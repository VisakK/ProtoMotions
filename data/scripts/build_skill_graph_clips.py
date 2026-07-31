# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cut a yoga ``.motion`` into **skill-graph** node and edge clips.

Motivation
----------
A Stage-1 mimic tracker imitates *one* reference trajectory, so it can only ever
reproduce transitions the dataset happens to contain.  There is no reference for
"handstand -> warrior III", so no amount of Stage-1/Stage-2 MaskedMimic training
produces it.  The skill-graph alternative is to factor a flow into

* **nodes**  -- quasi-static poses the character can hold *indefinitely*, and
* **edges**  -- the transitions between them,

train a small expert per node/edge, and let a planner (later, a distilled
student) sequence them.  This script produces the clips that experiment needs.

Node clips are **frozen**: a single representative ("medoid") frame of a
quasi-static hold window, repeated for ``--hold-seconds`` with **all velocities
zeroed**.  Three properties follow, and all three matter:

1. *Holdable for an arbitrary time.*  The tracking target never advances, so the
   episode length is bounded by ``max_episode_length``, not by the clip.
2. *Time-invariant value function.*  ``mimic_target_poses`` is a lookahead on the
   reference; with a constant reference it is constant, so the critic's V(s)
   depends only on the **physical** state.  That is what makes
   ``V(s) -> stability region`` well posed -- otherwise V would confound "how
   stable am I" with "where am I in the clip".
3. *No reset-velocity spike.*  Reference velocities are injected verbatim at
   reset (``notes/Addressing_grounding_PD_stiffness_issues.MD`` §10); a frozen
   frame injects zeros.

Edge clips are **real slices** of the source motion, cut with generous margins
*into* the quiet hold windows at both ends, so an edge starts inside its source
node's basin and finishes inside its target node's basin.  No splicing, so no
velocity discontinuities.

The source clip is expected to be per-frame grounded already (the
``yoga_motions_proto_yogi_grounded`` corpus).  Node clips are re-grounded anyway
-- freezing picks a single frame, and that frame's own clearance is what the
policy will see for the whole episode -- and every output is verified against
the collision geometry before it is written.

Usage::

    python data/scripts/build_skill_graph_clips.py \
        --src data/smpl/yoga_motions_proto_yogi_grounded_yogaonly/220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a.motion \
        --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \
        --out-dir data/smpl/skill_graph_handstand
"""

import argparse
import json
from pathlib import Path

import torch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_yoga_motions import load_char_geoms, compute_min_geom_z  # noqa: E402
from contact_detection import compute_contact_labels_from_pos_and_vel  # noqa: E402


# --------------------------------------------------------------------------- #
# Segment table.
#
# Frame numbers were located by scanning the source clip for quasi-static
# windows (mean whole-body speed over a 1 s window) restricted to a pose mask
# per node.  See notes/Skill_graph_nodes.MD for the derivation and the trace.
#
# The source clip 220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a is 2938
# frames @ 60 fps (48.97 s) and contains four handstand attempts.  The fourth
# (t = 31.9 - 48.97 s) is the only one that runs cleanly all the way from a
# three-legged down dog through a held handstand and out to standing, so every
# segment below is cut from it.
# --------------------------------------------------------------------------- #
NODES = {
    # name: (hold_window_start, hold_window_end, description)
    "node_downdog3": (
        1983,
        2043,
        "Three-legged down dog (Eka Pada Adho Mukha Svanasana), left leg lifted, "
        "right foot planted. Mean speed 0.017 m/s over the window.",
    ),
    "node_handstand": (
        2598,
        2658,
        "Handstand (Adho Mukha Vrksasana), legs together and vertical. "
        "Mean speed 0.029 m/s over the window.",
    ),
    "node_tadasana": (
        2877,
        2937,
        "Standing, arms overhead (Urdhva Hastasana / tadasana with raised arms) "
        "-- the pose the exit lands in. Mean speed 0.066 m/s over the window.",
    ),
}

EDGES = {
    # name: (start_frame, end_frame, source_node, target_node, description)
    "edge_kickup": (
        2000,
        2640,
        "node_downdog3",
        "node_handstand",
        "Kick-up: leg swings down for momentum, planted foot pushes off, both "
        "legs rise and converge into the handstand.",
    ),
    "edge_exit": (
        2620,
        2938,
        "node_handstand",
        "node_tadasana",
        "Exit: legs lower under control, pass through a deep forward fold, "
        "then roll up to standing with the arms overhead.",
    ),
}

FLOW = {
    "flow_downdog_handstand_tadasana": (
        1983,
        2938,
        "The full fluid sequence the graph factors: three-legged down dog -> "
        "kick up -> handstand -> exit -> tadasana.",
    ),
}

# Fields that are a per-frame time series and must be sliced/repeated together.
TIME_SERIES_KEYS = [
    "dof_pos",
    "dof_vel",
    "rigid_body_pos",
    "rigid_body_rot",
    "rigid_body_vel",
    "rigid_body_ang_vel",
    "rigid_body_contacts",
    "local_rigid_body_rot",
]
VELOCITY_KEYS = ["dof_vel", "rigid_body_vel", "rigid_body_ang_vel"]


def medoid_frame(motion: dict, lo: int, hi: int) -> int:
    """Index of the frame in ``[lo, hi)`` closest to every other frame in it.

    Distance is the mean per-body Euclidean distance on **root-relative** body
    positions, so a slow root drift across the window cannot decide the winner
    -- the pose shape does.
    """
    pos = motion["rigid_body_pos"][lo:hi]  # [W, B, 3]
    rel = pos - pos[:, 0:1, :]  # subtract the pelvis
    # pairwise: [W, W]
    d = (rel.unsqueeze(0) - rel.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
    return lo + int(d.sum(dim=-1).argmin().item())


def slice_motion(motion: dict, lo: int, hi: int) -> dict:
    """A plain time slice; everything else (fps, state_conversion) carried over."""
    out = {k: v for k, v in motion.items() if k not in TIME_SERIES_KEYS}
    for k in TIME_SERIES_KEYS:
        if k in motion:
            out[k] = motion[k][lo:hi].clone()
    return out


def freeze_motion(motion: dict, frame: int, num_frames: int) -> dict:
    """Repeat one frame ``num_frames`` times with every velocity zeroed."""
    out = {k: v for k, v in motion.items() if k not in TIME_SERIES_KEYS}
    for k in TIME_SERIES_KEYS:
        if k not in motion:
            continue
        f = motion[k][frame : frame + 1]
        rep = f.repeat(*([num_frames] + [1] * (f.dim() - 1))).clone()
        if k in VELOCITY_KEYS:
            rep.zero_()
        out[k] = rep
    return out


def ground_motion(motion: dict, char_geoms, clearance: float) -> float:
    """Shift every frame vertically so its lowest collision geom sits at
    ``clearance``.  Returns the mean applied offset (metres).

    A pure vertical translation leaves rotations, DOFs and horizontal motion
    invariant.  World-z velocity picks up d(offset)/dt so a time-varying shift
    does not silently inject vertical velocity; for a frozen clip the offset is
    constant, so this term is exactly zero.
    """
    pos = motion["rigid_body_pos"]
    rot = motion["rigid_body_rot"]
    min_z = compute_min_geom_z(char_geoms, pos, rot)  # [T]
    offset = clearance - min_z  # [T]
    pos[:, :, 2] += offset.unsqueeze(-1)

    if pos.shape[0] > 1:
        fps = float(motion["fps"])
        d_off = torch.gradient(offset, spacing=1.0 / fps)[0]
        motion["rigid_body_vel"][:, :, 2] += d_off.unsqueeze(-1)

    return float(offset.mean().item())


def recompute_contacts(motion: dict) -> None:
    motion["rigid_body_contacts"] = compute_contact_labels_from_pos_and_vel(
        motion["rigid_body_pos"], motion["rigid_body_vel"]
    )


def verify(motion: dict, char_geoms, name: str) -> dict:
    """Geometry + sanity report for a written clip."""
    min_z = compute_min_geom_z(
        char_geoms, motion["rigid_body_pos"], motion["rigid_body_rot"]
    )
    speed = motion["rigid_body_vel"].norm(dim=-1).mean(dim=-1)
    t = motion["rigid_body_pos"].shape[0]
    return {
        "name": name,
        "frames": t,
        "seconds": round(t / float(motion["fps"]), 3),
        "min_geom_z_cm": [
            round(float(min_z.min()) * 100, 4),
            round(float(min_z.mean()) * 100, 4),
            round(float(min_z.max()) * 100, 4),
        ],
        "pelvis_z_m": round(float(motion["rigid_body_pos"][:, 0, 2].mean()), 4),
        "mean_body_speed_mps": round(float(speed.mean()), 5),
        "max_body_speed_mps": round(float(speed.max()), 5),
        "contacts_per_frame": round(
            float(motion["rigid_body_contacts"].float().sum(dim=-1).mean()), 3
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Source .motion file.")
    p.add_argument("--mjcf", required=True, help="MJCF the clip was FK'd onto.")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--hold-seconds",
        type=float,
        default=40.0,
        help="Length of each frozen node clip. Must exceed "
        "max_episode_length * control_dt (1000 / 30 Hz = 33.3 s) so a full "
        "episode never runs off the end of the reference.",
    )
    p.add_argument("--clearance", type=float, default=0.005)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "nodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "edges").mkdir(parents=True, exist_ok=True)
    (out_dir / "flow").mkdir(parents=True, exist_ok=True)

    motion = torch.load(args.src, map_location="cpu", weights_only=False)
    fps = int(motion["fps"])
    total = motion["rigid_body_pos"].shape[0]
    print(f"source: {args.src}\n  {total} frames @ {fps} fps = {total/fps:.2f} s")

    from protomotions.components.pose_lib import extract_kinematic_info

    ki = extract_kinematic_info(args.mjcf)
    char_geoms = load_char_geoms(args.mjcf, ki.body_names, torch.device("cpu"))

    report = {"source": args.src, "fps": fps, "source_frames": total,
              "nodes": [], "edges": [], "flow": []}

    hold_frames = int(round(args.hold_seconds * fps))

    # ---------------- nodes ----------------
    for name, (lo, hi, desc) in NODES.items():
        med = medoid_frame(motion, lo, hi)
        clip = freeze_motion(motion, med, hold_frames)
        off = ground_motion(clip, char_geoms, args.clearance)
        recompute_contacts(clip)
        path = out_dir / "nodes" / f"{name}.motion"
        torch.save(clip, path)
        info = verify(clip, char_geoms, name)
        info.update(
            hold_window=[lo, hi],
            medoid_frame=med,
            medoid_time_s=round(med / fps, 3),
            ground_offset_cm=round(off * 100, 4),
            description=desc,
            path=str(path),
        )
        report["nodes"].append(info)
        print(f"\n[node] {name}: medoid frame {med} (t={med/fps:.2f}s) "
              f"from window {lo}-{hi}, grounded by {off*100:+.3f} cm")
        print(f"       {info}")

    # ---------------- edges ----------------
    for name, (lo, hi, src_node, dst_node, desc) in EDGES.items():
        clip = slice_motion(motion, lo, hi)
        off = ground_motion(clip, char_geoms, args.clearance)
        recompute_contacts(clip)
        path = out_dir / "edges" / f"{name}.motion"
        torch.save(clip, path)
        info = verify(clip, char_geoms, name)
        info.update(
            source_frames=[lo, hi],
            source_time_s=[round(lo / fps, 3), round(hi / fps, 3)],
            from_node=src_node,
            to_node=dst_node,
            mean_ground_offset_cm=round(off * 100, 4),
            description=desc,
            path=str(path),
        )
        report["edges"].append(info)
        print(f"\n[edge] {name}: frames {lo}-{hi} "
              f"({src_node} -> {dst_node})")
        print(f"       {info}")

    # ---------------- flow ----------------
    for name, (lo, hi, desc) in FLOW.items():
        clip = slice_motion(motion, lo, hi)
        off = ground_motion(clip, char_geoms, args.clearance)
        recompute_contacts(clip)
        path = out_dir / "flow" / f"{name}.motion"
        torch.save(clip, path)
        info = verify(clip, char_geoms, name)
        info.update(
            source_frames=[lo, hi],
            description=desc,
            path=str(path),
        )
        report["flow"].append(info)
        print(f"\n[flow] {name}: frames {lo}-{hi}")
        print(f"       {info}")

    # ------------- graph consistency: how far is each edge endpoint from the
    # ------------- node pose it is supposed to start/end in?
    print("\n=== graph consistency (mean per-body distance, root-relative) ===")
    node_pose = {}
    for name, (lo, hi, _) in NODES.items():
        med = medoid_frame(motion, lo, hi)
        node_pose[name] = motion["rigid_body_pos"][med]

    def rel(x):
        return x - x[0:1, :]

    consistency = []
    for name, (lo, hi, src_node, dst_node, _) in EDGES.items():
        d_start = (rel(motion["rigid_body_pos"][lo]) - rel(node_pose[src_node])).norm(dim=-1).mean()
        d_end = (rel(motion["rigid_body_pos"][hi - 1]) - rel(node_pose[dst_node])).norm(dim=-1).mean()
        entry = {
            "edge": name,
            "start_vs_%s_cm" % src_node: round(float(d_start) * 100, 2),
            "end_vs_%s_cm" % dst_node: round(float(d_end) * 100, 2),
        }
        consistency.append(entry)
        print(f"  {name}: start is {float(d_start)*100:.2f} cm from {src_node}, "
              f"end is {float(d_end)*100:.2f} cm from {dst_node}")
    report["graph_consistency"] = consistency

    with open(out_dir / "segments.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_dir/'segments.json'}")


if __name__ == "__main__":
    main()
