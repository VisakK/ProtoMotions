# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geom-aware ground-clearance cleaning for converted yoga ``.motion`` files.

The converter (:mod:`convert_yoga_frames_to_proto`) applies ``fix_height(0.015)``
which lifts the lowest *joint* to +1.5 cm, but the collision *geoms* (heel/toe
boxes, limb capsules) still hang below the ground plane -- e.g. Cobra -4.4 cm,
Tree -4.2 cm, Boat -3.7 cm, Scorpion -1.8 cm (see ``notes/Bring_up.md``).

This tool recomputes, per clip, the lowest world-z over **all** collision geoms
of the boxhands MJCF (spheres/capsules/boxes reduced to witness points + radius)
and applies a single constant whole-clip vertical lift so the deepest geom over
the clip rests at ``--clearance`` above the ground.  Because a pure translation
leaves rotations, velocities and DOFs invariant, only ``rigid_body_pos`` is
shifted; ``rigid_body_contacts`` is recomputed from the shifted heights with the
same heuristic the converter used.

Geom parsing is ported from MimicKit ``mimickit/anim/char_geoms.py``; the
quaternion math uses ProtoMotions' own :func:`quat_rotate` (xyzw, matching
``RobotState.rigid_body_rot``).

Three grounding modes:

* default -- one constant whole-clip lift so the *deepest frame of the clip*
  rests at ``--clearance``; every other frame still floats.  Safe for anything.
* ``--per-frame`` -- ground *every* frame to ``--clearance``.  Removes the
  reset-float that causes the post-reset drop/bounce, but pins genuinely
  airborne phases to the floor, so it is for grounded clips only.
* ``--per-frame --ballistic-guard`` -- as above, but airborne phases are
  detected and their arc preserved (the correction is interpolated across the
  flight).  Safe for mixed corpora containing jumps/flips.

Usage::

    python data/scripts/clean_yoga_motions.py \
        --mjcf data/assets/smpl/smpl_boxhands_lowtorque.xml \
        --in-dir data/smpl/yoga_motions_proto \
        --out-dir data/smpl/yoga_motions_proto_cleaned \
        --clearance 0.005
    # add --dry-run to only report per-clip penetration / lift.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.utils.rotations import quat_rotate

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_detection import compute_contact_labels_from_pos_and_vel


# --------------------------------------------------------------------------- #
# Collision-geom parsing (ported from MimicKit anim/char_geoms.py).
# Each geom -> body-local witness points + a radius; lowest world z of the geom
# is min_z(points rotated into world) - radius (exact for sphere/capsule/box on
# flat ground).
# --------------------------------------------------------------------------- #
def _mjcf_quat_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def _rotate_np(quat_xyzw, points):
    q = torch.tensor(quat_xyzw, dtype=torch.float32).expand(points.shape[0], 4)
    p = torch.tensor(points, dtype=torch.float32)
    return quat_rotate(q, p, w_last=True).numpy()


def _box_corners(half_extents):
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(
                    [sx * half_extents[0], sy * half_extents[1], sz * half_extents[2]]
                )
    return np.array(corners, dtype=np.float32)


def _parse_geom(geom_xml):
    geom_type = geom_xml.attrib.get("type", "sphere")

    pos = geom_xml.attrib.get("pos")
    pos = np.fromstring(pos, dtype=np.float32, sep=" ") if pos else np.zeros(3, np.float32)

    quat = geom_xml.attrib.get("quat")
    if quat:
        quat = _mjcf_quat_to_xyzw(np.fromstring(quat, dtype=np.float32, sep=" "))
    else:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    size = geom_xml.attrib.get("size")
    size = np.fromstring(size, dtype=np.float32, sep=" ") if size else np.zeros(1, np.float32)

    fromto = geom_xml.attrib.get("fromto")
    fromto = np.fromstring(fromto, dtype=np.float32, sep=" ") if fromto else None

    if geom_type == "sphere":
        points = pos[np.newaxis, :]
        radius = float(size[0])
    elif geom_type == "capsule":
        radius = float(size[0])
        if fromto is not None:
            points = np.stack([fromto[0:3], fromto[3:6]], axis=0)
        else:
            half_len = float(size[1])
            axis = _rotate_np(quat, np.array([[0.0, 0.0, half_len]], dtype=np.float32))[0]
            points = np.stack([pos + axis, pos - axis], axis=0)
    elif geom_type == "box":
        radius = 0.0
        corners = _box_corners(size[:3])
        points = pos[np.newaxis, :] + _rotate_np(quat, corners)
    else:
        print(f"clean_yoga_motions: unsupported geom type '{geom_type}', treating as point")
        points = pos[np.newaxis, :]
        radius = 0.0

    return {"points": points.astype(np.float32), "radius": radius}


def load_char_geoms(char_file, body_names, device):
    """List aligned with ``body_names``; entry b = list of {points[P,3], radius}."""
    tree = ET.parse(char_file)
    worldbody = tree.getroot().find("worldbody")
    geoms_by_name = {name: [] for name in body_names}

    def _recurse(body_xml):
        name = body_xml.attrib.get("name")
        if name in geoms_by_name:
            for geom_xml in body_xml.findall("geom"):
                geoms_by_name[name].append(_parse_geom(geom_xml))
        for child in body_xml.findall("body"):
            _recurse(child)

    for root_body in worldbody.findall("body"):
        _recurse(root_body)

    char_geoms = []
    for name in body_names:
        char_geoms.append(
            [
                {
                    "points": torch.tensor(g["points"], dtype=torch.float32, device=device),
                    "radius": g["radius"],
                }
                for g in geoms_by_name[name]
            ]
        )
    return char_geoms


def compute_min_geom_z(char_geoms, body_pos, body_rot):
    """Lowest world z over all collision geoms, per frame.

    body_pos: [N, B, 3], body_rot: [N, B, 4] (xyzw). Returns [N]."""
    n = body_pos.shape[0]
    min_z = torch.full([n], float("inf"), dtype=body_pos.dtype, device=body_pos.device)
    for b, body_geoms in enumerate(char_geoms):
        if len(body_geoms) == 0:
            continue
        b_pos = body_pos[:, b, :]
        b_rot = body_rot[:, b, :]
        for geom in body_geoms:
            pts = geom["points"]
            m = pts.shape[0]
            rot = b_rot.unsqueeze(1).expand(n, m, 4).reshape(-1, 4)
            p = pts.unsqueeze(0).expand(n, m, 3).reshape(-1, 3)
            world = quat_rotate(rot, p, w_last=True).reshape(n, m, 3) + b_pos.unsqueeze(1)
            geom_min_z = world[..., 2].min(dim=1)[0] - geom["radius"]
            min_z = torch.minimum(min_z, geom_min_z)
    return min_z


# --------------------------------------------------------------------------- #
# Ballistic guard: tell a *converter float* (ground it) apart from *genuine
# flight* (preserve the arc).
#
# Height alone cannot separate them -- a Wheel float is 16 cm and a jump apex is
# 24 cm.  Vertical *speed* of the lowest geom cannot either: it spikes whenever
# the lowest geom changes identity (a floating toe handing over to a descending
# hand), which reads as flight on perfectly static yoga poses.
#
# What does separate them is **free fall**: while airborne the only force on the
# body is gravity, so its COM accelerates at ~-9.81 m/s^2.  Measured over the
# 199-clip yogi corpus (COM proxy = mean body height, median-filtered):
#
#   * 0 / 179 yoga clips reach even 3 consecutive frames below -5 m/s^2
#     (their most negative COM acceleration anywhere is -2.5 m/s^2), while
#   * 15 / 20 smpl_* locomotion/acrobatic clips do (jump 26 frames, spin 15,
#     spinkick 10, backflip 7 at -124, sideflip 7 at -218).
#
# A lift backstop covers the rest: no yoga clip floats past 16.2 cm, so anything
# peaking above --max-float-lift (default 25 cm) is airborne by construction --
# this is what catches smpl_hop (41 cm) whose recorded dynamics are too damped to
# show a free-fall signature.
# --------------------------------------------------------------------------- #
def _median_filter(x, window):
    """Centered median filter with edge padding (window forced odd, >= 1).

    Used instead of a mean before differentiating: ``min_geom_z`` *steps* when the
    lowest geom changes identity (a toe at 14 cm handing over to a hand at 2 cm),
    and a mean smears that step into several frames of large apparent speed, which
    reads as flight.  A median keeps the step one frame wide so the
    consecutive-frames test below can reject it.
    """
    window = max(1, int(window) | 1)
    if window == 1:
        return x
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def _longest_true_run(mask):
    """Length of the longest run of consecutive True values."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _contiguous_runs(mask):
    """[(start, end_inclusive), ...] for each run of True in a 1-D bool array."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def com_vertical_acceleration(com_z, fps, smooth_ms):
    """Median-filtered vertical acceleration of the COM proxy [m/s^2]."""
    dt = 1.0 / fps
    smoothed = _median_filter(com_z, round(0.05 * fps))
    return _median_filter(np.gradient(np.gradient(smoothed, dt), dt), round(smooth_ms * 1e-3 * fps))


def classify_flight(
    min_z, com_z, fps, clearance, air_enter, accel_thresh, min_air_frames, max_float_lift, smooth_ms
):
    """bool[T]: True on frames belonging to a genuinely airborne phase.

    Decided per *run*, not per frame: a jump's apex has v_z ~ 0 and is still
    airborne, so any per-frame test would carve the apex out of its own flight.
    A maximal run of frames lifted above ``air_enter`` is declared flight if it
    either contains ``min_air_frames`` **consecutive** frames of free fall
    (COM acceleration below ``accel_thresh``) or peaks above ``max_float_lift``.
    """
    lift = min_z - clearance
    acc = com_vertical_acceleration(com_z, fps, smooth_ms)
    flight = np.zeros(len(min_z), dtype=bool)
    n_flight_runs = 0
    for a, b in _contiguous_runs(lift > air_enter):
        free_fall = _longest_true_run(acc[a : b + 1] < accel_thresh) >= min_air_frames
        if free_fall or lift[a : b + 1].max() > max_float_lift:
            flight[a : b + 1] = True
            n_flight_runs += 1
    return flight, n_flight_runs


def build_guarded_offset(min_z, flight, clearance):
    """Per-frame vertical offset that grounds supported frames and preserves flight.

    Supported frames get the exact grounding offset ``clearance - min_z``.
    Across a flight phase the offset is *linearly interpolated* between the
    offsets of the bracketing supported frames (constant-extrapolated at the clip
    ends), so the ballistic arc keeps its shape -- only its slowly-varying
    reference drift is removed.  With no supported frame anywhere, fall back to
    the safe constant whole-clip lift.
    """
    off = clearance - min_z
    support = ~flight
    if not support.any():
        return np.full_like(off, max(0.0, clearance - float(min_z.min())))
    idx = np.arange(len(off), dtype=np.float64)
    return np.interp(idx, idx[support], off[support])


# --------------------------------------------------------------------------- #
def clean_one(
    path,
    char_geoms,
    clearance,
    vel_thres,
    height_thresh,
    per_frame=False,
    guard=None,
):
    """Returns (g_min, offset, num_frames, changed_dict) -- dict is None if the
    file has no rigid_body_pos.  ``offset`` is the constant whole-clip lift, or the
    mean |per-frame lift| when ``per_frame=True``."""
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(d, dict) or "rigid_body_pos" not in d:
        return None
    rbp = d["rigid_body_pos"]  # [T, B, 3]
    rbr = d["rigid_body_rot"]  # [T, B, 4] xyzw
    min_z = compute_min_geom_z(char_geoms, rbp, rbr)
    g_min = float(min_z.min())

    rbp2 = rbp.clone()
    vel = d.get("rigid_body_vel", None)
    n_flight = n_flight_runs = 0
    if per_frame:
        # Ground EVERY frame: lower floating frames / lift penetrating ones so the
        # lowest geom sits at ``clearance`` each frame.  A per-frame *vertical*
        # translation leaves rotations / dof / horizontal motion invariant; only
        # world-z of each body shifts and world-z velocity gains d(offset)/dt.
        # WARNING without ``guard``: valid ONLY for clips with a ground contact
        # every frame (yoga / balance).  It will wrongly pin genuinely airborne
        # phases (jump / backflip) to the floor.  Pass ``guard`` (--ballistic-guard)
        # to detect and preserve those phases, which makes it safe for mixed sets.
        fps = int(d.get("fps", 60))
        if guard is not None:
            mz = min_z.cpu().numpy().astype(np.float64)
            com_z = rbp[:, :, 2].mean(dim=1).cpu().numpy().astype(np.float64)
            flight, n_flight_runs = classify_flight(mz, com_z, fps, clearance, **guard)
            n_flight = int(flight.sum())
            off_np = build_guarded_offset(mz, flight, clearance)
            off_t = torch.from_numpy(off_np).to(min_z.dtype)
        else:
            off_t = clearance - min_z  # [T] signed (negative lowers floating frames)
        rbp2[..., 2] += off_t.unsqueeze(1)
        offset = float(off_t.abs().mean())
        if vel is not None:
            doff = np.gradient(off_t.cpu().numpy().astype(np.float64), 1.0 / fps)
            vel = vel.clone()
            vel[..., 2] = vel[..., 2] + torch.from_numpy(doff).to(vel.dtype).unsqueeze(1)
            d["rigid_body_vel"] = vel
    else:
        offset = max(0.0, clearance - g_min)  # single constant whole-clip lift
        rbp2[..., 2] += offset

    d["rigid_body_pos"] = rbp2
    if vel is not None:
        d["rigid_body_contacts"] = compute_contact_labels_from_pos_and_vel(
            positions=rbp2,
            velocity=vel,
            vel_thres=vel_thres,
            height_thresh=height_thresh,
        ).to(torch.bool)
    return g_min, offset, rbp.shape[0], d, n_flight, n_flight_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--clearance",
        type=float,
        default=0.005,
        help="target world-z of the deepest geom over the clip after the lift (m)",
    )
    ap.add_argument("--vel-thres", type=float, default=0.15)
    ap.add_argument("--height-thresh", type=float, default=0.1)
    ap.add_argument(
        "--per-frame",
        action="store_true",
        help="ground EVERY frame (lower floats / lift penetrations) so the lowest geom "
        "sits at --clearance each frame, instead of one constant whole-clip lift; also "
        "recomputes world-z velocity + contacts. Removes the reset-float that causes the "
        "post-reset drop/bounce. Use ONLY for clips grounded every frame (yoga/balance), "
        "NOT jump/backflip/airborne clips -- unless --ballistic-guard is also passed.",
    )
    ap.add_argument(
        "--ballistic-guard",
        action="store_true",
        help="with --per-frame: detect genuinely airborne phases and preserve their arc "
        "(offset linearly interpolated across the flight) instead of pinning them to the "
        "floor. Makes per-frame grounding safe for mixed corpora that contain jumps/flips.",
    )
    ap.add_argument(
        "--accel-thresh",
        type=float,
        default=-5.0,
        help="ballistic guard: COM vertical acceleration below this (m/s^2) is free fall. "
        "Yogi corpus: no yoga clip goes below -2.5; airborne clips reach -10 .. -218.",
    )
    ap.add_argument(
        "--air-enter",
        type=float,
        default=0.03,
        help="ballistic guard: lift above --clearance (m) for a frame to be a flight candidate",
    )
    ap.add_argument(
        "--min-air-frames",
        type=int,
        default=3,
        help="ballistic guard: consecutive free-fall frames needed to call a run genuine flight",
    )
    ap.add_argument(
        "--max-float-lift",
        type=float,
        default=0.25,
        help="ballistic guard: a run peaking above this lift (m) is flight regardless of "
        "speed -- backstop against pinning a big arc (max observed yoga float is 0.162 m)",
    )
    ap.add_argument(
        "--smooth-ms",
        type=float,
        default=83.0,
        help="ballistic guard: median-filter window (ms) applied to the COM acceleration, "
        "so single-frame mocap spikes cannot trip the free-fall test",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="optional list of specific .motion filenames (basename) to process",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    kin = extract_kinematic_info(args.mjcf)
    char_geoms = load_char_geoms(args.mjcf, kin.body_names, device="cpu")

    files = sorted(in_dir.glob("*.motion"))
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if f.name in wanted]

    guard = None
    if args.ballistic_guard:
        if not args.per_frame:
            ap.error("--ballistic-guard only applies with --per-frame")
        guard = dict(
            air_enter=args.air_enter,
            accel_thresh=args.accel_thresh,
            min_air_frames=args.min_air_frames,
            max_float_lift=args.max_float_lift,
            smooth_ms=args.smooth_ms,
        )

    print(
        f"{'clip':70s} {'frames':>7s} {'pen(cm)':>8s} {'lift(cm)':>9s} {'post(cm)':>9s} {'flight':>12s}"
    )
    n_written = 0
    worst = []
    airborne = []
    for f in files:
        res = clean_one(
            f,
            char_geoms,
            args.clearance,
            args.vel_thres,
            args.height_thresh,
            args.per_frame,
            guard,
        )
        if res is None:
            print(f"SKIP {f.name} (no rigid_body_pos)")
            continue
        g_min, offset, nfr, d, n_flight, n_flight_runs = res
        pen_cm = -min(g_min, 0.0) * 100.0
        lift_cm = offset * 100.0
        post_cm = args.clearance * 100.0 if args.per_frame else (g_min + offset) * 100.0
        flight_s = f"{n_flight_runs}x {100.0 * n_flight / nfr:4.0f}%" if n_flight else ""
        print(f"{f.name:70s} {nfr:7d} {pen_cm:8.2f} {lift_cm:9.2f} {post_cm:9.2f} {flight_s:>12s}")
        worst.append((pen_cm, f.name))
        if n_flight:
            airborne.append((n_flight / nfr, n_flight_runs, f.name))
        if not args.dry_run:
            torch.save(d, str(out_dir / f.name))
            n_written += 1

    worst.sort(reverse=True)
    print("\nWorst 10 pre-clean penetrations (cm):")
    for pen_cm, name in worst[:10]:
        print(f"  {pen_cm:6.2f}  {name}")
    if guard is not None:
        airborne.sort(reverse=True)
        print(
            f"\nBallistic guard: {len(airborne)}/{len(files)} clips had preserved flight phases"
        )
        for frac, nruns, name in airborne:
            print(f"  {frac * 100:5.1f}% of frames, {nruns:2d} run(s)  {name}")
    if args.dry_run:
        print(f"\n[dry-run] processed {len(files)} clips, wrote nothing")
    else:
        print(f"\nWrote {n_written} cleaned .motion files to {out_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
