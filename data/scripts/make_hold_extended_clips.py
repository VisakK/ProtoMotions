# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bake extended holds into reference clips, in several duration variants.

``expert_revist/expert_revisit.MD`` §3.2: the hold command is trained on clips
whose reference *stays* at the pose. Rather than a separate frozen ``hold_*``
clip, the hold is inserted into the clip itself -- entry, tiled hold, exit --
so one episode carries the whole node-edge-node-edge structure and the dwell
channel has continuous semantics. And it is inserted in **several duration
variants** (``--variants 0 3 7`` by default): with one fixed duration the
policy can learn "after N frames of this pose, leave" from the pose alone --
round 9's anticipatory departure -- whereas with variable durations the only
reliable exit cue is ``dwell_remaining``.

For every clip in the manifest (``propose_hold_manifest.py`` output, after
review) and every variant ``d``:

* every hold flagged ``extend: true`` has its exemplar frame (``frame_hold``)
  repeated ``round(d * fps)`` times, immediately after the exemplar;
* velocities are **recomputed from the spliced kinematics** with the same
  routines the converter used (``pose_lib.compute_kinematics_velocities`` for
  the bodies, ``compute_angular_velocity`` on the local joint rotations for
  ``dof_vel``), so the tiled hold has exactly zero velocity and the boundaries
  are consistent by construction rather than blended by hand -- the
  multi-horizon minimum the converter uses already decelerates into a frozen
  block over its horizon;
* contacts are tiled; measured pressure fields, if any, are dropped (a tiled
  instantaneous pressure frame would be an invented measurement);
* ``d = 0`` re-emits the clip **byte-identical** under its own name (stored
  velocities are copied through, not recomputed), and the script checks on it
  that recomputing would reproduce the stored velocities to a small *mean*
  drift, so a change in the converter's conventions (an fps relabel, a
  different horizon) cannot pass silently. The max is reported, not gated:
  the stored velocities differ from a fresh finite difference by up to
  ~0.2 m/s on ~4 % of frames (multi-horizon selection flips), which is
  immaterial to the ``gv`` term and expected.

Outputs: ``<out-dir>/<stem>.motion`` (d = 0), ``<out-dir>/<stem>_x<d>s.motion``
(d > 0), and ``<out-dir>/holds_extended.yaml`` -- the manifest of *emitted*
clips with every hold's times shifted, which is what
:mod:`build_hold_graph` consumes.

Usage::

    PYTHONPATH=. python data/scripts/make_hold_extended_clips.py \
      --manifest data/smpl/expert60/holds.yaml \
      --out-dir data/smpl/yoga_motions_proto_yogi_expert60 --variants 0 3 7
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from protomotions.components.pose_lib import (  # noqa: E402
    compute_angular_velocity,
    compute_kinematics_velocities,
)
from protomotions.utils.rotations import quaternion_to_matrix  # noqa: E402

TILED_FIELDS = (
    "dof_pos",
    "rigid_body_pos",
    "rigid_body_rot",
    "local_rigid_body_rot",
    "rigid_body_contacts",
)
RECOMPUTED_FIELDS = ("dof_vel", "rigid_body_vel", "rigid_body_ang_vel")
DROPPED_FIELDS = ("ground_reaction", "rigid_body_ground_forces", "ground_reaction_valid")
VELOCITY_MAX_HORIZON = 3  # convert_yoga_frames_to_proto.py


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def insertion_plan(holds: list[dict], insert_frames: int) -> list[tuple[int, int]]:
    """``[(frame_hold, n_frames)]`` for the holds flagged ``extend``, in time order."""
    plan = [
        (int(h["frame_hold"]), int(insert_frames))
        for h in sorted(holds, key=lambda h: h["frame_hold"])
        if h.get("extend") and insert_frames > 0
    ]
    return plan


def splice_index(num_frames: int, plan: list[tuple[int, int]]) -> torch.Tensor:
    """Source-frame index for every output frame, after tiling per ``plan``."""
    index = []
    cursor = 0
    for frame, n in plan:
        index.extend(range(cursor, frame + 1))
        index.extend([frame] * n)
        cursor = frame + 1
    index.extend(range(cursor, num_frames))
    return torch.tensor(index, dtype=torch.long)


def shifted_hold(hold: dict, plan: list[tuple[int, int]], fps: int) -> dict:
    """The hold's frame/time fields after the insertions in ``plan``."""
    out = copy.deepcopy(hold)
    before_start = sum(n for f, n in plan if f < hold["frame_start"])
    before_hold = sum(n for f, n in plan if f < hold["frame_hold"])
    # An insertion *at* the hold's own exemplar lengthens the hold; one at or
    # before its start shifts it whole.
    within = sum(n for f, n in plan if hold["frame_start"] <= f <= hold["frame_end"])
    out["frame_start"] = hold["frame_start"] + before_start
    out["frame_hold"] = hold["frame_hold"] + before_hold
    out["frame_end"] = hold["frame_end"] + before_start + within
    out["t_start"] = round(out["frame_start"] / fps, 4)
    out["t_hold"] = round(out["frame_hold"] / fps, 4)
    out["t_end"] = round(out["frame_end"] / fps, 4)
    out["duration_s"] = round((out["frame_end"] - out["frame_start"] + 1) / fps, 3)
    return out


def recompute_velocities(motion: dict, fps: int) -> dict:
    """Body and joint velocities from the kinematic fields, converter-style."""
    pos = motion["rigid_body_pos"].float()
    rot_mats = quaternion_to_matrix(motion["rigid_body_rot"].float(), w_last=True)
    lin_vel, ang_vel = compute_kinematics_velocities(
        pos, rot_mats, fps, VELOCITY_MAX_HORIZON
    )
    local_mats = quaternion_to_matrix(motion["local_rigid_body_rot"].float(), w_last=True)
    dof_vel = compute_angular_velocity(local_mats[:, 1:], fps=fps)
    return {
        "rigid_body_vel": lin_vel,
        "rigid_body_ang_vel": ang_vel,
        "dof_vel": dof_vel.reshape(dof_vel.shape[0], -1),
    }


def extend_motion(motion: dict, plan: list[tuple[int, int]]) -> dict:
    """Tile the kinematic fields per ``plan`` and recompute the velocities.

    An empty ``plan`` returns the clip with its stored velocities untouched (the
    d = 0 variant is byte-identical to the source apart from dropped pressure
    fields).
    """
    num_frames = int(motion["rigid_body_pos"].shape[0])
    index = splice_index(num_frames, plan)
    fps = int(motion.get("fps", 60))
    unchanged = len(plan) == 0
    out = {}
    for key, value in motion.items():
        if key in DROPPED_FIELDS:
            continue
        if not torch.is_tensor(value):
            out[key] = value
        elif key in TILED_FIELDS:
            out[key] = value.index_select(0, index).clone()
        elif key in RECOMPUTED_FIELDS:
            if unchanged:
                out[key] = value.clone()
        else:
            raise ValueError(f"extend_motion does not know how to hold field '{key}'")
    if not unchanged:
        fresh = recompute_velocities(out, fps)
        for key, value in fresh.items():
            out[key] = value.to(motion[key].dtype)
    return out


def velocity_reproduction_error(motion: dict) -> dict:
    """``{field: (mean, max)}`` of |stored - recomputed| on an unmodified clip."""
    fresh = recompute_velocities(motion, int(motion.get("fps", 60)))
    out = {}
    for key in RECOMPUTED_FIELDS:
        diff = (motion[key].float() - fresh[key]).abs()
        out[key] = (float(diff.mean()), float(diff.max()))
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default="data/smpl/expert60/holds.yaml")
    parser.add_argument("--out-dir", default="data/smpl/yoga_motions_proto_yogi_expert60")
    parser.add_argument("--variants", type=float, nargs="+", default=[0.0, 3.0, 7.0])
    parser.add_argument(
        "--max-velocity-drift", type=float, default=0.02,
        help="Refuse if recomputed velocities differ from the stored ones on an "
             "unmodified clip by more than this MEAN absolute value (m/s, rad/s).",
    )
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = yaml.safe_load(open(manifest_path))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    emitted = []
    worst_drift = {key: {"mean": 0.0, "max": 0.0} for key in RECOMPUTED_FIELDS}
    for clip in manifest["clips"]:
        if args.only and clip["stem"] not in args.only:
            continue
        motion = torch.load(clip["source"], map_location="cpu", weights_only=False)
        fps = int(motion.get("fps", 60))
        if fps != int(clip["fps"]):
            raise ValueError(f"{clip['stem']}: manifest fps {clip['fps']} != file fps {fps}")

        drift = velocity_reproduction_error(motion)
        for key, (mean, peak) in drift.items():
            worst_drift[key]["mean"] = max(worst_drift[key]["mean"], mean)
            worst_drift[key]["max"] = max(worst_drift[key]["max"], peak)
        if max(mean for mean, _ in drift.values()) > args.max_velocity_drift:
            raise ValueError(
                f"{clip['stem']}: recomputed velocities drift from the stored ones "
                f"by (mean, max) {drift}; the converter's conventions have changed"
            )

        for variant in args.variants:
            insert = int(round(variant * fps))
            plan = insertion_plan(clip["holds"], insert)
            stem = clip["stem"] if insert == 0 else f"{clip['stem']}_x{int(round(variant))}s"
            extended = extend_motion(motion, plan)
            torch.save(extended, out_dir / f"{stem}.motion")
            holds = [shifted_hold(h, plan, fps) for h in clip["holds"]]
            num_frames = int(extended["rigid_body_pos"].shape[0])
            emitted.append(
                {
                    **{k: v for k, v in clip.items() if k not in ("holds", "source", "num_frames", "length_s")},
                    "stem": stem,
                    "source_stem": clip["stem"],
                    "source": clip["source"],
                    "variant_s": float(variant),
                    "inserted_frames": int(sum(n for _, n in plan)),
                    "num_frames": num_frames,
                    "length_s": round(num_frames / fps, 3),
                    "holds": holds,
                }
            )
            print(
                f"{stem}: {num_frames} frames (+{sum(n for _, n in plan)}), "
                f"{sum(1 for _, n in plan)} holds extended"
            )

    payload = {
        "version": 1,
        "source_manifest": str(manifest_path),
        "variants_s": [float(v) for v in args.variants],
        "velocity_drift_max": worst_drift,
        "clips": emitted,
    }
    with open(out_dir / "holds_extended.yaml", "w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, width=120)
    print(
        f"wrote {len(emitted)} clips to {out_dir}; velocity reproduction drift "
        f"{ {k: {m: round(v, 4) for m, v in d.items()} for k, d in worst_drift.items()} }"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
