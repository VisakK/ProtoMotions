# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthesize frozen-reference HOLD clips from the contact graph's held poses.

Why this exists, measured (``notes/Student_improvement_plan2.MD`` §3): the goal
schedule's sustained "stay at the configuration you are already in" command has
**median 0.27 s and p90 2.13 s**, and always ends in a departure -- so a deployed
5-10 s hold is 20-40x past anything training ever asked, and the observed drift
onset (~2 s) matches the p90. The cure is corpus-level: clips whose schedule
*actually says* "same node, long dwell".

For each selected graph node this takes the reference pose at the node's best
hold, tiles it for a randomized duration with **zero velocities**, and writes an
ordinary ``.motion`` file plus a manifest entry recording the source clip and
its owning expert. The clips then flow through the unchanged pipeline:

    package (select_student_subset ``student44h``) -> record expert rollouts ->
    build_contact_graph_from_rollouts -> train

Three deliberate properties:

* **The teacher is the owning Stage-1 expert tracking the frozen reference** --
  a stabilizer pursuing exactly the goal the student sees, never the original
  clip expert moving on (which would associate a hold command with leaving).
  Whether the expert can actually stabilize the pose is *measured* downstream:
  the recording's ``good_fraction`` gate drops failed holds from the graph.
* **Pressure channels are dropped.** The student corpus is packaged without the
  MOYO channel anyway (128 of 171 sources have none), and a tiled instantaneous
  pressure frame would be an invented measurement.
* **Hold clips are named ``hold_...``** so the graph builder places their hold
  frame late in the segment (``--synthetic-hold-prefix``): for a genuinely
  frozen reference every timestamp is equally "the held pose", and a late hold
  frame is what turns the clip into a long sustained stay-command instead of
  half stay, half zeroed-goal tail.

Usage::

    PYTHONPATH=. python data/scripts/make_hold_motions.py \
      --graph-dir data/smpl/yoga_contact_graph_student44_loadpath \
      --corpus data/smpl/yoga_yogi_student44.pt \
      --out-dir data/smpl/yoga_motions_proto_yogi_holds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Fields frozen frame-wise; velocity fields are zeroed instead of tiled.
TILED_FIELDS = (
    "dof_pos",
    "rigid_body_pos",
    "rigid_body_rot",
    "local_rigid_body_rot",
    "rigid_body_contacts",
)
ZEROED_FIELDS = ("dof_vel", "rigid_body_vel", "rigid_body_ang_vel")
DROPPED_FIELDS = ("ground_reaction", "rigid_body_ground_forces", "ground_reaction_valid")


def freeze_motion(motion: dict, frame: int, num_frames: int) -> dict:
    """A ``num_frames``-long motion holding ``motion``'s pose at ``frame``.

    Kinematic fields are tiled from the chosen frame, velocity fields are
    exactly zero (the reference *is* static, and the finite-difference of a
    constant is zero, so stored and derivable velocities agree), and measured
    pressure fields are dropped rather than fabricated.
    """
    out = {}
    for key, value in motion.items():
        if key in DROPPED_FIELDS:
            continue
        if not torch.is_tensor(value):
            out[key] = value
        elif key in ZEROED_FIELDS:
            out[key] = torch.zeros(
                (num_frames, *value.shape[1:]), dtype=value.dtype
            )
        elif key in TILED_FIELDS:
            out[key] = (
                value[frame : frame + 1]
                .expand(num_frames, *value.shape[1:])
                .clone()
            )
        else:
            # An unknown per-frame tensor would silently desynchronize the
            # packaged library; refuse rather than guess.
            raise ValueError(f"freeze_motion does not know how to hold field '{key}'")
    return out


def select_holds(graph: dict, min_dwell_s: float, min_good: float, max_holds: int):
    """One (node, best segment) per node, richest-dwell nodes first.

    The best occurrence of a node is its longest trusted segment with
    ``good_fraction >= min_good``: hold the pose only where the expert
    demonstrably tracked it.
    """
    best: dict[str, tuple] = {}
    dwell: dict[str, float] = {}
    for clip, rec in graph["clips"].items():
        for seg in rec["segments"]:
            if not seg["trusted"] or seg["good_fraction"] < min_good:
                continue
            key = seg["config"]
            dwell[key] = dwell.get(key, 0.0) + seg["duration_s"]
            current = best.get(key)
            if current is None or seg["duration_s"] > current[1]["duration_s"]:
                best[key] = (clip, seg)
    ranked = sorted(
        (k for k in best if dwell[k] >= min_dwell_s),
        key=lambda k: -dwell[k],
    )
    return [(k, dwell[k], *best[k]) for k in ranked[:max_holds]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", required=True,
                        help="contact graph whose held poses to freeze")
    parser.add_argument("--corpus", required=True,
                        help="packaged corpus the graph is keyed to; source .motion "
                             "paths are read from its motion_files list")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-node-dwell-s", type=float, default=4.0,
                        help="only nodes the corpus holds at least this long in total")
    parser.add_argument("--min-good-fraction", type=float, default=0.9,
                        help="a hold is cut only from a segment the expert tracked")
    parser.add_argument("--max-holds", type=int, default=36)
    parser.add_argument("--min-duration-s", type=float, default=6.0)
    parser.add_argument("--max-duration-s", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=0,
                        help="durations are randomized per hold but reproducible")
    args = parser.parse_args()

    graph = json.loads((Path(args.graph_dir) / "contact_graph.json").read_text())
    packaged = torch.load(args.corpus, map_location="cpu", weights_only=False)
    source_by_stem = {Path(f).stem: Path(f) for f in packaged["motion_files"]}

    picks = select_holds(
        graph, args.min_node_dwell_s, args.min_good_fraction, args.max_holds
    )
    if not picks:
        raise SystemExit("no nodes pass the dwell/good-fraction gates")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    entries = []
    print(f"{len(picks)} hold clips from {args.graph_dir}:")
    for config, node_dwell, clip, seg in picks:
        source_path = source_by_stem.get(clip)
        if source_path is None:
            raise SystemExit(f"graph clip '{clip}' is not in the corpus motion_files")
        motion = torch.load(source_path, map_location="cpu", weights_only=False)
        fps = float(motion["fps"])
        frame = int(round(seg["t_hold"] * fps))
        frame = min(max(frame, 0), motion["rigid_body_pos"].shape[0] - 1)
        duration = float(rng.uniform(args.min_duration_s, args.max_duration_s))
        num_frames = int(round(duration * fps))

        held = freeze_motion(motion, frame, num_frames)
        stem = f"hold_t{seg['t_hold']:07.2f}_{clip}"
        torch.save(held, out_dir / f"{stem}.motion")
        entries.append(
            {
                "stem": stem,
                "file": f"{out_dir.name}/{stem}.motion",
                "source_stem": clip,
                "config": config,
                "t_hold_src": seg["t_hold"],
                "duration_s": round(num_frames / fps, 3),
                "node_total_dwell_s": round(node_dwell, 2),
            }
        )
        print(f"  {duration:5.1f}s  {config[:78]:78s} <- {clip[:44]}")

    manifest = {
        "graph_dir": args.graph_dir,
        "corpus": args.corpus,
        "seed": args.seed,
        "holds": entries,
    }
    manifest_path = out_dir.parent / f"{out_dir.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {len(entries)} clips to {out_dir} and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
