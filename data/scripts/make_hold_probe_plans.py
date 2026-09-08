# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate one 12 s hold probe per contact configuration, for a controlled sweep.

The four hand-written hold probes rank exactly by how close the corpus's only
sustained-hold supervision is to the pose they command (side plank 0.00 m,
dolphin plank 0.00 m, plank 0.17 m, standing 0.47 m; survival 12.9 / 12.9 /
5.8 / 4.2 s).  That is four points, three of them chosen by hand, and it was
derived from the same clips it is being tested on.

This emits the same probe for *every* configuration the graph holds long enough
to cut one from, so the correlation can be measured over 20-40 of them in a
single batched panel — and each plan carries the two predictors in its
``_meta`` so ``join_hold_probe_predictors`` can regress survival on them
without re-deriving anything:

* ``frozen_gap_m``  — distance to the nearest frozen ``hold_`` clip's pose, in
  the 6-body goal representation. ``null`` when the corpus has none.
* ``corpus_dwell_s`` — the longest trusted segment the corpus demonstrates at
  this configuration, i.e. how long anyone has ever been seen holding it.

``--variant`` is what makes this a controlled test rather than a survey.
``make_hold_motions.select_holds`` cuts its frozen hold from each
configuration's **longest** segment, so ``--variant longest`` commands exactly
the pose that has 6-14 s of frozen supervision (``frozen_gap_m`` ~ 0) while
``--variant second`` commands a *different* pose of the **same configuration**,
which has none. Contact set, node, orientation bin and probe protocol are held
fixed; only whether the commanded pose was ever demonstrated as a sustained
hold changes. That is the round-9 §2 hypothesis as a paired experiment.

Every probe starts *at* its own pose (``start`` = the segment's own hold frame),
so arrival is not part of what is being measured.

Usage::

    PYTHONPATH=. python data/scripts/make_hold_probe_plans.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \\
      --out-dir data/scripts/plans/v9_holds --max 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_pose_separation import (  # noqa: E402
    CONDITIONABLE,
    GoalPoses,
    mean_body_distance,
)


def is_frozen(clip: str) -> bool:
    return os.path.basename(clip).startswith("hold_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max", type=int, default=30)
    parser.add_argument("--min-dwell-s", type=float, default=1.5,
                        help="skip configurations no clip demonstrates this long")
    parser.add_argument("--min-good-fraction", type=float, default=0.9,
                        help="cut only from segments the expert tracked, as "
                             "make_hold_motions.py does")
    parser.add_argument("--variant", choices=("longest", "second"),
                        default="longest",
                        help="which segment of each configuration to command: "
                             "the longest (what the frozen hold was cut from) "
                             "or the second-longest (same configuration, no "
                             "frozen supervision at that pose)")
    parser.add_argument("--configs", type=str, default=None,
                        help="file with one configuration string per line, to "
                             "pin both variants to the same set")
    parser.add_argument("--reach-s", type=float, default=1.0)
    parser.add_argument("--hold-s", type=float, default=12.0)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text())
    poses = GoalPoses(args.motion_file, args.mjcf)
    body_ids = poses.body_ids(CONDITIONABLE)

    # Best live segment per configuration, and every frozen hold's pose.
    best: dict = {}
    dwell: dict = {}
    frozen: list = []
    for clip, entry in graph["clips"].items():
        motion = entry["motion_id"]
        for seg in entry["segments"]:
            if not seg.get("trusted", True):
                continue
            pose = poses.at(motion, float(seg["t_hold"]), body_ids)
            if is_frozen(clip):
                frozen.append(pose)
                continue
            if seg.get("good_fraction", 1.0) < args.min_good_fraction:
                continue
            key = seg["config"]
            dwell[key] = max(dwell.get(key, 0.0), float(seg["duration_s"]))
            best.setdefault(key, []).append((clip, seg, pose))
    for key, entries in best.items():
        entries.sort(key=lambda e: -e[1]["duration_s"])

    frozen_stack = np.stack(frozen) if frozen else None
    rank_index = 0 if args.variant == "longest" else 1
    eligible = [
        k for k in best
        if dwell[k] >= args.min_dwell_s and len(best[k]) > rank_index
        and best[k][rank_index][1]["duration_s"] >= args.min_dwell_s
    ]
    if args.configs:
        wanted = [
            line.strip() for line in Path(args.configs).read_text().splitlines()
            if line.strip()
        ]
        eligible = [k for k in wanted if k in eligible]
    ranked = sorted(eligible, key=lambda k: -dwell[k])[: args.max]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, key in enumerate(ranked):
        clip, seg, pose = best[key][rank_index]
        gap = (
            float(min(mean_body_distance(pose, f) for f in frozen_stack))
            if frozen_stack is not None else None
        )
        # Start a little before the hold frame so the probe settles into the
        # pose rather than being teleported onto it mid-frame.
        start = max(float(seg["t_start"]), float(seg["t_hold"]) - 0.5)
        name = f"hp{index:02d}{args.variant[0]}_" + (
            key.replace(":G", "").replace("|", "_").replace("+", "x")
               .replace("@", "_")[:38]
        )
        plan = {
            "_comment": [
                "Generated by make_hold_probe_plans.py: a 12 s commanded hold of "
                "the corpus's longest demonstration of this contact configuration, "
                "started at that pose. Arrival is not what this measures.",
                f"configuration: {key}",
            ],
            "_meta": {
                "config": key,
                "variant": args.variant,
                "segment_s": round(float(seg["duration_s"]), 3),
                "source_clip": clip,
                "t_hold": round(float(seg["t_hold"]), 3),
                "corpus_dwell_s": round(float(dwell[key]), 3),
                "frozen_gap_m": None if gap is None else round(gap, 4),
                "has_frozen_hold": bool(gap is not None and gap < 0.05),
            },
            "start": {"clip": clip, "time": round(start, 3)},
            "goals": [{
                "name": "hold",
                "config": key,
                "pose_clip": clip,
                "pose_time": round(float(seg["t_hold"]), 3),
                "reach_s": args.reach_s,
                "hold_s": args.hold_s,
            }],
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(plan, indent=1))
        written.append((name, key, dwell[key], gap))

    print(f"wrote {len(written)} plans to {out_dir}\n")
    print(f"{'plan':44s} {'dwell':>7} {'frozen gap':>11}")
    for name, key, d, gap in written:
        print(f"{name[:44]:44s} {d:7.2f} "
              + ("       none" if gap is None else f"{gap:11.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
