# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus predictors for every hold-probe plan, and a spanning subset of them.

``probe_plan_feasibility.py`` prices one plan at a time and prints a report.
This tabulates the same quantities over a whole plan directory so a *set* of
commanded states can be chosen to span one predictor, which is what an
experiment like ``decode_all_codes.py`` needs: enumerating 625 codes at eight
states is only informative if those eight states differ in something measured.

The predictor of interest is **frozen share** -- of the training mass where this
node is commanded *and the body is already at this pose*, the fraction that came
from a frozen ``hold_`` clip, i.e. from the only place in the corpus where a
pose is commanded and the reference does not move.  It is a **node-level**
quantity and it is not the same thing as ``frozen_gap_m`` (the distance from the
commanded pose to the nearest frozen clip's pose), which
``Student_v9_tier0_report.MD`` §11 tested and refuted as a predictor of hold
survival.  Both are reported so the two can be told apart.

Usage::

    PYTHONPATH=. python data/scripts/frozen_share_table.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \\
      --plan-dir data/scripts/plans/v9_holds --extra data/scripts/plans/hold_probe_standing.json \\
      --select 8 --out-dir data/scripts/plans/z1_span
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_plan_feasibility import Corpus  # noqa: E402


def plan_rows(paths, corpus: Corpus, radius: float):
    rows = []
    for path in paths:
        try:
            plan = json.loads(Path(path).read_text())
        except Exception:
            continue
        goals = plan.get("goals") or []
        if len(goals) != 1:
            continue                      # single-goal hold probes only
        goal = goals[0]
        config = goal.get("config")
        if config not in corpus.node_id:
            continue
        node = corpus.node_id[config]
        try:
            motion = corpus.motion_id(Path(str(goal["pose_clip"])).stem)
        except KeyError:
            continue
        target = corpus.pose_at(motion, float(goal["pose_time"]))
        score = corpus.score_goal(node, target, radius)
        frozen = corpus.hold_clip_for(config, target)
        rows.append(dict(
            plan=Path(path).stem, path=str(path), config=config, node=node,
            pose_clip=Path(str(goal["pose_clip"])).stem,
            pose_time=float(goal["pose_time"]),
            frozen_share=score["frozen_share"],
            frozen_gap_m=None if frozen is None else round(frozen[1], 4),
            has_frozen=frozen is not None,
            at_pose_rate=score["at_pose_rate"],
            at_pose_share=score["at_pose_share"],
            node_share=score["node_share"],
            goal_near=score["goal_near"],
            d_centroid=score["d_centroid"],
            p_lead_lt_1_5=score["p_lead_lt_1_5"],
            members=score["members"],
        ))
    return rows


def spanning(rows, count: int, min_mass: float):
    """Pick ``count`` rows spread over frozen share, one per configuration.

    ``at_pose_share`` gates: a frozen share computed from almost no training
    mass is not a measurement of anything, and those rows would otherwise
    dominate the extremes of the range.
    """
    usable = [
        r for r in rows
        if np.isfinite(r["frozen_share"]) and r["at_pose_share"] >= min_mass
    ]
    usable.sort(key=lambda r: r["frozen_share"])
    seen, unique = set(), []
    for row in usable:
        if row["config"] in seen:
            continue
        seen.add(row["config"])
        unique.append(row)
    if len(unique) <= count:
        return unique
    # Even quantiles of the frozen-share order, so the subset spans the range
    # rather than clustering wherever the corpus happens to be dense.
    picks = np.linspace(0, len(unique) - 1, count).round().astype(int)
    return [unique[i] for i in dict.fromkeys(picks.tolist())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--plan-dir", type=str, default=None)
    parser.add_argument("--extra", type=str, nargs="*", default=[])
    parser.add_argument("--radius", type=float, default=0.15,
                        help="'at this pose' radius, the panel's pose_arrive_m")
    parser.add_argument("--min-at-pose-share", type=float, default=1e-4,
                        help="training mass floor for a frozen share to mean anything")
    parser.add_argument("--select", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="copy the selected plans here")
    parser.add_argument("--lead-s", type=float, default=0.2,
                        help="ContactGraphControl.min_lead_s")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text())
    corpus = Corpus(graph, args.motion_file, args.mjcf, args.lead_s)
    paths = sorted(Path(args.plan_dir).glob("*.json")) if args.plan_dir else []
    paths += [Path(p) for p in args.extra]
    rows = plan_rows(paths, corpus, args.radius)
    rows.sort(key=lambda r: (-1.0 if not np.isfinite(r["frozen_share"])
                             else r["frozen_share"]))

    print(f"{'plan':<34}{'frzShare':>9}{'frzGap':>8}{'atPose':>8}"
          f"{'mass':>9}{'goalNear':>9}{'dCentr':>8}{'lead<1.5':>9}  config")
    for r in rows:
        gap = "  none" if r["frozen_gap_m"] is None else f"{r['frozen_gap_m']:6.3f}"
        print(f"{r['plan'][:33]:<34}{r['frozen_share']:>9.3f}{gap:>8}"
              f"{r['at_pose_rate']:>8.3f}{r['at_pose_share']:>9.5f}"
              f"{r['goal_near']:>9.3f}{r['d_centroid']:>8.3f}"
              f"{r['p_lead_lt_1_5']:>9.3f}  {r['config'][:44]}")

    selected = spanning(rows, args.select, args.min_at_pose_share) if args.select else []
    if selected:
        print(f"\nspanning subset ({len(selected)}):")
        for r in selected:
            gap = "none" if r["frozen_gap_m"] is None else f"{r['frozen_gap_m']:.3f}"
            print(f"  frozen_share {r['frozen_share']:.3f}  gap {gap:>6}  {r['plan']}")
        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            for r in selected:
                shutil.copy(r["path"], out / Path(r["path"]).name)
            print(f"copied {len(selected)} plans to {out}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            dict(rows=rows, selected=[r["plan"] for r in selected]), indent=1))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
