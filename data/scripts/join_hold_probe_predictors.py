# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does frozen-hold coverage decide how long a commanded pose is held?

``make_hold_probe_plans.py`` emits matched pairs: for each contact
configuration, one 12 s hold probe at the pose ``make_hold_motions.py`` cut its
frozen ``hold_`` clip from (``variant: longest``, ``frozen_gap_m`` ~ 0) and one
at a *different* pose of the **same** configuration (``variant: second``, no
frozen supervision at that pose).  Node, contact set, orientation bin and probe
protocol are identical within a pair; the confound to watch is that the longest
segment is usually also the longer *live* demonstration, which this reports so
the near-matched pairs can be read separately.

Prints the paired contrast, the within-pair difference, and — because the
paired design is the point — a sign test over pairs rather than a correlation
over plans.

Usage::

    PYTHONPATH=. python data/scripts/join_hold_probe_predictors.py \\
      --plans data/scripts/plans/v9_holds \\
      --panel output/v9_paired/last/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def sign_test(differences) -> float:
    """Two-sided exact binomial p for 'the longest variant holds longer'."""
    positive = sum(1 for d in differences if d > 0)
    negative = sum(1 for d in differences if d < 0)
    n = positive + negative
    if n == 0:
        return float("nan")
    k = max(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--matched-within-s", type=float, default=1.0,
                        help="a pair counts as live-dwell-matched when the two "
                             "segments differ by less than this")
    args = parser.parse_args()

    meta = {}
    for path in sorted(Path(args.plans).glob("*.json")):
        plan = json.loads(path.read_text())
        if "_meta" not in plan:
            continue
        meta[path.stem] = plan["_meta"]
    panel = {
        r["sequence"]: r
        for r in json.loads(Path(args.panel).read_text())
    }

    pairs = {}
    for name, m in meta.items():
        row = panel.get(name)
        if row is None:
            continue
        goal = (row.get("per_goal_agg") or [{}])[0]
        pairs.setdefault(m["config"], {})[m["variant"]] = {
            "plan": name,
            "held_s": goal.get("time_held_s_p50"),
            "hold_rate": row.get("hold_success_rate"),
            "err_p50": row.get("final_goal_pose_err_p50"),
            "reach_rate": goal.get("reach_rate"),
            "segment_s": m["segment_s"],
            "gap": m["frozen_gap_m"],
        }

    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    print(f"{len(complete)} complete pairs of {len(pairs)} configurations\n")
    header = (
        f"{'configuration':44s} {'seg_l':>6} {'seg_s':>6} {'gap_s':>6} "
        f"{'held_l':>7} {'held_s':>7} {'Δheld':>7} {'rate_l':>7} {'rate_s':>7}"
    )
    print(header)
    print("-" * len(header))
    deltas, matched = [], []
    for config, both in sorted(
        complete.items(), key=lambda kv: -(kv[1]["longest"]["segment_s"])
    ):
        a, b = both["longest"], both["second"]
        if a["held_s"] is None or b["held_s"] is None:
            continue
        delta = a["held_s"] - b["held_s"]
        deltas.append(delta)
        if abs(a["segment_s"] - b["segment_s"]) < args.matched_within_s:
            matched.append(delta)
        print(
            f"{config[:44]:44s} {a['segment_s']:6.2f} {b['segment_s']:6.2f} "
            f"{(b['gap'] if b['gap'] is not None else float('nan')):6.3f} "
            f"{a['held_s']:7.2f} {b['held_s']:7.2f} {delta:+7.2f} "
            f"{_f(a['hold_rate']):>7} {_f(b['hold_rate']):>7}"
        )
    print("-" * len(header))
    if deltas:
        print(f"\nall {len(deltas)} pairs:  median Δheld = {np.median(deltas):+.2f} s, "
              f"longest-variant holds longer on {sum(d > 0 for d in deltas)}/"
              f"{len(deltas)}  (sign test p = {sign_test(deltas):.3f})")
    if matched:
        print(f"live-dwell-matched pairs (|Δsegment| < {args.matched_within_s} s), "
              f"n = {len(matched)}: median Δheld = {np.median(matched):+.2f} s, "
              f"longer on {sum(d > 0 for d in matched)}/{len(matched)}  "
              f"(sign test p = {sign_test(matched):.3f})")
        print("  ^ this is the controlled comparison: same configuration, same "
              "length of live demonstration, only the frozen hold differs.")

    # And the plain correlation, for the record.
    rows = [
        (v["gap"], v["held_s"], v["segment_s"])
        for both in pairs.values() for v in both.values()
        if v["held_s"] is not None and v["gap"] is not None
    ]
    if len(rows) >= 5:
        gap = np.array([r[0] for r in rows])
        held = np.array([r[1] for r in rows])
        seg = np.array([r[2] for r in rows])
        print(f"\nover all {len(rows)} probes: "
              f"corr(frozen gap, seconds held) = {np.corrcoef(gap, held)[0, 1]:+.3f}; "
              f"corr(live segment length, seconds held) = "
              f"{np.corrcoef(seg, held)[0, 1]:+.3f}")
    return 0


def _f(value) -> str:
    return "  -  " if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
