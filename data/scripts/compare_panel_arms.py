# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare sequence-panel arms: hold rate per plan, one column per arm.

Reads ``<dir>/<arm>/summary.json`` written by
``data/scripts/run_sequence_panel.py`` and prints the matrix the Tier-0 sweep
exists to produce.  ``hold_success_rate`` is the fraction of replicas whose
mean pose error over the FINAL goal's hold window is below ``pose_arrive_m``
(0.15 m) — the number the old panel could not report, because it scored one
replica and reported a minimum over time.

A rate is a binomial estimate over ``replicas`` draws; at ~36 draws the 95 %
interval is roughly ±0.16 at p = 0.5 and ±0.03 at p = 0.98, so read a change of
less than ~0.15 in the middle of the range as noise.

Usage::

    PYTHONPATH=. python data/scripts/compare_panel_arms.py --dir output/v9_tier0
    PYTHONPATH=. python data/scripts/compare_panel_arms.py --dir output/v9_tier0 \\
        --metric final_goal_pose_err_p50 --goals
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(root: Path):
    arms = {}
    for summary in sorted(root.glob("*/summary.json")):
        name = summary.parent.name
        rows = json.loads(summary.read_text())
        meta_path = summary.parent / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        arms[name] = {"rows": {r["sequence"]: r for r in rows}, "meta": meta}
    return arms


def wilson(successes: float, n: int) -> tuple:
    """95 % Wilson interval — a rate over ~36 draws needs its width shown."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--metric", default="hold_success_rate")
    parser.add_argument("--order", default=None,
                        help="comma-separated arm order (default: alphabetical)")
    parser.add_argument("--goals", action="store_true",
                        help="also print the per-goal hold rate for every plan")
    parser.add_argument("--ci", action="store_true",
                        help="annotate the baseline arm's 95 %% Wilson interval")
    args = parser.parse_args()

    root = Path(args.dir)
    arms = load(root)
    if not arms:
        raise SystemExit(f"no <arm>/summary.json under {root}")
    names = (
        [a for a in args.order.split(",") if a in arms]
        if args.order else sorted(arms)
    )
    sequences = sorted({s for a in arms.values() for s in a["rows"]})

    print(f"metric: {args.metric}")
    for name in names:
        meta = arms[name]["meta"]
        fsq = meta.get("fsq", {})
        print(
            f"  {name:12s} lead {meta.get('hold_lead_s', '?')}s"
            f"/{meta.get('hold_lead_mode', 'clamp')}  "
            f"T {fsq.get('temperature', '?')}/{fsq.get('top_p', '?')}  "
            f"argmax {fsq.get('inference_argmax', '?')}  "
            f"hyst {fsq.get('intent_hysteresis', 0.0)}  "
            f"legacy_settle {meta.get('legacy_settle', '?')}  "
            f"envs {meta.get('num_envs', '?')}"
        )
    print()
    header = f"{'sequence':36s}" + "".join(f"{n[:9]:>10}" for n in names)
    print(header)
    print("-" * len(header))
    def cell_value(row):
        """`hold_rate@X` recomputes the rate at another threshold from the
        stored per-replica errors; anything else is a plain field lookup."""
        if row is None:
            return None
        if args.metric.startswith("hold_rate@"):
            errs = row.get("final_goal_pose_err_by_replica")
            if not errs:
                return None
            threshold = float(args.metric.split("@", 1)[1])
            scored = [e for e in errs if e is not None]
            return (
                sum(e < threshold for e in scored) / len(scored)
                if scored else None
            )
        return row.get(args.metric)

    for sequence in sequences:
        cells = []
        for name in names:
            value = cell_value(arms[name]["rows"].get(sequence))
            cells.append("     -    " if value is None else f"{value:10.3f}")
        print(f"{sequence[:36]:36s}" + "".join(cells))
    print("-" * len(header))
    for label, fn in (("mean", lambda v: sum(v) / len(v)),):
        cells = []
        for name in names:
            vals = [
                v for v in (
                    cell_value(r) for r in arms[name]["rows"].values()
                ) if v is not None
            ]
            cells.append(f"{fn(vals):10.3f}" if vals else "     -    ")
        print(f"{label:36s}" + "".join(cells))

    if args.ci:
        base = names[0]
        print(f"\n95 % Wilson intervals for '{base}':")
        for sequence in sequences:
            row = arms[base]["rows"].get(sequence)
            if not row or row.get("hold_success_rate") is None:
                continue
            n = int(row.get("replicas", 0))
            lo, hi = wilson(row["hold_success_rate"] * n, n)
            print(
                f"  {sequence[:36]:36s} {row['hold_success_rate']:.3f} "
                f"[{lo:.2f}, {hi:.2f}]  n={n}"
            )

    if args.goals:
        print("\nper-goal hold rate (fraction of replicas whose mean pose error "
              "over that goal's own hold window is below pose_arrive_m)")
        for sequence in sequences:
            print(f"\n  {sequence}")
            goals = None
            for name in names:
                row = arms[name]["rows"].get(sequence)
                if row and row.get("per_goal_agg"):
                    goals = [g["goal"] for g in row["per_goal_agg"]]
                    break
            if not goals:
                continue
            print(f"    {'goal':26s}" + "".join(f"{n[:9]:>10}" for n in names))
            for gi, goal in enumerate(goals):
                cells = []
                for name in names:
                    row = arms[name]["rows"].get(sequence)
                    agg = (row or {}).get("per_goal_agg") or []
                    value = agg[gi].get("hold_rate") if gi < len(agg) else None
                    cells.append("     -    " if value is None else f"{value:10.3f}")
                print(f"    {goal[:26]:26s}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
