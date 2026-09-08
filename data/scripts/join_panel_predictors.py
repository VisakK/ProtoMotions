# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join a panel's MEASURED hold rates onto the corpus predictors that claim to
explain them.

``probe_plan_feasibility.py --json-out`` writes, per plan goal, four things the
round-9 diagnosis says decide whether a commanded pose can be held:

* ``edge_count``            — how often the corpus demonstrates the hop into it
* ``hold_clip_pose_gap_m``  — distance to the nearest frozen ``hold_`` clip for
                              that configuration (the only long-lead "stay"
                              supervision in the corpus)
* ``goal_near`` / ``d_centroid_m`` — how specific the goal is inside its node
* ``p_lead_lt_1_5``         — P(the commanded deadline is under 1.5 s | the body
                              is already at this pose and this node is
                              commanded): the learned stay-vs-go prior

``run_sequence_panel.py`` writes the measured counterpart: ``hold_rate`` per
goal over ~36 replicas.  This joins the two and reports Spearman correlations,
so the diagnosis is testable rather than illustrative.

Usage::

    PYTHONPATH=. python data/scripts/join_panel_predictors.py \\
      --predictors output/v9_tier0/plan_predictors.json \\
      --panel output/v9_tier0/base/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spearman(x, y) -> float:
    """Rank correlation, ties averaged; no scipy dependency."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if x.size < 3:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(v.size, dtype=float)
        ranks[order] = np.arange(v.size, dtype=float)
        # average ties so a constant predictor cannot look informative
        for value in np.unique(v):
            mask = v == value
            ranks[mask] = ranks[mask].mean()
        return ranks

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictors", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    predictors = json.loads(Path(args.predictors).read_text())
    panel = {r["sequence"]: r for r in json.loads(Path(args.panel).read_text())}

    rows = []
    for plan, goals in predictors.items():
        measured = panel.get(plan)
        if measured is None:
            continue
        agg = {g["goal"]: g for g in measured.get("per_goal_agg", [])}
        for index, goal in enumerate(goals):
            hit = agg.get(goal["goal"])
            if hit is None or hit.get("hold_rate") is None:
                continue
            rows.append({
                "plan": plan,
                "goal": goal["goal"],
                "final": index == len(goals) - 1,
                "hold_rate": float(hit["hold_rate"]),
                "reach_rate": hit.get("reach_rate"),
                "hold_pose_err_p50": hit.get("hold_pose_err_p50"),
                "time_held_s_p50": hit.get("time_held_s_p50"),
                **{k: goal[k] for k in (
                    "edge_count", "hold_clip_pose_gap_m", "goal_near",
                    "d_centroid_m", "p_lead_lt_1_5", "p_lead_gt_3",
                    "frozen_share", "node_members", "reach_s", "hold_s",
                    "native_gap_s",
                )},
            })

    # Conditional rate: among replicas that held the PREVIOUS goal, how many
    # hold this one? A chain's later goals otherwise read 0 whenever the policy
    # fell earlier, which is a property of the plan, not of the goal.
    arrive = 0.15
    for plan, goals in predictors.items():
        measured = panel.get(plan)
        if measured is None:
            continue
        agg = measured.get("per_goal_agg", [])
        by_name = {g["goal"]: g for g in agg}
        for index, goal in enumerate(goals):
            row = next(
                (r for r in rows
                 if r["plan"] == plan and r["goal"] == goal["goal"]), None
            )
            if row is None:
                continue
            here = by_name.get(goal["goal"], {}).get("hold_pose_err_by_replica")
            if here is None:
                continue
            if index == 0:
                row["hold_rate_cond"] = row["hold_rate"]
                row["n_cond"] = len(here)
                continue
            previous = by_name.get(
                goals[index - 1]["goal"], {}
            ).get("hold_pose_err_by_replica")
            if previous is None or len(previous) != len(here):
                continue
            survived = [
                h for h, p_ in zip(here, previous)
                if p_ is not None and p_ < arrive
            ]
            row["n_cond"] = len(survived)
            row["hold_rate_cond"] = (
                float(np.mean([h is not None and h < arrive for h in survived]))
                if survived else None
            )

    if not rows:
        raise SystemExit("no plan/goal joined: are the two files from the same set?")

    print(f"{len(rows)} (plan, goal) rows joined\n")
    header = (
        f"{'plan':30s} {'goal':16s} {'held':>5} {'reach':>6} {'errp50':>7} "
        f"{'edge':>5} {'holdgap':>8} {'near':>6} {'dcent':>6} {'P<1.5':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: r["hold_rate"]):
        print(
            f"{row['plan'][:30]:30s} {row['goal'][:16]:16s} "
            f"{row['hold_rate']:5.2f} {_f(row['reach_rate']):>6} "
            f"{_f(row['hold_pose_err_p50']):>7} "
            f"{_i(row['edge_count']):>5} {_f(row['hold_clip_pose_gap_m']):>8} "
            f"{_f(row['goal_near']):>6} {_f(row['d_centroid_m']):>6} "
            f"{_f(row['p_lead_lt_1_5']):>6}"
        )

    conditional = [
        r for r in rows
        if r.get("hold_rate_cond") is not None and r.get("n_cond", 0) >= 8
    ]
    if len(conditional) >= 5:
        print(f"\nCONDITIONAL on holding the previous goal ({len(conditional)} "
              f"rows with n_cond >= 8):")
        t = [r["hold_rate_cond"] for r in conditional]
        for key, flip in (("hold_clip_pose_gap_m", -1), ("p_lead_lt_1_5", -1),
                          ("goal_near", +1), ("edge_count", +1),
                          ("d_centroid_m", -1), ("frozen_share", +1)):
            values = [
                r[key] if r[key] is not None else float("nan")
                for r in conditional
            ]
            rho = spearman(values, t)
            arrow = "" if np.isnan(rho) else (
                "  <-- as predicted" if rho * flip > 0 else "  (opposite sign)"
            )
            print(f"  {key:24s} rho = {rho:+.3f}{arrow}")

    print("\nSpearman rho against the measured per-goal hold rate")
    print("(sign is 'more of this predictor -> higher hold rate')")
    targets = [r["hold_rate"] for r in rows]
    for key, flip in (
        ("hold_clip_pose_gap_m", -1),   # closer frozen hold -> better
        ("p_lead_lt_1_5", -1),          # 'this command expires soon' -> worse
        ("p_lead_gt_3", +1),
        ("frozen_share", +1),
        ("goal_near", +1),
        ("d_centroid_m", -1),
        ("edge_count", +1),
        ("node_members", -1),
        ("reach_s", +1),
        ("hold_s", -1),
    ):
        values = [
            r[key] if r[key] is not None else float("nan") for r in rows
        ]
        rho = spearman(values, targets)
        arrow = "" if np.isnan(rho) else ("  <-- as predicted" if rho * flip > 0
                                          else "  (opposite sign)")
        print(f"  {key:24s} rho = {rho:+.3f}{arrow}")

    finals = [r for r in rows if r["final"]]
    if len(finals) >= 3:
        print(f"\nsame, restricted to each plan's FINAL goal ({len(finals)} rows) —")
        print("  the one `hold_success_rate` scores:")
        t = [r["hold_rate"] for r in finals]
        for key in ("hold_clip_pose_gap_m", "p_lead_lt_1_5", "goal_near",
                    "edge_count"):
            values = [
                r[key] if r[key] is not None else float("nan") for r in finals
            ]
            print(f"  {key:24s} rho = {spearman(values, t):+.3f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


def _f(value) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def _i(value) -> str:
    return "  -  " if value is None else str(int(value))


if __name__ == "__main__":
    raise SystemExit(main())
