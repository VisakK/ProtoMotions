# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track a Stage-2 student run: the viz panel and the scalars, side by side.

Written after round 7 §11.5, where a hand-rolled partial extraction of the panel
produced a conclusion that had to be retracted. Two rules are baked in here so
the mistake is not repeatable:

* **Never report a single panel.** Each panel is one nucleus draw per sequence
  and is not seeded, so consecutive epochs are independent draws --
  `chain_handstand` swings 0.32 -> 0.16 -> 1.00 -> 0.00 -> 0.05 on successive
  panels. Everything below is aggregated over the epochs in the window, and the
  panel count is printed next to every number.
* **Never report `final_goal_iou` alone.** It is the mean over the *last* goal's
  hold window, so a policy that kicks up, holds hands-only for a second and then
  puts a foot down scores the same as one that never leaves the floor. Runs that
  carry the per-goal block (added at the same time as this script) also get
  `max_goal_best_iou` -- "did it ever get there, on any goal?" -- which is the
  number that separates an attempt from an absence.

The scalar side reports the ratio `env/action_gap/prior_mse / privileged_mse`,
which is how much a *wrong* intent code costs the action. One caveat printed with
it: it is **not comparable across the goal-conditioned-trunk change** -- with the
goal in the trunk a low ratio means the decoder can route around the code;
without it the code is the only goal->action path, so a low ratio means something
else.

Usage::

    PYTHONPATH=. python data/scripts/track_student_run.py \
      --runs smpl_yogi_contact_graph_student_s2_v7_1_nogoaltrunk \
             smpl_yogi_contact_graph_student_s2_v7_fsq_goaltrunk \
             smpl_yogi_contact_graph_student_s2_v6_fsq \
      --from-epoch 500 --to-epoch 2500 [--sequences handstand scorpion]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

SCALARS = [
    "env/action_gap/prior_mse",
    "env/action_gap/privileged_mse",
    "env/action_gap/latent_residual_l2",
    "model/fsq_code_perplexity",
    "model/fsq_full_match",
    "model/fsq_ce_loss_refresh",
    "eval/success_rate",
    "env/raw_r/diag_goal_pose_error_mean",
    "eval/action_rate_mean_rad_s",
]


def load_panels(run: str, lo: int, hi: int) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {}
    for path in glob.glob(f"results/{run}/viz/epoch_*/summary.json"):
        epoch = int(re.search(r"epoch_(\d+)", path).group(1))
        if not (lo <= epoch <= hi):
            continue
        try:
            rows = json.loads(Path(path).read_text())
        except Exception:
            continue
        for row in rows:
            out.setdefault(row["sequence"], {})[epoch] = row
    return out


def load_scalars(run: str) -> Dict[str, Dict[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    root = f"results/{run}/lightning_logs"
    versions = sorted(glob.glob(f"{root}/version_*"))
    if not versions:
        return {}
    acc = EventAccumulator(versions[-1], size_guidance={"scalars": 200000})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    return {t: {e.step: e.value for e in acc.Scalars(t)} for t in SCALARS if t in tags}


def newest_at(series: Dict[int, float], epoch: Optional[int]) -> Optional[float]:
    if not series:
        return None
    keys = [k for k in series if epoch is None or k <= epoch]
    return series[max(keys)] if keys else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--from-epoch", type=int, default=0)
    ap.add_argument("--to-epoch", type=int, default=10**9)
    ap.add_argument(
        "--sequences", nargs="*", default=None,
        help="substrings; default is every sequence in the panel",
    )
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    panels = {r: load_panels(r, args.from_epoch, args.to_epoch) for r in args.runs}
    scalars = {r: load_scalars(r) for r in args.runs}
    short = {r: r.replace("smpl_yogi_contact_graph_student_s2_", "") for r in args.runs}

    # ---- scalars, at the newest epoch <= --to-epoch --------------------- #
    print(f"SCALARS at the newest epoch <= {args.to_epoch}")
    print(f"{'metric':40s}" + "".join(f"{short[r][:14]:>16s}" for r in args.runs))
    a = [newest_at(scalars[r].get("env/action_gap/prior_mse", {}), args.to_epoch) for r in args.runs]
    b = [newest_at(scalars[r].get("env/action_gap/privileged_mse", {}), args.to_epoch) for r in args.runs]
    cells = [f"{x / y:16.2f}" if x and y else f"{'-':>16s}" for x, y in zip(a, b)]
    print(f"{'RATIO prior/privileged':40s}" + "".join(cells))
    for tag in SCALARS[2:]:
        vals = [newest_at(scalars[r].get(tag, {}), args.to_epoch) for r in args.runs]
        print(
            f"{tag:40s}"
            + "".join(f"{v:16.4g}" if v is not None else f"{'-':>16s}" for v in vals)
        )
    print(
        "\n  NOTE: the ratio is NOT comparable across the goal-conditioned-trunk change.\n"
        "  With the goal in the trunk a low value means the decoder routes around the\n"
        "  code; without it the code is the only goal->action path (round 7 §11.5).\n"
    )

    # ---- panel, aggregated --------------------------------------------- #
    names = sorted({s for r in args.runs for s in panels[r]})
    if args.sequences:
        names = [n for n in names if any(k.lower() in n.lower() for k in args.sequences)]
    print(f"PANEL, epochs {args.from_epoch}-{args.to_epoch}, aggregated over draws")
    print("  cells: mean final_goal_iou / exact-rate / mean max_goal_best_iou / n panels")
    print(f"{'sequence':38s}" + "".join(f"{short[r][:20]:>26s}" for r in args.runs))
    report: Dict[str, dict] = {}
    for name in names:
        cells = []
        for run in args.runs:
            rows = list(panels[run].get(name, {}).values())
            if not rows:
                cells.append(f"{'-':>26s}")
                continue
            iou = [r["final_goal_iou"] for r in rows if r.get("final_goal_iou") is not None]
            ex = [1.0 if r.get("reached_exact") else 0.0 for r in rows]
            best = [
                max((g["best_iou"] for g in r["per_goal"]), default=float("nan"))
                for r in rows
                if r.get("per_goal")
            ]
            m_iou = sum(iou) / len(iou) if iou else float("nan")
            m_ex = sum(ex) / len(ex) if ex else float("nan")
            m_best = sum(best) / len(best) if best else float("nan")
            report.setdefault(name, {})[run] = {
                "mean_final_goal_iou": m_iou,
                "exact_rate": m_ex,
                "mean_max_goal_best_iou": m_best,
                "n_panels": len(rows),
            }
            cells.append(
                f"{m_iou:6.2f} /{m_ex:5.2f} /"
                + (f"{m_best:6.2f}" if m_best == m_best else "   n/a")
                + f" /{len(rows):2d}"
            )
        print(f"{name[:38]:38s}" + "".join(f"{c:>26s}" for c in cells))

    print(
        "\n  'n/a' in the third column means the run predates the per-goal block, so an\n"
        "  attempt that was not held cannot be distinguished from no attempt at all."
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
