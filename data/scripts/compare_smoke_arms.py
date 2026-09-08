# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the training scalars of several short distillation arms.

Built for the XY-contract smoke (``notes/.../Student_v9_goal_causality_investigation.MD``
§18): three runs that differ only in whether the student's pose inputs are
floor-free (``root_relative_xy``) and whether the reference is re-anchored to the
robot each step (``realign_motion_with_humanoid_on_each_step``).

Reads the TensorBoard event files directly, so it does not need wandb, and prints
each scalar at matched epochs rather than at whatever step each run last logged.

One reading caveat is worth stating at the top: ``eval/success_rate`` and
``eval/gt_error`` are **not comparable across the re-anchoring arms**, because
re-anchoring removes the horizontal component from the very error those metrics
score.  ``supervised/loss`` is comparable -- it is "how well can the student
predict its own teacher" in every arm.

Usage::

    PYTHONPATH=. python data/scripts/compare_smoke_arms.py \\
      --runs results/smoke_xy_A_off_off results/smoke_xy_B_on_norea \\
             results/smoke_xy_C_on_realign --out-dir output/smoke_xy
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

DEFAULT_TAGS = (
    "supervised/loss",
    "supervised/model_loss",
    "model/fsq_full_match",
    "model/fsq_code_perplexity",
    "model/fsq_token_accuracy_refresh",
    "model/fsq_ce_loss_refresh",
    "info/episode_length",
    "eval/success_rate",
    "eval/gt_error/mean",
)


def load(run: Path) -> dict:
    from tensorboard.backend.event_processing import event_accumulator

    files = sorted(glob.glob(str(run / "lightning_logs" / "version_*" / "events*")))
    if not files:
        raise SystemExit(f"no event file under {run}")
    series = {}
    for path in files:
        ea = event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                series.setdefault(tag, {})[e.step] = e.value
    return series


def at(series: dict, tag: str, epoch: int, window: int = 10):
    """Median of the tag over [epoch-window, epoch], so a noisy scalar is readable."""
    if tag not in series:
        return None
    values = [v for s, v in series[tag].items() if epoch - window <= s <= epoch]
    return float(np.median(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, nargs="+",
                        default=[25, 50, 100, 150, 200, 250])
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    runs = {Path(r).name: load(Path(r)) for r in args.runs}
    report = {"runs": {}, "epochs": args.epochs}
    for tag in args.tags:
        if not any(tag in s for s in runs.values()):
            continue
        print(f"\n--- {tag} ---")
        print(f"{'run':<26}" + "".join(f"{'e' + str(e):>13}" for e in args.epochs))
        for name, series in runs.items():
            values = [at(series, tag, e) for e in args.epochs]
            cells = "".join(
                (f"{v:>13.5g}" if v is not None else f"{'-':>13}") for v in values
            )
            print(f"{name[:25]:<26}{cells}")
            report["runs"].setdefault(name, {})[tag] = values

    final = args.epochs[-1]
    losses = {n: at(s, "supervised/loss", final) for n, s in runs.items()}
    losses = {n: v for n, v in losses.items() if v is not None}
    if len(losses) > 1:
        best = min(losses, key=losses.get)
        print(f"\nsupervised/loss at epoch {final}:")
        for name, value in sorted(losses.items(), key=lambda kv: kv[1]):
            print(f"  {name:<26}{value:.5g}"
                  + ("   <- lowest" if name == best else
                     f"   ({value / losses[best]:.2f}x)"))

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_arms.json").write_text(json.dumps(report, indent=1))
        print(f"\nwrote {out / 'smoke_arms.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
