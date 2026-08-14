# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-motion evaluation sweep: rank every clip in a library by tracking error.

``inference_agent.py --full-eval`` runs exactly this evaluation but prints only
corpus aggregates (``eval/gt_error/mean`` and friends), which cannot tell you
*which* clips the policy is failing.  The evaluator does hold the per-motion
breakdown -- ``_component_value_sum[name]`` and friends are indexed by motion id
-- but ``cleanup_after_evaluation()`` frees them before ``evaluate()`` returns.

So this wraps that teardown, snapshots the buffers, and writes a per-clip table.
The rollout itself is untouched: same evaluator, same batching, same metrics as
training-time eval, so the numbers here are directly comparable to the wandb
curves.

Output: a CSV ranked worst-first, plus a JSON of the aggregate log.  Feed the CSV
to ``render_policy_videos.py --worst N`` to look at the failures.

Usage::

    python data/scripts/eval_per_motion.py \
        --checkpoint results/smpl_yogi_easy128_contact_rich/inspect_snapshot.ckpt \
        --simulator isaaclab --headless --num-envs 128 \
        --overrides env.ref_respawn_offset=0.005 \
        --out results/smpl_yogi_easy128_contact_rich/per_motion_eval.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--out", type=str, default=None, help="CSV path")
parser.add_argument(
    "--sort-by", type=str, default="gt_error",
    help="metric column to rank by (default gt_error = mean tracking error, m)",
)
args = parser.parse_args()

# Simulator packages must be imported before torch.
from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build, motion_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def snapshot_per_motion(evaluator) -> dict:
    """Copy the evaluator's per-motion buffers before they are freed."""
    snap = {"components": {}}
    if evaluator._motion_failed is not None:
        snap["failed"] = evaluator._motion_failed.detach().cpu().clone()
        snap["evaluated"] = evaluator._eval_mask.detach().cpu().clone()
    for name in evaluator._component_value_sum:
        steps = evaluator._component_step_count[name].detach().cpu().clone()
        snap["components"][name] = {
            "mean": (
                evaluator._component_value_sum[name].detach().cpu()
                / steps.clamp(min=1).float()
            ).clone(),
            "min": evaluator._component_value_min[name].detach().cpu().clone(),
            "max": evaluator._component_value_max[name].detach().cpu().clone(),
            "steps": steps,
            "failed": evaluator._per_component_failures[name].detach().cpu().clone(),
        }
    return snap


def main() -> int:
    built = build(args, AppLauncher)
    agent, motion_lib = built["agent"], built["motion_lib"]
    evaluator = agent.evaluator

    names = motion_names(motion_lib)
    lengths = motion_lib.get_motion_length(None).detach().cpu()
    log.info("Evaluating %d motions (%.1f s total)", len(names), lengths.sum())

    captured = {}
    original_cleanup = evaluator.cleanup_after_evaluation

    def cleanup_with_snapshot():
        captured.update(snapshot_per_motion(evaluator))
        original_cleanup()

    evaluator.cleanup_after_evaluation = cleanup_with_snapshot

    try:
        evaluator.eval_count = 0
        evaluation_log, evaluated_score, num_eval_items = evaluator.evaluate()
    finally:
        evaluator.cleanup_after_evaluation = original_cleanup
        if hasattr(built["env"].simulator, "shutdown"):
            built["env"].simulator.shutdown()

    if not captured:
        log.error("No per-motion buffers captured -- evaluator returned early.")
        return 1

    comps = captured["components"]
    if args.sort_by not in comps:
        log.warning(
            "--sort-by %r not among %s; falling back to the first",
            args.sort_by, list(comps),
        )
    sort_key = args.sort_by if args.sort_by in comps else next(iter(comps))

    n = len(names)
    rows = []
    for i in range(n):
        row = {
            "motion_id": i,
            "clip": names[i],
            "length_s": round(float(lengths[i]), 3),
            "evaluated": bool(captured["evaluated"][i]),
            "failed": bool(captured["failed"][i]),
        }
        for name, d in comps.items():
            row[f"{name}_mean"] = round(float(d["mean"][i]), 6)
            row[f"{name}_max"] = round(float(d["max"][i]), 6)
            row[f"{name}_failed"] = bool(d["failed"][i])
            row[f"{name}_steps"] = int(d["steps"][i])
        rows.append(row)

    rows.sort(key=lambda r: (-int(r["failed"]), -r[f"{sort_key}_mean"]))

    out = Path(args.out or (Path(args.checkpoint).parent / "per_motion_eval.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    Path(str(out).replace(".csv", "_aggregate.json")).write_text(
        json.dumps(
            {
                "aggregate": {k: float(v) for k, v in evaluation_log.items()},
                "overall_score": (
                    float(evaluated_score) if evaluated_score is not None else None
                ),
                "num_evaluated": int(num_eval_items),
                "checkpoint": str(args.checkpoint),
                "sorted_by": f"{sort_key}_mean",
            },
            indent=1,
        )
    )

    n_failed = sum(r["failed"] for r in rows)
    print("\n" + "=" * 78)
    print(f"PER-MOTION EVALUATION  ({n} clips, {n_failed} failed, "
          f"success rate {1 - n_failed / max(n, 1):.3f})")
    print("=" * 78)
    print(f"{'#':>3} {'clip':<52} {sort_key:>9} {'fail':>5}")
    print("-" * 78)
    for k, r in enumerate(rows[:25]):
        print(f"{k + 1:>3} {r['clip'][:52]:<52} {r[f'{sort_key}_mean']:>9.4f} "
              f"{'YES' if r['failed'] else '':>5}")
    if n > 25:
        print(f"    ... {n - 25} more, best is {rows[-1]['clip'][:44]} "
              f"({rows[-1][f'{sort_key}_mean']:.4f})")
    print("-" * 78)
    for key in sorted(evaluation_log):
        print(f"  {key}: {evaluation_log[key]:.6f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        raise SystemExit(main())
