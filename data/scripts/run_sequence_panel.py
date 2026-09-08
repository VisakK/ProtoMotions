# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the sequence panel once, off a checkpoint, headless — with replicas.

``SequenceVizRunner`` already drives every plan concurrently: env ``e`` follows
sequence ``e % S``, so at 1024 envs and 28 plans each plan is rolled out ~36
times per call.  In training only the first row of each was ever scored, which
is why every panel scalar was one unseeded nucleus draw
(``notes/Student_improvement_round7_1.MD`` §5.2) — the FSQ student's
``forward_inference`` runs the *sampled* token stream, so a probe is one draw of
its commitment behaviour.  Scoring all the replicas turns the same rollout into
a success **rate**, and this script exposes it as a standalone measurement so a
finished run can be swept without retraining:

* the **stay-vs-go** sweep — ``--hold-lead`` — because a pinned goal's deadline
  is the one channel that says whether the command is about to change;
* the **stream** sweep — ``--overrides agent.model.fsq.inference_argmax=True``
  or ``... fsq.temperature=0.7 fsq.top_p=0.8`` — because a 12 s hold is 45
  independent nucleus draws.

Usage::

    PYTHONPATH=. python data/scripts/run_sequence_panel.py \\
      --checkpoint results/<run>/last.ckpt --headless \\
      --num-envs 1024 --out-dir output/panel/base \\
      --plans data/scripts/plans/hold_probe_standing.json ...

Writes ``<out-dir>/summary.json`` (per sequence: replica-0 scalars, the
percentiles and rates over replicas, and a per-goal aggregate) plus one
stick-figure video per sequence from replica 0.  ``--no-video`` skips the
matplotlib pass, which is most of the wall clock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--plans", type=str, nargs="+", required=True,
                    help="plan JSONs, in panel order")
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--max-seconds", type=float, default=24.0)
parser.add_argument("--hold-lead", type=float, default=1.2,
                    help="deadline held during a goal's hold phase. Training's "
                         "deadline is a strict countdown, so a pinned goal in "
                         "[0.7, 1.4] s sits where no training goal ever stayed "
                         "for more than 0.70 s.")
parser.add_argument("--hold-lead-mode", choices=("clamp", "park"),
                    default="clamp",
                    help="clamp = max(remaining, hold_lead) (shipped); park "
                         "substitutes hold_lead only after the reach window "
                         "expires, so a large lead does not also inflate the "
                         "reach deadlines of a multi-goal plan")
parser.add_argument("--reissue-every", type=float, default=0.5)
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--max-replicas", type=int, default=0,
                    help="0 = score every env (num_envs // num_plans per plan)")
parser.add_argument("--pose-arrive-m", type=float, default=0.15)
parser.add_argument("--pose-depart-m", type=float, default=0.30)
parser.add_argument("--legacy-settle", action="store_true",
                    help="control arm: pre-fix panel protocol (settle under "
                         "the clip schedule, install the goal after, never flush)")
parser.add_argument("--dump-traces", action="store_true",
                    help="write pose_error_traces.npz for survival analysis")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--label", type=str, default=None)
args = parser.parse_args()

args.headless = True

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("run_sequence_panel")


def main() -> int:
    torch.manual_seed(args.seed)
    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]

    from protomotions.agents.evaluators.sequence_viz import (
        SequenceVizConfig,
        SequenceVizRunner,
    )

    config = SequenceVizConfig(
        viz_every=1,
        plan_files=list(args.plans),
        num_sequences=len(args.plans),
        # Plans only: the hold and edge-round-trip generators would pad the set
        # with sequences the sweep is not about.
        num_hold_sequences=0,
        max_seconds=args.max_seconds,
        settle_steps=args.settle_steps,
        reissue_every_s=args.reissue_every,
        hold_lead_s=args.hold_lead,
        hold_lead_mode=args.hold_lead_mode,
        max_replicas=args.max_replicas,
        pose_arrive_m=args.pose_arrive_m,
        pose_depart_m=args.pose_depart_m,
        legacy_settle=args.legacy_settle,
        dump_traces=args.dump_traces,
        log_scalars=False,
    )
    runner = SequenceVizRunner(agent, config)
    log.info(
        "%d sequences over %d envs -> ~%d replicas each",
        len(runner.sequences), env.num_envs,
        max(env.num_envs // max(len(runner.sequences), 1), 1),
    )
    if args.no_video:
        # The matplotlib pass is most of the wall clock and a sweep does not
        # look at it; the scoring is unaffected.
        import protomotions.agents.evaluators.sequence_viz as viz_module

        viz_module.render_stick_video = lambda *a, **k: None

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _videos, _scalars = runner.run(0, out_dir=out_dir)
    log.info("panel done in %.1f s -> %s", time.time() - started, out_dir)

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "label": args.label or out_dir.name,
        "num_envs": int(env.num_envs),
        "hold_lead_s": args.hold_lead,
        "hold_lead_mode": args.hold_lead_mode,
        "reissue_every_s": args.reissue_every,
        "settle_steps": args.settle_steps,
        "max_seconds": args.max_seconds,
        "pose_arrive_m": args.pose_arrive_m,
        "seed": args.seed,
        "legacy_settle": bool(args.legacy_settle),
        "overrides": list(args.overrides),
        "plans": [str(Path(p).name) for p in args.plans],
    }
    fsq = getattr(getattr(agent.model, "config", None), "fsq", None)
    if fsq is not None:
        meta["fsq"] = {
            "temperature": float(fsq.temperature),
            "top_p": float(fsq.top_p),
            "inference_argmax": bool(fsq.inference_argmax),
            "chunk_steps": int(fsq.chunk_steps),
            "intent_hysteresis": float(
                getattr(fsq, "intent_hysteresis", 0.0) or 0.0
            ),
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    rows = json.loads((out_dir / "summary.json").read_text())
    print(f"\n=== {meta['label']}  ({meta.get('fsq', {})})")
    print(f"{'sequence':34s} {'n':>3} {'hold_rate':>9} {'p50 err':>8} "
          f"{'p10':>6} {'p90':>6} {'exact':>6} {'minGoal':>8}")
    for row in sorted(rows, key=lambda r: r.get("hold_success_rate") or -1):
        print(
            f"{row['sequence'][:34]:34s} {row.get('replicas', 0):3d} "
            f"{_fmt(row.get('hold_success_rate')):>9} "
            f"{_fmt(row.get('final_goal_pose_err_p50')):>8} "
            f"{_fmt(row.get('final_goal_pose_err_p10')):>6} "
            f"{_fmt(row.get('final_goal_pose_err_p90')):>6} "
            f"{_fmt(row.get('reached_exact_rate')):>6} "
            f"{_fmt(row.get('min_goal_hold_rate')):>8}"
        )
    return 0


def _fmt(value) -> str:
    return "  -  " if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
