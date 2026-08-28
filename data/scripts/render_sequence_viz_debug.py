# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the in-training sequence-viz panel offline, from a checkpoint.

Drives exactly the code path ``SupervisedAgent._maybe_run_sequence_viz`` runs
during training — same ``SequenceVizRunner``, same sequence derivation against
the run's own graph, same goal-slot protocol — but standalone, so renderer
changes can be inspected on a finished checkpoint without launching (or
waiting for) a training run.  Built for the frozen-frame / bone-order debug
round of ``notes/wandb_video_logging.MD`` §5.

Usage::

    PYTHONPATH=. python data/scripts/render_sequence_viz_debug.py \
      --checkpoint results/smpl_yogi_contact_graph_student_s2_v4/last.ckpt \
      --out-dir output/video_debugging
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--out-dir", type=str, default="output/video_debugging")
parser.add_argument(
    "--plan-files", type=str, nargs="*",
    default=[
        "data/scripts/plans/standing_downdog.json",
        "data/scripts/plans/handstand_chain.json",
    ],
    help="Same default plans the training panel uses.",
)
parser.add_argument("--num-sequences", type=int, default=10)
parser.add_argument("--video-px", type=int, default=640)
parser.add_argument("--epoch", type=int, default=0,
                    help="Label only; stamped into the log line.")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

args.headless = True
if args.num_envs is None:
    args.num_envs = args.num_sequences

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build  # noqa: E402

from protomotions.agents.evaluators.sequence_viz import (  # noqa: E402
    SequenceVizConfig,
    SequenceVizRunner,
)


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    built = build(args, AppLauncher)
    agent, simulator = built["agent"], built["simulator"]

    config = SequenceVizConfig(
        viz_every=1,
        num_sequences=args.num_sequences,
        plan_files=args.plan_files,
        video_px=args.video_px,
    )
    runner = SequenceVizRunner(agent, config)

    out_dir = Path(args.out_dir)
    videos, scalars = runner.run(args.epoch, out_dir=out_dir)

    print("\n" + "=" * 72)
    print(f"checkpoint : {args.checkpoint}")
    print(f"out dir    : {out_dir}")
    for key, path in videos.items():
        name = key.split("/", 1)[1]
        iou = scalars.get(f"viz/{name}/final_goal_iou", float("nan"))
        reached = scalars.get(f"viz/{name}/reached_exact", 0.0)
        print(f"  {name:44s} final_goal_iou {iou:5.2f}  "
              f"reached_exact {int(reached)}")
    print("=" * 72)

    if hasattr(simulator, "shutdown"):
        simulator.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
