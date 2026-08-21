# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How often does each slot of ``contact_state_obs`` actually fire?

The channel can be plumbed correctly and still be useless: if the body-body half
never crosses its threshold in practice, the student is handed 89 constant zeros
and the goal's body-body half stays unanswerable in effect if not in principle.
That is the same trap `Singleleg14_pressure_experiment.MD` §3.2 records for the
zone-share reward, where a target that looked healthy was a constant.

So this rolls a checkpoint out over its own corpus and reports, per contact pair,
how often the measured state has it and how often the *goal* asks for it. A
body-body slot that the goal requests and the state never reports is the failure
to look for.

Usage::

    DISPLAY= PYTHONPATH=. python data/scripts/probe_contact_state_obs.py \\
      --checkpoint results/smoke_pairs_on/last.ckpt --steps 400
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--top", type=int, default=25, help="rows to print")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

if args.num_envs is None:
    args.num_envs = 256
args.headless = True

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


def log(fmt, *fmt_args) -> None:
    print("probe_contact_state_obs: " + (fmt % fmt_args if fmt_args else fmt),
          flush=True)


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]

    control = None
    for component in env.control_manager.components.values():
        if hasattr(component, "graph"):
            control = component
            break
    if control is None:
        raise SystemExit("this checkpoint has no contact-graph control component")
    pair_names = list(control.graph.pair_names)

    obs = env.get_obs()
    if "contact_state_obs" not in obs:
        raise SystemExit(
            "this checkpoint's env does not compute contact_state_obs; it "
            f"produces {sorted(obs)}"
        )

    agent.eval()
    agent.pre_collect_step(0)
    obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    state_hits = torch.zeros(len(pair_names), device=env.device)
    goal_hits = torch.zeros(len(pair_names), device=env.device)
    both_hits = torch.zeros(len(pair_names), device=env.device)
    samples = 0

    for _ in range(args.steps):
        with torch.no_grad():
            forward = getattr(agent.model, "forward_inference", agent.model)
            outputs = forward(obs_td)
        action = outputs.get("mean_action", outputs.get("action"))
        obs, _, _, _, _ = env.step(action)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        state = obs["contact_state_obs"] > 0.5
        # The nearest goal's contact set, unmasked by visibility: the question is
        # what the graph asks for, not what this step happened to reveal.
        goal = control._gathered["contact"][:, 0] > 0.5
        goal = goal & control.goal_valid[:, 0:1]
        state_hits += state.float().sum(dim=0)
        goal_hits += goal.float().sum(dim=0)
        both_hits += (state & goal).float().sum(dim=0)
        samples += state.shape[0]

    rows = []
    for index, name in enumerate(pair_names):
        rows.append(
            (
                name,
                float(state_hits[index]) / samples,
                float(goal_hits[index]) / samples,
                float(both_hits[index]) / max(float(goal_hits[index]), 1.0),
            )
        )

    ground = [r for r in rows if r[0].endswith(":G")]
    body = [r for r in rows if not r[0].endswith(":G")]
    log("%d env-steps over %d pairs", samples, len(pair_names))
    for title, subset in (("GROUND", ground), ("BODY-BODY", body)):
        active = [r for r in subset if r[1] > 0 or r[2] > 0]
        log(
            "%s: %d/%d pairs ever active in state, %d ever requested by a goal",
            title,
            sum(1 for r in subset if r[1] > 0),
            len(subset),
            sum(1 for r in subset if r[2] > 0),
        )
        print(f"    {'pair':34s} {'state%':>8s} {'goal%':>8s} {'recall':>8s}")
        for name, s, g, recall in sorted(active, key=lambda r: -r[2])[: args.top]:
            print(f"    {name:34s} {s:8.1%} {g:8.1%} {recall:8.1%}")

    requested_never_seen = [
        r[0] for r in body if r[2] > 0.001 and r[1] == 0.0
    ]
    if requested_never_seen:
        log(
            "WARNING: %d body-body pairs are requested by goals and NEVER "
            "reported by the state: %s",
            len(requested_never_seen),
            requested_never_seen[:8],
        )
        return 1
    log("every body-body pair a goal asks for is one the state can report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
