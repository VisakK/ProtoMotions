# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 FSQ-C student, round 10_1: make the command mean something.

Branches from v9/v10 -- same corpus family, same experts, same FSQ head, same
DAgger, and both XY-contract flags on. **Only the goal schedule changes.** The
full brief is ``notes/V9_crucial_investigations/V10_1_plan.MD``.

**The disease.** v9's student reaches a commanded pose and then does the
corpus's most likely *next* thing instead of the commanded thing. Measured
cause: the commanded goal is a near-deterministic function of the current pose
on the training distribution -- conditioning on the pose removes **87-92 %** of
the goal's spread within a clip (0.705 m marginal -> 0.056-0.089 m) -- so the
imitation loss is never forced to make the goal a *cause*. The policy can
satisfy it by learning the clip-continuation map and ignoring the command, and
at deployment, where the command deliberately is *not* the continuation, that is
exactly what it does.

Three schedule changes attack that, all of them label-safe: the expert never
changes, and every goal remains a pose the playing clip actually reaches.

**1. ``--dwell-channels``.** Append ``[hold_duration, dwell_remaining]`` to each
goal slot. The command has never carried how long to stay. The *deadline* says
when to *be* somewhere, and Tier-0 §5 showed raising it makes holds
monotonically **worse** (4.20 -> 1.23 -> 1.03 -> 0.87 s) -- it is the wrong
channel for the job. Duration is the right one and it is informative: p10
0.60 s, median 1.70 s, p90 9.07 s across the corpus.

**2. ``--include-current-segment``.** Slot 0 becomes the segment the clip is
*inside*, not the next hold. Today **42.1 % of all trusted dwell** is spent
inside a segment while already commanded to leave it, which is where
"anticipatory departure at a commanded pose" is learned. Paired with (1) because
"stay for X more seconds" is meaningless for a goal you have not reached, and
without this slot 0 is always a goal you have not reached.

**3. ``--far-goal-prob``.** With some probability an episode's forward window
starts a few holds later. Nothing in training has ever placed a distant goal in
the nearest slot -- ``require_first_goal_specified`` guarantees slot 0 is the
imminent hold -- so a policy asked for one at inference is off-distribution.
The far-handstand probe's handstand goal has reach rate **0.028**.

Every flag defaults **off**, so this file with no extra arguments reproduces v9
exactly. That is deliberate: it is what makes each change ablatable, and the
tests assert the off-path is bit-identical.

**Not in this run, on purpose:** ``rare_node`` segment sampling (held back as the
v10_2 single-variable ablation), and any change to the FSQ head, the code size,
the trunk's inputs or the graph identity rule -- see the plan's §9 for the
measurement that rules each of those out.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


def _load_base_module():
    """Load the v9 experiment as this file's base, by path.

    The experiment files are loaded by path rather than imported as a package,
    so a plain ``import`` would not resolve; v9 does the same against v6.
    """
    path = Path(__file__).with_name("contact_graph_fsq_v9.py")
    spec = importlib.util.spec_from_file_location(
        "contact_graph_fsq_v9_base", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

NUM_GOAL_STEPS = base.NUM_GOAL_STEPS
STATE_KEYS = base.STATE_KEYS
UNNORMALIZED_STATE_KEYS = base.UNNORMALIZED_STATE_KEYS
CONTACT_EVENT_FLAG_KEY = base.CONTACT_EVENT_FLAG_KEY
ENCODER_FUTURE_STEPS = base.ENCODER_FUTURE_STEPS


def _flag(value) -> bool:
    return str(value).lower() not in ("0", "false", "no")


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Everything v9 takes, plus the three goal-schedule levers."""
    base.additional_experiment_arguments(parser)
    parser.add_argument(
        "--dwell-channels", type=_flag, default=False,
        help="Append [hold_duration, dwell_remaining] to every goal slot of "
             "contact_goal_obs, scaled to [0,1]. False keeps the observation "
             "block byte-identical to v9. Pairs with "
             "--include-current-segment.",
    )
    parser.add_argument(
        "--include-current-segment", type=_flag, default=False,
        help="Goal slot 0 becomes the segment the clip is currently inside "
             "rather than the next hold, pushing the forward window into slots "
             "1..K-1. A no-op until the current segment's own hold has been "
             "passed -- which is the 42.1 %% of dwell that today teaches "
             "anticipatory departure.",
    )
    parser.add_argument(
        "--far-goal-prob", type=float, default=0.0,
        help="Probability that an episode's forward goal window starts some "
             "holds later than usual (far-goal promotion). Drawn once per "
             "episode. 0 disables.",
    )
    parser.add_argument(
        "--far-goal-max-skip", type=int, default=3,
        help="Upper bound of the uniform skip when promotion fires. No "
             "measurement backs this default; it is a first guess.",
    )


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides
agent_config = base.agent_config


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """v9's environment plus the goal-schedule settings on the control component.

    The settings are written onto the control component's config so they are
    serialized into ``resolved_configs.yaml`` with the run -- an inference build
    of this checkpoint then reproduces the same schedule semantics without
    needing the flags again, exactly as v9's XY flag does.
    """
    cfg = base.env_config(robot_cfg, args)
    control = cfg.control_components["contact_graph"]
    control.dwell_channels = _flag(getattr(args, "dwell_channels", False))
    control.include_current_segment = _flag(
        getattr(args, "include_current_segment", False)
    )
    control.far_goal_prob = float(getattr(args, "far_goal_prob", 0.0) or 0.0)
    control.far_goal_max_skip = int(getattr(args, "far_goal_max_skip", 3))
    return cfg
