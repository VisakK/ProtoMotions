# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 FSQ-C student, round 11: put a multi-step target in the loss.

Branches from **v9**, which is still the incumbent -- rounds 7, 7_1, 8, 8_1 and
10_1 each patched the previous round and none beat it. Corpus, graph, experts,
code size, chunk clock, CE, DAgger mixture, temperature and the goal-free trunk
are all v9's, unchanged. **Exactly one thing changes: the shape of the target.**

**Why.** Ten rounds have edited the goal *channel* -- code size, KL,
predictability, masking, the schedule, the deadline, the dwell, the graph. The
measurement that says why none of it could work is about the *label*:

* Fitting a 214-dim linear map (heading-normalised pelvis-relative body
  positions and velocities plus the current DOFs) to the reference DOF one step
  ahead -- which ``normalized_pd_fixed_gains_action`` makes the expert action a
  fixed affine map of -- gives **R^2 = 0.999797** over 75,019 corpus rows. At
  h = 0 there is **2.03e-4** of variance left for *any* extra channel to
  explain. That is the causal root of the measured "the code's total action
  authority is 1.1 % of action variance": the target itself forbids more.
* The same map leaves 0.016665 at h = 8 (0.27 s), 0.051720 at h = 16 (0.53 s)
  and 0.090844 at h = 24 (0.80 s) -- **82x, 255x and 448x** the h = 0 residual.
* Controlled at a *bit-identical* state, the different-goal-minus-same-goal
  label excess is **+0.0023 rad at one control step (95 % CI [-0.0004,
  +0.0051] -- it contains zero)**, +0.0174 at 8 steps and +0.0633 at 24.

So at the horizon the imitation loss is evaluated on, the commanded goal
changes the correct action by an amount statistically indistinguishable from
zero. No bottleneck, code size, KL or prior surgery can matter there.

**The change.** The trunk's final layer emits ``len(ladder_offsets) *
number_of_actions`` instead of ``number_of_actions``. Rung 0 is what the
simulator executes and populates ``action`` / ``mean_action`` /
``privileged_action`` byte-identically to v9; the later rungs are pure
auxiliary targets against ``expert_actions`` that many control steps ahead,
each normalised by its own target variance. The executed action stays a
per-step closed-loop decode, so none of the open-loop-chunk stability risk a
naive action-chunking port would carry into a 30 Hz balancing humanoid.

**Why A1's rungs sit at {0, 5, 10, 15} rather than {0, 8, 16, 24}.** Those are
the encoder's *existing* future offsets, so this arm changes the target and
nothing else: no observation width moves, and the six-offset window that tipped
the 24 GB card in v5 is never approached. It is strictly single-variable. The
top rung at 0.5 s already carries ~13 % of DOF sd of goal-attributable content.
``--ladder-offsets 0 8 16 24`` with ``--encoder-future-steps 1 8 16 24`` is the
A2 follow-up, and it is contingent on A1 passing its mechanism gate.

**The pre-registered mechanism metric** is ``model/code_ablation_gap_h15``: the
rise in rung 15's normalised MSE when ``vae_latent`` is rolled across the
batch. PASS is >= 0.015 by epoch 4,000; NULL is <= 0.005. The matched control
is ``model/code_ablation_gap_h0``, whose ceiling is 2.03e-4 -- a large far-rung
gap against a near-zero rung-0 gap is the signature that the ladder engaged and
nothing else did. ``masked_mimic/mse`` must stay within 1.5x of v9's power law
(log-log slope -0.655) or the ladder is stealing rung-0 capacity.

Pre-registered as *allowed to get worse*: ``model/fsq_full_match`` and
``model/fsq_ce_loss_refresh``. The code now carries more state-unexplained
information, so the prior may predict it less well -- and code predictability
has been measured not to be the frontier (v5 drove the deployable code to the
teacher's exactly and the frontier did not move).

``--ladder-loss-coeff 0`` reproduces v9 exactly: the trunk goes back to one
action head and every ladder branch is a no-op.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


def _load_base_module():
    """Load v9 as this file's base, by path (the experiments are not a package)."""
    path = Path(__file__).with_name("contact_graph_fsq_v9.py")
    spec = importlib.util.spec_from_file_location("contact_graph_fsq_v9_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

NUM_GOAL_STEPS = base.NUM_GOAL_STEPS
STATE_KEYS = base.STATE_KEYS
UNNORMALIZED_STATE_KEYS = base.UNNORMALIZED_STATE_KEYS
CONTACT_EVENT_FLAG_KEY = base.CONTACT_EVENT_FLAG_KEY
ENCODER_FUTURE_STEPS = base.ENCODER_FUTURE_STEPS

# A1's rungs. These are exactly the encoder's default future offsets, which is
# what makes the arm single-variable.
DEFAULT_LADDER_OFFSETS = (0, 5, 10, 15)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Everything v9 takes, plus the ladder."""
    base.additional_experiment_arguments(parser)
    parser.add_argument(
        "--ladder-offsets",
        type=int,
        nargs="+",
        default=list(DEFAULT_LADDER_OFFSETS),
        help="Action-ladder rungs in control steps ahead; must start at 0 and "
             "increase. '0' alone reproduces v9's single action head. A1 runs "
             "0 5 10 15 (the encoder's own offsets, so nothing else moves); "
             "A2 runs 0 8 16 24 alongside --encoder-future-steps 1 8 16 24.",
    )
    parser.add_argument(
        "--ladder-balance",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=False,
        help="Interpret --ladder-loss-coeff as a MULTIPLE OF THE IMITATION "
             "LOSS rather than an absolute weight. A1 measured why this "
             "matters: the variance-normalised ladder ran at 32.9x the "
             "imitation term, 97 % of the trunk's gradient. False reproduces "
             "A1 exactly.",
    )
    parser.add_argument(
        "--ladder-loss-coeff",
        type=float,
        default=1.0,
        help="Weight of the auxiliary far-rung MSE. Each rung is normalised by "
             "its own target variance before the mean, so this is a single "
             "scale against the imitation term. 0 disables the ladder "
             "entirely and reproduces v9.",
    )


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides
env_config = base.env_config


def _resolve_offsets(args: argparse.Namespace) -> tuple:
    offsets = tuple(
        int(o) for o in getattr(args, "ladder_offsets", DEFAULT_LADDER_OFFSETS)
    )
    if not offsets or offsets[0] != 0:
        raise ValueError(
            f"--ladder-offsets must start with 0 (the executed action), got {offsets}"
        )
    if sorted(set(offsets)) != list(offsets):
        raise ValueError(
            f"--ladder-offsets must be strictly increasing, got {offsets}"
        )
    if float(getattr(args, "ladder_loss_coeff", 1.0) or 0.0) <= 0.0:
        # A zero coefficient and several rungs would widen the trunk and train
        # the extra outputs on nothing. Collapse to v9 rather than waste them.
        return (0,)
    return offsets


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    """v9's agent with the trunk widened to the ladder and the term switched on.

    The trunk's ``num_out`` and ``model.fsq.ladder_offsets`` are set from one
    resolved tuple, so the decode split and the auxiliary targets cannot
    disagree with the head width.
    """
    cfg = base.agent_config(robot_config, env_config, args)
    offsets = _resolve_offsets(args)
    cfg.model.fsq.ladder_offsets = offsets
    if len(offsets) == 1:
        cfg.ladder_loss_coeff = 0.0
        return cfg

    cfg.ladder_loss_coeff = float(getattr(args, "ladder_loss_coeff", 1.0))
    cfg.ladder_balance_to_imitation = bool(getattr(args, "ladder_balance", False))
    num_actions = robot_config.number_of_actions
    widened = False
    for model_cfg in cfg.model.trunk.models:
        # The action head is the one module that emits exactly the action width;
        # everything before it is a normaliser or an obs processor.
        if getattr(model_cfg, "num_out", None) == num_actions:
            model_cfg.num_out = len(offsets) * num_actions
            widened = True
    if not widened:
        raise ValueError(
            "no trunk module emits number_of_actions "
            f"({num_actions}); the ladder cannot find the action head"
        )
    return cfg
