# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 FSQ-C student, round 9: small code and sampled-action imitation.

New configurations also default to ``--student-root-relative-xy True``:
both sparse goals and the encoder's dense future poses describe body shape
and height without a reference floor location. Each target root XY is aligned
to the current root before encoding. Rotation and velocity cues are retained.
Frozen expert observations and legacy saved configurations keep their original
contract. Set the flag False to reproduce the original v9 observation inputs.
This changes the input distribution: train a separate run rather than silently
reinterpreting a checkpoint trained with world-anchored targets.

The text below records the original round-9 rationale; its metric claims were
subsequently qualified in notes/Student_v9_investigation.MD.

**This branches from v6, not from v7/v8.** Rounds 7, 7_1, 8 and 8_1 each patched
the previous round and none beat v6 on the two numbers that track commitment;
the round-7_1 §7c ledger is five runs long and v6 is still the incumbent. So
this run takes the v6 recipe unchanged -- no latent ramp, no teacher-rate loss,
no goal in the trunk, CE over all rows, one shared 2e-5, T 1.0 / top-p 0.9 --
and changes exactly two things, both aimed at the *disease* rather than at the
symptom.

**The disease.** ``model/fsq_full_match`` is 0.03-0.18 across every round, so
82-97 % of deployed refreshes decode a code the encoder would not have issued
at that state -- and the trunk has never received a gradient for such a code,
because the imitation MSE is taken on ``privileged_action`` alone and the
replay path skips the generated streams entirely. Under that exposure gap the
decoder has only two stable strategies: react hard to the code and be wrong
most of the time (v6's 3.75 Hz jerk), or ignore it and lose commitment
(v7/v7_1/v8). Every knob tried so far -- the ramp, the rate loss, the chunk
phase, the goal in the trunk, the encoder learning rate -- moved *along* that
trade. Neither lever below does.

**Lever 1: shrink the code so the prior can actually predict it.**
``--fsq-scalars 4 --fsq-scalars-per-token 4`` gives **one** token over a
625-word vocabulary: 9.3 bits, against v6's 16 scalars packed into 4 tokens
(37 bits). With a single token ``fsq_full_match`` *is* the token accuracy,
which has run 0.42-0.72 in every round -- so the expected exposure gap closes
by roughly 15-20x. 9.3 bits still names any of the corpus's 411 trusted goals
exactly (``Student_v7_improvement_investigation.MD`` §5.1 made this point and
nothing acted on it). The measured token-independence ratio
(``full_match / accuracy**4`` ~ 1.7-2.3, where a genuine 5-mode code would read
~10) says the 4-token factorization was buying almost nothing anyway.

**Lever 2: train the decoder on codes the prior actually produces.**
``--dagger-action-loss-coeff`` adds an imitation MSE on the *sampled* action,
restricted to the rows the prior itself drove. The DAgger block already visits
prior-induced states and already receives expert labels there -- the label was
simply never applied to the deployable action. The model now publishes the
latent the sampled stream decoded, the agent stores it, and replay re-decodes
it through the trunk for one extra forward, so the gradient exists at all.
Restricted to prior-driven rows on purpose: there the sampled action was the
one applied and the expert labelled the resulting state, which is what DAgger
prescribes; applying it everywhere would just be pressure for every code to
decode to one action.

**Everything else is v6**, including the old contact graph -- so v6 is the
clean baseline and the round-7_1 rebuilt graph
(``yoga_contact_graph_student44h_bbdemoted``) stays a separate arm.

**Readouts, in order.** ``model/fsq_full_match`` should jump to roughly the
token accuracy (0.4-0.6, against v6's 0.096) -- that is lever 1 working, and it
is visible within a few hundred epochs. ``supervised/dagger_action_loss``
should appear once DAgger engages at epoch 1000 and then fall. Code gain
``(prior_mse/privileged_mse - 1) / latent_residual_l2`` should hold near v6's
0.33-0.38 rather than collapsing toward the v7/v8 family's 0.07-0.19.
**``model/fsq_code_perplexity`` is the guard for lever 2**: forcing sampled
codes toward the expert action is mode-collapse pressure by construction, so a
slide toward 1.0 means the coefficient is too high.

Train::

    bash data/scripts/run_student_distill_v9.sh
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.supervised.multi_expert import (
    MultiExpertMaskedMimicAgentConfig,
)


def _load_base_module():
    """Load the v6 FSQ experiment by file path -- NOT v7 or v8."""
    base_path = Path(__file__).resolve().parent / "contact_graph_fsq_transformer.py"
    spec = importlib.util.spec_from_file_location(
        "contact_graph_fsq_transformer_base", base_path
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


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Everything v6 takes, plus round-9 loss and student pose-frame settings.

    Lever 1 needs no new argument -- ``--fsq-scalars`` and
    ``--fsq-scalars-per-token`` are v6's own flags; only their defaults move,
    below, so ``--help`` shows what this round runs at.
    """
    base.additional_experiment_arguments(parser)
    parser.add_argument(
        "--dagger-action-loss-coeff", type=float, default=0.1,
        help="Weight of the imitation MSE on the SAMPLED action over "
             "prior-driven rows (lever 2). 0 reproduces every earlier round, "
             "where the deployable action carried no loss at all. Watch "
             "model/fsq_code_perplexity: this term is mode-collapse pressure "
             "by construction, so keep it small.",
    )
    parser.add_argument(
        "--student-root-relative-xy",
        type=lambda value: str(value).lower() not in ("0", "false", "no"),
        default=True,
        help="Remove reference floor XY placement from student sparse goals "
             "and privileged future poses; retain height, rotations and "
             "velocity cues. Frozen expert observations are unchanged. "
             "False reproduces the original v9 observation contract.",
    )
    # Lever 1. v6 ran 16 scalars packed 4-per-token = 4 tokens of vocab 625.
    parser.set_defaults(fsq_scalars=4, fsq_scalars_per_token=4)


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Configure XY-free student targets after the expert copies are built.

    Keep shared reference context and expert_* components unchanged. The flag
    is serialized in each student's observation component, so training and
    inference use the same transform without changing older saved configs.
    """
    cfg = base.env_config(robot_cfg, args)
    relative_xy = bool(getattr(args, "student_root_relative_xy", True))
    for key in ("masked_mimic_target_poses", "mimic_target_poses"):
        cfg.observation_components[key].static_params["root_relative_xy"] = relative_xy
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MultiExpertMaskedMimicAgentConfig:
    """The v6 agent with the two round-9 levers.

    Lever 1 is already expressed through v6's own FSQ arguments, so the only
    thing to add here is lever 2's coefficient.
    """
    cfg = base.agent_config(robot_config, env_config, args)
    cfg.dagger_action_loss_coeff = float(
        getattr(args, "dagger_action_loss_coeff", 0.1)
    )
    return cfg
