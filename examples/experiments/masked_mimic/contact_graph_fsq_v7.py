# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 FSQ-C student, round 7: the Tier-1 changes.

This is :mod:`contact_graph_fsq_transformer` (v6 -- same student44h corpus,
graph, experts, DAgger schedule, chunk length, vocabulary, fixed sampling) with
the five small orthogonal changes of
``notes/Student_v7_improvement_investigation.MD`` §7 Tier 1, in the priority
order §10.8 revised them into after the Tier-0 measurements. v6 is left
untouched on disk so it stays the clean baseline.

**1. Nothing opposed jerk, and the chunk clock is audible (§10.2, §10.3).**
v6's 12 s holds carry 2-8x v5's joint-acceleration RMS -- it *falls over*
holding a commanded stand -- and the excess sits at 3.71/4.11 Hz, the chunk
clock (30/8 = 3.75 Hz), in the greedy stream as well as the sampled one. So the
mechanism is the refresh itself, and two independent things now oppose it:

* ``--latent-ema-alpha`` ramps a newly committed code into the trunk over
  ~1/alpha steps instead of substituting it. It is rollout state, so the
  imitation loss trains *through* the ramp and replay reproduces it exactly.
* ``--action-rate-loss-coeff`` turns on teacher-rate matching in
  ``SupervisedAgent.calculate_extra_loss`` -- the hook that has returned 0.0
  since round 2, while ``action_smoothness`` sat in ``reward_components``
  never being differentiated. Matching the *expert's* increment rather than
  penalising ``||da||^2`` is what keeps a legitimate kick-up free.

**2. The trunk never saw the goal (§5.1).** In v6 the entire goal->action path
is ``vae_latent``: 16 scalars x 5 levels, held 8 steps and nucleus-sampled.
``--goal-conditioned-trunk`` hands the goal to the decoder directly, so the
code carries *which continuation* and the goal carries *where* -- VQ-BeT's
offset head with the offset goal-conditioned. §2 measured that the 6-body goal
already separates the confusable poses by 0.47 m per body, so the
representation was never the problem; the bandwidth to the action was.
**Guard: watch ``model/fsq_code_perplexity``.** If it collapses toward 1 the
code has gone vestigial and the fallback is a low-rank/FiLM injection rather
than a full concatenation.

**3. The token CE trained on stale targets (§5.3).** 86.8 % of CE rows were
hold rows whose label is a code issued up to 7 steps earlier, under a context
that has since moved on -- 87 % of the gradient training a decision the
deployed head never makes. ``--ce-refresh-rows-only`` restricts it to the
states at which the prior is actually asked to choose. (The chunk-phase scalar
§7 pairs with this is deliberately *not* added: with the CE masked to refresh
rows the phase is constant wherever it would be trained, so it could only be
noise. The refresh/hold split logging added in §10.7 stays on either way.)

**4. One learning rate for three very different jobs (§5.4).** The AR prior is
a small classifier whose in-repo counterpart trains at AdamW 1e-4; the encoder
sees the straight-through gradient on ~13 % of rows and now also scaled by the
smoothing alpha. Both get their own parameter group.

**5. Nothing scored commanded-goal *pose* attainment (§5.5).** The failure this
round is about has been caught by eye in rounds 1, 4, 6 and again in the v6
review, and never had a number. ``diag_goal_pose_error`` is that number, live,
at weight 0 -- and deliberately the same arithmetic
``data/scripts/score_probe_pose.py`` does offline, so the training curve and
the probe tables are commensurable.

Inference defaults move to **T 0.7 / top-p 0.8**, which §10.4 priced as a free
win (standing-hold jerk 3.4x lower, pose error 2.9x lower, handstand kept).
Training samples its DAgger block at the same setting, so the states the
student is corrected on are the states it will actually be deployed at.

Train::

    bash data/scripts/run_student_distill_v7.sh

or the underlying command with
``--experiment-path examples/experiments/masked_mimic/contact_graph_fsq_v7.py``.
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
    """Load the v6 FSQ experiment by file path (same pattern v6 uses for v5)."""
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

# The goal, as the trunk sees it. Continuous blocks are normalised into
# trunk-private keys rather than the encoder's, so two containers cannot race
# to define the same normalised tensor within one forward.
GOAL_TRUNK_KEYS = [
    "masked_mimic_target_poses",
    "masked_mimic_target_masks",
    "masked_mimic_target_times",
    "masked_mimic_target_poses_masks",
    "contact_goal_obs",
]
TRUNK_GOAL_POSES_NORM = "trunk_goal_poses_norm"
TRUNK_GOAL_TIMES_NORM = "trunk_goal_times_norm"


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Everything v6 takes, plus the round-7 knobs."""
    base.additional_experiment_arguments(parser)

    def _flag(value) -> bool:
        return str(value).lower() not in ("0", "false", "no")

    parser.add_argument(
        "--goal-conditioned-trunk", type=_flag, default=True,
        help="Feed the goal (pose block, masks, times, contact half) to the "
             "action trunk as well as the encoder/prior. False reproduces v6, "
             "where the only goal->action path is the 16x5 sampled code.",
    )
    parser.add_argument(
        "--latent-ema-alpha", type=float, default=0.5,
        help="Exponential ramp applied to the intent code before it reaches "
             "the trunk (1.0 = v6's hard substitution). Removes the chunk-clock "
             "step change measured at 3.71-4.11 Hz on v6's holds.",
    )
    parser.add_argument(
        "--ce-refresh-rows-only", type=_flag, default=True,
        help="Take the token cross-entropy over refresh rows only -- the "
             "states at which the deployed prior actually chooses a code.",
    )
    parser.add_argument(
        "--action-rate-loss-coeff", type=float, default=0.5,
        help="Weight of the teacher-rate-matching term "
             "||(da_student) - (da_expert)||^2 in calculate_extra_loss. "
             "0 reproduces v6, where nothing opposed jerk at all.",
    )
    parser.add_argument(
        "--ar-head-lr", type=float, default=1e-4,
        help="Learning rate for the AR token prior's own parameter group.",
    )
    parser.add_argument(
        "--ar-head-weight-decay", type=float, default=0.01,
        help="Weight decay for the AR-prior group (AdamW).",
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=5e-5,
        help="Learning rate for the privileged encoder's own parameter group, "
             "offsetting its ~13 %% straight-through duty cycle.",
    )
    # v6's own defaults, restated here so `python ... --help` shows what this
    # round actually runs at rather than what v6 did.
    parser.set_defaults(fsq_temperature=0.7, fsq_top_p=0.8)


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """The v6 environment, plus the commanded-goal pose-error diagnostic.

    Two weight-0 reward components, which is how this repo carries a metric
    that is logged but never optimized (``diag_contact_goal_ground_iou`` is the
    precedent). They are read together: the error is filled with the batch mean
    on rows whose nearest goal reveals no pose, so
    ``raw_r/diag_goal_pose_error`` is the conditional mean over *commanded*
    poses and ``raw_r/diag_goal_pose_visible`` says how much of the batch that
    rests on.
    """
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.obs import (
        compute_contact_goal_pose_error,
        compute_contact_goal_pose_error_visible,
    )

    cfg = base.env_config(robot_cfg, args)
    cfg.reward_components["diag_goal_pose_error"] = MdpComponent(
        compute_func=compute_contact_goal_pose_error,
        dynamic_vars={"pose_error": EnvContext.contact_goal.pose_error},
        static_params={"weight": 0.0},
    )
    cfg.reward_components["diag_goal_pose_visible"] = MdpComponent(
        compute_func=compute_contact_goal_pose_error_visible,
        dynamic_vars={
            "pose_error_visible": EnvContext.contact_goal.pose_error_visible
        },
        static_params={"weight": 0.0},
    )
    return cfg


# Goal blocks that need normalising before the trunk sees them, and the
# trunk-private key each is written to. Everything else in GOAL_TRUNK_KEYS is a
# mask or a binary multi-hot and goes in raw, for the same rare-slot reason
# ``contact_goal_obs`` is never normalised anywhere else.
_TRUNK_GOAL_NORMALIZED = {
    "masked_mimic_target_poses": TRUNK_GOAL_POSES_NORM,
    "masked_mimic_target_times": TRUNK_GOAL_TIMES_NORM,
}


def _trunk_head(trunk_config):
    """The single ``MLPWithConcatConfig`` that produces the action.

    Located by identity: these configs are dataclasses, so ``list.index`` would
    match on structural equality and could pick a different module.
    """
    positions = [
        i
        for i, module in enumerate(trunk_config.models)
        if "actor_trunk_out" in (module.out_keys or [])
    ]
    if len(positions) != 1:
        raise ValueError(
            "expected exactly one trunk module producing 'actor_trunk_out', "
            f"found {len(positions)}"
        )
    return positions[0], trunk_config.models[positions[0]]


def _goal_condition_trunk(trunk_config, keys=None) -> None:
    """Give the action trunk (a subset of) the goal, in place.

    ``keys`` defaults to the whole goal -- v7. Round 7_1 §6.3 runs it with the
    contact half removed: measured over every training frame, the goal's support
    set already equals the current one on 72 % of them and only 13.8 % ask the
    body to unload a limb, so handing the decoder the contact half hands it the
    intent code's one remaining job, while the pose half is what measurably
    fixed the Warrior III substitution (round 7 §10.2).
    """
    from protomotions.agents.common.config import (
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
    )

    keys = list(GOAL_TRUNK_KEYS if keys is None else keys)
    unknown = [key for key in keys if key not in GOAL_TRUNK_KEYS]
    if unknown:
        raise ValueError(f"not goal keys: {unknown}; expected a subset of {GOAL_TRUNK_KEYS}")

    trunk_config.in_keys = list(dict.fromkeys(trunk_config.in_keys + keys))
    normalizers = [
        ObsProcessorConfig(
            in_keys=[source],
            out_keys=[_TRUNK_GOAL_NORMALIZED[source]],
            normalize_obs=True,
            norm_clamp_value=5,
            module_operations=[ModuleOperationForwardConfig()],
        )
        for source in keys
        if source in _TRUNK_GOAL_NORMALIZED
    ]
    index, head = _trunk_head(trunk_config)
    trunk_config.models[index:index] = normalizers
    head.in_keys = list(
        dict.fromkeys(
            head.in_keys
            + [_TRUNK_GOAL_NORMALIZED.get(key, key) for key in keys]
        )
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MultiExpertMaskedMimicAgentConfig:
    """The v6 agent with the Tier-1 changes applied on top.

    Built by mutating v6's configuration rather than restating it, so anything
    v6 changes later is inherited and the diff between the two rounds stays
    exactly the list in this module's docstring.
    """
    from protomotions.agents.base_agent.config import OptimizerConfig

    cfg = base.agent_config(robot_config, env_config, args)
    model = cfg.model

    if bool(getattr(args, "goal_conditioned_trunk", True)):
        _goal_condition_trunk(model.trunk)

    model.fsq.latent_ema_alpha = float(getattr(args, "latent_ema_alpha", 0.5))
    model.fsq.ce_refresh_rows_only = bool(
        getattr(args, "ce_refresh_rows_only", True)
    )

    # AdamW rather than Adam so a per-group weight decay means anything; with
    # decay 0 on the shared group it is numerically the Adam v6 ran.
    model.optimizer = OptimizerConfig(
        _target_="torch.optim.AdamW", lr=2e-5, weight_decay=0.0
    )
    model.ar_head_lr = float(getattr(args, "ar_head_lr", 1e-4))
    model.ar_head_weight_decay = float(getattr(args, "ar_head_weight_decay", 0.01))
    model.encoder_lr = float(getattr(args, "encoder_lr", 5e-5))

    cfg.action_rate_loss_coeff = float(
        getattr(args, "action_rate_loss_coeff", 0.5)
    )
    return cfg
