# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-graph **edge** expert, v3 — velocity-aware tracking + realistic starts.

Decided 2026-07-29 (``notes/brainstorm_skill_graph.MD`` §4.2). Three changes
from v2 (``mlp_edge_v2.py``), two removals and the reasons they are removals:

1. **``node_value_join_rew`` REMOVED.** The V-band hand-off term was measured
   inert on the exit (raw 9.3e-5 even with Cauchy tails: edge terminal states
   sit ~59 deltas / ~130 V-units outside the band's calibration range, and no
   tail reshaping fixes extrapolation). V carries no information about hand-off
   success (V ≈ −40 at the switch in both the 0.000- and the 1.000-composition
   case). Nothing learned replaces it this round.

2. **``terminal_goal_rew`` REMOVED.** No end-pose / end-velocity shaping: this
   round tests whether *velocity-profile imitation* plus *realistic start
   states* suffice, without steering the terminal state directly.

3. **``topk_gv_rew`` ADDED** (k=4, coef −2, weight 0.3) — top-k *velocity*
   tracking, the velocity twin of the ``topk_gt_rew`` that fixed the
   crossed-legs landing. Aimed at the 1.83x limb-speed failure that composition
   1.000 could not see.

4. **Bank-based starts** (``bank_reset.BankResetEnv``): 50 % of episodes start
   from the SOURCE node's actual hold-state distribution (collected by
   ``data/scripts/collect_state_bank.py --mode node-hold``) instead of the
   reference's frame 0; the usual reset noise applies on top of both. In
   composed operation the edge always starts from a node state — this puts that
   distribution in training. The reference clock still starts at 0, so
   ``progress_buf``-indexed terms remain valid. Sources: kickup ← downdog3
   (epoch_1000, the last healthy checkpoint), exit ← handstand.

Kept from v2: standard tracking terms (1.6 total), ``topk_gt_rew`` (0.3), the
global 0.5 m termination AND the tail termination (0.25 m over the last 30
steps — a termination, not a reward), ``action_smoothness`` −0.02 (unchanged to
keep the diff minimal), same net, same optimiser, same disturbances.

Evaluation is pre-registered in ``data/scripts/criteria/edge_*_v3.json`` and
gated by ``data/scripts/eval_gate.py``. Composition rate alone is never again a
success claim.

Train (banks must exist first — see ``data/scripts/run_edge_v3.sh``)::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_edge_v3.py \
      --experiment-name edge_kickup_v3 \
      --motion-file data/smpl/skill_graph_handstand/edge_kickup.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig


BANK_DIR = "data/smpl/skill_graph_handstand"
EDGE_SPECS = {
    "edge_kickup": dict(
        source_node="node_downdog3",
        target_node="node_handstand",
        clip_steps=320,
        bank_file=f"{BANK_DIR}/bank_node_downdog3_hold.pt",
    ),
    "edge_exit": dict(
        source_node="node_handstand",
        target_node="node_tadasana",
        clip_steps=159,
        bank_file=f"{BANK_DIR}/bank_node_handstand_hold.pt",
    ),
}

W_TRACK_TOPK = 0.3
W_TRACK_TOPK_VEL = 0.3
TAIL_STEPS = 30
TAIL_TERM_THRESHOLD = 0.25
BANK_PROB = float(os.environ.get("EDGE_BANK_PROB", "0.5"))


def _edge_key(args) -> str:
    mf = os.path.basename(getattr(args, "motion_file", "") or "")
    for k in EDGE_SPECS:
        if k in mf:
            return k
    raise ValueError(f"Cannot infer edge from '{mf}'. Expected one of {list(EDGE_SPECS)}.")


def terrain_config(args):
    return TerrainConfig()


def scene_lib_config(args):
    return SceneLibConfig(scene_file=getattr(args, "scenes_file", None))


def motion_lib_config(args):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory, previous_actions_factory,
        mimic_target_poses_max_coords_factory, action_smoothness_factory,
        mimic_tracking_rewards_factory, pow_rew_factory,
        contact_match_rew_factory, tracking_error_term_factory,
    )
    from protomotions.envs.action import make_pd_action_config
    from bank_reset import BankResetEnvConfig
    from edge_terms import (
        topk_gt_rew_factory, topk_gv_rew_factory, tail_tracking_error_term_factory,
    )

    spec = EDGE_SPECS[_edge_key(args)]
    clip_steps = spec["clip_steps"]
    bank_file = spec["bank_file"]
    if BANK_PROB > 0 and not os.path.exists(bank_file):
        raise FileNotFoundError(
            f"state bank {bank_file} not found -- collect it first:\n"
            f"  python data/scripts/collect_state_bank.py --mode node-hold "
            f"--checkpoint results/{spec['source_node']}/"
            f"{'epoch_1000.ckpt' if spec['source_node'] == 'node_downdog3' else 'final.ckpt'} "
            f"--out {bank_file}"
        )

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        # --- tracking (dominant; 1.6 standard + 0.6 localised) ---
        **mimic_tracking_rewards_factory(
            gt_weight=0.5, gr_weight=0.3, gv_weight=0.1, gav_weight=0.2, rh_weight=0.2,
            gt_coef=-25.0, gr_coef=-5.0, gv_coef=-0.5, gav_coef=-0.1, rh_coef=-100.0,
        ),
        "topk_gt_rew": topk_gt_rew_factory(k=4, coefficient=-15.0, weight=W_TRACK_TOPK),
        "topk_gv_rew": topk_gv_rew_factory(k=4, coefficient=-2.0, weight=W_TRACK_TOPK_VEL),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(weight=0.0, zero_during_grace_period=True),
        # v2's node_value_join_rew and terminal_goal_rew are deliberately absent.
    }

    return BankResetEnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        ref_respawn_offset=0.005,
        control_components={"mimic": MimicControlConfig(bootstrap_on_episode_end=True)},
        observation_components={
            "max_coords_obs": max_coords_obs_factory(observe_contacts=True),
            "previous_actions": previous_actions_factory(history_steps=1),
            "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
        },
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.5),
            "tail_tracking_error": tail_tracking_error_term_factory(
                threshold=TAIL_TERM_THRESHOLD, clip_steps=clip_steps, tail_steps=TAIL_STEPS,
            ),
        },
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(init_start_prob=1.0, resample_on_reset=True),
        bank_file=bank_file,
        bank_prob=BANK_PROB,
    )


def agent_config(robot_config: RobotConfig, env_config: EnvConfig,
                 args: argparse.Namespace) -> PPOAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig, PPOModelConfig, AdvantageNormalizationConfig)
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig, MotionWeightsRulesConfig)
    from protomotions.envs.component_factories import (
        gt_error_factory, gr_error_factory, max_joint_error_factory)

    keys = ["max_coords_obs", "mimic_target_poses", "previous_actions"]
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs, actor_logstd=-2.9,
        in_keys=keys, mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=keys, normalize_obs=True, norm_clamp_value=5,
            out_keys=["actor_trunk_out"], num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(2)]),
    )
    critic_config = MLPWithConcatConfig(
        in_keys=keys, out_keys=["value"], normalize_obs=True, norm_clamp_value=5,
        num_out=1, layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(2)])

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=keys, out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config, critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4)),
        batch_size=args.batch_size, training_max_steps=args.training_max_steps,
        save_epoch_checkpoint_every=250, gradient_clip_val=50.0, clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory()},
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0)),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True),
    )


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    from protomotions.simulator.base_simulator.config import (
        DomainRandomizationConfig, PushDomainRandomizationConfig, RobotNoiseConfig)
    from mlp_node_small import (
        RESET_NOISE, PUSH_INTERVAL_RANGE,
        PUSH_MAX_LINEAR_VELOCITY, PUSH_MAX_ANGULAR_VELOCITY)

    robot_cfg.update_fields(contact_bodies="all", reset_noise=RobotNoiseConfig(**RESET_NOISE))
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=PUSH_INTERVAL_RANGE,
            max_linear_velocity=PUSH_MAX_LINEAR_VELOCITY,
            max_angular_velocity=PUSH_MAX_ANGULAR_VELOCITY))


def apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg,
                              terrain_cfg, motion_lib_cfg, scene_lib_cfg, args):
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
    if hasattr(env_cfg, "bank_prob"):
        env_cfg.bank_prob = 0.0
    robot_cfg.reset_noise = None
    simulator_cfg.domain_randomization = None
