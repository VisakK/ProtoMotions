# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-graph **edge** expert, v2 — localised tracking + terminal goal state.

v1 (``mlp_edge_small.py``) tracked its clip but failed to hand off: the exit
landed with the legs **crossed** in 100 % of envs and the kick-up slid its left
palm 19.5 cm. Neither failure was visible to the objective, because
``gt_rew = exp(-25 * MSE)`` averages over **24 bodies** -- one body wrong by
0.20 m costs ~4 % of the position reward -- while the termination uses **max**
at 0.5 m, which 0.22 m passes. See ``notes/Skill_graph_handstand_lessons.MD``.

Four changes from v1:

1. **Top-k tracking term** (``topk_gt_rew``, weight 0.3, k=4, coef -15).
   Penalises a few badly-placed bodies, which the mean-over-24 dilutes.
2. **Tail termination** at 0.25 m over the last 30 steps, *in addition to* the
   global 0.5 m bound. Measured first: a *uniform* 0.25 m bound is not viable --
   the trained v1 exit runs at 0.242 m mean / 0.505 m worst max-body error,
   peaking mid-clip, and **100 % of envs** exceed 0.25 m somewhere. The bad
   landing is a terminal-phase problem, so the tighter bound is terminal-phase
   only.
3. **Terminal goal state** (``terminal_goal_rew``, weight 0.1): pose *and*
   velocity against the target node's frozen pose, ramped over the last 30 steps.
   The node reference is at rest, so the velocity half is "arrive stopped".
4. **Cauchy tails on the hand-off reward** instead of Gaussian. Not cosmetic:
   on this edge the Gaussian form measured *identically zero* for the whole run
   (tadasana's V falls steeply; at V = 96.9 the Gaussian is 1.7e-5), so the join
   term was inert. Cauchy scores 0.083 at the same point.

Reward ordering is deliberate and matches the intent that imitation dominates:

    tracking 1.6  >  hand-off 0.5  >  terminal goal 0.1

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_edge_v2.py \
      --experiment-name edge_exit_v2 \
      --motion-file data/smpl/skill_graph_handstand/edge_exit.pt \
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


EDGE_TARGETS = {
    "edge_kickup": dict(node="node_handstand", v_nominal=109.20, delta=1.67, clip_steps=320),
    "edge_exit":   dict(node="node_tadasana",  v_nominal=105.03, delta=2.45, clip_steps=159),
}

W_TRACK_TOPK = 0.3      # on top of the 1.3 of standard tracking terms
W_JOIN = 0.5
W_TERMINAL = 0.1
TAIL_STEPS = 30
TAIL_TERM_THRESHOLD = 0.25


def _edge_key(args) -> str:
    mf = os.path.basename(getattr(args, "motion_file", "") or "")
    for k in EDGE_TARGETS:
        if k in mf:
            return k
    raise ValueError(f"Cannot infer edge from '{mf}'. Expected one of {list(EDGE_TARGETS)}.")


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
    from node_value_join import node_value_join_rew_factory
    from edge_terms import (
        topk_gt_rew_factory, terminal_goal_rew_factory,
        tail_tracking_error_term_factory,
    )

    spec = EDGE_TARGETS[_edge_key(args)]
    node, clip_steps = spec["node"], spec["clip_steps"]

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        # --- tracking (dominant, total weight 1.6) ---
        **mimic_tracking_rewards_factory(
            gt_weight=0.5, gr_weight=0.3, gv_weight=0.1, gav_weight=0.2, rh_weight=0.2,
            gt_coef=-25.0, gr_coef=-5.0, gv_coef=-0.5, gav_coef=-0.1, rh_coef=-100.0,
        ),
        "topk_gt_rew": topk_gt_rew_factory(k=4, coefficient=-15.0, weight=W_TRACK_TOPK),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(weight=0.0, zero_during_grace_period=True),
        # --- hand-off (weight 0.5) ---
        "node_value_join_rew": node_value_join_rew_factory(
            node_checkpoint=f"results/{node}/final.ckpt",
            node_motion_pt=f"data/smpl/skill_graph_handstand/{node}.pt",
            v_nominal=spec["v_nominal"], delta=spec["delta"],
            weight=W_JOIN, clip_steps=clip_steps, tail="cauchy",
        ),
        # --- terminal goal state (lowest, weight 0.1) ---
        "terminal_goal_rew": terminal_goal_rew_factory(
            node_motion_pt=f"data/smpl/skill_graph_handstand/{node}.pt",
            clip_steps=clip_steps, tail_steps=TAIL_STEPS, weight=W_TERMINAL,
        ),
    }

    return EnvConfig(
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
    robot_cfg.reset_noise = None
    simulator_cfg.domain_randomization = None
