# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-graph **edge** expert -- value-joined to its target node.

An edge is a transition between two nodes. Unlike a node it tracks a real
trajectory, so it keeps the standard mimic objective. What it adds is a
**hand-off** term: the edge is paid for finishing in a state the *target node's
own critic* recognises.

Why that term is needed, from `notes/Skill_graph_nodes.MD` §6.3: for both edges,
the source node's release window and the target node's capture window are
disjoint by a wide margin (8.3 s of the 10.7 s kick-up, 3.0 s of the 5.3 s exit).
The edge is the only thing that can cover that interval, and tracking error alone
does not express "end somewhere the next skill can catch you" -- at t = 2.0 s of
the kick-up the character passes through a near-perfect down-dog *pose*
(0.141 m error) that the down-dog node still drops in 4 steps, because the limb
is mid-swing.

The band form ``exp(-((V - V_nom)/delta)^2)`` is used rather than a one-sided
threshold because §6.2 measured V-thresholding to be *anti*-predictive on two of
three nodes (AUC 0.298 / 0.235) while ``-|V - V_nom|`` is consistently
informative (0.887 / 0.923 / 0.930). Constants come from ``value_band.json``.

Differences from ``mlp_node_small.py`` (same 2x2048 trunk, same 1027-wide
observation, same disturbances):

* the reference is a **real trajectory**, not a frozen frame;
* ``init_start_prob = 1.0`` -- every episode replays the edge from its start, so
  ``progress_buf`` indexes the clip directly and the terminal hand-off window is
  well defined. (Trade-off: less mid-trajectory state diversity than RSI would
  give; the reset noise and pushes still supply variation.)
* a ``node_value_join_rew`` term, ramped quadratically over the clip so almost
  all of its weight lands at the hand-off while still giving a gradient earlier.

**The reward is not the evaluation.** The certificate being optimised is
imperfect (best band accuracy 0.82 on the handstand node), and optimising an
imperfect certificate invites gaming it. Judge a trained edge with
``data/scripts/handoff_check.py``, which measures the hand-off empirically.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_edge_small.py \
      --experiment-name edge_kickup_join \
      --motion-file data/smpl/skill_graph_handstand/edge_kickup.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb
"""
import argparse
import os
import sys

# This directory must be importable before any hook runs: both
# configure_robot_and_simulator (first hook called) and env_config pull siblings
# from it (mlp_node_small, node_value_join).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig


# --------------------------------------------------------------------------- #
# Edge -> target-node wiring. V_nominal and delta are read off
# results/<node>/value_band.json (velocity family): V_nominal is the scale-0
# median, delta the half-width of the best two-sided band.
#
# gate_steps is the progress_buf index after which the hand-off term pays, i.e.
# clip_length_steps - 30 (the last ~1 s at 30 Hz control).
# --------------------------------------------------------------------------- #
EDGE_TARGETS = {
    "edge_kickup": dict(
        node="node_handstand",
        v_nominal=109.20,
        delta=1.67,          # band [107.49, 110.82]
        clip_steps=320,      # 10.667 s @ 30 Hz
    ),
    "edge_exit": dict(
        node="node_tadasana",
        v_nominal=105.03,
        delta=2.45,          # band [101.69, 106.59]
        clip_steps=159,      # 5.30 s @ 30 Hz
    ),
}
JOIN_WEIGHT = float(os.environ.get("EDGE_JOIN_WEIGHT", "0.5"))


def _edge_key(args) -> str:
    """Which edge is being trained, inferred from the motion file name."""
    mf = os.path.basename(getattr(args, "motion_file", "") or "")
    for k in EDGE_TARGETS:
        if k in mf:
            return k
    raise ValueError(
        f"Cannot infer edge from motion file '{mf}'. Expected one of {list(EDGE_TARGETS)}."
    )


def terrain_config(args: argparse.Namespace):
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        action_smoothness_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        contact_match_rew_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    from node_value_join import node_value_join_rew_factory

    edge = _edge_key(args)
    spec = EDGE_TARGETS[edge]
    node = spec["node"]

    control_components = {"mimic": MimicControlConfig(bootstrap_on_episode_end=True)}

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(observe_contacts=True),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
    }

    termination_components = {"tracking_error": tracking_error_term_factory(threshold=0.5)}

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5, gr_weight=0.3, gv_weight=0.1, gav_weight=0.2, rh_weight=0.2,
            gt_coef=-25.0, gr_coef=-5.0, gv_coef=-0.5, gav_coef=-0.1, rh_coef=-100.0,
        ),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(
            weight=0.0, zero_during_grace_period=True
        ),
        # The hand-off term. Logged as raw_r/node_value_join_rew even at weight 0,
        # so a weight-0 run is a clean baseline that still reports what the target
        # node's critic thinks of the edge's terminal states.
        "node_value_join_rew": node_value_join_rew_factory(
            node_checkpoint=f"results/{node}/final.ckpt",
            node_motion_pt=f"data/smpl/skill_graph_handstand/{node}.pt",
            v_nominal=spec["v_nominal"],
            delta=spec["delta"],
            weight=JOIN_WEIGHT,
            clip_steps=spec["clip_steps"],
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        ref_respawn_offset=0.005,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=1.0,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig, PPOModelConfig, AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig, MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory, gr_error_factory, max_joint_error_factory,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
            normalize_obs=True, norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(2)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        out_keys=["value"], normalize_obs=True, norm_clamp_value=5, num_out=1,
        layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(2)],
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config, critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        # Dense checkpoints: see notes 6.1 -- a near-perfect policy can blow the
        # PPO trust region and never recover, and score_based.ckpt is no help.
        save_epoch_checkpoint_every=250,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Same contact sensing and disturbance channels as the node experts."""
    from protomotions.simulator.base_simulator.config import (
        DomainRandomizationConfig, PushDomainRandomizationConfig, RobotNoiseConfig,
    )
    from mlp_node_small import (
        RESET_NOISE, PUSH_INTERVAL_RANGE,
        PUSH_MAX_LINEAR_VELOCITY, PUSH_MAX_ANGULAR_VELOCITY,
    )

    robot_cfg.update_fields(
        contact_bodies="all",
        reset_noise=RobotNoiseConfig(**RESET_NOISE),
    )
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=PUSH_INTERVAL_RANGE,
            max_linear_velocity=PUSH_MAX_LINEAR_VELOCITY,
            max_angular_velocity=PUSH_MAX_ANGULAR_VELOCITY,
        )
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, env_cfg, agent_cfg,
    terrain_cfg: TerrainConfig, motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig, args: argparse.Namespace,
):
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
    robot_cfg.reset_noise = None
    simulator_cfg.domain_randomization = None
