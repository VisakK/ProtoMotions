# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-graph **node** expert -- small MLP, frozen pose, disturbance training
==========================================================================

A *node* is a quasi-static yoga pose the character must hold **indefinitely**
and recover into after a small shove.  Its reference is a single frozen frame
(built by ``data/scripts/build_skill_graph_clips.py``), so this config trains a
stabilising controller for a fixed point rather than a trajectory tracker.

Derived from ``mlp_2x_no_contact_rew.py`` -- the current best `smpl_yogi`
balance tracker (``notes/Contact_obs_baseline.MD``).  **The observation space,
reward terms and termination are byte-for-byte the same** so a node expert is
directly comparable to that baseline.  Four deliberate changes:

1. **Trunk 6x2048 / 4x2048 -> 2x2048 for both actor and critic**
   (37.85M -> ~12.7M params).  One frozen pose does not need a 38M-parameter
   trunk, and a small net trains several times faster per epoch --
   which is the point of a skill graph: many cheap specialists rather than one
   expensive generalist.

2. **Disturbance training.**  Two independent channels, both small:

   * ``domain_randomization.push`` -- a random root velocity impulse every
     1.5-4 s (uniform in +/-0.30 m/s linear, +/-0.30 rad/s angular).  This is
     the *recovery* signal: the policy is shoved mid-hold and has to come back.
   * ``robot_cfg.reset_noise`` -- RSI noise on the initial state (DOF pos/vel,
     root pos/rot/vel).  This is the *basin-widening* signal: the policy sees
     initial conditions off the nominal pose, which is exactly the distribution
     the stability-region analysis later samples from (at larger magnitudes).

   Vertical root noise is kept at 3 mm on purpose.  The node references are
   grounded to +0.5 cm and ``ref_respawn_offset`` adds another +0.5 cm, so
   +/-3 mm can never spawn a support geom through the floor.

3. **``init_start_prob`` 0.2 -> 1.0** and ``ref_respawn_offset`` 0.05 -> 0.005
   baked in (Fix A, ``notes/Addressing_grounding_PD_stiffness_issues.MD``).
   The reference is constant, so a random start time buys no variety -- it only
   shortens the episode (the clip ends and the env resets).  Starting every
   episode at t=0 on a 40 s clip means every episode runs the full
   ``max_episode_length`` = 1000 steps = 33.3 s as one continuous hold.

4. ``save_epoch_checkpoint_every`` 2000 -> 1000 (checkpoints are ~150 MB here,
   not 454 MB).

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_node_small.py \
      --experiment-name node_handstand \
      --motion-file data/smpl/skill_graph_handstand/node_handstand.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


# --------------------------------------------------------------------------- #
# Disturbance magnitudes. Kept here as module constants so the stability-region
# analysis can import the exact training values and sweep *above* them.
# --------------------------------------------------------------------------- #
PUSH_INTERVAL_RANGE = (1.5, 4.0)          # seconds between shoves
PUSH_MAX_LINEAR_VELOCITY = (0.30, 0.30, 0.15)   # m/s, uniform in +/-
PUSH_MAX_ANGULAR_VELOCITY = (0.30, 0.30, 0.30)  # rad/s, uniform in +/-

RESET_NOISE = dict(
    dof_pos_noise=0.02,                  # rad  (~1.1 deg per DOF)
    dof_vel_noise=0.15,                  # rad/s
    root_pos_noise=[0.010, 0.010, 0.003],  # m -- vertical kept tiny, see docstring
    root_rot_noise=[0.015, 0.015, 0.015],  # rad (~0.86 deg)
    root_vel_noise=0.05,                 # m/s
    root_ang_vel_noise=0.15,             # rad/s
)


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults)."""
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

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    # Identical to mlp_2x_no_contact_rew.py -- obs width 1027 (contacts on all
    # 24 bodies). Keeping this fixed is what makes a node expert comparable to
    # the multi-pose baseline.
    observation_components = {
        "max_coords_obs": max_coords_obs_factory(observe_contacts=True),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
    }

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        # Weight 0.0: kept only so raw_r/contact_match_rew stays visible in
        # wandb as a diagnostic. Contributes nothing to the objective.
        "contact_match_rew": contact_match_rew_factory(
            weight=0.0, zero_during_grace_period=True
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        # Fix A baked in: node references are grounded per-frame, so the +5 cm
        # penetration-safety respawn is pure float and causes the reset bounce.
        ref_respawn_offset=0.005,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            # Constant reference => a random start time only truncates the
            # episode. Always start at t=0 so every episode is a full 33.3 s hold.
            init_start_prob=1.0,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build agent configuration."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(2)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(2)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        # 250, not 1000. A near-perfect hold policy is vulnerable to a PPO
        # trust-region blowout (see notes/Skill_graph_nodes.MD 6.1: node_downdog3
        # collapsed at epoch 1126 and never recovered), and `score_based.ckpt` is
        # no help because `best_evaluated_score` saturates at 1.0 here -- it was
        # written *after* the collapse. Dense epoch checkpoints are the only
        # reliable way back. This changes checkpointing only, never the
        # optimisation, so runs stay comparable.
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
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Contact sensors on every body + the two disturbance channels.

    ``contact_bodies="all"`` matches ``mlp_2x_no_contact_rew.py``: in yoga the
    load-bearing contacts are as often hands, forearms, head or knees as feet,
    and ``observe_contacts=True`` would otherwise be blind to the contacts
    actually holding the pose up.

    The push and reset-noise settings are what make this a *node* expert rather
    than a one-pose tracker: the policy is trained to be shoved and to start
    off-pose, so "hold the pose" becomes "stabilise about the pose".
    """
    from protomotions.simulator.base_simulator.config import (
        DomainRandomizationConfig,
        PushDomainRandomizationConfig,
        RobotNoiseConfig,
    )

    robot_cfg.update_fields(
        contact_bodies="all",
        # RSI noise: perturbed initial conditions. Applied AFTER the respawn
        # offset in BaseEnv.reset, i.e. on top of a +1.0 cm clearance.
        reset_noise=RobotNoiseConfig(**RESET_NOISE),
    )

    # Mid-episode shoves. Applied as a root velocity impulse
    # (simulator._apply_root_velocity_impulse), rescheduled after every push and
    # reset to zero on env reset.
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=PUSH_INTERVAL_RANGE,
            max_linear_velocity=PUSH_MAX_LINEAR_VELOCITY,
            max_angular_velocity=PUSH_MAX_ANGULAR_VELOCITY,
        )
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Evaluation-specific overrides.

    Terminations and disturbances are both cleared so plain
    ``inference_agent.py`` shows an undisturbed, non-terminating hold -- useful
    for eyeballing the pose. The stability-region analysis
    (``data/scripts/node_stability_region.py``) does **not** go through this
    hook; it re-enables terminations and sets its own reset noise, because it
    needs failures in order to have a boundary to find.
    """
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    robot_cfg.reset_noise = None
    simulator_cfg.domain_randomization = None
