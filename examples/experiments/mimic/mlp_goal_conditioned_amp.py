# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Goal-conditioned expert + an AMP style reward (graph-growth round: card E2, run G1).

``expert_revist/graph_growth_2026_10_03/PLAN.MD`` §1.2 and cards E2/G1. Everything the
base expert builds is built by ``mlp_goal_conditioned.py``, which this file loads and
leaves byte-identical: the robot and sensors, the deployable actor contract
(``ACTOR_KEYS``), the critic's privileged window, the tracking + support + physics
reward stack, the hold-aware evaluator and its curriculum, and the viz panel. It adds:

* **env**: an ``amp_obs`` observation -- ``amp_features_v1``, heading-free, 163 values
  per frame over 8 frames at steps ``[1, 2, 3, 4, 6, 9, 14, 20]`` (0.67 s at 30 Hz;
  ``protomotions/envs/obs/amp_features.py``) -- and ``num_state_history_steps``
  15 -> 20 so the window fits. The actor's own ``[1, 8, 15]`` pose history reads the
  same buffer indices as before. Neither the actor nor the critic consumes ``amp_obs``.
* **agent**: ``GoalConditionedAMP`` (``protomotions/agents/amp/goal_conditioned.py``) --
  a discriminator (MLP 1024-512 on ``amp_obs``) and a discriminator critic (3 x 1024 on
  the critic's keys + ``amp_obs``); demonstrations are the x0 clips only, uniform per
  frame, never Scorpion -b; the style weight is 0 for ``--amp-w-start-epoch`` epochs
  and ramps linearly to ``--amp-reward-w`` at ``--amp-w-full-epoch``; no discriminator
  termination (threshold 0).

Warm start from a non-AMP checkpoint (``--checkpoint ... --warm-start-optimization-state``,
a new experiment name): the actor, critic, their normalisers, the PPO optimisers, the
advantage EMA and the task-reward normaliser load; the discriminator networks start
random (``AMPAgentMixin._load_model_state_dict``). The task reward is unchanged, which is
what makes restoring the optimisation state correct. Launcher:
``data/scripts/run_expert_amp_ft.sh`` (``CONTROL=1`` runs the matched no-AMP
continuation with the base experiment file).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig


def _load_base_module():
    """The base expert, by file path (this file may run from a copy in results/<exp>/)."""
    here = Path(__file__).resolve().parent / "mlp_goal_conditioned.py"
    repo = Path.cwd() / "examples" / "experiments" / "mimic" / "mlp_goal_conditioned.py"
    path = here if here.exists() else repo
    spec = importlib.util.spec_from_file_location("mlp_goal_conditioned_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

ACTOR_KEYS = base.ACTOR_KEYS
CRITIC_KEYS = base.CRITIC_KEYS
AMP_OBS_KEY = "amp_obs"
# The window must fit the history buffer: step k reads buffer index k.
AMP_STATE_HISTORY_STEPS = 20

terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator


def _bool(v) -> bool:
    return str(v).lower() not in ("0", "false", "no")


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    base.additional_experiment_arguments(parser)
    g = parser.add_argument_group("AMP (graph-growth card E2)")
    g.add_argument("--amp-reward-w", type=float, default=0.1,
                   help="discriminator_reward_w once the ramp completes (PLAN.MD E2: 0.1; 0.2 only if jerk "
                        "does not move and no group regresses).")
    g.add_argument("--amp-w-start-epoch", type=int, default=200,
                   help="Epochs [0, this) pay no style reward; the discriminator trains against the policy.")
    g.add_argument("--amp-w-full-epoch", type=int, default=500,
                   help="The style weight reaches its target here (linear ramp).")
    g.add_argument("--amp-calibrate-style-ratio", type=float, default=0.0,
                   help="> 0: at --amp-w-start-epoch, set the target weight so the style advantage's std is this "
                        "fraction of the task advantage's (median over the preceding --amp-calibrate-window w = 0 "
                        "epochs; PLAN.MD calibrates to 0.2-0.35), clamped to [--amp-w-min, --amp-w-max]. "
                        "0 = use --amp-reward-w.")
    g.add_argument("--amp-calibrate-window", type=int, default=100)
    g.add_argument("--amp-w-min", type=float, default=0.05)
    g.add_argument("--amp-w-max", type=float, default=0.5)
    g.add_argument("--amp-disc-batch-size", type=int, default=4096)
    g.add_argument("--amp-grad-penalty", type=float, default=5.0,
                   help="Discriminator gradient penalty (raise to 10 if agent accuracy stays above 0.9).")
    g.add_argument("--amp-demo-exclude-motions", type=str, nargs="*",
                   default=["Scorpion_pose_or_vrischikasana-b"],
                   help="Motion-stem substrings never used as demonstrations (x3s/x7s variants are always excluded).")
    g.add_argument("--amp-parity-check", type=_bool, default=True,
                   help="Compare agent and demonstration features once at the first rollout step.")


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.amp_features import (
        AMP_FEATURES_V1_STEPS,
        amp_features_v1_params,
        compute_amp_features_v1_from_state,
    )

    cfg = base.env_config(robot_cfg, args)
    if cfg.num_state_history_steps > AMP_STATE_HISTORY_STEPS:
        raise ValueError("the base expert already stores a longer history than the AMP window needs")
    cfg.num_state_history_steps = AMP_STATE_HISTORY_STEPS
    ki = robot_cfg.kinematic_info
    hist = EnvContext.historical
    cfg.observation_components[AMP_OBS_KEY] = MdpComponent(
        compute_func=compute_amp_features_v1_from_state,
        dynamic_vars={
            "historical_rigid_body_pos": hist.rigid_body_pos,
            "historical_rigid_body_rot": hist.rigid_body_rot,
            "historical_rigid_body_vel": hist.rigid_body_vel,
            "historical_rigid_body_ang_vel": hist.rigid_body_ang_vel,
            "historical_ground_heights": hist.ground_heights,
        },
        static_params={
            "history_steps": list(AMP_FEATURES_V1_STEPS),
            **amp_features_v1_params(ki.body_names, ki.parent_indices),
        },
    )
    if max(AMP_FEATURES_V1_STEPS) > cfg.num_state_history_steps:
        raise ValueError("amp_features_v1 window does not fit the state history buffer")
    return cfg


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace):
    from protomotions.agents.amp.config import AMPModelConfig, AMPParametersConfig, DiscriminatorConfig
    from protomotions.agents.amp.goal_conditioned import GoalConditionedAMPAgentConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig, ModuleContainerConfig
    from protomotions.agents.ppo.config import PPOAgentConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.amp_features import (
        AMP_FEATURES_V1_STEPS,
        amp_features_v1_params,
        compute_amp_features_v1_from_motion_lib,
    )

    ppo = base.agent_config(robot_config, env_config, args)
    if not isinstance(ppo, PPOAgentConfig):
        raise TypeError(f"base agent_config returned {type(ppo).__name__}, expected PPOAgentConfig")
    ki = robot_config.kinematic_info
    feature_params = amp_features_v1_params(ki.body_names, ki.parent_indices)

    discriminator = DiscriminatorConfig(
        in_keys=[AMP_OBS_KEY],
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=[AMP_OBS_KEY],
                out_keys=["disc_logits"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[MLPLayerConfig(units=1024, activation="relu"),
                        MLPLayerConfig(units=512, activation="relu")],
            )
        ],
    )
    disc_critic_keys = list(CRITIC_KEYS) + [AMP_OBS_KEY]
    disc_critic = ModuleContainerConfig(
        in_keys=disc_critic_keys,
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=disc_critic_keys,
                out_keys=["disc_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
            )
        ],
    )
    model = AMPModelConfig(
        in_keys=list(CRITIC_KEYS) + [AMP_OBS_KEY],
        out_keys=["action", "mean_action", "neglogp", "value", "disc_logits", "disc_value"],
        actor=ppo.model.actor,
        critic=ppo.model.critic,
        actor_optimizer=ppo.model.actor_optimizer,
        critic_optimizer=ppo.model.critic_optimizer,
        discriminator=discriminator,
        disc_critic=disc_critic,
        discriminator_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        disc_critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
    )
    amp_parameters = AMPParametersConfig(
        discriminator_reward_w=0.0,                 # set every epoch by the schedule
        discriminator_weight_decay=1e-4,
        discriminator_logit_weight_decay=0.01,
        discriminator_batch_size=int(getattr(args, "amp_disc_batch_size", 4096)),
        discriminator_grad_penalty=float(getattr(args, "amp_grad_penalty", 5.0)),
        discriminator_optimization_ratio=1,
        discriminator_replay_keep_prob=0.01,
        discriminator_replay_size=200000,
        discriminator_reward_threshold=0.0,         # no discriminator termination
        use_disc_critic=True,
    )
    reference_obs_components = {
        AMP_OBS_KEY: MdpComponent(
            compute_func=compute_amp_features_v1_from_motion_lib,
            dynamic_vars={},                         # motion_lib / ids / times / dt come from the agent
            static_params={"history_steps": list(AMP_FEATURES_V1_STEPS), **feature_params},
        ),
    }

    # Every PPO setting the base expert uses (evaluator + curriculum, viz panel, batch size, clipping,
    # advantage normalisation, checkpoint cadence, ...) carries over unchanged.
    carried = {f.name: getattr(ppo, f.name) for f in dataclasses.fields(PPOAgentConfig)
               if f.init and f.name not in ("_target_", "model")}
    return GoalConditionedAMPAgentConfig(
        **carried,
        model=model,
        amp_parameters=amp_parameters,
        reference_obs_components=reference_obs_components,
        amp_reward_w_target=float(getattr(args, "amp_reward_w", 0.1)),
        amp_reward_w_start_epoch=int(getattr(args, "amp_w_start_epoch", 200)),
        amp_reward_w_full_epoch=int(getattr(args, "amp_w_full_epoch", 500)),
        amp_calibrate_style_ratio=float(getattr(args, "amp_calibrate_style_ratio", 0.0) or 0.0),
        amp_calibrate_window=int(getattr(args, "amp_calibrate_window", 100)),
        amp_reward_w_min=float(getattr(args, "amp_w_min", 0.05)),
        amp_reward_w_max=float(getattr(args, "amp_w_max", 0.5)),
        demo_exclude_motions=list(getattr(args, "amp_demo_exclude_motions", None) or []),
        demo_min_time_steps=max(AMP_FEATURES_V1_STEPS),
        amp_parity_check=bool(getattr(args, "amp_parity_check", True)),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """The base expert's evaluation overrides; AMP never terminates an episode."""
    base.apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
                                   motion_lib_cfg, scene_lib_cfg, args)
    if agent_cfg is not None and hasattr(agent_cfg, "amp_parameters"):
        agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0
