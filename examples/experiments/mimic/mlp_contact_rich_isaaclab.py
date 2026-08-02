# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab-specific contact-rich Stage-1 MaskedMimic expert.

This experiment is an opt-in sibling of :mod:`mlp_contact_rich`.  It preserves
the 2x no-contact-reward tracker, all tracking rewards, the feet-only
contact-match diagnostic, and the nearest-surface input, but replaces the
backend-common ``contact_obs_v1`` key with two explicitly IsaacLab contracts:

* ``isaaclab_contact_obs_v1`` (``20*K+6``): body-weight-normalized aggregate
  normal force, a full control interval of physics-substep statistics, and
  force-only temporal state; and
* ``isaaclab_contact_pair_obs_v1`` (``15*K*F``): filtered normal force, true
  filtered friction force, and IsaacLab's average contact position.

The default pair contract is terrain-only (``F=1``).  This keeps checkpoint
shape and filter meaning portable between the Stage-1 ground-only run and a
later MaskedMimic distillation run.  Scene-object filters can be enabled in a
derived experiment, but doing so is a distinct checkpoint contract.
IsaacLab contact-point and friction extraction may materially reduce throughput
with all 24 sensors; benchmark the default pair mode at the intended environment
count, or use ``--disable-isaaclab-contact-pair`` for the aggregate-only ablation.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_contact_rich_isaaclab.py \
      --experiment-name smpl_yogi_balance_contact_rich_isaaclab \
      --motion-file data/smpl/yoga_yogi_balance_grounded.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb \
      --overrides env.ref_respawn_offset=0.005

One-epoch smoke::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_contact_rich_isaaclab.py \
      --experiment-name smpl_yogi_contact_rich_isaaclab_smoke \
      --motion-file data/smpl/yoga_yogi_balance_grounded.pt \
      --num-envs 8 --batch-size 32 --training-max-steps 256 --headless True \
      --overrides env.ref_respawn_offset=0.005 env.contact_diagnostics_interval=1
"""

from __future__ import annotations

import argparse

from examples.experiments.mimic import mlp_2x_no_contact_rew as base
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
apply_inference_overrides = base.apply_inference_overrides


INCLUDE_PAIR_OBS = True
INCLUDE_PROXIMITY_OBS = True

ISAACLAB_CONTACT_OBS_V1_PARAMS = {
    "contact_on_threshold_n": 5.0,
    "contact_off_threshold_n": 2.0,
    "force_clip_bodyweights": 10.0,
    "force_delta_clip_bodyweights": 10.0,
    "velocity_clip_mps": 5.0,
    "duration_clip_s": 2.0,
    "fallback_force_reference_n": 600.0,
    "force_epsilon_n": 1.0e-4,
    "load_fraction_epsilon": 1.0e-6,
}

ISAACLAB_CONTACT_PAIR_OBS_V1_PARAMS = {
    "contact_on_threshold_n": 5.0,
    "force_clip_bodyweights": 10.0,
    "tangential_normal_ratio_clip": 2.0,
    "contact_point_scale_m": 1.0,
    "fallback_force_reference_n": 600.0,
    "force_epsilon_n": 1.0e-4,
}


def _input_keys(
    *, include_pair_obs: bool, include_proximity_obs: bool
) -> list[str]:
    keys = ["max_coords_obs", "isaaclab_contact_obs_v1"]
    if include_pair_obs:
        keys.append("isaaclab_contact_pair_obs_v1")
    if include_proximity_obs:
        keys.append("contact_proximity_obs")
    keys.extend(("mimic_target_poses", "previous_actions"))
    return keys


CONTACT_INPUT_KEYS = _input_keys(
    include_pair_obs=INCLUDE_PAIR_OBS,
    include_proximity_obs=INCLUDE_PROXIMITY_OBS,
)


def additional_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose structural ablations before observation/model configs are built."""
    parser.add_argument(
        "--disable-isaaclab-contact-pair",
        action="store_true",
        help=(
            "Use only the 20*K+6 aggregate IsaacLab contact tensor. This is a "
            "new checkpoint shape and disables filtered friction/contact points."
        ),
    )
    parser.add_argument(
        "--disable-contact-proximity",
        action="store_true",
        help="Remove the separate nearest-sampled-surface contact input.",
    )


def _include_pair_obs(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "disable_isaaclab_contact_pair", False))


def _include_proximity_obs(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "disable_contact_proximity", False))


def _contact_observation_body_ids(robot_cfg: RobotConfig) -> list[int]:
    body_names = robot_cfg.contact_observation_bodies or []
    if not body_names:
        raise ValueError(
            "mlp_contact_rich_isaaclab requires contact observation bodies; "
            "call configure_robot_and_simulator() first"
        )
    name_to_id = {
        name: body_id
        for body_id, name in enumerate(robot_cfg.kinematic_info.body_names)
    }
    return [name_to_id[name] for name in body_names]


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    """Enable all-body sensing and the explicit IsaacLab pair capability."""
    robot_cfg.update_fields(
        contact_bodies="all",
        contact_observation_bodies="all",
        contact_reward_bodies=[
            "all_left_foot_bodies",
            "all_right_foot_bodies",
        ],
    )

    contact_cfg = getattr(simulator_cfg, "contact_sensor_observation", None)
    target = getattr(simulator_cfg, "_target_", type(simulator_cfg).__name__)
    if contact_cfg is None or "isaaclab" not in target.lower():
        raise ValueError(
            "mlp_contact_rich_isaaclab requires the IsaacLab simulator; "
            f"received '{target}'"
        )
    contact_cfg.enabled = True
    include_pair_obs = _include_pair_obs(args)
    contact_cfg.track_pair_data = include_pair_obs
    contact_cfg.track_contact_points = include_pair_obs
    contact_cfg.track_friction_forces = include_pair_obs
    contact_cfg.track_air_time = True
    contact_cfg.force_threshold_n = 2.0
    contact_cfg.max_contact_data_count_per_prim = 16
    contact_cfg.include_terrain_filter = True
    contact_cfg.include_scene_object_filters = False


def build_contact_observation_components(
    robot_cfg: RobotConfig,
    baseline_observations: dict,
    *,
    include_pair_obs: bool = INCLUDE_PAIR_OBS,
    include_proximity_obs: bool = INCLUDE_PROXIMITY_OBS,
) -> dict:
    """Build a reusable mapping for aggregate/pair/proximity ablations."""
    from protomotions.envs.component_factories import (
        isaaclab_contact_obs_v1_factory,
        isaaclab_contact_pair_obs_v1_factory,
        max_coords_obs_factory,
        nearest_surface_obs_factory,
    )

    body_ids = _contact_observation_body_ids(robot_cfg)
    observations = {
        "max_coords_obs": max_coords_obs_factory(observe_contacts=False),
        "isaaclab_contact_obs_v1": isaaclab_contact_obs_v1_factory(
            body_ids=body_ids,
            **ISAACLAB_CONTACT_OBS_V1_PARAMS,
        ),
    }
    if include_pair_obs:
        observations["isaaclab_contact_pair_obs_v1"] = (
            isaaclab_contact_pair_obs_v1_factory(
                body_ids=body_ids,
                **ISAACLAB_CONTACT_PAIR_OBS_V1_PARAMS,
            )
        )
    if include_proximity_obs:
        observations["contact_proximity_obs"] = nearest_surface_obs_factory(
            body_ids=body_ids
        )
    observations.update(
        {
            "mimic_target_poses": baseline_observations["mimic_target_poses"],
            "previous_actions": baseline_observations["previous_actions"],
        }
    )
    return observations


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build the unchanged tracker with IsaacLab-specific contact inputs."""
    cfg = base.env_config(robot_cfg, args)
    cfg.observation_components = build_contact_observation_components(
        robot_cfg,
        cfg.observation_components,
        include_pair_obs=_include_pair_obs(args),
        include_proximity_obs=_include_proximity_obs(args),
    )
    cfg.contact_diagnostics_interval = 100
    return cfg


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> PPOAgentConfig:
    """Use the new keys for actor and critic without changing network widths."""
    cfg = base.agent_config(robot_config, env_config, args)
    input_keys = _input_keys(
        include_pair_obs=_include_pair_obs(args),
        include_proximity_obs=_include_proximity_obs(args),
    )
    cfg.model.in_keys = list(input_keys)
    cfg.model.actor.in_keys = list(input_keys)
    cfg.model.actor.mu_model.in_keys = list(input_keys)
    cfg.model.critic.in_keys = list(input_keys)
    return cfg
