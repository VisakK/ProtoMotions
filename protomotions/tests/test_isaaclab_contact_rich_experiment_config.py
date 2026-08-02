# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration regression tests for the opt-in IsaacLab contact expert."""

import argparse
from types import SimpleNamespace

import pytest

from examples.experiments.mimic import mlp_2x_no_contact_rew as baseline
from examples.experiments.mimic import mlp_contact_rich_isaaclab as experiment
from protomotions.envs.obs import (
    isaaclab_contact_obs_v1_dim,
    isaaclab_contact_pair_obs_v1_dim,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        motion_file="data/smpl/yoga_yogi_balance_grounded.pt",
        scenes_file=None,
        batch_size=32,
        training_max_steps=256,
    )


def test_isaaclab_contact_experiment_is_opt_in_and_preserves_rewards():
    args = _args()
    robot_cfg = robot_config("smpl_yogi")
    simulator_cfg = simulator_config(
        "isaaclab", robot_cfg, True, 8, "unit_contact_rich_isaaclab"
    )
    experiment.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)

    sensor_cfg = simulator_cfg.contact_sensor_observation
    assert sensor_cfg.enabled is True
    assert sensor_cfg.track_pair_data is True
    assert sensor_cfg.track_contact_points is True
    assert sensor_cfg.track_friction_forces is True
    assert sensor_cfg.track_air_time is True
    assert sensor_cfg.include_terrain_filter is True
    assert sensor_cfg.include_scene_object_filters is False
    assert sensor_cfg.max_contact_data_count_per_prim == 16

    env_cfg = experiment.env_config(robot_cfg, args)
    agent_cfg = experiment.agent_config(robot_cfg, env_cfg, args)
    baseline_env_cfg = baseline.env_config(robot_cfg, args)
    baseline_agent_cfg = baseline.agent_config(robot_cfg, baseline_env_cfg, args)

    assert robot_cfg.contact_bodies == robot_cfg.kinematic_info.body_names
    assert robot_cfg.contact_observation_bodies == robot_cfg.kinematic_info.body_names
    assert robot_cfg.contact_reward_bodies == [
        "L_Ankle",
        "L_Toe",
        "R_Ankle",
        "R_Toe",
    ]
    assert list(env_cfg.observation_components) == experiment.CONTACT_INPUT_KEYS
    assert "contact_obs_v1" not in env_cfg.observation_components
    assert (
        env_cfg.observation_components["max_coords_obs"].static_params[
            "observe_contacts"
        ]
        is False
    )

    body_ids = list(range(24))
    aggregate = env_cfg.observation_components["isaaclab_contact_obs_v1"]
    pair = env_cfg.observation_components["isaaclab_contact_pair_obs_v1"]
    proximity = env_cfg.observation_components["contact_proximity_obs"]
    assert aggregate.static_params["body_ids"].tolist() == body_ids
    assert pair.static_params["body_ids"].tolist() == body_ids
    assert "sensor_data_valid" in pair.dynamic_vars
    assert proximity.static_params["body_ids"] == body_ids
    assert isaaclab_contact_obs_v1_dim(24) == 486
    assert isaaclab_contact_pair_obs_v1_dim(24, 1) == 360

    assert {
        name: component.to_dict()
        for name, component in env_cfg.reward_components.items()
    } == {
        name: component.to_dict()
        for name, component in baseline_env_cfg.reward_components.items()
    }
    assert env_cfg.reward_components["contact_match_rew"].static_params["weight"] == 0.0
    assert agent_cfg.model.in_keys == experiment.CONTACT_INPUT_KEYS
    assert agent_cfg.model.actor.in_keys == experiment.CONTACT_INPUT_KEYS
    assert agent_cfg.model.actor.mu_model.in_keys == experiment.CONTACT_INPUT_KEYS
    assert agent_cfg.model.critic.in_keys == experiment.CONTACT_INPUT_KEYS
    assert agent_cfg.model.actor_optimizer == baseline_agent_cfg.model.actor_optimizer
    assert agent_cfg.model.critic_optimizer == baseline_agent_cfg.model.critic_optimizer


def test_isaaclab_contact_experiment_rejects_other_backends():
    robot_cfg = robot_config("smpl_yogi")
    with pytest.raises(ValueError, match="requires the IsaacLab simulator"):
        experiment.configure_robot_and_simulator(
            robot_cfg,
            SimpleNamespace(_target_="fake.GenesisSimulator"),
            _args(),
        )


def test_contact_component_builder_supports_future_ablation_variants():
    args = _args()
    robot_cfg = robot_config("smpl_yogi")
    simulator_cfg = simulator_config(
        "isaaclab", robot_cfg, True, 8, "unit_contact_ablation"
    )
    experiment.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    base_env = baseline.env_config(robot_cfg, args)

    aggregate_only = experiment.build_contact_observation_components(
        robot_cfg,
        base_env.observation_components,
        include_pair_obs=False,
        include_proximity_obs=False,
    )

    assert list(aggregate_only) == [
        "max_coords_obs",
        "isaaclab_contact_obs_v1",
        "mimic_target_poses",
        "previous_actions",
    ]
