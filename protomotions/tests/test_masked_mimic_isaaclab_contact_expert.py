# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 propagation tests for an IsaacLab-rich Stage-1 expert."""

import argparse
from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.experiments.masked_mimic import transformer
from examples.experiments.mimic import mlp_contact_rich_isaaclab as rich_experiment
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config
from protomotions.utils import config_utils


def _args(expert_model_path="/tmp/rich_expert/last.ckpt"):
    return argparse.Namespace(
        expert_model_path=expert_model_path,
        motion_file="data/smpl/yoga_yogi_balance_grounded.pt",
        scenes_file=None,
        batch_size=32,
        training_max_steps=256,
    )


def _rich_configs():
    args = _args()
    expert_robot = robot_config("smpl_yogi")
    expert_simulator = simulator_config(
        "isaaclab", expert_robot, True, 8, "expert_unit"
    )
    rich_experiment.configure_robot_and_simulator(
        expert_robot, expert_simulator, args
    )
    expert_env = rich_experiment.env_config(expert_robot, args)
    expert_agent = rich_experiment.agent_config(expert_robot, expert_env, args)
    return {
        "robot": expert_robot,
        "simulator": expert_simulator,
        "env": expert_env,
        "agent": expert_agent,
    }


def test_no_expert_path_is_a_strict_noop():
    args = _args(expert_model_path=None)
    robot = robot_config("smpl_yogi")
    simulator = simulator_config("isaaclab", robot, True, 8, "student_noop")
    sensor_bodies = deepcopy(robot.contact_bodies)
    observation_bodies = deepcopy(robot.contact_observation_bodies)
    reward_bodies = deepcopy(robot.contact_reward_bodies)
    contact_cfg = deepcopy(simulator.contact_sensor_observation)

    transformer.configure_robot_and_simulator(robot, simulator, args)

    assert robot.contact_bodies == sensor_bodies
    assert robot.contact_observation_bodies == observation_bodies
    assert robot.contact_reward_bodies == reward_bodies
    assert simulator.contact_sensor_observation == contact_cfg


def test_rich_expert_propagates_sensors_but_preserves_student_rewards(
    monkeypatch,
):
    expert_configs = _rich_configs()
    monkeypatch.setattr(
        config_utils,
        "load_resolved_configs_from_checkpoint",
        lambda path: expert_configs,
    )
    args = _args()
    robot = robot_config("smpl_yogi")
    simulator = simulator_config("isaaclab", robot, True, 8, "student_rich")
    original_rewards = list(robot.contact_reward_bodies or [])

    transformer.configure_robot_and_simulator(robot, simulator, args)

    assert robot.contact_bodies == robot.kinematic_info.body_names
    assert robot.contact_observation_bodies == robot.kinematic_info.body_names
    assert robot.contact_reward_bodies == original_rewards
    assert simulator.contact_sensor_observation.enabled is True
    assert simulator.contact_sensor_observation.track_pair_data is True
    assert simulator.contact_sensor_observation.track_contact_points is True
    assert simulator.contact_sensor_observation.track_friction_forces is True
    assert (
        simulator.contact_sensor_observation
        is not expert_configs["simulator"].contact_sensor_observation
    )

    student_env = transformer.env_config(robot, args)
    assert "expert_isaaclab_contact_obs_v1" in student_env.observation_components
    assert (
        "expert_isaaclab_contact_pair_obs_v1"
        in student_env.observation_components
    )


def test_rich_expert_rejects_non_isaaclab_student(monkeypatch):
    expert_configs = _rich_configs()
    monkeypatch.setattr(
        config_utils,
        "load_resolved_configs_from_checkpoint",
        lambda path: expert_configs,
    )
    robot = robot_config("smpl_yogi")

    with pytest.raises(ValueError, match="both expert and student.*IsaacLab"):
        transformer.configure_robot_and_simulator(
            robot,
            SimpleNamespace(_target_="fake.GenesisSimulator"),
            _args(),
        )


def test_pair_expert_requires_saved_friction_and_point_tracking(monkeypatch):
    expert_configs = _rich_configs()
    expert_configs["simulator"].contact_sensor_observation.track_friction_forces = (
        False
    )
    monkeypatch.setattr(
        config_utils,
        "load_resolved_configs_from_checkpoint",
        lambda path: expert_configs,
    )
    robot = robot_config("smpl_yogi")
    simulator = simulator_config("isaaclab", robot, True, 8, "student_bad_pair")

    with pytest.raises(ValueError, match="does not track pair data"):
        transformer.configure_robot_and_simulator(robot, simulator, _args())
