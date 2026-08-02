# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused BaseEnv lifecycle tests for the rich IsaacLab contact capability."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.component_factories import (
    isaaclab_contact_obs_v1_factory,
    isaaclab_contact_pair_obs_v1_factory,
)
from protomotions.simulator.base_simulator.contact_sensor_state import (
    ContactFilterMetadata,
    ContactSensorState,
)


def _contact_state(
    *,
    force_n: float = 10.0,
    lifecycle_valid: torch.Tensor | None = None,
) -> ContactSensorState:
    num_envs, num_sensors, history, num_filters = 3, 2, 2, 1
    if lifecycle_valid is None:
        lifecycle_valid = torch.tensor([True, True, False])
    normal = torch.zeros(num_envs, num_sensors, 3)
    # Sensor order is hand, root. Observation order below is root, hand.
    normal[:, 0, 2] = force_n
    normal[:, 1, 2] = force_n * 2.0
    normal_history = normal[:, None].repeat(1, history, 1, 1)
    normal_valid = lifecycle_valid[:, None].expand(-1, num_sensors).clone()
    history_valid = normal_valid[:, None].expand(-1, history, -1).clone()

    filtered = normal[:, :, None, :].clone()
    filtered_history = filtered[:, None].repeat(1, history, 1, 1, 1)
    pair_measurement_valid = normal_valid[:, :, None].expand(
        -1, -1, num_filters
    ).clone()
    pair_slot_valid = torch.ones(
        num_envs, num_sensors, num_filters, dtype=torch.bool
    )
    friction = torch.zeros_like(filtered)
    friction[..., 0] = filtered[..., 2] * 0.25
    mean_point = torch.zeros_like(filtered)
    mean_point[..., 0] = 0.1

    return ContactSensorState(
        body_names=("hand", "root"),
        sim_body_indices=torch.tensor([1, 0]),
        common_body_indices=torch.tensor([2, 0]),
        filter_metadata=ContactFilterMetadata(
            labels=("terrain",),
            prim_path_exprs=("/World/ground/terrain/mesh",),
            scene_object_indices=(None,),
        ),
        normal_force_w=normal,
        normal_force_history_w=normal_history,
        sensor_data_valid=lifecycle_valid,
        normal_force_valid=normal_valid,
        normal_force_history_valid=history_valid,
        filtered_normal_force_w=filtered,
        filtered_normal_force_history_w=filtered_history,
        filtered_normal_force_valid=pair_measurement_valid,
        filtered_normal_force_history_valid=pair_measurement_valid[:, None].expand(
            -1, history, -1, -1
        ).clone(),
        friction_force_w=friction,
        friction_force_valid=pair_measurement_valid.clone(),
        mean_contact_point_w=mean_point,
        mean_contact_point_valid=pair_measurement_valid.clone(),
        pair_slot_valid=pair_slot_valid,
        body_weight_n=torch.full((num_envs, 1), 100.0),
    )


def _env(*, with_pair: bool = True) -> BaseEnv:
    env = object.__new__(BaseEnv)
    env.device = torch.device("cpu")
    env.num_envs = 3
    env.dt = 0.05
    env.config = EnvConfig()
    components = {
        # Prefix verifies Stage-2 expert component discovery is not key-literal.
        "expert_isaaclab_contact_obs_v1": isaaclab_contact_obs_v1_factory(
            [0, 2]
        )
    }
    if with_pair:
        components["expert_isaaclab_contact_pair_obs_v1"] = (
            isaaclab_contact_pair_obs_v1_factory([0, 2])
        )
    env.config.observation_components = components
    env.robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(
            num_bodies=3,
            body_names=["root", "middle", "hand"],
        ),
        contact_bodies=["root", "hand"],
        contact_observation_bodies=["root", "hand"],
    )
    state = _contact_state()
    env.simulator = SimpleNamespace(
        config=SimpleNamespace(_target_="fake.IsaacLabSimulator"),
        decimation=2,
        get_contact_sensor_state=lambda: state,
    )
    env.isaaclab_previous_normal_force_w = None
    env.isaaclab_previous_active = None
    env.isaaclab_contact_age_s = None
    env.isaaclab_air_age_s = None
    env.isaaclab_contact_temporal_valid = None
    env._isaaclab_contact_sensor_indices = None
    env._isaaclab_contact_body_ids = None
    env._initialize_isaaclab_contact_buffers()
    return env


def test_prefixed_components_validate_and_select_common_body_order():
    env = _env()
    state = env.simulator.get_contact_sensor_state()
    core, pair = env._isaaclab_contact_components()

    env._validate_isaaclab_contact_observation_support(state, core, pair)
    context = env._build_isaaclab_contact_context(state)

    assert env._isaaclab_contact_sensor_indices.tolist() == [1, 0]
    assert torch.equal(
        context.normal_force_w[:, :, 2],
        torch.tensor([[20.0, 10.0], [20.0, 10.0], [20.0, 10.0]]),
    )
    assert context.filtered_normal_force_w.shape == (3, 2, 1, 3)
    assert context.friction_force_w.shape == (3, 2, 1, 3)
    assert context.mean_contact_point_w.shape == (3, 2, 1, 3)
    assert torch.equal(
        context.sensor_data_valid, torch.tensor([True, True, False])
    )


def test_rich_contact_commit_and_partial_reset_are_policy_boundary_safe():
    env = _env()
    first = env.simulator.get_contact_sensor_state()
    core, pair = env._isaaclab_contact_components()
    env._validate_isaaclab_contact_observation_support(first, core, pair)

    env._finalize_isaaclab_contact_state(first)
    assert env.isaaclab_contact_temporal_valid.tolist() == [True, True, False]
    assert env.isaaclab_previous_active[:2].all()
    # First valid sample classifies contact but cannot fabricate an age/event.
    assert not env.isaaclab_contact_age_s.any()

    second = _contact_state(
        force_n=3.0,
        lifecycle_valid=torch.tensor([True, True, True]),
    )
    env._finalize_isaaclab_contact_state(second)
    # 3/6 N sensor loads retain prior active contacts through 2 N hysteresis;
    # the previously invalid third environment starts without a false onset.
    assert env.isaaclab_previous_active[:2].all()
    assert torch.allclose(
        env.isaaclab_contact_age_s[:2], torch.full((2, 2), env.dt)
    )
    assert env.isaaclab_previous_active[2, 0]
    assert not env.isaaclab_previous_active[2, 1]
    assert not env.isaaclab_contact_age_s[2].any()

    preserved = env.isaaclab_previous_normal_force_w[[0, 2]].clone()
    env._reset_isaaclab_contact_state(torch.tensor([1]))
    assert not env.isaaclab_previous_normal_force_w[1].any()
    assert not env.isaaclab_previous_active[1].any()
    assert not env.isaaclab_contact_age_s[1].any()
    assert env.isaaclab_contact_temporal_valid[1].item() is False
    assert torch.equal(
        env.isaaclab_previous_normal_force_w[[0, 2]], preserved
    )


def test_invalid_sensor_interval_clears_temporal_state_before_recovery():
    env = _env()
    first = env.simulator.get_contact_sensor_state()
    core, pair = env._isaaclab_contact_components()
    env._validate_isaaclab_contact_observation_support(first, core, pair)
    env._finalize_isaaclab_contact_state(first)
    assert env.isaaclab_previous_active[1].all()

    invalid_gap = _contact_state(
        force_n=10.0,
        lifecycle_valid=torch.tensor([True, False, False]),
    )
    env._finalize_isaaclab_contact_state(invalid_gap)
    assert env.isaaclab_contact_temporal_valid.tolist() == [True, False, False]
    assert not env.isaaclab_previous_normal_force_w[1:].any()
    assert not env.isaaclab_previous_active[1:].any()
    assert not env.isaaclab_contact_age_s[1:].any()
    assert not env.isaaclab_air_age_s[1:].any()

    recovered = _contact_state(
        force_n=3.0,
        lifecycle_valid=torch.tensor([True, True, True]),
    )
    recovered_context = env._build_isaaclab_contact_context(recovered)
    assert recovered_context.temporal_valid.tolist() == [True, False, False]
    env._finalize_isaaclab_contact_state(recovered)
    # Sensor order is hand/root; policy order is root/hand.  The 6 N root
    # classifies active, while the 3 N hand cannot inherit pre-gap hysteresis.
    assert env.isaaclab_previous_active[1:].tolist() == [
        [True, False],
        [True, False],
    ]
    assert not env.isaaclab_contact_age_s[1:].any()


def test_rich_contact_validation_fails_without_capability_or_pair_measurements():
    env = _env()
    core, pair = env._isaaclab_contact_components()
    with pytest.raises(RuntimeError, match="rich contact-sensor capability"):
        env._validate_contact_observation_support(
            current_state=SimpleNamespace(), isaaclab_contact_state=None
        )

    state = _contact_state()
    state.friction_force_w = None
    with pytest.raises(RuntimeError, match="missing fields.*friction_force_w"):
        env._validate_isaaclab_contact_observation_support(state, core, pair)


def test_rich_contact_pair_body_order_must_match_aggregate():
    env = _env(with_pair=False)
    env.config.observation_components["bad_pair"] = (
        isaaclab_contact_pair_obs_v1_factory([2, 0])
    )
    with pytest.raises(ValueError, match="must use the aggregate.*body order"):
        env._initialize_isaaclab_contact_buffers()
