# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic tests for current-contact observation kernels."""

import math

import pytest
import torch

from protomotions.envs.obs.contact import (
    CONTACT_OBS_V1_GLOBAL_DIM,
    CONTACT_OBS_V1_GLOBAL_LAYOUT,
    CONTACT_OBS_V1_LAYOUT,
    CONTACT_OBS_V1_PER_BODY_DIM,
    compute_contact_obs_v1,
    contact_obs_v1_dim,
    signed_log_compress,
    unflatten_contact_obs_v1,
    unsigned_log_compress,
    update_contact_state,
)
from protomotions.utils import rotations


def _identity_quat(
    num_envs: int, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    quat = torch.zeros(num_envs, 4, dtype=dtype)
    quat[:, 3] = 1.0
    return quat


def _contact_inputs(
    num_envs: int = 2,
    num_bodies: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
):
    return {
        "root_rot": _identity_quat(num_envs, dtype=dtype),
        "rigid_body_vel": torch.zeros(num_envs, num_bodies, 3, dtype=dtype),
        "rigid_body_contact_forces": torch.zeros(num_envs, num_bodies, 3, dtype=dtype),
        "previous_contact_forces": torch.zeros(num_envs, num_bodies, 3, dtype=dtype),
        "contact_active_state": torch.zeros(num_envs, num_bodies, dtype=torch.bool),
        "contact_age_steps": torch.zeros(num_envs, num_bodies, dtype=torch.long),
        "contact_air_age_steps": torch.zeros(num_envs, num_bodies, dtype=torch.long),
        "contact_temporal_valid": torch.zeros(num_envs, dtype=torch.bool),
        "dt": 0.1,
    }


def _feature(per_body: torch.Tensor, name: str) -> torch.Tensor:
    return per_body[..., CONTACT_OBS_V1_LAYOUT[name]].squeeze(-1)


def _global_feature(global_obs: torch.Tensor, name: str) -> torch.Tensor:
    return global_obs[..., CONTACT_OBS_V1_GLOBAL_LAYOUT[name]].squeeze(-1)


def test_contact_obs_v1_shape_finite_dtype_and_debug_split():
    inputs = _contact_inputs(num_envs=3, num_bodies=5, dtype=torch.float64)
    inputs["rigid_body_contact_forces"][0, 4] = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float64
    )
    body_ids = torch.tensor([4, 1, 3], dtype=torch.long)

    observation = compute_contact_obs_v1(**inputs, body_ids=body_ids)
    per_body, global_obs = unflatten_contact_obs_v1(observation, 3)

    assert CONTACT_OBS_V1_PER_BODY_DIM == 17
    assert CONTACT_OBS_V1_GLOBAL_DIM == 4
    assert contact_obs_v1_dim(3) == 55
    assert observation.shape == (3, 55)
    assert per_body.shape == (3, 3, 17)
    assert global_obs.shape == (3, 4)
    assert observation.dtype == torch.float64
    assert torch.isfinite(observation).all()

    with pytest.raises(ValueError, match="positive integer"):
        contact_obs_v1_dim(0)
    with pytest.raises(ValueError, match="must have shape"):
        unflatten_contact_obs_v1(torch.zeros(3, 54), 3)


def test_contact_obs_v1_zero_contact_is_exactly_zero_and_finite():
    inputs = _contact_inputs(num_envs=2, num_bodies=3)

    observation = compute_contact_obs_v1(**inputs, body_ids=[2, 0])

    assert torch.equal(observation, torch.zeros_like(observation))
    assert torch.isfinite(observation).all()


def test_contact_obs_v1_static_load_friction_and_load_distribution():
    inputs = _contact_inputs(num_envs=1, num_bodies=2)
    inputs["rigid_body_contact_forces"][0, 0] = torch.tensor([4.0, 0.0, 8.0])
    inputs["rigid_body_contact_forces"][0, 1] = torch.tensor([0.0, 0.0, 2.0])
    inputs["contact_active_state"][0] = True

    observation = compute_contact_obs_v1(
        **inputs,
        body_ids=[0, 1],
        force_reference_n=1.0,
        force_clip_n=100.0,
        friction_mu=2.0,
        friction_utilization_clip=2.0,
    )
    per_body, global_obs = unflatten_contact_obs_v1(observation, 2)

    expected_force_heading = signed_log_compress(
        torch.tensor([4.0, 0.0, 8.0]), 1.0, 100.0
    )
    assert torch.allclose(
        per_body[0, 0, CONTACT_OBS_V1_LAYOUT["net_force_heading"]],
        expected_force_heading,
    )
    assert _feature(per_body, "upward_force_proxy")[0, 0] == pytest.approx(
        unsigned_log_compress(torch.tensor(8.0), 1.0, 100.0).item()
    )
    assert _feature(per_body, "horizontal_force_proxy")[0, 0] == pytest.approx(
        unsigned_log_compress(torch.tensor(4.0), 1.0, 100.0).item()
    )
    assert _feature(per_body, "ground_friction_utilization_proxy")[
        0, 0
    ] == pytest.approx((4.0 / (2.0 * 8.0 + 1e-6)) / 2.0)
    assert torch.allclose(
        _feature(per_body, "support_load_fraction_proxy"),
        torch.tensor([[0.8, 0.2]]),
    )
    assert torch.equal(
        _feature(per_body, "body_origin_normal_velocity_proxy"),
        torch.zeros(1, 2),
    )
    assert torch.equal(
        _feature(per_body, "body_origin_tangent_speed_proxy"),
        torch.zeros(1, 2),
    )
    assert _global_feature(global_obs, "any_selected_contact").item() == 1.0
    assert _global_feature(global_obs, "active_body_fraction").item() == 1.0
    assert _global_feature(
        global_obs, "total_upward_force_proxy"
    ).item() == pytest.approx(
        unsigned_log_compress(torch.tensor(10.0), 1.0, 100.0).item()
    )
    assert _global_feature(
        global_obs, "total_horizontal_force_proxy"
    ).item() == pytest.approx(
        unsigned_log_compress(torch.tensor(4.0), 1.0, 100.0).item()
    )


def test_total_horizontal_force_is_norm_of_summed_horizontal_vector():
    inputs = _contact_inputs(num_envs=1, num_bodies=2)
    inputs["rigid_body_contact_forces"][0, 0, 0] = 10.0
    inputs["rigid_body_contact_forces"][0, 1, 0] = -10.0

    observation = compute_contact_obs_v1(**inputs)
    per_body, global_obs = unflatten_contact_obs_v1(observation, 2)

    assert torch.all(_feature(per_body, "horizontal_force_proxy") > 0)
    assert _global_feature(global_obs, "total_horizontal_force_proxy").item() == 0.0


def test_contact_obs_v1_heading_invariance_under_common_world_yaw():
    inputs = _contact_inputs(num_envs=2, num_bodies=3)
    inputs["rigid_body_contact_forces"] = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]],
            [[-2.0, 3.0, 4.0], [5.0, 6.0, -7.0], [8.0, 9.0, 10.0]],
        ]
    )
    inputs["previous_contact_forces"] = inputs["rigid_body_contact_forces"] * 0.25
    inputs["rigid_body_vel"] = torch.tensor(
        [
            [[0.5, -1.0, 0.2], [1.5, 2.0, -0.3], [-0.5, 0.2, 0.7]],
            [[2.0, 1.0, 0.1], [-1.0, 0.5, -0.4], [0.3, -0.8, 0.6]],
        ]
    )
    inputs["contact_active_state"][:, [0, 2]] = True
    inputs["contact_temporal_valid"][:] = True

    baseline = compute_contact_obs_v1(**inputs, body_ids=[2, 0])

    yaw_angle = torch.tensor([0.7, -1.2])
    yaw_axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    yaw_quat = rotations.quat_from_angle_axis(yaw_angle, yaw_axis, w_last=True)
    expanded_yaw = yaw_quat[:, None, :].expand(-1, 3, -1)
    rotated_inputs = dict(inputs)
    rotated_inputs["root_rot"] = yaw_quat
    rotated_inputs["rigid_body_contact_forces"] = rotations.quat_rotate(
        expanded_yaw, inputs["rigid_body_contact_forces"], w_last=True
    )
    rotated_inputs["previous_contact_forces"] = rotations.quat_rotate(
        expanded_yaw, inputs["previous_contact_forces"], w_last=True
    )
    rotated_inputs["rigid_body_vel"] = rotations.quat_rotate(
        expanded_yaw, inputs["rigid_body_vel"], w_last=True
    )

    rotated = compute_contact_obs_v1(**rotated_inputs, body_ids=[2, 0])

    assert torch.allclose(rotated, baseline, atol=1e-6, rtol=1e-6)


def test_force_rate_validity_duration_encoding_and_selected_body_order():
    inputs = _contact_inputs(num_envs=2, num_bodies=3)
    inputs["rigid_body_contact_forces"][:, 2] = torch.tensor([10.0, -5.0, 2.0])
    inputs["previous_contact_forces"][:, 2] = torch.tensor([2.0, -1.0, 0.0])
    inputs["rigid_body_contact_forces"][:, 0] = torch.tensor([1.0, 2.0, 3.0])
    inputs["contact_temporal_valid"] = torch.tensor([False, True])
    inputs["contact_age_steps"][:, 2] = torch.tensor([2, 20])
    inputs["contact_air_age_steps"][:, 2] = torch.tensor([3, 40])
    inputs["dt"] = 0.25

    observation = compute_contact_obs_v1(
        **inputs,
        body_ids=[2, 0],
        force_rate_reference_n_per_s=1.0,
        force_rate_clip_n_per_s=100.0,
        contact_age_clip_s=1.0,
        air_age_clip_s=2.0,
    )
    per_body, _ = unflatten_contact_obs_v1(observation, 2)

    assert torch.equal(
        per_body[0, :, CONTACT_OBS_V1_LAYOUT["force_rate_heading"]],
        torch.zeros(2, 3),
    )
    expected_rate = signed_log_compress(torch.tensor([32.0, -16.0, 8.0]), 1.0, 100.0)
    assert torch.allclose(
        per_body[1, 0, CONTACT_OBS_V1_LAYOUT["force_rate_heading"]],
        expected_rate,
    )
    assert torch.equal(
        _feature(per_body, "temporal_valid"),
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    )
    assert torch.equal(
        _feature(per_body, "contact_age"),
        torch.tensor([[0.5, 0.0], [1.0, 0.0]]),
    )
    assert torch.equal(
        _feature(per_body, "air_age"),
        torch.tensor([[0.375, 0.0], [1.0, 0.0]]),
    )
    expected_first_body = signed_log_compress(
        torch.tensor([10.0, -5.0, 2.0]), 100.0, 5000.0
    )
    expected_second_body = signed_log_compress(
        torch.tensor([1.0, 2.0, 3.0]), 100.0, 5000.0
    )
    assert torch.allclose(
        per_body[0, 0, CONTACT_OBS_V1_LAYOUT["net_force_heading"]],
        expected_first_body,
    )
    assert torch.allclose(
        per_body[0, 1, CONTACT_OBS_V1_LAYOUT["net_force_heading"]],
        expected_second_body,
    )


def test_log_compression_contract_and_parameter_validation():
    values = torch.tensor([-200.0, -100.0, -10.0, 0.0, 10.0, 100.0, 200.0])
    compressed = signed_log_compress(values, reference=10.0, clip=100.0)

    assert compressed[0].item() == -1.0
    assert compressed[1].item() == -1.0
    assert compressed[3].item() == 0.0
    assert compressed[5].item() == 1.0
    assert compressed[6].item() == 1.0
    assert torch.all(compressed[1:] >= compressed[:-1])
    assert torch.equal(
        unsigned_log_compress(torch.tensor([-5.0, 0.0]), reference=10.0, clip=100.0),
        torch.zeros(2),
    )
    assert unsigned_log_compress(
        torch.tensor(10.0), reference=10.0, clip=100.0
    ).item() == pytest.approx(math.log(2.0) / math.log(11.0))

    with pytest.raises(ValueError, match="greater than zero"):
        signed_log_compress(values, reference=0.0, clip=100.0)
    with pytest.raises(ValueError, match="greater than reference"):
        signed_log_compress(values, reference=10.0, clip=10.0)


def test_contact_obs_v1_sanitizes_nonfinite_samples_and_validates_inputs():
    inputs = _contact_inputs(num_envs=1, num_bodies=2)
    inputs["rigid_body_contact_forces"][0, 0] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")]
    )
    inputs["rigid_body_vel"][0, 0] = torch.tensor(
        [float("inf"), float("nan"), -float("inf")]
    )

    observation = compute_contact_obs_v1(**inputs)
    assert torch.isfinite(observation).all()

    with pytest.raises(ValueError, match="requires at least one selected body"):
        compute_contact_obs_v1(**inputs, body_ids=[])
    with pytest.raises(ValueError, match="body_ids must be in"):
        compute_contact_obs_v1(**inputs, body_ids=[2])
    with pytest.raises(ValueError, match="simulator backend"):
        compute_contact_obs_v1(**{**inputs, "rigid_body_contact_forces": None})
    with pytest.raises(ValueError, match="friction_mu"):
        compute_contact_obs_v1(**inputs, friction_mu=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_contact_obs_v1(**{**inputs, "dt": -0.1})


def test_update_contact_state_hysteresis_ages_raw_contact_and_no_mutation():
    raw_contacts = torch.tensor([[False, False, True, False]])
    forces = torch.zeros(1, 4, 3)
    forces[0, :, 0] = torch.tensor([6.0, 3.0, 0.0, 0.0])
    previous_active = torch.zeros(1, 4, dtype=torch.bool)
    previous_contact_age = torch.zeros(1, 4, dtype=torch.long)
    previous_air_age = torch.zeros(1, 4, dtype=torch.long)
    prior_copies = (
        previous_active.clone(),
        previous_contact_age.clone(),
        previous_air_age.clone(),
    )

    active, contact_age, air_age = update_contact_state(
        raw_contacts,
        forces,
        previous_active,
        previous_contact_age,
        previous_air_age,
        force_on_threshold_n=5.0,
        force_off_threshold_n=2.0,
    )

    assert torch.equal(active, torch.tensor([[True, False, True, False]]))
    assert torch.equal(contact_age, torch.tensor([[1, 0, 1, 0]]))
    assert torch.equal(air_age, torch.tensor([[0, 1, 0, 1]]))
    assert torch.equal(previous_active, prior_copies[0])
    assert torch.equal(previous_contact_age, prior_copies[1])
    assert torch.equal(previous_air_age, prior_copies[2])

    raw_contacts.fill_(False)
    forces.zero_()
    forces[0, :, 0] = torch.tensor([3.0, 6.0, 0.0, 0.0])
    active, contact_age, air_age = update_contact_state(
        raw_contacts,
        forces,
        active,
        contact_age,
        air_age,
        force_on_threshold_n=5.0,
        force_off_threshold_n=2.0,
    )

    assert torch.equal(active, torch.tensor([[True, True, False, False]]))
    assert torch.equal(contact_age, torch.tensor([[2, 1, 0, 0]]))
    assert torch.equal(air_age, torch.tensor([[0, 0, 1, 2]]))

    raw_contacts[0, 3] = True
    active, contact_age, air_age = update_contact_state(
        raw_contacts,
        torch.zeros_like(forces),
        active,
        contact_age,
        air_age,
    )
    assert torch.equal(active, torch.tensor([[False, False, False, True]]))
    assert torch.equal(contact_age, torch.tensor([[0, 0, 0, 1]]))
    assert torch.equal(air_age, torch.tensor([[1, 1, 2, 0]]))


@pytest.mark.parametrize(
    ("force_on", "force_off", "match"),
    [
        (1.0, 2.0, "greater than or equal"),
        (1.0, -1.0, "non-negative"),
        (float("inf"), 1.0, "finite"),
    ],
)
def test_update_contact_state_validates_thresholds(
    force_on: float, force_off: float, match: str
):
    with pytest.raises(ValueError, match=match):
        update_contact_state(
            raw_contacts=torch.zeros(1, 1, dtype=torch.bool),
            contact_forces=torch.zeros(1, 1, 3),
            previous_active=torch.zeros(1, 1, dtype=torch.bool),
            previous_contact_age_steps=torch.zeros(1, 1, dtype=torch.long),
            previous_air_age_steps=torch.zeros(1, 1, dtype=torch.long),
            force_on_threshold_n=force_on,
            force_off_threshold_n=force_off,
        )
