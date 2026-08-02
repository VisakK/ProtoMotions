# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure tensor tests for the parallel IsaacLab contact-observation contract."""

import math

import pytest
import torch

from protomotions.envs.obs.contact_isaaclab import (
    ISAACLAB_CONTACT_OBS_V1_GLOBAL_LAYOUT,
    ISAACLAB_CONTACT_OBS_V1_LAYOUT,
    ISAACLAB_CONTACT_OBS_V1_PER_BODY_DIM,
    ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT,
    ISAACLAB_CONTACT_PAIR_OBS_V1_PER_PAIR_DIM,
    compute_isaaclab_contact_obs_v1,
    compute_isaaclab_contact_pair_obs_v1,
    isaaclab_contact_obs_v1_dim,
    isaaclab_contact_pair_obs_v1_dim,
    unflatten_isaaclab_contact_obs_v1,
    unflatten_isaaclab_contact_pair_obs_v1,
    update_isaaclab_contact_state_v1,
)
from protomotions.utils import rotations


def _identity_quat(num_envs: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    root_rot = torch.zeros(num_envs, 4, dtype=dtype)
    root_rot[:, 3] = 1.0
    return root_rot


def _aggregate_inputs(
    num_envs: int = 2,
    num_bodies: int = 3,
    history_length: int = 4,
    dtype: torch.dtype = torch.float32,
):
    return {
        "root_rot": _identity_quat(num_envs, dtype),
        "rigid_body_vel": torch.zeros(num_envs, num_bodies, 3, dtype=dtype),
        "normal_force_w": torch.zeros(num_envs, num_bodies, 3, dtype=dtype),
        "normal_force_history_w": torch.zeros(
            num_envs, history_length, num_bodies, 3, dtype=dtype
        ),
        "previous_normal_force_w": torch.zeros(num_envs, num_bodies, 3, dtype=dtype),
        "previous_active": torch.zeros(num_envs, num_bodies, dtype=torch.bool),
        "previous_contact_age_s": torch.zeros(num_envs, num_bodies, dtype=dtype),
        "previous_air_age_s": torch.zeros(num_envs, num_bodies, dtype=dtype),
        "temporal_valid": torch.zeros(num_envs, dtype=torch.bool),
        "body_weight_n": torch.full((num_envs, 1), 10.0, dtype=dtype),
        "dt": 0.1,
    }


def _pair_inputs(
    num_envs: int = 2,
    num_bodies: int = 3,
    num_filters: int = 2,
    dtype: torch.dtype = torch.float32,
):
    pair_shape = (num_envs, num_bodies, num_filters)
    return {
        "root_rot": _identity_quat(num_envs, dtype),
        "rigid_body_pos": torch.zeros(num_envs, num_bodies, 3, dtype=dtype),
        "filtered_normal_force_w": torch.zeros(*pair_shape, 3, dtype=dtype),
        "friction_force_w": torch.zeros(*pair_shape, 3, dtype=dtype),
        "mean_contact_point_w": torch.full((*pair_shape, 3), float("nan"), dtype=dtype),
        "pair_slot_valid": torch.ones(*pair_shape, dtype=torch.bool),
        "mean_contact_point_valid": torch.zeros(*pair_shape, dtype=torch.bool),
        "body_weight_n": torch.full((num_envs, 1), 10.0, dtype=dtype),
    }


def _body_feature(per_body: torch.Tensor, name: str) -> torch.Tensor:
    return per_body[..., ISAACLAB_CONTACT_OBS_V1_LAYOUT[name]].squeeze(-1)


def _global_feature(global_obs: torch.Tensor, name: str) -> torch.Tensor:
    return global_obs[..., ISAACLAB_CONTACT_OBS_V1_GLOBAL_LAYOUT[name]].squeeze(-1)


def _pair_feature(per_pair: torch.Tensor, name: str) -> torch.Tensor:
    return per_pair[..., ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT[name]].squeeze(-1)


def _unsigned_log(value: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    return torch.log1p(torch.clamp(value, 0.0, clip)) / math.log1p(clip)


def _signed_log(value: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    value = torch.clamp(value, -clip, clip)
    return torch.sign(value) * torch.log1p(torch.abs(value)) / math.log1p(clip)


def test_layout_dimensions_are_exact_immutable_and_unflattened_body_major():
    assert ISAACLAB_CONTACT_OBS_V1_PER_BODY_DIM == 20
    assert ISAACLAB_CONTACT_PAIR_OBS_V1_PER_PAIR_DIM == 15
    assert isaaclab_contact_obs_v1_dim(3) == 66
    assert isaaclab_contact_pair_obs_v1_dim(3, 2) == 90

    aggregate = torch.arange(2 * 66, dtype=torch.float32).reshape(2, 66)
    per_body, global_obs = unflatten_isaaclab_contact_obs_v1(aggregate, 3)
    assert per_body.shape == (2, 3, 20)
    assert global_obs.shape == (2, 6)
    assert torch.equal(per_body[0, 1], aggregate[0, 20:40])

    pairs = torch.arange(90, dtype=torch.float32).reshape(1, 90)
    per_pair = unflatten_isaaclab_contact_pair_obs_v1(pairs, 3, 2)
    assert per_pair.shape == (1, 3, 2, 15)
    assert torch.equal(per_pair[0, 1, 0], pairs[0, 30:45])
    assert torch.equal(per_pair[0, 1, 1], pairs[0, 45:60])

    with pytest.raises(TypeError):
        ISAACLAB_CONTACT_OBS_V1_LAYOUT["new"] = slice(0, 1)
    with pytest.raises(ValueError, match="positive integer"):
        isaaclab_contact_obs_v1_dim(0)
    with pytest.raises(ValueError, match="positive integer"):
        isaaclab_contact_pair_obs_v1_dim(1, 0)
    with pytest.raises(ValueError, match="must have shape"):
        unflatten_isaaclab_contact_obs_v1(torch.zeros(2, 65), 3)


def test_zero_contact_outputs_are_exactly_zero_finite_and_dtype_preserving():
    aggregate_inputs = _aggregate_inputs(dtype=torch.float64)
    observation = compute_isaaclab_contact_obs_v1(**aggregate_inputs)
    assert observation.shape == (2, 66)
    assert observation.dtype == torch.float64
    assert torch.equal(observation, torch.zeros_like(observation))
    assert torch.isfinite(observation).all()

    pair_inputs = _pair_inputs(dtype=torch.float64)
    pair_observation = compute_isaaclab_contact_pair_obs_v1(**pair_inputs)
    per_pair = unflatten_isaaclab_contact_pair_obs_v1(pair_observation, 3, 2)
    assert pair_observation.dtype == torch.float64
    assert torch.isfinite(pair_observation).all()
    assert torch.equal(_pair_feature(per_pair, "pair_active"), torch.zeros(2, 3, 2))
    assert torch.equal(
        _pair_feature(per_pair, "tangential_to_normal_ratio"),
        torch.zeros(2, 3, 2),
    )
    assert torch.equal(
        per_pair[
            ...,
            ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT["mean_contact_point_rel_body_heading"],
        ],
        torch.zeros(2, 3, 2, 3, dtype=torch.float64),
    )
    assert torch.equal(
        _pair_feature(per_pair, "mean_contact_point_valid"),
        torch.zeros(2, 3, 2, dtype=torch.float64),
    )


def test_force_only_hysteresis_events_ages_and_first_reset_semantics():
    previous_active = torch.zeros(1, 1, dtype=torch.bool)
    contact_age = torch.full((1, 1), 99.0)
    air_age = torch.full((1, 1), 99.0)

    def step(force_n, valid, prior_active, prior_contact_age, prior_air_age):
        force = torch.tensor([[[force_n, 0.0, 0.0]]])
        return update_isaaclab_contact_state_v1(
            force,
            prior_active,
            prior_contact_age,
            prior_air_age,
            torch.tensor([valid]),
            dt=0.1,
            contact_on_threshold_n=5.0,
            contact_off_threshold_n=2.0,
        )

    active, contact_age, air_age, onset, release = step(
        0.0, False, previous_active, contact_age, air_age
    )
    assert not active.item()
    assert contact_age.item() == 0.0
    assert air_age.item() == 0.0
    assert not onset.item() and not release.item()

    active, contact_age, air_age, onset, release = step(
        6.0, True, active, contact_age, air_age
    )
    assert active.item() and onset.item() and not release.item()
    assert contact_age.item() == pytest.approx(0.1)
    assert air_age.item() == 0.0

    active, contact_age, air_age, onset, release = step(
        3.0, True, active, contact_age, air_age
    )
    assert active.item() and not onset.item() and not release.item()
    assert contact_age.item() == pytest.approx(0.2)

    active, contact_age, air_age, onset, release = step(
        1.0, True, active, contact_age, air_age
    )
    assert not active.item() and not onset.item() and release.item()
    assert contact_age.item() == 0.0
    assert air_age.item() == pytest.approx(0.1)


def test_current_contact_classifies_on_reset_but_suppresses_age_delta_and_event():
    inputs = _aggregate_inputs(num_envs=1, num_bodies=1)
    inputs["normal_force_w"][0, 0, 2] = 8.0
    inputs["previous_normal_force_w"][0, 0, 0] = 100.0

    observation = compute_isaaclab_contact_obs_v1(**inputs)
    per_body, _ = unflatten_isaaclab_contact_obs_v1(observation, 1)

    assert _body_feature(per_body, "active_contact").item() == 1.0
    assert _body_feature(per_body, "temporal_valid").item() == 0.0
    assert _body_feature(per_body, "contact_age").item() == 0.0
    assert _body_feature(per_body, "air_age").item() == 0.0
    assert _body_feature(per_body, "contact_onset").item() == 0.0
    assert _body_feature(per_body, "contact_release").item() == 0.0
    assert torch.equal(
        per_body[0, 0, ISAACLAB_CONTACT_OBS_V1_LAYOUT["normal_force_delta_heading"]],
        torch.zeros(3),
    )


def test_exact_aggregate_features_substep_statistics_bodyweight_and_order():
    inputs = _aggregate_inputs(num_envs=1, num_bodies=2, history_length=4)
    inputs["normal_force_w"][0, 0] = torch.tensor([6.0, 0.0, 8.0])
    inputs["normal_force_w"][0, 1] = torch.tensor([0.0, 0.0, 2.0])
    inputs["previous_normal_force_w"][0, 0] = torch.tensor([1.0, 0.0, 0.0])
    inputs["normal_force_history_w"][0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    inputs["normal_force_history_w"][0, :, 1, 2] = 2.0
    inputs["rigid_body_vel"][0, 0] = torch.tensor([3.0, 4.0, -2.5])
    inputs["previous_active"][0, 1] = True
    inputs["previous_contact_age_s"][0, 1] = 0.3
    inputs["temporal_valid"][:] = True

    observation = compute_isaaclab_contact_obs_v1(
        **inputs,
        velocity_clip_mps=5.0,
        duration_clip_s=2.0,
    )
    per_body, global_obs = unflatten_isaaclab_contact_obs_v1(observation, 2)

    assert torch.allclose(
        per_body[0, 0, ISAACLAB_CONTACT_OBS_V1_LAYOUT["normal_force_heading"]],
        _signed_log(torch.tensor([0.6, 0.0, 0.8])),
    )
    assert torch.allclose(
        per_body[0, 0, ISAACLAB_CONTACT_OBS_V1_LAYOUT["normal_force_delta_heading"]],
        _signed_log(torch.tensor([0.5, 0.0, 0.8])),
    )
    assert _body_feature(per_body, "normal_force_norm")[0, 0].item() == pytest.approx(
        _unsigned_log(torch.tensor(1.0)).item()
    )
    assert _body_feature(per_body, "substep_mean_force_norm")[
        0, 0
    ].item() == pytest.approx(_unsigned_log(torch.tensor(0.25)).item())
    assert _body_feature(per_body, "substep_peak_force_norm")[
        0, 0
    ].item() == pytest.approx(_unsigned_log(torch.tensor(0.4)).item())
    assert _body_feature(per_body, "substep_force_std")[0, 0].item() == pytest.approx(
        _unsigned_log(torch.tensor(math.sqrt(1.25) / 10.0)).item()
    )
    assert _body_feature(per_body, "support_load_fraction")[
        0, 0
    ].item() == pytest.approx(8.0 / (10.0 + 1.0e-6))
    assert _body_feature(per_body, "body_vertical_velocity")[0, 0].item() == -0.5
    assert _body_feature(per_body, "body_planar_speed")[0, 0].item() == 1.0
    assert _body_feature(per_body, "contact_onset")[0, 0].item() == 1.0
    assert _body_feature(per_body, "active_contact")[0, 1].item() == 1.0
    assert _body_feature(per_body, "contact_age")[0, 1].item() == pytest.approx(0.2)
    assert _global_feature(global_obs, "net_normal_force_norm").item() == pytest.approx(
        _unsigned_log(torch.tensor(math.sqrt(6.0**2 + 10.0**2) / 10.0)).item()
    )


@pytest.mark.parametrize(
    "upward, expected_entropy",
    [
        ([0.0, 0.0], 0.0),
        ([10.0, 0.0], 0.0),
        ([10.0, 10.0], 1.0),
    ],
)
def test_support_load_entropy(upward, expected_entropy):
    inputs = _aggregate_inputs(num_envs=1, num_bodies=2)
    inputs["normal_force_w"][0, :, 2] = torch.tensor(upward)
    observation = compute_isaaclab_contact_obs_v1(**inputs)
    _, global_obs = unflatten_isaaclab_contact_obs_v1(observation, 2)
    assert _global_feature(global_obs, "support_load_entropy").item() == pytest.approx(
        expected_entropy, abs=3.0e-6
    )


def test_partial_reset_isolation_for_state_delta_and_events():
    inputs = _aggregate_inputs(num_envs=2, num_bodies=1)
    inputs["normal_force_w"][:, 0, 0] = 7.0
    inputs["previous_normal_force_w"][:, 0, 0] = 3.0
    inputs["previous_active"][:] = True
    inputs["previous_contact_age_s"][:] = 0.4
    inputs["temporal_valid"][:] = torch.tensor([False, True])

    observation = compute_isaaclab_contact_obs_v1(**inputs)
    per_body, _ = unflatten_isaaclab_contact_obs_v1(observation, 1)
    delta = per_body[..., ISAACLAB_CONTACT_OBS_V1_LAYOUT["normal_force_delta_heading"]]
    assert torch.equal(delta[0], torch.zeros(1, 3))
    assert torch.any(delta[1] != 0.0)
    assert _body_feature(per_body, "contact_age")[0, 0].item() == 0.0
    assert _body_feature(per_body, "contact_age")[1, 0].item() == pytest.approx(0.25)
    assert not _body_feature(per_body, "contact_onset").any()
    assert not _body_feature(per_body, "contact_release").any()


def test_heading_invariance_for_aggregate_vectors_and_velocity_proxies():
    inputs = _aggregate_inputs(num_envs=2, num_bodies=2)
    inputs["normal_force_w"] = torch.tensor(
        [[[3.0, 4.0, 5.0], [-2.0, 7.0, 1.0]], [[1.0, -4.0, 2.0], [6.0, 3.0, 8.0]]]
    )
    inputs["normal_force_history_w"] = (
        inputs["normal_force_w"][:, None].expand(-1, 4, -1, -1).clone()
    )
    inputs["previous_normal_force_w"] = inputs["normal_force_w"] * 0.25
    inputs["rigid_body_vel"] = torch.tensor(
        [[[1.0, 2.0, 3.0], [-2.0, 1.0, -1.0]], [[4.0, -3.0, 1.0], [2.0, 5.0, -2.0]]]
    )
    inputs["temporal_valid"][:] = True
    baseline = compute_isaaclab_contact_obs_v1(**inputs)

    yaw = torch.tensor([0.7, -1.1])
    yaw_quat = rotations.quat_from_angle_axis(
        yaw, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), w_last=True
    )
    body_quat = yaw_quat[:, None, :].expand(-1, 2, -1)
    history_quat = yaw_quat[:, None, None, :].expand(-1, 4, 2, -1)
    rotated = dict(inputs)
    rotated["root_rot"] = yaw_quat
    rotated["normal_force_w"] = rotations.quat_rotate(
        body_quat, inputs["normal_force_w"], w_last=True
    )
    rotated["previous_normal_force_w"] = rotations.quat_rotate(
        body_quat, inputs["previous_normal_force_w"], w_last=True
    )
    rotated["normal_force_history_w"] = rotations.quat_rotate(
        history_quat, inputs["normal_force_history_w"], w_last=True
    )
    rotated["rigid_body_vel"] = rotations.quat_rotate(
        body_quat, inputs["rigid_body_vel"], w_last=True
    )
    actual = compute_isaaclab_contact_obs_v1(**rotated)
    assert torch.allclose(actual, baseline, atol=1.0e-6, rtol=1.0e-6)


def test_pair_layout_uses_true_friction_ratio_relative_point_and_slot_masks():
    inputs = _pair_inputs(num_envs=1, num_bodies=2, num_filters=2)
    inputs["rigid_body_pos"][0, 0] = torch.tensor([1.0, 2.0, 3.0])
    inputs["filtered_normal_force_w"][0, 0, 0] = torch.tensor([6.0, 0.0, 8.0])
    inputs["friction_force_w"][0, 0, 0] = torch.tensor([0.0, 3.0, 4.0])
    inputs["mean_contact_point_w"][0, 0, 0] = torch.tensor([2.0, 4.0, 6.0])
    inputs["mean_contact_point_valid"][0, 0, 0] = True
    # An invalid slot must not leak otherwise nonzero simulator data.
    inputs["pair_slot_valid"][0, 0, 1] = False
    inputs["filtered_normal_force_w"][0, 0, 1] = 1000.0
    inputs["friction_force_w"][0, 0, 1] = 1000.0

    observation = compute_isaaclab_contact_pair_obs_v1(
        **inputs, contact_point_scale_m=2.0
    )
    pair = unflatten_isaaclab_contact_pair_obs_v1(observation, 2, 2)

    assert _pair_feature(pair, "pair_active")[0, 0, 0].item() == 1.0
    assert torch.allclose(
        pair[
            0,
            0,
            0,
            ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT["filtered_normal_force_heading"],
        ],
        _signed_log(torch.tensor([0.6, 0.0, 0.8])),
    )
    assert _pair_feature(pair, "filtered_normal_force_norm")[
        0, 0, 0
    ].item() == pytest.approx(_unsigned_log(torch.tensor(1.0)).item())
    assert _pair_feature(pair, "friction_force_norm")[0, 0, 0].item() == pytest.approx(
        _unsigned_log(torch.tensor(0.5)).item()
    )
    assert _pair_feature(pair, "tangential_to_normal_ratio")[
        0, 0, 0
    ].item() == pytest.approx((5.0 / (10.0 + 1.0e-4)) / 2.0)
    assert torch.equal(
        pair[
            0,
            0,
            0,
            ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT["mean_contact_point_rel_body_heading"],
        ],
        torch.tensor([0.5, 1.0, 1.0]),
    )
    assert _pair_feature(pair, "mean_contact_point_valid")[0, 0, 0].item() == 1.0
    assert torch.equal(pair[0, 0, 1], torch.zeros(15))
    # A valid, inactive slot stays semantically valid.
    assert _pair_feature(pair, "pair_slot_valid")[0, 1, 0].item() == 1.0
    assert _pair_feature(pair, "pair_active")[0, 1, 0].item() == 0.0


def test_horizontal_wall_normal_does_not_create_friction_and_nan_point_is_safe():
    inputs = _pair_inputs(num_envs=1, num_bodies=1, num_filters=1)
    inputs["filtered_normal_force_w"][0, 0, 0] = torch.tensor([10.0, 0.0, 0.0])
    inputs["friction_force_w"][0, 0, 0] = 0.0
    inputs["mean_contact_point_w"][0, 0, 0] = float("nan")
    inputs["mean_contact_point_valid"][0, 0, 0] = True

    observation = compute_isaaclab_contact_pair_obs_v1(**inputs)
    pair = unflatten_isaaclab_contact_pair_obs_v1(observation, 1, 1)
    assert torch.isfinite(pair).all()
    assert torch.any(
        pair[
            0,
            0,
            0,
            ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT["filtered_normal_force_heading"],
        ]
        != 0.0
    )
    assert torch.equal(
        pair[0, 0, 0, ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT["friction_force_heading"]],
        torch.zeros(3),
    )
    assert _pair_feature(pair, "friction_force_norm").item() == 0.0
    assert _pair_feature(pair, "tangential_to_normal_ratio").item() == 0.0
    assert _pair_feature(pair, "mean_contact_point_valid").item() == 0.0


def test_measurement_validity_nonfinite_sanitization_and_bodyweight_fallback():
    inputs = _aggregate_inputs(num_envs=1, num_bodies=2)
    # Finite but stale samples must also be suppressed by lifecycle validity.
    inputs["normal_force_w"][0, 0] = torch.tensor([10.0, 20.0, 30.0])
    inputs["normal_force_w"][0, 1, 2] = 10.0
    inputs["normal_force_history_w"][0, :, 1, 2] = float("inf")
    inputs["normal_force_valid"] = torch.tensor([[True, False]])
    inputs["normal_force_history_valid"] = torch.ones(1, 4, 2, dtype=torch.bool)
    inputs["body_weight_n"][:] = float("nan")
    inputs["sensor_data_valid"] = torch.tensor([False])

    observation = compute_isaaclab_contact_obs_v1(**inputs)
    assert torch.isfinite(observation).all()
    assert torch.equal(observation, torch.zeros_like(observation))

    pair_inputs = _pair_inputs(num_envs=1, num_bodies=1, num_filters=1)
    pair_inputs["filtered_normal_force_w"][:] = float("inf")
    pair_inputs["friction_force_w"][:] = float("nan")
    pair_inputs["filtered_normal_force_valid"] = torch.tensor([[[False]]])
    pair_inputs["friction_force_valid"] = torch.tensor([[[False]]])
    pair_inputs["sensor_data_valid"] = torch.tensor([False])
    pair_observation = compute_isaaclab_contact_pair_obs_v1(**pair_inputs)
    assert torch.isfinite(pair_observation).all()
    pair = unflatten_isaaclab_contact_pair_obs_v1(pair_observation, 1, 1)
    assert _pair_feature(pair, "pair_slot_valid").item() == 1.0
    assert torch.equal(pair[..., 2:], torch.zeros_like(pair[..., 2:]))


def test_body_ids_preserve_explicit_kinematic_selection_order():
    inputs = _aggregate_inputs(num_envs=1, num_bodies=2)
    inputs["rigid_body_vel"] = torch.tensor(
        [[[0.0, 0.0, 5.0], [0.0, 0.0, 1.0], [3.0, 4.0, 0.0]]]
    )
    observation = compute_isaaclab_contact_obs_v1(
        **inputs, body_ids=[2, 0], velocity_clip_mps=5.0
    )
    per_body, _ = unflatten_isaaclab_contact_obs_v1(observation, 2)
    assert _body_feature(per_body, "body_planar_speed")[0, 0].item() == 1.0
    assert _body_feature(per_body, "body_vertical_velocity")[0, 1].item() == 1.0


def test_strict_validation_rejects_ambiguous_shapes_dtypes_and_configuration():
    inputs = _aggregate_inputs(num_envs=1, num_bodies=2)
    with pytest.raises(ValueError, match=r"\[E, H, K, 3\]"):
        compute_isaaclab_contact_obs_v1(
            **{**inputs, "normal_force_history_w": torch.zeros(1, 2, 3)}
        )
    with pytest.raises(TypeError, match="torch.bool"):
        compute_isaaclab_contact_obs_v1(**{**inputs, "temporal_valid": torch.zeros(1)})
    with pytest.raises(ValueError, match="body_ids is required"):
        compute_isaaclab_contact_obs_v1(
            **{**inputs, "rigid_body_vel": torch.zeros(1, 3, 3)}
        )
    with pytest.raises(ValueError, match="greater than or equal"):
        compute_isaaclab_contact_obs_v1(
            **inputs,
            contact_on_threshold_n=1.0,
            contact_off_threshold_n=2.0,
        )
    pair_inputs = _pair_inputs(num_envs=1, num_bodies=1, num_filters=1)
    with pytest.raises(ValueError, match="contact_point_scale_m"):
        compute_isaaclab_contact_pair_obs_v1(**pair_inputs, contact_point_scale_m=0.0)


def test_kernels_execute_through_torch_compile_eager_backend():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")

    aggregate_inputs = _aggregate_inputs(num_envs=1, num_bodies=2)
    compiled_aggregate = torch.compile(
        compute_isaaclab_contact_obs_v1, backend="eager", fullgraph=True
    )
    expected = compute_isaaclab_contact_obs_v1(**aggregate_inputs)
    actual = compiled_aggregate(**aggregate_inputs)
    assert torch.equal(actual, expected)

    pair_inputs = _pair_inputs(num_envs=1, num_bodies=2, num_filters=1)
    compiled_pair = torch.compile(
        compute_isaaclab_contact_pair_obs_v1, backend="eager", fullgraph=True
    )
    expected_pair = compute_isaaclab_contact_pair_obs_v1(**pair_inputs)
    actual_pair = compiled_pair(**pair_inputs)
    assert torch.equal(actual_pair, expected_pair)
