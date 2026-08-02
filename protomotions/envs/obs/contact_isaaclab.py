# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab-specific, physics-informed contact observation kernels.

This module intentionally lives beside, rather than replacing, ``contact.py``.
The common-backend ``contact_obs_v1`` contract is ``17*K + 4``; the contracts
defined here use IsaacLab contact-sensor history and optional filtered pair
data and therefore have different names and dimensions.

IsaacLab's aggregate and filtered ``normal_force_w`` tensors contain normal
contact forces.  Friction is consumed only from the separate filtered
``friction_force_w`` tensor.  ``mean_contact_point_w`` is an average contact
position reported by the sensor, not a center of pressure.

All functions are pure, batch-vectorized tensor operations.  They do not
mutate temporal state or sensor tensors, and contain no Python loop over
environments, bodies, filters, or history samples.
"""

from __future__ import annotations

import math
from numbers import Integral
from types import MappingProxyType
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from protomotions.utils import rotations


ISAACLAB_CONTACT_OBS_V1_PER_BODY_DIM = 20
ISAACLAB_CONTACT_OBS_V1_GLOBAL_DIM = 6
ISAACLAB_CONTACT_PAIR_OBS_V1_PER_PAIR_DIM = 15

ISAACLAB_CONTACT_OBS_V1_LAYOUT: Mapping[str, slice] = MappingProxyType(
    {
        "active_contact": slice(0, 1),
        "temporal_valid": slice(1, 2),
        "normal_force_heading": slice(2, 5),
        "normal_force_delta_heading": slice(5, 8),
        "normal_force_norm": slice(8, 9),
        "substep_mean_force_norm": slice(9, 10),
        "substep_peak_force_norm": slice(10, 11),
        "substep_force_std": slice(11, 12),
        "upward_normal_support": slice(12, 13),
        "support_load_fraction": slice(13, 14),
        "body_vertical_velocity": slice(14, 15),
        "body_planar_speed": slice(15, 16),
        "contact_age": slice(16, 17),
        "air_age": slice(17, 18),
        "contact_onset": slice(18, 19),
        "contact_release": slice(19, 20),
    }
)

ISAACLAB_CONTACT_OBS_V1_GLOBAL_LAYOUT: Mapping[str, slice] = MappingProxyType(
    {
        "any_active_contact": slice(0, 1),
        "active_contact_fraction": slice(1, 2),
        "total_upward_normal_support": slice(2, 3),
        "net_normal_force_norm": slice(3, 4),
        "max_substep_peak_force": slice(4, 5),
        "support_load_entropy": slice(5, 6),
    }
)

ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT: Mapping[str, slice] = MappingProxyType(
    {
        "pair_active": slice(0, 1),
        "pair_slot_valid": slice(1, 2),
        "filtered_normal_force_heading": slice(2, 5),
        "filtered_normal_force_norm": slice(5, 6),
        "friction_force_heading": slice(6, 9),
        "friction_force_norm": slice(9, 10),
        "tangential_to_normal_ratio": slice(10, 11),
        "mean_contact_point_rel_body_heading": slice(11, 14),
        "mean_contact_point_valid": slice(14, 15),
    }
)

ISAACLAB_CONTACT_DEFAULT_ON_THRESHOLD_N = 5.0
ISAACLAB_CONTACT_DEFAULT_OFF_THRESHOLD_N = 2.0

BodyIds = Optional[Union[Sequence[int], Tensor]]


def _validate_finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite scalar, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _validate_positive_scalar(name: str, value: float) -> None:
    _validate_finite_scalar(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {value}")


def _validate_thresholds(on_threshold_n: float, off_threshold_n: float) -> None:
    _validate_finite_scalar("contact_on_threshold_n", on_threshold_n)
    _validate_finite_scalar("contact_off_threshold_n", off_threshold_n)
    if off_threshold_n < 0.0:
        raise ValueError(
            "contact_off_threshold_n must be non-negative, " f"got {off_threshold_n}"
        )
    if on_threshold_n < off_threshold_n:
        raise ValueError(
            "contact_on_threshold_n must be greater than or equal to "
            "contact_off_threshold_n, got "
            f"on={on_threshold_n}, off={off_threshold_n}"
        )


def _positive_integer(name: str, value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def isaaclab_contact_obs_v1_dim(num_contact_bodies: int) -> int:
    """Return the flattened ``isaaclab_contact_obs_v1`` dimension."""

    num_bodies = _positive_integer("num_contact_bodies", num_contact_bodies)
    return (
        num_bodies * ISAACLAB_CONTACT_OBS_V1_PER_BODY_DIM
        + ISAACLAB_CONTACT_OBS_V1_GLOBAL_DIM
    )


def isaaclab_contact_pair_obs_v1_dim(num_contact_bodies: int, num_filters: int) -> int:
    """Return the flattened ``isaaclab_contact_pair_obs_v1`` dimension."""

    num_bodies = _positive_integer("num_contact_bodies", num_contact_bodies)
    filters = _positive_integer("num_filters", num_filters)
    return num_bodies * filters * ISAACLAB_CONTACT_PAIR_OBS_V1_PER_PAIR_DIM


def unflatten_isaaclab_contact_obs_v1(
    observation: Tensor, num_contact_bodies: int
) -> Tuple[Tensor, Tensor]:
    """Split ``[E, 20*K+6]`` into ``[E,K,20]`` and ``[E,6]`` views."""

    expected_dim = isaaclab_contact_obs_v1_dim(num_contact_bodies)
    if not isinstance(observation, Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if observation.ndim != 2 or observation.shape[-1] != expected_dim:
        raise ValueError(
            "isaaclab_contact_obs_v1 must have shape "
            f"[E, {expected_dim}] for K={num_contact_bodies}, "
            f"got {tuple(observation.shape)}"
        )
    body_dim = int(num_contact_bodies) * ISAACLAB_CONTACT_OBS_V1_PER_BODY_DIM
    return (
        observation[:, :body_dim].reshape(
            observation.shape[0],
            int(num_contact_bodies),
            ISAACLAB_CONTACT_OBS_V1_PER_BODY_DIM,
        ),
        observation[:, body_dim:],
    )


def unflatten_isaaclab_contact_pair_obs_v1(
    observation: Tensor, num_contact_bodies: int, num_filters: int
) -> Tensor:
    """View body-major/filter-major pair data as ``[E,K,F,15]``."""

    expected_dim = isaaclab_contact_pair_obs_v1_dim(num_contact_bodies, num_filters)
    if not isinstance(observation, Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if observation.ndim != 2 or observation.shape[-1] != expected_dim:
        raise ValueError(
            "isaaclab_contact_pair_obs_v1 must have shape "
            f"[E, {expected_dim}] for K={num_contact_bodies}, F={num_filters}, "
            f"got {tuple(observation.shape)}"
        )
    return observation.reshape(
        observation.shape[0],
        int(num_contact_bodies),
        int(num_filters),
        ISAACLAB_CONTACT_PAIR_OBS_V1_PER_PAIR_DIM,
    )


def _validate_float_tensor(
    name: str,
    tensor: Tensor,
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must have a floating dtype")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _validate_mask_tensor(
    name: str, tensor: Tensor, shape: Tuple[int, ...], *, device: torch.device
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {tensor.dtype}")


def _sanitize_float(tensor: Tensor) -> Tensor:
    """Replace invalid simulator samples without mutating the source tensor."""

    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))


def _body_weight_view(
    body_weight_n: Tensor,
    num_envs: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fallback_force_reference_n: float,
    force_epsilon_n: float,
) -> Tensor:
    if not isinstance(body_weight_n, Tensor):
        raise TypeError("body_weight_n must be a torch.Tensor")
    if tuple(body_weight_n.shape) not in ((num_envs,), (num_envs, 1)):
        raise ValueError(
            "body_weight_n must have shape [E] or [E, 1], "
            f"got {tuple(body_weight_n.shape)}"
        )
    if body_weight_n.device != device:
        raise ValueError(
            f"body_weight_n must be on device {device}, got {body_weight_n.device}"
        )
    if not torch.is_floating_point(body_weight_n):
        raise TypeError("body_weight_n must have a floating dtype")
    if body_weight_n.dtype != dtype:
        raise TypeError(
            f"body_weight_n must have dtype {dtype}, got {body_weight_n.dtype}"
        )

    flat_weight = body_weight_n.reshape(num_envs)
    valid_weight = torch.isfinite(flat_weight) & (flat_weight > force_epsilon_n)
    fallback = torch.full_like(flat_weight, fallback_force_reference_n)
    safe_weight = torch.where(valid_weight, flat_weight, fallback)
    return safe_weight[:, None, None]


def _resolve_body_ids(
    body_ids: BodyIds,
    *,
    num_input_bodies: int,
    num_selected_bodies: int,
    device: torch.device,
) -> Optional[Tensor]:
    if body_ids is None:
        if num_input_bodies != num_selected_bodies:
            raise ValueError(
                "body_ids is required when rigid-body state has B != K; "
                f"got B={num_input_bodies}, K={num_selected_bodies}"
            )
        return None

    if isinstance(body_ids, Tensor):
        ids = body_ids
        if ids.ndim != 1:
            raise ValueError(
                f"body_ids must be one-dimensional, got {tuple(ids.shape)}"
            )
        if ids.dtype != torch.long:
            raise TypeError(f"body_ids must have dtype torch.long, got {ids.dtype}")
        if ids.device != device:
            raise ValueError(f"body_ids must be on device {device}, got {ids.device}")
    else:
        entries: List[int] = list(body_ids)
        if any(
            not isinstance(body_id, Integral) or isinstance(body_id, bool)
            for body_id in entries
        ):
            raise TypeError("every body_ids entry must be an integer")
        ids = torch.tensor(entries, dtype=torch.long, device=device)

    if ids.numel() != num_selected_bodies:
        raise ValueError(
            f"body_ids must contain exactly K={num_selected_bodies} entries, "
            f"got {ids.numel()}"
        )
    if ids.numel() == 0:
        raise ValueError("at least one contact body is required")
    # Bounds are deliberately checked without converting device data to Python;
    # index_select will raise for an out-of-bounds dynamic tensor.
    if not isinstance(body_ids, Tensor):
        if min(entries) < 0 or max(entries) >= num_input_bodies:
            raise ValueError(
                f"body_ids must be in [0, {num_input_bodies}), got {entries}"
            )
    return ids


def _select_body_state(
    name: str,
    state: Tensor,
    *,
    num_envs: int,
    num_selected_bodies: int,
    device: torch.device,
    dtype: torch.dtype,
    body_ids: BodyIds,
) -> Tensor:
    if not isinstance(state, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if state.ndim != 3 or state.shape[0] != num_envs or state.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [E, B, 3], got {tuple(state.shape)}")
    if state.device != device:
        raise ValueError(f"{name} must be on device {device}, got {state.device}")
    if not torch.is_floating_point(state):
        raise TypeError(f"{name} must have a floating dtype")
    if state.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {state.dtype}")
    ids = _resolve_body_ids(
        body_ids,
        num_input_bodies=state.shape[1],
        num_selected_bodies=num_selected_bodies,
        device=device,
    )
    return state if ids is None else state.index_select(1, ids)


def _signed_log_compress_bodyweights(value: Tensor, clip: float) -> Tensor:
    finite = _sanitize_float(value)
    clipped = torch.clamp(finite, min=-clip, max=clip)
    return torch.sign(clipped) * torch.log1p(torch.abs(clipped)) / math.log1p(clip)


def _unsigned_log_compress_bodyweights(value: Tensor, clip: float) -> Tensor:
    finite = _sanitize_float(value)
    clipped = torch.clamp(finite, min=0.0, max=clip)
    return torch.log1p(clipped) / math.log1p(clip)


def update_isaaclab_contact_state_v1(
    normal_force_w: Tensor,
    previous_active: Tensor,
    previous_contact_age_s: Tensor,
    previous_air_age_s: Tensor,
    temporal_valid: Tensor,
    dt: float,
    contact_on_threshold_n: float = ISAACLAB_CONTACT_DEFAULT_ON_THRESHOLD_N,
    contact_off_threshold_n: float = ISAACLAB_CONTACT_DEFAULT_OFF_THRESHOLD_N,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute force-only hysteresis, ages, and one-step transition pulses.

    The first post-reset sample is identified by ``temporal_valid=False``.
    Current contact is still classified on that sample, but ages and events are
    exactly zero.  The caller owns state storage and commits the returned active
    state and ages only after the current observation/reward evaluation.
    """

    if not isinstance(normal_force_w, Tensor):
        raise TypeError("normal_force_w must be a torch.Tensor")
    if normal_force_w.ndim != 3 or normal_force_w.shape[-1] != 3:
        raise ValueError(
            "normal_force_w must have shape [E, K, 3], "
            f"got {tuple(normal_force_w.shape)}"
        )
    if not torch.is_floating_point(normal_force_w):
        raise TypeError("normal_force_w must have a floating dtype")
    num_envs, num_bodies, _ = normal_force_w.shape
    if num_bodies <= 0:
        raise ValueError("normal_force_w must contain at least one contact body")
    device = normal_force_w.device
    dtype = normal_force_w.dtype

    _validate_mask_tensor(
        "previous_active", previous_active, (num_envs, num_bodies), device=device
    )
    _validate_float_tensor(
        "previous_contact_age_s",
        previous_contact_age_s,
        (num_envs, num_bodies),
        device=device,
        dtype=dtype,
    )
    _validate_float_tensor(
        "previous_air_age_s",
        previous_air_age_s,
        (num_envs, num_bodies),
        device=device,
        dtype=dtype,
    )
    _validate_mask_tensor("temporal_valid", temporal_valid, (num_envs,), device=device)
    _validate_positive_scalar("dt", dt)
    _validate_thresholds(contact_on_threshold_n, contact_off_threshold_n)

    force = _sanitize_float(normal_force_w)
    force_norm = torch.linalg.vector_norm(force, dim=-1)
    new_active = torch.where(
        previous_active,
        force_norm >= contact_off_threshold_n,
        force_norm >= contact_on_threshold_n,
    )

    valid = temporal_valid[:, None]
    onset = valid & (~previous_active) & new_active
    release = valid & previous_active & (~new_active)

    previous_contact_age = torch.clamp(_sanitize_float(previous_contact_age_s), min=0.0)
    previous_air_age = torch.clamp(_sanitize_float(previous_air_age_s), min=0.0)
    dt_tensor = torch.full_like(previous_contact_age, dt)
    contact_age_s = torch.where(
        valid,
        torch.where(
            new_active,
            torch.where(previous_active, previous_contact_age + dt_tensor, dt_tensor),
            torch.zeros_like(previous_contact_age),
        ),
        torch.zeros_like(previous_contact_age),
    )
    air_age_s = torch.where(
        valid,
        torch.where(
            new_active,
            torch.zeros_like(previous_air_age),
            torch.where(~previous_active, previous_air_age + dt_tensor, dt_tensor),
        ),
        torch.zeros_like(previous_air_age),
    )
    return new_active, contact_age_s, air_age_s, onset, release


def compute_isaaclab_contact_obs_v1(
    root_rot: Tensor,
    rigid_body_vel: Tensor,
    normal_force_w: Tensor,
    normal_force_history_w: Tensor,
    previous_normal_force_w: Tensor,
    previous_active: Tensor,
    previous_contact_age_s: Tensor,
    previous_air_age_s: Tensor,
    temporal_valid: Tensor,
    body_weight_n: Tensor,
    dt: float,
    body_ids: BodyIds = None,
    normal_force_valid: Optional[Tensor] = None,
    normal_force_history_valid: Optional[Tensor] = None,
    sensor_data_valid: Optional[Tensor] = None,
    contact_on_threshold_n: float = ISAACLAB_CONTACT_DEFAULT_ON_THRESHOLD_N,
    contact_off_threshold_n: float = ISAACLAB_CONTACT_DEFAULT_OFF_THRESHOLD_N,
    force_clip_bodyweights: float = 10.0,
    force_delta_clip_bodyweights: float = 10.0,
    velocity_clip_mps: float = 5.0,
    duration_clip_s: float = 2.0,
    fallback_force_reference_n: float = 600.0,
    force_epsilon_n: float = 1.0e-4,
    load_fraction_epsilon: float = 1.0e-6,
    w_last: bool = True,
) -> Tensor:
    """Encode IsaacLab aggregate contact data as exactly ``[E, 20*K+6]``.

    ``normal_force_history_w`` is ``[E,H,K,3]`` with history index zero the
    newest physics sample.  Statistics span the full supplied latest-control-
    interval history and use population standard deviation.  Current contact
    state is based only on the current aggregate normal-force magnitude, never
    on a binary simulator flag or history peak.
    """

    if not isinstance(normal_force_w, Tensor):
        raise TypeError("normal_force_w must be a torch.Tensor")
    if normal_force_w.ndim != 3 or normal_force_w.shape[-1] != 3:
        raise ValueError(
            "normal_force_w must have shape [E, K, 3], "
            f"got {tuple(normal_force_w.shape)}"
        )
    if not torch.is_floating_point(normal_force_w):
        raise TypeError("normal_force_w must have a floating dtype")
    num_envs, num_bodies, _ = normal_force_w.shape
    if num_bodies <= 0:
        raise ValueError("normal_force_w must contain at least one contact body")
    device = normal_force_w.device
    dtype = normal_force_w.dtype

    _validate_float_tensor(
        "root_rot", root_rot, (num_envs, 4), device=device, dtype=dtype
    )
    if not isinstance(normal_force_history_w, Tensor):
        raise TypeError("normal_force_history_w must be a torch.Tensor")
    if (
        normal_force_history_w.ndim != 4
        or normal_force_history_w.shape[0] != num_envs
        or normal_force_history_w.shape[1] <= 0
        or normal_force_history_w.shape[2:] != (num_bodies, 3)
    ):
        raise ValueError(
            "normal_force_history_w must have shape [E, H, K, 3] with H > 0, "
            f"got {tuple(normal_force_history_w.shape)}"
        )
    if normal_force_history_w.device != device:
        raise ValueError(
            "normal_force_history_w must be on device "
            f"{device}, got {normal_force_history_w.device}"
        )
    if not torch.is_floating_point(normal_force_history_w):
        raise TypeError("normal_force_history_w must have a floating dtype")
    if normal_force_history_w.dtype != dtype:
        raise TypeError(
            f"normal_force_history_w must have dtype {dtype}, "
            f"got {normal_force_history_w.dtype}"
        )
    history_length = normal_force_history_w.shape[1]

    _validate_float_tensor(
        "previous_normal_force_w",
        previous_normal_force_w,
        (num_envs, num_bodies, 3),
        device=device,
        dtype=dtype,
    )
    _validate_mask_tensor(
        "previous_active", previous_active, (num_envs, num_bodies), device=device
    )
    _validate_float_tensor(
        "previous_contact_age_s",
        previous_contact_age_s,
        (num_envs, num_bodies),
        device=device,
        dtype=dtype,
    )
    _validate_float_tensor(
        "previous_air_age_s",
        previous_air_age_s,
        (num_envs, num_bodies),
        device=device,
        dtype=dtype,
    )
    _validate_mask_tensor("temporal_valid", temporal_valid, (num_envs,), device=device)

    if normal_force_valid is not None:
        _validate_mask_tensor(
            "normal_force_valid",
            normal_force_valid,
            (num_envs, num_bodies),
            device=device,
        )
    if normal_force_history_valid is not None:
        _validate_mask_tensor(
            "normal_force_history_valid",
            normal_force_history_valid,
            (num_envs, history_length, num_bodies),
            device=device,
        )
    if sensor_data_valid is not None:
        _validate_mask_tensor(
            "sensor_data_valid", sensor_data_valid, (num_envs,), device=device
        )

    _validate_positive_scalar("dt", dt)
    _validate_thresholds(contact_on_threshold_n, contact_off_threshold_n)
    _validate_positive_scalar("force_clip_bodyweights", force_clip_bodyweights)
    _validate_positive_scalar(
        "force_delta_clip_bodyweights", force_delta_clip_bodyweights
    )
    _validate_positive_scalar("velocity_clip_mps", velocity_clip_mps)
    _validate_positive_scalar("duration_clip_s", duration_clip_s)
    _validate_positive_scalar("fallback_force_reference_n", fallback_force_reference_n)
    _validate_positive_scalar("force_epsilon_n", force_epsilon_n)
    _validate_positive_scalar("load_fraction_epsilon", load_fraction_epsilon)

    selected_velocity_w = _select_body_state(
        "rigid_body_vel",
        rigid_body_vel,
        num_envs=num_envs,
        num_selected_bodies=num_bodies,
        device=device,
        dtype=dtype,
        body_ids=body_ids,
    )
    weight = _body_weight_view(
        body_weight_n,
        num_envs,
        device=device,
        dtype=dtype,
        fallback_force_reference_n=fallback_force_reference_n,
        force_epsilon_n=force_epsilon_n,
    )

    force = _sanitize_float(normal_force_w)
    history = _sanitize_float(normal_force_history_w)
    previous_force = _sanitize_float(previous_normal_force_w)
    if normal_force_valid is not None:
        force = torch.where(
            normal_force_valid[..., None], force, torch.zeros_like(force)
        )
    if normal_force_history_valid is not None:
        history = torch.where(
            normal_force_history_valid[..., None], history, torch.zeros_like(history)
        )
    effective_temporal_valid = temporal_valid
    if sensor_data_valid is not None:
        force = torch.where(
            sensor_data_valid[:, None, None], force, torch.zeros_like(force)
        )
        history = torch.where(
            sensor_data_valid[:, None, None, None],
            history,
            torch.zeros_like(history),
        )
        effective_temporal_valid = temporal_valid & sensor_data_valid

    active, contact_age_s, air_age_s, onset, release = update_isaaclab_contact_state_v1(
        normal_force_w=force,
        previous_active=previous_active,
        previous_contact_age_s=previous_contact_age_s,
        previous_air_age_s=previous_air_age_s,
        temporal_valid=effective_temporal_valid,
        dt=dt,
        contact_on_threshold_n=contact_on_threshold_n,
        contact_off_threshold_n=contact_off_threshold_n,
    )

    heading_inverse = rotations.calc_heading_quat_inv(root_rot, w_last=w_last)
    heading_for_bodies = heading_inverse[:, None, :].expand(-1, num_bodies, -1)
    force_heading = rotations.quat_rotate(heading_for_bodies, force, w_last=w_last)
    force_delta_w = torch.where(
        effective_temporal_valid[:, None, None],
        force - previous_force,
        torch.zeros_like(force),
    )
    force_delta_heading = rotations.quat_rotate(
        heading_for_bodies, force_delta_w, w_last=w_last
    )

    force_norm = torch.linalg.vector_norm(force, dim=-1)
    history_force_norm = torch.linalg.vector_norm(history, dim=-1)
    history_mean = history_force_norm.mean(dim=1)
    history_peak = history_force_norm.amax(dim=1)
    history_variance = torch.mean(
        torch.square(history_force_norm - history_mean[:, None, :]), dim=1
    )
    history_std = torch.sqrt(torch.clamp(history_variance, min=0.0))

    upward_support = torch.clamp(force[..., 2], min=0.0)
    total_upward_support = upward_support.sum(dim=1, keepdim=True)
    load_fraction = upward_support / (total_upward_support + load_fraction_epsilon)

    velocity = _sanitize_float(selected_velocity_w)
    vertical_velocity = torch.clamp(
        velocity[..., 2] / velocity_clip_mps, min=-1.0, max=1.0
    )
    planar_speed = torch.clamp(
        torch.linalg.vector_norm(velocity[..., :2], dim=-1) / velocity_clip_mps,
        min=0.0,
        max=1.0,
    )
    contact_age = torch.clamp(contact_age_s / duration_clip_s, min=0.0, max=1.0)
    air_age = torch.clamp(air_age_s / duration_clip_s, min=0.0, max=1.0)

    force_bodyweights = force_heading / weight
    force_delta_bodyweights = force_delta_heading / weight
    scalar_weight = weight.squeeze(-1)

    temporal_valid_obs = effective_temporal_valid[:, None].expand(-1, num_bodies)
    per_body = torch.cat(
        (
            active.to(dtype=dtype).unsqueeze(-1),
            temporal_valid_obs.to(dtype=dtype).unsqueeze(-1),
            _signed_log_compress_bodyweights(force_bodyweights, force_clip_bodyweights),
            _signed_log_compress_bodyweights(
                force_delta_bodyweights, force_delta_clip_bodyweights
            ),
            _unsigned_log_compress_bodyweights(
                force_norm / scalar_weight, force_clip_bodyweights
            ).unsqueeze(-1),
            _unsigned_log_compress_bodyweights(
                history_mean / scalar_weight, force_clip_bodyweights
            ).unsqueeze(-1),
            _unsigned_log_compress_bodyweights(
                history_peak / scalar_weight, force_clip_bodyweights
            ).unsqueeze(-1),
            _unsigned_log_compress_bodyweights(
                history_std / scalar_weight, force_clip_bodyweights
            ).unsqueeze(-1),
            _unsigned_log_compress_bodyweights(
                upward_support / scalar_weight, force_clip_bodyweights
            ).unsqueeze(-1),
            load_fraction.unsqueeze(-1),
            vertical_velocity.unsqueeze(-1),
            planar_speed.unsqueeze(-1),
            contact_age.unsqueeze(-1),
            air_age.unsqueeze(-1),
            onset.to(dtype=dtype).unsqueeze(-1),
            release.to(dtype=dtype).unsqueeze(-1),
        ),
        dim=-1,
    )

    if num_bodies <= 1:
        support_entropy = torch.zeros_like(total_upward_support)
    else:
        entropy_numerator = -torch.sum(
            load_fraction
            * torch.log(
                torch.clamp(
                    load_fraction + load_fraction_epsilon, min=load_fraction_epsilon
                )
            ),
            dim=1,
            keepdim=True,
        )
        support_entropy = entropy_numerator / math.log(num_bodies)
        support_entropy = torch.where(
            total_upward_support >= load_fraction_epsilon,
            support_entropy,
            torch.zeros_like(support_entropy),
        )
        support_entropy = torch.clamp(
            _sanitize_float(support_entropy), min=0.0, max=1.0
        )

    net_force_norm = torch.linalg.vector_norm(force.sum(dim=1), dim=-1, keepdim=True)
    max_history_peak = history_peak.amax(dim=1, keepdim=True)
    global_features = torch.cat(
        (
            active.any(dim=1, keepdim=True).to(dtype=dtype),
            active.to(dtype=dtype).mean(dim=1, keepdim=True),
            _unsigned_log_compress_bodyweights(
                total_upward_support / weight[:, 0, :], force_clip_bodyweights
            ),
            _unsigned_log_compress_bodyweights(
                net_force_norm / weight[:, 0, :], force_clip_bodyweights
            ),
            _unsigned_log_compress_bodyweights(
                max_history_peak / weight[:, 0, :], force_clip_bodyweights
            ),
            support_entropy,
        ),
        dim=-1,
    )
    observation = torch.cat((per_body.reshape(num_envs, -1), global_features), dim=-1)
    return _sanitize_float(observation)


def compute_isaaclab_contact_pair_obs_v1(
    root_rot: Tensor,
    rigid_body_pos: Tensor,
    filtered_normal_force_w: Tensor,
    friction_force_w: Tensor,
    mean_contact_point_w: Tensor,
    pair_slot_valid: Tensor,
    mean_contact_point_valid: Tensor,
    body_weight_n: Tensor,
    body_ids: BodyIds = None,
    filtered_normal_force_valid: Optional[Tensor] = None,
    friction_force_valid: Optional[Tensor] = None,
    sensor_data_valid: Optional[Tensor] = None,
    contact_on_threshold_n: float = ISAACLAB_CONTACT_DEFAULT_ON_THRESHOLD_N,
    force_clip_bodyweights: float = 10.0,
    tangential_normal_ratio_clip: float = 2.0,
    contact_point_scale_m: float = 1.0,
    fallback_force_reference_n: float = 600.0,
    force_epsilon_n: float = 1.0e-4,
    w_last: bool = True,
) -> Tensor:
    """Encode filtered IsaacLab pair data as exactly ``[E, 15*K*F]``.

    Flattening is body-major, then filter-major, then feature-major.  A valid
    filter slot with no current contact remains a valid slot.  Contact-point
    validity is independent and the relative point is exactly zero when its
    source average is invalid.  Friction is never inferred from horizontal
    components of the normal force.
    """

    if not isinstance(filtered_normal_force_w, Tensor):
        raise TypeError("filtered_normal_force_w must be a torch.Tensor")
    if filtered_normal_force_w.ndim != 4 or filtered_normal_force_w.shape[-1] != 3:
        raise ValueError(
            "filtered_normal_force_w must have shape [E, K, F, 3], "
            f"got {tuple(filtered_normal_force_w.shape)}"
        )
    if not torch.is_floating_point(filtered_normal_force_w):
        raise TypeError("filtered_normal_force_w must have a floating dtype")
    num_envs, num_bodies, num_filters, _ = filtered_normal_force_w.shape
    if num_bodies <= 0 or num_filters <= 0:
        raise ValueError(
            "filtered_normal_force_w must contain at least one body and filter"
        )
    device = filtered_normal_force_w.device
    dtype = filtered_normal_force_w.dtype

    _validate_float_tensor(
        "root_rot", root_rot, (num_envs, 4), device=device, dtype=dtype
    )
    pair_vector_shape = (num_envs, num_bodies, num_filters, 3)
    _validate_float_tensor(
        "friction_force_w",
        friction_force_w,
        pair_vector_shape,
        device=device,
        dtype=dtype,
    )
    _validate_float_tensor(
        "mean_contact_point_w",
        mean_contact_point_w,
        pair_vector_shape,
        device=device,
        dtype=dtype,
    )
    pair_shape = (num_envs, num_bodies, num_filters)
    _validate_mask_tensor("pair_slot_valid", pair_slot_valid, pair_shape, device=device)
    _validate_mask_tensor(
        "mean_contact_point_valid",
        mean_contact_point_valid,
        pair_shape,
        device=device,
    )
    if filtered_normal_force_valid is not None:
        _validate_mask_tensor(
            "filtered_normal_force_valid",
            filtered_normal_force_valid,
            pair_shape,
            device=device,
        )
    if friction_force_valid is not None:
        _validate_mask_tensor(
            "friction_force_valid",
            friction_force_valid,
            pair_shape,
            device=device,
        )
    if sensor_data_valid is not None:
        _validate_mask_tensor(
            "sensor_data_valid", sensor_data_valid, (num_envs,), device=device
        )

    _validate_finite_scalar("contact_on_threshold_n", contact_on_threshold_n)
    if contact_on_threshold_n < 0.0:
        raise ValueError(
            "contact_on_threshold_n must be non-negative, "
            f"got {contact_on_threshold_n}"
        )
    _validate_positive_scalar("force_clip_bodyweights", force_clip_bodyweights)
    _validate_positive_scalar(
        "tangential_normal_ratio_clip", tangential_normal_ratio_clip
    )
    _validate_positive_scalar("contact_point_scale_m", contact_point_scale_m)
    _validate_positive_scalar("fallback_force_reference_n", fallback_force_reference_n)
    _validate_positive_scalar("force_epsilon_n", force_epsilon_n)

    selected_body_pos = _select_body_state(
        "rigid_body_pos",
        rigid_body_pos,
        num_envs=num_envs,
        num_selected_bodies=num_bodies,
        device=device,
        dtype=dtype,
        body_ids=body_ids,
    )
    weight = _body_weight_view(
        body_weight_n,
        num_envs,
        device=device,
        dtype=dtype,
        fallback_force_reference_n=fallback_force_reference_n,
        force_epsilon_n=force_epsilon_n,
    )[:, :, None, :]

    normal_force = _sanitize_float(filtered_normal_force_w)
    friction_force = _sanitize_float(friction_force_w)
    if filtered_normal_force_valid is not None:
        normal_force = torch.where(
            filtered_normal_force_valid[..., None],
            normal_force,
            torch.zeros_like(normal_force),
        )
    if friction_force_valid is not None:
        friction_force = torch.where(
            friction_force_valid[..., None],
            friction_force,
            torch.zeros_like(friction_force),
        )
    slot_valid = pair_slot_valid
    measurement_valid = slot_valid
    if sensor_data_valid is not None:
        measurement_valid = slot_valid & sensor_data_valid[:, None, None]
    normal_force = torch.where(
        measurement_valid[..., None], normal_force, torch.zeros_like(normal_force)
    )
    friction_force = torch.where(
        measurement_valid[..., None], friction_force, torch.zeros_like(friction_force)
    )

    heading_inverse = rotations.calc_heading_quat_inv(root_rot, w_last=w_last)
    heading_for_pairs = heading_inverse[:, None, None, :].expand(
        -1, num_bodies, num_filters, -1
    )
    normal_force_heading = rotations.quat_rotate(
        heading_for_pairs, normal_force, w_last=w_last
    )
    friction_force_heading = rotations.quat_rotate(
        heading_for_pairs, friction_force, w_last=w_last
    )
    normal_force_norm = torch.linalg.vector_norm(normal_force, dim=-1)
    friction_force_norm = torch.linalg.vector_norm(friction_force, dim=-1)
    pair_active = measurement_valid & (normal_force_norm >= contact_on_threshold_n)
    ratio = friction_force_norm / (normal_force_norm + force_epsilon_n)
    ratio_obs = torch.clamp(
        _sanitize_float(ratio / tangential_normal_ratio_clip), min=0.0, max=1.0
    )

    finite_point = torch.isfinite(mean_contact_point_w).all(dim=-1)
    point_valid = measurement_valid & mean_contact_point_valid & finite_point
    safe_point = _sanitize_float(mean_contact_point_w)
    relative_point_w = safe_point - _sanitize_float(selected_body_pos)[:, :, None, :]
    relative_point_w = torch.where(
        point_valid[..., None], relative_point_w, torch.zeros_like(relative_point_w)
    )
    relative_point_heading = rotations.quat_rotate(
        heading_for_pairs, relative_point_w, w_last=w_last
    )
    relative_point_obs = torch.clamp(
        _sanitize_float(relative_point_heading / contact_point_scale_m),
        min=-1.0,
        max=1.0,
    )

    per_pair = torch.cat(
        (
            pair_active.to(dtype=dtype).unsqueeze(-1),
            slot_valid.to(dtype=dtype).unsqueeze(-1),
            _signed_log_compress_bodyweights(
                normal_force_heading / weight, force_clip_bodyweights
            ),
            _unsigned_log_compress_bodyweights(
                normal_force_norm / weight.squeeze(-1), force_clip_bodyweights
            ).unsqueeze(-1),
            _signed_log_compress_bodyweights(
                friction_force_heading / weight, force_clip_bodyweights
            ),
            _unsigned_log_compress_bodyweights(
                friction_force_norm / weight.squeeze(-1), force_clip_bodyweights
            ).unsqueeze(-1),
            ratio_obs.unsqueeze(-1),
            relative_point_obs,
            point_valid.to(dtype=dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    return _sanitize_float(per_pair.reshape(num_envs, -1))


__all__ = [
    "ISAACLAB_CONTACT_OBS_V1_PER_BODY_DIM",
    "ISAACLAB_CONTACT_OBS_V1_GLOBAL_DIM",
    "ISAACLAB_CONTACT_PAIR_OBS_V1_PER_PAIR_DIM",
    "ISAACLAB_CONTACT_OBS_V1_LAYOUT",
    "ISAACLAB_CONTACT_OBS_V1_GLOBAL_LAYOUT",
    "ISAACLAB_CONTACT_PAIR_OBS_V1_LAYOUT",
    "ISAACLAB_CONTACT_DEFAULT_ON_THRESHOLD_N",
    "ISAACLAB_CONTACT_DEFAULT_OFF_THRESHOLD_N",
    "isaaclab_contact_obs_v1_dim",
    "isaaclab_contact_pair_obs_v1_dim",
    "unflatten_isaaclab_contact_obs_v1",
    "unflatten_isaaclab_contact_pair_obs_v1",
    "update_isaaclab_contact_state_v1",
    "compute_isaaclab_contact_obs_v1",
    "compute_isaaclab_contact_pair_obs_v1",
]
