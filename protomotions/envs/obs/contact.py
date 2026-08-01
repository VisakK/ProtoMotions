# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics-informed observations from aggregate per-body contact forces.

``contact_obs_v1`` deliberately uses only state that is available from the
common simulator API: a net three-dimensional force for each rigid body,
rigid-body origin velocity, and temporal state maintained by the environment.
The force is an aggregate, not a contact manifold.  In particular, it does not
identify contact points, contact normals, collision pairs, pressure, moments,
or centers of pressure.

Vertical/horizontal force and body-origin velocity features are explicitly
flat-ground proxies.  They are not exact normal/tangential contact quantities
for tilted surfaces, object contacts, or self-contact.
"""

from __future__ import annotations

import math
from numbers import Integral
from types import MappingProxyType
from typing import List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

from protomotions.utils import rotations


CONTACT_OBS_V1_PER_BODY_DIM = 17
CONTACT_OBS_V1_GLOBAL_DIM = 4

CONTACT_OBS_V1_LAYOUT: Mapping[str, slice] = MappingProxyType(
    {
        "active": slice(0, 1),
        "net_force_heading": slice(1, 4),
        "force_rate_heading": slice(4, 7),
        "force_magnitude": slice(7, 8),
        "upward_force_proxy": slice(8, 9),
        "horizontal_force_proxy": slice(9, 10),
        "ground_friction_utilization_proxy": slice(10, 11),
        "support_load_fraction_proxy": slice(11, 12),
        "body_origin_normal_velocity_proxy": slice(12, 13),
        "body_origin_tangent_speed_proxy": slice(13, 14),
        "contact_age": slice(14, 15),
        "air_age": slice(15, 16),
        "temporal_valid": slice(16, 17),
    }
)

CONTACT_OBS_V1_GLOBAL_LAYOUT: Mapping[str, slice] = MappingProxyType(
    {
        "any_selected_contact": slice(0, 1),
        "active_body_fraction": slice(1, 2),
        "total_upward_force_proxy": slice(2, 3),
        "total_horizontal_force_proxy": slice(3, 4),
    }
)

DEFAULT_CONTACT_FORCE_ON_THRESHOLD_N = 5.0
DEFAULT_CONTACT_FORCE_OFF_THRESHOLD_N = 2.0

BodyIds = Optional[Union[List[int], Tensor]]


def _validate_finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite scalar, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _validate_positive_scalar(name: str, value: float) -> None:
    _validate_finite_scalar(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")


def _validate_log_compression(
    reference: float,
    clip: float,
    reference_name: str = "reference",
    clip_name: str = "clip",
) -> None:
    _validate_positive_scalar(reference_name, reference)
    _validate_finite_scalar(clip_name, clip)
    if clip <= reference:
        raise ValueError(
            f"{clip_name} must be greater than {reference_name}, "
            f"got {clip_name}={clip}, {reference_name}={reference}"
        )


def signed_log_compress(x: Tensor, reference: float, clip: float) -> Tensor:
    """Signed logarithmic compression with a fixed ``[-1, 1]`` range.

    Finite values at or beyond ``clip`` map to ``-1`` or ``1``.  Non-finite
    samples are converted to finite boundary/zero values so a transient bad
    simulator sample cannot inject NaN or Inf into a policy observation.
    """

    _validate_log_compression(reference, clip)
    finite_x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    clipped_x = torch.clamp(finite_x, min=-clip, max=clip)
    denominator = math.log1p(clip / reference)
    return (
        torch.sign(clipped_x)
        * torch.log1p(torch.abs(clipped_x) / reference)
        / denominator
    )


def unsigned_log_compress(x: Tensor, reference: float, clip: float) -> Tensor:
    """Logarithmically compress a non-negative quantity into ``[0, 1]``."""

    _validate_log_compression(reference, clip)
    finite_x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=0.0)
    clipped_x = torch.clamp(finite_x, min=0.0, max=clip)
    denominator = math.log1p(clip / reference)
    return torch.log1p(clipped_x / reference) / denominator


def contact_obs_v1_dim(num_contact_bodies: int) -> int:
    """Return the flattened ``contact_obs_v1`` size for a positive body count."""

    if (
        not isinstance(num_contact_bodies, Integral)
        or isinstance(num_contact_bodies, bool)
        or num_contact_bodies <= 0
    ):
        raise ValueError(
            "num_contact_bodies must be a positive integer, "
            f"got {num_contact_bodies!r}"
        )
    return (
        int(num_contact_bodies) * CONTACT_OBS_V1_PER_BODY_DIM
        + CONTACT_OBS_V1_GLOBAL_DIM
    )


def unflatten_contact_obs_v1(
    observation: Tensor, num_contact_bodies: int
) -> Tuple[Tensor, Tensor]:
    """Split a flat observation into ``[E, K, 17]`` and ``[E, 4]`` views."""

    expected_dim = contact_obs_v1_dim(num_contact_bodies)
    if observation.ndim != 2 or observation.shape[-1] != expected_dim:
        raise ValueError(
            "contact_obs_v1 must have shape "
            f"[E, {expected_dim}] for K={num_contact_bodies}, "
            f"got {tuple(observation.shape)}"
        )

    per_body_flat_dim = num_contact_bodies * CONTACT_OBS_V1_PER_BODY_DIM
    per_body = observation[:, :per_body_flat_dim].reshape(
        observation.shape[0],
        num_contact_bodies,
        CONTACT_OBS_V1_PER_BODY_DIM,
    )
    global_features = observation[:, per_body_flat_dim:]
    return per_body, global_features


def _validate_body_tensor(
    name: str,
    tensor: Tensor,
    expected_shape: Tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on device {device}, got device {tensor.device}"
        )


def _body_id_tensor(
    body_ids: BodyIds,
    num_bodies: int,
    device: torch.device,
) -> Optional[Tensor]:
    if body_ids is None:
        if num_bodies <= 0:
            raise ValueError("contact_obs_v1 requires at least one rigid body")
        return None

    if isinstance(body_ids, Tensor):
        if body_ids.ndim != 1:
            raise ValueError(
                f"body_ids must be one-dimensional, got shape {tuple(body_ids.shape)}"
            )
        if body_ids.numel() == 0:
            raise ValueError("contact_obs_v1 requires at least one selected body")
        if body_ids.dtype != torch.long:
            raise TypeError(
                f"body_ids tensor must have dtype torch.long, got {body_ids.dtype}"
            )
        if body_ids.device != device:
            raise ValueError(
                f"body_ids must be on device {device}, got device {body_ids.device}"
            )
        return body_ids

    selected_ids = list(body_ids)
    if not selected_ids:
        raise ValueError("contact_obs_v1 requires at least one selected body")
    if any(
        not isinstance(body_id, Integral) or isinstance(body_id, bool)
        for body_id in selected_ids
    ):
        raise TypeError("every body_ids entry must be an integer")
    selected_ids = [int(body_id) for body_id in selected_ids]
    if min(selected_ids) < 0 or max(selected_ids) >= num_bodies:
        raise ValueError(f"body_ids must be in [0, {num_bodies}), got {selected_ids}")
    return torch.tensor(selected_ids, dtype=torch.long, device=device)


def _validate_contact_obs_inputs(
    root_rot: Tensor,
    rigid_body_vel: Tensor,
    rigid_body_contact_forces: Tensor,
    previous_contact_forces: Tensor,
    contact_active_state: Tensor,
    contact_age_steps: Tensor,
    contact_air_age_steps: Tensor,
    contact_temporal_valid: Tensor,
) -> Tuple[int, int]:
    if rigid_body_contact_forces is None:
        raise ValueError(
            "rigid_body_contact_forces is unavailable; contact_obs_v1 requires "
            "a simulator backend that reports per-body net contact forces"
        )
    if not isinstance(rigid_body_contact_forces, Tensor):
        raise TypeError("rigid_body_contact_forces must be a torch.Tensor")
    if rigid_body_contact_forces.ndim != 3 or rigid_body_contact_forces.shape[-1] != 3:
        raise ValueError(
            "rigid_body_contact_forces must have shape [E, B, 3], "
            f"got {tuple(rigid_body_contact_forces.shape)}"
        )
    if not torch.is_floating_point(rigid_body_contact_forces):
        raise TypeError("rigid_body_contact_forces must have a floating dtype")

    num_envs, num_bodies, _ = rigid_body_contact_forces.shape
    device = rigid_body_contact_forces.device
    floating_dtype = rigid_body_contact_forces.dtype

    _validate_body_tensor("root_rot", root_rot, (num_envs, 4), device)
    _validate_body_tensor(
        "rigid_body_vel",
        rigid_body_vel,
        (num_envs, num_bodies, 3),
        device,
    )
    _validate_body_tensor(
        "previous_contact_forces",
        previous_contact_forces,
        (num_envs, num_bodies, 3),
        device,
    )
    _validate_body_tensor(
        "contact_active_state",
        contact_active_state,
        (num_envs, num_bodies),
        device,
    )
    _validate_body_tensor(
        "contact_age_steps",
        contact_age_steps,
        (num_envs, num_bodies),
        device,
    )
    _validate_body_tensor(
        "contact_air_age_steps",
        contact_air_age_steps,
        (num_envs, num_bodies),
        device,
    )
    _validate_body_tensor(
        "contact_temporal_valid",
        contact_temporal_valid,
        (num_envs,),
        device,
    )

    for name, tensor in (
        ("root_rot", root_rot),
        ("rigid_body_vel", rigid_body_vel),
        ("previous_contact_forces", previous_contact_forces),
    ):
        if tensor.dtype != floating_dtype:
            raise TypeError(
                f"{name} must have dtype {floating_dtype}, got {tensor.dtype}"
            )

    return num_envs, num_bodies


def _validate_contact_obs_config(
    dt: float,
    force_reference_n: float,
    force_clip_n: float,
    force_rate_reference_n_per_s: float,
    force_rate_clip_n_per_s: float,
    friction_mu: float,
    friction_utilization_clip: float,
    velocity_reference_mps: float,
    contact_age_clip_s: float,
    air_age_clip_s: float,
    eps: float,
) -> None:
    _validate_finite_scalar("dt", dt)
    if dt < 0:
        raise ValueError(f"dt must be non-negative, got {dt}")
    _validate_log_compression(
        force_reference_n,
        force_clip_n,
        reference_name="force_reference_n",
        clip_name="force_clip_n",
    )
    _validate_log_compression(
        force_rate_reference_n_per_s,
        force_rate_clip_n_per_s,
        reference_name="force_rate_reference_n_per_s",
        clip_name="force_rate_clip_n_per_s",
    )
    _validate_positive_scalar("friction_mu", friction_mu)
    _validate_positive_scalar("friction_utilization_clip", friction_utilization_clip)
    _validate_positive_scalar("velocity_reference_mps", velocity_reference_mps)
    _validate_positive_scalar("contact_age_clip_s", contact_age_clip_s)
    _validate_positive_scalar("air_age_clip_s", air_age_clip_s)
    _validate_positive_scalar("eps", eps)


def compute_contact_obs_v1(
    root_rot: Tensor,
    rigid_body_vel: Tensor,
    rigid_body_contact_forces: Tensor,
    previous_contact_forces: Tensor,
    contact_active_state: Tensor,
    contact_age_steps: Tensor,
    contact_air_age_steps: Tensor,
    contact_temporal_valid: Tensor,
    dt: float,
    body_ids: BodyIds = None,
    force_reference_n: float = 100.0,
    force_clip_n: float = 5000.0,
    force_rate_reference_n_per_s: float = 1000.0,
    force_rate_clip_n_per_s: float = 50000.0,
    friction_mu: float = 1.0,
    friction_utilization_clip: float = 2.0,
    velocity_reference_mps: float = 2.0,
    contact_age_clip_s: float = 2.0,
    air_age_clip_s: float = 2.0,
    eps: float = 1e-6,
    w_last: bool = True,
) -> Tensor:
    """Compute the versioned current-contact observation.

    Args:
        root_rot: Root orientation quaternions, shape ``[E, 4]``.
        rigid_body_vel: World-frame rigid-body origin velocities, ``[E, B, 3]``.
        rigid_body_contact_forces: World-frame aggregate per-body net forces,
            shape ``[E, B, 3]``.
        previous_contact_forces: Previous-step aggregate forces, ``[E, B, 3]``.
        contact_active_state: Hysteretic active state, shape ``[E, B]``.
        contact_age_steps: Consecutive active-contact step counts, ``[E, B]``.
        contact_air_age_steps: Consecutive inactive step counts, ``[E, B]``.
        contact_temporal_valid: Previous-sample validity per environment,
            shape ``[E]``. Invalid samples have a zero force rate.
        dt: Policy/environment step duration in seconds.
        body_ids: Selected rigid bodies in deterministic output order. ``None``
            selects all bodies. A device-resident ``torch.long`` tensor avoids
            constructing an index tensor in a GPU forward path.
        w_last: Whether quaternions use XYZW rather than WXYZ ordering.

    Returns:
        A tensor of shape ``[E, K * 17 + 4]``. Each ``[E, K, 17]`` body
        block is ordered as:

        ``active, net_force_heading[3], force_rate_heading[3],
        force_magnitude, upward_force_proxy, horizontal_force_proxy,
        ground_friction_utilization_proxy, support_load_fraction_proxy,
        body_origin_normal_velocity_proxy,
        body_origin_tangent_speed_proxy, contact_age, air_age,
        temporal_valid``.

        The final four values are ``any_selected_contact``,
        ``active_body_fraction``, log-compressed total positive world-up
        force, and log-compressed norm of the *summed* world-horizontal force.
        The last definition is net horizontal support force, not a sum of
        per-body horizontal magnitudes.

    Notes:
        Force vectors are expressed in the inverse root-heading frame. Force
        magnitude and world-up/horizontal quantities are compressed separately.
        The world-up decomposition and origin-velocity values are flat-ground
        proxies; aggregate force alone cannot provide a true contact normal or
        contact-point velocity. Inputs are not mutated.
    """

    num_envs, num_bodies = _validate_contact_obs_inputs(
        root_rot,
        rigid_body_vel,
        rigid_body_contact_forces,
        previous_contact_forces,
        contact_active_state,
        contact_age_steps,
        contact_air_age_steps,
        contact_temporal_valid,
    )
    _validate_contact_obs_config(
        dt,
        force_reference_n,
        force_clip_n,
        force_rate_reference_n_per_s,
        force_rate_clip_n_per_s,
        friction_mu,
        friction_utilization_clip,
        velocity_reference_mps,
        contact_age_clip_s,
        air_age_clip_s,
        eps,
    )

    selected_ids = _body_id_tensor(
        body_ids, num_bodies, rigid_body_contact_forces.device
    )
    if selected_ids is None:
        force_w = rigid_body_contact_forces
        previous_force_w = previous_contact_forces
        velocity_w = rigid_body_vel
        active_bool = contact_active_state.bool()
        selected_contact_age_steps = contact_age_steps
        selected_air_age_steps = contact_air_age_steps
    else:
        force_w = rigid_body_contact_forces.index_select(1, selected_ids)
        previous_force_w = previous_contact_forces.index_select(1, selected_ids)
        velocity_w = rigid_body_vel.index_select(1, selected_ids)
        active_bool = contact_active_state.index_select(1, selected_ids).bool()
        selected_contact_age_steps = contact_age_steps.index_select(1, selected_ids)
        selected_air_age_steps = contact_air_age_steps.index_select(1, selected_ids)

    num_selected_bodies = force_w.shape[1]
    dtype = force_w.dtype

    heading_inverse = rotations.calc_heading_quat_inv(root_rot, w_last=w_last)
    heading_inverse = heading_inverse.unsqueeze(1).expand(-1, num_selected_bodies, -1)
    force_heading = rotations.quat_rotate(heading_inverse, force_w, w_last=w_last)

    effective_dt = max(dt, eps)
    temporal_valid_bool = contact_temporal_valid.bool()
    force_rate_w = (force_w - previous_force_w) / effective_dt
    force_rate_w = torch.where(
        temporal_valid_bool[:, None, None],
        force_rate_w,
        torch.zeros_like(force_rate_w),
    )
    force_rate_heading = rotations.quat_rotate(
        heading_inverse, force_rate_w, w_last=w_last
    )

    force_magnitude = torch.linalg.vector_norm(force_w, dim=-1)
    upward_force = torch.clamp(force_w[..., 2], min=0.0)
    horizontal_force = torch.linalg.vector_norm(force_w[..., :2], dim=-1)

    friction_utilization = horizontal_force / (friction_mu * upward_force + eps)
    friction_utilization_obs = torch.clamp(
        torch.nan_to_num(
            friction_utilization / friction_utilization_clip,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        min=0.0,
        max=1.0,
    )

    total_upward_force = upward_force.sum(dim=1, keepdim=True)
    load_fraction = upward_force / torch.clamp(total_upward_force, min=eps)
    load_fraction = torch.nan_to_num(load_fraction, nan=0.0, posinf=0.0, neginf=0.0)

    normal_velocity_obs = torch.clamp(
        torch.nan_to_num(
            velocity_w[..., 2] / velocity_reference_mps,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ),
        min=-1.0,
        max=1.0,
    )
    tangent_speed_obs = torch.clamp(
        torch.nan_to_num(
            torch.linalg.vector_norm(velocity_w[..., :2], dim=-1)
            / velocity_reference_mps,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        min=0.0,
        max=1.0,
    )

    contact_age_obs = torch.clamp(
        selected_contact_age_steps.to(dtype=dtype) * dt / contact_age_clip_s,
        min=0.0,
        max=1.0,
    )
    air_age_obs = torch.clamp(
        selected_air_age_steps.to(dtype=dtype) * dt / air_age_clip_s,
        min=0.0,
        max=1.0,
    )
    contact_age_obs = torch.nan_to_num(contact_age_obs, nan=0.0, posinf=1.0, neginf=0.0)
    air_age_obs = torch.nan_to_num(air_age_obs, nan=0.0, posinf=1.0, neginf=0.0)

    active_obs = active_bool.to(dtype=dtype)
    temporal_valid_obs = (
        temporal_valid_bool[:, None].expand(-1, num_selected_bodies).to(dtype=dtype)
    )

    per_body = torch.cat(
        (
            active_obs.unsqueeze(-1),
            signed_log_compress(force_heading, force_reference_n, force_clip_n),
            signed_log_compress(
                force_rate_heading,
                force_rate_reference_n_per_s,
                force_rate_clip_n_per_s,
            ),
            unsigned_log_compress(
                force_magnitude, force_reference_n, force_clip_n
            ).unsqueeze(-1),
            unsigned_log_compress(
                upward_force, force_reference_n, force_clip_n
            ).unsqueeze(-1),
            unsigned_log_compress(
                horizontal_force, force_reference_n, force_clip_n
            ).unsqueeze(-1),
            friction_utilization_obs.unsqueeze(-1),
            load_fraction.unsqueeze(-1),
            normal_velocity_obs.unsqueeze(-1),
            tangent_speed_obs.unsqueeze(-1),
            contact_age_obs.unsqueeze(-1),
            air_age_obs.unsqueeze(-1),
            temporal_valid_obs.unsqueeze(-1),
        ),
        dim=-1,
    )

    net_horizontal_force = torch.linalg.vector_norm(
        force_w[..., :2].sum(dim=1), dim=-1, keepdim=True
    )
    global_features = torch.cat(
        (
            active_bool.any(dim=1, keepdim=True).to(dtype=dtype),
            active_obs.mean(dim=1, keepdim=True),
            unsigned_log_compress(total_upward_force, force_reference_n, force_clip_n),
            unsigned_log_compress(
                net_horizontal_force, force_reference_n, force_clip_n
            ),
        ),
        dim=-1,
    )

    return torch.cat(
        (per_body.reshape(num_envs, -1), global_features),
        dim=-1,
    )


def update_contact_state(
    raw_contacts: Tensor,
    contact_forces: Tensor,
    previous_active: Tensor,
    previous_contact_age_steps: Tensor,
    previous_air_age_steps: Tensor,
    force_on_threshold_n: float = DEFAULT_CONTACT_FORCE_ON_THRESHOLD_N,
    force_off_threshold_n: float = DEFAULT_CONTACT_FORCE_OFF_THRESHOLD_N,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Update hysteretic contact state and duration counters for all bodies.

    ``raw_contacts`` and state/counter tensors have shape ``[E, B]``;
    ``contact_forces`` has shape ``[E, B, 3]``. An inactive body turns on when
    a raw contact is reported or its force norm reaches the on threshold. An
    active body stays on while raw contact remains or force reaches the lower
    off threshold. Contact age increments only while active, and air age
    increments only while inactive.

    The operation is batched, does not mutate its inputs, and returns
    ``(active, contact_age_steps, air_age_steps)``.
    """

    _validate_finite_scalar("force_on_threshold_n", force_on_threshold_n)
    _validate_finite_scalar("force_off_threshold_n", force_off_threshold_n)
    if force_off_threshold_n < 0:
        raise ValueError(
            "force_off_threshold_n must be non-negative, "
            f"got {force_off_threshold_n}"
        )
    if force_on_threshold_n < force_off_threshold_n:
        raise ValueError(
            "force_on_threshold_n must be greater than or equal to "
            "force_off_threshold_n, got "
            f"on={force_on_threshold_n}, off={force_off_threshold_n}"
        )

    if not isinstance(contact_forces, Tensor):
        raise TypeError("contact_forces must be a torch.Tensor")
    if contact_forces.ndim != 3 or contact_forces.shape[-1] != 3:
        raise ValueError(
            f"contact_forces must have shape [E, B, 3], got {tuple(contact_forces.shape)}"
        )
    if not torch.is_floating_point(contact_forces):
        raise TypeError("contact_forces must have a floating dtype")

    num_envs, num_bodies, _ = contact_forces.shape
    state_shape = (num_envs, num_bodies)
    device = contact_forces.device
    _validate_body_tensor("raw_contacts", raw_contacts, state_shape, device)
    _validate_body_tensor("previous_active", previous_active, state_shape, device)
    _validate_body_tensor(
        "previous_contact_age_steps",
        previous_contact_age_steps,
        state_shape,
        device,
    )
    _validate_body_tensor(
        "previous_air_age_steps",
        previous_air_age_steps,
        state_shape,
        device,
    )
    if previous_contact_age_steps.dtype != previous_air_age_steps.dtype:
        raise TypeError(
            "contact and air age counters must have the same dtype, got "
            f"{previous_contact_age_steps.dtype} and "
            f"{previous_air_age_steps.dtype}"
        )

    raw_contact_bool = raw_contacts.bool()
    previous_active_bool = previous_active.bool()
    force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
    turn_on = raw_contact_bool | (force_norm >= force_on_threshold_n)
    stay_on = raw_contact_bool | (force_norm >= force_off_threshold_n)
    active = torch.where(previous_active_bool, stay_on, turn_on)

    contact_age_steps = torch.where(
        active,
        previous_contact_age_steps + torch.ones_like(previous_contact_age_steps),
        torch.zeros_like(previous_contact_age_steps),
    )
    air_age_steps = torch.where(
        active,
        torch.zeros_like(previous_air_age_steps),
        previous_air_age_steps + torch.ones_like(previous_air_age_steps),
    )
    return active, contact_age_steps, air_age_steps


__all__ = [
    "CONTACT_OBS_V1_PER_BODY_DIM",
    "CONTACT_OBS_V1_GLOBAL_DIM",
    "CONTACT_OBS_V1_LAYOUT",
    "CONTACT_OBS_V1_GLOBAL_LAYOUT",
    "DEFAULT_CONTACT_FORCE_ON_THRESHOLD_N",
    "DEFAULT_CONTACT_FORCE_OFF_THRESHOLD_N",
    "signed_log_compress",
    "unsigned_log_compress",
    "contact_obs_v1_dim",
    "unflatten_contact_obs_v1",
    "compute_contact_obs_v1",
    "update_contact_state",
]
