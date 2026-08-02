# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contact-sensor capability shared by simulator backends.

The generic :class:`RobotState` contact-force field remains the compatibility
interface used throughout ProtoMotions.  This module describes richer sensor
data that only some backends can provide without pretending that every
simulator has the same contact-reporting semantics.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def sanitize_contact_vectors(
    tensor: torch.Tensor,
    lifecycle_valid: torch.Tensor,
    static_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero invalid vector measurements and return their explicit valid mask.

    Args:
        tensor: Vector samples with shape ``[E, ..., 3]``.
        lifecycle_valid: Per-environment validity with shape ``[E]``. This is
            false after a sensor reset until the backend has filled fresh history.
        static_valid: Optional validity tensor matching ``tensor.shape[:-1]``
            (for example, scene-object filter existence).
    """

    if tensor.ndim < 2 or tensor.shape[-1] != 3:
        raise ValueError("Contact vectors must have shape [E, ..., 3].")
    if lifecycle_valid.shape != (tensor.shape[0],):
        raise ValueError("lifecycle_valid must have shape [E].")

    vector_valid = torch.isfinite(tensor).all(dim=-1)
    lifecycle_shape = (tensor.shape[0],) + (1,) * (vector_valid.ndim - 1)
    vector_valid &= lifecycle_valid.view(lifecycle_shape)
    if static_valid is not None:
        if tuple(static_valid.shape) != tuple(vector_valid.shape):
            raise ValueError(
                "Static contact validity shape does not match measurement "
                f"shape: {tuple(static_valid.shape)} versus "
                f"{tuple(vector_valid.shape)}."
            )
        vector_valid &= static_valid

    finite_tensor = torch.nan_to_num(
        tensor, nan=0.0, posinf=0.0, neginf=0.0
    )
    finite_tensor = torch.where(
        vector_valid.unsqueeze(-1), finite_tensor, torch.zeros_like(finite_tensor)
    )
    return finite_tensor, vector_valid


@dataclass(frozen=True)
class ContactFilterMetadata:
    """Static meaning and ordering of pair-resolved contact filter slots.

    ``scene_object_indices`` uses ``None`` for non-object filters (currently the
    terrain) and the scene-object slot for object filters.  The three tuples are
    parallel and are intentionally metadata rather than policy inputs.
    """

    labels: Tuple[str, ...] = ()
    prim_path_exprs: Tuple[str, ...] = ()
    scene_object_indices: Tuple[Optional[int], ...] = ()

    def __post_init__(self) -> None:
        num_filters = len(self.labels)
        if len(self.prim_path_exprs) != num_filters:
            raise ValueError(
                "Contact filter labels and prim path expressions must have the "
                "same length."
            )
        if len(self.scene_object_indices) != num_filters:
            raise ValueError(
                "Contact filter labels and scene object indices must have the "
                "same length."
            )

    @property
    def num_filters(self) -> int:
        return len(self.labels)


@dataclass
class ContactSensorState:
    """Rich contact sensor data in deterministic common body-name order.

    The body axis contains the configured sensor bodies, not necessarily every
    robot rigid body.  ``common_body_indices`` therefore provides the reliable
    mapping into full common :class:`RobotState` tensors.  Aggregate and pair
    force values are world-frame *normal* contact forces.  Friction is exposed
    only through ``friction_force_w``.

    All history tensors are newest-first.  Numeric samples are finite: a
    backend replaces non-finite values with zero and preserves the corresponding
    validity mask so missing data are never confused with a measured zero.
    """

    body_names: Tuple[str, ...]
    sim_body_indices: torch.Tensor
    common_body_indices: torch.Tensor
    filter_metadata: ContactFilterMetadata

    normal_force_w: torch.Tensor
    normal_force_history_w: torch.Tensor
    sensor_data_valid: torch.Tensor
    normal_force_valid: torch.Tensor
    normal_force_history_valid: torch.Tensor

    filtered_normal_force_w: Optional[torch.Tensor] = None
    filtered_normal_force_history_w: Optional[torch.Tensor] = None
    filtered_normal_force_valid: Optional[torch.Tensor] = None
    filtered_normal_force_history_valid: Optional[torch.Tensor] = None

    friction_force_w: Optional[torch.Tensor] = None
    friction_force_valid: Optional[torch.Tensor] = None

    mean_contact_point_w: Optional[torch.Tensor] = None
    mean_contact_point_valid: Optional[torch.Tensor] = None
    pair_slot_valid: Optional[torch.Tensor] = None

    current_contact_time_s: Optional[torch.Tensor] = None
    current_air_time_s: Optional[torch.Tensor] = None
    last_contact_time_s: Optional[torch.Tensor] = None
    last_air_time_s: Optional[torch.Tensor] = None

    body_weight_n: Optional[torch.Tensor] = None
    history_newest_first: bool = True

    def __post_init__(self) -> None:
        if not self.history_newest_first:
            raise ValueError("Contact sensor histories must be newest-first.")

        num_bodies = len(self.body_names)
        if self.sim_body_indices.shape != (num_bodies,):
            raise ValueError(
                "sim_body_indices must have shape [num_sensor_bodies]."
            )
        if self.common_body_indices.shape != (num_bodies,):
            raise ValueError(
                "common_body_indices must have shape [num_sensor_bodies]."
            )
        if self.normal_force_w.ndim != 3 or self.normal_force_w.shape[1:] != (
            num_bodies,
            3,
        ):
            raise ValueError("normal_force_w must have shape [E, K, 3].")

        num_envs = self.normal_force_w.shape[0]
        if (
            self.normal_force_history_w.ndim != 4
            or self.normal_force_history_w.shape[0] != num_envs
            or self.normal_force_history_w.shape[2:] != (num_bodies, 3)
        ):
            raise ValueError(
                "normal_force_history_w must have shape [E, H, K, 3]."
            )
        history_length = self.normal_force_history_w.shape[1]
        self._require_shape(self.sensor_data_valid, (num_envs,), "sensor_data_valid")
        self._require_shape(
            self.normal_force_valid,
            (num_envs, num_bodies),
            "normal_force_valid",
        )
        self._require_shape(
            self.normal_force_history_valid,
            (num_envs, history_length, num_bodies),
            "normal_force_history_valid",
        )

        pair_shape = (num_envs, num_bodies, self.filter_metadata.num_filters)
        pair_history_shape = (
            num_envs,
            history_length,
            num_bodies,
            self.filter_metadata.num_filters,
        )
        self._validate_optional_vector(
            self.filtered_normal_force_w,
            pair_shape,
            "filtered_normal_force_w",
        )
        self._validate_optional_vector(
            self.filtered_normal_force_history_w,
            pair_history_shape,
            "filtered_normal_force_history_w",
        )
        self._validate_optional_vector(
            self.friction_force_w, pair_shape, "friction_force_w"
        )
        self._validate_optional_vector(
            self.mean_contact_point_w, pair_shape, "mean_contact_point_w"
        )

        for tensor, shape, name in (
            (self.filtered_normal_force_valid, pair_shape, "filtered_normal_force_valid"),
            (
                self.filtered_normal_force_history_valid,
                pair_history_shape,
                "filtered_normal_force_history_valid",
            ),
            (self.friction_force_valid, pair_shape, "friction_force_valid"),
            (self.mean_contact_point_valid, pair_shape, "mean_contact_point_valid"),
            (self.pair_slot_valid, pair_shape, "pair_slot_valid"),
        ):
            if tensor is not None:
                self._require_shape(tensor, shape, name)

        timer_shape = (num_envs, num_bodies)
        for tensor, name in (
            (self.current_contact_time_s, "current_contact_time_s"),
            (self.current_air_time_s, "current_air_time_s"),
            (self.last_contact_time_s, "last_contact_time_s"),
            (self.last_air_time_s, "last_air_time_s"),
        ):
            if tensor is not None:
                self._require_shape(tensor, timer_shape, name)

        if self.body_weight_n is not None:
            self._require_shape(self.body_weight_n, (num_envs, 1), "body_weight_n")

    @staticmethod
    def _require_shape(
        tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str
    ) -> None:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {list(expected_shape)}, got "
                f"{list(tensor.shape)}."
            )

    @classmethod
    def _validate_optional_vector(
        cls,
        tensor: Optional[torch.Tensor],
        leading_shape: Tuple[int, ...],
        name: str,
    ) -> None:
        if tensor is not None:
            cls._require_shape(tensor, leading_shape + (3,), name)
