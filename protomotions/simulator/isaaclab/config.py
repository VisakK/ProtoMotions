# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for IsaacLab simulator."""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig
from protomotions.simulator.isaacgym.config import IsaacGymPhysXParams
from protomotions.simulator.base_simulator.contact_sensor_state import (
    ContactFilterMetadata,
)
import torch


@dataclass
class IsaacLabContactSensorObservationCfg:
    """Opt-in extraction settings for IsaacLab-rich contact observations.

    This configuration changes only the data retained by humanoid contact
    sensors.  Policy encoding is configured independently by the environment.
    """

    enabled: bool = False
    track_pair_data: bool = False
    track_contact_points: bool = False
    track_friction_forces: bool = False
    track_air_time: bool = True
    force_threshold_n: float = 2.0
    max_contact_data_count_per_prim: int = 16
    include_terrain_filter: bool = True
    include_scene_object_filters: bool = True

    # Populated during scene construction and declared here so resolved config
    # snapshots/checkpoints retain the exact pair-filter contract.
    resolved_filter_labels: Tuple[str, ...] = field(
        default_factory=tuple, init=False
    )
    resolved_filter_prim_path_exprs: Tuple[str, ...] = field(
        default_factory=tuple, init=False
    )
    resolved_filter_scene_object_indices: Tuple[Optional[int], ...] = field(
        default_factory=tuple, init=False
    )
    resolved_num_filters: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate constructor values and any later CLI/experiment mutations."""

        if self.max_contact_data_count_per_prim < 1:
            raise ValueError(
                "max_contact_data_count_per_prim must be at least one."
            )
        if self.force_threshold_n < 0.0:
            raise ValueError("force_threshold_n must be non-negative.")
        if (
            self.track_contact_points or self.track_friction_forces
        ) and not self.track_pair_data:
            raise ValueError(
                "track_contact_points and track_friction_forces require "
                "track_pair_data=True."
            )
        if self.track_pair_data and not (
            self.include_terrain_filter or self.include_scene_object_filters
        ):
            raise ValueError(
                "track_pair_data requires at least one enabled filter category."
            )


@dataclass
class ProtoMotionsIsaacLabMarkers:
    """Configuration for a single marker instance."""

    marker: Any = field(
        default=None,
        metadata={"help": "Marker object reference."}
    )
    scale: torch.Tensor = field(
        default=None,
        metadata={"help": "Marker scale tensor."}
    )


@dataclass
class IsaacLabPhysXParams(IsaacGymPhysXParams):
    """PhysX physics engine parameters with IsaacLab extensions."""

    gpu_found_lost_pairs_capacity: int = field(
        default=2**21,
        metadata={"help": "GPU capacity for found/lost collision pairs."}
    )
    gpu_max_rigid_contact_count: int = field(
        default=2**23,
        metadata={"help": "Maximum GPU rigid contact count."}
    )
    gpu_found_lost_aggregate_pairs_capacity: int = field(
        default=2**25,
        metadata={"help": "GPU capacity for aggregate found/lost pairs."}
    )
    gpu_max_rigid_patch_count: int = field(
        default=5 * 2**15,
        metadata={"help": "Maximum GPU rigid patch count."}
    )


@dataclass
class IsaacLabSimParams(SimParams):
    """PhysX-specific simulation parameters for IsaacLab."""

    physx: IsaacLabPhysXParams = field(
        default_factory=IsaacLabPhysXParams,
        metadata={"help": "PhysX engine parameters."}
    )


@dataclass
class IsaacLabSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacLab simulator."""

    _target_: str = "protomotions.simulator.isaaclab.simulator.IsaacLabSimulator"
    w_last: bool = field(
        default=False,
        metadata={"help": "Quaternion format: False for wxyz (IsaacLab convention)."}
    )
    sim: IsaacLabSimParams = field(
        default_factory=IsaacLabSimParams,
        metadata={"help": "IsaacLab-specific simulation parameters."}
    )
    contact_sensor_observation: IsaacLabContactSensorObservationCfg = field(
        default_factory=IsaacLabContactSensorObservationCfg,
        metadata={
            "help": "Opt-in IsaacLab-rich contact sensor extraction settings."
        },
    )


def get_contact_sensor_observation_cfg(
    config: IsaacLabSimulatorConfig,
) -> IsaacLabContactSensorObservationCfg:
    """Return the rich-contact config, upgrading legacy pickles in place.

    A dataclass ``default_factory`` is not materialized when an instance saved
    by an older class definition is unpickled.  IsaacLab checkpoints created
    before this capability therefore lack the instance attribute.  They retain
    their original behavior by receiving the current feature-disabled default.
    """

    contact_cfg = getattr(config, "contact_sensor_observation", None)
    if contact_cfg is None:
        contact_cfg = IsaacLabContactSensorObservationCfg()
        config.contact_sensor_observation = contact_cfg
    return contact_cfg


def build_contact_filter_metadata(
    config: IsaacLabSimulatorConfig,
    num_scene_objects: int,
    terrain_available: bool,
) -> ContactFilterMetadata:
    """Resolve deterministic terrain-then-object pair-filter metadata."""

    contact_cfg = get_contact_sensor_observation_cfg(config)
    contact_cfg.validate()
    if not (contact_cfg.enabled and contact_cfg.track_pair_data):
        metadata = ContactFilterMetadata()
        # Disabling pair data is an explicit ablation override. Clear any
        # checkpoint-carried pair contract because no pair tensor is live.
        contact_cfg.resolved_filter_labels = ()
        contact_cfg.resolved_filter_prim_path_exprs = ()
        contact_cfg.resolved_filter_scene_object_indices = ()
        contact_cfg.resolved_num_filters = 0
        return metadata

    labels = []
    prim_path_exprs = []
    scene_object_indices = []
    if contact_cfg.include_terrain_filter and terrain_available:
        labels.append("terrain")
        prim_path_exprs.append("/World/ground/terrain/mesh")
        scene_object_indices.append(None)
    if contact_cfg.include_scene_object_filters:
        for object_idx in range(num_scene_objects):
            labels.append(f"scene_object_{object_idx}")
            prim_path_exprs.append(f"/World/envs/env_.*/Object_{object_idx}")
            scene_object_indices.append(object_idx)

    if not labels:
        raise ValueError(
            "IsaacLab pair contact data were enabled, but no contact filters "
            "resolved. Enable an available terrain or scene-object filter."
        )

    metadata = ContactFilterMetadata(
        labels=tuple(labels),
        prim_path_exprs=tuple(prim_path_exprs),
        scene_object_indices=tuple(scene_object_indices),
    )
    _store_resolved_contact_filter_metadata(contact_cfg, metadata)
    return metadata


def build_humanoid_contact_sensor_kwargs(
    config: IsaacLabSimulatorConfig,
    prim_path: str,
    legacy_filter_prim_path_exprs: Tuple[str, ...],
    filter_metadata: ContactFilterMetadata,
) -> dict:
    """Build ContactSensorCfg kwargs without importing the IsaacLab runtime."""

    contact_cfg = get_contact_sensor_observation_cfg(config)
    contact_cfg.validate()
    if contact_cfg.enabled:
        if contact_cfg.track_pair_data and filter_metadata.num_filters < 1:
            raise ValueError(
                "IsaacLab pair contact sensors require non-empty filter metadata."
            )
        sensor_kwargs = {
            "prim_path": prim_path,
            "filter_prim_paths_expr": list(filter_metadata.prim_path_exprs),
            "history_length": config.sim.decimation,
            "update_period": 0.0,
            "track_air_time": contact_cfg.track_air_time,
            "force_threshold": contact_cfg.force_threshold_n,
        }
        if contact_cfg.track_pair_data:
            sensor_kwargs.update(
                track_contact_points=contact_cfg.track_contact_points,
                track_friction_forces=contact_cfg.track_friction_forces,
                max_contact_data_count_per_prim=(
                    contact_cfg.max_contact_data_count_per_prim
                ),
            )
        return sensor_kwargs

    # Exact legacy behavior: do not opt into timer, point, friction, or enlarged
    # contact-data buffers when the rich capability is disabled.
    return {
        "prim_path": prim_path,
        "filter_prim_paths_expr": list(legacy_filter_prim_path_exprs),
        "history_length": config.sim.decimation,
    }


def _store_resolved_contact_filter_metadata(
    contact_cfg: IsaacLabContactSensorObservationCfg,
    metadata: ContactFilterMetadata,
) -> None:
    """Persist the live filter contract in serializable declared config fields."""

    has_saved_contract = bool(
        contact_cfg.resolved_num_filters
        or contact_cfg.resolved_filter_labels
        or contact_cfg.resolved_filter_prim_path_exprs
        or contact_cfg.resolved_filter_scene_object_indices
    )
    if has_saved_contract:
        saved_metadata = ContactFilterMetadata(
            labels=contact_cfg.resolved_filter_labels,
            prim_path_exprs=contact_cfg.resolved_filter_prim_path_exprs,
            scene_object_indices=(
                contact_cfg.resolved_filter_scene_object_indices
            ),
        )
        if (
            saved_metadata != metadata
            or contact_cfg.resolved_num_filters != metadata.num_filters
        ):
            raise ValueError(
                "The live IsaacLab contact filters do not match the resolved "
                "checkpoint/config contract. Saved labels/paths are "
                f"{saved_metadata.labels}/{saved_metadata.prim_path_exprs}; "
                f"live labels/paths are {metadata.labels}/"
                f"{metadata.prim_path_exprs}."
            )

    contact_cfg.resolved_filter_labels = metadata.labels
    contact_cfg.resolved_filter_prim_path_exprs = metadata.prim_path_exprs
    contact_cfg.resolved_filter_scene_object_indices = (
        metadata.scene_object_indices
    )
    contact_cfg.resolved_num_filters = metadata.num_filters
