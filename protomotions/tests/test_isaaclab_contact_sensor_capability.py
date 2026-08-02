# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused pure tests for the optional IsaacLab contact sensor contract."""

import io

import pytest
import torch

from protomotions.simulator.base_simulator.contact_sensor_state import (
    ContactFilterMetadata,
    ContactSensorState,
    sanitize_contact_vectors,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabContactSensorObservationCfg,
    IsaacLabSimParams,
    IsaacLabSimulatorConfig,
    build_contact_filter_metadata,
    build_humanoid_contact_sensor_kwargs,
)


def _simulator_config(contact_cfg=None):
    return IsaacLabSimulatorConfig(
        headless=True,
        num_envs=2,
        sim=IsaacLabSimParams(fps=60, decimation=4),
        experiment_name="contact-capability-unit",
        contact_sensor_observation=(
            contact_cfg
            if contact_cfg is not None
            else IsaacLabContactSensorObservationCfg()
        ),
    )


def _contact_state(**overrides):
    num_envs, history, num_bodies, num_filters = 2, 4, 2, 2
    kwargs = {
        "body_names": ("pelvis", "left_foot"),
        "sim_body_indices": torch.tensor([1, 0]),
        "common_body_indices": torch.tensor([0, 3]),
        "filter_metadata": ContactFilterMetadata(
            labels=("terrain", "scene_object_0"),
            prim_path_exprs=(
                "/World/ground/terrain/mesh",
                "/World/envs/env_.*/Object_0",
            ),
            scene_object_indices=(None, 0),
        ),
        "normal_force_w": torch.zeros(num_envs, num_bodies, 3),
        "normal_force_history_w": torch.zeros(
            num_envs, history, num_bodies, 3
        ),
        "sensor_data_valid": torch.tensor([True, False]),
        "normal_force_valid": torch.ones(
            num_envs, num_bodies, dtype=torch.bool
        ),
        "normal_force_history_valid": torch.ones(
            num_envs, history, num_bodies, dtype=torch.bool
        ),
        "filtered_normal_force_w": torch.zeros(
            num_envs, num_bodies, num_filters, 3
        ),
        "filtered_normal_force_history_w": torch.zeros(
            num_envs, history, num_bodies, num_filters, 3
        ),
        "filtered_normal_force_valid": torch.ones(
            num_envs, num_bodies, num_filters, dtype=torch.bool
        ),
        "filtered_normal_force_history_valid": torch.ones(
            num_envs, history, num_bodies, num_filters, dtype=torch.bool
        ),
        "friction_force_w": torch.zeros(
            num_envs, num_bodies, num_filters, 3
        ),
        "friction_force_valid": torch.ones(
            num_envs, num_bodies, num_filters, dtype=torch.bool
        ),
        "mean_contact_point_w": torch.zeros(
            num_envs, num_bodies, num_filters, 3
        ),
        "mean_contact_point_valid": torch.zeros(
            num_envs, num_bodies, num_filters, dtype=torch.bool
        ),
        "pair_slot_valid": torch.ones(
            num_envs, num_bodies, num_filters, dtype=torch.bool
        ),
        "current_contact_time_s": torch.zeros(num_envs, num_bodies),
        "current_air_time_s": torch.zeros(num_envs, num_bodies),
        "last_contact_time_s": torch.zeros(num_envs, num_bodies),
        "last_air_time_s": torch.zeros(num_envs, num_bodies),
        "body_weight_n": torch.tensor([[600.0], [700.0]]),
    }
    kwargs.update(overrides)
    return ContactSensorState(**kwargs)


def test_isaaclab_contact_sensor_config_is_opt_in_and_validated():
    config = _simulator_config()
    assert config.contact_sensor_observation.enabled is False
    assert config.contact_sensor_observation.track_pair_data is False
    assert config.contact_sensor_observation.max_contact_data_count_per_prim == 16

    for kwargs, message in (
        ({"max_contact_data_count_per_prim": 0}, "at least one"),
        ({"force_threshold_n": -0.1}, "non-negative"),
        ({"track_contact_points": True}, "track_pair_data=True"),
        ({"track_friction_forces": True}, "track_pair_data=True"),
        (
            {
                "track_pair_data": True,
                "include_terrain_filter": False,
                "include_scene_object_filters": False,
            },
            "at least one enabled filter",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            IsaacLabContactSensorObservationCfg(**kwargs)

    mutated = IsaacLabContactSensorObservationCfg()
    mutated.track_contact_points = True
    with pytest.raises(ValueError, match="track_pair_data=True"):
        build_contact_filter_metadata(
            _simulator_config(mutated),
            num_scene_objects=0,
            terrain_available=True,
        )


def test_filter_metadata_is_terrain_then_scene_objects_and_opt_in():
    disabled = build_contact_filter_metadata(
        _simulator_config(), num_scene_objects=2, terrain_available=True
    )
    assert disabled == ContactFilterMetadata()

    contact_cfg = IsaacLabContactSensorObservationCfg(
        enabled=True,
        track_pair_data=True,
        track_contact_points=True,
        track_friction_forces=True,
    )
    metadata = build_contact_filter_metadata(
        _simulator_config(contact_cfg),
        num_scene_objects=2,
        terrain_available=True,
    )
    assert metadata.labels == ("terrain", "scene_object_0", "scene_object_1")
    assert metadata.scene_object_indices == (None, 0, 1)
    assert metadata.prim_path_exprs == (
        "/World/ground/terrain/mesh",
        "/World/envs/env_.*/Object_0",
        "/World/envs/env_.*/Object_1",
    )
    assert contact_cfg.resolved_filter_labels == metadata.labels
    assert contact_cfg.resolved_filter_prim_path_exprs == metadata.prim_path_exprs
    assert (
        contact_cfg.resolved_filter_scene_object_indices
        == metadata.scene_object_indices
    )
    assert contact_cfg.resolved_num_filters == 3

    with pytest.raises(ValueError, match="checkpoint/config contract"):
        build_contact_filter_metadata(
            _simulator_config(contact_cfg),
            num_scene_objects=1,
            terrain_available=True,
        )

    checkpoint_buffer = io.BytesIO()
    torch.save(contact_cfg, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    restored_cfg = torch.load(
        checkpoint_buffer, map_location="cpu", weights_only=False
    )
    assert restored_cfg.resolved_filter_labels == metadata.labels
    assert restored_cfg.resolved_filter_prim_path_exprs == metadata.prim_path_exprs
    assert restored_cfg.resolved_filter_scene_object_indices == (None, 0, 1)
    assert restored_cfg.resolved_num_filters == 3

    objects_only_cfg = IsaacLabContactSensorObservationCfg(
        enabled=True,
        track_pair_data=True,
        include_terrain_filter=False,
    )
    objects_only = build_contact_filter_metadata(
        _simulator_config(objects_only_cfg),
        num_scene_objects=1,
        terrain_available=False,
    )
    assert objects_only.labels == ("scene_object_0",)

    with pytest.raises(ValueError, match="no contact filters resolved"):
        build_contact_filter_metadata(
            _simulator_config(objects_only_cfg),
            num_scene_objects=0,
            terrain_available=False,
        )


def test_pre_feature_pickled_config_is_upgraded_as_disabled():
    legacy_config = _simulator_config()
    del legacy_config.__dict__["contact_sensor_observation"]
    checkpoint_buffer = io.BytesIO()
    torch.save(legacy_config, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    restored = torch.load(
        checkpoint_buffer, map_location="cpu", weights_only=False
    )
    assert not hasattr(restored, "contact_sensor_observation")

    metadata = build_contact_filter_metadata(
        restored, num_scene_objects=1, terrain_available=True
    )

    assert metadata == ContactFilterMetadata()
    assert restored.contact_sensor_observation.enabled is False


def test_sanitize_contact_vectors_preserves_explicit_validity():
    vectors = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0], [float("nan"), 2.0, 3.0]],
                [[4.0, 5.0, 6.0], [7.0, 8.0, float("inf")]],
            ],
            [
                [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                [[15.0, 16.0, 17.0], [18.0, 19.0, 20.0]],
            ],
        ]
    )
    lifecycle_valid = torch.tensor([True, False])
    static_valid = torch.tensor(
        [
            [[True, True], [False, True]],
            [[True, True], [True, True]],
        ]
    )

    clean, valid = sanitize_contact_vectors(
        vectors, lifecycle_valid, static_valid
    )

    assert valid.tolist() == [
        [[True, False], [False, False]],
        [[False, False], [False, False]],
    ]
    assert torch.equal(clean[0, 0, 0], vectors[0, 0, 0])
    assert torch.equal(clean[~valid], torch.zeros_like(clean[~valid]))
    assert torch.isfinite(clean).all()

    with pytest.raises(ValueError, match=r"shape \[E"):
        sanitize_contact_vectors(torch.zeros(2, 4), lifecycle_valid)
    with pytest.raises(ValueError, match="lifecycle_valid"):
        sanitize_contact_vectors(torch.zeros(2, 3), torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError, match="Static contact validity"):
        sanitize_contact_vectors(
            torch.zeros(2, 1, 3),
            lifecycle_valid,
            torch.ones(2, 2, dtype=torch.bool),
        )


def test_sensor_kwargs_preserve_legacy_and_enable_only_requested_features():
    legacy_filters = ("/terrain", "/object")
    disabled = build_humanoid_contact_sensor_kwargs(
        _simulator_config(),
        prim_path="/Robot/foot",
        legacy_filter_prim_path_exprs=legacy_filters,
        filter_metadata=ContactFilterMetadata(),
    )
    assert disabled == {
        "prim_path": "/Robot/foot",
        "filter_prim_paths_expr": list(legacy_filters),
        "history_length": 4,
    }

    aggregate_cfg = IsaacLabContactSensorObservationCfg(enabled=True)
    aggregate = build_humanoid_contact_sensor_kwargs(
        _simulator_config(aggregate_cfg),
        prim_path="/Robot/foot",
        legacy_filter_prim_path_exprs=legacy_filters,
        filter_metadata=ContactFilterMetadata(),
    )
    assert aggregate["filter_prim_paths_expr"] == []
    assert aggregate["track_air_time"] is True
    assert aggregate["force_threshold"] == 2.0
    assert "track_contact_points" not in aggregate
    assert "track_friction_forces" not in aggregate
    assert "max_contact_data_count_per_prim" not in aggregate

    pair_cfg = IsaacLabContactSensorObservationCfg(
        enabled=True,
        track_pair_data=True,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=32,
        include_scene_object_filters=False,
    )
    pair_metadata = build_contact_filter_metadata(
        _simulator_config(pair_cfg),
        num_scene_objects=0,
        terrain_available=True,
    )
    pair = build_humanoid_contact_sensor_kwargs(
        _simulator_config(pair_cfg),
        prim_path="/Robot/foot",
        legacy_filter_prim_path_exprs=legacy_filters,
        filter_metadata=pair_metadata,
    )
    assert pair["filter_prim_paths_expr"] == ["/World/ground/terrain/mesh"]
    assert pair["track_contact_points"] is True
    assert pair["track_friction_forces"] is True
    assert pair["max_contact_data_count_per_prim"] == 32


def test_contact_sensor_state_shape_contract_and_metadata():
    state = _contact_state()

    assert state.body_names == ("pelvis", "left_foot")
    assert state.normal_force_w.shape == (2, 2, 3)
    assert state.normal_force_history_w.shape == (2, 4, 2, 3)
    assert state.filtered_normal_force_w.shape == (2, 2, 2, 3)
    assert state.filtered_normal_force_history_w.shape == (2, 4, 2, 2, 3)
    assert state.history_newest_first is True
    assert state.filter_metadata.num_filters == 2

    with pytest.raises(ValueError, match="normal_force_w"):
        _contact_state(normal_force_w=torch.zeros(2, 3, 3))
    with pytest.raises(ValueError, match="newest-first"):
        _contact_state(history_newest_first=False)
    with pytest.raises(ValueError, match="friction_force_valid"):
        _contact_state(friction_force_valid=torch.ones(2, 2, dtype=torch.bool))


def test_filter_metadata_rejects_misaligned_parallel_fields():
    with pytest.raises(ValueError, match="prim path"):
        ContactFilterMetadata(
            labels=("terrain",),
            prim_path_exprs=(),
            scene_object_indices=(None,),
        )
    with pytest.raises(ValueError, match="scene object"):
        ContactFilterMetadata(
            labels=("terrain",),
            prim_path_exprs=("/terrain",),
            scene_object_indices=(),
        )
