# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Student XY pose settings must not change a frozen expert's contract."""

import argparse
from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.experiments.masked_mimic import contact_graph_fsq_v9 as experiment
from protomotions.envs.component_factories import mimic_target_poses_max_coords_factory
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.masked_mimic import compute_target_poses_only


@pytest.mark.parametrize("relative_xy", [True, False])
def test_xy_transform_is_student_only_after_expert_copy(monkeypatch, relative_xy):
    dense = mimic_target_poses_max_coords_factory()
    sparse = MdpComponent(compute_func=compute_target_poses_only, dynamic_vars={})
    expert = deepcopy(dense)
    expert.static_params["future_steps"] = 1
    expert_before = deepcopy(expert.static_params)
    dense_bindings = dict(dense.dynamic_vars)
    cfg = SimpleNamespace(observation_components={
        "mimic_target_poses": dense,
        "masked_mimic_target_poses": sparse,
        "expert_mimic_target_poses": expert,
    })
    monkeypatch.setattr(experiment.base, "env_config", lambda robot, args: cfg)

    result = experiment.env_config(
        object(), argparse.Namespace(student_root_relative_xy=relative_xy)
    )

    assert result is cfg
    for key in ("mimic_target_poses", "masked_mimic_target_poses"):
        assert result.observation_components[key].static_params["root_relative_xy"] is relative_xy
    assert expert.static_params == expert_before
    assert "root_relative_xy" not in expert.static_params
    assert dense.dynamic_vars == dense_bindings


@pytest.mark.parametrize("argument, expected", [(None, True), ("False", False)])
def test_v9_xy_flag_is_wired_through_actual_environment(tmp_path, argument, expected):
    from protomotions.tests.test_contact_graph import _StubRobotConfig
    from protomotions.tests.test_fsq_masked_mimic import _fsq_experiment_args

    parser = argparse.ArgumentParser()
    experiment.additional_experiment_arguments(parser)
    argv = [] if argument is None else ["--student-root-relative-xy", argument]
    values = vars(parser.parse_args(argv))
    values.update(vars(_fsq_experiment_args(tmp_path)))
    args = argparse.Namespace(**values)
    robot = _StubRobotConfig()
    experiment.configure_robot_and_simulator(robot, SimpleNamespace(), args)
    cfg = experiment.env_config(robot, args)

    for key in ("mimic_target_poses", "masked_mimic_target_poses"):
        assert cfg.observation_components[key].static_params["root_relative_xy"] is expected
    assert cfg.control_components["contact_graph"].future_steps == [1, 5, 10, 15]
