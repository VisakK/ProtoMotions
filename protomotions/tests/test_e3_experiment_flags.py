# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card E3's flags through ``mlp_goal_conditioned.py`` (and the AMP experiment built on it).

``--segment-end-prob`` -> ``ContactGraphMotionManagerConfig.segment_end_prob``,
``--motion-prior`` -> ``HoldCurriculumConfig.motion_prior``,
``--freeze-obs-normalizers`` -> ``PPOAgentConfig.freeze_obs_normalizers``. With none given,
G1's own CLI arguments rebuild G1's motion manager, curriculum and freeze setting exactly.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json

import pytest
import torch

from protomotions.tests.test_e3_departure_anchoring import G1_DIR, REPO

EXPERIMENTS = REPO / "examples" / "experiments" / "mimic"
# E6's launcher passes exactly these (PLAN.MD card E6 item 4).
E6_FLAGS = ["--segment-start-prob", "0.4", "--segment-end-prob", "0.2", "--segment-pre-roll-s", "0.5",
            "--motion-prior", "package", "--freeze-obs-normalizers", "True"]


def _experiment(name):
    spec = importlib.util.spec_from_file_location(f"e3_{name}", EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(module, argv):
    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)
    return parser.parse_args(argv)


def test_defaults_are_off_and_e6_flags_parse():
    base = _experiment("mlp_goal_conditioned")
    off = _parse(base, [])
    assert (off.segment_end_prob, off.motion_prior, off.freeze_obs_normalizers) == (0.0, "none", False)
    on = _parse(base, E6_FLAGS)
    assert (on.segment_start_prob, on.segment_end_prob, on.segment_pre_roll_s) == (0.4, 0.2, 0.5)
    assert (on.motion_prior, on.freeze_obs_normalizers) == ("package", True)
    assert _parse(base, ["--freeze-obs-normalizers", "False"]).freeze_obs_normalizers is False
    with pytest.raises(SystemExit):
        _parse(base, ["--motion-prior", "yaml"])


def _g1():
    if not (G1_DIR / "config.yaml").is_file() or not (G1_DIR / "resolved_configs.pt").is_file():
        pytest.skip("G1's run directory is not on this machine")
    args = argparse.Namespace(**json.load(open(G1_DIR / "config.yaml")))
    resolved = torch.load(G1_DIR / "resolved_configs.pt", map_location="cpu", weights_only=False)
    return args, resolved


@pytest.mark.parametrize("name", ["mlp_goal_conditioned", "mlp_goal_conditioned_amp"])
def test_g1_arguments_rebuild_g1s_sampling_curriculum_and_normalisers(name):
    args, resolved = _g1()
    module = _experiment(name)
    env = module.env_config(resolved["robot"], args)
    agent = module.agent_config(resolved["robot"], env, args)
    # Dataclass equality reads the pickled configs' missing new fields as their class defaults.
    assert env.motion_manager == resolved["env"].motion_manager
    assert env.motion_manager.segment_end_prob == 0.0
    assert agent.evaluator.curriculum == resolved["agent"].evaluator.curriculum
    assert agent.evaluator.curriculum.motion_prior == "none"
    assert agent.freeze_obs_normalizers is False


def _train_agent_parser():
    """``train_agent.create_parser()`` without importing train_agent (it parses argv at import)."""
    from protomotions.utils.cli_utils import parse_bool

    path = REPO / "protomotions" / "train_agent.py"
    tree = ast.parse(path.read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_parser"]
    namespace = {"argparse": argparse, "parse_bool": parse_bool}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["create_parser"]()


def test_e6_flags_parse_through_train_agents_parser_and_the_amp_experiment():
    # The parser a real launch builds: train_agent's, extended by the experiment file's.
    required = ["--robot-name", "smpl_yogi_v2", "--simulator", "isaaclab", "--num-envs", "8",
                "--batch-size", "8", "--motion-file", "x.pt", "--experiment-path", "x.py",
                "--experiment-name", "x"]
    for name in ("mlp_goal_conditioned", "mlp_goal_conditioned_amp"):
        parser = _train_agent_parser()
        _experiment(name).additional_experiment_arguments(parser)
        on = parser.parse_args(required + E6_FLAGS)
        assert (on.segment_start_prob, on.segment_end_prob, on.segment_pre_roll_s) == (0.4, 0.2, 0.5), name
        assert (on.motion_prior, on.freeze_obs_normalizers) == ("package", True), name
        off = parser.parse_args(required)
        assert (off.segment_end_prob, off.motion_prior, off.freeze_obs_normalizers) == (0.0, "none", False)


def test_a_motion_prior_without_the_mixture_curriculum_is_refused():
    args, resolved = _g1()
    args.curriculum, args.motion_prior = "legacy", "package"
    base = _experiment("mlp_goal_conditioned")
    env = base.env_config(resolved["robot"], args)
    with pytest.raises(ValueError, match="--motion-prior package needs --curriculum mixture"):
        base.agent_config(resolved["robot"], env, args)


def test_e6_flags_reach_the_amp_experiments_configs():
    args, resolved = _g1()
    amp = _experiment("mlp_goal_conditioned_amp")
    parsed = vars(_parse(amp, E6_FLAGS))
    for key in ("segment_start_prob", "segment_end_prob", "segment_pre_roll_s", "motion_prior",
                "freeze_obs_normalizers"):
        setattr(args, key, parsed[key])
    env = amp.env_config(resolved["robot"], args)
    agent = amp.agent_config(resolved["robot"], env, args)
    manager = env.motion_manager
    assert (manager.segment_start_prob, manager.segment_end_prob, manager.pre_roll_s) == (0.4, 0.2, 0.5)
    assert agent.evaluator.curriculum.motion_prior == "package"
    assert agent.freeze_obs_normalizers is True
    assert type(agent).__name__ == "GoalConditionedAMPAgentConfig"
    # Inference zeroes both anchoring probabilities (every clip starts at t = 0).
    amp.apply_inference_overrides(resolved["robot"], resolved["simulator"], env, agent, None, None, None, args)
    assert (manager.segment_start_prob, manager.segment_end_prob, manager.init_start_prob) == (0.0, 0.0, 1.0)
