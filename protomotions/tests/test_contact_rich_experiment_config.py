# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-level coverage for the opt-in contact-rich Stage-1 expert."""

import argparse
from types import SimpleNamespace

import pytest

from examples.experiments.mimic import mlp_2x_no_contact_rew as baseline
from examples.experiments.mimic import mlp_contact_rich as contact_rich
from protomotions.envs.obs import contact_obs_v1_dim
from protomotions.robot_configs.factory import robot_config


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        motion_file="data/smpl/yoga_yogi_balance_grounded.pt",
        scenes_file=None,
        batch_size=32,
        training_max_steps=256,
    )


def test_smpl_yogi_contact_rich_config_preserves_baseline_except_inputs():
    args = _args()
    robot_cfg = robot_config("smpl_yogi")
    contact_rich.configure_robot_and_simulator(
        robot_cfg,
        SimpleNamespace(),
        args,
    )

    env_cfg = contact_rich.env_config(robot_cfg, args)
    agent_cfg = contact_rich.agent_config(robot_cfg, env_cfg, args)
    baseline_env_cfg = baseline.env_config(robot_cfg, args)
    baseline_agent_cfg = baseline.agent_config(robot_cfg, baseline_env_cfg, args)

    body_names = robot_cfg.kinematic_info.body_names
    assert len(body_names) == 24
    assert robot_cfg.contact_bodies == body_names
    assert robot_cfg.contact_observation_bodies == body_names
    assert robot_cfg.contact_reward_bodies == [
        "L_Ankle",
        "L_Toe",
        "R_Ankle",
        "R_Toe",
    ]

    assert list(env_cfg.observation_components) == contact_rich.CONTACT_INPUT_KEYS
    assert (
        env_cfg.observation_components["max_coords_obs"].static_params[
            "observe_contacts"
        ]
        is False
    )
    body_ids = list(range(24))
    contact_component = env_cfg.observation_components["contact_obs_v1"]
    proximity_component = env_cfg.observation_components["contact_proximity_obs"]
    assert contact_component.static_params["body_ids"].tolist() == body_ids
    assert proximity_component.static_params["body_ids"] == body_ids
    for name, value in contact_rich.CONTACT_OBS_V1_PARAMS.items():
        assert contact_component.static_params[name] == value
    assert contact_obs_v1_dim(len(body_ids)) == 412
    assert len(body_ids) * 3 == 72

    # Every baseline reward component, including the zero-weight contact-match
    # diagnostic, remains exactly the same.
    assert {
        name: component.to_dict()
        for name, component in env_cfg.reward_components.items()
    } == {
        name: component.to_dict()
        for name, component in baseline_env_cfg.reward_components.items()
    }
    assert env_cfg.reward_components["contact_match_rew"].static_params["weight"] == 0.0

    assert env_cfg.contact_force_on_threshold_n == 5.0
    assert env_cfg.contact_force_off_threshold_n == 2.0
    assert env_cfg.contact_diagnostics_interval == 100

    assert agent_cfg.model.in_keys == contact_rich.CONTACT_INPUT_KEYS
    assert agent_cfg.model.actor.in_keys == contact_rich.CONTACT_INPUT_KEYS
    assert agent_cfg.model.actor.mu_model.in_keys == contact_rich.CONTACT_INPUT_KEYS
    assert agent_cfg.model.critic.in_keys == contact_rich.CONTACT_INPUT_KEYS
    assert (
        len(agent_cfg.model.actor.mu_model.layers)
        == len(baseline_agent_cfg.model.actor.mu_model.layers)
        == 6
    )
    assert (
        len(agent_cfg.model.critic.layers)
        == len(baseline_agent_cfg.model.critic.layers)
        == 4
    )
    assert {layer.units for layer in agent_cfg.model.actor.mu_model.layers} == {2048}
    assert {layer.units for layer in agent_cfg.model.critic.layers} == {2048}
    assert agent_cfg.model.actor_optimizer == baseline_agent_cfg.model.actor_optimizer
    assert agent_cfg.model.critic_optimizer == baseline_agent_cfg.model.critic_optimizer
    assert (
        agent_cfg.save_epoch_checkpoint_every
        == baseline_agent_cfg.save_epoch_checkpoint_every
        == 2000
    )


def test_contact_match_rew_normalize_scales_by_body_count():
    """Normalized scoring must not grow with the number of scored bodies."""
    import torch

    from protomotions.envs.rewards import compute_contact_match_rew

    sim = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]])
    ref = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    feet = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    everything = torch.arange(6, dtype=torch.long)

    # Raw counts: scoring more bodies mechanically inflates the penalty.
    assert compute_contact_match_rew(sim, ref, feet).item() == 2.0
    assert compute_contact_match_rew(sim, ref, everything).item() == 3.0

    # Normalized: a mean mismatch fraction, comparable across body sets.
    assert compute_contact_match_rew(sim, ref, feet, normalize=True).item() == 0.5
    assert compute_contact_match_rew(
        sim, ref, everything, normalize=True
    ).item() == pytest.approx(0.5)

    # A perfect match is zero either way; total disagreement saturates at 1.
    assert compute_contact_match_rew(sim, sim, everything, normalize=True).item() == 0.0
    assert compute_contact_match_rew(
        sim, 1.0 - sim, everything, normalize=True
    ).item() == 1.0


def test_contact_match_rew_handles_smoothed_float_reference():
    """Motion libraries smooth contact labels into [0, 1]; the kernel must cope."""
    import torch

    from protomotions.envs.rewards import compute_contact_match_rew

    sim = torch.tensor([[1.0, 0.0]])
    ref = torch.tensor([[0.75, 0.25]])
    ids = torch.arange(2, dtype=torch.long)
    assert compute_contact_match_rew(sim, ref, ids, normalize=True).item() == pytest.approx(0.25)
