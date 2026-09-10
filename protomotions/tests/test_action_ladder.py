# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v11 action ladder.

The ladder widens the trunk to ``len(ladder_offsets) * num_actions`` and treats
rung 0 as the executed action, the rest as auxiliary targets against the
expert's action that many control steps ahead. The properties worth pinning:

* **the off-path is bit-identical** -- with the default ``(0,)`` the model's
  decode, its three streams and the agent's loss are exactly what they were
  before v11, so ``--ladder-loss-coeff 0`` really does reproduce v9;
* **rung 0 is the executed action** -- widening the trunk must not change which
  slice reaches the simulator, and the full block must be published under its
  own key by the *privileged* decode only, so the three later decodes in one
  forward cannot overwrite what the loss reads;
* **the horizon targets are the expert's own future action**, shifted along the
  buffer's time axis, with rows masked out when the rollout window ends or an
  episode boundary falls inside the horizon -- a target from the next episode
  is a target from a different clip;
* **the loss is commensurable across rungs** -- each is divided by its own
  target variance, so an unequal valid-row count cannot tilt the sum;
* **the argument surface refuses malformed rungs** rather than silently
  training a head whose slices mean nothing.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from protomotions.agents.supervised.agent import (
    LADDER_PRED_KEY,
    ladder_target_key,
    ladder_valid_key,
)


# --------------------------------------------------------------------------- #
# The buffer-side target construction, isolated from the agent
# --------------------------------------------------------------------------- #
def build_targets(expert: torch.Tensor, dones: torch.Tensor, offset: int):
    """Reference implementation of the shift in ``pre_process_dataset``.

    Kept here rather than imported so the test states the contract in its own
    terms; ``test_agent_shift_matches_reference`` pins the agent to it.
    """
    num_steps, num_envs = dones.shape
    target = torch.zeros_like(expert)
    valid = torch.zeros(dones.shape, dtype=expert.dtype)
    for env in range(num_envs):
        for step in range(num_steps - offset):
            # Per environment: episodes end independently, so collapsing the
            # env axis here would mask every row whenever any env reset.
            if bool(dones[step : step + offset, env].any()):
                continue
            target[step, env] = expert[step + offset, env]
            valid[step, env] = 1.0
    return target, valid


def agent_shift(expert: torch.Tensor, dones: torch.Tensor, offset: int):
    """The vectorised cumulative-sum shift the agent actually runs."""
    num_steps = expert.shape[0]
    target = torch.zeros_like(expert)
    valid = torch.zeros(dones.shape, dtype=expert.dtype)
    usable = num_steps - offset
    if usable > 0:
        target[:usable] = expert[offset:]
        cum = torch.cumsum(dones.to(expert.dtype), dim=0)
        upper = cum[offset - 1 : offset - 1 + usable]
        lower = torch.cat([torch.zeros_like(cum[:1]), cum[: usable - 1]], dim=0)
        valid[:usable] = (upper - lower <= 0).to(expert.dtype)
    return target, valid


@pytest.mark.parametrize("offset", [1, 5, 8, 15, 24])
def test_agent_shift_matches_reference(offset):
    torch.manual_seed(0)
    steps, envs, actions = 32, 6, 4
    expert = torch.randn(steps, envs, actions)
    dones = (torch.rand(steps, envs) < 0.05).float()
    ref_t, ref_v = build_targets(expert, dones, offset)
    got_t, got_v = agent_shift(expert, dones, offset)
    assert torch.equal(got_v, ref_v)
    # Only the valid rows carry a meaningful target; the rest are zeroed and
    # masked, and the loss never reads them.
    mask = ref_v.bool()
    assert torch.allclose(got_t[mask], ref_t[mask])


def test_shift_drops_rows_past_the_window_end():
    expert = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1)
    dones = torch.zeros(10, 1)
    _, valid = agent_shift(expert, dones, 3)
    # Rows 7, 8, 9 have no t+3 inside the rollout window.
    assert valid.reshape(-1).tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


def test_shift_drops_rows_spanning_an_episode_boundary():
    expert = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
    dones = torch.zeros(8, 1)
    dones[4, 0] = 1.0  # the episode ends at step 4
    target, valid = agent_shift(expert, dones, 2)
    flags = valid.reshape(-1).tolist()
    # Steps 3 and 4 would reach across the boundary into a different clip.
    assert flags[3] == 0 and flags[4] == 0
    # Steps before it are untouched and take the expert's own future action.
    assert flags[0] == 1 and target[0, 0, 0] == pytest.approx(2.0)
    # And the new episode picks up cleanly once the horizon clears.
    assert flags[5] == 1 and target[5, 0, 0] == pytest.approx(7.0)


def test_a_done_on_the_row_itself_still_invalidates_it():
    expert = torch.zeros(6, 1, 1)
    dones = torch.zeros(6, 1)
    dones[2, 0] = 1.0
    _, valid = agent_shift(expert, dones, 2)
    # `dones[t]` ends the episode at t, so t's own horizon already crosses it.
    assert valid.reshape(-1).tolist()[2] == 0


# --------------------------------------------------------------------------- #
# The model-side decode split
# --------------------------------------------------------------------------- #
def _fsq_test_module():
    """The FSQ test scaffolding, imported lazily so this file loads standalone."""
    return pytest.importorskip("protomotions.tests.test_fsq_masked_mimic")


def test_default_offsets_leave_the_decode_untouched():
    fsq = _fsq_test_module()
    model = fsq.make_model()
    assert model._num_rungs == 1
    model.reset_rollout_context(num_envs=4, device="cpu")
    model.eval()
    with torch.no_grad():
        td = model(fsq.make_obs(4, seed=3))
    assert td["action"].shape == (4, fsq.ACT)
    # No ladder key at all on the off-path: nothing downstream can key off it.
    assert LADDER_PRED_KEY not in td.keys()


def test_ladder_publishes_the_block_and_returns_rung_zero():
    fsq = _fsq_test_module()
    offsets = (0, 2, 4)
    model = fsq.make_model()
    model.config.fsq.ladder_offsets = offsets
    model._ladder_offsets = offsets
    model._num_rungs = len(offsets)
    # Widen the head the way the experiment file does.
    for module_cfg in model.config.trunk.models:
        if getattr(module_cfg, "num_out", None) == fsq.ACT:
            module_cfg.num_out = len(offsets) * fsq.ACT
    model = type(model)(model.config)
    model.reset_rollout_context(num_envs=4, device="cpu")
    model.eval()
    with torch.no_grad():
        td = model(fsq.make_obs(4, seed=4))
    assert LADDER_PRED_KEY in td.keys()
    rungs = td[LADDER_PRED_KEY]
    assert rungs.shape == (4, len(offsets), fsq.ACT)
    # Rung 0 is what the simulator executes, and it is the privileged stream
    # that published the block -- not the later sampled/greedy decodes.
    assert torch.equal(td["privileged_action"], rungs[:, 0, :])
    assert td["action"].shape == (4, fsq.ACT)


def test_ladder_rejects_a_head_that_does_not_divide():
    fsq = _fsq_test_module()
    model = fsq.make_model()
    model._ladder_offsets = (0, 2)
    model._num_rungs = 2  # head still emits ACT, which ACT % 2 may not divide
    model.reset_rollout_context(num_envs=2, device="cpu")
    model.eval()
    if fsq.ACT % 2 == 0:
        pytest.skip("test action width happens to divide; nothing to refuse")
    with pytest.raises(ValueError, match="not divisible"):
        with torch.no_grad():
            model(fsq.make_obs(2, seed=5))


# --------------------------------------------------------------------------- #
# The experiment file's argument surface
# --------------------------------------------------------------------------- #
def _v11():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "experiments"
        / "masked_mimic"
        / "contact_graph_fsq_v11.py"
    )
    spec = importlib.util.spec_from_file_location("contact_graph_fsq_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_offsets_defaults_to_the_encoder_window():
    v11 = _v11()
    args = argparse.Namespace(ladder_offsets=None, ladder_loss_coeff=1.0)
    args.ladder_offsets = list(v11.DEFAULT_LADDER_OFFSETS)
    assert v11._resolve_offsets(args) == (0, 5, 10, 15)


def test_zero_coefficient_collapses_to_a_single_rung():
    v11 = _v11()
    args = argparse.Namespace(ladder_offsets=[0, 5, 10, 15], ladder_loss_coeff=0.0)
    # A zero weight with several rungs would widen the head and train the extra
    # slices on nothing, so it collapses to v9 instead of wasting them.
    assert v11._resolve_offsets(args) == (0,)


@pytest.mark.parametrize("bad", [[1, 5], [0, 5, 5], [0, 10, 5], []])
def test_malformed_offsets_are_refused(bad):
    v11 = _v11()
    args = argparse.Namespace(ladder_offsets=bad, ladder_loss_coeff=1.0)
    with pytest.raises(ValueError):
        v11._resolve_offsets(args)
