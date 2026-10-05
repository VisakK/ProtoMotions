# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PPOAgentConfig.freeze_obs_normalizers`` (``--freeze-obs-normalizers``), graph_growth PLAN.MD card E3.

CPU only. The last two tests build G1's real AMP model from its frozen config, load its
epoch-5000 weights and statistics, and check that the PPO hook freezes exactly
``_actor.mu.norm`` and ``_critic.norm`` -- the discriminator's and its critic's stay free.
The 10-epoch GPU check is the integration stage's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.common.common import NormObsBase, freeze_obs_normalizers, obs_normalizer_statistics
from protomotions.agents.common.config import NormObsBaseConfig
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.tests.test_e3_departure_anchoring import G1_DIR, REPO

E15500_DIR = REPO / "results" / "smpl_yogi_v2_expert56_a2dda5d2ac"


def _norm(ema_decay=0.999):
    norm = NormObsBase(NormObsBaseConfig(normalize_obs=True, norm_clamp_value=5, norm_ema_decay=ema_decay))
    norm(torch.randn(8, 5))      # materialise the lazy statistics (train mode: one recorded batch)
    return norm


@pytest.mark.parametrize("ema_decay", [0.999, None])
def test_a_frozen_normaliser_keeps_its_statistics_bit_identical(ema_decay):
    torch.manual_seed(0)
    frozen, free = _norm(ema_decay), _norm(ema_decay)
    frozen.train()
    free.train()
    frozen._freeze_running = True
    before_frozen = obs_normalizer_statistics({"n": frozen})
    before_free = obs_normalizer_statistics({"n": free})
    for _ in range(200):
        x = torch.randn(64, 5) * 3.0 + 2.0
        out = frozen(x)
        free(x)
    after_frozen = obs_normalizer_statistics({"n": frozen})
    after_free = obs_normalizer_statistics({"n": free})
    assert set(before_frozen) == {"n.running_obs_norm.mean", "n.running_obs_norm.var", "n.running_obs_norm.count"}
    assert all(torch.equal(before_frozen[k], after_frozen[k]) for k in before_frozen)
    assert not torch.equal(before_free["n.running_obs_norm.mean"], after_free["n.running_obs_norm.mean"])
    # EMA mode never moves count, which is why the check compares mean and var as well.
    if ema_decay is not None:
        assert torch.equal(before_free["n.running_obs_norm.count"], after_free["n.running_obs_norm.count"])
    # Frozen still normalises, with the frozen statistics.
    running = frozen.running_obs_norm
    expected = ((x - running.mean.float()) / torch.sqrt(running.var.float() + running.epsilon)).clamp(-5, 5)
    assert torch.equal(out, expected)


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _norm()
        self.inner = nn.Sequential(nn.Identity())
        self.inner.add_module("extra", _norm())


class _Model(nn.Module):
    """``_actor`` / ``_critic`` like PPOModel's, plus a discriminator-like sibling."""

    def __init__(self):
        super().__init__()
        self._actor = nn.Module()
        self._actor.mu = _Tower()
        self._critic = _Tower()
        self._discriminator = _Tower()


def _agent(model, freeze=True, loaded=True):
    agent = object.__new__(PPO)
    agent.model = model
    agent.config = SimpleNamespace(freeze_obs_normalizers=freeze)
    agent.just_loaded_checkpoint_should_evaluate = loaded
    agent.current_epoch = 1
    return agent


def test_freeze_obs_normalizers_names_every_normaliser_under_the_module():
    model = _Model()
    frozen = freeze_obs_normalizers(model._actor, prefix="_actor")
    assert sorted(frozen) == ["_actor.mu.inner.extra", "_actor.mu.norm"]
    assert all(norm._freeze_running for norm in frozen.values())
    assert not model._critic.norm._freeze_running


def test_the_hook_freezes_actor_and_critic_only_and_verifies(capsys, monkeypatch):
    model = _Model()
    agent = _agent(model)
    agent._before_first_rollout()
    assert sorted(agent._frozen_obs_norms) == [
        "_actor.mu.inner.extra", "_actor.mu.norm", "_critic.inner.extra", "_critic.norm"]
    assert not model._discriminator.norm._freeze_running
    disc_before = obs_normalizer_statistics({"d": model._discriminator.norm})
    model.train()
    for module in (model._actor.mu.norm, model._critic.norm, model._discriminator.norm):
        for _ in range(20):
            module(torch.randn(32, 5) + 4.0)
    assert not torch.equal(disc_before["d.running_obs_norm.mean"],
                           model._discriminator.norm.running_obs_norm.mean)
    # post_epoch_logging runs the check once; BaseAgent's logging itself is stubbed out.
    monkeypatch.setattr(BaseAgent, "post_epoch_logging", lambda self, log: None)
    agent.post_epoch_logging({})
    out = capsys.readouterr().out
    assert "freeze_obs_normalizers: froze" in out and "freeze_obs_normalizers: verified" in out
    assert agent._frozen_obs_norm_stats is None
    agent.post_epoch_logging({})                     # once per launch: nothing more to check
    assert "verified" not in capsys.readouterr().out


def test_the_check_raises_when_the_statistics_moved(monkeypatch):
    model = _Model()
    agent = _agent(model)
    agent._before_first_rollout()
    model._critic.norm._freeze_running = False      # something un-froze it
    model.train()
    model._critic.norm(torch.randn(32, 5) + 4.0)
    monkeypatch.setattr(BaseAgent, "post_epoch_logging", lambda self, log: None)
    with pytest.raises(RuntimeError, match=r"freeze_obs_normalizers: FAILED .*_critic\.norm\.running_obs_norm\.mean"):
        agent.post_epoch_logging({})


def test_a_resume_defers_the_check_to_the_first_policy_update(capsys, monkeypatch):
    # A resume skips its first epoch's update and the post-evaluation epoch's: no
    # training-mode forward has run, so "verified" there would mean nothing. The
    # check keeps comparing through the skipped epochs and verifies after the
    # first epoch that updated the policy.
    model = _Model()
    agent = _agent(model)
    agent._before_first_rollout()
    monkeypatch.setattr(BaseAgent, "post_epoch_logging", lambda self, log: None)
    for epoch in (1, 2):
        agent.current_epoch = epoch
        agent.post_epoch_logging({"skipped_policy_update": 1.0})
        out = capsys.readouterr().out
        assert "verified" not in out and f"epoch {epoch - 1} skipped its policy update" in out
        assert agent._frozen_obs_norm_stats is not None
    model.train()
    model._actor.mu.norm(torch.randn(32, 5) + 4.0)          # frozen: records nothing
    agent.current_epoch = 3
    agent.post_epoch_logging({"epoch": 2})
    out = capsys.readouterr().out
    assert "freeze_obs_normalizers: verified" in out and "(3 epoch(s); epoch counter now 3)" in out
    assert agent._frozen_obs_norm_stats is None


def test_a_change_during_a_skipped_epoch_still_raises(monkeypatch):
    model = _Model()
    agent = _agent(model)
    agent._before_first_rollout()
    monkeypatch.setattr(BaseAgent, "post_epoch_logging", lambda self, log: None)
    model._actor.mu.norm._freeze_running = False
    model.train()
    model._actor.mu.norm(torch.randn(32, 5) + 4.0)
    with pytest.raises(RuntimeError, match=r"freeze_obs_normalizers: FAILED over this launch's first 1 epoch"):
        agent.post_epoch_logging({"skipped_policy_update": 1.0})


def test_default_off_freezes_nothing_and_fresh_runs_warn(capsys):
    model = _Model()
    _agent(model, freeze=False)._before_first_rollout()
    assert not any(m._freeze_running for m in model.modules() if isinstance(m, NormObsBase))
    _agent(_Model(), loaded=False)._before_first_rollout()
    assert "WARNING no checkpoint was loaded" in capsys.readouterr().out


def test_old_pickled_agent_configs_read_no_freeze():
    cfg = object.__new__(PPOAgentConfig)             # unpickling sets __dict__ without __init__
    assert "freeze_obs_normalizers" not in cfg.__dict__ and cfg.freeze_obs_normalizers is False


# --------------------------------------------------------------------------- #
# G1's real model (AMP: actor, critic, discriminator, disc critic) on the CPU
# --------------------------------------------------------------------------- #
def _checkpoint_model_keys(path):
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)["model"]


def test_the_frozen_paths_are_the_checkpoints_normaliser_keys():
    for path, expected in (
        (G1_DIR / "epoch_5000.ckpt", {"_actor.mu.norm", "_critic.norm", "_discriminator.models.0.norm",
                                      "_disc_critic.models.0.norm"}),
        (E15500_DIR / "epoch_15500.ckpt", {"_actor.mu.norm", "_critic.norm"}),
    ):
        keys = _checkpoint_model_keys(path)
        norms = {k[: -len(".running_obs_norm.mean")] for k in keys if k.endswith(".running_obs_norm.mean")}
        assert norms == expected, (path.name, norms)


def test_g1_model_freezes_actor_and_critic_and_leaves_the_discriminator_free(capsys):
    resolved = G1_DIR / "resolved_configs.pt"
    if not resolved.is_file():
        pytest.skip("G1's resolved config is not on this machine")
    from protomotions.agents.utils.normalization import materialize_lazy_running_stats_from_state_dict
    from protomotions.utils.hydra_replacement import get_class

    config = torch.load(resolved, map_location="cpu", weights_only=False)["agent"]
    assert "freeze_obs_normalizers" not in config.__dict__ and config.freeze_obs_normalizers is False
    model = get_class(config.model._target_)(config=config.model)
    state = _checkpoint_model_keys(G1_DIR / "epoch_5000.ckpt")
    materialize_lazy_running_stats_from_state_dict(model, state)
    model.materialize_from_state_dict(state)
    model.load_state_dict(state)

    agent = _agent(model)
    agent._before_first_rollout()
    assert sorted(agent._frozen_obs_norms) == ["_actor.mu.norm", "_critic.norm"]
    assert not model._discriminator.models[0].norm._freeze_running
    assert not model._disc_critic.models[0].norm._freeze_running
    # The frozen statistics are the checkpoint's.
    assert torch.equal(agent._frozen_obs_norm_stats["_actor.mu.norm.running_obs_norm.mean"],
                       state["_actor.mu.norm.running_obs_norm.mean"])

    # Training-mode forwards of the whole model: every key's width is folded into the
    # first one (MLPWithConcat concatenates), which is all the normalisers see.
    def obs(keys, width, batch=64):
        td = {k: torch.zeros(batch, 0) for k in keys}
        td[keys[0]] = torch.randn(batch, width) * 2.0 + 1.0
        return td

    from tensordict import TensorDict

    actor_keys = list(config.model.actor.in_keys)
    widths = {name: state[f"{name}.running_obs_norm.mean"].numel()
              for name in ("_actor.mu.norm", "_critic.norm", "_discriminator.models.0.norm")}
    disc_before = model._discriminator.models[0].norm.running_obs_norm.mean.clone()
    disc_critic_before = model._disc_critic.models[0].norm.running_obs_norm.mean.clone()
    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(5):
            td = obs(actor_keys, widths["_actor.mu.norm"])
            td["mimic_target_poses"] = torch.randn(64, widths["_critic.norm"] - widths["_actor.mu.norm"])
            td["amp_obs"] = torch.randn(64, widths["_discriminator.models.0.norm"])
            model(TensorDict(td, batch_size=64))
    assert not torch.equal(disc_before, model._discriminator.models[0].norm.running_obs_norm.mean)
    assert not torch.equal(disc_critic_before, model._disc_critic.models[0].norm.running_obs_norm.mean)
    agent._verify_frozen_obs_normalizers()
    assert "freeze_obs_normalizers: verified -- 6 statistics buffers" in capsys.readouterr().out
