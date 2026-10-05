# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Departure anchoring (``segment_end_prob``), graph_growth PLAN.MD card E3.

The synthetic graph is ``test_contact_graph_motion_manager``'s. The regression
tests against the code before E3 load that code from git (the pre-E3 commit, so
they keep meaning the same thing after E3 is committed) and run it beside the new
code on G1's frozen config, release v2's graph and its packed library.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from protomotions.envs.motion_manager.config import ContactGraphMotionManagerConfig
from protomotions.envs.motion_manager.contact_graph_motion_manager import (
    KIND_ARRIVAL,
    KIND_DEPARTURE,
    KIND_T0,
    KIND_UNIFORM,
    START_KINDS,
    ContactGraphMotionManager,
)
from protomotions.tests.test_contact_graph_motion_manager import _MotionLib, _write_graph

REPO = Path(__file__).resolve().parents[2]
# HEAD when card E3 was built: the code every "unchanged with the flags off" test compares against.
PRE_E3_COMMIT = "5fdbd50f115d2e8da666c683a475d69c3ac43045"
G1_DIR = REPO / "results" / "smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac"
RELEASE_V2 = REPO / "data" / "smpl" / "reference_curation" / "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
ENV_DT = 1.0 / 30.0


def module_at_pre_e3_commit(rel_path: str, name: str, tmp_path: Path):
    """Import ``rel_path`` as it was at PRE_E3_COMMIT (skips when git cannot show it)."""
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_E3_COMMIT}:{rel_path}"], cwd=REPO,
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git cannot show {rel_path} at {PRE_E3_COMMIT[:7]}")
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses resolve string annotations through it
    spec.loader.exec_module(module)
    return module


def _manager(tmp_path, num_envs=4096, env_dt=ENV_DT, motion_lib=None, **overrides):
    kwargs = dict(init_start_prob=0.0, resample_on_reset=True, graph_file=_write_graph(tmp_path))
    kwargs.update(overrides)
    return ContactGraphMotionManager(
        ContactGraphMotionManagerConfig(**kwargs), num_envs=num_envs, env_dt=env_dt,
        device=torch.device("cpu"), motion_lib=motion_lib or _MotionLib(),
    )


# --------------------------------------------------------------------------- #
# The departure draw
# --------------------------------------------------------------------------- #
def test_eligibility_excludes_segments_that_end_the_clip(tmp_path):
    # clip_b's only segment ends 1.2 s before its 3.0 s end: eligible. Shorten the
    # clip to the segment's end and the same segment is out.
    manager = _manager(tmp_path, segment_start_prob=0.0, segment_end_prob=1.0)
    eligible = manager._departure_eligible
    assert eligible[0].tolist() == [True, True, True]      # ends 1.5, 5.0, 9.0 in a 10 s clip
    assert eligible[1].tolist() == [True, False, False]
    assert not bool(eligible[2].any())                     # no segments at all
    lib = _MotionLib()
    lib.motion_lengths = torch.tensor([9.0 + ENV_DT / 2, 1.8, 4.0, 3.2], dtype=torch.float)
    short = _manager(tmp_path, motion_lib=lib, segment_start_prob=0.0, segment_end_prob=1.0)
    assert short._departure_eligible[0].tolist() == [True, True, False]   # 9.0 ends within env_dt
    assert short._departure_eligible[1].tolist() == [False, False, False]
    assert short._departure_eligible[3].tolist() == [True, False, False]  # 3.2 == clip end


def test_departure_starts_sit_just_before_an_eligible_segment_end(tmp_path):
    torch.manual_seed(0)
    manager = _manager(tmp_path, segment_start_prob=0.0, segment_end_prob=1.0, pre_roll_s=0.5)
    envs = torch.arange(manager.num_envs)
    manager.sample_motions(envs)
    kind = manager.start_kind
    departed = kind == KIND_DEPARTURE
    has_eligible = manager._departure_eligible[manager.motion_ids].any(dim=-1)
    assert torch.equal(departed, has_eligible)     # p_end = 1: every clip that can, departs
    # How far each start sits before the nearest eligible segment end at or after it.
    ends = manager.seg_end[manager.motion_ids]
    eligible = manager._departure_eligible[manager.motion_ids]
    delta = ends - manager.motion_times.unsqueeze(-1)
    delta = torch.where(eligible & (delta >= -1e-6), delta, torch.full_like(delta, math.inf))
    lead = delta.min(dim=-1).values[departed]
    assert bool((lead <= 0.5 + 1e-5).all())
    lengths = manager._motion_lengths[manager.motion_ids]
    assert bool((manager.motion_times[departed] < lengths[departed] - ENV_DT).all())
    # The offset is spread, not fixed.
    assert float(lead.max() - lead.min()) > 0.4
    # A clip without an eligible segment keeps the time MimicMotionManager drew.
    stay = manager.motion_ids == 2
    assert bool(stay.any()) and bool((kind[stay] == KIND_UNIFORM).all())
    assert float(manager.motion_times[stay].max()) > 1.0


def test_one_uniform_splits_arrival_departure_and_drawn_time(tmp_path):
    torch.manual_seed(1)
    manager = _manager(tmp_path, num_envs=20000, segment_start_prob=0.4, segment_end_prob=0.2,
                       init_start_prob=0.2)
    manager.sample_motions(torch.arange(manager.num_envs))
    kind = manager.start_kind
    on = manager.motion_ids != 2           # clip_c has no segment: never anchored
    shares = [float((kind[on] == k).float().mean()) for k in range(len(START_KINDS))]
    expected = {KIND_ARRIVAL: 0.4, KIND_DEPARTURE: 0.2, KIND_T0: 0.4 * 0.2, KIND_UNIFORM: 0.4 * 0.8}
    for k, p in expected.items():
        assert abs(shares[k] - p) < 0.015, (START_KINDS[k], shares[k], p)
    assert bool((kind[~on] != KIND_ARRIVAL).all()) and bool((kind[~on] != KIND_DEPARTURE).all())
    t0 = kind == KIND_T0
    assert bool((manager.motion_times[t0] == 0.0).all())


def test_zero_end_probability_is_the_arrival_only_stream(tmp_path):
    old = module_at_pre_e3_commit(
        "protomotions/envs/motion_manager/contact_graph_motion_manager.py", "cgmm_pre_e3", tmp_path)
    config = ContactGraphMotionManagerConfig(
        init_start_prob=0.2, resample_on_reset=True, graph_file=_write_graph(tmp_path),
        segment_start_prob=0.6, pre_roll_s=0.5, segment_weighting="rare_node")
    new = ContactGraphMotionManager(config, 512, ENV_DT, torch.device("cpu"), _MotionLib())
    ref = old.ContactGraphMotionManager(config, 512, ENV_DT, torch.device("cpu"), _MotionLib())
    picks = torch.Generator().manual_seed(7)
    for step in range(40):
        envs = torch.nonzero(torch.rand(512, generator=picks) < 0.3).flatten()
        torch.manual_seed(100 + step)
        ref.sample_motions(envs)
        ref_state = torch.get_rng_state()
        torch.manual_seed(100 + step)
        new.sample_motions(envs)
        assert torch.equal(torch.get_rng_state(), ref_state)          # same calls, same count
        assert torch.equal(new.motion_ids, ref.motion_ids)
        assert torch.equal(new.motion_times, ref.motion_times)


def test_departure_cdf_keeps_the_arrival_weights(tmp_path):
    # rare_node counts every live segment, eligible or not; the departure CDF only
    # zeroes the ineligible ones. clip_b's one segment (node 0) ends its 1.8 s clip:
    # node 0 still counts 4, so clip_d's two segments keep the arrival ratio 1/2 : 1.
    manager = _manager(tmp_path, segment_start_prob=0.3, segment_end_prob=0.3, segment_weighting="rare_node")
    lib = _MotionLib()
    lib.motion_lengths = torch.tensor([9.0, 1.8, 4.0, 5.0], dtype=torch.float)
    short = _manager(tmp_path, motion_lib=lib, segment_start_prob=0.3, segment_end_prob=0.3,
                     segment_weighting="rare_node")
    assert not bool(short._departure_eligible[1].any())
    assert torch.equal(short._departure_cdf[3], manager._segment_cdf[3])
    assert torch.allclose(short._departure_cdf[3, :2], torch.tensor([1.0 / 3.0, 1.0]))
    # clip_a's last segment ends its 9 s clip: weight 0, the other two keep theirs.
    depart = torch.diff(short._departure_cdf[0], prepend=torch.zeros(1))
    arrive = torch.diff(manager._segment_cdf[0], prepend=torch.zeros(1))
    assert float(depart[2]) == 0.0
    assert torch.allclose(depart[:2], arrive[:2] / arrive[:2].sum())
    assert torch.equal(short._segment_cdf, manager._segment_cdf)     # arrivals untouched


def test_only_departures_when_the_start_probability_is_zero(tmp_path):
    torch.manual_seed(2)
    manager = _manager(tmp_path, segment_start_prob=0.0, segment_end_prob=0.5)
    manager.sample_motions(torch.arange(manager.num_envs))
    kind = manager.start_kind
    assert not bool((kind == KIND_ARRIVAL).any())
    assert 0.3 < float((kind == KIND_DEPARTURE).float().mean()) < 0.5


# --------------------------------------------------------------------------- #
# The per-step start-kind counts (env/anchor/* through BaseEnv's extras)
# --------------------------------------------------------------------------- #
LOG_KEYS = {f"anchor/{k}_resets" for k in START_KINDS} | {"anchor/resets"}


def test_step_logs_count_the_drawn_kinds_and_reset(tmp_path):
    torch.manual_seed(3)
    manager = _manager(tmp_path, num_envs=1000, segment_start_prob=0.4, segment_end_prob=0.2,
                       init_start_prob=0.2)
    manager.sample_motions(torch.arange(600))
    first = manager.start_kind[:600].clone()
    manager.sample_motions(torch.arange(600, 1000))
    kinds = torch.cat([first, manager.start_kind[600:]])
    logs = manager.pop_step_logs()
    assert set(logs) == LOG_KEYS
    assert float(logs["anchor/resets"]) == 1000.0
    for k, name in enumerate(START_KINDS):
        assert int(logs[f"anchor/{name}_resets"]) == int((kinds == k).sum())
    assert sum(int(logs[f"anchor/{k}_resets"]) for k in START_KINDS) == 1000
    # Popped: a step without resets reports the same keys, all zero (never a missing key).
    again = manager.pop_step_logs()
    assert list(again) == list(logs) and all(float(v) == 0.0 for v in again.values())


def test_the_epoch_ratio_of_the_logged_means_is_the_pooled_share(tmp_path):
    # Through the agent's own path (record_rollout_step's scalar branch into a
    # TensorAverageMeterDict, float16 storage included): the means of
    # anchor/<kind>_resets and anchor/resets over an epoch's steps -- most of them
    # without a reset -- divide to the pooled share exactly, and the key order the
    # meter sees is the same whether or not the first step had a reset.
    from protomotions.agents.utils.metering import TensorAverageMeterDict

    manager = _manager(tmp_path, num_envs=64, segment_start_prob=0.4, segment_end_prob=0.2,
                       init_start_prob=0.2)
    torch.manual_seed(4)
    picks = torch.Generator().manual_seed(5)
    orders = []
    for first_step_resets in (False, True):
        meter, pooled = TensorAverageMeterDict(), torch.zeros(len(START_KINDS))
        for step in range(300):
            envs = torch.nonzero(torch.rand(64, generator=picks) < 0.02).flatten()
            if step == 0:
                envs = torch.arange(8) if first_step_resets else envs[:0]
            manager.sample_motions(envs)
            pooled += torch.bincount(manager.start_kind[envs], minlength=len(START_KINDS)).float()
            meter.add({key: value.float().flatten() for key, value in manager.pop_step_logs().items()})
        orders.append(list(meter.data))
        means = meter.mean()
        assert float(means["anchor/resets"]) * 300 == pytest.approx(float(pooled.sum()), rel=1e-6)
        for k, name in enumerate(START_KINDS):
            ratio = float(means[f"anchor/{name}_resets"]) / float(means["anchor/resets"])
            assert ratio == pytest.approx(float(pooled[k] / pooled.sum()), rel=1e-6)
    assert orders[0] == orders[1]


def test_the_end_probability_is_read_live_like_the_start_probability(tmp_path):
    # Editing the config (as inference-style code does for segment_start_prob)
    # changes departures exactly as it changes arrivals.
    torch.manual_seed(6)
    manager = _manager(tmp_path, segment_start_prob=0.4, segment_end_prob=0.2)
    manager.config.segment_end_prob = 0.0
    manager.sample_motions(torch.arange(manager.num_envs))
    assert not bool((manager.start_kind == KIND_DEPARTURE).any())
    assert bool((manager.start_kind == KIND_ARRIVAL).any())
    manager.config.segment_start_prob, manager.config.segment_end_prob = 0.0, 1.0
    manager.sample_motions(torch.arange(manager.num_envs))
    assert not bool((manager.start_kind == KIND_ARRIVAL).any())
    assert bool((manager.start_kind == KIND_DEPARTURE).any())
    manager.config.segment_end_prob = 0.0          # both off: nothing anchored, nothing logged
    manager.pop_step_logs()
    manager.sample_motions(torch.arange(manager.num_envs))
    assert manager.pop_step_logs() == {}


def test_step_logs_are_empty_when_nothing_is_anchored(tmp_path):
    manager = _manager(tmp_path, segment_start_prob=0.0, segment_end_prob=0.0)
    manager.sample_motions(torch.arange(16))
    assert manager.pop_step_logs() == {}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"segment_start_prob": 0.6, "segment_end_prob": 0.5}, "must be <= 1"),
        ({"segment_end_prob": -0.1}, "segment_end_prob"),
        ({"segment_end_prob": 1.5}, "segment_end_prob"),
    ],
)
def test_invalid_departure_configuration_is_rejected(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        _manager(tmp_path, num_envs=4, **overrides)


def test_sum_exactly_one_is_accepted(tmp_path):
    _manager(tmp_path, num_envs=4, segment_start_prob=0.4, segment_end_prob=0.6)
    _manager(tmp_path, num_envs=4, segment_start_prob=0.7, segment_end_prob=0.3)


def test_old_pickled_configs_read_no_departures(tmp_path):
    config = ContactGraphMotionManagerConfig(graph_file=_write_graph(tmp_path))
    del config.__dict__["segment_end_prob"]          # a config frozen before the field existed
    assert config.segment_end_prob == 0.0
    manager = ContactGraphMotionManager(config, 8, ENV_DT, torch.device("cpu"), _MotionLib())
    assert manager._segment_end_prob == 0.0


# --------------------------------------------------------------------------- #
# Regression on G1's frozen config, release v2's graph and packed library
# --------------------------------------------------------------------------- #
def _g1_config_and_library():
    resolved = G1_DIR / "resolved_configs.pt"
    library = RELEASE_V2 / "motions.pt"
    if not resolved.is_file() or not library.is_file():
        pytest.skip("G1's resolved config or release v2's library is not on this machine")
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    configs = torch.load(resolved, map_location="cpu", weights_only=False)
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(library)), device="cpu")
    return configs, motion_lib


def test_g1_config_samples_exactly_as_before_e3(tmp_path):
    configs, motion_lib = _g1_config_and_library()
    config = configs["env"].motion_manager
    assert "segment_end_prob" not in config.__dict__ and config.segment_end_prob == 0.0
    old = module_at_pre_e3_commit(
        "protomotions/envs/motion_manager/contact_graph_motion_manager.py", "cgmm_pre_e3_g1", tmp_path)
    num_envs = 4096
    new = ContactGraphMotionManager(config, num_envs, ENV_DT, torch.device("cpu"), motion_lib)
    ref = old.ContactGraphMotionManager(config, num_envs, ENV_DT, torch.device("cpu"), motion_lib)
    picks = torch.Generator().manual_seed(11)
    for step in range(60):
        envs = (torch.arange(num_envs) if step == 0
                else torch.nonzero(torch.rand(num_envs, generator=picks) < 0.05).flatten())
        torch.manual_seed(1000 + step)
        ref.sample_motions(envs)
        ref_state = torch.get_rng_state()
        torch.manual_seed(1000 + step)
        new.sample_motions(envs)
        assert torch.equal(torch.get_rng_state(), ref_state)
        assert torch.equal(new.motion_ids, ref.motion_ids)
        assert torch.equal(new.motion_times, ref.motion_times)
    # And G1's own shares: 60 % arrivals, 8 % t = 0, 32 % uniform.
    kinds = new.start_kind
    assert abs(float((kinds == KIND_ARRIVAL).float().mean()) - 0.6) < 0.03
    assert not bool((kinds == KIND_DEPARTURE).any())


def test_base_env_publishes_the_managers_step_logs_in_extras():
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.tests.test_base_env_helpers import _make_env

    def stepped(env):
        env.compute_observations = lambda context: None
        env.compute_reward = lambda context: None
        env.check_resets_and_terminations = lambda context: (torch.zeros(3, dtype=torch.bool),) * 2
        BaseEnv.post_physics_step(env)
        return env.extras

    plain = stepped(_make_env(motions=2, history=True))
    assert not any(k.startswith("anchor/") for k in plain)    # managers without the hook: nothing new
    env = _make_env(motions=2, history=True)
    env.motion_manager.pop_step_logs = lambda: {"anchor/arrival_resets": torch.tensor(2),
                                                "anchor/resets": torch.tensor(5.0)}
    extras = stepped(env)
    assert int(extras["anchor/arrival_resets"]) == 2
    assert float(extras["anchor/resets"]) == 5.0
