# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for contact-graph-anchored reference-state initialisation."""

from __future__ import annotations

import math

import pytest
import torch

from protomotions.envs.motion_manager.config import ContactGraphMotionManagerConfig
from protomotions.envs.motion_manager.contact_graph_motion_manager import (
    ContactGraphMotionManager,
)

# Motion 0: three segments. Motion 1: one. Motion 2: none (the graph dropped it
# because the expert was not tracking). Motion 3: two, one of them the only
# occurrence of node 2 in the corpus, which is what 'rare_node' should find.
SEG_START = [
    [0.0, 2.0, 6.0],
    [1.0, math.inf, math.inf],
    [math.inf, math.inf, math.inf],
    [0.5, 3.0, math.inf],
]
SEG_END = [
    [1.5, 5.0, 9.0],
    [1.8, math.inf, math.inf],
    [math.inf, math.inf, math.inf],
    [2.5, 3.2, math.inf],
]
SEG_NODE = [[0, 0, 1], [0, -1, -1], [-1, -1, -1], [0, 2, -1]]
SEG_COUNT = [3, 1, 0, 2]
MOTION_NAMES = ["clip_a", "clip_b", "clip_c", "clip_d"]
MOTION_LENGTHS = [10.0, 3.0, 4.0, 5.0]
NUM_NODES = 3


class _MotionLib:
    def __init__(self):
        self.motion_weights = torch.ones(len(MOTION_NAMES), dtype=torch.float)
        self.motion_lengths = torch.tensor(MOTION_LENGTHS, dtype=torch.float)
        self.motion_files = [f"/data/{name}.motion" for name in MOTION_NAMES]
        self.motion_file = "motions.yaml"

    def num_motions(self):
        return len(MOTION_NAMES)


def _write_graph(tmp_path, motion_names=None, name="contact_graph.pt"):
    """A graph payload; ``motion_names`` shorter than the default truncates the rows.

    Truncating rather than only renaming keeps the payload self-consistent, so a
    library-mismatch test exercises the name check rather than tripping the row
    count check that runs before it.
    """
    names = list(motion_names or MOTION_NAMES)
    rows = len(names)
    hold = [
        [(s + e) / 2 for s, e in zip(row_s, row_e)]
        for row_s, row_e in zip(SEG_START, SEG_END)
    ]
    payload = {
        "motion_names": names,
        "pair_names": ["L_FOOT:G", "R_FOOT:G"],
        "orientation_names": ["upright", "inverted"],
        "node_keys": ["a@upright", "b@upright", "c@inverted"],
        "node_contact": torch.zeros(NUM_NODES, 2),
        "node_orient": torch.zeros(NUM_NODES, dtype=torch.long),
        "seg_node": torch.tensor(SEG_NODE, dtype=torch.long)[:rows],
        "seg_start": torch.tensor(SEG_START)[:rows],
        "seg_end": torch.tensor(SEG_END)[:rows],
        "seg_hold": torch.tensor(hold)[:rows],
        "seg_count": torch.tensor(SEG_COUNT, dtype=torch.long)[:rows],
        "min_lead_s": 0.2,
    }
    path = tmp_path / name
    torch.save(payload, path)
    return str(path)


def _manager(tmp_path, num_envs=256, env_dt=1.0 / 30.0, **overrides):
    config = ContactGraphMotionManagerConfig(
        init_start_prob=0.0,
        resample_on_reset=True,
        graph_file=_write_graph(tmp_path),
        **overrides,
    )
    return ContactGraphMotionManager(
        config,
        num_envs=num_envs,
        env_dt=env_dt,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
    )


def _all_envs(manager):
    return torch.arange(manager.num_envs)


def _distance_to_a_segment(manager) -> torch.Tensor:
    """How far each env's start time sits before its clip's nearest segment."""
    starts = manager.seg_start[manager.motion_ids]
    delta = starts - manager.motion_times.unsqueeze(-1)
    delta = torch.where(delta >= -1e-6, delta, torch.full_like(delta, math.inf))
    return delta.min(dim=-1).values


def test_anchored_starts_sit_just_before_a_segment_boundary(tmp_path):
    torch.manual_seed(0)
    manager = _manager(tmp_path, segment_start_prob=1.0, pre_roll_s=0.5)
    manager.sample_motions(_all_envs(manager))

    has_segments = manager.seg_count[manager.motion_ids] > 0
    distance = _distance_to_a_segment(manager)[has_segments]
    # Every anchored start is at most pre_roll_s before a boundary, and never
    # after one -- an "anchor" that landed mid-hold would defeat the purpose.
    assert torch.all(distance <= 0.5 + 1e-5)
    assert torch.all(distance >= -1e-6)
    # And the offset is spread rather than fixed: a constant pre-roll would put
    # every episode at the same phase relative to the transition.
    assert float(distance.max() - distance.min()) > 0.3


def test_zero_probability_leaves_plain_rsi_untouched(tmp_path):
    torch.manual_seed(0)
    manager = _manager(tmp_path, segment_start_prob=0.0, pre_roll_s=0.5)
    manager.sample_motions(_all_envs(manager))

    torch.manual_seed(0)
    baseline = _manager(tmp_path, segment_start_prob=0.0)
    baseline.sample_motions(_all_envs(baseline))
    assert torch.allclose(manager.motion_times, baseline.motion_times)
    # Uniform RSI spreads over the whole clip, which is what "untouched" means.
    fraction = manager.motion_times / manager._motion_lengths[manager.motion_ids]
    assert float(fraction.max()) > 0.8


def test_partial_probability_mixes_anchored_and_uniform_starts(tmp_path):
    torch.manual_seed(0)
    manager = _manager(tmp_path, num_envs=4096, segment_start_prob=0.5, pre_roll_s=0.25)
    manager.sample_motions(_all_envs(manager))

    has_segments = manager.seg_count[manager.motion_ids] > 0
    anchored = (_distance_to_a_segment(manager) <= 0.25 + 1e-5) & has_segments
    share = float(anchored[has_segments].float().mean())
    # Uniform starts land near a boundary by chance too, so this is a loose band
    # around 0.5 rather than an equality.
    assert 0.45 < share < 0.75


def test_start_times_stay_inside_the_clip(tmp_path):
    torch.manual_seed(0)
    manager = _manager(tmp_path, segment_start_prob=1.0, pre_roll_s=5.0)
    manager.sample_motions(_all_envs(manager))
    lengths = manager._motion_lengths[manager.motion_ids]
    assert torch.all(manager.motion_times >= 0.0)
    assert torch.all(manager.motion_times <= lengths - manager.env_dt + 1e-6)


def test_motions_with_no_segments_are_never_anchored(tmp_path):
    torch.manual_seed(0)
    manager = _manager(tmp_path, num_envs=2048, segment_start_prob=1.0)
    manager.sample_motions(_all_envs(manager))
    # clip_c has no trusted segments; its padded seg_start row is +inf, so an
    # unguarded anchor would produce inf or clamp every episode to the clip end.
    empty = manager.motion_ids == 2
    assert bool(empty.any())
    times = manager.motion_times[empty]
    assert torch.all(torch.isfinite(times))
    assert float(times.max()) > 1.0  # still uniformly spread, not pinned


@pytest.mark.parametrize("weighting", ["uniform", "dwell", "rare_node"])
def test_every_weighting_produces_a_finite_normalised_cdf(tmp_path, weighting):
    manager = _manager(tmp_path, segment_start_prob=1.0, segment_weighting=weighting)
    cdf = manager._segment_cdf
    assert torch.all(torch.isfinite(cdf)), "inf padding leaked into the CDF"
    assert torch.all(cdf[:, 1:] >= cdf[:, :-1] - 1e-6), "CDF is not monotone"
    assert torch.allclose(cdf[:, -1], torch.ones_like(cdf[:, -1]))


def test_rare_node_weighting_prefers_the_least_common_configuration(tmp_path):
    torch.manual_seed(0)
    rare = _manager(
        tmp_path, num_envs=4096, segment_start_prob=1.0, segment_weighting="rare_node"
    )
    uniform = _manager(
        tmp_path, num_envs=4096, segment_start_prob=1.0, segment_weighting="uniform"
    )
    # Within clip_d, segment 1 holds node 2 (one occurrence corpus-wide) and
    # segment 0 holds node 0 (four). rare_node should shift mass onto segment 1.
    def second_segment_share(manager) -> float:
        return float(manager._segment_cdf[3, 1] - manager._segment_cdf[3, 0])

    assert second_segment_share(rare) > second_segment_share(uniform) + 0.05


def test_graph_built_for_another_library_is_refused(tmp_path):
    path = _write_graph(tmp_path, motion_names=["other_a", "other_b"])
    config = ContactGraphMotionManagerConfig(
        init_start_prob=0.0, graph_file=path, segment_start_prob=1.0
    )
    with pytest.raises(ValueError, match="different motion library"):
        ContactGraphMotionManager(
            config,
            num_envs=4,
            env_dt=1.0 / 30.0,
            device=torch.device("cpu"),
            motion_lib=_MotionLib(),
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"graph_file": ""}, "graph_file is required"),
        ({"segment_start_prob": 1.5}, "segment_start_prob"),
        ({"pre_roll_s": -1.0}, "pre_roll_s"),
        ({"segment_weighting": "nonsense"}, "segment_weighting"),
    ],
)
def test_invalid_configuration_is_rejected_at_construction(tmp_path, overrides, match):
    kwargs = {"graph_file": _write_graph(tmp_path)}
    kwargs.update(overrides)
    config = ContactGraphMotionManagerConfig(init_start_prob=0.0, **kwargs)
    with pytest.raises(ValueError, match=match):
        ContactGraphMotionManager(
            config,
            num_envs=4,
            env_dt=1.0 / 30.0,
            device=torch.device("cpu"),
            motion_lib=_MotionLib(),
        )
