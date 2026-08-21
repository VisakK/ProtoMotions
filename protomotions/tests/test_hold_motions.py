# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the frozen-reference hold-clip synthesis (make_hold_motions.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "data" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from make_hold_motions import freeze_motion, select_holds  # noqa: E402


def _motion(frames=10):
    return {
        "state_conversion": "COMMON",
        "fps": 60,
        "dof_pos": torch.arange(frames * 3, dtype=torch.float32).reshape(frames, 3),
        "dof_vel": torch.ones(frames, 3),
        "rigid_body_pos": torch.rand(frames, 4, 3),
        "rigid_body_rot": torch.rand(frames, 4, 4),
        "rigid_body_vel": torch.ones(frames, 4, 3),
        "rigid_body_ang_vel": torch.ones(frames, 4, 3),
        "rigid_body_contacts": torch.zeros(frames, 4, dtype=torch.bool),
        "local_rigid_body_rot": torch.rand(frames, 4, 4),
        "ground_reaction": torch.rand(frames, 3),
        "rigid_body_ground_forces": torch.rand(frames, 4, 3),
        "ground_reaction_valid": torch.ones(frames, 3),
    }


def test_freeze_motion_tiles_pose_zeroes_velocity_drops_pressure():
    motion = _motion()
    held = freeze_motion(motion, frame=4, num_frames=7)

    assert held["rigid_body_pos"].shape == (7, 4, 3)
    assert torch.equal(held["rigid_body_pos"][0], motion["rigid_body_pos"][4])
    assert torch.equal(held["rigid_body_pos"][6], motion["rigid_body_pos"][4])
    assert torch.equal(held["dof_pos"][3], motion["dof_pos"][4])
    for key in ("dof_vel", "rigid_body_vel", "rigid_body_ang_vel"):
        assert held[key].shape[0] == 7
        assert held[key].abs().max() == 0
    for key in ("ground_reaction", "rigid_body_ground_forces", "ground_reaction_valid"):
        assert key not in held
    assert held["fps"] == 60
    # The tile is a copy, not a view of the source clip.
    held["rigid_body_pos"][0, 0, 0] += 1.0
    assert not torch.equal(held["rigid_body_pos"][0], motion["rigid_body_pos"][4])


def test_freeze_motion_refuses_unknown_per_frame_fields():
    motion = _motion()
    motion["mystery_channel"] = torch.zeros(10, 2)
    with pytest.raises(ValueError, match="mystery_channel"):
        freeze_motion(motion, frame=0, num_frames=5)


def _segment(config, duration, good=1.0, trusted=True, t_hold=1.0):
    return {
        "config": config,
        "duration_s": duration,
        "good_fraction": good,
        "trusted": trusted,
        "t_hold": t_hold,
    }


def test_select_holds_picks_best_segment_per_node_and_gates():
    graph = {
        "clips": {
            "clip_a": {
                "segments": [
                    _segment("STAND", 3.0, t_hold=1.0),
                    _segment("STAND", 5.0, t_hold=8.0),
                    _segment("CROW", 2.0, t_hold=4.0),
                ]
            },
            "clip_b": {
                "segments": [
                    # Longer than clip_a's best stand but poorly tracked.
                    _segment("STAND", 9.0, good=0.5, t_hold=2.0),
                    # Untrusted segments never become holds.
                    _segment("WHEEL", 30.0, trusted=False),
                ]
            },
        }
    }

    picks = select_holds(graph, min_dwell_s=4.0, min_good=0.9, max_holds=10)

    configs = [p[0] for p in picks]
    assert configs == ["STAND"]  # CROW total dwell 2.0 < 4.0; WHEEL untrusted
    config, dwell, clip, seg = picks[0]
    assert clip == "clip_a" and seg["t_hold"] == 8.0  # the longest clean segment
    assert dwell == pytest.approx(8.0)  # 3.0 + 5.0; the bad clip_b one is excluded


def test_select_holds_caps_and_orders_by_dwell():
    graph = {
        "clips": {
            "c": {
                "segments": [
                    _segment("A", 10.0),
                    _segment("B", 6.0),
                    _segment("C", 8.0),
                ]
            }
        }
    }

    picks = select_holds(graph, min_dwell_s=1.0, min_good=0.9, max_holds=2)

    assert [p[0] for p in picks] == ["A", "C"]
