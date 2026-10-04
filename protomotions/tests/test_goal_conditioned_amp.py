# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GoalConditionedAMP: weight schedule, demonstration set, one-shot weight calibration (PLAN.MD card E2)."""

from types import SimpleNamespace

import pytest

from protomotions.agents.amp.goal_conditioned import (
    GoalConditionedAMPComponent,
    amp_reward_weight,
    demo_motion_mask,
)


def test_schedule_is_zero_then_linear_then_flat():
    w = [amp_reward_weight(e, 0.3, 200, 500) for e in (0, 199, 200, 350, 499, 500, 4999)]
    assert w[0] == w[1] == w[2] == 0.0
    assert w[3] == pytest.approx(0.15)
    assert w[4] == pytest.approx(0.3 * 299 / 300)
    assert w[5] == w[6] == pytest.approx(0.3)
    assert amp_reward_weight(10, 0.2, 5, 5) == pytest.approx(0.2)        # degenerate ramp = step


def test_demonstrations_are_x0_clips_without_scorpion_b():
    stems = ["220923_Tree_Pose_or_Vrksasana_-a", "220923_Tree_Pose_or_Vrksasana_-a_x3s",
             "220923_Tree_Pose_or_Vrksasana_-a_x7s", "220923_Scorpion_pose_or_vrischikasana-b",
             "220923_Scorpion_pose_or_vrischikasana-a"]
    keep = demo_motion_mask(stems, r"_x\d+s$", ["Scorpion_pose_or_vrischikasana-b"])
    assert keep == [True, False, False, False, True]


def _component(ratio=0.25, start=200, window=100, target=0.1, w_min=0.05, w_max=0.5):
    comp = object.__new__(GoalConditionedAMPComponent)
    cfg = SimpleNamespace(amp_calibrate_style_ratio=ratio, amp_reward_w_start_epoch=start,
                          amp_calibrate_window=window, amp_reward_w_target=target,
                          amp_reward_w_min=w_min, amp_reward_w_max=w_max)
    comp.agent = SimpleNamespace(config=cfg)
    comp.ratio_history, comp.calibrated_target = [], None
    return comp


def test_calibration_freezes_ratio_over_median_at_the_start_epoch():
    comp = _component()
    comp.ratio_history = [[e, 0.75] for e in range(0, 100)] + [[e, 0.8] for e in range(100, 200)]
    comp._maybe_calibrate(199)                        # before the start epoch: nothing
    assert comp.calibrated_target is None and comp.target_w() == pytest.approx(0.1)
    comp._maybe_calibrate(200)                        # median over epochs 100-199 only
    assert comp.calibrated_target == pytest.approx(0.25 / 0.8)
    comp.ratio_history.append([200, 10.0])
    comp._maybe_calibrate(201)                        # frozen: never recomputed
    assert comp.target_w() == pytest.approx(0.25 / 0.8)


def test_calibration_clamps_and_can_be_disabled():
    lo = _component()
    lo.ratio_history = [[e, 100.0] for e in range(100, 200)]
    lo._maybe_calibrate(200)
    assert lo.calibrated_target == pytest.approx(0.05)
    hi = _component()
    hi.ratio_history = [[e, 0.01] for e in range(100, 200)]
    hi._maybe_calibrate(250)                          # also fires when first seen after the start (resume)
    assert hi.calibrated_target == pytest.approx(0.5)
    off = _component(ratio=0.0)
    off.ratio_history = [[e, 0.01] for e in range(100, 200)]
    off._maybe_calibrate(300)
    assert off.calibrated_target is None and off.target_w() == pytest.approx(0.1)


def test_calibration_without_history_falls_back_to_the_configured_target():
    comp = _component(target=0.12)
    comp._maybe_calibrate(400)
    assert comp.calibrated_target == pytest.approx(0.12)
