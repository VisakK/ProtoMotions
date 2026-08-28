# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the online contact-event history tracker."""

from __future__ import annotations

import pytest
import torch

from protomotions.envs.control.contact_event_tracker import (
    NUM_ORIENT_BINS,
    ContactEventTracker,
    orientation_scores,
)
from protomotions.envs.obs.contact_goal import (
    compute_contact_history_masks,
    compute_contact_history_obs,
)

P = 6           # pair-vocabulary width used by these tests
DT = 1.0 / 30.0
UPRIGHT = torch.tensor([0.0, 0.0, 0.0, 1.0])          # identity, xyzw
INVERTED = torch.tensor([1.0, 0.0, 0.0, 0.0])         # 180 deg about x, xyzw


def make_tracker(num_envs=2, num_events=4, min_dwell_s=0.3):
    return ContactEventTracker(
        num_envs=num_envs,
        num_pairs=P,
        num_events=num_events,
        min_dwell_s=min_dwell_s,
        time_clip_s=10.0,
        orient_margin=0.15,
        device=torch.device("cpu"),
    )


def contacts(*on, num_envs=2):
    v = torch.zeros(num_envs, P, dtype=torch.bool)
    for pair in on:
        v[:, pair] = True
    return v


def run(tracker, contact, rot, steps):
    for _ in range(steps):
        tracker.update(contact, rot.expand(contact.shape[0], 4), DT)


def test_orientation_scores_pick_the_right_bins():
    scores = orientation_scores(torch.stack([UPRIGHT, INVERTED]))
    assert scores.shape == (2, NUM_ORIENT_BINS)
    assert int(scores[0].argmax()) == 0  # upright
    assert int(scores[1].argmax()) == 1  # inverted


def test_first_update_opens_a_segment_without_advancing_time():
    tracker = make_tracker()
    tracker.update(contacts(0), UPRIGHT.expand(2, 4), DT)
    features, valid = tracker.features()

    assert valid[:, 0].all() and not valid[:, 1:].any()
    assert features[0, 0, 0] == 1.0                       # contact pair 0
    assert features[0, 0, P] == 1.0                       # upright one-hot
    assert features[0, 0, P + NUM_ORIENT_BINS + 2] == 1.0  # is_open
    assert float(tracker.time[0]) == 0.0                  # fresh rows start at 0
    # Invalid slots are all-zero, like a hidden goal slot.
    assert features[:, 1:].abs().sum() == 0.0


def test_flicker_shorter_than_dwell_never_commits():
    tracker = make_tracker()
    run(tracker, contacts(0), UPRIGHT, 10)
    run(tracker, contacts(1), UPRIGHT, 5)   # 0.167 s < 0.3 s dwell
    run(tracker, contacts(0), UPRIGHT, 10)

    features, valid = tracker.features()
    assert not valid[:, 1:].any()                       # nothing committed
    assert bool(tracker.open_contact[0, 0])             # still the original
    assert float(tracker.open_since[0]) == 0.0          # segment never closed


def test_persistent_change_commits_one_event_with_the_boundary_at_first_sight():
    tracker = make_tracker()
    run(tracker, contacts(0), UPRIGHT, 10)   # t reaches 9*dt
    run(tracker, contacts(1), UPRIGHT, 12)   # candidate at t=10*dt, commits at +0.3 s

    features, valid = tracker.features()
    assert valid[:, 0].all() and valid[:, 1].all() and not valid[:, 2:].any()
    # Closed event is the original configuration, ending when the NEW one first
    # appeared (10*dt), not when it survived the dwell check.
    assert bool(tracker.closed_contact[0, 0, 0])
    assert float(tracker.closed_start[0, 0]) == pytest.approx(0.0)
    assert float(tracker.closed_end[0, 0]) == pytest.approx(10 * DT, abs=1e-6)
    assert bool(tracker.open_contact[0, 1])
    assert float(tracker.open_since[0]) == pytest.approx(10 * DT, abs=1e-6)
    # Feature times are clip-scaled: dwell of the closed event = 10*dt / 10 s.
    dwell = features[0, 1, P + NUM_ORIENT_BINS + 1]
    assert float(dwell) == pytest.approx(10 * DT / 10.0, abs=1e-6)


def test_ring_keeps_newest_first_and_drops_the_oldest():
    tracker = make_tracker(num_events=4)
    for pair in (0, 1, 2, 3, 4):            # A,B,C,D held 0.5 s each, E open
        run(tracker, contacts(pair), UPRIGHT, 15)

    _features, valid = tracker.features()
    assert valid.all()
    assert bool(tracker.open_contact[0, 4])
    # Newest-first ring of the last three closed segments: D, C, B (A dropped).
    assert bool(tracker.closed_contact[0, 0, 3])
    assert bool(tracker.closed_contact[0, 1, 2])
    assert bool(tracker.closed_contact[0, 2, 1])


def test_orientation_change_alone_is_an_event():
    tracker = make_tracker()
    run(tracker, contacts(0), UPRIGHT, 15)
    run(tracker, contacts(0), INVERTED, 15)  # same pairs, flipped trunk

    _features, valid = tracker.features()
    assert valid[:, 1].all()
    assert int(tracker.closed_orient[0, 0]) == 0   # upright closed
    assert int(tracker.open_orient[0]) == 1        # inverted open


def test_reset_clears_only_the_given_rows():
    tracker = make_tracker()
    run(tracker, contacts(0), UPRIGHT, 15)
    run(tracker, contacts(1), UPRIGHT, 15)
    tracker.reset(torch.tensor([0]))

    features, valid = tracker.features()
    assert not valid[0].any() and features[0].abs().sum() == 0.0
    assert valid[1, 0] and valid[1, 1]

    # The cleared row re-opens from the next observation at t=0.
    tracker.update(contacts(2), UPRIGHT.expand(2, 4), DT)
    features, valid = tracker.features()
    assert valid[0, 0] and not valid[0, 1:].any()
    assert bool(tracker.open_contact[0, 2])
    assert float(tracker.time[0]) == 0.0
    # The untouched row kept its history and its clock.
    assert valid[1, 1] and float(tracker.time[1]) > 0.0


def test_initialize_fresh_is_idempotent_and_never_advances():
    tracker = make_tracker()
    tracker.initialize_fresh(contacts(0), UPRIGHT.expand(2, 4))
    tracker.initialize_fresh(contacts(1), UPRIGHT.expand(2, 4))  # no-op: not fresh

    assert bool(tracker.open_contact[0, 0]) and not bool(tracker.open_contact[0, 1])
    assert float(tracker.time[0]) == 0.0
    _features, valid = tracker.features()
    assert valid[:, 0].all() and not valid[:, 1:].any()


def test_history_obs_kernels_flatten_and_cast():
    tracker = make_tracker()
    run(tracker, contacts(0), UPRIGHT, 15)
    run(tracker, contacts(1), UPRIGHT, 15)
    features, valid = tracker.features()

    flat = compute_contact_history_obs(features)
    assert flat.shape == (2, tracker.num_events * tracker.feature_size)
    masks = compute_contact_history_masks(valid)
    assert masks.dtype == torch.float32 and masks.shape == (2, tracker.num_events)
    assert masks[:, 0].eq(1.0).all() and masks[:, 2:].eq(0.0).all()
