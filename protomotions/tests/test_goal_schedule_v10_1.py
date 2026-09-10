# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The v10_1 goal-schedule changes: dwell channels, current segment, promotion.

Every one of these defaults off, and the first thing each group asserts is that
the off-path is **bit-identical** to the pre-v10_1 behaviour -- that is what makes
the changes ablatable, and a silent drift there would invalidate the comparison
the whole round rests on.

Plan: ``notes/V9_crucial_investigations/V10_1_plan.MD``.
"""

import torch

from protomotions.components.contact_graph import ContactGraph
from protomotions.envs.obs.contact_goal import compute_contact_goal_obs


def _graph(num_motions: int = 2, device: str = "cpu") -> ContactGraph:
    """Two motions of three segments each, with an intentional gap in motion 1.

    motion 0: [0.0,1.0] hold 0.5 | [1.0,2.0] hold 1.5 | [2.0,3.0] hold 2.5
    motion 1: [0.0,1.0] hold 0.5 | [2.0,3.0] hold 2.5 | [3.0,4.0] hold 3.5
              -- nothing covers (1.0, 2.0), so `include_current` must fall back.
    """
    inf = float("inf")
    seg_start = torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 3.0]])
    seg_end = torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 4.0]])
    seg_hold = torch.tensor([[0.5, 1.5, 2.5], [0.5, 2.5, 3.5]])
    payload = {
        "motion_names": [f"m{i}" for i in range(num_motions)],
        "pair_names": ["A:G", "B:G"],
        "orientation_names": ["upright", "prone"],
        "node_keys": ["n0", "n1", "n2"],
        "min_lead_s": 0.2,
        "node_contact": torch.eye(3, 2),
        "node_orient": torch.tensor([0, 1, 0]),
        "seg_node": torch.tensor([[0, 1, 2], [0, 2, 1]]),
        "seg_start": seg_start,
        "seg_end": seg_end,
        "seg_hold": seg_hold,
        "seg_count": torch.tensor([3, 3]),
    }
    del inf
    return ContactGraph(payload, device=device)


# --------------------------------------------------------------------------- #
# next_goal_indices
# --------------------------------------------------------------------------- #
def test_include_current_off_is_bit_identical():
    """The legacy path must be untouched -- this is the ablation's foundation."""
    g = _graph()
    ids = torch.tensor([0, 0, 0, 1, 1, 1])
    times = torch.tensor([0.0, 0.9, 1.9, 0.4, 1.5, 2.9])
    legacy_idx, legacy_valid = g.next_goal_indices(ids, times, 3)
    new_idx, new_valid = g.next_goal_indices(ids, times, 3, include_current=False)
    assert torch.equal(legacy_idx, new_idx)
    assert torch.equal(legacy_valid, new_valid)


def test_include_current_is_a_noop_while_the_hold_is_still_ahead():
    """Early in a segment `first` already IS the current segment.

    The change is designed to bite only once the hold has been passed, which is
    the 42.1 %-of-dwell case; anywhere else it must not perturb the schedule.
    """
    g = _graph()
    ids = torch.tensor([0])
    times = torch.tensor([0.1])          # inside segment 0, hold 0.5 still ahead
    off, _ = g.next_goal_indices(ids, times, 3, include_current=False)
    on, _ = g.next_goal_indices(ids, times, 3, include_current=True)
    assert torch.equal(off, on)


def test_include_current_prepends_the_segment_once_its_hold_has_passed():
    g = _graph()
    ids = torch.tensor([0])
    times = torch.tensor([0.9])          # inside segment 0, hold 0.5 behind us
    off, _ = g.next_goal_indices(ids, times, 3, include_current=False)
    on, _ = g.next_goal_indices(ids, times, 3, include_current=True)
    assert off[0, 0].item() == 1, "legacy already commands the NEXT hold"
    assert on[0, 0].item() == 0, "slot 0 should be the segment we are inside"
    # The forward window is preserved, just shifted along by one slot.
    assert on[0, 1].item() == 1 and on[0, 2].item() == 2
    # ...and never re-serves the segment already sitting in slot 0.
    assert (on[0, 1:] > on[0, 0]).all()


def test_include_current_falls_back_in_a_gap():
    """Motion 1 has no segment covering t = 1.5; legacy behaviour must resume."""
    g = _graph()
    ids = torch.tensor([1])
    times = torch.tensor([1.5])
    off, _ = g.next_goal_indices(ids, times, 3, include_current=False)
    on, _ = g.next_goal_indices(ids, times, 3, include_current=True)
    assert torch.equal(off, on)


def test_promotion_shifts_the_window_and_never_invalidates_slot_zero():
    g = _graph()
    ids = torch.tensor([0, 0])
    times = torch.tensor([0.0, 0.0])
    base_idx, _ = g.next_goal_indices(ids, times, 2)
    promoted, valid = g.next_goal_indices(
        ids, times, 2, promote_k=torch.tensor([1, 99])
    )
    assert promoted[0, 0].item() == base_idx[0, 0].item() + 1
    # A skip larger than the clip can serve is clamped, not allowed to push the
    # nearest goal off the end and silently leave the student with no goal.
    assert promoted[1, 0].item() == 2
    assert bool(valid[1, 0])


def test_promotion_zero_is_bit_identical():
    g = _graph()
    ids = torch.tensor([0, 1])
    times = torch.tensor([0.3, 0.3])
    a, va = g.next_goal_indices(ids, times, 3)
    b, vb = g.next_goal_indices(ids, times, 3, promote_k=torch.zeros(2, dtype=torch.long))
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_seg_start_sortedness_is_asserted():
    """A graph whose segments are out of order must be refused, not mis-served."""
    g = _graph()
    payload = {
        "motion_names": g.motion_names, "pair_names": g.pair_names,
        "orientation_names": g.orientation_names, "node_keys": g.node_keys,
        "node_contact": g.node_contact, "node_orient": g.node_orient,
        "seg_node": g.seg_node, "seg_hold": g.seg_hold, "seg_end": g.seg_end,
        "seg_count": g.seg_count,
        "seg_start": torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 3.0]]),
    }
    try:
        ContactGraph(payload)
    except ValueError as exc:
        assert "start times are not sorted" in str(exc)
    else:
        raise AssertionError("out-of-order seg_start was accepted")


# --------------------------------------------------------------------------- #
# the observation kernel
# --------------------------------------------------------------------------- #
def _spec(envs=2, steps=3, pairs=4, bins=2):
    contact = torch.rand(envs, steps, pairs).round()
    orient = torch.zeros(envs, steps, bins)
    orient[..., 0] = 1.0
    visible = torch.ones(envs, steps)
    return contact, orient, visible


def test_dwell_absent_is_bit_identical():
    contact, orient, visible = _spec()
    legacy = compute_contact_goal_obs(contact, orient, visible)
    empty = compute_contact_goal_obs(
        contact, orient, visible, contact.new_zeros((2, 3, 0))
    )
    assert torch.equal(legacy, empty)
    assert legacy.shape[1] == 3 * (4 + 2 + 1)


def test_dwell_widens_the_slot_and_lands_at_the_end():
    contact, orient, visible = _spec()
    dwell = torch.stack(
        [torch.full((2, 3), 0.25), torch.full((2, 3), 0.75)], dim=-1
    )
    out = compute_contact_goal_obs(contact, orient, visible, dwell)
    per_slot = 4 + 2 + 1 + 2
    assert out.shape[1] == 3 * per_slot
    block = out.view(2, 3, per_slot)
    # The two channels are the LAST two columns of each slot, in order.
    assert torch.allclose(block[..., -2], torch.full((2, 3), 0.25))
    assert torch.allclose(block[..., -1], torch.full((2, 3), 0.75))
    # ...and everything to their left is untouched.
    legacy = compute_contact_goal_obs(contact, orient, visible).view(2, 3, 7)
    assert torch.equal(block[..., :7], legacy)


def test_dwell_survives_a_hidden_contact_half():
    """Gated by slot validity, NOT by contact visibility -- the design decision.

    How long a command lasts is a property of its timing, like the deadline. If
    it were gated by `visible` it would vanish on ~15 % of slots for no reason.
    """
    contact, orient, visible = _spec()
    visible[:, 1] = 0.0
    contact[:, 1] = 0.0                       # the control component pre-zeroes these
    orient[:, 1] = 0.0
    dwell = torch.full((2, 3, 2), 0.5)
    block = compute_contact_goal_obs(contact, orient, visible, dwell).view(2, 3, 9)
    assert torch.allclose(block[:, 1, :6], torch.zeros(2, 6)), "contact half hidden"
    assert float(block[0, 1, 6]) == 0.0, "visible flag off"
    assert torch.allclose(block[:, 1, 7:], torch.full((2, 2), 0.5)), (
        "dwell must survive a hidden contact half"
    )


# --------------------------------------------------------------------------- #
# the dwell arithmetic itself
# --------------------------------------------------------------------------- #
class _StubControl:
    """Just enough of ContactGraphControl to exercise _compute_dwell_features."""

    from protomotions.envs.control.contact_graph_control import (  # noqa: E402
        ContactGraphControl as _Real,
    )

    _compute_dwell_features = _Real._compute_dwell_features

    def __init__(self, valid, dwell_channels=True, clip=10.0):
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.dwell_channels = dwell_channels
        self.config.history_time_clip_s = clip
        self.goal_valid = valid


def test_dwell_features_disabled_is_empty():
    ctl = _StubControl(torch.ones(2, 3, dtype=torch.bool), dwell_channels=False)
    out = ctl._compute_dwell_features(
        torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2)
    )
    assert out.shape == (2, 3, 0)


def test_dwell_features_values_and_scaling():
    ctl = _StubControl(torch.ones(1, 2, dtype=torch.bool))
    t_hold = torch.tensor([[1.0, 5.0]])
    t_end = torch.tensor([[4.0, 9.0]])
    now = torch.tensor([2.0])
    out = ctl._compute_dwell_features(t_hold, t_end, now)
    # duration = t_end - t_hold = [3, 4]; remaining = t_end - now = [2, 7]
    assert torch.allclose(out[0, :, 0], torch.tensor([0.3, 0.4]))
    assert torch.allclose(out[0, :, 1], torch.tensor([0.2, 0.7]))
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0


def test_dwell_features_clamp_and_padding():
    """Padded slots carry +inf; a nan/inf reaching the observation is a bug."""
    ctl = _StubControl(torch.tensor([[True, False]]))
    t_hold = torch.tensor([[0.0, float("inf")]])
    t_end = torch.tensor([[999.0, float("inf")]])
    out = ctl._compute_dwell_features(t_hold, t_end, torch.tensor([0.0]))
    assert torch.isfinite(out).all(), "inf padding leaked into the observation"
    assert float(out[0, 0, 0]) == 1.0, "a very long hold saturates at the clip"
    assert torch.allclose(out[0, 1], torch.zeros(2)), "invalid slot must be zeroed"


def test_dwell_remaining_never_negative_after_the_segment_ends():
    ctl = _StubControl(torch.ones(1, 1, dtype=torch.bool))
    out = ctl._compute_dwell_features(
        torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([5.0])
    )
    assert float(out[0, 0, 1]) == 0.0
