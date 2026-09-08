# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contact-configuration graph and its student conditioning.

Three layers, all CPU-only:

* the force-annotation helpers in ``data/scripts/build_contact_graph_from_rollouts``;
* :class:`protomotions.components.contact_graph.ContactGraph`'s goal lookup;
* :class:`protomotions.envs.control.contact_graph_control.ContactGraphControl`'s
  schedule and mask bookkeeping, against a stub environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "data" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from build_contact_graph_from_rollouts import (  # noqa: E402
    absorb_short_runs,
    demote_supported_pairs,
    node_identity_pairs,
    force_hysteresis,
    hold_frame,
    pair_forces_from_bodies,
    pair_index,
    segment_runs,
    substeps_to_policy_steps,
    zone_sim_indices,
)
from protomotions.components.contact_graph import ContactGraph  # noqa: E402
from protomotions.envs.obs.contact_goal import (  # noqa: E402
    compute_contact_goal_masks,
    compute_contact_goal_obs,
)

SIM_BODY_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax",
    "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


# --------------------------------------------------------------------------- #
# Force annotation
# --------------------------------------------------------------------------- #
def test_pair_index_masks_adjacent_zone_pairs():
    names, decomposed = pair_index()
    assert len(names) == len(decomposed) == 104
    assert names[:15] == [n for n in names if n.endswith(":G")]
    # Anatomically adjacent pairs are permanently near and never a real contact.
    assert "PELVIS+TRUNK" not in names
    assert "L_FOOT+L_SHANK" not in names
    # Cross-side and non-adjacent pairs are eligible.
    assert "L_SHANK+L_UPPER_ARM" in names
    assert "L_FOOT+R_THIGH" in names


def test_zone_sim_indices_resolves_by_name_not_position():
    zones = zone_sim_indices(SIM_BODY_NAMES)
    assert zones["L_FOOT"] == [SIM_BODY_NAMES.index("L_Ankle"), SIM_BODY_NAMES.index("L_Toe")]
    assert len(zones["TRUNK"]) == 5
    shuffled = list(reversed(SIM_BODY_NAMES))
    other = zone_sim_indices(shuffled)
    assert other["L_FOOT"] == [shuffled.index("L_Ankle"), shuffled.index("L_Toe")]


def test_zone_sim_indices_rejects_a_missing_body():
    with pytest.raises(ValueError, match="L_Toe"):
        zone_sim_indices([n for n in SIM_BODY_NAMES if n != "L_Toe"])


def test_substeps_to_policy_steps_uses_the_preceding_window():
    # Three policy steps of two substeps each; the snapshot is taken *before*
    # stepping, so entry i closes the window opened by entry i - 1.
    values = np.arange(6, dtype=np.float64).reshape(6, 1)
    index = np.array([0, 2, 4, 6])
    out = substeps_to_policy_steps(values, index)
    assert out.shape == (3, 1)
    assert out[:, 0].tolist() == [0.5, 2.5, 4.5]


def test_substeps_to_policy_steps_tolerates_a_truncated_tail():
    values = np.arange(4, dtype=np.float64).reshape(4, 1)
    index = np.array([0, 2, 4, 9])  # last window runs past the recorded data
    out = substeps_to_policy_steps(values, index)
    assert out.shape == (3, 1)
    # Falls back to the last recorded substep rather than inventing a
    # contact-free frame, which the segmenter would take at face value.
    assert out[2, 0] == pytest.approx(3.0)


def test_substeps_to_policy_steps_last_takes_the_end_of_each_window():
    # Orientations must not be averaged: the reduction has to be able to return
    # the sample the control-step timestamp was read at.
    values = np.arange(6, dtype=np.float64).reshape(6, 1)
    index = np.array([0, 2, 4, 6])
    out = substeps_to_policy_steps(values, index, reduce="last")
    assert out[:, 0].tolist() == [1.0, 3.0, 5.0]


def test_substeps_to_policy_steps_rejects_an_unknown_reduction():
    with pytest.raises(ValueError, match="reduce must be"):
        substeps_to_policy_steps(np.zeros((2, 1)), np.array([0, 2]), reduce="median")


def test_pair_forces_symmetrise_body_body_without_double_counting():
    names, decomposed = pair_index()
    zones = zone_sim_indices(SIM_BODY_NAMES)
    n_bodies = len(SIM_BODY_NAMES)
    ground = np.zeros((1, n_bodies))
    ground[0, SIM_BODY_NAMES.index("L_Ankle")] = 300.0
    ground[0, SIM_BODY_NAMES.index("L_Toe")] = 100.0
    bb = np.zeros((1, n_bodies, n_bodies))
    knee = SIM_BODY_NAMES.index("L_Knee")
    shoulder = SIM_BODY_NAMES.index("L_Shoulder")
    bb[0, knee, shoulder] = 200.0
    bb[0, shoulder, knee] = 180.0

    forces = pair_forces_from_bodies(ground, bb, zones, decomposed)
    assert forces[0, names.index("L_FOOT:G")] == pytest.approx(400.0)
    # Mean of the two sensors' readings, counted once.
    assert forces[0, names.index("L_SHANK+L_UPPER_ARM")] == pytest.approx(190.0)


def test_pair_forces_clamp_negative_ground_noise():
    names, decomposed = pair_index()
    zones = zone_sim_indices(SIM_BODY_NAMES)
    ground = np.zeros((1, len(SIM_BODY_NAMES)))
    ground[0, SIM_BODY_NAMES.index("L_Ankle")] = -5.0
    bb = np.zeros((1, len(SIM_BODY_NAMES), len(SIM_BODY_NAMES)))
    forces = pair_forces_from_bodies(ground, bb, zones, decomposed)
    assert forces[0, names.index("L_FOOT:G")] == 0.0


def _pair_slot(names, name):
    return names.index(name)


def test_demote_supported_pairs_drops_the_standing_feet_touch():
    names, decomposed = pair_index()
    active = np.zeros((3, len(names)), dtype=bool)
    lf, rf = _pair_slot(names, "L_FOOT:G"), _pair_slot(names, "R_FOOT:G")
    touch = _pair_slot(names, "L_FOOT+R_FOOT")
    # Frame 0: both feet grounded and touching -> the touch is demoted.
    active[0, [lf, rf, touch]] = True
    # Frame 1: Eagle's hook -- only the right foot grounded -> touch is identity.
    active[1, [rf, touch]] = True
    # Frame 2: both grounded, no touch.
    active[2, [lf, rf]] = True

    identity, demoted = demote_supported_pairs(active, decomposed)

    assert not identity[0, touch] and demoted[0, touch]
    assert identity[1, touch] and not demoted[1, touch]
    assert not demoted[2].any()
    # Ground pairs are never demoted.
    assert identity[:, lf].tolist() == active[:, lf].tolist()
    assert (identity | demoted).tolist() == active.tolist()
    assert not (identity & demoted).any()


def test_demote_supported_pairs_keeps_arm_balance_load_paths():
    names, decomposed = pair_index()
    active = np.zeros((1, len(names)), dtype=bool)
    # Crow: hands down, shins pressed on upper arms. The arms have no ground
    # pair of their own... they do (UPPER_ARM:G exists) but it is inactive.
    for pair in ("L_HAND:G", "R_HAND:G", "L_SHANK+L_UPPER_ARM", "R_SHANK+R_UPPER_ARM"):
        active[0, _pair_slot(names, pair)] = True

    identity, demoted = demote_supported_pairs(active, decomposed)

    assert identity[0, _pair_slot(names, "L_SHANK+L_UPPER_ARM")]
    assert identity[0, _pair_slot(names, "R_SHANK+R_UPPER_ARM")]
    assert not demoted.any()


def test_demote_supported_pairs_follows_ground_state_frame_by_frame():
    names, decomposed = pair_index()
    active = np.zeros((2, len(names)), dtype=bool)
    ls, rs = _pair_slot(names, "L_SHANK:G"), _pair_slot(names, "R_SHANK:G")
    pair = _pair_slot(names, "L_SHANK+R_SHANK")
    # Kneeling with shanks together: demoted. Lifting one shank off the
    # ground turns the same press into a load path.
    active[:, [rs, pair]] = True
    active[0, ls] = True

    identity, demoted = demote_supported_pairs(active, decomposed)

    assert demoted[0, pair] and not identity[0, pair]
    assert identity[1, pair] and not demoted[1, pair]


def test_force_hysteresis_latches_and_releases():
    force = np.array([[0.0], [30.0], [20.0], [5.0], [20.0]])
    active = force_hysteresis(force, make_n=25.0, break_n=10.0)
    # on at 30, stays on at 20 (above break), off at 5, stays off at 20
    # (below make), which is what a hysteresis is for.
    assert active[:, 0].tolist() == [False, True, True, False, False]


def test_force_hysteresis_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        force_hysteresis(np.zeros((2, 1)), make_n=1.0, break_n=2.0)


def test_force_hysteresis_accepts_per_pair_thresholds():
    force = np.array([[30.0, 30.0]])
    active = force_hysteresis(force, make_n=np.array([25.0, 40.0]),
                              break_n=np.array([10.0, 20.0]))
    assert active[0].tolist() == [True, False]


def test_segment_runs_and_short_run_absorption_keep_a_total_tiling():
    ids = np.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 2])
    assert segment_runs(ids) == [(0, 3, 0), (4, 4, 1), (5, 9, 2)]
    absorbed = absorb_short_runs(ids, min_frames=3)
    assert len(absorbed) == len(ids)
    assert 1 not in absorbed.tolist()  # the singleton is gone
    # and it went to the longer neighbour, not automatically the previous one
    assert absorbed[4] == 2


def test_absorb_short_runs_is_a_no_op_on_long_runs():
    ids = np.array([0, 0, 0, 1, 1, 1])
    assert absorb_short_runs(ids, min_frames=3).tolist() == ids.tolist()


def test_hold_frame_avoids_the_transition_boundaries():
    # Slowest frame is at the very start, but that is the make event.
    speed = np.array([0.0, 5.0, 5.0, 5.0, 0.1, 5.0, 5.0, 5.0, 5.0, 5.0])
    assert hold_frame(0, 9, speed) == 4


def test_hold_frame_smoothing_prefers_a_static_stretch_to_a_lucky_frame():
    # One isolated near-zero sample in a moving stretch, against a genuinely
    # still run. Unsmoothed, argmin takes the lucky frame; smoothed, it takes
    # the stretch -- which is what survives PhysX's run-to-run variation.
    speed = np.array([9.0] * 3 + [1.0, 1.0, 1.0, 1.0, 1.0] + [9.0, 0.0, 9.0] + [9.0] * 4)
    assert hold_frame(0, 14, speed) == 9            # the lucky single frame
    assert 3 <= hold_frame(0, 14, speed, window=5) <= 7  # the still stretch


def test_hold_frame_breaks_a_flat_hold_toward_the_centre():
    # A perfectly flat hold: every frame is equally "most static", so the pick
    # must not depend on which one numpy happens to see first.
    speed = np.zeros(21)
    assert hold_frame(0, 20, speed) == 10
    # ...and it stays put when the flat region is perturbed below tolerance,
    # which is what makes it stable across two rollouts of the same clip.
    speed[4] = -0.0
    speed[16] = 0.0
    assert hold_frame(0, 20, speed) == 10


def test_hold_frame_on_a_degenerate_segment():
    speed = np.array([1.0, 2.0])
    assert hold_frame(0, 1, speed) in (0, 1)


def _segment(node_pairs, t_start, t_hold, trusted=True, orientation="upright"):
    return {
        "pairs": list(node_pairs),
        "pair_ids": [],
        "orientation_bin": orientation,
        "orientation_id": 0,
        "t_start": t_start,
        "t_end": t_hold + 1.0,
        "t_hold": t_hold,
        "duration_s": 2.0,
        "trusted": trusted,
    }


def test_edges_carry_both_hold_times_and_their_motion():
    from build_contact_graph_from_rollouts import build_graph

    records = {
        "clip_a": {
            "segments": [
                _segment(["L_FOOT:G", "R_FOOT:G"], 0.0, 1.0),
                _segment(["L_HAND:G", "R_HAND:G"], 3.0, 4.0),
            ]
        }
    }
    _ids, nodes, _per_motion, edges = build_graph(records, ["clip_a"], [])
    assert len(edges) == 1
    edge = next(iter(edges.values()))
    occurrence = edge["occurrences"][0]
    assert occurrence["motion"] == "clip_a" and occurrence["motion_id"] == 0
    assert occurrence["t_hold_src"] == 1.0
    assert occurrence["t_hold_dst"] == 4.0
    assert nodes[edge["src"]]["pairs"] == ["L_FOOT:G", "R_FOOT:G"]


def test_edges_do_not_bridge_a_stretch_the_expert_did_not_track():
    from build_contact_graph_from_rollouts import build_graph

    records = {
        "clip_a": {
            "segments": [
                _segment(["L_FOOT:G", "R_FOOT:G"], 0.0, 1.0),
                # The policy fell over here; joining across it would invent a
                # transition that never happened.
                _segment(["TRUNK:G"], 3.0, 4.0, trusted=False),
                _segment(["L_HAND:G", "R_HAND:G"], 6.0, 7.0),
            ]
        }
    }
    _ids, _nodes, per_motion, edges = build_graph(records, ["clip_a"], [])
    assert edges == {}
    # The untrusted segment is still recorded, just never packed or linked.
    assert len(per_motion[0]) == 3
    assert sum(s["trusted"] for s in per_motion[0]) == 2


# --------------------------------------------------------------------------- #
# Runtime graph
# --------------------------------------------------------------------------- #
def _toy_graph_payload():
    # Two motions. Motion 0 has three holds, motion 1 has one.
    return {
        "motion_names": ["clip_a", "clip_b"],
        "pair_names": ["L_FOOT:G", "R_FOOT:G", "L_HAND:G"],
        "orientation_names": ["upright", "inverted"],
        "node_keys": ["stand", "one_leg", "handstand"],
        "node_contact": torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "node_orient": torch.tensor([0, 0, 1]),
        "seg_node": torch.tensor([[0, 1, 2], [2, -1, -1]]),
        "seg_start": torch.tensor([[0.0, 2.0, 5.0], [0.0, float("inf"), float("inf")]]),
        "seg_end": torch.tensor([[2.0, 5.0, 9.0], [4.0, float("inf"), float("inf")]]),
        "seg_hold": torch.tensor([[1.0, 3.0, 7.0], [2.0, float("inf"), float("inf")]]),
        "seg_count": torch.tensor([3, 1]),
        "min_lead_s": 0.2,
        "zone_order": ["L_FOOT", "R_FOOT", "L_HAND"],
        "zone_bodies": {
            "L_FOOT": ["L_Ankle", "L_Toe"],
            "R_FOOT": ["R_Ankle", "R_Toe"],
            "L_HAND": ["L_Wrist", "L_Hand"],
        },
    }


def test_contact_graph_lookup_returns_the_upcoming_holds():
    graph = ContactGraph(_toy_graph_payload())
    ids = torch.tensor([0, 0])
    times = torch.tensor([0.0, 3.5])
    indices, valid = graph.next_goal_indices(ids, times, num_steps=3)
    assert indices[0].tolist() == [0, 1, 2]
    assert valid[0].tolist() == [True, True, True]
    # At t=3.5 the 3.0 s hold is behind us, so the next goal is the 7.0 s one.
    assert indices[1].tolist() == [2, 2, 2]
    assert valid[1].tolist() == [True, False, False]


def test_contact_graph_min_lead_skips_a_hold_that_is_upon_us():
    graph = ContactGraph(_toy_graph_payload())
    ids = torch.tensor([0])
    near, _ = graph.next_goal_indices(ids, torch.tensor([0.9]), 1, min_lead_s=0.0)
    late, _ = graph.next_goal_indices(ids, torch.tensor([0.9]), 1, min_lead_s=0.5)
    assert near[0, 0].item() == 0  # the 1.0 s hold is still ahead
    assert late[0, 0].item() == 1  # 0.1 s is not enough lead; aim past it


def test_contact_graph_gather_matches_the_node_table():
    graph = ContactGraph(_toy_graph_payload())
    ids = torch.tensor([0])
    indices = torch.tensor([[0, 2]])
    gathered = graph.gather(ids, indices)
    assert gathered["node"].tolist() == [[0, 2]]
    assert gathered["contact"][0, 0].tolist() == [1.0, 1.0, 0.0]
    assert gathered["contact"][0, 1].tolist() == [0.0, 0.0, 1.0]
    assert gathered["orient"].tolist() == [[0, 1]]
    assert gathered["t_hold"].tolist() == [[1.0, 7.0]]


def test_contact_graph_rejects_a_mismatched_motion_library():
    graph = ContactGraph(_toy_graph_payload())
    graph.validate_against_motion_lib(["dir/clip_a.motion", "dir/clip_b.motion"])
    with pytest.raises(ValueError, match="different motion library"):
        graph.validate_against_motion_lib(["dir/clip_a.motion"])


def test_contact_graph_rejects_unsorted_hold_times():
    payload = _toy_graph_payload()
    payload["seg_hold"] = torch.tensor(
        [[5.0, 3.0, 7.0], [2.0, float("inf"), float("inf")]]
    )
    with pytest.raises(ValueError, match="not sorted"):
        ContactGraph(payload)


def test_contact_graph_coverage_counts_motions_with_segments():
    payload = _toy_graph_payload()
    payload["seg_count"] = torch.tensor([3, 0])
    assert ContactGraph(payload).coverage() == (1, 2)


def test_contact_graph_carries_its_own_zone_definition():
    graph = ContactGraph(_toy_graph_payload())
    assert graph.zone_order == ["L_FOOT", "R_FOOT", "L_HAND"]
    assert graph.zone_bodies["L_FOOT"] == ["L_Ankle", "L_Toe"]


def test_control_falls_back_when_the_graph_predates_zone_storage(tmp_path, monkeypatch):
    # Graphs written before zone_order/zone_bodies were stored must still load,
    # using the extractor's own definition -- which is what they were built with.
    from protomotions.envs.control import contact_graph_control as module
    from protomotions.envs.control.mimic_control import MimicControl

    payload = _toy_graph_payload()
    del payload["zone_order"]
    del payload["zone_bodies"]
    graph_path = tmp_path / "contact_graph.pt"
    torch.save(payload, graph_path)
    monkeypatch.setattr(MimicControl, "populate_context", lambda self, ctx: None)
    control = module.ContactGraphControl(
        module.ContactGraphControlConfig(
            graph_file=str(graph_path), num_goal_steps=3, num_masked_future_steps=3
        ),
        _stub_env(),
    )
    assert control._ground_zone_names == ["L_FOOT", "R_FOOT", "L_HAND"]


# --------------------------------------------------------------------------- #
# Observation kernel
# --------------------------------------------------------------------------- #
def test_contact_goal_obs_layout_is_one_row_per_goal_step():
    contact = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])       # [1, 2 steps, 2 pairs]
    orient = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])        # [1, 2 steps, 2 bins]
    visible = torch.tensor([[1.0, 0.0]])
    obs = compute_contact_goal_obs(contact, orient, visible)
    assert obs.shape == (1, 2 * (2 + 2 + 1))
    assert obs[0, :5].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert obs[0, 5:].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_contact_goal_masks_are_float():
    assert compute_contact_goal_masks(torch.tensor([[1.0, 0.0]])).dtype == torch.float32


# --------------------------------------------------------------------------- #
# Control component
# --------------------------------------------------------------------------- #
class _StubMotionLib:
    def __init__(self, names, lengths, reference_pose=None):
        self.motion_files = [f"dir/{n}.motion" for n in names]
        self._lengths = torch.tensor(lengths)
        self.num_bodies = len(SIM_BODY_NAMES)
        # [1, bodies, 3] returned for every requested (motion, time), so a test
        # can hand the goal a known pose to score against.
        self.reference_pose = reference_pose

    def get_motion_length(self, motion_ids):
        return self._lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        if self.reference_pose is None:
            pos = torch.zeros(n, self.num_bodies, 3)
        else:
            pos = self.reference_pose.expand(n, -1, -1).clone()
        # Identity (w-last), not zeros: the goal pose-error metric normalises by
        # the reference's own heading, and a zero quaternion is a 180-degree
        # rotation rather than a no-op.
        rot = torch.zeros(n, self.num_bodies, 4)
        rot[..., 3] = 1.0
        return SimpleNamespace(rigid_body_pos=pos, rigid_body_rot=rot)


def _stub_env(num_envs=4):
    motion_lib = _StubMotionLib(["clip_a", "clip_b"], [10.0, 5.0])
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        dt=1.0 / 30.0,
        motion_lib=motion_lib,
        motion_manager=SimpleNamespace(
            motion_ids=torch.zeros(num_envs, dtype=torch.long),
            motion_times=torch.zeros(num_envs),
        ),
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=list(SIM_BODY_NAMES)),
            trackable_bodies_subset=list(SIM_BODY_NAMES),
            mimic_small_marker_bodies=None,
        ),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=(
            lambda pos: torch.zeros_like(pos)
        ),
    )


def _current_state(contact_forces=None, num_envs=4, body_pos=None):
    """The slice of ``ctx.current`` the contact-goal half reads.

    Body positions and rotations are here because the commanded-goal pose
    error is scored from them every step; the rotation is a real identity
    quaternion (w-last) rather than zeros so the heading normalisation is
    well defined.
    """
    if body_pos is None:
        body_pos = torch.zeros(num_envs, len(SIM_BODY_NAMES), 3)
    body_rot = torch.zeros(num_envs, len(SIM_BODY_NAMES), 4)
    body_rot[..., 3] = 1.0
    return SimpleNamespace(
        rigid_body_contact_forces=contact_forces,
        rigid_body_pos=body_pos,
        rigid_body_rot=body_rot,
    )


def _make_control(tmp_path, monkeypatch, seed: int = 0, **overrides):
    from protomotions.envs.control import contact_graph_control as module
    from protomotions.envs.control.mimic_control import MimicControl

    # Mask sampling is stochastic; pin it so a test added later cannot shift the
    # RNG stream and turn an unrelated assertion red.
    torch.manual_seed(seed)
    graph_path = tmp_path / "contact_graph.pt"
    torch.save(_toy_graph_payload(), graph_path)

    # The mimic half of populate_context is upstream code needing the full env;
    # these tests are about the contact-goal half.
    monkeypatch.setattr(MimicControl, "populate_context", lambda self, ctx: None)

    kwargs = {
        "graph_file": str(graph_path),
        "num_goal_steps": 3,
        "num_masked_future_steps": 3,
    }
    kwargs.update(overrides)
    return module.ContactGraphControl(
        module.ContactGraphControlConfig(**kwargs), _stub_env()
    )


def test_control_requires_matching_goal_and_token_counts(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="num_goal_steps"):
        _make_control(tmp_path, monkeypatch, num_goal_steps=4)


def test_control_reset_populates_a_full_goal_schedule(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    assert control.goal_index.shape == (4, 3)
    assert control.goal_index[0].tolist() == [0, 1, 2]
    assert torch.allclose(control.target_times[0], torch.tensor([1.0, 3.0, 7.0]))
    assert control.goal_valid[0].all()


def test_control_masks_follow_a_goal_as_it_shifts_down_the_slots(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    before_pose = control.pose_visible.clone()
    before_contact = control.contact_visible.clone()
    before_bodies = control.goal_body_masks.clone()

    # Advance past the first hold so every env's schedule shifts by exactly one.
    control.env.motion_manager.motion_times[:] = 2.0
    control.step()

    assert control.goal_index[0].tolist() == [1, 2, 2]
    # Slot 0 now holds what slot 1 held: same specification, not a fresh sample.
    assert torch.equal(control.pose_visible[:, 0], before_pose[:, 1])
    assert torch.equal(control.goal_body_masks[:, 0], before_bodies[:, 1])
    # ...with one deliberate exception: a slot that specified neither half has
    # its contact set revealed once it becomes the nearest goal, so the student
    # is never asked to act on an empty specification.
    forced = ~before_pose[:, 1] & ~before_contact[:, 1]
    assert torch.equal(control.contact_visible[:, 0], before_contact[:, 1] | forced)


def test_control_does_not_resample_when_the_schedule_has_not_moved(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    before = control.contact_visible.clone()
    control.env.motion_manager.motion_times[:] = 0.5  # still before the first hold
    control.step()
    assert torch.equal(control.contact_visible, before)


def test_control_always_specifies_the_nearest_goal(tmp_path, monkeypatch):
    control = _make_control(
        tmp_path, monkeypatch, pose_visible_prob=0.0, contact_visible_prob=0.0
    )
    control.reset(torch.arange(4))
    # Both halves would have been hidden; the nearest slot is forced to carry
    # its contact set so the student is never asked to act on nothing.
    assert control.contact_visible[:, 0].all()
    assert not control.contact_visible[:, 1:].any()


def test_control_context_zeroes_hidden_and_invalid_goals(tmp_path, monkeypatch):
    from protomotions.envs.context_views import EnvContext

    control = _make_control(
        tmp_path, monkeypatch, pose_visible_prob=1.0, contact_visible_prob=1.0
    )
    # Motion 1 has a single segment, so slots 1 and 2 are past the end.
    control.env.motion_manager.motion_ids[:] = 1
    control.reset(torch.arange(4))

    ctx = EnvContext(
        current=_current_state(),
        noisy=None,
        dt=control.env.dt,
    )
    control.populate_context(ctx)

    assert ctx.contact_goal.contact_spec.shape == (4, 3, 3)
    assert ctx.contact_goal.visible[:, 0].all()
    assert not ctx.contact_goal.visible[:, 1:].any()
    assert ctx.contact_goal.contact_spec[:, 1:].abs().sum() == 0
    assert ctx.contact_goal.node_ids[:, 0].tolist() == [2, 2, 2, 2]
    assert (ctx.contact_goal.node_ids[:, 1:] == -1).all()
    # An exhausted slot clamps to a hold that is behind us; the raw offset would
    # be negative, which never occurs on a live goal, so it is zeroed.
    assert (ctx.masked_mimic.time_offsets[:, 1:] == 0).all()
    assert (ctx.masked_mimic.time_offsets[:, 0] > 0).all()
    # The pose token for an invalid slot carries no visible bodies either.
    masks = ctx.masked_mimic.target_bodies_masks.view(4, 3, len(SIM_BODY_NAMES), 2)
    assert masks[:, 0].any()
    assert not masks[:, 1:].any()


def test_control_pose_token_stays_open_when_only_contacts_are_given(tmp_path, monkeypatch):
    from protomotions.envs.context_views import EnvContext

    control = _make_control(
        tmp_path, monkeypatch, pose_visible_prob=0.0, contact_visible_prob=1.0
    )
    control.reset(torch.arange(4))
    ctx = EnvContext(
        current=_current_state(),
        noisy=None,
        dt=control.env.dt,
    )
    control.populate_context(ctx)
    # No body is revealed, but the slot is still worth attending to because the
    # contact configuration is specified.
    masks = ctx.masked_mimic.target_bodies_masks.view(4, 3, len(SIM_BODY_NAMES), 2)
    assert not masks.any()
    assert ctx.masked_mimic.target_poses_masks[:, 0].all()


def test_control_ground_iou_scores_the_nearest_goal(tmp_path, monkeypatch):
    from protomotions.envs.context_views import EnvContext

    control = _make_control(
        tmp_path, monkeypatch, pose_visible_prob=1.0, contact_visible_prob=1.0
    )
    control.reset(torch.arange(4))

    # Goal 0 on clip_a is "stand" = L_FOOT:G and R_FOOT:G. The toy graph's pair
    # list only has three entries, so the control component's ground mapping is
    # exercised over the zones those name.
    forces = torch.zeros(4, len(SIM_BODY_NAMES), 3)
    forces[0, SIM_BODY_NAMES.index("L_Ankle"), 2] = 400.0
    forces[0, SIM_BODY_NAMES.index("R_Ankle"), 2] = 400.0
    ctx = EnvContext(
        current=_current_state(forces),
        noisy=None,
        dt=control.env.dt,
    )
    control.populate_context(ctx)
    assert ctx.contact_goal.reached[0].item() == pytest.approx(1.0)
    # Env 1 has no ground contact at all against a two-zone goal.
    assert ctx.contact_goal.reached[1].item() == pytest.approx(0.0)


def test_goal_pose_error_measures_distance_to_the_commanded_pose(
    tmp_path, monkeypatch
):
    """The pose half of the goal, scored -- what ground IoU cannot see.

    At a degenerate node every member has the same contact set, so ``reached``
    is 1.00 whatever pose is held; this is the number that tells Warrior III
    from Lord of the Dance.
    """
    from protomotions.envs.context_views import EnvContext

    control = _make_control(
        tmp_path, monkeypatch, pose_visible_prob=1.0, contact_visible_prob=1.0
    )
    # Goal pose: every body at the origin. Achieved pose: every non-pelvis body
    # displaced 0.10 m along +z, so each conditionable body but the pelvis is
    # exactly 0.10 m off and the pelvis (the reference point) is exact.
    control.env.motion_lib.reference_pose = torch.zeros(
        1, len(SIM_BODY_NAMES), 3
    )
    achieved = torch.zeros(4, len(SIM_BODY_NAMES), 3)
    achieved[:, 1:, 2] = 0.10
    control.reset(torch.arange(4))
    # Reset samples a random body subset per env; pin it so the expected value
    # is arithmetic rather than a draw.
    control.goal_body_masks[:] = True

    def error(body_pos):
        ctx = EnvContext(
            current=_current_state(body_pos=body_pos),
            noisy=None,
            dt=control.env.dt,
        )
        control.populate_context(ctx)
        return ctx.contact_goal

    goal = error(achieved)
    num_bodies = len(control.conditionable_body_ids)
    expected = 0.10 * (num_bodies - 1) / num_bodies  # the pelvis contributes 0
    assert torch.allclose(
        goal.pose_error, torch.full((4,), expected), atol=1e-5
    )
    assert torch.equal(goal.pose_error_visible, torch.ones(4))

    # Exact pose -> zero error, so the metric is not measuring a constant.
    assert error(torch.zeros(4, len(SIM_BODY_NAMES), 3)).pose_error.abs().max() < 1e-6

    # It averages over the bodies the goal actually names, not over all of
    # them: with only the head conditioned the error is that body's own.
    control.goal_body_masks[:] = False
    control.goal_body_masks[:, :, SIM_BODY_NAMES.index("Head"), 0] = True
    assert torch.allclose(
        error(achieved).pose_error, torch.full((4,), 0.10), atol=1e-5
    )


def test_goal_pose_error_reports_visibility_and_does_not_dilute(
    tmp_path, monkeypatch
):
    """Rows with no commanded pose take the batch mean, not a zero."""
    from protomotions.envs.context_views import EnvContext

    control = _make_control(
        tmp_path, monkeypatch, pose_visible_prob=1.0, contact_visible_prob=1.0
    )
    control.env.motion_lib.reference_pose = torch.zeros(
        1, len(SIM_BODY_NAMES), 3
    )
    achieved = torch.zeros(4, len(SIM_BODY_NAMES), 3)
    achieved[:, 1:, 2] = 0.10
    control.reset(torch.arange(4))
    control.goal_body_masks[:] = True
    # Hide the nearest goal's pose on half the envs, after the reset that
    # sampled the masks.
    control.pose_visible[2:, 0] = False

    ctx = EnvContext(
        current=_current_state(body_pos=achieved),
        noisy=None,
        dt=control.env.dt,
    )
    control.populate_context(ctx)

    assert torch.equal(
        ctx.contact_goal.pose_error_visible, torch.tensor([1.0, 1.0, 0.0, 0.0])
    )
    # All four rows carry the same value, so the mean over the batch is the
    # mean over the *measured* rows rather than being halved by the hidden ones.
    errors = ctx.contact_goal.pose_error
    assert torch.allclose(errors, errors[0].expand(4), atol=1e-6)
    assert errors[0] > 0


def test_manual_goal_overrides_the_clip_schedule(tmp_path, monkeypatch):
    from protomotions.envs.context_views import EnvContext

    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))

    ones = torch.ones(4, 3, dtype=torch.bool)
    control.set_manual_goal(
        node_ids=torch.full((4, 3), -1).index_put_(
            (torch.arange(4), torch.zeros(4, dtype=torch.long)),
            torch.full((4,), 2),
        ),
        # Take the goal pose from the *other* clip than the one being played.
        pose_motion_ids=torch.ones(4, 3, dtype=torch.long),
        pose_times=torch.full((4, 3), 2.0),
        time_offsets=torch.full((4, 3), 3.0),
        pose_visible=ones,
        contact_visible=ones,
    )

    ctx = EnvContext(
        current=_current_state(),
        noisy=None,
        dt=control.env.dt,
    )
    control.populate_context(ctx)

    assert ctx.contact_goal.node_ids[:, 0].tolist() == [2, 2, 2, 2]
    assert (ctx.contact_goal.node_ids[:, 1:] == -1).all()
    # Node 2 is the handstand: L_HAND:G only, inverted.
    assert ctx.contact_goal.contact_spec[0, 0].tolist() == [0.0, 0.0, 1.0]
    assert ctx.contact_goal.orient_spec[0, 0].tolist() == [0.0, 1.0]
    assert torch.allclose(ctx.masked_mimic.time_offsets[:, 0], torch.tensor(3.0))
    # The clip being played is 0; the pose was taken from clip 1.
    assert control._goal_motion_ids[:, 0].tolist() == [1, 1, 1, 1]


def test_manual_goal_counts_down_and_survives_steps(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    ones = torch.ones(4, 3, dtype=torch.bool)
    control.set_manual_goal(
        node_ids=torch.full((4, 3), 2),
        pose_motion_ids=torch.zeros(4, 3, dtype=torch.long),
        pose_times=torch.full((4, 3), 1.0),
        time_offsets=torch.full((4, 3), 1.0),
        pose_visible=ones,
        contact_visible=ones,
    )
    control.step()
    expected = 1.0 - control.env.dt
    assert torch.allclose(control._time_offsets, torch.full((4, 3), expected))
    # And it never counts below the lead time the policy was trained with.
    for _ in range(200):
        control.step()
    assert torch.allclose(
        control._time_offsets, torch.full((4, 3), control.config.min_lead_s)
    )


def test_manual_goal_survives_an_episode_reset(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    control.set_manual_goal(
        node_ids=torch.full((4, 3), 2),
        pose_motion_ids=torch.ones(4, 3, dtype=torch.long),
        pose_times=torch.full((4, 3), 2.0),
        time_offsets=torch.full((4, 3), 3.0),
        pose_visible=torch.zeros(4, 3, dtype=torch.bool),
        contact_visible=torch.ones(4, 3, dtype=torch.bool),
    )
    # An episode ending mid-query must not resample the specification.
    control.reset(torch.arange(4))
    assert not control.pose_visible.any()
    assert control.contact_visible.all()
    assert control._goal_motion_ids[:, 0].tolist() == [1, 1, 1, 1]


def test_clear_manual_goal_returns_to_the_clip_schedule(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    ones = torch.ones(4, 3, dtype=torch.bool)
    control.set_manual_goal(
        node_ids=torch.full((4, 3), 2),
        pose_motion_ids=torch.ones(4, 3, dtype=torch.long),
        pose_times=torch.full((4, 3), 2.0),
        time_offsets=torch.full((4, 3), 3.0),
        pose_visible=ones,
        contact_visible=ones,
    )
    control.clear_manual_goal()
    control.reset(torch.arange(4))
    assert control.goal_index[0].tolist() == [0, 1, 2]
    assert control._goal_motion_ids[0].tolist() == [0, 0, 0]


def test_manual_goal_requires_a_reset_first(tmp_path, monkeypatch):
    # step() returns early until _initialized, so a goal set before the first
    # reset would silently never count down.
    control = _make_control(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="before the first reset"):
        control.set_manual_goal(
            node_ids=torch.zeros(4, 3, dtype=torch.long),
            pose_motion_ids=torch.zeros(4, 3, dtype=torch.long),
            pose_times=torch.zeros(4, 3),
            time_offsets=torch.ones(4, 3),
            pose_visible=torch.ones(4, 3, dtype=torch.bool),
            contact_visible=torch.ones(4, 3, dtype=torch.bool),
        )


def test_manual_goal_rejects_a_node_outside_the_graph(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    kwargs = dict(
        pose_motion_ids=torch.zeros(4, 3, dtype=torch.long),
        pose_times=torch.zeros(4, 3),
        time_offsets=torch.ones(4, 3),
        pose_visible=torch.ones(4, 3, dtype=torch.bool),
        contact_visible=torch.ones(4, 3, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="outside"):
        control.set_manual_goal(node_ids=torch.full((4, 3), 99), **kwargs)
    with pytest.raises(ValueError, match="outside"):
        control.set_manual_goal(node_ids=torch.full((4, 3), -2), **kwargs)
    # -1 is the padding sentinel and must stay legal.
    control.set_manual_goal(node_ids=torch.full((4, 3), -1), **kwargs)


def test_manual_goal_does_not_alias_the_callers_buffers(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    node = torch.full((4, 3), 2, dtype=torch.long)
    control.set_manual_goal(
        node_ids=node,
        pose_motion_ids=torch.zeros(4, 3, dtype=torch.long),
        pose_times=torch.zeros(4, 3),
        time_offsets=torch.ones(4, 3),
        pose_visible=torch.ones(4, 3, dtype=torch.bool),
        contact_visible=torch.ones(4, 3, dtype=torch.bool),
    )
    # Mutating the caller's tensor must not change the live goal: the manual
    # tables are re-read every step, so an alias would swap the goal silently.
    node[:, 0] = 0
    control.step()
    assert control._gathered["node"][:, 0].tolist() == [2, 2, 2, 2]


def test_clear_manual_goal_restores_a_usable_clip_schedule(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    control.set_manual_goal(
        node_ids=torch.full((4, 3), 2),
        pose_motion_ids=torch.zeros(4, 3, dtype=torch.long),
        pose_times=torch.zeros(4, 3),
        time_offsets=torch.ones(4, 3),
        pose_visible=torch.ones(4, 3, dtype=torch.bool),
        contact_visible=torch.ones(4, 3, dtype=torch.bool),
    )
    assert control.goal_body_masks.all()          # forced on by the manual path
    control.clear_manual_goal()
    # The forced all-revealed masks are gone and the roll index is current, so
    # the clip schedule resumes cleanly rather than with stale state.
    assert not control.goal_body_masks.all()
    assert torch.equal(control.prev_first_index, control.goal_index[:, 0])
    assert control._goal_motion_ids[:, 0].tolist() == [0, 0, 0, 0]


def test_manual_goal_rejects_a_wrong_shape(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    with pytest.raises(ValueError, match="node_ids must be"):
        control.set_manual_goal(
            node_ids=torch.zeros(4, 2, dtype=torch.long),
            pose_motion_ids=torch.zeros(4, 3, dtype=torch.long),
            pose_times=torch.zeros(4, 3),
            time_offsets=torch.zeros(4, 3),
            pose_visible=torch.ones(4, 3, dtype=torch.bool),
            contact_visible=torch.ones(4, 3, dtype=torch.bool),
        )


# --------------------------------------------------------------------------- #
# Ghost character (goal-pose visualization robot)
# --------------------------------------------------------------------------- #
class _GhostSimulator:
    """Captures set_ghost_state calls the way the real simulator receives them."""

    def __init__(self, ghost_enabled=True):
        self.ghost_enabled = ghost_enabled
        self.headless = True  # keeps the marker half of get_markers_state inert
        self.calls = []

    def set_ghost_state(self, reset_state, active=None):
        self.calls.append((reset_state, active))


def _attach_ghost_env(control, ghost_enabled=True):
    """Wire a ghost-capable fake simulator and a pose-serving motion lib."""
    simulator = _GhostSimulator(ghost_enabled)
    control.env.simulator = simulator
    num_bodies = control.env.motion_lib.num_bodies
    num_dofs = (num_bodies - 1) * 3

    def get_motion_state(motion_ids, motion_times):
        from protomotions.simulator.base_simulator.simulator_state import (
            StateConversion,
        )

        n = len(motion_ids)
        body_pos = torch.zeros(n, num_bodies, 3)
        # Make the served pose identifiable: x encodes the clip, z the time.
        body_pos[:, :, 0] = motion_ids.float().reshape(-1, 1)
        body_pos[:, :, 2] = motion_times.reshape(-1, 1)
        return SimpleNamespace(
            rigid_body_pos=body_pos,
            rigid_body_rot=torch.zeros(n, num_bodies, 4),
            root_pos=body_pos[:, 0],
            root_rot=torch.zeros(n, 4),
            root_vel=torch.ones(n, 3),  # deliberately non-zero: must be ignored
            root_ang_vel=torch.ones(n, 3),
            dof_pos=torch.full((n, num_dofs), 0.25),
            dof_vel=torch.ones(n, num_dofs),
            state_conversion=StateConversion.COMMON,
            fps=None,
        )

    control.env.motion_lib.get_motion_state = get_motion_state
    # A constant spawn offset, to prove it reaches the ghost's root.
    control.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction = (
        lambda pos: torch.tensor([1.0, 2.0, 0.5]).expand_as(pos)
    )
    return simulator


def test_ghost_char_poses_the_nearest_goal_with_the_spawn_offset(
    tmp_path, monkeypatch
):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    simulator = _attach_ghost_env(control)

    control.get_markers_state()

    assert len(simulator.calls) == 1
    reset_state, active = simulator.calls[0]
    # Slot 0 of the clip schedule: clip 0's first hold at t=1.0 (x encodes the
    # clip, z the hold time), plus the (1.0, 2.0, 0.5) spawn offset.
    assert torch.allclose(reset_state.root_pos[0], torch.tensor([1.0, 2.0, 1.5]))
    assert torch.equal(active, control.goal_valid[:, 0])
    assert active.all()


def test_ghost_char_follows_a_manual_goal_from_another_clip(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    simulator = _attach_ghost_env(control)

    ones = torch.ones(4, 3, dtype=torch.bool)
    control.set_manual_goal(
        node_ids=torch.full((4, 3), 2),
        pose_motion_ids=torch.ones(4, 3, dtype=torch.long),  # the OTHER clip
        pose_times=torch.full((4, 3), 2.0),
        time_offsets=torch.full((4, 3), 3.0),
        pose_visible=ones,
        contact_visible=ones,
    )
    control.get_markers_state()

    reset_state, active = simulator.calls[-1]
    # x = clip 1, z = pose time 2.0, plus the spawn offset.
    assert torch.allclose(reset_state.root_pos[0], torch.tensor([2.0, 2.0, 2.5]))
    assert active.all()


def test_ghost_char_reports_invalid_slots_inactive(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    simulator = _attach_ghost_env(control)

    node = torch.full((4, 3), -1)
    node[0, 0] = 2  # only env 0 gets a real goal
    control.set_manual_goal(
        node_ids=node,
        pose_motion_ids=torch.zeros(4, 3, dtype=torch.long),
        pose_times=torch.zeros(4, 3),
        time_offsets=torch.ones(4, 3),
        pose_visible=torch.ones(4, 3, dtype=torch.bool),
        contact_visible=torch.ones(4, 3, dtype=torch.bool),
    )
    control.get_markers_state()

    _, active = simulator.calls[-1]
    assert active.tolist() == [True, False, False, False]


def test_ghost_char_is_silent_without_a_ghost_robot(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    simulator = _attach_ghost_env(control, ghost_enabled=False)

    control.get_markers_state()

    assert simulator.calls == []


def test_ghost_config_defaults_are_off():
    import dataclasses

    from protomotions.simulator.base_simulator.config import SimulatorConfig

    fields = {f.name: f for f in dataclasses.fields(SimulatorConfig)}
    assert fields["ghost_robot"].default is False
    assert tuple(fields["ghost_offset"].default) == (1.8, 0.0)


# --------------------------------------------------------------------------- #
# Multi-expert routing
# --------------------------------------------------------------------------- #
def _routing_agent(motion_expert, motion_ids, num_experts=3, action_dim=4):
    """A MultiExpertSupervisedAgent with only the routing collaborators wired."""
    from tensordict import TensorDict

    from protomotions.agents.supervised.multi_expert import MultiExpertSupervisedAgent

    agent = object.__new__(MultiExpertSupervisedAgent)
    agent.device = torch.device("cpu")
    agent.expert_actor_in_keys = ["max_coords_obs"]
    agent._motion_expert = torch.tensor(motion_expert, dtype=torch.long)
    agent.env = SimpleNamespace(
        motion_manager=SimpleNamespace(motion_ids=torch.tensor(motion_ids))
    )

    def make_actor(index):
        def actor(obs_td):
            n = obs_td.batch_size[0]
            # Each expert signs its output so the routing is checkable.
            return TensorDict(
                {"mean_action": torch.full((n, action_dim), float(index))},
                batch_size=[n],
            )

        return actor

    agent.expert_actors = [make_actor(i) for i in range(num_experts)]
    return agent


def test_expert_routing_labels_each_env_with_its_clips_owner():
    from tensordict import TensorDict

    # 4 clips owned by experts 0, 1, 2, 0; 5 envs playing clips 3, 0, 1, 2, 1.
    agent = _routing_agent(motion_expert=[0, 1, 2, 0], motion_ids=[3, 0, 1, 2, 1])
    obs = TensorDict({"expert_max_coords_obs": torch.zeros(5, 7)}, batch_size=[5])
    actions = agent._collect_external_expert_action(obs)
    assert actions.shape == (5, 4)
    assert actions[:, 0].tolist() == [0.0, 0.0, 1.0, 2.0, 1.0]


def test_expert_routing_skips_experts_with_no_environments():
    from tensordict import TensorDict

    # Nothing routes to expert 2; its actor must never be called with an empty
    # batch (lazy modules choke on one) and the result must still be complete.
    called = []
    agent = _routing_agent(motion_expert=[0, 1, 2], motion_ids=[0, 0, 1])
    inner = agent.expert_actors[2]

    def tracking_actor(obs_td):
        called.append(obs_td.batch_size[0])
        return inner(obs_td)

    agent.expert_actors[2] = tracking_actor
    obs = TensorDict({"expert_max_coords_obs": torch.zeros(3, 7)}, batch_size=[3])
    actions = agent._collect_external_expert_action(obs)
    assert called == []
    assert actions[:, 0].tolist() == [0.0, 0.0, 1.0]


def test_routing_table_rejects_a_relabelled_library(tmp_path):
    from protomotions.agents.supervised.multi_expert import MultiExpertSupervisedAgent

    agent = object.__new__(MultiExpertSupervisedAgent)
    agent.device = torch.device("cpu")
    agent.env = SimpleNamespace(
        motion_lib=SimpleNamespace(motion_files=["d/a.motion", "d/b.motion"])
    )
    path = tmp_path / "experts.json"

    import json

    path.write_text(json.dumps({"motion_expert": [0, 1], "motion_names": ["a", "c"]}))
    agent.config = SimpleNamespace(motion_expert_file=str(path))
    with pytest.raises(ValueError, match="different clip order"):
        agent._load_routing_table(2)

    path.write_text(json.dumps({"motion_expert": [0], "motion_names": ["a"]}))
    with pytest.raises(ValueError, match="covers 1 motions"):
        agent._load_routing_table(2)

    path.write_text(json.dumps({"motion_expert": [0, 5], "motion_names": ["a", "b"]}))
    with pytest.raises(ValueError, match="indexes experts"):
        agent._load_routing_table(2)


# --------------------------------------------------------------------------- #
# Experiment wiring
# --------------------------------------------------------------------------- #
class _StubRobotConfig:
    number_of_actions = 69

    def __init__(self):
        self.kinematic_info = SimpleNamespace(
            hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
            dof_limits_lower=torch.zeros(69),
            dof_limits_upper=torch.ones(69),
            dof_names=[f"dof_{i}" for i in range(69)],
            body_names=list(SIM_BODY_NAMES),
        )
        self.trackable_bodies_subset = list(SIM_BODY_NAMES)
        self.contact_observation_bodies = list(SIM_BODY_NAMES)
        self.control = SimpleNamespace(
            control_info={
                name: SimpleNamespace(stiffness=10.0, damping=1.0)
                for name in self.kinematic_info.dof_names
            }
        )
        self.updated = []

    def update_fields(self, **kwargs):
        self.updated.append(kwargs)


def _experiment_args(**overrides):
    values = {
        "motion_file": "motions.pt",
        "scenes_file": None,
        "batch_size": 32,
        "training_max_steps": 1024,
        "contact_graph_file": "graph.pt",
        "motion_expert_file": None,
        "expert_model_path": None,
        "expert_model_paths": None,
        "goal_pose_visible_prob": 0.75,
        "goal_contact_visible_prob": 0.85,
        "goal_full_pose_prob": 0.75,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_experiment_builds_contact_goal_wiring_without_experts(tmp_path):
    from examples.experiments.masked_mimic import contact_graph_transformer as module

    # env_config reads the graph to build contact_state_obs's vocabulary from
    # the same pair list the goals use, so it needs a real file.
    graph_path = tmp_path / "graph.pt"
    torch.save(_toy_graph_payload(), graph_path)

    args = _experiment_args(contact_graph_file=str(graph_path))
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    assert robot_cfg.updated[0]["contact_observation_bodies"] == "all"
    assert robot_cfg.updated[0]["contact_pair_bodies"] == "all"

    env_cfg = module.env_config(robot_cfg, args)
    control = env_cfg.control_components["contact_graph"]
    assert control.graph_file == str(graph_path)
    assert control.num_goal_steps == module.NUM_GOAL_STEPS
    assert control.num_masked_future_steps == module.NUM_GOAL_STEPS
    for key in (
        "contact_goal_obs",
        "contact_goal_masks",
        "contact_state_obs",
        "masked_mimic_target_poses",
    ):
        assert key in env_cfg.observation_components
    # The measured block must be built over the SAME pair list as the goal, or
    # slot k means one thing in the goal and another in the state.
    assert (
        env_cfg.observation_components["contact_state_obs"].static_params["num_pairs"]
        == len(_toy_graph_payload()["pair_names"])
    )
    # No experts requested -> no expert_* block was added.
    assert not [k for k in env_cfg.observation_components if k.startswith("expert_")]
    assert env_cfg.reward_components["diag_contact_goal_ground_iou"].static_params[
        "weight"
    ] == 0.0

    agent_cfg = module.agent_config(robot_cfg, env_cfg, args)
    assert agent_cfg._target_.endswith("MultiExpertSupervisedAgent")
    # Both halves of the goal reach the prior, and they share one token stream.
    assert "contact_goal_obs" in agent_cfg.model.prior.in_keys
    assert "contact_goal_obs" in agent_cfg.model.encoder.in_keys
    token = next(
        m for m in agent_cfg.model.prior.models
        if m.out_keys == ["masked_mimic_target_poses_token"]
    )
    assert "contact_goal_seq" in token.in_keys
    assert "contact_obs_v1" in agent_cfg.model.trunk.in_keys
    # The measured contact state reaches all three heads, and never through a
    # running normaliser: it is binary and most pairs are almost always zero.
    for part in (agent_cfg.model.encoder, agent_cfg.model.prior, agent_cfg.model.trunk):
        assert "contact_state_obs" in part.in_keys
    consumers = [
        m
        for part in (agent_cfg.model.encoder, agent_cfg.model.prior, agent_cfg.model.trunk)
        for m in part.models
        if "contact_state_obs" in getattr(m, "in_keys", [])
    ]
    assert consumers, "contact_state_obs reached no module"
    assert all(not getattr(m, "normalize_obs", False) for m in consumers)


# --------------------------------------------------------------------------- #
# Round 7_1: node identity coarser than the segment's own contact set
# --------------------------------------------------------------------------- #
def test_gather_prefers_the_per_segment_contact_target():
    """A node may be coarser than the set a segment actually held.

    ``--node-identity ground`` keys a node by its ground support set alone so a
    marginal body-body press stops creating a singleton node; the *goal* must
    still name what that segment held, or the coarsening would give back the
    body-body goal channel round 2 added.
    """
    payload = _toy_graph_payload()
    # Both of motion 0's first two segments now live in node 0, but segment 1
    # additionally held L_HAND:G.
    payload["seg_node"] = torch.tensor([[0, 0, 2], [2, -1, -1]])
    payload["seg_contact"] = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    graph = ContactGraph(payload)
    gathered = graph.gather(torch.tensor([0]), torch.tensor([[0, 1]]))
    assert gathered["node"].tolist() == [[0, 0]]
    # Same node, different goal: this is the whole point of the table.
    assert gathered["contact"][0, 0].tolist() == [1.0, 1.0, 0.0]
    assert gathered["contact"][0, 1].tolist() == [1.0, 1.0, 1.0]


def test_gather_falls_back_to_the_node_table_without_seg_contact():
    """Graphs written before ``seg_contact`` keep working unchanged."""
    payload = _toy_graph_payload()
    assert "seg_contact" not in payload
    graph = ContactGraph(payload)
    assert graph.seg_contact is None
    gathered = graph.gather(torch.tensor([0]), torch.tensor([[0, 2]]))
    assert gathered["contact"][0, 0].tolist() == [1.0, 1.0, 0.0]
    assert gathered["contact"][0, 1].tolist() == [0.0, 0.0, 1.0]


def test_seg_contact_layout_must_match_seg_node():
    payload = _toy_graph_payload()
    payload["seg_contact"] = torch.zeros(2, 2, 3)  # one segment column short
    with pytest.raises(ValueError, match="disagree on the segment layout"):
        ContactGraph(payload)


def test_body_pair_identity_none_drops_every_body_body_pair_from_identity():
    """``none`` demotes the whole body-body half; ``load_path`` only the
    both-grounded ones; ``all`` demotes nothing."""
    # Every zone needs its own ``:G`` pair -- the load-path rule asks whether
    # each member of a body-body pair is independently grounded, and the real
    # vocabulary always carries all 15 ground zones (``pair_index``).
    decomposed = [
        ("L_FOOT", None), ("R_FOOT", None), ("L_SHANK", None),
        ("L_UPPER_ARM", None),
        ("L_FOOT", "R_FOOT"), ("L_SHANK", "L_UPPER_ARM"),
    ]
    # Frame 0: standing, feet touching (both grounded) and a shin on an arm
    # (neither grounded).
    active = np.array([[True, True, False, False, True, True]])

    identity, demoted = demote_supported_pairs(active, decomposed, "all")
    assert identity[0].tolist() == [True, True, False, False, True, True]
    assert not demoted.any()

    identity, demoted = demote_supported_pairs(active, decomposed, "load_path")
    # L_FOOT+R_FOOT demoted (both grounded); the shin-on-arm survives.
    assert identity[0].tolist() == [True, True, False, False, False, True]
    assert demoted[0].tolist() == [False, False, False, False, True, False]

    identity, demoted = demote_supported_pairs(active, decomposed, "none")
    assert identity[0].tolist() == [True, True, False, False, False, False]
    assert demoted[0].tolist() == [False, False, False, False, True, True]
    for rule in ("all", "load_path", "none"):
        i, d = demote_supported_pairs(active, decomposed, rule)
        assert not (i & d).any()          # disjoint
        assert ((i | d) == active).all()  # and a partition of `active`
    with pytest.raises(ValueError, match="identity rule must be one of"):
        demote_supported_pairs(active, decomposed, "nonsense")


def test_node_identity_ground_keys_a_node_by_its_ground_set():
    segment = {
        "pairs": ["L_HAND:G", "R_HAND:G", "L_THIGH+L_UPPER_ARM"],
        "pair_ids": [2, 3, 7],
        "ground_pairs": ["L_HAND:G", "R_HAND:G"],
        "ground_pair_ids": [2, 3],
    }
    assert node_identity_pairs(segment, "segment") == (
        ["L_HAND:G", "R_HAND:G", "L_THIGH+L_UPPER_ARM"], [2, 3, 7]
    )
    assert node_identity_pairs(segment, "ground") == (["L_HAND:G", "R_HAND:G"], [2, 3])
    with pytest.raises(ValueError, match="unknown node identity"):
        node_identity_pairs(segment, "nonsense")
