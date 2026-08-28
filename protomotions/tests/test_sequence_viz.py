# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-training stick-figure sequence visualization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protomotions.agents.evaluators.sequence_viz import (
    VizGoal,
    VizSequence,
    derive_edges,
    derive_node_dwell,
    fill_goal_slots,
    render_stick_video,
    resolve_config,
    short_config,
    skeleton_bones,
    strip_supported_pairs,
)
from protomotions.agents.supervised.agent import SupervisedAgent

LOADPATH_GRAPH = (
    Path(__file__).resolve().parents[2]
    / "data/smpl/yoga_contact_graph_student44_loadpath/contact_graph.pt"
)


def test_strip_supported_pairs_applies_the_demotion_rule_to_strings():
    flagged = "L_FOOT:G|R_FOOT:G|L_FOOT+R_FOOT@upright"
    assert strip_supported_pairs(flagged) == "L_FOOT:G|R_FOOT:G@upright"
    # Eagle's hook: the left foot has no ground contact, so the pair stays.
    hook = "R_FOOT:G|L_FOOT+R_SHANK@upright"
    assert strip_supported_pairs(hook) == hook
    # Crow: arms are not grounded, the load path stays.
    crow = "L_HAND:G|R_HAND:G|L_SHANK+L_UPPER_ARM|R_SHANK+R_UPPER_ARM@upright"
    assert strip_supported_pairs(crow) == crow


@pytest.mark.skipif(not LOADPATH_GRAPH.is_file(), reason="load-path graph not built")
def test_resolve_config_falls_back_across_graph_builds():
    from protomotions.components.contact_graph import ContactGraph

    graph = ContactGraph.from_file(str(LOADPATH_GRAPH), device="cpu")
    plain = resolve_config(graph, "L_FOOT:G|R_FOOT:G@upright")
    assert plain is not None
    # A plan written against the pre-load-path build names the flagged stand;
    # it resolves to the merged node here.
    flagged = resolve_config(graph, "L_FOOT:G|R_FOOT:G|L_FOOT+R_FOOT@upright")
    assert flagged == plain
    assert resolve_config(graph, "HEAD:G|PELVIS:G@supine_nonsense") is None


@pytest.mark.skipif(not LOADPATH_GRAPH.is_file(), reason="load-path graph not built")
def test_derived_holds_and_edges_match_the_known_graph_shape():
    from protomotions.components.contact_graph import ContactGraph

    graph = ContactGraph.from_file(str(LOADPATH_GRAPH), device="cpu")
    dwell = derive_node_dwell(graph)
    top_node = max(dwell, key=lambda n: dwell[n][0])
    assert graph.node_keys[top_node] == "L_FOOT:G|R_FOOT:G@upright"
    _total, motion, t_hold = dwell[top_node]
    assert motion >= 0 and t_hold > 0.0

    edges = derive_edges(graph)
    assert edges, "the graph has transitions"
    src, dst, count, motion, t_src, t_dst = edges[0]
    assert count >= 10  # the all-fours <-> lift-foot edges are x18-19
    assert src != dst and t_dst > 0.0 and motion >= 0


def _sequences():
    walk = VizSequence(
        name="walk",
        start_motion=3,
        start_time=0.0,
        goals=[
            VizGoal("a", node=5, pose_motion=3, pose_time=1.0, reach_s=2.0, hold_s=1.0),
            VizGoal("b", node=7, pose_motion=3, pose_time=4.0, reach_s=1.0, hold_s=1.0),
        ],
    )
    hold = VizSequence(
        name="hold",
        start_motion=9,
        start_time=2.0,
        goals=[VizGoal("h", node=2, pose_motion=9, pose_time=2.0, reach_s=0.5, hold_s=8.0)],
    )
    return [walk, hold]


def test_fill_goal_slots_tiles_sequences_and_arms_deadlines():
    sequences = _sequences()
    env_sequence = torch.tensor([0, 1, 0, 1])
    slots = fill_goal_slots(
        sequences, env_sequence, t=0.0, slots=5, hold_lead_s=1.2,
        device=torch.device("cpu"),
    )

    # Env 0 follows the walk: slot 0 = goal a, slot 1 = goal b, rest padded.
    assert slots["node_ids"][0].tolist() == [5, 7, -1, -1, -1]
    assert slots["node_ids"][1].tolist() == [2, -1, -1, -1, -1]
    assert torch.equal(slots["node_ids"][0], slots["node_ids"][2])
    # Goal a's reach ends at 2.0 s; goal b's at 4.0 s.
    assert slots["time_offsets"][0, 0] == pytest.approx(2.0)
    assert slots["time_offsets"][0, 1] == pytest.approx(4.0)
    assert bool(slots["pose_visible"][0, 0]) and not bool(slots["pose_visible"][0, 2])


def test_fill_goal_slots_parks_on_the_final_goal_after_the_end():
    sequences = _sequences()
    env_sequence = torch.tensor([0, 1])
    slots = fill_goal_slots(
        sequences, env_sequence, t=30.0, slots=5, hold_lead_s=1.2,
        device=torch.device("cpu"),
    )
    # Both sequences are past their end: final goal, deadline parked at lead.
    assert slots["node_ids"][0, 0] == 7
    assert slots["node_ids"][1, 0] == 2
    assert slots["time_offsets"][0, 0] == pytest.approx(1.2)
    assert slots["time_offsets"][1, 0] == pytest.approx(1.2)


def test_skeleton_bones_joins_by_name_across_body_orders():
    common = ["Pelvis", "L_Hip", "R_Hip", "Torso"]
    parents = [-1, 0, 0, 0]
    sim = ["Torso", "R_Hip", "Pelvis", "L_Hip"]  # deliberately shuffled

    bones = skeleton_bones(sim, common, parents)

    by_child = {child: parent for child, parent, _c in bones}
    assert by_child[sim.index("L_Hip")] == sim.index("Pelvis")
    assert by_child[sim.index("Torso")] == sim.index("Pelvis")
    colors = {sim[child]: color for child, _p, color in bones}
    assert colors["L_Hip"] != colors["R_Hip"]


def test_short_config_compacts_zone_names():
    text = short_config("L_FOOT:G|R_FOOT:G|L_SHANK+L_UPPER_ARM@upright")
    assert "LF" in text and "RF" in text and "LS" in text and "@upri" in text


def test_render_stick_video_writes_an_mp4(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, 1.0, size=(6, 4, 3)).astype(np.float32)
    bones = [(1, 0, "#d62728"), (2, 0, "#1f77b4"), (3, 2, "#444444")]
    out = tmp_path / "seq.mp4"

    render_stick_video(
        positions, bones, [f"t={i}" for i in range(6)], out, fps=5, video_px=200
    )

    assert out.is_file() and out.stat().st_size > 1000


def test_render_stick_video_frames_are_not_frozen(tmp_path):
    # buffer_rgba() returns a view into the reused canvas buffer; without an
    # explicit copy every frame aliases the final draw and the video plays as
    # a single frozen image (the v4 panel defect). An articulating pose must
    # therefore decode into frames that actually differ.
    steps = 8
    angles = np.linspace(0.0, np.pi, steps, dtype=np.float32)
    positions = np.zeros((steps, 3, 3), dtype=np.float32)
    positions[:, 1] = [0.0, 0.0, 1.0]
    positions[:, 2, 0] = 0.6 * np.cos(angles)
    positions[:, 2, 2] = 1.0 + 0.6 * np.sin(angles)
    bones = [(1, 0, "#444444"), (2, 1, "#d62728")]
    out = tmp_path / "move.mp4"

    render_stick_video(positions, bones, [""] * steps, out, fps=5, video_px=200)

    import imageio.v2 as imageio

    frames = imageio.mimread(str(out), memtest=False)
    first = np.asarray(frames[0], dtype=np.int16)
    last = np.asarray(frames[-1], dtype=np.int16)
    assert np.abs(first - last).max() > 30


def test_skeleton_bones_is_the_identity_join_for_common_order_positions():
    # get_robot_state() returns bodies in COMMON order, so the runner joins
    # the tree onto kinematic_info's own names: bone indices must equal the
    # (child, parent) pairs of parent_indices verbatim.
    common = ["Pelvis", "L_Hip", "R_Hip", "Torso"]
    parents = [-1, 0, 0, 0]

    bones = skeleton_bones(common, common, parents)

    assert [(child, parent) for child, parent, _c in bones] == [(1, 0), (2, 0), (3, 0)]


def test_resolve_clip_prefers_the_original_over_derived_hold_clips():
    from protomotions.agents.evaluators.sequence_viz import SequenceVizRunner

    runner = object.__new__(SequenceVizRunner)
    runner.motion_names = [
        "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
        "hold_t0007.10_220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
        "220923_Cobra_Pose_or_Bhujangasana_-a",
    ]

    assert runner._resolve_clip("Adho_Mukha_Svanasana_-a") == 0
    assert runner._resolve_clip("Cobra") == 2
    # A genuinely ambiguous needle (two non-hold matches) still refuses.
    runner.motion_names.append("220926_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a")
    assert runner._resolve_clip("Adho_Mukha_Svanasana_-a") is None


def _viz_agent(epoch, every=500, rank=0):
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        sequence_viz=SimpleNamespace(viz_every=every) if every else None
    )
    agent.current_epoch = epoch
    agent.fabric = SimpleNamespace(global_rank=rank, loggers=[])
    agent._skip_next_policy_update = False
    return agent


def test_sequence_viz_trigger_gating(monkeypatch):
    ran = []

    class _Runner:
        def __init__(self, agent, config):
            pass

        def run(self, epoch):
            ran.append(epoch)
            return {}, {}

    import protomotions.agents.evaluators.sequence_viz as viz_module

    monkeypatch.setattr(viz_module, "SequenceVizRunner", _Runner)

    # Off-cycle epochs, epoch 0, other ranks, and no config never run.
    for agent in (
        _viz_agent(epoch=499),
        _viz_agent(epoch=0),
        _viz_agent(epoch=500, rank=1),
        _viz_agent(epoch=500, every=None),
    ):
        SupervisedAgent._maybe_run_sequence_viz(agent)
        assert not ran and not agent._skip_next_policy_update

    agent = _viz_agent(epoch=500)
    SupervisedAgent._maybe_run_sequence_viz(agent)
    assert ran == [500]
    assert agent._skip_next_policy_update


def test_sequence_viz_failure_disables_without_raising(monkeypatch):
    class _ExplodingRunner:
        def __init__(self, agent, config):
            raise RuntimeError("boom")

    import protomotions.agents.evaluators.sequence_viz as viz_module

    monkeypatch.setattr(viz_module, "SequenceVizRunner", _ExplodingRunner)

    agent = _viz_agent(epoch=500)
    SupervisedAgent._maybe_run_sequence_viz(agent)
    assert agent._sequence_viz_disabled

    # Subsequent triggers are no-ops.
    agent.current_epoch = 1000
    SupervisedAgent._maybe_run_sequence_viz(agent)
