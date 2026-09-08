# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-training stick-figure sequence visualization."""

from __future__ import annotations

import json
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


def _pose_error_runner(reference_pose, num_bodies=5, body_ids=(0, 2, 4)):
    """A SequenceVizRunner stub carrying only what `_goal_pose_errors` reads."""
    from protomotions.agents.evaluators.sequence_viz import SequenceVizRunner

    def get_motion_state(motion_ids, motion_times):
        n = len(motion_ids)
        rot = torch.zeros(n, num_bodies, 4)
        rot[..., 3] = 1.0
        return SimpleNamespace(
            rigid_body_pos=reference_pose.expand(n, -1, -1).clone(),
            rigid_body_rot=rot,
        )

    runner = object.__new__(SequenceVizRunner)
    runner.device = torch.device("cpu")
    runner.pelvis_index = 0
    runner.control = SimpleNamespace(
        conditionable_body_ids=torch.tensor(body_ids, dtype=torch.long)
    )
    runner.env = SimpleNamespace(
        motion_lib=SimpleNamespace(get_motion_state=get_motion_state)
    )
    return runner


def test_goal_pose_error_scores_the_pose_half_the_iou_cannot_see():
    """A pose the IoU would call perfect, scored by how far off it actually is."""
    num_bodies = 5
    reference = torch.zeros(1, num_bodies, 3)
    runner = _pose_error_runner(reference, num_bodies=num_bodies)

    sequences = [
        VizSequence(
            name="seq",
            start_motion=0,
            start_time=0.0,
            goals=[VizGoal("a", 0, 0, 0.0, reach_s=1.0, hold_s=1.0)],
        )
    ]
    # Frame 0 sits on the goal; frame 1 has every non-pelvis body 0.30 m off.
    positions = torch.zeros(2, 1, num_bodies, 3)
    positions[1, 0, 1:, 2] = 0.30
    root_rots = torch.zeros(2, 1, 4)
    root_rots[..., 3] = 1.0

    errors = runner._goal_pose_errors(sequences, positions, root_rots, [0.0, 0.5])

    assert errors.shape == (2, 1)
    assert errors[0, 0] == pytest.approx(0.0, abs=1e-6)
    # Conditionable bodies are (0, 2, 4); body 0 is the pelvis reference, so two
    # of the three are 0.30 m out.
    assert errors[1, 0] == pytest.approx(0.30 * 2 / 3, abs=1e-6)


def test_goal_pose_error_follows_the_active_goal_through_the_sequence():
    num_bodies = 5
    reference = torch.zeros(1, num_bodies, 3)
    runner = _pose_error_runner(reference, num_bodies=num_bodies)
    sequences = [
        VizSequence(
            name="seq",
            start_motion=0,
            start_time=0.0,
            goals=[
                VizGoal("first", 0, 0, 0.0, reach_s=1.0, hold_s=0.0),
                VizGoal("second", 0, 0, 0.0, reach_s=1.0, hold_s=0.0),
            ],
        )
    ]
    positions = torch.zeros(2, 1, num_bodies, 3)
    root_rots = torch.zeros(2, 1, 4)
    root_rots[..., 3] = 1.0

    # t = 0.5 is inside goal 0's window, t = 1.5 inside goal 1's.
    assert sequences[0].active_index(0.5) == 0
    assert sequences[0].active_index(1.5) == 1
    errors = runner._goal_pose_errors(sequences, positions, root_rots, [0.5, 1.5])
    assert np.allclose(errors, 0.0, atol=1e-6)


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


# --------------------------------------------------------------------------- #
# Round 7_1: the panel's per-sequence summary must name the HARDEST goal
# --------------------------------------------------------------------------- #
def _encode_runner(tmp_path, monkeypatch, pose_errors):
    """A ``SequenceVizRunner`` stub wired for ``_encode`` only.

    ``render_stick_video`` is replaced because this test is about the scalars,
    not the pixels, and rendering is by far the slowest part of the panel.
    """
    from protomotions.agents.evaluators import sequence_viz as module

    monkeypatch.setattr(module, "render_stick_video", lambda *a, **k: None)
    runner = module.SequenceVizRunner.__new__(module.SequenceVizRunner)
    runner.agent = SimpleNamespace(root_dir=str(tmp_path))
    runner.config = SimpleNamespace(render_fps=10, video_px=64)
    runner.bones = []
    runner.pelvis_index = 0
    runner.head_index = None
    # Two ground zones; goal node 0 wants the first, node 1 wants both.
    runner.control = SimpleNamespace(
        _ground_zone_names=["L_FOOT", "R_FOOT"],
        _ground_pair_ids=torch.tensor([0, 1]),
    )
    runner.graph = SimpleNamespace(
        node_contact=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        node_keys=["one_foot@upright", "two_feet@upright"],
    )
    monkeypatch.setattr(
        module.SequenceVizRunner, "_goal_pose_errors",
        lambda self, *a, **k: pose_errors,
    )
    return runner


def test_panel_reports_the_hardest_goal_not_just_the_easiest(tmp_path, monkeypatch):
    """``best_pose_err`` is a minimum over the whole sequence, so any plan with a
    trivial goal saturates it -- measured at 0.018-0.077 m for every v7_1
    sequence at every epoch, including ones that never reach anything.
    ``worst_goal_pose_err`` is the counterpart that cannot be saturated."""
    # Goal 0 (frames 0-1) is nailed; goal 1 (frames 2-3) is missed by 0.6 m.
    pose_errors = np.array([[0.02], [0.03], [0.60], [0.65]], dtype=np.float32)
    runner = _encode_runner(tmp_path, monkeypatch, pose_errors)

    sequences = [
        VizSequence(
            name="seq",
            start_motion=0,
            start_time=0.0,
            goals=[
                VizGoal("easy", 0, 0, 0.0, reach_s=0.5, hold_s=0.5),
                VizGoal("hard", 1, 0, 0.0, reach_s=0.5, hold_s=0.5),
            ],
        )
    ]
    frame_times = [0.0, 0.5, 1.0, 1.5]
    positions = [torch.zeros(1, 3, 3) for _ in frame_times]
    root_rots = [torch.tensor([[0.0, 0.0, 0.0, 1.0]]) for _ in frame_times]
    # Only the left foot is ever down: exact for goal 0, half for goal 1.
    zones = [torch.tensor([[1.0, 0.0]]) for _ in frame_times]

    _, scalars = runner._encode(
        7, sequences, positions, root_rots, zones, frame_times, out_dir=tmp_path
    )

    assert scalars["viz/seq/goal0_easy/best_pose_err"] == pytest.approx(0.02)
    assert scalars["viz/seq/goal1_hard/best_pose_err"] == pytest.approx(0.60)
    # The saturating summary, and the one that is not.
    assert scalars["viz/seq/best_pose_err"] == pytest.approx(0.02)
    assert scalars["viz/seq/worst_goal_pose_err"] == pytest.approx(0.60)
    # Same shape on the contact half: goal 0 is exact, goal 1 never is.
    assert scalars["viz/seq/max_goal_best_iou"] == pytest.approx(1.0)
    assert scalars["viz/seq/min_goal_best_iou"] == pytest.approx(0.5)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary[0]["worst_goal"] == "hard"
    assert summary[0]["worst_goal_pose_err"] == pytest.approx(0.60, abs=1e-3)
    assert [g["goal"] for g in summary[0]["per_goal"]] == ["easy", "hard"]


# --------------------------------------------------------------------------- #
# Round 8_1: the panel can log videos without the scalar charts
# --------------------------------------------------------------------------- #
class _FakeWandbLogger:
    """Stands in for lightning's WandbLogger, which the agent finds by class name."""

    def __init__(self):
        self.logged = []
        self.experiment = SimpleNamespace(
            log=lambda payload, step=None: self.logged.append((payload, step))
        )


def _viz_logging_agent(log_scalars, monkeypatch, tmp_path):
    import sys, types

    # `_log_sequence_viz` imports wandb for the Video wrapper only.
    fake = types.ModuleType("wandb")
    fake.Video = lambda path, format=None: f"<video {Path(path).name}>"
    monkeypatch.setitem(sys.modules, "wandb", fake)

    logger = _FakeWandbLogger()
    logger.__class__.__name__ = "WandbLogger"
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        sequence_viz=SimpleNamespace(log_scalars=log_scalars)
    )
    agent.fabric = SimpleNamespace(loggers=[logger])
    agent.current_epoch = 500
    videos = {"viz/seq_a": tmp_path / "seq_a.mp4"}
    scalars = {"viz/seq_a/best_pose_err": 0.03, "viz/seq_a/goal0_stand/best_iou": 1.0}
    SupervisedAgent._log_sequence_viz(agent, videos, scalars)
    return logger


def test_panel_logs_videos_without_the_scalar_charts(monkeypatch, tmp_path):
    """`log_scalars=False` ships the videos and drops the charts.

    Early in training each scalar is one unseeded nucleus draw per epoch, so the
    charts are a coin flip (round 7_1 §5.2) and invite the single-draw reading
    round 7 §11.5 had to retract.
    """
    logger = _viz_logging_agent(False, monkeypatch, tmp_path)
    (payload, step) = logger.logged[0]
    assert step == 500
    assert set(payload) == {"viz/seq_a"}          # the video, and nothing else


def test_panel_logs_scalars_by_default(monkeypatch, tmp_path):
    logger = _viz_logging_agent(True, monkeypatch, tmp_path)
    (payload, _) = logger.logged[0]
    assert "viz/seq_a" in payload
    assert payload["viz/seq_a/best_pose_err"] == 0.03
    assert payload["viz/seq_a/goal0_stand/best_iou"] == 1.0


def test_panel_scalars_still_reach_summary_json_when_charts_are_off(
    tmp_path, monkeypatch
):
    """The gate is at the logger, so the offline record is untouched."""
    pose_errors = np.array([[0.02], [0.60]], dtype=np.float32)
    runner = _encode_runner(tmp_path, monkeypatch, pose_errors)
    sequences = [
        VizSequence(
            name="seq", start_motion=0, start_time=0.0,
            goals=[VizGoal("a", 0, 0, 0.0, reach_s=0.5, hold_s=0.5)],
        )
    ]
    frame_times = [0.0, 0.5]
    _, scalars = runner._encode(
        1, sequences, [torch.zeros(1, 3, 3)] * 2,
        [torch.tensor([[0.0, 0.0, 0.0, 1.0]])] * 2,
        [torch.tensor([[1.0, 0.0]])] * 2, frame_times, out_dir=tmp_path,
    )
    assert scalars  # `_encode` always returns them; only the logger drops them
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary[0]["per_goal"][0]["best_pose_err"] == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Round 9 diagnosis §6.1/§6.2: replica scoring and the per-goal hold metrics
# --------------------------------------------------------------------------- #
def _hold_sequence():
    """One goal, reached at frame 1 and abandoned at frame 3."""
    return [
        VizSequence(
            name="hold",
            start_motion=0,
            start_time=0.0,
            goals=[VizGoal("stay", 0, 0, 0.0, reach_s=0.5, hold_s=1.5)],
        )
    ]


def test_per_goal_hold_metrics_see_a_hold_that_best_pose_err_cannot(
    tmp_path, monkeypatch
):
    """A minimum over time cannot distinguish "touched it once" from "held it".

    v9's ``hold_probe_standing`` read ``worst_goal_pose_err`` 0.02-0.03 m at all
    26 panels while its hold-window mean was bimodal and failed on 19 of them.
    ``hold_pose_err`` / ``end_pose_err`` / ``time_held_s`` are the per-goal
    twins of the ``hold_iou`` the contact half has had since round 7_1.
    """
    # Frames at 0.0/0.5/1.0/1.5 s; the goal's hold window is [0.5, 2.0).
    pose_errors = np.array([[0.90], [0.05], [0.70], [0.80]], dtype=np.float32)
    runner = _encode_runner(tmp_path, monkeypatch, pose_errors)
    frame_times = [0.0, 0.5, 1.0, 1.5]
    positions = [torch.zeros(1, 3, 3) for _ in frame_times]
    root_rots = [torch.tensor([[0.0, 0.0, 0.0, 1.0]]) for _ in frame_times]
    zones = [torch.tensor([[1.0, 0.0]]) for _ in frame_times]

    _, scalars = runner._encode(
        1, _hold_sequence(), positions, root_rots, zones, frame_times,
        out_dir=tmp_path,
    )

    goal = json.loads((tmp_path / "summary.json").read_text())[0]["per_goal"][0]
    # The old number says the pose was nailed...
    assert goal["best_pose_err"] == pytest.approx(0.05, abs=1e-3)
    # ...the new ones say it did not stay there.
    assert goal["hold_pose_err"] == pytest.approx((0.05 + 0.70 + 0.80) / 3, abs=1e-3)
    assert goal["end_pose_err"] == pytest.approx(0.80, abs=1e-3)
    # Arrived at t=0.5 (0.05 <= 0.15), left at t=1.0 (0.70 > 0.30).
    assert goal["time_held_s"] == pytest.approx(0.5, abs=1e-6)
    assert scalars["viz/hold/goal0_stay/hold_pose_err"] > 0.15


def test_replicas_are_scored_as_a_rate_not_a_single_draw(tmp_path, monkeypatch):
    """``env_sequence = env_ids % num_seq``, so column ``s + k*S`` is a replica.

    The panel stepped ~36 of them per plan and scored the first; the rate over
    all of them is what a 12 s hold (45 nucleus draws at chunk_steps=8) has to
    be judged on.
    """
    # Four replicas of one sequence: two hold the pose, two leave it.
    pose_errors = np.array(
        [
            [0.90, 0.90, 0.90, 0.90],
            [0.05, 0.05, 0.05, 0.05],
            [0.04, 0.60, 0.05, 0.70],
            [0.05, 0.65, 0.04, 0.75],
        ],
        dtype=np.float32,
    )
    runner = _encode_runner(tmp_path, monkeypatch, pose_errors)
    frame_times = [0.0, 0.5, 1.0, 1.5]
    positions = [torch.zeros(4, 3, 3) for _ in frame_times]
    root_rots = [torch.zeros(4, 4) for _ in frame_times]
    for rot in root_rots:
        rot[:, 3] = 1.0
    zones = [torch.tensor([[1.0, 0.0]] * 4) for _ in frame_times]

    _, scalars = runner._encode(
        1, _hold_sequence(), positions, root_rots, zones, frame_times,
        out_dir=tmp_path, replicas=4,
    )

    entry = json.loads((tmp_path / "summary.json").read_text())[0]
    assert entry["replicas"] == 4
    assert entry["hold_success_rate"] == pytest.approx(0.5)
    assert scalars["viz/hold/hold_success_rate"] == pytest.approx(0.5)
    # Replica 0 held, so the scalar that names the video still says so — the
    # video and its numbers must describe the same rollout.
    assert entry["final_goal_pose_err"] < 0.15
    assert entry["per_goal_agg"][0]["hold_rate"] == pytest.approx(0.5)
    assert entry["per_goal_agg"][0]["n"] == 4


def test_hold_lead_park_mode_leaves_the_reach_countdown_alone():
    """`clamp` inflates every reach window shorter than the lead, which
    confounded the first deadline sweep: a 5-goal plan with 2 s reaches never
    saw a deadline below the lead and stopped meeting its own waypoints."""
    sequences = [
        VizSequence(
            name="two",
            start_motion=0,
            start_time=0.0,
            goals=[
                VizGoal("reach", 0, 0, 0.0, reach_s=2.0, hold_s=2.0),
                VizGoal("next", 1, 0, 0.0, reach_s=2.0, hold_s=2.0),
            ],
        )
    ]
    env_sequence = torch.zeros(1, dtype=torch.long)
    # t = 1.0 s is inside the first reach window: 1.0 s left to the target.
    clamp = fill_goal_slots(sequences, env_sequence, 1.0, 2, 5.0,
                            torch.device("cpu"), "clamp")
    park = fill_goal_slots(sequences, env_sequence, 1.0, 2, 5.0,
                           torch.device("cpu"), "park")
    assert clamp["time_offsets"][0, 0].item() == pytest.approx(5.0)
    assert park["time_offsets"][0, 0].item() == pytest.approx(1.0)
    # ...and both park at the lead once the reach window has expired (t = 3 s).
    for mode in ("clamp", "park"):
        held = fill_goal_slots(sequences, env_sequence, 3.0, 2, 5.0,
                               torch.device("cpu"), mode)
        assert held["time_offsets"][0, 0].item() == pytest.approx(5.0)
