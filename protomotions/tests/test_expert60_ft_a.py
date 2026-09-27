"""Tests for expert60 fine-tune A: hold repair, monotonic schedule, hold-aware curriculum.

``expert_revist/run1_gap_analysis.MD`` §3 step 2. Pure CPU; the data-level checks
skip when the generated artefacts are not on disk.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "data" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from build_hold_graph import build_graph  # noqa: E402
from repair_hold_manifest import best_yaw_dist, repaired_end_frame  # noqa: E402

from protomotions.agents.evaluators.hold_curriculum import (  # noqa: E402
    HoldWindow,
    ScoreParams,
    best_yaw_distance,
    mixture_sampling_probs,
    score_clip,
    update_score_ema,
)
from protomotions.components.contact_graph import ContactGraph  # noqa: E402

PAIRS = ["L_FOOT:G", "R_FOOT:G", "L_HAND:G", "R_HAND:G"]
ORIENTS = ["upright", "prone"]
FT_A_GRAPH = REPO / "data/smpl/yoga_hold_graph_expert60_ftA/contact_graph.pt"
FT_A_MANIFEST = REPO / "data/smpl/expert60/holds_repaired.yaml"


# --------------------------------------------------------------------------- #
# Hold repair
# --------------------------------------------------------------------------- #
def test_repaired_end_frame_cuts_before_first_exceedance():
    dist = np.array([0.0, 0.02, 0.05, 0.11, 0.04, 0.2])
    assert repaired_end_frame(dist, frame_hold=100, delta=0.10) == 102
    assert repaired_end_frame(np.array([0.0, 0.01]), 7, 0.10) == 8  # nothing to cut
    assert repaired_end_frame(np.array([0.0, 0.5]), 7, 0.10) == 7   # leaves the exemplar


def test_best_yaw_distance_ignores_heading_in_both_implementations():
    rng = np.random.default_rng(0)
    pose = rng.normal(size=(6, 3))
    pose -= pose[:1]
    th = 1.3
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    turned = pose @ rot.T
    assert best_yaw_dist(turned[None], pose)[0] < 1e-9
    d = best_yaw_distance(torch.tensor(turned[None]), torch.tensor(pose))
    assert float(d[0]) < 1e-6


@pytest.mark.skipif(not FT_A_MANIFEST.exists(), reason="repaired manifest not generated")
def test_repaired_manifest_holds_stay_within_delta_after_the_exemplar():
    import yaml
    from repair_hold_manifest import post_exemplar_distances

    manifest = yaml.safe_load(open(FT_A_MANIFEST))
    delta = manifest["repair"]["delta_m"]
    for clip in manifest["clips"][:12]:
        pos = torch.load(clip["source"], map_location="cpu", weights_only=False)["rigid_body_pos"].numpy()
        for h in clip["holds"]:
            assert post_exemplar_distances(pos, h).max() <= delta + 1e-9, (clip["stem"], h["name"])


# --------------------------------------------------------------------------- #
# Monotonic schedule
# --------------------------------------------------------------------------- #
def _graph_with_near_start_hold(tmp_path) -> ContactGraph:
    def hold(name, t_start, t_hold, t_end, pairs, orient="upright", extend=False):
        return {"name": name, "t_start": t_start, "t_hold": t_hold, "t_end": t_end, "pairs": pairs,
                "orientation": orient, "extend": extend, "frame_start": int(t_start * 60),
                "frame_hold": int(t_hold * 60), "frame_end": int(t_end * 60),
                "speed_at_hold": 0.01, "pelvis_z": 0.9}

    clips = [{
        "stem": "clip", "group": "g", "family": "X", "source_stem": "clip",
        "holds": [
            hold("standing", 0.0, 0.0, 1.0, ["L_FOOT:G", "R_FOOT:G"]),
            # hold frame 0.05 s after its start: the legacy lookahead skips it
            hold("X", 3.0, 3.05, 5.0, ["L_HAND:G", "R_HAND:G"], "prone", extend=True),
            hold("standing", 8.0, 8.5, 9.0, ["L_FOOT:G", "R_FOOT:G"]),
        ],
    }]
    payload, _ = build_graph(clips, ["clip"], PAIRS, ORIENTS)
    torch.save(payload, tmp_path / "g.pt")
    return ContactGraph.from_file(tmp_path / "g.pt")


def test_interval_schedule_removes_the_backward_step(tmp_path):
    g = _graph_with_near_start_hold(tmp_path)
    t = torch.arange(0.0, 9.5, 1 / 30)
    ids = torch.zeros(len(t), dtype=torch.long)
    legacy, lv = g.next_goal_indices(ids, t, 2, include_current=True)
    interval, iv = g.next_goal_indices(ids, t, 2, include_current=True, interval=True)
    s_legacy = legacy[:, 0][lv[:, 0]].numpy()
    s_interval = interval[:, 0][iv[:, 0]].numpy()
    assert (np.diff(s_legacy) < 0).any(), "fixture should reproduce the legacy backward step"
    assert (np.diff(s_interval) >= 0).all()
    # in the gap before X the interval rule already commands X, then the next hold
    gap = (t > 1.0) & (t < 3.0)
    assert (interval[gap, 0] == 1).all() and (interval[gap, 1] == 2).all()
    # inside a hold both rules agree
    inside = (t >= 3.0) & (t <= 5.0)
    assert torch.equal(interval[inside], legacy[inside])


def test_interval_schedule_guards(tmp_path):
    g = _graph_with_near_start_hold(tmp_path)
    t = torch.tensor([0.5])
    ids = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError):
        g.next_goal_indices(ids, t, 2, include_current=False, interval=True)
    with pytest.raises(ValueError):
        g.next_goal_indices(ids, t, 2, include_current=True, interval=True,
                            promote_k=torch.zeros(1, dtype=torch.long))


@pytest.mark.skipif(not FT_A_GRAPH.exists(), reason="fine-tune A graph not generated")
def test_interval_schedule_is_monotonic_on_every_ft_a_variant():
    g = ContactGraph.from_file(FT_A_GRAPH)
    for m in range(g.seg_count.shape[0]):
        end = float(g.seg_end[m][torch.isfinite(g.seg_end[m])].max()) + 1.0
        t = torch.arange(0.0, end, 1 / 30)
        idx, valid = g.next_goal_indices(torch.full((len(t),), m), t, 2, include_current=True, interval=True)
        s0 = idx[:, 0][valid[:, 0]].numpy()
        assert (np.diff(s0) >= 0).all(), m


# --------------------------------------------------------------------------- #
# Performance score and curriculum
# --------------------------------------------------------------------------- #
NAMES = ["Pelvis", "L_Ankle", "L_Toe", "R_Ankle", "R_Toe", "L_Wrist", "L_Hand",
         "R_Wrist", "R_Hand", "Head"]
GOAL = [NAMES.index(b) for b in ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")]
ZONES = {"L_FOOT": [1, 2], "R_FOOT": [3, 4], "L_HAND": [5, 6], "R_HAND": [7, 8]}


def _crow_pose(feet_z: float) -> torch.Tensor:
    p = torch.zeros(len(NAMES), 3)
    p[0] = torch.tensor([0.0, 0.0, 0.8])
    for i in (1, 2, 3, 4):
        p[i] = torch.tensor([0.1 * i, 0.2, feet_z])
    for i in (5, 6, 7, 8):
        p[i] = torch.tensor([0.1 * i, -0.3, 0.03])
    p[9] = torch.tensor([0.0, -0.4, 0.5])
    return p


def test_score_is_one_for_perfect_tracking_and_ignores_xy_offset():
    T = 60
    ref = _crow_pose(0.4).expand(T, -1, -1).clone()
    sim = ref.clone()
    sim[..., 0] += 25.0  # terrain offset
    times = (torch.arange(T) + 1) / 30.0
    s = score_clip(sim, ref, times, [HoldWindow(0.5, 1.5, True)], ref[:1], GOAL, ZONES, ScoreParams())
    assert s["p_track"] == 1.0 and s["p_hold"] == 1.0 and s["score"] == 1.0
    assert s["support_violation"] == 0.0


def test_feet_down_crow_fails_the_hold_even_with_a_small_pose_error():
    T = 60
    ref = _crow_pose(0.4).expand(T, -1, -1).clone()
    sim = _crow_pose(0.04).expand(T, -1, -1).clone()  # feet on the floor
    times = (torch.arange(T) + 1) / 30.0
    params = ScoreParams(pose_threshold_m=0.5)  # pose alone would pass
    s = score_clip(sim, ref, times, [HoldWindow(0.5, 1.5, True)], ref[:1], GOAL, ZONES, params)
    assert s["p_hold"] == 0.0 and s["support_violation"] == 1.0
    assert s["p_track"] == 1.0 and math.isclose(s["score"], 0.5)


def test_feet_down_is_not_penalised_when_the_reference_feet_are_down():
    T = 30
    ref = _crow_pose(0.05).expand(T, -1, -1).clone()  # e.g. plow's float-bias mislabel
    times = (torch.arange(T) + 1) / 30.0
    s = score_clip(ref.clone(), ref, times, [HoldWindow(0.1, 0.9)], ref[:1], GOAL, ZONES, ScoreParams())
    assert s["support_violation"] == 0.0 and s["p_hold"] == 1.0


def test_tracking_failure_fraction_and_no_hold_fallback():
    T = 30
    ref = _crow_pose(0.4).expand(T, -1, -1).clone()
    sim = ref.clone()
    sim[15:, 9, 2] += 1.0  # head 1 m off for half the clip
    times = (torch.arange(T) + 1) / 30.0
    s = score_clip(sim, ref, times, [], None, GOAL, ZONES, ScoreParams())
    assert math.isclose(s["p_track"], 0.5) and math.isnan(s["p_hold"]) and math.isclose(s["score"], 0.5)


def test_score_ema_and_mixture_probabilities():
    first = update_score_ema(None, torch.tensor([0.2, float("nan"), 1.0]), 0.5)
    assert torch.allclose(first, torch.tensor([0.2, 1.0, 1.0]))
    second = update_score_ema(first, torch.tensor([0.6, float("nan"), 0.0]), 0.5)
    assert torch.allclose(second, torch.tensor([0.4, 1.0, 0.5]))

    scores = torch.tensor([0.0, 0.5, 1.0, 1.0])
    p = mixture_sampling_probs(scores, uniform_fraction=0.8, eps=0.0)
    assert math.isclose(float(p.sum()), 1.0, rel_tol=1e-6)
    assert (p >= 0.8 / 4 - 1e-9).all()
    assert p[0] > p[1] > p[2] and math.isclose(float(p[2]), 0.2, rel_tol=1e-6)
    # the prioritized share never exceeds 1 - uniform_fraction
    assert math.isclose(float((p - 0.2).sum()), 0.2, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# Experiment wiring
# --------------------------------------------------------------------------- #
def test_goal_conditioned_experiment_wires_fine_tune_a(tmp_path):
    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.agents.evaluators.config import (
        HoldCurriculumEvaluatorConfig,
        MimicEvaluatorConfig,
    )
    from protomotions.robot_configs.factory import robot_config

    g = _graph_with_near_start_hold(tmp_path)  # writes tmp_path/g.pt
    del g
    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)

    def build(argv):
        args = parser.parse_args(["--hold-graph-file", str(tmp_path / "g.pt")] + argv)
        args.motion_file = "unused.pt"
        args.scenes_file = None
        args.batch_size = 32
        args.training_max_steps = 256
        robot_cfg = robot_config("smpl_yogi")
        exp.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
        env_cfg = exp.env_config(robot_cfg, args)
        return env_cfg, exp.agent_config(robot_cfg, env_cfg, args)

    env_cfg, agent_cfg = build([])  # round-1 defaults
    assert type(agent_cfg.evaluator) is MimicEvaluatorConfig
    assert agent_cfg.evaluator.max_eval_steps == 600 and agent_cfg.evaluator.eval_metrics_every == 200
    assert agent_cfg.save_epoch_checkpoint_every == 1000
    assert env_cfg.control_components["contact_graph"].interval_schedule is False

    env_cfg, agent_cfg = build([
        "--curriculum", "mixture", "--hold-manifest", "m.yaml", "--uniform-fraction", "0.8",
        "--eval-every", "500", "--eval-max-steps", "2250", "--interval-schedule", "True",
        "--save-every", "500",
    ])
    ev = agent_cfg.evaluator
    assert isinstance(ev, HoldCurriculumEvaluatorConfig)
    assert ev.hold_manifest == "m.yaml" and ev.curriculum.uniform_fraction == 0.8
    assert ev.max_eval_steps == 2250 and ev.eval_metrics_every == 500
    assert agent_cfg.save_epoch_checkpoint_every == 500
    assert env_cfg.control_components["contact_graph"].interval_schedule is True
    # the observation contract (what a warm start loads into) is unchanged
    base_env, base_agent = build([])
    assert base_agent.model.actor.in_keys == agent_cfg.model.actor.in_keys
    assert set(base_env.observation_components) == set(env_cfg.observation_components)

    with pytest.raises(ValueError):
        build(["--curriculum", "mixture"])  # no manifest
