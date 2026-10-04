# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support rule v2 of the hold-curriculum evaluator (card E1, graph_growth PLAN.MD), on the CPU."""

import math
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from protomotions.agents.evaluators.config import HoldCurriculumConfig
from protomotions.agents.evaluators.hold_curriculum import (
    HoldWindow,
    ScoreParams,
    SupportV2Params,
    best_yaw_distance,
    dilate,
    score_clip,
    support_v2_holds,
    tracked_frames,
    zone_lowest_points,
    zone_vertical_load,
)
from protomotions.agents.evaluators.hold_curriculum_evaluator import HoldCurriculumEvaluator


# --------------------------------------------------------------------------- #
# A frozen copy of score_clip as it was before support rule v2 (git 46e0f30),
# so "the v1 default is bit-identical" is tested against the old code itself.
# --------------------------------------------------------------------------- #
def _score_clip_before_v2(sim_pos, ref_pos, times, holds, exemplars, goal_ids, zone_ids, params):
    sim = sim_pos.clone()
    sim[..., :2] -= (sim[0, 0, :2] - ref_pos[0, 0, :2])
    body_err = (sim - ref_pos).norm(dim=-1)
    track_fail = body_err.max(dim=-1).values > params.track_fail_m
    p_track = 1.0 - track_fail.float().mean().item()
    goal = list(goal_ids)
    hold_scores: List[float] = []
    family_scores: List[float] = []
    family_event: List[float] = []
    violations = 0
    hold_frames = 0
    for h, hold in enumerate(holds):
        sel = (times >= hold.t_hold) & (times <= hold.t_end)
        if not bool(sel.any()):
            continue
        s = sim[sel]
        r = ref_pos[sel]
        ex = exemplars[h]
        dist = best_yaw_distance(s[:, goal] - s[:, :1], ex[goal] - ex[:1])
        attained = dist < params.pose_threshold_m
        violation = torch.zeros_like(attained)
        for ids in zone_ids.values():
            ids = list(ids)
            if float(r[:, ids, 2].min()) > params.unloaded_ref_min_z:
                violation |= s[:, ids, 2].min(dim=-1).values < params.foot_down_z
        event = attained & ~dilate(violation, params.event_dilate_frames)
        attained &= ~violation
        share = attained.float().mean().item()
        hold_scores.append(share)
        if hold.family:
            family_scores.append(share)
            family_event.append(event.float().mean().item())
        violations += int(violation.sum())
        hold_frames += int(sel.sum())
    p_hold = sum(hold_scores) / len(hold_scores) if hold_scores else float("nan")
    p_family = sum(family_scores) / len(family_scores) if family_scores else float("nan")
    p_family_event = sum(family_event) / len(family_event) if family_event else float("nan")
    if hold_scores:
        score = params.track_weight * p_track + (1.0 - params.track_weight) * p_hold
    else:
        score = p_track
    return dict(p_track=p_track, p_hold=p_hold, p_family=p_family, p_family_event=p_family_event,
                support_violation=violations / hold_frames if hold_frames else float("nan"),
                track_fail_frac=1.0 - p_track, score=score, holds_scored=float(len(hold_scores)))


GOAL = [0, 1, 2, 3, 4, 5]
ZONES = {"L_FOOT": [1, 6], "R_FOOT": [2, 7], "L_HAND": [3], "R_HAND": [4]}


def _rollout(seed: int = 0, T: int = 120, B: int = 8):
    g = torch.Generator().manual_seed(seed)
    ref = torch.rand(T, B, 3, generator=g) * 0.5 + 0.3          # every zone above 15 cm -> v1 checks it
    sim = ref + 0.04 * torch.randn(T, B, 3, generator=g)
    sim[..., :2] += 3.0                                           # a constant XY offset, removed by score_clip
    sim[40:60, 2, 2] = 0.03                                       # R_FOOT origin on the floor: a v1 violation
    sim[100:104] += 0.6                                           # a tracking failure
    times = (torch.arange(T, dtype=torch.float32) + 1.0) / 30.0
    holds = [HoldWindow(0.5, 2.5, True, 0.3), HoldWindow(2.6, 3.9, False, 2.0), HoldWindow(9.0, 9.5, True)]
    exemplars = ref[[15, 80, 0]]
    return sim, ref, times, holds, exemplars


def _same(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return a.keys() == b.keys() and all(
        (math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a
    )


def test_v1_default_is_bit_identical_to_the_code_before_v2():
    for seed in range(5):
        sim, ref, times, holds, ex = _rollout(seed)
        params = ScoreParams()
        new = score_clip(sim, ref, times, holds, ex, GOAL, ZONES, params)
        old = _score_clip_before_v2(sim, ref, times, holds, ex, GOAL, ZONES, params)
        assert _same(new, old), (new, old)
        assert old["support_violation"] > 0.0        # the fixture does exercise the v1 rule


def test_tracked_frames_is_score_clips_p_track():
    sim, ref, times, holds, ex = _rollout(1)
    tracked = tracked_frames(sim, ref, 0.5)
    s = score_clip(sim, ref, times, holds, ex, GOAL, ZONES, ScoreParams())
    assert abs(tracked.float().mean().item() - s["p_track"]) < 1e-7
    assert not bool(tracked[100:104].any())


def _window(T=100, start=0, end=79, hold_at=19):
    times = (torch.arange(T, dtype=torch.float32) + 1.0) / 30.0
    hold = HoldWindow(float(times[hold_at]), float(times[end]), True, float(times[start]))
    return times, hold


def test_dilation_turns_tapping_into_support_and_the_20_percent_share():
    times, hold = _window()                                       # window = frames 0..79 (80 frames)
    T, Z = times.numel(), 4
    loaded = torch.zeros(T, Z, dtype=torch.bool)
    loaded[0:80:10, 0] = True       # A: 8 single-frame taps -> raw 10 %, dilated +-7 -> 78/80
    loaded[40, 1] = True            # B: one touch -> 15/80 = 18.75 % < 20 %
    loaded[[10, 50], 2] = True      # C: two touches -> 30/80 = 37.5 %
    loaded[[10, 50], 3] = True      # D: same as C, but not known-free
    low = torch.full((T, Z), 0.5)
    commanded = torch.zeros(1, Z, dtype=torch.bool)
    known_free = torch.tensor([[True, True, True, False]])
    tracked = torch.ones(T, dtype=torch.bool)
    rows, masks = support_v2_holds(times, loaded, low, [hold], commanded, known_free, tracked, SupportV2Params())
    r = rows[0]
    assert r["frames"] == 80
    assert r["flagged"] == [0, 2] and r["substitution"]
    assert abs(float(r["zone_share"][0]) - 78 / 80) < 1e-6
    assert abs(float(r["zone_share"][1]) - 15 / 80) < 1e-6
    assert float(r["zone_share"][3]) == 0.0                       # not known-free: never shared
    # the violation mask covers [t_hold, t_end] (frames 19..79) where A or C is (dilated) loaded
    post = (times >= hold.t_hold) & (times <= hold.t_end)
    assert masks[0].numel() == int(post.sum()) == 61
    assert bool(masks[0][:59].all())                              # frames 19..77: A's dilated taps
    assert not bool(masks[0][59:].any())                          # frames 78, 79: nothing loaded within 7


def test_share_threshold_is_inclusive_and_window_local():
    times, hold = _window(T=100, start=20, end=94, hold_at=20)   # 75 frames
    loaded = torch.zeros(100, 1, dtype=torch.bool)
    loaded[50, 0] = True                                          # 15 frames dilated -> exactly 20 %
    loaded[5, 0] = True                                           # outside the window: must not count
    rows, _ = support_v2_holds(times, loaded, torch.full((100, 1), 0.5), [hold],
                               torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool),
                               torch.ones(100, dtype=torch.bool), SupportV2Params())
    assert abs(float(rows[0]["zone_share"][0]) - 0.2) < 1e-6 and rows[0]["flagged"] == [0]


def test_geometry_alone_never_counts():
    times, hold = _window()
    T = times.numel()
    loaded = torch.zeros(T, 1, dtype=torch.bool)
    low = torch.full((T, 1), 0.005)                               # hovering within 2 cm the whole window
    rows, masks = support_v2_holds(times, loaded, low, [hold], torch.zeros(1, 1, dtype=torch.bool),
                                   torch.ones(1, 1, dtype=torch.bool), torch.ones(T, dtype=torch.bool),
                                   SupportV2Params())
    assert not rows[0]["substitution"] and not bool(masks[0].any())


def test_commanded_supports_realised():
    times, hold = _window()
    T = times.numel()
    low = torch.full((T, 2), 0.5)
    low[0:76, 0] = 0.01             # zone 0 down on 76/80 = 95 % of the window
    low[0:68, 1] = 0.01             # zone 1 down on 68/80 = 85 %
    args = (torch.zeros(T, 2, dtype=torch.bool), low, [hold])
    free = torch.zeros(1, 2, dtype=torch.bool)
    tracked = torch.ones(T, dtype=torch.bool)
    p = SupportV2Params()
    only_0 = torch.tensor([[True, False]])
    both = torch.tensor([[True, True]])
    none = torch.zeros(1, 2, dtype=torch.bool)
    assert support_v2_holds(times, *args, only_0, free, tracked, p)[0][0]["realised"] is True
    assert support_v2_holds(times, *args, both, free, tracked, p)[0][0]["realised"] is False
    assert support_v2_holds(times, *args, none, free, tracked, p)[0][0]["realised"] is None
    tracked[0:40] = False
    assert support_v2_holds(times, *args, only_0, free, tracked, p)[0][0]["tracked_share"] == 0.5


def test_empty_window_and_v2_masks_drive_score_clip():
    sim, ref, times, holds, ex = _rollout(2)
    T = times.numel()
    loaded = torch.zeros(T, 3, dtype=torch.bool)
    loaded[30:45, 1] = True                                       # inside hold 0's window [0.3, 2.5] s
    known_free = torch.ones(len(holds), 3, dtype=torch.bool)
    rows, masks = support_v2_holds(times, loaded, torch.full((T, 3), 0.5), holds,
                                   torch.zeros(len(holds), 3, dtype=torch.bool), known_free,
                                   torch.ones(T, dtype=torch.bool), SupportV2Params())
    assert rows[2] is None and masks[2] is None                   # hold 2 lies beyond the rollout
    still = ref[:1].expand_as(ref).clone()                        # a static pose: every frame is its exemplar
    still_ex = still[:3]
    clean = score_clip(still, still, times, holds, still_ex, GOAL, ZONES, ScoreParams(),
                       hold_violation=[torch.zeros_like(m) if m is not None else None for m in masks])
    hit = score_clip(still, still, times, holds, still_ex, GOAL, ZONES, ScoreParams(), hold_violation=masks)
    assert clean["p_hold"] == 1.0 and clean["support_violation"] == 0.0
    share = masks[0].float().mean().item()
    assert share > 0.0
    assert abs(hit["p_hold"] - (1.0 - share + 1.0) / 2.0) < 1e-6  # hold 0 loses its violating frames
    assert abs(hit["p_family_event"] - (1.0 - share)) < 1e-6      # not dilated a second time


def test_zone_geometry_and_load_kernels():
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    tables = SimpleNamespace(
        geom_type=torch.tensor([2, 1, 0]),                        # sphere, capsule, box
        radius=torch.tensor([0.05, 0.03, 0.0]),
        sph_center=torch.zeros(3, 3),
        cap_a=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0]]),
        cap_b=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]),
        box_center=torch.zeros(3, 3),
        box_quat=q.expand(3, 4).clone(),
        box_half=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.05, 0.02]]),
    )
    pos = torch.tensor([[[0.0, 0.0, 0.5], [1.0, 0.0, 1.0], [2.0, 0.0, 0.3]]])
    rot = q.expand(1, 3, 4).clone()
    low = zone_lowest_points(pos, rot, tables, [[0], [1, 2], [1]])
    assert torch.allclose(low, torch.tensor([[0.45, 0.28, 0.87]]), atol=1e-6)
    forces = torch.tensor([[[0.0, 0.0, 30.0], [5.0, 0.0, -10.0], [0.0, 0.0, 12.0]]])
    load = zone_vertical_load(forces, [[0], [1, 2]])
    assert torch.allclose(load, torch.tensor([[30.0, 12.0]]))     # fz clamped at zero, summed
    assert abs(SupportV2Params(body_weight_n=74.0 * 9.81).load_threshold_n - 21.7782) < 1e-3


# --------------------------------------------------------------------------- #
# Evaluator routing: v2 drives only the sampling; logged v1 keys stay v1.
# --------------------------------------------------------------------------- #
class _MotionManager:
    def __init__(self, n):
        self.motion_weights = torch.zeros(n)

    def update_sampling_weights(self, w):
        self.motion_weights[:] = w


def _stub_evaluator(rule: str, n: int = 2):
    ev = object.__new__(HoldCurriculumEvaluator)
    ev.config = SimpleNamespace(curriculum=HoldCurriculumConfig(support_rule=rule))
    ev.agent = SimpleNamespace(env=SimpleNamespace(motion_manager=_MotionManager(n)))
    ev._support_rule = rule
    ev._score_ema = None
    ev._groups = ["arm_balance", "single_leg"]
    ev._report_excluded = [False, False]
    ev._stems = ["clip_a", "clip_a_x3s"]
    ev._x0 = [True, False]
    ev._drag_tables = None
    return ev


def _scores():
    nan = float("nan")
    s = {k: torch.tensor([1.0, 0.5]) for k in ("score", "p_track", "p_hold", "p_family", "p_family_event",
                                                 "support_violation", "track_fail_frac")}
    s.update({"score_v2": torch.tensor([0.2, 0.5]), "p_hold_v2": torch.tensor([0.1, 0.5]),
              "p_family_v2": torch.tensor([0.1, nan]), "p_family_event_v2": torch.tensor([0.1, nan]),
              "support_violation_v2": torch.tensor([0.4, 0.0]), "holds_tracked_v2": torch.tensor([3.0, 4.0]),
              "supports_scored_v2": torch.tensor([3.0, 4.0]), "supports_realised_v2": torch.tensor([2.0, 4.0]),
              "substitution_holds_v2": torch.tensor([1.0, 2.0])})
    return s


@pytest.mark.parametrize("rule,expected", [("v1", [1.0, 0.5]), ("v2", [0.2, 0.5])])
def test_curriculum_runs_on_the_selected_rule(rule, expected):
    ev = _stub_evaluator(rule)
    probs = ev._update_curriculum(_scores())
    assert torch.allclose(ev._score_ema, torch.tensor(expected))
    assert torch.allclose(ev.env.motion_manager.motion_weights, probs)


def test_logged_v1_keys_ignore_the_rule_and_v2_keys_are_new():
    for rule in ("v1", "v2"):
        ev = _stub_evaluator(rule)
        scores = _scores()
        probs = ev._update_curriculum(scores)
        logs = ev._curriculum_logs(scores, probs)
        assert logs["eval/perf/score"] == pytest.approx(0.75)                     # v1, whatever the rule
        assert logs["eval/perf_group/arm_balance_score"] == pytest.approx(1.0)
        assert logs["eval/perf_v2/score"] == pytest.approx(0.35)
        assert logs["eval/perf_group_v2/arm_balance_score"] == pytest.approx(0.2)
        assert logs["eval/perf/support_realised_v2"] == pytest.approx(6 / 7)
        assert logs["eval/perf/support_realised_v2_x0"] == pytest.approx(2 / 3)
        assert logs["eval/perf/substitution_holds_v2"] == 3.0
        assert logs["eval/perf/substitution_holds_v2_x0"] == 1.0
    no_v2 = {k: v for k, v in _scores().items() if not k.endswith("_v2")}
    ev = _stub_evaluator("v1")
    logs = ev._curriculum_logs(no_v2, ev._update_curriculum(no_v2))
    assert not any("_v2" in k for k in logs)                                       # absent, not NaN


def test_old_pickled_configs_read_the_v1_defaults():
    cfg = HoldCurriculumConfig()
    for name in ("support_rule", "support_v2_load_frac_bw", "support_v2_min_share", "support_v2_down_m",
                 "support_v2_realised_share", "support_v2_tracked_share"):
        del cfg.__dict__[name]                                    # a config frozen before the fields existed
    assert cfg.support_rule == "v1" and cfg.support_v2_load_frac_bw == 0.03


def test_predicted_library_saves_sim_ground_forces_under_its_own_key(tmp_path):
    from protomotions.tests.test_mimic_evaluator_helpers import _evaluator, _packed_metric

    evaluator = _evaluator(tmp_path)
    motion_lens = torch.tensor([2, 1, 0])
    metrics = {k: _packed_metric(motion_lens, features=f) for k, f in (
        ("dof_pos", 2), ("dof_vel", 2), ("rigid_body_pos", 3), ("rigid_body_rot", 4),
        ("rigid_body_vel", 3), ("rigid_body_ang_vel", 3), ("rigid_body_contacts", 1))}
    evaluator.motion_manager.fixed = (torch.tensor([1]), torch.tensor([0]))
    unrecorded = _packed_metric(motion_lens, features=3)
    unrecorded.frame_counts[:] = 0                                # the simulator reported no ground forces
    metrics["rigid_body_ground_forces"] = unrecorded
    evaluator._save_predicted_motion_lib(metrics, epoch=1)
    saved = torch.load(tmp_path / "results" / "predicted_motion_lib_epoch_1.pt")
    assert "sim_rigid_body_ground_forces" not in saved and "gnf" not in saved

    metrics["rigid_body_ground_forces"] = _packed_metric(motion_lens, features=3)
    evaluator._save_predicted_motion_lib(metrics, epoch=2)
    saved = torch.load(tmp_path / "results" / "predicted_motion_lib_epoch_2.pt")
    forces = saved["sim_rigid_body_ground_forces"]
    assert forces.shape == saved["gts"].shape == (3, 1, 3)
    assert torch.equal(forces[:, 0], torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]))
    assert "gnf" not in saved                                     # never passed off as measured force
