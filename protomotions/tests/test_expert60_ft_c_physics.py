# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune C physics terms (``expert_revist/ft_c/README.MD``): kernels, tables, control, wiring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protomotions.agents.evaluators.hold_curriculum import HoldWindow, ScoreParams, dilate, score_clip
from protomotions.envs.control.physics_terms import (
    PhysicsTables,
    clip_drag,
    corner_slip_speed,
    lean_shortfall,
    patch_points,
    polygon_margin,
    swing_charged_load,
    whole_body_com,
)
from protomotions.tests.test_contact_graph import SIM_BODY_NAMES, _current_state, _make_control, _toy_graph_payload

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
MG = 74.0 * 9.81


def _numpy_margin(p, pts):
    """Reference signed margin: + inside, distance to the hull boundary."""
    from repair_armbalance_holds import hull, signed_margin

    return signed_margin(np.asarray(p, float), hull(np.asarray(pts, float)))


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def test_polygon_margin_matches_a_reference_hull_inside_and_outside():
    gen = torch.Generator().manual_seed(0)
    E, N = 64, 12
    pts = torch.rand(E, N, 2, generator=gen)
    valid = torch.rand(E, N, generator=gen) > 0.25
    valid[:, :3] = True
    p = torch.rand(E, 2, generator=gen) * 1.6 - 0.3
    got = polygon_margin(pts, valid, p)
    for e in range(E):
        ref = _numpy_margin(p[e].numpy(), pts[e][valid[e]].numpy())
        if ref >= 0:     # inside: exact
            assert got[e].item() == pytest.approx(ref, abs=1e-5)
        else:            # outside: negative, and never more negative than the true distance
            assert got[e].item() < 0 and got[e].item() >= ref - 1e-5


def test_polygon_margin_degenerate_rows():
    pts = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]])
    valid = torch.tensor([[True, True, False]])      # one distinct point
    m = polygon_margin(pts, valid, torch.tensor([[3.0, 4.0]]))
    assert m.item() == pytest.approx(-5.0)


def _tables(tmp_path, body_names=None):
    """A physics-tables payload for the toy graph + stub env of ``test_contact_graph``."""
    from build_physics_tables import body_constants

    body_names = list(body_names or SIM_BODY_NAMES)
    g = _toy_graph_payload()
    Z, M, S, P = len(g["zone_order"]), 2, g["seg_node"].shape[1], len(g["pair_names"])
    swing = torch.zeros(M, 700, Z, dtype=torch.bool)
    swing[1, 250:300, 0] = True          # motion 1 swings L_FOOT at 4.2-5.0 s (after its only hold)
    seg_lean_gate = torch.zeros(M, S, dtype=torch.bool)
    seg_lean_gate[0, 2] = True           # motion 0's "handstand" (L_HAND:G) hold
    payload = dict(
        motion_names=["clip_a", "clip_b"], fps=60, zone_order=g["zone_order"], zone_bodies=g["zone_bodies"],
        swing=swing, swing_len=torch.tensor([600, 600]),
        seg_cop_rel=torch.zeros(M, S, 2), seg_cop_valid=torch.zeros(M, S, dtype=torch.bool),
        seg_com_rel=torch.zeros(M, S, 2), seg_zone_share=torch.zeros(M, S, Z),
        seg_share_valid=torch.zeros(M, S, dtype=torch.bool), seg_lean_gate=seg_lean_gate,
        seg_pair_consequential=torch.zeros(M, S, P, dtype=torch.bool),
        **body_constants(body_names),
    )
    path = tmp_path / "physics_tables.pt"
    torch.save(payload, path)
    return path


def test_tables_validate_the_library_and_look_up_frames(tmp_path):
    path = _tables(tmp_path)
    with pytest.raises(ValueError, match="different motion library"):
        PhysicsTables(path, ["x", "y"], SIM_BODY_NAMES, "cpu")
    with pytest.raises(ValueError, match="body order"):
        PhysicsTables(path, ["clip_a", "clip_b"], list(reversed(SIM_BODY_NAMES)), "cpu")
    t = PhysicsTables(path, ["clip_a", "clip_b"], SIM_BODY_NAMES, "cpu")
    got = t.swing_at(torch.tensor([1, 1, 1, 0]), torch.tensor([4.0, 4.5, 99.0, 4.5]))
    assert got[:, 0].tolist() == [False, True, False, False]   # 99 s clamps to the last frame
    assert float(t.body_mass.sum()) == pytest.approx(74.0, abs=0.05)


def test_com_and_patch_points_match_mujoco_and_the_offline_builder(tmp_path):
    """Runtime kernels on a real reference pose, in the MJCF body order."""
    import static_hold_lp as S
    from build_physics_tables import body_constants, patch_points as offline_points

    names = S.BODY
    t = PhysicsTables(_tables(tmp_path, names), ["clip_a", "clip_b"], names, "cpu")
    pos, rot = S.load_frame(REPO / "data/smpl/yoga_motions_proto_yogi_expert60_ftC/220923_Crane_Crow_Pose_or_Bakasana_-a.motion", 9.5)
    S.set_pose(pos, rot)
    pos_t = torch.tensor(pos, dtype=torch.float32)[None]
    rot_t = torch.tensor(rot, dtype=torch.float32)[None]
    com = whole_body_com(pos_t, rot_t, t.body_mass, t.body_com_local)
    assert np.allclose(com[0].numpy(), S.D.subtree_com[1], atol=2e-3)   # geom-centre COM model
    bodies = [names.index(b) for b in ("L_Wrist", "L_Hand", "L_Elbow", "Head")]
    ours = patch_points(pos_t, rot_t, t, bodies)
    theirs, _ = offline_points(pos_t, rot_t, body_constants(names), bodies)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_corner_slip_reads_a_translation_and_ignores_a_pivot(tmp_path):
    t = PhysicsTables(_tables(tmp_path), ["clip_a", "clip_b"], SIM_BODY_NAMES, "cpu")
    B = len(SIM_BODY_NAMES)
    ank = SIM_BODY_NAMES.index("L_Ankle")
    pos = torch.zeros(2, B, 3)
    rot = torch.zeros(2, B, 4)
    rot[..., 3] = 1.0
    vel = torch.zeros(2, B, 3)
    ang = torch.zeros(2, B, 3)
    vel[0, ank] = torch.tensor([0.8, 0.0, 0.0])                   # env 0: dragged at 0.8 m/s
    # env 1: yawing about one bottom corner of the box, COM velocity consistent with that pivot
    local = t.box_center[ank] + t.box_half[ank] * torch.tensor([-1.0, -1.0, -1.0])
    w = torch.tensor([0.0, 0.0, 2.0])
    ang[1, ank] = w
    vel[1, ank] = torch.linalg.cross(w, t.body_com_local[ank] - local)
    s = corner_slip_speed(pos, rot, vel, ang, t, [ank])
    assert s[0].item() == pytest.approx(0.8, abs=1e-5)
    assert s[1].item() == pytest.approx(0.0, abs=1e-5)


def test_clip_drag_counts_loaded_slides_only(tmp_path):
    """drag_corpus.py's definition: load > 50 N and slowest-corner slip > 0.1 m/s."""
    t = PhysicsTables(_tables(tmp_path), ["clip_a", "clip_b"], SIM_BODY_NAMES, "cpu")
    B, T, dt = len(SIM_BODY_NAMES), 30, 1.0 / 30.0
    ank = SIM_BODY_NAMES.index("L_Ankle")
    pos, vel, ang, gf = (torch.zeros(T, B, 3) for _ in range(4))
    rot = torch.zeros(T, B, 4)
    rot[..., 3] = 1.0
    vel[:10, ank, 0] = 0.8
    gf[:10, ank, 2] = 200.0                     # frames 0-9: dragged under load -> counted
    local = t.box_center[ank] + t.box_half[ank] * torch.tensor([-1.0, -1.0, -1.0])
    w = torch.tensor([0.0, 0.0, 2.0])
    ang[10:20, ank] = w
    vel[10:20, ank] = torch.linalg.cross(w, t.body_com_local[ank] - local)
    gf[10:20, ank, 2] = 300.0                   # frames 10-19: pivoting under load -> not drag
    vel[20:, ank, 0] = 0.8
    gf[20:, ank, 2] = 20.0                      # frames 20-29: sliding a barely-loaded foot -> not drag
    swing = torch.zeros(T, 1, dtype=torch.bool)
    swing[5:10] = True
    d = clip_drag(pos, rot, vel, ang, gf, t, [[ank]], swing, dt)
    assert d["drag_s"] == pytest.approx(10 * dt)
    assert d["drag_J"] == pytest.approx(0.75 * 200.0 * 0.8 * 10 * dt, rel=1e-5)
    assert d["drag_J_swing"] == pytest.approx(d["drag_J"] / 2, rel=1e-5)
    assert clip_drag(pos, rot, vel, ang, gf, t, [[ank]], None, dt)["drag_J_swing"] == 0.0


def test_evaluator_drag_logs_pool_the_gate_set_without_excluded_motions():
    from protomotions.agents.evaluators.hold_curriculum_evaluator import HoldCurriculumEvaluator as H

    stub = SimpleNamespace(
        _drag_tables=object(), _nanmean=H._nanmean,
        _stems=["Warrior_II_-a", "Warrior_II_-a_x3s", "Plow_-b", "Scale_-a", "Tree_-a"],
        _groups=["connective", "connective", "inversion", "arm_balance", "single_leg"],
        _report_excluded=[False, False, False, True, False],
        _drag_patterns=["Warrior_II_-a", "Plow_-b", "Scale_-a"],
    )
    J = torch.tensor([10.0, 30.0, 200.0, 999.0, float("nan")])
    logs = H._drag_logs(stub, {"drag_J": J, "drag_J_swing": J / 2, "drag_s": J / 100})
    assert logs["eval/drag/Warrior_II_-a_J"] == pytest.approx(20.0)
    assert logs["eval/drag/top_J"] == pytest.approx(80.0)          # Scale excluded
    assert "eval/drag/Scale_-a_J" not in logs
    assert logs["eval/drag/all_J"] == pytest.approx(80.0)          # NaN (unscored) skipped
    assert logs["eval/drag_group/inversion_swing_J"] == pytest.approx(100.0)
    stub._drag_tables = None
    assert H._drag_logs(stub, {}) == {}


def test_swing_load_and_lean_shortfall_are_gated_and_bounded():
    forces = torch.zeros(2, 3, 3)
    forces[:, 0, 2] = 50.0
    forces[:, 1, 2] = -5.0                                         # noise, not load
    zm = torch.eye(3)
    swing = torch.tensor([[True, True, False], [True, False, False]])
    out = swing_charged_load(forces, swing, zm, torch.tensor([True, False]))
    assert out.tolist() == [50.0, 0.0]
    m = torch.tensor([-0.05, 0.01, 0.2])
    p = lean_shortfall(m, torch.tensor([True, True, True]), 0.03, 0.10)
    assert p.tolist() == pytest.approx([0.8, 0.2, 0.0])
    assert lean_shortfall(m, torch.tensor([False] * 3), 0.03, 0.10).abs().sum() == 0


# --------------------------------------------------------------------------- #
# Control
# --------------------------------------------------------------------------- #
def _ctx(num_envs, foot_n=0.0):
    cur = _current_state(num_envs=num_envs)
    forces = torch.zeros(num_envs, len(SIM_BODY_NAMES), 3)
    forces[:, SIM_BODY_NAMES.index("L_Toe"), 2] = foot_n
    cur.rigid_body_ground_forces = forces
    cur.rigid_body_vel = torch.zeros(num_envs, len(SIM_BODY_NAMES), 3)
    cur.rigid_body_ang_vel = torch.zeros(num_envs, len(SIM_BODY_NAMES), 3)
    cur.rigid_body_pair_contact_forces = None
    return SimpleNamespace(current=cur, mimic=SimpleNamespace(ref_state=None))


def test_control_without_tables_leaves_every_physics_field_none(tmp_path, monkeypatch):
    control = _make_control(tmp_path, monkeypatch, include_current_segment=True, interval_schedule=True)
    control.reset(torch.arange(4))
    out = control._physics_terms(_ctx(4))
    assert all(v is None for v in out.values())


def test_control_prices_the_swing_load_once_per_step_outside_holds(tmp_path, monkeypatch):
    path = _tables(tmp_path)
    control = _make_control(tmp_path, monkeypatch, include_current_segment=True, interval_schedule=True,
                            physics_tables_file=str(path), swing_ema_tau_s=0.1)
    mm = control.env.motion_manager
    mm.motion_ids[:] = torch.tensor([1, 1, 0, 0])
    mm.motion_times[:] = torch.tensor([4.5, 3.0, 4.5, 7.0])       # env0 swing, env1 inside its hold
    control.reset(torch.arange(4))
    control.step()
    out = control._physics_terms(_ctx(4, foot_n=50.0))
    assert out["swing_gate"].tolist() == [1.0, 0.0, 0.0, 0.0]     # motion 0 is holds end to end
    assert out["swing_load_n"].tolist() == [50.0, 0.0, 0.0, 0.0]
    a = control._swing_ema.alpha
    assert out["swing_penalty"][0].item() == pytest.approx(a * 50.0 / (0.1 * MG))
    again = control._physics_terms(_ctx(4, foot_n=50.0))           # rebuild in the same step
    assert torch.equal(out["swing_penalty"], again["swing_penalty"])
    control.reset(torch.tensor([0]))
    assert control._physics_terms(_ctx(4, foot_n=50.0))["swing_penalty"][0].item() == 0.0
    # the lean term is armed only in motion 0's lean-gated hold (env 3 at 7 s)
    assert out["lean_gate"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert 0.0 <= out["lean_penalty"][3].item() <= 1.0 and out["lean_penalty"][:3].abs().sum() == 0
    for v in out.values():
        assert torch.isfinite(v).all()


def test_control_excludes_motions_and_is_off_under_a_manual_goal(tmp_path, monkeypatch):
    path = _tables(tmp_path)
    control = _make_control(tmp_path, monkeypatch, include_current_segment=True, interval_schedule=True,
                            physics_tables_file=str(path), physics_exclude_motions=["clip_b"])
    mm = control.env.motion_manager
    mm.motion_ids[:] = 1
    mm.motion_times[:] = 4.5
    control.reset(torch.arange(4))
    control.step()
    assert control._physics_terms(_ctx(4, foot_n=50.0))["swing_gate"].abs().sum() == 0
    with pytest.raises(ValueError, match="match no motion"):
        _make_control(tmp_path, monkeypatch, include_current_segment=True, interval_schedule=True,
                      physics_tables_file=str(path), physics_exclude_motions=["nope"])


# --------------------------------------------------------------------------- #
# Evaluator: event-aware family holds
# --------------------------------------------------------------------------- #
def test_dilate_and_the_event_aware_family_score_discount_tapping():
    assert dilate(torch.tensor([False, False, True, False, False]), 1).tolist() == [False, True, True, True, False]
    T, B = 120, 24
    times = torch.arange(T, dtype=torch.float32) / 30.0
    ref = torch.zeros(T, B, 3)
    ref[:, 3, 2] = 0.4                                  # L_Ankle kept up by the reference
    sim = ref.clone()
    sim[::10, 3, 2] = 0.0                               # the policy's foot touches down 3x a second
    holds = [HoldWindow(0.0, 4.0, family=True)]
    zone_ids = {"L_FOOT": [3]}
    per_frame = score_clip(sim, ref, times, holds, ref[:1], [0, 3], zone_ids, ScoreParams(event_dilate_frames=0))
    event = score_clip(sim, ref, times, holds, ref[:1], [0, 3], zone_ids, ScoreParams(event_dilate_frames=7))
    assert per_frame["p_family"] == pytest.approx(0.9, abs=0.01)   # credits the frames between strikes
    # never lifted for 0.23 s: only the last 2 frames sit beyond the final touch's dilation
    assert event["p_family_event"] == pytest.approx(2 / 120)
    assert event["p_family"] == per_frame["p_family"]              # the scored number is unchanged


# --------------------------------------------------------------------------- #
# Experiment wiring
# --------------------------------------------------------------------------- #
def test_goal_conditioned_experiment_wires_the_physics_terms(tmp_path):
    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.robot_configs.factory import robot_config
    from protomotions.tests.test_expert60_ft_a import _graph_with_near_start_hold

    _graph_with_near_start_hold(tmp_path)
    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)

    def build(argv):
        args = parser.parse_args(["--hold-graph-file", str(tmp_path / "g.pt")] + argv)
        args.motion_file, args.scenes_file, args.batch_size, args.training_max_steps = "unused.pt", None, 32, 256
        robot_cfg = robot_config("smpl_yogi")
        exp.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
        env_cfg = exp.env_config(robot_cfg, args)
        return env_cfg, exp.agent_config(robot_cfg, env_cfg, args)

    base_env, base_agent = build([])
    assert not any(k.startswith(("swing", "lean", "diag_swing", "diag_lean", "diag_slip", "diag_pair"))
                   for k in base_env.reward_components)
    assert base_env.control_components["contact_graph"].physics_tables_file == ""
    with pytest.raises(ValueError, match="need --physics-tables"):
        build(["--swing-penalty-weight", "-0.3"])
    env_cfg, agent_cfg = build([
        "--physics-tables", "t.pt", "--swing-penalty-weight", "-0.3", "--lean-penalty-weight", "-0.3",
        "--physics-exclude-motions", "Cockerel_Pose-b", "--report-exclude-motions", "Scale_Pose",
        "--drag-report-motions", "Warrior_II", "Plow", "--curriculum", "mixture", "--hold-manifest", "h.yaml",
    ])
    cur = agent_cfg.evaluator.curriculum
    assert cur.drag_report_motions == ["Warrior_II", "Plow"] and cur.report_exclude_motions == ["Scale_Pose"]
    rew = env_cfg.reward_components
    assert rew["swing_penalty_rew"].get_params()["weight"] == -0.3
    assert rew["lean_penalty_rew"].get_params()["zero_during_grace_period"] is True
    for k in ("diag_swing_load_n", "diag_lean_margin_m", "diag_lean_error_m", "diag_slip_power_w", "diag_pair_load_n"):
        assert rew[k].get_params()["weight"] == 0.0
    ctrl = env_cfg.control_components["contact_graph"]
    assert ctrl.physics_tables_file == "t.pt" and ctrl.physics_exclude_motions == ["Cockerel_Pose-b"]
    # the observation contract a warm start loads into is untouched
    assert agent_cfg.model.actor.in_keys == base_agent.model.actor.in_keys
    assert set(env_cfg.observation_components) == set(base_env.observation_components)
    with pytest.raises(ValueError, match="penalty"):
        build(["--physics-tables", "t.pt", "--lean-penalty-weight", "0.3"])
