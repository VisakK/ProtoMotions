# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Card S1 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: the student port to the release-v3 expert, and the
three fixes S0 found first.

* ``BaseEnv.rebuild_observations`` rebuilds against the contact sample the last build used;
* ``set_manual_goal`` takes a separate dwell duration and a deadline floor, and the panel's ``timing='training'``
  serves a plan goal as the segment a scheduled slot carries;
* route plans mirror a clip's own holds (``make_route_probe_plans.py``);
* the expert view: ``ContactGraphControl`` publishes the expert's own window, and the copied components are rewired
  onto it; the merge loop skips a critic-only future window.

Toy graphs on the stub envs of ``test_contact_graph`` / ``test_release_v2_runtime`` / ``test_base_env_helpers``. The
GPU parity tests are ``data/scripts/s1_port_parity.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.sequence_viz import VizGoal, VizSequence, fill_goal_slots
from protomotions.agents.supervised import expert_port
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.tests.test_base_env_helpers import _make_env
from protomotions.tests.test_contact_graph import SIM_BODY_NAMES, _current_state, _stub_env
from protomotions.tests.test_release_v2_runtime import _LibWithLayout, _graph_v2
from protomotions.utils.release_identity import file_sha256

SCHEDULE = dict(include_current_segment=True, interval_schedule=True, dwell_channels=True)


# --------------------------------------------------------------------------- #
# BaseEnv.rebuild_observations
# --------------------------------------------------------------------------- #
def _capturing_env():
    env = _make_env(motions=0, history=False)
    env._obs_reset_rows = torch.zeros(env.num_envs, dtype=torch.bool)
    env._obs_contact_sample = None
    seen = []

    def capture(env_ids=None, context=None):
        seen.append(dict(
            previous=context.previous_contact_forces.clone(),
            valid=context.contact_temporal_valid.clone(),
            prev_mag=context.prev_contact_force_magnitudes.clone(),
            forces=context.current.rigid_body_contact_forces.clone(),
        ))

    env.compute_observations = capture
    env.compute_reward = lambda context: None
    env.check_resets_and_terminations = lambda context: (
        torch.zeros(env.num_envs, dtype=torch.bool), torch.zeros(env.num_envs, dtype=torch.bool))
    return env, seen


def test_rebuild_after_a_step_reads_the_pre_step_contact_sample():
    env, seen = _capturing_env()
    env.previous_contact_forces[:] = 7.0
    env.prev_contact_force_magnitudes[:] = 3.0
    env.contact_temporal_valid[:] = True
    BaseEnv.post_physics_step(env)
    stepped = seen[-1]
    # _finalize_contact_state has promoted this step's force: a naive rebuild would read a zero force rate.
    assert torch.equal(env.previous_contact_forces, env.simulator.state.rigid_body_contact_forces)
    BaseEnv.rebuild_observations(env)
    rebuilt = seen[-1]
    for key in stepped:
        assert torch.equal(rebuilt[key], stepped[key]), key
    # ... and the env's own buffers are as the step left them.
    assert torch.equal(env.previous_contact_forces, env.simulator.state.rigid_body_contact_forces)
    assert env.contact_temporal_valid.all()


def test_rebuild_after_a_reset_clears_the_reset_rows_and_keeps_the_others():
    env, seen = _capturing_env()
    env.previous_contact_forces[:] = 7.0
    env.contact_temporal_valid[:] = True
    BaseEnv.post_physics_step(env)
    stepped = seen[-1]
    BaseEnv.reset(env, env_ids=torch.tensor([1]))
    reset_build = seen[-1]
    assert not reset_build["forces"][1].any()           # the reset's build used a cleared sample
    BaseEnv.rebuild_observations(env)
    rebuilt = seen[-1]
    # Row 1 is rebuilt as the reset built it; rows 0 and 2 as the step did.
    assert not rebuilt["forces"][1].any() and not rebuilt["previous"][1].any() and not bool(rebuilt["valid"][1])
    for row in (0, 2):
        for key in stepped:
            assert torch.equal(rebuilt[key][row], stepped[key][row]), (row, key)
    # The live sensors still carry the pre-teleport force; the rebuild did not write them.
    assert env.simulator.state.rigid_body_contact_forces[1].abs().sum() > 0


# --------------------------------------------------------------------------- #
# The expert view
# --------------------------------------------------------------------------- #
class _TimedLib(_LibWithLayout):
    """Reference poses that depend on (motion, time), so a slot mix-up shows."""

    def get_motion_state(self, motion_ids, motion_times):
        n, b = len(motion_ids), len(SIM_BODY_NAMES)
        pos = (motion_times.view(n, 1, 1) + 10.0 * motion_ids.view(n, 1, 1).float()
               + 0.01 * torch.arange(b).view(1, b, 1)).expand(n, b, 3).clone()
        rot = torch.zeros(n, b, 4)
        rot[..., 3] = 1.0
        rot[..., 0] = 0.001 * motion_times.view(n, 1)
        return SimpleNamespace(rigid_body_pos=pos, rigid_body_rot=rot)


def _graph_file(tmp_path):
    path = tmp_path / "contact_graph.pt"
    if not path.exists():
        torch.save(_graph_v2(), path)
    return path


def _env(subset=None):
    env = _stub_env()
    env.motion_lib = _TimedLib(env.motion_lib)
    if subset is not None:
        env.robot_config.trackable_bodies_subset = list(subset)
    return env


def _contract(graph_path, **edit):
    contract = dict(num_goal_steps=2, full_visibility=True, min_lead_s=0.2, history_time_clip_s=10.0,
                    far_goal_prob=0.0, num_history_events=0, graph_file=str(graph_path),
                    graph_sha256=file_sha256(graph_path), pair_names=list(_graph_v2()["pair_names"]), **SCHEDULE)
    contract.update(edit)
    return contract


def _controls(tmp_path, monkeypatch, student=None, contract=None):
    """A student control (3 slots, 3 bodies, masked, expert view on) and the expert's own (2 slots, every body)."""
    from protomotions.envs.control import contact_graph_control as module
    from protomotions.envs.control.mimic_control import MimicControl

    monkeypatch.setattr(MimicControl, "populate_context", lambda self, ctx: None)
    torch.manual_seed(0)
    graph_path = _graph_file(tmp_path)
    student_kwargs = dict(graph_file=str(graph_path), num_goal_steps=3, num_masked_future_steps=3,
                          pose_visible_prob=0.5, contact_visible_prob=0.5, full_pose_prob=0.5,
                          expert_view_steps=2, expert_view_num_bodies=len(SIM_BODY_NAMES),
                          expert_view_contract=contract if contract is not None else _contract(graph_path),
                          **SCHEDULE)
    student_kwargs.update(student or {})
    s = module.ContactGraphControl(module.ContactGraphControlConfig(**student_kwargs), _env(SIM_BODY_NAMES[:3]))
    e = module.ContactGraphControl(module.ContactGraphControlConfig(
        graph_file=str(graph_path), num_goal_steps=2, num_masked_future_steps=2, pose_visible_prob=1.0,
        contact_visible_prob=1.0, full_pose_prob=1.0, **SCHEDULE), _env())
    return s, e


def _ctx(control):
    ctx = EnvContext(current=_current_state(), noisy=None, dt=control.env.dt)
    control.populate_context(ctx)
    return ctx


def _same_view(s_ctx, e_ctx):
    a, b = s_ctx.expert_masked_mimic, e_ctx.masked_mimic
    for key in ("ref_pos", "ref_rot", "target_times", "time_offsets", "target_poses_masks", "target_bodies_masks"):
        assert torch.equal(getattr(a, key), getattr(b, key)), key
    a, b = s_ctx.expert_contact_goal, e_ctx.contact_goal
    for key in ("contact_spec", "orient_spec", "visible", "time_offsets", "dwell_features", "node_ids"):
        assert torch.equal(getattr(a, key), getattr(b, key)), key


def test_expert_view_is_the_expert_controls_own_window(tmp_path, monkeypatch):
    s, e = _controls(tmp_path, monkeypatch)
    for motion in (0, 1):
        for step in range(0, 181, 3):
            for c in (s, e):
                c.env.motion_manager.motion_ids[:] = motion
                c.env.motion_manager.motion_times[:] = step / 30.0 + torch.tensor([0.0, 0.013, 0.021, 0.04])
                c.reset(torch.arange(4)) if step == 0 else c.step()
            _same_view(_ctx(s), _ctx(e))
    # The student's own view is masked and has more slots; the expert view is not.
    assert _ctx(s).masked_mimic.ref_pos.shape[1] == 3


def test_expert_view_passes_a_manual_goal_through(tmp_path, monkeypatch):
    s, e = _controls(tmp_path, monkeypatch)
    for c in (s, e):
        c.reset(torch.arange(4))
    n = 4
    node = torch.tensor([[1, 0, 0]]).repeat(n, 1)
    args = dict(pose_motion_ids=torch.zeros(n, 3, dtype=torch.long), pose_times=torch.tensor([[1.0, 3.0, 3.5]]).repeat(n, 1),
                time_offsets=torch.tensor([[0.5, 2.0, 3.0]]).repeat(n, 1),
                hold_seconds=torch.tensor([[1.5, 4.0, 5.0]]).repeat(n, 1),
                hold_duration=torch.tensor([[1.0, 2.0, 2.0]]).repeat(n, 1))
    visible = torch.tensor([[True, False, True]]).repeat(n, 1)     # the student's masks do not reach the expert
    s.set_manual_goal(node_ids=node, pose_visible=visible, contact_visible=~visible, deadline_floor=0.0, **args)
    e.set_manual_goal(node_ids=node[:, :2], pose_visible=torch.ones(n, 2, dtype=torch.bool),
                      contact_visible=torch.ones(n, 2, dtype=torch.bool), deadline_floor=0.0,
                      **{k: v[:, :2] for k, v in args.items()})
    for _ in range(25):
        _same_view(_ctx(s), _ctx(e))
        s.step()
        e.step()


def test_expert_view_off_publishes_nothing(tmp_path, monkeypatch):
    s, _ = _controls(tmp_path, monkeypatch, student=dict(expert_view_steps=0))
    s.reset(torch.arange(4))
    ctx = _ctx(s)
    assert ctx.expert_masked_mimic is None and ctx.expert_contact_goal is None


@pytest.mark.parametrize("student, contract, match", [
    (dict(include_current_segment=False, interval_schedule=False), None, "include_current_segment"),
    (dict(dwell_channels=False), None, "dwell_channels"),
    (dict(min_lead_s=0.3), None, "min_lead_s"),
    (dict(history_time_clip_s=5.0), None, "history_time_clip_s"),
    (dict(expert_view_steps=4), None, "num_goal_steps"),
    ({}, dict(full_visibility=False), "visible"),
    ({}, dict(num_goal_steps=3), "3 goal slots"),
    ({}, dict(pair_names=["X:G"]), "pair vocabulary"),
    ({}, dict(graph_sha256="0" * 64), "graph sha256"),
    (dict(expert_view_contract={}), None, "contract is empty"),
])
def test_expert_view_refuses_a_contract_it_does_not_meet(tmp_path, monkeypatch, student, contract, match):
    graph_path = _graph_file(tmp_path)
    with pytest.raises(ValueError, match=match):
        _controls(tmp_path, monkeypatch, student=student,
                  contract=None if contract is None else _contract(graph_path, **contract))


def test_expert_view_checks_the_env_settings_the_expert_reads(tmp_path, monkeypatch):
    from protomotions.envs.control import contact_graph_control as module
    from protomotions.envs.control.mimic_control import MimicControl

    monkeypatch.setattr(MimicControl, "populate_context", lambda self, ctx: None)
    graph_path = _graph_file(tmp_path)
    contract = _contract(graph_path, ref_respawn_offset=0.005, contact_force_on_threshold_n=5.0,
                         contact_force_off_threshold_n=2.0, ref_contact_smooth_window=7, num_state_history_steps=20,
                         realign_motion_with_humanoid_on_each_step=False)
    good = dict(ref_respawn_offset=0.005, contact_force_on_threshold_n=5.0, contact_force_off_threshold_n=2.0,
                ref_contact_smooth_window=7, num_state_history_steps=60,
                motion_manager=SimpleNamespace(realign_motion_with_humanoid_on_each_step=False))
    cfg = lambda: module.ContactGraphControlConfig(  # noqa: E731
        graph_file=str(graph_path), num_goal_steps=3, num_masked_future_steps=3, expert_view_steps=2,
        expert_view_num_bodies=len(SIM_BODY_NAMES), expert_view_contract=contract, **SCHEDULE)
    for edit, match in ((dict(ref_respawn_offset=0.05), "ref_respawn_offset"),
                        (dict(num_state_history_steps=15), "num_state_history_steps"),
                        (dict(ref_contact_smooth_window=0), "ref_contact_smooth_window"),
                        (dict(motion_manager=SimpleNamespace(realign_motion_with_humanoid_on_each_step=True)),
                         "realign")):
        env = _env(SIM_BODY_NAMES[:3])
        env.config = SimpleNamespace(**{**good, **edit})
        with pytest.raises(ValueError, match=match):
            module.ContactGraphControl(cfg(), env)
    env = _env(SIM_BODY_NAMES[:3])
    env.config = SimpleNamespace(**good)
    module.ContactGraphControl(cfg(), env)


# --------------------------------------------------------------------------- #
# Manual goals: the dwell duration and the deadline floor
# --------------------------------------------------------------------------- #
def _single(tmp_path, monkeypatch, **overrides):
    from protomotions.envs.control import contact_graph_control as module
    from protomotions.envs.control.mimic_control import MimicControl

    monkeypatch.setattr(MimicControl, "populate_context", lambda self, ctx: None)
    kwargs = dict(graph_file=str(_graph_file(tmp_path)), num_goal_steps=2, num_masked_future_steps=2,
                  pose_visible_prob=1.0, contact_visible_prob=1.0, full_pose_prob=1.0, **SCHEDULE)
    kwargs.update(overrides)
    control = module.ContactGraphControl(module.ContactGraphControlConfig(**kwargs), _env())
    control.reset(torch.arange(4))
    return control


def _manual(control, **extra):
    n = control.env.num_envs
    ones = torch.ones(n, 2, dtype=torch.bool)
    control.set_manual_goal(node_ids=torch.tensor([[1, 0]]).repeat(n, 1),
                            pose_motion_ids=torch.zeros(n, 2, dtype=torch.long),
                            pose_times=torch.tensor([[1.0, 3.0]]).repeat(n, 1),
                            time_offsets=torch.tensor([[0.1, 2.0]]).repeat(n, 1),
                            pose_visible=ones, contact_visible=ones, **extra)


def test_manual_goal_duration_is_its_own_channel_and_stays_put(tmp_path, monkeypatch):
    control = _single(tmp_path, monkeypatch)
    _manual(control, hold_seconds=torch.full((4, 2), 6.0), hold_duration=torch.tensor([[2.0, 3.0]]).repeat(4, 1),
            deadline_floor=0.0)
    first = control._dwell_features.clone()
    assert torch.allclose(first[0], torch.tensor([[0.2, 0.6], [0.3, 0.6]]))
    for _ in range(30):
        control.step()
    after = control._dwell_features
    assert torch.equal(after[..., 0], first[..., 0])                       # duration: constant
    assert torch.allclose(after[..., 1], first[..., 1] - 1.0 / 10.0, atol=1e-5)   # remaining: 1 s less
    assert torch.allclose(control._time_offsets[:, 0], torch.zeros(4))     # the deadline reached 0 and stopped


def test_manual_goal_legacy_call_is_unchanged(tmp_path, monkeypatch):
    control = _single(tmp_path, monkeypatch)
    _manual(control, hold_seconds=torch.full((4, 2), 6.0))
    assert torch.equal(control._dwell_features[..., 0], control._dwell_features[..., 1])   # one value, both channels
    for _ in range(30):
        control.step()
    assert torch.allclose(control._time_offsets[:, 0], torch.full((4,), 0.2))   # floored at min_lead_s
    assert torch.allclose(control._dwell_features[..., 0], torch.full((4, 2), 0.6))


def test_manual_goal_rejects_a_bad_duration_or_floor(tmp_path, monkeypatch):
    control = _single(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="hold_duration"):
        _manual(control, hold_seconds=torch.ones(4, 2), hold_duration=torch.ones(4, 3))
    with pytest.raises(ValueError, match="deadline_floor"):
        _manual(control, hold_seconds=torch.ones(4, 2), deadline_floor=-1.0)


def test_the_schedule_re_issued_as_a_manual_goal_changes_no_goal_input(tmp_path, monkeypatch):
    """The CPU analogue of S0's manual-path check (``goal_causality_x0.manual_path_check``, ``fixed``)."""
    control = _single(tmp_path, monkeypatch)
    for motion, t in ((0, 0.4), (0, 1.5), (0, 2.6), (0, 4.9), (1, 0.3), (1, 3.0)):
        control.clear_manual_goal()
        control.env.motion_manager.motion_ids[:] = motion
        control.env.motion_manager.motion_times[:] = t
        control._refresh_goal_indices()
        own = _ctx(control)
        g = control._gathered
        now = control.env.motion_manager.motion_times.unsqueeze(-1)
        finite = lambda x: torch.where(torch.isfinite(x), x, torch.zeros_like(x))  # noqa: E731
        ones = torch.ones_like(control.goal_valid)
        control.set_manual_goal(
            node_ids=torch.where(control.goal_valid, g["node"], torch.full_like(g["node"], -1)),
            pose_motion_ids=control._goal_motion_ids.clone(), pose_times=control.target_times.clone(),
            time_offsets=control._time_offsets.clone(), pose_visible=ones, contact_visible=ones,
            hold_seconds=finite((g["t_end"] - now).clamp(min=0.0)), hold_duration=finite(g["t_end"] - g["t_hold"]),
            deadline_floor=0.0)
        man = _ctx(control)
        for key in ("ref_pos", "ref_rot", "target_times", "time_offsets", "target_poses_masks", "target_bodies_masks"):
            assert torch.equal(getattr(man.masked_mimic, key), getattr(own.masked_mimic, key)), (motion, t, key)
        for key in ("contact_spec", "orient_spec", "visible", "dwell_features"):
            assert torch.equal(getattr(man.contact_goal, key), getattr(own.contact_goal, key)), (motion, t, key)


# --------------------------------------------------------------------------- #
# The panel's timing
# --------------------------------------------------------------------------- #
def _sequence():
    return VizSequence("s", 0, 0.0, [VizGoal("a", 1, 0, 1.0, reach_s=1.0, hold_s=2.0),
                                     VizGoal("b", 0, 0, 3.0, reach_s=0.5, hold_s=4.0)])


@pytest.mark.parametrize("t", [0.0, 0.6, 1.0, 2.2, 3.0, 3.2, 7.4, 9.0])
def test_training_timing_serves_each_goal_as_a_segment(t):
    out = fill_goal_slots([_sequence()], torch.zeros(1, dtype=torch.long), t, 2, 1.2, torch.device("cpu"),
                          timing="training")
    seq = _sequence()
    first = seq.active_index(t)
    for slot, index in enumerate(range(first, min(first + 2, 2))):
        goal, end = seq.goals[index], seq.ends[index]
        assert out["time_offsets"][0, slot] == pytest.approx(max(end - goal.hold_s - t, 0.0))
        assert out["hold_seconds"][0, slot] == pytest.approx(max(end - t, 0.0))
        assert out["hold_duration"][0, slot] == pytest.approx(goal.hold_s)
    assert out["deadline_floor"] == 0.0


@pytest.mark.parametrize("t", [0.0, 0.6, 1.0, 2.2, 3.2, 7.4])
def test_legacy_timing_is_the_shipped_protocol(t):
    out = fill_goal_slots([_sequence()], torch.zeros(1, dtype=torch.long), t, 2, 1.2, torch.device("cpu"))
    assert "hold_duration" not in out and "deadline_floor" not in out
    seq = _sequence()
    first = seq.active_index(t)
    for slot, index in enumerate(range(first, min(first + 2, 2))):
        goal, end = seq.goals[index], seq.ends[index]
        assert out["time_offsets"][0, slot] == pytest.approx(max(end - goal.hold_s - t, 1.2))
        assert out["hold_seconds"][0, slot] == pytest.approx(min(max(end - t, 0.0), goal.hold_s))


# --------------------------------------------------------------------------- #
# Route plans
# --------------------------------------------------------------------------- #
def _graph_json():
    seg = lambda name, node, t0, th, t1: dict(name=name, node=node, t_start=t0, t_hold=th, t_end=t1,  # noqa: E731
                                              duration_s=t1 - t0, hold_id=f"x@{int(th * 60)}")
    return {
        "nodes": [{"key": "standing|L_FOOT:G|R_FOOT:G@upright"}, {"key": "Fam_h1|L_FOOT:G@upright"},
                  {"key": "Fam|R_FOOT:G@prone"}],
        "clips": {
            "c_a": dict(family="Fam", variant_s=0.0, source_stem="c_a", segments=[
                seg("standing", 0, 0.0, 0.0, 0.9), seg("Fam_h1", 1, 1.5, 1.6, 2.2), seg("Fam", 2, 4.0, 9.0, 10.0),
                seg("standing", 0, 12.0, 12.5, 12.6)]),
            "c_b": dict(family="Fam", variant_s=0.0, source_stem="c_b", segments=[
                seg("standing", 0, 0.0, 0.2, 0.5), seg("Fam", 2, 1.0, 2.0, 3.0)]),        # a shorter family hold
            "c_a_x3s": dict(family="Fam", variant_s=3.0, source_stem="c_a", segments=[
                seg("Fam", 2, 4.0, 12.0, 13.0)]),                                          # extended: never chosen
        },
    }


def test_route_goals_mirror_the_clips_own_segments():
    from data.scripts.make_route_probe_plans import route_goals

    goals = route_goals(_graph_json(), "c_a", target=2, exit_holds=1)
    assert [g["config"].split("|")[0] for g in goals] == ["standing", "Fam_h1", "Fam", "standing"]
    ends, t = [], 0.0
    for g in goals:
        t += g["reach_s"] + g["hold_s"]
        ends.append(round(t, 4))
    assert ends == [0.9, 2.2, 10.0, 12.6]                       # each goal ends at its segment's t_end
    assert [g["reach_s"] for g in goals] == [0.0, 0.7, 6.8, 2.5]   # to the hold frame, from the previous end
    assert [g["pose_time"] for g in goals] == [0.0, 1.6, 9.0, 12.5]


def test_route_plans_take_the_release_fork_exemplar_and_mark_the_target():
    from data.scripts.make_route_probe_plans import route_plans

    plans = route_plans(_graph_json())
    assert list(plans) == ["fork_Fam"]
    plan = plans["fork_Fam"]
    assert plan["start"] == {"clip": "c_a", "time": 0.0}         # the longest family hold, an unextended clip
    assert plan["target"] == {"goal": 2, "config": "Fam|R_FOOT:G@prone", "window_s": [9.0, 10.0]}


# --------------------------------------------------------------------------- #
# The port helpers
# --------------------------------------------------------------------------- #
def test_field_path_resolves_context_paths_and_refuses_others():
    assert expert_port.field_path("expert_masked_mimic.ref_pos").path == "expert_masked_mimic.ref_pos"
    assert expert_port.field_path("expert_contact_goal.dwell_features").path == "expert_contact_goal.dwell_features"
    assert expert_port.field_path("masked_mimic.mimic.ref_state.rigid_body_pos").path == (
        "masked_mimic.mimic.ref_state.rigid_body_pos")
    for bad in ("expert_masked_mimic", "nope.ref_pos", "masked_mimic.nope"):
        with pytest.raises(ValueError):
            expert_port.field_path(bad)


def test_rewire_moves_every_goal_binding_and_nothing_else():
    poses = MdpComponent(compute_func=lambda **k: None, dynamic_vars={
        "current_state_body_pos": EnvContext.current.rigid_body_pos,
        "masked_mimic_ref_pos": EnvContext.masked_mimic.ref_pos,
        "masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks})
    goal = MdpComponent(compute_func=lambda **k: None, dynamic_vars={
        "contact_spec": EnvContext.contact_goal.contact_spec, "dwell_features": EnvContext.contact_goal.dwell_features})
    state = MdpComponent(compute_func=lambda **k: None, dynamic_vars={"x": EnvContext.current.rigid_body_ground_forces})
    comps = {"expert_masked_mimic_target_poses": poses, "expert_contact_goal_obs": goal, "expert_contact_state_obs": state}
    assert set(expert_port.goal_view_bindings(comps)) == {"expert_masked_mimic_target_poses", "expert_contact_goal_obs"}
    rewired = expert_port.rewire_expert_goal_view(comps)
    assert rewired == {"expert_masked_mimic_target_poses": ["masked_mimic_ref_pos", "masked_mimic_target_bodies_masks"],
                       "expert_contact_goal_obs": ["contact_spec", "dwell_features"]}
    assert poses.dynamic_vars["masked_mimic_ref_pos"].path == "expert_masked_mimic.ref_pos"
    assert poses.dynamic_vars["current_state_body_pos"].path == "current.rigid_body_pos"
    assert state.dynamic_vars["x"].path == "current.rigid_body_ground_forces"
    assert expert_port.goal_view_bindings(comps) == {}


def _expert_env_config(graph_path, **ctrl_edit):
    from protomotions.envs.control.contact_graph_control import ContactGraphControlConfig

    ctrl = ContactGraphControlConfig(graph_file=str(graph_path), num_goal_steps=2, num_masked_future_steps=2,
                                     pose_visible_prob=1.0, contact_visible_prob=1.0, full_pose_prob=1.0,
                                     force_max_conditioned_bodies_prob=0.0,
                                     force_small_num_conditioned_bodies_prob=0.0, **SCHEDULE)
    for k, v in ctrl_edit.items():
        setattr(ctrl, k, v)
    ids = torch.arange(len(SIM_BODY_NAMES))
    return SimpleNamespace(
        control_components={"contact_graph": ctrl}, ref_respawn_offset=0.005, contact_force_on_threshold_n=5.0,
        contact_force_off_threshold_n=2.0, ref_contact_smooth_window=7, num_state_history_steps=20,
        motion_manager=SimpleNamespace(realign_motion_with_humanoid_on_each_step=False),
        observation_components={
            "masked_mimic_target_poses": SimpleNamespace(static_params={"conditionable_body_ids": ids}),
            "masked_mimic_target_masks": SimpleNamespace(static_params={"conditionable_body_ids": ids})})


def test_expert_goal_contract_reads_the_experts_resolved_config(tmp_path):
    graph_path = _graph_file(tmp_path)
    contract = expert_port.expert_goal_contract(_expert_env_config(graph_path))
    assert contract["num_goal_steps"] == 2 and contract["full_visibility"] is True
    assert {k: contract[k] for k in SCHEDULE} == SCHEDULE
    assert contract["graph_sha256"] == file_sha256(graph_path)
    assert contract["pair_names"] == _graph_v2()["pair_names"]
    assert contract["ref_respawn_offset"] == 0.005 and contract["num_state_history_steps"] == 20
    json.dumps(contract)                                     # plain values: it is stored in a resolved config
    assert expert_port.expert_conditionable_body_count(_expert_env_config(graph_path)) == len(SIM_BODY_NAMES)
    masked = expert_port.expert_goal_contract(_expert_env_config(graph_path, pose_visible_prob=0.75))
    assert masked["full_visibility"] is False


def test_port_identity_problems_name_every_difference(tmp_path):
    graph_path = _graph_file(tmp_path)
    contract = expert_port.expert_goal_contract(_expert_env_config(graph_path))
    package = tmp_path / "motions.pt"
    package.write_bytes(b"package")
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps(expert_port.single_expert_routing(_graph_v2()["motion_names"])))

    class Robot:
        def __init__(self, name="a.xml"):
            self.asset = SimpleNamespace(asset_root=str(tmp_path), asset_file_name=None, usd_asset_file_name=name)

    ok = dict(robot_config=Robot(), expert_robot_config=Robot(), motion_file=str(package),
              expert_motion_file=str(package), graph_file=str(graph_path), contract=contract,
              expert_paths=["e.ckpt"], routing_file=str(routing))
    assert expert_port.port_identity_problems(**ok) == []
    bad_routing = tmp_path / "bad_routing.json"
    bad_routing.write_text(json.dumps({"motion_expert": [0, 1], "motion_names": _graph_v2()["motion_names"]}))
    for edit, match in ((dict(expert_robot_config=Robot("b.xml")), "usd_asset_file_name"),
                        (dict(expert_motion_file=str(other)), "motion package"),
                        (dict(expert_paths=["a", "b"]), "one expert"),
                        (dict(routing_file=None), "routing table"),
                        (dict(routing_file=str(bad_routing)), "single-expert"),
                        (dict(contract={**contract, "graph_sha256": "0" * 64}), "not the expert's")):
        problems = expert_port.port_identity_problems(**{**ok, **edit})
        assert any(match in p for p in problems), (edit, problems)


# --------------------------------------------------------------------------- #
# The merge loop
# --------------------------------------------------------------------------- #
def _fake_expert(tmp_path, actor_keys, future_steps):
    from protomotions.envs.obs import to_float

    expert_dir = tmp_path / "expert"
    expert_dir.mkdir(exist_ok=True)
    component = lambda: MdpComponent(compute_func=to_float,  # noqa: E731  (picklable: it is saved below)
                                     dynamic_vars={"x": EnvContext.current.rigid_body_pos})
    env = SimpleNamespace(
        num_state_history_steps=15,
        control_components={"mimic": SimpleNamespace(future_steps=future_steps)},
        observation_components={k: component() for k in actor_keys + ["mimic_target_poses"]})
    agent = SimpleNamespace(model=SimpleNamespace(actor=SimpleNamespace(in_keys=list(actor_keys))))
    torch.save({"env": env, "agent": agent}, expert_dir / "resolved_configs.pt")
    return str(expert_dir / "last.ckpt")


@pytest.mark.parametrize("actor_keys, future_steps, want", [
    (["max_coords_obs", "masked_mimic_target_poses"], [1, 5, 10, 15], 1),   # Design-B: critic-only window, skipped
    (["max_coords_obs", "mimic_target_poses"], 3, 3),                       # a tracker: the student's window grows
])
def test_the_merge_loop_reads_an_experts_future_window_only_if_its_actor_does(tmp_path, actor_keys, future_steps, want):
    from examples.experiments.masked_mimic import contact_graph_transformer as module
    from protomotions.tests.test_contact_graph import _StubRobotConfig, _experiment_args, _toy_graph_payload

    graph_path = tmp_path / "graph.pt"
    torch.save(_toy_graph_payload(), graph_path)
    args = _experiment_args(contact_graph_file=str(graph_path),
                            expert_model_path=_fake_expert(tmp_path, actor_keys, future_steps))
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    env_cfg = module.env_config(robot_cfg, args)
    assert env_cfg.control_components["contact_graph"].future_steps == want
    assert {f"expert_{k}" for k in actor_keys} <= set(env_cfg.observation_components)
