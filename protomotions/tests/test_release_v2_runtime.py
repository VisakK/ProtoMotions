# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime half of a curation release (BodyFix Step 5; BUILD_PLAN Steps 9-10): graph v2's manual-goal rule,
the contact-target sidecar, the physics tables' library checks, the release record check and the experiment
wiring. Toy tables on the stub env of ``test_contact_graph``; the release itself is tested in
``data/scripts/reference_curation/tests/test_release_v2.py``."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
import torch

from protomotions.components.contact_graph import ContactGraph
from protomotions.envs.control.contact_targets import ROLE_NAMES, ContactTargets
from protomotions.envs.control.physics_terms import PhysicsTables
from protomotions.envs.control.support_penalty import unwanted_support
from protomotions.tests.test_contact_graph import SIM_BODY_NAMES, _current_state, _stub_env
from protomotions.utils.release_identity import (
    ReleaseMismatchError,
    file_sha256,
    load_release,
    require_artifact,
)

PAIRS = ["L_FOOT:G", "R_FOOT:G", "L_HAND:G", "L_FOOT+L_HAND"]
ZONES = ["L_FOOT", "R_FOOT", "L_HAND"]
INF = float("inf")


def _graph_v2():
    """Two clips. ``crow`` (L_HAND:G) has two source holds: clip_a's carries the L_FOOT+L_HAND pair, clip_b's
    does not -- the side-specific case a 0.5-vote union would merge."""
    return {
        "motion_names": ["clip_a", "clip_b"], "pair_names": list(PAIRS), "orientation_names": ["upright", "prone"],
        "node_keys": ["crow|L_HAND:G@prone", "stand|L_FOOT:G|R_FOOT:G@upright"],
        "node_contact": torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),   # consensus: no pair
        "node_orient": torch.tensor([1, 0]),
        "seg_node": torch.tensor([[1, 0], [0, -1]]),
        "seg_contact": torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                                     [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        "seg_start": torch.tensor([[0.0, 2.0], [0.0, INF]]),
        "seg_end": torch.tensor([[2.0, 5.0], [4.0, INF]]),
        "seg_hold": torch.tensor([[1.0, 3.0], [2.0, INF]]),
        "seg_count": torch.tensor([2, 1]),
        "min_lead_s": 0.2,
        "zone_order": list(ZONES),
        "zone_bodies": {"L_FOOT": ["L_Ankle", "L_Toe"], "R_FOOT": ["R_Ankle", "R_Toe"], "L_HAND": ["L_Wrist", "L_Hand"]},
        "graph_version": 2, "manual_goal_rule": "segment", "fps": 60,
        "hold_ids": ["clip_a@180", "clip_a@60", "clip_b@120"],
        "seg_hold_index": torch.tensor([[1, 0], [2, -1]]),
        "motion_num_frames": torch.tensor([600, 300]),
    }


def _targets(graph_path, **edit):
    """A sidecar consistent with ``_graph_v2``: clip_a's crow requires L_HAND support and the pair (closed by the
    reference on frames 150-239 only), R_FOOT known free; clip_b's crow masks nothing."""
    M, S, P, Z = 2, 2, len(PAIRS), len(ZONES)
    g = _graph_v2()
    role = torch.zeros(M, S, P, dtype=torch.int8)
    role[0, 0, :2] = ROLE_NAMES.index("required_support")
    role[0, 1, 2] = ROLE_NAMES.index("required_support")
    role[0, 1, 3] = ROLE_NAMES.index("required_touch")
    role[1, 0, 2] = ROLE_NAMES.index("required_support")
    commanded = g["seg_contact"] > 0.5
    free = torch.zeros(M, S, Z, dtype=torch.bool)
    free[0, 1, 1] = True
    ok = torch.zeros(M, 600, 1, dtype=torch.bool)
    ok[0, 150:240, 0] = True
    payload = {
        "kind": "contact_targets", "version": 1, "release_id": "toy", "fps": 60, "plant_sha256": None,
        "graph_sha256": file_sha256(graph_path), "motion_names": ["clip_a", "clip_b"], "pair_names": list(PAIRS),
        "zone_order": list(ZONES), "hold_ids": list(g["hold_ids"]), "seg_hold_index": g["seg_hold_index"].clone(),
        "role_names": list(ROLE_NAMES), "flag_names": ["fit_closer"],
        "seg_commanded": commanded, "seg_configured": commanded.clone(), "seg_masked": torch.zeros(M, S, P, dtype=torch.bool),
        "seg_role": role, "seg_critical": commanded & torch.tensor([False, False, False, True]),
        "seg_restored": torch.zeros(M, S, P, dtype=torch.bool), "seg_ground_free": free,
        "seg_flags": torch.zeros(M, S, 1, dtype=torch.bool),
        "frame_pair_slots": torch.tensor([3]), "frame_pair_ok": ok, "frame_len": torch.tensor([600, 300]),
    }
    payload.update(edit)
    path = graph_path.parent / "contact_targets.pt"
    torch.save(payload, path)
    return path


class _LibWithLayout:
    """The stub library with the frame counts and frame rate a real MotionLib exposes."""

    def __init__(self, base, frames=(600, 300), fps=60.0):
        self._base = base
        self.motion_files = base.motion_files
        self.motion_num_frames = torch.tensor(list(frames))
        self.motion_dt = torch.full((len(frames),), 1.0 / fps)
        self.motion_file = None

    def __getattr__(self, name):
        return getattr(self._base, name)


def _control(tmp_path, monkeypatch, graph=None, frames=(600, 300), fps=60.0, sense_pairs=False, **overrides):
    from protomotions.envs.control import contact_graph_control as module
    from protomotions.envs.control.mimic_control import MimicControl

    torch.manual_seed(0)
    graph_path = tmp_path / "contact_graph.pt"
    if graph is not None or not graph_path.exists():   # tables, sidecars and records hash this file
        torch.save(graph or _graph_v2(), graph_path)
    monkeypatch.setattr(MimicControl, "populate_context", lambda self, ctx: None)
    env = _stub_env()
    env.motion_lib = _LibWithLayout(env.motion_lib, frames, fps)
    if sense_pairs:
        env.robot_config.contact_pair_bodies = list(SIM_BODY_NAMES)
    kwargs = {"graph_file": str(graph_path), "num_goal_steps": 2, "num_masked_future_steps": 2,
              "include_current_segment": True, "interval_schedule": True}
    kwargs.update(overrides)
    return module.ContactGraphControl(module.ContactGraphControlConfig(**kwargs), env), graph_path


# --------------------------------------------------------------------------- #
# Graph v2: manual goals resolve the side-specific segment
# --------------------------------------------------------------------------- #
def test_manual_goals_resolve_the_segment_holding_the_pose():
    graph = ContactGraph(_graph_v2())
    node = torch.tensor([0, 0, 0, 1])
    motion = torch.tensor([0, 1, 0, 0])
    time = torch.tensor([3.0, 2.0, 1.0, 1.0])
    contact, resolved = graph.manual_contact(node, motion, time)
    assert resolved.tolist() == [True, True, False, True]
    assert contact[0].tolist() == [0.0, 0.0, 1.0, 1.0]     # clip_a's crow: with its pair
    assert contact[1].tolist() == [0.0, 0.0, 1.0, 0.0]     # clip_b's crow: without
    assert contact[2].tolist() == [0.0, 0.0, 1.0, 0.0]     # the pose is a stand, not a crow: the node's consensus
    legacy = ContactGraph({**_graph_v2(), "manual_goal_rule": "node", "graph_version": 1})
    assert legacy.manual_contact(node, motion, time)[0][0].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_graph_v2_checks_the_library_layout():
    graph = ContactGraph(_graph_v2())
    graph.validate_against_motion_lib(["d/clip_a.motion", "d/clip_b.motion"], motion_num_frames=[600, 300], fps=60)
    with pytest.raises(ValueError, match="frame counts"):
        graph.validate_against_motion_lib(["d/clip_a.motion", "d/clip_b.motion"], motion_num_frames=[600, 301])
    with pytest.raises(ValueError, match="fps"):
        graph.validate_against_motion_lib(["d/clip_a.motion", "d/clip_b.motion"], fps=30)
    bad = _graph_v2()
    bad["seg_hold_index"] = torch.tensor([[1, 0], [2, 0]])            # a hold id on padding
    with pytest.raises(ValueError, match="seg_hold_index"):
        ContactGraph(bad)


def test_the_control_serves_a_manual_goal_its_segments_contacts(tmp_path, monkeypatch):
    control, _ = _control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    shape = (4, 2)
    node = torch.tensor([[0, 1], [0, 1], [1, -1], [0, -1]])
    motion = torch.tensor([[0, 0], [1, 0], [0, 0], [0, 0]])
    times = torch.tensor([[3.0, 1.0], [2.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    control.set_manual_goal(node, motion, times, torch.ones(shape), torch.ones(shape, dtype=torch.bool),
                            torch.ones(shape, dtype=torch.bool), hold_seconds=torch.ones(shape))
    got = control._gathered["contact"][:, 0]
    assert got[0].tolist() == [0.0, 0.0, 1.0, 1.0] and got[1].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert got[3].tolist() == [0.0, 0.0, 1.0, 0.0]          # crow commanded at a standing pose: consensus
    assert control._manual_resolved[:, 0].tolist() == [True, True, True, False]


def test_the_control_refuses_a_library_of_another_layout(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="frame counts"):
        _control(tmp_path, monkeypatch, frames=(601, 300))
    with pytest.raises(ValueError, match="fps"):
        _control(tmp_path, monkeypatch, fps=30.0)


# --------------------------------------------------------------------------- #
# Physics tables: the library's frames and fps, v2's graph identity
# --------------------------------------------------------------------------- #
def _tables(tmp_path, graph_path, **edit):
    from protomotions.tests.test_expert60_ft_c_physics import _tables as ft_c_tables

    payload = torch.load(ft_c_tables(tmp_path), weights_only=False)
    M, S = 2, 2
    payload.update(swing=torch.zeros(M, 600, 3, dtype=torch.bool), swing_len=torch.tensor([600, 300]),
                   seg_cop_rel=torch.zeros(M, S, 2), seg_cop_valid=torch.zeros(M, S, dtype=torch.bool),
                   seg_com_rel=torch.zeros(M, S, 2), seg_zone_share=torch.zeros(M, S, 3),
                   seg_share_valid=torch.zeros(M, S, dtype=torch.bool), seg_lean_gate=torch.zeros(M, S, dtype=torch.bool),
                   seg_pair_consequential=torch.zeros(M, S, len(PAIRS), dtype=torch.bool),
                   version=2, pair_names=list(PAIRS), graph_sha256=file_sha256(graph_path))
    payload.update(edit)
    path = tmp_path / "physics_tables_v2.pt"
    torch.save(payload, path)
    return path


def test_tables_refuse_another_library_layout(tmp_path):
    torch.save(_graph_v2(), tmp_path / "g.pt")
    path = _tables(tmp_path, tmp_path / "g.pt")
    names = ["clip_a", "clip_b"]
    PhysicsTables(path, names, SIM_BODY_NAMES, "cpu", motion_num_frames=torch.tensor([600, 300]), fps=60)
    with pytest.raises(ValueError, match="frame counts"):
        PhysicsTables(path, names, SIM_BODY_NAMES, "cpu", motion_num_frames=torch.tensor([600, 299]), fps=60)
    with pytest.raises(ValueError, match="fps"):
        PhysicsTables(path, names, SIM_BODY_NAMES, "cpu", fps=30)


def test_the_control_refuses_v2_tables_of_another_graph(tmp_path, monkeypatch):
    torch.save(_graph_v2(), tmp_path / "contact_graph.pt")
    good = _tables(tmp_path, tmp_path / "contact_graph.pt")
    control, _ = _control(tmp_path, monkeypatch, physics_tables_file=str(good))
    assert control._physics is not None and control._physics.version == 2
    other = _tables(tmp_path, tmp_path / "contact_graph.pt", graph_sha256="0" * 64)
    with pytest.raises(ValueError, match="graph sha256"):
        _control(tmp_path, monkeypatch, physics_tables_file=str(other))
    shuffled = _tables(tmp_path, tmp_path / "contact_graph.pt", pair_names=list(reversed(PAIRS)))
    with pytest.raises(ValueError, match="pair vocabulary"):
        _control(tmp_path, monkeypatch, physics_tables_file=str(shuffled))


# --------------------------------------------------------------------------- #
# The contact-target sidecar
# --------------------------------------------------------------------------- #
def test_targets_validate_against_the_graph(tmp_path):
    gpath = tmp_path / "g.pt"
    torch.save(_graph_v2(), gpath)
    graph = ContactGraph(_graph_v2())
    names = ["clip_a", "clip_b"]
    kw = dict(motion_num_frames=torch.tensor([600, 300]), fps=60, graph_sha256=file_sha256(gpath))
    t = ContactTargets(_targets(gpath), graph, names, "cpu", **kw)
    assert t.frame_pairs(torch.tensor([0, 0, 1]), torch.tensor([3.0, 1.0, 3.0]))[:, 3].tolist() == [True, False, False]
    bad = {
        "pair vocabulary": dict(pair_names=list(reversed(PAIRS))),
        "hold ids": dict(hold_ids=["x", "y", "z"]),
        "commanded": dict(seg_commanded=torch.zeros(2, 2, 4, dtype=torch.bool)),
        "frame tables": dict(frame_len=torch.tensor([600, 299])),
        "fps": dict(fps=30),
        "graph sha256": dict(graph_sha256="0" * 64),
        "role vocabulary": dict(role_names=["none"]),
    }
    for match, edit in bad.items():
        with pytest.raises(ValueError, match=match):
            ContactTargets(_targets(gpath, **edit), graph, names, "cpu", **kw)


def test_support_term_charges_only_known_free_zones():
    forces = torch.zeros(2, len(SIM_BODY_NAMES), 3)
    forces[:, SIM_BODY_NAMES.index("R_Toe"), 2] = 100.0
    ref = torch.zeros(2, len(SIM_BODY_NAMES), 3)
    ref[..., 2] = 0.5                                              # the reference keeps every zone up
    zm = torch.zeros(3, len(SIM_BODY_NAMES))
    for z, bodies in enumerate((("L_Ankle", "L_Toe"), ("R_Ankle", "R_Toe"), ("L_Wrist", "L_Hand"))):
        zm[z, [SIM_BODY_NAMES.index(b) for b in bodies]] = 1.0
    args = dict(ground_forces=forces, ref_body_pos=ref, goal_ground=torch.zeros(2, 3, dtype=torch.bool),
                in_hold=torch.ones(2, dtype=torch.bool), zone_matrix=zm, clear_height=0.25, load_ref_n=72.6)
    assert unwanted_support(**args)[1].tolist() == [100.0, 100.0]           # fine-tune C's complement rule
    known = torch.tensor([[False, True, False], [False, False, False]])     # R_FOOT known free on env 0 only
    assert unwanted_support(**args, known_free=known)[1].tolist() == [100.0, 0.0]


def _ctx(num_envs=4, hand_n=0.0, toe_n=0.0, pair_n=0.0):
    cur = _current_state(num_envs=num_envs)
    forces = torch.zeros(num_envs, len(SIM_BODY_NAMES), 3)
    forces[:, SIM_BODY_NAMES.index("L_Hand"), 2] = hand_n
    forces[:, SIM_BODY_NAMES.index("R_Toe"), 2] = toe_n
    cur.rigid_body_ground_forces = forces
    pair = torch.zeros(num_envs, len(SIM_BODY_NAMES), len(SIM_BODY_NAMES), 3)
    pair[:, SIM_BODY_NAMES.index("L_Toe"), SIM_BODY_NAMES.index("L_Hand"), 2] = pair_n
    cur.rigid_body_pair_contact_forces = pair
    ref = torch.zeros(num_envs, len(SIM_BODY_NAMES), 3)
    ref[..., 2] = 0.5
    return SimpleNamespace(current=cur, mimic=SimpleNamespace(ref_state=SimpleNamespace(rigid_body_pos=ref)))


def test_target_diagnostics_and_the_known_free_support_gate(tmp_path, monkeypatch):
    torch.save(_graph_v2(), tmp_path / "contact_graph.pt")
    tpath = _targets(tmp_path / "contact_graph.pt")
    control, _ = _control(tmp_path, monkeypatch, sense_pairs=True, contact_targets_file=str(tpath))
    mm = control.env.motion_manager
    mm.motion_ids[:] = torch.tensor([0, 0, 1, 0])
    mm.motion_times[:] = torch.tensor([3.0, 4.5, 2.0, 1.0])   # crow (pair closed), crow (pair open), crow b, stand
    control.reset(torch.arange(4))
    out = control._target_terms(_ctx(hand_n=300.0, toe_n=80.0, pair_n=50.0))
    assert out["required_support_gate"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert out["required_support_met"].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.5])   # standing: one foot down
    assert out["pair_target_gate"].tolist() == [1.0, 0.0, 0.0, 0.0]        # closed by the reference at 3.0 s only
    assert out["pair_target_met"].tolist() == [1.0, 1.0, 1.0, 1.0]        # ungated rows carry the gated mean
    assert out["known_free_load_n"].tolist() == [80.0, 80.0, 0.0, 0.0]     # R_FOOT is known free in clip_a's crow
    _, charged, _ = control._unwanted_support(_ctx(hand_n=300.0, toe_n=80.0))
    assert charged.tolist() == [80.0, 80.0, 0.0, 0.0]                      # clip_b's crow: R_FOOT not known free
    no_pairs = control._target_terms(_ctx(hand_n=300.0, toe_n=80.0, pair_n=0.0))
    assert no_pairs["pair_target_met"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_without_a_sidecar_nothing_changes(tmp_path, monkeypatch):
    control, _ = _control(tmp_path, monkeypatch)
    control.reset(torch.arange(4))
    assert all(v is None for v in control._target_terms(_ctx()).values())
    assert control.release is None


# --------------------------------------------------------------------------- #
# The release record
# --------------------------------------------------------------------------- #
def _record(tmp_path, **artifacts):
    rec = {"kind": "reference_release", "release_id": "toy.release_v2.0", "plant": {"plant_sha256": None},
           "artifacts": {role: {"path": str(p), "sha256": file_sha256(p)} for role, p in artifacts.items()}}
    path = tmp_path / "release.json"
    path.write_text(json.dumps(rec))
    return path


def test_require_artifact_compares_contents(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    rec = load_release(_record(tmp_path, graph=a))
    assert require_artifact(rec, "graph", a) == file_sha256(a)
    with pytest.raises(ReleaseMismatchError, match="not release"):
        require_artifact(rec, "graph", b)
    with pytest.raises(ReleaseMismatchError, match="no physics_tables"):
        require_artifact(rec, "physics_tables", a)
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ReleaseMismatchError, match="not a release record"):
        load_release(tmp_path / "x.json")


def test_the_control_refuses_artifacts_that_are_not_its_releases(tmp_path, monkeypatch):
    torch.save(_graph_v2(), tmp_path / "contact_graph.pt")
    package = tmp_path / "motions.pt"
    package.write_bytes(b"library")
    rec = _record(tmp_path, package=package, graph=tmp_path / "contact_graph.pt")
    control, _ = _control(tmp_path, monkeypatch)
    control.env.motion_lib.motion_file = str(package)
    control.config.release_file = str(rec)
    control._init_release()
    assert control.release["release_id"] == "toy.release_v2.0"
    other = tmp_path / "other.pt"
    other.write_bytes(b"another library")
    control.env.motion_lib.motion_file = str(other)
    with pytest.raises(ReleaseMismatchError, match="motion library"):
        control._init_release()


# --------------------------------------------------------------------------- #
# Experiment wiring
# --------------------------------------------------------------------------- #
def test_the_experiment_wires_the_sidecar_and_the_release(tmp_path):
    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.robot_configs.factory import robot_config
    from protomotions.tests.test_expert60_ft_a import _graph_with_near_start_hold

    _graph_with_near_start_hold(tmp_path)
    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)

    def build(argv):
        args = parser.parse_args(["--hold-graph-file", str(tmp_path / "g.pt")] + argv)
        args.motion_file, args.scenes_file, args.batch_size, args.training_max_steps = "unused.pt", None, 32, 256
        robot = robot_config("smpl_yogi_v2")
        exp.configure_robot_and_simulator(robot, SimpleNamespace(), args)
        return exp.env_config(robot, args)

    plain = build([])
    assert not any(k.startswith("diag_required") for k in plain.reward_components)
    assert plain.control_components["contact_graph"].contact_targets_file == ""
    env = build(["--contact-targets", "t.pt", "--release-record", "r.json"])
    cfg = env.control_components["contact_graph"]
    assert cfg.contact_targets_file == "t.pt" and cfg.release_file == "r.json"
    diags = ["diag_required_support_met", "diag_required_support_gate", "diag_pair_target_met",
             "diag_pair_target_gate", "diag_known_free_load_n"]
    for d in diags:
        assert env.reward_components[d].static_params["weight"] == 0.0
