# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hold-graph pipeline behind the goal-conditioned expert.

Covers the pure functions of ``propose_hold_manifest.py``,
``make_hold_extended_clips.py`` and ``build_hold_graph.py`` on synthetic
inputs, the schedule semantics the built graph produces through the runtime
``ContactGraph``, and the asymmetric actor/critic contract of
``mlp_goal_conditioned.py`` (``expert_revist/expert_revisit.MD``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "data" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from build_hold_graph import build_graph, node_key  # noqa: E402
from make_hold_graph_probe_plans import family_plans  # noqa: E402
from make_hold_extended_clips import (  # noqa: E402
    extend_motion,
    insertion_plan,
    shifted_hold,
    splice_index,
)
from propose_hold_manifest import (  # noqa: E402
    demote_supported_pairs,
    drop_standing_leg_flags,
    median_filter,
    name_holds,
    split_by_config,
    stillness_windows,
    vote_pairs,
)

PAIRS = ["L_FOOT:G", "R_FOOT:G", "L_HAND:G", "R_HAND:G", "L_SHANK+L_UPPER_ARM", "L_FOOT+R_SHANK"]
ORIENTS = ["upright", "inverted", "prone"]


# --------------------------------------------------------------------------- #
# propose_hold_manifest
# --------------------------------------------------------------------------- #
def test_median_filter_removes_single_frame_spikes():
    x = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0])
    assert np.allclose(median_filter(x, 3), 0.0)
    assert np.array_equal(median_filter(x, 1), x)


def test_stillness_windows_merge_then_drop():
    # still 0-9, blip at 10, still 11-19, moving 20-29, still 30-32 (too short)
    speed = np.full(33, 0.05)
    speed[10] = 1.0
    speed[20:30] = 1.0
    windows = stillness_windows(speed, v_still=0.15, min_frames=5, merge_frames=3)
    assert windows == [(0, 19)]
    # No merging: two windows, and the trailing 3-frame one is dropped.
    assert stillness_windows(speed, 0.15, 5, 0) == [(0, 9), (11, 19)]


def test_split_by_config_cuts_on_support_change_and_drops_transients():
    ids = np.array([0] * 10 + [1] * 2 + [2] * 10)
    assert split_by_config((0, 21), ids, min_frames=5) == [(0, 9), (12, 21)]
    assert split_by_config((3, 8), ids, min_frames=5) == [(3, 8)]


def test_pair_rules():
    assert demote_supported_pairs({"L_FOOT:G", "R_FOOT:G", "L_FOOT+R_FOOT"}) == {
        "L_FOOT:G",
        "R_FOOT:G",
    }
    # Crow's shin-on-arm pair survives: the upper arm is not grounded.
    crow = {"L_HAND:G", "R_HAND:G", "L_SHANK+L_UPPER_ARM"}
    assert demote_supported_pairs(crow) == crow
    standing = {"L_FOOT:G", "R_FOOT:G", "L_FOOT+R_SHANK"}
    assert drop_standing_leg_flags(standing, "upright") == {"L_FOOT:G", "R_FOOT:G"}
    # Only while standing on both feet: a tree pose keeps its foot-on-thigh pair.
    tree = {"R_FOOT:G", "L_FOOT+R_THIGH"}
    assert drop_standing_leg_flags(tree, "upright") == tree
    assert drop_standing_leg_flags(standing, "prone") == standing


def test_vote_pairs_majority_and_ordering():
    active = {
        "R_HAND:G": np.ones(10, dtype=bool),
        "L_HAND:G": np.array([1] * 6 + [0] * 4, dtype=bool),
        "L_SHANK+L_UPPER_ARM": np.array([1] * 4 + [0] * 6, dtype=bool),
    }
    assert vote_pairs(active, 0, 9, 0.5, "prone") == ["L_HAND:G", "R_HAND:G"]
    assert vote_pairs(active, 0, 3, 0.5, "prone") == [
        "L_HAND:G",
        "R_HAND:G",
        "L_SHANK+L_UPPER_ARM",
    ]


def _hold(name, start, end, pairs, orient="upright", rest=1.0, pose=None):
    hold = int((start + end) / 2)
    return {
        "name": name,
        "frame_start": start,
        "frame_end": end,
        "frame_hold": hold,
        "t_start": start / 60,
        "t_end": end / 60,
        "t_hold": hold / 60,
        "duration_s": (end - start + 1) / 60,
        "pairs": pairs,
        "pairs_ground": [p for p in pairs if p.endswith(":G")],
        "orientation": orient,
        "rest_pose_m": rest,
        "rest_pose_threshold_m": 0.25,
        "same_hold_m": 0.25,
        "_pose": np.zeros((6, 3)) if pose is None else pose,
        "auto": True,
        "auto_rest": False,
    }


def test_name_holds_standing_family_and_grouping():
    far = np.ones((6, 3))
    holds = [
        _hold(None, 0, 60, ["L_FOOT:G", "R_FOOT:G"], rest=0.0),
        _hold(None, 100, 160, ["L_FOOT:G", "R_FOOT:G"], rest=0.4, pose=far * 0.1),  # prep
        _hold(None, 200, 500, ["L_FOOT:G", "R_FOOT:G"], rest=0.3, pose=far),  # warrior
        _hold(None, 520, 700, ["L_FOOT:G", "R_FOOT:G"], rest=0.3, pose=far * 1.05),  # again
        _hold(None, 800, 860, ["L_FOOT:G", "R_FOOT:G"], rest=0.0),
    ]
    name_holds(holds, "Warrior")
    names = [h["name"] for h in holds]
    assert names == ["standing", "Warrior_h1", "Warrior", "Warrior", "standing"]
    assert [h["extend"] for h in holds] == [False, False, True, True, False]


def test_name_holds_rest_end_naming():
    holds = [
        _hold(None, 0, 60, ["L_FOOT:G", "L_HAND:G"], orient="prone", rest=0.0),
        _hold(None, 100, 400, ["L_HAND:G", "R_HAND:G"], orient="inverted", rest=1.0),
    ]
    holds[0]["auto_rest"] = True
    name_holds(holds, "Handstand")
    assert [h["name"] for h in holds] == ["rest_start", "Handstand"]


# --------------------------------------------------------------------------- #
# make_hold_extended_clips
# --------------------------------------------------------------------------- #
def _motion(frames=20, bodies=4):
    torch.manual_seed(0)
    t = torch.linspace(0, 1, frames).view(frames, 1, 1)
    pos = t * torch.ones(frames, bodies, 3)
    quat = torch.zeros(frames, bodies, 4)
    quat[..., 3] = 1.0
    return {
        "state_conversion": "COMMON",
        "fps": 60,
        "dof_pos": torch.arange(frames * 9, dtype=torch.float32).reshape(frames, 9),
        "dof_vel": torch.ones(frames, 9),
        "rigid_body_pos": pos,
        "rigid_body_rot": quat,
        "rigid_body_vel": torch.ones(frames, bodies, 3),
        "rigid_body_ang_vel": torch.ones(frames, bodies, 3),
        "rigid_body_contacts": torch.zeros(frames, bodies, dtype=torch.bool),
        "local_rigid_body_rot": quat.clone(),
        "ground_reaction": torch.rand(frames, 3),
    }


def test_splice_index_and_shifted_hold():
    plan = [(5, 3), (12, 2)]
    index = splice_index(20, plan).tolist()
    assert index == list(range(0, 6)) + [5, 5, 5] + list(range(6, 13)) + [12, 12] + list(range(13, 20))
    hold = {"frame_start": 10, "frame_hold": 12, "frame_end": 15, "t_start": 0, "t_hold": 0, "t_end": 0, "duration_s": 0}
    shifted = shifted_hold(hold, plan, fps=60)
    assert (shifted["frame_start"], shifted["frame_hold"], shifted["frame_end"]) == (13, 15, 20)
    earlier = shifted_hold({"frame_start": 0, "frame_hold": 2, "frame_end": 4}, plan, 60)
    assert (earlier["frame_start"], earlier["frame_hold"], earlier["frame_end"]) == (0, 2, 4)
    assert insertion_plan([{"frame_hold": 7, "extend": True}, {"frame_hold": 3, "extend": False}], 4) == [(7, 4)]
    assert insertion_plan([{"frame_hold": 7, "extend": True}], 0) == []


def test_extend_motion_tiles_and_zeroes_velocity_in_the_hold():
    motion = _motion()
    out = extend_motion(motion, [(5, 6)])
    assert out["rigid_body_pos"].shape[0] == 26
    assert torch.equal(out["rigid_body_pos"][5:12], motion["rigid_body_pos"][5:6].expand(7, -1, -1))
    # Strictly inside the tiled block every velocity is exactly zero.
    assert torch.all(out["rigid_body_vel"][7:10] == 0)
    assert torch.all(out["dof_vel"][7:10] == 0)
    assert "ground_reaction" not in out
    # An empty plan keeps the stored velocities byte-identical.
    same = extend_motion(motion, [])
    assert torch.equal(same["rigid_body_vel"], motion["rigid_body_vel"])
    assert torch.equal(same["dof_vel"], motion["dof_vel"])


# --------------------------------------------------------------------------- #
# build_hold_graph + runtime schedule
# --------------------------------------------------------------------------- #
def _manifest_clips():
    def hold(name, t_start, t_hold, t_end, pairs, orient="upright", extend=False):
        return {
            "name": name, "t_start": t_start, "t_hold": t_hold, "t_end": t_end,
            "pairs": pairs, "orientation": orient, "extend": extend,
            "frame_start": int(t_start * 60), "frame_hold": int(t_hold * 60),
            "frame_end": int(t_end * 60), "speed_at_hold": 0.01, "pelvis_z": 0.9,
        }

    crow = ["L_HAND:G", "R_HAND:G", "L_SHANK+L_UPPER_ARM"]
    return [
        {
            "stem": "crow", "group": "arm_balance", "family": "Crow", "source_stem": "crow",
            "holds": [
                hold("standing", 0.0, 0.0, 1.0, ["L_FOOT:G", "R_FOOT:G"]),
                hold("Crow", 3.0, 3.5, 8.0, crow, "prone", extend=True),
                hold("standing", 10.0, 10.5, 11.0, ["L_FOOT:G", "R_FOOT:G", "L_FOOT+R_SHANK"]),
            ],
        },
        {
            "stem": "crow_x3s", "group": "arm_balance", "family": "Crow", "source_stem": "crow",
            "variant_s": 3.0,
            "holds": [
                hold("standing", 0.0, 0.0, 1.0, ["L_FOOT:G", "R_FOOT:G"]),
                hold("Crow", 3.0, 3.5, 11.0, crow, "prone", extend=True),
                hold("standing", 13.0, 13.5, 14.0, ["L_FOOT:G", "R_FOOT:G"]),
            ],
        },
        {
            "stem": "handstand", "group": "inversion", "family": "Handstand", "source_stem": "handstand",
            "holds": [
                hold("standing", 0.0, 0.0, 1.0, ["L_FOOT:G", "R_FOOT:G"]),
                hold("Handstand", 4.0, 5.0, 9.0, ["L_HAND:G", "R_HAND:G"], "inverted", extend=True),
                hold("standing", 12.0, 12.5, 13.0, ["L_FOOT:G", "R_FOOT:G"]),
            ],
        },
    ]


def test_build_graph_identity_goal_and_edges():
    clips = _manifest_clips()
    payload, description = build_graph(clips, ["crow", "crow_x3s", "handstand"], PAIRS, ORIENTS)
    keys = payload["node_keys"]
    # Standing is one node despite the feet-on-shin flag on one of its segments.
    assert keys.count(node_key("standing", ["L_FOOT:G", "R_FOOT:G"], "upright")) == 1
    assert len(keys) == 3
    crow_id = keys.index(node_key("Crow", ["L_HAND:G", "R_HAND:G"], "prone"))
    # Identity is the ground set; the shin-on-arm pair is in the goal vectors.
    assert payload["node_contact"][crow_id].tolist() == [0, 0, 1, 1, 1, 0]
    assert payload["seg_contact"][0, 1].tolist() == [0, 0, 1, 1, 1, 0]
    # The standing end of clip 0 keeps its incidental flag in seg_contact but not in the node.
    standing_id = keys.index(node_key("standing", ["L_FOOT:G", "R_FOOT:G"], "upright"))
    assert payload["seg_contact"][0, 2].tolist() == [1, 1, 0, 0, 0, 1]
    assert payload["node_contact"][standing_id].tolist() == [1, 1, 0, 0, 0, 0]
    assert payload["seg_count"].tolist() == [3, 3, 3]
    # Edges: standing->Crow (2 occurrences, 1 source), Crow->standing, standing->Handstand, Handstand->standing.
    edges = {(e["src"], e["dst"]): e for e in description["edges"]}
    assert edges[(standing_id, crow_id)]["count"] == 2
    assert edges[(standing_id, crow_id)]["source_count"] == 1
    assert description["nodes"][standing_id]["source_count"] == 2
    assert payload["min_lead_s"] == 0.2


def test_build_graph_refuses_library_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        build_graph(_manifest_clips(), ["crow", "other"], PAIRS, ORIENTS)


def test_runtime_schedule_is_entry_hold_exit(tmp_path):
    from protomotions.components.contact_graph import ContactGraph

    payload, _ = build_graph(_manifest_clips(), ["crow", "crow_x3s", "handstand"], PAIRS, ORIENTS)
    torch.save(payload, tmp_path / "contact_graph.pt")
    graph = ContactGraph.from_file(tmp_path / "contact_graph.pt")
    graph.validate_against_motion_lib(["/x/crow.motion", "/x/crow_x3s.motion", "/x/handstand.motion"])
    crow_id = graph.node_id_for_key(node_key("Crow", ["L_HAND:G", "R_HAND:G"], "prone"))
    standing_id = graph.node_id_for_key(node_key("standing", ["L_FOOT:G", "R_FOOT:G"], "upright"))
    mid = torch.tensor([1])  # crow_x3s: standing 0-1, Crow 3-11 (hold 3.5), standing 13-14

    def at(t):
        ids, valid = graph.next_goal_indices(mid, torch.tensor([t]), 2, include_current=True)
        g = graph.gather(mid, ids)
        return [int(g["node"][0, k]) for k in range(2)], [bool(valid[0, k]) for k in range(2)], g

    # Inside the standing rest: slot 0 is the rest itself, slot 1 the crow.
    nodes, valid, g = at(0.5)
    assert nodes == [standing_id, crow_id] and valid == [True, True]
    assert float(g["t_end"][0, 0]) == 1.0
    # In the entry gap: the crow with a countdown to its exemplar.
    nodes, valid, g = at(2.0)
    assert nodes[0] == crow_id and float(g["t_hold"][0, 0]) - 2.0 == pytest.approx(1.5)
    # Inside the extended hold: still the crow, dwell remaining = t_end - now.
    nodes, valid, g = at(7.0)
    assert nodes == [crow_id, standing_id]
    assert float(g["t_end"][0, 0]) - 7.0 == pytest.approx(4.0)
    # In the exit: the final standing hold; nothing after it.
    nodes, valid, _ = at(12.0)
    assert nodes[0] == standing_id and valid == [True, False]


# --------------------------------------------------------------------------- #
# mlp_goal_conditioned experiment contract
# --------------------------------------------------------------------------- #
def test_goal_conditioned_expert_config_is_asymmetric(tmp_path):
    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.robot_configs.factory import robot_config

    payload, _ = build_graph(_manifest_clips(), ["crow", "crow_x3s", "handstand"], PAIRS, ORIENTS)
    torch.save(payload, tmp_path / "contact_graph.pt")

    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)
    args = parser.parse_args(["--hold-graph-file", str(tmp_path / "contact_graph.pt")])
    args.motion_file = "unused.pt"
    args.scenes_file = None
    args.batch_size = 32
    args.training_max_steps = 256

    robot_cfg = robot_config("smpl_yogi")
    exp.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    assert len(robot_cfg.trackable_bodies_subset) == 24
    assert robot_cfg.contact_pair_bodies == robot_cfg.kinematic_info.body_names

    env_cfg = exp.env_config(robot_cfg, args)
    agent_cfg = exp.agent_config(robot_cfg, env_cfg, args)

    assert agent_cfg.model.actor.in_keys == exp.ACTOR_KEYS
    assert "mimic_target_poses" not in agent_cfg.model.actor.in_keys
    assert "mimic_target_poses" not in agent_cfg.model.actor.mu_model.in_keys
    assert agent_cfg.model.critic.in_keys == exp.ACTOR_KEYS + ["mimic_target_poses"]
    assert set(agent_cfg.model.in_keys) == set(agent_cfg.model.critic.in_keys)
    assert set(env_cfg.observation_components) == set(agent_cfg.model.in_keys)

    ctrl = env_cfg.control_components["contact_graph"]
    assert ctrl.num_goal_steps == 2 and ctrl.num_masked_future_steps == 2
    assert ctrl.future_steps == [1, 5, 10, 15]
    assert ctrl.dwell_channels and ctrl.include_current_segment
    assert ctrl.pose_visible_prob == 1.0 and ctrl.contact_visible_prob == 1.0
    assert ctrl.full_pose_prob == 1.0
    assert env_cfg.termination_components["tracking_error"].static_params["threshold"] == 0.5
    weights = {k: v.static_params["weight"] for k, v in env_cfg.reward_components.items()}
    assert weights["gt_rew"] == 0.5 and weights["diag_goal_pose_error"] == 0.0
    assert "contact_match_rew" not in weights
    assert env_cfg.motion_manager.graph_file == str(tmp_path / "contact_graph.pt")
    # The goal-pose kernel is bound to all 24 bodies.
    goal = env_cfg.observation_components["masked_mimic_target_poses"]
    assert goal.static_params["conditionable_body_ids"].numel() == 24


def test_goal_conditioned_expert_subset_bodies_and_ground_only(tmp_path):
    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.robot_configs.factory import robot_config

    payload, _ = build_graph(_manifest_clips(), ["crow", "crow_x3s", "handstand"], PAIRS, ORIENTS)
    torch.save(payload, tmp_path / "contact_graph.pt")
    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)
    args = parser.parse_args(
        [
            "--hold-graph-file", str(tmp_path / "contact_graph.pt"),
            "--goal-bodies", "subset",
            "--sense-body-pair-contacts", "false",
        ]
    )
    args.motion_file = "unused.pt"
    args.scenes_file = None
    args.batch_size = 32
    args.training_max_steps = 256
    robot_cfg = robot_config("smpl_yogi")
    exp.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    assert len(robot_cfg.trackable_bodies_subset) == 6
    assert robot_cfg.contact_pair_bodies is None
    env_cfg = exp.env_config(robot_cfg, args)
    goal = env_cfg.observation_components["masked_mimic_target_poses"]
    assert goal.static_params["conditionable_body_ids"].numel() == 6


# --------------------------------------------------------------------------- #
# Probe plans + the in-training panel on the PPO expert
# --------------------------------------------------------------------------- #
def test_family_plans_fork_and_dwell_from_the_graph_description():
    _, description = build_graph(_manifest_clips(), ["crow", "crow_x3s", "handstand"], PAIRS, ORIENTS)
    plans = family_plans(description, hold_s=4.0, dwell_seconds=(3.0, 6.0))
    assert set(plans) == {
        "fork_Crow", "dwell_Crow_3s", "dwell_Crow_6s",
        "fork_Handstand", "dwell_Handstand_3s", "dwell_Handstand_6s",
    }
    fork = plans["fork_Crow"]
    # Poses come from the unextended clip, nodes are named by key, the entry
    # deadline is the reference's own (3.0 s hold start - 1.0 s rest end).
    assert fork["start"] == {"clip": "crow", "time": 0.0}
    assert fork["goals"][0]["pose_clip"] == "crow"
    assert fork["goals"][0]["config"] == node_key("Crow", ["L_HAND:G", "R_HAND:G"], "prone")
    assert fork["goals"][0]["reach_s"] == pytest.approx(2.0)
    assert fork["goals"][0]["hold_s"] == 4.0
    assert fork["goals"][1]["config"] == node_key("standing", ["L_FOOT:G", "R_FOOT:G"], "upright")
    assert fork["goals"][1]["reach_s"] == pytest.approx(2.0)  # 10.0 - 8.0
    dwell = plans["dwell_Crow_6s"]
    assert dwell["start"] == {"clip": "crow", "time": 3.5}
    assert dwell["goals"][0]["reach_s"] == 0.5 and dwell["goals"][0]["hold_s"] == 6.0


def _expert_args(exp, tmp_path, extra=()):
    payload, _ = build_graph(_manifest_clips(), ["crow", "crow_x3s", "handstand"], PAIRS, ORIENTS)
    torch.save(payload, tmp_path / "contact_graph.pt")
    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)
    args = parser.parse_args(["--hold-graph-file", str(tmp_path / "contact_graph.pt"), *extra])
    args.motion_file = "unused.pt"
    args.scenes_file = None
    args.batch_size = 32
    args.training_max_steps = 256
    return args


def test_goal_conditioned_expert_carries_the_sequence_viz_panel(tmp_path):
    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.robot_configs.factory import robot_config

    args = _expert_args(exp, tmp_path)
    robot_cfg = robot_config("smpl_yogi")
    exp.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    env_cfg = exp.env_config(robot_cfg, args)
    agent_cfg = exp.agent_config(robot_cfg, env_cfg, args)
    viz = agent_cfg.sequence_viz
    assert viz is not None and viz.viz_every == 500
    assert viz.plan_files == exp.DEFAULT_VIZ_PLANS
    assert all(Path(p).is_file() for p in viz.plan_files)
    assert viz.num_sequences == 12 and viz.log_scalars is False

    off = _expert_args(exp, tmp_path, extra=["--viz-sequences-every", "0"])
    assert exp.agent_config(robot_cfg, env_cfg, off).sequence_viz is None


def test_ppo_model_forward_inference_runs_the_actor_only():
    from tensordict import TensorDict
    from protomotions.agents.ppo.model import PPOModel

    calls = []

    class _Actor:
        def __call__(self, td, log_internals=False):
            calls.append("actor")
            td["action"] = torch.ones(2, 3)
            td["mean_action"] = torch.zeros(2, 3)
            return td

    class _Critic:
        def __call__(self, td):
            raise AssertionError("the critic must not run at inference")

    model = object.__new__(PPOModel)
    model._actor = _Actor()
    model._critic = _Critic()
    out = PPOModel.forward_inference(model, TensorDict({"obs": torch.zeros(2, 4)}, batch_size=[2]))
    assert calls == ["actor"]
    assert torch.equal(out["mean_action"], torch.zeros(2, 3))


def test_sequence_viz_trigger_is_agent_agnostic(monkeypatch):
    """The panel hook lives on BaseAgent, so a PPO agent fires it too."""
    from protomotions.agents.ppo.agent import PPO
    import protomotions.agents.evaluators.sequence_viz as viz_module

    ran = []

    class _Runner:
        def __init__(self, agent, config):
            pass

        def run(self, epoch):
            ran.append(epoch)
            return {}, {}

    monkeypatch.setattr(viz_module, "SequenceVizRunner", _Runner)
    agent = object.__new__(PPO)
    agent.config = SimpleNamespace(sequence_viz=SimpleNamespace(viz_every=500))
    agent.current_epoch = 1000
    agent.fabric = SimpleNamespace(global_rank=0, loggers=[])
    agent._skip_next_policy_update = False
    PPO._maybe_run_sequence_viz(agent)
    assert ran == [1000] and agent._skip_next_policy_update
    # No config at all (any agent whose experiment never sets it) is a no-op.
    plain = object.__new__(PPO)
    plain.config = SimpleNamespace()
    plain.current_epoch = 1000
    plain.fabric = SimpleNamespace(global_rank=0, loggers=[])
    plain._skip_next_policy_update = False
    PPO._maybe_run_sequence_viz(plain)
    assert ran == [1000] and not plain._skip_next_policy_update
