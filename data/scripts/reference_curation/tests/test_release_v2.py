# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BodyFix Step 5 (BUILD_PLAN Steps 9-10): the training release on plant v2.

Pure rules on synthetic inputs (hold extension v2, graph v2's ids and consensus, the sidecar's known-free rule,
tables v2's column rules), then the committed release: its record, its identity checks (TODO C5), the numbers the
build measured, the side-specific manual goals graph v2 exists for, the swing re-check (TODO C3) and negative
controls that must fail loudly.

    OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest \\
        data/scripts/reference_curation/tests/test_release_v2.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from build_hold_graph import build_graph as build_graph_v1
from build_hold_graph_v2 import build_graph_v2
from build_physics_tables_v2 import swing_rules
from extract_contact_configs import ORIENT_BINS, ZONE_ORDER, mjcf_body_names, zone_pairs
from reference_curation import contact_targets_v2 as ct, hold_extension_v2 as hx, ids, release_v2 as R

# The committed release (BodyFix.MD "What Step 5 found"); never "the newest folder".
RELEASE_ID = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
SIDE_CROW_C = "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c"


# --------------------------------------------------------------------------- #
# Hold extension v2
# --------------------------------------------------------------------------- #
def _motion(T=12, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    rot = torch.zeros(T, B, 4)
    rot[..., 3] = 1.0
    return {
        "fps": 60, "plant_sha256": "x",
        "rigid_body_pos": torch.randn(T, B, 3, generator=g), "rigid_body_rot": rot.clone(),
        "local_rigid_body_rot": rot.clone(), "dof_pos": torch.randn(T, 6, generator=g),
        "dof_vel": torch.zeros(T, 6), "rigid_body_vel": torch.zeros(T, B, 3), "rigid_body_ang_vel": torch.zeros(T, B, 3),
        "rigid_body_contacts": torch.rand(T, B, generator=g) > 0.5,
        "ground_reaction": torch.rand(T, 3, generator=g) + 1.0,
        "rigid_body_ground_forces": torch.rand(T, B, 3, generator=g) + 1.0,
        "ground_reaction_valid": torch.rand(T, 3, generator=g) * 0.5 + 0.5,
    }


def test_hold_extension_v2_keeps_real_frame_pressure_and_masks_inserted_frames():
    m = _motion()
    plan = [(4, 3)]
    out, index, inserted = hx.extend_motion_v2(m, plan)
    assert index.tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11]
    assert inserted.nonzero().flatten().tolist() == [5, 6, 7]
    for k in ("rigid_body_pos", "dof_pos", "rigid_body_contacts"):
        assert torch.equal(out[k], m[k][index])
    real = ~inserted
    for k in hx.PRESSURE_FIELDS:
        assert torch.equal(out[k][real], m[k][index][real])
        assert out[k][inserted].abs().sum() == 0               # zero, with every validity column 0
    assert out["ground_reaction_valid"][4].min() > 0           # the exemplar itself is a real frame
    partial = {k: v for k, v in m.items() if k != "ground_reaction"}
    with pytest.raises(ValueError, match="only"):
        hx.extend_motion_v2(partial, plan)
    same, idx0, ins0 = hx.extend_motion_v2(m, [])
    assert not ins0.any() and all(torch.equal(same[k], m[k]) for k in m if torch.is_tensor(m[k]))
    assert hx.variant_stem("a", 0.0) == "a" and hx.variant_stem("a", 7.0) == "a_x7s"


# --------------------------------------------------------------------------- #
# Graph v2
# --------------------------------------------------------------------------- #
def _hold(hid, t_hold, pairs, name="crow", orient="prone", t0=None, t1=None):
    t0 = t_hold - 0.5 if t0 is None else t0
    t1 = t_hold + 0.5 if t1 is None else t1
    return {"name": name, "hold_id": hid, "t_start": t0, "t_hold": t_hold, "t_end": t1, "pairs": pairs,
            "orientation": orient, "frame_start": int(t0 * 60), "frame_hold": int(t_hold * 60), "frame_end": int(t1 * 60)}


def _clips():
    """Side Crow -c's shape: one clip, two crow holds, the left shelf in one and the right in the other (each in
    half the node's segments, so v1's 0.5 vote merged them); and a second clip with another node. Each clip in two
    variants."""
    left = ["L_HAND:G", "R_HAND:G", "L_SHANK+L_UPPER_ARM"]
    right = ["L_HAND:G", "R_HAND:G", "R_SHANK+R_UPPER_ARM"]
    out = []
    for stem, variant in (("c", 0.0), ("c_x3s", 3.0), ("d", 0.0), ("d_x3s", 3.0)):
        src = stem.split("_")[0]
        holds = ([_hold("c@60", 1.0, left), _hold("c@240", 4.0, right)] if src == "c"
                 else [_hold("d@120", 2.0, ["L_HAND:G"], name="another")])
        out.append({"stem": stem, "source_stem": src, "variant_s": variant, "fps": 60, "num_frames": 600, "holds": holds})
    return out


def test_graph_v2_ids_are_stable_and_nodes_take_the_consensus_of_source_holds():
    pairs, orients = list(zone_pairs()), list(ORIENT_BINS)
    clips = _clips()
    names = [c["stem"] for c in clips]
    payload, desc = build_graph_v2(clips, names, pairs, orients, motion_num_frames=[600] * 4)
    shuffled, _ = build_graph_v2(clips, list(reversed(names)), pairs, orients, motion_num_frames=[600] * 4)
    assert payload["node_keys"] == shuffled["node_keys"]                      # ids do not follow the clip order
    assert payload["hold_ids"] == ["c@240", "c@60", "d@120"]
    node = desc["nodes"][payload["node_keys"].index("crow|L_HAND:G|R_HAND:G@prone")]
    assert node["num_source_holds"] == 2 and node["goal_pairs"] == []          # no shelf every hold commands
    assert node["pair_conflicts"] == {"L_SHANK+L_UPPER_ARM": ["c@60"], "R_SHANK+R_UPPER_ARM": ["c@240"]}
    # v1's 0.5 vote over segments: each shelf is in 2 of the 4 crow segments, so a manual goal got both shelves,
    # and v1 numbered nodes by first appearance
    v1, _ = build_graph_v1(clips, names, pairs, orients)
    v1_reversed, _ = build_graph_v1(clips, list(reversed(names)), pairs, orients)
    assert v1["node_keys"] != v1_reversed["node_keys"]
    i = pairs.index
    crow = v1["node_keys"].index(node["key"])
    assert v1["node_contact"][crow, [i("L_SHANK+L_UPPER_ARM"), i("R_SHANK+R_UPPER_ARM")]].tolist() == [1.0, 1.0]
    from protomotions.components.contact_graph import ContactGraph

    graph = ContactGraph(payload)
    n = torch.tensor([payload["node_keys"].index(node["key"])] * 2)
    contact, resolved = graph.manual_contact(n, torch.tensor([0, 0]), torch.tensor([1.0, 4.0]))
    assert resolved.all()
    assert contact[0, i("L_SHANK+L_UPPER_ARM")] == 1 and contact[0, i("R_SHANK+R_UPPER_ARM")] == 0
    assert contact[1, i("L_SHANK+L_UPPER_ARM")] == 0 and contact[1, i("R_SHANK+R_UPPER_ARM")] == 1


def test_graph_v2_refuses_what_would_mislabel_a_goal():
    pairs, orients = list(zone_pairs()), list(ORIENT_BINS)
    clips = _clips()
    names = [c["stem"] for c in clips]
    with pytest.raises(ValueError, match="frame counts"):
        build_graph_v2(clips, names, pairs, orients, motion_num_frames=[600, 600, 600, 599])
    no_id = json.loads(json.dumps(clips))
    del no_id[0]["holds"][0]["hold_id"]
    with pytest.raises(ValueError, match="no hold_id"):
        build_graph_v2(no_id, names, pairs, orients)
    drift = json.loads(json.dumps(clips))
    drift[1]["holds"][0]["pairs"] = ["L_HAND:G", "R_HAND:G"]                  # the x3s variant lost the shelf
    with pytest.raises(ValueError, match="variants disagree"):
        build_graph_v2(drift, names, pairs, orients)


# --------------------------------------------------------------------------- #
# The sidecar's known-free rule and tables v2's column rules
# --------------------------------------------------------------------------- #
def test_known_free_needs_the_human_separated_and_the_zone_unconfigured():
    T, Z = 100, len(ZONE_ORDER)
    state = np.zeros((T, Z), np.int8)
    state[:, ZONE_ORDER.index("L_FOOT")] = 1
    state[:10, ZONE_ORDER.index("HEAD")] = -1                               # 90 % decided: not known
    state[:3, ZONE_ORDER.index("TRUNK")] = -1                               # 97 %: known
    hold = {"frame_start": 0, "frame_end": T - 1, "pairs_configured": ["L_FOOT:G"]}
    free, share = ct.known_free(state, hold)
    assert not free[ZONE_ORDER.index("L_FOOT")] and not free[ZONE_ORDER.index("HEAD")]
    assert free[ZONE_ORDER.index("TRUNK")] and share[ZONE_ORDER.index("TRUNK")] == pytest.approx(0.97)
    assert free.sum() == Z - 2


def test_tables_v2_use_column_1_for_newtons_and_the_attribution_visibility_gate():
    T = 40
    args = Namespace(fps=60, speed_filter=1, lift_window_s=0.05, lift_m=0.01, swing_speed=0.25, moving_speed=0.05,
                     near_floor_m=0.10, unloaded_n=14.0, loaded_n=70.0)
    speed = np.full(T, 0.1)                       # moving, not fast enough for the velocity rule
    zmin = np.full(T, 0.03)
    visible = np.ones(T, bool)
    load = np.full(T, 5.0)
    valid = np.ones((T, 3))
    valid[:10, 1] = 0.0                           # frames 0-9: only the on-mat column 2 is valid
    visible[10:20] = False                        # frames 10-19: the zone is outside the attribution's view
    load[30:] = 100.0                             # frames 30-39: loaded
    r = swing_rules(speed, zmin, visible, load, valid, "L_FOOT", args)
    assert r["unl"].nonzero()[0].tolist() == list(range(20, 30))
    assert r["unl_v1"].nonzero()[0].tolist() == list(range(0, 30))           # v1: column 2 and invisible frames too
    assert r["veto"].nonzero()[0].tolist() == list(range(30, 40))
    valid[30:, 1] = 0.0
    assert not swing_rules(speed, zmin, visible, load, valid, "L_FOOT", args)["veto"].any()  # col 2 cannot veto


# --------------------------------------------------------------------------- #
# The committed release
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def release():
    path = R.RECORD_ROOT / f"{RELEASE_ID}.json"
    if not path.exists() or not (ids.REPO / json.loads(path.read_text())["dir"]).exists():
        pytest.skip("the release is not on disk")
    rec = json.loads(path.read_text())
    return SimpleNamespace(rec=rec, out=ids.REPO / rec["dir"], inp=R.pinned_inputs(rec["gate_id"]))


def test_the_record_names_every_artifact_by_content(release):
    rec = release.rec
    assert rec["kind"] == "reference_release" and rec["release_id"] == RELEASE_ID
    assert rec["plant"]["plant"] == "v2" and rec["robot"] == "smpl_yogi_v2"
    assert rec["gate_id"] == "holds_repaired_ftC_posefix.gate_v2.b2f76f2606"
    assert rec["labels_id"] == "holds_repaired_ftC_posefix.labels_v2.e03994ad3c"
    for role, a in rec["artifacts"].items():
        assert ids.sha256_file(ids.REPO / a["path"]) == a["sha256"], role
    for stem, sha in rec["motions"].items():
        assert ids.sha256_file(release.out / "motions" / f"{stem}.motion") == sha
    assert rec["checks"]["failed"] == [] and len(rec["checks"]["passed"]) == 26


def test_the_release_passes_its_identity_checks(release):
    passed, failed, facts = R.check(release.out, release.inp)
    assert failed == []
    assert len(passed) == 26
    assert facts["graph"]["segments_v1_vote_served_another_set"] == 18


def test_the_numbers_the_build_measured(release):
    c = release.rec["counts"]
    assert (c["clips"], c["holds"], c["extendable_holds"]) == (56, 270, 64)
    assert c["groups"] == {"single_leg": 13, "inversion": 13, "arm_balance": 9, "connective": 21}
    assert (c["motions"], c["frames"], c["inserted_frames"]) == (168, 282936, 38400)
    g = c["graph"]
    assert (g["nodes"], g["edges"], g["segments"], g["hold_ids"]) == (123, 196, 810, 270)
    assert g["nodes_with_pair_conflicts"] == 5
    assert c["tables"] == {"swing_zone_frames": 345854, "lean_gated_segments": 90, "cop_valid_segments": 768,
                           "share_valid_segments": 735}
    s = c["sidecar"]
    assert (s["critical"], s["restored"], s["masked"]) == (59, 4, {"R_HAND:G": 1})
    assert s["roles"]["required_support"]["ground"] == 515 and s["roles"]["required_touch"] == {"ground": 299, "pair": 63}
    assert s["roles"]["forbidden_support"]["ground"] == 16
    assert {k: v for k, v in s["flags"].items() if v} == {"fit_closer": 24, "critical_partly_realised": 1,
                                                           "statue_moved_single_leg": 17, "pose_preparation": 30}
    assert s["pose_roles"] == {"family": 71, "standing": 112, "variant": 49, "preparation": 30, "undecided": 8}
    assert s["known_free_zone_holds"] == 3233 and len(s["free_unknown"]) == 3
    assert s["frame_pair_partly_realised"] == {
        "220923_Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_-a@914:TRUNK+L_HAND": 0.471}
    assert c["plans"] == 148
    assert release.rec["training"] == {**release.rec["training"], "longest_motion_s": 62.967, "eval_max_steps": 1890}


def test_side_crow_c_commands_each_hold_its_own_shelf(release):
    from protomotions.components.contact_graph import ContactGraph

    graph = ContactGraph.from_file(release.out / "contact_graph.pt")
    m = graph.motion_names.index(SIDE_CROW_C)
    holds = {graph.hold_ids[int(graph.seg_hold_index[m, k])]: k for k in range(int(graph.seg_count[m]))}
    k1, k2 = holds[f"{SIDE_CROW_C}@606"], holds[f"{SIDE_CROW_C}@1296"]
    assert int(graph.seg_node[m, k1]) == int(graph.seg_node[m, k2])        # one node
    node = graph.seg_node[m, [k1, k2]]
    contact, resolved = graph.manual_contact(node, torch.tensor([m, m]), graph.seg_hold[m, [k1, k2]])
    i = graph.pair_names.index
    assert resolved.all()
    assert contact[0, i("L_SHANK+L_UPPER_ARM")] == 1 and contact[0, i("R_SHANK+R_UPPER_ARM")] == 0
    assert contact[1, i("L_SHANK+L_UPPER_ARM")] == 0 and contact[1, i("R_SHANK+R_UPPER_ARM")] == 1
    assert graph.node_contact[node[0], [i("L_SHANK+L_UPPER_ARM"), i("R_SHANK+R_UPPER_ARM")]].sum() == 0


def test_the_swing_labels_agree_with_the_human(release):
    cal = release.rec["calibration_c3"]
    a = cal["all_motions"]
    assert (a["labels"], a["pressure_only"], a["pressure_only_v1_rule"]) == (345854, 17649, 28147)
    assert a["pressure_only"] <= cal["baselines_180_motions"]["ftC"]["pressure_only"]   # back below ftC's 18,254
    x = cal["x0_against_human_evidence"]
    assert x["velocity"]["human_separated_share"] >= 0.998
    assert x["pressure"]["mat_over_14n"] == 0 and x["both"]["mat_over_14n"] == 0
    h = cal["human_load_on_labelled_feet_hands_n"]
    assert h["p99"] < 14.0 and h["max"] < 0.1 * 74 * 9.81    # a perfect imitator is never charged to saturation


def test_mismatched_artifacts_fail_loudly(release, tmp_path):
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.envs.control.contact_targets import ContactTargets
    from protomotions.envs.control.physics_terms import PhysicsTables
    from protomotions.utils.release_identity import ReleaseMismatchError, load_release, require_artifact

    out = release.out
    graph = ContactGraph.from_file(out / "contact_graph.pt")
    names = list(graph.motion_names)
    frames = graph.motion_num_frames
    with pytest.raises(ValueError, match="different motion library"):
        graph.validate_against_motion_lib([f"{n}.motion" for n in reversed(names)])
    with pytest.raises(ValueError, match="frame counts"):
        graph.validate_against_motion_lib([f"{n}.motion" for n in names], motion_num_frames=frames + 1)
    body_names = mjcf_body_names(str(R.plant_mjcf()))
    with pytest.raises(ValueError, match="frame counts"):
        PhysicsTables(str(out / "physics_tables.pt"), names, body_names, "cpu", motion_num_frames=frames - 1, fps=60)
    from protomotions.utils.plant_identity import PlantMismatchError, mjcf_path

    with pytest.raises(PlantMismatchError):
        PhysicsTables(str(out / "physics_tables.pt"), names, body_names, "cpu", plant_mjcf=str(mjcf_path("v1")))
    with pytest.raises(ValueError, match="graph sha256"):
        ContactTargets(str(out / "contact_targets.pt"), graph, names, "cpu", graph_sha256="0" * 64)
    rec = load_release(R.RECORD_ROOT / f"{RELEASE_ID}.json")
    with pytest.raises(ReleaseMismatchError):
        require_artifact(rec, "graph", out / "physics_tables.pt")
    edited = yaml.safe_load(open(out / "holds_extended.yaml"))
    edited["clips"][0]["holds"][0]["t_end"] += 0.1
    (tmp_path / "h.yaml").write_text(yaml.safe_dump(edited, sort_keys=False))
    with pytest.raises(ReleaseMismatchError):
        require_artifact(rec, "holds_extended", tmp_path / "h.yaml")


def test_realisability_of_the_reference_is_the_baseline_a_policy_is_compared_with(release):
    from reference_curation import realisability_v2 as rz

    p = rz.release_paths(RELEASE_ID)
    stems = [c["stem"] for c in yaml.safe_load(open(p["ext"]))["clips"]]
    q = rz.evaluate(RELEASE_ID, rz.reference_rollouts(p["motions"], stems))["pooled"]
    assert q["tracked_share"] == 1.0
    assert q["float_share_over_2cm"] == 0.0           # the one floating support (Triangle -b@669's hand) is masked
    assert q["slide_share_over_5cm_s"] == pytest.approx(0.059, abs=0.001)
    assert q["slide_cm_s_p90"] == pytest.approx(2.54, abs=0.01)
    assert q["com_cop_cm_p50"] == pytest.approx(1.44, abs=0.01)
