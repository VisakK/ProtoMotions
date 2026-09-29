# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the source states (capture store v3) and labels v1 (BUILD_PLAN Step 6).

They run on the real clips and pin what the review and the build measured: the crow family's
exemplars move off toe-down frames (README §3.3), Peacock -a is toes-assisted (§3.2), Chaturanga's
``HEAD:G`` and Bridge's hands are not down (§3.1-3.2), and the mesh's wrist-crease "forearm" on
Standing Split -a is a seam artifact. The stores and the audit come from the session fixture
``conftest.stores`` (a temp dir); the labels of ``STEMS`` are built from them. Nothing under
``output/``, ``data/reference_curation/`` or ``REVIEW_ROOT`` is written.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, human_mesh as hm, ids, labels, sources

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
SIDE_CROW_C = "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c"
STANDING_SPLIT = "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a"
PEACOCK = "220923_Peacock_Pose_or_Mayurasana_-a"
CHATURANGA = "220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a"
BRIDGE = "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a"
PLOW_B = "220926_Plow_Pose_or_Halasana_-b"
DOLPHIN = "220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a"
BIG_TOE_C = "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c"  # its MoSh pkl is corrupt
STEMS = (CROW, SIDE_CROW_C, STANDING_SPLIT, PEACOCK, CHATURANGA, BRIDGE, PLOW_B, DOLPHIN, BIG_TOE_C)

pytestmark = pytest.mark.skipif(
    not (ids.MOYO_DATA / "mosh").exists() or not hm.MODEL_PATH.exists() or not ids.SHIPPED_DIR.exists(),
    reason="the MOYO MoSh fits, the SMPL-X model or the shipped ftC clips are not on disk")


def _zi(zone: str) -> int:
    return ZONE_ORDER.index(zone)


@pytest.fixture(scope="module")
def run(stores):
    """Labels of ``STEMS`` over the session's stores (``conftest.stores``)."""
    result = labels.build(stores.aud, stores.stores, stems=list(STEMS))
    assert result["failures"] == []
    by_hold = {}
    for a in result["annotations"]:
        by_hold.setdefault(a["hold_id"], {})[a["contact"]] = a
    return SimpleNamespace(out=stores.out, aud=stores.aud, stores=stores.stores, v3=stores.v3, result=result,
                           holds=result["holds"], anns=by_hold)


# --------------------------------------------------------------------------- #
# Store v3: seam arbitration
# --------------------------------------------------------------------------- #
def test_the_wrist_crease_is_not_a_forearm_contact(run):
    """Standing Split -a at 9.0 s: v2's left forearm "touches" through three wrist-crease vertices 1.9 cm
    up next to a palm pressed into the floor; the wrist joint is 3.5 cm up."""
    rec = run.v3[STANDING_SPLIT]
    f = rec.frame(9.0)
    assert rec["human_ground_state"][f, _zi("L_FOREARM")] == 1 and rec["human_min_z"][f, _zi("L_FOREARM")] < 0.02
    assert rec["human_floor_state"][f, _zi("L_FOREARM")] == 0 and rec["human_floor_z"][f, _zi("L_FOREARM")] > 0.04
    assert (rec["human_floor_state"][:, _zi("L_FOREARM")] == 1).sum() == 0          # v2: 341 frames
    for zone in ("L_HAND", "R_HAND", "L_FOOT", "HEAD", "TRUNK"):                     # never arbitrated
        assert np.array_equal(rec["human_floor_state"][:, _zi(zone)], rec["human_ground_state"][:, _zi(zone)])


def test_arbitration_keeps_real_limb_contacts_and_drops_elbow_seams(run):
    counts = {s: run.v3[s].meta["counts"] for s in (DOLPHIN, PLOW_B)}
    for side in "LR":  # Dolphin's forearms lie flat: >= 94 % of their v2 frames stay
        c = counts[DOLPHIN][f"{side}_FOREARM"]
        assert c["v2_contact"] > 400 and c["v3_contact"] >= 0.94 * c["v2_contact"]
    for side in "LR":  # Plow -b: the upper arms lie on the floor, the forearms rise from the elbow
        assert counts[PLOW_B][f"{side}_FOREARM"]["v3_contact"] < 0.45 * counts[PLOW_B][f"{side}_FOREARM"]["v2_contact"]
        assert counts[PLOW_B][f"{side}_UPPER_ARM"]["v3_contact"] >= 0.9 * counts[PLOW_B][f"{side}_UPPER_ARM"]["v2_contact"]


def test_seam_rule_on_synthetic_heights():
    """A seam goes to the neighbour whose core touches; between two seam-only sides (a kneecap) the
    deeper side keeps it, so the knee contact survives."""
    T, Z = 3, len(ZONE_ORDER)
    skin = np.full((T, Z), 0.5)
    core = np.full((T, Z), 0.5)
    seams = {(z, n): np.full(T, 0.5) for z in ZONE_ORDER for n in sources.NEIGHBOURS[z]}
    known = np.ones((T, Z), dtype=bool)
    ground = {"touch_m": 0.02, "separation_m": 0.03}
    # frame 0: palm on the floor, forearm only through its wrist seam
    core[0, _zi("L_HAND")] = skin[0, _zi("L_HAND")] = 0.0
    seams["L_FOREARM", "L_HAND"][0] = skin[0, _zi("L_FOREARM")] = 0.015
    # frame 1: kneeling on the kneecap, shank and thigh both only through their shared seam
    seams["L_SHANK", "L_THIGH"][1] = -0.004
    seams["L_THIGH", "L_SHANK"][1] = 0.012
    # frame 2: a forearm lying flat
    core[2, _zi("R_FOREARM")] = 0.001
    h = sources.arbitrate(skin, core, seams, known, ground)
    assert h[0, _zi("L_FOREARM")] == 0.5 and h[0, _zi("L_HAND")] == 0.0
    assert h[1, _zi("L_SHANK")] == -0.004 and h[1, _zi("L_THIGH")] == 0.5
    assert h[2, _zi("R_FOREARM")] == 0.001


def test_source_hierarchy_mesh_then_markers_then_mat(run):
    rec = run.v3[CROW]
    state, channel = sources.ground_source(rec)
    assert np.array_equal(state, rec["human_floor_state"])  # a readable fit decides every zone it can
    assert set(np.unique(channel)) == {1}
    toe = run.v3[BIG_TOE_C]  # no readable fit: markers and mesh are unknown, the mat says contact or nothing
    state, channel = sources.ground_source(toe)
    assert (state >= 0).any() and set(np.unique(channel[state >= 0])) == {3} and (state != 0).all()


# --------------------------------------------------------------------------- #
# Tier 1 (notes/Contact_label_consequence.MD §6)
# --------------------------------------------------------------------------- #
def test_tier1_reproduces_the_notes_verdicts():
    hands, feet = {"L_HAND", "R_HAND"}, {"L_FOOT", "R_FOOT"}
    cases = [  # (pair, ground, verdict): the note's six measured cases, the chair's knees, and two more
        (("R_SHANK", "R_UPPER_ARM"), hands, "consequential_internal"),     # crow knee-shoulder, 858 N
        (("R_THIGH", "TRUNK"), hands, "consequential_internal"),           # crow hip-chest, 1167 N
        (("R_THIGH", "L_UPPER_ARM"), hands, "consequential_internal"),     # side crow hip-shoulder
        (("L_SHANK", "R_SHANK"), hands, "internal_brace"),                 # crow knees
        (("L_THIGH", "R_THIGH"), hands, "internal_brace"),                 # side crow hips
        (("R_SHANK", "L_THIGH"), hands, "internal_brace"),                 # side crow, 0 N
        (("L_SHANK", "R_SHANK"), feet, "internal_brace"),                  # chair's knees, 754 N, no support
        (("HEAD", "L_UPPER_ARM"), feet, "internal_brace"),                 # arms by the ears
        (("R_FOOT", "L_THIGH"), {"L_FOOT"}, "consequential_internal"),     # tree: the foot on the thigh
        (("L_SHANK", "R_SHANK"), set(), "incidental"),
        (("R_THIGH", "L_THIGH"), {"L_FOOT"}, "consequential_internal"),    # eagle: the wrapped leg rests
        (("TRUNK", "L_FOREARM"), hands | feet, "consequential_internal"),  # peacock: elbows in the belly
    ]
    assert [labels.load_path_class(p, g) for p, g, _ in cases] == [v for _, _, v in cases]
    d = labels.joint_distances(hands)
    assert d[_zi("R_SHANK")] == 6 and labels.joint_distances(hands, [("R_SHANK", "R_UPPER_ARM")])[_zi("R_SHANK")] == 2
    # the side-crow shin on the other thigh is one joint shorter, but only through the pelvis
    assert labels.joint_distances(hands, [("R_SHANK", "L_THIGH")])[_zi("R_SHANK")] == 5
    assert labels.common_root("R_SHANK", "L_THIGH") == "PELVIS" and labels.common_root("TRUNK", "L_HAND") == "TRUNK"


# --------------------------------------------------------------------------- #
# The card's exit tests
# --------------------------------------------------------------------------- #
def test_labels_have_no_contradictory_roles_and_cite_their_evidence(run):
    assert labels.check(run.result, run.stores) == []
    ev = run.result["evidence"]
    for a in run.result["annotations"]:
        assert a["evidence_ids"] and all(i in ev for i in a["evidence_ids"])
        assert f"manifest:{run.aud.manifest.stem}" in a["evidence_ids"]
        if a["source_channel"] == "capture_fit_mesh":
            assert f"capture:v3:{a['stem']}" in a["evidence_ids"] or a["kind"] == "pair"
    kinds = {e["kind"] for e in ev.values()}
    assert {"source_manifest", "audit", "rule", "capture_store", "shipped_motion"} <= kinds
    for e in ev.values():
        if e["kind"] == "capture_store":
            assert ids.sha256_file(ids.REPO / e["path"] if not e["path"].startswith("/") else e["path"]) == e["sha256"]


def test_body_ground_disagreement_with_the_capture_is_zero(run):
    for hid, h in run.holds.items():
        state, _ = sources.ground_source(run.v3[h["hold_id"].rsplit("@", 1)[0]])
        ground = {p[:-2] for p in h["pairs_ground"]}
        for t in range(h["frame_start"], h["frame_end"] + 1):
            decided = {z for z in ZONE_ORDER if state[t, _zi(z)] >= 0}
            assert {z for z in decided if state[t, _zi(z)] == 1} == ground & decided, (hid, t)


def test_the_crow_family_exemplars_move_to_capture_stable_frames(run):
    """README §3.3: the label starts before the toe lifts, and two exemplars sit on toe-down frames."""
    expected = {  # hold: (status, window, exemplar); the mesh lifts the toe at 8.15, 7.87 and 21.97 s
        f"{CROW}@439": ("moved", (489, 719), 651),
        f"{SIDE_CROW_C}@606": ("kept", (472, 808), 606),
        f"{SIDE_CROW_C}@1296": ("moved", (1318, 1385), 1349),
    }
    for hid, (status, (s, e), ex) in expected.items():
        h = run.holds[hid]
        assert (h["labels"]["status"], h["frame_start"], h["frame_end"], h["frame_hold"]) == (status, s, e, ex), hid
        assert h["pairs_ground"] == ["L_HAND:G", "R_HAND:G"]
        state, _ = sources.ground_source(run.v3[hid.rsplit("@", 1)[0]])
        near = round(audit.STABLE_S * 60)
        window = state[ex - near:ex + near + 1]
        assert (labels.matching(window, {"L_HAND", "R_HAND"})).all() and (window[:, _zi("L_FOOT")] == 0).all()
    assert run.holds[f"{CROW}@439"]["t_hold"] == 10.85 and run.holds[f"{CROW}@439"]["labels"]["source"]["frame_hold"] == 439


def test_the_two_side_crow_c_holds_get_distinct_stable_ids(run):
    family = [h for h in run.holds.values() if h["hold_id"].startswith(SIDE_CROW_C) and h.get("extend")]
    assert sorted(h["hold_id"] for h in family) == [f"{SIDE_CROW_C}@1296", f"{SIDE_CROW_C}@606"]
    assert {h["name"] for h in family} == {"Side_Crane_Crow_Pose_or_Parsva_Bakasana"}  # same name, ground set
    assert family[0]["pairs_ground"] == family[1]["pairs_ground"]
    moved = run.holds[f"{SIDE_CROW_C}@1296"]
    assert moved["frame_hold"] == 1349 and moved["hold_id"].endswith("@1296")  # the id does not follow the exemplar
    shelves = {hid: {p for p in run.holds[hid]["pairs"] if "UPPER_ARM" in p and "SHANK" in p}
               for hid in (f"{SIDE_CROW_C}@606", f"{SIDE_CROW_C}@1296")}
    assert shelves == {f"{SIDE_CROW_C}@606": {"L_SHANK+L_UPPER_ARM"}, f"{SIDE_CROW_C}@1296": {"R_SHANK+R_UPPER_ARM"}}
    counts = {}
    for c in run.result["clips"]:
        for h in c["holds"]:
            counts[h["hold_id"]] = counts.get(h["hold_id"], 0) + 1
    assert max(counts.values()) == 1


# --------------------------------------------------------------------------- #
# What the review found, as labels
# --------------------------------------------------------------------------- #
def test_the_reviews_label_errors_are_corrected(run):
    peacock = run.holds[f"{PEACOCK}@619"]  # README §3.2: toes-assisted, not hands-only
    assert peacock["labels"]["status"] == "relabelled" and peacock["frame_hold"] == 619
    assert peacock["pairs_ground"] == ["L_FOOT:G", "R_FOOT:G", "L_HAND:G", "R_HAND:G"]
    head = run.anns[f"{CHATURANGA}@485"]["HEAD:G"]  # README §3.1: HEAD:G at 8.7 cm is a wrong label
    assert (head["label_action"], head["target_role"], head["source_state"]) == \
        ("removed", "forbidden_support", "observed_separation")
    for hand in ("L_HAND:G", "R_HAND:G"):  # README §3.2: Bridge -a's hands are lifted
        assert run.anns[f"{BRIDGE}@848"][hand]["label_action"] == "removed"
    plow = run.holds[f"{PLOW_B}@980"]  # README §3.2: Plow's planted feet are missing from the label
    assert {"L_FOOT:G", "R_FOOT:G"} <= set(plow["pairs_ground"])
    assert not any("FOREARM:G" in p for p in plow["pairs_ground"])  # the elbow seam is not a forearm contact
    split = run.holds[f"{STANDING_SPLIT}@540"]
    assert split["labels"]["status"] == "kept" and split["pairs_ground"] == ["L_FOOT:G", "L_HAND:G", "R_HAND:G"]


def test_crow_pairs_roles_and_the_knee_straddle(run):
    a = run.anns[f"{CROW}@439"]
    for pair in ("L_SHANK+L_UPPER_ARM", "R_SHANK+R_UPPER_ARM", "L_THIGH+L_UPPER_ARM", "R_THIGH+R_UPPER_ARM"):
        assert (a[pair]["target_role"], a[pair]["load_path_class"], a[pair]["in_configuration"]) == \
            ("required_touch", "consequential_internal", True), pair
    assert a["R_FOOT+L_SHANK"]["label_action"] == "removed" and a["R_FOOT+L_SHANK"]["reasons"] == ["proximity_latch"]
    assert a["L_FOOT+R_FOOT"]["label_action"] == "demoted" and a["L_FOOT+R_FOOT"]["target_role"] == "allowed"
    assert a["L_FOOT+R_FOOT"]["load_path_class"] == "internal_brace"
    # README §2: the knee straddles the shank and thigh colliders against the upper arm -> any_of
    assert a["L_SHANK+L_UPPER_ARM"]["alternative_group"] == a["L_THIGH+L_UPPER_ARM"]["alternative_group"] is not None
    assert a["L_SHANK+L_UPPER_ARM"]["alternative_group"] != a["R_SHANK+R_UPPER_ARM"]["alternative_group"]


def test_every_change_to_the_configuration_is_listed(run):
    """A hold's ``labels.changes`` names every contact that entered or left ``pairs``, demotions included."""
    for h in run.holds.values():
        before, after = set(h["labels"]["source"]["pairs"]), set(h["pairs"])
        listed = {x.split(":", 1)[1] for x in h["labels"]["changes"] if x.split(":", 1)[0] in ("added", "removed", "demoted")}
        assert before ^ after == listed, h["hold_id"]
    assert "demoted:L_FOOT+R_FOOT" in run.holds[f"{CROW}@439"]["labels"]["changes"]


def test_a_clip_without_capture_carries_its_labels(run):
    hold = run.holds[f"{BIG_TOE_C}@836"]
    assert hold["labels"]["status"] == "kept" and hold["labels"]["unverified"] == ["R_FOOT+L_FOREARM", "R_FOOT+L_HAND"]
    assert set(hold["pairs"]) >= {"R_FOOT+L_FOREARM", "R_FOOT+L_HAND"}
    foot = run.anns[f"{BIG_TOE_C}@836"]["L_FOOT:G"]
    assert foot["source_channel"] == "mat_attributed" and foot["source_state"] == "observed_contact"


def test_moved_exemplars_recompute_the_proposers_fields(run):
    """The recomputed orientation bin equals the manifest's at every exemplar that did not move and is
    not on a pose-repaired clip; a moved one gets fresh values."""
    src = {(c["stem"], i): h for c in ids.load_manifest()["clips"] for i, h in enumerate(c["holds"])}
    repaired = {c["stem"] for c in ids.load_manifest()["clips"] if c.get("pose_repair")}
    for c in run.result["clips"]:
        clip = labels.clip_evidence(c["stem"], ids.load_manifest(), run.stores)
        for i, h in enumerate(c["holds"]):
            fresh = labels.exemplar_fields(clip, h["frame_hold"])
            if h["frame_hold"] == src[c["stem"], i]["frame_hold"] and c["stem"] not in repaired:
                assert fresh["orientation"] == h["orientation"], h["hold_id"]
            elif h["frame_hold"] != src[c["stem"], i]["frame_hold"]:
                assert {k: h[k] for k in fresh} == fresh


def test_labels_are_deterministic_and_write_a_readable_manifest(run, tmp_path):
    again = labels.build(run.aud, run.stores, stems=list(STEMS))
    assert labels.labels_id(again) == labels.labels_id(run.result)
    a, b = labels.write(run.result, tmp_path / "a"), labels.write(again, tmp_path / "b")
    for name in ("annotations.jsonl", "holds.yaml", "evidence.jsonl", "summary.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    manifest = ids.load_manifest(a / "holds.yaml")
    source = ids.load_manifest(run.aud.manifest)
    assert [k for k in source if k != "clips"] == [k for k in manifest if k not in ("clips", "labels")]
    src_holds = {c["stem"]: c["holds"] for c in source["clips"]}
    for c in manifest["clips"]:  # every source field of every hold survives, plus hold_id and labels
        for h, s in zip(c["holds"], src_holds[c["stem"]], strict=True):
            assert set(h) == set(s) | {"hold_id", "labels"} and h["name"] == s["name"]
    lines = [json.loads(x) for x in (a / "annotations.jsonl").read_text().splitlines()]
    assert len(lines) == len(run.result["annotations"]) and {x["labels_id"] for x in lines} == {a.name}


# --------------------------------------------------------------------------- #
# Reviews
# --------------------------------------------------------------------------- #
def _claims(hold_id, roles, floor=None, frame=439):
    return {"verdict_id": "ledger:fake.B.1", "packet_id": "fake", "frame": frame, "candidates": list(roles),
            "roles": roles, "unanswered": [], "floor": floor or {}, "errors": [],
            "variant": {"matches_label": "yes", "note": "", "candidates": []},
            "timing": {"start": "capture", "end": "label", "main_moment": "move", "better_frame": 651},
            "repair": {"needed": "no", "kinds": []}, "path": "fake", "reviewer": {}}


def test_reviewer_roles_attach_and_decide_only_when_admitted(run):
    clip = labels.clip_evidence(CROW, ids.load_manifest(), run.stores)
    rec = run.aud.records[f"{CROW}@439"]
    entry = next(c for c in ids.load_manifest()["clips"] if c["stem"] == CROW)
    ev_ids = {k: f"{k}:x" for k in ("manifest", "audit", "rule", "v1", "v2", "v3", "motion")}
    roles = {"L_FOOT+R_FOOT": "required_touch", "R_FOOT+L_SHANK": "required_touch", "L_SHANK+L_UPPER_ARM": "allowed"}
    floor = {"L_HAND": {"human": 1, "avatar": 1}, "R_HAND": {"human": 0, "avatar": 1}}  # the hands touch at 439
    claims = _claims(rec["hold_id"], roles, floor=floor)
    hold, anns = labels.reconcile(clip, entry, rec["hold_index"], rec, ev_ids, claims)
    a = {x["contact"]: x for x in anns}
    assert a["R_FOOT+L_SHANK"]["review"]["conflict"] == "required_but_apart"
    assert a["L_SHANK+L_UPPER_ARM"]["review"] == {"verdict": "ledger:fake.B.1", "role": "allowed", "agrees": False,
                                                  "conflict": "role_differs"}
    assert a["L_SHANK+L_UPPER_ARM"]["target_role"] == "required_touch"          # advisory: nothing changes
    assert "ledger:fake.B.1" in a["L_SHANK+L_UPPER_ARM"]["evidence_ids"]
    assert a["L_HAND:G"]["review"]["agrees_with_source"] is True and a["R_HAND:G"]["review"]["agrees_with_source"] is False
    assert a["R_HAND:G"]["status"] == "resolved"                                 # floor_human is not admitted
    assert hold["labels"]["review"]["timing"]["better_frame_minus_exemplar_s"] == 0.0
    # an admitted class that contradicts the source at the corrected exemplar makes the zone unresolved
    at_exemplar = _claims(rec["hold_id"], roles, floor=floor, frame=651)
    _, conflicted = labels.reconcile(clip, entry, rec["hold_index"], rec, ev_ids, at_exemplar, admitted=("floor_human",))
    c = {x["contact"]: x for x in conflicted}
    assert c["R_HAND:G"]["status"] == "unresolved" and "reviewer_conflict" in c["R_HAND:G"]["reasons"]
    assert c["R_HAND:G"]["in_configuration"] and c["L_HAND:G"]["status"] == "resolved"  # the capture still decides
    _, admitted = labels.reconcile(clip, entry, rec["hold_index"], rec, ev_ids, claims, admitted=("roles",))
    b = {x["contact"]: x for x in admitted}
    assert b["L_FOOT+R_FOOT"]["target_role"] == "required_touch" and b["L_FOOT+R_FOOT"]["in_configuration"]
    assert b["R_FOOT+L_SHANK"]["target_role"] == "unresolved"                   # a touch the human did not make
    assert b["L_SHANK+L_UPPER_ARM"]["target_role"] == "allowed" and not b["L_SHANK+L_UPPER_ARM"]["in_configuration"]


def test_pair_role_rules():
    assert labels.pair_role(1, True, True, "consequential_internal", False) == "required_touch"
    assert labels.pair_role(1, True, False, "consequential_internal", False) == "allowed"
    assert labels.pair_role(1, True, True, "internal_brace", False) == "allowed"
    assert labels.pair_role(1, True, False, "internal_brace", False) == "incidental"
    assert labels.pair_role(1, False, True, "consequential_internal", False) == "unresolved"
    assert labels.pair_role(1, True, True, "consequential_internal", True) == "unresolved"
    assert labels.pair_role(0, True, True, "consequential_internal", False) == "unspecified"
    assert labels.pair_role(-1, False, True, "internal_brace", False) == "unspecified"
    assert labels.pair_role(0, True, True, "internal_brace", False, admitted_role="required_touch") == "unresolved"
    assert labels.pair_role(1, True, False, "internal_brace", False, admitted_role="required_touch") == "required_touch"
