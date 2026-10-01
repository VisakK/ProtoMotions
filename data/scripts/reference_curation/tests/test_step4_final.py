# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BodyFix Step 4's acceptance on the committed records: labels v2 (no body-ground disagreement with the capture,
every evidence id verifies by sha256), every new calibration table and B6 rebuilt from the ledger (re-scored, never
re-queried), and gate v2 (every exclusion has a reason; TODO A4: the commanded ground set is labels v2's configured
supports minus the gate's masks).

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_step4_final.py -q
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest
import yaml

from reference_curation import b6  # first: render (OSMesa)
from reference_curation import capture_v4, edits_v2, gate_v2, ids, labels as L1, labels_v2, packets, packets_v4
from reference_curation import sources, verdicts
from extract_contact_configs import ZONE_ORDER

ZI = {z: i for i, z in enumerate(ZONE_ORDER)}
# The committed ids (BodyFix.MD, "What Step 4 found"); never "the newest folder". The draft is B6's base; pass 1 is
# B6's critical-only configuration (statics v2 ran on it to find what the plant cannot hold without the demoted
# pairs); pass 2 adds the statics restoration (statics v2 ran on it); the final cites pass 2's statics and has pass 2's
# configuration exactly.
LABELS_V2 = ids.DATA_ROOT / "labels" / "holds_repaired_ftC_posefix.labels_v2.e03994ad3c"
LABELS_V2_PASS2 = ids.DATA_ROOT / "labels" / "holds_repaired_ftC_posefix.labels_v2.46024365fa"
LABELS_V2_PASS1 = ids.DATA_ROOT / "labels" / "holds_repaired_ftC_posefix.labels_v2.b61fad7e9d"
B6_DRAFT = ids.DATA_ROOT / "labels" / "holds_repaired_ftC_posefix.labels_v2.74a61979cf"
STATICS_DIR = ids.DATA_ROOT / "statics_v2"
STATICS_CRITICAL_ONLY = STATICS_DIR / "holds_repaired_ftC_posefix.labels_v2.b61fad7e9d.statics_v2.fb432fa15d"
STATICS_V11 = STATICS_DIR / "holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828.statics_v2.f57d85d3fa"


def _final_labels() -> Path:
    if not LABELS_V2.exists():
        pytest.skip("labels v2 is not on disk")
    return LABELS_V2


def _b6_path() -> Path:
    found = sorted(b6.B6_DIR.glob("*/b6.json"))
    if len(found) != 1:
        pytest.skip("no B6 record")
    return found[0]


def _b6_record() -> dict:
    return json.loads(_b6_path().read_text())


# --------------------------------------------------------------------------- #
# Labels v2
# --------------------------------------------------------------------------- #
def test_every_evidence_id_of_labels_v2_verifies_by_sha256():
    d = _final_labels()
    ev = {json.loads(l)["id"]: json.loads(l) for l in open(d / "evidence.jsonl")}
    assert labels_v2.verify_evidence(ev) == []
    for line in open(d / "annotations.jsonl"):
        a = json.loads(line)
        assert a["evidence_ids"] and all(i in ev for i in a["evidence_ids"]), (a["hold_id"], a["contact"])


def test_labels_v2_ground_sets_never_disagree_with_the_capture():
    d = _final_labels()
    m = ids.load_manifest(d / "holds.yaml")
    for c in m["clips"]:
        state, _ = sources.ground_source(capture_v4.load(c["stem"], rebuild=False))
        for h in c["holds"]:
            ground = {p[:-2] for p in h["pairs_ground"]}
            s, e, ex = h["frame_start"], h["frame_end"], h["frame_hold"]
            assert all(state[ex, ZI[z]] < 0 or (state[ex, ZI[z]] == 1) == (z in ground) for z in ZONE_ORDER), h["hold_id"]
            assert L1.matching(state[s:e + 1], ground).all(), h["hold_id"]


def test_the_final_configuration_is_the_one_statics_v2_ran_on():
    a, b = (ids.load_manifest(d / "holds.yaml") for d in (LABELS_V2_PASS2, _final_labels()))
    ha = {h["hold_id"]: h for c in a["clips"] for h in c["holds"]}
    hb = {h["hold_id"]: h for c in b["clips"] for h in c["holds"]}
    assert ha.keys() == hb.keys()
    for k in ha:
        assert (ha[k]["pairs"], ha[k]["frame_start"], ha[k]["frame_end"]) == (hb[k]["pairs"], hb[k]["frame_start"],
                                                                               hb[k]["frame_end"]), k
    assert b["labels"]["statics_id"].startswith(LABELS_V2_PASS2.name)


def test_the_statics_restoration_replays_and_is_the_only_change_to_b6s_configuration():
    """Re-derived from the two statics records: B6's critical-only configuration leaves Peacock -a@788 beyond the
    plant (its forearm braces were split TRUNK / PELVIS between the reviewer's samples, so none reproduced); labels
    v1.1's configuration holds it; the four braces come back, not as critical pairs."""
    from reference_curation import labels_v2_restore as R

    crit = json.loads(_b6_path().read_text())["critical"]
    restored = R.restorations(labels_v2.read_v11(), crit, STATICS_CRITICAL_ONLY, STATICS_V11)
    rec = json.loads((_final_labels() / "restored.json").read_text())
    assert {h: sorted(ps) for h, ps in restored.items()} == {h: r["pairs"] for h, r in rec["holds"].items()}
    assert sorted(restored) == ["220923_Peacock_Pose_or_Mayurasana_-a@788"]
    (h,) = restored
    assert rec["holds"][h]["critical_only"]["verdict"] == "beyond_plant" and rec["holds"][h]["statics_now"]["verdict"] == "feasible"
    p1, fin = (ids.load_manifest(d / "holds.yaml") for d in (LABELS_V2_PASS1, _final_labels()))
    c1 = {x["hold_id"]: set(x["pairs"]) for c in p1["clips"] for x in c["holds"]}
    cf = {x["hold_id"]: set(x["pairs"]) for c in fin["clips"] for x in c["holds"]}
    assert {k for k in cf if cf[k] != c1[k]} == {h} and cf[h] - c1[h] == set(restored[h])
    for line in open(_final_labels() / "annotations.jsonl"):
        a = json.loads(line)
        if a["hold_id"] == h and a["contact"] in restored[h]:
            assert a["label_action"] == "restored" and a["in_configuration"] and not a.get("critical")


def test_labels_v2_keeps_every_hold_id_of_the_corpus():
    v11 = ids.load_manifest(labels_v2.LABELS_V11 / "holds.yaml")
    m = ids.load_manifest(_final_labels() / "holds.yaml")
    corpus = {c["stem"] for c in m["clips"]}
    assert len(corpus) == 56
    assert {h["hold_id"] for c in v11["clips"] if c["stem"] in corpus for h in c["holds"]} == \
        {h["hold_id"] for c in m["clips"] for h in c["holds"]}


# --------------------------------------------------------------------------- #
# The new tables, rebuilt from the ledger
# --------------------------------------------------------------------------- #
def test_the_pass_b_v4_table_replays_the_ledger():
    if not b6.PASS_B_CALIBRATION.exists():
        pytest.skip("no Pass-B render_v4 table")
    stored = json.loads(b6.PASS_B_CALIBRATION.read_text())
    aud = packets.load_audit(labels_v2.default_audit_dir(labels_v2.read_v11()["id"]))
    table, failures = b6.calibrate_b(verdicts.read_ledger(), packets.read_index(packets_v4.index_path("B")), aud.records)
    assert failures == [] and json.loads(json.dumps(table["reviewers"])) == stored["reviewers"]
    for rv in stored["reviewers"]:
        assert "roles" in rv["advisory"] and "pose_identity" in rv["advisory"]


def test_b6_replays_the_ledger():
    rec = _b6_record()
    manifest, anns = b6.read_labels(B6_DRAFT)
    holds = b6.select_holds(manifest, anns)
    claims = b6.v4_claims(packets.read_index(packets_v4.index_path("B")), verdicts.read_ledger(), rec["reviewer_key"])
    claims = {h: cs for h, cs in claims.items() if h in holds}
    rep = b6.reproducibility({h: cs for h, cs in claims.items() if len(cs) >= 2})
    assert json.loads(json.dumps(rep)) == rec["reproducibility"]
    ok, why = b6.admitted(rep)
    assert (ok, why) == (rec["admitted"], rec["admission"])
    critical, votes, _, _ = b6.decide(manifest, anns, claims, b6.v3_votes(manifest), ok)
    assert json.loads(json.dumps(critical)) == rec["critical"] and votes == rec["votes"]


def test_the_render_c2_table_replays_the_ledger():
    path = edits_v2.CALIBRATION_DIR / f"{edits_v2.RENDER_V}.json"
    if not path.exists():
        pytest.skip("no render_c2 table")
    stored = json.loads(path.read_text())
    table = edits_v2.verdict_table(edits_v2.read_index())
    assert table["classes"] == stored["classes"] and table["edits"]["accepted"] == stored["edits"]["accepted"]
    assert stored["classes"]["pose_change"]["evidence"] and stored["classes"]["closer"]["evidence"]


# --------------------------------------------------------------------------- #
# Gate v2
# --------------------------------------------------------------------------- #
GATE_V2 = gate_v2.GATE_DIR / "holds_repaired_ftC_posefix.gate_v2.b2f76f2606"


def _gate() -> Path:
    if not (GATE_V2 / "gate.json").exists():
        pytest.skip("no gate v2")
    return GATE_V2


def test_gate_v2_counts_and_its_inputs_are_pinned():
    rec = json.loads((_gate() / "gate.json").read_text())
    assert rec["labels_id"] == LABELS_V2.name and rec["statics_id"] == ids.load_manifest(
        LABELS_V2 / "holds.yaml")["labels"]["statics_id"]
    c = rec["counts"]
    assert c["holds"] == 280 and c["decisions"] == {"pass": 204, "mask": 1, "flag": 65, "exclude": 10}
    assert rec["pass_c_admitted"] == {"pose_change": True, "artefact": False, "closer": True}
    rows = {json.loads(l)["hold_id"]: json.loads(l) for l in open(_gate() / "holds.jsonl")}
    excluded = {h for h, r in rows.items() if r["decision"] == "exclude"}
    transitions = {h for h, r in rows.items() if r["evidence"]["labels_status"] == "transition"}
    assert len(transitions) == 8 and excluded == transitions | {"220923_Scorpion_pose_or_vrischikasana-b@945",
                                                                "220923_Scorpion_pose_or_vrischikasana-b@1163"}
    assert {h: r["masks"] for h, r in rows.items() if r["masks"]} == {
        "220923_Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-b@669": ["R_HAND:G"]}
    table = json.loads((edits_v2.CALIBRATION_DIR / f"{edits_v2.RENDER_V}.json").read_text())
    rejected = {r["hold_id"] for r in table["edits"]["rejected"]}
    assert len(rejected) == 24 and rejected == {h for h, r in rows.items() if "fit_closer" in r["reasons"]["flag"]}
    for r in rows.values():                      # the plant's contract holds inside every released window
        if r["decision"] != "exclude":
            e = r["evidence"]
            assert e["box_max_deg"] == 0 and e["floor_min_cm"] > -1.0 and e["new_overlap_pair_frames"] == 0
            assert e["jerk_frames"] == 0


def test_every_gate_v2_exclusion_has_a_reason():
    rows = [json.loads(l) for l in open(_gate() / "holds.jsonl")]
    for r in rows:
        assert r["decision"] in gate_v2.DECISIONS
        if r["decision"] != "pass":
            assert r["reasons"][r["decision"]], r["hold_id"]
        assert all(gate_v2.reason_kind(x) in gate_v2.RULES[d] for d in ("exclude", "mask", "flag")
                   for x in r["reasons"][d]), r["hold_id"]


def test_the_released_holds_command_the_configured_supports_minus_the_masks():
    g = _gate()
    rec = json.loads((g / "gate.json").read_text())
    assert rec["a4_check"] == [] and rec["pass_c_missing"] == [] and rec["pass_c_stale"] == []
    labels = ids.load_manifest(ids.DATA_ROOT / "labels" / rec["labels_id"] / "holds.yaml")
    release = yaml.safe_load((g / "release_holds.yaml").read_text())
    rows = {json.loads(l)["hold_id"]: json.loads(l) for l in open(g / "holds.jsonl")}
    configured = {h["hold_id"]: h for c in labels["clips"] for h in c["holds"]}
    released = {h["hold_id"]: h for c in release["clips"] for h in c["holds"]}
    assert set(released) == {h for h, r in rows.items() if r["decision"] != "exclude"}
    for hid, h in released.items():
        masks = set(rows[hid]["masks"])
        assert h["pairs_ground"] == [p for p in configured[hid]["pairs_ground"] if p not in masks]
        assert h["pairs"] == [p for p in configured[hid]["pairs"] if p not in masks]
