# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the informed review, Pass B (BUILD_PLAN Step 6): its contract, its packets, a dry run
through Step 4's runner, the parse-time checks, the calibration at the packet's main moment, and
the replay of the committed Pass-B calibration table from the committed ledger.

The stores and the audit come from the session fixture ``conftest.stores`` (a temp dir). Packets
and ledgers are written to temp dirs; the replay test *reads* the committed ledger, the packets it
names in ``REVIEW_ROOT`` and the committed table.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import pytest

from extract_contact_configs import ZONE_ORDER
from reference_curation import ids, informed, labels, packets, render, review, verdicts

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a@439"
PEACOCK = "220923_Peacock_Pose_or_Mayurasana_-a@619"
BIG_TOE_C = "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c@836"  # no readable fit
HOLDS = (CROW, PEACOCK, BIG_TOE_C)
CALIBRATION_B = informed.CALIBRATION_DIR / f"{render.RENDER_V}.json"


@pytest.fixture(scope="module")
def built(stores, tmp_path_factory):
    out = tmp_path_factory.mktemp("informed")
    items = [q for q in stores.aud.queue if q["pass"] == "B" and q["hold_id"] in HOLDS]
    kw = dict(review_root=out / "review", index_root=out / "index")
    entries, failures, n = informed.build_packets(items, stores.aud, stores.stores, **kw)
    assert failures == [] and n == len(HOLDS)
    entries = {e["hold_id"]: e for e in entries}
    pkts = {h: json.loads((packets.packet_dir(e, kw["review_root"]) / "packet.json").read_text())
            for h, e in entries.items()}
    return SimpleNamespace(out=out, kw=kw, entries=entries, packets=pkts, items=items)


def _stub_ledger(built, ledger):
    reviewer = review.Reviewer(model="stub", pass_="B")
    out = review.run(list(built.entries.values()), reviewer, informed.call_stub, ledger_dir=ledger,
                     review_root=built.kw["review_root"], log=lambda *_: None)
    return reviewer, out


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_contract_extends_pass_a_unchanged():
    a, b = verdicts.load_schema("A"), verdicts.load_schema("B")
    for block in ("pose", "floor", "discrepancies", "body_body", "implausible"):  # validate and score apply as they are
        assert b["properties"][block] == a["properties"][block], block
    assert set(b["required"]) == set(a["required"]) | {"variant", "roles", "timing", "repair"} == set(b["properties"])
    role = b["properties"]["roles"]["items"]["properties"]
    assert role["role"]["enum"] == [*labels.REVIEW_ROLES, "cannot_tell"]
    assert role["part_a"]["enum"] == [render.ZONE_WORDS[z] for z in ZONE_ORDER]
    assert b["properties"]["roles"]["maxItems"] == informed.MAX_CANDIDATES
    text = verdicts.schema_path("B").read_text()  # the CLI checks draft 7 (Step 4's trap)
    assert "$schema" not in text and "$ref" not in text and "$defs" not in text


def test_prompt_and_schema_use_no_word_of_any_clip_name():
    tokens = set()
    for clip in ids.load_manifest()["clips"]:
        for name in [clip["stem"], clip.get("family") or "", ids.recording_id(clip["stem"]) or ""] + \
                [h["name"] for h in clip["holds"]] + [h.get("orientation") or "" for h in clip["holds"]]:
            tokens |= {t.lower() for t in re.split(r"[_\-()\s]+", name) if len(t) >= 4 and re.search("[A-Za-z]", t)}
    tokens -= packets.GENERIC_TOKENS
    text = (verdicts.prompt_path("B").read_text() + verdicts.schema_path("B").read_text()).lower()
    assert sorted(t for t in tokens if t in text) == []


# --------------------------------------------------------------------------- #
# Packets
# --------------------------------------------------------------------------- #
def test_the_packet_shows_the_corrected_exemplar_with_the_source_labels(built):
    p, e = built.packets[CROW], built.entries[CROW]
    assert p["pass"] == "B" and p["contract"] == informed.CONTRACT and "hold" not in p
    assert p["labels"]["hold_id"] == CROW and p["labels"]["label_ground"] == ["L_HAND", "R_HAND"]
    cap = p["capture"]
    assert (cap["main_frame"], cap["source_main_frame"], cap["window"]["frame_start"]) == (651, 439, 489)
    assert e["frame_hold"] == 651 and e["source_frame_hold"] == 439
    strip = [f["frame"] for f in p["evidence"]["frames"]]
    assert {425, 489, 651, 719} <= set(strip)             # label start, corrected start, main moment, end
    assert [f["frame"] for f in p["human_mesh"]["frames"]] == strip
    main_strip = [t for t in cap["main_panels"] if t.startswith(str(len(p["images"])))]
    assert len(main_strip) == 1                            # the main moment is one of the strip's panels
    labelled = [c["pair"] for c in p["candidates"] if c["labelled"]]
    assert labelled == sorted((x for x in p["labels"]["label_pairs"] if ":" not in x),
                              key=lambda x: labels.PAIR_COLUMN[labels.parse_pair(x)])
    assert [c["pair"] for c in p["candidates"]][:len(labelled)] == labelled
    shelf = next(c for c in p["candidates"] if c["pair"] == "L_SHANK+L_UPPER_ARM")
    assert shelf["skin_state_at_main"] == 1 and shelf["parts"] == "left shin + left upper arm"
    frame = next(f for f in p["human_mesh"]["frames"] if f["frame"] == 651)
    assert set(frame["floor"]) == set(ZONE_ORDER) and set(frame["pairs"]) == {c["pair"] for c in p["candidates"]}
    assert frame["floor"]["L_FOOT"]["state"] == 0 and frame["floor"]["L_HAND"]["state"] == 1
    assert len(p["images"]) <= packets.MAX_IMAGES and e["bytes"] <= packets.MAX_PACKET_BYTES


def test_a_packet_without_a_fit_says_so(built):
    p = built.packets[BIG_TOE_C]
    assert p["legend"]["markers"] == packets.NO_MARKERS
    assert all(c["skin_state_at_main"] == -1 for c in p["candidates"])


def test_a_rebuild_reproduces_every_packet_id(built, stores):
    before = {h: e["packet_id"] for h, e in built.entries.items()}
    entries, failures, n = informed.build_packets(built.items, stores.aud, stores.stores, force=True, **built.kw)
    assert failures == [] and n == len(HOLDS) and {e["hold_id"]: e["packet_id"] for e in entries} == before


# --------------------------------------------------------------------------- #
# Review and claims
# --------------------------------------------------------------------------- #
def test_a_dry_run_reviews_every_packet_through_step_4s_runner(built, tmp_path):
    ledger = tmp_path / "ledger"
    reviewer, out = _stub_ledger(built, ledger)
    assert out["status"] == {"valid": len(HOLDS)} and out["failures"] == []
    assert _stub_ledger(built, ledger)[1]["current"] == len(HOLDS)       # resumed: nothing to do
    records = verdicts.read_ledger(ledger)
    assert {r["pass"] for r in records} == {"B"} and all(r["reviewer"]["key"] == reviewer.key for r in records)
    assert reviewer.key != review.Reviewer(model="stub", pass_="A").key
    for r in records:
        claims = labels.parse_review(r, built.packets[r["hold_id"]])
        assert claims["errors"] == [] and claims["unanswered"] == [] and set(claims["roles"].values()) <= {None}
        assert claims["frame"] == r["frame_hold"] and claims["verdict_id"] == f"ledger:{r['packet_id']}.B.1"


def test_parse_review_rejects_what_the_validator_cannot(built):
    p = built.packets[CROW]
    answer = informed.stub_answer(p)
    assert verdicts.validate(answer, p, "B") == ([], [])
    dropped = answer["roles"].pop()
    answer["roles"].append(copy.deepcopy(answer["roles"][0]))
    answer["roles"].append({"part_a": "head", "part_b": "left foot", "role": "incidental", "note": "", "panels": []})
    answer["timing"].update(main_moment="move", better_panel="1a")
    assert verdicts.validate(answer, p, "B") == ([], [])  # the schema and Step 4's checks cannot see these
    claims = labels.parse_review({"answer": answer, "packet_id": "x", "n": 1}, p)
    assert any("given twice" in e for e in claims["errors"]) and any("not a candidate" in e for e in claims["errors"])
    assert any("not a time-strip panel" in e for e in claims["errors"]) and claims["timing"]["better_frame"] is None
    first = answer["roles"][0]
    assert claims["unanswered"] == [labels.pair_name(sorted((verdicts.PARTS[dropped["part_a"]],
                                                             verdicts.PARTS[dropped["part_b"]]), key=labels.ZI.get))]
    assert labels.pair_name(sorted((verdicts.PARTS[first["part_a"]], verdicts.PARTS[first["part_b"]]),
                                   key=labels.ZI.get)) in claims["roles"]
    answer["roles"][0]["note"] = "the shin carries 300 N"  # Step 4's validator still rejects force claims
    assert verdicts.validate(answer, p, "B")[0][0].startswith("force claim")


def test_labels_attach_the_reviews_and_cite_the_verdicts(built, stores, tmp_path):
    ledger = tmp_path / "ledger"
    reviewer, _ = _stub_ledger(built, ledger)
    current = {h: e["packet_id"] for h, e in built.entries.items()}
    reviews, failures = labels.load_reviews(ledger, reviewer, current=current)
    assert failures == [] and set(reviews) == set(HOLDS)
    rows = [{"id": c["verdict_id"], "kind": "verdict"} for c in reviews.values()]
    result = labels.build(stores.aud, stores.stores, stems=[h.rsplit("@", 1)[0] for h in HOLDS], reviews=reviews)
    assert labels.check(result, stores.stores) == []
    crow = [a for a in result["annotations"] if a["hold_id"] == CROW]
    shelf = next(a for a in crow if a["contact"] == "L_SHANK+L_UPPER_ARM")
    assert shelf["review"]["role"] is None and reviews[CROW]["verdict_id"] in shelf["evidence_ids"]
    assert result["evidence"][reviews[CROW]["verdict_id"]]["kind"] == "verdict" and rows
    assert result["holds"][CROW]["labels"]["review"]["variant"]["matches_label"] == "cannot_tell"


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_scores_at_the_packets_main_moment(built, stores, tmp_path):
    ledger = tmp_path / "ledger"
    _stub_ledger(built, ledger)
    table, rows, failures = informed.calibrate(verdicts.read_ledger(ledger), stores.aud, stores.stores)
    assert failures == [] and table["holds"] == len(HOLDS) and len(table["reviewers"]) == 1
    rv = table["reviewers"][0]
    assert rv["evidence"] == [] and {"roles", "variant", "timing", "repair"} <= set(rv["advisory"])
    assert rv["pass_b"]["roles"]["cannot_tell"] == sum(len(p["candidates"]) for p in built.packets.values())
    crow = [r for r in rows[rv["key"]] if r["hold_id"] == CROW and r["class"] == "floor_human"]
    # at the corrected exemplar (651) the toe is up; at the source exemplar (439) it was down
    assert {r["key"]: r["truth"] for r in crow}["L_FOOT"] == 0


@pytest.mark.skipif(not CALIBRATION_B.exists(), reason="no Pass-B calibration table yet")
def test_pass_b_calibration_replays_the_ledger(stores):
    """The committed table is what the committed ledger says: a rebuild re-scores, never re-queries."""
    stored = json.loads(CALIBRATION_B.read_text())
    table, _, failures = informed.calibrate(verdicts.read_ledger(), stores.aud, stores.stores)
    assert failures == [] and table["audit_ids"] == stored["audit_ids"] == [stores.aud.audit_id]
    assert json.loads(json.dumps(table["reviewers"])) == stored["reviewers"]
    for rv in stored["reviewers"]:
        assert rv["model"] == review.MODEL and "roles" in rv["advisory"] and "roles" not in rv["evidence"]
        for cls, m in rv["classes"].items():
            admit = (m["precision"] is not None and m["precision"] >= 0.9 and m["answered_rate"] >= 0.5
                     and m["items"] >= 30 and m["support"] >= 30)
            assert m["evidence"] == admit and (cls in rv["evidence"]) == admit, cls
