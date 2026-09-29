# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the corpus audit and review queue (BUILD_PLAN Step 2).

They run on the real clips and assert numbers the review already measured
(``expert_revist/reference_curation_review_2026_09_28/README.MD`` §3.1-3.3,
``output_capture_marker_audit.txt`` and ``output_support_hover_audit.txt``), so a regression shows
up as a wrong number. The module fixture builds the capture store into a temp dir, as
``test_capture.py`` does, and writes the audit there. Nothing under ``output/`` or
``data/reference_curation/`` is touched.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from reference_curation import audit, capture, ids

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
SIDE_CROW_C = "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c"
BIG_TOE_C = "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c"  # its MoSh pkl is corrupt
# The review's exemplar audit, hold by hold (README §3.2, output_capture_marker_audit.txt).
REVIEW_MISSED = {
    "220923_Plow_Pose_or_Halasana_-a@876": ("L_FOOT", "R_FOOT"),
    "220926_Plow_Pose_or_Halasana_-b@980": ("L_FOOT", "R_FOOT"),
    "220926_Plow_Pose_or_Halasana_-b@1384": ("L_FOOT", "R_FOOT"),
    f"{CROW}@439": ("L_FOOT",),
    "220923_Peacock_Pose_or_Mayurasana_-a@619": ("L_FOOT", "R_FOOT"),
    "220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b@873": ("L_FOOT",),
    f"{SIDE_CROW_C}@1296": ("L_FOOT",),
    "220923_Plank_Pose_or_Kumbhakasana_-a@541": ("R_FOOT",),
    "220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a@565": ("R_FOOT",),
    "220923_Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-a@770": ("R_FOOT",),
    "220923_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-a@570": ("R_FOOT",),
}
REVIEW_PHANTOM = {
    "220923_Side_Plank_Pose_or_Vasisthasana_-a@879": ("L_FOOT",),
    "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a@848": ("L_HAND",),
    "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a@1056": ("L_HAND", "R_HAND"),
}

pytestmark = pytest.mark.skipif(
    not (ids.MOYO_DATA / "mosh").exists() or not ids.SHIPPED_DIR.exists(),
    reason="the MOYO MoSh fits or the shipped ftC clips are not on disk")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("audit")
    cal, _, failures = capture.build_all(ids.manifest_stems(), out / "store", out / "capture_v1.json")
    assert failures == []
    rules = {}
    for rule in audit.RULES:
        records, context, failures = audit.audit(rule=rule, store_dir=out / "store", calibration=cal)
        assert failures == []
        rules[rule] = SimpleNamespace(records=records, context=context, by_id={r["hold_id"]: r for r in records})
    written = audit.write(rules["calibrated"].records, rules["calibrated"].context, out / "audits")
    return SimpleNamespace(out=out, cal=cal, written=written, **rules)


def _exemplar_disagreements(records, key):
    return {r["hold_id"]: tuple(r["disagreements"][key]) for r in records
            if r["family_hold"] and r["capture"]["available"] and r["disagreements"][key]}


def _reason(record, code):
    return next((x for x in record["reasons"] if x["code"] == code), None)


# --------------------------------------------------------------------------- #
# Pure pieces
# --------------------------------------------------------------------------- #
def test_capture_bounds_extend_past_the_window_while_the_match_holds():
    match = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    assert audit.capture_bounds(match, 2, 7) == (1, 9)   # runs on at both ends
    assert audit.capture_bounds(match, 4, 7) == (6, 9)   # the label starts 2 frames early
    assert audit.capture_bounds(match, 4, 5) == (None, None)


def test_support_changes_ignore_unknown_gaps_and_first_observations():
    s = np.array([[-1, 1], [1, 1], [-1, 1], [1, 1], [1, 0], [-1, 0], [0, 0]], dtype=np.int8)
    # frame 1: first observation of zone 0 (no change); frame 3: back from unknown, same state
    assert audit.support_changes(s).tolist() == [4, 6]


def test_severity_table():
    labels, hover = {"code": "missed_touch", "fix": "labels"}, {"code": "hover", "fix": "retarget", "max_cm": 11.6}
    assert audit.severity([labels], True) == "high" and audit.severity([labels], False) == "medium"
    assert audit.severity([hover], True) == "medium" and audit.severity([hover], False) == "low"
    assert audit.severity([dict(hover, max_cm=3.0)], True) == "low" and audit.severity([], True) == "none"


# --------------------------------------------------------------------------- #
# The review, reproduced
# --------------------------------------------------------------------------- #
def test_pooled_rule_reproduces_the_review_exemplar_audit(run):
    family = [r for r in run.pooled.records if r["family_hold"]]
    assert len(family) == 70 and sum(r["capture"]["available"] for r in family) == 69
    missed = _exemplar_disagreements(run.pooled.records, "missed_touch")
    phantom = _exemplar_disagreements(run.pooled.records, "phantom_support")
    assert missed == REVIEW_MISSED and phantom == REVIEW_PHANTOM
    assert (sum(map(len, missed.values())), len(missed)) == (15, 11)
    assert (sum(map(len, phantom.values())), len(phantom)) == (4, 3)
    m = audit.metrics(run.pooled.records)["family"]
    assert (m["missed_touch"], m["phantom_support"]) == ({"holds": 11, "zones": 15}, {"holds": 3, "zones": 4})


def test_calibrated_rule_counts(run):
    """Per-zone thresholds on the store's full marker map (BUILD_PLAN §4)."""
    missed = _exemplar_disagreements(run.calibrated.records, "missed_touch")
    phantom = _exemplar_disagreements(run.calibrated.records, "phantom_support")
    assert (sum(map(len, missed.values())), len(missed)) == (18, 12)
    assert (sum(map(len, phantom.values())), len(phantom)) == (6, 4)
    koundinya, dolphin = ("220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b@873",
                          "220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a@354")
    assert missed == {**REVIEW_MISSED, koundinya: ("L_FOOT", "R_FOOT", "HEAD"), dolphin: ("R_FOOT",)}
    # The head is decided now: Chaturanga's HEAD:G is a wrong label (README §3.1), and Bridge's
    # right hand reads its wrist at 4.9 cm, above the 2.5 cm hand touch.
    chaturanga = "220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a@485"
    bridge = "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a@848"
    assert phantom == {**REVIEW_PHANTOM, chaturanga: ("HEAD",), bridge: ("L_HAND", "R_HAND")}
    assert run.calibrated.by_id[chaturanga]["zones"]["HEAD"]["marker_cm"] == pytest.approx(26.2, abs=0.1)


def test_labelled_supports_hover_as_the_review_measured(run):
    """README §3.1: hold-median height of every labelled ground support."""
    for rule in ("calibrated", "pooled"):  # geometry only: the rule must not change it
        a = audit.metrics(getattr(run, rule).records)
        assert a["all"]["labelled_supports"] == 833
        assert a["all"]["hover_gt_cm"] == {"1": 467, "2": 367, "3": 237, "5": 120}
    by_zone = {z: (v["hover"], v["labelled"]) for z, v in a["labelled_hover_gt_2cm_by_zone"].items()}
    assert {z: by_zone[z] for z in ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND", "L_FOREARM", "R_FOREARM", "HEAD", "TRUNK")} \
        == {"L_FOOT": (139, 195), "R_FOOT": (34, 190), "L_HAND": (9, 120), "R_HAND": (66, 127),
            "L_FOREARM": (2, 38), "R_FOREARM": (32, 37), "HEAD": (30, 38), "TRUNK": (13, 13)}
    assert by_zone["L_SHANK"][0] + by_zone["R_SHANK"][0] == 31 == by_zone["L_SHANK"][1] + by_zone["R_SHANK"][1]


def test_crow_starts_its_hands_only_hold_late(run):
    """README §3.3: the toe stays down until ~8.3 s; the label starts at 7.08 s, exemplar 7.32 s."""
    r = run.calibrated.by_id[f"{CROW}@439"]
    assert r["window"]["t_start"] == pytest.approx(7.083, abs=0.001) and r["label_ground"] == ["L_HAND", "R_HAND"]
    assert 8.2 < r["capture"]["first_s"] < 8.35
    assert r["capture"]["ground_exemplar"] == ["L_FOOT", "L_HAND", "R_HAND"]
    assert r["capture"]["exemplar_stable"] and r["capture"]["nearest_change_s"] == pytest.approx(0.933, abs=0.02)
    assert [x["code"] for x in r["reasons"]] == ["missed_touch", "boundary_start"] and r["severity"] == "high"
    assert r["zones"]["L_FOOT"]["load_n"] is None  # the reference foot floats 7.8 cm: attribution-blind
    assert r["zones"]["L_FOOT"]["avatar_cm"] == pytest.approx(7.8, abs=0.1)
    assert 25 <= r["mat"]["unexplained_exemplar_n"] <= 60  # the toe's share at the exemplar
    assert audit.questions(r) == [{"id": "source_state", "frame": 439, "zones": ["L_FOOT"]},
                                  {"id": "boundary", "label": [425, 719], "capture": [495, 722]}]


def test_side_crow_c_holds_start_on_a_toe(run):
    """README §3.3: toe down until ~7.5-8 s (hold 1) and ~22.0 s (hold 2); hold 2's exemplar is toe-down."""
    h1, h2 = run.calibrated.by_id[f"{SIDE_CROW_C}@606"], run.calibrated.by_id[f"{SIDE_CROW_C}@1296"]
    assert (h1["window"]["t_start"], h2["window"]["t_start"]) == pytest.approx((6.783, 21.0), abs=0.001)
    assert 7.5 <= h1["capture"]["first_s"] <= 8.05 and 21.95 <= h2["capture"]["first_s"] <= 22.15
    assert _reason(h1, "boundary_start") and not h1["disagreements"]["missed_touch"]
    assert h2["disagreements"]["missed_touch"] == ["L_FOOT"] and not h2["capture"]["exemplar_stable"]
    assert 0 < h2["capture"]["nearest_change_s"] <= audit.STABLE_S
    assert h1["hold_id"] != h2["hold_id"]


@pytest.mark.parametrize("hold_id, hover, loads, total, unexplained", [
    # README §3.1's worst family holds: labelled support height (cm), attributed load (N), mat total and
    # unexplained (N). A support floating > 6 cm is attribution-blind: its load is None, not 0.
    ("220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a@540",
     {"L_FOOT": 11.6, "L_HAND": 0.5, "R_HAND": 2.3}, {"L_FOOT": None, "L_HAND": 40, "R_HAND": 37}, 664, 581),
    ("220923_Half_Moon_Pose_or_Ardha_Chandrasana_-b@556", {"L_FOOT": 9.2}, {"L_FOOT": None, "R_HAND": 155}, 676, 522),
    ("220923_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-a@570", {"L_FOOT": 8.7},
     {"L_FOOT": None, "R_HAND": 176}, 683, 510),
    ("220926_Upward_Plank_Pose_or_Purvottanasana_-a@854", {"L_FOOT": 10.2, "R_FOOT": 9.8},
     {"L_FOOT": None, "R_FOOT": None, "L_HAND": 25, "R_HAND": 389}, 681, 259),
    ("220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a@1051", {"HEAD": 14.0},
     {"HEAD": None, "L_FOREARM": 156, "R_FOREARM": 195}, 670, 221),
    ("220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a@485", {"HEAD": 8.7}, {"HEAD": None}, 654, 4),
])
def test_worst_family_holds_and_the_mat(run, hold_id, hover, loads, total, unexplained):
    r = run.calibrated.by_id[hold_id]
    assert r["family_hold"]
    for zone, cm in hover.items():
        assert r["zones"][zone]["hover_cm"] == pytest.approx(cm, abs=0.06), zone
    for zone, n in loads.items():
        assert (r["zones"][zone]["load_n"] is None) if n is None else r["zones"][zone]["load_n"] == pytest.approx(n, abs=1.5)
    assert (r["mat"]["total_n"], r["mat"]["unexplained_n"]) == pytest.approx((total, unexplained), abs=1.5)
    assert bool(_reason(r, "unexplained_load")) == (unexplained > audit.UNEXPLAINED_N)


def test_hover_the_markers_confirm_asks_no_question(run):
    """A floating support the human does touch is the retarget's job (Step 8), not the reviewer's."""
    r = run.calibrated.by_id["220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a@540"]
    assert r["disagreements"]["hover"] == ["L_FOOT", "R_HAND"] and r["zones"]["L_FOOT"]["exemplar"] == 1
    assert [q["id"] for q in audit.questions(r)] == ["hidden_support"]
    assert _reason(r, "unexplained_load")["zones"] == ["L_FOOT"]
    side_plank = run.calibrated.by_id["220923_Side_Plank_Pose_or_Vasisthasana_-a@879"]
    # The stacked top foot is lifted: a phantom, not a hover, and it cannot hide floor load.
    assert side_plank["disagreements"]["phantom_support"] == ["L_FOOT"]
    assert "L_FOOT" not in side_plank["disagreements"]["hover"]
    assert "L_FOOT" not in _reason(side_plank, "unexplained_load")["zones"]


def test_unreadable_fit_is_capture_unavailable(run):
    holds = [r for r in run.calibrated.records if r["stem"] == BIG_TOE_C]
    assert len(holds) == 3
    for r in holds:
        assert not r["capture"]["available"] and r["capture"]["decided"] == []
        assert [x["code"] for x in r["reasons"]] == ["capture_unavailable"] and r["severity"] == "low"
        assert all(z.get("exemplar") is None for z in r["zones"].values())
        assert all(r["zones"][z]["hover_cm"] is not None for z in r["label_ground"])
        assert audit.questions(r)[0] == {"id": "source_state", "frame": r["window"]["frame_hold"],
                                         "zones": r["label_ground"]}


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def test_written_audit(run):
    d, ctx = run.written, run.calibrated.context
    assert d.name == ctx["audit_id"] and ctx["audit_id"].startswith("holds_repaired_ftC_posefix.calibrated.")
    holds = [json.loads(line) for line in (d / "holds.jsonl").read_text().splitlines()]
    assert len(holds) == 303 and {h["schema_version"] for h in holds} == {audit.SCHEMA_VERSION}
    assert {h["audit_id"] for h in holds} == {ctx["audit_id"]}
    assert [h["hold_id"] for h in holds] == [r["hold_id"] for r in run.calibrated.records]
    record = json.loads((d / "audit.json").read_text())
    assert record["generator"]["sha256"] == ids.sha256_file(audit.__file__)
    assert record["inputs"]["data/smpl/expert60/holds_repaired_ftC_posefix.yaml"] == ids.sha256_file(ids.DEFAULT_MANIFEST)
    assert record["metrics"] == json.loads(json.dumps(audit.metrics(run.calibrated.records)))
    assert record["calibration_id"] == run.cal["id"]


def test_summary_lists_every_flagged_hold(run):
    text = (run.written / "summary.md").read_text()
    table = text[text.index("## Flagged holds"):]
    flagged = [r for r in run.calibrated.records if r["reasons"]]
    assert len(flagged) == sum(r["severity"] != "none" for r in run.calibrated.records) == 240
    for r in run.calibrated.records:
        assert (f"`{r['hold_id']}`" in table) == bool(r["reasons"]), r["hold_id"]
    assert "| Labelled ground supports hovering > 2 cm | 367 / 833 (44 %) |" in text
    assert "| `missed_touch` | 99 in 57 holds | 18 in 12 holds |" in text


def test_review_queue(run):
    queue = [json.loads(line) for line in (run.written / "queue.jsonl").read_text().splitlines()]
    records = run.calibrated.by_id
    for item in queue:
        assert {"hold_id", "window", "pass", "questions", "priority"} <= set(item) and item["hold_id"] in records
        w = item["window"]
        assert w["frame_start"] <= w["frame_hold"] <= w["frame_end"] and w["frame_hold"] in w["key_frames"]
    for p in ("A", "B"):
        assert sorted(q["priority"] for q in queue if q["pass"] == p) == list(range(1, 1 + sum(q["pass"] == p for q in queue)))
    pass_b = sorted((q for q in queue if q["pass"] == "B"), key=lambda q: q["priority"])
    flagged = [q for q in pass_b if q["purpose"] == "flagged"]
    asking = [r for r in run.calibrated.records if audit.questions(r)]
    assert {q["hold_id"] for q in flagged} == {r["hold_id"] for r in asking}
    ranks = [audit.SEVERITY.index(q["severity"]) for q in flagged]
    assert ranks == sorted(ranks, reverse=True) and flagged[0]["severity"] == "high"
    assert all(q["purpose"] == "flagged" for q in pass_b[:len(flagged)])  # controls come last
    controls = [q for q in pass_b if q["purpose"] == "control"]
    assert len(controls) == math.ceil(audit.CONTROL_FRACTION * (len(run.calibrated.records) - len(asking)))
    assert all(not audit.questions(records[q["hold_id"]]) for q in controls)
    pass_a = sorted((q for q in queue if q["pass"] == "A"), key=lambda q: q["priority"])
    assert all(records[q["hold_id"]]["capture"]["truth_unambiguous"] for q in pass_a)
    assert [q["stratum"] for q in pass_a[:len(audit.STRATA)]] == list(audit.STRATA)
    assert all(q["questions"] == [{"id": x} for x in audit.PASS_A_QUESTIONS] for q in pass_a)
    crow_b = next(q for q in queue if q["item_id"] == f"{CROW}@439#B")
    assert crow_b["severity"] == "high" and crow_b["window"]["key_frames"] == [425, 439, 495, 719, 722]


def test_audit_id_is_content_addressed(run):
    again, context, _ = audit.audit(rule="calibrated", store_dir=run.out / "store", calibration=run.cal)
    assert context["audit_id"] == run.calibrated.context["audit_id"] and again == run.calibrated.records
    assert run.pooled.context["audit_id"] != run.calibrated.context["audit_id"]


def test_cli_writes_and_fails_loudly(run, tmp_path, capsys):
    source = ids.load_manifest()
    small = dict(source, clips=[c for c in source["clips"] if c["stem"] in (CROW, BIG_TOE_C)])
    manifest = tmp_path / "two.yaml"
    manifest.write_text(yaml.safe_dump(small))
    argv = ["--manifest", str(manifest), "--store-dir", str(run.out / "store"),
            "--calibration", str(run.out / "capture_v1.json"), "--out-root", str(tmp_path / "audits")]
    assert audit.main(argv) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("audit two.calibrated.") and "\n" not in line
    assert len((tmp_path / "audits").glob("two.calibrated.*/holds.jsonl").__next__().read_text().splitlines()) == 7
    variant = dict(source, clips=[dict(next(c for c in source["clips"] if c["stem"] == CROW), stem=f"{CROW}_x3s")])
    (tmp_path / "variant.yaml").write_text(yaml.safe_dump(variant))
    before = sorted(p.name for p in (tmp_path / "audits").iterdir())
    argv[1] = str(tmp_path / "variant.yaml")
    assert audit.main(argv) == 1 and "x0 clips" in capsys.readouterr().err
    assert sorted(p.name for p in (tmp_path / "audits").iterdir()) == before  # nothing written
