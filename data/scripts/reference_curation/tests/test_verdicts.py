# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the headless reviewer, the verdict ledger and the calibration (BUILD_PLAN Step 4).

They run on the real clips and pin the truth the review measured. The Standing Split -a standing
foot floats 11.45 cm over markers at 3.0 cm (README §3.2, BUILD_PLAN §5), the Supported Headstand -b
head floats 13.5 cm, and Crow -a's shins rest on its upper arms. The module fixture builds the
capture store, the audit and three pilot packets into a temp dir, as ``test_render.py`` does.
Nothing under ``output/``, ``data/reference_curation/`` or ``REVIEW_ROOT`` is written. The
replay test *reads* the committed ledger and calibration table.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import stat
import sys
from types import SimpleNamespace

import pytest

from extract_contact_configs import ADJACENT, ZONE_ORDER
from reference_curation import audit, capture, ids, packets, render, review, verdicts

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
STANDING_SPLIT = "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a"
HEADSTAND_B = "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b"
HOLDS = (f"{STANDING_SPLIT}@540", f"{CROW}@439", f"{HEADSTAND_B}@1093")

pytestmark = pytest.mark.skipif(
    not (ids.MOYO_DATA / "mosh").exists() or not ids.SHIPPED_DIR.exists(),
    reason="the MOYO MoSh fits or the shipped ftC clips are not on disk")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("verdicts")
    cal_path = out / "capture_v1.json"
    cal, _, failures = capture.build_all(ids.manifest_stems(), out / "store", cal_path)
    assert failures == []
    records, context, failures = audit.audit(store_dir=out / "store", calibration=cal)
    assert failures == []
    audit_dir = audit.write(records, context, out / "audits")
    aud = packets.load_audit(audit_dir)
    kw = dict(manifest=aud.manifest, review_root=out / "review", index_root=out / "index",
              store_dir=out / "store", calibration=cal)
    items = packets.select_items(aud, "A", holds=list(HOLDS))
    entries, failures, _ = packets.build_packets(items, aud.records, **kw)
    assert failures == []
    entries = {e["hold_id"]: e for e in entries}
    truths = {h: verdicts.packet_truth(aud.records[h], capture.load(aud.records[h]["stem"], out / "store", cal), cal)
              for h in HOLDS}
    return SimpleNamespace(out=out, cal=cal, cal_path=cal_path, aud=aud, audit_dir=audit_dir, kw=kw,
                           entries=entries, truths=truths)


def _packet(run, hold_id) -> dict:
    return json.loads((packets.packet_dir(run.entries[hold_id], run.kw["review_root"]) / "packet.json").read_text())


def _answer(**floor) -> dict:
    """The abstaining answer, with ``floor`` overrides {part word: (avatar, markers)}."""
    answer = review.stub_answer()
    for part, (avatar, markers) in floor.items():
        answer["floor"][part.replace("_", " ")] = {"avatar": avatar, "markers": markers, "panels": ["1a"]}
    return answer


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_schema_names_parts_with_the_packets_words():
    schema = verdicts.load_schema()
    parts = [render.ZONE_WORDS[z] for z in ZONE_ORDER]
    assert schema["properties"]["floor"]["required"] == parts == list(verdicts.PARTS)
    assert [verdicts.PARTS[p] for p in parts] == ZONE_ORDER
    props = schema["properties"]
    for enum in (props["discrepancies"]["items"]["properties"]["part"]["enum"],
                 props["body_body"]["properties"]["contacts"]["items"]["properties"]["part_a"]["enum"],
                 props["implausible"]["items"]["properties"]["parts"]["items"]["enum"]):
        assert enum == parts
    # The CLI's validator knows draft 7 only: a 2020-12 "$schema" made every call exit 1 (the build).
    text = verdicts.schema_path().read_text()
    assert "$schema" not in text and "$ref" not in text and "$defs" not in text


def test_prompt_and_schema_use_no_word_of_any_clip_name():
    """Both reach the reviewer. The scan is test_render's: every name token of the corpus, as a
    substring ("holds" and "threshold" contain one; so do "inside", "crown" and "unsupported")."""
    tokens = set()
    for clip in ids.load_manifest()["clips"]:
        for name in [clip["stem"], clip.get("family") or "", ids.recording_id(clip["stem"]) or ""] + \
                [h["name"] for h in clip["holds"]] + [h.get("orientation") or "" for h in clip["holds"]]:
            tokens |= {t.lower() for t in re.split(r"[_\-()\s]+", name) if len(t) >= 4 and re.search("[A-Za-z]", t)}
    tokens -= packets.GENERIC_TOKENS
    text = (verdicts.prompt_path().read_text() + verdicts.schema_path().read_text()).lower()
    assert sorted(t for t in tokens if t in text) == []
    assert {"hold", "side", "crow", "supported", "legs", "shoulder"} <= tokens


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
def test_validator_accepts_a_good_answer(run):
    packet = _packet(run, f"{STANDING_SPLIT}@540")
    answer = _answer(left_foot=("off_floor", "touching"), left_hand=("touching", "touching"))
    answer["pose"]["candidates"] = [{"name": "a one-leg fold", "confidence": "low"}]
    answer["body_body"] = {"answer": "listed", "contacts": []}
    answer["discrepancies"] = [{"part": "left foot", "kind": "avatar_off_floor_markers_on_floor",
                                "note": "drop line under the foot; markers on the floor", "panels": ["3a", "2b"]}]
    assert verdicts.validate(answer, packet) == ([], [])
    assert verdicts.validate(review.stub_answer(), packet) == ([], [])


def test_validator_rejects_bad_names_missing_images_and_extra_fields(run):
    packet = _packet(run, f"{STANDING_SPLIT}@540")
    good = _answer(left_foot=("off_floor", "touching"))

    def errors(mutate):
        answer = copy.deepcopy(good)
        mutate(answer)
        return verdicts.validate(answer, packet)[0]

    # names outside the vocabulary
    assert errors(lambda a: a["floor"].update({"left paw": a["floor"].pop("left foot")}))
    assert errors(lambda a: a["floor"]["left foot"].update(avatar="hovering"))
    assert errors(lambda a: a["body_body"].update(answer="listed", contacts=[
        {"part_a": "left knee", "part_b": "torso", "certainty": "clear", "panels": ["1a"]}]))
    # images and panels the packet does not have
    assert errors(lambda a: a["floor"]["left foot"].update(panels=["5c"])) == ["panel 5c is not in the packet"]
    assert errors(lambda a: a["floor"]["left foot"].update(panels=["3z"]))  # not a tag at all (schema)
    assert errors(lambda a: a["pose"].update(description="see img_9.png")) == ["image img_9.png is not in the packet"]
    assert errors(lambda a: a["pose"].update(description="see img_1.png")) == []
    # extra fields, at the top and inside
    assert errors(lambda a: a.update(verdict="fine"))
    assert errors(lambda a: a["floor"]["head"].update(height_cm=3))
    assert errors(lambda a: a.pop("implausible"))
    # word limits
    assert errors(lambda a: a["pose"].update(description="x" * 481))


def test_validator_rejects_force_claims_and_bad_contacts(run):
    packet = _packet(run, f"{CROW}@439")
    for text in ("the hands carry about 300 N", "a force through the wrists", "0.4 kN on the feet",
                 "roughly 60% of body weight on the hands", "newtons"):
        answer = _answer()
        answer["pose"]["description"] = text
        errs, _ = verdicts.validate(answer, packet)
        assert errs and errs[0].startswith("force claim"), text
    answer = _answer()
    answer["pose"]["description"] = "the loaded hands bear the body"
    assert verdicts.validate(answer, packet) == ([], ["load word 'bear'", "load word 'loaded'"])

    def contacts(*pairs, answer="listed"):
        a = _answer()
        a["body_body"] = {"answer": answer, "contacts": [
            {"part_a": x, "part_b": y, "certainty": "clear", "panels": ["1a"]} for x, y in pairs]}
        return verdicts.validate(a, packet)

    assert contacts(("left shin", "left upper arm"), ("right shin", "right upper arm")) == ([], [])
    assert contacts(("left shin", "left shin"))[0] == ["body_body: left shin paired with itself"]
    assert contacts(("left shin", "left upper arm"), ("left upper arm", "left shin"))[0] == \
        ["body_body: left upper arm + left shin listed twice"]
    assert contacts(("left shin", "left upper arm"), answer="cannot_tell")[0] == \
        ["body_body: contacts listed under cannot_tell"]
    assert contacts(("left hand", "left forearm")) == ([], ["body_body: left hand + left forearm are joined; not scored"])


# --------------------------------------------------------------------------- #
# Truth and pose names
# --------------------------------------------------------------------------- #
def test_truth_pins_the_measured_floats_and_contacts(run):
    split = run.truths[f"{STANDING_SPLIT}@540"]
    assert split["human"]["L_FOOT"] == 1 and split["avatar"]["L_FOOT"] == 0
    assert split["avatar_cm"]["L_FOOT"] == 11.453 and split["marker_cm"]["L_FOOT"] == 2.992
    assert split["human"]["R_FOOT"] is None  # the raised foot's fit residual: unknown (Step 1)
    assert split["avatar"]["L_HAND"] == 1 and split["avatar_cm"]["R_HAND"] == 2.096
    assert split["pose"] == "Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana"
    head = run.truths[f"{HEADSTAND_B}@1093"]
    assert head["human"]["HEAD"] == 1 and head["avatar_cm"]["HEAD"] == 13.505 and head["avatar"]["HEAD"] == 0
    assert head["human"]["L_FOOT"] == head["human"]["R_FOOT"] == 0
    crow = run.truths[f"{CROW}@439"]
    assert crow["human"]["L_FOOT"] == 1 and crow["avatar_cm"]["L_FOOT"] == 7.797  # the toe-down exemplar
    assert {p for p, t in crow["pairs"].items() if t == 1} == {
        ("R_THIGH", "R_UPPER_ARM"), ("L_SHANK", "L_UPPER_ARM"), ("R_SHANK", "R_UPPER_ARM")}
    assert crow["pair_gap_cm"][("R_SHANK", "R_UPPER_ARM")] == 0.173
    assert all(frozenset(p) not in ADJACENT for p in crow["pairs"])


def test_gated_pair_gaps_are_exact_below_the_gate():
    exact = {p: 100.0 * min(float(verdicts.geom_pair_distance(ga, gb)[0][0])
                            for a in verdicts.ZONES[p[0]] for b in verdicts.ZONES[p[1]]
                            for ga in world[a] for gb in world[b])
             for world in [_world(CROW, 439)] for p in verdicts.BODY_PAIRS}
    gated = verdicts.pair_gaps_cm(CROW, 439)
    assert all(gated[p] <= exact[p] + 1e-9 for p in exact)
    assert all(gated[p] == exact[p] for p in exact if exact[p] < 100 * verdicts.PAIR_GATE_M)


def _world(stem, frame):
    clip, sk = render.load_clip(stem), capture.skeleton()
    pos, rot = verdicts.torch.as_tensor(clip.pos[frame:frame + 1]), verdicts.torch.as_tensor(clip.rot[frame:frame + 1])
    return {b: [verdicts.geom_to_world(g, pos[:, i], rot[:, i]) for g in sk.geoms[b]] for i, b in enumerate(sk.names)}


def test_pose_names_resolve_to_the_most_specific_pose():
    cases = {
        "Parsva Bakasana (Side Crow)": "Side_Crane_Crow_Pose_or_Parsva_Bakasana",
        "Kakasana": "Crane_Crow_Pose_or_Bakasana",
        "Tripod headstand (Sirsasana II)": "Supported_Headstand_pose_or_Salamba_Sirsasana",
        "Shirshasana": "Supported_Headstand_pose_or_Salamba_Sirsasana",
        "Setu Bandha Sarvangasana": "Bridge_Pose_or_Setu_Bandha_Sarvangasana",
        "Salamba Sarvangasana": "Supported_Shoulderstand_pose_or_Salamba_Sarvangasana",
        "Mountain pose (Tadasana)": "standing",
        "Warrior 2": "Warrior_II_Pose_or_Virabhadrasana_II",
        "Virabhadrasana III": "Warrior_III_Pose_or_Virabhadrasana_III",
        "Dolphin plank (Makara Adho Mukha Svanasana)": "Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana",
        "Adho Mukha Svanasana": "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana",
        "Pincha Mayurasana": "Feathered_Peacock_Pose_or_Pincha_Mayurasana",
        "Ardha Pincha Mayurasana": "Dolphin_Pose_or_Ardha_Pincha_Mayurasana",
        "Mayurasana": "Peacock_Pose_or_Mayurasana",
        "Vrikshasana": "Tree_Pose_or_Vrksasana",
        "Low plank": "Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana",
        "Boat pose (Navasana)": None,
    }
    assert {n: verdicts.resolve_pose(n) for n in cases} == cases
    for pose, aliases in verdicts.POSE_ALIASES.items():
        assert all(verdicts.resolve_pose(a) == pose for a in aliases), pose
    names = {h["name"] for c in ids.load_manifest()["clips"] for h in c["holds"]}
    assert all(verdicts.pose_truth(n) for n in names)
    assert verdicts.pose_truth("Plow_Pose_or_Halasana_h2") == "Plow_Pose_or_Halasana"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _oracle(truth, mirror=False) -> dict:
    """An answer that states the truth wherever there is one (sides swapped with ``mirror``)."""
    words = {v: k for k, v in verdicts.PARTS.items()}
    swap = (lambda z: z.replace("L_", "R_", 1) if z.startswith("L_") else z.replace("R_", "L_", 1)) \
        if mirror else (lambda z: z)
    answer = review.stub_answer()
    for z in ZONE_ORDER:
        src = swap(z)
        state = lambda t: {1: "touching", 0: "off_floor"}.get(t, "cannot_tell")  # noqa: E731
        answer["floor"][words[z]] = {"avatar": state(truth["avatar"][src]),
                                     "markers": state(truth["human"].get(src)), "panels": []}
    answer["body_body"] = {"answer": "listed", "contacts": [
        {"part_a": words[a], "part_b": words[b], "certainty": "clear", "panels": []}
        for (a, b), t in truth["pairs"].items() if t == 1]}
    answer["pose"]["candidates"] = [{"name": verdicts.POSE_ALIASES[truth["pose"]][0], "confidence": "high"}]
    return answer


def test_scoring_an_oracle_and_a_mirrored_reviewer(run):
    rows = [r for t in run.truths.values() for r in verdicts.score(_oracle(t), t)]
    by = {cls: verdicts.class_metrics([r for r in rows if r["class"] == cls], kind)
          for cls, kind in verdicts.CLASSES.items()}
    for cls in verdicts.CLASSES:
        assert by[cls]["precision"] == 1.0 and by[cls]["answered_rate"] == 1.0, cls
    assert by["float"]["positives"] == by["float"]["claims"] == 4  # split L_FOOT, R_HAND; crow L_FOOT; head
    assert by["body_body"]["recall"] == 1.0 and by["pose_identity"]["items"] == 3
    assert not by["float"]["evidence"] and not by["pose_identity"]["evidence"]  # perfect, but under 30 items

    mirrored = [r for t in run.truths.values() for r in verdicts.score(_oracle(t, mirror=True), t)]
    lr = verdicts.class_metrics([r for r in mirrored if r["class"] == "left_right"], "match")
    assert lr["answered"] > 0 and lr["precision"] == 0.0
    stub = [r for t in run.truths.values() for r in verdicts.score(review.stub_answer(), t)]
    assert all(r["claim"] is None for r in stub)


def test_admission_rule():
    rows = [{"stratum": "x", "truth": 1, "claim": 1, "correct": True}] * 27 + \
           [{"stratum": "x", "truth": 0, "claim": 1, "correct": False}] * 3
    m = verdicts.class_metrics(rows, "detect")
    assert m["precision"] == 0.9 and m["items"] == 30 and m["evidence"]
    assert not verdicts.class_metrics(rows[:29], "detect")["evidence"]              # 29 items
    abstain = [{"stratum": "x", "truth": 1, "claim": None, "correct": None}] * 31
    assert not verdicts.class_metrics(rows + abstain, "detect")["evidence"]         # answered 30 / 61
    # the body-body trap (render_v3, high effort): 4 perfect positive claims and 90 implicit "no"s make
    # a 1.000 precision at a 0.55 answered rate, over 4 claims; the claims floor keeps it advisory
    few = [{"stratum": "x", "truth": 1, "claim": 1, "correct": True}] * 4 + \
          [{"stratum": "x", "truth": 1, "claim": 0, "correct": False}] * 48 + \
          [{"stratum": "x", "truth": 1, "claim": None, "correct": None}] * 42
    m = verdicts.class_metrics(few, "detect")
    assert m["precision"] == 1.0 and m["answered_rate"] >= 0.5 and m["support"] == 4 and not m["evidence"]
    # a two-valued class is only as good as its worse claim value
    binary = [{"stratum": "x", "truth": 1, "claim": 1, "correct": True}] * 40 + \
             [{"stratum": "x", "truth": 1, "claim": 0, "correct": False}] * 3 + \
             [{"stratum": "x", "truth": 0, "claim": 0, "correct": True}] * 3
    m = verdicts.class_metrics(binary, "binary")
    assert m["accuracy"] == round(43 / 46, 4) and m["precision"] == 0.5 and m["support"] == 6 and not m["evidence"]
    assert verdicts.wilson_low(27, 30) == 0.7438


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def _main_args(run, ledger):
    return ["--ledger-dir", str(ledger), "--audit", str(run.audit_dir), "--index-root", str(run.kw["index_root"]),
            "--review-root", str(run.kw["review_root"])]


def test_dry_run_with_a_stub_reviewer_is_resumable_and_calibrates(run, tmp_path):
    ledger = tmp_path / "ledger"
    assert review.main(["--dry-run", *_main_args(run, ledger)]) == 0
    records = verdicts.read_ledger(ledger)
    assert len(records) == len(HOLDS) and {r["status"] for r in records} == {"valid"}
    assert {r["reviewer"]["model"] for r in records} == {"stub"} and all(r["n"] == 1 for r in records)
    assert [p.name for p in sorted(ledger.iterdir())] == sorted(f"{r['packet_id']}.A.1.json" for r in records)
    assert review.main(["--dry-run", *_main_args(run, ledger)]) == 0     # resumed: nothing to do
    assert len(verdicts.read_ledger(ledger)) == len(HOLDS)

    out = tmp_path / "calibration.json"
    args = ["--ledger-dir", str(ledger), "--audit", str(run.audit_dir), "--store-dir", str(run.out / "store"),
            "--calibration", str(run.cal_path), "--out", str(out), "--items-root", str(tmp_path / "items")]
    assert verdicts.main(args) == 0
    table = json.loads(out.read_text())
    assert table["holds"] == len(HOLDS) and len(table["reviewers"]) == 1
    rv = table["reviewers"][0]
    assert rv["evidence"] == [] and rv["calls"] == {"valid": len(HOLDS)}
    assert all(m["answered"] == 0 for m in rv["classes"].values())
    assert table["generator"]["module"] == verdicts.MODULE and table["inputs"]


FAKE_CLI = """#!{python}
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("0.0.0 (fake)")
    sys.exit(0)
with open({log!r}, "a") as f:
    f.write(json.dumps({{"cwd": os.getcwd(), "argv": args, "files": sorted(os.listdir("."))}}) + "\\n")
schema = json.loads(args[args.index("--json-schema") + 1])
answer = {{"pose": {{"description": "cannot_tell", "candidates": []}},
          "floor": {{p: {{"avatar": "touching", "markers": "touching", "panels": ["2b"]}}
                    for p in schema["properties"]["floor"]["required"]}},
          "discrepancies": [], "body_body": {{"answer": "cannot_tell", "contacts": []}}, "implausible": []}}
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "num_turns": 3,
                  "total_cost_usd": 0.0125, "session_id": "fake", "permission_denials": [],
                  "usage": {{"output_tokens": 42, "output_tokens_details": {{"thinking_tokens": 7}}}},
                  "structured_output": answer}}))
"""


def test_the_cli_runs_blind_in_the_packet_directory(run, tmp_path):
    log = tmp_path / "calls.jsonl"
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLI.format(python=sys.executable, log=str(log)))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    entry = run.entries[f"{CROW}@439"]
    reviewer = review.Reviewer(cli=str(fake), effort="medium")
    out = review.run([entry], reviewer, review.call_claude, ledger_dir=tmp_path / "ledger",
                     review_root=run.kw["review_root"], log=lambda *_: None)
    assert out["status"] == {"valid": 1} and out["failures"] == [] and out["spent"] == 0.0125
    call = json.loads(log.read_text())
    d = packets.packet_dir(entry, run.kw["review_root"])
    assert call["cwd"] == str(d.resolve()) and call["files"] == sorted([*entry["images"], "packet.json"])
    argv = call["argv"]
    assert argv[0] == "-p" and argv[1] == verdicts.prompt_path().read_text()   # before the variadic flags
    for flag in ("--restricted", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == "Read" and argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == verdicts.load_schema()
    record = verdicts.read_ledger(tmp_path / "ledger")[0]
    assert record["reviewer"]["cli_version"] == "0.0.0 (fake)" and record["reviewer"]["key"] == reviewer.key
    assert record["usage"]["thinking_tokens"] == 7 and record["prompt"]["sha256"] == ids.sha256_file(verdicts.prompt_path())
    assert record["packet"]["images"] == entry["images"] and record["hold_id"] == f"{CROW}@439"
    # the key moves with what the reviewer is asked, not with how the call is capped
    assert review.Reviewer(effort="high").key != reviewer.key == review.Reviewer(effort="medium",
                                                                                per_call_max_usd=9).key
    assert review.Reviewer().effort == "high"  # Step 4's calibration chose it


def test_check_blind_refuses_a_packet_that_could_leak(run, tmp_path, monkeypatch):
    entry = run.entries[f"{HEADSTAND_B}@1093"]
    src = packets.packet_dir(entry, run.kw["review_root"])
    assert review.check_blind(entry, run.kw["review_root"])[0] == src.resolve()

    def copy_to(root):
        dst = root / entry["render_v"] / entry["packet_id"]
        shutil.copytree(src, dst)
        return dst

    leaky = tmp_path / "leaky"
    dst = copy_to(leaky)
    (leaky / "CLAUDE.md").write_text("names")
    with pytest.raises(ValueError, match="CLAUDE.md"):
        review.check_blind(entry, leaky)
    stray = tmp_path / "stray"
    copy_to(stray).joinpath("notes.txt").write_text("hint")
    with pytest.raises(ValueError, match="notes.txt"):
        review.check_blind(entry, stray)
    edited = tmp_path / "edited"
    img = copy_to(edited) / "img_1.png"
    img.write_bytes(img.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        review.check_blind(entry, edited)
    repo = tmp_path / "repo"
    copy_to(repo / "review")
    monkeypatch.setattr(ids, "REPO", repo)
    with pytest.raises(ValueError, match="inside the repo"):
        review.check_blind(entry, repo / "review")
    assert dst.exists()


def test_the_ledger_never_overwrites(tmp_path):
    record = {"packet_id": "p" * 40, "pass": "A", "status": "valid"}
    paths = [verdicts.write_verdict(record, tmp_path) for _ in range(3)]
    assert [p.name for p in paths] == [f"{'p' * 40}.A.{n}.json" for n in (1, 2, 3)]
    assert [r["n"] for r in verdicts.read_ledger(tmp_path)] == [1, 2, 3]
    todo, blocked, current = review.pending(
        [{"packet_id": "p" * 40, "pass": "A", "item_id": "x"}],
        [{"packet_id": "p" * 40, "pass": "A", "reviewer": {"key": "k"}, "status": s} for s in ("error", "invalid")],
        "k")
    assert todo == [] and len(blocked) == 1 and current == 0


# --------------------------------------------------------------------------- #
# The calibration table
# --------------------------------------------------------------------------- #
CALIBRATION = verdicts.CALIBRATION_DIR / f"{render.RENDER_V}.json"


def test_the_current_render_v_has_a_calibration_table():
    assert CALIBRATION.exists(), f"run the reviewer and `-m reference_curation.verdicts` for {render.RENDER_V}"


@pytest.mark.parametrize("render_v", sorted(p.stem for p in verdicts.CALIBRATION_DIR.glob("render_v*.json")))
def test_calibration_table_replays_the_ledger(run, render_v):
    """Every committed table is what the committed ledger says: a rebuild re-scores, never re-queries."""
    path = verdicts.CALIBRATION_DIR / f"{render_v}.json"
    stored = json.loads(path.read_text())
    table, rows, failures = verdicts.calibrate(verdicts.read_ledger(), run.aud.records, render_v=render_v,
                                               store_dir=run.out / "store", calibration=run.cal)
    assert failures == [] and table["audit_ids"] == stored["audit_ids"] == [run.aud.audit_id]
    assert json.loads(json.dumps(table["reviewers"])) == stored["reviewers"]
    assert stored["rule"] == verdicts.RULE and stored["truth"] == verdicts.TRUTH
    for rv in stored["reviewers"]:
        assert rv["packets"] >= 40 and rv["model"] == review.MODEL
        strata = {s for m in rv["classes"].values() for s in m["by_stratum"]}
        assert {"arm_balance", "inversion", "float_5_15", "float_2_5"} <= strata  # the spike's failures
        for cls, m in rv["classes"].items():
            admit = (m["precision"] is not None and m["precision"] >= 0.9 and m["answered_rate"] >= 0.5
                     and m["items"] >= 30 and m["support"] >= 30)
            assert m["evidence"] == admit and (cls in rv["evidence"]) == admit, cls
    assert os.path.getsize(path) < 200_000
