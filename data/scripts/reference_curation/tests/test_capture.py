# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the capture evidence store (BUILD_PLAN Step 1) and the ids it is keyed by.

They run on the real clips and assert numbers the review already measured
(``expert_revist/reference_curation_review_2026_09_28/README.MD`` §3.2 and
``output_capture_marker_audit.txt``), so a regression shows up as a wrong number. The module
fixture builds the whole 60-clip store once into a temp dir; nothing under ``output/`` or
``data/reference_curation/`` is touched.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reference_curation import capture, ids

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
SIDE_CROW_C = "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c"
STANDING_SPLIT = "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a"
PEACOCK = "220923_Peacock_Pose_or_Mayurasana_-a"
HEADSTAND_B = "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b"
KOUNDINYA_B = "220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b"
BIG_TOE_C = "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c"  # its MoSh pkl is corrupt
DOWNDOG_A = "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a"    # ungated port: no column 2
ADMITTED = {"L_FOOT", "R_FOOT", "HEAD", "L_HAND", "R_HAND"}

pytestmark = pytest.mark.skipif(
    not (ids.MOYO_DATA / "mosh").exists() or not ids.SHIPPED_DIR.exists(),
    reason="the MOYO MoSh fits or the shipped ftC clips are not on disk")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    out = tmp_path_factory.mktemp("capture")
    start = time.time()
    cal, metas, failures = capture.build_all(ids.manifest_stems(), out / "store", out / "capture_v1.json")
    return SimpleNamespace(dir=out / "store", cal=cal, metas={m["stem"]: m for m in metas},
                           failures=failures, seconds=time.time() - start)


def _load(store, stem):
    return capture.load(stem, store.dir, store.cal)


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #
def test_name_mapping_keeps_moyo_punctuation():
    assert ids.recording_id(CROW) == "220923_yogi_body_hands_03596_Crane_(Crow)_Pose_or_Bakasana_-a"
    assert ids.recording_id(SIDE_CROW_C) == "220923_yogi_body_hands_03596_Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_-c"
    for stem in (CROW, SIDE_CROW_C, STANDING_SPLIT):
        assert ids.stem_of_recording(ids.recording_id(stem)) == stem
        assert ids.mosh_path(stem).name == f"{ids.recording_id(stem)}_stageii.pkl"
    # The 221004 session reuses the pose names; its date keeps it apart.
    assert ids.stem_of_recording("221004_yogi_nexus_body_hands_03596_Crane_(Crow)_Pose_or_Bakasana_-a") \
        == "221004_Crane_Crow_Pose_or_Bakasana_-a"
    assert ids.mosh_path("221004_Crane_Crow_Pose_or_Bakasana_-a") != ids.mosh_path(CROW)
    assert ids.mosh_path("220923_Crane_Crow_Pose_or_Bakasana_hold") is None  # hand-cut subclip
    with pytest.raises(ValueError):
        ids.stem_of_recording("Crane_(Crow)_Pose_or_Bakasana_-a")
    stems = ids.manifest_stems()
    assert len(stems) == 60 and len({ids.mosh_path(s) for s in stems} - {None}) == 60
    assert ids.pressure_paths(CROW)[0].parent == ids.PRESSURE_DIRS[0]
    assert ids.split_clip_name(f"{CROW}_x3s") == (CROW, 3.0) and ids.split_clip_name(CROW) == (CROW, 0.0)


def test_hold_ids_map_every_variant_back_to_source_frames():
    ext = ids.load_manifest(ids.EXTENDED_MANIFEST)
    source = {c["stem"]: c for c in ids.load_manifest(ids.DEFAULT_MANIFEST)["clips"]}
    checked = 0
    for clip in ext["clips"]:
        src_holds = source[clip["source_stem"]]["holds"]
        assert len(clip["holds"]) == len(src_holds)
        for hold, src in zip(clip["holds"], src_holds):
            assert ids.hold_id(clip["stem"], hold["frame_hold"]) == f"{clip['source_stem']}@{src['frame_hold']}"
            checked += 1
    assert len(ext["clips"]) == 180 and checked > 0
    assert ids.hold_id(f"{CROW}_x3s", 1151) == f"{CROW}@971"  # the last hold shifts by the 180 inserted frames
    assert ids.parse_hold_id(ids.hold_id(CROW, 439)) == (CROW, 439)
    with pytest.raises(ValueError):
        ids.hold_id(f"{CROW}_x3s", -1)


def test_provenance_block():
    p = ids.provenance(1, "reference_curation.capture", capture.__file__, [ids.MJCF, ids.mosh_path(CROW)])
    assert p["schema_version"] == 1 and p["generator"]["module"] == "reference_curation.capture"
    assert len(p["generator"]["sha256"]) == 64
    assert p["inputs"]["data/assets/smpl/smpl_yogi03596_lowtorque.xml"] == ids.sha256_file(ids.MJCF)
    assert str(ids.mosh_path(CROW)) in p["inputs"]  # outside the repo: absolute


# --------------------------------------------------------------------------- #
# Pure pieces
# --------------------------------------------------------------------------- #
def test_marker_map_covers_all_73_markers():
    fit, status, _ = capture.load_mosh(CROW)
    assert status == "ok" and len(fit["labels"]) == capture.NUM_MARKERS
    used = [m for ms in capture.ZONE_MARKERS.values() for m in ms]
    assert set(used) == set(fit["labels"])
    shared = {m for m in used if used.count(m) > 1}
    assert shared == {s + m for s in "LR" for m in ("ANK", "KNE", "ELB", "ELBIN", "IWR", "OWR")}


def test_contact_state_is_a_schmitt_trigger():
    h = np.array([0.05, 0.02, 0.05, 0.08, 0.05, 0.02, 0.9, 0.05, 0.08])
    known = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    state = capture.contact_state(h, known, touch=0.04, separation=0.065)
    # band before any decisive frame -> unknown; band keeps the last state; memory survives unknown frames
    assert state.tolist() == [-1, 1, 1, 0, 0, 1, -1, 1, 0]


# --------------------------------------------------------------------------- #
# The store, on the real corpus
# --------------------------------------------------------------------------- #
def test_full_corpus_builds_in_budget(store):
    assert store.failures == []
    assert len(store.metas) == 60
    assert store.seconds < 600, store.seconds
    assert sum(m["capture_available"] for m in store.metas.values()) == 59


def test_mosh_and_motion_frame_counts_match(store):
    statuses = {m["capture_status"] for m in store.metas.values()}
    assert statuses == {"ok", "unreadable"}  # a frame or fps mismatch would be its own status
    assert store.metas[STANDING_SPLIT]["num_frames"] == 1036  # README §3.2
    assert store.metas[HEADSTAND_B]["num_frames"] == 1790
    rec = _load(store, STANDING_SPLIT)
    fit, _, _ = capture.load_mosh(STANDING_SPLIT)
    assert fit["obs"].shape[0] == rec["ground_state"].shape[0] == 1036


def test_thresholds_match_the_preview(store):
    zones = store.cal["zones"]
    assert {z for z in zones if zones[z]["admitted"]} == ADMITTED
    preview_cm = {"L_FOOT": (4.3, 6.8), "R_FOOT": (4.3, 6.8), "L_HAND": (2.5, 5.0),
                  "R_HAND": (2.5, 5.0), "HEAD": (3.7, 6.2)}
    for zone, (touch, separation) in preview_cm.items():
        assert abs(100 * zones[zone]["touch_m"] - touch) <= 0.3, (zone, zones[zone]["touch_m"])
        assert abs(100 * zones[zone]["separation_m"] - separation) <= 0.3, (zone, zones[zone]["separation_m"])
    # Loaded thigh markers read 14-18 cm: markers there do not sit on the floor-facing surface.
    assert zones["L_THIGH"]["p50"] > 0.10 and not zones["L_THIGH"]["admitted"]
    rec = _load(store, CROW)
    for zone in ADMITTED:
        assert rec.meta["thresholds"][zone] == {k: zones[zone][k] for k in ("touch_m", "separation_m", "admitted")}
    assert (rec["ground_state"][:, [rec.zi(z) for z in capture.ZONE_ORDER if z not in ADMITTED]] == -1).all()


def test_standing_split_standing_foot_touches_while_the_avatar_floats(store):
    rec = _load(store, STANDING_SPLIT)
    f, z = rec.frame(9.0), rec.zi("L_FOOT")
    assert rec["ground_state"][f, z] == 1
    assert rec["avatar_min_z"][f, z] == pytest.approx(0.116, abs=0.005)
    assert np.median(rec["avatar_min_z"][rec.frame(4.3):rec.frame(10.9), z]) == pytest.approx(0.116, abs=0.001)
    assert rec["marker_min_z"][f, z] == pytest.approx(0.030, abs=0.003)  # LTOE 3.0 cm
    assert "L_FOOT" in rec.support(f)


def test_crow_left_foot_lifts_between_8_and_9_s(store):
    rec = _load(store, CROW)
    left, right = rec["ground_state"][:, rec.zi("L_FOOT")], rec["ground_state"][:, rec.zi("R_FOOT")]
    assert (left[: rec.frame(8.0) + 1] == 1).all()
    assert (left[rec.frame(9.0): rec.frame(11.5) + 1] == 0).all()
    assert (right[rec.frame(6.0): rec.frame(12.0) + 1] == 0).all()
    lift = int(np.argmax(left == 0)) / rec.fps
    assert 8.0 < lift < 8.5, lift  # label says hands-only from 7.08 s
    f = rec.frame(8.5)  # the review's lift timeline: 657 N total, 644 on the hands, 14 unexplained
    hands = rec["mat_zone_load"][f, [rec.zi("L_HAND"), rec.zi("R_HAND")]].sum()
    assert (rec["mat_total"][f], hands, rec["mat_unexplained"][f]) == pytest.approx((657, 644, 14), abs=1)


def test_peacock_is_toes_assisted(store):
    rec = _load(store, PEACOCK)
    window = slice(rec.frame(9.0), rec.frame(10.3) + 1)
    for zone in ("L_FOOT", "R_FOOT"):
        assert (rec["ground_state"][window, rec.zi(zone)] == 1).all(), zone
    hold = slice(rec.frame(9.0), rec.frame(10.32) + 1)
    feet = rec["mat_zone_load"][hold][:, [rec.zi("L_FOOT"), rec.zi("R_FOOT")]].sum(-1)
    assert np.median(feet) == pytest.approx(69, abs=1)  # README §3.2: 69 N of 614 N
    assert np.median(rec["mat_total"][hold]) == pytest.approx(614, abs=1)


def _exemplar_audit(zone_markers):
    """The review's pooled-threshold check at the family-hold exemplars (README §3.2)."""
    missed, lifted, n = [], [], 0
    for clip in ids.load_manifest()["clips"]:
        family = [h for h in clip["holds"] if h.get("extend")]
        fit, status, _ = capture.load_mosh(clip["stem"])
        if not family or status != "ok":
            continue
        min_z = capture.marker_arrays(fit["obs"], fit["sim"], fit["labels"], zone_markers)[0]
        for hold in family:
            n += 1
            f, ground = int(hold["frame_hold"]), {p[:-2] for p in hold["pairs"] if p.endswith(":G")}
            for zone in ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND"):
                low = np.nanmin(min_z[max(0, f - 3): f + 4, capture.ZONE_ORDER.index(zone)])
                if zone not in ground and low < 0.035:
                    missed.append((clip["stem"], f))
                if zone in ground and low > 0.06:
                    lifted.append((clip["stem"], f))
    return n, missed, lifted


def test_reproduces_the_review_exemplar_audit():
    n, missed, lifted = _exemplar_audit(capture.REVIEW_ZONE_MARKERS)
    assert n == 69
    assert (len(missed), len(set(missed))) == (15, 11)
    assert (len(lifted), len(set(lifted))) == (4, 3)
    # The store's map adds the wrist markers: Bridge -a's hands then read their wrists at 3.5-4.4 cm.
    _, missed, lifted = _exemplar_audit(capture.ZONE_MARKERS)
    assert (len(missed), len(set(missed))) == (15, 11)
    assert [stem for stem, _ in lifted] == ["220923_Side_Plank_Pose_or_Vasisthasana_-a"]


def test_reproduces_the_review_support_hover_count(store):
    heights = []  # README §3.1: hold-median height of every labelled ground support
    for clip in ids.load_manifest()["clips"]:
        rec = _load(store, clip["stem"])
        last = rec["avatar_min_z"].shape[0] - 1
        for hold in clip["holds"]:
            f0, f1 = (min(last, rec.frame(float(hold[k]))) for k in ("t_start", "t_end"))
            for pair in hold["pairs"]:
                if pair.endswith(":G"):
                    heights.append(np.median(rec["avatar_min_z"][f0:f1 + 1, rec.zi(pair[:-2])]))
    heights = np.asarray(heights)
    assert len(heights) == 833
    assert [int((heights > cm / 100).sum()) for cm in (1, 2, 3, 5)] == [467, 367, 237, 120]


def test_unreadable_fit_is_unavailable_not_a_crash(store):
    rec = _load(store, BIG_TOE_C)
    assert rec.meta["capture_available"] is False and rec.meta["capture_status"] == "unreadable"
    assert (rec["ground_state"] == -1).all()
    assert np.isnan(rec["marker_min_z"]).all() and np.isnan(rec["marker_resid"]).all()
    assert np.isfinite(rec["avatar_min_z"]).all() and rec.meta["mat_available"]


def test_record_fields_and_validity_columns(store):
    rec = _load(store, CROW)
    T, Z = 972, 15
    for name, shape in (("marker_min_z", (T, Z)), ("marker_resid", (T, Z)), ("avatar_min_z", (T, Z)),
                        ("mat_total", (T,)), ("mat_cop", (T, 2)), ("mat_valid_cov", (T,)),
                        ("mat_valid_body", (T,)), ("mat_valid_share", (T,)), ("mat_zone_load", (T, Z)),
                        ("mat_unexplained", (T,)), ("attr_visible", (T, Z)), ("ground_state", (T, Z))):
        assert rec[name].shape == shape, name
    assert rec["ground_state"].dtype == np.int8 and rec["attr_visible"].dtype == bool
    assert set(rec.meta["arrays"]) == set(rec.arrays)
    np.testing.assert_allclose(rec["mat_total"] - rec["mat_zone_load"].sum(-1), rec["mat_unexplained"], atol=0.05)
    assert (rec["mat_valid_cov"] != rec["mat_valid_body"]).any()  # separate columns, not one merged gate
    assert rec.meta["mat_share_available"] and np.isfinite(rec["mat_valid_share"]).all()
    ungated = _load(store, DOWNDOG_A)
    assert not ungated.meta["mat_share_available"] and np.isnan(ungated["mat_valid_share"]).all()
    meta = json.loads((store.dir / f"{CROW}.json").read_text())
    assert meta["schema_version"] == capture.SCHEMA_VERSION
    assert meta["generator"]["sha256"] == ids.sha256_file(capture.__file__)
    assert meta["inputs"][str(ids.mosh_path(CROW))] == ids.sha256_file(ids.mosh_path(CROW))
    assert meta["calibration"]["id"] == store.cal["id"]


def test_attr_visible_is_measured_on_the_pose_the_attribution_saw(store):
    repaired = store.metas[KOUNDINYA_B]  # repaired for ft_c; its pressure port keeps the grounded pose
    assert repaired["attr_kinematics"] == "pressure_port" and not repaired["attr_pose_matches_shipped"]
    assert repaired["attr_visible_vs_shipped_disagree"] == 6
    assert sum(not m["attr_pose_matches_shipped"] for m in store.metas.values()) == 8
    assert store.metas[STANDING_SPLIT]["attr_pose_matches_shipped"]


def test_markers_register_to_the_clip_frame():
    xyz, labels = capture.markers(STANDING_SPLIT)
    motion = torch.load(ids.motion_path(STANDING_SPLIT), map_location="cpu", weights_only=False)
    ankle = motion["rigid_body_pos"][:, capture.skeleton().names.index("L_Ankle"), :2].numpy()
    lank = xyz[:, labels.index("LANK"), :2]
    registered = np.median(np.linalg.norm(lank - ankle, axis=-1))
    raw = np.median(np.linalg.norm(lank - np.asarray(capture.VICON_TO_CLIP_XY) - ankle, axis=-1))
    assert registered < 0.08 < 0.25 < raw, (registered, raw)


def test_load_rebuilds_a_record_from_another_calibration(store):
    path = store.dir / f"{PEACOCK}.json"
    meta = json.loads(path.read_text())
    meta["calibration"]["id"] = "stale"
    path.write_text(json.dumps(meta))
    rec = _load(store, PEACOCK)
    assert rec.meta["calibration"]["id"] == store.cal["id"]
    assert json.loads(path.read_text())["calibration"]["id"] == store.cal["id"]


def test_variants_are_rejected():
    with pytest.raises(ValueError, match="x0 clips only"):
        capture.measure(f"{CROW}_x3s")
