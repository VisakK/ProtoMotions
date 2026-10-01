# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BodyFix Step 4b: render_v4 packets, the Pass-A recalibration, B6, Pass C on render_c2 and gate v2, against the
committed records and the shared ledger (re-scored, never re-queried).

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_step4_review.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from reference_curation import calibrate_v4  # first: render (OSMesa)
from reference_curation import capture, capture_v4, edits as E, edits_v2, ids, packets, packets_v4, render_v4, verdicts

TABLE = calibrate_v4.CALIBRATION_DIR / f"{packets_v4.RENDER_V}.json"

pytestmark = pytest.mark.skipif(not packets_v4.index_path("A").exists(), reason="no render_v4 packets on disk")


@pytest.fixture(scope="module")
def aud():
    return packets.load_audit(calibrate_v4.audit_dir())


def test_the_fixed_texts_use_no_word_of_any_clip_name():
    text = json.dumps([packets_v4.LEGEND, render_v4.palette_legend(), packets_v4.UNITS_B, edits_v2.LEGEND,
                       E.IMAGE_TEXT]).lower()
    assert sorted(t for t in E.name_tokens() if t in text) == []


def test_every_v4_pass_a_packet_is_blind(aud):
    index = packets.read_index(packets_v4.index_path("A"))
    assert len(index) == 101
    manifest = calibrate_v4.LABELS_V11 / "holds.yaml"
    for e in index.values():
        p = json.loads((packets.packet_dir(e) / "packet.json").read_text())
        assert set(p) == {"images", "legend", "packet_id", "pass", "render_v", "schema_version"}
        item = packets_v4.item_a(aud.records[e["hold_id"]], e["variant"], e["lift_m"])
        assert packets.leaks(json.dumps({k: p[k] for k in ("images", "legend", "pass", "render_v", "schema_version")}),
                             aud.records[e["hold_id"]], item, manifest) == []


def test_a_lift_floats_every_part_by_exactly_the_lift_and_moves_no_pair(aud):
    cal = capture.load_calibration()
    hid = "220923_Crane_Crow_Pose_or_Bakasana_-a@439"
    r = aud.records[hid]
    rec = capture_v4.load(r["stem"], rebuild=False)
    f = r["window"]["frame_hold"]
    t0 = packets_v4.truth(r, rec, cal, f, "reference")
    t1 = packets_v4.truth(r, rec, cal, f, "lift", 0.06)
    assert all(abs(t1["avatar_cm"][z] - t0["avatar_cm"][z] - 6.0) < 1e-6 for z in t0["avatar_cm"])
    assert t1["pair_gap_cm"] == t0["pair_gap_cm"] and t1["human"] == t0["human"]
    assert t0["avatar"]["L_HAND"] == 1 and t1["avatar"]["L_HAND"] == 0


def test_the_render_v4_table_replays_the_ledger(aud):
    stored = json.loads(TABLE.read_text())
    table, _, failures = calibrate_v4.calibrate(verdicts.read_ledger(), aud.records,
                                                packets.read_index(packets_v4.index_path("A")))
    assert failures == []
    assert json.loads(json.dumps(table["reviewers"])) == stored["reviewers"]
    (rv,) = stored["reviewers"]
    assert rv["packets"] == 101 and rv["packets_by_variant"] == {"reference": 60, "fit": 17, "lift": 24}
    assert set(rv["evidence"]) == {"floor_avatar", "floor_avatar_far", "floor_human", "float", "left_right"}
    for cls, m in rv["classes"].items():
        admit = (m["precision"] is not None and m["precision"] >= 0.9 and m["answered_rate"] >= 0.5
                 and m["items"] >= 30 and m["support"] >= 30)
        assert m["evidence"] == admit, cls
    fl = rv["classes"]["float"]
    assert fl["precision"] == 1.0 and fl["recall"] == 1.0
    assert fl["recall_by_size"]["2-5 cm"]["caught"] == fl["recall_by_size"]["2-5 cm"]["floats"] >= 50
    assert fl["recall_by_size"]["5-15 cm"]["caught"] == fl["recall_by_size"]["5-15 cm"]["floats"] >= 40


def test_render_v4_tables_live_outside_step_4s_replay_glob():
    assert not list(verdicts.CALIBRATION_DIR.glob("render_v4*.json"))
    assert TABLE.parent.name == "pass_a_v4"
