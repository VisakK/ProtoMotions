# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Pass C (BUILD_PLAN Step 8): the controls are what their truth says, the fixed texts leak no
name, the validator and the acceptance rule behave, and the committed calibration table replays the ledger.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from reference_curation import capture, edits, ids, retarget as rt, statics


@pytest.fixture(scope="module")
def labels():
    try:
        return statics.load_labels(statics.default_labels_dir())
    except FileNotFoundError:
        pytest.skip("no labels v1 folder")


def test_fixed_texts_use_no_word_of_any_clip_name():
    text = edits.fixed_texts()
    assert sorted(t for t in edits.name_tokens() if t in text) == []
    assert "side" in edits.name_tokens() and "hold" in edits.name_tokens()   # the traps are in the set


def test_controls_are_what_their_truth_says(labels):
    items = {i["kind"]: i for i in edits.control_items(labels, {k: 1 for k in edits.CONTROLS})}
    for kind in ("lift", "sink"):
        it = items[kind]
        pos, rot, _ = edits.load_motion(it["stem"], ids.SHIPPED_DIR)
        p2, _ = edits.control_motion(kind, it["stem"], it["frame"])
        dz = p2[..., 2] - pos[..., 2]
        assert np.allclose(dz, edits.LIFT_M if kind == "lift" else -edits.SINK_M)
    from scipy.spatial.transform import Rotation

    it = items["kink"]
    sk = rt.skeleton()
    _, rot0, _ = edits.load_motion(it["stem"], ids.SHIPPED_DIR)
    _, rot1 = edits.control_motion("kink", it["stem"], it["frame"])
    f = it["frame"]

    def local(rot, b):
        return Rotation.from_quat(rot[f, sk.parents[b]]).inv() * Rotation.from_quat(rot[f, b])

    turned = [np.degrees((local(rot0, b).inv() * local(rot1, b)).magnitude()) for b in range(1, sk.num_bodies)]
    assert max(turned) > 70 and sorted(turned)[-2] < 1e-3    # one joint turned 80 deg past its range, only one
    it = items["swap"]
    fam = lambda s: s.split("_", 1)[1].rsplit("_-", 1)[0].rsplit("-", 1)[0]   # noqa: E731
    assert fam(it["other"][0]) != fam(it["stem"])
    p2, r2 = edits.control_motion("swap", it["stem"], it["frame"], tuple(it["other"]))
    low = capture.body_min_z(p2[it["frame"]][None], r2[it["frame"]][None]).min()
    assert low == pytest.approx(0.005, abs=1e-6)
    assert {i["truth"]["same_pose"] for i in items.values()} == {"yes", "no"}
    assert items["lift"]["truth"]["closer"] != edits.modified_side(items["lift"])


def _packet(tags=("1a", "1b")):
    return {"images": [{"file": "img_1.png", "panels": list(tags)}]}


def _answer(**kw):
    a = {"same_pose": {"answer": "yes", "reason": "x"}, "closer_to_human": "equal", "more_natural": "equal",
         "artefacts": [], "floating_parts": {"A": [], "B": []}, "summary": "x"}
    a.update(kw)
    return a


def test_validator_rejects_bad_panels_and_schema():
    ok, _ = edits.validate(_answer(), _packet())
    assert ok == []
    bad, _ = edits.validate(_answer(artefacts=[{"avatar": "A", "kind": "through_floor", "part": "left foot",
                                               "severity": "major", "panels": ["9z"]}]), _packet())
    assert any("9z" in e for e in bad)
    bad, _ = edits.validate(_answer(closer_to_human="C"), _packet())
    assert bad and bad[0].startswith("schema")


def test_acceptance_rule():
    entry = {"a_is": "first"}          # A shipped, B the retarget
    art = lambda av, kind="through_floor": {"avatar": av, "kind": kind, "part": "left foot",  # noqa: E731
                                            "severity": "major", "panels": ["1a"]}
    assert edits.accept(_answer(closer_to_human="B"), entry)["accepted"]
    assert not edits.accept(_answer(same_pose={"answer": "no", "reason": "x"}), entry)["accepted"]
    assert not edits.accept(_answer(closer_to_human="A"), entry)["accepted"]
    assert not edits.accept(_answer(artefacts=[art("B")]), entry)["accepted"]
    assert edits.accept(_answer(artefacts=[art("A"), art("B")]), entry)["accepted"]   # not new in the edit


def test_calibration_table_replays_the_ledger():
    path = edits.CALIBRATION_DIR / f"{edits.RENDER_V}.json"
    if not path.exists() or not edits.index_path().exists():
        pytest.skip("no Pass C calibration yet")
    committed = json.loads(path.read_text())
    table = edits.verdict_table(edits.read_index())
    assert table["classes"] == committed["classes"]
    assert table["edits"]["accepted"] == committed["edits"]["accepted"]
