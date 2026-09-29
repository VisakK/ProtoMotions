# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gated statics and the statue witness (BUILD_PLAN Step 7).

They run on the shipped ftC references and the committed labels v1, and pin what the build measured:
the 0.5 m lift that the old solver ignored (README §3.5), the open pair that carries nothing, Side
Crow -c's shelves (useful, not required, in both holds), the three easy holds the statue holds, and the
corpus counts. Nothing under ``output/``, ``data/reference_curation/`` or ``REVIEW_ROOT`` is written.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import collections
import json
import math

import mujoco
import numpy as np
import pytest

import static_hold_lp as S
from reference_curation import ids, statics, verdicts, witness

SIDE_CROW_C = "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c"
WARRIOR_II = "220926_Warrior_II_Pose_or_Virabhadrasana_II_-a"
UTTANASANA = "220923_Standing_Forward_Bend_pose_or_Uttanasana_-a"
HANDS = ["L_HAND", "R_HAND"]
H1_SHELF, H2_SHELF = "L_SHANK+L_UPPER_ARM", "R_SHANK+R_UPPER_ARM"


def _labels_dir():
    try:
        return statics.default_labels_dir()
    except FileNotFoundError:
        return None


pytestmark = pytest.mark.skipif(not ids.SHIPPED_DIR.exists() or _labels_dir() is None,
                                reason="the shipped ftC clips or the labels v1 folder are not on disk")


@pytest.fixture(scope="module")
def labels():
    return statics.load_labels(_labels_dir())


def _hold(labels, hold_id):
    stem = hold_id.rsplit("@", 1)[0]
    return stem, next(h for s, hs in labels["clips"] if s == stem for h in hs if h["hold_id"] == hold_id)


def _pose(stem, frame):
    return statics.load_pose(stem, frame)


# --------------------------------------------------------------------------- #
# The card's exit tests
# --------------------------------------------------------------------------- #
def test_the_half_metre_lift_returns_support_not_realised():
    """The old solver balanced the lifted pose to 16 digits (README §3.5); the gated one refuses it."""
    pos, rot = _pose(SIDE_CROW_C, 1349)
    up = pos.copy()
    up[:, 2] += 0.50
    closed = ["L_HAND:G", "R_HAND:G", H2_SHELF]
    r = statics.analyse(up, rot, HANDS, [H2_SHELF])
    assert r["status"] == "support_not_realised"
    assert r["unrealised"] == ["L_HAND:G", "R_HAND:G"] and r["s_star"] is None
    assert all(r["contacts"][f"{z}:G"]["necessity"] == "not_realised" for z in HANDS)
    assert r["contacts"]["L_HAND:G"]["height_cm"] == pytest.approx(50.5, abs=0.01)
    # the old solver: the same numbers lifted as on the floor
    old = S.solve(up, rot, HANDS, [H2_SHELF])
    assert old["feasible"] and old["s_star"] == pytest.approx(S.solve(pos, rot, HANDS, [H2_SHELF])["s_star"], abs=1e-9)
    # the counterfactual cannot close a support 50 cm up into evidence: it says so
    cf = statics.analyse(up, rot, HANDS, [H2_SHELF], closed=closed)
    assert cf["counterfactual"] and cf["contacts"]["R_HAND:G"]["counterfactual"]


def test_an_open_pair_carries_zero_load():
    """Hold 2's shelf is the right one; the left shelf is 12 cm open and gets no columns."""
    pos, rot = _pose(SIDE_CROW_C, 1349)
    closed = ["L_HAND:G", "R_HAND:G", H2_SHELF]
    base = statics.analyse(pos, rot, HANDS, [H2_SHELF], closed=closed)
    with_open = statics.analyse(pos, rot, HANDS, [H2_SHELF, H1_SHELF], closed=closed)
    c = with_open["contacts"][H1_SHELF]
    assert c["gap_cm"] == pytest.approx(11.98, abs=0.01) and not c["realised"] and not c["loaded"]
    assert c["load_n"] == 0.0 and c["necessity"] == "open" and with_open["open"] == [H1_SHELF]
    assert with_open["s_star"] == pytest.approx(base["s_star"], abs=1e-9)
    # the old solver gives the same open pair force columns and loads it
    old = S.solve(pos, rot, HANDS, [H2_SHELF, H1_SHELF])
    assert old["gaps_cm"][H1_SHELF] == pytest.approx(12.0, abs=0.05)


@pytest.mark.parametrize("hold_id,frame,shelf,s_without,s_star,gated", [
    (f"{SIDE_CROW_C}@606", 606, H1_SHELF, 0.5780, 0.5135, "support_not_realised"),
    (f"{SIDE_CROW_C}@1296", 1349, H2_SHELF, 0.5241, 0.3606, "infeasible"),
])
def test_side_crow_c_shelf_is_useful_not_required(labels, hold_id, frame, shelf, s_without, s_star, gated):
    """Both holds' shelves: useful (the effort falls with it), not required (the hands alone hold the
    pose within the limits, min load 0). The reference itself cannot say so: hold 1's right hand floats
    3.85 cm, and hold 2's tilted right hand leaves the COM outside its in-band corners. The
    counterfactual reproduces the investigation's what-if-closed numbers (README §3.5)."""
    stem, hold = _hold(labels, hold_id)
    assert hold["frame_hold"] == frame and hold["pairs"] == ["L_HAND:G", "R_HAND:G", shelf]
    r = statics.audit_hold(stem, hold, labels["anns"][hold_id], run_witness=False)
    assert r["gated"]["status"] == gated and r["verdict"] == gated
    cf = r["counterfactual"]
    assert cf["status"] == "optimal" and cf["counterfactual"] and cf["s_star"] == pytest.approx(s_star, abs=1e-3)
    c = cf["contacts"][shelf]
    assert c["necessity"] == "useful" and c["load_min_n"] == 0.0 and c["load_max_n"] > 1000
    assert c["s_without"] == pytest.approx(s_without, abs=1e-3) and c["effort_relief"] >= statics.RELIEF_MIN
    assert cf["contacts"]["L_HAND:G"]["necessity"] == cf["contacts"]["R_HAND:G"]["necessity"] == "required"
    row = next(x for x in statics.contact_rows([r], labels) if x["contact"] == shelf)
    assert row["necessity"] == "useful" and row["necessity_basis"] == "counterfactual"
    assert row["target_role"] == "required_touch" and row["review_role"] == "required_touch"


@pytest.mark.parametrize("hold_id,settle,drift", [
    (f"{WARRIOR_II}@572", 2.41, 0.22),
    (f"{UTTANASANA}@25", 3.03, 0.65),
    (f"{UTTANASANA}@719", 3.08, 0.75),
])
def test_the_witness_holds_three_easy_holds(labels, hold_id, settle, drift):
    stem, hold = _hold(labels, hold_id)
    r = statics.audit_hold(stem, hold, labels["anns"][hold_id])
    w = r["witness"]
    assert r["verdict"] == "held" and w["passed"] and w["feed_forward"]
    assert w["settle_cm"] == pytest.approx(settle, abs=0.05) and w["drift_cm"] == pytest.approx(drift, abs=0.05)
    assert not (w["lifted"] or w["landed"] or w["pairs_opened"] or w["saturated"])
    assert {"L_FOOT:G", "R_FOOT:G"} <= set(w["end_contacts"])


# --------------------------------------------------------------------------- #
# The witness: what it rejects, and why it is stiff
# --------------------------------------------------------------------------- #
def test_the_witness_drops_a_lifted_pose():
    pos, rot = _pose(WARRIOR_II, 572)
    r = statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])
    up = pos.copy()
    up[:, 2] += 0.5
    w = witness.run(up, rot, r["tau"])
    assert not w["passed"] and w["settle_cm"] > 45 and w["start_contacts"] == {}
    assert set(w["landed"]) >= {"L_FOOT:G", "R_FOOT:G"}


def test_the_statue_buckles_at_the_training_gains():
    """Why ``STIFFNESS_SCALE`` is 10: at 1x the ankle-knee-hip springs in series are softer than
    gravity (m g h), and Warrior II -a, which the 10x statue holds, falls."""
    pos, rot = _pose(WARRIOR_II, 572)
    r = statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])
    soft = witness.run(pos, rot, r["tau"], stiffness_scale=1.0)
    stiff = witness.run(pos, rot, r["tau"])
    assert stiff["passed"] and not soft["passed"] and soft["final_cm"] > 50


def test_the_witness_does_not_hold_what_the_gated_lp_rejects():
    """Side Crow -c hold 2 is gated-infeasible (tilted right hand); fed the counterfactual torques, the
    statue falls off its hands."""
    pos, rot = _pose(SIDE_CROW_C, 1349)
    cf = statics.analyse(pos, rot, HANDS, [H2_SHELF], closed=["L_HAND:G", "R_HAND:G", H2_SHELF])
    w = witness.run(pos, rot, cf["tau"], [H2_SHELF])
    assert not w["passed"] and w["drift_cm"] > 20


def test_the_witness_plant_is_the_training_plant(labels):
    p = witness.plant()
    m = p.model
    by_joint = {m.joint(m.actuator_trnid[a, 0]).name: a for a in range(m.nu)}
    for joint, (kp, kd) in {"L_Hip_x": (800, 80), "R_Toe_y": (500, 50), "Chest_z": (1000, 100),
                            "L_Shoulder_y": (500, 50), "R_Elbow_z": (500, 50), "L_Wrist_x": (300, 30),
                            "R_Hand_z": (300, 30), "Head_x": (500, 50)}.items():
        a = by_joint[joint]
        assert p.kp[a] == pytest.approx(witness.STIFFNESS_SCALE * kp)
        assert p.kd[a] == pytest.approx(math.sqrt(witness.STIFFNESS_SCALE) * kd)
    np.testing.assert_allclose(p.limit, S.TAU_MAX[p.dof])
    assert m.body_mass.sum() == pytest.approx(74.0, abs=1e-3)
    assert not m.jnt_limited.any() and not m.jnt_stiffness.any() and not m.dof_damping.any()
    # friction as MuJoCo mixes it: 0.75 on the floor, 0.5 between bodies
    pos, rot = _pose(SIDE_CROW_C, 0)   # standing, the feet overlapping by 0.94 cm
    S.set_pose(pos, rot)
    d = mujoco.MjData(m)
    d.qpos[:] = S.D.qpos
    d.qpos[2] -= 0.006   # the right foot, 0.5 cm up, into the floor
    mujoco.mj_forward(m, d)
    mu = {("floor" if p.floor in (c.geom1, c.geom2) else "body"): c.friction[0] for c in d.contact[:d.ncon]}
    assert mu == {"floor": pytest.approx(0.75), "body": pytest.approx(0.5)}


# --------------------------------------------------------------------------- #
# The LP: statuses, checks, necessity
# --------------------------------------------------------------------------- #
def test_typed_statuses(monkeypatch):
    pos, rot = _pose(WARRIOR_II, 572)
    assert statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])["status"] == "optimal"
    assert statics.analyse(pos, rot, ["L_FOOT"])["status"] == "infeasible"   # the COM is between the feet
    real = statics.linprog

    def failing(code):
        def fake(*args, **kwargs):
            res = real(*args, **kwargs)
            res.status = code
            return res
        return fake
    monkeypatch.setattr(statics, "linprog", failing(1))
    assert statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])["status"] == "iteration_limit"
    monkeypatch.setattr(statics, "linprog", failing(4))
    assert statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])["status"] == "numerical"

    def nan(*args, **kwargs):
        res = real(*args, **kwargs)
        res.x = res.x * np.nan
        return res
    monkeypatch.setattr(statics, "linprog", nan)
    r = statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])
    assert r["status"] == "numerical" and r["contacts"]["L_FOOT:G"]["necessity"] == "undecided"


def test_a_com_on_the_polygon_edge_is_decided_by_the_interior_point():
    """Plow -b's first frame: the COM is 0.4 mm outside its in-band foot corners, and HiGHS's simplex
    returns 'model status unknown'. The retry decides it."""
    pos, rot = _pose("220926_Plow_Pose_or_Halasana_-b", 0)
    assert statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])["status"] == "infeasible"


def test_finite_and_unit_checks():
    pos, rot = _pose(WARRIOR_II, 572)
    bad = pos.copy()
    bad[3, 1] = np.nan
    with pytest.raises(FloatingPointError):
        statics.analyse(bad, rot, ["L_FOOT", "R_FOOT"])
    with pytest.raises(ValueError):
        statics.analyse(pos, rot * 1.1, ["L_FOOT", "R_FOOT"])


def test_necessity_is_strict_and_its_two_tests_agree():
    """Warrior II's feet are each required (the COM lies between them), with tight load intervals that
    sum to the body weight."""
    pos, rot = _pose(WARRIOR_II, 572)
    r = statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"])
    for z, (lo, hi) in {"L_FOOT:G": (365.7, 374.0), "R_FOOT:G": (352.0, 360.3)}.items():
        c = r["contacts"][z]
        assert c["necessity"] == "required" and c["infeasible_without"] and not c.get("dual_mismatch")
        assert (c["load_min_n"], c["load_max_n"]) == pytest.approx((lo, hi), abs=0.2)
    total = sum(r["contacts"][z]["load_n"] for z in ("L_FOOT:G", "R_FOOT:G"))
    assert total == pytest.approx(9.81 * S.M.body_mass.sum(), abs=0.1)   # the ground carries the weight (0.1 N rounding)


def test_joint_stops_only_relieve():
    pos, rot = _pose(WARRIOR_II, 572)
    r = statics.analyse(pos, rot, ["L_FOOT", "R_FOOT"], stops=True)
    st = r["stops"]
    assert st["status"] == "optimal" and st["candidates"] > 0 and st["s_star"] <= r["s_star"] + 1e-9
    assert r["limit_violations_deg"]["L_Knee_z"] == pytest.approx(11.0, abs=0.1)   # 41 deg against +-30


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corpus(labels):
    records, failures = statics.audit_holds(labels, workers=8)
    assert failures == []
    return records, statics.contact_rows(records, labels)


def test_corpus_contract_and_counts(corpus):
    records, rows = corpus
    assert statics.check(records) == []
    assert len(records) == 303 and len(rows) == 1122
    assert collections.Counter(r["verdict"] for r in records) == {
        "support_not_realised": 226, "feasible": 34, "infeasible": 31, "beyond_plant": 8, "held": 4}
    assert sorted(r["hold_id"] for r in records if r["verdict"] == "held") == [
        f"{UTTANASANA}@25", f"{UTTANASANA}@719", f"{UTTANASANA}@896", f"{WARRIOR_II}@572"]
    assert collections.Counter(r["counterfactual"]["status"] for r in records if r["counterfactual"]) == {
        "optimal": 263, "infeasible": 7}
    # no required_touch pair is required: body-body forces are internal, the hands and feet carry the pose
    pairs = collections.Counter(x["necessity"] for x in rows if x["kind"] == "pair" and x["target_role"] == "required_touch")
    assert pairs == {"useful": 114, "redundant": 82}
    touch = collections.Counter(x["necessity"] for x in rows if x["kind"] == "ground" and x["target_role"] == "required_touch")
    assert touch == {"required": 82, "useful": 276, "redundant": 101, "undecided": 20}


def test_gating_agrees_with_the_capture_store(corpus, labels):
    """The LP's zone height is the store's ``avatar_min_z`` at the exemplar (labels v1's
    ``evidence.avatar.cm``), so ``realised`` is the labels' float rule applied to the exemplar."""
    records, _ = corpus
    n = 0
    for r in records:
        anns = {a["contact"]: a for a in labels["anns"][r["hold_id"]]}
        for name, c in r["gated"]["contacts"].items():
            if c["kind"] == "ground":
                assert c["height_cm"] == pytest.approx(anns[name]["evidence"]["avatar"]["cm"], abs=0.011)
                if abs(c["height_cm"] - 100 * statics.GROUND_BAND_M) > 0.005:   # stored to 0.01 cm
                    assert c["realised"] == (c["height_cm"] <= 100 * statics.GROUND_BAND_M)
                n += 1
            elif c["gap_cm"] < 100 * verdicts.PAIR_GATE_M:   # beyond the gate the store keeps a lower bound
                assert c["gap_cm"] == pytest.approx(anns[name]["evidence"]["avatar"]["gap_cm"], abs=0.011)
    assert n == 924


def test_cli_writes_a_reproducible_folder(tmp_path):
    argv = ["--stem", WARRIOR_II, SIDE_CROW_C, "--workers", "1", "--out-root", str(tmp_path)]
    assert statics.main(argv) == 0
    (out,) = tmp_path.iterdir()
    first = {f: (out / f).read_bytes() for f in ("holds.jsonl", "contacts.jsonl", "summary.md")}
    assert statics.main(argv) == 0 and [p.name for p in tmp_path.iterdir()] == [out.name]
    assert {f: (out / f).read_bytes() for f in first} == first   # HiGHS and MuJoCo are deterministic here
    meta = json.loads((out / "statics.json").read_text())
    assert meta["statics_id"] == out.name and meta["labels_id"].startswith("holds_repaired_ftC_posefix.labels_v1.")
    assert meta["generator"]["module"] == statics.MODULE and meta["witness"]["stiffness_scale"] == 10.0
    holds = [json.loads(line) for line in (out / "holds.jsonl").read_text().splitlines()]
    assert {h["hold_id"] for h in holds if h["verdict"] == "held"} == {f"{WARRIOR_II}@572"}
    assert "## Worklist" in (out / "summary.md").read_text()
