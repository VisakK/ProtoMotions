# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BodyFix Step 4a: the plant-v2 machine evidence (mat attribution, capture store v4, audit v2, statics v2 and the
plant-v2 statue). The tests read the committed records and re-measure a few clips; they pin the numbers the step
measured, so a regression shows up as a wrong number.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_step4_evidence.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, capture, capture_v4, fit_writer as fw, human_mesh as hm, ids, pressure_v2
from reference_curation import retarget as rt
from reference_curation import statics_v2

REPO = ids.REPO
from protomotions.utils import plant_identity  # noqa: E402

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
RID = capture_v4.RETARGET_ID
ZI = {z: i for i, z in enumerate(ZONE_ORDER)}

pytestmark = pytest.mark.skipif(not capture_v4.motion_path(CROW).exists() or not (capture_v4.STORE_DIR / f"{CROW}.json").exists(),
                                reason="the Step 3 references or capture store v4 are not on disk")


def _one(glob: str):
    found = sorted((ids.DATA_ROOT).glob(glob))
    assert len(found) == 1, f"{glob}: {found}"
    return found[0]


# Statics v2 runs once per configuration, so its folders are pinned (BodyFix.MD, "What Step 4 found"): labels v1.1's
# configuration, B6's critical-only one (labels v2 pass 1) and the final one (pass 2, with the statics restoration).
STATICS_V11 = "statics_v2/holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828.statics_v2.f57d85d3fa"
STATICS_CRITICAL_ONLY = "statics_v2/holds_repaired_ftC_posefix.labels_v2.b61fad7e9d.statics_v2.fb432fa15d"
STATICS_FINAL = "statics_v2/holds_repaired_ftC_posefix.labels_v2.46024365fa.statics_v2.aab60189f4"


# --------------------------------------------------------------------------- #
# The mat attribution on plant v2
# --------------------------------------------------------------------------- #
def test_the_mat_explains_the_plant_v2_references_better_than_any_earlier_geometry():
    rec = json.loads((pressure_v2.RECORD_ROOT / RID / "pressure.json").read_text())
    po = rec["pooled"]
    assert rec["plant"]["plant"] == "v2" and len(rec["stems"]) == 56
    assert po["v2"]["loaded_frames"] == po["ftR"]["loaded_frames"] == po["shipped_port"]["loaded_frames"] == 81140
    assert po["v2"]["explained_ge_0_9"] >= 0.998           # 99.9 %: ftR 89.5 %, the shipped port's pose 71.2 %
    assert po["ftR"]["explained_ge_0_9"] < 0.9 and po["shipped_port"]["explained_ge_0_9"] < 0.75
    assert po["v2"]["cop_residual_cm_p90"] < 0.2 < 3.0 < po["ftR"]["cop_residual_cm_p90"] < po["shipped_port"]["cop_residual_cm_p90"]


def test_every_gated_port_is_the_reference_on_plant_v2_with_three_columns():
    p = pressure_v2.paths(RID)
    assert pressure_v2.check(fw.corpus()[0], p) == []


# --------------------------------------------------------------------------- #
# Capture store v4
# --------------------------------------------------------------------------- #
def test_store_v4_is_plant_v2s_and_refused_elsewhere():
    meta = json.loads((capture_v4.STORE_DIR / f"{CROW}.json").read_text())
    assert ids.require_plant(meta, "v4", capture_v4.plant_mjcf()) == "v2"
    with pytest.raises(plant_identity.PlantMismatchError):
        ids.require_plant(meta, "v4", plant_identity.mjcf_path("v1"))


def test_the_human_side_is_not_read_under_plant_v2():
    from reference_curation import retarget_v2

    with retarget_v2.on_plant():
        with pytest.raises(RuntimeError, match="outside on_plant"):
            capture_v4.human_base(CROW)


def test_crow_on_plant_v2_rests_on_its_hands_with_its_shins_on_its_arms():
    r = capture_v4.load(CROW, rebuild=False)
    f = 651                                               # labels v1.1's corrected exemplar (10.85 s)
    z = 100 * r["avatar_min_z"][f]
    assert z[ZI["L_HAND"]] < 0.5 and z[ZI["R_HAND"]] < 0.5 and z[ZI["L_FOOT"]] > 20 and z[ZI["R_FOOT"]] > 30
    col = {n: k for k, n in enumerate(hm.PAIR_NAMES)}
    for pair in ("L_SHANK+L_UPPER_ARM", "R_SHANK+R_UPPER_ARM"):
        assert r["avatar_pair_gap"][f, col[pair]] < 0.01 and r["human_pair_gap"][f, col[pair]] < 0.01
    load = r["mat_zone_load"][f]
    assert 300 < load[ZI["L_HAND"]] < 315 and 335 < load[ZI["R_HAND"]] < 350 and load[ZI["L_FOOT"]] == 0.0
    assert bool(r["attr_visible"][f, ZI["L_HAND"]]) and not bool(r["attr_visible"][f, ZI["R_FOOT"]])


def test_the_plant_free_mat_fields_are_capture_v1s():
    s = json.loads((capture_v4.STORE_DIR / "_summary.json").read_text())
    assert s["clips"] == 56 and s["failures"] == []
    assert s["plant_free_mat_max_diff"] == {"mat_total": 0.0, "mat_cop": 0.0, "mat_valid_cov": 0.0}


def test_the_human_side_thresholds_survive_the_new_attribution():
    """Re-derived on the plant-v2 attribution's mat-confirmed frames, every touch threshold moves < 0.1 cm. The
    head's marker test was admitted on 911 frames of 3 clips; with the crowns realised, 2,379 frames of 10 clips
    confirm it, and their spread (p99 - p50 2.1 cm) fails the admission rule, while its touch threshold holds."""
    cc = json.loads(capture_v4.CROSSCHECK_PATH.read_text())
    for z in ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND", "HEAD"):
        row = cc["markers"][z]
        assert row["admitted_v1"] and abs(row["touch_cm_v4"] - row["touch_cm_v1"]) < 0.1
        assert row["admitted_v4"] == (z != "HEAD")
    assert cc["markers"]["HEAD"]["n_frames_v4"] > 2 * cc["markers"]["HEAD"]["n_frames_v1"]
    assert abs(cc["mesh"]["touch_cm_v4"] - cc["mesh"]["touch_cm_committed"]) < 0.1


def test_zone_pair_gaps_are_the_retarget_kernels_minimum():
    from reference_curation.retarget_v2 import motion_state

    sk = fw.skeleton("v2")
    m = torch.load(capture_v4.motion_path(CROW), map_location="cpu", weights_only=False)
    pos, rot = motion_state(m)
    frames = np.arange(600, 700, 7)
    P, R = torch.as_tensor(pos[frames]), torch.as_tensor(rot[frames])
    g = capture_v4.zone_pair_gaps(sk, P, R)
    for k, (za, zb) in enumerate(hm.PAIRS):
        if za in ("L_SHANK", "R_SHANK") and zb in ("L_UPPER_ARM", "R_UPPER_ARM"):
            pairs = [(sk.names.index(a), sk.names.index(b)) for a in capture_v4.ZONES[za] for b in capture_v4.ZONES[zb]]
            exact = torch.stack([rt.body_gaps(sk, P, R, np.arange(len(frames)), np.full(len(frames), a), np.full(len(frames), b))
                                 for a, b in pairs], 1).min(1).values.numpy()
            assert np.allclose(g[:, k], np.minimum(exact, capture_v4.PAIR_CAP_M), atol=1e-9)


# --------------------------------------------------------------------------- #
# Audit v2
# --------------------------------------------------------------------------- #
def test_audit_v2_counts_the_floats_on_the_new_references():
    rec = json.loads((_one("audits/*.audit_v2.calibrated.*") / "audit.json").read_text())
    a, b = rec["metrics"]["all"], rec["baseline_v1"]["metrics"]["all"]
    assert a["holds"] == 280 and a["labelled_supports"] == 864
    assert b["hover_gt_cm"]["2"] == 434 and a["hover_gt_cm"]["2"] == 1 and a["hover_gt_cm"]["3"] == 0
    assert b["unexplained_load"]["holds"] == 33 and a["unexplained_load"]["holds"] == 0
    assert rec["plant"]["plant"] == "v2" and len(rec["dropped"]) == 4


def test_audit_v2_under_the_labels_own_evidence_finds_no_disagreement():
    rec = json.loads((_one("audits/*.audit_v2.sources.*") / "audit.json").read_text())
    a = rec["metrics"]["all"]
    assert a["missed_touch"]["holds"] == 0 and a["phantom_support"]["holds"] == 0
    assert a["exemplar_unstable"]["holds"] == 8           # the eight unresolved (transition) holds


def test_audit_v2_names_holds_by_the_labels_stable_id():
    rows = {json.loads(l)["hold_id"]: json.loads(l) for l in open(_one("audits/*.audit_v2.calibrated.*") / "holds.jsonl")}
    r = rows[f"{CROW}@439"]                                  # the source exemplar's id ...
    assert r["window"]["frame_hold"] == 651 and r["exemplar_hold_id"] == f"{CROW}@651"   # ... its corrected exemplar


# --------------------------------------------------------------------------- #
# Statics v2 and the plant-v2 statue
# --------------------------------------------------------------------------- #
def test_the_statue_takes_plant_v2s_training_gains():
    from reference_curation import witness_v2

    info = witness_v2.control_info()
    neck = [ci for name, ci in info.items() if name.startswith("Neck")]
    assert neck and all(ci.stiffness == 1158 and ci.damping == 116 for ci in neck)
    assert all(ci.effort_limit == 30 for name, ci in info.items() if "Wrist" in name)
    assert all(ci.effort_limit == 15 for name, ci in info.items() if "_Hand_" in name)


def test_the_statue_refuses_to_run_off_plant_v2():
    from reference_curation import witness_v2

    with pytest.raises(RuntimeError, match="on_plant"):
        witness_v2.run(np.zeros((24, 3)), np.tile([0, 0, 0, 1.0], (24, 1)), None)


def test_statics_v2_on_the_new_references():
    st = statics_v2.load(ids.DATA_ROOT / STATICS_V11)
    rows = st["holds"]
    v = {}
    for r in rows.values():
        v[r["verdict"]] = v.get(r["verdict"], 0) + 1
    assert len(rows) == 280 and v.get("held", 0) + v.get("feasible", 0) == 276
    assert {h for h, r in rows.items() if r["verdict"] == "beyond_plant"} == {
        "220923_Scorpion_pose_or_vrischikasana-b@945", "220923_Scorpion_pose_or_vrischikasana-b@1163",
        "220923_Peacock_Pose_or_Mayurasana_-a@1183"}
    (snr,) = [h for h, r in rows.items() if r["verdict"] == "support_not_realised"]
    assert snr == "220923_Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-b@669"
    assert rows[snr]["gated"]["realised_s_star"] < 1.0       # held on the realised supports: maskable
    wm = st["record"]["witness_metrics"]
    assert (wm["run"], wm["passed"], wm["passed_strict"], wm["still"]) == (276, 194, 180, 231)


def test_statics_v1_records_are_refused_by_the_v2_loader():
    old = _one("statics/*labels_v1.249a920cea.statics_v1.7d3b9e15b4")
    with pytest.raises(plant_identity.PlantMismatchError):
        statics_v2.load(old)


def test_d4_lets_an_unloaded_resting_zone_lift():
    """A forearm hold the strict rule failed because an unloaded hand rose: D4 passes it, the LP's loads decide."""
    st = statics_v2.load(ids.DATA_ROOT / STATICS_V11)
    hid = "220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a@354"
    w = st["holds"][hid]["witness"]
    assert hid in st["record"]["witness_metrics"]["passed_only_with_d4"]
    assert w["passed"] and not w["passed_strict"] and w["lifted_unloaded"] and not w["lifted"]
    assert set(w["lifted_unloaded"]).isdisjoint(w["must_stay"])


def test_statics_v2_on_the_final_configuration():
    """The configuration labels v2 ships: B6's critical pairs and Peacock -a@788's restored forearm braces. Only the
    two Scorpion -b holds stay beyond the plant (wrists); Peacock -a@1183 is holdable once B6 names the realised
    TRUNK braces instead of labels v1.1's unrealised PELVIS ones."""
    st = statics_v2.load(ids.DATA_ROOT / STATICS_FINAL)
    rows = st["holds"]
    v = {}
    for r in rows.values():
        v[r["verdict"]] = v.get(r["verdict"], 0) + 1
    assert len(rows) == 280 and v["held"] + v["feasible"] == 277
    assert {h for h, r in rows.items() if r["verdict"] == "beyond_plant"} == {
        "220923_Scorpion_pose_or_vrischikasana-b@945", "220923_Scorpion_pose_or_vrischikasana-b@1163"}
    wm = st["record"]["witness_metrics"]
    assert (wm["run"], wm["passed"], wm["passed_strict"], wm["still"]) == (277, 198, 183, 231)
    p = "220923_Peacock_Pose_or_Mayurasana_-a@"
    v11 = statics_v2.load(ids.DATA_ROOT / STATICS_V11)["holds"]
    assert v11[p + "1183"]["verdict"] == "beyond_plant" and rows[p + "1183"]["verdict"] == "feasible"
    assert set(rows[p + "1183"]["request"]["pairs"]) == {"TRUNK+L_FOREARM", "TRUNK+R_FOREARM"}


def test_one_configuration_change_moves_only_its_own_hold():
    """Statics v2 is deterministic per hold: pass 2 differs from the critical-only run only at the restored hold,
    whose row is labels v1.1's configuration's row exactly."""
    def strip(r):
        return {k: v for k, v in r.items() if k not in ("statics_id", "schema_version")}

    crit = statics_v2.load(ids.DATA_ROOT / STATICS_CRITICAL_ONLY)["holds"]
    final = statics_v2.load(ids.DATA_ROOT / STATICS_FINAL)["holds"]
    v11 = statics_v2.load(ids.DATA_ROOT / STATICS_V11)["holds"]
    changed = {h for h in final if strip(final[h]) != strip(crit[h])}
    assert changed == {"220923_Peacock_Pose_or_Mayurasana_-a@788"}
    (h,) = changed
    assert crit[h]["verdict"] == "beyond_plant" and strip(final[h]) == strip(v11[h])
