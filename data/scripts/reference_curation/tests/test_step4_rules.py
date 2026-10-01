# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BodyFix Step 4: the pure rules of labels v2, B6, render_v4 and gate v2, on synthetic inputs.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_step4_rules.py -q
"""

from __future__ import annotations

import colorsys
import json
from types import SimpleNamespace

import numpy as np
import pytest

from extract_contact_configs import ZONE_ORDER
from reference_curation import b6, gate_v2, human_mesh as hm, labels as L1, labels_v2, render_v4

ZI = {z: i for i, z in enumerate(ZONE_ORDER)}


# --------------------------------------------------------------------------- #
# labels v2
# --------------------------------------------------------------------------- #
def test_any_of_groups_are_pairwise_not_transitive():
    # a leg wrapped against one arm: foot, shin and thigh each on the upper arm
    chain = [("L_FOOT", "R_UPPER_ARM"), ("L_SHANK", "R_UPPER_ARM"), ("L_THIGH", "R_UPPER_ARM")]
    groups = labels_v2.any_of_groups("h@1", chain)
    members = sorted(tuple(g["members"]) for g in groups)
    assert members == [("L_FOOT+R_UPPER_ARM", "L_SHANK+R_UPPER_ARM"), ("L_SHANK+R_UPPER_ARM", "L_THIGH+R_UPPER_ARM")]
    # labels v1's union-find put all three in one "any one of" group
    v1 = set(L1.alternative_groups("h@1", chain).values())
    assert len(v1) == 1
    # the foot alone does not satisfy the shin-thigh group
    st = SimpleNamespace(state=np.full((1, len(ZONE_ORDER)), -1), pairs=np.zeros((1, len(hm.PAIRS)), int))
    st.pairs[0, L1.PAIR_COLUMN[("L_FOOT", "R_UPPER_ARM")]] = 1
    demanded = [L1.pair_name(p) for p in chain]
    assert not labels_v2.configuration_holds(st, set(), demanded, groups)[0]
    st.pairs[0, L1.PAIR_COLUMN[("L_SHANK", "R_UPPER_ARM")]] = 1
    assert labels_v2.configuration_holds(st, set(), demanded, groups)[0]


def test_the_window_is_where_the_configuration_holds():
    T = 20
    st = SimpleNamespace(state=np.zeros((T, len(ZONE_ORDER)), int), pairs=np.zeros((T, len(hm.PAIRS)), int))
    st.state[:, ZI["L_FOOT"]] = 1
    k = L1.PAIR_COLUMN[("L_FOOT", "R_THIGH")]
    st.pairs[6:15, k] = 1           # the foot reaches the thigh at 6, leaves it at 15
    st.pairs[3, k] = -1             # an undecided frame agrees with anything
    ok = labels_v2.configuration_holds(st, {"L_FOOT"}, ["L_FOOT+R_THIGH"], [])
    assert labels_v2.window_around(ok, 0, 19, 10) == (6, 14)
    assert labels_v2.window_around(ok, 8, 12, 10) == (8, 12)          # never outside the old window
    assert labels_v2.window_around(ok, 0, 19, 2) is None              # the exemplar must hold it


def test_a_transition_reports_its_longest_constant_support():
    st = SimpleNamespace(state=np.zeros((60, len(ZONE_ORDER)), int))
    st.state[:20, ZI["L_FOOT"]] = 1
    st.state[20:50, ZI["R_FOOT"]] = 1
    assert labels_v2.longest_constant_run(st, 0, 59) == (30, 20)


@pytest.mark.parametrize("name,ground,votes,role", [
    ("standing", ["L_FOOT:G"], [], "standing"),
    ("Crow", ["L_HAND:G"], [], "family"),
    ("Crow_h1", ["L_FOOT:G", "L_HAND:G"], ["no"], "preparation"),
    ("Crow_h1", ["L_HAND:G"], ["no"], "undecided"),                    # the family's own ground set: no corroboration
    ("Crow_h1", ["L_HAND:G"], ["yes"], "family"),
    ("Crow_h1", ["L_FOOT:G", "L_HAND:G"], ["yes"], "variant"),
    ("Crow_h1", ["L_FOOT:G"], ["variant", "variant", "no"], "variant"),
    ("Crow_h1", ["L_FOOT:G"], ["no", "yes"], "undecided"),              # no majority
    ("Crow_h1", ["L_FOOT:G"], ["cannot_tell"], "undecided"),
])
def test_pose_role(name, ground, votes, role):
    family = [frozenset(["L_HAND:G"])]
    assert labels_v2.pose_role({"name": name, "pairs_ground": ground}, family, votes)[0] == role


def test_ground_role_rule():
    a = {"in_configuration": True, "source_state": "observed_contact", "target_role": "required_touch"}
    assert labels_v2.ground_role(a, 60.0, None) == ("required_support", "mat")
    assert labels_v2.ground_role(a, 10.0, {"necessity": "required"}) == ("required_support", "statics")
    assert labels_v2.ground_role(a, None, {"necessity": "useful"}) == ("required_touch", None)
    carried = {"in_configuration": True, "source_state": "unknown", "target_role": "unspecified"}
    assert labels_v2.ground_role(carried, 500.0, {"necessity": "required"}) == ("unspecified", None)


# --------------------------------------------------------------------------- #
# B6
# --------------------------------------------------------------------------- #
def _claims(*roles_per_sample, pair="L_SHANK+L_UPPER_ARM"):
    return [{"candidates": [pair], "roles": {pair: r}} for r in roles_per_sample]


def test_reproducibility_counts_ordered_sample_pairs():
    claims = {"a@1": _claims("required_touch", "required_touch"), "b@1": _claims("required_touch", "allowed"),
              "c@1": _claims("allowed", "incidental"), "d@1": _claims(None, "allowed")}
    rep = b6.reproducibility(claims)
    assert rep["critical"] == {"claims": 3, "agree": 2, "precision": 0.6667, "precision_lo95": rep["critical"]["precision_lo95"]}
    assert rep["not_critical"]["claims"] == 4 and rep["not_critical"]["agree"] == 2   # c both ways; b's and d's not
    assert rep["items"] == 4 and rep["answered_rate"] == 0.875


def test_admission_needs_both_values_to_reproduce_on_enough_claims():
    good = {"items": 60, "answered_rate": 0.9, "critical": {"claims": 40, "precision": 0.95},
            "not_critical": {"claims": 80, "precision": 0.93}}
    assert b6.admitted(good)[0]
    assert not b6.admitted({**good, "critical": {"claims": 29, "precision": 1.0}})[0]
    assert not b6.admitted({**good, "not_critical": {"claims": 80, "precision": 0.85}})[0]
    assert not b6.admitted({**good, "answered_rate": 0.4})[0]


def test_mirror_and_family_make_takes_comparable():
    assert b6.mirror("L_SHANK+R_UPPER_ARM") == "R_SHANK+L_UPPER_ARM"
    assert b6.mirror("HEAD+L_HAND") == "HEAD+R_HAND"
    assert b6.family("220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c") == \
        b6.family("220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a")
    assert b6.consensus(["required_touch", "required_touch"]) == "required_touch"
    assert b6.consensus(["required_touch", None]) is None


# --------------------------------------------------------------------------- #
# render_v4
# --------------------------------------------------------------------------- #
def _hue(rgb) -> float:
    return colorsys.rgb_to_hsv(*rgb)[0] * 360.0


def test_every_segment_has_its_own_colour_and_neighbours_differ_in_hue():
    rgb = {z: render_v4.zone_rgb(z) for z in ZONE_ORDER}
    assert rgb["PELVIS"] == rgb["TRUNK"]                       # "torso and pelvis", as render_v3
    assert len(set(rgb.values())) == len(ZONE_ORDER) - 1
    for side in "LR":
        for a, b in (("UPPER_ARM", "FOREARM"), ("FOREARM", "HAND"), ("THIGH", "SHANK"), ("SHANK", "FOOT")):
            d = abs(_hue(rgb[f"{side}_{a}"]) - _hue(rgb[f"{side}_{b}"]))
            assert min(d, 360 - d) > 20, (side, a, b)
    legend = render_v4.palette_legend()
    assert legend["left upper arm"] == "dark purple" and legend["right forearm"] == "light orange"
    assert legend["left thigh"] == "dark yellow" and legend["head and neck"] == "magenta"


def test_render_v4_refuses_to_draw_off_plant_v2():
    from reference_curation import render

    clip = render.Clip("x", 60, np.zeros((1, 24, 3)), np.tile([0, 0, 0, 1.0], (1, 24, 1)), None, None, (), ())
    with pytest.raises(RuntimeError, match="plant_context"):
        render_v4.Scene(clip)


# --------------------------------------------------------------------------- #
# gate v2
# --------------------------------------------------------------------------- #
WINDOW = {"box_max_deg": 0.0, "floor_min_cm": 0.0, "new_overlap_pair_frames": 0, "jerk_frames": 0,
          "edit_mean_p95_cm": 1.0, "edit_body_max_cm": 5.0}


def _hold(pairs, status="kept", not_realised=(), pose_role="family"):
    return {"hold_id": "c@1", "pairs": list(pairs), "pairs_ground": [p for p in pairs if p.endswith(":G")],
            "labels": {"status": status, "not_realised": list(not_realised), "pose_role": pose_role}}


def _passc(same_pose, closer="edited"):
    return {"accepted": same_pose == "yes" and closer != "fit", "same_pose": same_pose, "closer": closer,
            "more_natural": "edited", "artefacts": [], "verdict_id": "ledger:x.C.1"}


def test_a_pass_c_claim_decides_only_when_its_class_is_admitted():
    h = _hold(["L_FOOT:G", "R_FOOT:G"])
    both = {"pose_change": True, "closer": True}
    r = gate_v2.decide({}, h, [], None, _passc("no"), WINDOW, both)
    assert r["decision"] == "exclude" and r["reasons"]["exclude"] == ["not_same_pose"]
    r = gate_v2.decide({}, h, [], None, _passc("no"), WINDOW, {"pose_change": False, "closer": True})
    assert r["decision"] == "flag" and r["reasons"]["flag"] == ["not_same_pose_unadmitted"]
    assert gate_v2.decide({}, h, [], None, _passc("yes", "fit"), WINDOW, both)["reasons"]["flag"] == ["fit_closer"]
    assert gate_v2.decide({}, h, [], None, _passc("yes", "fit"), WINDOW, {"closer": False})["decision"] == "pass"


@pytest.mark.parametrize("realised_s_star,decision", [(0.27, "mask"), (1.3, "exclude"), (None, "exclude")])
def test_an_unrealised_support_is_masked_only_if_the_hold_stands_without_it(realised_s_star, decision):
    h = _hold(["L_FOOT:G", "R_FOOT:G", "R_HAND:G"], not_realised=["R_HAND:G"])
    srow = {"verdict": "support_not_realised", "gated": {"unrealised": ["R_HAND:G"], "realised_s_star": realised_s_star,
                                                          "s_star": 0.2}, "witness": None}
    r = gate_v2.decide({}, h, [], srow, _passc("yes"), WINDOW, {"pose_change": True, "closer": True})
    assert r["decision"] == decision
    if decision == "mask":
        assert r["masks"] == ["R_HAND:G"] and r["commanded"] == ["L_FOOT:G", "R_FOOT:G"]
        assert r["reasons"]["mask"] == ["unrealisable_R_HAND:G"]          # once, though statics and labels both name it
    else:
        assert r["reasons"]["exclude"] == ["unholdable_without_R_HAND:G"]


def test_a_single_leg_hold_is_flagged_when_its_statue_moves():
    h = _hold(["L_FOOT:G"])
    srow = {"verdict": "held", "gated": {"s_star": 0.1, "realised_s_star": None, "unrealised": []},
            "witness": {"passed": False, "passed_strict": False, "still": False, "settle_cm": 30.0, "drift_cm": 40.0}}
    r = gate_v2.decide({"group": "single_leg"}, h, [], srow, _passc("yes"), WINDOW, {"pose_change": True})
    assert r["decision"] == "flag" and r["reasons"]["flag"] == ["statue_moved_single_leg"]
    assert gate_v2.decide({"group": "standing"}, h, [], srow, _passc("yes"), WINDOW, {})["decision"] == "pass"


def test_the_a4_check_catches_a_commanded_set_that_is_not_configured_minus_masks():
    hold = _hold(["L_FOOT:G", "R_HAND:G", "L_SHANK+L_UPPER_ARM"])
    result = {"labels": {"manifest": {"clips": [{"stem": "c", "holds": [hold]}]}},
              "holds": [{"hold_id": "c@1", "decision": "mask", "masks": ["R_HAND:G"],
                         "commanded": ["L_FOOT:G", "L_SHANK+L_UPPER_ARM"], "reasons": {"flag": []}}]}
    release = gate_v2.release_manifest(result)
    assert gate_v2.check_commanded(result, release) == []
    assert release["clips"][0]["holds"][0]["pairs_configured"] == hold["pairs"]
    release["clips"][0]["holds"][0]["pairs_ground"] = ["L_FOOT:G", "R_HAND:G"]
    assert any("commanded ground" in p for p in gate_v2.check_commanded(result, release))
    result["holds"][0]["decision"] = "exclude"
    assert gate_v2.check_commanded(result, gate_v2.release_manifest(result)) == []
    assert any("excluded but released" in p for p in gate_v2.check_commanded(result, release))


def test_every_gate_reason_names_a_rule():
    assert gate_v2.reason_kind("unrealisable_R_HAND:G") == "unrealisable_<support>"
    assert gate_v2.reason_kind("not_realised_L_SHANK+L_UPPER_ARM") == "not_realised_<pair>"
    assert gate_v2.reason_kind("unholdable_without_HEAD:G") == "unholdable_without_<support>"
    assert gate_v2.reason_kind("transition") == "transition"
