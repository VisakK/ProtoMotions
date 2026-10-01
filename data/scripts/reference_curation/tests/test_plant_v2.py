# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for plant v2 (BodyFix Step 1): the performer's skeleton and masses with commensurate colliders.

They pin what the build (``data/scripts/build_subject_plant_v2.py``) and its acceptance
(``reference_curation.plant_v2``) measured: the XML pair is the shipped plant with only body offsets, the one
geom per body and 23 joint ranges changed; its rest FK is her SMPL-X joints; no colliding pair overlaps at
rest; the head reaches her crown, the foot boxes her sole, the hand boxes her palm plane; the masses are her
template's; the joint box follows the decision table; and, driven by her own motion over the 59 clips, the
card's acceptance numbers. The build is re-derived from the template and compared with the committed XML.
Nothing under ``output/``, ``data/reference_curation/`` or ``data/assets/`` is written.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_plant_v2.py -q
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import build_subject_plant_v2 as B
from reference_curation import human_mesh as hm, ids, mosh_replay as mr, plant_v2 as P, retarget as rt

needs_plant = pytest.mark.skipif(not (mr.V2_XML.exists() and mr.V2_FLAT.exists() and P.RECORD.exists()),
                                 reason="plant v2 is not built (run build_subject_plant_v2.py)")
needs_data = pytest.mark.skipif(not (hm.MODEL_PATH.exists() and (ids.MOYO_DATA / "mosh").exists()),
                                reason="the SMPL-X model or the MoSh fits are not on disk")
pytestmark = [needs_plant, needs_data]

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
# The 23 joint ranges the decision table changes (degrees), mirror-symmetric (x, z swap sides L/R, y does not).
CHANGED = {"L_Knee_x": (-50, 30), "R_Knee_x": (-30, 50), "L_Knee_y": (-5, 160), "R_Knee_y": (-5, 160),
           "L_Knee_z": (-40, 40), "R_Knee_z": (-40, 40), "L_Ankle_y": (-55, 75), "R_Ankle_y": (-55, 75),
           "Neck_y": (-55, 60), "L_Thorax_x": (-40, 50), "R_Thorax_x": (-50, 40), "L_Thorax_y": (-35, 30),
           "R_Thorax_y": (-35, 30), "L_Thorax_z": (-60, 30), "R_Thorax_z": (-30, 60), "L_Elbow_x": (-30, 70),
           "R_Elbow_x": (-70, 30), "L_Elbow_y": (-70, 70), "R_Elbow_y": (-70, 70), "L_Elbow_z": (-160, 20),
           "R_Elbow_z": (-20, 160), "L_Hand_x": (-110, 90), "R_Hand_x": (-90, 110)}


@pytest.fixture(scope="module")
def record():
    return json.loads(P.RECORD.read_text())


@pytest.fixture(scope="module")
def models():
    import mujoco

    return {k: mujoco.MjModel.from_xml_path(str(p)) for k, p in
            (("v2", mr.V2_XML), ("v2_flat", mr.V2_FLAT), ("shipped", mr.SHIPPED_XML))}


@pytest.fixture(scope="module")
def her():
    from extract_contact_configs import mjcf_body_names

    mdl, vt, J = B.load_template()
    names = mjcf_body_names(str(B.SRC))
    parents = mr.skeleton_for(B.SRC).parents
    skin, area, nrm = B.skin_arrays(mdl, vt)
    geo, rec, O, off = B.build_colliders(mdl, vt, J, names, parents, skin, area, nrm)
    return dict(mdl=mdl, vt=vt, J=J, names=names, parents=parents, skin=skin, area=area, nrm=nrm, geo=geo, rec=rec,
                O=O, off=off)


# --------------------------------------------------------------------------- #
# The files
# --------------------------------------------------------------------------- #
def test_record_describes_these_files(record):
    assert record["outputs"]["xml_sha256"] == ids.sha256_file(mr.V2_XML)
    assert record["outputs"]["flat_sha256"] == ids.sha256_file(mr.V2_FLAT)
    assert record["acceptance"]["xml_sha256"] == ids.sha256_file(mr.V2_XML)      # measured on these very files
    assert record["template"]["sha1"] == B.TEMPLATE_SHA1


def test_xml_pair_loads_at_74_kg_and_compiles_alike(models):
    v2, flat, ship = models["v2"], models["v2_flat"], models["shipped"]
    assert abs(v2.body_mass.sum() - 74.0) < 1e-4 and abs(flat.body_mass.sum() - 74.0) < 1e-4
    for f in ("body_pos", "body_mass", "body_inertia", "body_ipos", "geom_size", "geom_pos", "geom_quat", "jnt_range",
              "jnt_actfrcrange", "actuator_gear"):
        assert np.abs(getattr(v2, f) - getattr(flat, f)).max() == 0.0, f
    # the shipped plant's joints, actuators and primitives: one geom per body, same type and size
    assert [v2.joint(j).name for j in range(v2.njnt)] == [ship.joint(j).name for j in range(ship.njnt)]
    assert np.array_equal(v2.jnt_axis, ship.jnt_axis) and np.array_equal(v2.jnt_type, ship.jnt_type)
    # torque limits: the shipped ones, but wrist and hand raised 50 % (the user's decision at Step 2); gear follows
    wrist_hand = np.array([any(k in v2.joint(j).name for k in ("Wrist", "Hand")) for j in range(v2.njnt)])
    assert np.array_equal(v2.jnt_actfrcrange[~wrist_hand], ship.jnt_actfrcrange[~wrist_hand])
    assert np.allclose(v2.jnt_actfrcrange[wrist_hand], 1.5 * ship.jnt_actfrcrange[wrist_hand]) and wrist_hand.sum() == 12
    assert np.array_equal(v2.actuator_gear[:, 0], v2.jnt_actfrcrange[v2.actuator_trnid[:, 0], 1])
    assert np.array_equal(np.bincount(v2.geom_bodyid), np.bincount(ship.geom_bodyid))
    assert np.array_equal(v2.geom_type, ship.geom_type)
    # no collider grew: sphere and capsule radii, box half-extents (a capsule's half-length follows its bone)
    import mujoco

    for g in range(v2.ngeom):
        k = 3 if v2.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 1
        assert np.abs(v2.geom_size[g, :k] - ship.geom_size[g, :k]).max() < 1e-6, g
    assert np.abs(v2.body_quat - [1, 0, 0, 0]).max() == 0.0                         # identity rest orientations


def test_only_the_decided_joint_ranges_changed(record, models):
    from protomotions.components.pose_lib import extract_kinematic_info

    k0, k1 = extract_kinematic_info(str(mr.SHIPPED_XML)), extract_kinematic_info(str(mr.V2_XML))
    assert list(k1.dof_names) == list(k0.dof_names) and list(k1.body_names) == list(k0.body_names)
    v2, ship = models["v2"], models["shipped"]
    for j in range(1, v2.njnt):
        name = v2.joint(j).name
        got = tuple(np.degrees(v2.jnt_range[j]).round(4))
        want = CHANGED.get(name, tuple(np.degrees(ship.jnt_range[j]).round(4)))
        assert got == pytest.approx(want, abs=1e-4), name
    assert set(record["checks"]["dof_limits_changed"]) == set(CHANGED)
    limits = json.loads(B.LIMITS_JSON.read_text())
    assert {k: tuple(v["v2_deg"]) for k, v in limits["joint_ranges_changed_deg"].items()} == {k: tuple(map(float, v)) for k, v in CHANGED.items()}
    assert limits["torque_limits_changed_Nm"] == {f"{side}_{j}_{a}": {"shipped": t, "v2": 1.5 * t}
                                                for side in "LR" for j, t in (("Wrist", 20.0), ("Hand", 10.0)) for a in "xyz"}
    # the kept sides are the fit's right-foot artefacts (toe roll, ankle inversion)
    assert record["joint_box"]["totals"]["v2_coordinate_sides_past_on_1pct"] == ["R_Ankle_x:hi", "R_Toe_x:lo"]


# --------------------------------------------------------------------------- #
# Her skeleton
# --------------------------------------------------------------------------- #
def test_rest_fk_is_her_smplx_joints(her):
    sk = mr.skeleton_for(mr.V2_XML)
    pos, _ = mr.rest_pose(sk)
    target = np.stack([(her["O"][b] - her["O"]["Pelvis"]) @ rt.AXES for b in her["names"]])
    assert np.abs(pos[0] - target).max() < 1e-5                                     # measured 1.6e-8 m


def test_her_fit_replays_exactly_on_plant_v2():
    kin = mr.mosh_kinematics(CROW)
    pos, _ = mr.fk_bodies(mr.skeleton_for(mr.V2_XML), kin)
    assert np.linalg.norm(pos - mr.body_joint_targets(kin), axis=-1).max() < 1e-5  # 2.9e-8 m over the corpus
    pos_s, _ = mr.fk_bodies(mr.skeleton_for(mr.SHIPPED_XML), kin)
    assert np.linalg.norm(pos_s - mr.body_joint_targets(kin), axis=-1).max() > 0.05  # the old skeleton misses by cm


def test_no_colliding_pair_overlaps_at_rest(record):
    sk = mr.skeleton_for(mr.V2_XML)
    pos, rot = mr.rest_pose(sk)
    g = mr.pair_gaps(sk, pos, rot)[0]
    assert len(g) == 253 and g.min() == pytest.approx(0.0292, abs=5e-4)            # Torso-Chest, 2.92 cm
    assert record["checks"]["rest_pelvis_above_lowest_collider_m"] == pytest.approx(0.9745, abs=5e-4)
    assert record["checks"]["rest_collider_stature_m"] == pytest.approx(1.749, abs=1e-3)   # her 1.748 m


# --------------------------------------------------------------------------- #
# Colliders: shipped primitives, placed on her segments
# --------------------------------------------------------------------------- #
def test_the_build_reproduces_the_committed_colliders(her):
    sk = mr.skeleton_for(mr.V2_XML)
    for b in her["names"]:
        g, w = her["geo"][b], sk.geoms[b][0]
        for k in ("center", "seg", "half", "radius"):
            if k in g:
                assert np.abs(np.asarray(g[k]) - np.asarray(w[k])).max() < 2e-6, (b, k)
        if "quat" in g:
            assert abs(abs(float(np.dot(g["quat"], w["quat"]))) - 1) < 1e-6, b


def test_head_sphere_reaches_her_crown(her, record):
    h = her["rec"]["Head"]
    assert h["radius_m"] == pytest.approx(0.095)
    assert abs(h["crown_minus_surface_m"]) < 0.005 and h["crown_minus_surface_m"] == pytest.approx(-0.0012, abs=2e-4)
    assert np.allclose(h["center_head_frame_m"], [-0.0092, -0.0036, 0.0466], atol=2e-4)   # BodyFix §1: (-0.9, -0.4, +4.7) cm
    assert h["chin_minus_surface_m"] > 0.05                                         # the chin stays outside
    hr = record["colliders"]["rules"]["Head"]                                       # no rest overlap with Neck or Chest
    assert hr["rest_gap_to_chest_m"] == pytest.approx(0.149, abs=1e-3) and hr["rest_gap_to_neck_m"] > 0.02


def test_neck_sphere_splits_the_column(her):
    n = her["rec"]["Neck"]
    assert n["gaps_at_shipped_offset_m"]["head"] == pytest.approx(0.0503, abs=2e-4)
    assert n["gaps_v2_m"]["chest"] == pytest.approx(n["gaps_v2_m"]["head"], abs=1e-9)
    assert n["gaps_v2_m"]["chest"] == pytest.approx(0.0296, abs=2e-4)


def test_foot_boxes_rest_on_her_sole(her):
    for s, depth in (("L", 0.0681), ("R", 0.0755)):                                  # BodyFix: 6.8 / 7.6 cm
        a, t = her["rec"][f"{s}_Ankle"], her["rec"][f"{s}_Toe"]
        assert a["ankle_above_sole_m"] == pytest.approx(depth, abs=2e-4)
        g = her["geo"][f"{s}_Ankle"]
        assert g["center"][2] - g["half"][2] == pytest.approx(-a["ankle_above_sole_m"], abs=1e-9)
        assert t["toe_bottom_minus_foot_bottom_m"] == pytest.approx(0.0011, abs=1e-4)   # flush, as shipped
    assert her["rec"]["L_Ankle"]["drop_m"] == pytest.approx(0.0152, abs=2e-4)
    assert her["rec"]["R_Ankle"]["drop_m"] == pytest.approx(0.0226, abs=2e-4)


def test_hand_boxes_lie_on_her_palm_plane(her):
    from scipy.spatial.transform import Rotation

    for s in "LR":
        pc, n, _ = B.palm_plane(her["mdl"], her["vt"], her["O"], her["skin"], her["area"], her["nrm"], s)
        h = her["geo"][f"{s}_Hand"]
        bottom = Rotation.from_quat(h["quat"]).as_matrix() @ [0, 0, -1.0]
        assert math.degrees(math.acos(min(1.0, bottom @ n))) < 1e-6                 # rotated onto the plane
        assert her["rec"][f"{s}_Hand"]["shipped_bottom_vs_palm_deg"] == pytest.approx(14.4, abs=0.1)
        centre_w = h["center"] + her["off"][f"{s}_Hand"]
        assert n @ (centre_w + h["half"][2] * n - pc) == pytest.approx(0.0, abs=1e-9)
        w = her["geo"][f"{s}_Wrist"]
        assert n @ (w["center"] - [0, 0, w["half"][2]] - pc) == pytest.approx(0.0, abs=1e-9)


def test_capsules_keep_their_radius_and_fraction_of_the_bone(her):
    ship = B.shipped_geoms(her["names"])
    sk0 = mr.skeleton_for(mr.SHIPPED_XML)
    for b, child in B.CAPSULE_CHILD.items():
        g = her["geo"][b]
        assert g["radius"] == ship[b]["radius"]
        bone_s = sk0.offsets[her["names"].index(child)].numpy().astype(float)
        bone_v = her["off"][child]
        frac = lambda seg, bone: (seg @ bone) / (bone @ bone)  # noqa: E731
        assert np.allclose(frac(g["seg"], bone_v), frac(ship[b]["seg"], bone_s), atol=1e-6), b


# --------------------------------------------------------------------------- #
# Masses and inertia
# --------------------------------------------------------------------------- #
def test_masses_are_her_template_cut_at_the_joints(her, models, record):
    pts, vol = B.voxelize(her["mdl"], her["vt"])
    assert vol.sum() == pytest.approx(0.07248, abs=1e-5)                               # her 72.4 L
    dom = B.dominant_body(her["mdl"], her["vt"], pts, her["names"])
    body, _ = B.anatomical_partition(pts, vol, dom, her["O"], her["names"])
    rho = B.TARGET_MASS / vol.sum()
    v2 = models["v2"]
    for i, b in enumerate(her["names"]):
        assert v2.body_mass[i + 1] == pytest.approx(vol[body == i].sum() * rho, abs=1e-4), b
    m = {b: float(v2.body_mass[i + 1]) for i, b in enumerate(her["names"])}
    assert m["Pelvis"] == pytest.approx(13.79, abs=0.01)                               # was 15.28 by skinning
    assert (m["L_Hip"], m["R_Hip"]) == pytest.approx((9.14, 8.88), abs=0.01)
    off = {k for k, v in record["masses"]["de_leva_female"].items() if v["off_by_more_than_tol"]}
    assert off == {"L_upper_arm", "R_upper_arm"}                                       # both with a recorded reason
    assert all(record["masses"]["de_leva_female"][k].get("reason") for k in off)


def test_whole_body_inertia_is_hers(record):
    wb = record["checks"]["whole_body"]
    assert np.allclose(wb["v2_over_human_diag"], [1.022, 1.026, 0.909], atol=0.005)    # prototype 1.08/1.09/0.94
    assert min(wb["shipped_over_human_diag"]) > 1.14                                  # shipped 1.15-1.17


def test_pd_gains_move_only_for_the_neck(record):
    pd = record["pd_gains"]
    assert pd["groups_beyond_threshold"] == ["Neck"]
    assert pd["joint_space_inertia_rest"]["Neck"]["ratio"] == pytest.approx(2.315, abs=0.01)
    ch = pd["changes_for_robot_config"]["Neck"]
    assert ch["stiffness"][0] == 500 and ch["stiffness"][1] == pytest.approx(1157.7, abs=1.0)
    assert ch["natural_frequency_rad_s"]["v2_scaled_gains"] == pytest.approx(ch["natural_frequency_rad_s"]["shipped"])


# --------------------------------------------------------------------------- #
# Joint box: the decision table on her motion
# --------------------------------------------------------------------------- #
def test_joint_box_decisions_reproduce_the_committed_ranges(her, models):
    ship = models["shipped"]
    dof_names = [f"{b}_{a}" for b in her["names"][1:] for a in "xyz"]
    rng = {ship.joint(j).name: np.degrees(ship.jnt_range[j]).round(6) for j in range(1, ship.njnt)}
    lo, hi = np.radians([rng[n][0] for n in dof_names]), np.radians([rng[n][1] for n in dof_names])
    stems, dof, clip = B.corpus_dof(workers=8)
    assert len(stems) == 57 and len(dof) == 82592
    ev, tot, _ = B.joint_box_evidence(dof, clip, stems, her["names"], her["parents"], her["off"], lo, hi)
    assert tot["frames_any_past"] == 71073                                           # 86 % of her frames
    assert ev["L_Thorax_z:lo"]["share_past"] == pytest.approx(0.3935, abs=1e-3)       # the card's hotspots
    assert ev["R_Toe_x:lo"]["share_past"] == pytest.approx(0.2651, abs=1e-3)
    new_lo, new_hi, _ = B.decide_ranges(ev, her["names"], lo, hi)
    for k, n in enumerate(dof_names):
        assert (new_lo[k], new_hi[k]) == pytest.approx(CHANGED.get(n, tuple(rng[n])), abs=1e-6), n


# --------------------------------------------------------------------------- #
# Acceptance: her own motion over the corpus (about a minute on 8 workers)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def acceptance():
    acc, _ = P.measure(workers=8)
    return acc


def test_acceptance_passes(acceptance):
    assert acceptance["failures"] == [] and acceptance["acceptance_failures"] == []
    assert acceptance["clips"] == 59 and acceptance["replay"]["frames"] == 87888


def test_self_overlap_is_no_worse_than_the_shipped_plant(acceptance):
    ov = acceptance["replay"]["overlaps"]
    assert ov["C1"]["deep_pair_frames"] == 80058                                      # the card's baseline, reproduced
    assert ov["C1"]["by_human_state"]["human_separated"] == 7746
    assert ov["V2"]["deep_pair_frames"] == 79943
    assert ov["V2"]["by_human_state"]["human_separated"] == 1962                      # a quarter of C1's
    assert ov["V2"]["acc_spike_frames"] == 2                                          # the fit's own


def test_crowns_and_floats(acceptance):
    cr, sup = acceptance["replay"]["crowns"], acceptance["replay"]["supports"]
    assert cr["n"] == 27 and cr["within_crown_m"]["V2"] == 27 and cr["within_crown_m"]["C1"] == 0
    assert sup["n"] == 919
    assert sup["C1g"]["over2"] == 398                                                 # shipped plant, same fit
    assert sup["V2g"]["over2"] == 254                                                 # <= 261; the target 200 is not met
    if "C3g" in sup:
        assert sup["C3g"]["over2"] == 261                                             # the card's hybrid reference
    assert sup["V2"]["over2"] == 56                                                   # her own root, ungrounded


def test_statics_and_com(acceptance):
    st = acceptance["statics"]
    assert st["shipped_reground"]["lp_holdable"] == 41                                # the card's baseline, reproduced
    assert (st["v2_reground"]["lp_holdable"], st["v2_reground0"]["lp_holdable"], st["v2_raw"]["lp_holdable"]) == (95, 164, 251)
    # the raised 30 / 15 N m wrist / hand limits still bind on 1 / 2 / 3 LP-optimal holds, all cleared by <= 1.3x
    assert [st[a]["torque_sweep"]["wrist_hand_limited"] for a in ("v2_raw", "v2_reground", "v2_reground0")] == [1, 2, 3]
    assert max(max(st[a]["torque_sweep"]["k_needed_sorted"]) for a in ("v2_raw", "v2_reground", "v2_reground0")) < 1.3
    assert acceptance["com_cop"]["v2"]["p50"] <= 1.3 and acceptance["com_cop"]["v2"]["p50"] == pytest.approx(1.0, abs=0.02)
    assert acceptance["com_cop"]["shipped"]["p50"] == pytest.approx(1.91, abs=0.02)
    jb = acceptance["joint_box_exemplars"]
    assert (jb["shipped"]["past_2deg"], jb["v2"]["past_2deg"]) == (282, 100)
