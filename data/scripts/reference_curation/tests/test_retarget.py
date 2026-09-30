# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plant checks and the contact-constrained retarget (BUILD_PLAN Step 8).

They pin what the build measured: PhysX's exp-map joint coordinates and hard limits on the recorded
rollouts, the references outside the plant's box, the avatar's rotations being the performer's, the
analytic Jacobians against finite differences, the support plan's faces on a flat foot and on the toes, the
head collider's refusal, and one full retarget (Standing Split -a, whose standing foot floats 11.45 cm)
with its round trip. Nothing under ``output/``, ``data/reference_curation/`` or ``REVIEW_ROOT`` is written.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import dataclasses
import glob

import numpy as np
import pytest
import torch

from reference_curation import human_mesh as hm, ids, plant, retarget as rt, statics

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
WARRIOR_II = "220926_Warrior_II_Pose_or_Virabhadrasana_II_-a"
PLANK = "220923_Plank_Pose_or_Kumbhakasana_-a"
STANDING_SPLIT = "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a"
HEADSTAND_B = "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b"

needs_data = pytest.mark.skipif(not ids.SHIPPED_DIR.exists() or not hm.MODEL_PATH.exists(),
                                reason="the shipped ftC clips or the SMPL-X model are not on disk")


@pytest.fixture(scope="module")
def labels():
    try:
        return statics.load_labels(statics.default_labels_dir())
    except FileNotFoundError:
        pytest.skip("no labels v1 folder")


@pytest.fixture(scope="module")
def crow_problem(labels):
    torch.set_num_threads(4)
    return rt.build_problem(CROW, labels["anns"])


# --------------------------------------------------------------------------- #
# The training plant
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rollout():
    paths = [p for p in plant.rollouts() if CROW in str(p)]
    if not paths:
        pytest.skip("no recorded PhysX rollout on disk")
    return plant._load(paths[0])


def test_physx_joint_coordinates_are_the_exp_map(rollout):
    c = plant.joint_convention(rollout)
    assert c["max_err_rotvec_rad"] < 1e-5                 # measured 7e-7 on every frame
    assert c["max_err_xyz_euler_rad"] > 1.0                # not XYZ Euler: up to 3 rad apart


def test_physx_limits_are_a_hard_box(rollout):
    lim = plant.limit_hardness(rollout)
    assert lim["max_violation_deg"] < 1.1                  # measured 0.79 on this rollout, 1.04 over all 13
    assert lim["frames_past_2deg"] == 0
    assert lim["at_stop_with_saturated_actuator"] > 0      # the actuator pushes into the stop and loses


def test_the_references_ask_for_joint_positions_outside_the_box(labels):
    r = plant.reference_violations(labels)
    assert (r["exemplars_past_range_expmap"], r["exemplars_past_range_xyz"], r["holds"]) == (285, 290, 303)


def test_the_gated_lp_explains_physx_ground_loads(rollout):
    pos, quat, perm = plant.common_order(rollout)
    weight = float(rollout["body_masses"].sum()) * 9.81
    inside = n = 0
    for f in (300, 600, 900, 1200):
        g, p = plant.physx_contacts(rollout, perm, f)
        r = plant.force_frame(pos[f - 1:f + 2], quat[f - 1:f + 2], float(rollout["dt_phys"]), g, p, weight)
        for row in r["rows"]:
            if row["kind"] == "ground" and "inside" in row:
                inside, n = inside + row["inside"], n + 1
    assert n >= 8 and inside / n >= 0.9                    # 95.3 % of 20,015 ground contact-frames corpus-wide


# --------------------------------------------------------------------------- #
# Kinematics and geometry
# --------------------------------------------------------------------------- #
@needs_data
def test_fk_reproduces_the_shipped_bodies():
    ref = rt.load_reference(CROW)
    pos, rot = rt.fk(rt.skeleton(), ref["pos"][:, 0], ref["rot"][:, 0], ref["dof"])
    assert float((pos - ref["pos"]).abs().max()) < 1e-6    # measured 3.3e-7 m (float32 storage)
    assert float((rot - ref["rot"]).abs().max()) < 2e-5


@needs_data
def test_avatar_rotations_are_the_performers_up_to_one_permutation():
    from scipy.spatial.transform import Rotation as R

    human, _, _ = hm.load_human(WARRIOR_II)
    ref = rt.load_reference(WARRIOR_II)
    frames = np.arange(0, ref["dof"].shape[0], 30)
    _, A = human.model._joints(human.fit["v_template"], human.fit["fullpose"][frames])
    sk = rt.skeleton()
    for body in ("Pelvis", "L_Hip", "R_Knee", "Chest", "Head", "L_Elbow"):
        g = A[:, rt.BODY_JOINT[body], :3, :3]
        c = np.einsum("fji,fjk->fik", g, ref["rot"].numpy()[frames, sk.names.index(body)])
        dev = R.from_matrix(np.einsum("ji,fjk->fik", rt.AXES, c)).magnitude()
        assert np.degrees(np.median(dev)) < 3.0            # measured 0.2-1.9 deg


def test_jacobians_match_finite_differences(crow_problem):
    prob = crow_problem
    a0, a1 = 640, 660
    sl = slice(a0, a1)

    def cut(d):
        n = len(d["frames"])
        m = (d["frames"] >= a0) & (d["frames"] < a1)
        out = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in d.items()}
        out["frames"] = out["frames"] - a0
        return out

    close = cut(prob.close)
    keep = np.unique(close["item"])
    close["w"] = prob.close["w"][keep] if len(keep) else np.zeros(0)
    close["item"] = np.searchsorted(keep, close["item"])
    plan = {k: (v[sl] if isinstance(v, np.ndarray) else v) for k, v in prob.plan.items()}
    sub = dataclasses.replace(prob, root_pos0=prob.root_pos0[sl], root_rot0=prob.root_rot0[sl], dof0=prob.dof0[sl],
                              pos0=prob.pos0[sl], rot0=prob.rot0[sl], plan=plan, close=close)
    torch.manual_seed(0)
    x = rt.initial(sub) + 0.01 * torch.randn(sub.T, rt.NV, dtype=torch.float64)
    x[:, :3] *= 0.1
    res = rt.residuals(sub, x, jacobian=True)
    eps = 1e-6
    checked = 0
    for name, blk in res.items():
        if not len(blk["r"]):
            continue
        scale = float(blk["J"].abs().max())
        for col in (0, 4, 10, 30, 57, 70):
            xp, xm = x.clone(), x.clone()
            xp[:, col] += eps
            xm[:, col] -= eps
            rp, rm = rt.residuals(sub, xp)[name]["r"], rt.residuals(sub, xm)[name]["r"]
            if len(rp) != len(blk["r"]) or len(rm) != len(blk["r"]):
                continue                                    # an active set changed: not a smooth point
            fd = (rp - rm) / (2 * eps)
            assert float((fd - blk["J"][:, col]).abs().max()) < 1e-6 * max(scale, 1.0), name
            checked += 1
    assert checked >= 20


@needs_data
def test_a_knee_folded_past_180_deg_stays_folded():
    sk = rt.skeleton()
    ref = rt.load_reference("220923_Cockerel_Pose-b")
    rep = rt.nearest_representative(ref["dof"], sk.lower + rt.LIMIT_MARGIN_RAD, sk.upper - rt.LIMIT_MARGIN_RAD)
    changed = (rep != ref["dof"]).reshape(ref["dof"].shape[0], -1, 3).any(-1)
    assert int(changed.sum()) == 914 and int(changed[:, sk.names.index("R_Knee") - 1].sum()) == 914
    same = rt.so3_exp(rep.reshape(-1, 3)).transpose(-1, -2) @ rt.so3_exp(ref["dof"].reshape(-1, 3))
    assert torch.allclose(same, torch.eye(3, dtype=same.dtype).expand_as(same), atol=1e-9)   # the same rotations


def test_intersecting_boxes_separate_the_way_they_came():
    from scipy.spatial.transform import Rotation

    F = torch.float64
    half = torch.tensor([0.1, 0.05, 0.03], dtype=F)
    a = {"type": "box", "c": torch.zeros(1, 3, dtype=F), "R": torch.eye(3, dtype=F)[None], "half": half}
    b = {"type": "box", "c": torch.tensor([[0.02, 0.0855, 0.005]], dtype=F), "R": torch.eye(3, dtype=F)[None],
         "half": half}
    gap, _, _, u = rt.geom_gap(a, b)                          # two feet side by side, 1.45 cm into each other
    assert float(gap[0]) == pytest.approx(-0.0145) and torch.allclose(u[0], torch.tensor([0.0, 1.0, 0.0], dtype=F))
    torch.manual_seed(0)
    n = 200
    rb = torch.as_tensor(Rotation.random(n, random_state=1).as_matrix())
    a = {"type": "box", "c": torch.zeros(n, 3, dtype=F), "R": torch.as_tensor(Rotation.random(n, random_state=0).as_matrix()),
         "half": torch.tensor([0.12, 0.05, 0.04], dtype=F)}
    cb, w, v = (torch.randn(n, 3, dtype=F) * k for k in (0.05, 1.0, 1.0))
    moved = lambda t: {"type": "box", "c": cb + t * v, "half": torch.tensor([0.08, 0.06, 0.03], dtype=F),  # noqa: E731
                       "R": torch.as_tensor(Rotation.from_rotvec((t * w).numpy()).as_matrix()) @ rb}
    gap, pa, pb, u = rt.geom_gap(a, moved(0.0))
    inside = gap < 0
    assert int(inside.sum()) > 100
    fd = (rt.geom_gap(a, moved(1e-6))[0] - rt.geom_gap(a, moved(-1e-6))[0]) / 2e-6
    pred = (u * (v + torch.linalg.cross(w, pb - cb))).sum(1)     # d gap = u . (d pb - d pa), a fixed
    assert float((fd - pred)[inside].abs().max()) < 1e-6
    assert torch.allclose((u * (pb - pa)).sum(1), gap)


def test_the_guard_covers_every_pair_the_plant_collides():
    sk = rt.skeleton()
    pairs = {tuple(p) for p in rt.plant_pairs().tolist()}
    assert len(pairs) == 253                                # 276 pairs of 24 bodies, less 23 jointed ones
    n = sk.names.index
    assert (n("Chest"), n("L_Shoulder")) in pairs and (n("Pelvis"), n("Spine")) in pairs   # adjacent zones
    assert (n("L_Knee"), n("L_Ankle")) not in pairs         # a joint: PhysX filters it
    pos, rot = rt.fk(sk, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
                     torch.eye(3, dtype=torch.float64)[None], torch.zeros(1, 69, dtype=torch.float64))
    assert len(rt.near_body_pairs(sk, pos, rot, 0.0)[0]) == 0                    # the rest pose overlaps nothing


def test_support_plan_rests_a_flat_foot_flat_and_a_plank_on_its_toes(labels):
    sk = rt.skeleton()
    ankle, toe = sk.names.index("L_Ankle"), sk.names.index("L_Toe")
    wp = rt.build_problem(WARRIOR_II, labels["anns"])
    t = wp.plan["target"][572]
    assert t[sk.cand_body == ankle].sum() == 4 and t[sk.cand_body == toe].sum() == 4   # both boxes flat
    pp = rt.build_problem(PLANK, labels["anns"])
    t = pp.plan["target"][541]
    assert t[sk.cand_body == ankle].sum() == 0             # the heel is up
    assert t[sk.cand_body == toe].sum() >= 2               # on the toes


def test_the_head_collider_cannot_take_a_crown(labels):
    hb = rt.build_problem(HEADSTAND_B, labels["anns"])
    assert int(hb.plan["head_blocked"].sum()) == 1115       # every crown-down frame of the headstand
    wp = rt.build_problem(WARRIOR_II, labels["anns"])
    assert int(wp.plan["head_blocked"].sum()) == 0


def test_a_jerk_is_counted_and_the_reference_is_not(labels):
    prob = rt.build_problem(WARRIOR_II, labels["anns"])
    x = rt.initial(prob)
    assert rt.spike_frames(prob, x) == 0                    # the reference, clipped into the box: no jerk
    x[400, 6 + 3 * (rt.skeleton().names.index("R_Elbow") - 1)] += 0.3    # one elbow 17 deg off on one frame
    assert rt.spike_frames(prob, x) >= 1


def test_runs_are_cleaned_and_ramped():
    m = np.zeros((40, 2), bool)
    m[10:25, 0] = True
    m[0:3, 1] = m[30:33, 1] = m[35:40, 1] = True
    c = rt.clean_runs(m)
    assert c[:, 0].sum() == 15 and not c[:3, 1].any() and c[30:40, 1].all()    # gap filled, short run dropped
    w = rt.ramp_weights(c)
    assert w[5, 0] < 0.05 and w[10, 0] == pytest.approx(0.5, abs=0.1) and w[17, 0] == 1.0
    assert np.all(np.diff(w[:18, 0]) >= 0) and w[39, 1] == 1.0                  # no ramp at the clip's end


# --------------------------------------------------------------------------- #
# One full retarget
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def standing_split(labels):
    torch.set_num_threads(4)
    anns = {h: a for h, a in labels["anns"].items() if h.startswith(STANDING_SPLIT + "@")}
    return rt.retarget_clip(STANDING_SPLIT, anns)


def test_standing_split_stands_on_its_foot(standing_split):
    m = standing_split["record"]["metrics"]
    assert m["before"]["floating_gt_2cm"] == pytest.approx(0.4321, abs=1e-4)
    assert m["after"]["floating_gt_2cm"] < 0.01             # measured 0.0067
    assert m["after"]["agreement"] > 0.99 and m["before"]["agreement"] < 0.93
    assert m["before"]["limit_excess_max_deg"] > 40 and m["after"]["limit_excess_max_deg"] == 0.0
    assert m["after"]["floor_min_cm"] > -0.5 and m["new_deep_overlaps"] == 0
    assert m["before"]["overlap_max_cm"] > 3.0 and m["after"]["overlap_max_cm"] < 1.0   # the head in the collar
    motion = standing_split["motion"]
    z = rt.zone_min_z(motion["rigid_body_pos"].double().numpy()[540:541], motion["rigid_body_rot"].double().numpy()[540:541])
    assert z[0, rt.ZI["L_FOOT"]] <= 0.01                    # shipped: 11.45 cm at 9.0 s


def test_the_retarget_round_trips(standing_split):
    rt_ = rt.round_trip(standing_split["motion"])
    assert rt_["finite"] and rt_["pos_m"] < rt.ROUND_TRIP_M and rt_["rot"] < rt.ROUND_TRIP_M and rt_["local"] < rt.ROUND_TRIP_M


def test_an_unchanged_clip_is_returned_bit_for_bit(labels):
    prob = rt.build_problem(WARRIOR_II, labels["anns"])
    ref = rt.load_reference(WARRIOR_II)
    inside = dataclasses.replace(prob, dof0=torch.maximum(torch.minimum(prob.dof0, prob.upper), prob.lower))
    out = rt.regenerate(inside, rt.initial(inside), ref["motion"])
    assert all(torch.equal(out[k], ref["motion"][k]) for k in ref["motion"] if torch.is_tensor(ref["motion"][k]))


def test_the_committed_corpus_meets_the_exit_criterion():
    import json

    mine = ids.sha256_file(rt.__file__)
    recs = [json.load(open(p)) for p in sorted(glob.glob(str(rt.RECORD_ROOT / "*" / "retarget.json")))]
    recs = [r for r in recs if r["generator"]["sha256"] == mine]
    if not recs:
        pytest.skip("no corpus retarget record from this retarget.py")
    m = recs[-1]["metrics"]
    src = m["exit"]["source"]
    assert src["before_over"] == 367 and src["supports"] == 833      # Step 2's 44 %
    assert src["after_over"] / src["supports"] < 0.05                  # BUILD_PLAN Step 8's exit
    assert src["after_over_reachable"] == 0                            # every float left is a crown or a removed support
    assert m["pooled"]["after_limit_frames"] == 0
