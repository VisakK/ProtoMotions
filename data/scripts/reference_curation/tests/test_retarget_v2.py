# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the references on plant v2 (BodyFix Step 3): the writer (``fit_writer``) and the small retarget
(``retarget_v2``).

They pin what the build measured: the corpus and its drops; the writer putting every body on her SMPL-X joint
with her own root (Tree -a's standing foot 1.3 cm in the floor, ungrounded) and the plant's identity; the plant
context redirecting ``retarget``'s caches and the stores refused inside it; every residual's analytic Jacobian,
the new cross-frame ones included, against finite differences, and the banded normal equations against a dense
``J^T J``; the balance gate's stable-support rule; one full retarget (Garland -a) with the card's exit numbers;
and the committed corpus record's acceptance. Nothing under ``output/`` or ``data/reference_curation/`` is
written.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from reference_curation import fit_writer as fw
from reference_curation import human_mesh as hm
from reference_curation import ids
from reference_curation import mosh_replay as mr
from reference_curation import retarget as rt
from reference_curation import retarget_v2 as r2

TREE = "220923_Tree_Pose_or_Vrksasana_-a"
GARLAND = "220923_Garland_Pose_or_Malasana_-a"

needs_data = pytest.mark.skipif(not (ids.MOYO_DATA / "mosh").exists() or not hm.MODEL_PATH.exists()
                                or not mr.V2_XML.exists(), reason="the MoSh fits, the SMPL-X model or plant v2 are not on disk")


@pytest.fixture(scope="module")
def labels():
    try:
        return r2.load_labels()
    except FileNotFoundError:
        pytest.skip("no labels v1.1 folder")


# --------------------------------------------------------------------------- #
# The writer
# --------------------------------------------------------------------------- #
@needs_data
def test_the_corpus_drops_the_lotus_clips_the_unreadable_fit_and_firefly_b():
    stems, dropped = fw.corpus()
    assert len(stems) == 56 and set(dropped) == set(fw.DROPPED)
    assert not set(stems) & set(dropped)


@needs_data
def test_the_writer_writes_her_fit_unchanged_on_plant_v2():
    shipped = torch.load(ids.motion_path(TREE), map_location="cpu", weights_only=False)
    m, kin = fw.fit_motion(TREE)
    assert set(m) == set(shipped) | {"plant_sha256"}
    assert m["plant_sha256"] == fw.plant_identity.sha256("v2") and m["fps"] == 60
    for k, v in shipped.items():
        if torch.is_tensor(v):
            assert m[k].dtype == v.dtype and m[k].shape[1:] == v.shape[1:], k
    rtp = fw.round_trip(m)
    assert fw.round_trip_ok(rtp) and rtp["pos_m"] < 1e-6               # measured 4.4e-7 (float32 storage)
    # her own skeleton: every body origin on her SMPL-X joint (hands: the finger bases' mean), her own root
    assert np.abs(m["rigid_body_pos"].double().numpy() - mr.body_joint_targets(kin)).max() < 1e-6
    assert np.abs(m["rigid_body_pos"][:, 0].double().numpy() - kin["root_pos"]).max() < 1e-6
    # nothing grounded: the standing foot sits in the floor as her skin does (Step 2: median -1.0 cm, worst -4.6)
    low = fw.body_lowest(m["rigid_body_pos"].double().numpy(), m["rigid_body_rot"].double().numpy())
    assert round(100 * float(np.median(low.min(1))), 1) == -1.3
    assert torch.equal(m["rigid_body_contacts"], torch.as_tensor(low <= fw.AVATAR_TOUCH_M))
    # the fit asks for coordinates past the box (the right toe's roll): the retarget brings them in
    assert int((fw.box_excess_deg(m["dof_pos"].double().numpy()) > 1).any(1).sum()) == 138


@needs_data
def test_the_plant_context_redirects_and_restores_the_caches():
    v1 = rt.skeleton().offsets.clone()
    with r2.on_plant() as sk:
        assert torch.equal(sk.offsets, mr.skeleton_for(mr.V2_XML).offsets)
        assert not torch.equal(sk.offsets, v1)
        assert abs(float(rt.mass_model()[0].sum()) - 74.0) < 1e-3
        with pytest.raises(RuntimeError):
            r2.human_evidence(TREE)                    # a stale store would be rebuilt on plant v2
    assert torch.equal(rt.skeleton().offsets, v1)


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
def _window(prob: r2.Problem, a0: int, a1: int) -> r2.Problem:
    """``prob`` restricted to frames ``a0 .. a1 - 1`` (every frame-indexed plan re-based)."""
    sl = slice(a0, a1)
    plan = {k: (v[sl] if isinstance(v, np.ndarray) and v.shape[:1] == (prob.T,) else v) for k, v in prob.plan.items()}
    fl = prob.plan["flat"]
    m = (fl["frames"] >= a0) & (fl["frames"] < a1)
    plan["flat"] = {**{k: v[m] for k, v in fl.items()}, "frames": fl["frames"][m] - a0}
    v = prob.vel
    m = (v["frames"] >= a0) & (v["frames"] < a1 - 1)
    vel = {"frames": v["frames"][m] - a0, "cand": v["cand"][m], "w": v["w"][m], "d0": v["d0"][torch.as_tensor(m)]}
    b = prob.balance
    m = (b["frames"] >= a0) & (b["frames"] < a1)
    bal = {"frames": b["frames"][m] - a0, "w": b["w"][m], "margin": b["margin"][m] + 0.05}   # wide: rows active
    c = prob.close
    m = (c["frames"] >= a0) & (c["frames"] < a1)
    close = {k: (val[m] if isinstance(val, np.ndarray) and len(val) == len(c["frames"]) else val) for k, val in c.items()}
    close["frames"] = close["frames"] - a0
    keep = np.unique(close["item"])
    close["w"] = c["w"][keep] if len(keep) else np.zeros(0)
    close["item"] = np.searchsorted(keep, close["item"])
    pen = {**prob.pen, "floor_tp": prob.pen["floor_tp"][sl], "soft_tp": prob.pen["soft_tp"][sl]}
    smooth_w = None if prob.smooth_w is None else prob.smooth_w[a0:a1 - 2]
    return dataclasses.replace(prob, root_pos0=prob.root_pos0[sl], root_rot0=prob.root_rot0[sl], dof0=prob.dof0[sl],
                               pos0=prob.pos0[sl], rot0=prob.rot0[sl], plan=plan, close=close, vel=vel, balance=bal,
                               pen=pen, smooth_w=smooth_w)


@pytest.fixture(scope="module")
def tree_window(labels):
    torch.set_num_threads(4)
    anns = {h: a for h, a in labels["anns"].items() if h.startswith(TREE + "@")}
    ev = r2.human_evidence(TREE)
    human, _, _ = hm.load_human(TREE)
    with r2.on_plant():
        fit, _ = fw.fit_motion(TREE)
        prob = r2.build_problem(TREE, anns, ev, human, fit)
        sub = _window(prob, 380, 400)
        torch.manual_seed(0)
        x = rt.initial(sub) + 0.01 * torch.randn(sub.T, r2.NV, dtype=torch.float64)
        x[:, :3] *= 0.1
    return prob, sub, x


@needs_data
def test_the_plan_reads_her_mesh(tree_window):
    prob, _, _ = tree_window
    # her request for the shin on the thigh is the colliders' geometry (3.2 cm apart on her own pose), her foot
    # in the thigh is closed by the guard
    st = {r["contact"]: r["status"] for r in prob.requests}
    assert st == {"L_FOOT+R_THIGH": "closed", "L_SHANK+R_THIGH": "geometry_incompatible"}
    # the standing right foot rests flat on every frame (all four corners of its sole touch; the box is 11 deg
    # off it), its toes on 95 % of them, the lifted left foot only before the lift
    flat = prob.plan["flat"]
    names = mr.skeleton_for(mr.V2_XML).names
    full = flat["w"] > 0.999
    n = {names[b]: int((flat["body"][full] == b).sum()) for b in np.unique(flat["body"][full])}
    assert n["R_Ankle"] == prob.T and n["R_Toe"] > 0.9 * prob.T and n["L_Ankle"] < 200
    # the toes rest on their sole (her toe roll removed from the face choice), not on a side
    toes = np.isin(flat["body"], [names.index("L_Toe"), names.index("R_Toe")])
    assert (flat["axis"][toes] == 2).all() and (flat["sign"][toes] == -1).all()
    assert len(prob.balance["frames"]) > 0.9 * prob.T                   # a slow pose: quasi-static throughout
    # the guard: her left sole is pressed 5 cm into the right thigh (soft tissue, the human in contact), so that
    # pair may not get deeper than in her fit and is only asked apart by preference (pen_soft); everywhere else
    # no overlap deeper than 5 mm
    pen = prob.pen
    k = pen["index"][names.index("L_Ankle"), names.index("R_Hip")]
    assert pen["soft_tp"][:, k].sum() > 500 and pen["floor_tp"][:, k].min() < -0.045
    assert (pen["floor_tp"][~pen["soft_tp"]] == -r2.PEN_TOL_M).all()
    assert (pen["floor_tp"] <= -r2.PEN_TOL_M + 1e-12).all()


def test_the_weighted_smoothness_is_d_transpose_w_d():
    rng = np.random.default_rng(0)
    T = 11
    ws = rng.uniform(0.5, 30.0, T - 2)
    D = np.zeros((T - 2, T))
    for r in range(T - 2):
        D[r, r:r + 3] = (1.0, -2.0, 1.0)
    M = D.T @ np.diag(ws) @ D
    d0, d1, d2 = r2.smooth_diagonals(T, ws)
    assert np.allclose(d0, np.diag(M)) and np.allclose(d1, np.diag(M, 1)) and np.allclose(d2, np.diag(M, 2))
    for a, b in zip(r2.smooth_diagonals(T, np.ones(T - 2)), rt._smooth_diagonals(T)):
        assert np.array_equal(a, b)


@needs_data
def test_every_residual_matches_finite_differences(tree_window):
    _, sub, x = tree_window
    with r2.on_plant():
        res = r2.residuals(sub, x, jacobian=True)
        assert all(len(res[k]["r"]) for k in ("support", "pen_soft", "flat", "balance", "vel", "anchor", "ends"))
        eps, checked = 1e-6, 0
        for name, blk in res.items():
            if not len(blk["r"]):
                continue
            scale = max(float(blk["J"].abs().max()), 1.0)
            for col in (0, 4, 10, 30, 57, 70):
                for fr in (3, 7):
                    xp, xm = x.clone(), x.clone()
                    xp[fr, col] += eps
                    xm[fr, col] -= eps
                    rp, rm = r2.residuals(sub, xp)[name]["r"], r2.residuals(sub, xm)[name]["r"]
                    if len(rp) != len(blk["r"]) or len(rm) != len(blk["r"]):
                        continue                            # an active set changed: not a smooth point
                    an = torch.zeros(len(rp), dtype=torch.float64)
                    on = torch.as_tensor(blk["frames"] == fr)
                    an[on] = blk["J"][on, col]
                    if "J2" in blk:                         # the cross-frame rows of vel
                        on2 = torch.as_tensor(blk["frames2"] == fr)
                        an[on2] += blk["J2"][on2, col]
                    assert float(((rp - rm) / (2 * eps) - an).abs().max()) < 1e-6 * scale, (name, col, fr)
                    checked += 1
        assert checked >= 60


@needs_data
def test_the_banded_normal_equations_are_the_dense_ones(tree_window):
    _, sub, x = tree_window
    nv, T = r2.NV, sub.T
    with r2.on_plant():
        res = r2.residuals(sub, x, jacobian=True)
        ab, _ = r2.normal_equations(sub, x)
    rows = []
    for blk in res.values():
        if not len(blk["r"]):
            continue
        M = torch.zeros(len(blk["r"]), T * nv, dtype=torch.float64)
        for i, f in enumerate(blk["frames"]):
            M[i, f * nv:(f + 1) * nv] = blk["J"][i]
        if "J2" in blk:
            for i, f in enumerate(blk["frames2"]):
                M[i, f * nv:(f + 1) * nv] += blk["J2"][i]
        rows.append(M)
    J = torch.cat(rows)
    dense = (J.T @ J).numpy()
    n = T * nv
    full = np.zeros((n, n))
    for k in range(r2.BAND + 1):
        d = r2.BAND - k
        i = np.arange(d, n)
        full[i - d, i] = ab[k, i]
    # the linear terms (dof, root, smoothness) add only same-variable entries: compare everything else
    off_var = np.ones((n, n), bool)
    v = np.arange(n) % nv
    off_var[v[:, None] == v[None, :]] = False
    up = np.triu(np.ones((n, n), bool)) & (np.subtract.outer(np.arange(n), np.arange(n)) >= -r2.BAND)
    sel = off_var & up
    assert np.abs(full[sel] - dense[sel]).max() < 1e-9 * np.abs(dense).max()
    cross = [np.abs(full[t * nv:(t + 1) * nv, (t + 1) * nv:(t + 2) * nv]).max() for t in range(T - 1)]
    assert min(cross) > 0                                 # vel couples every neighbouring pair of frames here


@needs_data
def test_the_gradient_is_the_energys_with_local_smoothing(tree_window):
    _, sub, x = tree_window
    sub = dataclasses.replace(sub, smooth_w=np.random.default_rng(1).uniform(0.5, 50.0, sub.T - 2))
    with r2.on_plant():
        _, g = r2.normal_equations(sub, x)

        def half_sq(xx):
            return 0.5 * r2.energy(sub, xx)[0] * sub.T

        eps = 1e-6
        for fr, col in ((2, 0), (5, 4), (9, 10), (13, 33), (17, 70), (0, 57)):
            xp, xm = x.clone(), x.clone()
            xp[fr, col] += eps
            xm[fr, col] -= eps
            fd = (half_sq(xp) - half_sq(xm)) / (2 * eps)
            assert abs(fd - g[fr * r2.NV + col]) < 1e-6 * max(abs(g[fr * r2.NV + col]), 1.0)   # measured 2e-8


def test_the_balance_gate_wants_a_stable_support_set():
    sk = mr.skeleton_for(mr.V2_XML)
    T, K = 100, len(sk.cand_body)
    tw = np.zeros((T, K))
    feet = [c for z in ("L_FOOT", "R_FOOT") for c in sk.zone_cands[r2.ZONE_ORDER.index(z)]]
    tw[:, feet] = 1.0
    left = sk.zone_cands[r2.ZONE_ORDER.index("L_FOOT")]
    tw[50:, left] = 0.0                                   # the left foot lifts at frame 50
    st = r2.stable_support(sk, tw, half=15)
    assert st[:35].all() and not st[35:65].any() and st[65:].all()


# --------------------------------------------------------------------------- #
# One clip, end to end, and the corpus record
# --------------------------------------------------------------------------- #
@needs_data
def test_garland_retargets_within_the_cards_limits(labels, tmp_path):
    corpus, dropped = fw.corpus()
    fit_dir, fit_rec, failures = fw.build([GARLAND], dropped, workers=1, out_root=tmp_path / "fit",
                                          record_root=tmp_path / "fit_rec")
    assert failures == []
    out, records, failures = r2.run(labels, [GARLAND], fit_dir, "test", workers=1, out_root=tmp_path / "rt")
    assert failures == [] and r2.check(records, out) == []
    m = records[0]["metrics"]
    f, a = m["fit"], m["after"]
    assert (f["frames_past_box_1deg"], a["frames_past_box_1deg"]) == (274, 0)
    assert a["pen_zone_frames"] == 0 and a["float2_zone_frames"] == 0 and f["pen_zone_frames"] > 800
    assert m["new_spike_frames"] == 0 and m["new_deep_overlaps"] == 0 and a["deep_overlap_pair_frames"] == 0
    assert a["floor_min_cm"] >= -0.5
    assert a["slide"]["p90_cm_s"] <= 4.5
    assert m["edit"]["mean_disp_cm_p50"] < 1.2 and m["edit"]["body_disp_cm_max"] < 7.0      # measured 0.85, 6.40
    assert a["flat_face_tilt_deg_p50_p90"][0] < 1.0 < f["flat_face_tilt_deg_p50_p90"][0]   # 13.3 -> 0.2 deg
    assert a["com_outside_frames"] == 0
    motion = torch.load(out / f"{GARLAND}.motion", map_location="cpu", weights_only=False)
    assert motion["plant_sha256"] == fw.plant_identity.sha256("v2")


RECORD_ID = "holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828.retarget_v2.e4278ada02"


def _record_dir():
    d = ids.DATA_ROOT / "retarget_v2" / RECORD_ID
    if not d.exists():
        pytest.skip(f"the committed retarget_v2 record {RECORD_ID} is not on disk")
    return d


def _record() -> dict:
    return json.loads((_record_dir() / "retarget.json").read_text())


def test_the_corpus_record_meets_the_cards_acceptance_but_one():
    """The committed record (``retarget_v2.e4278ada02``): 9 of the card's 10 tests pass; the miss is pinned so a
    change in either direction shows. Worst body per clip: p50 10.6 cm against Step 8's 15.5 (the bar is half).
    Firefly -b is dropped (the user's decision): its own motion folds the trunk 24.9 deg past the box (Torso_y), and
    it alone kept new overlaps (14 pair-frames) in the 57-clip run, whose other 56 motions this one reproduces."""
    rec = _record()
    acc = {k: v for k, v in rec["acceptance"].items()}
    assert {k for k, v in acc.items() if not v["passed"]} == {"edit_body_disp_cm_max_p50"}
    m = rec["metrics"]
    assert m["clips"] == 56 and m["frames"] == 81512 and set(rec["dropped"]) == set(fw.DROPPED)
    assert (m["pooled"]["fit_frames_past_box_1deg"], m["pooled"]["after_frames_past_box_1deg"]) == (26532, 0)
    assert (m["supports"]["n"], m["supports"]["fit"]["over2"], m["supports"]["after"]["over2"]) == (864, 48, 1)
    assert m["supports"]["crowns"] == 21 and m["supports"]["after"]["crowns_within_2cm"] == 21
    assert m["touch_penetration"]["fit_share"] == 0.4304 and m["touch_penetration"]["after_share"] <= 0.0001
    assert m["slide"]["after"]["clip_median_of_p90_cm_s"] <= 3.0 and m["slide"]["after"]["share_gt_5cm_s"] <= 0.07
    assert m["pooled"]["new_spike_frames"] == 0
    assert (m["pooled"]["fit_deep_overlap_pair_frames"], m["pooled"]["new_deep_overlaps"]) == (63201, 0)
    new = {json.loads(line)["stem"] for line in open(_record_dir() / "clips.jsonl")
           if json.loads(line)["metrics"]["new_deep_overlaps"]}
    assert not new
    assert m["edit"]["mean_disp_cm_p50"]["p50"] == 1.368 and m["edit"]["body_disp_cm_max"]["p50"] == 10.587
    px = json.loads((_record_dir() / "physx.json").read_text())
    assert px["failures"] == [] and px["fk"]["pos_max_err_m"] < 1.5e-6 and px["reset"]["launched"] == 0
    assert px["reset"]["positive_control"]["seen_as_launch"] == 3 and px["library"]["v1_refused"]
