# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card R2 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: the synthetic clip writer (``splice_v3``).

Pure rules first (the variant's clock, the blend weights and the no-repeated-frame rule, the layout, the names and
their collisions, the root transform, the holds on a small synthetic case), then one real clip spliced in memory
(placement, lineage, a negative control), then ``--check`` on the committed build.

    OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest \\
        data/scripts/reference_curation/tests/test_splice_v3.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from reference_curation import hold_extension_v2 as hx
from reference_curation import splice_v3 as R

# The committed build (expert_revist/graph_growth_2026_10_03/r2_splice/README.MD); never "the newest folder".
SYNTHETIC_ID = "synthetic_v3.d034199f23"
HAVE_INPUTS = R.SELECTED.exists() and R.RELEASE_V2_RECORD.exists()


# --------------------------------------------------------------------------- #
# The variant's clock and the blends
# --------------------------------------------------------------------------- #
def test_variant_clock_matches_the_card():
    t_first = -0.3 + 1 / 60
    mid = R.variant_clock(204, t_first, 1.075, 0.3, 2.0)          # the card's example: B1 mid
    assert (mid["k_d"], mid["k_a"], mid["N"], mid["hold_frames"]) == (17, 82, 204, 122)
    high = R.variant_clock(234, t_first, 1.6, 0.3, 2.0)           # 60 * 1.6 is not rounded up past 96
    assert (high["k_a"], high["hold_frames"]) == (113, 121)
    assert R.variant_clock(408, t_first, 4.5, 0.3, 2.0)["k_a"] == 287
    with pytest.raises(ValueError):
        R.variant_clock(204, -0.3, 1.075, 0.3, 2.0)                  # t_first must be -0.3 + 1/60
    with pytest.raises(ValueError):
        R.variant_clock(200, t_first, 1.075, 0.3, 2.0)                # the hold must run 2 s past T


def test_blend_weights_and_no_repeated_frame():
    ws = R.s_weights(17)
    assert ws.shape == (17,) and ws[-1] == 1.0 and 0 < ws[0] < 0.01 and np.all(np.diff(ws) > 0)
    k_a, N = 82, 204
    wd = R.d_weights(k_a, N)
    assert wd.shape == (N - k_a,) and 0 < wd[0] and wd[-1] < 1 and np.all(np.diff(wd) > 0)
    # the card's formula, smoothstep((k - k_a + 1) / (N - k_a)), reaches 1 on the last variant frame: that frame
    # would be D' itself and the lead-out's first frame (D') would repeat it
    card = R.smoothstep((np.arange(k_a, N) - k_a + 1.0) / (N - k_a))
    assert card[-1] == 1.0
    # a 1-D splice: variant values, then the D blend toward D' = 1, then D' as the lead-out's first frame
    variant = np.linspace(0.0, 0.2, N - k_a)
    spliced = np.r_[(1 - wd) * variant + wd * 1.0, 1.0]
    assert np.all(np.diff(spliced) != 0)
    repeated = np.r_[(1 - card) * variant + card * 1.0, 1.0]
    assert repeated[-2] == repeated[-1]


def test_layout_indices():
    clock = {"k_d": 17, "k_a": 82, "N": 204, "hold_frames": 122}
    lay = R.layout(335, clock, 389)
    assert lay["s_exemplar"] == 335 and lay["settle"] == [336, 352] and lay["departure"] == 335 + 18
    assert lay["transition"] == [353, 335 + 82] and lay["arrival"] == 335 + 1 + 82
    assert lay["d_hold"] == [418, 335 + 204] and lay["lead_out"] == [335 + 205, 335 + 204 + 389]
    assert lay["num_frames"] == 335 + 1 + 204 + 389
    parts = [lay["lead_in"], [lay["s_exemplar"]] * 2, lay["settle"], lay["transition"], lay["d_hold"], lay["lead_out"]]
    assert [p[0] for p in parts[1:]] == [p[1] + 1 for p in parts[:-1]]      # contiguous, in order
    assert lay["junctions"]["d_hold|lead_out"] == [539, 540]


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def _row(edge, timing, seed, tag):
    return {"edge": edge, "variant": f"SYN_{edge}_{edge}_{edge}_{timing}_s{seed}_{tag}",
            "recipe": {"timing": f"{edge}_{timing}", "seed": seed}}


def test_names_and_their_collisions():
    assert R.synthetic_name(_row("E1", "high", 0, "t6px")) == "SYN_E1_press_high_s0_t6px"
    assert R.synthetic_name(_row("B1", "high", 1, "t6rpx")) == "SYN_B1_jumpplank_high_s1_t6rpx"
    assert R.synthetic_name(_row("E5", "mid", 0, "t6rpx12")) == "SYN_E5_floatdown_mid_s0_t6rpx12"
    names = [R.synthetic_name(_row("E2", "mid", 1, t)) for t in R.TAGS]
    real = ["220923_Crane_Crow_Pose_or_Bakasana_-a"]
    chk = R.name_checks(names, real)
    assert chk["pass"] and chk["unique"]
    pairs = {tuple(p) for p in chk["name_collisions"]}
    assert ("SYN_E2_jumpback_mid_s1_t6px", "SYN_E2_jumpback_mid_s1_t6px12") in pairs
    assert ("SYN_E2_jumpback_mid_s1_t6px", "SYN_E2_jumpback_mid_s1_t6px12_x3s") in pairs
    assert ("SYN_E2_jumpback_mid_s1_t6rpx", "SYN_E2_jumpback_mid_s1_t6rpx12_x7s") in pairs
    assert not any(a == b.rsplit("_x", 1)[0] for a, b in pairs)      # never a clip's own duration variant
    assert sorted(names, key=R.order_key) == names                   # (edge, timing, seed, tag)
    assert not R.name_checks(names + [names[0]], real)["pass"]
    assert not R.name_checks(names + ["SYN_E2_x_220923_Crane_Crow_Pose_or_Bakasana_-a"], real)["pass"]
    assert not R.name_checks(names + ["SYN_E2_jumpback_mid_s1_x3s"], real)["pass"]
    assert not R.name_checks(names, real + ["SYN_human"])["pass"]
    with pytest.raises(ValueError):
        R.synthetic_name(_row("E1", "high", 0, "t7px"))


# --------------------------------------------------------------------------- #
# The transform
# --------------------------------------------------------------------------- #
def test_place_is_a_rigid_yaw_and_shift():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(5, 3))
    rot = Rotation.random(5, random_state=1).as_quat()
    yaw, t = 0.3, np.array([0.2, -0.1])
    p, q = R.place(pos, rot, yaw, t)
    assert np.allclose(p[:, 2], pos[:, 2])                              # no lift
    back, qb = R.place(p - np.r_[t, 0.0], q, -yaw, np.zeros(2))
    assert np.allclose(back, pos, atol=1e-12)
    assert np.allclose((Rotation.from_quat(qb).inv() * Rotation.from_quat(rot)).magnitude(), 0, atol=1e-12)
    v = rng.normal(size=(4, 3))
    assert np.allclose(R.rotate_vectors(v, yaw)[:, 2], v[:, 2])


@pytest.mark.skipif(not HAVE_INPUTS, reason="release v2 or the selection is not on disk")
def test_root_transform_fk_on_a_real_frame():
    """FK of the transformed root with unchanged dof_pos = the rigidly transformed bodies (float64)."""
    from reference_curation import fit_writer as fw

    sk = fw.skeleton("v2")
    rel = R.release_v2()
    m = R.release_motion(rel, "220923_Plank_Pose_or_Kumbhakasana_-a")
    r, q, d = R.frames_of(m, [541, 600])
    p0, _ = R.fk(sk, r, q, d)
    yaw, t = math.radians(-11.4), np.array([0.31, -0.07])
    rp, rq = R.place(r, q, yaw, t)
    p1, _ = R.fk(sk, rp, rq, d)
    want, _ = R.place(p0.reshape(-1, 3), np.tile(q[:1], (48, 1)), yaw, t)
    assert np.abs(p1.reshape(-1, 3) - want).max() < 1e-10
    assert np.abs(p0 - m["rigid_body_pos"][[541, 600]].double().numpy()).max() < 1e-5       # the stored FK


@pytest.mark.skipif(not HAVE_INPUTS, reason="release v2 or the selection is not on disk")
def test_blend_places_the_hands_midpoint_and_keeps_the_ends_exact():
    from reference_curation import fit_writer as fw

    sk = fw.skeleton("v2")
    rel = R.release_v2()
    m = R.release_motion(rel, "220923_Crane_Crow_Pose_or_Bakasana_-a")
    a = R.frames_of(m, [651, 651, 651, 651])
    b = R.frames_of(m, [700, 700, 700, 700])
    w = np.array([0.0, 0.3, 0.7, 1.0])
    rp, rq, dd = R.blend(sk, a, b, w)
    mid = (1 - w)[:, None] * R.hands_mid(sk, *a) + w[:, None] * R.hands_mid(sk, *b)
    assert np.abs(R.hands_mid(sk, rp, rq, dd) - mid).max() < 1e-9
    assert all(np.array_equal(x[0], y[0]) for x, y in zip((rp, rq, dd), a))
    assert all(np.array_equal(x[-1], y[-1]) for x, y in zip((rp, rq, dd), b))
    sk_lo, sk_hi = sk.lower.numpy(), sk.upper.numpy()
    assert np.all(dd[1:3] >= sk_lo - math.radians(1)) and np.all(dd[1:3] <= sk_hi + math.radians(1))


# --------------------------------------------------------------------------- #
# Holds on a small synthetic case
# --------------------------------------------------------------------------- #
def _hold(stem, fs, fh, fe, name, extend=False):
    return {"name": name, "frame_start": fs, "frame_hold": fh, "frame_end": fe, "hold_id": f"{stem}@{fh}",
            "pairs": ["L_HAND:G", "R_HAND:G"], "orientation": "prone", "extend": extend, "labels": {"x": 1},
            "gate": {"decision": "pass"}, "speed_at_hold": 0.01, "pelvis_z": 0.7}


def test_splice_holds_inherit_and_place_every_window():
    s_clip = {"stem": "S", "holds": [_hold("S", 0, 5, 9, "a"), _hold("S", 20, 30, 34, "b"),
                                     _hold("S", 40, 60, 80, "s", True), _hold("S", 90, 95, 99, "z")]}
    d_clip = {"stem": "D", "holds": [_hold("D", 0, 3, 5, "q"), _hold("D", 8, 15, 22, "d", True),
                                     _hold("D", 30, 33, 36, "e"), _hold("D", 40, 44, 49, "f")]}
    s_hold, d_hold = s_clip["holds"][2], d_clip["holds"][1]
    clock = {"k_d": 17, "k_a": 82, "N": 204, "hold_frames": 122}
    lead_in_start, lead_out_end = 20, 36                     # the last hold before S starts at 20; D's next ends at 36
    L = s_hold["frame_hold"] - lead_in_start
    lay = R.layout(L, clock, lead_out_end - d_hold["frame_hold"] + 1)
    holds = R.splice_holds("SYN_X", s_clip, s_hold, d_clip, d_hold, lay, lead_in_start, lead_out_end)
    assert [h["inherits"] for h in holds] == ["S@30", "S@60", "D@15", "D@33"]
    assert [h["extend"] for h in holds] == [False, True, False, False]
    b, s, d, e = holds
    assert (b["frame_start"], b["frame_hold"], b["frame_end"]) == (0, 10, 14)
    assert (s["frame_start"], s["frame_hold"], s["frame_end"]) == (20, L, L + 18)
    out0 = L + 1 + 204
    assert (d["frame_start"], d["frame_hold"], d["frame_end"]) == (L + 1 + 82, out0, out0 + 22 - 15)
    assert (e["frame_start"], e["frame_hold"], e["frame_end"]) == (out0 + 15, out0 + 18, out0 + 21)
    assert all(h["hold_id"] == f"SYN_X@{h['frame_hold']}" for h in holds)
    assert s["t_hold"] == round(L / 60, 4) and s["duration_s"] == round((L + 18 - 20 + 1) / 60, 3)
    assert all(h["labels"] == {"x": 1} and h["speed_at_hold"] == 0.01 for h in holds)
    assert R.holds_well_formed(holds, lay["num_frames"]) == []
    # hold extension then lengthens S only, before the departure
    plan = hx.insertion_plan(holds, 180)
    assert plan == [(L, 180)]
    ext = hx.variant_holds(holds, plan, 60)
    assert ext[1]["frame_end"] - ext[1]["frame_start"] == s["frame_end"] - s["frame_start"] + 180
    assert ext[2]["frame_start"] == d["frame_start"] + 180 and ext[0] == {**holds[0]}
    assert R.holds_well_formed(ext, lay["num_frames"] + 180, x0=False) == []
    # a broken case is reported
    bad = [dict(h) for h in holds]
    bad[2]["frame_start"] = bad[1]["frame_end"]
    assert R.holds_well_formed(bad, lay["num_frames"])
    # trimmed windows (content_windows): S ends earlier, D starts later, everything else as before
    trimmed = R.splice_holds("SYN_X", s_clip, s_hold, d_clip, d_hold, lay, lead_in_start, lead_out_end,
                             s_end=L + 5, d_start=L + 1 + 82 + 23)
    assert trimmed[1]["frame_end"] == L + 5 and trimmed[2]["frame_start"] == L + 1 + 82 + 23
    assert [h["frame_hold"] for h in trimmed] == [h["frame_hold"] for h in holds]
    assert R.holds_well_formed(trimmed, lay["num_frames"]) == []
    with pytest.raises(ValueError):
        R.splice_holds("SYN_X", s_clip, s_hold, d_clip, d_hold, lay, lead_in_start, lead_out_end, d_start=out0 + 1)


def test_content_windows_trim_to_the_label():
    """A hold's window may not hold a non-real frame touching the floor with a zone it does not label."""
    clock = {"k_d": 17, "k_a": 82, "N": 204, "hold_frames": 122}
    lay = R.layout(335, clock, 389)
    T = lay["num_frames"]
    contacts = np.zeros((T, 24), bool)
    contacts[:, [R.BI[b] for b in ("L_Hand", "R_Hand")]] = True           # the hands everywhere
    zc = R.zone_contacts(contacts)
    win = R.content_windows(zc, lay, ["L_HAND", "R_HAND"], ["L_HAND", "R_HAND", "L_FOOT", "R_FOOT"])
    assert (win["s_end"], win["d_start"]) == (lay["departure"], lay["arrival"])           # the card's windows
    contacts[lay["arrival"]:lay["arrival"] + 23, R.BI["L_Knee"]] = True                    # E2 s3 t6rpx12's knee
    contacts[lay["settle"][0] + 9, R.BI["L_Toe"]] = True                                   # a toe in the settle
    contacts[lay["transition"][0] + 5, R.BI["L_Hip"]] = True                               # transition: not a hold
    zc = R.zone_contacts(contacts)
    win = R.content_windows(zc, lay, ["L_HAND", "R_HAND"], ["L_HAND", "R_HAND", "L_FOOT", "R_FOOT"])
    assert win["d_start"] == lay["arrival"] + 23 and win["d_unlabelled"] == {
        "L_SHANK": list(range(lay["arrival"], lay["arrival"] + 23))}
    assert win["s_end"] == lay["settle"][0] + 8 and list(win["s_unlabelled"]) == ["L_FOOT"]
    tr = R.unlabelled(zc, np.arange(lay["transition"][0], lay["transition"][1] + 1), {"L_HAND", "R_HAND"})
    assert tr == {"L_THIGH": [lay["transition"][0] + 5]}


def test_near_repeats_only_inside_holds():
    pos = np.cumsum(np.full((10, 24, 3), 1e-3), axis=0)
    pos[4] = pos[3] + 1e-7                                                  # frames 3 -> 4 nearly repeat
    assert R.near_repeats(pos, [{"frame_start": 2, "frame_end": 5}])["pass"]
    out = R.near_repeats(pos, [{"frame_start": 4, "frame_end": 6}])
    assert not out["pass"] and out["outside_holds"] == [[3, 4]] and out["min_step_m"] < R.NEAR_REPEAT_M


# --------------------------------------------------------------------------- #
# One real clip, in memory
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def spliced():
    if not HAVE_INPUTS:
        pytest.skip("release v2 or the selection is not on disk")
    from reference_curation import fit_writer as fw

    torch.set_num_threads(1)
    inp = R.inputs()
    sk = fw.skeleton("v2")
    v = next(v for v in inp["variants"] if v["name"] == "SYN_E3_lower_high_s0_t6px")
    c = R.splice(sk, v, inp, {})
    R.attach_endpoints(c, inp)
    return c, R.clip_entry(c, v, inp, R.HEAVY_ROOT / "x.motion"), inp, sk


def test_lineage_partitions_the_clip(spliced):
    c, entry, _, _ = spliced
    lin, lay, clock = c["lineage"], c["info"]["layout"], c["info"]["clock"]
    T = lay["num_frames"]
    assert entry["num_frames"] == T == len(lin["kind"]) == c["motion"]["dof_pos"].shape[0]
    kind = lin["kind"]
    L, N, k_d, k_a = lay["s_exemplar"], clock["N"], clock["k_d"], clock["k_a"]
    assert (kind[:L + 1] == 0).all() and (kind[lay["lead_out"][0]:] == 0).all()
    assert (kind[L + 1:L + 1 + k_d] == 2).all() and (kind[L + 1 + k_d:L + 1 + k_a] == 1).all()
    assert (kind[L + 1 + k_a:L + 1 + N] == 2).all()
    assert list(lin["source_frame"][L + 1:L + 1 + N]) == list(range(N))
    real = kind == 0
    for s in (0, 1):                                      # consecutive source frames inside each real part
        sf = lin["source_frame"][real & (lin["source"] == s)]
        assert np.all(np.diff(sf) == 1)
    assert lin["source_frame"][L] == c["info"]["S"]["frame_hold"]
    assert lin["source_frame"][lay["lead_out"][0]] == c["info"]["D"]["frame_hold"]
    yaw = c["info"]["transform"]["yaw_rad"]
    assert np.all(lin["yaw"][:L + 1 + k_a] == 0) and np.all(lin["yaw"][lay["d_hold"][0]:] == yaw)
    assert np.all(lin["blend_w"][kind != 2] == 0) and np.all((lin["blend_source"] >= 0) == (kind == 2))
    assert lin["blend_w"][L + k_d] == 0 and np.all(lin["blend_w"][L + 1:L + k_d] > 0)      # settle: S's weight
    assert np.all((lin["blend_w"][L + 1 + k_a:L + 1 + N] > 0) & (lin["blend_w"][L + 1 + k_a:L + 1 + N] < 1))
    assert np.all(np.isnan(lin["variant_t"]) == (kind == 0))
    assert [str(x) for x in lin["seg_name"]] == ["lead_in", "s_exemplar", "settle", "transition", "d_hold", "lead_out"]


def test_spliced_clip_passes_every_clip_check(spliced):
    c, entry, inp, sk = spliced
    chk = R.check_clip(sk, c, entry, inp["release"])
    assert chk["failed"] == [], chk["failed"]
    assert chk["real_frames"]["dof_exact"] and chk["real_frames"]["pos_m"] < 1e-5
    assert chk["seams"]["repeated_frames"]["identical_pairs"] == []
    assert chk["seams"]["acc_rule"]["card"]["pass"] and chk["seams"]["acc_rule"]["parts"]["pass"]
    assert chk["transform"]["yaw_vs_run_deg"] <= 0.005
    # the contract is per part: the settle (S = handstand) has no brace to exempt, the hold at D (crow) has crow's
    assert chk["blend_frames"]["settle"]["braces_exempt"] == []
    assert chk["blend_frames"]["d_hold"]["braces_exempt"] == ["L_SHANK+L_UPPER_ARM", "R_SHANK+R_UPPER_ARM"]
    assert chk["hold_content"]["trimmed"] == []


def test_negative_control_an_unlabelled_contact_in_a_hold_fails(spliced):
    c, entry, inp, _ = spliced
    f = c["info"]["layout"]["d_hold"][0] + 5                    # a blend frame inside D's window
    mot = dict(c["motion"])
    mot["rigid_body_contacts"] = c["motion"]["rigid_body_contacts"].clone()
    mot["rigid_body_contacts"][f, R.BI["L_Knee"]] = True
    out = R.check_hold_content(entry, mot, c["lineage"], c["info"], inp["release"])
    assert not out["pass"] and any("L_SHANK" in p for p in out["problems"])


def test_negative_control_a_non_tie_horizon_fails(spliced):
    """An angular velocity equal to a horizon candidate that does not tie the smallest is a wrong choice."""
    c, _, _, _ = spliced
    lin = c["lineage"]
    sources = {str(lin["stems"][0]): c["sources"]["S"], str(lin["stems"][1]): c["sources"]["D"]}
    m = c["sources"]["S"]
    real = np.nonzero((lin["kind"] == 0) & (lin["source"] == 0))[0][50:-10]
    for f in real:                          # a frame and body whose largest candidate is far from the smallest
        sf = int(lin["source_frame"][f])
        for b in range(24):
            cand = R.horizon_candidates(m["rigid_body_rot"][:, b].double().numpy(), sf, 0.0)
            mags = [np.linalg.norm(x) for x in cand]
            if max(mags) - min(mags) > 0.05:
                break
        else:
            continue
        break
    mot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in c["motion"].items()}
    mot["rigid_body_ang_vel"][f, b] = torch.as_tensor(cand[int(np.argmax(mags))], dtype=torch.float32)
    out = R.check_real_frames(mot, lin, sources)
    assert not out["pass"] and out["ang_vel_unexplained"]


def test_negative_control_a_perturbed_real_frame_fails(spliced):
    c, _, _, _ = spliced
    lin = c["lineage"]
    mot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in c["motion"].items()}
    f = int(np.nonzero(lin["kind"] == 0)[0][100])
    mot["rigid_body_pos"][f, 5] += torch.tensor([0.0, 0.0, 1e-4])
    sources = {str(lin["stems"][0]): c["sources"]["S"], str(lin["stems"][1]): c["sources"]["D"]}
    assert R.check_real_frames(c["motion"], lin, sources)["pass"]
    assert not R.check_real_frames(mot, lin, sources)["pass"]
    mot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in c["motion"].items()}
    mot["dof_pos"][f, 7] += 1e-6
    assert not R.check_real_frames(mot, lin, sources)["dof_exact"]


# --------------------------------------------------------------------------- #
# The committed build
# --------------------------------------------------------------------------- #
BUILT = (R.RECORD_ROOT / f"{SYNTHETIC_ID}.json").exists()


@pytest.mark.skipif(not BUILT, reason="the build is not on disk")
def test_check_passes_on_the_committed_build():
    passed, failed = R.check(SYNTHETIC_ID, log=lambda *_: None)
    assert failed == [], failed
    assert len(passed) >= 29 * 4


@pytest.mark.skipif(not BUILT, reason="the build is not on disk")
def test_the_build_reproduces_in_memory(spliced):
    """Re-splicing a clip gives the written clip bit for bit, and its record has no wall-clock field."""
    import json

    c, _, _, _ = spliced
    rec = json.loads((R.RECORD_ROOT / f"{SYNTHETIC_ID}.json").read_text())
    disk = R.load(R.REPO / rec["clips"][c["name"]]["motion"]["path"])
    assert set(disk) == set(c["motion"])
    assert all(torch.equal(disk[k], v) if torch.is_tensor(v) else disk[k] == v for k, v in c["motion"].items())
    info = json.loads((R.REPO / rec["clips"][c["name"]]["json"]["path"]).read_text())
    assert "seconds" not in json.dumps(info)


@pytest.mark.skipif(not BUILT, reason="the build is not on disk")
def test_the_knee_down_landing_is_out_of_ds_window():
    """SYN_E2_jumpback_mid_s3_t6rpx12 lands knee down; D's window starts after the blend lifts the knee."""
    import json

    import yaml

    rec = json.loads((R.RECORD_ROOT / f"{SYNTHETIC_ID}.json").read_text())
    name = "SYN_E2_jumpback_mid_s3_t6rpx12"
    info = json.loads((R.REPO / rec["clips"][name]["json"]["path"]).read_text())
    win = info["windows"]
    assert win["d_start"] > win["d_start_card"] and set(win["d_unlabelled"]) == {"L_SHANK", "L_THIGH"}
    entry = yaml.safe_load(open(R.REPO / rec["clips"][name]["holds"]["path"]))
    d = next(h for h in entry["holds"] if h["inherits"] == info["D"]["hold_id"])
    assert d["frame_start"] == win["d_start"]
    assert rec["windows_trimmed"] == {name: ["D"]}
    assert "L_SHANK" in rec["transition_contacts"][name]
