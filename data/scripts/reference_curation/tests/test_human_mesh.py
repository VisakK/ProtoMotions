# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the human mesh layer and capture store v2 (BUILD_PLAN Step 5).

They run on the real fits and pin what the check script printed
(``expert_revist/reference_curation_review_2026_09_28/output_smplx_model_check.txt``) and what the
review measured (README §3.2-3.3). The module fixture builds the v1 and v2 records of four clips
into a temp dir with the committed calibrations; nothing under ``output/`` or
``data/reference_curation/`` is touched.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from extract_contact_configs import ADJACENT, ZONE_ORDER
from reference_curation import capture, human_mesh as hm, ids

CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
STANDING_SPLIT = "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a"
HEADSTAND_B = "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b"
BIG_TOE_C = "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c"  # its MoSh pkl is corrupt
DANCER_C = "220926_Lord_of_the_Dance_Pose_or_Natarajasana_-c"
CHECK_SCRIPT = ids.REPO / "expert_revist/reference_curation_review_2026_09_28/smplx_model_check.py"
SEGMENTS = ids.MOYO_DATA / "essentials/yogi_segments/smplx/smplx_segments_vertex_ids.pkl"  # JSON inside

pytestmark = pytest.mark.skipif(
    not (ids.MOYO_DATA / "mosh").exists() or not hm.MODEL_PATH.exists() or not ids.SHIPPED_DIR.exists(),
    reason="the MOYO MoSh fits, the SMPL-X model or the shipped ftC clips are not on disk")


def _zi(zone: str) -> int:
    return ZONE_ORDER.index(zone)


def _pk(name: str) -> int:
    return hm.PAIR_NAMES.index(name)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("capture_v2")
    cal_v1, cal = capture.load_calibration(), hm.load_calibration()
    bases, records = {}, {}
    for stem in (CROW, STANDING_SPLIT, HEADSTAND_B, BIG_TOE_C):
        bases[stem] = capture.build(stem, cal_v1, out / "v1")
        records[stem] = hm.build(stem, cal, out / "v2", base=bases[stem])
    return SimpleNamespace(out=out, cal=cal, cal_v1=cal_v1, bases=bases, records=records)


# --------------------------------------------------------------------------- #
# The body: port, registration, verification
# --------------------------------------------------------------------------- #
def _check_script():
    spec = importlib.util.spec_from_file_location("smplx_model_check", CHECK_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_port_poses_the_mesh_as_the_check_script_and_the_fast_path_agrees():
    chk = _check_script()
    fit, status, _ = hm.load_fit(CROW)
    assert status == "ok" and fit["fullpose"].shape == (972, 165) and fit["v_template"].shape == (10475, 3)
    frames = [0, 439, 570]
    ref = chk.SMPLX(chk.MODEL).vertices(fit["v_template"], fit["fullpose"][frames], fit["trans"][frames])
    ported = hm.model().vertices(fit["v_template"], fit["fullpose"][frames], fit["trans"][frames])
    assert np.array_equal(ported, ref)
    fast = np.concatenate([v for _, v in hm.model().posed(fit["v_template"], fit["fullpose"][frames],
                                                          fit["trans"][frames], chunk=2)])
    assert np.abs(fast - ref).max() < 1e-12


def test_skinning_rule_reproduces_the_check_script_output():
    """``output_smplx_model_check.txt``: offset, non-finger median, share under 2 mm, finger median."""
    printed = {STANDING_SPLIT: ([9.0], (0.0, 0.02, 10.16), 0.42, 0.96, 0.21),
               HEADSTAND_B: ([15.0], (-0.02, -0.01, 10.19), 0.59, 0.89, 0.22),
               CROW: ([7.3167, 9.5], (0.0, -0.02, 10.18), 0.70, 0.85, 0.20)}
    for stem, (times, offset, body, share, finger) in printed.items():
        fit, _, _ = hm.load_fit(stem)
        T = fit["fullpose"].shape[0]
        frames = sorted(set(np.linspace(0, T - 1, 24).astype(int).tolist() + [round(t * fit["fps"]) for t in times]))
        _, v = hm.register(hm.model(), fit, frames, attach="skinning")
        assert np.allclose(v["offset_mm"], offset, atol=0.005), stem
        assert round(v["nonfinger_median_mm"], 2) == body and round(v["nonfinger_share_under_2mm"], 2) == share
        assert round(v["finger_median_mm"], 2) == finger and v["ok"], stem


def test_surface_rule_passes_where_the_check_scripts_rule_fails():
    """The one fit of 59 where the skinning rule misses the 80 % criterion (79.2 %)."""
    fit, _, _ = hm.load_fit(DANCER_C)
    _, skin = hm.register(hm.model(), fit, attach="skinning")
    _, surf = hm.register(hm.model(), fit)
    assert skin["nonfinger_share_under_2mm"] == pytest.approx(0.7925, abs=1e-4) and not skin["markers_ok"]
    assert surf["nonfinger_share_under_2mm"] == pytest.approx(0.9057, abs=1e-4) and surf["ok"]
    assert surf["nonfinger_median_mm"] < skin["nonfinger_median_mm"] and surf["finger_median_mm"] < 0.15
    assert skin["offset_ok"]  # the registration itself is fine either way


def test_marker_error_below_the_threshold(built):
    """The card's verification on every built clip, and the committed corpus numbers (59 fits)."""
    for stem in (CROW, STANDING_SPLIT, HEADSTAND_B):
        meta = built.records[stem].meta
        v = meta["verification"]
        assert v["ok"] and v["attach"] == "surface" and v["frames"] == hm.VERIFY_FRAMES
        assert v["nonfinger_median_mm"] < 0.6 and v["finger_median_mm"] < 0.16
        assert v["nonfinger_share_under_2mm"] >= 0.83
        assert abs(v["offset_mm"][2] - 10.17) < 0.05 and max(map(abs, v["offset_mm"][:2])) < 0.05
        clip_xy0 = np.array([*capture.VICON_TO_CLIP_XY, 0.0])
        assert meta["registration"]["shift_to_clip_m"] == pytest.approx(np.array(v["offset_mm"]) / 1000 + clip_xy0,
                                                                        abs=2e-6)
    corpus = built.cal["verification"]
    assert corpus["clips"] == 59 and corpus["attach"] == "surface"
    assert 10.12 <= corpus["offset_z_mm"][0] <= corpus["offset_z_mm"][1] <= 10.22 and corpus["offset_xy_max_mm"] < 0.05
    assert corpus["nonfinger_median_mm_max"] < 0.7 and corpus["nonfinger_share_under_2mm_min"] >= 0.83
    assert corpus["finger_median_mm_max"] < 0.21


def test_registration_refuses_a_shifted_fit():
    fit, _, _ = hm.load_fit(CROW)
    shifted = dict(fit, sim=fit["sim"] + np.array([0.0, 0.0, 0.002]))  # 2 mm more than MoSh's frame
    _, v = hm.register(hm.model(), shifted)
    assert not v["offset_ok"] and v["markers_ok"] and not v["ok"]


# --------------------------------------------------------------------------- #
# Zones, pairs and the seam
# --------------------------------------------------------------------------- #
def test_zone_map_keeps_sides_and_explains_why_ipman_segments_cannot_be_the_zones():
    m = hm.model()
    assert len(hm.JOINT_ZONE) == 55 and sorted(set(m.zone.tolist())) == list(range(len(ZONE_ORDER)))
    segments = {k: set(v) for k, v in json.loads(SEGMENTS.read_text()).items()}
    for zone in ZONE_ORDER:
        verts = set(m.zone_vertices[_zi(zone)].tolist())
        touched = {s for s, vs in segments.items() if verts & vs}
        if zone.startswith("L_"):
            assert touched <= {"leftBicep", "leftForeArm", "leftLowerLeg", "leftThigh", "torso"}, zone
        elif zone.startswith("R_"):
            assert touched <= {"rightBicep", "rightForeArm", "rightLowerLeg", "rightThigh", "torso"}, zone
    # IPMAN's ten volume parts merge zones the labels keep apart.
    for zone, segment in (("L_HAND", "leftForeArm"), ("R_HAND", "rightForeArm"), ("L_FOOT", "leftLowerLeg"),
                          ("R_FOOT", "rightLowerLeg"), ("PELVIS", "torso")):
        assert set(m.zone_vertices[_zi(zone)].tolist()) <= segments[segment], zone


def test_pairs_cover_every_label_and_skip_exactly_the_shared_skin():
    m = hm.model()
    assert len(hm.PAIRS) == 91
    labelled = {p for c in ids.load_manifest()["clips"] for h in c["holds"] for p in h["pairs"] if ":" not in p}
    assert len(labelled) == 65 and labelled <= set(hm.PAIR_NAMES)
    assert {"HEAD+L_UPPER_ARM", "HEAD+R_UPPER_ARM", "L_THIGH+R_THIGH"} <= set(hm.PAIR_NAMES)
    # Zones that share mesh edges: the kinematic adjacency, plus the thighs at the crotch.
    shared = set()
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = m.zone[m.faces[:, i]], m.zone[m.faces[:, j]]
        shared |= {frozenset((ZONE_ORDER[x], ZONE_ORDER[y])) for x, y in zip(a[a != b], b[a != b])}
    assert shared == hm.KINEMATIC_ADJACENT | {frozenset(("L_THIGH", "R_THIGH"))}
    assert hm.KINEMATIC_ADJACENT == ADJACENT - {frozenset(("HEAD", "L_UPPER_ARM")), frozenset(("HEAD", "R_UPPER_ARM"))}


def test_seam_rule_trims_only_the_thighs():
    fit, _, _ = hm.load_fit(CROW)
    m = hm.model()
    geo = hm.zone_geodesics(m, fit["v_template"])
    trimmed = []
    for k, (a, b) in enumerate(hm.PAIRS):
        query, tree, _ = hm.pair_vertex_sets(m, fit["v_template"])[k]
        if len(query) + len(tree) < len(m.zone_vertices[_zi(a)]) + len(m.zone_vertices[_zi(b)]):
            trimmed.append(hm.PAIR_NAMES[k])
    assert trimmed == ["L_THIGH+R_THIGH"]
    along = lambda a, b: 100 * geo[_zi(b)][m.zone_vertices[_zi(a)]].min()  # noqa: E731 -- cm
    assert along("L_THIGH", "R_THIGH") < 1.0
    assert 14.0 < min(along("HEAD", "L_UPPER_ARM"), along("HEAD", "R_UPPER_ARM")) < 15.0
    assert 19.0 < along("L_THIGH", "TRUNK") < 20.0


def test_states_are_schmitt_triggers_on_the_calibrated_bands():
    ground = {"touch_m": 0.02, "separation_m": 0.03}
    z = np.array([0.05, 0.01, 0.025, 0.035, 0.025, -0.03])
    min_z = np.repeat(z[:, None], len(ZONE_ORDER), 1)
    gap = np.repeat(np.array([0.10, 0.015, 0.025, 0.04, 0.02, 0.0])[:, None], len(hm.PAIRS), 1)
    known = np.ones_like(min_z, dtype=bool)
    known[4, _zi("L_SHANK")] = False
    g, p = hm.states(min_z, gap, known, ground)
    assert g[:, _zi("L_FOOT")].tolist() == [0, 1, 1, 0, 0, 1]  # penetration is contact
    assert g[:, _zi("L_SHANK")].tolist() == [0, 1, 1, 0, -1, 1]
    assert p[:, _pk("L_SHANK+L_UPPER_ARM")].tolist() == [0, 1, 1, 0, -1, 1]  # unknown if either zone is
    assert p[:, _pk("R_FOOT+L_THIGH")].tolist() == [0, 1, 1, 0, 1, 1]


# --------------------------------------------------------------------------- #
# The card's human-side truths
# --------------------------------------------------------------------------- #
def test_standing_split_standing_sole_is_on_the_floor_while_the_avatar_floats(built):
    rec = built.records[STANDING_SPLIT]
    f = rec.frame(9.0)
    assert 0.0 < rec["human_min_z"][f, _zi("L_FOOT")] < 0.01                     # check script: 0.4 cm
    assert rec["human_ground_state"][f, _zi("L_FOOT")] == 1
    assert rec["avatar_min_z"][f, _zi("L_FOOT")] == pytest.approx(0.1145, abs=0.002)  # the float (Step 3)
    assert rec["human_min_z"][f, _zi("R_FOOT")] > 1.7                            # the raised leg: 174.6 cm
    for hand in ("L_HAND", "R_HAND"):                                           # 0.1 and -1.1 cm
        assert -0.02 < rec["human_min_z"][f, _zi(hand)] < 0.01 and rec["human_ground_state"][f, _zi(hand)] == 1


def test_headstand_crown_is_on_the_floor(built):
    rec = built.records[HEADSTAND_B]
    f = rec.frame(15.0)
    assert abs(rec["human_min_z"][f, _zi("HEAD")]) < 0.01                          # check script: -0.0 cm
    assert rec["human_ground_state"][f, _zi("HEAD")] == 1
    assert rec["avatar_min_z"][f, _zi("HEAD")] > 0.13                              # the head collider hovers
    assert min(rec["human_min_z"][f, _zi("L_FOOT")], rec["human_min_z"][f, _zi("R_FOOT")]) > 1.5


def test_crow_knee_triceps_contact_through_the_hands_only_hold(built):
    rec = built.records[CROW]
    hold = slice(495, 720)  # the capture's hands-only run (Step 2: 8.25 s to the label's end)
    for side in "LR":
        k = _pk(f"{side}_SHANK+{side}_UPPER_ARM")
        assert (rec["human_pair_state"][hold, k] == 1).mean() >= 0.95, side
    f = rec.frame(9.5)
    assert rec["human_pair_gap"][f, _pk("L_SHANK+L_UPPER_ARM")] == pytest.approx(0.0051, abs=0.001)  # 0.5 cm
    assert rec["human_pair_gap"][f, _pk("R_SHANK+R_UPPER_ARM")] == pytest.approx(0.0047, abs=0.001)
    assert rec["human_ground_state"][f, _zi("L_FOOT")] == 0 and rec["human_ground_state"][f, _zi("R_FOOT")] == 0
    early = rec.frame(7.3167)                                          # the label's exemplar: toe still down
    assert rec["human_ground_state"][early, _zi("L_FOOT")] == 1
    assert rec["human_min_z"][early, _zi("L_FOOT")] == pytest.approx(0.0063, abs=0.001)


def test_mesh_agrees_with_the_marker_touch_states(built):
    for stem in (CROW, STANDING_SPLIT, HEADSTAND_B):
        rec = built.records[stem]
        for zone in hm.CALIB_ZONES:
            mk, hs = rec["ground_state"][:, _zi(zone)], rec["human_ground_state"][:, _zi(zone)]
            both = (mk >= 0) & (hs >= 0)
            assert (mk[both] == hs[both]).mean() >= 0.97, (stem, zone)


def test_calibration_is_steps_1_rule_on_the_pooled_extremities(built):
    cal = built.cal
    assert cal["ground"]["touch_m"] == pytest.approx(cal["pooled"]["p99"] + capture.TOUCH_MARGIN_M)
    assert cal["ground"]["separation_m"] == pytest.approx(cal["ground"]["touch_m"] + hm.SEPARATION_GAP_M)
    assert 0.0225 < cal["ground"]["touch_m"] < 0.0230                   # measured 2.27 cm
    assert cal["pooled"]["n_frames"] == sum(cal["zones"][z]["n_frames"] for z in hm.CALIB_ZONES)
    assert cal["zones"]["PELVIS"]["p50"] < -0.04                        # soft tissue: the mesh sinks in
    assert cal["pairs"]["names"] == list(hm.PAIR_NAMES) and cal["pairs"]["touch_m"] == hm.PAIR_TOUCH_M
    assert built.records[CROW].meta["calibration"]["id"] == cal["id"]


# --------------------------------------------------------------------------- #
# Store v2
# --------------------------------------------------------------------------- #
def test_store_v2_extends_v1_without_touching_it(built):
    v1_dir, v2_dir = built.out / "v1", built.out / "v2"
    before = {p.name: (p.stat().st_mtime_ns, ids.sha256_file(p)) for p in v1_dir.iterdir()}
    rec = hm.load(CROW, v2_dir, built.cal, capture.load(CROW, v1_dir, built.cal_v1))
    base = built.bases[CROW]
    for k, v in base.arrays.items():
        assert np.array_equal(rec[k], v, equal_nan=True), k
    assert set(rec.arrays) - set(base.arrays) == set(hm.HUMAN_ARRAYS)
    assert rec["human_pair_gap"].shape == (972, 91) and rec["human_pair_state"].dtype == np.int8
    with np.load(v2_dir / f"{CROW}.npz") as npz:
        assert set(npz.files) == set(hm.HUMAN_ARRAYS)
    assert {p.name: (p.stat().st_mtime_ns, ids.sha256_file(p)) for p in v1_dir.iterdir()} == before
    assert rec.meta["base"] == {"store": "capture/v1", "generator": ids.sha256_file(capture.__file__),
                                "calibration": built.cal_v1["id"]}


def test_store_v2_reloads_and_rebuilds_on_a_new_calibration(built):
    path = built.out / "v2" / f"{CROW}.json"
    stamp = path.stat().st_mtime_ns
    rec = hm.load(CROW, built.out / "v2", built.cal, built.bases[CROW])
    assert path.stat().st_mtime_ns == stamp and rec.meta == built.records[CROW].meta
    other = dict(built.cal, id="another-calibration", ground={"touch_m": 0.05, "separation_m": 0.06})
    rebuilt = hm.load(CROW, built.out / "v2", other, built.bases[CROW])
    assert rebuilt.meta["calibration"]["id"] == "another-calibration" and path.stat().st_mtime_ns != stamp
    assert np.array_equal(rebuilt["human_min_z"], built.records[CROW]["human_min_z"])
    hm.build(CROW, built.cal, built.out / "v2", base=built.bases[CROW])  # leave the fixture as it was


def test_unreadable_fit_is_unavailable_not_a_crash(built):
    rec = built.records[BIG_TOE_C]
    assert not rec.meta["human_available"] and rec.meta["human_status"] == "unreadable"
    assert (rec["human_ground_state"] == -1).all() and (rec["human_pair_state"] == -1).all()
    assert np.isnan(rec["human_min_z"]).all() and rec["human_pair_gap"].shape[1] == 91


# --------------------------------------------------------------------------- #
# Render layer
# --------------------------------------------------------------------------- #
def test_render_layer_is_co_registered_with_the_markers_and_the_scene():
    from reference_curation import render

    human, _, _ = hm.load_human(CROW)
    frame = 439
    rebuilt = hm.rebuild_markers(human.model, human.fit, [frame])[0] + human.shift
    sim, labels = capture.markers(CROW, "sim")
    assert labels == human.fit["labels"]
    assert np.median(np.linalg.norm(rebuilt - sim[frame], axis=-1)) < 0.0015
    layer = hm.render_layer(CROW)
    v = layer.vertices(frame)
    assert np.abs(v - human.model.vertices(human.fit["v_template"], human.fit["fullpose"][[frame]],
                                           human.fit["trans"][[frame]])[0] - human.shift).max() < 1e-12
    clip = render.load_clip(CROW)
    sheets = render.plan(clip, frame, ("L_FOOT",))
    with render.Scene(clip, human_mesh=layer) as scene:
        scene.pose(frame)
        g = scene.model.geom("human_mesh").id
        start = scene.model.mesh_vertadr[0]
        mv = scene.model.mesh_vert[start:start + scene.model.mesh_vertnum[0]]
        world = mv @ scene.data.geom_xmat[g].reshape(3, 3).T + scene.data.geom_xpos[g]
        assert np.abs(world - v).max() < 1e-5          # MuJoCo does not move the mesh (5-decimal XML)
        assert scene.model.geom_rgba[g][3] < 0.5       # translucent grey
        eye = sheets[0].panels[0]
        seg = scene.render(eye, segmentation=True)
        assert ((seg[..., 0] == g) & (seg[..., 1] == int(render.mujoco.mjtObj.mjOBJ_GEOM))).sum() > 1000
    for panel in (p for s in sheets for p in s.panels if p.kind in ("eye", "high", "grazing")):
        uv, depth = panel.camera.project(v)
        cam = panel.camera
        assert (depth > 0).all() and (uv >= 0).all()
        assert (uv[:, 0] <= cam.width).all() and (uv[:, 1] <= cam.height).all()


def test_no_fit_no_layer():
    assert hm.render_layer(BIG_TOE_C) is None
    assert os.path.exists(hm.MODEL_PATH)
