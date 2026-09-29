# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the evidence renderer and packets (BUILD_PLAN Step 3).

They run on the real clips and pin what the review measured (README §3.1-3.2, BUILD_PLAN §5): the
Standing Split -a standing foot floats 11.6 cm at 9.0 s, the Supported Headstand -b head hovers
14 cm, and so on. The module fixture builds the capture store and the audit into a temp dir, as
``test_audit.py`` does, then renders the Pass-A packets of every pilot hold there. Nothing under
``output/``, ``data/reference_curation/`` or ``REVIEW_ROOT`` is touched.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, capture, ids, packets, render

mujoco = render.mujoco
CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a"
STANDING_SPLIT = "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a"
HEADSTAND_B = "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b"
PEACOCK = "220923_Peacock_Pose_or_Mayurasana_-a"
PLOW_B = "220926_Plow_Pose_or_Halasana_-b"
BRIDGE = "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a"
SIDE_CROW_C = "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c"
# The pilot holds (README §6): every family hold of the ten pilot clips.
PILOT_HOLDS = {
    f"{CROW}@439", f"{SIDE_CROW_C}@606", f"{SIDE_CROW_C}@1296", f"{STANDING_SPLIT}@540",
    "220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a@1051", f"{HEADSTAND_B}@1093",
    f"{PLOW_B}@980", f"{PLOW_B}@1384", f"{PEACOCK}@619",
    "220926_Upward_Plank_Pose_or_Purvottanasana_-a@854", "220926_Upward_Plank_Pose_or_Purvottanasana_-a@990",
    "220926_Upward_Plank_Pose_or_Purvottanasana_-a@1409", "220926_Warrior_II_Pose_or_Virabhadrasana_II_-a@572",
    f"{BRIDGE}@848", f"{BRIDGE}@1056",
}
FULL_BODY = ("eye", "high", "grazing", "strip")

pytestmark = pytest.mark.skipif(
    not (ids.MOYO_DATA / "mosh").exists() or not ids.SHIPPED_DIR.exists(),
    reason="the MOYO MoSh fits or the shipped ftC clips are not on disk")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("packets")
    cal, _, failures = capture.build_all(ids.manifest_stems(), out / "store", out / "capture_v1.json")
    assert failures == []
    records, context, failures = audit.audit(store_dir=out / "store", calibration=cal)
    assert failures == []
    aud = packets.load_audit(audit.write(records, context, out / "audits"))
    kw = dict(manifest=aud.manifest, review_root=out / "review", index_root=out / "index",
              store_dir=out / "store", calibration=cal)
    items = packets.select_items(aud, "A", pilot=True)
    entries, failures, built = packets.build_packets(items, aud.records, **kw)
    assert failures == []
    return SimpleNamespace(out=out, cal=cal, aud=aud, kw=kw, items={i["hold_id"]: i for i in items},
                           pilot={e["hold_id"]: e for e in entries}, built=built)


def _packet(run, entry) -> tuple[dict, str]:
    text = (packets.packet_dir(entry, run.kw["review_root"]) / "packet.json").read_text()
    return json.loads(text), text


def _mujoco_corners(scene) -> np.ndarray:
    """Every avatar geom's bounding-box corners from MuJoCo's own posed model (not ``render``'s)."""
    m, d = scene.model, scene.data
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    pts = [d.geom_xpos[g] + (m.geom_aabb[g, :3] + signs * m.geom_aabb[g, 3:]) @ d.geom_xmat[g].reshape(3, 3).T
           for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    return np.concatenate(pts)


# --------------------------------------------------------------------------- #
# Scene and cameras
# --------------------------------------------------------------------------- #
def test_camera_projection_matches_mujoco():
    model = mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="1024" offheight="1024"/>'
                                           '</visual><worldbody/></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cases = [(510, 510, 45, 30, -20, (0.4, -0.3, 0.9)), (800, 400, 45, 200, -60, (-0.5, 0.2, 0.1)),
             (338, 338, 30, 75, 12, (0.1, 0.35, 0.02)), (600, 300, 45, 120, 5, (0.6, 0.1, 1.2))]
    for w, h, fovy, az, el, point in cases:
        cam = render.Camera((0.1, 0.0, 0.5), 2.5, az, el, fovy, w, h)
        model.vis.global_.fovy = fovy
        r = mujoco.Renderer(model, h, w)
        try:
            r.update_scene(data, camera=cam.mjv())
            render._glyph(r.scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.01, 0, 0), point, np.eye(3), (1, 0, 0, 1), 1.0)
            img = r.render().astype(int)
        finally:
            r.close()
        ys, xs = np.nonzero((img[..., 0] > 150) & (img[..., 1] < 80))
        uv, depth = cam.project(point)
        assert depth[0] > 0 and len(xs) > 3
        assert abs(xs.mean() + 0.5 - uv[0, 0]) < 0.5 and abs(ys.mean() + 0.5 - uv[0, 1]) < 0.5


def test_glyphs_cast_no_shadow():
    """The spike's reviewer read marker shadows as markers (BUILD_PLAN §5). A decor glyph casts
    none; the control shows the same sphere as a non-decor scene geom does."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="256" offheight="256"/><quality shadowsize="4096"/></visual>'
        '<worldbody><light directional="true" pos="1 0 3" dir="-0.3 0 -1" castshadow="true"/>'
        '<geom type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = render.Camera((0.0, 0.0, 0.0), 3.0, 90.0, -89.9, 45.0, 256, 256)
    r = mujoco.Renderer(model, 256, 256)

    def shot(category):
        r.update_scene(data, camera=cam.mjv())
        if category is not None:
            render._glyph(r.scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.1, 0, 0), (0, 0, 0.3), np.eye(3), (0, 1, 0, 1))
            r.scene.geoms[r.scene.ngeom - 1].category = category
        return r.render().astype(int)

    try:
        base, decor = shot(None), shot(render._DECOR)
        static = shot(int(mujoco.mjtCatBit.mjCAT_STATIC))
    finally:
        r.close()
    body = ndimage.binary_dilation(decor[..., 1] > decor[..., 0] + 30, iterations=3)
    darker = lambda img: ((base.sum(-1) - img.sum(-1)) > 30) & ~body  # noqa: E731
    assert darker(decor).sum() == 0
    assert darker(static).sum() > 50  # 108 px measured; only mjCAT_DECOR (4) is exempt, 0/1/2 all cast


def test_renders_on_the_cpu_without_shadows():
    """EGL drifted by one level in two pixels after a few hundred renders in one process; OSMesa
    is bit-exact (``test_images_are_deterministic``). Nothing casts a shadow."""
    info = render.gl_info()
    assert info["backend"] == "osmesa" and "llvmpipe" in info["renderer"]
    model = mujoco.MjModel.from_xml_string(render.scene_xml())
    assert model.nlight == 1 and not model.light_castshadow.any()


def test_palette_tells_every_segment_family_apart():
    rgb = {z: np.array(render.zone_rgb(z)) for z in ZONE_ORDER}
    for a in ZONE_ORDER:
        for b in ZONE_ORDER:
            if render.ZONE_FAMILY[a] != render.ZONE_FAMILY[b]:
                assert np.linalg.norm(rgb[a] - rgb[b]) > 0.2, (a, b)
            elif a[:2] != b[:2] and "_" in a[:2] and "_" in b[:2]:  # the two shades of one family
                assert np.linalg.norm(rgb[a] - rgb[b]) > 0.25, (a, b)
    for hand in ("L_HAND", "R_HAND"):  # the spike's Crow was read upside down: hands must not look like feet
        for foot in ("L_FOOT", "R_FOOT"):
            assert np.linalg.norm(rgb[hand] - rgb[foot]) > 0.45
    legend = render.palette_legend()
    assert len(legend) == 10 and len(set(legend.values())) == 10
    assert legend["left hand"] == "dark red" and legend["right foot"] == "light blue"


def test_scene_is_the_clip_frame_and_drop_lines_are_the_store_heights(run):
    """The trap: a render's centring offset must never reach a measurement. The world is the clip
    frame, and the drop line of a zone is exactly the store's ``avatar_min_z``."""
    clip = render.load_clip(STANDING_SPLIT)
    rec = capture.load(STANDING_SPLIT, run.out / "store", run.cal)
    for f in (0, 540, 700, clip.num_frames - 1):
        heights, _ = render.zone_lowest(clip.pos[f], clip.rot[f])
        assert np.array_equal(heights.astype(np.float32), rec["avatar_min_z"][f])
    heights, low = render.zone_lowest(clip.pos[540], clip.rot[540])
    lines = {(round(p[0], 6), round(p[1], 6)): 2 * size[1] for t, size, p, *_ in render.glyphs(clip, 540)
             if t == mujoco.mjtGeom.mjGEOM_CYLINDER}
    foot = ZONE_ORDER.index("L_FOOT")
    length = lines[round(low[foot, 0], 6), round(low[foot, 1], 6)]
    assert length == pytest.approx(heights[foot]) == pytest.approx(0.116, abs=0.005)  # README §3.2: 11.6 cm
    raised = ZONE_ORDER.index("R_FOOT")  # 1.7 m up: it gets no pole through the insets
    assert heights[raised] > 1.5 and (round(low[raised, 0], 6), round(low[raised, 1], 6)) not in lines
    hand = ZONE_ORDER.index("L_HAND")  # render_v2: at the reference's 5 mm clearance it touches, so no line
    assert heights[hand] <= render.DROP_MIN_M and (round(low[hand, 0], 6), round(low[hand, 1], 6)) not in lines
    drawn_lines = {z: h for z, h, _ in render.drop_lines(clip.pos[540], clip.rot[540])}
    assert sorted(drawn_lines) == ["HEAD", "L_FOOT", "L_FOREARM", "R_FOREARM", "R_HAND"] and len(lines) == 5
    for z, h in drawn_lines.items():  # every line, the forearms' (4.8 and 8.0 cm) included, is the store's height
        zi = ZONE_ORDER.index(z)
        assert lines[round(low[zi, 0], 6), round(low[zi, 1], 6)] == pytest.approx(h) == pytest.approx(heights[zi])
    obs, _ = capture.markers(STANDING_SPLIT)
    drawn, _ = render.drawn_markers(clip, 540)
    spheres = np.array([p for t, _, p, *_ in render.glyphs(clip, 540) if t == mujoco.mjtGeom.mjGEOM_SPHERE])
    assert np.array_equal(spheres, drawn) and np.array_equal(drawn, obs[540][clip.trusted[540]])
    assert clip.trusted[540].sum() < 73  # the raised right foot's markers fail the fit (Step 1 hygiene)
    with render.Scene(clip) as scene:
        scene.pose(540)
        assert np.allclose(scene.data.xpos[1:], clip.pos[540], atol=1e-9)


def test_drop_lines_mark_every_gap_the_calibration_needs():
    """render_v2, from Step 4's render_v1 calibration: no line within 1 cm of the floor (the reference's
    5 mm clearance read as a gap), and a line under any other part up to 15 cm unless another part is
    below it (forearms 2-5 cm up, which had none, were read as resting on the floor)."""
    def lines(stem, frame):
        clip = render.load_clip(stem)
        return {z: round(100 * h, 2) for z, h, _ in render.drop_lines(clip.pos[frame], clip.rot[frame])}

    dolphin = lines("220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a", 565)
    assert dolphin["L_FOREARM"] == 1.3 and dolphin["R_FOREARM"] == 2.09 and dolphin["R_HAND"] == 2.54
    assert "L_HAND" not in dolphin  # 0.5 cm: touching
    # render_v3: a line and its cross wear their part's colour, so a forearm's line beside a planted hand
    # cannot pass for the hand's (render_v2's black lines did, in the calibration)
    clip = render.load_clip("220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a")
    drawn = [(t, rgba[:3]) for t, _, _, _, rgba, _ in render.glyphs(clip, 565) if t != mujoco.mjtGeom.mjGEOM_SPHERE]
    cylinders = sorted(tuple(np.round(c, 3)) for t, c in drawn if t == mujoco.mjtGeom.mjGEOM_CYLINDER)
    by_line = {z: h for z, h, _ in render.drop_lines(clip.pos[565], clip.rot[565])}
    assert cylinders == sorted(tuple(np.round(render.zone_rgb(z), 3)) for z in by_line)
    assert len(drawn) == 3 * len(by_line)  # a line and a two-bar cross each, and no bar without a camera
    plow = lines(PLOW_B, 980)
    assert {"TRUNK", "L_FOREARM", "R_FOREARM", "R_UPPER_ARM"} <= set(plow) and plow["HEAD"] == 1.16
    # Warrior II: both shins are within 15 cm but sit over their own feet, so only the 1.15 cm foot
    assert lines("220926_Warrior_II_Pose_or_Virabhadrasana_II_-a", 572) == {"L_FOOT": 1.15}


def test_human_mesh_slot_renders_translucent_and_follows_the_frame():
    clip = render.load_clip(CROW)
    tet = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    calls = []

    panel = render.plan(clip, 439)[0].panels[0]
    toward_camera = -render._basis(panel.camera.azimuth, panel.camera.elevation)[0]

    def vertices(frame):
        calls.append(frame)
        return tet - tet.mean(0) + np.array(panel.camera.lookat) + 0.5 * toward_camera

    mesh = render.HumanMesh(faces, vertices)
    with render.Scene(clip, human_mesh=mesh) as scene:
        seg = scene.render(panel, segmentation=True)
        mesh_id = scene.model.geom("human_mesh").id
        rgba = scene.model.geom_rgba[mesh_id]
    with render.Scene(clip) as scene:
        plain = scene.render(panel)
    assert calls == [439] and rgba[3] < 0.5
    assert ((seg[..., 0] == mesh_id) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))).sum() > 50
    with render.Scene(clip, human_mesh=mesh) as scene:
        assert not np.array_equal(scene.render(panel), plain)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def test_every_body_is_inside_every_full_body_view(run):
    """Checked with MuJoCo's own posed geometry and with segmentation: no avatar pixel on a border."""
    for hold_id in sorted(PILOT_HOLDS):
        stem, frame = ids.parse_hold_id(hold_id)
        item = run.items[hold_id]
        clip = render.load_clip(stem)
        sheets = render.plan(clip, frame, tuple(item["zones"]), packets.strip_frames(item, clip.num_frames))
        with render.Scene(clip) as scene:
            avatar = set(range(scene.model.ngeom)) - {scene.model.geom("floor").id}
            for panel in (p for s in sheets for p in s.panels if p.kind in FULL_BODY):
                scene.pose(panel.frame)
                uv, depth = panel.camera.project(_mujoco_corners(scene))
                cam = panel.camera
                assert (depth > 0).all() and (uv >= 0).all(), (hold_id, panel.kind)
                assert (uv[:, 0] <= cam.width).all() and (uv[:, 1] <= cam.height).all(), (hold_id, panel.kind)
                seg = scene.render(panel, segmentation=True)
                border = np.concatenate([seg[0], seg[-1], seg[:, 0], seg[:, -1]])
                on_edge = (border[:, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(border[:, 0], list(avatar))
                assert not on_edge.any(), (hold_id, panel.kind)
                if panel.kind == "grazing":
                    assert 0 < cam.position[2] <= 0.03  # the card: camera height <= 3 cm


def test_insets_are_floor_level_and_the_card_width(run):
    clip = render.load_clip(STANDING_SPLIT)
    insets = [p for s in render.plan(clip, 540, ("L_FOOT",)) if s.kind == "insets" for p in s.panels]
    assert any("L_FOOT" in p.zones for p in insets)
    for p in insets:
        cam = p.camera
        assert cam.position[2] == pytest.approx(render.FLOOR_CAMERA_M)
        right = render._basis(cam.azimuth, cam.elevation)[2]
        uv, _ = cam.project(np.array([p.target, np.add(p.target, 0.3 * right), np.subtract(p.target, 0.3 * right)]))
        assert uv[0] == pytest.approx([cam.width / 2, (1 - render.INSET_FLOOR) * cam.height], abs=1e-6)
        assert uv[1, 0] == pytest.approx(cam.width, abs=1e-6) and uv[2, 0] == pytest.approx(0, abs=1e-6)


def test_inset_targets_answer_the_pilot_questions(run):
    """README §6: each pilot's question needs its support in a close-up."""
    zones = lambda hold: {z for s in run.pilot[hold]["sheets"] if s["kind"] == "insets"  # noqa: E731
                          for p in s["panels"] for z in p["zones"]}
    assert "L_FOOT" in zones(f"{STANDING_SPLIT}@540")     # the 11.6 cm standing-foot float
    assert "HEAD" in zones(f"{HEADSTAND_B}@1093")          # the head collider hovering
    assert {"L_FOOT", "R_FOOT"} <= zones(f"{PEACOCK}@619")  # the toes-assisted hold
    assert {"L_FOOT", "R_FOOT"} <= zones(f"{PLOW_B}@980")   # planted feet missing from the label
    assert {"L_HAND", "R_HAND"} <= zones(f"{BRIDGE}@1056")  # labelled hands that are not down
    assert {"L_HAND", "R_HAND", "L_FOOT"} <= zones(f"{CROW}@439")  # the label starts while a foot is down
    clip = render.load_clip(HEADSTAND_B)
    heights, _ = render.zone_lowest(clip.pos[1093], clip.rot[1093])
    assert heights[ZONE_ORDER.index("HEAD")] == pytest.approx(0.14, abs=0.01)  # BUILD_PLAN §5 p3: 14 cm


def test_strip_frames():
    item = lambda keys, hold: {"window": {"key_frames": keys, "frame_hold": hold, "fps": 60}}  # noqa: E731
    # Crow -a@439: 425/439 and 719/722 sit within 0.25 s; the exemplar and the earlier survive
    assert packets.strip_frames(item([425, 439, 495, 719, 722], 439), 972) == [439, 495, 551, 607, 719]
    assert packets.strip_frames(item([0, 21, 54, 175], 21), 1000) == [0, 21, 54, 114, 175]  # every key kept
    frames = packets.strip_frames(item(list(range(100, 1000, 100)), 450), 2000)
    assert len(frames) == 5 and {100, 450, 900} <= set(frames)
    assert packets.strip_frames(item([5], 5), 1000) == [5, 12, 20, 27, 35]  # widened by 0.5 s each side


# --------------------------------------------------------------------------- #
# Packets
# --------------------------------------------------------------------------- #
def test_pilot_packets_exist_and_fit_the_budget(run):
    assert set(run.pilot) == PILOT_HOLDS and run.built == len(PILOT_HOLDS)
    index = packets.read_index(packets.index_path(render.RENDER_V, run.kw["index_root"]))
    assert {e["hold_id"] for e in index.values()} == PILOT_HOLDS
    for hold_id, entry in run.pilot.items():
        d = packets.packet_dir(entry, run.kw["review_root"])
        files = sorted(p.name for p in d.iterdir())
        assert files == [f"img_{k}.png" for k in range(1, len(files))] + ["packet.json"]
        assert 4 <= len(files) - 1 <= packets.MAX_IMAGES
        for name in files[:-1]:
            with Image.open(d / name) as im:
                assert max(im.size) <= 1024
            assert ids.sha256_file(d / name) == entry["images"][name]
        assert sum(p.stat().st_size for p in d.iterdir()) == entry["bytes"] <= packets.MAX_PACKET_BYTES
        pkt, _ = _packet(run, entry)
        assert pkt["packet_id"] == entry["packet_id"] == d.name
        content = {k: v for k, v in pkt.items() if k != "packet_id"}
        assert packets.hashlib.sha1(packets._canonical(content)).hexdigest() == d.name


def test_pass_a_packets_are_blind(run):
    for hold_id, entry in run.pilot.items():
        pkt, text = _packet(run, entry)
        record = run.aud.records[hold_id]
        assert set(pkt) == {"packet_id", "schema_version", "render_v", "pass", "images", "legend"}
        for name in (record["stem"], record["name"], hold_id, ids.recording_id(record["stem"])):
            assert name not in text
        assert record["name"].split("_h")[0].replace("_", " ").lower() not in text.lower()
        for label in record["label_ground"]:
            assert label not in text and f"{label}:G" not in text
        assert packets.leaks(text, record, run.items[hold_id], run.aud.manifest) == []
        for name in entry["images"]:  # no PNG text chunks either
            data = (packets.packet_dir(entry, run.kw["review_root"]) / name).read_bytes()
            assert not any(chunk in data for chunk in (b"tEXt", b"iTXt", b"zTXt"))


def test_fixed_texts_use_no_word_of_any_clip_name():
    """The legend and view texts are shared by every packet, so they avoid every name token in the
    corpus (several are plain English: scale, four, side, angle, hold, facing, standing, ...)."""
    tokens = set()
    for clip in ids.load_manifest()["clips"]:
        for name in [clip["stem"], clip.get("family") or "", ids.recording_id(clip["stem"]) or ""] + \
                [h["name"] for h in clip["holds"]] + [h.get("orientation") or "" for h in clip["holds"]]:
            tokens |= {t.lower() for t in re.split(r"[_\-()\s]+", name) if len(t) >= 4 and re.search("[A-Za-z]", t)}
    tokens -= packets.GENERIC_TOKENS
    panel = lambda zones=(), frame=5: SimpleNamespace(zones=zones, frame=frame)  # noqa: E731
    sheets = [SimpleNamespace(kind="eye", panels=[panel()] * 4),
              SimpleNamespace(kind="high_grazing", panels=[panel()] * 2),
              SimpleNamespace(kind="insets", panels=[panel((z,)) for z in ZONE_ORDER]),
              SimpleNamespace(kind="strip", panels=[panel(frame=f) for f in (1, 5, 9)])]
    text = json.dumps([packets.LEGEND, packets.NO_MARKERS, render.palette_legend(),
                       packets.describe(sheets, [1, 2, 3, 4], 5)]).lower()
    assert sorted(t for t in tokens if t in text) == []
    assert {"prone", "upright", "inverted", "supine"} <= tokens


def test_leak_guard_catches_names_labels_and_frames(run):
    hold_id = f"{CROW}@439"
    record, item = run.aud.records[hold_id], run.items[hold_id]
    assert packets.leaks(json.dumps(packets.LEGEND), record, item, run.aud.manifest) == []
    for bad in (CROW, "a bakasana hold", "L_HAND:G", "L_SHANK+L_UPPER_ARM", "R_THIGH", "prone",
                "frame 439 of the clip", "at 7.317 s"):
        assert packets.leaks(bad, record, item, run.aud.manifest), bad


def test_images_are_deterministic(run, tmp_path):
    hold_id = f"{CROW}@439"
    kw = dict(run.kw)
    entries = []
    for k in range(2):
        kw.update(review_root=tmp_path / f"review{k}", index_root=tmp_path / f"index{k}")
        entry, built = packets.build_packet(run.items[hold_id], run.aud.records[hold_id], force=True, **kw)
        entries.append(entry)
    assert entries[0]["packet_id"] == entries[1]["packet_id"] == run.pilot[hold_id]["packet_id"]
    assert entries[0]["images"] == entries[1]["images"] == run.pilot[hold_id]["images"]


def test_rebuild_skips_current_packets_and_repairs_missing_ones(run):
    """A rebuild late in a busy process must land on the same packet id: this is the check the EGL
    backend failed (two pixels one level off after a few hundred renders), and Step 4's ledger keys
    its verdicts on these ids."""
    items = [run.items[f"{CROW}@439"], run.items[f"{STANDING_SPLIT}@540"]]
    entries, failures, built = packets.build_packets(items, run.aud.records, **run.kw)
    assert failures == [] and built == 0
    packets.shutil.rmtree(packets.packet_dir(entries[0], run.kw["review_root"]))
    entries2, failures, built = packets.build_packets(items, run.aud.records, **run.kw)
    assert failures == [] and built == 1 and entries2[0]["packet_id"] == entries[0]["packet_id"]


def test_pass_b_carries_labels_and_numbers_over_the_same_images(run):
    hold_id = f"{STANDING_SPLIT}@540"
    item = packets.select_items(run.aud, "B", holds=[hold_id])[0]
    assert item["pass"] == "B" and item["purpose"] == "flagged"
    entry, built = packets.build_packet(item, run.aud.records[hold_id], **run.kw)
    pkt, _ = _packet(run, entry)
    assert built and entry["images"] == run.pilot[hold_id]["images"]
    assert pkt["hold"]["label_ground"] == ["L_FOOT", "L_HAND", "R_HAND"] and "L_FOOT:G" in pkt["hold"]["label_pairs"]
    assert pkt["item"]["questions"] and pkt["legend"] == _packet(run, run.pilot[hold_id])[0]["legend"]
    exemplar = pkt["evidence"]["exemplar"]["zones"]["L_FOOT"]
    assert exemplar["exemplar"] == 1 and exemplar["hover_cm"] == pytest.approx(11.6, abs=0.05)  # README §3.1
    now = next(f for f in pkt["evidence"]["frames"] if f["frame"] == 540)
    assert now["zones"]["L_FOOT"]["human"] == 1 and now["zones"]["L_FOOT"]["avatar_cm"] == pytest.approx(11.5, abs=0.05)
    assert now["panels"][:4] == ["1a", "1b", "1c", "1d"] and now["t_s"] == 9.0
