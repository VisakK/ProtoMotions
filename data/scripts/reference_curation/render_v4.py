# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence renders v4 (BodyFix Step 4, item 3): render_v3 on plant v2, with one colour per limb segment.

Two things change, and nothing else (cameras, framing, views, insets, the drop-line rules, the markers, the floor,
the bar, the human-mesh layer, OSMesa and its bit-exactness are ``render.py``'s, called unchanged):

* **The avatar is plant v2** (her skeleton, the shipped collider primitives on her segments, the head sphere on
  the cranium) posed by the Step 3 references. Every geometric function of ``render.py`` reads the plant through
  ``capture.skeleton()``, so they run inside ``retarget_v2.on_plant()`` (``plant_context``), which points it at
  plant v2 and is checked. The scene's own geometry (``scene_xml``) is read from plant v2's MJCF here:
  ``render._body_geoms`` caches the MJCF it first saw and ``mosh_replay.use_plant`` does not reset it.
* **Distinct hues per limb segment** (render_v3's still-open item, BUILD_PLAN Step 3's card): render_v3 gave a
  limb's two segments one colour (upper arm and forearm orange, thigh and shin green), so a forearm's drop line
  beside a planted upper arm read as the upper arm's (Plow, three times). Each chain is now three hues, each
  segment's neighbours different: arm = purple (upper arm), orange (forearm), red (hand); leg = yellow (thigh),
  green (shin), blue (foot); the head is magenta, the torso and pelvis grey. Left is still the dark shade and right
  the light one, the rule the left/right calibration admitted.

Packets and calibration use it through ``packets_v4``.
"""

from __future__ import annotations

import contextlib
import functools
import math
import xml.etree.ElementTree as ET

from reference_curation import render as R  # first: fixes MuJoCo's GL backend (OSMesa) before anything imports it
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from extract_contact_configs import ZONE_ORDER  # noqa: E402
from reference_curation import capture, fit_writer as fw, ids  # noqa: E402

MODULE = "reference_curation.render_v4"
RENDER_V = "render_v4"
PLANT = "v2"

# Segment families: the colour word for the legend, then the left (dark) and right (light) shade.
FAMILIES = {
    "foot": ("blue", (0.122, 0.471, 0.706), (0.651, 0.808, 0.890)),
    "shin": ("green", (0.200, 0.627, 0.173), (0.698, 0.875, 0.541)),
    "thigh": ("yellow", (0.776, 0.639, 0.000), (1.000, 0.929, 0.420)),
    "hand": ("red", (0.890, 0.102, 0.110), (0.984, 0.604, 0.600)),
    "forearm": ("orange", (1.000, 0.498, 0.000), (0.992, 0.749, 0.435)),
    "upper_arm": ("purple", (0.416, 0.239, 0.604), (0.792, 0.698, 0.839)),
    "head": ("magenta", (0.808, 0.110, 0.565), None),
    "torso": ("grey", (0.600, 0.600, 0.620), None),
}
ZONE_FAMILY = {"L_FOOT": "foot", "R_FOOT": "foot", "L_SHANK": "shin", "R_SHANK": "shin", "L_THIGH": "thigh",
               "R_THIGH": "thigh", "PELVIS": "torso", "TRUNK": "torso", "HEAD": "head", "L_UPPER_ARM": "upper_arm",
               "R_UPPER_ARM": "upper_arm", "L_FOREARM": "forearm", "R_FOREARM": "forearm", "L_HAND": "hand",
               "R_HAND": "hand"}
PART_WORDS = {"foot": "foot", "shin": "shin", "thigh": "thigh", "hand": "hand", "forearm": "forearm",
              "upper_arm": "upper arm", "head": "head and neck", "torso": "torso and pelvis"}


def zone_rgb(zone: str) -> tuple[float, float, float]:
    _, left, right = FAMILIES[ZONE_FAMILY[zone]]
    return right if zone.startswith("R_") and right is not None else left


def colour_words(zone: str) -> str:
    word, _, right = FAMILIES[ZONE_FAMILY[zone]]
    if right is None:
        return word
    return f"{'light' if zone.startswith('R_') else 'dark'} {word}"


def palette_legend() -> dict:
    """Plain-language colour key, one entry per drawn segment family and side."""
    out = {}
    for fam in ("head", "torso", "upper_arm", "forearm", "hand", "thigh", "shin", "foot"):
        zone = next(z for z, f in ZONE_FAMILY.items() if f == fam)
        if FAMILIES[fam][2] is None:
            out[PART_WORDS[fam]] = colour_words(zone)
        else:
            base = zone[2:]
            out[f"left {PART_WORDS[fam]}"] = colour_words(f"L_{base}")
            out[f"right {PART_WORDS[fam]}"] = colour_words(f"R_{base}")
    return out


# --------------------------------------------------------------------------- #
# The plant
# --------------------------------------------------------------------------- #
def mjcf_path():
    return fw.plant_paths(PLANT)[0]


@contextlib.contextmanager
def plant_context():
    """``retarget_v2.on_plant()``, checked: ``capture.skeleton()`` (every ``render`` geometry function) is plant v2."""
    from reference_curation import retarget_v2

    with retarget_v2.on_plant(PLANT):
        sk = capture.skeleton()
        ref = fw.skeleton(PLANT)
        if [g for b in sk.names for g in [sk.geoms[b][0]["type"]]] != [ref.geoms[b][0]["type"] for b in ref.names] or \
                str(ids.MJCF) != str(mjcf_path().resolve()):
            raise RuntimeError("capture.skeleton() is not plant v2 inside the plant context")
        yield


def _require_plant() -> None:
    if str(ids.MJCF) != str(mjcf_path().resolve()):
        raise RuntimeError("render_v4 draws plant v2: call it inside render_v4.plant_context()")


@functools.lru_cache(maxsize=1)
def body_geoms() -> dict:
    """Plant v2's MJCF: body name -> its ``<geom>`` attributes (body frame)."""
    out = {}

    def walk(body):
        out[body.attrib["name"]] = [dict(g.attrib) for g in body.findall("geom")]
        for child in body.findall("body"):
            walk(child)

    for body in ET.parse(str(mjcf_path())).getroot().find("worldbody").findall("body"):
        walk(body)
    return out


def scene_xml(mesh_vertices: np.ndarray | None = None, mesh: R.HumanMesh | None = None) -> str:
    """``render.scene_xml`` with plant v2's geoms and this module's palette."""
    rgb = lambda c: " ".join(f"{v:.3f}" for v in c)  # noqa: E731
    sk = capture.skeleton()
    zone_of = {sk.names[i]: z for z, idx in zip(ZONE_ORDER, sk.zone_bodies) for i in idx}
    geoms = body_geoms()
    materials = "".join(f'<material name="{z}" rgba="{rgb(zone_rgb(z))} 1" specular="0.15" shininess="0.2"/>'
                        for z in ZONE_ORDER)
    parts = [
        '<mujoco model="evidence">',
        f'<visual><global offwidth="{R.SHEET_MAX_PX}" offheight="{R.SHEET_MAX_PX}" fovy="{R.FOVY_DEG}"/>'
        '<quality shadowsize="512" offsamples="8"/><map znear="0.01" zfar="50"/>'
        '<headlight ambient=".42 .42 .42" diffuse=".45 .45 .45" specular="0 0 0"/></visual>',
        '<statistic extent="2" center="0 0 0.8"/>',
        '<asset><texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .89 .93" '
        'width="64" height="384"/>'
        '<texture name="checks" type="2d" builtin="checker" rgb1=".80 .80 .78" rgb2=".66 .66 .64" '
        'width="256" height="256"/>'
        f'<material name="floor" texture="checks" texrepeat="{1 / (2 * R.CHECK_M):g} {1 / (2 * R.CHECK_M):g}" '
        'texuniform="true" specular="0" shininess="0"/>'
        f'{materials}',
    ]
    if mesh is not None:
        v = " ".join(f"{x:.5f}" for x in np.asarray(mesh_vertices, dtype=float).ravel())
        f = " ".join(str(int(i)) for i in np.asarray(mesh.faces).ravel())
        parts.append(f'<mesh name="human" vertex="{v}" face="{f}" inertia="shell"/>')
    parts += ['</asset><worldbody>',
              '<light name="overhead" directional="true" pos="0 0 6" dir="0 0 -1" diffuse=".38 .38 .38" '
              'specular="0 0 0" castshadow="false"/>',
              f'<geom name="floor" type="plane" size="{R.FLOOR_HALF_M:g} {R.FLOOR_HALF_M:g} 0.5" material="floor" '
              'contype="0" conaffinity="0"/>']
    for body in sk.names:
        parts.append(f'<body name="{body}" mocap="true">')
        for g in geoms[body]:
            attrs = " ".join(f'{k}="{v}"' for k, v in g.items() if k in ("type", "size", "pos", "quat", "fromto"))
            parts.append(f'<geom {attrs} material="{zone_of[body]}" contype="0" conaffinity="0"/>')
        parts.append("</body>")
    if mesh is not None:
        parts.append(f'<geom name="human_mesh" type="mesh" mesh="human" rgba="{rgb(mesh.rgba[:3])} '
                     f'{mesh.rgba[3]}" contype="0" conaffinity="0"/>')
    parts.append("</worldbody></mujoco>")
    return "".join(parts)


def glyphs(clip: R.Clip, frame: int, bar=None, drop_zones=None) -> list[tuple]:
    """``render.glyphs`` in this module's palette. ``drop_zones``: draw drop lines only under these zones
    (``None``: every zone, render_v3's rule)."""
    out = []
    eye3 = np.eye(3)
    mk, zones = R.drawn_markers(clip, frame)
    for p, z in zip(mk, zones):
        out.append((mujoco.mjtGeom.mjGEOM_SPHERE, (R.MARKER_RADIUS_M, 0, 0), p, eye3, (*zone_rgb(z), 1.0),
                    R.MARKER_EMISSION))
    for z, h, (x, y, _) in R.drop_lines(clip.pos[frame], clip.rot[frame]):
        if drop_zones is not None and z not in drop_zones:
            continue
        rgba = (*zone_rgb(z), 1.0)
        out.append((mujoco.mjtGeom.mjGEOM_CYLINDER, (R.DROP_RADIUS_M, h / 2, 0), (x, y, h / 2), eye3, rgba,
                    R.MARKER_EMISSION))
        for turn in (0.0, math.pi / 2):
            out.append((mujoco.mjtGeom.mjGEOM_BOX, R.CROSS_HALF_M, (x, y, R.CROSS_HALF_M[2] + 1e-4),
                        R._heading_matrix(math.cos(turn), math.sin(turn)), rgba, R.MARKER_EMISSION))
    if bar is not None:
        (cx, cy, _), (dx, dy) = bar
        mat = R._heading_matrix(dx, dy)
        for k in range(R.BAR_SEGMENTS):
            s = (k + 0.5) / R.BAR_SEGMENTS * R.BAR_LENGTH_M - R.BAR_LENGTH_M / 2
            out.append((mujoco.mjtGeom.mjGEOM_BOX, R.BAR_HALF_M, (cx + s * dx, cy + s * dy, R.BAR_HALF_M[2]), mat,
                        R._BLACK if k % 2 == 0 else R._WHITE, 0.0))
    return out


class Scene(R.Scene):
    """``render.Scene`` on plant v2 with this module's palette (and, for Pass C, drop lines limited to
    ``drop_zones(frame)``)."""

    def __init__(self, clip: R.Clip, human_mesh: R.HumanMesh | None = None, markers: bool = True, drop_zones=None):
        _require_plant()
        self.drop_zones = drop_zones
        super().__init__(clip, human_mesh, markers)

    def _build(self, frame: int | None) -> None:
        self.close()
        mesh_v = None if frame is None else self.human_mesh.vertices(frame)
        self.model = mujoco.MjModel.from_xml_string(scene_xml(mesh_v, self.human_mesh))
        self.data = mujoco.MjData(self.model)
        self._mocap = [self.model.body(b).mocapid[0] for b in capture.skeleton().names]
        self._mesh_frame = frame

    def render(self, panel: R.Panel, segmentation: bool = False) -> np.ndarray:
        cam = panel.camera
        self.pose(panel.frame)
        self.model.vis.global_.fovy = cam.fovy
        r = self._renderer(cam.width, cam.height)
        r.update_scene(self.data, camera=cam.mjv())
        zones = None if self.drop_zones is None else self.drop_zones(panel.frame)
        for gtype, size, pos, mat, rgba, emission in glyphs(self.clip, panel.frame, panel.bar, zones):
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE and not self.markers:
                continue
            R._glyph(r.scene, gtype, size, pos, mat, rgba, emission)
        if segmentation:
            r.enable_segmentation_rendering()
            try:
                return r.render().copy()
            finally:
                r.disable_segmentation_rendering()
        return r.render().copy()


def render_sheets(clip: R.Clip, sheets: list, first_number: int = 1, human_mesh: R.HumanMesh | None = None,
                  markers: bool = True) -> list:
    """``render.render_sheets`` on plant v2 (inside ``plant_context``)."""
    out = []
    with Scene(clip, human_mesh, markers) as scene:
        for k, sheet in enumerate(sheets):
            out.append(R.compose([scene.render(p) for p in sheet.panels], sheet, first_number + k))
    return out


def lifted(clip: R.Clip, lift_m: float) -> R.Clip:
    """A copy of ``clip`` with the avatar raised by ``lift_m`` (markers untouched): a float of known size."""
    pos = clip.pos.copy()
    pos[..., 2] += lift_m
    return R.Clip(clip.stem, clip.fps, pos, clip.rot, clip.markers, clip.trusted, clip.marker_labels, clip.marker_zone)
