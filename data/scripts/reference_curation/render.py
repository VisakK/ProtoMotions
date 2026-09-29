# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence renders (BUILD_PLAN Step 3): the same reproducible views of any clip frame, for you and
for the VLM reviewer. ``packets.py`` assembles them into review packets.

Scene
-----
The shipped reference avatar is drawn with its MJCF collision geoms: one mocap body per MJCF body,
posed straight from ``rigid_body_pos`` / ``rigid_body_rot``. It stands on a checked floor, and
evidence glyphs are drawn on top:

* **Palette.** Each segment family has its own hue, and left and right are a dark and a light
  shade of it (``FAMILIES``, ColorBrewer *Paired*). Hands, feet, head, arms and legs never share a
  colour. The spike's two-colour scheme made Crow read as a standing forward fold (BUILD_PLAN §5).
* **Markers.** The human's MoSh markers (``capture.markers``) are small glowing spheres, coloured
  like the avatar segment they belong to. A shared marker (ANK, KNE, ELB, IWR, ...) takes the
  more distal segment. Markers the fit cannot explain (NaN, or residual > 5 cm, the Step-1 hygiene
  rule) are not drawn.
* **Drop lines** (``drop_lines``). A thin black vertical line runs from a part's lowest point
  down to the floor and ends in a black cross. The lowest point comes from the kernels behind the
  store's ``avatar_min_z`` (``capture.body_min_z``), so the line's length *is* the store's height.
  Every hand, foot and the head gets one up to ``DROP_MAX_M`` (60 cm): a raised foot's 1.7 m line
  read as a pole in the insets. Since render_v2 every other part gets one up to ``DROP_MAX_BODY_M``
  (15 cm, the float range), unless the line would pass through another part (a shin over its own
  foot), and no part within ``DROP_MIN_M`` (1 cm) of the floor gets a line or a cross: it touches.
  Both changes come from render_v1's Step 4 calibration. The shipped reference is grounded with a
  5 mm clearance, so v1's line under every planted hand and foot read as a gap: 28 false floats, all
  at 0.50-0.97 cm, and float precision 0.72 at recall 0.99. And forearms had no line: 7 of the 14
  forearms 2-5 cm up were read as resting on the floor, against none of them in v2 and v3.
  Since render_v3 a line and its cross have the colour of the part they hang from, and glow like
  the markers. render_v2's black lines could not say whose they were: a forearm line beside a
  planted hand read as the hand's ("Hand box has drop lines", Dolphin -a), and a raised foot's line
  landing next to the standing foot read as the standing foot's (Tree -b).
* **Floor.** 10 cm checks, aligned to the clip frame's origin. A 1 m scale bar of ten black and
  white segments lies in the eye-level and high views.
* **Human mesh (Step 5).** ``HumanMesh`` is the slot: a posed mesh in the clip frame, rendered in
  translucent grey.

Nothing casts a shadow. The spike's reviewer read marker shadows as markers (BUILD_PLAN §5), and
the drop lines, the grazing view and the insets carry the float cue a shadow would. Under OSMesa an
avatar shadow also brought artifacts: a false shadow bar on the floor beyond the directional
shadow map, or jagged edges and acne streaks once the map was widened to cover the floor
(measured). The glyphs are ``mjCAT_DECOR`` scene geoms anyway, which is the one category that casts
no shadow (measured: 0, 1 and 2 all do), so a shadowed render_v cannot bring the marker shadows back.
Nothing is recentred: the world *is* the clip frame, so no render offset can enter a measurement.

Views
-----
``plan()`` returns the sheets of one frame. Each sheet is one image of at most 1024 px.

* ``eye``: four panels at eye level (elevation -12 deg), 90 deg apart. The first looks across the
  body's long horizontal axis (``side_azimuth``) from the side the pelvis faces.
* ``high_grazing``: the high view (-55 deg), and the grazing view. The grazing camera is
  ``FLOOR_CAMERA_M`` (2 cm) above the floor, tilted up so the horizon sits 12 % above the panel's
  bottom edge; any gap under a body part shows against the far floor.
* ``insets``: floor-level close-ups, ``INSET_WIDTH_M`` (0.6 m) wide at the target. They show each
  hand, foot and the head within 20 cm of the floor (avatar or marker), plus every zone the caller
  asks for (the queue item's labelled and flagged zones). Targets within 25 cm share an inset. The
  camera is 2 cm above the floor and looks in from outside the body.
* ``strip``: one eye-level camera at up to five frames, framed on all of them.

Every full-body panel is auto-framed. The camera's distance and centre are solved in closed form
from the corners of every collision geom's bounding box, the drawn markers, the drop-line feet and
the scale bar, so nothing is cropped. ``Camera.project`` reproduces MuJoCo's free camera to
sub-pixel accuracy, and the tests check it against a render.

CLI (not blind: for looking at an arbitrary frame; packets are built by ``packets.py``)::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.render \\
        --stem 220923_Crane_Crow_Pose_or_Bakasana_-a --frame 439 --zones L_FOOT
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# MuJoCo fixes its GL backend when it is imported. OSMesa (Mesa llvmpipe, on the CPU) renders
# bit-exact whatever the process did before. EGL on the NVIDIA driver drifted by one level in two
# pixels of a shaded sphere after a few hundred renders in one process (measured), which changes
# content-addressed packet ids. The CPU path is as fast for these scenes (~1 s per packet) and
# leaves the GPU to training. REFERENCE_CURATION_GL overrides it; gl_info() records what ran.
GL_BACKEND = os.environ.get("REFERENCE_CURATION_GL", "osmesa")
os.environ["MUJOCO_GL"] = GL_BACKEND
if GL_BACKEND == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco  # noqa: E402
from mujoco import gl_context as _gl_context  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from contact_geometry import geom_ground_distance, geom_pair_distance, geom_to_world  # noqa: E402
from extract_contact_configs import ZONE_ORDER  # noqa: E402
from reference_curation import capture, ids  # noqa: E402

MODULE = "reference_curation.render"
RENDER_V = "render_v3"

SHEET_MAX_PX = 1024
GUTTER_PX = 4
FOVY_DEG = 45.0
INSET_FOVY_DEG = 30.0
EYE_ELEVATION_DEG = -12.0
HIGH_ELEVATION_DEG = -55.0
HIGH_AZIMUTH_OFFSET_DEG = 45.0
FLOOR_CAMERA_M = 0.02      # grazing and inset cameras; the card allows <= 3 cm
GRAZING_HORIZON = 0.12     # the horizon sits this fraction of the panel height above the bottom edge
INSET_WIDTH_M = 0.6
CHECK_M = 0.10             # floor check size; the builtin checker texture is 2 x 2 checks
FLOOR_HALF_M = 6.0         # the corpus's bodies and markers stay within 2.54 m of the origin
INSET_FLOOR = 0.15         # an inset's floor point sits this fraction of the panel height up
NEAR_FLOOR_M = 0.20        # a hand, foot or head this close to the floor gets an inset
MERGE_M = 0.25             # inset targets closer than this share one inset
OUTSIDE_M = 0.15           # an inset target this close to the body centre is viewed from the side
MAX_INSETS = 12
FRAME_MARGIN = 0.05        # fraction of every half-extent kept clear at the panel edges
MIN_DEPTH_M = 0.10         # nothing framed may sit closer than this to a camera
MARKER_RADIUS_M = 0.009
MARKER_EMISSION = 0.2      # higher washes the light shades out to white
DROP_RADIUS_M = 0.0035     # render_v3: coloured, so a little thicker than v1-v2's black 2.5 mm
DROP_MAX_M = 0.60          # a longer line (a raised foot) reads as a pole in the insets
DROP_MIN_M = 0.01          # render_v2: a part this close to the floor touches it (the reference's clearance is 5 mm)
DROP_MAX_BODY_M = 0.15     # render_v2: the parts other than hands, feet and head get a line up to here
CROSS_HALF_M = (0.025, 0.003, 0.001)
BAR_LENGTH_M = 1.0
BAR_SEGMENTS = 10
BAR_HALF_M = (0.05, 0.015, 0.002)   # one 10 cm segment
BAR_CLEARANCE_M = 0.25
EXTREMITIES = ("L_HAND", "R_HAND", "L_FOOT", "R_FOOT", "HEAD")
INSET_ORDER = ("HEAD", "L_HAND", "R_HAND", "L_FOOT", "R_FOOT") + tuple(
    z for z in ZONE_ORDER if z not in EXTREMITIES)
DISTAL = ("L_HAND", "R_HAND", "L_FOOT", "R_FOOT", "HEAD", "L_FOREARM", "R_FOREARM", "L_SHANK",
          "R_SHANK", "L_UPPER_ARM", "R_UPPER_ARM", "L_THIGH", "R_THIGH", "PELVIS", "TRUNK")

# Segment families: colour words for the legend, then the left (dark) and right (light) shade.
FAMILIES = {
    "foot": ("blue", (0.122, 0.471, 0.706), (0.651, 0.808, 0.890)),
    "leg": ("green", (0.200, 0.627, 0.173), (0.698, 0.875, 0.541)),
    "hand": ("red", (0.890, 0.102, 0.110), (0.984, 0.604, 0.600)),
    "arm": ("orange", (1.000, 0.498, 0.000), (0.992, 0.749, 0.435)),
    "head": ("purple", (0.416, 0.239, 0.604), None),
    "torso": ("grey", (0.600, 0.600, 0.620), None),
}
ZONE_FAMILY = {"L_FOOT": "foot", "R_FOOT": "foot", "L_SHANK": "leg", "R_SHANK": "leg", "L_THIGH": "leg",
               "R_THIGH": "leg", "PELVIS": "torso", "TRUNK": "torso", "HEAD": "head", "L_UPPER_ARM": "arm",
               "R_UPPER_ARM": "arm", "L_FOREARM": "arm", "R_FOREARM": "arm", "L_HAND": "hand", "R_HAND": "hand"}
ZONE_WORDS = {"L_FOOT": "left foot", "R_FOOT": "right foot", "L_SHANK": "left shin", "R_SHANK": "right shin",
              "L_THIGH": "left thigh", "R_THIGH": "right thigh", "PELVIS": "pelvis", "TRUNK": "torso",
              "HEAD": "head", "L_UPPER_ARM": "left upper arm", "R_UPPER_ARM": "right upper arm",
              "L_FOREARM": "left forearm", "R_FOREARM": "right forearm", "L_HAND": "left hand",
              "R_HAND": "right hand"}
_DECOR = int(mujoco.mjtCatBit.mjCAT_DECOR)
_ACTIVE_GL = _gl_context.GLContext.__module__.rsplit(".", 1)[-1]  # mujoco imported earlier keeps its own
_BLACK = (0.05, 0.05, 0.05, 1.0)
_WHITE = (0.97, 0.97, 0.97, 1.0)


def zone_rgb(zone: str) -> tuple[float, float, float]:
    _, left, right = FAMILIES[ZONE_FAMILY[zone]]
    return right if zone.startswith("R_") and right is not None else left


def colour_words(zone: str) -> str:
    """``dark blue`` / ``light blue`` for a sided zone, the plain hue for the head and torso."""
    word, _, right = FAMILIES[ZONE_FAMILY[zone]]
    if right is None:
        return word
    return f"{'light' if zone.startswith('R_') else 'dark'} {word}"


def palette_legend() -> dict:
    """Plain-language colour key, one entry per drawn segment family and side."""
    parts = {"head": "head and neck", "torso": "torso and pelvis", "arm": "arm (upper arm and forearm)",
             "leg": "leg (thigh and shin)", "hand": "hand", "foot": "foot"}
    out = {}
    for fam in ("head", "torso", "arm", "hand", "leg", "foot"):
        zone = next(z for z, f in ZONE_FAMILY.items() if f == fam)
        if FAMILIES[fam][2] is None:
            out[parts[fam]] = colour_words(zone)
        else:
            base = zone[2:]
            out[f"left {parts[fam]}"] = colour_words(f"L_{base}")
            out[f"right {parts[fam]}"] = colour_words(f"R_{base}")
    return out


# --------------------------------------------------------------------------- #
# Clip data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Clip:
    stem: str
    fps: int
    pos: np.ndarray              # [T, B, 3] clip frame, COMMON (MJCF) body order
    rot: np.ndarray              # [T, B, 4] xyzw
    markers: np.ndarray | None   # [T, M, 3] clip frame (Vicon XY + capture.VICON_TO_CLIP_XY)
    trusted: np.ndarray | None   # [T, M] the Step-1 hygiene rule: finite and residual <= 5 cm
    marker_labels: tuple         # [M]
    marker_zone: tuple           # [M] zone each marker is coloured as

    @property
    def num_frames(self) -> int:
        return self.pos.shape[0]


def _marker_zones(labels: list[str]) -> tuple:
    owners = {m: [z for z in DISTAL if m in capture.ZONE_MARKERS[z]] for m in labels}
    return tuple(owners[m][0] for m in labels)


@functools.lru_cache(maxsize=8)
def load_clip(stem: str, motion_dir: Path = ids.SHIPPED_DIR) -> Clip:
    motion = capture._load_motion(ids.motion_path(stem, motion_dir))  # checks COMMON body order
    pos = motion["rigid_body_pos"].double().numpy()
    rot = motion["rigid_body_rot"].double().numpy()
    obs, sim = capture.markers(stem, "obs"), capture.markers(stem, "sim")
    if obs is None or obs[0].shape[0] != pos.shape[0]:
        return Clip(stem, int(motion["fps"]), pos, rot, None, None, (), ())
    resid = np.linalg.norm(obs[0] - sim[0], axis=-1)
    trusted = np.isfinite(obs[0]).all(-1) & (np.nan_to_num(resid, nan=np.inf) <= capture.RESID_MAX_M)
    return Clip(stem, int(motion["fps"]), pos, rot, obs[0], trusted, tuple(obs[1]), _marker_zones(obs[1]))


# --------------------------------------------------------------------------- #
# Pose geometry (pure)
# --------------------------------------------------------------------------- #
def _quat_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _world_geoms(pos, rot):
    """``[(body index, world typed geom)]`` for one frame (float32, as ``capture.body_min_z``)."""
    sk = capture.skeleton()
    p = torch.as_tensor(pos, dtype=torch.float32)[None]
    r = torch.as_tensor(rot, dtype=torch.float32)[None]
    return [(i, geom_to_world(g, p[:, i], r[:, i])) for i, b in enumerate(sk.names) for g in sk.geoms[b]]


def zone_lowest(pos, rot) -> tuple[np.ndarray, np.ndarray]:
    """``(height [Z], point [Z, 3])``: every zone's lowest collision surface and where it is. The
    kernels are ``capture.body_min_z``'s, so ``height`` equals the store's ``avatar_min_z``."""
    sk = capture.skeleton()
    best = {}
    for i, g in _world_geoms(pos, rot):
        gap, witness = geom_ground_distance(g)
        if i not in best or float(gap[0]) < best[i][0]:
            best[i] = (float(gap[0]), witness[0].double().numpy())
    heights, points = [], []
    for idx in sk.zone_bodies:
        h, p = min((best[i] for i in idx), key=lambda hp: hp[0])
        heights.append(h)
        points.append(p)
    return np.array(heights), np.stack(points)


def drop_lines(pos, rot) -> list[tuple[str, float, np.ndarray]]:
    """``[(zone, height, lowest point)]``: the frame's drop lines. Every hand, foot and head
    between ``DROP_MIN_M`` and ``DROP_MAX_M`` up gets one; every other part between ``DROP_MIN_M``
    and ``DROP_MAX_BODY_M`` up gets one unless its line would pass through another zone's geom."""
    heights, low = zone_lowest(pos, rot)
    zone_of = {i: z for z, idx in zip(ZONE_ORDER, capture.skeleton().zone_bodies) for i in idx}
    geoms, out = None, []
    for zi, z in enumerate(ZONE_ORDER):
        h, (x, y, top) = float(heights[zi]), low[zi]
        if not DROP_MIN_M < h <= (DROP_MAX_M if z in EXTREMITIES else DROP_MAX_BODY_M):
            continue
        if z not in EXTREMITIES:
            geoms = _world_geoms(pos, rot) if geoms is None else geoms
            line = {"type": "capsule", "radius": DROP_RADIUS_M,
                    "a": torch.tensor([[x, y, top - 0.001]], dtype=torch.float32),
                    "b": torch.tensor([[x, y, 0.0]], dtype=torch.float32)}
            if any(zone_of[i] != z and float(geom_pair_distance(line, g)[0][0]) < 0 for i, g in geoms):
                continue
        out.append((z, h, low[zi]))
    return out


def geom_corners(pos, rot) -> np.ndarray:
    """``[N, 3]`` corners of every collision geom's world bounding box: a superset of the avatar."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    out = []
    for _, g in _world_geoms(pos, rot):
        if g["type"] == "sphere":
            c = g["center"][0].double().numpy()
            lo, hi = c - g["radius"], c + g["radius"]
        elif g["type"] == "capsule":
            ends = np.stack([g["a"][0].double().numpy(), g["b"][0].double().numpy()])
            lo, hi = ends.min(0) - g["radius"], ends.max(0) + g["radius"]
        else:
            box = g["center"][0].double().numpy() + (signs * g["half"].double().numpy()) @ _quat_matrix(g["quat"][0]).T
            lo, hi = box.min(0), box.max(0)
        out.append(lo + (signs + 1) / 2 * (hi - lo))
    return np.concatenate(out)


def drawn_markers(clip: Clip, frame: int) -> tuple[np.ndarray, list[str]]:
    """``(positions [K, 3], zone per marker)`` of the markers drawn on ``frame``: the trusted ones."""
    if clip.markers is None:
        return np.zeros((0, 3)), []
    keep = np.nonzero(clip.trusted[frame])[0]
    return clip.markers[frame, keep], [clip.marker_zone[k] for k in keep]


def marker_min_z(clip: Clip, frame: int) -> dict[str, float]:
    """Lowest drawn marker per zone of ``capture.ZONE_MARKERS``: the zone's own marker map."""
    if clip.markers is None:
        return {}
    labels = clip.marker_labels
    out = {}
    for zone, names in capture.ZONE_MARKERS.items():
        idx = [labels.index(m) for m in names if clip.trusted[frame, labels.index(m)]]
        if idx:
            out[zone] = float(clip.markers[frame, idx, 2].min())
    return out


def side_azimuth(pos, rot) -> float:
    """Azimuth (deg) of a camera looking across the body's long horizontal axis, from the side the
    pelvis faces (its local x axis). The long axis is the principal axis of the bodies' xy."""
    xy = pos[:, :2] - pos[:, :2].mean(0)
    axis = np.linalg.eigh(xy.T @ xy)[1][:, -1]
    look = np.array([-axis[1], axis[0]])            # the camera's horizontal forward direction
    heading = _quat_matrix(rot[0])[:2, 0]
    side = float(look @ heading)
    if side > 1e-9 or (abs(side) <= 1e-9 and look[0] + 1e-3 * look[1] < 0):  # a fixed tie-break
        look = -look                                  # the camera sits at -look: in front of the pelvis
    return math.degrees(math.atan2(look[1], look[0])) % 360.0


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def _basis(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MuJoCo's free camera: ``(forward, up, right)``; negative elevation looks down."""
    a, e = math.radians(azimuth), math.radians(elevation)
    forward = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    return forward, up, np.cross(forward, up)


@dataclass(frozen=True)
class Camera:
    lookat: tuple
    distance: float
    azimuth: float
    elevation: float
    fovy: float
    width: int
    height: int

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.lookat) - self.distance * _basis(self.azimuth, self.elevation)[0]

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """``(uv [N, 2] in pixels, depth [N])``: u right, v down, pixel ``i`` spans ``[i, i + 1)``."""
        f, up, right = _basis(self.azimuth, self.elevation)
        q = np.atleast_2d(points) - self.position
        z, t = q @ f, math.tan(math.radians(self.fovy) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.width / 2 + (q @ right) / z / t * self.height / 2
            v = self.height / 2 - (q @ up) / z / t * self.height / 2
        return np.stack([u, v], -1), z

    def mjv(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        cam.distance, cam.azimuth, cam.elevation = self.distance, self.azimuth, self.elevation
        return cam

    def record(self) -> dict:
        return {"lookat": [round(float(v), 5) for v in self.lookat], "distance": round(self.distance, 5),
                "azimuth": round(self.azimuth, 4), "elevation": round(self.elevation, 4), "fovy": self.fovy,
                "size": [self.width, self.height]}


def _tangents(fovy: float, width: int, height: int, margin: float) -> tuple[float, float]:
    ty = math.tan(math.radians(fovy) / 2) * (1 - margin)
    return ty * width / height, ty


def fit_free(points, azimuth: float, elevation: float, width: int, height: int, fovy: float = FOVY_DEG,
             margin: float = FRAME_MARGIN) -> Camera:
    """The closest camera with this orientation that frames every point, centred on them."""
    f, up, right = _basis(azimuth, elevation)
    tx, ty = _tangents(fovy, width, height, margin)
    pts = np.asarray(points, dtype=float)
    lookat = (pts.min(0) + pts.max(0)) / 2

    def distance(at):
        q = pts - at
        x, y, d = q @ right, q @ up, q @ f
        return max(float(np.max(np.maximum(np.abs(x) / tx, np.abs(y) / ty) - d)), float(np.max(MIN_DEPTH_M - d)))

    for _ in range(4):  # centre the projected extents; the final distance is exact for the final centre
        D = distance(lookat)
        q = pts - lookat
        z = D + q @ f
        xn, yn = (q @ right) / z, (q @ up) / z
        lookat = lookat + right * D * (xn.max() + xn.min()) / 2 + up * D * (yn.max() + yn.min()) / 2
    return Camera(tuple(lookat), distance(lookat), azimuth % 360.0, elevation, fovy, width, height)


def _fits(cam: Camera, pts: np.ndarray, margin: float) -> bool:
    uv, z = cam.project(pts)
    lo = np.array([cam.width, cam.height]) * margin / 2
    hi = np.array([cam.width, cam.height]) - lo
    return bool((z > MIN_DEPTH_M).all() and (uv >= lo).all() and (uv <= hi).all())


def fit_floor(points, azimuth: float, width: int, height: int, fovy: float = FOVY_DEG,
              camera_height: float = FLOOR_CAMERA_M, horizon: float = GRAZING_HORIZON,
              margin: float = FRAME_MARGIN) -> Camera:
    """Grazing view: the camera is ``camera_height`` above the floor and tilted up so the horizon sits
    ``horizon`` of the panel height above its bottom edge; the closest such camera that frames every
    point, centred left-right."""
    elevation = math.degrees(math.atan((1 - 2 * horizon) * math.tan(math.radians(fovy) / 2)))
    f, up, right = _basis(azimuth, elevation)
    pts = np.asarray(points, dtype=float)
    centre = (pts.min(0) + pts.max(0)) / 2
    ground = np.array([centre[0], centre[1], camera_height])
    g = np.array([f[0], f[1], 0.0]) / math.hypot(f[0], f[1])

    def camera(back: float) -> Camera:
        lateral = 0.0
        for _ in range(4):  # centre left-right at this distance
            c = ground - back * g + lateral * right
            q = pts - c
            xn = (q @ right) / (q @ f)
            lateral += float((xn.max() + xn.min()) / 2) * back
        c = ground - back * g + lateral * right
        return Camera(tuple(c + back * f), back, azimuth % 360.0, elevation, fovy, width, height)

    lo, hi = 0.3, 60.0
    if not _fits(camera(hi), pts, margin):
        raise ValueError("points cannot be framed from the floor")
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if _fits(camera(mid), pts, margin) else (mid, hi)
    return camera(hi)


def fit_inset(target_xy, azimuth: float, width: int, height: int, fovy: float = INSET_FOVY_DEG,
              width_m: float = INSET_WIDTH_M, camera_height: float = FLOOR_CAMERA_M,
              floor: float = INSET_FLOOR) -> Camera:
    """Floor-level close-up: the floor point ``target_xy`` lands centred left-right and ``floor`` of
    the panel height above its bottom edge, at the depth where the panel is ``width_m`` wide. The
    camera is ``camera_height`` above the floor, looking along ``azimuth``."""
    t = math.tan(math.radians(fovy) / 2)
    depth = width_m / 2 / (t * width / height)
    k = 1 - 2 * floor
    back = math.sqrt(depth ** 2 * (1 + (k * t) ** 2) - camera_height ** 2)   # horizontal distance
    elevation = math.degrees(math.atan(k * t) - math.atan(camera_height / back))
    f, _, _ = _basis(azimuth, elevation)
    g = np.array([f[0], f[1], 0.0]) / math.hypot(f[0], f[1])
    c = np.array([target_xy[0], target_xy[1], camera_height]) - back * g
    return Camera(tuple(c + depth * f), depth, azimuth % 360.0, elevation, fovy, width, height)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Panel:
    kind: str                 # eye | high | grazing | inset | strip
    frame: int
    camera: Camera
    bar: tuple | None = None  # (centre xyz, direction xy) of the scale bar, if drawn
    zones: tuple = ()         # inset: the zones it is centred under
    target: tuple | None = None  # inset: the floor point it is centred on


@dataclass(frozen=True)
class Sheet:
    kind: str                 # eye | high_grazing | insets | strip
    panels: tuple
    cols: int
    rows: int


def grid(n: int) -> tuple[int, int]:
    """``(cols, rows)`` of a sheet of ``n`` panels (at most 6)."""
    return {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2), 6: (3, 2)}[n]


def panel_px(cols: int) -> int:
    return (SHEET_MAX_PX - (cols - 1) * GUTTER_PX) // cols


def framing_points(clip: Clip, frame: int) -> np.ndarray:
    """Everything a full-body view must show: geom boxes, drawn markers (with their radius) and
    the feet of the drop lines."""
    pos, rot = clip.pos[frame], clip.rot[frame]
    mk, _ = drawn_markers(clip, frame)
    heights, low = zone_lowest(pos, rot)
    drawn = [ZONE_ORDER.index(z) for z in EXTREMITIES if heights[ZONE_ORDER.index(z)] <= DROP_MAX_M]
    feet = low[drawn] * np.array([1.0, 1.0, 0.0])
    blob = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * MARKER_RADIUS_M
    return np.concatenate([geom_corners(pos, rot), (mk[:, None] + blob).reshape(-1, 3), feet])


def scale_bar(points: np.ndarray, azimuth: float) -> tuple:
    """The bar lies on the floor between the first eye-level camera and the body, across its image."""
    f, _, right = _basis(azimuth, 0.0)
    xy = points[:, :2]
    centre = (xy.min(0) + xy.max(0)) / 2
    reach = float(np.max((xy - centre) @ -f[:2]))
    c = centre - f[:2] * (reach + BAR_CLEARANCE_M)
    return (float(c[0]), float(c[1]), 0.0), (float(right[0]), float(right[1]))


def bar_points(bar) -> np.ndarray:
    (cx, cy, _), (dx, dy) = bar
    c, d, n = np.array([cx, cy]), np.array([dx, dy]), np.array([-dy, dx])
    ends = [c + s * BAR_LENGTH_M / 2 * d + w * BAR_HALF_M[1] * n for s in (-1, 1) for w in (-1, 1)]
    return np.array([[p[0], p[1], h] for p in ends for h in (0.0, 2 * BAR_HALF_M[2])])


def inset_targets(clip: Clip, frame: int, zones=()) -> list[tuple[tuple, np.ndarray]]:
    """``[(zones, floor xy)]``: the extremities within ``NEAR_FLOOR_M`` of the floor (avatar or
    markers) and every requested zone, in ``INSET_ORDER``; targets within ``MERGE_M`` merge."""
    heights, low = zone_lowest(clip.pos[frame], clip.rot[frame])
    marker = marker_min_z(clip, frame)
    wanted = set(zones) | {z for z in EXTREMITIES if min(heights[ZONE_ORDER.index(z)],
                                                             marker.get(z, np.inf)) <= NEAR_FLOOR_M}
    groups: list[list] = []
    for z in (z for z in INSET_ORDER if z in wanted):
        xy = low[ZONE_ORDER.index(z), :2]
        near = next((g for g in groups if np.linalg.norm(np.mean(g[1], 0) - xy) < MERGE_M), None)
        if near is None:
            groups.append([[z], [xy]])
        else:
            near[0].append(z)
            near[1].append(xy)
    return [(tuple(g[0]), np.mean(g[1], 0)) for g in groups][:MAX_INSETS]


def inset_panels(clip: Clip, frame: int, zones=(), side: float | None = None) -> list[list[Panel]]:
    """Inset panels, chunked into sheets of at most six."""
    pos = clip.pos[frame]
    side = side_azimuth(pos, clip.rot[frame]) if side is None else side
    centre = pos[:, :2].mean(0)
    targets = inset_targets(clip, frame, zones)
    sheets = []
    for start in range(0, len(targets), 6):
        chunk = targets[start:start + 6]
        size = panel_px(grid(len(chunk))[0])
        panels = []
        for zs, xy in chunk:
            inward = centre - xy
            az = side if np.linalg.norm(inward) < OUTSIDE_M else math.degrees(math.atan2(inward[1], inward[0]))
            panels.append(Panel("inset", frame, fit_inset(xy, az, size, size), zones=zs,
                                target=(float(xy[0]), float(xy[1]), 0.0)))
        sheets.append(panels)
    return sheets


def plan(clip: Clip, frame: int, zones=(), strip=None) -> list[Sheet]:
    """The sheets of one frame: ``eye``, ``high_grazing``, ``insets`` (0-2 sheets), then ``strip``
    if ``strip`` lists frames."""
    if not 0 <= frame < clip.num_frames:
        raise IndexError(f"{clip.stem}: frame {frame} outside 0..{clip.num_frames - 1}")
    unknown = set(zones) - set(ZONE_ORDER)
    if unknown:
        raise ValueError(f"unknown zones {sorted(unknown)}")
    pos, rot = clip.pos[frame], clip.rot[frame]
    side = side_azimuth(pos, rot)
    pts = framing_points(clip, frame)
    bar = scale_bar(pts, side)
    with_bar = np.concatenate([pts, bar_points(bar)])
    size = panel_px(2)
    eye = tuple(Panel("eye", frame, fit_free(with_bar, side + 90 * k, EYE_ELEVATION_DEG, size, size), bar)
                for k in range(4))
    high = Panel("high", frame, fit_free(with_bar, side + HIGH_AZIMUTH_OFFSET_DEG, HIGH_ELEVATION_DEG, size, size),
                 bar)
    grazing = Panel("grazing", frame, fit_floor(pts, side, size, size))
    sheets = [Sheet("eye", eye, 2, 2), Sheet("high_grazing", (high, grazing), 2, 1)]
    for panels in inset_panels(clip, frame, zones, side):
        sheets.append(Sheet("insets", tuple(panels), *grid(len(panels))))
    if strip:
        frames = sorted({int(f) for f in strip})
        if len(frames) > 6 or not all(0 <= f < clip.num_frames for f in frames):
            raise ValueError(f"strip frames {frames}: at most 6, inside the clip")
        union = np.concatenate([framing_points(clip, f) for f in frames])
        cols, rows = grid(len(frames))
        size = panel_px(cols)
        cam = fit_free(union, side, EYE_ELEVATION_DEG, size, size)
        sheets.append(Sheet("strip", tuple(Panel("strip", f, cam) for f in frames), cols, rows))
    return sheets


# --------------------------------------------------------------------------- #
# Scene and rendering
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HumanMesh:
    """Step 5's layer: ``vertices(frame)`` is the posed mesh ``[V, 3]`` in the clip frame."""
    faces: np.ndarray
    vertices: Callable[[int], np.ndarray]
    rgba: tuple = (0.55, 0.55, 0.55, 0.35)


@functools.lru_cache(maxsize=1)
def _body_geoms() -> dict:
    """MJCF body name -> its ``<geom>`` attributes (body frame)."""
    out = {}

    def walk(body):
        out[body.attrib["name"]] = [dict(g.attrib) for g in body.findall("geom")]
        for child in body.findall("body"):
            walk(child)

    for body in ET.parse(str(ids.MJCF)).getroot().find("worldbody").findall("body"):
        walk(body)
    return out


def scene_xml(mesh_vertices: np.ndarray | None = None, mesh: HumanMesh | None = None) -> str:
    """The MJCF of the render scene: floor, light, the avatar's mocap bodies, optional human mesh."""
    rgb = lambda c: " ".join(f"{v:.3f}" for v in c)  # noqa: E731
    sk = capture.skeleton()
    zone_of = {sk.names[i]: z for z, idx in zip(ZONE_ORDER, sk.zone_bodies) for i in idx}
    geoms = _body_geoms()
    materials = "".join(f'<material name="{z}" rgba="{rgb(zone_rgb(z))} 1" specular="0.15" shininess="0.2"/>'
                        for z in ZONE_ORDER)
    parts = [
        '<mujoco model="evidence">',
        f'<visual><global offwidth="{SHEET_MAX_PX}" offheight="{SHEET_MAX_PX}" fovy="{FOVY_DEG}"/>'
        '<quality shadowsize="512" offsamples="8"/><map znear="0.01" zfar="50"/>'
        '<headlight ambient=".42 .42 .42" diffuse=".45 .45 .45" specular="0 0 0"/></visual>',
        '<statistic extent="2" center="0 0 0.8"/>',
        '<asset><texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .89 .93" '
        'width="64" height="384"/>'
        '<texture name="checks" type="2d" builtin="checker" rgb1=".80 .80 .78" rgb2=".66 .66 .64" '
        'width="256" height="256"/>'
        f'<material name="floor" texture="checks" texrepeat="{1 / (2 * CHECK_M):g} {1 / (2 * CHECK_M):g}" '
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
              f'<geom name="floor" type="plane" size="{FLOOR_HALF_M:g} {FLOOR_HALF_M:g} 0.5" material="floor" '
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


def _glyph(scn, gtype, size, pos, mat, rgba, emission: float = 0.0) -> None:
    if scn.ngeom >= scn.maxgeom:
        raise RuntimeError("scene geom buffer full")
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                        np.asarray(mat, dtype=float).ravel(), np.asarray(rgba, dtype=np.float32))
    g.category = _DECOR  # decor geoms cast no shadow (measured; categories 0-2 all do)
    g.emission, g.specular, g.shininess, g.reflectance = emission, 0.0, 0.0, 0.0
    # mjv_initGeom leaves category, objtype, objid, segid, transparent and camdist as whatever the
    # malloc'd scene buffer held (measured). A garbage segid made segmentation index out of range; a
    # garbage transparent flag would send the glyph through the depth-sorted transparency pass.
    g.segid, g.objid, g.objtype = scn.ngeom, -1, int(mujoco.mjtObj.mjOBJ_UNKNOWN)
    g.transparent = int(rgba[3] < 1.0)
    g.camdist = float(np.linalg.norm(np.asarray(pos, dtype=float) - np.asarray(scn.camera[0].pos)))
    scn.ngeom += 1


def _heading_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[dx, -dy, 0.0], [dy, dx, 0.0], [0.0, 0.0, 1.0]])


def glyphs(clip: Clip, frame: int, bar=None) -> list[tuple]:
    """The decor geoms of one frame: markers, drop lines with floor crosses, and the scale bar."""
    out = []
    eye3 = np.eye(3)
    mk, zones = drawn_markers(clip, frame)
    for p, z in zip(mk, zones):
        out.append((mujoco.mjtGeom.mjGEOM_SPHERE, (MARKER_RADIUS_M, 0, 0), p, eye3, (*zone_rgb(z), 1.0),
                    MARKER_EMISSION))
    for z, h, (x, y, _) in drop_lines(clip.pos[frame], clip.rot[frame]):
        rgba = (*zone_rgb(z), 1.0)
        out.append((mujoco.mjtGeom.mjGEOM_CYLINDER, (DROP_RADIUS_M, h / 2, 0), (x, y, h / 2), eye3, rgba,
                    MARKER_EMISSION))
        for turn in (0.0, math.pi / 2):
            out.append((mujoco.mjtGeom.mjGEOM_BOX, CROSS_HALF_M, (x, y, CROSS_HALF_M[2] + 1e-4),
                        _heading_matrix(math.cos(turn), math.sin(turn)), rgba, MARKER_EMISSION))
    if bar is not None:
        (cx, cy, _), (dx, dy) = bar
        mat = _heading_matrix(dx, dy)
        for k in range(BAR_SEGMENTS):
            s = (k + 0.5) / BAR_SEGMENTS * BAR_LENGTH_M - BAR_LENGTH_M / 2
            out.append((mujoco.mjtGeom.mjGEOM_BOX, BAR_HALF_M, (cx + s * dx, cy + s * dy, BAR_HALF_M[2]), mat,
                        _BLACK if k % 2 == 0 else _WHITE, 0.0))
    return out


class Scene:
    """A MuJoCo scene of one clip. ``render(panel)`` poses the avatar at the panel's frame, adds its
    glyphs and renders it. Use it as a context manager: the GL contexts are released on exit. With a
    ``human_mesh`` the model is rebuilt whenever the frame changes, the mesh baked in."""

    def __init__(self, clip: Clip, human_mesh: HumanMesh | None = None, markers: bool = True):
        self.clip, self.human_mesh, self.markers = clip, human_mesh, markers
        self._renderers: dict = {}
        self._mesh_frame = None
        self.model = self.data = None
        if human_mesh is None:
            self._build(None)

    def _build(self, frame: int | None) -> None:
        self.close()
        mesh_v = None if frame is None else self.human_mesh.vertices(frame)
        self.model = mujoco.MjModel.from_xml_string(scene_xml(mesh_v, self.human_mesh))
        self.data = mujoco.MjData(self.model)
        self._mocap = [self.model.body(b).mocapid[0] for b in capture.skeleton().names]
        self._mesh_frame = frame

    def pose(self, frame: int) -> None:
        if self.human_mesh is not None and frame != self._mesh_frame:
            self._build(frame)
        self.data.mocap_pos[self._mocap] = self.clip.pos[frame]
        self.data.mocap_quat[self._mocap] = self.clip.rot[frame][:, [3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)

    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        if (width, height) not in self._renderers:
            self._renderers[width, height] = mujoco.Renderer(self.model, height, width)
        return self._renderers[width, height]

    def render(self, panel: Panel, segmentation: bool = False) -> np.ndarray:
        """``[H, W, 3]`` uint8, or ``[H, W, 2]`` (object id, object type) with ``segmentation``."""
        cam = panel.camera
        self.pose(panel.frame)
        self.model.vis.global_.fovy = cam.fovy
        r = self._renderer(cam.width, cam.height)
        r.update_scene(self.data, camera=cam.mjv())
        for gtype, size, pos, mat, rgba, emission in glyphs(self.clip, panel.frame, panel.bar):
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE and not self.markers:
                continue
            _glyph(r.scene, gtype, size, pos, mat, rgba, emission)
        if segmentation:
            r.enable_segmentation_rendering()
            try:
                return r.render().copy()
            finally:
                r.disable_segmentation_rendering()
        return r.render().copy()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@functools.lru_cache(maxsize=1)
def gl_info() -> dict:
    """The GL backend and renderer that draw the images, for provenance and build keys."""
    from OpenGL import GL

    r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8)
    try:
        r.render()  # makes its context current
        return {"backend": _ACTIVE_GL, "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
                "version": GL.glGetString(GL.GL_VERSION).decode()}
    finally:
        r.close()


@functools.lru_cache(maxsize=1)
def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default(size=18)


def compose(images: list[np.ndarray], sheet: Sheet, number: int) -> tuple[Image.Image, list[dict]]:
    """Tile a sheet's panels on white, tag each ``<number><letter>`` in its top-left corner, and
    return the image with each panel's tag and pixel box."""
    size = images[0].shape[0]
    W = sheet.cols * size + (sheet.cols - 1) * GUTTER_PX
    H = sheet.rows * size + (sheet.rows - 1) * GUTTER_PX
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    boxes = []
    for k, img in enumerate(images):
        col, row = k % sheet.cols, k // sheet.cols
        x, y = col * (size + GUTTER_PX), row * (size + GUTTER_PX)
        canvas.paste(Image.fromarray(img), (x, y))
        tag = f"{number}{chr(ord('a') + k)}"
        l, t, r, b = draw.textbbox((x + 8, y + 6), tag, font=_font())
        draw.rectangle((l - 4, t - 3, r + 4, b + 3), fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((x + 8, y + 6), tag, fill=(0, 0, 0), font=_font())
        boxes.append({"tag": tag, "box": [x, y, x + size, y + size]})
    return canvas, boxes


def render_sheets(clip: Clip, sheets: list[Sheet], first_number: int = 1,
                  human_mesh: HumanMesh | None = None, markers: bool = True) -> list[tuple[Image.Image, list]]:
    """Render and compose every sheet; sheet ``k`` is image ``first_number + k``."""
    out = []
    with Scene(clip, human_mesh, markers) as scene:
        for k, sheet in enumerate(sheets):
            out.append(compose([scene.render(p) for p in sheet.panels], sheet, first_number + k))
    return out


def save_png(image: Image.Image, path: Path) -> None:
    """Deterministic PNG: no metadata chunks, fixed compression."""
    image.save(path, format="PNG", optimize=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stem", required=True, help="an x0 clip of the shipped directory")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--zones", nargs="*", default=[], help="zones that also get a floor inset")
    ap.add_argument("--strip", type=int, nargs="*", default=[], help="up to six frames for a time strip")
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ids.OUTPUT_ROOT / "renders")
    args = ap.parse_args(argv)

    start = time.time()
    try:
        clip = load_clip(args.stem)
        sheets = plan(clip, args.frame, args.zones, args.strip or None)
        rendered = render_sheets(clip, sheets, markers=not args.no_markers)
    except Exception as exc:  # noqa: BLE001 -- one line, non-zero exit
        print(f"FAILED {args.stem}@{args.frame}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, ((image, _), sheet) in enumerate(zip(rendered, sheets), 1):
        save_png(image, out / f"{args.stem}@{args.frame}_{k}_{sheet.kind}.png")
    print(f"render {RENDER_V}: {args.stem}@{args.frame}: {len(sheets)} sheets, "
          f"{sum(len(s.panels) for s in sheets)} panels in {time.time() - start:.1f} s -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
