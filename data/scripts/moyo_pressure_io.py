# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readers and geometry for the MOYO pressure-mat modality.

MOYO ships, per capture, a Noraxon myoPressure mat recording alongside the Vicon
markers used to build the yoga clips.  Three files make up one clip
(``../../moyo_toolkit/data/pressure/{train,val}/``):

``xml/<name>.xml``
    The full pressure field: **37 x 110 cells of 12.7 mm, in N/cm2, at 60 Hz**.
    Each frame stores only a cropped bounding box of loaded cells
    (``cell_begin`` + ``cell_count`` + whitespace-separated ``cells``), so the
    reader un-crops back to the full mat.  Parsed with ``iterparse`` -- the moyo
    toolkit uses BeautifulSoup, which is ~100x slower on these 2 MB files.

``single_csv/<name>.csv``
    Per-frame total ``Force (N)`` and centre of pressure (mm, mat-local), plus
    the sync ``time`` column that locates these rows inside the XML stream.

``pressure_mat_c3d/<name>.c3d``
    The **full 83-marker Vicon set** -- 73 body markers *and* the four
    ``SensorMat:M_1..4`` mat-corner markers, in one file, one frame.  This is
    what makes registration exact rather than approximate.

Verified numerically (see ``notes/Moyo_pressure_port.MD``):

* ``sum(P) * 1.6129 cm^2`` reproduces the CSV ``Force (N)`` to 0.04 %.
* COP recomputed from the grid reproduces the CSV COP to 0.1 mm.
* After ``frame_ids`` filtering, pressure frame *i* corresponds to c3d frame *i*
  and to ProtoMotions frame *i* (all 60 Hz).

Mat frame convention
--------------------
Mat-local ``+x`` runs along columns and ``+y`` along rows, origin at the corner
of cell ``(0, 0)``.  In the Vicon world these map to **-x** and **-y**
respectively.  The four corner markers form a rectangle that is the sensing area
plus exactly **one cell (12.7 mm) of bezel on every side** (measured 0.4942 x
1.4234 m vs a 0.4699 x 1.397 m sensing area; residual ~1 mm across 173 clips),
so :func:`mat_frame_from_markers` insets by one cell.  The moyo toolkit instead
stretches the marker rectangle onto the sensing area, a ~5 % x-scale error worth
up to 1.2 cm at the mat edge.

Requires ``c3d`` (pure Python, numpy-only): ``pip install --no-deps c3d``.
"""

from __future__ import annotations

import csv
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Mat hardware constants (asserted against every clip by the readers below).
CELL_M = 0.0127  # sensel pitch, metres
MAT_NX = 37  # cells along mat-local +x (columns)
MAT_NY = 110  # cells along mat-local +y (rows)
CELL_AREA_CM2 = (CELL_M * 100.0) ** 2  # 1.6129 cm^2
PRESSURE_FPS = 60.0

# The subject's ground-reaction weight, read off single-foot balance poses which
# cannot lose load off the mat (Tree 694 N, Eagle 710 N).  Used to turn total mat
# force into a per-frame coverage fraction.  Note this is ~4 % below the 74 kg
# the smpl_yogi humanoid was built to.
SUBJECT_WEIGHT_N = 700.0

# Constant offset between the ProtoMotions clip frame and the Vicon lab frame.
# ``smpl_to_mimickit`` writes AMASS ``trans`` straight through as the pelvis
# position, but SMPL defines ``pelvis_world = J_0 + transl``; the subject's
# personalized origin-centred ``v_template`` puts J_0 about 0.34 m below centre.
# Measured +0.3412 m over 170 clips (the 12 mm spread is estimator noise, not
# per-clip variation: arm- and leg-marker estimates correlate -0.01).
# Refined corpus-wide by ``calibrate_pressure_alignment.py``.
DEFAULT_PROTO_MINUS_VICON_XY = np.array([0.0, 0.3412], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Pressure field (XML)
# --------------------------------------------------------------------------- #
@dataclass
class PressureField:
    """Un-cropped pressure field for a whole capture."""

    frames: np.ndarray  # (count, MAT_NY, MAT_NX) float32, N/cm^2
    fps: float
    count: int
    units: str
    cell_size_m: Tuple[float, float]
    mat_cells: Tuple[int, int]  # (nx, ny)

    def total_force_n(self) -> np.ndarray:
        """Per-frame total normal force, N."""
        return self.frames.sum(axis=(1, 2)) * CELL_AREA_CM2


def parse_pressure_xml(path: str) -> PressureField:
    """Parse a Noraxon ``myopressure`` XML into a dense (T, NY, NX) field."""
    nx = ny = None
    sx = sy = None
    fps = count = units = None
    frames = None
    t = 0
    in_data = False

    for ev, el in ET.iterparse(path, events=("start", "end")):
        if ev == "start":
            if el.tag == "data":
                in_data = True
            continue
        tag = el.tag
        # The header <cell_count> appears twice (measurement + clip) before
        # <data>; per-frame ones inside <data> are crop extents, not mat size.
        if tag == "cell_count" and not in_data and nx is None:
            x = el.find("x")
            if x is not None:
                nx, ny = int(x.text), int(el.find("y").text)
        elif tag == "cell_size" and sx is None:
            sx, sy = float(el.find("x").text), float(el.find("y").text)
        elif tag == "frequency" and fps is None:
            fps = float(el.text)
        elif tag == "count" and count is None:
            count = int(el.text)
        elif tag == "units" and units is None:
            units = el.text
        elif tag == "quant":
            if frames is None:
                if None in (nx, ny, count):
                    raise ValueError(f"{path}: malformed header")
                frames = np.zeros((count, ny, nx), dtype=np.float32)
            cbx = int(el.find("cell_begin/x").text)
            cby = int(el.find("cell_begin/y").text)
            ccx = int(el.find("cell_count/x").text)
            ccy = int(el.find("cell_count/y").text)
            txt = el.find("cells").text
            if ccx > 0 and ccy > 0 and txt is not None:
                vals = np.fromstring(txt.replace("\n", " "), sep=" ", dtype=np.float32)
                if vals.size != ccx * ccy:
                    raise ValueError(
                        f"{path}: frame {t} has {vals.size} values, expected {ccx * ccy}"
                    )
                # The crop is stored bottom-up relative to its own placement.
                frames[t, cby : cby + ccy, cbx : cbx + ccx] = np.flipud(
                    vals.reshape(ccy, ccx)
                )
            t += 1
            el.clear()

    if t != count:
        raise ValueError(f"{path}: {t} frames parsed, header says {count}")
    if (nx, ny) != (MAT_NX, MAT_NY):
        raise ValueError(f"{path}: unexpected mat size {(nx, ny)}")
    if abs(sx / 1000.0 - CELL_M) > 1e-6 or abs(sy / 1000.0 - CELL_M) > 1e-6:
        raise ValueError(f"{path}: unexpected cell size {(sx, sy)} mm")
    if units != "N/cm2":
        raise ValueError(f"{path}: unexpected units {units!r}")

    return PressureField(
        frames=frames,
        fps=fps,
        count=count,
        units=units,
        cell_size_m=(sx / 1000.0, sy / 1000.0),
        mat_cells=(nx, ny),
    )


# --------------------------------------------------------------------------- #
# Force / COP trace (CSV)
# --------------------------------------------------------------------------- #
@dataclass
class PressureTrace:
    """Per-frame scalars, already restricted to the recorded window."""

    force_n: np.ndarray  # (T,) float64
    cop_mat: np.ndarray  # (T, 2) float64, mat-local metres
    time_s: np.ndarray  # (T,) float64
    frame_ids: np.ndarray  # (T,) int, indices into PressureField.frames
    fps: float
    begin_time: float
    meta: Dict[str, str]


_COP_X = "Pressure, Raw Pressure-distribution-Pressure center-x (mm)"
_COP_Y = "Pressure, Raw Pressure-distribution-Pressure center-y (mm)"


def parse_pressure_csv(path: str) -> PressureTrace:
    """Parse the single-channel CSV export (force + COP + sync clock)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().split("\n")

    meta = dict(zip(next(csv.reader([lines[0]])), next(csv.reader([lines[1]]))))
    header = next(csv.reader([lines[3]]))
    rows = [r for r in csv.reader(lines[4:]) if r and r[0].strip()]

    cols: Dict[str, List[str]] = {h: [] for h in header}
    for r in rows:
        for h, v in zip(header, r):
            cols[h].append(v)

    def num(name: str) -> np.ndarray:
        return np.array(
            [float(v) if v not in ("", "nan", "NaN") else np.nan for v in cols[name]],
            dtype=np.float64,
        )

    fps = float(meta["frequency"])
    if fps != PRESSURE_FPS:
        raise ValueError(f"{path}: pressure fps {fps} != {PRESSURE_FPS}")
    time_s = num("time")
    frame_ids = np.rint(time_s * fps).astype(int)
    if frame_ids.shape[0] != frame_ids[-1] - frame_ids[0] + 1:
        raise ValueError(f"{path}: gaps in the pressure timestamps")

    return PressureTrace(
        force_n=num("Force (N)"),
        cop_mat=np.stack([num(_COP_X) / 1000.0, num(_COP_Y) / 1000.0], axis=1),
        time_s=time_s,
        frame_ids=frame_ids,
        fps=fps,
        begin_time=float(meta["begin_time"]),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Vicon markers (C3D)
# --------------------------------------------------------------------------- #
def read_marker_c3d(path: str) -> Tuple[Dict[str, int], np.ndarray]:
    """-> (label -> index, positions (T, n_markers, 3) in metres)."""
    import c3d  # local import: optional dependency, only data-prep needs it

    with open(path, "rb") as fh:
        reader = c3d.Reader(fh)
        labels = [l.strip() for l in reader.point_labels]
        pts = np.stack([f[1][:, :3] for f in reader.read_frames()])
    return {l: i for i, l in enumerate(labels)}, pts / 1000.0


@dataclass
class MatFrame:
    """Placement of the *sensing area* in some world frame.

    ``world_xy = origin + mat_x * ex + mat_y * ey`` for mat-local metres.
    """

    origin: np.ndarray  # (2,) corner of cell (0, 0)
    ex: np.ndarray  # (2,) unit, mat-local +x
    ey: np.ndarray  # (2,) unit, mat-local +y
    marker_span: Tuple[float, float]  # measured marker-rectangle size, m
    marker_z: float  # mean marker height above the lab floor, m

    def to_world(self, mat_xy: np.ndarray) -> np.ndarray:
        mat_xy = np.atleast_2d(mat_xy)
        return self.origin + np.outer(mat_xy[:, 0], self.ex) + np.outer(mat_xy[:, 1], self.ey)

    def cell_centres_world(self) -> np.ndarray:
        """(NY, NX, 2) world XY of every sensel centre."""
        jj, ii = np.meshgrid(np.arange(MAT_NX), np.arange(MAT_NY))
        return (
            self.origin[None, None, :]
            + ((jj + 0.5) * CELL_M)[..., None] * self.ex
            + ((ii + 0.5) * CELL_M)[..., None] * self.ey
        )

    def translated(self, delta_xy: np.ndarray) -> "MatFrame":
        return MatFrame(
            origin=self.origin + np.asarray(delta_xy, dtype=np.float64),
            ex=self.ex,
            ey=self.ey,
            marker_span=self.marker_span,
            marker_z=self.marker_z,
        )


def mat_frame_from_markers(
    marker_index: Dict[str, int], points: np.ndarray, inset: float = CELL_M
) -> MatFrame:
    """Locate the sensing area from the four ``SensorMat`` corner markers.

    The corners are identified by quadrant sign exactly as the moyo toolkit does
    (the lab origin sits inside the mat outline).  Positions are the per-clip
    median over frames -- the mat is static within a clip (sigma <= 0.7 mm) but
    *moves between clips* (sigma 20 mm), so this must not be hoisted to a global.
    """
    keys = [f"SensorMat:M_{k}" for k in (1, 2, 3, 4)]
    missing = [k for k in keys if k not in marker_index]
    if missing:
        raise KeyError(f"missing mat markers: {missing}")

    corners = np.stack(
        [np.nanmedian(points[:, marker_index[k]], axis=0) for k in keys]
    )  # (4, 3)
    if not np.isfinite(corners).all():
        raise ValueError("mat markers contain non-finite values")

    xy = corners[:, :2]
    quad = [
        xy[(xy[:, 0] > 0) & (xy[:, 1] > 0)],  # bottom-left  (mat-local origin side)
        xy[(xy[:, 0] > 0) & (xy[:, 1] < 0)],  # top-left
        xy[(xy[:, 0] < 0) & (xy[:, 1] < 0)],  # top-right
        xy[(xy[:, 0] < 0) & (xy[:, 1] > 0)],  # bottom-right
    ]
    if any(len(q) != 1 for q in quad):
        raise ValueError(f"mat corners do not fall one per quadrant: {[len(q) for q in quad]}")
    bl, tl, tr, br = (q[0] for q in quad)

    ex = ((br - bl) + (tr - tl)) / 2.0  # mat-local +x  (world -x)
    ey = ((tl - bl) + (tr - br)) / 2.0  # mat-local +y  (world -y)
    span = (float(np.linalg.norm(ex)), float(np.linalg.norm(ey)))
    ex = ex / span[0]
    ey = ey / span[1]

    return MatFrame(
        origin=bl + ex * inset + ey * inset,
        ex=ex,
        ey=ey,
        marker_span=span,
        marker_z=float(np.nanmean(corners[:, 2])),
    )


# --------------------------------------------------------------------------- #
# Clip name mapping
# --------------------------------------------------------------------------- #
# ProtoMotions clips drop the subject infix and the parenthesised aliases:
#   220923_Crane_Crow_Pose_or_Bakasana_-a
#     <-> 220923_yogi_body_hands_03596_Crane_(Crow)_Pose_or_Bakasana_-a
# Fuzzy matching is NOT safe here: difflib happily pairs Gomukhasana "-c" with
# "-d" and the hand-trimmed "Bakasana_hold" subclip with "-b".  Match on a
# punctuation-normalised key only, then gate on exact frame-count equality.
_MOYO_INFIX = "yogi_body_hands_03596"
_SESSIONS = ("220923", "220926")


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def proto_to_moyo_key(proto_stem: str) -> Optional[str]:
    """``220923_Tree_Pose_-a`` -> normalised MOYO key, or None if not a MOYO clip."""
    if "_" not in proto_stem:
        return None
    session, rest = proto_stem.split("_", 1)
    if session not in _SESSIONS:
        return None
    return _normalise(f"{session}_{_MOYO_INFIX}_{rest}")


def build_moyo_index(pressure_root: str) -> Dict[str, Tuple[str, str]]:
    """normalised key -> (moyo stem, split) for every clip with all three files."""
    import glob

    index: Dict[str, Tuple[str, str]] = {}
    for xml_path in glob.glob(os.path.join(pressure_root, "*", "xml", "*.xml")):
        stem = os.path.basename(xml_path)[:-4]
        split = xml_path.split(os.sep)[-3]
        csv_path = os.path.join(pressure_root, split, "single_csv", stem + ".csv")
        c3d_path = os.path.join(pressure_root, split, "pressure_mat_c3d", stem + ".c3d")
        if os.path.exists(csv_path) and os.path.exists(c3d_path):
            index[_normalise(stem)] = (stem, split)
    return index


def moyo_paths(pressure_root: str, stem: str, split: str) -> Dict[str, str]:
    return {
        "xml": os.path.join(pressure_root, split, "xml", stem + ".xml"),
        "csv": os.path.join(pressure_root, split, "single_csv", stem + ".csv"),
        "c3d": os.path.join(pressure_root, split, "pressure_mat_c3d", stem + ".c3d"),
    }
