# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared geometry for relating the MOYO pressure mat to the humanoid's bodies.

Loads the Tier-0 archive written by :mod:`extract_moyo_pressure` and answers the
one question every downstream product needs: *for a given frame, which collision
geoms are near the ground and how far is each mat cell from them, in XY?*

Box geoms matter disproportionately here -- the heel/toe/hand boxes are what
actually touch the floor -- so their XY footprint is handled exactly (project the
eight oriented corners, take the convex hull, signed distance to that polygon)
rather than by a bounding-circle approximation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_geometry import (  # noqa: E402
    _box_corners_world,
    geom_ground_distance,
    geom_to_world,
)

# Common-order SMPL body names (matches RobotState / MotionLib body ordering).
BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]
NUM_BODIES = len(BODY_NAMES)


# --------------------------------------------------------------------------- #
# Archive access
# --------------------------------------------------------------------------- #
def load_archive(path: str | Path) -> Dict:
    """Open a Tier-0 ``.npz`` and add cheap per-frame accessors.

    ``dense(t)`` rebuilds the (NY, NX) field for one frame; the COO frame index
    is stored ascending so the slice is a pair of ``searchsorted`` lookups.
    """
    z = np.load(path, allow_pickle=False)
    ny, nx = (int(v) for v in z["mat_shape"])
    p_frame = z["p_frame"]
    n = int(z["n_frames"])
    starts = np.searchsorted(p_frame, np.arange(n + 1))

    origin, ex, ey = z["mat_origin_xy"], z["mat_ex"], z["mat_ey"]
    cell = float(z["cell_size_m"])
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    centres = (
        origin[None, None, :]
        + ((jj + 0.5) * cell)[..., None] * ex
        + ((ii + 0.5) * cell)[..., None] * ey
    )

    def dense(t: int) -> np.ndarray:
        out = np.zeros((ny, nx), dtype=np.float32)
        a, b = starts[t], starts[t + 1]
        out[z["p_row"][a:b].astype(int), z["p_col"][a:b].astype(int)] = z["p_value"][a:b]
        return out

    def sparse(t: int) -> Tuple[np.ndarray, np.ndarray]:
        """(cell_xy (M,2), pressure (M,)) for one frame, no dense allocation."""
        a, b = starts[t], starts[t + 1]
        r, c = z["p_row"][a:b].astype(int), z["p_col"][a:b].astype(int)
        return centres[r, c], z["p_value"][a:b]

    return dict(
        n_frames=n, mat_shape=(ny, nx), cell_size_m=cell,
        cell_centres=centres, dense=dense, sparse=sparse,
        total_force_n=z["total_force_n"], cop_world=z["cop_world"],
        cop_mat=z["cop_mat"], coverage=z["coverage"], edge_frac=z["edge_frac"],
        subject_weight_n=float(z["subject_weight_n"]), fps=float(z["fps"]),
        mat_origin_xy=origin, mat_ex=ex, mat_ey=ey,
        proto_minus_vicon_xy=z["proto_minus_vicon_xy"],
    )


# --------------------------------------------------------------------------- #
# World geoms
# --------------------------------------------------------------------------- #
def world_geoms_for_clip(geoms: Dict[str, List[dict]], rbp: torch.Tensor,
                         rbr: torch.Tensor) -> List[Tuple[int, dict, np.ndarray]]:
    """Transform every collision geom to world for the whole clip, once.

    -> list of (body_index, batched world geom, ground gap (T,) numpy).
    """
    out = []
    for bi, name in enumerate(BODY_NAMES):
        for g in geoms.get(name, []):
            gw = geom_to_world(g, rbp[:, bi], rbr[:, bi])
            gap, _ = geom_ground_distance(gw)
            out.append((bi, gw, gap.numpy()))
    return out


def _slice_geom(gw: dict, t: int) -> dict:
    s = {"type": gw["type"]}
    for k, v in gw.items():
        if k == "type":
            continue
        s[k] = v[t : t + 1] if torch.is_tensor(v) and v.dim() > 1 else v
    return s


def near_ground_geoms(geoms, rbp, rbr, t: int, z_thresh: float = 0.06):
    """Convenience single-frame wrapper (recomputes; use the batched path in loops)."""
    out = []
    for bi, name in enumerate(BODY_NAMES):
        for g in geoms.get(name, []):
            gw = geom_to_world(g, rbp[t : t + 1, bi], rbr[t : t + 1, bi])
            gap, _ = geom_ground_distance(gw)
            if float(gap[0]) <= z_thresh:
                out.append((bi, gw))
    return out


# --------------------------------------------------------------------------- #
# XY distance to a geom footprint
# --------------------------------------------------------------------------- #
def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; returns CCW hull vertices of a small point set."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(p) < 3:
        return p

    def half(points):
        stack = []
        for q in points:
            while len(stack) >= 2:
                a, b = stack[-2], stack[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 1e-12:
                    break
                stack.pop()
            stack.append(q)
        return stack

    lower, upper = half(p), half(p[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def _dist_to_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed XY distance to a CCW convex polygon; negative inside."""
    n = len(poly)
    if n < 3:
        return np.linalg.norm(pts[:, None, :] - poly[None, :, :], axis=2).min(axis=1)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a  # (n, 2)
    ap = pts[:, None, :] - a[None, :, :]  # (M, n, 2)
    tt = np.clip(
        (ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1)[None], 1e-12), 0.0, 1.0
    )
    closest = a[None] + tt[..., None] * ab[None]
    d = np.linalg.norm(pts[:, None, :] - closest, axis=2).min(axis=1)
    cross = ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0]
    inside = (cross >= -1e-12).all(axis=1)
    return np.where(inside, -d, d)


def xy_distance_to_geom(g: dict, pts: np.ndarray) -> np.ndarray:
    """Signed XY distance from ``pts`` (M,2) to a single-frame world geom's
    footprint.  Negative inside.

    This is the geom's *whole* horizontal projection.  Which of several
    overlapping projections should actually receive load is decided by the
    caller's vertical-gap weighting, not here -- see the note below.

    .. note::
       An earlier version clipped the footprint axially to the part of the geom
       within a band of its own lowest point, to stop a shin capsule (knee to
       ankle) claiming the sensels under the foot in Chair Pose.  Measured
       head-to-head, that clipping was strictly worse than weighting each geom's
       claim by its ground gap: gap-weighting alone puts 100 % of Chair Pose's
       load on the feet *and* keeps the corpus' explainable load at 0.908, while
       adding the axial clip dropped it to 0.842 (and to 0.674 at a 2 cm band)
       with no anatomy benefit on any probe.  The clip was removed; do not
       reintroduce it without re-running that comparison.
    """
    if g["type"] == "sphere":
        c = g["center"][0, :2].numpy()
        return np.linalg.norm(pts - c, axis=1) - float(g["radius"])

    if g["type"] == "capsule":
        a = g["a"][0, :2].numpy().astype(np.float64)
        b = g["b"][0, :2].numpy().astype(np.float64)
        ab = b - a
        t = np.clip((pts - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) - float(g["radius"])

    if g["type"] == "box":
        corners = _box_corners_world(g)[0, :, :2].numpy().astype(np.float64)
        return _dist_to_polygon(_convex_hull_2d(corners), pts)

    raise ValueError(g["type"])
