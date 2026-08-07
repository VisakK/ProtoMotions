# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Balance and contact quantities derived from a ``record_contact_physics`` rollout.

Pure numpy (plus an optional torch import to read the MJCF collision primitives),
so the whole analysis can be re-run and iterated on without a GPU or IsaacLab.

The quantities here are the physically meaningful ones the raw log does not
contain directly:

* whole-body centre of mass, its velocity, and the extrapolated centre of mass
  (capture point) -- the balance-relevant point for a held pose,
* centre of pressure from the *actual* PhysX contact points, weighted by their
  normal forces,
* the support polygon (convex hull of the ground contact points) and the signed
  margin of COM / COP / XCoM to its boundary, with the bodies that own the hull
  vertices,
* two effective contact areas per body: the convex hull of the contact manifold
  PhysX actually solved (measured, but structurally degenerate for curved
  colliders), and the cross-section of the collision primitive inside a
  compliance band above the ground (a model, but continuous and comparable),
* per-pair impulse and friction-cone utilisation.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GRAVITY = 9.81

# Every actuated joint is three co-located hinges (``_x``/``_y``/``_z``).
# ``L/R_Thorax`` (the clavicle) sits in the arm panels: its torque is what the
# arm's load is carried through.
LIMB_GROUPS: dict[str, tuple[str, ...]] = {
    "left_arm": ("L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"),
    "right_arm": ("R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"),
    "left_leg": ("L_Hip", "L_Knee", "L_Ankle", "L_Toe"),
    "right_leg": ("R_Hip", "R_Knee", "R_Ankle", "R_Toe"),
    "torso_neck_head": ("Torso", "Spine", "Chest", "Neck", "Head"),
}
GROUP_TITLES = {
    "left_arm": "Left arm",
    "right_arm": "Right arm",
    "left_leg": "Left leg",
    "right_leg": "Right leg",
    "torso_neck_head": "Torso, neck and head",
}


def joint_of_dof(dof_name: str) -> str:
    """``L_Hip_x`` -> ``L_Hip``."""
    return re.sub(r"_[xyz]$", "", dof_name)


def group_dof_indices(dof_names: list[str]) -> dict[str, list[tuple[str, list[int]]]]:
    """group -> ordered ``(joint_name, dof_indices)``, asserting a full partition."""
    joints = [joint_of_dof(n) for n in dof_names]
    grouped: dict[str, list[tuple[str, list[int]]]] = {}
    assigned = 0
    for group, members in LIMB_GROUPS.items():
        entries = []
        for joint in members:
            idx = [i for i, j in enumerate(joints) if j == joint]
            if idx:
                entries.append((joint, idx))
                assigned += len(idx)
        grouped[group] = entries
    if assigned != len(dof_names):
        missing = sorted({j for j in joints} - {m for g in LIMB_GROUPS.values() for m in g})
        raise ValueError(f"limb groups do not partition the DOFs; unassigned joints: {missing}")
    return grouped


# --------------------------------------------------------------------------- #
# 2-D convex hull utilities (no scipy dependency, matching the repo's style)
# --------------------------------------------------------------------------- #
def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain; returns the hull vertices counter-clockwise.

    Fewer than three distinct points are returned as-is (a point or a segment).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    pts = np.unique(np.round(pts, 9), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def _half(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower = _half(pts)
    upper = _half(pts[::-1])
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if len(hull) >= 3 else pts


def polygon_area(hull: np.ndarray) -> float:
    """Shoelace area of a polygon; 0 for degenerate (point / segment) inputs."""
    hull = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-18:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def signed_margin(point: np.ndarray, hull: np.ndarray) -> float:
    """Distance from ``point`` to the hull boundary, positive inside.

    A degenerate hull (a single contact point, or a line of them) has no
    interior, so the margin is the negative distance to the point / segment --
    which is the physically right statement: there is no static support there.
    """
    p = np.asarray(point, dtype=np.float64).reshape(2)
    hull = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if len(hull) == 0 or not np.all(np.isfinite(p)):
        return float("nan")
    if len(hull) == 1:
        return -float(np.linalg.norm(p - hull[0]))
    if len(hull) == 2:
        return -_point_segment_distance(p, hull[0], hull[1])
    inside = True
    best = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < 0:
            inside = False
        best = min(best, _point_segment_distance(p, a, b))
    return float(best if inside else -best)


def hull_vertex_owners(points: np.ndarray, owners: np.ndarray, hull: np.ndarray) -> list[int]:
    """Owner ids of the points that ended up as hull vertices (order preserved)."""
    if len(hull) == 0 or len(points) == 0:
        return []
    out: list[int] = []
    for vertex in np.atleast_2d(hull):
        d = np.linalg.norm(points[:, :2] - vertex[None, :], axis=1)
        out.append(int(owners[int(np.argmin(d))]))
    return out


# --------------------------------------------------------------------------- #
# rollout container
# --------------------------------------------------------------------------- #
@dataclass
class Rollout:
    """A recorded rollout plus everything derived from it."""

    path: Path
    raw: dict
    motion_name: str
    body_names: list[str]
    dof_names: list[str]
    filter_names: list[str]
    masses: np.ndarray
    dt_phys: float
    dt_ctrl: float
    decimation: int
    mu: float
    mu_terrain: float
    mu_robot: float
    friction_combine_mode: str
    contact_offset: float

    # substep time series
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    com: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    com_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    com_acc: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    xcom: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    @property
    def num_substeps(self) -> int:
        return int(self.raw["net_force_w"].shape[0])


def combine_friction(a: float, b: float, mode: str) -> float:
    """PhysX's material combine rules, as configured on the terrain."""
    mode = str(mode).lower()
    if mode in ("min", "minimum"):
        return min(a, b)
    if mode in ("max", "maximum"):
        return max(a, b)
    if mode in ("multiply", "product"):
        return a * b
    return 0.5 * (a + b)  # "average" is PhysX's default and the repo's setting


def effective_friction(raw: dict) -> tuple[float, float, float, str]:
    """(mu_effective, mu_terrain, mu_robot, combine_mode).

    The terrain's ``static_friction`` on its own is not what the contact sees:
    PhysX combines it with the robot shapes' material, which nothing in the
    ProtoMotions config sets, so it keeps PhysX's 0.5 default.
    """
    mu_terrain = float(raw.get("static_friction", np.array(1.0)))
    mode = str(raw.get("friction_combine_mode", np.array("average")))
    material = np.asarray(raw.get("robot_material", np.zeros((0, 3))))
    mu_robot = float(np.median(material[:, 0])) if material.size else mu_terrain
    return combine_friction(mu_terrain, mu_robot, mode), mu_terrain, mu_robot, mode


def load_rollout(path: str | Path) -> Rollout:
    path = Path(path)
    if path.is_dir():
        path = path / "rollout.npz"
    z = np.load(path, allow_pickle=True)
    raw = {k: z[k] for k in z.files}
    masses = raw["body_masses"].astype(np.float64)
    mu, mu_terrain, mu_robot, combine_mode = effective_friction(raw)
    roll = Rollout(
        path=path,
        raw=raw,
        motion_name=str(raw["motion_name"]),
        body_names=[str(b) for b in raw["body_names"]],
        dof_names=[str(d) for d in raw["dof_names"]],
        filter_names=[str(f) for f in raw["filter_names"]],
        masses=masses,
        dt_phys=float(raw["dt_phys"]),
        dt_ctrl=float(raw["dt_ctrl"]),
        decimation=int(raw["decimation"]),
        mu=mu,
        mu_terrain=mu_terrain,
        mu_robot=mu_robot,
        friction_combine_mode=combine_mode,
        contact_offset=float(raw.get("contact_offset", np.array(0.02))),
    )
    n = roll.num_substeps
    roll.t = np.arange(n) * roll.dt_phys
    m = masses[None, :, None]
    total = masses.sum()
    roll.com = (raw["body_com_pos_w"] * m).sum(1) / total
    roll.com_vel = (raw["body_com_vel_w"] * m).sum(1) / total
    roll.com_acc = np.gradient(roll.com_vel, roll.dt_phys, axis=0) if n > 1 else np.zeros_like(roll.com_vel)
    return roll


# --------------------------------------------------------------------------- #
# contact-point bookkeeping
# --------------------------------------------------------------------------- #
def step_slices(step_ids: np.ndarray, num_steps: int) -> np.ndarray:
    """Start index of every step in a step-sorted flat record array (len n+1)."""
    counts = np.bincount(step_ids.astype(np.int64), minlength=num_steps)
    starts = np.zeros(num_steps + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return starts


@dataclass
class ContactSeries:
    """Everything needed for the balance figures, per physics substep."""

    cop: np.ndarray            # [T,2] force-weighted centre of pressure (NaN in flight)
    grf: np.ndarray            # [T,3] resultant ground normal force
    grf_tangential: np.ndarray  # [T,3] resultant ground friction force
    hulls: list[np.ndarray]    # support polygon vertices per step
    hull_owners: list[list[int]]  # body index owning each hull vertex
    margin_com: np.ndarray     # [T] signed distance COM_xy -> hull boundary
    margin_cop: np.ndarray     # [T]
    margin_xcom: np.ndarray    # [T]
    ground_contact: np.ndarray  # [T,B] bool, body has >=1 ground contact point
    n_points: np.ndarray       # [T,B] contact points per body against the ground
    area_manifold: np.ndarray  # [T,B] hull area of the body's ground contact manifold
    support_area: np.ndarray   # [T] area of the support polygon


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / max(np.linalg.norm(normal), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= max(np.linalg.norm(u), 1e-12)
    return u, np.cross(n, u)


def manifold_area(points: np.ndarray, normals: np.ndarray) -> float:
    """Convex-hull area of a contact manifold, in the plane of its mean normal."""
    if len(points) < 3:
        return 0.0
    u, v = _plane_basis(normals.mean(0))
    flat = np.stack([points @ u, points @ v], axis=1)
    return polygon_area(convex_hull_2d(flat))


def contact_series(roll: Rollout, ground_filter: int = 0) -> ContactSeries:
    raw = roll.raw
    n, num_bodies = roll.num_substeps, len(roll.body_names)
    cp_step, cp_body, cp_filter = raw["cp_step"], raw["cp_body"], raw["cp_filter"]
    cp_pos, cp_normal, cp_force = raw["cp_pos"], raw["cp_normal"], raw["cp_force"]
    ground = cp_filter == ground_filter

    g_step = cp_step[ground].astype(np.int64)
    order = np.argsort(g_step, kind="stable")
    g_step = g_step[order]
    g_body = cp_body[ground][order].astype(np.int64)
    g_pos = cp_pos[ground][order].astype(np.float64)
    g_normal = cp_normal[ground][order].astype(np.float64)
    g_force = cp_force[ground][order].astype(np.float64)
    starts = step_slices(g_step, n)

    fr_step, fr_filter = raw["fr_step"], raw["fr_filter"]
    fr_ground = fr_filter == ground_filter
    f_step = fr_step[fr_ground].astype(np.int64)
    f_order = np.argsort(f_step, kind="stable")
    f_step = f_step[f_order]
    f_force = raw["fr_force"][fr_ground][f_order].astype(np.float64)
    f_starts = step_slices(f_step, n)

    cop = np.full((n, 2), np.nan)
    grf = np.zeros((n, 3))
    grf_t = np.zeros((n, 3))
    hulls: list[np.ndarray] = []
    owners: list[list[int]] = []
    margin_com = np.full(n, np.nan)
    margin_cop = np.full(n, np.nan)
    margin_xcom = np.full(n, np.nan)
    in_contact = np.zeros((n, num_bodies), dtype=bool)
    n_points = np.zeros((n, num_bodies), dtype=np.int32)
    area_manifold = np.zeros((n, num_bodies))
    support_area = np.zeros(n)

    height = np.maximum(roll.com[:, 2], 1e-3)
    xcom = roll.com[:, :2] + roll.com_vel[:, :2] * np.sqrt(height / GRAVITY)[:, None]
    roll.xcom = xcom

    for t in range(n):
        lo, hi = starts[t], starts[t + 1]
        flo, fhi = f_starts[t], f_starts[t + 1]
        if fhi > flo:
            grf_t[t] = f_force[flo:fhi].sum(0)
        if hi <= lo:
            hulls.append(np.zeros((0, 2)))
            owners.append([])
            continue
        pos, force, normal, body = g_pos[lo:hi], g_force[lo:hi], g_normal[lo:hi], g_body[lo:hi]
        vec = force[:, None] * normal
        grf[t] = vec.sum(0)
        fz = vec[:, 2]
        total_fz = fz.sum()
        if total_fz > 1e-6:
            cop[t] = (pos[:, :2] * fz[:, None]).sum(0) / total_fz
        np.add.at(n_points[t], body, 1)
        in_contact[t, np.unique(body)] = True
        for b in np.unique(body):
            sel = body == b
            area_manifold[t, b] = manifold_area(pos[sel], normal[sel])
        hull = convex_hull_2d(pos[:, :2])
        hulls.append(hull)
        owners.append(hull_vertex_owners(pos, body, hull))
        support_area[t] = polygon_area(hull)
        margin_com[t] = signed_margin(roll.com[t, :2], hull)
        margin_xcom[t] = signed_margin(xcom[t], hull)
        if np.all(np.isfinite(cop[t])):
            margin_cop[t] = signed_margin(cop[t], hull)

    return ContactSeries(
        cop=cop,
        grf=grf,
        grf_tangential=grf_t,
        hulls=hulls,
        hull_owners=owners,
        margin_com=margin_com,
        margin_cop=margin_cop,
        margin_xcom=margin_xcom,
        ground_contact=in_contact,
        n_points=n_points,
        area_manifold=area_manifold,
        support_area=support_area,
    )


# --------------------------------------------------------------------------- #
# per-pair series
# --------------------------------------------------------------------------- #
@dataclass
class Pair:
    body: int
    filt: int
    body_name: str
    filter_name: str
    force: np.ndarray        # [T,3] normal force, world frame
    friction: np.ndarray     # [T,3] tangential force, world frame
    n_points: np.ndarray     # [T]
    area: np.ndarray         # [T] contact-manifold hull area
    centroid: np.ndarray     # [T,3] force-weighted contact-point centroid

    @property
    def is_ground(self) -> bool:
        return self.filter_name == "ground"

    @property
    def label(self) -> str:
        return f"{self.body_name} ↔ {self.filter_name}"

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.force, axis=-1)

    def impulse(self, dt: float) -> np.ndarray:
        return np.cumsum(self.magnitude) * dt

    def vertical_impulse(self, dt: float) -> np.ndarray:
        return np.cumsum(self.force[:, 2]) * dt

    def friction_impulse(self, dt: float) -> np.ndarray:
        return np.cumsum(np.linalg.norm(self.friction, axis=-1)) * dt

    def cone_utilisation(self, mu: float, floor_n: float = 5.0) -> np.ndarray:
        """|F_t| / (mu |F_n|); NaN where the normal force is too small to mean anything."""
        normal = self.magnitude
        tangential = np.linalg.norm(self.friction, axis=-1)
        out = np.full_like(normal, np.nan)
        ok = normal > floor_n
        out[ok] = tangential[ok] / (mu * normal[ok])
        return out


def active_pairs(roll: Rollout, force_threshold: float = 1.0,
                 min_duty_steps: int = 2, dedupe_symmetric: bool = True) -> list[Pair]:
    """Pairs that carry a real force at some point in the clip.

    Body-body contacts are reported twice by PhysX (once from each side, with
    opposite sign); the duplicate is dropped and the surviving copy keeps the
    lower body index as the "sensor" side.
    """
    raw = roll.raw
    pair_force = raw["pair_force_w"].astype(np.float64)
    n, num_bodies, num_filters, _ = pair_force.shape
    magnitude = np.linalg.norm(pair_force, axis=-1)
    hot = (magnitude > force_threshold).sum(0) >= min_duty_steps  # [B,F]

    body_index = {name: i for i, name in enumerate(roll.body_names)}
    friction = np.zeros_like(pair_force)
    np.add.at(
        friction,
        (raw["fr_step"].astype(np.int64), raw["fr_body"].astype(np.int64),
         raw["fr_filter"].astype(np.int64)),
        raw["fr_force"].astype(np.float64),
    )
    n_points = np.zeros((n, num_bodies, num_filters), dtype=np.int32)
    np.add.at(
        n_points,
        (raw["cp_step"].astype(np.int64), raw["cp_body"].astype(np.int64),
         raw["cp_filter"].astype(np.int64)),
        1,
    )

    cp_step = raw["cp_step"].astype(np.int64)
    cp_body = raw["cp_body"].astype(np.int64)
    cp_filter = raw["cp_filter"].astype(np.int64)
    cp_pos = raw["cp_pos"].astype(np.float64)
    cp_normal = raw["cp_normal"].astype(np.float64)
    cp_force = np.abs(raw["cp_force"].astype(np.float64))

    pairs: list[Pair] = []
    seen: set[tuple[int, int]] = set()
    for b in range(num_bodies):
        for f in range(num_filters):
            if not hot[b, f]:
                continue
            filter_name = roll.filter_names[f]
            if dedupe_symmetric and filter_name in body_index:
                other = body_index[filter_name]
                key = tuple(sorted((b, other)))
                if key in seen:
                    continue
                seen.add(key)
            sel = (cp_body == b) & (cp_filter == f)
            area = np.zeros(n)
            centroid = np.full((n, 3), np.nan)
            if sel.any():
                steps = cp_step[sel]
                pos, nrm, frc = cp_pos[sel], cp_normal[sel], cp_force[sel]
                order = np.argsort(steps, kind="stable")
                steps, pos, nrm, frc = steps[order], pos[order], nrm[order], frc[order]
                bounds = step_slices(steps, n)
                for t in range(n):
                    lo, hi = bounds[t], bounds[t + 1]
                    if hi <= lo:
                        continue
                    area[t] = manifold_area(pos[lo:hi], nrm[lo:hi])
                    w = frc[lo:hi]
                    total = w.sum()
                    centroid[t] = (
                        (pos[lo:hi] * w[:, None]).sum(0) / total
                        if total > 1e-9
                        else pos[lo:hi].mean(0)
                    )
            pairs.append(
                Pair(
                    body=b,
                    filt=f,
                    body_name=roll.body_names[b],
                    filter_name=filter_name,
                    force=pair_force[:, b, f],
                    friction=friction[:, b, f],
                    n_points=n_points[:, b, f],
                    area=area,
                    centroid=centroid,
                )
            )
    pairs.sort(key=lambda p: (not p.is_ground, -float(np.linalg.norm(p.force, axis=-1).max())))
    return pairs


# --------------------------------------------------------------------------- #
# geometric contact-area model
# --------------------------------------------------------------------------- #
def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` [...,N,3] by unit quaternion ``q`` [...,4] in xyzw order."""
    u = q[..., None, :3]
    w = q[..., None, 3:4]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def load_collision_geoms(mjcf_path: str | Path, body_names: list[str]) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from contact_geometry import parse_typed_geoms  # noqa: E402  (local module)

    return parse_typed_geoms(str(mjcf_path), body_names)


def _band_points_sphere(center: np.ndarray, radius: float, band_top: float) -> np.ndarray:
    """XY outline of a sphere's widest cross-section below ``band_top``."""
    if center[2] - radius > band_top:
        return np.zeros((0, 2))
    z = min(center[2], band_top)
    r = float(np.sqrt(max(radius**2 - (z - center[2]) ** 2, 0.0)))
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    return center[None, :2] + r * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _band_points_capsule(a: np.ndarray, b: np.ndarray, radius: float,
                         band_top: float, samples: int = 48) -> np.ndarray:
    s = np.linspace(0.0, 1.0, samples)[:, None]
    axis = a[None, :] * (1 - s) + b[None, :] * s
    out = [_band_points_sphere(p, radius, band_top) for p in axis]
    out = [o for o in out if len(o)]
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


def _band_points_box(corners: np.ndarray, band_top: float) -> np.ndarray:
    """Corners inside the band plus where the box's edges cross its top."""
    pts = [corners[corners[:, 2] <= band_top][:, :2]]
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    for i, j in edges:
        zi, zj = corners[i, 2], corners[j, 2]
        if (zi - band_top) * (zj - band_top) < 0:
            u = (band_top - zi) / (zj - zi)
            pts.append((corners[i] + u * (corners[j] - corners[i]))[None, :2])
    pts = [p for p in pts if len(p)]
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


_BOX_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
)


def geometric_band_area(roll: Rollout, geoms: dict, band: float = 0.005,
                        stride: int | None = None,
                        ground_z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional area of each body's collision primitive inside a band of
    height ``band`` above the ground plane.

    Rigid-body contact has no deformation, so there is no true contact area to
    measure; this answers instead "how much of the collider is within ``band``
    of the floor", which is continuous, comparable across bodies and independent
    of how many contact points the solver happened to generate.

    Returns ``(sample_indices, area[len(sample_indices), num_bodies])``.
    """
    stride = stride or roll.decimation
    idx = np.arange(0, roll.num_substeps, stride)
    pos = roll.raw["body_pos_w"][idx].astype(np.float64)
    quat_wxyz = roll.raw["body_quat_w"][idx].astype(np.float64)
    quat = np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., :1]], axis=-1)
    band_top = ground_z + band
    area = np.zeros((len(idx), len(roll.body_names)))

    for b, name in enumerate(roll.body_names):
        for geom in geoms.get(name, []):
            q, p = quat[:, b], pos[:, b]
            if geom["type"] == "sphere":
                centers = _quat_rotate_xyzw(q, geom["center"][None, None, :].astype(np.float64))[:, 0] + p
                for k, c in enumerate(centers):
                    if c[2] - geom["radius"] <= band_top:
                        area[k, b] += polygon_area(
                            convex_hull_2d(_band_points_sphere(c, geom["radius"], band_top))
                        )
            elif geom["type"] == "capsule":
                seg = geom["seg"].astype(np.float64)[None, :, :]
                world = _quat_rotate_xyzw(q, seg) + p[:, None, :]
                for k in range(len(idx)):
                    a, c = world[k, 0], world[k, 1]
                    if min(a[2], c[2]) - geom["radius"] <= band_top:
                        area[k, b] += polygon_area(
                            convex_hull_2d(
                                _band_points_capsule(a, c, geom["radius"], band_top)
                            )
                        )
            elif geom["type"] == "box":
                local = _BOX_SIGNS * geom["half"].astype(np.float64)[None, :]
                gq = np.asarray(geom["quat"], dtype=np.float64)
                local = _quat_rotate_xyzw(gq[None, :], local[None, ...])[0]
                local = local + np.asarray(geom["center"], dtype=np.float64)[None, :]
                world = _quat_rotate_xyzw(q, local[None, ...]) + p[:, None, :]
                for k in range(len(idx)):
                    corners = world[k]
                    if corners[:, 2].min() <= band_top:
                        area[k, b] += polygon_area(
                            convex_hull_2d(_band_points_box(corners, band_top))
                        )
    return idx, area


# --------------------------------------------------------------------------- #
@dataclass
class LeanSeries:
    """Is the body far enough over its hands to release the trailing foot?

    An arm balance is only reachable once the centre of mass sits inside the
    polygon the *hands alone* make.  While it does not, a trailing foot is
    load-bearing and the pose cannot be held -- so this, not the full support
    polygon, is the quantity that decides whether lift-off is available.
    """

    hands_down: np.ndarray      # [T] bool, at least one hand/wrist in contact
    hand_margin: np.ndarray     # [T] signed distance COM_xy -> hands-only hull (+ inside)
    xcom_margin: np.ndarray     # [T] same for the capture point
    lean: np.ndarray            # [T] COM offset along the feet->hands axis (+ = past the hands)
    foot_load: np.ndarray       # [T] share of vertical ground force carried by the feet
    hand_hulls: list[np.ndarray]


def lean_series(roll: Rollout, hand_names=("L_Hand", "R_Hand", "L_Wrist", "R_Wrist"),
                foot_names=("L_Ankle", "R_Ankle", "L_Toe", "R_Toe"),
                ground_filter: int = 0) -> LeanSeries:
    raw = roll.raw
    n = roll.num_substeps
    index = {name: i for i, name in enumerate(roll.body_names)}
    hands = {index[b] for b in hand_names if b in index}
    feet = {index[b] for b in foot_names if b in index}

    ground = raw["cp_filter"] == ground_filter
    step = raw["cp_step"][ground].astype(np.int64)
    order = np.argsort(step, kind="stable")
    step = step[order]
    body = raw["cp_body"][ground][order].astype(np.int64)
    pos = raw["cp_pos"][ground][order].astype(np.float64)
    force = raw["cp_force"][ground][order].astype(np.float64)
    normal = raw["cp_normal"][ground][order].astype(np.float64)
    starts = step_slices(step, n)

    hands_down = np.zeros(n, dtype=bool)
    hand_margin = np.full(n, np.nan)
    xcom_margin = np.full(n, np.nan)
    lean = np.full(n, np.nan)
    foot_load = np.full(n, np.nan)
    hulls: list[np.ndarray] = []

    for t in range(n):
        lo, hi = starts[t], starts[t + 1]
        if hi <= lo:
            hulls.append(np.zeros((0, 2)))
            continue
        b, p = body[lo:hi], pos[lo:hi]
        fz = force[lo:hi] * normal[lo:hi][:, 2]
        is_hand = np.isin(b, list(hands))
        is_foot = np.isin(b, list(feet))
        total = fz.sum()
        if total > 1e-6:
            foot_load[t] = float(fz[is_foot].sum() / total)
        if not is_hand.any():
            hulls.append(np.zeros((0, 2)))
            continue
        hands_down[t] = True
        hull = convex_hull_2d(p[is_hand][:, :2])
        hulls.append(hull)
        hand_margin[t] = signed_margin(roll.com[t, :2], hull)
        xcom_margin[t] = signed_margin(roll.xcom[t], hull)
        # "Forward" is whichever way the hands lie from the feet; with no foot
        # contact the body is already past them, so fall back to the COM's own
        # offset direction.
        hand_centre = p[is_hand][:, :2].mean(0)
        if is_foot.any():
            axis = hand_centre - p[is_foot][:, :2].mean(0)
            norm = np.linalg.norm(axis)
            if norm > 1e-6:
                lean[t] = float((roll.com[t, :2] - hand_centre) @ (axis / norm))
        else:
            lean[t] = float(np.linalg.norm(roll.com[t, :2] - hand_centre))

    return LeanSeries(
        hands_down=hands_down,
        hand_margin=hand_margin,
        xcom_margin=xcom_margin,
        lean=lean,
        foot_load=foot_load,
        hand_hulls=hulls,
    )


def force_balance(roll: Rollout, series: ContactSeries) -> dict[str, np.ndarray]:
    """Newton's second law on the whole body: sum(F_contact) + m*g = m*a_com.

    On a flat floor every contact normal is +z, so the vertical residual is a
    direct check that the recorded contact set is complete and that the motion
    is dynamically consistent.
    """
    mass = roll.total_mass
    weight = mass * GRAVITY
    measured = series.grf[:, 2]
    predicted = mass * (GRAVITY + roll.com_acc[:, 2])
    return {
        "grf_z": measured,
        "predicted_z": predicted,
        "residual_z": measured - predicted,
        "weight": np.full_like(measured, weight),
        "grf_over_weight": measured / weight,
    }
