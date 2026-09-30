# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-constrained retarget (BUILD_PLAN Step 8): reference motions that realise the human's contacts
on the training plant.

Why the shipped references float
--------------------------------
The shipped clips copy the performer's joint rotations onto the avatar: every avatar body's world rotation
is its SMPL-X joint's times a fixed axis permutation (``AXES``; spread 0.2-2 deg p50 over a clip). The
skeletons differ. Relative to the pelvis the avatar's ankles sit 5.9 cm higher than the performer's
(its legs are shorter) and its wrists 4.5 cm further out (its arms longer), so a pose with hands and a
foot on the floor cannot put both down with the performer's angles: Standing Split -a's standing foot
floats 11.6 cm with the hands pinned by the per-frame grounding. The foot boxes are tilted against the
performer's sole (about 10 deg of pitch plus a clip-dependent roll on flat feet), and 285 of 303
exemplars ask for joint positions outside the plant's hard limits (``plant.py``).

What is solved
--------------
Per clip, one least-squares problem over every frame's 75 variables (root offset, root rotation
perturbation, the 69 exp-map joint coordinates, the plant's own coordinates), with the plant's joint box
and an edit budget as bounds (``bounds``) and these residuals (``residuals``):

* **Supports.** ``support_plan`` reads, frame by frame, what the performer's mesh (capture store v3) does
  under every candidate point of the avatar (box corners, capsule ends, sphere bottoms; ``correspondence``
  maps each to the performer's surface facing the same way). A zone the performer touches puts its resting
  face on the floor at ``CLEARANCE_M``: a box rests **flat** on the face that points down in the shipped
  pose when the performer's surface under its four corners is flat (spread <= ``FLAT_SPREAD_M``: the MoSh
  heel reads 2-3 cm under a flat foot), else on the corners within ``EDGE_BAND_M`` of the lowest (toes, an
  edge). Zones the performer holds off the floor stay above ``OFF_FLOOR_M``; nothing goes below
  ``FLOOR_MIN_M``. Every term fades in and out over ``RAMP`` frames and ignores runs shorter than
  ``MIN_RUN``: a stiff term switched on in one frame made the edit jump (+115 m/s^2 of body acceleration).
* **The head** is left alone while inverted (``head_inverted``): its collider is a 9.5 cm sphere on the
  head joint, about 20 cm short of the crown (README §3.6), so a headstand's crown cannot be realised;
  those frames are reported (``head_blocked``). The back of the head and the chin are realised.
* **Body-body contacts** the labels configure and the informed reviewer calls critical are closed to
  ``PAIR_TARGET_M`` over the performer's contact runs, on the smallest gap among the two zones' bodies in
  the current pose; a pair farther than ``PAIR_REACH_M`` in the reference is the avatar's geometry and is
  reported (``geometry_incompatible``), not forced.
* **No self-penetration.** No body pair the plant collides (``plant_pairs``: every pair but a parent and
  its child) overlaps by more than ``PEN_TOL_M``, re-evaluated on every frame at every step; a shipped
  overlap is removed, not kept. The first pilot guarded only the contact labels' zone pairs, which leave
  out adjacent zones, and closing Peacock's forearms on its belly drove both upper arms 11.6 cm into the
  chest box; a guard fixed on the frames near the reference's own contacts let the overlap move to the
  frames it did not cover.
* **Balance.** During the labels' hold windows the plant's COM lies ``BALANCE_MARGIN_M`` inside the hull of
  the targeted supports: flattening the box hands rotated Crow's support polygon 2 cm off its COM.
* **The edit stays small and smooth**: joint coordinates, the root, hands/feet/head orientation and every
  body's position anchored to the shipped ones, and the correction's second difference penalised. A
  solution that makes a body jerk (``spike_frames``: > ``SPIKE_ACC`` on a frame the reference does not) is
  solved again with the pair and penetration terms brought in by continuation under a stiffer smoothness
  (``solve_gently``), and the smoother of the two is kept: solved at full weight at once, a limb the
  reference buries in another leaves on each frame by its own nearest side.

The solver is Levenberg-Marquardt with analytic Jacobians (``point_jacobian``: exp-map chains,
``right_jacobian``; verified against finite differences to 1e-9 relative) and one banded solve per step
(``normal_equations``, bandwidth 150), the box enforced by projection with the bound-active coordinates
frozen. L-BFGS-B on the same objective did not converge (a 2 mm scale is stiff), and without the anchor
and the budget the solve slid a clip 35 cm sideways (Warrior II), rotated one 49 deg (Plow) and flipped a
hip 136 deg (Bridge) to buy a few millimetres of support.

The output is every ``.motion`` field regenerated from the solved coordinates (``regenerate``) with a
per-frame lineage file, under ``output/reference_curation/retarget/<retarget_id>/``, and the records under
``data/reference_curation/retarget/<retarget_id>/``.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.retarget --pilot
    ... --all --workers 8
"""

from __future__ import annotations

import collections
import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from contact_geometry import _pt_seg, _seg_seg, parse_typed_geoms
from extract_contact_configs import ZONES, ZONE_ORDER, mjcf_body_names
from reference_curation import human_mesh as hm, ids

MODULE = "reference_curation.retarget"
SCHEMA_VERSION = 1
RETARGET_VERSION = "v1"
F64 = torch.float64

ZI = {z: i for i, z in enumerate(ZONE_ORDER)}
# The SMPL-X joint behind every avatar body: the correspondence origin. The avatar's hand body sits at
# SMPL's hand joint, which SMPL-X replaces with the fingers; its origin is the mean of four finger bases.
BODY_JOINT = {"Pelvis": 0, "L_Hip": 1, "L_Knee": 4, "L_Ankle": 7, "L_Toe": 10, "R_Hip": 2, "R_Knee": 5,
              "R_Ankle": 8, "R_Toe": 11, "Torso": 3, "Spine": 6, "Chest": 9, "Neck": 12, "Head": 15,
              "L_Thorax": 13, "L_Shoulder": 16, "L_Elbow": 18, "L_Wrist": 20, "R_Thorax": 14,
              "R_Shoulder": 17, "R_Elbow": 19, "R_Wrist": 21}
FINGER_BASES = {"L_Hand": (25, 28, 31, 34), "R_Hand": (40, 43, 46, 49)}
JOINT_BODY = {**{j: b for b, j in BODY_JOINT.items()}, 22: "Head", 23: "Head", 24: "Head",
              **{j: "L_Hand" for j in range(25, 40)}, **{j: "R_Hand" for j in range(40, 55)}}
# Avatar local axes -> SMPL-X canonical axes: ``AXES @ l_avatar = l_smplx``. Measured: every avatar body's
# world rotation is its SMPL-X joint's global rotation times AXES (spread 0.2-2 deg p50 over a clip), so a
# human vertex expressed in the avatar's body frame is ``AXES.T @ (v - joint)``, constant up to skinning.
AXES = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
CORNER_NEIGHBOURS = 8          # human vertices read under a box corner


# --------------------------------------------------------------------------- #
# The plant's kinematics, in its own joint coordinates
# --------------------------------------------------------------------------- #
@dataclass
class Skeleton:
    """The MJCF skeleton in COMMON order (the ``.motion`` body order) with the plant's joint limits.

    PhysX's joint coordinates of a 3-dof joint are the exponential map of its local rotation (measured:
    ``dof_pos`` of 11 recorded rollouts equals ``rotvec(R_parent^T R_child)`` to 0.0), and its limits act
    on those coordinates as a box: over 29k rollout frames no coordinate passed its MJCF range by more
    than 1.0 deg, even with the actuator at its torque limit against it. ``lower``/``upper`` are that box.
    """

    names: list
    parents: list
    offsets: torch.Tensor          # [B, 3] body position in its parent's frame
    lower: torch.Tensor            # [69] rad, exp-map box
    upper: torch.Tensor
    geoms: dict                    # body -> [typed geom], body-local
    cand_body: np.ndarray          # [K] body index of every candidate floor point
    cand_local: np.ndarray         # [K, 3] body-local point (a box corner, a capsule end, a sphere centre)
    cand_radius: np.ndarray        # [K] its surface is this far below it (0 for a corner)
    cand_zone: np.ndarray          # [K] zone index
    cand_kind: list                # [K] "corner" | "cap" | "sphere"
    zone_cands: list = field(default_factory=list)   # per zone, its candidate indices

    @property
    def num_bodies(self) -> int:
        return len(self.names)


BODY_ZONE = {b: z for z, bodies in ZONES.items() for b in bodies}


@functools.lru_cache(maxsize=1)
def skeleton() -> Skeleton:
    from protomotions.components.pose_lib import extract_kinematic_info

    kin = extract_kinematic_info(str(ids.MJCF))
    names = list(kin.body_names)
    if not torch.allclose(kin.local_rot_ref_mat.double(), torch.eye(3, dtype=F64).expand(len(names), 3, 3)):
        raise ValueError("the MJCF has rotated bodies; the FK here assumes identity rest orientations")
    geoms = parse_typed_geoms(str(ids.MJCF), mjcf_body_names(str(ids.MJCF)))
    body, local, radius, zone, kind = [], [], [], [], []
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    for i, b in enumerate(names):
        for g in geoms[b]:
            if g["type"] == "box":
                from scipy.spatial.transform import Rotation

                rg = Rotation.from_quat(g["quat"]).as_matrix()
                pts, r, k = g["center"] + (rg @ (signs * g["half"]).T).T, 0.0, "corner"
            elif g["type"] == "capsule":
                pts, r, k = np.asarray(g["seg"], float), float(g["radius"]), "cap"
            else:
                pts, r, k = np.asarray(g["center"], float)[None], float(g["radius"]), "sphere"
            for p in pts:
                body.append(i)
                local.append(p)
                radius.append(r)
                zone.append(ZI[BODY_ZONE[b]])
                kind.append(k)
    sk = Skeleton(names, list(kin.parent_indices), kin.local_pos.to(F64), kin.dof_limits_lower.to(F64),
                  kin.dof_limits_upper.to(F64), geoms, np.array(body), np.array(local, float),
                  np.array(radius, float), np.array(zone), kind)
    sk.zone_cands = [np.nonzero(sk.cand_zone == zi)[0] for zi in range(len(ZONE_ORDER))]
    return sk


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] rotation vector -> [..., 3, 3], smooth through zero."""
    th2 = (v * v).sum(-1)[..., None, None]
    th = torch.sqrt(th2.clamp_min(1e-30))
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6 + th2 * th2 / 120, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24 + th2 * th2 / 720, (1 - torch.cos(th)) / th2.clamp_min(1e-30))
    k = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    k[..., 1, 0], k[..., 2, 0], k[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return torch.eye(3, dtype=v.dtype, device=v.device) + a * k + b * (k @ k)


def nearest_representative(dof: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """``[T, 69]``: each joint's rotation vector ``v``, or its other representative ``v - 2 pi v / |v|`` (the
    same rotation, turned the other way round) when that one, clipped into the box, lands closer to the
    rotation. A lotus knee folded past 180 deg is stored as a 160 deg turn the other way: clipping that
    coordinate into the knee's range straightens the leg, clipping the other keeps it folded (Cockerel -b's
    right knee, 914 frames, 143 deg closer; the only joint of the corpus where the choice matters)."""
    v = dof.reshape(dof.shape[0], -1, 3)
    alt = v - 2 * math.pi * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    lo, hi = lower.reshape(-1, 3), upper.reshape(-1, 3)
    rv = so3_exp(v)

    def angle(c):
        m = so3_exp(torch.maximum(torch.minimum(c, hi), lo)).transpose(-1, -2) @ rv
        return torch.arccos(((m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1) / 2).clamp(-1.0, 1.0))

    return torch.where((angle(alt) < angle(v))[..., None], alt, v).reshape(dof.shape)


def fk(sk: Skeleton, root_pos: torch.Tensor, root_rot: torch.Tensor, dof: torch.Tensor):
    """``(pos [T,B,3], rot [T,B,3,3])`` from the root and the exp-map joint coordinates ``dof [T,69]``:
    ``pose_lib.extract_transforms_from_qpos(..., qpos_is_exp_map_on_3dof_joints=True)`` then
    ``compute_forward_kinematics_from_transforms``, differentiably (measured on the shipped clips: the
    stored ``dof_pos`` reproduce ``rigid_body_pos`` to 3e-7 m and ``rigid_body_rot`` to 1e-5)."""
    local = so3_exp(dof.reshape(dof.shape[0], -1, 3))
    rots, poss = [root_rot], [root_pos]
    for i in range(1, sk.num_bodies):
        p = sk.parents[i]
        rots.append(rots[p] @ local[:, i - 1])
        poss.append(poss[p] + rots[p] @ sk.offsets[i])
    return torch.stack(poss, 1), torch.stack(rots, 1)


def candidate_points(sk: Skeleton, pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """``[T, K, 3]`` world position of every candidate floor point; its surface height is
    ``z - cand_radius``."""
    body = torch.as_tensor(sk.cand_body)
    local = torch.as_tensor(sk.cand_local, dtype=pos.dtype)
    return pos[:, body] + (rot[:, body] @ local[..., None]).squeeze(-1)


def candidate_heights(sk: Skeleton, pts: torch.Tensor) -> torch.Tensor:
    return pts[..., 2] - torch.as_tensor(sk.cand_radius, dtype=pts.dtype)


# --------------------------------------------------------------------------- #
# The human under every candidate point
# --------------------------------------------------------------------------- #
def _origin(joints: np.ndarray, body: str) -> np.ndarray:
    if body in FINGER_BASES:
        return joints[list(FINGER_BASES[body])].mean(0)
    return joints[BODY_JOINT[body]]


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``[V, 3]`` area-weighted unit vertex normals."""
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    out = np.zeros_like(verts)
    for i in range(3):
        np.add.at(out, faces[:, i], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


FACE_COS = math.cos(math.radians(50.0))   # a vertex lies on a box face whose normal is within 50 deg of its own
BOX_FACES = tuple((a, s) for a in range(3) for s in (-1, 1))


@dataclass
class Correspondence:
    """Which human vertices every candidate reads (``correspondence``). ``sets`` is a flat list of vertex
    sets; ``face_sets[box candidate block][face]`` names the 4 (candidate, set) pairs of each box face,
    ``point_sets[k]`` the set of capsule end or sphere candidate ``k``."""

    sets: list
    face_sets: dict       # body index -> {(axis, sign): [(candidate k, set index)] * 4}
    point_sets: dict      # candidate k -> set index


def correspondence(sk: Skeleton, mdl: hm.SMPLX, v_template: np.ndarray) -> Correspondence:
    """The human vertices every candidate point follows.

    Every vertex belongs to the avatar body of its dominant skinning joint (``JOINT_BODY``, the rule the
    zones follow) and is expressed, with its surface normal, in that body's frame: ``AXES.T (v - joint)``
    on the rest template (the avatar's rotations are the human's times ``AXES``).

    * **A box face** reads the human surface facing the same way: the body's vertices whose normal is
      within 50 deg of the face's outward normal (the sole for the bottom face, the back of the heel for
      the back face). That surface is mapped onto the face in its two tangent axes (its extent onto the
      face's: the human's foot is longer than the avatar's box, its toes twice as long), and each of the
      face's corners reads the ``CORNER_NEIGHBOURS`` surface vertices nearest to it there. A plain
      nearest-neighbour map of the whole cloud read the rounded back of the heel under a flat foot's heel
      corners and a sole vertex under a top corner (measured on Warrior II), so faces are matched first.
    * **A capsule end** reads the half of the cloud on its side of the segment's midpoint.
    * **A sphere** reads the whole cloud: its lowest point is the body's lowest point in any orientation.
    """
    from scipy.spatial.transform import Rotation

    joints = mdl.J_regressor @ v_template
    normals = vertex_normals(v_template, mdl.faces)
    body_of = np.array([JOINT_BODY[j] for j in mdl.part])
    sets, face_sets, point_sets = [], {}, {}
    for i, b in enumerate(sk.names):
        g = sk.geoms[b][0]
        ks = np.nonzero(sk.cand_body == i)[0]
        vids = np.nonzero(body_of == b)[0]
        u = (v_template[vids] - _origin(joints, b)) @ AXES
        n = normals[vids] @ AXES
        if g["type"] == "box":
            rg = Rotation.from_quat(g["quat"]).as_matrix()
            u, n = (u - g["center"]) @ rg, n @ rg                  # box axes, centred on the box
            half = np.asarray(g["half"], float)
            corners = (sk.cand_local[ks] - g["center"]) @ rg       # +-half
            face_sets[i] = {}
            for a, s in BOX_FACES:
                on_face = s * n[:, a] >= FACE_COS
                t = [j for j in range(3) if j != a]
                entries = []
                if on_face.sum() >= CORNER_NEIGHBOURS:
                    uf = u[on_face][:, t]
                    lo, hi = uf.min(0), uf.max(0)
                    mapped = -half[t] + (uf - lo) / np.maximum(hi - lo, 1e-9) * 2 * half[t]
                for k, c in zip(ks, corners):
                    if np.sign(c[a]) != s:
                        continue
                    if on_face.sum() >= CORNER_NEIGHBOURS:
                        d = np.linalg.norm(mapped - c[t], axis=1)
                        sets.append(vids[on_face][np.argsort(d)[:CORNER_NEIGHBOURS]])
                    else:   # no human surface faces this way (the wrist box's ends): the nearest vertices
                        sets.append(vids[np.argsort(np.linalg.norm(u - c, axis=1))[:CORNER_NEIGHBOURS]])
                    entries.append((int(k), len(sets) - 1))
                face_sets[i][(a, s)] = entries
        elif g["type"] == "capsule":
            a0, a1 = np.asarray(g["seg"], float)
            sp = (u - a0) @ (a1 - a0) / max(float((a1 - a0) @ (a1 - a0)), 1e-12)
            sp = (sp - sp.min()) / max(sp.max() - sp.min(), 1e-9)
            for k in ks:
                mine = sp <= 0.5 if np.allclose(sk.cand_local[k], a0) else sp > 0.5
                sets.append(vids[mine])
                point_sets[int(k)] = len(sets) - 1
        else:
            for k in ks:
                sets.append(vids)
                point_sets[int(k)] = len(sets) - 1
    return Correspondence(sets, face_sets, point_sets)


def human_heights(human: hm.Human, corr: Correspondence) -> np.ndarray:
    """``[T, S]`` lowest height of every correspondence set, in the clip frame (m)."""
    out = np.empty((human.num_frames, len(corr.sets)))
    for s, verts in human.chunks():
        z = verts[..., 2]
        for j, vids in enumerate(corr.sets):
            out[s:s + len(verts), j] = z[:, vids].min(1)
    return out


# --------------------------------------------------------------------------- #
# Body-body gaps with their witnesses (rotation matrices; ``contact_geometry`` uses quaternions)
# --------------------------------------------------------------------------- #
def _world_geom(sk: Skeleton, body: int, pos: torch.Tensor, rot: torch.Tensor) -> dict:
    """Body ``body``'s geom at frames ``pos [N,3], rot [N,3,3]`` (its own rows)."""
    g = sk.geoms[sk.names[body]][0]
    dt = pos.dtype
    if g["type"] == "sphere":
        return {"type": "sphere", "c": pos + rot @ torch.as_tensor(g["center"], dtype=dt), "r": g["radius"]}
    if g["type"] == "capsule":
        seg = torch.as_tensor(np.asarray(g["seg"]), dtype=dt)
        return {"type": "capsule", "a": pos + rot @ seg[0], "b": pos + rot @ seg[1], "r": g["radius"]}
    from scipy.spatial.transform import Rotation

    rg = torch.as_tensor(Rotation.from_quat(g["quat"]).as_matrix(), dtype=dt)
    return {"type": "box", "c": pos + rot @ torch.as_tensor(g["center"], dtype=dt), "R": rot @ rg,
            "half": torch.as_tensor(np.asarray(g["half"]), dtype=dt)}


def _pt_box(p: torch.Tensor, box: dict):
    """``(sdf [N], surface witness [N,3], outward normal [N,3])`` of points ``p`` against boxes (one per row)."""
    pl = (box["R"].transpose(-1, -2) @ (p - box["c"])[..., None]).squeeze(-1)
    half = box["half"]
    cl = torch.maximum(torch.minimum(pl, half), -half)
    out = (pl - cl).norm(dim=-1)
    gap = half - pl.abs()
    inside = out <= 0.0
    ax = gap.argmin(-1)
    onehot = torch.nn.functional.one_hot(ax, 3).to(pl.dtype)
    face = pl * (1 - onehot) + onehot * torch.sign(pl) * half
    surf = torch.where(inside[..., None], face, cl)
    sdf = torch.where(inside, -gap.gather(-1, ax[..., None]).squeeze(-1), out)
    n_local = torch.where(inside[..., None], onehot * torch.sign(pl), (pl - cl) / out.clamp_min(1e-12)[..., None])
    return sdf, box["c"] + (box["R"] @ surf[..., None]).squeeze(-1), (box["R"] @ n_local[..., None]).squeeze(-1)


def _seg_box(a: torch.Tensor, b: torch.Tensor, box: dict, iters: int = 40):
    """The point of segment ``ab`` deepest toward (or nearest to) the box, by ternary search (the box SDF
    is convex along a line)."""
    lo = torch.zeros(a.shape[0], dtype=a.dtype)
    hi = torch.ones_like(lo)
    for _ in range(iters):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        g1 = _pt_box(a + m1[:, None] * (b - a), box)[0]
        g2 = _pt_box(a + m2[:, None] * (b - a), box)[0]
        left = g1 <= g2
        hi, lo = torch.where(left, m2, hi), torch.where(left, lo, m1)
    return a + ((lo + hi) / 2)[:, None] * (b - a)


_UNIT_BOX = torch.tensor([[x, y, z] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)
                          if (x, y, z) != (0.0, 0.0, 0.0)])


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n > 1e-12, v / n.clamp_min(1e-12), torch.tensor([0.0, 0.0, 1.0], dtype=v.dtype))


def _box_box_sat(ga: dict, gb: dict):
    """``(gap, pa, pb, u)`` of two boxes by the separating-axis test over their 15 axes (the 3 + 3 face
    normals and the 9 crossed edge directions): ``gap`` is the largest separation along any axis, which for
    intersecting boxes is exactly minus the penetration depth (the least translation that separates them).
    The witnesses make ``d gap = u . (d pb - d pa)`` exact: on a face axis, the other box's deepest vertex
    and its projection on the face plane; on an edge axis, the closest points of the two edge lines."""
    cA, cB = ga["c"], gb["c"]
    hA, hB = ga["half"].to(cA.dtype), gb["half"].to(cA.dtype)
    a, b = ga["R"].transpose(-1, -2), gb["R"].transpose(-1, -2)        # [N, 3, 3], rows are the axes
    N = cA.shape[0]
    d = cB - cA
    cross = torch.linalg.cross(a[:, :, None].expand(N, 3, 3, 3), b[:, None].expand(N, 3, 3, 3)).reshape(N, 9, 3)
    axes = torch.cat([a, b, cross], 1)                                  # [N, 15, 3]
    norm = axes.norm(dim=-1)
    n = axes / norm.clamp_min(1e-12)[..., None]
    n = n * torch.where((n * d[:, None]).sum(-1) >= 0, 1.0, -1.0).to(cA.dtype)[..., None]   # from A toward B
    na, nb = n @ a.transpose(-1, -2), n @ b.transpose(-1, -2)            # [N, 15, 3]: n . a_i, n . b_j
    gap_axis = (n * d[:, None]).sum(-1) - (hA * na.abs()).sum(-1) - (hB * nb.abs()).sum(-1)
    gap_axis = torch.where(norm > 1e-6, gap_axis, torch.full_like(gap_axis, -math.inf))   # parallel edges
    gap, k = gap_axis.max(1)
    rows = torch.arange(N)
    u = n[rows, k]
    sa = torch.where(na[rows, k] >= 0, 1.0, -1.0).to(cA.dtype)
    sb = torch.where(nb[rows, k] >= 0, 1.0, -1.0).to(cA.dtype)
    vA = cA + ((hA * sa)[..., None] * a).sum(1)                         # A's vertex farthest along u
    vB = cB - ((hB * sb)[..., None] * b).sum(1)                         # B's vertex farthest along -u
    i, j = (k - 6).clamp(min=0) // 3, (k - 6).clamp(min=0) % 3
    ai, bj = a[rows, i], b[rows, j]
    eA = vA - (hA[i] * sa[rows, i])[:, None] * ai                        # the edges' midpoints
    eB = vB + (hB[j] * sb[rows, j])[:, None] * bj
    w0, c = eA - eB, (ai * bj).sum(-1)
    den = (1 - c * c).clamp_min(1e-12)
    t = (c * (bj * w0).sum(-1) - (ai * w0).sum(-1)) / den
    v = ((bj * w0).sum(-1) - c * (ai * w0).sum(-1)) / den
    face_a, face_b = (k < 3)[:, None], ((k >= 3) & (k < 6))[:, None]
    pa = torch.where(face_a, vB - gap[:, None] * u, torch.where(face_b, vA, eA + t[:, None] * ai))
    pb = torch.where(face_a, vB, torch.where(face_b, vA + gap[:, None] * u, eB + v[:, None] * bj))
    return gap, pa, pb, u


@torch.no_grad()
def geom_gap(ga: dict, gb: dict):
    """``(gap [N], pa [N,3], pb [N,3], u [N,3])``: the signed surface gap of two world geoms (negative:
    overlap), a point fixed to each body where it is attained, and the unit direction with
    ``d gap = u . (d pb - d pa)`` for small motions of the two bodies (the envelope theorem)."""
    ta, tb = ga["type"], gb["type"]
    if ta == "box" and tb != "box":
        gap, pb, pa, u = geom_gap(gb, ga)
        return gap, pa, pb, -u
    if tb != "box":
        if ta == "sphere" and tb == "sphere":
            ca, cb = ga["c"], gb["c"]
        elif ta == "sphere":
            ca, cb = ga["c"], _pt_seg(ga["c"], gb["a"], gb["b"])
        elif tb == "sphere":
            ca, cb = _pt_seg(gb["c"], ga["a"], ga["b"]), gb["c"]
        else:
            ca, cb = _seg_seg(ga["a"], ga["b"], gb["a"], gb["b"])
        u = _unit(cb - ca)
        return (cb - ca).norm(dim=-1) - ga["r"] - gb["r"], ca, cb, u
    if ta != "box":        # a round geom against a box
        core = ga["c"] if ta == "sphere" else _seg_box(ga["a"], ga["b"], gb)
        sdf, w, n_out = _pt_box(core, gb)
        return sdf - ga["r"], core, w, -n_out
    # Box-box. Apart: 26 surface points of each box against the other's exact SDF. Intersecting: the
    # separating-axis depth, exact, where the sampled SDF was not: a corner inside the other box is closer to
    # its top face than to the side it came through, and on two feet side by side that gradient pushed up,
    # against the floor, where the separation is sideways (Crow -a's ankles stuck 2.2 cm deep).
    sat = _box_box_sat(ga, gb)
    best = None
    for src, dst, a_is_src in ((ga, gb, True), (gb, ga, False)):
        pts = src["c"][:, None] + (src["R"][:, None] @ (_UNIT_BOX.to(src["c"].dtype) * src["half"])[None, :, :, None]).squeeze(-1)
        n = pts.shape[1]
        sdf, w, n_out = _pt_box(pts.reshape(-1, 3), {k: (v.repeat_interleave(n, 0) if k in ("c", "R") else v)
                                                     for k, v in dst.items() if k != "type"})
        sdf, w, n_out = sdf.reshape(-1, n), w.reshape(-1, n, 3), n_out.reshape(-1, n, 3)
        k = sdf.argmin(1)
        rows = torch.arange(len(k))
        g, p, wd, no = sdf[rows, k], pts[rows, k], w[rows, k], n_out[rows, k]
        cand = (g, p, wd, -no) if a_is_src else (g, wd, p, no)
        if best is None:
            best = cand
        else:
            take = cand[0] < best[0]
            best = tuple(torch.where(take[:, None], c, b) if c.dim() > 1 else torch.where(take, c, b)
                         for c, b in zip(cand, best))
    inside = sat[0] < 0
    return tuple(torch.where(inside[:, None], x, y) if x.dim() > 1 else torch.where(inside, x, y)
                 for x, y in zip(sat, best))


def body_gaps(sk: Skeleton, pos: torch.Tensor, rot: torch.Tensor, frames, body_a, body_b, witnesses: bool = False):
    """Gaps of the body pairs ``(body_a[i], body_b[i])`` at ``frames[i]``, batched by body pair: ``[N]``,
    or with ``witnesses`` the tuple ``(gap, pa, pb, u)`` of ``geom_gap``."""
    frames, body_a, body_b = (np.asarray(x, int) for x in (frames, body_a, body_b))
    N = len(frames)
    out = [torch.empty(N, dtype=pos.dtype), torch.empty(N, 3, dtype=pos.dtype), torch.empty(N, 3, dtype=pos.dtype),
           torch.empty(N, 3, dtype=pos.dtype)]
    if N:
        order = np.lexsort((body_b, body_a))
        keys = np.stack([body_a[order], body_b[order]], 1)
        cut = np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
        for grp in np.split(order, cut):
            ba, bb, f = int(body_a[grp[0]]), int(body_b[grp[0]]), torch.as_tensor(frames[grp])
            res = geom_gap(_world_geom(sk, ba, pos[f, ba], rot[f, ba]), _world_geom(sk, bb, pos[f, bb], rot[f, bb]))
            idx = torch.as_tensor(grp)
            for o, r in zip(out, res):
                o[idx] = r
    return tuple(out) if witnesses else out[0]


def zone_pair_bodies(za: str, zb: str) -> list:
    sk = skeleton()
    return [(sk.names.index(a), sk.names.index(b)) for a in ZONES[za] for b in ZONES[zb]]


def zone_pair_gaps(sk: Skeleton, pos: torch.Tensor, rot: torch.Tensor, za: str, zb: str, frames=None):
    """``(gap [F], body_a [F], body_b [F])``: the closest body pair of two zones at every frame (all frames
    by default), with no gradient."""
    frames = np.arange(pos.shape[0]) if frames is None else np.asarray(frames, int)
    pairs = zone_pair_bodies(za, zb)
    with torch.no_grad():
        g = torch.stack([body_gaps(sk, pos, rot, frames, np.full(len(frames), a), np.full(len(frames), b))
                         for a, b in pairs], 1)
    k = g.argmin(1).numpy()
    arr = np.array(pairs)
    return g.min(1).values.numpy(), arr[k, 0], arr[k, 1]


# --------------------------------------------------------------------------- #
# What the human asks of every candidate point, frame by frame (pure)
# --------------------------------------------------------------------------- #
CLEARANCE_M = 0.005        # the corpus grounding convention: a resting surface sits 5 mm up
FLOOR_MIN_M = 0.002        # no collision surface below this
OFF_FLOOR_M = 0.025        # a zone the human holds off the floor stays this high: the 2 cm float threshold + 5 mm
FLAT_SPREAD_M = 0.035      # a box face rests flat when the human under its corners spans at most this
EDGE_BAND_M = 0.015        # otherwise it rests on the corners within this of the lowest (an edge, the toes)
RELEASE_M = 0.01           # a box stays down until the human under it rises this far above the touch height
MODE_FRAMES = 7            # the resting face is the mode over this many frames
HEAD_INVERTED_COS = 0.5    # the head's up axis within 60 deg of straight down: the crown is toward the floor


def hysteresis(h: np.ndarray, on: float, off: float) -> np.ndarray:
    """``[T]`` bool: down once ``h <= on``, up once ``h > off``, unchanged between (up at the start)."""
    out = np.zeros(len(h), bool)
    state = False
    for t, x in enumerate(h):
        state = True if x <= on else (False if x > off else state)
        out[t] = state
    return out


def mode_filter(x: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    out = x.copy()
    for t in range(len(x)):
        vals, counts = np.unique(x[max(0, t - half): t + half + 1], return_counts=True)
        out[t] = vals[counts.argmax()]
    return out


def box_face_normals_z(sk: Skeleton, rot: np.ndarray, body: int) -> np.ndarray:
    """``[T, 6]`` world z of the outward normal of each face (``BOX_FACES`` order) of ``body``'s box."""
    from scipy.spatial.transform import Rotation

    rg = Rotation.from_quat(sk.geoms[sk.names[body]][0]["quat"]).as_matrix()
    axes_z = (rot[:, body] @ rg)[:, 2, :]                 # [T, 3]: world z of the box's local axes
    return np.stack([s * axes_z[:, a] for a, s in BOX_FACES], 1)


def support_plan(sk: Skeleton, corr: Correspondence, heights: np.ndarray, state: np.ndarray,
                 head_down: np.ndarray, touch_m: float, rot0: np.ndarray) -> dict:
    """The floor terms of every candidate on every frame, from the human heights under it (``heights``,
    ``[T, S]``) and the zone's touch state (``state [T, Z]``, capture store v3: the mesh with the seam
    arbitration of the limb zones).

    * ``target [T, K]``: rest at ``CLEARANCE_M``. Only on a zone the human touches, and there on:
      - **a box**: the face it rests on is the one facing most nearly down in the shipped pose ``rot0``
        (mode over ``MODE_FRAMES``). The shipped rotations are the human's to within the box's tilt
        (10-26 deg), so the face is never ambiguous; the human heights under the thin toe box read its
        top and bottom faces alike, and choosing the face from them flipped toes. The face is down while
        its lowest corner reads at most ``touch_m`` (released above ``touch_m + RELEASE_M``). Down, it
        rests **flat** (all four corners) when its corners' readings span at most ``FLAT_SPREAD_M`` (the
        MoSh sole reads the heel 2-3 cm up under a flat foot, so a tighter band would tilt every standing
        foot), and otherwise on its corners within ``EDGE_BAND_M`` of the lowest: the toes, a foot's edge,
        the heel of a hand;
      - **a capsule end or a sphere**: while its reading is at most ``touch_m`` (same release);
      - never the head while it is inverted (``head_down``): the collider is a 9.5 cm sphere on the head
        joint, 20 cm short of the crown (README §3.6), so a headstand's crown cannot be realised.
    * ``off [T, K]``: stay at or above ``OFF_FLOOR_M``, on every frame the human holds the zone off
      the floor.
    * ``head_blocked [T]``: frames the head's contact is refused as a plant limit.
    """
    T, K = heights.shape[0], len(sk.cand_body)
    target = np.zeros((T, K), bool)
    off = np.zeros((T, K), bool)
    release = touch_m + RELEASE_M
    for zi, z in enumerate(ZONE_ORDER):
        on = state[:, zi] == 1
        off[np.ix_(state[:, zi] == 0, sk.zone_cands[zi])] = True
        if z == "HEAD":
            on = on & ~head_down
        bodies = sorted(set(sk.cand_body[sk.zone_cands[zi]]))
        for b in bodies:
            if b in corr.face_sets:
                faces = [(f, corr.face_sets[b][f]) for f in BOX_FACES]
                h = np.stack([heights[:, [s for _, s in ent]] for _, ent in faces], 1)     # [T, 6, 4]
                face = mode_filter(box_face_normals_z(sk, rot0, b).argmin(1), MODE_FRAMES)
                hf = h[np.arange(T), face]                                                  # [T, 4]
                lo = hf.min(1)
                down = hysteresis(lo, touch_m, release) & on
                flat = hf.max(1) - hf.min(1) <= FLAT_SPREAD_M
                for t in np.nonzero(down)[0]:
                    for j, (k, _) in enumerate(faces[face[t]][1]):
                        if flat[t] or hf[t, j] <= lo[t] + EDGE_BAND_M:
                            target[t, k] = True
            else:
                for k in np.nonzero(sk.cand_body == b)[0]:
                    down = hysteresis(heights[:, corr.point_sets[int(k)]], touch_m, release) & on
                    target[down, k] = True
    head_blocked = (state[:, ZI["HEAD"]] == 1) & head_down
    target, off = clean_runs(target), clean_runs(off)
    return {"target": target, "off": off, "head_blocked": head_blocked,
            "target_w": ramp_weights(target), "off_w": ramp_weights(off)}


MIN_RUN = 6                # frames: runs of a term shorter than this are dropped, gaps shorter are filled
RAMP = 10                  # frames: a term fades in and out over this many, centred on its onset and release


def clean_runs(mask: np.ndarray) -> np.ndarray:
    """``[T, K]``: per column, gaps shorter than ``MIN_RUN`` filled, then runs shorter than it dropped."""
    out = mask.copy()
    for k in np.nonzero(mask.any(0))[0]:
        col = out[:, k]
        for fill in (True, False):
            runs = _runs(col if not fill else ~col)
            for r0, r1 in runs:
                if r1 - r0 < MIN_RUN and (not fill or (r0 > 0 and r1 < len(col))):
                    col[r0:r1] = fill
        out[:, k] = col
    return out


def ramp_weights(mask: np.ndarray) -> np.ndarray:
    """``[T, K]`` in [0, 1]: 1 inside each run, a raised cosine over ``RAMP`` frames centred on every onset
    and release (a stiff term switched on in one frame made the correction jump: up to +115 m/s^2 of body
    acceleration in the pilot)."""
    T = mask.shape[0]
    half = RAMP / 2
    t = np.arange(T) + 0.5
    w = np.zeros(mask.shape)
    for k in np.nonzero(mask.any(0))[0]:
        col = np.zeros(T)
        for r0, r1 in _runs(mask[:, k]):
            rise = np.ones(T) if r0 == 0 else np.clip((t - (r0 - half)) / RAMP, 0, 1)
            fall = np.ones(T) if r1 == T else np.clip(((r1 + half) - t) / RAMP, 0, 1)
            col = np.maximum(col, 0.5 - 0.5 * np.cos(np.pi * np.minimum(rise, fall)))
        w[:, k] = col
    return w


def head_inverted(rot: np.ndarray, sk: Skeleton) -> np.ndarray:
    """``[T]`` bool: the head's up axis (its local z) points within 60 deg of straight down."""
    up = rot[:, sk.names.index("Head"), :, 2]
    return up[:, 2] <= -HEAD_INVERTED_COS


# --------------------------------------------------------------------------- #
# One clip's problem
# --------------------------------------------------------------------------- #
PAIR_TARGET_M = 0.003      # a pair the human closes rests at most 3 mm apart (HoldIK's closure target)
PAIR_REACH_M = 0.06        # ... if the reference's median gap over the human's contact run is at most this;
                           #     farther is the avatar's geometry (the head sphere, the pelvis), reported
PEN_TOL_M = 0.005          # no body pair the plant collides overlaps by more than this
LIMIT_MARGIN_RAD = math.radians(0.5)
END_BODIES = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe", "L_Wrist", "L_Hand", "R_Wrist", "R_Hand", "Head")
WEIGHTS = {"support": 10.0, "floor": 10.0, "off": 10.0, "pair": 10.0, "pen": 10.0, "balance": 10.0,
           "dof": 1.0, "root": 1.0, "ends": 1.0, "anchor": 1.0, "smooth": 1.0}
SCALES = {"support": 0.005, "floor": 0.001, "off": 0.005, "pair": 0.005, "pen": 0.001, "balance": 0.005, "dof": 0.2,
          "root_pos": 0.05, "root_rot": 0.1, "ends": 0.2, "anchor": 0.05, "acc_dof": 0.003, "acc_pos": 0.0005,
          "acc_rot": 0.003}
# The edit budget, as bounds: no joint coordinate moves further than this from its shipped value (clipped into
# the plant's box), and the root no further than these. With a 2 mm support scale the first pilot's solves
# found it cheaper to flip a hip 136 deg (Bridge -a) than to leave a support 2 cm up.
BALANCE_MARGIN_M = 0.015   # during a hold the whole-body COM lies this far inside its targeted supports' hull
BUDGET_JOINT_RAD = math.radians(45.0)
BUDGET_ROOT_M = 0.25
BUDGET_ROOT_RAD = math.radians(30.0)


@dataclass
class Problem:
    stem: str
    fps: int
    root_pos0: torch.Tensor        # [T, 3]
    root_rot0: torch.Tensor        # [T, 3, 3]
    dof0: torch.Tensor             # [T, 69] as shipped, each joint's nearest_representative (it may lie
                                   # outside the plant's box)
    pos0: torch.Tensor             # [T, B, 3] shipped
    rot0: torch.Tensor             # [T, B, 3, 3]
    plan: dict                     # support_plan
    close: dict                    # frames, body_a, body_b (numpy), per-item request names
    pen: dict                      # pairs [P, 2] (the plant's), floor (m): the self-penetration guard
    requests: list                 # the pair requests and what became of them
    lower: torch.Tensor            # [69] the box with LIMIT_MARGIN_RAD
    upper: torch.Tensor
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))

    @property
    def T(self) -> int:
        return self.dof0.shape[0]


def load_reference(stem: str, motion_dir=ids.SHIPPED_DIR) -> dict:
    from protomotions.utils.rotations import quaternion_to_matrix

    mot = torch.load(ids.motion_path(stem, motion_dir), map_location="cpu", weights_only=False)
    pos, rot = mot["rigid_body_pos"].to(F64), quaternion_to_matrix(mot["rigid_body_rot"].to(F64), w_last=True)
    return {"motion": mot, "fps": int(mot["fps"]), "pos": pos, "rot": rot, "dof": mot["dof_pos"].to(F64)}


def body_spheres(sk: Skeleton, pos: torch.Tensor, rot: torch.Tensor):
    """``(centre [T,B,3], radius [B])`` of a sphere around every body's geom."""
    cs, rs = [], []
    for i, b in enumerate(sk.names):
        g = sk.geoms[b][0]
        if g["type"] == "box":
            c, r = np.asarray(g["center"], float), float(np.linalg.norm(g["half"]))
        elif g["type"] == "capsule":
            seg = np.asarray(g["seg"], float)
            c, r = seg.mean(0), float(np.linalg.norm(seg[1] - seg[0]) / 2 + g["radius"])
        else:
            c, r = np.asarray(g["center"], float), float(g["radius"])
        cs.append(pos[:, i] + rot[:, i] @ torch.as_tensor(c, dtype=pos.dtype))
        rs.append(r)
    return torch.stack(cs, 1), np.array(rs)


@functools.lru_cache(maxsize=1)
def plant_pairs() -> np.ndarray:
    """``[P, 2]`` every body pair the training plant collides: all but a parent and its child (253 of the
    24 bodies' 276). The MJCF gives every geom ``contype = conaffinity = 1`` and excludes nothing, and
    PhysX's articulation self-collision filters only jointed links, so a chest box collides with an
    upper-arm capsule and a pelvis box with a spine box, pairs the contact labels' ``BODY_PAIRS`` leave out
    (adjacent zones)."""
    sk = skeleton()
    return np.array([(a, b) for a in range(sk.num_bodies) for b in range(a + 1, sk.num_bodies)
                     if sk.parents[b] != a and sk.parents[a] != b])


def near_body_pairs(sk: Skeleton, pos: torch.Tensor, rot: torch.Tensor, within: float, pairs=None):
    """``(frames, body_a, body_b, gap)`` of every body pair of ``pairs`` (default ``plant_pairs``) whose gap
    is below ``within`` on some frame (bounding spheres first, then the exact gap), ordered by frame."""
    pairs = plant_pairs() if pairs is None else np.asarray(pairs)
    a, b = pairs[:, 0], pairs[:, 1]
    with torch.no_grad():
        cen, rad = body_spheres(sk, pos, rot)
        bound = (cen[:, a] - cen[:, b]).norm(dim=-1).numpy() - rad[a] - rad[b]       # [T, P]
        f, k = np.nonzero(bound < within)
        gap = body_gaps(sk, pos, rot, f, a[k], b[k]).numpy() if len(f) else np.zeros(0)
    keep = gap < within
    return f[keep], a[k][keep], b[k][keep], gap[keep]


def hold_windows(stem: str, anns: dict, T: int) -> np.ndarray:
    """``[T]`` in [0, 1]: the labels' hold windows of the clip (``interval`` of its annotations), ramped
    like the floor terms: where the human holds a pose, the plant must be able to hold it too."""
    mask = np.zeros(T, bool)
    for hid, rows in anns.items():
        if hid.startswith(stem + "@") and rows:
            iv = rows[0]["interval"]
            mask[iv["start_frame"]:min(iv["end_frame_exclusive"], T)] = True
    return ramp_weights(mask[:, None])[:, 0]


@functools.lru_cache(maxsize=1)
def mass_model() -> tuple[np.ndarray, np.ndarray]:
    """``(mass [B], centre [B, 3])``: the plant's body masses and body-frame centres of mass (the MJCF the
    statics and the witness use, 74 kg)."""
    import static_hold_lp as S

    return S.M.body_mass[1:].copy(), S.M.body_ipos[1:].copy()


def pair_requests(stem: str, anns: dict, pair_state: np.ndarray) -> list:
    """The body-body contacts the retarget closes, per (hold, pair): the labels' configured pairs
    (``required_touch`` and carried) and the informed reviewer's critical ones (``review.role`` =
    ``required_touch`` on a stable human contact, BUILD_PLAN Step 6's caveat), each over the human's
    contact runs that overlap the hold's window."""
    col = {frozenset(n.split("+")): k for k, n in enumerate(hm.PAIR_NAMES)}
    out = []
    for hid, rows in anns.items():
        if not hid.startswith(stem + "@"):
            continue
        for a in rows:
            if a["kind"] != "pair":
                continue
            critical = ((a.get("review") or {}).get("role") == "required_touch"
                        and a["source_state"] == "observed_contact" and a["stable"])
            if not (a["in_configuration"] or critical):
                continue
            s, e = a["interval"]["start_frame"], a["interval"]["end_frame_exclusive"]
            on = pair_state[:, col[frozenset(a["zones"])]] == 1
            frames = np.zeros(len(on), bool)
            for r0, r1 in _runs(on):
                if r1 > s and r0 < e:
                    frames[r0:r1] = True
            w = ramp_weights(clean_runs(frames[:, None]))[:, 0]    # a flickering grip must not blip the arm
            out.append({"hold_id": hid, "contact": a["contact"], "zones": a["zones"],
                        "why": "configured" if a["in_configuration"] else "critical",
                        "target_role": a["target_role"], "frames": np.nonzero(w > 0)[0], "w": w[w > 0]})
    return out


def _runs(mask: np.ndarray) -> list:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def build_problem(stem: str, anns: dict, motion_dir=ids.SHIPPED_DIR, human=None, store=None) -> Problem:
    """Everything the solve needs for one x0 clip: the shipped reference, the human heights under every
    candidate, the support plan, the pair requests and the self-penetration guard."""
    from reference_curation import sources

    sk = skeleton()
    ref = load_reference(stem, motion_dir)
    store = sources.load(stem) if store is None else store
    if human is None:
        human, status, error = hm.load_human(stem)
        if human is None:
            raise ValueError(f"{stem}: no human mesh ({status}: {error})")
    corr = correspondence(sk, human.model, human.fit["v_template"])
    heights = human_heights(human, corr)
    T = ref["dof"].shape[0]
    if heights.shape[0] != T:
        raise ValueError(f"{stem}: the fit has {heights.shape[0]} frames, the reference {T}")
    touch = hm.load_calibration()["ground"]["touch_m"]
    rot0 = ref["rot"].numpy()
    plan = support_plan(sk, corr, heights, store["human_floor_state"], head_inverted(rot0, sk), touch, rot0)
    plan["balance_w"] = hold_windows(stem, anns, T)
    plan["heights"] = heights

    requests = pair_requests(stem, anns, store["human_pair_state"])
    items = {}                      # (contact, frame) -> weight: holds sharing a pair share its items
    for r in requests:
        if not len(r["frames"]):
            r["status"] = "no_human_contact"
            continue
        g, _, _ = zone_pair_gaps(sk, ref["pos"], ref["rot"], *r["zones"], r["frames"])
        r["gap_before_cm"] = round(100 * float(np.median(g)), 2)
        if np.median(g) > PAIR_REACH_M:
            r["status"] = "geometry_incompatible"
            continue
        r["status"] = "closed"
        for f, w in zip(r["frames"].tolist(), r["w"].tolist()):
            key = ("+".join(r["zones"]), f)
            items[key] = max(items.get(key, 0.0), w)
    # Every body pair of the two zones is an item; the residual closes the smallest gap among them, in the
    # current pose. A body pair fixed per frame from the reference switched between the ankle and the toe
    # box on a few frames of a hand-on-ankle grip and swung the arm 10 cm there and back (Bridge -a).
    rows = [(k, f, a, b) for k, (c, f) in enumerate(sorted(items)) for a, b in zone_pair_bodies(*c.split("+"))]
    keys = sorted(items)
    close = {"item": np.array([r[0] for r in rows], int), "frames": np.array([r[1] for r in rows], int),
             "body_a": np.array([r[2] for r in rows], int), "body_b": np.array([r[3] for r in rows], int),
             "w": np.array([items[k] for k in keys]), "contact": [c for c, _ in keys]}
    pen = {"pairs": plant_pairs(), "floor": -PEN_TOL_M}
    lower, upper = sk.lower + LIMIT_MARGIN_RAD, sk.upper - LIMIT_MARGIN_RAD
    return Problem(stem, ref["fps"], ref["pos"][:, 0].clone(), ref["rot"][:, 0].clone(),
                   nearest_representative(ref["dof"], lower, upper), ref["pos"], ref["rot"], plan, close, pen,
                   requests, lower, upper)


# --------------------------------------------------------------------------- #
# The objective: residuals, analytic Jacobians and a Gauss-Newton solve
# --------------------------------------------------------------------------- #
NV = 75    # per frame: root offset (3), root rotation perturbation (3), the 69 exp-map joint coordinates


def unpack(prob: Problem, x: torch.Tensor):
    """``x [T, 75]`` = root translation offset, root rotation perturbation (rotation vector, applied in the
    root's frame), exp-map joint coordinates -> ``(root_pos, root_rot, dof)``."""
    return prob.root_pos0 + x[:, :3], prob.root_rot0 @ so3_exp(x[:, 3:6]), x[:, 6:]


def hat(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1), torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def right_jacobian(v: torch.Tensor) -> torch.Tensor:
    """``[..., 3, 3]``: ``exp(v + d) = exp(v) exp(J_r(v) d)`` to first order."""
    th2 = (v * v).sum(-1)[..., None, None]
    th = torch.sqrt(th2.clamp_min(1e-30))
    small = th2 < 1e-8
    a = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-30))
    b = torch.where(small, 1.0 / 6 - th2 / 120, (th - torch.sin(th)) / (th2 * th).clamp_min(1e-30))
    k = hat(v)
    return torch.eye(3, dtype=v.dtype) - a * k + b * (k @ k)


def right_jacobian_inv(v: torch.Tensor) -> torch.Tensor:
    """``[..., 3, 3]``: ``log(exp(v) exp(d)) = v + J_r^-1(v) d`` to first order."""
    th2 = (v * v).sum(-1)[..., None, None]
    th = torch.sqrt(th2.clamp_min(1e-30))
    small = th2 < 1e-6
    c = torch.where(small, 1.0 / 12 + th2 / 720,
                    1 / th2.clamp_min(1e-30) - (1 + torch.cos(th)) / (2 * th * torch.sin(th)).clamp_min(1e-30))
    k = hat(v)
    return torch.eye(3, dtype=v.dtype) + 0.5 * k + c * (k @ k)


def so3_log(r: torch.Tensor) -> torch.Tensor:
    """``[..., 3]`` rotation vector of ``[..., 3, 3]`` rotations (angles below about 170 deg)."""
    cos = ((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.arccos(cos)
    w = torch.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    f = torch.where(th < 1e-4, 0.5 + th * th / 12, th / (2 * torch.sin(th)))
    return f[..., None] * w


@functools.lru_cache(maxsize=1)
def ancestors() -> np.ndarray:
    """``[B, B]`` bool: ``[b, i]`` when body ``i`` is ``b`` or one of its ancestors (``i = 0`` is the root)."""
    sk = skeleton()
    out = np.zeros((sk.num_bodies, sk.num_bodies), bool)
    for b in range(sk.num_bodies):
        i = b
        while i >= 0:
            out[b, i] = True
            i = sk.parents[i]
    return out


@dataclass
class State:
    pos: torch.Tensor      # [T, B, 3]
    rot: torch.Tensor      # [T, B, 3, 3]
    axes: torch.Tensor     # [T, B, 3, 3]: column c = world angular velocity per unit of the body's coordinate c
    pts: torch.Tensor      # [T, K, 3] candidate points
    h: torch.Tensor        # [T, K] their surface heights


@torch.no_grad()
def kinematics(prob: Problem, x: torch.Tensor) -> State:
    sk = skeleton()
    root_pos, root_rot, dof = unpack(prob, x)
    pos, rot = fk(sk, root_pos, root_rot, dof)
    e = torch.cat([x[:, None, 3:6], dof.reshape(dof.shape[0], -1, 3)], 1)     # [T, B, 3]: root perturbation, joints
    axes = rot @ right_jacobian(e)          # d R_i = [axes d e_i]_x R_i : R_i = R_parent exp(e_i)
    pts = candidate_points(sk, pos, rot)
    return State(pos, rot, axes, pts, candidate_heights(sk, pts))


def point_jacobian(st: State, frames, bodies, points: torch.Tensor) -> torch.Tensor:
    """``[N, 3, 75]``: how points fixed to ``bodies`` at ``frames`` move with the frame's 75 variables."""
    frames, bodies = torch.as_tensor(np.asarray(frames)), torch.as_tensor(np.asarray(bodies))
    N = len(frames)
    d = points[:, None, :] - st.pos[frames]                                     # [N, B, 3]
    ax = st.axes[frames].transpose(-1, -2)                                     # [N, B, 3c, 3xyz]
    cr = torch.linalg.cross(ax, d[:, :, None, :].expand_as(ax), dim=-1)          # [N, B, 3c, 3xyz]
    cr = cr * torch.as_tensor(ancestors())[bodies][:, :, None, None]
    out = torch.zeros(N, 3, NV, dtype=points.dtype)
    out[:, :, :3] = torch.eye(3, dtype=points.dtype)
    out[:, :, 3:] = cr.permute(0, 3, 1, 2).reshape(N, 3, NV - 3)              # body i -> columns 3 + 3i .. 5 + 3i
    return out


def rotation_jacobian(st: State, frames, bodies) -> torch.Tensor:
    """``[N, 3, 75]``: the world angular velocity of ``bodies`` at ``frames`` per variable."""
    frames, bodies = torch.as_tensor(np.asarray(frames)), torch.as_tensor(np.asarray(bodies))
    ax = st.axes[frames] * torch.as_tensor(ancestors())[bodies][:, :, None, None]          # [N, B, 3xyz, 3c]
    out = torch.zeros(len(frames), 3, NV, dtype=ax.dtype)
    out[:, :, 3:] = ax.permute(0, 2, 1, 3).reshape(len(frames), 3, NV - 3)
    return out


def _sq(v: torch.Tensor) -> float:
    return float((v * v).sum())


def residuals(prob: Problem, x: torch.Tensor, jacobian: bool = False) -> dict:
    """Every scale-normalised, weight-rooted residual of the objective (``energy`` is their squared sum).

    Per frame (``frames``, ``r`` and with ``jacobian`` the ``[N, 75]`` rows ``J``):

    ``support``  targeted candidates at ``CLEARANCE_M``;   ``floor``  nothing below ``FLOOR_MIN_M``;
    ``off``      zones the human holds up at or above ``OFF_FLOOR_M``;
    ``pair``     requested pairs at most ``PAIR_TARGET_M`` apart;
    ``pen``      no plant pair overlapping by more than ``PEN_TOL_M``, found anew in the current pose;
    ``ends``     hands, feet and head keep their shipped orientation (a clipped elbow twist moves to the
                 wrist, not into the hand);
    ``anchor``   every body stays near its shipped position: the edit is local. Without it nothing holds
                 the body in the horizontal plane (the floor terms act on heights), and the pilot's first
                 solve slid Warrior II 35 cm sideways and rotated Plow 49 deg.

    Linear, handled in closed form: ``dof`` (joint coordinates near the shipped ones), ``root`` (the
    root's offset) and ``smooth`` (second differences of the correction). A no-slip term (a candidate
    targeted on consecutive frames keeps its horizontal position) was tried and removed: real supports
    pivot, and pinning every corner for a whole contact run dragged the body with it.
    """
    sk, s, w, plan = skeleton(), SCALES, prob.weights, prob.plan
    st = kinematics(prob, x)
    out = {}
    tw, ow = torch.as_tensor(plan["target_w"]), torch.as_tensor(plan["off_w"])
    ones = torch.ones_like(tw)
    for name, mask, level, scale, fade in (("support", tw > 0, CLEARANCE_M, s["support"], tw),
                                           ("floor", st.h < FLOOR_MIN_M, FLOOR_MIN_M, s["floor"], ones),
                                           ("off", (ow > 0) & (st.h < OFF_FLOOR_M), OFF_FLOOR_M, s["off"], ow)):
        f, k = torch.nonzero(mask, as_tuple=True)
        c = math.sqrt(w[name]) / scale * torch.sqrt(fade[f, k])
        blk = {"frames": f.numpy(), "r": c * (st.h[f, k] - level)}
        if jacobian:
            blk["J"] = c[:, None] * point_jacobian(st, f, sk.cand_body[k.numpy()], st.pts[f, k])[:, 2]
        out[name] = blk
    nf, na, nb, _ = near_body_pairs(sk, st.pos, st.rot, prob.pen["floor"], prob.pen["pairs"])
    pen = {"frames": nf, "body_a": na, "body_b": nb, "floor": np.full(len(nf), prob.pen["floor"])}
    for name, items in (("pair", prob.close), ("pen", pen)):
        c = math.sqrt(w[name]) / s[name]
        if not len(items["frames"]):
            out[name] = {"frames": np.zeros(0, int), "r": torch.zeros(0, dtype=F64), "J": torch.zeros(0, NV, dtype=F64)}
            continue
        g, pa, pb, u = body_gaps(sk, st.pos, st.rot, items["frames"], items["body_a"], items["body_b"], witnesses=True)
        if name == "pair":          # the closest body pair of each item's two zones
            order = np.lexsort((g.numpy(), items["item"]))
            first = order[np.r_[True, np.diff(items["item"][order]) != 0]]
            items = {k: (v[first] if isinstance(v, np.ndarray) and len(v) == len(g) else v) for k, v in items.items()}
            g, pa, pb, u = g[first], pa[first], pb[first], u[first]
        lim = torch.full_like(g, PAIR_TARGET_M) if name == "pair" else torch.as_tensor(items["floor"])
        act = (g > lim) if name == "pair" else (g < lim)
        idx = torch.nonzero(act, as_tuple=True)[0]
        c = c * (torch.sqrt(torch.as_tensor(items["w"])[idx]) if name == "pair" else torch.ones(len(idx), dtype=F64))
        blk = {"frames": items["frames"][idx.numpy()], "r": c * (g[idx] - lim[idx])}
        if jacobian:
            fr = items["frames"][idx.numpy()]
            ja = point_jacobian(st, fr, items["body_a"][idx.numpy()], pa[idx])
            jb = point_jacobian(st, fr, items["body_b"][idx.numpy()], pb[idx])
            blk["J"] = c[:, None] * (u[idx][:, :, None] * (jb - ja)).sum(1)
        out[name] = blk
    ends = np.array([sk.names.index(b) for b in END_BODIES])
    f = np.repeat(np.arange(prob.T), len(ends))
    b = np.tile(ends, prob.T)
    rel = prob.rot0[f, b].transpose(-1, -2) @ st.rot[f, b]
    c = math.sqrt(w["ends"]) / s["ends"]
    lg = so3_log(rel)
    blk = {"frames": np.repeat(f, 3), "r": c * lg.reshape(-1)}
    if jacobian:   # R' = R exp(R^T omega): d log(R0^T R) = J_r^-1(log) R^T omega
        blk["J"] = c * (right_jacobian_inv(lg) @ st.rot[f, b].transpose(-1, -2)
                        @ rotation_jacobian(st, f, b)).reshape(-1, NV)
    out["ends"] = blk
    c = math.sqrt(w["anchor"]) / s["anchor"]
    f = np.repeat(np.arange(prob.T), sk.num_bodies)
    b = np.tile(np.arange(sk.num_bodies), prob.T)
    blk = {"frames": np.repeat(f, 3), "r": c * (st.pos - prob.pos0).reshape(-1)}
    if jacobian:
        blk["J"] = torch.cat([c * point_jacobian(st, f[i:i + 8192], b[i:i + 8192], st.pos[f[i:i + 8192], b[i:i + 8192]])
                              for i in range(0, len(f), 8192)]).reshape(-1, NV)
    out["anchor"] = blk
    out["balance"] = _balance(prob, st, jacobian)
    return out


def _balance(prob: Problem, st: State, jacobian: bool) -> dict:
    """The COM against the targeted supports' hull, on hold-window frames. For every hull edge ``v1 v2``
    (candidate points, counter-clockwise) with the COM ``c`` within ``BALANCE_MARGIN_M`` of it or outside,
    ``r = sqrt(w) (d - margin) / scale`` with ``d = (e x (c - v1)) / |e|``, ``e = v2 - v1``, the signed
    distance inside. The Jacobian runs through the COM and both edge points: ``d`` does not change when the
    body translates, so a Jacobian through the COM alone would move the root in vain."""
    from scipy.spatial import ConvexHull, QhullError

    sk, w = skeleton(), prob.weights
    mass, centre = mass_model()
    bw, tw = prob.plan["balance_w"], prob.plan["target_w"]
    frames = np.nonzero(bw > 0)[0]
    empty = {"frames": np.zeros(0, int), "r": torch.zeros(0, dtype=F64), "J": torch.zeros(0, NV, dtype=F64)}
    if not len(frames):
        return empty
    c_local = torch.as_tensor(centre, dtype=F64)
    m = torch.as_tensor(mass, dtype=F64)
    com_b = st.pos[frames] + (st.rot[frames] @ c_local[..., None]).squeeze(-1)             # [F, B, 3]
    com = ((m[None, :, None] * com_b).sum(1) / m.sum())[:, :2].numpy()                   # [F, 2]
    rows = []                                                  # (frame index, k1, k2, d, g_c, g_1, g_2)
    for i, f in enumerate(frames):
        ks = np.nonzero(tw[f] > 0.5)[0]
        if len(ks) < 3:
            continue
        xy = st.pts[f, ks, :2].numpy()
        try:
            hull = ConvexHull(xy)
        except QhullError:
            continue
        c, mid = com[i], xy[hull.vertices].mean(0)
        for i1, i2 in hull.simplices:
            v1, v2 = xy[i1], xy[i2]
            e = v2 - v1
            L = float(np.hypot(*e))
            if L < 1e-6:
                continue
            if (e[0] * (mid - v1)[1] - e[1] * (mid - v1)[0]) < 0:    # orient so the hull's inside is positive
                i1, i2, v1, v2, e = i2, i1, v2, v1, -e
            wv = c - v1
            d = (e[0] * wv[1] - e[1] * wv[0]) / L
            if d >= BALANCE_MARGIN_M:
                continue
            g_c = np.array([-e[1], e[0]]) / L
            g_1 = np.array([-wv[1] + e[1], wv[0] - e[0]]) / L + d * e / L ** 2
            g_2 = np.array([wv[1], -wv[0]]) / L - d * e / L ** 2
            rows.append((i, ks[i1], ks[i2], d, g_c, g_1, g_2))
    if not rows:
        return empty
    rf = np.array([r[0] for r in rows])
    scale = math.sqrt(w["balance"]) / SCALES["balance"] * torch.sqrt(torch.as_tensor(bw[frames[rf]]))
    blk = {"frames": frames[rf], "r": scale * torch.as_tensor([r[3] - BALANCE_MARGIN_M for r in rows], dtype=F64)}
    if jacobian:
        uf = np.unique(rf)
        n_b = sk.num_bodies
        f_rep = np.repeat(frames[uf], n_b)
        b_rep = np.tile(np.arange(n_b), len(uf))
        jb = point_jacobian(st, f_rep, b_rep, com_b[uf].reshape(-1, 3)).reshape(len(uf), n_b, 3, NV)
        jcom = ((m[None, :, None, None] * jb).sum(1) / m.sum())[:, :2]                   # [U, 2, 75]
        pos_of = {int(u): k for k, u in enumerate(uf)}
        fr = frames[rf]
        k1 = np.array([r[1] for r in rows]); k2 = np.array([r[2] for r in rows])
        j1 = point_jacobian(st, fr, sk.cand_body[k1], st.pts[fr, k1])[:, :2]
        j2 = point_jacobian(st, fr, sk.cand_body[k2], st.pts[fr, k2])[:, :2]
        gc = torch.as_tensor(np.array([r[4] for r in rows]))
        g1 = torch.as_tensor(np.array([r[5] for r in rows]))
        g2 = torch.as_tensor(np.array([r[6] for r in rows]))
        jc = jcom[[pos_of[int(i)] for i in rf]]
        blk["J"] = scale[:, None] * ((gc[:, :, None] * jc).sum(1) + (g1[:, :, None] * j1).sum(1)
                                     + (g2[:, :, None] * j2).sum(1))
    return blk


def _linear_terms(prob: Problem, x: torch.Tensor) -> dict:
    """The linear residuals' values: ``dof``, ``root``, ``smooth`` (each ``[..]`` tensors)."""
    s, w = SCALES, prob.weights
    dofc = x[:, 6:] - prob.dof0
    corr = torch.cat([x[:, :3] / s["acc_pos"], x[:, 3:6] / s["acc_rot"], dofc / s["acc_dof"]], 1)
    return {"dof": math.sqrt(w["dof"]) * dofc / s["dof"],
            "root": math.sqrt(w["root"]) * torch.cat([x[:, :3] / s["root_pos"], x[:, 3:6] / s["root_rot"]], 1),
            "smooth": math.sqrt(w["smooth"]) * (corr[2:] - 2 * corr[1:-1] + corr[:-2])}


def energy(prob: Problem, x: torch.Tensor) -> tuple[float, dict]:
    """``(total, {term: value})``: the squared residuals, each term divided by the frame count."""
    terms = {k: _sq(v["r"]) for k, v in residuals(prob, x).items()}
    terms.update({k: _sq(v) for k, v in _linear_terms(prob, x).items()})
    terms = {k: v / prob.T for k, v in terms.items()}
    return sum(terms.values()), terms


def _smooth_diagonals(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three upper diagonals of ``D^T D`` for the second difference ``D`` over ``T`` frames."""
    d0, d1, d2 = np.zeros(T), np.zeros(max(T - 1, 0)), np.zeros(max(T - 2, 0))
    for t in range(T - 2):
        for i, ci in enumerate((1.0, -2.0, 1.0)):
            d0[t + i] += ci * ci
            for j, cj in enumerate((1.0, -2.0, 1.0)):
                if j == i + 1:
                    d1[t + i] += ci * cj
                elif j == i + 2:
                    d2[t + i] += ci * cj
    return d0, d1, d2


BAND = 2 * NV      # upper bandwidth: the smoothness couples a coordinate at t with itself at t + 2


def normal_equations(prob: Problem, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(ab, g)``: ``J^T J`` in LAPACK upper-band storage ``[BAND + 1, 75 T]`` and ``J^T r`` ``[75 T]``."""
    T = prob.T
    s, w = SCALES, prob.weights
    res = residuals(prob, x, jacobian=True)
    A = torch.zeros(T, NV, NV, dtype=F64)
    g = torch.zeros(T, NV, dtype=F64)
    for blk in res.values():
        if len(blk["r"]):
            _gram(A, g, torch.as_tensor(blk["frames"]), blk["J"], blk["r"])
    lin = _linear_terms(prob, x)
    cd = w["dof"] / s["dof"] ** 2
    A[:, 6:, 6:] += cd * torch.eye(69, dtype=F64)
    g[:, 6:] += math.sqrt(w["dof"]) / s["dof"] * lin["dof"]
    cr = torch.tensor([w["root"] / s["root_pos"] ** 2] * 3 + [w["root"] / s["root_rot"] ** 2] * 3, dtype=F64)
    A[:, :6, :6] += torch.diag(cr)
    g[:, :6] += torch.sqrt(cr) * lin["root"]
    # smoothness: r = sqrt(w) D (c / scale): add its Gram D^T D per variable, and its gradient
    vs = torch.tensor([s["acc_pos"]] * 3 + [s["acc_rot"]] * 3 + [s["acc_dof"]] * 69, dtype=F64)
    cs = w["smooth"] / vs ** 2
    if T > 2:
        r = lin["smooth"]                                       # [T-2, 75], already sqrt(w)-scaled / scale
        gs = torch.zeros(T, NV, dtype=F64)
        k = math.sqrt(w["smooth"]) / vs
        gs[2:] += k * r
        gs[1:-1] += -2 * k * r
        gs[:-2] += k * r
        g += gs
    ab = np.zeros((BAND + 1, NV * T))
    ia, ib = np.triu_indices(NV)
    cols = NV * np.arange(T)[:, None] + ib[None, :]
    ab[(BAND + ia - ib)[None, :].repeat(T, 0), cols] += A.numpy()[:, ia, ib]
    if T > 2:
        d0, d1, d2 = _smooth_diagonals(T)
        csn = cs.numpy()
        ab[BAND] += (d0[:, None] * csn[None, :]).ravel()
        ab[BAND - NV, NV:] += (d1[:, None] * csn[None, :]).ravel()
        ab[0, 2 * NV:] += (d2[:, None] * csn[None, :]).ravel()
    return ab, g.numpy().ravel()


def _gram(A: torch.Tensor, g: torch.Tensor, frames: torch.Tensor, J: torch.Tensor, r: torch.Tensor, chunk: int = 4096):
    for i in range(0, len(r), chunk):
        f, j, rr = frames[i:i + chunk], J[i:i + chunk], r[i:i + chunk]
        A.index_add_(0, f, j[:, :, None] * j[:, None, :])
        g.index_add_(0, f, j * rr[:, None])


def initial(prob: Problem) -> torch.Tensor:
    x = torch.zeros(prob.T, NV, dtype=F64)
    x[:, 6:] = torch.maximum(torch.minimum(prob.dof0, prob.upper), prob.lower)
    return x


def bounds(prob: Problem) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lo, hi) [T, 75]``: the plant's joint box intersected with the edit budget around the shipped pose
    clipped into it; the root's offset and perturbation within their budgets."""
    T = prob.T
    q0 = torch.maximum(torch.minimum(prob.dof0, prob.upper), prob.lower)
    root = torch.tensor([BUDGET_ROOT_M] * 3 + [BUDGET_ROOT_RAD] * 3, dtype=F64).expand(T, 6)
    lo = torch.cat([-root, torch.maximum(prob.lower.expand(T, -1), q0 - BUDGET_JOINT_RAD)], 1)
    hi = torch.cat([root, torch.minimum(prob.upper.expand(T, -1), q0 + BUDGET_JOINT_RAD)], 1)
    return lo, hi


def solve(prob: Problem, x0: torch.Tensor | None = None, iters: int = 60, rtol: float = 1e-7,
          log=None) -> tuple[torch.Tensor, dict]:
    """Levenberg-Marquardt over the whole clip: the band of ``normal_equations`` solved exactly
    (``solveh_banded``), the plant's joint box enforced by projection with the bound-active coordinates
    frozen for the step (projected Newton). From ``x0`` (default: the shipped reference clipped into the
    box). Returns the solution and the solver's report."""
    import time

    from scipy.linalg import LinAlgError, solveh_banded

    start = time.time()
    T = prob.T
    x = initial(prob) if x0 is None else x0.clone()
    lo, hi = bounds(prob)
    E, _ = energy(prob, x)
    E0, lam, history = E, 1e-3, []
    for it in range(iters):
        t_it, tries = time.time(), 0
        ab, g = normal_equations(prob, x)
        gt = torch.as_tensor(g).reshape(T, NV)
        frozen = (((x <= lo + 1e-12) & (gt > 0)) | ((x >= hi - 1e-12) & (gt < 0))).numpy().ravel()
        diag = ab[BAND].copy()
        accepted = False
        while lam < 1e8:
            band = ab.copy()
            band[BAND] += lam * np.maximum(diag, 1e-6) + 1e-9
            fz = np.nonzero(frozen)[0]
            if len(fz):
                band[:, fz] = 0.0
                for k in range(1, BAND + 1):
                    cols = fz + k
                    cols = cols[cols < band.shape[1]]
                    band[BAND - k, cols] = 0.0
                band[BAND, fz] = 1.0
            rhs = -g.copy()
            rhs[frozen] = 0.0
            try:
                step = solveh_banded(band, rhs, lower=False, check_finite=False)
            except LinAlgError:
                lam *= 10
                continue
            xn = torch.minimum(torch.maximum(x + torch.as_tensor(step).reshape(T, NV), lo), hi)
            En, _ = energy(prob, xn)
            tries += 1
            if En < E:
                accepted = True
                break
            lam *= 10
        if not accepted:
            break
        history.append(round(En, 4))
        if log is not None:
            log(f"it {it} E {En:.4f} lambda {lam:.1e} tries {tries} {time.time() - t_it:.1f}s")
        dec, x, E = E - En, xn, En
        lam = max(lam / 5, 1e-6)
        if dec < rtol * max(E, 1e-9) or np.abs(step).max() < 1e-7:
            break
    _, terms = energy(prob, x)
    return x, {"iterations": len(history), "seconds": round(time.time() - start, 1), "energy_start": round(E0, 4),
               "energy": round(E, 5), "terms": {k: round(v, 5) for k, v in terms.items()}, "history": history[-8:]}


# --------------------------------------------------------------------------- #
# The retargeted motion, and what it realises
# --------------------------------------------------------------------------- #
AVATAR_TOUCH_M = 0.01      # the avatar touches at <= 1 cm (verdicts.AVATAR_TOUCH_CM)
REALISED_M = 0.02          # a support floats above 2 cm (the labels' threshold, audit.HOVER_M)
DEEP_OVERLAP_M = 0.01      # a body pair overlapping by more than this is a penetration
# The edit budget as a validation flag (the solver's own bounds are per coordinate): the whole body moves less
# than EDIT_MEAN_P95_M on 95 % of frames and no body further than EDIT_BODY_MAX_M, the root's budget. A clip
# over it is flagged for Pass C, not failed: the largest supports float 15 cm and a joint brought back into
# the plant's box can move a hand 20 cm.
EDIT_MEAN_P95_M = 0.10
EDIT_BODY_MAX_M = BUDGET_ROOT_M
SPIKE_ACC = 100.0          # m/s^2: a body accelerating harder than this (10 g) on a frame the reference does not is a jerk
GENTLE_SMOOTH = 5.0        # solve_gently: the smoothness weight's multiplier
GENTLE_STAGES = (0.01, 0.1, 1.0)   # ... and the pair and penetration weights' continuation


def _quat(rot: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """xyzw quaternions of ``rot``, each signed like the matching one of ``like`` (an unchanged pose
    keeps its stored quaternion's sign)."""
    from protomotions.utils.rotations import matrix_to_quaternion

    q = matrix_to_quaternion(rot, w_last=True)
    flip = (q * like.to(q.dtype)).sum(-1, keepdim=True) < 0
    return torch.where(flip, -q, q)


def regenerate(prob: Problem, x: torch.Tensor, motion: dict) -> dict:
    """Every field of the ``.motion`` from the solved joint coordinates: forward kinematics for the body
    transforms, the exp-map for ``local_rigid_body_rot`` and ``dof_pos`` (the plant's coordinates, the reset
    path's convention), the converter's velocity routine (``make_hold_extended_clips.recompute_velocities``)
    and ``rigid_body_contacts`` from the geometry (a body whose lowest collision surface is within
    ``AVATAR_TOUCH_M`` of the floor; the shipped field is a speed-and-height heuristic, 0-16 % precise).
    Non-tensor fields are copied. A clip the solve left unchanged is returned as it was, bit for bit."""
    from make_hold_extended_clips import recompute_velocities
    from reference_curation import capture

    if torch.equal(x, initial(prob)) and torch.equal(x[:, 6:], prob.dof0):
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in motion.items()}
    sk = skeleton()
    root_pos, root_rot, dof = unpack(prob, x)
    with torch.no_grad():
        pos, rot = fk(sk, root_pos, root_rot, dof)
        local = torch.cat([rot[:, :1], so3_exp(dof.reshape(prob.T, -1, 3))], 1)
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in motion.items()}
    out["rigid_body_pos"] = pos.to(motion["rigid_body_pos"].dtype)
    out["rigid_body_rot"] = _quat(rot, motion["rigid_body_rot"]).to(motion["rigid_body_rot"].dtype)
    out["local_rigid_body_rot"] = _quat(local, motion["local_rigid_body_rot"]).to(motion["local_rigid_body_rot"].dtype)
    out["dof_pos"] = dof.to(motion["dof_pos"].dtype)
    for k, v in recompute_velocities(out, prob.fps).items():
        out[k] = v.to(motion[k].dtype)
    low = capture.body_min_z(out["rigid_body_pos"].double().numpy(), out["rigid_body_rot"].double().numpy())
    out["rigid_body_contacts"] = torch.as_tensor(low <= AVATAR_TOUCH_M)
    return out


def zone_min_z(pos: np.ndarray, rot_quat: np.ndarray) -> np.ndarray:
    """``[T, Z]`` lowest collision surface per zone: the kernels behind the store's ``avatar_min_z``."""
    from reference_curation import capture

    return capture.zone_min(capture.body_min_z(pos, rot_quat))


def limit_excess_deg(dof: np.ndarray) -> np.ndarray:
    """``[T, 69]`` degrees beyond the plant's box."""
    sk = skeleton()
    lo, hi = sk.lower.numpy(), sk.upper.numpy()
    return np.degrees(np.maximum(lo - dof, 0) + np.maximum(dof - hi, 0))


def deep_overlaps(pos: torch.Tensor, rot: torch.Tensor) -> set:
    """``{(frame, body_a, body_b)}`` of the plant's body pairs overlapping by more than ``DEEP_OVERLAP_M``."""
    f, a, b, g = near_body_pairs(skeleton(), pos, rot, 0.0)
    keep = g < -DEEP_OVERLAP_M
    return set(zip(f[keep].tolist(), a[keep].tolist(), b[keep].tolist()))


def clip_metrics(prob: Problem, x: torch.Tensor, motion_before: dict, motion_after: dict, store) -> dict:
    """What the edit did, before and after: capture agreement, floor and self penetration, the plant's
    joint box, the edit's size and smoothness, and the body-body requests."""
    from protomotions.utils.rotations import quaternion_to_matrix

    sk = skeleton()
    state = store["human_floor_state"]
    head = prob.plan["head_blocked"]
    out = {}
    per = {}
    for tag, mot in (("before", motion_before), ("after", motion_after)):
        pos, q = mot["rigid_body_pos"].double().numpy(), mot["rigid_body_rot"].double().numpy()
        zmin = zone_min_z(pos, q)
        touch = state == 1
        touch[:, ZI["HEAD"]] &= ~head
        sep = state == 0
        rotm = quaternion_to_matrix(torch.as_tensor(q), w_last=True)
        h = candidate_heights(sk, candidate_points(sk, torch.as_tensor(pos), rotm)).numpy()
        of, oa, ob, og = near_body_pairs(sk, torch.as_tensor(pos), rotm, 0.0)
        deep = og < -DEEP_OVERLAP_M
        per[tag] = {"zmin": zmin, "pos": pos,
                    "overlaps": set(zip(of[deep].tolist(), oa[deep].tolist(), ob[deep].tolist()))}
        worst = int(og.argmin()) if len(og) else None
        out[tag] = {
            "human_touch_zone_frames": int(touch.sum()),
            "floating_gt_2cm": _frac((zmin > REALISED_M)[touch]),
            "floating_gt_5cm": _frac((zmin > 0.05)[touch]),
            "hover_cm_p50_p90_p99": ([round(100 * float(v), 2) for v in np.percentile(zmin[touch], [50, 90, 99])]
                                     if touch.any() else None),
            "separated_zone_frames": int(sep.sum()),
            "phantom_le_2cm": _frac((zmin <= REALISED_M)[sep]),
            "agreement": _frac(np.r_[(zmin <= REALISED_M)[touch], (zmin > REALISED_M)[sep]]),
            "floor_min_cm": round(100 * float(h.min()), 2),
            "deep_overlap_pair_frames": len(per[tag]["overlaps"]),
            "overlap_frames_gt_tol": int(len(np.unique(of[og < -PEN_TOL_M]))),
            "overlap_max_cm": round(-100 * float(og[worst]), 2) if worst is not None else 0.0,
            "overlap_worst": [sk.names[oa[worst]], sk.names[ob[worst]], int(of[worst])] if worst is not None else None,
            "limit_excess_max_deg": round(float(limit_excess_deg(mot["dof_pos"].double().numpy()).max()), 2),
            "frames_past_limit_2deg": int((limit_excess_deg(mot["dof_pos"].double().numpy()) > 2).any(1).sum()),
            "frames_acc_gt_spike": int((np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1).max(1)
                                        * prob.fps ** 2 > SPIKE_ACC).sum()),
        }
    disp = np.linalg.norm(per["after"]["pos"] - per["before"]["pos"], axis=-1)          # [T, B]
    dq = np.degrees(np.abs((x[:, 6:] - prob.dof0).numpy()))
    acc = lambda p: np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1) * prob.fps ** 2
    out["edit"] = {"mean_disp_cm_p50": round(100 * float(np.median(disp.mean(1))), 2),
                   "mean_disp_cm_p95": round(100 * float(np.percentile(disp.mean(1), 95)), 2),
                   "mean_disp_cm_max": round(100 * float(disp.mean(1).max()), 2),
                   "body_disp_cm_max": round(100 * float(disp.max()), 2),
                   "body_disp_worst": sk.names[int(disp.max(0).argmax())],
                   "joint_change_deg_p50": round(float(np.median(dq)), 2),
                   "joint_change_deg_p99": round(float(np.percentile(dq, 99)), 2),
                   "joint_change_deg_max": round(float(dq.max()), 2),
                   "body_acc_rms_before": round(float(np.sqrt((acc(per["before"]["pos"]) ** 2).mean())), 3),
                   "body_acc_rms_after": round(float(np.sqrt((acc(per["after"]["pos"]) ** 2).mean())), 3)}
    out["edit"]["over_budget"] = [k for k, v, lim in (("mean_p95", np.percentile(disp.mean(1), 95), EDIT_MEAN_P95_M),
                                                      ("body_max", disp.max(), EDIT_BODY_MAX_M)) if v > lim]
    out["new_deep_overlaps"] = len(per["after"]["overlaps"] - per["before"]["overlaps"])
    out["head_blocked_frames"] = int(head.sum())
    reqs = []
    for r in prob.requests:
        row = {k: r[k] for k in ("hold_id", "contact", "why", "target_role", "status")}
        row["frames"] = int(len(r["frames"]))
        if len(r["frames"]):
            for tag in ("before", "after"):
                mot = motion_before if tag == "before" else motion_after
                g, _, _ = zone_pair_gaps(sk, mot["rigid_body_pos"].double(),
                                         quaternion_to_matrix(mot["rigid_body_rot"].double(), w_last=True),
                                         *r["zones"], r["frames"])
                row[f"gap_cm_{tag}"] = round(100 * float(np.median(g)), 2)
        reqs.append(row)
    out["pair_requests"] = reqs
    return out


def _frac(mask: np.ndarray) -> float | None:
    return round(float(mask.mean()), 4) if mask.size else None


# --------------------------------------------------------------------------- #
# Clips, the corpus and the release of the retargeted motions
# --------------------------------------------------------------------------- #
OUT_ROOT = ids.OUTPUT_ROOT / "retarget"            # the motions and per-frame lineage (bulky, regenerable)
RECORD_ROOT = ids.DATA_ROOT / "retarget"           # the records (small)
ITERS = 60
CONFIG = {"clearance_m": CLEARANCE_M, "floor_min_m": FLOOR_MIN_M, "off_floor_m": OFF_FLOOR_M,
          "flat_spread_m": FLAT_SPREAD_M, "edge_band_m": EDGE_BAND_M, "release_m": RELEASE_M,
          "mode_frames": MODE_FRAMES, "head_inverted_cos": HEAD_INVERTED_COS, "corner_neighbours": CORNER_NEIGHBOURS,
          "face_cos": FACE_COS, "pair_target_m": PAIR_TARGET_M, "pair_reach_m": PAIR_REACH_M,
          "pen_tol_m": PEN_TOL_M, "pen_pairs": "plant", "limit_margin_deg": math.degrees(LIMIT_MARGIN_RAD),
          "weights": WEIGHTS, "scales": SCALES, "end_bodies": list(END_BODIES), "iters": ITERS,
          "budget_joint_deg": math.degrees(BUDGET_JOINT_RAD), "budget_root_m": BUDGET_ROOT_M,
          "budget_root_deg": math.degrees(BUDGET_ROOT_RAD), "min_run": MIN_RUN, "ramp": RAMP,
          "edit_mean_p95_m": EDIT_MEAN_P95_M, "edit_body_max_m": EDIT_BODY_MAX_M,
          "spike_acc": SPIKE_ACC, "gentle_smooth": GENTLE_SMOOTH, "gentle_stages": list(GENTLE_STAGES),
          "balance_margin_m": BALANCE_MARGIN_M}


def spike_frames(prob: Problem, x: torch.Tensor) -> int:
    """Frames on which some body's acceleration exceeds ``SPIKE_ACC`` in the solution and not in the
    shipped reference: a jerk the edit made."""
    root_pos, root_rot, dof = unpack(prob, x)
    with torch.no_grad():
        pos, _ = fk(skeleton(), root_pos, root_rot, dof)

    def acc(p):
        return ((p[2:] - 2 * p[1:-1] + p[:-2]).norm(dim=-1) * prob.fps ** 2).max(1).values

    return int(((acc(pos) > SPIKE_ACC) & (acc(prob.pos0) <= SPIKE_ACC)).sum())


def solve_gently(prob: Problem, iters: int = ITERS) -> tuple[torch.Tensor, dict]:
    """The fallback for a solution with jerks: the pair and penetration terms brought in by continuation
    (1 %, 10 %, then all of their weight, each stage starting from the last) under a five times stiffer
    smoothness. Solved at full weight from the start, a limb the reference buries in another (a forearm in
    a shin, a lotus foot in a thigh) left on each frame by its own nearest side, and neighbouring frames that
    left by different sides jumped between them (Standing Forward Bend -a: 536 m/s^2 at an elbow)."""
    base = dict(prob.weights)
    smooth = dict(base, smooth=GENTLE_SMOOTH * base["smooth"])
    x, reps = None, []
    try:
        for k in GENTLE_STAGES:
            prob.weights = dict(smooth, pen=k * base["pen"], pair=k * base["pair"])
            x, rep = solve(prob, x0=x, iters=iters if k == 1.0 else max(10, iters // 2))
            reps.append(rep)
    finally:
        prob.weights = base
    rep = dict(reps[-1], iterations=sum(r["iterations"] for r in reps), seconds=round(sum(r["seconds"] for r in reps), 1),
               energy_start=reps[0]["energy_start"], stages=[{"weight": k, "iterations": r["iterations"],
                                                             "energy": r["energy"]} for k, r in zip(GENTLE_STAGES, reps)])
    return x, rep


def retarget_clip(stem: str, anns: dict, motion_dir=ids.SHIPPED_DIR, iters: int = ITERS) -> dict:
    """One x0 clip: ``{"motion", "record", "lineage"}``. A solution with jerks (``spike_frames``) is solved
    again gently (``solve_gently``); the gentle one is kept if it has fewer."""
    import time

    from reference_curation import sources

    start = time.time()
    store = sources.load(stem)
    prob = build_problem(stem, anns, motion_dir, store=store)
    motion = load_reference(stem, motion_dir)["motion"]
    x, rep = solve(prob, iters=iters)
    rep["spike_frames"] = spikes = spike_frames(prob, x)
    if spikes:
        xg, gentle = solve_gently(prob, iters)
        gentle["spike_frames"] = spike_frames(prob, xg)
        rep["gentle"] = {k: gentle[k] for k in ("iterations", "seconds", "energy", "spike_frames", "stages")}
        rep["kept"] = "gentle" if gentle["spike_frames"] < spikes else "direct"
        if rep["kept"] == "gentle":
            x = xg
            rep.update({k: gentle[k] for k in ("energy", "terms", "history")})
    after = regenerate(prob, x, motion)
    metrics = clip_metrics(prob, x, motion, after, store)
    tgt = prob.plan["target"]
    record = {"stem": stem, "frames": prob.T, "fps": prob.fps, "solver": rep, "metrics": metrics,
              "targets_by_zone": {z: int(tgt[:, sk_z].any(1).sum()) for z, sk_z in zip(ZONE_ORDER, skeleton().zone_cands)
                                  if tgt[:, sk_z].any()},
              "inputs": {"motion": ids.sha256_file(ids.motion_path(stem, motion_dir)),
                         "store_v3": sources.identity(store)},
              "seconds": round(time.time() - start, 1)}
    lineage = {"target": tgt, "off": prob.plan["off"], "head_blocked": prob.plan["head_blocked"],
               "x": x.numpy().astype(np.float32),
               "body_disp_m": np.linalg.norm(after["rigid_body_pos"].double().numpy()
                                             - motion["rigid_body_pos"].double().numpy(), axis=-1).astype(np.float16),
               "dof_change_rad": (x[:, 6:] - prob.dof0).numpy().astype(np.float16)}
    return {"motion": after, "record": record, "lineage": lineage}


@functools.lru_cache(maxsize=None)
def has_human(stem: str) -> bool:
    """The capture store v3 holds the performer's mesh for ``stem`` (``human_available``): false for a clip
    MOYO has no fit for and for Standing big toe hold -c, whose fit on disk is unreadable."""
    from reference_curation import sources

    return bool(sources.load(stem).meta.get("human_available"))


def _worker(job: tuple) -> tuple[str, dict | None, str | None]:
    stem, anns, motion_dir, out_dir, iters = job
    torch.set_num_threads(1)
    try:
        res = retarget_clip(stem, anns, motion_dir, iters)
        torch.save(res["motion"], Path(out_dir) / f"{stem}.motion")
        np.savez_compressed(Path(out_dir) / f"{stem}.lineage.npz", **res["lineage"])
        return stem, res["record"], None
    except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
        import traceback

        return stem, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"


def retarget_id(labels: dict, stems: list[str], motion_dir) -> str:
    from reference_curation import sources

    key = {"schema": SCHEMA_VERSION, "config": CONFIG, "labels": labels["id"],
           "generators": {Path(f).name: ids.sha256_file(f) for f in (__file__, sources.__file__, hm.__file__)},
           "motions": {s: ids.sha256_file(ids.motion_path(s, motion_dir)) for s in sorted(stems)}}
    return f"{labels['id']}.retarget_{RETARGET_VERSION}.{ids.sha256_json(key)[:10]}"


def run(labels: dict, stems: list[str], motion_dir=ids.SHIPPED_DIR, workers: int = 1, iters: int = ITERS,
        out_root: Path = OUT_ROOT) -> tuple[Path, list[dict], list[str]]:
    """Retarget ``stems`` into ``out_root/<retarget_id>/`` (motions and lineage); returns the directory, the
    clip records and the failures."""
    import concurrent.futures
    import multiprocessing

    rid = retarget_id(labels, stems, motion_dir)
    out = Path(out_root) / rid
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, {h: a for h, a in labels["anns"].items() if h.startswith(s + "@")}, str(motion_dir), str(out), iters)
            for s in stems]
    if workers > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]
    records = [r for _, r, e in results if r is not None]
    failures = [f"{s}: {e}" for s, _, e in results if e is not None]
    return out, records, failures


# --------------------------------------------------------------------------- #
# The contract, the corpus metrics and the records
# --------------------------------------------------------------------------- #
ROUND_TRIP_M = 1e-4
FLOOR_TOL_M = -0.005
RECORD_KEYS = ("stem", "frames", "fps", "solver", "metrics", "targets_by_zone", "inputs", "seconds")


def round_trip(motion: dict) -> dict:
    """The stored joint coordinates reproduce the stored bodies through the plant's FK (the reset path),
    ``local_rigid_body_rot`` is their exponential map, and every field is finite."""
    from protomotions.utils.rotations import quaternion_to_matrix

    sk = skeleton()
    dof = motion["dof_pos"].double()
    rot = quaternion_to_matrix(motion["rigid_body_rot"].double(), w_last=True)
    pos, rotm = fk(sk, motion["rigid_body_pos"][:, 0].double(), rot[:, 0], dof)
    local = quaternion_to_matrix(motion["local_rigid_body_rot"].double(), w_last=True)
    finite = all(bool(torch.isfinite(v).all()) for v in motion.values() if torch.is_tensor(v) and v.is_floating_point())
    return {"pos_m": float((pos - motion["rigid_body_pos"].double()).abs().max()),
            "rot": float((rotm - rot).abs().max()),
            "local": float((local[:, 1:] - so3_exp(dof.reshape(dof.shape[0], -1, 3))).abs().max()),
            "finite": finite}


def check(records: list[dict], out_dir: Path) -> list[str]:
    """Every violation of the retarget's contract: a motion outside the plant's box, below the floor
    beyond ``FLOOR_TOL_M``, failing the round trip, with non-finite fields, or a solve that found nothing."""
    problems = []
    for r in records:
        s, a = r["stem"], r["metrics"]["after"]
        if a["limit_excess_max_deg"] > 0.01:
            problems.append(f"{s}: joint coordinates {a['limit_excess_max_deg']} deg outside the plant's box")
        if a["floor_min_cm"] < 100 * FLOOR_TOL_M:
            problems.append(f"{s}: a surface {a['floor_min_cm']} cm below the floor")
        rt_ = round_trip(torch.load(Path(out_dir) / f"{s}.motion", map_location="cpu", weights_only=False))
        r["round_trip"] = rt_
        if not rt_["finite"] or rt_["pos_m"] > ROUND_TRIP_M or rt_["rot"] > ROUND_TRIP_M or rt_["local"] > ROUND_TRIP_M:
            problems.append(f"{s}: round trip {rt_}")
    return problems


def _window_hover(zmin: np.ndarray, f0: int, f1: int, zi: int) -> float:
    return float(np.median(zmin[f0:f1 + 1, zi]))


def corpus_metrics(labels: dict, out_dir: Path, records: list[dict], motion_dir=ids.SHIPPED_DIR) -> dict:
    """The step's exit numbers, before and after, measured exactly as Steps 2 and 6 measured them:
    labelled ground supports whose avatar window-median height exceeds 2 cm, over the source manifest
    (Step 2: 367 of 833) and over labels v1's configured supports the human touches (461 of 924). Each
    also without the supports the retarget must not bring down: the crowns the head collider cannot reach
    (head-blocked frames at the hold) and, in the source manifest, the 19 supports labels v1 removed
    (``forbidden_support``: Chaturanga's head, Bridge -a's hands). Plus the capture agreement, the
    penetrations and the edit sizes pooled over the clips."""
    src = ids.load_manifest(ids.DEFAULT_MANIFEST)
    stems = [c["stem"] for c in src["clips"] if ids.motion_path(c["stem"], out_dir).exists()]
    removed = {(hid, a["contact"]) for hid, rows in labels["anns"].items() for a in rows
               if a["kind"] == "ground" and a["target_role"] == "forbidden_support"}
    out = {"source": collections.Counter(), "labels": collections.Counter()}
    for stem in stems:
        z = {}
        for tag, d in (("before", motion_dir), ("after", out_dir)):
            m = torch.load(ids.motion_path(stem, d), map_location="cpu", weights_only=False)
            z[tag] = zone_min_z(m["rigid_body_pos"].double().numpy(), m["rigid_body_rot"].double().numpy())
        lineage = Path(out_dir) / f"{stem}.lineage.npz"
        blocked = np.load(lineage)["head_blocked"] if lineage.exists() else np.zeros(z["before"].shape[0], bool)
        for c in (c for c in src["clips"] if c["stem"] == stem):
            for h in c["holds"]:
                f0, f1 = int(h["frame_start"]), int(h["frame_end"])
                for p in h["pairs"]:
                    if not p.endswith(":G"):
                        continue
                    zn = p[:-2]
                    head_out = bool(zn == "HEAD" and blocked[f0:f1 + 1].mean() > 0.5)
                    gone = (f"{stem}@{int(h['frame_hold'])}", p) in removed
                    for tag in ("before", "after"):
                        over = _window_hover(z[tag], f0, f1, ZI[zn]) > REALISED_M
                        out["source"][f"{tag}_over"] += over
                        if not (head_out or gone):
                            out["source"][f"{tag}_over_reachable"] += over
                    out["source"]["supports"] += 1
                    out["source"]["reachable"] += not (head_out or gone)
                    out["source"]["head_blocked"] += head_out
                    out["source"]["removed_by_labels"] += gone
        for hid, rows in labels["anns"].items():
            if not hid.startswith(stem + "@"):
                continue
            for a in rows:
                if a["kind"] != "ground" or not a["in_configuration"] or a["source_state"] != "observed_contact":
                    continue
                iv = a["interval"]
                f0, f1 = iv["start_frame"], iv["end_frame_exclusive"] - 1
                zn = a["zones"][0]
                head_out = bool(zn == "HEAD" and blocked[f0:f1 + 1].mean() > 0.5)
                for tag in ("before", "after"):
                    over = _window_hover(z[tag], f0, f1, ZI[zn]) > REALISED_M
                    out["labels"][f"{tag}_over"] += over
                    if not head_out:
                        out["labels"][f"{tag}_over_reachable"] += over
                out["labels"]["supports"] += 1
                out["labels"]["reachable"] += not head_out
                out["labels"]["head_blocked"] += head_out
    pooled = collections.Counter()
    for r in records:
        m = r["metrics"]
        for tag in ("before", "after"):
            b = m[tag]
            pooled[f"{tag}_touch"] += b["human_touch_zone_frames"]
            pooled[f"{tag}_float"] += round((b["floating_gt_2cm"] or 0) * b["human_touch_zone_frames"])
            pooled[f"{tag}_sep"] += b["separated_zone_frames"]
            pooled[f"{tag}_phantom"] += round((b["phantom_le_2cm"] or 0) * b["separated_zone_frames"])
            pooled[f"{tag}_overlaps"] += b["deep_overlap_pair_frames"]
            pooled[f"{tag}_limit_frames"] += b["frames_past_limit_2deg"]
            pooled[f"{tag}_spike_frames"] += b["frames_acc_gt_spike"]
        pooled["new_overlaps"] += m["new_deep_overlaps"]
        pooled["resolved_gently"] += r["solver"].get("kept") == "gentle"
        pooled["jerky_direct"] += bool(r["solver"].get("spike_frames"))
        pooled["head_blocked"] += m["head_blocked_frames"]
        pooled["frames"] += r["frames"]
    edit = {k: [r["metrics"]["edit"][k] for r in records] for k in records[0]["metrics"]["edit"]
            if k not in ("body_disp_worst", "over_budget")} if records else {}
    over = [r["stem"] for r in records if r["metrics"]["edit"]["over_budget"]]
    pairs = collections.Counter(q["status"] for r in records for q in r["metrics"]["pair_requests"])
    closed = [q for r in records for q in r["metrics"]["pair_requests"] if q["status"] == "closed"]
    return {"exit": {k: dict(v) for k, v in out.items()}, "pooled": dict(pooled),
            "edit": {k: {"p50": float(np.median(v)), "max": float(np.max(v))} for k, v in edit.items()},
            "over_edit_budget": over,
            "pair_requests": dict(pairs),
            "pairs_closed_after_le_1cm": sum(q.get("gap_cm_after", 99) <= 1.0 for q in closed),
            "pairs_closed": len(closed), "clips": len(records)}


def _json_default(o):
    """numpy scalars and arrays in a record, as JSON."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def retarget_record_id(rid: str) -> Path:
    return RECORD_ROOT / rid


def summary_markdown(rid: str, m: dict, records: list[dict]) -> str:
    ex, po = m["exit"], m["pooled"]

    def pct(a, b):
        return f"{a} / {b} ({100 * a / max(b, 1):.1f} %)"

    lines = [f"# Retarget `{rid}`", "", "Generated by `reference_curation.retarget` (BUILD_PLAN Step 8); the rules are in its "
             "docstring. Before = the shipped ftC references, after = the retarget.", "",
             "| Metric | Before | After |", "|---|---|---|"]
    for name, k in (("Source manifest labelled supports > 2 cm (Step 2's 367 / 833)", "source"),
                    ("Labels v1 configured supports the human touches, > 2 cm", "labels")):
        e = ex[k]
        lines.append(f"| {name} | {pct(e['before_over'], e['supports'])} | {pct(e['after_over'], e['supports'])} |")
        what = ("the head collider's unreachable crowns and the supports labels v1 removed" if k == "source"
                else "the head collider's unreachable crowns")
        lines.append(f"| ... without {what} | {pct(e['before_over_reachable'], e['reachable'])} | "
                     f"{pct(e['after_over_reachable'], e['reachable'])} |")
    lines += [f"| Human-touch zone-frames floating > 2 cm | {pct(po['before_float'], po['before_touch'])} | "
              f"{pct(po['after_float'], po['after_touch'])} |",
              f"| Separated zone-frames touching (<= 2 cm) | {pct(po['before_phantom'], po['before_sep'])} | "
              f"{pct(po['after_phantom'], po['after_sep'])} |",
              f"| Frames past a joint range by > 2 deg (exp-map) | {pct(po['before_limit_frames'], po['frames'])} | "
              f"{pct(po['after_limit_frames'], po['frames'])} |",
              f"| Plant body-pair overlaps > 1 cm (pair-frames) | {po['before_overlaps']} | {po['after_overlaps']} (new {po['new_overlaps']}) |",
              f"| Frames with a body accelerating > {SPIKE_ACC:.0f} m/s^2 | {po['before_spike_frames']} | "
              f"{po['after_spike_frames']} |",
              f"| Head contact frames refused (collider) | | {po['head_blocked']} |", "",
              f"Solved again gently (`solve_gently`) for jerks: {po['resolved_gently']} of the {po['jerky_direct']} "
              "clips whose direct solve had some.", "",
              f"Pair requests: {m['pair_requests']}; closed within 1 cm after: {m['pairs_closed_after_le_1cm']} / "
              f"{m['pairs_closed']}.", "",
              f"Over the edit budget (whole-body displacement p95 > {100 * EDIT_MEAN_P95_M:.0f} cm or a body "
              f"> {100 * EDIT_BODY_MAX_M:.0f} cm; flagged for Pass C, not failed): {len(m['over_edit_budget'])} "
              f"clips{': ' + ', '.join(f'`{s}`' for s in m['over_edit_budget']) if m['over_edit_budget'] else ''}.",
              "", "## Clips", "",
              "| Clip | Frames | Float > 2 cm | Agreement | Floor min (cm) | Deepest overlap (cm) | New overlaps | Mean disp p95 (cm) | Max body disp (cm) | Joint change p99 / max (deg) | Body acc RMS before / after |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        b, a, e = r["metrics"]["before"], r["metrics"]["after"], r["metrics"]["edit"]
        lines.append(f"| `{r['stem']}` | {r['frames']} | {b['floating_gt_2cm']} -> {a['floating_gt_2cm']} | "
                     f"{b['agreement']} -> {a['agreement']} | {a['floor_min_cm']} | "
                     f"{b['overlap_max_cm']} -> {a['overlap_max_cm']} | {r['metrics']['new_deep_overlaps']} | "
                     f"{e['mean_disp_cm_p95']} | {e['body_disp_cm_max']} ({e['body_disp_worst']}) | "
                     f"{e['joint_change_deg_p99']} / {e['joint_change_deg_max']} | {e['body_acc_rms_before']} / "
                     f"{e['body_acc_rms_after']} |")
    return "\n".join(lines) + "\n"


def write(labels: dict, rid: str, out_dir: Path, records: list[dict], metrics: dict, motion_dir=ids.SHIPPED_DIR,
          record_root: Path = RECORD_ROOT) -> Path:
    from reference_curation import sources

    out = Path(record_root) / rid
    out.mkdir(parents=True, exist_ok=True)
    head = {"schema_version": SCHEMA_VERSION, "retarget_id": rid}
    (out / "clips.jsonl").write_text("".join(json.dumps({**head, **{k: r[k] for k in RECORD_KEYS},
                                                         "round_trip": r.get("round_trip")}, default=_json_default)
                                             + "\n" for r in records))
    inputs = [labels["dir"] / "holds.yaml", labels["dir"] / "annotations.jsonl", ids.MJCF]
    inputs += [ids.motion_path(r["stem"], motion_dir) for r in records]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "retarget_id": rid,
              "labels_id": labels["id"], "motion_dir": ids.display_path(motion_dir),
              "out_dir": ids.display_path(out_dir), "config": CONFIG,
              "generators": {ids.display_path(p): ids.sha256_file(p) for p in (Path(__file__), Path(sources.__file__),
                                                                            Path(hm.__file__))},
              "motions": {r["stem"]: ids.sha256_file(Path(out_dir) / f"{r['stem']}.motion") for r in records},
              "metrics": metrics}
    (out / "retarget.json").write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
    (out / "summary.md").write_text(summary_markdown(rid, metrics, records))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    import time

    from reference_curation import packets, statics

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="every clip of the labels")
    what.add_argument("--pilot", action="store_true", help="the pilot clips (packets.PILOT_STEMS)")
    what.add_argument("--stem", nargs="+")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--motion-dir", type=Path, default=ids.SHIPPED_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--record-root", type=Path, default=RECORD_ROOT)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        labels = statics.load_labels(args.labels or statics.default_labels_dir())
        stems = ([s for s, _ in labels["clips"]] if args.all else list(packets.PILOT_STEMS) if args.pilot else args.stem)
        kept = [s for s in stems if not has_human(s)]                # no human mesh: nothing to retarget to
        stems = [s for s in stems if s not in kept]
        out_dir, records, failures = run(labels, stems, args.motion_dir, args.workers, args.iters, args.out_root)
        for s in kept:   # the release keeps its shipped motion, so the directory is complete for statics
            import shutil

            shutil.copyfile(ids.motion_path(s, args.motion_dir), out_dir / f"{s}.motion")
        failures += check(records, out_dir)
        metrics = corpus_metrics(labels, out_dir, records, args.motion_dir) if records else {}
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if not records:
        print("retarget: nothing retargeted", file=sys.stderr)
        return 1
    rid = out_dir.name
    rec = write(labels, rid, out_dir, records, metrics, args.motion_dir, args.record_root)
    ex = metrics["exit"]
    print(f"retarget {rid}: {len(records)} clips in {time.time() - start:.0f} s; labelled supports > 2 cm "
          f"{ex['source']['before_over']}/{ex['source']['supports']} -> {ex['source']['after_over']}/{ex['source']['supports']} "
          f"(reachable {ex['source']['after_over_reachable']}/{ex['source']['reachable']}); new overlaps "
          f"{metrics['pooled']['new_overlaps']}; {len(failures)} failures -> {ids.display_path(rec)}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
