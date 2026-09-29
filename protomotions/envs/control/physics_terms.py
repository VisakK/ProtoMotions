# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics terms for expert60 fine-tune C (``expert_revist/ft_c/README.MD``).

Two trained terms and three weight-0 diagnostics, all fed by the offline tables that
``data/scripts/build_physics_tables.py`` writes next to the hold graph:

* **Swing-gated unloaded-limb penalty.** Outside commanded holds, the terrain-filtered ground
  load on every zone the reference is swinging (a limb the human lifted and moved, or one the
  mat says carried nothing while moving), averaged over ``tau`` before a saturating clamp --
  the unwanted-support term's pricing (``support_penalty.ChargedLoadEMA``), extended from holds
  to transitions. It prices the drag measured in ``expert_revist/contact_balance_investigation``
  §2: Warrior II's stepping foot carried 0 N on the mat while lifting only 2.5-4.3 cm, and the
  policy slid it along the floor at 0.8-2.1 m/s under up to 87 N.
* **Commanded-support lean penalty.** Inside a commanded hold whose ground set is hands (and
  possibly forearms and head) with no foot, the COM must sit at least ``min_margin`` inside the
  polygon of the *commanded* supports' contact patches -- measured on the policy's own body.
  One-sided and linear in the shortfall, so it never pulls a lifted policy back, and it gives a
  gradient toward the lean before the foot comes up: the feet-down arm balances sit 1-8 cm
  outside, the lifted ones and every working inversion 4.7-6.4 cm inside.
* Diagnostics: loaded slip power (ground force x slowest-corner slip speed on feet and hands),
  lean error against the human COP where the mat measured it, and the load through the
  commanded hold's leg-on-arm / leg-on-trunk pairs.

Everything is pure tensor code; ``ContactGraphControl`` owns the tables and the EMA state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from protomotions.utils.rotations import quat_rotate

BOX, CAPSULE, SPHERE = 0, 1, 2
_BOX_SIGNS = torch.tensor(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
class PhysicsTables:
    """The offline tables, validated against the environment's motion library and body order."""

    def __init__(self, path: str, motion_names: List[str], body_names: List[str], device):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if list(payload["motion_names"]) != list(motion_names):
            raise ValueError(
                f"{path} was built for a different motion library "
                f"({len(payload['motion_names'])} vs {len(motion_names)} clips)"
            )
        if list(payload["body_names"]) != list(body_names):
            raise ValueError(f"{path} body order {payload['body_names']} != robot {body_names}")
        self.fps = float(payload["fps"])
        self.zone_order = list(payload["zone_order"])
        self.zone_bodies = {z: list(v) for z, v in payload["zone_bodies"].items()}
        self.swing = payload["swing"].to(device=device, dtype=torch.bool)          # [M, T, Z]
        self.swing_len = payload["swing_len"].to(device)                              # [M]
        for key in ("seg_cop_rel", "seg_cop_valid", "seg_com_rel", "seg_lean_gate",
                    "seg_pair_consequential", "seg_zone_share", "seg_share_valid"):
            setattr(self, key, payload[key].to(device))
        for key in ("body_mass", "body_com_local", "geom_type", "box_center", "box_half",
                    "box_quat", "cap_a", "cap_b", "radius", "sph_center"):
            setattr(self, key, payload[key].to(device))
        self.body_names = list(body_names)

    def swing_at(self, motion_ids: Tensor, motion_times: Tensor) -> Tensor:
        """``[E, Z]`` bool swing labels at each env's clip frame."""
        frame = torch.round(motion_times * self.fps).long()
        frame = torch.minimum(frame.clamp(min=0), self.swing_len[motion_ids] - 1)
        return self.swing[motion_ids, frame]


# --------------------------------------------------------------------------- #
# Geometry kernels
# --------------------------------------------------------------------------- #
def whole_body_com(pos: Tensor, rot: Tensor, mass: Tensor, com_local: Tensor) -> Tensor:
    """``[E, 3]`` COM from link poses (``pos [E,B,3]``, ``rot [E,B,4]`` xyzw) and the mass model."""
    E, B = pos.shape[:2]
    offset = quat_rotate(rot.reshape(-1, 4), com_local.expand(E, B, 3).reshape(-1, 3), w_last=True)
    body_com = pos + offset.view(E, B, 3)
    return (body_com * mass.view(1, B, 1)).sum(dim=1) / mass.sum()


def patch_points(pos: Tensor, rot: Tensor, tables: PhysicsTables, bodies: List[int]) -> Tensor:
    """``[E, n, 3]`` candidate contact points of ``bodies`` (fixed count per body).

    Box: its 4 lowest corners; capsule: both ends lowered by the radius; sphere: its bottom --
    the selection ``build_physics_tables.patch_points`` uses offline.
    """
    E = pos.shape[0]
    out = []
    signs = _BOX_SIGNS.to(pos.device, pos.dtype)
    for i in bodies:
        p, q = pos[:, i], rot[:, i]
        t = int(tables.geom_type[i])
        if t == BOX:
            local = tables.box_center[i] + quat_rotate(
                tables.box_quat[i].expand(8, 4), signs * tables.box_half[i], w_last=True
            )
            world = quat_rotate(
                q.unsqueeze(1).expand(E, 8, 4).reshape(-1, 4),
                local.unsqueeze(0).expand(E, 8, 3).reshape(-1, 3), w_last=True,
            ).view(E, 8, 3) + p.unsqueeze(1)
            low = world[..., 2].argsort(dim=1)[:, :4]
            out.append(torch.gather(world, 1, low.unsqueeze(-1).expand(E, 4, 3)))
        elif t == CAPSULE:
            for e in (tables.cap_a[i], tables.cap_b[i]):
                w = quat_rotate(q, e.expand(E, 3), w_last=True) + p
                w = torch.cat([w[:, :2], w[:, 2:] - tables.radius[i]], dim=-1)
                out.append(w.unsqueeze(1))
        else:
            w = quat_rotate(q, tables.sph_center[i].expand(E, 3), w_last=True) + p
            w = torch.cat([w[:, :2], w[:, 2:] - tables.radius[i]], dim=-1)
            out.append(w.unsqueeze(1))
    return torch.cat(out, dim=1)


def polygon_margin(points: Tensor, valid: Tensor, p: Tensor) -> Tensor:
    """Signed distance of ``p [E,2]`` to the convex hull of ``points [E,N,2]`` (``valid [E,N]``).

    Positive inside. A directed pair (i, j) is a hull edge when every other valid point lies on
    its left; the margin is the minimum signed distance of ``p`` to those edge lines (exact
    inside, a lower bound on the violation outside -- continuous and monotone, which is what a
    penalty needs). Rows with fewer than two distinct valid points fall back to minus the
    distance to the nearest one.
    """
    E, N, _ = points.shape
    a = points.unsqueeze(2)                          # [E, N, 1, 2] edge start
    b = points.unsqueeze(1)                          # [E, 1, N, 2] edge end
    ab = b - a                                       # [E, N, N, 2]
    length = ab.norm(dim=-1)                         # [E, N, N]
    # side of every point k relative to edge (i, j): cross(ab, pk - a)
    pk = points.view(E, 1, 1, N, 2)
    rel = pk - a.unsqueeze(3)                        # [E, N, 1, N, 2] -> broadcast over j
    cross = ab.unsqueeze(3)[..., 0] * rel[..., 1] - ab.unsqueeze(3)[..., 1] * rel[..., 0]  # [E,N,N,N]
    tol = -1e-6 * (1.0 + length.unsqueeze(-1))
    others_left = ((cross >= tol) | ~valid.view(E, 1, 1, N)).all(dim=-1)                # [E, N, N]
    edge = others_left & valid.unsqueeze(2) & valid.unsqueeze(1) & (length > 1e-6)
    rp = p.view(E, 1, 1, 2) - a                                                          # [E, N, 1->N, 2]
    signed = (ab[..., 0] * rp[..., 1] - ab[..., 1] * rp[..., 0]) / length.clamp(min=1e-9)
    big = torch.finfo(p.dtype).max
    margin = torch.where(edge, signed, torch.full_like(signed, big)).flatten(1).min(dim=1).values
    nearest = torch.where(valid, (points - p.unsqueeze(1)).norm(dim=-1), torch.full_like(valid, big, dtype=p.dtype))
    fallback = -nearest.min(dim=1).values
    return torch.where(edge.flatten(1).any(dim=1), margin, fallback)


def corner_slip_speed(pos: Tensor, rot: Tensor, vel: Tensor, ang_vel: Tensor,
                      tables: PhysicsTables, bodies: List[int]) -> Tensor:
    """``[E]`` slowest horizontal speed over the bottom-face corners of the zone's box bodies.

    ``vel`` is the simulator's per-body COM velocity (IsaacLab aliases ``body_lin_vel_w`` to the
    COM), so a corner moves at ``v_com + w x (corner - com)``. A pivot about any corner reads ~0,
    a dragged foot reads its drag speed.
    """
    E = pos.shape[0]
    best = None
    signs = _BOX_SIGNS.to(pos.device, pos.dtype)
    for i in bodies:
        if int(tables.geom_type[i]) != BOX:
            continue
        p, q = pos[:, i], rot[:, i]
        local = tables.box_center[i] + quat_rotate(tables.box_quat[i].expand(8, 4), signs * tables.box_half[i], w_last=True)
        world = quat_rotate(q.unsqueeze(1).expand(E, 8, 4).reshape(-1, 4),
                            local.unsqueeze(0).expand(E, 8, 3).reshape(-1, 3), w_last=True).view(E, 8, 3) + p.unsqueeze(1)
        com = quat_rotate(q, tables.body_com_local[i].expand(E, 3), w_last=True) + p
        low = world[..., 2].argsort(dim=1)[:, :4]
        corners = torch.gather(world, 1, low.unsqueeze(-1).expand(E, 4, 3))
        v = vel[:, i].unsqueeze(1) + torch.linalg.cross(
            ang_vel[:, i].unsqueeze(1).expand(E, 4, 3), corners - com.unsqueeze(1), dim=-1)
        s = v[..., :2].norm(dim=-1).min(dim=1).values
        best = s if best is None else torch.minimum(best, s)
    return best if best is not None else torch.zeros(E, device=pos.device, dtype=pos.dtype)


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #
def swing_charged_load(ground_forces: Optional[Tensor], swing: Tensor, zone_matrix: Tensor,
                       gate: Tensor) -> Tensor:
    """``[E]`` newtons of terrain-filtered load on swinging zones, zero where ``gate`` is off."""
    if ground_forces is None:
        return torch.zeros(swing.shape[0], device=swing.device)
    fz = ground_forces[..., 2].clamp_min(0.0)
    load = fz @ zone_matrix.to(fz.dtype).t()                  # [E, Z]
    return (load * swing.to(load.dtype)).sum(dim=-1) * gate.to(load.dtype)


def lean_shortfall(margin: Tensor, gate: Tensor, min_margin: float, scale: float) -> Tensor:
    """``[E]`` in [0, 1]: how far the COM is from being ``min_margin`` inside the support."""
    return ((min_margin - margin) / scale).clamp(0.0, 1.0) * gate.to(margin.dtype)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def clip_drag(pos: Tensor, rot: Tensor, vel: Tensor, ang_vel: Tensor, ground_forces: Tensor,
              tables: PhysicsTables, zone_bodies: List[List[int]], swing: Optional[Tensor], dt: float,
              load_n: float = 50.0, slip_mps: float = 0.10, mu: float = 0.75) -> Dict[str, float]:
    """Loaded sliding of feet/hands over one rollout (``[T, B, ...]`` per-frame body state).

    The investigation's drag (``contact_balance_investigation/drag_corpus.py``): a zone drags on
    a frame when its terrain-filtered load exceeds ``load_n`` and its slowest bottom corner slides
    faster than ``slip_mps``; ``drag_J`` integrates ``mu * load * slip`` over those frames (the
    friction work at the effective ground friction), ``drag_s`` counts their time, and
    ``drag_J_swing`` keeps the frames where the reference is swinging that zone (``swing
    [T, len(zone_bodies)]``, the swing term's own labels) -- the part the swing term prices.
    """
    fz = ground_forces[..., 2].clamp_min(0.0)
    out = {"drag_s": 0.0, "drag_J": 0.0, "drag_J_swing": 0.0}
    for z, bodies in enumerate(zone_bodies):
        load = fz[:, bodies].sum(-1)
        speed = corner_slip_speed(pos, rot, vel, ang_vel, tables, bodies)
        mask = (load > load_n) & (speed > slip_mps)
        work = mu * load * speed * mask.to(load.dtype) * dt
        out["drag_s"] += float(mask.sum()) * dt
        out["drag_J"] += float(work.sum())
        if swing is not None:
            out["drag_J_swing"] += float((work * swing[:, z].to(work.dtype)).sum())
    return out
