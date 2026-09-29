# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact- and balance-aware repair of the arm-balance hold poses (minimal edit, LP-checked).

``expert_revist/contact_balance_investigation/README.MD`` §0: the crow contacts the pose depends
on are labelled but left open *in the reference itself* (Crow -a shin on upper arm 1.8 cm,
Side Crow -c 3.6 cm), and on this plant's mass model the reference COM sits where it cannot
hold: at the edge of the small box-hand polygon (Crow -a +0.8 cm) or 2-4 cm further forward than
the human's measured COP (Firefly 9.0 vs 4.7 cm, wrist-bound LP s* 1.04-1.22). Tracking the
reference therefore asks for a pose the plant cannot hold hands-only.

For every family hold (``extend: true``) of a non-excluded arm balance whose commanded ground set
is hands only, this solves a small inverse-kinematics problem on keyframes of the hold window:

* the commanded supports (hand + wrist box bottom faces) stay exactly where they are;
* labelled leg-on-arm / leg-on-trunk pairs (the Tier-1 "consequential" topology of
  ``notes/Contact_label_consequence.MD`` §6) whose reference gap is 0.5-5 cm are closed to
  0.3 cm;
* the COM goes to the human's measured COP when the reference COM is far from it, pushed at
  least ``--min-margin`` inside the support polygon (the human COP for Crow -a sits 0.7 cm
  inside the sim's hand polygon: the plant's hands are smaller than a person's);
* zones that are not supports may not dip toward the floor;
* everything else changes as little as possible.

The recipe is decided on the exemplar frame and accepted only if the static LP
(``static_hold_lp.py``) does not get worse; pair closure is dropped first if it does. The
correction is applied at keyframes every ``--keyframe-s`` through the hold and blended in and
out over ``--ramp-s``. Every field is rewritten consistently: body transforms by forward
kinematics, ``local_rigid_body_rot``, ``dof_pos`` as the per-joint exponential map (the reset
path), and all velocities recomputed with the converter's routines.

    PYTHONPATH=. python data/scripts/repair_armbalance_holds.py \\
      --manifest data/smpl/expert60/holds_repaired_ftC.yaml \\
      --out-dir data/smpl/yoga_motions_proto_yogi_expert60_ftC_src \\
      --out-manifest data/smpl/expert60/holds_repaired_ftC.yaml \\
      --report expert_revist/ft_c/data/pose_repair.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import static_hold_lp as S  # noqa: E402
from extract_contact_configs import ZONES  # noqa: E402
from make_hold_extended_clips import recompute_velocities  # noqa: E402
from protomotions.components.pose_lib import (  # noqa: E402
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
)
from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix  # noqa: E402

M = S.M
LEG = {"L_FOOT", "R_FOOT", "L_SHANK", "R_SHANK", "L_THIGH", "R_THIGH"}
ARM_OR_TRUNK = {"L_UPPER_ARM", "R_UPPER_ARM", "L_FOREARM", "R_FOREARM", "TRUNK"}
HANDS = {"L_HAND", "R_HAND"}
GOAL_IDS = [S.BODY.index(b) for b in ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")]
DEFAULT_EXCLUDE = ("Cockerel_Pose-b", "Scale_Pose_or_Tolasana_-a")
PRESSURE = REPO / "data/smpl/yoga_motions_proto_yogi_pressure"
PRESSURE_GATED = REPO / "data/smpl/yoga_motions_proto_yogi_pressure_gated"


# --------------------------------------------------------------------------- #
# Geometry helpers (pure numpy)
# --------------------------------------------------------------------------- #
def consequential(pair: str) -> bool:
    """A leg zone resting on an arm or trunk zone: a load path that bypasses joints."""
    a, b = pair.split("+")
    return (a in LEG and b in ARM_OR_TRUNK) or (b in LEG and a in ARM_OR_TRUNK)


def is_shelf(pair: str) -> bool:
    """Shin on an arm: the crow-family shelf the pose is built on. Closed from further away
    than the incidental thigh/trunk proximities, which the label audit found are mostly
    2.5-4.5 cm proximity latches (``notes/Contact_label_consequence.MD`` §4)."""
    a, b = pair.split("+")
    arms = {"L_UPPER_ARM", "R_UPPER_ARM", "L_FOREARM", "R_FOREARM"}
    return (a in {"L_SHANK", "R_SHANK"} and b in arms) or (b in {"L_SHANK", "R_SHANK"} and a in arms)


def _cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, np.asarray(points, float)))
    if len(pts) <= 2:
        return np.array(pts)
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and _cross2(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and _cross2(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def signed_margin(p: np.ndarray, h: np.ndarray) -> float:
    """Distance from ``p`` to the boundary of convex polygon ``h``; + inside, - outside."""
    m = len(h)
    if m < 3:
        return -min(float(np.linalg.norm(p - q)) for q in h)
    inside = all(_cross2(h[i], h[(i + 1) % m], p) >= 0 for i in range(m))
    d = min(_seg_dist(p, h[i], h[(i + 1) % m]) for i in range(m))
    return d if inside else -d


def push_inside(p: np.ndarray, h: np.ndarray, margin: float) -> np.ndarray:
    """Move ``p`` toward the polygon centroid until it is ``margin`` inside (bisection)."""
    if signed_margin(p, h) >= margin:
        return p
    c = h.mean(axis=0)
    if signed_margin(c, h) < margin:
        return c
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if signed_margin(p + mid * (c - p), h) >= margin:
            hi = mid
        else:
            lo = mid
    return p + hi * (c - p)


def best_yaw_dist(frames: np.ndarray, exemplar: np.ndarray) -> np.ndarray:
    """``repair_hold_manifest.best_yaw_dist``: mean per-body distance after a best-fit yaw."""
    a = frames[..., :2]
    b = exemplar[None, :, :2]
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    th = np.arctan2(num, den)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    x = c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] + c * a[..., 1]
    rot = np.stack([x, y, frames[..., 2]], -1)
    return np.linalg.norm(rot - exemplar[None], axis=-1).mean(-1)


def lowest_points(ik: "HoldIK") -> np.ndarray:
    """[B] lowest surface height of every body's collision geom at ``ik.d``'s pose."""
    out = np.zeros(len(S.BODY))
    from scipy.spatial.transform import Rotation as Rot
    for i, b in enumerate(S.BODY):
        g = S.TYPED[b][0]
        bid = i + 1
        R = ik.d.xmat[bid].reshape(3, 3)
        if g["type"] == "box":
            Rg = Rot.from_quat(g["quat"]).as_matrix()
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
            local = g["center"][None] + (Rg @ (signs * g["half"]).T).T
            out[i] = (ik.d.xpos[bid][2] + (R @ local.T)[2]).min()
        elif g["type"] == "capsule":
            z = [(ik.d.xpos[bid] + R @ e)[2] for e in g["seg"]]
            out[i] = min(z) - g["radius"]
        else:
            out[i] = (ik.d.xpos[bid] + R @ g["center"])[2] - g["radius"]
    return out


def seg_seg_closest(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = float(np.clip((b * f - c * e) / den, 0, 1)) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0:
        t, s = 0.0, float(np.clip(-c / a, 0, 1)) if a > 1e-12 else 0.0
    elif t > 1:
        t, s = 1.0, float(np.clip((b - c) / a, 0, 1)) if a > 1e-12 else 0.0
    return p1 + d1 * s, p2 + d2 * t


# --------------------------------------------------------------------------- #
# The IK problem on one frame
# --------------------------------------------------------------------------- #
class HoldIK:
    """Damped Gauss-Newton in MuJoCo's velocity space with analytic Jacobians."""

    def __init__(self, w_reg=1.0, w_root=3.0, w_sup=1000.0, w_pair=30.0, w_com=30.0, w_clear=30.0):
        self.d = mujoco.MjData(M)
        self.w_reg, self.w_root, self.w_sup = w_reg, w_root, w_sup
        self.w_pair, self.w_com, self.w_clear = w_pair, w_com, w_clear
        self.reg = np.r_[np.full(3, w_root), np.full(3, w_reg), np.full(M.nv - 6, w_reg)]

    def _fk(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(M, self.d)
        mujoco.mj_comPos(M, self.d)

    def _jac(self, body_id, point):
        jp = np.zeros((3, M.nv))
        mujoco.mj_jac(M, self.d, jp, None, point, body_id)
        return jp

    def _world(self, body_id, local):
        return self.d.xpos[body_id] + self.d.xmat[body_id].reshape(3, 3) @ local

    def _core(self, body):
        g = S.TYPED[body][0]
        bid = M.body(body).id
        if g["type"] == "capsule":
            return bid, self._world(bid, g["seg"][0]), self._world(bid, g["seg"][1]), g["radius"]
        if g["type"] == "sphere":
            c = self._world(bid, g["center"])
            return bid, c, c, g["radius"]
        raise ValueError(f"{body}: box geoms are not supported as a closure pair")

    def pair_gap(self, ba, bb):
        ia, a0, a1, ra = self._core(ba)
        ib, b0, b1, rb = self._core(bb)
        c1, c2 = seg_seg_closest(a0, a1, b0, b1)
        v = c2 - c1
        dist = float(np.linalg.norm(v))
        return dist - ra - rb, c1, c2, ia, ib, (v / dist if dist > 1e-9 else np.array([0.0, 0.0, 1.0]))

    def residual(self, q, spec, with_jac=True):
        self._fk(q)
        rs, js = [], []
        dq = np.zeros(M.nv)
        mujoco.mj_differentiatePos(M, dq, 1.0, spec["q_ref"], q)
        rs.append(self.reg * dq)
        if with_jac:
            js.append(np.diag(self.reg))
        for bid, local, p0 in spec["supports"]:
            p = self._world(bid, local)
            rs.append(self.w_sup * (p - p0))
            if with_jac:
                js.append(self.w_sup * self._jac(bid, p))
        for ba, bb in spec["pairs"]:
            gap, c1, c2, ia, ib, u = self.pair_gap(ba, bb)
            rs.append(np.array([self.w_pair * 10.0 * (gap - spec["g_target"])]))
            if with_jac:
                js.append(self.w_pair * 10.0 * (u @ (self._jac(ib, c2) - self._jac(ia, c1)))[None])
        if spec.get("com_target") is not None:
            rs.append(self.w_com * 10.0 * (self.d.subtree_com[1][:2] - spec["com_target"]))
            if with_jac:
                jc = np.zeros((3, M.nv))
                mujoco.mj_jacSubtreeCom(M, self.d, jc, 1)
                js.append(self.w_com * 10.0 * jc[:2])
        for bid, local, zmin in spec["clear"]:
            p = self._world(bid, local)
            viol = min(0.0, p[2] - zmin)
            rs.append(np.array([self.w_clear * 10.0 * viol]))
            if with_jac:
                row = self._jac(bid, p)[2] if viol < 0 else np.zeros(M.nv)
                js.append(self.w_clear * 10.0 * row[None])
        r = np.concatenate(rs)
        return (r, np.concatenate(js, axis=0)) if with_jac else (r, None)

    def solve(self, q0, spec, iters=150):
        q = q0.copy()
        r, J = self.residual(q, spec)
        cost = float(r @ r)
        lam = 1e-2
        for _ in range(iters):
            H = J.T @ J + lam * np.eye(M.nv)
            step = -np.linalg.solve(H, J.T @ r)
            qn = q.copy()
            mujoco.mj_integratePos(M, qn, step, 1.0)
            rn, _ = self.residual(qn, spec, with_jac=False)
            cn = float(rn @ rn)
            if cn < cost:
                q, cost, lam = qn, cn, max(lam / 3.0, 1e-6)
                r, J = self.residual(q, spec)
                if np.linalg.norm(step) < 1e-7:
                    break
            else:
                lam *= 10.0
                if lam > 1e6:
                    break
        self._fk(q)
        return q


# --------------------------------------------------------------------------- #
def frame_supports(ik: HoldIK, q, zones):
    """(body id, local point, world point) for the bottom face of every support zone's boxes."""
    ik._fk(q)
    out = []
    for z in zones:
        for b in ZONES[z]:
            g = S.TYPED[b][0]
            if g["type"] != "box":
                raise ValueError(f"support zone {z} body {b} is not a box")
            bid = M.body(b).id
            R = ik.d.xmat[bid].reshape(3, 3)
            from scipy.spatial.transform import Rotation as Rot
            Rg = Rot.from_quat(g["quat"]).as_matrix()
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
            local = g["center"][None] + (Rg @ (signs * g["half"]).T).T
            world = ik.d.xpos[bid] + (R @ local.T).T
            for k in np.argsort(world[:, 2])[:4]:
                out.append((bid, local[k].copy(), world[k].copy()))
    return out


def frame_clearance(ik: HoldIK, q, support_zones, floor_band=0.10):
    """Low points of every non-support zone that sits within ``floor_band`` of the floor:
    it may not go lower than min(its height, 3 cm)."""
    ik._fk(q)
    out = []
    for z, bodies in ZONES.items():
        if z in support_zones:
            continue
        for b in bodies:
            g = S.TYPED[b][0]
            bid = M.body(b).id
            if g["type"] == "box":
                from scipy.spatial.transform import Rotation as Rot
                Rg = Rot.from_quat(g["quat"]).as_matrix()
                signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
                cands = g["center"][None] + (Rg @ (signs * g["half"]).T).T
                offs = np.zeros(len(cands))
            elif g["type"] == "capsule":
                cands, offs = np.stack([g["seg"][0], g["seg"][1]]), np.full(2, g["radius"])
            else:
                cands, offs = g["center"][None], np.array([g["radius"]])
            for local, off in zip(cands, offs):
                z0 = ik._world(bid, local)[2] - off
                if z0 < floor_band:
                    # the material point is the geom's centre-line point; its low surface is `off` below
                    out.append((bid, local.copy(), min(z0, 0.03) + off))
    return out


def pressure_cop(stem: str, pos: np.ndarray):
    """``(cop [T,2], valid [T])`` from the MOYO port, or ``(None, None)``; refuses a misaligned clip."""
    for path, gate_col in ((PRESSURE_GATED / f"{stem}.motion", 2), (PRESSURE / f"{stem}.motion", 0)):
        if not path.exists():
            continue
        pm = torch.load(str(path), map_location="cpu", weights_only=False)
        if "ground_reaction" not in pm:
            continue
        if pm["rigid_body_pos"].shape[0] != pos.shape[0] or \
                float((pm["rigid_body_pos"].numpy() - pos).__abs__().max()) > 1e-4:
            raise ValueError(f"{stem}: the pressure clip's kinematics differ from the source")
        v = pm["ground_reaction_valid"].numpy()
        valid = v[:, 0] >= 0.9
        if v.shape[1] > gate_col:
            valid |= v[:, gate_col] >= 0.9
        return pm["ground_reaction"].numpy()[:, 1:3], valid
    return None, None


def plan_com_target(com, cop, hull_xy, min_margin, cop_far_m, lp_hands, cop_lp_min):
    """The COM target and why: keep, push inside, or move to the human COP.

    The COM is moved to the human COP only when the reference hold is near or past the
    plant's torque limits (``lp_hands > cop_lp_min``, the Firefly case: the sim's COM sits
    4 cm further forward than the human's and overloads the wrists). A hold the plant can
    already carry keeps its COM -- policies hold several of them today and a moved
    reference would ask them to change a working balance.
    """
    m = signed_margin(com, hull_xy)
    if m < min_margin:
        base = cop if (cop is not None and signed_margin(cop, hull_xy) >= min_margin) else com
        return push_inside(base, hull_xy, min_margin), "margin"
    if cop is not None and lp_hands > cop_lp_min and float(np.linalg.norm(com - cop)) > cop_far_m:
        return push_inside(cop, hull_xy, min_margin), "cop"
    return None, "keep"


# --------------------------------------------------------------------------- #
def repair_clip(clip, args, ik, kin):
    src = torch.load(clip["source"], map_location="cpu", weights_only=False)
    fps = int(src["fps"])
    pos = src["rigid_body_pos"].double().numpy()
    rot = src["rigid_body_rot"].double().numpy()
    T = pos.shape[0]
    cop, cop_valid = pressure_cop(clip["stem"], pos)
    reports = []
    # per-frame velocity-space correction and its blend weight
    delta = np.zeros((T, M.nv))
    weight = np.zeros(T)
    q_orig = np.zeros((T, M.nq))
    for f in range(T):
        S.set_pose(pos[f], rot[f])
        q_orig[f] = S.D.qpos.copy()

    for h in clip["holds"]:
        ground = sorted(p.split(":")[0] for p in h["pairs"] if p.endswith(":G"))
        if not h.get("extend") or set(ground) != HANDS:
            continue
        f0, fh, f1 = int(h["frame_start"]), int(h["frame_hold"]), int(h["frame_end"])
        # Correct only the quasi-static part of the window. The manifest trims holds after
        # the exemplar (repair_hold_manifest.py) but not before it, so a family window can
        # open on the entry itself (Firefly -b: legs still lifting 7 s before the exemplar).
        drift = best_yaw_dist(pos[f0:fh + 1][:, GOAL_IDS] - pos[f0:fh + 1][:, :1],
                              pos[fh][GOAL_IDS] - pos[fh][:1])
        over = np.nonzero(drift > args.window_drift_m)[0]
        f0 = f0 + (int(over.max()) + 1 if len(over) else 0)
        cand_pairs = [p for p in h["pairs"] if ":G" not in p and consequential(p)]
        # closest member bodies at the exemplar; capsule/sphere only
        pairs = []
        ik._fk(q_orig[fh])
        for p in cand_pairs:
            za, zb = p.split("+")
            best = None
            for ba in ZONES[za]:
                for bb in ZONES[zb]:
                    if "box" in (S.TYPED[ba][0]["type"], S.TYPED[bb][0]["type"]):
                        continue
                    gap = ik.pair_gap(ba, bb)[0]
                    if best is None or gap < best[2]:
                        best = (ba, bb, gap)
            limit = args.close_max_m if is_shelf(p) else args.close_max_other_m
            if best is not None and args.close_min_m < best[2] <= limit:
                pairs.append((p, best[0], best[1], best[2]))

        def spec_for(f, with_pairs, mode):
            sup = frame_supports(ik, q_orig[f], ground)
            hull_xy = hull(np.array([p0[:2] for _, _, p0 in sup]))
            ik._fk(q_orig[f])
            com = ik.d.subtree_com[1][:2].copy()
            c = None
            if cop is not None:
                window = np.arange(max(f0, f - 6), min(f1, f + 6) + 1)
                ok = window[cop_valid[window]]
                if len(ok):
                    c = np.median(cop[ok], axis=0)
            if mode == "keep":
                target = None
            elif mode == "margin":
                base = c if (c is not None and signed_margin(c, hull_xy) >= args.min_margin) else com
                target = push_inside(base, hull_xy, args.min_margin) if signed_margin(com, hull_xy) < args.min_margin else None
            else:  # "cop"
                target = push_inside(c, hull_xy, args.min_margin) if c is not None else None
            return dict(
                q_ref=q_orig[f], supports=[(b, l, p0) for b, l, p0 in sup],
                pairs=[(ba, bb) for _, ba, bb, _ in pairs] if with_pairs else [],
                g_target=args.target_gap_m, com_target=target,
                clear=frame_clearance(ik, q_orig[f], set(ground)),
            ), hull_xy, com, c

        # ---- decide the recipe on the exemplar -------------------------- #
        sup0 = frame_supports(ik, q_orig[fh], ground)
        hull0 = hull(np.array([p0[:2] for _, _, p0 in sup0]))
        ik._fk(q_orig[fh])
        com0 = ik.d.subtree_com[1][:2].copy()
        _, _, _, cop0 = spec_for(fh, False, "keep")
        lp_pairs = [p for p in cand_pairs]
        before_g = S.solve(pos[fh], rot[fh], ground)
        before_p = S.solve(pos[fh], rot[fh], ground, lp_pairs) if lp_pairs else before_g
        lp_hands0 = before_g["s_star"] if before_g.get("feasible") else math.inf
        target0, mode = plan_com_target(com0, cop0, hull0, args.min_margin, args.cop_far_m,
                                        lp_hands0, args.cop_lp_min)

        def attempt(with_pairs):
            spec, _, _, _ = spec_for(fh, with_pairs, mode)
            if not spec["pairs"] and spec["com_target"] is None:
                return None
            q = ik.solve(q_orig[fh], spec)
            new_pos = ik.d.xpos[1:].copy()
            new_rot = np.array([[x[1], x[2], x[3], x[0]] for x in ik.d.xquat[1:]])
            after_g = S.solve(new_pos, new_rot, ground)
            after_p = S.solve(new_pos, new_rot, ground, lp_pairs) if lp_pairs else after_g
            return q, new_pos, new_rot, after_g, after_p, spec

        def s_of(r):
            return r["s_star"] if r.get("feasible") else math.inf

        chosen, reason = None, "no edit needed"
        for with_pairs in ([True, False] if pairs else [False]):
            res = attempt(with_pairs)
            if res is None:
                continue
            q, new_pos, new_rot, after_g, after_p, spec = res
            ok = (s_of(after_p) <= s_of(before_p) + args.lp_slack) or (s_of(after_p) <= args.lp_ok)
            disp = np.linalg.norm(new_pos - pos[fh], axis=1)
            dqx = np.zeros(M.nv)
            mujoco.mj_differentiatePos(M, dqx, 1.0, q_orig[fh], q)
            small = disp.mean() <= args.max_mean_disp_m and np.degrees(np.abs(dqx[6:]).max()) <= args.max_joint_deg
            if ok and small:
                chosen = (with_pairs, res)
                reason = "accepted"
                break
            reason = (f"LP worse ({s_of(before_p):.2f} -> {s_of(after_p):.2f})" if not ok else
                      f"edit too large (mean {100 * disp.mean():.1f} cm, "
                      f"max joint {np.degrees(np.abs(dqx[6:]).max()):.0f} deg)")
        rec = dict(stem=clip["stem"], hold=h["name"], t=[h["t_start"], h["t_hold"], h["t_end"]],
                   mode=mode, pairs_labelled=cand_pairs,
                   pairs_to_close={p: round(100 * g, 1) for p, _, _, g in pairs},
                   com_margin_before_cm=round(100 * signed_margin(com0, hull0), 1),
                   cop_margin_cm=None if cop0 is None else round(100 * signed_margin(cop0, hull0), 1),
                   lp_hands_before=round(s_of(before_g), 3), lp_pairs_before=round(s_of(before_p), 3),
                   decision=reason)
        if chosen is None:
            reports.append(rec)
            continue
        with_pairs, (q, new_pos, new_rot, after_g, after_p, spec) = chosen
        ik._fk(q)
        com1 = ik.d.subtree_com[1][:2].copy()
        disp = np.linalg.norm(new_pos - pos[fh], axis=1)
        dq_ex = np.zeros(M.nv)
        mujoco.mj_differentiatePos(M, dq_ex, 1.0, q_orig[fh], q)
        rec.update(closed=bool(with_pairs),
                   gaps_after_cm={p: round(100 * ik.pair_gap(ba, bb)[0], 1) for p, ba, bb, _ in pairs},
                   com_margin_after_cm=round(100 * signed_margin(com1, hull0), 1),
                   com_move_cm=round(100 * float(np.linalg.norm(com1 - com0)), 1),
                   lp_hands_after=round(s_of(after_g), 3), lp_pairs_after=round(s_of(after_p), 3),
                   body_disp_cm_mean=round(100 * float(disp.mean()), 1),
                   body_disp_cm_max=round(100 * float(disp.max()), 1),
                   body_disp_top=[(S.BODY[i], round(100 * float(disp[i]), 1)) for i in np.argsort(-disp)[:3]],
                   joint_change_deg_max=round(float(np.degrees(np.abs(dq_ex[6:]).max())), 1),
                   support_drift_mm=round(1000 * max(float(np.linalg.norm(ik._world(b, l) - p0))
                                                     for b, l, p0 in spec["supports"]), 2))
        # ---- apply at keyframes through the hold, warm-started ------------ #
        step = max(1, int(round(args.keyframe_s * fps)))
        keys = sorted(set([f0, f1, fh] + list(range(f0, f1 + 1, step))))
        dks = {}
        q_prev = q
        for f in sorted(keys, key=lambda k: abs(k - fh)):
            spec_f, _, _, _ = spec_for(f, with_pairs, mode)
            start = q_orig[f].copy()
            # warm start: the exemplar's correction carried to this frame
            dq0 = np.zeros(M.nv)
            mujoco.mj_differentiatePos(M, dq0, 1.0, q_orig[fh], q_prev)
            mujoco.mj_integratePos(M, start, dq0, 1.0)
            qf = ik.solve(start, spec_f) if (spec_f["pairs"] or spec_f["com_target"] is not None) else q_orig[f]
            d = np.zeros(M.nv)
            mujoco.mj_differentiatePos(M, d, 1.0, q_orig[f], qf)
            dks[f] = d
        ks = np.array(sorted(dks))
        dk = np.stack([dks[k] for k in ks])
        ramp = int(round(args.ramp_s * fps))
        for f in range(max(0, f0 - ramp), min(T, f1 + ramp + 1)):
            if f <= ks[0]:
                d, w = dk[0], (1.0 if f >= f0 else max(0.0, 1.0 - (f0 - f) / max(ramp, 1)))
            elif f >= ks[-1]:
                d, w = dk[-1], (1.0 if f <= f1 else max(0.0, 1.0 - (f - f1) / max(ramp, 1)))
            else:
                j = int(np.searchsorted(ks, f))
                a = (f - ks[j - 1]) / (ks[j] - ks[j - 1])
                d, w = (1 - a) * dk[j - 1] + a * dk[j], 1.0
            w = 0.5 - 0.5 * math.cos(math.pi * w)   # smooth ramp
            if w > weight[f]:
                delta[f], weight[f] = d, w
        rec["keyframes"] = len(ks)
        reports.append(rec)

    if not weight.any():
        return None, reports
    # ---- rebuild the clip ------------------------------------------------- #
    new_pos = pos.copy()
    new_rot = rot.copy()
    # Floor safety: the largest blend weight per frame that keeps every body at or above its
    # own grounding, then a running minimum (+-15 frames) and a moving average (+-10) so the
    # weight profile has no steps (a step is a velocity spike the tracking reward would feel).
    # The average of a +-15 running minimum over +-10 never exceeds any frame's admissible value.
    frames = np.nonzero(weight)[0]
    floors = {}
    admissible = weight.copy()
    for f in frames:
        ik._fk(q_orig[f])
        floors[f] = np.minimum(lowest_points(ik), 0.005) - 0.002
        w = weight[f]
        for _ in range(14):
            qf = q_orig[f].copy()
            mujoco.mj_integratePos(M, qf, w * delta[f], 1.0)
            ik._fk(qf)
            if (lowest_points(ik) >= floors[f]).all():
                break
            w *= 0.7
        else:
            w = 0.0
        admissible[f] = w
    run_min = np.array([admissible[max(0, f - 15): f + 16].min() for f in range(T)])
    smooth = np.array([run_min[max(0, f - 10): f + 11].mean() for f in range(T)])
    final = np.minimum(smooth, admissible)
    shrunk = 0
    for f in frames:
        w = final[f]
        qf = q_orig[f].copy()
        mujoco.mj_integratePos(M, qf, w * delta[f], 1.0)
        ik._fk(qf)
        if not (lowest_points(ik) >= floors[f]).all():   # cannot happen; belt and braces
            w = 0.0
            ik._fk(q_orig[f])
        shrunk += int(w < weight[f] - 1e-6)
        new_pos[f] = ik.d.xpos[1:]
        new_rot[f] = np.array([[x[1], x[2], x[3], x[0]] for x in ik.d.xquat[1:]])
    if reports:
        reports[-1]["floor_shrunk_frames"] = shrunk
    out = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in src.items()}
    rb_pos = torch.tensor(new_pos, dtype=src["rigid_body_pos"].dtype)
    rb_rot = torch.tensor(new_rot, dtype=src["rigid_body_rot"].dtype)
    world = quaternion_to_matrix(rb_rot.double(), w_last=True)
    local = world.clone()
    for b in range(1, world.shape[1]):
        local[:, b] = world[:, S.PARENT[b]].transpose(-1, -2) @ world[:, b]
    qpos = extract_qpos_from_transforms(kin, rb_pos[:, 0].double(), local, multi_dof_decomposition_method="exp_map")
    out["rigid_body_pos"] = rb_pos
    out["rigid_body_rot"] = rb_rot
    out["local_rigid_body_rot"] = matrix_to_quaternion(local, w_last=True).to(src["local_rigid_body_rot"].dtype)
    out["dof_pos"] = qpos[:, 7:].to(src["dof_pos"].dtype)
    fresh = recompute_velocities(out, fps)
    for k, v in fresh.items():
        out[k] = v.to(src[k].dtype)
    # consistency: the reset path (exp-map dof_pos -> FK) must land on the new bodies
    root_pos, jr = extract_transforms_from_qpos(
        kin, torch.cat([rb_pos[:, :1, :].squeeze(1).double(),
                        torch.tensor(np.array([[q[3], q[0], q[1], q[2]] for q in new_rot[:, 0]])),
                        out["dof_pos"].double()], dim=1), qpos_is_exp_map_on_3dof_joints=True)
    wr = jr.clone()
    for b in range(1, wr.shape[1]):
        wr[:, b] = wr[:, S.PARENT[b]] @ jr[:, b]
    rot_err = float((wr - world).abs().max())
    if rot_err > 1e-4:
        raise ValueError(f"{clip['stem']}: exp-map dof_pos does not reproduce the repaired rotations ({rot_err})")
    return out, reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--min-margin", type=float, default=0.03, help="COM margin inside the hand polygon (m)")
    ap.add_argument("--cop-far-m", type=float, default=0.03,
                    help="move the COM to the human COP when they are further apart than this")
    ap.add_argument("--cop-lp-min", type=float, default=0.9,
                    help="...and only when the reference hold's hands-only LP exceeds this")
    ap.add_argument("--close-min-m", type=float, default=0.005)
    ap.add_argument("--close-max-m", type=float, default=0.05, help="shin-on-arm shelf pairs")
    ap.add_argument("--close-max-other-m", type=float, default=0.025, help="other leg-on-arm/trunk pairs")
    ap.add_argument("--max-mean-disp-m", type=float, default=0.06, help="reject larger edits")
    ap.add_argument("--max-joint-deg", type=float, default=25.0, help="reject larger edits")
    ap.add_argument("--target-gap-m", type=float, default=0.003)
    ap.add_argument("--lp-slack", type=float, default=0.02)
    ap.add_argument("--lp-ok", type=float, default=0.6)
    ap.add_argument("--keyframe-s", type=float, default=0.2)
    ap.add_argument("--window-drift-m", type=float, default=0.10,
                    help="edit only the part of the hold within this 6-body drift of the exemplar")
    ap.add_argument("--ramp-s", type=float, default=0.5)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    manifest = yaml.safe_load(open(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kin = extract_kinematic_info(str(S.MJCF))
    ik = HoldIK()
    reports, new_manifest = [], copy.deepcopy(manifest)
    for clip, new_clip in zip(manifest["clips"], new_manifest["clips"]):
        if clip["group"] != "arm_balance" or any(x in clip["stem"] for x in args.exclude):
            continue
        if args.only and clip["stem"] not in args.only:
            continue
        motion, rec = repair_clip(clip, args, ik, kin)
        reports += rec
        for r in rec:
            print(json.dumps({k: v for k, v in r.items() if k not in ("pairs_labelled",)}))
        if motion is None:
            continue
        path = out_dir / f"{clip['stem']}.motion"
        torch.save(motion, path)
        new_clip["source"] = str(path.resolve())
        new_clip["pose_repair"] = {"from": clip["source"], "report": str(Path(args.report).resolve())}
    new_manifest["pose_repair"] = {k: v for k, v in vars(args).items()}
    with open(args.out_manifest, "w") as f:
        yaml.safe_dump(new_manifest, f, sort_keys=False)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(reports, indent=1))
    print(f"repaired {sum(1 for c in new_manifest['clips'] if 'pose_repair' in c)} clips; "
          f"report {args.report}; manifest {args.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
