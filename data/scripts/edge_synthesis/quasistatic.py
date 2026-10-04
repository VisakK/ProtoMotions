"""Card T2 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: the quasi-static generator (edges E1, E3, E4, B2).

A quasi-static edge keeps its COM over the planted hands throughout, so it is generated kinematically and certified
by statics, then executed in MuJoCo by T3's planner (``edge_mppi`` with ``--reference``) as the dynamic check.

1. **Endpoints.** S and D from ``edges.json``; D is moved rigidly so its hands land on S's (``sketch.endpoints``).
2. **Keyframe drafts** (``drafts``), 2-4 per edge, written as a yoga teacher would describe the passage and built
   from the endpoints' own joint angles: a *hybrid* takes some bodies' local rotations from one exemplar and the rest
   from another (the press's tuck is the handstand's arms, shoulders and trunk with the crow's folded legs), then the
   root is placed so the hands stay where S has them. The solver realises them; statics judges them.
3. **The anchor** is the drafts joined at 60 fps by minimum-jerk slerp on the edge's phase clock (``sketch.Sketch``
   with the hands as the anchor), in PhysX's exp-map coordinates.
4. **The problem** is ``retarget_v2``'s objective (Levenberg-Marquardt, analytic Jacobians, one banded solve per
   step; the joint box a hard bound, every one of the plant's 253 body pairs guarded, SAT depth for box pairs, jerk
   continuation) with every human-evidence input replaced by the edge's schedule (``build_problem``):

   * ``support``/``flat``: on every frame each planted zone rests its resting face (the box face whose normal points
     most nearly down; the head sphere's bottom) on the floor, palms flat;
   * ``off``: every other zone whose state is known at both ends stays ``OFF_FLOOR_M`` up;
   * ``vel``: a planted point does not move horizontally between frames (no slide);
   * ``pair``: a brace the schedule closes is closed (``retarget.PAIR_TARGET_M``), and a brace it opens keeps
     ``OPEN_GAP_M`` (ramped over ``OPEN_RAMP_S`` from its break), through the penetration guard's per-pair floor;
   * ``balance``: on every frame with three or more planted points the COM lies max(1.5 cm, 25 % of the support
     hull's inradius) inside that hull;
   * ``anchor``/``ends``/``dof``/``smooth``: stay near the drafts, smoothly.
5. **Certification** (``certify``): statics (``statics.analyse``, gated: ground points within 2 cm, pairs within 1
   cm) on every ``STATICS_EVERY``-th frame with that frame's contact set -- ``s* <= 1`` is the certificate, ``s* > 1``
   a negative with its binding joint -- and the card's contract: box, floor, new overlaps against the endpoints,
   COM margin, brace closure at a crow end, acceleration spikes.

Everything plant-dependent runs inside ``retarget_v2.on_plant("v2")``; solves run single-threaded (BLAS threads make
``solveh_banded`` ~1000x slower, BodyFix Step 3).

CLI::

    PYTHONPATH=.:data/scripts python -m edge_synthesis.quasistatic --edge E1 [--timing mid]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from extract_contact_configs import ZONE_ORDER, ZONES
from edge_synthesis import gpu_guard
from edge_synthesis import sketch as SK
from reference_curation import ids

F64 = torch.float64
FPS = 60
STATICS_EVERY = 3
OPEN_GAP_M = 0.02          # a brace the schedule opens keeps this gap (the labels' 2 cm float threshold)
OPEN_RAMP_S = 0.3          # ... reached this long after its break
PLANTED_BAND_M = 0.03      # a box of a planted zone rests when its resting face is within this of the zone's lowest
PRELOAD_S = 0.5            # a breaking support is unloaded (out of the balance hull) this long before its break,
PRELOAD_MARGIN_M = 0.04    # ... with the COM this deep inside the remaining hull: inside with E4's 1.2 cm margin the
                           # wrists still bound at the head's break (s* 1.10) -- the hull is necessary, not sufficient
PEN_WEIGHT = 100.0         # the penetration guard, 10x retarget_v2's (E4's folded heel sat 2.3 mm in the thigh at 10)
GUARD_M = 0.0008           # ... aimed 0.2 mm inside the contract's 1 mm (the solver stopped at 1.1 mm on E4's thigh)
OVERLAP_M = 0.001          # the contract: no new body-pair overlap deeper than 1 mm ...
OVERLAP_TOL_M = 0.0001     # ... beyond the solver's tolerance on the guard's -1 mm floor
OUT_ROOT = ids.REPO / "output/edge_synthesis/quasistatic"
LEGS = ("L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe")
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)   # mosh_replay's corners


# --------------------------------------------------------------------------- #
# Poses in PhysX's coordinates (root position, root rotation matrix, exp-map dof)
# --------------------------------------------------------------------------- #
@dataclass
class Pose:
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof: np.ndarray

    def copy(self) -> "Pose":
        return Pose(self.root_pos.copy(), self.root_rot.copy(), self.dof.copy())


def pose_from_bodies(sk, pos: np.ndarray, rot_xyzw: np.ndarray) -> Pose:
    R = Rotation.from_quat(np.asarray(rot_xyzw, float)).as_matrix()
    par = sk.parents
    dof = np.concatenate([Rotation.from_matrix(R[par[i]].T @ R[i]).as_rotvec() for i in range(1, len(sk.names))])
    return Pose(np.asarray(pos[0], float).copy(), R[0].copy(), dof)


def bodies(sk, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    from reference_curation import retarget as rt

    with torch.no_grad():
        p, r = rt.fk(sk, torch.as_tensor(pose.root_pos[None]), torch.as_tensor(pose.root_rot[None]),
                     torch.as_tensor(pose.dof[None]))
    return p[0].numpy(), r[0].numpy()


def hybrid(sk, base: Pose, other: Pose, from_other: tuple, root_from_other: bool) -> Pose:
    """``base`` with the local rotations of the bodies ``from_other`` (and optionally the root's) taken from
    ``other``."""
    out = base.copy()
    for b in from_other:
        i = sk.names.index(b) - 1
        out.dof[3 * i:3 * i + 3] = other.dof[3 * i:3 * i + 3]
    if root_from_other:
        out.root_rot = other.root_rot.copy()
    return out


def anchored(sk, pose: Pose, point: np.ndarray, names=("L_Hand", "R_Hand")) -> Pose:
    """``pose`` translated so the midpoint of ``names`` sits at ``point``."""
    p, _ = bodies(sk, pose)
    idx = [sk.names.index(n) for n in names]
    out = pose.copy()
    out.root_pos = pose.root_pos + (np.asarray(point) - p[idx].mean(0))
    return out


def endpoint_poses(sk, e: dict) -> tuple[Pose, Pose, dict]:
    """S and the re-anchored D (``sketch.hand_anchor_transform``) as Poses."""
    sp, sr, _ = SK.exemplar(e["source"])
    dp, dr, _ = SK.exemplar(e["destination"])
    bi = {n: i for i, n in enumerate(sk.names)}
    yaw, t = SK.hand_anchor_transform(sp, dp, bi)
    dp2, dr2 = SK.apply_planar(dp, dr, yaw, t)
    return pose_from_bodies(sk, sp, sr), pose_from_bodies(sk, dp2, dr2), {"yaw_deg": round(math.degrees(yaw), 3)}


# --------------------------------------------------------------------------- #
# Keyframe drafts (the model proposes, the instruments decide)
# --------------------------------------------------------------------------- #
def drafts(edge_id: str, sk, S: Pose, D: Pose, sched: SK.Schedule) -> list[dict]:
    """``[{name, t, pose, note}]`` intermediate keyframes of an edge, at times on the schedule's clock."""
    hands = bodies(sk, S)[0][[sk.names.index("L_Hand"), sk.names.index("R_Hand")]].mean(0)
    b = sched.bounds
    if edge_id == "E1":         # crow -> handstand: lift, tuck inverted, extend
        tuck = anchored(sk, hybrid(sk, D, S, LEGS, root_from_other=False), hands)
        return [{"name": "lift", "t": b[0] + 0.25 * (b[1] - b[0]), "pose": anchored(sk, S, hands),
                 "note": "the crow with its braces open: shins off the upper arms, the COM over the hands"},
                {"name": "tuck_inverted", "t": b[1], "pose": tuck,
                 "note": "the handstand's arms, shoulders and trunk with the crow's folded legs: hips over the "
                         "shoulders, knees to the chest"}]
    if edge_id == "E3":         # handstand -> crow: tuck inverted, lower, braces close
        tuck = anchored(sk, hybrid(sk, S, D, LEGS, root_from_other=False), hands)
        return [{"name": "tuck_inverted", "t": b[0] + 0.45 * (b[1] - b[0]), "pose": tuck,
                 "note": "the legs fold to the chest over a vertical trunk"},
                {"name": "pre_crow", "t": b[1], "pose": anchored(sk, D, hands),
                 "note": "the crow with the shins just off the upper arms (the braces close in the settle phase)"}]
    if edge_id == "E4":         # tripod -> crow: fold the legs onto the arms, shift forward, lift the head
        folded = anchored(sk, hybrid(sk, S, D, LEGS, root_from_other=False), hands)
        return [{"name": "tripod_tuck", "t": b[1], "pose": folded,
                 "note": "the tripod's head, arms and trunk with the crow's folded legs, knees on the upper arms"},
                {"name": "head_light", "t": b[2], "pose": anchored(sk, D, hands),
                 "note": "the crow with the head still touching (the head breaks in the settle phase)"}]
    if edge_id == "B2":         # firefly -> crow: bend the knees, lift the hips, shins onto the arms
        mid = anchored(sk, hybrid(sk, S, D, ("L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"),
                                  root_from_other=False), hands)
        return [{"name": "knees_bent", "t": b[1], "pose": mid,
                 "note": "the firefly with the knees bent to the crow's angle, thighs still on the arms"}]
    return []


# --------------------------------------------------------------------------- #
# The anchor trajectory
# --------------------------------------------------------------------------- #
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return np.clip(x * x * x * (10 - 15 * x + 6 * x * x), 0.0, 1.0)


def anchor_trajectory(sk, keys: list[tuple[float, Pose]], t0: float, t1: float, hand_point: np.ndarray,
                      fps: int = FPS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(times [T], root_pos [T,3], root_rot [T,3,3], dof [T,69])``: the keyframes joined by minimum-jerk slerp of
    the root and local rotations, the root placed so the hands' midpoint stays at ``hand_point``."""
    times = np.arange(int(round(t0 * fps)), int(round(t1 * fps)) + 1) / fps
    kt = np.array([k[0] for k in keys])
    T = len(times)
    root_rot = np.empty((T, 3, 3))
    dof = np.empty((T, 69))
    nb = len(sk.names) - 1
    for i, t in enumerate(times):
        if t <= kt[0]:
            k, s = 0, 0.0
        elif t >= kt[-1]:
            k, s = len(kt) - 2, 1.0
        else:
            k = int(np.searchsorted(kt, t, side="right") - 1)
            s = float(_smoothstep((t - kt[k]) / (kt[k + 1] - kt[k])))
        a, b = keys[k][1], keys[k + 1][1]
        rr = Slerp([0, 1], Rotation.from_matrix(np.stack([a.root_rot, b.root_rot])))([s])[0]
        root_rot[i] = rr.as_matrix()
        La = Rotation.from_rotvec(a.dof.reshape(nb, 3))
        Lb = Rotation.from_rotvec(b.dof.reshape(nb, 3))
        rel = (La.inv() * Lb).as_rotvec()
        dof[i] = (La * Rotation.from_rotvec(s * rel)).as_rotvec().reshape(-1)
    from reference_curation import retarget as rt

    with torch.no_grad():
        p, _ = rt.fk(sk, torch.zeros(T, 3, dtype=F64), torch.as_tensor(root_rot), torch.as_tensor(dof))
    idx = [sk.names.index("L_Hand"), sk.names.index("R_Hand")]
    root_pos = np.asarray(hand_point)[None] - p[:, idx].mean(1).numpy()
    return times, root_pos, root_rot, dof


# --------------------------------------------------------------------------- #
# The problem
# --------------------------------------------------------------------------- #
def _box_cands(sk) -> dict:
    """``{body: (first candidate index, geom rotation)}`` of every box body (8 corners in ``SIGNS`` order)."""
    out = {}
    for i, b in enumerate(sk.names):
        g = sk.geoms[b][0]
        if g["type"] == "box":
            k = np.nonzero(sk.cand_body == i)[0]
            out[i] = (int(k[0]), Rotation.from_quat(g["quat"]).as_matrix())
    return out


@dataclass
class Contacts:
    """The contact configuration of every frame: ``ground [T, Z]`` planted zones, ``braces [T]`` closed body-body
    pairs, ``known [Z]`` zones whose state is known (planted or known free) and ``balance [T]`` frames on which the
    COM must lie over the planted supports (quasi-static frames)."""
    ground: np.ndarray
    braces: list
    known: np.ndarray
    balance: np.ndarray
    off_exempt: np.ndarray | None = None      # [T, Z]: frames on which a known zone is not kept off (touchdowns)
    unload: np.ndarray | None = None          # [T, Z]: planted zones the balance hull leaves out (about to break)


def schedule_frames(sched: SK.Schedule, times: np.ndarray) -> Contacts:
    """The schedule's configuration on every frame; balance on its quasi-static phases (and the held ends)."""
    Z = len(ZONE_ORDER)
    ground = np.zeros((len(times), Z), bool)
    braces = []
    qs = np.zeros(len(times), bool)
    for i, t in enumerate(times):
        g, br, q = sched.config_at(float(t))
        ground[i] = [z in g for z in ZONE_ORDER]
        braces.append(set(br))
        qs[i] = q
    known = np.array([z in sched.known for z in ZONE_ORDER])
    # pre-load: a support that breaks must already be unloadable -- the COM over the remaining supports -- for
    # PRELOAD_S before its break (E4's head lifted with the COM still behind the hands: s* 1.003 at the event)
    unload = np.zeros_like(ground)
    for te, z, kind in sched.events:
        if kind == "break":
            unload[(times >= te - PRELOAD_S) & (times <= te), ZONE_ORDER.index(z)] = True
    return Contacts(ground, braces, known, qs, None, unload)


def build_problem(sk, stem: str, times, root_pos0, root_rot0, dof0, sched: SK.Schedule | Contacts,
                  weights: dict | None = None):
    """``retarget_v2.Problem`` of an anchor trajectory under a contact configuration (a ``Schedule``, read per
    frame, or explicit ``Contacts``)."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    _install_balance_mask()
    T = len(times)
    lower, upper = sk.lower + rv2.LIMIT_MARGIN_RAD, sk.upper - rv2.LIMIT_MARGIN_RAD
    r0 = torch.as_tensor(root_pos0, dtype=F64)
    R0 = torch.as_tensor(root_rot0, dtype=F64)
    dof_rep = rt.nearest_representative(torch.as_tensor(dof0, dtype=F64), sk.lower, sk.upper)
    with torch.no_grad():
        pos0, rot0 = rt.fk(sk, r0, R0, dof_rep)
        pos_i, rot_i = rt.fk(sk, r0, R0, torch.maximum(torch.minimum(dof_rep, upper), lower))
    con = sched if isinstance(sched, Contacts) else schedule_frames(sched, times)
    ground, braces, known = con.ground, con.braces, con.known
    K = len(sk.cand_body)
    pts = rt.candidate_points(sk, pos0, rot0).numpy()
    h = pts[..., 2] - sk.cand_radius
    boxes = _box_cands(sk)
    target = np.zeros((T, K), bool)
    off = np.zeros((T, K), bool)
    flat_f, flat_b, flat_a, flat_s = [], [], [], []
    rot_np = rot0.numpy()
    for f in range(T):
        for zi, z in enumerate(ZONE_ORDER):
            cands = sk.zone_cands[zi]
            if ground[f, zi]:
                low = h[f, cands].min()
                for b in sorted({int(sk.cand_body[k]) for k in cands}):
                    kb = cands[sk.cand_body[cands] == b]
                    if h[f, kb].min() > low + PLANTED_BAND_M:
                        continue
                    if b in boxes:
                        k0, rg = boxes[b]
                        nz = (rot_np[f, b] @ rg)[2]
                        faces = [(a, s) for a in range(3) for s in (-1, 1)]
                        a, s = min(faces, key=lambda fs: fs[1] * nz[fs[0]])
                        corners = [k0 + c for c in range(8) if SIGNS[c, a] == s]
                        target[f, corners] = True
                        flat_f.append(f); flat_b.append(b); flat_a.append(a); flat_s.append(float(s))
                    else:
                        target[f, kb[np.argmin(h[f, kb])]] = True
            elif known[zi] and (con.off_exempt is None or not con.off_exempt[f, zi]):
                off[f, cands] = True
    target, off = rv2.clean_runs(target), rv2.clean_runs(off)
    tw, ow = rv2.ramp_weights(target), rv2.ramp_weights(off)
    fl = {"frames": np.array(flat_f, int), "body": np.array(flat_b, int), "axis": np.array(flat_a, int),
          "sign": np.array(flat_s, float), "w": np.ones(len(flat_f))}
    plan = {"target": target, "off": off, "target_w": tw, "off_w": ow, "flat": fl, "heights": None,
            "head_inverted": None}
    # braces: closed ones are pair requests; opened ones keep OPEN_GAP_M through the guard's per-pair floor
    pairs = rt.plant_pairs()
    index = np.full((sk.num_bodies, sk.num_bodies), -1)
    index[pairs[:, 0], pairs[:, 1]] = index[pairs[:, 1], pairs[:, 0]] = np.arange(len(pairs))
    floor_tp = np.full((T, len(pairs)), -GUARD_M)
    all_braces = sorted(set().union(*braces))
    items, rows = {}, []
    for f in range(T):
        for br in braces[f]:
            items[(br, f)] = 1.0
    keys = sorted(items)
    for k_, (br, f) in enumerate(keys):
        for a, b in rt.zone_pair_bodies(*br.split("+")):
            rows.append((k_, f, a, b))
    close = {"item": np.array([r[0] for r in rows], int), "frames": np.array([r[1] for r in rows], int),
             "body_a": np.array([r[2] for r in rows], int), "body_b": np.array([r[3] for r in rows], int),
             "w": np.array([items[k] for k in keys]), "contact": [c for c, _ in keys]}
    for br in all_braces:
        closed = np.array([br in braces[f] for f in range(T)])
        if closed.all():
            continue
        # the open frames: a ramp from the guard's floor at a break/make to OPEN_GAP_M within OPEN_RAMP_S
        dist = np.full(T, np.inf)
        for f in np.nonzero(closed)[0]:
            dist = np.minimum(dist, np.abs(times - times[f]))
        ramp = np.clip(dist / OPEN_RAMP_S, 0, 1)
        flo = -GUARD_M + (OPEN_GAP_M + GUARD_M) * ramp
        for a, b in rt.zone_pair_bodies(*br.split("+")):
            j = index[a, b]
            if j >= 0:
                floor_tp[~closed, j] = flo[~closed]
    pen = {"pairs": pairs, "floor": OPEN_GAP_M, "floor_tp": floor_tp, "soft_tp": np.zeros_like(floor_tp, bool),
           "index": index}
    # no slide: every targeted point stays where it was on the previous frame
    w = np.minimum(tw[:-1], tw[1:])
    vf, vk = np.nonzero(w > 0)
    vel = {"frames": vf, "cand": vk, "w": w[vf, vk], "d0": torch.zeros(len(vf), 2, dtype=F64)}
    # balance on every quasi-static frame with >= 3 planted points
    pts_i = rt.candidate_points(sk, pos_i, rot_i).numpy()
    bfr, bw, bm = [], [], []
    keep = np.ones((T, K), bool)
    if con.unload is not None:
        for zi in range(len(ZONE_ORDER)):
            keep[np.ix_(con.unload[:, zi], sk.zone_cands[zi])] = False
    for f in range(T):
        sel = (tw[f] > 0.5) & keep[f]
        if sel.sum() >= 3 and con.balance[f]:
            bfr.append(f)
            bw.append(1.0)
            margin = max(rv2.BALANCE_MARGIN_M, rv2.MARGIN_SHARE * rv2.hull_inradius(pts_i[f, sel, :2]))
            if con.unload is not None and con.unload[f].any():
                margin = max(margin, PRELOAD_MARGIN_M)
            bm.append(margin)
    balance = {"frames": np.array(bfr, int), "w": np.array(bw), "margin": np.array(bm), "speed": None}
    plan["balance_keep"] = keep
    prob = rv2.Problem(stem, FPS, r0, R0, dof_rep, pos0, rot0, plan, close, pen, [], vel, balance, lower, upper)
    prob.weights["pen"] = PEN_WEIGHT
    if weights:
        prob.weights.update(weights)
    return prob


def _install_balance_mask():
    """``retarget_v2._balance`` builds each frame's support hull from every targeted point; a problem built here may
    carry ``plan["balance_keep"]`` (the pre-load: a support about to break leaves the hull). Wrap the function in this
    process so it reads the masked weights -- ``retarget_v2.py`` itself stays untouched (its source hash is part of
    the retarget record) and problems without the mask are unchanged."""
    from reference_curation import retarget_v2 as rv2

    if getattr(rv2._balance, "_edge_synthesis", False):
        return
    orig = rv2._balance

    def _balance(prob, st, jacobian):
        keep = prob.plan.get("balance_keep")
        if keep is None:
            return orig(prob, st, jacobian)
        tw = prob.plan["target_w"]
        prob.plan["target_w"] = np.where(keep, tw, 0.0)
        try:
            return orig(prob, st, jacobian)
        finally:
            prob.plan["target_w"] = tw

    _balance._edge_synthesis = True
    rv2._balance = _balance


# --------------------------------------------------------------------------- #
# Certification
# --------------------------------------------------------------------------- #
def certify(sk, prob, x: torch.Tensor, sched: SK.Schedule | Contacts, times: np.ndarray, S: Pose, D: Pose,
            every: int = STATICS_EVERY, dst_braces=None) -> dict:
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2
    from reference_curation import statics

    root_pos, root_rot, dof = rt.unpack(prob, x)
    with torch.no_grad():
        pos, rot = rt.fk(sk, root_pos, root_rot, dof)
    pos_np, rot_np = pos.numpy(), rot.numpy()
    quat = Rotation.from_matrix(rot_np.reshape(-1, 3, 3)).as_quat().reshape(rot_np.shape[:2] + (4,))
    con = sched if isinstance(sched, Contacts) else schedule_frames(sched, times)
    ground, braces = con.ground, con.braces
    dst_braces = sorted(sched.dst_braces) if dst_braces is None and not isinstance(sched, Contacts) else \
        sorted(dst_braces or [])
    rows = []
    for f in range(0, len(times), every):
        if not con.balance[f]:          # statics certifies quasi-static frames only (a flight has no static answer)
            continue
        g = [z for zi, z in enumerate(ZONE_ORDER) if ground[f, zi]]
        res = statics.analyse(pos_np[f], quat[f], g, sorted(braces[f]), with_necessity=False)
        top = (res.get("top_joints") or [[None, None]])[0]
        rows.append({"frame": f, "t": round(float(times[f]), 3), "status": res["status"], "s_star": res["s_star"],
                     "binding": top[0], "unrealised": res["unrealised"], "open": res["open"]})
    s = [r["s_star"] for r in rows if r["s_star"] is not None]
    # the contract
    exc = np.degrees(np.maximum(sk.lower.numpy() - dof.numpy(), 0) + np.maximum(dof.numpy() - sk.upper.numpy(), 0))
    pts = rt.candidate_points(sk, pos, rot)
    hgt = rt.candidate_heights(sk, pts).numpy()
    # new overlaps deeper than 1 mm (0.1 mm solver tolerance: the guard's floor is exactly -1 mm), against the
    # endpoints' own overlaps; a brace the schedule closes may touch
    nf, na, nb, gap = rt.near_body_pairs(sk, pos, rot, -OVERLAP_M - OVERLAP_TOL_M)
    end_pairs = set()
    for P in (S, D):
        pe, re_ = (torch.as_tensor(v)[None] for v in bodies(sk, P))
        _, ea, eb, _ = rt.near_body_pairs(sk, pe, re_, -OVERLAP_M - OVERLAP_TOL_M)
        end_pairs |= {(int(a), int(b)) for a, b in zip(ea, eb)}
    zone_of = {i: z for z, bs in ZONES.items() for i in [sk.names.index(b) for b in bs]}

    def brace(f, a, b):
        return f"{zone_of[a]}+{zone_of[b]}" in braces[f] or f"{zone_of[b]}+{zone_of[a]}" in braces[f]
    new = [(int(f), sk.names[a], sk.names[b], round(100 * float(g_), 2)) for f, a, b, g_ in zip(nf, na, nb, gap)
           if (int(a), int(b)) not in end_pairs and not brace(int(f), int(a), int(b))]
    acc = rv2.body_acc(pos_np, FPS)
    mass, centre = rt.mass_model()
    com = ((mass[None, :, None] * (pos_np + np.einsum("tbij,bj->tbi", rot_np, centre))).sum(1) / mass.sum())
    margins = []
    keep = prob.plan.get("balance_keep")
    for f in range(len(times)):
        sel = prob.plan["target_w"][f] > 0.5
        if keep is not None:
            sel &= keep[f]
        if sel.sum() >= 3 and con.balance[f]:
            rows_ = rv2.hull_rows(pts[f, sel, :2].numpy(), com[f, :2], 1e9)
            if rows_:
                margins.append(min(r[2] for r in rows_))
    # braces at a crow end: closed within 1 cm on the last frames
    brace_gap = {}
    for br in dst_braces:
        g_, _, _ = rt.zone_pair_gaps(sk, pos, rot, *br.split("+"), np.arange(len(times) - 10, len(times)))
        brace_gap[br] = round(100 * float(g_.max()), 2)
    hands = [sk.names.index(n) for n in ("L_Hand", "R_Hand", "L_Wrist", "R_Wrist")]
    drift = np.linalg.norm(pos_np[:, hands, :2] - pos_np[:1, hands, :2], axis=-1).max()
    out = {
        "statics": {"checked": len(rows), "s_star_max": max(s) if s else None,
                    "beyond": [r for r in rows if r["s_star"] is not None and r["s_star"] > 1.0],
                    "not_optimal": [r for r in rows if r["status"] != "optimal"],
                    "rows": rows},
        "box_frames_past_1deg": int((exc > 1.0).any(1).sum()), "box_max_deg": round(float(exc.max()), 2),
        "floor_min_cm": round(100 * float(hgt.min()), 2),
        "new_overlaps_1mm": len(new), "new_overlap_examples": new[:8],
        "com_margin_min_cm": round(100 * float(min(margins)), 2) if margins else None,
        "acc_over_100_frames": int((acc > rv2.SPIKE_ACC).sum()),
        "brace_gap_end_cm": brace_gap, "hand_drift_cm": round(100 * float(drift), 2),
    }
    out["pass"] = bool(out["box_frames_past_1deg"] == 0 and out["floor_min_cm"] >= -0.5 and out["new_overlaps_1mm"] == 0
                       and (out["com_margin_min_cm"] or 0) > 0 and not out["statics"]["beyond"]
                       and not out["statics"]["not_optimal"] and all(v <= 1.0 for v in brace_gap.values()))
    return out, pos_np, quat


# --------------------------------------------------------------------------- #
# One edge
# --------------------------------------------------------------------------- #
def generate(edge_id: str, timing="mid", iters: int = 60, out_root: Path = OUT_ROOT, pre: float = 0.3,
             post: float = 0.5, weights: dict | None = None, log=print) -> dict:
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    spec = SK.load_edges()
    e = SK.edge(spec, edge_id)
    durs = SK.timing(e, timing)
    sched = SK.schedule(e, durs)
    with rv2.on_plant("v2") as sk:
        S, D, info = endpoint_poses(sk, e)
        hands = bodies(sk, S)[0][[sk.names.index("L_Hand"), sk.names.index("R_Hand")]].mean(0)
        kf = drafts(edge_id, sk, S, D, sched)
        keys = [(0.0, S)] + [(k["t"], k["pose"]) for k in kf] + [(sched.T, anchored(sk, D, hands))]
        times, rp, rr, dof = anchor_trajectory(sk, keys, -pre, sched.T + post, hands)
        prob = build_problem(sk, f"{edge_id}_{timing}", times, rp, rr, dof, sched, weights)
        t0 = time.time()
        x, rep = rv2.solve(prob, iters=iters)
        spikes = rv2.spike_frames(prob, x)
        if spikes:
            log(f"{edge_id}: {spikes} jerk frames after the direct solve; solving gently")
            x2, rep2 = rv2.solve_gently(prob, iters=iters, x0=x)
            if rv2.spike_frames(prob, x2) <= spikes:
                x, rep = x2, rep2
        solve_s = time.time() - t0
        spikes_after = rv2.spike_frames(prob, x)
        cert, pos, quat = certify(sk, prob, x, sched, times, S, D)
        root_pos, root_rot, dofx = (v.numpy() for v in rt.unpack(prob, x))
    import edge_synthesis
    rec = {"provenance": edge_synthesis.provenance(),
           "config": {"open_gap_m": OPEN_GAP_M, "open_ramp_s": OPEN_RAMP_S, "planted_band_m": PLANTED_BAND_M,
                      "preload_s": PRELOAD_S, "preload_margin_m": PRELOAD_MARGIN_M, "pen_weight": PEN_WEIGHT,
                      "guard_m": GUARD_M, "statics_every": STATICS_EVERY, "iters": iters, "pre_s": pre, "post_s": post},
           "edge": edge_id, "label": e["label"], "timing": timing, "durations_s": [round(d, 3) for d in durs],
           "T": round(sched.T, 3), "frames": len(times), "t0": float(times[0]), "fps": FPS, "anchor": info,
           "keyframes": [{"name": k["name"], "t": round(float(k["t"]), 3), "note": k["note"]} for k in kf],
           "solver": {k: rep[k] for k in ("iterations", "seconds", "energy_start", "energy", "terms")},
           "solve_s": round(solve_s, 1), "spike_frames": spikes_after, "certification": cert}
    out = Path(out_root) / f"{edge_id}_{timing if isinstance(timing, str) else 'custom'}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reference.npz", times=times, root_pos=root_pos, root_rot=root_rot, dof=dofx,
                        body_pos=pos, body_quat=quat, anchor_root_pos=rp, anchor_root_rot=rr, anchor_dof=dof)
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edge", required=True)
    ap.add_argument("--timing", default="mid")
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args(argv)
    print("cpu policy", gpu_guard.be_polite(), flush=True)
    torch.set_num_threads(1)
    rec = generate(args.edge, args.timing, args.iters)
    c = rec["certification"]
    print(json.dumps({k: v for k, v in rec.items() if k != "certification"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in c.items() if k != "statics"}, indent=1, default=float))
    st = c["statics"]
    print("statics: checked", st["checked"], "s* max", st["s_star_max"], "beyond", len(st["beyond"]),
          "not optimal", len(st["not_optimal"]))
    for r in st["rows"][:: max(1, len(st["rows"]) // 20)]:
        print("  ", r["t"], r["status"], r["s_star"], r["binding"], r["unrealised"], r["open"])
    print("T2 certification", "PASSED" if c["pass"] else "FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
