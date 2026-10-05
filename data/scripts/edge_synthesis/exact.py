"""Card T6 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: endpoint-exact references for the D1 edges.

Why (PLAN.MD §1.4, "What the Step 3 build must handle"; measured on the D1 selection of PhysX variants):

* **E1, E3** started from their T2 reference's first frame, whose solve moved the arms: the hands sat 4-6 cm off
  S's exemplar at both ends (``flat``/``support`` re-solved palms that were already where the human put them, with
  the crow's wrist at its box limit, so the arm moved instead of the wrist).
* **B1, E2, E5** started at S's exemplar but put the feet 5-14 cm from D's. Measured here (2026-10-04): the landing
  sketch (``edge_mppi --keyframes land``, a minimum-jerk slerp from the crow's tuck to D) drives the feet 4-7 cm
  *below the floor* while they are still 41-47 cm short of D's feet. Every execution touched down short (20-68 cm
  along the hands-to-feet axis on B1, 2-11 cm on E2, 13-24 cm on E5) and then dragged the loaded feet back for about
  a second. E5's sketch also lifts the feet 5 cm off the plank (its ``--via``) on the way into chaturanga.

The references are built with T2's solver (``quasistatic``: ``retarget_v2``'s LM, the joint box a hard bound, the
253-pair guard, the schedule's support/off/no-slide/brace/balance rows) plus these rules:

* **Frozen ends.** Every frame up to the departure (the settle) *is* S's exemplar, and every frame from the arrival
  on *is* D' (below): those frames' variables are bounds-pinned (lo = hi).
* **Pinned supports.** The hands keep S's exemplar pose -- the position and orientation of ``L_Wrist``, ``L_Hand``,
  ``R_Wrist``, ``R_Hand`` -- on every frame. A zone pinned so drops out of the ``support`` and ``flat`` rows (it is
  already where the human put it) and stays in the balance hull. A landing's feet are pinned from their touchdown on
  to where D' has them (ramped in over ``FOOT_PIN_RAMP_S``).
* **D', the edited destination.** D's exemplar re-anchored by ``sketch.hand_anchor_transform`` as before, then
  IK-edited by a static solve of the same objective so that its hands are S's (and, on a landing edge, its feet D's
  own). Statics runs on D'. E5's touchdown keyframe (the plank, ``--via``) is edited the same way onto S's hands and
  D's toes, so it touches down where chaturanga keeps its feet.
* **The landing cone** (landing edges). Before the touchdown, every collider point that touches the floor at the
  touchdown keyframe stays at least ``CONE_SLOPE`` x its horizontal distance from its own spot above that spot's
  height: a foot can come down only where it lands.

If the joint box makes a pin infeasible on some frame, the frame and the binding joints are reported (as T2 reports
its statics negatives): that is a finding, not a knob.

The output has ``quasistatic``'s layout (``reference.npz``, ``record.json``) in ``output/edge_synthesis/exact/
<edge>_<timing>/``, so ``edge_mppi_physx --reference <dir> --start exemplar`` tracks it unchanged.

CLI::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts python -m edge_synthesis.exact --edge E1 --timing high
    ... --edge E5 --timing mid --via 220923_Plank_Pose_or_Kumbhakasana_-a@541
    ... --d1          # every (edge, timing) of the D1 selection
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from extract_contact_configs import ZONE_ORDER, ZONES
from edge_synthesis import gpu_guard
from edge_synthesis import quasistatic as Q
from edge_synthesis import sketch as SK
from reference_curation import ids

F64 = torch.float64
FPS = Q.FPS
OUT_ROOT = ids.REPO / "output/edge_synthesis/exact"
HAND_ZONES = ("L_HAND", "R_HAND")
FOOT_ZONES = ("L_FOOT", "R_FOOT")
HAND_BODIES = tuple(b for z in HAND_ZONES for b in ZONES[z])          # L_Wrist L_Hand R_Wrist R_Hand
FOOT_BODIES = tuple(b for z in FOOT_ZONES for b in ZONES[z])          # L_Ankle L_Toe R_Ankle R_Toe
TOE_BODIES = ("L_Toe", "R_Toe")
PIN_POS_M = 0.0005           # the pin rows' scales: 0.5 mm and 0.5 deg cost what 2.5 mm costs a support row ...
PIN_ROT_RAD = math.radians(0.5)
PIN_WEIGHT = 10.0            # ... at retarget_v2's contact weight
PIN_TOL_M = 0.001            # a pinned body is "on its pin" within 1 mm ...
PIN_TOL_DEG = 1.0            # ... and 1 deg; beyond, the frame is reported with its binding joints
FOOT_PIN_RAMP_S = 0.1        # a landing foot's pin ramps in over this long from its touchdown
CONE_SLOPE = 0.25            # a landing point at height h above its spot lies within h / CONE_SLOPE of it
CONE_SCALE = 0.005
CONE_WEIGHT = 10.0
CONE_TOUCH_M = 0.015         # the touchdown keyframe's collider points this close to the floor are landing points
EDIT_FRAMES = 8              # the static solve that makes D' (identical frames)
# The D1 selection's (edge, timing) recipes (selected_physx.json), with E5's touchdown hold.
D1_RECIPES = (("E1", "high", None), ("E1", "mid", None), ("E3", "high", None), ("B1", "high", None),
              ("B1", "mid", None), ("E2", "mid", None), ("E5", "mid", "220923_Plank_Pose_or_Kumbhakasana_-a@541"))
QUASI_STATIC = ("E1", "E3", "E4", "B2")


# --------------------------------------------------------------------------- #
# The added rows: pins, the landing cone; frozen frames through the bounds
# --------------------------------------------------------------------------- #
def _install():
    """Wrap ``retarget_v2.residuals`` and ``retarget_v2.bounds`` in this process (``retarget_v2.py`` stays untouched:
    its source hash is part of the retarget record; ``quasistatic._install_balance_mask`` does the same). A problem
    carrying ``plan["t6"]`` gets the pin and cone rows and its frozen frames; every other problem is unchanged."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    Q._install_balance_mask()
    if getattr(rv2.residuals, "_edge_synthesis_t6", False):
        return
    orig_res, orig_bounds = rv2.residuals, rv2.bounds

    def residuals(prob, x, jacobian=False):
        out = orig_res(prob, x, jacobian)
        ex = prob.plan.get("t6")
        if ex is None:
            return out
        st = rt.kinematics(prob, x)
        out.update(pin_rows(ex.get("pins"), st, jacobian))
        out["cone"] = cone_rows(ex.get("cone"), st, jacobian)
        return out

    def bounds(prob):
        lo, hi = orig_bounds(prob)
        ex = prob.plan.get("t6")
        fz = None if ex is None else ex.get("freeze")
        if fz is not None:
            m = torch.as_tensor(fz["mask"])
            lo, hi = lo.clone(), hi.clone()
            lo[m] = fz["x"][m]
            hi[m] = fz["x"][m]
        return lo, hi

    residuals._edge_synthesis_t6 = True
    rv2.residuals, rv2.bounds = residuals, bounds


def pin_rows(pins: dict | None, st, jacobian: bool) -> dict:
    """``pin_pos``: ``sqrt(w) (p - p*) / PIN_POS_M``; ``pin_rot``: ``sqrt(w) log(R*^T R) / PIN_ROT_RAD`` (the
    ``ends`` rows' form), for every ``(frame, body)`` row of ``pins`` with a nonzero weight."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    out = {"pin_pos": rv2._empty(), "pin_rot": rv2._empty()}
    if pins is None or not len(pins["frames"]):
        return out
    f, b = pins["frames"], pins["body"]
    sel = np.nonzero(pins["w_pos"] > 0)[0]
    if len(sel):
        c = math.sqrt(PIN_WEIGHT) / PIN_POS_M * torch.sqrt(torch.as_tensor(pins["w_pos"][sel], dtype=F64))
        p = st.pos[f[sel], b[sel]]
        blk = {"frames": np.repeat(f[sel], 3),
               "r": (c[:, None] * (p - torch.as_tensor(pins["pos"][sel], dtype=F64))).reshape(-1)}
        if jacobian:
            blk["J"] = (c[:, None, None] * rv2._point_jacobian(st, f[sel], b[sel], p)).reshape(-1, rt.NV)
        out["pin_pos"] = blk
    sel = np.nonzero(pins["w_rot"] > 0)[0]
    if len(sel):
        fr, br = f[sel], b[sel]
        c = math.sqrt(PIN_WEIGHT) / PIN_ROT_RAD * torch.sqrt(torch.as_tensor(pins["w_rot"][sel], dtype=F64))
        R = st.rot[fr, br]
        lg = rt.so3_log(torch.as_tensor(pins["rot"][sel], dtype=F64).transpose(-1, -2) @ R)
        blk = {"frames": np.repeat(fr, 3), "r": (c[:, None] * lg).reshape(-1)}
        if jacobian:
            blk["J"] = (c[:, None, None] * (rt.right_jacobian_inv(lg) @ R.transpose(-1, -2)
                                            @ rt.rotation_jacobian(st, fr, br))).reshape(-1, rt.NV)
        out["pin_rot"] = blk
    return out


def cone_rows(cone: dict | None, st, jacobian: bool) -> dict:
    """The landing cone: ``g = h - h0 - CONE_SLOPE |xy - xy*|`` for every ``(frame, candidate)`` row, a residual
    ``sqrt(w) g / CONE_SCALE`` where ``g < 0``."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    if cone is None or not len(cone["frames"]):
        return rv2._empty()
    sk = rt.skeleton()
    f, k = cone["frames"], cone["cand"]
    pts = st.pts[f, k]
    dxy = pts[:, :2] - torch.as_tensor(cone["xy"], dtype=F64)
    d = dxy.norm(dim=-1)
    g = st.h[f, k] - torch.as_tensor(cone["h0"], dtype=F64) - CONE_SLOPE * d
    idx = torch.nonzero(g < 0, as_tuple=True)[0]
    if not len(idx):
        return rv2._empty()
    ii = idx.numpy()
    c = math.sqrt(CONE_WEIGHT) / CONE_SCALE * torch.sqrt(torch.as_tensor(cone["w"][ii], dtype=F64))
    blk = {"frames": f[ii], "r": c * g[idx]}
    if jacobian:
        Jp = rv2._point_jacobian(st, f[ii], sk.cand_body[k[ii]], pts[idx])          # [n, 3, NV]
        u = dxy[idx] / d[idx].clamp(min=1e-9)[:, None]
        blk["J"] = c[:, None] * (Jp[:, 2] - CONE_SLOPE * (u[:, :, None] * Jp[:, :2]).sum(1))
    return blk


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def pins(sk, frames: np.ndarray, poses: dict, w_pos=1.0, w_rot=1.0) -> dict:
    """Pin rows: every ``(frame, body)`` of ``frames x poses`` (``{body: (pos [3], rot [3,3])}``), with per-frame
    weights (scalars or ``[len(frames)]``)."""
    frames = np.asarray(frames, int)
    names = sorted(poses)
    nb = len(names)
    wp = np.broadcast_to(np.asarray(w_pos, float), frames.shape)
    wr = np.broadcast_to(np.asarray(w_rot, float), frames.shape)
    return {"frames": np.repeat(frames, nb), "body": np.tile([sk.names.index(n) for n in names], len(frames)),
            "pos": np.tile(np.stack([poses[n][0] for n in names]), (len(frames), 1)),
            "rot": np.tile(np.stack([poses[n][1] for n in names]), (len(frames), 1, 1)),
            "w_pos": np.repeat(wp, nb), "w_rot": np.repeat(wr, nb)}


def cat_pins(*ps) -> dict:
    ps = [p for p in ps if p is not None and len(p["frames"])]
    return {k: np.concatenate([p[k] for p in ps]) for k in ps[0]} if ps else None


def body_poses(sk, pose: Q.Pose, names) -> dict:
    p, r = Q.bodies(sk, pose)
    return {n: (p[sk.names.index(n)].copy(), r[sk.names.index(n)].copy()) for n in names}


def drop_pinned(sk, prob, zone_frames: dict) -> None:
    """Zones pinned on some frames (``{zone: [T] bool}``) leave the ``support`` and ``flat`` rows on those frames;
    the balance hull keeps them (``plan["balance_w"]``)."""
    plan = prob.plan
    tw = plan["target_w"]
    plan["balance_w"] = tw.copy()
    tw = tw.copy()
    fl = plan["flat"]
    keep = np.ones(len(fl["frames"]), bool)
    for z, m in zone_frames.items():
        cands = sk.zone_cands[ZONE_ORDER.index(z)]
        tw[np.ix_(np.asarray(m, bool), cands)] = 0.0
        bodies = [sk.names.index(b) for b in ZONES[z]]
        keep &= ~(np.asarray(m, bool)[fl["frames"]] & np.isin(fl["body"], bodies))
    plan["target_w"] = tw
    plan["flat"] = {k: v[keep] for k, v in fl.items()}


def freeze_x(prob, mask: np.ndarray, pose_of) -> dict:
    """The bounds pin of the frames in ``mask``: ``x`` such that ``retarget.unpack`` gives ``pose_of(frame)``."""
    from reference_curation import retarget as rt

    x = torch.zeros(prob.T, rt.NV, dtype=F64)
    for f in np.nonzero(mask)[0]:
        P = pose_of(int(f))
        x[f, :3] = torch.as_tensor(P.root_pos, dtype=F64) - prob.root_pos0[f]
        x[f, 3:6] = rt.so3_log(prob.root_rot0[f].T @ torch.as_tensor(P.root_rot, dtype=F64))
        x[f, 6:] = rt.nearest_representative(torch.as_tensor(P.dof, dtype=F64)[None], rt.skeleton().lower,
                                             rt.skeleton().upper)[0]
    return {"mask": np.asarray(mask, bool), "x": x}


def pin_report(sk, prob, x, pins_: dict | None, times: np.ndarray) -> dict:
    """Per pinned body and frame, the distance to its pin; frames beyond ``PIN_TOL_*`` with the joints of the pinned
    bodies' chains that sit on a bound of the box (the binding joints)."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    if pins_ is None:
        return {"rows": 0}
    st = rt.kinematics(prob, x)
    f, b = pins_["frames"], pins_["body"]
    on = (pins_["w_pos"] >= 0.999) | (pins_["w_rot"] >= 0.999)
    dp = (st.pos[f, b] - torch.as_tensor(pins_["pos"], dtype=F64)).norm(dim=-1).numpy()
    dr = np.degrees(rt.so3_log(torch.as_tensor(pins_["rot"], dtype=F64).transpose(-1, -2) @ st.rot[f, b])
                    .norm(dim=-1).numpy())
    dp = np.where((pins_["w_pos"] >= 0.999) & on, dp, 0.0)
    dr = np.where((pins_["w_rot"] >= 0.999) & on, dr, 0.0)
    bad = (dp > PIN_TOL_M) | (dr > PIN_TOL_DEG)
    lo, hi = rv2.bounds(prob)
    at_bound = ((x <= lo + 1e-9) | (x >= hi - 1e-9)).numpy()
    anc = rt.ancestors()
    dof_names = [f"{n}_{a}" for n in sk.names[1:] for a in "xyz"]
    frames_bad = []
    for fr in sorted(set(f[bad].tolist())):
        rows = (f == fr) & bad
        chain = sorted({int(i) for bb in b[rows] for i in np.nonzero(anc[bb])[0] if i > 0})
        cols = [6 + 3 * (i - 1) + a for i in chain for a in range(3)]
        frames_bad.append({"frame": int(fr), "t": round(float(times[fr]), 3),
                           "bodies": sorted({sk.names[int(bb)] for bb in b[rows]}),
                           "pos_mm": round(1e3 * float(dp[rows].max()), 2), "rot_deg": round(float(dr[rows].max()), 2),
                           "binding": [dof_names[c - 6] for c in cols if at_bound[fr, c]]})
    return {"rows": int(on.sum()), "pos_max_mm": round(1e3 * float(dp.max()), 3), "rot_max_deg": round(float(dr.max()), 3),
            "frames_beyond_tol": len(frames_bad), "beyond": frames_bad[:40]}


def statics_of(sk, pose: Q.Pose, ground: list, braces: list) -> dict:
    from reference_curation import statics

    p, r = Q.bodies(sk, pose)
    quat = Rotation.from_matrix(r).as_quat()
    res = statics.analyse(p, quat, sorted(ground), sorted(braces), with_necessity=False)
    top = (res.get("top_joints") or [[None, None]])[0]
    return {"status": res["status"], "s_star": res["s_star"], "binding": top[0], "unrealised": res["unrealised"],
            "open": res["open"]}


def edit_pose(sk, pose: Q.Pose, ground: list, braces: list, known: np.ndarray, pins_poses: dict, stem: str,
              pin_rot: bool | dict = True, iters: int = 60) -> tuple[Q.Pose, dict]:
    """The static IK edit: ``pose`` held over ``EDIT_FRAMES`` identical frames under the configuration
    ``(ground, braces)`` (support, flat, off, brace, balance, floor, pair-guard rows) with the bodies of
    ``pins_poses`` (``{body: (pos, rot)}``) pinned -- orientation too where ``pin_rot`` (or ``pin_rot[body]``) is
    true. Zones all of whose bodies are pinned with orientation leave the support and flat rows."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    _install()
    T = EDIT_FRAMES
    times = np.zeros(T)
    con = Q.Contacts(np.tile([z in ground for z in ZONE_ORDER], (T, 1)), [set(braces)] * T, known, np.ones(T, bool))
    prob = Q.build_problem(sk, stem, times, np.repeat(pose.root_pos[None], T, 0), np.repeat(pose.root_rot[None], T, 0),
                           np.repeat(pose.dof[None], T, 0), con)
    rot_of = (lambda n: bool(pin_rot.get(n, False))) if isinstance(pin_rot, dict) else (lambda n: bool(pin_rot))
    frames = np.arange(T)
    p_rot = {n: v for n, v in pins_poses.items() if rot_of(n)}
    p_pos = {n: v for n, v in pins_poses.items() if not rot_of(n)}
    pn = cat_pins(pins(sk, frames, p_rot) if p_rot else None, pins(sk, frames, p_pos, w_rot=0.0) if p_pos else None)
    full = [z for z in ground if all(b in p_rot for b in ZONES[z])]
    drop_pinned(sk, prob, {z: np.ones(T, bool) for z in full})
    prob.plan["t6"] = {"pins": pn, "cone": None, "freeze": None}
    x, rep = rv2.solve(prob, iters=iters)
    rp, rr, dd = (v.numpy() for v in rt.unpack(prob, x))
    out = Q.Pose(rp[0].copy(), rr[0].copy(), dd[0].copy())
    p0, _ = Q.bodies(sk, pose)
    p1, _ = Q.bodies(sk, out)
    edit = np.linalg.norm(p1 - p0, axis=-1)
    report = {"solver": {k: rep[k] for k in ("iterations", "seconds", "energy_start", "energy", "terms")},
              "edit_body_max_cm": round(100 * float(edit.max()), 2),
              "edit_by_body_cm": {sk.names[i]: round(100 * float(edit[i]), 2) for i in np.argsort(-edit)[:8]},
              "pins": pin_report(sk, prob, x, pn, times),
              "frames_identical_m": float(np.abs(rp - rp[:1]).max()),
              "statics": statics_of(sk, out, ground, braces)}
    return out, report


def first_make(sched: SK.Schedule) -> float | None:
    makes = sorted({te for te, _, kind in sched.events if kind == "make" and te > 0})
    return makes[0] if makes and makes[0] < sched.T else None


def via_pose(sk, S: Q.Pose, hold_id: str) -> Q.Pose:
    """``edge_mppi.via_qpos`` in PhysX coordinates: the exemplar of ``hold_id`` (an endpoint of an edge in
    edges.json) re-anchored on S's hands."""
    sp, _ = Q.bodies(sk, S)
    bi = {n: i for i, n in enumerate(sk.names)}
    for e in SK.load_edges()["edges"]:
        for k in ("source", "destination"):
            if e[k]["hold_id"] == hold_id:
                vp, vr, _ = SK.exemplar(e[k])
                yaw, t = SK.hand_anchor_transform(sp, vp, bi)
                vp2, vr2 = SK.apply_planar(vp, vr, yaw, t)
                return Q.pose_from_bodies(sk, vp2, vr2)
    raise KeyError(f"{hold_id} is not an endpoint in edges.json")


def known_zones(sched: SK.Schedule) -> np.ndarray:
    return np.array([z in sched.known for z in ZONE_ORDER])


def release_bodies(ep: dict) -> np.ndarray:
    return SK.exemplar(ep)[0]


# --------------------------------------------------------------------------- #
# One edge
# --------------------------------------------------------------------------- #
def generate(edge_id: str, timing: str = "mid", via: str | None = None, iters: int = 60, pre: float = 0.3,
             post: float = 0.5, out_root: Path = OUT_ROOT, log=print) -> dict:
    """The endpoint-exact reference of one (edge, timing): quasi-static edges keep T2's keyframe drafts, landing
    edges the landing sketch (S, the touchdown keyframe at the first make, D'), both under the rules above."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    _install()
    e = SK.edge(SK.load_edges(), edge_id)
    durs = SK.timing(e, timing)
    sched = SK.schedule(e, durs)
    T_e = sched.T
    te = first_make(sched)
    landing = te is not None and any(z in FOOT_ZONES for _, z, kind in sched.events if kind == "make")
    if edge_id in QUASI_STATIC and landing:
        raise ValueError(f"{edge_id}: a quasi-static edge with a foot touchdown is not a case this module covers")
    known = known_zones(sched)
    t0 = time.time()
    with rv2.on_plant("v2") as sk:
        S, D, info = Q.endpoint_poses(sk, e)
        hands = Q.bodies(sk, S)[0][[sk.names.index("L_Hand"), sk.names.index("R_Hand")]].mean(0)
        S_hands = body_poses(sk, S, HAND_BODIES)
        # D': D's exemplar (re-anchored on S's hands) edited onto S's hands, and on a landing onto D's own feet
        D_pins = dict(S_hands)
        if landing:
            D_pins.update(body_poses(sk, D, FOOT_BODIES))
        Dp, d_edit = edit_pose(sk, D, sched.dst_ground, sched.dst_braces, known, D_pins, f"{edge_id}_Dprime")
        log(f"{edge_id}: D' edit {d_edit['edit_body_max_cm']} cm, pins {d_edit['pins'].get('pos_max_mm')} mm / "
            f"{d_edit['pins'].get('rot_max_deg')} deg, statics {d_edit['statics']['status']} s* {d_edit['statics']['s_star']}")
        keyframes, land, land_edit = [], None, None
        if landing:
            if via is not None:
                V = via_pose(sk, S, via)
                g_land, br_land, _ = sched.config_at(te + 1e-6)
                Dp_toes = body_poses(sk, Dp, TOE_BODIES)
                land, land_edit = edit_pose(sk, V, sorted(g_land), sorted(br_land), known, {**S_hands, **Dp_toes},
                                            f"{edge_id}_via", pin_rot={**{n: True for n in HAND_BODIES},
                                                                       **{n: False for n in TOE_BODIES}})
                log(f"{edge_id}: touchdown keyframe ({via}) edit {land_edit['edit_body_max_cm']} cm, statics "
                    f"{land_edit['statics']['status']} s* {land_edit['statics']['s_star']}")
            else:
                land = Dp
            keys = [(0.0, S), (te, land), (T_e, Dp)]
        else:
            kf = Q.drafts(edge_id, sk, S, Dp, sched)
            keyframes = [{"name": k["name"], "t": round(float(k["t"]), 3), "note": k["note"]} for k in kf]
            keys = [(0.0, S)] + [(k["t"], k["pose"]) for k in kf] + [(T_e, Dp)]
        times, rp, rr, dof = Q.anchor_trajectory(sk, keys, -pre, T_e + post, hands)
        prob = Q.build_problem(sk, f"{edge_id}_{timing}_exact", times, rp, rr, dof, sched)
        T = len(times)
        start = times <= 1e-9
        end = times >= T_e - 1e-9
        all_f = np.ones(T, bool)
        zone_pins = {z: all_f for z in HAND_ZONES}
        pn = pins(sk, np.arange(T), S_hands)
        cone = None
        if landing:
            after = times >= te - 1e-9
            ramp = np.clip((times - te) / FOOT_PIN_RAMP_S, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * ramp)
            fr = np.nonzero(after)[0]
            if via is None:      # B1, E2: the feet land in D' and stay (position and orientation)
                pn = cat_pins(pn, pins(sk, fr, body_poses(sk, Dp, FOOT_BODIES), w_pos=w[fr], w_rot=w[fr]))
                for z in FOOT_ZONES:
                    zone_pins[z] = after & (w >= 0.999)
            else:                # E5: the toes land where D' has them; the feet roll on them into chaturanga
                pn = cat_pins(pn, pins(sk, fr, body_poses(sk, Dp, TOE_BODIES), w_pos=w[fr], w_rot=0.0))
            # the cone: every collider point of the feet touching at the touchdown keyframe
            lp, lr = Q.bodies(sk, land)
            pts = rt.candidate_points(sk, torch.as_tensor(lp)[None], torch.as_tensor(lr)[None])[0]
            hgt = rt.candidate_heights(sk, pts[None])[0].numpy()
            cands = np.concatenate([sk.zone_cands[ZONE_ORDER.index(z)] for z in FOOT_ZONES])
            cands = cands[hgt[cands] < CONE_TOUCH_M]
            cf = np.nonzero(~after & ~start)[0]
            cone = {"frames": np.repeat(cf, len(cands)), "cand": np.tile(cands, len(cf)),
                    "xy": np.tile(pts[cands, :2].numpy(), (len(cf), 1)), "h0": np.tile(hgt[cands], len(cf)),
                    "w": np.ones(len(cf) * len(cands))}
        drop_pinned(sk, prob, zone_pins)

        def pose_at(f):
            return S if times[f] <= 1e-9 else Dp
        prob.plan["t6"] = {"pins": pn, "cone": cone, "freeze": freeze_x(prob, start | end, pose_at)}
        x, rep = rv2.solve(prob, iters=iters)
        spikes = rv2.spike_frames(prob, x)
        if spikes:
            log(f"{edge_id}: {spikes} jerk frames after the direct solve; solving gently")
            x2, rep2 = rv2.solve_gently(prob, iters=iters, x0=x)
            if rv2.spike_frames(prob, x2) <= spikes:
                x, rep = x2, rep2
        solve_s = time.time() - t0
        spikes_after = rv2.spike_frames(prob, x)
        cert, pos, quat = Q.certify(sk, prob, x, sched, times, S, Dp)
        pr = pin_report(sk, prob, x, pn, times)
        root_pos, root_rot, dofx = (v.numpy() for v in rt.unpack(prob, x))
        # the ends against the release exemplars (S in its clip frame, D placed by the hand-anchor transform)
        sp_rel = release_bodies(e["source"])
        dp_rel, dr_rel, _ = SK.exemplar(e["destination"])
        bi = {n: i for i, n in enumerate(sk.names)}
        yaw, txy = SK.hand_anchor_transform(sp_rel, dp_rel, bi)
        dp2, _ = SK.apply_planar(dp_rel, dr_rel, yaw, txy)
        planted_end = [b for z in sched.dst_ground for b in ZONES[z]]
        ends = {"start_all_bodies_max_cm": round(100 * float(np.linalg.norm(pos[0] - sp_rel, axis=-1).max()), 4),
                "end_planted_max_cm": round(100 * float(np.linalg.norm(
                    pos[-1, [bi[b] for b in planted_end]] - dp2[[bi[b] for b in planted_end]], axis=-1).max()), 3),
                "end_planted_bodies": planted_end,
                "end_all_bodies_max_cm": round(100 * float(np.linalg.norm(pos[-1] - dp2, axis=-1).max()), 3)}
        if landing:
            ends["touchdown"] = touchdown_report(sk, pos, quat, times, land, te)
        cone_out = None if cone is None else {"slope": CONE_SLOPE, "points": int(len(cands)),
                                              "frames": int(len(cf)), "touch_m": CONE_TOUCH_M}
    import edge_synthesis
    rec = {"provenance": edge_synthesis.provenance(), "card": "T6",
           "config": {"pin_pos_m": PIN_POS_M, "pin_rot_deg": math.degrees(PIN_ROT_RAD), "pin_weight": PIN_WEIGHT,
                      "pin_tol_m": PIN_TOL_M, "pin_tol_deg": PIN_TOL_DEG, "foot_pin_ramp_s": FOOT_PIN_RAMP_S,
                      "cone_slope": CONE_SLOPE, "cone_scale": CONE_SCALE, "cone_weight": CONE_WEIGHT,
                      "cone_touch_m": CONE_TOUCH_M, "edit_frames": EDIT_FRAMES, "iters": iters, "pre_s": pre,
                      "post_s": post, "quasistatic": {"open_gap_m": Q.OPEN_GAP_M, "pen_weight": Q.PEN_WEIGHT,
                                                      "guard_m": Q.GUARD_M, "statics_every": Q.STATICS_EVERY}},
           "edge": edge_id, "label": e["label"], "kind": "landing" if landing else "quasi_static", "timing": timing,
           "via": via, "durations_s": [round(d, 3) for d in durs], "T": round(T_e, 3), "touchdown_s": te,
           "frames": len(times), "t0": float(times[0]), "fps": FPS, "anchor": info, "keyframes": keyframes,
           "frozen": {"start_frames": int(start.sum()), "end_frames": int(end.sum())},
           "d_prime": d_edit, "touchdown_keyframe": land_edit, "cone": cone_out,
           "solver": {k: rep[k] for k in ("iterations", "seconds", "energy_start", "energy", "terms")},
           "solve_s": round(solve_s, 1), "spike_frames": spikes_after, "pins": pr, "ends": ends,
           "certification": cert}
    out = Path(out_root) / f"{edge_id}_{timing}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reference.npz", times=times, root_pos=root_pos, root_rot=root_rot, dof=dofx,
                        body_pos=pos, body_quat=quat, anchor_root_pos=rp, anchor_root_rot=rr, anchor_dof=dof)
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    return rec


def touchdown_report(sk, pos: np.ndarray, quat: np.ndarray, times: np.ndarray, land: Q.Pose, te: float) -> dict:
    """Per foot, the reference's first frame with a collider point below 1 cm, and how far the touching points are
    (horizontally) from their own spots on the touchdown keyframe (the executions before T6 touched down 20-68 cm
    short on B1)."""
    from reference_curation import retarget as rt

    R = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(quat.shape[:2] + (3, 3))
    pts = rt.candidate_points(sk, torch.as_tensor(pos), torch.as_tensor(R)).numpy()
    h = rt.candidate_heights(sk, torch.as_tensor(pts)).numpy()
    lp, lr = Q.bodies(sk, land)
    lpts = rt.candidate_points(sk, torch.as_tensor(lp)[None], torch.as_tensor(lr)[None])[0].numpy()
    out = {}
    for z in FOOT_ZONES:
        cz = np.asarray(sk.zone_cands[ZONE_ORDER.index(z)])
        low = h[:, cz].min(1)
        f = int(np.argmax(low < 0.01)) if (low < 0.01).any() else None
        if f is None:
            out[z] = None
            continue
        touching = cz[h[f, cz] < 0.01]
        out[z] = {"t": round(float(times[f]), 3), "scheduled_t": round(float(te), 3),
                  "points_to_spot_cm": round(100 * float(np.linalg.norm(pts[f, touching, :2] - lpts[touching, :2],
                                                                        axis=-1).max()), 2),
                  "min_height_before_cm": (round(100 * float(low[times < te - 0.2].min()), 2)
                                           if (times < te - 0.2).any() else None)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edge", default=None)
    ap.add_argument("--timing", default="mid")
    ap.add_argument("--via", default=None)
    ap.add_argument("--d1", action="store_true", help="every (edge, timing) recipe of the D1 selection")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args(argv)
    print("cpu policy", gpu_guard.be_polite(), flush=True)
    torch.set_num_threads(1)
    jobs = list(D1_RECIPES) if args.d1 else [(args.edge, args.timing, args.via)]
    ok = True
    for edge_id, timing, via in jobs:
        rec = generate(edge_id, timing, via, iters=args.iters, out_root=args.out_root)
        c, p, en = rec["certification"], rec["pins"], rec["ends"]
        print(json.dumps({"edge": edge_id, "timing": timing, "solve_s": rec["solve_s"],
                          "solver_terms": rec["solver"]["terms"], "spike_frames": rec["spike_frames"],
                          "pins": {k: v for k, v in p.items() if k != "beyond"}, "ends": en,
                          "certification": {k: v for k, v in c.items() if k != "statics"},
                          "statics_s_star_max": c["statics"]["s_star_max"], "statics_beyond": len(c["statics"]["beyond"]),
                          "d_prime": {k: rec["d_prime"][k] for k in ("edit_body_max_cm", "statics")}},
                         indent=1, default=float), flush=True)
        for b in p.get("beyond", [])[:10]:
            print("   pin beyond tolerance:", b)
        ok &= bool(c["pass"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
