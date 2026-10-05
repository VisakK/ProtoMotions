"""Card T3's cost library: what a sampled transition on plant v2 is charged, term by term.

Every term is dimensionless (a residual divided by its ``SCALES`` entry, squared or hinged) and weighted by
``EdgeWeights``; MPPI normalises the summed cost per replan, so only the ratios matter. All terms are evaluated at the
control-step boundaries of a ``mppi.Rollouts`` batch, except the landing force (the peak over the step's physics
steps). Per term, with its source in PLAN.MD's T3 card:

==================  =========================================================================================
term                charge
==================  =========================================================================================
``track``           mean squared body-origin distance to the sketch (all 24 bodies), decaying from
                    ``track`` to ``track_end`` over the edge ("decaying tracking of the nominal")
``terminal``        from the arrival on: the 24 bodies, the root's height and up axis against D, and the COM
                    speed ("terminal attainment ... plus COM velocity")
``support``         a planted zone's lowest collider point above 1 cm ("scheduled supports kept planted")
``anchor``          a hand planted since the departure drifting horizontally from where S had it ("no slip")
``slip``            any zone carrying load near the floor while its bodies move horizontally (making zones
                    included; body origins, since a zone's lowest point jumps between box corners as it rocks)
``place``           a making zone landing away from where D has it (from its event on)
``flat``            a hand planted since the departure rocking off its palm: the tilt of its boxes' down faces
                    (a rocking palm loads two corners and shuffles across the floor; measured in the first E2 run)
``free``            a known-free zone's lowest point below 2 cm, or carrying load, outside its event windows
                    ("known-free zones kept above 2 cm": the toe kickstand); the head and trunk below 6 cm
                    (``CLEAR_M``), where a fall onto the face starts
``landing``         a zone's peak normal load above 3 BW ("a landing peak-force cap of <= 3 BW")
``balance``         on quasi-static phases, the COM outside the planted supports' hull with margin
                    ("COM over the support hull"), and the angular momentum about the COM
``capture``         on quasi-static phases, the capture point (COM + COM velocity x sqrt(h / g)) away from the
                    sketch's COM: a hull term is blind to a slow topple until the COM leaves the hull, and the
                    first press execution toppled exactly so (its COM drifted 8 -> 97 cm from the hands)
``torque``          actuator utilisation above 0.8 ("torque utilisation above 0.8")
``speed``           a body faster than the corpus p99 of that body ("joint velocity and jerk, against corpus
                    percentiles"); ``smooth`` charges the PD targets' second difference
``box``             PhysX's exp-map box excess ("exp-map box excess")
``cone``            card T6's landing cone (weight 0 unless a recipe sets it): every collider point of a making
                    zone that touches the floor at D lies at least ``CONE_SLOPE`` x its horizontal distance from
                    its own spot at D (less ``CONE_TOL_M``) above that spot's height. A short touchdown and a
                    loaded foot sliding toward its spot (both measured on the first PhysX B1 runs: 15-26 cm short,
                    then 10-20 cm of slide) cost the same as their distance; landing on the spot costs nothing
==================  =========================================================================================

Self-penetration is left to the physics (MuJoCo's contacts keep every pair the plant collides apart) and checked
on the executed motion by T5's contract; braces are asked for by the sketch and the terminal term.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from extract_contact_configs import ZONE_ORDER
from edge_synthesis import plant_mj as pm
from edge_synthesis.mppi import Rollouts
from edge_synthesis.sketch import Schedule, Sketch
from reference_curation import ids

CORPUS_KIN = ids.REPO / "output/edge_synthesis/corpus_body_kinematics.json"
SCALES = {"pos": 0.05, "support": 0.01, "anchor": 0.01, "slip": 0.05, "place": 0.03, "free": 0.01, "landing": 1.0,
          "flat": 0.05, "capture": 0.02,
          "balance": 0.02, "ang_mom": 5.0, "torque": 0.1, "speed": 0.25, "smooth": 0.05, "box": 0.05,
          "height": 0.05, "up": 0.1, "com_vel": 0.1, "cone": 0.01}
LOAD_N = 5.0                 # a zone "carries load" above this normal force
FREE_M = 0.02                # known-free zones stay this far above the floor
CLEAR_M = {"HEAD": 0.06, "TRUNK": 0.06}   # ... and these this far (a face plant in the first E5 run)
PLANTED_M = 0.01             # a planted support's lowest point lies within this of the floor
LANDING_BW = 3.0
TORQUE_UTIL = 0.8
BALANCE_MARGIN_M = 0.015
CONE_SLOPE = 0.25            # exact.CONE_SLOPE: a landing point at height h lies within h / CONE_SLOPE of its spot ...
CONE_TOL_M = 0.015           # ... less the card's 1.5 cm placement tolerance
CONE_TOUCH_M = 0.015         # D's collider points this close to the floor are a making zone's landing points


@dataclass
class EdgeWeights:
    track: float = 10.0
    track_end: float = 2.0
    terminal: float = 8.0
    support: float = 4.0
    anchor: float = 30.0
    flat: float = 4.0
    slip: float = 2.0
    place: float = 8.0
    free: float = 20.0
    landing: float = 10.0
    balance: float = 4.0
    capture: float = 10.0
    ang_mom: float = 0.2
    torque: float = 0.2
    speed: float = 1.0
    smooth: float = 0.05
    box: float = 20.0
    cone: float = 0.0


def corpus_speed_p99() -> np.ndarray:
    """``[24]`` per-body speed p99 (m/s, 30 Hz finite differences, x0 release motions)."""
    return np.asarray(json.load(open(CORPUS_KIN))["speed30"]["p99"], float)


def zone_lowest(sk, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(low [.., 15], xy [.., 15, 2])``: each zone's lowest collider point (height, horizontal position)."""
    body = sk.cand_body
    pts = pos[..., body, :] + np.einsum("...kij,kj->...ki", rot[..., body, :, :], sk.cand_local)
    h = pts[..., 2] - sk.cand_radius
    low = np.empty(h.shape[:-1] + (len(ZONE_ORDER),))
    xy = np.empty(h.shape[:-1] + (len(ZONE_ORDER), 2))
    for z, idx in enumerate(sk.zone_cands):
        hz = h[..., idx]
        k = np.argmin(hz, axis=-1)
        low[..., z] = np.take_along_axis(hz, k[..., None], -1)[..., 0]
        xy[..., z, :] = np.take_along_axis(pts[..., idx, :2], k[..., None, None], -2)[..., 0, :]
    return low, xy


def support_hull(sk, pos: np.ndarray, rot: np.ndarray, zones: list[str], touch_m: float = 0.015) -> np.ndarray:
    """``[M, 3]`` half-planes ``a x + b y + c >= 0`` (inside) of the convex hull of the touching candidate points of
    ``zones`` in one pose (``pos [24,3]``, ``rot [24,3,3]``); empty if fewer than 3 points."""
    from scipy.spatial import ConvexHull, QhullError

    idx = np.concatenate([sk.zone_cands[ZONE_ORDER.index(z)] for z in zones]) if zones else np.zeros(0, int)
    if not len(idx):
        return np.zeros((0, 3))
    pts = pos[sk.cand_body[idx]] + np.einsum("kij,kj->ki", rot[sk.cand_body[idx]], sk.cand_local[idx])
    h = pts[:, 2] - sk.cand_radius[idx]
    xy = pts[h <= touch_m, :2]
    if len(xy) < 3:
        return np.zeros((0, 3))
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return np.zeros((0, 3))
    eq = hull.equations                       # n.x + d <= 0 inside
    return np.c_[-eq[:, 0], -eq[:, 1], -eq[:, 2]]


class EdgeCost:
    """The transition objective of one edge at one timing (``Schedule``), tracking ``sketch`` to the re-anchored
    destination pose ``dst_qpos``."""

    def __init__(self, plant: pm.Plant, sketch: Sketch, sched: Schedule, src_qpos: np.ndarray, dst_qpos: np.ndarray,
                 weights: EdgeWeights | None = None):
        from reference_curation import mosh_replay as mr

        self.plant, self.sketch, self.sched = plant, sketch, sched
        self.w = weights or EdgeWeights()
        self.sk = mr.skeleton_for(mr.V2_XML)
        if list(self.sk.names) != plant.body_names:
            raise ValueError("collider skeleton body order differs from the MuJoCo plant's")
        self.v99 = corpus_speed_p99()
        self.bw = plant.weight_n
        m = plant.model
        self.body_mass = m.body_mass[1:].copy()
        self.body_ipos = m.body_ipos[1:].copy()
        sp, sr = plant.fk(src_qpos[None])
        dp, dr = plant.fk(dst_qpos[None])
        self.dst_pos, self.dst_up, self.dst_z = dp[0], dr[0, 0, :, 2], float(dst_qpos[2])
        self.planted_from_start = [z for z in sched.src_ground if all(z in g for g in sched.ground)
                                   and z in sched.dst_ground]
        from extract_contact_configs import ZONES
        self.anchor_bodies = [plant.body_index[b] for z in self.planted_from_start for b in ZONES[z]]
        self.anchor_xy = sp[0, self.anchor_bodies, :2]            # where S has them
        # the box geometry of the anchored bodies: each box's down face (in its body frame) at S
        from scipy.spatial.transform import Rotation
        self.flat_rows = []
        for b in self.anchor_bodies:
            g = self.sk.geoms[self.sk.names[b]][0]
            if g["type"] != "box":
                continue
            rg = Rotation.from_quat(g["quat"]).as_matrix()
            axes_z = (sr[0, b] @ rg)[2]                           # world z of the box axes at S
            a = int(np.argmax(np.abs(axes_z)))
            self.flat_rows.append((b, rg[:, a] * -np.sign(axes_z[a])))   # body-frame normal of the down face
        self.zone_bodies = [[plant.body_index[b] for b in ZONES[z]] for z in ZONE_ORDER]
        # making zones: from their event on they should land where D has them
        self.place = [(te, ZONE_ORDER.index(z)) for te, z, kind in sched.events if kind == "make"]
        # the landing cone's points: the making zones' collider points that touch the floor at D
        sk = self.sk
        cands = np.concatenate([sk.zone_cands[zi] for _, zi in self.place]).astype(int) if self.place else \
            np.zeros(0, int)
        cb = sk.cand_body[cands]
        pts = dp[0, cb] + np.einsum("kij,kj->ki", dr[0, cb], sk.cand_local[cands])
        h = pts[:, 2] - sk.cand_radius[cands]
        keep = h < CONE_TOUCH_M
        self.cone_body, self.cone_local = cb[keep], sk.cand_local[cands][keep]
        self.cone_radius, self.cone_xy, self.cone_h0 = sk.cand_radius[cands][keep], pts[keep, :2], h[keep]
        # the support hull of every phase, from the sketch at the phase's midpoint (planted zones do not move)
        self.hulls = {}
        for p in range(-1, len(sched.phase_names) + 1):
            if p < 0:
                t, zones = 0.0, sched.src_ground
            elif p >= len(sched.phase_names):
                t, zones = sched.T, sched.dst_ground
            else:
                t, zones = 0.5 * (sched.bounds[p] + sched.bounds[p + 1]), sched.ground[p]
            q = sketch.qpos(np.array([t]))[0]
            pp, rr = plant.fk(q[None])
            self.hulls[p] = support_hull(self.sk, pp[0], rr[0], list(zones))

    def __call__(self, r: Rollouts):
        w, S, sched = self.w, SCALES, self.sched
        pos, rot = r.fk()                                          # [N, H, 24, 3], [N, H, 24, 3, 3]
        n, h = r.n, r.h
        tb = r.t                                                   # each boundary's time (s since the departure)
        m = sched.masks(tb)
        T = sched.T
        terms = {}
        # tracking (decaying) and terminal attainment
        ref = self.plant.fk(self.sketch.qpos(tb))[0]               # [H, 24, 3]
        d2 = ((pos - ref[None]) ** 2).sum(-1).mean(-1) / S["pos"] ** 2
        frac = np.clip(tb / max(T, 1e-6), 0.0, 1.0)
        w_tr = w.track + (w.track_end - w.track) * frac
        after = tb >= T
        terms["track"] = (w_tr[None] * d2 * ~after[None]).sum(1)
        e_dst = ((pos - self.dst_pos[None, None]) ** 2).sum(-1).mean(-1) / S["pos"] ** 2
        up = rot[:, :, 0, :, 2]
        term = e_dst + ((r.qpos[..., 2] - self.dst_z) / S["height"]) ** 2 + \
            ((1.0 - (up * self.dst_up).sum(-1)) / S["up"]) + (np.linalg.norm(r.com_vel, axis=-1) / S["com_vel"]) ** 2
        terms["terminal"] = w.terminal * (term * after[None]).sum(1)
        # contacts
        low, xy = zone_lowest(self.sk, pos, rot)                   # [N, H, 15]
        fz = r.force[..., 2]                                       # [N, H, 15] mean normal load
        g, fr = m["ground"][None], m["free"][None]
        terms["support"] = w.support * ((np.maximum(low - PLANTED_M, 0) / S["support"]) ** 2 * g).sum((1, 2))
        dxy = np.linalg.norm(pos[:, :, self.anchor_bodies, :2] - self.anchor_xy, axis=-1)       # [N, H, A]
        terms["anchor"] = w.anchor * ((dxy / S["anchor"]) ** 2).sum((1, 2))
        fl = np.zeros(n)
        for b, nb in self.flat_rows:
            nz = np.einsum("nhij,j->nhi", rot[:, :, b], nb)[..., 2]     # world z of the down face's normal (-1 flat)
            fl += (((1.0 + nz) / S["flat"]) ** 2).sum(1)
        terms["flat"] = w.flat * fl
        loaded = fz > LOAD_N
        vb = np.zeros((n, h, 24))
        vb[:, 1:] = np.linalg.norm(pos[:, 1:, :, :2] - pos[:, :-1, :, :2], axis=-1) * pm.CTRL_HZ
        vxy = np.stack([vb[..., b].mean(-1) for b in self.zone_bodies], -1)              # [N, H, 15]
        terms["slip"] = w.slip * ((vxy / S["slip"]) ** 2 * (loaded & (low < PLANTED_M))).sum((1, 2))
        pl = np.zeros(n)
        for te, zi in self.place:
            on = tb >= te - 0.05
            if not on.any():
                continue
            bz = self.zone_bodies[zi]
            dxy = np.linalg.norm(pos[:, on][:, :, bz, :2] - self.dst_pos[bz, :2], axis=-1).mean(-1)
            pl += ((dxy / S["place"]) ** 2).sum(1)
        terms["place"] = w.place * pl
        clear = np.array([CLEAR_M.get(z, FREE_M) for z in ZONE_ORDER])
        terms["free"] = w.free * (((np.maximum(clear - low, 0) / S["free"]) ** 2 + 4.0 * loaded) * fr).sum((1, 2))
        terms["landing"] = w.landing * (np.maximum(r.force_z_max / self.bw - LANDING_BW, 0) ** 2).sum((1, 2))
        # balance on quasi-static phases: the COM inside the phase's support hull with margin
        bal = np.zeros((n, h))
        for k in range(h):
            if not m["quasi_static"][k]:
                continue
            hp = self.hulls.get(int(m["phase"][k]))
            if hp is None or not len(hp):
                continue
            dist = (r.com[:, k, :2] @ hp[:, :2].T + hp[:, 2]).min(-1)     # signed distance inside
            bal[:, k] = (np.maximum(BALANCE_MARGIN_M - dist, 0) / S["balance"]) ** 2
        # capture point against the sketch's COM, on quasi-static phases
        rp, rr = self.plant.fk(self.sketch.qpos(tb))
        com_ref = (self.body_mass[:, None] * (rp + np.einsum("hbij,bj->hbi", rr, self.body_ipos))).sum(1) / \
            self.body_mass.sum()                                                       # [H, 3]
        h_c = np.maximum(r.com[..., 2], 0.2)
        cp = r.com[..., :2] + r.com_vel[..., :2] * np.sqrt(h_c / 9.81)[..., None]
        cap = (np.linalg.norm(cp - com_ref[None, :, :2], axis=-1) / S["capture"]) ** 2 * m["quasi_static"][None]
        terms["capture"] = w.capture * cap.sum(1)
        angm = (np.linalg.norm(r.ang_mom, axis=-1) / S["ang_mom"]) ** 2 * m["quasi_static"][None]
        terms["balance"] = w.balance * bal.sum(1) + w.ang_mom * angm.sum(1)
        # effort, speed, smoothness, the joint box
        tau = self.plant.kp * (r.ctrl - r.qpos[..., 7:]) - self.plant.kd * r.qvel[..., 6:]
        util = np.abs(tau) / self.plant.torque_limit
        terms["torque"] = w.torque * ((np.maximum(util - TORQUE_UTIL, 0) / S["torque"]) ** 2).sum((1, 2))
        spd = np.zeros((n, h, 24))
        spd[:, 1:] = np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1) * pm.CTRL_HZ
        terms["speed"] = w.speed * ((np.maximum(spd - self.v99, 0) / S["speed"]) ** 2).sum((1, 2))
        if h > 2:
            dd = r.ctrl[:, 2:] - 2 * r.ctrl[:, 1:-1] + r.ctrl[:, :-2]
            terms["smooth"] = w.smooth * ((dd / S["smooth"]) ** 2).sum((1, 2)) / self.plant.nu
        _, _, dof = self.plant.expmap_from_qpos(r.qpos)
        terms["box"] = w.box * ((self.plant.box_excess(dof) / S["box"]) ** 2).sum((1, 2))
        terms["cone"] = np.zeros(n)
        if w.cone and len(self.cone_body):
            cp = pos[:, :, self.cone_body] + np.einsum("nhkij,kj->nhki", rot[:, :, self.cone_body], self.cone_local)
            d = np.linalg.norm(cp[..., :2] - self.cone_xy, axis=-1)
            g = cp[..., 2] - self.cone_radius - self.cone_h0 - CONE_SLOPE * np.maximum(d - CONE_TOL_M, 0.0)
            terms["cone"] = w.cone * ((np.minimum(g, 0.0) / S["cone"]) ** 2).sum((1, 2))
        total = sum(terms.values())
        return total, terms

    def describe(self) -> dict:
        return {"weights": asdict(self.w), "scales": SCALES, "planted_from_start": self.planted_from_start,
                "hull_sizes": {int(k): int(len(v)) for k, v in self.hulls.items()},
                "cone": {"points": int(len(self.cone_body)), "slope": CONE_SLOPE, "tol_m": CONE_TOL_M}}
