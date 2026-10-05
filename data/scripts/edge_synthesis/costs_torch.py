"""``costs.py``'s transition objective on the GPU, for MPPI in PhysX (``edge_mppi_physx``).

Every term, scale and weight is ``costs.EdgeCost``'s; this module only moves the evaluation to torch. To keep the
two from drifting, ``EdgeCostTorch`` *builds* lane T's numpy ``EdgeCost`` (the anchor positions, the flat palm rows,
the making zones, the per-phase support hulls, the collider candidate points) and samples everything that depends
on time -- the sketch's bodies and COM, the contact schedule's masks -- once on the run's control grid. The
evaluation then reads a ``View``: the plant-specific quantities at each control-step boundary.

Two inputs differ from the MuJoCo plant's by construction, and are the plant's own:

* ``fz``/``fz_max``: MuJoCo averages the zone loads over the 8 physics steps of a control step and takes their
  peak; the PhysX plant reads its contact sensors once, after the last substep (``PhysXPlant.control_step``), so
  both are that reading during planning. The executed motion records every substep and is judged on those.
* ``util`` and ``box``: torque utilisation and box excess in each plant's own joint coordinates (MuJoCo's hinges,
  PhysX's exp-map DOFs), computed by the caller.

``tests/test_physx_port.py`` checks this module against ``costs.EdgeCost`` on MuJoCo rollouts, term by term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from extract_contact_configs import ZONE_ORDER
from edge_synthesis import costs as C


@dataclass
class View:
    """What a cost reads at each control-step boundary, ``[M, H, ...]``."""
    pos: torch.Tensor          # [M, H, 24, 3] body origins
    R: torch.Tensor            # [M, H, 24, 3, 3]
    root_z: torch.Tensor       # [M, H]
    com: torch.Tensor          # [M, H, 3]
    com_vel: torch.Tensor      # [M, H, 3]
    ang_mom: torch.Tensor      # [M, H, 3]
    fz: torch.Tensor           # [M, H, 15] zone normal load (N)
    fz_max: torch.Tensor       # [M, H, 15] peak zone normal load within the step (N)
    ctrl: torch.Tensor         # [M, H, nu] PD targets
    util: torch.Tensor         # [M, H, nu] |PD torque| / limit (unclipped)
    box: torch.Tensor          # [M, H, nu] rad past the exp-map box


class ZoneLowest:
    """``costs.zone_lowest`` (the lowest collider point of each zone) in torch."""

    def __init__(self, sk, device):
        self.body = torch.as_tensor(np.asarray(sk.cand_body), dtype=torch.long, device=device)
        self.local = torch.as_tensor(np.asarray(sk.cand_local), dtype=torch.float32, device=device)
        self.radius = torch.as_tensor(np.asarray(sk.cand_radius), dtype=torch.float32, device=device)
        mask = torch.zeros(len(ZONE_ORDER), len(self.body), dtype=torch.bool, device=device)
        for z, idx in enumerate(sk.zone_cands):
            mask[z, torch.as_tensor(np.asarray(idx), dtype=torch.long)] = True
        self.mask = mask

    def __call__(self, pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        pts = pos[..., self.body, :] + (R[..., self.body, :, :] @ self.local[:, :, None])[..., 0]
        h = pts[..., 2] - self.radius                                                 # [..., C]
        h = torch.where(self.mask, h[..., None, :], torch.full_like(h[..., None, :], float("inf")))
        return h.min(-1).values                                                       # [..., 15]


class Grid:
    """Values of the run on its control grid: index ``i`` <-> time ``t_start + i * dt``."""

    def __init__(self, t_start: float, dt: float, n: int):
        self.t_start, self.dt, self.n = float(t_start), float(dt), int(n)
        self.t = t_start + dt * np.arange(n)


class EdgeCostTorch:
    """``costs.EdgeCost`` on a control grid. ``__call__(view, j)``: the horizon whose first control step starts at
    grid index ``j`` (its boundaries are grid indices ``j + 1 .. j + H``)."""

    def __init__(self, plant_mj, sketch, sched, src_qpos, dst_qpos, weights: C.EdgeWeights, grid: Grid, device,
                 horizon: int):
        self.np_cost = C.EdgeCost(plant_mj, sketch, sched, src_qpos, dst_qpos, weights)
        npc = self.np_cost
        self.w, self.S, self.sched, self.H = npc.w, C.SCALES, sched, int(horizon)
        self.device = device
        f32 = dict(dtype=torch.float32, device=device)
        self.zone_lowest = ZoneLowest(npc.sk, device)
        self.v99 = torch.as_tensor(npc.v99, **f32)
        self.bw = float(npc.bw)
        self.dst_pos = torch.as_tensor(npc.dst_pos, **f32)
        self.dst_up = torch.as_tensor(npc.dst_up, **f32)
        self.dst_z = float(npc.dst_z)
        self.anchor_bodies = torch.as_tensor(npc.anchor_bodies, dtype=torch.long, device=device)
        self.anchor_xy = torch.as_tensor(npc.anchor_xy, **f32)
        self.flat_b = torch.as_tensor([b for b, _ in npc.flat_rows], dtype=torch.long, device=device)
        self.flat_n = torch.as_tensor(np.array([nb for _, nb in npc.flat_rows]).reshape(-1, 3), **f32)
        zb = torch.zeros(len(ZONE_ORDER), 24, **f32)
        for z, bs in enumerate(npc.zone_bodies):
            zb[z, bs] = 1.0 / len(bs)
        self.zone_mean = zb                                                          # [15, 24] body -> zone mean
        self.clear = torch.as_tensor([C.CLEAR_M.get(z, C.FREE_M) for z in ZONE_ORDER], **f32)
        # time-dependent values on the grid (padded by the horizon so any window slices)
        self.grid = grid
        t = grid.t
        q = sketch.qpos(t)
        rp, rr = plant_mj.fk(q)
        self.ref_pos = torch.as_tensor(rp, **f32)                                    # [G, 24, 3]
        mass, ipos = npc.body_mass, npc.body_ipos
        com_ref = (mass[:, None] * (rp + np.einsum("hbij,bj->hbi", rr, ipos))).sum(1) / mass.sum()
        self.com_ref = torch.as_tensor(com_ref, **f32)
        m = sched.masks(t)
        self.g_mask = torch.as_tensor(m["ground"], device=device)
        self.free_mask = torch.as_tensor(m["free"], device=device)
        self.qs = torch.as_tensor(m["quasi_static"], device=device)
        T = sched.T
        self.t = torch.as_tensor(t, **f32)
        frac = np.clip(t / max(T, 1e-6), 0.0, 1.0)
        self.w_track = torch.as_tensor(npc.w.track + (npc.w.track_end - npc.w.track) * frac, **f32)
        self.after = torch.as_tensor(t >= T, device=device)
        # the balance hull of each grid instant (its phase's), padded with always-satisfied rows
        rows = []
        for i in range(len(t)):
            hp = npc.hulls.get(int(m["phase"][i])) if m["quasi_static"][i] else None
            rows.append(np.zeros((0, 3)) if hp is None else np.asarray(hp))
        rmax = max(1, max(len(r) for r in rows))
        hull = np.zeros((len(t), rmax, 3))
        hull[:, :, 2] = 1e3                                                           # a + b . 0 + 1e3 >= 0
        valid = np.zeros(len(t), bool)
        for i, r in enumerate(rows):
            if len(r):
                hull[i, :len(r)] = r
                valid[i] = True
        self.hull = torch.as_tensor(hull, **f32)
        self.hull_valid = torch.as_tensor(valid, device=device)
        # making zones: from their event on they should land where D has them
        on = np.zeros((len(t), max(1, len(npc.place))), bool)
        self.place_zone = []
        for k, (te, zi) in enumerate(npc.place):
            on[:, k] = t >= te - 0.05
            self.place_zone.append(torch.as_tensor(npc.zone_bodies[zi], dtype=torch.long, device=device))
        self.place_on = torch.as_tensor(on, device=device)
        # the landing cone's points (costs.EdgeCost's)
        self.cone_body = torch.as_tensor(np.asarray(npc.cone_body), dtype=torch.long, device=device)
        self.cone_local = torch.as_tensor(np.asarray(npc.cone_local).reshape(-1, 3), **f32)
        self.cone_radius = torch.as_tensor(np.asarray(npc.cone_radius), **f32)
        self.cone_xy = torch.as_tensor(np.asarray(npc.cone_xy).reshape(-1, 2), **f32)
        self.cone_h0 = torch.as_tensor(np.asarray(npc.cone_h0), **f32)

    def _window(self, j: int) -> slice:
        return slice(j + 1, j + 1 + self.H)

    def __call__(self, v: View, j: int):
        w, S = self.w, self.S
        sl = self._window(j)
        M, H = v.pos.shape[:2]
        terms = {}
        ref = self.ref_pos[sl]
        d2 = ((v.pos - ref[None]) ** 2).sum(-1).mean(-1) / S["pos"] ** 2               # [M, H]
        after = self.after[sl][None]
        terms["track"] = (self.w_track[sl][None] * d2 * ~after).sum(1)
        e_dst = ((v.pos - self.dst_pos) ** 2).sum(-1).mean(-1) / S["pos"] ** 2
        up = v.R[:, :, 0, :, 2]
        term = e_dst + ((v.root_z - self.dst_z) / S["height"]) ** 2 + (1.0 - (up * self.dst_up).sum(-1)) / S["up"] \
            + (v.com_vel.norm(dim=-1) / S["com_vel"]) ** 2
        terms["terminal"] = w.terminal * (term * after).sum(1)
        low = self.zone_lowest(v.pos, v.R)                                              # [M, H, 15]
        g, fr = self.g_mask[sl][None], self.free_mask[sl][None]
        terms["support"] = w.support * (((low - C.PLANTED_M).clamp(min=0) / S["support"]) ** 2 * g).sum((1, 2))
        dxy = (v.pos[:, :, self.anchor_bodies, :2] - self.anchor_xy).norm(dim=-1)
        terms["anchor"] = w.anchor * ((dxy / S["anchor"]) ** 2).sum((1, 2))
        if len(self.flat_b):
            nz = (v.R[:, :, self.flat_b] @ self.flat_n[:, :, None])[..., 2, 0]          # [M, H, F]
            terms["flat"] = w.flat * (((1.0 + nz) / S["flat"]) ** 2).sum((1, 2))
        else:
            terms["flat"] = torch.zeros(M, device=self.device)
        loaded = v.fz > C.LOAD_N
        vb = torch.zeros(M, H, 24, device=self.device)
        vb[:, 1:] = (v.pos[:, 1:, :, :2] - v.pos[:, :-1, :, :2]).norm(dim=-1) * 30.0
        vxy = vb @ self.zone_mean.T                                                     # [M, H, 15]
        terms["slip"] = w.slip * ((vxy / S["slip"]) ** 2 * (loaded & (low < C.PLANTED_M))).sum((1, 2))
        pl = torch.zeros(M, device=self.device)
        for k, bz in enumerate(self.place_zone):
            on = self.place_on[sl, k]
            dxy = (v.pos[:, :, bz, :2] - self.dst_pos[bz, :2]).norm(dim=-1).mean(-1)   # [M, H]
            pl = pl + ((dxy / S["place"]) ** 2 * on[None]).sum(1)
        terms["place"] = w.place * pl
        terms["free"] = w.free * ((((self.clear - low).clamp(min=0) / S["free"]) ** 2 + 4.0 * loaded) * fr).sum((1, 2))
        terms["landing"] = w.landing * ((v.fz_max / self.bw - C.LANDING_BW).clamp(min=0) ** 2).sum((1, 2))
        # balance: the COM inside its phase's support hull with margin, on quasi-static instants
        hp = self.hull[sl]                                                              # [H, R, 3]
        dist = (torch.einsum("mhk,hrk->mhr", v.com[..., :2], hp[..., :2]) + hp[None, :, :, 2]).min(-1).values
        bal = ((C.BALANCE_MARGIN_M - dist).clamp(min=0) / S["balance"]) ** 2 * self.hull_valid[sl][None]
        qs = self.qs[sl][None].float()
        h_c = v.com[..., 2].clamp(min=0.2)
        cp = v.com[..., :2] + v.com_vel[..., :2] * torch.sqrt(h_c / 9.81)[..., None]
        cap = ((cp - self.com_ref[sl][None, :, :2]).norm(dim=-1) / S["capture"]) ** 2 * qs
        terms["capture"] = w.capture * cap.sum(1)
        angm = (v.ang_mom.norm(dim=-1) / S["ang_mom"]) ** 2 * qs
        terms["balance"] = w.balance * bal.sum(1) + w.ang_mom * angm.sum(1)
        terms["torque"] = w.torque * (((v.util - C.TORQUE_UTIL).clamp(min=0) / S["torque"]) ** 2).sum((1, 2))
        spd = torch.zeros(M, H, 24, device=self.device)
        spd[:, 1:] = (v.pos[:, 1:] - v.pos[:, :-1]).norm(dim=-1) * 30.0
        terms["speed"] = w.speed * (((spd - self.v99).clamp(min=0) / S["speed"]) ** 2).sum((1, 2))
        if H > 2:
            dd = v.ctrl[:, 2:] - 2 * v.ctrl[:, 1:-1] + v.ctrl[:, :-2]
            terms["smooth"] = w.smooth * ((dd / S["smooth"]) ** 2).sum((1, 2)) / v.ctrl.shape[-1]
        terms["box"] = w.box * ((v.box / S["box"]) ** 2).sum((1, 2))
        terms["cone"] = torch.zeros(M, device=self.device)
        if w.cone and len(self.cone_body):
            cp = v.pos[:, :, self.cone_body] + (v.R[:, :, self.cone_body] @ self.cone_local[:, :, None])[..., 0]
            d = (cp[..., :2] - self.cone_xy).norm(dim=-1)
            g = cp[..., 2] - self.cone_radius - self.cone_h0 - C.CONE_SLOPE * (d - C.CONE_TOL_M).clamp(min=0)
            terms["cone"] = w.cone * ((g.clamp(max=0) / S["cone"]) ** 2).sum((1, 2))
        total = sum(terms.values())
        return total, terms

    def describe(self) -> dict:
        return self.np_cost.describe()


class HoldCostTorch:
    """``mppi.hold_cost`` (the benchmark's hold objective, T1's regression) in torch."""

    WATCH = ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "Head", "L_Elbow", "R_Elbow")

    WEIGHTS = {"com_over_hands": 50.0, "root_height": 20.0, "upright": 5.0, "joints": 0.05, "watch_down": 100.0,
               "ctrl": 0.01}

    def __init__(self, body_names, qref: torch.Tensor, zref: float, upref: torch.Tensor, weights: dict | None = None):
        self.watch = [body_names.index(b) for b in self.WATCH]
        self.wrists = [body_names.index("L_Wrist"), body_names.index("R_Wrist")]
        self.qref, self.zref, self.upref = qref, float(zref), upref
        self.w = {**self.WEIGHTS, **(weights or {})}

    def __call__(self, v: View, j: int, dof: torch.Tensor):
        hands = v.pos[:, :, self.wrists].mean(2)
        up = v.R[:, :, 0, :, 2]
        w = self.w
        terms = {
            "com_over_hands": w["com_over_hands"] * ((v.com[..., :2] - hands[..., :2]) ** 2).sum(-1),
            "root_height": w["root_height"] * (v.root_z - self.zref) ** 2,
            "upright": w["upright"] * (1.0 - (up * self.upref).sum(-1)),
            "joints": w["joints"] * ((dof - self.qref) ** 2).sum(-1),
            "watch_down": w["watch_down"] * (v.pos[:, :, self.watch, 2] < 0.06).float().sum(-1),
            "ctrl": w["ctrl"] * ((v.ctrl - self.qref) ** 2).sum(-1),
        }
        terms = {k: x.sum(1) for k, x in terms.items()}
        return sum(terms.values()), terms


class PoseHoldCostTorch:
    """Holding a pose that is not a hands-only balance (chaturanga, plank, tripod): ``costs.EdgeCost``'s terminal
    attainment term -- the 24 bodies against the exemplar, root height, root up axis, COM speed -- plus its torque
    and box terms, at ``EdgeWeights``' weights. (``HoldCostTorch`` centres the COM over the hands, which is the
    benchmark's objective for crow and the handstand and wrong for a pose resting on hands and feet.)"""

    def __init__(self, pos_ref: torch.Tensor, up_ref: torch.Tensor, z_ref: float, weights: C.EdgeWeights | None = None):
        self.pos_ref, self.up_ref, self.z_ref = pos_ref, up_ref, float(z_ref)
        self.w = weights or C.EdgeWeights()

    def __call__(self, v: View, j: int, dof: torch.Tensor):
        S, w = C.SCALES, self.w
        e = ((v.pos - self.pos_ref) ** 2).sum(-1).mean(-1) / S["pos"] ** 2
        up = v.R[:, :, 0, :, 2]
        term = e + ((v.root_z - self.z_ref) / S["height"]) ** 2 + (1.0 - (up * self.up_ref).sum(-1)) / S["up"] \
            + (v.com_vel.norm(dim=-1) / S["com_vel"]) ** 2
        terms = {"terminal": w.terminal * term.sum(1),
                 "torque": w.torque * (((v.util - C.TORQUE_UTIL).clamp(min=0) / S["torque"]) ** 2).sum((1, 2)),
                 "box": w.box * ((v.box / S["box"]) ** 2).sum((1, 2))}
        return sum(terms.values()), terms
