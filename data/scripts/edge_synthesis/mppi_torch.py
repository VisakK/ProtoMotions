"""``mppi.py``'s sampling MPC, run in PhysX on the GPU (``physx_plant.PhysXPlant``), several seeds at once.

The algorithm is lane T's, step for step (``mppi.MPPI``): PD-target offsets on ``knots`` spline knots over a
``horizon``-step window around a base plan (the sketch's PD targets); ``samples`` perturbations ~ N(0, noise^2) per
knot and joint, sample 0 the current offset itself; ``iters`` updates per replan with weights
``softmax(-(c - c_min) / (lam (c_max - c_min)))``; the first ``replan`` steps of the updated plan executed, then the
offset shifted forward. Two things differ, both because the plant is PhysX:

* **Blocks.** The plant's ``num_envs`` envs are ``B`` blocks of ``samples`` envs, one block per seed, each with its
  own random stream (``torch.Generator`` seeded with the seed), offset and state. A control step costs almost the
  same for 256 or 2048 envs (fixed overhead), so B seeds run in the wall time of one.
* **Execution.** The executed plan is rolled out in every env of its block from the same restored state; the
  block's first env is the executed motion (recorded at every physics substep) and the next replan starts from its
  final state. The other envs repeat it from their own grid positions, which measures how far float rounding
  alone carries the same plan in 0.1 s (``exec_spread_mm``: their largest body distance to the executed env;
  p50 4-12 mm, up to 40 mm on the landings). A whole run is reproducible: the same env count, grid and seeds
  repeat it bit for bit.

Controls are joint PD targets in PhysX's exp-map DOF coordinates (COMMON order), clamped to the range the expert's
tanh action can command (``PhysXPlant.act_lo/act_hi``), so any executed motion is one the policy's action space
contains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from edge_synthesis import costs_torch as CT
from edge_synthesis import mppi as M
from edge_synthesis.physx_plant import Frames, PhysXPlant, State

MPPIConfig = M.MPPIConfig


def make_view(plant: PhysXPlant, f: Frames) -> tuple[CT.View, dict]:
    """The cost's view of a rollout (``Frames`` at control-step boundaries) on the PhysX plant."""
    mq = plant.mass_quantities(f)
    fz = plant.zone_forces(f.ground)[..., 2]
    tau = plant.pd_torque(f.ctrl, f.dof, f.dof_vel, clip=False)
    view = CT.View(pos=f.pos, R=mq["R"], root_z=f.pos[..., 0, 2], com=mq["com"], com_vel=mq["com_vel"],
                   ang_mom=mq["ang_mom"], fz=fz, fz_max=fz, ctrl=f.ctrl, util=tau.abs() / plant.tau_lim,
                   box=plant.box_excess(f.dof))
    return view, mq


@dataclass
class Executed:
    """The executed motion of every block, at the physics rate (lists of ``Frames`` chunks, one per replan)."""
    chunks: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def frames(self) -> dict:
        """``{name: [B, T, ...]}`` numpy arrays over the whole run."""
        keys = self.chunks[0].numpy().keys()
        parts = [c.numpy() for c in self.chunks]
        return {k: np.concatenate([p[k] for p in parts], 1) for k in keys}


class MPPITorch:
    def __init__(self, plant: PhysXPlant, cost_fn, cfg: MPPIConfig, base: torch.Tensor, seeds: list[int]):
        """``base [G, nu]``: the base plan's PD targets on the run's control grid (index i = the control step
        starting at grid time i); ``cost_fn(view, j, frames) -> (cost [N], terms {name: [N]})``."""
        self.plant, self.cost_fn, self.cfg = plant, cost_fn, cfg
        self.B, self.n = len(seeds), int(cfg.samples)
        if self.B * self.n != plant.N:
            raise ValueError(f"{self.B} blocks x {self.n} samples != {plant.N} envs")
        dev = plant.device
        self.device = dev
        self.seeds = list(seeds)
        self.gens = [torch.Generator(device=dev).manual_seed(int(s)) for s in seeds]
        H, K = cfg.horizon, cfg.knots
        kt = np.linspace(0, H - 1, K)
        self.knot_idx = torch.as_tensor(np.round(kt).astype(int), device=dev)
        self.W = torch.as_tensor(M.basis(np.arange(H), kt, cfg.interp), dtype=torch.float32, device=dev)
        self.W_shift = torch.as_tensor(M.basis(np.arange(H) + cfg.replan, kt, cfg.interp), dtype=torch.float32,
                                       device=dev)
        nu = base.shape[1]
        self.offset = torch.zeros(self.B, K, nu, device=dev)
        sig = torch.full((nu,), float(cfg.noise), device=dev)
        if cfg.noise_scale is not None:
            sig = sig * torch.as_tensor(cfg.noise_scale, dtype=torch.float32, device=dev)
        self.sigma = sig
        self.base = base.to(dev).float()
        self.first = torch.arange(self.B, device=dev) * self.n         # each block's executed env

    def controls(self, knots: torch.Tensor, j: int) -> torch.Tensor:
        """``knots [B, s, K, nu]`` -> PD targets ``[B, s, H, nu]`` for the horizon starting at grid index ``j``."""
        H = self.cfg.horizon
        base = self.base[j:j + H]
        if base.shape[0] < H:
            base = torch.cat([base, base[-1:].expand(H - base.shape[0], -1)], 0)
        return base[None, None] + torch.einsum("hk,bsku->bshu", self.W, knots)

    def plan(self, state: State, j: int, iters: int | None = None, sigma_scale: float = 1.0) -> list[dict]:
        cfg = self.cfg
        infos = [{} for _ in range(self.B)]
        for _ in range(cfg.iters if iters is None else iters):
            eps = torch.stack([torch.randn((self.n, cfg.knots, self.sigma.numel()), generator=g, device=self.device)
                               for g in self.gens]) * (self.sigma * sigma_scale)
            eps[:, 0] = 0.0
            knots = self.offset[:, None] + eps                                     # [B, n, K, nu]
            ctrl = self.controls(knots, j).reshape(self.B * self.n, cfg.horizon, -1)
            frames, _ = self.plant.rollout(state, ctrl)
            cost, terms = self.cost_fn(frames, j)
            bad = ~torch.isfinite(cost) | ~torch.isfinite(frames.pos).flatten(1).all(1)
            cost = torch.where(bad, torch.full_like(cost, float("inf")), cost).reshape(self.B, self.n)
            fin = torch.isfinite(cost)
            cmax = torch.where(fin, cost, torch.full_like(cost, -float("inf"))).max(1, keepdim=True).values
            c = torch.where(fin, cost, cmax)
            cmin = c.min(1, keepdim=True).values
            cn = (c - cmin) / (cmax - cmin + 1e-12)
            w = torch.exp(-cn / cfg.lam) * fin
            w = w / w.sum(1, keepdim=True).clamp(min=1e-30)
            upd = torch.einsum("bn,bnku->bku", w, knots)
            ok = fin.any(1)
            self.offset = torch.where(ok[:, None, None], upd, self.offset)
            best = c.argmin(1)
            for b in range(self.B):
                kb = int(b * self.n + best[b])
                infos[b] = {"cost_min": float(c[b].min()), "cost_nominal": float(c[b, 0]),
                            "ess": float(1.0 / (w[b] ** 2).sum()), "unstable": float((~fin[b]).float().mean()),
                            "terms_best": {k: float(v[kb]) for k, v in terms.items()}}
        return infos

    def execute(self, state: State, j: int) -> tuple[State, Frames, torch.Tensor]:
        cfg = self.cfg
        ctrl = self.controls(self.offset[:, None], j)[:, :, :cfg.replan]           # [B, 1, r, nu]
        ctrl = ctrl.expand(self.B, self.n, cfg.replan, -1).reshape(self.B * self.n, cfg.replan, -1)
        frames, sub = self.plant.rollout(state, ctrl, sub_rows=self.first)
        p = frames.pos[:, -1].reshape(self.B, self.n, 24, 3)
        spread = (p - p[:, :1]).norm(dim=-1).amax((1, 2))                            # [B]
        new_state = self.plant.read(self.first)
        vals = self.W_shift @ self.offset                                            # [B, H, nu]
        self.offset = vals[:, self.knot_idx]
        return new_state, sub, spread

    def optimise_start(self, state: State, j: int, iters: int, sigma_from: float = 3.0) -> list[dict]:
        """``iters`` updates of the first plan from the fixed start ``state``, nothing executed, the noise annealed
        geometrically from ``sigma_from`` x to 1 x. On PhysX the PD-at-pose plan sags at once (crow: root -12 cm in
        0.3 s, before 0.1 s replanning can react; MuJoCo's crow sagged 2.4 cm), so the offsets that hold the start
        must be in the plan before the first step is taken. Returns one info row per update."""
        rows = []
        for k in range(iters):
            scale = sigma_from ** (1.0 - k / max(iters - 1, 1))
            infos = self.plan(state, j, iters=1, sigma_scale=scale)
            rows.append({"k": k, "sigma_scale": round(scale, 3),
                         "cost_min": [round(i["cost_min"], 4) for i in infos],
                         "cost_nominal": [round(i["cost_nominal"], 4) for i in infos]})
        return rows

    def run(self, state0: State, n_steps: int, j0: int = 0, monitor=None, verbose: bool = False) -> Executed:
        """``n_steps`` control steps from grid index ``j0`` (a multiple of ``replan``), every block from its row of
        ``state0`` (``[B]``)."""
        ex = Executed()
        state, j = state0, j0
        while j - j0 < n_steps:
            tic = time.perf_counter()
            infos = self.plan(state, j)
            state, sub, spread = self.execute(state, j)
            ex.chunks.append(sub)
            j += self.cfg.replan
            dt = time.perf_counter() - tic
            row = {"j": j, "plan_s": round(dt, 3), "blocks": []}
            for b in range(self.B):
                info = {**infos[b], "exec_spread_mm": round(1e3 * float(spread[b]), 3)}
                if monitor is not None:
                    info.update(monitor(state, b))
                row["blocks"].append(info)
            ex.log.append(row)
            if verbose:
                print({"j": j, "plan_s": row["plan_s"],
                       "blocks": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in bi.items()
                                   if k != "terms_best"} for bi in row["blocks"]]}, flush=True)
        return ex
