"""Sampling MPC on plant v2 in MuJoCo (card T3's core; card T1's hold regression).

MPPI over PD-target splines (Williams et al. 2017; the setting measured in ``evidence/to_bench/mppi_grid.py``):

* **Controls** are the 69 hinge PD targets per 30 Hz control step, held for the plant's 8 physics steps (PhysX's
  decimation). A plan is ``base(t) + offset(t)``: ``base`` is the reference sketch's PD targets at each step (a
  constant pose for a hold) and ``offset`` a spline through ``knots`` knots spread over the horizon -- linear
  ("hat" weights, the benchmark's) or natural cubic.
* **Sampling**: ``samples`` knot perturbations ~ N(0, ``noise``^2) around the current offset (sample 0 = the offset
  itself); every sample is rolled out from the current state (``Plant.rollout``, a persistent thread pool).
* **Update**: ``w = softmax(-(c - c_min) / (lam (c_max - c_min)))`` on the summed cost, offset = sum w * knots,
  ``iters`` times per replan.
* **Execution**: the updated plan's first ``replan`` control steps are rolled out from the current state (the same
  rollout function, so the executed motion is exactly a planned one), then the offset is shifted forward in time
  (past the horizon it holds its last value) and the next replan starts there.

A cost function takes a ``Rollouts`` batch (states at the control-step boundaries, forces over the physics steps,
FK on demand) and returns ``(cost [N], terms {name: [N]})``. ``hold_cost`` is the benchmark's hold objective
(``evidence/to_bench/ps_hold.py``); T3's transition costs live in ``costs.py``.

CLI: ``PYTHONPATH=.:data/scripts python -m edge_synthesis.mppi --hold`` (T1's regression: crow and handstand held
5 s on 3 of 3 seeds at training gains).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from edge_synthesis import gpu_guard
from edge_synthesis import plant_mj as pm


@dataclass
class MPPIConfig:
    horizon: int = 24            # control steps (0.8 s)
    knots: int = 4
    samples: int = 192
    iters: int = 2
    lam: float = 0.05            # temperature on the min-max normalised cost
    noise: float = 0.05          # rad, per knot and hinge (a scalar or a [69] array via noise_scale)
    replan: int = 3              # control steps executed per replan (0.1 s)
    interp: str = "linear"       # "linear" | "cubic"
    seed: int = 0
    noise_scale: list | None = None   # optional per-hinge multiplier of ``noise`` (69)


def basis(times: np.ndarray, knot_times: np.ndarray, kind: str) -> np.ndarray:
    """``[len(times), K]`` weights evaluating a spline with knots at ``knot_times`` at ``times`` (clamped to the knot
    span: a spline holds its end values outside it)."""
    t = np.clip(np.asarray(times, float), knot_times[0], knot_times[-1])
    k = len(knot_times)
    if kind == "linear":
        out = np.zeros((len(t), k))
        for i, x in enumerate(t):
            j = int(np.clip(np.searchsorted(knot_times, x, side="right") - 1, 0, k - 2))
            a = (x - knot_times[j]) / (knot_times[j + 1] - knot_times[j])
            out[i, j], out[i, j + 1] = 1 - a, a
        return out
    if kind == "cubic":
        return CubicSpline(knot_times, np.eye(k), bc_type="natural")(t)
    raise ValueError(kind)


class Rollouts:
    """A batch of rollouts at the control-step boundaries (``[N, H, ...]``)."""

    def __init__(self, plant: pm.Plant, state: np.ndarray, sens: np.ndarray, ctrl: np.ndarray, t0: float):
        sub = plant.sub
        n, steps = state.shape[:2]
        h = steps // sub
        self.plant = plant
        self.n, self.h = n, h
        b = state[:, sub - 1::sub]
        self.time = b[0, :, 0]
        self.qpos = b[..., plant.qpos_slice]
        self.qvel = b[..., plant.qvel_slice]
        self.ctrl = ctrl
        self.t = t0 + plant.dt * sub * (np.arange(h) + 1)          # absolute time of each boundary
        s = sens.reshape(n, h, sub, -1)
        self.com = plant.com(s[:, :, -1])
        self.com_vel = plant.com_vel(s[:, :, -1])
        self.ang_mom = plant.ang_mom(s[:, :, -1])
        f = plant.zone_forces(s)                                    # [N, H, sub, 15, 3]
        self.force = f.mean(2)                                      # mean over the control step
        self.force_z_max = f[..., 2].max(2)                         # peak normal load inside the step
        start_t = state[:, 0, 0] - plant.dt
        self.unstable = ~np.isclose(state[:, :, 0], start_t[:, None] + plant.dt * (np.arange(steps) + 1)[None],
                                    atol=1e-6).all(1) | ~np.isfinite(state).all((1, 2))
        self._fk = None

    def fk(self) -> tuple[np.ndarray, np.ndarray]:
        if self._fk is None:
            self._fk = self.plant.fk(self.qpos)
        return self._fk


@dataclass
class Trajectory:
    """The executed motion at the physics rate (``qpos``/``qvel``/``ctrl``/``sensordata`` per physics step)."""
    qpos: list = field(default_factory=list)
    qvel: list = field(default_factory=list)
    ctrl: list = field(default_factory=list)
    sens: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def arrays(self) -> dict:
        return {k: np.concatenate(getattr(self, k), 0) for k in ("qpos", "qvel", "ctrl", "sens")}


class MPPI:
    def __init__(self, plant: pm.Plant, cost_fn, cfg: MPPIConfig, base_fn):
        """``base_fn(t [H]) -> [H, 69]`` the sketch's PD targets at absolute times (the control step's start)."""
        self.plant, self.cost_fn, self.cfg, self.base_fn = plant, cost_fn, cfg, base_fn
        self.rng = np.random.default_rng(cfg.seed)
        self.knot_times = np.linspace(0, cfg.horizon - 1, cfg.knots)
        self.W = basis(np.arange(cfg.horizon), self.knot_times, cfg.interp)            # [H, K]
        self.W_shift = basis(np.arange(cfg.horizon) + cfg.replan, self.knot_times, cfg.interp)
        self.offset = np.zeros((cfg.knots, plant.nu))                                 # knot offsets
        sig = np.full(plant.nu, cfg.noise)
        if cfg.noise_scale is not None:
            sig = sig * np.asarray(cfg.noise_scale, float)
        self.sigma = sig
        self.dt_ctrl = plant.dt * plant.sub

    def controls(self, knots: np.ndarray, t0: float) -> np.ndarray:
        """``knots [N, K, 69]`` -> PD targets ``[N, H, 69]`` from absolute time ``t0``."""
        base = self.base_fn(t0 + self.dt_ctrl * np.arange(self.cfg.horizon))
        return base[None] + np.einsum("hk,nku->nhu", self.W, knots)

    def plan(self, state: np.ndarray, t0: float) -> dict:
        cfg = self.cfg
        info = {}
        for it in range(cfg.iters):
            eps = self.rng.standard_normal((cfg.samples, cfg.knots, self.plant.nu)) * self.sigma
            eps[0] = 0.0
            knots = self.offset[None] + eps
            ctrl = self.controls(knots, t0)
            st, sd = self.plant.rollout(state, ctrl)
            r = Rollouts(self.plant, st, sd, ctrl, t0)
            cost, terms = self.cost_fn(r)
            cost = np.where(r.unstable, np.inf, cost)
            fin = np.isfinite(cost)
            if not fin.any():
                info = {"all_unstable": True}
                break
            c = np.where(fin, cost, cost[fin].max())
            cn = (c - c.min()) / (c.max() - c.min() + 1e-12)
            w = np.exp(-cn / cfg.lam)
            w[~fin] = 0.0
            w /= w.sum()
            self.offset = np.einsum("n,nku->ku", w, knots)
            best = int(np.argmin(c))
            info = {"cost_min": float(c.min()), "cost_nominal": float(c[0]), "ess": float(1.0 / (w ** 2).sum()),
                    "unstable": float(r.unstable.mean()), "terms_best": {k: float(v[best]) for k, v in terms.items()}}
        return info

    def execute(self, state: np.ndarray, t0: float, traj: Trajectory) -> np.ndarray:
        cfg = self.cfg
        ctrl = self.controls(self.offset[None], t0)[:, : cfg.replan]
        st, sd = self.plant.rollout(state, ctrl)
        traj.qpos.append(st[0, :, self.plant.qpos_slice])
        traj.qvel.append(st[0, :, self.plant.qvel_slice])
        traj.ctrl.append(np.repeat(ctrl[0], self.plant.sub, 0))
        traj.sens.append(sd[0])
        self.offset = self.W_shift_knots()
        return st[0, -1]

    def W_shift_knots(self) -> np.ndarray:
        """The offset re-knotted ``replan`` steps later (holding its last value past the horizon)."""
        vals = self.W_shift @ self.offset                       # [H, 69] the old spline at the shifted times
        idx = np.round(self.knot_times).astype(int)
        return vals[idx]

    def run(self, state0: np.ndarray, seconds: float, t0: float = 0.0, monitor=None, verbose: bool = False) -> Trajectory:
        traj = Trajectory()
        state, t = state0, t0
        steps = int(round(seconds / self.dt_ctrl))
        k = 0
        while k < steps:
            tic = time.perf_counter()
            info = self.plan(state, t)
            if info.get("all_unstable"):
                traj.log.append({"t": t, "all_unstable": True})
                break
            state = self.execute(state, t, traj)
            t += self.cfg.replan * self.dt_ctrl
            k += self.cfg.replan
            info["t"] = round(t, 4)
            info["plan_s"] = round(time.perf_counter() - tic, 3)
            if monitor is not None:
                info.update(monitor(state))
            traj.log.append(info)
            if verbose:
                print(json.dumps(info), flush=True)
            if not np.isfinite(state).all():
                break
        return traj


# --------------------------------------------------------------------------- #
# The benchmark's hold objective (evidence/to_bench/ps_hold.py) and T1's regression
# --------------------------------------------------------------------------- #
WATCH = ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "Head", "L_Elbow", "R_Elbow")
HOLD_POSES = {"crow": ("220923_Crane_Crow_Pose_or_Bakasana_-a", 651),
              "handstand": ("220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a", 1193)}


def hold_cost(plant: pm.Plant, qref: np.ndarray, zref: float, upref: np.ndarray):
    bi = plant.body_index
    watch = [bi[b] for b in WATCH]
    wrists = [bi["L_Wrist"], bi["R_Wrist"]]

    def cost(r: Rollouts):
        pos, rot = r.fk()
        hands = pos[:, :, wrists].mean(2)
        up = rot[:, :, 0, :, 2]
        terms = {
            "com_over_hands": 50.0 * ((r.com[..., :2] - hands[..., :2]) ** 2).sum(-1),
            "root_height": 20.0 * (r.qpos[..., 2] - zref) ** 2,
            "upright": 5.0 * (1.0 - (up * upref).sum(-1)),
            "joints": 0.05 * ((r.qpos[..., 7:] - qref) ** 2).sum(-1),
            "watch_down": 100.0 * (pos[:, :, watch, 2] < 0.06).sum(-1),
            "ctrl": 0.01 * ((r.ctrl - qref) ** 2).sum(-1),
        }
        terms = {k: v.sum(1) for k, v in terms.items()}
        return sum(terms.values()), terms
    return cost


def hold(plant: pm.Plant, pose: tuple, seconds: float = 5.0, cfg: MPPIConfig | None = None) -> dict:
    cfg = cfg or MPPIConfig()
    pos, rot, _ = pm.release_frame(*pose)
    q = plant.qpos_from_bodies(pos, rot)
    s0, d = plant.make_state(q)
    qref, zref = q[7:].copy(), float(q[2])
    upref = d.xmat[1].reshape(3, 3)[:, 2].copy()
    bi = plant.body_index
    watch = [bi[b] for b in WATCH]

    def monitor(state):
        qp = state[plant.qpos_slice]
        p, r = plant.fk(qp[None])
        up = r[0, 0, :, 2]
        return {"root_z": round(float(qp[2]), 4), "tilt_deg": round(float(np.degrees(np.arccos(np.clip(up @ upref, -1, 1)))), 2),
                "watch_min_z": round(float(p[0, watch, 2].min()), 4)}

    ctl = MPPI(plant, hold_cost(plant, qref, zref, upref), cfg, lambda t: np.tile(qref, (len(t), 1)))
    tic = time.perf_counter()
    traj = ctl.run(s0, seconds, monitor=monitor)
    wall = time.perf_counter() - tic
    log = [l for l in traj.log if "root_z" in l]
    ok = len(log) >= int(round(seconds * pm.CTRL_HZ)) // cfg.replan and \
        all(l["root_z"] > zref - 0.10 for l in log) and max(l["tilt_deg"] for l in log) < 30 and \
        min(l["watch_min_z"] for l in log) > 0.06
    return {"held": bool(ok), "seconds": seconds, "wall_s": round(wall, 1),
            "plan_ms_per_0.1s": round(1e3 * float(np.mean([l["plan_s"] for l in log])), 0) if log else None,
            "root_z_drop_max_m": round(zref - min(l["root_z"] for l in log), 4) if log else None,
            "tilt_max_deg": max(l["tilt_deg"] for l in log) if log else None,
            "watch_min_z": min(l["watch_min_z"] for l in log) if log else None}


def hold_acceptance(nthread: int = gpu_guard.POLITE_THREADS, seeds=(0, 1, 2), seconds: float = 5.0) -> dict:
    out = {}
    for name, pose in HOLD_POSES.items():
        runs = []
        for seed in seeds:
            plant = pm.Plant(nthread=nthread)
            runs.append({"seed": seed, **hold(plant, pose, seconds, MPPIConfig(seed=seed))})
            plant.close()
        out[name] = {"held": sum(r["held"] for r in runs), "of": len(runs), "runs": runs}
    out["config"] = asdict(MPPIConfig())
    out["pass"] = all(out[n]["held"] == out[n]["of"] for n in HOLD_POSES)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--threads", type=int, default=gpu_guard.POLITE_THREADS)
    args = ap.parse_args(argv)
    if args.threads > 20:
        print("refusing > 20 threads while a GPU run may be active", file=sys.stderr)
        return 2
    print("cpu policy", gpu_guard.be_polite())
    if args.hold:
        res = hold_acceptance(args.threads)
        print(json.dumps(res, indent=1))
        return 0 if res["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
