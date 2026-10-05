"""Stability vs timestep/integrator for noisy PD-target rollouts from the crow, and throughput when stable."""
import sys
import time

import mujoco
import numpy as np
from mujoco import rollout

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import plant_v2_mj as P
from bench import CROW, STAND, HAND, setup

CTRL_HZ = 30


def run(dt, integrator, noise, pose=CROW, H=30, nbatch=256, nthread=22, limits=False, smooth=False):
    m = P.build(dt=dt, integrator=integrator, limits=limits)
    sub = int(round(1 / (dt * CTRL_HZ)))
    d, qp, _, _ = setup(m, pose)
    s0 = P.full_state(m, d)
    rng = np.random.default_rng(1)
    if smooth:
        # cubic-ish: 4 knots linearly interpolated over the horizon
        knots = qp[7:][None, None, :] + noise * rng.standard_normal((nbatch, 4, m.nu))
        tk = np.linspace(0, H - 1, 4)
        ctrl = np.stack([np.stack([np.interp(np.arange(H), tk, knots[b, :, j]) for j in range(m.nu)], 1)
                         for b in range(nbatch)])
    else:
        ctrl = qp[7:][None, None, :] + noise * rng.standard_normal((nbatch, H, m.nu))
    ctrl = np.repeat(ctrl, sub, axis=1)
    init = np.tile(s0, (nbatch, 1))
    datas = [mujoco.MjData(m) for _ in range(nthread)]
    t0 = time.perf_counter()
    st, _ = rollout.rollout(m, datas, init, ctrl, nstep=H * sub, persistent_pool=True)
    el = time.perf_counter() - t0
    # state layout FULLPHYSICS: time, qpos, qvel, act, ... -> time is col 0
    tcol = st[:, :, 0]
    expected = s0[0] + dt * np.arange(1, H * sub + 1)
    reset = ~np.isclose(tcol, expected[None, :], atol=1e-6).all(1)
    finite = np.isfinite(st).all((1, 2))
    unstable = reset | ~finite
    sps = nbatch * H * sub / el
    rz = st[:, -1, 1 + 2]
    return dict(dt=dt, integ=integrator, noise=noise, sub=sub, unstable=float(unstable.mean()), ms=el * 1e3,
                ctrl_steps_per_s=sps / sub, phys_steps_per_s=sps, root_z_end_med=float(np.median(rz[~unstable]))
                if (~unstable).any() else float('nan'))


if __name__ == "__main__":
    rows = []
    for integ in ("implicitfast", "implicit"):
        for dt in (1 / 120, 1 / 240, 1 / 480):
            for noise in (0.0, 0.05, 0.15):
                r = run(dt, integ, noise)
                rows.append(r)
                print(f"{integ:12s} dt=1/{round(1/dt):3d} noise {noise:.2f}: unstable {r['unstable']:.3f}  "
                      f"{r['ms']:7.1f} ms/256x{30}ctrl  {r['ctrl_steps_per_s']:9,.0f} ctrl-steps/s  "
                      f"root z end (stable, med) {r['root_z_end_med']:.3f}", flush=True)
    rollout.shutdown_persistent_pool()
