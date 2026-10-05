"""MPPI variant: softmax-weighted knots (lambda on normalised cost), 2 iterations per replan, H=24 (0.8 s), K=4,
optionally stiffer PD in the sampler (gains are an action parameterisation for reference generation; torque limits
unchanged)."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np, mujoco
from mujoco import rollout
import plant_v2_mj as P
import ps_hold as S
from bench import CROW, HAND, setup

def hold_mppi(pose, T=5.0, N=192, H=24, K=4, iters=2, noise=0.05, lam=0.05, gain_scale=1.0, seed=0, nthread=22, dt=1/240):
    S.DT = dt; S.SUB = int(round(1/(dt*30)))
    old_build = P.build
    P.build = lambda dt=dt, **kw: old_build(dt=dt, gain_scale=gain_scale)
    try:
        m = S.build_with_sensors()
    finally:
        P.build = old_build
    d, qp, _, _ = setup(m, pose)
    qref, zref = qp[7:].copy(), qp[2]
    mujoco.mj_forward(m, d)
    upref = d.xmat[1].reshape(3, 3)[:, 2].copy()
    datas = [mujoco.MjData(m) for _ in range(nthread)]
    rng = np.random.default_rng(seed)
    nominal = np.tile(qref, (K, 1))
    steps, k, log, tps = int(T*30), 0, [], []
    tk = np.linspace(0, H-1, K)
    while k < steps:
        s0 = P.full_state(m, d); t0 = time.perf_counter()
        for it in range(iters):
            knots, ctrl = S.plan_controls(nominal, noise, rng, N, H, K)
            st, sd = rollout.rollout(m, datas, np.tile(s0, (N, 1)), ctrl, nstep=H*S.SUB, persistent_pool=True)
            c = S.cost(sd, st, qref, zref, upref, m, ctrl)
            cn = (c - c.min()) / (c.max() - c.min() + 1e-9)
            w = np.exp(-cn / lam); w /= w.sum()
            nominal = np.einsum("n,nku->ku", w, knots)
        _, ctrl = S.plan_controls(nominal, 0.0, rng, 1, H, K)
        tps.append(time.perf_counter() - t0)
        for j in range(3 * S.SUB):
            d.ctrl[:] = ctrl[0, j]; mujoco.mj_step(m, d)
        k += 3
        best_ctrl = ctrl[0, ::S.SUB]
        shifted = np.vstack([best_ctrl[3:], np.repeat(best_ctrl[-1:], 3, 0)])
        nominal = shifted[np.round(tk).astype(int)]
        mujoco.mj_forward(m, d)
        up = d.xmat[1].reshape(3, 3)[:, 2]
        wmin = min(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b), 2] for b in S.WATCH)
        log.append((k/30, d.qpos[2], np.degrees(np.arccos(np.clip(up @ upref, -1, 1))), wmin))
        if not np.isfinite(d.qpos).all(): break
    rollout.shutdown_persistent_pool()
    a = np.array(log)
    ok = (a[:, 1] > a[0, 1] - 0.10).all() and a[:, 2].max() < 30 and a[:, 3].min() > 0.06 and len(a) >= steps//3
    return ok, np.mean(tps), a

for name, pose in (("crow", CROW), ("handstand", HAND)):
    for gs in (1.0, 10.0):
        oks, tp = [], []
        for seed in (0, 1, 2):
            ok, t, a = hold_mppi(pose, gain_scale=gs, seed=seed)
            oks.append(ok); tp.append(t)
        print(f"MPPI {name:9s} gains x{gs:<4}: held 5 s on {sum(oks)}/3 seeds, replan {1e3*np.mean(tp):.0f} ms per 0.1 s of motion", flush=True)
