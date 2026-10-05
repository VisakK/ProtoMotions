"""Can plain predictive sampling (no policy prior) hold the crow / handstand on plant v2 in MuJoCo?

Receding horizon: H control steps (30 Hz), K linear-interpolated knots of PD-target offsets (69-D each), N samples
around the shifted nominal (sample 0 = the nominal), best-of-N (predictive sampling, Howell et al. 2022). Cost from
sensors: whole-body COM over the hands' midpoint, root height and uprightness vs the reference, joint deviation,
control deviation, and a large penalty for any non-hand body (toes, knees, head, elbows) below 6 cm.
"""
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mujoco import rollout

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import plant_v2_mj as P
from bench import CROW, HAND, setup

DT = 1 / 240
SUB = 8                      # 30 Hz control
WATCH = ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "Head", "L_Elbow", "R_Elbow"]


def build_with_sensors():
    m0 = P.build(dt=DT)
    # rebuild via spec-free route: append sensors to an XML copy and re-apply the PD conversion
    tree = ET.parse(P.XML)
    root = tree.getroot()
    sens = ET.SubElement(root, "sensor")
    ET.SubElement(sens, "subtreecom", name="com", body="Pelvis")
    ET.SubElement(sens, "framezaxis", name="up", objtype="body", objname="Pelvis")
    for b in ["L_Wrist", "R_Wrist"] + WATCH:
        ET.SubElement(sens, "framepos", name=f"p_{b}", objtype="body", objname=b)
    tmp = __file__.rsplit("/", 1)[0] + "/_with_sensors.xml"
    tree.write(tmp)
    old = P.XML
    P.XML = tmp
    try:
        m = P.build(dt=DT)
    finally:
        P.XML = old
    return m


def plan_controls(nominal, noise, rng, N, H, K):
    """nominal [K, nu] knots -> samples [N, H*SUB, nu]."""
    knots = nominal[None] + noise * rng.standard_normal((N, K, nominal.shape[1]))
    knots[0] = nominal
    tk = np.linspace(0, H - 1, K)
    w = np.clip(1 - np.abs(np.arange(H)[:, None] - tk[None, :]) / (tk[1] - tk[0]), 0, 1)   # [H, K] hat weights
    ctrl = np.einsum("hk,nku->nhu", w, knots)
    return knots, np.repeat(ctrl, SUB, axis=1)


def cost(sd, st, qref, zref, upref, m, ctrl):
    # sensordata layout: com(3) up(3) p_L_Wrist(3) p_R_Wrist(3) then WATCH bodies (3 each)
    sd = sd[:, SUB - 1::SUB]          # control-step boundaries
    st = st[:, SUB - 1::SUB]
    com, up = sd[..., 0:3], sd[..., 3:6]
    hands = 0.5 * (sd[..., 6:9] + sd[..., 9:12])
    watch_z = sd[..., 14::3][..., :len(WATCH)]
    qj = st[..., 1 + 7: 1 + m.nq]
    rz = st[..., 1 + 2]
    c = 50.0 * ((com[..., :2] - hands[..., :2]) ** 2).sum(-1)
    c += 20.0 * (rz - zref) ** 2
    c += 5.0 * (1.0 - (up * upref).sum(-1))
    c += 0.05 * ((qj - qref) ** 2).sum(-1)
    c += 100.0 * (np.clip(0.06 - watch_z, 0, None) > 0).sum(-1)
    u = ctrl[:, SUB - 1::SUB]
    c += 0.01 * ((u - qref) ** 2).sum(-1)
    return c.sum(1)


def hold(pose, T=3.0, N=128, H=15, K=3, replan_every=3, noise=0.04, nthread=22, seed=0):
    m = build_with_sensors()
    d, qp, _, _ = setup(m, pose)
    qref, zref = qp[7:].copy(), qp[2]
    mujoco.mj_forward(m, d)
    upref = d.xmat[1].reshape(3, 3)[:, 2].copy()
    datas = [mujoco.MjData(m) for _ in range(nthread)]
    rng = np.random.default_rng(seed)
    nominal = np.tile(qref, (K, 1))
    steps = int(T * 30)
    t_plan, log = [], []
    k = 0
    while k < steps:
        s0 = P.full_state(m, d)
        t0 = time.perf_counter()
        knots, ctrl = plan_controls(nominal, noise, rng, N, H, K)
        st, sd = rollout.rollout(m, datas, np.tile(s0, (N, 1)), ctrl, nstep=H * SUB, persistent_pool=True)
        cst = cost(sd, st, qref, zref, upref, m, ctrl)
        best = int(np.argmin(cst))
        t_plan.append(time.perf_counter() - t0)
        # execute the first replan_every control steps of the best sample
        for j in range(replan_every * SUB):
            d.ctrl[:] = ctrl[best, j]
            mujoco.mj_step(m, d)
        k += replan_every
        # shift nominal: re-knot the best plan from t = replan_every
        tk = np.linspace(0, H - 1, K)
        best_ctrl = ctrl[best, ::SUB]                       # [H, nu]
        shifted = np.vstack([best_ctrl[replan_every:], np.repeat(best_ctrl[-1:], replan_every, 0)])
        nominal = shifted[np.round(tk).astype(int)]
        mujoco.mj_forward(m, d)
        up = d.xmat[1].reshape(3, 3)[:, 2]
        watch_min = min(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b), 2] for b in WATCH)
        log.append((k / 30, d.qpos[2], np.degrees(np.arccos(np.clip(up @ upref, -1, 1))), watch_min, cst[best]))
    rollout.shutdown_persistent_pool()
    arr = np.array(log)
    return arr, np.array(t_plan)


if __name__ == "__main__":
    for name, pose in (("crow", CROW), ("handstand", HAND)):
        for N, noise in ((128, 0.04), (256, 0.08)):
            arr, tp = hold(pose, N=N, noise=noise)
            fell = arr[:, 1] < arr[0, 1] - 0.10
            t_fall = arr[np.argmax(fell), 0] if fell.any() else None
            print(f"{name:9s} N={N} noise={noise}: replan {1e3*tp.mean():.0f} ms (p90 {1e3*np.percentile(tp,90):.0f}); "
                  f"root z {arr[0,1]:.3f}->{arr[-1,1]:.3f}, max tilt {arr[:,2].max():.1f} deg, "
                  f"min non-hand body z {arr[:,3].min():.3f} m, fell(-10cm) at {t_fall}", flush=True)
