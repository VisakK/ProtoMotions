import sys
import time

import mujoco
import numpy as np
from mujoco import rollout

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import plant_v2_mj as P

CROW = ("220923_Crane_Crow_Pose_or_Bakasana_-a", 651)
STAND = ("220923_Crane_Crow_Pose_or_Bakasana_-a", 22)
HAND = ("220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a", 1193)
CHAT = ("220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a", 485)


def setup(m, pose, z_lift=0.0):
    pos, rot, _ = P.load_frame(*pose)
    qp, err, viol = P.qpos_from_bodies(m, pos, rot)
    d = mujoco.MjData(m)
    d.qpos[:] = qp
    d.qpos[2] += z_lift
    d.ctrl[:] = qp[7:]        # actuators are in joint order (motor i -> joint i+1)
    mujoco.mj_forward(m, d)
    return d, qp, err, viol


def lowest_geom_z(m, d):
    # crude: min over geom centres minus a size bound is not exact; use contact-free lowest body origin instead
    return float(d.xpos[1:25, 2].min())


def main():
    m = P.build(dt=1 / 120)
    # actuator order must equal joint order
    assert all(m.actuator_trnid[a, 0] == a + 1 for a in range(m.nu)), "actuator/joint order"
    print("nq nv nu ngeom", m.nq, m.nv, m.nu, m.ngeom, "mass", round(float(m.body_mass.sum()), 3))
    for name, pose in (("crow", CROW), ("stand", STAND), ("handstand", HAND), ("chaturanga", CHAT)):
        d, qp, err, viol = setup(m, pose)
        mujoco.mj_forward(m, d)
        print(f"{name}: FK err {err:.2e} m, worst hinge-range violation {viol:.1f} deg, ncon at t0 {d.ncon}, "
              f"lowest body origin {lowest_geom_z(m, d):.3f}")

    # --- single-thread mj_step throughput, crow pose with contacts ---
    for dt in (1 / 120, 1 / 240, 1 / 480):
        m2 = P.build(dt=dt)
        d, qp, _, _ = setup(m2, CROW)
        n = int(2.0 / dt)
        t0 = time.perf_counter()
        ncon = []
        for i in range(n):
            mujoco.mj_step(m2, d)
            ncon.append(d.ncon)
        el = time.perf_counter() - t0
        bad = not np.isfinite(d.qpos).all()
        print(f"dt=1/{round(1/dt)}: {n/el:,.0f} steps/s single thread, mean ncon {np.mean(ncon):.1f}, "
              f"root z after 2 s {d.qpos[2]:.3f} (start {qp[2]:.3f}), nan={bad}")

    # --- mujoco.rollout throughput ---
    m = P.build(dt=1 / 120)
    d, qp, _, _ = setup(m, CROW)
    s0 = P.full_state(m, d)
    rng = np.random.default_rng(0)
    for nthread in (1, 8, 16, 22):
        datas = [mujoco.MjData(m) for _ in range(nthread)]
        for H in (15, 30, 60):        # control steps at 30 Hz
            nstep = H * 4             # 4 physics substeps of 1/120 s
            nbatch = 256
            ctrl = qp[7:][None, None, :] + 0.05 * rng.standard_normal((nbatch, H, m.nu))
            ctrl = np.repeat(ctrl, 4, axis=1)
            init = np.tile(s0, (nbatch, 1))
            rollout.rollout(m, datas, init, ctrl, nstep=nstep, persistent_pool=True)   # warm
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                st, _ = rollout.rollout(m, datas, init, ctrl, nstep=nstep, persistent_pool=True)
            el = (time.perf_counter() - t0) / reps
            sps = nbatch * nstep / el
            print(f"threads {nthread:2d} H={H:2d} ctrl steps (nstep {nstep}), nbatch {nbatch}: {el*1e3:7.1f} ms per "
                  f"batch -> {sps:,.0f} physics steps/s = {sps/4:,.0f} control steps/s")
        rollout.shutdown_persistent_pool()


if __name__ == "__main__":
    main()
