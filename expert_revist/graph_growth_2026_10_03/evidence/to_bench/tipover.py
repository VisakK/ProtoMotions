"""Open-loop: hold a reference pose with constant PD targets; how long until it falls?"""
import sys

import mujoco
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import plant_v2_mj as P
from bench import CROW, STAND, HAND, CHAT, setup


def floor_bodies(m, d, floor_id):
    s = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == floor_id:
            s.add(m.geom_bodyid[c.geom2])
        elif c.geom2 == floor_id:
            s.add(m.geom_bodyid[c.geom1])
    return s


def run(pose, gain_scale=1.0, T=4.0, dt=1 / 240, settle=0.25):
    m = P.build(dt=dt, gain_scale=gain_scale)
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    d, qp, _, _ = setup(m, pose)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    z0 = d.qpos[2]
    up0 = d.xmat[1].reshape(3, 3)[:, 2].copy()
    n = int(T / dt)
    base = None
    t_fall = t_new = t_tilt = None
    new_body = None
    for i in range(n):
        mujoco.mj_step(m, d)
        t = (i + 1) * dt
        fb = floor_bodies(m, d, floor_id)
        if base is None and t >= settle:
            base = set(fb)
        if base is not None and t_new is None and (fb - base):
            t_new = t
            new_body = [names[b] for b in sorted(fb - base)]
        if t_fall is None and d.qpos[2] < z0 - 0.10:
            t_fall = t
        up = d.xmat[1].reshape(3, 3)[:, 2]
        if t_tilt is None and np.degrees(np.arccos(np.clip(up @ up0, -1, 1))) > 20:
            t_tilt = t
    return dict(z0=float(z0), z_end=float(d.qpos[2]), t_drop10cm=t_fall, t_tilt20=t_tilt, t_new_floor_contact=t_new,
                new_bodies=new_body, base=[names[b] for b in sorted(base or [])])


if __name__ == "__main__":
    for gs in (1.0, 10.0):
        for name, pose in (("crow", CROW), ("handstand", HAND), ("standing", STAND), ("chaturanga", CHAT)):
            r = run(pose, gs)
            print(f"gains x{gs:<4} {name:10s}: z0 {r['z0']:.3f} -> z(4 s) {r['z_end']:.3f}; root -10 cm at "
                  f"{r['t_drop10cm']}; root tilt 20 deg at {r['t_tilt20']}; first new floor contact at "
                  f"{r['t_new_floor_contact']} {r['new_bodies']}; base {r['base']}", flush=True)
