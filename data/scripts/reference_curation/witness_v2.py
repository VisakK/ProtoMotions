# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The statue witness on plant v2 (BodyFix Step 4, item 4): Step 7's MuJoCo statue (``witness.py``) with two changes.

* **Plant v2's training gains.** ``witness.control_info`` resolves the gains from ``SmplYogiRobotConfig``. Plant v2
  trains with ``SmplYogiV2RobotConfig``, identical except the neck, whose joint-space inertia rose x2.32 when the
  head sphere moved onto the cranium (1158 / 116 instead of 500 / 50). The statue takes those, at the same 10x
  stiffening (``witness.STIFFNESS_SCALE``), with plant v2's torque limits (wrist 30, hand 15 N m).
* **TODO D4: only loaded zones must stay down.** Step 7 failed a statue whenever a zone touching the floor in the
  pose (<= 1 cm) had lifted by the end. On the pilot, 10 of the 14 statues that held still (drift <= 0.04 cm) failed
  that way: static supine holds in which an *unloaded* resting zone (a forearm beside the body in Plow, the back of
  a head in Bridge) rose past 1 cm. The rule is now the gated LP's: a zone must stay down if it touches in the pose
  **and** the LP's representative solution loads it (more than ``statics.LOAD_EPS_N``). Every other test is
  Step 7's: the settle (< 5 cm in 0.5 s), the drift (< 2 cm after it), no zone landing from beyond 2 cm, every
  realised pair the LP was given still closed. ``passed_strict`` keeps Step 7's verdict beside it.

Everything else is ``witness.py``'s, and its docstring is the reference: the plant (``static_hold_lp``'s model with a
floor; friction 0.75 / 0.5; self-collision on; joint limits off), the PD targets (pose + Kp^-1 tau_LP, clipped to the
training action range), 3 s at 1 kHz. It must run inside ``retarget_v2.on_plant()`` (``static_hold_lp`` posed on
plant v2), and refuses otherwise.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import math

import mujoco
import numpy as np

import static_hold_lp as S
from reference_curation import fit_writer as fw, ids, statics
from reference_curation import witness as W

REPO = ids.REPO
MODULE = "reference_curation.witness_v2"
PLANT = "v2"
CONFIG = {**W.CONFIG, "gains": "SmplYogiV2RobotConfig", "plant": PLANT,
          "contact_rule": "a zone touching the floor in the pose (<= touch_m) that the gated LP loads (> load_eps_n) "
                          "must stay down (TODO D4); Step 7's rule (every touching zone) is reported as passed_strict",
          "load_eps_n": statics.LOAD_EPS_N}


def _require_plant() -> None:
    flat = fw.plant_paths(PLANT)[1]
    if S.FLAT.resolve() != flat.resolve():
        raise RuntimeError(f"static_hold_lp is posed on {S.FLAT}, not plant {PLANT}: run inside retarget_v2.on_plant()")


def control_info() -> dict:
    """``{joint: ControlInfo}`` as training on ``smpl_yogi_v2`` resolves it."""
    from protomotions.components.pose_lib import extract_control_info
    from protomotions.robot_configs.smpl_yogi_v2 import SmplYogiV2RobotConfig

    overrides = SmplYogiV2RobotConfig.__dataclass_fields__["control"].default_factory().override_control_info
    with contextlib.redirect_stdout(io.StringIO()):
        return extract_control_info(str(fw.plant_paths(PLANT)[0]), overrides)


@functools.lru_cache(maxsize=2)
def plant(stiffness_scale: float = W.STIFFNESS_SCALE):
    """``witness.plant`` on plant v2 with ``control_info()``'s gains (the body is ``witness.plant``'s, line for
    line, with the gains swapped)."""
    _require_plant()
    spec = mujoco.MjSpec.from_file(str(S.FLAT))
    spec.option.timestep = W.DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    floor = spec.worldbody.add_geom()
    floor.name, floor.type, floor.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [0.0, 0.0, 0.05]
    floor.friction, floor.condim = [W.MU_GROUND, 0.005, 0.0001], 3
    floor.solref, floor.solimp = [0.015, 1.0], [0.9, 0.99, 0.003, 0.5, 2.0]
    m = spec.compile()
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot = np.arange(m.ngeom) != floor_id
    m.geom_friction[robot, 0] = W.MU_BODY
    m.geom_condim[:] = 3
    m.jnt_stiffness[:] = 0.0
    m.dof_damping[:] = 0.0
    m.jnt_limited[:] = 0
    info = control_info()
    kp, kd, lim = np.zeros(m.nu), np.zeros(m.nu), np.zeros(m.nu)
    dof = np.zeros(m.nu, int)
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        ci = info[m.joint(j).name]
        kp[a], kd[a], lim[a] = stiffness_scale * ci.stiffness, math.sqrt(stiffness_scale) * ci.damping, ci.effort_limit
        dof[a] = m.jnt_dofadr[j] - 6
        m.actuator_gear[a] = [1, 0, 0, 0, 0, 0]
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a] = 0.0
        m.actuator_gainprm[a, 0] = kp[a]
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[a] = 0.0
        m.actuator_biasprm[a, 1:3] = [-kp[a], -kd[a]]
        m.actuator_ctrllimited[a] = 0
        m.actuator_forcelimited[a] = 1
        m.actuator_forcerange[a] = [-lim[a], lim[a]]
    if not np.allclose(lim, S.TAU_MAX[dof]):
        raise ValueError("the actuator limits differ from the LP's")
    probe = copy.deepcopy(m)
    probe.geom_margin[:] = statics.GROUND_BAND_M
    lo, hi = W.action_range()
    zone_of = [W.BODY_ZONE.get(m.body(m.geom_bodyid[g]).name) if g != floor_id else None for g in range(m.ngeom)]
    from types import SimpleNamespace
    return SimpleNamespace(model=m, probe=probe, floor=floor_id, kp=kp, kd=kd, limit=lim, dof=dof,
                           target_low=lo[dof], target_high=hi[dof], zone_of=zone_of, stiffness_scale=stiffness_scale)


def run(pos: np.ndarray, rot: np.ndarray, tau: np.ndarray | None, pairs=(), loaded=None,
        duration: float = W.DURATION_S, stiffness_scale: float = W.STIFFNESS_SCALE) -> dict:
    """The statue of one pose (``witness.run``'s test, line for line, on plant v2's gains) with TODO D4's contact
    rule: ``loaded`` names the ``ZONE:G`` contacts the gated LP loads; only those must stay down (``None``: every
    touching zone, Step 7's rule)."""
    _require_plant()          # the cached plant outlives a plant context; static_hold_lp must still be on v2
    p = plant(stiffness_scale)
    m = p.model
    S.set_pose(np.asarray(pos, float), np.asarray(rot, float))
    qpos = S.D.qpos.copy()
    q = qpos[7:]
    ff = np.zeros(len(p.dof)) if tau is None else np.asarray(tau, float)[p.dof] / p.kp
    target = q[p.dof] + ff
    clipped = int(((target < p.target_low) | (target > p.target_high)).sum())
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = np.clip(target, p.target_low, p.target_high)
    mujoco.mj_forward(m, d)
    start = W.pose_contacts(qpos, p)
    x0 = d.xpos[1:].copy()
    n, tail, k_settle = int(round(duration / W.DT)), int(round(W.TAIL_S / W.DT)), int(round(W.SETTLE_S / W.DT))
    x_settle, force, f6 = None, {}, np.zeros(6)
    for k in range(n):
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return {"passed": False, "passed_strict": False, "error": "non-finite state", "step": k}
        if k + 1 == k_settle:
            x_settle = d.xpos[1:].copy()
        if k >= n - tail:
            for i in range(d.ncon):
                c = d.contact[i]
                key = W._contact_key(p, c.geom1, c.geom2)
                if key:
                    mujoco.mj_contactForce(m, d, i, f6)
                    force[key] = force.get(key, 0.0) + f6[0] / tail
    x_end = d.xpos[1:]
    settle = float(np.linalg.norm(x_settle - x0, axis=-1).max())
    moved = np.linalg.norm(x_end - x_settle, axis=-1)
    end = {k: round(v, 1) for k, v in force.items() if v > W.CONTACT_N}
    ground_start = {k: v for k, v in start.items() if k.endswith(":G")}
    touching = {k for k, v in ground_start.items() if v <= W.TOUCH_M}
    must = touching if loaded is None else touching & set(loaded)
    lifted = sorted(must - set(end))
    lifted_unloaded = sorted((touching - must) - set(end))
    landed = sorted(k for k in end if k.endswith(":G") and k not in ground_start)
    opened = sorted(k for k in pairs if k in start and k not in end)
    pair_changes = sorted(set(k for k in start if not k.endswith(":G")) ^ set(k for k in end if not k.endswith(":G"))
                          - set(pairs))
    saturated = [m.actuator(a).name for a in range(m.nu) if abs(d.actuator_force[a]) >= p.limit[a] - 1e-6]
    still = bool(settle < W.SETTLE_M and moved.max() < W.DRIFT_M)
    return {"passed": bool(still and not (lifted or landed or opened)),
            "passed_strict": bool(still and not (lifted or lifted_unloaded or landed or opened)),
            "still": still, "settle_cm": round(100 * settle, 2), "drift_cm": round(100 * float(moved.max()), 2),
            "drift_body": S.BODY[int(moved.argmax())],
            "final_cm": round(100 * float(np.linalg.norm(x_end - x0, axis=-1).max()), 2),
            "start_contacts": {k: round(100 * v, 2) for k, v in sorted(start.items())}, "end_contacts": end,
            "must_stay": sorted(must), "lifted": lifted, "lifted_unloaded": lifted_unloaded, "landed": landed,
            "pairs_opened": opened, "other_pair_changes": pair_changes, "saturated": saturated,
            "targets_clipped": clipped, "feed_forward": tau is not None, "stiffness_scale": stiffness_scale,
            "gains": "SmplYogiV2RobotConfig"}
