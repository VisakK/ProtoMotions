# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statue witness (BUILD_PLAN Step 7): does the plant, simulated, actually hold a reference pose?

``run(pos, rot, tau, pairs)`` puts the plant in the pose at rest, sets every PD target to the pose
plus ``Kp^-1 tau`` (``tau``: the gated LP's representative joint torques, ``statics.analyse``),
clipped to the training action range, and simulates ``DURATION_S`` (3 s) in MuJoCo with gravity on
and no root assistance. It passes when the statue holds the pose on the contacts it started with:

* it **settles** by less than ``SETTLE_M`` (5 cm) in the first ``SETTLE_S`` (0.5 s): the supports the
  reference leaves up to 2 cm above the floor, and the tilted boxes they rest on, come down;
* it then **drifts** by less than ``DRIFT_M`` (2 cm) over the remaining 2.5 s (every body, from its
  settled position);
* its **contacts are unchanged**, judged with the package's three-way truth
  (``verdicts.packet_truth``): every zone touching the floor in the pose (<= 1 cm) is still on it at
  the end (mean force > 1 N over the last 0.1 s), no zone lands that was beyond the band (2 cm), and
  every realised pair the LP was given (``pairs``) still touches. Pairs nobody asked for (feet
  resting together, pushed apart by their overlap in the pose) are reported, not judged.

The measured case for the settle: the statues that hold (Warrior II -a, the Uttanasana standing
holds) settle 2.4-3.1 cm and then move 0.2-0.8 cm; the ones that fail keep going (tens of cm to a
fall) or collapse at once (Cockerel -b, 49 cm). A pass is strong positive evidence that the plant can
hold the pose on those contacts. A failure is weaker evidence, because a statue has no balance
feedback (README §4.3): narrow supports and tilted boxes fail for it first (the shipped standing
references rest on the outer edges of their foot boxes, supinated about 12 deg).

The plant
---------
The LP's MuJoCo model (``static_hold_lp.FLAT``, 74 kg) with a floor plane at z = 0, and:

* friction 0.75 against the floor and 0.5 between bodies, ``condim`` 3 everywhere: PhysX's effective
  values (the robot's shapes keep PhysX's default 0.5, the terrain's 1.0 is averaged with it);
* self-collision on, parent-child pairs filtered (MuJoCo's default, and PhysX's for an articulation);
* passive joint stiffness and damping zeroed; the actuators are position servos, as the ProtoMotions
  MuJoCo backend makes them, with the training gains (``SmplYogiRobotConfig``, resolved by
  ``pose_lib.extract_control_info``) and the MJCF torque limits (hip 300 ... wrist 20, hand 10 N m);
* 1 kHz, ``implicitfast`` (the servos' damping is integrated implicitly).

Two deliberate departures from the training plant, both measured:

* **The servos are ``STIFFNESS_SCALE`` (10x) stiffer, damping x sqrt(10), torque limits unchanged.**
  At the training gains a statue buckles whatever the pose: ankle, knee and hip in series give each
  leg about 270 N m/rad against a gravitational stiffness m g h of about 600 N m/rad. Warrior II -a
  falls at 1x-3x and stands from 5x; a welded (rigid) statue stands. The policy balances with
  feedback; the statue has none, so it gets stiffness instead. The limits keep the torque test real.
* **Joint limits are off.** 290 of 303 exemplars violate some hinge range of the MuJoCo XYZ
  decomposition by > 2 deg (elbow twist most often, lotus knees at 175 deg), and the training plant's
  limits are soft (the USD's ``physxLimit:rot*:stiffness`` is 300-500). Hard hinge limits would test
  the decomposition, not the pose; ``statics.limit_violations`` reports the violations instead.

Contacts in the pose come from MuJoCo's own collision detection with a probe margin: a zone is on the
floor within ``statics.GROUND_BAND_M`` and a pair touches within ``statics.PAIR_BAND_M``, the LP's
bands, so the witness and the LP judge the same geometry. Pairs are the labels' 91 non-adjacent zone
pairs (``human_mesh.PAIRS``); contacts inside a zone or between adjacent zones are ignored.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

import static_hold_lp as S
from extract_contact_configs import ZONES
from reference_curation import human_mesh as hm, ids, statics, verdicts

MODULE = "reference_curation.witness"
DURATION_S = 3.0
DT = 0.001
STIFFNESS_SCALE = 10.0
DRIFT_M = 0.02                  # after the settle
SETTLE_S = 0.5
SETTLE_M = 0.05
TOUCH_M = verdicts.AVATAR_TOUCH_CM / 100.0   # 0.01: touching in the pose
CONTACT_N = 1.0                 # mean normal force that makes a contact at the end
TAIL_S = 0.1                    # the end: the last 0.1 s
MU_GROUND, MU_BODY = S.MU_GROUND, S.MU_BODY
CONFIG = {"duration_s": DURATION_S, "dt": DT, "integrator": "implicitfast", "stiffness_scale": STIFFNESS_SCALE,
          "damping_scale": math.sqrt(STIFFNESS_SCALE), "drift_m": DRIFT_M, "settle_s": SETTLE_S, "settle_m": SETTLE_M,
          "touch_m": TOUCH_M, "contact_n": CONTACT_N, "tail_s": TAIL_S,
          "mu_ground": MU_GROUND, "mu_body": MU_BODY, "condim": 3, "joint_limits": False, "self_collision": True,
          "action_range": "build_pd_action_offset_scale(action_scale=1)", "gains": "SmplYogiRobotConfig"}

BODY_ZONE = {b: z for z, bodies in ZONES.items() for b in bodies}
PAIR_NAME = {frozenset(p): "+".join(p) for p in hm.PAIRS}


def control_info() -> dict:
    """``{joint: ControlInfo}`` as training resolves it: the MJCF plus the robot config's overrides."""
    from protomotions.components.pose_lib import extract_control_info
    from protomotions.robot_configs.smpl_yogi import SmplYogiRobotConfig

    overrides = SmplYogiRobotConfig.__dataclass_fields__["control"].default_factory().override_control_info
    with contextlib.redirect_stdout(io.StringIO()):   # it prints one line per joint
        return extract_control_info(str(ids.MJCF), overrides)


def action_range() -> tuple[np.ndarray, np.ndarray]:
    """``(low, high)`` [69] of the training PD targets (``make_pd_action_config``, action_scale 1)."""
    from protomotions.envs.action.action_functions import build_pd_action_offset_scale

    hinges = S.M.jnt_range[1:]
    offset, scale = build_pd_action_offset_scale({b: [0, 1, 2] for b in range(len(hinges) // 3)},
                                                 torch.as_tensor(hinges[:, 0]), torch.as_tensor(hinges[:, 1]),
                                                 1.0, torch.device("cpu"))
    return (offset - scale).numpy(), (offset + scale).numpy()


@functools.lru_cache(maxsize=2)
def plant(stiffness_scale: float = STIFFNESS_SCALE) -> SimpleNamespace:
    """The witness plant (see the module docstring), plus a copy with a probe margin for the pose's
    contacts, and the per-actuator gains, limits, dof addresses and target range."""
    spec = mujoco.MjSpec.from_file(str(S.FLAT))
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    floor = spec.worldbody.add_geom()
    floor.name, floor.type, floor.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [0.0, 0.0, 0.05]
    floor.friction, floor.condim = [MU_GROUND, 0.005, 0.0001], 3
    floor.solref, floor.solimp = [0.015, 1.0], [0.9, 0.99, 0.003, 0.5, 2.0]   # the robot geoms' values
    m = spec.compile()
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot = np.arange(m.ngeom) != floor_id
    m.geom_friction[robot, 0] = MU_BODY          # MuJoCo takes the max: 0.75 on the floor, 0.5 between bodies
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
    lo, hi = action_range()
    zone_of = [BODY_ZONE.get(m.body(m.geom_bodyid[g]).name) if g != floor_id else None for g in range(m.ngeom)]
    return SimpleNamespace(model=m, probe=probe, floor=floor_id, kp=kp, kd=kd, limit=lim, dof=dof,
                           target_low=lo[dof], target_high=hi[dof], zone_of=zone_of, stiffness_scale=stiffness_scale)


def _contact_key(p: SimpleNamespace, g1: int, g2: int) -> str | None:
    """``ZONE:G`` for a floor contact, ``A+B`` for a labelled zone pair, ``None`` otherwise."""
    if p.floor in (g1, g2):
        return f"{p.zone_of[g2 if g1 == p.floor else g1]}:G"
    return PAIR_NAME.get(frozenset((p.zone_of[g1], p.zone_of[g2])))


def pose_contacts(qpos: np.ndarray, p: SimpleNamespace) -> dict:
    """``{contact: distance m}`` of the pose within the LP's bands, from MuJoCo's collision detection."""
    d = mujoco.MjData(p.probe)
    d.qpos[:] = qpos
    mujoco.mj_forward(p.probe, d)
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        key = _contact_key(p, c.geom1, c.geom2)
        band = statics.GROUND_BAND_M if key and key.endswith(":G") else statics.PAIR_BAND_M
        if key and c.dist <= band:
            out[key] = min(out.get(key, math.inf), float(c.dist))
    return out


def run(pos: np.ndarray, rot: np.ndarray, tau: np.ndarray | None, pairs=(), duration: float = DURATION_S,
        stiffness_scale: float = STIFFNESS_SCALE) -> dict:
    """The statue witness of one pose (module docstring). ``tau`` [69] in hinge (dof) order, or ``None``
    for no feed-forward; ``pairs`` the body-body contacts the LP balanced the pose on."""
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
    start = pose_contacts(qpos, p)
    x0 = d.xpos[1:].copy()
    n, tail, k_settle = int(round(duration / DT)), int(round(TAIL_S / DT)), int(round(SETTLE_S / DT))
    x_settle, force, f6 = None, {}, np.zeros(6)
    for k in range(n):
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return {"passed": False, "error": "non-finite state", "step": k}
        if k + 1 == k_settle:
            x_settle = d.xpos[1:].copy()
        if k >= n - tail:
            for i in range(d.ncon):
                c = d.contact[i]
                key = _contact_key(p, c.geom1, c.geom2)
                if key:
                    mujoco.mj_contactForce(m, d, i, f6)
                    force[key] = force.get(key, 0.0) + f6[0] / tail
    x_end = d.xpos[1:]
    settle = float(np.linalg.norm(x_settle - x0, axis=-1).max())
    moved = np.linalg.norm(x_end - x_settle, axis=-1)
    end = {k: round(v, 1) for k, v in force.items() if v > CONTACT_N}
    ground_start = {k: v for k, v in start.items() if k.endswith(":G")}
    must = {k for k, v in ground_start.items() if v <= TOUCH_M}
    lifted = sorted(must - set(end))
    landed = sorted(k for k in end if k.endswith(":G") and k not in ground_start)
    opened = sorted(k for k in pairs if k in start and k not in end)
    pair_changes = sorted(set(k for k in start if not k.endswith(":G")) ^ set(k for k in end if not k.endswith(":G"))
                          - set(pairs))
    saturated = [m.actuator(a).name for a in range(m.nu) if abs(d.actuator_force[a]) >= p.limit[a] - 1e-6]
    return {"passed": bool(settle < SETTLE_M and moved.max() < DRIFT_M and not (lifted or landed or opened)),
            "settle_cm": round(100 * settle, 2), "drift_cm": round(100 * float(moved.max()), 2),
            "drift_body": S.BODY[int(moved.argmax())],
            "final_cm": round(100 * float(np.linalg.norm(x_end - x0, axis=-1).max()), 2),
            "start_contacts": {k: round(100 * v, 2) for k, v in sorted(start.items())}, "end_contacts": end,
            "lifted": lifted, "landed": landed, "pairs_opened": opened, "other_pair_changes": pair_changes,
            "saturated": saturated, "targets_clipped": clipped, "feed_forward": tau is not None,
            "stiffness_scale": stiffness_scale}
