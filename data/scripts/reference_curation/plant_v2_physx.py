# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plant v2 in PhysX (BodyFix Step 2, the simulated half): IsaacLab simulates the MJCF, its FK is the
references' FK, and a standing reset does not launch.

The robot is built exactly as training builds it (robot config -> ``build_all_components`` -> IsaacLab,
headless), with no policy, and three things are measured:

1. **Mass and actuation, read from the PhysX view at full precision** (``check_simulated_mass.py`` prints
   grams, which alone puts the 0.11 kg hands 0.2 % off). For every body: mass, body-frame COM and inertia
   tensor against MuJoCo's (the same primitive at the same uniform density). For every dof: limits against
   the MJCF range (PhysX's hard box), max force against the MJCF ``actuatorfrcrange``, and stiffness, damping,
   armature and velocity limit against the robot config's resolved ``control_info``.
2. **PhysX FK equals the references' FK.** Step 3's writer stores ``rigid_body_pos`` from pose_lib FK:
   ``extract_transforms_from_qpos(..., qpos_is_exp_map_on_3dof_joints=True)`` then
   ``fk_from_transforms_with_velocities``, in float32 (``convert_yoga_frames_to_proto.py``), applied to the
   female fit (``mosh_replay.mosh_kinematics(hand="fingers")``) inside the plant's joint box. Here the same
   root and dof go through training's reset (``Simulator.reset_envs``), and PhysX's link poses
   (``update_articulations_kinematic``, with no step) must equal the stored bodies to 1e-5 m. The frames are
   every labels-v1.1 hold with a fit, plus two transition frames per clip.
3. **Reset stability**, through ``reset_envs`` and then ``Simulator.step`` for ``--duration`` s, replicated
   over every env (PhysX is not deterministic):
   * ``default``: the rest pose at ``default_root_height``, zero action. For 3-dof joints the action offset
     is 0 (``build_pd_action_offset_scale``), so a zero action holds the rest pose. This is the card's test.
   * ``standing``: her two-feet standing holds (required supports exactly L_FOOT + R_FOOT), with the lowest
     collider at 0 plus the training scripts' 5 mm ``ref_respawn_offset``, and the PD targets held at the
     pose (what the env does on reset).
   * ``thrown_1ms``: the rest pose with the root thrown up at 1 m/s (training's ``max_depenetration_velocity``).
     This is the positive control: the detector must call it a launch on every replica.
   * ``sunk_1cm`` / ``sunk_2cm``: the rest pose 1 / 2 cm into the floor. The ankle box (2.5 cm half-height)
     still straddles the surface. Measured: PhysX corrects the position to the surface (the root rises by the
     sink, at <= 0.05 m/s) and nothing is thrown. Gated as a fact about the configuration: restored (a rise of
     >= 80 % of the sink) and not launched.
   * ``sunk_5cm``: how a v1 reference starts on v2 (5-6.6 cm into the floor). The foot boxes start entirely
     under the one-sided terrain mesh and nothing pushes them out: the body sinks through the floor
     (recorded, not a gate).
   A **launch** is the root overshooting the height that just clears the floor (its reset height, plus the
   sink for a pose that starts inside it) by more than 1 cm, or moving up faster than 0.25 m/s, within the
   first 0.5 s.

Writes ``data/reference_curation/plant_v2/physx_<robot>.json`` and exits non-zero on any failure. Run it
as its own process (IsaacLab launches before torch)::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python data/scripts/reference_curation/plant_v2_physx.py \\
        [--robot smpl_yogi_v2]

``--robot smpl_yogi`` measures the shipped plant the same way (its FK frames use v1's skeleton).
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--robot", default="smpl_yogi_v2")
parser.add_argument("--out", default=None, help="default: data/reference_curation/plant_v2/physx_<robot>.json")
parser.add_argument("--duration", type=float, default=1.0, help="reset test length (s)")
parser.add_argument("--transition-frames", type=int, default=2, help="non-hold FK frames per clip")
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.WARNING)

SCHEMA_VERSION = 1
MODULE = "reference_curation.plant_v2_physx"
MASS_RTOL = 1e-3            # the card: every body within 0.1 % of the MJCF
TOTAL_MASS_ATOL = 5e-3      # "74.00 kg"
COM_TOL_M = 1e-6
INERTIA_RTOL = 1e-3
LIMIT_TOL_RAD = 1e-5
GAIN_RTOL = 1e-5
FK_TOL_M = 1e-5             # the card
RESPAWN_M = 0.005           # env.ref_respawn_offset in every training script (the config default is 0.05)
SINKS_M = (0.01, 0.02, 0.05)  # rest pose this far into the floor (the ankle box is 2.5 cm half-height, sole at +0.5 mm)
SHALLOW_SINKS_M = (0.01, 0.02)  # the box straddles the surface: restored to it, without a launch
RESTORED_FRAC = 0.8             # "restored": the root rises by at least this fraction of the sink
THROW_VZ = 1.0                  # the positive control's upward root speed (= max_depenetration_velocity)
WINDOW_S = 0.5
LAUNCH_RISE_M = 0.01
LAUNCH_VZ = 0.25
GRID_M = 2.5


def _mj_reference(flat: Path):
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(flat))
    bodies = {}
    for i in range(1, m.nbody):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_iquat[i])
        R = R.reshape(3, 3)
        bodies[m.body(i).name] = {"mass": float(m.body_mass[i]), "com": m.body_ipos[i].copy(),
                                  "inertia": R @ np.diag(m.body_inertia[i]) @ R.T}
    joints = {m.joint(j).name: {"range": m.jnt_range[j].copy(), "frc": float(m.jnt_actfrcrange[j][1]),
                                "armature": float(m.dof_armature[m.jnt_dofadr[j]])}
              for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    return bodies, joints


def physx_properties(sim, robot_config, flat: Path) -> dict:
    """Masses, COMs, inertias, dof limits and drive parameters from ``root_physx_view``, against the MJCF and
    the robot config (env 0; every env is a clone)."""
    view = sim._robot.root_physx_view
    body_names = list(sim._robot.data.body_names)
    dof_names = list(sim._robot.data.joint_names)
    mj_bodies, mj_joints = _mj_reference(flat)
    masses = view.get_masses()[0].double().cpu().numpy()
    coms = view.get_coms()[0].double().cpu().numpy()                  # pos + quat, link frame
    inert = view.get_inertias()[0].double().cpu().numpy().reshape(-1, 3, 3)
    rows = {}
    for i, b in enumerate(body_names):
        ref = mj_bodies[b]
        rows[b] = {"physx_kg": masses[i], "mjcf_kg": ref["mass"],
                   "mass_rel_err": abs(masses[i] - ref["mass"]) / ref["mass"],
                   "com_err_m": float(np.linalg.norm(coms[i, :3] - ref["com"])),
                   "inertia_rel_err": float(np.linalg.norm(inert[i] - ref["inertia"]) / np.linalg.norm(ref["inertia"]))}
    lim = view.get_dof_limits()[0].double().cpu().numpy()
    got = {"max_force": view.get_dof_max_forces()[0], "stiffness": view.get_dof_stiffnesses()[0],
           "damping": view.get_dof_dampings()[0], "armature": view.get_dof_armatures()[0],
           "max_velocity": view.get_dof_max_velocities()[0]}
    got = {k: v.double().cpu().numpy() for k, v in got.items()}
    ci = robot_config.control.control_info
    dofs, err = {}, {"limit_rad": 0.0, "max_force_vs_mjcf": 0.0}
    want_of = {"stiffness": "stiffness", "damping": "damping", "armature": "armature",
               "max_velocity": "velocity_limit", "max_force": "effort_limit"}
    for k in want_of:
        err[f"{k}_vs_config_rel"] = 0.0
    for d, name in enumerate(dof_names):
        ref = mj_joints[name]
        e_lim = float(np.abs(lim[d] - ref["range"]).max())
        e_frc = abs(got["max_force"][d] - ref["frc"])
        err["limit_rad"] = max(err["limit_rad"], e_lim)
        err["max_force_vs_mjcf"] = max(err["max_force_vs_mjcf"], e_frc)
        row = {"limit_rad": lim[d].tolist(), "mjcf_range_rad": ref["range"].tolist(),
               **{k: float(got[k][d]) for k in got}}
        for k, attr in want_of.items():
            want = getattr(ci[name], attr)
            if want is not None:
                rel = abs(got[k][d] - want) / max(abs(want), 1e-9)
                err[f"{k}_vs_config_rel"] = max(err[f"{k}_vs_config_rel"], float(rel))
                row[f"config_{attr}"] = float(want)
        dofs[name] = row
    return {"bodies": rows, "dofs": dofs, "total_physx_kg": float(masses.sum()),
            "total_mjcf_kg": float(sum(v["mass"] for v in mj_bodies.values())),
            "mass_max_rel_err": max(r["mass_rel_err"] for r in rows.values()),
            "mass_worst_body": max(rows, key=lambda b: rows[b]["mass_rel_err"]),
            "com_max_err_m": max(r["com_err_m"] for r in rows.values()),
            "inertia_max_rel_err": max(r["inertia_rel_err"] for r in rows.values()),
            "dof_max_err": err, "effort_by_joint": {n[:-2]: float(got["max_force"][d]) for d, n in enumerate(dof_names)},
            "neck_gains": {n: [float(got["stiffness"][d]), float(got["damping"][d])]
                           for d, n in enumerate(dof_names) if n.startswith("Neck_")}}


# --------------------------------------------------------------------------- #
# The frames: her fit, inside the plant's joint box, through pose_lib FK
# --------------------------------------------------------------------------- #
def reference_frames(plant_xml: Path, n_transition: int) -> dict:
    """Every labels-v1.1 hold with a fit plus ``n_transition`` evenly spaced frames per clip, as Step 3's
    writer would store them: female fit (fingers) -> nearest exp-map representative -> clipped into the box."""
    from reference_curation import mosh_replay as mr
    from reference_curation import plant_v2 as P

    sk = mr.skeleton_for(plant_xml)
    lo, hi = sk.lower.numpy(), sk.upper.numpy()
    labels = P._labels()[1]
    rows, standing = [], []
    root_pos, root_quat, dof = [], [], []
    clipped, clip_deg = 0, 0.0
    for stem, holds in labels["clips"]:
        try:
            T = mr._load(stem)[0]["fullpose"].shape[0]
        except FileNotFoundError:
            continue
        frames = [int(h["frame_hold"]) for h in holds]
        frames += [int(round((k + 1) * (T - 1) / (n_transition + 1))) for k in range(n_transition)]
        kin = mr.mosh_kinematics(stem, hand="fingers", representative="nearest", xml_path=plant_xml,
                                 frames=np.array(frames))
        d = kin["dof"]
        over = np.maximum(d - hi, lo - d)
        clipped += int((over > 0).sum())
        clip_deg = max(clip_deg, float(np.degrees(over.max())))
        d = np.clip(d, lo, hi)
        for k, f in enumerate(frames):
            hold = holds[k] if k < len(holds) else None
            if hold is not None:
                anns = labels["anns"][hold["hold_id"]]
                ground = {z for a in anns if a["kind"] == "ground" and a.get("target_role") == "required_support"
                          for z in a["zones"]}
                if ground == {"L_FOOT", "R_FOOT"}:
                    standing.append(len(rows))
            rows.append({"stem": stem, "frame": f, "hold_id": hold["hold_id"] if hold else None})
        root_pos.append(kin["root_pos"])
        root_quat.append(kin["root_quat"])
        dof.append(d)
    return {"rows": rows, "standing": standing, "root_pos": np.concatenate(root_pos),
            "root_quat_xyzw": np.concatenate(root_quat), "dof": np.concatenate(dof),
            "clipped_coords": clipped, "clip_max_deg": clip_deg}


def poselib_fk(kinematic_info, root_pos, root_quat_xyzw, dof, dtype):
    """The writer's FK: qpos = [root, root quat wxyz, exp-map dof] -> pose_lib (COMMON order, rot xyzw)."""
    from protomotions.components.pose_lib import extract_transforms_from_qpos, fk_from_transforms_with_velocities

    q = torch.as_tensor(root_quat_xyzw, dtype=torch.float64)
    qpos = torch.cat([torch.as_tensor(root_pos, dtype=torch.float64), q[:, [3, 0, 1, 2]],
                      torch.as_tensor(dof, dtype=torch.float64)], 1).to(dtype)
    rp, jrm = extract_transforms_from_qpos(kinematic_info, qpos, qpos_is_exp_map_on_3dof_joints=True)
    st = fk_from_transforms_with_velocities(kinematic_info, rp, jrm, fps=None, compute_velocities=False)
    return st.rigid_body_pos, st.rigid_body_rot, qpos


def _quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between unit quaternions (xyzw): 2 atan2(|v|, |w|) of conj(a) b, accurate at small angles
    (``2 acos(|a.b|)`` reads float32 rounding as ~0.05 deg)."""
    a, b = a.double(), b.double()
    w = (a * b).sum(-1)
    v = a[..., 3:4] * b[..., :3] - b[..., 3:4] * a[..., :3] - torch.cross(a[..., :3], b[..., :3], dim=-1)
    return 2 * torch.atan2(v.norm(dim=-1), w.abs())


def _reset(sim, root_pos, root_rot_xyzw, dof_pos, root_vel=None):
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion

    n = root_pos.shape[0]
    dev = sim.device
    z3 = torch.zeros(n, 3, device=dev)
    root_vel = z3.clone() if root_vel is None else root_vel.float().to(dev)
    state = ResetState(root_pos=root_pos.float().to(dev), root_rot=root_rot_xyzw.float().to(dev),
                       root_vel=root_vel, root_ang_vel=z3.clone(), dof_pos=dof_pos.float().to(dev),
                       dof_vel=torch.zeros_like(dof_pos, dtype=torch.float32, device=dev),
                       state_conversion=StateConversion.COMMON)
    sim.reset_envs(state, env_ids=torch.arange(n, device=dev))


def fk_check(sim, robot_config, frames: dict) -> dict:
    n = sim.num_envs
    pos32, rot32, qpos = poselib_fk(robot_config.kinematic_info, frames["root_pos"], frames["root_quat_xyzw"],
                                    frames["dof"], torch.float32)
    pos64, rot64, _ = poselib_fk(robot_config.kinematic_info, frames["root_pos"], frames["root_quat_xyzw"],
                                 frames["dof"], torch.float64)
    assert pos32.shape[0] == n
    _reset(sim, pos32[:, 0], rot32[:, 0], qpos[:, 7:])
    rs = sim.get_robot_state()                               # COMMON order; link poses via PhysX FK, no step
    got_p, got_q = rs.rigid_body_pos.double().cpu(), rs.rigid_body_rot.double().cpu()
    e32 = (got_p - pos32.double()).norm(dim=-1)
    e64 = (got_p - pos64).norm(dim=-1)
    a32 = _quat_angle(got_q, rot32)
    a64 = _quat_angle(got_q, rot64)
    names = robot_config.kinematic_info.body_names
    worst = np.unravel_index(int(e32.argmax()), tuple(e32.shape))
    wr = np.unravel_index(int(a32.argmax()), tuple(a32.shape))
    back = rs.dof_pos.double().cpu()
    dof_err = (back - qpos[:, 7:].double()).abs()
    wd = np.unravel_index(int(dof_err.argmax()), tuple(dof_err.shape))
    dof_names = robot_config.kinematic_info.dof_names
    v = qpos[:, 7:].double().reshape(n, -1, 3)
    return {"frames": n, "holds": sum(r["hold_id"] is not None for r in frames["rows"]),
            "clipped_coords": frames["clipped_coords"], "clip_max_deg": frames["clip_max_deg"],
            "pos_max_err_m": float(e32.max()), "pos_p99_err_m": float(e32.flatten().quantile(0.99)),
            "pos_max_err_vs_float64_fk_m": float(e64.max()),
            "writer_float32_vs_float64_m": float((pos32.double() - pos64).norm(dim=-1).max()),
            "rot_max_err_deg": float(np.degrees(a32.max())),
            "rot_max_err_vs_float64_fk_deg": float(np.degrees(a64.max())),
            "rot_worst": {**frames["rows"][wr[0]], "body": names[wr[1]],
                          "joint_angle_deg": float(np.degrees(v[wr[0], wr[1] - 1].norm())) if wr[1] > 0 else None},
            "rot_max_err_by_body_deg": {b: float(np.degrees(a32[:, i].max())) for i, b in enumerate(names)},
            "dof_readback_max_err_rad": float(dof_err.max()),
            "dof_readback_worst": {**frames["rows"][wd[0]], "dof": dof_names[wd[1]],
                                   "written_rad": float(qpos[wd[0], 7 + wd[1]]), "read_rad": float(back[wd])},
            "worst": {**frames["rows"][worst[0]], "body": names[worst[1]]},
            "per_body_max_err_m": {b: float(e32[:, i].max()) for i, b in enumerate(names)}}


# --------------------------------------------------------------------------- #
# Reset stability
# --------------------------------------------------------------------------- #
def reset_cases(sim, robot_config, frames: dict, plant_xml: Path) -> tuple[list[dict], dict]:
    from reference_curation import mosh_replay as mr
    from scipy.spatial.transform import Rotation

    sk = mr.skeleton_for(plant_xml)
    h0 = float(robot_config.default_root_height)
    n_dof = len(robot_config.kinematic_info.dof_names)
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    cases = [{"case": "default", "root_z": h0, "quat": ident, "dof": np.zeros(n_dof), "hold": "zero"}]
    cases += [{"case": f"sunk_{round(100 * d)}cm", "root_z": h0 - d, "quat": ident, "dof": np.zeros(n_dof),
               "hold": "zero", "sink_m": d} for d in SINKS_M]
    cases.append({"case": "thrown_1ms", "root_z": h0, "quat": ident, "dof": np.zeros(n_dof), "hold": "zero",
                  "root_vz": THROW_VZ})
    idx = frames["standing"]
    pos, rot, _ = poselib_fk(robot_config.kinematic_info, frames["root_pos"][idx], frames["root_quat_xyzw"][idx],
                             frames["dof"][idx], torch.float64)
    rotm = Rotation.from_quat(rot.reshape(-1, 4).numpy()).as_matrix().reshape(*rot.shape[:2], 3, 3)
    low = mr.body_lowest(sk, pos.numpy(), rotm).min(1)
    gaps = mr.pair_gaps(sk, pos.numpy(), rotm)                      # [S, 253] PhysX's colliding pairs
    pairs = mr.colliding_pairs(sk)
    for k, i in enumerate(idx):
        r = frames["rows"][i]
        cases.append({"case": f"standing:{r['hold_id']}", "root_z": float(pos[k, 0, 2]) - float(low[k]) + RESPAWN_M,
                      "quat": frames["root_quat_xyzw"][i], "dof": frames["dof"][i], "hold": "pose",
                      "raw_lowest_cm": round(100 * float(low[k]), 3),
                      "self_overlap_cm": round(-100 * min(0.0, float(gaps[k].min())), 3),
                      "self_overlap_pair": "+".join(sk.names[j] for j in pairs[int(gaps[k].argmin())])})
    stats = {"standing_holds": len(idx), "standing_raw_lowest_cm": {
        "min": round(100 * float(low.min()), 3), "median": round(100 * float(np.median(low)), 3),
        "max": round(100 * float(low.max()), 3)} if len(idx) else None,
        "standing_self_overlapping": int((gaps.min(1) < 0).sum()) if len(idx) else None}
    return cases, stats


def reset_test(sim, robot_config, cases: list[dict], plant_xml: Path, duration: float) -> dict:
    from reference_curation import mosh_replay as mr
    from scipy.spatial.transform import Rotation

    n, dev = sim.num_envs, sim.device
    sk = mr.skeleton_for(plant_xml)
    which = np.arange(n) % len(cases)
    side = int(math.ceil(math.sqrt(n)))
    grid = np.array([[2.0 + GRID_M * (e % side), 2.0 + GRID_M * (e // side)] for e in range(n)])
    root = np.array([[*grid[e], cases[which[e]]["root_z"]] for e in range(n)])
    quat = np.stack([cases[c]["quat"] for c in which])
    dof = torch.as_tensor(np.stack([cases[c]["dof"] for c in which]), dtype=torch.float32)
    targets = dof.clone()
    for e in range(n):
        if cases[which[e]]["hold"] == "zero":
            targets[e] = 0.0
    vel = torch.zeros(n, 3)
    vel[:, 2] = torch.as_tensor([cases[c].get("root_vz", 0.0) for c in which])
    _reset(sim, torch.as_tensor(root), torch.as_tensor(quat), dof, vel)
    rs = sim.get_robot_state()
    p0 = rs.rigid_body_pos.double().cpu()
    q0 = rs.rigid_body_rot.double().cpu().numpy()
    lowest0 = mr.body_lowest(sk, p0.numpy(), Rotation.from_quat(q0.reshape(-1, 4)).as_matrix().reshape(*q0.shape[:2], 3, 3)).min(1)
    dt = sim.dt if hasattr(sim, "dt") else sim._sim.get_physics_dt() * sim.decimation
    steps = int(round(duration / dt))
    win = int(round(WINDOW_S / dt))
    targets = targets.to(dev)
    z, vz, vmax, jmax, jarg, disp_win = [], [], [], [], [], None
    for s in range(steps):
        sim.step(targets)
        st = sim.get_robot_state()
        p = st.rigid_body_pos.double().cpu()
        z.append(p[:, 0, 2])
        vz.append(st.rigid_body_vel[:, 0, 2].double().cpu())
        vmax.append(st.rigid_body_vel.double().norm(dim=-1).max(-1).values.cpu())
        jv = st.dof_vel.double().abs().cpu()
        jmax.append(jv.max(-1).values)
        jarg.append(jv.argmax(-1))
        if s + 1 == win:
            disp_win = (p - p0).norm(dim=-1).max(-1).values
    z, vz, vmax, jmax, jarg = (torch.stack(x, 1) for x in (z, vz, vmax, jmax, jarg))
    z0 = p0[:, 0, 2]
    rise = (z[:, :win].max(1).values - z0).clamp(min=0)
    # a pose that starts inside the floor is owed the rise that just clears it; a launch is anything beyond
    owed = torch.as_tensor(np.maximum(0.0, -lowest0))
    overshoot = (z[:, :win].max(1).values - z0 - owed).clamp(min=0)
    vz_up = vz[:, :win].max(1).values.clamp(min=0)
    launch = (overshoot > LAUNCH_RISE_M) | (vz_up > LAUNCH_VZ)
    dof_names = robot_config.kinematic_info.dof_names
    out = []
    for c, case in enumerate(cases):
        e = np.flatnonzero(which == c)
        if len(e) == 0:
            continue
        jm = jmax[e, :win]
        ew, sw = np.unravel_index(int(jm.argmax()), tuple(jm.shape))
        row = {"case": case["case"], "replicas": int(len(e)), "reset_root_z_m": round(float(z0[e].mean()), 5),
               "lowest_collider_at_reset_cm": round(100 * float(lowest0[e].min()), 3),
               "launched": int(launch[e].sum()),
               "rise_max_cm": round(100 * float(rise[e].max()), 3),
               "rise_min_cm": round(100 * float(rise[e].min()), 3),
               "overshoot_max_cm": round(100 * float(overshoot[e].max()), 3),
               "root_vz_up_max_m_s": round(float(vz_up[e].max()), 4),
               "root_drop_max_cm": round(100 * float((z0[e] - z[e, :win].min(1).values).max()), 3),
               "body_speed_max_m_s": round(float(vmax[e, :win].max()), 4),
               "joint_speed_max_rad_s": round(float(jm.max()), 3),
               "joint_speed_max_dof": dof_names[int(jarg[e, :win][ew, sw])],
               "joint_speed_max_step": int(sw) + 1,
               "joint_speed_step1_max_rad_s": round(float(jmax[e, 0].max()), 3),
               "moved_by_0_5s_max_cm": round(100 * float(disp_win[e].max()), 3),
               "root_z_end_m": round(float(z[e, -1].min()), 4),
               "root_drop_at_end_max_cm": round(100 * float((z0[e] - z[e, -1]).max()), 3)}
        for k in ("raw_lowest_cm", "self_overlap_cm", "self_overlap_pair"):
            if k in case:
                row[k] = case[k]
        out.append(row)
    standing = [r for r in out if r["case"].startswith("standing:")]
    by = {r["case"]: r for r in out}
    summary = {"policy_dt_s": dt, "steps": steps, "window_steps": win, "default": by.get("default"),
               "sunk": {r["case"]: r for r in out if r["case"].startswith("sunk_")}, "thrown_1ms": by.get("thrown_1ms"),
               "standing": {"cases": len(standing), "launched_cases": sum(r["launched"] > 0 for r in standing),
                            "rise_max_cm": max((r["rise_max_cm"] for r in standing), default=None),
                            "overshoot_max_cm": max((r["overshoot_max_cm"] for r in standing), default=None),
                            "self_overlapping_cases": sum(r["self_overlap_cm"] > 0 for r in standing),
                            "root_drop_at_end_median_cm": float(np.median([r["root_drop_at_end_max_cm"] for r in standing]))
                            if standing else None,
                            "root_vz_up_max_m_s": max((r["root_vz_up_max_m_s"] for r in standing), default=None),
                            "body_speed_max_m_s": max((r["body_speed_max_m_s"] for r in standing), default=None),
                            "joint_speed_max_rad_s": max((r["joint_speed_max_rad_s"] for r in standing), default=None)}}
    return {"summary": summary, "cases": out}


def failures(rec: dict) -> list[str]:
    bad = []
    p = rec["properties"]
    if p["mass_max_rel_err"] > MASS_RTOL or abs(p["total_physx_kg"] - p["total_mjcf_kg"]) > TOTAL_MASS_ATOL:
        bad.append(f"mass: worst body {p['mass_worst_body']} {p['mass_max_rel_err']:.2e}, total {p['total_physx_kg']:.4f}")
    if p["com_max_err_m"] > COM_TOL_M or p["inertia_max_rel_err"] > INERTIA_RTOL:
        bad.append(f"COM {p['com_max_err_m']:.2e} m / inertia {p['inertia_max_rel_err']:.2e}")
    e = p["dof_max_err"]
    if e["limit_rad"] > LIMIT_TOL_RAD or e["max_force_vs_mjcf"] > 1e-3:
        bad.append(f"dof limits {e['limit_rad']:.2e} rad / max force {e['max_force_vs_mjcf']:.2e}")
    if any(v > GAIN_RTOL for k, v in e.items() if k.endswith("_vs_config_rel")):
        bad.append(f"drive parameters differ from the robot config: {e}")
    fk = rec["fk"]
    if fk["pos_max_err_m"] > FK_TOL_M:
        bad.append(f"FK: {fk['pos_max_err_m']:.2e} m at {fk['worst']}")
    s = rec["reset"]["summary"]
    if s["default"]["launched"] or s["standing"]["launched_cases"]:
        bad.append(f"launch: default {s['default']['launched']}, standing {s['standing']['launched_cases']}")
    c = s["thrown_1ms"]
    if c["launched"] != c["replicas"]:
        bad.append(f"positive control: a root thrown up at {THROW_VZ} m/s was not seen as a launch: {c}")
    for d in SHALLOW_SINKS_M:
        c = s["sunk"][f"sunk_{round(100 * d)}cm"]
        if c["launched"] or c["rise_min_cm"] < RESTORED_FRAC * 100 * d:
            bad.append(f"a {100 * d:.0f} cm sink was not restored to the surface without a launch: {c}")
    return bad


def main() -> int:
    from lightning.fabric import Fabric

    from protomotions.robot_configs.factory import robot_config as make_robot_config
    from protomotions.simulator.factory import simulator_config as make_simulator_config
    from protomotions.utils.fabric_config import FabricConfig

    t0 = time.time()
    fabric = Fabric(**FabricConfig(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]).as_kwargs())
    fabric.launch()
    simulation_app = AppLauncher({"headless": True, "device": str(fabric.device)}).app

    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.scene_lib import SceneLibConfig
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils import plant_identity
    from protomotions.utils.component_builder import build_all_components
    from reference_curation import ids

    robot_config = make_robot_config(args.robot)
    asset = robot_config.asset
    plant_xml = Path(asset.asset_root) / asset.asset_file_name
    plant_xml = plant_xml if plant_xml.is_absolute() else ids.REPO / plant_xml
    flat = plant_identity.flat_path(str(plant_xml))
    usd = ids.REPO / asset.asset_root / asset.usd_asset_file_name
    frames = reference_frames(plant_xml, args.transition_frames)
    n = len(frames["rows"])
    simulator_config = make_simulator_config(simulator="isaaclab", robot_config=robot_config, headless=True,
                                             num_envs=n, experiment_name="plant_v2_physx")
    terrain_config, simulator_config = convert_friction_for_simulator(TerrainConfig(), simulator_config)
    comps = build_all_components(terrain_config=terrain_config, scene_lib_config=SceneLibConfig(),
                                 motion_lib_config=MotionLibConfig(), simulator_config=simulator_config,
                                 robot_config=robot_config, device=fabric.device, save_dir=None,
                                 simulation_app=simulation_app)
    sim = comps["simulator"]
    sim._initialize_with_markers(None)

    rec = {"robot": args.robot, "plant": plant_identity.identity(plant_xml), "usd": ids.display_path(usd),
           "num_envs": n, "physics_dt_s": sim._sim.get_physics_dt(), "decimation": sim.decimation}
    rec["properties"] = physx_properties(sim, robot_config, flat)
    rec["fk"] = fk_check(sim, robot_config, frames)
    cases, stats = reset_cases(sim, robot_config, frames, plant_xml)
    rec["reset"] = {**reset_test(sim, robot_config, cases, plant_xml, args.duration), **stats}
    layers = sorted((usd.parent / "configuration").glob("*.usd"))
    rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [plant_xml, flat, usd, *layers]), **rec,
           "seconds": round(time.time() - t0, 1)}
    rec["failures"] = failures(rec)
    out = Path(args.out) if args.out else ids.DATA_ROOT / "plant_v2" / f"physx_{args.robot}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    p, fk, s = rec["properties"], rec["fk"], rec["reset"]["summary"]
    print(f"plant physx {args.robot}: {p['total_physx_kg']:.4f} kg (body err <= {p['mass_max_rel_err']:.1e}, COM <= "
          f"{p['com_max_err_m']:.1e} m, inertia <= {p['inertia_max_rel_err']:.1e}); limits <= "
          f"{p['dof_max_err']['limit_rad']:.1e} rad; FK {fk['frames']} frames <= {fk['pos_max_err_m']:.1e} m; reset "
          f"default rise {s['default']['rise_max_cm']} cm / vz {s['default']['root_vz_up_max_m_s']} m/s, standing "
          f"{s['standing']['launched_cases']}/{s['standing']['cases']} launched, sinks "
          + ", ".join(f"{k[5:]} rise {v['rise_max_cm']} cm / {v['launched']} launched" for k, v in s["sunk"].items())
          + f", thrown {s['thrown_1ms']['launched']}/{s['thrown_1ms']['replicas']} launched -> {ids.display_path(out)}"
          + (f"; FAILED: {rec['failures']}" if rec["failures"] else ""), flush=True)
    os._exit(1 if rec["failures"] else 0)   # simulation_app.close() can hang (the known IsaacLab exit hang)


if __name__ == "__main__":
    sys.exit(main())
