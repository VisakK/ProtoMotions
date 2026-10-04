"""Card T1 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: plant v2 in MuJoCo, for batched CPU rollouts.

The trajectory generator (T2's dynamic check, T3's sampling MPC, T4) needs thousands of rollouts of the training
plant per second without the GPU. ProtoMotions' own MuJoCo backend cannot provide them (§1.3: one env, actuator
``gear`` ignored so every joint is bang-bang, PhysX exp-map coordinates written into XYZ hinges, 50 Hz control), so
this module builds its own copy of plant v2, matched to PhysX where a reference depends on it. It starts from the
benchmark ``evidence/to_bench/plant_v2_mj.py``; every setting below was measured there or in BodyFix.

=====================  ==========================================================================================
setting                value and reason
=====================  ==========================================================================================
model                  ``data/assets/smpl/smpl_yogi03596_v2_flat.xml`` (the plant's MJCF: 74 kg, her skeleton)
                       plus a floor plane
actuation              one PD servo per hinge at the smpl_yogi_v2 training gains (``GAINS``: leg 800/80, toe
                       500/50, spine 1000/100, neck 1158/116, arm and head 500/50, wrist and hand 300/30), gear
                       reset to 1, force clamped to the joint's ``actuatorfrcrange`` (hip 300 ... hand 15 N m)
integration            dt 1/240 s, ``implicitfast``, 8 physics steps per 30 Hz control step, PD targets held
                       (zero-order hold) across them like PhysX's decimation. At 1/120 noisy controls blew up
                       35 % of rollouts at 0.05 rad and 99.6 % at 0.15 rad (benchmark)
friction               floor 0.75, robot 0.5: MuJoCo combines by max, so robot-floor contacts see PhysX's
                       effective 0.75 and robot-robot contacts its default material's 0.5; ``condim`` 3 on the
                       robot geoms (the MJCF's 1 makes self-contact, e.g. crow's shin-on-arm brace, frictionless).
                       The friction cone is elliptic (``CONE``): PhysX's patch friction is isotropic, MuJoCo's
                       default pyramid is a diamond that loses 29 % on the diagonals (``slide_test``)
joints                 passive springs and damping zeroed; MuJoCo hinge limits **off**: PhysX's hard box is on
                       the exp-map coordinates, and the XYZ hinge chain is singular near 90 deg where crow's knees
                       and hips sit. The box is enforced as a cost (``box_excess``) and in the export projection
                       (``export_dof``: ``retarget.nearest_representative`` then a clip)
self-collision         every non-parent-child body pair (MuJoCo's parent filter = PhysX's 253 pairs)
=====================  ==========================================================================================

Coordinates. A pose is either *bodies* (``.motion`` COMMON order = MJCF order: ``pos [.., 24, 3]``,
``rot [.., 24, 4]`` xyzw), *exp-map* (root pos, root quat xyzw, ``dof [.., 69]`` = rotvec(R_parent^T R_child), PhysX's
own coordinates) or *hinge* (MuJoCo ``qpos [.., 76]``: root pos, root quat wxyz, 69 intrinsic-XYZ Euler angles).
Converters go through rotation matrices; the Euler branch is ``static_hold_lp``'s choice (the representation that
violates the MJCF hinge range least) or, along a trajectory, the branch nearest the previous frame
(``hinge_from_local``), so interpolated sketches never jump by pi.

Sensors (per physics step, ``SENSORS``): the whole-body COM, its linear velocity and the angular momentum about it,
and the ground contact force on every zone of ``extract_contact_configs.ZONES`` (MuJoCo ``contact`` sensors against
the floor, net force). Body poses are not sensed: ``fk`` recomputes them from ``qpos`` in numpy, which is cheaper than
returning them for every physics step.

CLI (the card's acceptance; CPU only, ``--threads`` <= 20 while a GPU run is active, PLAN.MD §2 Operations)::

    PYTHONPATH=.:data/scripts python -m edge_synthesis.plant_mj --acceptance [--threads 16]
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout
from scipy.spatial.transform import Rotation

from extract_contact_configs import ZONE_ORDER, ZONES
from edge_synthesis import gpu_guard
from reference_curation import ids

REPO = ids.REPO
MODULE = "edge_synthesis.plant_mj"
XML = REPO / "data/assets/smpl/smpl_yogi03596_v2_flat.xml"
MJCF = REPO / "data/assets/smpl/smpl_yogi03596_v2.xml"          # the plant's MJCF (pose_lib reads this one)
RELEASE_DIR = REPO / "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
RECORD = REPO / "expert_revist/graph_growth_2026_10_03/t1_plant_mj.json"

DT = 1.0 / 240.0
CTRL_HZ = 30
SUBSTEPS = 8
MU_FLOOR = 0.75
MU_BODY = 0.5
CONE = "elliptic"
GRAVITY = 9.81
# smpl_yogi_v2's PD gains (protomotions/robot_configs/smpl_yogi_v2.py; neck x2.3154 rounded as there)
GAINS = (
    (r".*_(Hip|Knee|Ankle)_.*", 800.0, 80.0),
    (r".*_Toe_.*", 500.0, 50.0),
    (r"(Torso|Spine|Chest)_.*", 1000.0, 100.0),
    (r"Neck_.*", 1158.0, 116.0),
    (r"(Head|.*_Thorax|.*_Shoulder|.*_Elbow)_.*", 500.0, 50.0),
    (r".*_(Wrist|Hand)_.*", 300.0, 30.0),
)
# The ground contact sensors: one MuJoCo body or subtree per sensor, summed per zone.
ZONE_SENSORS = {
    "L_FOOT": [("subtree1", "L_Ankle")], "R_FOOT": [("subtree1", "R_Ankle")],
    "L_SHANK": [("body1", "L_Knee")], "R_SHANK": [("body1", "R_Knee")],
    "L_THIGH": [("body1", "L_Hip")], "R_THIGH": [("body1", "R_Hip")],
    "PELVIS": [("body1", "Pelvis")],
    "TRUNK": [("body1", b) for b in ZONES["TRUNK"]],
    "HEAD": [("subtree1", "Neck")],
    "L_UPPER_ARM": [("body1", "L_Shoulder")], "R_UPPER_ARM": [("body1", "R_Shoulder")],
    "L_FOREARM": [("body1", "L_Elbow")], "R_FOREARM": [("body1", "R_Elbow")],
    "L_HAND": [("subtree1", "L_Wrist")], "R_HAND": [("subtree1", "R_Wrist")],
}


# Per body, the order of its three hinges (intrinsic, as MuJoCo composes the joints of one body). An XYZ-type chain
# is singular where its middle angle reaches +-90 deg; the MJCF's x, y, z order puts y -- the largest range of the
# hips (-140..60), knees (-5..160), ankles, toes, spine and neck -- in the middle, so every edge that folds a knee
# or a hip through 90 deg (crow <-> handstand, the jump-backs) would cross the singularity. Here the axis with the
# smallest range is the middle one; within the joint box it stays below 70 deg everywhere except the shoulders and
# hands, whose three ranges all span >= 180 deg (their y stays in the middle, as in the MJCF).
HINGE_ORDER = {
    "L_Hip": "yzx", "R_Hip": "yzx", "L_Knee": "yxz", "R_Knee": "yxz", "L_Ankle": "yxz", "R_Ankle": "yxz",
    "L_Toe": "yxz", "R_Toe": "yxz", "Torso": "yzx", "Spine": "yzx", "Chest": "yzx", "Neck": "yzx", "Head": "xzy",
    "L_Thorax": "xyz", "R_Thorax": "xyz", "L_Shoulder": "xyz", "R_Shoulder": "xyz", "L_Elbow": "zxy",
    "R_Elbow": "zxy", "L_Wrist": "xzy", "R_Wrist": "xzy", "L_Hand": "xyz", "R_Hand": "xyz",
}


def gains_for(joint: str) -> tuple[float, float]:
    for pat, kp, kd in GAINS:
        if re.fullmatch(pat, joint):
            return kp, kd
    raise KeyError(joint)


def model_xml(dt: float = DT, mu_floor: float = MU_FLOOR, mu_body: float = MU_BODY, cone: str = CONE,
              sensors: bool = True) -> str:
    """The plant's MJCF with the floor, solver options and sensors added (actuators are set on the compiled model)."""
    root = ET.parse(XML).getroot()
    qpos_names = []
    for body in root.iter("body"):
        joints = body.findall("joint")
        name = body.get("name")
        if name not in HINGE_ORDER:
            continue
        order = HINGE_ORDER[name]
        by_axis = {j.get("name")[-1]: j for j in joints}
        if sorted(by_axis) != ["x", "y", "z"] or len(joints) != 3:
            raise ValueError(f"{name}: expected hinges _x, _y, _z")
        first = list(body).index(joints[0])
        for j in joints:
            body.remove(j)
        for k, ax in enumerate(order):
            body.insert(first + k, by_axis[ax])
            qpos_names.append(f"{name}_{ax}")
    act = root.find("actuator")
    motors = {mt.get("joint"): mt for mt in act.findall("motor")}
    for mt in list(act):
        act.remove(mt)
    for jn in qpos_names:
        act.append(motors[jn])
    opt = ET.SubElement(root, "option")
    opt.set("timestep", repr(float(dt)))
    opt.set("integrator", "implicitfast")
    opt.set("cone", cone)
    for g in root.iter("geom"):
        g.set("condim", "3")
        g.set("friction", f"{mu_body} 0.005 0.0001")
    wb = root.find("worldbody")
    floor = ET.SubElement(wb, "geom")
    for k, v in (("name", "floor"), ("type", "plane"), ("size", "0 0 0.05"), ("condim", "3"),
                 ("friction", f"{mu_floor} 0.005 0.0001")):
        floor.set(k, v)
    if sensors:
        sens = ET.SubElement(root, "sensor")
        ET.SubElement(sens, "subtreecom", name="com", body="Pelvis")
        ET.SubElement(sens, "subtreelinvel", name="com_vel", body="Pelvis")
        ET.SubElement(sens, "subtreeangmom", name="ang_mom", body="Pelvis")
        for zone in ZONE_ORDER:
            for i, (kind, body) in enumerate(ZONE_SENSORS[zone]):
                ET.SubElement(sens, "contact", {"name": f"g_{zone}_{i}", kind: body, "geom2": "floor",
                                                "data": "force", "reduce": "netforce"})
    return ET.tostring(root, encoding="unicode")


def build_model(dt: float = DT, gain_scale: float = 1.0, mu_floor: float = MU_FLOOR, mu_body: float = MU_BODY,
                cone: str = CONE, limits: bool = False, sensors: bool = True) -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_string(model_xml(dt, mu_floor, mu_body, cone, sensors))
    m.jnt_stiffness[:] = 0.0
    m.dof_damping[:] = 0.0
    if not limits:
        m.jnt_limited[:] = 0
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0])
        if j != a + 1:
            raise ValueError(f"actuator {a} drives joint {j}: the actuator order is not the joint order")
        kp, kd = gains_for(m.joint(j).name)
        kp, kd = kp * gain_scale, kd * math.sqrt(gain_scale)
        eff = float(m.jnt_actfrcrange[j, 1])
        m.actuator_gear[a, :] = 0.0
        m.actuator_gear[a, 0] = 1.0
        m.actuator_gainprm[a, :] = 0.0
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[a, :] = 0.0
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kd
        m.actuator_ctrllimited[a] = 0
        m.actuator_forcelimited[a] = 1
        m.actuator_forcerange[a] = (-eff, eff)
    return m


# --------------------------------------------------------------------------- #
# Rotations (numpy, batched)
# --------------------------------------------------------------------------- #
def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(R.shape[:-2] + (4,))


def _elementary(axis: str, a: np.ndarray) -> np.ndarray:
    c, s_ = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i = "xyz".index(axis)
    j, k = (i + 1) % 3, (i + 2) % 3
    R[..., i, i] = 1.0
    R[..., j, j] = c
    R[..., k, k] = c
    R[..., j, k] = -s_
    R[..., k, j] = s_
    return R


def euler_to_mat(a: np.ndarray, order: str) -> np.ndarray:
    """``[.., 3]`` intrinsic angles in ``order`` (MuJoCo's hinge chain of one body) -> ``[.., 3, 3]``."""
    a = np.asarray(a, np.float64)
    return _elementary(order[0], a[..., 0]) @ _elementary(order[1], a[..., 1]) @ _elementary(order[2], a[..., 2])


def rotvec_of(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# The plant
# --------------------------------------------------------------------------- #
class Plant:
    """Plant v2 in MuJoCo: the model, its coordinates and batched rollouts.

    ``nthread`` worker ``MjData`` back ``rollout`` (a persistent pool; keep it <= 20 while a GPU run is active).
    """

    def __init__(self, dt: float = DT, gain_scale: float = 1.0, cone: str = CONE, mu_floor: float = MU_FLOOR,
                 mu_body: float = MU_BODY, nthread: int = 16, sensors: bool = True):
        self.dt, self.gain_scale, self.cone = float(dt), float(gain_scale), cone
        self.sub = int(round(1.0 / (dt * CTRL_HZ)))
        if abs(self.sub * dt * CTRL_HZ - 1.0) > 1e-9:
            raise ValueError(f"dt {dt} is not a divisor of the 30 Hz control step")
        self.model = build_model(dt, gain_scale, mu_floor, mu_body, cone, limits=False, sensors=sensors)
        m = self.model
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.body_names = [m.body(i).name for i in range(1, m.nbody)]          # 24, COMMON order
        self.body_index = {n: i for i, n in enumerate(self.body_names)}
        self.parents = [int(m.body_parentid[i]) - 1 for i in range(1, m.nbody)]
        self.offsets = m.body_pos[1:].copy()
        if not np.allclose(m.body_quat[1:], [1, 0, 0, 0]):
            raise ValueError("rotated bodies: fk assumes identity rest orientations")
        self.joint_names = [m.joint(j).name for j in range(1, m.njnt)]        # 69 hinges, qpos order
        self.hinge_range = m.jnt_range[1:].copy()                              # rad, qpos order (float64)
        self.orders = [HINGE_ORDER[b] for b in self.body_names[1:]]
        expect = [f"{b}_{ax}" for b, o in zip(self.body_names[1:], self.orders) for ax in o]
        if self.joint_names != expect:
            raise ValueError("the compiled hinge order is not HINGE_ORDER")
        # PhysX's exp-map box in the .motion dof order (x, y, z per body): the MJCF range of each named axis
        self.dof_names = [f"{b}_{ax}" for b in self.body_names[1:] for ax in "xyz"]
        jidx = {n: i for i, n in enumerate(self.joint_names)}
        self.dof_from_hinge = np.array([jidx[n] for n in self.dof_names])     # dof[k] <-> hinge[dof_from_hinge[k]]
        self.lower = self.hinge_range[self.dof_from_hinge, 0].copy()
        self.upper = self.hinge_range[self.dof_from_hinge, 1].copy()
        self.groups = {}
        for i, o in enumerate(self.orders):
            self.groups.setdefault(o, []).append(i)
        self.groups = {o: np.array(v) for o, v in self.groups.items()}
        self.torque_limit = m.jnt_actfrcrange[1:, 1].copy()
        self.kp = np.array([gains_for(n)[0] for n in self.joint_names]) * gain_scale
        self.kd = np.array([gains_for(n)[1] for n in self.joint_names]) * math.sqrt(gain_scale)
        self.mass = float(m.body_subtreemass[1])
        self.weight_n = self.mass * GRAVITY
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(m, self.spec)
        self.qpos_slice = slice(1, 1 + self.nq)
        self.qvel_slice = slice(1 + self.nq, 1 + self.nq + self.nv)
        self.sensor = self._sensor_layout() if sensors else {}
        self.nthread = int(nthread)
        self._datas = None
        self.zone_bodies = {z: [self.body_index[b] for b in ZONES[z]] for z in ZONE_ORDER}

    # ------------------------------------------------------------------ sensors
    def _sensor_layout(self) -> dict:
        m = self.model
        adr = {m.sensor(i).name: (int(m.sensor_adr[i]), int(m.sensor_dim[i])) for i in range(m.nsensor)}
        zone_cols = []
        for zone in ZONE_ORDER:
            zone_cols.append([adr[f"g_{zone}_{i}"][0] for i in range(len(ZONE_SENSORS[zone]))])
        return {"com": adr["com"][0], "com_vel": adr["com_vel"][0], "ang_mom": adr["ang_mom"][0], "zones": zone_cols,
                "n": int(m.nsensordata)}

    def zone_forces(self, sensordata: np.ndarray) -> np.ndarray:
        """``[.., 15, 3]`` net floor force on each zone (N, world frame, + up = the floor pushing the body)."""
        out = np.zeros(sensordata.shape[:-1] + (len(ZONE_ORDER), 3))
        for z, cols in enumerate(self.sensor["zones"]):
            for c in cols:
                out[..., z, :] += sensordata[..., c:c + 3]
        return -out      # MuJoCo reports the force body1 exerts on geom2 (the floor) in the contact frame's sense

    def com(self, sensordata: np.ndarray) -> np.ndarray:
        c = self.sensor["com"]
        return sensordata[..., c:c + 3]

    def com_vel(self, sensordata: np.ndarray) -> np.ndarray:
        c = self.sensor["com_vel"]
        return sensordata[..., c:c + 3]

    def ang_mom(self, sensordata: np.ndarray) -> np.ndarray:
        c = self.sensor["ang_mom"]
        return sensordata[..., c:c + 3]

    # ------------------------------------------------------------------ coordinates
    def hinge_from_local(self, R_local: np.ndarray, prev: np.ndarray | None = None) -> np.ndarray:
        """``[.., 23, 3, 3]`` local rotations -> ``[.., 69]`` hinge angles. Without ``prev`` the branch is
        ``static_hold_lp``'s (least hinge-range violation); with ``prev`` (``[.., 69]``) the branch and the 2 pi
        turn nearest ``prev`` (continuity along a trajectory)."""
        shp = R_local.shape[:-3]
        a = np.empty(shp + (23, 3))
        for o, idx in self.groups.items():
            Rg = R_local[..., idx, :, :]
            a[..., idx, :] = Rotation.from_matrix(Rg.reshape(-1, 3, 3)).as_euler(o.upper()).reshape(Rg.shape[:-2] + (3,))
        b = _wrap(np.stack([a[..., 0] + np.pi, np.pi - a[..., 1], a[..., 2] + np.pi], -1))
        if prev is None:
            rng = self.hinge_range.reshape(23, 3, 2)
            va = (np.maximum(0, rng[..., 0] - a) + np.maximum(0, a - rng[..., 1])).sum(-1)
            vb = (np.maximum(0, rng[..., 0] - b) + np.maximum(0, b - rng[..., 1])).sum(-1)
            out = np.where((va <= vb)[..., None], a, b)
        else:
            p = np.asarray(prev, np.float64).reshape(shp + (23, 3))
            a = p + _wrap(a - p)
            b = p + _wrap(b - p)
            da = np.abs(a - p).sum(-1)
            db = np.abs(b - p).sum(-1)
            out = np.where((da <= db)[..., None], a, b)
        return out.reshape(shp + (69,))

    def qpos_from_bodies(self, pos: np.ndarray, rot_xyzw: np.ndarray, prev: np.ndarray | None = None) -> np.ndarray:
        """``pos [.., 24, 3]``, ``rot [.., 24, 4]`` xyzw -> ``qpos [.., 76]``."""
        pos = np.asarray(pos, np.float64)
        Rw = quat_xyzw_to_mat(rot_xyzw)
        par = self.parents[1:]
        R_local = np.swapaxes(Rw[..., par, :, :], -1, -2) @ Rw[..., 1:, :, :]
        q = np.empty(pos.shape[:-2] + (self.nq,))
        q[..., :3] = pos[..., 0, :]
        rq = np.asarray(rot_xyzw, np.float64)[..., 0, :]
        q[..., 3:7] = np.concatenate([rq[..., 3:4], rq[..., :3]], -1)
        q[..., 7:] = self.hinge_from_local(R_local, None if prev is None else np.asarray(prev)[..., 7:])
        return q

    def qpos_from_expmap(self, root_pos, root_quat_xyzw, dof, prev=None) -> np.ndarray:
        dof = np.asarray(dof, np.float64)
        R_local = Rotation.from_rotvec(dof.reshape(-1, 3)).as_matrix().reshape(dof.shape[:-1] + (23, 3, 3))
        q = np.empty(dof.shape[:-1] + (self.nq,))
        q[..., :3] = np.asarray(root_pos, np.float64)
        rq = np.asarray(root_quat_xyzw, np.float64)
        q[..., 3:7] = np.concatenate([rq[..., 3:4], rq[..., :3]], -1)
        q[..., 7:] = self.hinge_from_local(R_local, None if prev is None else np.asarray(prev)[..., 7:])
        return q

    def local_rotations(self, qpos: np.ndarray) -> np.ndarray:
        a = np.asarray(qpos, np.float64)[..., 7:].reshape(np.shape(qpos)[:-1] + (23, 3))
        out = np.empty(a.shape[:-1] + (3, 3))
        for o, idx in self.groups.items():
            out[..., idx, :, :] = euler_to_mat(a[..., idx, :], o)
        return out

    def fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``qpos [.., 76]`` -> ``(pos [.., 24, 3], rot [.., 24, 3, 3])`` world body frames (numpy, batched)."""
        qpos = np.asarray(qpos, np.float64)
        loc = self.local_rotations(qpos)
        shp = qpos.shape[:-1]
        pos = np.empty(shp + (24, 3))
        rot = np.empty(shp + (24, 3, 3))
        pos[..., 0, :] = qpos[..., :3]
        w = qpos[..., 3:7]
        rot[..., 0, :, :] = quat_xyzw_to_mat(np.concatenate([w[..., 1:], w[..., :1]], -1))
        for i in range(1, 24):
            p = self.parents[i]
            pos[..., i, :] = pos[..., p, :] + np.einsum("...ij,j->...i", rot[..., p, :, :], self.offsets[i])
            rot[..., i, :, :] = rot[..., p, :, :] @ loc[..., i - 1, :, :]
        return pos, rot

    def expmap_from_qpos(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``qpos [.., 76]`` -> ``(root_pos, root_quat_xyzw, dof [.., 69])``: the principal rotation vectors."""
        qpos = np.asarray(qpos, np.float64)
        dof = rotvec_of(self.local_rotations(qpos)).reshape(qpos.shape[:-1] + (69,))
        w = qpos[..., 3:7]
        return qpos[..., :3].copy(), np.concatenate([w[..., 1:], w[..., :1]], -1), dof

    def box_excess(self, dof: np.ndarray) -> np.ndarray:
        """``[.., 69]`` rad past PhysX's exp-map box, taking each joint's better representative (``v`` or
        ``v - 2 pi v / |v|``, ``retarget.nearest_representative``'s alternative)."""
        v = np.asarray(dof, np.float64).reshape(dof.shape[:-1] + (23, 3))
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        alt = v - 2 * np.pi * v / np.maximum(n, 1e-9)
        lo, hi = self.lower.reshape(23, 3), self.upper.reshape(23, 3)

        def exc(x):
            return np.maximum(lo - x, 0) + np.maximum(x - hi, 0)
        ev, ea = exc(v), exc(alt)
        pick = (ea.sum(-1) < ev.sum(-1))[..., None]
        return np.where(pick, ea, ev).reshape(dof.shape)

    def export_dof(self, qpos: np.ndarray, margin_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """``qpos [T, 76]`` -> ``(root_pos, root_quat_xyzw, dof [T, 69], report)``: rotation vectors brought into
        the box by ``retarget.nearest_representative`` and clipped (``margin_rad`` inside). ``report`` gives the
        largest clip and the frames it touched (an export that clips more than 1 deg fails T5's contract)."""
        import torch

        from reference_curation import retarget as rt

        root_pos, root_q, dof = self.expmap_from_qpos(qpos)
        lo = self.lower + margin_rad
        hi = self.upper - margin_rad
        rep = rt.nearest_representative(torch.as_tensor(dof), torch.as_tensor(lo), torch.as_tensor(hi)).numpy()
        clipped = np.clip(rep, lo, hi)
        d = np.abs(clipped - rep)
        return root_pos, root_q, clipped, {"clip_max_deg": float(np.degrees(d.max())) if d.size else 0.0,
                                            "frames_clipped_1deg": int((np.degrees(d) > 1.0).any(-1).sum())}

    # ------------------------------------------------------------------ states and rollouts
    def data(self) -> mujoco.MjData:
        return mujoco.MjData(self.model)

    def state_of(self, d: mujoco.MjData) -> np.ndarray:
        s = np.empty(self.nstate)
        mujoco.mj_getState(self.model, d, s, self.spec)
        return s

    def make_state(self, qpos: np.ndarray, qvel: np.ndarray | None = None, ctrl: np.ndarray | None = None) -> tuple:
        """``(state, data)`` at ``qpos`` (and ``qvel``), forward-evaluated; ``ctrl`` defaults to the pose's own
        hinge angles (a PD servo holding the pose)."""
        d = self.data()
        d.qpos[:] = qpos
        d.qvel[:] = 0.0 if qvel is None else qvel
        d.ctrl[:] = np.asarray(qpos)[7:] if ctrl is None else ctrl
        mujoco.mj_forward(self.model, d)
        return self.state_of(d), d

    def _pool(self):
        if self._datas is None:
            self._datas = [mujoco.MjData(self.model) for _ in range(self.nthread)]
        return self._datas

    def rollout(self, init_state: np.ndarray, ctrl: np.ndarray, warmstart: np.ndarray | None = None) -> tuple:
        """Open-loop rollouts. ``init_state [N or 1, nstate]``, ``ctrl [N, H, 69]`` PD targets per **control** step
        (held for ``sub`` physics steps). Returns ``(state [N, H*sub, nstate], sensordata [N, H*sub, ns])``."""
        ctrl = np.asarray(ctrl, np.float64)
        n, h = ctrl.shape[:2]
        full = np.repeat(ctrl, self.sub, axis=1)
        init = np.asarray(init_state, np.float64)
        if init.ndim == 1:
            init = init[None]
        if init.shape[0] == 1 and n > 1:
            init = np.repeat(init, n, axis=0)
        kw = {}
        if warmstart is not None:
            ws = np.asarray(warmstart, np.float64)
            kw["initial_warmstart"] = np.repeat(ws[None], n, 0) if ws.ndim == 1 else ws
        return mj_rollout.rollout(self.model, self._pool(), init, full, nstep=h * self.sub, persistent_pool=True,
                                  **kw)

    def close(self):
        mj_rollout.shutdown_persistent_pool()
        self._datas = None


# --------------------------------------------------------------------------- #
# Release frames
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=64)
def load_motion(stem: str):
    import torch

    return torch.load(RELEASE_DIR / "motions" / f"{stem}.motion", map_location="cpu", weights_only=False)


def release_frame(stem: str, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(pos [24,3], rot [24,4] xyzw, dof [69])`` of a release v2 frame (float64 of the stored float32)."""
    mot = load_motion(stem)
    return (mot["rigid_body_pos"][frame].double().numpy(), mot["rigid_body_rot"][frame].double().numpy(),
            mot["dof_pos"][frame].double().numpy())


# --------------------------------------------------------------------------- #
# Acceptance (the card): FK, throughput, slide, (MPPI hold in mppi.py)
# --------------------------------------------------------------------------- #
def fk_test(plant: Plant, n: int = 100, seed: int = 0) -> dict:
    """``n`` release frames: exp-map -> hinge -> MuJoCo FK (``mj_kinematics``) and this module's numpy ``fk``,
    against pose_lib's FK of the same exp-map coordinates in float64 (the ``.motion`` writer's FK)."""
    import torch

    rng = np.random.default_rng(seed)
    stems = sorted(p.stem for p in (RELEASE_DIR / "motions").glob("*.motion") if not re.search(r"_x\d+s$", p.stem))
    rows = []
    for _ in range(n):
        stem = stems[rng.integers(len(stems))]
        mot = load_motion(stem)
        f = int(rng.integers(mot["dof_pos"].shape[0]))
        rows.append((stem, f))
    root_pos = np.stack([load_motion(s)["rigid_body_pos"][f, 0].double().numpy() for s, f in rows])
    root_rot = np.stack([load_motion(s)["rigid_body_rot"][f, 0].double().numpy() for s, f in rows])
    dof = np.stack([load_motion(s)["dof_pos"][f].double().numpy() for s, f in rows])
    # pose_lib FK in float64: plant_v2_physx.poselib_fk's lines. Never import that module here: it launches
    # IsaacLab (the GPU) at import.
    from protomotions.components.pose_lib import (extract_kinematic_info, extract_transforms_from_qpos,
                                                  fk_from_transforms_with_velocities)

    kin = extract_kinematic_info(str(MJCF))
    q = torch.as_tensor(root_rot)
    qpos_pl = torch.cat([torch.as_tensor(root_pos), q[:, [3, 0, 1, 2]], torch.as_tensor(dof)], 1).double()
    rp, jrm = extract_transforms_from_qpos(kin, qpos_pl, qpos_is_exp_map_on_3dof_joints=True)
    st = fk_from_transforms_with_velocities(kin, rp, jrm, fps=None, compute_velocities=False)
    ref = st.rigid_body_pos.double().numpy()
    qpos = plant.qpos_from_expmap(root_pos, root_rot, dof)
    d = plant.data()
    err_mj = 0.0
    for i in range(n):
        d.qpos[:] = qpos[i]
        mujoco.mj_kinematics(plant.model, d)
        err_mj = max(err_mj, float(np.abs(d.xpos[1:] - ref[i]).max()))
    pos_np, rot_np = plant.fk(qpos)
    err_np = float(np.abs(pos_np - ref).max())
    # round trip hinge -> exp-map: the rotation is reproduced (the representative may differ by 2 pi turns)
    _, _, dof_back = plant.expmap_from_qpos(qpos)
    R1 = Rotation.from_rotvec(dof.reshape(-1, 3)).as_matrix()
    R2 = Rotation.from_rotvec(dof_back.reshape(-1, 3)).as_matrix()
    rot_rt = float(np.abs(R1 - R2).max())
    stored = np.stack([load_motion(s)["rigid_body_pos"][f].double().numpy() for s, f in rows])
    return {"frames": n, "clips": len({s for s, _ in rows}), "mj_kinematics_vs_poselib_m": err_mj,
            "numpy_fk_vs_poselib_m": err_np, "stored_float32_vs_poselib_m": float(np.abs(stored - ref).max()),
            "expmap_round_trip_rot": rot_rt, "pass": max(err_mj, err_np) <= 1e-6 and rot_rt <= 1e-9}


def throughput_test(plant: Plant, nbatch: int = 256, h: int = 30, noise: float = 0.05, reps: int = 3,
                    pose=("220923_Crane_Crow_Pose_or_Bakasana_-a", 651)) -> dict:
    pos, rot, _ = release_frame(*pose)
    qpos = plant.qpos_from_bodies(pos, rot)
    s0, _ = plant.make_state(qpos)
    rng = np.random.default_rng(0)
    ctrl = qpos[7:][None, None] + noise * rng.standard_normal((nbatch, h, plant.nu))
    plant.rollout(s0, ctrl)                       # warm the pool
    t0 = time.perf_counter()
    for _ in range(reps):
        st, _ = plant.rollout(s0, ctrl)
    el = (time.perf_counter() - t0) / reps
    finite = bool(np.isfinite(st).all())
    tcol = st[:, :, 0]
    expected = s0[0] + plant.dt * np.arange(1, h * plant.sub + 1)
    reset = ~np.isclose(tcol, expected[None], atol=1e-6).all(1)
    return {"threads": plant.nthread, "nbatch": nbatch, "horizon_ctrl_steps": h, "noise_rad": noise,
            "ms_per_batch": round(1e3 * el, 1), "ctrl_steps_per_s": round(nbatch * h / el),
            "physics_steps_per_s": round(nbatch * h * plant.sub / el), "unstable_share": float(reset.mean()),
            "finite": finite, "pass": nbatch * h / el >= 20000 and finite and not reset.any()}


def slide_test(cone: str = CONE, tilts=(0.70, 0.74, 0.76, 0.80), azimuths_deg=(0.0, 45.0, 90.0),
               seconds: float = 1.0) -> dict:
    """Is the robot-floor friction coefficient 0.75, in every direction? Two of the plant's own colliders -- the
    left foot box and the left palm box, with their MJCF size, density, solref and solimp and this module's robot
    friction (0.5, condim 3) -- rest on this module's floor (0.75) under gravity tilted by ``atan(t)`` towards each
    azimuth. MuJoCo combines friction by max, so they must stick below t = 0.75 (creep < 1 cm in ``seconds``) and
    slide above it (> 2 cm). A whole-body test (the robot supine, tilted) is not used: the body rolls and yaws
    10-30 deg before it slides, so its displacement does not measure friction."""
    root = ET.parse(XML).getroot()
    boxes = {}
    for body in root.iter("body"):
        if body.get("name") in ("L_Ankle", "L_Hand"):
            boxes[body.get("name")] = next(g for g in body.findall("geom") if g.get("type") == "box")
    out, ok_stick, ok_slip = {}, True, True
    for name, g in boxes.items():
        size = [float(x) for x in g.get("size").split()]
        xml = (f'<mujoco><option timestep="{DT}" integrator="implicitfast" cone="{cone}"/><worldbody>'
               f'<geom name="floor" type="plane" size="0 0 0.05" condim="3" friction="{MU_FLOOR} 0.005 0.0001"/>'
               f'<body name="b" pos="0 0 {size[2] + 1e-4}"><freejoint/><geom type="box" size="{g.get("size")}" '
               f'density="{g.get("density")}" condim="3" friction="{MU_BODY} 0.005 0.0001" solimp="{g.get("solimp")}" '
               f'solref="{g.get("solref")}"/></body></worldbody></mujoco>')
        for az in azimuths_deg:
            for t in tilts:
                m = mujoco.MjModel.from_xml_string(xml)
                th, a = math.atan(t), math.radians(az)
                m.opt.gravity[:] = GRAVITY * np.array([math.sin(th) * math.cos(a), math.sin(th) * math.sin(a),
                                                       -math.cos(th)])
                d = mujoco.MjData(m)
                for _ in range(int(0.2 / DT)):
                    mujoco.mj_step(m, d)
                x0 = d.qpos[:2].copy()
                for _ in range(int(seconds / DT)):
                    mujoco.mj_step(m, d)
                disp = float(np.linalg.norm(d.qpos[:2] - x0))
                out[f"{name}_az{az:.0f}_t{t:.2f}"] = round(disp, 4)
                if t < MU_FLOOR:
                    ok_stick &= disp < 0.01
                else:
                    ok_slip &= disp > 0.02
    return {"cone": cone, "displacement_m_after_s": seconds, "displacement": out, "sticks_below_0.75": ok_stick,
            "slides_above_0.75": ok_slip, "pass": ok_stick and ok_slip}


def _geom_lowest_z(m, d, g) -> float:
    """Lowest world z of geom ``g`` (sphere, capsule, box)."""
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    s = m.geom_size[g]
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(c[2] - s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(c[2] - abs(R[2, 2]) * s[1] - s[0])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return float(c[2] - np.abs(R[2, :]) @ s)
    raise ValueError(f"geom type {t}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acceptance", action="store_true")
    ap.add_argument("--threads", type=int, default=gpu_guard.POLITE_THREADS)
    ap.add_argument("--skip-mppi", action="store_true")
    args = ap.parse_args(argv)
    if args.threads > 20:
        print("refusing > 20 threads while a GPU run may be active (PLAN.MD §2 Operations)", file=sys.stderr)
        return 2
    print("cpu policy", gpu_guard.be_polite())
    rec = {**ids.provenance(1, MODULE, __file__, [XML]), "dt": DT, "substeps": SUBSTEPS, "cone": CONE,
           "mu_floor": MU_FLOOR, "mu_body": MU_BODY}
    plant = Plant(nthread=args.threads)
    rec["model"] = {"nq": plant.nq, "nv": plant.nv, "nu": plant.nu, "mass_kg": round(plant.mass, 4),
                    "nsensordata": plant.sensor["n"], "nstate": plant.nstate}
    rec["fk"] = fk_test(plant)
    print("fk", json.dumps(rec["fk"]))
    rec["throughput"] = {}
    for c in ("elliptic", "pyramidal"):
        p = Plant(nthread=args.threads, cone=c)
        rec["throughput"][c] = throughput_test(p)
        p.close()
    print("throughput", json.dumps(rec["throughput"]))
    rec["slide"] = {c: slide_test(c) for c in ("elliptic", "pyramidal")}
    print("slide", json.dumps(rec["slide"]))
    if not args.skip_mppi:
        from edge_synthesis import mppi

        rec["mppi_hold"] = mppi.hold_acceptance(nthread=args.threads)
        print("mppi_hold", json.dumps(rec["mppi_hold"]))
    plant.close()
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    ok = rec["fk"]["pass"] and rec["throughput"][CONE]["pass"] and rec["slide"][CONE]["pass"] and \
        (args.skip_mppi or rec["mppi_hold"]["pass"])
    print("T1 acceptance", "PASSED" if ok else "FAILED", "->", ids.display_path(RECORD))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
