"""Plant v2 in MuJoCo for sampling-MPC benchmarks (scratch; CPU only).

Builds data/assets/smpl/smpl_yogi03596_v2_flat.xml + a floor, with:
  * effective ground friction 0.75 (MuJoCo combines by max: floor 0.75, robot 0.5 -> robot-robot 0.5 like PhysX's
    default material), condim 3 everywhere (the MJCF's condim=1 makes self-contact frictionless),
  * passive joint stiffness/damping zeroed (the backend does the same),
  * motors converted to PD position actuators at the smpl_yogi_v2 training gains, gear reset to 1, force limit =
    actuatorfrcrange (the MJCF's gear 20-300 is NOT reset by protomotions' MuJoCo backend),
  * optional joint limits off (PhysX's box is on exp-map coordinates, MuJoCo's on XYZ hinge angles).
Poses come from release-v2 .motion frames (COMMON body order = MJCF order), converted to hinge angles by
intrinsic XYZ Euler with static_hold_lp's branch choice.
"""
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO = "/home/visakii/Documents/moves/ProtoMotions"
XML = f"{REPO}/data/assets/smpl/smpl_yogi03596_v2_flat.xml"
REL = f"{REPO}/data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac/motions"

# smpl_yogi_v2 training gains (protomotions/robot_configs/smpl_yogi_v2.py)
GAINS = [
    (r".*_(Hip|Knee|Ankle)_.*", 800, 80),
    (r".*_Toe_.*", 500, 50),
    (r"(Torso|Spine|Chest)_.*", 1000, 100),
    (r"Neck_.*", 1158, 116),
    (r"(Head|.*_Thorax|.*_Shoulder|.*_Elbow)_.*", 500, 50),
    (r".*_(Wrist|Hand)_.*", 300, 30),
]


def gains_for(name):
    for pat, kp, kd in GAINS:
        if re.fullmatch(pat, name):
            return kp, kd
    raise KeyError(name)


def build(dt=1 / 120, integrator="implicitfast", limits=False, gain_scale=1.0, mu_floor=0.75, mu_body=0.5):
    tree = ET.parse(XML)
    root = tree.getroot()
    opt = ET.SubElement(root, "option")
    opt.set("timestep", repr(dt))
    opt.set("integrator", integrator)
    for g in root.iter("geom"):
        g.set("condim", "3")
        g.set("friction", f"{mu_body} 0.005 0.0001")
    wb = root.find("worldbody")
    fl = ET.SubElement(wb, "geom")
    fl.set("name", "floor"); fl.set("type", "plane"); fl.set("size", "0 0 0.05")
    fl.set("friction", f"{mu_floor} 0.005 0.0001"); fl.set("condim", "3")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    m.jnt_stiffness[:] = 0.0
    m.dof_damping[:] = 0.0
    if not limits:
        m.jnt_limited[:] = 0
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        kp, kd = gains_for(name)
        kp, kd = kp * gain_scale, kd * np.sqrt(gain_scale)
        eff = m.jnt_actfrcrange[j, 1]
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


def euler_branch(loc, rng):
    a = loc.as_euler("XYZ")
    b = np.array([a[0] + np.pi, np.pi - a[1], a[2] + np.pi])
    b = (b + np.pi) % (2 * np.pi) - np.pi

    def viol(x):
        return np.sum(np.maximum(0, rng[:, 0] - x) + np.maximum(0, x - rng[:, 1]))
    return a if viol(a) <= viol(b) else b


def load_frame(stem, frame):
    mot = torch.load(f"{REL}/{stem}.motion", map_location="cpu", weights_only=False)
    pos = mot["rigid_body_pos"][frame].double().numpy()
    rot = mot["rigid_body_rot"][frame].double().numpy()      # xyzw
    return pos, rot, mot


def qpos_from_bodies(m, pos, rot):
    parent = [m.body_parentid[i] - 1 for i in range(1, m.nbody)]
    nb = m.nbody - 1 - 0
    qp = np.zeros(m.nq)
    qp[:3] = pos[0]
    q = rot[0]
    qp[3:7] = [q[3], q[0], q[1], q[2]]
    k = 7
    rng_all = m.jnt_range[1:]
    viol = 0.0
    for b in range(1, 24):
        loc = R.from_quat(rot[parent[b]]).inv() * R.from_quat(rot[b])
        rng = rng_all[(k - 7):(k - 7) + 3]
        e = euler_branch(loc, rng)
        viol = max(viol, float(np.max(np.maximum(0, rng[:, 0] - e) + np.maximum(0, e - rng[:, 1]))))
        qp[k:k + 3] = e
        k += 3
    d = mujoco.MjData(m)
    d.qpos[:] = qp
    mujoco.mj_forward(m, d)
    err = float(np.abs(d.xpos[1:25] - pos).max())
    return qp, err, np.degrees(viol)


def full_state(m, d):
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    s = np.zeros(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, s, spec)
    return s
