# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quasi-static feasibility of a hold pose on the smpl_yogi plant (MuJoCo model, 74 kg).

Maintained copy of ``expert_revist/contact_balance_investigation/static_hold_lp.py`` (the
investigation's record); ``repair_armbalance_holds.py`` uses this one.

For a pose q (qvel = qacc = 0) and a candidate contact set, find contact forces inside linearised
friction cones that balance gravity, minimising the peak joint-torque utilisation

    s* = min_f max_i |tau_i| / tau_max_i,   tau = g(q) - sum_c J_c(q)^T f_c,   root rows of tau = 0.

s* > 1: the plant cannot hold this pose with these supports, whatever the policy does.
Infeasible: the supports cannot balance gravity at all (COM outside the support polygon).
Body-body contacts are internal forces (f on body A, -f on body B): they never change the
root-row balance, only the joint torques along the chain between the two bodies.

Also reports the per-joint utilisation at the optimum and the contact normal forces.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
sys.path.insert(0, str(REPO))
from contact_geometry import parse_typed_geoms, geom_to_world, geom_pair_distance, _box_corners_world  # noqa
from extract_contact_configs import mjcf_body_names, ZONES  # noqa

from protomotions.utils import plant_identity  # noqa: E402

# The plant: REFERENCE_PLANT (v1 default, the shipped smpl_yogi03596_lowtorque; v2 the performer's own body).
FLAT = plant_identity.flat_path()
MJCF = plant_identity.mjcf_path()
M = mujoco.MjModel.from_xml_path(str(FLAT))
D = mujoco.MjData(M)
BODY = [M.body(i).name for i in range(1, M.nbody)]
PARENT = [M.body_parentid[i] - 1 for i in range(1, M.nbody)]
TYPED = parse_typed_geoms(str(MJCF), mjcf_body_names(str(MJCF)))
MU_GROUND, MU_BODY = 0.75, 0.5
TAU_MAX = np.array([M.jnt_actfrcrange[j][1] for j in range(1, M.njnt)])      # 69 hinges
JNAMES = [M.joint(j).name for j in range(1, M.njnt)]


def _euler_branch(loc, rng):
    a = loc.as_euler("XYZ")
    b = np.array([a[0] + np.pi, np.pi - a[1], a[2] + np.pi])
    b = (b + np.pi) % (2 * np.pi) - np.pi

    def viol(x):
        return np.sum(np.maximum(0, rng[:, 0] - x) + np.maximum(0, x - rng[:, 1]))
    return a if viol(a) <= viol(b) else b


def set_pose(pos, rot):
    qp = np.zeros(M.nq)
    qp[:3] = pos[0]
    q = rot[0]
    qp[3:7] = [q[3], q[0], q[1], q[2]]
    k = 7
    for b in range(1, 24):
        loc = R.from_quat(rot[PARENT[b]]).inv() * R.from_quat(rot[b])
        rng = M.jnt_range[1 + (k - 7): 1 + (k - 7) + 3]
        qp[k:k + 3] = _euler_branch(loc, rng)
        k += 3
    D.qpos[:] = qp
    D.qvel[:] = 0
    D.qacc[:] = 0
    mujoco.mj_forward(M, D)
    err = np.abs(D.xpos[1:] - pos).max()
    assert err < 1e-4, f"FK mismatch {err}"
    bias = np.zeros(M.nv)
    mujoco.mj_rne(M, D, 0, bias)          # g(q) at qvel = 0
    return bias


def pyramid(n, mu, k=4):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return [n + mu * t for t in (t1, -t1, t2, -t2)][:k]


def jac_point(body_name, point):
    jp = np.zeros((3, M.nv))
    mujoco.mj_jac(M, D, jp, None, point, M.body(body_name).id)
    return jp


def ground_points(pos, rot, zone, face_tol=0.012):
    """Bottom-face corners of a zone's box geoms (hands / feet) -> list of (body, point)."""
    out = []
    for b in ZONES[zone]:
        g = TYPED[b][0]
        i = BODY.index(b)
        wg = geom_to_world(g, torch.tensor(pos[i:i + 1]), torch.tensor(rot[i:i + 1]))
        if g["type"] == "box":
            c = _box_corners_world(wg)[0].numpy()
            keep = np.argsort(c[:, 2])[:4]          # the bottom face, whatever the fit's tilt
            out += [(b, p) for p in c[keep]]
        elif g["type"] == "capsule":
            for p in (wg["a"][0].numpy(), wg["b"][0].numpy()):
                q = p.copy(); q[2] -= g["radius"]; out.append((b, q))
        else:
            q = wg["center"][0].numpy().copy(); q[2] -= g["radius"]; out.append((b, q))
    return out


def body_pair_point(pos, rot, za, zb):
    """Closest member-body pair of two zones -> (body_a, pa, body_b, pb, gap)."""
    best = None
    for a in ZONES[za]:
        for b in ZONES[zb]:
            ia, ib = BODY.index(a), BODY.index(b)
            ga = geom_to_world(TYPED[a][0], torch.tensor(pos[ia:ia + 1]), torch.tensor(rot[ia:ia + 1]))
            gb = geom_to_world(TYPED[b][0], torch.tensor(pos[ib:ib + 1]), torch.tensor(rot[ib:ib + 1]))
            g, wa, wb = geom_pair_distance(ga, gb)
            if best is None or float(g[0]) < best[4]:
                best = (a, wa[0].numpy(), b, wb[0].numpy(), float(g[0]))
    return best


def solve(pos, rot, ground_zones, body_pairs=(), verbose=False):
    bias = set_pose(pos, rot)
    cols, meta = [], []
    for z in ground_zones:
        for b, p in ground_points(pos, rot, z):
            J = jac_point(b, p)
            for e in pyramid(np.array([0, 0, 1.0]), MU_GROUND):
                cols.append(J.T @ e)
                meta.append(("G", z, e[2]))
    gaps = {}
    for pair in body_pairs:
        za, zb = pair.split("+")
        a, pa, b, pb, gap = body_pair_point(pos, rot, za, zb)
        gaps[pair] = gap
        # witness vector pb - pa = gap * u, u the unit normal from A to B, for either sign of gap
        if abs(gap) > 1e-4:
            u = (pb - pa) / gap
        else:
            u = D.xpos[M.body(b).id] - D.xpos[M.body(a).id]
        n = -u / np.linalg.norm(u)            # force on A pushes A away from B
        pm = 0.5 * (pa + pb)                  # one application point: the pair wrench is internal
        Ja, Jb = jac_point(a, pm), jac_point(b, pm)
        for e in pyramid(n, MU_BODY):
            cols.append(Ja.T @ e - Jb.T @ e)
            meta.append(("B", pair, float(np.dot(e, n))))
    A = np.array(cols).T if cols else np.zeros((M.nv, 0))    # [nv, K]
    K = A.shape[1]
    # tau = bias - A lam ; root rows: A_r lam = bias_r
    Ar, br = A[:6], bias[:6]
    Aj, bj = A[6:], bias[6:]
    # variables [lam (K), s]
    c = np.zeros(K + 1); c[-1] = 1.0
    A_ub = np.vstack([np.hstack([-Aj, -TAU_MAX[:, None]]), np.hstack([Aj, -TAU_MAX[:, None]])])
    b_ub = np.concatenate([-bj, bj])
    A_eq = np.hstack([Ar, np.zeros((6, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=br, bounds=[(0, None)] * (K + 1), method="highs")
    out = {"feasible": bool(res.status == 0), "gaps_cm": {k: round(100 * v, 1) for k, v in gaps.items()}}
    if res.status != 0:
        return out
    lam, s = res.x[:K], res.x[-1]
    tau = bj - Aj @ lam
    util = np.abs(tau) / TAU_MAX
    order = np.argsort(-util)[:6]
    out["s_star"] = float(s)
    out["top_joints"] = [(JNAMES[i], round(float(util[i]), 2), round(float(tau[i]), 1)) for i in order]
    forces = {}
    for (kind, name, nz), l in zip(meta, lam):
        forces[name] = forces.get(name, 0.0) + l * nz     # normal component
    out["normal_forces_N"] = {k: round(v, 1) for k, v in forces.items()}
    # joint group utilisations
    groups = {"shoulders": ["Shoulder"], "hips": ["Hip"], "spine": ["Torso", "Spine", "Chest"], "wrists": ["Wrist", "Hand"],
              "elbows": ["Elbow"], "knees": ["Knee"]}
    out["group_max_util"] = {g: round(float(max(util[i] for i, n in enumerate(JNAMES) if any(k in n for k in ks))), 2)
                             for g, ks in groups.items()}
    return out


def load_frame(path, t, fps_override=None):
    mot = torch.load(str(path), map_location="cpu", weights_only=False)
    fps = float(mot["fps"])
    i = int(np.clip(round(t * fps), 0, mot["rigid_body_pos"].shape[0] - 1))
    return mot["rigid_body_pos"][i].double().numpy(), mot["rigid_body_rot"][i].double().numpy()


