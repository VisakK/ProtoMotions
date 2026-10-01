# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The performer's own MoSh++ fit (SMPL-X female) as plant coordinates, on any MJCF plant.

Adopted from the subject-body check's prototype (``output/reference_curation/subject_body_check/prototype/
replay.py``, BodyFix §3); the functions and their conventions are unchanged, ``use_plant`` also redirects the
statics module's copied joint box. BodyFix Step 1 validates plant v2 with it and Step 3's writer builds on
``mosh_kinematics``.

    kin = mosh_replay.mosh_kinematics("220923_Crane_Crow_Pose_or_Bakasana_-a")          # hand="fingers"
    sk = mosh_replay.skeleton_for(mosh_replay.V2_XML)
    pos, rot = mosh_replay.fk_bodies(sk, kin)                # [T,24,3], [T,24,3,3] clip frame, COMMON order
    low = mosh_replay.zone_lowest(sk, pos, rot)              # [T,15] lowest collider surface per zone (m)

Conventions (all measured by the subject-body check, ``prototype/validate_replay.py``):

* **Bodies** are in COMMON order (``extract_contact_configs.mjcf_body_names``), the ``.motion`` order.
* **Rotations.** Every avatar body's world rotation is its SMPL-X joint's global rotation times
  ``retarget.AXES`` (``R_b = G_j @ AXES``; ``retarget.BODY_JOINT``). The hand bodies have no SMPL-X joint:
  ``hand="fingers"`` (default) gives them the chordal mean of the four proximal finger joints' global
  rotations (index1, middle1, pinky1, ring1), so the finger box follows the fingers; ``hand="wrist"`` gives
  them the wrist's rotation (local identity), which is what every shipped ``.motion`` stores.
* **dof** ``[T,69]``: for body ``i >= 1`` columns ``3(i-1):3i`` are ``rotvec(R_parent^T R_i)``, the
  principal exp-map (|v| <= pi), PhysX's own joint coordinates (``retarget.fk``). Pass
  ``representative="nearest"`` to apply ``retarget.nearest_representative`` against a plant's joint box.
* **Positions.** ``root_pos`` is the posed pelvis joint in the clip frame:
  ``LBS pelvis + trans + (0, 0, +10.17 mm) + (capture.VICON_TO_CLIP_XY, 0)`` (``human_mesh``'s
  registration). ``human_joints [T,55,3]`` are the posed SMPL-X joints in the same frame.

On a plant whose offsets are the performer's own joints (plant v2) ``fk_bodies`` reproduces
``human_joints`` exactly (body origins = SMPL-X joints; hands = mean of 4 finger bases); on the shipped plant
the residual is the skeleton mismatch.
"""

from __future__ import annotations

import contextlib
import functools
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from contact_geometry import geom_ground_distance, geom_to_world, parse_typed_geoms
from extract_contact_configs import ZONE_ORDER, ZONES, mjcf_body_names
from reference_curation import capture, ids
from reference_curation import human_mesh as hm
from reference_curation import retarget as rt

MODULE = "reference_curation.mosh_replay"
F64 = torch.float64
SHIPPED_XML = ids.MJCF
SHIPPED_FLAT = ids.REPO / "data/assets/smpl/smpl_yogi03596_lowtorque_flat.xml"
V2_XML = ids.REPO / "data/assets/smpl/smpl_yogi03596_v2.xml"
V2_FLAT = ids.REPO / "data/assets/smpl/smpl_yogi03596_v2_flat.xml"
MOSH_TO_MARKER_Z = hm.EXPECTED_OFFSET_M[2]                    # +10.17 mm (human_mesh.register)
CLIP_SHIFT = np.array([capture.VICON_TO_CLIP_XY[0], capture.VICON_TO_CLIP_XY[1], MOSH_TO_MARKER_Z])
HAND_MODES = ("wrist", "fingers")


# --------------------------------------------------------------------------- #
# Skeletons for any MJCF
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=8)
def _skeleton_cached(xml: str) -> rt.Skeleton:
    from protomotions.components.pose_lib import extract_kinematic_info
    from scipy.spatial.transform import Rotation

    kin = extract_kinematic_info(xml)
    names = list(kin.body_names)
    if names != mjcf_body_names(xml):
        raise ValueError(f"{xml}: pose_lib body order differs from mjcf_body_names")
    if not torch.allclose(kin.local_rot_ref_mat.double(), torch.eye(3, dtype=F64).expand(len(names), 3, 3)):
        raise ValueError(f"{xml}: rotated bodies; the FK here assumes identity rest orientations")
    geoms = parse_typed_geoms(xml, names)
    body, local, radius, zone, kind = [], [], [], [], []
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    for i, b in enumerate(names):
        for g in geoms[b]:
            if g["type"] == "box":
                rg = Rotation.from_quat(g["quat"]).as_matrix()
                pts, r, k = g["center"] + (rg @ (signs * g["half"]).T).T, 0.0, "corner"
            elif g["type"] == "capsule":
                pts, r, k = np.asarray(g["seg"], float), float(g["radius"]), "cap"
            else:
                pts, r, k = np.asarray(g["center"], float)[None], float(g["radius"]), "sphere"
            for p in pts:
                body.append(i)
                local.append(p)
                radius.append(r)
                zone.append(rt.ZI[rt.BODY_ZONE[b]])
                kind.append(k)
    sk = rt.Skeleton(names, list(kin.parent_indices), kin.local_pos.to(F64), kin.dof_limits_lower.to(F64),
                     kin.dof_limits_upper.to(F64), geoms, np.array(body), np.array(local, float),
                     np.array(radius, float), np.array(zone), kind)
    sk.zone_cands = [np.nonzero(sk.cand_zone == zi)[0] for zi in range(len(ZONE_ORDER))]
    sk.zone_bodies = [[names.index(b) for b in ZONES[z]] for z in ZONE_ORDER]
    sk.xml = xml
    return sk


def skeleton_for(xml_path) -> rt.Skeleton:
    """A ``retarget.Skeleton`` (names, parents, offsets [24,3] float64, lower/upper [69] exp-map box, typed
    geoms, the candidate floor points) built from ``xml_path`` exactly as ``retarget.skeleton()`` builds it
    from ``ids.MJCF``, plus ``zone_bodies`` (per ``ZONE_ORDER``, body indices) and ``xml``. Cached per path.
    ``offsets`` come from pose_lib's float32 parse (rounding <= 3e-8 m)."""
    return _skeleton_cached(str(Path(xml_path).resolve()))


# --------------------------------------------------------------------------- #
# The MoSh fit, as plant coordinates
# --------------------------------------------------------------------------- #
def _load(stem_or_path) -> tuple[dict, str]:
    p = Path(str(stem_or_path))
    if p.suffix == ".pkl" and p.exists():
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        s2 = d["stageii_debug_details"]
        fit = dict(v_template=np.asarray(d["stagei_debug_details"]["v_template"], np.float64),
                   fullpose=np.asarray(d["fullpose"], np.float64), trans=np.asarray(d["trans"], np.float64),
                   fps=float(s2["mocap_frame_rate"]), path=p)
        return fit, p.name
    fit, status, err = hm.load_fit(str(stem_or_path))
    if fit is None:
        raise FileNotFoundError(f"{stem_or_path}: {status}: {err}")
    return fit, str(stem_or_path)


def smplx_globals(v_template: np.ndarray, fullpose: np.ndarray, mdl: hm.SMPLX | None = None):
    """``(G [T,55,3,3], J [T,55,3])``: every SMPL-X joint's global rotation and posed position relative to
    ``trans`` (the pelvis joint sits at ``J_rest[0]``). ``fullpose`` as stored (hand pose absolute)."""
    mdl = hm.model() if mdl is None else mdl
    T = fullpose.shape[0]
    R = hm.rodrigues(fullpose.reshape(T, 55, 3))
    Jr = mdl.J_regressor @ v_template
    G = np.empty((T, 55, 3, 3))
    Jp = np.empty((T, 55, 3))
    for i in range(55):
        p = mdl.parents[i]
        if p < 0:
            G[:, i], Jp[:, i] = R[:, i], Jr[i]
        else:
            G[:, i] = G[:, p] @ R[:, i]
            Jp[:, i] = Jp[:, p] + G[:, p] @ (Jr[i] - Jr[p])
    return G, Jp


def _project_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.ones(M.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def body_rotations(G: np.ndarray, names: list, hand: str = "fingers") -> np.ndarray:
    """``[T,B,3,3]`` avatar body world rotations ``G_j @ AXES`` in ``names`` order (see module doc)."""
    if hand not in HAND_MODES:
        raise ValueError(f"hand must be one of {HAND_MODES}")
    out = np.empty(G.shape[:1] + (len(names), 3, 3))
    for i, b in enumerate(names):
        if b in rt.BODY_JOINT:
            out[:, i] = G[:, rt.BODY_JOINT[b]] @ rt.AXES
        elif b in rt.FINGER_BASES:
            if hand == "wrist":
                out[:, i] = G[:, rt.BODY_JOINT[b.replace("Hand", "Wrist")]] @ rt.AXES
            else:
                out[:, i] = _project_so3(G[:, list(rt.FINGER_BASES[b])].mean(1)) @ rt.AXES
        else:
            raise KeyError(f"no SMPL-X joint for body {b}")
    return out


def rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def mosh_kinematics(stem_or_path, hand: str = "fingers", register: bool = False, representative: str = "principal",
                    xml_path=None, frames=None) -> dict:
    """The clip's MoSh++ SMPL-X female fit as plant coordinates.

    ``stem_or_path``: a project clip stem (``ids.mosh_path``) or a ``*_stageii.pkl`` path.
    Returns a dict:
      ``stem``, ``fps`` (float), ``hand``, ``body_names`` [24] (COMMON order),
      ``root_pos`` [T,3] clip frame (m), ``root_rot`` [T,3,3], ``root_quat`` [T,4] xyzw,
      ``dof`` [T,69] exp-map in the plant's joint order, ``body_rot`` [T,24,3,3] (the target world rotations),
      ``human_joints`` [T,55,3] posed SMPL-X joints in the clip frame, ``shift`` [3] (clip = LBS + trans +
      shift), ``trans`` [T,3], ``frames`` [T] source frame indices.
    ``representative``: ``"principal"`` (|v| <= pi) or ``"nearest"`` (``retarget.nearest_representative``
    against ``xml_path``'s joint box, default the shipped plant).
    ``frames``: optional index array/slice to subset the clip.
    """
    fit, stem = _load(stem_or_path)
    if frames is not None:
        idx = np.arange(fit["fullpose"].shape[0])[frames]
        fit = {**fit, "fullpose": fit["fullpose"][idx], "trans": fit["trans"][idx]}
    else:
        idx = np.arange(fit["fullpose"].shape[0])
    mdl = hm.model()
    G, Jp = smplx_globals(fit["v_template"], fit["fullpose"], mdl)
    if register:
        full, _, _ = hm.load_fit(stem)
        offset, v = hm.register(mdl, full)
        if not v["ok"]:
            raise ValueError(f"{stem}: registration failed {v}")
        shift = offset + np.array([*capture.VICON_TO_CLIP_XY, 0.0])
    else:
        shift = CLIP_SHIFT.copy()
    joints = Jp + fit["trans"][:, None] + shift
    sk = skeleton_for(SHIPPED_XML if xml_path is None else xml_path)
    names = sk.names
    Rb = body_rotations(G, names, hand)
    T = Rb.shape[0]
    local = np.einsum("tbji,tbjk->tbik", Rb[:, sk.parents[1:]], Rb[:, 1:])      # R_parent^T R_child
    dof = rotvec(local).reshape(T, -1)
    if representative == "nearest":
        dof = rt.nearest_representative(torch.as_tensor(dof), sk.lower, sk.upper).numpy()
    elif representative != "principal":
        raise ValueError("representative must be 'principal' or 'nearest'")
    from scipy.spatial.transform import Rotation

    return {"stem": stem, "fps": float(fit["fps"]), "hand": hand, "body_names": list(names),
            "root_pos": joints[:, 0].copy(), "root_rot": Rb[:, 0].copy(),
            "root_quat": Rotation.from_matrix(Rb[:, 0]).as_quat(), "dof": dof, "body_rot": Rb,
            "human_joints": joints, "shift": shift, "trans": fit["trans"], "frames": idx}


def with_wrist_hands(kin: dict) -> dict:
    """``kin`` (any hand mode) re-expressed with ``hand="wrist"``: the hand bodies take their wrist's world
    rotation and their local dof is 0; everything else unchanged (the shipped ``.motion`` convention)."""
    names = kin["body_names"]
    br, dof = kin["body_rot"].copy(), kin["dof"].copy()
    for h in ("L_Hand", "R_Hand"):
        hi = names.index(h)
        br[:, hi] = br[:, names.index(h.replace("Hand", "Wrist"))]
        dof[:, 3 * (hi - 1):3 * hi] = 0.0
    return {**kin, "body_rot": br, "dof": dof, "hand": "wrist"}


# --------------------------------------------------------------------------- #
# Forward kinematics and floor heights on any plant
# --------------------------------------------------------------------------- #
def fk_bodies(sk: rt.Skeleton, kin: dict, dof=None) -> tuple[np.ndarray, np.ndarray]:
    """``(pos [T,24,3], rot [T,24,3,3])`` body origins and world rotations (float64 numpy) from
    ``kin["root_pos"], kin["root_rot"]`` and ``dof`` (default ``kin["dof"]``) through ``retarget.fk``."""
    dof = kin["dof"] if dof is None else dof
    with torch.no_grad():
        pos, rot = rt.fk(sk, torch.as_tensor(kin["root_pos"], dtype=F64), torch.as_tensor(kin["root_rot"], dtype=F64),
                         torch.as_tensor(np.asarray(dof), dtype=F64))
    return pos.numpy(), rot.numpy()


def body_joint_targets(kin: dict, names: list | None = None) -> np.ndarray:
    """``[T,24,3]`` the SMPL-X point every body origin should sit on: ``human_joints[BODY_JOINT]``, the hand
    bodies at the mean of their four finger bases."""
    names = kin["body_names"] if names is None else names
    J = kin["human_joints"]
    return np.stack([J[:, list(rt.FINGER_BASES[b])].mean(1) if b in rt.FINGER_BASES else J[:, rt.BODY_JOINT[b]]
                     for b in names], 1)


def _quat_xyzw(rot: np.ndarray) -> torch.Tensor:
    from protomotions.utils.rotations import matrix_to_quaternion

    return matrix_to_quaternion(torch.as_tensor(rot, dtype=F64), w_last=True)


def body_lowest(sk: rt.Skeleton, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``[T,24]`` lowest collision surface of every body above z = 0 (m): ``contact_geometry``'s
    ``geom_to_world`` + ``geom_ground_distance`` (the kernels behind ``capture.body_min_z``), in float64,
    with ``sk``'s own geoms."""
    pos_t = torch.as_tensor(pos, dtype=F64)
    q = _quat_xyzw(rot)
    per_body = []
    for i, b in enumerate(sk.names):
        gaps = [geom_ground_distance(geom_to_world(g, pos_t[:, i], q[:, i]))[0] for g in sk.geoms[b]]
        per_body.append(torch.stack(gaps, -1).min(-1).values)
    return torch.stack(per_body, -1).numpy()


def zone_lowest(sk: rt.Skeleton, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``[T,15]`` lowest collider point per zone (``ZONE_ORDER``) above the floor, the store's
    ``avatar_min_z`` rule (min over the zone's bodies of ``body_lowest``)."""
    per_body = body_lowest(sk, pos, rot)
    zb = getattr(sk, "zone_bodies", None) or [[sk.names.index(b) for b in ZONES[z]] for z in ZONE_ORDER]
    return np.stack([per_body[:, idx].min(-1) for idx in zb], -1)


def rest_pose(sk: rt.Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """``(pos [1,24,3], rot [1,24,3,3])`` of the plant at rest (identity rotations, pelvis at the origin)."""
    return fk_bodies(sk, {"root_pos": np.zeros((1, 3)), "root_rot": np.eye(3)[None], "dof": np.zeros((1, 69))})


def colliding_pairs(sk: rt.Skeleton) -> np.ndarray:
    """``[P,2]`` every body pair PhysX collides on this plant: all but a parent and its child (253)."""
    n = sk.num_bodies
    return np.array([(a, b) for a in range(n) for b in range(a + 1, n)
                     if sk.parents[b] != a and sk.parents[a] != b])


def pair_gaps(sk: rt.Skeleton, pos: np.ndarray, rot: np.ndarray, pairs=None) -> np.ndarray:
    """``[T,P]`` signed surface gap of every pair (default ``colliding_pairs``), ``retarget.body_gaps``
    kernels (negative: overlap)."""
    pairs = colliding_pairs(sk) if pairs is None else np.asarray(pairs)
    T = pos.shape[0]
    f = np.repeat(np.arange(T), len(pairs))
    a = np.tile(pairs[:, 0], T)
    b = np.tile(pairs[:, 1], T)
    g = rt.body_gaps(sk, torch.as_tensor(pos, dtype=F64), torch.as_tensor(rot, dtype=F64), f, a, b)
    return g.numpy().reshape(T, len(pairs))


def mass_model_for(flat_xml) -> tuple[np.ndarray, np.ndarray]:
    """``(mass [24], body-frame COM [24,3])`` of a plant's MuJoCo model (COMMON order), like
    ``retarget.mass_model``."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(flat_xml))
    return m.body_mass[1:].copy(), m.body_ipos[1:].copy()


# --------------------------------------------------------------------------- #
# Running the repo's statics / capture code on another plant, in process only
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def use_plant(xml_path=V2_XML, flat_path=V2_FLAT):
    """Temporarily point the repo's plant-dependent module state at another plant, in this process only
    (no file is edited): ``ids.MJCF``; ``static_hold_lp``'s ``FLAT, MJCF, M, D, BODY, PARENT, TYPED,
    TAU_MAX, JNAMES``; ``statics.N_HINGE / JOINT_RANGE`` (copied at its import, and the joint box can differ
    between plants); and the lru caches of ``retarget.skeleton / plant_pairs / ancestors / mass_model``,
    ``capture.skeleton`` and (if imported) ``witness.plant``. Restored on exit. ``plant.PARENT`` is not
    redirected (the kinematic tree is the same on every plant here). Check anything else your caller reads."""
    import mujoco
    import static_hold_lp as S

    xml, flat = Path(xml_path).resolve(), Path(flat_path).resolve()
    saved_ids = ids.MJCF
    names = ("FLAT", "MJCF", "M", "D", "BODY", "PARENT", "TYPED", "TAU_MAX", "JNAMES")
    saved_S = {k: getattr(S, k) for k in names}
    st = sys.modules.get("reference_curation.statics")
    saved_st = {k: getattr(st, k) for k in ("N_HINGE", "JOINT_RANGE")} if st is not None else {}

    def clear():
        caches = [rt.skeleton, rt.plant_pairs, rt.ancestors, rt.mass_model, capture.skeleton]
        w = sys.modules.get("reference_curation.witness")
        if w is not None:
            caches.append(w.plant)
        for c in caches:
            c.cache_clear()

    try:
        M = mujoco.MjModel.from_xml_path(str(flat))
        S.FLAT, S.MJCF, S.M, S.D = flat, xml, M, mujoco.MjData(M)
        S.BODY = [M.body(i).name for i in range(1, M.nbody)]
        S.PARENT = [M.body_parentid[i] - 1 for i in range(1, M.nbody)]
        S.TYPED = parse_typed_geoms(str(xml), mjcf_body_names(str(xml)))
        S.TAU_MAX = np.array([M.jnt_actfrcrange[j][1] for j in range(1, M.njnt)])
        S.JNAMES = [M.joint(j).name for j in range(1, M.njnt)]
        if st is not None:
            st.N_HINGE, st.JOINT_RANGE = M.njnt - 1, M.jnt_range[1:].copy()
        ids.MJCF = xml
        clear()
        yield
    finally:
        ids.MJCF = saved_ids
        for k, v in saved_S.items():
            setattr(S, k, v)
        st = sys.modules.get("reference_curation.statics")
        if st is not None:        # also when statics was first imported inside the context
            st.N_HINGE, st.JOINT_RANGE = saved_st.get("N_HINGE", S.M.njnt - 1), saved_st.get(
                "JOINT_RANGE", S.M.jnt_range[1:].copy())
        clear()
