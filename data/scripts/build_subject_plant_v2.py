# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build plant v2 (BodyFix Step 1): the MOYO performer's skeleton and masses, commensurate colliders.

Writes ``data/assets/smpl/smpl_yogi03596_v2.xml`` and ``_v2_flat.xml`` by text surgery on the shipped pair
(``smpl_yogi03596_lowtorque{,_flat}.xml``), the same edits on both files, plus
``smpl_yogi03596_v2_limits.json`` (every changed joint range and torque limit) next to them and the record
``data/reference_curation/plant_v2/plant_v2.json``. ``build_subject_skeleton.py`` is not touched. What changes
and why (``expert_revist/reference_curation_review_2026_09_28/BodyFix.MD`` §1-§2, Step 1's card):

**Skeleton.** Every body ``pos`` is her own joint: the SMPL-X female ``J_regressor`` applied to her personal
template (``stagei_debug_details.v_template``, byte-identical in all 336 readable fits), ``pos = AXES.T
(J_child - J_parent)`` with identity rest orientations; the hand body's origin is the mean of the four finger
bases (``retarget.FINGER_BASES``), the toes SMPL-X joints 10/11. FK of this plant reproduces her joints, so the
female fit replays on it exactly. Joint names, axes and actuators are the shipped ones.

**Colliders: the shipped primitives and sizes, placed on her segments; nothing is fitted to her skin.**

* Capsules (thigh, shin, upper arm, forearm): the shipped ``fromto`` carried by the similarity that maps the
  shipped bone onto hers (minimal rotation times length ratio), radius unchanged.
* Trunk and thorax spheres ride her joints at their shipped body-local offsets (``TRUNK_NEIGHBOURS`` are
  checked: no new rest overlap, no gap wider than the shipped plant's).
* Head sphere: the shipped radius, re-centred on her cranium: the least-squares centre of a sphere of that
  radius through the area-weighted head skin above the head joint (``CRANIUM_MIN_Z``; the subject-body
  check's cranium fit, radius held), so it reaches her crown.
* Neck sphere: nudged (the card's exception for a gap between neighbouring spheres). Riding her neck joint at
  its shipped offset it leaves a 5.0 cm gap to the re-centred head sphere (her neck is 6.3 cm longer and the
  head sphere now sits on the cranium, where the shipped pair overlapped by 6.8 cm), and no placement of a
  sphere of its size closes both that gap and the one to the chest sphere. It sits on the line between the
  chest and head sphere centres where its two gaps are equal (3.0 cm each), the smallest largest hole in the
  column its size allows. At the shipped offset it also meets her more medial collar spheres: 4,763 deep
  overlap pair-frames with the thorax on the corpus, 528 once nudged.
* Foot and toe boxes: shipped sizes and orientation; the foot box moved down so its bottom face lies at her
  sole (the lowest plantar vertex below her ankle joint: 6.81 / 7.55 cm L/R against the shipped box's
  5.29 cm); the toe box stays where the shipped plant puts it relative to the ankle, moved down with the
  foot box, so the two stay flush (``pos = pos_shipped + toe_offset_shipped - toe_offset_v2 - drop``).
* Wrist (palm) and hand boxes: shipped sizes; her palm plane is the area-weighted least-squares plane of the
  wrist+hand skin facing within ``PALM_CONE_DEG`` of down, in the wrist frame. The hand box is rotated from
  the shipped 18.8 deg tilt (14.4 deg off her palm) to that plane (the minimal rotation of the body-frame
  bottom normal onto the plane normal) and, like the wrist box, placed with its bottom face on the plane;
  both keep their shipped in-plane position relative to the wrist.

**Masses.** Her template voxelised at ``VOXEL_PITCH`` (surface voxels half-weighted) at the uniform density
that gives ``TARGET_MASS``; a voxel belongs to the body of the dominant skinning joint of its ``VOXEL_K``
nearest vertices, then anatomical cut planes at the hip, knee, shoulder and elbow re-assign the tissue
between the two segments a joint separates (``anatomical_partition``). Each geom's density = its body's
mass / its volume, so MuJoCo and PhysX derive the same mass, COM (the collider centroid) and inertia (the
collider shape) from the primitive. ``DE_LEVA_FEMALE`` cross-checks every segment; the inertia is compared
with her voxel segments.

**Joint box and torque limits** (PhysX's hard box on the exp-map coordinates, ``retarget.Skeleton``):
``LIMIT_DECISIONS`` classifies every coordinate side that her female-fit motion passes by more than
``LIMIT_REPORT_DEG`` on at least ``LIMIT_SHARE_MIN`` of the frames (57 clips: the two lotus clips and the
unreadable Standing big toe hold -c are dropped, TODO B1/B2), with the swing-twist decomposition as the
evidence. A widened side moves to her ``LIMIT_QUANTILE`` rounded outward to ``LIMIT_ROUND_DEG``, taking the
larger of the left joint and the mirrored right one (the plant stays mirror-symmetric); a kept side is the
retarget's to clip. Knee flexion is unchanged (card). ``TORQUE_DECISIONS`` raises the wrist and hand limits 50 %
(the user's decision at Step 2, a safety margin over what the arm balances' gated statics on this plant need,
``reference_curation.plant_v2``); the actuator ``gear`` follows ``actuatorfrcrange``, as on the shipped plant.

**PD gains** live in the robot config; the joint-space inertia at rest is compared with the shipped plant's
and a joint group whose inertia moves by more than ``PD_INERTIA_RATIO`` is recorded for Step 2.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python data/scripts/build_subject_plant_v2.py [--workers 8]
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "data" / "scripts"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reference_curation import human_mesh as hm  # noqa: E402
from reference_curation import ids  # noqa: E402
from reference_curation import retarget as rt  # noqa: E402

SCHEMA_VERSION = 1
MODULE = "build_subject_plant_v2"
SRC = REPO / "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
SRC_FLAT = REPO / "data/assets/smpl/smpl_yogi03596_lowtorque_flat.xml"
OUT = REPO / "data/assets/smpl/smpl_yogi03596_v2.xml"
OUT_FLAT = REPO / "data/assets/smpl/smpl_yogi03596_v2_flat.xml"
LIMITS_JSON = REPO / "data/assets/smpl/smpl_yogi03596_v2_limits.json"
RECORD_DIR = ids.DATA_ROOT / "plant_v2"
RECORD = RECORD_DIR / "plant_v2.json"
SRC_MODEL_NAME = "smpl_yogi03596_lowtorque"
MODEL_NAME = "smpl_yogi03596_v2"

TEMPLATE_STEM = "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a"   # any fit: one template
TEMPLATE_SHA1 = "9f7d171132caf5fce695e07787d8fda14b1d4524"                         # BodyFix §1
TARGET_MASS = 74.0
# Clips left out of the release (TODO B1: the plant's knee cannot fold a lotus; B2: no readable fit).
DROPPED_STEMS = ("220923_Cockerel_Pose-b", "220923_Scale_Pose_or_Tolasana_-a",
                 "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c")

# --------------------------------------------------------------------------- #
# Collider rules
# --------------------------------------------------------------------------- #
CAPSULE_CHILD = {"L_Hip": "L_Knee", "L_Knee": "L_Ankle", "L_Shoulder": "L_Elbow", "L_Elbow": "L_Wrist",
                 "R_Hip": "R_Knee", "R_Knee": "R_Ankle", "R_Shoulder": "R_Elbow", "R_Elbow": "R_Wrist"}
RIDING_SPHERES = ("Pelvis", "Torso", "Spine", "Chest", "L_Thorax", "R_Thorax")
TRUNK_NEIGHBOURS = (("Pelvis", "Torso"), ("Torso", "Spine"), ("Spine", "Chest"), ("Chest", "L_Thorax"),
                    ("Chest", "R_Thorax"))
CRANIUM_MIN_Z = 0.02         # head-frame z above which the head skin is cranium (subject-body check's fit)
PALM_CONE_DEG = 30.0         # palm skin: vertex normals within this of straight down (wrist frame)
CROWN_TOL_M = 0.005          # the card: her crown within 5 mm of the head sphere's surface

# --------------------------------------------------------------------------- #
# Mass rules
# --------------------------------------------------------------------------- #
VOXEL_PITCH = 0.003
VOXEL_K = 8
# de Leva (1996), Table 4, female segment masses in % of body mass (Zatsiorsky-Seluyanov, adjusted).
DE_LEVA_FEMALE = {"head_neck": 6.68, "trunk": 42.57, "upper_arm": 2.55, "forearm": 1.38, "hand": 0.56,
                  "thigh": 14.78, "shank": 4.81, "foot": 1.29}
SEGMENTS = {"head_neck": ("Neck", "Head"), "trunk": ("Pelvis", "Torso", "Spine", "Chest", "L_Thorax", "R_Thorax"),
            **{f"{s}_{k}": v for s in "LR" for k, v in (
                ("upper_arm", (f"{s}_Shoulder",)), ("forearm", (f"{s}_Elbow",)), ("hand", (f"{s}_Wrist", f"{s}_Hand")),
                ("thigh", (f"{s}_Hip",)), ("shank", (f"{s}_Knee",)), ("foot", (f"{s}_Ankle", f"{s}_Toe")))}}
DE_LEVA_TOL = 0.20
# Why a segment is more than DE_LEVA_TOL off de Leva (the card: record it with the reason).
DE_LEVA_NOTES = {
    "upper_arm": "the shoulder plane (through the glenohumeral centre, perpendicular to the humerus) puts on the arm "
                 "the tissue SMPL-X skins to the collar that lies lateral of the joint, 0.57 / 0.44 kg (L/R), 71 % of "
                 "it below the joint centre: the proximal underside of the arm at the axilla. Without it the upper arm "
                 "is 1.95 / 1.98 kg (+3 / +5 %); de Leva's boundary (acromion to axilla) leaves that tissue on the "
                 "trunk, and tilting the plane 45 deg toward it still gives 2.33 / 2.22 kg. It moves 0.5 kg between two "
                 "bodies 11 cm apart (about 0.3 N m of shoulder gravity moment against its 150 N m limit).",
}

# --------------------------------------------------------------------------- #
# Joint box and torque limits
# --------------------------------------------------------------------------- #
LIMIT_REPORT_DEG = 2.0       # a frame passes a bound when it is more than this beyond it
LIMIT_SHARE_MIN = 0.01       # a coordinate side needs a decision when at least this share of frames passes it
LIMIT_QUANTILE = 0.5         # % of her frames a widened side may still clip (p0.5 / p99.5)
LIMIT_ROUND_DEG = 5.0
# One entry per coordinate side of the LEFT (or centre) body; the right body gets the mirror (axis y keeps
# its side, x and z swap sides and sign). The swing-twist evidence behind each is recomputed and recorded.
LIMIT_DECISIONS = (
    {"joint": "L_Thorax", "axis": "z", "side": "lo", "verdict": "widen", "kind": "anatomical",
     "why": "SMPL-X's collar joint carries the whole shoulder girdle (clavicle and scapula): on the frames past the "
            "box the collar protracts (swing about z p50 -21, p1 -38 deg) while it rotates about its own bone "
            "(twist p50 -43, p1 -58 deg), and the exp-map z coordinate carries their sum; the arms reach forward "
            "or overhead on 40 % of the corpus"},
    {"joint": "L_Thorax", "axis": "x", "side": "hi", "verdict": "widen", "kind": "anatomical",
     "why": "shoulder-girdle elevation (arms overhead, shrug): the elevation swing on the frames past the box is "
            "p50 27 / 35 deg (L / R), p99 38 / 48; the shipped box allows 30 deg of elevation but 40 of depression"},
    {"joint": "L_Thorax", "axis": "y", "side": "lo", "verdict": "widen", "kind": "anatomical",
     "why": "the collar bone's axial rotation during arm elevation (twist about it p50 -54 deg on the frames past "
            "the box), which the SMPL-X collar joint carries for the whole girdle"},
    {"joint": "L_Elbow", "axis": "x", "side": "hi", "verdict": "widen", "kind": "representation",
     "why": "SMPL-X leaves part of the humerus's axial rotation at the elbow, so a flexion appears in a rotated "
            "plane: on these frames the elbow flexes (swing magnitude p50 80, p99 118 deg, within 150) in a plane "
            "turned about 58 deg from the hinge's (swing about x p50 66, about z -40); the upper-arm capsule is "
            "axisymmetric, so moving that rotation between shoulder and elbow changes no collider"},
    {"joint": "L_Elbow", "axis": "y", "side": "lo", "verdict": "widen", "kind": "representation",
     "why": "forearm pronation / supination (twist about the forearm, p50 -44 / +55 deg L / R on these frames) is "
            "stored at the elbow in SMPL-X; the shipped +-30 deg is a third of the anatomical +-80-90"},
    {"joint": "L_Elbow", "axis": "y", "side": "hi", "verdict": "widen", "kind": "representation",
     "why": "as the other side: forearm pronation / supination stored at the elbow (twist p50 +56 / -53 deg)"},
    {"joint": "L_Elbow", "axis": "z", "side": "hi", "verdict": "widen", "kind": "anatomical",
     "why": "elbow hyperextension of the straight, loaded arm (plank, downward dog, arm balances), as in hypermobile "
            "adults: the flexion swing is past 0 on these frames (p50 10 / 5 deg L / R, p99 19 / 17) with a "
            "carrying-angle swing of about 13 deg"},
    {"joint": "L_Ankle", "axis": "y", "side": "hi", "verdict": "widen", "kind": "anatomical",
     "why": "plantarflexion of the whole foot segment (ankle plus midfoot in SMPL-X) with pointed feet in inversions "
            "and lying poses: the plantarflexion swing on these frames is p50 64 / 60 deg (L / R), p99 73 / 64"},
    {"joint": "L_Ankle", "axis": "x", "side": "lo", "verdict": "keep", "kind": "fit",
     "why": "the mirror of the right ankle's inversion (twist p50 31, p99 47 deg on 3.9 % of the frames), which the "
            "left foot never needs (0.2 %): the fit's right-foot supination (BodyFix §1 item 2), not her anatomy; "
            "the retarget corrects the foot orientation"},
    {"joint": "L_Knee", "axis": "x", "side": "lo", "verdict": "widen", "kind": "representation",
     "why": "as the elbow: part of the femur's axial rotation sits at the knee, so a deep flexion (swing magnitude "
            "p50 136, p99 156 deg, within the 160 stop) appears in a rotated plane (swing about x p50 -64 deg); the "
            "thigh capsule is axisymmetric"},
    {"joint": "L_Knee", "axis": "z", "side": "lo", "verdict": "widen", "kind": "anatomical",
     "why": "tibial axial rotation of the flexed knee: twist about the shank p50 34 / 38 deg (L / R) on these frames, "
            "within the anatomical 40"},
    {"joint": "L_Knee", "axis": "z", "side": "hi", "verdict": "widen", "kind": "anatomical",
     "why": "the mirror of the right knee's tibial rotation (twist about the shank p50 38 deg)"},
    {"joint": "L_Knee", "axis": "y", "side": "lo", "verdict": "widen", "kind": "anatomical",
     "why": "the extension stop, not the flexion stop the card keeps at 160 deg (the lotus clips are dropped): her "
            "locked standing knees hyperextend (flexion swing p50 -4 / -2 deg L / R, p1 -8 / -7), as in hypermobile "
            "adults, on 50 of the 300 hold exemplars (right knee)"},
    {"joint": "L_Toe", "axis": "x", "side": "hi", "verdict": "keep", "kind": "fit",
     "why": "the mirror of the right toe's roll about the foot axis (twist p50 -40, p1 -71 deg on 26.5 % of the "
            "frames; the left toe 0.5 %): a metatarsophalangeal joint has no roll, so it is the fit's right-foot "
            "supination compensated at the toe"},
    {"joint": "Neck", "axis": "y", "side": "lo", "verdict": "widen", "kind": "anatomical",
     "why": "cervical extension in back bends (cobra, bridge, upward plank): the extension swing on these frames is "
            "p50 -46, p1 -62 deg"},
    {"joint": "L_Hand", "axis": "x", "side": "lo", "verdict": "widen", "kind": "anatomical",
     "why": "finger flexion at the metacarpophalangeal joints in grips (big-toe hold, dancer): the hand body takes "
            "the mean rotation of the four finger bases, flexed p50 103 deg on these frames (the ring and little "
            "fingers reach 100-110)"},
)
# Wrist and hand torque limits (N m), decided from the arm balances' gated statics on this plant
# (``reference_curation.plant_v2``: ``acceptance.statics.<arm>.torque_sweep`` in plant_v2.json): kept.
TORQUE_SCALE_WRIST_HAND = 1.5
TORQUE_DECISIONS = {f"{s}_{j}_{a}": TORQUE_SCALE_WRIST_HAND * base for s in "LR" for j, base in (("Wrist", 20.0), ("Hand", 10.0))
                    for a in "xyz"}
TORQUE_NOTE = ("wrist 20 -> 30, hand 10 -> 15 N m: raised 50 % on the user's decision (2026-09-30, BodyFix Step 2) as a "
               "safety margin for subject-sized arm balances. The statics at the shipped 20 / 10 had already found "
               "them rarely binding: on plant v2 driven by her own motion (root at her pelvis) 38 of the 41 holds "
               "supported by hands, forearms or the head alone were LP-holdable within them (s* p50 about 0.4) and "
               "one (Scorpion -b@945) needed both limits scaled by 1.73; grounded per frame (+0.5 cm / 0 cm) they "
               "bound on 2 / 5 LP-optimal holds, cleared at 1.18-1.90x (Cockerel -b, a dropped lotus clip, among "
               "them). The torque sweep in the acceptance is now relative to the raised limits.")
PD_INERTIA_RATIO = 1.5       # BodyFix: keep the gains unless a joint's joint-space inertia moves by more


# --------------------------------------------------------------------------- #
# Her template, skin and joints
# --------------------------------------------------------------------------- #
def load_template():
    """``(mdl, v_template [10475,3], J [55,3])`` in SMPL-X canonical axes; asserts the template identity."""
    mdl = hm.model()
    fit, status, err = hm.load_fit(TEMPLATE_STEM)
    if fit is None:
        raise FileNotFoundError(f"{TEMPLATE_STEM}: {status}: {err}")
    vt = np.asarray(fit["v_template"], np.float64)
    sha1 = hashlib.sha1(np.ascontiguousarray(vt).tobytes()).hexdigest()
    if sha1 != TEMPLATE_SHA1:
        raise ValueError(f"template sha1 {sha1} is not the performer's ({TEMPLATE_SHA1})")
    return mdl, vt, mdl.J_regressor @ vt


def skin_arrays(mdl, vt):
    """``(skin mask, vertex area, vertex normal in avatar axes)`` of the main mesh component (the eyeball
    shells, which sit inside the head, are left out)."""
    import trimesh

    full = trimesh.Trimesh(vt, mdl.faces, process=False)
    lab = trimesh.graph.connected_component_labels(full.face_adjacency, node_count=len(mdl.faces))
    Fm = mdl.faces[lab == np.argmax(np.bincount(lab))]
    skin = np.zeros(len(vt), bool)
    skin[np.unique(Fm)] = True
    fa = 0.5 * np.linalg.norm(np.cross(vt[Fm[:, 1]] - vt[Fm[:, 0]], vt[Fm[:, 2]] - vt[Fm[:, 0]]), axis=1)
    area = np.zeros(len(vt))
    for i in range(3):
        np.add.at(area, Fm[:, i], fa / 3)
    return skin, area, rt.vertex_normals(vt, Fm) @ rt.AXES


def body_frames(J, names, parents):
    """``(O {body: origin, canonical}, offsets {body: pos in parent frame, avatar axes})``."""
    O = {b: rt._origin(J, b) for b in names}
    off = {b: (O[b] - O[names[parents[i]]]) @ rt.AXES for i, b in enumerate(names) if i > 0}
    return O, off


def local(O, body, v):
    """Canonical SMPL-X points in ``body``'s avatar frame (origin at her joint)."""
    return (v - O[body]) @ rt.AXES


# --------------------------------------------------------------------------- #
# Colliders
# --------------------------------------------------------------------------- #
def min_rotation(a, b):
    """The smallest rotation taking direction ``a`` to direction ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def plane_fit(p, w):
    """Area-weighted least-squares plane ``(centroid, unit normal)``."""
    c = np.average(p, 0, w)
    _, _, vh = np.linalg.svd((p - c) * np.sqrt(w / w.sum())[:, None], full_matrices=False)
    return c, vh[-1]


def shipped_geoms(names):
    """The shipped plant's one geom per body, body-local, float64 (``parse_typed_geoms`` layout)."""
    from contact_geometry import parse_typed_geoms

    out = {}
    for b, gs in parse_typed_geoms(str(SRC), names).items():
        if len(gs) != 1:
            raise ValueError(f"{b}: {len(gs)} geoms; the pipeline assumes one per body")
        g = {k: (np.asarray(v, np.float64) if isinstance(v, np.ndarray) else v) for k, v in gs[0].items() if k != "mass"}
        out[b] = g
    return out


def head_sphere(mdl, vt, O, skin, area, radius):
    """The centre (head frame) of the sphere of ``radius`` closest in the area-weighted least-squares sense to
    her cranium skin (head skin above ``CRANIUM_MIN_Z``), and the landmarks it is checked on."""
    from scipy.optimize import least_squares

    body_of = np.array([rt.JOINT_BODY[j] for j in mdl.part])
    m = (body_of == "Head") & skin
    p, w = local(O, "Head", vt[m]), area[m]
    cran = p[:, 2] > CRANIUM_MIN_Z
    sw = np.sqrt(w[cran] / w[cran].sum())
    c = least_squares(lambda x: sw * (np.linalg.norm(p[cran] - x, axis=1) - radius), np.average(p[cran], 0, w[cran])).x
    crown = p[p[:, 2].argmax()]
    back = p[p[:, 0].argmin()]
    chin_pts = p[np.isin(mdl.part[m], [22])]
    chin = chin_pts[(chin_pts[:, 0] - 0.5 * chin_pts[:, 2]).argmax()]
    d = lambda q: float(np.linalg.norm(q - c) - radius)  # noqa: E731
    return c, {"cranium_vertices": int(cran.sum()), "crown_head_frame_m": crown.tolist(),
               "crown_minus_surface_m": d(crown), "back_of_head_minus_surface_m": d(back),
               "chin_minus_surface_m": d(chin), "sphere_top_minus_crown_z_m": float(c[2] + radius - crown[2])}


def sole_depth(mdl, vt, O, skin, side):
    """Her ankle joint's height above her sole: the lowest plantar vertex of foot + toes, ankle frame."""
    body_of = np.array([rt.JOINT_BODY[j] for j in mdl.part])
    m = np.isin(body_of, [f"{side}_Ankle", f"{side}_Toe"]) & skin
    p = local(O, f"{side}_Ankle", vt[m])
    k = int(p[:, 2].argmin())
    return float(-p[k, 2]), p[k]


def palm_plane(mdl, vt, O, skin, area, nrm, side):
    """Her palm plane in the wrist frame: ``(point, unit normal pointing out of the palm)``."""
    body_of = np.array([rt.JOINT_BODY[j] for j in mdl.part])
    m = np.isin(body_of, [f"{side}_Wrist", f"{side}_Hand"]) & skin
    p, n, w = local(O, f"{side}_Wrist", vt[m]), nrm[m], area[m]
    down = np.array([0.0, 0.0, -1.0])
    sel = n @ down >= math.cos(math.radians(PALM_CONE_DEG))
    c, nn = plane_fit(p[sel], w[sel])
    return c, (nn if nn @ down > 0 else -nn), int(sel.sum())


def build_colliders(mdl, vt, J, names, parents, skin, area, nrm):
    """``(geoms {body: typed geom, body-local}, record {body: rule and numbers})`` of plant v2."""
    from scipy.spatial.transform import Rotation

    O, off = body_frames(J, names, parents)
    ship = shipped_geoms(names)
    from reference_curation.mosh_replay import skeleton_for

    sk0 = skeleton_for(SRC)
    off_s = {b: sk0.offsets[i].numpy().astype(np.float64) for i, b in enumerate(names) if i > 0}
    geo, rec = {}, {}
    for b in names:
        g = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ship[b].items()}
        if b in CAPSULE_CHILD:
            cs, cp = off_s[CAPSULE_CHILD[b]], off[CAPSULE_CHILD[b]]
            R = min_rotation(cs, cp)
            S = R * (np.linalg.norm(cp) / np.linalg.norm(cs))
            g["seg"] = ship[b]["seg"] @ S.T
            rec[b] = {"rule": "capsule re-spanned by the bone similarity (radius unchanged)",
                      "bone_length_ratio": float(np.linalg.norm(cp) / np.linalg.norm(cs)),
                      "bone_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                      "fromto_shipped": ship[b]["seg"].tolist(), "fromto_v2": g["seg"].tolist()}
        elif b in RIDING_SPHERES:
            rec[b] = {"rule": "sphere rides her joint at its shipped body-local offset"}
        geo[b] = g
    # head: shipped radius, re-centred on her cranium
    r_head = float(ship["Head"]["radius"])
    c, lm = head_sphere(mdl, vt, O, skin, area, r_head)
    geo["Head"]["center"] = c
    rec["Head"] = {"rule": "shipped radius, centre = fixed-radius least-squares fit to her cranium skin",
                   "radius_m": r_head, "center_head_frame_m": c.tolist(), **lm}
    # neck: on the chest-head centre line, equal gaps to both (neck frame)
    C = geo["Chest"]["center"] - off["Neck"]
    H = off["Head"] + geo["Head"]["center"]
    rc, rn, rh = float(geo["Chest"]["radius"]), float(geo["Neck"]["radius"]), r_head
    u = (H - C) / np.linalg.norm(H - C)
    t = (float(np.linalg.norm(H - C)) + rc - rh) / 2
    gap = lambda p: (float(np.linalg.norm(p - C)) - rc - rn, float(np.linalg.norm(H - p)) - rh - rn)  # noqa: E731
    shipped_neck = ship["Neck"]["center"].copy()
    geo["Neck"]["center"] = C + t * u
    rec["Neck"] = {"rule": "nudged onto the chest-head sphere centre line, equal gaps to both spheres",
                   "center_shipped_offset_m": shipped_neck.tolist(), "center_v2_m": geo["Neck"]["center"].tolist(),
                   "gaps_at_shipped_offset_m": {"chest": gap(shipped_neck)[0], "head": gap(shipped_neck)[1]},
                   "gaps_v2_m": {"chest": gap(geo["Neck"]["center"])[0], "head": gap(geo["Neck"]["center"])[1]}}
    for s in "LR":
        # feet: bottom face on her sole, toe box flush with it
        a, t = f"{s}_Ankle", f"{s}_Toe"
        depth, low_pt = sole_depth(mdl, vt, O, skin, s)
        bottom_s = float(ship[a]["center"][2] - ship[a]["half"][2])
        drop = depth + bottom_s
        geo[a]["center"] = ship[a]["center"] - np.array([0.0, 0.0, drop])
        geo[t]["center"] = ship[t]["center"] + off_s[t] - off[t] - np.array([0.0, 0.0, drop])
        toe_bottom = off[t][2] + geo[t]["center"][2] - geo[t]["half"][2]
        rec[a] = {"rule": "shipped box moved down to her sole", "ankle_above_sole_m": depth,
                  "lowest_plantar_vertex_ankle_frame_m": low_pt.tolist(), "shipped_ankle_above_box_bottom_m": -bottom_s,
                  "drop_m": drop, "center_shipped": ship[a]["center"].tolist(), "center_v2": geo[a]["center"].tolist()}
        rec[t] = {"rule": "shipped box kept in the ankle frame (flush with the foot box), moved down with it",
                  "center_shipped": ship[t]["center"].tolist(), "center_v2": geo[t]["center"].tolist(),
                  "toe_bottom_minus_foot_bottom_m": float(toe_bottom - (geo[a]["center"][2] - geo[a]["half"][2]))}
        # hands: bottom faces on her palm plane, the hand box rotated to it
        w, h = f"{s}_Wrist", f"{s}_Hand"
        pc, n, n_sel = palm_plane(mdl, vt, O, skin, area, nrm, s)
        gw = geo[w]
        cx, cy, hz = float(gw["center"][0]), float(gw["center"][1]), float(gw["half"][2])
        gw["center"] = np.array([cx, cy, hz + (n @ pc - n[0] * cx - n[1] * cy) / n[2]])
        Rs = Rotation.from_quat(ship[h]["quat"]).as_matrix()
        Rn = min_rotation(np.array([0.0, 0.0, -1.0]), n)
        cw = ship[h]["center"] + off_s[h]                       # its centre in the wrist frame (parent-anchored)
        hzz = float(ship[h]["half"][2])
        cw = np.array([cw[0], cw[1], (n @ pc - hzz - n[0] * cw[0] - n[1] * cw[1]) / n[2]])
        geo[h]["center"] = cw - off[h]
        q = Rotation.from_matrix(Rn).as_quat()
        geo[h]["quat"] = q if q[3] >= 0 else -q
        tilt_s = float(np.degrees(np.arccos(np.clip((Rs @ [0, 0, -1.0]) @ n, -1, 1))))
        tilt_w = float(np.degrees(np.arccos(np.clip(-n[2], -1, 1))))
        rec[w] = {"rule": "shipped box, bottom face centred on her palm plane", "palm_plane_point_wrist_frame_m": pc.tolist(),
                  "palm_plane_normal_out": n.tolist(), "palm_vertices": n_sel, "bottom_vs_palm_deg": tilt_w,
                  "center_shipped": ship[w]["center"].tolist(), "center_v2": gw["center"].tolist()}
        rec[h] = {"rule": "shipped box rotated to her palm plane, bottom face on it, kept in the wrist frame",
                  "shipped_bottom_vs_palm_deg": tilt_s, "shipped_rotation_deg": float(np.degrees(
                      Rotation.from_quat(ship[h]["quat"]).magnitude())),
                  "v2_rotation_deg": float(np.degrees(Rotation.from_matrix(Rn).magnitude())),
                  "center_shipped": ship[h]["center"].tolist(), "center_v2": geo[h]["center"].tolist()}
    return geo, rec, O, off


def rest_gaps(names, parents, off, geo):
    """``(pairs, gaps [P])`` of every colliding pair (all but parent-child) at rest, ``retarget.body_gaps``."""
    import torch
    from types import SimpleNamespace

    B = len(names)
    P = np.zeros((B, 3))
    for i in range(1, B):
        P[i] = P[parents[i]] + off[names[i]]
    rot = np.broadcast_to(np.eye(3), (1, B, 3, 3)).copy()
    sk = SimpleNamespace(names=names, geoms={b: [geo[b]] for b in names})
    pairs = [(a, b) for a in range(B) for b in range(a + 1, B) if parents[b] != a and parents[a] != b]
    g = rt.body_gaps(sk, torch.as_tensor(P[None]), torch.as_tensor(rot), np.zeros(len(pairs), int),
                     np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    return pairs, g.numpy(), P


def sphere_gap(P, names, geo, a, b):
    ca = P[names.index(a)] + geo[a]["center"]
    cb = P[names.index(b)] + geo[b]["center"]
    return float(np.linalg.norm(ca - cb) - geo[a]["radius"] - geo[b]["radius"])


# --------------------------------------------------------------------------- #
# Masses
# --------------------------------------------------------------------------- #
def voxelize(mdl, vt, pitch=VOXEL_PITCH):
    """``(points [N,3] canonical, volume [N] m^3)``: the closed template rasterised on a ``pitch`` grid;
    interior voxels count whole, surface voxels half (partial-volume correction)."""
    from scipy import ndimage

    lo = vt.min(0) - 2 * pitch
    shape = np.ceil((vt.max(0) + 2 * pitch - lo) / pitch).astype(int) + 1
    surf = np.zeros(shape, bool)
    F = mdl.faces
    a, b, c = vt[F[:, 0]], vt[F[:, 1]], vt[F[:, 2]]
    emax = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                            np.linalg.norm(a - c, axis=1)]), 0)
    steps = np.ceil(emax / (0.4 * pitch)).astype(int)
    for n in np.unique(steps):
        sel = steps == n
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        m = (i + j) <= n
        u, v = i[m] / n, j[m] / n
        pts = a[sel][:, None] + u[None, :, None] * (b[sel] - a[sel])[:, None] + v[None, :, None] * (c[sel] - a[sel])[:, None]
        idx = np.floor((pts.reshape(-1, 3) - lo) / pitch).astype(int)
        surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    inner = ndimage.binary_fill_holes(surf) & ~surf
    ii, ss = np.argwhere(inner), np.argwhere(surf)
    pts = np.r_[ii, ss] * pitch + lo + pitch / 2
    return pts, np.r_[np.ones(len(ii)), 0.5 * np.ones(len(ss))] * pitch ** 3


def dominant_body(mdl, vt, pts, names):
    """``[N]`` body index of each voxel: the dominant joint of the inverse-distance-weighted skinning weights
    of its ``VOXEL_K`` nearest template vertices, through ``retarget.JOINT_BODY``."""
    from scipy.spatial import cKDTree

    tree = cKDTree(vt)
    body_of_joint = np.array([names.index(rt.JOINT_BODY[j]) for j in range(55)])
    out = np.empty(len(pts), int)
    for s in range(0, len(pts), 400000):
        d, nn = tree.query(pts[s:s + 400000], k=VOXEL_K)
        w = 1.0 / np.maximum(d, 1e-4)
        out[s:s + 400000] = body_of_joint[np.einsum("nk,nkj->nj", w, mdl.weights[nn]).argmax(1)]
    return out


def anatomical_partition(pts, vol, dom, O, names):
    """``(body [N], transfers)``: the skinning partition with anatomical cut planes at the hip, knee, shoulder
    and elbow. Each plane passes through the joint centre, perpendicular to the distal bone at rest; it
    re-assigns the tissue of the two segments the joint separates:

    * leg (per side): tissue skinned to the pelvis or either leg chain that lies below the hip plane and on
      that side of the midsagittal plane (through the pelvis joint) is leg; leg-chain tissue above the hip
      plane is pelvis. Within the leg, the knee plane splits thigh from shank; the foot (ankle / toe
      skinning) is kept.
    * arm (per side): tissue skinned to the collar or the arm chain that lies lateral of the shoulder plane is
      arm, arm-chain tissue medial of it is collar (thorax body). Within the arm the elbow plane splits upper
      arm from forearm; the hand (wrist / finger skinning) is kept."""
    B = {b: i for i, b in enumerate(names)}
    X = (pts - O["Pelvis"]) @ rt.AXES
    Oa = {b: (O[b] - O["Pelvis"]) @ rt.AXES for b in names}
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    body = dom.copy()
    moved = {}
    for s in "LR":
        lat = 1.0 if s == "L" else -1.0
        other = "R" if s == "L" else "L"
        hip, knee, ankle, toe = (f"{s}_{k}" for k in ("Hip", "Knee", "Ankle", "Toe"))
        chain = [B[hip], B[knee], B[ankle], B[toe]]
        lower = chain + [B[f"{other}_{k}"] for k in ("Hip", "Knee", "Ankle", "Toe")] + [B["Pelvis"]]
        below_hip = (X - Oa[hip]) @ unit(Oa[knee] - Oa[hip]) > 0
        side = lat * X[:, 1] > 0
        leg = below_hip & side & np.isin(dom, lower)
        up = np.isin(dom, chain) & ~below_hip
        moved[f"{s}_leg"] = {"pelvis_to_leg_kg": float(vol[leg & (dom == B["Pelvis"])].sum()),
                             "other_leg_to_leg_kg": float(vol[leg & np.isin(dom, lower[4:8])].sum()),
                             "leg_to_pelvis_kg": float(vol[up].sum())}
        body[up] = B["Pelvis"]
        below_knee = (X - Oa[knee]) @ unit(Oa[ankle] - Oa[knee]) > 0
        foot = np.isin(dom, [B[ankle], B[toe]])
        body[leg & ~below_knee & ~foot] = B[hip]
        body[leg & below_knee & ~foot] = B[knee]
        th, sho, elb, wri, hand = (f"{s}_{k}" for k in ("Thorax", "Shoulder", "Elbow", "Wrist", "Hand"))
        achain = [B[sho], B[elb], B[wri], B[hand]]
        out_sho = (X - Oa[sho]) @ unit(Oa[elb] - Oa[sho]) > 0
        arm = out_sho & np.isin(dom, achain + [B[th]])
        inn = np.isin(dom, achain) & ~out_sho
        moved[f"{s}_arm"] = {"collar_to_arm_kg": float(vol[arm & (dom == B[th])].sum()),
                             "arm_to_collar_kg": float(vol[inn].sum()),
                             "trunk_lateral_of_plane_kept_kg": float(vol[out_sho & np.isin(
                                 dom, [B["Chest"], B["Spine"], B["Torso"]])].sum())}
        body[inn] = B[th]
        out_elb = (X - Oa[elb]) @ unit(Oa[wri] - Oa[elb]) > 0
        hnd = np.isin(dom, [B[wri], B[hand]])
        body[arm & ~out_elb & ~hnd] = B[sho]
        body[arm & out_elb & ~hnd] = B[elb]
    return body, moved


def segment_mass_properties(pts, vol, body, O, names, rho):
    """Per body ``{mass_kg, com_body_m (her body frame), inertia_com_kgm2 [3x3] (avatar axes)}`` and the
    whole body ``{com_pelvis_frame_m, inertia_com_kgm2}``, uniform density ``rho``."""
    out = {}
    X = (pts - O["Pelvis"]) @ rt.AXES
    for i, b in enumerate(names):
        k = body == i
        m = vol[k] * rho
        loc = (pts[k] - O[b]) @ rt.AXES
        com = (loc * m[:, None]).sum(0) / m.sum()
        r = loc - com
        I = (m[:, None, None] * ((r * r).sum(1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])).sum(0)
        out[b] = {"mass_kg": float(m.sum()), "com_body_m": com.tolist(), "inertia_com_kgm2": I.tolist()}
    m = vol * rho
    com = (X * m[:, None]).sum(0) / m.sum()
    r = X - com
    I = (m[:, None, None] * ((r * r).sum(1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])).sum(0)
    return out, {"com_pelvis_frame_m": com.tolist(), "inertia_com_kgm2": I.tolist(), "mass_kg": float(m.sum())}


def geom_volume(g) -> float:
    if g["type"] == "sphere":
        return 4 / 3 * math.pi * g["radius"] ** 3
    if g["type"] == "capsule":
        return math.pi * g["radius"] ** 2 * float(np.linalg.norm(g["seg"][1] - g["seg"][0])) + 4 / 3 * math.pi * g["radius"] ** 3
    return 8.0 * float(np.prod(g["half"]))


def de_leva_check(mass):
    rows = {}
    for seg, bodies in SEGMENTS.items():
        key = seg.split("_", 1)[1] if seg[:2] in ("L_", "R_") else seg
        ref = DE_LEVA_FEMALE[key] / 100.0 * TARGET_MASS
        m = sum(mass[b] for b in bodies)
        off = bool(abs(m / ref - 1) > DE_LEVA_TOL)
        rows[seg] = {"bodies": list(bodies), "mass_kg": m, "de_leva_kg": ref, "ratio": m / ref, "off_by_more_than_tol": off}
        if off:
            if key not in DE_LEVA_NOTES:
                raise ValueError(f"{seg} is {m / ref:.2f}x de Leva and has no recorded reason")
            rows[seg]["reason"] = DE_LEVA_NOTES[key]
    return rows


# --------------------------------------------------------------------------- #
# Joint box: her female-fit motion per exp-map coordinate
# --------------------------------------------------------------------------- #
def _clip_dof(stem):
    import torch

    torch.set_num_threads(1)
    from reference_curation import mosh_replay

    kin = mosh_replay.mosh_kinematics(stem, hand="fingers")
    return stem, kin["dof"].astype(np.float32)


def corpus_dof(workers: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(stems, dof [F,69] principal exp-map, clip index [F])`` of the female fit (hand = "fingers") over the
    manifest minus ``DROPPED_STEMS``."""
    stems = [s for s in ids.manifest_stems() if s not in DROPPED_STEMS]
    if workers > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                min(workers, 8), mp_context=multiprocessing.get_context("spawn")) as ex:
            res = list(ex.map(_clip_dof, stems))
    else:
        res = [_clip_dof(s) for s in stems]
    return stems, np.concatenate([d for _, d in res]).astype(np.float64), np.concatenate(
        [np.full(len(d), k) for k, (_, d) in enumerate(res)])


def swing_twist(rv, axis):
    """``(swing rotvec [F,3], twist [F])`` degrees: the local rotation = swing * twist, the twist about the
    distal segment's rest axis ``axis``."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_rotvec(rv).as_quat()
    p = (q[:, :3] @ axis)[:, None] * axis
    qt = np.c_[p, q[:, 3]]
    qt /= np.linalg.norm(qt, axis=1, keepdims=True)
    tw = 2 * np.arctan2(qt[:, :3] @ axis, qt[:, 3])
    tw = (tw + np.pi) % (2 * np.pi) - np.pi
    sw = (Rotation.from_quat(q) * Rotation.from_quat(qt).inv()).as_rotvec()
    return np.degrees(sw), np.degrees(tw)


def segment_axis(body, off, names, parents):
    """The distal segment's rest long axis in its own frame: toward its (first) child, or for the leaves the
    toe/finger direction (+x toes, the hand offset for fingers) and up for the head."""
    kids = [names[i] for i in range(len(names)) if parents[i] == names.index(body)]
    if body in ("L_Toe", "R_Toe"):
        v = np.array([1.0, 0.0, 0.0])
    elif body in ("L_Hand", "R_Hand"):
        v = off[body]
    elif body == "Head":
        v = np.array([0.0, 0.0, 1.0])
    else:
        kid = {"Chest": "Neck", "Pelvis": "Torso"}.get(body, kids[0] if kids else None)
        v = off[kid]
    return v / np.linalg.norm(v)


def mirror_side(joint, axis, side):
    """The right body's coordinate side that mirrors a left (or centre) body's: y keeps its side, x and z
    swap. Centre bodies mirror onto themselves."""
    j = joint.replace("L_", "R_", 1) if joint.startswith("L_") else joint
    s = side if axis == "y" else {"lo": "hi", "hi": "lo"}[side]
    return j, s


def joint_box_evidence(dof, clip, stems, names, parents, off, lo, hi):
    """Per coordinate side: the share of frames past the shipped box by more than ``LIMIT_REPORT_DEG``, the
    excess quantiles, her coordinate's quantiles and clip count, and for every joint the swing / twist
    quantiles on the frames past each side (the anatomical reading)."""
    import torch

    rep = rt.nearest_representative(torch.as_tensor(dof), torch.as_tensor(lo), torch.as_tensor(hi)).numpy()
    val = np.degrees(rep)
    lo_d, hi_d = np.degrees(lo), np.degrees(hi)
    ev = {}
    for i, b in enumerate(names[1:], start=1):
        cols = slice(3 * (i - 1), 3 * i)
        sw, tw = swing_twist(rep[:, cols], segment_axis(b, off, names, parents))
        for k, a in enumerate("xyz"):
            j = 3 * (i - 1) + k
            for side, exc in (("lo", lo_d[j] - val[:, j]), ("hi", val[:, j] - hi_d[j])):
                past = exc > LIMIT_REPORT_DEG
                e = np.maximum(exc, 0.0)
                rec = {"share_past": float(past.mean()), "frames_past": int(past.sum()),
                       "clips_past": int(len(np.unique(clip[past]))),
                       "excess_p95_deg": float(np.percentile(e, 95)), "excess_p99_deg": float(np.percentile(e, 99)),
                       "excess_max_deg": float(e.max()), "bound_shipped_deg": float(lo_d[j] if side == "lo" else hi_d[j]),
                       "value_quantile_deg": float(np.percentile(val[:, j], LIMIT_QUANTILE if side == "lo"
                                                                 else 100 - LIMIT_QUANTILE))}
                if past.sum() >= 20:
                    rec["on_frames_past"] = {"swing_rotvec_p1_p50_p99_deg": np.percentile(sw[past], [1, 50, 99], axis=0).T.round(1).tolist(),
                                             "twist_p1_p50_p99_deg": np.percentile(tw[past], [1, 50, 99]).round(1).tolist(),
                                             "swing_magnitude_p50_p99_deg": np.percentile(np.linalg.norm(sw[past], axis=1), [50, 99]).round(1).tolist()}
                ev[f"{b}_{a}:{side}"] = rec
    any_past = np.zeros(len(val), bool)
    for j in range(69):
        any_past |= (lo_d[j] - val[:, j] > LIMIT_REPORT_DEG) | (val[:, j] - hi_d[j] > LIMIT_REPORT_DEG)
    return ev, {"frames": int(len(val)), "clips": len(stems), "frames_any_past": int(any_past.sum())}, val


def decide_ranges(ev, names, lo, hi):
    """``(lower, upper [69] deg, decisions)`` from ``LIMIT_DECISIONS``; raises if a coordinate side that
    needs a decision (``LIMIT_SHARE_MIN``) has none."""
    lo_d, hi_d = np.degrees(lo).round(6), np.degrees(hi).round(6)
    new_lo, new_hi = lo_d.copy(), hi_d.copy()
    jidx = {f"{b}_{a}": 3 * (i - 1) + k for i, b in enumerate(names[1:], start=1) for k, a in enumerate("xyz")}
    covered, out = set(), []
    for d in LIMIT_DECISIONS:
        a, side = d["axis"], d["side"]
        sides = [(d["joint"], side)]
        mj, ms = mirror_side(d["joint"], a, side)
        if (mj, ms) != (d["joint"], side):
            sides.append((mj, ms))
        need = 0.0
        for j, s in sides:
            covered.add(f"{j}_{a}:{s}")
            q = ev[f"{j}_{a}:{s}"]["value_quantile_deg"]
            need = max(need, -q if s == "lo" else q)
        rec = {**d, "sides": [f"{j}_{a}:{s}" for j, s in sides],
               "evidence": {f"{j}_{a}:{s}": ev[f"{j}_{a}:{s}"] for j, s in sides}}
        if d["verdict"] == "widen":
            bound = LIMIT_ROUND_DEG * math.ceil(need / LIMIT_ROUND_DEG - 1e-9)
            rec["new_bound_abs_deg"] = bound
            for j, s in sides:
                k = jidx[f"{j}_{a}"]
                if s == "lo":
                    new_lo[k] = min(new_lo[k], -bound)
                else:
                    new_hi[k] = max(new_hi[k], bound)
        out.append(rec)
    missing = [k for k, e in ev.items() if e["share_past"] >= LIMIT_SHARE_MIN and k not in covered]
    if missing:
        raise ValueError(f"coordinate sides past the box on >= {LIMIT_SHARE_MIN:.0%} of frames without a decision: {missing}")
    return new_lo, new_hi, out


# --------------------------------------------------------------------------- #
# XML (text surgery: nothing but the listed attributes changes, identically in both files)
# --------------------------------------------------------------------------- #
def _fmt(vals, nd=6) -> str:
    return " ".join(f"{float(v):.{nd}f}" for v in np.atleast_1d(vals))


def _fmt_range(a, b) -> str:
    f = lambda x: str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.6f}"  # noqa: E731
    return f"{f(a)} {f(b)}"


def _body_span(text: str, name: str) -> tuple[int, int]:
    m = re.search(r'<body\s+name="%s"' % re.escape(name), text)
    if m is None:
        raise KeyError(name)
    nxt = [i for i in (text.find("<body ", m.end()), text.find("</body>", m.end())) if i != -1]
    return m.start(), min(nxt)


def _set_attr(tag: str, attr: str, value: str) -> str:
    pat = re.compile(r'(\s%s=")([^"]*)(")' % re.escape(attr))
    new, n = pat.subn(lambda m: m.group(1) + value + m.group(3), tag)
    if n != 1:
        raise ValueError(f"attribute {attr!r}: {n} matches in {tag[:80]}")
    return new


def geom_attrs(g, density) -> dict:
    """The MJCF attributes of one v2 geom (the geom keeps its shipped type)."""
    rho = f"{density:.6f}"
    if g["type"] == "sphere":
        return {"density": rho, "size": _fmt(g["radius"]), "pos": _fmt(g["center"])}
    if g["type"] == "capsule":
        return {"density": rho, "size": _fmt(g["radius"]), "fromto": _fmt(np.asarray(g["seg"]).ravel())}
    q = np.asarray(g["quat"], float)                                             # xyzw
    return {"density": rho, "size": _fmt(g["half"]), "pos": _fmt(g["center"]), "quat": _fmt([q[3], q[0], q[1], q[2]])}


def edit_xml(text, names, off, geo, density, ranges, torques):
    if f'<mujoco model="{SRC_MODEL_NAME}">' not in text:
        raise ValueError("not the shipped plant")
    text = text.replace(f'<mujoco model="{SRC_MODEL_NAME}">', f'<mujoco model="{MODEL_NAME}">', 1)
    for b in names:
        s, e = _body_span(text, b)
        block = text[s:e]
        head_end = block.index(">") + 1
        head, rest = block[:head_end], block[head_end:]
        if b != names[0]:
            head = _set_attr(head, "pos", _fmt(off[b], 8))
        tags = re.findall(r"<geom\b[^>]*/>", rest)
        if len(tags) != 1:
            raise ValueError(f"{b}: expected one geom in its own block, found {len(tags)}")
        tag = tags[0]
        gtype = re.search(r'type="(\w+)"', tag).group(1)
        if gtype != geo[b]["type"]:
            raise ValueError(f"{b}: geom type {gtype} -> {geo[b]['type']} is not allowed")
        new = tag
        for attr, val in geom_attrs(geo[b], density[b]).items():
            new = _set_attr(new, attr, val)
        rest = rest.replace(tag, new, 1)
        for jn in sorted(set(ranges) | set(torques)):
            if not jn.startswith(b + "_") or jn[len(b) + 1:] not in ("x", "y", "z"):
                continue
            m = re.search(r'<joint\s+name="%s"[^>]*/>' % re.escape(jn), rest)
            jt = m.group(0)
            if jn in ranges:
                jt = _set_attr(jt, "range", _fmt_range(*ranges[jn]))
            if jn in torques:
                jt = _set_attr(jt, "actuatorfrcrange", _fmt_range(-torques[jn], torques[jn]))
            rest = rest.replace(m.group(0), jt, 1)
        text = text[:s] + head + rest + text[e:]
    for jn, t in torques.items():
        m = re.search(r'<motor\s+name="%s"[^>]*/>' % re.escape(jn), text)
        text = text.replace(m.group(0), _set_attr(m.group(0), "gear", _fmt_range(t, t).split()[0]), 1)
    return text


# --------------------------------------------------------------------------- #
# Checks on the written plant
# --------------------------------------------------------------------------- #
def mj_rest(xml):
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def joint_space_inertia(flat) -> dict:
    """``{joint group: mean diagonal of the joint-space inertia over its three hinges}`` at rest (mj_fullM,
    armature included, as the PD servo sees it)."""
    import mujoco

    m, d = mj_rest(flat)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    out = collections.defaultdict(list)
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            out[m.joint(j).name.rsplit("_", 1)[0]].append(M[m.jnt_dofadr[j], m.jnt_dofadr[j]])
    return {k: float(np.mean(v)) for k, v in out.items()}


def training_gains() -> dict:
    """``{joint group: (stiffness, damping)}`` as training resolves them (``SmplYogiRobotConfig``'s overrides,
    the MJCF's own values beneath)."""
    import contextlib
    import io

    from protomotions.components.pose_lib import extract_control_info
    from protomotions.robot_configs.smpl_yogi import SmplYogiRobotConfig

    ov = SmplYogiRobotConfig.__dataclass_fields__["control"].default_factory().override_control_info
    with contextlib.redirect_stdout(io.StringIO()):
        ci = extract_control_info(str(SRC), ov)
    return {jn.rsplit("_", 1)[0]: (float(c.stiffness), float(c.damping)) for jn, c in ci.items()}


def pd_decisions(pd: dict) -> dict:
    """For every joint group whose joint-space inertia moved by more than ``PD_INERTIA_RATIO``: its gains
    scaled by the inertia ratio, which keeps the servo's natural frequency sqrt(k / I) and damping ratio
    c / (2 sqrt(k I)). They live in the robot config (BodyFix Step 2 applies them); the MJCF is unchanged."""
    gains = training_gains()
    out = {}
    for g, v in pd.items():
        if v["beyond_threshold"]:
            k, c = gains[g]
            out[g] = {"inertia_ratio": v["ratio"], "stiffness": [k, round(k * v["ratio"], 1)],
                      "damping": [c, round(c * v["ratio"], 2)],
                      "natural_frequency_rad_s": {"shipped": math.sqrt(k / v["shipped"]), "v2_unchanged_gains": math.sqrt(k / v["v2"]),
                                                  "v2_scaled_gains": math.sqrt(k * v["ratio"] / v["v2"])},
                      "applies_in": "the v2 robot config (Step 2)"}
    return out


def whole_body(m, d):
    """Rest COM (relative to the pelvis origin) and inertia about the COM, avatar axes."""
    mass = m.body_mass[1:]
    com = (d.xipos[1:] * mass[:, None]).sum(0) / mass.sum()
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - com
        I += R @ np.diag(m.body_inertia[b]) @ R.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    return com - d.xpos[1], I


def check_plant(names, O, off, geo, mass, ranges_deg, torques, voxel_props, whole_human):
    """Everything the card asks of the XML pair without PhysX: both load and compile alike, 74.00 kg with the
    decided per-body masses, pose_lib's dof names (and limits), rest FK = her joints, rest overlaps, and the
    mass-property comparison."""
    import torch
    from protomotions.components.pose_lib import extract_kinematic_info

    from reference_curation import mosh_replay

    m, d = mj_rest(OUT)
    mf, df = mj_rest(OUT_FLAT)
    out = {"total_mass_kg": float(m.body_mass.sum()), "flat_total_mass_kg": float(mf.body_mass.sum())}
    same = {k: float(np.abs(getattr(m, k) - getattr(mf, k)).max()) for k in
            ("body_pos", "body_mass", "body_inertia", "body_ipos", "geom_size", "geom_pos", "geom_quat", "jnt_range",
             "jnt_actfrcrange", "actuator_gear")}
    out["main_vs_flat_max_abs"] = same
    body_mass = {m.body(i).name: float(m.body_mass[i]) for i in range(1, m.nbody)}
    out["body_mass_max_abs_err_kg"] = max(abs(body_mass[b] - mass[b]) for b in names)
    k0, k1 = extract_kinematic_info(str(SRC)), extract_kinematic_info(str(OUT))
    out["dof_names_identical"] = list(k0.dof_names) == list(k1.dof_names)
    out["body_names_identical"] = list(k0.body_names) == list(k1.body_names)
    lo1, hi1 = np.degrees(k1.dof_limits_lower.numpy()), np.degrees(k1.dof_limits_upper.numpy())
    exp_lo = np.array([ranges_deg[n][0] for n in k1.dof_names])
    exp_hi = np.array([ranges_deg[n][1] for n in k1.dof_names])
    out["dof_limits_match_decisions_max_abs_deg"] = float(max(np.abs(lo1 - exp_lo).max(), np.abs(hi1 - exp_hi).max()))
    lo0, hi0 = np.degrees(k0.dof_limits_lower.numpy()), np.degrees(k0.dof_limits_upper.numpy())
    out["dof_limits_changed"] = [n for n, a, b, c, e in zip(k1.dof_names, lo0, hi0, lo1, hi1)
                                 if abs(a - c) > 1e-6 or abs(b - e) > 1e-6]
    import mujoco

    ms = mujoco.MjModel.from_xml_path(str(SRC))
    tau = {m.joint(j).name: (float(m.jnt_actfrcrange[j][1]), float(ms.jnt_actfrcrange[j][1])) for j in range(1, m.njnt)}
    gear = {m.actuator(a).name: float(m.actuator_gear[a][0]) for a in range(m.nu)}
    out["torque_limits_changed_Nm"] = {n: v for n, (v, v0) in tau.items() if v != v0}
    out["torque_limits_match_decisions"] = (set(out["torque_limits_changed_Nm"]) == set(torques)
                                            and all(tau[n][0] == t for n, t in torques.items()))
    out["gear_equals_torque_limit"] = all(gear[n] == tau[n][0] for n in tau)
    # rest FK of the written XML (pose_lib float32 parse -> float64 FK) against her joints
    sk = mosh_replay.skeleton_for(OUT)
    pos, rot = mosh_replay.rest_pose(sk)
    target = np.stack([(O[b] - O["Pelvis"]) @ rt.AXES for b in names])
    out["rest_fk_max_err_m"] = float(np.abs(pos[0] - target).max())
    Pm = d.xpos[1:] - d.xpos[1]
    out["rest_fk_mujoco_max_err_m"] = float(np.abs(Pm - target).max())
    # rest overlaps of the written XML (its own parse of the geoms)
    g = mosh_replay.pair_gaps(sk, pos, rot)[0]
    pairs = mosh_replay.colliding_pairs(sk)
    out["colliding_pairs"] = int(len(pairs))
    out["rest_overlaps"] = [[names[a], names[b], float(x)] for (a, b), x in zip(pairs, g) if x < 0]
    order = np.argsort(g)[:6]
    out["rest_min_gaps_m"] = [[names[pairs[k][0]], names[pairs[k][1]], float(g[k])] for k in order]
    # mass properties: collider (MuJoCo) vs her voxel segment
    from scipy.spatial.transform import Rotation

    per = {}
    for i in range(1, m.nbody):
        b = m.body(i).name
        vp = voxel_props[b]
        Ih = np.linalg.eigvalsh(np.asarray(vp["inertia_com_kgm2"]))
        Ip = np.sort(m.body_inertia[i])
        per[b] = {"mass_kg": float(m.body_mass[i]), "collider_com_body_m": m.body_ipos[i].tolist(),
                  "voxel_com_body_m": vp["com_body_m"],
                  "com_offset_m": float(np.linalg.norm(m.body_ipos[i] - np.asarray(vp["com_body_m"]))),
                  "collider_principal_inertia_kgm2": Ip.tolist(), "voxel_principal_inertia_kgm2": Ih.tolist(),
                  "principal_inertia_ratio": (Ip / Ih).tolist()}
    out["per_body_mass_properties"] = per
    com_p, I_p = whole_body(m, d)
    ms, ds = mj_rest(SRC)
    com_s, I_s = whole_body(ms, ds)
    Ih = np.asarray(whole_human["inertia_com_kgm2"])
    out["whole_body"] = {"human_com_pelvis_frame_m": whole_human["com_pelvis_frame_m"], "v2_com_pelvis_frame_m": com_p.tolist(),
                         "shipped_com_pelvis_frame_m": com_s.tolist(),
                         "v2_com_minus_human_m": (com_p - np.asarray(whole_human["com_pelvis_frame_m"])).tolist(),
                         "human_inertia_diag_kgm2": np.diag(Ih).tolist(), "v2_inertia_diag_kgm2": np.diag(I_p).tolist(),
                         "shipped_inertia_diag_kgm2": np.diag(I_s).tolist(),
                         "v2_over_human_diag": (np.diag(I_p) / np.diag(Ih)).tolist(),
                         "shipped_over_human_diag": (np.diag(I_s) / np.diag(Ih)).tolist()}
    # standing height: pelvis joint above the lowest collider at rest
    sk_s = mosh_replay.skeleton_for(SRC)
    ps, rs = mosh_replay.rest_pose(sk_s)
    out["rest_pelvis_above_lowest_collider_m"] = float(-mosh_replay.body_lowest(sk, pos, rot)[0].min())
    out["shipped_rest_pelvis_above_lowest_collider_m"] = float(-mosh_replay.body_lowest(sk_s, ps, rs)[0].min())
    top = max(float(pos[0, names.index("Head"), 2] + geo["Head"]["center"][2] + geo["Head"]["radius"]), 0.0)
    out["rest_collider_stature_m"] = top + out["rest_pelvis_above_lowest_collider_m"]
    return out


# --------------------------------------------------------------------------- #
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def build(workers: int = 8) -> dict:
    """Build plant v2; returns the record (also written)."""
    import mujoco
    import torch
    from extract_contact_configs import mjcf_body_names

    from reference_curation import mosh_replay

    t0 = time.time()
    mdl, vt, J = load_template()
    names = mjcf_body_names(str(SRC))
    sk0 = mosh_replay.skeleton_for(SRC)
    parents = sk0.parents
    skin, area, nrm = skin_arrays(mdl, vt)
    geo, col_rec, O, off = build_colliders(mdl, vt, J, names, parents, skin, area, nrm)
    ship = shipped_geoms(names)

    # rest geometry: overlaps, trunk-sphere neighbours
    pairs, gaps, P = rest_gaps(names, parents, off, geo)
    _, gaps0, P0 = rest_gaps(names, parents, {b: sk0.offsets[i].numpy().astype(float) for i, b in enumerate(names) if i},
                             ship)
    neighbours = {f"{a}+{b}": {"v2_gap_m": sphere_gap(P, names, geo, a, b), "shipped_gap_m": sphere_gap(P0, names, ship, a, b)}
                  for a, b in TRUNK_NEIGHBOURS + (("Chest", "Neck"), ("Neck", "Head"))}
    for a, b in TRUNK_NEIGHBOURS:
        n = neighbours[f"{a}+{b}"]
        if n["v2_gap_m"] > max(n["shipped_gap_m"], 0.0) + 1e-9:
            raise ValueError(f"{a}+{b}: a gap of {n['v2_gap_m']:.4f} m opens (shipped {n['shipped_gap_m']:.4f}); nudge a sphere")
    overlaps = [(names[a], names[b], float(g)) for (a, b), g in zip(pairs, gaps) if g < 0]
    if overlaps:
        raise ValueError(f"rest overlaps: {overlaps}")
    k = pairs.index((names.index("Chest"), names.index("Head")))
    col_rec["Head"].update(rest_gap_to_chest_m=float(gaps[k]), rest_gap_to_neck_m=sphere_gap(P, names, geo, "Neck", "Head"))

    # masses
    pts, vol = voxelize(mdl, vt)
    rho = TARGET_MASS / vol.sum()
    dom = dominant_body(mdl, vt, pts, names)
    skin_mass = {b: float(vol[dom == i].sum() * rho) for i, b in enumerate(names)}
    body_of, moved = anatomical_partition(pts, vol, dom, O, names)
    mass = {b: float(vol[body_of == i].sum() * rho) for i, b in enumerate(names)}
    props, whole = segment_mass_properties(pts, vol, body_of, O, names, rho)
    density = {b: mass[b] / geom_volume(geo[b]) for b in names}
    deleva = de_leva_check(mass)

    # joint box (the shipped ranges from MuJoCo's float64 parse, in the dof order)
    ms = mujoco.MjModel.from_xml_path(str(SRC))
    dof_names = [f"{b}_{a}" for b in names[1:] for a in "xyz"]
    rng_ship = {ms.joint(j).name: np.degrees(ms.jnt_range[j]).round(6) for j in range(1, ms.njnt)}
    lo = np.radians([rng_ship[n][0] for n in dof_names])
    hi = np.radians([rng_ship[n][1] for n in dof_names])
    stems, dof, clip = corpus_dof(workers)
    ev, ev_tot, _ = joint_box_evidence(dof, clip, stems, names, parents, off, lo, hi)
    new_lo, new_hi, decisions = decide_ranges(ev, names, lo, hi)
    ranges = {n: (float(a), float(b)) for n, a, b in zip(dof_names, new_lo, new_hi)}
    changed = {n: {"shipped_deg": [float(rng_ship[n][0]), float(rng_ship[n][1])], "v2_deg": list(ranges[n])}
               for n in dof_names if abs(ranges[n][0] - rng_ship[n][0]) > 1e-6 or abs(ranges[n][1] - rng_ship[n][1]) > 1e-6}
    rep_v2 = rt.nearest_representative(torch.as_tensor(dof), torch.as_tensor(np.radians(new_lo)),
                                       torch.as_tensor(np.radians(new_hi))).numpy()
    val2 = np.degrees(rep_v2)
    past2 = (new_lo - val2 > LIMIT_REPORT_DEG) | (val2 - new_hi > LIMIT_REPORT_DEG)
    ev_tot["v2_frames_any_past"] = int(past2.any(1).sum())
    ev_tot["v2_coordinate_sides_past_on_1pct"] = sorted(
        f"{dof_names[j]}:{s}" for j in range(69) for s, m_ in (("lo", new_lo[j] - val2[:, j] > LIMIT_REPORT_DEG),
                                                                ("hi", val2[:, j] - new_hi[j] > LIMIT_REPORT_DEG))
        if m_.mean() >= LIMIT_SHARE_MIN)

    # torque limits
    tau_ship = {ms.joint(j).name: float(ms.jnt_actfrcrange[j][1]) for j in range(1, ms.njnt)}
    torques = {jn: float(t) for jn, t in TORQUE_DECISIONS.items()}

    # write (only the changed ranges are rewritten)
    new_ranges = {n: tuple(v["v2_deg"]) for n, v in changed.items()}
    for src, dst in ((SRC, OUT), (SRC_FLAT, OUT_FLAT)):
        dst.write_text(edit_xml(src.read_text(), names, off, geo, density, new_ranges, torques))
    checks = check_plant(names, O, off, geo, mass, ranges, torques, props, whole)
    jsi_s, jsi_v = joint_space_inertia(SRC_FLAT), joint_space_inertia(OUT_FLAT)
    pd = {g: {"shipped": jsi_s[g], "v2": jsi_v[g], "ratio": jsi_v[g] / jsi_s[g],
              "beyond_threshold": bool(max(jsi_v[g] / jsi_s[g], jsi_s[g] / jsi_v[g]) > PD_INERTIA_RATIO)} for g in jsi_s}
    pd_changes = pd_decisions(pd)

    limits_rec = {
        "plant": ids.display_path(OUT), "flat": ids.display_path(OUT_FLAT),
        "joint_ranges_changed_deg": changed,
        "torque_limits_changed_Nm": {jn: {"shipped": tau_ship[jn], "v2": t} for jn, t in torques.items()},
        "torque_limits_decision": TORQUE_NOTE,
        "pd_gains_changed_in_robot_config": pd_changes,
        "rule": {"frames_past_deg": LIMIT_REPORT_DEG, "share_needing_a_decision": LIMIT_SHARE_MIN,
                 "widened_bound": f"her p{LIMIT_QUANTILE:g} / p{100 - LIMIT_QUANTILE:g} over the left joint and the "
                                  f"mirrored right one, rounded outward to {LIMIT_ROUND_DEG:g} deg",
                 "corpus": f"{len(stems)} clips of the manifest, female MoSh fit, hand = fingers, "
                           f"nearest exp-map representative; dropped {list(DROPPED_STEMS)}"},
        "decisions": [{k: v for k, v in d.items() if k != "evidence"} for d in decisions],
    }
    record = {
        **ids.provenance(SCHEMA_VERSION, MODULE, __file__,
                         [SRC, SRC_FLAT, hm.MODEL_PATH, ids.DEFAULT_MANIFEST, REPO / "data/scripts/reference_curation/mosh_replay.py"]),
        "outputs": {"xml": ids.display_path(OUT), "flat": ids.display_path(OUT_FLAT), "limits": ids.display_path(LIMITS_JSON),
                    "xml_sha256": ids.sha256_file(OUT), "flat_sha256": ids.sha256_file(OUT_FLAT)},
        "template": {"stem": TEMPLATE_STEM, "sha1": TEMPLATE_SHA1, "height_m": float(np.ptp(vt[:, 1]))},
        "skeleton": {"joints_smplx_m": {b: O[b].tolist() for b in names},
                     "offsets_m": {b: {"v2": off[b].tolist(), "shipped": sk0.offsets[i].numpy().astype(float).tolist(),
                                       "length_v2_m": float(np.linalg.norm(off[b])),
                                       "length_shipped_m": float(np.linalg.norm(sk0.offsets[i].numpy()))}
                                   for i, b in enumerate(names) if i}},
        "colliders": {"rules": col_rec, "geoms_v2": {b: {k: v for k, v in geo[b].items()} for b in names},
                      "rest_min_gap_m": float(gaps.min()), "shipped_rest_min_gap_m": float(gaps0.min()), "rest_overlaps": overlaps,
                      "trunk_sphere_neighbours": neighbours},
        "masses": {"target_kg": TARGET_MASS, "uniform_density_kg_m3": rho, "voxel_pitch_m": VOXEL_PITCH,
                   "voxel_volume_m3": float(vol.sum()), "skinning_partition_kg": skin_mass,
                   "anatomical_partition_kg": mass, "cut_plane_transfers_kg": {k: {kk: vv * rho for kk, vv in v.items()}
                                                                               for k, v in moved.items()},
                   "geom_volume_m3": {b: geom_volume(geo[b]) for b in names}, "geom_density_kg_m3": density,
                   "de_leva_female": deleva, "voxel_segments": props, "whole_body_human": whole},
        "joint_box": {"totals": ev_tot, "decisions": decisions, "changed_deg": changed},
        "torque_limits": {"shipped_Nm": tau_ship, "v2_changed_Nm": torques, "decision": TORQUE_NOTE},
        "pd_gains": {"threshold_ratio": PD_INERTIA_RATIO, "joint_space_inertia_rest": pd,
                     "groups_beyond_threshold": [g for g, v in pd.items() if v["beyond_threshold"]],
                     "changes_for_robot_config": pd_changes},
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    record = jsonable(record)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    acc = old.get("acceptance")
    if acc and acc.get("xml_sha256") == record["outputs"]["xml_sha256"] and acc.get("flat_sha256") == record["outputs"]["flat_sha256"]:
        record["acceptance"] = acc                  # still measured on these very files
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    LIMITS_JSON.write_text(json.dumps(jsonable(limits_rec), indent=1) + "\n")
    return record


def failures(rec: dict) -> list[str]:
    c = rec["checks"]
    out = []
    if abs(c["total_mass_kg"] - TARGET_MASS) > 0.005 or abs(c["flat_total_mass_kg"] - TARGET_MASS) > 0.005:
        out.append(f"total mass {c['total_mass_kg']:.4f} / flat {c['flat_total_mass_kg']:.4f} kg")
    if max(c["main_vs_flat_max_abs"].values()) > 1e-9:
        out.append(f"main and flat compile differently: {c['main_vs_flat_max_abs']}")
    if not (c["dof_names_identical"] and c["body_names_identical"]):
        out.append("dof or body names differ from the shipped plant")
    if c["dof_limits_match_decisions_max_abs_deg"] > 1e-4:
        out.append(f"pose_lib limits differ from the decisions by {c['dof_limits_match_decisions_max_abs_deg']} deg")
    if c["rest_fk_max_err_m"] > 1e-5:
        out.append(f"rest FK misses her joints by {c['rest_fk_max_err_m']:.2e} m")
    if c["rest_overlaps"]:
        out.append(f"rest overlaps {c['rest_overlaps']}")
    if not (c["torque_limits_match_decisions"] and c["gear_equals_torque_limit"]):
        out.append(f"torque limits {c['torque_limits_changed_Nm']} differ from TORQUE_DECISIONS or the actuator gears")
    if abs(rec["colliders"]["rules"]["Head"]["crown_minus_surface_m"]) > CROWN_TOL_M:
        out.append("the head sphere misses her crown")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    rec = build(args.workers)
    bad = failures(rec)
    c = rec["checks"]
    print(f"plant v2: {ids.display_path(OUT)} + _flat; {c['total_mass_kg']:.3f} kg; rest FK err "
          f"{c['rest_fk_max_err_m']:.1e} m; rest min gap {rec['colliders']['rest_min_gap_m'] * 100:.2f} cm; "
          f"{len(rec['joint_box']['changed_deg'])} joint ranges and {len(rec['torque_limits']['v2_changed_Nm'])} "
          f"torque limits changed; {rec['seconds']} s" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
