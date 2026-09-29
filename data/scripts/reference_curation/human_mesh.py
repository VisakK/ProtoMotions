# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human mesh layer (BUILD_PLAN Step 5): the performer's own body, posed as MoSh++ fitted it, in the
clip frame; the floor and self contacts it gives every zone; and its render layer.

``load(stem)`` returns capture store v2: the Step 1 record (``capture.load``) plus the four
``human_*`` arrays of ``HUMAN_ARRAYS``. ``output/reference_curation/capture/v2/<stem>.{npz,json}``
holds only the human arrays and the identity of the v1 record they extend. v1 is never rewritten:
the Step 2 audit id and the Step 4 calibration tables are keyed on its generator hash.

The body
--------
MoSh++ fitted SMPL-X *female* with the subject's own template (``stagei_debug_details.v_template``,
betas all zero, one template for all 59 readable fits), with no DMPLs and no expressions
(``cfg.moshpp.optimize_dynamics`` and ``optimize_face`` are False). So the mesh is plain LBS with
pose correctives. The model contributes J_regressor, weights, posedirs and kintree
(``SMPLX_FEMALE.npz``; the v1.1 file beside it has identical posing arrays and adds the hand mean).
``SMPLX``, ``rodrigues`` and the skinning-rotation marker rebuild are ported from
``expert_revist/reference_curation_review_2026_09_28/smplx_model_check.py``, which verified them.

* **``fullpose`` is used as stored.** The hand pose is absolute (MOYO's reader:
  ``flat_hand_mean=True``). Adding the SMPL-X hand mean puts the fingers ~5 cm off and into the floor.
* **The hand mean only rebuilds markers.** MoSh's latent markers live in a canonical space whose
  hands are mean-posed (``use_hands_mean=True``); the mesh never uses it.
* **Registration.** MoSh's marker frame sits (0, 0, +10.17 mm) from ``LBS + trans``: over the 59
  fits z is 10.13-10.19 mm and |xy| < 0.05 mm. ``register`` measures it per clip as the median offset
  of the rebuilt non-finger markers to ``markers_sim``, and refuses a clip 0.5 mm off. The clip
  frame then adds ``capture.VICON_TO_CLIP_XY``; z is shared, floor at 0.

Verification
------------
The check script's criteria, per clip over 24 evenly spaced frames: non-finger median < 1.5 mm with
at least 80 % of the markers under 2 mm, finger median < 1.5 mm. The rebuild attaches a latent marker
to its vertex's *surface frame* (normal and one edge). The check script's rule, which turns the offset
with the vertex's skinning rotation, is kept as ``attach="skinning"`` and reproduces its printed
numbers, but it fails the 80 % criterion on one of the 59 fits (Lord of the Dance -c, 79.2 %). The
surface frame passes all 59: worst share 83 %, non-finger median <= 0.67 mm (0.92 under the check's
rule), fingers <= 0.20 mm (0.31). A few shoulder and waist markers (RBSH, LFSH, MFWT) stay 3-15 mm
off under both rules: MoSh's attachment is exactly neither. The mesh is not in question.

Zones and contacts
------------------
A vertex belongs to the zone of its dominant skinning joint (``JOINT_ZONE``), which is how
``extract_contact_configs.ZONES`` maps the MJCF bodies: a body is named after its proximal joint, so
the ``L_Knee`` body is the shin, as the ``left_knee`` joint's vertices are. The IPMAN segments
(``yogi_segments/smplx``) cannot express the zones: they are ten closed parts made for volume, with
the hand inside the forearm, the foot inside the lower leg and the pelvis inside the torso. The
tests use them as an independent check of the sides.

* ``human_min_z``: the zone's lowest vertex (a mesh's lowest point over a plane is a vertex). The mesh
  is skin with no soft tissue, so a loaded part goes *into* the floor. Mat-confirmed medians: pelvis
  -5.1 cm (lying and seated poses), back -1.2 cm, back of the head -0.65 cm, foot +0.4 cm.
* ``human_ground_state``: Step 1's Schmitt trigger and touch rule, the loaded p99 + 0.5 cm. The p99 is
  pooled over the feet, hands and head on mat-confirmed frames, and one threshold serves all 15 zones.
  Skin has no per-zone marker offset, and the other zones' mat-confirmed frames are attributed on the
  *avatar* and read wide (loaded "thigh" p99 4.8 cm, 529 of its 594 left frames from one clip, Scale
  pose). The band is 1 cm, not Step 1's 2.5 cm. The mesh is smooth (7 runs shorter than 0.1 s in the
  corpus), and a 2.5 cm band kept Shoulder-Pressing -a's left foot touching for 3.8 s with its sole
  3.6 cm up.
* ``human_pair_gap``: the smallest vertex-to-vertex distance between two zones, capped at 10 cm, for
  the 91 zone pairs that are not kinematically adjacent (``PAIR_NAMES``). The labels' 65 pairs are all
  among them, and so are head-upper arm, which the avatar's list leaves out as a capsule artifact. A
  vertex within 10 cm of the other zone along the template surface does not count. That removes the
  thighs' shared crotch seam (0.5 cm apart along the surface) and nothing else: the next closest pair,
  head-upper arm, is 14.4 cm apart.
* ``human_pair_state``: contact at or below 2 cm, apart above 3 cm (Step 4's body-body "apart" band).
  Vertex to vertex overstates a skin gap by up to half an edge, and limb edges are 2-3 cm (median).

A zone is unknown (-1) where v1's marker hygiene fails it (median fit residual > 5 cm: the fit, and so
the mesh, is not following the markers there), and a pair where either zone is.

Render layer
------------
``render_layer(stem)`` fills ``render.HumanMesh``: the posed mesh in the clip frame, translucent grey,
co-registered with the avatar and the markers (MuJoCo keeps its vertices to 5 um).

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.human_mesh --all
    ... --stem <stem> [<stem> ...]                         # with the committed calibration
    ... --render --stem <stem> --frame 570 [--zones L_FOOT]  # a frame with the mesh, by eye
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import functools
import hashlib
import itertools
import json
import multiprocessing
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from extract_contact_configs import ADJACENT, ZONE_ORDER
from reference_curation import capture, ids

MODULE = "reference_curation.human_mesh"
SCHEMA_VERSION = 1
STORE_VERSION = "v2"
STORE_DIR = ids.OUTPUT_ROOT / "capture" / STORE_VERSION
CALIBRATION_PATH = ids.DATA_ROOT / "calibration" / f"capture_{STORE_VERSION}.json"
MODEL_PATH = ids.MOYO_DATA / "body_models/smplx/SMPLX_FEMALE.npz"

EXPECTED_OFFSET_M = (0.0, 0.0, 0.01017)
OFFSET_TOL_M = 0.0005
VERIFY_FRAMES = 24
VERIFY_MEDIAN_MM = 1.5
VERIFY_UNDER_MM = 2.0
VERIFY_SHARE = 0.8
CHUNK_FRAMES = 64
CALIB_ZONES = ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND", "HEAD")
SEPARATION_GAP_M = 0.010
SEAM_M = 0.10
PAIR_CAP_M = 0.10
PAIR_TOUCH_M = 0.02
PAIR_SEPARATION_M = 0.03

# SMPL-X joint -> zone: pelvis, hips, spine1, knees, spine2, ankles, spine3, feet, neck, collars, head,
# shoulders, elbows, wrists, jaw, eyes (0-24), then the 15 finger joints of each hand.
JOINT_ZONE = ("PELVIS", "L_THIGH", "R_THIGH", "TRUNK", "L_SHANK", "R_SHANK", "TRUNK", "L_FOOT", "R_FOOT",
              "TRUNK", "L_FOOT", "R_FOOT", "HEAD", "TRUNK", "TRUNK", "HEAD", "L_UPPER_ARM", "R_UPPER_ARM",
              "L_FOREARM", "R_FOREARM", "L_HAND", "R_HAND", "HEAD", "HEAD", "HEAD") \
    + ("L_HAND",) * 15 + ("R_HAND",) * 15
# The avatar's ADJACENT also lists head-upper arm, as a capsule-fit artifact; on the skin it is a contact.
KINEMATIC_ADJACENT = ADJACENT - {frozenset(("HEAD", f"{s}_UPPER_ARM")) for s in "LR"}
PAIRS = tuple((a, b) for a, b in itertools.combinations(ZONE_ORDER, 2) if frozenset((a, b)) not in KINEMATIC_ADJACENT)
PAIR_NAMES = tuple(f"{a}+{b}" for a, b in PAIRS)
_PAIR_ZI = tuple((ZONE_ORDER.index(a), ZONE_ORDER.index(b)) for a, b in PAIRS)

HUMAN_ARRAYS = {
    "human_min_z": ("m", "[T,Z] lowest skin point of the zone on the registered MoSh SMPL-X mesh "
                         "(negative: into the floor)"),
    "human_ground_state": ("int8", "[T,Z] human floor touch from the mesh: 1 contact, 0 separated, -1 unknown"),
    "human_pair_gap": ("m", f"[T,P] smallest skin distance between the pair's zones (PAIR_NAMES), capped at "
                            f"{PAIR_CAP_M:g} m: the cap means at least that far"),
    "human_pair_state": ("int8", "[T,P] human self contact: 1 contact, 0 apart, -1 unknown"),
}


# --------------------------------------------------------------------------- #
# SMPL-X (ported from smplx_model_check.py)
# --------------------------------------------------------------------------- #
def rodrigues(aa):
    """[..., 3] axis-angle -> [..., 3, 3]."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(theta, 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def find_hands_mean(model_path):
    """``(hands_mean [90], path)`` from the model file or a sibling ``*.npz``, else ``(None, None)``."""
    for p in [model_path] + sorted(q for q in model_path.parent.glob("*.npz") if q != model_path):
        m = np.load(p, allow_pickle=True)
        if "hands_meanl" in m.files and "hands_meanr" in m.files:
            return np.concatenate([m["hands_meanl"], m["hands_meanr"]]).astype(np.float64), p
    return None, None


class SMPLX:
    def __init__(self, path=MODEL_PATH):
        m = np.load(path, allow_pickle=True)
        self.path = Path(path)
        self.weights = m["weights"].astype(np.float64)                     # [V, 55]
        self.J_regressor = m["J_regressor"].astype(np.float64)             # [55, V]
        self.posedirs = m["posedirs"].astype(np.float64)                   # [V, 3, 486]
        self.faces = m["f"].astype(np.int64)
        parents = m["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        self.parents = parents
        self.part = self.weights.argmax(1)                                 # dominant joint per vertex
        self.hands_mean, self.hands_mean_source = find_hands_mean(Path(path))
        self.zone = np.array([ZONE_ORDER.index(JOINT_ZONE[j]) for j in self.part])
        self.zone_vertices = [np.nonzero(self.zone == zi)[0] for zi in range(len(ZONE_ORDER))]

    def _joints(self, v_template, fullpose):
        """``(R [F,55,3,3], A [F,55,4,4])``: joint rotations and the rest-to-posed joint transforms."""
        F = fullpose.shape[0]
        R = rodrigues(fullpose.reshape(F, 55, 3))
        J = self.J_regressor @ v_template
        G = np.zeros((F, 55, 4, 4))
        G[:, :, 3, 3] = 1.0
        for i in range(55):
            local = np.zeros((F, 4, 4))
            local[:, :3, :3] = R[:, i]
            local[:, :3, 3] = J[i] - (J[self.parents[i]] if i > 0 else 0.0)
            local[:, 3, 3] = 1.0
            G[:, i] = local if i == 0 else G[:, self.parents[i]] @ local
        A = G.copy()
        A[:, :, :3, 3] -= np.einsum("fjab,jb->fja", G[:, :, :3, :3], J)
        return R, A

    def transforms(self, v_template, fullpose, vids=None):
        """(skin transforms [F,V',4,4], pose-corrective offsets [F,V',3]) for vertices ``vids``.
        ``fullpose`` is used as stored: the hand pose is absolute (MOYO: ``flat_hand_mean=True``)."""
        F = fullpose.shape[0]
        R, A = self._joints(v_template, fullpose)
        feat = (R[:, 1:] - np.eye(3)).reshape(F, -1)
        pd = self.posedirs if vids is None else self.posedirs[vids]
        dv = np.einsum("fp,vcp->fvc", feat, pd)
        w = self.weights if vids is None else self.weights[vids]
        return np.einsum("vj,fjab->fvab", w, A), dv

    def vertices(self, v_template, fullpose, trans):
        T, dv = self.transforms(v_template, fullpose)
        v = v_template[None] + dv
        return np.einsum("fvab,fvb->fva", T[..., :3, :3], v) + T[..., :3, 3] + trans[:, None]

    def posed(self, v_template, fullpose, trans, chunk: int = CHUNK_FRAMES):
        """Yield ``(first frame, [F, V, 3])``: ``vertices`` for every frame, in chunks and through
        BLAS (a 1000-frame ``vertices`` call would hold 1.3 GB of transforms)."""
        pd = self.posedirs.reshape(-1, self.posedirs.shape[-1])            # [V*3, 486]
        V = v_template.shape[0]
        for s in range(0, fullpose.shape[0], chunk):
            fp = fullpose[s:s + chunk]
            F = fp.shape[0]
            R, A = self._joints(v_template, fp)
            v = v_template[None] + ((R[:, 1:] - np.eye(3)).reshape(F, -1) @ pd.T).reshape(F, V, 3)
            T = (self.weights @ A[:, :, :3].transpose(1, 0, 2, 3).reshape(55, -1)).reshape(V, F, 3, 4)
            T = T.transpose(1, 0, 2, 3)
            yield s, np.einsum("fvab,fvb->fva", T[..., :3], v) + T[..., 3] + trans[s:s + F, None]


@functools.lru_cache(maxsize=1)
def model() -> SMPLX:
    return SMPLX(MODEL_PATH)


# --------------------------------------------------------------------------- #
# The fit, its markers and the registration
# --------------------------------------------------------------------------- #
def load_fit(stem: str) -> tuple[dict | None, str, str | None]:
    """``(fit, status, error)`` like ``capture.load_mosh``: ``fit`` holds ``v_template``,
    ``fullpose``, ``trans``, the latent markers and ``sim`` ``[T,73,3]`` (Vicon frame)."""
    path = ids.mosh_path(stem)
    if path is None:
        return None, "no_fit", "MOYO has no MoSh fit for this clip"
    try:
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        s2 = d["stageii_debug_details"]
        fit = dict(v_template=np.asarray(d["stagei_debug_details"]["v_template"], np.float64),
                   fullpose=np.asarray(d["fullpose"], np.float64), trans=np.asarray(d["trans"], np.float64),
                   latent=np.asarray(d["markers_latent"], np.float64), labels=list(d["latent_labels"]),
                   vids=d["markers_latent_vids"], mtype=d["marker_meta"]["marker_type"],
                   sim=np.asarray(s2["markers_sim"], np.float64), fps=float(s2["mocap_frame_rate"]), path=path)
        obs_labels = [str(x) for x in s2["labels_obs"][0]]
    except Exception as exc:  # noqa: BLE001 -- one fit on disk is corrupt (Standing big toe hold -c)
        return None, "unreadable", f"{type(exc).__name__}: {exc}"
    if fit["labels"] != obs_labels:  # markers_sim is indexed like markers_obs; the rebuild like the latents
        raise ValueError(f"{path}: latent_labels differ from labels_obs")
    return fit, "ok", None


def _surface_frame(pos, marker_faces, neighbour):
    """``[F, M, 3, 3]`` rows (tangent, bitangent, normal) at each marker vertex: the area-weighted
    normal of its faces and the edge to one neighbour, projected into the tangent plane."""
    n = np.zeros(pos.shape[:1] + (len(marker_faces), 3))
    for m, faces in enumerate(marker_faces):
        a, b, c = pos[:, faces[:, 0]], pos[:, faces[:, 1]], pos[:, faces[:, 2]]
        n[:, m] = np.cross(b - a, c - a).sum(1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    e = pos[:, neighbour[:, 1]] - pos[:, neighbour[:, 0]]
    t = e - (e * n).sum(-1, keepdims=True) * n
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    return np.stack([t, np.cross(n, t), n], -2)


def rebuild_markers(mdl: SMPLX, fit: dict, frames, attach: str = "surface") -> np.ndarray:
    """``[F, 73, 3]`` MoSh's markers at ``frames`` (Vicon frame, before registration). Each latent
    marker is an offset from its vertex in the canonical space whose hands are mean-posed. With
    ``attach="skinning"`` the offset turns with the vertex's skinning rotation (the check script's
    rule); with ``"surface"`` it rides the vertex's surface frame (normal and one edge)."""
    if mdl.hands_mean is None:
        raise FileNotFoundError(f"no SMPL-X hand mean beside {mdl.path}: rebuilding markers needs the v1.1 file")
    frames = np.asarray(frames)
    vid = np.array([int(fit["vids"][lab]) for lab in fit["labels"]])
    vt = fit["v_template"]
    canon = np.zeros((1, 165))
    canon[0, 75:] = mdl.hands_mean
    if attach == "skinning":
        Tc, dvc = mdl.transforms(vt, canon, vid)
        S_c = np.einsum("mab,mb->ma", Tc[0, :, :3, :3], vt[vid] + dvc[0]) + Tc[0, :, :3, 3]
        off = fit["latent"] - S_c                                             # marker offset, canonical space
        T, dv = mdl.transforms(vt, fit["fullpose"][frames], vid)
        S = np.einsum("fmab,fmb->fma", T[..., :3, :3], vt[vid][None] + dv) + T[..., :3, 3]
        R_rel = np.einsum("fmab,mcb->fmac", T[..., :3, :3], Tc[0, :, :3, :3])  # R_pose R_canon^T
        return S + np.einsum("fmab,mb->fma", R_rel, off) + fit["trans"][frames][:, None]
    if attach != "surface":
        raise ValueError(f"unknown attach rule {attach!r}")
    marker_faces = [mdl.faces[(mdl.faces == v).any(1)] for v in vid]
    need = np.unique(np.concatenate([vid] + [f.ravel() for f in marker_faces]))   # skin only these
    faces_local = [np.searchsorted(need, f) for f in marker_faces]
    # the edge from the marker vertex to the first other vertex of its first face
    neighbour = np.searchsorted(need, [[v, next(u for u in f[0] if u != v)] for v, f in zip(vid, marker_faces)])
    at = neighbour[:, 0]

    def posed(pose, trans):
        T, dv = mdl.transforms(vt, pose, need)
        return np.einsum("fvab,fvb->fva", T[..., :3, :3], vt[need][None] + dv) + T[..., :3, 3] + trans[:, None]

    pc = posed(canon, np.zeros((1, 3)))
    coef = np.einsum("mab,mb->ma", _surface_frame(pc, faces_local, neighbour)[0], fit["latent"] - pc[0, at])
    pp = posed(fit["fullpose"][frames], fit["trans"][frames])
    return pp[:, at] + np.einsum("fmab,ma->fmb", _surface_frame(pp, faces_local, neighbour), coef)


def register(mdl: SMPLX, fit: dict, frames=None, attach: str = "surface") -> tuple[np.ndarray, dict]:
    """``(offset [3], verification)``: the Vicon-frame offset of MoSh's markers from ``LBS + trans``
    (the median over non-finger markers and ``frames``, default 24 evenly spaced) and the check
    script's grades. ``verification["ok"]`` needs the offset within 0.5 mm of +10.17 mm in z and the
    markers within the criteria."""
    T = fit["fullpose"].shape[0]
    frames = np.linspace(0, T - 1, VERIFY_FRAMES).astype(int) if frames is None else np.asarray(frames)
    rebuilt = rebuild_markers(mdl, fit, frames, attach)
    is_f = np.array([fit["mtype"][lab] == "finger" for lab in fit["labels"]])
    offset = np.median((fit["sim"][frames] - rebuilt)[:, ~is_f].reshape(-1, 3), axis=0)
    err = np.linalg.norm(rebuilt + offset - fit["sim"][frames], axis=-1) * 1000.0     # mm
    per_marker = np.median(err[:, ~is_f], axis=0)
    labels_nf = [lab for lab, f in zip(fit["labels"], is_f) if not f]
    v = {"attach": attach, "frames": len(frames), "offset_mm": [round(float(x), 3) for x in 1000.0 * offset],
         "nonfinger_median_mm": round(float(np.median(err[:, ~is_f])), 3),
         "nonfinger_share_under_2mm": round(float((per_marker < VERIFY_UNDER_MM).mean()), 4),
         "finger_median_mm": round(float(np.median(err[:, is_f])), 3),
         "by_type": {t: {"median_mm": round(float(np.median(e)), 3), "p95_mm": round(float(np.percentile(e, 95)), 3),
                         "max_mm": round(float(e.max()), 3)}
                     for t in ("body", "feet", "hand", "finger", "head")
                     for e in [err[:, [i for i, lab in enumerate(fit["labels"]) if fit["mtype"][lab] == t]]] if e.size},
         "outliers_mm": {labels_nf[i]: round(float(per_marker[i]), 1) for i in np.argsort(-per_marker)
                         if per_marker[i] >= VERIFY_UNDER_MM}}
    v["offset_ok"] = bool(np.abs(offset - np.asarray(EXPECTED_OFFSET_M)).max() <= OFFSET_TOL_M)
    v["markers_ok"] = bool(v["nonfinger_median_mm"] < VERIFY_MEDIAN_MM and v["finger_median_mm"] < VERIFY_MEDIAN_MM
                           and v["nonfinger_share_under_2mm"] >= VERIFY_SHARE)
    v["ok"] = v["offset_ok"] and v["markers_ok"]
    return offset, v


@dataclass(frozen=True)
class Human:
    """One clip's registered body. ``shift`` takes ``LBS + trans`` into the clip frame."""
    stem: str
    model: SMPLX
    fit: dict
    shift: np.ndarray
    verification: dict

    @property
    def num_frames(self) -> int:
        return self.fit["fullpose"].shape[0]

    def chunks(self, chunk: int = CHUNK_FRAMES):
        """Yield ``(first frame, [F, V, 3])`` over the whole clip, in the clip frame."""
        f = self.fit
        for s, v in self.model.posed(f["v_template"], f["fullpose"], f["trans"], chunk):
            yield s, v + self.shift

    def vertices(self, frame: int) -> np.ndarray:
        """``[V, 3]`` the posed mesh at ``frame``, in the clip frame."""
        if not 0 <= int(frame) < self.num_frames:
            raise IndexError(f"{self.stem}: frame {frame} outside 0..{self.num_frames - 1}")
        f, k = self.fit, slice(int(frame), int(frame) + 1)
        return next(self.model.posed(f["v_template"], f["fullpose"][k], f["trans"][k], 1))[1][0] + self.shift


def load_human(stem: str, mdl: SMPLX | None = None) -> tuple[Human | None, str, str | None]:
    """``(human, status, error)``. Raises if the fit is readable but fails registration or the
    marker verification: that is a wrong model or a wrong frame, not missing data."""
    mdl = model() if mdl is None else mdl
    fit, status, error = load_fit(stem)
    if fit is None:
        return None, status, error
    offset, v = register(mdl, fit)
    if not v["ok"]:
        raise ValueError(f"{stem}: verification failed: offset {v['offset_mm']} mm, non-finger median "
                         f"{v['nonfinger_median_mm']} mm, share {v['nonfinger_share_under_2mm']}, "
                         f"finger median {v['finger_median_mm']} mm")
    shift = offset + np.array([*capture.VICON_TO_CLIP_XY, 0.0])
    return Human(stem, mdl, fit, shift, v), "ok", None


# --------------------------------------------------------------------------- #
# Contacts on the mesh (pure)
# --------------------------------------------------------------------------- #
def zone_geodesics(mdl: SMPLX, v_template: np.ndarray) -> np.ndarray:
    """``[Z, V]`` distance of every vertex to each zone along the template's surface (edge graph)."""
    edges = np.concatenate([mdl.faces[:, [0, 1]], mdl.faces[:, [1, 2]], mdl.faces[:, [2, 0]]])
    e = np.unique(np.sort(edges, 1), axis=0)
    w = np.linalg.norm(v_template[e[:, 0]] - v_template[e[:, 1]], axis=1)
    V = v_template.shape[0]
    graph = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(V, V)).tocsr()
    return np.stack([dijkstra(graph, indices=idx, min_only=True) for idx in mdl.zone_vertices])


_PAIR_SETS: dict = {}


def pair_vertex_sets(mdl: SMPLX, v_template: np.ndarray) -> list[tuple]:
    """Per pair of ``PAIRS``: ``(query vertices, tree vertices, tree key)``, the smaller side queried
    against the larger. A vertex within ``SEAM_M`` of the other zone along the surface is left out."""
    key = (id(mdl), hashlib.sha1(np.ascontiguousarray(v_template).tobytes()).hexdigest())
    if key not in _PAIR_SETS:
        geo = zone_geodesics(mdl, v_template)
        sets = []
        for k, (ai, bi) in enumerate(_PAIR_ZI):
            sides = []
            for zi, other in ((ai, bi), (bi, ai)):
                idx = mdl.zone_vertices[zi]
                keep = idx[geo[other][idx] > SEAM_M]
                sides.append((keep, zi if len(keep) == len(idx) else (zi, k)))
            (qa, _), (qb, kb) = sorted(sides, key=lambda s: len(s[0]))
            sets.append((qa, qb, kb))
        _PAIR_SETS[key] = sets
    return _PAIR_SETS[key]


def zone_min_z(verts: np.ndarray, mdl: SMPLX) -> np.ndarray:
    """``[F, Z]`` lowest vertex of every zone."""
    return np.stack([verts[:, idx, 2].min(1) for idx in mdl.zone_vertices], -1)


def pair_gaps(v: np.ndarray, mdl: SMPLX, sets: list[tuple]) -> np.ndarray:
    """``[P]`` smallest distance between each pair's vertex sets on one frame ``v [V, 3]``, capped at
    ``PAIR_CAP_M``. Pairs whose zones' boxes are a cap apart skip the tree query."""
    lo = np.stack([v[idx].min(0) for idx in mdl.zone_vertices])
    hi = np.stack([v[idx].max(0) for idx in mdl.zone_vertices])
    out = np.full(len(PAIRS), PAIR_CAP_M)
    trees = {}
    for k, (ai, bi) in enumerate(_PAIR_ZI):
        box = np.maximum(0.0, np.maximum(lo[ai] - hi[bi], lo[bi] - hi[ai]))
        if box @ box >= PAIR_CAP_M ** 2:
            continue
        query, tree_vertices, tree_key = sets[k]
        if tree_key not in trees:
            trees[tree_key] = cKDTree(v[tree_vertices])
        out[k] = min(float(trees[tree_key].query(v[query], distance_upper_bound=PAIR_CAP_M)[0].min()), PAIR_CAP_M)
    return out


def contacts(human: Human) -> tuple[np.ndarray, np.ndarray]:
    """``(min_z [T, Z], pair_gap [T, P])`` over the whole clip, in metres."""
    sets = pair_vertex_sets(human.model, human.fit["v_template"])
    min_z = np.empty((human.num_frames, len(ZONE_ORDER)))
    gap = np.empty((human.num_frames, len(PAIRS)))
    for s, verts in human.chunks():
        min_z[s:s + len(verts)] = zone_min_z(verts, human.model)
        for k, v in enumerate(verts):
            gap[s + k] = pair_gaps(v, human.model, sets)
    return min_z, gap


def states(min_z: np.ndarray, gap: np.ndarray, known: np.ndarray, ground: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(human_ground_state [T, Z], human_pair_state [T, P])``: Step 1's Schmitt trigger, on the
    calibrated ground thresholds and on the pair thresholds; unknown where ``known`` [T, Z] is not."""
    g = np.stack([capture.contact_state(min_z[:, zi], known[:, zi], ground["touch_m"], ground["separation_m"])
                  for zi in range(len(ZONE_ORDER))], -1)
    p = np.stack([capture.contact_state(gap[:, k], known[:, ai] & known[:, bi], PAIR_TOUCH_M, PAIR_SEPARATION_M)
                  for k, (ai, bi) in enumerate(_PAIR_ZI)], -1)
    return g.astype(np.int8), p.astype(np.int8)


# --------------------------------------------------------------------------- #
# One clip, before thresholds
# --------------------------------------------------------------------------- #
def measure(stem: str) -> dict:
    """The human arrays of one x0 clip before thresholds: ``{"arrays", "meta", "inputs"}``."""
    if ids.split_clip_name(stem)[1]:
        raise ValueError(f"{stem}: build on x0 clips only; map variant frames with ids.source_frame_index")
    mdl = model()
    human, status, error = load_human(stem, mdl)
    inputs = [mdl.path] + ([mdl.hands_mean_source] if mdl.hands_mean_source else [])
    meta = {"stem": stem, "human_available": human is not None, "human_status": status, "human_error": error}
    if human is None:
        return {"arrays": None, "meta": meta, "inputs": inputs}
    min_z, gap = contacts(human)
    meta.update(num_frames=human.num_frames, fps=human.fit["fps"], verification=human.verification,
                registration={"offset_vicon_mm": human.verification["offset_mm"],
                              "shift_to_clip_m": [round(float(x), 6) for x in human.shift],
                              "expected_offset_mm": [1000.0 * x for x in EXPECTED_OFFSET_M],
                              "tolerance_mm": 1000.0 * OFFSET_TOL_M})
    return {"arrays": {"human_min_z": min_z, "human_pair_gap": gap}, "meta": meta,
            "inputs": inputs + [human.fit["path"]]}


# --------------------------------------------------------------------------- #
# Corpus calibration
# --------------------------------------------------------------------------- #
def confirmed_frames(base: capture.Capture) -> np.ndarray:
    """``[T, Z]`` Step 1's mat-confirmed frames: zone load > 50 N, attribution-visible, trustworthy
    markers and validity column 1 >= 0.9."""
    a = base.arrays
    known = capture.zone_known(a["marker_min_z"], a["marker_resid"])
    with np.errstate(invalid="ignore"):
        return ((a["mat_zone_load"] > capture.CALIB_LOAD_N) & a["attr_visible"] & known
                & (a["mat_valid_body"] >= capture.CALIB_BODY_GATE)[:, None])


def _quantiles(h: np.ndarray) -> dict:
    q = dict(zip(("p1", "p5", "p50", "p90", "p99"), np.percentile(h, [1, 5, 50, 90, 99]).tolist()))
    return {**q, "max": float(h.max())}


def calibrate(measured: list[dict], bases: dict) -> dict:
    """Ground thresholds from the mesh heights on mat-confirmed frames, pooled over ``CALIB_ZONES``;
    every zone's loaded distribution is reported beside them."""
    samples = {z: [] for z in ZONE_ORDER}
    clips = {z: set() for z in ZONE_ORDER}
    used, excluded = [], {}
    for m in measured:
        stem = m["meta"]["stem"]
        base = bases[stem]
        if not (m["meta"]["human_available"] and base.meta["mat_available"]):
            excluded[stem] = f"human {m['meta']['human_status']}, mat {base.meta['mat_status']}"
            continue
        used.append(stem)
        confirmed = confirmed_frames(base)
        for zi, z in enumerate(ZONE_ORDER):
            h = m["arrays"]["human_min_z"][confirmed[:, zi], zi]
            if h.size:
                samples[z].append(h)
                clips[z].add(stem)
    zones = {}
    for z in ZONE_ORDER:
        h = np.concatenate(samples[z]) if samples[z] else np.zeros(0)
        zones[z] = {"n_frames": int(h.size), "n_clips": len(clips[z]), **(_quantiles(h) if h.size else {})}
    pooled = np.concatenate([np.concatenate(samples[z]) for z in CALIB_ZONES if samples[z]] or [np.zeros(0)])
    if pooled.size == 0:
        raise ValueError("no mat-confirmed frame of a feet, hand or head zone to calibrate on")
    touch = float(np.percentile(pooled, 99)) + capture.TOUCH_MARGIN_M
    ground = {"touch_m": touch, "separation_m": touch + SEPARATION_GAP_M}
    pairs = {"touch_m": PAIR_TOUCH_M, "separation_m": PAIR_SEPARATION_M, "cap_m": PAIR_CAP_M, "seam_m": SEAM_M,
             "names": list(PAIR_NAMES)}
    rule = {"frames": "mat-confirmed as capture v1: load_n > {}, attr_visible, markers trusted, column 1 >= {}"
                      .format(capture.CALIB_LOAD_N, capture.CALIB_BODY_GATE),
            "pooled_zones": list(CALIB_ZONES), "touch": f"pooled p99 + {capture.TOUCH_MARGIN_M} m, every zone",
            "separation": f"touch + {SEPARATION_GAP_M} m", "zone_of_vertex": "dominant skinning joint (JOINT_ZONE)"}
    checks = [m["meta"]["verification"] for m in measured if m["meta"]["human_available"]]
    z_mm = [v["offset_mm"][2] for v in checks]
    return {"store": f"capture/{STORE_VERSION}", "rule": rule, "ground": ground, "pairs": pairs,
            "pooled": {"n_frames": int(pooled.size), **_quantiles(pooled)}, "zones": zones,
            "verification": {"clips": len(checks), "attach": checks[0]["attach"], "offset_z_mm": [min(z_mm), max(z_mm)],
                             "offset_xy_max_mm": max(abs(x) for v in checks for x in v["offset_mm"][:2]),
                             "nonfinger_median_mm_max": max(v["nonfinger_median_mm"] for v in checks),
                             "nonfinger_share_under_2mm_min": min(v["nonfinger_share_under_2mm"] for v in checks),
                             "finger_median_mm_max": max(v["finger_median_mm"] for v in checks)},
            "corpus": {"stems": used, "excluded": excluded},
            "id": ids.sha256_json({"rule": rule, "ground": ground, "pairs": pairs, "joint_zone": JOINT_ZONE})}


def write_calibration(cal: dict, measured: list[dict], path: Path = CALIBRATION_PATH) -> dict:
    inputs = sorted({p for m in measured for p in m["inputs"]} | {capture.CALIBRATION_PATH}, key=str)
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), **cal}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing; run `-m reference_curation.human_mesh --all` first")
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Store v2
# --------------------------------------------------------------------------- #
def _base_identity(base: capture.Capture) -> dict:
    return {"store": base.meta["store"], "generator": base.meta["generator"]["sha256"],
            "calibration": base.meta["calibration"]["id"]}


def _paths(stem: str, store_dir: Path) -> tuple[Path, Path]:
    return Path(store_dir) / f"{stem}.npz", Path(store_dir) / f"{stem}.json"


def build(stem: str, calibration: dict | None = None, store_dir: Path = STORE_DIR, measured: dict | None = None,
          base: capture.Capture | None = None) -> capture.Capture:
    """Measure ``stem``, apply the calibration, write ``<store_dir>/<stem>.{npz,json}`` and return
    the v1 record extended with the human arrays."""
    cal = load_calibration() if calibration is None else calibration
    base = capture.load(stem) if base is None else base
    m = measure(stem) if measured is None else measured
    T = base.meta["num_frames"]
    meta_h = dict(m["meta"])
    if meta_h["human_available"] and meta_h["num_frames"] != T:
        raise ValueError(f"{stem}: MoSh fit has {meta_h['num_frames']} frames, the motion {T}")
    if meta_h["human_available"]:
        min_z, gap = m["arrays"]["human_min_z"], m["arrays"]["human_pair_gap"]
        known = capture.zone_known(base["marker_min_z"], base["marker_resid"])
    else:
        min_z, gap = np.full((T, len(ZONE_ORDER)), np.nan), np.full((T, len(PAIRS)), np.nan)
        known = np.zeros((T, len(ZONE_ORDER)), dtype=bool)
    ground_state, pair_state = states(min_z, gap, known, cal["ground"])
    arrays = {"human_min_z": min_z.astype(np.float32), "human_ground_state": ground_state,
              "human_pair_gap": gap.astype(np.float32), "human_pair_state": pair_state}
    counts = {"ground": {z: {s: int((ground_state[:, zi] == v).sum()) for s, v in (("contact", 1), ("separated", 0),
                                                                                  ("unknown", -1))}
                         for zi, z in enumerate(ZONE_ORDER)},
              "pairs_in_contact": {n: int((pair_state[:, k] == 1).sum()) for k, n in enumerate(PAIR_NAMES)
                                   if (pair_state[:, k] == 1).any()}}
    meta = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, m["inputs"]), "store": f"capture/{STORE_VERSION}",
            **meta_h, "recording_id": ids.recording_id(stem), "fps": base.meta["fps"], "num_frames": T,
            "zone_order": list(ZONE_ORDER), "pair_order": list(PAIR_NAMES), "base": _base_identity(base),
            "calibration": {"id": cal["id"], "path": cal.get("path")},
            "thresholds": {"ground": cal["ground"], "pairs": {k: cal["pairs"][k] for k in ("touch_m", "separation_m",
                                                                                          "cap_m", "seam_m")}},
            "counts": counts, "arrays": {k: {"unit": u, "meaning": d} for k, (u, d) in HUMAN_ARRAYS.items()}}
    npz_path, json_path = _paths(stem, store_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, indent=1) + "\n")
    return capture.Capture(meta, {**base.arrays, **arrays})


def load(stem: str, store_dir: Path = STORE_DIR, calibration: dict | None = None,
         base: capture.Capture | None = None) -> capture.Capture:
    """Capture store v2 for ``stem``: the v1 record plus the human arrays. Rebuilt if missing, or made
    by other code, another calibration or another v1 record."""
    cal = load_calibration() if calibration is None else calibration
    base = capture.load(stem) if base is None else base
    npz_path, json_path = _paths(stem, store_dir)
    if npz_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        if (meta.get("schema_version") == SCHEMA_VERSION and meta["generator"]["sha256"] == ids.sha256_file(__file__)
                and meta["calibration"]["id"] == cal["id"] and meta["base"] == _base_identity(base)):
            with np.load(npz_path) as npz:
                return capture.Capture(meta, {**base.arrays, **{k: npz[k] for k in npz.files}})
    return build(stem, cal, store_dir, base=base)


@contextlib.contextmanager
def _single_threaded_children():
    """Spawned workers read these at import, so each one runs single-threaded BLAS."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: "1" for k in keys})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def measure_all(stems: list[str], workers: int = 1) -> tuple[list[dict], list[str]]:
    """``(measured, failures)`` over ``stems``, in ``workers`` spawned processes."""
    measured, failures = [], []
    if min(workers, len(stems)) <= 1:
        results = []
        for stem in stems:
            try:
                results.append((stem, measure(stem), None))
            except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
                results.append((stem, None, exc))
    else:
        with _single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {stem: pool.submit(measure, stem) for stem in stems}
            results = [(stem, f.result() if f.exception() is None else None, f.exception())
                       for stem, f in futures.items()]
    for stem, m, exc in results:
        if exc is None:
            measured.append(m)
        else:
            failures.append(f"{stem}: {type(exc).__name__}: {exc}")
    return measured, failures


def build_all(stems: list[str], store_dir: Path = STORE_DIR, calibration_path: Path | None = CALIBRATION_PATH,
              workers: int = 1) -> tuple[dict, list[capture.Capture], list[str]]:
    """Measure every clip, calibrate on them, build the store. ``(calibration, records, failures)``."""
    bases = {stem: capture.load(stem) for stem in stems}
    measured, failures = measure_all(stems, workers)
    cal = calibrate(measured, bases)
    if calibration_path is not None:
        cal = write_calibration(cal, measured, calibration_path)
        cal["path"] = ids.display_path(calibration_path)
    records = []
    for m in measured:
        try:
            records.append(build(m["meta"]["stem"], cal, store_dir, measured=m, base=bases[m["meta"]["stem"]]))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{m['meta']['stem']}: {type(exc).__name__}: {exc}")
    return cal, records, failures


def marker_agreement(records: list[capture.Capture]) -> dict:
    """Mesh against marker touch state on the frames and zones both decide (the admitted extremities)."""
    out = {}
    for z in CALIB_ZONES:
        zi = ZONE_ORDER.index(z)
        agree = total = 0
        for r in records:
            mk, hm = r["ground_state"][:, zi], r["human_ground_state"][:, zi]
            both = (mk >= 0) & (hm >= 0)
            agree += int((mk[both] == hm[both]).sum())
            total += int(both.sum())
        out[z] = {"frames": total, "agree": round(agree / total, 4) if total else None}
    agree = sum(v["agree"] * v["frames"] for v in out.values() if v["frames"])
    frames = sum(v["frames"] for v in out.values())
    return {"zones": out, "all": round(agree / frames, 4) if frames else None, "frames": frames}


def summarize(cal: dict, records: list[capture.Capture], failures: list[str], seconds: float) -> dict:
    available = [r for r in records if r.meta["human_available"]]
    contact_pairs = {}
    for r in available:
        for name, n in r.meta["counts"]["pairs_in_contact"].items():
            contact_pairs[name] = contact_pairs.get(name, 0) + n
    return {"clips": len(records), "frames": sum(r.meta["num_frames"] for r in records), "seconds": round(seconds, 1),
            "human_unavailable": {r.meta["stem"]: r.meta["human_error"] for r in records
                                  if not r.meta["human_available"]},
            "ground": cal["ground"], "verification": cal["verification"],
            "marker_agreement": marker_agreement(available),
            "pair_contact_frames": dict(sorted(contact_pairs.items(), key=lambda kv: -kv[1])), "failures": failures}


# --------------------------------------------------------------------------- #
# Render layer
# --------------------------------------------------------------------------- #
def render_layer(stem: str, rgba: tuple | None = None):
    """``render.HumanMesh`` of the clip's registered body, or ``None`` without a readable fit."""
    from reference_curation import render  # imports mujoco: keep it out of the measuring workers

    human, _, _ = load_human(stem)
    if human is None:
        return None
    kw = {} if rgba is None else {"rgba": tuple(rgba)}
    return render.HumanMesh(human.model.faces, human.vertices, **kw)


def render_frame(stem: str, frame: int, zones=(), strip=None, markers: bool = True,
                 out_dir: Path = ids.OUTPUT_ROOT / "renders") -> list[Path]:
    """Render one frame's sheets with the human mesh (not blind: for looking at a frame by eye)."""
    from reference_curation import render

    clip = render.load_clip(stem)
    layer = render_layer(stem)
    if layer is None:
        raise ValueError(f"{stem}: no readable MoSh fit, so no human mesh")
    sheets = render.plan(clip, frame, tuple(zones), strip or None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, ((image, _), sheet) in enumerate(zip(render.render_sheets(clip, sheets, human_mesh=layer, markers=markers),
                                                sheets), 1):
        paths.append(out / f"{stem}@{frame}_{k}_{sheet.kind}_human.png")
        render.save_png(image, paths[-1])
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="calibrate on the manifest's clips and build them all")
    what.add_argument("--stem", nargs="+", help="build these x0 clips with the committed calibration")
    ap.add_argument("--render", action="store_true", help="with one --stem and --frame: render it with the mesh")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--zones", nargs="*", default=[], help="--render: zones that also get a floor inset")
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--manifest", type=Path, default=ids.DEFAULT_MANIFEST)
    ap.add_argument("--store-dir", type=Path, default=STORE_DIR)
    ap.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--max-seconds", type=float, default=600.0, help="fail a full build slower than this")
    args = ap.parse_args(argv)

    start = time.time()
    if args.render:
        if not args.stem or len(args.stem) != 1 or args.frame is None:
            print("FAILED --render needs exactly one --stem and a --frame", file=sys.stderr)
            return 1
        try:
            paths = render_frame(args.stem[0], args.frame, args.zones, markers=not args.no_markers)
        except Exception as exc:  # noqa: BLE001 -- one line, non-zero exit
            print(f"FAILED {args.stem[0]}@{args.frame}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"human mesh: {args.stem[0]}@{args.frame}: {len(paths)} sheets in {time.time() - start:.1f} s -> "
              f"{ids.display_path(paths[0].parent)}")
        return 0
    if args.stem:
        cal = load_calibration(args.calibration)
        cal["path"] = ids.display_path(args.calibration)
        measured, failures = measure_all(args.stem, args.workers)
        records = []
        for m in measured:
            try:
                records.append(build(m["meta"]["stem"], cal, args.store_dir, measured=m))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{m['meta']['stem']}: {type(exc).__name__}: {exc}")
    else:
        cal, records, failures = build_all(ids.manifest_stems(args.manifest), args.store_dir, args.calibration,
                                           args.workers)
    seconds = time.time() - start
    if args.all and seconds > args.max_seconds:
        failures.append(f"took {seconds:.0f} s > budget {args.max_seconds:.0f} s")
    summary = summarize(cal, records, failures, seconds)
    if args.all:
        record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [args.manifest, args.calibration]),
                  "calibration_id": cal["id"], **summary}
        (Path(args.store_dir) / "_summary.json").write_text(json.dumps(record, indent=1) + "\n")
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    v, g, agree = cal["verification"], cal["ground"], summary["marker_agreement"]["all"]
    print(f"capture {STORE_VERSION} (human mesh): {len(records)} clips, {summary['frames']} frames in {seconds:.1f} s; "
          f"no fit: {len(summary['human_unavailable'])}; "
          f"offset z {v['offset_z_mm'][0]:.2f}-{v['offset_z_mm'][1]:.2f} mm; "
          f"markers: non-finger median <= {v['nonfinger_median_mm_max']:.2f} mm, share >= "
          f"{v['nonfinger_share_under_2mm_min']:.2f}, fingers <= {v['finger_median_mm_max']:.2f} mm; touch "
          f"{100 * g['touch_m']:.2f} / separation {100 * g['separation_m']:.2f} cm; agrees with the markers on "
          f"{'-' if agree is None else f'{100 * agree:.2f} %'} of decided extremity frames; {len(failures)} failures "
          f"-> {ids.display_path(args.store_dir)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
