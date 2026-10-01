# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The small retarget on plant v2 (BodyFix Step 3, part 2): the performer's own motion on her own skeleton,
inside the plant's joint box, its supports realised, not sliding, and without new jerk.

Why it is small
---------------
Step 8 (``retarget.py``) had to re-pose the shipped references because they copied the performer's angles onto
a body whose legs were 5 cm short and whose head collider sat 10 cm short of the crown. Plant v2 is her
skeleton (``build_subject_plant_v2.py``): the fit written on it (``fit_writer``) already puts every body origin
on her SMPL-X joint, the crowns within 2 cm of the floor, and 94 % of the labelled supports within 2 cm. What is
left (BodyFix "What Step 1 found"):

* **the joint box**: 26,532 of 81,512 frames ask for a coordinate more than 1 deg past PhysX's hard box, mostly
  the right toe's 40-71 deg roll (21,241 frames more than 2 deg past) and the right ankle's inversion (2,875): the
  fit's right-foot supination, not anatomy;
* **local floor penetration**: rigid colliders cannot compress like her skin, so 44 % of the human's touching
  zone-frames sit more than 5 mm in the floor (her skin: 15.6 %), standing feet by -1.0 cm (median), hands by
  -1.1 / -1.3 cm (the wrist box is 16 deg off her flat palm, the right foot box 11 deg off her flat sole);
* **soft supports** the commensurate colliders do not reach: the pelvis sphere (her buttocks, 7 supports) and
  the trunk spheres (her upper back in Plow, Bridge and Shoulderstand, 15 supports) float 3-7 cm;
* **self overlap** where her skin touches and the boxes are wider than her flesh (feet together, palms
  together): 79,943 pair-frames deeper than 1 cm.

What is solved
--------------
Step 8's least-squares problem (``retarget.Problem`` ... ``retarget.solve``: per frame the root offset, a root
rotation perturbation and the 69 exp-map joint coordinates; Levenberg-Marquardt with analytic Jacobians and one
banded solve per step), with these changes:

* **The anchor is her fit on plant v2** (the ``fit_writer`` motion), not a reference on another body. Every
  body stays near its position there, the joint coordinates near hers, hands, feet and head near her orientation.
* **The joint box is a hard bound** (``bounds``: the MJCF range less ``LIMIT_MARGIN_RAD``, intersected with the
  edit budget around her coordinates clipped into it).
* **Supports** (``support``/``floor``/``off``) are Step 8's support plan read on her mesh, two-sided at
  ``CLEARANCE_M``: a support she touches rests on the floor, a part she holds up stays up, nothing goes below
  ``FLOOR_MIN_M``. This is a penetration correction made **locally** by the solve (a sunk foot is lifted by its
  own leg) -- never a per-frame shift of the whole body, which re-introduced 2-5 cm floats on 19 % of supports.
  **The head is no longer refused**: plant v2's head sphere reaches the crown.
* **No slide (TODO A1)** (``vel``): every support point targeted on two consecutive frames moves horizontally by
  what the same material point of her fit moves. Her own pivots and rolls survive; a slide the edit would add
  costs. The term couples frame t with t+1, so the normal equations carry the cross-frame blocks
  (``normal_equations``; Step 8's band already spans two frames).
* **Balance margin tied to the support polygon, on every quasi-static frame (TODO A2)** (``balance``): wherever
  her COM moves slower than ``QS_SPEED_M_S`` on at least three targeted support points -- not only inside the
  labels' hold windows, so lift-offs are covered -- the plant's COM lies ``max(BALANCE_MARGIN_M,
  MARGIN_SHARE x inradius)`` inside the targeted supports' hull (the inradius of the initial pose's hull, fixed
  per frame).
* **Flatness where her sole or palm is flat** (``flat``): on the frames the support plan rests a box face flat
  (all four of its corners touch: each corner's human reading within the touch band), the face's outward normal
  points straight down. This is what turns the right foot's supination and the 16 deg wrist box flat; the toes'
  resting face is chosen with her toe roll removed (``face_rotations``), and the toes are left out of ``ends``,
  so a clipped toe roll is not carried into the ankle.
* **Body-body contacts.** A requested pair (labels v1.1's configured and the reviewer's critical) is closed only
  when her fit holds it within ``PAIR_REACH_M`` on plant v2. The guard against self-penetration keeps every pair
  she holds apart within ``PEN_TOL_M`` (``pen``); a pair her mesh shows in contact may not get deeper than in her
  fit, and is asked apart only by a light preference (``pen_soft``) that the anchor outvotes where separating
  would re-pose a limb (her flesh compresses where the colliders cannot).
* **No new jerk (TODO A3)**: a solution with frames on which a body accelerates past ``SPIKE_ACC`` where her fit
  does not (``spike_frames``) is solved again with *every* contact term brought in by continuation
  (``CONTACT_TERMS``; Step 8 continued only the pair and penetration terms) under a stiffer smoothness; if jerks
  remain, the smoothness is stiffened only around them (``smooth_locally``: x10, x30, x100 within 0.25 s) and the
  clip solved again; the solution with the fewest jerk frames is kept.

The output is every ``.motion`` field written by ``fit_writer.write_motion`` (the converter's FK path, with
``plant_sha256``) from the solved coordinates, a per-frame lineage, and the exit metrics of the card
(``clip_metrics``, ``corpus_metrics``, ``acceptance``), under ``output/reference_curation/retarget_v2/<rid>/``
and ``data/reference_curation/retarget_v2/<rid>/``.

The plant: everything plant-dependent runs inside ``on_plant()`` (``mosh_replay.use_plant``), which points
``retarget``'s cached skeleton, pairs and mass model at plant v2 in this process only. The capture stores are
read **outside** it (``human_evidence``) and only for their human-side arrays: a stale store rebuilt under plant v2
would write v2 avatar fields into the v1 records.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.retarget_v2 --pilot
    ... --all --workers 16
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import json
import math
import multiprocessing
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from extract_contact_configs import ZONE_ORDER, ZONES
from reference_curation import fit_writer as fw
from reference_curation import human_mesh as hm
from reference_curation import ids
from reference_curation import mosh_replay as mr
from reference_curation import retarget as rt
from protomotions.utils import plant_identity  # noqa: E402  (ids put the repo on sys.path)

MODULE = "reference_curation.retarget_v2"
SCHEMA_VERSION = 1
RETARGET_VERSION = "v2"
F64 = torch.float64
NV, BAND = rt.NV, rt.BAND
ZI = rt.ZI
PLANT = fw.PLANT

# Step 8's floor convention and support plan, unchanged.
CLEARANCE_M = rt.CLEARANCE_M
FLOOR_MIN_M = rt.FLOOR_MIN_M
OFF_FLOOR_M = rt.OFF_FLOOR_M
PAIR_TARGET_M = rt.PAIR_TARGET_M
# A requested body-body contact (labels v1.1's configured and the reviewer's critical pairs) is closed only when
# her fit already holds it within this on plant v2; farther is the commensurate colliders' geometry and is
# reported (``geometry_incompatible``), not forced. Measured on the 96 (clip, pair) requests: the gaps are bimodal,
# pressed contacts overlap 4-6 cm (thighs in upper arms, Tree's foot in the thigh: her soft tissue compresses) and
# thigh-trunk contacts sit 4-9 cm apart (the trunk spheres do not fill her belly); closing those moved Tree -a's
# knee 15 cm. Step 8's 6 cm reach was for a body whose limbs were the wrong length. Which body-body contacts are
# critical is Step 4's decision (TODO B6).
PAIR_REACH_M = 0.02
# No body pair the plant collides overlaps by more than this (Step 8 allowed 5 mm, and the corpus then kept 1,043
# pair-frames over 1 cm, many of them at exactly 5 mm: a sole on a thigh, a hand on a knee, palms together). PhysX
# keeps the colliders apart (rest offset 0), so a reference asking for overlap asks for what the plant cannot do.
# Measured on six clips (Tree -a/-b, Eagle, Side Crow -b, Garland, Firefly -b), 1 mm leaves <= 2.7 mm on the
# five that reach it, with no new jerk and the worst body moved 6.3 -> 6.7 cm (Tree) and 16.5 -> 14.7 cm (Eagle).
PEN_TOL_M = 0.001
# The self-penetration guard is split where her mesh says the two zones touch (``soft_floor``): there her flesh is
# compressed and the commensurate colliders overlap by what it compresses (Firefly's thighs 5.5 cm into her upper
# arms, Tree's foot 5 cm into the thigh, Eagle's wrapped legs 2-4 cm). The hard guard (``pen``) only forbids such
# a pair to get deeper than in her fit; a preference (``pen_soft``: weight 10, 1 cm scale, a tenth of the hard
# guard's stiffness) asks it apart to ``PEN_TOL_M``, and the anchor can outvote it where separating would re-pose
# a limb. Measured on the alternatives: as a hard guard it slid Firefly -b's legs to the other side of the arms
# (50 jerk frames, a toe 44 cm); a fixed 2 cm push on every pair summed through Eagle's wrap to a 19 cm toe over
# the whole hold; at weight 1 it left Tree's sole 1.5 cm in the thigh, which PhysX's reset kicked out at 0.78 m/s;
# at 10 Tree, Eagle and Side Crow -b clear (<= 1.05 cm) and the leg-on-arm balances halve (4.5 -> 2.3 cm). A pair
# she holds apart is cleared fully. The card's acceptance is the same rule: no new overlaps.
LIMIT_MARGIN_RAD = rt.LIMIT_MARGIN_RAD
# The contract's floor: no collision surface deeper than this. Where supports conflict on the plant's geometry
# (Cobra -a's prone leg cannot rest thigh, shin and dorsum each at +5 mm) the solve leaves a few millimetres in
# the floor (worst 0.53 cm); a 10x stiffer floor term floated both thighs 2.5 cm on 530 frames instead. BodyFix
# Step 2 measured PhysX lifting a pose 1-2 cm in the floor to its surface at <= 0.05 m/s with nothing thrown.
FLOOR_CONTRACT_M = -0.01
MIN_RUN = rt.MIN_RUN
RAMP = rt.RAMP
# Hands, feet and head keep her orientation; the toes do not (their roll past the box is the fit's supination).
END_BODIES = ("L_Ankle", "R_Ankle", "L_Wrist", "L_Hand", "R_Wrist", "R_Hand", "Head")
WEIGHTS = {"support": 10.0, "floor": 10.0, "off": 10.0, "pair": 10.0, "pen": 10.0, "pen_soft": 10.0, "balance": 10.0,
           "flat": 10.0, "vel": 10.0, "dof": 1.0, "root": 1.0, "ends": 1.0, "anchor": 1.0, "smooth": 1.0}
SCALES = {**rt.SCALES, "flat": 0.05, "vel": 0.001, "pen_soft": 0.01}
CONTACT_TERMS = ("support", "floor", "off", "pair", "pen", "pen_soft", "flat", "balance")
# Balance (TODO A2).
BALANCE_MARGIN_M = rt.BALANCE_MARGIN_M     # the floor of the margin
MARGIN_SHARE = 0.25                        # ... else this share of the support polygon's inradius
QS_SPEED_M_S = 0.15                        # quasi-static: her COM's horizontal speed (the labels' quasi_static_v)
QS_SMOOTH_FRAMES = 11                      # ... smoothed over this many frames
QS_STABLE_FRAMES = 15                      # ... and the targeted support zones unchanged this many frames either side:
                                           # a slow weight shift onto one foot is not static on it (the first
                                           # Garland solve dragged the pelvis 8 cm over the lifting foot's partner)
# The edit budget, as bounds (Step 8's).
BUDGET_JOINT_RAD = rt.BUDGET_JOINT_RAD
BUDGET_ROOT_M = rt.BUDGET_ROOT_M
BUDGET_ROOT_RAD = rt.BUDGET_ROOT_RAD
ITERS = 60
SPIKE_ACC = rt.SPIKE_ACC
GENTLE_SMOOTH = rt.GENTLE_SMOOTH
GENTLE_STAGES = rt.GENTLE_STAGES
# TODO A3's local post-pass: around every frame still jerking after the gentle solve, the smoothness weight is
# multiplied by each factor in turn (within LOCAL_HALF frames either side) and the clip solved again from there.
LOCAL_HALF = 15
LOCAL_FACTORS = (10.0, 30.0, 100.0)


# --------------------------------------------------------------------------- #
# The plant, the human and the labels
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def on_plant(plant: str = PLANT):
    """``retarget``'s cached plant (skeleton, pairs, ancestors, mass model), ``capture``'s and
    ``static_hold_lp``'s, pointed at ``plant`` for the duration (``mosh_replay.use_plant``), checked, and the
    skeleton yielded."""
    xml, flat = fw.plant_paths(plant)
    with mr.use_plant(xml, flat):
        sk = rt.skeleton()
        ref = mr.skeleton_for(xml)
        if not (torch.equal(sk.offsets, ref.offsets) and torch.equal(sk.lower, ref.lower)
                and torch.equal(sk.upper, ref.upper)):
            raise RuntimeError(f"retarget.skeleton() is not plant {plant}'s inside use_plant")
        mass, _ = rt.mass_model()
        if abs(float(mass.sum()) - 74.0) > 1e-3:
            raise RuntimeError(f"the mass model in use totals {mass.sum():.4f} kg")
        yield sk


HUMAN_FIELDS = ("human_floor_state", "human_floor_z", "human_pair_state", "mat_cop", "mat_total", "mat_valid_cov")


def human_evidence(stem: str) -> dict:
    """The capture store's human-side arrays (v3 ``human_floor_*``, v2 ``human_pair_state``; plant-free) and its
    identity. Refused under any plant but v1: a missing or stale store is rebuilt by ``sources.load``, and a
    rebuild under plant v2 would write v2 avatar fields into the v1 records (BodyFix Step 2, "Plant identity")."""
    from reference_curation import sources

    if plant_identity.name_of(plant_identity.sha256(ids.MJCF)) not in ("v1",):
        raise RuntimeError("load the capture stores outside on_plant() and without REFERENCE_PLANT=v2")
    rec = sources.load(stem)
    if not rec.meta.get("human_available"):
        raise ValueError(f"{stem}: the capture store has no human mesh")
    return {"arrays": {k: np.asarray(rec[k]) for k in HUMAN_FIELDS}, "identity": sources.identity(rec)}


def default_labels_dir(manifest: Path = ids.DEFAULT_MANIFEST) -> Path:
    """Labels v1.1 of ``manifest`` (exactly one folder): the human-side decisions this step reads (windows,
    configured supports and pairs, the reviewer's critical pairs); never its avatar evidence (plant v1)."""
    dirs = sorted((ids.DATA_ROOT / "labels").glob(f"{Path(manifest).stem}.labels_v1_1.*"))
    if len(dirs) != 1:
        raise FileNotFoundError(f"expected one labels v1.1 folder for {Path(manifest).stem}, found {len(dirs)}")
    return dirs[0]


def load_labels(labels_dir: Path | None = None) -> dict:
    from reference_curation import statics

    return statics.load_labels(default_labels_dir() if labels_dir is None else labels_dir)


# --------------------------------------------------------------------------- #
# Run masks and ramps (Step 8's, with the lengths as parameters)
# --------------------------------------------------------------------------- #
def clean_runs(mask: np.ndarray, min_run: int = MIN_RUN) -> np.ndarray:
    """``[T, K]``: per column, gaps shorter than ``min_run`` filled, then runs shorter than it dropped."""
    out = mask.copy()
    for k in np.nonzero(mask.any(0))[0]:
        col = out[:, k]
        for fill in (True, False):
            for r0, r1 in rt._runs(col if not fill else ~col):
                if r1 - r0 < min_run and (not fill or (r0 > 0 and r1 < len(col))):
                    col[r0:r1] = fill
        out[:, k] = col
    return out


def ramp_weights(mask: np.ndarray, ramp: int = RAMP) -> np.ndarray:
    """``[T, K]`` in [0, 1]: 1 inside each run, a raised cosine over ``ramp`` frames centred on every onset and
    release."""
    T = mask.shape[0]
    half = ramp / 2
    t = np.arange(T) + 0.5
    w = np.zeros(mask.shape)
    for k in np.nonzero(mask.any(0))[0]:
        col = np.zeros(T)
        for r0, r1 in rt._runs(mask[:, k]):
            rise = np.ones(T) if r0 == 0 else np.clip((t - (r0 - half)) / ramp, 0, 1)
            fall = np.ones(T) if r1 == T else np.clip(((r1 + half) - t) / ramp, 0, 1)
            col = np.maximum(col, 0.5 - 0.5 * np.cos(np.pi * np.minimum(rise, fall)))
        w[:, k] = col
    return w


# --------------------------------------------------------------------------- #
# What her mesh asks of every candidate point, frame by frame (pure)
# --------------------------------------------------------------------------- #
ROLL_FREE = ("L_Toe", "R_Toe")     # bodies whose resting face is chosen with their roll removed (``face_rotations``)


def face_rotations(sk: rt.Skeleton, root_pos: torch.Tensor, root_rot: torch.Tensor, dof: torch.Tensor) -> np.ndarray:
    """``[T, B, 3, 3]`` the body rotations ``support_plan`` chooses resting faces from: FK of ``dof`` (her fit
    clipped into the box) with the toes' roll coordinate (exp-map x, about the foot's long axis) set to 0. Her fit
    rolls the right toes 40-71 deg (the supination): with it, the clipped roll plus the foot's own put the toe
    box's side face down, and forcing that face flat swung Warrior II's ankle 14 cm. Without it the toe box
    follows the foot: its sole standing, its dorsum prone (Cobra), its side when the foot lies on its side (Side
    Plank), and its pitch (a raised heel on extended toes) is kept."""
    d = dof.clone()
    for b in ROLL_FREE:
        d[:, 3 * (sk.names.index(b) - 1)] = 0.0
    with torch.no_grad():
        return rt.fk(sk, root_pos, root_rot, d)[1].numpy()


def support_plan(sk: rt.Skeleton, corr: rt.Correspondence, heights: np.ndarray, state: np.ndarray, touch_m: float,
                 rot_face: np.ndarray) -> dict:
    """``retarget.support_plan`` with three changes, plus the frames on which each box rests flat (``flat``:
    ``frames``, ``body``, ``axis``, ``sign``, ``w``):

    * **No head refusal**: plant v2's head sphere reaches the crown.
    * **A face rests flat only while all four of its corners touch** (each corner's human reading at most
      ``touch_m + RELEASE_M``, the touch band's release): "only where her sole or palm is flat" (the card). Step
      8's rule (the readings spanning at most ``FLAT_SPREAD_M``) was sized for the 16 cm foot box; on the 7 cm
      palm box it held a peeling palm flat up to ~30 deg (Crow -a: the wrist turned 34 deg against her, the
      elbow 10 cm). Measured on planted feet and palms (Plank, Crow, Downward Dog, Warrior II, Tree), the palms
      and fingers qualify on 94-98 % of their down frames, the foot boxes on 80-100 % (the MoSh heel's 2-3 cm
      stays inside the band).
    * The resting face of a box is the one facing most nearly down in ``rot_face`` (``face_rotations``: her fit
      clipped into the box, the toes' roll removed)."""
    T, K = heights.shape[0], len(sk.cand_body)
    target = np.zeros((T, K), bool)
    off = np.zeros((T, K), bool)
    release = touch_m + rt.RELEASE_M
    flat_rows = []
    for zi in range(len(ZONE_ORDER)):
        on = state[:, zi] == 1
        off[np.ix_(state[:, zi] == 0, sk.zone_cands[zi])] = True
        for b in sorted(set(sk.cand_body[sk.zone_cands[zi]])):
            if b in corr.face_sets:
                faces = [(f, corr.face_sets[b][f]) for f in rt.BOX_FACES]
                h = np.stack([heights[:, [s for _, s in ent]] for _, ent in faces], 1)       # [T, 6, 4]
                face = rt.mode_filter(rt.box_face_normals_z(sk, rot_face, b).argmin(1), rt.MODE_FRAMES)
                hf = h[np.arange(T), face]
                lo = hf.min(1)
                down = rt.hysteresis(lo, touch_m, release) & on
                flat = hf.max(1) <= release
                for t in np.nonzero(down)[0]:
                    for j, (k, _) in enumerate(faces[face[t]][1]):
                        if flat[t] or hf[t, j] <= lo[t] + rt.EDGE_BAND_M:
                            target[t, k] = True
                flat_rows.append((b, down & flat, face))
            else:
                for k in np.nonzero(sk.cand_body == b)[0]:
                    down = rt.hysteresis(heights[:, corr.point_sets[int(k)]], touch_m, release) & on
                    target[down, k] = True
    target, off = clean_runs(target), clean_runs(off)
    fl = {"frames": [], "body": [], "axis": [], "sign": [], "w": []}
    for b, mask, face in flat_rows:
        w = ramp_weights(clean_runs(mask[:, None]))[:, 0]
        f = np.nonzero(w > 0)[0]
        fl["frames"].append(f)
        fl["body"].append(np.full(len(f), b))
        fl["axis"].append(np.array([rt.BOX_FACES[i][0] for i in face[f]], int))
        fl["sign"].append(np.array([rt.BOX_FACES[i][1] for i in face[f]], float))
        fl["w"].append(w[f])
    fl = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in fl.items()}
    fl["frames"], fl["body"], fl["axis"] = (fl[k].astype(int) for k in ("frames", "body", "axis"))
    return {"target": target, "off": off, "target_w": ramp_weights(target), "off_w": ramp_weights(off), "flat": fl}


def velocity_plan(sk: rt.Skeleton, target_w: np.ndarray, pos0: torch.Tensor, rot0: torch.Tensor) -> dict:
    """The no-slide rows (TODO A1): every candidate targeted on frames t and t + 1 (weight: the smaller ramp),
    with the horizontal displacement ``d0`` of the same material point in her fit."""
    with torch.no_grad():
        pts0 = rt.candidate_points(sk, pos0, rot0).numpy()
    w = np.minimum(target_w[:-1], target_w[1:])
    f, k = np.nonzero(w > 0)
    return {"frames": f, "cand": k, "w": w[f, k], "d0": torch.as_tensor(pts0[f + 1, k, :2] - pts0[f, k, :2])}


def com_xy(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """``[T, 2]`` whole-body COM over the plant's mass model (``retarget.mass_model``)."""
    mass, centre = rt.mass_model()
    m = torch.as_tensor(mass, dtype=pos.dtype)
    com_b = pos + (rot @ torch.as_tensor(centre, dtype=pos.dtype)[..., None]).squeeze(-1)
    return ((m[None, :, None] * com_b).sum(1) / m.sum())[:, :2]


def hull_inradius(xy: np.ndarray) -> float:
    """The largest circle inside the convex hull of ``xy`` (its Chebyshev centre's radius; 0 if degenerate)."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(xy)
    except QhullError:
        return 0.0
    eq = hull.equations                               # n . x + d <= 0 inside, |n| = 1
    res = linprog([0.0, 0.0, -1.0], A_ub=np.c_[eq[:, :2], np.ones(len(eq))], b_ub=-eq[:, 2],
                  bounds=[(None, None), (None, None), (0, None)], method="highs")
    return float(res.x[2]) if res.status == 0 else 0.0


def stable_support(sk: rt.Skeleton, target_w: np.ndarray, half: int = QS_STABLE_FRAMES) -> np.ndarray:
    """``[T]`` bool: the set of zones with a targeted support point (weight > 0.5) is the same on every frame
    within ``half`` frames either side."""
    T = target_w.shape[0]
    zones = np.stack([(target_w[:, c] > 0.5).any(1) for c in sk.zone_cands], 1)            # [T, Z]
    code = zones.astype(np.int64) @ (1 << np.arange(zones.shape[1], dtype=np.int64))
    change = np.r_[False, code[1:] != code[:-1]]                                          # a change between t-1 and t
    cum = np.r_[0, np.cumsum(change)]
    lo, hi = np.clip(np.arange(T) - half, 0, T - 1), np.clip(np.arange(T) + half, 0, T - 1)
    return (cum[hi + 1] - cum[lo + 1]) == 0


def balance_plan(sk: rt.Skeleton, target_w: np.ndarray, pos0, rot0, pos_i, rot_i, fps: int) -> dict:
    """The balance rows (TODO A2): her quasi-static frames on the targeted supports (COM speed <=
    ``QS_SPEED_M_S``, smoothed; at least three targeted support points; the targeted zones unchanged within
    ``QS_STABLE_FRAMES``), ramped, and each frame's margin from the initial pose's support hull."""
    with torch.no_grad():
        com = com_xy(pos0, rot0).numpy()
        pts = rt.candidate_points(sk, pos_i, rot_i).numpy()
    T = len(com)
    v = np.gradient(com, axis=0) * fps if T > 1 else np.zeros_like(com)
    k = np.ones(QS_SMOOTH_FRAMES) / QS_SMOOTH_FRAMES
    speed = np.convolve(np.linalg.norm(v, axis=1), k, mode="same")
    qs = (speed <= QS_SPEED_M_S) & ((target_w > 0.5).sum(1) >= 3) & stable_support(sk, target_w)
    w = ramp_weights(clean_runs(qs[:, None]))[:, 0]
    frames = np.nonzero(w > 0)[0]
    margin = np.array([max(BALANCE_MARGIN_M, MARGIN_SHARE * hull_inradius(pts[f, target_w[f] > 0.5, :2]))
                       if (target_w[f] > 0.5).sum() >= 3 else BALANCE_MARGIN_M for f in frames])
    return {"frames": frames, "w": w[frames], "margin": margin, "speed": speed}


def soft_floor(sk: rt.Skeleton, pos0: torch.Tensor, rot0: torch.Tensor, pair_state: np.ndarray) -> dict:
    """The self-penetration guard: every pair the plant collides (``retarget.plant_pairs``), the gap each pair-frame
    must keep (``floor [T, P]``: ``-PEN_TOL_M``, or her fit's gap where it overlaps deeper and her mesh does not
    hold the two zones apart -- ``human_pair_state`` not 0; same or adjacent zones have no reading) and those soft
    pair-frames (``soft [T, P]``), on which ``pen_soft`` asks for ``-PEN_TOL_M``."""
    pairs = rt.plant_pairs()
    T = pos0.shape[0]
    index = np.full((sk.num_bodies, sk.num_bodies), -1)
    index[pairs[:, 0], pairs[:, 1]] = index[pairs[:, 1], pairs[:, 0]] = np.arange(len(pairs))
    floor = np.full((T, len(pairs)), -PEN_TOL_M)
    soft_tp = np.zeros((T, len(pairs)), bool)
    f, a, b, g = rt.near_body_pairs(sk, pos0, rot0, -PEN_TOL_M, pairs)
    col = {frozenset(n.split("+")): k for k, n in enumerate(hm.PAIR_NAMES)}
    zone = [rt.BODY_ZONE[n] for n in sk.names]
    held_apart = np.array([(frozenset((zone[i], zone[j])) in col
                            and pair_state[fi, col[frozenset((zone[i], zone[j]))]] == 0) for fi, i, j in zip(f, a, b)], bool)
    soft = ~held_apart
    floor[f[soft], index[a[soft], b[soft]]] = g[soft]
    soft_tp[f[soft], index[a[soft], b[soft]]] = True
    return {"pairs": pairs, "floor": -PEN_TOL_M, "floor_tp": floor, "soft_tp": soft_tp, "index": index,
            "soft_pair_frames": int(soft.sum()), "held_apart_pair_frames": int(held_apart.sum())}


# --------------------------------------------------------------------------- #
# One clip's problem
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    """Field names follow ``retarget.Problem`` so its ``unpack``, ``kinematics``, ``initial`` and ``fk`` apply."""

    stem: str
    fps: int
    root_pos0: torch.Tensor        # [T, 3] her pelvis (the fit motion's root)
    root_rot0: torch.Tensor        # [T, 3, 3]
    dof0: torch.Tensor             # [T, 69] her coordinates (nearest representative; may lie outside the box)
    pos0: torch.Tensor             # [T, B, 3] her fit on the plant
    rot0: torch.Tensor             # [T, B, 3, 3]
    plan: dict                     # support_plan (+ heights, head_inverted)
    close: dict                    # the requested body-body contacts (Step 8's items)
    pen: dict                      # the self-penetration guard
    requests: list
    vel: dict                      # velocity_plan
    balance: dict                  # balance_plan
    lower: torch.Tensor            # [69] the box less LIMIT_MARGIN_RAD
    upper: torch.Tensor
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))
    smooth_w: np.ndarray | None = None     # [T - 2] per-row multiplier of the smoothness (None: 1)

    @property
    def T(self) -> int:
        return self.dof0.shape[0]


def anchor_coordinates(fit_motion: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(root_pos, root_rot, dof)`` float64 of a ``fit_writer`` motion: the solve's anchor is the written file."""
    from protomotions.utils.rotations import quaternion_to_matrix

    rot = quaternion_to_matrix(fit_motion["rigid_body_rot"][:, 0].double(), w_last=True)
    return fit_motion["rigid_body_pos"][:, 0].double().clone(), rot, fit_motion["dof_pos"].double().clone()


def build_problem(stem: str, anns: dict, ev: dict, human: hm.Human, fit_motion: dict) -> Problem:
    """Everything the solve needs for one clip, on the plant in use (call inside ``on_plant``)."""
    sk = rt.skeleton()
    fps = int(fit_motion["fps"])
    root_pos0, root_rot0, dof0 = anchor_coordinates(fit_motion)
    T = dof0.shape[0]
    lower, upper = sk.lower + LIMIT_MARGIN_RAD, sk.upper - LIMIT_MARGIN_RAD
    with torch.no_grad():
        pos0, rot0 = rt.fk(sk, root_pos0, root_rot0, dof0)
        pos_i, rot_i = rt.fk(sk, root_pos0, root_rot0, torch.maximum(torch.minimum(dof0, upper), lower))
    state = ev["arrays"]["human_floor_state"]
    if state.shape[0] != T:
        raise ValueError(f"{stem}: the store has {state.shape[0]} frames, the fit motion {T}")
    corr = rt.correspondence(sk, human.model, human.fit["v_template"])
    heights = rt.human_heights(human, corr)
    touch = hm.load_calibration()["ground"]["touch_m"]
    plan = support_plan(sk, corr, heights, state, touch,
                        face_rotations(sk, root_pos0, root_rot0, torch.maximum(torch.minimum(dof0, upper), lower)))
    plan["heights"] = heights
    plan["head_inverted"] = rt.head_inverted(rot0.numpy(), sk)
    requests = rt.pair_requests(stem, anns, ev["arrays"]["human_pair_state"])
    items = {}
    for r in requests:
        if not len(r["frames"]):
            r["status"] = "no_human_contact"
            continue
        g, _, _ = rt.zone_pair_gaps(sk, pos0, rot0, *r["zones"], r["frames"])
        r["gap_before_cm"] = round(100 * float(np.median(g)), 2)
        if np.median(g) > PAIR_REACH_M:
            r["status"] = "geometry_incompatible"
            continue
        r["status"] = "closed"
        for f, w in zip(r["frames"].tolist(), r["w"].tolist()):
            key = ("+".join(r["zones"]), f)
            items[key] = max(items.get(key, 0.0), w)
    rows = [(k, f, a, b) for k, (c, f) in enumerate(sorted(items)) for a, b in rt.zone_pair_bodies(*c.split("+"))]
    keys = sorted(items)
    close = {"item": np.array([r[0] for r in rows], int), "frames": np.array([r[1] for r in rows], int),
             "body_a": np.array([r[2] for r in rows], int), "body_b": np.array([r[3] for r in rows], int),
             "w": np.array([items[k] for k in keys]), "contact": [c for c, _ in keys]}
    return Problem(stem, fps, root_pos0, root_rot0, dof0, pos0, rot0, plan, close,
                   soft_floor(sk, pos0, rot0, ev["arrays"]["human_pair_state"]), requests,
                   velocity_plan(sk, plan["target_w"], pos0, rot0),
                   balance_plan(sk, plan["target_w"], pos0, rot0, pos_i, rot_i, fps), lower, upper)


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
def _empty() -> dict:
    return {"frames": np.zeros(0, int), "r": torch.zeros(0, dtype=F64), "J": torch.zeros(0, NV, dtype=F64)}


def _point_jacobian(st: rt.State, frames, bodies, points: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    frames, bodies = np.asarray(frames), np.asarray(bodies)
    if len(frames) <= chunk:
        return rt.point_jacobian(st, frames, bodies, points)
    return torch.cat([rt.point_jacobian(st, frames[i:i + chunk], bodies[i:i + chunk], points[i:i + chunk])
                      for i in range(0, len(frames), chunk)])


def box_rotations(sk: rt.Skeleton) -> torch.Tensor:
    """``[B, 3, 3]`` every body's box orientation in its body frame (identity for a round geom)."""
    from scipy.spatial.transform import Rotation

    out = np.tile(np.eye(3), (sk.num_bodies, 1, 1))
    for i, b in enumerate(sk.names):
        g = sk.geoms[b][0]
        if g["type"] == "box":
            out[i] = Rotation.from_quat(g["quat"]).as_matrix()
    return torch.as_tensor(out, dtype=F64)


def face_normals(sk: rt.Skeleton, rot: torch.Tensor, frames, bodies, axis, sign) -> torch.Tensor:
    """``[N, 3]`` world outward normal of the given box faces."""
    rg = box_rotations(sk)
    frames, bodies = np.asarray(frames, int), np.asarray(bodies, int)
    R = rot[frames, bodies] @ rg[bodies]
    n = R[torch.arange(len(frames)), :, torch.as_tensor(np.asarray(axis, int))]
    return n * torch.as_tensor(np.asarray(sign, float), dtype=rot.dtype)[:, None]


def residuals(prob: Problem, x: torch.Tensor, jacobian: bool = False) -> dict:
    """Every scale-normalised, weight-rooted residual (``energy`` is their squared sum). Per block: ``frames``,
    ``r`` and with ``jacobian`` the ``[N, 75]`` rows ``J``; a block coupling two frames (``vel``) also carries
    ``frames2`` and ``J2`` (its rows on frame t + 1).

    ``support``, ``floor``, ``off``, ``pair``, ``pen``, ``anchor``: ``retarget.residuals``'s, on the plan above
    (``pen`` with ``soft_floor``'s per pair-frame floors). ``pen_soft``: her contact overlaps asked apart to
    ``-PEN_TOL_M``. ``ends``: ``END_BODIES`` keep her orientation. ``flat``, ``balance``, ``vel``: the module doc."""
    sk, s, w, plan = rt.skeleton(), SCALES, prob.weights, prob.plan
    st = rt.kinematics(prob, x)
    out = {}
    tw, ow = torch.as_tensor(plan["target_w"]), torch.as_tensor(plan["off_w"])
    ones = torch.ones_like(tw)
    for name, mask, level, scale, fade in (("support", tw > 0, CLEARANCE_M, s["support"], tw),
                                           ("floor", st.h < FLOOR_MIN_M, FLOOR_MIN_M, s["floor"], ones),
                                           ("off", (ow > 0) & (st.h < OFF_FLOOR_M), OFF_FLOOR_M, s["off"], ow)):
        f, k = torch.nonzero(mask, as_tuple=True)
        c = math.sqrt(w[name]) / scale * torch.sqrt(fade[f, k])
        blk = {"frames": f.numpy(), "r": c * (st.h[f, k] - level)}
        if jacobian:
            blk["J"] = c[:, None] * _point_jacobian(st, f.numpy(), sk.cand_body[k.numpy()], st.pts[f, k])[:, 2]
        out[name] = blk
    nf, na, nb, _ = rt.near_body_pairs(sk, st.pos, st.rot, prob.pen["floor"], prob.pen["pairs"])
    ki = prob.pen["index"][na, nb]
    pen = {"frames": nf, "body_a": na, "body_b": nb, "floor": prob.pen["floor_tp"][nf, ki]}
    soft = prob.pen["soft_tp"][nf, ki]
    pen_soft = {"frames": nf[soft], "body_a": na[soft], "body_b": nb[soft], "floor": np.full(int(soft.sum()), -PEN_TOL_M)}
    for name, items in (("pair", prob.close), ("pen", pen), ("pen_soft", pen_soft)):
        c = math.sqrt(w[name]) / s[name]
        if not len(items["frames"]):
            out[name] = _empty()
            continue
        g, pa, pb, u = rt.body_gaps(sk, st.pos, st.rot, items["frames"], items["body_a"], items["body_b"], witnesses=True)
        if name == "pair":
            order = np.lexsort((g.numpy(), items["item"]))
            first = order[np.r_[True, np.diff(items["item"][order]) != 0]]
            items = {k: (v[first] if isinstance(v, np.ndarray) and len(v) == len(g) else v) for k, v in items.items()}
            g, pa, pb, u = g[first], pa[first], pb[first], u[first]
        lim = torch.full_like(g, PAIR_TARGET_M) if name == "pair" else torch.as_tensor(items["floor"], dtype=F64)
        act = (g > lim) if name == "pair" else (g < lim)
        idx = torch.nonzero(act, as_tuple=True)[0]
        c = c * (torch.sqrt(torch.as_tensor(items["w"])[idx]) if name == "pair" else torch.ones(len(idx), dtype=F64))
        blk = {"frames": items["frames"][idx.numpy()], "r": c * (g[idx] - lim[idx])}
        if jacobian:
            fr = items["frames"][idx.numpy()]
            ja = rt.point_jacobian(st, fr, items["body_a"][idx.numpy()], pa[idx])
            jb = rt.point_jacobian(st, fr, items["body_b"][idx.numpy()], pb[idx])
            blk["J"] = c[:, None] * (u[idx][:, :, None] * (jb - ja)).sum(1)
        out[name] = blk
    ends = np.array([sk.names.index(b) for b in END_BODIES])
    f = np.repeat(np.arange(prob.T), len(ends))
    b = np.tile(ends, prob.T)
    rel = prob.rot0[f, b].transpose(-1, -2) @ st.rot[f, b]
    c = math.sqrt(w["ends"]) / s["ends"]
    lg = rt.so3_log(rel)
    blk = {"frames": np.repeat(f, 3), "r": c * lg.reshape(-1)}
    if jacobian:
        blk["J"] = c * (rt.right_jacobian_inv(lg) @ st.rot[f, b].transpose(-1, -2)
                        @ rt.rotation_jacobian(st, f, b)).reshape(-1, NV)
    out["ends"] = blk
    c = math.sqrt(w["anchor"]) / s["anchor"]
    f = np.repeat(np.arange(prob.T), sk.num_bodies)
    b = np.tile(np.arange(sk.num_bodies), prob.T)
    blk = {"frames": np.repeat(f, 3), "r": c * (st.pos - prob.pos0).reshape(-1)}
    if jacobian:
        blk["J"] = _point_jacobian(st, f, b, st.pos[f, b]).mul_(c).reshape(-1, NV)
    out["anchor"] = blk
    out["flat"] = _flat(prob, st, jacobian)
    out["balance"] = _balance(prob, st, jacobian)
    out["vel"] = _vel(prob, st, jacobian)
    return out


def _flat(prob: Problem, st: rt.State, jacobian: bool) -> dict:
    """A box face resting flat points straight down: ``r = sqrt(w) n_xy / scale`` with ``n`` the face's outward
    normal; ``d n = omega x n``."""
    fl, sk = prob.plan["flat"], rt.skeleton()
    if not len(fl["frames"]):
        return _empty()
    n = face_normals(sk, st.rot, fl["frames"], fl["body"], fl["axis"], fl["sign"])
    c = math.sqrt(prob.weights["flat"]) / SCALES["flat"] * torch.sqrt(torch.as_tensor(fl["w"]))
    blk = {"frames": np.repeat(fl["frames"], 2), "r": (c[:, None] * n[:, :2]).reshape(-1)}
    if jacobian:
        jw = rt.rotation_jacobian(st, fl["frames"], fl["body"])                   # [N, 3, NV]
        blk["J"] = (c[:, None, None] * (-rt.hat(n) @ jw)[:, :2]).reshape(-1, NV)
    return blk


def _vel(prob: Problem, st: rt.State, jacobian: bool) -> dict:
    """A support point targeted on frames t and t + 1 moves horizontally as her fit's does:
    ``r = sqrt(w) ((p(t+1) - p(t))_xy - d0) / scale``."""
    v, sk = prob.vel, rt.skeleton()
    if not len(v["frames"]):
        blk = _empty()
        blk.update(frames2=np.zeros(0, int), J2=torch.zeros(0, NV, dtype=F64))
        return blk
    f, k = v["frames"], v["cand"]
    p0, p1 = st.pts[f, k], st.pts[f + 1, k]
    c = math.sqrt(prob.weights["vel"]) / SCALES["vel"] * torch.sqrt(torch.as_tensor(v["w"]))
    blk = {"frames": np.repeat(f, 2), "frames2": np.repeat(f + 1, 2),
           "r": (c[:, None] * ((p1 - p0)[:, :2] - v["d0"])).reshape(-1)}
    if jacobian:
        bodies = sk.cand_body[k]
        blk["J"] = (-c[:, None, None] * _point_jacobian(st, f, bodies, p0)[:, :2]).reshape(-1, NV)
        blk["J2"] = (c[:, None, None] * _point_jacobian(st, f + 1, bodies, p1)[:, :2]).reshape(-1, NV)
    return blk


def hull_rows(xy: np.ndarray, c: np.ndarray, margin: float) -> list:
    """``retarget._balance``'s edge rows of one frame: for every hull edge (points ``i1 i2`` of ``xy``,
    oriented so the inside is positive) with the COM ``c`` closer than ``margin`` or outside,
    ``(i1, i2, d, g_c, g_1, g_2)``: the signed distance inside and its gradients in the COM and both ends."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(xy)
    except QhullError:
        return []
    mid = xy[hull.vertices].mean(0)
    rows = []
    for i1, i2 in hull.simplices:
        v1, v2 = xy[i1], xy[i2]
        e = v2 - v1
        L = float(np.hypot(*e))
        if L < 1e-6:
            continue
        if (e[0] * (mid - v1)[1] - e[1] * (mid - v1)[0]) < 0:
            i1, i2, v1, v2, e = i2, i1, v2, v1, -e
        wv = c - v1
        d = (e[0] * wv[1] - e[1] * wv[0]) / L
        if d >= margin:
            continue
        g_c = np.array([-e[1], e[0]]) / L
        g_1 = np.array([-wv[1] + e[1], wv[0] - e[0]]) / L + d * e / L ** 2
        g_2 = np.array([wv[1], -wv[0]]) / L - d * e / L ** 2
        rows.append((i1, i2, d, g_c, g_1, g_2))
    return rows


def _balance(prob: Problem, st: rt.State, jacobian: bool) -> dict:
    """The COM against the targeted supports' hull on the balance frames, each with its own margin; the Jacobian
    runs through the COM and both edge points (``retarget._balance``)."""
    sk, bal = rt.skeleton(), prob.balance
    frames = bal["frames"]
    if not len(frames):
        return _empty()
    mass, centre = rt.mass_model()
    m = torch.as_tensor(mass, dtype=F64)
    tw = prob.plan["target_w"]
    com_b = st.pos[frames] + (st.rot[frames] @ torch.as_tensor(centre, dtype=F64)[..., None]).squeeze(-1)
    com = ((m[None, :, None] * com_b).sum(1) / m.sum())[:, :2].numpy()
    rows = []
    for i, f in enumerate(frames):
        ks = np.nonzero(tw[f] > 0.5)[0]
        if len(ks) < 3:
            continue
        for i1, i2, d, gc, g1, g2 in hull_rows(st.pts[f, ks, :2].numpy(), com[i], bal["margin"][i]):
            rows.append((i, ks[i1], ks[i2], d - bal["margin"][i], gc, g1, g2))
    if not rows:
        return _empty()
    rf = np.array([r[0] for r in rows])
    scale = math.sqrt(prob.weights["balance"]) / SCALES["balance"] * torch.sqrt(torch.as_tensor(bal["w"][rf]))
    blk = {"frames": frames[rf], "r": scale * torch.as_tensor([r[3] for r in rows], dtype=F64)}
    if jacobian:
        uf = np.unique(rf)
        n_b = sk.num_bodies
        jb = _point_jacobian(st, np.repeat(frames[uf], n_b), np.tile(np.arange(n_b), len(uf)),
                             com_b[uf].reshape(-1, 3)).reshape(len(uf), n_b, 3, NV)
        jcom = ((m[None, :, None, None] * jb).sum(1) / m.sum())[:, :2]
        pos_of = {int(u): k for k, u in enumerate(uf)}
        fr = frames[rf]
        k1 = np.array([r[1] for r in rows])
        k2 = np.array([r[2] for r in rows])
        j1 = rt.point_jacobian(st, fr, sk.cand_body[k1], st.pts[fr, k1])[:, :2]
        j2 = rt.point_jacobian(st, fr, sk.cand_body[k2], st.pts[fr, k2])[:, :2]
        gc = torch.as_tensor(np.array([r[4] for r in rows]))
        g1 = torch.as_tensor(np.array([r[5] for r in rows]))
        g2 = torch.as_tensor(np.array([r[6] for r in rows]))
        jc = jcom[[pos_of[int(i)] for i in rf]]
        blk["J"] = scale[:, None] * ((gc[:, :, None] * jc).sum(1) + (g1[:, :, None] * j1).sum(1)
                                     + (g2[:, :, None] * j2).sum(1))
    return blk


def _linear_terms(prob: Problem, x: torch.Tensor) -> dict:
    """``retarget._linear_terms`` with the smoothness rows weighted by ``prob.smooth_w``."""
    out = rt._linear_terms(prob, x)
    if prob.smooth_w is not None and len(out["smooth"]):
        out["smooth"] = out["smooth"] * torch.as_tensor(np.sqrt(prob.smooth_w), dtype=F64)[:, None]
    return out


def smooth_diagonals(T: int, ws: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three upper diagonals of ``D^T diag(ws) D`` for the second difference ``D`` over ``T`` frames
    (``retarget._smooth_diagonals`` when ``ws`` is None)."""
    if ws is None:
        return rt._smooth_diagonals(T)
    c = (1.0, -2.0, 1.0)
    d0, d1, d2 = np.zeros(T), np.zeros(max(T - 1, 0)), np.zeros(max(T - 2, 0))
    r = np.arange(T - 2)
    for i in range(3):
        np.add.at(d0, r + i, ws * c[i] ** 2)
        if i < 2:
            np.add.at(d1, r + i, ws * c[i] * c[i + 1])
    d2 += ws * c[0] * c[2]
    return d0, d1, d2


def energy(prob: Problem, x: torch.Tensor) -> tuple[float, dict]:
    """``(total, {term: value})``: the squared residuals, each term divided by the frame count."""
    terms = {k: rt._sq(v["r"]) for k, v in residuals(prob, x).items()}
    terms.update({k: rt._sq(v) for k, v in _linear_terms(prob, x).items()})
    terms = {k: v / prob.T for k, v in terms.items()}
    return sum(terms.values()), terms


def _by_frame(T: int, f: torch.Tensor, *rows: torch.Tensor, max_cells: int = 1 << 22):
    """The rows grouped by frame and zero-padded: yields ``(frames, padded...)`` chunks with ``padded`` of shape
    ``[n_frames, max rows of a frame, ...]``, so a Gram is one batched matmul per chunk instead of one outer
    product per row (``retarget._gram``: 3.4 s of a 1,212-frame clip's step, against 0.1 s for the solve)."""
    order = torch.argsort(f, stable=True)
    f = f[order]
    rows = [x[order] for x in rows]
    counts = torch.bincount(f, minlength=T)
    starts = torch.cumsum(counts, 0) - counts
    width = max(int(rows[0][0].numel()) if len(f) else 1, 1)
    step = max(1, max_cells // max(int(counts.max()) * width, 1)) if len(f) else T
    for t0 in range(0, T, step):
        t1 = min(T, t0 + step)
        sel = (f >= t0) & (f < t1)
        if not bool(sel.any()):
            continue
        ff = f[sel]
        pos = torch.arange(len(f))[sel] - starts[ff]
        n = int(counts[t0:t1].max())
        out = []
        for x in rows:
            P = torch.zeros((t1 - t0, n) + tuple(x.shape[1:]), dtype=x.dtype)
            P[ff - t0, pos] = x[sel]
            out.append(P)
        yield t0, t1, out


def _gram_rows(A, g, f, J, r):
    """``A[t] += sum J^T J`` and ``g[t] += sum J^T r`` over the rows of every frame t (``retarget._gram``)."""
    for t0, t1, (P, R) in _by_frame(A.shape[0], f, J, r):
        Pt = P.transpose(1, 2)
        A[t0:t1] += Pt @ P
        g[t0:t1] += (Pt @ R[..., None]).squeeze(-1)


def _gram_pair(A, C, g, f, J, J2, r):
    """Rows on frames ``f`` (``J``) and ``f + 1`` (``J2``): both diagonal blocks, the cross block ``C[f]``
    (frame f's variables against frame f + 1's) and the gradient."""
    for t0, t1, (P, P2, R) in _by_frame(A.shape[0] - 1, f, J, J2, r):
        Pt, P2t = P.transpose(1, 2), P2.transpose(1, 2)
        A[t0:t1] += Pt @ P
        A[t0 + 1:t1 + 1] += P2t @ P2
        C[t0:t1] += Pt @ P2
        g[t0:t1] += (Pt @ R[..., None]).squeeze(-1)
        g[t0 + 1:t1 + 1] += (P2t @ R[..., None]).squeeze(-1)


def normal_equations(prob: Problem, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(ab, g)``: ``J^T J`` in LAPACK upper-band storage ``[BAND + 1, 75 T]`` (the frame blocks, the cross-frame
    blocks of ``vel`` and the smoothness' closed form) and ``J^T r`` ``[75 T]``."""
    T = prob.T
    s, w = SCALES, prob.weights
    res = residuals(prob, x, jacobian=True)
    A = torch.zeros(T, NV, NV, dtype=F64)
    C = torch.zeros(max(T - 1, 1), NV, NV, dtype=F64)
    g = torch.zeros(T, NV, dtype=F64)
    for blk in res.values():
        if not len(blk["r"]):
            continue
        if "J2" in blk:
            _gram_pair(A, C, g, torch.as_tensor(blk["frames"]), blk["J"], blk["J2"], blk["r"])
        else:
            _gram_rows(A, g, torch.as_tensor(blk["frames"]), blk["J"], blk["r"])
    lin = _linear_terms(prob, x)
    A[:, 6:, 6:] += w["dof"] / s["dof"] ** 2 * torch.eye(69, dtype=F64)
    g[:, 6:] += math.sqrt(w["dof"]) / s["dof"] * lin["dof"]
    cr = torch.tensor([w["root"] / s["root_pos"] ** 2] * 3 + [w["root"] / s["root_rot"] ** 2] * 3, dtype=F64)
    A[:, :6, :6] += torch.diag(cr)
    g[:, :6] += torch.sqrt(cr) * lin["root"]
    vs = torch.tensor([s["acc_pos"]] * 3 + [s["acc_rot"]] * 3 + [s["acc_dof"]] * 69, dtype=F64)
    cs = w["smooth"] / vs ** 2
    if T > 2:
        r = lin["smooth"]
        gs = torch.zeros(T, NV, dtype=F64)
        k = math.sqrt(w["smooth"]) / vs
        if prob.smooth_w is not None:
            k = k * torch.as_tensor(np.sqrt(prob.smooth_w), dtype=F64)[:, None]
        gs[2:] += k * r
        gs[1:-1] += -2 * k * r
        gs[:-2] += k * r
        g += gs
    ab = np.zeros((BAND + 1, NV * T))
    ia, ib = np.triu_indices(NV)
    cols = NV * np.arange(T)[:, None] + ib[None, :]
    ab[(BAND + ia - ib)[None, :].repeat(T, 0), cols] += A.numpy()[:, ia, ib]
    if T > 1:                     # cross blocks (t, t + 1): row NV t + a, column NV (t + 1) + b, all a, b
        a_, b_ = np.meshgrid(np.arange(NV), np.arange(NV), indexing="ij")
        rows_ = np.broadcast_to(BAND - NV + a_ - b_, (T - 1, NV, NV))
        cols_ = NV * (np.arange(1, T)[:, None, None]) + b_[None]
        ab[rows_, cols_] += C.numpy()[:T - 1]
    if T > 2:
        d0, d1, d2 = smooth_diagonals(T, prob.smooth_w)
        csn = cs.numpy()
        ab[BAND] += (d0[:, None] * csn[None, :]).ravel()
        ab[BAND - NV, NV:] += (d1[:, None] * csn[None, :]).ravel()
        ab[0, 2 * NV:] += (d2[:, None] * csn[None, :]).ravel()
    return ab, g.numpy().ravel()


def bounds(prob: Problem) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lo, hi) [T, 75]``: the plant's box intersected with the edit budget around her coordinates clipped into
    it; the root's offset and perturbation within their budgets."""
    T = prob.T
    q0 = torch.maximum(torch.minimum(prob.dof0, prob.upper), prob.lower)
    root = torch.tensor([BUDGET_ROOT_M] * 3 + [BUDGET_ROOT_RAD] * 3, dtype=F64).expand(T, 6)
    lo = torch.cat([-root, torch.maximum(prob.lower.expand(T, -1), q0 - BUDGET_JOINT_RAD)], 1)
    hi = torch.cat([root, torch.minimum(prob.upper.expand(T, -1), q0 + BUDGET_JOINT_RAD)], 1)
    return lo, hi


def solve(prob: Problem, x0: torch.Tensor | None = None, iters: int = ITERS, rtol: float = 1e-7,
          log=None) -> tuple[torch.Tensor, dict]:
    """``retarget.solve`` on this objective: Levenberg-Marquardt, one banded solve per step (``solveh_banded``),
    the box by projection with the bound-active coordinates frozen. From ``x0`` (default: her fit clipped into
    the box)."""
    from scipy.linalg import LinAlgError, solveh_banded

    start = time.time()
    T = prob.T
    x = rt.initial(prob) if x0 is None else x0.clone()
    lo, hi = bounds(prob)
    x = torch.minimum(torch.maximum(x, lo), hi)
    E, _ = energy(prob, x)
    E0, lam, history = E, 1e-3, []
    for it in range(iters):
        t_it, tries = time.time(), 0
        ab, g = normal_equations(prob, x)
        gt = torch.as_tensor(g).reshape(T, NV)
        frozen = (((x <= lo + 1e-12) & (gt > 0)) | ((x >= hi - 1e-12) & (gt < 0))).numpy().ravel()
        diag = ab[BAND].copy()
        accepted = False
        while lam < 1e8:
            band = ab.copy()
            band[BAND] += lam * np.maximum(diag, 1e-6) + 1e-9
            fz = np.nonzero(frozen)[0]
            if len(fz):
                band[:, fz] = 0.0
                for k in range(1, BAND + 1):
                    cols = fz + k
                    cols = cols[cols < band.shape[1]]
                    band[BAND - k, cols] = 0.0
                band[BAND, fz] = 1.0
            rhs = -g.copy()
            rhs[frozen] = 0.0
            try:
                step = solveh_banded(band, rhs, lower=False, check_finite=False)
            except LinAlgError:
                lam *= 10
                continue
            xn = torch.minimum(torch.maximum(x + torch.as_tensor(step).reshape(T, NV), lo), hi)
            En, _ = energy(prob, xn)
            tries += 1
            if En < E:
                accepted = True
                break
            lam *= 10
        if not accepted:
            break
        history.append(round(En, 4))
        if log is not None:
            log(f"it {it} E {En:.4f} lambda {lam:.1e} tries {tries} {time.time() - t_it:.1f}s")
        dec, x, E = E - En, xn, En
        lam = max(lam / 5, 1e-6)
        if dec < rtol * max(E, 1e-9) or np.abs(step).max() < 1e-7:
            break
    _, terms = energy(prob, x)
    return x, {"iterations": len(history), "seconds": round(time.time() - start, 1), "energy_start": round(E0, 4),
               "energy": round(E, 5), "terms": {k: round(v, 5) for k, v in terms.items()}, "history": history[-8:]}


def solve_gently(prob: Problem, iters: int = ITERS, x0: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """The fallback for a solution with new jerks (TODO A3): every contact term (``CONTACT_TERMS``) brought in by
    continuation (1 %, 10 %, then all of its weight, each stage from the last) under ``GENTLE_SMOOTH`` times the
    smoothness. Step 8 continued only the pair and penetration terms; a support switched on at full weight
    still lifts or drops a limb within its ramp."""
    base = dict(prob.weights)
    smooth = dict(base, smooth=GENTLE_SMOOTH * base["smooth"])
    x, reps = x0, []
    try:
        for k in GENTLE_STAGES:
            prob.weights = dict(smooth, **{t: k * base[t] for t in CONTACT_TERMS})
            x, rep = solve(prob, x0=x, iters=iters if k == 1.0 else max(10, iters // 2))
            reps.append(rep)
    finally:
        prob.weights = base
    rep = dict(reps[-1], iterations=sum(r["iterations"] for r in reps), seconds=round(sum(r["seconds"] for r in reps), 1),
               energy_start=reps[0]["energy_start"], stages=[{"weight": k, "iterations": r["iterations"],
                                                             "energy": r["energy"]} for k, r in zip(GENTLE_STAGES, reps)])
    return x, rep


def body_acc(pos: np.ndarray, fps: float) -> np.ndarray:
    """``[T]`` the largest body acceleration of every frame (second difference; 0 at the ends)."""
    out = np.zeros(pos.shape[0])
    if pos.shape[0] > 2:
        out[1:-1] = np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1).max(1) * fps ** 2
    return out


def spike_mask(prob: Problem, x: torch.Tensor) -> np.ndarray:
    """``[T]`` frames on which some body accelerates past ``SPIKE_ACC`` in the solution and not in her fit."""
    root_pos, root_rot, dof = rt.unpack(prob, x)
    with torch.no_grad():
        pos, _ = rt.fk(rt.skeleton(), root_pos, root_rot, dof)
    return (body_acc(pos.numpy(), prob.fps) > SPIKE_ACC) & (body_acc(prob.pos0.numpy(), prob.fps) <= SPIKE_ACC)


def spike_frames(prob: Problem, x: torch.Tensor) -> int:
    return int(spike_mask(prob, x).sum())


def smooth_locally(prob: Problem, x: torch.Tensor, iters: int = ITERS) -> tuple[torch.Tensor, dict]:
    """TODO A3's local post-pass: the smoothness stiffened ``LOCAL_FACTORS`` times (in turn) within ``LOCAL_HALF``
    frames of every frame still jerking, the clip solved again from the best solution so far; a stage is kept
    when it removes jerks. The contact terms stay at full weight, so a jerk gives way to a small constraint
    residual near it, which the clip metrics report. ``prob.smooth_w`` is left as the kept solution used it."""
    T, rows = prob.T, []
    best, best_n, best_w = x, spike_frames(prob, x), prob.smooth_w
    for factor in LOCAL_FACTORS:
        if not best_n:
            break
        near = np.zeros(T, bool)
        for t in np.nonzero(spike_mask(prob, best))[0]:
            near[max(0, t - LOCAL_HALF):t + LOCAL_HALF + 1] = True
        ws = np.ones(max(T - 2, 0)) if best_w is None else best_w.copy()
        ws[near[1:-1]] = np.maximum(ws[near[1:-1]], factor)     # second-difference row r is centred on frame r + 1
        prob.smooth_w = ws
        x2, rep = solve(prob, x0=best, iters=iters)
        n2 = spike_frames(prob, x2)
        rows.append({"factor": factor, "frames": int(near.sum()), "spike_frames": n2, "iterations": rep["iterations"],
                     "energy": rep["energy"]})
        if n2 < best_n:
            best, best_n, best_w = x2, n2, ws
    prob.smooth_w = best_w
    return best, {"stages": rows, "spike_frames": best_n}


# --------------------------------------------------------------------------- #
# What the edit did (the card's exit metrics), per clip
# --------------------------------------------------------------------------- #
FLOAT_M = 0.02             # a labelled support floats above 2 cm (Step 8's exit test, audit.HOVER_M)
FLOAT5_M = 0.05
PEN_M = -0.005             # a touching zone-frame penetrates below -0.5 cm (the card; her skin: 15.6 %)
DEEP_OVERLAP_M = 0.01      # a body pair overlapping by more than 1 cm
CROWN_M = 0.02             # a crown realised: the head within 2 cm of the floor
SLIDE_ZONES = ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND")
SLIDE_FAST = 0.05          # m/s
BOX_TOL_DEG = 1.0          # the card: 0 frames more than 1 deg past the joint box
MAT_COV_MIN, MAT_BW_MIN = 0.9, 0.6        # plant_v2.com_cop's gate: the mat sees the whole body
BW_N = 74.0 * 9.81
BEYOND_BOX_DEG = 10.0      # the summary lists clips whose own motion is this far past the box somewhere ...
SUPINATION_COORDS = ("R_Toe_x", "R_Ankle_x")    # ... except on the fit's right-foot supination (BodyFix Step 1)


def motion_state(mot: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(pos [T,B,3], rot [T,B,3,3])`` float64 numpy of a stored ``.motion``."""
    from protomotions.utils.rotations import quaternion_to_matrix

    pos = mot["rigid_body_pos"].double()
    return pos.numpy(), quaternion_to_matrix(mot["rigid_body_rot"].double(), w_last=True).numpy()


def supports_of(stem: str, anns: dict) -> list[dict]:
    """Labels v1.1's configured ground supports the human is seen making (Step 8's exit test, BodyFix's 919):
    ``hold_id``, ``zone``, window ``f0`` .. ``f1``."""
    out = []
    for hid, rows in anns.items():
        if not hid.startswith(stem + "@"):
            continue
        for a in rows:
            if a["kind"] == "ground" and a["in_configuration"] and a["source_state"] == "observed_contact":
                iv = a["interval"]
                out.append({"hold_id": hid, "zone": a["zones"][0], "f0": int(iv["start_frame"]),
                            "f1": int(iv["end_frame_exclusive"]) - 1, "exemplar": int(iv["exemplar_frame"])})
    return out


def flat_tilt_deg(sk: rt.Skeleton, rot: np.ndarray, fl: dict) -> np.ndarray:
    """``[N]`` tilt from horizontal (deg) of the flat-targeted faces of ``fl`` (``support_plan``'s ``flat``)."""
    if not len(fl["frames"]):
        return np.zeros(0)
    n = face_normals(sk, torch.as_tensor(rot), fl["frames"], fl["body"], fl["axis"], fl["sign"]).numpy()
    return np.degrees(np.arccos(np.clip(-n[:, 2], -1.0, 1.0)))


def com_margin(sk: rt.Skeleton, prob: Problem, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``[F]`` on the balance frames, the COM's signed distance inside the targeted supports' hull (m; the
    smallest over its edges, negative outside; NaN with fewer than three points)."""
    from scipy.spatial import ConvexHull, QhullError

    frames = prob.balance["frames"]
    tw = prob.plan["target_w"]
    with torch.no_grad():
        com = com_xy(torch.as_tensor(pos[frames]), torch.as_tensor(rot[frames])).numpy()
        pts = rt.candidate_points(sk, torch.as_tensor(pos[frames]), torch.as_tensor(rot[frames])).numpy()
    out = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        ks = np.nonzero(tw[f] > 0.5)[0]
        if len(ks) < 3:
            continue
        try:
            hull = ConvexHull(pts[i, ks, :2])
        except QhullError:
            continue
        out[i] = float(-(hull.equations[:, :2] @ com[i] + hull.equations[:, 2]).max())
    return out


def clip_metrics(prob: Problem, x: torch.Tensor, fit_motion: dict, motion: dict, ev: dict, anns: dict) -> dict:
    """What the edit did, on the stored motions: her fit written on the plant (``fit``) against the retarget
    (``after``). Every count the card's acceptance pools (``corpus_metrics``), plus the diagnostics."""
    from reference_curation import plant_v2

    sk = rt.skeleton()
    arr = ev["arrays"]
    state, hz = arr["human_floor_state"], arr["human_floor_z"].astype(np.float64)
    touch, sep = state == 1, state == 0
    fps = prob.fps
    sup = supports_of(prob.stem, anns)
    head_inv = prob.plan["head_inverted"]
    out, per = {"clip": {}}, {}
    zone_idx = [ZONE_ORDER.index(z) for z in SLIDE_ZONES]
    for tag, mot in (("fit", fit_motion), ("after", motion)):
        pos, rot = motion_state(mot)
        z = mr.zone_lowest(sk, pos, rot)
        h = rt.candidate_heights(sk, rt.candidate_points(sk, torch.as_tensor(pos), torch.as_tensor(rot))).numpy()
        of, oa, ob, og = rt.near_body_pairs(sk, torch.as_tensor(pos), torch.as_tensor(rot), 0.0)
        deep = og < -DEEP_OVERLAP_M
        exc = fw.box_excess_deg(mot["dof_pos"].double().numpy())
        acc = body_acc(pos, fps)
        slide = np.stack([plant_v2.zone_slide(sk, pos, rot, [sk.names.index(b) for b in ZONES[zn]], fps)
                          for zn in SLIDE_ZONES], 1)
        sl = np.concatenate([slide[touch[:, zi], k] for k, zi in enumerate(zone_idx)])
        per[tag] = {"pos": pos, "rot": rot, "z": z, "acc": acc,
                    "overlaps": set(zip(of[deep].tolist(), oa[deep].tolist(), ob[deep].tolist()))}
        wst = int(og.argmin()) if len(og) else None
        out[tag] = {
            "touch_zone_frames": int(touch.sum()),
            "float2_zone_frames": int(((z > FLOAT_M) & touch).sum()),
            "pen_zone_frames": int(((z < PEN_M) & touch).sum()),
            "pen_zone_frames_by_zone": {zn: int(((z[:, i] < PEN_M) & touch[:, i]).sum()) for i, zn in enumerate(ZONE_ORDER)
                                        if touch[:, i].any()},
            "phantom_zone_frames": int(((z <= FLOAT_M) & sep).sum()), "separated_zone_frames": int(sep.sum()),
            "floor_min_cm": round(100 * float(h.min()), 3),
            "frames_past_box_1deg": int((exc > BOX_TOL_DEG).any(1).sum()),
            "frames_past_box_2deg": int((exc > 2.0).any(1).sum()),
            "box_excess_max_deg": round(float(exc.max()), 3),
            "deep_overlap_pair_frames": int(deep.sum()),
            "overlap_max_cm": round(-100 * float(og.min()), 2) if len(og) else 0.0,
            "overlap_worst": [sk.names[oa[wst]], sk.names[ob[wst]], int(of[wst])] if wst is not None else None,
            "acc_spike_frames": int((acc > SPIKE_ACC).sum()),
            "acc_rms": round(float(np.sqrt((acc[1:-1] ** 2).mean())), 4) if len(acc) > 2 else 0.0,
            "slide": {"n": int(len(sl)), "fast": int((sl > SLIDE_FAST).sum()),
                      "p50_cm_s": round(100 * float(np.median(sl)), 3) if len(sl) else None,
                      "p90_cm_s": round(100 * float(np.percentile(sl, 90)), 3) if len(sl) else None},
            "flat_face_tilt_deg_p50_p90": ([round(float(v), 2) for v in np.percentile(
                flat_tilt_deg(sk, rot, _full_flat(prob)), [50, 90])] if len(_full_flat(prob)["frames"]) else None)}
        cm = com_margin(sk, prob, pos, rot)
        ok = np.isfinite(cm)
        out[tag]["com_margin_cm_p10_p50"] = ([round(100 * float(v), 2) for v in np.percentile(cm[ok], [10, 50])]
                                             if ok.any() else None)
        out[tag]["com_outside_frames"] = int((cm[ok] < 0).sum())
    out["skin"] = {"pen_zone_frames": int(((hz < PEN_M) & touch).sum())}
    # where her motion asks for more than the plant's joint box allows (the retarget must take it elsewhere)
    exc = fw.box_excess_deg(fit_motion["dof_pos"].double().numpy())
    coords = [f"{b}_{a}" for b in sk.names[1:] for a in "xyz"]
    out["fit_box_excess_deg"] = {coords[j]: [round(float(exc[:, j].max()), 1), int((exc[:, j] > BOX_TOL_DEG).sum())]
                                 for j in np.argsort(-exc.max(0)) if exc[:, j].max() > BOX_TOL_DEG}
    out["new_spike_frames"] = int(((per["after"]["acc"] > SPIKE_ACC) & (per["fit"]["acc"] <= SPIKE_ACC)).sum())
    out["new_deep_overlaps"] = len(per["after"]["overlaps"] - per["fit"]["overlaps"])
    rows = []
    for s in sup:
        win = slice(s["f0"], s["f1"] + 1)
        zi = ZONE_ORDER.index(s["zone"])
        row = {**{k: s[k] for k in ("hold_id", "zone")},
               "crown": bool(s["zone"] == "HEAD" and head_inv[win].mean() > 0.5),
               "human_cm": round(100 * float(np.median(hz[win, zi])), 2)}
        for tag in ("fit", "after"):
            row[f"{tag}_cm"] = round(100 * float(np.median(per[tag]["z"][win, zi])), 2)
        rows.append(row)
    out["supports"] = rows
    disp = np.linalg.norm(per["after"]["pos"] - per["fit"]["pos"], axis=-1)                     # [T, B]
    dq = np.degrees(np.abs((x[:, 6:] - prob.dof0).numpy()))
    out["edit"] = {"mean_disp_cm_p50": round(100 * float(np.median(disp.mean(1))), 3),
                   "mean_disp_cm_p95": round(100 * float(np.percentile(disp.mean(1), 95)), 3),
                   "body_disp_cm_max": round(100 * float(disp.max()), 3),
                   "body_disp_worst": sk.names[int(disp.max(0).argmax())],
                   "body_disp_cm_p95_by_body": {b: round(100 * float(np.percentile(disp[:, i], 95)), 2)
                                                for i, b in enumerate(sk.names)},
                   "joint_change_deg_p50": round(float(np.median(dq)), 3),
                   "joint_change_deg_p99": round(float(np.percentile(dq, 99)), 3),
                   "joint_change_deg_max": round(float(dq.max()), 3)}
    reqs = []
    for r in prob.requests:
        row = {k: r[k] for k in ("hold_id", "contact", "why", "target_role", "status")}
        row["frames"] = int(len(r["frames"]))
        if len(r["frames"]):
            for tag in ("fit", "after"):
                g, _, _ = rt.zone_pair_gaps(sk, torch.as_tensor(per[tag]["pos"]), torch.as_tensor(per[tag]["rot"]),
                                            *r["zones"], r["frames"])
                row[f"gap_cm_{tag}"] = round(100 * float(np.median(g)), 2)
        reqs.append(row)
    out["pair_requests"] = reqs
    out["com_cop"] = com_cop_rows(prob, per, arr, anns)
    return out


def _full_flat(prob: Problem) -> dict:
    """The flat rows at full weight (the ramps excluded)."""
    fl = prob.plan["flat"]
    m = fl["w"] > 0.999
    return {k: v[m] for k, v in fl.items()}


def com_cop_rows(prob: Problem, per: dict, arr: dict, anns: dict) -> list[dict]:
    """At every hold exemplar where the mat sees the whole body (``plant_v2.com_cop``'s gate), the horizontal
    distance between the plant's COM and the mat COP, for the fit and the retarget (cm)."""
    rows = []
    for hid in sorted({h for h in anns if h.startswith(prob.stem + "@")}):
        f = ids.parse_hold_id(hid)[1]
        cop, tot, cov = arr["mat_cop"][f], float(arr["mat_total"][f]), float(arr["mat_valid_cov"][f])
        ok = bool(np.isfinite(cop).all() and cov >= MAT_COV_MIN and tot >= MAT_BW_MIN * BW_N)
        row = {"hold_id": hid, "mat_ok": ok}
        if ok:
            for tag in ("fit", "after"):
                c = com_xy(torch.as_tensor(per[tag]["pos"][f:f + 1]), torch.as_tensor(per[tag]["rot"][f:f + 1]))[0].numpy()
                row[f"{tag}_cm"] = round(100 * float(np.linalg.norm(c - cop)), 2)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Clips, the corpus and the release of the motions
# --------------------------------------------------------------------------- #
OUT_ROOT = ids.OUTPUT_ROOT / "retarget_v2"          # motions and per-frame lineage (bulky, regenerable)
RECORD_ROOT = ids.DATA_ROOT / "retarget_v2"         # the records (small)
CONFIG = {"plant": PLANT, "clearance_m": CLEARANCE_M, "floor_min_m": FLOOR_MIN_M, "off_floor_m": OFF_FLOOR_M,
          "flat_rule": "all four corners within touch_m + release_m", "roll_free_faces": list(ROLL_FREE),
          "edge_band_m": rt.EDGE_BAND_M, "floor_contract_m": FLOOR_CONTRACT_M, "release_m": rt.RELEASE_M,
          "mode_frames": rt.MODE_FRAMES, "corner_neighbours": rt.CORNER_NEIGHBOURS, "face_cos": rt.FACE_COS,
          "head_refused": False, "pair_target_m": PAIR_TARGET_M, "pair_reach_m": PAIR_REACH_M, "pen_tol_m": PEN_TOL_M,
          "pen_soft": "her contact overlaps: no deeper (pen), apart by preference (pen_soft)",
          "pen_pairs": "plant", "limit_margin_deg": math.degrees(LIMIT_MARGIN_RAD), "weights": WEIGHTS, "scales": SCALES,
          "end_bodies": list(END_BODIES), "iters": ITERS, "budget_joint_deg": math.degrees(BUDGET_JOINT_RAD),
          "budget_root_m": BUDGET_ROOT_M, "budget_root_deg": math.degrees(BUDGET_ROOT_RAD), "min_run": MIN_RUN,
          "ramp": RAMP, "spike_acc": SPIKE_ACC, "gentle_smooth": GENTLE_SMOOTH, "gentle_stages": list(GENTLE_STAGES),
          "contact_terms": list(CONTACT_TERMS), "local_half": LOCAL_HALF, "local_factors": list(LOCAL_FACTORS), "balance_margin_m": BALANCE_MARGIN_M, "margin_share": MARGIN_SHARE,
          "qs_speed_m_s": QS_SPEED_M_S, "qs_smooth_frames": QS_SMOOTH_FRAMES, "qs_stable_frames": QS_STABLE_FRAMES}


def retarget_clip(stem: str, anns: dict, fit_dir: Path, iters: int = ITERS) -> dict:
    """One clip: ``{"motion", "record", "lineage"}``. The anchor is the ``fit_writer`` motion in ``fit_dir``; a
    solution with new jerks is solved again gently (``solve_gently``) and the one with fewer is kept."""
    start = time.time()
    ev = human_evidence(stem)                          # outside the plant context (see human_evidence)
    human, status, err = hm.load_human(stem)
    if human is None:
        raise ValueError(f"{stem}: no human mesh ({status}: {err})")
    fit_path = Path(fit_dir) / f"{stem}.motion"
    fit_motion = torch.load(fit_path, map_location="cpu", weights_only=False)
    plant_identity.require(fit_motion.get(plant_identity.KEY), fw.plant_paths(PLANT)[0], f"{fit_path}")
    with on_plant():
        prob = build_problem(stem, anns, ev, human, fit_motion)
        x, rep = solve(prob, iters=iters)
        rep["spike_frames"] = spikes = spike_frames(prob, x)
        if spikes:
            xg, gentle = solve_gently(prob, iters)
            gentle["spike_frames"] = spike_frames(prob, xg)
            rep["gentle"] = {k: gentle[k] for k in ("iterations", "seconds", "energy", "spike_frames", "stages")}
            rep["kept"] = "gentle" if gentle["spike_frames"] < spikes else "direct"
            if rep["kept"] == "gentle":
                x = xg
                rep.update({k: gentle[k] for k in ("energy", "terms", "history")})
            if min(spikes, gentle["spike_frames"]):
                xl, local = smooth_locally(prob, x, iters)
                rep["local"] = local
                if local["spike_frames"] < min(spikes, gentle["spike_frames"]):
                    x = xl
                    rep["kept"] += "+local"
            rep["spike_frames_kept"] = spike_frames(prob, x)
        root_pos, root_rot, dof = rt.unpack(prob, x)
        motion = fw.write_motion(root_pos, root_rot, dof, prob.fps, PLANT)
        metrics = clip_metrics(prob, x, fit_motion, motion, ev, anns)
        tgt = prob.plan["target"]
        record = {"stem": stem, "frames": prob.T, "fps": prob.fps, "solver": rep, "metrics": metrics,
                  "targets_by_zone": {z: int(tgt[:, c].any(1).sum()) for z, c in zip(ZONE_ORDER, rt.skeleton().zone_cands)
                                      if tgt[:, c].any()},
                  "plan": {"flat_frames": int(len(np.unique(prob.plan["flat"]["frames"]))),
                           "soft_pair_frames": prob.pen["soft_pair_frames"],
                           "held_apart_pair_frames": prob.pen["held_apart_pair_frames"],
                           "vel_rows": int(len(prob.vel["frames"])), "balance_frames": int(len(prob.balance["frames"])),
                           "balance_margin_cm_p50": round(100 * float(np.median(prob.balance["margin"])), 2)
                           if len(prob.balance["frames"]) else None},
                  "inputs": {"fit_motion": ids.sha256_file(fit_path), "store": ev["identity"]},
                  "seconds": round(time.time() - start, 1)}
        lineage = {"target": tgt, "off": prob.plan["off"], "head_inverted": prob.plan["head_inverted"],
                   "flat_frames": prob.plan["flat"]["frames"], "flat_body": prob.plan["flat"]["body"],
                   "balance_frames": prob.balance["frames"], "balance_margin": prob.balance["margin"],
                   "x": x.numpy().astype(np.float32),
                   "body_disp_m": np.linalg.norm(motion["rigid_body_pos"].double().numpy()
                                                 - fit_motion["rigid_body_pos"].double().numpy(), axis=-1).astype(np.float16),
                   "dof_change_rad": (x[:, 6:] - prob.dof0).numpy().astype(np.float16)}
    return {"motion": motion, "record": record, "lineage": lineage}


def _worker(job: tuple) -> tuple[str, dict | None, str | None]:
    stem, anns, fit_dir, out_dir, iters = job
    torch.set_num_threads(1)
    try:
        res = retarget_clip(stem, anns, Path(fit_dir), iters)
        torch.save(res["motion"], Path(out_dir) / f"{stem}.motion")
        np.savez_compressed(Path(out_dir) / f"{stem}.lineage.npz", **res["lineage"])
        return stem, res["record"], None
    except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
        import traceback

        return stem, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"


def generators() -> list[Path]:
    from reference_curation import plant_v2, sources

    return [Path(__file__), Path(rt.__file__), Path(fw.__file__), Path(mr.__file__), Path(sources.__file__),
            Path(hm.__file__), Path(plant_v2.__file__)]


def retarget_id(labels: dict, fit_rec: dict, stems: list[str]) -> str:
    key = {"schema": SCHEMA_VERSION, "config": CONFIG, "labels": labels["id"], "fit_id": fit_rec["fit_id"],
           "plant": plant_identity.sha256(PLANT), "generators": {p.name: ids.sha256_file(p) for p in generators()},
           "stems": sorted(stems)}
    return f"{labels['id']}.retarget_{RETARGET_VERSION}.{ids.sha256_json(key)[:10]}"


def run(labels: dict, stems: list[str], fit_dir: Path, rid: str, workers: int = 8, iters: int = ITERS,
        out_root: Path = OUT_ROOT) -> tuple[Path, list[dict], list[str]]:
    """Retarget ``stems`` into ``out_root/<rid>/`` in ``workers`` spawned, single-threaded processes (BLAS threads
    make the banded solve 1000x slower: 123 s against 0.11 s on a 1,212-frame clip)."""
    out = Path(out_root) / rid
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, {h: a for h, a in labels["anns"].items() if h.startswith(s + "@")}, str(fit_dir), str(out), iters)
            for s in stems]
    with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
            max(1, workers), mp_context=multiprocessing.get_context("spawn")) as ex:
        results = list(ex.map(_worker, jobs))
    records = [r for _, r, e in results if r is not None]
    failures = [f"{s}: {e}" for s, _, e in results if e is not None]
    return out, records, failures


# --------------------------------------------------------------------------- #
# The contract, the corpus metrics and the acceptance
# --------------------------------------------------------------------------- #
def check(records: list[dict], out_dir: Path) -> list[str]:
    """Every violation of the motions' contract: a frame past the joint box, a surface below the floor beyond
    ``FLOOR_CONTRACT_M``, a failed round trip or plant identity, non-finite fields."""
    problems = []
    for r in records:
        s, a = r["stem"], r["metrics"]["after"]
        if a["frames_past_box_1deg"] or a["box_excess_max_deg"] > 0.01:
            problems.append(f"{s}: joint coordinates {a['box_excess_max_deg']} deg outside the plant's box")
        if a["floor_min_cm"] < 100 * FLOOR_CONTRACT_M:
            problems.append(f"{s}: a surface {a['floor_min_cm']} cm below the floor")
        rtp = fw.round_trip(torch.load(Path(out_dir) / f"{s}.motion", map_location="cpu", weights_only=False))
        r["round_trip"] = rtp
        if not fw.round_trip_ok(rtp):
            problems.append(f"{s}: round trip {rtp}")
    return problems


def corpus_metrics(records: list[dict]) -> dict:
    """The card's exit numbers pooled over the clips, for her fit on the plant and for the retarget."""
    out = {"clips": len(records), "frames": sum(r["frames"] for r in records)}
    sup = [s for r in records for s in r["metrics"]["supports"]]
    crowns = [s for s in sup if s["crown"]]
    out["supports"] = {"n": len(sup), "crowns": len(crowns), "human_over2": sum(s["human_cm"] > 100 * FLOAT_M for s in sup)}
    for tag in ("fit", "after"):
        v = np.array([s[f"{tag}_cm"] for s in sup])
        over = [s for s in sup if s[f"{tag}_cm"] > 100 * FLOAT_M]
        out["supports"][tag] = {
            "over2": len(over), "over2_share": round(len(over) / max(len(sup), 1), 4),
            "over5": int((v > 100 * FLOAT5_M).sum()), "below_minus2": int((v < -2.0).sum()),
            "over2_by_zone": dict(collections.Counter(s["zone"] for s in over)),
            "crowns_over2": sum(s[f"{tag}_cm"] > 100 * CROWN_M for s in crowns),
            "crowns_within_2cm": sum(abs(s[f"{tag}_cm"]) <= 100 * CROWN_M for s in crowns),
            "median_cm": round(float(np.median(v)), 2) if len(v) else None}
    pooled = collections.Counter()
    for r in records:
        m = r["metrics"]
        pooled["skin_pen"] += m["skin"]["pen_zone_frames"]
        pooled["new_spike_frames"] += m["new_spike_frames"]
        pooled["new_deep_overlaps"] += m["new_deep_overlaps"]
        for tag in ("fit", "after"):
            for k in ("touch_zone_frames", "float2_zone_frames", "pen_zone_frames", "phantom_zone_frames",
                      "separated_zone_frames", "frames_past_box_1deg", "frames_past_box_2deg",
                      "deep_overlap_pair_frames", "acc_spike_frames", "com_outside_frames"):
                pooled[f"{tag}_{k}"] += m[tag][k]
            pooled[f"{tag}_slide_n"] += m[tag]["slide"]["n"]
            pooled[f"{tag}_slide_fast"] += m[tag]["slide"]["fast"]
        pooled["gentle_kept"] += r["solver"].get("kept") == "gentle"
        pooled["jerky_direct"] += bool(r["solver"].get("spike_frames"))
    out["pooled"] = dict(pooled)
    touch = max(pooled["after_touch_zone_frames"], 1)
    out["touch_penetration"] = {"skin_share": round(pooled["skin_pen"] / touch, 4),
                                "fit_share": round(pooled["fit_pen_zone_frames"] / touch, 4),
                                "after_share": round(pooled["after_pen_zone_frames"] / touch, 4)}
    out["slide"] = {}
    for tag in ("fit", "after"):
        p90 = [r["metrics"][tag]["slide"]["p90_cm_s"] for r in records if r["metrics"][tag]["slide"]["n"]]
        out["slide"][tag] = {"clip_median_of_p90_cm_s": round(float(np.median(p90)), 3) if p90 else None,
                             "clip_max_of_p90_cm_s": round(float(np.max(p90)), 3) if p90 else None,
                             "share_gt_5cm_s": round(pooled[f"{tag}_slide_fast"] / max(pooled[f"{tag}_slide_n"], 1), 4)}
    edit = {k: [r["metrics"]["edit"][k] for r in records] for k in
            ("mean_disp_cm_p50", "mean_disp_cm_p95", "body_disp_cm_max", "joint_change_deg_p50", "joint_change_deg_p99",
             "joint_change_deg_max")}
    out["edit"] = {k: {"p50": round(float(np.median(v)), 3), "max": round(float(np.max(v)), 3)} for k, v in edit.items() if v}
    reqs = [q for r in records for q in r["metrics"]["pair_requests"]]
    out["pair_requests"] = dict(collections.Counter(q["status"] for q in reqs))
    closed = [q for q in reqs if q["status"] == "closed"]
    out["pairs_closed_within_1cm_after"] = sum(q.get("gap_cm_after", 99) <= 1.0 for q in closed)
    out["pairs_closed"] = len(closed)
    cc = [c for r in records for c in r["metrics"]["com_cop"] if c["mat_ok"]]
    out["com_cop"] = {"holds": len(cc), **{f"{tag}_p50_cm": round(float(np.median([c[f"{tag}_cm"] for c in cc])), 2)
                                           for tag in ("fit", "after") if cc}}
    flat = [r["metrics"][t]["flat_face_tilt_deg_p50_p90"] for r in records for t in ("fit", "after")]
    out["flat_tilt_deg_p50_over_clips"] = {t: round(float(np.median([r["metrics"][t]["flat_face_tilt_deg_p50_p90"][0]
                                                                     for r in records if r["metrics"][t]["flat_face_tilt_deg_p50_p90"]])), 2)
                                           for t in ("fit", "after") if any(flat)}
    return out


# The card's acceptance (BodyFix Step 3) and the baselines it quotes.
ACCEPT = {"box_frames_past_1deg": 0, "float_share_max": 0.05, "crowns_floating": 0,
          "slide_clip_p90_max_cm_s": 4.5, "slide_fast_share_max": 0.10, "new_spike_frames": 0, "new_deep_overlaps": 0,
          "step8_mean_disp_cm_p50": 3.61, "step8_body_disp_cm_max_p50": 15.52, "well_under": 0.5}


def acceptance(m: dict) -> dict:
    """``{test: {value, limit, passed}}`` of the card's acceptance on the corpus metrics."""
    s, po = m["supports"], m["pooled"]
    tests = {
        "frames_past_box_1deg": (po["after_frames_past_box_1deg"], ACCEPT["box_frames_past_1deg"],
                                 po["after_frames_past_box_1deg"] <= ACCEPT["box_frames_past_1deg"]),
        "labelled_float_share": (s["after"]["over2_share"], ACCEPT["float_share_max"],
                                 s["after"]["over2_share"] <= ACCEPT["float_share_max"]),
        "crowns_floating": (s["after"]["crowns_over2"], ACCEPT["crowns_floating"],
                            s["after"]["crowns_over2"] <= ACCEPT["crowns_floating"]),
        "touch_penetration_share": (m["touch_penetration"]["after_share"], m["touch_penetration"]["skin_share"],
                                    m["touch_penetration"]["after_share"] <= m["touch_penetration"]["skin_share"]),
        "slide_clip_p90_cm_s": (m["slide"]["after"]["clip_median_of_p90_cm_s"], ACCEPT["slide_clip_p90_max_cm_s"],
                                (m["slide"]["after"]["clip_median_of_p90_cm_s"] or 0) <= ACCEPT["slide_clip_p90_max_cm_s"]),
        "slide_fast_share": (m["slide"]["after"]["share_gt_5cm_s"], ACCEPT["slide_fast_share_max"],
                             m["slide"]["after"]["share_gt_5cm_s"] <= ACCEPT["slide_fast_share_max"]),
        "new_spike_frames": (po["new_spike_frames"], ACCEPT["new_spike_frames"],
                             po["new_spike_frames"] <= ACCEPT["new_spike_frames"]),
        "new_deep_overlaps": (po["new_deep_overlaps"], ACCEPT["new_deep_overlaps"],
                              po["new_deep_overlaps"] <= ACCEPT["new_deep_overlaps"]),
        "edit_mean_disp_cm_p50": (m["edit"]["mean_disp_cm_p50"]["p50"], ACCEPT["well_under"] * ACCEPT["step8_mean_disp_cm_p50"],
                                  m["edit"]["mean_disp_cm_p50"]["p50"] <= ACCEPT["well_under"] * ACCEPT["step8_mean_disp_cm_p50"]),
        "edit_body_disp_cm_max_p50": (m["edit"]["body_disp_cm_max"]["p50"],
                                      ACCEPT["well_under"] * ACCEPT["step8_body_disp_cm_max_p50"],
                                      m["edit"]["body_disp_cm_max"]["p50"]
                                      <= ACCEPT["well_under"] * ACCEPT["step8_body_disp_cm_max_p50"]),
    }
    return {k: {"value": v, "limit": lim, "passed": bool(ok)} for k, (v, lim, ok) in tests.items()}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


RECORD_KEYS = ("stem", "frames", "fps", "solver", "metrics", "targets_by_zone", "plan", "inputs", "seconds")


def summary_markdown(rid: str, m: dict, acc: dict, records: list[dict], dropped: dict) -> str:
    s, po, sl, tp = m["supports"], m["pooled"], m["slide"], m["touch_penetration"]

    def pct(a, b):
        return f"{a} / {b} ({100 * a / max(b, 1):.1f} %)"

    lines = [f"# Retarget v2 `{rid}`", "",
             "Generated by `reference_curation.retarget_v2` (BodyFix Step 3); the rules are in its docstring. "
             "Fit = her MoSh fit written on plant v2 unchanged (`fit_writer`), after = the retarget.", "",
             "## Acceptance", "", "| Test | Value | Limit | Passed |", "|---|---|---|---|"]
    for k, v in acc.items():
        lines.append(f"| {k} | {v['value']} | {v['limit']} | {'yes' if v['passed'] else '**no**'} |")
    lines += ["", "## Fit vs retarget", "", "| Metric | Fit | After |", "|---|---|---|",
              f"| Labelled supports > 2 cm (labels v1.1, configured, observed) | {pct(s['fit']['over2'], s['n'])} | "
              f"{pct(s['after']['over2'], s['n'])} |",
              f"| ... > 5 cm | {s['fit']['over5']} | {s['after']['over5']} |",
              f"| Crowns within 2 cm of the floor (of {s['crowns']}) | {s['fit']['crowns_within_2cm']} | "
              f"{s['after']['crowns_within_2cm']} |",
              f"| Touching zone-frames below -0.5 cm (her skin {100 * tp['skin_share']:.1f} %) | {100 * tp['fit_share']:.1f} % | "
              f"{100 * tp['after_share']:.1f} % |",
              f"| Touching zone-frames floating > 2 cm | {pct(po['fit_float2_zone_frames'], po['fit_touch_zone_frames'])} | "
              f"{pct(po['after_float2_zone_frames'], po['after_touch_zone_frames'])} |",
              f"| Frames > 1 deg past the joint box | {pct(po['fit_frames_past_box_1deg'], m['frames'])} | "
              f"{pct(po['after_frames_past_box_1deg'], m['frames'])} |",
              f"| Slide, clip median of p90 (cm/s) | {sl['fit']['clip_median_of_p90_cm_s']} | {sl['after']['clip_median_of_p90_cm_s']} |",
              f"| Slide, touch frames > 5 cm/s | {100 * sl['fit']['share_gt_5cm_s']:.1f} % | {100 * sl['after']['share_gt_5cm_s']:.1f} % |",
              f"| Frames with a body > {SPIKE_ACC:.0f} m/s^2 | {po['fit_acc_spike_frames']} | {po['after_acc_spike_frames']} "
              f"(new {po['new_spike_frames']}) |",
              f"| Body-pair overlaps > 1 cm (pair-frames) | {po['fit_deep_overlap_pair_frames']} | "
              f"{po['after_deep_overlap_pair_frames']} (new {po['new_deep_overlaps']}) |",
              f"| Balance frames with the COM outside the targeted supports | {po['fit_com_outside_frames']} | "
              f"{po['after_com_outside_frames']} |",
              f"| COM vs mat COP at exemplars, p50 cm ({m['com_cop']['holds']} holds) | {m['com_cop'].get('fit_p50_cm')} | "
              f"{m['com_cop'].get('after_p50_cm')} |",
              f"| Flat-face tilt, clip median of p50 (deg) | {m['flat_tilt_deg_p50_over_clips'].get('fit')} | "
              f"{m['flat_tilt_deg_p50_over_clips'].get('after')} |",
              "", f"Edit (clip p50 / max): mean body displacement p50 {m['edit']['mean_disp_cm_p50']['p50']} / "
              f"{m['edit']['mean_disp_cm_p50']['max']} cm (Step 8: 3.61 / 7.99); worst body {m['edit']['body_disp_cm_max']['p50']} / "
              f"{m['edit']['body_disp_cm_max']['max']} cm (Step 8: 15.52 / 42.2).", "",
              f"Solved again gently for new jerks: kept on {po['gentle_kept']} of the {po['jerky_direct']} clips whose direct "
              "solve had some.", "",
              f"Pair requests: {m['pair_requests']}; closed within 1 cm after: {m['pairs_closed_within_1cm_after']} / "
              f"{m['pairs_closed']}.", "",
              "Dropped: " + "; ".join(f"`{k}` ({v})" for k, v in dropped.items()) + ".", "",
              "## Beyond the plant's joint box", "",
              f"Clips whose own motion asks for a coordinate more than {BEYOND_BOX_DEG:.0f} deg past the plant's box, other "
              f"than {' and '.join(SUPINATION_COORDS)} (her fit's right-foot supination, which Step 1 kept out of the box "
              "for this step's flatness to correct): poses the plant cannot reach, so the retarget must take the "
              "difference elsewhere.", "",
              "| Clip | Coordinate: max deg past (frames > 1 deg) | New overlaps | New spikes | Worst body (cm) |",
              "|---|---|---|---|---|"]
    for r in records:
        top = {k: v for k, v in r["metrics"].get("fit_box_excess_deg", {}).items()
               if v[0] > BEYOND_BOX_DEG and k not in SUPINATION_COORDS}
        if top:
            lines.append(f"| `{r['stem']}` | " + ", ".join(f"{k}: {v[0]} ({v[1]})" for k, v in top.items())
                         + f" | {r['metrics']['new_deep_overlaps']} | {r['metrics']['new_spike_frames']} | "
                         f"{r['metrics']['edit']['body_disp_cm_max']} ({r['metrics']['edit']['body_disp_worst']}) |")
    lines += ["", "## Clips", "",
              "| Clip | Frames | Pen < -0.5 cm fit -> after | Float > 2 cm | Box > 1 deg | New spikes | New overlaps | Slide p90 (cm/s) | Mean disp p50 (cm) | Worst body (cm) | Kept |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        f, a, e = r["metrics"]["fit"], r["metrics"]["after"], r["metrics"]["edit"]
        t = max(a["touch_zone_frames"], 1)
        lines.append(f"| `{r['stem']}` | {r['frames']} | {100 * f['pen_zone_frames'] / t:.1f} -> {100 * a['pen_zone_frames'] / t:.1f} % | "
                     f"{100 * f['float2_zone_frames'] / t:.1f} -> {100 * a['float2_zone_frames'] / t:.1f} % | "
                     f"{f['frames_past_box_1deg']} -> {a['frames_past_box_1deg']} | {r['metrics']['new_spike_frames']} | "
                     f"{r['metrics']['new_deep_overlaps']} | {f['slide']['p90_cm_s']} -> {a['slide']['p90_cm_s']} | "
                     f"{e['mean_disp_cm_p50']} | {e['body_disp_cm_max']} ({e['body_disp_worst']}) | "
                     f"{r['solver'].get('kept', 'direct')} |")
    return "\n".join(lines) + "\n"


def write(labels: dict, fit_rec: dict, rid: str, out_dir: Path, records: list[dict], metrics: dict, acc: dict,
          dropped: dict, record_root: Path = RECORD_ROOT) -> Path:
    out = Path(record_root) / rid
    out.mkdir(parents=True, exist_ok=True)
    head = {"schema_version": SCHEMA_VERSION, "retarget_id": rid}
    (out / "clips.jsonl").write_text("".join(json.dumps({**head, **{k: r[k] for k in RECORD_KEYS},
                                                         "round_trip": r.get("round_trip")}, default=_json_default)
                                             + "\n" for r in records))
    inputs = [labels["dir"] / "holds.yaml", labels["dir"] / "annotations.jsonl", fw.plant_paths(PLANT)[0],
              fw.plant_paths(PLANT)[1]]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "retarget_id": rid,
              "labels_id": labels["id"], "fit_id": fit_rec["fit_id"], "fit_dir": fit_rec["out_dir"],
              "plant": plant_identity.identity(PLANT), "out_dir": ids.display_path(out_dir), "config": CONFIG,
              "generators": {ids.display_path(p): ids.sha256_file(p) for p in generators()},
              "motions": {r["stem"]: ids.sha256_file(Path(out_dir) / f"{r['stem']}.motion") for r in records},
              "dropped": dropped, "metrics": metrics, "acceptance": acc}
    (out / "retarget.json").write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
    (out / "summary.md").write_text(summary_markdown(rid, metrics, acc, records, dropped))
    return out


PILOT = ("220923_Crane_Crow_Pose_or_Bakasana_-a", "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c",
         "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a",
         "220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a",
         "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b", "220926_Plow_Pose_or_Halasana_-b",
         "220923_Peacock_Pose_or_Mayurasana_-a", "220926_Upward_Plank_Pose_or_Purvottanasana_-a",
         "220926_Warrior_II_Pose_or_Virabhadrasana_II_-a", "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a",
         # TODO_before_step9 F.1: the ft_c probe's worst clips
         "220923_Tree_Pose_or_Vrksasana_-a", "220923_Tree_Pose_or_Vrksasana_-b",
         "220926_Lord_of_the_Dance_Pose_or_Natarajasana_-c", "220923_Firefly_Pose_or_Tittibhasana_-a",
         "220923_Firefly_Pose_or_Tittibhasana_-b")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="every clip of the writer's corpus (56)")
    what.add_argument("--pilot", action="store_true", help="BUILD_PLAN's pilot clips plus the ft_c probe's worst")
    what.add_argument("--stem", nargs="+")
    ap.add_argument("--labels", type=Path, default=None, help="labels v1.1 folder (default: the one of the manifest)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--record-root", type=Path, default=RECORD_ROOT)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        labels = load_labels(args.labels)
        corpus, dropped = fw.corpus()
        stems = corpus if args.all else list(PILOT) if args.pilot else list(args.stem)
        unknown = [s for s in stems if s not in corpus]
        if unknown:
            raise ValueError(f"not in the writer's corpus: {unknown}")
        fit_dir, fit_rec, failures = fw.build(corpus, dropped, workers=args.workers)
        if failures:
            raise RuntimeError(f"fit writer: {failures}")
        rid = retarget_id(labels, fit_rec, stems)
        out_dir, records, failures = run(labels, stems, fit_dir, rid, args.workers, args.iters, args.out_root)
        failures += check(records, out_dir)
        metrics = corpus_metrics(records) if records else {}
        acc = acceptance(metrics) if records else {}
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if not records:
        print("retarget_v2: nothing retargeted", file=sys.stderr)
        return 1
    rec = write(labels, fit_rec, rid, out_dir, records, metrics, acc, dropped, args.record_root)
    bad = [k for k, v in acc.items() if not v["passed"]]
    s = metrics["supports"]
    print(f"retarget_v2 {rid}: {len(records)} clips in {time.time() - start:.0f} s; labelled floats "
          f"{s['fit']['over2']}/{s['n']} -> {s['after']['over2']}/{s['n']}; crowns {s['after']['crowns_within_2cm']}/"
          f"{s['crowns']}; touch penetration {metrics['touch_penetration']['fit_share']} -> "
          f"{metrics['touch_penetration']['after_share']} (skin {metrics['touch_penetration']['skin_share']}); new spikes "
          f"{metrics['pooled']['new_spike_frames']}; new overlaps {metrics['pooled']['new_deep_overlaps']}; "
          f"{len(failures)} failures; acceptance {'passed' if not bad else 'FAILED ' + ', '.join(bad)} -> "
          f"{ids.display_path(rec)}")
    return 1 if failures or bad else 0


if __name__ == "__main__":
    sys.exit(main())
