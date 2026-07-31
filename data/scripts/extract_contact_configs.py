# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract contact-configuration events from kinematic ``.motion`` clips.

A *contact configuration* is a set of active contact pairs -- (zone, GROUND) or
(zone_a, zone_b) -- over ~15 semantic body zones (feet, shanks, thighs, pelvis,
trunk, head, upper arms, forearms, hands), plus a coarse trunk-orientation bin
(upright / inverted / prone / supine / side_l / side_r).  Contacts are detected
geometrically from collision-geom surface distances (:mod:`contact_geometry`),
NOT from the stored ``rigid_body_contacts`` labels (joint-center heuristic,
0-16 % precision on this corpus, no body-body notion).

Per clip the extractor emits:

* ``segments`` -- maximal runs of a constant (contact set, orientation bin),
  with per-contact geometry (heading-canonical root-relative witness points,
  mean gap, static/sliding) and dwell time;
* ``events`` -- every make / break / reorient between segments, with the
  approach velocity of the pair at the event.

Activation uses hysteresis (make < eps_make, break > eps_break) plus
morphological cleanup (short breaks merged, short makes dropped), so boundary
chatter does not read as configuration churn.

Frames are assumed grounded (z=0 floor; the *_grounded_yogaonly corpus sits at
0.5 cm lowest-geom clearance).  Quaternions xyzw.  Root frame of the smpl_yogi
Pelvis: +x forward (toes), +y left, +z cranial.

Usage::

    PYTHONPATH=. python data/scripts/extract_contact_configs.py \
        --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \
        --in-dir data/smpl/yoga_motions_proto_yogi_grounded_yogaonly \
        --out-dir data/smpl/yoga_contact_configs [--calibrate] [--workers 8]
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_geometry import (
    geom_pair_distance,
    geom_to_world,
    geom_ground_distance,
    geom_ground_patch,
    parse_typed_geoms,
    witness_velocity,
    world_geom_center,
)

from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate, quat_rotate_inverse


def mjcf_body_names(mjcf_path):
    """Depth-first body-name order of the MJCF -- matches rigid_body_pos indexing
    (same traversal as protomotions.components.pose_lib.extract_kinematic_info,
    without its dm_control dependency)."""
    import xml.etree.ElementTree as ET

    names = []

    def _rec(b):
        names.append(b.attrib["name"])
        for c in b.findall("body"):
            _rec(c)

    for b in ET.parse(mjcf_path).getroot().find("worldbody").findall("body"):
        _rec(b)
    return names


# --------------------------------------------------------------------------- #
# Zones and masks.
# --------------------------------------------------------------------------- #
ZONE_ORDER = [
    "L_FOOT", "R_FOOT", "L_SHANK", "R_SHANK", "L_THIGH", "R_THIGH",
    "PELVIS", "TRUNK", "HEAD",
    "L_UPPER_ARM", "R_UPPER_ARM", "L_FOREARM", "R_FOREARM", "L_HAND", "R_HAND",
]
ZONES = {
    "L_FOOT": ["L_Ankle", "L_Toe"],
    "R_FOOT": ["R_Ankle", "R_Toe"],
    "L_SHANK": ["L_Knee"],
    "R_SHANK": ["R_Knee"],
    "L_THIGH": ["L_Hip"],
    "R_THIGH": ["R_Hip"],
    "PELVIS": ["Pelvis"],
    "TRUNK": ["Torso", "Spine", "Chest", "L_Thorax", "R_Thorax"],
    "HEAD": ["Neck", "Head"],
    "L_UPPER_ARM": ["L_Shoulder"],
    "R_UPPER_ARM": ["R_Shoulder"],
    "L_FOREARM": ["L_Elbow"],
    "R_FOREARM": ["R_Elbow"],
    "L_HAND": ["L_Wrist", "L_Hand"],
    "R_HAND": ["R_Wrist", "R_Hand"],
}

# Kinematically adjacent zone pairs: permanently near by construction, never a
# meaningful contact.  Everything else is eligible (cross-side included).
ADJACENT = {
    frozenset(p)
    for p in [
        ("L_FOOT", "L_SHANK"), ("R_FOOT", "R_SHANK"),
        ("L_SHANK", "L_THIGH"), ("R_SHANK", "R_THIGH"),
        ("L_THIGH", "PELVIS"), ("R_THIGH", "PELVIS"),
        ("PELVIS", "TRUNK"), ("TRUNK", "HEAD"),
        ("TRUNK", "L_UPPER_ARM"), ("TRUNK", "R_UPPER_ARM"),
        ("L_UPPER_ARM", "L_FOREARM"), ("R_UPPER_ARM", "R_FOREARM"),
        ("L_FOREARM", "L_HAND"), ("R_FOREARM", "R_HAND"),
        # Fit artifacts, not anatomy: with arms overhead the head sphere
        # *penetrates* the L shoulder capsule (calibration: 11 % of all corpus
        # frames < 0.5 cm, mass at -3..0 cm) while the R side rarely fires --
        # asymmetric chronic labels for symmetric poses, never load-bearing.
        ("HEAD", "L_UPPER_ARM"), ("HEAD", "R_UPPER_ARM"),
    ]
}

ORIENT_BINS = ["upright", "inverted", "prone", "supine", "side_l", "side_r"]

# Per-zone-category ground thresholds (make, break), calibrated on the corpus
# gap histograms (see calibration.json): hands plant anywhere in 0-4 cm (per-side
# SMPL-fit bias), forearms are cleanly bimodal at 2 cm, everything else has a
# sharp planted mode below 1 cm.
GROUND_THRESHOLDS = {
    "FOOT": (0.02, 0.05),
    "HAND": (0.045, 0.075),
    "FOREARM": (0.02, 0.045),
    "SHANK": (0.025, 0.055),
    "THIGH": (0.025, 0.055),
    "DEFAULT": (0.02, 0.05),
}

DEFAULT_THRESHOLDS = {
    "body_make": 0.02,
    "body_break": 0.05,
    "merge_s": 0.15,     # active runs separated by a shorter break are merged
    "dwell_s": 0.20,     # active runs shorter than this are dropped
    "reorient_s": 0.30,  # orientation-bin runs shorter than this are absorbed
    "static_slip": 0.05, # median witness slip speed below this = static contact
    # Recovery tiers for the corpus-wide SMPL-fit float bias: the per-frame
    # grounding pins only the lowest geom, so genuinely load-bearing parts float
    # 2-16 cm (kneeling shanks 9-11, supine trunk 3-7, wheel feet 14-16).
    # Stillness-gated recovery is driven by the static-support model below
    # (COM/support-polygon feasibility + orientation/bin rules), not by blanket
    # proximity, so deliberately hovering limbs (e.g. firefly's feet at 9 cm)
    # are not latched.
    "body_loose_make": 0.045,   # compressed body-body supports (crow knee 2.8-3.8)
    "body_loose_break": 0.07,
    "body_still_v": 0.06,
    "loose_dwell_s": 0.40,
    "promote_still_v": 0.10,    # witness speed for support-candidate stillness
    "quasi_static_v": 0.15,     # smoothed root speed gating the static model
    # Bilateral completion: COM-sufficiency never adds a statically REDUNDANT
    # support (in downdog, hands + one foot already contain the COM), so when a
    # zone's mirror is strictly planted and the zone itself hovers stationary
    # within this band, it is load-bearing by symmetry (downdog/plank feet at
    # 2-4 cm).  Zones with no planted mirror (firefly's tucked feet) never fire.
    "bilateral_band": 0.06,
    "promote_margin": 0.06,     # COM-to-support-hull distance that triggers promotion
    # Wider than promote_margin: the collision geoms understate real support
    # extent (fingers reach ~10 cm past the hand box, toes ~7 cm past the toe
    # box) and the density-model COM carries a few cm of systematic error, so
    # single-support balance poses legitimately measure up to ~0.15 outside the
    # patch hull (tree max 0.147), while physically impossible support sets
    # measure >= 0.32 (Scale/Tolasana float).  0.20 splits the two bands.
    "inconsistent_margin": 0.20,
}

# Chronically 2.5-4.5 cm apart whenever the legs are together while standing
# (calibration histogram) -- a stillness-gated loose latch would pollute every
# standing config, and no validated pose needs the pair at loose range.
BODY_LOOSE_EXCLUDE = {frozenset(("L_THIGH", "R_THIGH"))}

# Max ground gap per zone category inside which a *stationary* zone may be
# promoted to support, sized from the measured float bands per pose family.
PROMOTE_BANDS = {
    "FOOT": 0.175,      # wheel feet 14.4-15.6 cm (median at band edge needs headroom)
    "SHANK": 0.13,      # low lunge / pigeon kneel 8.7-11.4 cm
    "THIGH": 0.09,      # splits / staff 4.8-6.1 cm
    "PELVIS": 0.125,    # seated: pelvis-sphere bottom sits 8-11.6 cm up
    "TRUNK": 0.10,      # shoulderstand 4.8-6.6, supine 3-7 cm
    "HEAD": 0.15,       # headstand head fit error 13.5-14.2 cm
    "FOREARM": 0.07,    # right-forearm fit elevation 2.9-3.7 cm
    "UPPER_ARM": 0.05,
    "HAND": 0.08,
}

# Capsule-limb zones may only be promoted while near-horizontal (|axis_z|<0.45):
# a kneeling shank is horizontal, a standing shank is vertical at the same 6-9 cm
# hover, so height alone cannot tell them apart.
HORIZONTAL_BLANKET_CATS = {"SHANK", "THIGH", "FOREARM", "UPPER_ARM"}

# Lying-pose blanket: in these orientation bins, stationary zones within their
# band latch directly (fit float is the only reason they read off the floor).
# FOOT is deliberately absent from "prone" (firefly/crow-family hover their feet
# while prone) and "inverted" (tucked inversions).
BIN_BLANKET = {
    "supine": {"FOOT", "PELVIS", "TRUNK", "HEAD", "HAND"},
    "side_l": {"FOOT", "PELVIS", "TRUNK", "HEAD", "HAND"},
    "side_r": {"FOOT", "PELVIS", "TRUNK", "HEAD", "HAND"},
    "prone": {"PELVIS", "TRUNK", "HEAD"},
    "inverted": {"TRUNK", "HEAD"},
    "upright": set(),
}


def zone_cat(zone):
    return zone[2:] if zone[:2] in ("L_", "R_") else zone


def zone_pairs(include_masked=False):
    """Ordered list of pair keys: 'ZONE:G' ground pairs then 'A+B' body pairs."""
    pairs = [f"{z}:G" for z in ZONE_ORDER]
    for i, za in enumerate(ZONE_ORDER):
        for zb in ZONE_ORDER[i + 1 :]:
            if include_masked or frozenset((za, zb)) not in ADJACENT:
                pairs.append(f"{za}+{zb}")
    return pairs


# --------------------------------------------------------------------------- #
# Per-clip geometry pass.
# --------------------------------------------------------------------------- #
class ClipGeometry:
    """World-frame typed geoms + per-pair gap/witness/velocity series for one clip."""

    def __init__(self, motion, body_names, typed_geoms, gate=0.15):
        self.pos = motion["rigid_body_pos"]  # [T,B,3]
        self.rot = motion["rigid_body_rot"]
        self.vel = motion["rigid_body_vel"]
        self.ang_vel = motion["rigid_body_ang_vel"]
        self.T = self.pos.shape[0]
        self.body_index = {n: i for i, n in enumerate(body_names)}
        self.gate = gate
        self._world = {}
        self._typed = typed_geoms

    def world_geom(self, body):
        if body not in self._world:
            b = self.body_index[body]
            geoms = self._typed[body]
            assert len(geoms) == 1, f"{body}: expected exactly 1 collision geom"
            self._world[body] = geom_to_world(geoms[0], self.pos[:, b], self.rot[:, b])
        return self._world[body]

    @staticmethod
    def _bound(g):
        """(center [T,3], bounding radius) of a world geom."""
        if g["type"] == "sphere":
            return g["center"], g["radius"]
        if g["type"] == "capsule":
            mid = (g["a"] + g["b"]) / 2
            return mid, (g["a"] - g["b"]).norm(dim=-1).max().item() / 2 + g["radius"]
        return g["center"], float(g["half"].norm())

    def _combo_distance(self, body_a, body_b):
        """Gap/witnesses for one body pair, exact only where a cheap lower bound
        says the surfaces could be within ``gate`` meters."""
        ga, gb = self.world_geom(body_a), self.world_geom(body_b)
        ca, ra = self._bound(ga)
        cb, rb = self._bound(gb)
        lb = (ca - cb).norm(dim=-1) - ra - rb
        near = lb < self.gate
        gap = lb.clone().clamp_min(self.gate)  # placeholder for far frames
        # Far-frame witnesses fall back to the bounding centers, so a merge-filled
        # far frame can never emit a zero-vector witness (review finding).
        wa = ca.clone()
        wb = cb.clone()
        if near.any():
            idx = near.nonzero(as_tuple=True)[0]
            # Per-frame geom tensors are all 2-D ([T,3] / [T,4]); frame-invariant
            # fields (box half [3], radius, mass) must pass through untouched.
            sub_a = {k: (v[idx] if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in ga.items()}
            sub_b = {k: (v[idx] if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in gb.items()}
            g, a, b = geom_pair_distance(sub_a, sub_b)
            gap[idx], wa[idx], wb[idx] = g, a, b
        return gap, wa, wb

    def pair_series(self, pair_key):
        """Min over member-body combos; returns dict of [T]-tensors."""
        if ":G" in pair_key:
            zone = pair_key.split(":")[0]
            gaps, was, bodies_a = [], [], []
            for body in ZONES[zone]:
                g, w = geom_ground_distance(self.world_geom(body))
                gaps.append(g)
                was.append(w)
                bodies_a.append(body)
            gaps = torch.stack(gaps)  # [C,T]
            was = torch.stack(was)  # [C,T,3]
            best = gaps.argmin(dim=0)  # [T]
            t_idx = torch.arange(self.T)
            gap = gaps[best, t_idx]
            # Zone contact patch: blend member-geom witnesses that are within
            # 1 cm of the zone's min gap, so a flat foot's patch spans the
            # ankle+toe boxes instead of snapping to whichever box is lowest.
            wgt = ((gaps - gap.unsqueeze(0)) < 0.01).float().unsqueeze(-1)  # [C,T,1]
            wa = (was * wgt).sum(dim=0) / wgt.sum(dim=0)
            wa[:, 2] = gap
            wb = wa.clone()
            wb[:, 2] = 0.0
            idx_a = torch.tensor([self.body_index[b] for b in bodies_a])[best]
            idx_b = torch.full((self.T,), -1, dtype=torch.long)  # ground
        else:
            za, zb = pair_key.split("+")
            combos = [(a, b) for a in ZONES[za] for b in ZONES[zb]]
            gaps, was, wbs = [], [], []
            for a, b in combos:
                g, p, q = self._combo_distance(a, b)
                gaps.append(g)
                was.append(p)
                wbs.append(q)
            gaps = torch.stack(gaps)
            best = gaps.argmin(dim=0)
            t_idx = torch.arange(self.T)
            gap = gaps[best, t_idx]
            wa = torch.stack(was)[best, t_idx]
            wb = torch.stack(wbs)[best, t_idx]
            idx_a = torch.tensor([self.body_index[a] for a, _ in combos])[best]
            idx_b = torch.tensor([self.body_index[b] for _, b in combos])[best]

        # Witness-point relative velocity (rigid-body formula on the argmin combo).
        va = witness_velocity(
            wa, self.pos[t_idx, idx_a], self.vel[t_idx, idx_a], self.ang_vel[t_idx, idx_a]
        )
        if ":G" in pair_key:
            vb = torch.zeros_like(va)
        else:
            vb = witness_velocity(
                wb, self.pos[t_idx, idx_b], self.vel[t_idx, idx_b], self.ang_vel[t_idx, idx_b]
            )
        v_rel = va - vb
        normal = wb - wa
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal_speed = (v_rel * normal).sum(-1)  # >0: approaching
        slip = (v_rel - normal_speed.unsqueeze(-1) * normal).norm(dim=-1)
        return {
            "gap": gap, "wa": wa, "wb": wb, "idx_a": idx_a, "idx_b": idx_b,
            "v_rel": v_rel, "normal": normal, "normal_speed": normal_speed, "slip": slip,
        }


# --------------------------------------------------------------------------- #
# Activation logic.
# --------------------------------------------------------------------------- #
def _runs(mask):
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def hysteresis_active(gap, make, brk):
    g = gap.numpy()
    active = np.zeros(len(g), dtype=bool)
    on = False
    for t in range(len(g)):
        on = (g[t] < brk) if on else (g[t] < make)
        active[t] = on
    return active


def loose_hysteresis_active(gap, speed, make, brk, v_still):
    """Stillness-gated activation for the loose foot-ground tier."""
    g, v = gap.numpy(), speed.numpy()
    active = np.zeros(len(g), dtype=bool)
    on = False
    for t in range(len(g)):
        on = (g[t] < brk and v[t] < 2 * v_still) if on else (g[t] < make and v[t] < v_still)
        active[t] = on
    return active


def merge_short_breaks(active, merge_n):
    """Fill inactive gaps shorter than merge_n between active runs."""
    a = active.copy()
    runs = _runs(a)
    for (s1, e1), (s2, _) in zip(runs, runs[1:]):
        if s2 - e1 - 1 < merge_n:
            a[e1 + 1 : s2] = True
    return a


def drop_short_runs(active, dwell_n):
    """Drop active runs shorter than dwell_n."""
    a = active.copy()
    for s, e in _runs(a):
        if e - s + 1 < dwell_n:
            a[s : e + 1] = False
    return a


def clean_active(active, merge_n, dwell_n):
    return drop_short_runs(merge_short_breaks(active, merge_n), dwell_n)


# --------------------------------------------------------------------------- #
# Static-support model: COM vs support polygon.
# --------------------------------------------------------------------------- #
def _median_filter_1d(x, window):
    window = max(1, int(window) | 1)
    if window == 1:
        return x
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def _pt_seg_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _convex_hull(pts):
    """Andrew monotone chain; pts [N,2] -> hull vertices CCW."""
    pts = sorted(map(tuple, pts))
    if len(pts) <= 2:
        return np.array(pts)
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def _point_hull_dist(p, pts):
    """Distance from 2-D point p to the convex hull of pts (0 if inside)."""
    n = len(pts)
    if n == 0:
        return float("inf")
    if n == 1:
        return float(np.linalg.norm(p - pts[0]))
    if n == 2:
        return _pt_seg_dist(p, pts[0], pts[1])
    hull = _convex_hull(np.asarray(pts))
    m = len(hull)
    if m == 1:
        return float(np.linalg.norm(p - hull[0]))
    if m == 2:
        return _pt_seg_dist(p, hull[0], hull[1])
    inside = True
    for i in range(m):
        a, b = hull[i], hull[(i + 1) % m]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:
            inside = False
            break
    if inside:
        return 0.0
    return min(_pt_seg_dist(p, hull[i], hull[(i + 1) % m]) for i in range(m))


def com_xy(clip):
    """Mass-weighted center of mass (XY), [T,2], from the MJCF geom densities."""
    total, acc = 0.0, None
    for body in clip.body_index:
        g = clip.world_geom(body)
        c = world_geom_center(g) * g["mass"]
        acc = c if acc is None else acc + c
        total += g["mass"]
    return (acc / total)[:, :2].numpy()


def zone_horizontal(clip, zone):
    """bool[T]: True where the zone may be treated as a lying/kneeling support.
    Capsule-limb zones require a near-horizontal axis; other zones always True."""
    if zone_cat(zone) not in HORIZONTAL_BLANKET_CATS:
        return np.ones(clip.T, dtype=bool)
    body = ZONES[zone][0]
    g = clip.world_geom(body)
    axis = g["b"] - g["a"]
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return (axis[:, 2].abs() < 0.45).numpy()


def support_promotion(clip, series, strict, bins, th, fps):
    """Promote stationary near-floor zones to ground support where the static
    model requires or licenses it.  Returns (promoted [Z,T] bool aligned with
    ZONE_ORDER, inconsistent [T] bool)."""
    T = clip.T
    Z = len(ZONE_ORDER)
    gaps = np.stack([series[f"{z}:G"]["gap"].numpy() for z in ZONE_ORDER])
    speed = np.stack([series[f"{z}:G"]["v_rel"].norm(dim=-1).numpy() for z in ZONE_ORDER])
    # Support PATCHES: all near-ground surface points of a zone (box corners,
    # capsule endpoints), so a planted foot contributes its whole sole to the
    # hull instead of a single centroid (a one-leg COM legitimately rides
    # anywhere over the foot area).
    patch_pts, patch_near = [], []
    for zi, z in enumerate(ZONE_ORDER):
        pts = torch.cat([geom_ground_patch(clip.world_geom(b)) for b in ZONES[z]], dim=1)
        near = (pts[:, :, 2] - torch.as_tensor(gaps[zi]).unsqueeze(1)) < 0.015
        patch_pts.append(pts[:, :, :2].numpy())
        patch_near.append(near.numpy())
    strict_g = np.stack([strict[f"{z}:G"] for z in ZONE_ORDER])
    cats = [zone_cat(z) for z in ZONE_ORDER]
    band = np.array([PROMOTE_BANDS[c] for c in cats])[:, None]
    horiz = np.stack([zone_horizontal(clip, z) for z in ZONE_ORDER])

    com = com_xy(clip)
    root_speed = _median_filter_1d(
        np.linalg.norm(clip.vel[:, 0].numpy(), axis=-1), round(0.25 * fps)
    )
    quasi = root_speed < th["quasi_static_v"]

    cand = (speed < th["promote_still_v"]) & (gaps < band) & horiz & ~strict_g
    promoted = np.zeros((Z, T), dtype=bool)

    # R2: horizontal capsule limbs (kneeling shank, seated thigh, planted forearm)
    for zi, c in enumerate(cats):
        if c in HORIZONTAL_BLANKET_CATS:
            promoted[zi] = cand[zi] & quasi

    # R3: lying-pose blanket by orientation bin
    for zi, c in enumerate(cats):
        allow = np.array([c in BIN_BLANKET[ORIENT_BINS[b]] for b in bins])
        promoted[zi] |= cand[zi] & quasi & allow

    # R5: bilateral completion (mirror strictly planted, self stationary + near)
    for zi, z in enumerate(ZONE_ORDER):
        if z[:2] not in ("L_", "R_"):
            continue
        mirror = ("R_" if z[:2] == "L_" else "L_") + z[2:]
        mi = ZONE_ORDER.index(mirror)
        promoted[zi] |= (
            quasi
            & strict_g[mi]
            & ~strict_g[zi]
            & (speed[zi] < th["promote_still_v"])
            & (gaps[zi] < th["bilateral_band"])
            & horiz[zi]
        )

    # R4: COM-feasibility greedy promotion + inconsistency flag
    margin = th["promote_margin"]
    inconsistent = np.zeros(T, dtype=bool)

    def zone_patch(zi, t):
        pts = patch_pts[zi][t][patch_near[zi][t]]
        return pts if len(pts) else patch_pts[zi][t][:1]

    for t in np.nonzero(quasi)[0]:
        sup = [zi for zi in range(Z) if strict_g[zi, t] or promoted[zi, t]]
        pts = np.vstack([zone_patch(zi, t) for zi in sup]) if sup else np.zeros((0, 2))
        d = _point_hull_dist(com[t], pts)
        if d <= margin:
            continue
        rem = [zi for zi in range(Z) if cand[zi, t] and not promoted[zi, t]]
        while d > margin and rem:
            best, best_d = None, d
            for zi in rem:
                dd = _point_hull_dist(com[t], np.vstack([pts, zone_patch(zi, t)]))
                if dd < best_d - 1e-9:
                    best, best_d = zi, dd
            if best is None:
                break
            promoted[best, t] = True
            pts = np.vstack([pts, zone_patch(best, t)])
            rem.remove(best)
            d = best_d
        if d > th["inconsistent_margin"]:
            inconsistent[t] = True
    return promoted, inconsistent


def orientation_bins(root_rot, margin=0.15):
    """Sticky-argmax orientation bin: switch only when the challenger's score
    beats the incumbent's by ``margin`` (~8.6 deg past the 45-deg boundary), so
    a trunk pitching around a bin boundary does not flap."""
    down = torch.tensor([0.0, 0.0, -1.0]).expand(root_rot.shape[0], 3)
    g = quat_rotate_inverse(root_rot, down, w_last=True)  # gravity in root frame
    scores = torch.stack([-g[:, 2], g[:, 2], g[:, 0], -g[:, 0], g[:, 1], -g[:, 1]], dim=1).numpy()
    out = np.empty(scores.shape[0], dtype=np.int64)
    cur = int(scores[0].argmax())
    for t in range(scores.shape[0]):
        top = int(scores[t].argmax())
        if top != cur and scores[t, top] > scores[t, cur] + margin:
            cur = top
        out[t] = cur
    return out, g


def stabilize_bins(raw, min_n):
    """Absorb orientation-bin runs shorter than min_n into the surrounding bin."""
    out = raw.copy()
    runs, s = [], 0
    for t in range(1, len(raw) + 1):
        if t == len(raw) or raw[t] != raw[s]:
            runs.append((s, t - 1, raw[s]))
            s = t
    long_runs = [r for r in runs if r[1] - r[0] + 1 >= min_n]
    if not long_runs:
        vals, counts = np.unique(raw, return_counts=True)
        out[:] = vals[counts.argmax()]
        return out
    cur = long_runs[0][2]
    for s, e, b in runs:
        if e - s + 1 >= min_n:
            cur = b
        out[s : e + 1] = cur
    return out


# --------------------------------------------------------------------------- #
# Snapshot helpers.
# --------------------------------------------------------------------------- #
def _f(x):
    if torch.is_tensor(x):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [round(float(v), 5) for v in x]
    return round(float(x), 5)


def config_string(active_pairs, obin):
    tokens = sorted(active_pairs, key=lambda p: (":G" not in p, p))
    return ("|".join(tokens) if tokens else "NONE") + "@" + obin


def config_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]


class HeadingFrame:
    def __init__(self, motion):
        self.root_pos = motion["rigid_body_pos"][:, 0]
        self.root_rot = motion["rigid_body_rot"][:, 0]
        self.root_vel = motion["rigid_body_vel"][:, 0]
        self.h_inv = calc_heading_quat_inv(self.root_rot, w_last=True)

    def to_root(self, world_pts, t):
        rel = world_pts - self.root_pos[t]
        return quat_rotate(self.h_inv[t : t + 1].expand(rel.shape[0] if rel.dim() > 1 else 1, 4),
                           rel.unsqueeze(0) if rel.dim() == 1 else rel, w_last=True).squeeze(0)

    def vec_to_heading(self, vec, t):
        v = vec.unsqueeze(0) if vec.dim() == 1 else vec
        return quat_rotate(self.h_inv[t : t + 1].expand(v.shape[0], 4), v, w_last=True).squeeze(0)


# --------------------------------------------------------------------------- #
# Per-clip extraction.
# --------------------------------------------------------------------------- #
def extract_clip(path, mjcf, thresholds, calibrate=False):
    torch.set_num_threads(1)
    motion = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(motion, dict) or "rigid_body_pos" not in motion:
        return None
    fps = int(motion.get("fps", 60))
    body_names = mjcf_body_names(mjcf)
    typed = parse_typed_geoms(mjcf, body_names)
    clip = ClipGeometry(motion, body_names, typed)
    T = clip.T

    pairs = zone_pairs(include_masked=calibrate)
    series = {p: clip.pair_series(p) for p in pairs}

    if calibrate:
        edges = np.arange(-0.05, 0.6001, 0.002)
        out = {}
        for p, s in series.items():
            g = s["gap"].numpy()
            out[p] = {
                "hist": np.histogram(np.clip(g, -0.049, 0.599), bins=edges)[0].tolist(),
                "min": float(g.min()),
                "frames": int(T),
                "frac_below": {str(th): float((g < th).mean()) for th in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)},
            }
        return {"motion": path.stem, "fps": fps, "pairs": out, "hist_edges_mm": [-50, 600, 2]}

    th = thresholds
    merge_n = round(th["merge_s"] * fps)
    dwell_n = round(th["dwell_s"] * fps)
    loose_dwell_n = round(th["loose_dwell_s"] * fps)

    raw_bins, gravity_root = orientation_bins(clip.rot[:, 0])
    bins = stabilize_bins(raw_bins, round(th["reorient_s"] * fps))

    # Strict geometric tier.
    strict, make_th = {}, {}
    for p, s in series.items():
        if ":G" in p:
            cat = zone_cat(p.split(":")[0])
            make, brk = GROUND_THRESHOLDS.get(cat, GROUND_THRESHOLDS["DEFAULT"])
        else:
            make, brk = th["body_make"], th["body_break"]
        make_th[p] = make
        strict[p] = clean_active(hysteresis_active(s["gap"], make, brk), merge_n, dwell_n)

    # Loose tiers: stillness-gated body-body (compressed supports), and the
    # static-support model for ground contacts.  Each loose mask is cleaned on
    # its own dwell, then the union gets a final merge-only pass so the OR of
    # independently cleaned tiers cannot re-introduce sub-merge_n churn.
    active = {}
    for p, s in series.items():
        a = strict[p]
        if ":G" not in p and frozenset(p.split("+")) not in BODY_LOOSE_EXCLUDE:
            loose = loose_hysteresis_active(
                s["gap"], s["v_rel"].norm(dim=-1),
                th["body_loose_make"], th["body_loose_break"], th["body_still_v"],
            )
            a = merge_short_breaks(a | clean_active(loose, merge_n, loose_dwell_n), merge_n)
        active[p] = a

    promoted, inconsistent_frames = support_promotion(clip, series, strict, bins, th, fps)
    for zi, z in enumerate(ZONE_ORDER):
        p = f"{z}:G"
        promo = clean_active(promoted[zi], merge_n, loose_dwell_n)
        active[p] = merge_short_breaks(active[p] | promo, merge_n)

    hf = HeadingFrame(motion)

    # ------- segment boundaries: any change in (active set, orientation bin) --
    state_change = np.zeros(T, dtype=bool)
    state_change[0] = True
    for p in pairs:
        a = active[p]
        state_change[1:] |= a[1:] != a[:-1]
    state_change[1:] |= bins[1:] != bins[:-1]
    boundaries = np.nonzero(state_change)[0].tolist() + [T]

    def snapshot_root(t):
        return {
            "root_height": _f(hf.root_pos[t, 2]),
            "gravity_in_root": _f(gravity_root[t]),
            "root_vel_heading": _f(hf.vec_to_heading(hf.root_vel[t], t)),
            "orientation_bin": ORIENT_BINS[bins[t]],
        }

    def contact_detail(p, t0, t1, mid):
        s = series[p]
        seg_gap = s["gap"][t0:t1]
        # Representative frame = deepest contact in the segment: guaranteed to be
        # a real geometry frame, never a merge-filled far frame with placeholder
        # witnesses (review finding).
        k = t0 + int(seg_gap.argmin())
        near = seg_gap[seg_gap < 0.149]  # exclude gated placeholder values
        mean_gap = near.mean() if near.numel() else seg_gap.mean()
        seg_slip = s["slip"][t0:t1]
        body_a = body_names[s["idx_a"][k]]
        body_b = None if ":G" in p else body_names[s["idx_b"][k]]
        return {
            "pair": p,
            "bodies": [body_a, body_b],
            "mean_gap": _f(mean_gap),
            "tier": "planted" if float(mean_gap) < make_th[p] else "inferred",
            "witness_root": _f(hf.to_root(s["wa"][k], k)),
            "slip_p50": _f(seg_slip.median()),
            "static": bool(seg_slip.median() < th["static_slip"]),
        }

    segments, events = [], []
    prev_active = {p: False for p in pairs}
    prev_bin = None
    win = max(1, round(0.1 * fps))  # approach-velocity window
    for k in range(len(boundaries) - 1):
        t0, t1 = boundaries[k], boundaries[k + 1]  # segment [t0, t1)
        mid = (t0 + t1) // 2
        act = [p for p in pairs if active[p][t0]]
        obin = ORIENT_BINS[bins[t0]]
        cfg = config_string(act, obin)
        segments.append(
            {
                "config": cfg,
                "hash": config_hash(cfg),
                "start_frame": int(t0),
                "end_frame": int(t1 - 1),
                "duration_s": _f((t1 - t0) / fps),
                "contacts": [contact_detail(p, t0, t1, mid) for p in act],
                # True when the detected+promoted support set still cannot place
                # the COM within the support hull for most of the segment: the
                # fit floats every real support out of band (known cases: Scale/
                # Tolasana, cockerel press) -- treat the segment's contacts as
                # unreliable.
                "statically_inconsistent": bool(inconsistent_frames[t0:t1].mean() > 0.3),
                **snapshot_root(mid),
            }
        )
        # ------- events at this boundary (vs previous segment state) ----------
        if k > 0:
            prev_cfg = segments[-2]["config"]
            for p in pairs:
                now = active[p][t0]
                if now == prev_active[p]:
                    continue
                s = series[p]
                etype = "make" if now else "break"
                if etype == "make":
                    w0, w1 = max(0, t0 - win), max(1, t0)
                else:
                    w0, w1 = t0, min(T, t0 + win)
                events.append(
                    {
                        "frame": int(t0),
                        "t": _f(t0 / fps),
                        "type": etype,
                        "pair": p,
                        "bodies": [
                            body_names[s["idx_a"][t0]],
                            None if ":G" in p else body_names[s["idx_b"][t0]],
                        ],
                        "config_before": prev_cfg,
                        "config_after": cfg,
                        "gap": _f(s["gap"][t0]),
                        "witness_root": _f(hf.to_root(s["wa"][t0], t0)),
                        "witness_world": _f(s["wa"][t0]),
                        "approach_speed": _f(s["normal_speed"][w0:w1].mean()),
                        "rel_vel_heading": _f(hf.vec_to_heading(s["v_rel"][t0], t0)),
                        "slip_speed": _f(s["slip"][t0]),
                        **snapshot_root(t0),
                    }
                )
            if bins[t0] != prev_bin:
                events.append(
                    {
                        "frame": int(t0),
                        "t": _f(t0 / fps),
                        "type": "reorient",
                        "pair": None,
                        "bodies": [None, None],
                        "config_before": prev_cfg,
                        "config_after": cfg,
                        **snapshot_root(t0),
                    }
                )
        prev_active = {p: active[p][t0] for p in pairs}
        prev_bin = bins[t0]

    return {
        "motion": path.stem,
        "fps": fps,
        "num_frames": int(T),
        "thresholds": thresholds,
        "segments": segments,
        "events": events,
    }


# --------------------------------------------------------------------------- #
def _worker(args):
    path, mjcf, thresholds, calibrate = args
    try:
        with torch.no_grad():
            return extract_clip(path, mjcf, thresholds, calibrate)
    except Exception as e:  # surface per-clip failures without killing the pool
        return {"motion": path.stem, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calibrate", action="store_true", help="emit per-pair gap statistics instead of events")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    for k, v in DEFAULT_THRESHOLDS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()
    thresholds = {k: getattr(args, k) for k in DEFAULT_THRESHOLDS}

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.motion"))
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if f.name in wanted]
    print(f"{len(files)} clips, calibrate={args.calibrate}")

    jobs = [(f, args.mjcf, thresholds, args.calibrate) for f in files]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_worker, jobs)):
            if r is None:
                continue
            results.append(r)
            tag = "ERR " + r["error"] if "error" in r else (
                f"{len(r.get('segments', r.get('pairs', [])))} segs, {len(r.get('events', []))} events"
                if not args.calibrate else "calibrated"
            )
            print(f"[{i + 1}/{len(files)}] {r['motion'][:60]:60s} {tag}")

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    if args.calibrate:
        agg = {}
        for r in ok:
            for p, st in r["pairs"].items():
                a = agg.setdefault(
                    p, {"hist": np.zeros(len(st["hist"]), dtype=np.int64), "mins": [], "frames": 0,
                        "frac_below": {k: 0.0 for k in st["frac_below"]}}
                )
                a["hist"] += np.array(st["hist"])
                a["mins"].append(st["min"])
                a["frames"] += st["frames"]
                for k, v in st["frac_below"].items():
                    a["frac_below"][k] += v * st["frames"]
        report = {}
        for p, a in agg.items():
            report[p] = {
                "frames": a["frames"],
                "frac_below": {k: round(v / a["frames"], 4) for k, v in a["frac_below"].items()},
                "clip_min_p10_p50_p90": [
                    round(float(np.percentile(a["mins"], q)), 4) for q in (10, 50, 90)
                ],
                "hist": a["hist"].tolist(),
            }
        (out_dir / "calibration.json").write_text(json.dumps(
            {"hist_edges": "[-0.05, 0.60] step 0.002", "pairs": report}, indent=1))
        print(f"\nwrote {out_dir / 'calibration.json'}")
        print("\nMost chronically close pairs (frac of frames < 2 cm):")
        rows = sorted(report.items(), key=lambda kv: -kv[1]["frac_below"]["0.02"])
        for p, st in rows[:25]:
            adj = " [masked-adjacent]" if "+" in p and frozenset(p.split("+")) in ADJACENT else ""
            print(f"  {p:28s} {st['frac_below']['0.02']:6.3f}{adj}")
    else:
        for r in ok:
            (out_dir / f"{r['motion']}.json").write_text(json.dumps(r, indent=1))
        # corpus summary: config occurrence + directed transitions
        configs, transitions = {}, {}
        for r in ok:
            for seg in r["segments"]:
                c = configs.setdefault(
                    seg["config"],
                    {"hash": seg["hash"], "clips": set(), "total_dwell_s": 0.0, "n_segments": 0},
                )
                c["clips"].add(r["motion"])
                c["total_dwell_s"] += seg["duration_s"]
                c["n_segments"] += 1
            for ev in r["events"]:
                key = ev["config_before"] + " -> " + ev["config_after"]
                t = transitions.setdefault(
                    key, {"count": 0, "clips": set(), "via": {}})
                t["count"] += 1
                t["clips"].add(r["motion"])
                via = f"{ev['type']}:{ev['pair']}" if ev["pair"] else "reorient"
                t["via"][via] = t["via"].get(via, 0) + 1
        for c in configs.values():
            c["clips"] = sorted(c["clips"])
            c["total_dwell_s"] = round(c["total_dwell_s"], 2)
        for t in transitions.values():
            t["clips"] = sorted(t["clips"])
        summary = {
            "num_clips": len(ok),
            "num_configs": len(configs),
            "num_transitions": len(transitions),
            "configs": dict(sorted(configs.items(), key=lambda kv: -kv[1]["total_dwell_s"])),
            "transitions": dict(sorted(transitions.items(), key=lambda kv: -kv[1]["count"])),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"\n{len(ok)} clips -> {len(configs)} distinct configs, "
              f"{len(transitions)} distinct transitions")
        print(f"wrote per-clip JSON + summary.json to {out_dir}")

    if errors:
        print(f"\n{len(errors)} clips FAILED:")
        for r in errors:
            print(f"  {r['motion']}: {r['error']}")


if __name__ == "__main__":
    main()
