# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Labels v1 (BUILD_PLAN Step 6): the corrected contact labels, reconciled deterministically from the
capture, and the informed reviewer's (Pass B) claims attached to them.

``build()`` takes every hold of the Step 2 audit and ``write()`` puts the result in
``data/reference_curation/labels/<labels_id>/``:

* ``holds.yaml``: the source manifest with corrected ground sets, pairs, boundaries and exemplars.
  Every source field is kept, so the existing consumers read it; each hold adds ``hold_id`` and a
  ``labels`` block (status, changes, the source values, roles, review).
* ``annotations.jsonl``: one record per (hold, zone) and per (hold, pair) the source label names or
  the human makes at the exemplar: source state and channel, load-path class, target role, status,
  evidence values and the ``evidence_ids`` they come from.
* ``evidence.jsonl``: what every evidence id is (kind, path, sha256 or store identity).
* ``labels.json`` (provenance, configuration, metrics) and ``summary.md``.

``labels_id`` is ``<manifest stem>.labels_v1.<hash>``; the hash covers this module, the stores, the
audit, the manifest and every verdict used, so a rerun on the same inputs rewrites the same folder.

Hold identity
-------------
``hold_id`` is Step 2's ``<stem>@<frame_hold>`` of the *source* hold. It never follows a moved
exemplar, so a label revision renames nothing, and two holds of one clip keep distinct ids even when
they share a name, a ground set and an orientation (Side Crow -c's two hands-only holds). The
corrected exemplar is ``frame_hold``.

Source state
------------
``sources.ground_source``: the mesh with seam arbitration (store v3), then the markers, then the
mat's contact. ``sources.pair_source``: the mesh. A frame *matches* a ground set when every zone the
source decides agrees with it (unknown agrees with anything). It is *stable* when no zone changes
state within ±0.5 s (``audit.support_changes``).

Window, exemplar and ground set
-------------------------------
Per hold, inside the source window:

``kept``        the source exemplar matches the label's ground set and is stable;
``moved``       another frame does: the stillest of them (the proposer's own exemplar rule, minimum
                smoothed mean body speed, restricted to capture-good frames);
``relabelled``  no frame does: the ground set becomes the support set held on most stable frames
                (the exemplar's set on a tie), and the exemplar the source one if it matches it and
                is stable, else the stillest frame that does;
``unresolved``  no frame is stable: the exemplar is the frame farthest from a change (then the
                nearest to the source one), and the ground set its support set.

The ground set is what the source says touches at the exemplar, plus labelled zones it cannot
decide there (carried forward, unverified: unknown is not absent). So on every decided zone it
equals the capture, at the exemplar and on every frame of the window. The window is the run of
matching frames around the exemplar, inside the source window: it can shrink, never grow. A moved
exemplar gets its speed, pelvis height, orientation bin and rest-pose distance recomputed as the
proposer computes them, on the shipped motion.

Load paths (Tier 1, ``notes/Contact_label_consequence.MD`` §6)
------------------------------------------------------------
``external_support``        a ground contact of the human.
``consequential_internal``  a body-body contact that, added alone to the kinematic tree, gives some
                            zone a strictly shorter route to the human's ground set, counted in
                            joints (the contact itself bypasses joints), and for two zones on
                            different chains a route that avoids their common root: it can carry
                            weight to the floor (a shin on the upper arm, the trunk on the thighs).
``internal_brace``          one that does not: a loop between chains that meet at the pelvis or
                            reach the floor anyway (leg-leg, the chair's knees at 754 N). Never a
                            support.
``incidental``              one with no ground to reach (the ground set is empty).

Shortening alone is not enough: a shin on the other thigh skips its own knee and hip but re-enters
through the other hip, one joint shorter, and the note measured 0 N there (both legs hang from the
pelvis). Avoiding the root alone is not enough either: the chair's knees reach the floor through
the other shin's foot, no shorter than through their own. The note's six measured cases and the
chair come out as it reports them (tests).

Roles (provisional, deterministic)
----------------------------------
Ground, in the ground set: ``required_support`` when the mat confirms load (the attributed load's
window median >= 50 N over the coverage-valid frames where the zone is attribution-visible, if at
least half qualify), else ``required_touch``: the touch is measured, the load is not (a floating
avatar zone is invisible to the attribution). A carried zone is ``unspecified``. A labelled zone the
source says is lifted is ``forbidden_support``.

Pairs the human touches (stable at the exemplar): ``required_touch`` when consequential and labelled
(the knee shelves), ``allowed`` when consequential but unlabelled or labelled but a brace,
``incidental`` when an unlabelled brace or with no ground. A contact that flips within ±0.5 s, or
any contact of an ``unresolved`` hold, is ``unresolved``. A pair the human does not touch is
``unspecified`` (a labelled one is a proximity latch); an unknown labelled pair is ``unspecified``
and carried. Statics (Step 7) refines these; the reviewer's roles are attached, and decide nothing
until a Pass-B calibration admits the ``roles`` class.

The configuration (``pairs`` in ``holds.yaml``) is what every current consumer reads as *demanded*,
so it carries only positive goal contacts (the design's legacy-vector rule): the ground set, the
``required_touch`` pairs, and the labels the capture cannot decide (carried, listed as
``unverified``). Every other role lives in the annotations for the Step 9 sidecar. A labelled pair
the human touches but that is not required is *demoted* out of the configuration, never deleted; an
unlabelled one is *recorded*, and enters only through an admitted reviewer role. Two configured
pairs that share a zone and whose other zones are adjacent segments of one limb (a knee straddling
shin and thigh against one upper arm) form an ``any_of`` alternative group.

Review (Pass B)
---------------
``load_reviews`` reads the valid Pass-B verdicts of the configured reviewer from the ledger, one per
hold (its current packet if it has one, else the most recent), and ``parse_review`` turns each into
claims. A claim class decides only if the Pass-B calibration table admits it as evidence. Otherwise
the claims are attached as ``review`` fields and counted in the metrics. ``floor_human`` claims are
compared with the source at the packet's main moment (the corrected exemplar, for ``informed``
packets). An admitted class that disagrees there makes the annotation ``unresolved`` (the capture
still decides the ground set), but only on the zones the class was calibrated on: the feet, hands
and head, where ``verdicts.packet_truth`` has marker truth. Elsewhere its claims stay advisory: the
reviewer reads a seated pelvis's waist markers 11 cm up as "off the floor" (Scale -a), where the
prompt's "1 to 4 cm" holds only for hands and feet.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.labels
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from extract_contact_configs import ORIENT_BINS, ZONE_ORDER, orientation_bins, stabilize_bins
from propose_hold_manifest import goal_frame_pose, median_filter, pose_distance
from reference_curation import audit, capture, human_mesh as hm, ids, packets, review, sources, verdicts

MODULE = "reference_curation.labels"
SCHEMA_VERSION = 1
LABELS_VERSION = "v1"
LABELS_DIR = ids.DATA_ROOT / "labels"
PASS_B_CALIBRATION = ids.DATA_ROOT / "calibration" / "pass_b"

SUPPORT_LOAD_N = capture.CALIB_LOAD_N
ZI = {z: i for i, z in enumerate(ZONE_ORDER)}
PAIR_COLUMN = {p: k for k, p in enumerate(hm.PAIRS)}
TREE_EDGES = tuple(tuple(ZI[z] for z in sorted(e, key=ZI.get)) for e in hm.KINEMATIC_ADJACENT)
LIMB_CHAINS = {frozenset((f"{s}_{a}", f"{s}_{b}")) for s in "LR"
               for a, b in (("FOOT", "SHANK"), ("SHANK", "THIGH"), ("HAND", "FOREARM"), ("FOREARM", "UPPER_ARM"))}
STATES = {1: "observed_contact", 0: "observed_separation", -1: "unknown"}
HOLD_STATUS = ("kept", "moved", "relabelled", "unresolved")
LOAD_PATH_CLASSES = ("external_support", "consequential_internal", "internal_brace", "incidental")
ROLES = ("required_support", "required_touch", "allowed", "incidental", "forbidden_support", "unspecified",
         "unresolved")
REVIEW_ROLES = ("required_touch", "allowed", "incidental")
CONFIG = {"stable_s": audit.STABLE_S, "support_load_n": SUPPORT_LOAD_N, "coverage_gate_col0": audit.COVERAGE_GATE,
          "min_valid": audit.MIN_VALID, "hover_cm": 100 * audit.HOVER_M, "exemplar": "stillest capture-good frame",
          "sources": sources.CONFIG}


def _num(x, nd: int = 3):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _cm(x):
    return _num(100.0 * float(x), 2) if x is not None else None


@functools.lru_cache(maxsize=1)
def floor_human_scope() -> frozenset:
    """The zones ``floor_human`` is calibrated on: where ``verdicts.packet_truth`` has human truth,
    the zones capture v1 admits (feet, hands, head)."""
    return frozenset(z for z, row in capture.load_calibration()["zones"].items() if row["admitted"])


def pair_name(pair) -> str:
    return "+".join(pair)


def parse_pair(name: str) -> tuple[str, str]:
    a, b = sorted(name.split("+"), key=ZI.get)
    if (a, b) not in PAIR_COLUMN:
        raise ValueError(f"{name}: not a non-adjacent zone pair")
    return a, b


# --------------------------------------------------------------------------- #
# Tier 1: load paths over the kinematic tree (pure)
# --------------------------------------------------------------------------- #
def _tree_parents() -> dict:
    """Each zone's parent in the kinematic tree rooted at the pelvis."""
    parent, todo = {"PELVIS": None}, ["PELVIS"]
    while todo:
        z = todo.pop()
        for e in hm.KINEMATIC_ADJACENT:
            if z in e:
                (n,) = e - {z}
                if n not in parent:
                    parent[n] = z
                    todo.append(n)
    return parent


PARENT = _tree_parents()


def common_root(a: str, b: str) -> str:
    """The lowest common ancestor of two zones (one of them, if they share a chain)."""
    up = [a]
    while PARENT[up[-1]] is not None:
        up.append(PARENT[up[-1]])
    while b not in up:
        b = PARENT[b]
    return b


def joint_distances(ground, extra=(), removed: str | None = None) -> np.ndarray:
    """``[Z]`` the fewest joints from each zone to a grounded zone (``inf`` with no ground). Tree edges
    are joints (1); ``extra`` body-body contacts bypass joints (0); a ``removed`` zone is not passed."""
    d = np.full(len(ZONE_ORDER), np.inf)
    d[[ZI[z] for z in ground if z != removed]] = 0.0
    cut = None if removed is None else ZI[removed]
    edges = [(i, j, w) for i, j, w in [(i, j, 1.0) for i, j in TREE_EDGES] + [(ZI[a], ZI[b], 0.0) for a, b in extra]
             if cut not in (i, j)]
    for _ in range(len(ZONE_ORDER)):
        changed = False
        for i, j, w in edges:
            if d[i] + w < d[j]:
                d[j], changed = d[i] + w, True
            if d[j] + w < d[i]:
                d[i], changed = d[j] + w, True
        if not changed:
            break
    return d


def load_path_class(pair, ground) -> str:
    """Tier 1: consequential when the contact gives some zone a strictly shorter route to the ground,
    and for two chains a route that avoids their common root. Both legs hang from the pelvis, so a
    shin on the other thigh reaches no floor the pelvis could not (side crow, 0 N measured), and the
    chair's knees give no shorter route than each shin's own foot (754 N, no support)."""
    if not ground:
        return "incidental"
    root = common_root(*pair)
    detour = joint_distances(ground, [pair], removed=None if root in pair else root)
    return "consequential_internal" if (detour < joint_distances(ground)).any() else "internal_brace"


# --------------------------------------------------------------------------- #
# One clip's evidence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stores:
    v1: Path = capture.STORE_DIR
    v2: Path = hm.STORE_DIR
    v3: Path = sources.STORE_DIR


@dataclass(frozen=True)
class Clip:
    stem: str
    rec: capture.Capture
    state: np.ndarray       # [T,Z] source ground state
    channel: np.ndarray     # [T,Z] its channel (sources.CHANNELS)
    pairs: np.ndarray       # [T,P] source pair state
    change_dist: np.ndarray  # [T] frames to the nearest support change (inf: none)
    zone_dist: np.ndarray   # [T,Z] frames to the nearest change of each zone
    pair_dist: np.ndarray   # [T,P] frames to the nearest change of each pair
    speed: np.ndarray       # [T] the proposer's smoothed mean body speed, shipped motion
    bins: np.ndarray        # [T] the proposer's stabilised orientation bin
    pos: np.ndarray         # [T,B,3] shipped motion
    rot: np.ndarray         # [T,B,4]

    @property
    def fps(self) -> int:
        return self.rec.fps

    @property
    def num_frames(self) -> int:
        return self.state.shape[0]


def _distance_to(changes: np.ndarray, T: int) -> np.ndarray:
    """``[T]`` frames from each frame to the nearest of ``changes``."""
    if not len(changes):
        return np.full(T, np.inf)
    t = np.arange(T)
    k = np.clip(np.searchsorted(changes, t), 1, len(changes)) - 1
    left = np.abs(t - changes[k])
    right = np.abs(changes[np.minimum(k + 1, len(changes) - 1)] - t)
    return np.minimum(left, right).astype(float)


def _per_column_dist(states: np.ndarray) -> np.ndarray:
    return np.stack([_distance_to(audit.support_changes(states[:, [k]]), len(states))
                     for k in range(states.shape[1])], -1)


def load_record(stem: str, stores: Stores = Stores()) -> capture.Capture:
    """Capture store v3 of ``stem`` over the given v1 / v2 / v3 directories, with the committed
    calibrations."""
    base_v1 = capture.load(stem, stores.v1)
    base_v2 = hm.load(stem, stores.v2, base=base_v1)
    return sources.load(stem, stores.v3, base_v2)


def clip_evidence(stem: str, manifest: dict, stores: Stores = Stores()) -> Clip:
    rec = load_record(stem, stores)
    state, channel = sources.ground_source(rec)
    pairs = sources.pair_source(rec)
    T, fps = state.shape[0], rec.fps
    motion = capture._load_motion(ids.motion_path(stem))
    if motion["rigid_body_pos"].shape[0] != T:
        raise ValueError(f"{stem}: motion has {motion['rigid_body_pos'].shape[0]} frames, the store {T}")
    speed = median_filter(motion["rigid_body_vel"].norm(dim=-1).mean(dim=-1).numpy(),
                          round(manifest["stillness"]["smooth_s"] * fps))
    raw, _ = orientation_bins(motion["rigid_body_rot"][:, 0])
    bins = stabilize_bins(raw, round(manifest["detector_thresholds"]["reorient_s"] * fps))
    return Clip(stem, rec, state, channel, pairs, _distance_to(audit.support_changes(state), T),
                _per_column_dist(state), _per_column_dist(pairs), speed, bins,
                motion["rigid_body_pos"].double().numpy(), motion["rigid_body_rot"].double().numpy())


# --------------------------------------------------------------------------- #
# Window, exemplar and ground set (pure)
# --------------------------------------------------------------------------- #
def matching(state: np.ndarray, ground) -> np.ndarray:
    """``[T]`` frames on which every decided zone agrees with ``ground``."""
    g = np.array([z in ground for z in ZONE_ORDER])
    return (((state == 1) == g) | (state == -1)).all(-1)


def support_at(clip: Clip, t: int) -> frozenset:
    return frozenset(z for z in ZONE_ORDER if clip.state[t, ZI[z]] == 1)


def choose(clip: Clip, f0: int, f1: int, fh: int, label) -> SimpleNamespace:
    """``status``, ``frame_hold``, ``ground``, ``frame_start``, ``frame_end`` of one hold (see the
    module docstring)."""
    if not f0 <= fh <= f1:
        raise ValueError(f"exemplar {fh} outside the window {f0}-{f1}")
    near = round(audit.STABLE_S * clip.fps)
    win = np.arange(f0, f1 + 1)
    stable = clip.change_dist[win] > near
    ok = matching(clip.state, label)[win] & stable

    def stillest(frames):
        return int(frames[np.argmin(clip.speed[frames])])

    if ok[fh - f0]:
        status, ex = "kept", fh
    elif ok.any():
        status, ex = "moved", stillest(win[ok])
    elif stable.any():
        sets = collections.Counter(support_at(clip, int(t)) for t in win[stable])
        at_fh = support_at(clip, fh)
        best = max(sets, key=lambda s: (sets[s], s == at_fh, tuple(sorted(-ZI[z] for z in s))))
        ok = matching(clip.state, best)[win] & stable
        status, ex = "relabelled", fh if ok[fh - f0] else stillest(win[ok])
    else:
        status, ex = "unresolved", int(max(win, key=lambda t: (clip.change_dist[t], -abs(int(t) - fh), -int(t))))
    ground = support_at(clip, ex) | {z for z in label if clip.state[ex, ZI[z]] == -1}
    m = matching(clip.state, ground)
    s = e = ex
    while s > f0 and m[s - 1]:
        s -= 1
    while e < f1 and m[e + 1]:
        e += 1
    return SimpleNamespace(status=status, frame_hold=ex, ground=frozenset(ground), frame_start=s, frame_end=e)


def exemplar_fields(clip: Clip, frame: int) -> dict:
    """The proposer's per-exemplar fields, recomputed on the shipped motion."""
    pose = goal_frame_pose(clip.pos, clip.rot, frame)
    rest = min(pose_distance(pose, goal_frame_pose(clip.pos, clip.rot, 0)),
               pose_distance(pose, goal_frame_pose(clip.pos, clip.rot, clip.num_frames - 1)))
    return {"orientation": ORIENT_BINS[int(clip.bins[frame])], "speed_at_hold": _num(clip.speed[frame], 4),
            "pelvis_z": _num(clip.pos[frame, 0, 2], 3), "rest_pose_m": _num(rest, 3)}


# --------------------------------------------------------------------------- #
# Annotations of one hold (pure)
# --------------------------------------------------------------------------- #
def _window_fraction(states: np.ndarray) -> float | None:
    known = states >= 0
    return _num((states[known] == 1).mean()) if known.any() else None


def mat_load(rec: capture.Capture, s: int, e: int, zi: int) -> tuple[float | None, float]:
    """``(load_n, visible_fraction)``: the attributed load's median over the window's coverage-valid
    frames where the zone is attribution-visible, if at least half the window qualifies (Step 2's
    gate)."""
    cov = np.nan_to_num(rec["mat_valid_cov"][s:e + 1], nan=0.0) >= audit.COVERAGE_GATE
    seen = cov & rec["attr_visible"][s:e + 1, zi]
    load = _num(np.median(rec["mat_zone_load"][s:e + 1, zi][seen]), 1) if seen.mean() >= audit.MIN_VALID else None
    return load, _num(rec["attr_visible"][s:e + 1, zi].mean())


def ground_role(in_ground: bool, state: int, load_n: float | None) -> str:
    if in_ground:
        if state == -1:
            return "unspecified"
        return "required_support" if load_n is not None and load_n >= SUPPORT_LOAD_N else "required_touch"
    return "forbidden_support" if state == 0 else "unspecified"


def pair_role(state: int, stable: bool, labelled: bool, lp: str, hold_unresolved: bool,
              admitted_role: str | None = None) -> str:
    """The provisional role (module docstring). ``admitted_role`` is a reviewer's role from a class the
    Pass-B calibration admits; it decides a stable contact, and asking for a touch the human did not
    make leaves the pair unresolved."""
    if state != 1:
        return "unresolved" if admitted_role == "required_touch" and state == 0 else "unspecified"
    if not stable or hold_unresolved:
        return "unresolved"
    if admitted_role is not None:
        return admitted_role
    if lp == "consequential_internal":
        return "required_touch" if labelled else "allowed"
    if lp == "internal_brace" and labelled:
        return "allowed"
    return "incidental"


def alternative_groups(hold_id: str, configured: list[tuple[str, str]]) -> dict:
    """``{pair: group id}``: configured pairs sharing a zone whose other zones are adjacent segments of
    one limb, merged by union-find into ``any_of`` groups."""
    parent = {p: p for p in configured}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for i, p in enumerate(configured):
        for q in configured[i + 1:]:
            shared = set(p) & set(q)
            if len(shared) == 1 and frozenset(set(p) ^ set(q)) in LIMB_CHAINS:
                parent[find(p)] = find(q)
    groups = collections.defaultdict(list)
    for p in configured:
        groups[find(p)].append(p)
    out = {}
    for members in groups.values():
        if len(members) > 1:
            gid = f"{hold_id}/any_of/" + "|".join(pair_name(p) for p in sorted(members, key=PAIR_COLUMN.get))
            out.update({p: gid for p in members})
    return out


def annotate(clip: Clip, record: dict, hold: dict, choice: SimpleNamespace, avatar_gaps: dict, ev_ids: dict,
             review_claims: dict | None, admitted: tuple = ()) -> list[dict]:
    """The annotations of one hold: its ground zones, then its pairs."""
    rec, hold_id = clip.rec, record["hold_id"]
    ex, s, e = choice.frame_hold, choice.frame_start, choice.frame_end
    near = round(audit.STABLE_S * clip.fps)
    label = set(record["label_ground"])
    label_pairs = {parse_pair(p) for p in hold["pairs"] if ":" not in p}
    hold_unresolved = choice.status == "unresolved"
    interval = {"clock": "x0_source", "fps": clip.fps, "start_frame": s, "end_frame_exclusive": e + 1,
                "exemplar_frame": ex}
    base = {"hold_id": hold_id, "stem": clip.stem, "recording_id": ids.recording_id(clip.stem),
            "hold_name": record["name"], "family_hold": record["family_hold"], "interval": interval}
    common_ids = [ev_ids["manifest"], ev_ids["audit"], ev_ids["rule"]]
    mat_at_exemplar = sources.mat_contact(rec)[ex]
    out = []

    for z in sorted(label | choice.ground, key=ZI.get):
        zi = ZI[z]
        state, channel = int(clip.state[ex, zi]), sources.CHANNELS[clip.channel[ex, zi]]
        in_ground, labelled = z in choice.ground, z in label
        load, visible = mat_load(rec, s, e, zi)
        role = ground_role(in_ground, state, load)
        action = ("carried" if state == -1 else "kept") if labelled and in_ground else \
            ("added" if in_ground else "removed")
        reasons = {"added": ["missed_touch"], "removed": ["phantom_support"], "carried": ["source_unknown"]}.get(action, [])
        hover = np.median(rec["avatar_min_z"][s:e + 1, zi])
        evidence = {
            "mesh": {"state": int(rec["human_floor_state"][ex, zi]), "floor_cm": _cm(rec["human_floor_z"][ex, zi]),
                     "skin_cm": _cm(rec["human_min_z"][ex, zi]), "v2_state": int(rec["human_ground_state"][ex, zi])},
            "markers": {"state": int(rec["ground_state"][ex, zi]), "marker_cm": _cm(rec["marker_min_z"][ex, zi])},
            "mat": {"load_n": load, "visible_fraction": visible,
                    "contact_at_exemplar": bool(mat_at_exemplar[zi])},
            "avatar": {"cm": _cm(rec["avatar_min_z"][ex, zi]), "hover_cm": _cm(hover),
                       "realised": bool(100 * hover <= CONFIG["hover_cm"])}}
        cited = common_ids + [ev_ids["v1"]] + ([ev_ids["v2"], ev_ids["v3"]] if channel == "mesh" else [])
        ann = {**base, "kind": "ground", "contact": f"{z}:G", "zones": [z], "source_label": labelled,
               "source_state": STATES[state], "source_channel": sources.CHANNEL_NAMES[channel],
               "stable": bool(clip.zone_dist[ex, zi] > near),
               "window_contact_fraction": _window_fraction(clip.state[s:e + 1, zi]),
               "in_configuration": in_ground, "label_action": action,
               "load_path_class": "external_support" if in_ground and state == 1 else None,
               "target_role": role, "status": "unresolved" if hold_unresolved else "resolved",
               "reasons": reasons + (["hold_unresolved"] if hold_unresolved else []),
               "evidence": evidence, "evidence_ids": cited, "review": None, "alternative_group": None}
        if review_claims is not None:
            claim = review_claims["floor"].get(z, {}).get("human")
            frame = review_claims["frame"]
            truth = int(clip.state[frame, zi])
            agrees = None if claim is None or truth == -1 else claim == truth
            ann["review"] = {"verdict": review_claims["verdict_id"], "frame": frame, "human_claim": claim,
                             "agrees_with_source": agrees}
            ann["evidence_ids"] = ann["evidence_ids"] + [review_claims["verdict_id"]]
            if agrees is False and "floor_human" in admitted and frame == ex and z in floor_human_scope():
                ann["status"] = "unresolved"
                ann["reasons"].append("reviewer_conflict")
        out.append(ann)

    ground = set(choice.ground)
    contact_now = {hm.PAIRS[k] for k in np.nonzero(clip.pairs[ex] == 1)[0]}
    pair_anns = []
    for p in sorted(label_pairs | contact_now, key=PAIR_COLUMN.get):
        k = PAIR_COLUMN[p]
        state = int(clip.pairs[ex, k])
        labelled = p in label_pairs
        stable = bool(clip.pair_dist[ex, k] > near)
        lp = load_path_class(p, ground)
        claim = (review_claims or {}).get("roles", {}).get(pair_name(p))
        role = pair_role(state, stable, labelled, lp, hold_unresolved,
                         claim if "roles" in admitted and claim in REVIEW_ROLES else None)
        configured = role == "required_touch" or (state == -1 and labelled)
        if state == -1:
            action = "carried" if labelled else "recorded"
        elif labelled:
            action = "kept" if configured else ("removed" if state == 0 else "demoted")
        else:
            action = "added" if configured else "recorded"
        reasons = {"removed": ["proximity_latch"], "demoted": [f"touch_{role}"], "carried": ["source_unknown"],
                   "added": ["admitted_review_role"]}.get(action, [])
        if state == 1 and not stable:
            reasons.append("unstable_at_exemplar")
        gap = avatar_gaps.get(p)
        evidence = {"mesh": {"state": state, "gap_cm": _cm(rec["human_pair_gap"][ex, k]),
                             "window_gap_cm": _cm(np.median(rec["human_pair_gap"][s:e + 1, k]))},
                    "avatar": {"gap_cm": _num(gap, 2), "realised": None if gap is None else bool(gap <= verdicts.PAIR_TOUCH_CM)}}
        ann = {**base, "kind": "pair", "contact": pair_name(p), "zones": list(p), "source_label": labelled,
               "source_state": STATES[state], "source_channel": sources.CHANNEL_NAMES["mesh" if state != -1 else "none"],
               "stable": stable, "window_contact_fraction": _window_fraction(clip.pairs[s:e + 1, k]),
               "in_configuration": configured, "label_action": action,
               "load_path_class": lp if state == 1 else None, "load_path_if_touching": lp,
               "target_role": role,
               "status": "unresolved" if role == "unresolved" or hold_unresolved else "resolved",
               "reasons": reasons + (["hold_unresolved"] if hold_unresolved else []),
               "evidence": evidence,
               "evidence_ids": common_ids + [ev_ids["v2"], ev_ids["motion"]], "review": None,
               "alternative_group": None}
        if review_claims is not None and pair_name(p) in review_claims["candidates"]:
            claim = review_claims["roles"].get(pair_name(p))
            conflict = None
            if claim == "required_touch" and state == 0:
                conflict = "required_but_apart"
            elif claim is not None and state == 1 and claim != role and role in REVIEW_ROLES:
                conflict = "role_differs"
            ann["review"] = {"verdict": review_claims["verdict_id"], "role": claim, "agrees": None if claim is None
                             else claim == role, "conflict": conflict}
            ann["evidence_ids"] = ann["evidence_ids"] + [review_claims["verdict_id"]]
        pair_anns.append(ann)
    groups = alternative_groups(hold_id, [parse_pair(a["contact"]) for a in pair_anns if a["in_configuration"]])
    for a in pair_anns:
        a["alternative_group"] = groups.get(parse_pair(a["contact"]))
    return out + pair_anns


# --------------------------------------------------------------------------- #
# Reviews (Pass B)
# --------------------------------------------------------------------------- #
CLAIM = {"touching": 1, "off_floor": 0}


def pass_b_reviewer(effort: str = "high") -> review.Reviewer:
    return review.Reviewer(effort=effort, pass_="B")


def parse_review(record: dict, packet: dict) -> dict:
    """The claims of one valid Pass-B verdict, and ``errors`` for the parts the contract rejects: a role
    for a pair that is not a candidate or given twice, a panel the packet lacks."""
    ans, errors = record["answer"], []
    tags = verdicts.panel_tags(packet)
    frame_of = {t: f["frame"] for f in packet["evidence"]["frames"] for t in f["panels"]}
    strip = {t for im in packet["images"] for t, what in im["panels"].items() if what == "time strip"}
    candidates = [c["pair"] for c in packet["candidates"]]
    roles = {}
    for r in ans["roles"]:
        name = pair_name(sorted((verdicts.PARTS[r["part_a"]], verdicts.PARTS[r["part_b"]]), key=ZI.get))
        if name not in candidates:
            errors.append(f"role for {name}, which is not a candidate")
        elif name in roles:
            errors.append(f"role for {name} given twice")
        else:
            roles[name] = None if r["role"] == "cannot_tell" else r["role"]
        errors += [f"panel {t} is not in the packet" for t in r["panels"] if t not in tags]
    better = ans["timing"]["better_panel"]
    if better and better not in strip:
        errors.append(f"better_panel {better} is not a time-strip panel")
    return {"verdict_id": f"ledger:{record['packet_id']}.B.{record['n']}", "packet_id": record["packet_id"],
            "frame": int(packet["capture"]["main_frame"]), "candidates": candidates,
            "roles": {k: v for k, v in roles.items()}, "unanswered": [c for c in candidates if c not in roles],
            "floor": {verdicts.PARTS[p]: {"avatar": CLAIM.get(v["avatar"]), "human": CLAIM.get(v["markers"])}
                      for p, v in ans["floor"].items()},
            "variant": {"matches_label": ans["variant"]["matches_label"], "note": ans["variant"]["note"],
                        "candidates": [c["name"] for c in ans["pose"]["candidates"]]},
            "timing": {"start": ans["timing"]["start"], "end": ans["timing"]["end"],
                       "main_moment": ans["timing"]["main_moment"],
                       "better_frame": frame_of.get(better) if better in strip else None},
            "repair": {"needed": ans["repair"]["needed"], "kinds": list(ans["repair"]["kinds"])},
            "errors": errors}


def load_reviews(ledger_dir: Path = verdicts.LEDGER_DIR, reviewer: review.Reviewer | None = None,
                 render_v: str = "render_v3", current: dict | None = None) -> tuple[dict, list[str]]:
    """``({hold_id: claims}, failures)``: one valid Pass-B verdict per hold under ``reviewer``'s key, from
    its current packet (``current``: hold id -> packet id) if it has one, else the most recent."""
    reviewer = pass_b_reviewer() if reviewer is None else reviewer
    mine = [r for r in verdicts.read_ledger(ledger_dir) if r["pass"] == "B" and r["render_v"] == render_v
            and r["reviewer"]["key"] == reviewer.key and r["status"] == "valid"]
    by_hold = collections.defaultdict(list)
    for r in mine:
        by_hold[r["hold_id"]].append(r)
    out, failures = {}, []
    for hold_id, rs in sorted(by_hold.items()):
        own = [r for r in rs if r["packet_id"] == (current or {}).get(hold_id)]
        r = min(own, key=lambda r: r["n"]) if own else max(rs, key=lambda r: (r["started_utc"], r["n"]))
        path = Path(r["packet"]["dir"]) / "packet.json"
        try:
            packet = json.loads(path.read_text())
            claims = parse_review(r, packet)
        except Exception as exc:  # noqa: BLE001 -- a verdict whose packet is gone cannot be read
            failures.append(f"{hold_id}: {type(exc).__name__}: {exc}")
            continue
        claims["path"] = str(verdicts.verdict_path(ledger_dir, r["packet_id"], "B", r["n"]))
        claims["reviewer"] = {"key": r["reviewer"]["key"], "model": r["reviewer"]["model"],
                              "effort": r["reviewer"]["effort"]}
        out[hold_id] = claims
    return out, failures


def admitted_classes(reviewer: review.Reviewer | None = None, render_v: str = "render_v3",
                     calibration_dir: Path = PASS_B_CALIBRATION) -> tuple[tuple, Path | None]:
    """The Pass-B claim classes the calibration table admits as evidence for ``reviewer``."""
    reviewer = pass_b_reviewer() if reviewer is None else reviewer
    path = Path(calibration_dir) / f"{render_v}.json"
    if not path.exists():
        return (), None
    table = json.loads(path.read_text())
    rv = next((x for x in table["reviewers"] if x["key"] == reviewer.key), None)
    return (tuple(rv["evidence"]) if rv else ()), path


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
def _hold_review(claims: dict | None, choice: SimpleNamespace, fps: int) -> dict | None:
    if claims is None:
        return None
    t = claims["timing"]
    better = t["better_frame"]
    return {"verdict": claims["verdict_id"], "variant": claims["variant"], "repair": claims["repair"],
            "timing": {**t, "better_frame_minus_exemplar_s": None if better is None
                       else _num((better - choice.frame_hold) / fps, 3)},
            "unanswered_roles": claims["unanswered"], "contract_errors": claims["errors"]}


def reconcile(clip: Clip, entry: dict, index: int, record: dict, ev_ids: dict, claims: dict | None,
              admitted: tuple = ()) -> tuple[dict, list[dict]]:
    """``(hold for holds.yaml, annotations)`` of one source hold."""
    hold = entry["holds"][index]
    w = record["window"]
    label = set(record["label_ground"])
    choice = choose(clip, w["frame_start"], w["frame_end"], w["frame_hold"], label)
    gaps = {p: g for p, g in verdicts.pair_gaps_cm(clip.stem, choice.frame_hold).items()}
    anns = annotate(clip, record, hold, choice, gaps, ev_ids, claims, admitted)
    fps = clip.fps
    out = dict(hold)
    out.update(t_start=_num(choice.frame_start / fps, 4), t_end=_num(choice.frame_end / fps, 4),
               t_hold=_num(choice.frame_hold / fps, 4), duration_s=_num((choice.frame_end - choice.frame_start + 1) / fps, 3),
               frame_start=choice.frame_start, frame_end=choice.frame_end, frame_hold=choice.frame_hold)
    if choice.frame_hold != w["frame_hold"]:
        out.update(exemplar_fields(clip, choice.frame_hold))
    ground_pairs = [a["contact"] for a in anns if a["kind"] == "ground" and a["in_configuration"]]
    body_pairs = [a["contact"] for a in anns if a["kind"] == "pair" and a["in_configuration"]]
    out["pairs"] = ground_pairs + body_pairs
    out["pairs_ground"] = ground_pairs
    changes = []
    if choice.status != "kept":
        changes.append(choice.status)
    if choice.frame_start > w["frame_start"]:
        changes.append("start_trimmed")
    if choice.frame_end < w["frame_end"]:
        changes.append("end_trimmed")
    for a in anns:  # every contact that enters or leaves the configuration
        if a["label_action"] in ("added", "removed", "demoted"):
            changes.append(f"{a['label_action']}:{a['contact']}")
    out["hold_id"] = record["hold_id"]
    out["labels"] = {
        "status": choice.status, "changes": changes,
        "source": {k: hold[k] for k in ("frame_start", "frame_end", "frame_hold", "pairs")},
        "unverified": [a["contact"] for a in anns if a["label_action"] == "carried"],
        "roles": {a["contact"]: a["target_role"] for a in anns},
        "alternative_groups": sorted({a["alternative_group"] for a in anns if a["alternative_group"]}),
        "review": _hold_review(claims, choice, fps)}
    return out, anns


def _store_ids(stem: str, rec: capture.Capture, stores: Stores) -> list[dict]:
    rows = []
    for v, d in (("v1", stores.v1), ("v2", stores.v2), ("v3", stores.v3)):
        path = Path(d) / f"{stem}.json"
        meta = json.loads(path.read_text())
        rows.append({"id": f"capture:{v}:{stem}", "kind": "capture_store", "store": f"capture/{v}",
                     "path": ids.display_path(path), "sha256": ids.sha256_file(path),
                     "identity": {"generator": meta["generator"]["sha256"], "calibration": meta["calibration"]["id"],
                                  **({"base": meta["base"]} if "base" in meta else {"inputs": meta["inputs"]})}})
    motion = ids.motion_path(stem)
    rows.append({"id": f"motion:{stem}", "kind": "shipped_motion", "path": ids.display_path(motion),
                 "sha256": ids.sha256_file(motion)})
    return rows


def build(aud: SimpleNamespace, stores: Stores = Stores(), stems: list[str] | None = None,
          reviews: dict | None = None, admitted: tuple = (), extra_evidence: list[dict] = ()) -> dict:
    """Reconcile every hold of the audit (of ``stems``, if given). Returns ``holds`` (the manifest),
    ``annotations``, ``evidence``, ``context`` and ``failures``."""
    manifest = ids.load_manifest(aud.manifest)
    reviews = reviews or {}
    ev = {f"manifest:{aud.manifest.stem}": {"id": f"manifest:{aud.manifest.stem}", "kind": "source_manifest",
                                             "path": ids.display_path(aud.manifest), "sha256": ids.sha256_file(aud.manifest)},
          f"audit:{aud.audit_id}": {"id": f"audit:{aud.audit_id}", "kind": "audit",
                                     "path": ids.display_path(aud.dir / "holds.jsonl"),
                                     "sha256": ids.sha256_file(aud.dir / "holds.jsonl")},
          f"rule:{MODULE}": {"id": f"rule:{MODULE}", "kind": "rule", "path": ids.display_path(__file__),
                             "sha256": ids.sha256_file(__file__),
                             "rules": ["timing", "tier1_load_path", "roles", "configuration"]}}
    for row in extra_evidence:
        ev[row["id"]] = row
    holds_out, annotations, failures, clips_out, identities = {}, [], [], [], {}
    by_stem = collections.defaultdict(list)
    for r in aud.records.values():
        by_stem[r["stem"]].append(r)
    for entry in manifest["clips"]:
        stem = entry["stem"]
        if stems is not None and stem not in stems:
            continue
        try:
            clip = clip_evidence(stem, manifest, stores)
            for row in _store_ids(stem, clip.rec, stores):
                ev[row["id"]] = row
            identities[stem] = sources.identity(clip.rec)
            ev_ids = {"manifest": f"manifest:{aud.manifest.stem}", "audit": f"audit:{aud.audit_id}",
                      "rule": f"rule:{MODULE}", "v1": f"capture:v1:{stem}", "v2": f"capture:v2:{stem}",
                      "v3": f"capture:v3:{stem}", "motion": f"motion:{stem}"}
            records = sorted(by_stem[stem], key=lambda r: r["hold_index"])
            if [r["hold_index"] for r in records] != list(range(len(entry["holds"]))):
                raise ValueError("the audit's holds do not match the manifest's")
            new_holds = []
            for r in records:
                claims = reviews.get(r["hold_id"])
                if claims is not None:
                    ev[claims["verdict_id"]] = {"id": claims["verdict_id"], "kind": "verdict",
                                                "path": ids.display_path(claims["path"]),
                                                "sha256": ids.sha256_file(claims["path"]), **claims["reviewer"],
                                                "packet_id": claims["packet_id"]}
                hold, anns = reconcile(clip, entry, r["hold_index"], r, ev_ids, claims, admitted)
                new_holds.append(hold)
                annotations += anns
                holds_out[r["hold_id"]] = hold
        except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
            failures.append(f"{stem}: {type(exc).__name__}: {exc}")
            continue
        clips_out.append({**entry, "recording_id": ids.recording_id(stem),
                          "capture_available": bool(clip.rec.meta["human_available"]), "holds": new_holds})
    return {"clips": clips_out, "holds": holds_out, "annotations": annotations, "evidence": ev,
            "identities": identities, "failures": failures, "manifest": manifest, "aud": aud,
            "reviews": reviews, "admitted": admitted}


# --------------------------------------------------------------------------- #
# Checks (the tests' invariants, also run by the CLI)
# --------------------------------------------------------------------------- #
def check(result: dict, stores: Stores = Stores()) -> list[str]:
    """Every violation of the labels' contract: contradictory roles, uncited or unknown evidence,
    a ground set that disagrees with the capture, duplicate ids."""
    problems = []
    seen, ev = set(), result["evidence"]
    for a in result["annotations"]:
        key = (a["hold_id"], a["contact"])
        if key in seen:
            problems.append(f"{key}: annotated twice")
        seen.add(key)
        role, state = a["target_role"], a["source_state"]
        if role not in ROLES or (a["load_path_class"] or "external_support") not in LOAD_PATH_CLASSES:
            problems.append(f"{key}: unknown role or class")
        unstable_pair = a["kind"] == "pair" and not a["stable"]
        if role in ("required_support", "required_touch") and (state != "observed_contact" or unstable_pair):
            problems.append(f"{key}: {role} without a stable observed contact")
        if role == "required_support" and a["load_path_class"] != "external_support":
            problems.append(f"{key}: required_support on a {a['load_path_class']}")
        if role == "forbidden_support" and (a["in_configuration"] or state != "observed_separation"):
            problems.append(f"{key}: forbidden_support on a configured or unseparated zone")
        carried = a["label_action"] == "carried"
        if carried != (state == "unknown" and a["source_label"]) or (carried and not a["in_configuration"]):
            problems.append(f"{key}: an unknown label must be carried in the configuration")
        if a["kind"] == "ground" and a["in_configuration"] != (state == "observed_contact" or carried):
            problems.append(f"{key}: the ground set must be the observed contacts and the carried labels")
        if a["kind"] == "pair" and a["in_configuration"] != (role == "required_touch" or carried):
            problems.append(f"{key}: a configured pair must be required_touch or carried")
        if role == "unresolved" and a["status"] != "unresolved":
            problems.append(f"{key}: role unresolved with status {a['status']}")
        if not a["evidence_ids"]:
            problems.append(f"{key}: no evidence ids")
        problems += [f"{key}: evidence id {i} is not in the index" for i in a["evidence_ids"] if i not in ev]
    by_hold = collections.defaultdict(list)
    for a in result["annotations"]:
        by_hold[a["hold_id"]].append(a)
    for stem in {h["hold_id"].rsplit("@", 1)[0] for h in result["holds"].values()}:
        state, _ = sources.ground_source(load_record(stem, stores))
        for hid, h in result["holds"].items():
            if not hid.startswith(stem + "@"):
                continue
            ground = {p[:-2] for p in h["pairs_ground"]}
            s, e, ex = h["frame_start"], h["frame_end"], h["frame_hold"]
            bad = [z for z in ZONE_ORDER if state[ex, ZI[z]] >= 0 and (state[ex, ZI[z]] == 1) != (z in ground)]
            if bad:
                problems.append(f"{hid}: ground set disagrees with the capture at the exemplar on {bad}")
            if not matching(state[s:e + 1], ground).all():
                problems.append(f"{hid}: ground set disagrees with the capture inside the window")
            if not s <= ex <= e:
                problems.append(f"{hid}: exemplar outside the window")
            configured = {a["contact"] for a in by_hold[hid] if a["in_configuration"]}
            if set(h["pairs"]) != configured:
                problems.append(f"{hid}: pairs differ from the configured annotations")
    counts = collections.Counter(h["hold_id"] for c in result["clips"] for h in c["holds"])
    problems += [f"hold id {h} used {n} times" for h, n in counts.items() if n > 1]
    return problems


# --------------------------------------------------------------------------- #
# Metrics and output
# --------------------------------------------------------------------------- #
def metrics(result: dict) -> dict:
    anns = result["annotations"]
    holds = [h for c in result["clips"] for h in c["holds"]]

    def block(hs):
        hid = {h["hold_id"] for h in hs}
        mine = [a for a in anns if a["hold_id"] in hid]
        g = [a for a in mine if a["kind"] == "ground"]
        p = [a for a in mine if a["kind"] == "pair"]
        configured_support = [a for a in g if a["in_configuration"] and a["source_state"] == "observed_contact"]
        hover = [a["evidence"]["avatar"]["hover_cm"] for a in configured_support]
        return {
            "holds": len(hs), "status": dict(collections.Counter(h["labels"]["status"] for h in hs)),
            "exemplar_moved": sum(h["frame_hold"] != h["labels"]["source"]["frame_hold"] for h in hs),
            "window_trimmed": sum(("start_trimmed" in h["labels"]["changes"]) or ("end_trimmed" in h["labels"]["changes"])
                                  for h in hs),
            "ground_actions": dict(collections.Counter(a["label_action"] for a in g)),
            "ground_roles": dict(collections.Counter(a["target_role"] for a in g)),
            "pair_actions": dict(collections.Counter(a["label_action"] for a in p)),
            "pair_roles": dict(collections.Counter(a["target_role"] for a in p)),
            "pair_load_paths_of_contacts": dict(collections.Counter(a["load_path_class"] for a in p
                                                                     if a["load_path_class"])),
            "labelled_pairs": sum(a["source_label"] for a in p),
            "labelled_pairs_touching": sum(a["source_label"] and a["source_state"] == "observed_contact" for a in p),
            "labelled_leg_leg_braces": sum(a["source_label"] and a["load_path_if_touching"] == "internal_brace"
                                           and all(z[2:] in ("FOOT", "SHANK", "THIGH") for z in a["zones"]) for a in p),
            "alternative_groups": len({a["alternative_group"] for a in p if a["alternative_group"]}),
            "supports_configured": len(configured_support),
            "supports_hover_gt_2cm": sum(x is not None and x > 2 for x in hover),
            "supports_hover_gt_5cm": sum(x is not None and x > 5 for x in hover),
            "unresolved_annotations": sum(a["status"] == "unresolved" for a in mine),
        }

    reviewed = [h for h in holds if h["labels"]["review"]]
    rev_pairs = [a for a in anns if a["kind"] == "pair" and a["review"]]
    rev_ground = [a for a in anns if a["kind"] == "ground" and a["review"] and a["review"]["agrees_with_source"] is not None]
    review_block = {
        "holds_reviewed": len(reviewed), "admitted_classes": list(result["admitted"]),
        "role_claims": sum(a["review"]["role"] is not None for a in rev_pairs),
        "role_agrees": sum(bool(a["review"]["agrees"]) for a in rev_pairs),
        "role_conflicts": dict(collections.Counter(a["review"]["conflict"] for a in rev_pairs if a["review"]["conflict"])),
        "floor_human_claims": len(rev_ground),
        "floor_human_agrees": sum(a["review"]["agrees_with_source"] for a in rev_ground),
        "variant": dict(collections.Counter(h["labels"]["review"]["variant"]["matches_label"] for h in reviewed)),
        "repair_needed": dict(collections.Counter(h["labels"]["review"]["repair"]["needed"] for h in reviewed)),
        "main_moment": dict(collections.Counter(h["labels"]["review"]["timing"]["main_moment"] for h in reviewed)),
    }
    return {"all": block(holds), "family": block([h for h in holds if h.get("extend")]), "review": review_block}


CROW_FAMILY = ("220923_Crane_Crow_Pose_or_Bakasana_-a@439", "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c@606",
               "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c@1296")


def labels_id(result: dict) -> str:
    aud = result["aud"]
    key = {"schema": SCHEMA_VERSION, "generator": ids.sha256_file(__file__), "config": CONFIG,
           "manifest": ids.sha256_file(aud.manifest), "audit": aud.audit_id, "clips": result["identities"],
           "reviews": sorted((c["verdict_id"], ids.sha256_file(c["path"])) for c in result["reviews"].values()),
           "admitted": list(result["admitted"])}
    return f"{aud.manifest.stem}.labels_{LABELS_VERSION}.{ids.sha256_json(key)[:10]}"


def summary_markdown(result: dict, lid: str, m: dict) -> str:
    a, f = m["all"], m["family"]
    rows = [("Holds", a["holds"], f["holds"])]
    rows += [(f"Status `{s}`", a["status"].get(s, 0), f["status"].get(s, 0)) for s in HOLD_STATUS]
    rows += [("Exemplar moved", a["exemplar_moved"], f["exemplar_moved"]),
             ("Window trimmed", a["window_trimmed"], f["window_trimmed"])]
    rows += [(f"Ground zones `{k}`", a["ground_actions"].get(k, 0), f["ground_actions"].get(k, 0))
             for k in ("kept", "added", "removed", "carried")]
    rows += [(f"Pairs `{k}`", a["pair_actions"].get(k, 0), f["pair_actions"].get(k, 0))
             for k in ("kept", "demoted", "removed", "carried", "added", "recorded")]
    rows += [(f"Pair role `{r}`", a["pair_roles"].get(r, 0), f["pair_roles"].get(r, 0))
             for r in ("required_touch", "allowed", "incidental", "unresolved", "unspecified")]
    rows += [("Configured supports floating > 2 cm (Step 8's worklist)",
              f"{a['supports_hover_gt_2cm']} / {a['supports_configured']}",
              f"{f['supports_hover_gt_2cm']} / {f['supports_configured']}")]
    lines = [f"# Labels `{lid}`", "", "Generated by `reference_curation.labels` (BUILD_PLAN Step 6); the rules are in its "
             "docstring. `hold_id` names the source hold; `frame_hold` is the corrected exemplar.", "",
             "| Metric | All holds | Family holds |", "|---|---|---|"]
    lines += [f"| {x} | {y} | {z} |" for x, y, z in rows]
    rv = m["review"]
    lines += ["", f"Pass-B review: {rv['holds_reviewed']} holds; admitted classes {rv['admitted_classes'] or 'none'}; "
              f"role claims {rv['role_claims']} (agree with the provisional role {rv['role_agrees']}); conflicts "
              f"{rv['role_conflicts'] or 'none'}; floor-human claims agreeing with the source {rv['floor_human_agrees']} / "
              f"{rv['floor_human_claims']}.", "", "## Changed holds", "",
              "| Hold | Name | Status | Source window / exemplar (s) | Corrected | Changes |", "|---|---|---|---|---|---|"]
    for c in result["clips"]:
        fps = c["fps"]
        for h in c["holds"]:
            lab = h["labels"]
            if not lab["changes"]:
                continue
            s = lab["source"]
            lines.append(f"| `{h['hold_id']}` | {h['name']} | {lab['status']} | {s['frame_start'] / fps:.2f}-"
                         f"{s['frame_end'] / fps:.2f} / {s['frame_hold'] / fps:.2f} | {h['t_start']:.2f}-{h['t_end']:.2f} / "
                         f"{h['t_hold']:.2f} | {', '.join(x for x in lab['changes'] if x not in HOLD_STATUS)} |")
    return "\n".join(lines + review_worklist(result)) + "\n"


def review_worklist(result: dict) -> list[str]:
    """What a person should look at: where the reviewer's advisory claims and the labels differ."""
    anns = [a for a in result["annotations"] if a["review"]]
    roles = [a for a in anns if a["kind"] == "pair" and a["review"]["role"] and not a["review"]["agrees"]
             and (a["review"]["role"] == "required_touch" or a["target_role"] == "required_touch")]
    floor = [a for a in anns if a["kind"] == "ground" and a["review"]["agrees_with_source"] is False]
    variants = [h for h in result["holds"].values() if h["labels"]["review"]
                and h["labels"]["review"]["variant"]["matches_label"] in ("no", "variant")]
    if not (roles or floor or variants):
        return []
    out = ["", "## Review worklist (advisory claims that differ from the labels)", "",
           f"Roles where the reviewer or the labels say `required_touch` and the other does not ({len(roles)}):", "",
           "| Hold | Pair | Labels | Reviewer | Human skin | Load path |", "|---|---|---|---|---|---|"]
    out += [f"| `{a['hold_id']}` | {a['contact']} | {a['target_role']} ({a['label_action']}) | {a['review']['role']} | "
            f"{a['source_state']} | {a['load_path_if_touching']} |" for a in roles]
    out += ["", f"Variants the reviewer calls different from the label ({len(variants)}):", "",
            "| Hold | Label | Reviewer | Note |", "|---|---|---|---|"]
    out += [f"| `{h['hold_id']}` | {h['name']} | {h['labels']['review']['variant']['matches_label']} | "
            f"{h['labels']['review']['variant']['note']} |" for h in variants]
    out += ["", f"Floor contacts where the reviewer contradicts the capture at the packet's moment ({len(floor)}):", "",
            "| Hold | Zone | Capture | Reviewer (human) |", "|---|---|---|---|"]
    out += [f"| `{a['hold_id']}` | {a['zones'][0]} | {a['source_state']} | {a['review']['human_claim']} |" for a in floor]
    return out


def write(result: dict, out_root: Path = LABELS_DIR) -> Path:
    lid = labels_id(result)
    out = Path(out_root) / lid
    out.mkdir(parents=True, exist_ok=True)
    head = {"schema_version": SCHEMA_VERSION, "labels_id": lid}
    (out / "annotations.jsonl").write_text("".join(json.dumps({**head, **a}) + "\n" for a in result["annotations"]))
    (out / "evidence.jsonl").write_text("".join(json.dumps(result["evidence"][k]) + "\n" for k in sorted(result["evidence"])))
    manifest = {k: v for k, v in result["manifest"].items() if k != "clips"}
    manifest["labels"] = {"labels_id": lid, "schema_version": SCHEMA_VERSION, "module": MODULE,
                          "git_rev": ids.git_rev(), "source_manifest": ids.display_path(result["aud"].manifest),
                          "audit_id": result["aud"].audit_id, "annotations": "annotations.jsonl",
                          "hold_id": "names the source hold (its source exemplar); frame_hold is the corrected exemplar"}
    manifest["clips"] = result["clips"]
    (out / "holds.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))
    m = metrics(result)
    inputs = [result["aud"].manifest, result["aud"].dir / "holds.jsonl", capture.CALIBRATION_PATH, hm.CALIBRATION_PATH]
    inputs += [Path(c["path"]) for c in result["reviews"].values()]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "labels_id": lid,
              "audit_id": result["aud"].audit_id, "config": CONFIG, "metrics": m,
              "crow_family": {h: {k: result["holds"][h][k] for k in ("frame_start", "frame_end", "frame_hold", "t_hold")}
                              | {"source_frame_hold": result["holds"][h]["labels"]["source"]["frame_hold"]}
                              for h in CROW_FAMILY if h in result["holds"]},
              "stores": {"capture/v3": ids.sha256_file(sources.__file__)}}
    (out / "labels.json").write_text(json.dumps(record, indent=1) + "\n")
    (out / "summary.md").write_text(summary_markdown(result, lid, m))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit", type=Path, help="an audit directory (default: the newest calibrated audit)")
    ap.add_argument("--stem", nargs="*", help="only these clips")
    ap.add_argument("--no-review", action="store_true", help="leave the Pass-B verdicts out")
    ap.add_argument("--ledger-dir", type=Path, default=verdicts.LEDGER_DIR)
    ap.add_argument("--out-root", type=Path, default=LABELS_DIR)
    args = ap.parse_args(argv)

    start = time.time()
    try:
        aud = packets.load_audit(args.audit or packets.default_audit_dir())
        reviews, rfail, admitted, extra = {}, [], (), []
        if not args.no_review:
            from reference_curation import informed  # its packets index names each hold's current packet
            reviews, rfail = load_reviews(args.ledger_dir, current=informed.current_packets())
            admitted, cal_path = admitted_classes()
            if cal_path is not None:
                extra.append({"id": "calibration:pass_b/render_v3", "kind": "calibration",
                              "path": ids.display_path(cal_path), "sha256": ids.sha256_file(cal_path)})
        result = build(aud, stems=args.stem or None, reviews=reviews, admitted=admitted, extra_evidence=extra)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    failures = result["failures"] + rfail + check(result)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures:
        print(f"labels: {len(failures)} failures; nothing written", file=sys.stderr)
        return 1
    out = write(result, args.out_root)
    a = metrics(result)["all"]
    print(f"labels {out.name}: {a['holds']} holds ({', '.join(f'{k} {v}' for k, v in a['status'].items())}); ground "
          f"{a['ground_actions']}; pairs {a['pair_actions']}; roles {a['pair_roles']}; {len(reviews)} reviewed, admitted "
          f"{list(admitted) or 'none'}; {len(result['annotations'])} annotations in {time.time() - start:.0f} s "
          f"-> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
