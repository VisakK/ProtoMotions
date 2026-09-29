# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus audit and review queue (BUILD_PLAN Step 2): every hold of a manifest, checked against the
capture evidence store (Step 1).

It produces the review worklist, the first fix list, and the progress metrics that later steps
drive to zero. ``audit(manifest, rule)`` returns one record per hold. ``write()`` puts them in
``data/reference_curation/audits/<audit_id>/``:

* ``holds.jsonl``: one record per hold, flagged or not;
* ``summary.md``: the progress metrics, then every flagged hold;
* ``queue.jsonl``: review items, each with hold, window, pass, questions and priority;
* ``audit.json``: provenance, configuration and metrics.

``audit_id`` is ``<manifest stem>.<rule>.<hash>``, where the hash covers the manifest, this module,
the rule and every store record's own identity. A rerun on the same inputs rewrites the same folder.

A hold record
-------------
``label_ground`` is the manifest's ground set, taken from its ``:G`` pairs. The capture reads
only the zones its rule can decide (``capture.decided``). Every other zone is -1, *undecided*, and
never counts as "absent".

* ``capture.ground_exemplar``: the human's support at ``frame_hold``.
* ``capture.ground_window``: the zones in contact on most of the window's decided frames.
* ``capture.first`` / ``last``: the capture-derived boundaries. They are the first and last window
  frames whose support set equals the label's, where an undecided zone agrees with anything. Each
  one extends past the window while the match continues. ``start_lag_s`` > 0 means the label starts
  before the human reaches its support; ``end_lead_s`` > 0 means it ends after the human leaves it.
* ``capture.exemplar_stable``: no support change within ±0.5 s of ``frame_hold``. A zone's
  unknown frames keep its last known state, as the Schmitt trigger does.
* ``zones``: one row per labelled or decided zone:
  - the exemplar state and the marker height the rule read;
  - the window's contact fraction;
  - the avatar's lowest collision surface at the exemplar (``avatar_cm``), and for labelled zones
    its window median (``hover_cm``, README §3.1);
  - ``load_n``: the attributed mat load over coverage-valid frames where the zone is
    attribution-visible. It is ``None`` when fewer than half the window's frames qualify: the mat
    cannot speak for a zone more than 6 cm up.
* ``mat``: window medians over coverage-valid frames (validity column 0) of the total and the
  unexplained load. Never gate the unexplained load on column 1: that column is low exactly when
  the unexplained share is high.

Rules
-----
``calibrated`` (default)
    The store's ``ground_state``: Step 1's per-zone Schmitt trigger on the feet, hands and head.
    The exemplar state is the state at ``frame_hold``.
``pooled``
    The review's exemplar audit (README §3.2), kept to reproduce it. It reads the feet and hands
    through ``capture.REVIEW_ZONE_MARKERS``, with touch below 3.5 cm and lifted above 6 cm. The band
    between is undecided, and the exemplar reads the lowest marker over ±3 frames. Pushing these
    thresholds through the Schmitt trigger instead gives 17 in 12 and 5 in 3, not 15 in 11 and 4 in 3.

Reasons, fixes and severity
---------------------------
=====================  ==========  ===================================================================
code                   fix         meaning
=====================  ==========  ===================================================================
missed_touch           labels      the human touches a decided zone at the exemplar; the label omits it
phantom_support        labels      the label has a decided zone that the human has lifted at the exemplar
no_capture_match       labels      the label's support set never occurs in the window
boundary_start / _end  boundaries  the label's window runs > 0.25 s outside the capture's matching run
exemplar_unstable      boundaries  the support changes within ±0.5 s of the exemplar
hover                  retarget    a labelled support floats > 2 cm (window median), and the human is
                                   not seen lifting it
unexplained_load       evidence    > 100 N of mat load goes to no body (window median)
capture_unavailable    evidence    no readable MoSh fit, so nothing can be checked against the human
=====================  ==========  ===================================================================

Severity is the worst over a hold's reasons:
- **high:** a ``labels`` reason on a family hold (``extend``). Hold extension freezes that exemplar
  for 3 s and 7 s.
- **medium:** a ``labels`` reason elsewhere; a ``boundaries`` reason on a family hold; or a family
  hold floating > 5 cm.
- **low:** everything else.

Queue
-----
- **Pass B.** Every hold whose reasons ask a question, ordered by severity; then controls, a
  hash-chosen 15 % of the holds that ask none. A hover the markers confirm as touch asks nothing,
  because the retarget (Step 8) realises it. A hover on a zone the markers cannot decide asks for
  its source state.
- **Pass A.** The calibration candidates of Step 4: holds whose human truth at the exemplar is
  unambiguous (every decided zone decisive outside its hysteresis band, stable for ±0.5 s). They are
  interleaved over ``STRATA``: arm balances, inversions, floats of 5–15 cm, floats of 2–5 cm, then
  the rest, because the VLM spike failed on folded poses and floats (BUILD_PLAN §5). Step 4 takes the top ~40, and
  the whole pool is there if a claim class needs more items. Pass-A questions are the fixed blind
  set. Nothing in the item may reach a packet except the frames and zones.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.audit [--rule pooled]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from extract_contact_configs import ZONE_ORDER
from reference_curation import capture, ids

MODULE = "reference_curation.audit"
SCHEMA_VERSION = 1
AUDITS_DIR = ids.DATA_ROOT / "audits"

HOVER_M = 0.02            # README §3.1
HOVER_SEVERE_M = 0.05
HOVER_REPORT_CM = (1, 2, 3, 5)
FLOAT_MAX_M = 0.15        # Step 4's float stratum: 2-15 cm
STABLE_S = 0.5
BOUNDARY_TOL_S = 0.25
UNEXPLAINED_N = 100.0     # corpus hold medians are bimodal: 75 % at <= 2 N, 35 holds above 100 N
COVERAGE_GATE = 0.9       # validity column 0
MIN_VALID = 0.5           # a window median over fewer qualifying frames is not reported
CONTROL_FRACTION = 0.15   # Step 6: a random 15 % of clean holds as Pass-B controls
CONFIG = {"hover_m": HOVER_M, "hover_severe_m": HOVER_SEVERE_M, "float_max_m": FLOAT_MAX_M, "stable_s": STABLE_S,
          "boundary_tol_s": BOUNDARY_TOL_S, "unexplained_n": UNEXPLAINED_N, "coverage_gate_col0": COVERAGE_GATE,
          "min_valid": MIN_VALID, "control_fraction": CONTROL_FRACTION}
POOLED = {"zones": ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND"), "touch_lt_m": 0.035, "lifted_gt_m": 0.06,
          "exemplar_half_width": 3, "markers": "capture.REVIEW_ZONE_MARKERS"}
RULES = ("calibrated", "pooled")
REASONS = {"missed_touch": "labels", "phantom_support": "labels", "no_capture_match": "labels",
           "boundary_start": "boundaries", "boundary_end": "boundaries", "exemplar_unstable": "boundaries",
           "hover": "retarget", "unexplained_load": "evidence", "capture_unavailable": "evidence"}
SEVERITY = ("none", "low", "medium", "high")
PASS_A_QUESTIONS = ("pose", "floor_contacts", "discrepancies", "body_body", "implausible")  # Step 4
STRATA = ("arm_balance", "inversion", "float_5_15", "float_2_5", "other")
ZI = {z: i for i, z in enumerate(ZONE_ORDER)}


def _num(x, nd: int):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _cm(x):  # 10 um: two labelled supports sit at 1.002 / 1.004 cm, and the metrics count from records
    return _num(100.0 * float(x), 3)


def _zones(zs) -> list[str]:
    return sorted(set(zs), key=ZI.__getitem__)


def _hash_order(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Rules: how the human's touch is read
# --------------------------------------------------------------------------- #
def make_rule(name: str, calibration: dict) -> SimpleNamespace:
    """``zones`` the rule decides, ``bands`` {zone: (touch, separation)} and a ``config`` to hash."""
    if name == "calibrated":
        zones = tuple(z for z in ZONE_ORDER if calibration["zones"][z]["admitted"])
        bands = {z: (calibration["zones"][z]["touch_m"], calibration["zones"][z]["separation_m"]) for z in zones}
        return SimpleNamespace(name=name, zones=zones, bands=bands,
                               config={"calibration_id": calibration["id"], "bands_m": bands})
    if name == "pooled":
        band = (POOLED["touch_lt_m"], POOLED["lifted_gt_m"])
        return SimpleNamespace(name=name, zones=POOLED["zones"], bands={z: band for z in POOLED["zones"]},
                               config=dict(POOLED))
    raise ValueError(f"unknown rule {name!r}; expected one of {RULES}")


def clip_evidence(stem: str, rec: capture.Capture, rule) -> SimpleNamespace:
    """Per-frame ``states`` [T,Z] int8, the ``heights`` read, and ``exemplar(f)`` -> (state [Z], height [Z])."""
    if rule.name == "calibrated" or not rec.meta["capture_available"]:
        states, heights = rec["ground_state"], rec["marker_min_z"]
        return SimpleNamespace(states=states, heights=heights, exemplar=lambda f: (states[f], heights[f]))
    fit, _, _ = capture.load_mosh(stem)
    z, resid, _ = capture.marker_arrays(fit["obs"], fit["sim"], fit["labels"], capture.REVIEW_ZONE_MARKERS)
    heights = np.where(capture.zone_known(z, resid), z, np.nan)
    decided = np.array([zone in rule.zones for zone in ZONE_ORDER])

    def band(h):
        with np.errstate(invalid="ignore"):
            s = np.where(h < POOLED["touch_lt_m"], 1, np.where(h > POOLED["lifted_gt_m"], 0, -1))
        return np.where(decided & np.isfinite(h), s, -1).astype(np.int8)

    def exemplar(f):
        hw = POOLED["exemplar_half_width"]
        with warnings.catch_warnings():  # an all-NaN column stays NaN: undecided
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.nanmin(heights[max(0, f - hw): f + hw + 1], 0)
        return band(low), low

    return SimpleNamespace(states=band(heights), heights=heights, exemplar=exemplar)


# --------------------------------------------------------------------------- #
# Pure measurements
# --------------------------------------------------------------------------- #
def capture_bounds(match: np.ndarray, f0: int, f1: int) -> tuple[int | None, int | None]:
    """First and last matching frame of ``[f0, f1]``, each extended outward while ``match`` holds."""
    inside = np.nonzero(match[f0:f1 + 1])[0]
    if not len(inside):
        return None, None
    first, last = f0 + int(inside[0]), f0 + int(inside[-1])
    if first == f0:
        miss = np.nonzero(~match[:f0])[0]
        first = int(miss[-1]) + 1 if len(miss) else 0
    if last == f1:
        miss = np.nonzero(~match[f1 + 1:])[0]
        last = f1 + int(miss[0]) if len(miss) else len(match) - 1
    return first, last


def support_changes(states: np.ndarray) -> np.ndarray:
    """Frames where some zone flips contact <-> separated; unknown frames keep the last known state."""
    T = len(states)
    last = np.maximum.accumulate(np.where(states >= 0, np.arange(T)[:, None], -1), axis=0)
    filled = np.where(last >= 0, np.take_along_axis(states, np.maximum(last, 0), axis=0), -1)
    flip = (filled[1:] != filled[:-1]) & (filled[:-1] >= 0)
    return np.nonzero(flip.any(-1))[0] + 1


def severity(reasons: list[dict], family_hold: bool) -> str:
    level = 0
    for r in reasons:
        if r["fix"] == "labels":
            lv = 3 if family_hold else 2
        elif r["fix"] == "boundaries" or (r["code"] == "hover" and r["max_cm"] > 100 * HOVER_SEVERE_M):
            lv = 2 if family_hold else 1
        else:
            lv = 1
        level = max(level, lv)
    return SEVERITY[level]


# --------------------------------------------------------------------------- #
# One hold
# --------------------------------------------------------------------------- #
def audit_hold(clip: dict, index: int, hold: dict, rec: capture.Capture, ev, rule) -> dict:
    T, fps = rec.meta["num_frames"], rec.fps
    f0, f1, fh = (min(int(hold[k]), T - 1) for k in ("frame_start", "frame_end", "frame_hold"))
    win = slice(f0, f1 + 1)
    label = [z for z in ZONE_ORDER if f"{z}:G" in hold["pairs"]]
    decided = list(rule.zones) if rec.meta["capture_available"] else []
    ex_state, ex_height = ev.exemplar(fh)
    s = ev.states[:, [ZI[z] for z in decided]]
    known = s[win] >= 0
    with np.errstate(invalid="ignore"):
        contact = (s[win] == 1).sum(0) / known.sum(0)  # NaN where the window has no decided frame
    lifted = {z for z, c in zip(decided, contact) if c < 0.5}

    cap = {"available": bool(decided), "decided": decided}
    first = last = None
    if decided:
        match = (((s == 1) == np.array([z in label for z in decided])) | (s == -1)).all(-1)
        first, last = capture_bounds(match, f0, f1)
        changes = support_changes(s)
        near = int(changes[np.argmin(np.abs(changes - fh))]) if len(changes) else None
        stable = near is None or abs(near - fh) > round(STABLE_S * fps)
        decisive = all(ex_state[ZI[z]] >= 0 and not (rule.bands[z][0] < ex_height[ZI[z]] <= rule.bands[z][1])
                       for z in decided)
        cap.update(
            ground_exemplar=[z for z in decided if ex_state[ZI[z]] == 1],
            undecided_exemplar=[z for z in decided if ex_state[ZI[z]] == -1],
            ground_window=[z for z, c in zip(decided, contact) if c > 0.5],
            match_fraction=_num(match[win].mean(), 3), first=first, last=last,
            first_s=None if first is None else _num(first / fps, 3),
            last_s=None if last is None else _num(last / fps, 3),
            start_lag_s=None if first is None else _num((first - f0) / fps, 3),
            end_lead_s=None if last is None else _num((f1 - last) / fps, 3),
            nearest_change=near, nearest_change_s=None if near is None else _num((near - fh) / fps, 3),
            exemplar_stable=bool(stable), truth_unambiguous=bool(stable and decisive))

    cov = np.nan_to_num(rec["mat_valid_cov"][win], nan=0.0) >= COVERAGE_GATE
    valid = cov.mean() >= MIN_VALID
    zones = {}
    for z in _zones(label + decided):
        zi, row = ZI[z], {"label": z in label}
        if z in decided:
            row.update(exemplar=int(ex_state[zi]), marker_cm=_cm(ex_height[zi]),
                       contact_frac=_num(contact[decided.index(z)], 3))
        row["avatar_cm"] = _cm(rec["avatar_min_z"][fh, zi])
        if z in label:
            row["hover_cm"] = _cm(np.median(rec["avatar_min_z"][win, zi]))
        seen = cov & rec["attr_visible"][win, zi]
        row["load_n"] = _num(np.median(rec["mat_zone_load"][win, zi][seen]), 1) if seen.mean() >= MIN_VALID else None
        row["attr_visible_frac"] = _num(rec["attr_visible"][win, zi].mean(), 3)
        zones[z] = row
    mat = {"available": rec.meta["mat_available"], "coverage_valid": _num(cov.mean(), 3),
           "total_n": _num(np.median(rec["mat_total"][win][cov]), 1) if valid else None,
           "unexplained_n": _num(np.median(rec["mat_unexplained"][win][cov]), 1) if valid else None,
           "unexplained_exemplar_n": (_num(rec["mat_unexplained"][fh], 1)
                                      if np.nan_to_num(rec["mat_valid_cov"][fh]) >= COVERAGE_GATE else None)}

    missed = [z for z in decided if z not in label and ex_state[ZI[z]] == 1]
    phantom = [z for z in decided if z in label and ex_state[ZI[z]] == 0]
    hover = [z for z in label if zones[z]["hover_cm"] is not None
             and zones[z]["hover_cm"] > 100 * HOVER_M and z not in lifted]
    reasons = []

    def add(code, zs=(), **detail):
        reasons.append({"code": code, "fix": REASONS[code], "zones": list(zs), **detail})

    if not decided:
        add("capture_unavailable", label)
    if missed:
        add("missed_touch", missed)
    if phantom:
        add("phantom_support", phantom)
    if decided and first is None:
        add("no_capture_match")
    if first is not None and cap["start_lag_s"] > BOUNDARY_TOL_S:
        add("boundary_start", lag_s=cap["start_lag_s"])
    if last is not None and cap["end_lead_s"] > BOUNDARY_TOL_S:
        add("boundary_end", lead_s=cap["end_lead_s"])
    if decided and not cap["exemplar_stable"]:
        add("exemplar_unstable", change_s=cap["nearest_change_s"])
    if hover:
        add("hover", hover, max_cm=max(zones[z]["hover_cm"] for z in hover))
    if mat["unexplained_n"] is not None and mat["unexplained_n"] > UNEXPLAINED_N:
        blind = [z for z, r in zones.items() if r["attr_visible_frac"] < 0.5 and z not in lifted
                 and (r["label"] or (r.get("contact_frac") or 0) > 0.5)]
        add("unexplained_load", blind, unexplained_n=mat["unexplained_n"])

    family_hold = bool(hold.get("extend"))
    return {"hold_id": ids.hold_id(clip["stem"], fh), "stem": clip["stem"], "hold_index": index,
            "name": hold["name"], "group": clip.get("group"), "orientation": hold.get("orientation"),
            "family_hold": family_hold,
            "window": {"frame_start": f0, "frame_end": f1, "frame_hold": fh, "fps": fps,
                       "t_start": _num(f0 / fps, 3), "t_end": _num(f1 / fps, 3), "t_hold": _num(fh / fps, 3)},
            "label_ground": label, "capture": cap, "zones": zones, "mat": mat,
            "disagreements": {"missed_touch": missed, "phantom_support": phantom, "hover": hover},
            "reasons": reasons, "severity": severity(reasons, family_hold)}


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
def audit(manifest: Path = ids.DEFAULT_MANIFEST, rule: str = "calibrated", store_dir: Path = capture.STORE_DIR,
          calibration: dict | None = None) -> tuple[list[dict], dict, list[str]]:
    """``(records, context, failures)``: one record per hold of ``manifest`` under ``rule``."""
    cal = capture.load_calibration() if calibration is None else calibration
    r = make_rule(rule, cal)
    records, failures, stores, inputs = [], [], {}, [Path(manifest)]
    for clip in ids.load_manifest(Path(manifest))["clips"]:
        stem = clip["stem"]
        try:
            if ids.split_clip_name(stem)[1]:
                raise ValueError("audit x0 clips; a variant's frames map back through ids.source_frame_index")
            rec = capture.load(stem, store_dir, cal)
            if rec.meta["num_frames"] != int(clip["num_frames"]) or rec.fps != int(clip["fps"]):
                raise ValueError(f"store has {rec.meta['num_frames']} frames at {rec.fps} fps, the manifest "
                                 f"{clip['num_frames']} at {clip['fps']}")
            ev = clip_evidence(stem, rec, r)
            clip_records = [audit_hold(clip, i, h, rec, ev, r) for i, h in enumerate(clip["holds"])]
        except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
            failures.append(f"{stem}: {type(exc).__name__}: {exc}")
            continue
        records += clip_records
        # A store record is determined by its generator, its calibration and its sources; its npz
        # bytes are not (zip timestamps), so the audit id hashes this identity instead.
        stores[stem] = {"generator": rec.meta["generator"]["sha256"], "calibration": rec.meta["calibration"]["id"],
                        "inputs": rec.meta["inputs"]}
        inputs += [Path(store_dir) / f"{stem}.json", Path(store_dir) / f"{stem}.npz"]
    counts = collections.Counter(x["hold_id"] for x in records)
    failures += [f"duplicate hold_id {h}" for h, n in sorted(counts.items()) if n > 1]
    key = {"schema": SCHEMA_VERSION, "generator": ids.sha256_file(__file__), "rule": r.config, "config": CONFIG,
           "manifest": ids.sha256_file(manifest), "stores": stores}
    context = {"audit_id": f"{Path(manifest).stem}.{rule}.{ids.sha256_json(key)[:10]}", "manifest": Path(manifest),
               "rule": rule, "rule_config": r.config, "calibration_id": cal["id"], "inputs": inputs}
    return records, context, failures


def metrics(records: list[dict]) -> dict:
    """The progress metrics, over all holds and over family holds."""
    def block(rs):
        sup = [z for r in rs for z in r["zones"].values() if z["label"]]
        hov = np.array([z["hover_cm"] for z in sup], dtype=float)

        def count(code):
            hit = [x for r in rs for x in r["reasons"] if x["code"] == code]
            return {"holds": len(hit), "zones": sum(len(x["zones"]) for x in hit)}

        return {"holds": len(rs), "holds_with_capture": sum(r["capture"]["available"] for r in rs),
                "labelled_supports": len(sup),
                "hover_gt_cm": {str(c): int((hov > c).sum()) for c in HOVER_REPORT_CM},
                **{code: count(code) for code in REASONS},
                "severity": {lv: sum(r["severity"] == lv for r in rs) for lv in SEVERITY}}

    by_zone = collections.defaultdict(lambda: {"labelled": 0, "hover": 0})
    for r in records:
        for z in r["label_ground"]:
            by_zone[z]["labelled"] += 1
            by_zone[z]["hover"] += int((r["zones"][z]["hover_cm"] or 0) > 100 * HOVER_M)
    return {"all": block(records), "family": block([r for r in records if r["family_hold"]]),
            "labelled_hover_gt_2cm_by_zone": {z: by_zone[z] for z in ZONE_ORDER if z in by_zone}}


# --------------------------------------------------------------------------- #
# Queue
# --------------------------------------------------------------------------- #
def questions(r: dict) -> list[dict]:
    """Pass-B questions a hold's reasons ask; ``[]`` when its fixes are already determined."""
    w, cap, qs, state_zones = r["window"], r["capture"], [], []
    for x in r["reasons"]:
        if x["code"] in ("missed_touch", "phantom_support", "capture_unavailable"):
            state_zones += x["zones"]
        elif x["code"] == "hover":
            state_zones += [z for z in x["zones"] if z not in cap["decided"]]
        elif x["code"] == "no_capture_match":
            qs.append({"id": "support_set", "frames": [w["frame_start"], w["frame_end"]]})
        elif x["code"] in ("boundary_start", "boundary_end") and not any(q["id"] == "boundary" for q in qs):
            qs.append({"id": "boundary", "label": [w["frame_start"], w["frame_end"]],
                       "capture": [cap["first"], cap["last"]]})
        elif x["code"] == "exemplar_unstable":
            qs.append({"id": "exemplar", "frame": w["frame_hold"], "change_frame": cap["nearest_change"]})
        elif x["code"] == "unexplained_load":
            qs.append({"id": "hidden_support", "zones": x["zones"], "unexplained_n": x["unexplained_n"]})
    if state_zones:
        qs.insert(0, {"id": "source_state", "frame": w["frame_hold"], "zones": _zones(state_zones)})
    return qs


def stratum(r: dict) -> str:
    """Step 4's calibration strata. A clip's ``standing`` holds are not its group's pose, and the
    floats are split by size, because a random draw is mostly 2-3 cm and Step 4 needs 2-15 cm."""
    if r["group"] in ("arm_balance", "inversion") and r["name"] != "standing":
        return r["group"]
    floats = [z["avatar_cm"] for z in r["zones"].values() if z.get("exemplar") == 1 and z["avatar_cm"] is not None
              and 100 * HOVER_M < z["avatar_cm"] <= 100 * FLOAT_MAX_M]
    if floats:
        return "float_5_15" if max(floats) > 100 * HOVER_SEVERE_M else "float_2_5"
    return "other"


def build_queue(records: list[dict]) -> list[dict]:
    def item(r, pass_, purpose, qs):
        w, cap = r["window"], r["capture"]
        frames = [w["frame_start"], w["frame_hold"], w["frame_end"], cap.get("first"), cap.get("last")]
        if not cap.get("exemplar_stable", True):
            frames.append(cap["nearest_change"])
        focus = r["label_ground"] + [z for x in r["reasons"] for z in x["zones"]]
        return {"item_id": f"{r['hold_id']}#{pass_}", "hold_id": r["hold_id"], "stem": r["stem"],
                "pass": pass_, "purpose": purpose, "severity": r["severity"],
                "reasons": [x["code"] for x in r["reasons"]],
                "window": {**{k: w[k] for k in ("frame_start", "frame_end", "frame_hold", "fps")},
                           "key_frames": sorted({f for f in frames if f is not None})},
                "zones": _zones(focus), "questions": qs}

    asked = [(r, questions(r)) for r in records]
    flagged = sorted([(r, q) for r, q in asked if q],
                     key=lambda rq: (-SEVERITY.index(rq[0]["severity"]), not rq[0]["family_hold"], rq[0]["hold_id"]))
    clean = sorted([r for r, q in asked if not q], key=lambda r: _hash_order(r["hold_id"]))
    controls = clean[: math.ceil(CONTROL_FRACTION * len(clean))]
    pass_b = [item(r, "B", "flagged", q) for r, q in flagged]
    pass_b += [item(r, "B", "control", [{"id": "source_state", "frame": r["window"]["frame_hold"],
                                         "zones": r["label_ground"]}]) for r in controls]
    by_stratum = {s: sorted([r for r in records if r["capture"].get("truth_unambiguous") and stratum(r) == s],
                            key=lambda r: _hash_order(r["hold_id"])) for s in STRATA}
    interleaved = [by_stratum[s][i] for i in range(max(map(len, by_stratum.values()), default=0))
                   for s in STRATA if i < len(by_stratum[s])]
    pass_a = [dict(item(r, "A", "calibration", [{"id": q} for q in PASS_A_QUESTIONS]), stratum=stratum(r))
              for r in interleaved]
    for items in (pass_a, pass_b):
        for i, it in enumerate(items, 1):
            it["priority"] = i
    return pass_a + pass_b


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _evidence(r: dict) -> str:
    cap, parts = r["capture"], []
    if not cap["available"]:
        parts.append("no readable MoSh fit")
    elif cap["ground_exemplar"] != [z for z in r["label_ground"] if z in cap["decided"]]:
        parts.append(f"human at exemplar {{{', '.join(cap['ground_exemplar'])}}} vs label "
                     f"{{{', '.join(r['label_ground'])}}}")
    if any(x["code"] in ("boundary_start", "boundary_end") for x in r["reasons"]):
        parts.append(f"capture {cap['first_s']:.2f}-{cap['last_s']:.2f} s vs label "
                     f"{r['window']['t_start']:.2f}-{r['window']['t_end']:.2f} s")
    if cap.get("exemplar_stable") is False:
        parts.append(f"support changes {cap['nearest_change_s']:+.2f} s from the exemplar")
    for x in r["reasons"]:
        if x["code"] == "hover":
            parts.append("hover " + ", ".join(f"{z} {r['zones'][z]['hover_cm']:.1f} cm" for z in x["zones"]))
        if x["code"] == "unexplained_load":
            parts.append(f"unexplained {x['unexplained_n']:.0f} of {r['mat']['total_n']:.0f} N")
    return "; ".join(parts)


def summary_markdown(records: list[dict], context: dict, m: dict, queue: list[dict]) -> str:
    a, f = m["all"], m["family"]
    pct = (lambda n, d: f"{n} / {d} ({100 * n / d:.0f} %)" if d else "-")
    hz = (lambda b, code: f"{b[code]['zones']} in {b[code]['holds']} holds")
    rows = [("Holds (with capture)", f"{a['holds']} ({a['holds_with_capture']})", f"{f['holds']} ({f['holds_with_capture']})")]
    rows += [(f"Labelled ground supports hovering > {c} cm", pct(a["hover_gt_cm"][str(c)], a["labelled_supports"]),
              pct(f["hover_gt_cm"][str(c)], f["labelled_supports"])) for c in HOVER_REPORT_CM]
    rows += [(f"`{code}`", hz(a, code), hz(f, code)) for code in ("missed_touch", "phantom_support", "hover")]
    rows += [(f"`{code}` (holds)", str(a[code]["holds"]), str(f[code]["holds"]))
             for code in ("no_capture_match", "boundary_start", "boundary_end", "exemplar_unstable",
                          "unexplained_load", "capture_unavailable")]
    rows += [("Severity high / medium / low / none", " / ".join(str(a["severity"][s]) for s in SEVERITY[::-1]),
              " / ".join(str(f["severity"][s]) for s in SEVERITY[::-1]))]
    flagged = sorted([r for r in records if r["reasons"]],
                     key=lambda r: (-SEVERITY.index(r["severity"]), not r["family_hold"], r["hold_id"]))
    n_q = {(p, u): sum(1 for q in queue if q["pass"] == p and q["purpose"] == u)
           for p, u in (("A", "calibration"), ("B", "flagged"), ("B", "control"))}
    lines = [f"# Hold audit `{context['audit_id']}`", "",
             f"Manifest `{ids.display_path(context['manifest'])}`, rule `{context['rule']}`, calibration "
             f"`{context['calibration_id'][:12]}`, capture store `{capture.STORE_VERSION}`, git `{(ids.git_rev() or '-')[:10]}`. "
             "Generated by `reference_curation.audit` (BUILD_PLAN Step 2); the reason codes and severity "
             "rules are in its docstring.", "",
             "## Progress metrics", "", "Later steps drive these to zero. Zones are counted over the holds that carry the code.", "",
             "| Metric | All holds | Family holds |", "|---|---|---|"]
    lines += [f"| {a_} | {b_} | {c_} |" for a_, b_, c_ in rows]
    lines += ["", "Labelled supports hovering > 2 cm, by zone: " + ", ".join(
        f"{z} {v['hover']}/{v['labelled']}" for z, v in m["labelled_hover_gt_2cm_by_zone"].items()) + ".", "",
        "## Queue", "", f"Pass A (calibration candidates, truth unambiguous): {n_q['A', 'calibration']}. "
        f"Pass B: {n_q['B', 'flagged']} holds with questions plus {n_q['B', 'control']} controls.", "",
        f"## Flagged holds ({len(flagged)})", "", "| Severity | Hold | Name | Reasons | Evidence |", "|---|---|---|---|---|"]
    for r in flagged:
        why = "; ".join(x["code"] + (f" {'/'.join(x['zones'])}" if x["zones"] and x["code"] != "unexplained_load" else "")
                        for x in r["reasons"])
        lines.append(f"| {r['severity']}{' (family)' if r['family_hold'] else ''} | `{r['hold_id']}` | "
                     f"{r['name']} | {why} | {_evidence(r)} |")
    return "\n".join(lines) + "\n"


def write(records: list[dict], context: dict, out_root: Path = AUDITS_DIR) -> Path:
    out = Path(out_root) / context["audit_id"]
    out.mkdir(parents=True, exist_ok=True)
    queue, m = build_queue(records), metrics(records)
    head = {"schema_version": SCHEMA_VERSION, "audit_id": context["audit_id"]}
    for name, rows in (("holds.jsonl", records), ("queue.jsonl", queue)):
        (out / name).write_text("".join(json.dumps({**head, **row}) + "\n" for row in rows))
    (out / "summary.md").write_text(summary_markdown(records, context, m, queue))
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, context["inputs"]), "audit_id": context["audit_id"],
              "manifest": ids.display_path(context["manifest"]), "rule": context["rule"],
              "rule_config": context["rule_config"], "calibration_id": context["calibration_id"],
              "config": CONFIG, "metrics": m,
              "queue": {p: sum(q["pass"] == p for q in queue) for p in ("A", "B")}}
    (out / "audit.json").write_text(json.dumps(record, indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=ids.DEFAULT_MANIFEST)
    ap.add_argument("--rule", choices=RULES, default="calibrated")
    ap.add_argument("--store-dir", type=Path, default=capture.STORE_DIR)
    ap.add_argument("--calibration", type=Path, default=capture.CALIBRATION_PATH)
    ap.add_argument("--out-root", type=Path, default=AUDITS_DIR)
    args = ap.parse_args(argv)

    records, context, failures = audit(args.manifest, args.rule, args.store_dir, capture.load_calibration(args.calibration))
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures:
        print(f"audit: {len(failures)} failures; nothing written", file=sys.stderr)
        return 1
    context["inputs"].append(args.calibration)
    out = write(records, context, args.out_root)
    a = metrics(records)["all"]
    queue = [json.loads(line) for line in (out / "queue.jsonl").read_text().splitlines()]
    print(f"audit {context['audit_id']}: {a['holds']} holds ({a['holds_with_capture']} with capture); "
          f"flagged {sum(bool(r['reasons']) for r in records)} (high {a['severity']['high']}, medium "
          f"{a['severity']['medium']}, low {a['severity']['low']}); missed touch {a['missed_touch']['zones']} in "
          f"{a['missed_touch']['holds']} holds, phantom {a['phantom_support']['zones']} in "
          f"{a['phantom_support']['holds']}; hover > 2 cm {a['hover_gt_cm']['2']}/{a['labelled_supports']}; queue "
          f"A {sum(q['pass'] == 'A' for q in queue)}, B {sum(q['pass'] == 'B' for q in queue)} -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
