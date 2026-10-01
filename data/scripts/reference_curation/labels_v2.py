# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Labels v2 (BodyFix Step 4, items 5-6): labels v1.1's human-side decisions with the plant-v2 evidence, and the
open label problems of Steps 6-8 decided by machine evidence.

``build()`` takes every hold of labels v1.1 on the writer's 56-clip corpus and ``write()`` puts the result in
``data/reference_curation/labels/<manifest stem>.labels_v2.<hash>/`` (``holds.yaml``, ``annotations.jsonl``,
``evidence.jsonl``, ``labels.json``, ``summary.md``), the layout of labels v1.

What is kept (the human side, which no plant changes)
-----------------------------------------------------
Hold ids, exemplars, ground sets, the Tier-1 load-path classes, the source states and channels, and every
configuration decision of labels v1.1 (``labels.py``'s rules). The ground source (``sources.ground_source``) is the
same on capture store v4 as on v3 for every zone-frame of the corpus (the mat's tier never decides here), so the
ground sets still agree with the capture on every frame of every window.

What is re-measured on plant v2 (capture store v4, statics v2, the ``retarget_v2`` references)
--------------------------------------------------------------------------------------------
* every annotation's **avatar evidence**: a ground zone's height at the exemplar, its window median and
  ``realised`` (<= 2 cm); a pair's avatar gap at the exemplar, its window median, ``realised`` (<= 1 cm at the
  exemplar) and the share of the human's contact frames it is realised on;
* the **mat evidence** (``labels.mat_load`` on the plant-v2 attribution) and with it the **ground roles**:
  ``required_support`` where the mat confirms >= 50 N, else ``required_touch``; then, as labels v1.1 did,
  ``required_touch`` -> ``required_support`` where statics v2 finds the support required (``role_from``);
* the **statics block** (statics v2: the gated verdict on the plant-v2 reference, necessity, load interval);
* the hold's **exemplar fields** (``speed_at_hold``, ``pelvis_z``, ``rest_pose_m``, ``orientation``), recomputed
  on the reference the manifest now points at (labels v1 recomputed them only for a moved exemplar).

What is decided anew (item 6, machine evidence only)
----------------------------------------------------
* **Transitions.** labels v1.1 left 8 holds of the corpus ``unresolved``: no frame of their window is stable
  (a support change within +-0.5 s of every frame). Their support set changes every 0.3-0.9 s: they are
  transitions between holds, not holds. Status ``transition`` (``labels.transition``: the longest constant
  support run inside the source window); the gate excludes them.
* **The window is where the configuration is held.** A hold's window was the run of frames on which the human
  matches its ground set; a demanded pair (a configured ``required_touch``) could hold on a fraction of it (11 pairs
  under 90 %: Tree's foot placed on the thigh 2 s after the foot leaves the floor). The window is now the run
  around the exemplar, inside labels v1.1's window, on which the ground set **and** every demanded pair hold (an
  ``any_of`` group: any member; an undecided frame agrees), so a demanded pair holds on every frame of it.
* **``any_of`` groups are pairwise** (``any_of``): two configured pairs that share a zone and whose other zones are
  adjacent segments of one limb form a group of two; a pair can sit in two groups. Labels v1 merged them by
  union-find, which let a chain (foot-shank-thigh against one arm) collapse into "any one".
* **Not realised on the plant** (``labels.not_realised``): a configured pair the reference does not close at the
  exemplar (avatar gap > 1 cm). These are the commensurate colliders' geometry (Step 3's ``geometry_incompatible``
  requests: her fit 2-10 cm apart on plant v2, the trunk spheres short of the belly, a shin short of a thigh), not
  a defect the reference could repair, so they are carried as configured and the gate masks them, unless the
  critical-contact route (``critical``) admits one.
* **Pose role of secondary holds** (``pose_role``): an ``_h`` hold is ``preparation`` when the reviewer's samples
  call it not the labelled pose (majority) and its ground set is not the ground set of any family hold of the clip
  (a different contact configuration corroborates it), ``variant`` / ``family`` on the reviewer's majority, else
  ``undecided``. The name is kept (node identity downstream); ``pose_role`` says what it shows.
* **Critical body-body contacts (TODO B6)** come in as ``critical`` (``b6.py``): when the reviewer's
  self-consistency admits the class, a configured pair is demanded only if it is critical. Without it, the
  configuration is the deterministic one (labels v1's ``required_touch``), as here.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.labels_v2 [--critical <b6 json>]
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from extract_contact_configs import ORIENT_BINS, ZONE_ORDER, orientation_bins, stabilize_bins
from propose_hold_manifest import median_filter
from reference_curation import audit, capture_v4, fit_writer as fw, human_mesh as hm, ids, labels as L1, pressure_v2
from reference_curation import sources, statics_v2, verdicts

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.labels_v2"
SCHEMA_VERSION = 1
LABELS_VERSION = "v2"
LABELS_DIR = ids.DATA_ROOT / "labels"
LABELS_V11 = LABELS_DIR / "holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828"
PLANT = "v2"
ZI = L1.ZI
COL = L1.PAIR_COLUMN
SUPPORT_LOAD_N = L1.SUPPORT_LOAD_N
FLOAT_CM = 100 * audit.HOVER_M                 # 2.0: a ground zone is realised at <= 2 cm (window median)
PAIR_REALISED_CM = verdicts.PAIR_TOUCH_CM      # 1.0: a pair is realised at <= 1 cm
H_SUFFIX = re.compile(r"_h\d+$")
STATUS = ("kept", "moved", "relabelled", "transition")
POSE_ROLES = ("family", "standing", "variant", "preparation", "undecided")
CONFIG = {"support_load_n": SUPPORT_LOAD_N, "float_cm": FLOAT_CM, "pair_realised_cm": PAIR_REALISED_CM,
          "stable_s": audit.STABLE_S, "coverage_gate_col0": audit.COVERAGE_GATE, "min_valid": audit.MIN_VALID,
          "window": "the run around the exemplar, inside labels v1.1's window, where the ground set and every demanded "
                    "pair hold (any_of: any member; undecided agrees)",
          "any_of": "pairwise: two configured pairs sharing one zone, the other zones adjacent segments of one limb",
          "transition": "labels v1.1 'unresolved': no frame of the window stable within +-stable_s",
          "pose_role": "_h holds: reviewer majority over every valid Pass-B sample + the ground set against the clip's "
                       "family holds"}


def _num(x, nd: int = 3):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _cm(x):
    return _num(100.0 * float(x), 2) if x is not None else None


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def read_v11(labels_dir: Path = LABELS_V11) -> dict:
    manifest = copy.deepcopy(ids.load_manifest(Path(labels_dir) / "holds.yaml"))
    anns = collections.defaultdict(list)
    for line in open(Path(labels_dir) / "annotations.jsonl"):
        a = json.loads(line)
        anns[a["hold_id"]].append(a)
    return {"dir": Path(labels_dir), "id": manifest["labels"]["labels_id"], "manifest": manifest, "anns": dict(anns)}


def default_audit_dir(labels_id: str, rule: str = "calibrated") -> Path:
    found = sorted((audit.AUDITS_DIR).glob(f"{labels_id}.audit_v2.{rule}.*"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one audit v2 ({rule}) of {labels_id}, found {len(found)}")
    return found[0]


def default_statics_dir(labels_id: str) -> Path:
    found = sorted(statics_v2.STATICS_DIR.glob(f"{labels_id}.statics_v2.*"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one statics v2 folder of {labels_id}, found {len(found)}")
    return found[0]


# --------------------------------------------------------------------------- #
# One clip's evidence
# --------------------------------------------------------------------------- #
def clip_evidence(stem: str, manifest: dict, rid: str = capture_v4.RETARGET_ID) -> SimpleNamespace:
    """The human side (ground and pair sources, their change distances) on capture store v4, and the reference's
    proposer fields (speed, orientation bins) on the ``retarget_v2`` motion."""
    rec = capture_v4.load(stem, rid=rid, rebuild=False)
    state, channel = sources.ground_source(rec)
    pairs = sources.pair_source(rec)
    T, fps = state.shape[0], rec.fps
    mot = torch.load(capture_v4.motion_path(stem, rid), map_location="cpu", weights_only=False)
    if mot["rigid_body_pos"].shape[0] != T:
        raise ValueError(f"{stem}: the reference has {mot['rigid_body_pos'].shape[0]} frames, the store {T}")
    speed = median_filter(mot["rigid_body_vel"].norm(dim=-1).mean(dim=-1).numpy(),
                          round(manifest["stillness"]["smooth_s"] * fps))
    raw, _ = orientation_bins(mot["rigid_body_rot"][:, 0])
    bins = stabilize_bins(raw, round(manifest["detector_thresholds"]["reorient_s"] * fps))
    return SimpleNamespace(stem=stem, rec=rec, state=state, channel=channel, pairs=pairs, fps=fps, num_frames=T,
                           change_dist=L1._distance_to(audit.support_changes(state), T),
                           zone_dist=L1._per_column_dist(state), pair_dist=L1._per_column_dist(pairs),
                           speed=speed, bins=bins, pos=mot["rigid_body_pos"].double().numpy(),
                           rot=mot["rigid_body_rot"].double().numpy())


# --------------------------------------------------------------------------- #
# The window, any_of, transitions and pose roles (pure)
# --------------------------------------------------------------------------- #
def any_of_groups(hold_id: str, configured: list[tuple[str, str]]) -> list[dict]:
    """Pairwise alternative groups: ``[{id, members: [p, q]}]`` for every two configured pairs that share exactly one
    zone and whose other zones are adjacent segments of one limb (``labels.LIMB_CHAINS``). Not transitive."""
    out = []
    for i, p in enumerate(configured):
        for q in configured[i + 1:]:
            if len(set(p) & set(q)) == 1 and frozenset(set(p) ^ set(q)) in L1.LIMB_CHAINS:
                a, b = sorted((p, q), key=COL.get)
                out.append({"id": f"{hold_id}/any_of/{L1.pair_name(a)}|{L1.pair_name(b)}",
                            "members": [L1.pair_name(a), L1.pair_name(b)]})
    return out


def configuration_holds(clip: SimpleNamespace, ground, demanded: list[str], groups: list[dict]) -> np.ndarray:
    """``[T]`` frames on which the human holds the configuration: the ground set (``labels.matching``), every
    demanded pair outside a group, and at least one member of every group (an undecided state agrees)."""
    ok = L1.matching(clip.state, ground)
    grouped = {m for g in groups for m in g["members"]}
    for p in demanded:
        if p not in grouped:
            ok &= clip.pairs[:, COL[L1.parse_pair(p)]] != 0
    for g in groups:
        ok &= np.any([clip.pairs[:, COL[L1.parse_pair(m)]] != 0 for m in g["members"]], axis=0)
    return ok


def window_around(ok: np.ndarray, s: int, e: int, ex: int) -> tuple[int, int] | None:
    """The run of ``ok`` containing ``ex`` inside ``[s, e]``; ``None`` if ``ok[ex]`` is False."""
    if not ok[ex]:
        return None
    a = b = ex
    while a > s and ok[a - 1]:
        a -= 1
    while b < e and ok[b + 1]:
        b += 1
    return a, b


def longest_constant_run(clip: SimpleNamespace, s: int, e: int) -> tuple[int, int]:
    """``(frames, start)`` of the longest run of one support set inside ``[s, e]``."""
    best, start, run_start = 0, s, s
    for t in range(s, e + 2):
        if t == e + 1 or (t > s and not np.array_equal(clip.state[t], clip.state[t - 1])):
            if t - run_start > best:
                best, start = t - run_start, run_start
            run_start = t
    return best, start


def is_secondary(name: str) -> bool:
    return bool(H_SUFFIX.search(name))


def family_ground_sets(holds: list[dict]) -> list[frozenset]:
    return [frozenset(h["pairs_ground"]) for h in holds if not is_secondary(h["name"]) and h["name"] != "standing"]


def pose_role(hold: dict, family_sets: list[frozenset], votes: list[str]) -> tuple[str, dict]:
    """``(role, evidence)`` of one hold (module docstring). ``votes``: the reviewer's ``variant.matches_label`` of
    every valid Pass-B sample of the hold (``yes``, ``variant``, ``no``, ``cannot_tell``)."""
    if hold["name"] == "standing":
        return "standing", {}
    if not is_secondary(hold["name"]):
        return "family", {}
    answered = [v for v in votes if v in ("yes", "variant", "no")]
    counts = collections.Counter(answered)
    same_ground = frozenset(hold["pairs_ground"]) in family_sets
    ev = {"votes": dict(collections.Counter(votes)), "ground_equals_a_family_hold": same_ground}
    if not answered:
        return "undecided", ev
    top, n = counts.most_common(1)[0]
    if n * 2 <= len(answered):               # no majority
        return "undecided", ev
    if top == "no":
        return ("preparation" if not same_ground else "undecided"), ev
    if top == "yes":
        return ("family" if same_ground else "variant"), ev
    return "variant", ev


# --------------------------------------------------------------------------- #
# One hold
# --------------------------------------------------------------------------- #
def exemplar_fields(clip: SimpleNamespace, frame: int) -> dict:
    """The proposer's per-exemplar fields on the reference the manifest points at (``labels.exemplar_fields``)."""
    return L1.exemplar_fields(clip, frame)


def ground_evidence(clip: SimpleNamespace, s: int, e: int, ex: int, zi: int) -> tuple[dict, dict, float | None]:
    rec = clip.rec
    load, visible = L1.mat_load(rec, s, e, zi)
    hover = float(np.median(rec["avatar_min_z"][s:e + 1, zi]))
    mat_contact = bool(sources.mat_contact(rec)[ex, zi])
    return ({"cm": _cm(rec["avatar_min_z"][ex, zi]), "hover_cm": _cm(hover), "realised": bool(100 * hover <= FLOAT_CM),
             "plant": PLANT},
            {"load_n": load, "visible_fraction": visible, "contact_at_exemplar": mat_contact, "attribution": "plant v2"},
            load)


def pair_evidence(clip: SimpleNamespace, s: int, e: int, ex: int, k: int) -> dict:
    g = clip.rec["avatar_pair_gap"][:, k].astype(np.float64)
    human = clip.pairs[s:e + 1, k] == 1
    on = g[s:e + 1][human] <= PAIR_REALISED_CM / 100.0 if human.any() else np.zeros(0, bool)
    return {"gap_cm": _cm(g[ex]), "window_gap_cm": _cm(np.median(g[s:e + 1])),
            "realised": bool(100 * g[ex] <= PAIR_REALISED_CM),
            "realised_on_human_contact": _num(on.mean(), 3) if len(on) else None, "plant": PLANT}


def ground_role(a: dict, load_n: float | None, statics_row: dict | None) -> tuple[str, str | None]:
    """``(role, role_from)``: labels v1's rule on the plant-v2 mat, then the statics promotion."""
    if not a["in_configuration"] or a["source_state"] != "observed_contact":
        return a["target_role"], None            # carried (unspecified) and forbidden supports are human-side
    if load_n is not None and load_n >= SUPPORT_LOAD_N:
        return "required_support", "mat"
    if statics_row is not None and statics_row["necessity"] == "required":
        return "required_support", "statics"
    return "required_touch", None


STATICS_KEYS = ("load_n", "load_min_n", "load_max_n", "s_without", "effort_relief", "height_cm", "gap_cm", "realised")


def statics_block(row: dict | None, hold_row: dict | None, sid: str) -> dict | None:
    if row is None:
        return None
    basis = row["necessity_basis"]
    v = row[basis] or {}
    return {"statics_id": sid, "hold_verdict": (hold_row or {}).get("verdict"), "necessity": row["necessity"],
            "basis": f"{basis}_plant_v2", **{k: v.get(k) for k in STATICS_KEYS}}


def reconcile(clip: SimpleNamespace, hold: dict, anns: list[dict], ctx: SimpleNamespace,
              family_sets: list[frozenset]) -> tuple[dict, list[dict], dict]:
    """``(hold for holds.yaml, annotations, notes)`` of one hold of labels v1.1."""
    hid = hold["hold_id"]
    lab = hold["labels"]
    s0, e0, ex = int(hold["frame_start"]), int(hold["frame_end"]), int(hold["frame_hold"])
    ground = {p[:-2] for p in hold["pairs_ground"]}
    status = "transition" if lab["status"] == "unresolved" else lab["status"]
    notes = {}
    configured_pairs = [a["contact"] for a in anns if a["kind"] == "pair" and a["in_configuration"]]
    critical = (ctx.critical or {}).get(hid)
    if ctx.critical is not None:            # B6 admitted: a configured pair is demanded only when critical
        demanded = [p for p in configured_pairs if critical and p in critical] + \
                   [p for p in (critical or []) if p not in configured_pairs]
    else:
        demanded = list(configured_pairs)
    groups = any_of_groups(hid, [L1.parse_pair(p) for p in demanded])
    s, e = s0, e0
    if status != "transition":
        ok = configuration_holds(clip, ground, demanded, groups)
        run = window_around(ok, s0, e0, ex)
        if run is None:
            raise ValueError(f"{hid}: the configuration does not hold at the exemplar")
        s, e = run
        if (s, e) != (s0, e0):
            notes["window_trimmed_for_pairs"] = {"from": [s0, e0], "to": [s, e]}
    else:
        n, at = longest_constant_run(clip, s0, e0)
        notes["transition"] = {"longest_constant_support_s": _num(n / clip.fps, 3), "at": at}
    fps = clip.fps
    roles, not_realised, out = {}, [], []
    statics_rows = ctx.statics["contacts"]
    for a in anns:
        a = copy.deepcopy(a)
        a.pop("labels_id", None)
        a["role_v1_1"] = a["target_role"]
        a["interval"] = {**a["interval"], "start_frame": s, "end_frame_exclusive": e + 1}
        zs = a["zones"]
        srow = statics_rows.get((hid, a["contact"]))
        a["statics"] = statics_block(srow, ctx.statics["holds"].get(hid), ctx.statics_id)
        if a["kind"] == "ground":
            zi = ZI[zs[0]]
            avatar, mat, load = ground_evidence(clip, s, e, ex, zi)
            a["evidence"] = {**a["evidence"], "avatar": avatar, "mat": mat}
            a["window_contact_fraction"] = L1._window_fraction(clip.state[s:e + 1, zi])
            role, frm = ground_role(a, load, srow) if status != "transition" else (a["target_role"], None)
            a["target_role"], a["role_from"] = role, frm
            if a["in_configuration"] and a["source_state"] == "observed_contact" and not avatar["realised"]:
                not_realised.append(a["contact"])
            a["evidence_ids"] = ctx.ev_common + [f"capture:v1:{clip.stem}", f"capture:v4:{clip.stem}",
                                                 f"motion:{clip.stem}", f"pressure:{ctx.rid}"] + \
                ([f"capture:v2:{clip.stem}", f"capture:v3:{clip.stem}"] if a["source_channel"] == "capture_fit_mesh" else []) + \
                ([f"statics:{ctx.statics_id}"] if srow is not None else [])
        else:
            k = COL[L1.parse_pair(a["contact"])]
            a["evidence"] = {**a["evidence"], "avatar": pair_evidence(clip, s, e, ex, k)}
            a["window_contact_fraction"] = L1._window_fraction(clip.pairs[s:e + 1, k])
            a["evidence"]["mesh"]["window_gap_cm"] = _cm(np.median(clip.rec["human_pair_gap"][s:e + 1, k]))
            was = a["in_configuration"]
            a["in_configuration"] = a["contact"] in demanded or (a["source_state"] == "unknown" and a["source_label"])
            if critical is not None and a["contact"] in critical:
                a["critical"] = critical[a["contact"]]
            if ctx.critical is not None and was and not a["in_configuration"]:
                a["label_action"] = "demoted"
                a["reasons"] = a["reasons"] + ["not_critical"]
                a["target_role"] = "allowed" if a["target_role"] == "required_touch" else a["target_role"]
            if ctx.critical is not None and not was and a["in_configuration"]:
                a["label_action"] = "added"
                a["reasons"] = a["reasons"] + ["critical"]
                a["target_role"] = "required_touch"
            a["any_of"] = [g["id"] for g in groups if a["contact"] in g["members"]]
            a["alternative_group"] = None   # superseded by the pairwise any_of
            if a["in_configuration"] and a["source_state"] == "observed_contact" and not a["evidence"]["avatar"]["realised"]:
                not_realised.append(a["contact"])
            a["evidence_ids"] = ctx.ev_common + [f"capture:v2:{clip.stem}", f"capture:v4:{clip.stem}",
                                                 f"motion:{clip.stem}"] + \
                ([f"statics:{ctx.statics_id}"] if srow is not None else []) + \
                [x for x in ((a.get("critical") or {}).get("evidence_ids") or [])]
        if status == "transition":
            a["status"] = "unresolved"
            a["reasons"] = [r for r in a["reasons"] if r != "hold_unresolved"] + ["hold_is_a_transition"]
        a["evidence_ids"] = list(dict.fromkeys(a["evidence_ids"]))
        roles[a["contact"]] = a["target_role"]
        out.append(a)
    h = copy.deepcopy(hold)
    h.update(frame_start=s, frame_end=e, t_start=_num(s / fps, 4), t_end=_num(e / fps, 4),
             duration_s=_num((e - s + 1) / fps, 3))
    fields_v11 = {k: hold.get(k) for k in ("orientation", "speed_at_hold", "pelvis_z", "rest_pose_m")}
    h.update(exemplar_fields(clip, ex))
    changed_fields = {k: v for k, v in fields_v11.items() if k == "orientation" and v != h["orientation"]}
    configured = [a["contact"] for a in out if a["in_configuration"]]
    h["pairs"] = [c for c in configured if c.endswith(":G")] + [c for c in configured if not c.endswith(":G")]
    h["pairs_ground"] = [c for c in configured if c.endswith(":G")]
    votes = ctx.votes.get(hid, [])
    prole, pev = pose_role(hold, family_sets, votes)
    if ctx.vote_ids.get(hid):
        pev["verdicts"] = list(ctx.vote_ids[hid])
    changes = [x for x in lab["changes"] if x not in L1.HOLD_STATUS]
    if status != lab["status"]:
        changes.append(f"status:{lab['status']}->{status}")
    if "window_trimmed_for_pairs" in notes:
        changes.append("window_trimmed_for_pairs")
    changes += [f"role:{a['contact']}:{a['role_v1_1']}->{a['target_role']}" for a in out if a["role_v1_1"] != a["target_role"]]
    if changed_fields:
        changes.append(f"orientation:{fields_v11['orientation']}->{h['orientation']}")
    h["labels"] = {
        "status": status, "changes": changes,
        "source": lab["source"],
        "source_v1_1": {"status": lab["status"], "frame_start": s0, "frame_end": e0, "frame_hold": ex,
                        "pairs": list(hold["pairs"]), "exemplar_fields": fields_v11},
        "unverified": [a["contact"] for a in out if a["label_action"] == "carried"],
        "roles": roles, "any_of": groups, "not_realised": sorted(set(not_realised), key=configured.index),
        "pose_role": prole, "pose_role_evidence": pev, "review_v3": lab.get("review"),
        "statics": None if hid not in ctx.statics["holds"] else
        {"statics_id": ctx.statics_id, "verdict": ctx.statics["holds"][hid]["verdict"],
         "s_star": ctx.statics["holds"][hid]["gated"]["s_star"]},
        **notes}
    return h, out, notes


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
def _store_rows(stem: str, rid: str) -> list[dict]:
    rows = []
    for v, d in (("v1", ids.OUTPUT_ROOT / "capture" / "v1"), ("v2", hm.STORE_DIR), ("v3", sources.STORE_DIR),
                 ("v4", capture_v4.STORE_DIR)):
        path = Path(d) / f"{stem}.json"
        rows.append({"id": f"capture:{v}:{stem}", "kind": "capture_store", "store": f"capture/{v}",
                     "path": ids.display_path(path), "sha256": ids.sha256_file(path)})
    m = capture_v4.motion_path(stem, rid)
    rows.append({"id": f"motion:{stem}", "kind": "reference_motion", "plant": PLANT, "retarget_id": rid,
                 "path": ids.display_path(m), "sha256": ids.sha256_file(m)})
    return rows


def build(v11: dict, audit_dir: Path, statics_dir: Path, rid: str = capture_v4.RETARGET_ID, critical: dict | None = None,
          votes: dict | None = None, vote_ids: dict | None = None, extra_evidence: list[dict] = ()) -> dict:
    """Labels v2 of every hold of ``v11`` on the writer's corpus. ``critical``: B6's ``{hold_id: {pair: block}}`` when
    the critical-contact class is admitted (``None``: the deterministic configuration). ``votes``: per hold, the
    reviewer's variant answers (every valid Pass-B sample)."""
    manifest = v11["manifest"]
    corpus, dropped = fw.corpus()
    st = statics_v2.load(statics_dir)
    statics_id = st["record"]["statics_id"]
    aud = json.loads((Path(audit_dir) / "audit.json").read_text())
    pres = pressure_v2.RECORD_ROOT / rid / "pressure.json"
    ev = {f"labels:{v11['id']}": {"id": f"labels:{v11['id']}", "kind": "labels_human_side",
                                   "path": ids.display_path(v11["dir"] / "annotations.jsonl"),
                                   "sha256": ids.sha256_file(v11["dir"] / "annotations.jsonl")},
          f"manifest:{v11['id']}": {"id": f"manifest:{v11['id']}", "kind": "labels_manifest",
                                     "path": ids.display_path(v11["dir"] / "holds.yaml"),
                                     "sha256": ids.sha256_file(v11["dir"] / "holds.yaml")},
          f"audit:{aud['audit_id']}": {"id": f"audit:{aud['audit_id']}", "kind": "audit",
                                        "path": ids.display_path(Path(audit_dir) / "holds.jsonl"),
                                        "sha256": ids.sha256_file(Path(audit_dir) / "holds.jsonl")},
          f"rule:{MODULE}": {"id": f"rule:{MODULE}", "kind": "rule", "path": ids.display_path(__file__),
                             "sha256": ids.sha256_file(__file__)},
          f"statics:{statics_id}": {"id": f"statics:{statics_id}", "kind": "statics", "plant": PLANT,
                                    "path": ids.display_path(Path(statics_dir) / "contacts.jsonl"),
                                    "sha256": ids.sha256_file(Path(statics_dir) / "contacts.jsonl")},
          f"pressure:{rid}": {"id": f"pressure:{rid}", "kind": "mat_attribution", "plant": PLANT,
                              "path": ids.display_path(pres), "sha256": ids.sha256_file(pres)}}
    for row in extra_evidence:
        ev[row["id"]] = row
    ctx = SimpleNamespace(statics=st, statics_id=statics_id, rid=rid, critical=critical, votes=votes or {},
                          vote_ids=vote_ids or {},
                          ev_common=[f"labels:{v11['id']}", f"manifest:{v11['id']}", f"audit:{aud['audit_id']}",
                                     f"rule:{MODULE}"])
    clips_out, holds_out, annotations, failures, identities, notes = [], {}, [], [], {}, {}
    for entry in manifest["clips"]:
        stem = entry["stem"]
        if stem not in corpus:
            continue
        try:
            clip = clip_evidence(stem, manifest, rid)
            for row in _store_rows(stem, rid):
                ev[row["id"]] = row
            identities[stem] = capture_v4.identity(clip.rec)
            fam = family_ground_sets(entry["holds"])
            new_holds = []
            for hold in entry["holds"]:
                h, anns, n = reconcile(clip, hold, v11["anns"][hold["hold_id"]], ctx, fam)
                new_holds.append(h)
                holds_out[h["hold_id"]] = h
                annotations += anns
                if n:
                    notes[h["hold_id"]] = n
        except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
            import traceback

            failures.append(f"{stem}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
            continue
        fitm = fw.OUT_ROOT / FIT_ID / f"{stem}.motion"
        c = {k: v for k, v in entry.items() if k != "holds"}
        c["labels"] = {"source_v1_1": entry["source"], "v1_1": entry.get("labels"), "fit": str(fitm.resolve()),
                       "retarget_id": rid, "retargeted": True, "plant": PLANT,
                       "plant_sha256": plant_identity.sha256(PLANT)}
        c["source"] = str(capture_v4.motion_path(stem, rid).resolve())
        clips_out.append({**c, "holds": new_holds})
    return {"clips": clips_out, "holds": holds_out, "annotations": annotations, "evidence": ev, "identities": identities,
            "failures": failures, "notes": notes, "v11": v11, "dropped": dropped, "audit_id": aud["audit_id"],
            "statics_id": statics_id, "rid": rid, "critical": critical, "manifest": manifest}


FIT_ID = "fit_v2.v1.b8f5e86fc9"


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def verify_evidence(evidence: dict) -> list[str]:
    """Every evidence row with a path verifies by sha256."""
    bad = []
    for k, row in evidence.items():
        p = row.get("path")
        if p is None:
            continue
        path = Path(p) if Path(p).is_absolute() else ids.REPO / p
        if not path.exists():
            bad.append(f"evidence {k}: {p} is missing")
        elif ids.sha256_file(path) != row["sha256"]:
            bad.append(f"evidence {k}: {p} sha256 differs")
    return bad


def check(result: dict) -> list[str]:
    """Labels v1's contract (``labels.check``'s rules) on labels v2, plus: every annotation's evidence ids exist and
    verify by sha256; every demanded pair holds on every frame of its window; ground sets agree with capture store
    v4 at the exemplar and on every window frame."""
    problems = []
    ev = result["evidence"]
    seen = set()
    for a in result["annotations"]:
        key = (a["hold_id"], a["contact"])
        if key in seen:
            problems.append(f"{key}: annotated twice")
        seen.add(key)
        role, state = a["target_role"], a["source_state"]
        if role not in L1.ROLES:
            problems.append(f"{key}: unknown role {role}")
        unstable_pair = a["kind"] == "pair" and not a["stable"]
        if role in ("required_support", "required_touch") and (state != "observed_contact" or unstable_pair):
            problems.append(f"{key}: {role} without a stable observed contact")
        if role == "required_support" and a["kind"] != "ground":
            problems.append(f"{key}: required_support on a pair")
        if a["kind"] == "ground" and a["in_configuration"] != (state == "observed_contact" or a["label_action"] == "carried"):
            problems.append(f"{key}: the ground set must be the observed contacts and the carried labels")
        if a["kind"] == "pair" and a["in_configuration"] and not (role == "required_touch" or a["label_action"] == "carried"):
            problems.append(f"{key}: a configured pair must be required_touch or carried")
        if not a["evidence_ids"]:
            problems.append(f"{key}: no evidence ids")
        problems += [f"{key}: evidence id {i} is not in the index" for i in a["evidence_ids"] if i not in ev]
    problems += verify_evidence(ev)
    by_hold = collections.defaultdict(list)
    for a in result["annotations"]:
        by_hold[a["hold_id"]].append(a)
    stems = sorted({h["hold_id"].rsplit("@", 1)[0] for h in result["holds"].values()})
    for stem in stems:
        rec = capture_v4.load(stem, rid=result["rid"], rebuild=False)
        state, _ = sources.ground_source(rec)
        pairs = sources.pair_source(rec)
        for hid, h in result["holds"].items():
            if not hid.startswith(stem + "@"):
                continue
            ground = {p[:-2] for p in h["pairs_ground"]}
            s, e, ex = h["frame_start"], h["frame_end"], h["frame_hold"]
            bad = [z for z in ZONE_ORDER if state[ex, ZI[z]] >= 0 and (state[ex, ZI[z]] == 1) != (z in ground)]
            if bad:
                problems.append(f"{hid}: ground set disagrees with the capture at the exemplar on {bad}")
            if not L1.matching(state[s:e + 1], ground).all():
                problems.append(f"{hid}: ground set disagrees with the capture inside the window")
            if not s <= ex <= e:
                problems.append(f"{hid}: exemplar outside the window")
            configured = {a["contact"] for a in by_hold[hid] if a["in_configuration"]}
            if set(h["pairs"]) != configured:
                problems.append(f"{hid}: pairs differ from the configured annotations")
            problems += [f"{hid}: pose-role verdict {v} is not in the evidence index"
                         for v in h["labels"]["pose_role_evidence"].get("verdicts", []) if v not in ev]
            if h["labels"]["status"] != "transition":
                demanded = [p for p in h["pairs"] if ":" not in p]
                clip = SimpleNamespace(state=state, pairs=pairs)
                ok = configuration_holds(clip, ground, demanded, h["labels"]["any_of"])
                if not ok[s:e + 1].all():
                    problems.append(f"{hid}: a demanded pair does not hold on every frame of the window")
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
        sup = [a for a in g if a["in_configuration"] and a["source_state"] == "observed_contact"]
        conf_p = [a for a in p if a["in_configuration"]]
        return {
            "holds": len(hs), "status": dict(collections.Counter(h["labels"]["status"] for h in hs)),
            "pose_role": dict(collections.Counter(h["labels"]["pose_role"] for h in hs)),
            "windows_trimmed_for_pairs": sum("window_trimmed_for_pairs" in h["labels"] for h in hs),
            "orientation_changed": sum(any(c.startswith("orientation:") for c in h["labels"]["changes"]) for h in hs),
            "ground_roles": dict(collections.Counter(a["target_role"] for a in g)),
            "ground_roles_v1_1": dict(collections.Counter(a["role_v1_1"] for a in g)),
            "ground_role_from": dict(collections.Counter(a.get("role_from") for a in g if a.get("role_from"))),
            "ground_role_changes": dict(collections.Counter(f"{a['role_v1_1']}->{a['target_role']}" for a in g
                                                          if a["role_v1_1"] != a["target_role"])),
            "supports_configured": len(sup),
            "supports_not_realised": sum(not a["evidence"]["avatar"]["realised"] for a in sup),
            "pair_roles": dict(collections.Counter(a["target_role"] for a in p)),
            "pairs_configured": len(conf_p),
            "pairs_configured_not_realised": sum(a["source_state"] == "observed_contact" and
                                                 not a["evidence"]["avatar"]["realised"] for a in conf_p),
            "pairs_configured_window_fraction_lt_0_9": sum((a["window_contact_fraction"] or 1.0) < 0.9 for a in conf_p
                                                          if a["source_state"] == "observed_contact"),
            "any_of_groups": sum(len(h["labels"]["any_of"]) for h in hs),
            "critical_pairs": sum(bool(a.get("critical")) for a in p),
        }

    return {"all": block(holds), "family": block([h for h in holds if h.get("extend")])}


def labels_id(result: dict) -> str:
    key = {"schema": SCHEMA_VERSION, "generator": ids.sha256_file(__file__), "config": CONFIG,
           "base": result["v11"]["id"], "audit": result["audit_id"], "statics": result["statics_id"], "retarget": result["rid"],
           "clips": result["identities"], "critical": result["critical"] and ids.sha256_json(result["critical"]),
           "evidence": sorted((k, r.get("sha256")) for k, r in result["evidence"].items())}
    stem = Path(result["manifest"]["labels"]["source_manifest"]).stem
    return f"{stem}.labels_{LABELS_VERSION}.{ids.sha256_json(key)[:10]}"


def summary_markdown(result: dict, lid: str, m: dict) -> str:
    a, f = m["all"], m["family"]

    def row(name, key, sub=None):
        va, vf = (a[key], f[key]) if sub is None else (a[key].get(sub, 0), f[key].get(sub, 0))
        return f"| {name} | {va} | {vf} |"

    lines = [f"# Labels `{lid}`", "",
             f"Generated by `{MODULE}` (BodyFix Step 4): labels v1.1 `{result['v11']['id']}`'s human-side decisions with "
             f"the plant-v2 evidence (capture store v4, statics `{result['statics_id']}`, references `{result['rid']}`). "
             "The rules are in the module docstring. `hold_id` names the source hold; `frame_hold` is the corrected "
             "exemplar.", "", "| Metric | All holds | Family holds |", "|---|---|---|", row("Holds", "holds")]
    lines += [row(f"Status `{s}`", "status", s) for s in STATUS]
    lines += [row(f"Pose role `{r}`", "pose_role", r) for r in POSE_ROLES]
    lines += [row("Windows trimmed to the demanded pairs", "windows_trimmed_for_pairs"),
              row("Orientation bin changed on the new reference", "orientation_changed")]
    lines += [row(f"Ground role `{r}`", "ground_roles", r) for r in ("required_support", "required_touch", "unspecified",
                                                                     "forbidden_support")]
    lines += [row(f"... labels v1.1 `{r}`", "ground_roles_v1_1", r) for r in ("required_support", "required_touch")]
    lines += [f"| Ground supports `required_support` from the mat / statics | {a['ground_role_from'].get('mat', 0)} / "
              f"{a['ground_role_from'].get('statics', 0)} | {f['ground_role_from'].get('mat', 0)} / "
              f"{f['ground_role_from'].get('statics', 0)} |",
              f"| Configured supports not realised (window median > 2 cm) | {a['supports_not_realised']} / "
              f"{a['supports_configured']} | {f['supports_not_realised']} / {f['supports_configured']} |",
              f"| Configured pairs not realised at the exemplar (> 1 cm) | {a['pairs_configured_not_realised']} / "
              f"{a['pairs_configured']} | {f['pairs_configured_not_realised']} / {f['pairs_configured']} |",
              row("Configured pairs held on < 90 % of the window", "pairs_configured_window_fraction_lt_0_9"),
              row("`any_of` groups (pairwise)", "any_of_groups"), row("Critical pairs (B6)", "critical_pairs")]
    lines += ["", f"Ground role changes from labels v1.1: {a['ground_role_changes'] or 'none'}.", "",
              f"Dropped clips: {', '.join(f'`{s}` ({r})' for s, r in result['dropped'].items())}.", "",
              "## Holds whose window or status changed", "", "| Hold | Status | Window (s) v1.1 -> v2 | Note |", "|---|---|---|---|"]
    for c in result["clips"]:
        fps = c["fps"]
        for h in c["holds"]:
            lab = h["labels"]
            src = lab["source_v1_1"]
            if lab["status"] == "transition" or "window_trimmed_for_pairs" in lab:
                note = (f"longest constant support {lab['transition']['longest_constant_support_s']} s" if "transition" in lab
                        else "trimmed to the demanded pairs")
                lines.append(f"| `{h['hold_id']}` | {lab['status']} | {src['frame_start'] / fps:.2f}-{src['frame_end'] / fps:.2f} "
                             f"-> {h['t_start']:.2f}-{h['t_end']:.2f} | {note} |")
    lines += ["", "## Configured contacts the plant-v2 reference does not realise (the gate masks them)", "",
              "| Hold | Contacts |", "|---|---|"]
    for c in result["clips"]:
        for h in c["holds"]:
            if h["labels"]["not_realised"]:
                lines.append(f"| `{h['hold_id']}` | {', '.join(h['labels']['not_realised'])} |")
    return "\n".join(lines) + "\n"


def write(result: dict, out_root: Path = LABELS_DIR) -> Path:
    lid = labels_id(result)
    out = Path(out_root) / lid
    out.mkdir(parents=True, exist_ok=True)
    head = {"schema_version": SCHEMA_VERSION, "labels_id": lid}
    (out / "annotations.jsonl").write_text("".join(json.dumps({**head, **a}) + "\n" for a in result["annotations"]))
    (out / "evidence.jsonl").write_text("".join(json.dumps(result["evidence"][k]) + "\n" for k in sorted(result["evidence"])))
    manifest = {k: v for k, v in result["manifest"].items() if k not in ("clips", "labels")}
    manifest["mjcf"] = str(fw.plant_paths(PLANT)[0].resolve())
    manifest["plant_sha256"] = plant_identity.sha256(PLANT)
    manifest["labels"] = {"labels_id": lid, "schema_version": SCHEMA_VERSION, "module": MODULE, "git_rev": ids.git_rev(),
                          "source_manifest": result["manifest"]["labels"]["source_manifest"],
                          "base_labels_id": result["v11"]["id"], "audit_id": result["audit_id"],
                          "statics_id": result["statics_id"], "retarget_id": result["rid"], "fit_id": FIT_ID,
                          "plant": plant_identity.identity(PLANT), "dropped": result["dropped"],
                          "annotations": "annotations.jsonl", "mjcf_v1_1": result["manifest"].get("mjcf"),
                          "hold_id": "names the source hold (its source exemplar); frame_hold is the corrected exemplar"}
    manifest["clips"] = result["clips"]
    (out / "holds.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))
    m = metrics(result)
    inputs = [result["v11"]["dir"] / "holds.yaml", result["v11"]["dir"] / "annotations.jsonl", *fw.plant_paths(PLANT)]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "labels_id": lid,
              "base_labels_id": result["v11"]["id"], "audit_id": result["audit_id"], "statics_id": result["statics_id"],
              "retarget_id": result["rid"], "plant": plant_identity.identity(PLANT), "config": CONFIG, "metrics": m,
              "critical": None if result["critical"] is None else {"pairs": sum(len(v) for v in result["critical"].values())},
              "notes": result["notes"]}
    (out / "labels.json").write_text(json.dumps(record, indent=1) + "\n")
    (out / "summary.md").write_text(summary_markdown(result, lid, m))
    return out


def verdict_row(verdict_id: str, render_v: str) -> dict:
    """``ledger:<packet>.B.<n>`` -> its evidence row (the ledger file's path and sha256)."""
    pid, _, n = verdict_id.split(":", 1)[1].rsplit(".", 2)
    path = verdicts.verdict_path(verdicts.LEDGER_DIR, pid, "B", int(n))
    return {"id": verdict_id, "kind": "verdict", "render_v": render_v, "path": ids.display_path(path),
            "sha256": ids.sha256_file(path), "packet_id": pid}


def load_votes(path: Path | None) -> tuple[dict, dict, list[dict]]:
    """``(votes, vote_verdicts, evidence rows)``: per hold the reviewer's variant answers and their verdict ids,
    from a votes JSON (``b6.py`` writes one), else from labels v1.1's single render_v3 Pass-B verdicts."""
    if path is not None:
        rec = json.loads(Path(path).read_text())
        return rec["votes"], rec.get("vote_verdicts", {}), rec.get("evidence", [])
    v11 = read_v11()
    votes, vids, rows = {}, {}, {}
    for c in v11["manifest"]["clips"]:
        for h in c["holds"]:
            r = h["labels"].get("review")
            if r:
                votes[h["hold_id"]] = [r["variant"]["matches_label"]]
                vids[h["hold_id"]] = [r["verdict"]]
                rows[r["verdict"]] = verdict_row(r["verdict"], "render_v3")
    return votes, vids, list(rows.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, default=LABELS_V11, help="labels v1.1 (the human-side decisions)")
    ap.add_argument("--audit", type=Path, help="audit v2 (calibrated) of those labels (default: the only one)")
    ap.add_argument("--statics", type=Path, help="statics v2 of those labels (default: the only one)")
    ap.add_argument("--critical", type=Path, help="B6's admitted critical contacts (b6.py), if any")
    ap.add_argument("--votes", type=Path, help="the reviewer's variant votes per hold (b6.py), default: labels v1.1's")
    ap.add_argument("--out-root", type=Path, default=LABELS_DIR)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        v11 = read_v11(args.labels)
        audit_dir = args.audit or default_audit_dir(v11["id"])
        statics_dir = args.statics or default_statics_dir(v11["id"])
        critical, extra = None, []
        if args.critical is not None:
            crit = json.loads(args.critical.read_text())
            critical = crit["critical"] if crit.get("admitted") else None
            extra = crit.get("evidence", [])
        votes, vote_ids, vote_rows = load_votes(args.votes)
        result = build(v11, audit_dir, statics_dir, critical=critical, votes=votes, vote_ids=vote_ids,
                       extra_evidence=extra + vote_rows)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    failures = result["failures"] + check(result)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures:
        print(f"labels_v2: {len(failures)} failures; nothing written", file=sys.stderr)
        return 1
    out = write(result, args.out_root)
    a = metrics(result)["all"]
    print(f"labels_v2 {out.name}: {a['holds']} holds {a['status']}; pose roles {a['pose_role']}; ground roles {a['ground_roles']} "
          f"(v1.1 {a['ground_roles_v1_1']}); windows trimmed {a['windows_trimmed_for_pairs']}; pairs configured "
          f"{a['pairs_configured']} (not realised {a['pairs_configured_not_realised']}); any_of {a['any_of_groups']}; "
          f"{len(result['annotations'])} annotations in {time.time() - start:.0f} s -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
