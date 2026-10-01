# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-hold release gate v2 (BodyFix Step 4, item 8): what of the plant-v2 corpus a release may use.

Gate v1 (``gate.py``) decided the Step 8 corpus. This decides labels v2's holds on the Step 3 references, from
pinned inputs only: the labels folder is named on the command line, and it names the rest (statics v2, the
retarget, the fit); nothing is read from "the newest folder". Every input's id or sha256 is in ``gate.json``.

Evidence
--------
* **labels v2**: status (``transition``), configured contacts, ``not_realised`` (configured contacts the plant-v2
  reference does not close), ``pose_role``, critical contacts (B6), roles;
* **statics v2** on the reference: verdict, ``realised_s_star`` of an unrealised support, and the plant-v2 statue's
  motion verdict (``still``: settle < 5 cm, drift < 2 cm);
* **Pass C on render_c2** (``edits_v2``): the reviewer's same-pose and closer-to-the-human verdicts, the classes its
  controls admit. A verdict counts only if its packet was built on the motions in use (the index entry's sha256 of
  the before and the after equal the files'), so a stale verdict is never read;
* **the motions** themselves, inside the hold window: the joint box (no coordinate more than 1 deg past plant v2's
  box), the floor contract (no surface below ``retarget_v2.FLOOR_CONTRACT_M``), the collision scan (body pairs the
  plant collides overlapping more than 1 cm where her fit does not: *new* overlaps, as Step 3's acceptance counts
  them, because every overlap would drop most binds), jerk (a body > 100 m/s^2 where her fit is not), and the edit
  against her fit (the lineage's per-body displacement, the retarget's budget).

Decisions (the first that applies decides; every reason is kept)
---------------------------------------------------------------
``exclude``  ``transition``; ``not_same_pose`` (Pass C, its ``pose_change`` class admitted by render_c2's controls;
             unadmitted it is the flag ``not_same_pose_unadmitted``); ``statics_beyond_plant``;
             ``statics_infeasible``; ``unholdable_without_<support>`` (a support the plant cannot realise, without
             which the LP needs more than the plant's torque); ``joint_box``, ``floor_contract``, ``new_overlap``
             (the motion breaks the plant's contract inside the window);
``mask``     ``unrealisable_<support>`` (the hold is held without it); ``not_realised_<pair>`` (a configured pair the
             reference does not close: the commensurate colliders' geometry, BodyFix Step 3); masked contacts leave
             every contact target;
``flag``     ``fit_closer`` (Pass C, ``closer`` admitted: the same pose, her unedited fit closer to her: expected
             where Step 3 had to edit), ``critical_not_realised`` (a B6-critical contact among the masks), ``critical_partly_realised``
             (a commanded critical contact the reference closes on less than 90 % of the frames the human makes it in
             the window: Step 5's per-frame masks), ``statue_moved_single_leg``
             (a single-leg hold whose plant-v2 statue settles or drifts: it caught 5 of the 6 single-leg holds
             ft_c failed), ``pose_preparation`` (the hold is a preparation, not the named pose), ``jerk_in_window``,
             ``over_edit_budget``;
``pass``     none of the above.

Outputs (``data/reference_curation/gate_v2/<labels stem>.gate_v2.<hash>/``; not ``gate/``, which gate v1's test reads): ``holds.jsonl``, ``clips.jsonl``,
``gate.json``, ``summary.md`` and ``release_holds.yaml``: labels v2's manifest without the excluded holds, each
hold's ``pairs`` / ``pairs_ground`` its *commanded* contacts (configured minus masks) and a ``gate`` block.
``check_commanded`` asserts TODO A4: every hold's commanded ground set equals labels v2's configured supports minus
the gate's masks.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.gate_v2 --labels <labels_v2 dir>
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from reference_curation import capture_v4, edits as E, edits_v2, fit_writer as fw, ids, statics_v2
from reference_curation import retarget as rt, retarget_v2 as rv2

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.gate_v2"
SCHEMA_VERSION = 1
GATE_VERSION = "v2"
GATE_DIR = ids.DATA_ROOT / "gate_v2"           # not gate/: gate v1's test reads the newest folder there
DECISIONS = ("pass", "mask", "flag", "exclude")
RULES = {
    "exclude": ["transition", "not_same_pose", "statics_beyond_plant", "statics_infeasible",
                "unholdable_without_<support>", "joint_box", "floor_contract", "new_overlap"],
    "mask": ["unrealisable_<support>", "not_realised_<pair>"],
    "flag": ["fit_closer", "critical_not_realised", "critical_partly_realised", "statue_moved_single_leg",
             "pose_preparation", "jerk_in_window", "over_edit_budget", "not_same_pose_unadmitted"],
}
PARTLY_REALISED = 0.9      # a critical contact the reference closes on less than this share of the human's contact frames
BOX_TOL_DEG = rv2.BOX_TOL_DEG
FLOOR_CONTRACT_M = rv2.FLOOR_CONTRACT_M
DEEP_OVERLAP_M = rv2.DEEP_OVERLAP_M
SPIKE_ACC = rv2.SPIKE_ACC


def read_labels(labels_dir: Path) -> dict:
    m = ids.load_manifest(Path(labels_dir) / "holds.yaml")
    lab = m["labels"]
    if lab.get("module") != "reference_curation.labels_v2":
        raise ValueError(f"{labels_dir} is not a labels v2 folder")
    ids.require_plant(lab, f"labels {lab['labels_id']}", fw.plant_paths("v2")[0])
    anns = collections.defaultdict(list)
    for line in open(Path(labels_dir) / "annotations.jsonl"):
        a = json.loads(line)
        anns[a["hold_id"]].append(a)
    return {"dir": Path(labels_dir), "id": lab["labels_id"], "manifest": m, "anns": dict(anns),
            "statics_id": lab["statics_id"], "retarget_id": lab["retarget_id"], "fit_id": lab["fit_id"]}


# --------------------------------------------------------------------------- #
# Pass C on render_c2, current verdicts only
# --------------------------------------------------------------------------- #
def pass_c_by_hold(labels: dict) -> tuple[dict, list[str]]:
    """``{hold_id: accept row}`` from render_c2's edit verdicts (the lowest valid ``n`` per packet), only where the
    packet was built on the before and after in use; ``stale`` lists the rest."""
    from reference_curation import verdicts

    index = edits_v2.read_index()
    if not index:
        raise FileNotFoundError(f"no Pass C index at {edits_v2.index_path()}")
    fit_dir = fw.OUT_ROOT / labels["fit_id"]
    ref_dir = rv2.OUT_ROOT / labels["retarget_id"]
    by_pid, stale = {}, []
    for e in index.values():
        if e["kind"] != "edit":
            continue
        cur = {"before": ids.sha256_file(ids.motion_path(e["stem"], fit_dir)),
               "after": ids.sha256_file(ids.motion_path(e["stem"], ref_dir))}
        if e.get("motions") != cur:
            stale.append(e["hold_id"])
            continue
        by_pid[e["packet_id"]] = e
    chosen = {}
    for r in verdicts.read_ledger():
        if r["pass"] == E.PASS and r["status"] == "valid" and r["packet_id"] in by_pid:
            chosen.setdefault(r["packet_id"], r)
    out = {}
    for pid, r in chosen.items():
        e = by_pid[pid]
        row = E.accept(r["answer"], e)
        row["closer"] = "fit" if row["closer"] == "shipped" else row["closer"]
        row["more_natural"] = "fit" if row["more_natural"] == "shipped" else row["more_natural"]
        row["artefacts"] = [[a["avatar"] == E.modified_side(e) and "edit" or "fit", a["kind"], a["part"], a["severity"]]
                            for a in r["answer"]["artefacts"]]
        row["verdict_id"] = f"ledger:{pid}.C.{r['n']}"
        out[e["hold_id"]] = row
    return out, stale


def pass_c_admitted() -> dict:
    """``{class: admitted}`` from render_c2's control table (``edits_v2 --calibrate``): a Pass C claim decides only
    if its class is evidence there (``pose_change`` for ``not_same_pose``, ``closer`` for ``fit_closer``)."""
    path = E.CALIBRATION_DIR / f"{edits_v2.RENDER_V}.json"
    t = json.loads(path.read_text())
    return {k: bool(v["evidence"]) for k, v in t["classes"].items()}


# --------------------------------------------------------------------------- #
# Per-frame measures of a clip (the motion against her fit)
# --------------------------------------------------------------------------- #
def frame_measures(stem: str, labels: dict) -> dict:
    """``{"box": [T] max deg past the box, "floor": [T] lowest surface m, "overlap": [T] new deep overlaps,
    "jerk": [T] bool, "disp": [T, B] edit m}``: the reference against her fit on plant v2."""
    from reference_curation import mosh_replay as mr

    sk = fw.skeleton("v2")
    ref = torch.load(ids.motion_path(stem, rv2.OUT_ROOT / labels["retarget_id"]), map_location="cpu", weights_only=False)
    fit = torch.load(ids.motion_path(stem, fw.OUT_ROOT / labels["fit_id"]), map_location="cpu", weights_only=False)
    out, pairs = {}, {}
    for tag, m in (("ref", ref), ("fit", fit)):
        pos, rot = rv2.motion_state(m)
        a = rv2.body_acc(pos, int(m["fps"]))
        P, R = torch.as_tensor(pos), torch.as_tensor(rot)
        f, ba, bb, g = rt.near_body_pairs(sk, P, R, -DEEP_OVERLAP_M, mr.colliding_pairs(sk))
        pairs[tag] = set(zip(f.tolist(), ba.tolist(), bb.tolist()))
        out[tag] = {"pos": pos, "rot": rot, "acc": a}
    T = ref["rigid_body_pos"].shape[0]
    overlap = np.zeros(T, int)
    for f, _, _ in pairs["ref"] - pairs["fit"]:
        overlap[f] += 1
    low = rt.candidate_heights(sk, rt.candidate_points(sk, torch.as_tensor(out["ref"]["pos"]),
                                                       torch.as_tensor(out["ref"]["rot"]))).numpy().min(1)
    return {"box": fw.box_excess_deg(ref["dof_pos"].double().numpy()).max(1), "floor": low, "overlap": overlap,
            "jerk": (out["ref"]["acc"] > SPIKE_ACC) & (out["fit"]["acc"] <= SPIKE_ACC),
            "disp": np.linalg.norm(out["ref"]["pos"] - out["fit"]["pos"], axis=-1)}


# --------------------------------------------------------------------------- #
# One hold
# --------------------------------------------------------------------------- #
def single_leg(clip: dict, hold: dict) -> bool:
    ground = {p[:-2] for p in hold["pairs_ground"]}
    return clip.get("group") == "single_leg" and len(ground & {"L_FOOT", "R_FOOT"}) == 1


def decide(clip: dict, hold: dict, anns: list[dict], srow: dict | None, passc: dict | None, window: dict,
           admitted: dict | None = None) -> dict:
    """The decision of one hold (module docstring). ``admitted``: ``pass_c_admitted()``."""
    admitted = admitted or {}
    lab = hold["labels"]
    reasons = {"exclude": [], "mask": [], "flag": []}
    masks = []
    if lab["status"] == "transition":
        reasons["exclude"].append("transition")
    if passc is not None and passc["same_pose"] == "no":
        reasons["exclude" if admitted.get("pose_change") else "flag"].append(
            "not_same_pose" if admitted.get("pose_change") else "not_same_pose_unadmitted")
    verdict = (srow or {}).get("verdict")
    if verdict in ("beyond_plant", "infeasible"):
        reasons["exclude"].append(f"statics_{verdict}")
    if verdict == "support_not_realised":
        g = srow["gated"]
        missing = sorted(g["unrealised"])
        if g.get("realised_s_star") is None or g["realised_s_star"] > 1.0:
            reasons["exclude"] += [f"unholdable_without_{m}" for m in missing]
        else:
            reasons["mask"] += [f"unrealisable_{m}" for m in missing]
            masks += missing
    if window["box_max_deg"] > BOX_TOL_DEG:
        reasons["exclude"].append("joint_box")
    if window["floor_min_cm"] < 100 * FLOOR_CONTRACT_M:
        reasons["exclude"].append("floor_contract")
    if window["new_overlap_pair_frames"]:
        reasons["exclude"].append("new_overlap")
    for c in lab["not_realised"]:
        if c not in masks:
            reasons["mask"].append(f"unrealisable_{c}" if c.endswith(":G") else f"not_realised_{c}")
            masks.append(c)
    crit = {a["contact"] for a in anns if a["kind"] == "pair" and a.get("critical")}
    if crit & set(masks):
        reasons["flag"].append("critical_not_realised")
    partly = [a["contact"] for a in anns if a["contact"] in crit and a["contact"] not in masks
              and (a["evidence"]["avatar"].get("realised_on_human_contact") or 0.0) < PARTLY_REALISED]
    if partly:
        reasons["flag"].append("critical_partly_realised")
    if passc is not None and passc["same_pose"] == "yes" and passc["closer"] == "fit" and admitted.get("closer"):
        reasons["flag"].append("fit_closer")
    wit = (srow or {}).get("witness") or {}
    if single_leg(clip, hold) and lab["status"] != "transition" and not wit.get("still", False):
        reasons["flag"].append("statue_moved_single_leg")
    if lab.get("pose_role") == "preparation":
        reasons["flag"].append("pose_preparation")
    if window["jerk_frames"]:
        reasons["flag"].append("jerk_in_window")
    if window["edit_mean_p95_cm"] > 100 * rt.EDIT_MEAN_P95_M or window["edit_body_max_cm"] > 100 * rt.EDIT_BODY_MAX_M:
        reasons["flag"].append("over_edit_budget")
    decision = next((d for d in ("exclude", "mask", "flag") if reasons[d]), "pass")
    configured = list(hold["pairs"])
    commanded = [c for c in configured if c not in masks]
    return {"decision": decision, "reasons": reasons, "masks": masks, "commanded": commanded,
            "critical": sorted(crit), "pose_role": lab.get("pose_role"),
            "evidence": {"labels_status": lab["status"], "statics_verdict": verdict,
                         "s_star": (srow or {}).get("gated", {}).get("s_star"),
                         "realised_s_star": (srow or {}).get("gated", {}).get("realised_s_star"),
                         "statue": None if not wit else {k: wit.get(k) for k in ("passed", "passed_strict", "still",
                                                                                 "settle_cm", "drift_cm")},
                         "pass_c": None if passc is None else {k: passc[k] for k in ("accepted", "same_pose", "closer",
                                                                                     "more_natural", "artefacts",
                                                                                     "verdict_id")},
                         **window}}


def run(labels_dir: Path) -> dict:
    labels = read_labels(labels_dir)
    st = statics_v2.load(statics_v2.STATICS_DIR / labels["statics_id"])
    passc, stale = pass_c_by_hold(labels)
    admitted = pass_c_admitted()
    holds, clips = [], []
    for clip in labels["manifest"]["clips"]:
        stem = clip["stem"]
        fm = frame_measures(stem, labels)
        inside = np.zeros(len(fm["jerk"]), bool)
        for h in clip["holds"]:
            f0, f1 = int(h["frame_start"]), int(h["frame_end"]) + 1
            inside[f0:f1] = True
            d = fm["disp"][f0:f1]
            window = {"window": [f0, f1 - 1], "box_max_deg": round(float(fm["box"][f0:f1].max()), 3),
                      "floor_min_cm": round(100 * float(fm["floor"][f0:f1].min()), 3),
                      "new_overlap_pair_frames": int(fm["overlap"][f0:f1].sum()), "jerk_frames": int(fm["jerk"][f0:f1].sum()),
                      "edit_mean_p95_cm": round(100 * float(np.percentile(d.mean(1), 95)), 2),
                      "edit_body_max_cm": round(100 * float(d.max()), 2)}
            row = decide(clip, h, labels["anns"][h["hold_id"]], st["holds"].get(h["hold_id"]), passc.get(h["hold_id"]), window,
                         admitted)
            holds.append({"hold_id": h["hold_id"], "stem": stem, "name": h["name"], "family_hold": bool(h.get("extend")), **row})
        clips.append({"stem": stem, "frames": int(len(inside)),
                      "transition_jerk_frames": int(fm["jerk"][~inside].sum()),
                      "transition_new_overlap_pair_frames": int(fm["overlap"][~inside].sum()),
                      "box_max_deg": round(float(fm["box"].max()), 3), "floor_min_cm": round(100 * float(fm["floor"].min()), 3),
                      "decisions": dict(collections.Counter(r["decision"] for r in holds if r["stem"] == stem))})
    missing = [r["hold_id"] for r in holds if r["evidence"]["pass_c"] is None and r["evidence"]["labels_status"] != "transition"]
    return {"labels": labels, "statics": st, "holds": holds, "clips": clips, "pass_c_stale": stale,
            "pass_c_missing": missing, "pass_c_admitted": admitted}


# --------------------------------------------------------------------------- #
# The release manifest and TODO A4
# --------------------------------------------------------------------------- #
def release_manifest(result: dict) -> dict:
    """labels v2's manifest with the excluded holds dropped and every kept hold's contacts its commanded ones."""
    m = copy.deepcopy(result["labels"]["manifest"])
    by = {r["hold_id"]: r for r in result["holds"]}
    clips = []
    for c in m["clips"]:
        kept = []
        for h in c["holds"]:
            r = by[h["hold_id"]]
            if r["decision"] == "exclude":
                continue
            h["pairs_configured"] = list(h["pairs"])
            h["pairs"] = list(r["commanded"])
            h["pairs_ground"] = [p for p in r["commanded"] if p.endswith(":G")]
            h["gate"] = {"decision": r["decision"], "masks": r["masks"], "flags": r["reasons"]["flag"]}
            kept.append(h)
        if kept:
            clips.append({**c, "holds": kept})
    m["clips"] = clips
    return m


def check_commanded(result: dict, release: dict) -> list[str]:
    """TODO A4: every released hold commands exactly labels v2's configured ground supports minus the gate's masks
    (and its configured pairs minus the masked pairs); every excluded hold is gone."""
    problems = []
    labels_holds = {h["hold_id"]: h for c in result["labels"]["manifest"]["clips"] for h in c["holds"]}
    by = {r["hold_id"]: r for r in result["holds"]}
    released = {h["hold_id"]: h for c in release["clips"] for h in c["holds"]}
    for hid, r in by.items():
        if r["decision"] == "exclude":
            if hid in released:
                problems.append(f"{hid}: excluded but released")
            continue
        h = released.get(hid)
        if h is None:
            problems.append(f"{hid}: kept but not released")
            continue
        conf = labels_holds[hid]
        want_ground = [p for p in conf["pairs_ground"] if p not in r["masks"]]
        want_pairs = [p for p in conf["pairs"] if p not in r["masks"]]
        if h["pairs_ground"] != want_ground:
            problems.append(f"{hid}: commanded ground {h['pairs_ground']} != configured minus masks {want_ground}")
        if h["pairs"] != want_pairs:
            problems.append(f"{hid}: commanded contacts {h['pairs']} != configured minus masks {want_pairs}")
    return problems


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def reason_kind(reason: str) -> str:
    """The rule a reason instantiates (``unrealisable_R_HAND:G`` -> ``unrealisable_<support>``)."""
    for rules in RULES.values():
        for rule in rules:
            if "<" in rule and reason.startswith(rule.split("<")[0]):
                return rule
    return reason


def counts(result: dict) -> dict:
    hs = result["holds"]
    reasons = collections.Counter(x for r in hs for d in ("exclude", "mask", "flag") for x in r["reasons"][d])
    kinds = collections.Counter(reason_kind(x) for r in hs for d in ("exclude", "mask", "flag") for x in r["reasons"][d])
    return {"holds": len(hs), "decisions": {d: sum(r["decision"] == d for r in hs) for d in DECISIONS},
            "family_decisions": {d: sum(r["decision"] == d and r["family_hold"] for r in hs) for d in DECISIONS},
            "reasons": dict(sorted(reasons.items())), "reason_kinds": dict(sorted(kinds.items())),
            "masks": dict(collections.Counter(m for r in hs if r["decision"] != "exclude" for m in r["masks"])),
            "clips_all_excluded": sorted(c["stem"] for c in result["clips"] if set(c["decisions"]) == {"exclude"}),
            "transition_jerk_frames": sum(c["transition_jerk_frames"] for c in result["clips"]),
            "transition_new_overlap_pair_frames": sum(c["transition_new_overlap_pair_frames"] for c in result["clips"]),
            "pass_c_stale": len(result["pass_c_stale"]), "pass_c_missing": len(result["pass_c_missing"])}


def gate_id(result: dict) -> str:
    key = {"schema": SCHEMA_VERSION, "labels": result["labels"]["id"], "rules": RULES, "generator": ids.sha256_file(__file__),
           "statics": result["labels"]["statics_id"], "retarget": result["labels"]["retarget_id"],
           "pass_c": sorted((r["hold_id"], (r["evidence"]["pass_c"] or {}).get("verdict_id")) for r in result["holds"])}
    stem = Path(result["labels"]["manifest"]["labels"]["source_manifest"]).stem
    return f"{stem}.gate_{GATE_VERSION}.{ids.sha256_json(key)[:10]}"


def summary_markdown(gid: str, c: dict, result: dict) -> str:
    lab = result["labels"]
    lines = [f"# Gate v2 `{gid}`", "", f"Generated by `{MODULE}` (BodyFix Step 4, item 8) from labels `{lab['id']}`, statics "
             f"`{lab['statics_id']}`, references `{lab['retarget_id']}` and her fit `{lab['fit_id']}` (pinned), with Pass C "
             "on render_c2. The rules are in the module docstring.", "",
             "| Decision | Holds | Family holds |", "|---|---|---|"]
    lines += [f"| `{d}` | {c['decisions'][d]} | {c['family_decisions'][d]} |" for d in DECISIONS]
    lines += ["", "| Reason | Holds |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in c["reason_kinds"].items()]
    lines += ["", f"Masked contacts on released holds: {c['masks'] or 'none'}.", "",
              f"Outside every hold window: {c['transition_jerk_frames']} jerk frames and {c['transition_new_overlap_pair_frames']} "
              "new overlap pair-frames.", "",
              f"Pass C verdicts on stale motions: {c['pass_c_stale']}; holds without a current Pass C verdict: {c['pass_c_missing']}.",
              "", "## Excluded and masked holds", "", "| Hold | Decision | Reasons |", "|---|---|---|"]
    for r in result["holds"]:
        if r["decision"] in ("exclude", "mask"):
            why = "; ".join(x for d in ("exclude", "mask", "flag") for x in r["reasons"][d])
            lines.append(f"| `{r['hold_id']}` | {r['decision']} | {why} |")
    return "\n".join(lines) + "\n"


def write(result: dict, out_root: Path = GATE_DIR) -> Path:
    gid = gate_id(result)
    out = Path(out_root) / gid
    out.mkdir(parents=True, exist_ok=True)
    head = {"schema_version": SCHEMA_VERSION, "gate_id": gid}
    (out / "holds.jsonl").write_text("".join(json.dumps({**head, **r}) + "\n" for r in result["holds"]))
    (out / "clips.jsonl").write_text("".join(json.dumps({**head, **r}) + "\n" for r in result["clips"]))
    release = release_manifest(result)
    release["labels"] = {**release["labels"], "gate_id": gid, "release_manifest": "commanded contacts: configured minus "
                         "the gate's masks; excluded holds dropped"}
    (out / "release_holds.yaml").write_text(yaml.safe_dump(release, sort_keys=False, width=120))
    c = counts(result)
    lab = result["labels"]
    inputs = [lab["dir"] / "holds.yaml", lab["dir"] / "annotations.jsonl",
              statics_v2.STATICS_DIR / lab["statics_id"] / "holds.jsonl",
              rv2.RECORD_ROOT / lab["retarget_id"] / "retarget.json", edits_v2.index_path(),
              E.CALIBRATION_DIR / f"{edits_v2.RENDER_V}.json", *fw.plant_paths("v2")]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "gate_id": gid, "labels_id": lab["id"],
              "statics_id": lab["statics_id"], "retarget_id": lab["retarget_id"], "fit_id": lab["fit_id"],
              "pass_c_render_v": edits_v2.RENDER_V, "plant": plant_identity.identity("v2"), "rules": RULES,
              "thresholds": {"box_tol_deg": BOX_TOL_DEG, "floor_contract_m": FLOOR_CONTRACT_M,
                             "deep_overlap_m": DEEP_OVERLAP_M, "spike_acc": SPIKE_ACC,
                             "edit_mean_p95_m": rt.EDIT_MEAN_P95_M, "edit_body_max_m": rt.EDIT_BODY_MAX_M},
              "counts": c, "pass_c_admitted": result["pass_c_admitted"], "pass_c_stale": result["pass_c_stale"],
              "pass_c_missing": result["pass_c_missing"],
              "a4_check": check_commanded(result, release)}
    (out / "gate.json").write_text(json.dumps(record, indent=1) + "\n")
    (out / "summary.md").write_text(summary_markdown(gid, c, result))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, required=True, help="the labels v2 folder (pinned, never the newest)")
    ap.add_argument("--out-root", type=Path, default=GATE_DIR)
    args = ap.parse_args(argv)
    try:
        result = run(args.labels)
        if result["pass_c_missing"]:
            raise ValueError(f"{len(result['pass_c_missing'])} holds have no current Pass C verdict, e.g. "
                             f"{result['pass_c_missing'][:3]}")
        problems = check_commanded(result, release_manifest(result))
        if problems:
            raise ValueError(f"A4: {problems[:3]}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = write(result, args.out_root)
    c = counts(result)
    print(f"gate_v2: {c['holds']} holds {c['decisions']} (family {c['family_decisions']}); masks {sum(c['masks'].values())}; "
          f"A4 check passed -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
