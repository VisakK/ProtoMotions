# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit v2 (BodyFix Step 4, item 2): every hold of labels v1.1's manifest checked against capture store v4, so
the float, missed-touch and phantom metrics speak for the Step 3 references on plant v2.

It is Step 2's audit (``audit.audit_hold``, ``audit.build_queue``, ``audit.metrics``, unchanged) with three
differences:

* **The manifest is labels v1.1's** ``holds.yaml``: the corrected ground sets, windows and exemplars, not the source
  manifest's. Holds of the clips the writer drops (the lotus clips, Standing big toe hold -c, Firefly -b) are listed
  as dropped, not audited.
* **The store is capture v4** (``capture_v4.load``): the human side unchanged, the avatar side and the mat's
  attribution measured on plant v2 and the ``retarget_v2`` motions.
* **Hold ids are the labels'.** ``audit.audit_hold`` names a hold by the manifest's ``frame_hold``; in labels v1.1
  that is the *corrected* exemplar, while ``hold_id`` names the source hold, and the labels keep it. The record
  takes the labels' ``hold_id``.

Two rules. ``calibrated`` is Step 2's: the markers' Schmitt trigger on the feet, hands and head (the human truth
``verdicts.packet_truth`` scores the reviewer against, so the Pass-A calibration queue comes from it). ``sources``
reads the labels' own evidence hierarchy (``sources.ground_source``: the seam-arbitrated mesh on every zone, then
the markers, then the mat), so its missed touches and phantoms are the labels' disagreements with the capture.

Beside every audit the same holds are audited on capture store v1 (the shipped references on plant v1:
``baseline_v1``), so the two are compared hold for hold.

Output: ``data/reference_curation/audits/<labels_id>.audit_v2.<rule>.<hash>/`` (``holds.jsonl``, ``queue.jsonl``,
``summary.md``, ``audit.json``), the layout ``packets.load_audit`` reads.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.audit_v2 [--rule sources]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, capture, capture_v4, fit_writer as fw, human_mesh as hm, ids, sources

MODULE = "reference_curation.audit_v2"
SCHEMA_VERSION = 1
AUDIT_VERSION = "audit_v2"
LABELS_V11 = ids.DATA_ROOT / "labels" / "holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828"
RULES = ("calibrated", "sources")


def make_rule(name: str) -> SimpleNamespace:
    """``calibrated``: Step 2's (capture v1's admitted zones and bands). ``sources``: every zone, the mesh's bands."""
    if name == "calibrated":
        return audit.make_rule("calibrated", capture.load_calibration())
    if name == "sources":
        g = hm.load_calibration()["ground"]
        band = (g["touch_m"], g["separation_m"])
        return SimpleNamespace(name=name, zones=tuple(ZONE_ORDER), bands={z: band for z in ZONE_ORDER},
                               config={"source": "sources.ground_source", "bands_m": band,
                                       "hierarchy": sources.CONFIG["hierarchy"]})
    raise ValueError(f"unknown rule {name!r}; expected one of {RULES}")


def clip_evidence(stem: str, rec: capture.Capture, rule: SimpleNamespace) -> SimpleNamespace:
    if rule.name == "calibrated":
        return audit.clip_evidence(stem, rec, rule)
    states, _ = sources.ground_source(rec)
    heights = np.asarray(rec["human_floor_z"], np.float64)
    return SimpleNamespace(states=states, heights=heights, exemplar=lambda f: (states[f], heights[f]))


def audit_records(manifest: dict, stores: dict, rule: str) -> tuple[list[dict], list[str]]:
    """One record per hold of ``manifest``'s clips found in ``stores`` (stem -> Capture), under ``rule``."""
    r = make_rule(rule)
    out, failures = [], []
    for clip in manifest["clips"]:
        stem = clip["stem"]
        if stem not in stores:
            continue
        rec = stores[stem]
        try:
            if rec.meta["num_frames"] != int(clip["num_frames"]) or rec.fps != int(clip["fps"]):
                raise ValueError(f"store has {rec.meta['num_frames']} frames at {rec.fps} fps, the manifest "
                                 f"{clip['num_frames']} at {clip['fps']}")
            ev = clip_evidence(stem, rec, r)
            for i, h in enumerate(clip["holds"]):
                row = audit.audit_hold(clip, i, h, rec, ev, r)
                row["exemplar_hold_id"] = row["hold_id"]
                row["hold_id"] = h["hold_id"]        # the labels' stable id: the source hold's
                out.append(row)
        except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
            failures.append(f"{stem}: {type(exc).__name__}: {exc}")
    counts = collections.Counter(x["hold_id"] for x in out)
    failures += [f"duplicate hold_id {h}" for h, n in sorted(counts.items()) if n > 1]
    return out, failures


def load_stores(stems: list[str]) -> tuple[dict, dict]:
    """``(v4, v1)``: capture store v4 (plant v2) and the v3 record whose avatar side is capture v1's (plant v1, the
    shipped references) for every stem; read outside plant v2."""
    capture_v4.require_v1_process()
    v4, v1 = {}, {}
    for s in stems:
        base = capture_v4.human_base(s)
        v4[s] = capture_v4.load(s, base=base, rebuild=False)
        v1[s] = capture.Capture(capture.load(s).meta, dict(base.arrays))   # v1's meta keys, v3's arrays
    return v4, v1


def audit_id(labels_id: str, rule: str, manifest_path: Path, stores: dict) -> str:
    key = {"schema": SCHEMA_VERSION, "generator": ids.sha256_file(__file__), "audit": ids.sha256_file(audit.__file__),
           "rule": make_rule(rule).config, "config": audit.CONFIG, "manifest": ids.sha256_file(manifest_path),
           "stores": {s: capture_v4.identity(r) for s, r in sorted(stores.items())}}
    return f"{labels_id}.{AUDIT_VERSION}.{rule}.{ids.sha256_json(key)[:10]}"


def compare(v2: list[dict], v1: list[dict]) -> dict:
    """Hold-for-hold changes of the avatar-side reasons between the shipped references (v1) and plant v2."""
    by1 = {r["hold_id"]: r for r in v1}
    out = collections.Counter()
    for r in v2:
        o = by1[r["hold_id"]]
        for code in ("hover", "unexplained_load"):
            a, b = any(x["code"] == code for x in o["reasons"]), any(x["code"] == code for x in r["reasons"])
            out[f"{code}:{'kept' if a and b else 'fixed' if a else 'new' if b else 'absent'}"] += 1
    return dict(sorted(out.items()))


def summary_markdown(records: list[dict], context: dict, m: dict, m1: dict, queue: list[dict], cmp: dict) -> str:
    text = audit.summary_markdown(records, context, m, queue)
    head = (f"Manifest `{ids.display_path(context['manifest'])}` (labels v1.1), rule `{context['rule']}`, capture store "
            f"`v4` (plant v2, `{capture_v4.RETARGET_ID}`), git `{(ids.git_rev() or '-')[:10]}`. Generated by "
            f"`{MODULE}` (BodyFix Step 4) with `reference_curation.audit`'s rules, which are in its docstring.")
    lines = text.splitlines()
    lines[2] = head
    a, b = m["all"], m1["all"]
    extra = ["", "## Against the shipped references (capture store v1, plant v1, the same holds and rule)", "",
             "| Metric | Shipped (v1) | Step 3 (plant v2) |", "|---|---|---|"]
    for c in audit.HOVER_REPORT_CM:
        extra.append(f"| Labelled supports hovering > {c} cm (of {a['labelled_supports']}) | {b['hover_gt_cm'][str(c)]} | "
                     f"{a['hover_gt_cm'][str(c)]} |")
    for code in ("missed_touch", "phantom_support", "hover", "unexplained_load", "boundary_start", "boundary_end",
                 "exemplar_unstable", "no_capture_match"):
        extra.append(f"| `{code}` (holds) | {b[code]['holds']} | {a[code]['holds']} |")
    extra += ["", "Hold for hold: " + ", ".join(f"{k} {v}" for k, v in cmp.items()) + ".", "",
              f"Dropped clips (not audited): {', '.join(f'`{s}`' for s in context['dropped'])}.", ""]
    at = lines.index("## Queue")
    return "\n".join(lines[:at] + extra[1:] + lines[at:]) + "\n"


def write(records: list[dict], records_v1: list[dict], context: dict, out_root: Path = audit.AUDITS_DIR) -> Path:
    out = Path(out_root) / context["audit_id"]
    out.mkdir(parents=True, exist_ok=True)
    queue, m, m1 = audit.build_queue(records), audit.metrics(records), audit.metrics(records_v1)
    cmp = compare(records, records_v1)
    head = {"schema_version": SCHEMA_VERSION, "audit_id": context["audit_id"]}
    for name, rows in (("holds.jsonl", records), ("queue.jsonl", queue)):
        (out / name).write_text("".join(json.dumps({**head, **row}) + "\n" for row in rows))
    (out / "summary.md").write_text(summary_markdown(records, context, m, m1, queue, cmp))
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, context["inputs"]), "audit_id": context["audit_id"],
              "manifest": ids.display_path(context["manifest"]), "labels_id": context["labels_id"], "rule": context["rule"],
              "rule_config": context["rule_config"], "store": "capture/v4", "retarget_id": capture_v4.RETARGET_ID,
              "plant": capture_v4.plant_identity.identity(capture_v4.PLANT), "config": audit.CONFIG, "metrics": m,
              "baseline_v1": {"store": "capture/v1", "metrics": m1}, "compare_v1": cmp, "dropped": context["dropped"],
              "queue": {p: sum(q["pass"] == p for q in queue) for p in ("A", "B")}}
    (out / "audit.json").write_text(json.dumps(record, indent=1) + "\n")
    return out


def run(rule: str = "calibrated", labels_dir: Path = LABELS_V11) -> tuple[list[dict], list[dict], dict, list[str]]:
    manifest_path = Path(labels_dir) / "holds.yaml"
    manifest = ids.load_manifest(manifest_path)
    corpus, dropped = fw.corpus()
    v4, v1 = load_stores(corpus)
    records, failures = audit_records(manifest, v4, rule)
    records_v1, f1 = audit_records(manifest, v1, rule)
    failures += [f"(v1 baseline) {f}" for f in f1]
    lid = manifest["labels"]["labels_id"]
    inputs = [manifest_path, capture.CALIBRATION_PATH, hm.CALIBRATION_PATH]
    inputs += [capture_v4.STORE_DIR / f"{s}.json" for s in corpus]
    context = {"audit_id": audit_id(lid, rule, manifest_path, v4), "manifest": manifest_path, "labels_id": lid,
               "rule": rule, "rule_config": make_rule(rule).config, "calibration_id": capture.load_calibration()["id"],
               "inputs": inputs, "dropped": sorted(dropped)}
    return records, records_v1, context, failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rule", choices=RULES, default="calibrated")
    ap.add_argument("--labels", type=Path, default=LABELS_V11)
    ap.add_argument("--out-root", type=Path, default=audit.AUDITS_DIR)
    args = ap.parse_args(argv)
    try:
        records, records_v1, context, failures = run(args.rule, args.labels)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures:
        print(f"audit_v2: {len(failures)} failures; nothing written", file=sys.stderr)
        return 1
    out = write(records, records_v1, context, args.out_root)
    a, b = audit.metrics(records)["all"], audit.metrics(records_v1)["all"]
    print(f"audit_v2 {context['audit_id']}: {a['holds']} holds; hover > 2 cm {b['hover_gt_cm']['2']} -> "
          f"{a['hover_gt_cm']['2']} of {a['labelled_supports']} (shipped -> plant v2); missed touch {a['missed_touch']['zones']} "
          f"in {a['missed_touch']['holds']} holds, phantom {a['phantom_support']['zones']} in {a['phantom_support']['holds']}; "
          f"flagged {sum(bool(r['reasons']) for r in records)} (high {a['severity']['high']}, medium {a['severity']['medium']}, "
          f"low {a['severity']['low']}) -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
