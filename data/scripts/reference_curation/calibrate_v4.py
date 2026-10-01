# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Pass-A recalibration on render_v4 (BodyFix Step 4, item 3): which of the blind reviewer's claim classes are
evidence about plant v2's avatar.

render_v3's admissions (float detection, avatar and human floor contact, left/right) were measured on the shipped
avatar. Plant v2 changed the body, the head sphere and the colours, so they are measured again, with Step 4's rule
(``verdicts.class_metrics``: precision >= 0.9 on >= 30 of a class's own claims, answered rate >= 0.5 over >= 30
items), on three groups of blind packets (``packets_v4``), every truth by machine:

* ``reference`` (48): render_v3's calibration holds that are still in the corpus, on the Step 3 references at
  labels v1.1's exemplar (audit v2's records): the same poses, so the table compares with render_v3's;
* ``fit`` (17): her unedited fit on plant v2 at the Pass-A candidates where it floats a support 2-15 cm (a
  supinated foot, a tilted hand: natural floats of 2-3 cm);
* ``lift`` (24): reference packets with the avatar raised by a known 2.5-12 cm (``packets_v4.LIFTS_M``, by hash),
  so that the float class has claims across sizes: the plant-v2 references float on 1 of 864 supports.

The verdicts go to the shared ledger (``<packet_id>.A.<n>.json``) with the Step 4 reviewer (``review.py``, high
effort, blind). The table (``data/reference_curation/calibration/pass_a_v4/render_v4.json``, a subdirectory because
Step 4's replay test globs ``calibration/render_v*.json``) holds every class per variant group and pooled; it is
rebuilt from the ledger, never by re-querying (``--calibrate``).

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.calibrate_v4 --build
    ... --review --max-cost 15
    ... --calibrate
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from reference_curation import packets_v4  # first: render (OSMesa)
from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, capture, capture_v4, fit_writer as fw, ids, packets, review, verdicts

MODULE = "reference_curation.calibrate_v4"
SCHEMA_VERSION = 1
RENDER_V = packets_v4.RENDER_V
CALIBRATION_DIR = verdicts.CALIBRATION_DIR / "pass_a_v4"
LABELS_V11 = ids.DATA_ROOT / "labels" / "holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828"
N_LIFT = 24
FLOAT_RANGE_M = (0.02, 0.15)


def audit_dir() -> Path:
    found = sorted(audit.AUDITS_DIR.glob("*.labels_v1_1.*.audit_v2.calibrated.*"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one calibrated audit v2, found {len(found)}")
    return found[0]


def v3_calibration_holds() -> list[str]:
    """render_v3's Pass-A calibration holds (one valid verdict each in the ledger), restricted to the corpus."""
    corpus, _ = fw.corpus()
    idx = packets.read_index(packets.index_path("render_v3"))
    used = {r["packet_id"] for r in verdicts.read_ledger() if r["pass"] == "A" and r["render_v"] == "render_v3"
            and r["status"] == "valid"}
    return sorted({e["hold_id"] for e in idx.values() if e["pass"] == "A" and e["packet_id"] in used
                   and e["stem"] in corpus})


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def fit_float_holds(records: dict, queue: list[dict]) -> list[str]:
    """Pass-A candidates whose unedited fit floats a support the human touches by 2-15 cm at the exemplar."""
    out = []
    for q in queue:
        if q["pass"] != "A":
            continue
        r = records[q["hold_id"]]
        z, _ = packets_v4._variant_geometry(r["stem"], "fit", 0.0)
        rec = capture_v4.load(r["stem"], rebuild=False)
        f = r["window"]["frame_hold"]
        if any(rec["ground_state"][f, ZONE_ORDER.index(zn)] == 1 and FLOAT_RANGE_M[0] <= z[f, ZONE_ORDER.index(zn)]
               <= FLOAT_RANGE_M[1] for zn in r["capture"]["decided"]):
            out.append(q["hold_id"])
    return sorted(set(out))


def sided_holds(records: dict, queue: list[dict], exclude: set, n: int) -> list[str]:
    """Pass-A candidates (truth unambiguous) whose two feet or two hands differ at the exemplar in the human's
    decisive state, by hash: the left/right class's items (render_v3 rested its admission on 30 such claims)."""
    out = []
    for q in sorted((q for q in queue if q["pass"] == "A"), key=lambda q: _h(q["hold_id"] + "#sided")):
        r = records[q["hold_id"]]
        if q["hold_id"] in exclude:
            continue
        ex = {z: v["exemplar"] for z, v in r["zones"].items() if "exemplar" in v}
        if any({ex.get(f"L_{p}"), ex.get(f"R_{p}")} == {0, 1} for p in ("FOOT", "HAND")):
            out.append(q["hold_id"])
        if len(out) == n:
            break
    return out


N_SIDED = 12


def select_items(records: dict, queue: list[dict]) -> list[dict]:
    ref = v3_calibration_holds()
    items = [packets_v4.item_a(records[h], "reference") for h in ref]
    items += [packets_v4.item_a(records[h], "fit") for h in fit_float_holds(records, queue)]
    lifts = sorted(ref, key=lambda h: _h(h + "#lift"))[:N_LIFT]
    items += [packets_v4.item_a(records[h], "lift", packets_v4.LIFTS_M[k % len(packets_v4.LIFTS_M)])
              for k, h in enumerate(lifts)]
    items += [packets_v4.item_a(records[h], "reference", purpose="calibration_sided")
              for h in sided_holds(records, queue, set(ref), N_SIDED)]
    return items


# --------------------------------------------------------------------------- #
# Calibration (replays the ledger)
# --------------------------------------------------------------------------- #
def calibrate(ledger: list[dict], records: dict, index: dict) -> tuple[dict, dict, list[str]]:
    """``(table, rows by reviewer key, failures)``: every reviewer key's classes over the v4 Pass-A packets, pooled
    and per variant group. One verdict per packet: the lowest valid ``n``."""
    cal = capture.load_calibration()
    by_pid = {e["packet_id"]: e for e in index.values()}
    mine = [r for r in ledger if r["pass"] == "A" and r["render_v"] == RENDER_V and r["packet_id"] in by_pid]
    truths, failures = {}, []
    for r in mine:
        e = by_pid[r["packet_id"]]
        if r["status"] != "valid" or e["packet_id"] in truths:
            continue
        try:
            rec = capture_v4.load(e["stem"], rebuild=False)
            truths[e["packet_id"]] = packets_v4.truth(records[e["hold_id"]], rec, cal, e["frame_hold"], e["variant"],
                                                      e["lift_m"])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{e['item_id']}: {type(exc).__name__}: {exc}")
    by_key = collections.defaultdict(list)
    for r in mine:
        by_key[r["reviewer"]["key"]].append(r)
    reviewers, all_rows = [], {}
    for key, calls in sorted(by_key.items()):
        first = {}
        for r in calls:
            if r["status"] == "valid" and r["packet_id"] in truths:
                first.setdefault(r["packet_id"], r)
        if not first:
            continue
        chosen = list(first.values())
        # reviewer_table keys truths by hold id; a hold appears in several variants, so key them by packet
        rows = [row for v in chosen for row in verdicts.score(v["answer"], truths[v["packet_id"]],
                                                              {"packet_id": v["packet_id"], "n": v["n"],
                                                               "variant": by_pid[v["packet_id"]]["variant"],
                                                               "lift_m": by_pid[v["packet_id"]]["lift_m"]})]
        pooled = {cls: verdicts.class_metrics([x for x in rows if x["class"] == cls], kind)
                  for cls, kind in verdicts.CLASSES.items()}
        pooled["float"]["recall_by_size"] = verdicts._float_recall_by_size([x for x in rows if x["class"] == "float"])
        groups = {v: {cls: verdicts.class_metrics([x for x in rows if x["class"] == cls and x["variant"] == v], kind)
                      for cls, kind in verdicts.CLASSES.items()} for v in packets_v4.VARIANTS}
        head = chosen[0]["reviewer"]
        usage = [c.get("usage") or {} for c in calls]
        reviewers.append({
            "key": key, "model": head["model"], "effort": head["effort"],
            "prompt_sha256": chosen[0]["prompt"]["sha256"], "schema_sha256": chosen[0]["schema"]["sha256"],
            "packets": len(chosen), "packets_by_variant": dict(collections.Counter(by_pid[v["packet_id"]]["variant"] for v in chosen)),
            "calls": dict(collections.Counter(c["status"] for c in calls)),
            "cost_usd": verdicts._stats([c.get("cost_usd") for c in calls]),
            "duration_s": verdicts._stats([c.get("duration_s") for c in calls]),
            "output_tokens": verdicts._stats([u.get("output_tokens") for u in usage]),
            "classes": pooled, "by_variant": groups,
            "evidence": [c for c, m in pooled.items() if m["evidence"]],
            "advisory": [c for c, m in pooled.items() if not m["evidence"]]})
        all_rows[key] = rows
    table = {"render_v": RENDER_V, "pass": "A", "rule": verdicts.RULE, "truth": {**verdicts.TRUTH,
             "avatar": "the variant drawn (reference, fit or lift) on plant v2"},
             "audit_ids": sorted({records[by_pid[p]["hold_id"]]["audit_id"] for p in truths}),
             "packets": len(truths), "reviewers": reviewers}
    return table, all_rows, failures


def write_table(table: dict, rows: dict, out: Path | None = None) -> Path:
    out = out or CALIBRATION_DIR / f"{RENDER_V}.json"
    inputs = [Path(__file__), verdicts.prompt_path("A"), verdicts.schema_path("A"), capture.CALIBRATION_PATH,
              packets_v4.index_path("A")]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), **table}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    for key, rs in rows.items():
        p = verdicts.ITEMS_ROOT / RENDER_V / f"pass_a.{key}.items.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, sort_keys=True, default=float) + "\n" for r in rs))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--build", action="store_true")
    what.add_argument("--review", action="store_true")
    what.add_argument("--calibrate", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=15.0)
    ap.add_argument("--effort", choices=review.EFFORTS, default="high")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    start = time.time()
    aud = packets.load_audit(audit_dir())
    manifest = LABELS_V11 / "holds.yaml"
    if args.build:
        items = select_items(aud.records, aud.queue)[:args.limit]
        entries, failures, built = packets_v4.build_batch_a(items, aud.records, manifest)
        for f in failures:
            print(f"FAILED {f}", file=sys.stderr)
        print(f"calibrate_v4 build: {len(entries)} of {len(items)} packets ready ({built} built; "
              f"{dict(collections.Counter(e['variant'] for e in entries))}), {len(failures)} failed, "
              f"{sum(e['bytes'] for e in entries) / 1e6:.1f} MB in {time.time() - start:.0f} s")
        return 1 if failures else 0
    index = packets.read_index(packets_v4.index_path("A"))
    if args.review:
        model = "stub" if args.dry_run else review.MODEL
        reviewer = review.Reviewer(model=model, effort=args.effort, pass_="A")
        entries = sorted(index.values(), key=lambda e: (packets_v4.VARIANTS.index(e["variant"]), e["item_id"]))
        entries = entries[:args.limit] if args.limit else entries
        if not args.dry_run and not review.cli_version(reviewer.cli):
            print("FAILED the reviewer CLI does not run", file=sys.stderr)
            return 1
        ledger_dir = review.DRY_LEDGER_DIR if args.dry_run else verdicts.LEDGER_DIR
        out = review.run(entries, reviewer, review.call_stub if args.dry_run else review.call_claude,
                         ledger_dir=ledger_dir, max_cost=args.max_cost, est_cost=0.2)
        for f in out["failures"]:
            print(f"FAILED {f}", file=sys.stderr)
        print(f"calibrate_v4 review {RENDER_V} {model} effort {args.effort} key {reviewer.key}: {len(entries)} packets, "
              f"{out['current']} current, {sum(out['status'].values())} made {out['status']}, {out['unstarted']} left by the "
              f"budget, {len(out['blocked'])} blocked; ${out['spent']:.2f}")
        return 1 if out["failures"] or out["blocked"] else (2 if out["unstarted"] else 0)
    table, rows, failures = calibrate(verdicts.read_ledger(), aud.records, index)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures or not table["reviewers"]:
        print("calibrate_v4: nothing written", file=sys.stderr)
        return 1
    out = write_table(table, rows)
    for rv in table["reviewers"]:
        c = rv["cost_usd"]
        print(f"{rv['model']} effort {rv['effort']} (key {rv['key']}): {rv['packets']} packets {rv['packets_by_variant']}, "
              f"${c.get('total', 0):.2f}")
        for cls, m in rv["classes"].items():
            prec = "-" if m["precision"] is None else f"{m['precision']:.3f}"
            rate = "-" if m["answered_rate"] is None else f"{m['answered_rate']:.2f}"
            print(f"  {cls:17s} {'EVIDENCE' if m['evidence'] else 'advisory':8s} precision {prec:>5s} on {m['support']:3d} "
                  f"claims, answered {rate:>4s} of {m['items']:4d}" + (f", recall {m['recall']:.3f}" if m.get("recall") is not None else ""))
    print(f"calibrate_v4: {table['packets']} packets -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
