# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Informed review (BUILD_PLAN Step 6, Pass B): packets that show the reviewer the labels and the
measurements, the Pass-B contract, the headless run and the Pass-B calibration.

Packets
-------
One per Pass-B item of Step 2's queue: every flagged hold, plus a hash-chosen 15 % of the clean
holds as controls. They are rendered with the current ``render_v`` (render_v3, whose claims Step 4
calibrated). Three differences from Step 3's Pass-B packets:

* The main moment is the labels' corrected exemplar (``labels.choose``), so the reviewer judges the
  configuration the labels will ship. The source exemplar and the ends of the source window and of
  the corrected window are time-strip moments too.
* ``packet.json`` adds three blocks. ``capture`` holds the corrected window and moments.
  ``human_mesh`` holds, per rendered moment, every part's skin floor touch and height, and every
  candidate pair's skin state and gap with the avatar's gap. ``candidates`` lists the source label's
  pairs and the pairs the skin touches at the main moment, labelled ones first, at most 16.
* The block Step 3 calls ``hold`` is ``labels`` here: "hold" occurs in a clip name, and the prompt
  names the block.

The reviewer is not shown the deterministic roles, so its roles are independent evidence. The index
is ``output/reference_curation/packets/<render_v>/_index_informed.jsonl``, and the directory is
``REVIEW_ROOT/<render_v>/<packet_id>/`` like every packet's.

Contract
--------
``schemas/pass_b.json`` holds Pass A's five blocks unchanged: pose, floor, discrepancies, body_body
and implausible. So ``verdicts.validate`` and ``verdicts.score`` apply as they are. It adds
``variant``, ``roles``, ``timing`` and ``repair``, which ``prompts/pass_b.md`` asks for. The runner
is Step 4's (``review.run`` with ``Reviewer(pass_="B")``). The validator cannot check a role per
candidate or a strip panel for the better moment; ``labels.parse_review`` does, and drops a failing
part from the claims, never the whole verdict.

Calibration
-----------
``calibrate`` scores the Pass-A classes of every valid Pass-B verdict with Step 4's admission rule,
against ``verdicts.packet_truth`` at the packet's main moment (``verdicts.calibrate`` would use the
source exemplar). ``pose_identity`` is informed here, since the label names the pose, so it
measures nothing. The Pass-B blocks have no truth set. They are reported as agreement with the
deterministic labels, so ``roles`` stays advisory by construction. The table goes to
``data/reference_curation/calibration/pass_b/<render_v>.json``, a subdirectory because Step 4's
replay test globs ``render_v*.json``.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.informed --build
    ... --review --limit 4 --max-cost 2        # a pilot; --dry-run uses the abstaining stub
    ... --review --max-cost 45                 # every built packet, at high effort
    ... --calibrate
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from extract_contact_configs import ZONE_ORDER
from reference_curation import capture, human_mesh as hm, ids, labels, packets, render, review, sources, verdicts

MODULE = "reference_curation.informed"
SCHEMA_VERSION = 1
CONTRACT = "pass_b.v1"
INDEX_NAME = "_index_informed.jsonl"
MAX_CANDIDATES = 16
CALIBRATION_DIR = labels.PASS_B_CALIBRATION
DRY_LEDGER_DIR = ids.OUTPUT_ROOT / "ledger_dry_run" / "verdicts"
UNITS = {"skin_cm": "centimetres above the floor of the performer's lowest skin point",
         "gap_cm": "centimetres between two parts' skin", "avatar_gap_cm": "centimetres between two avatar parts",
         "floor, state": "1 touching, 0 apart, -1 not decided"}


def index_path(render_v: str = render.RENDER_V, index_root: Path = packets.INDEX_ROOT) -> Path:
    return Path(index_root) / render_v / INDEX_NAME


def current_packets(render_v: str = render.RENDER_V, index_root: Path = packets.INDEX_ROOT) -> dict:
    """``{hold_id: packet_id}`` of the built informed packets."""
    return {e["hold_id"]: e["packet_id"] for e in packets.read_index(index_path(render_v, index_root)).values()}


# --------------------------------------------------------------------------- #
# Packet content (pure)
# --------------------------------------------------------------------------- #
def words(pair) -> str:
    return " + ".join(render.ZONE_WORDS[z] for z in pair)


def candidate_pairs(clip: labels.Clip, hold: dict, frame: int) -> list[tuple[str, str]]:
    """The source label's pairs, then the pairs the skin touches at ``frame``, at most 16."""
    labelled = sorted({labels.parse_pair(p) for p in hold["pairs"] if ":" not in p}, key=labels.PAIR_COLUMN.get)
    touching = [hm.PAIRS[k] for k in np.nonzero(clip.pairs[frame] == 1)[0] if hm.PAIRS[k] not in labelled]
    return (labelled + touching)[:MAX_CANDIDATES]


def informed_item(item: dict, choice) -> dict:
    """The queue item with the corrected exemplar as its main moment and the corrected window's ends
    among its key frames."""
    w = dict(item["window"])
    w["key_frames"] = sorted(set(w["key_frames"]) | {w["frame_hold"], choice.frame_start, choice.frame_end,
                                                     choice.frame_hold})
    w["frame_hold"] = choice.frame_hold
    zones = sorted(set(item["zones"]) | set(choice.ground), key=ZONE_ORDER.index)
    return {**item, "window": w, "zones": zones}


def human_block(clip: labels.Clip, frames: list[int], tags: dict, cands: list, gaps: dict) -> dict:
    rec = clip.rec
    out = []
    for f in frames:
        out.append({"frame": f, "t_s": round(f / clip.fps, 3), "panels": tags.get(f, []),
                    "floor": {z: {"state": int(rec["human_floor_state"][f, labels.ZI[z]]),
                                  "skin_cm": labels._cm(rec["human_floor_z"][f, labels.ZI[z]])} for z in ZONE_ORDER},
                    "pairs": {labels.pair_name(p): {"state": int(clip.pairs[f, labels.PAIR_COLUMN[p]]),
                                                    "gap_cm": labels._cm(rec["human_pair_gap"][f, labels.PAIR_COLUMN[p]]),
                                                    "avatar_gap_cm": labels._num(gaps[f].get(p), 1)}
                              for p in cands}})
    return {"frames": out}


def content(item: dict, record: dict, clip: labels.Clip, choice, sheets, rendered, manifest: Path,
            has_markers: bool = True) -> dict:
    """``packet.json`` without its id."""
    numbers = list(range(1, len(sheets) + 1))
    tags = {}
    for sheet, n in zip(sheets, numbers):
        for k, p in enumerate(sheet.panels):
            tags.setdefault(p.frame, []).append(f"{n}{chr(ord('a') + k)}")
    frames = sorted(tags)
    hold = packets.manifest_hold(record, manifest)[1]
    cands = candidate_pairs(clip, hold, choice.frame_hold)
    gaps = {f: verdicts.pair_gaps_cm(clip.stem, f) for f in frames}
    informed = packets.informed(item, record, clip.rec, frames, tags, manifest)
    informed["labels"] = informed.pop("hold")
    fps = clip.fps
    src = record["window"]
    return {
        "schema_version": SCHEMA_VERSION, "contract": CONTRACT, "render_v": render.RENDER_V, "pass": "B",
        "images": [dict(im, **t) for im, t in zip(rendered, packets.describe(sheets, numbers, choice.frame_hold))],
        "legend": dict(packets.LEGEND, palette=render.palette_legend(),
                       **({} if has_markers else {"markers": packets.NO_MARKERS})),
        **informed,
        "capture": {"main_frame": choice.frame_hold, "main_s": round(choice.frame_hold / fps, 3),
                    "main_panels": tags.get(choice.frame_hold, []),
                    "source_main_frame": src["frame_hold"], "source_main_s": src["t_hold"],
                    "window": {"frame_start": choice.frame_start, "frame_end": choice.frame_end,
                               "t_start": round(choice.frame_start / fps, 3), "t_end": round(choice.frame_end / fps, 3)},
                    "floor_contacts_at_main": [z for z in ZONE_ORDER if z in choice.ground]},
        "human_mesh": human_block(clip, frames, tags, cands, gaps),
        "candidates": [{"pair": labels.pair_name(p), "parts": words(p), "labelled": labels.pair_name(p) in hold["pairs"],
                        "skin_state_at_main": int(clip.pairs[choice.frame_hold, labels.PAIR_COLUMN[p]]),
                        "skin_gap_cm_at_main": labels._cm(clip.rec["human_pair_gap"][choice.frame_hold,
                                                                                    labels.PAIR_COLUMN[p]]),
                        "avatar_gap_cm_at_main": labels._num(gaps[choice.frame_hold].get(p), 1)} for p in cands],
        "units": {**informed["units"], **UNITS},
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_key(item: dict, record: dict, clip: labels.Clip, manifest: Path) -> str:
    stable = {k: v for k, v in item.items() if k not in ("priority", "schema_version")}
    files = [__file__, labels.__file__, sources.__file__, packets.__file__, render.__file__]
    inputs = [ids.MJCF, ids.motion_path(clip.stem), manifest] + [p for p in [ids.mosh_path(clip.stem)] if p]
    return ids.sha256_json({"render_v": render.RENDER_V, "gl": render.gl_info(), "contract": CONTRACT,
                            "code": [ids.sha256_file(f) for f in files], "item": stable, "record": record,
                            "store": sources.identity(clip.rec),
                            "inputs": {ids.display_path(p): ids.sha256_file(p) for p in inputs}})


def build_packet(item: dict, record: dict, clip: labels.Clip, *, manifest: Path, index: dict,
                 review_root: Path = ids.REVIEW_ROOT, index_root: Path = packets.INDEX_ROOT,
                 force: bool = False) -> tuple[dict, bool]:
    """``(index entry, built)``: the informed packet of a Pass-B queue item. Reuses ``index``'s
    entry when its build key is unchanged and its directory complete."""
    start = time.time()
    w = record["window"]
    choice = labels.choose(clip, w["frame_start"], w["frame_end"], w["frame_hold"], set(record["label_ground"]))
    item = informed_item(item, choice)
    key = build_key(item, record, clip, manifest)
    old = index.get(item["item_id"])
    if not force and old and old["build_key"] == key and packets._complete(old, review_root):
        return old, False
    rclip = render.load_clip(clip.stem)
    frames = packets.strip_frames(item, rclip.num_frames)
    sheets = render.plan(rclip, choice.frame_hold, tuple(item["zones"]), frames)
    if len(sheets) > packets.MAX_IMAGES:
        raise ValueError(f"{len(sheets)} images > {packets.MAX_IMAGES}")
    rendered = render.render_sheets(rclip, sheets)
    staging = Path(index_root) / render.RENDER_V / ".staging_informed" / hashlib.sha1(item["item_id"].encode()).hexdigest()
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        images = []
        for n, (image, _) in enumerate(rendered, 1):
            path = staging / f"img_{n}.png"
            render.save_png(image, path)
            images.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        body = content(item, record, clip, choice, sheets, images, manifest, rclip.markers is not None)
        packet_id = hashlib.sha1(packets._canonical(body)).hexdigest()
        (staging / "packet.json").write_text(json.dumps({"packet_id": packet_id, **body}, indent=1, sort_keys=True) + "\n")
        total = sum(f.stat().st_size for f in staging.iterdir())
        if total > packets.MAX_PACKET_BYTES:
            raise ValueError(f"packet is {total} bytes > budget {packets.MAX_PACKET_BYTES}")
        final = Path(review_root) / render.RENDER_V / packet_id
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(final, ignore_errors=True)
        shutil.move(str(staging), str(final))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    inputs = [ids.MJCF, ids.motion_path(clip.stem), manifest, render.__file__, labels.__file__]
    entry = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "item_id": item["item_id"],
             "packet_id": packet_id, "render_v": render.RENDER_V, "pass": "B", "contract": CONTRACT,
             "dir": str(final), "gl": render.gl_info(), "hold_id": record["hold_id"], "stem": record["stem"],
             "audit_id": record["audit_id"], "purpose": item.get("purpose"), "stratum": None,
             "frame_hold": choice.frame_hold, "source_frame_hold": w["frame_hold"], "strip": frames,
             "candidates": [c["pair"] for c in body["candidates"]],
             "images": {im["file"]: im["sha256"] for im in images}, "bytes": total, "build_key": key,
             "seconds": round(time.time() - start, 2)}
    index[item["item_id"]] = entry
    return entry, True


def build_packets(items: list[dict], aud, stores: labels.Stores = labels.Stores(), *,
                  review_root: Path = ids.REVIEW_ROOT, index_root: Path = packets.INDEX_ROOT,
                  force: bool = False) -> tuple[list[dict], list[str], int]:
    """``(entries, failures, built)``; the index is rewritten once, at the end."""
    path = index_path(render.RENDER_V, index_root)
    index = packets.read_index(path)
    manifest = ids.load_manifest(aud.manifest)
    clips, entries, failures, built = {}, [], [], 0
    for item in items:
        try:
            record = aud.records[item["hold_id"]]
            if record["stem"] not in clips:
                clips[record["stem"]] = labels.clip_evidence(record["stem"], manifest, stores)
            entry, new = build_packet(item, record, clips[record["stem"]], manifest=aud.manifest, index=index,
                                      review_root=review_root, index_root=index_root, force=force)
            entries.append(entry)
            built += new
        except Exception as exc:  # noqa: BLE001 -- report every broken item, then fail
            failures.append(f"{item['item_id']}: {type(exc).__name__}: {exc}")
    packets.write_index(path, index)
    return entries, failures, built


# --------------------------------------------------------------------------- #
# The stub reviewer and selection
# --------------------------------------------------------------------------- #
def stub_answer(packet: dict) -> dict:
    """A schema-valid Pass-B answer that abstains everywhere, with a role entry per candidate."""
    parts = verdicts.load_schema("B")["properties"]["floor"]["required"]
    return {"pose": {"description": "cannot_tell", "candidates": []},
            "variant": {"matches_label": "cannot_tell", "note": ""},
            "floor": {p: {"avatar": "cannot_tell", "markers": "cannot_tell", "panels": []} for p in parts},
            "discrepancies": [], "body_body": {"answer": "cannot_tell", "contacts": []}, "implausible": [],
            "roles": [{"part_a": render.ZONE_WORDS[c["pair"].split("+")[0]],
                       "part_b": render.ZONE_WORDS[c["pair"].split("+")[1]], "role": "cannot_tell", "note": "",
                       "panels": []} for c in packet["candidates"]],
            "timing": {"start": "cannot_tell", "end": "cannot_tell", "main_moment": "cannot_tell", "better_panel": "",
                       "note": ""},
            "repair": {"needed": "cannot_tell", "kinds": [], "note": ""}}


def call_stub(reviewer: review.Reviewer, packet_dir: Path) -> dict:
    """The dry run's reviewer: no process, no cost, the abstaining answer."""
    packet = json.loads((Path(packet_dir) / "packet.json").read_text())
    envelope = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 0, "duration_ms": 0,
                "total_cost_usd": 0.0, "session_id": "stub", "permission_denials": [],
                "usage": {"input_tokens": 0, "output_tokens": 0}, "structured_output": stub_answer(packet)}
    return {"envelope": envelope, "returncode": 0, "stderr": "", "seconds": 0.0, "started": review._now()}


def select_entries(index: dict, queue: list[dict], holds=(), limit: int | None = None) -> list[dict]:
    """The built informed packets in Pass-B queue priority order, narrowed to ``holds``, cut to ``limit``."""
    return review.select_entries(index, queue, "B", holds, (), limit)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _agreement(rows: list[tuple]) -> dict:
    n = len(rows)
    return {"items": n, "agree": sum(bool(a) for a, _ in rows), "rate": labels._num(sum(bool(a) for a, _ in rows) / n, 4)
            if n else None}


def pass_b_metrics(chosen: list[dict], claims: dict, result: dict) -> dict:
    """The Pass-B blocks against the deterministic labels: agreement, not accuracy (no truth set)."""
    anns = collections.defaultdict(dict)
    for a in result["annotations"]:
        anns[a["hold_id"]][a["contact"]] = a
    roles, by_class, conflicts = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter()
    agree_rows, timing, variant, repair, float_rows = [], collections.Counter(), collections.Counter(), collections.Counter(), []
    errors = collections.Counter()
    for v in chosen:
        c, hold = claims[v["hold_id"]], result["holds"].get(v["hold_id"])
        errors["verdicts_with_contract_errors"] += bool(c["errors"])
        variant[c["variant"]["matches_label"]] += 1
        timing[c["timing"]["main_moment"]] += 1
        repair[c["repair"]["needed"]] += 1
        for pair, role in c["roles"].items():
            roles[role or "cannot_tell"] += 1
            a = anns[v["hold_id"]].get(pair)
            if a is None or role is None:
                continue
            by_class[a["load_path_if_touching"]][role] += 1
            if a["source_state"] == "observed_separation" and role == "required_touch":
                conflicts["required_but_apart"] += 1
            if a["source_state"] == "observed_contact" and a["target_role"] in labels.REVIEW_ROLES:
                agree_rows.append((role == a["target_role"], pair))
        if hold is not None:
            floating = any(a["kind"] == "ground" and a["in_configuration"] and a["source_state"] == "observed_contact"
                           and not a["evidence"]["avatar"]["realised"] for a in anns[v["hold_id"]].values())
            if c["repair"]["needed"] in ("yes", "no"):
                float_rows.append(((c["repair"]["needed"] == "yes" and "float_support" in c["repair"]["kinds"]) == floating,
                                   v["hold_id"]))
    return {"roles": dict(roles), "roles_by_load_path": {k: dict(x) for k, x in by_class.items()},
            "role_agreement_with_provisional": _agreement(agree_rows), "role_conflicts": dict(conflicts),
            "variant": dict(variant), "main_moment": dict(timing), "repair_needed": dict(repair),
            "float_support_agreement_with_hover": _agreement(float_rows), "contract": dict(errors),
            "note": "no truth set: agreement with the deterministic labels, never admitted as evidence"}


def calibrate(ledger: list[dict], aud, stores: labels.Stores = labels.Stores(), render_v: str = render.RENDER_V) -> tuple[dict, dict, list[str]]:
    """``(table, rows by reviewer key, failures)`` from the valid Pass-B verdicts on ``render_v``
    packets, one per (packet, reviewer key), each scored at its packet's main moment."""
    cal = capture.load_calibration()
    mine = [r for r in ledger if r["pass"] == "B" and r["render_v"] == render_v]
    failures, truths = [], {}
    for r in mine:
        if r["status"] != "valid" or r["hold_id"] in truths:
            continue
        try:
            record = aud.records[r["hold_id"]]
            truths[r["hold_id"]] = verdicts.packet_truth(record, labels.load_record(record["stem"], stores), cal,
                                                         frame=r["frame_hold"])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{r['hold_id']}: {type(exc).__name__}: {exc}")
    by_key = collections.defaultdict(list)
    for r in mine:
        by_key[r["reviewer"]["key"]].append(r)
    stems = sorted({aud.records[h]["stem"] for h in truths})
    result = labels.build(aud, stores, stems=stems) if stems else {"annotations": [], "holds": {}}
    failures += result.get("failures", [])
    reviewers, all_rows = [], {}
    for key, calls in sorted(by_key.items()):
        first = {}
        for r in calls:
            if r["status"] == "valid" and r["hold_id"] in truths:
                first.setdefault(r["packet_id"], r)
        if not first:
            continue
        chosen = list(first.values())
        table, rows = verdicts.reviewer_table(chosen, calls, truths)
        claims = {}
        for v in chosen:
            packet = json.loads((Path(v["packet"]["dir"]) / "packet.json").read_text())
            claims[v["hold_id"]] = labels.parse_review(v, packet)
        table["pass_b"] = pass_b_metrics(chosen, claims, result)
        table["advisory"] = sorted(set(table["advisory"]) | {"roles", "variant", "timing", "repair"})
        table["purposes"] = dict(collections.Counter(v.get("purpose") for v in chosen))
        reviewers.append(table)
        all_rows[key] = rows
    return ({"render_v": render_v, "pass": "B", "contract": CONTRACT, "rule": verdicts.RULE, "truth": verdicts.TRUTH,
             "truth_frame": "the packet's main moment (the labels' corrected exemplar)",
             "audit_ids": sorted({aud.records[h]["audit_id"] for h in truths}), "holds": len(truths),
             "reviewers": reviewers}, all_rows, failures)


def write_calibration(table: dict, rows: dict, ledger_dir: Path, out: Path,
                      items_root: Path = verdicts.ITEMS_ROOT) -> Path:
    inputs = [Path(__file__), verdicts.prompt_path("B"), verdicts.schema_path("B"), capture.CALIBRATION_PATH]
    inputs += sorted(Path(ledger_dir).glob("*.B.*.json"))
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), **table}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=1) + "\n")
    for key, rs in rows.items():
        path = Path(items_root) / table["render_v"] / f"pass_b.{key}.items.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rs))
    return Path(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--build", action="store_true", help="build the informed packets of the Pass-B queue")
    what.add_argument("--review", action="store_true", help="review the built packets with the headless reviewer")
    what.add_argument("--calibrate", action="store_true", help="rebuild the Pass-B calibration from the ledger")
    ap.add_argument("--audit", type=Path, help="an audit directory (default: the newest calibrated audit)")
    ap.add_argument("--hold", nargs="*", default=[], help="only these holds")
    ap.add_argument("--limit", type=int, help="the first N items in queue priority order")
    ap.add_argument("--force", action="store_true", help="--build: rebuild even if the index is current")
    ap.add_argument("--effort", choices=review.EFFORTS, default="high")
    ap.add_argument("--max-cost", type=float, default=45.0, help="--review: USD for the whole run")
    ap.add_argument("--per-call-max", type=float, default=2.0)
    ap.add_argument("--est-cost", type=float, default=0.35)
    ap.add_argument("--parallel", type=int, default=review.MAX_PARALLEL)
    ap.add_argument("--dry-run", action="store_true", help="--review: the abstaining stub, into its own ledger")
    ap.add_argument("--ledger-dir", type=Path)
    ap.add_argument("--review-root", type=Path, default=ids.REVIEW_ROOT)
    ap.add_argument("--index-root", type=Path, default=packets.INDEX_ROOT)
    ap.add_argument("--out", type=Path, help="--calibrate: default calibration/pass_b/<render_v>.json")
    args = ap.parse_args(argv)

    start = time.time()
    try:
        aud = packets.load_audit(args.audit or packets.default_audit_dir())
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ledger_dir = args.ledger_dir or (DRY_LEDGER_DIR if args.dry_run else verdicts.LEDGER_DIR)

    if args.build:
        queue = sorted((q for q in aud.queue if q["pass"] == "B"), key=lambda q: q["priority"])
        items = [q for q in queue if not args.hold or q["hold_id"] in args.hold][:args.limit]
        entries, failures, built = build_packets(items, aud, review_root=args.review_root, index_root=args.index_root,
                                                 force=args.force)
        for f in failures:
            print(f"FAILED {f}", file=sys.stderr)
        mb = sum(e["bytes"] for e in entries) / 1e6
        print(f"informed packets {render.RENDER_V}: {len(entries)} of {len(items)} ready ({built} built), "
              f"{len(failures)} failed, {mb:.1f} MB in {time.time() - start:.0f} s -> {args.review_root / render.RENDER_V} "
              f"(index {ids.display_path(index_path(render.RENDER_V, args.index_root))})")
        return 1 if failures else 0

    if args.review:
        model = "stub" if args.dry_run else review.MODEL
        reviewer = review.Reviewer(model=model, effort=args.effort, pass_="B", per_call_max_usd=args.per_call_max)
        index = packets.read_index(index_path(render.RENDER_V, args.index_root))
        entries = select_entries(index, aud.queue, args.hold, args.limit)
        if not entries:
            print(f"FAILED no built informed packets selected in {args.index_root}", file=sys.stderr)
            return 1
        if not args.dry_run and not review.cli_version(reviewer.cli):
            print("FAILED the reviewer CLI does not run", file=sys.stderr)
            return 1
        out = review.run(entries, reviewer, call_stub if args.dry_run else review.call_claude, ledger_dir=ledger_dir,
                         review_root=args.review_root, max_cost=args.max_cost, est_cost=args.est_cost,
                         parallel=args.parallel)
        for f in out["failures"]:
            print(f"FAILED {f}", file=sys.stderr)
        print(f"review pass B {render.RENDER_V} {model} effort {args.effort} key {reviewer.key}: {len(entries)} packets, "
              f"{out['current']} current, {sum(out['status'].values())} made {out['status']}, {out['unstarted']} left by "
              f"the ${args.max_cost:g} budget, {len(out['blocked'])} blocked; ${out['spent']:.2f} -> "
              f"{ids.display_path(ledger_dir)}")
        if out["failures"] or out["blocked"]:
            return 1
        return 2 if out["unstarted"] else 0

    table, rows, failures = calibrate(verdicts.read_ledger(ledger_dir), aud)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures or not table["reviewers"]:
        print(f"calibration pass B: {len(failures)} failures, {len(table['reviewers'])} reviewers; nothing written",
              file=sys.stderr)
        return 1
    out = write_calibration(table, rows, ledger_dir, args.out or CALIBRATION_DIR / f"{render.RENDER_V}.json")
    print(verdicts.format_table(table))
    evidence = {rv["effort"]: rv["evidence"] for rv in table["reviewers"]}
    print(f"calibration {render.RENDER_V} pass B: {table['holds']} holds; evidence {evidence} -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
