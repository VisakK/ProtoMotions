# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence packets v4 (BodyFix Step 4, items 3 and 6): Steps 3-6's packets on render_v4 (plant v2, one hue per limb
segment), the Pass-A recalibration set, the informed (Pass B) packets of the label review, and the truth both are
scored against.

Packets are built as ``packets.py`` (Pass A, blind) and ``informed.py`` (Pass B) build them, with these changes:

* **render_v4** (``render_v4``): the Step 3 references on plant v2 (``capture_v4.RETARGET_ID``); every geometric
  call runs inside ``render_v4.plant_context()``. The legend's colour text names the new segment hues.
* **Variants** (Pass A, the calibration's machine truth): ``reference`` (the release motion), ``fit`` (her unedited
  fit on plant v2, ``fit_writer``: natural 2-3 cm floats of a supinated foot or a tilted hand), ``lift`` (the
  reference raised by a known ``lift_m``, 2.5-12 cm: every support floats by exactly that much). The plant-v2
  references float on 1 of 864 labelled supports, so the calibration's float class needs the last two. A packet
  carries no trace of its variant; the private index does.
* **Informed packets** (Pass B) show the main moment at labels v2's exemplar, the *source* labels (the original
  manifest's ground set and pairs, as render_v3's did, never labels v1/v2's deterministic roles, so the reviewer's
  roles stay independent evidence), labels v2's window and configured ground set as the capture's, the store v4
  numbers, the skin and the plant-v2 avatar gaps per candidate pair (source label pairs, then the pairs the skin
  touches at the main moment, at most 16). Contract unchanged (``prompts/pass_b.md``, ``schemas/pass_b.json``).

Truth (``truth``) is ``verdicts.packet_truth``'s rule on what the packet shows: human touch from the markers
(decisive and stable), the avatar's three-way floor state and pair state from the **variant's own plant-v2 geometry**
(``verdicts.packet_truth`` would read the shipped reference on plant v1).

Index: ``output/reference_curation/packets/render_v4/_index.jsonl`` (Pass A) and ``_index_informed.jsonl`` (Pass B);
packets in ``REVIEW_ROOT/render_v4/<packet_id>/``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from reference_curation import render_v4  # first: render (OSMesa) before anything imports mujoco
from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, capture, capture_v4, fit_writer as fw, human_mesh as hm, ids, labels as L1
from reference_curation import packets, render, verdicts

MODULE = "reference_curation.packets_v4"
SCHEMA_VERSION = 1
RENDER_V = render_v4.RENDER_V
CONTRACT_B = "pass_b.v1"
INDEX_A, INDEX_B = "_index.jsonl", "_index_informed.jsonl"
VARIANTS = ("reference", "fit", "lift")
LIFTS_M = (0.025, 0.04, 0.06, 0.09, 0.12)
FIT_ID = "fit_v2.v1.b8f5e86fc9"
LEGEND = {**packets.LEGEND,
          "colours": "Every body segment of the avatar has its own colour (see palette): the upper arm, forearm and hand "
                     "of an arm have three different colours, and so do the thigh, shin and foot of a leg. Left parts are "
                     "the dark shade and right parts the light shade of their colour."}
UNITS_B = {"skin_cm": "centimetres above the floor of the performer's lowest skin point",
           "gap_cm": "centimetres between two parts' skin", "avatar_gap_cm": "centimetres between two avatar parts",
           "floor, state": "1 touching, 0 apart, -1 not decided"}


def motion_dir(variant: str) -> Path:
    return fw.OUT_ROOT / FIT_ID if variant == "fit" else capture_v4.motion_path("x").parent


def index_path(which: str = "A", index_root: Path = packets.INDEX_ROOT) -> Path:
    return Path(index_root) / RENDER_V / (INDEX_A if which == "A" else INDEX_B)


def variant_clip(stem: str, variant: str = "reference", lift_m: float = 0.0) -> render.Clip:
    """The clip a packet draws: the reference, her unedited fit, or the reference raised by ``lift_m``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    clip = render.load_clip(stem, motion_dir("fit" if variant == "fit" else "reference"))
    return render_v4.lifted(clip, lift_m) if variant == "lift" else clip


# --------------------------------------------------------------------------- #
# Truth on what the packet shows (pure given the geometry)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=64)
def _variant_geometry(stem: str, variant: str, lift_m: float) -> tuple[np.ndarray, np.ndarray]:
    """``(zone heights [T,Z], zone pair gaps [T,P])`` of a variant on plant v2 (``human_mesh.PAIRS`` order)."""
    from reference_curation import mosh_replay as mr
    from reference_curation.retarget_v2 import motion_state

    sk = fw.skeleton("v2")
    m = torch.load(ids.motion_path(stem, motion_dir("fit" if variant == "fit" else "reference")), map_location="cpu",
                   weights_only=False)
    pos, rot = motion_state(m)
    z = mr.zone_lowest(sk, pos, rot) + (lift_m if variant == "lift" else 0.0)
    with torch.no_grad():
        g = capture_v4.zone_pair_gaps(sk, torch.as_tensor(pos), torch.as_tensor(rot))
    return z, g


def truth(record: dict, rec: capture.Capture, cal: dict, frame: int, variant: str = "reference",
          lift_m: float = 0.0) -> dict:
    """``verdicts.packet_truth``'s rule at ``frame`` with the avatar side of the variant drawn, on plant v2."""
    fh = int(frame)
    near = round(audit.STABLE_S * rec.fps)
    human, marker_cm = {}, {}
    for z in record["capture"]["decided"]:
        zi = verdicts.ZI[z]
        state, height = int(rec["ground_state"][fh, zi]), float(rec["marker_min_z"][fh, zi])
        band = cal["zones"][z]
        decisive = state in (0, 1) and not (band["touch_m"] < height <= band["separation_m"])
        flips = audit.support_changes(rec["ground_state"][:, [zi]])
        stable = not len(flips) or int(np.abs(flips - fh).min()) > near
        human[z] = state if decisive and stable else None
        marker_cm[z] = round(100.0 * height, 3)
    zh, pg = _variant_geometry(record["stem"], variant, float(lift_m))
    avatar_cm = {z: round(100.0 * float(zh[fh, verdicts.ZI[z]]), 3) for z in ZONE_ORDER}
    col = {p: k for k, p in enumerate(hm.PAIRS)}
    gaps = {p: round(100.0 * float(pg[fh, col[p]]), 3) for p in verdicts.BODY_PAIRS}
    return {"hold_id": record["hold_id"], "frame": fh, "stratum": audit.stratum(record),
            "family_hold": record["family_hold"], "pose": verdicts.pose_truth(record["name"]),
            "human": human, "marker_cm": marker_cm, "avatar_cm": avatar_cm,
            "avatar": {z: verdicts._three_way(a, verdicts.AVATAR_TOUCH_CM, verdicts.AVATAR_OFF_CM) for z, a in avatar_cm.items()},
            "pair_gap_cm": gaps, "pairs": {p: verdicts._three_way(g, verdicts.PAIR_TOUCH_CM, verdicts.PAIR_APART_CM)
                                           for p, g in gaps.items()}, "variant": variant, "lift_m": lift_m}


# --------------------------------------------------------------------------- #
# Pass A (blind)
# --------------------------------------------------------------------------- #
def item_a(record: dict, variant: str = "reference", lift_m: float = 0.0, purpose: str = "calibration") -> dict:
    """A Pass-A item of any hold (``packets.item_for``), for one variant."""
    item = packets.item_for(record, "A")
    tag = variant if variant != "lift" else f"lift{round(100 * lift_m, 1):g}"
    return {**item, "item_id": f"{record['hold_id']}#A4:{tag}", "purpose": purpose, "variant": variant,
            "lift_m": float(lift_m)}


def _key(item: dict, record: dict, clip_inputs: list[Path], manifest: Path, extra=None) -> str:
    stable = {k: v for k, v in item.items() if k not in ("priority", "schema_version")}
    files = [__file__, render_v4.__file__, render.__file__, packets.__file__]
    return ids.sha256_json({"render_v": RENDER_V, "code": [ids.sha256_file(f) for f in files], "gl": render.gl_info(),
                            "item": stable, "record": record, "extra": extra, "plant": fw.plant_paths("v2")[0].name,
                            "inputs": {ids.display_path(p): ids.sha256_file(p) for p in clip_inputs + [manifest]}})


def _stage(items_png: list, staging: Path) -> list[dict]:
    images = []
    for n, (image, _) in enumerate(items_png, 1):
        if max(image.size) > render.SHEET_MAX_PX:
            raise ValueError(f"image {n} is {image.size}, over {render.SHEET_MAX_PX} px")
        path = staging / f"img_{n}.png"
        render.save_png(image, path)
        images.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return images


def _finish(staging: Path, content: dict, review_root: Path) -> tuple[str, Path, int]:
    packet_id = hashlib.sha1(packets._canonical(content)).hexdigest()
    (staging / "packet.json").write_text(json.dumps({"packet_id": packet_id, **content}, indent=1, sort_keys=True) + "\n")
    total = sum(f.stat().st_size for f in staging.iterdir())
    if total > packets.MAX_PACKET_BYTES:
        raise ValueError(f"packet is {total} bytes > budget {packets.MAX_PACKET_BYTES}")
    final = Path(review_root) / RENDER_V / packet_id
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(final, ignore_errors=True)
    shutil.move(str(staging), str(final))
    return packet_id, final, total


def build_packet_a(item: dict, record: dict, *, manifest: Path, index: dict, review_root: Path = ids.REVIEW_ROOT,
                   index_root: Path = packets.INDEX_ROOT, force: bool = False) -> tuple[dict, bool]:
    """A blind Pass-A packet of ``item`` on render_v4 (call inside ``render_v4.plant_context()``)."""
    start = time.time()
    stem, variant, lift_m = record["stem"], item["variant"], item["lift_m"]
    mpath = ids.motion_path(stem, motion_dir(variant))
    key = _key(item, record, [mpath] + [p for p in [ids.mosh_path(stem)] if p], manifest)
    old = index.get(item["item_id"])
    if not force and old and old["build_key"] == key and packets._complete(old, review_root):
        return old, False
    clip = variant_clip(stem, variant, lift_m)
    hold = int(item["window"]["frame_hold"])
    frames = packets.strip_frames(item, clip.num_frames)
    sheets = render.plan(clip, hold, tuple(item["zones"]), frames)
    if len(sheets) > packets.MAX_IMAGES:
        raise ValueError(f"{len(sheets)} images > {packets.MAX_IMAGES}")
    rendered = render_v4.render_sheets(clip, sheets)
    numbers = list(range(1, len(sheets) + 1))
    staging = Path(index_root) / RENDER_V / ".staging" / hashlib.sha1(item["item_id"].encode()).hexdigest()
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        images = _stage(rendered, staging)
        content = {"schema_version": SCHEMA_VERSION, "render_v": RENDER_V, "pass": "A",
                   "images": [dict(im, **t) for im, t in zip(images, packets.describe(sheets, numbers, hold))],
                   "legend": dict(LEGEND, palette=render_v4.palette_legend(),
                                  **({} if clip.markers is not None else {"markers": packets.NO_MARKERS}))}
        found = packets.leaks(json.dumps(content), record, item, manifest)
        if found:
            raise ValueError(f"pass-A packet would leak {found}")
        packet_id, final, total = _finish(staging, content, review_root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    entry = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [mpath, manifest, render_v4.__file__]),
             "item_id": item["item_id"], "packet_id": packet_id, "render_v": RENDER_V, "pass": "A", "dir": str(final),
             "gl": render.gl_info(), "hold_id": record["hold_id"], "stem": stem, "audit_id": record["audit_id"],
             "purpose": item.get("purpose"), "stratum": item.get("stratum"), "variant": variant, "lift_m": lift_m,
             "frame_hold": hold, "strip": frames, "images": {im["file"]: im["sha256"] for im in images}, "bytes": total,
             "build_key": key, "seconds": round(time.time() - start, 2)}
    index[item["item_id"]] = entry
    return entry, True


# --------------------------------------------------------------------------- #
# Pass B (informed)
# --------------------------------------------------------------------------- #
def candidate_pairs(source_pairs: list[str], pair_state: np.ndarray, frame: int) -> list[tuple[str, str]]:
    """The source label's pairs, then the pairs the skin touches at ``frame``, at most 16 (``informed``'s rule)."""
    labelled = sorted({L1.parse_pair(p) for p in source_pairs if ":" not in p}, key=L1.PAIR_COLUMN.get)
    touching = [hm.PAIRS[k] for k in np.nonzero(pair_state[frame] == 1)[0] if hm.PAIRS[k] not in labelled]
    return (labelled + touching)[:16]


def informed_content(item: dict, record: dict, rec: capture.Capture, hold_v2: dict, clip: render.Clip, sheets,
                     images: list[dict], manifest: Path) -> dict:
    """``packet.json`` (without its id) of an informed v4 packet: ``informed.content``'s blocks on store v4."""
    numbers = list(range(1, len(sheets) + 1))
    tags = {}
    for sheet, n in zip(sheets, numbers):
        for k, p in enumerate(sheet.panels):
            tags.setdefault(p.frame, []).append(f"{n}{chr(ord('a') + k)}")
    frames = sorted(tags)
    ex = int(hold_v2["frame_hold"])
    src = hold_v2["labels"]["source"]
    pairs = rec["human_pair_state"]
    cands = candidate_pairs(src["pairs"], pairs, ex)
    col = {p: k for k, p in enumerate(hm.PAIRS)}
    clip_entry, mh = packets.manifest_hold(record, manifest)
    zones = [z for z in ZONE_ORDER if z in set(item["zones"]) | set(render.EXTREMITIES)]
    fps = rec.fps
    hold_block = {"hold_id": record["hold_id"], "stem": record["stem"], "name": record["name"],
                  "family": clip_entry.get("family"), "group": record["group"], "orientation": record["orientation"],
                  "family_hold": record["family_hold"],
                  "label_ground": [p[:-2] for p in src["pairs"] if p.endswith(":G")],
                  "label_pairs": list(src["pairs"]),
                  "window": {"frame_start": src["frame_start"], "frame_end": src["frame_end"], "frame_hold": src["frame_hold"],
                             "fps": fps, "t_start": round(src["frame_start"] / fps, 3), "t_end": round(src["frame_end"] / fps, 3),
                             "t_hold": round(src["frame_hold"] / fps, 3)}}

    def avatar_gap(f, p):
        return L1._num(100.0 * float(rec["avatar_pair_gap"][f, col[p]]), 1)

    human = {"frames": [{"frame": f, "t_s": round(f / fps, 3), "panels": tags.get(f, []),
                         "floor": {z: {"state": int(rec["human_floor_state"][f, L1.ZI[z]]),
                                       "skin_cm": L1._cm(rec["human_floor_z"][f, L1.ZI[z]])} for z in ZONE_ORDER},
                         "pairs": {L1.pair_name(p): {"state": int(pairs[f, col[p]]),
                                                     "gap_cm": L1._cm(rec["human_pair_gap"][f, col[p]]),
                                                     "avatar_gap_cm": avatar_gap(f, p)} for p in cands}}
                        for f in frames]}
    return {
        "schema_version": SCHEMA_VERSION, "contract": CONTRACT_B, "render_v": RENDER_V, "pass": "B",
        "images": [dict(im, **t) for im, t in zip(images, packets.describe(sheets, numbers, ex))],
        "legend": dict(LEGEND, palette=render_v4.palette_legend(),
                       **({} if clip.markers is not None else {"markers": packets.NO_MARKERS})),
        "labels": hold_block,
        "item": {k: item.get(k) for k in ("item_id", "purpose", "severity", "reasons", "questions")},
        "audit_id": record["audit_id"],
        "evidence": {"exemplar": {k: record[k] for k in ("capture", "zones", "mat", "disagreements")},
                     "frames": [dict(packets.frame_evidence(rec, f, zones), panels=tags.get(f, [])) for f in frames]},
        "units": {"*_cm": "centimetres above the floor", "*_n": "newtons, over mat-coverage-valid frames",
                  "human": "the capture's touch state: 1 contact, 0 separated, -1 unknown or undecided",
                  "panels": "tags of the panels that show the frame", **UNITS_B},
        "capture": {"main_frame": ex, "main_s": round(ex / fps, 3), "main_panels": tags.get(ex, []),
                    "source_main_frame": src["frame_hold"], "source_main_s": round(src["frame_hold"] / fps, 3),
                    "window": {"frame_start": hold_v2["frame_start"], "frame_end": hold_v2["frame_end"],
                               "t_start": round(hold_v2["frame_start"] / fps, 3), "t_end": round(hold_v2["frame_end"] / fps, 3)},
                    "floor_contacts_at_main": [p[:-2] for p in hold_v2["pairs_ground"]]},
        "human_mesh": human,
        "candidates": [{"pair": L1.pair_name(p), "parts": " + ".join(render.ZONE_WORDS[z] for z in p),
                        "labelled": L1.pair_name(p) in src["pairs"], "skin_state_at_main": int(pairs[ex, col[p]]),
                        "skin_gap_cm_at_main": L1._cm(rec["human_pair_gap"][ex, col[p]]),
                        "avatar_gap_cm_at_main": avatar_gap(ex, p)} for p in cands],
    }


def informed_item(item: dict, hold_v2: dict) -> dict:
    """The queue item with labels v2's exemplar as the main moment and its window's ends among the key frames."""
    w = dict(item["window"])
    w["key_frames"] = sorted(set(w["key_frames"]) | {w["frame_hold"], hold_v2["frame_start"], hold_v2["frame_end"],
                                                     hold_v2["frame_hold"]})
    w["frame_hold"] = int(hold_v2["frame_hold"])
    zones = sorted(set(item["zones"]) | {p[:-2] for p in hold_v2["pairs_ground"]}, key=ZONE_ORDER.index)
    return {**item, "item_id": f"{hold_v2['hold_id']}#B4", "window": w, "zones": zones}


def build_packet_b(item: dict, record: dict, rec: capture.Capture, hold_v2: dict, *, manifest: Path, index: dict,
                   review_root: Path = ids.REVIEW_ROOT, index_root: Path = packets.INDEX_ROOT,
                   force: bool = False, purpose: str = "b6") -> tuple[dict, bool]:
    """An informed packet of one hold on render_v4 (call inside ``render_v4.plant_context()``; ``rec`` is its
    capture store v4 record, read outside it)."""
    start = time.time()
    stem = record["stem"]
    item = informed_item(item, hold_v2)
    mpath = ids.motion_path(stem, motion_dir("reference"))
    key = _key(item, record, [mpath] + [p for p in [ids.mosh_path(stem)] if p], manifest,
               extra={"contract": CONTRACT_B, "hold_v2": hold_v2, "store": capture_v4.identity(rec)})
    old = index.get(item["item_id"])
    if not force and old and old["build_key"] == key and packets._complete(old, review_root):
        return old, False
    clip = variant_clip(stem, "reference")
    ex = int(hold_v2["frame_hold"])
    frames = packets.strip_frames(item, clip.num_frames)
    sheets = render.plan(clip, ex, tuple(item["zones"]), frames)
    if len(sheets) > packets.MAX_IMAGES:
        raise ValueError(f"{len(sheets)} images > {packets.MAX_IMAGES}")
    rendered = render_v4.render_sheets(clip, sheets)
    staging = Path(index_root) / RENDER_V / ".staging_informed" / hashlib.sha1(item["item_id"].encode()).hexdigest()
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        images = _stage(rendered, staging)
        body = informed_content(item, record, rec, hold_v2, clip, sheets, images, manifest)
        packet_id, final, total = _finish(staging, body, review_root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    entry = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [mpath, manifest, render_v4.__file__]),
             "item_id": item["item_id"], "packet_id": packet_id, "render_v": RENDER_V, "pass": "B", "contract": CONTRACT_B,
             "dir": str(final), "gl": render.gl_info(), "hold_id": record["hold_id"], "stem": stem,
             "audit_id": record["audit_id"], "purpose": purpose, "stratum": None, "frame_hold": ex,
             "source_frame_hold": hold_v2["labels"]["source"]["frame_hold"], "strip": frames,
             "candidates": [c["pair"] for c in body["candidates"]], "images": {im["file"]: im["sha256"] for im in images},
             "bytes": total, "build_key": key, "seconds": round(time.time() - start, 2)}
    index[item["item_id"]] = entry
    return entry, True


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #
def build_batch_a(items: list[dict], records: dict, manifest: Path, **kw) -> tuple[list[dict], list[str], int]:
    path = index_path("A", kw.get("index_root", packets.INDEX_ROOT))
    index = packets.read_index(path)
    entries, failures, built = [], [], 0
    with render_v4.plant_context():
        for item in items:
            try:
                e, new = build_packet_a(item, records[item["hold_id"]], manifest=manifest, index=index, **kw)
                entries.append(e)
                built += new
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{item['item_id']}: {type(exc).__name__}: {exc}")
    packets.write_index(path, index)
    return entries, failures, built


def build_batch_b(jobs: list[tuple], manifest: Path, **kw) -> tuple[list[dict], list[str], int]:
    """``jobs``: ``(item, record, store v4 record, labels v2 hold, purpose)``; stores are read by the caller,
    outside the plant context."""
    path = index_path("B", kw.get("index_root", packets.INDEX_ROOT))
    index = packets.read_index(path)
    entries, failures, built = [], [], 0
    with render_v4.plant_context():
        for item, record, rec, hold_v2, purpose in jobs:
            try:
                e, new = build_packet_b(item, record, rec, hold_v2, manifest=manifest, index=index, purpose=purpose, **kw)
                entries.append(e)
                built += new
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{item['item_id']}: {type(exc).__name__}: {exc}")
    packets.write_index(path, index)
    return entries, failures, built
