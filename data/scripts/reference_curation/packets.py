# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence packets (BUILD_PLAN Step 3): a queue item's rendered views, as the reviewer reads them.

A packet is ``REVIEW_ROOT/<render_v>/<packet_id>/`` with ``img_1.png`` ... ``img_n.png`` (n <= 6,
each at most 1024 px on its long side) and ``packet.json``. ``REVIEW_ROOT`` is outside the repo, so
``claude -p`` run there loads no ``CLAUDE.md`` (BUILD_PLAN §1).

The images show the item's exemplar (``frame_hold``) in the sheets ``render.plan`` makes: image 1
holds four eye-level views, image 2 the high and the grazing view, image 3 (and 4, past six) the
floor-level insets of the extremities near the floor and of the item's ``zones``, and the last
image a time strip over ``strip_frames(item)``: up to five of the item's ``key_frames``, the
exemplar always among them.

``packet.json``
---------------
* **Pass A (blind).** ``images`` (file, sha256 and a plain-language description of every panel) and
  ``legend``: nothing else. No clip, pose or recording name, label, orientation, frame number, time
  or measurement. ``leaks()`` is the guard, and ``build_packet`` refuses to write a Pass-A packet
  that fails it. The legend's fixed text avoids every word of every clip name in the corpus
  ("scale", "four", "side", "angle", "hold" all occur in one), so it speaks of the "1 m bar".
* **Pass B (informed).** The same images and legend, plus ``hold`` (names, the label ground set and
  every label pair, the window), ``item`` (reasons, severity, questions), ``evidence`` (the audit
  record's exemplar numbers, and per rendered frame the store's human touch states, marker and
  avatar heights and mat loads, gated as ``audit.py`` gates them) and ``units``.

``packet_id`` is the sha1 of ``packet.json``'s canonical content without the id, and that content
lists every image's sha256. A packet is therefore content-addressed: a rebuild of the same inputs
lands on the same directory.

Index
-----
``output/reference_curation/packets/<render_v>/_index.jsonl`` is the private map, one line per
``item_id``: the packet id and directory, hold, stem, audit, the panels' frames and cameras, image
hashes and sizes, the build key and provenance. An item whose build key is unchanged and whose
directory is complete is skipped, so a batch can be resumed.

Items
-----
Items come from Step 2's ``queue.jsonl``. Any hold can also get an item, built by
``audit.build_queue`` from its record (``item_for``). The pilot holds (``PILOT_STEMS``' family holds,
README §6) need that, because several are not in the Pass-A pool (Step 2's card).

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.packets --pass A --pilot
    ... --pass A --top 40          # Step 4's calibration candidates
    ... --pass B --hold <hold_id>  # any hold, by eye
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, capture, ids, render

MODULE = "reference_curation.packets"
SCHEMA_VERSION = 1
INDEX_ROOT = ids.OUTPUT_ROOT / "packets"
MAX_IMAGES = 6
MAX_PACKET_BYTES = 1_500_000   # BUILD_PLAN §1 budget
STRIP_FRAMES = 5
PASSES = ("A", "B")
# README §6's pilot clips; their family holds are the pilot holds.
PILOT_STEMS = (
    "220923_Crane_Crow_Pose_or_Bakasana_-a",
    "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c",
    "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a",
    "220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a",
    "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b",
    "220926_Plow_Pose_or_Halasana_-b",
    "220923_Peacock_Pose_or_Mayurasana_-a",
    "220926_Upward_Plank_Pose_or_Purvottanasana_-a",
    "220926_Warrior_II_Pose_or_Virabhadrasana_II_-a",
    "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a",
)
# Recording-name words every clip shares; the leak check ignores them.
GENERIC_TOKENS = frozenset({"yogi", "body", "hands", "nexus", "pose"})

LEGEND = {
    "what": "Renders of one moment of a motion. A simulated avatar made of rigid capsules, spheres "
            "and boxes shows the reference motion. Small glowing spheres show the motion-capture "
            "markers of the human performer at the same moment.",
    "colours": "Every body segment of the avatar has its own colour (see palette). Left parts are the "
               "dark shade and right parts the light shade of their colour.",
    "markers": "Each marker is coloured like the avatar segment it belongs to. Markers sit on the skin, "
               "so a marker on a limb that rests on the floor is a little above the floor. Markers the "
               "capture fit could not explain are not drawn.",
    "drop_lines": "A thin vertical line in the colour of an avatar part runs from the part's lowest point "
                  "straight down to the floor, and ends in a small cross of the same colour on the floor. Every "
                  "hand, foot and the head "
                  f"between {100 * render.DROP_MIN_M:.0f} cm and {100 * render.DROP_MAX_M:.0f} cm above the floor "
                  f"has one; every other part between {100 * render.DROP_MIN_M:.0f} cm and "
                  f"{100 * render.DROP_MAX_BODY_M:.0f} cm up has one unless another part is below it. A part "
                  f"within {100 * render.DROP_MIN_M:.0f} cm of the floor touches it and has no line.",
    "floor": f"The floor is flat. Its checks are {100 * render.CHECK_M:.0f} cm squares.",
    "bar": f"In the eye-level and high views, a {render.BAR_LENGTH_M:g} m bar of ten alternating black and "
           f"white {100 * render.BAR_LENGTH_M / render.BAR_SEGMENTS:.0f} cm segments lies on the floor.",
    "shadows": "Nothing in the images casts a shadow.",
    "tags": "Every panel carries a tag in its top-left corner: its image number and a letter.",
}
NO_MARKERS = "This packet has no capture markers."
CAMERA_CM = f"{100 * render.FLOOR_CAMERA_M:.0f}"
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
# Numbers the fixed texts use (legend constants, image numbers, moment counts): not evidence.
FIXED_NUMBERS = (frozenset(_NUMBER.findall(json.dumps(LEGEND))) | {str(n) for n in range(1, MAX_IMAGES + 1)}
                 | {CAMERA_CM, f"{render.INSET_WIDTH_M:g}"})


# --------------------------------------------------------------------------- #
# Audit and items
# --------------------------------------------------------------------------- #
def default_audit_dir(manifest: Path = ids.DEFAULT_MANIFEST, rule: str = "calibrated") -> Path:
    """The newest audit of ``manifest`` under ``rule``."""
    found = sorted(audit.AUDITS_DIR.glob(f"{Path(manifest).stem}.{rule}.*/audit.json"),
                   key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no {rule} audit of {Path(manifest).name}; run `-m reference_curation.audit`")
    return found[-1].parent


def load_audit(audit_dir: Path) -> SimpleNamespace:
    audit_dir = Path(audit_dir)
    rows = lambda name: [json.loads(x) for x in (audit_dir / name).read_text().splitlines() if x]  # noqa: E731
    records = rows("holds.jsonl")
    meta = json.loads((audit_dir / "audit.json").read_text())
    return SimpleNamespace(dir=audit_dir, audit_id=meta["audit_id"], manifest=ids.REPO / meta["manifest"],
                           records={r["hold_id"]: r for r in records}, queue=rows("queue.jsonl"))


def pilot_holds(records: dict) -> list[str]:
    """The family holds of ``PILOT_STEMS``, in that order."""
    return [h for s in PILOT_STEMS for h, r in records.items() if r["stem"] == s and r["family_hold"]]


def item_for(record: dict, pass_: str) -> dict:
    """An item for any hold, built as ``audit.build_queue`` builds the queue's (same window,
    key frames and zones). Pass A gets the fixed blind questions and purpose ``pilot``."""
    item = next(i for i in audit.build_queue([record]) if i["pass"] == "B")
    if pass_ == "A":
        item = {**item, "item_id": f"{record['hold_id']}#A", "pass": "A", "purpose": "pilot",
                "questions": [{"id": q} for q in audit.PASS_A_QUESTIONS], "stratum": audit.stratum(record)}
    return {"schema_version": record.get("schema_version"), "audit_id": record["audit_id"], **item}


def select_items(aud: SimpleNamespace, pass_: str, top: int | None = None, pilot: bool = False,
                 holds: list[str] | None = None) -> list[dict]:
    """Queue items of ``pass_`` in priority order (``top`` of them), plus the pilot holds and
    ``holds``; a hold the queue lacks gets ``item_for``. Each item appears once."""
    queue = sorted((i for i in aud.queue if i["pass"] == pass_), key=lambda i: i["priority"])
    by_hold = {i["hold_id"]: i for i in queue}
    chosen = queue[:top] if top is not None else []
    wanted = (pilot_holds(aud.records) if pilot else []) + list(holds or [])
    for h in wanted:
        if h not in aud.records:
            raise KeyError(f"{h} is not a hold of audit {aud.audit_id}")
        chosen.append(by_hold.get(h) or item_for(aud.records[h], pass_))
    seen, out = set(), []
    for item in chosen:
        if item["item_id"] not in seen:
            seen.add(item["item_id"])
            out.append(item)
    return out


def strip_frames(item: dict, num_frames: int, n: int = STRIP_FRAMES) -> list[int]:
    """Up to ``n`` frames for the time strip: the item's key frames plus the exemplar. Key frames
    closer than the audit's boundary tolerance (0.25 s) merge into the earlier one, or into the
    exemplar. More than ``n`` are thinned to the first, the last, the exemplar and the ones farthest
    from those; fewer are topped up with the frames farthest from the chosen ones. A window shorter
    than ``n`` frames is widened by 0.5 s each side."""
    w = item["window"]
    hold, fps = int(w["frame_hold"]), int(w["fps"])
    raw = sorted({int(f) for f in w["key_frames"]} | {hold})
    tol = round(audit.BOUNDARY_TOL_S * fps)
    keys: list[int] = []
    for f in raw:
        if keys and f - keys[-1] < tol:
            if f == hold:
                keys[-1] = hold
            continue
        keys.append(f)
    lo, hi = raw[0], raw[-1]
    if hi - lo < n - 1:
        lo, hi = max(0, lo - fps // 2), min(num_frames - 1, hi + fps // 2)
    chosen = set(keys) if len(keys) <= n else {keys[0], keys[-1], hold}
    pool = keys if len(keys) > n else range(lo, hi + 1)
    while len(chosen) < min(n, len(pool)):
        # the pool frame farthest from every chosen one; ties go to the earlier frame
        chosen.add(max(pool, key=lambda f: (min(abs(f - c) for c in chosen), -f)))
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Packet content
# --------------------------------------------------------------------------- #
def _words(zones) -> str:
    names = [render.ZONE_WORDS[z] for z in zones]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def describe(sheets: list, numbers: list[int], hold_frame: int) -> list[dict]:
    """Pass-A text for every image: what its panels show, without a number that is evidence."""
    first_eye = f"{numbers[0]}a"
    body_images = [n for s, n in zip(sheets, numbers) if s.kind != "strip"]
    out = []
    for sheet, number in zip(sheets, numbers):
        tags = [f"{number}{chr(ord('a') + k)}" for k in range(len(sheet.panels))]
        if sheet.kind == "eye":
            text = ("The whole avatar at eye level, looking slightly down. From panel a to panel d the camera "
                    "moves a quarter turn anticlockwise around the avatar, seen from above. Panel a looks "
                    "across the avatar's long horizontal axis.")
            panels = {t: "eye level" for t in tags}
        elif sheet.kind == "high_grazing":
            text = ("Panel a: the whole avatar from high above, looking steeply down. Panel b: the whole avatar "
                    f"from floor level. Its camera is {CAMERA_CM} cm above the floor and tilted up, so a gap "
                    "between a body part and the floor shows as floor seen under it. It looks the same way as "
                    f"panel {first_eye}.")
            panels = {tags[0]: "high", tags[1]: "floor level"}
        elif sheet.kind == "insets":
            text = (f"Close-ups at floor level, each {render.INSET_WIDTH_M:g} m wide where it is centred, with "
                    f"the camera {CAMERA_CM} cm above the floor looking in toward the avatar's middle. Each panel "
                    "is centred under the parts it names.")
            panels = {t: f"close-up under the {_words(p.zones)}" for t, p in zip(tags, sheet.panels)}
        else:
            frames = [p.frame for p in sheet.panels]
            now = tags[frames.index(hold_frame)]
            text = (f"The avatar from the direction of panel {first_eye}, framed to fit every moment, at "
                    f"{len(frames)} moments in time order: panel a is the earliest and panel {tags[-1][-1]} the "
                    f"latest. Images {body_images[0]} to {body_images[-1]} show the moment of panel {now}.")
            panels = {t: "time strip" for t in tags}
        out.append({"text": text, "panels": panels})
    return out


def _cm(x) -> float | None:
    x = float(x)
    return round(100.0 * x, 1) if np.isfinite(x) else None


def frame_evidence(rec: capture.Capture, frame: int, zones) -> dict:
    """Pass-B numbers of one frame from the store, gated as ``audit.py`` gates them."""
    a = rec.arrays
    cov = float(np.nan_to_num(a["mat_valid_cov"][frame], nan=0.0)) >= audit.COVERAGE_GATE
    row = {"frame": frame, "t_s": round(frame / rec.fps, 3), "zones": {}}
    for z in zones:
        zi = ZONE_ORDER.index(z)
        row["zones"][z] = {
            "human": int(a["ground_state"][frame, zi]),
            "marker_cm": _cm(a["marker_min_z"][frame, zi]),
            "avatar_cm": _cm(a["avatar_min_z"][frame, zi]),
            "load_n": (round(float(a["mat_zone_load"][frame, zi]), 1)
                       if cov and a["attr_visible"][frame, zi] else None)}
    row["mat_total_n"] = round(float(a["mat_total"][frame]), 1) if cov else None
    row["mat_unexplained_n"] = round(float(a["mat_unexplained"][frame]), 1) if cov else None
    return row


def manifest_hold(record: dict, manifest: Path) -> tuple[dict, dict]:
    """``(clip, hold)`` entries of ``manifest`` behind an audit record, checked against it."""
    clip = next(c for c in ids.load_manifest(manifest)["clips"] if c["stem"] == record["stem"])
    hold = clip["holds"][record["hold_index"]]
    if int(hold["frame_hold"]) != record["window"]["frame_hold"] or hold["name"] != record["name"]:
        raise ValueError(f"{record['hold_id']}: {Path(manifest).name} changed since the audit")
    return clip, hold


def informed(item: dict, record: dict, rec: capture.Capture, frames: list[int], tags: dict,
             manifest: Path = ids.DEFAULT_MANIFEST) -> dict:
    """The Pass-B additions: labels, the item's questions and the numeric companions."""
    clip, hold = manifest_hold(record, manifest)
    zones = [z for z in ZONE_ORDER if z in set(item["zones"]) | set(render.EXTREMITIES)]
    return {
        "hold": {"hold_id": record["hold_id"], "stem": record["stem"], "name": record["name"],
                 "family": clip.get("family"), "group": record["group"], "orientation": record["orientation"],
                 "family_hold": record["family_hold"], "label_ground": record["label_ground"],
                 "label_pairs": list(hold["pairs"]), "window": record["window"]},
        "item": {k: item.get(k) for k in ("item_id", "purpose", "severity", "reasons", "questions")},
        "audit_id": record["audit_id"],
        "evidence": {"exemplar": {k: record[k] for k in ("capture", "zones", "mat", "disagreements")},
                     "frames": [dict(frame_evidence(rec, f, zones), panels=tags.get(f, [])) for f in frames]},
        "units": {"*_cm": "centimetres above the floor", "*_n": "newtons, over mat-coverage-valid frames",
                  "human": "the capture's touch state: 1 contact, 0 separated, -1 unknown or undecided",
                  "panels": "tags of the panels that show the frame"},
    }


def forbidden(record: dict, manifest: Path = ids.DEFAULT_MANIFEST) -> SimpleNamespace:
    """What a Pass-A packet must not contain: ``words`` (case-insensitive) and ``exact`` strings."""
    clip, hold = manifest_hold(record, manifest)
    names = [record["stem"], record["hold_id"], record["name"], clip.get("family") or "",
             ids.recording_id(record["stem"]) or ""]
    words = {n.lower() for n in names if n}
    words |= {t.lower() for n in names for t in re.split(r"[_\-()\s]+", n)
              if len(t) >= 4 and re.search("[A-Za-z]", t)} - GENERIC_TOKENS
    words.add(str(hold.get("orientation") or "").lower())
    exact = set(hold["pairs"]) | {p.split(":")[0] for p in hold["pairs"] if ":" in p}
    exact |= {z for p in hold["pairs"] for z in re.split(r"[+:]", p) if z in ZONE_ORDER}
    return SimpleNamespace(words=sorted(w for w in words if w), exact=sorted(exact))


def leaks(text: str, record: dict, item: dict, manifest: Path = ids.DEFAULT_MANIFEST) -> list[str]:
    """Everything in ``text`` a Pass-A packet may not carry: names, labels, and the item's frame
    numbers and times as standalone numbers. A number the fixed legend or view texts use (2, 10,
    60, ...) cannot be told apart, so it is not counted."""
    bad = forbidden(record, manifest)
    low = text.lower()
    hits = [w for w in bad.words if w in low] + [e for e in bad.exact if e in text]
    w = item["window"]
    numbers = set(_NUMBER.findall(text)) - FIXED_NUMBERS
    frames = set(w["key_frames"]) | {w["frame_hold"], w["frame_start"], w["frame_end"]}
    hits += [f"frame {f}" for f in sorted(frames) if str(f) in numbers]
    times = [f / w["fps"] for f in frames] + [w.get(k) for k in ("t_start", "t_hold", "t_end")]
    hits += [f"time {t:g}" for t in sorted({round(t, 3) for t in times if t is not None})
             if any(f"{t:.{d}f}" in numbers for d in (1, 2, 3))]
    return hits


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_key(item: dict, record: dict, strip: bool, clip: render.Clip, manifest: Path,
              store: capture.Capture | None = None) -> str:
    """Everything a packet's content depends on: code, render version, GL renderer, item and inputs;
    for Pass B also the record and the capture store record its numbers come from."""
    stable = {k: v for k, v in item.items() if k not in ("priority", "schema_version")}
    inputs = [ids.MJCF, ids.motion_path(clip.stem), manifest] + [p for p in [ids.mosh_path(clip.stem)] if p]
    informed_by = None
    if item["pass"] == "B":
        informed_by = {"record": record, "store": {k: store.meta[k] for k in ("generator", "calibration", "inputs")}}
    return ids.sha256_json({"render_v": render.RENDER_V, "render": ids.sha256_file(render.__file__),
                            "packets": ids.sha256_file(__file__), "gl": render.gl_info(), "strip": strip,
                            "item": stable, "informed_by": informed_by,
                            "inputs": {ids.display_path(p): ids.sha256_file(p) for p in inputs}})


def index_path(render_v: str = render.RENDER_V, index_root: Path = INDEX_ROOT) -> Path:
    return Path(index_root) / render_v / "_index.jsonl"


def read_index(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    return {e["item_id"]: e for e in (json.loads(x) for x in Path(path).read_text().splitlines() if x)}


def write_index(path: Path, entries: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(entries[k], sort_keys=True) + "\n" for k in sorted(entries)))
    tmp.replace(path)


def packet_dir(entry: dict, review_root: Path = ids.REVIEW_ROOT) -> Path:
    return Path(review_root) / entry["render_v"] / entry["packet_id"]


def _complete(entry: dict | None, review_root: Path) -> bool:
    if not entry:
        return False
    d = packet_dir(entry, review_root)
    return (d / "packet.json").exists() and all((d / f).exists() for f in entry["images"])


def build_packet(item: dict, record: dict, *, manifest: Path = ids.DEFAULT_MANIFEST,
                 review_root: Path = ids.REVIEW_ROOT, index_root: Path = INDEX_ROOT, strip: bool = True,
                 force: bool = False, index: dict | None = None, store_dir: Path = capture.STORE_DIR,
                 calibration: dict | None = None) -> tuple[dict, bool]:
    """Render ``item`` into a packet and return ``(index entry, built)``. ``built`` is False when
    ``index`` already holds a complete packet with the same build key. The packet is assembled in
    ``index_root`` and moved to ``review_root`` only once it passes the leak and budget checks.
    Pass B reads its numbers from the capture store at ``store_dir``."""
    pass_ = item["pass"]
    if pass_ not in PASSES:
        raise ValueError(f"unknown pass {pass_!r}")
    start = time.time()
    clip = render.load_clip(record["stem"])
    store = capture.load(record["stem"], store_dir, calibration) if pass_ == "B" else None
    key = build_key(item, record, strip, clip, manifest, store)
    index = read_index(index_path(render.RENDER_V, index_root)) if index is None else index
    old = index.get(item["item_id"])
    if not force and old and old["build_key"] == key and _complete(old, review_root):
        return old, False

    hold = int(item["window"]["frame_hold"])
    frames = strip_frames(item, clip.num_frames) if strip else None
    sheets = render.plan(clip, hold, tuple(item["zones"]), frames)
    if len(sheets) > MAX_IMAGES:
        raise ValueError(f"{len(sheets)} images > {MAX_IMAGES}")
    numbers = list(range(1, len(sheets) + 1))
    rendered = render.render_sheets(clip, sheets)
    staging = Path(index_root) / render.RENDER_V / ".staging" / hashlib.sha1(item["item_id"].encode()).hexdigest()
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        images = []
        for n, (image, _) in zip(numbers, rendered):
            if max(image.size) > render.SHEET_MAX_PX:
                raise ValueError(f"image {n} is {image.size}, over {render.SHEET_MAX_PX} px")
            path = staging / f"img_{n}.png"
            render.save_png(image, path)  # hashed from its bytes: ids' cache keys on (path, size, mtime)
            images.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        content = {"schema_version": SCHEMA_VERSION, "render_v": render.RENDER_V, "pass": pass_,
                   "images": [dict(im, **t) for im, t in zip(images, describe(sheets, numbers, hold))],
                   "legend": dict(LEGEND, palette=render.palette_legend(),
                                  **({} if clip.markers is not None else {"markers": NO_MARKERS}))}
        tags = {}
        for sheet, n in zip(sheets, numbers):
            for k, p in enumerate(sheet.panels):
                tags.setdefault(p.frame, []).append(f"{n}{chr(ord('a') + k)}")
        if pass_ == "B":
            content.update(informed(item, record, store, sorted(tags), tags, manifest))
        else:
            found = leaks(json.dumps(content), record, item, manifest)
            if found:
                raise ValueError(f"pass-A packet would leak {found}")
        packet_id = hashlib.sha1(_canonical(content)).hexdigest()
        (staging / "packet.json").write_text(json.dumps({"packet_id": packet_id, **content}, indent=1,
                                                        sort_keys=True) + "\n")
        total = sum(f.stat().st_size for f in staging.iterdir())
        if total > MAX_PACKET_BYTES:
            raise ValueError(f"packet is {total} bytes > budget {MAX_PACKET_BYTES}")
        final = Path(review_root) / render.RENDER_V / packet_id
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(final, ignore_errors=True)  # content-addressed: same id, same content
        shutil.move(str(staging), str(final))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    inputs = [ids.MJCF, ids.motion_path(clip.stem), manifest, render.__file__]
    inputs += [p for p in [ids.mosh_path(clip.stem)] if p]
    entry = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs),
             "item_id": item["item_id"], "packet_id": packet_id, "render_v": render.RENDER_V, "pass": pass_,
             "dir": str(final), "gl": render.gl_info(),
             "hold_id": record["hold_id"], "stem": record["stem"], "audit_id": record["audit_id"],
             "purpose": item.get("purpose"), "stratum": item.get("stratum"), "frame_hold": hold, "strip": frames,
             "images": {im["file"]: im["sha256"] for im in images}, "bytes": total,
             "sheets": [{"image": f"img_{n}.png", "kind": s.kind,
                         "panels": [{"tag": b["tag"], "kind": p.kind, "frame": p.frame, "zones": list(p.zones),
                                     "target": p.target, "camera": p.camera.record(), "box": b["box"]}
                                    for p, b in zip(s.panels, boxes)]}
                        for s, n, (_, boxes) in zip(sheets, numbers, rendered)],
             "build_key": key, "seconds": round(time.time() - start, 2)}
    index[item["item_id"]] = entry
    return entry, True


def build_packets(items: list[dict], records: dict, *, index_root: Path = INDEX_ROOT,
                  **kw) -> tuple[list[dict], list[str], int]:
    """``(entries, failures, built)`` over ``items``; ``kw`` goes to ``build_packet``. The index is
    rewritten once, at the end."""
    path = index_path(render.RENDER_V, index_root)
    index = read_index(path)
    entries, failures, built = [], [], 0
    for item in items:
        try:
            entry, new = build_packet(item, records[item["hold_id"]], index_root=index_root, index=index, **kw)
            entries.append(entry)
            built += new
        except Exception as exc:  # noqa: BLE001 -- report every broken item, then fail
            failures.append(f"{item['item_id']}: {type(exc).__name__}: {exc}")
    write_index(path, index)
    return entries, failures, built


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pass", dest="pass_", choices=PASSES, default="A")
    ap.add_argument("--audit", type=Path, help="an audit directory (default: the newest calibrated audit)")
    ap.add_argument("--top", type=int, help="the first N queue items of the pass, in priority order")
    ap.add_argument("--all", action="store_true", help="every queue item of the pass")
    ap.add_argument("--pilot", action="store_true", help="the pilot holds (README §6)")
    ap.add_argument("--hold", nargs="*", default=[], help="these holds, queued or not")
    ap.add_argument("--no-strip", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild even if the index is current")
    ap.add_argument("--review-root", type=Path, default=ids.REVIEW_ROOT)
    ap.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    args = ap.parse_args(argv)

    start = time.time()
    try:
        aud = load_audit(args.audit or default_audit_dir())
        top = None if args.all else args.top
        items = select_items(aud, args.pass_, len(aud.queue) if args.all else top, args.pilot, args.hold)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not items:
        print("FAILED nothing selected: pass --pilot, --top N, --all or --hold", file=sys.stderr)
        return 1
    entries, failures, built = build_packets(items, aud.records, manifest=aud.manifest, review_root=args.review_root,
                                             index_root=args.index_root, strip=not args.no_strip, force=args.force)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    mb = sum(e["bytes"] for e in entries) / 1e6
    print(f"packets {render.RENDER_V} pass {args.pass_} from {aud.audit_id}: {len(entries)} of {len(items)} ready "
          f"({built} built, {len(entries) - built} current), {len(failures)} failed, {mb:.1f} MB in "
          f"{time.time() - start:.0f} s -> {args.review_root / render.RENDER_V} "
          f"(index {ids.display_path(index_path(render.RENDER_V, args.index_root))})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
