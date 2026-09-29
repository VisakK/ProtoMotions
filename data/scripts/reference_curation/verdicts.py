# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verdicts (BUILD_PLAN Step 4): the Pass-A answer contract, its validator, the verdict ledger and
the calibration that measures how far each kind of claim can be trusted.

Contract
--------
``schemas/pass_a.json`` is the answer the blind reviewer returns (``claude -p --json-schema``) and
``prompts/pass_a.md`` the question it is asked. Both reach the reviewer, so neither may contain a
word of any clip name (the tests scan them). Body parts are named with the packets' own words
(``render.ZONE_WORDS``: "left foot", "right shin", ...); ``PARTS`` maps them back to zones.

Validator
---------
``validate(answer, packet)`` returns ``(errors, warnings)``. An error rejects the verdict: the JSON
schema (types, enums, required and extra fields, length limits); a panel tag or image file the
packet does not have; a body-body contact naming one part twice, or a pair twice; contacts listed
under ``cannot_tell``; any force claim (newtons, "force", a percentage of body weight). Warnings
stay with the verdict: load or weight words, and body-body pairs of parts joined at a joint (which
scoring ignores).

Ledger
------
``data/reference_curation/ledger/verdicts/<packet_id>.<pass>.<n>.json``: one file per reviewer
call, valid or not, since each one costs money. It holds the answer, the reviewer (model, effort,
CLI version, config ``key``), the prompt, schema, packet and image sha256, and cost, duration and
tokens. ``n`` counts a packet's calls in one pass from 1. Files are created exclusively and never
rewritten. Everything downstream replays the ledger, and the runner never re-queries a packet that
has a valid verdict under the same reviewer key.

Truth (Pass A)
--------------
Truth covers only what the evidence settles at the packet's main moment (``frame_hold``):

* ``human[zone]``: the capture's touch state, for the zones Step 1 admits (feet, hands and head).
  It is used only when it is *decisive* (the marker height lies outside the zone's hysteresis
  band) and *stable* (the zone does not flip within ±0.5 s). Other zones have no human truth
  until the mesh (Step 5).
* ``avatar[zone]``: the reference's lowest collision surface, read three ways:
  - touching at ≤ 1 cm;
  - off the floor at ≥ 2 cm (README §3.1's hover threshold);
  - undecided in between.
* ``pairs[(a, b)]``: the reference's non-adjacent zone pairs, from their surface gap:
  - in contact at ≤ 1 cm (penetration included);
  - apart at ≥ 3 cm;
  - undecided in between.

  A pair whose bounding spheres are 10 cm apart keeps that lower bound as its gap, 7x faster and
  with identical truth.
* ``pose``: the hold's name without its ``_h<k>`` suffix, which is its family or ``standing``.
  A candidate name counts when ``resolve_pose`` maps it to that pose. It maps a candidate to the
  pose whose alias is its longest contiguous match, after lowercasing, stripping accents and
  folding Sanskrit ``h`` / ``ri`` spellings. A tie between two poses stays unresolved.

Claim classes
-------------
=================  ==========================================  =====================================
class              items                                       precision
=================  ==========================================  =====================================
floor_avatar       (packet, zone), reference decided, ≤ 15 cm  min over the two claims, touching / off
floor_avatar_far   the same, above 15 cm (a sanity check)      min over the two claims
floor_human        (packet, zone) with human truth             min over the two claims (markers)
float              (packet, zone) the human touches, the       of "off_floor" claims, i.e. "floats"
                   reference decided
body_body          pairs truly in contact, plus every pair     of pairs listed with certainty "clear"
                   the reviewer lists with decided truth
pose_identity      packets with a pose truth                   of the first candidate name (also
                                                               reported over family holds only)
left_right         (packet, part, avatar or human) whose two   answered = both sides decisive and
                   sides' truths differ                        different; correct = not mirrored
=================  ==========================================  =====================================

``cannot_tell`` and ``no_markers`` are abstentions, and so is a ``possible`` body-body contact.

A class is **evidence** when (BUILD_PLAN Step 4):
- its precision is ≥ 0.9, measured over ≥ 30 of its own claims (``support``);
- its answered (non-abstained) rate is ≥ 0.5, over ≥ 30 items.

Otherwise it stays **advisory**. The claims floor matters for a detect class, whose answered rate
also counts its implicit "no" claims. Without it, render_v3 at high effort admitted body-body on 4
positive claims. For a two-valued class the precision is the lower of its two claim values'
precisions, and ``support`` the smaller of their claim counts, so a majority value cannot hide a
broken or untested one. Every precision also carries its 95 % Wilson lower bound.

``calibrate()`` replays the ledger:
- one verdict per (packet, reviewer key): the lowest valid ``n``;
- scored against truth rebuilt from the audit, the capture store and the shipped reference.

It writes ``data/reference_curation/calibration/<render_v>.json``. The per-item rows go to
``output/reference_curation/calibration/<render_v>/``.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.verdicts
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import jsonschema
import numpy as np
import torch

from contact_geometry import geom_pair_distance, geom_to_world
from extract_contact_configs import ADJACENT, ZONE_ORDER, ZONES
from reference_curation import audit, capture, ids, packets, render

MODULE = "reference_curation.verdicts"
SCHEMA_VERSION = 1
PACKAGE_DIR = Path(__file__).resolve().parent
PASSES = ("A",)
LEDGER_DIR = ids.DATA_ROOT / "ledger" / "verdicts"
CALIBRATION_DIR = ids.DATA_ROOT / "calibration"
ITEMS_ROOT = ids.OUTPUT_ROOT / "calibration"

PARTS = {render.ZONE_WORDS[z]: z for z in ZONE_ORDER}   # the reviewer's words -> zones
ZI = {z: i for i, z in enumerate(ZONE_ORDER)}
CLAIM = {"touching": 1, "off_floor": 0}                  # every other answer abstains
SIDED = ("FOOT", "SHANK", "THIGH", "UPPER_ARM", "FOREARM", "HAND")
BODY_PAIRS = tuple((a, b) for i, a in enumerate(ZONE_ORDER) for b in ZONE_ORDER[i + 1:]
                   if frozenset((a, b)) not in ADJACENT)

AVATAR_TOUCH_CM = 1.0
AVATAR_OFF_CM = 2.0        # README §3.1
NEAR_FLOOR_CM = 100 * audit.FLOAT_MAX_M
PAIR_TOUCH_CM = 1.0
PAIR_APART_CM = 3.0
PAIR_GATE_M = 0.10         # exact pair gaps below this; a far pair keeps its bounding-sphere lower bound
PRECISION_MIN = 0.9
ANSWERED_MIN = 0.5
ITEMS_MIN = 30
TRUTH = {"avatar_touch_max_cm": AVATAR_TOUCH_CM, "avatar_off_min_cm": AVATAR_OFF_CM,
         "near_floor_max_cm": NEAR_FLOOR_CM, "pair_touch_max_cm": PAIR_TOUCH_CM,
         "pair_apart_min_cm": PAIR_APART_CM, "human": "capture ground_state, decisive and stable +-0.5 s"}
CLAIMS_MIN = 30           # the claims a precision rests on (render_v3 high "admitted" body-body on 4)
RULE = {"precision_min": PRECISION_MIN, "answered_rate_min": ANSWERED_MIN, "items_min": ITEMS_MIN,
        "claims_min": CLAIMS_MIN}
CLASSES = {"floor_avatar": "binary", "floor_avatar_far": "binary", "floor_human": "binary",
           "float": "detect", "body_body": "detect", "pose_identity": "match", "left_right": "match"}

_FORCE_WORDS = re.compile(r"\b(?:kilo)?newtons?\b|\bforces?\b|\bkgf\b|\blbf\b"
                          r"|\d\s?%\s*(?:of\s+)?(?:the\s+|its\s+)?(?:body\s?weight|weight|bw)\b", re.I)
_FORCE_UNITS = re.compile(r"\b\d+(?:\.\d+)?\s?k?N\b")
_LOAD_WORDS = re.compile(r"\b(?:load(?:s|ed|ing)?|weight(?:s|ed)?|bear(?:s|ing)?)\b", re.I)
_IMAGE_REF = re.compile(r"\bimg_\d+\.png\b")


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def prompt_path(pass_: str = "A") -> Path:
    return PACKAGE_DIR / "prompts" / f"pass_{pass_.lower()}.md"


def schema_path(pass_: str = "A") -> Path:
    return PACKAGE_DIR / "schemas" / f"pass_{pass_.lower()}.json"


@functools.lru_cache(maxsize=4)
def _schema(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text())


def load_schema(pass_: str = "A") -> dict:
    p = schema_path(pass_)
    return _schema(str(p), p.stat().st_mtime_ns)


def schema_text(pass_: str = "A") -> str:
    """The schema as the reviewer's CLI gets it: compact JSON."""
    return json.dumps(load_schema(pass_), separators=(",", ":"))


def panel_tags(packet: dict) -> set:
    return {tag for image in packet["images"] for tag in image["panels"]}


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def validate(answer, packet: dict, pass_: str = "A") -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` of a reviewer's answer to ``packet`` (its ``packet.json``)."""
    validator = jsonschema.Draft202012Validator(load_schema(pass_))
    errors = [f"schema: /{'/'.join(map(str, e.absolute_path))}: {e.message}"
              for e in sorted(validator.iter_errors(answer), key=lambda e: list(map(str, e.absolute_path)))]
    if errors or not isinstance(answer, dict):
        return errors or ["schema: not an object"], []
    warnings = []
    tags, files = panel_tags(packet), {im["file"] for im in packet["images"]}
    cited = [t for part in answer["floor"].values() for t in part["panels"]]
    for key in ("discrepancies", "implausible"):
        cited += [t for x in answer[key] for t in x["panels"]]
    cited += [t for c in answer["body_body"]["contacts"] for t in c["panels"]]
    errors += [f"panel {t} is not in the packet" for t in sorted(set(cited) - tags)]
    for text in _strings(answer):
        errors += [f"image {f} is not in the packet" for f in _IMAGE_REF.findall(text) if f not in files]
        found = _FORCE_WORDS.findall(text) + _FORCE_UNITS.findall(text)
        errors += [f"force claim {m!r}" for m in found]
        warnings += [f"load word {m!r}" for m in _LOAD_WORDS.findall(text)]
    bb, seen = answer["body_body"], set()
    if bb["answer"] == "cannot_tell" and bb["contacts"]:
        errors.append("body_body: contacts listed under cannot_tell")
    for c in bb["contacts"]:
        a, b = PARTS[c["part_a"]], PARTS[c["part_b"]]
        pair = frozenset((a, b))
        if a == b:
            errors.append(f"body_body: {c['part_a']} paired with itself")
        elif pair in seen:
            errors.append(f"body_body: {c['part_a']} + {c['part_b']} listed twice")
        elif pair in ADJACENT:
            warnings.append(f"body_body: {c['part_a']} + {c['part_b']} are joined; not scored")
        seen.add(pair)
    return errors, sorted(set(warnings))


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def verdict_path(ledger_dir: Path, packet_id: str, pass_: str, n: int) -> Path:
    return Path(ledger_dir) / f"{packet_id}.{pass_}.{n}.json"


def read_ledger(ledger_dir: Path = LEDGER_DIR) -> list[dict]:
    """Every verdict record, ordered by (packet, pass, n)."""
    records = [json.loads(p.read_text()) for p in sorted(Path(ledger_dir).glob("*.json"))]
    return sorted(records, key=lambda r: (r["packet_id"], r["pass"], r["n"]))


def next_n(ledger_dir: Path, packet_id: str, pass_: str) -> int:
    taken = [int(p.name.split(".")[-2]) for p in Path(ledger_dir).glob(f"{packet_id}.{pass_}.*.json")]
    return max(taken, default=0) + 1


def write_verdict(record: dict, ledger_dir: Path = LEDGER_DIR) -> Path:
    """Write ``record`` as the packet's next call; never overwrites (``n`` moves on instead)."""
    Path(ledger_dir).mkdir(parents=True, exist_ok=True)
    while True:
        n = next_n(ledger_dir, record["packet_id"], record["pass"])
        path = verdict_path(ledger_dir, record["packet_id"], record["pass"], n)
        try:
            with open(path, "x") as f:
                f.write(json.dumps({**record, "n": n}, indent=1, sort_keys=True) + "\n")
            return path
        except FileExistsError:  # another runner took this n
            continue


# --------------------------------------------------------------------------- #
# Pose names
# --------------------------------------------------------------------------- #
# Every pose truth in the manifest (the families, plus "standing"), with the names a reviewer may
# use. A candidate resolves to the pose of its longest matching alias.
POSE_ALIASES = {
    # MOYO takes open and close in the capture's T-pose, which is what the "standing" holds show
    "standing": ["standing", "stand", "tadasana", "mountain", "samasthiti", "t pose"],
    "Half_Moon_Pose_or_Ardha_Chandrasana": ["half moon", "ardha chandrasana"],
    "Standing_big_toe_hold_pose_or_Utthita_Padangusthasana": [
        "standing big toe", "big toe hold", "hand to big toe", "utthita hasta padangusthasana",
        "utthita padangusthasana", "hasta padangusthasana"],
    "Side_Crane_Crow_Pose_or_Parsva_Bakasana": ["side crow", "side crane", "parsva bakasana"],
    "Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana": ["standing split", "urdhva prasarita eka padasana"],
    "Tree_Pose_or_Vrksasana": ["tree", "vrksasana"],
    "Lord_of_the_Dance_Pose_or_Natarajasana": ["lord of the dance", "dancer", "natarajasana"],
    "Warrior_III_Pose_or_Virabhadrasana_III": ["warrior iii", "virabhadrasana iii"],
    "Plow_Pose_or_Halasana": ["plow", "plough", "halasana"],
    "Scorpion_pose_or_vrischikasana": ["scorpion", "vrischikasana", "vrschikasana"],
    "Supported_Headstand_pose_or_Salamba_Sirsasana": [
        "headstand", "head stand", "sirsasana", "salamba sirsasana", "tripod headstand"],
    "Firefly_Pose_or_Tittibhasana": ["firefly", "tittibhasana"],
    "Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II": [
        "koundinya", "kaundinya", "koundinyasana", "kaundinyasana", "koundinyanasana"],
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana": [
        "downward dog", "downward facing dog", "down dog", "adho mukha svanasana"],
    "Side_Plank_Pose_or_Vasisthasana": ["side plank", "vasisthasana"],
    "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana": [
        "triangle", "trikonasana", "revolved triangle", "parivrtta trikonasana"],
    "Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana": [
        "side angle", "extended side angle", "revolved side angle", "parsvakonasana", "parivrtta parsvakonasana"],
    "Eagle_Pose_or_Garudasana": ["eagle", "garudasana"],
    "Handstand_pose_or_Adho_Mukha_Vrksasana": ["handstand", "hand stand", "adho mukha vrksasana"],
    "Legs-Up-the-Wall_Pose_or_Viparita_Karani": ["legs up the wall", "viparita karani"],
    "Supported_Shoulderstand_pose_or_Salamba_Sarvangasana": [
        "shoulderstand", "shoulder stand", "sarvangasana", "salamba sarvangasana"],
    "Cockerel_Pose": ["cockerel", "rooster", "kukkutasana"],
    "Crane_Crow_Pose_or_Bakasana": ["crow", "crane", "bakasana", "kakasana"],
    "Peacock_Pose_or_Mayurasana": ["peacock", "mayurasana"],
    "Scale_Pose_or_Tolasana": ["scale", "tolasana"],
    "Shoulder-Pressing_Pose_or_Bhujapidasana": ["shoulder pressing", "shoulder press", "bhujapidasana"],
    "Plank_Pose_or_Kumbhakasana": ["plank", "high plank", "kumbhakasana", "phalakasana"],
    "Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana": [
        "chaturanga", "chaturanga dandasana", "four limbed staff", "low plank"],
    # notes/Contact_config_def.MD: both Cobra takes are performed as up-dog, so that name counts too
    "Cobra_Pose_or_Bhujangasana": ["cobra", "bhujangasana", "upward facing dog", "urdhva mukha svanasana", "up dog"],
    "Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana": [
        "dolphin plank", "forearm plank", "makara adho mukha svanasana"],
    "Dolphin_Pose_or_Ardha_Pincha_Mayurasana": ["dolphin", "ardha pincha mayurasana"],
    "Standing_Forward_Bend_pose_or_Uttanasana": [
        "standing forward bend", "standing forward fold", "forward fold", "uttanasana"],
    "Garland_Pose_or_Malasana": ["garland", "malasana", "yogi squat"],
    "Low_Lunge_pose_or_Anjaneyasana": ["low lunge", "crescent lunge", "anjaneyasana"],
    "Warrior_II_Pose_or_Virabhadrasana_II": ["warrior ii", "virabhadrasana ii"],
    "Upward_Plank_Pose_or_Purvottanasana": ["upward plank", "reverse plank", "purvottanasana"],
    "Bridge_Pose_or_Setu_Bandha_Sarvangasana": ["bridge", "setu bandha", "setu bandha sarvangasana", "setu bandhasana"],
    "viparita_virabhadrasana_or_reverse_warrior_pose": ["reverse warrior", "viparita virabhadrasana", "peaceful warrior"],
    "Intense_Side_Stretch_Pose_or_Parsvottanasana": ["intense side stretch", "pyramid", "parsvottanasana"],
    "Feathered_Peacock_Pose_or_Pincha_Mayurasana": [
        "feathered peacock", "pincha mayurasana", "forearm stand", "forearm balance"],
}
_NUMERALS = {"1": "i", "2": "ii", "3": "iii"}


def name_tokens(text: str) -> tuple:
    """Lowercase ASCII words, digits 1-3 as numerals, Sanskrit ``h`` after a consonant and ``ri``
    folded (sirsasana = shirshasana, vrksasana = vrikshasana)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    out = []
    for tok in re.split(r"[^a-z0-9]+", text):
        if tok:
            tok = _NUMERALS.get(tok, tok)
            out.append(re.sub(r"([bcdgjkpst])h", r"\1", tok).replace("ri", "r"))
    return tuple(out)


@functools.lru_cache(maxsize=1)
def _alias_index() -> list[tuple[tuple, str]]:
    return [(name_tokens(a), pose) for pose, aliases in POSE_ALIASES.items() for a in aliases]


def resolve_pose(name: str) -> str | None:
    """The pose whose alias is the longest contiguous token match in ``name``; ``None`` for no
    match or a tie between two poses."""
    toks = name_tokens(name)
    best, poses = (0, 0), set()
    for alias, pose in _alias_index():
        k = len(alias)
        if any(toks[i:i + k] == alias for i in range(len(toks) - k + 1)):
            score = (k, len("".join(alias)))
            if score > best:
                best, poses = score, {pose}
            elif score == best:
                poses.add(pose)
    return next(iter(poses)) if len(poses) == 1 else None


def pose_truth(name: str) -> str | None:
    key = re.sub(r"_h\d+$", "", name)
    return key if key in POSE_ALIASES else None


# --------------------------------------------------------------------------- #
# Truth
# --------------------------------------------------------------------------- #
def _bound(g: dict) -> tuple[np.ndarray, float]:
    """Centre and radius of a sphere enclosing a world geom (one frame)."""
    if g["type"] == "sphere":
        return g["center"][0].numpy(), float(g["radius"])
    if g["type"] == "capsule":
        a, b = g["a"][0].numpy(), g["b"][0].numpy()
        return (a + b) / 2, float(np.linalg.norm(a - b)) / 2 + float(g["radius"])
    return g["center"][0].numpy(), float(np.linalg.norm(g["half"].numpy()))


def _gap_m(ga: dict, gb: dict) -> float:
    (ca, ra), (cb, rb) = _bound(ga), _bound(gb)
    low = float(np.linalg.norm(ca - cb)) - ra - rb
    return low if low >= PAIR_GATE_M else float(geom_pair_distance(ga, gb)[0][0])


@functools.lru_cache(maxsize=512)
def pair_gaps_cm(stem: str, frame: int) -> dict:
    """Surface gap (cm) of every non-adjacent zone pair of the shipped reference at ``frame``: the
    minimum over the zones' bodies and their collision geoms (negative = penetration). A pair whose
    bounding spheres are ``PAIR_GATE_M`` apart gets that lower bound instead of the exact gap."""
    clip, sk = render.load_clip(stem), capture.skeleton()
    pos = torch.as_tensor(clip.pos[frame:frame + 1])
    rot = torch.as_tensor(clip.rot[frame:frame + 1])
    world = {b: [geom_to_world(g, pos[:, i], rot[:, i]) for g in sk.geoms[b]] for i, b in enumerate(sk.names)}
    return {(za, zb): 100.0 * min(_gap_m(ga, gb) for a in ZONES[za] for b in ZONES[zb]
                                  for ga in world[a] for gb in world[b])
            for za, zb in BODY_PAIRS}


def _three_way(x: float, low: float, high: float) -> int | None:
    return 1 if x <= low else (0 if x >= high else None)


def packet_truth(record: dict, store: capture.Capture, calibration: dict, frame: int | None = None) -> dict:
    """What the evidence settles at the hold's exemplar (see the module docstring)."""
    fh = int(record["window"]["frame_hold"] if frame is None else frame)
    near = round(audit.STABLE_S * store.fps)
    human, marker_cm = {}, {}
    for z in record["capture"]["decided"]:
        zi = ZI[z]
        state, height = int(store["ground_state"][fh, zi]), float(store["marker_min_z"][fh, zi])
        band = calibration["zones"][z]
        decisive = state in (0, 1) and not (band["touch_m"] < height <= band["separation_m"])
        flips = audit.support_changes(store["ground_state"][:, [zi]])
        stable = not len(flips) or int(np.abs(flips - fh).min()) > near
        human[z] = state if decisive and stable else None
        marker_cm[z] = round(100.0 * height, 3)
    avatar_cm = {z: round(100.0 * float(store["avatar_min_z"][fh, ZI[z]]), 3) for z in ZONE_ORDER}
    gaps = {p: round(g, 3) for p, g in pair_gaps_cm(record["stem"], fh).items()}
    return {"hold_id": record["hold_id"], "frame": fh, "stratum": audit.stratum(record),
            "family_hold": record["family_hold"], "pose": pose_truth(record["name"]),
            "human": human, "marker_cm": marker_cm, "avatar_cm": avatar_cm,
            "avatar": {z: _three_way(a, AVATAR_TOUCH_CM, AVATAR_OFF_CM) for z, a in avatar_cm.items()},
            "pair_gap_cm": gaps, "pairs": {p: _three_way(g, PAIR_TOUCH_CM, PAIR_APART_CM) for p, g in gaps.items()}}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(answer: dict, truth: dict, meta: dict | None = None) -> list[dict]:
    """One row per scored item: ``class``, ``key``, ``truth``, ``claim`` (``None`` = abstained) and
    ``correct``, plus ``meta`` (packet id, hold, stratum) and the numbers behind the truth."""
    base = {"hold_id": truth["hold_id"], "stratum": truth["stratum"], "family_hold": truth["family_hold"],
            **(meta or {})}
    rows = []

    def add(cls, key, true, claim, correct=None, **detail):
        if correct is None and claim is not None:
            correct = claim == true
        rows.append({**base, "class": cls, "key": key, "truth": true, "claim": claim, "correct": correct, **detail})

    floor = {PARTS[p]: v for p, v in answer["floor"].items()}
    avatar = {z: CLAIM.get(v["avatar"]) for z, v in floor.items()}
    markers = {z: CLAIM.get(v["markers"]) for z, v in floor.items()}
    for z in ZONE_ORDER:
        t, cm = truth["avatar"][z], truth["avatar_cm"][z]
        if t is not None:
            add("floor_avatar" if cm <= NEAR_FLOOR_CM else "floor_avatar_far", z, t, avatar[z], avatar_cm=cm)
        h = truth["human"].get(z)
        if h is not None:
            add("floor_human", z, h, markers[z], marker_cm=truth["marker_cm"][z])
            if h == 1 and t is not None:  # the human touches: does the reviewer see the reference float?
                add("float", z, 1 - t, None if avatar[z] is None else 1 - avatar[z], avatar_cm=cm)
    for part in SIDED:
        left, right = f"L_{part}", f"R_{part}"
        for view, true, said in (("avatar", truth["avatar"], avatar), ("human", truth["human"], markers)):
            tl, tr = true.get(left), true.get(right)
            if tl is None or tr is None or tl == tr:
                continue
            cl, cr = said[left], said[right]
            answered = cl is not None and cr is not None and cl != cr
            add("left_right", f"{view}:{part}", [tl, tr], [cl, cr] if answered else None,
                correct=(cl == tl) if answered else None)

    bb, listed = answer["body_body"], {}
    for c in bb["contacts"]:
        a, b = sorted((PARTS[c["part_a"]], PARTS[c["part_b"]]), key=ZI.__getitem__)
        if (a, b) in truth["pairs"]:
            listed[(a, b)] = c["certainty"]
    keys = {p for p, t in truth["pairs"].items() if t == 1} | {p for p in listed if truth["pairs"][p] is not None}
    for p in sorted(keys, key=lambda p: (ZI[p[0]], ZI[p[1]])):
        if bb["answer"] == "cannot_tell":
            claim = None
        elif p in listed:
            claim = 1 if listed[p] == "clear" else None
        else:
            claim = 0
        add("body_body", "+".join(p), truth["pairs"][p], claim, gap_cm=truth["pair_gap_cm"][p])

    if truth["pose"] is not None:  # a name that resolves to no pose is an answer, and a wrong one
        names = [c["name"] for c in answer["pose"]["candidates"]]
        resolved = [resolve_pose(n) for n in names]
        add("pose_identity", "pose", truth["pose"], (resolved[0] or "unresolved") if names else None,
            correct=(resolved[0] == truth["pose"]) if names else None,
            names=names, resolved=resolved, any_of_three=truth["pose"] in resolved)
    return rows


def wilson_low(k: int, n: int, z: float = 1.96) -> float | None:
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre - spread) / (1 + z * z / n), 4)


def _ratio(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def class_metrics(rows: list[dict], kind: str) -> dict:
    """Precision, answered rate and evidence verdict of one class (see the module docstring)."""
    n = len(rows)
    answered = [r for r in rows if r["claim"] is not None]
    m = {"items": n, "answered": len(answered), "answered_rate": _ratio(len(answered), n),
         "correct": sum(bool(r["correct"]) for r in answered)}
    if kind == "binary":
        by = {}
        for value, word in ((1, "touching"), (0, "off_floor")):
            claims = [r for r in answered if r["claim"] == value]
            k, truths = sum(r["correct"] for r in claims), sum(r["truth"] == value for r in rows)
            by[word] = {"claims": len(claims), "correct": k, "precision": _ratio(k, len(claims)),
                        "precision_lo95": wilson_low(k, len(claims)), "truth_items": truths,
                        "recall": _ratio(sum(r["correct"] and r["claim"] == value for r in answered), truths)}
        used = [b for b in by.values() if b["claims"]]
        worst = min(used, key=lambda b: b["precision"]) if used else None
        m.update(accuracy=_ratio(m["correct"], len(answered)), by_value=by,
                 precision=worst["precision"] if worst else None,
                 precision_lo95=worst["precision_lo95"] if worst else None,
                 support=min((b["claims"] for b in used), default=0))
    elif kind == "detect":
        claims = [r for r in answered if r["claim"] == 1]
        negatives = [r for r in answered if r["claim"] == 0]
        tp, positives = sum(r["truth"] == 1 for r in claims), sum(r["truth"] == 1 for r in rows)
        m.update(claims=len(claims), precision=_ratio(tp, len(claims)), precision_lo95=wilson_low(tp, len(claims)),
                 positives=positives, recall=_ratio(tp, positives),
                 npv=_ratio(sum(r["truth"] == 0 for r in negatives), len(negatives)), support=len(claims))
    else:
        m.update(precision=_ratio(m["correct"], len(answered)), precision_lo95=wilson_low(m["correct"], len(answered)),
                 support=len(answered))
    m["evidence"] = bool(m["precision"] is not None and m["precision"] >= PRECISION_MIN
                         and m["answered_rate"] is not None and m["answered_rate"] >= ANSWERED_MIN
                         and n >= ITEMS_MIN and m["support"] >= CLAIMS_MIN)
    strata = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        s = strata[r["stratum"]]
        s[0] += 1
        s[1] += r["claim"] is not None
        s[2] += bool(r["correct"])
    m["by_stratum"] = {k: {"items": v[0], "answered": v[1], "correct": v[2]} for k, v in sorted(strata.items())}
    return m


def _float_recall_by_size(rows: list[dict]) -> dict:
    out = {}
    for lo, hi in ((2, 5), (5, 15), (15, 60)):
        pos = [r for r in rows if r["truth"] == 1 and lo <= r["avatar_cm"] < hi]
        out[f"{lo}-{hi} cm"] = {"floats": len(pos), "caught": sum(r["claim"] == 1 for r in pos),
                                "abstained": sum(r["claim"] is None for r in pos)}
    return out


def _stats(values: list[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "total": round(float(v.sum()), 4), "median": round(float(np.median(v)), 4),
            "max": round(float(v.max()), 4)}


def advisory_counts(verdicts: list[dict], truths: dict) -> dict:
    """What has no truth yet: marker claims on zones the capture cannot decide, discrepancy and
    implausibility kinds."""
    kinds = collections.Counter(x["kind"] for v in verdicts for x in v["answer"]["discrepancies"])
    implausible = collections.Counter(x["kind"] for v in verdicts for x in v["answer"]["implausible"])
    undecided = collections.Counter()
    for v in verdicts:
        decided = truths[v["hold_id"]]["human"]
        for part, a in v["answer"]["floor"].items():
            if PARTS[part] not in decided:
                undecided[f"{PARTS[part]}:{a['markers']}"] += 1
    return {"discrepancy_kinds": dict(sorted(kinds.items())), "implausible_kinds": dict(sorted(implausible.items())),
            "human_marker_claims_without_truth": dict(sorted(undecided.items()))}


def reviewer_table(verdicts: list[dict], calls: list[dict], truths: dict) -> tuple[dict, list[dict]]:
    """The calibration of one reviewer key: per-class metrics over ``verdicts`` (one per packet),
    and the rows behind them."""
    rows = [row for v in verdicts for row in score(v["answer"], truths[v["hold_id"]],
                                                   {"packet_id": v["packet_id"], "n": v["n"]})]
    classes = {cls: class_metrics([r for r in rows if r["class"] == cls], kind) for cls, kind in CLASSES.items()}
    classes["float"]["recall_by_size"] = _float_recall_by_size([r for r in rows if r["class"] == "float"])
    pose = [r for r in rows if r["class"] == "pose_identity"]
    classes["pose_identity"]["any_of_three"] = sum(r["any_of_three"] for r in pose)
    classes["pose_identity"]["unresolved_names"] = sum(x is None for r in pose for x in r["resolved"])
    # a secondary (_h<k>) hold's label is its clip's family, which it need not show
    family = [r for r in pose if r["family_hold"] and r["claim"] is not None]
    classes["pose_identity"]["family_holds"] = {"answered": len(family), "correct": sum(r["correct"] for r in family)}
    head = verdicts[0]["reviewer"]
    usage = [c.get("usage") or {} for c in calls]
    table = {
        "key": head["key"], "model": head["model"], "effort": head["effort"],
        "prompt_sha256": verdicts[0]["prompt"]["sha256"], "schema_sha256": verdicts[0]["schema"]["sha256"],
        "cli_versions": sorted({c["reviewer"].get("cli_version") or "" for c in calls}),
        "packets": len(verdicts),
        "calls": dict(collections.Counter(c["status"] for c in calls)),
        "cost_usd": _stats([c.get("cost_usd") for c in calls]),
        "duration_s": _stats([c.get("duration_s") for c in calls]),
        "output_tokens": _stats([u.get("output_tokens") for u in usage]),
        "thinking_tokens": _stats([u.get("thinking_tokens") for u in usage]),
        "classes": classes,
        "evidence": [c for c, m in classes.items() if m["evidence"]],
        "advisory": [c for c, m in classes.items() if not m["evidence"]],
        "advisory_counts": advisory_counts(verdicts, truths),
    }
    return table, rows


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(ledger: list[dict], records: dict, *, render_v: str = render.RENDER_V, pass_: str = "A",
              store_dir: Path = capture.STORE_DIR, calibration: dict | None = None) -> tuple[dict, dict, list[str]]:
    """``(table, rows by reviewer key, failures)`` from the ledger's ``pass_`` verdicts on
    ``render_v`` packets. ``records`` are the audit records by hold id. Never queries anything."""
    cal = capture.load_calibration() if calibration is None else calibration
    mine = [r for r in ledger if r["pass"] == pass_ and r["render_v"] == render_v]
    failures, truths = [], {}
    for hold_id in sorted({r["hold_id"] for r in mine if r["status"] == "valid"}):
        try:
            record = records[hold_id]
            truths[hold_id] = packet_truth(record, capture.load(record["stem"], store_dir, cal), cal)
        except Exception as exc:  # noqa: BLE001 -- report every hold, then fail
            failures.append(f"{hold_id}: {type(exc).__name__}: {exc}")
    by_key = collections.defaultdict(list)
    for r in mine:
        by_key[r["reviewer"]["key"]].append(r)
    reviewers, all_rows = [], {}
    for key, calls in sorted(by_key.items()):
        first = {}
        for r in calls:  # the ledger is ordered by (packet, pass, n)
            if r["status"] == "valid" and r["hold_id"] in truths:
                first.setdefault(r["packet_id"], r)
        if not first:
            continue
        table, rows = reviewer_table(list(first.values()), calls, truths)
        reviewers.append(table)
        all_rows[key] = rows
    table = {"render_v": render_v, "pass": pass_, "rule": RULE, "truth": TRUTH,
             "audit_ids": sorted({records[h]["audit_id"] for h in truths}),
             "holds": len(truths), "reviewers": reviewers}
    return table, all_rows, failures


def write_calibration(table: dict, rows: dict, ledger_dir: Path, out: Path, items_root: Path = ITEMS_ROOT,
                      calibration_path: Path = capture.CALIBRATION_PATH) -> Path:
    inputs = [Path(__file__), prompt_path(table["pass"]), schema_path(table["pass"]), calibration_path]
    inputs += sorted(Path(ledger_dir).glob(f"*.{table['pass']}.*.json"))
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), **table}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=1) + "\n")
    for key, rs in rows.items():
        path = Path(items_root) / table["render_v"] / f"pass_{table['pass'].lower()}.{key}.items.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rs))
    return Path(out)


def format_table(table: dict) -> str:
    """The admission table, one line per class and reviewer."""
    lines = []
    for rv in table["reviewers"]:
        c = rv["cost_usd"]
        lines.append(f"{rv['model']} effort {rv['effort']} (key {rv['key']}): {rv['packets']} packets, "
                     f"${c.get('total', 0):.2f} (median ${c.get('median', 0):.3f}/packet)")
        for cls, m in rv["classes"].items():
            prec = "-" if m["precision"] is None else f"{m['precision']:.3f}"
            rate = "-" if m["answered_rate"] is None else f"{m['answered_rate']:.2f}"
            extra = f", recall {m['recall']:.3f}" if m.get("recall") is not None else ""
            lines.append(f"  {cls:17s} {'EVIDENCE' if m['evidence'] else 'advisory':8s} precision {prec:>5s} on "
                         f"{m['support']:3d} claims, answered {rate:>4s} of {m['items']:4d}{extra}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pass", dest="pass_", choices=PASSES, default="A")
    ap.add_argument("--render-v", default=render.RENDER_V)
    ap.add_argument("--ledger-dir", type=Path, default=LEDGER_DIR)
    ap.add_argument("--audit", type=Path, help="an audit directory (default: the newest calibrated audit)")
    ap.add_argument("--store-dir", type=Path, default=capture.STORE_DIR)
    ap.add_argument("--calibration", type=Path, default=capture.CALIBRATION_PATH, help="the capture calibration")
    ap.add_argument("--out", type=Path, help="default: data/reference_curation/calibration/<render_v>.json")
    ap.add_argument("--items-root", type=Path, default=ITEMS_ROOT)
    args = ap.parse_args(argv)

    try:
        aud = packets.load_audit(args.audit or packets.default_audit_dir())
        ledger = read_ledger(args.ledger_dir)
        table, rows, failures = calibrate(ledger, aud.records, render_v=args.render_v, pass_=args.pass_,
                                          store_dir=args.store_dir,
                                          calibration=capture.load_calibration(args.calibration))
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures or not table["reviewers"]:
        print(f"calibration {args.render_v}: {len(failures)} failures, {len(table['reviewers'])} reviewers "
              f"with valid verdicts in {args.ledger_dir}; nothing written", file=sys.stderr)
        return 1
    out = write_calibration(table, rows, args.ledger_dir, args.out or CALIBRATION_DIR / f"{args.render_v}.json",
                            args.items_root, args.calibration)
    print(format_table(table))
    evidence = {rv["effort"]: rv["evidence"] for rv in table["reviewers"]}
    print(f"calibration {args.render_v} pass {args.pass_}: {table['holds']} holds, {len(table['reviewers'])} reviewer "
          f"configs; evidence {evidence} -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
