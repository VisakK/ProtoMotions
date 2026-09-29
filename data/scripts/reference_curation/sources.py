# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source states (BUILD_PLAN Step 6): what the human did with every zone and every pair on every
frame, taken from the best evidence that decides it.

``load(stem)`` returns capture store v3: the v2 record (``human_mesh.load``: v1 plus the mesh) plus
the two arrays of ``FLOOR_ARRAYS``. ``output/reference_curation/capture/v3/<stem>.{npz,json}`` holds
only those two and the identity of the v2 record they extend; v1 and v2 are never rewritten.

Seam arbitration (why v3 exists)
--------------------------------
v2's ``human_ground_state`` gives a zone the floor contact of its lowest vertex, and a vertex belongs
to the zone of its dominant skinning joint. Where two zones meet at a joint, the seam vertices of one
zone sit next to the other zone's surface. So when one side is on the floor, the other side's seam
reads as touching too:

* Standing Split -a at 9.0 s: the palm is pressed 1 cm into the floor, and the "forearm" touches
  through three wrist-crease vertices (weights 0.53 elbow / 0.47 wrist, 1.9 cm up), while the wrist
  joint is 3.5 cm up and the wrist markers 6 cm. v2 has the left forearm on the floor for 5.6 s.
* Plow, Shoulderstand and Bridge: the upper arms lie on the floor and the forearms rise from the
  elbow, which v2 reads as forearm contacts (about 800 frames per arm in Plow -b).
* Scale -a: the gluteal fold makes the thighs touch next to a seated pelvis.

v3 re-reads the eight limb zones (shanks, thighs, upper arms, forearms). A limb zone's vertices within
``SEAM_M`` (5 cm, along the template surface) of an adjacent zone are its seam with that zone. The
limb zone touches if its core (every vertex that is in no seam) touches, or through a seam, but only
while the neighbour's own core is not on the floor and the limb's side of the seam is the deeper one.
So a kneecap is still a knee contact, whichever zone it falls in, and a palm takes its wrist crease.

The trunk, the pelvis and the extremities (feet, hands, head) keep v2's state. The extremities agree
with the markers on 99 % of decided frames (Step 5), and the torso's seams are real contact areas:
Bridge's upper back touches next to the upper arms, and in the avatar that is the thorax (TRUNK).

Evidence hierarchy
------------------
``ground_source`` combines the channels per zone and frame, first decided wins: the mesh (v3), then
the markers (v1, feet, hands and head), then the mat. The mat can only say *contact*: a zone the
attribution can see (``attr_visible``, coverage column 0 >= 0.9) carrying more than 50 N. A zone at
0 N may be resting or invisible, so the mat never says *separated*. With a readable fit the markers
add nothing (the mesh is unknown exactly where the marker hygiene fails); they and the mat decide
the one clip with no readable fit. ``pair_source`` is the mesh's ``human_pair_state``: pairs have no
other channel.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.sources --all
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

from extract_contact_configs import ZONE_ORDER
from reference_curation import audit, capture, human_mesh as hm, ids

MODULE = "reference_curation.sources"
SCHEMA_VERSION = 1
STORE_VERSION = "v3"
STORE_DIR = ids.OUTPUT_ROOT / "capture" / STORE_VERSION

SEAM_M = 0.05              # seam-only contacts lie within 0.9-4.0 cm of the neighbour (all 303 exemplars)
MAT_CONTACT_N = capture.CALIB_LOAD_N
LIMB_ZONES = ("L_SHANK", "R_SHANK", "L_THIGH", "R_THIGH", "L_UPPER_ARM", "R_UPPER_ARM", "L_FOREARM", "R_FOREARM")
ZI = {z: i for i, z in enumerate(ZONE_ORDER)}
NEIGHBOURS = {z: tuple(n for n in ZONE_ORDER if frozenset((z, n)) in hm.KINEMATIC_ADJACENT) for z in ZONE_ORDER}
CHANNELS = ("none", "mesh", "markers", "mat")   # source channel codes 0..3
CHANNEL_NAMES = {"mesh": "capture_fit_mesh", "markers": "capture_markers", "mat": "mat_attributed", "none": "none"}
CONFIG = {"seam_m": SEAM_M, "limb_zones": list(LIMB_ZONES), "mat_contact_n": MAT_CONTACT_N,
          "coverage_gate_col0": audit.COVERAGE_GATE, "hierarchy": ["mesh", "markers", "mat"]}

FLOOR_ARRAYS = {
    "human_floor_z": ("m", "[T,Z] lowest skin point that is the zone's own after seam arbitration (limb zones); "
                           "human_min_z for the trunk, pelvis and extremities"),
    "human_floor_state": ("int8", "[T,Z] human floor touch from human_floor_z: 1 contact, 0 separated, -1 unknown"),
}


# --------------------------------------------------------------------------- #
# Seam arbitration (pure)
# --------------------------------------------------------------------------- #
_SUBSETS: dict = {}


def seam_subsets(mdl: hm.SMPLX, v_template: np.ndarray) -> tuple[dict, dict]:
    """``(core, seam)``: per limb zone its core vertices, and per (limb zone, neighbour) the zone's
    vertices within ``SEAM_M`` of the neighbour along the template surface; also the neighbours'
    cores and their seams toward the limb, which the arbitration compares against."""
    key = (id(mdl), hashlib.sha1(np.ascontiguousarray(v_template).tobytes()).hexdigest())
    if key not in _SUBSETS:
        geo = hm.zone_geodesics(mdl, v_template)
        core, seam = {}, {}
        for z in ZONE_ORDER:
            idx = mdl.zone_vertices[ZI[z]]
            near = np.zeros(len(idx), dtype=bool)
            for n in NEIGHBOURS[z]:
                s = geo[ZI[n]][idx] <= SEAM_M
                seam[z, n] = idx[s]
                near |= s
            core[z] = idx[~near]
        _SUBSETS[key] = (core, seam)
    return _SUBSETS[key]


def floor_heights(human: hm.Human) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(skin [T,Z], core [T,Z], seams {(zone, neighbour): [T]})``: per frame the lowest vertex of each
    zone, of its core, and of each of its seams, in metres (``inf`` for an empty set)."""
    mdl = human.model
    core, seam = seam_subsets(mdl, human.fit["v_template"])
    T = human.num_frames
    skin = np.empty((T, len(ZONE_ORDER)))
    lo_core = np.full((T, len(ZONE_ORDER)), np.inf)
    lo_seam = {k: np.full(T, np.inf) for k in seam}
    for s, verts in human.chunks():
        z = verts[..., 2]
        k = slice(s, s + len(verts))
        skin[k] = hm.zone_min_z(verts, mdl)
        for zone in ZONE_ORDER:
            if len(core[zone]):
                lo_core[k, ZI[zone]] = z[:, core[zone]].min(1)
        for key, idx in seam.items():
            if len(idx):
                lo_seam[key][k] = z[:, idx].min(1)
    return skin, lo_core, lo_seam


def arbitrate(skin: np.ndarray, lo_core: np.ndarray, lo_seam: dict, known: np.ndarray, ground: dict) -> np.ndarray:
    """``[T,Z]`` floor height after seam arbitration. A limb zone keeps a seam contact only while the
    neighbour's core is off the floor (its Schmitt state, so a core hovering at the threshold cannot
    make the limb flicker) and the limb's side of the seam is the deeper one. Other zones: ``skin``."""
    out = skin.copy()
    core_state = np.stack([capture.contact_state(lo_core[:, zi], known[:, zi], ground["touch_m"],
                                                 ground["separation_m"]) for zi in range(len(ZONE_ORDER))], -1)
    for zone in LIMB_ZONES:
        h = lo_core[:, ZI[zone]].copy()
        for n in NEIGHBOURS[zone]:
            mine = lo_seam[zone, n]
            h = np.minimum(h, np.where((core_state[:, ZI[n]] != 1) & (mine <= lo_seam[n, zone]), mine, np.inf))
        out[:, ZI[zone]] = h
    return out


# --------------------------------------------------------------------------- #
# Store v3
# --------------------------------------------------------------------------- #
def measure(stem: str) -> dict:
    """``{"arrays": {"skin", "core", "seams"} | None, "meta"}`` for one x0 clip, before thresholds."""
    human, status, error = hm.load_human(stem)
    meta = {"stem": stem, "human_available": human is not None, "human_status": status, "human_error": error}
    if human is None:
        return {"arrays": None, "meta": meta}
    skin, lo_core, lo_seam = floor_heights(human)
    return {"arrays": {"skin": skin, "core": lo_core, "seams": lo_seam}, "meta": meta}


def _base_identity(base: capture.Capture) -> dict:
    return {"store": base.meta["store"], "generator": base.meta["generator"]["sha256"],
            "calibration": base.meta["calibration"]["id"], "base": base.meta["base"]}


def _paths(stem: str, store_dir: Path) -> tuple[Path, Path]:
    return Path(store_dir) / f"{stem}.npz", Path(store_dir) / f"{stem}.json"


def build(stem: str, store_dir: Path = STORE_DIR, base: capture.Capture | None = None,
          measured: dict | None = None) -> capture.Capture:
    """Arbitrate ``stem``'s floor contacts, write ``<store_dir>/<stem>.{npz,json}`` and return the v2
    record extended with ``FLOOR_ARRAYS``."""
    base = hm.load(stem) if base is None else base
    cal = hm.load_calibration()
    if base.meta["calibration"]["id"] != cal["id"]:
        raise ValueError(f"{stem}: the v2 record was built with calibration {base.meta['calibration']['id']}, "
                         f"the committed one is {cal['id']}")
    m = measure(stem) if measured is None else measured
    T = base.meta["num_frames"]
    if m["arrays"] is None:
        floor_z = np.full((T, len(ZONE_ORDER)), np.nan)
        state = np.full((T, len(ZONE_ORDER)), -1, dtype=np.int8)
    else:
        a = m["arrays"]
        if a["skin"].shape[0] != T or not np.allclose(a["skin"], base["human_min_z"], atol=1e-5, equal_nan=True):
            raise ValueError(f"{stem}: the mesh posed here differs from the v2 record's human_min_z")
        known = capture.zone_known(base["marker_min_z"], base["marker_resid"])
        floor_z = arbitrate(a["skin"], a["core"], a["seams"], known, cal["ground"])
        state = np.stack([capture.contact_state(floor_z[:, zi], known[:, zi], cal["ground"]["touch_m"],
                                                cal["ground"]["separation_m"]) for zi in range(len(ZONE_ORDER))], -1)
        state = state.astype(np.int8)
        state[:, [ZI[z] for z in ZONE_ORDER if z not in LIMB_ZONES]] = \
            base["human_ground_state"][:, [ZI[z] for z in ZONE_ORDER if z not in LIMB_ZONES]]
    v2 = base["human_ground_state"]
    counts = {z: {"v2_contact": int((v2[:, ZI[z]] == 1).sum()), "v3_contact": int((state[:, ZI[z]] == 1).sum())}
              for z in LIMB_ZONES}
    arrays = {"human_floor_z": floor_z.astype(np.float32), "human_floor_state": state}
    meta = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [hm.MODEL_PATH]), "store": f"capture/{STORE_VERSION}",
            **m["meta"], "fps": base.meta["fps"], "num_frames": T, "zone_order": list(ZONE_ORDER),
            "base": _base_identity(base), "calibration": base.meta["calibration"], "config": CONFIG,
            "counts": counts, "arrays": {k: {"unit": u, "meaning": d} for k, (u, d) in FLOOR_ARRAYS.items()}}
    npz_path, json_path = _paths(stem, store_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, indent=1) + "\n")
    return capture.Capture(meta, {**base.arrays, **arrays})


def load(stem: str, store_dir: Path = STORE_DIR, base: capture.Capture | None = None) -> capture.Capture:
    """Capture store v3 for ``stem``: v2 plus ``FLOOR_ARRAYS``, rebuilt if missing, made by other code
    or on another v2 record."""
    base = hm.load(stem) if base is None else base
    npz_path, json_path = _paths(stem, store_dir)
    if npz_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        if (meta.get("schema_version") == SCHEMA_VERSION and meta["generator"]["sha256"] == ids.sha256_file(__file__)
                and meta["base"] == _base_identity(base)):
            with np.load(npz_path) as npz:
                return capture.Capture(meta, {**base.arrays, **{k: npz[k] for k in npz.files}})
    return build(stem, store_dir, base)


def identity(rec: capture.Capture) -> dict:
    """What a v3 record is determined by: this module, its v2 record and that record's own bases."""
    return {"generator": rec.meta["generator"]["sha256"], "base": rec.meta["base"]}


def build_all(stems: list[str], store_dir: Path = STORE_DIR, workers: int = 1,
              bases: dict | None = None) -> tuple[list[capture.Capture], list[str]]:
    """``(records, failures)``: every clip measured in ``workers`` spawned processes, then written on
    its v2 record (``bases[stem]``, default ``human_mesh.load``)."""
    records, failures = [], []
    if min(workers, len(stems)) <= 1:
        results = []
        for stem in stems:
            try:
                results.append((stem, measure(stem), None))
            except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
                results.append((stem, None, exc))
    else:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {stem: pool.submit(measure, stem) for stem in stems}
            results = [(stem, f.result() if f.exception() is None else None, f.exception())
                       for stem, f in futures.items()]
    for stem, m, exc in results:
        try:
            if exc is not None:
                raise exc
            records.append(build(stem, store_dir, base=(bases or {}).get(stem), measured=m))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{stem}: {type(exc).__name__}: {exc}")
    return records, failures


# --------------------------------------------------------------------------- #
# The evidence hierarchy (pure)
# --------------------------------------------------------------------------- #
def mat_contact(rec: capture.Capture) -> np.ndarray:
    """``[T,Z]`` the mat's only positive statement: a visible zone carries more than ``MAT_CONTACT_N``
    on a coverage-valid frame."""
    cov = np.nan_to_num(rec["mat_valid_cov"], nan=0.0) >= audit.COVERAGE_GATE
    with np.errstate(invalid="ignore"):
        return cov[:, None] & rec["attr_visible"] & (np.nan_to_num(rec["mat_zone_load"], nan=0.0) > MAT_CONTACT_N)


def ground_source(rec: capture.Capture) -> tuple[np.ndarray, np.ndarray]:
    """``(state [T,Z] int8, channel [T,Z] int8)``: 1 contact, 0 separated, -1 unknown, from the first
    channel that decides it (``CHANNELS`` codes: mesh, then markers, then the mat's contact)."""
    mesh, markers, mat = rec["human_floor_state"], rec["ground_state"], mat_contact(rec)
    state = np.where(mesh >= 0, mesh, np.where(markers >= 0, markers, np.where(mat, 1, -1))).astype(np.int8)
    channel = np.where(mesh >= 0, 1, np.where(markers >= 0, 2, np.where(mat, 3, 0))).astype(np.int8)
    return state, channel


def pair_source(rec: capture.Capture) -> np.ndarray:
    """``[T,P]`` int8 in ``human_mesh.PAIR_NAMES`` order: the mesh's skin contact."""
    return rec["human_pair_state"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="every clip of the manifest")
    what.add_argument("--stem", nargs="+")
    ap.add_argument("--manifest", type=Path, default=ids.DEFAULT_MANIFEST)
    ap.add_argument("--store-dir", type=Path, default=STORE_DIR)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    start = time.time()
    stems = ids.manifest_stems(args.manifest) if args.all else args.stem
    records, failures = build_all(stems, args.store_dir, args.workers)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    moved = {z: sum(r.meta["counts"][z]["v2_contact"] - r.meta["counts"][z]["v3_contact"] for r in records)
             for z in LIMB_ZONES}
    total = sum(r.meta["counts"][z]["v2_contact"] for r in records for z in LIMB_ZONES)
    print(f"capture {STORE_VERSION} (seam arbitration): {len(records)} clips in {time.time() - start:.1f} s; "
          f"limb contact frames {total} -> {total - sum(moved.values())} "
          f"({', '.join(f'{z} -{n}' for z, n in moved.items() if n)}); {len(failures)} failures "
          f"-> {ids.display_path(args.store_dir)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
