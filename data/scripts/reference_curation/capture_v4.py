# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture store v4 (BodyFix Step 4, item 1): the avatar side of the capture store, measured on plant v2 and the
Step 3 references; the human side reused unchanged.

Capture stores v1-v3 answer three questions per clip, frame and zone: what the human did (markers, mesh), what the
reference does (``avatar_min_z``) and what the mat says (``mat_*``). The human side does not depend on the plant.
The reference side and the mat's *attribution* do: they were measured on the shipped plant (v1) and its references.
Store v4 re-measures them on plant v2 (her skeleton) and the ``retarget_v2`` motions, and nothing else:

=================  ======  =========================================================================================
array              unit    meaning (all measured on plant v2 and the Step 3 reference)
=================  ======  =========================================================================================
avatar_min_z       m       [T,Z] the zone's lowest collision surface (``mosh_replay.zone_lowest``, float64)
avatar_pair_gap    m       [T,P] the surface gap of every ``human_mesh.PAIRS`` zone pair, min over the zones' body
                           pairs (``retarget`` kernels: the separating-axis depth for box-box), capped at 10 cm
mat_zone_load      N       [T,Z] the load the re-run attribution (``pressure_v2``) put on the zone's bodies
mat_unexplained    N       [T] mat total minus all attributed load
mat_valid_body     -       [T] validity column 1, coverage x explained
mat_valid_share    -       [T] validity column 2, on-mat x explained (now on every clip)
attr_visible       bool    [T,Z] the zone within 6 cm of the floor in the pose the attribution ran on: the reference
=================  ======  =========================================================================================

``load(stem)`` returns the full record: capture store v3 (``sources.load``: the v1 markers and mat totals, the v2
mesh, the v3 seam arbitration) with these arrays **in place of** v1's avatar-side ones, so every function written
against a ``capture.Capture`` (``audit.audit_hold``, ``sources.ground_source``, ``verdicts.packet_truth``) reads the
plant-v2 evidence. The plant-free mat fields (``mat_total``, ``mat_cop``, ``mat_valid_cov``) stay v1's; ``build``
checks they equal the re-run's to float32 (the same archive, the same registration).

Plant identity
--------------
The human-side stores are read **outside** plant v2 (``retarget_v2.human_evidence``'s rule): ``sources.load``
rebuilds a stale record, and a rebuild under ``REFERENCE_PLANT=v2`` would write plant-v2 avatar fields into the v1
records. Every v4 record carries plant v2's identity (``plant``, and the MJCF among its ``inputs``) and ``load``
refuses it on any other plant through ``ids.require_plant``. The geometry is computed with plant v2's skeleton
passed explicitly (``fit_writer.skeleton``), so nothing here depends on the process's selected plant.

What the human-side thresholds rest on
--------------------------------------
Capture v1's marker thresholds and v2's mesh touch threshold were calibrated on *mat-confirmed* frames, which the
attribution decides (zone load > 50 N, ``attr_visible``, column 1 >= 0.9). They are kept (the human side does not
change), and ``crosscheck`` re-derives both on the plant-v2 attribution's mat-confirmed frames and reports the
difference, as evidence that they still hold.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.capture_v4 --all [--workers 8]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from extract_contact_configs import ZONE_ORDER, ZONES
from reference_curation import capture, fit_writer as fw, human_mesh as hm, ids, pressure_v2
from reference_curation import mosh_replay as mr
from reference_curation import retarget as rt

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.capture_v4"
SCHEMA_VERSION = 1
STORE_VERSION = "v4"
STORE_DIR = ids.OUTPUT_ROOT / "capture" / STORE_VERSION
PLANT = "v2"
RETARGET_ID = pressure_v2.RETARGET_ID
CROSSCHECK_PATH = ids.DATA_ROOT / "calibration" / "capture_v4_crosscheck.json"
PAIR_CAP_M = hm.PAIR_CAP_M          # 0.10, as the human's pair gap
ATTR_BAND_M = capture.ATTR_BAND_M   # 0.06

AVATAR_ARRAYS = {
    "avatar_min_z": ("m", "[T,Z] lowest collision surface of the zone: plant v2, the Step 3 reference"),
    "avatar_pair_gap": ("m", "[T,P] surface gap of the human_mesh.PAIRS zone pairs on plant v2 (min over body pairs, "
                             "negative = overlap), capped at 0.1 m"),
    "mat_zone_load": ("N", "[T,Z] load the plant-v2 attribution put on the zone's bodies"),
    "mat_unexplained": ("N", "[T] mat_total minus all attributed load"),
    "mat_valid_body": ("-", "[T] validity column 1, coverage x explained (plant-v2 attribution)"),
    "mat_valid_share": ("-", "[T] validity column 2, on-mat x explained (plant v2)"),
    "attr_visible": ("bool", "[T,Z] zone within 6 cm of the floor in the pose the attribution ran on (the reference)"),
}
PLANT_FREE = ("mat_total", "mat_cop", "mat_valid_cov")


def plant_mjcf() -> Path:
    return fw.plant_paths(PLANT)[0]


# --------------------------------------------------------------------------- #
# The human side (plant-free), read outside plant v2
# --------------------------------------------------------------------------- #
def require_v1_process() -> None:
    """The human-side stores may be rebuilt on load, so read them only where ``ids.MJCF`` is plant v1."""
    if plant_identity.name_of(plant_identity.sha256(ids.MJCF)) != "v1":
        raise RuntimeError("read the human-side capture stores outside on_plant() and without REFERENCE_PLANT=v2: "
                           "a stale v1-v3 record rebuilt now would carry plant-v2 avatar fields")


def human_base(stem: str) -> capture.Capture:
    """Capture store v3 of ``stem`` (``sources.load``), refused under plant v2."""
    from reference_curation import sources

    require_v1_process()
    rec = sources.load(stem)
    if not rec.meta.get("human_available"):
        raise ValueError(f"{stem}: the capture store has no human mesh")
    return rec


# --------------------------------------------------------------------------- #
# The avatar side on plant v2 (pure given the motion)
# --------------------------------------------------------------------------- #
def motion_path(stem: str, rid: str = RETARGET_ID) -> Path:
    return ids.motion_path(stem, pressure_v2.RETARGET_ROOT / rid)


def zone_pair_gaps(sk: rt.Skeleton, pos: torch.Tensor, rot: torch.Tensor, cap: float = PAIR_CAP_M) -> np.ndarray:
    """``[T, P]`` the surface gap of every ``human_mesh.PAIRS`` zone pair on skeleton ``sk``: the smallest over the
    zones' body pairs, exact (``retarget.body_gaps``) where the bodies' bounding spheres come within ``cap``, else
    ``cap``. ``pos [T,B,3]``, ``rot [T,B,3,3]`` float64."""
    zb = {z: [sk.names.index(b) for b in ZONES[z]] for z in ZONE_ORDER}
    members = [[(min(x, y), max(x, y)) for x in zb[za] for y in zb[zb_]] for za, zb_ in hm.PAIRS]
    body_pairs = sorted({p for m in members for p in m})
    col = {p: k for k, p in enumerate(body_pairs)}
    if any(sk.parents[b] == a or sk.parents[a] == b for a, b in body_pairs):
        raise ValueError("a zone pair includes a parent-child body pair, which the plant does not collide")
    f, a, b, gap = rt.near_body_pairs(sk, pos, rot, cap, np.array(body_pairs))
    G = np.full((pos.shape[0], len(body_pairs)), cap)
    G[f, [col[(int(x), int(y))] for x, y in zip(a, b)]] = np.minimum(gap, cap)
    return np.stack([G[:, [col[p] for p in m]].min(1) for m in members], 1)


def avatar_side(stem: str, rid: str = RETARGET_ID) -> dict:
    """``{"arrays", "meta", "inputs"}``: the avatar-side arrays of one clip on plant v2."""
    from reference_curation.retarget_v2 import motion_state

    sk = fw.skeleton(PLANT)
    mpath, port_path = motion_path(stem, rid), pressure_v2.gated_port(stem, rid)
    bodies_path = pressure_v2.paths(rid)["bodies"] / f"{stem}.npz"
    mot = torch.load(mpath, map_location="cpu", weights_only=False)
    plant_identity.require(mot.get(plant_identity.KEY), plant_mjcf(), str(mpath))
    port = torch.load(port_path, map_location="cpu", weights_only=False)
    plant_identity.require(port.get(plant_identity.KEY), plant_mjcf(), str(port_path))
    if not torch.equal(port["rigid_body_pos"], mot["rigid_body_pos"]):
        raise ValueError(f"{stem}: the gated port's kinematics are not the reference's")
    pos, rot = motion_state(mot)
    zmin = mr.zone_lowest(sk, pos, rot)
    with torch.no_grad():
        pgap = zone_pair_gaps(sk, torch.as_tensor(pos), torch.as_tensor(rot))
    gr = port["ground_reaction"].double().numpy()
    force = port["rigid_body_ground_forces"][..., 2].double().numpy()
    valid = port["ground_reaction_valid"].double().numpy()
    zone_bodies = [[sk.names.index(b) for b in ZONES[z]] for z in ZONE_ORDER]
    arrays = {"avatar_min_z": zmin, "avatar_pair_gap": pgap,
              "mat_zone_load": np.stack([force[:, idx].sum(-1) for idx in zone_bodies], -1),
              "mat_unexplained": gr[:, 0] - force.sum(-1), "mat_valid_body": valid[:, 1], "mat_valid_share": valid[:, 2],
              "attr_visible": zmin <= ATTR_BAND_M,
              # plant-free, compared with the v1 record by ``build``, not stored
              "_mat_total": gr[:, 0], "_mat_cop": gr[:, 1:3], "_mat_valid_cov": valid[:, 0]}
    meta = {"stem": stem, "retarget_id": rid, "num_frames": int(pos.shape[0]), "fps": int(mot["fps"]),
            "motion": ids.display_path(mpath), "motion_sha256": ids.sha256_file(mpath),
            "port": ids.display_path(port_path), "port_sha256": ids.sha256_file(port_path)}
    return {"arrays": arrays, "meta": meta, "inputs": [plant_mjcf(), fw.plant_paths(PLANT)[1], mpath, port_path,
                                                       bodies_path]}


# --------------------------------------------------------------------------- #
# Store v4
# --------------------------------------------------------------------------- #
def _base_identity(base: capture.Capture) -> dict:
    from reference_curation import sources

    return sources.identity(base)


def _paths(stem: str, store_dir: Path) -> tuple[Path, Path]:
    return Path(store_dir) / f"{stem}.npz", Path(store_dir) / f"{stem}.json"


def _plant_free_matches(base: capture.Capture, a: dict) -> dict:
    """The re-run's plant-free mat fields against the v1 record's (stored float32): max abs differences."""
    out = {}
    for k in PLANT_FREE:
        mine, theirs = np.asarray(a["_" + k], np.float64), np.asarray(base[k], np.float64)
        both = np.isfinite(mine) & np.isfinite(theirs)
        out[k] = float(np.abs(mine[both] - theirs[both]).max()) if both.any() else 0.0
    return out


def build(stem: str, store_dir: Path = STORE_DIR, base: capture.Capture | None = None, measured: dict | None = None,
          rid: str = RETARGET_ID) -> capture.Capture:
    """Measure ``stem`` on plant v2, write ``<store_dir>/<stem>.{npz,json}`` and return the full record."""
    base = human_base(stem) if base is None else base
    m = avatar_side(stem, rid) if measured is None else measured
    T = base.meta["num_frames"]
    if m["meta"]["num_frames"] != T or m["meta"]["fps"] != base.fps:
        raise ValueError(f"{stem}: the reference has {m['meta']['num_frames']} frames at {m['meta']['fps']} fps, the "
                         f"capture store {T} at {base.fps}")
    match = _plant_free_matches(base, m["arrays"])
    if max(match.values()) > 1e-3:
        raise ValueError(f"{stem}: the re-run's plant-free mat fields differ from capture v1's: {match}")
    arrays = {k: (v if v.dtype == bool else v.astype(np.float32)) for k, v in m["arrays"].items() if not k.startswith("_")}
    hz = base["human_floor_z"]
    counts = {"attr_visible_frames": {z: int(arrays["attr_visible"][:, zi].sum()) for zi, z in enumerate(ZONE_ORDER)},
              "attr_visible_changed_vs_v1": int((arrays["attr_visible"] != base["attr_visible"]).sum()),
              "pairs_within_1cm_frames": {n: int((arrays["avatar_pair_gap"][:, k] <= 0.01).sum())
                                          for k, n in enumerate(hm.PAIR_NAMES) if (arrays["avatar_pair_gap"][:, k] <= 0.01).any()},
              "mat_valid_share_available": bool(np.isfinite(arrays["mat_valid_share"]).any())}
    meta = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, m["inputs"]), "store": f"capture/{STORE_VERSION}",
            **m["meta"], "plant": plant_identity.identity(PLANT),
            "recording_id": base.meta.get("recording_id"), "zone_order": list(ZONE_ORDER), "pair_order": list(hm.PAIR_NAMES),
            "base": _base_identity(base),
            # the keys every consumer of a Capture reads from its meta (v1's live under v3's base; every clip of the
            # corpus has a readable fit, so its markers, and a pressure port)
            "capture_available": base.meta["human_available"], "capture_status": base.meta.get("human_status"),
            "human_available": base.meta["human_available"], "mat_available": True, "mat_status": "ok",
            "mat_share_available": counts["mat_valid_share_available"], "attr_kinematics": "reference",
            "calibration": base.meta["calibration"], "plant_free_mat_max_diff": match,
            "human_floor_z_frames": int(np.isfinite(hz).all(1).sum()), "counts": counts,
            "arrays": {k: {"unit": u, "meaning": d} for k, (u, d) in AVATAR_ARRAYS.items()}}
    npz_path, json_path = _paths(stem, store_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, indent=1) + "\n")
    return capture.Capture(meta, {**base.arrays, **arrays})


def identity(rec: capture.Capture) -> dict:
    """What a v4 record is determined by: this module, its human-side base, the reference and the attribution."""
    return {"generator": rec.meta["generator"]["sha256"], "base": rec.meta["base"],
            "motion_sha256": rec.meta["motion_sha256"], "port_sha256": rec.meta["port_sha256"],
            "plant_sha256": rec.meta["plant"]["plant_sha256"]}


def load(stem: str, store_dir: Path = STORE_DIR, base: capture.Capture | None = None, rid: str = RETARGET_ID,
         rebuild: bool = True) -> capture.Capture:
    """Capture store v4 of ``stem``: store v3 with the plant-v2 avatar side in place of v1's. Rebuilt when missing
    or stale (other code, base, reference or attribution) unless ``rebuild`` is off; refused unless it was built on
    plant v2 (``ids.require_plant``)."""
    base = human_base(stem) if base is None else base
    npz_path, json_path = _paths(stem, store_dir)
    if npz_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        ids.require_plant(meta, f"capture store v4 {stem}", plant_mjcf())
        fresh = (meta.get("schema_version") == SCHEMA_VERSION and meta["generator"]["sha256"] == ids.sha256_file(__file__)
                 and meta["base"] == _base_identity(base) and meta["retarget_id"] == rid
                 and meta["motion_sha256"] == ids.sha256_file(motion_path(stem, rid))
                 and meta["port_sha256"] == ids.sha256_file(pressure_v2.gated_port(stem, rid)))
        if fresh:
            with np.load(npz_path) as npz:
                return capture.Capture(meta, {**base.arrays, **{k: npz[k] for k in npz.files}})
        if not rebuild:
            raise ValueError(f"{json_path} is stale")
    elif not rebuild:
        raise FileNotFoundError(json_path)
    return build(stem, store_dir, base, rid=rid)


def _measure_job(job: tuple) -> tuple[str, dict | None, str | None]:
    stem, rid = job
    torch.set_num_threads(1)
    try:
        return stem, avatar_side(stem, rid), None
    except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
        import traceback

        return stem, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"


def build_all(stems: list[str], store_dir: Path = STORE_DIR, workers: int = 8,
              rid: str = RETARGET_ID) -> tuple[list[capture.Capture], list[str]]:
    """Measure every clip in ``workers`` spawned processes, then write each on its human-side base (read here,
    outside plant v2)."""
    bases = {s: human_base(s) for s in stems}
    jobs = [(s, rid) for s in stems]
    if workers > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_measure_job, jobs))
    else:
        results = [_measure_job(j) for j in jobs]
    records, failures = [], []
    for stem, m, err in results:
        try:
            if err:
                raise RuntimeError(err)
            records.append(build(stem, store_dir, bases[stem], m, rid))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{stem}: {type(exc).__name__}: {exc}")
    return records, failures


# --------------------------------------------------------------------------- #
# The human-side thresholds, re-derived on the plant-v2 attribution (a check, never applied)
# --------------------------------------------------------------------------- #
def crosscheck(records: list[capture.Capture]) -> dict:
    """Capture v1's marker thresholds and v2's mesh touch threshold re-derived by their own rules on the frames the
    plant-v2 attribution confirms (zone load > 50 N, ``attr_visible``, column 1 >= 0.9, markers trusted)."""
    measured = [{"meta": {"stem": r.meta["stem"], "capture_available": True, "mat_available": True,
                          "capture_status": "ok", "mat_status": "ok"},
                 "arrays": {k: r[k] for k in ("marker_min_z", "marker_resid", "mat_zone_load", "attr_visible",
                                               "mat_valid_body")}} for r in records]
    markers = capture.calibrate(measured)
    committed = capture.load_calibration()
    samples = []
    for r in records:
        conf = hm.confirmed_frames(r)
        for z in hm.CALIB_ZONES:
            zi = ZONE_ORDER.index(z)
            samples.append(r["human_min_z"][conf[:, zi], zi])
    pooled = np.concatenate(samples)
    mesh_touch = float(np.percentile(pooled, 99)) + capture.TOUCH_MARGIN_M
    cal_v2 = hm.load_calibration()
    zones = {}
    for z in ZONE_ORDER:
        new, old = markers["zones"][z], committed["zones"][z]
        zones[z] = {"admitted_v1": old["admitted"], "admitted_v4": new["admitted"],
                    "n_frames_v1": old["n_frames"], "n_frames_v4": new["n_frames"],
                    "touch_cm_v1": None if old["touch_m"] is None else round(100 * old["touch_m"], 3),
                    "touch_cm_v4": None if new["touch_m"] is None else round(100 * new["touch_m"], 3)}
    return {"rule": "capture.calibrate and human_mesh.calibrate, unchanged, on the plant-v2 attribution's mat-confirmed "
                    "frames; reported only, the committed thresholds stay in force",
            "markers": zones,
            "mesh": {"touch_cm_committed": round(100 * cal_v2["ground"]["touch_m"], 3),
                     "touch_cm_v4": round(100 * mesh_touch, 3), "n_frames_v4": int(pooled.size),
                     "n_frames_committed": cal_v2["pooled"]["n_frames"]}}


def summarize(records: list[capture.Capture], failures: list[str], seconds: float) -> dict:
    changed = sum(r.meta["counts"]["attr_visible_changed_vs_v1"] for r in records)
    return {"clips": len(records), "frames": sum(r.meta["num_frames"] for r in records), "seconds": round(seconds, 1),
            "attr_visible_zone_frames_changed_vs_v1": changed,
            "plant_free_mat_max_diff": {k: max(r.meta["plant_free_mat_max_diff"][k] for r in records) for k in PLANT_FREE}
            if records else {}, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="every clip of the writer's corpus (56)")
    what.add_argument("--stem", nargs="+")
    ap.add_argument("--store-dir", type=Path, default=STORE_DIR)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    start = time.time()
    try:
        corpus, _ = fw.corpus()
        stems = corpus if args.all else list(args.stem)
        unknown = [s for s in stems if s not in corpus]
        if unknown:
            raise ValueError(f"not in the writer's corpus: {unknown}")
        records, failures = build_all(stems, args.store_dir, args.workers)
        summary = summarize(records, failures, time.time() - start)
        if args.all and not failures:
            cc = crosscheck(records)
            rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [capture.CALIBRATION_PATH, hm.CALIBRATION_PATH]),
                   "store": f"capture/{STORE_VERSION}", "retarget_id": RETARGET_ID, **cc}
            CROSSCHECK_PATH.write_text(json.dumps(rec, indent=1) + "\n")
            summary["crosscheck"] = cc
            (Path(args.store_dir) / "_summary.json").write_text(json.dumps({**ids.provenance(
                SCHEMA_VERSION, MODULE, __file__, [CROSSCHECK_PATH]), **summary}, indent=1) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    extra = ""
    if "crosscheck" in summary:
        mk = summary["crosscheck"]["markers"]
        extra = ("; re-derived touch (cm) " + ", ".join(f"{z} {mk[z]['touch_cm_v1']}->{mk[z]['touch_cm_v4']}"
                                                         for z in hm.CALIB_ZONES)
                 + f", mesh {summary['crosscheck']['mesh']['touch_cm_committed']}->{summary['crosscheck']['mesh']['touch_cm_v4']}")
    print(f"capture {STORE_VERSION}: {len(records)} clips, {summary['frames']} frames on plant v2 in {summary['seconds']:.0f} s; "
          f"attr_visible changed on {summary['attr_visible_zone_frames_changed_vs_v1']} zone-frames vs v1{extra}; "
          f"{len(failures)} failures -> {ids.display_path(args.store_dir)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
