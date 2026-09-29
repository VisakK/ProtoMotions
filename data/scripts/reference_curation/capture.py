# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture evidence store (BUILD_PLAN Step 1): what the human did, what the reference does and
what the mat says, for every clip, frame and zone.

``build(stem)`` writes ``output/reference_curation/capture/v1/<stem>.{npz,json}``; ``load(stem)``
returns it as a :class:`Capture`, rebuilding it if the code or the calibration changed. Arrays are
``[T]`` per frame or ``[T, Z]`` per zone, in ``extract_contact_configs.ZONE_ORDER`` (Z = 15);
``ARRAYS`` below documents each one, and every JSON repeats that table.

Evidence and what it can say
----------------------------
* **Human, from the markers.** ``markers_obs`` of the MoSh++ stage-II fit are the raw Vicon
  markers (metres, Z up, floor at z = 0), frame-aligned 1:1 with the 60 fps clips.
  ``ground_state`` is *touch*, not load: a relaxed limb resting on the floor reads the same height
  as a loaded one (unloaded p5 ~ loaded p50). It is a Schmitt trigger on the zone's lowest
  trustworthy marker: contact once it is at or below ``touch``, separated once it rises above
  ``separation``, unchanged in between. A marker is untrustworthy on a frame where it is NaN or
  sits more than 5 cm from its fitted position (``markers_sim``); the zone is *unknown* (-1) where
  none is left, where the zone's median residual exceeds 5 cm, and before its first decisive frame.
* **Reference, from the shipped ``.motion``.** ``avatar_min_z`` is the lowest collision surface of
  the zone's bodies (all their geoms, ``contact_geometry``).
* **Mat, from the MOYO pressure port.** Load comes only from here: the attributed ``mat_zone_load``
  where ``attr_visible``, else ``mat_unexplained``. The validity columns stay separate, because
  they gate different things: column 0 the total and COP, column 1 absolute per-body load,
  column 2 load *shares* (only the gated port has it).

Thresholds
----------
Calibrated once, over a corpus (default: the manifest's 60 ftC clips), on mat-confirmed frames:
zone load > 50 N, ``attr_visible``, column 1 >= 0.9, markers trustworthy. Then
``touch = p99 + 0.5 cm`` of the loaded lowest-marker height and ``separation = touch + 2.5 cm``.
A zone's marker test is **admitted** only if that loaded distribution is tight
(``p99 - p50 <= 1 cm``) over >= 500 frames from >= 3 clips. On the 60 ftC clips that admits the
feet, hands and head. The other ten zones fail it: their markers do not sit on the surface that
touches the floor. Thigh markers read 14-18 cm when loaded, and trunk markers under a lying back
are gap-filled to 1-2.5 cm *below* the floor. Those zones stay -1: their human-side contact comes
from the mesh (Step 5), not from markers.

Traps
-----
* Clip XY = Vicon XY + ``VICON_TO_CLIP_XY``; Z is shared, floor at 0 (``markers()``).
* x0 clips only; a variant's frames map back through ``ids.source_frame_index``.
* The markers are gap-filled upstream (no NaN anywhere in this data), so an occluded marker is
  interpolated, not missing. That is why the hygiene rule is on the fit residual.
* The attribution ran on the pressure port's own kinematics: the unrepaired grounded source, which
  differs from the shipped pose on the 8 repaired clips. ``attr_visible`` is measured on that
  pose, never on the shipped one.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.capture --all
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from contact_geometry import geom_ground_distance, geom_to_world, parse_typed_geoms
from extract_contact_configs import ZONE_ORDER, ZONES, mjcf_body_names
from reference_curation import ids

MODULE = "reference_curation.capture"
SCHEMA_VERSION = 1
STORE_VERSION = "v1"
STORE_DIR = ids.OUTPUT_ROOT / "capture" / STORE_VERSION
CALIBRATION_PATH = ids.DATA_ROOT / "calibration" / f"capture_{STORE_VERSION}.json"

VICON_TO_CLIP_XY = (-0.002, 0.3452)  # README §3.2 / notes/Moyo_pressure_port.MD
ATTR_BAND_M = 0.06        # attribute_pressure_to_bodies.py: only geoms this close compete for load
RESID_MAX_M = 0.05        # marker hygiene
CALIB_LOAD_N = 50.0
CALIB_BODY_GATE = 0.9     # column 1, the gate build_physics_tables.py applies to it
TOUCH_MARGIN_M = 0.005
SEPARATION_GAP_M = 0.025
ADMIT_SPREAD_M = 0.010
ADMIT_MIN_FRAMES = 500
ADMIT_MIN_CLIPS = 3

# 73 MOYO markers -> 15 zones. ANK, KNE, ELB, ELBIN, IWR and OWR each serve two zones.
# The middle finger carries MID0/MID6 (not 3/6) in this layout.
_SIDED = {
    "FOOT": ("HEE", "TOE", "MT1", "MT5", "ANK"),
    "HAND": ("FIN", "THMB", "IDX3", "IDX6", "MID0", "MID6", "PNK3", "PNK6", "RNG3", "RNG6",
             "THM3", "THM6", "IWR", "OWR"),
    "SHANK": ("SHN", "KNE", "KNI", "ANK"),
    "THIGH": ("THI", "KNE"),
    "UPPER_ARM": ("UPA", "ELB", "ELBIN"),
    "FOREARM": ("FRM", "ELB", "ELBIN", "IWR", "OWR"),
}
_CENTRAL = {
    "HEAD": ("LFHD", "RFHD", "LBHD", "RBHD", "ARIEL"),
    "PELVIS": ("LFWT", "RFWT", "MFWT", "LBWT", "RBWT", "MBWT"),
    "TRUNK": ("C7", "T10", "CLAV", "STRN", "LFSH", "RFSH", "LBSH", "RBSH"),
}
ZONE_MARKERS = {z: list(_CENTRAL[z]) if z in _CENTRAL else [z[0] + m for m in _SIDED[z[2:]]]
                for z in ZONE_ORDER}
# The review's exemplar audit (README §3.2) read only the floor-facing markers of the feet and
# hands: no ANK, IWR or OWR. Its pooled-threshold counts need this map. With the wrist markers,
# Bridge -a's lowest "hand" marker is the outer wrist at 3.5-4.4 cm (its forearms lie on the
# floor) instead of the fingers at 6.6-7.3 cm, and 3 of the review's 4 lifted supports go.
REVIEW_ZONE_MARKERS = {z: [m for m in ms if m[1:] not in ("ANK", "IWR", "OWR")]
                       if z in ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND") else ms
                       for z, ms in ZONE_MARKERS.items()}
NUM_MARKERS = 73

ARRAYS = {
    "marker_min_z": ("m", "[T,Z] height of the zone's lowest trustworthy observed marker"),
    "marker_resid": ("m", "[T,Z] median |obs - fit| of the zone's markers"),
    "avatar_min_z": ("m", "[T,Z] lowest collision surface of the zone in the shipped .motion"),
    "mat_total": ("N", "[T] total mat force (ground_reaction[:, 0])"),
    "mat_cop": ("m", "[T,2] centre of pressure, clip frame"),
    "mat_valid_cov": ("-", "[T] validity column 0, coverage: gates mat_total and mat_cop"),
    "mat_valid_body": ("-", "[T] validity column 1, coverage x explained: gates mat_zone_load"),
    "mat_valid_share": ("-", "[T] validity column 2, on-mat x explained: gates load shares; "
                             "NaN if the port has no column 2"),
    "mat_zone_load": ("N", "[T,Z] load the attribution put on the zone's bodies"),
    "mat_unexplained": ("N", "[T] mat_total minus all attributed load"),
    "attr_visible": ("bool", "[T,Z] zone within 6 cm of the floor in the pose the attribution ran on"),
    "ground_state": ("int8", "[T,Z] human touch from the markers: 1 contact, 0 separated, -1 unknown"),
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_motion(path: Path) -> dict:
    motion = torch.load(str(path), map_location="cpu", weights_only=False)
    order = getattr(motion.get("state_conversion"), "value", motion.get("state_conversion"))
    if order not in (None, "common"):  # per-body arrays below assume COMMON (MJCF) body order
        raise ValueError(f"{path}: state_conversion {order!r}, expected COMMON body order")
    return motion


def load_mosh(stem: str) -> tuple[dict | None, str, str | None]:
    """``(fit, status, error)``: ``fit`` holds ``obs``/``sim`` ``[T,73,3]``, ``labels``, ``fps``,
    ``path``; ``status`` is ``ok`` / ``no_fit`` / ``unreadable``."""
    path = ids.mosh_path(stem)
    if path is None:
        return None, "no_fit", "MOYO has no MoSh fit for this clip"
    try:
        with open(path, "rb") as f:
            debug = pickle.load(f, encoding="latin1")["stageii_debug_details"]
        obs = np.asarray(debug["markers_obs"], dtype=np.float64)
        sim = np.asarray(debug["markers_sim"], dtype=np.float64)
        labels_per_frame = np.asarray(debug["labels_obs"])
        fps = float(debug["mocap_frame_rate"])
    except Exception as exc:  # noqa: BLE001 -- one fit on disk is corrupt (Standing big toe hold -c)
        return None, "unreadable", f"{type(exc).__name__}: {exc}"
    labels = [str(x) for x in labels_per_frame[0]]
    if (labels_per_frame != labels_per_frame[0]).any():
        raise ValueError(f"{path}: marker labels change across frames")
    missing = sorted({m for ms in ZONE_MARKERS.values() for m in ms} - set(labels))
    if missing or len(labels) != NUM_MARKERS:
        raise ValueError(f"{path}: unexpected marker layout ({len(labels)} labels, missing {missing})")
    return {"obs": obs, "sim": sim, "labels": labels, "fps": fps, "path": path}, "ok", None


def load_pressure(stem: str) -> tuple[dict | None, Path | None]:
    """The clip's pressure port (gated first) with all three measured fields, or ``(None, None)``."""
    for path in ids.pressure_paths(stem):
        port = _load_motion(path)
        if all(k in port for k in ("ground_reaction", "rigid_body_ground_forces", "ground_reaction_valid")):
            return port, path
    return None, None


def markers(stem: str, which: str = "obs") -> tuple[np.ndarray, list[str]] | None:
    """``([T,73,3] markers in the clip frame, labels)``; ``which`` is ``obs`` or ``sim``."""
    fit, _, _ = load_mosh(stem)
    if fit is None:
        return None
    xyz = fit[which].copy()
    xyz[..., :2] += np.asarray(VICON_TO_CLIP_XY)
    return xyz, fit["labels"]


# --------------------------------------------------------------------------- #
# Pure measurements
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def skeleton() -> SimpleNamespace:
    """The MJCF's bodies (COMMON order), their typed collision geoms and each zone's body indices."""
    names = mjcf_body_names(str(ids.MJCF))
    zone_bodies = [[names.index(b) for b in ZONES[z]] for z in ZONE_ORDER]
    if sorted(i for idx in zone_bodies for i in idx) != list(range(len(names))):
        raise ValueError("ZONES must partition the bodies once each")
    return SimpleNamespace(names=names, geoms=parse_typed_geoms(str(ids.MJCF), names), zone_bodies=zone_bodies)


def body_min_z(pos, rot) -> np.ndarray:
    """``[T, B]`` lowest collision surface of every body (COMMON order), min over its geoms."""
    sk = skeleton()
    pos, rot = torch.as_tensor(pos).float(), torch.as_tensor(rot).float()
    if pos.shape[1] != len(sk.names):
        raise ValueError(f"{pos.shape[1]} bodies, the MJCF has {len(sk.names)}")
    per_body = []
    for i, body in enumerate(sk.names):
        gaps = [geom_ground_distance(geom_to_world(g, pos[:, i], rot[:, i]))[0] for g in sk.geoms[body]]
        per_body.append(torch.stack(gaps, -1).min(-1).values)
    return torch.stack(per_body, -1).numpy()


def zone_min(per_body: np.ndarray) -> np.ndarray:
    return np.stack([per_body[:, idx].min(-1) for idx in skeleton().zone_bodies], -1)


def zone_sum(per_body: np.ndarray) -> np.ndarray:
    return np.stack([per_body[:, idx].sum(-1) for idx in skeleton().zone_bodies], -1)


def marker_arrays(obs: np.ndarray, sim: np.ndarray, labels: list[str], zone_markers: dict = ZONE_MARKERS):
    """``(min_z [T,Z], resid [T,Z], excluded [T,Z])``: lowest trustworthy marker, median residual,
    and how many of the zone's markers were untrustworthy (NaN or residual > 5 cm)."""
    resid = np.linalg.norm(obs - sim, axis=-1)
    trusted = np.isfinite(obs).all(-1) & (np.nan_to_num(resid, nan=np.inf) <= RESID_MAX_M)
    height = np.where(trusted, obs[..., 2], np.inf)
    min_z, med, excluded = [], [], []
    for zone in ZONE_ORDER:
        idx = [labels.index(m) for m in zone_markers[zone]]
        low = height[:, idx].min(-1)
        min_z.append(np.where(np.isfinite(low), low, np.nan))
        with warnings.catch_warnings():  # all-NaN rows -> NaN, which is what we want
            warnings.simplefilter("ignore", RuntimeWarning)
            med.append(np.nanmedian(resid[:, idx], -1))
        excluded.append((~trusted[:, idx]).sum(-1))
    return np.stack(min_z, -1), np.stack(med, -1), np.stack(excluded, -1)


def contact_state(height: np.ndarray, known: np.ndarray, touch: float, separation: float) -> np.ndarray:
    """``[T]`` int8 Schmitt trigger: 1 from a frame at or below ``touch``, 0 from one above
    ``separation``, the last decisive state in between. -1 where not ``known`` and before the first
    decisive frame. Memory carries across unknown frames."""
    with np.errstate(invalid="ignore"):
        touching = known & (height <= touch)
        decisive = touching | (known & (height > separation))
    last = np.maximum.accumulate(np.where(decisive, np.arange(len(height)), -1))
    state = np.where(last >= 0, touching[np.maximum(last, 0)], -1).astype(np.int8)
    state[~known] = -1
    return state


def zone_known(min_z: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """``[T,Z]`` marker hygiene: a trustworthy marker exists and the zone's median residual <= 5 cm."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(min_z) & (resid <= RESID_MAX_M)


def ground_state(min_z: np.ndarray, resid: np.ndarray, zones: dict) -> np.ndarray:
    """``[T,Z]`` int8 human touch; ``zones`` is the calibration's per-zone table."""
    known = zone_known(min_z, resid)
    state = np.full(min_z.shape, -1, dtype=np.int8)
    for zi, zone in enumerate(ZONE_ORDER):
        th = zones[zone]
        if th["admitted"]:
            state[:, zi] = contact_state(min_z[:, zi], known[:, zi], th["touch_m"], th["separation_m"])
    return state


# --------------------------------------------------------------------------- #
# One clip, before thresholds
# --------------------------------------------------------------------------- #
def measure(stem: str, motion_dir: Path = ids.SHIPPED_DIR) -> dict:
    """Everything but ``ground_state`` for one x0 clip: ``{"arrays", "meta", "inputs"}``."""
    if ids.split_clip_name(stem)[1]:
        raise ValueError(f"{stem}: build on x0 clips only; map variant frames with ids.source_frame_index")
    motion_file = ids.motion_path(stem, motion_dir)
    motion = _load_motion(motion_file)
    T, fps = motion["rigid_body_pos"].shape[0], int(motion["fps"])
    avatar = zone_min(body_min_z(motion["rigid_body_pos"], motion["rigid_body_rot"]))
    Z = len(ZONE_ORDER)
    nan_tz, nan_t = np.full((T, Z), np.nan), np.full(T, np.nan)
    inputs = [ids.MJCF, motion_file]
    meta = {"stem": stem, "recording_id": ids.recording_id(stem), "fps": fps, "num_frames": T,
            "zone_order": list(ZONE_ORDER)}

    # Human.
    fit, status, error = load_mosh(stem)
    if fit is not None:
        inputs.append(fit["path"])
        if fit["obs"].shape[0] != T or fit["fps"] != fps:
            status, error = "frame_mismatch", (f"MoSh {fit['obs'].shape[0]} frames at {fit['fps']:g} fps, "
                                               f"motion {T} at {fps} fps")
    if status == "ok":
        min_z, resid, excluded = marker_arrays(fit["obs"], fit["sim"], fit["labels"])
    else:
        min_z, resid, excluded = nan_tz.copy(), nan_tz.copy(), np.zeros((T, Z), dtype=int)
    known = zone_known(min_z, resid)
    meta.update(capture_available=status == "ok", capture_status=status, capture_error=error)
    meta["hygiene"] = {z: {"markers_excluded": int(excluded[:, zi].sum()),
                           "frames_no_marker": int((~np.isfinite(min_z[:, zi])).sum()) if status == "ok" else T,
                           "frames_resid_over": int((np.isfinite(min_z[:, zi]) & ~known[:, zi]).sum())}
                       for zi, z in enumerate(ZONE_ORDER)}

    # Mat.
    port, port_file = load_pressure(stem)
    mat_status = "no_port" if port is None else "ok"
    if port is not None:
        inputs.append(port_file)
        if port["rigid_body_pos"].shape[0] != T:
            mat_status = "frame_mismatch"
    if mat_status == "ok":
        gr = port["ground_reaction"].double().numpy()
        force = port["rigid_body_ground_forces"][..., 2].double().numpy()
        valid = port["ground_reaction_valid"].double().numpy()
        mat = {"mat_total": gr[:, 0], "mat_cop": gr[:, 1:3],
               "mat_valid_cov": valid[:, 0], "mat_valid_body": valid[:, 1],
               "mat_valid_share": valid[:, 2] if valid.shape[1] > 2 else nan_t.copy(),
               "mat_zone_load": zone_sum(force), "mat_unexplained": gr[:, 0] - force.sum(-1)}
        same_pose = bool(torch.equal(port["rigid_body_pos"], motion["rigid_body_pos"]))
        attr_z = avatar if same_pose else zone_min(body_min_z(port["rigid_body_pos"], port["rigid_body_rot"]))
    else:
        mat = {"mat_total": nan_t.copy(), "mat_cop": np.full((T, 2), np.nan), "mat_valid_cov": nan_t.copy(),
               "mat_valid_body": nan_t.copy(), "mat_valid_share": nan_t.copy(),
               "mat_zone_load": nan_tz.copy(), "mat_unexplained": nan_t.copy()}
        same_pose, attr_z = True, avatar
    attr_visible = attr_z <= ATTR_BAND_M
    meta.update(mat_available=mat_status == "ok", mat_status=mat_status,
                mat_port=None if port_file is None else ids.display_path(port_file),
                mat_share_available=bool(mat_status == "ok" and np.isfinite(mat["mat_valid_share"]).any()),
                attr_kinematics="pressure_port" if mat_status == "ok" else "shipped",
                attr_pose_matches_shipped=same_pose,
                attr_visible_vs_shipped_disagree=int(((avatar <= ATTR_BAND_M) != attr_visible).sum()))

    arrays = {"marker_min_z": min_z, "marker_resid": resid, "avatar_min_z": avatar, **mat,
              "attr_visible": attr_visible}
    arrays = {k: (v if v.dtype == bool else v.astype(np.float32)) for k, v in arrays.items()}
    return {"arrays": arrays, "meta": meta, "inputs": inputs}


# --------------------------------------------------------------------------- #
# Corpus calibration
# --------------------------------------------------------------------------- #
def calibrate(measured: list[dict]) -> dict:
    """Per-zone touch/separation thresholds and admission from mat-confirmed frames."""
    samples = {z: [] for z in ZONE_ORDER}
    clips = {z: set() for z in ZONE_ORDER}
    used, excluded = [], {}
    for m in measured:
        meta, a = m["meta"], m["arrays"]
        if not (meta["capture_available"] and meta["mat_available"]):
            excluded[meta["stem"]] = f"capture {meta['capture_status']}, mat {meta['mat_status']}"
            continue
        used.append(meta["stem"])
        known = zone_known(a["marker_min_z"], a["marker_resid"])
        with np.errstate(invalid="ignore"):
            confirmed = ((a["mat_zone_load"] > CALIB_LOAD_N) & a["attr_visible"] & known
                         & (a["mat_valid_body"] >= CALIB_BODY_GATE)[:, None])
        for zi, z in enumerate(ZONE_ORDER):
            h = a["marker_min_z"][confirmed[:, zi], zi].astype(np.float64)
            if h.size:
                samples[z].append(h)
                clips[z].add(meta["stem"])
    zones = {}
    for z in ZONE_ORDER:
        h = np.concatenate(samples[z]) if samples[z] else np.zeros(0)
        row = {"n_frames": int(h.size), "n_clips": len(clips[z])}
        if h.size:
            q = dict(zip(("p1", "p5", "p50", "p90", "p99"), np.percentile(h, [1, 5, 50, 90, 99]).tolist()))
            row.update(q, max=float(h.max()), spread_m=q["p99"] - q["p50"])
            row["touch_m"] = q["p99"] + TOUCH_MARGIN_M
            row["separation_m"] = row["touch_m"] + SEPARATION_GAP_M
        else:
            row.update(touch_m=None, separation_m=None, spread_m=None)
        failed = [msg for bad, msg in (
            (row["n_frames"] < ADMIT_MIN_FRAMES, f"{row['n_frames']} < {ADMIT_MIN_FRAMES} frames"),
            (row["n_clips"] < ADMIT_MIN_CLIPS, f"{row['n_clips']} < {ADMIT_MIN_CLIPS} clips"),
            (row["spread_m"] is not None and row["spread_m"] > ADMIT_SPREAD_M,
             f"p99 - p50 = {100 * (row['spread_m'] or 0):.2f} cm > {100 * ADMIT_SPREAD_M:.1f} cm"),
        ) if bad]
        row.update(admitted=not failed, reason="; ".join(failed) or "admitted")
        zones[z] = row
    rule = {"load_n": CALIB_LOAD_N, "body_gate_col1": CALIB_BODY_GATE, "attr_band_m": ATTR_BAND_M,
            "resid_max_m": RESID_MAX_M, "touch": f"p99 + {TOUCH_MARGIN_M} m",
            "separation": f"touch + {SEPARATION_GAP_M} m",
            "admit": {"spread_p99_p50_max_m": ADMIT_SPREAD_M, "min_frames": ADMIT_MIN_FRAMES,
                      "min_clips": ADMIT_MIN_CLIPS}}
    decisive = {z: {k: zones[z][k] for k in ("touch_m", "separation_m", "admitted")} for z in ZONE_ORDER}
    return {"store": f"capture/{STORE_VERSION}", "rule": rule, "marker_zones": ZONE_MARKERS,
            "corpus": {"stems": used, "excluded": excluded},
            "zones": zones, "id": ids.sha256_json({"rule": rule, "zones": decisive, "markers": ZONE_MARKERS})}


def write_calibration(calibration: dict, measured: list[dict], path: Path = CALIBRATION_PATH) -> dict:
    inputs = sorted({p for m in measured for p in m["inputs"]}, key=str)
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), **calibration}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing; run `-m reference_curation.capture --all` first")
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Capture:
    meta: dict
    arrays: dict

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    @property
    def fps(self) -> int:
        return self.meta["fps"]

    def frame(self, t_s: float) -> int:
        return int(round(t_s * self.fps))

    @staticmethod
    def zi(zone: str) -> int:
        return ZONE_ORDER.index(zone)

    def support(self, frame: int) -> frozenset:
        """Zones the human touches the floor with on ``frame``."""
        return frozenset(z for zi, z in enumerate(ZONE_ORDER) if self.arrays["ground_state"][frame, zi] == 1)

    def unknown(self, frame: int) -> frozenset:
        return frozenset(z for zi, z in enumerate(ZONE_ORDER) if self.arrays["ground_state"][frame, zi] == -1)


def _paths(stem: str, store_dir: Path) -> tuple[Path, Path]:
    return Path(store_dir) / f"{stem}.npz", Path(store_dir) / f"{stem}.json"


def build(stem: str, calibration: dict | None = None, store_dir: Path = STORE_DIR,
          measured: dict | None = None) -> Capture:
    """Measure ``stem``, apply the calibration and write ``<store_dir>/<stem>.{npz,json}``."""
    cal = load_calibration() if calibration is None else calibration
    m = measure(stem) if measured is None else measured
    arrays = dict(m["arrays"])
    arrays["ground_state"] = ground_state(arrays["marker_min_z"], arrays["marker_resid"], cal["zones"])
    counts = {z: {"contact": int((arrays["ground_state"][:, zi] == 1).sum()),
                  "separated": int((arrays["ground_state"][:, zi] == 0).sum()),
                  "unknown": int((arrays["ground_state"][:, zi] == -1).sum())}
              for zi, z in enumerate(ZONE_ORDER)}
    meta = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, m["inputs"]),
            "store": f"capture/{STORE_VERSION}", **m["meta"],
            "calibration": {"id": cal["id"], "path": cal.get("path")},
            "thresholds": {z: {k: cal["zones"][z][k] for k in ("touch_m", "separation_m", "admitted")}
                           for z in ZONE_ORDER},
            "registration": {"vicon_to_clip_xy": list(VICON_TO_CLIP_XY), "z": "shared, floor at 0"},
            "ground_state_counts": counts,
            "arrays": {k: {"unit": u, "meaning": d} for k, (u, d) in ARRAYS.items()}}
    npz_path, json_path = _paths(stem, store_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, indent=1) + "\n")
    return Capture(meta, arrays)


def load(stem: str, store_dir: Path = STORE_DIR, calibration: dict | None = None) -> Capture:
    """The stored record, rebuilt if missing or made by other code or another calibration."""
    cal = load_calibration() if calibration is None else calibration
    npz_path, json_path = _paths(stem, store_dir)
    if npz_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        if (meta.get("schema_version") == SCHEMA_VERSION
                and meta["generator"]["sha256"] == ids.sha256_file(__file__)
                and meta["calibration"]["id"] == cal["id"]):
            with np.load(npz_path) as npz:
                return Capture(meta, {k: npz[k] for k in npz.files})
    return build(stem, cal, store_dir)


def build_all(stems: list[str], store_dir: Path = STORE_DIR,
              calibration_path: Path | None = CALIBRATION_PATH) -> tuple[dict, list[dict], list[str]]:
    """Measure every clip, calibrate on them, build the store. ``(calibration, metas, failures)``."""
    measured, failures = [], []
    for stem in stems:
        try:
            measured.append(measure(stem))
        except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
            failures.append(f"{stem}: {type(exc).__name__}: {exc}")
    cal = calibrate(measured)
    if calibration_path is not None:
        cal = write_calibration(cal, measured, calibration_path)
        cal["path"] = ids.display_path(calibration_path)
    metas = [build(m["meta"]["stem"], cal, store_dir, measured=m).meta for m in measured]
    for m in metas:  # misaligned evidence is a failure; a missing or unreadable fit is data
        if m["capture_status"] == "frame_mismatch":
            failures.append(f"{m['stem']}: {m['capture_error']}")
        if m["mat_status"] == "frame_mismatch":
            failures.append(f"{m['stem']}: pressure port frame count differs from the motion")
    return cal, metas, failures


def summarize(cal: dict, metas: list[dict], failures: list[str], seconds: float) -> dict:
    admitted = [z for z in ZONE_ORDER if cal["zones"][z]["admitted"]]
    frames = sum(m["num_frames"] for m in metas)
    with_capture = [m for m in metas if m["capture_available"]]
    known = sum(m["ground_state_counts"][z]["contact"] + m["ground_state_counts"][z]["separated"]
                for m in with_capture for z in admitted)
    total = sum(m["num_frames"] for m in with_capture) * len(admitted)
    return {
        "clips": len(metas), "frames": frames, "seconds": round(seconds, 1),
        "capture_unavailable": {m["stem"]: m["capture_error"] for m in metas if not m["capture_available"]},
        "mat_unavailable": [m["stem"] for m in metas if not m["mat_available"]],
        "no_share_column": [m["stem"] for m in metas if m["mat_available"] and not m["mat_share_available"]],
        "admitted_zones": admitted,
        "unknown_fraction_admitted": 1.0 - known / total if total else None,
        "hygiene": {z: {k: sum(m["hygiene"][z][k] for m in with_capture)
                        for k in ("markers_excluded", "frames_no_marker", "frames_resid_over")}
                    for z in ZONE_ORDER},
        "attr_visible_vs_shipped_disagree": {m["stem"]: m["attr_visible_vs_shipped_disagree"]
                                             for m in metas if m["attr_visible_vs_shipped_disagree"]},
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="calibrate on the manifest's clips and build them all")
    what.add_argument("--stem", nargs="+", help="build these x0 clips with the existing calibration")
    ap.add_argument("--manifest", type=Path, default=ids.DEFAULT_MANIFEST)
    ap.add_argument("--store-dir", type=Path, default=STORE_DIR)
    ap.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    ap.add_argument("--max-seconds", type=float, default=600.0, help="fail a full build slower than this")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print the threshold table")
    args = ap.parse_args(argv)

    start = time.time()
    if args.stem:
        cal = load_calibration(args.calibration)
        cal["path"] = ids.display_path(args.calibration)
        metas, failures = [], []
        for stem in args.stem:
            try:
                metas.append(build(stem, cal, args.store_dir).meta)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{stem}: {type(exc).__name__}: {exc}")
    else:
        cal, metas, failures = build_all(ids.manifest_stems(args.manifest), args.store_dir, args.calibration)
    seconds = time.time() - start
    if args.all and seconds > args.max_seconds:
        failures.append(f"took {seconds:.0f} s > budget {args.max_seconds:.0f} s")
    summary = summarize(cal, metas, failures, seconds)
    if args.all:
        record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [args.manifest, args.calibration]),
                  "calibration_id": cal["id"], **summary}
        (Path(args.store_dir) / "_summary.json").write_text(json.dumps(record, indent=1) + "\n")
    if args.verbose:
        for z in ZONE_ORDER:
            r = cal["zones"][z]
            cm = (lambda v: "   -" if v is None else f"{100 * v:5.2f}")
            print(f"  {z:12s} n={r['n_frames']:6d} clips={r['n_clips']:2d} p50={cm(r.get('p50'))} "
                  f"p99={cm(r.get('p99'))} touch={cm(r['touch_m'])} sep={cm(r['separation_m'])}  {r['reason']}")
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    unknown = summary["unknown_fraction_admitted"]
    print(f"capture {STORE_VERSION}: {len(metas)} clips, {summary['frames']} frames in {seconds:.1f} s; "
          f"no capture: {len(summary['capture_unavailable'])}; admitted {'/'.join(summary['admitted_zones'])}; "
          f"unknown {'-' if unknown is None else f'{100 * unknown:.2f} %'} of admitted zone-frames; "
          f"{len(failures)} failures -> {ids.display_path(args.store_dir)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
