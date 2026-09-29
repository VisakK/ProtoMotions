# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identities: clip names, MOYO recordings, file paths, hold ids and provenance records.

Every module in the package gets its paths and ids from here, so the naming rules live in
one place.

Names
-----
``recording_id``
    The MOYO recording name, punctuation kept:
    ``220923_yogi_body_hands_03596_Crane_(Crow)_Pose_or_Bakasana_-a``. The MoSh++ fit is
    ``<MOYO>/mosh/{train,val}/<recording_id>_stageii.pkl``.
``stem``
    The project clip name: the session, then the MOYO pose name with ``(`` and ``)`` dropped:
    ``220923_Crane_Crow_Pose_or_Bakasana_-a``. Those two characters are the only punctuation
    in any MOYO name (all 337 fits), so the mapping is exact and nothing is fuzzy-matched
    (``moyo_pressure_io.py`` records why fuzzy matching pairs the wrong takes). A second
    session (``221004_yogi_nexus_…``) reuses the pose names, and its date keeps the stems
    distinct.
clip name
    A file stem in a motion directory: ``<stem>`` for the x0 clip, ``<stem>_x3s`` /
    ``<stem>_x7s`` for the hold-extended variants (``make_hold_extended_clips.py``).
``hold_id``
    ``<stem>@<frame_hold>``, always in x0 source frames. A variant's frames map back through
    ``make_hold_extended_clips.splice_index`` (``source_frame_index``).

Provenance
----------
``provenance()`` returns the block every JSON record carries: ``schema_version``,
``generator`` {module, git rev, file sha256} and ``inputs`` {path: sha256}. Paths inside the
repo are repo-relative; the MOYO files outside it are absolute.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
MOYO_DATA = Path(os.environ.get("MOYO_DATA", REPO.parents[1] / "moyo_toolkit" / "data"))
MJCF = REPO / "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
DEFAULT_MANIFEST = REPO / "data/smpl/expert60/holds_repaired_ftC_posefix.yaml"
SHIPPED_DIR = REPO / "data/smpl/yoga_motions_proto_yogi_expert60_ftC"
EXTENDED_MANIFEST = SHIPPED_DIR / "holds_extended.yaml"
# The gated port carries validity column 2 (on-mat x explained); prefer it.
PRESSURE_DIRS = (REPO / "data/smpl/yoga_motions_proto_yogi_pressure_gated",
                 REPO / "data/smpl/yoga_motions_proto_yogi_pressure")

OUTPUT_ROOT = REPO / "output/reference_curation"  # bulky, regenerable (git-ignored)
DATA_ROOT = REPO / "data/reference_curation"       # small, precious
# Packets for blind review live outside the repo, so `claude -p` loads no CLAUDE.md.
REVIEW_ROOT = Path(os.environ.get("REVIEW_ROOT", REPO.parent / "reference_review"))

_RECORDING_RE = re.compile(r"^(\d{6})_yogi(?:_nexus)?_body_hands_03596_(.+)$")
_VARIANT_RE = re.compile(r"^(.+)_x(\d+(?:\.\d+)?)s$")
_HOLD_RE = re.compile(r"^(.+)@(\d+)$")
_MOSH_SUFFIX = "_stageii.pkl"


# --------------------------------------------------------------------------- #
# Clips and recordings
# --------------------------------------------------------------------------- #
def stem_of_recording(recording: str) -> str:
    """``220923_yogi_body_hands_03596_Crane_(Crow)_…`` -> ``220923_Crane_Crow_…``."""
    m = _RECORDING_RE.match(recording)
    if m is None:
        raise ValueError(f"not a MOYO recording name: {recording!r}")
    session, pose = m.groups()
    return f"{session}_{pose.replace('(', '').replace(')', '')}"


@functools.lru_cache(maxsize=None)
def _mosh_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(MOYO_DATA.glob(f"mosh/*/*{_MOSH_SUFFIX}")):
        stem = stem_of_recording(path.name[: -len(_MOSH_SUFFIX)])
        if stem in index:
            raise ValueError(f"two MoSh fits map to {stem}: {index[stem]} and {path}")
        index[stem] = path
    return index


def mosh_path(stem: str) -> Path | None:
    """The clip's MoSh++ stage-II fit, or ``None`` if MOYO has none (e.g. hand-cut subclips)."""
    return _mosh_index().get(stem)


def recording_id(stem: str) -> str | None:
    path = mosh_path(stem)
    return None if path is None else path.name[: -len(_MOSH_SUFFIX)]


def pressure_paths(stem: str) -> list[Path]:
    """The clip's MOYO pressure ports that exist, gated first."""
    return [d / f"{stem}.motion" for d in PRESSURE_DIRS if (d / f"{stem}.motion").exists()]


def split_clip_name(name: str) -> tuple[str, float]:
    """``<stem>_x3s`` -> ``(<stem>, 3.0)``; an x0 name -> ``(name, 0.0)``."""
    m = _VARIANT_RE.match(name)
    return (m.group(1), float(m.group(2))) if m else (name, 0.0)


def motion_path(name: str, motion_dir: Path = SHIPPED_DIR) -> Path:
    """The shipped ``.motion`` of an x0 clip or a variant."""
    return Path(motion_dir) / f"{name}.motion"


# --------------------------------------------------------------------------- #
# Manifests and holds
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=8)
def _load_yaml(path: str, size: int, mtime_ns: int) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    """A hold manifest (``propose_hold_manifest.py`` format), cached until the file changes.
    Do not mutate it."""
    st = os.stat(path)
    return _load_yaml(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


def manifest_stems(path: Path = DEFAULT_MANIFEST) -> list[str]:
    return [c["stem"] for c in load_manifest(path)["clips"]]


def source_frame_index(name: str, extended_manifest: Path = EXTENDED_MANIFEST) -> np.ndarray:
    """``[T_clip]`` x0 source frame of every frame of clip ``name`` (the identity for x0)."""
    from make_hold_extended_clips import insertion_plan, splice_index  # heavy import, used rarely

    ext = load_manifest(extended_manifest)
    entry = next((c for c in ext["clips"] if c["stem"] == name), None)
    if entry is None:
        raise KeyError(f"{name} is not in {extended_manifest}")
    source = next(c for c in load_manifest(Path(ext["source_manifest"]))["clips"]
                  if c["stem"] == entry["source_stem"])
    plan = insertion_plan(source["holds"], int(round(entry["variant_s"] * entry["fps"])))
    index = splice_index(int(source["num_frames"]), plan).numpy()
    if len(index) != entry["num_frames"] or len(index) - source["num_frames"] != entry["inserted_frames"]:
        raise ValueError(f"{name}: splice index has {len(index)} frames, the manifest says "
                         f"{entry['num_frames']} ({entry['inserted_frames']} inserted)")
    return index


def hold_id(name: str, frame_hold: int, extended_manifest: Path = EXTENDED_MANIFEST) -> str:
    """``<stem>@<frame_hold>`` in x0 source frames; a variant's frame is mapped back first."""
    stem, variant_s = split_clip_name(name)
    frame = int(frame_hold)
    if frame < 0:  # checked before indexing: numpy would wrap it
        raise ValueError(f"negative frame {frame_hold} for {name}")
    if variant_s:
        frame = int(source_frame_index(name, extended_manifest)[frame])
    return f"{stem}@{frame}"


def parse_hold_id(hid: str) -> tuple[str, int]:
    m = _HOLD_RE.match(hid)
    if m is None:
        raise ValueError(f"not a hold id: {hid!r}")
    return m.group(1), int(m.group(2))


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path | str) -> str:
    st = os.stat(path)
    return _sha256(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


def sha256_json(obj) -> str:
    """Content hash of a JSON-serialisable object (canonical key order)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def git_rev() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def display_path(path: Path | str) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def provenance(schema_version: int, module: str, module_file: str, inputs) -> dict:
    """``{schema_version, generator: {module, git_rev, sha256}, inputs: {path: sha256}}``."""
    return {
        "schema_version": schema_version,
        "generator": {"module": module, "git_rev": git_rev(), "sha256": sha256_file(module_file)},
        "inputs": {display_path(p): sha256_file(p) for p in inputs},
    }
