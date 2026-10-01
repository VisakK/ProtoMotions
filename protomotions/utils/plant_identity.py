# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which simulated body ("plant") a piece of data was built for, and a loud error when it meets another.

The yoga pipeline has two plants (``PLANTS``): **v1**, the shipped ``smpl_yogi03596_lowtorque`` (the SMPL mean
shape with rescaled limbs, robot ``smpl_yogi``), and **v2**, the MOYO performer's own skeleton and masses
(``smpl_yogi03596_v2``, robot ``smpl_yogi_v2``; BodyFix Steps 1-2). Everything computed on a plant's geometry is
wrong on the other one and nothing noticed: MotionLib serves the stored ``rigid_body_pos`` rather than FK, so a
v1 clip starts 5-6.6 cm inside the floor on v2; physics tables, capture and statics records carry body masses,
colliders and heights of the plant they were measured on.

So data carries the sha256 of the MJCF it was built on (``identity``), and every consumer compares it with the
plant it runs (``require``). A plant is identified by its MJCF's bytes: rebuilding the XML is a new plant.

* The offline scripts pick the plant with ``REFERENCE_PLANT`` (``v1``, ``v2`` or a path to an MJCF); the default
  stays ``v1`` so every finished record (built on v1) keeps meaning what it meant. ``--mjcf`` flags default to it.
* The training side takes the plant from the robot config's ``asset_file_name``.
* Legacy data (no identity) predates plant v2 and was built on v1: accepted with v1, refused with v2. For an MJCF
  outside the registry (another robot) no plant rule applies.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from typing import Mapping, Optional, Union

REPO = Path(__file__).resolve().parents[2]
ENV_VAR = "REFERENCE_PLANT"
DEFAULT = "v1"
LEGACY = "v1"            # the plant every record without an identity was built on
PLANTS = {
    "v1": ("data/assets/smpl/smpl_yogi03596_lowtorque.xml", "data/assets/smpl/smpl_yogi03596_lowtorque_flat.xml"),
    "v2": ("data/assets/smpl/smpl_yogi03596_v2.xml", "data/assets/smpl/smpl_yogi03596_v2_flat.xml"),
}
KEY = "plant_sha256"     # the field name data carries (a .motion dict, a motion library, physics tables)

PathLike = Union[str, os.PathLike]


class PlantMismatchError(RuntimeError):
    """Data built for one plant was about to be used with another."""


def _abs(path: PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def selected() -> str:
    """The plant the offline scripts use: ``REFERENCE_PLANT`` (a name in ``PLANTS`` or an MJCF path)."""
    value = os.environ.get(ENV_VAR, DEFAULT).strip() or DEFAULT
    if value not in PLANTS and not _abs(value).is_file():
        raise ValueError(f"{ENV_VAR}={value!r} is neither a plant ({sorted(PLANTS)}) nor an MJCF file")
    return value


def mjcf_path(plant: Optional[str] = None) -> Path:
    """The MJCF of ``plant`` (default: ``selected()``), absolute."""
    plant = selected() if plant is None else plant
    return _abs(PLANTS[plant][0]) if plant in PLANTS else _abs(plant)


def default_mjcf(plant: Optional[str] = None) -> str:
    """``mjcf_path`` as a repo-relative string: the ``--mjcf`` default of the offline scripts."""
    return _rel(mjcf_path(plant))


def flat_path(plant: Optional[str] = None) -> Path:
    """The flattened twin (``_flat.xml``) of ``plant``'s MJCF: the one MuJoCo-side tools (statics) load."""
    plant = selected() if plant is None else plant
    if plant in PLANTS:
        return _abs(PLANTS[plant][1])
    p = _abs(plant)
    return p.with_name(p.stem + "_flat.xml")


@functools.lru_cache(maxsize=64)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256(plant_or_path: PathLike) -> str:
    """The sha256 of an MJCF's bytes (a plant name or an MJCF path): the plant's identity."""
    p = mjcf_path(str(plant_or_path)) if str(plant_or_path) in PLANTS else _abs(plant_or_path)
    st = p.stat()
    return _sha256(str(p), st.st_size, st.st_mtime_ns)


def name_of(sha: Optional[str]) -> Optional[str]:
    """The registered plant with this MJCF sha256, or None."""
    if not sha:
        return None
    for name, (xml, _) in PLANTS.items():
        p = _abs(xml)
        if p.is_file() and sha256(p) == sha:
            return name
    return None


def identity(plant_or_path: Optional[PathLike] = None) -> dict:
    """``{"plant": name or None, "mjcf": repo-relative path, "plant_sha256": sha}`` of a plant name or an MJCF."""
    if plant_or_path is None or str(plant_or_path) in PLANTS:
        p = mjcf_path(None if plant_or_path is None else str(plant_or_path))
    else:
        p = _abs(plant_or_path)
    sha = sha256(p)
    try:
        rel = str(p.resolve().relative_to(REPO))
    except ValueError:
        rel = str(p)
    return {"plant": name_of(sha), "mjcf": rel, KEY: sha}


def robot_mjcf(robot_config) -> Optional[str]:
    """The MJCF a robot config simulates (``asset.asset_root/asset.asset_file_name``); None when it names none
    (a test stub), in which case its consumers skip the plant check."""
    asset = getattr(robot_config, "asset", None)
    name = getattr(asset, "asset_file_name", None)
    return os.path.join(asset.asset_root, name) if name else None


def recorded_sha(data: Union[Mapping, None]) -> Optional[str]:
    """The plant sha256 a piece of data records: a ``plant_sha256`` field, a nested ``plant`` identity, or
    (curation records) the provenance ``inputs`` entry of a registered plant's MJCF. None: no identity."""
    if not data:
        return None
    if data.get(KEY):
        return str(data[KEY])
    plant = data.get("plant")
    if isinstance(plant, Mapping) and plant.get(KEY):
        return str(plant[KEY])
    inputs = data.get("inputs")
    if isinstance(inputs, Mapping):
        for xml, _ in PLANTS.values():
            if xml in inputs:
                return str(inputs[xml])
    return None


def require(recorded: Optional[str], mjcf: PathLike, what: str) -> Optional[str]:
    """Raise ``PlantMismatchError`` unless data recording plant sha ``recorded`` (None: legacy) may be used
    with the plant whose MJCF is ``mjcf``. Returns the plant name in use (None when unregistered)."""
    actual = sha256(mjcf)
    actual_name = name_of(actual)
    if recorded is None:
        if actual_name is None or actual_name == LEGACY:
            return actual_name
        raise PlantMismatchError(
            f"{what} carries no plant identity, so it predates plant {actual_name} and was built on plant "
            f"{LEGACY} ({PLANTS[LEGACY][0]}); the plant in use is {actual_name} ({_rel(mjcf)}). Rebuild it on "
            f"{actual_name} (BodyFix Steps 3-5) or run with plant {LEGACY}."
        )
    if recorded != actual:
        was = name_of(recorded)
        raise PlantMismatchError(
            f"{what} was built for plant {was or 'with MJCF sha256 ' + recorded[:12]} but the plant in use is "
            f"{actual_name or ''} ({_rel(mjcf)}, sha256 {actual[:12]}). Rebuild it on this plant, or select its "
            f"plant ({ENV_VAR} / the robot config)."
        )
    return actual_name


def _rel(path: PathLike) -> str:
    p = mjcf_path(str(path)) if str(path) in PLANTS else _abs(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)
