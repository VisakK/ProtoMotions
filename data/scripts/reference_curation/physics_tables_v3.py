# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics tables v2 on release v3 (card R3 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``): v2's builder, run
unchanged, with the synthetic clips sent down its no-pressure path.

A synthetic clip (an extended-manifest entry with a ``synthetic`` block, stem ``SYN_*``) carries the three measured
channels as zeros with validity 0 on every frame: MotionLib packs the pressure channel all-or-nothing, so the clips
must carry it. ``build_physics_tables_v2.main`` treats any motion with ``ground_reaction`` as measured and opens the
mat archive of its ``source_stem`` (``:224-226``), which a synthetic clip has none of, then gates "unloaded" with
``on_mat``. This wrapper runs that ``main`` in-process with one change: the builder's view of ``torch`` loads a
synthetic clip's ``.motion`` without the three channels, so ``pressure_fields`` returns None for it and the clip
takes the path every pressure-free motion takes -- swing labels from the velocity rule only, no COP or zone-share
signature, no archive. Human motions are loaded untouched. ``build_physics_tables_v2.py`` is not edited (its sha256 is
part of release v2's id).

The arguments are ``build_physics_tables_v2.py``'s; the wrapper refuses a library whose synthetic stems it cannot
account for (every synthetic stem must be stripped exactly once, nothing else may be).

    PYTHONPATH=.:data/scripts python -m reference_curation.physics_tables_v3 --extended-manifest ... (v2's flags)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

PRESSURE_FIELDS = ("ground_reaction", "rigid_body_ground_forces", "ground_reaction_valid")


def synthetic_stems(manifest: dict) -> list[str]:
    """The extended manifest's synthetic motions: entries with a ``synthetic`` block. Each must be named ``SYN_*``
    and no human motion may be."""
    out, bad = [], []
    for c in manifest["clips"]:
        syn = c.get("synthetic") is not None
        if syn != c["stem"].startswith("SYN_"):
            bad.append(c["stem"])
        if syn:
            out.append(c["stem"])
    if bad:
        raise ValueError(f"synthetic block and SYN_ prefix disagree on {bad[:3]}")
    return out


class NoPressureTorch:
    """``torch`` as ``build_physics_tables_v2`` sees it, except that ``load`` of one of ``paths`` returns the motion
    without its measured channels. Everything else is the real module."""

    def __init__(self, real, paths):
        self._real = real
        self._paths = {str(Path(p).resolve()): Path(p).name[: -len(".motion")] for p in paths}
        self.stripped: list[str] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def load(self, f, *args, **kwargs):
        obj = self._real.load(f, *args, **kwargs)
        if isinstance(f, (str, os.PathLike)):
            stem = self._paths.get(str(Path(f).resolve()))
            if stem is not None:
                obj = {k: v for k, v in obj.items() if k not in PRESSURE_FIELDS}
                self.stripped.append(stem)
        return obj


def main(argv: list[str] | None = None) -> int:
    import build_physics_tables_v2 as tables

    argv = list(sys.argv[1:] if argv is None else argv)

    def flag(name: str) -> str:
        if name not in argv or argv.index(name) + 1 >= len(argv):
            raise ValueError(f"physics_tables_v3 needs {name}")
        return argv[argv.index(name) + 1]

    manifest = yaml.safe_load(open(flag("--extended-manifest")))
    stems = synthetic_stems(manifest)
    proxy = NoPressureTorch(tables.torch, [Path(flag("--motion-dir")) / f"{s}.motion" for s in stems])
    real, saved_argv = tables.torch, sys.argv
    tables.torch = proxy
    sys.argv = ["build_physics_tables_v2.py", *argv]
    try:
        code = tables.main()
    finally:
        tables.torch, sys.argv = real, saved_argv
    if sorted(proxy.stripped) != sorted(stems):
        raise RuntimeError(f"the no-pressure path took {len(proxy.stripped)} loads for {len(stems)} synthetic motions")
    print(f"physics_tables_v3: {len(stems)} synthetic motions took the no-pressure path (velocity swing rule only)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
