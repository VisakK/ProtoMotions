# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 post-pass: attach the measured ground reaction to ``.motion`` files.

Non-destructive, in the same style as ``clean_yoga_motions.py`` -- reads a
converted clip plus its pressure archive and writes a new ``.motion`` into a
parallel folder.  Kinematics are copied through untouched; only three fields are
added, so the attribution algorithm can be revised and this re-run without
touching motion conversion.

Fields written (all consumed by ``MotionLib`` and surfaced on ``RobotState``):

``ground_reaction`` (T, 3)
    ``[total vertical force N, cop_x, cop_y]``.  Measured directly by the mat;
    this is the **primary** signal.  COP is in the clip's world frame.

``rigid_body_ground_forces`` (T, 24, 3)
    Per-body vertical force in newtons (x = y = 0; the mat measures normal
    pressure only).  **EXPERIMENTAL** -- it depends on the attribution in
    ``attribute_pressure_to_bodies.py``, which in turn depends on the reference
    pose being right.  Its column of ``ground_reaction_valid`` says when to
    believe it.  Note the per-body forces sum to the *explained* load, not to
    ``ground_reaction[:, 0]``: unattributed load is deliberately left out rather
    than smeared onto whichever body happened to be closest.

``ground_reaction_valid`` (T, 2)
    ``[coverage, coverage * explained]``.  Column 0 gates ``ground_reaction``,
    column 1 gates ``rigid_body_ground_forces``.

Clips with no pressure capture are **skipped, not zero-filled** -- a library
mixing measured and unmeasured clips cannot tell "no load" from "not measured",
and ``MotionLib`` drops the fields entirely if they are only partly present.

Usage::

    python data/scripts/add_pressure_to_motions.py \
        --in-dir data/smpl/yoga_motions_proto_yogi_grounded_yogaonly \
        --archive-dir data/smpl/yoga_pressure \
        --bodies-dir data/smpl/yoga_pressure_bodies \
        --out-dir data/smpl/yoga_motions_proto_yogi_pressure
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pressure_bodies import NUM_BODIES, load_archive  # noqa: E402


def build_fields(arch, bodies, n: int, with_body_forces: bool):
    force = np.nan_to_num(arch["total_force_n"][:n], nan=0.0)
    cop = arch["cop_world"][:n].copy()
    # A NaN COP means the mat saw no load at all that frame; park it at the
    # measured origin and let the validity column carry the "ignore me".
    bad = ~np.isfinite(cop).all(axis=1)
    cop[bad] = 0.0
    coverage = np.clip(np.nan_to_num(arch["coverage"][:n], nan=0.0), 0.0, 1.0)
    coverage[bad] = 0.0

    ground_reaction = np.stack([force, cop[:, 0], cop[:, 1]], axis=1).astype(np.float32)

    if with_body_forces and bodies is not None:
        bf = bodies["body_force_n"][:n]
        explained = np.clip(np.nan_to_num(bodies["explained"][:n], nan=0.0), 0.0, 1.0)
    else:
        bf = np.zeros((n, NUM_BODIES), dtype=np.float32)
        explained = np.zeros(n, dtype=np.float32)

    body_forces = np.zeros((n, NUM_BODIES, 3), dtype=np.float32)
    body_forces[:, :, 2] = bf  # normal (vertical) component only

    valid = np.stack([coverage, coverage * explained], axis=1).astype(np.float32)
    return ground_reaction, body_forces, valid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--bodies-dir", default=None,
                    help="attribution output; omit to write zero per-body forces")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-body-forces", action="store_true",
                    help="write ground_reaction only (per-body forces are experimental)")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="skip clips whose median coverage is below this "
                         "(default: keep everything, gated by the validity channel)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with_bf = not args.no_body_forces and args.bodies_dir is not None
    if with_bf:
        print("per-body ground forces: ENABLED (experimental - gated by "
              "ground_reaction_valid[:, 1])")
    else:
        print("per-body ground forces: disabled (zeros, validity column 1 = 0)")

    records = []
    written = skipped = 0
    for mp in sorted(Path(p) for p in glob.glob(os.path.join(args.in_dir, "*.motion"))):
        npz = Path(args.archive_dir) / f"{mp.stem}.npz"
        if not npz.exists():
            records.append({"clip": mp.stem, "status": "no_pressure"})
            skipped += 1
            continue
        arch = load_archive(npz)
        motion = torch.load(mp, map_location="cpu", weights_only=False)
        n = int(motion["rigid_body_pos"].shape[0])
        if n != arch["n_frames"]:
            records.append({"clip": mp.stem, "status": "frame_mismatch"})
            skipped += 1
            continue

        bodies = None
        if with_bf:
            bnpz = Path(args.bodies_dir) / f"{mp.stem}.npz"
            if not bnpz.exists():
                records.append({"clip": mp.stem, "status": "no_attribution"})
                skipped += 1
                continue
            bodies = np.load(bnpz, allow_pickle=False)

        gr, bf, valid = build_fields(arch, bodies, n, with_bf)
        cov_med = float(np.median(valid[:, 0]))
        if args.min_coverage is not None and cov_med < args.min_coverage:
            records.append({"clip": mp.stem, "status": "below_min_coverage",
                            "coverage_med": round(cov_med, 4)})
            skipped += 1
            continue

        motion["ground_reaction"] = torch.from_numpy(gr)
        motion["rigid_body_ground_forces"] = torch.from_numpy(bf)
        motion["ground_reaction_valid"] = torch.from_numpy(valid)
        torch.save(motion, out_dir / mp.name)
        written += 1
        records.append({
            "clip": mp.stem, "status": "ok", "n_frames": n,
            "coverage_med": round(cov_med, 4),
            "body_valid_med": round(float(np.median(valid[:, 1])), 4),
            "force_med": round(float(np.median(gr[:, 0])), 2),
        })
        print(f"  OK  {mp.stem[:56]:56s} n={n:5d} F={gr[:, 0].mean():6.1f}N "
              f"cov={cov_med:.3f} bodyvalid={np.median(valid[:, 1]):.3f}")

    (out_dir / "index.json").write_text(json.dumps(records, indent=1))
    print(f"\nwrote {written} motions to {out_dir}, skipped {skipped}")
    for r in records:
        if r["status"] != "ok":
            print(f"   SKIP {r['status']:20s} {r['clip']}")
    ok = [r for r in records if r["status"] == "ok"]
    if ok:
        c = np.array([r["coverage_med"] for r in ok])
        b = np.array([r["body_valid_med"] for r in ok])
        print(f"\n   coverage      median {np.median(c):.3f}  "
              f"[{c.min():.2f}, {c.max():.2f}]  {(c < 0.9).sum()} clips < 0.90")
        print(f"   body validity median {np.median(b):.3f}  "
              f"[{b.min():.2f}, {b.max():.2f}]  {(b < 0.5).sum()} clips < 0.50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
