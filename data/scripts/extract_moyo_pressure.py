# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-0 archive: MOYO pressure mat -> per-clip ``.npz``, in the clip's frame.

This is the lossless artifact every downstream product regenerates from; it is
never loaded at training time.  For each ProtoMotions yoga clip that has a MOYO
capture it stores the sparse pressure field, the scalar traces, and the mat
placement **already expressed in the ProtoMotions clip frame** so that no
consumer has to redo the registration.

Registration (all links verified numerically, see ``notes/Moyo_pressure_port.MD``):

* The clip's XY is AMASS ``trans`` byte-identical; only Z is touched downstream.
* The clip frame differs from the Vicon lab frame by a pure constant translation
  ``proto_xy = vicon_xy + (0, +0.3412) m`` (``--offset-xy`` to override with the
  value refined by ``calibrate_pressure_alignment.py``).
* Pressure frame *i* == c3d frame *i* == motion frame *i*, all at 60 Hz.

Matching is exact (punctuation-normalised) and then **gated on frame counts**;
a clip whose CSV is one row longer than the motion is trimmed, anything else is
rejected rather than guessed at.  See ``moyo_pressure_io.proto_to_moyo_key``
for why fuzzy matching is unsafe.

Usage::

    python data/scripts/extract_moyo_pressure.py \
        --motion-dir data/smpl/yoga_motions_proto_yogi_grounded_yogaonly \
        --pressure-root ../../moyo_toolkit/data/pressure \
        --out-dir data/smpl/yoga_pressure
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
from moyo_pressure_io import (  # noqa: E402
    CELL_AREA_CM2,
    CELL_M,
    DEFAULT_PROTO_MINUS_VICON_XY,
    MAT_NX,
    MAT_NY,
    SUBJECT_WEIGHT_N,
    build_moyo_index,
    mat_frame_from_markers,
    moyo_paths,
    parse_pressure_csv,
    parse_pressure_xml,
    proto_to_moyo_key,
    read_marker_c3d,
)

# Cells below this carry no meaningful load (the mat's own noise floor is ~0.1).
SPARSE_EPS = 0.05  # N/cm^2


def edge_ring_mask() -> np.ndarray:
    m = np.zeros((MAT_NY, MAT_NX), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


def extract_one(motion_path: Path, stem: str, split: str, pressure_root: str,
                offset_xy: np.ndarray) -> dict:
    """Return a record dict; ``status == 'ok'`` means ``payload`` is present."""
    rec = {"clip": motion_path.stem, "moyo": stem, "split": split}

    motion = torch.load(motion_path, map_location="cpu", weights_only=False)
    n_motion = int(motion["rigid_body_pos"].shape[0])
    fps = float(motion.get("fps", 0.0))
    rec.update(n_motion=n_motion, motion_fps=fps)

    paths = moyo_paths(pressure_root, stem, split)
    field = parse_pressure_xml(paths["xml"])
    trace = parse_pressure_csv(paths["csv"])
    marker_index, points = read_marker_c3d(paths["c3d"])

    n_csv, n_c3d = int(trace.force_n.shape[0]), int(points.shape[0])
    rec.update(n_csv=n_csv, n_c3d=n_c3d, n_xml=int(field.count))

    if fps != trace.fps:
        rec["status"] = f"fps_mismatch motion={fps} pressure={trace.fps}"
        return rec

    # Frame-count gate.  A single trailing row in the CSV/c3d is a known benign
    # off-by-one in the MOYO export (9 clips); anything else is a real mismatch
    # and must not be silently truncated into alignment.
    n = n_motion
    if not (n_csv - n_motion in (0, 1) and n_c3d - n_motion in (0, 1)):
        rec["status"] = "frame_mismatch"
        return rec
    if n_csv != n_motion or n_c3d != n_motion:
        rec["trimmed_rows"] = max(n_csv, n_c3d) - n_motion

    mat = mat_frame_from_markers(marker_index, points[:n]).translated(offset_xy)
    rec["mat_marker_span"] = [round(v, 5) for v in mat.marker_span]
    rec["mat_marker_z"] = round(mat.marker_z, 5)

    dense = field.frames[trace.frame_ids[:n]]  # (n, NY, NX) N/cm^2
    force_n = dense.sum(axis=(1, 2)) * CELL_AREA_CM2
    cop_mat = trace.cop_mat[:n]
    cop_world = mat.to_world(np.nan_to_num(cop_mat))
    cop_world[~np.isfinite(cop_mat).all(axis=1)] = np.nan

    # Cross-check our field parse against the vendor's own scalar export.
    csv_force = trace.force_n[:n]
    ok = np.isfinite(csv_force) & (csv_force > 1.0)
    rec["force_vs_csv_max_rel_err"] = (
        float(np.abs(force_n[ok] - csv_force[ok]).max() / max(csv_force[ok].max(), 1e-9))
        if ok.any() else None
    )

    ring = edge_ring_mask()
    total = dense.sum(axis=(1, 2)) + 1e-12
    edge_frac = dense[:, ring].sum(axis=1) / total

    # Coverage: how much of the subject's weight the mat actually saw.  Poses
    # that hang off the 0.47 x 1.40 m mat read low; this is the validity signal
    # that stops a force reward charging the policy for unmeasured load.
    coverage = force_n / SUBJECT_WEIGHT_N

    sparse_mask = dense >= SPARSE_EPS
    idx = np.nonzero(sparse_mask)
    rec.update(
        force_med=round(float(np.median(force_n)), 2),
        force_p05=round(float(np.percentile(force_n, 5)), 2),
        force_p95=round(float(np.percentile(force_n, 95)), 2),
        coverage_med=round(float(np.median(coverage)), 4),
        coverage_frames_below_90pct=round(float((coverage < 0.90).mean()), 4),
        edge_frac_med=round(float(np.median(edge_frac)), 5),
        edge_frames_pct=round(float((edge_frac > 0.02).mean() * 100), 2),
        nan_cop_pct=round(float((~np.isfinite(cop_mat).all(axis=1)).mean() * 100), 3),
        sparse_nnz=int(idx[0].size),
        sparse_density=round(float(idx[0].size / dense.size), 5),
        status="ok",
    )

    rec["payload"] = dict(
        # sparse (COO) pressure field, N/cm^2
        p_frame=idx[0].astype(np.int32),
        p_row=idx[1].astype(np.uint8),
        p_col=idx[2].astype(np.uint8),
        p_value=dense[idx].astype(np.float32),
        n_frames=np.int64(n),
        mat_shape=np.array([MAT_NY, MAT_NX], dtype=np.int32),
        cell_size_m=np.float32(CELL_M),
        # placement, already in the ProtoMotions clip frame
        mat_origin_xy=mat.origin.astype(np.float64),
        mat_ex=mat.ex.astype(np.float64),
        mat_ey=mat.ey.astype(np.float64),
        mat_marker_z=np.float32(mat.marker_z),
        proto_minus_vicon_xy=offset_xy.astype(np.float64),
        # scalar traces
        total_force_n=force_n.astype(np.float32),
        csv_force_n=csv_force.astype(np.float32),
        cop_mat=cop_mat.astype(np.float32),
        cop_world=cop_world.astype(np.float32),
        edge_frac=edge_frac.astype(np.float32),
        coverage=coverage.astype(np.float32),
        subject_weight_n=np.float32(SUBJECT_WEIGHT_N),
        fps=np.float32(trace.fps),
        frame_ids=trace.frame_ids[:n].astype(np.int32),
        # provenance
        moyo_stem=np.str_(stem),
        moyo_split=np.str_(split),
        motion_file=np.str_(motion_path.name),
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--pressure-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--offset-xy", type=float, nargs=2, default=None,
        help="proto_xy - vicon_xy, metres (default: the measured +0.3412 m in Y)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    offset = (
        np.array(args.offset_xy, dtype=np.float64)
        if args.offset_xy is not None
        else DEFAULT_PROTO_MINUS_VICON_XY.copy()
    )
    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    index = build_moyo_index(args.pressure_root)
    motions = sorted(Path(p) for p in glob.glob(os.path.join(args.motion_dir, "*.motion")))
    print(f"{len(motions)} motions, {len(index)} MOYO captures, offset_xy={offset}")

    records = []
    for mp in motions:
        key = proto_to_moyo_key(mp.stem)
        if key is None:
            records.append({"clip": mp.stem, "status": "synthetic"})
            continue
        hit = index.get(key)
        if hit is None:
            records.append({"clip": mp.stem, "status": "no_pressure_file"})
            continue
        try:
            rec = extract_one(mp, hit[0], hit[1], args.pressure_root, offset)
        except Exception as e:  # a bad capture must not abort the corpus
            records.append({"clip": mp.stem, "status": f"error: {type(e).__name__}: {e}"})
            continue
        payload = rec.pop("payload", None)
        records.append(rec)
        if payload is not None and not args.dry_run:
            np.savez_compressed(out_dir / f"{mp.stem}.npz", **payload)
        print(
            f"{rec['status']:16s} {mp.stem[:56]:56s} "
            + (
                f"n={rec['n_motion']:5d} F={rec.get('force_med', 0):6.1f}N "
                f"cov={rec.get('coverage_med', 0):.2f} nnz={rec.get('sparse_density', 0):.4f}"
                if rec["status"] == "ok" else ""
            )
        )

    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n{len(ok)} / {len(motions)} clips archived")
    for r in records:
        if r["status"] not in ("ok", "synthetic"):
            print(f"   SKIP {r['status']:20s} {r['clip']}")
    if ok:
        err = [r["force_vs_csv_max_rel_err"] for r in ok if r["force_vs_csv_max_rel_err"]]
        print(f"   field-vs-CSV force agreement: worst {max(err) * 100:.3f} %")
        cov = np.array([r["coverage_med"] for r in ok])
        print(f"   coverage: median {np.median(cov):.3f}, {(cov < 0.9).sum()} clips below 0.90")
    if not args.dry_run:
        (out_dir / "index.json").write_text(json.dumps(records, indent=1))
        print(f"   wrote {out_dir}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
