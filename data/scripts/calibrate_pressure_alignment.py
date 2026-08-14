# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration gate: refit the clip-frame <-> Vicon-frame XY offset corpus-wide.

The extractor bakes in ``proto_xy = vicon_xy + (0, +0.3412) m``, estimated from
Vicon marker positions against FK'd joint centres.  That estimator is only as
unbiased as the marker-to-joint-centre correspondence, so this script re-derives
the offset from the quantity we actually care about -- how well the measured
pressure footprint sits on the reference's own ground-contact geometry -- and
reports whether the two independent estimates agree.

Two estimators
--------------
``marker``
    Median of ``FK_body - marker`` over the three medial/lateral marker pairs
    whose midpoint really is a joint centre (knee LKNE/LKNI, elbow LELB/LIEL,
    wrist LIWR/LOWR), both sides.  Deliberately excludes single-sided markers
    such as LANK -- the lateral malleolus sits ~3 cm off the ankle joint centre
    and biases the estimate.

``pressure``
    Grid search (coarse then fine) over the residual translation that maximises
    the **explained load fraction**: the share of measured force falling within
    ``--tol`` of a collision geom that the reference has near the ground.  The
    reference geometry does not move with the search parameter, so this is sharp
    rather than a plateau, and it needs no marker correspondence at all.

Usage::

    python data/scripts/calibrate_pressure_alignment.py \
        --archive-dir data/smpl/yoga_pressure \
        --motion-dir data/smpl/yoga_motions_proto_yogi_grounded_yogaonly \
        --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \
        --pressure-root ../../moyo_toolkit/data/pressure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moyo_pressure_io import (  # noqa: E402
    CELL_AREA_CM2,
    build_moyo_index,
    moyo_paths,
    proto_to_moyo_key,
    read_marker_c3d,
)
from pressure_bodies import (  # noqa: E402
    BODY_NAMES,
    load_archive,
    near_ground_geoms,
    xy_distance_to_geom,
)

# Medial/lateral marker pairs whose midpoint is a genuine joint centre.
JOINT_PAIRS = [
    (("_03596:LKNE", "_03596:LKNI"), "L_Knee"),
    (("_03596:RKNE", "_03596:RKNI"), "R_Knee"),
    (("_03596:LELB", "_03596:LIEL"), "L_Elbow"),
    (("_03596:RELB", "_03596:RIEL"), "R_Elbow"),
    (("_03596:LIWR", "_03596:LOWR"), "L_Wrist"),
    (("_03596:RIWR", "_03596:ROWR"), "R_Wrist"),
]


def marker_offset(motion_path: Path, c3d_path: str) -> np.ndarray | None:
    motion = torch.load(motion_path, map_location="cpu", weights_only=False)
    rbp = motion["rigid_body_pos"].numpy()
    idx, pts = read_marker_c3d(c3d_path)
    n = min(rbp.shape[0], pts.shape[0])
    deltas = []
    for (m1, m2), body in JOINT_PAIRS:
        if m1 in idx and m2 in idx:
            centre = (pts[:n, idx[m1]] + pts[:n, idx[m2]]) / 2.0
            deltas.append(rbp[:n, BODY_NAMES.index(body)] - centre)
    if not deltas:
        return None
    return np.nanmedian(np.concatenate(deltas, axis=0), axis=0)[:2]


def frame_samples(arch, motion, geoms, n_frames, min_coverage, min_force):
    """Yield (cell_xy (M,2), weight (M,), geom list) for well-measured frames."""
    rbp, rbr = motion["rigid_body_pos"], motion["rigid_body_rot"]
    ok = np.where(
        (arch["coverage"] >= min_coverage) & (arch["total_force_n"] >= min_force)
    )[0]
    if ok.size == 0:
        return
    for t in ok[:: max(1, ok.size // n_frames)][:n_frames]:
        dense = arch["dense"](int(t))
        mask = dense > 0.2
        if mask.sum() < 5:
            continue
        w = dense[mask] * CELL_AREA_CM2
        cells = arch["cell_centres"][mask]
        gs = near_ground_geoms(geoms, rbp, rbr, int(t), z_thresh=0.06)
        if not gs:
            continue
        yield cells, w, gs


def explained_fraction(cells, w, gs, deltas, tol):
    """(D,) share of load within ``tol`` of a near-ground geom, per candidate delta."""
    # shifting the mat by +delta == shifting the query points by -delta
    pts = cells[None, :, :] - deltas[:, None, :]  # (D, M, 2)
    flat = pts.reshape(-1, 2)
    best = np.full(flat.shape[0], np.inf)
    for _, g in gs:
        np.minimum(best, xy_distance_to_geom(g, flat), out=best)
    inside = (best.reshape(len(deltas), -1) <= tol).astype(np.float64)
    return inside @ w / max(w.sum(), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--pressure-root", required=True)
    ap.add_argument("--frames-per-clip", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--min-force", type=float, default=200.0)
    ap.add_argument("--tol", type=float, default=0.02, help="explained-load radius, m")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    args = ap.parse_args()

    from contact_geometry import parse_typed_geoms

    geoms = parse_typed_geoms(args.mjcf, BODY_NAMES)
    index = build_moyo_index(args.pressure_root)
    archives = sorted(Path(args.archive_dir).glob("*.npz"))
    print(f"{len(archives)} archived clips")

    # ---- estimator 1: markers ------------------------------------------------
    marker_est = []
    for ap_ in archives:
        key = proto_to_moyo_key(ap_.stem)
        hit = index.get(key) if key else None
        if hit is None:
            continue
        d = marker_offset(Path(args.motion_dir) / f"{ap_.stem}.motion",
                          moyo_paths(args.pressure_root, hit[0], hit[1])["c3d"])
        if d is not None:
            marker_est.append(d)
    M = np.asarray(marker_est)
    print(f"\nmarker estimator ({len(M)} clips, ML-pair joint centres only):")
    print(f"   median {np.round(np.median(M, 0), 4)}   "
          f"sd-across-clips {np.round(M.std(0), 4)}   sem {np.round(M.std(0) / np.sqrt(len(M)), 4)}")

    # ---- estimator 2: pressure footprint ------------------------------------
    samples = []
    for ap_ in archives:
        arch = load_archive(ap_)
        motion = torch.load(Path(args.motion_dir) / f"{ap_.stem}.motion",
                            map_location="cpu", weights_only=False)
        samples.extend(
            frame_samples(arch, motion, geoms, args.frames_per_clip,
                          args.min_coverage, args.min_force)
        )
    print(f"\npressure estimator: {len(samples)} sampled frames "
          f"(coverage >= {args.min_coverage}, force >= {args.min_force} N)")

    def search(centre, half, step):
        g = np.arange(-half, half + 1e-9, step)
        deltas = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) + centre
        score = np.zeros(len(deltas))
        for cells, w, gs in samples:
            score += explained_fraction(cells, w, gs, deltas, args.tol)
        score /= max(len(samples), 1)
        best = int(score.argmax())
        return deltas[best], score[best], deltas, score

    base = np.zeros(2)
    d1, s1, _, _ = search(base, 0.06, 0.01)
    print(f"   coarse (+-6 cm, 1 cm): residual {np.round(d1, 4)}  explained {s1:.4f}")
    d2, s2, deltas, score = search(d1, 0.012, 0.002)
    print(f"   fine   (+-1.2 cm, 2 mm): residual {np.round(d2, 4)}  explained {s2:.4f}")
    s0 = float(np.mean([explained_fraction(c, w, g, base[None], args.tol)[0]
                        for c, w, g in samples]))
    print(f"   explained load at the shipped offset: {s0:.4f}  ->  gain {s2 - s0:+.4f}")

    baked = load_archive(archives[0])["proto_minus_vicon_xy"]
    total = baked + d2
    print(f"\nshipped offset {np.round(baked, 4)}  +  residual {np.round(d2, 4)}"
          f"  =  refined {np.round(total, 4)}")
    print(f"independent marker estimate            =           {np.round(np.median(M, 0), 4)}")
    print(f"disagreement                           =           "
          f"{np.round(total - np.median(M, 0), 4)} m")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "marker_median": np.median(M, 0).tolist(),
            "marker_sd": M.std(0).tolist(),
            "marker_n_clips": len(M),
            "pressure_residual": d2.tolist(),
            "pressure_explained_at_residual": float(s2),
            "pressure_explained_at_shipped": s0,
            "shipped_offset": baked.tolist(),
            "refined_offset": total.tolist(),
            "n_sampled_frames": len(samples),
            "tol_m": args.tol,
        }, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
