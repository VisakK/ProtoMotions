# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split the measured mat load across the humanoid's bodies (EXPERIMENTAL).

The mat measures a scalar pressure field on the floor; the reward wants a
per-body vertical force.  Bridging the two means deciding, for every loaded
sensel, which body is standing on it.  That decision is only as good as the
reference pose, so this stage is flagged experimental and always reports how
much load it could *not* explain.

Assignment is **soft**, not nearest-wins.  Each cell's force is shared among the
near-ground collision geoms with weight

    exp(-(d_xy / sigma)^2) * exp(-(gap / sigma_z)^2)

where ``d_xy`` is the signed XY distance from the cell to the geom's footprint
(so a cell well inside a foot box gets ~1) and ``gap`` is that geom's clearance
above the floor.  A hard argmax would put a discontinuity down the middle of a
two-foot stance, where the true split is genuinely gradual; the kernel also
degrades gracefully where two bodies overlap in XY (toes under fingers in
Down-Dog).  Load further than ``--max-dist`` from every near-ground geom is left
**unassigned** and reported rather than forced onto the closest body.

The ``gap`` factor is what stops an elongated segment stealing load from the
body actually standing on the floor.  A shin capsule runs knee-to-ankle, so in
Chair Pose its projection covers the foot; without the factor it took 31 % of
the load, and the pose scored 69 % on the feet instead of 100 %.  ``sigma_z``
must not be tightened much below 2 cm: at 1 cm it also rejects genuine broad
contacts, because the reference carries a known body-wide float bias, and the
corpus' explainable load falls from 0.908 to 0.313.

Three diagnostics are written per frame, and they are the point of this stage as
much as the forces are:

``explained``     share of measured force that landed on some body
``cop_residual``  distance between the measured COP and the COP implied by the
                  attributed per-body forces -- a self-consistency check that
                  fails loudly if the kernel smears load across the mat
``n_bodies``      how many bodies carry >2 % of the load

Usage::

    python data/scripts/attribute_pressure_to_bodies.py \
        --archive-dir data/smpl/yoga_pressure \
        --motion-dir data/smpl/yoga_motions_proto_yogi_grounded_yogaonly \
        --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \
        --out-dir data/smpl/yoga_pressure_bodies
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
from moyo_pressure_io import CELL_AREA_CM2  # noqa: E402
from pressure_bodies import (  # noqa: E402
    BODY_NAMES,
    NUM_BODIES,
    _slice_geom,
    load_archive,
    world_geoms_for_clip,
    xy_distance_to_geom,
)


def attribute_clip(arch, motion, geoms, sigma: float, max_dist: float,
                   z_thresh: float, min_force: float, sigma_z: float = 0.02):
    """-> (force (T, NB) N, explained (T,), cop_residual (T,), n_bodies (T,))."""
    rbp, rbr = motion["rigid_body_pos"], motion["rigid_body_rot"]
    world = world_geoms_for_clip(geoms, rbp, rbr)
    gaps = np.stack([g[2] for g in world])  # (G, T)

    n = arch["n_frames"]
    force = np.zeros((n, NUM_BODIES), dtype=np.float32)
    explained = np.zeros(n, dtype=np.float32)
    cop_res = np.full(n, np.nan, dtype=np.float32)
    n_bodies = np.zeros(n, dtype=np.int16)

    for t in range(n):
        cells, pressure = arch["sparse"](t)
        if cells.shape[0] == 0:
            continue
        w = pressure.astype(np.float64) * CELL_AREA_CM2  # N per cell
        # Normalise against the archive's total, which is summed over the *full*
        # field including any cell below the sparse threshold. That keeps the
        # invariant sum(per-body force) == explained * total_force_n exact by
        # construction rather than relying on the two sums happening to agree.
        total = float(arch["total_force_n"][t])
        if total < min_force:
            continue

        active = np.where(gaps[:, t] <= z_thresh)[0]
        if active.size == 0:
            continue

        d = np.stack([
            xy_distance_to_geom(_slice_geom(world[gi][1], t), cells)
            for gi in active
        ])  # (A, M)
        d = np.maximum(d, 0.0)
        kernel = np.exp(-((d / sigma) ** 2))
        kernel[d > max_dist] = 0.0
        # Load flows through what is actually touching. Without this factor a
        # geom hovering several centimetres up competes on equal terms with one
        # resting on the floor whenever their footprints overlap in XY.
        gap = np.maximum(gaps[active, t], 0.0)
        kernel *= np.exp(-((gap / sigma_z) ** 2))[:, None]
        norm = kernel.sum(axis=0)  # (M,)

        assigned = norm > 1e-9
        share = np.zeros_like(kernel)
        share[:, assigned] = kernel[:, assigned] / norm[assigned]
        contrib = share * w[None, :]  # (A, M) newtons

        per_geom = contrib.sum(axis=1)
        for k, gi in enumerate(active):
            force[t, world[gi][0]] += per_geom[k]

        explained[t] = w[assigned].sum() / total
        n_bodies[t] = int((force[t] > 0.02 * total).sum())

        # Self-consistency: COP implied by the attribution vs the measured COP.
        if assigned.any():
            implied = (cells[assigned] * w[assigned, None]).sum(0) / w[assigned].sum()
            meas = arch["cop_world"][t]
            if np.isfinite(meas).all():
                cop_res[t] = float(np.linalg.norm(implied - meas))

    return force, explained, cop_res, n_bodies


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sigma", type=float, default=0.02,
                    help="soft-assignment length scale on XY distance, m")
    ap.add_argument("--max-dist", type=float, default=0.06,
                    help="beyond this a cell is left unassigned, m")
    ap.add_argument("--z-thresh", type=float, default=0.06,
                    help="a geom counts as near-ground below this gap, m")
    ap.add_argument("--sigma-z", type=float, default=0.02,
                    help="how fast a geom's claim decays with its ground gap, m")
    ap.add_argument("--min-force", type=float, default=20.0)
    args = ap.parse_args()

    from contact_geometry import parse_typed_geoms

    geoms = parse_typed_geoms(args.mjcf, BODY_NAMES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(Path(p) for p in glob.glob(os.path.join(args.archive_dir, "*.npz")))
    print(f"{len(archives)} clips, sigma={args.sigma} sigma_z={args.sigma_z} "
          f"max_dist={args.max_dist} z_thresh={args.z_thresh}")

    records = []
    for ap_ in archives:
        arch = load_archive(ap_)
        motion = torch.load(Path(args.motion_dir) / f"{ap_.stem}.motion",
                            map_location="cpu", weights_only=False)
        if motion["rigid_body_pos"].shape[0] != arch["n_frames"]:
            print(f"SKIP {ap_.stem}: frame mismatch")
            continue
        force, explained, cop_res, n_bodies = attribute_clip(
            arch, motion, geoms, args.sigma, args.max_dist, args.z_thresh,
            args.min_force, args.sigma_z
        )
        loaded = arch["total_force_n"] >= args.min_force
        rec = dict(
            clip=ap_.stem,
            n_frames=int(arch["n_frames"]),
            explained_med=round(float(np.median(explained[loaded])), 4) if loaded.any() else None,
            explained_p10=round(float(np.percentile(explained[loaded], 10)), 4) if loaded.any() else None,
            cop_residual_med_cm=round(float(np.nanmedian(cop_res) * 100), 2),
            n_bodies_med=int(np.median(n_bodies[loaded])) if loaded.any() else 0,
            top_bodies=sorted(
                {BODY_NAMES[i]: round(float(force[:, i].sum() / max(force.sum(), 1e-9)) * 100, 1)
                 for i in range(NUM_BODIES) if force[:, i].sum() > 0}.items(),
                key=lambda kv: -kv[1],
            )[:5],
        )
        records.append(rec)
        np.savez_compressed(
            out_dir / f"{ap_.stem}.npz",
            body_force_n=force,
            explained=explained,
            cop_residual_m=cop_res,
            n_bodies=n_bodies,
            body_names=np.array(BODY_NAMES),
            sigma=np.float32(args.sigma),
            sigma_z=np.float32(args.sigma_z),
            max_dist=np.float32(args.max_dist),
            z_thresh=np.float32(args.z_thresh),
        )
        print(f"  {ap_.stem[:52]:52s} expl={rec['explained_med']:.3f} "
              f"copres={rec['cop_residual_med_cm']:5.2f}cm nb={rec['n_bodies_med']} "
              f"{rec['top_bodies'][:3]}")

    (out_dir / "index.json").write_text(json.dumps(records, indent=1))
    e = np.array([r["explained_med"] for r in records if r["explained_med"] is not None])
    c = np.array([r["cop_residual_med_cm"] for r in records])
    print(f"\n{len(records)} clips attributed")
    print(f"   explained load: median {np.median(e):.3f}, p10 {np.percentile(e, 10):.3f}, "
          f"{int((e < 0.5).sum())} clips below 0.50")
    print(f"   COP self-consistency residual: median {np.nanmedian(c):.2f} cm, "
          f"p90 {np.nanpercentile(c, 90):.2f} cm")
    print(f"   wrote {out_dir}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
