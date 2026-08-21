# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add a third validity column: 'every body that could be loaded is on the mat'.

Why
---
``ground_reaction_valid`` ships two columns — ``coverage`` and
``coverage x explained``.  Coverage is ``measured_total / body_weight``, and it
conflates two failures with **opposite** consequences for a load-*share* target:

* a **uniform gain deficit** — the mat under-reads but sees every contact.
  Shares are unaffected; the frame is fully usable.
* **load off the mat** — a limb is outside the 0.47 x 1.40 m sensing area.
  That limb's load is missing, so every share is wrong.

Coverage cannot tell them apart, and gating on it throws away the first case.
Measured on the 29 inversion / arm-balance clips, that costs more than half the
supervised hold frames: all four Pincha takes read 0.74-0.77 BW during the hold
with **2.85 % of frames touching the outer sensel ring**, i.e. a gain deficit
with everything on the mat, and a ``coverage >= 0.90`` gate rejects 100 % of
them.

The Tier-0 archive stores the mat rectangle *already expressed in the clip
frame* (``mat_origin_xy`` / ``mat_ex`` / ``mat_ey`` / ``mat_shape`` /
``cell_size_m``), so the second case is decidable geometrically: is any body
whose lowest collision geom is within ``--near-ground`` of the floor outside the
sensing area?  The threshold is deliberately generous (15 cm default) because
the reference carries the body-wide SMPL float bias documented in
``notes/Contact_config_def.MD`` 4 — a genuinely loaded foot can hover 12 cm.

Effect, pooled over the 29 clips (usable *hold* frames):

    coverage >= 0.90 and explained >= 0.90     25 %
    on_mat and explained >= 0.90               48 %

and it still correctly rejects the poses that really do overrun the mat
(Plow, Shoulderstand, Legs-Up-the-Wall, Peacock all drop toward 0).

Output
------
Rewrites each ``.motion`` with ``ground_reaction_valid`` widened from (T, 2) to
(T, 3):

    [0] coverage                     gates ground_reaction        (unchanged)
    [1] coverage x explained         gates rigid_body_ground_forces (unchanged)
    [2] on_mat x explained           gates per-body SHARES          (new)

Column 2 deliberately drops coverage and keeps ``explained``: a share target
does not care how much total load was measured, only that what was measured was
placed on the right bodies and that nothing was missed off the edge.

Usage::

    PYTHONPATH=. python data/scripts/add_onmat_gate_to_motions.py \
      --in-dir data/smpl/yoga_motions_proto_yogi_pressure \
      --archive-dir data/smpl/yoga_pressure \
      --out-dir data/smpl/yoga_motions_proto_yogi_pressure_gated
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
from contact_geometry import geom_ground_distance, geom_to_world, parse_typed_geoms  # noqa: E402

BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle",
    "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder",
    "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Hand",
]


def lowest_geom_per_body(geoms, body_pos: torch.Tensor, body_rot: torch.Tensor):
    """(gap [T,B], xy [T,B,2]) of each body's lowest collision geom.

    Bodies with no collision geom fall back to their origin and an infinite gap,
    so they never trigger the off-mat test.
    """
    T, B = body_pos.shape[0], body_pos.shape[1]
    gap = torch.full((T, B), float("inf"), dtype=torch.float64)
    xy = body_pos[:, :, :2].clone().to(torch.float64)
    for bi, name in enumerate(BODY_NAMES):
        for g in geoms[name]:
            gw = geom_to_world(g, body_pos[:, bi], body_rot[:, bi])
            g_gap, witness = geom_ground_distance(gw)
            g_gap = g_gap.to(torch.float64)
            upd = g_gap < gap[:, bi]
            gap[upd, bi] = g_gap[upd]
            xy[upd, bi] = witness[upd][:, :2].to(torch.float64)
    return gap.numpy(), xy.numpy()


def on_mat_mask(archive, gap: np.ndarray, xy: np.ndarray,
                near_ground_m: float, margin_m: float) -> np.ndarray:
    """[T] True when no body within ``near_ground_m`` of the floor is off the mat."""
    origin = np.asarray(archive["mat_origin_xy"], np.float64)
    ex = np.asarray(archive["mat_ex"], np.float64)
    ey = np.asarray(archive["mat_ey"], np.float64)
    n_rows, n_cols = [int(v) for v in archive["mat_shape"]]
    cell = float(archive["cell_size_m"])
    # mat_ex runs along columns, mat_ey along rows (moyo_pressure_io convention)
    len_x, len_y = n_cols * cell, n_rows * cell

    rel = xy - origin[None, None, :]
    u = rel[..., 0] * ex[0] + rel[..., 1] * ex[1]
    v = rel[..., 0] * ey[0] + rel[..., 1] * ey[1]
    inside = (
        (u >= -margin_m) & (u <= len_x + margin_m)
        & (v >= -margin_m) & (v <= len_y + margin_m)
    )
    could_load = gap < near_ground_m
    return ~np.any(could_load & ~inside, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="data/smpl/yoga_motions_proto_yogi_pressure")
    ap.add_argument("--archive-dir", default="data/smpl/yoga_pressure")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mjcf", default="data/assets/smpl/smpl_yogi03596_lowtorque.xml")
    ap.add_argument("--near-ground", type=float, default=0.15,
                    help="A body this close to the floor could be carrying load.")
    ap.add_argument("--mat-margin", type=float, default=0.02,
                    help="Slack on the sensing rectangle, metres.")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--clips-yaml", default=None)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geoms = parse_typed_geoms(args.mjcf, BODY_NAMES)

    if args.clips_yaml:
        import yaml
        names = [Path(e["file"]).stem
                 for e in yaml.safe_load(Path(args.clips_yaml).read_text())["motions"]]
    elif args.clips:
        names = list(args.clips)
    else:
        names = sorted(p.stem for p in in_dir.glob("*.motion"))

    index, skipped = [], []
    print(f"{'clip':56s} {'T':>6s} {'onmat%':>7s} {'colA%':>6s} {'colC%':>6s}")
    print("-" * 86)
    for name in names:
        src = in_dir / f"{name}.motion"
        arch = Path(args.archive_dir) / f"{name}.npz"
        if not src.is_file():
            skipped.append((name, "no .motion")); continue
        d = torch.load(src, map_location="cpu", weights_only=False)
        if d.get("ground_reaction_valid") is None:
            skipped.append((name, "no pressure")); continue
        if not arch.is_file():
            skipped.append((name, "no Tier-0 archive")); continue

        gv = d["ground_reaction_valid"].to(torch.float64)
        if gv.shape[1] >= 3:
            skipped.append((name, "already has 3 columns")); continue
        a = np.load(arch, allow_pickle=True)
        gap, xy = lowest_geom_per_body(
            geoms, d["rigid_body_pos"].to(torch.float64),
            d["rigid_body_rot"].to(torch.float64),
        )
        on_mat = on_mat_mask(a, gap, xy, args.near_ground, args.mat_margin)
        cov = gv[:, 0].numpy()
        explained = gv[:, 1].numpy() / np.maximum(cov, 1e-6)
        col2 = np.clip(on_mat.astype(np.float64) * explained, 0.0, 1.0)

        d["ground_reaction_valid"] = torch.cat(
            [gv.to(torch.float32),
             torch.from_numpy(col2).to(torch.float32).unsqueeze(1)], dim=1
        )
        torch.save(d, out_dir / f"{name}.motion")

        a_pass = float(((cov >= 0.90) & (explained >= 0.90)).mean())
        c_pass = float((col2 >= 0.90).mean())
        index.append(dict(clip=name, T=int(len(cov)), on_mat_frac=float(on_mat.mean()),
                          gate_cov_expl=a_pass, gate_onmat_expl=c_pass))
        print(f"{name[:56]:56s} {len(cov):6d} {100*on_mat.mean():7.1f} "
              f"{100*a_pass:6.1f} {100*c_pass:6.1f}")

    if index:
        # Merge, do not clobber. Run on a subset -- which is the normal case when
        # a new pose group is added -- a plain write would replace the record of
        # every clip already in this directory with a record of just these ones,
        # while leaving their .motion files in place. The index would then
        # silently under-report the directory it describes. Refuse to merge
        # across different geometry thresholds, since those numbers would not
        # mean the same thing.
        index_path = out_dir / "index.json"
        payload = {"near_ground_m": args.near_ground,
                   "mat_margin_m": args.mat_margin,
                   "source": str(in_dir), "clips": index}
        if index_path.is_file():
            prev = json.loads(index_path.read_text())
            same = (prev.get("near_ground_m") == args.near_ground
                    and prev.get("mat_margin_m") == args.mat_margin)
            if same:
                by = {c["clip"]: c for c in prev.get("clips", [])}
                by.update({c["clip"]: c for c in index})
                payload["clips"] = sorted(by.values(), key=lambda c: c["clip"])
                kept = len(payload["clips"]) - len(index)
                if kept:
                    print(f"merged with the existing index ({kept} clips kept)")
            else:
                backup = index_path.with_suffix(".json.superseded")
                index_path.replace(backup)
                print(f"WARNING existing index used near_ground="
                      f"{prev.get('near_ground_m')} / mat_margin="
                      f"{prev.get('mat_margin_m')}, this pass used "
                      f"{args.near_ground} / {args.mat_margin}; not merging. "
                      f"Old index moved to {backup.name} -- the .motion files in "
                      f"this directory are now a MIXTURE of two thresholds.")
        index_path.write_text(json.dumps(payload, indent=1))
        a = np.mean([r["gate_cov_expl"] for r in index])
        c = np.mean([r["gate_onmat_expl"] for r in index])
        print(f"\n{len(index)} clips rewritten to {out_dir}")
        print(f"pooled usable frames: coverage gate {100*a:.0f}% -> on-mat gate {100*c:.0f}%")
    for name, why in skipped:
        print(f"SKIPPED {name}: {why}")


if __name__ == "__main__":
    main()
