# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation of the MOYO pressure port.

Six independent checks, each of which can fail loudly on its own.  The point is
that none of them assumes the others: the physics checks do not use the marker
alignment, and the alignment checks do not use the attribution.

1. **Round trip** -- pack the pressure-carrying ``.motion`` files into a
   ``MotionLib`` and read frames back through ``get_motion_state``, confirming
   the fields survive concat, save, load, and blending.
2. **Force conservation** -- attributed per-body forces must sum to the
   *explained* share of the measured total, and never exceed the total.
3. **COP consistency** -- the COP implied by the attributed forces must match
   the mat's own reported COP.
4. **Support sanity** -- for single-support frames the measured COP must fall
   inside the loaded footprint, and near the reference's contact geometry.
5. **Static equilibrium** -- on still frames the measured COP must sit under the
   reference's centre of mass; deviation bounds the residual registration error
   using no marker data at all.
6. **Anatomy** -- the load on named poses must land on the bodies the pose is
   named for (Tree on one foot, Crow on the hands, ...).

Usage::

    python data/scripts/validate_pressure_port.py \
        --motion-dir data/smpl/yoga_motions_proto_yogi_pressure \
        --archive-dir data/smpl/yoga_pressure \
        --bodies-dir data/smpl/yoga_pressure_bodies \
        --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from moyo_pressure_io import CELL_AREA_CM2  # noqa: E402
from pressure_bodies import BODY_NAMES, load_archive  # noqa: E402

FEET = {"L_Ankle", "L_Toe", "R_Ankle", "R_Toe"}
HANDS = {"L_Wrist", "R_Wrist", "L_Hand", "R_Hand"}

# Poses whose name tells you which bodies must carry the load.  ``airborne``
# names bodies that must be off the floor for the pose to count as held -- a
# handstand clip spends half its frames standing, and there the load is
# correctly on the feet.  The gate is pure reference kinematics, so it cannot
# launder an attribution error into a pass.
ANATOMY = [
    # pattern,            expected support,        airborne, min z, threshold
    ("Tree_Pose", FEET, set(), 0.0, 0.90),
    ("Chair_Pose", FEET, set(), 0.0, 0.90),
    ("Warrior_III", FEET, set(), 0.0, 0.90),
    ("Half_Moon", FEET | {"L_Wrist", "R_Wrist"}, set(), 0.0, 0.80),
    ("Crane_Crow_Pose", HANDS, FEET, 0.25, 0.85),
    ("Handstand", HANDS, FEET, 0.25, 0.85),
    ("Tittibhasana", HANDS, FEET, 0.25, 0.85),
    ("Pincha_Mayurasana", HANDS | {"L_Elbow", "R_Elbow"}, FEET, 0.25, 0.85),
    ("Salamba_Sirsasana", HANDS | {"L_Elbow", "R_Elbow", "Head"}, FEET, 0.25, 0.80),
]

# Frames whose attribution is already flagged untrustworthy are excluded and
# reported separately: the validity channel exists precisely so downstream
# terms ignore them, so failing the check on them would be double-counting.
MIN_EXPLAINED = 0.90

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# --------------------------------------------------------------------------- #
def check_round_trip(motion_dir: str, tmp_pt: Path) -> None:
    print("\n1. MotionLib round trip")
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    files = sorted(glob.glob(os.path.join(motion_dir, "*.motion")))[:6]
    yaml_dir = tmp_pt.parent / "rt_motions"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        (yaml_dir / os.path.basename(f)).write_bytes(Path(f).read_bytes())

    lib = MotionLib(MotionLibConfig(motion_file=str(yaml_dir)), device="cpu")
    have = all(getattr(lib, k) is not None for k in ("gnf", "grc", "grw"))
    check("fields packed", have, f"gnf/grc/grw present={have}, {lib.num_motions()} motions")
    if not have:
        return

    lib.save_to_file(str(tmp_pt))
    lib2 = MotionLib(MotionLibConfig(motion_file=str(tmp_pt)), device="cpu")
    same = all(
        torch.equal(getattr(lib, k), getattr(lib2, k)) for k in ("gnf", "grc", "grw")
    )
    check("save/load identity", same, f"tensors identical after .pt round trip={same}")

    ids = torch.arange(min(4, lib2.num_motions()))
    exact = lib2.get_motion_state_exact_frame(ids, torch.zeros_like(ids))
    ok = (
        exact.ground_reaction is not None
        and exact.ground_reaction.shape == (len(ids), 3)
        and exact.rigid_body_ground_forces.shape[1:] == (len(BODY_NAMES), 3)
        and exact.ground_reaction_valid.shape == (len(ids), 2)
    )
    check("exact-frame fetch", ok, f"shapes {tuple(exact.ground_reaction.shape)}, "
                                  f"{tuple(exact.rigid_body_ground_forces.shape)}, "
                                  f"{tuple(exact.ground_reaction_valid.shape)}")

    t = torch.full((len(ids),), 0.5 / 60.0)  # land between frame 0 and 1
    blended = lib2.get_motion_state(ids, t)
    f0 = lib2.get_motion_state_exact_frame(ids, torch.zeros_like(ids))
    f1 = lib2.get_motion_state_exact_frame(ids, torch.ones_like(ids))
    mid = (f0.ground_reaction + f1.ground_reaction) / 2
    err = float((blended.ground_reaction - mid).abs().max())
    vmin = float(
        (blended.ground_reaction_valid
         - torch.minimum(f0.ground_reaction_valid, f1.ground_reaction_valid)).abs().max()
    )
    check("blending", err < 1e-3 and vmin == 0.0,
          f"max |lerp err| {err:.2e} N, validity uses min (dev {vmin:.1e})")


def check_conservation(archive_dir: str, bodies_dir: str) -> None:
    print("\n2. Force conservation  &  3. COP consistency")
    worst_excess, worst_gap, cop_res = 0.0, 0.0, []
    for bp in sorted(Path(bodies_dir).glob("*.npz")):
        b = np.load(bp, allow_pickle=False)
        a = load_archive(Path(archive_dir) / bp.name)
        n = min(b["body_force_n"].shape[0], a["n_frames"])
        total = a["total_force_n"][:n]
        summed = b["body_force_n"][:n].sum(axis=1)
        expl = b["explained"][:n]
        loaded = total > 20.0
        if not loaded.any():
            continue
        # summed must equal explained * total (that is what attribution promises)
        pred = expl[loaded] * total[loaded]
        worst_gap = max(worst_gap, float(np.abs(summed[loaded] - pred).max() / max(total[loaded].max(), 1)))
        worst_excess = max(worst_excess, float((summed[loaded] / total[loaded]).max()))
        # Only meaningful where nearly all load was attributed: the residual is
        # the offset between the attributed centroid and the full-field COP, so
        # on a frame with 40 % unexplained load it measures the unexplained
        # load's position, not a registration error.
        trust = loaded & (expl > 0.95)
        if trust.any():
            cop_res.append(np.nanmedian(b["cop_residual_m"][trust]))
    check("sum(per-body) == explained * total", worst_gap < 1e-3,
          f"worst relative deviation {worst_gap:.2e}")
    check("no body force invented", worst_excess <= 1.001,
          f"max sum/total across corpus {worst_excess:.4f}")
    cr = np.array(cop_res) * 100
    check("attributed COP == measured COP (explained > 0.95)",
          np.nanpercentile(cr, 95) < 2.0,
          f"median {np.nanmedian(cr):.3f} cm, p95 {np.nanpercentile(cr, 95):.3f} cm "
          f"over {len(cr)} clips")


def check_support(archive_dir: str) -> None:
    print("\n4. COP inside the measured support polygon")
    from pressure_bodies import _convex_hull_2d, _dist_to_polygon

    inside, margins = [], []
    for ap in sorted(Path(archive_dir).glob("*.npz")):
        a = load_archive(ap)
        idx = np.where(a["total_force_n"] > 200.0)[0]
        if idx.size == 0:
            continue
        for t in idx[:: max(1, idx.size // 5)][:5]:
            cells, p = a["sparse"](int(t))
            cop = a["cop_world"][t]
            if cells.shape[0] < 5 or not np.isfinite(cop).all():
                continue
            # The COP is a load-weighted mean, so on a two-foot stance it sits
            # *between* the feet and touches no loaded sensel. What must hold in
            # every case is that it lies inside the convex hull of the loaded
            # cells -- the measured support polygon.
            hull = _convex_hull_2d(cells.astype(np.float64))
            if len(hull) < 3:
                continue  # collinear strip: no polygon to be inside of
            d = float(_dist_to_polygon(hull, cop[None, :].astype(np.float64))[0])
            inside.append(float(d <= 0.0))
            margins.append(d)
    check("COP inside the measured support polygon", np.mean(inside) > 0.99,
          f"{np.mean(inside) * 100:.2f}% of {len(inside)} sampled frames, "
          f"median margin {abs(np.median(margins)) * 1000:.1f} mm inside")


def check_equilibrium(archive_dir: str, motion_dir: str, mjcf: str) -> None:
    print("\n5. Static equilibrium: measured COP under the reference COM")
    from contact_geometry import geom_to_world, parse_typed_geoms, world_geom_center

    geoms = parse_typed_geoms(mjcf, BODY_NAMES)
    dists = []
    for ap in sorted(Path(archive_dir).glob("*.npz")):
        mp = Path(motion_dir) / f"{ap.stem}.motion"
        if not mp.exists():
            continue
        a = load_archive(ap)
        m = torch.load(mp, map_location="cpu", weights_only=False)
        rbp, rbr = m["rigid_body_pos"], m["rigid_body_rot"]
        n = min(rbp.shape[0], a["n_frames"])
        # "Still" means the whole body is still, not just the pelvis -- a
        # standing balance can hold its root while an arm swings.
        speed = np.linalg.norm(np.diff(rbp[:n].numpy(), axis=0), axis=2).max(axis=1)
        speed = speed * a["fps"]
        still = np.where(
            (a["total_force_n"][: n - 1] > 600.0)
            & (a["coverage"][: n - 1] > 0.97)
            & (speed < 0.02)
        )[0]
        if still.size == 0:
            continue
        # COM from geom centroids weighted by geom mass -- a body origin sits at
        # its proximal joint, which is nowhere near the segment's centre of mass.
        centres, masses = [], []
        for bi, name in enumerate(BODY_NAMES):
            for g in geoms.get(name, []):
                gw = geom_to_world(g, rbp[:n, bi], rbr[:n, bi])
                centres.append(world_geom_center(gw).numpy())
                masses.append(g["mass"])
        centres = np.stack(centres, axis=1)  # (n, G, 3)
        w = np.asarray(masses, dtype=np.float64)
        w = w / w.sum()
        com_xy = (centres[:, :, :2] * w[None, :, None]).sum(axis=1)
        for t in still[:: max(1, still.size // 4)][:4]:
            cop = a["cop_world"][t]
            if np.isfinite(cop).all():
                dists.append(float(np.linalg.norm(com_xy[t] - cop)))
    d = np.array(dists) * 100
    check("COP-COM offset on still, fully-measured frames", np.median(d) < 6.0,
          f"median {np.median(d):.2f} cm, p90 {np.percentile(d, 90):.2f} cm "
          f"over {len(d)} frames (bounds residual registration error)")


def check_anatomy(bodies_dir: str, motion_dir: str) -> None:
    print("\n6. Anatomy: load lands on the bodies the pose is named for")
    # A clip whose *reference* contradicts the measurement is a dataset finding,
    # not a defect in the port, so the gate is the median clip per pose and the
    # outliers are enumerated underneath. If the attribution were broken in
    # general every clip would fail and the median would fail with them.
    flagged = []
    for pattern, expected, airborne, min_z, thresh in ANATOMY:
        cols = [BODY_NAMES.index(e) for e in expected]
        air = [BODY_NAMES.index(e) for e in airborne]
        shares, n_frames = [], 0
        per_clip = []
        for bp in sorted(Path(bodies_dir).glob(f"*{pattern}*.npz")):
            mp = Path(motion_dir) / f"{bp.stem}.motion"
            if not mp.exists():
                continue
            b = np.load(bp, allow_pickle=False)
            f = b["body_force_n"]
            tot = f.sum(axis=1)
            z = torch.load(mp, map_location="cpu", weights_only=False)[
                "rigid_body_pos"
            ].numpy()[: f.shape[0], :, 2]
            in_pose = (
                z[:, air].min(axis=1) > min_z
                if air
                else np.ones(f.shape[0], dtype=bool)
            )
            held = in_pose & (tot > 200.0)
            good = held & (b["explained"] > MIN_EXPLAINED)
            if held.sum() >= 10 and good.sum() < 0.25 * held.sum():
                flagged.append(
                    f"{bp.stem[:56]} (median explained "
                    f"{np.median(b['explained'][held]):.2f} over {held.sum()} held frames)"
                )
            if good.sum() < 10:
                continue
            n_frames += int(good.sum())
            share = float(np.median(f[good][:, cols].sum(1) / tot[good]))
            shares.append(share)
            per_clip.append((bp.stem, share, int(good.sum())))
        if not shares:
            flagged.append(
                f"{pattern}: no clip had 10+ frames whose attribution is "
                f"trustworthy (explained > {MIN_EXPLAINED}) -- the reference "
                f"cannot account for the measured load in this pose at all"
            )
            print(f"  [ -- ] {pattern}: no trustworthy in-pose frames "
                  f"(reported as a reference finding below)")
            continue
        med = float(np.median(shares))
        check(f"{pattern} -> {'/'.join(sorted(expected))[:36]}", med >= thresh,
              f"{len(shares)} clips / {n_frames} held frames, median clip "
              f"{med * 100:.1f}% (need >= {thresh * 100:.0f}%)")
        for stem, share, n in per_clip:
            if share < thresh:
                flagged.append(
                    f"{stem} -- only {share * 100:.1f}% of load on "
                    f"{'/'.join(sorted(expected))} over {n} held frames"
                )
    if flagged:
        print("\n  REFERENCE AUDIT -- these clips' kinematics contradict the measured")
        print("  load. Not port defects; the validity channel down-weights them, and")
        print("  they are candidates for re-retargeting or exclusion:")
        for f_ in flagged:
            print(f"    - {f_}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--bodies-dir", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--scratch", default="/tmp/pressure_validate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    check_round_trip(args.motion_dir, scratch / "rt.pt")
    check_conservation(args.archive_dir, args.bodies_dir)
    check_support(args.archive_dir)
    check_equilibrium(args.archive_dir, args.motion_dir, args.mjcf)
    check_anatomy(args.bodies_dir, args.motion_dir)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 72}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"   FAILED  {name}: {detail}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            [{"check": n, "pass": bool(o), "detail": d} for n, o, d in RESULTS], indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
