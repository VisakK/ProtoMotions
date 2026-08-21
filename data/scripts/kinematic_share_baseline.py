# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How much of the measured load distribution is already implied by the pose?

The pressure terms are worth their weight only where the mat says something the
reference kinematics does not.  ``notes/Pressure_supervision_design.MD`` §3
established that for crow it does -- the reference foot sits 9.2 cm above the
floor while carrying 636 N, so *no* kinematic threshold separates the loaded
frames from the unloaded ones.  Whether the same holds for a standing balance is
a different question, and it decides whether a pressure-supervised run on those
clips is measuring the supervision or measuring the tracker.

This answers it **without training anything**.  For every frame it builds a
purely kinematic prediction of the zone load shares -- each body weighted by how
close its lowest collision geom is to the floor -- and reports the total-variation
distance to the *measured* shares, on gated frames only.  Small distance means
the pose already determines the load distribution and the mat is redundant there;
large distance means the mat carries information the reference cannot.

The proxy is deliberately given every advantage: sigma is swept and the **best**
value per group is reported, so a large residual cannot be an artefact of a badly
chosen kernel width.  Two nulls are reported alongside it, because a TV distance
means nothing without knowing what chance looks like:

* ``best constant`` -- put every newton on whichever single zone carries the most
  measured load, chosen **per clip**.  This is the strongest constant available
  and is deliberately generous: if it already scores near 0, the target is a
  constant and the reward term asking to match it is vacuous.
* ``uniform``       -- spread load equally over the zones.

``--both-zone-sets`` reruns each group with left and right split out of HANDS and
FEET, which is the question that matters once a target turns out to be constant:
is the information absent, or merely pooled away?  Measured answer, 2026-08-16 --
single-leg 0.006 -> 0.162 (27x), hard-29 0.364 -> 0.535 (+47 %): pooled away, and
splitting helps *both* groups.

Usage::

    PYTHONPATH=. python data/scripts/kinematic_share_baseline.py
    PYTHONPATH=. python data/scripts/kinematic_share_baseline.py --both-zone-sets
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
from add_onmat_gate_to_motions import BODY_NAMES, lowest_geom_per_body  # noqa: E402
from contact_geometry import parse_typed_geoms  # noqa: E402
from pressure_policy_report import ZID, ZK  # noqa: E402

GATED_DIR = "data/smpl/yoga_motions_proto_yogi_pressure_gated"
MJCF = "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
GATE = 0.90
MIN_TOTAL_N = 30.0
SIGMAS_CM = (1.0, 2.0, 3.0, 5.0, 8.0)

# The shipped 6-zone set pools left and right into one HANDS and one FEET. That
# is right for the leakage it was designed against -- a shin capsule stealing
# from the foot it runs to -- but left and right limbs are different chains half
# a metre apart, and pooling them discards the only thing the mat says about a
# standing balance. `--split-lr` scores the same clips with the sides separated.
SPLIT_ZONES = {
    "L_HAND": ["L_Wrist", "L_Hand"],
    "R_HAND": ["R_Wrist", "R_Hand"],
    "FOREARM": ["L_Elbow", "R_Elbow"],
    "L_FOOT": ["L_Ankle", "L_Toe"],
    "R_FOOT": ["R_Ankle", "R_Toe"],
    "SHANK": ["L_Knee", "R_Knee"],
    "HEAD": ["Neck", "Head"],
    "TORSO": ["Pelvis", "Torso", "Spine", "Chest", "L_Thorax", "R_Thorax",
              "L_Hip", "R_Hip", "L_Shoulder", "R_Shoulder"],
}


def zone_index(split_lr: bool):
    """-> (zone names, list of body-index arrays)."""
    if not split_lr:
        return list(ZK), [ZID[k] for k in ZK]
    from pressure_policy_report import COMMON_BODY_NAMES as C
    names = list(SPLIT_ZONES)
    return names, [np.array([C.index(b) for b in SPLIT_ZONES[k]]) for k in names]


def zone_shares(per_body: np.ndarray, idx) -> np.ndarray:
    """[T, 24] non-negative -> [T, Z] shares that sum to 1 (0 where nothing)."""
    z = np.stack([np.maximum(per_body, 0.0)[:, i].sum(1) for i in idx], 1)
    tot = z.sum(1, keepdims=True)
    return np.divide(z, tot, out=np.zeros_like(z), where=tot > 0)


def kinematic_shares(gap: np.ndarray, sigma_m: float, idx) -> np.ndarray:
    """Predict zone shares from geometry alone: closer to the floor, more load.

    A Gaussian in the ground gap, the same functional form and scale as the
    attribution kernel in ``attribute_pressure_to_bodies.py``. It has no access
    to any force.
    """
    w = np.exp(-((np.minimum(gap, 1.0) / sigma_m) ** 2))
    return zone_shares(w, idx)


def tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(-1)


def score_clip(path: Path, geoms, names, idx) -> dict | None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    gv = d.get("ground_reaction_valid")
    if gv is None or gv.shape[1] < 3:
        return None
    fz = np.maximum(d["rigid_body_ground_forces"].numpy()[:, :, 2], 0.0)
    live = (gv[:, 2].numpy() >= GATE) & (fz.sum(1) >= MIN_TOTAL_N)
    if live.sum() < 20:
        return {"clip": path.stem, "n_live": int(live.sum())}

    gap, _ = lowest_geom_per_body(
        geoms, d["rigid_body_pos"].to(torch.float64), d["rigid_body_rot"].to(torch.float64)
    )
    s_ref = zone_shares(fz, idx)[live]
    out = {"clip": path.stem, "n_live": int(live.sum())}
    out["tv_by_sigma"] = {
        f"{s:g}cm": float(tv(kinematic_shares(gap, s / 100.0, idx)[live], s_ref).mean())
        for s in SIGMAS_CM
    }
    best = min(out["tv_by_sigma"], key=out["tv_by_sigma"].get)
    out["tv_best"], out["sigma_best"] = out["tv_by_sigma"][best], best

    # The strongest constant available, chosen per group in the caller's favour:
    # whichever single zone carries the most measured load on average. Anything
    # weaker would understate how predictable the target is.
    best = int(s_ref.mean(0).argmax())
    const = np.zeros_like(s_ref); const[:, best] = 1.0
    out["tv_null_best_constant"] = float(tv(const, s_ref).mean())
    out["null_zone"] = names[best]
    out["tv_null_uniform"] = float(tv(np.full_like(s_ref, 1 / len(names)), s_ref).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gated-dir", default=GATED_DIR)
    ap.add_argument("--mjcf", default=MJCF)
    ap.add_argument("--groups", nargs="*", default=["single_leg", "hard29"])
    ap.add_argument("--split-lr", action="store_true",
                    help="score with left/right split out of HANDS and FEET (8 zones)")
    ap.add_argument("--both-zone-sets", action="store_true",
                    help="run each group under both zone sets and print the contrast")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from select_hard_pose_subsets import ARM_BALANCES, INVERSIONS, SINGLE_LEG_BALANCES, flat

    groups = {"single_leg": list(SINGLE_LEG_BALANCES),
              "hard29": flat(INVERSIONS) + flat(ARM_BALANCES)}
    geoms = parse_typed_geoms(args.mjcf, BODY_NAMES)
    gated = Path(args.gated_dir)

    if args.both_zone_sets:
        print("Does splitting left from right make the target informative?\n")
        print(f"{'group':14s} {'zones':>7s} {'best-constant TV':>18s} {'that zone':>12s}")
        print("-" * 56)
        for split in (False, True):
            names, idx = zone_index(split)
            for gname in args.groups:
                rows = [r for r in (score_clip(gated / f"{c}.motion", geoms, names, idx)
                                    for c in groups[gname]
                                    if (gated / f"{c}.motion").is_file())
                        if r and "tv_best" in r]
                if not rows:
                    continue
                w = np.array([r["n_live"] for r in rows], float)
                const = float(np.average([r["tv_null_best_constant"] for r in rows],
                                         weights=w))
                zone = max(set(r["null_zone"] for r in rows),
                           key=[r["null_zone"] for r in rows].count)
                print(f"{gname:14s} {len(names):7d} {const:18.3f} {zone:>12s}")
        print("\nA best-constant TV near 0 means the target is a constant and the "
              "term is vacuous.")
        return

    names, idx = zone_index(args.split_lr)
    results = {}
    for gname in args.groups:
        rows = []
        print(f"\n### {gname}")
        print(f"{'clip':56s} {'frames':>7s} {'TV(kin)':>8s} {'sigma':>6s} "
              f"{'TV(const)':>9s} {'TV(unif)':>9s}")
        for clip in groups[gname]:
            p = gated / f"{clip}.motion"
            if not p.is_file():
                continue
            r = score_clip(p, geoms, names, idx)
            if r is None or "tv_best" not in r:
                print(f"{clip[:56]:56s} {'--':>7s}  (no gated frames)")
                continue
            rows.append(r)
            print(f"{clip[:56]:56s} {r['n_live']:7d} {r['tv_best']:8.3f} "
                  f"{r['sigma_best']:>6s} {r['tv_null_best_constant']:9.3f} "
                  f"{r['tv_null_uniform']:9.3f}")
        if rows:
            w = np.array([r["n_live"] for r in rows], float)
            def wm(key):
                return float(np.average([r[key] for r in rows], weights=w))
            results[gname] = {
                "clips": len(rows), "frames": int(w.sum()),
                "tv_kinematic": wm("tv_best"),
                "tv_null_best_constant": wm("tv_null_best_constant"),
                "tv_null_uniform": wm("tv_null_uniform"),
                "per_clip": rows,
            }
            g = results[gname]
            print(f"{'POOLED (frame-weighted)':56s} {g['frames']:7d} "
                  f"{g['tv_kinematic']:8.3f} {'':>6s} {g['tv_null_best_constant']:9.3f} "
                  f"{g['tv_null_uniform']:9.3f}")

    if results:
        print("\n" + "=" * 86)
        print("Does the zone-share target carry information at all?")
        print("  TV(const) is what the best CONSTANT prediction scores. If it is ~0 the")
        print("  target is a constant and matching it demonstrates nothing.")
        for gname, g in results.items():
            const = g["tv_null_best_constant"]
            verdict = ("TARGET IS ~CONSTANT: the zone term is vacuous here"
                       if const < 0.05 else
                       "target varies: the zone term has something to ask for")
            print(f"\n  {gname:12s} TV(best constant) {const:.3f}   "
                  f"TV(kinematic proxy) {g['tv_kinematic']:.3f}")
            print(f"  {'':12s} -> {verdict}")
            if const >= 0.05:
                gain = 1.0 - g["tv_kinematic"] / const
                print(f"  {'':12s}    geometry alone removes {100 * gain:.0f}% of the "
                      "constant predictor's error")
        print("=" * 86)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
