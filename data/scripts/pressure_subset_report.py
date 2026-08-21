# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-clip usability report for the measured MOYO pressure on a motion subset.

Answers the questions a reward built on this channel has to answer first:

* **How many frames actually carry supervision?**  Both measured channels are
  gated: ``ground_reaction`` by mat coverage, ``rigid_body_ground_forces`` by
  coverage x attribution.  A term that ignores the gates charges the policy for
  load the mat never recorded.
* **What does the reference say during the hold?**  For an inversion or an arm
  balance the whole point is which zone carries the load once the feet leave,
  so the hold-phase zone shares are reported separately from the clip average.
* **Which clips are unusable and why?**  Three distinct failures show up:
  the mat misses load (low coverage), the attribution cannot place it (low
  ``explained``), or the retargeted reference disagrees with the measurement
  (load on the wrong zone for the named pose).

Measured caveat this report makes visible
-----------------------------------------
During the *hold* of a forearm-supported pose the mat reads well below body
weight -- Pincha 0.77, Bakasana -b 0.76, Scorpion -b 0.81 -- while hand-supported
poses read 0.93-0.97.  These clips are not running off the mat edge, so this is a
measurement bias, not a field-of-view loss.  **Do not use absolute newtons as a
target.**  Zone *shares* and the COP are invariant to a uniform gain error;
absolute force is not.

Usage::

    PYTHONPATH=. python data/scripts/pressure_subset_report.py \
      --clips-yaml data/smpl/yoga_yogi_hard29_pressure.yaml --markdown
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
from contact_geometry import parse_typed_geoms  # noqa: E402

BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle",
    "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder",
    "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Hand",
]
IDX = {n: i for i, n in enumerate(BODY_NAMES)}

# Support zones. Deliberately coarser than bodies: the attribution's dominant
# error is leakage *within* a limb chain (a shin capsule stealing from the foot
# it runs to), which zone pooling absorbs.
ZONES = {
    "HANDS": ["L_Wrist", "L_Hand", "R_Wrist", "R_Hand"],
    "FOREARM": ["L_Elbow", "R_Elbow"],
    "FEET": ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"],
    "SHANK": ["L_Knee", "R_Knee"],
    "HEAD": ["Neck", "Head"],
    "TORSO": ["Pelvis", "Torso", "Spine", "Chest", "L_Thorax", "R_Thorax",
              "L_Hip", "R_Hip", "L_Shoulder", "R_Shoulder"],
}
ZID = {k: np.array([IDX[n] for n in v]) for k, v in ZONES.items()}
ZK = list(ZONES)

SUBJECT_WEIGHT_N = 700.0
FEET_UP_M = 0.25  # ankle height above which both feet count as off the floor


def zone_shares(fz: np.ndarray) -> np.ndarray:
    tot = np.maximum(fz.sum(-1, keepdims=True), 1e-6)
    return np.stack([fz[..., ZID[k]].sum(-1) for k in ZK], -1) / tot


def analyse(path: Path, cov_min: float, expl_min: float) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("ground_reaction") is None:
        return {"clip": path.stem, "status": "no_pressure"}
    gr = d["ground_reaction"].numpy().astype(np.float64)
    gv = d["ground_reaction_valid"].numpy().astype(np.float64)
    gnf = np.maximum(d["rigid_body_ground_forces"].numpy()[:, :, 2], 0.0).astype(np.float64)
    bp = d["rigid_body_pos"].numpy().astype(np.float64)
    T = len(gr)

    cov, covex = gv[:, 0], gv[:, 1]
    expl = covex / np.maximum(cov, 1e-6)
    gate_cop = cov >= cov_min                      # gates ground_reaction
    gate_body = gate_cop & (expl >= expl_min)      # gates rigid_body_ground_forces
    hold = bp[:, [IDX["L_Ankle"], IDX["R_Ankle"]], 2].min(1) > FEET_UP_M
    sh = zone_shares(gnf)

    r = {
        "clip": path.stem, "status": "ok", "T": T, "dur_s": T / 60.0,
        "cov_p50": float(np.percentile(cov, 50)),
        "cov_p10": float(np.percentile(cov, 10)),
        "gate_cop_frac": float(gate_cop.mean()),
        "gate_body_frac": float(gate_body.mean()),
        "hold_frac": float(hold.mean()),
        "fz_hold_bw": float(np.median(gr[hold, 0]) / SUBJECT_WEIGHT_N) if hold.any() else np.nan,
        "gate_cop_hold": float(gate_cop[hold].mean()) if hold.any() else np.nan,
        "gate_body_hold": float(gate_body[hold].mean()) if hold.any() else np.nan,
    }
    hg = hold & gate_body
    for k in ZK:
        j = ZK.index(k)
        r[f"hold_{k}"] = float(np.median(sh[hg, j])) if hg.sum() >= 20 else np.nan
    # zones the reference says carry NOTHING during the gated hold: the
    # one-sided constraint a reward can enforce without any absolute force.
    if hg.sum() >= 20:
        r["zero_zones"] = [k for k in ZK
                           if float(np.percentile(sh[hg, ZK.index(k)], 90)) < 0.02]
    else:
        r["zero_zones"] = []
    return r


def flags(r: dict) -> list[str]:
    f = []
    if r["status"] != "ok":
        return ["NO PRESSURE"]
    if r["gate_body_hold"] < 0.25:
        f.append("per-body channel mostly gated off in the hold")
    if r["fz_hold_bw"] < 0.85:
        f.append(f"mat reads {r['fz_hold_bw']:.2f} BW in the hold (gain bias)")
    if r["hold_frac"] < 0.05:
        f.append("no feet-up hold detected")
    if not np.isnan(r.get("hold_HEAD", np.nan)) and r["clip"].count("Sirsasana") \
            and "Janu" not in r["clip"] and r["hold_HEAD"] < 0.02:
        f.append("headstand with ZERO head load - reference does not ground the head")
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-dir", default="data/smpl/yoga_motions_proto_yogi_pressure")
    ap.add_argument("--clips-yaml", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--cov-min", type=float, default=0.90)
    ap.add_argument("--expl-min", type=float, default=0.90)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    mdir = Path(args.motion_dir)
    if args.clips_yaml:
        import yaml
        names = [Path(e["file"]).stem
                 for e in yaml.safe_load(Path(args.clips_yaml).read_text())["motions"]]
    elif args.clips:
        names = list(args.clips)
    else:
        names = sorted(p.stem for p in mdir.glob("*.motion"))

    rows = [analyse(mdir / f"{n}.motion", args.cov_min, args.expl_min)
            for n in names if (mdir / f"{n}.motion").is_file()]

    sep = " | " if args.markdown else "  "
    cols = ["clip", "s", "hold%", "Fz/BW", "cov50", "gCOP%", "gBODY%",
            "gCOPh%", "gBODYh%", "HANDS", "FOREARM", "FEET", "HEAD", "TORSO"]
    if args.markdown:
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * len(cols)) + "|")
    else:
        print(sep.join(f"{c:>8s}" if i else f"{c:<52s}" for i, c in enumerate(cols)))

    def fmt(v, w=8, p=2):
        return " " * (w - 3) + "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) \
            else f"{v:{w}.{p}f}"

    for r in rows:
        if r["status"] != "ok":
            print(f"{r['clip'][:52]:<52s}{sep}NO PRESSURE")
            continue
        vals = [f"{r['dur_s']:8.0f}", f"{100*r['hold_frac']:8.0f}", fmt(r["fz_hold_bw"]),
                fmt(r["cov_p50"]), f"{100*r['gate_cop_frac']:8.0f}",
                f"{100*r['gate_body_frac']:8.0f}", fmt(100*r["gate_cop_hold"], 8, 0),
                fmt(100*r["gate_body_hold"], 8, 0)] + \
               [fmt(r.get(f"hold_{k}")) for k in ("HANDS", "FOREARM", "FEET", "HEAD", "TORSO")]
        if args.markdown:
            print("| " + " | ".join([r["clip"]] + [v.strip() for v in vals]) + " |")
        else:
            print(f"{r['clip'][:52]:<52s}" + sep + sep.join(vals))

    print("\nFlags:")
    any_flag = False
    for r in rows:
        f = flags(r)
        if f:
            any_flag = True
            print(f"  {r['clip']}\n      - " + "\n      - ".join(f))
    if not any_flag:
        print("  (none)")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        print(f"\n{len(ok)} clips; COP channel usable on "
              f"{100*np.mean([r['gate_cop_frac'] for r in ok]):.0f}% of frames "
              f"({100*np.nanmean([r['gate_cop_hold'] for r in ok]):.0f}% of hold frames); "
              f"per-body channel on {100*np.mean([r['gate_body_frac'] for r in ok]):.0f}% "
              f"({100*np.nanmean([r['gate_body_hold'] for r in ok]):.0f}% of hold).")
        print(f"gates: coverage >= {args.cov_min}, explained >= {args.expl_min}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
