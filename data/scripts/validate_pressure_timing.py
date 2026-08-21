# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent temporal check that the MOYO pressure channel is frame-aligned
with the kinematics of the clip it is attached to.

``notes/Moyo_pressure_port.MD`` establishes alignment by *construction* (the
CSV's ``begin_time x 60`` indexes the XML stream, then frame counts must match
exactly).  That argument is about bookkeeping.  This script asks the data,
using no marker file, no mat geometry and no per-body attribution.

Two tests
---------
**COP test (primary).**  The measured centre of pressure and the reference COM
must move together.  Correlating *velocities* rather than positions is what
makes the lag identifiable: positions drift slowly, so their correlation is
nearly flat in lag and an ``argmax`` over it is meaningless.

**Force test (secondary, often under-powered).**  For a rigid body
``Fz = m (g + a_com,z)``.  The reference gives ``a_com,z`` by twice
differentiating a *retargeted SMPL fit*, which is noisy enough that most yoga
clips simply do not contain enough vertical dynamics to resolve timing.  It is
reported, and only asserted on clips that have the dynamics to support it.

Reading the result
------------------
Do **not** gate on ``argmax == 0``.  The correlation surface is flat near the
optimum, so the argmax wanders by a few frames on perfectly aligned data — the
easy-pose controls (chair, tree, warrior III) wander exactly as much as the hard
clips do.  The identifiable statistic is the *penalty for assuming lag 0*:
``r(best) - r(0)``.  If that is small, the data cannot distinguish the shipped
alignment from the best one, which is the strongest statement a correlation test
can make in favour of it.

Usage::

    PYTHONPATH=. python data/scripts/validate_pressure_timing.py \
      --clips-yaml data/smpl/yoga_yogi_hard29_pressure.yaml --with-controls
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
G = 9.81
SUBJECT_WEIGHT_N = 700.0  # measured, Moyo_pressure_port.MD 4.1
MAX_LAG = 12  # +-200 ms at 60 fps

# Easy, well-measured clips used as a null: whatever wander they show is the
# test's own resolution, not a defect of the clip under scrutiny.
CONTROL_CLIPS = [
    "220923_Chair_Pose_or_Utkatasana_-b",
    "220923_Tree_Pose_or_Vrksasana_-a",
    "220926_Warrior_III_Pose_or_Virabhadrasana_III_-a",
    "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
]


def body_masses(mjcf: str) -> np.ndarray:
    geoms = parse_typed_geoms(mjcf, BODY_NAMES)
    return np.array(
        [sum(g.get("mass", 0.0) for g in geoms[n]) for n in BODY_NAMES], np.float64
    )


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average along axis 0 (edge-replicated)."""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    k = np.ones(w) / w
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, xp)


def xcorr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, max_lag: int = MAX_LAG):
    """Correlation of ``a`` with ``b`` shifted by each integer lag.

    Positive lag means ``b`` lags ``a``.  Returns an array of length
    ``2*max_lag+1`` with NaN where too few frames survive the mask.
    """
    out = np.full(2 * max_lag + 1, np.nan)
    for j, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            x, y = a[lag:], b[: len(b) - lag]
            m = mask[lag:] & mask[: len(mask) - lag]
        else:
            x, y = a[: len(a) + lag], b[-lag:]
            m = mask[: len(mask) + lag] & mask[-lag:]
        if m.sum() < 60:
            continue
        x, y = x[m], y[m]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        out[j] = float(np.corrcoef(x, y)[0, 1])
    return out


def _peak(cc: np.ndarray):
    if np.all(np.isnan(cc)):
        return None, np.nan, np.nan
    i = int(np.nanargmax(cc))
    return i - MAX_LAG, float(cc[i]), float(cc[MAX_LAG])


def check_clip(path: Path, masses: np.ndarray, smooth_w: int) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if any(d.get(k) is None for k in
           ("ground_reaction", "ground_reaction_valid", "rigid_body_pos")):
        return dict(clip=path.stem, status="no_pressure")

    gr = d["ground_reaction"].numpy().astype(np.float64)
    gv = d["ground_reaction_valid"].numpy().astype(np.float64)
    bp = d["rigid_body_pos"].numpy().astype(np.float64)
    dt = 1.0 / float(d.get("fps", 60))
    T = len(gr)

    com = (bp * masses[None, :, None]).sum(1) / masses.sum()
    ok = gv[:, 0] >= 0.90  # frames the mat plausibly saw in full

    # ---- COP test: velocities, on the frames where the COM actually moves ----
    cop = smooth(gr[:, 1:3], 5)
    com_xy = smooth(com[:, :2], 5)
    v_cop = np.gradient(cop, dt, axis=0)
    v_com = np.gradient(com_xy, dt, axis=0)
    speed = np.linalg.norm(v_com, axis=1)
    fast = ok & (speed > np.percentile(speed[ok], 60)) if ok.any() else ok
    cc = np.nanmean(
        np.stack([xcorr(v_com[:, 0], v_cop[:, 0], fast),
                  xcorr(v_com[:, 1], v_cop[:, 1], fast)]), axis=0
    )
    cop_lag, cop_rmax, cop_r0 = _peak(cc)

    # ---- Force test: Fz vs m(g + a_com,z) on frames with vertical dynamics ---
    com_s = smooth(com, smooth_w)
    acc = np.zeros_like(com_s)
    acc[1:-1] = (com_s[2:] - 2 * com_s[1:-1] + com_s[:-2]) / dt**2
    acc[0], acc[-1] = acc[1], acc[-2]
    fz_pred = (SUBJECT_WEIGHT_N / G) * (G + acc[:, 2])
    fz_meas = smooth(gr[:, 0:1], smooth_w)[:, 0]
    dyn = ok & (np.abs(fz_pred - SUBJECT_WEIGHT_N) > 60.0)
    cf = xcorr(fz_pred, fz_meas, dyn)
    f_lag, f_rmax, f_r0 = _peak(cf)

    return dict(
        clip=path.stem, status="ok", T=T,
        n_fast=int(fast.sum()), n_dyn=int(dyn.sum()), n_ok=int(ok.sum()),
        cop_lag=cop_lag, cop_r0=cop_r0, cop_rmax=cop_rmax,
        cop_penalty=(cop_rmax - cop_r0) if np.isfinite(cop_rmax) else np.nan,
        f_lag=f_lag, f_r0=f_r0, f_rmax=f_rmax,
        f_penalty=(f_rmax - f_r0) if np.isfinite(f_rmax) else np.nan,
        cop_com_med_cm=float(
            np.median(np.linalg.norm(cop[ok] - com_xy[ok], axis=1)) * 100
        ) if ok.any() else float("nan"),
    )


def load_clip_names(args) -> list[str]:
    if args.clips_yaml:
        import yaml
        man = yaml.safe_load(Path(args.clips_yaml).read_text())
        return [Path(e["file"]).stem for e in man["motions"]]
    if args.clips:
        return list(args.clips)
    return sorted(p.stem for p in Path(args.motion_dir).glob("*.motion"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-dir", default="data/smpl/yoga_motions_proto_yogi_pressure")
    ap.add_argument("--mjcf", default="data/assets/smpl/smpl_yogi03596_lowtorque.xml")
    ap.add_argument("--clips-yaml", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--with-controls", action="store_true",
                    help="Also run known-good easy clips as a null.")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--max-lag0-penalty", type=float, default=0.05,
                    help="Max acceptable r(best) - r(0) on the COP test.")
    ap.add_argument("--min-cop-r0", type=float, default=0.45)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    masses = body_masses(args.mjcf)
    mdir = Path(args.motion_dir)
    names = load_clip_names(args)
    if args.with_controls:
        names = names + ["__CONTROLS__"] + CONTROL_CLIPS

    print(f"# humanoid mass from MJCF densities: {masses.sum():.2f} kg")
    print(f"# criterion: COP-velocity r(0) >= {args.min_cop_r0}, "
          f"r(best) - r(0) <= {args.max_lag0_penalty}")
    print("# the argmax lag is printed for information only - the surface is flat\n")

    hdr = (f"{'clip':56s} {'nFast':6s} {'lag':4s} {'r(0)':6s} {'rmax':6s} "
           f"{'pen':6s} | {'nDyn':5s} {'Flag':4s} {'Fr(0)':6s} {'Fpen':6s} | "
           f"{'|COP-COM|cm':11s}")
    print(hdr); print("-" * len(hdr))

    rows, bad, missing = [], [], []
    in_controls = False
    for n in names:
        if n == "__CONTROLS__":
            print("-" * len(hdr) + "   (controls below: easy, well-measured clips)")
            in_controls = True
            continue
        p = mdir / f"{n}.motion"
        if not p.is_file():
            missing.append(n)
            print(f"{n[:56]:56s}  MISSING .motion")
            continue
        r = check_clip(p, masses, args.smooth)
        r["control"] = in_controls
        rows.append(r)
        if r["status"] != "ok":
            print(f"{r['clip'][:56]:56s}  {r['status']}")
            continue
        flag = ""
        if not (np.isfinite(r["cop_r0"]) and r["cop_r0"] >= args.min_cop_r0):
            flag += " LOW-COP-R"
        if not (np.isfinite(r["cop_penalty"]) and r["cop_penalty"] <= args.max_lag0_penalty):
            flag += " COP-LAG"
        if flag and not in_controls:
            bad.append((r["clip"], flag.strip()))
        fr0 = "  n/a " if not np.isfinite(r["f_r0"]) else f"{r['f_r0']:6.3f}"
        fpn = "  n/a " if not np.isfinite(r["f_penalty"]) else f"{r['f_penalty']:6.3f}"
        print(f"{r['clip'][:56]:56s} {r['n_fast']:6d} {str(r['cop_lag']):4s} "
              f"{r['cop_r0']:6.3f} {r['cop_rmax']:6.3f} {r['cop_penalty']:6.3f} | "
              f"{r['n_dyn']:5d} {str(r['f_lag']):4s} {fr0} {fpn} | "
              f"{r['cop_com_med_cm']:11.2f}{flag}")

    tested = [r for r in rows if r["status"] == "ok" and not r["control"]]
    ctrl = [r for r in rows if r["status"] == "ok" and r["control"]]
    if ctrl:
        pens = [r["cop_penalty"] for r in ctrl if np.isfinite(r["cop_penalty"])]
        lags = [abs(r["cop_lag"]) for r in ctrl if r["cop_lag"] is not None]
        print(f"\ncontrol null: COP lag0-penalty max {max(pens):.3f}, "
              f"|argmax lag| up to {max(lags)} frames")
    print(f"\n{len(tested) - len(bad)}/{len(tested)} clips pass the alignment criterion.")
    if missing:
        print(f"MISSING pressure motion file ({len(missing)}): {missing}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json_out}")
    if bad:
        print("\nOUT OF TOLERANCE:")
        for c, f in bad:
            print(f"  {c}: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
