# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a rollout against the measured pressure: the metrics that discriminate.

``notes/Pressure_supervision_design.MD`` 4 establishes that neither tracking
error nor the shipped lift-off percentages separate a good arm balance from a
bad one — but two measured quantities do:

1. **Non-hand ground load while the hands are down**, in body weights. The
   trailing toe of the best crow checkpoint carries 34 % BW where the human
   carries 12 N. Aggregate metrics are blind to this.
2. **Zone-share total variation** against the measured human on gated frames.
   It recovers the documented crow/side-crow ordering with no tuning.

Both are computed here from a ``rollout.npz`` written by
``record_contact_physics.py``, so this is CPU-only and needs no simulator.

The body-order trap
-------------------
``rollout.npz`` stores ``pair_force_w`` in **simulator** body order
(``npz["body_names"]``), which is *not* the COMMON/MJCF order the ``.motion``
fields use — sim index 10 is ``L_Toe``, common index 10 is ``Spine``. Reading it
with the wrong order produces a confident, completely wrong answer (it once
reported a policy "resting on its spine"). This script remaps by name and
asserts the two orders differ, so the remap can never be silently dropped.

Usage::

    # 1) record (needs a GPU, num_envs=1 -- see notes/Physics_insights.md)
    python data/scripts/record_contact_physics.py \
      --checkpoint results/<run>/last.ckpt \
      --motion-file data/smpl/yoga_motions_proto_yogi_pressure_gated/<clip>.motion \
      --out-dir results/<run>_pressure_eval

    # 2) score (CPU only)
    PYTHONPATH=. python data/scripts/pressure_policy_report.py \
      --in-dir results/<run>_pressure_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

COMMON_BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle",
    "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder",
    "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Hand",
]
SUPPORT_ZONES = {
    "HANDS": ["L_Wrist", "L_Hand", "R_Wrist", "R_Hand"],
    "FOREARM": ["L_Elbow", "R_Elbow"],
    "FEET": ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"],
    "SHANK": ["L_Knee", "R_Knee"],
    "HEAD": ["Neck", "Head"],
    "TORSO": ["Pelvis", "Torso", "Spine", "Chest", "L_Thorax", "R_Thorax",
              "L_Hip", "R_Hip", "L_Shoulder", "R_Shoulder"],
}
ZK = list(SUPPORT_ZONES)
ZID = {k: np.array([COMMON_BODY_NAMES.index(n) for n in v])
       for k, v in SUPPORT_ZONES.items()}
ROBOT_WEIGHT_N = 74.0 * 9.81
ON_N = 5.0            # a body counts as grounded above this
FEET_UP_M = 0.25


def to_common_order(fz_sim: np.ndarray, sim_names: list) -> np.ndarray:
    """[T, B_sim] in simulator order -> [T, 24] in COMMON order."""
    missing = [n for n in COMMON_BODY_NAMES if n not in sim_names]
    if missing:
        raise ValueError(f"rollout is missing bodies {missing}")
    return fz_sim[:, [sim_names.index(n) for n in COMMON_BODY_NAMES]]


def zone_shares(fz: np.ndarray, min_total_n: float = 30.0):
    fz = np.maximum(fz, 0.0)
    zone = np.stack([fz[..., ZID[k]].sum(-1) for k in ZK], -1)
    total = zone.sum(-1)
    return zone / np.maximum(total, min_total_n)[..., None], total


def score(npz_path: Path, motion_dir: Path, valid_threshold: float = 0.90) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    sim_names = [str(x) for x in d["body_names"]]
    if sim_names == COMMON_BODY_NAMES:
        raise AssertionError(
            "rollout body_names already equal the COMMON order — the remap this "
            "script exists for has become a no-op. Verify before trusting output."
        )
    clip = str(d["motion_name"])
    ref_path = motion_dir / f"{clip}.motion"
    if not ref_path.is_file():
        return {"clip": clip, "status": f"no reference at {ref_path}"}
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    if ref.get("ground_reaction_valid") is None:
        return {"clip": clip, "status": "reference has no measured pressure"}

    ref_fz = np.maximum(ref["rigid_body_ground_forces"].numpy()[:, :, 2], 0.0)
    ref_bp = ref["rigid_body_pos"].numpy()
    gv = ref["ground_reaction_valid"].numpy()
    n_cols = gv.shape[1]

    n_sub = d["pair_force_w"].shape[0]
    sub = np.clip(d["ctrl_substep_index"].astype(int), 0, n_sub - 1)
    t = d["ctrl_motion_time"].astype(float)
    fz_all = to_common_order(np.maximum(d["pair_force_w"][:, :, 0, 2], 0.0), sim_names)
    fz = fz_all[sub]
    fi = np.clip(np.round(t * 60).astype(int), 0, len(ref_fz) - 1)

    hands = fz[:, ZID["HANDS"]].sum(1)
    hands_down = hands > ON_N
    other = fz.copy(); other[:, ZID["HANDS"]] = 0.0

    out = {"clip": clip, "status": "ok", "n_ctrl": int(len(t)),
           "checkpoint": str(d["checkpoint"]),
           "hands_down_pct": 100 * float(hands_down.mean())}

    # --- metric 1: non-hand ground load while the hands are down -------------
    if hands_down.any():
        nh = other[hands_down].sum(1)
        out["nonhand_load_mean_bw"] = float(nh.mean() / ROBOT_WEIGHT_N)
        out["nonhand_load_p90_bw"] = float(np.percentile(nh, 90) / ROBOT_WEIGHT_N)
        out["hands_only_pct"] = 100 * float(
            (hands_down & (other.sum(1) <= ON_N)).mean())
        duty = (other[hands_down] > ON_N).mean(0) * 100
        out["top_offenders"] = [
            {"body": COMMON_BODY_NAMES[i], "duty_pct": float(duty[i]),
             "mean_when_on_bw": float(
                 other[hands_down][:, i][other[hands_down][:, i] > ON_N].mean()
                 / ROBOT_WEIGHT_N)}
            for i in np.argsort(-duty)[:4] if duty[i] >= 1.0
        ]
        ref_nh = ref_fz[fi][hands_down].copy(); ref_nh[:, ZID["HANDS"]] = 0.0
        out["reference_nonhand_load_bw"] = float(ref_nh.sum(1).mean() / ROBOT_WEIGHT_N)

    # --- metric 2: zone-share TV on gated frames -----------------------------
    cov = gv[fi, 0]
    explained = gv[fi, 1] / np.maximum(cov, 1e-6)
    share_gate = (gv[fi, 2] >= valid_threshold) if n_cols >= 3 else \
        ((cov >= valid_threshold) & (explained >= valid_threshold))
    out["share_gate_column"] = 2 if n_cols >= 3 else "coverage (legacy 2-col)"
    body_gate = (gv[fi, 1] >= valid_threshold * valid_threshold)
    s_sim, tot_sim = zone_shares(fz)
    s_ref, tot_ref = zone_shares(ref_fz[fi])
    live = share_gate & (tot_sim >= 30.0) & (tot_ref >= 30.0)
    out["share_gate_pct"] = 100 * float(live.mean())
    if live.sum() >= 20:
        tv = 0.5 * np.abs(s_sim[live] - s_ref[live]).sum(-1)
        out["zone_share_tv"] = float(tv.mean())
        out["zone_share_rew"] = float(np.exp(-3.0 * tv).mean())
        out["per_zone_abs_diff"] = {k: float(v) for k, v in
                                    zip(ZK, np.abs(s_sim[live] - s_ref[live]).mean(0))}

    # --- metric 3: the option-B penalty, replayed ----------------------------
    empty = (s_ref < 0.02)
    zone_sim = np.stack([fz[..., ZID[k]].sum(-1) for k in ZK], -1)
    viol = (empty * zone_sim).sum(-1) / ROBOT_WEIGHT_N
    m = body_gate & (tot_ref >= 30.0)
    if m.sum() >= 20:
        out["unloaded_violation_mean_bw"] = float(viol[m].mean())
        out["unloaded_violation_frac_over_5pct"] = float((viol[m] > 0.05).mean())

    # --- hold phase (feet up in the reference) -------------------------------
    hold = ref_bp[fi][:, [COMMON_BODY_NAMES.index("L_Ankle"),
                          COMMON_BODY_NAMES.index("R_Ankle")], 2].min(1) > FEET_UP_M
    out["hold_pct"] = 100 * float(hold.mean())
    hl = hold & live
    if hl.sum() >= 20:
        tv = 0.5 * np.abs(s_sim[hl] - s_ref[hl]).sum(-1)
        out["hold_zone_share_tv"] = float(tv.mean())
        out["hold_feet_load_bw"] = float(
            fz[hl][:, ZID["FEET"]].sum(1).mean() / ROBOT_WEIGHT_N)
        out["hold_ref_feet_load_bw"] = float(
            ref_fz[fi][hl][:, ZID["FEET"]].sum(1).mean() / ROBOT_WEIGHT_N)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True,
                    help="Directory of <clip>/rollout.npz from record_contact_physics.py")
    ap.add_argument("--motion-dir",
                    default="data/smpl/yoga_motions_proto_yogi_pressure_gated")
    ap.add_argument("--valid-threshold", type=float, default=0.90)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rolls = sorted(Path(args.in_dir).glob("*/rollout.npz"))
    if not rolls:
        raise SystemExit(f"no rollout.npz under {args.in_dir}")
    rows = [score(p, Path(args.motion_dir), args.valid_threshold) for p in rolls]

    print(f"{'clip':48s} {'hands':>6s} {'only':>6s} {'nonhandBW':>10s} "
          f"{'refBW':>6s} {'TV':>6s} {'holdTV':>7s} {'violBW':>7s} {'gate%':>6s}")
    print("-" * 112)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['clip'][:48]:48s}  {r['status']}"); continue
        g = lambda k, d="   n/a": f"{r[k]:.3f}" if k in r else d  # noqa: E731
        print(f"{r['clip'][:48]:48s} {r['hands_down_pct']:6.1f} "
              f"{r.get('hands_only_pct', float('nan')):6.1f} "
              f"{g('nonhand_load_mean_bw'):>10s} {g('reference_nonhand_load_bw'):>6s} "
              f"{g('zone_share_tv'):>6s} {g('hold_zone_share_tv'):>7s} "
              f"{g('unloaded_violation_mean_bw'):>7s} {r['share_gate_pct']:6.1f}")
    print("\nPer-clip offenders (bodies carrying ground load while the hands are down):")
    for r in rows:
        if r.get("top_offenders"):
            s = "  ".join(f"{o['body']}={o['duty_pct']:.0f}%/{o['mean_when_on_bw']:.2f}BW"
                          for o in r["top_offenders"])
            print(f"  {r['clip'][:46]:46s} {s}")
    ok = [r for r in rows if r["status"] == "ok" and "zone_share_tv" in r]
    if ok:
        print(f"\nmean zone-share TV {np.mean([r['zone_share_tv'] for r in ok]):.3f}; "
              f"mean unloaded violation "
              f"{np.mean([r.get('unloaded_violation_mean_bw', np.nan) for r in ok]):.3f} BW")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
