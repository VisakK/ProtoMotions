# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surgically repair foot-ground contact labels on hands-planted frames.

The geometric annotator (``annotate_contacts_geometric.py``) includes the
static-support model's PROMOTIONS in the ``ground`` mask: slow near-ground
zones are promoted to "supporting" (band 17.5 cm for feet).  During
feet-only standing (entry/exit) those promotions recover genuine contact
hidden by the SMPL float bias and must be kept.  But during hands-planted
frames of an arm balance the same model labels the hovering trailing foot
as grounded (crow ``L_FOOT:G`` 44.4 % dwell, of which only 12.3 % is the
strict geometric tier), and a symmetric contact-match reward then *pays*
the policy to keep the toe down (``notes/Contact_balance_reward_design.MD``
section 2.4).

The repair is deliberately surgical -- foot bodies only, hands-planted
frames only:

    hands_planted[t] = (L_Hand|L_Wrist grounded) & (R_Hand|R_Wrist grounded)

    on hands_planted frames, for b in {L_Ankle, L_Toe, R_Ankle, R_Toe}:
        ground[t, b] &= strict_zone[t] & (member_gap_b[t] < GAP_MAX)
    on all other frames: unchanged.

``strict_zone`` is the strict geometric tier of ``compute_active_pairs``
(calibrated hysteresis on surface distance, no promotions), and the
per-member gap gate resolves the zone down to the individual foot body.
Body-body bits (the real foot-foot / foot-shank contacts, 66-98 % dwell in
crow) and every non-foot body are untouched, and the repair only ever
*clears* bits, never sets them.

Outputs, per clip, into ``--out-dir``: a full copy of the ``.motion`` dict
with ``rigid_body_contacts`` rebuilt as ``body | repaired_ground``, and a
new ``<stem>.contacts.npz`` (any/ground/body/body_names) with the repaired
ground mask.

Usage::

    PYTHONPATH=. python data/scripts/repair_foot_contact_labels.py \
      --in-dir data/smpl/yoga_yogi_balance_subset_v2_contacts \
      --clips 220923_Crane_Crow_Pose_or_Bakasana_-a \
              220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a \
      --out-dir data/smpl/yoga_yogi_crow_pair_v3_contacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_contact_configs import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    ZONES,
    compute_active_pairs,
)

FOOT_ZONES = ("L_FOOT", "R_FOOT")  # members: [L_Ankle, L_Toe], [R_Ankle, R_Toe]
LEFT_HAND = ("L_Hand", "L_Wrist")
RIGHT_HAND = ("R_Hand", "R_Wrist")
GAP_MAX = 0.05  # a member body must be within 5 cm of the floor to keep its bit


def repair_clip(motion_path: Path, npz_path: Path, mjcf: str, gap_max: float):
    """Returns (motion, any, ground, body, stats) with the repaired masks."""
    npz = np.load(npz_path, allow_pickle=True)
    names = [str(b) for b in npz["body_names"]]
    col = {n: i for i, n in enumerate(names)}
    ground = npz["ground"].copy()
    body = npz["body"].copy()
    T = ground.shape[0]

    resolved = compute_active_pairs(motion_path, mjcf, DEFAULT_THRESHOLDS)
    if resolved is None:
        raise SystemExit(f"{motion_path}: not a motion dict")
    if resolved["T"] != T:
        raise SystemExit(f"{motion_path}: {resolved['T']} frames vs npz {T}")
    motion = resolved["motion"]
    stored = torch.as_tensor(motion["rigid_body_contacts"], dtype=torch.bool).numpy()
    if not np.array_equal(stored, npz["any"]):
        raise SystemExit(f"{motion_path}: .motion contacts disagree with sidecar 'any'")

    # Hands planted: BOTH hands have a grounded hand-or-wrist body.
    left = ground[:, col["L_Hand"]] | ground[:, col["L_Wrist"]]
    right = ground[:, col["R_Hand"]] | ground[:, col["R_Wrist"]]
    hands_planted = left & right

    repaired = ground.copy()
    stats = {"frames": T, "hands_planted_frames": int(hands_planted.sum()), "zones": {}}
    for zone in FOOT_ZONES:
        pair = f"{zone}:G"
        strict = np.asarray(resolved["strict"][pair], dtype=bool)  # [T]
        active = np.asarray(resolved["active"][pair], dtype=bool)
        gaps, members = resolved["clip"].member_gaps(pair)  # [C,T], [(body, None)]
        member_bodies = [a for a, _ in members]
        assert member_bodies == ZONES[zone], (zone, member_bodies)
        stats["zones"][zone] = {
            "active_dwell": float(active.mean()),
            "strict_dwell": float(strict.mean()),
        }
        for ci, b in enumerate(member_bodies):
            keep = strict & (gaps[ci].numpy() < gap_max)
            j = col[b]
            repaired[:, j] = np.where(hands_planted, ground[:, j] & keep, ground[:, j])

    # Invariants: only foot-body bits on hands-planted frames may change, and
    # bits are only ever cleared.
    foot_cols = [col[b] for z in FOOT_ZONES for b in ZONES[z]]
    delta = repaired != ground
    assert not delta[~hands_planted].any(), "non-hands-planted frame changed"
    non_foot = np.ones(len(names), dtype=bool)
    non_foot[foot_cols] = False
    assert not delta[:, non_foot].any(), "non-foot body changed"
    assert not (repaired & ~ground).any(), "repair set a bit it should only clear"

    any_mask = body | repaired
    stats["hands_planted"] = {
        n: {
            "before": float(ground[hands_planted, col[n]].mean()) if hands_planted.any() else 0.0,
            "after": float(repaired[hands_planted, col[n]].mean()) if hands_planted.any() else 0.0,
        }
        for z in FOOT_ZONES
        for n in ZONES[z]
    }
    return motion, any_mask, repaired, body, names, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--clips", nargs="+", required=True, help="Clip stems (no extension).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--mjcf",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
        ),
    )
    ap.add_argument("--gap-max", type=float, default=GAP_MAX)
    ap.add_argument("--report", default=None, help="Optional JSON stats path.")
    a = ap.parse_args()

    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for stem in a.clips:
        motion_path = in_dir / f"{stem}.motion"
        npz_path = in_dir / f"{stem}.contacts.npz"
        for p in (motion_path, npz_path):
            if not p.is_file():
                raise SystemExit(f"missing input: {p}")
        motion, any_mask, ground, body, names, stats = repair_clip(
            motion_path, npz_path, a.mjcf, a.gap_max
        )

        out_motion = dict(motion)
        out_motion["rigid_body_contacts"] = torch.from_numpy(any_mask).to(torch.bool)
        torch.save(out_motion, out_dir / motion_path.name)
        np.savez_compressed(
            out_dir / npz_path.name,
            any=any_mask, ground=ground, body=body,
            body_names=np.array(names),
        )
        reports[stem] = stats

        hp = stats["hands_planted_frames"]
        print(f"{stem}: {stats['frames']} frames, hands_planted {hp} "
              f"({100 * hp / stats['frames']:.1f}%)")
        for zone, zs in stats["zones"].items():
            print(f"  {zone}:G active {100 * zs['active_dwell']:.1f}% "
                  f"strict {100 * zs['strict_dwell']:.1f}%")
        for b, d in stats["hands_planted"].items():
            print(f"  {b:8s} ground dwell on hands-planted frames: "
                  f"{100 * d['before']:5.1f}% -> {100 * d['after']:5.1f}%")

    if a.report:
        rp = Path(a.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(
            {"in_dir": str(in_dir), "mjcf": a.mjcf, "gap_max": a.gap_max,
             "clips": reports}, indent=2))
        print(f"report -> {rp}")
    print(f"wrote {len(reports)} repaired clips to {out_dir}")


if __name__ == "__main__":
    main()
