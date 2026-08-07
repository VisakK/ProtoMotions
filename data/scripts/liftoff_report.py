# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How much of a clip the policy spends on each support state.

For an arm balance the question "did it reach the pose" reduces to: while the
hands are down, how many feet are still on the floor?  A single trailing toe is
the difference between a crow pose and a squat with the hands down, and no
aggregate reward or tracking error separates the two.

Reads ``rollout.npz`` files written by ``record_contact_physics.py`` and prints
one row per rollout.  Give it several checkpoints of the same run to see whether
the pose is being unlocked or the policy has settled.

Usage::

    python data/scripts/liftoff_report.py results/Contact_Physics_analysis_crow_pair/*/
    python data/scripts/liftoff_report.py --label-from-parent sweep/epoch_*/*/
    python data/scripts/liftoff_report.py --reference \
        data/smpl/yoga_yogi_balance_subset_v2_contacts/220923_Crane_Crow_Pose_or_Bakasana_-a.contacts.npz \
        results/.../220923_Crane_Crow_Pose_or_Bakasana_-a
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_physics_support import (  # noqa: E402
    active_pairs,
    contact_series,
    lean_series,
    load_rollout,
)

# Down-facing supports of an arm balance vs the parts that should leave the floor.
HANDS = ("L_Hand", "R_Hand", "L_Wrist", "R_Wrist")
FEET = ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")

HEADER = (
    f"{'run':<26}{'hands down':>11}{'hands only':>12}{'+1 foot':>9}"
    f"{'+2 feet':>9}{'no hands':>10}"
)
# ``limb→arm`` is a SUM of duties over every knee/hip-against-arm pair, so it can
# exceed 100%: it is total contact time across pairs, not a fraction of the clip.
LEAN_HEADER = (
    f"{'run':<26}{'COM→hand poly':>15}{'COM inside':>12}{'lean':>9}"
    f"{'foot load':>11}{'limb→arm Σ':>12}{'track err':>11}"
)


def breakdown(contact: np.ndarray, names: list[str]) -> tuple[float, float, float, float, float]:
    """(hands_down, hands_only, plus_one_foot, plus_more_feet, no_hands) as fractions."""
    hands = [i for i, n in enumerate(names) if n in HANDS]
    feet = [i for i, n in enumerate(names) if n in FEET]
    on_hands = contact[:, hands].any(1)
    n_feet = contact[:, feet].sum(1)
    total = max(len(contact), 1)
    return (
        float(on_hands.mean()),
        float((on_hands & (n_feet == 0)).mean()),
        float((on_hands & (n_feet == 1)).mean()),
        float((on_hands & (n_feet >= 2)).mean()),
        float((~on_hands).mean()),
    )


def limb_on_arm_duty(roll) -> float:
    """Total duty of knee/hip-against-upper-arm or forearm contacts."""
    duty = 0.0
    for pair in active_pairs(roll):
        if pair.is_ground:
            continue
        joined = f"{pair.body_name}|{pair.filter_name}"
        limb = any(k in joined for k in ("Knee", "Hip"))
        arm = any(k in joined for k in ("Shoulder", "Elbow"))
        if limb and arm:
            duty += float((pair.magnitude > 1.0).mean())
    return duty


def row(label: str, values, extra: str = "") -> str:
    hd, ho, h1, h2, nh = values
    return (
        f"{label:<26}{100 * hd:10.1f}%{100 * ho:11.1f}%{100 * h1:8.1f}%"
        f"{100 * h2:8.1f}%{100 * nh:9.1f}%{extra}"
    )


def lean_row(label: str, roll) -> str:
    """Whether the body is over its hands, and how much load the feet still take."""
    lean = lean_series(roll)
    down = lean.hands_down
    if not down.any():
        return f"{label:<26}{'(hands never down)':>15}"
    margin = lean.hand_margin[down]
    inside = float(np.nanmean(margin > 0))
    foot = np.nanmean(lean.foot_load[down])
    duty = sum(
        float((p.magnitude > 1.0).mean())
        for p in active_pairs(roll)
        if not p.is_ground
        and any(k in f"{p.body_name}|{p.filter_name}" for k in ("Knee", "Hip"))
        and any(k in f"{p.body_name}|{p.filter_name}" for k in ("Shoulder", "Elbow"))
    )
    return (
        f"{label:<26}{np.nanmean(margin):+14.3f}m{100 * inside:11.1f}%"
        f"{np.nanmean(lean.lean):+8.3f}m{100 * foot:10.1f}%{100 * duty:11.1f}%"
        f"{float(np.mean(roll.raw['ctrl_track_err'])):10.3f}m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", help="Folders holding a rollout.npz.")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="A .contacts.npz to print as the target row (uses its ground labels).",
    )
    parser.add_argument(
        "--label-from-parent",
        action="store_true",
        help="Label rows by the parent folder (e.g. the checkpoint) instead of the clip.",
    )
    args = parser.parse_args()

    loaded = []
    for target in args.rollouts:
        path = Path(target)
        roll = load_rollout(path)
        label = (path.parent.name if args.label_from_parent else path.name)[:26]
        loaded.append((label, roll))

    print(HEADER)
    if args.reference:
        ref = np.load(args.reference, allow_pickle=True)
        names = [str(b) for b in ref["body_names"]]
        print(row("reference (target)", breakdown(ref["ground"], names)))
    for label, roll in loaded:
        series = contact_series(roll)
        print(row(label, breakdown(series.ground_contact, roll.body_names)))

    print()
    print("Is the body over its hands?  (averaged over frames with a hand down)")
    print(LEAN_HEADER)
    for label, roll in loaded:
        print(lean_row(label, roll))


if __name__ == "__main__":
    main()
