# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the inversion and arm-balance clips — the complement of the easy-128
set — and emit the file lists to package with measured MOYO pressure attached.

``notes/Easy_128_experiment.MD`` split the 179-clip yoga corpus into 128 "easy"
clips and 51 excluded ones: 14 inversions, 16 arm balances, 14 single-leg
standing balances, and 7 synthetic duplicates.  This script reproduces the first
two groups explicitly, *cross-checks them against the shipped easy-128 manifest*
(so the two sets provably partition the corpus), and reports which clips carry
the measured ground reaction.

Why the cross-check and the trap assertions exist
-------------------------------------------------
Naive substring matching gets two clips wrong, and both are in the easy set:

* ``Janu Sirsasana`` (Head-to-Knee Forward Bend, seated) matches ``Sirsasana``
  but is not ``Salamba Sirsasana`` (headstand).
* ``Dolphin Pose (Ardha Pincha Mayurasana)`` matches ``Pincha Mayurasana`` but
  is forearms-and-feet-down, not the forearm balance.

Pressure caveat
---------------
``220923_Crane_Crow_Pose_or_Bakasana_hold`` is an arm balance with **no usable
pressure capture** (it is a hand-trimmed subclip whose frame offset into the
parent was never recovered).  ``MotionLib`` packs the ground reaction
all-or-nothing, so including it would silently strip the measured channel from
every other clip.  It is therefore reported and excluded from the pressure
packages — arm balances go 16 -> 15.

Usage::

    PYTHONPATH=. python data/scripts/select_hard_pose_subsets.py
    PYTHONPATH=. python data/scripts/select_hard_pose_subsets.py --group inversions --print-paths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# The two groups, by pose family.  Names are clip stems in the 179-clip corpus.
# --------------------------------------------------------------------------- #
INVERSIONS = {
    "Pincha Mayurasana (forearm balance)": [
        "220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a",
        "220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-b",
        "220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-c",
        "220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-d",
    ],
    "Adho Mukha Vrksasana (handstand)": [
        "220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a",
    ],
    "Viparita Karani (legs up the wall)": [
        "220923_Legs-Up-the-Wall_Pose_or_Viparita_Karani_-a",
    ],
    "Halasana (plow)": [
        "220923_Plow_Pose_or_Halasana_-a",
        "220926_Plow_Pose_or_Halasana_-b",
    ],
    "Vrischikasana (scorpion)": [
        "220923_Scorpion_pose_or_vrischikasana-a",
        "220923_Scorpion_pose_or_vrischikasana-b",
        "220923_Scorpion_pose_or_vrischikasana-c",
    ],
    "Salamba Sirsasana (supported headstand)": [
        "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b",
        "220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a",
    ],
    "Salamba Sarvangasana (supported shoulderstand)": [
        "220923_Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_-a",
    ],
}

ARM_BALANCES = {
    "Kukkutasana (cockerel)": [
        "220923_Cockerel_Pose-b",
    ],
    "Bakasana (crow)": [
        "220923_Crane_Crow_Pose_or_Bakasana_-a",
        "220923_Crane_Crow_Pose_or_Bakasana_-b",
        "220923_Crane_Crow_Pose_or_Bakasana_hold",
    ],
    "Tittibhasana (firefly)": [
        "220923_Firefly_Pose_or_Tittibhasana_-a",
        "220923_Firefly_Pose_or_Tittibhasana_-b",
    ],
    "Mayurasana (peacock)": [
        "220923_Peacock_Pose_or_Mayurasana_-a",
        "220923_Peacock_Pose_or_Mayurasana_-b",
    ],
    "Eka Pada Koundinyasana": [
        "220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a",
        "220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b",
    ],
    "Tolasana (scale)": [
        "220923_Scale_Pose_or_Tolasana_-a",
    ],
    "Bhujapidasana (shoulder-pressing)": [
        "220923_Shoulder-Pressing_Pose_or_Bhujapidasana_-a",
    ],
    "Parsva Bakasana (side crow)": [
        "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a",
        "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-b",
        "220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c",
    ],
    "Astavakrasana (eight-angle)": [
        "220926_Eight-Angle_Pose_or_Astavakrasana_-a",
    ],
}

# Excluded from easy-128 as a deliberate judgement call rather than by the
# literal "no inversions or arm balances" rule (Easy_128_experiment.MD 1), so
# they were never trained on anything. Selectable as their own group.
SINGLE_LEG_BALANCES = [
    "220923_Half_Moon_Pose_or_Ardha_Chandrasana_-a",
    "220923_Half_Moon_Pose_or_Ardha_Chandrasana_-b",
    "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a",
    "220923_Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-b",
    "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-a",
    "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-b",
    "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c",
    "220923_Tree_Pose_or_Vrksasana_-a",
    "220923_Tree_Pose_or_Vrksasana_-b",
    "220926_Eagle_Pose_or_Garudasana_-a",
    "220926_Lord_of_the_Dance_Pose_or_Natarajasana_-a",
    "220926_Lord_of_the_Dance_Pose_or_Natarajasana_-c",
    "220926_Warrior_III_Pose_or_Virabhadrasana_III_-a",
    "220926_Warrior_III_Pose_or_Virabhadrasana_III_-b",
]
SYNTHETIC = [
    "crow_pose", "kound_a_pose", "one_legged_crow_slow", "one_legged_crow_synth",
    "pincha_synth", "scorpion_pose", "yoga_pose",
]

# Kept in easy-128 on purpose despite matching an inversion/arm-balance substring.
SUBSTRING_TRAPS = {
    "Janu_Sirsasana": "Head-to-Knee Forward Bend (seated), not Salamba Sirsasana",
    "Dolphin_Pose_or_Ardha_Pincha_Mayurasana": (
        "forearms AND feet down, not the Pincha Mayurasana forearm balance"
    ),
}

CORPUS_DIR = "data/smpl/yoga_motions_proto_yogi_grounded_yogaonly"
PRESSURE_DIR = "data/smpl/yoga_motions_proto_yogi_pressure"
EASY128_YAML = "data/smpl/yoga_yogi_easy128_grounded.yaml"


def flat(groups: dict) -> list[str]:
    return [c for v in groups.values() for c in v]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default=CORPUS_DIR)
    ap.add_argument("--pressure-dir", default=PRESSURE_DIR)
    ap.add_argument("--easy128-yaml", default=EASY128_YAML)
    ap.add_argument("--group",
                    choices=["inversions", "arm_balances", "both", "single_leg"],
                    default="both")
    ap.add_argument("--require-pressure", action="store_true", default=True)
    ap.add_argument("--no-require-pressure", dest="require_pressure",
                    action="store_false")
    ap.add_argument("--print-paths", action="store_true",
                    help="Print only the .motion paths, for use in a package command.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    corpus = sorted(p.stem for p in Path(args.corpus_dir).glob("*.motion"))
    corpus_set = set(corpus)
    inv, arm = flat(INVERSIONS), flat(ARM_BALANCES)

    # --- structural checks --------------------------------------------------
    unknown = [c for c in inv + arm if c not in corpus_set]
    assert not unknown, f"clips not in the corpus: {unknown}"
    assert len(inv) == 14, f"expected 14 inversions, got {len(inv)}"
    assert len(arm) == 16, f"expected 16 arm balances, got {len(arm)}"
    assert len(SINGLE_LEG_BALANCES) == 14, (
        f"expected 14 single-leg balances, got {len(SINGLE_LEG_BALANCES)}")
    assert not (set(inv) & set(arm)), "a clip is in both groups"
    assert not (set(SINGLE_LEG_BALANCES) & (set(inv) | set(arm))), (
        "a clip is in single-leg and in a hard group")
    missing_sl = [c for c in SINGLE_LEG_BALANCES if c not in corpus_set]
    assert not missing_sl, f"single-leg clips not in the corpus: {missing_sl}"

    import yaml
    easy = {Path(e["file"]).stem
            for e in yaml.safe_load(Path(args.easy128_yaml).read_text())["motions"]}
    assert len(easy) == 128, f"easy-128 manifest has {len(easy)} clips"
    overlap = (set(inv) | set(arm) | set(SINGLE_LEG_BALANCES)) & easy
    assert not overlap, f"clips claimed excluded but in easy-128: {sorted(overlap)}"
    partition = easy | set(inv) | set(arm) | set(SINGLE_LEG_BALANCES) | set(SYNTHETIC)
    leftover = corpus_set - partition
    assert not leftover, f"corpus clips in no group: {sorted(leftover)}"
    assert len(partition) == len(corpus_set) == 179, (
        f"partition covers {len(partition)} of {len(corpus_set)} clips"
    )

    # --- the two substring traps, asserted rather than trusted ---------------
    for needle, why in SUBSTRING_TRAPS.items():
        matched = [c for c in corpus if needle in c]
        assert matched, f"trap clip {needle} vanished from the corpus"
        wrong = [c for c in matched if c in inv or c in arm]
        assert not wrong, f"{wrong} wrongly classified as hard: {why}"
        assert all(c in easy for c in matched), (
            f"{needle} should be in easy-128 ({why})"
        )

    # --- pressure availability ---------------------------------------------
    pdir = Path(args.pressure_dir)
    has_pressure = {c: (pdir / f"{c}.motion").is_file()
                    for c in inv + arm + SINGLE_LEG_BALANCES}
    no_pressure = [c for c, v in has_pressure.items() if not v]

    sel = {"inversions": inv, "arm_balances": arm, "both": inv + arm,
           "single_leg": list(SINGLE_LEG_BALANCES)}[args.group]
    if args.require_pressure:
        sel = [c for c in sel if has_pressure[c]]
    paths = [str(pdir / f"{c}.motion") if args.require_pressure
             else str(Path(args.corpus_dir) / f"{c}.motion") for c in sel]

    if args.print_paths:
        print(" ".join(paths))
        return

    print(f"corpus {len(corpus)} clips; easy-128 {len(easy)}; "
          f"inversions {len(inv)}; arm balances {len(arm)}; "
          f"single-leg {len(SINGLE_LEG_BALANCES)}; synthetic {len(SYNTHETIC)} "
          f"-> partition closes at {len(partition)}")
    print(f"substring traps verified: {', '.join(SUBSTRING_TRAPS)}\n")

    for title, groups in (("INVERSIONS", INVERSIONS), ("ARM BALANCES", ARM_BALANCES)):
        print(f"### {title} ({len(flat(groups))})")
        for family, clips in groups.items():
            marks = ["" if has_pressure[c] else "   <-- NO PRESSURE" for c in clips]
            print(f"  {family}")
            for c, m in zip(clips, marks):
                print(f"    {c}{m}")
        print()

    if no_pressure:
        print(f"EXCLUDED from the pressure packages ({len(no_pressure)}): {no_pressure}")
        print("  MotionLib packs the ground reaction all-or-nothing; including these "
              "would strip it from every clip.\n")
    print(f"selected for --group {args.group}"
          f"{' (pressure only)' if args.require_pressure else ''}: {len(sel)} clips")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"inversions": inv, "arm_balances": arm,
             "has_pressure": has_pressure, "selected": sel, "paths": paths}, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
