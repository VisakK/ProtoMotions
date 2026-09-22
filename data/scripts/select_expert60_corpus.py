# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the 60-clip mixed corpus for the goal-conditioned Stage-1 expert.

``expert_revist/expert_revisit.MD`` §3.4: every single-leg standing balance (14),
every inversion (14), every arm balance (15 -- ``Bakasana_hold`` is a hand-trimmed
subclip of ``Bakasana_-a`` and is redundant once holds are extended), and 17
easy-128 clips chosen as the connective tissue the hard poses pass through:
the downdog / plank / chaturanga / cobra chain, the side-plank and forearm
supports, the forward fold and squat the arm balances enter from, and the
standing lunge family the single-leg balances leave from. Chair is excluded on
purpose: the SMPL thighs press together at >1 BW there with both hip actuators
saturated (``notes/Physics_insights.md`` §5).

The group lists are imported from :mod:`select_hard_pose_subsets`, which is the
script that partitioned the corpus for the previous experts and carries the two
substring traps (``Janu Sirsasana`` is seated; ``Dolphin Pose`` is not the
forearm balance). Every clip is resolved against the grounded yoga-only source
directory and must exist; the output is an explicit manifest, so the corpus is a
reviewed list rather than a rule.

Usage::

    PYTHONPATH=. python data/scripts/select_expert60_corpus.py \
      --out data/smpl/expert60/corpus.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_hard_pose_subsets import (  # noqa: E402
    ARM_BALANCES,
    CORPUS_DIR,
    INVERSIONS,
    SINGLE_LEG_BALANCES,
    flat,
)

# The connective clips. Variants were chosen to match the ones the student44
# corpus already tracks cleanly where that information exists (side plank
# a/b, downdog a/b), else the ``-a`` take.
CONNECTIVE = [
    "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
    "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-b",
    "220923_Plank_Pose_or_Kumbhakasana_-a",
    "220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a",
    "220923_Cobra_Pose_or_Bhujangasana_-a",
    "220923_Side_Plank_Pose_or_Vasisthasana_-a",
    "220923_Side_Plank_Pose_or_Vasisthasana_-b",
    "220923_Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a",
    "220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a",
    "220923_Standing_Forward_Bend_pose_or_Uttanasana_-a",
    "220923_Garland_Pose_or_Malasana_-a",
    "220923_Low_Lunge_pose_or_Anjaneyasana_-a",
    "220926_Warrior_II_Pose_or_Virabhadrasana_II_-a",
    "220923_Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-a",
    "220923_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-a",
    "220926_Upward_Plank_Pose_or_Purvottanasana_-a",
    "220923_Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a",
]

EXCLUDED = {
    # A hand-trimmed subclip of Bakasana_-a with no recoverable frame offset;
    # the extended-hold variants make it redundant.
    "220923_Crane_Crow_Pose_or_Bakasana_hold": "subclip of Bakasana_-a",
}


def family_of(stem: str) -> str:
    """``220923_Tree_Pose_or_Vrksasana_-a`` -> ``Tree_Pose_or_Vrksasana``."""
    body = stem.split("_", 1)[1] if stem[:6].isdigit() and "_" in stem else stem
    for suffix in ("_-a", "_-b", "_-c", "_-d", "_-e", "-a", "-b", "-c", "-d", "-e"):
        if body.endswith(suffix):
            return body[: -len(suffix)]
    return body


def build_corpus(corpus_dir: Path) -> list[dict]:
    groups = [
        ("single_leg", SINGLE_LEG_BALANCES),
        ("inversion", flat(INVERSIONS)),
        ("arm_balance", [s for s in flat(ARM_BALANCES) if s not in EXCLUDED]),
        ("connective", CONNECTIVE),
    ]
    entries, seen = [], set()
    for group, stems in groups:
        for stem in stems:
            if stem in seen:
                raise ValueError(f"{stem} listed twice")
            seen.add(stem)
            source = corpus_dir / f"{stem}.motion"
            if not source.exists():
                raise FileNotFoundError(f"{group}: {source} does not exist")
            entries.append(
                {
                    "stem": stem,
                    "group": group,
                    "family": family_of(stem),
                    "source": str(source.resolve()),
                }
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", default=CORPUS_DIR)
    parser.add_argument("--out", default="data/smpl/expert60/corpus.yaml")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = REPO_ROOT / corpus_dir
    entries = build_corpus(corpus_dir)

    counts = {}
    for entry in entries:
        counts[entry["group"]] = counts.get(entry["group"], 0) + 1
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        yaml.safe_dump(
            {
                "corpus_dir": str(corpus_dir),
                "excluded": EXCLUDED,
                "counts": counts,
                "motions": entries,
            },
            handle,
            sort_keys=False,
        )
    print(f"wrote {out}: {len(entries)} clips {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
