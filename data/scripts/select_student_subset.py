# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named subsets of the Stage-2 student corpus, with sampling weights.

``package_student_corpus.py`` builds the full 171-clip union of the three
Stage-1 expert domains.  That was the right first experiment and it is the wrong
second one: the first student
(``notes/Student_distill_experiment1.MD``) plateaued at 0.60 success and, watched,
could not invert.  This module names smaller corpora built only from clips whose
owning expert tracks them cleanly, so the student is distilled from behaviour
that is worth copying.

**Why a subset alone is not enough, measured.**  Episodes sample a clip uniformly
by weight and then a *uniform time within it*, so at weight 1.0 each clip owns
``1/N`` of episodes regardless of length.  Scoring "what fraction of frames have
an **inverted** nearest goal" -- the thing the handstand failure is actually about:

===========================================  ==============
corpus                                       inverted goal
===========================================  ==============
full 171 clips, uniform                              4.2 %
``student44``, uniform                               5.3 %
``student44``, this module's weights                **11.7 %**
===========================================  ==============

The subset on its own buys 1.3x.  The weights buy 2.8x over the 171-clip run.
That is why the weights live here next to the clip list rather than being left at
1.0: choosing the clips and choosing how often they are visited are the same
decision, and only the second one moves the number that matters.

**Selection rule.** Every clip is one the Stage-1 expert that owns it tracks with
``good_fraction >= 0.98`` and max tracking error <= 0.52 m in the recorded
rollouts the contact graph was built from.  The clips the graph itself mostly
discards -- Supported Headstand ``-a`` (0.38), Scorpion ``-c`` (0.37), Cockerel
``-b`` (0.50) -- are absent by construction rather than by hand.

Usage::

    # the clip list, as absolute .motion paths, in packaging order
    PYTHONPATH=. python data/scripts/select_student_subset.py --subset student44 --print-paths

    # what it contains and what the weights do to the exposure numbers
    PYTHONPATH=. python data/scripts/select_student_subset.py --subset student44 --summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Manifest per expert, in the order that defines the expert index. Must stay in
# step with EXPERT_GROUPS in package_student_corpus.py, which is asserted there.
EXPERT_MANIFESTS = {
    "easy128": "data/smpl/yoga_yogi_easy128_grounded.yaml",
    "hard29": "data/smpl/yoga_yogi_hard29_pressure.yaml",
    "singleleg14": "data/smpl/yoga_yogi_singleleg14_pressure.yaml",
}


# --------------------------------------------------------------------------- #
# student44
# --------------------------------------------------------------------------- #

# Inversions and arm balances. Eight clips, and eight is all there is: the corpus
# holds exactly ONE handstand take, ONE supported shoulderstand and ONE Warrior II
# (below), so "handstands" plural is not available at any price. Supported
# Headstand `-b` is included and `-a` is not, because the expert tracks `-b` on
# 100 % of frames and `-a` on 38 %.
STUDENT44_HARD = [
    "220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a",
    "220923_Firefly_Pose_or_Tittibhasana_-a",
    "220923_Firefly_Pose_or_Tittibhasana_-b",
    "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b",
    "220923_Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_-a",
    "220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a",
    "220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b",
    "220923_Scorpion_pose_or_vrischikasana-b",
]

# All fourteen single-leg standing balances. The Stage-1 run holds every one of
# them without falling (`Singleleg14_pressure_experiment.MD` §3.3: 0 falls in 14
# videos, same standing foot as the human on 99.7 % of decisive frames), so there
# is no failing clip to drop. Eagle's trailing-foot load is a reward defect, not
# a tracking one.
STUDENT44_SINGLE_LEG = [
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

# Non-balance poses: plank family, downdog, chaturanga, warrior II, chair, cobra.
#
# "All variations of plank" resolves to nine clips across four distinct contact
# configurations, and all four are worth having: Plank and Side Plank are
# hands-down prone/side, Dolphin Plank is FOREARMS down (the only easy clip that
# is, and the natural stepping stone toward pincha and scorpion), and Upward
# Plank is hands-and-heels SUPINE, which nothing else in the subset reaches.
#
# Warrior II is one clip. Note it is only separable from a plain stand by its
# POSE -- both are `L_FOOT:G|R_FOOT:G@upright` -- so it must never be scored by
# its contact set (`Student_distill_experiment1.MD` §7.1 point 4).
STUDENT44_EASY = [
    "220923_Plank_Pose_or_Kumbhakasana_-a",
    "220923_Side_Plank_Pose_or_Vasisthasana_-a",
    "220923_Side_Plank_Pose_or_Vasisthasana_-b",
    "220923_Side_Plank_Pose_or_Vasisthasana_-c",
    "220923_Side_Plank_Pose_or_Vasisthasana_-d",
    "220923_Side_Plank_Pose_or_Vasisthasana_-e",
    "220923_Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a",
    "220926_Upward_Plank_Pose_or_Purvottanasana_-a",
    "220926_Upward_Plank_Pose_or_Purvottanasana_-b",
    "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
    "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-b",
    "220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-c",
    "220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a",
    "220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-b",
    "220926_Warrior_II_Pose_or_Virabhadrasana_II_-a",
    "220923_Chair_Pose_or_Utkatasana_-a",
    "220923_Chair_Pose_or_Utkatasana_-b",
    "220923_Chair_Pose_or_Utkatasana_-c",
    "220923_Cobra_Pose_or_Bhujangasana_-a",
    "220923_Cobra_Pose_or_Bhujangasana_-b",
    "220923_Cobra_Pose_or_Bhujangasana_-c",
    "220923_Cobra_Pose_or_Bhujangasana_-d",
]


@dataclass
class Subset:
    """A named corpus: which clips, whose expert owns them, how often to visit."""

    name: str
    description: str
    # expert name -> clip stems, in packaging order within that expert.
    clips: dict
    # expert name -> sampling weight applied to every clip it owns. May also
    # carry a "holds" entry for the frozen-reference hold clips.
    group_weights: dict
    # clip stem -> weight, overriding the group weight.
    clip_weights: dict = field(default_factory=dict)
    # manifest written by make_hold_motions.py; None means no hold clips.
    # Each hold is routed to the expert that owns its *source* clip, so the
    # frozen pose is stabilized by the tracker that demonstrated it.
    hold_manifest: str = None

    def weight_for(self, expert: str, stem: str) -> float:
        return float(self.clip_weights.get(stem, self.group_weights[expert]))


SUBSETS = {
    "student44": Subset(
        name="student44",
        description=(
            "44 clips the Stage-1 experts all track cleanly: 8 inversions/arm "
            "balances, 14 single-leg standing balances, 22 non-balance poses"
        ),
        clips={
            "easy128": STUDENT44_EASY,
            "hard29": STUDENT44_HARD,
            "singleleg14": STUDENT44_SINGLE_LEG,
        },
        # 3.0 on the hard group takes the inverted-nearest-goal share from 5.3 %
        # to 11.7 % while still leaving the easy poses 37 % of episodes. Pushing
        # further (hard 4.0, easy 0.7) reaches 15.3 % but starves the broad poses
        # to a quarter of episodes, which is how the first student lost skills it
        # already had. Re-measure with --summary if these lists change.
        group_weights={"easy128": 1.0, "hard29": 3.0, "singleleg14": 1.0},
    ),
    "student44h": Subset(
        name="student44h",
        description=(
            "student44 plus the frozen-reference HOLD clips: the corpus-level "
            "cure for the measured stay-command gap (median 0.27 s, p90 2.13 s "
            "in the clip schedule vs the 5-10 s a deployed hold needs)"
        ),
        clips={
            "easy128": STUDENT44_EASY,
            "hard29": STUDENT44_HARD,
            "singleleg14": STUDENT44_SINGLE_LEG,
        },
        # Holds at 0.5 give ~30 hold clips about a fifth of episodes without
        # starving the transition data the vinyasa failure needs. Re-measure
        # with --summary after regenerating the hold set.
        group_weights={
            "easy128": 1.0,
            "hard29": 3.0,
            "singleleg14": 1.0,
            "holds": 0.5,
        },
        hold_manifest="data/smpl/yoga_motions_proto_yogi_holds.manifest.json",
    ),
}


def read_manifest_stems(expert: str) -> dict:
    """``stem -> absolute .motion path`` for one expert's manifest."""
    manifest = REPO_ROOT / EXPERT_MANIFESTS[expert]
    if not manifest.is_file():
        raise SystemExit(f"missing manifest for {expert}: {manifest}")
    doc = yaml.safe_load(manifest.read_text())
    out = {}
    for entry in doc["motions"]:
        path = (manifest.parent / entry["file"]).resolve()
        out[path.stem] = path
    return out


def resolve(subset: Subset) -> list:
    """``[(expert, stem, path, weight), ...]`` in packaging order.

    Every clip is looked up in its declared expert's manifest rather than
    searched for, so naming a clip under the wrong expert is an error here and
    not a silently mis-routed action label in the middle of a training run.
    """
    resolved, seen = [], set()
    for expert in EXPERT_MANIFESTS:  # fixed order == the expert index order
        stems = subset.clips.get(expert, [])
        if not stems:
            continue
        available = read_manifest_stems(expert)
        missing = [s for s in stems if s not in available]
        if missing:
            raise SystemExit(
                f"subset '{subset.name}': {len(missing)} clips are not in the "
                f"{expert} manifest, e.g. {missing[:3]}. A clip listed under the "
                f"wrong expert would be labelled by a tracker that never trained "
                f"on it."
            )
        for stem in stems:
            if stem in seen:
                raise SystemExit(f"subset '{subset.name}': '{stem}' listed twice")
            seen.add(stem)
            resolved.append((expert, stem, available[stem], subset.weight_for(expert, stem)))

    unknown = set(subset.clips) - set(EXPERT_MANIFESTS)
    if unknown:
        raise SystemExit(f"subset '{subset.name}': unknown experts {sorted(unknown)}")
    resolved.extend(resolve_holds(subset, seen))
    return resolved


def resolve_holds(subset: Subset, seen: set) -> list:
    """Hold clips, each routed to the expert that owns its source clip.

    The routing is re-derived here from the expert manifests rather than
    trusted from the hold manifest, so a hold cut from a clip an expert never
    trained on is an error at packaging time, not a silently mis-routed label.
    """
    if subset.hold_manifest is None:
        return []
    manifest_path = REPO_ROOT / subset.hold_manifest
    if not manifest_path.is_file():
        raise SystemExit(
            f"subset '{subset.name}' names hold manifest {manifest_path}, which "
            f"does not exist; run make_hold_motions.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    owner_of = {}
    for expert in EXPERT_MANIFESTS:
        for stem in read_manifest_stems(expert):
            owner_of[stem] = expert

    resolved = []
    for entry in manifest["holds"]:
        stem = entry["stem"]
        if stem in seen:
            raise SystemExit(f"subset '{subset.name}': '{stem}' listed twice")
        seen.add(stem)
        expert = owner_of.get(entry["source_stem"])
        if expert is None:
            raise SystemExit(
                f"hold '{stem}' was cut from '{entry['source_stem']}', which no "
                f"expert manifest contains"
            )
        path = (manifest_path.parent / entry["file"]).resolve()
        if not path.is_file():
            raise SystemExit(f"hold clip missing on disk: {path}")
        weight = float(
            subset.clip_weights.get(stem, subset.group_weights["holds"])
        )
        resolved.append((expert, stem, path, weight))
    return resolved


def summarize(subset: Subset, resolved: list) -> None:
    """Print the composition and the exposure the weights actually buy."""
    total_weight = sum(w for *_, w in resolved)
    print(f"subset '{subset.name}': {len(resolved)} clips")
    print(f"  {subset.description}\n")
    holds = [r for r in resolved if r[1].startswith("hold_")]
    for expert in EXPERT_MANIFESTS:
        rows = [r for r in resolved if r[0] == expert and not r[1].startswith("hold_")]
        if not rows:
            continue
        share = sum(w for *_, w in rows) / total_weight
        print(
            f"  {expert:12s} {len(rows):3d} clips  weight {rows[0][3]:.1f}  "
            f"-> {share:5.1%} of episodes"
        )
    if holds:
        share = sum(w for *_, w in holds) / total_weight
        print(
            f"  {'holds':12s} {len(holds):3d} clips  weight {holds[0][3]:.1f}  "
            f"-> {share:5.1%} of episodes  (routed to their source experts)"
        )

    graph = REPO_ROOT / "data/smpl/yoga_contact_graph/contact_graph.json"
    if not graph.is_file():
        print("\n  (contact_graph.json absent; skipping the exposure measurement)")
        return

    import json

    import numpy as np

    clips = dict(json.loads(graph.read_text())["clips"])
    weighted_inverted = 0.0
    covered = 0
    for _expert, stem, _path, weight in resolved:
        clip = clips.get(stem)
        if clip is None:
            continue
        covered += 1
        trusted = sorted(
            (s for s in clip["segments"] if s.get("trusted")), key=lambda s: s["t_hold"]
        )
        if not trusted:
            continue
        holds = np.array([s["t_hold"] for s in trusted])
        bins = [s["orientation_bin"] for s in trusted]
        length = max(s["t_end"] for s in clip["segments"])
        times = np.arange(0.0, length, 1.0 / 30.0)
        index = np.searchsorted(holds, times + 0.2)
        live = index < len(holds)
        if not live.any():
            continue
        share = float(np.mean([bins[i] == "inverted" for i in index[live]]))
        weighted_inverted += weight * share
    print(
        f"\n  frames whose nearest goal is INVERTED: "
        f"{weighted_inverted / total_weight:.1%} "
        f"(uniform over the full 171-clip corpus: 4.2 %)"
    )
    if covered != len(resolved):
        print(
            f"  note: {len(resolved) - covered} clips are absent from the existing "
            f"graph, so the number above is over the {covered} that are present"
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subset", type=str, default="student44", choices=sorted(SUBSETS))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--print-paths", action="store_true", help=".motion paths, one per line")
    group.add_argument("--print-weights", action="store_true", help="weights, one per line")
    group.add_argument("--print-stems", action="store_true", help="clip names, one per line")
    group.add_argument("--summary", action="store_true", help="composition and exposure")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    subset = SUBSETS[args.subset]
    resolved = resolve(subset)

    if args.print_paths:
        print("\n".join(str(p) for _e, _s, p, _w in resolved))
    elif args.print_weights:
        print("\n".join(f"{w:g}" for *_, w in resolved))
    elif args.print_stems:
        print("\n".join(s for _e, s, _p, _w in resolved))
    else:
        summarize(subset, resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
