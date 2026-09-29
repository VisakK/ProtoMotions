# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a hold manifest for a changed corpus from already-reviewed pieces.

Fine-tune C (``expert_revist/ft_c/README.MD``) keeps 56 of expert60's clips with their
reviewed and repaired holds untouched, drops four whose reference fails the physics audit,
and adds four easy-128 connectives whose holds were proposed and repaired with the same
tools and thresholds. This script only *combines*: every clip's hold list is copied
verbatim from the manifest it comes from, and the output follows the corpus manifest's
clip order so the packaged library and the graph line up with it.

    PYTHONPATH=. python data/scripts/merge_hold_manifests.py \\
      --corpus data/smpl/expert60/corpus_ftC.yaml \\
      --manifests data/smpl/expert60/holds_repaired.yaml data/smpl/expert60/holds_ftC_new4_repaired.yaml \\
      --out data/smpl/expert60/holds_repaired_ftC.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def merge(corpus: dict, manifests: list[dict]) -> dict:
    """Clips in corpus order, each taken from the first manifest that has it."""
    by_stem = {}
    for manifest in manifests:
        for clip in manifest["clips"]:
            by_stem.setdefault(clip["stem"], clip)
    stems = [m["stem"] for m in corpus["motions"]]
    missing = [s for s in stems if s not in by_stem]
    if missing:
        raise ValueError(f"no manifest has holds for {missing}")
    clips = []
    for entry in corpus["motions"]:
        clip = dict(by_stem[entry["stem"]])
        if clip.get("group") != entry["group"]:
            raise ValueError(f"{entry['stem']}: group {clip.get('group')} != corpus {entry['group']}")
        clips.append(clip)
    out = {k: v for k, v in manifests[0].items() if k != "clips"}
    out["clips"] = clips
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    corpus = yaml.safe_load(open(args.corpus))
    manifests = [yaml.safe_load(open(p)) for p in args.manifests]
    merged = merge(corpus, manifests)
    merged["merged_from"] = {"corpus": str(Path(args.corpus).resolve()),
                             "manifests": [str(Path(p).resolve()) for p in args.manifests]}
    with open(args.out, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    counts = {}
    for c in merged["clips"]:
        counts[c["group"]] = counts.get(c["group"], 0) + 1
    print(f"wrote {args.out}: {len(merged['clips'])} clips {counts}, "
          f"{sum(len(c['holds']) for c in merged['clips'])} holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
