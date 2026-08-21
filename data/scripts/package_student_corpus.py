# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package the three Stage-1 expert domains into one Stage-2 student corpus.

The MaskedMimic student is distilled from *three* trackers, each trained on a
different slice of the yoga corpus:

    easy-128      smpl_yogi_easy128_contact_rich        (non-balance poses)
    hard-29       smpl_yogi_hard29_pressure_ab_s1       (inversions + arm balances)
    single-leg 14 smpl_yogi_singleleg14_pressure_ab_s1  (standing balances)

Distillation needs one motion library covering all three, plus a map from motion
id to the expert that owns that clip -- the student rolls out on a clip and the
*owning* expert labels the action.  This writes both:

* ``<out>.pt`` / ``<out>.yaml``  -- the packaged 171-clip library and its manifest,
  built and verified by :mod:`package_motion_subset`;
* ``<out>.experts.json``         -- ``{"experts": [...], "motion_expert": [...]}``,
  indexed by motion id in the packaged library's own order.

Two things this refuses to guess at:

* **Clip order.**  ``MotionLib`` stores ``motion_files`` in manifest order, but
  the expert map is only correct if it is derived from the *packaged* file rather
  than from the input list, so the map is built by re-reading the ``.pt``.
* **Pressure.**  ``MotionLib`` packs the measured MOYO ground reaction
  all-or-nothing, and 128 of the 171 clips have none, so the packaged library has
  no pressure channel at all.  That is correct here (no student or expert
  observation reads it -- the pressure terms were *rewards* on the Stage-1 runs)
  but it has to be passed explicitly, so ``--allow-mixed-pressure`` is forwarded
  and the outcome asserted rather than silently accepted.

Usage::

    PYTHONPATH=. python data/scripts/package_student_corpus.py \
      --out data/smpl/yoga_yogi_student171.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# (expert name, checkpoint, source manifest). Order defines the expert index.
EXPERT_GROUPS = [
    (
        "easy128",
        "results/smpl_yogi_easy128_contact_rich/last.ckpt",
        "data/smpl/yoga_yogi_easy128_grounded.yaml",
    ),
    (
        "hard29",
        "results/smpl_yogi_hard29_pressure_ab_s1/last.ckpt",
        "data/smpl/yoga_yogi_hard29_pressure.yaml",
    ),
    (
        "singleleg14",
        "results/smpl_yogi_singleleg14_pressure_ab_s1/last.ckpt",
        "data/smpl/yoga_yogi_singleleg14_pressure.yaml",
    ),
]


def read_manifest(manifest: Path) -> list[Path]:
    """Absolute ``.motion`` paths listed by a MotionLib yaml manifest."""
    doc = yaml.safe_load(manifest.read_text())
    return [(manifest.parent / entry["file"]).resolve() for entry in doc["motions"]]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None,
                        help="output .pt (default depends on --subset)")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="named subset from select_student_subset.py (e.g. student44). "
             "Omit for the full 171-clip union of the three expert domains.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def collect_full() -> tuple[list[Path], list[int], list[float]]:
    """Every clip of all three expert domains, at weight 1.0."""
    motions: list[Path] = []
    owner: list[int] = []
    for index, (name, _checkpoint, manifest) in enumerate(EXPERT_GROUPS):
        group = read_manifest(REPO_ROOT / manifest)
        print(f"{name:12s} {len(group):3d} clips  <- {manifest}")
        motions.extend(group)
        owner.extend([index] * len(group))
    return motions, owner, [1.0] * len(motions)


def collect_subset(subset_name: str) -> tuple[list[Path], list[int], list[float]]:
    """One named subset, with its sampling weights.

    The subset module owns the clip lists and the weights together, because
    choosing which clips to train on and choosing how often to visit them are
    the same decision -- see its docstring for the measured exposure numbers.
    """
    sys.path.insert(0, str(REPO_ROOT / "data/scripts"))
    from select_student_subset import SUBSETS, resolve  # noqa: E402

    if subset_name not in SUBSETS:
        raise SystemExit(
            f"unknown subset '{subset_name}'; known: {sorted(SUBSETS)}"
        )
    subset = SUBSETS[subset_name]
    # The routing table indexes experts by their position in EXPERT_GROUPS, so a
    # subset that named them in a different order would mis-route every label.
    expert_index = {name: i for i, (name, _c, _m) in enumerate(EXPERT_GROUPS)}
    unknown = set(subset.clips) - set(expert_index)
    if unknown:
        raise SystemExit(
            f"subset '{subset_name}' names experts this packager does not build: "
            f"{sorted(unknown)}"
        )

    resolved = resolve(subset)
    motions = [path for _e, _s, path, _w in resolved]
    owner = [expert_index[expert] for expert, _s, _p, _w in resolved]
    weights = [weight for *_, weight in resolved]
    for index, (name, _c, _m) in enumerate(EXPERT_GROUPS):
        count = owner.count(index)
        if count:
            group_weights = {w for o, w in zip(owner, weights) if o == index}
            shown = ", ".join(f"{w:g}" for w in sorted(group_weights))
            print(f"{name:12s} {count:3d} clips  weight {shown}")
    return motions, owner, weights


def main() -> int:
    args = create_parser().parse_args()
    default_out = (
        f"data/smpl/yoga_yogi_{args.subset}.pt"
        if args.subset
        else "data/smpl/yoga_yogi_student171.pt"
    )
    raw_out = args.out or default_out
    out = (REPO_ROOT / raw_out).resolve() if not Path(raw_out).is_absolute() else Path(raw_out)

    for name, checkpoint, manifest in EXPERT_GROUPS:
        if not (REPO_ROOT / manifest).is_file():
            raise SystemExit(f"missing manifest for {name}: {REPO_ROOT / manifest}")
        if not (REPO_ROOT / checkpoint).is_file():
            raise SystemExit(f"missing expert checkpoint for {name}: {checkpoint}")

    if args.subset:
        motions, owner, weights = collect_subset(args.subset)
    else:
        motions, owner, weights = collect_full()

    missing = [p for p in motions if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips do not exist, e.g. {missing[:2]}")

    stems = [p.stem for p in motions]
    duplicates = {s for s in stems if stems.count(s) > 1}
    if duplicates:
        raise SystemExit(
            f"the three groups are not disjoint: {sorted(duplicates)[:5]} "
            f"({len(duplicates)} clips appear in more than one group)"
        )
    print(f"\ntotal {len(motions)} clips, all distinct")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "data/scripts/package_motion_subset.py"),
        "--out",
        str(out),
        "--allow-mixed-pressure",
    ]
    if args.force:
        cmd.append("--force")
    cmd.extend(str(p) for p in motions)
    # --weights goes AFTER the positionals, and that is not cosmetic: `motions`
    # is nargs="+" and `--weights` is a float nargs="*", so putting the weights
    # first makes argparse swallow the first .motion path as a weight and die on
    # "invalid float value".
    if any(w != 1.0 for w in weights):
        cmd.append("--weights")
        cmd.extend(f"{w:g}" for w in weights)
    print(f"\n$ {' '.join(cmd[:6])} ... ({len(motions)} motions)\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        return result.returncode

    # Build the expert map from the PACKAGED file, not from the input list: the
    # map is only meaningful in the library's own motion-id order.
    import torch

    packaged = torch.load(out, map_location="cpu", weights_only=False)
    packaged_stems = [Path(f).stem for f in packaged["motion_files"]]
    if packaged_stems != stems:
        raise SystemExit(
            "packaged motion order does not match the requested order; refusing "
            "to write an expert map that would label the wrong clips"
        )
    if packaged.get("grc") is not None:
        raise SystemExit(
            "packaged library unexpectedly carries the measured ground reaction; "
            "the expert observation contract does not include it -- investigate"
        )

    side = out.with_suffix(".experts.json")
    payload = {
        "motion_file": str(out.relative_to(REPO_ROOT)),
        "experts": [
            {"name": name, "checkpoint": checkpoint, "num_motions": owner.count(i)}
            for i, (name, checkpoint, _) in enumerate(EXPERT_GROUPS)
        ],
        "motion_expert": owner,
        "motion_names": stems,
    }
    side.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {side}")
    for i, entry in enumerate(payload["experts"]):
        print(f"  expert {i} {entry['name']:12s} {entry['num_motions']:3d} clips  {entry['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
