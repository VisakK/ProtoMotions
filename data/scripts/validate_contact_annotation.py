# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check geometric contact labels against the corpus config database and pose expectations.

Three independent checks, because a label set that only agrees with itself proves
nothing:

1. **Against the corpus database.** ``data/smpl/yoga_contact_configs/*.json`` was
   built by the same detector but on a *separately grounded* copy of the corpus and
   was validated by 44 scripted assertions (``notes/Contact_config_def.MD`` s6).
   Its dominant segment's zone set is compared with the zones the new per-frame
   labels report during the clip's hold.
2. **Against domain expectations.** Hand-coded signatures for poses whose support
   set is not in dispute (tree stands on one foot, crow is on two hands, handstand
   is two hands and nothing else).
3. **Against the old heuristic**, to quantify what changed rather than assert it.

Usage::

    PYTHONPATH=. python data/scripts/validate_contact_annotation.py \
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \
      --new-dir data/smpl/yoga_yogi_balance_subset_contacts \
      --old-dir data/smpl/yoga_yogi_balance_subset_grounded \
      --configs-dir data/smpl/yoga_contact_configs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_contact_configs import ZONES, mjcf_body_names  # noqa: E402

BODY_TO_ZONE = {b: z for z, bodies in ZONES.items() for b in bodies}

# Support sets these poses must show while held, stated as GROUND contact --
# "the feet are off the floor in crow" is a claim about the floor, while the
# feet are still touching each other, and the label the reward uses means
# "touching anything".  Checking must_not against any-contact is what produced
# five spurious failures on the first pass.  Only poses whose ground-support
# signature is not in dispute (Contact_config_def.MD s6 or anatomically plain).
EXPECTED = {
    "220923_Tree_Pose_or_Vrksasana_-a": {
        "must": {"R_FOOT"}, "must_not": {"L_HAND", "R_HAND", "HEAD"},
        "note": "stands on one foot; raised foot rests on the opposite thigh",
    },
    "220923_Crane_Crow_Pose_or_Bakasana_hold": {
        "must": {"L_HAND", "R_HAND"}, "must_not": {"L_FOOT", "R_FOOT", "HEAD"},
        "note": "arm balance: hands on floor, shins on upper arms",
    },
    "220923_Crane_Crow_Pose_or_Bakasana_-a": {
        "must": {"L_HAND", "R_HAND"}, "must_not": {"HEAD"},
        "note": "same, with entry/exit",
    },
    "220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a": {
        "must": {"L_HAND", "R_HAND"}, "must_not": {"HEAD"},
        "note": "fully inverted on two hands",
    },
    "220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a": {
        "must": {"HEAD"}, "must_not": {"L_FOOT", "R_FOOT"},
        "note": "crown plus forearms bear the load",
    },
    "220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a": {
        "must": {"L_FOREARM", "R_FOREARM"}, "must_not": {"L_FOOT", "R_FOOT"},
        "note": "forearm balance",
    },
    "220926_Warrior_III_Pose_or_Virabhadrasana_III_-a": {
        "must_not": {"L_HAND", "R_HAND", "HEAD"},
        "note": "standing balance on one leg, hands free",
    },
    "220923_Half_Moon_Pose_or_Ardha_Chandrasana_-a": {
        "must_not": {"HEAD"},
        "note": "one foot and one hand on the floor",
    },
}


def load_split(path: Path):
    """(any, ground, body) masks from the annotator's sidecar, if present."""
    side = path.with_suffix("").with_suffix(".contacts.npz")
    if not side.exists():
        return None
    import numpy as np
    z = np.load(side, allow_pickle=False)
    return (torch.as_tensor(z["any"]), torch.as_tensor(z["ground"]),
            torch.as_tensor(z["body"]))


def zones_on(labels: torch.Tensor, body_names, frac: float = 0.6) -> set:
    """Zones whose bodies are in contact for at least ``frac`` of the given frames."""
    on = labels.float().mean(0)
    zones = set()
    for i, name in enumerate(body_names):
        if float(on[i]) >= frac:
            zones.add(BODY_TO_ZONE[name])
    return zones


def hold_slice(n: int) -> slice:
    return slice(int(n * 0.35), int(n * 0.75))


def dominant_config_zones(cfg_path: Path):
    """Zone set of the longest segment in a corpus contact-config JSON."""
    if not cfg_path.exists():
        return None, None
    data = json.loads(cfg_path.read_text())
    segs = [s for s in data.get("segments", []) if s.get("duration_s")]
    if not segs:
        return None, None
    best = max(segs, key=lambda s: s["duration_s"])
    zones = set()
    for pair in best["config"].split("@")[0].split("|"):
        if not pair:
            continue
        if pair.endswith(":G"):
            zones.add(pair[:-2])
        else:
            zones.update(pair.split("+"))
    return zones, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--configs-dir", required=True)
    a = ap.parse_args()

    body_names = mjcf_body_names(a.mjcf)
    new_dir, old_dir, cfg_dir = Path(a.new_dir), Path(a.old_dir), Path(a.configs_dir)
    failures, warnings_ = [], []

    for path in sorted(new_dir.glob("*.motion")):
        stem = path.stem
        new = torch.as_tensor(
            torch.load(path, map_location="cpu", weights_only=False)["rigid_body_contacts"],
            dtype=torch.bool,
        )
        old_p = old_dir / path.name
        old = None
        if old_p.exists():
            old = torch.as_tensor(
                torch.load(old_p, map_location="cpu", weights_only=False)["rigid_body_contacts"],
                dtype=torch.bool,
            )

        hs = hold_slice(new.shape[0])
        new_zones = zones_on(new[hs], body_names)
        old_zones = zones_on(old[hs], body_names) if old is not None else set()
        split = load_split(path)
        ground_zones = zones_on(split[1][hs], body_names) if split else None

        print(f"\n=== {stem}")
        print(f"  new labels, held phase : {sorted(new_zones) or '(none)'}")
        if ground_zones is not None:
            print(f"    of which ground      : {sorted(ground_zones) or '(none)'}")
        print(f"  old labels, held phase : {sorted(old_zones) or '(none)'}")

        corpus_zones, seg = dominant_config_zones(cfg_dir / f"{stem}.json")
        if corpus_zones is not None:
            shared = new_zones & corpus_zones
            print(f"  corpus dominant config : {sorted(corpus_zones)} "
                  f"({seg['duration_s']:.1f}s)")
            missed = corpus_zones - new_zones
            if missed:
                warnings_.append(f"{stem}: corpus zones not in new labels: {sorted(missed)}")
                print(f"  ! corpus zones missing from new labels: {sorted(missed)}")
            else:
                print(f"  OK corpus support set fully covered ({len(shared)} zones)")

        exp = EXPECTED.get(stem)
        check_zones = ground_zones if ground_zones is not None else new_zones
        if exp:
            for z in exp.get("must", set()):
                if z not in check_zones:
                    failures.append(f"{stem}: expected {z} on the ground ({exp['note']})")
                    print(f"  FAIL expected {z} in ground contact")
            for z in exp.get("must_not", set()):
                if z in check_zones:
                    failures.append(f"{stem}: {z} should NOT be on the ground ({exp['note']})")
                    print(f"  FAIL {z} should not be in ground contact")
            if not any(f.startswith(stem) for f in failures):
                print(f"  OK pose expectation: {exp['note']}")

        if old is not None:
            inter = (old & new).float().sum()
            prec = float(inter / old.float().sum().clamp_min(1))
            rec = float(inter / new.float().sum().clamp_min(1))
            print(f"  old-vs-new: old precision {prec*100:.1f}%, "
                  f"old covers {rec*100:.1f}% of new contacts")

    print("\n" + "=" * 70)
    if warnings_:
        print(f"{len(warnings_)} warning(s):")
        for w in warnings_:
            print("  -", w)
    if failures:
        print(f"{len(failures)} FAILURE(s):")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("all pose expectations satisfied")


if __name__ == "__main__":
    main()
