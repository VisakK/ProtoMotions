# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a clip's ``rigid_body_contacts`` using geometric contact detection.

The labels shipped in the motion files come from
``contact_detection.compute_contact_labels_from_pos_and_vel`` -- joint-centre
height < 8 cm and speed < 0.1 m/s.  That heuristic audits at 0-16 % precision on
this corpus (``notes/Contact_obs_baseline.MD``), has no notion of body-body
contact at all, and cannot see a load-bearing part that the SMPL fit floats
(a headstand head sits 13.5-14.2 cm off the floor).

This replaces them with the validated geometric pipeline from
``extract_contact_configs.py``: exact surface distances between typed collision
geoms, per-zone calibrated hysteresis, a stillness-gated tier for compressed
body-body supports, and the COM/support-polygon static model that recovers
float-biased supports.  See ``notes/Contact_config_def.MD``.

Two semantics matter for a contact-matching reward:

* **Every body, not just feet.** In yoga the load-bearing contacts are routinely
  hands, forearms, head, knees or trunk.
* **Contact with anything.** The simulator's ``rigid_body_contacts`` is "net
  contact force on this body exceeds a threshold", which self-collision makes
  true for body-body contact too.  The reference labels therefore mark a body if
  it touches the ground **or** another body, so both sides of the reward mean the
  same thing.

Zone activity is resolved down to individual bodies by proximity: when a zone
pair is active, every member body whose own gap is within ``--patch-slack`` of
the zone's minimum is marked.  That is the same 1 cm contact-patch convention
``pair_series`` already uses to span a flat foot's ankle and toe boxes.

Usage::

    PYTHONPATH=. python data/scripts/annotate_contacts_geometric.py \
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \
      --in-dir data/smpl/yoga_yogi_balance_subset_grounded \
      --out-dir data/smpl/yoga_yogi_balance_subset_contacts \
      --report results/contact_annotation/report.json
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
    mjcf_body_names,
)


def per_body_contacts(resolved, patch_slack: float):
    """Per-body contact masks, split by what the body is touching.

    Returns ``(any_contact, ground_only, body_only)``, each ``[T, B]`` bool.
    ``any_contact`` is what the reward uses -- it matches the simulator's
    "net contact force on this body" semantics.  The split is kept for
    diagnostics: a pose expectation like "the feet are off the floor in crow"
    is a statement about *ground* contact, while the feet are still touching
    each other, and conflating the two makes validation meaningless.
    """
    clip = resolved["clip"]
    body_names = resolved["body_names"]
    T = resolved["T"]
    index = {n: i for i, n in enumerate(body_names)}
    ground = torch.zeros(T, len(body_names), dtype=torch.bool)
    body = torch.zeros(T, len(body_names), dtype=torch.bool)

    for pair, active in resolved["active"].items():
        active_t = torch.as_tensor(np.asarray(active), dtype=torch.bool)
        if not bool(active_t.any()):
            continue
        gaps, members = clip.member_gaps(pair)          # [C,T], [(a,b)]
        near = gaps <= (gaps.min(dim=0).values.unsqueeze(0) + patch_slack)
        touching = near & active_t.unsqueeze(0)          # [C,T]
        target = ground if ":G" in pair else body
        for ci, (body_a, body_b) in enumerate(members):
            hit = touching[ci]
            if not bool(hit.any()):
                continue
            target[hit, index[body_a]] = True
            if body_b is not None:
                target[hit, index[body_b]] = True
    return ground | body, ground, body


def summarise(name, old, new, body_names) -> dict:
    old_f, new_f = old.float(), new.float()
    both = (old & new).float().sum()
    return {
        "motion": name,
        "frames": int(old.shape[0]),
        "old_on_fraction": float(old_f.mean()),
        "new_on_fraction": float(new_f.mean()),
        "agreement": float((old == new).float().mean()),
        "old_only": float((old & ~new).float().mean()),
        "new_only": float((~old & new).float().mean()),
        "old_recall_of_new": float(both / new_f.sum().clamp_min(1)),
        "per_body_new": {
            n: round(float(new_f[:, i].mean()), 4)
            for i, n in enumerate(body_names)
            if float(new_f[:, i].mean()) > 0.01
        },
        "per_body_old": {
            n: round(float(old_f[:, i].mean()), 4)
            for i, n in enumerate(body_names)
            if float(old_f[:, i].mean()) > 0.01
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument(
        "--patch-slack",
        type=float,
        default=0.01,
        help="A member body counts as touching if within this of its zone's min gap.",
    )
    a = ap.parse_args()

    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(in_dir.rglob("*.motion"))
    if not clips:
        raise SystemExit(f"no .motion files under {in_dir}")
    body_names = mjcf_body_names(a.mjcf)

    reports = []
    for i, path in enumerate(clips, 1):
        resolved = compute_active_pairs(path, a.mjcf, DEFAULT_THRESHOLDS)
        if resolved is None:
            print(f"[{i}/{len(clips)}] {path.name}: not a motion dict, skipped")
            continue
        motion = resolved["motion"]
        new, ground, body = per_body_contacts(resolved, a.patch_slack)
        old = motion.get("rigid_body_contacts")
        old = (
            torch.zeros_like(new)
            if old is None
            else torch.as_tensor(old, dtype=torch.bool)
        )
        rep = summarise(path.stem, old, new, body_names)
        rep["ground_on_fraction"] = float(ground.float().mean())
        rep["body_on_fraction"] = float(body.float().mean())
        reports.append(rep)

        motion["rigid_body_contacts"] = new
        torch.save(motion, out_dir / path.name)
        # Sidecar with the ground/body split, for validation and diagnostics.
        np.savez_compressed(
            out_dir / f"{path.stem}.contacts.npz",
            any=new.numpy(), ground=ground.numpy(), body=body.numpy(),
            body_names=np.array(body_names),
        )
        print(
            f"[{i}/{len(clips)}] {path.stem[:52]:54s} "
            f"on {rep['old_on_fraction']*100:5.2f}% -> {rep['new_on_fraction']*100:5.2f}%  "
            f"(ground {rep['ground_on_fraction']*100:4.1f}% / body {rep['body_on_fraction']*100:4.1f}%)"
        )

    if a.report:
        rp = Path(a.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(
            {"patch_slack": a.patch_slack, "mjcf": a.mjcf,
             "in_dir": str(in_dir), "clips": reports}, indent=2))
        print(f"\nreport -> {rp}")
    print(f"wrote {len(reports)} clips to {out_dir}")


if __name__ == "__main__":
    main()
