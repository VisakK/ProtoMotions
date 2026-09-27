# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repair a hold manifest so a hold command is never paired with the reference leaving the hold.

``propose_hold_manifest.py`` produced holds whose post-exemplar part contains motion:
the auto-inserted ``standing`` rests extend over a whole same-support run with no
stillness check (audit P1: 34/120 standing windows drift > 0.25 m after their
exemplar), and a few family holds contain their own slow exit (Headstand -a/-b,
Dancer -c). With ``include_current_segment`` the actor is told "deadline 0, stay
here for X s" on those frames while the tracking reward requires leaving -- a
contradiction the policy resolves in favour of the reward (run1_gap_analysis §2 G4,
§5.3: 10 of 10 rendered clips).

The repair is deliberately minimal: names, exemplars (``t_hold``), contact sets,
orientations, ``t_start`` and ``extend`` flags are untouched. Only ``t_end`` moves:
it becomes the last frame before the reference first drifts more than ``--delta``
from the hold's exemplar after ``t_hold``. Pose distance is the 6-goal-body,
pelvis-relative distance after a per-frame best-fit yaw (root-quaternion heading
flips on prone / inverted poses). By construction every frame the schedule serves
with deadline 0 is then within ``--delta`` of the commanded pose.

    PYTHONPATH=. python data/scripts/repair_hold_manifest.py --stats-only --delta 0.08 0.10 0.12
    PYTHONPATH=. python data/scripts/repair_hold_manifest.py --delta 0.10 \
        --out data/smpl/expert60/holds_repaired.yaml
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from propose_hold_manifest import COMMON_BODY_ORDER, GOAL_BODIES  # noqa: E402

GID = [COMMON_BODY_ORDER.index(b) for b in GOAL_BODIES]


def best_yaw_dist(frames: np.ndarray, exemplar: np.ndarray) -> np.ndarray:
    """Mean per-body distance of ``frames`` [T, B, 3] to ``exemplar`` [B, 3], both
    pelvis-relative, after rotating each frame by its best-fitting yaw."""
    a = frames[..., :2]
    b = exemplar[None, :, :2]
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    th = np.arctan2(num, den)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    x = c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] + c * a[..., 1]
    rot = np.stack([x, y, frames[..., 2]], -1)
    return np.linalg.norm(rot - exemplar[None], axis=-1).mean(-1)


def post_exemplar_distances(pos: np.ndarray, hold: dict) -> np.ndarray:
    """Distance of frames ``frame_hold .. frame_end`` (inclusive) to the exemplar."""
    fh, fe = int(hold["frame_hold"]), int(hold["frame_end"])
    ex = pos[fh][GID] - pos[fh][0]
    seg = pos[fh:fe + 1][:, GID] - pos[fh:fe + 1][:, :1]
    return best_yaw_dist(seg, ex)


def repaired_end_frame(dist: np.ndarray, frame_hold: int, delta: float) -> int:
    """Last frame before the first exceedance of ``delta`` (``frame_hold`` if the
    very next frame already exceeds)."""
    over = np.nonzero(dist > delta)[0]
    if len(over) == 0:
        return frame_hold + len(dist) - 1
    return frame_hold + max(int(over[0]) - 1, 0)


def repair_clip(clip: dict, pos: np.ndarray, fps: float, delta: float) -> tuple[dict, list]:
    out = copy.deepcopy(clip)
    changes = []
    for h in out["holds"]:
        dist = post_exemplar_distances(pos, h)
        new_end = repaired_end_frame(dist, int(h["frame_hold"]), delta)
        if new_end < int(h["frame_end"]):
            old_t_end = h["t_end"]
            h["repair"] = dict(
                t_end_before=old_t_end,
                frame_end_before=int(h["frame_end"]),
                delta_m=delta,
                max_post_exemplar_drift_m=round(float(dist.max()), 4),
            )
            h["frame_end"] = int(new_end)
            h["t_end"] = round(new_end / fps, 4)
            h["duration_s"] = round(h["t_end"] - h["t_start"], 3)
            changes.append(dict(name=h["name"], auto_rest=bool(h.get("auto_rest")),
                                extend=bool(h.get("extend")),
                                cut_s=round(old_t_end - h["t_end"], 3),
                                dwell_before_s=round(old_t_end - h["t_hold"], 3),
                                dwell_after_s=round(h["t_end"] - h["t_hold"], 3),
                                drift_m=round(float(dist.max()), 3)))
    return out, changes


def load_positions(clip: dict) -> tuple[np.ndarray, float]:
    m = torch.load(clip["source"], map_location="cpu", weights_only=False)
    return m["rigid_body_pos"].numpy(), float(m["fps"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="data/smpl/expert60/holds.yaml")
    ap.add_argument("--delta", type=float, nargs="+", default=[0.10],
                    help="max post-exemplar drift (m); several values with --stats-only")
    ap.add_argument("--out", default=None, help="repaired manifest path (single --delta)")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    manifest = yaml.safe_load(open(args.manifest))
    cache = {c["stem"]: load_positions(c) for c in manifest["clips"]}

    for delta in args.delta:
        all_changes = []
        for clip in manifest["clips"]:
            pos, fps = cache[clip["stem"]]
            _, changes = repair_clip(clip, pos, fps, delta)
            all_changes += [dict(clip=clip["stem"], **c) for c in changes]
        n_holds = sum(len(c["holds"]) for c in manifest["clips"])
        fam = [c for c in all_changes if c["extend"]]
        rest = [c for c in all_changes if c["auto_rest"]]
        other = [c for c in all_changes if not c["extend"] and not c["auto_rest"]]
        print(f"delta {delta:.2f} m: {len(all_changes)}/{n_holds} holds shortened "
              f"(auto-rest {len(rest)}, family {len(fam)}, other {len(other)}); "
              f"total cut {sum(c['cut_s'] for c in all_changes):.1f} s")
        if fam:
            print(f"   family holds: median dwell {statistics.median(c['dwell_before_s'] for c in fam):.2f}"
                  f" -> {statistics.median(c['dwell_after_s'] for c in fam):.2f} s")
            for c in sorted(fam, key=lambda c: -c["cut_s"])[:8]:
                print(f"     {c['clip'][7:55]:48} {c['name'][:28]:28} dwell {c['dwell_before_s']:5.2f} -> "
                      f"{c['dwell_after_s']:5.2f} s (drift {c['drift_m']:.2f} m)")
        if rest:
            print(f"   auto-rests: median dwell {statistics.median(c['dwell_before_s'] for c in rest):.2f}"
                  f" -> {statistics.median(c['dwell_after_s'] for c in rest):.2f} s")
        zero = [c for c in all_changes if c["dwell_after_s"] <= 0.05]
        print(f"   holds left with <= 0.05 s of post-exemplar dwell: {len(zero)}")

    if args.stats_only:
        return 0
    if len(args.delta) != 1 or args.out is None:
        raise SystemExit("write mode needs exactly one --delta and --out")
    delta = args.delta[0]
    repaired = copy.deepcopy(manifest)
    repaired["clips"] = []
    for clip in manifest["clips"]:
        pos, fps = cache[clip["stem"]]
        new_clip, _ = repair_clip(clip, pos, fps, delta)
        repaired["clips"].append(new_clip)
    repaired["repair"] = dict(
        source_manifest=str(Path(args.manifest).resolve()), delta_m=delta,
        rule="t_end := last frame before post-exemplar 6-body best-yaw drift first exceeds delta",
    )
    with open(args.out, "w") as f:
        yaml.safe_dump(repaired, f, sort_keys=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
