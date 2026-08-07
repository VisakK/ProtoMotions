# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does the learned motion make the same contact configuration as the reference?

``extract_contact_configs.py`` builds the contact-configuration database from
*kinematic* clips by geometry alone.  A trained policy produces a different
motion, and the interesting question is whether it lands in the same
configuration -- the same zone pairs touching, in the same orientation bin -- or
only looks similar.

This runs three detectors over the same clip and lines them up:

1. **reference (kinematic)** -- ``extract_contact_configs`` on the reference
   ``.motion``, i.e. exactly what ``data/smpl/yoga_contact_configs/`` holds;
2. **learned (geometry)** -- the *same* extractor, same thresholds, run on the
   simulated body poses recorded by ``record_contact_physics.py``.  Same
   instrument, different subject, so a difference is a difference in motion and
   not in method;
3. **learned (measured force)** -- the zone pairs PhysX actually reported a
   contact force on during that rollout.

(3) is the control: the geometric detector was designed around the SMPL fit's
float bias and has an inference tier for supports it cannot see.  On simulated
motion there is no float bias and the forces are ground truth, so agreement
between (2) and (3) says the detector reads the sim correctly, which is what
licenses comparing (1) against (2).

Usage::

    PYTHONPATH=. python data/scripts/compare_learned_contact_configs.py \
      --rollout results/Contact_Physics_analysis_crow_pair_final/<clip> \
      --reference-motion data/smpl/yoga_yogi_balance_subset_v2_contacts/<clip>.motion \
      --reference-json data/smpl/yoga_contact_configs/<clip>.json \
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \
      --out results/.../contact_config_match.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_physics_support import load_rollout, step_slices  # noqa: E402
from extract_contact_configs import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    ZONE_ORDER,
    ZONES,
    clean_active,
    extract_clip,
    mjcf_body_names,
    zone_pairs,
)

BODY_TO_ZONE = {b: z for z, bodies in ZONES.items() for b in bodies}
ZONE_RANK = {z: i for i, z in enumerate(ZONE_ORDER)}


def body_pair_key(zone_a: str, zone_b: str) -> str:
    """Canonical body-body pair key, ordered the way ``zone_pairs()`` builds them.

    ZONE_ORDER, not alphabetical -- e.g. the side-crow contact is
    ``R_THIGH+L_UPPER_ARM``, which sorts the other way as a string.
    """
    first, second = sorted((zone_a, zone_b), key=ZONE_RANK.__getitem__)
    return f"{first}+{second}"


# --------------------------------------------------------------------------- #
# rollout -> kinematic .motion
# --------------------------------------------------------------------------- #
def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def _angular_velocity(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a quaternion series, by finite difference.

    The recorder stores poses, not link angular velocities; the extractor needs
    omega only for witness slip speeds and the stillness gates, where a central
    difference at 60 Hz is plenty.
    """
    q = quat_xyzw / np.linalg.norm(quat_xyzw, axis=-1, keepdims=True).clip(1e-9)
    q_next = np.roll(q, -1, axis=0)
    q_next[-1] = q[-1]
    q_prev = np.roll(q, 1, axis=0)
    q_prev[0] = q[0]
    # delta = q_next * conj(q_prev), xyzw
    v0, w0 = q_prev[..., :3], q_prev[..., 3:4]
    v1, w1 = q_next[..., :3], q_next[..., 3:4]
    conj_v, conj_w = -v0, w0
    vec = w1 * conj_v + conj_w * v1 + np.cross(v1, conj_v)
    scalar = w1 * conj_w - (v1 * conj_v).sum(-1, keepdims=True)
    sign = np.where(scalar < 0, -1.0, 1.0)
    vec = vec * sign
    span = np.full((len(q), 1, 1), 2.0 * dt)
    span[0] = span[-1] = dt
    return 2.0 * vec / span


def rollout_to_motion(roll, mjcf: str, target_fps: int = 60) -> tuple[dict, np.ndarray]:
    """Kinematic motion dict in MJCF body order; also returns the substep indices used."""
    mjcf_names = mjcf_body_names(mjcf)
    sim_names = roll.body_names
    missing = set(mjcf_names) - set(sim_names)
    if missing:
        raise SystemExit(f"rollout is missing bodies present in the MJCF: {sorted(missing)}")
    perm = np.array([sim_names.index(n) for n in mjcf_names])

    stride = max(1, int(round((1.0 / roll.dt_phys) / target_fps)))
    idx = np.arange(0, roll.num_substeps, stride)
    dt = roll.dt_phys * stride

    pos = roll.raw["body_pos_w"][idx][:, perm].astype(np.float32)
    quat = _quat_wxyz_to_xyzw(roll.raw["body_quat_w"][idx][:, perm]).astype(np.float32)
    vel = np.gradient(pos, dt, axis=0).astype(np.float32)
    ang_vel = _angular_velocity(quat, dt).astype(np.float32)

    motion = {
        "fps": int(round(1.0 / dt)),
        "rigid_body_pos": torch.from_numpy(pos),
        "rigid_body_rot": torch.from_numpy(quat),
        "rigid_body_vel": torch.from_numpy(vel),
        "rigid_body_ang_vel": torch.from_numpy(ang_vel),
    }
    return motion, idx


# --------------------------------------------------------------------------- #
# measured-force zone contacts
# --------------------------------------------------------------------------- #
def measured_zone_pairs(roll, idx: np.ndarray, fps: int, threshold: float = 5.0) -> dict:
    """Per-frame activity of every zone pair, from the recorded contact forces.

    Cleaned with the extractor's own merge/dwell cadence so the comparison is
    like-for-like rather than raw-threshold vs hysteresis-and-dwell.
    """
    force = np.linalg.norm(roll.raw["pair_force_w"][idx], axis=-1)  # [T, body, filter]
    filters = roll.filter_names
    body_zone = [BODY_TO_ZONE.get(b) for b in roll.body_names]
    filter_zone = [None if f == "ground" else BODY_TO_ZONE.get(f) for f in filters]

    active = {p: np.zeros(len(idx), dtype=bool) for p in zone_pairs()}
    for bi, zb in enumerate(body_zone):
        if zb is None:
            continue
        for fi, zf in enumerate(filter_zone):
            hot = force[:, bi, fi] > threshold
            if not hot.any():
                continue
            if filters[fi] == "ground":
                key = f"{zb}:G"
            else:
                if zf is None or zf == zb:
                    continue
                key = body_pair_key(zb, zf)
            if key not in active:
                # Masked-adjacent pairs (see ADJACENT) have no config slot; a
                # real pair silently vanishing here is what the assert catches.
                assert "+" in key, f"unexpected missing pair key {key}"
                continue
            active[key] |= hot

    merge_n = round(DEFAULT_THRESHOLDS["merge_s"] * fps)
    dwell_n = round(DEFAULT_THRESHOLDS["dwell_s"] * fps)
    return {p: clean_active(a, merge_n, dwell_n) for p, a in active.items()}


# --------------------------------------------------------------------------- #
def dwell_fractions(result: dict) -> dict[str, float]:
    """Fraction of the clip each pair is active, from an extract_clip result."""
    total = result["num_frames"]
    out: dict[str, float] = {}
    for seg in result["segments"]:
        n = seg["end_frame"] - seg["start_frame"] + 1
        for contact in seg["contacts"]:
            out[contact["pair"]] = out.get(contact["pair"], 0.0) + n / total
    return out


def dominant_segment(result: dict) -> dict:
    return max(result["segments"], key=lambda s: s["duration_s"])


def config_zone_pairs(config: str) -> set[str]:
    body = config.rsplit("@", 1)[0]
    return set() if body == "NONE" else set(body.split("|"))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout", required=True, help="Folder holding rollout.npz.")
    ap.add_argument("--reference-motion", required=True)
    ap.add_argument("--reference-json", default=None,
                    help="Stored corpus config JSON, compared as a third opinion.")
    ap.add_argument("--mjcf", default="data/assets/smpl/smpl_yogi03596_lowtorque.xml")
    ap.add_argument("--out", default=None, help="Write the markdown report here.")
    ap.add_argument("--force-threshold", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    roll = load_rollout(Path(args.rollout))
    motion, idx = rollout_to_motion(roll, args.mjcf, target_fps=args.fps)
    fps = motion["fps"]

    scratch = Path(args.rollout) / "_learned_kinematic.motion"
    torch.save(motion, scratch)
    learned = extract_clip(scratch, args.mjcf, DEFAULT_THRESHOLDS)
    reference = extract_clip(Path(args.reference_motion), args.mjcf, DEFAULT_THRESHOLDS)
    measured = measured_zone_pairs(roll, idx, fps, args.force_threshold)

    ref_dwell = dwell_fractions(reference)
    learn_dwell = dwell_fractions(learned)
    meas_dwell = {p: float(a.mean()) for p, a in measured.items() if a.any()}

    ref_dom, learn_dom = dominant_segment(reference), dominant_segment(learned)
    ref_set, learn_set = config_zone_pairs(ref_dom["config"]), config_zone_pairs(learn_dom["config"])

    lines = [f"# Contact configuration: reference vs learned — {Path(args.rollout).name}", ""]
    lines.append(f"- rollout: `{roll.raw['checkpoint']}`")
    lines.append(f"- compared at {fps} fps ({len(idx)} frames); force threshold {args.force_threshold:g} N")
    lines += ["", "## Dominant configuration", ""]
    lines.append(f"**reference** ({ref_dom['duration_s']:.2f}s, "
                 f"orientation `{ref_dom['orientation_bin']}`)")
    lines.append(f"```\n{ref_dom['config']}\n```")
    lines.append(f"**learned** ({learn_dom['duration_s']:.2f}s, "
                 f"orientation `{learn_dom['orientation_bin']}`)")
    lines.append(f"```\n{learn_dom['config']}\n```")
    lines.append("")
    lines.append(f"- orientation bin match: "
                 f"**{ref_dom['orientation_bin'] == learn_dom['orientation_bin']}**")
    lines.append(f"- zone-pair Jaccard: **{jaccard(ref_set, learn_set):.2f}** "
                 f"({len(ref_set & learn_set)} shared, {len(ref_set - learn_set)} missing, "
                 f"{len(learn_set - ref_set)} extra)")
    if ref_set - learn_set:
        lines.append(f"- in reference, not learned: `{'`, `'.join(sorted(ref_set - learn_set))}`")
    if learn_set - ref_set:
        lines.append(f"- in learned, not reference: `{'`, `'.join(sorted(learn_set - ref_set))}`")

    if args.reference_json:
        stored = json.load(open(args.reference_json))
        stored_dom = dominant_segment(stored)
        stored_set = config_zone_pairs(stored_dom["config"])
        lines += ["", f"Stored corpus config (`{Path(args.reference_json).name}`, "
                      f"{stored_dom['duration_s']:.2f}s, `{stored_dom['orientation_bin']}`) "
                      f"vs this reference run: Jaccard "
                      f"**{jaccard(stored_set, ref_set):.2f}**; vs learned: "
                      f"**{jaccard(stored_set, learn_set):.2f}**"]
        lines.append(f"```\n{stored_dom['config']}\n```")

    lines += ["", "## Per-pair dwell (fraction of clip)", "",
              "| pair | reference | learned (geometry) | learned (measured force) |",
              "|---|---|---|---|"]
    keys = sorted(set(ref_dwell) | set(learn_dwell) | set(meas_dwell),
                  key=lambda p: -(max(ref_dwell.get(p, 0), learn_dwell.get(p, 0),
                                      meas_dwell.get(p, 0))))
    for p in keys:
        r, l, m = ref_dwell.get(p, 0.0), learn_dwell.get(p, 0.0), meas_dwell.get(p, 0.0)
        if max(r, l, m) < 0.02:
            continue
        lines.append(f"| `{p}` | {100 * r:5.1f}% | {100 * l:5.1f}% | {100 * m:5.1f}% |")

    geom = {p for p, v in learn_dwell.items() if v > 0.05}
    meas = {p for p, v in meas_dwell.items() if v > 0.05}
    lines += ["", "## Detector control: geometry vs measured force on the same rollout", "",
              f"- agreement (Jaccard over pairs active >5% of the clip): "
              f"**{jaccard(geom, meas):.2f}**"]
    if geom - meas:
        lines.append(f"- geometry only (no force ever recorded): `{'`, `'.join(sorted(geom - meas))}`")
    if meas - geom:
        lines.append(f"- force only (geometry missed): `{'`, `'.join(sorted(meas - geom))}`")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    scratch.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
