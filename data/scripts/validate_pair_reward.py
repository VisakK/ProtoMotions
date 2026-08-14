# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline validation of the pair-aware contact reward kernels (design §2.6).

Run BEFORE any training with ``mlp_pair_contact_rew.py``. Three passes:

1. **Geometry cross-check** — the training kernels inline a segment-segment
   surface-gap routine; verify it against ``contact_geometry.geom_pair_distance``
   (the annotation pipeline's validated kernels) on real reference frames for
   every vocabulary pair. Hard assert: max |gap difference| < 1e-4 m.

2. **Reference replay** — feed the reference kinematics through the kernels as
   if they were sim state (synthetic forces from the contact labels). Expected:
   masks fire in the annotated windows; crow ``L_THIGH+TRUNK`` gap statistics
   reproduce the measured min ~2.45 cm / median ~4.0 cm; the forbid term reads
   ~0 on the reference (its mask is ref-gap-gated by construction).

3. **Rollout replay** — feed the RECORDED policy rollouts
   (``results/Contact_Physics_analysis_crow_pair_final``) through the kernels.
   Expected: the kernels reproduce the known deficits — crow ``L_THIGH+TRUNK``
   stays open (the missing pair), side ``R_THIGH+L_UPPER_ARM`` is closed (the
   achieved pose-defining pair), the side forbid pair fires with substantial
   duty (the invented thigh-thigh load path; measured 37 % force duty / 58.8 %
   loose-band geometric dwell — the 2 cm psi ramp should land in between).

Usage::

    python data/scripts/validate_pair_reward.py \
        [--reftargets data/smpl/yoga_yogi_crow_pair_reftargets.pt] \
        [--clips-dir data/smpl/yoga_yogi_crow_pair_v3_contacts] \
        [--rollouts results/Contact_Physics_analysis_crow_pair_final]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data" / "scripts"))

from contact_geometry import (  # noqa: E402
    geom_pair_distance,
    geom_to_world,
    parse_typed_geoms,
)
from examples.experiments.mimic.pair_contact_terms import (  # noqa: E402
    PairEncourageReward,
    PairForbidPenalty,
    _load_reftargets,
    _pair_surface_gap,
)

MJCF = "data/assets/smpl/smpl_yogi03596_lowtorque.xml"

HARD_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str, hard: bool = True) -> None:
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{tag}] {name}: {detail}")
    if not ok and hard:
        HARD_FAILURES.append(name)


def load_motion(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def kernel_gaps(rt, body_pos: torch.Tensor, body_rot: torch.Tensor,
                body_a: int, body_b: int) -> torch.Tensor:
    ids = torch.tensor([body_a, body_b], dtype=torch.long)
    p0, p1, radii = rt.world_segments(body_pos, body_rot, ids)
    return _pair_surface_gap(p0[:, 0], p1[:, 0], radii[0], p0[:, 1], p1[:, 1], radii[1])


def reference_gaps_via_contact_geometry(
    motion: dict, body_names: list, body_a: str, body_b: str
) -> torch.Tensor:
    geoms = parse_typed_geoms(_REPO / MJCF, body_names)
    ia, ib = body_names.index(body_a), body_names.index(body_b)
    pos, rot = motion["rigid_body_pos"], motion["rigid_body_rot"]
    ga = geom_to_world(geoms[body_a][0], pos[:, ia], rot[:, ia])
    gb = geom_to_world(geoms[body_b][0], pos[:, ib], rot[:, ib])
    gap, _, _ = geom_pair_distance(ga, gb)
    return gap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reftargets", default="data/smpl/yoga_yogi_crow_pair_reftargets.pt")
    ap.add_argument("--clips-dir", default="data/smpl/yoga_yogi_crow_pair_v3_contacts")
    ap.add_argument("--rollouts", default="results/Contact_Physics_analysis_crow_pair_final")
    args = ap.parse_args()

    rt = _load_reftargets(str(_REPO / args.reftargets), torch.device("cpu"))
    raw = torch.load(_REPO / args.reftargets, map_location="cpu", weights_only=False)
    stems = list(raw["motion_names"])  # motion_id order is authoritative here
    clips_dir = _REPO / args.clips_dir
    print(f"reftargets: {args.reftargets}")
    print(f"{len(rt.pair_names)} encourage pairs, {len(rt.forbid_names)} forbid; "
          f"motions: {stems}\n")

    enc_kernel = PairEncourageReward(str(_REPO / args.reftargets))
    forbid_kernel = PairForbidPenalty(str(_REPO / args.reftargets))

    # ---------------------------------------------------------------- pass 1+2
    for clip_idx, stem in enumerate(stems):
        motion = load_motion(clips_dir / f"{stem}.motion")
        npz = np.load(clips_dir / f"{stem}.contacts.npz", allow_pickle=True)
        T = motion["rigid_body_pos"].shape[0]
        is_crow = "Side" not in stem
        print(f"\n=== {stem} (clip {clip_idx}, {T} frames, "
              f"{'crow' if is_crow else 'side crow'}) ===")

        # Pass 1: kernel gap vs contact_geometry gap, every body-body pair.
        worst = 0.0
        for p, pname in enumerate(rt.pair_names):
            b = int(rt.pair_body_b[p])
            if b < 0:
                continue
            a = int(rt.pair_body_a[p])
            mine = kernel_gaps(rt, motion["rigid_body_pos"],
                               motion["rigid_body_rot"], a, b)
            ref = reference_gaps_via_contact_geometry(
                motion, rt.body_names, rt.body_names[a], rt.body_names[b])
            worst = max(worst, float((mine - ref).abs().max()))
        for f, fname in enumerate(rt.forbid_names):
            mine = kernel_gaps(rt, motion["rigid_body_pos"],
                               motion["rigid_body_rot"],
                               int(rt.forbid_body_a[f]), int(rt.forbid_body_b[f]))
            ref = reference_gaps_via_contact_geometry(
                motion, rt.body_names,
                rt.body_names[int(rt.forbid_body_a[f])],
                rt.body_names[int(rt.forbid_body_b[f])])
            worst = max(worst, float((mine - ref).abs().max()))
        check("geometry-crosscheck", worst < 1e-4,
              f"max |kernel gap - contact_geometry gap| = {worst:.2e} m")

        # Pass 2: reference replay with synthetic forces from the labels.
        body_order = [str(b) for b in npz["body_names"]]
        perm = [body_order.index(n) for n in rt.body_names]
        any_m = torch.from_numpy(npz["any"][:, perm].astype(np.float32))
        gnd_m = torch.from_numpy(npz["ground"][:, perm].astype(np.float32))
        net = torch.zeros(T, 24, 3)
        net[..., 2] = 100.0 * any_m
        gnd = torch.zeros(T, 24, 3)
        gnd[..., 2] = 100.0 * gnd_m
        ids = torch.full((T,), clip_idx, dtype=torch.long)
        times = torch.arange(T, dtype=torch.float32) / rt.fps

        r_enc = enc_kernel(motion["rigid_body_pos"], motion["rigid_body_rot"],
                           net, gnd, ids, times)
        r_forb = forbid_kernel(motion["rigid_body_pos"], motion["rigid_body_rot"],
                               net, gnd, ids, times)
        fl = rt.frame_lookup(ids, times)
        active = rt.encourage_mask[fl].sum(-1) > 0.5
        print(f"  reference encourage reward: mean {r_enc.mean():.3f}, "
              f"in-annotated-frames mean {r_enc[active].mean():.3f}")
        check(f"{stem}: ref forbid ~ 0", float(r_forb.mean()) < 0.02,
              f"reference forbid mean {r_forb.mean():.4f} (mask is ref-gap-gated)")
        check(f"{stem}: frame lookup identity",
              bool((fl == rt.frame_starts[clip_idx] +
                    torch.arange(T)).all()),
              "frame_lookup(arange/fps) == arange")

        if is_crow:
            p = rt.pair_names.index("L_THIGH+TRUNK")
            gap = kernel_gaps(rt, motion["rigid_body_pos"],
                              motion["rigid_body_rot"],
                              int(rt.pair_body_a[p]), int(rt.pair_body_b[p]))
            mask = rt.encourage_mask[fl][:, p] > 0.5
            g = gap[mask]
            check("crow ref L_THIGH+TRUNK gap stats",
                  0.015 < float(g.min()) < 0.035 and 0.025 < float(g.median()) < 0.055,
                  f"min {g.min()*100:.2f} cm (measured ~2.45), "
                  f"median {g.median()*100:.2f} cm (measured ~4.0), "
                  f"mask dwell {mask.float().mean()*100:.1f}% (annotated 61.1%)")

    # ------------------------------------------------------------------ pass 3
    print("\n=== rollout replay ===")
    from contact_physics_support import load_rollout  # noqa: E402
    from compare_learned_contact_configs import rollout_to_motion  # noqa: E402

    for clip_idx, stem in enumerate(stems):
        roll_dir = _REPO / args.rollouts / stem
        if not roll_dir.is_dir():
            check(f"{stem}: rollout dir", False, f"{roll_dir} missing", hard=False)
            continue
        roll = load_rollout(roll_dir)
        motion, _idx = rollout_to_motion(roll, str(_REPO / MJCF), target_fps=60)
        pos, rot = motion["rigid_body_pos"], motion["rigid_body_rot"]
        T = pos.shape[0]
        Tm = int(rt.num_frames[clip_idx])
        T = min(T, Tm)
        pos, rot = pos[:T], rot[:T]
        ids = torch.full((T,), clip_idx, dtype=torch.long)
        times = torch.arange(T, dtype=torch.float32) / rt.fps
        fl = rt.frame_lookup(ids, times)
        is_crow = "Side" not in stem
        print(f"\n--- {stem} rollout ({T} frames used) ---")

        if is_crow:
            p = rt.pair_names.index("L_THIGH+TRUNK")
            gap = kernel_gaps(rt, pos, rot, int(rt.pair_body_a[p]),
                              int(rt.pair_body_b[p]))
            mask = rt.encourage_mask[fl][:, p] > 0.5
            closed = float((gap[mask] < 0.045).float().mean())
            check("rollout: crow L_THIGH+TRUNK stays open",
                  closed < 0.15,
                  f"within-4.5cm dwell {closed*100:.1f}% in annotated window "
                  f"(measured geometric dwell 0.0%); gap p50 "
                  f"{gap[mask].median()*100:.1f} cm")
        else:
            p = rt.pair_names.index("R_THIGH+L_UPPER_ARM")
            gap = kernel_gaps(rt, pos, rot, int(rt.pair_body_a[p]),
                              int(rt.pair_body_b[p]))
            mask = rt.encourage_mask[fl][:, p] > 0.5
            closed = float((gap[mask] < 0.045).float().mean())
            check("rollout: side R_THIGH+L_UPPER_ARM is made",
                  closed > 0.6,
                  f"within-4.5cm dwell {closed*100:.1f}% in annotated window "
                  f"(measured geometric dwell ~50.8% of clip / higher in-window)")

            f0 = 0
            fgap = kernel_gaps(rt, pos, rot, int(rt.forbid_body_a[f0]),
                               int(rt.forbid_body_b[f0]))
            psi = (1.0 - fgap.clamp_min(0.0) / 0.02).clamp(0.0, 1.0)
            raw_duty = float((psi > 0).float().mean())
            duty = float(((psi > 0) & (rt.forbid_mask[fl][:, f0] > 0.5))
                         .float().mean())
            check("rollout: side forbid pair fires", 0.15 < duty < 0.75,
                  f"thigh-thigh psi>0 duty: masked {duty*100:.1f}%, "
                  f"unmasked {raw_duty*100:.1f}% "
                  f"(measured: 37% force duty, 58.8% loose-band dwell); "
                  f"mask-ON fraction "
                  f"{float((rt.forbid_mask[fl][:, f0] > 0.5).float().mean())*100:.1f}%",
                  hard=False)

    print()
    if HARD_FAILURES:
        print(f"HARD FAILURES ({len(HARD_FAILURES)}): {HARD_FAILURES}")
        sys.exit(1)
    print("all hard checks passed")


if __name__ == "__main__":
    main()
