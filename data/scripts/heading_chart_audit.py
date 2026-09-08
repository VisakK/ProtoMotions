# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where is the heading frame every student observation lives in ill-conditioned?

``rotations.calc_heading`` takes the yaw of the root's **local x axis projected
onto the world XY plane**.  When that axis approaches vertical the projection
length goes to zero and the yaw is undefined: a small physical wobble in pitch or
roll becomes a large rotation of the extracted heading, with gain ``1/|fwd_xy|``.

That matters more here than in a locomotion setting, because *every* student
input is expressed in this frame -- ``max_coords_obs``, ``contact_obs_v1``,
``contact_proximity_obs``, ``historical_pose_obs``, the sparse pose goals and the
privileged dense future (see ``notes/V9_crucial_investigations/Student_v9_input_frames.MD``).
A single-frame flip therefore rotates the *entire* observation vector at once,
and the student has no absolute-yaw channel with which to notice.  Yoga spends a
great deal of time in exactly the postures that trigger it: any pose whose pelvis
is pitched to roughly vertical -- plank, chaturanga, cobra, downward dog,
inversions.

The graph/goal audit in ``notes/V9_crucial_investigations/yogi_v9_investigation_2026-09-06/graph_goal_findings.md``
found two individual instances of this (121.3 deg in Cobra -c for 0.42 deg of
real rotation, 97.5 deg in Koundinyasana -a for 1.97 deg).  This measures the
rate over the whole corpus and localises it by clip, so the defect can be ranked
against the others rather than anecdotally quoted.

A note on what this is and is not: it is a **representation defect with a cheap
fix** (a swing-twist decomposition about world Z is well defined except at a
180-degree rotation about a horizontal axis, a far smaller set).  It is not, on
this evidence, the cause of the v9 deployment failures -- ``chain_scorpion``
passes through two singular commanded goals and is the run's best skill, and the
standing hold is well conditioned and fails anyway.

Usage::

    PYTHONPATH=. python data/scripts/heading_chart_audit.py \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --plan-dir data/scripts/plans \\
      --json-out output/heading_chart_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def forward_xy(root_quat: torch.Tensor) -> torch.Tensor:
    """``|proj_xy(R x_hat)|`` per frame -- the conditioning of ``calc_heading``.

    ``R x_hat`` has components ``(1 - 2(y^2+z^2), 2(xy + wz), 2(xz - wy))``; the
    heading is ``atan2`` of the second over the first, so this norm is exactly
    the magnitude that ``atan2`` divides by.
    """
    x, y, z, w = root_quat.unbind(-1)
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    return torch.stack([fx, fy], dim=-1).norm(dim=-1)


def heading(root_quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = root_quat.unbind(-1)
    return torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def wrapped_diff_deg(angles: torch.Tensor) -> torch.Tensor:
    d = (angles[1:] - angles[:-1] + math.pi) % (2.0 * math.pi) - math.pi
    return d.abs() * 180.0 / math.pi


def audit_corpus(motion_file: str, singular: float, weak: float, jump_deg: float):
    lib = torch.load(motion_file, map_location="cpu", weights_only=False)
    grs, starts, frames = lib["grs"], lib["length_starts"], lib["motion_num_frames"]
    files = lib["motion_files"]
    rows = []
    total_frames = total_singular = total_jumps = 0
    for m in range(len(files)):
        s, n = int(starts[m]), int(frames[m])
        q = grs[s : s + n, 0].float()
        cond = forward_xy(q)
        jumps = wrapped_diff_deg(heading(q)) if n > 1 else torch.zeros(0)
        total_frames += n
        total_singular += int((cond < singular).sum())
        total_jumps += int((jumps > jump_deg).sum())
        rows.append(
            dict(
                clip=Path(files[m]).stem,
                frames=n,
                frac_singular=float((cond < singular).float().mean()),
                frac_weak=float((cond < weak).float().mean()),
                min_cond=float(cond.min()),
                jumps=int((jumps > jump_deg).sum()),
                max_jump_deg=float(jumps.max()) if jumps.numel() else 0.0,
            )
        )
    summary = dict(
        clips=len(files),
        frames=total_frames,
        frac_singular=total_singular / max(total_frames, 1),
        jumps=total_jumps,
        singular_threshold=singular,
        weak_threshold=weak,
        jump_threshold_deg=jump_deg,
    )
    return rows, summary


def audit_plans(plan_dir: str, motion_file: str):
    """Conditioning at every commanded goal frame of every probe plan."""
    lib = torch.load(motion_file, map_location="cpu", weights_only=False)
    grs, starts, frames = lib["grs"], lib["length_starts"], lib["motion_num_frames"]
    dt = lib["motion_dt"]
    stems = {Path(f).stem: i for i, f in enumerate(lib["motion_files"])}

    def resolve(needle: str):
        needle = Path(str(needle)).stem
        if needle in stems:
            return stems[needle]
        hits = [k for k in stems if k.endswith(needle) or needle.endswith(k)]
        return stems[hits[0]] if len(hits) >= 1 else None

    out = []
    paths = sorted(Path(plan_dir).rglob("*.json"))
    for path in paths:
        try:
            plan = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(plan, dict):
            continue
        for goal in plan.get("goals") or []:
            clip = goal.get("pose_clip") or goal.get("clip")
            t = goal.get("pose_time")
            if clip is None or t is None:
                continue
            m = resolve(clip)
            if m is None:
                continue
            n = int(frames[m])
            f = min(max(int(round(float(t) / float(dt[m]))), 0), n - 1)
            cond = float(forward_xy(grs[int(starts[m]) + f, 0].float().unsqueeze(0))[0])
            out.append(
                dict(plan=path.stem, goal=goal.get("name"), clip=Path(clip).stem,
                     pose_time=float(t), cond=cond, gain=1.0 / max(cond, 1e-6))
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--plan-dir", type=str, default=None)
    parser.add_argument("--singular", type=float, default=0.20,
                        help="|fwd_xy| below this is >=5x yaw gain")
    parser.add_argument("--weak", type=float, default=0.35)
    parser.add_argument("--jump-deg", type=float, default=30.0)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    rows, summary = audit_corpus(
        args.motion_file, args.singular, args.weak, args.jump_deg
    )
    print(f"corpus: {summary['clips']} clips, {summary['frames']} frames")
    print(f"frames with |fwd_xy| < {args.singular}: {summary['frac_singular']*100:.2f}%")
    print(f"one-frame heading jumps > {args.jump_deg:g} deg: {summary['jumps']}\n")

    print(f"{'clip':<66}{'sing%':>7}{'weak%':>7}{'jumps':>7}{'maxdeg':>8}")
    for r in sorted(rows, key=lambda r: -r["max_jump_deg"])[: args.top]:
        print(f"{r['clip'][:64]:<66}{100*r['frac_singular']:>7.1f}"
              f"{100*r['frac_weak']:>7.1f}{r['jumps']:>7}{r['max_jump_deg']:>8.1f}")
    print(f"\nmost time near the singularity:")
    for r in sorted(rows, key=lambda r: -r["frac_singular"])[: args.top]:
        print(f"{r['clip'][:64]:<66}{100*r['frac_singular']:>7.1f}{100*r['frac_weak']:>7.1f}")

    plans = audit_plans(args.plan_dir, args.motion_file) if args.plan_dir else []
    if plans:
        bad = [p for p in plans if p["cond"] < args.singular]
        print(f"\n{len(bad)} of {len(plans)} commanded goal frames across "
              f"{len({p['plan'] for p in plans})} plans are singular:")
        for p in sorted(bad, key=lambda p: p["cond"])[: args.top]:
            print(f"  {p['plan'][:40]:<42}{str(p['goal'])[:18]:<20}"
                  f"cond {p['cond']:.3f}  gain {p['gain']:.1f}x")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(dict(summary=summary, clips=rows, plan_goals=plans), indent=1)
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
