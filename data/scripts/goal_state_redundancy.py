# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How much does the CURRENT pose already determine the COMMANDED goal?

Every round of this project has asked whether the goal representation is rich
enough (``goal_pose_separation.py``), whether the contact half is redundant with
the pose half (``contact_goal_baseline.py``), and how often each goal is
commanded (``goal_exposure_baseline.py``).  None of them asked the question that
decides whether an imitation loss can make the goal *causal*:

    on the training distribution, is the commanded goal already a function of
    the state the policy is standing in?

If it is, the cheapest way to drive the per-frame action error to zero is the
clip-continuation map ``state -> action``, and the goal channel is free to be
ignored -- which is exactly the failure mode the deployment probes report
("commanded to stand, performs a shoulderstand"; "commanded a cobra chain,
performs Lord of the Dance").  This is textbook causal confusion in imitation
learning, and the diagnostic for it is a conditional spread, not a loss curve.

The measurement reproduces the training schedule exactly --
``ContactGraph.next_goal_indices`` with the frozen ``min_lead_s``, over the
packaged corpus -- and reports the goal's spread three ways:

* **marginal**: how far apart are two randomly drawn commanded goals;
* **conditional, any clip**: how far apart are the goals of two frames that are
  near neighbours *in pose*;
* **conditional, cross-clip only**: the same with same-clip neighbours excluded,
  because the nearest neighbour of a frame is otherwise its own next frame and
  the number measures temporal leakage rather than generalisation.

The gap between the last two is the whole finding: within a clip the goal is
nearly determined by the pose, across clips it is not.  A network that
generalises across clips can use the goal; a network that takes the shortcut
never has to.

Distances are the project's own goal metric -- the mean over the 6 conditionable
bodies of the distance between poses each taken pelvis-relative and rotated into
the heading-normalised frame of its own root (``GoalPoses`` in
``goal_pose_separation.py``, which is byte-for-byte what
``ContactGraphControl._goal_pose_error`` logs live).

Usage::

    PYTHONPATH=. python data/scripts/goal_state_redundancy.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.pt \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \\
      --json-out output/goal_state_redundancy.json
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
from goal_pose_separation import CONDITIONABLE, GoalPoses  # noqa: E402


# --------------------------------------------------------------------------- #
# Vectorised twin of GoalPoses.at
# --------------------------------------------------------------------------- #
def normalize_batch(pos: torch.Tensor, rot: torch.Tensor, body_ids) -> torch.Tensor:
    """``[N, len(body_ids), 3]`` pelvis-relative, heading-normalised positions.

    Same arithmetic as ``GoalPoses.at`` / ``score_probe_pose.normalize_frame``,
    batched.  The heading is the yaw of the root's rotated x axis --
    ``atan2(2(xy + wz), 1 - 2(y^2 + z^2))`` -- which is ``calc_heading`` written
    out, so this agrees with the live metric and inherits its singularity (see
    ``heading_chart_audit.py``).
    """
    root = rot[:, 0] if rot.dim() == 3 else rot  # [N, 4] xyzw, body 0 is Pelvis
    x, y, z, w = root[:, 0], root[:, 1], root[:, 2], root[:, 3]
    yaw = torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    rel = pos[:, body_ids, :] - pos[:, 0:1, :]
    return torch.stack(
        [
            rel[..., 0] * c[:, None] - rel[..., 1] * s[:, None],
            rel[..., 0] * s[:, None] + rel[..., 1] * c[:, None],
            rel[..., 2],
        ],
        dim=-1,
    )


def mean_body_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean over bodies of the per-body distance, for ``[N, B, 3]`` inputs."""
    return (a - b).norm(dim=-1).mean(dim=-1)


# --------------------------------------------------------------------------- #
def build_frames(graph, poses: GoalPoses, body_ids, stride: int, min_lead_s: float):
    """``(current [N,B,3], goal [N,B,3], clip_id [N], motion [N], time [N])``.

    One row per sampled frame that still has a scheduled goal, with the goal
    resolved the way ``next_goal_indices`` resolves it: the first hold whose
    ``t_hold`` exceeds ``motion_time + min_lead_s``.
    """
    seg_hold, seg_count = graph["seg_hold"], graph["seg_count"]
    cur, goal, clip, motion, tvals = [], [], [], [], []
    for m in range(len(poses.frames)):
        n = int(poses.frames[m])
        dt = float(poses.dt[m])
        holds = seg_hold[m][: int(seg_count[m])]
        if holds.numel() == 0:
            continue
        t = torch.arange(n) * dt
        j = torch.searchsorted(holds.contiguous(), (t + min_lead_s).contiguous())
        live = j < holds.numel()
        sel = torch.nonzero(live).squeeze(-1)[::stride]
        if sel.numel() == 0:
            continue
        goal_t = holds[j[sel]]
        goal_f = (goal_t / dt).round().long().clamp(0, n - 1)
        start = int(poses.starts[m])
        pos = torch.as_tensor(poses.pos[start : start + n]).float()
        rot = torch.as_tensor(poses.rot[start : start + n]).float()
        norm = normalize_batch(pos, rot, body_ids)
        cur.append(norm[sel])
        goal.append(norm[goal_f])
        clip.append(torch.full((sel.numel(),), m, dtype=torch.long))
        motion.append(torch.full((sel.numel(),), m, dtype=torch.long))
        tvals.append(t[sel])
    return (
        torch.cat(cur),
        torch.cat(goal),
        torch.cat(clip),
        torch.cat(motion),
        torch.cat(tvals),
    )


def conditional_spread(cur, goal, clip, k_list, sample, seed, exclude_same_clip):
    """Mean goal distance to the goals of the K nearest neighbours in state."""
    generator = torch.Generator().manual_seed(seed)
    n = cur.shape[0]
    idx = torch.randperm(n, generator=generator)[: min(sample, n)]
    flat_c = cur.reshape(n, -1)
    # cdist over the flattened 6x3 block; /sqrt(B) puts it in per-body metres,
    # which is the same scale the pairwise metric below reports.
    dist = torch.cdist(flat_c[idx], flat_c) / np.sqrt(cur.shape[1])
    dist[torch.arange(len(idx)), idx] = float("inf")
    if exclude_same_clip:
        dist[clip[idx].unsqueeze(1) == clip.unsqueeze(0)] = float("inf")
    out = {}
    for k in k_list:
        best = dist.topk(k, largest=False)
        gaps = torch.stack(
            [mean_body_dist(goal[idx], goal[best.indices[:, j]]) for j in range(k)], 1
        )
        states = torch.stack(
            [mean_body_dist(cur[idx], cur[best.indices[:, j]]) for j in range(k)], 1
        )
        out[k] = dict(
            goal_spread_m=float(gaps.mean()),
            state_gap_m=float(states.mean()),
            per_row=gaps.mean(dim=1),
        )
    return out, idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--graph", required=True, help="contact_graph.pt")
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--min-lead-s", type=float, default=None,
                        help="default: the value stored in the graph")
    parser.add_argument("--stride", type=int, default=3,
                        help="frame subsample (3 = ~20 Hz on a 60 fps corpus)")
    parser.add_argument("--sample", type=int, default=2500,
                        help="query rows for the conditional spreads")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    poses = GoalPoses(args.motion_file, args.mjcf)
    body_ids = poses.body_ids(CONDITIONABLE)
    min_lead = (
        float(graph.get("min_lead_s", 0.2))
        if args.min_lead_s is None
        else args.min_lead_s
    )

    cur, goal, clip, motion, tvals = build_frames(
        graph, poses, body_ids, args.stride, min_lead
    )
    n = cur.shape[0]

    # Agreement with the reference implementation, on a handful of rows: this
    # metric is quoted against round 7's 0.47 m Warrior III / Dancer separation,
    # so it has to be the same metric.
    check = []
    for row in (0, n // 3, 2 * n // 3):
        m = int(motion[row])
        ref = torch.as_tensor(poses.at(m, float(tvals[row]), body_ids)).float()
        check.append(float((ref - cur[row]).abs().max()))
    max_dev = max(check)
    assert max_dev < 1e-4, f"vectorised metric disagrees with GoalPoses.at: {max_dev}"

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=generator)
    marginal = float(mean_body_dist(goal, goal[perm]).mean())
    to_go = float(mean_body_dist(goal, cur).mean())

    any_clip, _ = conditional_spread(
        cur, goal, clip, args.k, args.sample, args.seed, exclude_same_clip=False
    )
    cross, idx = conditional_spread(
        cur, goal, clip, args.k, args.sample, args.seed, exclude_same_clip=True
    )

    print(f"{n} (state, commanded-goal) frames over {len(poses.frames)} clips")
    print(f"metric agrees with GoalPoses.at to {max_dev:.2e} m\n")
    print(f"marginal   E|goal_i - goal_j|            {marginal:.3f} m")
    print(f"           E|goal - current|             {to_go:.3f} m\n")
    print(f"{'K':>4}  {'any-clip':>10}  {'cross-clip':>11}  {'state gap':>10}  {'removed':>8}")
    for k in args.k:
        removed = 100.0 * (1.0 - cross[k]["goal_spread_m"] / marginal)
        print(
            f"{k:>4}  {any_clip[k]['goal_spread_m']:>10.3f}  "
            f"{cross[k]['goal_spread_m']:>11.3f}  {cross[k]['state_gap_m']:>10.3f}  "
            f"{removed:>7.1f}%"
        )
    print()
    for k in args.k:
        removed = 100.0 * (1.0 - any_clip[k]["goal_spread_m"] / marginal)
        print(f"  within-clip neighbours remove {removed:.1f}% of the goal spread at K={k}")

    ref_k = args.k[len(args.k) // 2]
    resid = cross[ref_k]["per_row"]
    quant = torch.quantile(resid, torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99]))
    print()
    print(f"cross-clip residual (K={ref_k})  p50 {quant[0]:.3f}  p75 {quant[1]:.3f}  "
          f"p90 {quant[2]:.3f}  p95 {quant[3]:.3f}  p99 {quant[4]:.3f} m")
    print(f"frames with >0.25 m residual goal ambiguity: {float((resid > 0.25).float().mean()):.3f}")
    print(f"frames with >0.47 m (Warrior III vs Dancer): {float((resid > 0.47).float().mean()):.3f}")

    if args.json_out:
        payload = dict(
            frames=n,
            clips=int(len(poses.frames)),
            min_lead_s=min_lead,
            stride=args.stride,
            metric_max_deviation_m=max_dev,
            marginal_goal_spread_m=marginal,
            mean_distance_to_goal_m=to_go,
            any_clip={
                str(k): dict(goal_spread_m=v["goal_spread_m"], state_gap_m=v["state_gap_m"])
                for k, v in any_clip.items()
            },
            cross_clip={
                str(k): dict(goal_spread_m=v["goal_spread_m"], state_gap_m=v["state_gap_m"])
                for k, v in cross.items()
            },
            cross_clip_residual_percentiles_m={
                q: float(v) for q, v in zip(("p50", "p75", "p90", "p95", "p99"), quant)
            },
            frac_residual_gt_0_25=float((resid > 0.25).float().mean()),
            frac_residual_gt_0_47=float((resid > 0.47).float().mean()),
        )
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
