# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where does the corpus already contain the counterfactual the training task lacks?

``goal_state_redundancy.py`` measures the problem: within a clip the commanded
goal is a near-deterministic function of the pose, so an imitation loss is never
forced to make the goal causal.  The standard remedy is to break that
correlation with examples where the *same* state carries a *different* command.
The usual objection is that you cannot do that here without invalidating the
action label, because the expert is the playing clip's tracker
(``multi_expert.py:161``) and a goal from another clip would be labelled "keep
following this one".

That objection has an exception, and this script prices it: **two clips that pass
through the same physical state**.  At such a point the episode can switch clip
*and expert* together.  The state is unchanged, the new goal is the new clip's
own next hold, and the new label is the new clip's own expert tracking its own
reference -- so the label stays valid while the (state, goal) correlation breaks.
This is the label-safe form of "walk the graph instead of following one clip".

A pair is a **candidate** when, across two different clips:

* the 6-body goal-metric pose distance is under ``--pose-m``,
* the mean per-body linear-velocity difference is under ``--vel``,
* the root heights agree to ``--height-m``,
* both frames sit in the **same contact node** (so the support set does not have
  to change during the switch),

and **counterfactual** when the two frames' commanded goals differ by more than
``--goal-m``.  Only counterfactual candidates carry new information; the rest are
switches that teach nothing.

Two caveats the numbers cannot carry, both of which Z3 tests:

* a matched *pose* is not a matched *state* -- joint configuration, contact load
  allocation and angular momentum are not in the filter;
* the two clips have different world **yaw**, and ``align_motion_with_humanoid``
  re-anchors only XY, so a switch either needs a yaw re-anchor or must be
  restricted to pairs whose headings already agree (``--max-yaw-deg``).

Usage::

    PYTHONPATH=. python data/scripts/matched_state_pairs.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.pt \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \\
      --out-dir output/matched_state_pairs
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
from goal_state_redundancy import mean_body_dist, normalize_batch  # noqa: E402


def heading_deg(root_quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = root_quat.unbind(-1)
    return torch.rad2deg(
        torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    )


def build_index(graph, poses: GoalPoses, body_ids, stride: int, min_lead_s: float):
    """Per-frame table: normalised pose, goal pose, velocity, height, node, yaw."""
    seg_hold, seg_count = graph["seg_hold"], graph["seg_count"]
    seg_node, seg_start, seg_end = graph["seg_node"], graph["seg_start"], graph["seg_end"]
    velocities = torch.as_tensor(poses.velocity)
    cur, goal, vel, height, node, yaw, motion, tval = [], [], [], [], [], [], [], []
    for m in range(len(poses.frames)):
        n = int(poses.frames[m])
        dt = float(poses.dt[m])
        count = int(seg_count[m])
        holds = seg_hold[m][:count]
        if holds.numel() == 0:
            continue
        t = torch.arange(n) * dt
        j = torch.searchsorted(holds.contiguous(), (t + min_lead_s).contiguous())
        live = j < holds.numel()
        sel = torch.nonzero(live).squeeze(-1)[::stride]
        if sel.numel() == 0:
            continue
        start = int(poses.starts[m])
        pos = torch.as_tensor(poses.pos[start : start + n]).float()
        rot = torch.as_tensor(poses.rot[start : start + n]).float()
        norm = normalize_batch(pos, rot, body_ids)
        goal_frame = (holds[j[sel]] / dt).round().long().clamp(0, n - 1)

        # Which segment (hence node) each sampled frame sits in.
        starts_m, ends_m, nodes_m = seg_start[m][:count], seg_end[m][:count], seg_node[m][:count]
        k = (torch.searchsorted(starts_m.contiguous(), t[sel].contiguous(), right=True) - 1)
        k = k.clamp(0, count - 1)
        inside = t[sel] <= ends_m[k]

        cur.append(norm[sel])
        goal.append(norm[goal_frame])
        vel.append(velocities[start : start + n][sel].float())
        height.append(pos[sel, 0, 2])
        node.append(torch.where(inside, nodes_m[k], torch.full_like(nodes_m[k], -1)))
        yaw.append(heading_deg(rot[sel, 0]))
        motion.append(torch.full((sel.numel(),), m, dtype=torch.long))
        tval.append(t[sel])
    return dict(
        cur=torch.cat(cur), goal=torch.cat(goal), vel=torch.cat(vel),
        height=torch.cat(height), node=torch.cat(node), yaw=torch.cat(yaw),
        motion=torch.cat(motion), t=torch.cat(tval),
    )


def draw_pairs(path: Path, pairs, poses, names, common_names, parent_indices,
               count=8):
    """Strip of matched pairs: same body, different command.

    Each row has two panels sharing one scale. LEFT overlays clip A's and clip
    B's frames -- they should be indistinguishable, that is what "matched" means.
    RIGHT overlays the goal each of them is commanded at that instant, which is
    the supervision the training task never presents together.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    from decode_all_codes_report import bones_of

    bones = bones_of(common_names, parent_indices)
    rows = pairs[:count]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10.0, 2.6 * len(rows)), dpi=115)
    if len(rows) == 1:
        axes = [axes]
    for ax, pair in zip(axes, rows):
        frames = {
            ("state", "a"): poses.full_frame(pair["motion_a"], pair["t_a"]),
            ("state", "b"): poses.full_frame(pair["motion_b"], pair["t_b"]),
            ("goal", "a"): poses.full_frame(pair["motion_a"], pair["goal_t_a"]),
            ("goal", "b"): poses.full_frame(pair["motion_b"], pair["goal_t_b"]),
        }
        stack = np.stack(list(frames.values()))
        span = max(
            float(stack[..., 0].max() - stack[..., 0].min()),
            float(stack[..., 2].max() - stack[..., 2].min()), 1e-3,
        )
        scale = 1.0 / span
        mid_x = float(stack[..., 0].mean())
        low_z = float(stack[..., 2].min())
        panel = {"state": 0.0, "goal": 1.35}
        colour = {"a": "#1f77b4", "b": "#d62728"}
        segs, colours = [], []
        for (kind, side), pose in frames.items():
            ox = panel[kind]
            for child, parent, _c in bones:
                segs.append([
                    (ox + (pose[parent, 0] - mid_x) * scale,
                     (pose[parent, 2] - low_z) * scale),
                    (ox + (pose[child, 0] - mid_x) * scale,
                     (pose[child, 2] - low_z) * scale),
                ])
                colours.append(colour[side])
        ax.add_collection(
            LineCollection(segs, colors=colours, linewidths=1.7, alpha=0.8)
        )
        ax.set_xlim(-0.78, 2.16)
        ax.set_ylim(-0.30, 1.62)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(-0.70, 1.46,
                f"node {pair['node']}   state gap {pair['pose_gap_m']:.3f} m, "
                f"|dv| {pair['vel_gap']:.2f} m/s, yaw {pair['yaw_gap_deg']:.0f} deg"
                f"   ->   GOAL GAP {pair['goal_gap_m']:.3f} m",
                fontsize=8.5, family="monospace")
        ax.text(-0.70, 1.30, f"A  {names[pair['motion_a']][:60]} @{pair['t_a']:.2f}s",
                fontsize=7, color="#1f77b4", family="monospace")
        ax.text(-0.70, 1.16, f"B  {names[pair['motion_b']][:60]} @{pair['t_b']:.2f}s",
                fontsize=7, color="#d62728", family="monospace")
        ax.text(0.0, -0.25, "MATCHED STATE  (A over B)", fontsize=7.5,
                ha="center", color="#555555", family="monospace")
        ax.text(1.35, -0.25, "EACH CLIP'S OWN GOAL", fontsize=7.5,
                ha="center", color="#555555", family="monospace")
    fig.suptitle(
        "Label-safe counterfactuals already in the corpus: one physical state, "
        "two different commands", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--graph", required=True, help="contact_graph.pt")
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--sample", type=int, default=3000,
                        help="query frames (all frames are searched as partners)")
    parser.add_argument("--candidates", type=int, default=32,
                        help="nearest cross-clip partners examined per query")
    parser.add_argument("--pose-m", type=float, default=0.10)
    parser.add_argument("--vel", type=float, default=0.30)
    parser.add_argument("--height-m", type=float, default=0.05)
    parser.add_argument("--goal-m", type=float, nargs="+", default=[0.25, 0.47])
    parser.add_argument("--max-yaw-deg", type=float, default=30.0,
                        help="reported as a second, stricter tier: a mid-episode "
                             "switch has no yaw re-anchor today")
    parser.add_argument("--export", type=int, default=64,
                        help="candidate switch points written for the handoff screen")
    parser.add_argument("--max-per-clip-pair", type=int, default=2,
                        help="diversity cap on the export: one clip pair sampled at "
                             "20 Hz otherwise fills the whole list with near-duplicates")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    poses = GoalPoses(args.motion_file, args.mjcf)
    lib = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    poses.velocity = lib["gvs"].numpy()
    names = [Path(f).stem for f in lib["motion_files"]]

    def full_frame(motion, t):
        n = int(poses.frames[motion])
        f = min(max(int(round(t / float(poses.dt[motion]))), 0), n - 1)
        row = int(poses.starts[motion]) + f
        pos = torch.as_tensor(poses.pos[row]).float().unsqueeze(0)
        rot = torch.as_tensor(poses.rot[row]).float().unsqueeze(0)
        return normalize_batch(pos, rot, list(range(pos.shape[1])))[0].numpy()

    poses.full_frame = full_frame
    body_ids = poses.body_ids(CONDITIONABLE)
    min_lead = float(graph.get("min_lead_s", 0.2))

    table = build_index(graph, poses, body_ids, args.stride, min_lead)
    n = table["cur"].shape[0]
    flat = table["cur"].reshape(n, -1)
    print(f"{n} frames indexed over {len(names)} clips (stride {args.stride})")

    generator = torch.Generator().manual_seed(args.seed)
    query = torch.randperm(n, generator=generator)[: min(args.sample, n)]
    dist = torch.cdist(flat[query], flat) / np.sqrt(len(body_ids))
    dist[table["motion"][query].unsqueeze(1) == table["motion"].unsqueeze(0)] = float("inf")
    best = dist.topk(args.candidates, largest=False)
    idx = best.indices

    pose_gap = best.values
    vel_gap = (table["vel"][query].unsqueeze(1) - table["vel"][idx]).norm(dim=-1).mean(-1)
    height_gap = (table["height"][query].unsqueeze(1) - table["height"][idx]).abs()
    same_node = (table["node"][query].unsqueeze(1) == table["node"][idx]) & (
        table["node"][idx] >= 0
    )
    yaw_gap = (table["yaw"][query].unsqueeze(1) - table["yaw"][idx]).abs()
    yaw_gap = torch.minimum(yaw_gap, 360.0 - yaw_gap)
    goal_gap = torch.stack(
        [mean_body_dist(table["goal"][query], table["goal"][idx[:, k]])
         for k in range(args.candidates)], dim=1
    )

    stages = [
        ("pose match", pose_gap < args.pose_m),
    ]
    stages.append(("  + velocity", stages[-1][1] & (vel_gap < args.vel)))
    stages.append(("  + root height", stages[-1][1] & (height_gap < args.height_m)))
    stages.append(("  + same contact node", stages[-1][1] & same_node))
    matched = stages[-1][1]
    for threshold in args.goal_m:
        stages.append((f"  + goal differs > {threshold:g} m", matched & (goal_gap > threshold)))
    stages.append(
        (f"  + and yaw within {args.max_yaw_deg:g} deg",
         matched & (goal_gap > args.goal_m[0]) & (yaw_gap < args.max_yaw_deg))
    )

    print(f"\n{'filter':<40}{'frames':>10}{'partners/frame':>17}")
    summary = {}
    for label, mask in stages:
        frac = float(mask.any(dim=1).float().mean())
        rate = float(mask.sum(dim=1).float().mean())
        print(f"{label:<40}{100*frac:>9.1f}%{rate:>17.1f}")
        summary[label.strip()] = dict(frac_of_frames=frac, partners_per_frame=rate)

    counterfactual = matched & (goal_gap > args.goal_m[0])
    if bool(counterfactual.any()):
        gaps = goal_gap[counterfactual]
        print(f"\ngoal gap among counterfactual pairs: "
              f"p50 {gaps.median():.3f}  p90 {torch.quantile(gaps, 0.9):.3f}  "
              f"max {gaps.max():.3f} m")

    # ---- export candidate switch points, best first ---- #
    rows = []
    q_idx, c_idx = torch.nonzero(counterfactual, as_tuple=True)
    score = goal_gap[q_idx, c_idx] - 2.0 * pose_gap[q_idx, c_idx]
    seen: dict = {}
    for rank in torch.argsort(score, descending=True).tolist():
        qi, ci = int(q_idx[rank]), int(c_idx[rank])
        a, b = int(query[qi]), int(idx[qi, ci])
        key = tuple(sorted((int(table["motion"][a]), int(table["motion"][b]))))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > args.max_per_clip_pair:
            continue
        rows.append(dict(
            node=int(table["node"][a]),
            motion_a=int(table["motion"][a]), t_a=float(table["t"][a]),
            clip_a=names[int(table["motion"][a])],
            motion_b=int(table["motion"][b]), t_b=float(table["t"][b]),
            clip_b=names[int(table["motion"][b])],
            pose_gap_m=float(pose_gap[qi, ci]), vel_gap=float(vel_gap[qi, ci]),
            height_gap_m=float(height_gap[qi, ci]), yaw_gap_deg=float(yaw_gap[qi, ci]),
            goal_gap_m=float(goal_gap[qi, ci]),
            goal_t_a=float(_goal_time(graph, table, a, min_lead, poses)),
            goal_t_b=float(_goal_time(graph, table, b, min_lead, poses)),
        ))
        if len(rows) >= args.export:
            break

    print(f"\ntop counterfactual switch points ({len(rows)} exported):")
    print(f"{'node':>5}  {'clip A @t':<44}{'clip B @t':<44}{'pose':>6}{'goal':>7}{'yaw':>7}")
    for row in rows[:10]:
        print(f"{row['node']:>5}  {row['clip_a'][:36]}@{row['t_a']:>5.2f}  "
              f"{row['clip_b'][:36]}@{row['t_b']:>5.2f}  "
              f"{row['pose_gap_m']:>5.3f}{row['goal_gap_m']:>7.3f}{row['yaw_gap_deg']:>7.1f}")

    (out_dir / "summary.json").write_text(json.dumps(
        dict(frames=n, clips=len(names), stride=args.stride,
             filters=dict(pose_m=args.pose_m, vel=args.vel, height_m=args.height_m,
                          goal_m=args.goal_m, max_yaw_deg=args.max_yaw_deg),
             stages=summary), indent=1))
    (out_dir / "switch_points.json").write_text(json.dumps(rows, indent=1))

    kin_names = poses.body_names
    parents = _parent_indices(args.mjcf, kin_names)
    draw_pairs(out_dir / "matched_pairs.png", rows, poses, names, kin_names, parents)
    print(f"\nwrote {out_dir}")
    return 0


def _goal_time(graph, table, row, min_lead, poses):
    motion = int(table["motion"][row])
    holds = graph["seg_hold"][motion][: int(graph["seg_count"][motion])]
    t = float(table["t"][row])
    j = int(torch.searchsorted(holds.contiguous(), torch.tensor(t + min_lead)))
    j = min(j, holds.numel() - 1)
    return float(holds[j])


def _parent_indices(mjcf: str, body_names):
    """Parent index per body, from the MJCF tree (COMMON order)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(mjcf).getroot()
    order, parent = [], {}

    def walk(node, parent_name):
        for body in node.findall("body"):
            name = body.get("name")
            order.append(name)
            parent[name] = parent_name
            walk(body, name)

    walk(root.find("worldbody"), None)
    index = {name: i for i, name in enumerate(body_names)}
    return [index.get(parent[name], -1) if parent[name] else -1 for name in body_names]


if __name__ == "__main__":
    raise SystemExit(main())
