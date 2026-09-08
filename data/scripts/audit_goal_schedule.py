# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What does the contact-graph goal schedule actually ask the student for?

Four questions the student's failures turn on, all answerable offline from the
graph JSON plus the packaged corpus.  In the spirit of
``contact_goal_baseline.py`` / ``kinematic_share_baseline.py``: decide it in
seconds on a CPU before spending a GPU-night.

**A. Orientation bin.**  A node's key carries a 6-way trunk-orientation bin
chosen by a *sticky* argmax over gravity-in-root with a 0.15 hysteresis margin
(``extract_contact_configs.orientation_bins``).  Poses whose torso sits near a
bin boundary therefore get whichever bin they were approached from.  Warrior III
is the worst case in this corpus and it is the pose four rounds of review have
caught being substituted.

**B. Deadline.**  ``ContactGraph.next_goal_indices`` returns the first hold more
than ``min_lead_s`` ahead, so the commanded deadline never falls below 0.2 s in
training -- while at inference ``ContactGraphControl.step`` clamps a *pinned*
manual goal at exactly that floor and holds it there indefinitely.  This section
prices how much of training looks like the inference-time hold regime.

**C. Support change.**  How often the commanded goal asks the body to unload a
grounded zone at all -- the commitment decision the FSQ intent code exists for.

**D. Goal velocity.**  Whether a target COM velocity would carry information, or
would be the constant that ``Singleleg14_pressure_experiment.MD`` §3.2 warns
about (a target so nearly constant that scoring it measures the target).

Usage::

    PYTHONPATH=. python data/scripts/audit_goal_schedule.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_pose_separation import (  # noqa: E402
    CONDITIONABLE,
    GoalPoses,
    mean_body_distance,
)

from protomotions.utils import rotations  # noqa: E402

# The margin ``orientation_bins`` requires before a challenger bin takes over.
HYSTERESIS = 0.15


def trusted_segments(graph: dict) -> list:
    rows = []
    for name, clip in graph["clips"].items():
        for segment in clip["segments"]:
            if segment.get("trusted"):
                rows.append({**segment, "clip": name, "motion_id": clip["motion_id"]})
    return rows


def orientation_scores(quats: torch.Tensor) -> np.ndarray:
    """``[N, 6]`` bin scores, in ``ORIENT_BINS`` order."""
    down = torch.tensor([[0.0, 0.0, -1.0]]).expand(quats.shape[0], 3)
    g = rotations.quat_rotate_inverse(quats, down, w_last=True).numpy()
    return np.stack([-g[:, 2], g[:, 2], g[:, 0], -g[:, 0], g[:, 1], -g[:, 1]], axis=1)


def section_a(graph, rows, lib, args):
    print("=" * 92)
    print("A. ORIENTATION BIN -- how decisive is the 6-way argmax at the corpus's hold frames?")
    print("=" * 92)
    names = graph["orientation_names"]
    starts, frames, dts = lib["length_starts"].numpy(), lib["motion_num_frames"].numpy(), lib["motion_dt"].numpy()
    index = []
    for r in rows:
        f = int(round(r["t_hold"] / dts[r["motion_id"]]))
        index.append(int(starts[r["motion_id"]]) + max(0, min(f, int(frames[r["motion_id"]]) - 1)))
    scores = orientation_scores(lib["grs"][index, 0])
    order = np.argsort(-scores, axis=1)
    margin = scores[np.arange(len(scores)), order[:, 0]] - scores[np.arange(len(scores)), order[:, 1]]
    for r, m, o in zip(rows, margin, order):
        r["margin"], r["argmax"], r["runner"] = float(m), names[o[0]], names[o[1]]

    dwell = np.array([r["duration_s"] for r in rows])
    print(f"  {len(rows)} trusted holds.  margin = top bin score - runner-up "
          f"(the builder's hysteresis is {HYSTERESIS})")
    print(f"    median {np.median(margin):.3f}   p25 {np.percentile(margin, 25):.3f}   "
          f"p10 {np.percentile(margin, 10):.3f}   min {margin.min():.3f}")
    for threshold in (HYSTERESIS, 0.30):
        share = margin < threshold
        print(f"    holds within {threshold:.2f} of a bin flip: {share.sum():3d} / {len(rows)} "
              f"({100 * share.mean():.1f} %), {100 * dwell[share].sum() / dwell.sum():.1f} % of dwell")
    flipped = [r for r in rows if r["argmax"] != r["orientation_bin"]]
    print(f"\n  holds whose STORED bin != their own instantaneous argmax (the sticky rule "
          f"carried another bin in):\n    {len(flipped)} of {len(rows)} "
          f"({100 * len(flipped) / len(rows):.1f} %), "
          f"{100 * sum(r['duration_s'] for r in flipped) / dwell.sum():.1f} % of dwell")
    for r in sorted(flipped, key=lambda r: -r["duration_s"])[: args.top]:
        print(f"      {r['duration_s']:6.2f}s  stored {r['orientation_bin']:8s} <- argmax "
              f"{r['argmax']:8s} (margin {r['margin']:.3f} over {r['runner']})  {r['clip'][:46]}")
    if args.pose_filter:
        print(f"\n  --- holds matching /{args.pose_filter}/ ---")
        pattern = re.compile(args.pose_filter, re.I)
        for r in sorted((r for r in rows if pattern.search(r["clip"]) and r["duration_s"] >= 0.9),
                        key=lambda r: (r["clip"], r["t_hold"])):
            flag = "   <<< STORED != ARGMAX" if r["argmax"] != r["orientation_bin"] else ""
            print(f"    {r['clip'][:52]:52s} t{r['t_hold']:6.2f} {r['duration_s']:5.2f}s "
                  f"node{r['node']:4d} stored {r['orientation_bin']:8s} argmax {r['argmax']:8s} "
                  f"marg {r['margin']:5.3f}{flag}")


def _per_clip_schedule(graph, lib, lead):
    """Yield ``(clip, motion_id, times, goal_index, valid, current_node)`` per clip."""
    frames, dts = lib["motion_num_frames"].numpy(), lib["motion_dt"].numpy()
    weights = lib["motion_weights"].numpy()
    for name, clip in graph["clips"].items():
        motion_id = clip["motion_id"]
        segments = [s for s in clip["segments"] if s.get("trusted")]
        if not segments:
            continue
        holds = np.array([s["t_hold"] for s in segments])
        n, dt = int(frames[motion_id]), float(dts[motion_id])
        t = np.arange(n) * dt
        # Exactly ContactGraph.next_goal_indices for slot 0.
        j = np.searchsorted(holds, t + lead)
        valid = j < len(holds)
        current = np.full(n, -1)
        for s in segments:
            a, b = int(round(s["t_start"] / dt)), int(round(s["t_end"] / dt))
            current[max(0, a) : min(n, b + 1)] = s["node"]
        yield name, motion_id, t, np.clip(j, 0, len(holds) - 1), valid, current, segments, float(weights[motion_id]) / n


def section_b(graph, lib, poses, args):
    print()
    print("=" * 92)
    print("B. DEADLINE -- is the goal ever 'stay where you are'?")
    print("=" * 92)
    deadlines, weights, pose_err, pose_w = [], [], [], []
    total = same_node = no_goal = hold_total = hold_no_goal = 0.0
    ids = poses.body_ids(CONDITIONABLE) if poses else None
    for name, motion_id, t, j, valid, current, segments, w in _per_clip_schedule(graph, lib, args.min_lead_s):
        holds = np.array([s["t_hold"] for s in segments])
        total += w * len(t)
        no_goal += w * (~valid).sum()
        if name.startswith("hold_"):
            hold_total += w * len(t)
            hold_no_goal += w * (~valid).sum()
        deadlines.append((holds[j] - t)[valid])
        weights.append(np.full(int(valid.sum()), w))
        goal_node = np.array([segments[k]["node"] for k in j])
        same_node += w * ((current == goal_node) & valid).sum()
        if ids is None:
            continue
        for f in range(0, len(t), args.pose_stride):
            if not valid[f]:
                continue
            pose_err.append(mean_body_distance(
                poses.at(motion_id, t[f], ids), poses.at(motion_id, holds[j[f]], ids)
            ))
            pose_w.append(w * args.pose_stride)

    deadlines, weights = np.concatenate(deadlines), np.concatenate(weights)
    weights = weights / weights.sum()
    order = np.argsort(deadlines)
    d, cw = deadlines[order], np.cumsum(weights[order])
    q = lambda p: d[np.searchsorted(cw, p)]  # noqa: E731
    print("  slot-0 deadline (t_hold - now), weighted by the corpus's own motion_weights:")
    print(f"    p10 {q(.10):.2f}s   p25 {q(.25):.2f}s   MEDIAN {q(.50):.2f}s   "
          f"p75 {q(.75):.2f}s   p90 {q(.90):.2f}s   max {d[-1]:.1f}s")
    for threshold in (0.25, 0.5, 1.0):
        print(f"    frames with deadline < {threshold:4.2f}s : "
              f"{100 * weights[order][d < threshold].sum():5.2f} %")
    print(f"\n  frames with NO goal at all (past the clip's last hold): {100 * no_goal / total:.2f} % "
          f"(on hold_ clips: {100 * hold_no_goal / max(hold_total, 1e-9):.2f} %)")
    print(f"  frames whose CURRENT node already equals the commanded goal node: "
          f"{100 * same_node / total:.1f} %")
    if pose_err:
        pe = np.array(pose_err)
        pw = np.array(pose_w)
        pw = pw / pw.sum()
        o = np.argsort(pe)
        e, c = pe[o], np.cumsum(pw[o])
        qp = lambda p: e[np.searchsorted(c, p)]  # noqa: E731
        print(f"\n  distance from the CURRENT frame to the commanded goal frame "
              f"({len(CONDITIONABLE)} bodies, m):")
        print(f"    p10 {qp(.10):.3f}  MEDIAN {qp(.50):.3f}  p90 {qp(.90):.3f};  "
              f"already within 0.05 m on {100 * pw[o][e < 0.05].sum():.1f} % of frames")
    print(f"\n  NOTE at inference ContactGraphControl.step clamps a pinned manual goal's "
          f"deadline at\n       min_lead_s = {args.min_lead_s:.2f}s and holds it there for "
          f"the rest of the episode.")


def section_c(graph, lib, args):
    print()
    print("=" * 92)
    print("C. SUPPORT CHANGE -- how often does the goal ask the body to unload a limb?")
    print("=" * 92)
    ground = [set(p for p in n["pairs"] if p.endswith(":G")) for n in graph["nodes"]]
    release, acquire, weights = [], [], []
    for _, _, t, j, valid, current, segments, w in _per_clip_schedule(graph, lib, args.min_lead_s):
        for f in range(0, len(t), args.pose_stride):
            if not valid[f] or current[f] < 0:
                continue
            goal = segments[j[f]]["node"]
            release.append(len(ground[current[f]] - ground[goal]))
            acquire.append(len(ground[goal] - ground[current[f]]))
            weights.append(w * args.pose_stride)
    release, acquire = np.array(release), np.array(acquire)
    weights = np.array(weights)
    weights = weights / weights.sum()
    for label, v in (("must RELEASE a grounded zone", release), ("must ACQUIRE a grounded zone", acquire)):
        print(f"  {label:32s}" + "".join(
            f"  {k if k < 3 else '3+'}: {100 * weights[(v == k) if k < 3 else (v >= 3)].sum():5.1f} %"
            for k in range(4)
        ))
    identical = (release == 0) & (acquire == 0)
    print(f"  goal support set IDENTICAL to the current one: {100 * weights[identical].sum():.1f} % "
          f"-> on those frames only the POSE half separates 'now' from 'the goal'.")


def section_d(graph, rows, lib):
    print()
    print("=" * 92)
    print("D. GOAL VELOCITY -- would a target COM velocity carry information?")
    print("=" * 92)
    starts, frames, dts = lib["length_starts"].numpy(), lib["motion_num_frames"].numpy(), lib["motion_dt"].numpy()
    speed, duration = [], []
    for r in rows:
        f = int(round(r["t_hold"] / dts[r["motion_id"]]))
        row = int(starts[r["motion_id"]]) + max(0, min(f, int(frames[r["motion_id"]]) - 1))
        speed.append(float(np.linalg.norm(lib["gvs"][row].numpy().mean(axis=0))))
        duration.append(r["duration_s"])
    speed, duration = np.array(speed), np.array(duration)
    print(f"  COM speed at the goal frame (m/s): median {np.median(speed):.4f}  "
          f"p90 {np.percentile(speed, 90):.4f}  max {speed.max():.4f}")
    print(f"    below 0.10 m/s on {100 * (speed < 0.10).mean():.1f} % of goals, "
          f"{100 * duration[speed < 0.10].sum() / duration.sum():.1f} % of dwell")
    print("    -> goals sit at t_hold, the most-static half-second of a segment by "
          "construction. A target\n       velocity is very nearly the constant zero: the "
          "Singleleg14 §3.2 trap.")
    print(f"\n  what DOES vary, and is not in the goal -- how long the pose is held:")
    print(f"    duration (s): p10 {np.percentile(duration, 10):.2f}  median {np.median(duration):.2f}  "
          f"p90 {np.percentile(duration, 90):.2f};  under 1 s: {100 * (duration < 1.0).mean():.0f} %")
    print(f"    corr(COM speed at goal, segment duration) = {np.corrcoef(speed, duration)[0, 1]:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--mjcf", default=None,
                        help="asset MJCF; enables the pose-distance column of section B")
    parser.add_argument("--min-lead-s", type=float, default=0.2)
    parser.add_argument("--pose-stride", type=int, default=10,
                        help="frame stride for the pose-distance sweep (section B)")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--pose-filter", default=None,
                        help="regex over clip names; lists those holds in section A")
    parser.add_argument("--sections", default="abcd")
    args = parser.parse_args()

    graph = json.loads(open(args.graph).read())
    lib = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    rows = trusted_segments(graph)
    poses = GoalPoses(args.motion_file, args.mjcf) if args.mjcf else None

    print(f"graph {args.graph}")
    print(f"  rule={graph.get('body_pair_identity', '?')} "
          f"node_identity={graph.get('node_identity', 'segment')}  "
          f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
          f"{len(rows)} trusted holds\n")
    if "a" in args.sections:
        section_a(graph, rows, lib, args)
    if "b" in args.sections:
        section_b(graph, lib, poses, args)
    if "c" in args.sections:
        section_c(graph, lib, args)
    if "d" in args.sections:
        section_d(graph, rows, lib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
