# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When two poses share a contact node, can the *goal* tell them apart?

``contact_goal_baseline.py`` asked whether the contact half is redundant with the
pose half, and answered no.  This asks the mirror question, which the student's
observed failures actually turn on: inside one contact node -- where the contact
half is by construction identical and the reached-goal IoU is 1.00 for every
member -- how far apart are the members in the **pose half the policy is given**?

Two readings come out of it, and they point in opposite directions:

* **If the members are far apart**, the goal representation is sufficient and a
  policy that produces the wrong one is losing the distinction *downstream* --
  in the bottleneck between the goal and the action, or to sampling.  Enriching
  the input would be treating the wrong stage.
* **If they are close**, the representation genuinely cannot express the request
  and more conditionable bodies (or pose sub-nodes) are the fix.

The second output is the one that names the failure.  Weighting each member by
how often training commands it (``goal_exposure_baseline.py``) gives the node's
**conditional-mean pose**, and the distance from each member to it is how wrong
a mean-regressor is on that member.  The mode farthest from the centroid is the
one that gets replaced by its neighbours -- and, because the centroid usually
lands near some *other real pose*, the substitution looks like a competent
performance of the wrong asana rather than like a failure.

Body order is COMMON (the packed library's ``gts`` order), read from the MJCF
rather than from a remembered list -- the trap recorded in
``notes/Student_improvement_round4.MD`` §6.

Usage::

    PYTHONPATH=. python data/scripts/goal_pose_separation.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \\
      --node "L_FOOT:G@upright"
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
from extract_contact_configs import mjcf_body_names  # noqa: E402
from goal_exposure_baseline import (  # noqa: E402
    goal_exposures,
    load_corpus_weights,
)

# ``RobotConfig.trackable_bodies_subset`` for the SMPL family -- the bodies the
# pose half of a goal can name (protomotions/robot_configs/smpl.py).
CONDITIONABLE = ["Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head"]
# A plausible enrichment, for the "would more bodies help?" column: the joints
# that distinguish poses sharing the same endpoints (a scorpion is a handstand
# with bent knees and an arched back).
CONDITIONABLE_PLUS = CONDITIONABLE + [
    "L_Knee", "R_Knee", "L_Elbow", "R_Elbow", "Chest", "L_Toe", "R_Toe",
]


def heading_quat_inv(quat_xyzw: np.ndarray) -> np.ndarray:
    """Inverse yaw-only quaternion, matching ``rotations.calc_heading_quat_inv``."""
    x, y, z, w = quat_xyzw
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = -yaw / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)])


def quat_rotate(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    u = quat_xyzw[:3]
    w = quat_xyzw[3]
    return vec + 2.0 * np.cross(u, np.cross(u, vec) + w * vec)


class GoalPoses:
    """Heading-normalised, pelvis-relative body positions at any (clip, time)."""

    def __init__(self, motion_file: str, mjcf: str):
        lib = torch.load(motion_file, map_location="cpu", weights_only=False)
        self.pos = lib["gts"].numpy()
        self.rot = lib["grs"].numpy()
        self.starts = lib["length_starts"].numpy()
        self.frames = lib["motion_num_frames"].numpy()
        self.dt = lib["motion_dt"].numpy()
        self.body_names = mjcf_body_names(mjcf)
        if len(self.body_names) != self.pos.shape[1]:
            raise ValueError(
                f"{mjcf} has {len(self.body_names)} bodies but the library packs "
                f"{self.pos.shape[1]}: wrong asset for this corpus"
            )
        self.index = {name: i for i, name in enumerate(self.body_names)}

    def body_ids(self, names) -> list:
        missing = [n for n in names if n not in self.index]
        if missing:
            raise ValueError(f"bodies not in {self.body_names}: {missing}")
        return [self.index[n] for n in names]

    def at(self, motion_id: int, t: float, body_ids) -> np.ndarray:
        frame = int(round(t / float(self.dt[motion_id])))
        frame = max(0, min(frame, int(self.frames[motion_id]) - 1))
        row = int(self.starts[motion_id]) + frame
        pos = self.pos[row]
        heading = heading_quat_inv(self.rot[row, 0])
        return np.stack([quat_rotate(heading, p - pos[0]) for p in pos[body_ids]])


def mean_body_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over bodies of the per-body euclidean distance, in metres."""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def report_node(key, graph, poses, rows, args):
    node_ids = {n["key"]: i for i, n in enumerate(graph["nodes"])}
    nid = node_ids.get(key)
    if nid is None:
        print(f"\n!! node key not in this graph: {key}")
        return
    node = graph["nodes"][nid]
    members = [r for r in rows if r.node == nid and r.segment_s >= args.min_dwell_s]
    members.sort(key=lambda r: -r.segment_s)
    if len(members) < 2:
        print(f"\n=== node {nid} {key}: fewer than two segments over "
              f"{args.min_dwell_s}s, nothing to separate")
        return

    ids = poses.body_ids(CONDITIONABLE)
    ids_plus = poses.body_ids(CONDITIONABLE_PLUS)
    goal = np.array([poses.at(r.motion_id, r.t_hold, ids) for r in members])
    goal_plus = np.array([poses.at(r.motion_id, r.t_hold, ids_plus) for r in members])

    print(f"\n=== node {nid}  {key} ===")
    print(f"    {node['total_dwell_s']:.1f}s dwell, {len(members)} segments over "
          f"{args.min_dwell_s}s. Every member has the SAME contact set, so contact")
    print("    IoU is 1.00 for all of them: only the pose half can discriminate.\n")

    labels = [f"{r.clip.replace('220923_', '').replace('220926_', '')[:40]:40s} "
              f"{r.segment_s:5.1f}s" for r in members]
    header = "".join(f"{i:6d}" for i in range(len(members)))
    print(f"    {'':48s}{header}")
    for i, label in enumerate(labels):
        dists = "".join(
            f"{mean_body_distance(goal[i], goal[j]):6.2f}" for j in range(len(members))
        )
        print(f" {i:2d} {label:48s}{dists}")
    print(f"    (mean per-body distance, m, over the {len(CONDITIONABLE)} conditionable "
          "bodies)")

    # Would a richer body subset help? Report only where it matters -- pairs the
    # current subset already separates well need no help.
    gains = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            d6 = mean_body_distance(goal[i], goal[j])
            d13 = mean_body_distance(goal_plus[i], goal_plus[j])
            if d6 > 1e-6 and d13 / d6 > 1.15 and d6 < args.near_degenerate_m:
                gains.append((d13 / d6, i, j, d6, d13))
    if gains:
        print(f"\n    pairs where {len(CONDITIONABLE_PLUS)} bodies separate materially "
              f"better than {len(CONDITIONABLE)}:")
        for ratio, i, j, d6, d13 in sorted(gains, reverse=True)[:6]:
            print(f"      {i:2d} vs {j:2d}: {d6:.3f} -> {d13:.3f} m  ({ratio:.2f}x)")
    else:
        print(f"\n    no pair gains materially from the richer {len(CONDITIONABLE_PLUS)}-body "
              "subset: the endpoints already carry the distinction")

    # The conditional mean, and who is farthest from it.
    weights = np.array([r.share for r in members], dtype=float)
    if weights.sum() <= 0:
        return
    weights = weights / weights.sum()
    centroid = (goal * weights[:, None, None]).sum(axis=0)
    print("\n    distance to the EXPOSURE-WEIGHTED centroid — how wrong a "
          "conditional-mean\n    regressor is on each member:")
    ranked = sorted(
        ((mean_body_distance(goal[i], centroid), weights[i], labels[i])
         for i in range(len(members))),
        reverse=True,
    )
    for dist, weight, label in ranked:
        marker = "  <-- farthest from the mean" if dist == ranked[0][0] else ""
        print(f"      {dist:5.3f} m   commanded {weight * 100:5.1f} % of the node   "
              f"{label}{marker}")


def summarize(graph, poses, rows, args):
    """Corpus-wide: which commanded poses sit farthest from their node's mean?

    One line per (node, member); sorted by distance to the node's
    exposure-weighted centroid. The head of this list is the set of goals a
    conditional-mean regressor cannot express, ranked -- which is the shortlist
    a curriculum, a probe suite, or an RL stage should be pointed at.
    """
    ids = poses.body_ids(CONDITIONABLE)
    ranked = []
    for nid, node in enumerate(graph["nodes"]):
        members = [r for r in rows
                   if r.node == nid and r.segment_s >= args.min_dwell_s]
        if len(members) < 2:
            continue
        weights = np.array([r.share for r in members], dtype=float)
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()
        goal = np.array([poses.at(r.motion_id, r.t_hold, ids) for r in members])
        centroid = (goal * weights[:, None, None]).sum(axis=0)
        for i, member in enumerate(members):
            ranked.append((mean_body_distance(goal[i], centroid), member.share,
                           node["key"], member.clip, member.segment_s))

    dists = np.array([r[0] for r in ranked])
    print(f"{len(ranked)} (node, pose) members across "
          f"{len({r[2] for r in ranked})} multi-pose nodes")
    print(f"  distance to the node's exposure-weighted centroid: "
          f"median {np.median(dists):.3f} m, p90 {np.percentile(dists, 90):.3f} m, "
          f"max {dists.max():.3f} m")
    print(f"\n=== the {args.top_nodes * 5} commanded poses farthest from their own "
          "node's conditional mean ===")
    print(f"  {'dist':>6} {'exposure':>9}  node / clip")
    for dist, share, key, clip, seg in sorted(ranked, reverse=True)[: args.top_nodes * 5]:
        short = clip.replace("220923_", "").replace("220926_", "")[:46]
        print(f"  {dist:6.3f} {share * 100:8.3f}%  {key:46s}  {short} ({seg:.1f}s)")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--motion-file", type=str, required=True)
    parser.add_argument("--mjcf", type=str, required=True,
                        help="the robot MJCF, for COMMON body order")
    parser.add_argument("--node", type=str, action="append", default=[],
                        help="node key to analyse; repeatable. Default: the "
                             "highest-dwell nodes.")
    parser.add_argument("--top-nodes", type=int, default=4,
                        help="how many nodes to analyse when --node is not given")
    parser.add_argument("--min-dwell-s", type=float, default=0.4,
                        help="ignore segments shorter than this (they are passes, "
                             "not holds)")
    parser.add_argument("--near-degenerate-m", type=float, default=0.35,
                        help="only report richer-body gains for pairs the current "
                             "subset separates by less than this")
    parser.add_argument("--lead-s", type=float, default=0.2)
    parser.add_argument("--summary", action="store_true",
                        help="corpus-wide ranking of commanded poses by distance "
                             "to their own node's conditional mean")
    return parser


def main():
    args = create_parser().parse_args()
    graph = json.loads(Path(args.graph).read_text())
    poses = GoalPoses(args.motion_file, args.mjcf)
    p_clip, lengths = load_corpus_weights(args.motion_file)
    rows = goal_exposures(graph, p_clip, lengths, args.lead_s)

    if args.summary:
        summarize(graph, poses, rows, args)
        return

    keys = args.node
    if not keys:
        ranked = sorted(graph["nodes"], key=lambda n: -n["total_dwell_s"])
        keys = [n["key"] for n in ranked[: args.top_nodes]]
    for key in keys:
        report_node(key, graph, poses, rows, args)


if __name__ == "__main__":
    main()
