# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price a probe plan against the corpus, before spending a GPU-hour on it.

Round 9's visual review asked, of half a dozen failing sequences, "is this a goal
definition issue?  a contact-graph issue?  or the policy?"  Four numbers answer it
offline, and each is a *different* failure:

**1. edge count** -- how many times the corpus demonstrates this hop.  Round 6
measured probe feasibility tracking it (>= 6 clean, 4-5 attempts/falls, 1 never
attempts).  ``count 0`` means the plan asks for a transition the graph does not
contain at all -- ``seq_plank_sideplank``'s plank -> side plank is one.

**2. a frozen hold clip for the destination** -- ``make_hold_motions.py`` cut 36
of them and they carry 23 % of the sampling mass.  They are the *only* place in
training where a pose is commanded and the reference does not move for ~10 s, so
they are the only source of "stay here" supervision.  A destination with no hold
clip has never been held on command.

**3. goal specificity** -- two numbers over the destination node's own goal
members, weighted by how often each is commanded.  ``near`` is the share of the
node's commanded exposure whose *goal pose* lies within ``--pose-radius`` of this
one: high means the contact set already implies the pose (a standing goal at the
standing hub, 80 % at 0.15 m -- the goal adds nothing where the fork is widest),
low means the pose half is carrying the whole specification alone.
``d_centroid`` is the distance from the commanded pose to the node's
exposure-weighted centroid -- how wrong a conditional-mean regressor over this
node would be (downdog sits 0.744 m from node 1's centroid, and node 1 contains
five side-plank segments).

**4. the stay-vs-go prior** -- P(commanded deadline < 1.5 s | the body is already
at this pose and this node is commanded), and the share of those frames that come
from a frozen hold clip.  ``next_goal_indices`` returns the first
hold more than ``min_lead_s`` ahead, so the deadline is a strict countdown and
42 % of trusted dwell is spent already commanded to the *next* hold.  Where that
probability is high the corpus has taught "arriving here means leaving soon".

Usage::

    PYTHONPATH=. python data/scripts/probe_plan_feasibility.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \\
      --plan data/scripts/plans/seq_plank_sideplank.json

Give ``--plan`` more than once, or ``--plan-dir`` for a whole set.
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
from goal_pose_separation import (  # noqa: E402
    CONDITIONABLE,
    GoalPoses,
    mean_body_distance,
)

DT = 1.0 / 30.0


def load_graph(path: str) -> dict:
    return json.loads(Path(path).read_text())


def is_frozen(clip: str) -> bool:
    return os.path.basename(clip).startswith("hold_")


class Corpus:
    """Commanded-goal exposure and the deadline distribution, per frame."""

    def __init__(self, graph: dict, motion_file: str, mjcf: str, lead_s: float,
                 stride: int = 3):
        lib = torch.load(motion_file, map_location="cpu", weights_only=False)
        weights = lib["motion_weights"].numpy().astype(float)
        self.p_clip = weights / weights.sum()
        self.lengths = lib["motion_lengths"].numpy().astype(float)
        self.poses = GoalPoses(motion_file, mjcf)
        self.body_ids = self.poses.body_ids(CONDITIONABLE)
        self.graph = graph
        self.node_key = [n["key"] for n in graph["nodes"]]
        self.node_id = {key: i for i, key in enumerate(self.node_key)}
        self.edges = {(e["src"], e["dst"]): e["count"] for e in graph["edges"]}

        weight, node, lead, pose, frozen = [], [], [], [], []
        # Goal-side exposure: how often each trusted hold is the commanded goal,
        # which is what "what else could this node's goal have been" is over.
        self.goal_members: dict = {}
        for clip, entry in graph["clips"].items():
            mid = entry["motion_id"]
            segs = sorted(
                (s for s in entry["segments"] if s.get("trusted", True)),
                key=lambda s: s["t_hold"],
            )
            if not segs:
                continue
            holds = np.array([s["t_hold"] for s in segs])
            steps = int(self.lengths[mid] / DT)
            times = np.arange(0, steps, stride) * DT
            first = np.searchsorted(holds, times + lead_s)
            per_frame = float(self.p_clip[mid]) * stride / max(steps, 1)
            for i, t in enumerate(times):
                k = int(first[i])
                if k >= len(segs):
                    continue
                member = self.goal_members.setdefault(int(segs[k]["node"]), {})
                key = (mid, round(float(segs[k]["t_hold"]), 3))
                if key not in member:
                    member[key] = [0.0, self.poses.at(
                        mid, float(segs[k]["t_hold"]), self.body_ids)]
                member[key][0] += per_frame
                weight.append(per_frame)
                node.append(int(segs[k]["node"]))
                lead.append(float(holds[k] - t))
                pose.append(self.poses.at(mid, float(t), self.body_ids))
                frozen.append(is_frozen(clip))
        self.w = np.array(weight)
        self.w /= self.w.sum()
        self.node = np.array(node)
        self.lead = np.array(lead)
        self.pose = np.stack(pose)
        self.frozen = np.array(frozen)

        # Frozen hold clips, by the configuration their SOURCE segment carried:
        # that is the goal a query names, and the frozen clip is what taught it.
        self.hold_clips = {}
        for clip, entry in graph["clips"].items():
            if not is_frozen(clip):
                continue
            stem = os.path.basename(clip)
            t_hold = float(stem.split("_")[1][1:])
            source = graph["clips"].get(stem.split("_", 2)[2])
            if source is None:
                continue
            seg = next(
                (s for s in source["segments"]
                 if s["t_start"] <= t_hold <= s["t_end"]),
                None,
            )
            if seg is None:
                continue
            self.hold_clips.setdefault(seg["config"], []).append(
                (source["motion_id"], t_hold, stem)
            )

    def motion_id(self, substring: str, live_only: bool = True) -> int:
        """Live clip first -- a frozen hold embeds its source clip's stem, so a
        plan naming the source would otherwise be ambiguous. Some generated
        plans name a frozen clip outright, hence the fallback."""
        for allow_frozen in ((False, True) if live_only else (True,)):
            for clip, entry in self.graph["clips"].items():
                if substring in clip and (allow_frozen or not is_frozen(clip)):
                    return entry["motion_id"]
        raise KeyError(substring)

    def pose_at(self, motion_id: int, t: float) -> np.ndarray:
        return self.poses.at(motion_id, t, self.body_ids)

    def score_goal(self, node: int, target: np.ndarray, radius: float) -> dict:
        commanded = self.node == node
        mass = float(self.w[commanded].sum())
        distance = np.linalg.norm(self.pose - target[None], axis=-1).mean(-1)
        near = commanded & (distance < radius)
        near_mass = float(self.w[near].sum())
        out = {
            "node_share": mass,
            "at_pose_rate": near_mass / mass if mass > 0 else float("nan"),
            "at_pose_share": near_mass,
        }
        # Goal specificity, over the node's own commanded goal poses.
        members = self.goal_members.get(node, {})
        total = sum(m[0] for m in members.values())
        if total > 0:
            share = np.array([m[0] for m in members.values()]) / total
            poses = np.stack([m[1] for m in members.values()])
            gaps = np.linalg.norm(poses - target[None], axis=-1).mean(-1)
            centroid = (poses * share[:, None, None]).sum(0)
            out["goal_near"] = float(share[gaps < radius].sum())
            out["d_centroid"] = mean_body_distance(target, centroid)
            out["members"] = len(members)
        else:
            out.update(goal_near=float("nan"), d_centroid=float("nan"), members=0)
        if near_mass > 0:
            out["p_lead_lt_1_5"] = float(
                self.w[near & (self.lead < 1.5)].sum()) / near_mass
            out["p_lead_gt_3"] = float(
                self.w[near & (self.lead > 3.0)].sum()) / near_mass
            out["frozen_share"] = float(
                self.w[near & self.frozen].sum()) / near_mass
        else:
            out.update(p_lead_lt_1_5=float("nan"), p_lead_gt_3=float("nan"),
                       frozen_share=float("nan"))
        return out

    def hold_clip_for(self, config: str, target: np.ndarray):
        """``(stem, pose distance)`` of the closest frozen hold naming ``config``."""
        best = None
        for motion_id, t_hold, stem in self.hold_clips.get(config, []):
            d = mean_body_distance(self.pose_at(motion_id, t_hold), target)
            if best is None or d < best[1]:
                best = (stem, d)
        return best


def report(plan_path: Path, corpus: Corpus, radius: float, collect=None) -> None:
    plan = json.loads(plan_path.read_text())
    # `--plan-dir` sweeps a directory, and the generated plan sets ship a
    # manifest.json beside the plans themselves.
    if not isinstance(plan, dict) or "goals" not in plan:
        return
    start = plan.get("start", {})
    start_mid = corpus.motion_id(start["clip"]) if start else None
    print(f"\n=== {plan_path.name} ===")
    if start_mid is not None:
        print(f"  start: {start['clip'][:56]} @ {start['time']} s")
    previous_node = None
    previous_time = None
    previous_mid = start_mid
    for goal in plan["goals"]:
        config = goal["config"]
        node = corpus.node_id.get(config)
        name = goal.get("name", "?")
        if node is None:
            print(f"  {name:14s} config does not resolve against this graph: {config}")
            continue
        mid = corpus.motion_id(goal.get("pose_clip", start.get("clip", "")))
        target = corpus.pose_at(mid, float(goal["pose_time"]))
        s = corpus.score_goal(node, target, radius)
        edge = (
            corpus.edges.get((previous_node, node), 0)
            if previous_node is not None and previous_node != node
            else None
        )
        hold = corpus.hold_clip_for(config, target)
        native = (
            float(goal["pose_time"]) - previous_time
            if previous_time is not None and mid == previous_mid
            else None
        )
        print(f"  {name:14s} node {node:3d}  {config[:46]}")
        print(
            f"      edge from previous  : "
            + ("(same node)" if edge is None else f"{edge:3d} occurrence(s)"
               + ("   << count 0: this hop is not in the graph" if edge == 0 else ""))
        )
        print(
            f"      frozen hold clip    : "
            + (f"{hold[0][:52]} ({hold[1]:.2f} m from the commanded pose)"
               if hold else "NONE for this configuration   << never held on command")
        )
        print(
            f"      goal specificity    : {s['goal_near']*100:5.1f} % of this node's "
            f"{s['members']} commanded goals are within {radius:.2f} m of this pose; "
            f"it sits {s['d_centroid']:.2f} m from the node's exposure-weighted centroid"
        )
        print(
            f"      at-pose rate        : the body is already within {radius:.2f} m of "
            f"it on {s['at_pose_rate']*100:5.1f} % of frames commanding this node"
        )
        print(
            f"      stay-vs-go prior    : P(deadline < 1.5 s | at pose, commanded) = "
            f"{s['p_lead_lt_1_5']*100:5.1f} %   P(> 3 s) = {s['p_lead_gt_3']*100:5.1f} %"
            f"   (frozen-hold share {s['frozen_share']*100:4.1f} %)"
        )
        if native is not None:
            print(
                f"      timing              : plan allows {goal['reach_s']:.1f} s; the "
                f"clip's own hold-to-hold gap is {native:.1f} s"
            )
        if collect is not None:
            collect.setdefault(plan_path.stem, []).append({
                "goal": name,
                "config": config,
                "node": node,
                "edge_count": edge,
                "hold_clip": None if hold is None else hold[0],
                "hold_clip_pose_gap_m": None if hold is None else round(hold[1], 3),
                "goal_near": round(s["goal_near"], 4),
                "d_centroid_m": round(s["d_centroid"], 3),
                "node_members": s["members"],
                "node_share": round(s["node_share"], 5),
                "at_pose_rate": round(s["at_pose_rate"], 4),
                "p_lead_lt_1_5": round(s["p_lead_lt_1_5"], 4),
                "p_lead_gt_3": round(s["p_lead_gt_3"], 4),
                "frozen_share": round(s["frozen_share"], 4),
                "reach_s": float(goal["reach_s"]),
                "hold_s": float(goal["hold_s"]),
                "native_gap_s": None if native is None else round(native, 3),
            })
        previous_node, previous_time, previous_mid = node, float(goal["pose_time"]), mid


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--plan-dir", default=None)
    parser.add_argument("--lead-s", type=float, default=0.2,
                        help="ContactGraphControl.min_lead_s")
    parser.add_argument("--pose-radius", type=float, default=0.20,
                        help="metres, 6-body mean; 0.20 is roughly 'the same pose'")
    parser.add_argument("--stride", type=int, default=3,
                        help="corpus frame stride (3 = 10 Hz)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="also write the per-goal rows as JSON, so a panel "
                             "run's measured hold rates can be joined onto the "
                             "predictors")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    graph = load_graph(args.graph)
    corpus = Corpus(graph, args.motion_file, args.mjcf, args.lead_s, args.stride)
    plans = [Path(p) for p in args.plan]
    if args.plan_dir:
        plans += sorted(Path(args.plan_dir).glob("*.json"))
    if not plans:
        raise SystemExit("give --plan or --plan-dir")
    collect = {} if args.json_out else None
    for plan in plans:
        report(plan, corpus, args.pose_radius, collect)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(collect, indent=1))
        print(f"\nwrote {args.json_out} ({len(collect)} plans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
