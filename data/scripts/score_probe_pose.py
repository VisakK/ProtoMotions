# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Did the probe reach the commanded POSE, or a different pose at the same node?

`summarize_probe_traces.py` scores contact IoU and root height. Neither can see
the failure that has been caught by eye in rounds 1, 4 and 6 and again in the v6
review: **the right support set, held in the wrong pose.** At a degenerate node
-- standing, single-leg, four-point -- the contact half is identical for every
member, so IoU is 1.00 whatever the policy does, and the only thing that
distinguishes Warrior III from Lord of the Dance is the pose half.

So this scores the pose half, offline, from the `.motion` a probe already
records. Two numbers per goal-hold window, both in the **student's own goal
representation** (the 6 conditionable bodies of ``trackable_bodies_subset``,
heading-normalised and pelvis-relative -- what ``build_sparse_target_poses``
hands the network):

* ``err`` -- mean per-body distance to the pose that was **commanded**.
* the **nearest node member** -- of all the trusted hold poses the graph records
  at the commanded node, which one is the achieved pose closest to? If a probe
  commanded Warrior III and lands nearest Dancer, that is the mode-averaging
  hypothesis of ``notes/Student_v7_improvement_investigation.MD`` §3 turned into
  a measurement rather than a viewing.

Read `err` against the node's own scale, which is printed: a 0.15 m error inside
a node whose members are 0.05 m apart is a different statement from the same
error inside one that spans 0.5 m.

Usage::

    PYTHONPATH=. python data/scripts/score_probe_pose.py \\
      --motion output/renderings/v6_probe/v6last_seq_warrior3_standing.motion \\
      --plan data/scripts/plans/seq_warrior3_standing.json \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml
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
    heading_quat_inv,
    mean_body_distance,
    quat_rotate,
)
from goal_exposure_baseline import goal_exposures, load_corpus_weights  # noqa: E402


def normalize_frame(pos: np.ndarray, root_rot: np.ndarray, body_ids) -> np.ndarray:
    """Heading-normalised, pelvis-relative positions of `body_ids` for one frame."""
    heading = heading_quat_inv(root_rot)
    return np.stack([quat_rotate(heading, p - pos[0]) for p in pos[body_ids]])


def resolve_clip(needle: str, names) -> str:
    """Same rule the probe drivers use: prefer the unique non-``hold_`` match."""
    hits = [n for n in names if needle in n]
    if len(hits) == 1:
        return hits[0]
    plain = [n for n in hits if not n.startswith("hold_")]
    if len(plain) == 1:
        return plain[0]
    raise ValueError(f"ambiguous or missing clip {needle!r}: {hits}")


def node_members(graph, rows, node_id, min_dwell_s):
    """Trusted hold segments at `node_id`, longest first."""
    members = [r for r in rows if r.node == node_id and r.segment_s >= min_dwell_s]
    members.sort(key=lambda r: -r.segment_s)
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=str, required=True,
                        help="the .motion a probe recorded")
    parser.add_argument("--plan", type=str, required=True)
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--motion-file", type=str, required=True,
                        help="the packaged corpus (for the commanded pose frames)")
    parser.add_argument("--mjcf", type=str, required=True)
    parser.add_argument("--min-dwell-s", type=float, default=0.4)
    parser.add_argument("--corpus-min-dwell-s", type=float, default=0.8,
                        help="minimum segment length for the corpus-wide "
                             "'what did it actually do?' ranking")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text())
    poses = GoalPoses(args.motion_file, args.mjcf)
    p_clip, lengths = load_corpus_weights(args.motion_file)
    rows = goal_exposures(graph, p_clip, lengths, 0.2)
    ids = poses.body_ids(CONDITIONABLE)
    name_to_mid = {k: c["motion_id"] for k, c in graph["clips"].items()}
    node_of_key = {n["key"]: i for i, n in enumerate(graph["nodes"])}

    rec = torch.load(args.motion, map_location="cpu", weights_only=False)
    pos = rec["rigid_body_pos"].numpy()
    rot = rec["rigid_body_rot"].numpy()
    fps = float(rec["fps"])

    plan = json.loads(Path(args.plan).read_text())
    label = args.label or Path(args.motion).stem
    print(f"=== {label}")
    print(f"    {pos.shape[0]} frames @ {fps:.0f} fps ({pos.shape[0] / fps:.1f} s), "
          f"plan {Path(args.plan).name}")
    print(f"    pose metric: mean per-body distance over "
          f"{CONDITIONABLE}, heading-normalised\n")

    results = []
    t_cursor = 0.0
    for goal in plan["goals"]:
        t_start = t_cursor
        t_end = t_cursor + float(goal["reach_s"]) + float(goal["hold_s"])
        hold_from = t_end - float(goal["hold_s"])
        t_cursor = t_end
        # Score the WHOLE goal window (reach + hold), not just the hold: a probe
        # can reach the pose and lose it, and the hold-window mean alone cannot
        # tell that apart from never reaching it.
        lo, hi = int(round(t_start * fps)), min(int(round(t_end * fps)), pos.shape[0])
        h_lo = int(round(hold_from * fps))
        if hi <= lo:
            continue

        clip = resolve_clip(goal["pose_clip"], name_to_mid.keys())
        commanded = poses.at(name_to_mid[clip], float(goal["pose_time"]), ids)
        achieved = np.stack([normalize_frame(pos[f], rot[f, 0], ids)
                             for f in range(lo, hi)])
        err = np.array([mean_body_distance(a, commanded) for a in achieved])
        hold_err = err[max(h_lo - lo, 0):]
        best = int(err.argmin())
        t_best = t_start + best / fps

        node_id = node_of_key.get(goal["config"])
        members = node_members(graph, rows, node_id, args.min_dwell_s) if node_id is not None else []
        # Rank node members against the achieved pose at its CLOSEST approach to
        # the command -- "at its best moment, what was it actually doing?"
        ranking = []
        for member in members:
            ref = poses.at(member.motion_id, member.t_hold, ids)
            ranking.append((mean_body_distance(achieved[best], ref), member))
        ranking.sort(key=lambda r: r[0])

        spread = float("nan")
        if len(members) > 1:
            member_poses = [poses.at(m.motion_id, m.t_hold, ids) for m in members]
            spread = float(np.mean([
                mean_body_distance(member_poses[i], member_poses[j])
                for i in range(len(members)) for j in range(i + 1, len(members))
            ]))

        commanded_stem = clip.replace("220923_", "").replace("220926_", "")
        print(f"  goal '{goal['name']}'  {t_start:.1f}-{t_end:.1f}s "
              f"(hold from {hold_from:.1f})  node {node_id} ({goal['config']})")
        print(f"      commanded: {commanded_stem} @ {goal['pose_time']}")
        print(f"      pose err:  best {err.min():.3f} m @ t={t_best:.1f}s | "
              f"hold mean {hold_err.mean():.3f} | final {err[-1]:.3f}"
              + (f"   [node spread {spread:.3f} m]" if len(members) > 1 else ""))
        # The nearest-member verdict only says something when the node has real
        # spread AND the winner beats the commanded pose by a real margin. On a
        # 91-member standing node whose members are 0.15 m apart, "nearest" is
        # noise -- report it, but do not flag it.
        flagged = False
        if ranking:
            best_dist, best_member = ranking[0]
            commanded_dist = next(
                (d for d, m in ranking if m.clip == clip and
                 abs(m.t_hold - float(goal["pose_time"])) < 0.05), None)
            if commanded_dist is None:
                commanded_dist = next((d for d, m in ranking if m.clip == clip), None)
            margin = (commanded_dist - best_dist) if commanded_dist is not None else 0.0
            flagged = (best_member.clip != clip and not np.isnan(spread)
                       and margin > 0.5 * spread)
            stem = best_member.clip.replace("220923_", "").replace("220926_", "")
            verdict = ("*** landed on a DIFFERENT node member ***" if flagged else
                       "matches the command" if best_member.clip == clip else
                       "(nearest is another member, but within the node's own noise)")
            print(f"      nearest member at its best frame: {stem} "
                  f"({best_dist:.3f} m) -- {verdict}")
            for dist, member in ranking[:4]:
                stem = member.clip.replace("220923_", "").replace("220926_", "")
                mark = " <- commanded" if (
                    member.clip == clip
                    and abs(member.t_hold - float(goal["pose_time"])) < 0.05) else ""
                print(f"        {dist:6.3f} m  {stem} ({member.segment_s:.1f}s, "
                      f"{member.share * 100:.3f}% exposure){mark}")

        # What did it settle into? Rank the mid-hold pose against EVERY trusted
        # hold in the corpus, not just the commanded node's members -- the
        # substitution a policy makes need not stay inside the node it was
        # asked for (a dancer and a warrior III share a ground contact set but
        # not an orientation bin).
        settled = achieved[min(max((h_lo + hi) // 2 - lo, 0), len(achieved) - 1)]
        corpus = sorted(
            (mean_body_distance(settled, poses.at(r.motion_id, r.t_hold, ids)), r)
            for r in rows if r.segment_s >= args.corpus_min_dwell_s
        )
        err_settled = mean_body_distance(settled, commanded)
        print(f"      settled pose (mid-hold) is {err_settled:.3f} m from the command; "
              "nearest holds anywhere in the corpus:")
        for dist, member in corpus[:3]:
            stem = member.clip.replace("220923_", "").replace("220926_", "")[:50]
            print(f"        {dist:6.3f} m  {stem}  [node {member.node} "
                  f"{graph['nodes'][member.node]['key'][:36]}]")
        print()
        results.append({
            "goal": goal["name"], "node": node_id, "config": goal["config"],
            "commanded_clip": clip, "pose_time": goal["pose_time"],
            "window_s": [round(t_start, 2), round(t_end, 2)],
            "pose_err_best": round(float(err.min()), 4),
            "t_best": round(float(t_best), 2),
            "pose_err_hold_mean": round(float(hold_err.mean()), 4),
            "pose_err_final": round(float(err[-1]), 4),
            "node_member_spread": None if np.isnan(spread) else round(spread, 4),
            "nearest_member": ranking[0][1].clip if ranking else None,
            "nearest_member_dist": round(ranking[0][0], 4) if ranking else None,
            "landed_on_different_member": bool(flagged),
            "pose_err_settled": round(float(err_settled), 4),
            "settled_nearest_corpus": corpus[0][1].clip,
            "settled_nearest_corpus_dist": round(float(corpus[0][0]), 4),
            "settled_nearest_corpus_node": corpus[0][1].node,
        })

    wrong = [r for r in results if r["landed_on_different_member"]]
    print(f"  SUMMARY: {len(results)} goals | best pose err "
          f"{np.mean([r['pose_err_best'] for r in results]):.3f} m mean, "
          f"hold-window {np.mean([r['pose_err_hold_mean'] for r in results]):.3f} m | "
          f"{len(wrong)} landed on a different node member"
          + (": " + ", ".join(r["goal"] for r in wrong) if wrong else ""))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"label": label, "plan": args.plan, "goals": results}, indent=1))
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
