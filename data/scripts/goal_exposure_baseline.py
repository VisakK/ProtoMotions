# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How often does training ever *ask* for each goal?

Every exposure argument in the student work so far has been made at the level of
clips (``select_student_subset.py --summary``) or edges
(``notes/Student_improvement_round6.MD`` §5).  Neither answers the question a
mode-averaging failure actually turns on: *of all the frames the student trains
on, what fraction carries this particular goal?*  A pose the schedule commands
on 0.3 % of frames is competing, inside its own contact node, with siblings
commanded five times more often -- and a regressor asked for the rare one will
be pulled toward the common ones.

This reproduces the goal schedule exactly (``ContactGraph.next_goal_indices``:
the first trusted hold later than ``t + min_lead_s`` of the clip being played,
``protomotions/components/contact_graph.py``) and weights it by the corpus's own
episode-sampling weights.  CPU only, seconds, no GPU and no training -- the same
role ``contact_goal_baseline.py`` and ``kinematic_share_baseline.py`` play for
their questions.

Two modelling choices, stated rather than buried:

* **Episodes are assumed uniform in clip time.**  A clip is drawn by weight and
  then a time inside it, so the share of training *frames* in clip ``c`` is
  ``w_c / sum(w)``.  Segment-anchored starts (``segment_start_prob`` 0.6) bias
  the first fraction of a second of each episode toward boundaries; over an
  episode of a few hundred steps that is a second-order correction, and it is
  the *same* convention the round-2 exposure numbers used.
* **Dwell past the final hold is excluded** (the schedule zeroes the goal there),
  which is why the accounted share is ~0.91 rather than 1.0.

Usage::

    PYTHONPATH=. python data/scripts/goal_exposure_baseline.py \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --node "L_FOOT:G@upright" --node "R_FOOT:G@upright"
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import torch


class GoalExposure(NamedTuple):
    """One (clip, hold) goal and the share of training frames that command it."""

    share: float          # fraction of all training frames whose nearest goal is this
    clip: str
    motion_id: int
    t_hold: float
    node: int
    segment_s: float


def load_corpus_weights(motion_file: str):
    """Per-motion sampling weight and length from the packaged library."""
    lib = torch.load(motion_file, map_location="cpu", weights_only=False)
    weights = lib["motion_weights"].numpy().astype(float)
    lengths = lib["motion_lengths"].numpy().astype(float)
    return weights / weights.sum(), lengths


def goal_exposures(graph: dict, p_clip, lengths, lead_s: float) -> List[GoalExposure]:
    """Share of training frames commanding each trusted hold.

    Within a clip the nearest goal is hold ``i`` for every ``t`` in
    ``[t_hold[i-1] - lead, t_hold[i] - lead)`` -- the searchsorted semantics of
    ``next_goal_indices``, written out.
    """
    out: List[GoalExposure] = []
    for clip, entry in graph["clips"].items():
        mid = entry["motion_id"]
        segments = [s for s in entry["segments"] if s.get("trusted", True)]
        if not segments:
            continue
        segments.sort(key=lambda s: s["t_hold"])
        length = float(lengths[mid])
        if length <= 0:
            continue
        bounds = [0.0] + [s["t_hold"] - lead_s for s in segments] + [length]
        for i, seg in enumerate(segments):
            lo = max(0.0, bounds[i])
            hi = min(length, bounds[i + 1])
            span = max(0.0, hi - lo)
            out.append(
                GoalExposure(
                    share=float(p_clip[mid]) * span / length,
                    clip=clip,
                    motion_id=mid,
                    t_hold=float(seg["t_hold"]),
                    node=int(seg["node"]),
                    segment_s=float(seg["t_end"] - seg["t_start"]),
                )
            )
    out.sort(key=lambda r: -r.share)
    return out


def node_exposures(rows: List[GoalExposure]) -> Dict[int, float]:
    totals: Dict[int, float] = collections.Counter()
    for row in rows:
        totals[row.node] += row.share
    return dict(totals)


def _short(clip: str) -> str:
    return clip.replace("220923_", "").replace("220926_", "")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=str, required=True,
                        help="contact_graph.json built from expert rollouts")
    parser.add_argument("--motion-file", type=str, required=True,
                        help="the packaged corpus the graph was built against")
    parser.add_argument("--lead-s", type=float, default=0.2,
                        help="ContactGraphControlConfig.min_lead_s (default 0.2)")
    parser.add_argument("--node", type=str, action="append", default=[],
                        help="node key to break down goal by goal; repeatable")
    parser.add_argument("--top", type=int, default=15,
                        help="how many nodes/goals to list")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main():
    args = create_parser().parse_args()
    graph = json.loads(Path(args.graph).read_text())
    nodes = graph["nodes"]
    p_clip, lengths = load_corpus_weights(args.motion_file)

    rows = goal_exposures(graph, p_clip, lengths, args.lead_s)
    per_node = node_exposures(rows)
    accounted = sum(r.share for r in rows)

    print(f"{len(rows)} trusted goals over {len(graph['clips'])} clips; "
          f"{accounted:.3f} of training frames carry one "
          f"(the rest is past each clip's final hold, where the goal is zeroed)\n")

    print("=== goal exposure by NODE ===")
    for nid, share in sorted(per_node.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  {share * 100:6.2f}%   node {nid:3d}  {nodes[nid]['key']}")
    thin = sum(1 for v in per_node.values() if v < 0.001)
    print(f"  ... {len(per_node)} nodes; {thin} below 0.1 %")

    key_to_id = {n["key"]: i for i, n in enumerate(nodes)}
    for key in args.node:
        nid = key_to_id.get(key)
        if nid is None:
            print(f"\n!! node key not in this graph: {key}")
            continue
        print(f"\n=== node {nid}  {key}  "
              f"({per_node.get(nid, 0.0) * 100:.2f} % of training frames) ===")
        members = [r for r in rows if r.node == nid]
        top = max((r.share for r in members), default=0.0)
        for r in members:
            ratio = f"{top / r.share:5.1f}x" if r.share > 0 else "   inf"
            print(f"  {r.share * 100:6.3f}%  ({ratio} rarer than the node's most-commanded)"
                  f"  hold@{r.t_hold:6.2f}  seg{r.segment_s:5.1f}s  {_short(r.clip)}")

    print(f"\n=== rarest {args.top} goals overall ===")
    for r in rows[-args.top:]:
        print(f"  {r.share * 100:7.4f}%  node{r.node:3d}  hold@{r.t_hold:6.2f}  "
              f"seg{r.segment_s:5.1f}s  {_short(r.clip)}")

    shares = np.array([r.share for r in rows])
    print("\n=== distribution ===")
    print(f"  median {np.median(shares) * 100:.3f} %, p10 {np.percentile(shares, 10) * 100:.4f} %, "
          f"p90 {np.percentile(shares, 90) * 100:.3f} %, max {shares.max() * 100:.2f} %")
    ordered = np.sort(shares)[::-1]
    cum = np.cumsum(ordered) / ordered.sum()
    for frac in (0.5, 0.8, 0.9):
        k = int(np.searchsorted(cum, frac)) + 1
        print(f"  top {k:3d} goals ({k / len(shares) * 100:2.0f} %) carry "
              f"{frac * 100:.0f} % of goal exposure")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"goals": [r._asdict() for r in rows],
             "per_node": {str(k): v for k, v in per_node.items()}}, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
