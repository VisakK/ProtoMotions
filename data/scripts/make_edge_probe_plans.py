# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the probe plans of the synthesised edges (card R4a of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``).

Release v3 (``reference_curation.release_v3``) calls this in its plans step, after ``make_hold_graph_probe_plans.py``.
Per admitted edge (S -> D) it writes three plans in the format that generator writes (``start`` and ``pose_clip``
name clips; ``config`` names a node by its graph key):

* ``edge_<edge>.json`` -- start in the plan clip at S's ``t_hold``; hold S (``reach_s`` 0.5, ``hold_s`` 3, the dwell
  plans' convention); then D, its pose from the plan clip at D's ``t_hold``, with ``reach_s`` = T + 0.5 s (T the
  variant's transition time); hold D 3 s.
* ``nohijack_<edge>.json`` -- the same start and the same first goal; then the S clip's own next hold (Crow -a's
  standing @971, Handstand -a's @1632), its pose from the S clip, with ``reach_s`` = that clip's own transition time
  (S's ``t_end`` to the next hold's ``t_start``) + 0.5 s; hold 3 s. With the edge plan it is the hub causality pair:
  one state, two commands that differ only in the second goal.
* ``fork_edge_<edge>.json`` -- start in the S clip at t = 0 (its standing start); S at the S clip's own entry time
  (``make_hold_graph_probe_plans``' fork rule: S's ``t_start`` minus the opening hold's ``t_end``, clamped to 1-10 s),
  pose from the plan clip, held 3 s; then D as in the edge plan.

The panel (``run_expert_graph_ft.sh``) runs ``edge_*`` and ``nohijack_*``; the forks go to the funnel battery.

**The plan clip** of an edge is one spliced clip, so every pose of a plan comes from one world frame: the spliced clip
is in S's clip frame (its lead-in and S's exemplar are S's clip, untransformed), so a pose from the S clip is in that
frame too. Among the edge's variants in the release it is the slowest (largest T), then the one with the lowest final
6-body error (``admission.row.endpoints.final_d6_m``), then the first in the release's order.

    PYTHONPATH=. python data/scripts/make_edge_probe_plans.py --graph <release>/contact_graph.json \\
        --manifest <release>/holds_extended.yaml --out-dir <release>/plans
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EDGE_ORDER = ("E1", "E3", "B1", "E2", "E5")
HOLD_S = 3.0
FIRST_REACH_S = 0.5
REACH_MARGIN_S = 0.5
MAX_REACH_S = 10.0           # make_hold_graph_probe_plans.family_plans' fork cap


def plan_clips(clips: list[dict]) -> dict:
    """``{edge: x0 entry}`` of each edge's plan clip (module doc), from extended-manifest entries."""
    by_edge: dict = {}
    for i, c in enumerate(clips):
        syn = c.get("synthetic")
        if syn is None or float(c.get("variant_s", 0.0)) != 0.0:
            continue
        key = (-float(syn["T"]), float(syn["admission"]["row"]["endpoints"]["final_d6_m"]), i)
        if syn["edge"] not in by_edge or key < by_edge[syn["edge"]][0]:
            by_edge[syn["edge"]] = (key, c)
    return {e: c for e, (_, c) in sorted(by_edge.items(), key=lambda kv: EDGE_ORDER.index(kv[0]))}


def _segment(graph: dict, stem: str, hold_id: str) -> dict:
    for s in graph["clips"][stem]["segments"]:
        if s["hold_id"] == hold_id:
            return s
    raise KeyError(f"{stem} has no segment {hold_id}")


def _goal(graph: dict, seg: dict, pose_clip: str, reach_s: float, hold_s: float = HOLD_S) -> dict:
    return {"name": seg["name"][:24], "config": graph["nodes"][seg["node"]]["key"], "pose_clip": pose_clip,
            "pose_time": seg["t_hold"], "reach_s": round(float(reach_s), 2), "hold_s": float(hold_s)}


def edge_plans(graph: dict, clips: list[dict]) -> tuple[dict, dict]:
    """``({plan_name: plan}, {edge: plan clip stem})`` (pure). ``graph`` is the release's ``contact_graph.json``,
    ``clips`` its extended manifest's entries."""
    plans, picks = {}, {}
    for edge, clip in plan_clips(clips).items():
        stem, syn = clip["stem"], clip["synthetic"]
        s_hold = next(h for h in clip["holds"] if h["inherits"] == syn["S"]["hold_id"] and h.get("extend"))
        d_hold = next(h for h in clip["holds"] if h["inherits"] == syn["D"]["hold_id"]
                      and h["frame_start"] > s_hold["frame_end"])
        s_seg, d_seg = _segment(graph, stem, s_hold["hold_id"]), _segment(graph, stem, d_hold["hold_id"])
        # the S clip: S's own segment, its opening, and its next hold
        s_clip = syn["S"]["stem"]
        segments = graph["clips"][s_clip]["segments"]
        k = next(i for i, s in enumerate(segments) if s["hold_id"] == syn["S"]["hold_id"])
        opening = segments[0] if segments and segments[0]["t_start"] == 0.0 else None
        entry = min(max(segments[k]["t_start"] - (opening["t_end"] if opening else 0.0), 1.0), MAX_REACH_S)
        if k + 1 >= len(segments):
            raise ValueError(f"{s_clip}: S {syn['S']['hold_id']} is its last hold; no next hold for no-hijack")
        nxt = segments[k + 1]
        first = _goal(graph, s_seg, stem, FIRST_REACH_S)
        to_d = _goal(graph, d_seg, stem, float(syn["T"]) + REACH_MARGIN_S)
        own = _goal(graph, nxt, s_clip, nxt["t_start"] - segments[k]["t_end"] + REACH_MARGIN_S)
        label = f"{edge} ({syn['label']})" if syn.get("label") else edge
        source = "Generated by data/scripts/make_edge_probe_plans.py (PLAN.MD R4a)."
        plans[f"edge_{edge}"] = {
            "_comment": [f"Edge test for {label}: placed at S in {stem}, hold S {HOLD_S:.0f} s, then D within",
                         f"T + {REACH_MARGIN_S} s = {to_d['reach_s']} s and hold it {HOLD_S:.0f} s.", source],
            "start": {"clip": stem, "time": s_seg["t_hold"]}, "goals": [first, to_d]}
        plans[f"nohijack_{edge}"] = {
            "_comment": [f"No-hijack test for {label}: the edge plan's start and first goal, then the S clip's own next",
                         f"hold ({nxt['hold_id']}) within its own transition + {REACH_MARGIN_S} s. Paired with edge_{edge}.",
                         source],
            "start": {"clip": stem, "time": s_seg["t_hold"]}, "goals": [first, own]}
        plans[f"fork_edge_{edge}"] = {
            "_comment": [f"Fork test for {label}: from {s_clip}'s start (t = 0), enter S in that clip's own entry time,",
                         f"hold it {HOLD_S:.0f} s, then D as in edge_{edge}. Poses from {stem} (S's clip frame).", source],
            "start": {"clip": s_clip, "time": 0.0},
            "goals": [_goal(graph, s_seg, stem, entry), to_d]}
        picks[edge] = stem
    return plans, picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True, help="the release's contact_graph.json")
    parser.add_argument("--manifest", required=True, help="the release's holds_extended.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    graph = json.load(open(args.graph))
    clips = yaml.safe_load(open(args.manifest))["clips"]
    plans, picks = edge_plans(graph, clips)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, plan in plans.items():
        path = out / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"{path} exists: an edge plan would overwrite another plan")
        path.write_text(json.dumps(plan, indent=2))
    print(f"wrote {len(plans)} edge plans for {len(picks)} edges to {out}: " +
          ", ".join(f"{e} <- {s}" for e, s in picks.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
