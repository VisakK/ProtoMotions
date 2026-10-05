# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fork plans on training's deadline: every plan mirrors one clip's own holds (S1 of
``expert_revist/graph_growth_2026_10_03/PLAN.MD``).

The release's fork plans (``make_hold_graph_probe_plans.py``) command a family hold straight from the clip's
standing start, with ``reach_s`` measured from the opening hold's end to the family segment's *start* and a 4 s
hold. Training's deadline runs to the hold *frame*, through the clip's intermediate holds, and each hold lasts its
own ``t_end - t_hold``: Warrior III is asked for in 3.25 s where training reaches its frame 12.9 s after the
opening hold, through two intermediate holds (S0, ``s0_x0/README.MD`` §4). That generator is one of the release's
hash-pinned builders, so this is a separate one.

**A route plan** starts in its clip at t = 0 and commands the clip's own segments in order, through the target
hold and the next ``--exit-holds`` holds after it. Segment ``k`` becomes the goal

    reach_s = t_hold[k] - t_end[k - 1]   (t_end[-1] = 0)
    hold_s  = t_end[k] - t_hold[k]

so goal ``k``'s reach ends at ``t_hold[k]`` and the goal ends at ``t_end[k]`` on the clip's own clock: under the
interval schedule a segment is slot 0 from the previous segment's end to its own, and the panel's
``timing='training'`` then serves exactly the deadline and dwell channels the scheduled slot carried at that clip
time. Poses are the clip's own exemplars (``pose_time = t_hold``), nodes are named by graph key.

* ``fork_<family>.json``: the family's exemplar clip and hold, chosen by the release's fork rule (the longest
  family-named segment among clips that are their own unextended source), then the next hold;
* ``route_edge_<edge>.json``: each synthetic edge's plan clip (``make_edge_probe_plans.plan_clips``), from its own
  t = 0 through D's hold and the next hold. It commands the expert's own training clip.

Usage::

    PYTHONPATH=. python data/scripts/make_route_probe_plans.py \\
        --graph data/smpl/reference_curation/<release>/contact_graph.json \\
        --manifest data/smpl/reference_curation/<release>/holds_extended.yaml \\
        --out-dir data/scripts/plans_release_v3_route
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR = "data/scripts/make_route_probe_plans.py"


def _slug(text: str, limit: int = 40) -> str:
    """The release fork plans' file-name rule (``make_hold_graph_probe_plans._slug``)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug[:limit].strip("_")


def route_goals(graph: dict, stem: str, target: int, exit_holds: int = 1) -> List[dict]:
    """The goals that mirror ``stem``'s segments ``0 .. target + exit_holds`` (module docstring)."""
    segments = graph["clips"][stem]["segments"]
    if not 0 <= target < len(segments):
        raise IndexError(f"{stem} has {len(segments)} segments, not {target + 1}")
    goals, previous_end = [], 0.0
    for k in range(min(target + 1 + max(exit_holds, 0), len(segments))):
        seg = segments[k]
        reach = float(seg["t_hold"]) - previous_end
        hold = float(seg["t_end"]) - float(seg["t_hold"])
        if reach < -1e-6 or hold < -1e-6:
            raise ValueError(f"{stem} segment {k}: hold frame outside its window or segments overlap")
        goals.append({
            "name": seg["name"][:24],
            "config": graph["nodes"][seg["node"]]["key"],
            "pose_clip": stem,
            "pose_time": seg["t_hold"],
            "reach_s": round(max(reach, 0.0), 4),
            "hold_s": round(max(hold, 0.0), 4),
        })
        previous_end = float(seg["t_end"])
    return goals


def family_exemplars(graph: dict) -> Dict[str, Tuple[str, int]]:
    """``{family: (stem, segment index)}`` by the release's fork rule (``make_hold_graph_probe_plans.family_plans``)."""
    best: Dict[str, Tuple[float, str, int]] = {}
    for stem, clip in graph["clips"].items():
        if float(clip.get("variant_s", 0.0)) != 0.0 or clip.get("source_stem", stem) != stem:
            continue
        family = clip.get("family") or stem
        for k, seg in enumerate(clip["segments"]):
            if seg["name"] != family:
                continue
            if family not in best or seg["duration_s"] > best[family][0]:
                best[family] = (seg["duration_s"], stem, k)
    return {family: (stem, k) for family, (_d, stem, k) in sorted(best.items())}


def _plan(start: str, goals: List[dict], comment: List[str], target: int) -> dict:
    """``target`` names the scored goal (the family hold, or D): its index, node key and hold window on the plan's
    clock, ``[t_hold, t_end]``. A scorer that drops empty hold windows (``hold_s`` 0) shifts goal positions, so
    it matches the key and window instead. The panel ignores the field."""
    end = sum(g["reach_s"] + g["hold_s"] for g in goals[: target + 1])
    return {"_comment": comment, "start": {"clip": start, "time": 0.0}, "goals": goals,
            "target": {"goal": target, "config": goals[target]["config"],
                       "window_s": [round(end - goals[target]["hold_s"], 4), round(end, 4)]}}


def route_plans(graph: dict, clips: Optional[list] = None, exit_holds: int = 1) -> Dict[str, dict]:
    """``{plan_name: plan}`` (pure). ``graph`` is a release's ``contact_graph.json``; ``clips`` its extended
    manifest's entries (for the synthetic edges' plan clips), or None to write the family forks only."""
    plans: Dict[str, dict] = {}
    for family, (stem, k) in family_exemplars(graph).items():
        goals = route_goals(graph, stem, k, exit_holds)
        seg = graph["clips"][stem]["segments"][k]
        plans[f"fork_{_slug(family)}"] = _plan(stem, goals, [
            f"Fork test for {family} on training's deadline: {stem} from t = 0 through its own holds to the",
            f"family hold (frame {seg['t_hold']} s), each held its own t_end - t_hold, then {exit_holds} more.",
            f"Run with timing='training'. Generated by {GENERATOR} (S1).",
        ], target=k)
    if clips is not None:
        from make_edge_probe_plans import plan_clips

        for edge, clip in plan_clips(clips).items():
            stem, syn = clip["stem"], clip["synthetic"]
            # D's hold in the plan clip, by make_edge_probe_plans.edge_plans' own rule.
            s_hold = next(h for h in clip["holds"] if h["inherits"] == syn["S"]["hold_id"] and h.get("extend"))
            d_hold = next(h for h in clip["holds"] if h["inherits"] == syn["D"]["hold_id"]
                          and h["frame_start"] > s_hold["frame_end"])
            segments = graph["clips"][stem]["segments"]
            d_index = next(i for i, s in enumerate(segments) if s["hold_id"] == d_hold["hold_id"])
            goals = route_goals(graph, stem, d_index, exit_holds)
            plans[f"route_edge_{edge}"] = _plan(stem, goals, [
                f"Edge {edge} on its own training clip {stem}: from t = 0 through S and D, each hold its own",
                f"t_end - t_hold, then {exit_holds} more. Run with timing='training'. Generated by {GENERATOR} (S1).",
            ], target=d_index)
    return plans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True, help="the release's contact_graph.json")
    parser.add_argument("--manifest", default=None,
                        help="the release's holds_extended.yaml; given, the route_edge_* plans are written too")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exit-holds", type=int, default=1)
    args = parser.parse_args()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    graph = json.loads(resolve(args.graph).read_text())
    clips = None
    if args.manifest:
        import yaml

        clips = yaml.safe_load(open(resolve(args.manifest)))["clips"]
    plans = route_plans(graph, clips, exit_holds=args.exit_holds)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, plan in plans.items():
        (out_dir / f"{name}.json").write_text(json.dumps(plan, indent=2))
    lengths = {n: round(sum(g["reach_s"] + g["hold_s"] for g in p["goals"]), 2) for n, p in plans.items()}
    longest = max(lengths, key=lengths.get)
    print(f"wrote {len(plans)} plans to {out_dir}; longest {longest} {lengths[longest]} s "
          f"(the panel's max_seconds must cover it, or its last goals are dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
