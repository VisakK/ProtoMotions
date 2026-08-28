# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive feasible goal-sequence probe plans from a contact graph.

Every emitted plan is *feasible by construction* in the graph's own terms —
each hop is something the graph actually observed, with the provenance written
into the plan's ``_comment``. Three families:

* **chains** — a clip's own trusted hold sequence (the most in-distribution
  multi-goal command there is: the schedule trained on exactly this order, at
  roughly this pacing). Clips are ranked by how many distinct nodes their
  chain visits; ``hold_*`` clips are excluded (single-node schedules).
* **trips** — standing -> X -> standing, where both edges (0 -> X, X -> 0)
  were observed. This is the cross-clip entry/exit question: can the student
  reach a configuration from a stand and come back, outside any single clip's
  arc. X ranked by node dwell.
* **walks** — standing -> A -> B -> standing with all three hops observed
  edges; the 2-hop compositional version of a trip. Ranked by the weakest
  hop's observation count.

Pose exemplars come from the graph's own hold frames (longest trusted segment
of the node), the standing exemplar from the same clip when it has one. All
selection is deterministic (pure sorts, no randomness).

Known scoring trap, stamped into every plan that touches it: standing-family
goals cannot be judged by contact IoU (the documented degeneracy) — judge
those on the video / pose, not the IoU column.

Usage::

    PYTHONPATH=. python data/scripts/make_graph_probe_plans.py \
        --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \
        --out-dir data/scripts/plans/v6_probe \
        --num-chains 8 --num-trips 6 --num-walks 3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STANDING_KEY = "L_FOOT:G|R_FOOT:G@upright"

ZONE_ABBREV = {
    "L_FOOT": "LF", "R_FOOT": "RF", "L_HAND": "LH", "R_HAND": "RH",
    "L_FOREARM": "LFA", "R_FOREARM": "RFA", "L_SHANK": "LS", "R_SHANK": "RS",
    "L_THIGH": "LT", "R_THIGH": "RT", "L_UPPER_ARM": "LUA", "R_UPPER_ARM": "RUA",
    "HEAD": "HD", "PELVIS": "PV", "TORSO": "TO",
}


def slug(key: str) -> str:
    """Short filesystem-safe name for a node key."""
    pairs, orient = key.rsplit("@", 1)
    parts = []
    for pair in pairs.split("|"):
        if pair.endswith(":G"):
            parts.append(ZONE_ABBREV.get(pair[:-2], pair[:-2]))
        else:
            a, b = pair.split("+")
            parts.append(f"{ZONE_ABBREV.get(a, a)}x{ZONE_ABBREV.get(b, b)}")
    return "_".join(parts) + "_" + orient[:4]


def clip_family(stem: str) -> str:
    """Pose-family key for diversity dedup: strip date prefix and take suffix.

    Take suffixes come both as ``_-a`` and bare ``-a`` (e.g.
    ``...Koundinyanasana_I_and_II-b``), so the dash alone is the marker.
    """
    stem = re.sub(r"^\d+_", "", stem)
    stem = re.sub(r"-[a-z]$", "", stem)
    return stem.rstrip("_")


def load_graph(path: Path):
    graph = json.loads(path.read_text())
    node_of_key = {n["key"]: i for i, n in enumerate(graph["nodes"])}
    if STANDING_KEY not in node_of_key:
        raise SystemExit(f"graph has no standing node '{STANDING_KEY}'")
    return graph, node_of_key


def trusted_segments(graph):
    """Per-node list of (duration, clip, seg) trusted segments, plus per-clip."""
    by_node = defaultdict(list)
    by_clip = {}
    for clip, entry in graph["clips"].items():
        segs = [s for s in entry["segments"] if s.get("trusted")]
        by_clip[clip] = segs
        for seg in segs:
            by_node[seg["node"]].append((seg["duration_s"], clip, seg))
    for rows in by_node.values():
        rows.sort(key=lambda r: -r[0])
    return by_node, by_clip


def best_exemplar(by_node, node):
    """Longest trusted segment of a node -> (clip, seg)."""
    rows = by_node.get(node)
    if not rows:
        return None
    _, clip, seg = rows[0]
    return clip, seg


def standing_exemplar(by_clip, by_node, standing_node, prefer_clip):
    """A standing hold frame, from ``prefer_clip`` when it has one."""
    for seg in by_clip.get(prefer_clip, []):
        if seg["node"] == standing_node:
            return prefer_clip, seg
    return best_exemplar(by_node, standing_node)


def clamp(value, low, high):
    return max(low, min(high, value))


STANDING_NOTE = (
    "standing-family goals share the plain two-feet contact set - the IoU "
    "column is vacuous on them by the documented degeneracy; judge on video."
)


def build_chain_plan(clip, segs):
    goals = []
    prev_hold = None
    for seg in segs[:6]:
        reach = (seg["t_hold"] - prev_hold) if prev_hold is not None else max(
            seg["t_hold"] - seg["t_start"], 0.5
        )
        goals.append(
            {
                "name": f"g{len(goals)}_{slug(seg['config'])}"[:40],
                "config": seg["config"],
                "pose_clip": clip,
                "pose_time": round(seg["t_hold"], 2),
                "reach_s": round(clamp(reach, 1.0, 3.5), 2),
                "hold_s": round(clamp(seg["t_end"] - seg["t_hold"], 0.8, 3.0), 2),
            }
        )
        prev_hold = seg["t_hold"]
    return {
        "_comment": [
            f"GRAPH-DERIVED chain: {clip}'s own trusted hold sequence "
            f"({len(goals)} goals) - in-distribution by construction.",
            STANDING_NOTE,
        ],
        "start": {"clip": clip, "time": round(max(segs[0]["t_start"] - 0.3, 0.03), 2)},
        "goals": goals,
    }


def build_trip_plan(graph, by_clip, by_node, standing_node, node, counts):
    key = graph["nodes"][node]["key"]
    clip_x, seg_x = best_exemplar(by_node, node)
    clip_s, seg_s = standing_exemplar(by_clip, by_node, standing_node, clip_x)
    out_count, back_count = counts
    return {
        "_comment": [
            f"GRAPH-DERIVED round trip: standing -> {key} -> standing. "
            f"Observed edges: standing->X x{out_count}, X->standing x{back_count}. "
            f"Exemplar: {clip_x} @ {seg_x['t_hold']:.2f}s "
            f"(dwell {graph['nodes'][node]['total_dwell_s']:.1f}s over corpus).",
            STANDING_NOTE,
        ],
        "start": {"clip": clip_s, "time": round(max(seg_s["t_start"], 0.03), 2)},
        "goals": [
            {
                "name": "stand",
                "config": STANDING_KEY,
                "pose_clip": clip_s,
                "pose_time": round(seg_s["t_hold"], 2),
                "reach_s": 1.5,
                "hold_s": 1.5,
            },
            {
                "name": f"reach_{slug(key)}"[:40],
                "config": key,
                "pose_clip": clip_x,
                "pose_time": round(seg_x["t_hold"], 2),
                "reach_s": 2.5,
                "hold_s": 3.0,
            },
            {
                "name": "stand_up",
                "config": STANDING_KEY,
                "pose_clip": clip_s,
                "pose_time": round(seg_s["t_hold"], 2),
                "reach_s": 2.5,
                "hold_s": 2.0,
            },
        ],
    }


def build_walk_plan(graph, by_clip, by_node, standing_node, a, b, counts):
    key_a = graph["nodes"][a]["key"]
    key_b = graph["nodes"][b]["key"]
    clip_a, seg_a = best_exemplar(by_node, a)
    clip_b, seg_b = best_exemplar(by_node, b)
    clip_s, seg_s = standing_exemplar(by_clip, by_node, standing_node, clip_a)
    return {
        "_comment": [
            f"GRAPH-DERIVED 2-hop walk: standing -> {key_a} -> {key_b} -> "
            f"standing. Observed edge counts: {counts[0]} / {counts[1]} / "
            f"{counts[2]}. Exemplars: {clip_a} @ {seg_a['t_hold']:.2f}s, "
            f"{clip_b} @ {seg_b['t_hold']:.2f}s.",
            STANDING_NOTE,
        ],
        "start": {"clip": clip_s, "time": round(max(seg_s["t_start"], 0.03), 2)},
        "goals": [
            {
                "name": "stand",
                "config": STANDING_KEY,
                "pose_clip": clip_s,
                "pose_time": round(seg_s["t_hold"], 2),
                "reach_s": 1.5,
                "hold_s": 1.0,
            },
            {
                "name": f"a_{slug(key_a)}"[:40],
                "config": key_a,
                "pose_clip": clip_a,
                "pose_time": round(seg_a["t_hold"], 2),
                "reach_s": 2.5,
                "hold_s": 2.5,
            },
            {
                "name": f"b_{slug(key_b)}"[:40],
                "config": key_b,
                "pose_clip": clip_b,
                "pose_time": round(seg_b["t_hold"], 2),
                "reach_s": 2.5,
                "hold_s": 2.5,
            },
            {
                "name": "stand_up",
                "config": STANDING_KEY,
                "pose_clip": clip_s,
                "pose_time": round(seg_s["t_hold"], 2),
                "reach_s": 2.5,
                "hold_s": 2.0,
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-chains", type=int, default=8)
    parser.add_argument("--num-trips", type=int, default=6)
    parser.add_argument("--num-walks", type=int, default=3)
    args = parser.parse_args()

    graph, node_of_key = load_graph(Path(args.graph))
    standing_node = node_of_key[STANDING_KEY]
    by_node, by_clip = trusted_segments(graph)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    # ---------------- chains ---------------- #
    candidates = []
    for clip, segs in by_clip.items():
        if clip.startswith("hold_") or len(segs) < 4:
            continue
        distinct = len({s["node"] for s in segs})
        if distinct < 3:
            continue
        span = segs[-1]["t_end"] - segs[0]["t_start"]
        candidates.append((distinct, span, clip, segs))
    candidates.sort(key=lambda r: (-r[0], -r[1], r[2]))
    seen_families, chains = set(), []
    for distinct, span, clip, segs in candidates:
        family = clip_family(clip)
        if family in seen_families:
            continue
        seen_families.add(family)
        chains.append((clip, segs, distinct))
        if len(chains) >= args.num_chains:
            break
    for clip, segs, distinct in chains:
        take = clip[-1] if re.search(r"-[a-z]$", clip) else "x"
        name = f"chain_{clip_family(clip)[:26].lower()}_{take}"
        plan = build_chain_plan(clip, segs)
        (out_dir / f"{name}.json").write_text(json.dumps(plan, indent=1))
        manifest.append(
            {"name": name, "type": "chain", "hops": len(plan["goals"]),
             "distinct_nodes": distinct, "source": clip}
        )

    # ---------------- edges index ---------------- #
    edge_count = {}
    for edge in graph["edges"]:
        edge_count[(edge["src"], edge["dst"])] = edge["count"]

    # ---------------- trips ---------------- #
    trip_nodes = []
    for node_id, node in enumerate(graph["nodes"]):
        if node_id == standing_node:
            continue
        out_c = edge_count.get((standing_node, node_id))
        back_c = edge_count.get((node_id, standing_node))
        if out_c and back_c and by_node.get(node_id):
            trip_nodes.append((node["total_dwell_s"], node_id, out_c, back_c))
    trip_nodes.sort(key=lambda r: -r[0])
    picked_trip_nodes = []
    for dwell, node_id, out_c, back_c in trip_nodes[: args.num_trips]:
        picked_trip_nodes.append(node_id)
        key = graph["nodes"][node_id]["key"]
        name = f"trip_{slug(key)[:30].lower()}"
        plan = build_trip_plan(
            graph, by_clip, by_node, standing_node, node_id, (out_c, back_c)
        )
        (out_dir / f"{name}.json").write_text(json.dumps(plan, indent=1))
        manifest.append(
            {"name": name, "type": "trip", "node": key,
             "edges": f"{out_c}/{back_c}", "dwell_s": round(dwell, 1)}
        )

    # ---------------- walks ---------------- #
    walk_candidates = []
    for (a, b), count_ab in edge_count.items():
        if a in (standing_node,) or b in (standing_node,) or a == b:
            continue
        count_sa = edge_count.get((standing_node, a))
        count_bs = edge_count.get((b, standing_node))
        if not (count_sa and count_bs):
            continue
        if not (by_node.get(a) and by_node.get(b)):
            continue
        # Diversity: prefer walks over nodes the trips did not already cover.
        overlap = int(a in picked_trip_nodes) + int(b in picked_trip_nodes)
        weakest = min(count_sa, count_ab, count_bs)
        dwell = (
            graph["nodes"][a]["total_dwell_s"] + graph["nodes"][b]["total_dwell_s"]
        )
        walk_candidates.append(
            (overlap, -weakest, -dwell, a, b, (count_sa, count_ab, count_bs))
        )
    walk_candidates.sort()
    used_pairs = set()
    walks_done = 0
    for overlap, _, _, a, b, counts in walk_candidates:
        if walks_done >= args.num_walks:
            break
        if (a, b) in used_pairs or (b, a) in used_pairs:
            continue
        used_pairs.add((a, b))
        key_a, key_b = graph["nodes"][a]["key"], graph["nodes"][b]["key"]
        name = f"walk_{slug(key_a)[:14].lower()}__{slug(key_b)[:14].lower()}"
        plan = build_walk_plan(graph, by_clip, by_node, standing_node, a, b, counts)
        (out_dir / f"{name}.json").write_text(json.dumps(plan, indent=1))
        manifest.append(
            {"name": name, "type": "walk", "path": f"{key_a} -> {key_b}",
             "edges": "/".join(map(str, counts))}
        )
        walks_done += 1

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} plans to {out_dir}")
    for row in manifest:
        print(" ", json.dumps(row))


if __name__ == "__main__":
    main()
