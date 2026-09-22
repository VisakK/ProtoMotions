# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a hold graph from a reviewed hold manifest.

``expert_revist/expert_revisit.MD`` §3.4: the goal-conditioned expert's graph has
**named holds as nodes** and the transitions between consecutive holds of a
clip as edges. It is built from the kinematic reference via the manifest
:mod:`propose_hold_manifest` proposes and a reviewer corrects (and
:mod:`make_hold_extended_clips` re-times for the extended variants) -- not from
expert rollouts, because the rollout graph records what the *experts* do, and
for the poses this round exists to fix that is exactly the wrong target (crow
recorded with a foot down).

Node identity is ``(hold name, ground contact set, trunk orientation)``. The
body-body pairs a hold carries are **not** identity -- a feet-together flag
must not fragment the standing hub -- but they are the goal: every segment's
full voted pair set goes into ``seg_contact`` (what a scheduled goal serves),
and a node's ``node_contact`` (what a manual goal serves) is its ground set plus
the body-body pairs present in at least half of its segments, so a manual crow
goal still asks for the shins on the upper arms. This is the split the shipped
graphs make with ``--node-identity ground`` and ``seg_contact``.

The output is the layout :class:`protomotions.components.contact_graph.ContactGraph`
loads unchanged, so ``ContactGraphControl`` and ``ContactGraphMotionManager``
work as they are: segments are the holds only, the transitions are the gaps
between them, and ``include_current_segment=True`` turns that into the
entry / hold / exit schedule (outside every hold: the next hold with a
countdown; inside one: that hold with ``dwell_remaining``). Node keys are
``NAME|GROUND_PAIRS@ORIENT``; ``node_id_for_key`` matches them exactly and the
existing key parsers tolerate the leading name token.

Usage::

    PYTHONPATH=. python data/scripts/build_hold_graph.py \
      --manifest data/smpl/yoga_motions_proto_yogi_expert60/holds_extended.yaml \
      --motion-file data/smpl/yoga_yogi_expert60.pt \
      --out-dir data/smpl/yoga_hold_graph_expert60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from extract_contact_configs import ORIENT_BINS, ZONE_ORDER, ZONES, zone_pairs  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure builder (unit-tested on synthetic manifests)
# --------------------------------------------------------------------------- #
def node_key(name: str, ground_pairs: list[str], orientation: str) -> str:
    ground = "|".join(sorted(ground_pairs)) if ground_pairs else "NONE"
    return f"{name}|{ground}@{orientation}"


def build_graph(
    clips: list[dict],
    motion_names: list[str],
    pair_names: list[str],
    orientation_names: list[str],
    min_lead_s: float = 0.2,
    goal_pair_vote: float = 0.5,
) -> tuple[dict, dict]:
    """``(payload for ContactGraph, json-able description)``.

    ``clips`` are manifest entries keyed by ``stem`` with a ``holds`` list;
    ``motion_names`` is the packaged library's clip order and every name must
    have a manifest entry (a graph keyed to the wrong library mislabels every
    goal silently, which is why ``ContactGraph.validate_against_motion_lib``
    exists and why this refuses to build without a one-to-one match).
    """
    by_stem = {c["stem"]: c for c in clips}
    missing = [m for m in motion_names if m not in by_stem]
    extra = [s for s in by_stem if s not in motion_names]
    if missing or extra:
        raise ValueError(
            f"manifest/library mismatch: {len(missing)} library clips without a "
            f"manifest entry {missing[:3]}, {len(extra)} manifest clips not in "
            f"the library {extra[:3]}"
        )
    pair_index = {p: i for i, p in enumerate(pair_names)}
    orient_index = {o: i for i, o in enumerate(orientation_names)}

    nodes: "OrderedDict[str, dict]" = OrderedDict()
    per_motion: list[list[dict]] = []
    edges: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for motion_id, stem in enumerate(motion_names):
        clip = by_stem[stem]
        holds = sorted(clip["holds"], key=lambda h: float(h["t_hold"]))
        segments = []
        for hold in holds:
            for pair in hold["pairs"]:
                if pair not in pair_index:
                    raise ValueError(f"{stem}: unknown contact pair '{pair}'")
            if hold["orientation"] not in orient_index:
                raise ValueError(f"{stem}: unknown orientation '{hold['orientation']}'")
            ground = sorted(p for p in hold["pairs"] if p.endswith(":G"))
            key = node_key(hold["name"], ground, hold["orientation"])
            if key not in nodes:
                nodes[key] = {
                    "key": key,
                    "name": hold["name"],
                    "pairs": ground,
                    "pair_ids": [pair_index[p] for p in ground],
                    "orientation_bin": hold["orientation"],
                    "orientation_id": orient_index[hold["orientation"]],
                    "total_dwell_s": 0.0,
                    "num_segments": 0,
                    "motions": [],
                    "source_stems": [],
                    "_body_pair_votes": defaultdict(float),
                }
            node = nodes[key]
            node_id = list(nodes.keys()).index(key)
            duration = float(hold["t_end"]) - float(hold["t_start"])
            node["total_dwell_s"] += duration
            node["num_segments"] += 1
            node["motions"].append(stem)
            source = clip.get("source_stem", stem)
            if source not in node["source_stems"]:
                node["source_stems"].append(source)
            for pair in hold["pairs"]:
                if not pair.endswith(":G"):
                    node["_body_pair_votes"][pair] += 1.0
            segments.append(
                {
                    "name": hold["name"],
                    "t_start": float(hold["t_start"]),
                    "t_end": float(hold["t_end"]),
                    "t_hold": float(hold["t_hold"]),
                    "duration_s": duration,
                    "frame_start": int(hold.get("frame_start", -1)),
                    "frame_end": int(hold.get("frame_end", -1)),
                    "frame_hold": int(hold.get("frame_hold", -1)),
                    "pairs": list(hold["pairs"]),
                    "pair_ids": [pair_index[p] for p in hold["pairs"]],
                    "orientation_bin": hold["orientation"],
                    "orientation_id": orient_index[hold["orientation"]],
                    "extend": bool(hold.get("extend", False)),
                    "hold_speed": float(hold.get("speed_at_hold", 0.0)),
                    "pelvis_z": float(hold.get("pelvis_z", 0.0)),
                    # Kinematic holds carry no rollout trust statistics; the
                    # fields exist so tools written against the rollout graph
                    # keep reading. Trust here means "reviewed manifest".
                    "trusted": True,
                    "good_fraction": 1.0,
                    "mean_track_err": 0.0,
                    "node": node_id,
                    "config": key,
                }
            )
        for k in range(1, len(segments)):
            src, dst = segments[k - 1], segments[k]
            edges[(src["node"], dst["node"])].append(
                {
                    "motion_id": motion_id,
                    "motion": stem,
                    "t": dst["t_start"],
                    "t_hold_src": src["t_hold"],
                    "t_hold_dst": dst["t_hold"],
                    "transition_s": round(dst["t_start"] - src["t_end"], 4),
                }
            )
        per_motion.append(segments)

    node_rows = []
    for key, node in nodes.items():
        votes = node.pop("_body_pair_votes")
        goal_pairs = sorted(
            p for p, v in votes.items() if v / max(node["num_segments"], 1) >= goal_pair_vote
        )
        node["goal_pairs"] = goal_pairs
        node["goal_pair_ids"] = [pair_index[p] for p in goal_pairs]
        node["goal_pair_dwell_frac"] = {
            p: round(v / max(node["num_segments"], 1), 3) for p, v in sorted(votes.items())
        }
        node["motions"] = sorted(set(node["motions"]))
        node["total_dwell_s"] = round(node["total_dwell_s"], 4)
        node_rows.append(node)

    # Extended-hold variants are the same demonstration three times over, so
    # occurrence counts overstate support; ``source_count`` is the number of
    # distinct source clips, which is what round 6's edge-count finding was
    # measured in.
    edge_rows = []
    for (s, d), occ in sorted(edges.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sources = sorted({by_stem[o["motion"]].get("source_stem", o["motion"]) for o in occ})
        edge_rows.append(
            {"src": s, "dst": d, "count": len(occ), "source_count": len(sources),
             "sources": sources, "occurrences": occ}
        )
    for node in node_rows:
        node["source_count"] = len(node["source_stems"])

    payload = pack_tensors(node_rows, per_motion, pair_names, motion_names, min_lead_s)
    description = {
        "builder": "build_hold_graph",
        "node_identity": "hold",
        "goal_pair_vote": goal_pair_vote,
        "min_lead_s": min_lead_s,
        "pair_names": list(pair_names),
        "orientation_names": list(orientation_names),
        "nodes": node_rows,
        "edges": edge_rows,
        "clips": {
            stem: {
                "motion_id": motion_id,
                "group": by_stem[stem].get("group"),
                "family": by_stem[stem].get("family"),
                "source_stem": by_stem[stem].get("source_stem", stem),
                "variant_s": by_stem[stem].get("variant_s", 0.0),
                "good_fraction": 1.0,
                "max_track_err": 0.0,
                "segments": per_motion[motion_id],
            }
            for motion_id, stem in enumerate(motion_names)
        },
    }
    return payload, description


def pack_tensors(node_rows, per_motion, pair_names, motion_names, min_lead_s):
    """Padded per-motion segment tables, in the layout ``ContactGraph`` loads."""
    num_nodes = len(node_rows)
    num_pairs = len(pair_names)
    node_contact = torch.zeros(num_nodes, num_pairs, dtype=torch.float32)
    node_orient = torch.zeros(num_nodes, dtype=torch.long)
    for i, row in enumerate(node_rows):
        node_contact[i, row["pair_ids"]] = 1.0
        node_contact[i, row["goal_pair_ids"]] = 1.0
        node_orient[i] = row["orientation_id"]

    max_segments = max((len(s) for s in per_motion), default=0)
    max_segments = max(max_segments, 1)
    num_motions = len(motion_names)
    seg_node = torch.full((num_motions, max_segments), -1, dtype=torch.long)
    seg_start = torch.full((num_motions, max_segments), float("inf"))
    seg_end = torch.full((num_motions, max_segments), float("inf"))
    seg_hold = torch.full((num_motions, max_segments), float("inf"))
    seg_count = torch.zeros(num_motions, dtype=torch.long)
    seg_contact = torch.zeros(num_motions, max_segments, num_pairs, dtype=torch.float32)
    for motion_id, segments in enumerate(per_motion):
        seg_count[motion_id] = len(segments)
        for k, segment in enumerate(segments):
            seg_node[motion_id, k] = segment["node"]
            seg_start[motion_id, k] = segment["t_start"]
            seg_end[motion_id, k] = segment["t_end"]
            seg_hold[motion_id, k] = segment["t_hold"]
            seg_contact[motion_id, k, segment["pair_ids"]] = 1.0
        if len(segments) > 1:
            holds = seg_hold[motion_id, : len(segments)]
            starts = seg_start[motion_id, : len(segments)]
            if bool((holds[1:] <= holds[:-1]).any()) or bool((starts[1:] <= starts[:-1]).any()):
                raise ValueError(
                    f"{motion_names[motion_id]}: holds are not strictly ordered in time"
                )
            ends = seg_end[motion_id, : len(segments)]
            if bool((starts[1:] <= ends[:-1]).any()):
                raise ValueError(f"{motion_names[motion_id]}: holds overlap")

    return {
        "motion_names": list(motion_names),
        "pair_names": list(pair_names),
        "orientation_names": list(ORIENT_BINS),
        "zone_order": list(ZONE_ORDER),
        "zone_bodies": {zone: list(ZONES[zone]) for zone in ZONE_ORDER},
        "node_keys": [row["key"] for row in node_rows],
        "node_names": [row["name"] for row in node_rows],
        "node_contact": node_contact,
        "node_orient": node_orient,
        "seg_node": seg_node,
        "seg_contact": seg_contact,
        "seg_start": seg_start,
        "seg_end": seg_end,
        "seg_hold": seg_hold,
        "seg_count": seg_count,
        "min_lead_s": float(min_lead_s),
    }


def summarize(description: dict) -> str:
    nodes, edges = description["nodes"], description["edges"]
    degree_out = defaultdict(int)
    degree_in = defaultdict(int)
    for e in edges:
        degree_out[e["src"]] += 1
        degree_in[e["dst"]] += 1
    singletons = sum(1 for n in nodes if n.get("source_count", n["num_segments"]) == 1)
    lines = [
        f"{len(nodes)} nodes ({singletons} from a single source clip), {len(edges)} edges "
        f"({sum(1 for e in edges if e.get('source_count', e['count']) == 1)} from a single "
        f"source clip), {len(description['clips'])} clips",
        "top nodes by dwell:",
    ]
    for i, n in sorted(enumerate(nodes), key=lambda kv: -kv[1]["total_dwell_s"])[:15]:
        lines.append(
            f"  [{i:3d}] {n['total_dwell_s']:7.1f} s  {n['num_segments']:3d} segs "
            f"({n.get('source_count', '?')} src)  out {degree_out[i]:2d} in {degree_in[i]:2d}  {n['key']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest", default="data/smpl/yoga_motions_proto_yogi_expert60/holds_extended.yaml"
    )
    parser.add_argument("--motion-file", default="data/smpl/yoga_yogi_expert60.pt")
    parser.add_argument("--out-dir", default="data/smpl/yoga_hold_graph_expert60")
    parser.add_argument("--min-lead-s", type=float, default=0.2)
    parser.add_argument("--goal-pair-vote", type=float, default=0.5)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = yaml.safe_load(open(manifest_path))
    motion_path = Path(args.motion_file)
    if not motion_path.is_absolute():
        motion_path = REPO_ROOT / motion_path
    library = torch.load(str(motion_path), map_location="cpu", weights_only=False)
    motion_names = [Path(str(f)).stem for f in library["motion_files"]]

    payload, description = build_graph(
        manifest["clips"],
        motion_names,
        list(zone_pairs()),
        list(ORIENT_BINS),
        min_lead_s=args.min_lead_s,
        goal_pair_vote=args.goal_pair_vote,
    )
    description["motion_file"] = str(motion_path)
    description["manifest"] = str(manifest_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "contact_graph.pt")
    with open(out_dir / "contact_graph.json", "w") as handle:
        json.dump(description, handle, indent=1)

    # Round-trip through the runtime loader: the layout is only right if the
    # class that consumes it accepts it.
    from protomotions.components.contact_graph import ContactGraph

    graph = ContactGraph.from_file(out_dir / "contact_graph.pt")
    graph.validate_against_motion_lib([str(f) for f in library["motion_files"]])
    print(summarize(description))
    print(f"wrote {out_dir}/contact_graph.{{pt,json}}; runtime load OK "
          f"({graph.num_nodes} nodes, {graph.num_pairs} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
