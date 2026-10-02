# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hold graph v2: the hold graph keyed on stable hold ids (BUILD_PLAN Step 9, BodyFix Step 5).

``build_hold_graph.py`` (v1) built the expert60 graphs. v2 keeps its layout, its node identity
(``NAME|GROUND@ORIENT``) and its schedule semantics, so ``ContactGraph``, ``ContactGraphControl`` and
``ContactGraphMotionManager`` load it unchanged. It changes four things:

1. **Stable ids.** Every segment carries the ``hold_id`` of its hold: ``<stem>@<x0 exemplar frame>``,
   labels v1's stable id (it never follows a moved exemplar, and the three duration variants of a hold
   share it). The payload adds ``hold_ids`` (sorted) and ``seg_hold_index [M, S]`` (-1 on padding).
   Node ids are the sorted order of the node keys, so they depend on the set of configurations, not on
   the clip order (v1 numbered nodes by first appearance).
2. **Nodes aggregate unique source holds, not variant votes.** v1's ``node_contact`` was the ground set
   plus every body-body pair present in at least half of the node's *segments*, so the three variants of
   one clip voted three times, and a node holding two side-specific holds (Side Crow -c's left and right
   shelf) served a union or neither. v2's ``node_contact`` is the ground set plus the body-body pairs
   **every** source hold of the node commands; a pair only some of them command is recorded as a
   conflict in the description (``pair_votes``, ``pair_conflicts``), never merged.
3. **Manual goals resolve the segment.** ``manual_goal_rule: "segment"`` tells ``ContactGraph`` to serve
   a manual goal (probe plans, the viz panel) the contact set of the segment that holds its pose --
   ``(pose clip, pose time)`` inside ``[t_start, t_end]`` -- when that segment belongs to the requested
   node, and the node's consensus set otherwise. So a commanded Side Crow -c hold 2 asks for the right
   shelf, as its scheduled goal does.
4. **Identity.** ``graph_version`` 2, ``fps``, ``motion_num_frames`` and the sha256 of the library and
   manifest it was built from. It refuses a manifest/library mismatch in names, frame counts or fps, a
   hold without a ``hold_id``, and a hold id whose variants disagree on name, contact set or orientation
   (the target topology must be identical across duration variants).

The description (``contact_graph.json``) keeps v1's fields, so ``make_hold_graph_probe_plans.py`` and the
viz panel read it unchanged, and adds the per-segment ``hold_id`` and the per-node source holds.

Usage::

    PYTHONPATH=. python data/scripts/build_hold_graph_v2.py \\
      --manifest <release>/holds_extended.yaml --motion-file <release>/motions.pt --out-dir <release>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from build_hold_graph import node_key, pack_tensors, summarize  # noqa: E402
from extract_contact_configs import ORIENT_BINS, zone_pairs  # noqa: E402

GRAPH_VERSION = 2


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_graph_v2(
    clips: list[dict],
    motion_names: list[str],
    pair_names: list[str],
    orientation_names: list[str],
    motion_num_frames: list[int] | None = None,
    fps: int = 60,
    min_lead_s: float = 0.2,
) -> tuple[dict, dict]:
    """``(payload for ContactGraph, json-able description)``; pure, unit-tested on synthetic manifests.

    ``clips`` are extended-manifest entries keyed by ``stem`` (every hold with a ``hold_id``),
    ``motion_names`` the packaged library's clip order and ``motion_num_frames`` its frame counts.
    """
    by_stem = {c["stem"]: c for c in clips}
    missing = [m for m in motion_names if m not in by_stem]
    extra = [s for s in by_stem if s not in motion_names]
    if missing or extra:
        raise ValueError(
            f"manifest/library mismatch: {len(missing)} library clips without a manifest entry "
            f"{missing[:3]}, {len(extra)} manifest clips not in the library {extra[:3]}"
        )
    if motion_num_frames is not None:
        if len(motion_num_frames) != len(motion_names):
            raise ValueError("motion_num_frames does not match motion_names")
        wrong = [(s, by_stem[s].get("num_frames"), int(n)) for s, n in zip(motion_names, motion_num_frames)
                 if int(by_stem[s].get("num_frames", -1)) != int(n)]
        if wrong:
            raise ValueError(f"manifest frame counts differ from the library: {wrong[:3]}")
    wrong_fps = [s for s in motion_names if int(by_stem[s].get("fps", fps)) != int(fps)]
    if wrong_fps:
        raise ValueError(f"clips not at {fps} fps: {wrong_fps[:3]}")
    pair_index = {p: i for i, p in enumerate(pair_names)}
    orient_index = {o: i for i, o in enumerate(orientation_names)}

    # ---- pass 1: segments, node keys, the hold each segment is --------------------------- #
    per_motion_raw: list[list[dict]] = []
    hold_topology: dict[str, tuple] = {}
    for stem in motion_names:
        clip = by_stem[stem]
        holds = sorted(clip["holds"], key=lambda h: float(h["t_hold"]))
        segments = []
        for hold in holds:
            hid = hold.get("hold_id")
            if not hid:
                raise ValueError(f"{stem}: a hold at t_hold {hold.get('t_hold')} has no hold_id")
            for pair in hold["pairs"]:
                if pair not in pair_index:
                    raise ValueError(f"{stem}: unknown contact pair '{pair}'")
            if hold["orientation"] not in orient_index:
                raise ValueError(f"{stem}: unknown orientation '{hold['orientation']}'")
            ground = sorted(p for p in hold["pairs"] if p.endswith(":G"))
            topology = (hold["name"], tuple(sorted(hold["pairs"])), hold["orientation"])
            if hold_topology.setdefault(hid, topology) != topology:
                raise ValueError(
                    f"hold {hid}: its duration variants disagree on (name, contacts, orientation): "
                    f"{hold_topology[hid]} vs {topology}"
                )
            segments.append({"hold": hold, "hold_id": hid, "key": node_key(hold["name"], ground, hold["orientation"]),
                             "ground": ground})
        per_motion_raw.append(segments)

    # ---- nodes: sorted keys, aggregated over unique source holds -------------------------- #
    keys = sorted({s["key"] for segs in per_motion_raw for s in segs})
    node_id = {k: i for i, k in enumerate(keys)}
    nodes = {
        k: {"key": k, "total_dwell_s": 0.0, "num_segments": 0, "motions": [], "source_stems": [],
            "_holds": {}}
        for k in keys
    }
    per_motion: list[list[dict]] = []
    edges: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for motion_id, (stem, raw) in enumerate(zip(motion_names, per_motion_raw)):
        clip = by_stem[stem]
        source = clip.get("source_stem", stem)
        segments = []
        for s in raw:
            hold, key = s["hold"], s["key"]
            node = nodes[key]
            node.setdefault("name", hold["name"])
            node.setdefault("pairs", s["ground"])
            node.setdefault("orientation_bin", hold["orientation"])
            duration = float(hold["t_end"]) - float(hold["t_start"])
            node["total_dwell_s"] += duration
            node["num_segments"] += 1
            node["motions"].append(stem)
            if source not in node["source_stems"]:
                node["source_stems"].append(source)
            node["_holds"][s["hold_id"]] = sorted(p for p in hold["pairs"] if not p.endswith(":G"))
            segments.append({
                "name": hold["name"],
                "hold_id": s["hold_id"],
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
                # A reviewed, gated release: every segment is trusted. The fields exist so tools
                # written against the rollout graph keep reading.
                "trusted": True,
                "good_fraction": 1.0,
                "mean_track_err": 0.0,
                "node": node_id[key],
                "config": key,
            })
        for k in range(1, len(segments)):
            src, dst = segments[k - 1], segments[k]
            edges[(src["node"], dst["node"])].append({
                "motion_id": motion_id, "motion": stem, "t": dst["t_start"],
                "t_hold_src": src["t_hold"], "t_hold_dst": dst["t_hold"],
                "hold_src": src["hold_id"], "hold_dst": dst["hold_id"],
                "transition_s": round(dst["t_start"] - src["t_end"], 4),
            })
        per_motion.append(segments)

    node_rows = []
    for key in keys:
        node = nodes[key]
        holds = node.pop("_holds")
        votes = Counter(p for pairs in holds.values() for p in pairs)
        consensus = sorted(p for p, v in votes.items() if v == len(holds))
        node["pair_ids"] = [pair_index[p] for p in node["pairs"]]
        node["orientation_id"] = orient_index[node["orientation_bin"]]
        node["source_holds"] = sorted(holds)
        node["num_source_holds"] = len(holds)
        node["source_count"] = len(node["source_stems"])
        node["goal_pairs"] = consensus
        node["goal_pair_ids"] = [pair_index[p] for p in consensus]
        node["pair_votes"] = {p: votes[p] for p in sorted(votes)}
        node["pair_conflicts"] = {p: sorted(h for h, pairs in holds.items() if p in pairs)
                                  for p in sorted(votes) if votes[p] < len(holds)}
        node["motions"] = sorted(set(node["motions"]))
        node["total_dwell_s"] = round(node["total_dwell_s"], 4)
        node_rows.append(node)

    edge_rows = []
    for (s, d), occ in sorted(edges.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sources = sorted({by_stem[o["motion"]].get("source_stem", o["motion"]) for o in occ})
        hold_pairs = sorted({(o["hold_src"], o["hold_dst"]) for o in occ})
        edge_rows.append({"src": s, "dst": d, "count": len(occ), "source_count": len(sources),
                          "source_hold_pairs": len(hold_pairs), "sources": sources, "occurrences": occ})

    payload = pack_tensors(node_rows, per_motion, pair_names, motion_names, min_lead_s)
    if payload["orientation_names"] != list(orientation_names):
        raise ValueError("orientation vocabulary differs from the extractor's ORIENT_BINS")
    hold_ids = sorted({s["hold_id"] for segs in per_motion for s in segs})
    hold_index = {h: i for i, h in enumerate(hold_ids)}
    seg_hold_index = torch.full(payload["seg_node"].shape, -1, dtype=torch.long)
    for motion_id, segments in enumerate(per_motion):
        for k, segment in enumerate(segments):
            seg_hold_index[motion_id, k] = hold_index[segment["hold_id"]]
    payload.update(
        graph_version=GRAPH_VERSION,
        manual_goal_rule="segment",
        hold_ids=hold_ids,
        seg_hold_index=seg_hold_index,
        fps=int(fps),
        motion_num_frames=(torch.tensor([int(n) for n in motion_num_frames], dtype=torch.long)
                           if motion_num_frames is not None else None),
    )
    description = {
        "builder": "build_hold_graph_v2",
        "graph_version": GRAPH_VERSION,
        "node_identity": "hold",
        "node_order": "sorted keys",
        "node_contact": "ground set + body-body pairs every source hold of the node commands",
        "manual_goal_rule": "segment",
        "min_lead_s": min_lead_s,
        "fps": int(fps),
        "pair_names": list(pair_names),
        "orientation_names": list(orientation_names),
        "hold_ids": hold_ids,
        "nodes": node_rows,
        "edges": edge_rows,
        "clips": {
            stem: {
                "motion_id": motion_id,
                "group": by_stem[stem].get("group"),
                "family": by_stem[stem].get("family"),
                "source_stem": by_stem[stem].get("source_stem", stem),
                "variant_s": by_stem[stem].get("variant_s", 0.0),
                "num_frames": int(by_stem[stem].get("num_frames", -1)),
                "good_fraction": 1.0,
                "max_track_err": 0.0,
                "segments": per_motion[motion_id],
            }
            for motion_id, stem in enumerate(motion_names)
        },
    }
    return payload, description


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", required=True, help="holds_extended.yaml of the library")
    parser.add_argument("--motion-file", required=True, help="the packaged library (.pt)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-lead-s", type=float, default=0.2)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    motion_path = Path(args.motion_file).resolve()
    manifest = yaml.safe_load(open(manifest_path))
    library = torch.load(str(motion_path), map_location="cpu", weights_only=False)
    motion_names = [Path(str(f)).stem for f in library["motion_files"]]
    num_frames = [int(n) for n in library["motion_num_frames"]]
    fps_values = {round(1.0 / float(dt)) for dt in library["motion_dt"]}
    if len(fps_values) != 1:
        raise ValueError(f"the library mixes frame rates {sorted(fps_values)}")
    fps = fps_values.pop()

    payload, description = build_graph_v2(
        manifest["clips"], motion_names, list(zone_pairs()), list(ORIENT_BINS),
        motion_num_frames=num_frames, fps=fps, min_lead_s=args.min_lead_s,
    )
    identity = {"package_sha256": _sha256(motion_path), "manifest_sha256": _sha256(manifest_path),
                "plant_sha256": library.get("plant_sha256")}
    payload.update(identity)
    description.update(identity, motion_file=str(motion_path), manifest=str(manifest_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "contact_graph.pt")
    with open(out_dir / "contact_graph.json", "w") as handle:
        json.dump(description, handle, indent=1)

    # Round trip through the runtime loader, against the library it was built for.
    from protomotions.components.contact_graph import ContactGraph

    graph = ContactGraph.from_file(out_dir / "contact_graph.pt")
    graph.validate_against_motion_lib([str(f) for f in library["motion_files"]],
                                      motion_num_frames=num_frames, fps=fps)
    if graph.graph_version != GRAPH_VERSION or graph.manual_goal_rule != "segment":
        raise RuntimeError("the runtime loader did not read the v2 fields")
    print(summarize(description))
    conflicts = sum(1 for n in description["nodes"] if n["pair_conflicts"])
    print(f"wrote {out_dir}/contact_graph.{{pt,json}}; runtime load OK ({graph.num_nodes} nodes, "
          f"{len(payload['hold_ids'])} holds, {graph.num_pairs} pairs, {conflicts} nodes with side-specific "
          f"pair conflicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
