# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price a change of ``--body-pair-identity`` before training on it.

Round 7_1 demoted every body-body pair out of node identity so that a marginal
press stops fragmenting a pose into singleton nodes.  The gain is edge
consolidation -- the measured predictor of probe feasibility (round 6: >=6
observations -> clean, 4-5 -> attempts and falls, 1 -> never attempts) -- and
the cost is node degeneracy, because poses that were separated only by a
body-body pair now share a node.  Both are computable from the two graph JSONs
without a GPU, and this prints them side by side.

The goal vector is *not* part of that trade: it is served per segment
(``seg_contact``), pinned to the load-path definition, so the coarsening never
takes a body-body pair out of what a scheduled goal can ask for.  The last
section verifies that, and it is the check that must pass before training.

Usage::

    PYTHONPATH=. python data/scripts/compare_graph_identity_rules.py \\
      --before data/smpl/yoga_contact_graph_student44h \\
      --after  data/smpl/yoga_contact_graph_student44h_bbdemoted
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def load(directory: str) -> tuple[dict, dict]:
    path = Path(directory)
    graph = json.loads((path / "contact_graph.json").read_text())
    tensors = torch.load(path / "contact_graph.pt", map_location="cpu", weights_only=False)
    return graph, tensors


def node_stats(graph: dict) -> dict:
    nodes = graph["nodes"]
    counts = [e["count"] for e in graph["edges"]]
    dwell = sum(n["total_dwell_s"] for n in nodes)
    segments = [s for c in graph["clips"].values() for s in c["segments"] if s["trusted"]]
    return {
        "rule": graph.get("body_pair_identity", graph.get("supported_pair_rule", "?")),
        "nodes": len(nodes),
        "singleton nodes": sum(1 for n in nodes if n["num_segments"] == 1),
        "singleton dwell s": round(
            sum(n["total_dwell_s"] for n in nodes if n["num_segments"] == 1), 1
        ),
        "total dwell s": round(dwell, 1),
        "trusted segments": len(segments),
        "median segment s": round(
            sorted(s["duration_s"] for s in segments)[len(segments) // 2], 2
        ),
        "edges": len(counts),
        "edges seen once": sum(1 for c in counts if c == 1),
        "edges seen 4+": sum(1 for c in counts if c >= 4),
        "edges seen 6+": sum(1 for c in counts if c >= 6),
        "edge observations": sum(counts),
    }


def clips_per_node(graph: dict) -> dict[int, set]:
    out: dict[int, set] = defaultdict(set)
    for name, clip in graph["clips"].items():
        for segment in clip["segments"]:
            if segment["trusted"]:
                out[segment["node"]].add(name)
    return out


def short(name: str) -> str:
    return (
        name.replace("220923_", "")
        .replace("220926_", "")
        .replace("hold_t", "H")[:30]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="graph directory, old rule")
    parser.add_argument("--after", required=True, help="graph directory, new rule")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    before, before_t = load(args.before)
    after, after_t = load(args.after)

    a, b = node_stats(before), node_stats(after)
    print("=" * 78)
    print("SHAPE OF THE GRAPH")
    print("=" * 78)
    print(f"  {'':22s}{'before':>12}{'after':>12}{'change':>12}")
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, str):
            print(f"  {key:22s}{va:>12}{vb:>12}")
            continue
        delta = vb - va
        print(f"  {key:22s}{va:>12}{vb:>12}{delta:>+12}")

    print()
    print("=" * 78)
    print("WHAT MERGED: nodes that became one, and the clips now sharing them")
    print("=" * 78)
    after_keys = {n["key"]: i for i, n in enumerate(after["nodes"])}
    before_clips, after_clips = clips_per_node(before), clips_per_node(after)
    merged = []
    for i, node in enumerate(after["nodes"]):
        sources = [
            j for j, old in enumerate(before["nodes"])
            if tuple(sorted(p for p in old["pairs"] if p.endswith(":G"))) ==
               tuple(sorted(p for p in node["pairs"] if p.endswith(":G")))
            and old["orientation_bin"] == node["orientation_bin"]
        ]
        if len(sources) > 1:
            merged.append((node["total_dwell_s"], i, node, sources))
    merged.sort(reverse=True, key=lambda r: r[0])
    print(f"  {len(merged)} of {len(after['nodes'])} nodes absorbed more than one "
          f"predecessor.\n")
    for dwell, i, node, sources in merged[: args.top]:
        gained = sorted(after_clips[i] - set().union(
            *[before_clips[j] for j in sources[:1]] or [set()]
        ))
        print(f"  {node['key'][:64]:64s} {dwell:6.1f}s  {len(sources)} -> 1")
        print(f"      clips now sharing this node: "
              f"{', '.join(short(c) for c in sorted(after_clips[i]))[:150]}")
    print()
    print("=" * 78)
    print("THE CHECK THAT MUST PASS: the goal did not lose the body-body half")
    print("=" * 78)
    for label, tensors, graph in (("before", before_t, before), ("after", after_t, after)):
        contact = tensors.get("seg_contact")
        pairs = tensors["pair_names"]
        body = torch.tensor([1.0 if "+" in p else 0.0 for p in pairs])
        counts = tensors["seg_count"]
        if contact is None:
            node_contact = tensors["node_contact"]
            contact = node_contact[tensors["seg_node"].clamp(min=0)]
        rows = torch.cat([contact[m, : int(counts[m])] for m in range(len(counts))])
        with_body = int(((rows * body).sum(dim=-1) > 0).sum())
        print(f"  {label:6s}: {with_body} of {rows.shape[0]} scheduled goals name at "
              f"least one body-body pair "
              f"({100 * with_body / max(rows.shape[0], 1):.1f} %); "
              f"{int((rows * body).sum())} body-body slots set in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
