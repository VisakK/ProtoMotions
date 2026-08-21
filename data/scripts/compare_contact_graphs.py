# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two contact-graph builds of the same corpus.

Built for the load-path node rebuild (``demote_supported_pairs`` in
``build_contact_graph_from_rollouts.py``): it quantifies what a change to node
identity does to the graph the student trains against, using the measurements
the improvement plans read (``notes/Student_improvement_plan2.MD`` §4, §7).

Reported per graph, and as a delta:

* node / edge / trusted-segment counts, singleton-edge and clip-unique-edge
  shares (the "bundle of clip paths" measurements);
* the standing-state next-goal table (plan 2's P2 probe): from a plain two-feet
  stand, what configuration does the schedule pair as the next goal, and how
  much of it is the feet-touching proximity flag;
* the share of dwell whose nearest goal is inverted (the exposure number the
  corpus weights were tuned for);
* configurations that exist in one build and not the other.

Usage::

    PYTHONPATH=. python data/scripts/compare_contact_graphs.py \
      --old data/smpl/yoga_contact_graph_student44 \
      --new data/smpl/yoga_contact_graph_student44_loadpath
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

STAND_KEY = "L_FOOT:G|R_FOOT:G@upright"
STAND_FLAG_KEY = "L_FOOT:G|R_FOOT:G|L_FOOT+R_FOOT@upright"


def load(graph_dir: str) -> dict:
    return json.loads((Path(graph_dir) / "contact_graph.json").read_text())


def edge_stats(graph: dict) -> dict:
    edges = graph["edges"]
    singleton = sum(1 for e in edges if e["count"] == 1)
    clip_unique = sum(
        1 for e in edges if len({o["motion"] for o in e["occurrences"]}) == 1
    )
    return {
        "edges": len(edges),
        "singleton": singleton,
        "clip_unique": clip_unique,
    }


def trusted_segments(graph: dict):
    for clip, rec in graph["clips"].items():
        ordered = sorted(rec["segments"], key=lambda s: s["t_start"])
        yield clip, [s for s in ordered if s["trusted"]]


def standing_next_goal_table(graph: dict, top: int = 8):
    """P2: over consecutive trusted segments, what follows a plain stand?

    A stand is a segment whose config is exactly the two-feet upright set,
    with or without the feet-touch flag (so the table is comparable across
    the two identity rules).
    """
    table = Counter()
    dwell = Counter()
    total_dwell = 0.0
    transitions = 0
    for _clip, segments in trusted_segments(graph):
        for a, b in zip(segments, segments[1:]):
            if a["config"] not in (STAND_KEY, STAND_FLAG_KEY):
                continue
            table[b["config"]] += 1
            dwell[b["config"]] += a["duration_s"]
            total_dwell += a["duration_s"]
            transitions += 1
    rows = []
    for config, count in table.most_common(top):
        rows.append((dwell[config] / max(total_dwell, 1e-9), count, config))
    return rows, transitions


def inverted_goal_dwell_share(graph: dict) -> float:
    """Share of trusted dwell whose *next* segment is inverted.

    A cheap stand-in for the schedule's nearest-goal statistic: within a
    segment the nearest goal flips from the own hold to the next segment's
    hold, so the next segment's orientation is the goal orientation for the
    leave-phase of the dwell.
    """
    inverted = 0.0
    total = 0.0
    for _clip, segments in trusted_segments(graph):
        for a, b in zip(segments, segments[1:]):
            total += a["duration_s"]
            if b["orientation_bin"] == "inverted":
                inverted += a["duration_s"]
    return inverted / max(total, 1e-9)


def secondary_pair_summary(graph: dict, top: int = 6):
    counts = Counter()
    for _clip, segments in trusted_segments(graph):
        for s in segments:
            for pair in s.get("secondary_pairs", []):
                counts[pair] += 1
    return counts.most_common(top)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="baseline graph directory")
    parser.add_argument("--new", required=True, help="candidate graph directory")
    args = parser.parse_args()

    graphs = {"old": load(args.old), "new": load(args.new)}

    print(f"{'':24s} {'old':>10s} {'new':>10s}")
    for label, fn in (
        ("nodes", lambda g: len(g["nodes"])),
        ("edges", lambda g: edge_stats(g)["edges"]),
        ("singleton edges", lambda g: edge_stats(g)["singleton"]),
        ("clip-unique edges", lambda g: edge_stats(g)["clip_unique"]),
        (
            "trusted segments",
            lambda g: sum(len(s) for _c, s in trusted_segments(g)),
        ),
        (
            "inverted-goal dwell",
            lambda g: f"{100 * inverted_goal_dwell_share(g):.1f}%",
        ),
        ("rule", lambda g: g.get("supported_pair_rule", "keep")),
    ):
        print(f"{label:24s} {str(fn(graphs['old'])):>10s} {str(fn(graphs['new'])):>10s}")

    for name, graph in graphs.items():
        rows, transitions = standing_next_goal_table(graph)
        print(f"\nstanding-state next goal ({name}, {transitions} transitions):")
        for share, count, config in rows:
            marker = "  <- proximity flag" if config == STAND_FLAG_KEY else ""
            print(f"  {100 * share:5.1f}%  x{count:<3d} {config[:80]}{marker}")

    old_keys = {n["key"] for n in graphs["old"]["nodes"]}
    new_keys = {n["key"] for n in graphs["new"]["nodes"]}
    gone = sorted(old_keys - new_keys)
    born = sorted(new_keys - old_keys)
    print(f"\nconfigs only in old ({len(gone)}):")
    for key in gone[:15]:
        print(f"  {key}")
    print(f"configs only in new ({len(born)}):")
    for key in born[:15]:
        print(f"  {key}")

    secondary = secondary_pair_summary(graphs["new"])
    if secondary:
        print("\nsecondary (demoted) pairs by segment count (new):")
        for pair, count in secondary:
            print(f"  x{count:<4d} {pair}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
