# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite probe/viz plan ``config`` strings onto another graph's node keys.

A plan names its goals by node key (``"R_FOOT:G|R_HAND:G@side_r"``), so
changing what defines a node -- ``--node-identity ground`` in
``build_contact_graph_from_rollouts.py`` -- invalidates every plan whose key
carried a body-body pair.  The rewrite is mechanical (drop the body-body
tokens, keep the ground set and the orientation bin) and the point of doing it
in a script is the **validation**: every rewritten key is checked against the
target graph, and a plan that still does not resolve is reported rather than
silently skipped at render time.

Two things this deliberately does *not* touch: ``pose_clip``/``pose_time``,
because the ``--node-identity ground`` rebuild leaves segmentation and hold
times bit-identical, and the reach/hold budgets.

Usage::

    PYTHONPATH=. python data/scripts/migrate_probe_plans.py \\
      --graph data/smpl/yoga_contact_graph_student44h_bbdemoted/contact_graph.pt \\
      --plans data/scripts/plans --out data/scripts/plans_bbdemoted

    # or, after taking a copy, rewrite where the launch scripts already look:
    PYTHONPATH=. python data/scripts/migrate_probe_plans.py --in-place ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def ground_key(config: str) -> str:
    """``config`` with every body-body token dropped."""
    pairs, _, orientation = config.rpartition("@")
    if not orientation:
        return config
    kept = [token for token in pairs.split("|") if token and "+" not in token]
    return ("|".join(sorted(kept, key=lambda p: (":G" not in p, p))) or "NONE") + "@" + orientation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="target contact_graph.pt")
    parser.add_argument("--plans", default="data/scripts/plans")
    parser.add_argument("--out", default=None, help="output directory (default: alongside)")
    parser.add_argument("--in-place", action="store_true",
                        help="rewrite --plans itself, after copying it to <plans>_backup")
    args = parser.parse_args()

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    keys = set(payload["node_keys"])

    source = Path(args.plans)
    if args.in_place:
        backup = source.with_name(source.name + "_backup")
        if not backup.exists():
            shutil.copytree(source, backup)
            print(f"copied {source} -> {backup}")
        destination = source
    else:
        destination = Path(args.out or (str(source) + "_migrated"))
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    rewritten = unresolved = plans = 0
    for path in sorted(destination.rglob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "goals" not in data:
            continue
        plans += 1
        changed = False
        for goal in data["goals"]:
            if not isinstance(goal, dict) or "config" not in goal:
                continue
            config = goal["config"]
            if config in keys:
                continue
            candidate = ground_key(config)
            if candidate in keys:
                goal["config"] = candidate
                changed = True
                rewritten += 1
            else:
                unresolved += 1
                print(f"  UNRESOLVED {path.relative_to(destination)}  "
                      f"{goal.get('name')}: {config}  ->  {candidate}")
        if changed:
            path.write_text(json.dumps(data, indent=1) + "\n")

    print(f"\n{plans} plans in {destination}: {rewritten} configs rewritten, "
          f"{unresolved} still unresolved")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
