# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One table pairing each rendered probe with its replica rate.

A render is **one draw**. The v9 plank render holds the pose cleanly (hold-window
error 0.061 m) while the panel says only 3-25 % of replicas do; the v9 standing
render settles 0.69 m away and the panel says 83 % of replicas eventually come
back. Reading either alone is how this project has repeatedly drawn the wrong
conclusion, so this joins them:

* from ``score_probe_pose.py``'s ``<label>.posescore.json`` — what the rendered
  draw actually did, and which corpus pose it ended up nearest;
* from ``run_sequence_panel.py``'s ``summary.json`` — how often that happens,
  over ~36 replicas of the same plan and checkpoint.

Usage::

    PYTHONPATH=. python data/scripts/join_render_and_panel.py \\
      --renders output/renderings/v9_probe --prefix v9last_ \\
      --panel output/v9_tier0/base/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--arrive", type=float, default=0.15)
    args = parser.parse_args()

    panel = {
        r["sequence"]: r
        for r in json.loads(Path(args.panel).read_text())
    }
    rows = []
    for path in sorted(Path(args.renders).glob(f"{args.prefix}*.posescore.json")):
        score = json.loads(path.read_text())
        name = score["label"]
        seq = panel.get(name, {})
        goals = score.get("goals", [])
        if not goals:
            continue
        final = goals[-1]
        rows.append({
            "probe": name,
            "goals": len(goals),
            "render_best": final.get("pose_err_best"),
            "render_hold": final.get("pose_err_hold_mean"),
            "render_settled_dist": final.get("pose_err_settled"),
            "settled_near": final.get("settled_nearest_corpus"),
            "settled_near_m": final.get("settled_nearest_corpus_dist"),
            "panel_rate": seq.get("hold_success_rate"),
            "panel_p50": seq.get("final_goal_pose_err_p50"),
            "replicas": seq.get("replicas"),
        })

    if not rows:
        raise SystemExit(f"no {args.prefix}*.posescore.json under {args.renders}")

    header = (
        f"{'probe':34s} {'g':>2} {'render best':>11} {'render hold':>11} "
        f"{'panel rate':>10} {'panel p50':>9} {'n':>4}   settled nearest"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["panel_rate"] is None, r["panel_rate"] or 0)):
        near = ""
        if r["render_hold"] is not None and r["render_hold"] > args.arrive:
            near = (
                f"{r['settled_near_m']:.2f} m from "
                f"{str(r['settled_near'])[8:44]}"
                if r["settled_near"] else ""
            )
        print(
            f"{r['probe'][:34]:34s} {r['goals']:2d} {_f(r['render_best']):>11} "
            f"{_f(r['render_hold']):>11} {_f(r['panel_rate']):>10} "
            f"{_f(r['panel_p50']):>9} {str(r['replicas'] or '-'):>4}   {near}"
        )
    print("-" * len(header))
    agree = sum(
        1 for r in rows
        if r["render_hold"] is not None and r["panel_rate"] is not None
        and ((r["render_hold"] < args.arrive) == (r["panel_rate"] >= 0.5))
    )
    scored = sum(
        1 for r in rows
        if r["render_hold"] is not None and r["panel_rate"] is not None
    )
    print(f"\nthe rendered draw agrees with the majority of replicas on "
          f"{agree}/{scored} probes.")
    print("A disagreement is not an error: the render is one nucleus draw, and "
          "§11.2 puts the per-plan reproducibility at about +/-0.15.")
    return 0


def _f(value) -> str:
    return "   -  " if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
