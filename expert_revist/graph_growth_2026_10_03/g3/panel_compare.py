"""The goal-conditioned panel: G3's six training panels (500-3,000) and the standalone epoch-3,420 panels.

Training panels command each of 22 plans to 186 replicas (``results/<run>/viz/epoch_*/summary.json``), with the stock
heading-chart pose error. At epoch 3,420 the same 20 plans plus the five ``fork_edge_*`` plans ran standalone at
163 replicas, twice: with the stock metric (``panel_e3420_heading``) and with the best-yaw metric
(``panel_e3420_bestyaw``; ``run_panel_bestyaw.py``), which is what tells a missed pose from a chart artefact at
chaturanga. G1's panels (``g1_amp_eval/data/panel.json``) are the reference for the plans both runs share.

Per goal: hold rate (share of replicas within 0.15 m over the goal's hold window), reach rate, hold-window pose
error p50, contact IoU p50 and the longest contiguous hold. Writes ``data/panel.json``.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/panel_compare.py
"""

from __future__ import annotations

import ast
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_VIZ = ROOT / "results/smpl_yogi_v2_expert56_g3_2f132f4299/viz"
OUT = ROOT / "output/renderings/expert56_v2_g3_e3420"
STANDALONE = {"3420h": OUT / "panel_e3420_heading/summary.json", "3420b": OUT / "panel_e3420_bestyaw/summary.json"}
G1_PANEL = HERE.parent / "g1_amp_eval/data/panel.json"


def plans_of(path: Path) -> dict[str, dict]:
    plans = {}
    for p in json.load(open(path)):
        agg = p.get("per_goal_agg")
        if isinstance(agg, str):
            agg = ast.literal_eval(agg)
        goals = [dict(goal=g["goal"], hold_rate=g.get("hold_rate"), reach_rate=g.get("reach_rate"),
                      hold_err=g.get("hold_pose_err_p50"), best_err=g.get("best_pose_err_p50"),
                      held_s=g.get("time_held_s_p50"), hold_iou=g.get("hold_iou_p50")) for g in agg or []]
        plans[p["sequence"]] = dict(replicas=p.get("replicas"), hold_success_rate=p.get("hold_success_rate"),
                                    reached_exact_rate=p.get("reached_exact_rate"), goals=goals)
    return plans


def main() -> int:
    panels = {}
    for f in sorted(glob.glob(str(RUN_VIZ / "epoch_*/summary.json"))):
        panels[str(int(re.search(r"epoch_(\d+)", f).group(1)))] = plans_of(Path(f))
    for k, p in STANDALONE.items():
        if p.exists():
            panels[k] = plans_of(p)
    g1 = json.load(open(G1_PANEL))["g1"]
    out = {"g3": panels, "g1": g1, "columns": list(panels)}
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/panel.json").write_text(json.dumps(out, indent=1) + "\n")

    names = sorted({n for p in panels.values() for n in p}, key=lambda n: (n.split("_")[0], n))
    cols = list(panels)
    print("hold rate of each goal (share of replicas), G3 panels; 3420h = standalone, heading chart; "
          "3420b = standalone, best yaw; G1 = its epoch-5,000 panel")
    print(f"{'plan / goal':52s} {'G1 5000':>7s} " + " ".join(f"{c:>6s}" for c in cols))
    for n in names:
        first = next(p[n] for p in panels.values() if n in p)
        for gi, g in enumerate(first["goals"]):
            vals = []
            for c in cols:
                pl = panels[c].get(n)
                vals.append(f"{pl['goals'][gi]['hold_rate']:6.2f}" if pl and gi < len(pl["goals"]) else f"{'-':>6s}")
            g1v = g1.get("5000", {}).get(n)
            g1s = f"{g1v['goals'][gi]['hold_rate']:7.2f}" if g1v and gi < len(g1v["goals"]) else f"{'-':>7s}"
            print(f"{(n[:30] + ' / ' + g['goal'][:18]):52s} {g1s} " + " ".join(vals))
    print(f"\n-> {HERE / 'data/panel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
