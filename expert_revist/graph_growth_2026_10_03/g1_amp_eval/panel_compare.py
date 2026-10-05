"""The goal-conditioned viz panel: e15500 (epochs 13,000-15,500) against every G1 panel (500-5,000).

Each plan is commanded to 341 replicas (``results/<run>/viz/epoch_*/summary.json``). Per goal it reports the hold rate
(the share of replicas within 0.15 m over the goal's window), the reach rate, the hold-window pose error (p50 over
replicas) and the longest contiguous hold. Writes ``data/panel.json`` and prints README §6's tables.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g1_amp_eval/panel_compare.py
"""

from __future__ import annotations

import ast
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {"e15500": ROOT / "results/smpl_yogi_v2_expert56_a2dda5d2ac/viz",
        "g1": ROOT / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/viz"}
E15_FROM = 13000


def load(folder: Path, min_epoch: int = 0) -> dict[int, dict[str, dict]]:
    out = {}
    for f in glob.glob(str(folder / "epoch_*/summary.json")):
        epoch = int(re.search(r"epoch_(\d+)", f).group(1))
        if epoch < min_epoch:
            continue
        plans = {}
        for p in json.load(open(f)):
            agg = p.get("per_goal_agg")
            if isinstance(agg, str):
                agg = ast.literal_eval(agg)
            goals = []
            for g in agg or []:
                goals.append(dict(goal=g["goal"], hold_rate=g.get("hold_rate"), reach_rate=g.get("reach_rate"),
                                  hold_err=g.get("hold_pose_err_p50"), best_err=g.get("best_pose_err_p50"),
                                  held_s=g.get("time_held_s_p50"), hold_iou=g.get("hold_iou_p50")))
            plans[p["sequence"]] = dict(replicas=p.get("replicas"), hold_success_rate=p.get("hold_success_rate"),
                                        min_goal_hold_rate=p.get("min_goal_hold_rate"),
                                        final_err_p50=p.get("final_goal_pose_err_p50"), goals=goals)
        out[epoch] = plans
    return dict(sorted(out.items()))


def main() -> None:
    e15, g1 = load(RUNS["e15500"], E15_FROM), load(RUNS["g1"])
    (HERE / "data/panel.json").write_text(json.dumps({"e15500": e15, "g1": g1}, indent=1) + "\n")
    plans = list(g1[max(g1)])
    print("final-goal hold rate per plan (share of 341 replicas):")
    print(f"{'plan':48s} " + " ".join(f"{'e'+str(e//100):>6s}" for e in e15) + " | "
          + " ".join(f"{'g'+str(e//100):>6s}" for e in g1))
    for p in plans:
        a = [e15[e].get(p, {}).get("hold_success_rate") for e in e15]
        b = [g1[e].get(p, {}).get("hold_success_rate") for e in g1]
        fmt = lambda v: f"{v:6.2f}" if v is not None else f"{'-':>6s}"  # noqa: E731
        print(f"{p[:48]:48s} " + " ".join(fmt(v) for v in a) + " | " + " ".join(fmt(v) for v in b))

    last_e15, last_g1 = max(e15), max(g1)
    print(f"\nper goal, e15500 @{last_e15} vs G1 @{last_g1}: hold rate / reach rate / hold-window error p50 / held s p50")
    for p in plans:
        print(p)
        ga = {g["goal"]: g for g in e15[last_e15].get(p, {}).get("goals", [])}
        for g in g1[last_g1][p]["goals"]:
            a = ga.get(g["goal"], {})
            fa = lambda k, f="{:.2f}": f.format(a[k]) if a.get(k) is not None else "-"  # noqa: E731
            print(f"    {g['goal'][:26]:26s} e15500 {fa('hold_rate')} / {fa('reach_rate')} / {fa('hold_err', '{:.3f}')} m"
                  f" / {fa('held_s', '{:.1f}')} s   G1 {g['hold_rate']:.2f} / {g['reach_rate']:.2f} / "
                  f"{g['hold_err']:.3f} m / {g['held_s']:.1f} s")


if __name__ == "__main__":
    main()
