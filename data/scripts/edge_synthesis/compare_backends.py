"""The same edges executed by MPPI on the MuJoCo plant (``edge_mppi``) and on the PhysX plant (``edge_mppi_physx``),
side by side on ``edge_mppi.evaluate``'s metrics (computed the same way on both).

A row is an (edge, timing) recipe; per backend it lists the seeds' rate of holding D and the medians over seeds of:
the final 6-body error to D, the mean body error to the tracked sketch, the planted hands' drift, the peak landing
load, the worst joint's torque-saturated share, the box excess and the largest per-body speed / corpus p99.

CLI: ``PYTHONPATH=.:data/scripts python -m edge_synthesis.compare_backends [--out <json>]``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from reference_curation import ids

MJ_RUNS = ids.REPO / "output/edge_synthesis/runs"
PX_RUNS = ids.REPO / "output/edge_synthesis/physx/runs"
# the MuJoCo recipe runs lane T admitted from or reported (t_edges/README.MD); tags fix the recipe
MJ_RECIPES = {
    ("E1", "E1_high"): r"^E1_E1_high_s\d_refff$", ("E1", "E1_mid"): r"^E1_E1_mid_s\d_refff$",
    ("E3", "E3_high"): r"^E3_E3_high_s0_refff$", ("E3", "E3_mid"): r"^E3_E3_mid_s\d_refff$",
    ("E4", "E4_high"): r"^E4_E4_high_s\d_refff$", ("E4", "E4_mid"): r"^E4_E4_mid_s\d_refff$",
    ("B2", "B2_mid"): r"^B2_B2_mid_s\d_refff$",
    ("E2", "mid"): r"^E2_mid_s\d_(land2|v2w)$", ("E5", "mid"): r"^E5_mid_s\d_v3$",
    ("B1", "mid"): r"^B1_mid_s\d_(land2|v4)$", ("B1", "high"): r"^B1_high_s0_v4$",
}


def _metrics(run_dir: Path) -> dict:
    r = json.load(open(run_dir / "run.json"))
    m = r["metrics"]
    land = max(m.get("landing_peak_bw", {}).values(), default=0.0)
    sat = max(m.get("torque_saturated_share", {}).values(), default=0.0)
    spd = max(m.get("speed_over_p99", {}).values(), default=0.0)
    timing = r["timing"] if isinstance(r["timing"], str) else Path(r.get("reference") or "x").name
    return {"run": run_dir.name, "edge": r["edge"], "timing": timing, "seed": r["seed"], "held": bool(m["held"]),
            "final_d6_m": m["final_d6_m"], "sketch_err_cm": m.get("sketch_body_err_mean_cm"),
            "hand_drift_cm": m.get("hand_drift_cm"), "landing_peak_bw": land, "sat_max": sat,
            "box_max_deg": m.get("box_excess_max_deg"), "speed_over_p99_max": spd, "wall_s": r.get("wall_s"),
            "exec_spread_mm_max": r.get("exec_spread_mm_max")}


def _summary(rows: list) -> dict:
    if not rows:
        return {}

    def med(k):
        v = [x[k] for x in rows if x.get(k) is not None]
        return round(float(np.median(v)), 4) if v else None
    return {"seeds": len(rows), "held": sum(x["held"] for x in rows),
            **{k: med(k) for k in ("final_d6_m", "sketch_err_cm", "hand_drift_cm", "landing_peak_bw", "sat_max",
                                    "box_max_deg", "speed_over_p99_max")},
            "best_final_d6_m": min(x["final_d6_m"] for x in rows), "runs": [x["run"] for x in rows]}


def compare(px_tag: str = "px") -> dict:
    px = defaultdict(list)
    for d in sorted(PX_RUNS.glob("*/run.json")):
        if not d.parent.name.endswith(f"_{px_tag}"):
            continue
        m = _metrics(d.parent)
        px[(m["edge"], m["timing"])].append(m)
    mj = defaultdict(list)
    for (edge, timing), pat in MJ_RECIPES.items():
        for d in sorted(MJ_RUNS.glob("*/run.json")):
            if re.match(pat, d.parent.name):
                mj[(edge, timing)].append(_metrics(d.parent))
    out = {}
    for key in sorted(set(px) | set(mj)):
        out[f"{key[0]} {key[1]}"] = {"mujoco": _summary(mj.get(key, [])), "physx": _summary(px.get(key, []))}
    return out


def markdown(table: dict) -> str:
    cols = ("held", "final_d6_m", "sketch_err_cm", "hand_drift_cm", "landing_peak_bw", "sat_max", "box_max_deg",
            "speed_over_p99_max")
    lines = ["| Edge, timing | Backend | Seeds | " + " | ".join(cols[1:]) + " |",
             "|---|---|---|" + "---|" * (len(cols) - 1)]
    for k, v in table.items():
        for be in ("mujoco", "physx"):
            s = v[be]
            if not s:
                continue
            lines.append(f"| {k} | {be} | {s['held']}/{s['seeds']} held | " +
                         " | ".join("" if s[c] is None else f"{s[c]:g}" for c in cols[1:]) + " |")
    return "\n".join(lines)


# D1 (PLAN.MD, Step 3 runbook): which T5 groups stay hard under each admission policy
POLICIES = {"strict": ("contract", "statics", "dynamics", "naturalness_p99", "endpoints", "physx_cpu"),
            "naturalness_advisory": ("contract", "statics", "dynamics", "endpoints", "physx_cpu"),
            "default": ("contract", "statics", "endpoints", "physx_cpu")}


def admission_policies(admitted: Path) -> dict:
    """Per D1 policy, the variants an ``admit.py`` record admits (every hard group passes)."""
    rec = json.load(open(admitted))
    out = {}
    for pol, hard in POLICIES.items():
        ok = [v["variant"] for v in rec["variants"] if not set(v["failed"]) & set(hard)]
        edges = sorted({v["edge"] for v in rec["variants"] if v["variant"] in ok})
        out[pol] = {"variants": ok, "count": len(ok), "edges": edges}
    out["failed_groups"] = {v["variant"]: v["failed"] for v in rec["variants"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="px")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--admitted", type=Path, default=None, help="an admit.py record: summarise it under D1's policies")
    args = ap.parse_args(argv)
    if args.admitted:
        pol = admission_policies(args.admitted)
        for k in POLICIES:
            print(f"{k}: {pol[k]['count']} variants, edges {pol[k]['edges']}")
        return 0
    t = compare(args.tag)
    print(markdown(t))
    if args.out:
        args.out.write_text(json.dumps(t, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
