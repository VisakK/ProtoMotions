"""Card T6's record: every re-exported variant's T5 groups against its result before T6, and its seam group.

Pairs each T6 variant (``admitted_physx.json``, tags ``t6px``/``t6px12``, or ``t6rpx``/``t6rpx12`` for the landing
recipes run on the new reference with their recorded weights) with the pre-T6 variant of the same
(edge, timing, seed, pass) (``admitted_physx_pre_t6.json``, tags ``px``/``px12``; seed 4 has no pre-T6 run), and
writes ``expert_revist/graph_growth_2026_10_03/t6_exact/t6_report.json`` plus a markdown summary on stdout.

CLI: ``CUDA_VISIBLE_DEVICES= PYTHONPATH=.:data/scripts python -m edge_synthesis.t6_report``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from reference_curation import ids

PLAN_DIR = ids.REPO / "expert_revist/graph_growth_2026_10_03"
GROUPS = ("contract", "statics", "dynamics", "naturalness_p99", "endpoints", "physx_cpu")


def key(v: dict, t6: bool) -> tuple:
    """(edge, timing, seed, pass) of an admission row, from its run name."""
    run = Path(v["source_run"]).name
    m = re.match(r"^(?P<edge>[A-Z]\d)_(?:[A-Z]\d_)?(?P<timing>high|mid|low)_s(?P<seed>\d+)_(?P<tag>\w+)$", run)
    if m is None:
        raise ValueError(run)
    tag = re.sub(r"^t6r?", "", m["tag"]) if t6 else m["tag"]      # t6px, t6px12; t6rpx, t6rpx12 (reference only)
    return m["edge"], m["timing"], int(m["seed"]), tag


def variant_set(v: dict) -> str:
    """``t6`` (the T6 recipe), ``t6r`` (a landing recipe on the new reference with its recorded weights)."""
    return "t6r" if re.search(r"_t6r(px|px12)$", Path(v["source_run"]).name) else "t6"


def run_metrics(v: dict) -> dict:
    r = json.load(open(ids.REPO / v["source_run"] / "run.json"))["metrics"]
    return {"final_d6_m": r["final_d6_m"], "held": r["held"], "hand_drift_cm": r["hand_drift_cm"]}


def row(v: dict) -> dict:
    out = {"variant": v["variant"], "failed": [g for g in v["failed"] if g in GROUPS], **run_metrics(v),
           "dynamics_sat_max": v["dynamics"]["torque_saturated_share_max"],
           "statics_s_star_max": v["statics"]["s_star_max"]}
    sm = v.get("seams")
    out["legacy_seams"] = sm["legacy"] if sm else None
    if sm:
        out["seams"] = {"pass": sm["pass"], "pass_attainable": sm["pass_attainable"], "failed": sm["failed"],
                        "start_all_cm": sm["start"]["all_max_cm"], "start_planted_cm": sm["start"]["planted_max_cm"],
                        "end_planted_cm": sm["end"]["planted_max_cm"], "end_beyond": sm["end"]["beyond"],
                        "end_exemplar_bodies": sm["end"]["end_exemplar_bodies"], "drift_cm": sm["drift"]["max_cm"],
                        "drift_zones": {z: r.get("max_cm") for z, r in sm["drift"]["zones"].items()}}
    if "d1" in v:
        out["d1"] = v["d1"]
    return out


def report(t6: Path, pre: Path) -> dict:
    new = json.load(open(t6))
    old = json.load(open(pre))
    before = {}
    for v in old["variants"]:
        try:
            before[key(v, False)] = v
        except ValueError:
            continue
    rows = []
    for v in new["variants"]:
        k = key(v, True)
        b = before.get(k)
        legacy_b = None
        if b is not None:
            import torch

            from edge_synthesis import seams as SM
            from edge_synthesis import sketch as SK
            mot = torch.load(ids.REPO / b["motion"], map_location="cpu", weights_only=False)
            legacy_b = SM.legacy(mot, SK.edge(SK.load_edges(), b["edge"]))
        rows.append({"edge": k[0], "timing": k[1], "seed": k[2], "pass": k[3], "set": variant_set(v), "t6": row(v),
                     "before": None if b is None else {**{x: y for x, y in row(b).items() if x != "legacy_seams"},
                                                       "legacy_seams": legacy_b}})
    by = defaultdict(list)
    for r in rows:
        by[(r["edge"], r["timing"], r["pass"], r["set"])].append(r)
    summary = {}
    for (edge, timing, p, vs), rs in sorted(by.items()):
        t = [r["t6"] for r in rs]
        summary[f"{edge} {timing} {p}" + (" (reference only)" if vs == "t6r" else "")] = {
            "seeds": len(t), "held": sum(x["held"] for x in t),
            "d1_selected": sum(bool(x.get("d1", {}).get("selected")) for x in t),
            "seams_pass": sum(x["seams"]["pass"] for x in t),
            "seams_pass_attainable": sum(x["seams"]["pass_attainable"] for x in t),
            "seam_failures": sorted({f for x in t for f in x["seams"]["failed"]}),
            "start_planted_cm_max": max(x["seams"]["start_planted_cm"] for x in t),
            "start_all_cm_max": max(x["seams"]["start_all_cm"] for x in t),
            "end_planted_cm": [x["seams"]["end_planted_cm"] for x in t],
            "drift_cm": [x["seams"]["drift_cm"] for x in t],
            "final_d6_m": [x["final_d6_m"] for x in t],
            "before_final_d6_m": [r["before"]["final_d6_m"] for r in rs if r["before"]],
            "before_hands_start_cm": [r["before"]["legacy_seams"]["hands_start_cm"] for r in rs if r["before"]],
            "legacy_hands_start_cm": [x["legacy_seams"]["hands_start_cm"] for x in t],
            "before_hands_end_cm": [r["before"]["legacy_seams"]["hands_end_cm"] for r in rs if r["before"]],
            "legacy_hands_end_cm": [x["legacy_seams"]["hands_end_cm"] for x in t],
            "before_feet_end_cm": [r["before"]["legacy_seams"]["feet_end_cm"] for r in rs if r["before"]],
            "legacy_feet_end_cm": [x["legacy_seams"]["feet_end_cm"] for x in t],
            "groups_changed": sorted({g for r in rs if r["before"]
                                      for g in set(r["t6"]["failed"]) ^ set(r["before"]["failed"])})}
    return {**ids.provenance(1, "edge_synthesis.t6_report", __file__, [t6, pre]), "kind": "t6_report",
            "summary": summary, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--t6", type=Path, default=PLAN_DIR / "admitted_physx.json")
    ap.add_argument("--pre", type=Path, default=PLAN_DIR / "admitted_physx_pre_t6.json")
    ap.add_argument("--out", type=Path, default=PLAN_DIR / "t6_exact/t6_report.json")
    args = ap.parse_args(argv)
    rec = report(args.t6, args.pre)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("| Recipe | Held | Seams (literal / attainable) | Seam failures | D1 selected | Final 6-body (m) | before T6 |")
    print("|---|---|---|---|---|---|---|")
    for k, s in rec["summary"].items():
        print(f"| {k} | {s['held']}/{s['seeds']} | {s['seams_pass']} / {s['seams_pass_attainable']} | "
              f"{', '.join(s['seam_failures']) or '-'} | {s['d1_selected']} | "
              f"{', '.join(f'{x:.3f}' for x in s['final_d6_m'])} | {', '.join(f'{x:.3f}' for x in s['before_final_d6_m'])} |")
    print("wrote", ids.display_path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
