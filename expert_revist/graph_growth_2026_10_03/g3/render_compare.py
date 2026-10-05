"""Do the renders show what the evaluator measured? Each render against the standalone evaluation's own rollout.

A render is a fresh one-env PhysX rollout of the same checkpoint (``render_review.py``), so it can differ from the
evaluator's 4,096-env rollout where the outcome depends on the simulation itself (G1's review: 16 of 20 within
5 mm). Per x0 motion this compares the rendered rollout (``part_*/render_summary.json``, ``analyze_rollouts_v2.py
--render-dir``) with the evaluator's (``timelines/library_e3420.json``, ``--library``): mean body error, tracked
share and the first time past the 0.5 m gate. Writes ``data/render_compare.json`` and prints the clips that diverge.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/render_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "output/renderings/expert56_v2_g3_e3420"
DIVERGE_MM = 5.0


def main() -> int:
    lib = {c["stem"]: c for c in json.load(open(OUT / "timelines/library_e3420.json"))["clips"]}
    rows = []
    for part in ("part_a", "part_b"):
        for c in json.load(open(OUT / part / "render_summary.json"))["clips"]:
            L = lib[c["stem"]]
            rows.append({"stem": c["stem"], "label": c["label"], "group": c["group"],
                         "render_mean_err": c["mean_err"], "eval_mean_err": L["mean_err"],
                         "diff_mm": round(1000 * abs(c["mean_err"] - L["mean_err"]), 1),
                         "render_tracked": c["tracked_share"], "eval_tracked": L["tracked_share"],
                         "render_first_gate_s": c["first_termination_s"], "eval_first_gate_s": L["first_termination_s"],
                         "recorded_s": c["recorded_s"], "clip_s": c["clip_len"]})
    rows.sort(key=lambda r: -r["diff_mm"])
    close = [r for r in rows if r["diff_mm"] <= DIVERGE_MM]
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/render_compare.json").write_text(json.dumps({"diverge_mm": DIVERGE_MM, "rows": rows}, indent=1) + "\n")
    syn = [r for r in rows if r["stem"].startswith("SYN_")]
    hum = [r for r in rows if not r["stem"].startswith("SYN_")]
    print(f"{len(close)} of {len(rows)} renders reproduce the evaluator's rollout within {DIVERGE_MM} mm of mean body "
          f"error (human {sum(r['diff_mm'] <= DIVERGE_MM for r in hum)}/{len(hum)}, synthetic "
          f"{sum(r['diff_mm'] <= DIVERGE_MM for r in syn)}/{len(syn)}); median "
          f"{sorted(r['diff_mm'] for r in rows)[len(rows) // 2]} mm")
    print(f"\n{'motion':60s} {'render':>7s} {'eval':>7s} {'diff mm':>8s} {'trk r/e':>11s} {'gate r / e (s)':>16s}")
    for r in rows:
        if r["diff_mm"] > DIVERGE_MM or (r["render_tracked"] < 0.99) or (r["eval_tracked"] < 0.99):
            print(f"{r['stem'][:60]:60s} {r['render_mean_err']:7.3f} {r['eval_mean_err']:7.3f} {r['diff_mm']:8.1f} "
                  f"{r['render_tracked']:5.2f}/{r['eval_tracked']:4.2f} "
                  f"{str(r['render_first_gate_s'])[:6]:>7s} / {str(r['eval_first_gate_s'])[:6]:6s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
