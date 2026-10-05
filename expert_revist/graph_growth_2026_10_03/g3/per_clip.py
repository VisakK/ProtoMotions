"""Per-clip comparison on the 56 human clips: G1's epoch 5,000 (the warm start) against every G3 evaluation.

Scores are the evaluator's per-motion ``score`` averaged over a clip's x0 / x3s / x7s variants (G1's
``per_clip_compare.py`` convention). Beside them, from the epoch-3,420 standalone library
(``analyze_rollouts_v2.py`` on release v3, x0 only): tracked share, mean body error, the first frame past the 0.5 m
gate, holds with their commanded supports down, and geometric substitutions. Writes ``data/per_clip.json``.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/per_clip.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = REPO / "results/smpl_yogi_v2_expert56_g3_2f132f4299"
STANDALONE = REPO / "output/renderings/expert56_v2_g3_e3420/eval_standalone_e3420/curriculum/eval_epoch_000000.csv"
G1_STANDALONE = REPO / "output/renderings/expert56_v2_amp_g1_e5000/eval_standalone_e5000/curriculum/eval_epoch_000000.csv"
G1_CSV_DIR = REPO / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/curriculum"
FLIP = 0.2


def base(stem: str) -> str:
    return re.sub(r"_x\d+s$", "", stem)


def short(stem: str) -> str:
    s = re.sub(r"^2209\d\d_", "", stem)
    name, _, var = s.rpartition("-")
    return f"{name.split('_or_')[0].replace('_', ' ')} -{var}"


def per_clip(path: Path) -> dict:
    by: dict[str, dict[str, list[float]]] = {}
    for r in csv.DictReader(open(path)):
        if r["motion"].startswith("SYN_"):
            continue
        d = by.setdefault(base(r["motion"]), {"group": r["group"], "score": [], "p_track": [], "drag_J": [],
                                               "p_family": []})
        for k in ("score", "p_track", "drag_J", "p_family"):
            if r[k] not in ("", "nan"):
                d[k].append(float(r[k]))
    return {k: {"group": v["group"], **{m: (float(np.mean(v[m])) if v[m] else None)
                                         for m in ("score", "p_track", "drag_J", "p_family")},
                "score_x0": None} for k, v in by.items()}


def main() -> int:
    evals = sorted(RUN.glob("curriculum/eval_epoch_*.csv"))
    epoch_of = lambda p: int(re.search(r"(\d+)", p.stem).group(1))  # noqa: E731
    cols = {"g3_" + str(epoch_of(p)): per_clip(p) for p in evals}
    cols["g3_3420s"] = per_clip(STANDALONE)
    g1 = per_clip(G1_STANDALONE)
    g1_evals = {epoch_of(p): per_clip(p) for p in sorted(G1_CSV_DIR.glob("eval_epoch_*.csv"))}
    lib = {c["stem"]: c for c in json.load(open(HERE / "data/library_g3_e3420.json"))["clips"]}
    lib_g1 = {c["stem"]: c for c in json.load(open(HERE.parent / "g1_amp_eval/data/library_g1_e5000.json"))["clips"]}

    order = list(cols)
    clips = []
    for stem in sorted(g1, key=lambda s: (g1[s]["group"], s)):
        series = [cols[c][stem]["score"] for c in order]
        g3_evals = series[1:]                                 # 500 ... 3000, 3420s (epoch 1 is G1 + one update)
        flips = int(sum(abs(a - b) > FLIP for a, b in zip(g3_evals, g3_evals[1:])))
        g1_band = [g1_evals[e][stem]["score"] for e in (4000, 4500, 5000) if e in g1_evals]
        row = {"stem": stem, "clip": short(stem), "group": g1[stem]["group"],
               "g1_5000s": round(g1[stem]["score"], 4), "g1_last3_mean": round(float(np.mean(g1_band)), 4),
               "g3": {c: round(cols[c][stem]["score"], 4) for c in order},
               "g3_last3_mean": round(float(np.mean([cols[f"g3_{e}"][stem]["score"] for e in (2000, 2500, 3000)])), 4),
               "g3_3420s": round(cols["g3_3420s"][stem]["score"], 4),
               "delta_3420s_vs_g1": round(cols["g3_3420s"][stem]["score"] - g1[stem]["score"], 4),
               "delta_last3": round(float(np.mean([cols[f"g3_{e}"][stem]["score"] for e in (2000, 2500, 3000)])
                                          - np.mean(g1_band)), 4),
               "flips": flips, "g3_min": round(min(g3_evals), 4),
               "drag_J_g1": round(g1[stem]["drag_J"] or 0.0, 1), "drag_J_3420s": round(cols["g3_3420s"][stem]["drag_J"] or 0.0, 1)}
        L, L1 = lib.get(stem), lib_g1.get(stem)
        if L:
            holds = [h for h in L["holds"] if h.get("recorded")]
            row.update(tracked=L["tracked_share"], mean_err=L["mean_err"], first_gate_s=L["first_termination_s"],
                       holds=len(holds), holds_supported=sum((h.get("support_min") is None) or h["support_min"] >= 0.9
                                                             for h in holds),
                       substitutions=[f"@{h['t_hold']:.1f} " + "+".join(h["substitutions"]) for h in holds
                                      if h.get("substitutions") and h["tracked_share"] >= 0.9])
        if L1:
            row.update(tracked_g1=L1["tracked_share"], mean_err_g1=L1["mean_err"])
        clips.append(row)

    out = {"columns": order, "flip_threshold": FLIP, "clips": clips}
    (HERE / "data/per_clip.json").write_text(json.dumps(out, indent=1) + "\n")

    below = lambda key: [c["clip"] for c in clips if c[key] < 0.95]  # noqa: E731
    print(f"clips below 0.95: G1 5000s {len(below('g1_5000s'))} {below('g1_5000s')}")
    print(f"                  G3 3420s {len(below('g3_3420s'))} {below('g3_3420s')}")
    print(f"flips (> {FLIP} between consecutive G3 evaluations 500..3000, 3420s): "
          f"{sum(c['flips'] > 0 for c in clips)} clips: {[c['clip'] for c in clips if c['flips']]}")
    print(f"\n{'clip':34s} {'group':11s} {'G1 5000':>7s} {'G3 3420':>7s} {'delta':>6s} {'G3 min':>6s} {'flips':>5s} "
          f"{'trk':>5s} {'err':>6s} {'gate s':>6s} {'drag':>6s} subst")
    for c in sorted(clips, key=lambda c: (c["delta_3420s_vs_g1"])):
        print(f"{c['clip'][:34]:34s} {c['group']:11s} {c['g1_5000s']:7.3f} {c['g3_3420s']:7.3f} "
              f"{c['delta_3420s_vs_g1']:+6.3f} {c['g3_min']:6.3f} {c['flips']:5d} {c.get('tracked', 0):5.2f} "
              f"{c.get('mean_err', 0):6.3f} {str(c.get('first_gate_s'))[:6]:>6s} {c['drag_J_3420s']:6.1f} "
              f"{'; '.join(c.get('substitutions', []))}")
    print(f"\n-> {HERE / 'data/per_clip.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
