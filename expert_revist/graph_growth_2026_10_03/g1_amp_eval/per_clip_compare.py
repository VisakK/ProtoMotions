"""Per-clip scores and drag: e15500 (and its 13,000-15,500 band) against every G1 evaluation.

A clip's value is the mean over its x0 / x3s / x7s variants, read from each run's ``curriculum/eval_epoch_*.csv``
(the in-training ``HoldCurriculumEvaluator``: deterministic actions, every motion from t = 0). Writes
``data/per_clip.json`` and prints README §3's tables: the largest changes, the flips between consecutive evaluations
and the drag movers.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g1_amp_eval/per_clip_compare.py
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E15 = ROOT / "results/smpl_yogi_v2_expert56_a2dda5d2ac/curriculum"
G1 = ROOT / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/curriculum"
BAND_FROM, LAST3, FLIP = 13000, (4000, 4500, 5000), 0.2
VARIANTS = ("", "_x3s", "_x7s")


def tables(folder: Path) -> dict[int, dict[str, dict]]:
    out = {}
    for f in glob.glob(str(folder / "eval_epoch_*.csv")):
        epoch = int(re.search(r"(\d+)\.csv", f).group(1))
        out[epoch] = {r["motion"]: r for r in csv.DictReader(open(f))}
    return dict(sorted(out.items()))


def clip_mean(table: dict, stem: str, col: str) -> float:
    return float(np.mean([float(table[stem + v][col]) for v in VARIANTS]))


def short(stem: str) -> str:
    name = stem[7:]
    name = re.sub(r"_pose_or_|_Pose_or_", " | ", name)
    return name[:60]


def main() -> None:
    e15, g1 = tables(E15), tables(G1)
    stems = sorted(m for m in e15[15500] if not re.search(r"_x\d+s$", m))
    band_epochs = [e for e in e15 if e >= BAND_FROM]
    g1_epochs = list(g1)
    rows = []
    for s in stems:
        group = e15[15500][s]["group"]
        e15_band = [clip_mean(e15[e], s, "score") for e in band_epochs]
        e15_flip_series = [clip_mean(e15[e], s, "score") for e in e15 if e >= 11000]
        g1_series = [clip_mean(g1[e], s, "score") for e in g1_epochs]
        drag_e15 = clip_mean(e15[15500], s, "drag_J")
        drag_g1 = [clip_mean(g1[e], s, "drag_J") for e in g1_epochs]
        subs = sorted({x for v in VARIANTS for x in g1[5000][s + v].get("substitutions_v2", "").split(";") if x})
        last3 = [g1_series[g1_epochs.index(e)] for e in LAST3]
        rows.append(dict(
            stem=s, group=group,
            e15500=clip_mean(e15[15500], s, "score"), e15_band_mean=float(np.mean(e15_band)),
            e15_band_min=float(np.min(e15_band)), e15_band_max=float(np.max(e15_band)),
            e15_flips=int((np.abs(np.diff(e15_flip_series)) > FLIP).sum()),
            g1=dict(zip(g1_epochs, g1_series)), g1_5000=g1_series[-1], g1_last3=float(np.mean(last3)),
            g1_min=float(np.min(g1_series[1:])), g1_max=float(np.max(g1_series[1:])),
            g1_flips=int((np.abs(np.diff(g1_series[1:])) > FLIP).sum()),
            drag_e15500=drag_e15, drag_g1=dict(zip(g1_epochs, drag_g1)), drag_g1_5000=drag_g1[-1],
            drag_g1_last3=float(np.mean([drag_g1[g1_epochs.index(e)] for e in LAST3])),
            p_track_g1_5000=clip_mean(g1[5000], s, "p_track"), p_family_g1_5000=clip_mean(g1[5000], s, "p_family"),
            substitutions_v2_5000=subs,
        ))
    (HERE / "data/per_clip.json").write_text(json.dumps(dict(epochs_g1=g1_epochs, band_epochs=band_epochs,
                                                             clips=rows), indent=1) + "\n")

    print(f"G1 evaluations {g1_epochs}; e15500 band {band_epochs}")
    print("\nclips by change, G1 last-3 mean minus e15500 band mean (x0/x3s/x7s mean score)")
    print(f"{'clip':60s} {'group':11s} {'e15500':>6s} {'band':>6s} | {'G1 5000':>7s} {'last3':>6s} {'min':>5s} {'max':>5s} "
          f"| {'delta':>6s} flips e15/G1")
    for r in sorted(rows, key=lambda r: r["g1_last3"] - r["e15_band_mean"]):
        d = r["g1_last3"] - r["e15_band_mean"]
        if abs(d) < 0.03 and r["g1_min"] > 0.9 and r["e15_band_min"] > 0.9:
            continue
        print(f"{short(r['stem']):60s} {r['group']:11s} {r['e15500']:6.3f} {r['e15_band_mean']:6.3f} | "
              f"{r['g1_5000']:7.3f} {r['g1_last3']:6.3f} {r['g1_min']:5.2f} {r['g1_max']:5.2f} | {d:+6.3f} "
              f"{r['e15_flips']:>3d}/{r['g1_flips']:<3d}")
    fails15 = [r for r in rows if r["e15500"] < 0.95]
    fails_g1 = [r for r in rows if r["g1_5000"] < 0.95]
    print(f"\nclips below 0.95: e15500 {len(fails15)}, G1@5000 {len(fails_g1)}")
    print("  e15500:", ", ".join(short(r["stem"])[:28] for r in fails15))
    print("  G1@5000:", ", ".join(short(r["stem"])[:28] for r in fails_g1))
    print(f"clips with a > {FLIP} jump between consecutive evaluations: e15500 (11,000-15,500, 10 evals) "
          f"{sum(r['e15_flips'] > 0 for r in rows)}, G1 (500-5,000, 10 evals) {sum(r['g1_flips'] > 0 for r in rows)}")

    print("\ndrag per clip (J per rollout, variant mean): largest movers")
    print(f"{'clip':60s} {'e15500':>7s} {'G1 5000':>8s} {'G1 last3':>8s}")
    for r in sorted(rows, key=lambda r: -(r["drag_g1_last3"] - r["drag_e15500"]))[:12]:
        print(f"{short(r['stem']):60s} {r['drag_e15500']:7.1f} {r['drag_g1_5000']:8.1f} {r['drag_g1_last3']:8.1f}")
    print("...")
    for r in sorted(rows, key=lambda r: (r["drag_g1_last3"] - r["drag_e15500"]))[:6]:
        print(f"{short(r['stem']):60s} {r['drag_e15500']:7.1f} {r['drag_g1_5000']:8.1f} {r['drag_g1_last3']:8.1f}")
    tot15 = np.mean([r["drag_e15500"] for r in rows])
    tot5 = np.mean([r["drag_g1_5000"] for r in rows])
    print(f"mean over 56 clips: e15500 {tot15:.1f} J, G1@5000 {tot5:.1f} J")
    print("\nG1@5000 substitutions (v2 force rule, any variant):")
    for r in rows:
        if r["substitutions_v2_5000"]:
            print(f"  {short(r['stem']):60s} {', '.join(r['substitutions_v2_5000'])}")


if __name__ == "__main__":
    main()
