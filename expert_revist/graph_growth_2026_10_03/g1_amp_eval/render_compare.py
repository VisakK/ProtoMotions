"""README §8's table: the 20 rendered clips, e15500 against G1 epoch 5,000, one row per clip.

Numbers come from each batch's ``render_summary.json`` (``analyze_rollouts_v2.py --render-dir``: the rendered rollout
itself) and each run's evaluation CSV; the notes from ``render_notes_g1.json`` (written after watching each
side-by-side video against both timelines). Also writes ``data/render_compare.json``.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g1_amp_eval/render_compare.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1] / "expert56_v2_e15500"))
from render_table import display_name  # noqa: E402

E15_SUM = ROOT / "output/renderings/expert56_v2_e15500/render_summary.json"
G1_SUM = ROOT / "output/renderings/expert56_v2_amp_g1_e5000/render_summary.json"
E15_CSV = ROOT / "results/smpl_yogi_v2_expert56_a2dda5d2ac/curriculum/eval_epoch_015500.csv"
G1_CSV = ROOT / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/curriculum/eval_epoch_005000.csv"


def fam_text(c: dict) -> str:
    fam = [h for h in c["holds"] if h.get("recorded") and h.get("pose_role") == "family" and h["err6_p50"] is not None]
    return "; ".join(f"@{h['t_hold']:.1f} {h['err6_p50']:.3f}/{h['support_min']:.2f}"
                     + (" " + ",".join(h["substitutions"]) if h["substitutions"] else "") for h in fam) or "—"


def main() -> None:
    e15 = {c["stem"]: c for c in json.load(open(E15_SUM))["clips"]}
    g1 = {c["stem"]: c for c in json.load(open(G1_SUM))["clips"]}
    s15 = {r["motion"]: float(r["score"]) for r in csv.DictReader(open(E15_CSV))}
    sg1 = {r["motion"]: float(r["score"]) for r in csv.DictReader(open(G1_CSV))}
    notes_path = HERE / "render_notes_g1.json"
    notes = json.load(open(notes_path)) if notes_path.exists() else {}
    out = []
    print("| # | clip | eval score x0: e15500 → G1 | rendered: mean err (m) e15500 → G1 | tracked | leaves 0.5 m gate | "
          "family-pose holds e15500 (@t err/supports) | family-pose holds G1 | what changed (video) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for stem, c in sorted(g1.items(), key=lambda kv: kv[1]["label"]):
        a = e15[stem]
        num = c["label"].split("_", 1)[0]
        gate = lambda r: f"{r['first_termination_s']:.1f} s" if r["first_termination_s"] is not None else "never"  # noqa: E731
        row = dict(num=num, stem=stem, score_e15500=s15[stem], score_g1=sg1[stem], mean_err=[a["mean_err"], c["mean_err"]],
                   tracked=[a["tracked_share"], c["tracked_share"]], gate=[gate(a), gate(c)],
                   family_e15500=fam_text(a), family_g1=fam_text(c), note=notes.get(num, ""))
        out.append(row)
        print(f"| {num} | {display_name(stem)} | {s15[stem]:.3f} → {sg1[stem]:.3f} | {a['mean_err']:.3f} → {c['mean_err']:.3f} | "
              f"{a['tracked_share']:.2f} → {c['tracked_share']:.2f} | {gate(a)} → {gate(c)} | {row['family_e15500']} | "
              f"{row['family_g1']} | {row['note']} |")
    (HERE / "data/render_compare.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
