"""The README's appendices: one row per human clip (A) and one per synthetic variant (B), from ``data/``.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/appendix_tables.py > /tmp/appendix.md
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUP = {"single_leg": "single-leg", "inversion": "inversion", "arm_balance": "arm balance", "connective": "connective"}
ORDER = ("single_leg", "inversion", "arm_balance", "connective")
SHORT = {"Pose Dedicated to the Sage Koundinya": "Koundinya", "Feathered Peacock Pose": "Pincha (Feathered Peacock)",
         "Shoulder-Pressing Pose": "Shoulder-Pressing", "viparita virabhadrasana": "Reverse Warrior",
         "Extended Revolved Side Angle Pose": "Revolved Side Angle", "Extended Revolved Triangle Pose": "Revolved Triangle"}


def nice(clip: str) -> str:
    for long, short in SHORT.items():
        if clip.startswith(long):
            return short + clip[len(long):]
    return clip


def main() -> int:
    clips = json.load(open(HERE / "data/per_clip.json"))["clips"]
    print("### Appendix A: every human clip\n")
    print("Scores are the evaluator's, mean over x0/x3s/x7s (G1: its standalone epoch 5,000; G3: the standalone "
          "3,420). Min is G3's lowest evaluation from 500 on. The rollout columns are the x0 rollout of the 3,420 "
          "library: tracked share, mean body error, holds with their commanded supports down (≥ 90 % of the window, "
          "of tracked holds), and substitutions by geometry.\n")
    print("| # | Clip | Group | G1 | G3 3,420 | G3 min | Flips | Tracked | Mean err (m) | Holds supported | Drag (J) | Substitutions |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    n = 0
    for g in ORDER:
        for c in sorted([c for c in clips if c["group"] == g], key=lambda c: c["clip"]):
            n += 1
            subs = "; ".join(c.get("substitutions", [])) or "—"
            print(f"| {n} | {nice(c['clip'])} | {GROUP[g]} | {c['g1_5000s']:.2f} | {c['g3_3420s']:.2f} | "
                  f"{c['g3_min']:.2f} | {c['flips']} | {c.get('tracked', float('nan')):.2f} | "
                  f"{c.get('mean_err', float('nan')):.3f} | {c.get('holds_supported', 0)}/{c.get('holds', 0)} | "
                  f"{c['drag_J_3420s']:.0f} | {subs} |")
    e = json.load(open(HERE / "data/edge_tracking.json"))["libraries"]
    before = {r["stem"]: r for r in e["e1"]["rows"]}
    print("\n### Appendix B: every synthetic variant (x0)\n")
    print("Epoch 1 (before G3) → epoch 3,420. Transition tracked = share of `[departure, arrival)` frames with every "
          "body within 0.5 m; D error = 6-body best-yaw error over D's hold (p50); D supports = worst commanded zone's "
          "share of D's window down (geometry); speed = peak body speed over the transition (+0.5 s) ÷ the human "
          "references' per-body p99 (policy, reference).\n")
    print("| Variant | T (s) | Transition tracked | Mean / peak err (m) at 3,420 | D error (m) | D supports | Speed ÷ p99 | Reached D |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted([r for r in e["e3420"]["rows"] if r["x"] == 0], key=lambda r: ("E1 E3 B1 E2 E5".split().index(r["edge"]), r["stem"])):
        b = before[r["stem"]]
        print(f"| `{r['stem'][4:]}` | {r['T']:.2f} | {b['trans_tracked']:.2f} → **{r['trans_tracked']:.2f}** | "
              f"{r['trans_mean_err_p50']:.3f} / {r['trans_max_err_peak']:.2f} | {b['d_err6_p50']:.3f} → {r['d_err6_p50']:.3f} | "
              f"{b['d_support_geom']:.2f} → {r['d_support_geom']:.2f} | {r['nat_policy']['speed_over_ref_p99_max']:.2f} "
              f"({r['nat_ref']['speed_over_ref_p99_max']:.2f}) | {'no' if not b['reached_D'] else 'yes'} → "
              f"**{'yes' if r['reached_D'] else 'no'}** |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
