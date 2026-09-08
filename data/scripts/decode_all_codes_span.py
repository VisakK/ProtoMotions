# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join several ``decode_all_codes.py`` runs into one cross-state comparison.

Z1 at a single pair of states answered "is the decoder or the prior at fault
here".  Run over a *span* of commanded states it answers a different question:
**is the standing failure a property of that pose, or of a family?** -- and it
lets the corpus predictors be regressed on two separable responses that a
survival measurement conflates:

* ``held`` -- how many of the 625 codes hold the pose. A property of the
  **decoder** and of the pose's own stability.
* ``alignment`` = (prior probability mass on the holding codes) / (fraction of
  codes that hold). A property of the **prior**, normalised so that 1.0 is
  "picks as well as chance at this state" and it cannot be inflated by an easy
  pose. This is the number a survival metric cannot see.

Both are reported at two arrival thresholds because 0.15 m is a cliff
(``Student_v9_tier0_report.MD`` §13) and it changes conclusions: Eagle scores
0/625 at 0.15 m and 92/625 at 0.20 m purely because the policy settles 0.170 m
from the commanded frame.

Usage::

    PYTHONPATH=. python data/scripts/decode_all_codes_span.py \\
      --runs output/decode_all_codes/last output/decode_all_codes/pairA ... \\
      --predictors output/frozen_share.json \\
      --out-dir output/decode_all_codes/span_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(directory: Path, predictors: dict):
    summary = json.loads((directory / "summary.json").read_text())
    data = np.load(directory / "rollouts.npz")
    state_of_env = data["state_of_env"]
    slot_of_env = data["slot_of_env"]
    num_codes = int(summary["num_codes"])
    controls = int(summary["controls"])
    token = data["code_token"]
    logits = data["prior_logits_step0"]
    rows = []
    for index, state in enumerate(summary["states"]):
        sel = np.nonzero(state_of_env == index)[0]
        slots = slot_of_env[sel]
        code_envs = sel[slots < num_codes]
        code_envs = code_envs[np.argsort(slots[slots < num_codes])]
        sampled = sel[(slots >= num_codes) & (slots < num_codes + controls)]
        greedy = sel[slots >= num_codes + controls]
        worst = data["pose_err"][:, code_envs].max(axis=0)

        row = logits[code_envs[0]]
        probs = np.exp(row - row.max())
        probs /= probs.sum()
        code_probs = probs[token]
        order = np.argsort(-code_probs)
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(len(order))

        pred = predictors.get(state["plan"], {})
        entry = dict(
            plan=state["plan"], run=directory.name, node=state["node"],
            node_key=state["node_key"], pose_clip=state["pose_clip"],
            pose_time=state["pose_time"],
            frozen_share=pred.get("frozen_share"),
            frozen_gap_m=pred.get("frozen_gap_m"),
            prior_entropy_bits=float(-(probs * np.log2(np.clip(probs, 1e-12, None))).sum()),
            best_pose_err_m=float(worst.min()),
            code_probs=code_probs, rank_of=rank_of, worst=worst,
        )
        for threshold in (0.15, 0.20):
            held = worst <= threshold
            key = f"{threshold:.2f}"
            entry[f"held_{key}"] = int(held.sum())
            entry[f"mass_{key}"] = float(code_probs[held].sum())
            entry[f"rank_{key}"] = int(rank_of[held].min()) + 1 if held.any() else -1
            entry[f"align_{key}"] = (
                float(code_probs[held].sum() / (held.sum() / len(held)))
                if held.any() else float("nan")
            )
            entry[f"sampled_{key}"] = float(
                (data["pose_err"][:, sampled].max(axis=0) <= threshold).mean()
            )
            entry[f"greedy_{key}"] = float(
                (data["pose_err"][:, greedy].max(axis=0) <= threshold).mean()
            )
        rows.append(entry)
    return rows


def figure(path: Path, rows, threshold: str):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=125)

    ax = axes[0]
    for r in rows:
        align = r[f"align_{threshold}"]
        if not np.isfinite(align):
            ax.scatter(r["frozen_share"], 1e-3, marker="x", s=70, color="#777777")
            ax.annotate(f"{short(r)}\n(no code holds)", (r["frozen_share"], 1e-3),
                        fontsize=7, xytext=(4, -14), textcoords="offset points",
                        color="#777777")
            continue
        held = r[f"held_{threshold}"] / 625
        colour = "#d62728" if align < 0.3 else "#2ca02c"
        ax.scatter(r["frozen_share"], align, s=40 + 300 * held, alpha=0.75, color=colour)
        ax.annotate(short(r), (r["frozen_share"], align), fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(1.0, color="#333333", ls="--", lw=1.0)
    ax.text(0.02, 1.06, "prior picks as well as chance", fontsize=7.5, color="#333333")
    ax.set_yscale("log")
    ax.set_xlabel("frozen share of the node's at-pose training mass")
    ax.set_ylabel("alignment = prior mass on holding codes / fraction that hold")
    ax.set_title(f"Prior alignment vs frozen coverage (threshold {threshold} m)\n"
                 "marker area = fraction of the 625 codes that hold", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for r in rows:
        cum = np.cumsum(np.sort(r["code_probs"])[::-1])
        ax.plot(np.arange(1, len(cum) + 1), cum, lw=1.3,
                label=f"{short(r)}  (hold @ rank {r[f'rank_{threshold}']})")
        rank = r[f"rank_{threshold}"]
        if rank > 0:
            ax.scatter([rank], [cum[rank - 1]], s=28, zorder=5,
                       color=ax.lines[-1].get_color())
    ax.axhline(0.9, color="#333333", ls="--", lw=1.0)
    ax.text(1.1, 0.915, "deployed top-p = 0.9", fontsize=7.5, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("code rank under the prior at the commanded state")
    ax.set_ylabel("cumulative probability mass")
    ax.set_title("Where the first holding code sits in the prior's ranking\n"
                 "(dot = best holding code)", fontsize=10)
    ax.legend(fontsize=6.4, loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def short(row) -> str:
    name = row["plan"].replace("hold_probe_", "").replace("_L_FOOT", "").split("_")[0]
    return {"hp00l": "single-leg", "hp01l": "upward plank", "hp03l": "Eagle",
            "hp07l": "handstand", "hp09l": "Chaturanga", "hp09s": "Cobra -b",
            "hp11l": "firefly", "standing": "STANDING",
            "sideplank": "side plank"}.get(name, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--predictors", required=True,
                        help="frozen_share_table.py --json-out")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    predictors = {
        r["plan"]: r for r in json.loads(Path(args.predictors).read_text())["rows"]
    }
    rows = []
    for run in args.runs:
        rows.extend(load_run(Path(run), predictors))
    # One entry per state; a plan measured in two batches keeps both for the
    # reproducibility check but only the first enters the correlations.
    seen, unique = set(), []
    for r in rows:
        if r["plan"] in seen:
            continue
        seen.add(r["plan"])
        unique.append(r)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for threshold in ("0.15", "0.20"):
        figure(out / f"span_alignment_{threshold}.png", unique, threshold)

    from scipy.stats import spearmanr

    report = {"states": [], "correlations": {}}
    print(f"{'state':<15}{'frzShr':>7}{'frzGap':>7}{'held@.20':>9}{'mass':>8}"
          f"{'rank':>6}{'align':>8}{'samp':>6}{'H(bits)':>8}{'best':>7}")
    for r in rows:
        report["states"].append({
            k: v for k, v in r.items()
            if k not in ("code_probs", "rank_of", "worst")
        })
        print(f"{short(r):<15}{nan(r['frozen_share']):>7.3f}{nan(r['frozen_gap_m']):>7.3f}"
              f"{r['held_0.20']:>9}{r['mass_0.20']:>8.4f}{r['rank_0.20']:>6}"
              f"{r['align_0.20']:>8.3f}{r['sampled_0.20']:>6.2f}"
              f"{r['prior_entropy_bits']:>8.2f}{r['best_pose_err_m']:>7.3f}")

    for threshold in ("0.15", "0.20"):
        block = {}
        for label, xk, yk, log in (
            ("frozen_share_vs_held", "frozen_share", f"held_{threshold}", False),
            ("frozen_share_vs_alignment", "frozen_share", f"align_{threshold}", True),
            ("frozen_gap_vs_alignment", "frozen_gap_m", f"align_{threshold}", True),
            ("held_vs_alignment", f"held_{threshold}", f"align_{threshold}", True),
        ):
            x = np.array([nan(r[xk]) for r in unique], float)
            y = np.array([nan(r[yk]) for r in unique], float)
            if log:
                y = np.log10(y)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                rho, p = spearmanr(x[mask], y[mask])
                block[label] = dict(rho=float(rho), p=float(p), n=int(mask.sum()))
        report["correlations"][threshold] = block
        print(f"\nthreshold {threshold} m")
        for label, v in block.items():
            print(f"  {label:<32} rho={v['rho']:+.3f}  p={v['p']:.3f}  n={v['n']}")

    (out / "span_summary.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


def nan(value):
    return float("nan") if value is None else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
