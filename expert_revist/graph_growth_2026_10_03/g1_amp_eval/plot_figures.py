"""README figures: evaluation curves across the warm start, the style measurements, per joint and per clip.

Reads only this folder's ``data/`` (and e15500's ``tb_eval_scalars.json``); writes ``figures/*.png``.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g1_amp_eval/plot_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
E15_TB = HERE.parents[1] / "expert56_v2_e15500/data/tb_eval_scalars.json"
WARM = 15500                       # G1 epoch e is plotted at 15,500 + e
E15_FROM = 9000
# reference palette, categorical slots 1-2 (validated: CVD dE 24.7, normal 33.6) and the chrome inks
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold", "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0, "lines.markersize": 4.5,
})


def series(tb: dict, tag: str, offset: int = 0, min_epoch: int = 0):
    pts = [(int(a) + offset, b) for a, b in tb.get(tag, []) if int(a) >= min_epoch]
    return [p[0] for p in pts], [p[1] for p in pts]


def gate_line(ax, y, text):
    ax.axhline(y, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.99, y, text, transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=INK2, fontsize=7.5)


def warm_marker(ax):
    ax.axvline(WARM, color=AXIS, lw=1.0, zorder=1)


def eval_curves() -> None:
    e15, g1 = json.load(open(E15_TB)), json.load(open(DATA / "tb_eval_scalars.json"))
    panels = [("eval/perf_group/single_leg_score", "Single-leg score", 0.948, "gate 0.948"),
              ("eval/perf_group/inversion_score", "Inversion score", 0.843, "gate 0.843"),
              ("eval/perf_group/arm_balance_score", "Arm-balance score", 0.939, "gate 0.939"),
              ("eval/perf_group/connective_score", "Connective score", 0.920, "gate 0.920"),
              ("eval/success_rate", "Success rate", None, None),
              ("eval/perf/worst10_score", "Worst-10 % score", None, None),
              ("eval/normalized_jerk_mean", "Normalised jerk", 72.2, "gate < 72.2"),
              ("eval/drag/all_J", "Drag (J per rollout)", 39.6, "gate <= 39.6")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
    for ax, (tag, title, gate, gtext) in zip(axes.flat, panels):
        x, y = series(e15, tag, 0, E15_FROM)
        ax.plot(x, y, "-o", color=BLUE, label="e15500 run (no AMP)", zorder=3)
        x2, y2 = series(g1, tag, WARM)
        ax.plot(x2, y2, "-o", color=ORANGE, label="G1 (AMP fine-tune)", zorder=3)
        if gate is not None:
            gate_line(ax, gate, gtext)
        if tag == "eval/normalized_jerk_mean":
            gate_line(ax, 68.6, "improvement < 68.6")
        warm_marker(ax)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training epoch (G1 = 15,500 + its epoch)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("In-training evaluation across the warm start (every motion from t = 0, deterministic)",
                 x=0.01, ha="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(FIG / "eval_curves.png", dpi=130)
    plt.close(fig)


def lib_epoch(label: str) -> int:
    run, e = label.split(":")
    return int(e) + (WARM if run == "g1" else 0)


def style() -> None:
    st = json.load(open(DATA / "style_shift.json"))["libraries"]
    sep = {}
    for lb, f in (("e15500:15500", HERE.parent / "e2_amp/separability_e15500.json"),
                  *[(f"g1:{m.group(1)}", p) for p in sorted(DATA.glob("separability_g1_e*.json"))
                    if (m := re.search(r"_e(\d+)\.json", p.name))]):
        sep[lb] = json.load(open(f))["leave_clips_out"]["tracked"]
    tb = json.load(open(DATA / "tb_training_binned.json"))["tags"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    for key, ax, title, unit in (("median_joint_angle_deg", axes[0], "Joint rotation off the human's", "deg, median"),
                                 ("mean_bias_deg", axes[1], "Systematic (droop-like) bias", "deg, mean over joints")):
        for run, color, name in (("e15500", BLUE, "e15500 run"), ("g1", ORANGE, "G1")):
            pts = sorted((lib_epoch(lb), r["posture"][key]) for lb, r in st.items() if lb.startswith(run + ":"))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color, label=name, zorder=3)
        ax.set_ylim(bottom=0)
        warm_marker(ax)
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.set_xlabel("training epoch")
    axes[0].legend(loc="lower left")
    ax = axes[2]
    for run, color in (("e15500", BLUE), ("g1", ORANGE)):
        pts = sorted((lib_epoch(lb), v) for lb, v in sep.items() if lb.startswith(run + ":"))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color, zorder=3)
    gate_line(ax, 0.5, "chance 0.5")
    ax.set_ylim(0.45, 1.02)
    warm_marker(ax)
    ax.set_title("Policy vs human: classifier on unseen clips", loc="left")
    ax.set_ylabel("AUROC, leave-clips-out")
    ax.set_xlabel("training epoch")
    ax = axes[3]
    acc = tb["discriminator/agent_acc"]
    ax.plot([int(k) + WARM for k in acc], list(acc.values()), "-o", color=ORANGE, zorder=3)
    ax.axhspan(0.55, 0.85, color=GRID, alpha=0.6, zorder=0)
    ax.text(0.02, 0.86, "plan's health band 0.55-0.85", transform=ax.get_yaxis_transform(), color=INK2, fontsize=7.5,
            va="bottom")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("G1's discriminator: agent accuracy", loc="left")
    ax.set_ylabel("500-epoch mean")
    ax.set_xlabel("training epoch")
    fig.savefig(FIG / "style.png", dpi=130)
    plt.close(fig)


def joints() -> None:
    st = json.load(open(DATA / "style_shift.json"))["libraries"]
    g1_last = max((lb for lb in st if lb.startswith("g1:")), key=lambda lb: int(lb.split(":")[1]))
    a, b = st["e15500:15500"]["posture"]["joints"], st[g1_last]["posture"]["joints"]
    order = sorted(a, key=lambda j: a[j]["median_deg"])
    fig, ax = plt.subplots(figsize=(7.5, 6.6), constrained_layout=True)
    for i, j in enumerate(order):
        ax.plot([b[j]["median_deg"], a[j]["median_deg"]], [i, i], color=AXIS, lw=1.5, zorder=1)
    ax.scatter([a[j]["median_deg"] for j in order], range(len(order)), color=BLUE, s=40, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="e15500, epoch 15,500")
    ax.scatter([b[j]["median_deg"] for j in order], range(len(order)), color=ORANGE, s=40, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label=f"G1, epoch {g1_last.split(':')[1]}")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(left=0)
    ax.set_xlabel("median angle between the policy's and the human's parent-relative rotation (deg), tracked frames")
    ax.set_title("Every joint moved toward the human's posture", loc="left")
    ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "joints.png", dpi=130)
    plt.close(fig)


def clips() -> None:
    pc = json.load(open(DATA / "per_clip.json"))["clips"]
    keep = [r for r in pc if abs(r["g1_last3"] - r["e15_band_mean"]) >= 0.03 or min(r["g1_min"], r["e15_band_min"]) < 0.9]
    keep.sort(key=lambda r: r["g1_last3"] - r["e15_band_mean"])
    names = [re.sub(r"_(pose|Pose)_or_.*?(-[a-d])$", r" \2", r["stem"][7:]).replace("_", " ") for r in keep]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(keep) + 1.4), constrained_layout=True)
    for i, r in enumerate(keep):
        ax.plot([r["e15_band_mean"], r["g1_last3"]], [i, i], color=AXIS, lw=1.5, zorder=1)
    ax.scatter([r["e15_band_mean"] for r in keep], range(len(keep)), color=BLUE, s=40, zorder=3, edgecolor=SURFACE,
               linewidth=1.5, label="e15500: mean of evaluations 13,000-15,500")
    ax.scatter([r["g1_last3"] for r in keep], range(len(keep)), color=ORANGE, s=40, zorder=3, edgecolor=SURFACE,
               linewidth=1.5, label="G1: mean of evaluations 4,000-5,000")
    ax.set_yticks(range(len(keep)), names)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("clip score (mean over its x0 / x3s / x7s variants)")
    ax.set_title("Clips that moved, or that sit below 0.9 in either run", loc="left")
    ax.legend(loc="center left")
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "clips.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    eval_curves()
    style()
    joints()
    clips()
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
