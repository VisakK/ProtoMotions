"""The G3 evaluation's figures (README), from the JSON the analysis scripts write into ``data/``.

* ``figures/gates.png``: the pre-registered gates on human clips across G3's evaluations, small multiples.
* ``figures/edge_traces.png``: each edge's worst-body tracking error around its transition, epoch 1 (G1's policy plus
  one update) against epoch 3,420, every x0 variant.
* ``figures/edge_learning.png``: per edge, the transition's tracked share and the arrival error at D, by evaluation.
* ``figures/clips.png``: every human clip's score, G1 epoch 5,000 against G3 epoch 3,420.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/plot_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
# reference palette (dataviz skill, light mode): categorical slots in fixed order, ink and chrome
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
EDGES = ("E1", "E3", "B1", "E2", "E5")
EDGE_NAME = {"E1": "E1 crow → handstand (press)", "E3": "E3 handstand → crow (lower)",
             "B1": "B1 crow → plank (jump)", "E2": "E2 crow → chaturanga (jump-back)",
             "E5": "E5 handstand → chaturanga (float-down)"}
EDGE_SHORT = {"E1": "E1 press: crow → handstand", "E3": "E3 lower: handstand → crow",
              "B1": "B1 jump: crow → plank", "E2": "E2 jump-back: crow → chaturanga",
              "E5": "E5 float-down: handstand → chaturanga"}
SHORT_NAMES = {"Pose Dedicated to the Sage Koundinya": "Koundinya", "Extended Revolved Side Angle Pose":
               "Revolved Side Angle", "Extended Revolved Triangle Pose": "Revolved Triangle",
               "Supported Shoulderstand pose": "Supported Shoulderstand", "Feathered Peacock Pose": "Pincha (Feathered Peacock)",
               "Shoulder-Pressing Pose": "Shoulder-Pressing (Bhujapidasana)", "Standing big toe hold pose": "Standing Big Toe Hold",
               "Downward-Facing Dog pose": "Downward-Facing Dog", "Standing Forward Bend pose": "Standing Forward Bend",
               "viparita virabhadrasana": "Reverse Warrior"}
GROUP_NAME = {"single_leg": "Single-leg", "inversion": "Inversion", "arm_balance": "Arm balance",
              "connective": "Connective"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False,
})


def dot(ax, x, y, color, **kw):
    ax.plot(x, y, "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=4, **kw)


def gates() -> None:
    g = json.load(open(DATA / "gates.json"))
    cols = g["epochs"]                                       # "1", "500", ..., "3000", "3420s"
    x = [3420 if c == "3420s" else int(c) for c in cols]
    panels = [("single_leg_score", "Single-leg score", g["floors"]["single_leg"], "≥"),
              ("inversion_score", "Inversion score", g["floors"]["inversion"], "≥"),
              ("arm_balance_score", "Arm-balance score", g["floors"]["arm_balance"], "≥"),
              ("connective_score", "Connective score", g["floors"]["connective"], "≥"),
              ("edge_score", "Edge score (28 synthetic clips)", None, None),
              ("human_drag_J", "Drag, human clips (J per rollout)", g["gates"]["drag_max"], "≤"),
              ("human_subst_v2_x0", "Substitutions, human x0 holds", g["gates"]["subst_max"], "≤"),
              ("human_success", "Success, human motions", None, None)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    g1 = g["g1_5000_standalone"]
    for ax, (key, title, thr, op) in zip(axes.flat, panels):
        y = [g["series"][c].get(key) for c in cols]
        xs = [a for a, b in zip(x, y) if b is not None]
        ys = [b for b in y if b is not None]
        ax.plot(xs[:-1], ys[:-1], color=SLOTS[0], zorder=3)
        for a, b in zip(xs[:-1], ys[:-1]):
            dot(ax, a, b, SLOTS[0])
        # the standalone evaluation of the analysed checkpoint, set apart
        ax.plot(xs[-1], ys[-1], "D", ms=7, color=SLOTS[0], mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{ys[-1]:.3g}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=8)
        if thr is not None:
            ax.axhline(thr, color=MUTED, lw=1, zorder=2)
            ax.annotate(f"gate {op} {thr:g}", (0, thr), xytext=(4, 3), textcoords="offset points", color=INK2,
                        fontsize=7.5)
        if g1.get(key) is not None:
            ax.plot(-250, g1[key], "s", ms=6, color=MUTED, mec=SURFACE, mew=2, zorder=4)
            ax.annotate("G1", (-250, g1[key]), xytext=(0, 7), textcoords="offset points", ha="center",
                        color=INK2, fontsize=7.5)
        ax.set_title(title, loc="left")
        ax.set_xlim(-600, 3900)
        ax.set_xticks([0, 1000, 2000, 3000])
        ax.set_xlabel("G3 epoch")
    fig.suptitle("G3 against its pre-registered gates, human clips only (◆ = standalone evaluation of epoch 3,420; "
                 "■ = G1 epoch 5,000)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "gates.png", dpi=140)
    plt.close(fig)


def edge_traces() -> None:
    e = json.load(open(DATA / "edge_tracking.json"))
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=True)
    for ax, edge in zip(axes, EDGES):
        for lab, color, name in (("e1", MUTED, "epoch 1 (G1)"), ("e3420", SLOTS[0], "epoch 3,420")):
            rows = [r for r in e["libraries"][lab]["rows"] if r["edge"] == edge and r["x"] == 0]
            for i, r in enumerate(rows):
                tr = r["_trace"]
                ax.plot(tr["t_rel"], tr["max_err"], color=color, lw=1.2 if lab == "e1" else 1.6, alpha=0.9,
                        label=name if i == 0 else None, zorder=3 if lab == "e3420" else 2)
        T = rows[0]["t_arrival"] - rows[0]["t_departure"]
        ax.axvspan(0, T, color=SLOTS[0], alpha=0.07, lw=0, zorder=1)
        ax.axhline(0.5, color=INK2, lw=1, zorder=2)
        ax.annotate("0.5 m gate", (-1.4, 0.5), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=7.5)
        ax.set_title(EDGE_SHORT[edge], loc="left", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(0.03, 3.0)
        ax.set_xlim(-1.5, T + 3.0)
    axes[0].set_ylabel("worst-body error to the reference (m)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.supxlabel("time from the departure from S (s); shaded = the transition of the slowest variant", fontsize=9,
                  color=INK2)
    fig.suptitle("Each synthesised edge, tracked from the policy's own arrival at S: every x0 variant, before "
                 "(G3 epoch 1 ≈ G1 epoch 5,000) and after (epoch 3,420)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(FIG / "edge_traces.png", dpi=140)
    plt.close(fig)


def edge_learning() -> None:
    e = json.load(open(DATA / "edge_tracking.json"))
    labs = [("e1", 1), ("e1500", 1500), ("e3000", 3000), ("e3420", 3420)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9))
    for k, edge in enumerate(EDGES):
        tt, de = [], []
        for lab, _ in labs:
            rows = [r for r in e["libraries"][lab]["rows"] if r["edge"] == edge]
            tt.append(float(np.mean([r["trans_tracked"] for r in rows])))
            de.append(float(np.mean([r["reached_D"] for r in rows])))
        xs = [x for _, x in labs]
        off = (k - 2) * 25                                 # small horizontal offset so coincident points stay visible
        for ax, ys in ((axes[0], tt), (axes[1], de)):
            ax.plot([x + off for x in xs], ys, color=SLOTS[k], zorder=3)
            for a, b in zip(xs, ys):
                dot(ax, a + off, b, SLOTS[k])
        if edge == "E1":                                   # the late learner; the rest overlap at 1.0 (legend below)
            axes[0].annotate("E1 press", (xs[1] + off, tt[1]), xytext=(8, -3), textcoords="offset points",
                             color=INK, fontsize=8)
            axes[1].annotate("E1 press", (xs[1] + off, de[1]), xytext=(8, 4), textcoords="offset points",
                             color=INK, fontsize=8)
    axes[0].axhline(0.6, color=INK2, lw=1)
    axes[0].annotate("G2 gate 0.6", (0, 0.6), xytext=(4, 3), textcoords="offset points", color=INK2, fontsize=7.5)
    axes[0].set_title("Transition frames tracked (every body within 0.5 m)", loc="left")
    axes[0].set_ylim(-0.03, 1.05)
    axes[1].set_title("Share of motions that execute the edge", loc="left")
    axes[1].set_ylim(-0.03, 1.05)
    for ax in axes:
        ax.set_xlabel("G3 epoch (evaluations with a saved rollout library; 3,420 standalone)")
        ax.set_xticks([1, 1500, 3000, 3420])
        ax.set_xticklabels(["1", "1,500", "3,000", "3,420"])
    handles = [plt.Line2D([], [], color=SLOTS[k], marker="o", mec=SURFACE, mew=2, label=EDGE_NAME[ed])
               for k, ed in enumerate(EDGES)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Edge learning: all 84 synthetic motions (28 variants × x0 / x3s / x7s), from the evaluator's own "
                 "rollouts", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(FIG / "edge_learning.png", dpi=140)
    plt.close(fig)


def clips() -> None:
    d = json.load(open(DATA / "per_clip.json"))["clips"]
    order = ["single_leg", "inversion", "arm_balance", "connective"]
    rows = sorted(d, key=lambda c: (order.index(c["group"]), -c["delta_3420s_vs_g1"], c["clip"]))
    fig, ax = plt.subplots(figsize=(8.5, 13.5))
    ypos, cur, prev_g = [], 0.0, None
    for c in rows:                                         # one blank row above each group for its header
        if c["group"] != prev_g:
            cur += 1.0
            prev_g = c["group"]
        ypos.append(cur)
        cur += 1.0
    y = np.array([cur - v for v in ypos])
    for yi, c in zip(y, rows):
        a, b = c["g1_5000s"], c["g3_3420s"]
        ax.plot([a, b], [yi, yi], color=GRID if abs(b - a) < 0.01 else BASE, lw=2, zorder=2)
        ax.plot(a, yi, "o", ms=6, color=MUTED, mec=SURFACE, mew=1.5, zorder=3)
        ax.plot(b, yi, "o", ms=7, color=SLOTS[0], mec=SURFACE, mew=1.5, zorder=4)
        if abs(b - a) >= 0.05:
            ax.annotate(f"{b - a:+.2f}", (max(a, b), yi), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK)
    ax.set_yticks(y)
    def nice(name: str) -> str:
        for long, short in SHORT_NAMES.items():
            if name.startswith(long):
                return short + name[len(long):]
        return name
    ax.set_yticklabels([nice(c["clip"]) for c in rows], fontsize=7.5, color=INK2)
    ax.set_xlim(0.3, 1.06)
    ax.set_xlabel("evaluator score, mean of the x0 / x3s / x7s variants")
    prev = None
    for yi, c in zip(y, rows):
        if c["group"] != prev:
            ax.annotate(GROUP_NAME[c["group"]], (0.305, yi + 1.0), va="center", fontsize=8.5, color=INK,
                        fontweight="semibold")
            prev = c["group"]
    ax.plot([], [], "o", color=MUTED, label="G1 epoch 5,000 (warm start)")
    ax.plot([], [], "o", color=SLOTS[0], label="G3 epoch 3,420")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Every human clip, before and after G3 (standalone evaluations)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "clips.png", dpi=140)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    gates()
    edge_traces()
    edge_learning()
    clips()
    print(f"-> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
