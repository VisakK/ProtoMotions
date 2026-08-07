# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot how an arm balance is acquired over training.

Takes one rollout per checkpoint of the same run and shows the two quantities
that matter together: how far the centre of mass is from the polygon the *hands
alone* make, and how much of the clip is spent with no foot on the floor.  The
first is the mechanism (lean forward until the COM is over the hands), the
second is the result (the trailing foot becomes releasable).

Usage::

    python data/scripts/plot_liftoff_progress.py --out results/.../fig_09_liftoff.png \
      --baseline results/Contact_Physics_analysis/220923_Crane_Crow_Pose_or_Bakasana_-a \
      2000:sweep/epoch_2000/<clip> 4000:sweep/epoch_4000/<clip> ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_physics_support import contact_series, lean_series, load_rollout  # noqa: E402
from liftoff_report import FEET, HANDS, breakdown  # noqa: E402

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "text.color": INK, "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.titleweight": "semibold", "legend.frameon": False,
    "lines.linewidth": 1.6, "figure.dpi": 130,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def measure(path: str) -> dict:
    roll = load_rollout(Path(path))
    series = contact_series(roll)
    lean = lean_series(roll)
    down = lean.hands_down
    hd, ho, h1, h2, nh = breakdown(series.ground_contact, roll.body_names)
    return {
        "hands_only": 100 * ho,
        "one_foot": 100 * h1,
        "two_feet": 100 * h2,
        "hand_margin": float(np.nanmean(lean.hand_margin[down])),
        "inside": 100 * float(np.nanmean(lean.hand_margin[down] > 0)),
        "lean": float(np.nanmean(lean.lean[down])),
        "foot_load": 100 * float(np.nanmean(lean.foot_load[down])),
        "track": float(np.mean(roll.raw["ctrl_track_err"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", nargs="+", help="EPOCH:path/to/rollout_dir entries.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--baseline", type=str, default=None,
                        help="Rollout dir for a reference policy, drawn as a flat line.")
    parser.add_argument("--baseline-label", type=str, default="19-motion tracker")
    parser.add_argument("--target", type=float, default=None,
                        help="Reference clip's own hands-only %, drawn as the target.")
    parser.add_argument("--title", type=str, default="Acquiring the arm balance")
    args = parser.parse_args()

    epochs, rows = [], []
    for entry in args.points:
        epoch, _, path = entry.partition(":")
        epochs.append(float(epoch))
        rows.append(measure(path))
    order = np.argsort(epochs)
    epochs = np.array(epochs)[order]
    rows = [rows[i] for i in order]
    base = measure(args.baseline) if args.baseline else None

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    ax = style(axes[0])
    ax.plot(epochs, [r["hand_margin"] for r in rows], color=CAT[0], marker="o", ms=4,
            label="COM → hands-only polygon")
    ax.plot(epochs, [r["lean"] for r in rows], color=CAT[0], ls="--", lw=1.0, alpha=0.6,
            marker=".", ms=4, label="COM offset along feet→hands axis")
    ax.axhline(0, color=RED, lw=0.9, ls=":")
    ax.text(epochs[0], 0, " COM over the hands", fontsize=7, color=RED, va="bottom")
    if base:
        ax.axhline(base["hand_margin"], color=INK_3, lw=0.9, ls="-.")
        ax.text(epochs[-1], base["hand_margin"], f" {args.baseline_label}", fontsize=7,
                color=INK_3, ha="right", va="bottom")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("m")
    ax.set_title("1 · The lean", loc="left")
    ax.legend(loc="lower right", fontsize=7)

    ax = style(axes[1])
    ax.plot(epochs, [r["hands_only"] for r in rows], color=CAT[2], marker="o", ms=4,
            label="hands only (lift-off)")
    ax.plot(epochs, [r["two_feet"] for r in rows], color=CAT[1], marker=".", ms=4,
            lw=1.0, label="hands + 2 feet")
    ax.plot(epochs, [r["inside"] for r in rows], color=CAT[0], ls="--", lw=1.0,
            label="COM inside hand polygon")
    if args.target is not None:
        ax.axhline(args.target, color=RED, lw=0.9, ls=":")
        ax.text(epochs[0], args.target, " reference target", fontsize=7, color=RED,
                va="bottom")
    if base:
        ax.axhline(base["hands_only"], color=INK_3, lw=0.9, ls="-.")
        ax.text(epochs[-1], base["hands_only"], f" {args.baseline_label}", fontsize=7,
                color=INK_3, ha="right", va="bottom")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("% of clip")
    ax.set_title("2 · The trailing foot comes off", loc="left")
    ax.legend(loc="upper left", fontsize=7)

    ax = style(axes[2])
    ax.plot(epochs, [r["foot_load"] for r in rows], color=CAT[3], marker="o", ms=4,
            label="share of ground force on the feet")
    ax2 = ax.twinx()
    ax2.plot(epochs, [r["track"] for r in rows], color=INK_3, lw=1.0, ls="--",
             marker=".", ms=4, label="tracking error")
    ax2.set_ylabel("worst-body error (m)", color=INK_3)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("% of vertical GRF")
    ax.set_title("3 · Load leaves the feet", loc="left")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=7)

    fig.suptitle(args.title, fontsize=12, fontweight="semibold", color=INK, y=1.03)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
