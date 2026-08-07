# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the ablation sensitivities produced by ablate_contact_obs.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "text.color": INK, "font.size": 8.5,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "legend.frameon": False,
    "figure.dpi": 130,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = np.load(a.npz)
    names = [str(s) for s in d["spec_names"]]
    da = d["delta_action"]
    ref = da[:, names.index("key:mimic_target_poses")].mean()
    pav = da[:, names.index("key:previous_actions")].mean() / ref * 100

    keys = [n for n in names if n.startswith("key:")]
    chans = [n for n in names if n.startswith("chan:")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6),
                             gridspec_kw={"width_ratios": [1, 1.12]})
    steps, _, envs = da.shape
    mass = float(d["total_mass"]) if "total_mass" in d.files else float("nan")
    fig.suptitle(
        "Does the contact-rich policy actually use its contact observations?\n"
        "Each block replaced by its running mean (= zero after the model's own "
        f"normalisation); action change measured on-policy, {steps} steps x {envs} "
        f"poses, {mass:.2f} kg plant\n{Path(a.npz).name}",
        fontsize=11, fontweight="semibold", color=INK, y=1.08,
    )

    ax = style(axes[0])
    vals = [da[:, names.index(n)].mean() / ref * 100 for n in keys]
    labels = [n.replace("key:", "") for n in keys]
    order = np.argsort(vals)
    cols = [ORANGE if "contact" in labels[i] else BLUE for i in order]
    y = np.arange(len(order))
    ax.barh(y, [vals[i] for i in order], color=cols, height=0.62)
    for yi, i in enumerate(order):
        ax.text(vals[i] + 1.5, yi, f"{vals[i]:.0f}%", va="center", fontsize=8, color=INK_2)
    ax.axvline(pav, color=INK_3, lw=1.1, ls="-")
    ax.text(pav, len(order) - 0.4, " previous_actions floor", fontsize=7.6, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_xlim(0, 118)
    ax.set_xlabel("action change, % of ablating mimic_target_poses")
    ax.set_title("whole observation blocks (orange = contact)", loc="left")

    ax = style(axes[1])
    cv = [da[:, names.index(n)].mean() / ref * 100 for n in chans]
    cl = [n.replace("chan:", "").replace("_proxy", "").replace("__", "") for n in chans]
    order = np.argsort(cv)
    y = np.arange(len(order))
    ax.barh(y, [cv[i] for i in order], color=ORANGE, height=0.62)
    for yi, i in enumerate(order):
        ax.text(cv[i] + 0.08, yi, f"{cv[i]:.1f}%", va="center", fontsize=7.6, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([cl[i] for i in order], fontsize=8)
    ax.tick_params(length=0)
    ax.set_xlim(0, max(cv) * 1.25)
    ax.set_xlabel("action change, % of ablating mimic_target_poses")
    ax.set_title("channels within contact_obs_v1, all 24 bodies at once", loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
