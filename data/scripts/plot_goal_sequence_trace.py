# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot what a student actually touched during a scripted goal sequence.

``render_contact_goal_sequence.py`` writes a per-step trace and a video. The
video shows *what happened*; this shows **why the numbers say what they say**,
which the aggregate per-goal IoU cannot:

* a **contact raster**, one row per ground zone, coloured by whether the goal
  asked for that contact (blue) or not (orange) -- so "stuck on all fours" reads
  as a solid orange band across the foot rows rather than as a mid-range IoU;
* the **ground IoU** against the nearest goal;
* the **pelvis height**, which is what separates a downdog from a plank and a
  handstand from a forward fold -- none of which the contact set can express.

Two traces can be overlaid as columns to compare checkpoints.

Usage::

    PYTHONPATH=. python data/scripts/plot_goal_sequence_trace.py \\
      --trace output/renderings/v2_probe/peak_flow.json \\
      --trace output/renderings/v2_probe/collapsed_flow.json \\
      --title "standing -> downdog -> 3-leg dog -> handstand -> ... -> warrior II" \\
      --out output/renderings/v2_probe/flow_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots 1 and 2 (see the dataviz reference palette):
# adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the light surface.
WANTED = "#2a78d6"       # in contact, and the goal asked for it
UNWANTED = "#eb6834"     # in contact, and the goal did not
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedddb"

# Bottom-up, so the raster reads like a body: feet at the bottom, hands above.
ZONE_ORDER = [
    "HEAD", "TRUNK", "PELVIS",
    "L_FOREARM", "R_FOREARM", "L_UPPER_ARM", "R_UPPER_ARM",
    "L_HAND", "R_HAND",
    "L_THIGH", "R_THIGH", "L_SHANK", "R_SHANK",
    "L_FOOT", "R_FOOT",
]


def goal_spans(goals: list) -> list:
    """``[(name, i_start, i_end), ...]`` for consecutive runs of the same goal."""
    spans, start = [], 0
    for i in range(1, len(goals) + 1):
        if i == len(goals) or goals[i] != goals[start]:
            spans.append((goals[start], start, i))
            start = i
    return spans


def goal_zone_sets(payload: dict) -> dict:
    """``goal name -> set of ground zones its configuration asks for``.

    Read from the trace's own ``summary``, which the renderer resolved against
    the graph the policy was actually queried with. Re-parsing the plan file's
    configuration strings here would silently disagree with it if a plan named a
    node by id, or was written against a different graph build.
    """
    out = {}
    for entry in payload.get("summary", []):
        out[entry.get("goal", "?")] = set(entry.get("wanted_ground_zones", []))
    return out


def draw(ax_raster, ax_iou, ax_z, payload: dict, title: str, show_ylabels: bool,
         rows: list, z_top: float):
    trace = payload.get("trace", payload)
    t = np.asarray(trace["t"], dtype=float)
    goals = list(trace["goal"])
    iou = np.asarray(trace["iou"], dtype=float)
    root_z = np.asarray(trace["root_z"], dtype=float)
    zones_per_step = [set(z) for z in trace["zones"]]
    wanted_by_goal = goal_zone_sets(payload)

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 30.0
    for row, zone in enumerate(rows):
        active = np.array([zone in s for s in zones_per_step])
        want = np.array([zone in wanted_by_goal.get(g, set()) for g in goals])
        # One bar per contiguous run keeps the PNG small and the edges crisp.
        for value, colour in ((True, WANTED), (False, UNWANTED)):
            mask = active & (want == value)
            if not mask.any():
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax_raster.add_patch(
                    mpatches.Rectangle(
                        (t[a], row + 0.12), max((b - a) * dt, dt), 0.76,
                        facecolor=colour, edgecolor="none",
                    )
                )

    ax_raster.set_ylim(0, len(rows))
    ax_raster.set_yticks(np.arange(len(rows)) + 0.5)
    ax_raster.set_yticklabels(rows if show_ylabels else [""] * len(rows),
                              fontsize=7, color=INK_2)
    # Headroom for the goal-name band, which is placed in axes coordinates so it
    # lands identically in every column regardless of row count.
    ax_raster.set_title(title, fontsize=10, color=INK, pad=42, loc="left")

    ax_iou.plot(t, iou, lw=2, color=WANTED, solid_capstyle="round")
    ax_iou.set_ylim(-0.05, 1.05)
    ax_iou.set_yticks([0, 0.5, 1.0])

    ax_z.plot(t, root_z, lw=2, color=WANTED, solid_capstyle="round")
    ax_z.set_ylim(0, z_top)
    ax_z.set_xlabel("seconds", fontsize=8, color=INK_2)

    spans = goal_spans(goals)
    for axis in (ax_raster, ax_iou, ax_z):
        axis.set_xlim(t[0], t[-1])
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_2, labelsize=7, length=3)
        for _name, a, _b in spans[1:]:
            axis.axvline(t[a], color=GRID, lw=1, zorder=0)
    ax_raster.grid(False)
    for axis in (ax_iou, ax_z):
        axis.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        axis.set_axisbelow(True)

    # Goal names above the raster: direct labels, so the reader never has to map
    # a colour back to a phase through a legend. Rotated, because seven goal
    # names across a 33 s span overlap when set horizontally, and placed in axes
    # coordinates so every column's band sits at the same height.
    for name, a, b in spans:
        mid = (t[a] + t[min(b, len(t) - 1)]) / 2
        ax_raster.annotate(
            name.replace("_", " "),
            xy=(mid, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 4), textcoords="offset points",
            ha="left", va="bottom", rotation=30, rotation_mode="anchor",
            fontsize=7, color=INK_2, annotation_clip=False,
        )

    if show_ylabels:
        ax_iou.set_ylabel("ground IoU", fontsize=8, color=INK_2)
        ax_z.set_ylabel("pelvis z (m)", fontsize=8, color=INK_2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trace", action="append", required=True,
                        help="a .json written by render_contact_goal_sequence (repeatable)")
    parser.add_argument("--label", action="append", default=None,
                        help="column title per trace (default: the trace's label)")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    payloads = [json.loads(Path(p).read_text()) for p in args.trace]
    labels = args.label or [p.get("label", Path(t).stem)
                            for p, t in zip(payloads, args.trace)]
    if len(labels) != len(payloads):
        raise SystemExit(f"{len(labels)} labels for {len(payloads)} traces")

    # One row set and one pelvis scale across every column: a comparison figure
    # whose rows or axes differ per column is not a comparison. The union is
    # taken over all traces so a zone only one checkpoint ever touches still has
    # a row -- its absence in the other column is then a readable fact rather
    # than a missing row.
    touched = set()
    z_top = 1.25
    for payload in payloads:
        trace = payload.get("trace", payload)
        touched |= {z for step in trace["zones"] for z in step}
        z_top = max(z_top, float(np.max(trace["root_z"])) * 1.1)
    rows = list(reversed([z for z in ZONE_ORDER if z in touched])) or ["L_FOOT", "R_FOOT"]

    ncols = len(payloads)
    fig, axes = plt.subplots(
        3, ncols, figsize=(7.4 * ncols, 7.8), squeeze=False,
        gridspec_kw={"height_ratios": [2.6, 1.0, 1.0], "hspace": 0.34,
                     "wspace": 0.10},
    )
    fig.patch.set_facecolor(SURFACE)

    for col, (payload, label) in enumerate(zip(payloads, labels)):
        draw(axes[0][col], axes[1][col], axes[2][col], payload, label,
             col == 0, rows, z_top)

    handles = [
        mpatches.Patch(facecolor=WANTED, label="in contact — the goal asks for it"),
        mpatches.Patch(facecolor=UNWANTED, label="in contact — the goal does not"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))
    if args.title:
        fig.suptitle(args.title, fontsize=11, color=INK, x=0.008, ha="left", y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
