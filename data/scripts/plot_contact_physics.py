# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a ``record_contact_physics`` rollout into contact/balance figures.

Usage::

    python data/scripts/plot_contact_physics.py \
      --in-dir results/Contact_Physics_analysis          # every motion sub-folder
    python data/scripts/plot_contact_physics.py \
      --rollout results/Contact_Physics_analysis/<motion>/rollout.npz --animate

Figures written next to the rollout:

    fig_01_pair_normal_forces.png   normal force vs time, one panel per contact pair
    pairs/<n>_<body>__<other>.png   the same, one standalone plot per pair
    fig_02_torques_<group>.png      joint torques, one figure per limb, one panel per joint
    fig_03_support_bev.png          COP / COM / support polygon, bird's eye
    fig_04_stability.png            margins, support area, GRF vs weight
    fig_05_contact_area.png         effective contact area per body
    fig_06_impulse.png              cumulative impulse per pair
    fig_07_force_balance.png        sum(F) vs m(g+a), friction-cone utilisation
    fig_08_joint_power.png          mechanical power per limb
    summary.json / summary.md       the numbers behind the figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_physics_support import (  # noqa: E402
    GRAVITY,
    GROUP_TITLES,
    active_pairs,
    contact_series,
    convex_hull_2d,
    force_balance,
    geometric_band_area,
    group_dof_indices,
    load_collision_geoms,
    load_rollout,
    polygon_area,
)

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RED = "#e34948"
SHADE = "#f0efec"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f5f3", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#6da7ec", "#f4f5f3", "#f2a58c", "#c0392b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "xtick.color": INK_3, "ytick.color": INK_3,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "text.color": INK,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
    "legend.frameon": False, "lines.linewidth": 1.4, "figure.dpi": 130,
})

DEFAULT_MJCF = "data/assets/smpl/smpl_yogi03596_lowtorque.xml"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# A clip can have well over a dozen active pairs, so extend the six-colour
# categorical set and fall back to dashes once even that wraps.
CAT_EXT = CAT + ["#7a4fbf", "#00838f", "#a1651a", "#c2185b", "#4c6b1f", "#37474f"]


def body_color(index: int) -> str:
    return CAT_EXT[index % len(CAT_EXT)]


def series_style(index: int) -> dict:
    return {
        "color": body_color(index),
        "ls": ["-", "--", ":"][(index // len(CAT_EXT)) % 3],
    }


def _grid(n: int, cols: int, width: float, row_height: float, sharex=True):
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(width, row_height * rows), sharex=sharex, squeeze=False
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat, rows


def _label_bottom(axes, n: int, cols: int, text: str = "time (s)") -> None:
    """Put the x label on the last used axis of every column."""
    for column in range(cols):
        used = [i for i in range(column, n, cols)]
        if used:
            axes[used[-1]].set_xlabel(text)
            axes[used[-1]].tick_params(labelbottom=True)


# --------------------------------------------------------------------------- #
# 1 - per-pair normal force
# --------------------------------------------------------------------------- #
def figure_pair_forces(roll, pairs, out_dir: Path) -> list[Path]:
    t = roll.t
    written = []
    n = len(pairs)
    cols = min(3, max(n, 1))
    fig, axes, _ = _grid(n, cols, 13.5, 2.4)
    for k, pair in enumerate(pairs):
        ax = style(axes[k])
        magnitude = pair.magnitude
        ax.plot(t, magnitude, color=body_color(k), lw=1.2, label="|F| normal")
        ax.plot(t, pair.force[:, 2], color=INK_3, lw=0.8, alpha=0.8, label="F_z")
        ax.fill_between(t, 0, magnitude, color=body_color(k), alpha=0.12, lw=0)
        weight = roll.total_mass * GRAVITY
        ax.axhline(weight, color=INK_3, lw=0.7, ls=":")
        ax.text(t[-1], weight, " mg", fontsize=6.8, color=INK_3, va="center")
        kind = "body ↔ ground" if pair.is_ground else "body ↔ body"
        ax.set_title(f"{pair.label}   ({kind})", loc="left")
        duty = float((magnitude > 1.0).mean())
        ax.text(
            0.99, 0.93,
            f"peak {magnitude.max():.0f} N = {magnitude.max() / weight:.2f}·mg\n"
            f"contact {100 * duty:.0f}% of clip",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=INK_2,
        )
        ax.set_ylabel("N")
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)
    _label_bottom(axes, n, cols)
    fig.suptitle(
        f"1 · Normal contact force per pair — {roll.motion_name}",
        fontsize=12, fontweight="semibold", color=INK, y=1.0,
    )
    written.append(save(fig, out_dir, "fig_01_pair_normal_forces.png"))

    pair_dir = out_dir / "pairs"
    for k, pair in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
        style(ax)
        ax.plot(t, pair.magnitude, color=body_color(k), lw=1.3, label="normal |F|")
        ax.plot(t, pair.force[:, 2], color=INK_3, lw=0.9, label="F_z")
        ax.plot(
            t, np.linalg.norm(pair.friction, axis=-1), color=RED, lw=0.9,
            alpha=0.75, label="friction |F_t|",
        )
        ax.fill_between(t, 0, pair.magnitude, color=body_color(k), alpha=0.12, lw=0)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("force (N)")
        ax.legend(loc="upper right", fontsize=7.5, ncol=3)
        kind = "body ↔ ground" if pair.is_ground else "body ↔ body"
        ax.set_title(f"{pair.label}  ·  {kind}", loc="left")
        slug = f"{pair.body_name}__{pair.filter_name}".replace(" ", "")
        written.append(save(fig, pair_dir, f"{k:02d}_{slug}.png"))
    return written


# --------------------------------------------------------------------------- #
# 2 - joint torques per limb
# --------------------------------------------------------------------------- #
def figure_torques(roll, out_dir: Path) -> list[Path]:
    t = roll.t
    applied = roll.raw["torque_applied"]
    measured = roll.raw["torque_measured"]
    limits = roll.raw["effort_limit"]
    groups = group_dof_indices(roll.dof_names)
    written = []
    for gi, (group, entries) in enumerate(groups.items()):
        if not entries:
            continue
        fig, axes, _ = _grid(len(entries), 1, 11.5, 2.0)
        saturating = []
        for k, (joint, dofs) in enumerate(entries):
            ax = style(axes[k])
            for a, dof in enumerate(dofs):
                axis_name = roll.dof_names[dof][-1]
                ax.plot(
                    t, applied[:, dof], color=CAT[a % len(CAT)], lw=1.1,
                    label=f"τ_{axis_name} commanded",
                )
                ax.plot(
                    t, measured[:, dof], color=CAT[a % len(CAT)], lw=0.7, ls="--",
                    alpha=0.45, label=f"τ_{axis_name} measured",
                )
            limit = float(np.nanmax(limits[dofs])) if len(dofs) else np.nan
            if np.isfinite(limit) and limit > 0:
                ax.axhline(limit, color=RED, lw=0.7, ls=":")
                ax.axhline(-limit, color=RED, lw=0.7, ls=":")
                ax.text(t[-1], limit, " limit", fontsize=6.8, color=RED, va="center")
                sat = float((np.abs(applied[:, dofs]) >= 0.98 * limit).any(-1).mean())
                if sat > 0.01:
                    saturating.append((joint, sat))
                ax.text(
                    0.995, 0.06, f"saturated {100 * sat:.1f}% of clip",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    color=RED if sat > 0.05 else INK_3,
                )
            ax.set_title(joint, loc="left")
            ax.set_ylabel("N·m")
            if k == 0:
                ax.legend(loc="upper left", fontsize=6.6, ncol=3)
        axes[len(entries) - 1].set_xlabel("time (s)")
        note = (
            "  ·  saturating: " + ", ".join(f"{j} {100 * s:.0f}%" for j, s in saturating)
            if saturating else ""
        )
        fig.suptitle(
            f"2.{gi + 1} · Joint torque — {GROUP_TITLES[group]} — {roll.motion_name}{note}",
            fontsize=12, fontweight="semibold", color=INK, y=1.0,
        )
        written.append(save(fig, out_dir, f"fig_02_torques_{group}.png"))
    return written


# --------------------------------------------------------------------------- #
# 3 - bird's eye support view
# --------------------------------------------------------------------------- #
def _draw_polygon(ax, hull, color, alpha=0.18, lw=1.2, label=None):
    hull = np.asarray(hull)
    if len(hull) >= 3:
        ax.add_patch(MplPolygon(hull, closed=True, facecolor=color, alpha=alpha,
                                edgecolor=color, lw=lw, label=label))
    elif len(hull) == 2:
        ax.plot(hull[:, 0], hull[:, 1], color=color, lw=lw, label=label)
    elif len(hull) == 1:
        ax.plot(hull[:, 0], hull[:, 1], "o", color=color, ms=4, label=label)


def _contact_points_at(roll, step: int, ground_filter: int = 0):
    raw = roll.raw
    sel = (raw["cp_step"] == step) & (raw["cp_filter"] == ground_filter)
    return raw["cp_pos"][sel], raw["cp_body"][sel].astype(int), raw["cp_force"][sel]


def support_reference(roll) -> np.ndarray:
    """Origin for the bird's-eye plots: the mean ground contact point of the clip.

    World coordinates run to ~100 m (envs are laid out on a terrain grid), which
    swamps the centimetre-scale detail these plots are about.
    """
    ground = roll.raw["cp_pos"][roll.raw["cp_filter"] == 0]
    if len(ground) == 0:
        return roll.com[0, :2].copy()
    return ground[:, :2].mean(0)


def _key_frames(roll, series, count: int = 5, settle_s: float = 0.25):
    """Frames with a real (>=3 vertex) support polygon, past the reset transient."""
    ok = np.array([len(h) >= 3 for h in series.hulls]) & (roll.t > settle_s)
    if not ok.any():
        ok = np.array([len(h) > 0 for h in series.hulls])
    if not ok.any():
        return 0, np.zeros(0, dtype=int)
    idx = np.where(ok)[0]
    margin = np.where(ok, series.margin_com, np.inf)
    worst = int(np.nanargmin(margin))
    keys = idx[np.linspace(0, len(idx) - 1, min(count, len(idx))).astype(int)]
    return worst, keys


def figure_support_bev(roll, series, out_dir: Path) -> Path:
    ref = support_reference(roll)
    worst, keys = _key_frames(roll, series)
    com = roll.com[:, :2] - ref
    xcom = roll.xcom - ref
    cop = series.cop - ref

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.34, wspace=0.22)
    ax = style(fig.add_subplot(gs[0, 0]))
    ax.set_aspect("equal")

    finite = np.isfinite(cop[:, 0])
    for i, k in enumerate(keys):
        shade = SEQ(0.25 + 0.6 * i / max(len(keys) - 1, 1))
        _draw_polygon(ax, series.hulls[k] - ref, shade, alpha=0.16, lw=0.8)
    _draw_polygon(ax, series.hulls[worst] - ref, RED, alpha=0.16, lw=1.5,
                  label=f"support polygon @ worst margin (t={roll.t[worst]:.2f}s)")
    scatter = ax.scatter(
        cop[finite, 0], cop[finite, 1], c=roll.t[finite], cmap=SEQ, s=6, lw=0,
        zorder=3, label="centre of pressure",
    )
    ax.plot(com[:, 0], com[:, 1], color=RED, lw=1.4, label="COM (xy)")
    ax.plot(xcom[:, 0], xcom[:, 1], color=RED, lw=0.7, ls="--", alpha=0.5,
            label="XCoM (capture point)")
    bar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.015, shrink=0.72, aspect=28)
    bar.set_label("time (s)", fontsize=7.5)
    bar.outline.set_visible(False)
    ax.set_title("Bird's eye — COP, COM and support polygon", loc="left")
    ax.set_xlabel("x − x_ref (m)")
    ax.set_ylabel("y − y_ref (m)")
    ax.legend(loc="best", fontsize=7)

    ax2 = style(fig.add_subplot(gs[0, 1]))
    ax2.set_aspect("equal")
    active = np.where(series.ground_contact.any(0))[0]
    color_of = {int(b): body_color(i) for i, b in enumerate(active)}
    pos, bodies, forces = _contact_points_at(roll, worst)
    _draw_polygon(ax2, series.hulls[worst] - ref, INK_3, alpha=0.14, lw=1.3)
    seen: set[int] = set()
    for p, b, f in zip(pos, bodies, forces):
        label = roll.body_names[b] if b not in seen else None
        seen.add(int(b))
        ax2.scatter(
            p[0] - ref[0], p[1] - ref[1],
            s=20 + 110 * f / max(float(forces.max()), 1e-6),
            color=color_of.get(int(b), INK_3), zorder=3, label=label,
        )
    labelled: set[int] = set()
    for vertex, owner in zip(np.atleast_2d(series.hulls[worst] - ref),
                             series.hull_owners[worst]):
        if owner in labelled:
            continue
        labelled.add(owner)
        ax2.annotate(roll.body_names[owner], vertex, textcoords="offset points",
                     xytext=(4, 4), fontsize=7, color=INK_2)
    ax2.plot(*com[worst], "X", color=RED, ms=10, label="COM", zorder=4)
    if np.all(np.isfinite(cop[worst])):
        ax2.plot(*cop[worst], "o", color=INK, ms=6, label="COP", zorder=4)
    ax2.plot(*xcom[worst], "^", color=RED, ms=7, alpha=0.6, label="XCoM", zorder=4)
    ax2.set_title(
        f"Worst margin among frames with a real polygon: t={roll.t[worst]:.2f}s, "
        f"COM {series.margin_com[worst]:+.3f} m, "
        f"support {1e4 * series.support_area[worst]:.0f} cm² (marker size ∝ force)",
        loc="left",
    )
    ax2.set_xlabel("x − x_ref (m)")
    ax2.set_ylabel("y − y_ref (m)")
    ax2.legend(loc="best", fontsize=7)

    ax3 = style(fig.add_subplot(gs[1, :]))
    vertex_mask = np.zeros_like(series.ground_contact)
    for t_idx, owners_t in enumerate(series.hull_owners):
        for owner in owners_t:
            vertex_mask[t_idx, owner] = True
    for row, b in enumerate(active):
        ax3.fill_between(roll.t, row - 0.38, row + 0.38,
                         where=series.ground_contact[:, b],
                         color=color_of[int(b)], alpha=0.22, lw=0, step="mid")
        ax3.fill_between(roll.t, row - 0.24, row + 0.24, where=vertex_mask[:, b],
                         color=color_of[int(b)], alpha=0.95, lw=0, step="mid")
    ax3.set_yticks(range(len(active)))
    ax3.set_yticklabels([roll.body_names[b] for b in active])
    ax3.set_xlabel("time (s)")
    ax3.set_ylim(-0.7, len(active) - 0.3)
    ax3.grid(axis="y", visible=False)
    ax3.set_title(
        "Which contacts build the support polygon — pale: touching the ground, "
        "solid: a vertex of the polygon",
        loc="left",
    )
    fig.suptitle(
        f"3 · Support geometry — {roll.motion_name}",
        fontsize=12, fontweight="semibold", color=INK, y=0.99,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_03_support_bev.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_stability(roll, series, out_dir: Path) -> Path:
    t = roll.t
    fig, axes = plt.subplots(4, 1, figsize=(12, 8.4), sharex=True)
    ax = style(axes[0])
    ax.plot(t, series.margin_com, color=RED, lw=1.2, label="COM → polygon edge")
    ax.plot(t, series.margin_xcom, color=RED, lw=0.8, ls="--", alpha=0.6,
            label="XCoM → polygon edge")
    ax.plot(t, series.margin_cop, color=CAT[0], lw=0.9, alpha=0.8,
            label="COP → polygon edge")
    ax.axhline(0, color=INK_3, lw=0.8)
    ax.fill_between(t, series.margin_com, 0, where=series.margin_com < 0,
                    color=RED, alpha=0.15, lw=0)
    ax.set_ylabel("m")
    ax.set_title(
        "Static stability margin (positive = inside the support polygon)", loc="left"
    )
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax = style(axes[1])
    ax.plot(t, series.support_area, color=CAT[2], lw=1.2)
    ax.fill_between(t, 0, series.support_area, color=CAT[2], alpha=0.15, lw=0)
    ax.set_ylabel("m²")
    ax.set_title("Support polygon area", loc="left")

    ax = style(axes[2])
    ax.plot(t, series.ground_contact.sum(1), color=CAT[0], lw=1.1, label="bodies touching")
    ax.plot(t, series.n_points.sum(1), color=INK_3, lw=0.8, label="PhysX contact points")
    ax.set_ylabel("count")
    ax.set_title("Ground contact count", loc="left")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = style(axes[3])
    weight = roll.total_mass * GRAVITY
    ax.plot(t, series.grf[:, 2] / weight, color=INK, lw=1.0, label="ΣF_z / mg")
    ax.plot(t, np.linalg.norm(series.grf_tangential, axis=-1) / weight, color=RED,
            lw=0.9, alpha=0.8, label="|ΣF_tangential| / mg")
    ax.axhline(1.0, color=INK_3, lw=0.7, ls=":")
    ax.set_ylabel("× body weight")
    ax.set_xlabel("time (s)")
    ax.set_title("Ground reaction force", loc="left")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(
        f"4 · Balance over time — {roll.motion_name}",
        fontsize=12, fontweight="semibold", color=INK, y=1.0,
    )
    return save(fig, out_dir, "fig_04_stability.png")


# --------------------------------------------------------------------------- #
# 5 - contact area
# --------------------------------------------------------------------------- #
def figure_contact_area(roll, series, pairs, out_dir: Path, band: float,
                        geoms) -> Path:
    idx, band_area = geometric_band_area(roll, geoms, band=band)
    idx_loose, band_area_loose = geometric_band_area(roll, geoms, band=roll.contact_offset)
    active = np.where(series.ground_contact.any(0) | (band_area.max(0) > 0))[0]
    if len(active) == 0:
        active = np.array([0])
    cols = min(2, len(active))
    fig, axes, _ = _grid(len(active), cols, 12.5, 2.5)
    for k, b in enumerate(active):
        ax = style(axes[k])
        ax.fill_between(
            roll.t[idx_loose], 0, 1e4 * band_area_loose[:, b], color=CAT[0], alpha=0.14,
            lw=0,
            label=f"collider within {roll.contact_offset * 1000:.0f} mm (PhysX contact offset)",
        )
        ax.plot(roll.t[idx], 1e4 * band_area[:, b], color=CAT[0], lw=1.2,
                label=f"collider within {band * 1000:.0f} mm of the floor")
        ax.plot(roll.t, 1e4 * series.area_manifold[:, b], color=RED, lw=1.0, alpha=0.85,
                label="PhysX contact manifold hull (measured)")
        ax.set_title(roll.body_names[b], loc="left")
        ax.set_ylabel("cm²")
        peak = max(band_area[:, b].max(), series.area_manifold[:, b].max())
        ax.text(
            0.99, 0.92,
            f"peak {1e4 * peak:.1f} cm²  ·  max {series.n_points[:, b].max()} contact points",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_2,
        )
        if k == 0:
            ax.legend(loc="upper left", fontsize=6.6)
    _label_bottom(axes, len(active), cols)
    fig.suptitle(
        f"5 · Effective contact area per body — {roll.motion_name}  ·  rigid contact has "
        "no true area; the band curves are a geometric model, the manifold hull is measured",
        fontsize=11.5, fontweight="semibold", color=INK, y=1.0,
    )
    return save(fig, out_dir, "fig_05_contact_area.png")


# --------------------------------------------------------------------------- #
# 6 - impulse
# --------------------------------------------------------------------------- #
def figure_impulse(roll, pairs, out_dir: Path) -> Path:
    t = roll.t
    dt = roll.dt_phys
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), sharex=True)
    ax = style(axes[0])
    for k, pair in enumerate(pairs):
        ax.plot(t, pair.impulse(dt), lw=1.2, label=pair.label, **series_style(k))
    ax.set_ylabel("N·s")
    ax.set_title("Cumulative normal impulse ∫|F| dt per pair", loc="left")
    ax.legend(loc="upper left", fontsize=6.8, ncol=3)

    ax = style(axes[1])
    for k, pair in enumerate(pairs):
        ax.plot(t, pair.magnitude * dt * 1e3, lw=0.9, alpha=0.85, **series_style(k))
    ax.set_ylabel("mN·s per physics step")
    ax.set_xlabel("time (s)")
    ax.set_title(
        f"Per-step impulse (physics step = {1e3 * dt:.2f} ms) — spikes are impacts",
        loc="left",
    )
    fig.suptitle(
        f"6 · Contact impulse — {roll.motion_name}",
        fontsize=12, fontweight="semibold", color=INK, y=1.0,
    )
    return save(fig, out_dir, "fig_06_impulse.png")


# --------------------------------------------------------------------------- #
# 7 - force balance + friction cone
# --------------------------------------------------------------------------- #
def figure_force_balance(roll, series, pairs, out_dir: Path) -> Path:
    t = roll.t
    balance = force_balance(roll, series)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), sharex=True)
    ax = style(axes[0])
    ax.plot(t, balance["grf_z"], color=INK, lw=1.1, label="measured ΣF_z (contacts)")
    ax.plot(t, balance["predicted_z"], color=CAT[0], lw=0.9, ls="--",
            label="m·(g + a_z of COM)")
    ax.axhline(roll.total_mass * GRAVITY, color=INK_3, lw=0.7, ls=":")
    ax.set_ylabel("N")
    residual = balance["residual_z"]
    ax.set_title(
        "Vertical force balance — residual mean "
        f"{residual.mean():+.1f} N, rms {np.sqrt((residual ** 2).mean()):.1f} N "
        f"({100 * np.sqrt((residual ** 2).mean()) / (roll.total_mass * GRAVITY):.1f}% of mg)",
        loc="left",
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = style(axes[1])
    ax.plot(t, roll.com[:, 2], color=CAT[2], lw=1.2, label="COM height")
    ax.set_ylabel("m")
    ax.set_title("Centre-of-mass height", loc="left")
    ax.legend(loc="upper right", fontsize=7)

    ax = style(axes[2])
    for k, pair in enumerate(pairs):
        if not pair.is_ground:
            continue
        ax.plot(t, pair.cone_utilisation(roll.mu), lw=0.9, label=pair.label,
                **series_style(k))
    ax.axhline(1.0, color=RED, lw=0.8, ls=":")
    ax.text(t[-1], 1.0, " slip", fontsize=7, color=RED, va="center")
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("|F_t| / (μ·F_n)")
    ax.set_xlabel("time (s)")
    saturated = [
        float(np.nanmean(p.cone_utilisation(roll.mu) > 0.98))
        for p in pairs if p.is_ground
    ]
    ax.set_title(
        f"Friction-cone utilisation — effective μ = {roll.mu:g} "
        f"(terrain {roll.mu_terrain:g} combined '{roll.friction_combine_mode}' with the "
        f"robot's {roll.mu_robot:g}); at 1.0 the contact is sliding. "
        f"Ground pairs ride the cone {100 * float(np.nanmax(saturated or [0])):.0f}% "
        "of their contact time at worst",
        loc="left",
    )
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.suptitle(
        f"7 · Is the motion dynamically consistent? — {roll.motion_name}",
        fontsize=12, fontweight="semibold", color=INK, y=1.0,
    )
    return save(fig, out_dir, "fig_07_force_balance.png")


def figure_joint_power(roll, out_dir: Path) -> Path:
    t = roll.t
    power = roll.raw["torque_applied"] * roll.raw["dof_vel"]
    groups = group_dof_indices(roll.dof_names)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    style(ax)
    for k, (group, entries) in enumerate(groups.items()):
        dofs = [d for _, idx in entries for d in idx]
        ax.plot(t, np.abs(power[:, dofs]).sum(1), color=CAT[k % len(CAT)], lw=1.1,
                label=GROUP_TITLES[group])
    ax.set_ylabel("W")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.set_title("Absolute mechanical power |τ·ω| summed per limb", loc="left")
    fig.suptitle(
        f"8 · Actuation effort — {roll.motion_name}",
        fontsize=12, fontweight="semibold", color=INK, y=1.02,
    )
    return save(fig, out_dir, "fig_08_joint_power.png")


# --------------------------------------------------------------------------- #
def animate_support(roll, series, out_dir: Path, max_frames: int = 400) -> Path | None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    stride = max(1, roll.num_substeps // max_frames)
    frames = range(0, roll.num_substeps, stride)
    all_pts = roll.raw["cp_pos"][roll.raw["cp_filter"] == 0]
    if len(all_pts) == 0:
        return None
    ref = support_reference(roll)
    pts = all_pts[:, :2] - ref
    com = roll.com[:, :2] - ref
    pad = 0.15
    xlim = (min(pts[:, 0].min(), com[:, 0].min()) - pad,
            max(pts[:, 0].max(), com[:, 0].max()) + pad)
    ylim = (min(pts[:, 1].min(), com[:, 1].min()) - pad,
            max(pts[:, 1].max(), com[:, 1].max()) + pad)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    style(ax)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def draw(step: int):
        ax.clear()
        style(ax)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _draw_polygon(ax, series.hulls[step] - ref, INK_3, alpha=0.18, lw=1.2)
        pos, bodies, forces = _contact_points_at(roll, step)
        if len(pos):
            ax.scatter(pos[:, 0] - ref[0], pos[:, 1] - ref[1],
                       s=15 + 120 * forces / max(float(forces.max()), 1e-6),
                       color=CAT[0], zorder=3)
            for vertex, owner in zip(np.atleast_2d(series.hulls[step] - ref),
                                     series.hull_owners[step]):
                ax.annotate(roll.body_names[owner], vertex, textcoords="offset points",
                            xytext=(3, 3), fontsize=6.5, color=INK_2)
        ax.plot(*com[step], "X", color=RED, ms=10)
        if np.all(np.isfinite(series.cop[step])):
            ax.plot(*(series.cop[step] - ref), "o", color=INK, ms=6)
        ax.plot(*(roll.xcom[step] - ref), "^", color=RED, ms=7, alpha=0.6)
        ax.set_title(
            f"t = {roll.t[step]:5.2f} s   COM margin {series.margin_com[step]:+.3f} m   "
            f"support {series.support_area[step]:.4f} m²",
            loc="left",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 * roll.dt_phys * stride)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "support_bev.gif"
    anim.save(path, writer=PillowWriter(fps=min(30, int(1 / (roll.dt_phys * stride)))))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def support_sets(roll, series, top: int = 5) -> list[dict]:
    """How long the body spends on each distinct set of grounded bodies.

    This is the cheapest test of whether the pose was actually reached: crow is
    hands-only, warrior III is one foot, and anything else in the list is the
    policy propping itself on something the pose does not use.
    """
    counts: dict[tuple[str, ...], int] = {}
    for t in range(roll.num_substeps):
        key = tuple(
            sorted(roll.body_names[b] for b in np.where(series.ground_contact[t])[0])
        )
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return [
        {"bodies": list(bodies), "frac_of_clip": count / roll.num_substeps}
        for bodies, count in ordered
    ]


def build_summary(roll, series, pairs, geoms, band: float) -> dict:
    balance = force_balance(roll, series)
    weight = roll.total_mass * GRAVITY
    supported = np.array([len(h) >= 3 for h in series.hulls])
    idx, band_area = geometric_band_area(roll, geoms, band=band)
    applied = roll.raw["torque_applied"]
    limits = roll.raw["effort_limit"]
    with np.errstate(invalid="ignore"):
        saturation = np.abs(applied) >= 0.98 * limits[None, :]
    per_joint_sat = {
        roll.dof_names[d]: float(saturation[:, d].mean())
        for d in range(applied.shape[1])
        if saturation[:, d].mean() > 0.01
    }
    return {
        "motion": roll.motion_name,
        "checkpoint": str(roll.raw["checkpoint"]),
        "duration_s": float(roll.t[-1]),
        "physics_dt_s": roll.dt_phys,
        "control_dt_s": roll.dt_ctrl,
        "total_mass_kg": roll.total_mass,
        "body_weight_N": weight,
        "friction": {
            "mu_effective": roll.mu,
            "mu_terrain": roll.mu_terrain,
            "mu_robot_shapes": roll.mu_robot,
            "combine_mode": roll.friction_combine_mode,
        },
        "force_balance": {
            "grf_over_weight_mean": float(np.mean(balance["grf_over_weight"])),
            "residual_z_mean_N": float(balance["residual_z"].mean()),
            "residual_z_rms_N": float(np.sqrt((balance["residual_z"] ** 2).mean())),
            "residual_z_rms_pct_of_mg": float(
                100 * np.sqrt((balance["residual_z"] ** 2).mean()) / weight
            ),
        },
        "support": {
            "frac_time_with_polygon": float(supported.mean()),
            "frac_time_support_degenerate": float(
                np.mean([0 < len(h) < 3 for h in series.hulls])
            ),
            "frac_time_airborne": float(np.mean([len(h) == 0 for h in series.hulls])),
            "com_margin_min_while_supported_m": float(
                np.nanmin(np.where(supported, series.margin_com, np.nan))
            ) if supported.any() else float("nan"),
            "support_area_mean_m2": float(np.mean(series.support_area[supported]))
            if supported.any() else 0.0,
            "support_area_max_m2": float(series.support_area.max()),
            "com_margin_mean_m": float(np.nanmean(series.margin_com)),
            "com_margin_min_m": float(np.nanmin(series.margin_com))
            if np.isfinite(series.margin_com).any() else float("nan"),
            "frac_time_com_outside": float(
                np.nanmean((series.margin_com < 0).astype(float))
            ),
            "frac_time_xcom_outside": float(
                np.nanmean((series.margin_xcom < 0).astype(float))
            ),
            "cop_margin_min_m": float(np.nanmin(series.margin_cop))
            if np.isfinite(series.margin_cop).any() else float("nan"),
            "support_sets": support_sets(roll, series),
        },
        "pairs": [
            {
                "pair": pair.label,
                "kind": "body-ground" if pair.is_ground else "body-body",
                "peak_normal_N": float(pair.magnitude.max()),
                "peak_over_bodyweight": float(pair.magnitude.max() / weight),
                # A single-substep maximum can be a solver recovery impulse; the
                # 42 ms moving average is the load the body actually carries.
                "peak_normal_42ms_N": float(
                    np.convolve(pair.magnitude, np.ones(5) / 5, mode="same").max()
                ),
                "p99_normal_when_active_N": float(
                    np.percentile(pair.magnitude[pair.magnitude > 1], 99)
                ) if (pair.magnitude > 1).any() else 0.0,
                "mean_normal_when_active_N": float(
                    pair.magnitude[pair.magnitude > 1].mean()
                ) if (pair.magnitude > 1).any() else 0.0,
                "contact_duty": float((pair.magnitude > 1).mean()),
                "total_impulse_Ns": float(pair.impulse(roll.dt_phys)[-1]),
                "friction_impulse_Ns": float(pair.friction_impulse(roll.dt_phys)[-1]),
                "cone_utilisation_p95": float(
                    np.nanpercentile(pair.cone_utilisation(roll.mu), 95)
                ) if np.isfinite(pair.cone_utilisation(roll.mu)).any() else float("nan"),
                "frac_contact_time_at_slip_boundary": float(
                    np.nanmean(pair.cone_utilisation(roll.mu) > 0.98)
                ) if np.isfinite(pair.cone_utilisation(roll.mu)).any() else float("nan"),
                "max_manifold_points": int(pair.n_points.max()),
                "max_manifold_area_cm2": float(1e4 * pair.area.max()),
            }
            for pair in pairs
        ],
        "contact_area": {
            roll.body_names[b]: {
                "max_band_area_cm2": float(1e4 * band_area[:, b].max()),
                "max_manifold_area_cm2": float(1e4 * series.area_manifold[:, b].max()),
                "band_mm": 1e3 * band,
            }
            for b in range(len(roll.body_names))
            if band_area[:, b].max() > 0 or series.area_manifold[:, b].max() > 0
        },
        "torque": {
            "peak_abs_Nm": float(np.abs(applied).max()),
            "mean_abs_Nm": float(np.abs(applied).mean()),
            "joints_saturating_frac": per_joint_sat,
            "peak_limb_power_W": {
                GROUP_TITLES[g]: float(
                    np.abs(
                        roll.raw["torque_applied"][:, [d for _, i in e for d in i]]
                        * roll.raw["dof_vel"][:, [d for _, i in e for d in i]]
                    ).sum(1).max()
                )
                for g, e in group_dof_indices(roll.dof_names).items()
                if e
            },
        },
        "tracking": {
            "max_body_error_mean_m": float(np.mean(roll.raw["ctrl_track_err"])),
            "max_body_error_max_m": float(np.max(roll.raw["ctrl_track_err"])),
        },
    }


def summary_markdown(summary: dict) -> str:
    lines = [f"# Contact physics — {summary['motion']}", ""]
    lines.append(
        f"- clip {summary['duration_s']:.2f} s, mass {summary['total_mass_kg']:.1f} kg "
        f"(weight {summary['body_weight_N']:.0f} N), physics {1e3 * summary['physics_dt_s']:.2f} ms"
    )
    fb = summary["force_balance"]
    lines.append(
        f"- vertical balance: ΣF_z/mg = {fb['grf_over_weight_mean']:.3f}, residual rms "
        f"{fb['residual_z_rms_N']:.1f} N ({fb['residual_z_rms_pct_of_mg']:.1f}% of mg)"
    )
    sup = summary["support"]
    lines.append(
        f"- support: a real polygon exists {100 * sup['frac_time_with_polygon']:.0f}% of the "
        f"clip (point/line support {100 * sup['frac_time_support_degenerate']:.0f}%, "
        f"airborne {100 * sup['frac_time_airborne']:.0f}%), mean area "
        f"{1e4 * sup['support_area_mean_m2']:.0f} cm²"
    )
    lines.append(
        f"- COM margin: mean {sup['com_margin_mean_m']:+.3f} m, min while supported "
        f"{sup['com_margin_min_while_supported_m']:+.3f} m; COM outside the polygon "
        f"{100 * sup['frac_time_com_outside']:.0f}% of the clip, XCoM outside "
        f"{100 * sup['frac_time_xcom_outside']:.0f}%"
    )
    tr = summary["tracking"]
    lines.append(
        f"- tracking: worst-body error mean {tr['max_body_error_mean_m']:.3f} m, "
        f"max {tr['max_body_error_max_m']:.3f} m"
    )
    lines += ["", "## What is on the floor", ""]
    for entry in sup["support_sets"]:
        bodies = ", ".join(entry["bodies"]) if entry["bodies"] else "(airborne)"
        lines.append(f"- {100 * entry['frac_of_clip']:4.1f}% of the clip: {bodies}")
    fr = summary["friction"]
    lines.append(
        f"- friction: effective μ = {fr['mu_effective']:g} (terrain {fr['mu_terrain']:g} "
        f"combined '{fr['combine_mode']}' with the robot shapes' {fr['mu_robot_shapes']:g})"
    )
    lines += ["", "## Contact pairs", "",
              "| pair | kind | peak (N) | peak / mg | peak over 42 ms (N) | duty | "
              "impulse (N·s) | friction cone p95 | at slip boundary | max points | "
              "max manifold area (cm²) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in summary["pairs"]:
        lines.append(
            f"| {p['pair']} | {p['kind']} | {p['peak_normal_N']:.0f} | "
            f"{p['peak_over_bodyweight']:.2f} | {p['peak_normal_42ms_N']:.0f} | "
            f"{100 * p['contact_duty']:.0f}% | "
            f"{p['total_impulse_Ns']:.0f} | {p['cone_utilisation_p95']:.2f} | "
            f"{100 * p['frac_contact_time_at_slip_boundary']:.0f}% | "
            f"{p['max_manifold_points']} | {p['max_manifold_area_cm2']:.1f} |"
        )
    area = summary["contact_area"]
    if area:
        band_mm = next(iter(area.values()))["band_mm"]
        lines += ["", "## Effective contact area", "",
                  f"| body | geometric band ({band_mm:.0f} mm) cm² | PhysX manifold hull cm² |",
                  "|---|---|---|"]
        for body, values in area.items():
            lines.append(
                f"| {body} | {values['max_band_area_cm2']:.1f} | "
                f"{values['max_manifold_area_cm2']:.1f} |"
            )
    torque = summary["torque"]
    lines += ["", "## Actuation", "",
              f"- peak |τ| {torque['peak_abs_Nm']:.0f} N·m, mean |τ| {torque['mean_abs_Nm']:.1f} N·m"]
    if torque["joints_saturating_frac"]:
        worst = sorted(torque["joints_saturating_frac"].items(), key=lambda kv: -kv[1])[:8]
        lines.append(
            "- torque saturation: "
            + ", ".join(f"{k} {100 * v:.0f}%" for k, v in worst)
        )
    lines.append(
        "- peak limb power: "
        + ", ".join(f"{k} {v:.0f} W" for k, v in torque["peak_limb_power_W"].items())
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def process(rollout_path: Path, band: float, mjcf: str, animate: bool) -> dict:
    roll = load_rollout(rollout_path)
    out_dir = rollout_path.parent if rollout_path.is_file() else rollout_path
    series = contact_series(roll)
    pairs = active_pairs(roll)
    geoms = load_collision_geoms(mjcf, roll.body_names)

    figure_pair_forces(roll, pairs, out_dir)
    figure_torques(roll, out_dir)
    figure_support_bev(roll, series, out_dir)
    figure_stability(roll, series, out_dir)
    figure_contact_area(roll, series, pairs, out_dir, band, geoms)
    figure_impulse(roll, pairs, out_dir)
    figure_force_balance(roll, series, pairs, out_dir)
    figure_joint_power(roll, out_dir)
    if animate:
        animate_support(roll, series, out_dir)

    summary = build_summary(roll, series, pairs, geoms, band)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(summary_markdown(summary))
    print(f"wrote {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=str, default=None,
                        help="A single rollout.npz (or the folder holding it).")
    parser.add_argument("--in-dir", type=str, default=None,
                        help="Root holding one sub-folder per motion.")
    parser.add_argument("--band", type=float, default=0.005,
                        help="Compliance band above the floor for the geometric area model (m).")
    parser.add_argument("--mjcf", type=str, default=DEFAULT_MJCF)
    parser.add_argument("--animate", action="store_true",
                        help="Also render support_bev.gif (slow).")
    args = parser.parse_args()

    targets: list[Path] = []
    if args.rollout:
        path = Path(args.rollout)
        targets.append(path if path.is_file() else path / "rollout.npz")
    if args.in_dir:
        targets += sorted(Path(args.in_dir).glob("*/rollout.npz"))
    if not targets:
        raise SystemExit("nothing to plot: pass --rollout or --in-dir")

    summaries = []
    for target in targets:
        summaries.append(process(target, args.band, args.mjcf, args.animate))
    if args.in_dir and len(summaries) > 1:
        root = Path(args.in_dir)
        (root / "summary_all.json").write_text(json.dumps(summaries, indent=2))
        (root / "summary_all.md").write_text(
            "\n\n---\n\n".join(summary_markdown(s) for s in summaries)
        )
        print(f"wrote {root / 'summary_all.md'}")


if __name__ == "__main__":
    main()
