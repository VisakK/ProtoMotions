# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the reward landscape recorded by ``record_reward_terms.py``.

Every panel is mean over the N rollouts with a +/-1 sd band. Reward terms are
plotted in **scaled** (post-weight, post-clamp) units so panels are directly
comparable and sum to the total; raw values go in the summary table.

Figures per clip (written next to the npz):

===================================  ====================================================
``fig_01_reward_terms.png``          one panel per active term, scaled contribution vs time
``fig_02_reward_composition.png``    stacked composition of the mean total reward
``fig_03_diagnostics.png``           the weight-0 channels + rollout survival
``fig_04_contact_match.png``         contact_match split into false-positive / false-negative
``fig_05_contact_bodies.png``        per-body sim-vs-reference contact, worst offenders
``fig_06_pair_encourage.png``        per-pair earned vs available contribution
``fig_07_pair_gaps.png``             per-pair surface gap vs the phi full-credit band
``fig_08_forbid.png``                the forbid channel: gate, gap, psi, load, penalty
===================================  ====================================================

Usage::

    python data/scripts/plot_reward_terms.py --in-dir results/Reward_landscape
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical palette, validated with the dataviz skill's checker (light surface,
# 8 slots): lightness band PASS, chroma floor PASS, CVD separation PASS
# (worst adjacent dE 9.6 deutan), normal-vision floor PASS. Assigned in fixed
# order, never cycled -- past 8 series we go to small multiples instead.
PALETTE = [
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#7570B3",
    "#1B9E77",
]
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"
SURFACE = "#fcfcfb"
POS_HUE = "#0072B2"
NEG_HUE = "#D55E00"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "font.size": 9,
    }
)


def band(ax, t, arr, color, label=None, ls="-"):
    """Mean line + /-1 sd shaded band over the rollout axis."""
    m, s = np.nanmean(arr, axis=1), np.nanstd(arr, axis=1)
    ax.fill_between(t, m - s, m + s, color=color, alpha=0.16, linewidth=0)
    ax.plot(t, m, color=color, label=label, linestyle=ls)
    return m, s


def tidy(ax, ylabel=None, title=None):
    ax.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")


def gate_spans(ax, t, gate_mean, color="#000000", alpha=0.05):
    """Shade the frames where the reference demands this pair (tau > 0.5)."""
    on = gate_mean > 0.5
    if not on.any():
        return
    edges = np.diff(on.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(on) - 1]
    for a, b in zip(starts, ends):
        ax.axvspan(t[a], t[b], color=color, alpha=alpha, linewidth=0)


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def fig_reward_terms(d, out):
    t = d["t"]
    names = [str(x) for x in d["reward_names"]]
    w = d["reward_weights"]
    active = [i for i, n in enumerate(names) if w[i] != 0]
    n_p = len(active) + 1
    ncol = 3
    nrow = int(np.ceil(n_p / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.5 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for k, i in enumerate(active):
        ax = axes[k]
        c = PALETTE[k % len(PALETTE)]
        m, _ = band(ax, t, d["scaled"][:, :, i], c)
        ax.axhline(0, color=INK_MUTED, lw=0.8, ls=":")
        tidy(ax, "scaled reward", f"{names[i]}  (w={w[i]:g})")
        ax.text(
            0.98,
            0.06,
            f"mean {m.mean():+.4f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color=INK_MUTED,
        )

    ax = axes[len(active)]
    m, _ = band(ax, t, d["total"], INK)
    tidy(ax, "reward", "total_env_reward")
    ax.text(
        0.98,
        0.06,
        f"mean {m.mean():+.4f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    for ax in axes[len(active) + 1 :]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        if ax.get_visible():
            ax.set_xlabel("rollout time (s)")
    fig.suptitle(
        f"Reward terms vs rollout time — {d['motion_name']}  "
        f"(mean ± 1 sd over {int(d['num_rollouts'])} rollouts)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_composition(d, out):
    t = d["t"]
    names = [str(x) for x in d["reward_names"]]
    w = d["reward_weights"]
    active = [i for i, n in enumerate(names) if w[i] != 0]
    means = {names[i]: d["scaled"][:, :, i].mean(axis=1) for i in active}
    pos = [n for n in means if means[n].mean() >= 0]
    neg = [n for n in means if means[n].mean() < 0]
    pos.sort(key=lambda n: -means[n].mean())
    neg.sort(key=lambda n: means[n].mean())

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.stackplot(
        t,
        *[means[n] for n in pos],
        labels=pos,
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(pos))],
        edgecolor=SURFACE,
        linewidth=0.8,
    )
    if neg:
        ax.stackplot(
            t,
            *[means[n] for n in neg],
            labels=neg,
            colors=[PALETTE[(len(pos) + i) % len(PALETTE)] for i in range(len(neg))],
            edgecolor=SURFACE,
            linewidth=0.8,
        )
    ax.plot(t, d["total"].mean(axis=1), color=INK, lw=2.2, label="total (net)")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    tidy(ax, "scaled reward contribution")
    ax.set_xlabel("rollout time (s)")
    ax.set_title(
        f"What the total reward is made of — {d['motion_name']} "
        f"(mean over {int(d['num_rollouts'])} rollouts)",
        loc="left",
        fontsize=12,
    )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_diagnostics(d, out):
    t = d["t"]
    names = [str(x) for x in d["reward_names"]]
    idx = {n: i for i, n in enumerate(names)}
    diags = [n for n in names if n.startswith("diag_")]
    fig, axes = plt.subplots(1, len(diags) + 1, figsize=(3.6 * (len(diags) + 1), 3.0))
    axes = np.atleast_1d(axes).ravel()

    for k, n in enumerate(diags):
        ax = axes[k]
        arr = d["raw"][:, :, idx[n]]
        unit = "m" if "gap" in n or "err" in n else ""
        band(ax, t, arr, PALETTE[k % len(PALETTE)])
        if n == "diag_worst_body_err":
            ax.axhline(0.5, color=NEG_HUE, ls="--", lw=1.2)
            ax.text(
                t[-1], 0.5, " termination threshold", color=NEG_HUE, fontsize=7, va="bottom", ha="right"
            )
        if n == "diag_ground_fz":
            ax.axhline(1.0, color=INK_MUTED, ls=":", lw=1.0)
        if n.startswith("diag_gap"):
            ax.axhline(
                float(d["gap_target"]), color=PALETTE[2], ls="--", lw=1.2
            )
            ax.text(
                t[0],
                float(d["gap_target"]),
                " phi full credit (1.5 cm)",
                color=PALETTE[2],
                fontsize=7,
                va="bottom",
            )
        tidy(ax, unit, n)
        ax.set_xlabel("rollout time (s)")

    # survival: an env that ever exceeded the 0.5 m tracking-error threshold
    # would have been terminated during training.
    err = d["raw"][:, :, idx["diag_worst_body_err"]]
    alive = np.cumprod((err <= 0.5).astype(np.float32), axis=0)
    ax = axes[-1]
    ax.plot(t, alive.mean(axis=1) * 100, color=PALETTE[0])
    ax.set_ylim(-3, 103)
    tidy(ax, "% of rollouts", "would-still-be-alive under training terminations")
    ax.set_xlabel("rollout time (s)")

    fig.suptitle(
        f"Weight-0 diagnostics — {d['motion_name']}",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_contact_match(d, out):
    t = d["t"]
    names = [str(x) for x in d["reward_names"]]
    i_cm = names.index("contact_match_rew")
    w_cm = float(d["reward_weights"][i_cm])
    nb = d["contact_fp"].shape[2]
    fp_term = d["contact_fp"].sum(axis=2) / nb  # mean mismatch fraction, FP half
    fn_term = d["contact_fn"].sum(axis=2) / nb

    # Explicit gridspec margins rather than tight_layout: the two heatmaps carry
    # colorbars, which tight_layout cannot lay out.
    fig = plt.figure(figsize=(13, 8.8))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.15, 1, 1], hspace=0.45, top=0.92, bottom=0.06,
        left=0.09, right=0.97,
    )

    ax = fig.add_subplot(gs[0])
    band(ax, t, fp_term * w_cm, NEG_HUE, "false positive: policy touches, reference does not")
    band(ax, t, fn_term * w_cm, POS_HUE, "false negative: reference touches, policy does not")
    band(ax, t, d["scaled"][:, :, i_cm], INK, "contact_match_rew (total)")
    tidy(ax, "scaled reward")
    ax.set_title(
        f"contact_match_rew decomposed (w={w_cm:g}) — the two halves sum to the term",
        loc="left",
    )
    ax.legend(loc="lower left", ncol=3)
    ax.set_xlabel("rollout time (s)")

    body_names = [str(b) for b in d["contact_body_names"]]
    for row, (arr, cmap, what) in enumerate(
        [
            (d["contact_fp"], "Reds", "FALSE POSITIVE (sim contact, no reference contact)"),
            (d["contact_fn"], "Blues", "FALSE NEGATIVE (reference contact, no sim contact)"),
        ]
    ):
        ax = fig.add_subplot(gs[row + 1])
        img = arr.mean(axis=1).T  # [bodies, T]
        order = np.argsort(-img.mean(axis=1))
        im = ax.imshow(
            img[order],
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[t[0], t[-1], -0.5, len(body_names) - 0.5],
            interpolation="nearest",
        )
        ax.set_yticks(range(len(body_names)))
        ax.set_yticklabels([body_names[i] for i in order], fontsize=6)
        ax.set_title(f"{what} — per body, mean over rollouts", loc="left")
        ax.set_xlabel("rollout time (s)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02, label="mismatch")

    fig.suptitle(
        f"Contact matching — {d['motion_name']}  "
        f"(mean ± 1 sd over {int(d['num_rollouts'])} rollouts)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_contact_bodies(d, out, top=8):
    t = d["t"]
    body_names = [str(b) for b in d["contact_body_names"]]
    mism = (d["contact_fp"] + d["contact_fn"]).mean(axis=(0, 1))
    order = np.argsort(-mism)[:top]
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.5 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k, b in enumerate(order):
        ax = axes[k]
        band(ax, t, d["ref_contact"][:, :, b], POS_HUE, "reference label")
        band(ax, t, d["sim_contact"][:, :, b], NEG_HUE, "simulated contact")
        ax.set_ylim(-0.05, 1.05)
        tidy(ax, "contact", f"{body_names[b]}   mismatch {mism[b]:.3f}")
        if k == 0:
            ax.legend(loc="center left", fontsize=7)
    for ax in axes[len(order) :]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        if ax.get_visible():
            ax.set_xlabel("rollout time (s)")
    fig.suptitle(
        f"Where the contact mismatch lives — {d['motion_name']} "
        f"(top {len(order)} bodies by mean mismatch)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _available(d):
    """Max attainable per-pair contribution w_p/max(sum w, 0.1) -- same units as
    the earned contribution, so the two can share one axis."""
    gate, kappa = d["pair_gate"], d["pair_kappa"]
    w = gate * kappa[None, None, :]
    return w / np.maximum(w.sum(axis=2, keepdims=True), 0.1)


def fig_pair_encourage(d, out):
    t = d["t"]
    pairs = [str(p) for p in d["pair_names"]]
    avail = _available(d)
    contrib = d["pair_contrib"]
    active = [i for i in range(len(pairs)) if avail[:, :, i].mean() > 1e-6]
    ncol = 3
    nrow = int(np.ceil((len(active) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.5 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for k, i in enumerate(active):
        ax = axes[k]
        gate_spans(ax, t, d["pair_gate"][:, :, i].mean(axis=1))
        ax.plot(
            t,
            avail[:, :, i].mean(axis=1),
            color=INK_MUTED,
            ls="--",
            lw=1.4,
            label="asked for (max)",
        )
        band(ax, t, contrib[:, :, i], PALETTE[k % len(PALETTE)], "earned")
        e, a = contrib[:, :, i].mean(), avail[:, :, i].mean()
        tidy(ax, "share of term", f"{pairs[i]}  (k={d['pair_kappa'][i]:g})")
        ax.text(
            0.98,
            0.08,
            f"earned {e:.3f} / asked {a:.3f}  = {100 * e / max(a, 1e-9):.0f}%",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.5,
            color=INK_MUTED,
        )
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)

    ax = axes[len(active)]
    names = [str(x) for x in d["reward_names"]]
    band(ax, t, d["scaled"][:, :, names.index("pair_contact_rew")], INK)
    ax.set_ylim(-0.01, 0.31)
    tidy(ax, "scaled reward", "pair_contact_rew (total, w=+0.3)")
    for ax in axes[len(active) + 1 :]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        if ax.get_visible():
            ax.set_xlabel("rollout time (s)")
    fig.suptitle(
        f"Pair-encourage, per pair: earned vs asked — {d['motion_name']}\n"
        f"shaded span = the reference demands this pair",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_pair_gaps(d, out):
    t = d["t"]
    pairs = [str(p) for p in d["pair_names"]]
    gap = d["pair_gap"] * 100.0  # cm
    gt = float(d["gap_target"]) * 100
    half = (float(d["gap_target"]) + float(d["gap_sigma"]) * np.sqrt(np.log(2))) * 100
    is_g = d["pair_is_ground"]
    active = [
        i
        for i in range(len(pairs))
        if not is_g[i] and np.isfinite(gap[:, :, i]).any()
    ]
    ncol = 3
    nrow = int(np.ceil(len(active) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.5 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(active):
        ax = axes[k]
        gm = d["pair_gate"][:, :, i].mean(axis=1)
        gate_spans(ax, t, gm)
        band(ax, t, gap[:, :, i], PALETTE[k % len(PALETTE)])
        ax.axhline(gt, color=PALETTE[2], ls="--", lw=1.2)
        ax.axhline(half, color=INK_MUTED, ls=":", lw=1.0)
        tidy(ax, "surface gap (cm)", pairs[i])
        ax.set_ylim(-1, min(30, np.nanpercentile(gap[:, :, i], 98) + 2))
        # Per-element gate mask (not the rollout-mean one used for shading), so
        # this matches the summary table exactly.
        sel = d["pair_gate"][:, :, i] > 0.5
        if sel.any():
            ax.text(
                0.98,
                0.9,
                f"median gap while demanded {np.nanmedian(gap[:, :, i][sel]):.1f} cm",
                transform=ax.transAxes,
                ha="right",
                fontsize=7.5,
                color=INK_MUTED,
            )
        if k == 0:
            ax.text(
                t[0], gt, " full phi credit (1.5 cm)", color=PALETTE[2], fontsize=7, va="bottom"
            )
            ax.text(t[0], half, " phi = 0.5", color=INK_MUTED, fontsize=7, va="bottom")
    for ax in axes[len(active) :]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        if ax.get_visible():
            ax.set_xlabel("rollout time (s)")
    fig.suptitle(
        f"Body-body surface gaps vs the phi shaping band — {d['motion_name']}\n"
        f"shaded span = the reference demands this pair",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_forbid(d, out):
    t = d["t"]
    fnames = [str(x) for x in d["forbid_names"]]
    names = [str(x) for x in d["reward_names"]]
    i_f = names.index("pair_forbid_rew")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    gm = d["forbid_gate"][:, :, 0].mean(axis=1)

    ax = axes[0]
    gate_spans(ax, t, gm)
    band(ax, t, d["forbid_gap"][:, :, 0] * 100, PALETTE[0])
    ax.axhline(float(d["forbid_margin"]) * 100, color=NEG_HUE, ls="--", lw=1.2)
    ax.text(
        t[0],
        float(d["forbid_margin"]) * 100,
        " psi ramp starts (2 cm)",
        color=NEG_HUE,
        fontsize=7,
        va="bottom",
    )
    tidy(ax, "surface gap (cm)", f"{fnames[0]} gap")

    ax = axes[1]
    gate_spans(ax, t, gm)
    band(ax, t, d["forbid_psi"][:, :, 0], PALETTE[1])
    ax.set_ylim(-0.05, 1.05)
    tidy(ax, "psi", "psi(d) proximity ramp")

    ax = axes[2]
    gate_spans(ax, t, gm)
    band(ax, t, d["forbid_load"][:, :, 0], PALETTE[3])
    ax.set_ylim(-0.05, 1.05)
    tidy(ax, "load factor", "load gate (min of the two bodies)")

    ax = axes[3]
    band(ax, t, d["scaled"][:, :, i_f], NEG_HUE)
    ax.axhline(0, color=INK_MUTED, lw=0.8, ls=":")
    tidy(ax, "scaled reward", "pair_forbid_rew (w=-0.1)")

    for ax in axes:
        ax.set_xlabel("rollout time (s)")
    fig.suptitle(
        f"Forbid channel — {d['motion_name']}  (shaded span = forbid gate on)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summarize(d, out_dir):
    names = [str(x) for x in d["reward_names"]]
    w = d["reward_weights"]
    idx = {n: i for i, n in enumerate(names)}
    nb = d["contact_fp"].shape[2]
    body_names = [str(b) for b in d["contact_body_names"]]
    pairs = [str(p) for p in d["pair_names"]]
    avail = _available(d)
    err = d["raw"][:, :, idx["diag_worst_body_err"]]
    alive = np.cumprod((err <= 0.5).astype(np.float32), axis=0)

    s = {
        "motion": str(d["motion_name"]),
        "num_rollouts": int(d["num_rollouts"]),
        "steps": int(len(d["t"])),
        "clip_length_s": float(d["clip_length_s"]),
        "checkpoint": str(d["checkpoint"]),
        "decomposition_max_abs_err": float(d["probe_max_abs_err"]),
        "survival_frac_at_end": float(alive[-1].mean()),
        "worst_body_err_mean_m": float(err.mean()),
        "total_reward_mean": float(d["total"].mean()),
        "terms": {
            n: {
                "weight": float(w[i]),
                "raw_mean": float(d["raw"][:, :, i].mean()),
                "scaled_mean": float(d["scaled"][:, :, i].mean()),
            }
            for i, n in enumerate(names)
        },
        "contact_match": {
            "fp_mean": float(d["contact_fp"].sum(axis=2).mean() / nb),
            "fn_mean": float(d["contact_fn"].sum(axis=2).mean() / nb),
            "per_body": {
                body_names[b]: {
                    "fp": float(d["contact_fp"][:, :, b].mean()),
                    "fn": float(d["contact_fn"][:, :, b].mean()),
                    "ref_dwell": float((d["ref_contact"][:, :, b] > 0.5).mean()),
                    "sim_dwell": float((d["sim_contact"][:, :, b] > 0.5).mean()),
                }
                for b in range(nb)
            },
        },
        "pairs": {
            pairs[i]: {
                "kappa": float(d["pair_kappa"][i]),
                "is_ground": bool(d["pair_is_ground"][i]),
                "asked_mean": float(avail[:, :, i].mean()),
                "earned_mean": float(d["pair_contrib"][:, :, i].mean()),
                "fill_rate": float(
                    d["pair_contrib"][:, :, i].mean()
                    / max(avail[:, :, i].mean(), 1e-9)
                ),
                "gate_dwell": float((d["pair_gate"][:, :, i] > 0.5).mean()),
                "gap_median_while_demanded_cm": (
                    float(
                        np.nanmedian(
                            d["pair_gap"][:, :, i][d["pair_gate"][:, :, i] > 0.5]
                        )
                        * 100
                    )
                    if (d["pair_gate"][:, :, i] > 0.5).any()
                    and np.isfinite(d["pair_gap"][:, :, i]).any()
                    else None
                ),
                "load_median_while_demanded": (
                    float(
                        np.nanmedian(
                            d["pair_load"][:, :, i][d["pair_gate"][:, :, i] > 0.5]
                        )
                    )
                    if (d["pair_gate"][:, :, i] > 0.5).any()
                    and np.isfinite(d["pair_load"][:, :, i]).any()
                    else None
                ),
            }
            for i in range(len(pairs))
        },
        "forbid": {
            str(d["forbid_names"][0]): {
                "gate_dwell": float((d["forbid_gate"][:, :, 0] > 0.5).mean()),
                "psi_fire_duty": float((d["forbid_psi"][:, :, 0] > 0).mean()),
                "gap_median_cm": float(np.nanmedian(d["forbid_gap"][:, :, 0]) * 100),
                "penalty_mean_scaled": float(d["scaled"][:, :, idx["pair_forbid_rew"]].mean()),
            }
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))

    L = [
        f"# Reward landscape — {s['motion']}",
        "",
        f"- {s['num_rollouts']} rollouts x {s['steps']} policy steps "
        f"({s['clip_length_s']:.2f} s clip), deterministic mean actions",
        f"- checkpoint: `{s['checkpoint']}`",
        f"- decomposition self-check: max |sum(parts) − env reward| = "
        f"{s['decomposition_max_abs_err']:.2e}",
        f"- worst-body tracking error mean **{s['worst_body_err_mean_m']:.3f} m**; "
        f"**{100 * s['survival_frac_at_end']:.0f}%** of rollouts would still be alive "
        "at clip end under the training termination (0.5 m)",
        f"- mean total reward **{s['total_reward_mean']:.4f}**",
        "",
        "## Reward terms (time- and rollout-averaged)",
        "",
        "| term | weight | raw mean | scaled mean | % of |total| |",
        "|---|---|---|---|---|",
    ]
    denom = sum(abs(v["scaled_mean"]) for v in s["terms"].values()) or 1.0
    for n, v in sorted(
        s["terms"].items(), key=lambda kv: -abs(kv[1]["scaled_mean"])
    ):
        L.append(
            f"| `{n}` | {v['weight']:g} | {v['raw_mean']:.4f} | "
            f"{v['scaled_mean']:+.4f} | {100 * abs(v['scaled_mean']) / denom:.1f}% |"
        )

    L += [
        "",
        "## Contact matching",
        "",
        f"Mean mismatch fraction {s['contact_match']['fp_mean'] + s['contact_match']['fn_mean']:.4f} "
        f"= **{s['contact_match']['fp_mean']:.4f} false-positive** + "
        f"**{s['contact_match']['fn_mean']:.4f} false-negative**.",
        "",
        "| body | FP | FN | ref dwell | sim dwell |",
        "|---|---|---|---|---|",
    ]
    for b, v in sorted(
        s["contact_match"]["per_body"].items(), key=lambda kv: -(kv[1]["fp"] + kv[1]["fn"])
    )[:10]:
        L.append(
            f"| `{b}` | {v['fp']:.3f} | {v['fn']:.3f} | "
            f"{100 * v['ref_dwell']:.0f}% | {100 * v['sim_dwell']:.0f}% |"
        )

    L += [
        "",
        "## Pair-encourage, per pair",
        "",
        "`asked` is the max attainable share of the term at that instant "
        "(kappa*tau / sum kappa*tau); `earned` is what the policy actually got.",
        "",
        "| pair | k | gate dwell | asked | earned | fill | median gap while demanded | median load |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, v in sorted(s["pairs"].items(), key=lambda kv: -kv[1]["asked_mean"]):
        if v["asked_mean"] < 1e-6:
            continue
        g = (
            f"{v['gap_median_while_demanded_cm']:.1f} cm"
            if v["gap_median_while_demanded_cm"] is not None
            else "— (ground)"
        )
        ld = (
            f"{v['load_median_while_demanded']:.2f}"
            if v["load_median_while_demanded"] is not None
            else "—"
        )
        L.append(
            f"| `{p}` | {v['kappa']:g} | {100 * v['gate_dwell']:.0f}% | "
            f"{v['asked_mean']:.3f} | {v['earned_mean']:.3f} | "
            f"**{100 * v['fill_rate']:.0f}%** | {g} | {ld} |"
        )

    fk = list(s["forbid"])[0]
    fv = s["forbid"][fk]
    L += [
        "",
        "## Forbid channel",
        "",
        f"`{fk}`: gate on {100 * fv['gate_dwell']:.0f}% of frames, median gap "
        f"{fv['gap_median_cm']:.2f} cm, psi fires {100 * fv['psi_fire_duty']:.1f}% "
        f"of frames, mean scaled penalty {fv['penalty_mean_scaled']:+.4f}.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(L))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=str, default="results/Reward_landscape")
    a = ap.parse_args()
    root = Path(a.in_dir)
    npzs = sorted(root.glob("*/rollout_rewards.npz"))
    assert npzs, f"no rollout_rewards.npz under {root}"
    for p in npzs:
        d = load(p)
        out = p.parent
        print(f"plotting {d['motion_name']} ...", flush=True)
        fig_reward_terms(d, out / "fig_01_reward_terms.png")
        fig_composition(d, out / "fig_02_reward_composition.png")
        fig_diagnostics(d, out / "fig_03_diagnostics.png")
        fig_contact_match(d, out / "fig_04_contact_match.png")
        fig_contact_bodies(d, out / "fig_05_contact_bodies.png")
        fig_pair_encourage(d, out / "fig_06_pair_encourage.png")
        fig_pair_gaps(d, out / "fig_07_pair_gaps.png")
        fig_forbid(d, out / "fig_08_forbid.png")
        summarize(d, out)
        print(f"  wrote 8 figures + summary.md/json to {out}", flush=True)


if __name__ == "__main__":
    main()
