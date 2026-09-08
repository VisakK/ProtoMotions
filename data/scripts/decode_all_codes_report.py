# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring and pictures for ``decode_all_codes.py``.

Kept separate so the rollout script owns the simulator and this owns matplotlib,
and so a finished ``rollouts.npz`` can be re-scored and re-drawn without an Isaac
launch (``python data/scripts/decode_all_codes_report.py --npz <dir>``).

The montage is the point of the experiment as much as the numbers are.  The FSQ
code is 4 scalars, so the 625 codes tile a 25x25 grid exactly: the outer 5x5
block index is ``(scalar0, scalar1)`` and the position inside a block is
``(scalar2, scalar3)``.  Each cell draws that code's terminal pose as a sagittal
stick figure -- heading-normalised and pelvis-centred, i.e. the same frame the
goal metric uses -- over a ghost of the pose that was commanded.  A code that
held is a green figure on top of its ghost; a code that left is a red figure
somewhere else entirely, and *where* it went is legible at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np


# --------------------------------------------------------------------------- #
# Geometry (the project's goal metric, vectorised)
# --------------------------------------------------------------------------- #
def heading_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
    """``calc_heading`` written out: yaw of the root's rotated x axis."""
    x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]
    return np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def normalize(pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    """``[..., B, 3]`` pelvis-relative, heading-normalised body positions."""
    yaw = heading_yaw(root_quat)
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel = pos - pos[..., 0:1, :]
    return np.stack(
        [
            rel[..., 0] * c[..., None] - rel[..., 1] * s[..., None],
            rel[..., 0] * s[..., None] + rel[..., 1] * c[..., None],
            rel[..., 2],
        ],
        axis=-1,
    )


def bones_of(common_body_names: List[str], parent_indices: List[int]):
    """``(child, parent, colour)`` per bone; left red, right blue, axial grey."""
    out = []
    for child, parent in enumerate(parent_indices):
        if parent < 0:
            continue
        name = common_body_names[child]
        colour = "#d62728" if name.startswith("L_") else (
            "#1f77b4" if name.startswith("R_") else "#444444"
        )
        out.append((child, parent, colour))
    return out


def departure_step(err: np.ndarray, threshold: float) -> int:
    """First step at which the pose error crosses ``threshold``; -1 if never."""
    bad = np.nonzero(err > threshold)[0]
    return int(bad[0]) if bad.size else -1


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #
def montage(path: Path, poses, ghost, errors, num_levels, title, threshold):
    """25x25 sagittal stick figures, one per code, over the commanded ghost.

    Everything is drawn into ONE axes with per-cell offsets rather than 625
    subplots: matplotlib spends its time in axes construction, and a single
    LineCollection renders the whole code space in about a second.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    side = num_levels ** 2
    cell = 2.4
    fig, ax = plt.subplots(figsize=(side * 0.62, side * 0.66), dpi=110)
    segs, colours, widths = [], [], []
    ghost_segs = []
    bones = montage.bones

    err_lo, err_hi = threshold, max(float(np.nanpercentile(errors, 95)), threshold * 2)
    cmap = plt.get_cmap("RdYlGn_r")

    for code_index in range(poses.shape[0]):
        s0, s1, s2, s3 = np.unravel_index(code_index, (num_levels,) * 4)
        row = s0 * num_levels + s2
        col = s1 * num_levels + s3
        ox, oy = col * cell, -row * cell
        p = poses[code_index]
        g = ghost
        err = errors[code_index]
        frac = np.clip((err - err_lo) / max(err_hi - err_lo, 1e-6), 0.0, 1.0)
        colour = cmap(frac) if err > err_lo else "#1a9850"
        for child, parent, _c in bones:
            ghost_segs.append(
                [(ox + g[parent, 0], oy + g[parent, 2]), (ox + g[child, 0], oy + g[child, 2])]
            )
            segs.append(
                [(ox + p[parent, 0], oy + p[parent, 2]), (ox + p[child, 0], oy + p[child, 2])]
            )
            colours.append(colour)
            widths.append(1.0)

    ax.add_collection(LineCollection(ghost_segs, colors="#c8c8c8", linewidths=0.7, zorder=1))
    ax.add_collection(LineCollection(segs, colors=colours, linewidths=widths, zorder=2))
    # Block separators mark the (scalar0, scalar1) structure.
    for k in range(num_levels + 1):
        ax.axvline(k * num_levels * cell - cell * 0.5, color="#dddddd", lw=0.8, zorder=0)
        ax.axhline(-(k * num_levels * cell - cell * 0.5), color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlim(-cell, side * cell)
    ax.set_ylim(-side * cell, cell)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")
    fig.text(
        0.005, 0.005,
        "sagittal view, pelvis-centred and heading-normalised; grey = commanded pose. "
        f"green = held (<{threshold:.2f} m), red = departed. "
        "outer 5x5 grid = FSQ scalars 0,1; inner = scalars 2,3.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def heatmaps(path: Path, errors, depart_s, num_levels, title, threshold, horizon_s):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    side = num_levels ** 2
    grid_err = np.full((side, side), np.nan)
    grid_dep = np.full((side, side), np.nan)
    for code_index in range(errors.shape[0]):
        s0, s1, s2, s3 = np.unravel_index(code_index, (num_levels,) * 4)
        grid_err[s0 * num_levels + s2, s1 * num_levels + s3] = errors[code_index]
        grid_dep[s0 * num_levels + s2, s1 * num_levels + s3] = depart_s[code_index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=120)
    im0 = axes[0].imshow(grid_err, cmap="RdYlGn_r", vmin=0.0,
                         vmax=max(float(np.nanpercentile(grid_err, 95)), threshold * 2))
    axes[0].set_title("goal-pose error at the end of the window (m)", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(grid_dep, cmap="viridis", vmin=0.0, vmax=horizon_s)
    axes[1].set_title(f"seconds until the pose error crosses {threshold:.2f} m "
                      f"({horizon_s:.1f} = never)", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        for k in range(num_levels + 1):
            ax.axvline(k * num_levels - 0.5, color="w", lw=0.8)
            ax.axhline(k * num_levels - 0.5, color="w", lw=0.8)
        ax.set_xlabel("scalar 1 (blocks) / scalar 3 (within)", fontsize=8)
        ax.set_ylabel("scalar 0 (blocks) / scalar 2 (within)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sparklines(path: Path, err_traces, num_levels, title, threshold, dt):
    """Per-code goal-pose-error trace, tiled on the same 25x25 layout."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    side = num_levels ** 2
    steps = err_traces.shape[1]
    t = np.arange(steps) * dt
    cell_w, cell_h = 1.15, 1.0
    y_max = max(float(np.percentile(err_traces, 99)), threshold * 3)
    segs, colours, refs = [], [], []
    for code_index in range(err_traces.shape[0]):
        s0, s1, s2, s3 = np.unravel_index(code_index, (num_levels,) * 4)
        ox = (s1 * num_levels + s3) * cell_w
        oy = -(s0 * num_levels + s2) * cell_h
        x = ox + t / max(t[-1], 1e-6) * cell_w * 0.9
        y = oy + np.clip(err_traces[code_index], 0, y_max) / y_max * cell_h * 0.9
        segs.append(np.stack([x, y], axis=-1))
        held = err_traces[code_index].max() <= threshold
        colours.append("#1a9850" if held else "#d73027")
        ref = oy + threshold / y_max * cell_h * 0.9
        refs.append([(ox, ref), (ox + cell_w * 0.9, ref)])

    fig, ax = plt.subplots(figsize=(side * 0.42, side * 0.40), dpi=110)
    ax.add_collection(LineCollection(refs, colors="#bbbbbb", linewidths=0.5, zorder=1))
    ax.add_collection(LineCollection(segs, colors=colours, linewidths=0.8, zorder=2))
    ax.set_xlim(-cell_w * 0.2, side * cell_w)
    ax.set_ylim(-side * cell_h, cell_h)
    ax.axis("off")
    ax.set_title(f"{title}   (each cell: goal-pose error vs time, 0..{y_max:.2f} m; "
                 f"grey line = {threshold:.2f} m)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def report(*, out_dir, plans, pose_err, iou, positions, root_rot, terminated,
           codes, state_of_env, slot_of_env, num_codes, num_levels, num_scalars,
           controls, dt, hold_threshold, videos, body_names, common_body_names,
           parent_indices, conditionable, goal_poses):
    import csv

    from protomotions.agents.evaluators.sequence_viz import render_stick_video

    steps = pose_err.shape[0]
    horizon_s = steps * dt
    bones = bones_of(common_body_names, parent_indices)
    montage.bones = bones
    pelvis_index = 0
    head_index = (
        common_body_names.index("Head") if "Head" in common_body_names else None
    )

    summary = dict(
        horizon_s=horizon_s, dt=dt, num_codes=num_codes, num_levels=num_levels,
        num_scalars=num_scalars, controls=controls,
        hold_threshold_m=hold_threshold, states=[],
    )
    rows = []

    for s, meta in enumerate(plans):
        sel = np.nonzero(state_of_env == s)[0]
        slots = slot_of_env[sel]
        code_envs = sel[slots < num_codes]
        code_slots = slots[slots < num_codes]
        samp_envs = sel[(slots >= num_codes) & (slots < num_codes + controls)]
        greedy_envs = sel[slots >= num_codes + controls]

        # code_envs is already in slot order by construction, but sort so the
        # montage index is the code index no matter how the batch was laid out.
        order = np.argsort(code_slots)
        code_envs, code_slots = code_envs[order], code_slots[order]

        err = pose_err[:, code_envs].T                    # [codes, steps]
        final = err[:, -1]
        worst = err.max(axis=1)
        held = worst <= hold_threshold
        depart = np.array([departure_step(e, hold_threshold) for e in err])
        depart_s = np.where(depart < 0, horizon_s, depart * dt)

        def block(envs):
            e = pose_err[:, envs].T
            return dict(
                n=int(len(envs)),
                held_rate=float((e.max(axis=1) <= hold_threshold).mean()),
                final_p50=float(np.median(e[:, -1])),
                depart_s_p50=float(np.median(
                    [horizon_s if departure_step(x, hold_threshold) < 0
                     else departure_step(x, hold_threshold) * dt for x in e]
                )),
            )

        gp, gr = goal_poses[s]
        ghost = normalize(gp[None], gr[None])[0]
        terminal = normalize(positions[-1, code_envs], root_rot[-1, code_envs])

        state = dict(
            **meta,
            codes=dict(
                n=int(num_codes),
                held_rate=float(held.mean()),
                held_count=int(held.sum()),
                final_pose_err_p10=float(np.percentile(final, 10)),
                final_pose_err_p50=float(np.median(final)),
                final_pose_err_p90=float(np.percentile(final, 90)),
                best_final_pose_err=float(final.min()),
                depart_s_p50=float(np.median(depart_s)),
                depart_s_p90=float(np.percentile(depart_s, 90)),
                terminated_rate=float(terminated[:, code_envs].any(axis=0).mean()),
                iou_final_p50=float(np.median(iou[-1, code_envs])),
            ),
            sampled_control=block(samp_envs),
            greedy_control=block(greedy_envs),
        )
        summary["states"].append(state)

        for i in range(num_codes):
            rows.append(dict(
                state=meta["plan"], code_index=i,
                code=" ".join(f"{int(v):+d}" for v in codes[i]),
                final_pose_err_m=round(float(final[i]), 4),
                worst_pose_err_m=round(float(worst[i]), 4),
                depart_s=round(float(depart_s[i]), 3),
                held=int(held[i]),
                final_iou=round(float(iou[-1, code_envs[i]]), 3),
                terminated=int(terminated[:, code_envs[i]].any()),
            ))

        tag = meta["plan"]
        title = (f"{tag}: all {num_codes} intent codes, terminal pose after "
                 f"{horizon_s:.1f} s   |   held {int(held.sum())}/{num_codes}"
                 f"   sampled {state['sampled_control']['held_rate']:.2f}"
                 f"   greedy {state['greedy_control']['held_rate']:.2f}")
        montage(out_dir / f"codes_montage_{tag}.png", terminal, ghost, final,
                num_levels, title, hold_threshold)
        heatmaps(out_dir / f"codes_heatmap_{tag}.png", final, depart_s, num_levels,
                 tag, hold_threshold, horizon_s)
        sparklines(out_dir / f"codes_sparklines_{tag}.png", err, num_levels, tag,
                   hold_threshold, dt)

        # Videos: the extremes of the enumeration plus the two live streams.
        ranked = np.argsort(final)
        picks = []
        for label, code_i in (
            ("best", ranked[0]), ("p25", ranked[len(ranked) // 4]),
            ("median", ranked[len(ranked) // 2]), ("worst", ranked[-1]),
        ):
            picks.append((f"{label}_code{int(code_i)}", int(code_envs[code_i]),
                          " ".join(f"{int(v):+d}" for v in codes[code_i])))
        picks.append(("ctrl_sampled", int(samp_envs[0]), "sampled stream"))
        picks.append(("ctrl_greedy", int(greedy_envs[0]), "greedy stream"))
        for label, env_id, code_str in picks[: max(videos, 0)]:
            titles = [
                f"{tag} {label} [{code_str}]  t={k*dt:5.2f}s  err={pose_err[k, env_id]:.3f}m"
                for k in range(steps)
            ]
            render_stick_video(
                positions[:, env_id], bones, titles,
                out_dir / f"video_{tag}_{label}.mp4",
                fps=int(round(1.0 / dt)), video_px=480,
                pelvis_index=pelvis_index, head_index=head_index,
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    with (out_dir / "per_code.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary
