# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured human ground pressure vs the policy's, side by side, bird's eye.

The 29 hard poses were trained with the MOYO pressure mat as weak supervision
(``notes/Pressure_supervision_design.MD``).  The aggregate that run reported --
"load on limbs the human left unloaded fell 41.7 N -> 3.7 N" -- says the policy
improved but not *where*, and this project has twice been saved by looking
rather than by aggregating.  This is the look: for every frame, the mat's own
reading next to the same square metre of floor as PhysX reports it.

Both panels are the same grid in the same frame (the reference clip's), so a
disagreement is a real disagreement:

* the measured field comes from the Tier-0 archive, whose mat rectangle is
  already stored in the clip frame;
* the simulated field comes from the rollout's ground contact points, mapped
  back with ``clip_xy = world_xy - respawn_offset_xy``.

Read the images, trust the numbers
----------------------------------
A rigid-body simulator has **no pressure** -- PhysX puts a whole limb's load on
1-6 solver points.  Both fields are therefore blurred by the same 2 cm Gaussian
before being drawn, purely so the pictures are comparable; every number printed
on the figure (totals, COP, zone shares, TV distance) is computed from the
unsmoothed data.  ``--sigma-cm 0`` turns the blur off and shows the raw spikes.

Gating is not optional
----------------------
The mat is 0.47 x 1.40 m and misses load on 57 of 170 clips.  The measured panel
is hatched, and the metrics suppressed, on frames where
``ground_reaction_valid`` says the measurement cannot be trusted; the simulated
panel reports what share of its own load falls inside the sensed rectangle, so
"the human has nothing there" can be told apart from "the mat could not see it".

Usage::

    # 1) record (GPU)
    python data/scripts/record_pressure_rollout.py \
      --checkpoint results/smpl_yogi_hard29_pressure_ab_s1/last.ckpt \
      --motion-file data/smpl/yoga_yogi_hard29_pressure.pt \
      --overrides env.ref_respawn_offset=0.005 \
      --out-dir results/hard29_pressure_rollouts

    # 2) compare (CPU only)
    PYTHONPATH=. python data/scripts/plot_pressure_compare.py \
      --in-dir results/hard29_pressure_rollouts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pressure_bodies import load_archive  # noqa: E402
from pressure_field import (  # noqa: E402
    CELL_AREA_CM2,
    MatGrid,
    field_total_n,
    frame_for_time,
    measured_field,
    on_mat_fraction,
    reference_hold,
    simulated_field,
    spread,
    weighted_cop,
)
from pressure_policy_report import (  # noqa: E402
    COMMON_BODY_NAMES,
    ZID,
    ZK,
    to_common_order,
    zone_shares,
)

# --------------------------------------------------------------------------- #
# house style (matches plot_contact_physics.py)
# --------------------------------------------------------------------------- #
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
MEASURED, LEARNED = "#2a78d6", "#eb6834"   # validated CVD pair, dE 24.7 protan

# One sequential ramp for BOTH heatmaps: pressure means the same thing in each
# panel, so it gets one encoding. Panel identity is carried by the frame colour
# and the COP marker instead. The hue is deliberately neither of the two
# identity hues, so a dark cell is never mistaken for a series colour.
SEQ_P = LinearSegmentedColormap.from_list(
    "pressure",
    ["#f4f5f3", "#e6e0f2", "#cdbfe6", "#ae98d6", "#8d70c4", "#6b4aab", "#4c3186",
     "#301c5c"],
)
# Difference: two hues, neutral midpoint, poles keyed to the two identities.
DIV_P = LinearSegmentedColormap.from_list(
    "measured_vs_learned",
    ["#12457f", "#2a78d6", "#a8c7ec", "#f0efec", "#f6bfa4", "#eb6834", "#9c3d15"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "xtick.color": INK_3, "ytick.color": INK_3,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "text.color": INK,
    "font.size": 8.0, "axes.titlesize": 9.0, "axes.titleweight": "semibold",
    "legend.frameon": False, "lines.linewidth": 1.6, "figure.dpi": 110,
})

FEET_UP_M = 0.25      # reference ankle height that defines the hold phase
GATE = 0.90           # ground_reaction_valid threshold, as the reward uses
ON_N = 5.0            # a body counts as grounded above this


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Comparison:
    """Everything one clip needs, already aligned onto the policy-step timeline."""

    clip: str
    checkpoint: str
    grid: MatGrid
    archive: dict
    t: np.ndarray                 # [S] motion time of each policy step
    gt_frame: np.ndarray          # [S] index into the measured clip
    sub_lo: np.ndarray            # [S] first substep of the step
    sub_hi: np.ndarray            # [S] one past the last substep
    cp_step: np.ndarray
    cp_body: np.ndarray
    cp_pos_clip: np.ndarray       # [N, 2] contact points in the CLIP frame
    cp_force: np.ndarray
    sim_fz: np.ndarray            # [S, 24] common order, N, per policy step
    gt_fz: np.ndarray             # [S, 24] common order, N, measured
    valid: np.ndarray             # [S, 3] the three gate columns at gt_frame
    hold: np.ndarray              # [S] reference is in the feet-up hold
    track_err: np.ndarray
    root_drift: np.ndarray
    bb_force: np.ndarray          # [S, 24, 24] |body-body force|, sim order
    sim_body_names: list
    body_mass_total: float
    offset_xy: np.ndarray         # clip -> world XY
    offset_source: str
    motion_length_s: float
    dt_ctrl: float
    subject_weight_n: float      # the human's, ~700 N -- NOT the robot's 725.9
    settle_steps: int            # steps burned before recording began
    hold_mode: str               # how 'the pose' was defined: feet_up | single_leg

    @property
    def n_steps(self) -> int:
        return len(self.t)


def _sim_points(cmp: Comparison, s: int):
    """Ground contact points of policy step ``s``, time-averaged over its substeps.

    Averaging rather than sampling one substep matters: contact solvers chatter,
    and a single 8.3 ms sample of a 33 ms window can miss a foot entirely.
    """
    lo, hi = int(cmp.sub_lo[s]), int(cmp.sub_hi[s])
    n = max(hi - lo, 1)
    m = (cmp.cp_step >= lo) & (cmp.cp_step < hi)
    return cmp.cp_pos_clip[m], cmp.cp_force[m] / n


def _ground_channels(d):
    """(ground_fz [Ts,B], bb_force [Ts,B,B], cp mask) for either rollout format.

    :mod:`record_pressure_rollout` stores the two reduced channels directly.
    A ``rollout.npz`` from :mod:`record_contact_physics` keeps the full
    ``[Ts, B, F, 3]`` pair tensor instead, and reading it costs nothing -- which
    is what lets this compare *older* policies (the 19-motion tracker, the
    crow-pair expert) against the same measured mat.
    """
    if "ground_fz" in d.files:
        return (d["ground_fz"].astype(np.float64),
                d["bb_force"].astype(np.float32),
                np.ones(d["cp_step"].shape, dtype=bool))
    pair = d["pair_force_w"]
    return (pair[:, :, 0, 2].astype(np.float64),
            np.linalg.norm(pair[:, :, 1:, :], axis=-1).astype(np.float32),
            d["cp_filter"] == 0)


def _registration_offset(d, ref_bp, t_first: float):
    """Clip-frame -> world XY offset, and how it was obtained.

    The recorded value is ``env.respawn_root_offset[:2]``, i.e. the offset the
    env applied at reset.  Older rollouts predate it and store only
    ``origin_xy``, the pelvis at the first recorded substep; since the robot is
    spawned exactly at ``reference + offset``, that difference *is* the offset --
    but only if the rollout really started at motion time 0, so require it.
    """
    if "respawn_offset_xy" in d.files:
        return np.asarray(d["respawn_offset_xy"], dtype=np.float64), "recorded"
    if "origin_xy" not in d.files:
        raise ValueError("rollout carries neither respawn_offset_xy nor origin_xy; "
                         "it cannot be put in the clip frame")
    if abs(t_first) > 1e-3:
        raise ValueError(
            f"legacy rollout starts at motion time {t_first:.3f}s, not 0, so "
            "origin_xy cannot be used to recover the spawn offset"
        )
    return (np.asarray(d["origin_xy"], dtype=np.float64)
            - ref_bp[0, 0, :2].astype(np.float64)), "derived from origin_xy"


def load_comparison(npz_path: Path, archive_dir: Path, motion_dir: Path,
                    pad_cells: int, hold_mode: str = "auto") -> Comparison:
    d = np.load(npz_path, allow_pickle=True)
    clip = str(d["motion_name"])

    archive_path = archive_dir / f"{clip}.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"no measured pressure archive at {archive_path}")
    archive = load_archive(archive_path)
    grid = MatGrid.from_archive(archive, pad_cells=pad_cells)

    import torch

    ref_path = motion_dir / f"{clip}.motion"
    if not ref_path.is_file():
        raise FileNotFoundError(f"no reference clip at {ref_path}")
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    if ref.get("ground_reaction_valid") is None:
        raise ValueError(f"{clip}: reference carries no measured pressure")
    gt_fz_all = np.maximum(ref["rigid_body_ground_forces"].numpy()[:, :, 2], 0.0)
    gv = ref["ground_reaction_valid"].numpy()
    ref_bp = ref["rigid_body_pos"].numpy()

    # The archive and the .motion must be the same clock. They are written by
    # different scripts from the same source, so check rather than assume: a
    # one-frame drift would silently shift every comparison by 16 ms.
    if len(gt_fz_all) != archive["n_frames"]:
        raise ValueError(
            f"{clip}: reference has {len(gt_fz_all)} frames but the pressure "
            f"archive has {archive['n_frames']}; the two are not the same clock"
        )

    sim_names = [str(x) for x in d["body_names"]]
    if sim_names == COMMON_BODY_NAMES:
        raise AssertionError(
            "rollout body_names already equal the COMMON order -- the remap has "
            "become a no-op. Verify before trusting output."
        )

    fz_sub_raw, bb_sub, cp_keep = _ground_channels(d)
    offset, offset_source = _registration_offset(
        d, ref_bp, float(d["ctrl_motion_time"][0])
    )

    t = d["ctrl_motion_time"].astype(float)
    sub_idx = d["ctrl_substep_index"].astype(int)
    n_sub = int(fz_sub_raw.shape[0])
    sub_hi = np.clip(sub_idx, 0, n_sub)
    sub_lo = np.concatenate([[0], sub_hi[:-1]])
    # Both recorders snapshot the control state once *before* the first step, so
    # entry 0 spans no substeps at all. Keeping it would open every animation on
    # an empty frame.
    if sub_hi[0] == 0 and len(sub_hi) > 1:
        keep = slice(1, None)
        t, sub_hi, sub_lo = t[keep], sub_hi[keep], sub_lo[keep]
        ctrl_track = d["ctrl_track_err"].astype(float)[keep]
        ctrl_drift = (d["ctrl_root_drift"].astype(float)[keep]
                      if "ctrl_root_drift" in d.files else np.zeros_like(t))
    else:
        ctrl_track = d["ctrl_track_err"].astype(float)
        ctrl_drift = (d["ctrl_root_drift"].astype(float)
                      if "ctrl_root_drift" in d.files else np.zeros_like(t))

    # Per-body ground force, averaged over each policy step's substeps, then put
    # in COMMON order. Reading it in simulator order once reported a policy
    # "resting on its spine" (Pressure_supervision_design.MD 4).
    fz_sub = np.maximum(fz_sub_raw, 0.0)
    cumulative = np.concatenate([np.zeros((1, fz_sub.shape[1])), fz_sub.cumsum(0)])
    span = np.maximum(sub_hi - sub_lo, 1)[:, None]
    fz_step = (cumulative[sub_hi] - cumulative[sub_lo]) / span
    sim_fz = to_common_order(fz_step, sim_names)

    bb_cum = np.concatenate([np.zeros((1, *bb_sub.shape[1:])),
                             bb_sub.astype(np.float64).cumsum(0)])
    bb_step = (bb_cum[sub_hi] - bb_cum[sub_lo]) / span[:, :, None]

    n_frames = int(archive["n_frames"])
    gt_frame = np.array([frame_for_time(x, n_frames) for x in t])

    cp_pos_clip = d["cp_pos"][cp_keep][:, :2].astype(np.float64) - offset[None, :]

    ankles = [COMMON_BODY_NAMES.index("L_Ankle"), COMMON_BODY_NAMES.index("R_Ankle")]
    hold, hold_mode_used = reference_hold(ref_bp[gt_frame][:, ankles, 2], hold_mode,
                                          FEET_UP_M)

    return Comparison(
        clip=clip,
        checkpoint=str(d["checkpoint"]),
        grid=grid,
        archive=archive,
        t=t,
        gt_frame=gt_frame,
        sub_lo=sub_lo,
        sub_hi=sub_hi,
        cp_step=d["cp_step"][cp_keep].astype(int),
        cp_body=d["cp_body"][cp_keep].astype(int),
        cp_pos_clip=cp_pos_clip,
        cp_force=d["cp_force"][cp_keep].astype(np.float64),
        sim_fz=sim_fz,
        gt_fz=gt_fz_all[gt_frame],
        valid=gv[gt_frame],
        hold=hold,
        track_err=ctrl_track,
        root_drift=ctrl_drift,
        bb_force=bb_step,
        sim_body_names=sim_names,
        body_mass_total=float(np.sum(d["body_masses"])) if "body_masses" in d.files
        else 74.0,
        offset_xy=offset,
        offset_source=offset_source,
        motion_length_s=float(d["motion_length_s"]),
        dt_ctrl=float(d["dt_ctrl"]),
        subject_weight_n=float(archive["subject_weight_n"]),
        settle_steps=int(d["settle_steps"]) if "settle_steps" in d.files else 0,
        hold_mode=hold_mode_used,
    )


# --------------------------------------------------------------------------- #
# per-frame quantities
# --------------------------------------------------------------------------- #
MIN_TOTAL_N = 30.0    # below this a "share" is noise, so leave the field at zero


def _renormalise(field: np.ndarray, bw: float) -> np.ndarray:
    """Rescale an N/cm2 field so it integrates to exactly one body weight.

    **The two sides cannot be compared in absolute newtons.**  The mat reads
    0.74-0.83 BW under forearm-supported holds and 0.93-0.97 under
    hand-supported ones (``Pressure_supervision_design.MD`` 5.1) while the
    simulator integrates to 1.000 BW by construction, so a raw difference would
    paint the policy as overloading everything it touches -- an artefact of the
    measurement gain, not a finding.  The same note establishes that whatever
    causes the deficit is *proportional across zones*, which is exactly the
    invariance this rescale needs.  Absolute totals are printed as text instead
    of being encoded in the colour.
    """
    total = float(field.sum()) * CELL_AREA_CM2
    if total < MIN_TOTAL_N:
        return np.zeros_like(field)
    return field * (bw / total)


def frame_fields(cmp: Comparison, s: int, sigma_m: float, bw: Optional[float] = None):
    """(measured, simulated) fields for policy step ``s``, drawn-ready and raw.

    The first two are blurred and -- when ``bw`` is given -- renormalised to one
    body weight, i.e. what goes on screen.  The last two are untouched N/cm2 and
    are what every reported number comes from.
    """
    gt_raw = measured_field(cmp.archive, cmp.gt_frame[s], cmp.grid)
    pts, forces = _sim_points(cmp, s)
    sim_raw = simulated_field(pts, forces, cmp.grid)
    gt_draw = spread(gt_raw, sigma_m, cmp.grid.cell)
    sim_draw = spread(sim_raw, sigma_m, cmp.grid.cell)
    if bw is not None:
        gt_draw = _renormalise(gt_draw, bw)
        sim_draw = _renormalise(sim_draw, bw)
    return gt_draw, sim_draw, gt_raw, sim_raw


def series(cmp: Comparison) -> dict:
    """Time series computed from unsmoothed data, for the plots and the JSON.

    Two body weights, deliberately. The humanoid is built at 74 kg and the
    captured subject weighs ~71 kg (``Moyo_pressure_port.MD`` 4.1), so dividing
    the mat's newtons by the robot's weight would fold a 3.7 % mass mismatch
    into the mat's own gain deficit and make the measurement look worse than it
    is. ``bw`` scales the simulated side and the shared colour scale; ``bw_ref``
    scales anything measured.
    """
    bw = cmp.body_mass_total * 9.81
    bw_ref = cmp.subject_weight_n
    gt_total = cmp.archive["total_force_n"][cmp.gt_frame].astype(float)
    sim_total = cmp.sim_fz.sum(1)

    s_sim, tot_sim = zone_shares(cmp.sim_fz)
    s_gt, tot_gt = zone_shares(cmp.gt_fz)
    tv = 0.5 * np.abs(s_sim - s_gt).sum(-1)

    # One gate per channel, because they fail independently. The mat's own
    # reading -- the picture on the left, its total and its COP -- is gated by
    # COVERAGE alone; the attribution of that reading to bodies, which is what
    # the zone bars and the TV distance are made of, is gated by column 2. Using
    # the attribution gate to hide the heatmap would blank a perfectly good
    # measurement on every frame where only the *bookkeeping* is uncertain.
    cov_gate = cmp.valid[:, 0] >= GATE
    share_gate = cmp.valid[:, 2] >= GATE if cmp.valid.shape[1] >= 3 else \
        cmp.valid[:, 1] >= GATE * GATE
    body_gate = cmp.valid[:, 1] >= GATE * GATE
    live = share_gate & (tot_sim >= 30.0) & (tot_gt >= 30.0)

    feet_sim = s_sim[:, ZK.index("FEET")]
    feet_gt = s_gt[:, ZK.index("FEET")]

    hands = cmp.sim_fz[:, ZID["HANDS"]].sum(1)
    hands_down = hands > ON_N
    nonhand = cmp.sim_fz.copy()
    nonhand[:, ZID["HANDS"]] = 0.0

    # Option B replayed per clip: load the policy puts on support zones the
    # measurement says carry nothing. This is `condN`, the quantity the run in
    # Pressure_supervision_design.MD 8.2 drove from 41.7 N to 3.7 N -- and unlike
    # "non-hand load" it is pose-agnostic, which matters here because the
    # forearm-supported poses (pincha, scorpion) are not held up by their hands.
    zone_sim = np.stack([cmp.sim_fz[..., ZID[k]].sum(-1) for k in ZK], -1)
    violation = ((s_gt < 0.02) * zone_sim).sum(-1)

    on_mat = np.full(cmp.n_steps, np.nan)
    captured = np.full(cmp.n_steps, np.nan)
    for s in range(cmp.n_steps):
        pts, forces = _sim_points(cmp, s)
        if len(pts):
            field = simulated_field(pts, forces, cmp.grid)
            on_mat[s] = on_mat_fraction(field, cmp.grid)
            # Load that lands outside the *padded* grid is dropped by the
            # rasteriser, and the per-frame renormalisation would then scale
            # whatever is left back up to a full body weight -- a policy that
            # drifted a metre away would be drawn as a perfectly loaded one.
            # This is the number that catches it.
            if sim_total[s] > MIN_TOTAL_N:
                captured[s] = field_total_n(field) / sim_total[s]

    # Which bodies the off-mat load belongs to. "30 % of the load is off the mat"
    # is a measurement caveat; "and it is both hands" is a finding -- the crow
    # policy plants its hands wider than the human and overruns a mat the human
    # fitted inside.
    rc = cmp.grid.to_cell(cmp.cp_pos_clip)
    r0, c0, ny, nx = cmp.grid.sensed
    off = ~((rc[:, 0] >= r0) & (rc[:, 0] < r0 + ny)
            & (rc[:, 1] >= c0) & (rc[:, 1] < c0 + nx))
    off_by_body = {}
    if cmp.cp_force.sum() > 0 and off.any():
        total = cmp.cp_force.sum()
        for bi in np.unique(cmp.cp_body[off]):
            share = float(cmp.cp_force[off & (cmp.cp_body == bi)].sum() / total)
            if share >= 0.005:
                off_by_body[cmp.sim_body_names[bi]] = round(share, 4)

    return dict(
        bw=bw, bw_ref=bw_ref,
        gt_total=gt_total, sim_total=sim_total,
        gt_share=s_gt, sim_share=s_sim, tv=tv,
        cov_gate=cov_gate, share_gate=share_gate, body_gate=body_gate, live=live,
        feet_sim=feet_sim, feet_gt=feet_gt,
        hands_down=hands_down, nonhand=nonhand.sum(1), violation=violation,
        on_mat=on_mat, captured=captured, off_mat_by_body=off_by_body,
    )


def view_box(cmp: Comparison, margin: float, steps=None, samples: int = 120,
             keep: float = 0.995):
    """Clip-frame box worth drawing: where the load actually is.

    The mat is 0.47 x 1.40 m and most poses use a fraction of it, so drawing the
    whole rectangle would make a crow's two hands four pixels wide.  The bound is
    load-weighted rather than a plain extent: a single grazing cell during the
    walk-on would otherwise stretch the view over half a metre of empty floor.
    """
    centres = cmp.grid.cell_centres()
    pool = np.arange(cmp.n_steps) if steps is None else np.asarray(steps)
    if len(pool) == 0:
        pool = np.arange(cmp.n_steps)
    xs, ys, ws = [], [], []
    for s in np.unique(pool[np.linspace(0, len(pool) - 1, samples).astype(int)]):
        gt_raw = measured_field(cmp.archive, cmp.gt_frame[s], cmp.grid)
        pts, forces = _sim_points(cmp, s)
        sim_raw = simulated_field(pts, forces, cmp.grid)
        field = gt_raw + sim_raw
        mask = field > 0.01
        if not mask.any():
            continue
        xs.append(centres[..., 0][mask])
        ys.append(centres[..., 1][mask])
        ws.append(field[mask])
    if not xs:
        corners = cmp.grid.sensed_polygon()
        return corners.min(0) - margin, corners.max(0) + margin
    xs, ys, ws = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    lo, hi = [], []
    tail = 0.5 * (1.0 - keep)
    for coord in (xs, ys):
        order = np.argsort(coord)
        cum = np.cumsum(ws[order]) / ws.sum()
        lo.append(coord[order][np.searchsorted(cum, tail)])
        hi.append(coord[order][min(np.searchsorted(cum, 1.0 - tail),
                                   len(coord) - 1)])
    return np.array(lo) - margin, np.array(hi) + margin


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def _draw_heat(ax, field, corners, vmax, box, accent, title):
    ax.clear()
    style(ax)
    mesh = ax.pcolormesh(
        corners[..., 0], corners[..., 1], field,
        cmap=SEQ_P, vmin=0.0, vmax=vmax, shading="flat", rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xlim(box[0][0], box[1][0])
    ax.set_ylim(box[0][1], box[1][1])
    ax.set_title(title, loc="left", color=accent)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(accent)
        ax.spines[spine].set_linewidth(1.6)
    ax.grid(False)
    return mesh


def _draw_mat_outline(ax, grid: MatGrid, label: bool = False):
    """Outline the area the mat can actually see.

    Load outside it is not "the human had none there" -- it is unmeasured, and
    the distinction is the difference between a finding and an artefact.
    """
    poly = grid.sensed_polygon()
    ax.plot(poly[:, 0], poly[:, 1], color=INK_3, lw=1.0, ls="--", zorder=4)
    if label:
        # In axes coordinates: the view is cropped to the held pose, so the mat
        # corner this used to anchor to is usually outside the drawn box and the
        # label silently never appeared.
        ax.text(0.985, 0.012, "- - -  sensed mat", transform=ax.transAxes,
                fontsize=6.5, color=INK_3, ha="right", va="bottom", zorder=4)


def render_frame(cmp, ser, axes, artists, s, sigma_m, vmax, box, norm_bw):
    """Draw one policy step across every panel. Returns the changed artists."""
    ax_gt, ax_sim, ax_bar, ax_force, ax_feet = axes
    grid = cmp.grid
    corners = grid.cell_corners()
    gt_f, sim_f, gt_raw, sim_raw = frame_fields(cmp, s, sigma_m, norm_bw)

    gate_ok = bool(ser["share_gate"][s])       # attribution: bars, TV
    field_ok = bool(ser["cov_gate"][s])        # the mat reading itself
    gt_n = field_total_n(gt_raw)
    sim_n = float(cmp.sim_fz[s].sum())

    _draw_heat(ax_gt, gt_f, corners, vmax, box, MEASURED,
               f"MEASURED human   {gt_n:5.0f} N  ({gt_n / ser['bw_ref']:.2f} BW)")
    _draw_heat(ax_sim, sim_f, corners, vmax, box, LEARNED,
               f"LEARNED policy   {sim_n:5.0f} N  ({sim_n / ser['bw']:.2f} BW)")
    for ax in (ax_gt, ax_sim):
        _draw_mat_outline(ax, grid, label=True)
        ax.set_xlabel("x (m, clip frame)")
    ax_gt.set_ylabel("y (m, clip frame)")

    # The measured panel is only evidence where the gate says so.
    if not field_ok:
        lo, hi = box
        ax_gt.add_patch(Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            facecolor="none", edgecolor=INK_3, hatch="////", lw=0.0, alpha=0.55,
            zorder=5,
        ))
        ax_gt.text(0.5, 0.5, "mat coverage too low\nto read this frame",
                   transform=ax_gt.transAxes,
                   ha="center", va="center", fontsize=9, color=INK_2, zorder=6,
                   bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.3"))

    cop_gt = cmp.archive["cop_world"][cmp.gt_frame[s]]
    if np.isfinite(cop_gt).all() and gt_n > 1.0:
        ax_gt.plot(cop_gt[0], cop_gt[1], "o", ms=7, color=MEASURED,
                   mec=SURFACE, mew=1.5, zorder=6)
        ax_gt.annotate("COP", (cop_gt[0], cop_gt[1]), textcoords="offset points",
                       xytext=(7, 4), fontsize=7, color=INK_2, zorder=6)
    pts, forces = _sim_points(cmp, s)
    cop_sim = weighted_cop(pts, forces)
    if cop_sim is not None:
        ax_sim.plot(cop_sim[0], cop_sim[1], "o", ms=7, color=LEARNED,
                    mec=SURFACE, mew=1.5, zorder=6)
        ax_sim.annotate("COP", (cop_sim[0], cop_sim[1]), textcoords="offset points",
                        xytext=(7, 4), fontsize=7, color=INK_2, zorder=6)

    # Name what is on the floor -- the whole crow question is *which* body.
    loaded = np.argsort(-cmp.sim_fz[s])[:3]
    for bi in loaded:
        if cmp.sim_fz[s, bi] <= ON_N:
            continue
        name = COMMON_BODY_NAMES[bi]
        sim_index = cmp.sim_body_names.index(name)
        m = ((cmp.cp_step >= cmp.sub_lo[s]) & (cmp.cp_step < cmp.sub_hi[s])
             & (cmp.cp_body == sim_index))
        if not m.any():
            continue
        centroid = weighted_cop(cmp.cp_pos_clip[m], cmp.cp_force[m])
        if centroid is None:
            continue
        ax_sim.annotate(
            f"{name} {cmp.sim_fz[s, bi] / ser['bw']:.2f}BW",
            (centroid[0], centroid[1]), textcoords="offset points", xytext=(6, -10),
            fontsize=6.5, color=INK_2, zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.7, pad=1.0),
        )
    # Body-body contact, which no pressure mat can see and which is half of what
    # an arm balance *is*: crow is held by the knees on the upper arms. A policy
    # that produces the right footprint with no self-contact is not doing the
    # pose, and only this line says so.
    bb = cmp.bb_force[s]
    pairs = np.dstack(np.unravel_index(np.argsort(-bb, axis=None), bb.shape))[0]
    seen, lines = set(), []
    for a, b in pairs:
        if a >= b or bb[a, b] <= ON_N or len(lines) >= 3:
            break
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{cmp.sim_body_names[a]}<->{cmp.sim_body_names[b]} "
                     f"{bb[a, b]:.0f} N")
    ax_sim.text(
        0.02, 0.985, "body-body: " + (", ".join(lines) if lines else "none"),
        transform=ax_sim.transAxes, fontsize=6.5, color=INK_2, zorder=6, va="top",
        bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.25"),
    )

    notes = []
    if np.isfinite(ser["on_mat"][s]) and ser["on_mat"][s] < 0.98:
        notes.append(f"{100 * (1 - ser['on_mat'][s]):.0f}% of this load is off the mat")
    if np.isfinite(ser["captured"][s]) and ser["captured"][s] < 0.98:
        notes.append(f"{100 * (1 - ser['captured'][s]):.0f}% is outside the drawn grid")
    if notes:
        ax_sim.text(
            0.02, 0.02, "\n".join(notes), transform=ax_sim.transAxes, fontsize=6.5,
            color=INK_2, zorder=6, va="bottom",
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.25"),
        )

    # --- zone shares ------------------------------------------------------- #
    ax_bar.clear()
    style(ax_bar)
    x = np.arange(len(ZK))
    # Hatched, not hidden: on a gated-off frame the shares still exist, they
    # just are not evidence, and a bar drawn solid next to the policy's invites
    # exactly the comparison the gate says cannot be made.
    ax_bar.bar(x - 0.19, ser["gt_share"][s], 0.34,
               color=MEASURED if gate_ok else "none",
               edgecolor=MEASURED, hatch=None if gate_ok else "///",
               linewidth=0.0 if gate_ok else 0.8,
               label="measured" if gate_ok else "measured (not trusted here)")
    ax_bar.bar(x + 0.19, ser["sim_share"][s], 0.34, color=LEARNED, label="learned")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ZK, fontsize=7)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("share of ground load")
    ax_bar.legend(loc="upper right", fontsize=7, ncol=2)
    tv_txt = f"TV {ser['tv'][s]:.3f}" if gate_ok else "TV n/a (gated)"
    ax_bar.set_title(f"Where the load goes -- {tv_txt}", loc="left")

    # --- time series (one measure per axis; never a second y-scale) -------- #
    for ax, gt_series, sim_series, ylabel, title in (
        (ax_force, ser["gt_total"] / ser["bw_ref"], ser["sim_total"] / ser["bw"],
         "total ground force (BW)", "Total load"),
        (ax_feet, ser["feet_gt"], ser["feet_sim"],
         "FEET share", "Load through the feet -- the trailing-foot question"),
    ):
        ax.clear()
        style(ax)
        gated = ~ser["share_gate"]
        ax.fill_between(cmp.t, 0, 1, where=gated, transform=ax.get_xaxis_transform(),
                        color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.plot(cmp.t, gt_series, color=MEASURED, label="measured")
        ax.plot(cmp.t, sim_series, color=LEARNED, label="learned")
        ax.axvline(cmp.t[s], color=INK, lw=1.0, alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax_feet.set_xlabel("motion time (s)")
    return artists


def build_figure(box, height: float = 7.6):
    """Size the heat columns from the data's own aspect ratio.

    ``set_aspect('equal')`` is non-negotiable for a floor plan, so fixed column
    widths mean matplotlib shrinks the panel to fit and leaves half the figure
    blank -- which is exactly what makes two hand-prints unreadable.  Sizing the
    columns from ``dx/dy`` instead keeps the heatmaps as large as the figure
    allows for every pose, tall (headstand) or wide (side crow).
    """
    (x0, y0), (x1, y1) = box
    aspect = float(np.clip((x1 - x0) / max(y1 - y0, 1e-6), 0.18, 3.0))
    heat_h = height * 0.80
    heat_w = float(np.clip(heat_h * aspect, 1.7, 6.0))
    right_w, pad_w = 6.6, 1.5
    fig = plt.figure(figsize=(2 * heat_w + right_w + pad_w, height))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[heat_w, heat_w, right_w],
        height_ratios=[1.05, 1.0, 1.0],
        wspace=0.30, hspace=0.55, left=0.045, right=0.985, top=0.90, bottom=0.075,
    )
    ax_gt = fig.add_subplot(gs[:, 0])
    ax_sim = fig.add_subplot(gs[:, 1], sharex=ax_gt, sharey=ax_gt)
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_force = fig.add_subplot(gs[1, 2])
    ax_feet = fig.add_subplot(gs[2, 2], sharex=ax_force)
    return fig, (ax_gt, ax_sim, ax_bar, ax_force, ax_feet)


def colour_scale(cmp: Comparison, sigma_m: float, norm_bw, samples: int = 90) -> float:
    """Shared vmax across both panels and the whole clip.

    Shared and fixed, because a per-frame or per-panel scale would let the two
    panels look identical while carrying an order of magnitude difference in
    load -- the one thing this figure exists to show.

    The p99 of every loaded *cell* pooled over the clip, not a statistic of
    per-frame peaks.  Frames where PhysX collapses the whole body weight onto a
    single solver point are common -- flight, impact, a toe grazing -- and after
    renormalisation one such cell reads ~24 N/cm2 against a mat whose realistic
    peak is 5-8.  Taking any percentile *of per-frame peaks* is dominated by
    those frames and washes the held pose out to nothing.  Pooling cells lets a
    one-cell frame contribute one cell, which is what it is.
    """
    cells = []
    for s in np.unique(np.linspace(0, cmp.n_steps - 1, samples).astype(int)):
        gt_f, sim_f, _, _ = frame_fields(cmp, s, sigma_m, norm_bw)
        for f in (gt_f, sim_f):
            cells.append(f[f > 0.02])
    pooled = np.concatenate(cells) if cells else np.zeros(0)
    return float(np.percentile(pooled, 99)) if pooled.size else 1.0


def animate(cmp: Comparison, ser: dict, out_path: Path, sigma_m: float,
            stride: int, fps: int, vmax: float, box, norm_bw) -> Path:
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    fig, axes = build_figure(box)
    frames = list(range(0, cmp.n_steps, max(stride, 1)))

    def draw(s):
        render_frame(cmp, ser, axes, None, s, sigma_m, vmax, box, norm_bw)
        fig.suptitle(
            f"{cmp.clip}    t = {cmp.t[s]:6.2f} s / {cmp.t[-1]:.1f} s"
            f"     tracking err {cmp.track_err[s]:.3f} m"
            + (f", root drift {cmp.root_drift[s] * 100:.1f} cm"
               if np.any(cmp.root_drift) else ""),
            fontsize=11, fontweight="semibold", color=INK, x=0.02, ha="left",
        )
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / max(fps, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".gif":
        anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    return out_path


def static_summary(cmp: Comparison, ser: dict, out_path: Path, sigma_m: float,
                   vmax: float, box, norm_bw) -> Path:
    """Mean fields over the frames that matter, plus their difference.

    The animation shows what happened; this shows what it adds up to, and the
    difference panel is the one that names the defect -- an orange blob under a
    trailing toe is the 34 % BW that ``Pressure_supervision_design.MD`` 4 found
    by arithmetic.
    """
    sel = np.where(cmp.hold & ser["share_gate"])[0]
    scope = "hold frames, gate open"
    if len(sel) < 5:
        sel = np.where(ser["share_gate"])[0]
        scope = "all gated frames"
    if len(sel) < 5:
        sel = np.arange(cmp.n_steps)
        scope = "whole clip (NO usable gate)"
    take = np.unique(np.linspace(0, len(sel) - 1, min(len(sel), 200)).astype(int))
    sel = sel[take]

    gt_mean = np.zeros(cmp.grid.shape)
    sim_mean = np.zeros(cmp.grid.shape)
    for s in sel:
        gt_f, sim_f, _, _ = frame_fields(cmp, s, sigma_m, norm_bw)
        gt_mean += gt_f
        sim_mean += sim_f
    gt_mean /= len(sel)
    sim_mean /= len(sel)

    # The mean of many frames peaks lower than any single frame, so reusing the
    # animation's scale would wash the summary out. Same rule, own numbers.
    nz = np.concatenate([gt_mean[gt_mean > 0], sim_mean[sim_mean > 0]])
    vmax = float(np.percentile(nz, 99.5)) if nz.size else vmax

    box = view_box(cmp, 0.10, steps=sel)
    (x0, y0), (x1, y1) = box
    aspect = float(np.clip((x1 - x0) / max(y1 - y0, 1e-6), 0.2, 3.0))
    panel_h = 6.4
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * float(np.clip(panel_h * 0.78 * aspect, 2.0, 6.0)) + 1.6,
                       panel_h))
    corners = cmp.grid.cell_corners()
    m0 = _draw_heat(axes[0], gt_mean, corners, vmax, box, MEASURED, "MEASURED human")
    _draw_heat(axes[1], sim_mean, corners, vmax, box, LEARNED, "LEARNED policy")
    diff = sim_mean - gt_mean
    lim = float(np.abs(diff).max()) or 1.0
    axes[2].clear()
    style(axes[2])
    m2 = axes[2].pcolormesh(corners[..., 0], corners[..., 1], diff, cmap=DIV_P,
                           vmin=-lim, vmax=lim, shading="flat", rasterized=True)
    axes[2].set_aspect("equal")
    axes[2].set_xlim(box[0][0], box[1][0])
    axes[2].set_ylim(box[0][1], box[1][1])
    axes[2].grid(False)
    axes[2].set_title("LEARNED minus MEASURED", loc="left")
    for ax in axes:
        _draw_mat_outline(ax, cmp.grid)
        ax.set_xlabel("x (m, clip frame)")
    axes[0].set_ylabel("y (m, clip frame)")

    units = "N/cm2 at 1 BW (each side renormalised)" if norm_bw else "N/cm2"
    fig.colorbar(m0, ax=axes[:2], shrink=0.55, label=f"pressure, {units}",
                 location="bottom", pad=0.09)
    fig.colorbar(m2, ax=axes[2], shrink=0.55, label="orange = policy loads it more",
                 location="bottom", pad=0.09)
    live = ser["live"]
    tv = float(ser["tv"][live].mean()) if live.any() else float("nan")
    fig.suptitle(
        f"{cmp.clip} -- mean ground pressure over {len(sel)} {scope}"
        f"   |   zone-share TV {tv:.3f}",
        fontsize=11.5, fontweight="semibold", color=INK, x=0.02, ha="left",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_weighted(ratio: np.ndarray, weight: np.ndarray) -> Optional[float]:
    """Mean of a per-frame ratio, weighted by how much load that frame carried.

    An unweighted mean lets a frame with one newton on a toe count as much as a
    frame with a whole body weight on two hands, which for "how much of the
    policy's load is on the mat" is exactly backwards.
    """
    m = np.isfinite(ratio) & (weight > 0)
    if not m.any():
        return None
    return float(np.average(ratio[m], weights=weight[m]))


def summary_json(cmp: Comparison, ser: dict) -> dict:
    live, hold = ser["live"], cmp.hold
    hl = live & hold
    bw = ser["bw"]

    def stat(mask, key, arr):
        return {key: float(arr[mask].mean())} if mask.any() else {}

    top_bb = []
    bb = cmp.bb_force
    if bb.size:
        peak = bb.max(0)
        duty = (bb > ON_N).mean(0)
        idx = np.dstack(np.unravel_index(np.argsort(-peak, axis=None), peak.shape))[0]
        seen = set()
        for a, b in idx:
            if a >= b or peak[a, b] <= ON_N:
                continue
            key = (cmp.sim_body_names[a], cmp.sim_body_names[b])
            if key in seen:
                continue
            seen.add(key)
            top_bb.append({"pair": f"{key[0]} <-> {key[1]}",
                           "peak_n": float(peak[a, b]),
                           "duty_pct": float(100 * duty[a, b])})
            if len(top_bb) >= 6:
                break

    # The registration is asserted at reset by the recorder (pelvis residual);
    # this is the independent downstream check. It cannot separate "the frames
    # are misaligned" from "the policy genuinely stands somewhere else", so it
    # is reported rather than asserted -- but a value far above the ~3 cm seen
    # on a well-registered rollout means one of the two, and both matter.
    cop_gap = []
    gr = cmp.archive["cop_world"]
    for s in np.where(ser["share_gate"] & (ser["gt_total"] > 1.0))[0]:
        pts, forces = _sim_points(cmp, s)
        sim_cop = weighted_cop(pts, forces)
        gt_cop = gr[cmp.gt_frame[s]]
        if sim_cop is not None and np.isfinite(gt_cop).all():
            cop_gap.append(float(np.linalg.norm(sim_cop - gt_cop)))

    out = {
        "clip": cmp.clip,
        "checkpoint": cmp.checkpoint,
        "n_policy_steps": cmp.n_steps,
        "clip_seconds": float(cmp.t[-1]),
        "clip_coverage_pct": 100 * float(cmp.t[-1] / max(cmp.motion_length_s, 1e-6)),
        "registration": f"clip_xy = world_xy - offset ({cmp.offset_source})",
        "registration_offset_xy": [float(x) for x in cmp.offset_xy],
        "cop_gap_median_m": float(np.median(cop_gap)) if cop_gap else None,
        "cop_gap_p90_m": float(np.percentile(cop_gap, 90)) if cop_gap else None,
        "gate_policy": {
            "shares": "ground_reaction_valid[:,2] >= 0.90",
            "absolute_per_body": "ground_reaction_valid[:,1] >= 0.81",
        },
        "coverage_gate_open_pct": 100 * float(ser["cov_gate"].mean()),
        "share_gate_open_pct": 100 * float(ser["share_gate"].mean()),
        "ref_hold_pct": 100 * float(hold.mean()),
        "hold_mode": cmp.hold_mode,
        "hold_definition": (
            "REFERENCE frames with BOTH ankles above 0.25 m -- the human is in "
            "the pose. Says nothing about whether the policy got there; see "
            "sim_feet_up_in_ref_hold_pct."
            if cmp.hold_mode == "feet_up" else
            "REFERENCE frames with EXACTLY ONE ankle above 0.25 m. A standing "
            "balance never lifts both feet, so the feet-up definition selects 0 "
            "% of these clips; the matching policy-side metric is "
            "sim_one_foot_loaded_in_ref_hold_pct, not sim_feet_up_*."),
        "mean_track_err_m": float(cmp.track_err.mean()),
        "mean_root_drift_m": (float(cmp.root_drift.mean())
                              if np.any(cmp.root_drift) else None),
        "sim_load_on_mat_frac": _load_weighted(ser["on_mat"], ser["sim_total"]),
        "sim_load_captured_by_grid_frac": float(np.nanmin(ser["captured"]))
        if np.isfinite(ser["captured"]).any() else None,
        "off_mat_load_by_body": ser["off_mat_by_body"],
        # Only over frames the mat could actually read -- averaging the gated-off
        # ones in would blend a measurement with a non-measurement.
        "measured_total_bw": (float(ser["gt_total"][ser["cov_gate"]].mean()
                                    / ser["bw_ref"])
                              if ser["cov_gate"].any() else None),
        "measured_total_bw_scope": "frames with mat coverage >= 0.90",
        "subject_weight_n": cmp.subject_weight_n,
        "robot_weight_n": bw,
        "sim_total_bw": float(np.mean(ser["sim_total"]) / bw),
        "body_body_pairs": top_bb,
    }
    if live.any():
        out["zone_share_tv"] = float(ser["tv"][live].mean())
        out["per_zone_abs_diff"] = {
            k: float(v) for k, v in
            zip(ZK, np.abs(ser["sim_share"][live] - ser["gt_share"][live]).mean(0))
        }
    if hold.any():
        # The one thing a reference-defined hold cannot tell you: did the policy
        # do the same thing with its own feet while the human was in the pose?
        # Which question that is depends on the pose -- "are both feet clear" is
        # meaningless for a standing balance, where the answer is always no and
        # the real question is whether it is standing on ONE foot.
        l_n = cmp.sim_fz[:, [COMMON_BODY_NAMES.index("L_Ankle"),
                             COMMON_BODY_NAMES.index("L_Toe")]].sum(1)
        r_n = cmp.sim_fz[:, [COMMON_BODY_NAMES.index("R_Ankle"),
                             COMMON_BODY_NAMES.index("R_Toe")]].sum(1)
        if cmp.hold_mode == "single_leg":
            one = ((l_n > ON_N) ^ (r_n > ON_N))
            out["sim_one_foot_loaded_in_ref_hold_pct"] = 100 * float(one[hold].mean())
            out["sim_both_feet_loaded_in_ref_hold_pct"] = 100 * float(
                ((l_n > ON_N) & (r_n > ON_N))[hold].mean())
        else:
            feet_n = cmp.sim_fz[:, ZID["FEET"]].sum(1)
            out["sim_feet_up_in_ref_hold_pct"] = 100 * float(
                (feet_n[hold] <= ON_N).mean())
    if hl.any():
        out["hold_zone_share_tv"] = float(ser["tv"][hl].mean())
        out["hold_feet_share_sim"] = float(ser["feet_sim"][hl].mean())
        out["hold_feet_share_measured"] = float(ser["feet_gt"][hl].mean())
    hd = ser["hands_down"]
    if hd.any():
        out["hands_down_pct"] = 100 * float(hd.mean())
        # Two scopes. The whole-clip figure is dominated by the entry and exit,
        # where the feet are legitimately on the floor while the hands are
        # already down, so it reads several times the in-pose value. It is kept
        # because it is the definition Pressure_supervision_design.MD 4
        # tabulates, and comparing against that table needs the same definition.
        out["nonhand_load_while_hands_down_bw"] = float(ser["nonhand"][hd].mean() / bw)
        out["nonhand_load_p90_bw"] = float(np.percentile(ser["nonhand"][hd], 90) / bw)
        out["nonhand_load_scope"] = ("whole clip; matches "
                                     "Pressure_supervision_design.MD 4")
        hh = hd & hold
        if hh.any():
            out["nonhand_load_in_ref_hold_bw"] = float(ser["nonhand"][hh].mean() / bw)

    # The training objective's own metric, per clip. Reported in newtons as well
    # as body weights so it can be read straight against the 41.7 N -> 3.7 N of
    # Pressure_supervision_design.MD 8.2.
    m = ser["body_gate"] & (ser["gt_total"] > MIN_TOTAL_N)
    if m.sum() >= 20:
        v = ser["violation"][m]
        # Median *and* p90 and mean, because this distribution is bimodal --
        # the violation is exactly zero whenever the offending limb happens to
        # be clear, so on crow the median reads 0.0 N while the mean is 48 N and
        # the p90 is far higher. Pressure_supervision_design.MD 8.7 makes the
        # same point about reading condN.
        out["unloaded_zone_load_median_n"] = float(np.median(v))
        out["unloaded_zone_load_mean_n"] = float(v.mean())
        out["unloaded_zone_load_p90_n"] = float(np.percentile(v, 90))
        out["unloaded_zone_load_bw"] = float(v.mean() / bw)
        out["unloaded_zone_gate_pct"] = 100 * float(m.mean())
    return out


def composite_with_video(anim_path: Path, video_path: Path, out_path: Path,
                         lead_seconds: float = 0.0) -> Path:
    """Stack the rendered policy video beside the heatmaps, aligned in time.

    Both are one frame per policy step at 30 fps, so they run at the same rate --
    but they do not *start* at the same motion time. ``render_policy_videos.py``
    burns ``--settle-steps`` (2 by default) before it starts capturing, so its
    first frame is at ``(settle + 1) * dt`` while the heatmap's is at ``dt``. Left
    uncorrected the video runs two frames ahead, which on a lift-off shows the
    trailing toe already clear while the heatmap still paints load under it --
    precisely the wrong reading of precisely the moment the figure is for.
    ``lead_seconds`` trims that difference off the front of the heatmap.
    """
    from moviepy import VideoFileClip, clips_array

    heat = VideoFileClip(str(anim_path))
    vid = VideoFileClip(str(video_path))
    if lead_seconds > 1e-6:
        heat = heat.subclipped(min(lead_seconds, heat.duration * 0.5), heat.duration)
    duration = min(heat.duration, vid.duration)
    vid = vid.subclipped(0, duration).resized(height=heat.h)
    stacked = clips_array([[vid, heat.subclipped(0, duration)]])
    stacked.write_videofile(str(out_path), codec="libx264", audio=False,
                           preset="veryfast", logger=None)
    heat.close()
    vid.close()
    return out_path


# --------------------------------------------------------------------------- #
def process(npz_path: Path, args) -> Optional[dict]:
    out_dir = Path(args.out_dir) / npz_path.parent.name if args.out_dir \
        else npz_path.parent
    cmp = load_comparison(npz_path, Path(args.archive_dir), Path(args.motion_dir),
                          args.pad_cells, args.hold_mode)
    ser = series(cmp)
    sigma_m = args.sigma_cm / 100.0
    norm_bw = None if args.absolute else ser["bw"]
    vmax = colour_scale(cmp, sigma_m, norm_bw)

    # Crop to the held pose, not to the whole clip. The subject walks the length
    # of a 1.40 m mat to get into a handstand, so a clip-wide box spends three
    # quarters of the panel on footprints and renders the two hands the pose is
    # actually about at a few pixels each. The entry and exit can leave the
    # frame; `--crop-to clip` keeps everything in view instead.
    crop_steps = None
    if args.crop_to == "hold":
        pose = np.where(cmp.hold & ser["share_gate"])[0]
        if len(pose) < 20:
            pose = np.where(ser["share_gate"])[0]
        if len(pose) >= 20:
            crop_steps = pose
    box = view_box(cmp, args.crop_margin, steps=crop_steps)
    print(f"  {cmp.clip}: {cmp.n_steps} steps, vmax {vmax:.2f} N/cm2, "
          f"gate open {100 * ser['share_gate'].mean():.0f}%", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = summary_json(cmp, ser)
    static_summary(cmp, ser, out_dir / "pressure_compare_summary.png", sigma_m,
                   vmax, box, norm_bw)
    if not args.no_animate:
        anim_path = out_dir / f"pressure_compare.{args.format}"
        # Real time by default: the heatmaps and the rendered policy video are
        # both one frame per policy step, so equal playback rates make them
        # watchable side by side (and compositable).
        fps = args.fps or max(int(round(1.0 / (cmp.dt_ctrl * max(args.stride, 1)))), 1)
        animate(cmp, ser, anim_path, sigma_m, args.stride, fps, vmax, box, norm_bw)
        payload["animation_fps"] = fps
        payload["animation"] = str(anim_path)
        if args.video_dir:
            match = sorted(Path(args.video_dir).glob(f"*{cmp.clip}*.mp4"))
            if match and anim_path.suffix == ".mp4":
                lead = max(args.video_settle_steps - cmp.settle_steps, 0) * cmp.dt_ctrl
                payload["composite"] = str(composite_with_video(
                    anim_path, match[0], out_dir / "pressure_compare_with_video.mp4",
                    lead_seconds=lead / max(args.stride, 1),
                ))
                payload["composite_lead_seconds"] = lead
    (out_dir / "pressure_compare.json").write_text(json.dumps(payload, indent=1))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", default=None,
                    help="directory of <clip>/pressure_rollout.npz")
    ap.add_argument("--rollout", default=None, help="a single pressure_rollout.npz")
    ap.add_argument("--out-dir", default=None,
                    help="write next to each rollout if omitted")
    ap.add_argument("--archive-dir", default="data/smpl/yoga_pressure")
    ap.add_argument("--motion-dir",
                    default="data/smpl/yoga_motions_proto_yogi_pressure_gated")
    ap.add_argument("--sigma-cm", type=float, default=2.0,
                    help="Gaussian applied to BOTH fields, for the eye only; the "
                         "scale the pressure attribution already assumes. 0 = raw")
    ap.add_argument("--absolute", action="store_true",
                    help="colour the heatmaps by absolute N/cm2 instead of "
                         "renormalising each to 1 BW. The mat under-reads by "
                         "3-26%% depending on the support (5.1 of the design "
                         "note), so absolute colour makes the policy look like "
                         "it overloads everything. Use for debugging only.")
    ap.add_argument("--pad-cells", type=int, default=16,
                    help="cells (12.7 mm each) of margin outside the sensed mat "
                         "rectangle. 16 = 20 cm, enough that no measured policy "
                         "contact falls off the grid; at 8 the worst frame lost "
                         "14%% of its load, which the renormalisation would then "
                         "have quietly scaled back up to a full body weight.")
    ap.add_argument("--crop-margin", type=float, default=0.12,
                    help="metres of margin around the loaded region")
    ap.add_argument("--hold-mode", choices=("auto", "feet_up", "single_leg"),
                    default="auto",
                    help="what counts as being in the pose, from the REFERENCE: "
                         "both ankles up (inversions, arm balances) or exactly "
                         "one (standing balances). 'auto' prefers both and falls "
                         "back, because a standing balance never lifts both feet "
                         "and would otherwise report n/a for every hold metric.")
    ap.add_argument("--crop-to", choices=("hold", "clip"), default="hold",
                    help="'hold' (default) frames the held pose, so the two "
                         "contacts it stands on fill the panel; the walk-on may "
                         "leave the frame. 'clip' keeps every contact in view.")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth policy step")
    ap.add_argument("--fps", type=int, default=0,
                    help="0 (default) plays back in real time whatever the "
                         "stride, so the animation lines up with the rendered "
                         "policy video frame for frame")
    ap.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    ap.add_argument("--no-animate", action="store_true",
                    help="only the static summary + JSON (much faster)")
    ap.add_argument("--video-settle-steps", type=int, default=2,
                    help="--settle-steps the videos in --video-dir were rendered "
                         "with; the composite trims the difference so the two "
                         "halves show the same motion time")
    ap.add_argument("--video-dir", default=None,
                    help="directory of render_policy_videos.py mp4s; when a clip "
                         "matches, its video is stacked beside the heatmaps")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="case-insensitive substrings selecting clips")
    args = ap.parse_args()

    if args.rollout:
        targets = [Path(args.rollout)]
    elif args.in_dir:
        targets = sorted(Path(args.in_dir).glob("*/pressure_rollout.npz"))
    else:
        raise SystemExit("pass --in-dir or --rollout")
    if args.clips:
        needles = [c.lower() for c in args.clips]
        targets = [p for p in targets
                   if any(n in p.parent.name.lower() for n in needles)]
    if not targets:
        raise SystemExit("no rollouts matched")

    print(f"comparing {len(targets)} clips", flush=True)
    rows = []
    for path in targets:
        try:
            row = process(path, args)
        except Exception as exc:  # keep a bad clip from killing the batch
            print(f"  {path.parent.name}: FAILED -- {exc}", flush=True)
            continue
        if row:
            rows.append(row)

    # Sorted worst-first on the hold-phase zone-share distance, so the ranking
    # is the output rather than something the reader has to build by hand from
    # 29 JSON files. Clips whose measurement is gated off almost everywhere sink
    # to the bottom with an explicit marker rather than a flattering blank.
    rows.sort(key=lambda r: -(r.get("hold_zone_share_tv")
                              if r.get("hold_zone_share_tv") is not None
                              else r.get("zone_share_tv") or -1.0))
    header = (f"{'clip':44s} {'TV':>6s} {'holdTV':>7s} {'feet sim':>9s} "
              f"{'feet ref':>9s} {'unloadP90N':>10s} {'ftUp%':>6s} "
              f"{'onMat':>6s} {'gate%':>6s}")
    lines = []
    for r in rows:
        def g(k, w=6, p=3):
            v = r.get(k)
            return f"{v:{w}.{p}f}" if v is not None and v == v else " " * (w - 3) + "n/a"
        lines.append(
            f"{r['clip'][:44]:44s} {g('zone_share_tv')} {g('hold_zone_share_tv', 7)} "
            f"{g('hold_feet_share_sim', 9)} {g('hold_feet_share_measured', 9)} "
            f"{g('unloaded_zone_load_p90_n', 10, 1)} "
            f"{g('sim_feet_up_in_ref_hold_pct', 6, 1) if 'sim_feet_up_in_ref_hold_pct' in r else g('sim_one_foot_loaded_in_ref_hold_pct', 6, 1)} "
            f"{g('sim_load_on_mat_frac')} {r['share_gate_open_pct']:6.1f}"
        )
    print("\n" + "=" * 120)
    print(header)
    print("-" * 120)
    print("\n".join(lines))
    print("=" * 120)

    if args.out_dir and rows:
        root = Path(args.out_dir)
        root.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k, v in r.items()
                       if isinstance(v, (int, float, str)) or v is None})
        with open(root / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        (root / "summary.md").write_text(
            "# Measured vs learned ground pressure, worst-first by hold-phase "
            "zone-share distance\n\n"
            f"checkpoint: `{rows[0]['checkpoint']}`\n\n"
            "`holdTV` = zone-share total variation on gated hold frames (0 = the "
            "policy loads the same zones the human did).\n"
            "`unloadP90N` = p90 of the newtons the policy puts on zones the "
            "measurement says carry nothing -- the training objective's own "
            "metric. p90 and not the median, because the median is 0 whenever "
            "the offending limb is intermittently clear.\n"
            "`ftUp%` = share of the *reference's* hold during which the policy's "
            "own feet were off the floor.\n"
            "`gate%` = frames where the measured attribution is trustworthy; a low "
            "number means the mat, not the policy, is the problem.\n\n"
            "```\n" + header + "\n" + "-" * 120 + "\n" + "\n".join(lines) + "\n```\n"
        )
        print(f"wrote {root / 'summary.md'} and summary.csv")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
