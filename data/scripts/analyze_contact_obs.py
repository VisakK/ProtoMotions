# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode, check, and plot the contact observations recorded by record_contact_obs.py.

Every encoded channel is inverted back to SI units and compared against the raw
simulator state recorded alongside it, so the figures show whether the numbers
are physically right -- not merely what they look like.

    python data/scripts/analyze_contact_obs.py \
      --npz results/contact_obs_probe/rollout.npz \
      --out-dir results/contact_obs_probe
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

G = 9.81

# --- design tokens (validated palette; see dataviz/references/palette.md) -----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f5f3", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a6a5", "#e34948", "#8c2020"],
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "figure.dpi": 130,
    }
)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# --- inverse of the encoder in protomotions/envs/obs/contact.py --------------
def inv_unsigned_log(y, reference, clip):
    return reference * np.expm1(np.clip(y, 0.0, 1.0) * math.log1p(clip / reference))


def inv_signed_log(y, reference, clip):
    return np.sign(y) * reference * np.expm1(
        np.abs(np.clip(y, -1.0, 1.0)) * math.log1p(clip / reference)
    )


def fwd_unsigned_log(x, reference, clip):
    return np.log1p(np.clip(x, 0.0, clip) / reference) / math.log1p(clip / reference)


def mjcf_mass(path: Path) -> float:
    """Total mass a density-specified MJCF implies, for comparison with the USD."""
    import xml.etree.ElementTree as ET

    total = 0.0
    for geom in ET.parse(path).getroot().iter("geom"):
        density = float(geom.get("density", 0.0) or 0.0)
        kind = geom.get("type")
        if kind == "sphere":
            r = float(geom.get("size"))
            volume = 4 / 3 * math.pi * r**3
        elif kind == "capsule":
            r = float(geom.get("size"))
            ft = [float(v) for v in geom.get("fromto").split()]
            length = math.dist(ft[:3], ft[3:])
            volume = math.pi * r * r * length + 4 / 3 * math.pi * r**3
        elif kind == "box":
            s = [float(v) for v in geom.get("size").split()]
            volume = 8 * s[0] * s[1] * s[2]
        else:
            continue
        total += density * volume
    return total


class Probe:
    """Recorded rollout plus every decoded/derived quantity used downstream."""

    def __init__(self, npz_path: Path, asset_path: str | None = None,
                 fail_threshold: float = 0.5):
        self.asset_path = asset_path
        self.asset_mass = mjcf_mass(Path(asset_path)) if asset_path else None
        d = np.load(npz_path, allow_pickle=False)
        self.d = d
        self.bodies = [str(s) for s in d["obs_body_names"]]
        self.motions = [str(s) for s in d["motion_names"]]
        self.dt = float(d["dt"])
        # Terrain sampling grid, for interpreting the proximity channel. Older
        # captures predate these keys; the defaults match Terrain's own.
        self.sample_width = float(d["terrain_sample_width"]) if (
            "terrain_sample_width" in d.files
        ) else 1.0
        self.samples_per_axis = int(d["terrain_samples_per_axis"]) if (
            "terrain_samples_per_axis" in d.files
        ) else 16
        self.masses = d["body_masses"]
        self.total_mass = float(self.masses.sum())
        self.ch = {n: (int(a), int(b)) for n, (a, b) in zip(d["channel_names"], d["channel_slices"])}
        self.gch = {str(n): i for i, n in enumerate(d["global_channel_names"])}

        self.p = {k[len("param_"):]: float(d[k]) for k in d.files if k.startswith("param_")}
        self.f_ref, self.f_clip = self.p["force_reference_n"], self.p["force_clip_n"]
        self.r_ref, self.r_clip = (
            self.p["force_rate_reference_n_per_s"],
            self.p["force_rate_clip_n_per_s"],
        )

        self.obs = d["contact_obs"]            # [T,E,K,17]
        self.glob = d["contact_global"]        # [T,E,4]
        self.prox = d["proximity"]             # [T,E,K,3]
        self.force = d["contact_forces"]       # [T,E,K,3] world frame, raw
        self.flags = d["contact_flags"]
        self.pos = d["body_pos"]
        self.vel = d["body_vel"]
        self.ground = d["ground_heights"]
        self.active = d["active_state"]
        self.age = d["age_steps"]
        self.air = d["air_age_steps"]
        self.tvalid = d["temporal_valid"]
        self.done = d["done"]
        self.err = d["track_err"]
        self.T, self.E, self.K = self.obs.shape[:3]

        # decoded channels, SI units
        self.dec_mag = inv_unsigned_log(self.c("force_magnitude")[..., 0], self.f_ref, self.f_clip)
        self.dec_up = inv_unsigned_log(self.c("upward_force_proxy")[..., 0], self.f_ref, self.f_clip)
        self.dec_horiz = inv_unsigned_log(
            self.c("horizontal_force_proxy")[..., 0], self.f_ref, self.f_clip
        )
        self.dec_vec = inv_signed_log(self.c("net_force_heading"), self.f_ref, self.f_clip)
        self.dec_rate = inv_signed_log(self.c("force_rate_heading"), self.r_ref, self.r_clip)
        self.dec_total_up = inv_unsigned_log(
            self.glob[..., self.gch["total_upward_force_proxy"]], self.f_ref, self.f_clip
        )
        self.dec_net_horiz = inv_unsigned_log(
            self.glob[..., self.gch["total_horizontal_force_proxy"]], self.f_ref, self.f_clip
        )

        # raw references
        self.raw_mag = np.linalg.norm(self.force, axis=-1)
        self.raw_up = np.clip(self.force[..., 2], 0.0, None)
        self.raw_horiz = np.linalg.norm(self.force[..., :2], axis=-1)
        self.height = self.pos[..., 2] - self.ground[:, :, None]

        # Failure, and the separate matter of the clip simply running out.
        #
        # The experiment's apply_inference_overrides CLEARS termination_components,
        # so `done` here is only the motion manager reaching the end of the clip --
        # it is NOT a fall. Failure has to be judged the way training judged it:
        # max per-body tracking error past the tracking_error threshold.
        self.fail_threshold = fail_threshold
        self.clip_end = np.full(self.E, self.T, dtype=int)
        self.first_done = np.full(self.E, self.T, dtype=int)
        for e in range(self.E):
            hits = np.nonzero(self.done[:, e])[0]
            if hits.size:
                self.clip_end[e] = int(hits[0])
            lost = np.nonzero(self.err[:, e] > self.fail_threshold)[0]
            if lost.size:
                self.first_done[e] = int(lost[0])
        # Nothing downstream should see a "failure" after the reference ran out.
        self.first_done = np.minimum(self.first_done, self.clip_end)

        # Newton check: mass-weighted body-origin velocity as a COM proxy.
        w = self.masses[None, None, :] / self.total_mass
        self.v_com = (self.vel * w[..., None]).sum(axis=2)          # [T,E,3]
        self.a_com = np.zeros_like(self.v_com)
        self.a_com[1:] = (self.v_com[1:] - self.v_com[:-1]) / self.dt
        self.expected_up = self.total_mass * (G + self.a_com[..., 2])
        self.raw_signed_up_sum = self.force[..., 2].sum(axis=2)      # external only (3rd law)

    def c(self, name):
        a, b = self.ch[name]
        return self.obs[..., a:b]

    def t(self):
        return np.arange(self.T) * self.dt

    def held(self, e, lo=0.30, hi=0.95):
        """Steps in the middle of env e's episode, before it fell (if it did)."""
        end = self.first_done[e]
        return slice(int(end * lo), max(int(end * hi), int(end * lo) + 1))


def fmt(x, n=3):
    return f"{x:.{n}g}"


# ---------------------------------------------------------------- checks ----
def run_checks(p: Probe) -> list[tuple[str, str, str]]:
    """(name, verdict, detail). Verdict is OK / WARN / FAIL / INFO."""
    out = []

    def add(name, ok, detail, warn_only=False):
        out.append((name, "OK" if ok else ("WARN" if warn_only else "FAIL"), detail))

    e_mag = np.abs(p.dec_mag - np.minimum(p.raw_mag, p.f_clip))
    add(
        "force_magnitude decodes to the raw |F|",
        e_mag.max() < 0.05,
        f"max abs error {fmt(e_mag.max())} N over {e_mag.size:,} samples",
    )
    e_up = np.abs(p.dec_up - np.minimum(p.raw_up, p.f_clip))
    e_h = np.abs(p.dec_horiz - np.minimum(p.raw_horiz, p.f_clip))
    add(
        "upward / horizontal proxies decode to clamp(Fz,0) and |Fxy|",
        max(e_up.max(), e_h.max()) < 0.05,
        f"max abs error {fmt(e_up.max())} N (up), {fmt(e_h.max())} N (horizontal)",
    )

    # Heading rotation is yaw-only: it must preserve z and the norm.
    keep = p.raw_mag < p.f_clip
    dz = np.abs(p.dec_vec[..., 2] - p.force[..., 2])[keep]
    dn = np.abs(np.linalg.norm(p.dec_vec, axis=-1) - p.raw_mag)[keep]
    add(
        "net_force_heading is a pure yaw rotation of the world force",
        dz.max() < 0.5 and dn.max() < 0.5,
        f"max |dz| {fmt(dz.max())} N, max |d|F||  {fmt(dn.max())} N",
    )

    lf = p.c("support_load_fraction_proxy")[..., 0].sum(axis=-1)
    any_up = p.raw_up.sum(axis=-1) > 1e-3
    add(
        "support_load_fraction sums to 1 whenever anything is loaded",
        np.abs(lf[any_up] - 1.0).max() < 1e-3,
        f"max |sum-1| = {fmt(np.abs(lf[any_up] - 1.0).max())} on loaded steps; "
        f"{(~any_up).mean() * 100:.1f}% of steps carry no upward force at all",
    )

    act = p.c("active")[..., 0]
    add(
        "active channel matches the env-side hysteresis state",
        np.array_equal(act, p.active),
        "bit-identical" if np.array_equal(act, p.active) else "mismatch",
    )
    on = p.raw_mag >= 5.0
    off = p.raw_mag < 2.0
    add(
        "hysteresis respects its 5 N on / 2 N off thresholds",
        bool(act[on].min() == 1.0) if on.any() else True,
        f"active on all {on.sum():,} samples with |F|>=5 N; "
        f"{(act[off] > 0).mean() * 100:.2f}% still active in the 0-2 N band "
        f"(raw contact flag holding them on)",
    )

    age = p.c("contact_age")[..., 0]
    air = p.c("air_age")[..., 0]
    both = (age > 0) & (air > 0)
    add(
        "contact_age and air_age are mutually exclusive",
        not both.any(),
        f"{both.sum()} samples with both non-zero",
    )
    slope = np.diff(age, axis=0)
    ramping = slope[slope > 0]
    add(
        "contact_age ramps at dt / contact_age_clip_s",
        np.allclose(ramping, p.dt / p.p["contact_age_clip_s"], atol=1e-6) if ramping.size else True,
        f"observed step {fmt(np.median(ramping)) if ramping.size else 'n/a'} "
        f"vs expected {fmt(p.dt / p.p['contact_age_clip_s'])}",
    )

    # The env clears temporal_valid at reset and only promotes the current force
    # to "previous" at the END of post_physics_step, so the reset-return
    # observation AND the first post-physics observation both report invalid.
    tv = p.c("temporal_valid")[..., 0]
    add(
        "temporal_valid is 0 exactly on the two post-reset observations",
        bool((tv[:2] == 0).all() and (tv[2:] == 1).all()),
        f"steps 0-1: {tv[:2].max():.0f}, steps 2+: min {tv[2:].min():.0f} "
        "(force rate is deliberately dead for two steps, not one)",
    )
    add(
        "force_rate is exactly zero wherever temporal_valid is 0",
        bool((p.c("force_rate_heading")[tv == 0] == 0).all()),
        "no fictitious rate spike is injected from the masked reset sample",
    )

    g_any = p.glob[..., p.gch["any_selected_contact"]]
    g_frac = p.glob[..., p.gch["active_body_fraction"]]
    add(
        "global any/fraction channels agree with the per-body active bits",
        np.allclose(g_any, (act.max(-1) > 0)) and np.allclose(g_frac, act.mean(-1), atol=1e-6),
        "consistent",
    )
    e_tot = np.abs(p.dec_total_up - np.minimum(p.raw_up.sum(-1), p.f_clip))
    add(
        "total_upward_force decodes to the sum of per-body upward forces",
        e_tot.max() < 0.5,
        f"max abs error {fmt(e_tot.max())} N",
    )

    # Flat terrain: every sampled terrain point shares one z, so the proximity
    # z channel is exactly minus the body height above ground.
    pz = np.abs(p.prox[..., 2] + p.height)
    add(
        "proximity z equals -(height above ground) on flat terrain",
        pz.max() < 5e-3,
        f"max abs error {fmt(pz.max())} m",
    )

    # ---- physics, not encoding ----
    resid = np.abs(p.raw_signed_up_sum[1:] - p.expected_up[1:])
    add(
        "Newton's 2nd law: Σ Fz = m(g + a_com,z)",
        np.median(resid) < 0.05 * p.total_mass * G,
        f"residual {fmt(np.median(resid), 2)} N median "
        f"({np.median(resid) / (p.total_mass * G) * 100:.1f}% of weight), "
        f"{fmt(np.percentile(resid, 95), 2)} N at p95 "
        "(a_com from mass-weighted body-origin velocities, so the tail is proxy error)",
    )

    ratios = []
    for e in range(p.E):
        s = p.held(e)
        if s.stop > s.start:
            ratios.append(p.dec_total_up[s, e].mean() / (p.total_mass * G))
    ratios = np.array(ratios)
    out.append(
        (
            "held-pose Σ upward force vs body weight",
            "INFO",
            f"simulated mass {p.total_mass:.2f} kg -> weight {p.total_mass * G:.0f} N; "
            f"per-motion ratio median {np.median(ratios):.3f}, "
            f"range {ratios.min():.2f}-{ratios.max():.2f}",
        )
    )

    if p.asset_mass:
        add(
            "simulated mass matches the source MJCF asset",
            abs(p.total_mass - p.asset_mass) / p.asset_mass < 0.02,
            f"USD articulation is {p.total_mass:.2f} kg but "
            f"{Path(p.asset_path).name} specifies {p.asset_mass:.2f} kg "
            f"({p.total_mass / p.asset_mass * 100:.0f}%) -- every contact force is "
            "scaled by that ratio",
        )

    dbl = p.raw_up.sum(-1) - p.raw_signed_up_sum
    worst = int(np.argmax(np.median(dbl, axis=0)))
    out.append(
        (
            "clamp(Fz,0)-then-sum double-counts self-contact",
            "INFO",
            f"Σclamp - ΣFz is {fmt(np.median(dbl), 2)} N median, "
            f"{fmt(np.percentile(dbl, 99), 3)} N at p99; worst motion "
            f"{p.motions[worst]} (+{np.median(dbl[:, worst]):.0f} N, "
            f"{np.median(dbl[:, worst]) / (p.total_mass * G) * 100:.0f}% of weight)",
        )
    )

    # Isaac Lab's net_forces_w sums NORMAL contact forces only. On flat ground
    # every ground normal is +z, so the tangential channels can only ever be fed
    # by body-on-body contact.
    act = p.c("active")[..., 0].astype(bool)
    horiz = p.c("horizontal_force_proxy")[..., 0][act]
    vert = p.c("upward_force_proxy")[..., 0][act]
    fric = p.c("ground_friction_utilization_proxy")[..., 0][act]
    tilt = np.degrees(np.arctan2(p.raw_horiz[act], np.maximum(p.raw_up[act], 1e-9)))
    floor = p.height[act] <= 0.15

    add(
        "'active' means ground support",
        (~floor).mean() < 0.05,
        f"{(~floor).mean() * 100:.0f}% of in-contact samples are on a body more than "
        "0.15 m above the floor; self-collisions are enabled, so the channel means "
        "'net contact force', ground or body-on-body",
        warn_only=True,
    )
    add(
        "tangential channels carry usable signal on floor contacts",
        np.median(horiz[floor]) > 0.02,
        f"floor contacts have a summed normal tilted {np.median(tilt[floor]):.0f} deg off "
        f"vertical (median) and horizontal_force_proxy < 0.01 in "
        f"{(horiz[floor] < 0.01).mean() * 100:.0f}% of them -- Isaac Lab sums NORMAL "
        "forces only, so flat ground can only ever report +z",
        warn_only=True,
    )
    add(
        "friction_utilization_proxy behaves like a friction ratio",
        (fric >= 1.0).mean() < 0.02 and np.median(fric) > 0.01,
        f"median {np.median(fric):.3f} but saturated at 1.0 in {(fric >= 1.0).mean() * 100:.0f}% "
        f"of in-contact samples -- it pins high exactly where upward force is zero "
        f"({(vert == 0).mean() * 100:.0f}% of samples, i.e. the sideways body-on-body "
        "contacts), so it is a 'this contact is lateral' flag, not a slip margin",
        warn_only=True,
    )

    rate_enc = np.abs(p.c("force_rate_heading"))
    rate = np.abs(p.dec_rate).max(-1)
    add(
        "force_rate clipping is rare",
        (rate_enc >= 1.0).mean() < 1e-4,
        f"{(rate_enc >= 1.0).mean() * 100:.4f}% of rate components saturate the "
        f"{p.r_clip:.0f} N/s clip; p99.99 of |dF/dt| is {np.percentile(rate, 99.99):.0f} N/s",
    )

    mag_enc = p.c("force_magnitude")[..., 0][act]
    add(
        "force_magnitude uses a useful share of its encoded range",
        np.percentile(mag_enc, 99) > 0.6,
        f"among loaded bodies the encoded value reaches only "
        f"{np.percentile(mag_enc, 99):.2f} at p99 and {mag_enc.max():.2f} at max -- "
        f"forces above ~{inv_unsigned_log(np.percentile(mag_enc, 99), p.f_ref, p.f_clip):.0f} N "
        f"never occur, so the top of the [0,1] domain is unused",
        warn_only=True,
    )
    return out


# ---------------------------------------------------------------- figures ---
def fig_encoding(p: Probe, out: Path):
    act_all = p.c("active")[..., 0].astype(bool)
    enc_loaded = p.c("force_magnitude")[..., 0][act_all]
    enc_p99 = float(np.percentile(enc_loaded, 99))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    fig.suptitle(
        "1 · The force encoding is exact and its resolution is where the data is — "
        f"but 99% of loaded samples sit below {enc_p99:.2f} of the [0,1] domain",
        fontsize=12, fontweight="semibold", color=INK, y=0.985,
    )

    ax = style(axes[0, 0])
    xs = np.logspace(-1, np.log10(p.f_clip), 400)
    ys = fwd_unsigned_log(xs, p.f_ref, p.f_clip)
    ax.semilogx(xs, ys, color=CAT[0], zorder=3)
    mags = p.raw_mag[p.raw_mag > 1e-3]
    weight = p.total_mass * G
    marks = [
        (np.median(mags), "median |F|", CAT[1], 0.62),
        (np.percentile(mags, 99.9), "p99.9", CAT[2], 0.80),
        (weight, "body weight", INK_3, 0.30),
    ]
    for value, label, col, ytext in marks:
        ax.axvline(value, color=col, lw=1.1, alpha=0.9, zorder=2)
        ax.text(value * 1.12, ytext, f"{label}\n{value:.0f} N", fontsize=7.2, color=col,
                va="center", ha="left")
    ax.fill_between(xs, ys, 1.0, where=xs >= np.percentile(mags, 99.9),
                    color="#f0efec", zorder=1)
    ax.axhspan(fwd_unsigned_log(np.percentile(mags, 99.9), p.f_ref, p.f_clip), 1.0,
               color="#f0efec", zorder=0)
    ax.text(0.12, 0.93, "encoded range never used", fontsize=7.6, color=INK_2, va="center")
    ax.set_xlabel("contact force magnitude (N, log)")
    ax.set_ylabel("encoded value")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"compression curve   ref={p.f_ref:.0f} N, clip={p.f_clip:.0f} N", loc="left")

    ax = style(axes[0, 1])
    ax.hist(mags, bins=np.logspace(-1, np.log10(p.f_clip), 60), color=CAT[0], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("contact force magnitude (N, log)")
    ax.set_ylabel("samples (log)")
    ax.set_title(
        f"what the sim actually produces — {(mags > p.f_clip).mean() * 100:.3f}% clipped",
        loc="left",
    )

    ax = style(axes[1, 0])
    sub = slice(None, None, 37)
    raw = p.raw_mag.reshape(-1)[sub]
    dec = p.dec_mag.reshape(-1)[sub]
    keep = raw > 1e-2
    lim = [1e-2, max(raw.max(), 1.0) * 1.5]
    ax.plot(lim, lim, color=INK_3, lw=2.0, zorder=1, alpha=0.45)
    ax.loglog(raw[keep], dec[keep], ".", ms=2.2, color=CAT[0], alpha=0.75, mec="none",
              zorder=3)
    ax.text(lim[1], lim[1], " y = x", fontsize=7.5, color=INK_2, va="center")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("raw |F| from the simulator (N)")
    ax.set_ylabel("decoded from the observation (N)")
    ax.set_title("round-trip: encode → decode is lossless", loc="left")

    ax = style(axes[1, 1])
    enc = p.c("force_magnitude")[..., 0]
    bins = np.linspace(0, 1, 60)
    ax.hist(enc.reshape(-1), bins=bins, color=CAT[0], alpha=0.9,
            label="all sensed bodies")
    ax.hist(enc_loaded, bins=bins, color=CAT[1], alpha=0.75, label="bodies in contact")
    ax.axvline(enc_p99, color=INK_3, lw=1.1)
    ax.text(enc_p99, ax.get_ylim()[1], f" p99 of loaded bodies = {enc_p99:.2f}",
            fontsize=7.2, color=INK_2, va="top")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("encoded force_magnitude value")
    ax.set_ylabel("samples (log)")
    ax.set_title(f"largest value ever produced is {enc.max():.2f}", loc="left")
    ax.legend(fontsize=7.4, loc="center right")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out / "fig_01_encoding.png", bbox_inches="tight")
    plt.close(fig)


def fig_newton(p: Probe, out: Path):
    ncol, nrow = 4, int(np.ceil(p.E / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.5 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(
        "2 · Vertical force balance — decoded Σ upward force vs the weight it has to carry",
        fontsize=12, fontweight="semibold", color=INK, y=0.997,
    )
    t = p.t()
    for e in range(p.E):
        ax = style(axes[e])
        end = p.first_done[e]
        if end < p.T:
            ax.axvspan(t[end], t[-1], color="#f0efec", zorder=0)
            ax.text(t[end], 0.97, " lost ref", transform=ax.get_xaxis_transform(),
                    fontsize=7, color=INK_3, va="top")
        ax.plot(t, p.expected_up[:, e], color=CAT[1], lw=1.0, alpha=0.9,
                label="m(g + a$_z$)" if e == 0 else None)
        ax.plot(t, p.dec_total_up[:, e], color=CAT[0], lw=1.2,
                label="Σ upward (decoded)" if e == 0 else None)
        ax.axhline(p.total_mass * G, color=INK_3, lw=0.9, ls="-",
                   label="m·g" if e == 0 else None)
        ax.set_ylim(0, max(p.total_mass * G * 3.2, np.percentile(p.dec_total_up[:, e], 99)))
        ax.set_title(p.motions[e].replace("220923_", "").replace("220926_", "")[:38],
                     loc="left", fontsize=8)
        if e % ncol == 0:
            ax.set_ylabel("N")
        if e >= p.E - ncol:
            ax.set_xlabel("time (s)")
    for e in range(p.E, len(axes)):
        axes[e].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=3, bbox_to_anchor=(0.99, 0.005))
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    fig.savefig(out / "fig_02_force_balance.png", bbox_inches="tight")
    plt.close(fig)


def fig_support_map(p: Probe, out: Path):
    ncol, nrow = 2, int(np.ceil(p.E / 2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.0 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(
        "3 · Which bodies carry the load — support_load_fraction, all 24 sensed bodies\n"
        "red line = tracking error passed the 0.5 m threshold training terminates on",
        fontsize=12, fontweight="semibold", color=INK, y=0.999,
    )
    lf = p.c("support_load_fraction_proxy")[..., 0]
    im = None
    for e in range(p.E):
        ax = axes[e]
        im = ax.imshow(lf[:, e].T, aspect="auto", origin="lower", cmap=SEQ,
                       vmin=0, vmax=1, extent=[0, p.T * p.dt, -0.5, p.K - 0.5],
                       interpolation="nearest")
        end = p.first_done[e]
        if end < p.T:
            ax.axvline(end * p.dt, color="#e34948", lw=1.4)
        ax.set_yticks(range(p.K))
        ax.set_yticklabels(p.bodies, fontsize=6.4)
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_title(p.motions[e].replace("220923_", "").replace("220926_", ""),
                     loc="left", fontsize=9)
        if e >= p.E - ncol:
            ax.set_xlabel("time (s)")
    for e in range(p.E, len(axes)):
        axes[e].axis("off")
    fig.tight_layout(rect=(0, 0, 0.93, 0.975))
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.012, pad=0.012)
    cb.set_label("fraction of total upward force", color=INK_2)
    cb.outline.set_visible(False)
    fig.savefig(out / "fig_03_support_map.png", bbox_inches="tight")
    plt.close(fig)


def fig_channels(p: Probe, out: Path, env: int, bodies: list[str], tag: str):
    ids = [p.bodies.index(b) for b in bodies]
    t = p.t()
    end = p.first_done[env]
    rows = [
        ("active", "active", None, (-0.1, 1.1)),
        ("net_force_heading", "force in heading frame (x,y,z)", None, None),
        ("force_magnitude", "|F|", None, (0, 1)),
        ("upward_force_proxy", "upward force", None, (0, 1)),
        ("horizontal_force_proxy", "horizontal force", None, (0, 1)),
        ("ground_friction_utilization_proxy", "friction utilisation", None, (0, 1.05)),
        ("support_load_fraction_proxy", "load fraction", None, (0, 1.05)),
        ("force_rate_heading", "force rate (x,y,z)", None, None),
        ("body_origin_normal_velocity_proxy", "vertical velocity", None, (-1.05, 1.05)),
        ("body_origin_tangent_speed_proxy", "tangential speed", None, (0, 1.05)),
        ("contact_age", "contact age", None, (0, 1.05)),
        ("air_age", "air age", None, (0, 1.05)),
    ]
    # The three-component rows plot one body only; pick the one that actually
    # carries this pose so the panel is not a flat line.
    load = p.c("support_load_fraction_proxy")[: end, env, :, 0].mean(axis=0)
    lead = int(np.argmax([load[i] for i in ids]))
    lead_id = ids[lead]

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.28 * len(rows)), sharex=True)
    fig.suptitle(
        f"4 · Every contact channel the policy reads — {p.motions[env]}",
        fontsize=11.5, fontweight="semibold", color=INK, y=1.0,
    )
    handles = None
    for ax, (chan, label, _, ylim) in zip(axes, rows):
        style(ax)
        block = p.c(chan)
        if block.shape[-1] == 3:
            for k, (comp, col) in enumerate(zip("xyz", CAT[:3])):
                ax.plot(t, block[:, env, lead_id, k], color=col, lw=1.1, label=comp)
            ax.legend(loc="upper right", ncol=3, fontsize=7, handlelength=1.2,
                      columnspacing=1.0, borderaxespad=0.2,
                      title=f"{p.bodies[lead_id]} only", title_fontsize=6.8)
            label = f"{label}\n[{p.bodies[lead_id]}]"
        else:
            for j, bi in enumerate(ids):
                ax.plot(t, block[:, env, bi, 0], color=CAT[j % len(CAT)], lw=1.1,
                        label=p.bodies[bi])
            handles = ax.get_legend_handles_labels()
        if end < p.T:
            ax.axvspan(t[end], t[-1], color="#f0efec", zorder=0)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(label, fontsize=7.6)
    axes[-1].set_xlabel("time (s)")
    if handles:
        fig.legend(*handles, loc="upper center", ncol=len(ids), fontsize=8.4,
                   bbox_to_anchor=(0.5, 0.978), handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    fig.savefig(out / f"fig_04_channels_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_health(p: Probe, out: Path):
    """Which channels actually carry information *when a body is in contact*.

    Unconditional statistics are dominated by the ~86% of body-steps with no
    contact at all, which makes every force channel look dead.
    """
    act = p.c("active")[..., 0].astype(bool)
    cols, labels, domains = [], [], []
    for name in p.ch:
        block = p.c(name)
        for k in range(block.shape[-1]):
            cols.append(block[..., k][act])
            labels.append(name if block.shape[-1] == 1 else f"{name}[{'xyz'[k]}]")
            domains.append((-1.0, 1.0) if block[..., k].min() < -1e-6 else (0.0, 1.0))

    n = len(cols)
    y = np.arange(n)                       # ascending; row 0 drawn at the bottom
    lo = np.array([np.percentile(c, 1) for c in cols])
    hi = np.array([np.percentile(c, 99) for c in cols])
    med = np.array([np.median(c) for c in cols])
    stds = np.array([c.std() for c in cols])
    frac0 = np.array([(c == 0).mean() * 100 for c in cols])
    used = (hi - lo) / np.array([d[1] - d[0] for d in domains]) * 100
    # active / air_age / temporal_valid are constant *because* we conditioned on
    # contact; that is not a property of the channel, so do not flag it as dead.
    pinned = stds == 0.0
    dead = (stds < 0.05) & ~pinned

    def bar_color(i):
        return "#c9c8c3" if pinned[i] else (CAT[1] if dead[i] else CAT[0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6),
                             gridspec_kw={"width_ratios": [2.0, 1, 1]})
    fig.suptitle(
        "5 · Channel health, conditioned on the body actually being in contact\n"
        "orange = the channel barely moves (σ < 0.05) · grey = constant because of that "
        f"conditioning · unconditional stats would be swamped by the "
        f"{(~act).mean() * 100:.0f}% of body-steps with no contact at all",
        fontsize=11.5, fontweight="semibold", color=INK, y=1.02,
    )

    ax = style(axes[0])
    for i in range(n):
        d0, d1 = domains[i]
        ax.plot([d0, d1], [y[i], y[i]], color="#eceae5", lw=7, solid_capstyle="butt",
                zorder=1)
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=bar_color(i), lw=7,
                solid_capstyle="butt", zorder=2)
        ax.plot([med[i]], [y[i]], "|", color=SURFACE, ms=9, mew=1.8, zorder=3)
        if pinned[i]:
            ax.text(0.04, y[i], f"pinned at {med[i]:.0f}", fontsize=7, color=INK_3,
                    va="center")
        ax.text(1.04, y[i], f"{used[i]:.0f}%", fontsize=7, color=INK_2, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-1.08, 1.16)
    ax.set_xlabel("channel value — grey = declared domain, bar = p1…p99 seen, tick = median")
    ax.set_title("range exercised (right label = % of domain used)", loc="left")

    ax = style(axes[1])
    ax.barh(y, stds, color=[bar_color(i) for i in range(n)], height=0.6)
    ax.axvline(0.05, color=INK_3, lw=1.0)
    for yi, v in zip(y, stds):
        ax.text(v + stds.max() * 0.02, yi, f"{v:.3f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, stds.max() * 1.3)
    ax.tick_params(length=0)
    ax.set_xlabel("standard deviation")
    ax.set_title("signal (line = σ 0.05)", loc="left")

    ax = style(axes[2])
    ax.barh(y, frac0, color=[bar_color(i) for i in range(n)], height=0.6)
    for yi, v in zip(y, frac0):
        ax.text(v + 1.5, yi, f"{v:.0f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, 112)
    ax.tick_params(length=0)
    ax.set_xlabel("% of in-contact samples that are exactly zero")
    ax.set_title("dead mass", loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_05_channel_health.png", bbox_inches="tight")
    plt.close(fig)


def fig_proximity(p: Probe, out: Path, env: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    fig.suptitle(
        "6 · Proximity channel — vector from each body origin to the nearest sampled terrain point",
        fontsize=12, fontweight="semibold", color=INK, y=0.985,
    )

    ax = style(axes[0, 0])
    sub = slice(None, None, 29)
    x = (-p.prox[..., 2]).reshape(-1)[sub]
    y = p.height.reshape(-1)[sub]
    ax.plot(x, y, ".", ms=2, color=CAT[0], alpha=0.3, mec="none")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color=INK_3, lw=1.0)
    ax.text(lim[1], lim[1], " y = x", fontsize=7.5, color=INK_2, va="center")
    ax.set_xlabel("-(proximity z channel)  (m)")
    ax.set_ylabel("body height above ground (m)")
    ax.set_title("the z channel is exactly the body's height", loc="left")

    ax = style(axes[0, 1])
    xy = np.linalg.norm(p.prox[..., :2], axis=-1).reshape(-1)
    ax.hist(xy, bins=70, color=CAT[0])
    # The terrain sampling grid is num_samples_per_axis over +/- sample_width,
    # so the worst case is half a cell diagonal.
    pitch = 2 * p.sample_width / (p.samples_per_axis - 1)
    half_diag = pitch / 2 * math.sqrt(2)
    ax.axvline(half_diag, color=CAT[1], lw=1.2)
    ax.text(half_diag, ax.get_ylim()[1] * 0.95,
            f"half-diagonal of the\n{pitch * 100:.1f} cm sampling cell  ",
            fontsize=7.2, color=CAT[1], va="top", ha="right")
    ax.set_xlabel("horizontal offset to the nearest sampled point (m)")
    ax.set_ylabel("samples")
    ax.set_title("horizontal components are pure grid-sampling noise", loc="left")

    ax = style(axes[1, 0])
    t = p.t()
    feature = ["L_Toe", "R_Toe", "L_Hand", "R_Hand", "Head", "Pelvis"]
    for j, b in enumerate(feature):
        bi = p.bodies.index(b)
        ax.plot(t, p.height[:, env, bi], color=CAT[j % len(CAT)], lw=1.2, label=b)
    end = p.first_done[env]
    if end < p.T:
        ax.axvspan(t[end], t[-1], color="#f0efec", zorder=0)
    ax.axhline(0, color=INK_3, lw=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height above ground (m)")
    ax.set_title(f"height traces — {p.motions[env][:44]}", loc="left")
    ax.legend(ncol=3, fontsize=7.4, loc="upper right")

    ax = style(axes[1, 1])
    act = p.c("active")[..., 0].reshape(-1).astype(bool)
    h = p.height.reshape(-1)
    bins = np.linspace(-0.05, 0.9, 80)
    ax.hist(h[act], bins=bins, color=CAT[0], alpha=0.9, label="active = 1")
    ax.hist(h[~act], bins=bins, color=CAT[1], alpha=0.5, label="active = 0")
    ax.axvline(0.15, color=INK_3, lw=1.1)
    off_ground = (h[act] > 0.15).mean() * 100
    ax.text(0.15, ax.get_ylim()[1], f"  {off_ground:.0f}% of contacts are above 0.15 m\n"
            "  — those are body-on-body, not the floor",
            fontsize=7.4, color=INK_2, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("body height above ground (m)")
    ax.set_ylabel("samples (log)")
    ax.set_title("'active' means net contact force, not ground support", loc="left")
    ax.legend(fontsize=7.6, loc="center right")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out / "fig_06_proximity.png", bbox_inches="tight")
    plt.close(fig)


def fig_temporal(p: Probe, out: Path, env: int):
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True)
    fig.suptitle(
        f"7 · Temporal contact state and impact transients — {p.motions[env][:52]}",
        fontsize=12, fontweight="semibold", color=INK, y=0.995,
    )
    t = p.t()
    feature = ["L_Toe", "L_Ankle", "R_Toe", "R_Ankle"]
    ids = [p.bodies.index(b) for b in feature]
    end = p.first_done[env]

    ax = style(axes[0])
    for j, bi in enumerate(ids):
        ax.plot(t, p.c("contact_age")[:, env, bi, 0], color=CAT[j], lw=1.2,
                label=f"{p.bodies[bi]} contact")
        ax.plot(t, -p.c("air_age")[:, env, bi, 0], color=CAT[j], lw=1.0, alpha=0.45,
                label=f"{p.bodies[bi]} air")
    ax.axhline(0, color=INK_3, lw=0.9)
    ax.set_ylabel("age  (contact ↑ / air ↓)")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(ncol=4, fontsize=6.8, loc="lower right")
    ax.set_title("2 s saturating ramps; they are mutually exclusive by construction", loc="left")

    ax = style(axes[1])
    for j, bi in enumerate(ids):
        ax.plot(t, p.dec_mag[:, env, bi], color=CAT[j], lw=1.1, label=p.bodies[bi])
    ax.axhline(p.total_mass * G, color=INK_3, lw=0.9)
    ax.text(t[-1], p.total_mass * G, " m·g ", fontsize=7.2, color=INK_2, ha="right", va="bottom")
    ax.set_ylabel("decoded |F| (N)")
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(ncol=4, fontsize=7.2, loc="upper right")
    ax.set_title("decoded per-body force magnitude", loc="left")

    ax = style(axes[2])
    rate = np.linalg.norm(p.dec_rate[:, env], axis=-1)
    for j, bi in enumerate(ids):
        ax.plot(t, rate[:, bi], color=CAT[j], lw=1.1, label=p.bodies[bi])
    ax.axhline(p.r_clip, color="#e34948", lw=1.0)
    ax.text(t[-1], p.r_clip, f" clip {p.r_clip:.0f} N/s ", fontsize=7.2, color="#e34948",
            ha="right", va="bottom")
    ax.set_ylim(0, p.r_clip * 1.35)
    ax.set_ylabel("decoded |dF/dt| (N/s)")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("time (s)")
    ax.legend(ncol=4, fontsize=7.2, loc="upper right")
    ax.set_title(
        f"force rate — peaks at {rate.max():.0f} N/s, so the clip is never approached",
        loc="left",
    )

    for ax in axes:
        if end < p.T:
            ax.axvspan(t[end], t[-1], color="#f0efec", zorder=0)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / "fig_07_temporal.png", bbox_inches="tight")
    plt.close(fig)


def fig_tracking(p: Probe, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle(
        "0 · How the policy did on each reference motion\n"
        "terminations are disabled at inference, so a clip simply ending is not a failure",
        fontsize=12, fontweight="semibold", color=INK, y=1.0)
    t = p.t()
    lost = p.first_done < p.clip_end          # tracking actually diverged
    ax = style(axes[0])
    for e in range(p.E):
        col = "#e34948" if lost[e] else CAT[0]
        ax.plot(t, p.err[:, e], color=col, lw=1.0, alpha=0.55 if lost[e] else 0.9)
    ax.axhline(p.fail_threshold, color=INK_3, lw=1.0)
    ax.text(t[-1], p.fail_threshold,
            f" {p.fail_threshold} m — training's tracking_error termination ",
            fontsize=7.2, color=INK_2, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("max per-body tracking error (m)")
    ax.set_title("blue = held the reference, red = lost it", loc="left")

    ax = style(axes[1])
    held = p.first_done * p.dt
    idx = np.argsort(held)
    ax.barh(np.arange(p.E), held[idx],
            color=["#e34948" if lost[e] else CAT[0] for e in idx], height=0.66)
    ax.barh(np.arange(p.E), (p.clip_end * p.dt)[idx], color="#eceae5", height=0.66,
            zorder=0)
    ax.set_yticks(np.arange(p.E))
    ax.set_yticklabels([p.motions[e].replace("220923_", "").replace("220926_", "")[:40]
                        for e in idx], fontsize=7.2)
    for y, e in enumerate(idx):
        note = f"{held[e]:.1f}s" + ("" if lost[e] else " (clip end)")
        ax.text(held[e] + 0.2, y, note, va="center", fontsize=7, color=INK_2)
    ax.set_xlabel("time tracked within the window (s) — grey = clip length")
    n_lost = int(lost.sum())
    ax.set_title(f"{p.E - n_lost}/{p.E} tracked their whole clip", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_00_tracking.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fail-threshold", type=float, default=0.5,
                    help="Max per-body tracking error counted as losing the reference; "
                         "match the training tracking_error termination.")
    ap.add_argument(
        "--asset-mjcf",
        default="data/assets/smpl/smpl_yogi03596_lowtorque.xml",
        help="Source MJCF, to check the simulated mass against the intended one.",
    )
    a = ap.parse_args()

    out = Path(a.out_dir)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    asset = a.asset_mjcf if a.asset_mjcf and Path(a.asset_mjcf).exists() else None
    p = Probe(Path(a.npz), asset_path=asset, fail_threshold=a.fail_threshold)

    print(f"{p.T} steps x {p.E} motions x {p.K} bodies, dt={p.dt:.4f}s, "
          f"mass={p.total_mass:.2f} kg"
          + (f" (MJCF says {p.asset_mass:.2f} kg)" if p.asset_mass else ""))

    checks = run_checks(p)
    width = max(len(n) for n, _, _ in checks)
    lines = ["# Contact observation probe\n",
             f"`{a.npz}` — {p.T} steps ({p.T * p.dt:.1f} s) x {p.E} motions x {p.K} bodies, "
             f"model mass {p.total_mass:.1f} kg ({p.total_mass * G:.0f} N)\n",
             "| check | verdict | detail |", "|---|---|---|"]
    for name, verdict, detail in checks:
        print(f"  [{verdict:<4}] {name:<{width}}  {detail}")
        lines.append(f"| {name} | **{verdict}** | {detail} |")

    # Semantic check: does the observation say the pose is supported by the
    # bodies a human would use? Averaged over the held phase of each clip.
    lf = p.c("support_load_fraction_proxy")[..., 0]
    lines += ["", "## What the observation says is carrying each pose", "",
              "Mean `support_load_fraction` over the held phase (30–95% of the episode).",
              "", "| motion | tracking | bodies carrying > 10% of the load |", "|---|---|---|"]
    print("\n  what the observation says is carrying each pose:")
    for e in range(p.E):
        s = p.held(e)
        share = lf[s, e].mean(axis=0)
        top = [f"{p.bodies[i]} {share[i] * 100:.0f}%"
               for i in np.argsort(share)[::-1] if share[i] > 0.10]
        survived = ("tracked whole clip" if p.first_done[e] >= p.clip_end[e]
                    else f"lost ref at {p.first_done[e] * p.dt:.1f}s")
        lines.append(f"| {p.motions[e]} | {survived} | {', '.join(top) or '—'} |")
        print(f"    {p.motions[e][:46]:48s} {', '.join(top)}")

    (out / "checks.md").write_text("\n".join(lines) + "\n")

    def pick(fragment, default=0):
        for i, m in enumerate(p.motions):
            if fragment.lower() in m.lower():
                return i
        return default

    stand = pick("Standing_big_toe")
    crow = pick("Crane_Crow_Pose_or_Bakasana_-a")

    fig_tracking(p, figs)
    fig_encoding(p, figs)
    fig_newton(p, figs)
    fig_support_map(p, figs)
    fig_channels(p, figs, stand, ["L_Toe", "L_Ankle", "R_Toe", "R_Ankle", "L_Hand", "Head"],
                 "standing_big_toe")
    fig_channels(p, figs, crow, ["L_Hand", "R_Hand", "L_Knee", "R_Knee", "L_Toe", "Head"],
                 "crow")
    fig_health(p, figs)
    fig_proximity(p, figs, stand)
    fig_temporal(p, figs, stand)
    print(f"figures -> {figs}")
    print(f"checks   -> {out / 'checks.md'}")


if __name__ == "__main__":
    main()
