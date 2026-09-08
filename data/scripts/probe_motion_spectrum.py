# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does the FSQ chunk clock show up as a line in the motion's spectrum?

The v6 student holds one intent code for ``chunk_steps`` control steps and then
issues a new one. At 30 Hz with ``chunk_steps=8`` that is a **3.75 Hz** step
change in a trunk input, and with the measured code-space disagreement
(perplexity 4.8/5, full-code match 9.5 %) consecutive codes differ
substantially. Nothing in the supervised loss penalises the resulting
discontinuity (``calculate_extra_loss`` returns 0), so the hypothesis in
``notes/Student_v7_improvement_investigation.MD`` §5.7 is that a held pose
carries a periodic kick at the chunk rate.

This tests it from data already on disk. A probe's ``.motion`` stores ``dof_vel``
at the control rate, so the joint **acceleration** is one finite difference away,
and its power spectrum is directly comparable between a v5 render (continuous
per-step latent, no chunk clock) and a v6 one (chunked). A chunk artifact shows
up as a narrow line at ``fps / chunk_steps`` standing above the neighbouring
band; a policy that is merely noisy shows a broad spectrum with no such line.

Reported per file: the band-limited power around the chunk frequency and its
first harmonic, each as a ratio to the local background (the median of the
surrounding bins). **A ratio near 1 means no line.** Read the ratio, not the
absolute power -- the two runs differ in overall activity.

Usage::

    PYTHONPATH=. python data/scripts/probe_motion_spectrum.py \\
      --motion output/renderings/v5_probe/v5last_hold_plank.motion \\
      --motion output/renderings/v6_tier0/sampled/sampled_hold_probe_plank.motion \\
      --window 2 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def line_ratio(freqs, power, f0, half_width_hz=0.35, background_hz=2.0):
    """Power in a narrow band at f0 over the median power of its surroundings."""
    band = np.abs(freqs - f0) <= half_width_hz
    around = (np.abs(freqs - f0) > half_width_hz) & (np.abs(freqs - f0) <= background_hz)
    if not band.any() or not around.any():
        return float("nan"), float("nan")
    peak = float(power[band].max())
    background = float(np.median(power[around]))
    return peak / background if background > 0 else float("inf"), peak


def analyse(path: str, args):
    rec = torch.load(path, map_location="cpu", weights_only=False)
    fps = float(rec["fps"])
    vel = rec["dof_vel"].numpy()
    lo = int(round(args.window[0] * fps))
    hi = min(int(round(args.window[1] * fps)), vel.shape[0])
    seg = vel[lo:hi]
    if seg.shape[0] < 32:
        print(f"  {Path(path).name}: window too short ({seg.shape[0]} frames)")
        return
    accel = np.diff(seg, axis=0) * fps
    accel = accel - accel.mean(axis=0, keepdims=True)
    accel *= np.hanning(accel.shape[0])[:, None]
    spectrum = np.abs(np.fft.rfft(accel, axis=0)) ** 2
    power = spectrum.mean(axis=1)                     # mean over the 69 dofs
    freqs = np.fft.rfftfreq(accel.shape[0], d=1.0 / fps)

    chunk_f = fps / args.chunk_steps
    r1, _ = line_ratio(freqs, power, chunk_f)
    r2, _ = line_ratio(freqs, power, 2 * chunk_f)
    rms = float(np.sqrt((accel ** 2).mean()))
    # The band above `high_hz` is what a closed loop cannot absorb and what the
    # supervised objective does not oppose; report it as a share of total power
    # so it is comparable between runs of very different overall activity.
    high = float(power[freqs >= args.high_hz].sum() / power.sum())
    peak_f = float(freqs[int(np.argmax(power))])
    print(f"  {Path(path).name}")
    print(f"      window {args.window[0]}-{args.window[1]}s, {seg.shape[0]} frames @ {fps:.0f} Hz")
    print(f"      joint-accel RMS {rms:7.2f} rad/s^2   |   "
          f"power above {args.high_hz:.0f} Hz: {high * 100:5.1f} %   |   peak bin {peak_f:5.2f} Hz")
    # The line ratio is only meaningful when there IS a signal: on a very quiet
    # trace the background bins are ~0 and any bin towers over them.
    if rms < args.quiet_rms:
        print(f"      chunk-line test skipped: trace too quiet (RMS < {args.quiet_rms}) "
              "-- ratios against a ~zero background are meaningless")
    else:
        print(f"      line at {chunk_f:.2f} Hz (chunk clock): {r1:5.2f}x background   "
              f"| {2 * chunk_f:.2f} Hz (2nd harmonic): {r2:5.2f}x")
    if args.dump_bins:
        order = np.argsort(power)[::-1][:6]
        peaks = ", ".join(f"{freqs[i]:.2f}" for i in sorted(order, key=lambda i: -power[i]))
        print(f"      six strongest bins (Hz): {peaks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=str, action="append", required=True,
                        help="a probe's recorded .motion; repeatable")
    parser.add_argument("--window", type=float, nargs=2, default=[2.0, 12.0],
                        help="seconds of the episode to analyse (a held pose)")
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--high-hz", type=float, default=8.0,
                        help="the 'jerk band' floor; power above this as a share "
                             "of total is the run-to-run comparable number")
    parser.add_argument("--quiet-rms", type=float, default=2.0,
                        help="below this joint-accel RMS the chunk-line ratio is "
                             "suppressed as meaningless")
    parser.add_argument("--dump-bins", action="store_true")
    args = parser.parse_args()
    print(f"chunk frequency = fps / {args.chunk_steps}\n")
    for path in args.motion:
        analyse(path, args)
        print()


if __name__ == "__main__":
    main()
