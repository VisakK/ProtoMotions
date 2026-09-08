# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When does a commanded pose stop being held?

``run_sequence_panel.py --dump-traces`` writes ``pose_error_traces.npz``: the
goal-pose error of every replica of every plan, at every rendered frame.  The
panel's scalars compress that to a mean over the hold window, which answers
"did it hold" but not "for how long, and when did it leave" — and on v9 the
distinction is the whole finding, because **arrival is never the failure**
(every probe reaches its pose, best error ~0.03 m; the ones that fail arrive
and then depart).

Prints, per sequence, the median error against time and the survival curve
(fraction of replicas still within ``--arrive`` metres, having once been
there), and optionally writes a figure.

Usage::

    PYTHONPATH=. python data/scripts/plot_probe_survival.py \\
      --panel output/v9_tier0/hyst05 \\
      --sequences hold_probe_standing hold_probe_plank \\
                  hold_probe_sideplank hold_probe_dolphinplank \\
      --figure notes/figures/v9_hold_survival.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(panel: Path):
    data = np.load(panel / "pose_error_traces.npz", allow_pickle=False)
    names = [str(n) for n in data["sequences"]]
    summary = {
        r["sequence"]: r
        for r in json.loads((panel / "summary.json").read_text())
    }
    return data["pose_errors"], data["frame_times"], names, summary


def survival(errors: np.ndarray, times: np.ndarray, arrive: float,
             depart: float):
    """``(never_left, currently_at)`` fractions per frame.

    ``never_left`` is absorbing: once a replica exceeds ``depart`` it is gone
    for good. ``currently_at`` is instantaneous. **The gap between them is the
    measurement that matters**, because several probes leave the commanded pose
    and come back — on v9's standing hold the median error goes 0.03 m → 0.8 m
    by t = 2 s → back to 0.05 m by t = 10 s, which an absorbing curve reports as
    a flat 3 % and a longest-contiguous-run statistic reports as "held 4.2 s".
    Neither is wrong; both are unreadable without the other.

    Departure needs its own, looser threshold: without the hysteresis a replica
    sitting exactly at ``arrive`` flickers in and out and the curve measures
    noise instead of behaviour.
    """
    reps = errors.shape[1]
    alive = np.zeros(len(times))
    at = np.zeros(len(times))
    for r in range(reps):
        state = 0  # 0 = not yet arrived, 1 = holding, 2 = departed
        for f in range(len(times)):
            e = errors[f, r]
            if np.isnan(e):
                continue
            if e <= arrive:
                at[f] += 1
            if state == 0 and e <= arrive:
                state = 1
            elif state == 1 and e > depart:
                state = 2
            if state == 1:
                alive[f] += 1
    return alive / max(reps, 1), at / max(reps, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True,
                        help="a run_sequence_panel.py output directory")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--arrive", type=float, default=0.15)
    parser.add_argument("--depart", type=float, default=0.30)
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()

    panel = Path(args.panel)
    errors, times, names, summary = load(panel)
    num_seq = len(names)
    reps = errors.shape[1] // num_seq

    curves = {}
    for sequence in args.sequences:
        if sequence not in names:
            print(f"  (no trace for {sequence})")
            continue
        index = names.index(sequence)
        columns = np.arange(index, errors.shape[1], num_seq)
        block = errors[:, columns]
        median = np.nanmedian(block, axis=1)
        alive, at = survival(block, times, args.arrive, args.depart)
        curves[sequence] = (median, alive, at)
        half = np.argmax(alive < 0.5) if (alive < 0.5).any() else None
        print(f"\n{sequence}  ({block.shape[1]} replicas)")
        print(f"  median goal-pose error at t = "
              + "  ".join(
                  f"{t:.0f}s:{median[np.argmin(abs(times - t))]:.2f}"
                  for t in (0, 2, 4, 6, 8, 10, 12)
                  if t <= times[-1]
              ))
        print(f"  fraction never having left  at t = "
              + "  ".join(
                  f"{t:.0f}s:{alive[np.argmin(abs(times - t))]:.2f}"
                  for t in (0, 2, 4, 6, 8, 10, 12)
                  if t <= times[-1]
              ))
        print(f"  fraction AT the pose        at t = "
              + "  ".join(
                  f"{t:.0f}s:{at[np.argmin(abs(times - t))]:.2f}"
                  for t in (0, 2, 4, 6, 8, 10, 12)
                  if t <= times[-1]
              ))
        returned = at[-1] - alive[-1]
        if returned > 0.15:
            print(f"  ** {returned:.0%} of replicas LEFT the pose and came back **")
        if half is not None and alive.max() >= 0.5:
            print(f"  half the replicas have left by t = {times[half]:.2f} s")
        else:
            print("  more than half are still holding at the end of the window")

    if args.figure and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for sequence, (median, alive, at) in curves.items():
            label = sequence.replace("hold_probe_", "")
            line, = axes[0].plot(times, median, label=label)
            axes[1].plot(times, at, color=line.get_color(), label=label)
            axes[1].plot(times, alive, color=line.get_color(), ls=":", lw=1)
        axes[0].axhline(args.arrive, color="k", lw=0.6, ls=":")
        axes[0].set_xlabel("plan time (s)")
        axes[0].set_ylabel("median goal-pose error (m)")
        axes[0].set_title("distance to the commanded pose")
        axes[1].set_xlabel("plan time (s)")
        axes[1].set_ylabel("fraction of replicas at the pose")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title(f"solid: currently at the pose; dotted: never left\n"
                          f"({reps} replicas, arrive<{args.arrive} m, "
                          f"depart>{args.depart} m)")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=130)
        print(f"\nwrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
