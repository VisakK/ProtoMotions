#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One command that says where the v11 programme stands and what it means.

Run this first on getting back. It prints, in order: what the autopilot is
doing, the mechanism read, the health guard, and the paired panel comparison
against the free ``v10_xycontract`` baseline -- scored the way the protocol
work says it should be, which is per goal and averaged over checkpoints rather
than per sequence at one.

**Why per goal and averaged.** Measured on the v10 / v10_1 panels already on
disk, the minimum detectable effect at 80 % power is 0.174 per-sequence at one
checkpoint -- which is what every published comparison in this project has used
-- against **0.022** per-goal averaged over eight. 79.4 % of the estimator's
variance is checkpoint-level, so K is the lever; replica sampling is only 9 %
and more probes barely move it.

**And continuous rather than thresholded.** On identical rollouts the
positive control (paired epoch 500 -> 4,000 within one run) reads t = +2.10 as
a per-goal binary rate and **t = -4.49** as a per-goal continuous hold-window
pose error. The arrays for both are already in ``per_goal_agg``.

Four probes are quarantined by default and reported separately, because their
commanded frame is inside the training corpus as a frozen hold clip
(``hold_probe_standing`` and friends), and four more command Eagle Pose under
the goal name "stand_up" because the plan generator falls back to the longest
node-0 segment.

Usage::

    PYTHONPATH=. python data/scripts/v11_readout.py
    PYTHONPATH=. python data/scripts/v11_readout.py --arm smpl_yogi_..._v11_ladder_h24
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

BASELINE = "smpl_yogi_contact_graph_student_s2_v10_xycontract"
DEFAULT_ARM = "smpl_yogi_contact_graph_student_s2_v11_ladder"

# Commanded frames that v10_1 added to the corpus as frozen hold clips, so any
# arm trained on a corpus containing them is scored on its own training data.
LEAKED = {
    "hold_probe_standing",
    "standing_downdog",
    "seq_downdog_plank",
    "seq_warrior2",
}
# `make_graph_probe_plans.standing_exemplar` falls back to the longest node-0
# segment, which is Eagle Pose -a at 12.8 s -- a twisted single-leg pose
# commanded under the goal name "stand_up".
EAGLE_ARTIFACTS = {
    "trip_lf_pron",
    "trip_lf_lh_rf_rh_upri",
    "walk_lf_lh_rh_pron__lh_rf_rh_pron",
    "walk_lf_rf_rh_upri__lf_rf_rh_pron",
}


def per_goal(run: str) -> dict:
    """``(sequence, goal_index) -> {epoch: continuous hold-window pose error}``."""
    out: dict = {}
    for directory in sorted(glob.glob(str(REPO / "results" / run / "viz" / "epoch_*"))):
        epoch = int(os.path.basename(directory).split("_")[1])
        summary = Path(directory) / "summary.json"
        if not summary.is_file():
            continue
        for sequence in json.loads(summary.read_text()):
            for index, goal in enumerate(sequence.get("per_goal_agg", [])):
                value = goal.get("hold_pose_err_p50")
                if value is not None:
                    out.setdefault((sequence["sequence"], index), {})[epoch] = value
    return out


def paired(arm: dict, base: dict, keep) -> tuple:
    """Paired mean difference over goals, each averaged across its checkpoints."""
    deltas = []
    for key in sorted(set(arm) & set(base)):
        if not keep(key[0]):
            continue
        a = list(arm[key].values())
        b = list(base[key].values())
        if a and b:
            deltas.append(statistics.mean(a) - statistics.mean(b))
    if len(deltas) < 3:
        return None
    mean = statistics.mean(deltas)
    error = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean, error, mean / error if error else float("nan"), len(deltas)


def report(title: str, result) -> None:
    if result is None:
        print(f"  {title:<34} not enough matched goals yet")
        return
    mean, error, t, n = result
    verdict = "BETTER" if t < -2 else "worse" if t > 2 else "no difference"
    # Lower pose error is better, hence the sign convention.
    print(
        f"  {title:<34} {mean:+.4f} m  se {error:.4f}  t {t:+.2f}  "
        f"n={n:<4} {verdict}   (MDE ~{2.8 * error:.3f} m)"
    )


def scalars(run: str) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    versions = sorted((REPO / "results" / run / "lightning_logs").glob("version_*"))
    if not versions:
        return {}
    accumulator = EventAccumulator(str(versions[-1]), size_guidance={"scalars": 200000})
    accumulator.Reload()
    return {
        tag: accumulator.Scalars(tag)[-1] for tag in accumulator.Tags()["scalars"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()

    print("=" * 78)
    print("AUTOPILOT")
    print("=" * 78)
    decision = REPO / "results" / "v11_autopilot" / "decision.json"
    if decision.is_file():
        print(decision.read_text())
    else:
        print("  no decision.json yet -- stage 1 still running")
    running = subprocess.run(
        ["pgrep", "-af", "train_agent.py"], capture_output=True, text=True
    ).stdout.strip()
    for line in running.splitlines():
        name = [p for p in line.split() if "experiment-name" not in p]
        print("  training:", line.split("--experiment-name")[-1].split()[0]
              if "--experiment-name" in line else name[:1])
    if not running:
        print("  nothing is training right now")

    print()
    print("=" * 78)
    print(f"MECHANISM AND HEALTH -- {args.arm}")
    print("=" * 78)
    series = scalars(args.arm)
    if not series:
        print("  no TensorBoard scalars yet")
    else:
        for tag, label, note in [
            ("model/code_ablation_gap_h15", "code_ablation_gap_h15",
             "PASS >= 0.015, NULL <= 0.005 -- the pre-registered mechanism read"),
            ("model/code_ablation_gap_h24", "code_ablation_gap_h24",
             "the A2 rung; ceiling 0.08-0.10"),
            ("model/code_ablation_gap_h0", "code_ablation_gap_h0",
             "the matched control; ceiling 2.03e-4, so near-zero is CORRECT"),
            ("masked_mimic/mse", "imitation MSE",
             "guard: within ~1.5x of v9 at the same epoch"),
            ("model/fsq_code_perplexity", "code perplexity",
             "guard: v6 ~4.4 of 5. BUT a 4-scalar code has a ~22-epoch flat "
             "start where this reads exactly 1.0 and the CE ramp has not "
             "begun (ce_start_epoch=100) -- compare the WAKE-UP EPOCH, not "
             "the value. A slide toward 1.0 after epoch ~1000 is collapse."),
            ("ladder/valid_frac_h15", "valid_frac_h15",
             "~0.52 expected once episodes lengthen. Early in training it is "
             "far lower because an untrained policy terminates constantly and "
             "an episode boundary inside the horizon drops the row."),
        ]:
            point = series.get(tag)
            if point is not None:
                print(f"  {label:<24} {point.value:>12.6g}   @epoch {point.step}")
                print(f"  {'':<24} {note}")

    print()
    print("=" * 78)
    print(f"PAIRED PANEL -- {args.arm} vs {args.baseline}")
    print("  per goal, averaged over each goal's checkpoints, continuous error.")
    print("  Negative is better: it is a pose error, in metres.")
    print("=" * 78)
    arm, base = per_goal(args.arm), per_goal(args.baseline)
    if not arm:
        print("  no panels for the arm yet (they land every 500 epochs)")
        return 0
    report("all goals", paired(arm, base, lambda s: True))
    report(
        "clean goals only",
        paired(arm, base, lambda s: s not in LEAKED and s not in EAGLE_ARTIFACTS),
    )
    report("quarantined: train-on-test", paired(arm, base, lambda s: s in LEAKED))
    report("quarantined: Eagle artifact", paired(arm, base, lambda s: s in EAGLE_ARTIFACTS))
    print()
    print("  Read 'clean goals only'. The other two rows exist so the leakage is")
    print("  visible rather than averaged in: v10_1's +0.89 on hold_probe_standing")
    print("  was a frozen hold clip cut at that probe's own (clip, time).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
