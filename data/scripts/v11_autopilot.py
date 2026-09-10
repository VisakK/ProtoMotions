#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the v11 programme unattended: A1, a health gate, then the long arm.

Written to be started once and left alone for ten days. It owns the GPU for the
whole window, so every branch ends in *something* training -- an idle GPU is the
one outcome with no information in it.

**Stage 1 -- A1 (~1.5 days, 6,000 epochs).** The action ladder at the encoder's
own future offsets ``{0, 5, 10, 15}``. Strictly single-variable against
``results/smpl_yogi_contact_graph_student_s2_v10_xycontract``, which is the same
recipe without the ladder and already has eight matched checkpoints on disk --
so A1 buys an interpretable mechanism read *and* a paired baseline for the price
of one arm.

**The gate.** Deliberately a *health* gate, not a direction gate, because the
direction is already decided by measurement: at h = 15 the linear state map
leaves 0.0167 of the target's variance for the code to explain, and at h = 24 it
leaves 0.0908 -- 5.5x more. Whatever A1 reads, the escalated horizon is the
better week-long bet. So the gate only asks whether the ladder *broke* anything:

* ``masked_mimic/mse`` within ``MSE_GUARD_FACTOR`` of v9's own value at the same
  epoch (v9's power law is a log-log slope of -0.655). Worse than that and the
  ladder is stealing rung-0 capacity, which is the one way this change can do
  real harm.
* ``model/fsq_code_perplexity`` above ``PERPLEXITY_FLOOR``. v6 settled at ~4.4
  and v10_1 at ~4.8 of 5; a slide toward 1.0 is intent collapse.

It also *records* the pre-registered mechanism read -- ``code_ablation_gap_h15``
against its ``code_ablation_gap_h0`` control -- because that is the number the
round exists to produce, even though it does not change what runs next.

**Stage 2 -- the long arm (~7 days).** On a healthy A1, ``a2``: rungs
``{0, 8, 16, 24}`` with the encoder window moved to match, i.e. the horizon with
the most available signal. On an unhealthy A1, ``control``: the ladder off, so
the week at least produces a properly matched v9 baseline on this corpus rather
than a second broken run.

Everything it decides is written to ``results/v11_autopilot/decision.json`` and
appended to ``results/v11_autopilot/log.md`` as it happens, so the record
survives the process.

Usage::

    PYTHONPATH=. nohup setsid python data/scripts/v11_autopilot.py \\
        > results/v11_autopilot/autopilot.log 2>&1 &

    PYTHONPATH=. python data/scripts/v11_autopilot.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE_DIR = REPO / "results" / "v11_autopilot"
DECISION = STATE_DIR / "decision.json"
LOGBOOK = STATE_DIR / "log.md"

LAUNCHER = REPO / "data" / "scripts" / "run_student_distill_v11.py"

# Measured on this arm at launch: 20.8 s/epoch, i.e. 5.77 GPU-hours per 1,000
# epochs (v10_1 ran 5.3; the ladder's wider trunk and its target shift cost
# ~9 %). Stage 1 is therefore ~34.6 h and stage 2 ~7.7 days, 9.2 days together,
# inside a ten-day window.
#
# Stage 1 length. Long enough to read the mechanism -- the v9 Tier-0 checkpoint
# sweep put the deployable optimum at ~8,000 -- and short enough to leave the
# week for the long arm.
STAGE1_EPOCHS = 6000

# Stage 2 length. A cap, not a deadline: checkpoints land every 500 epochs, so
# overrunning the window costs nothing and stopping early loses nothing either.
# The round has no expected result; the instruction is to spend the week and
# read what comes out.
STAGE2_EPOCHS = 32000

# Gate thresholds.
MSE_GUARD_FACTOR = 1.5  # vs v9 at the same epoch
PERPLEXITY_FLOOR = 2.0  # of 5; v6 settled ~4.4, v10_1 ~4.8
LADDER_GAP_PASS = 0.015  # pre-registered PASS on code_ablation_gap_h15
LADDER_GAP_NULL = 0.005  # pre-registered NULL

# The reference run the guard compares against, and the metric it reads.
V9_RUN = "smpl_yogi_contact_graph_student_s2_v9_smallcode_dagger"

POLL_SECONDS = 300


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def note(message: str) -> None:
    """Append to the logbook and to stdout, both timestamped."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"- `{stamp}` {message}"
    print(line, flush=True)
    with open(LOGBOOK, "a") as handle:
        handle.write(line + "\n")


def read_state() -> dict:
    if DECISION.is_file():
        return json.loads(DECISION.read_text())
    return {}


def write_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(state, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# Metric access
# --------------------------------------------------------------------------- #
def scalars(run: str) -> dict:
    """``tag -> [(step, value)]`` for a run's latest TensorBoard version."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    versions = sorted((REPO / "results" / run / "lightning_logs").glob("version_*"))
    if not versions:
        return {}
    accumulator = EventAccumulator(str(versions[-1]), size_guidance={"scalars": 200000})
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def last(series: dict, tag: str):
    points = series.get(tag)
    return points[-1] if points else None


def value_near(series: dict, tag: str, step: int):
    """The value of ``tag`` at the recorded step closest to ``step``."""
    points = series.get(tag)
    if not points:
        return None
    return min(points, key=lambda point: abs(point[0] - step))[1]


# --------------------------------------------------------------------------- #
# Running an arm
# --------------------------------------------------------------------------- #
def gpu_is_free() -> bool:
    return not subprocess.run(
        ["pgrep", "-f", "[t]rain_agent.py"], capture_output=True
    ).stdout.strip()


def run_arm(mode: str, epochs: int, tag: str) -> int:
    """Launch one arm in the foreground and wait for it."""
    log_path = STATE_DIR / f"{tag}.log"
    # `run_student_distill_v11.py` forwards unknown arguments straight through
    # to the trainer, and the epoch budget reaches it as an env-step budget:
    # num_envs * num_steps env-steps per epoch.
    command = [
        sys.executable,
        str(LAUNCHER),
        mode,
        "--log",
        str(log_path),
        "--training-max-steps",
        str(epochs * 1024 * 32),
    ]
    note(f"launching **{mode}** for ~{epochs} epochs -> `{log_path}`")
    environment = dict(os.environ)
    environment.setdefault("PYTHONPATH", str(REPO))
    started = time.time()
    code = subprocess.call(command, cwd=REPO, env=environment)
    hours = (time.time() - started) / 3600.0
    note(f"**{mode}** exited with code {code} after {hours:.1f} h")
    return code


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def evaluate_gate(run: str) -> dict:
    """Health of the A1 arm, plus the pre-registered mechanism read."""
    series = scalars(run)
    reference = scalars(V9_RUN)
    verdict = {"run": run, "healthy": True, "reasons": []}

    mse = last(series, "masked_mimic/mse")
    if mse is None:
        verdict["healthy"] = False
        verdict["reasons"].append("no masked_mimic/mse logged at all")
    else:
        step, value = mse
        verdict["mse"] = value
        verdict["mse_epoch"] = step
        baseline = value_near(reference, "masked_mimic/mse", step)
        verdict["mse_v9_at_same_epoch"] = baseline
        if baseline and value > MSE_GUARD_FACTOR * baseline:
            verdict["healthy"] = False
            verdict["reasons"].append(
                f"imitation MSE {value:.3e} is more than {MSE_GUARD_FACTOR}x "
                f"v9's {baseline:.3e} at epoch {step}: the ladder is stealing "
                "rung-0 capacity"
            )

    perplexity = last(series, "model/fsq_code_perplexity")
    if perplexity is not None:
        verdict["perplexity"] = perplexity[1]
        if perplexity[1] < PERPLEXITY_FLOOR:
            verdict["healthy"] = False
            verdict["reasons"].append(
                f"code perplexity {perplexity[1]:.2f} is below {PERPLEXITY_FLOOR}: "
                "the intent channel has collapsed"
            )

    # The mechanism read. Recorded, not acted on -- the escalated horizon is the
    # better bet at h = 24 whatever h = 15 says, because it has 5.5x the
    # available variance (0.0908 against 0.0167).
    gap = last(series, "model/code_ablation_gap_h15")
    control = last(series, "model/code_ablation_gap_h0")
    if gap is not None:
        verdict["code_ablation_gap_h15"] = gap[1]
        verdict["mechanism"] = (
            "PASS" if gap[1] >= LADDER_GAP_PASS
            else "NULL" if gap[1] <= LADDER_GAP_NULL
            else "INCONCLUSIVE"
        )
    if control is not None:
        verdict["code_ablation_gap_h0"] = control[1]
    return verdict


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--status", action="store_true", help="Print state and exit.")
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    args = parser.parse_args()

    state = read_state()
    if args.status:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    note("autopilot start")

    while not gpu_is_free():
        note("another train_agent.py holds the GPU; waiting")
        time.sleep(POLL_SECONDS)

    # ---- Stage 1 -------------------------------------------------------- #
    if not state.get("stage1_done"):
        code = run_arm("a1", args.stage1_epochs, "stage1_a1")
        state["stage1_exit"] = code
        state["stage1_done"] = True
        state["stage1_run"] = "smpl_yogi_contact_graph_student_s2_v11_ladder"
        write_state(state)

    # ---- Gate ------------------------------------------------------------ #
    if "gate" not in state:
        gate = evaluate_gate(state["stage1_run"])
        state["gate"] = gate
        write_state(state)
        note(
            "gate: healthy=%s mechanism=%s gap_h15=%s gap_h0=%s mse=%s"
            % (
                gate.get("healthy"),
                gate.get("mechanism"),
                gate.get("code_ablation_gap_h15"),
                gate.get("code_ablation_gap_h0"),
                gate.get("mse"),
            )
        )
        for reason in gate.get("reasons", []):
            note(f"  gate reason: {reason}")

    # ---- Stage 2 --------------------------------------------------------- #
    if not state.get("stage2_done"):
        healthy = bool(state["gate"].get("healthy", False))
        mode = "a2" if healthy else "control"
        note(
            f"stage 2 = **{mode}** "
            + (
                "(A1 healthy: escalate the horizon to h=24, where 0.0908 of the "
                "target's variance is available against 0.0167 at h=15)"
                if healthy
                else "(A1 unhealthy: spend the week on a matched v9 baseline for "
                "this corpus rather than a second broken arm)"
            )
        )
        state["stage2_mode"] = mode
        write_state(state)
        code = run_arm(mode, args.stage2_epochs, f"stage2_{mode}")
        state["stage2_exit"] = code
        state["stage2_done"] = True
        write_state(state)

    note("autopilot done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
