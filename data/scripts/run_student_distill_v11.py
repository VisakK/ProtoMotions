#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the v11 action-ladder student, or a smoke test of it.

**The arm.** v11-A1 is ``v10_xycontract`` with one thing added: the action
ladder. Same corpus (``student44h``), same graph, same experts, same FSQ
settings, same DAgger mixture, same XY and re-anchor flags. That matters
because ``results/smpl_yogi_contact_graph_student_s2_v10_xycontract`` already
holds **8 matched checkpoints** of exactly that recipe without the ladder, so
this arm gets a paired baseline for free rather than costing a 1.5-GPU-day
control run.

Deliberately NOT carried over from v10_1: ``--dwell-channels``,
``--include-current-segment`` and ``--far-goal-prob``. Measured on the shipped
graph, ``include_current_segment`` makes the slot-0 contact set equal the
currently-held contact set on 100.0 % of in-segment frames (from 66.9 %) and
drops the "unload a limb" command from 17.5 % to 0.0 %; far-goal promotion
touches slot 0 on 0.19 % of frames. v11 branches from v9, which never had them.

**Modes.**

``smoke``   ~40 epochs on 64 envs. Proves the ladder wires end to end: the
            trunk widens, the targets build, the loss is finite, the ablation
            diagnostic emits. Minutes, and it is the abort gate.
``a1``      The real arm: rungs {0, 5, 10, 15}, the encoder's own offsets, so
            no observation width moves and the arm is strictly single-variable.
``a2``      The follow-up: rungs {0, 8, 16, 24} with the encoder window moved
            to match. Only worth running once A1 clears its mechanism gate.
``control`` The ladder off (``--ladder-loss-coeff 0``), i.e. v9 exactly. Only
            needed if the free v10_xycontract baseline is ever in doubt.

**What to read.** ``model/code_ablation_gap_h15`` is the pre-registered
mechanism metric: the rise in rung 15's normalised MSE when ``vae_latent`` is
rolled across the batch. PASS >= 0.015 by epoch 4,000; NULL <= 0.005. Its
matched control is ``model/code_ablation_gap_h0``, whose ceiling is 2.03e-4
because the h=0 target is state-linear to R^2 = 0.999797 -- a large far-rung
gap against a near-zero rung-0 gap is the signature that the ladder engaged
and nothing else did.

Also watch ``ladder/valid_frac_h*``. A rung's target lives ``h`` steps further
along the buffer's time axis, and the rollout window is ``num_steps = 32``, so
rows within ``h`` of the window's end have no target at all: h = 15 structurally
loses ~47 % of rows and h = 24 loses ~75 %, on top of the ~2 % lost to episode
boundaries. That is a uniform subsample rather than a biased one -- a 32-step
window starts at an arbitrary phase of a ~321-step episode -- and h = 15 still
supervises ~17 k rows an epoch, but it is the reason each rung is normalised by
its own target variance instead of pooling them.

Guard: ``masked_mimic/mse`` must stay within ~1.5x of v9's power law (log-log
slope -0.655) or the ladder is stealing rung-0 capacity. Pre-registered as
allowed to worsen: ``model/fsq_full_match`` and ``model/fsq_ce_loss_refresh``.

Ignore ``eval/success_rate``. It measures clip tracking on the greedy stream,
and the pose half of the goal is the played clip's own future frame.

Usage::

    PYTHONPATH=. python data/scripts/run_student_distill_v11.py smoke
    PYTHONPATH=. python data/scripts/run_student_distill_v11.py a1
    PYTHONPATH=. python data/scripts/run_student_distill_v11.py a1 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLANS = REPO / "data" / "scripts" / "plans"

# The 28 panel plans, in the order every previous round ran them, so the panel
# rows line up with the v10 / v10_1 summaries already on disk.
VIZ_PLANS = [
    "v6_probe/chain_handstand_pose_or_adho_muk_a.json",
    "v6_probe/chain_scorpion_pose_or_vrischika_b.json",
    "v6_probe/chain_firefly_pose_or_tittibhasa_a.json",
    "v6_probe/chain_pose_dedicated_to_the_sage_b.json",
    "v6_probe/chain_supported_headstand_pose_o_b.json",
    "v6_probe/chain_supported_shoulderstand_po_a.json",
    "handstand_chain.json",
    "handstand_from_prone.json",
    "seq_warrior2.json",
    "seq_warrior3_standing.json",
    "seq_warrior3_halfmoon.json",
    "handstand_far_goal_loadpath.json",
    "v6_probe/chain_side_plank_pose_or_vasisth_c.json",
    "v6_probe/chain_cobra_pose_or_bhujangasana_c.json",
    "seq_downdog_plank.json",
    "seq_plank_sideplank.json",
    "standing_downdog.json",
    "hold_probe_standing.json",
    "hold_probe_plank.json",
    "hold_probe_sideplank.json",
    "hold_probe_dolphinplank.json",
    "v6_probe/trip_lf_upri.json",
    "v6_probe/trip_rf_upri.json",
    "v6_probe/trip_lf_pron.json",
    "v6_probe/trip_lf_lh_rf_rh_pron.json",
    "v6_probe/trip_lf_lh_rf_rh_upri.json",
    "v6_probe/walk_lf_lh_rh_pron__lh_rf_rh_pron.json",
    "v6_probe/walk_lf_rf_rh_upri__lf_rf_rh_pron.json",
]

# `num_steps` in the agent config; only used to convert an epoch budget into
# the env-step budget `--training-max-steps` takes.
NUM_ROLLOUT_STEPS = 32

# The v10_xycontract corpus and graph -- NOT v10_1's v101 rebuild. Three of
# v101's added frozen holds are cut at exactly four panel plans' commanded
# (clip, time), which makes those probes train-on-test.
MOTIONS = "data/smpl/yoga_yogi_student44h.pt"
EXPERT_MAP = "data/smpl/yoga_yogi_student44h.experts.json"
GRAPH = "data/smpl/yoga_contact_graph_student44h/contact_graph.pt"

EXPERTS = [
    "results/smpl_yogi_easy128_contact_rich/last.ckpt",
    "results/smpl_yogi_hard29_pressure_ab_s1/last.ckpt",
    "results/smpl_yogi_singleleg14_pressure_ab_s1/last.ckpt",
]

MODES = {
    "smoke": dict(
        experiment="smpl_yogi_v11_ladder_smoke",
        ladder_offsets=[0, 5, 10, 15],
        ladder_coeff=1.0,
        encoder_future=None,
        num_envs=64,
        batch_size=512,
        max_epochs=40,
        viz_every=0,
        wandb=False,
    ),
    "a1": dict(
        experiment="smpl_yogi_contact_graph_student_s2_v11_ladder",
        ladder_offsets=[0, 5, 10, 15],
        ladder_coeff=1.0,
        encoder_future=None,
        num_envs=1024,
        batch_size=8192,
        max_epochs=None,
        viz_every=500,
        wandb=True,
    ),
    "a2": dict(
        experiment="smpl_yogi_contact_graph_student_s2_v11_ladder_h24",
        ladder_offsets=[0, 8, 16, 24],
        ladder_coeff=1.0,
        encoder_future=[1, 8, 16, 24],
        num_envs=1024,
        batch_size=8192,
        max_epochs=None,
        viz_every=500,
        wandb=True,
    ),
    "control": dict(
        experiment="smpl_yogi_contact_graph_student_s2_v11_control",
        ladder_offsets=[0],
        ladder_coeff=0.0,
        encoder_future=None,
        num_envs=1024,
        batch_size=8192,
        max_epochs=None,
        viz_every=500,
        wandb=True,
    ),
}


def build_command(mode: str, extra: list) -> list:
    cfg = MODES[mode]
    command = [
        sys.executable,
        "protomotions/train_agent.py",
        "--robot-name", "smpl_yogi",
        "--simulator", "isaaclab",
        "--experiment-path", "examples/experiments/masked_mimic/contact_graph_fsq_v11.py",
        "--experiment-name", cfg["experiment"],
        "--motion-file", MOTIONS,
        "--contact-graph-file", GRAPH,
        "--motion-expert-file", EXPERT_MAP,
        "--expert-model-paths", *EXPERTS,
        "--segment-start-prob", "0.6",
        "--sense-body-pair-contacts", "True",
        # v9's FSQ settings, unchanged: one token over a 625-word vocabulary.
        "--fsq-levels", "5",
        "--fsq-scalars", "4",
        "--fsq-scalars-per-token", "4",
        "--chunk-steps", "8",
        "--event-refresh", "True",
        "--ce-start-epoch", "100",
        "--ce-end-epoch", "600",
        "--fsq-temperature", "1.0",
        "--fsq-top-p", "0.9",
        "--dagger-action-loss-coeff", "0.1",
        "--student-root-relative-xy", "True",
        # The one change.
        "--ladder-offsets", *[str(o) for o in cfg["ladder_offsets"]],
        "--ladder-loss-coeff", str(cfg["ladder_coeff"]),
        "--num-envs", str(cfg["num_envs"]),
        "--batch-size", str(cfg["batch_size"]),
        "--headless", "True",
    ]
    if cfg["encoder_future"] is not None:
        command += ["--encoder-future-steps", *[str(s) for s in cfg["encoder_future"]]]
    if cfg["viz_every"]:
        command += [
            "--viz-sequences-every", str(cfg["viz_every"]),
            "--viz-num-sequences", str(len(VIZ_PLANS)),
            "--viz-max-seconds", "24.0",
            "--viz-log-scalars", "False",
            "--viz-plan-files", *[str(PLANS / p) for p in VIZ_PLANS],
        ]
    if cfg["wandb"]:
        command += ["--use-wandb"]
    overrides = [
        "env.ref_respawn_offset=0.005",
        # Matches v10_xycontract exactly, which is what makes the baseline free.
        "env.motion_manager.realign_motion_with_humanoid_on_each_step=True",
    ]
    command += ["--overrides", *overrides]
    if cfg["max_epochs"] is not None:
        # `training_max_steps` is in env-steps; the agent divides by
        # num_envs * num_steps to get epochs (base_agent/agent.py:122).
        command += [
            "--training-max-steps",
            str(cfg["max_epochs"] * cfg["num_envs"] * NUM_ROLLOUT_STEPS),
        ]
    command += extra
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command and stop."
    )
    parser.add_argument(
        "--log", default=None, help="Tee stdout/stderr to this file as well."
    )
    args, extra = parser.parse_known_args()

    command = build_command(args.mode, extra)
    print(" \\\n  ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "wb") as handle:
            process = subprocess.Popen(
                command, cwd=REPO, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            for line in process.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
            return process.wait()
    return subprocess.call(command, cwd=REPO, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
