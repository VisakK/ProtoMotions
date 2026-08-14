# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crow-pair Stage-1 expert + PAIR-AWARE contact reward (Stage A of
``notes/Contact_balance_reward_design.MD``).

Identical to :mod:`mlp_contact_rich_rew` (same network, tracking rewards,
terminations, contact obs, per-body contact-match at -1.0) plus two new terms
from :mod:`pair_contact_terms`:

* ``pair_contact_rew`` (+0.3, bounded [0,0.3]) — encourage the annotated
  body-against-THAT-body pairs, near-tangent and preferably loaded. Targets are
  per-clip, side-specific, time-indexed masks in the reftargets table.
* ``pair_forbid_rew`` (-0.1, clamped at -0.3) — penalize the force-verified
  invented load path (side crow L_Hip<->R_Hip), load-gated so unloaded
  proximity is never punished. Deliberately enters at -0.1 because the pair is
  a genuine load path in the only achieved lift-off today.

  **Escalating the forbid weight to -0.3: use a WARM START, never a resume.**
  A resume (relaunching with the same ``--experiment-name``) loads the frozen
  ``resolved_configs.pt`` and does NOT re-read this module or ``--overrides``
  — an edited ``PAIR_FORBID_WEIGHT`` would be silently ignored. Instead: edit
  the constant, then launch with a NEW ``--experiment-name`` and
  ``--checkpoint results/<previous>/last.ckpt``.

Plus weight-0 gap diagnostics so wandb shows the two decisive gaps per epoch:
``env/raw_r/diag_gap_L_THIGH_TRUNK_mean`` (target: toward ~0.015 m) and
``env/raw_r/diag_gap_L_THIGH_R_THIGH_mean`` (side crow; should NOT shrink
below ~0.03 m).

MUST be trained with the label-REPAIRED motion package (design §2.4) — the v2
labels pay for the trailing foot the pair terms are trying to lift::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_pair_contact_rew.py \
      --experiment-name smpl_yogi_crow_pair_pair_rew_s1 \
      --motion-file data/smpl/yoga_yogi_crow_pair_v3_contacts.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb \
      --overrides env.ref_respawn_offset=0.005 \
      --checkpoint results/smpl_yogi_crow_pair_contact_rich_rew/last.ckpt

Before the first training run, validate the kernels offline against the
reference clips and the recorded rollouts::

    python data/scripts/validate_pair_reward.py
"""

from __future__ import annotations

import argparse
import os

from examples.experiments.mimic import mlp_contact_rich_rew as base
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
apply_inference_overrides = base.apply_inference_overrides
agent_config = base.agent_config

REFTARGETS_PT = "data/smpl/yoga_yogi_crow_pair_reftargets.pt"

PAIR_ENCOURAGE_WEIGHT = 0.3
# Ramp target -0.3; starts low because side crow currently LOAD-BEARS on the
# forbidden thigh-thigh pair (743 N peak / 37% duty) — yanking a support at
# full price destabilizes the only achieved lift-off. Escalate at resume.
PAIR_FORBID_WEIGHT = -0.1


def _assert_reftargets_match_motion_file(motion_file) -> None:
    """The design's §2.3 loud version tie: the reftargets table encodes
    per-frame gates for SPECIFIC clips in a SPECIFIC order. A mismatched
    --motion-file must refuse to start, not silently mis-gate every term
    (v2 vs v3 packages have identical shapes) or die later in a cryptic
    device-side gather assert (>2-motion packages)."""
    import torch

    if not motion_file or not os.path.isfile(str(motion_file)):
        return  # motion lib will raise its own error for a missing file
    rt = torch.load(REFTARGETS_PT, map_location="cpu", weights_only=False)
    mp = torch.load(str(motion_file), map_location="cpu", weights_only=False)
    mp_names = [
        os.path.basename(str(f)).split(".")[0] for f in mp["motion_files"]
    ]
    mp_frames = [int(n) for n in mp["motion_num_frames"]]
    rt_names = list(rt["motion_names"])
    rt_frames = [int(n) for n in rt["num_frames"]]
    assert mp_names == rt_names and mp_frames == rt_frames, (
        f"reftargets/motion-file contract mismatch: {REFTARGETS_PT} was built "
        f"for {rt['motion_pt']} (clips {rt_names}, frames {rt_frames}) but "
        f"--motion-file {motion_file} contains clips {mp_names}, frames "
        f"{mp_frames}. Expected: data/smpl/yoga_yogi_crow_pair_v3_contacts.pt "
        "(the label-REPAIRED package). Rebuild the reftargets with "
        "data/scripts/package_pair_targets.py if the motion package changed."
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from examples.experiments.mimic.pair_contact_terms import (
        ground_fz_diag_factory,
        pair_encourage_rew_factory,
        pair_forbid_rew_factory,
        pair_gap_diag_factory,
    )

    assert os.path.isfile(REFTARGETS_PT), (
        f"{REFTARGETS_PT} missing — build it with "
        "data/scripts/package_pair_targets.py before training"
    )
    _assert_reftargets_match_motion_file(getattr(args, "motion_file", None))

    cfg = base.env_config(robot_cfg, args)
    cfg.reward_components["pair_contact_rew"] = pair_encourage_rew_factory(
        REFTARGETS_PT, weight=PAIR_ENCOURAGE_WEIGHT
    )
    cfg.reward_components["pair_forbid_rew"] = pair_forbid_rew_factory(
        REFTARGETS_PT, weight=PAIR_FORBID_WEIGHT, min_value=-0.3
    )
    # Free wandb diagnostics (weight 0 = computed + logged, not summed).
    cfg.reward_components["diag_gap_L_THIGH_TRUNK"] = pair_gap_diag_factory(
        REFTARGETS_PT, "L_THIGH+TRUNK"
    )
    cfg.reward_components["diag_gap_L_THIGH_R_THIGH"] = pair_gap_diag_factory(
        REFTARGETS_PT, "L_THIGH+R_THIGH"
    )
    # S2 health check: ~1.0 in any ground-supported steady state; 0 = the
    # filtered ground-force buffer silently died (design §2.2 failure mode).
    cfg.reward_components["diag_ground_fz"] = ground_fz_diag_factory()
    return cfg
