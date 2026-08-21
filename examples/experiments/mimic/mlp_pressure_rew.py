# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-rich Stage-1 expert + weak supervision from the measured MOYO
pressure mat (options A and B of ``notes/Pressure_supervision_design.MD``).

Identical to :mod:`mlp_contact_rich` — same 6x2048 actor / 4x2048 critic, same
tracking rewards, same contact observations, contact-match reward still at
**weight 0** — plus two terms from :mod:`pressure_terms`:

* ``pressure_share_rew`` (+0.3, bounded [0, 0.3]) — match the *measured human's*
  distribution of ground load across 6 support zones. Shares, not newtons, so
  the term is invariant to the mat's hold-phase gain bias (0.74-0.83 BW on
  forearm-supported poses) and to the 74 kg vs 71 kg mass mismatch.
* ``pressure_unloaded_rew`` (-0.3, clamped at -0.3) — charge simulated ground
  load on zones the measurement says carry nothing. One-sided, so it can only
  ask the policy to *stop* loading a limb, never to reach for one.

Plus three weight-0 diagnostics so wandb shows whether the supervision is
actually live: ``diag_pressure_gate_share`` and ``diag_pressure_gate_body`` are
the fraction of envs whose measurement is usable this step (expect ~0.3-0.6; a
flat 0 means the motion file has no measured channel, or was packaged before
the on-mat column existed), and ``diag_ground_fz`` is total ground load over
body weight (~1.0 in any supported steady state).

Requires a motion file whose clips carry the measured ground reaction **with
the third validity column**, i.e. packaged from
``data/smpl/yoga_motions_proto_yogi_pressure_gated/``. Training on an
unmeasured motion file is refused rather than silently running with both terms
at zero.

Why the contact-match reward stays at 0
---------------------------------------
These clips carry the old heuristic contact labels (0-16 % precision,
``notes/Contact_config_def.MD``), and the per-body "touching anything" reward
was a net negative on hard poses
(``notes/Contact_label_consequence.MD``). The pressure terms are the
replacement: they carry the same information where it is decidable, from
measurement rather than from proximity.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_pressure_rew.py \
      --experiment-name smpl_yogi_hard29_pressure_ab_s1 \
      --motion-file data/smpl/yoga_yogi_hard29_pressure.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb \
      --overrides env.ref_respawn_offset=0.005

Changing a weight requires a NEW ``--experiment-name``: a resume loads the
frozen ``resolved_configs.pt`` and does not re-read this module (the trap
recorded in :mod:`mlp_pair_contact_rew`).
"""

from __future__ import annotations

import argparse
import os

from examples.experiments.mimic import mlp_contact_rich as base
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
apply_inference_overrides = base.apply_inference_overrides
agent_config = base.agent_config
configure_robot_and_simulator = base.configure_robot_and_simulator

# Sized to sit below the tracking terms (gt 0.5 / gr 0.3) so the pressure
# shapes behaviour without overriding the pose. Both are bounded, and the
# penalty is one-sided, so the worst case is a 0.3 offset on frames where the
# policy is loading a limb the human was not.
PRESSURE_SHARE_WEIGHT = 0.3
PRESSURE_UNLOADED_WEIGHT = -0.3

# NOTE: the weight is unchanged but the term is now ~10x stronger, because
# `pressure_unloaded_rew_factory` was rescaled to saturate at 10 % of body weight
# rather than 100 %. Run `smpl_yogi_hard29_pressure_ab_s1` (wandb zgzf7z7t)
# measured the original scaling to be ~30x too weak: realistic violations are
# 5-45 N, so the penalty was 0.8 % of the tracking reward and the policy bought
# stability by leaning on unloaded limbs from epoch ~5600 onward
# (notes/Pressure_supervision_design.MD §7.6). Any run launched after
# 2026-08-14 therefore has a materially different objective from that one and
# must NOT be compared to it without accounting for the change.


def _assert_motion_file_is_measured_and_gated(motion_file) -> None:
    """Refuse to start if the terms would silently contribute nothing.

    Two distinct failures, both otherwise invisible until the run has burned an
    hour: a motion package with no measured ground reaction at all (MotionLib
    drops it all-or-nothing with only a log warning), and one packaged before
    ``add_onmat_gate_to_motions.py`` added the third validity column.
    """
    import torch

    if not motion_file or not os.path.isfile(str(motion_file)):
        return  # motion lib raises its own error for a missing file
    pack = torch.load(str(motion_file), map_location="cpu", weights_only=False)
    missing = [k for k in ("gnf", "grc", "grw") if pack.get(k) is None]
    assert not missing, (
        f"--motion-file {motion_file} has no measured ground reaction "
        f"(missing {missing}). Both pressure terms would be identically zero. "
        "Package from data/smpl/yoga_motions_proto_yogi_pressure_gated/ with "
        "data/scripts/package_motion_subset.py — and note MotionLib packs the "
        "channel ALL-OR-NOTHING, so one unmeasured clip drops it for every clip."
    )
    n_cols = int(pack["grw"].shape[1])
    assert n_cols >= 3, (
        f"--motion-file {motion_file} carries {n_cols} validity columns; the "
        "share term needs the third (on_mat x explained). Rebuild the clips "
        "with data/scripts/add_onmat_gate_to_motions.py and repackage. Gating "
        "shares on coverage instead would discard more than half the "
        "supervised hold frames (see notes/Pressure_supervision_design.MD 5.2)."
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from examples.experiments.mimic.pressure_terms import (
        VALID_ON_MAT,
        VALID_PER_BODY,
        pressure_gate_diag_factory,
        pressure_share_rew_factory,
        pressure_unloaded_rew_factory,
    )
    from examples.experiments.mimic.pair_contact_terms import ground_fz_diag_factory

    _assert_motion_file_is_measured_and_gated(getattr(args, "motion_file", None))

    cfg = base.env_config(robot_cfg, args)
    cfg.reward_components["pressure_share_rew"] = pressure_share_rew_factory(
        weight=PRESSURE_SHARE_WEIGHT
    )
    cfg.reward_components["pressure_unloaded_rew"] = pressure_unloaded_rew_factory(
        weight=PRESSURE_UNLOADED_WEIGHT, min_value=PRESSURE_UNLOADED_WEIGHT
    )
    # Weight 0 = computed and logged, not summed.
    cfg.reward_components["diag_pressure_gate_share"] = pressure_gate_diag_factory(
        column=VALID_ON_MAT
    )
    cfg.reward_components["diag_pressure_gate_body"] = pressure_gate_diag_factory(
        column=VALID_PER_BODY
    )
    cfg.reward_components["diag_ground_fz"] = ground_fz_diag_factory()
    return cfg
