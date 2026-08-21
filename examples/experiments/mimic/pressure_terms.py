# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weak supervision from the measured MOYO pressure mat.

Design and the measurements behind it: ``notes/Pressure_supervision_design.MD``.

Two terms, both comparing the *simulated* per-body ground force
(``rigid_body_ground_forces``, the terrain column of IsaacLab's
``force_matrix_w``) against the *measured human* per-body ground force carried
on the reference motion in the same field and the same units.

``pressure_share_rew`` (option A, positive, bounded [0, 1])
    Pool the 24 bodies into 6 support zones, normalise each side to shares of
    its own total ground load, and score ``exp(-k * TV)`` where TV is the
    total-variation distance between the two share vectors.

``pressure_unloaded_rew`` (option B, penalty, bounded [0, 1] before weight)
    Where the measurement says a zone carries essentially nothing, charge the
    simulated ground load on that zone, saturating at **10 % of body weight**
    (see the factory's note: normalising by full mg made it ~30x too weak).

Why these and not something else
--------------------------------
* **Shares, never newtons.** During the hold the mat reads 0.74-0.83 body
  weights on forearm-supported poses and 0.93-0.97 on hand-supported ones,
  with no edge clipping — a measurement bias, not a field-of-view loss. Shares
  are invariant to a uniform gain error and to the 74 kg vs 71 kg mass
  mismatch; absolute force is not.
* **Zones, not bodies.** The attribution's dominant error is leakage *within* a
  limb chain — a shin capsule that runs knee-to-ankle stealing load from the
  foot it ends at (``notes/Moyo_pressure_port.MD`` 4.2). Zone pooling absorbs
  it. ``HANDS`` includes the wrists; ``FOREARM`` is kept separate because the
  hand/forearm split is exactly what distinguishes crow from a collapsed crow
  and pincha from a headstand.
* **Both terms are gated per frame** by the measurement's own confidence, and
  by *different* columns, because the channels fail differently. Column 2
  (``on_mat x explained``) gates shares; column 1 (``coverage x explained``)
  gates the absolute-load reading the unloaded term needs. Using coverage to
  gate a share target would discard the frames where the mat merely under-reads
  — more than half the supervised hold frames, and 100 % of Pincha's.
* **No pair identity, no geometry, no side-channel table.** Unlike
  :mod:`pair_contact_terms`, everything here is already on ``EnvContext``:
  both sides are (E, 24, 3) force fields in the same body order.

Calibration (offline, on the rollouts in ``results/``, no training):

===================  ==============  =================  ==================
                     19-motion       crow-pair @6250    crow-pair @15635
===================  ==============  =================  ==================
crow  TV / feet-BW   0.031 / 3.2 %   0.274 / 27.3 %     0.016 / 2.5 %
side  TV / feet-BW   0.263 / 29.3 %  0.007 / 2.6 %      0.010 / 7.1 %
===================  ==============  =================  ==================

Both terms recover the ordering documented in ``notes/Physics_insights.md``
without tuning.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent

# Body order is the COMMON/MJCF order used by both RobotState and MotionLib.
# (The recorded-rollout npz files use SIMULATOR order instead; that mismatch is
# a documented trap, but it does not apply on this path.)
COMMON_BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle",
    "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder",
    "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Hand",
]

SUPPORT_ZONES = {
    "HANDS": ["L_Wrist", "L_Hand", "R_Wrist", "R_Hand"],
    "FOREARM": ["L_Elbow", "R_Elbow"],
    "FEET": ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"],
    "SHANK": ["L_Knee", "R_Knee"],
    "HEAD": ["Neck", "Head"],
    "TORSO": ["Pelvis", "Torso", "Spine", "Chest", "L_Thorax", "R_Thorax",
              "L_Hip", "R_Hip", "L_Shoulder", "R_Shoulder"],
}

# Validity columns of ground_reaction_valid.
VALID_COVERAGE = 0        # coverage                gates ground_reaction
VALID_PER_BODY = 1        # coverage x explained    gates absolute per-body load
VALID_ON_MAT = 2          # on_mat  x explained     gates per-body SHARES

ROBOT_WEIGHT_N = 74.0 * 9.81


def build_zone_matrix(
    zones: Optional[dict] = None, body_names: Optional[list] = None
) -> Tensor:
    """[Z, B] 0/1 matrix pooling bodies into support zones."""
    zones = zones or SUPPORT_ZONES
    names = body_names or COMMON_BODY_NAMES
    index = {n: i for i, n in enumerate(names)}
    m = torch.zeros(len(zones), len(names), dtype=torch.float32)
    for z, (_, members) in enumerate(zones.items()):
        for n in members:
            if n not in index:
                raise ValueError(f"zone body '{n}' is not in the robot's body list")
            m[z, index[n]] = 1.0
    # Every body must be pooled exactly once, or the shares do not sum to one.
    counts = m.sum(0)
    if not bool((counts == 1).all()):
        bad = [names[i] for i in (counts != 1).nonzero(as_tuple=True)[0].tolist()]
        raise ValueError(f"bodies pooled {counts.unique().tolist()} times: {bad}")
    return m


def _zone_shares(fz: Tensor, zone_matrix: Tensor, min_total_n: float) -> tuple:
    """(shares [E, Z], total [E]) from per-body vertical force [E, B].

    Negative forces are clamped away: the measured channel is normal load and
    the simulator's terrain column is too, so a negative value is noise.
    """
    fz = fz.clamp_min(0.0)
    zone_force = fz @ zone_matrix.t()  # [E, Z]
    total = zone_force.sum(-1)  # [E]
    # keepdim, NOT a bare [E] divisor: broadcasting a [E] against [E, Z] is a
    # shape error for E != Z and *silently transposes the meaning* when E == Z.
    shares = zone_force / total.unsqueeze(-1).clamp_min(min_total_n)
    return shares, total


def _zeros_like_envs(*candidates: Optional[Tensor]) -> Tensor:
    """[E] zeros inferred from whichever input is present.

    The fallback when a motion library carries no measured channel must still be
    per-env: ``combine_rewards`` broadcasts, so a size-1 return would silently
    apply one env's value to all of them.
    """
    for t in candidates:
        if t is not None:
            return torch.zeros(t.shape[0], device=t.device, dtype=torch.float32)
    return torch.zeros(1)


def _gate(valid: Optional[Tensor], column: int, threshold: float, like: Tensor) -> Tensor:
    """Per-env 0/1 confidence gate from a ground_reaction_valid column.

    A motion library without the measured channel returns an all-zero gate
    rather than raising: these terms are meant to be attachable to an
    experiment that may be run on an unmeasured motion file, and a silent zero
    is visible in wandb as a flat ``raw_r`` trace.
    """
    if valid is None:
        return torch.zeros_like(like)
    if valid.shape[-1] <= column:
        raise RuntimeError(
            f"ground_reaction_valid has {valid.shape[-1]} columns but column "
            f"{column} was requested. Column {VALID_ON_MAT} is added by "
            "data/scripts/add_onmat_gate_to_motions.py — repackage the motion "
            "file from the gated directory."
        )
    return (valid[:, column] >= threshold).to(like.dtype)


def compute_pressure_share_rew(
    sim_ground_forces: Optional[Tensor],
    ref_ground_forces: Optional[Tensor],
    ground_reaction_valid: Optional[Tensor],
    zone_matrix: Tensor,
    sharpness: float = 3.0,
    valid_threshold: float = 0.90,
    min_total_n: float = 30.0,
    valid_column: int = VALID_ON_MAT,
) -> Tensor:
    """Option A. exp(-sharpness * TV(sim zone shares, measured zone shares)).

    Returns zero on frames the measurement cannot support, and on frames where
    either side is in flight (total ground load below ``min_total_n``) — with
    no load there is no distribution to match, and normalising noise would
    manufacture a target.
    """
    if sim_ground_forces is None or ref_ground_forces is None:
        return _zeros_like_envs(
            sim_ground_forces, ref_ground_forces, ground_reaction_valid
        )
    sim_fz = sim_ground_forces[..., 2]
    ref_fz = ref_ground_forces[..., 2]
    s_sim, tot_sim = _zone_shares(sim_fz, zone_matrix, min_total_n)
    s_ref, tot_ref = _zone_shares(ref_fz, zone_matrix, min_total_n)

    tv = 0.5 * (s_sim - s_ref).abs().sum(-1)
    reward = torch.exp(-sharpness * tv)

    gate = _gate(ground_reaction_valid, valid_column, valid_threshold, tv)
    airborne = (tot_sim < min_total_n) | (tot_ref < min_total_n)
    return reward * gate * (~airborne).to(reward.dtype)


def compute_pressure_unloaded_rew(
    sim_ground_forces: Optional[Tensor],
    ref_ground_forces: Optional[Tensor],
    ground_reaction_valid: Optional[Tensor],
    zone_matrix: Tensor,
    unloaded_share: float = 0.02,
    valid_threshold: float = 0.90,
    min_total_n: float = 30.0,
    total_weight_n: float = 0.1 * ROBOT_WEIGHT_N,
    valid_column: int = VALID_PER_BODY,
) -> Tensor:
    """Option B. Simulated ground load, in units of ``total_weight_n``, on zones
    the measurement says carry nothing. Positive in [0, 1]; give it a NEGATIVE weight.

    ``total_weight_n`` defaults to **0.1 x body weight**, not body weight.
    Normalising by full mg was measured to make the term ~30x too weak to bite:
    realistic violations are 5-45 N, i.e. 0.007-0.06 of mg, so at weight -0.3 the
    penalty was 0.8 % of the tracking reward and the policy happily bought
    stability by leaning on limbs the human left unloaded (run
    ``smpl_yogi_hard29_pressure_ab_s1``, epoch ~5600 onward; see
    ``notes/Pressure_supervision_design.MD`` 7.6). Saturating at 10 % of body
    weight keeps the term bounded in [0, 1] and readable against the tracking
    terms while making a 25 N violation cost ~0.10 of reward.

    One-sided on purpose. It never asks the policy to *add* load, only to stop
    putting load where the human had none, so it cannot fight the tracking
    reward into an unreachable pose. The gate is the per-body column, not the
    share column: a *false* zero — measured load the attribution failed to
    place — would punish a correct contact, and ``explained`` is what rules
    that out.
    """
    if sim_ground_forces is None or ref_ground_forces is None:
        return _zeros_like_envs(
            sim_ground_forces, ref_ground_forces, ground_reaction_valid
        )
    sim_fz = sim_ground_forces[..., 2].clamp_min(0.0)
    s_ref, tot_ref = _zone_shares(ref_ground_forces[..., 2], zone_matrix, min_total_n)

    zone_sim = sim_fz @ zone_matrix.t()  # [E, Z], newtons
    empty = (s_ref < unloaded_share).to(zone_sim.dtype)
    violation = (empty * zone_sim).sum(-1) / total_weight_n

    gate = _gate(ground_reaction_valid, valid_column, valid_threshold, violation)
    # With no measured load there is no evidence that any zone is unloaded.
    has_ref_load = (tot_ref >= min_total_n).to(violation.dtype)
    return violation.clamp(0.0, 1.0) * gate * has_ref_load


def compute_pressure_gate_diag(
    ground_reaction_valid: Optional[Tensor],
    column: int = VALID_ON_MAT,
    valid_threshold: float = 0.90,
) -> Tensor:
    """Weight-0 diagnostic: the fraction of envs whose measurement is usable.

    If this pins to 0 in wandb, the terms above are silently contributing
    nothing — either the motion file has no measured channel or it was
    packaged before the on-mat column was added.
    """
    if ground_reaction_valid is None:
        return torch.zeros(1)
    return (ground_reaction_valid[:, column] >= valid_threshold).float()


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def _pressure_dynamic_vars():
    return {
        "sim_ground_forces": EnvContext.current.rigid_body_ground_forces,
        "ref_ground_forces": EnvContext.mimic.ref_state.rigid_body_ground_forces,
        "ground_reaction_valid": EnvContext.mimic.ref_state.ground_reaction_valid,
    }


def pressure_share_rew_factory(
    weight: float = 0.3,
    sharpness: float = 3.0,
    valid_threshold: float = 0.90,
    min_total_n: float = 30.0,
    zones: Optional[dict] = None,
) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_pressure_share_rew,
        dynamic_vars=_pressure_dynamic_vars(),
        static_params={
            "weight": weight,
            "max_value": abs(weight),
            "zero_during_grace_period": True,
            "zone_matrix": build_zone_matrix(zones),
            "sharpness": sharpness,
            "valid_threshold": valid_threshold,
            "min_total_n": min_total_n,
        },
    )


def pressure_unloaded_rew_factory(
    weight: float = -0.3,
    min_value: float = -0.3,
    unloaded_share: float = 0.02,
    valid_threshold: float = 0.90,
    min_total_n: float = 30.0,
    total_weight_n: float = 0.1 * ROBOT_WEIGHT_N,
    zones: Optional[dict] = None,
) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_pressure_unloaded_rew,
        dynamic_vars=_pressure_dynamic_vars(),
        static_params={
            "weight": weight,
            "min_value": min_value,
            "zero_during_grace_period": True,
            "zone_matrix": build_zone_matrix(zones),
            "unloaded_share": unloaded_share,
            "valid_threshold": valid_threshold,
            "min_total_n": min_total_n,
            "total_weight_n": total_weight_n,
        },
    )


def pressure_gate_diag_factory(
    column: int = VALID_ON_MAT, valid_threshold: float = 0.90
) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_pressure_gate_diag,
        dynamic_vars={
            "ground_reaction_valid": EnvContext.mimic.ref_state.ground_reaction_valid,
        },
        static_params={"weight": 0.0, "column": column,
                       "valid_threshold": valid_threshold},
    )
