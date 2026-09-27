# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-clip performance score and the uniform + performance sampling curriculum.

Pure tensor functions (no simulator), used by ``HoldCurriculumEvaluator`` and
unit-tested on CPU.

Why this replaces the stock verdict (``expert_revist/run1_gap_analysis.MD`` §2 G2-G3,
§4.2-4.3): the stock ``MimicEvaluator`` scored the first 600 steps of every clip
with one binary verdict -- *mean* body error < 0.5 m -- and reset a failing clip's
weight to 1.0 while decaying every passing clip by 0.819 per eval with no floor.
That verdict passed a Headstand that never inverted and four arm balances done
feet-down, and the curriculum it drove starved 102/180 variants below
p = 1e-4 and made the starved clips regress. The score here is continuous, covers
the whole clip, scores the holds the policy is commanded to reach (pose *and*
support), and only ever steers a fixed minority of the sampling mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

GOAL_BODY_NAMES = ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")
SUPPORT_ZONES = {
    "L_FOOT": ("L_Ankle", "L_Toe"),
    "R_FOOT": ("R_Ankle", "R_Toe"),
    "L_HAND": ("L_Wrist", "L_Hand"),
    "R_HAND": ("R_Wrist", "R_Hand"),
}


@dataclass
class HoldWindow:
    """One scored hold: the commanded pose is the reference at ``t_hold``."""

    t_hold: float
    t_end: float
    family: bool = False


@dataclass
class ScoreParams:
    track_fail_m: float = 0.5
    pose_threshold_m: float = 0.15
    foot_down_z: float = 0.08
    unloaded_ref_min_z: float = 0.15
    track_weight: float = 0.5


def best_yaw_distance(frames: Tensor, target: Tensor) -> Tensor:
    """Mean per-body distance after rotating each frame by its best-fitting yaw.

    Args:
        frames: ``[T, B, 3]`` pelvis-relative positions.
        target: ``[B, 3]`` or ``[T, B, 3]`` pelvis-relative positions.

    Returns:
        ``[T]`` distances. Root-quaternion heading normalisation is avoided on
        purpose: it flips on prone and inverted poses.
    """
    if target.dim() == 2:
        target = target.unsqueeze(0).expand_as(frames)
    a, b = frames[..., :2], target[..., :2]
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    th = torch.atan2(num, den)
    c, s = torch.cos(th).unsqueeze(-1), torch.sin(th).unsqueeze(-1)
    x = c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] + c * a[..., 1]
    rotated = torch.stack([x, y, frames[..., 2]], dim=-1)
    return (rotated - target).norm(dim=-1).mean(-1)


def score_clip(
    sim_pos: Tensor,
    ref_pos: Tensor,
    times: Tensor,
    holds: Sequence[HoldWindow],
    exemplars: Optional[Tensor],
    goal_ids: Sequence[int],
    zone_ids: Dict[str, Sequence[int]],
    params: ScoreParams,
) -> Dict[str, float]:
    """Score one rollout of one clip.

    Args:
        sim_pos: ``[T, B, 3]`` simulated body positions (any constant XY offset).
        ref_pos: ``[T, B, 3]`` reference body positions at the same clip times.
        times: ``[T]`` clip time of each frame.
        holds: the clip's holds (from the hold manifest).
        exemplars: ``[H, B, 3]`` reference positions at each hold's ``t_hold``.
        goal_ids / zone_ids: body indices of the goal bodies and support zones.

    Returns:
        ``p_track`` (share of frames under the training termination's max-body
        gate), ``p_hold`` (mean over scored holds of the share of hold frames that
        reach the commanded pose *and* keep every zone the reference holds clear
        of the floor off the floor), ``p_family`` (same, family holds only; NaN if
        none), ``support_violation`` (share of hold frames with an unwanted
        support), ``score`` (``track_weight * p_track + (1 - track_weight) *
        p_hold``, or ``p_track`` when no hold falls in the rollout).
    """
    sim = sim_pos.clone()
    sim[..., :2] -= (sim[0, 0, :2] - ref_pos[0, 0, :2])
    body_err = (sim - ref_pos).norm(dim=-1)
    track_fail = body_err.max(dim=-1).values > params.track_fail_m
    p_track = 1.0 - track_fail.float().mean().item()

    goal = list(goal_ids)
    hold_scores: List[float] = []
    family_scores: List[float] = []
    violations = 0
    hold_frames = 0
    for h, hold in enumerate(holds):
        sel = (times >= hold.t_hold) & (times <= hold.t_end)
        if not bool(sel.any()):
            continue
        s = sim[sel]
        r = ref_pos[sel]
        ex = exemplars[h]
        dist = best_yaw_distance(s[:, goal] - s[:, :1], ex[goal] - ex[:1])
        attained = dist < params.pose_threshold_m
        violation = torch.zeros_like(attained)
        for ids in zone_ids.values():
            ids = list(ids)
            if float(r[:, ids, 2].min()) > params.unloaded_ref_min_z:
                violation |= s[:, ids, 2].min(dim=-1).values < params.foot_down_z
        attained &= ~violation
        share = attained.float().mean().item()
        hold_scores.append(share)
        if hold.family:
            family_scores.append(share)
        violations += int(violation.sum())
        hold_frames += int(sel.sum())

    p_hold = sum(hold_scores) / len(hold_scores) if hold_scores else float("nan")
    p_family = sum(family_scores) / len(family_scores) if family_scores else float("nan")
    if hold_scores:
        score = params.track_weight * p_track + (1.0 - params.track_weight) * p_hold
    else:
        score = p_track
    return dict(
        p_track=p_track,
        p_hold=p_hold,
        p_family=p_family,
        support_violation=violations / hold_frames if hold_frames else float("nan"),
        track_fail_frac=1.0 - p_track,
        score=score,
        holds_scored=float(len(hold_scores)),
    )


def update_score_ema(previous: Optional[Tensor], current: Tensor, keep: float) -> Tensor:
    """``keep * previous + (1 - keep) * current``; unscored (NaN) entries keep the
    previous value; the first call returns ``current`` with NaN -> 1."""
    current = current.clone()
    if previous is None:
        return torch.nan_to_num(current, nan=1.0)
    fresh = torch.isfinite(current)
    out = previous.clone()
    out[fresh] = keep * previous[fresh] + (1.0 - keep) * current[fresh]
    return out


def mixture_sampling_probs(
    score_ema: Tensor, uniform_fraction: float, power: float = 1.0, eps: float = 1e-3
) -> Tensor:
    """``uniform_fraction`` of the mass uniform over clips, the rest in proportion to
    ``(1 - score) ** power + eps``. Every clip keeps at least ``uniform_fraction / N``."""
    n = score_ema.numel()
    need = (1.0 - score_ema).clamp(min=0.0).pow(power) + eps
    prioritized = need / need.sum()
    return uniform_fraction / n + (1.0 - uniform_fraction) * prioritized
