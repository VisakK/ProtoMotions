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

**Support rule v2** (``support_v2_holds``; card E1 of
``expert_revist/graph_growth_2026_10_03/PLAN.MD``). v1's support check is a body
*origin* below 8 cm, applied only to zones the reference keeps entirely above
15 cm; at epoch 15,500 it saw 1 of the 10 support substitutions the collider
geometry shows (``expert_revist/expert56_v2_e15500/README.MD`` §4: a pointed toe
tip on the floor leaves the toe origin at 8.2 cm). v2 checks every zone the
release's sidecar says the human keeps free in the hold, by *load*: a zone
violates when its terrain-filtered ground load is at least 3 % of body weight on
at least 20 % of the hold window, the load mask dilated by the event window
first (so a tapping limb counts as down). Geometry alone never counts -- a head
hovering within 2 cm of the floor is not a support. Alongside it, commanded
supports are scored as realised when every commanded zone's lowest collider
point is within 2 cm of the floor on at least 90 % of the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    """One scored hold: the commanded pose is the reference at ``t_hold``.

    ``t_start`` is the start of the labelled window (v2 scores supports over
    ``[t_start, t_end]``, v1 and the pose over ``[t_hold, t_end]``); NaN means
    ``t_hold``.
    """

    t_hold: float
    t_end: float
    family: bool = False
    t_start: float = float("nan")


@dataclass
class ScoreParams:
    track_fail_m: float = 0.5
    pose_threshold_m: float = 0.15
    foot_down_z: float = 0.08
    unloaded_ref_min_z: float = 0.15
    track_weight: float = 0.5
    event_dilate_frames: int = 7


@dataclass
class SupportV2Params:
    """Support rule v2 (module docstring). Defaults are card E1's."""

    load_frac_bw: float = 0.03       # a zone is loaded at >= this share of body weight (vertical, N)
    body_weight_n: float = 74.0 * 9.81
    min_share: float = 0.2           # ... on >= this share of the window (after dilation) -> violation
    dilate_frames: int = 7           # +-0.23 s at 30 Hz
    down_m: float = 0.02             # a commanded zone is down when its lowest collider point is this low
    realised_share: float = 0.9      # ... on >= this share of the window -> commanded supports realised
    tracked_share: float = 0.9       # a hold counts as tracked when this share of its window is tracked

    @property
    def load_threshold_n(self) -> float:
        return self.load_frac_bw * self.body_weight_n


def dilate(mask: Tensor, k: int) -> Tensor:
    """``mask`` [T] OR-ed with its shifts by up to ``k`` frames either side."""
    out = mask.clone()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


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
    hold_violation: Optional[Sequence[Optional[Tensor]]] = None,
) -> Dict[str, float]:
    """Score one rollout of one clip.

    Args:
        sim_pos: ``[T, B, 3]`` simulated body positions (any constant XY offset).
        ref_pos: ``[T, B, 3]`` reference body positions at the same clip times.
        times: ``[T]`` clip time of each frame.
        holds: the clip's holds (from the hold manifest).
        exemplars: ``[H, B, 3]`` reference positions at each hold's ``t_hold``.
        goal_ids / zone_ids: body indices of the goal bodies and support zones.
        hold_violation: ``None`` (the v1 rule, unchanged) or, per hold, a bool
            mask over that hold's scored frames (``t_hold <= t <= t_end``) that
            replaces v1's per-frame support violation -- support rule v2's
            ``support_v2_holds`` masks, already event-dilated, so the event
            metric does not dilate them again. ``zone_ids`` is then unused.

    Returns:
        ``p_track`` (share of frames under the training termination's max-body
        gate), ``p_hold`` (mean over scored holds of the share of hold frames that
        reach the commanded pose *and* keep every zone the reference holds clear
        of the floor off the floor), ``p_family`` (same, family holds only; NaN if
        none), ``p_family_event`` (``p_family`` with each support violation dilated
        ``params.event_dilate_frames`` either side -- logged, never scored),
        ``support_violation`` (share of hold frames with an unwanted
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
    family_event: List[float] = []
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
        if hold_violation is None:
            violation = torch.zeros_like(attained)
            for ids in zone_ids.values():
                ids = list(ids)
                if float(r[:, ids, 2].min()) > params.unloaded_ref_min_z:
                    violation |= s[:, ids, 2].min(dim=-1).values < params.foot_down_z
            event = attained & ~dilate(violation, params.event_dilate_frames)
        else:
            given = hold_violation[h]
            violation = (torch.zeros_like(attained) if given is None
                         else given.to(device=attained.device, dtype=torch.bool))
            event = attained & ~violation
        attained &= ~violation
        share = attained.float().mean().item()
        hold_scores.append(share)
        if hold.family:
            family_scores.append(share)
            family_event.append(event.float().mean().item())
        violations += int(violation.sum())
        hold_frames += int(sel.sum())

    p_hold = sum(hold_scores) / len(hold_scores) if hold_scores else float("nan")
    p_family = sum(family_scores) / len(family_scores) if family_scores else float("nan")
    p_family_event = sum(family_event) / len(family_event) if family_event else float("nan")
    if hold_scores:
        score = params.track_weight * p_track + (1.0 - params.track_weight) * p_hold
    else:
        score = p_track
    return dict(
        p_track=p_track,
        p_hold=p_hold,
        p_family=p_family,
        p_family_event=p_family_event,
        support_violation=violations / hold_frames if hold_frames else float("nan"),
        track_fail_frac=1.0 - p_track,
        score=score,
        holds_scored=float(len(hold_scores)),
    )


def tracked_frames(sim_pos: Tensor, ref_pos: Tensor, track_fail_m: float) -> Tensor:
    """``[T]`` bool: every body within ``track_fail_m`` of the reference, after
    ``score_clip``'s frame-0 XY alignment (its ``p_track`` is the mean of this)."""
    sim = sim_pos.clone()
    sim[..., :2] -= (sim[0, 0, :2] - ref_pos[0, 0, :2])
    return ~((sim - ref_pos).norm(dim=-1).max(dim=-1).values > track_fail_m)


def zone_lowest_points(pos: Tensor, rot: Tensor, tables, zone_body_ids: Sequence[Sequence[int]]) -> Tensor:
    """``[N, Z]`` height of each zone's lowest collider patch point.

    ``pos [N, B, 3]``, ``rot [N, B, 4]`` (xyzw); ``tables`` carries the plant's
    collider geometry (``physics_terms.PhysicsTables`` or the same fields). The
    points are ``physics_terms.patch_points``' -- the selection
    ``build_physics_tables.patch_points`` uses offline.
    """
    from protomotions.envs.control.physics_terms import patch_points

    low: Dict[int, Tensor] = {}
    for ids in zone_body_ids:
        for b in ids:
            if b not in low:
                low[b] = patch_points(pos, rot, tables, [int(b)])[..., 2].min(dim=1).values
    return torch.stack(
        [torch.stack([low[b] for b in ids], dim=-1).min(dim=-1).values for ids in zone_body_ids], dim=-1
    )


def zone_vertical_load(ground_forces: Tensor, zone_body_ids: Sequence[Sequence[int]]) -> Tensor:
    """``[N, Z]`` terrain-filtered vertical load (N) per zone: the bodies' ``fz``,
    clamped at zero and summed -- the pricing of the unwanted-support term."""
    fz = ground_forces[..., 2].clamp_min(0.0)
    return torch.stack([fz[:, list(ids)].sum(dim=-1) for ids in zone_body_ids], dim=-1)


def support_v2_holds(
    times: Tensor,
    zone_loaded: Tensor,
    zone_low: Tensor,
    holds: Sequence[HoldWindow],
    commanded: Tensor,
    known_free: Tensor,
    tracked: Tensor,
    params: SupportV2Params,
) -> Tuple[List[Optional[dict]], List[Optional[Tensor]]]:
    """Support rule v2 on one rollout (module docstring).

    Args:
        times: ``[T]`` clip time of each frame.
        zone_loaded: ``[T, Z]`` bool, the zone carries at least
            ``params.load_threshold_n`` (an offline re-score may pass a proxy).
        zone_low: ``[T, Z]`` lowest collider point of each zone (m).
        holds: the clip's holds; the window is ``[t_start, t_end]``.
        commanded: ``[H, Z]`` bool, the hold's commanded ground zones.
        known_free: ``[H, Z]`` bool, zones the human is known to keep free.
        tracked: ``[T]`` bool (``tracked_frames``).

    Returns:
        ``(rows, masks)``, one entry per hold, ``None`` when no frame falls in
        its window. A row holds ``frames``, ``tracked_share``, ``realised``
        (``None`` without commanded zones), ``support_share`` (per commanded
        zone index), ``zone_share`` (``[Z]`` dilated load share on the known-free
        zones, 0 elsewhere), ``flagged`` (zone indices) and ``substitution``.
        A mask is the per-frame violation (any flagged zone's dilated load) on
        the hold's scored frames ``t_hold <= t <= t_end``, aligned with
        ``score_clip``'s selection, for its ``hold_violation``.
    """
    rows: List[Optional[dict]] = []
    masks: List[Optional[Tensor]] = []
    for h, hold in enumerate(holds):
        t0 = hold.t_start if hold.t_start == hold.t_start else hold.t_hold   # NaN -> t_hold
        win = (times >= t0) & (times <= hold.t_end)
        if not bool(win.any()):
            rows.append(None)
            masks.append(None)
            continue
        free = known_free[h].to(device=zone_loaded.device, dtype=torch.bool)
        load = dilate(zone_loaded[win], params.dilate_frames)                   # [n, Z]
        share = load.float().mean(dim=0) * free.float()                          # [Z]
        flagged = free & (share >= params.min_share)
        frame_violation = torch.zeros_like(times, dtype=torch.bool)
        frame_violation[win] = (load & flagged.unsqueeze(0)).any(dim=-1)
        post = (times >= hold.t_hold) & (times <= hold.t_end)
        masks.append(frame_violation[post])

        cmd = commanded[h].to(device=zone_low.device, dtype=torch.bool)
        realised: Optional[bool] = None
        support_share: Dict[int, float] = {}
        if bool(cmd.any()):
            down = (zone_low[win][:, cmd] <= params.down_m).float().mean(dim=0)
            support_share = {int(z): float(v) for z, v in zip(torch.nonzero(cmd).flatten().tolist(), down)}
            realised = bool((down >= params.realised_share).all())
        rows.append(dict(
            frames=int(win.sum()),
            tracked_share=float(tracked[win].float().mean()),
            realised=realised,
            support_share=support_share,
            zone_share=share.cpu(),
            flagged=[int(z) for z in torch.nonzero(flagged).flatten().tolist()],
            substitution=bool(flagged.any()),
        ))
    return rows, masks


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
    score_ema: Tensor, uniform_fraction: float, power: float = 1.0, eps: float = 1e-3,
    prior: Optional[Tensor] = None,
) -> Tensor:
    """``uniform_fraction`` of the mass uniform over clips, the rest in proportion to
    ``(1 - score) ** power + eps``. Every clip keeps at least ``uniform_fraction / N``.

    ``prior`` (``[N]``, non-negative, card E3): the uniform share becomes
    ``uniform_fraction * w_m / sum(w)`` -- e.g. the package's yaml weights, so a clip
    weighted 3 is sampled like three. ``None`` is the path above, unchanged."""
    n = score_ema.numel()
    need = (1.0 - score_ema).clamp(min=0.0).pow(power) + eps
    prioritized = need / need.sum()
    if prior is not None:
        prior = prior.to(device=score_ema.device)
    return uniform_share(n, uniform_fraction, prior) + (1.0 - uniform_fraction) * prioritized


def uniform_share(n: int, uniform_fraction: float, prior: Optional[Tensor] = None):
    """The mixture's uniform share per clip: the float ``uniform_fraction / n``, or with a
    prior the tensor ``uniform_fraction * w_m / sum(w)``.

    The prior is applied as ``(uniform_fraction / n) * (w_m * n / sum(w))`` -- the same
    product, ordered so that equal weights give the factor 1.0 exactly and the result is
    bit-identical to the no-prior path (``u * w / sum(w)`` rounds differently)."""
    if prior is None:
        return uniform_fraction / n
    prior = prior.to(dtype=torch.float32)
    if prior.numel() != n:
        raise ValueError(f"motion prior has {prior.numel()} weights for {n} motions")
    if bool((prior < 0).any()) or not bool(torch.isfinite(prior).all()) or float(prior.sum()) <= 0.0:
        raise ValueError("motion prior weights must be finite, non-negative and not all zero")
    return uniform_fraction / n * (prior * n / prior.sum())
