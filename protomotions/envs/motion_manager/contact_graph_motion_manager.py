# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-state initialisation anchored to contact-graph segments.

Reference-state initialisation samples a clip time *uniformly*, so an episode
begins wherever the clip happens to be.  For a contact-conditioned student that
spends most of the budget in the middle of holds, which is the part of a clip
the policy finds easiest and learns least from.  The states worth starting in are
the ones just before the support set changes -- and those are exactly what the
contact graph already enumerates, one entry per trusted segment.

So this manager keeps the clip sampling of :class:`MimicMotionManager` and
replaces the *time* sampling: with probability ``segment_start_prob`` an episode
starts a little before a segment boundary rather than at a uniform time.

Three details that are deliberate rather than incidental:

* **The offset is randomised, not fixed.**  Starting exactly ``pre_roll_s``
  before every boundary would put every episode at the same phase relative to
  the transition; the offset is drawn uniformly from ``[0, pre_roll_s]`` so the
  approach *and* the entry frame are both covered.
* **Segments are chosen uniformly by default, not by duration.**  A clip's time
  is dominated by a few long holds, so uniform-over-time barely ever lands near a
  boundary.  Uniform-over-segments deliberately over-samples the boundaries.
  ``segment_weighting='dwell'`` recovers the uniform-in-time behaviour if that
  turns out to be wanted.
* **Departures as well as arrivals** (``segment_end_prob``, card E3 of
  ``expert_revist/graph_growth_2026_10_03/PLAN.MD``).  A transition out of a
  hold -- a synthetic edge, a get-up -- begins where the hold *ends*, which an
  arrival anchor never starts near.  With probability ``segment_end_prob`` a
  reset starts at ``t_end - U(0, pre_roll_s)`` of a segment instead.  The two
  probabilities share one uniform draw, the segment draw and the pre-roll, so
  with ``segment_end_prob = 0`` every random number and every start time is
  the arrival-only manager's, call for call.  A segment that ends within
  ``env_dt`` of its clip's end (every clip's last hold, which runs to the end)
  is never a departure: the episode would end before the departure began.
* **This changes the start state only.**  It does not touch what the goals are
  or which expert labels the action -- the episode still plays the clip forward
  and ``ContactGraphControl`` still serves that clip's own upcoming holds.  The
  adjacency structure of the graph cannot be used to pick *goals* without
  breaking the label (see ``ContactGraphControl``'s note on why goals always come
  from the clip being tracked), but it can be used to pick *starts*, and that is
  what this does.
"""

from typing import Dict, Optional

import torch

from protomotions.components.contact_graph import ContactGraph
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.motion_manager.config import ContactGraphMotionManagerConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager

WEIGHTINGS = ("uniform", "dwell", "rare_node")

# What each reset actually started at, in ``start_kind`` order: the time
# MimicMotionManager drew (uniform, or t = 0 under init_start_prob) unless an
# arrival or a departure anchor replaced it.
START_KINDS = ("uniform", "t0", "arrival", "departure")
KIND_UNIFORM, KIND_T0, KIND_ARRIVAL, KIND_DEPARTURE = range(len(START_KINDS))


class ContactGraphMotionManager(MimicMotionManager):
    """Mimic motion manager that starts episodes at contact-graph boundaries."""

    config: ContactGraphMotionManagerConfig

    def __init__(
        self,
        config: ContactGraphMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            config, num_envs, env_dt, device, motion_lib, fixed_motion_ids_per_env
        )
        if not config.graph_file:
            raise ValueError(
                "ContactGraphMotionManagerConfig.graph_file is required; without "
                "it there are no segments to anchor to"
            )
        if not 0.0 <= float(config.segment_start_prob) <= 1.0:
            raise ValueError(
                f"segment_start_prob must be in [0, 1], got {config.segment_start_prob}"
            )
        end_prob = self._segment_end_prob
        if not 0.0 <= end_prob <= 1.0:
            raise ValueError(f"segment_end_prob must be in [0, 1], got {end_prob}")
        if float(config.segment_start_prob) + end_prob > 1.0 + 1e-9:
            raise ValueError(
                "segment_start_prob + segment_end_prob must be <= 1 (they split one "
                f"draw), got {config.segment_start_prob} + {end_prob}"
            )
        if float(config.pre_roll_s) < 0.0:
            raise ValueError(f"pre_roll_s must be >= 0, got {config.pre_roll_s}")
        if config.segment_weighting not in WEIGHTINGS:
            raise ValueError(
                f"segment_weighting must be one of {WEIGHTINGS}, got "
                f"'{config.segment_weighting}'"
            )

        graph = ContactGraph.from_file(config.graph_file, device=device)
        # Same check the control component makes, repeated here rather than
        # assumed: this manager is often constructed before the component, and a
        # graph built for a different library would index the wrong clips'
        # segments into the wrong motions with no other symptom than odd starts.
        graph.validate_against_motion_lib(list(motion_lib.motion_files))

        self.seg_start = graph.seg_start
        self.seg_end = graph.seg_end  # +inf where padded, like seg_start
        self.seg_count = graph.seg_count
        self._segment_cdf = self._build_segment_cdf(graph)
        self._motion_lengths = motion_lib.motion_lengths.to(device)
        # Departure anchoring: the same segment weights over the segments a
        # departure may start from -- every live segment that ends at least
        # env_dt before its clip does (the arithmetic of `upper` below).
        slots = torch.arange(graph.seg_end.shape[1], device=device).unsqueeze(0)
        live = slots < self.seg_count.unsqueeze(-1)
        self._departure_eligible = live & (
            self.seg_end < (self._motion_lengths - self.env_dt).unsqueeze(-1)
        )
        self._departure_cdf = self._build_segment_cdf(
            graph, eligible=self._departure_eligible
        )

        # Per-env start kind of the last reset (START_KINDS), and the start kinds
        # drawn since the env last published them (pop_step_logs).
        self.start_kind = torch.full(
            (num_envs,), KIND_UNIFORM, dtype=torch.long, device=device
        )
        self._kind_counts = torch.zeros(len(START_KINDS), dtype=torch.long, device=device)
        self._pending_resets = 0

        covered = int((self.seg_count > 0).sum())
        print(
            f"ContactGraphMotionManager: anchoring {config.segment_start_prob:.0%} "
            f"of resets to {int(self.seg_count.sum())} segments over {covered}/"
            f"{len(graph.motion_names)} motions "
            f"(weighting '{config.segment_weighting}', "
            f"pre-roll 0-{config.pre_roll_s:.2f} s)"
        )
        if end_prob > 0.0:
            print(
                f"ContactGraphMotionManager: departure-anchoring {end_prob:.0%} of "
                f"resets to {int(self._departure_eligible.sum())} segments over "
                f"{int(self._departure_eligible.any(dim=-1).sum())}/"
                f"{len(graph.motion_names)} motions (segments ending within "
                f"env_dt of their clip's end excluded)"
            )
        if covered == 0:
            raise ValueError(
                "contact graph has no segments for any motion; nothing to anchor to"
            )

    # ------------------------------------------------------------------ #
    @property
    def _segment_end_prob(self) -> float:
        """``config.segment_end_prob``, read live like ``segment_start_prob`` so that
        editing the config zeroes both together. getattr: configs pickled before
        departure anchoring existed lack the field and read 0."""
        return float(getattr(self.config, "segment_end_prob", 0.0) or 0.0)

    def _build_segment_cdf(
        self, graph: ContactGraph, eligible: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Per-motion cumulative segment weights, ``[num_motions, max_segments]``.

        A row-wise CDF turns "sample a segment of this clip" into one batched
        ``searchsorted``, which is what keeps the anchoring free at 1024 envs.
        Padded slots get weight 0 so they can never be drawn, and every row is
        normalised to end at exactly 1.0 so a uniform draw cannot fall past the
        end through float error.

        ``eligible`` (``[num_motions, max_segments]`` bool) gives weight 0 to
        every live segment outside it (departure anchoring) after the weights
        are computed, so a 'rare_node' weight still counts every live segment;
        ``None`` is every live segment, exactly as before it existed. A row with
        no eligible segment is all zeros up to the forced final 1.0 and must
        never be used (the caller checks the eligibility of what it drew).
        """
        num_motions, max_segments = graph.seg_start.shape
        device = graph.seg_start.device
        slots = torch.arange(max_segments, device=device).unsqueeze(0)
        live = slots < graph.seg_count.unsqueeze(-1)

        mode = self.config.segment_weighting
        if mode == "uniform":
            weights = torch.ones_like(graph.seg_start)
        elif mode == "dwell":
            weights = (graph.seg_end - graph.seg_start).clamp(min=0.0)
        else:  # rare_node
            # 1/sqrt(corpus frequency of the node this segment holds). Softer
            # than 1/f, which would hand almost all the probability to the
            # long tail of configurations seen exactly once.
            nodes = graph.safe_seg_node
            counts = torch.bincount(
                nodes[live].reshape(-1), minlength=graph.num_nodes
            ).clamp(min=1)
            weights = counts[nodes].float().rsqrt()

        if eligible is not None:
            live = live & eligible.to(device)
        weights = torch.where(live, weights, torch.zeros_like(weights))
        # A live row whose weights all came out 0 (a zero-duration segment under
        # 'dwell') would produce a 0/0 CDF; fall back to uniform on those rows.
        totals = weights.sum(dim=-1, keepdim=True)
        degenerate = (totals <= 0) & live.any(dim=-1, keepdim=True)
        weights = torch.where(degenerate, live.float(), weights)

        cdf = weights.cumsum(dim=-1)
        totals = cdf[:, -1:].clamp(min=1e-9)
        cdf = cdf / totals
        # Force the last live slot to exactly 1.0: searchsorted on a row whose
        # maximum is 0.9999999 can return max_segments for u close to 1.
        cdf[:, -1] = 1.0
        return cdf.contiguous()

    # ------------------------------------------------------------------ #
    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        """Sample clips as usual, then move the start times onto segment entries."""
        # MimicMotionManager only resamples finished tracks when
        # resample_on_reset is False, and anchoring an env whose clip was NOT
        # resampled would rewind it mid-episode. Mirror that filter exactly.
        reset_env_ids = env_ids
        if not self.config.resample_on_reset:
            reset_env_ids = env_ids[self.get_done_tracks(env_ids)]

        super().sample_motions(env_ids, new_motion_ids)

        if len(reset_env_ids) == 0 or (
            self.config.segment_start_prob <= 0.0 and self._segment_end_prob <= 0.0
        ):
            return
        self._anchor_to_segments(reset_env_ids)

    def _anchor_to_segments(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        device = self.device
        motion_ids = self.motion_ids[env_ids]
        p_start = self.config.segment_start_prob
        p_end = self._segment_end_prob

        # ONE uniform picks the kind of start: an arrival if u < p_start, a
        # departure if p_start <= u < p_start + p_end, otherwise the time
        # MimicMotionManager drew. With p_end = 0 this is the Bernoulli draw the
        # arrival-only manager made, and the three torch.rand calls below are its
        # three, in its order -- the random stream is unchanged call for call.
        u = torch.rand(count, device=device)
        fire = u < p_start
        selected = fire & (self.seg_count[motion_ids] > 0)

        draw = torch.rand(count, device=device).unsqueeze(-1)
        index = torch.searchsorted(self._segment_cdf[motion_ids], draw).squeeze(-1)
        index = index.clamp(max=(self.seg_count[motion_ids] - 1).clamp(min=0))

        starts = self.seg_start[motion_ids, index]
        # Uniform in [0, pre_roll_s]: a fixed offset would put every episode at
        # the same phase relative to the transition it is about to make.
        pre_roll = torch.rand(count, device=device) * float(self.config.pre_roll_s)
        # truncate_time=env_dt matches MotionManager.sample_time, so an anchored
        # start can never be the frame that immediately ends the episode.
        upper = (self._motion_lengths[motion_ids] - self.env_dt).clamp(min=0.0)
        anchored = (starts - pre_roll).clamp(min=0.0).minimum(upper)

        drawn = self.motion_times[env_ids]
        times = torch.where(selected, anchored, drawn)
        kind = torch.where(
            drawn == 0.0,
            torch.full_like(motion_ids, KIND_T0),
            torch.full_like(motion_ids, KIND_UNIFORM),
        )
        kind = torch.where(selected, torch.full_like(kind, KIND_ARRIVAL), kind)

        if p_end > 0.0:
            # Same segment draw and pre-roll, on the departure CDF. Checking the
            # drawn slot (rather than only "has an eligible segment") also covers
            # a draw of exactly 0, which searchsorted maps to slot 0 even when
            # that slot's weight is 0.
            leave = torch.searchsorted(self._departure_cdf[motion_ids], draw).squeeze(-1)
            leave = leave.clamp(max=(self.seg_count[motion_ids] - 1).clamp(min=0))
            depart = (
                (u >= p_start)
                & (u < p_start + p_end)
                & self._departure_eligible[motion_ids, leave]
            )
            ends = self.seg_end[motion_ids, leave]
            departed = (ends - pre_roll).clamp(min=0.0).minimum(upper)
            times = torch.where(depart, departed, times)
            kind = torch.where(depart, torch.full_like(kind, KIND_DEPARTURE), kind)

        self.motion_times[env_ids] = times
        self.start_kind[env_ids] = kind
        # A one-hot sum, not bincount: bincount syncs the device to size its output.
        kinds = torch.arange(len(START_KINDS), device=device)
        self._kind_counts += (kind.unsqueeze(-1) == kinds).sum(dim=0)
        self._pending_resets += count

    def pop_step_logs(self) -> Dict[str, torch.Tensor]:
        """Start-kind counts of the resets anchored since the last call, then reset.

        ``BaseEnv.post_physics_step`` merges this into the step's extras and the
        agent logs each key's mean over the epoch's steps: ``env/anchor/<kind>_resets``
        (resets per step that started as ``<kind>``) and ``env/anchor/resets``
        (resets per step). Both are means over the same steps, so
        ``env/anchor/<kind>_resets / env/anchor/resets`` is the epoch's drawn share
        exactly -- the pooled count ratio, not an average of per-step shares.
        Every key is emitted on every step, zeros included: a key that appeared
        only on steps with resets would give two ranks different log keys (or
        key orders), and the per-key all_gather would then pair the wrong
        metrics or hang. Empty while the manager anchors nothing (both
        probabilities 0).
        """
        if self.config.segment_start_prob <= 0.0 and self._segment_end_prob <= 0.0:
            return {}
        logs = {
            f"anchor/{name}_resets": self._kind_counts[i].clone()
            for i, name in enumerate(START_KINDS)
        }
        logs["anchor/resets"] = torch.full(
            (), float(self._pending_resets), device=self.device
        )
        self._kind_counts.zero_()
        self._pending_resets = 0
        return logs
