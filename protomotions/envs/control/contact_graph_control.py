# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-graph goal conditioning for the MaskedMimic student.

Vanilla MaskedMimic conditions on *random* future frames of the clip being
tracked: a beta-distributed time offset and a random subset of bodies.  That
teaches "reach some pose, some time from now".

This component replaces the random schedule with the contact graph built off
expert rollouts (``data/scripts/build_contact_graph_from_rollouts.py``).  The
goals are no longer arbitrary frames -- they are the frames where the policy is
*holding* a particular contact configuration, and each one is handed to the
student as two halves that are masked independently:

    contact half   which contact pairs carry load, and which way up the trunk is
    pose  half     the body pose being held at that moment

Masking them independently is the whole point.  With both revealed the task is
MaskedMimic with a better-chosen target frame.  With only the pose revealed it is
vanilla MaskedMimic.  With only the *contact configuration* revealed the student
has to find a pose that realises it, which is the capability the graph exists to
provide -- and it is not redundant with the pose, because
``notes/Pressure_supervision_design.MD`` §3 shows load-bearing is not decidable
from the reference kinematics (a crow foot 9 cm off the floor carrying 636 N).

The goal *schedule* is a pure function of ``(motion_id, motion_time)`` -- the next
``num_goal_steps`` holds in the clip -- so nothing has to be kept in sync with the
motion manager.  Only the visibility masks are stateful, and they are deliberately
sticky: they follow a goal as it shifts down the slots, so the specification the
student is working towards does not flicker every step.

``ctx.masked_mimic`` is populated in exactly the shape ``MaskedMimicControl``
produces, so every masked-mimic observation kernel and the whole Stage-2 model
work unchanged; ``ctx.contact_goal`` carries the contact half.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import Tensor

from protomotions.components.contact_graph import ContactGraph
from protomotions.envs.control.contact_event_tracker import ContactEventTracker
from protomotions.envs.context_views import (
    ContactGoalContext,
    EnvContext,
    MaskedMimicContext,
)
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.envs.obs.contact_state import compute_contact_state_obs
from protomotions.simulator.base_simulator.config import MarkerState

log = logging.getLogger(__name__)
from protomotions.simulator.base_simulator.simulator_state import ResetState
from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class ContactGraphControlConfig(MaskedMimicControlConfig):
    """Configuration for contact-graph goal conditioning.

    Attributes:
        graph_file: ``contact_graph.pt`` produced by the rollout graph builder.
            Its motion order must match the environment's motion library.
        num_goal_steps: How many upcoming contact configurations to expose.
            Kept equal to ``num_masked_future_steps`` so the inherited buffers
            and the model's token count line up.
        min_lead_s: A hold nearer than this is treated as already passed, so the
            nearest goal is always one the policy still has time to act on.
        pose_visible_prob: Probability that a slot reveals the held pose.
        contact_visible_prob: Probability that a slot reveals the contact set.
        full_pose_prob: Given the pose is revealed, probability that *every*
            conditionable body is revealed rather than a random subset.
        require_first_goal_specified: Force the nearest slot to reveal at least
            its contact set when the sampler would have hidden both halves.
        ground_contact_threshold_n: Per-zone force above which the *simulated*
            ground contact counts as active, for the reached-goal diagnostic.
        body_contact_threshold_n: The same, for a body-body zone pair. Only used
            when ``RobotConfig.contact_pair_bodies`` makes those observable.
        num_history_events: Contact-event history tokens exposed to the policy
            (1 open segment + N-1 completed ones, newest first). 0 disables the
            tracker and the history context fields carry empty tensors.
        history_min_dwell_s: A new measured configuration must persist this long
            before it closes the open segment — the causal counterpart of the
            graph builder's debounce, and what keeps pair-threshold chatter from
            committing events.
        history_time_clip_s: Event times (age, dwell) are clamped here and
            scaled into [0, 1], so the history block needs no running
            normalizer and its binary channels stay binary. Also the scale for
            the dwell channels below, so every time channel in the contact
            block shares one unit.
        dwell_channels: Append ``[hold_duration, dwell_remaining]`` to each goal
            slot of ``contact_goal_obs`` -- how long the commanded configuration
            lasts, and how much of it is left, both clamped to
            ``history_time_clip_s`` and scaled to [0, 1]. False keeps the block
            byte-identical to every run before v10_1.

            The command has never carried this. The *deadline*
            (``masked_mimic_target_times``) says when to be somewhere, not how
            long to stay, and Tier-0 §5 showed raising it makes holds
            monotonically **worse** -- it is the wrong channel for the job. The
            right one is duration, which spans p10 0.60 s / median 1.70 s /
            p90 9.07 s across the corpus (round 7_1 §6.5).
        include_current_segment: Put the segment the clip is currently inside in
            goal slot 0 instead of always the next hold. Pairs with
            ``dwell_channels``: "stay for X more seconds" is meaningless for a
            goal you have not reached, and without this slot 0 is always a goal
            you have not reached. Removes the 42.1 % of trusted dwell that today
            is spent inside a segment while commanded to leave it (round 9 §3a).
        far_goal_prob: Probability that an episode's forward goal window starts
            some holds later than usual -- far-goal promotion. Nothing in
            training has ever put a distant goal in the nearest slot, so a
            policy asked for one at inference is off-distribution (the
            far-handstand probe reaches its goal on 0.028 of replicas). 0.0
            disables. Sampled per episode at reset, never mid-episode.
        far_goal_max_skip: Upper bound of the uniform skip when promotion fires.
            No measurement backs the default; it is a first guess.
    """

    _target_: str = "protomotions.envs.control.contact_graph_control.ContactGraphControl"

    graph_file: str = ""
    num_goal_steps: int = 5
    min_lead_s: float = 0.2
    pose_visible_prob: float = 0.75
    contact_visible_prob: float = 0.85
    full_pose_prob: float = 0.75
    require_first_goal_specified: bool = True
    ground_contact_threshold_n: float = 20.0
    body_contact_threshold_n: float = 20.0
    num_history_events: int = 0
    history_min_dwell_s: float = 0.3
    history_time_clip_s: float = 10.0
    dwell_channels: bool = False
    include_current_segment: bool = False
    far_goal_prob: float = 0.0
    far_goal_max_skip: int = 3


class ContactGraphControl(MaskedMimicControl):
    """Masked-mimic conditioning whose targets come from the contact graph."""

    config: ContactGraphControlConfig

    def __init__(self, config: ContactGraphControlConfig, env: "BaseEnv"):
        if config.num_goal_steps != config.num_masked_future_steps:
            raise ValueError(
                f"num_goal_steps ({config.num_goal_steps}) must equal "
                f"num_masked_future_steps ({config.num_masked_future_steps}); the "
                f"student's token count is built from the latter"
            )
        super().__init__(config, env)

        if not config.graph_file:
            raise ValueError("ContactGraphControlConfig.graph_file is required")
        self.graph = ContactGraph.from_file(config.graph_file, device=self.env.device)
        self.graph.validate_against_motion_lib(list(self.env.motion_lib.motion_files))
        covered, total = self.graph.coverage()
        print(
            f"ContactGraphControl: {self.graph.num_nodes} nodes, "
            f"{self.graph.num_pairs} contact pairs, "
            f"{covered}/{total} motions carry contact goals"
        )
        if covered == 0:
            raise ValueError("contact graph has no segments for any motion")

        num_envs = self.env.num_envs
        steps = self.config.num_goal_steps
        device = self.env.device

        # Raw body masks, before the per-slot pose visibility is applied. Kept
        # separate so hiding and re-revealing a pose does not resample which
        # bodies it constrains.
        self.goal_body_masks = torch.zeros(
            num_envs, steps, self.num_conditionable_bodies, 2,
            dtype=torch.bool, device=device,
        )
        self.pose_visible = torch.zeros(num_envs, steps, dtype=torch.bool, device=device)
        self.contact_visible = torch.zeros(num_envs, steps, dtype=torch.bool, device=device)
        self.goal_index = torch.zeros(num_envs, steps, dtype=torch.long, device=device)
        self.goal_valid = torch.zeros(num_envs, steps, dtype=torch.bool, device=device)
        self.prev_first_index = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        self._goal_motion_ids = torch.zeros(
            num_envs, steps, dtype=torch.long, device=device
        )
        self._time_offsets = torch.zeros(num_envs, steps, device=device)
        # [E, steps, C]; C is 2 when the dwell channels are on and 0 otherwise,
        # so the observation kernel concatenates nothing in the off case.
        self._dwell_features = torch.zeros(
            num_envs, steps, 2 if config.dwell_channels else 0, device=device
        )
        # Per-episode far-goal promotion, resampled in reset() only.
        self._promote_k = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Set by set_manual_goal() to answer an explicit query at inference.
        self._manual = None

        zone_rows, zone_cols, pair_ids = self._ground_zone_body_map()
        self._ground_zone_rows = zone_rows
        self._ground_zone_cols = zone_cols
        self._ground_pair_ids = pair_ids

        # Scatter maps taking simulated forces into the graph's own pair slots.
        # Body-body pairs join the diagnostic only when the robot is configured
        # to sense them; otherwise their thresholds stay +inf and the scored
        # subset is the ground half, exactly as before.
        pair_bodies = getattr(self.env.robot_config, "contact_pair_bodies", None)
        self._contact_maps = self.graph.contact_scatter_maps(
            body_names=list(self.env.robot_config.kinematic_info.body_names),
            ground_threshold_n=self.config.ground_contact_threshold_n,
            body_threshold_n=self.config.body_contact_threshold_n,
            pair_body_names=list(pair_bodies) if pair_bodies else None,
            device=device,
        )
        # A slot with a finite threshold is one this robot can actually measure;
        # scoring an IoU over slots that can never fire would put every
        # unobservable body-body pair permanently in the union's denominator.
        self._scored_slots = torch.isfinite(self._contact_maps["thresholds"])
        print(
            f"ContactGraphControl: scoring the reached-goal diagnostic over "
            f"{int(self._scored_slots.sum())}/{self.graph.num_pairs} contact pairs "
            f"({self._contact_maps['num_body_body_slots']} of them body-body)"
        )

        # Contact-event history: measured past in the goal's own vocabulary.
        # getattr defaults keep resolved configs frozen before these fields
        # existed loading cleanly with the tracker off.
        num_events = int(getattr(config, "num_history_events", 0) or 0)
        self._pelvis_body_index = list(
            self.env.robot_config.kinematic_info.body_names
        ).index("Pelvis")
        if num_events > 0:
            self._event_tracker = ContactEventTracker(
                num_envs=num_envs,
                num_pairs=self.graph.num_pairs,
                num_events=num_events,
                min_dwell_s=float(getattr(config, "history_min_dwell_s", 0.3)),
                time_clip_s=float(getattr(config, "history_time_clip_s", 10.0)),
                orient_margin=0.15,  # extract_contact_configs.orientation_bins
                device=device,
            )
            print(
                f"ContactGraphControl: contact-event history on — "
                f"{num_events} tokens x {self._event_tracker.feature_size} features, "
                f"min dwell {self._event_tracker.min_dwell_s:.2f} s"
            )
        else:
            self._event_tracker = None
        self._history_update_pending = False
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _ground_zone_body_map(self):
        """Scatter map pooling per-body ground forces into the graph's zones.

        Kept for the human-readable zone reporting the query tools do
        (``_ground_zone_names`` names the zones a rollout ended in). The
        reached-goal diagnostic itself now runs off ``_contact_maps``, which
        covers the body-body pairs too when the robot senses them.
        """
        zone_order, zones = self.graph.zone_definition()
        body_names = list(self.env.robot_config.kinematic_info.body_names)
        rows, cols, zone_names, pair_ids = [], [], [], []
        for zone in zone_order:
            pair = f"{zone}:G"
            # Only zones this graph actually names: a graph built with a
            # different zone set still gets a well-defined (smaller) diagnostic
            # instead of an index error at construction.
            if pair not in self.graph.pair_names:
                continue
            members = [body_names.index(b) for b in zones[zone] if b in body_names]
            if not members:
                continue
            zone_index = len(zone_names)
            zone_names.append(zone)
            pair_ids.append(self.graph.pair_names.index(pair))
            for body in members:
                rows.append(zone_index)
                cols.append(body)
        self._ground_zone_names = zone_names
        device = self.env.device
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(cols, dtype=torch.long, device=device),
            torch.tensor(pair_ids, dtype=torch.long, device=device),
        )

    # ------------------------------------------------------------------ #
    # Goal schedule
    # ------------------------------------------------------------------ #
    def _refresh_goal_indices(self) -> None:
        if self._manual is not None:
            self._refresh_manual_goals()
            return
        motion_ids = self.env.motion_manager.motion_ids
        now = self.env.motion_manager.motion_times
        indices, valid = self.graph.next_goal_indices(
            motion_ids,
            now,
            self.config.num_goal_steps,
            min_lead_s=self.config.min_lead_s,
            include_current=self.config.include_current_segment,
            promote_k=self._promote_k if self.config.far_goal_prob > 0.0 else None,
        )
        self.goal_index = indices
        self.goal_valid = valid
        self._gathered = self.graph.gather(motion_ids, indices)
        # Goal poses come from the clip the environment is playing.
        self._goal_motion_ids = motion_ids.unsqueeze(-1).expand_as(indices)
        target_times = self._gathered["t_hold"]
        # Padded slots hold +inf; make them a finite time so the motion-lib query
        # and the marker code stay well defined. They are masked out anyway.
        motion_lengths = self.env.motion_lib.get_motion_length(motion_ids).unsqueeze(-1)
        self.target_times = torch.where(
            torch.isfinite(target_times), target_times, motion_lengths
        ).minimum(motion_lengths)
        offsets = self.target_times - now.unsqueeze(-1)
        # An exhausted schedule clamps to the last segment, whose hold is behind
        # us, so the raw offset goes negative. The transformer masks those tokens
        # out, but the privileged encoder is a plain MLP that concatenates the
        # time channel unmasked -- so zero it here rather than feed the encoder a
        # value that never occurs on a live goal.
        #
        # `include_current_segment` creates a second source of negative offsets,
        # and this one is on a *valid* slot: the segment you are inside can have
        # its hold frame behind you. Clamp rather than zero -- 0 means "be there
        # now", which is exactly right, whereas a negative deadline is the value
        # round 9 §3 showed the network reads as "the command is about to
        # change".
        self._time_offsets = torch.where(
            self.goal_valid, offsets.clamp(min=0.0), torch.zeros_like(offsets)
        )
        self._dwell_features = self._compute_dwell_features(
            self._gathered["t_hold"], self._gathered["t_end"], now
        )

    def _compute_dwell_features(
        self, t_hold: Tensor, t_end: Tensor, now: Tensor
    ) -> Tensor:
        """``[E, steps, C]`` timing channels, scaled to [0, 1] and validity-gated.

        ``C == 0`` when the feature is off, which makes the observation block
        byte-identical to pre-v10_1 runs without a second code path.

        * ``hold_duration`` = ``t_end - t_hold``: how long the configuration
          persists past the frame being commanded. Defined for every slot,
          including ones far in the future.
        * ``dwell_remaining`` = ``clamp(t_end - now, 0)``: how much of it is
          left. Only non-trivial for a slot you are actually inside, which is
          why this pairs with ``include_current_segment``.

        Both are clamped to ``history_time_clip_s`` and divided by it, matching
        how ``ContactEventTracker`` scales its own age/dwell channels, so every
        time value in the contact block shares one unit.
        """
        if not self.config.dwell_channels:
            return t_hold.new_zeros((*t_hold.shape, 0))
        clip = float(self.config.history_time_clip_s)
        # Padding carries +inf; nan_to_num keeps the clamp well defined before
        # the validity gate zeroes those slots anyway.
        end = torch.nan_to_num(t_end, posinf=0.0, neginf=0.0)
        hold = torch.nan_to_num(t_hold, posinf=0.0, neginf=0.0)
        duration = (end - hold).clamp(0.0, clip) / clip
        remaining = (end - now.unsqueeze(-1)).clamp(0.0, clip) / clip
        gate = self.goal_valid.to(duration.dtype)
        return torch.stack([duration * gate, remaining * gate], dim=-1)

    # ------------------------------------------------------------------ #
    # Manual goals (inference)
    # ------------------------------------------------------------------ #
    def set_manual_goal(
        self,
        node_ids: Tensor,
        pose_motion_ids: Tensor,
        pose_times: Tensor,
        time_offsets: Tensor,
        pose_visible: Tensor,
        contact_visible: Tensor,
        hold_seconds: Optional[Tensor] = None,
    ) -> None:
        """Drive the goal from an explicit query instead of the clip's schedule.

        This is what makes a trained student answerable: "reach *this* contact
        configuration, holding *this* pose, within *this* long", where the pose is
        named as (clip, time) because that is the only pose representation the
        motion library can serve.  Every argument is ``[num_envs, num_goal_steps]``.

        ``hold_seconds`` is ``[num_envs, num_goal_steps]`` seconds the caller
        wants each configuration *held* once reached, and it is what feeds the
        dwell channels. It matters more than it looks: without it the manual
        path would report a hold duration of **zero**, because a manual goal has
        no segment and ``t_start = t_end = t_hold``. Every pinned probe would
        then be commanding "stay 0 seconds" -- the exact opposite of what a 12 s
        hold probe means -- and the dwell feature would look broken when it was
        the driver that was wrong. Pass the *remaining* hold at issue time and
        re-issue as the plan advances; the countdown between issues is handled
        here. None keeps the pre-v10_1 behaviour (duration 0), which is correct
        only when the dwell channels are off.

        Call :meth:`clear_manual_goal` to return to the clip schedule.

        Raises:
            RuntimeError: if called before the first ``reset()``. ``step()``
                returns early until then, so the deadline would silently never
                count down while the context still looked correct.
            ValueError: on a wrong shape, or a node id outside the graph. ``-1``
                is the padding sentinel; anything above ``num_nodes`` would be an
                out-of-bounds gather, which on CUDA surfaces as an async
                device-side assert at some unrelated later call.
        """
        if not self._initialized:
            raise RuntimeError(
                "set_manual_goal() before the first reset(): step() returns early "
                "until the component is initialised, so the goal's deadline would "
                "never count down. Call env.reset() first."
            )
        steps = self.config.num_goal_steps
        expected = (self.env.num_envs, steps)
        for name, tensor in (
            ("node_ids", node_ids),
            ("pose_motion_ids", pose_motion_ids),
            ("pose_times", pose_times),
            ("time_offsets", time_offsets),
            ("pose_visible", pose_visible),
            ("contact_visible", contact_visible),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")
        if hold_seconds is not None and tuple(hold_seconds.shape) != expected:
            raise ValueError(
                f"hold_seconds must be {expected}, got {tuple(hold_seconds.shape)}"
            )
        if hold_seconds is None and self.config.dwell_channels:
            log.warning(
                "set_manual_goal() without hold_seconds while dwell_channels is on: "
                "every commanded hold will read as 'stay 0 s'. Pass the remaining "
                "hold time from the plan."
            )
        if node_ids.numel():
            if int(node_ids.max()) >= self.graph.num_nodes or int(node_ids.min()) < -1:
                raise ValueError(
                    f"node_ids span [{int(node_ids.min())}, {int(node_ids.max())}], "
                    f"outside [-1, {self.graph.num_nodes - 1}] (-1 = unspecified slot)"
                )
        device = self.env.device
        # Cloned, not just moved: `.to(device).long()` is the identity when dtype
        # and device already match, which would leave `_manual` aliasing the
        # caller's buffers -- and `_refresh_manual_goals` re-reads them every
        # step, so a later in-place write would change the live goal.
        self._manual = {
            "node": node_ids.to(device).long().clone(),
            "pose_motion": pose_motion_ids.to(device).long().clone(),
            "pose_time": pose_times.to(device).float().clone(),
            "offset": time_offsets.to(device).float().clone(),
            "pose_visible": pose_visible.to(device).bool().clone(),
            "contact_visible": contact_visible.to(device).bool().clone(),
            # `total` is what was asked for and stays put; `remaining` counts
            # down between re-issues so the channel reads 12 -> 0 over a hold.
            "hold_total": (
                hold_seconds.to(device).float().clone()
                if hold_seconds is not None
                else torch.zeros_like(time_offsets.to(device).float())
            ),
            "hold_remaining": (
                hold_seconds.to(device).float().clone()
                if hold_seconds is not None
                else torch.zeros_like(time_offsets.to(device).float())
            ),
        }
        # A manual query specifies bodies explicitly: reveal all of them where
        # the pose half is on, so the goal is the pose that was asked for.
        self.goal_body_masks[:] = True
        self.pose_visible = self._manual["pose_visible"].clone()
        self.contact_visible = self._manual["contact_visible"].clone()
        self._refresh_goal_indices()

    def clear_manual_goal(self) -> None:
        """Return to the clip schedule, undoing what a manual goal forced.

        Dropping ``_manual`` is not enough on its own: ``set_manual_goal`` forces
        every body mask on and the manual path never advances
        ``prev_first_index``. Left alone, the clip schedule would resume with a
        fully-revealed pose and compute its slot-roll against an index from
        before the manual phase.
        """
        if self._manual is None:
            return
        self._manual = None
        # Manual goals mean an external driver (probe script, viz panel) has
        # been steering the sim; the contact history accumulated under it does
        # not describe whatever episode resumes now.
        if self._event_tracker is not None:
            self._event_tracker.reset_all()
        self._refresh_goal_indices()
        steps = self.config.num_goal_steps
        num_envs = self.env.num_envs
        bodies, pose_visible, contact_visible = self._sample_masks(num_envs * steps)
        self.goal_body_masks = bodies.view(
            num_envs, steps, self.num_conditionable_bodies, 2
        )
        self.pose_visible = pose_visible.view(num_envs, steps)
        self.contact_visible = contact_visible.view(num_envs, steps)
        self.prev_first_index = self.goal_index[:, 0]
        self._enforce_first_goal()

    def _refresh_manual_goals(self) -> None:
        manual = self._manual
        node = manual["node"]
        valid = node >= 0
        safe_node = node.clamp(min=0)
        self.goal_index = torch.zeros_like(node)
        self.goal_valid = valid
        self._goal_motion_ids = manual["pose_motion"]
        motion_lengths = self.env.motion_lib.get_motion_length(
            self._goal_motion_ids.reshape(-1)
        ).view_as(manual["pose_time"])
        self.target_times = manual["pose_time"].minimum(motion_lengths)
        self._time_offsets = manual["offset"]
        # A manual goal has no segment, so one is synthesised: it starts at the
        # commanded frame and lasts as long as the caller asked it to be held.
        # With hold_seconds omitted this collapses to the pre-v10_1 zero-length
        # segment.
        self._gathered = {
            "node": safe_node,
            "t_start": self.target_times,
            "t_end": self.target_times + manual["hold_total"],
            "t_hold": self.target_times,
            "contact": self.graph.node_contact[safe_node],
            "orient": self.graph.node_orient[safe_node],
        }
        if self.config.dwell_channels:
            clip = float(self.config.history_time_clip_s)
            gate = valid.to(self.target_times.dtype)
            self._dwell_features = torch.stack(
                [
                    (manual["hold_total"].clamp(0.0, clip) / clip) * gate,
                    (manual["hold_remaining"].clamp(0.0, clip) / clip) * gate,
                ],
                dim=-1,
            )
        else:
            self._dwell_features = self.target_times.new_zeros(
                (*self.target_times.shape, 0)
            )
        # Cloned, not aliased: the stored query must survive any in-place write
        # to the live visibility buffers.
        self.pose_visible = manual["pose_visible"].clone()
        self.contact_visible = manual["contact_visible"].clone()

    # ------------------------------------------------------------------ #
    # Mask sampling
    # ------------------------------------------------------------------ #
    def _sample_masks(self, num_rows: int):
        """Sample ``num_rows`` independent goal specifications."""
        device = self.env.device
        bodies = self._sample_new_body_masks(num_rows).view(
            num_rows, self.num_conditionable_bodies, 2
        )
        full = (
            torch.rand(num_rows, device=device) < self.config.full_pose_prob
        ).view(num_rows, 1, 1)
        bodies = torch.where(full, torch.ones_like(bodies), bodies)
        pose_visible = torch.rand(num_rows, device=device) < self.config.pose_visible_prob
        contact_visible = (
            torch.rand(num_rows, device=device) < self.config.contact_visible_prob
        )
        return bodies, pose_visible, contact_visible

    def _enforce_first_goal(self) -> None:
        """Never let the nearest goal specify nothing at all.

        Written branch-free: ``contact | ~pose`` reveals the contact set exactly
        when the pose is hidden, and leaves every other case alone. The obvious
        ``if blank.any()`` guard would cost a host synchronisation on every
        environment step.
        """
        if not self.config.require_first_goal_specified:
            return
        self.contact_visible[:, 0] |= ~self.pose_visible[:, 0]

    def reset(self, env_ids: Tensor):
        """Resample the whole goal specification for the given environments."""
        MimicControl.reset(self, env_ids)
        if self._event_tracker is not None:
            # Cleared rows re-open their first segment from the next
            # observation build, so history is episode-local by construction.
            self._event_tracker.reset(env_ids)
        # Far-goal promotion is drawn once per episode and then held. A window
        # that jumped around mid-episode would be a different, and worse,
        # intervention: the policy could never tell a distant command from a
        # near one that is about to be replaced. Drawn before the refresh below
        # so the first schedule of the episode already carries it.
        self._resample_promotion(env_ids)
        # Unconditional: the schedule is a pure function of the motion state, and
        # a zero-length reset still has to leave the tables populated for the
        # context build that follows it.
        self._refresh_goal_indices()
        if self._manual is not None:
            # A manual query outlives episode boundaries -- resampling here would
            # silently replace the goal that was asked for.
            self._initialized = True
            return
        if len(env_ids) == 0:
            self._initialized = True
            return

        steps = self.config.num_goal_steps
        rows = len(env_ids) * steps
        bodies, pose_visible, contact_visible = self._sample_masks(rows)
        self.goal_body_masks[env_ids] = bodies.view(
            len(env_ids), steps, self.num_conditionable_bodies, 2
        )
        self.pose_visible[env_ids] = pose_visible.view(len(env_ids), steps)
        self.contact_visible[env_ids] = contact_visible.view(len(env_ids), steps)
        self.prev_first_index[env_ids] = self.goal_index[env_ids, 0]
        self._enforce_first_goal()
        self._initialized = True

    def _resample_promotion(self, env_ids: Tensor) -> None:
        """Draw a per-episode far-goal skip for ``env_ids``.

        Zero for every environment when ``far_goal_prob`` is 0, which is the
        default and reproduces the original schedule.
        """
        if self.config.far_goal_prob <= 0.0 or len(env_ids) == 0:
            if self.config.far_goal_prob <= 0.0:
                self._promote_k.zero_()
            return
        device = self.env.device
        count = len(env_ids)
        fire = torch.rand(count, device=device) < self.config.far_goal_prob
        skip = torch.randint(
            1, max(int(self.config.far_goal_max_skip), 1) + 1, (count,), device=device
        )
        self._promote_k[env_ids] = torch.where(
            fire, skip, torch.zeros_like(skip)
        ).long()

    def step(self):
        """Advance the goal schedule, keeping each goal's mask attached to it."""
        MimicControl.step(self)
        # step() runs exactly once per env step (post-physics, pre-context);
        # the flag makes the history advance exactly once even though
        # populate_context can run again in the same step (probe drivers
        # rebuild observations after re-issuing goals).
        self._history_update_pending = True
        if not self._initialized:
            return

        if self._manual is not None:
            # A manual goal is a deadline, not a clip position: count it down so
            # the "time to target" the policy sees means the same thing it did
            # during training.
            self._manual["offset"] = (self._manual["offset"] - self.env.dt).clamp(
                min=self.config.min_lead_s
            )
            # The requested dwell counts down to zero and stops; unlike the
            # deadline it has no floor, because "0 s left" is a meaningful
            # command and `min_lead_s` is a property of the deadline only.
            self._manual["hold_remaining"] = (
                self._manual["hold_remaining"] - self.env.dt
            ).clamp(min=0.0)
            self._refresh_goal_indices()
            return

        previous = self.prev_first_index.clone()
        self._refresh_goal_indices()
        steps = self.config.num_goal_steps
        # No early-out on "nothing advanced": the check is a host synchronisation
        # and with a thousand environments some goal advances almost every step,
        # so the branch would pay for itself roughly never. delta == 0 makes the
        # gather below an identity and need_new all False.
        delta = (self.goal_index[:, 0] - previous).clamp(min=0, max=steps)
        device = self.env.device
        source = torch.arange(steps, device=device).unsqueeze(0) + delta.unsqueeze(1)
        need_new = source >= steps
        source = source.clamp(max=steps - 1)

        gather_bodies = source.view(*source.shape, 1, 1).expand(
            -1, -1, self.num_conditionable_bodies, 2
        )
        rolled_bodies = torch.gather(self.goal_body_masks, 1, gather_bodies)
        rolled_pose = torch.gather(self.pose_visible, 1, source)
        rolled_contact = torch.gather(self.contact_visible, 1, source)

        num_envs = self.env.num_envs
        fresh_bodies, fresh_pose, fresh_contact = self._sample_masks(num_envs * steps)
        fresh_bodies = fresh_bodies.view(
            num_envs, steps, self.num_conditionable_bodies, 2
        )
        fresh_pose = fresh_pose.view(num_envs, steps)
        fresh_contact = fresh_contact.view(num_envs, steps)

        self.goal_body_masks = torch.where(
            need_new.view(num_envs, steps, 1, 1), fresh_bodies, rolled_bodies
        )
        self.pose_visible = torch.where(need_new, fresh_pose, rolled_pose)
        self.contact_visible = torch.where(need_new, fresh_contact, rolled_contact)
        self.prev_first_index = self.goal_index[:, 0]
        self._enforce_first_goal()

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #
    def _marker_motion_ids(self, env_indices: Tensor, slot_indices: Tensor) -> Tensor:
        """The clip each goal pose actually comes from.

        Not the clip being played: a manual query can take its goal pose from any
        clip in the library, and the base implementation would then draw the
        *playing* clip's frame at the goal's timestamp -- a plausible-looking
        target that is not the one the policy was given.
        """
        return self._goal_motion_ids[env_indices, slot_indices]

    def _marker_time_to_target(
        self, env_indices: Tensor, slot_indices: Tensor, target_times: Tensor
    ) -> Tensor:
        """The offset the policy is actually conditioned on.

        In manual mode the goal is a countdown, not a position in the played
        clip, so differencing against ``motion_times`` would colour the markers
        by an unrelated quantity.
        """
        return self._time_offsets[env_indices, slot_indices]

    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Marker states, plus the ghost-character pose where one exists.

        The ghost is a visualization-only second robot the simulator can spawn
        (``SimulatorConfig.ghost_robot``); posing it here keeps it in lockstep
        with the goal markers — same update point, same goal slot, same spawn
        offset — so what the viewer sees beside the character is exactly the
        pose the sphere markers are sampled from.
        """
        markers_state = super().get_markers_state()
        self._update_ghost_char()
        return markers_state

    def _update_ghost_char(self) -> None:
        """Pose the simulator's ghost robot as the nearest goal's held pose.

        Deliberately shows slot 0's pose whenever the slot carries a valid
        goal, even when the pose half is masked from the policy (a
        contact-only goal still *has* a defining pose in the graph, and the
        analyst wants to see it). Whether the policy was actually shown the
        pose is recorded by the probe tools (``pose_given``).
        """
        simulator = getattr(self.env, "simulator", None)
        if simulator is None or not getattr(simulator, "ghost_enabled", False):
            return
        if not self._initialized:
            return

        motion_ids = self._goal_motion_ids[:, 0]
        motion_times = self.target_times[:, 0]
        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_state.rigid_body_pos
        )
        reset_state = ResetState.from_robot_state(ref_state)
        # Per-env offset: XY is shared by all bodies and Z is shared by all
        # bodies, so any body's row carries the whole correction.
        reset_state.root_pos = reset_state.root_pos + offset[:, 0]
        simulator.set_ghost_state(reset_state, active=self.goal_valid[:, 0])

    # ------------------------------------------------------------------ #
    # Context
    # ------------------------------------------------------------------ #
    def _effective_body_masks(self) -> Tensor:
        """Body masks after per-slot pose visibility and slot validity."""
        keep = (self.pose_visible & self.goal_valid).view(
            self.env.num_envs, self.config.num_goal_steps, 1, 1
        )
        return self.goal_body_masks & keep

    def _measured_contact_state(self, ctx: EnvContext) -> Optional[Tensor]:
        """Binary measured contact over the graph's pair vocabulary, ``[E, P]``.

        ``None`` when this backend reports no contact forces at all. This is
        the single source both the reached-goal diagnostic and the
        contact-event history read, so they cannot disagree on what "in
        contact" means.
        """
        ground = getattr(ctx.current, "rigid_body_ground_forces", None)
        if ground is None:
            # Backends without a terrain-filtered column: fall back to the net
            # per-body force, which over-reports (it also counts body-body
            # contact) but is the only thing available.
            ground = getattr(ctx.current, "rigid_body_contact_forces", None)
        if ground is None:
            return None
        return (
            compute_contact_state_obs(
                ground_forces=ground,
                pair_forces=getattr(
                    ctx.current, "rigid_body_pair_contact_forces", None
                ),
                ground_slot=self._contact_maps["ground_slot"],
                ground_body=self._contact_maps["ground_body"],
                pair_slot=self._contact_maps["pair_slot"],
                pair_body_a=self._contact_maps["pair_body_a"],
                pair_body_b=self._contact_maps["pair_body_b"],
                thresholds=self._contact_maps["thresholds"],
                num_pairs=self._contact_maps["num_pairs"],
            )
            > 0.5
        )

    def _update_contact_history(
        self, ctx: EnvContext, current: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Advance the event tracker and return ``(features, valid)``.

        The tracker advances exactly once per env step (the flag set by
        :meth:`step`); any further observation rebuild within the same step —
        probe drivers re-issue goals and recompute observations — only
        initializes rows freshly cleared by a reset.
        """
        num_envs = self.env.num_envs
        tracker = self._event_tracker
        if tracker is None:
            return (
                torch.zeros(num_envs, 0, 1, device=self.env.device),
                torch.zeros(num_envs, 0, dtype=torch.bool, device=self.env.device),
            )
        contact = current
        if contact is None:
            contact = torch.zeros(
                num_envs, self.graph.num_pairs, dtype=torch.bool,
                device=self.env.device,
            )
        pelvis_rot = ctx.current.rigid_body_rot[:, self._pelvis_body_index]
        if self._history_update_pending:
            self._history_update_pending = False
            tracker.update(contact, pelvis_rot, float(self.env.dt))
        else:
            tracker.initialize_fresh(contact, pelvis_rot)
        return tracker.features()

    def _contact_configuration_iou(self, current: Optional[Tensor]) -> Tensor:
        """IoU between the measured contact configuration and the nearest goal's.

        Scored over the pairs this robot can actually sense. With
        ``RobotConfig.contact_pair_bodies`` set that is the whole vocabulary --
        which matters, because the body-body half is what distinguishes crow
        from firefly from eight-angle, all of which are "two hands" on the
        ground. Without it the scored subset is the 15 ground pairs, as before.
        """
        num_envs = self.env.num_envs
        if current is None:
            return torch.zeros(num_envs, device=self.env.device)

        goal = (self._gathered["contact"][:, 0] > 0.5) & self.goal_valid[:, 0:1]
        scored = self._scored_slots.unsqueeze(0)
        current = current & scored
        goal = goal & scored

        intersection = (current & goal).sum(dim=-1).float()
        union = (current | goal).sum(dim=-1).float()
        return torch.where(union > 0, intersection / union.clamp(min=1.0), torch.ones_like(union))

    def _goal_pose_error(
        self, ctx: EnvContext, ref_pos: Tensor, ref_rot: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Distance to the commanded *pose*, in the student's own goal frame.

        ``_contact_configuration_iou`` scores the contact half, and at a
        degenerate node -- standing, single-leg, four-point -- every member has
        the same contact set, so it reads 1.00 whatever pose is being held.
        Nothing in the project scored the other half, which is how "commanded
        Warrior III, performs Lord of the Dance" survived four rounds of review
        (``notes/Student_v7_improvement_investigation.MD`` §5.5).

        This is the same arithmetic ``data/scripts/score_probe_pose.py`` does
        offline, so the live number and the probe numbers are commensurable:
        the mean over the conditionable bodies of the distance between current
        and reference positions, each taken pelvis-relative and rotated into
        the heading-normalised frame of its own root. Only bodies whose
        *position* mask is set are counted, and only slot 0 -- the nearest goal
        -- is scored.

        Rows whose nearest goal has no visible pose carry no measurement. They
        are filled with the mean over the rows that do, so the metric averages
        to the conditional mean over commanded poses instead of being diluted
        toward zero by rows that were never asked for a pose. The second
        return value is 1.0 exactly on the rows that *were* measured, so the
        log says how much of the batch the first number rests on.
        """
        current_pos = ctx.current.rigid_body_pos[:, self.conditionable_body_ids]
        goal_pos = ref_pos[:, 0][:, self.conditionable_body_ids]
        root_current = ctx.current.rigid_body_pos[:, self._pelvis_body_index]
        root_goal = ref_pos[:, 0, self._pelvis_body_index]

        heading_current = calc_heading_quat_inv(
            ctx.current.rigid_body_rot[:, self._pelvis_body_index], w_last=True
        )
        # The reference frame's own heading: the goal pose is scored as a shape,
        # not as a compass bearing, exactly as the goal observation presents it.
        heading_goal = calc_heading_quat_inv(
            ref_rot[:, 0, self._pelvis_body_index], w_last=True
        )
        num_bodies = current_pos.shape[1]
        local_current = quat_rotate(
            heading_current.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4),
            (current_pos - root_current.unsqueeze(1)).reshape(-1, 3),
            w_last=True,
        ).view(-1, num_bodies, 3)
        local_goal = quat_rotate(
            heading_goal.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4),
            (goal_pos - root_goal.unsqueeze(1)).reshape(-1, 3),
            w_last=True,
        ).view(-1, num_bodies, 3)

        distance = (local_current - local_goal).norm(dim=-1)
        # Slot 0's per-body position mask (index 0 of the [position, rotation]
        # pair), already gated by pose visibility and slot validity.
        body_mask = self._effective_body_masks()[:, 0, :, 0].float()
        counted = body_mask.sum(dim=-1)
        error = (distance * body_mask).sum(dim=-1) / counted.clamp(min=1.0)

        measured = counted > 0
        if bool(measured.any()):
            error = torch.where(measured, error, error[measured].mean())
        else:
            error = torch.zeros_like(error)
        return error, measured.float()

    def populate_context(self, ctx: EnvContext) -> None:
        """Populate ``ctx.mimic``, ``ctx.masked_mimic`` and ``ctx.contact_goal``."""
        MimicControl.populate_context(self, ctx)

        if not self._initialized:
            self._refresh_goal_indices()

        num_envs = self.env.num_envs
        steps = self.config.num_goal_steps

        # Goal poses may come from a different clip than the one being played,
        # which is what lets a query name any pose in the library.
        flat_motion_ids = self._goal_motion_ids.reshape(-1)
        flat_times = self.target_times.reshape(-1)
        target_state = self.env.motion_lib.get_motion_state(flat_motion_ids, flat_times)

        num_bodies = target_state.rigid_body_pos.shape[1]
        ref_pos = target_state.rigid_body_pos.view(
            num_envs, steps, num_bodies, 3
        ).clone()
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_pos[:, 0, :, :]
        )
        ref_pos += offset.unsqueeze(1)
        ref_rot = target_state.rigid_body_rot.view(num_envs, steps, num_bodies, 4)

        body_masks = self._effective_body_masks()
        self.masked_mimic_target_bodies_masks = body_masks.reshape(num_envs, -1)
        contact_visible = self.contact_visible & self.goal_valid
        # A token is worth attending to when either half of the goal is given.
        self.masked_mimic_target_poses_masks = (
            body_masks.any(dim=-1).any(dim=-1) | contact_visible
        )

        time_offsets = self._time_offsets

        ctx.masked_mimic = MaskedMimicContext(
            mimic=ctx.mimic,
            ref_pos=ref_pos,
            ref_rot=ref_rot,
            target_times=self.target_times,
            time_offsets=time_offsets,
            target_poses_masks=self.masked_mimic_target_poses_masks,
            target_bodies_masks=self.masked_mimic_target_bodies_masks,
        )

        visible = contact_visible.float().unsqueeze(-1)
        contact_spec = self._gathered["contact"] * visible
        orient_spec = torch.nn.functional.one_hot(
            self._gathered["orient"], self.graph.num_orientations
        ).float() * visible
        node_ids = torch.where(
            self.goal_valid, self._gathered["node"], torch.full_like(self._gathered["node"], -1)
        )

        current_contact = self._measured_contact_state(ctx)
        history_features, history_valid = self._update_contact_history(
            ctx, current_contact
        )
        # "The measured configuration just committed a change" — the tracker's
        # per-step debounced make/break flag, exposed so a policy (the FSQ
        # student's chunk refresh) can react to it as an event.
        if self._event_tracker is not None:
            event_commit = self._event_tracker.last_commit
        else:
            event_commit = torch.zeros(
                num_envs, dtype=torch.bool, device=self.env.device
            )

        pose_error, pose_error_visible = self._goal_pose_error(ctx, ref_pos, ref_rot)

        ctx.contact_goal = ContactGoalContext(
            contact_spec=contact_spec,
            orient_spec=orient_spec,
            visible=contact_visible.float(),
            time_offsets=time_offsets,
            dwell_features=self._dwell_features,
            node_ids=node_ids,
            reached=self._contact_configuration_iou(current_contact),
            pose_error=pose_error,
            pose_error_visible=pose_error_visible,
            history_features=history_features,
            history_valid=history_valid,
            event_commit=event_commit,
        )
