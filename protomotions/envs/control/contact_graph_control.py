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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.components.contact_graph import ContactGraph
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
        indices, valid = self.graph.next_goal_indices(
            motion_ids,
            self.env.motion_manager.motion_times,
            self.config.num_goal_steps,
            min_lead_s=self.config.min_lead_s,
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
        offsets = self.target_times - self.env.motion_manager.motion_times.unsqueeze(-1)
        # An exhausted schedule clamps to the last segment, whose hold is behind
        # us, so the raw offset goes negative. The transformer masks those tokens
        # out, but the privileged encoder is a plain MLP that concatenates the
        # time channel unmasked -- so zero it here rather than feed the encoder a
        # value that never occurs on a live goal.
        self._time_offsets = torch.where(self.goal_valid, offsets, torch.zeros_like(offsets))

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
    ) -> None:
        """Drive the goal from an explicit query instead of the clip's schedule.

        This is what makes a trained student answerable: "reach *this* contact
        configuration, holding *this* pose, within *this* long", where the pose is
        named as (clip, time) because that is the only pose representation the
        motion library can serve.  Every argument is ``[num_envs, num_goal_steps]``.

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
        self._gathered = {
            "node": safe_node,
            "t_start": self.target_times,
            "t_end": self.target_times,
            "t_hold": self.target_times,
            "contact": self.graph.node_contact[safe_node],
            "orient": self.graph.node_orient[safe_node],
        }
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

    def step(self):
        """Advance the goal schedule, keeping each goal's mask attached to it."""
        MimicControl.step(self)
        if not self._initialized:
            return

        if self._manual is not None:
            # A manual goal is a deadline, not a clip position: count it down so
            # the "time to target" the policy sees means the same thing it did
            # during training.
            self._manual["offset"] = (self._manual["offset"] - self.env.dt).clamp(
                min=self.config.min_lead_s
            )
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

    # ------------------------------------------------------------------ #
    # Context
    # ------------------------------------------------------------------ #
    def _effective_body_masks(self) -> Tensor:
        """Body masks after per-slot pose visibility and slot validity."""
        keep = (self.pose_visible & self.goal_valid).view(
            self.env.num_envs, self.config.num_goal_steps, 1, 1
        )
        return self.goal_body_masks & keep

    def _contact_configuration_iou(self, ctx: EnvContext) -> Tensor:
        """IoU between the measured contact configuration and the nearest goal's.

        Scored over the pairs this robot can actually sense. With
        ``RobotConfig.contact_pair_bodies`` set that is the whole vocabulary --
        which matters, because the body-body half is what distinguishes crow
        from firefly from eight-angle, all of which are "two hands" on the
        ground. Without it the scored subset is the 15 ground pairs, as before.
        """
        num_envs = self.env.num_envs
        ground = getattr(ctx.current, "rigid_body_ground_forces", None)
        if ground is None:
            # Backends without a terrain-filtered column: fall back to the net
            # per-body force, which over-reports (it also counts body-body
            # contact) but is the only thing available.
            ground = getattr(ctx.current, "rigid_body_contact_forces", None)
        if ground is None:
            return torch.zeros(num_envs, device=self.env.device)

        current = (
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

        goal = (self._gathered["contact"][:, 0] > 0.5) & self.goal_valid[:, 0:1]
        scored = self._scored_slots.unsqueeze(0)
        current = current & scored
        goal = goal & scored

        intersection = (current & goal).sum(dim=-1).float()
        union = (current | goal).sum(dim=-1).float()
        return torch.where(union > 0, intersection / union.clamp(min=1.0), torch.ones_like(union))

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

        ctx.contact_goal = ContactGoalContext(
            contact_spec=contact_spec,
            orient_spec=orient_spec,
            visible=contact_visible.float(),
            time_offsets=time_offsets,
            node_ids=node_ids,
            reached=self._contact_configuration_iou(ctx),
        )
