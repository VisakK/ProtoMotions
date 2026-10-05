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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from protomotions.components.contact_graph import ContactGraph
from protomotions.envs.control.contact_event_tracker import ContactEventTracker
from protomotions.envs.control.contact_targets import ContactTargets
from protomotions.utils.release_identity import file_sha256, load_release, require_artifact
from protomotions.envs.control.support_penalty import ChargedLoadEMA, unwanted_support
from protomotions.envs.control.physics_terms import (
    PhysicsTables,
    corner_slip_speed,
    lean_shortfall,
    patch_points,
    polygon_margin,
    swing_charged_load,
    whole_body_com,
)
from protomotions.envs.obs.contact_state import compute_contact_slot_forces
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


def _library_layout(motion_lib) -> Tuple[Optional[Tensor], Optional[float]]:
    """``(frame counts [M] on the CPU, the single frame rate)`` of a motion library; ``(None, None)``
    for a stub without them. A library mixing frame rates returns no rate: the frame-indexed tables
    (graph v2, physics tables, contact targets) then refuse it through their own checks."""
    frames = getattr(motion_lib, "motion_num_frames", None)
    dt = getattr(motion_lib, "motion_dt", None)
    if frames is None or dt is None or not torch.is_tensor(dt) or dt.numel() == 0:
        return None, None
    rates = torch.round(1.0 / dt.double()).unique()
    return frames.detach().long().cpu(), (float(rates[0]) if rates.numel() == 1 else None)


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
        interval_schedule: With ``include_current_segment``, pick goals by
            segment interval (the segment the clip is inside, else the first
            that has not begun) instead of by hold time at ``now +
            min_lead_s``. The legacy rule steps slot 0 backwards at 52 hold
            entries over 42 of the 180 expert60 variants; this one is monotonic
            by construction (``ContactGraph.next_goal_indices(interval=True)``).
            Default False keeps every existing experiment unchanged.
        support_clear_height: Unwanted-support penalty
            (``protomotions/envs/control/support_penalty.py``,
            ``expert_revist/contact_reward/README.MD``): a zone must be kept
            unloaded only if the commanded hold's ground set leaves it out **and**
            the reference keeps its lowest joint centre above this height (m).
            0.25 m was the lowest clearance with no false positive on the ft_a
            calibration (0.15 m still charged a float-biased foot).
        support_load_ref_frac: Charged ground load, as a fraction of body weight,
            at which the penalty saturates. 0.1 is the scale the pressure runs
            settled on; full body weight made the same kind of term ~30x too weak
            (``notes/Pressure_supervision_design.MD`` 8.6).
        support_body_weight_n: Body weight (N) that fraction is taken of.
        support_exclude_motions: Substrings of motion names never charged (all
            hold-duration variants of a stem match its base name). For
            references that cannot be performed with the support they label.
        support_ema_tau_s: Time constant (s) of an exponential moving average
            applied to the charged load *before* the saturating clamp
            (``support_penalty.ChargedLoadEMA``). 0.0, the default, is the
            per-frame clamp ft_b trained with, bit for bit. That clamp prices
            how often a free zone is down, not how much it carries, and ft_b
            learned to tap a foot at 3-4 Hz to exploit it. 0.25 s charges a
            tapping foot what a resting one pays while a held lift stays nearly
            free (``expert_revist/ft_b_support/report.MD`` §4, §8).
            The context fields are computed for every run; they only reach the
            reward when an experiment binds them to a nonzero weight.
        physics_tables_file: ``physics_tables.pt`` from
            ``data/scripts/build_physics_tables.py`` (keyed to the motion library and
            this graph). Empty, the default, builds nothing and leaves every
            physics field of ``ctx.contact_goal`` None -- runs before fine-tune C
            are unchanged. With it, ``ctx.contact_goal`` carries the swing-gated
            unloaded-limb penalty, the commanded-support lean penalty and three
            diagnostics (``protomotions/envs/control/physics_terms.py``).
        swing_ema_tau_s: Time constant of the average applied to the swing
            term's charged load before its clamp (the support term's pricing;
            shorter than its 0.25 s because a swing lasts 0.3-0.5 s).
        swing_load_ref_frac: Charged swing load, as a fraction of body weight,
            at which the swing penalty saturates.
        lean_min_margin: How far inside the commanded support polygon the COM
            must be before the lean penalty is zero (m).
        lean_scale: Shortfall (m) at which the lean penalty saturates.
        physics_exclude_motions: Substrings of motion names neither physics
            term charges (the support term's references that cannot be
            performed with their labelled support).
        contact_targets_file: ``contact_targets.pt``, a release's contact-target
            sidecar (``reference_curation.contact_targets_v2``; TODO C1), keyed to
            this graph. Empty, the default, changes nothing. With it, the
            unwanted-support term charges only zones the human is *known* to keep
            off the floor in the commanded hold (and still only where the
            reference keeps them above ``support_clear_height``), a masked
            contact is never charged, and ``ctx.contact_goal`` carries weight-0
            target diagnostics (``ContactTargets``).
        release_file: A release record (``data/reference_curation/releases/
            <id>.json``). Given, the motion library, the graph, the physics
            tables and the sidecar this control loads must be that release's
            artifacts by sha256, or construction fails
            (``protomotions/utils/release_identity.py``).
        expert_view_steps: Publish an unmasked copy of the first this-many goal
            slots as ``ctx.expert_masked_mimic`` / ``ctx.expert_contact_goal``,
            for a frozen goal-conditioned (Design-B) expert distilled in this
            env (card S1 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``).
            Such an expert was trained on a few slots, every body and every
            half always visible; the student's own view is masked and has more
            slots. 0, the default, publishes nothing.
        expert_view_num_bodies: Bodies per slot in the expert view's body
            masks: the length of the expert's own ``conditionable_body_ids``.
        expert_view_contract: The expert's schedule semantics and the env
            settings its goal observation depends on, from its resolved config
            (``protomotions.agents.supervised.expert_port.expert_goal_contract``).
            Construction fails unless this control and its env match them,
            since the expert view is a slice of this control's own schedule.
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
    interval_schedule: bool = False
    support_clear_height: float = 0.25
    support_load_ref_frac: float = 0.1
    support_body_weight_n: float = 74.0 * 9.81
    support_exclude_motions: List[str] = field(default_factory=list)
    support_ema_tau_s: float = 0.0
    physics_tables_file: str = ""
    swing_ema_tau_s: float = 0.1
    swing_load_ref_frac: float = 0.1
    lean_min_margin: float = 0.03
    lean_scale: float = 0.10
    physics_exclude_motions: List[str] = field(default_factory=list)
    contact_targets_file: str = ""
    release_file: str = ""
    expert_view_steps: int = 0
    expert_view_num_bodies: int = 0
    expert_view_contract: Dict[str, Any] = field(default_factory=dict)


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
        if config.interval_schedule and not config.include_current_segment:
            raise ValueError("interval_schedule requires include_current_segment=True")
        if config.interval_schedule and config.far_goal_prob > 0.0:
            raise ValueError("interval_schedule does not support far-goal promotion")
        super().__init__(config, env)

        if not config.graph_file:
            raise ValueError("ContactGraphControlConfig.graph_file is required")
        self.graph = ContactGraph.from_file(config.graph_file, device=self.env.device)
        frames, fps = _library_layout(self.env.motion_lib)
        if frames is not None and fps is None and self.graph.fps is not None:
            raise ValueError("the motion library mixes frame rates; the v2 contact graph was built at "
                             f"{self.graph.fps} fps for every clip")
        self.graph.validate_against_motion_lib(
            list(self.env.motion_lib.motion_files), motion_num_frames=frames, fps=fps
        )
        self._manual_resolved = None
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
        self._init_support_penalty()

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
        self._init_physics_terms()
        self._init_contact_targets()
        self._init_release()
        self._init_expert_view()

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

    def _init_support_penalty(self) -> None:
        """Tables for the unwanted-support penalty (``support_penalty.py``).

        Zones are the graph's own ground zones, in ``_ground_zone_names`` order, so
        the goal's ground slots (``_ground_pair_ids``) and the pooling matrix line
        up by construction. ``getattr`` defaults keep frozen configs written before
        these fields existed loading with the calibrated values.
        """
        device = self.env.device
        num_bodies = len(self.env.robot_config.kinematic_info.body_names)
        matrix = torch.zeros(len(self._ground_zone_names), num_bodies, device=device)
        matrix[self._ground_zone_rows, self._ground_zone_cols] = 1.0
        self._support_zone_matrix = matrix
        cfg = self.config
        self._support_clear_height = float(getattr(cfg, "support_clear_height", 0.25))
        self._support_load_ref_n = float(getattr(cfg, "support_load_ref_frac", 0.1)) * float(
            getattr(cfg, "support_body_weight_n", 74.0 * 9.81)
        )
        patterns = list(getattr(cfg, "support_exclude_motions", None) or [])
        files = list(self.env.motion_lib.motion_files)
        names = [f.replace("\\", "/").split("/")[-1].removesuffix(".motion") for f in files]
        excluded = torch.tensor(
            [any(p in n for p in patterns) for n in names], dtype=torch.bool, device=device
        )
        unmatched = [p for p in patterns if not any(p in n for n in names)]
        if unmatched:
            raise ValueError(
                f"support_exclude_motions entries match no motion: {unmatched}"
            )
        self._support_excluded_motion = excluded
        tau = float(getattr(cfg, "support_ema_tau_s", 0.0) or 0.0)
        # None keeps the per-frame term exactly (old frozen configs have no field).
        self._support_ema = (
            ChargedLoadEMA(self.env.num_envs, tau, float(self.env.dt), device) if tau > 0.0 else None
        )
        if tau > 0.0:
            print(
                f"ContactGraphControl: unwanted-support load averaged over {tau:.3f} s before "
                f"the clamp (alpha {self._support_ema.alpha:.4f} per step)"
            )
        if patterns:
            print(
                f"ContactGraphControl: unwanted-support penalty skips "
                f"{int(excluded.sum())}/{len(names)} motions matching {patterns}"
            )

    # Candidate support bodies for the lean polygon: hands, forearms and head -- the only
    # zones a lean-gated hold's ground set may contain (build_physics_tables.LEAN_SUPPORT).
    _LEAN_BODIES = ("L_Wrist", "L_Hand", "R_Wrist", "R_Hand", "L_Elbow", "R_Elbow", "Neck", "Head")
    _SLIP_BODIES = (("L_Ankle", "L_Toe"), ("R_Ankle", "R_Toe"), ("L_Wrist", "L_Hand"), ("R_Wrist", "R_Hand"))

    def _init_physics_terms(self) -> None:
        """Load the fine-tune C tables (``physics_terms.PhysicsTables``) if configured.

        ``getattr`` defaults keep frozen configs written before these fields existed loading
        with the terms off.
        """
        cfg = self.config
        self._physics = None
        path = getattr(cfg, "physics_tables_file", "") or ""
        if not path:
            return
        from pathlib import Path as _Path

        device = self.env.device
        body_names = list(self.env.robot_config.kinematic_info.body_names)
        names = [_Path(f).stem for f in self.env.motion_lib.motion_files]
        from protomotions.utils import plant_identity

        frames, fps = _library_layout(self.env.motion_lib)
        tables = PhysicsTables(path, names, body_names, device,
                               plant_mjcf=plant_identity.robot_mjcf(self.env.robot_config),
                               motion_num_frames=frames, fps=fps)
        if tables.version >= 2:
            # Built on a release: the tables name the graph they were built with.
            if tables.pair_names != list(self.graph.pair_names):
                raise ValueError(f"physics tables {path} use another pair vocabulary than the graph")
            graph_sha = file_sha256(self.config.graph_file)
            if tables.graph_sha256 != graph_sha:
                raise ValueError(
                    f"physics tables {path} were built with graph sha256 {str(tables.graph_sha256)[:12]}, "
                    f"the graph in use is {graph_sha[:12]} ({self.config.graph_file})"
                )
        zone_order, _ = self.graph.zone_definition()
        if list(zone_order) != tables.zone_order:
            raise ValueError(f"physics tables zone order {tables.zone_order} != graph {zone_order}")
        if list(tables.seg_pair_consequential.shape[1:]) != [self.graph.seg_node.shape[1], self.graph.num_pairs]:
            raise ValueError("physics tables were built for a different graph (segment/pair shape)")
        zm = torch.zeros(len(zone_order), len(body_names), device=device)
        body_zone = {}
        for z, zone in enumerate(zone_order):
            for b in tables.zone_bodies[zone]:
                zm[z, body_names.index(b)] = 1.0
                body_zone[b] = z
        self._physics = tables
        self._physics_zone_matrix = zm
        # -1 marks a zone this graph gives no ground slot / a body in no zone: never commanded
        self._zone_ground_slot = torch.tensor(
            [self.graph.pair_names.index(f"{zone}:G") if f"{zone}:G" in self.graph.pair_names else -1
             for zone in zone_order],
            dtype=torch.long, device=device,
        )
        self._lean_body_ids = [body_names.index(b) for b in self._LEAN_BODIES]
        per_point = []
        for b, i in zip(self._LEAN_BODIES, self._lean_body_ids):
            n = {0: 4, 1: 2, 2: 1}[int(tables.geom_type[i])]
            per_point += [body_zone.get(b, -1)] * n
        self._lean_point_zone = torch.tensor(per_point, dtype=torch.long, device=device)
        self._slip_zone_bodies = [
            [body_names.index(b) for b in group if b in body_names] for group in self._SLIP_BODIES
        ]
        patterns = list(getattr(cfg, "physics_exclude_motions", None) or [])
        unmatched = [p for p in patterns if not any(p in n for n in names)]
        if unmatched:
            raise ValueError(f"physics_exclude_motions entries match no motion: {unmatched}")
        self._physics_excluded = torch.tensor(
            [any(p in n for p in patterns) for n in names], dtype=torch.bool, device=device
        )
        weight_n = float(getattr(cfg, "support_body_weight_n", 74.0 * 9.81))
        self._swing_load_ref_n = float(getattr(cfg, "swing_load_ref_frac", 0.1)) * weight_n
        tau = float(getattr(cfg, "swing_ema_tau_s", 0.1) or 0.0)
        self._swing_ema = (
            ChargedLoadEMA(self.env.num_envs, tau, float(self.env.dt), device) if tau > 0.0 else None
        )
        self._lean_min_margin = float(getattr(cfg, "lean_min_margin", 0.03))
        self._lean_scale = float(getattr(cfg, "lean_scale", 0.10))
        print(
            f"ContactGraphControl: physics terms on ({path}): swing labels "
            f"{tuple(tables.swing.shape)}, swing tau {tau:.2f} s, lean margin "
            f"{self._lean_min_margin:.3f} m over {len(self._lean_point_zone)} candidate points, "
            f"{int(tables.seg_lean_gate.sum())} lean-gated segments, "
            f"{int(self._physics_excluded.sum())}/{len(names)} motions excluded"
        )

    def _init_contact_targets(self) -> None:
        """Load the release's contact-target sidecar (``ContactTargets``) if configured.

        ``getattr`` defaults keep frozen configs written before the field existed loading with
        the sidecar off, which leaves the unwanted-support term exactly fine-tune C's.
        """
        self._targets = None
        path = getattr(self.config, "contact_targets_file", "") or ""
        if not path:
            return
        from pathlib import Path as _Path
        from protomotions.utils import plant_identity

        frames, fps = _library_layout(self.env.motion_lib)
        names = [_Path(f).stem for f in self.env.motion_lib.motion_files]
        targets = ContactTargets(
            path, self.graph, names, self.env.device, motion_num_frames=frames, fps=fps,
            graph_sha256=file_sha256(self.config.graph_file),
            plant_mjcf=plant_identity.robot_mjcf(self.env.robot_config),
        )
        # the support term's zones (``_ground_zone_names``) as columns of the sidecar's [M, S, Z] tables
        self._targets_support_cols = torch.tensor(
            [targets.zone_order.index(z) for z in self._ground_zone_names], dtype=torch.long, device=self.env.device
        )
        self._targets = targets
        print(
            f"ContactGraphControl: contact targets on ({path}, release {targets.release_id}): "
            f"{int(targets.required_support.sum())} required-support, {int(targets.configured[..., targets.body_pair].sum())} "
            f"configured body-body and {int(targets.masked.sum())} masked segment contacts; "
            f"{int(targets.ground_free.sum())} known-free ground zone-segments"
        )

    def _init_release(self) -> None:
        """Check every artifact this control loads against the release record, if one is named."""
        self.release = None
        path = getattr(self.config, "release_file", "") or ""
        if not path:
            return
        from protomotions.utils import plant_identity

        release = load_release(path)
        motion_file = getattr(self.env.motion_lib, "motion_file", None)
        if not motion_file:
            raise ValueError("release_file is set but the motion library names no packaged file to check")
        require_artifact(release, "package", motion_file, "motion library")
        require_artifact(release, "graph", self.config.graph_file, "contact graph")
        tables = getattr(self.config, "physics_tables_file", "") or ""
        if tables:
            require_artifact(release, "physics_tables", tables, "physics tables")
        targets = getattr(self.config, "contact_targets_file", "") or ""
        if targets:
            require_artifact(release, "contact_targets", targets, "contact targets")
        mjcf = plant_identity.robot_mjcf(self.env.robot_config)
        if mjcf is not None:
            plant_identity.require(release["plant"].get(plant_identity.KEY), mjcf, f"release {release['release_id']}")
        self.release = release
        print(f"ContactGraphControl: release {release['release_id']} -- every loaded artifact matches its record")

    # Schedule semantics the expert view inherits from this control. The view is a
    # slice of this control's own schedule, so each must equal the expert's.
    EXPERT_SCHEDULE_KEYS = (
        "include_current_segment",
        "dwell_channels",
        "min_lead_s",
        "interval_schedule",
        "history_time_clip_s",
    )
    # Env settings the expert's inputs depend on (the goal poses' spawn offset,
    # contact_obs_v1's hysteresis) or that shape the reference it was trained
    # with (the reference contact smoothing), compared for equality.
    EXPERT_ENV_KEYS = (
        "ref_respawn_offset",
        "contact_force_on_threshold_n",
        "contact_force_off_threshold_n",
        "ref_contact_smooth_window",
    )

    def _init_expert_view(self) -> None:
        """Check the expert view's contract (``expert_view_*`` in the config docstring)."""
        cfg = self.config
        steps = int(getattr(cfg, "expert_view_steps", 0) or 0)
        self._expert_view_steps = steps
        self._expert_view_num_bodies = 0
        if steps <= 0:
            return
        bodies = int(getattr(cfg, "expert_view_num_bodies", 0) or 0)
        contract = dict(getattr(cfg, "expert_view_contract", None) or {})
        problems = []
        if not contract:
            problems.append("expert_view_contract is empty: record the expert's resolved config")
        if steps > cfg.num_goal_steps:
            problems.append(f"expert_view_steps {steps} > num_goal_steps {cfg.num_goal_steps}")
        if bodies <= 0:
            problems.append("expert_view_num_bodies must be the length of the expert's conditionable_body_ids")
        if contract and contract.get("num_goal_steps") != steps:
            problems.append(f"the expert has {contract.get('num_goal_steps')} goal slots, the view {steps}")
        if contract and not contract.get("full_visibility", False):
            problems.append("the expert was not trained with every body and both halves of every slot visible, "
                            "so an unmasked view is not its training distribution")
        for key in self.EXPERT_SCHEDULE_KEYS:
            if contract and key not in contract:
                problems.append(f"the contract lacks {key}")
            elif contract and getattr(cfg, key) != contract[key]:
                problems.append(f"{key}: this control {getattr(cfg, key)!r}, the expert {contract[key]!r}")
        if float(cfg.far_goal_prob) > 0.0:
            problems.append("far_goal_prob > 0 shifts the forward window, which the expert never saw")
        env_cfg = getattr(self.env, "config", None)
        if env_cfg is not None:
            for key in self.EXPERT_ENV_KEYS:
                if key in contract and getattr(env_cfg, key, None) != contract[key]:
                    problems.append(f"env.{key}: {getattr(env_cfg, key, None)!r}, the expert's {contract[key]!r}")
            need = contract.get("num_state_history_steps")
            have = int(getattr(env_cfg, "num_state_history_steps", 0) or 0)
            if need is not None and have < int(need):
                problems.append(f"env.num_state_history_steps {have} < the expert's {need}")
            key = "realign_motion_with_humanoid_on_each_step"
            manager = getattr(env_cfg, "motion_manager", None)
            if key in contract and bool(getattr(manager, key, False)) != bool(contract[key]):
                # Realignment re-anchors the reference, so the expert's world-anchored goal poses move.
                problems.append(f"env.motion_manager.{key}: {getattr(manager, key, False)!r}, "
                                f"the expert's {contract[key]!r}")
        if "pair_names" in contract and list(contract["pair_names"]) != list(self.graph.pair_names):
            problems.append("the graph's pair vocabulary is not the expert's")
        if "graph_sha256" in contract:
            have_sha = file_sha256(cfg.graph_file)
            if have_sha != contract["graph_sha256"]:
                problems.append(f"graph sha256 {have_sha[:12]} is not the expert's {str(contract['graph_sha256'])[:12]}")
        if problems:
            raise ValueError("ContactGraphControl expert view:\n  " + "\n  ".join(problems))
        self._expert_view_num_bodies = bodies
        print(
            f"ContactGraphControl: expert view on -- slots [:{steps}] of {cfg.num_goal_steps}, "
            f"{bodies} bodies, every half of a valid slot visible"
        )

    def _populate_expert_view(
        self, ctx: EnvContext, ref_pos: Tensor, ref_rot: Tensor, node_ids: Tensor
    ) -> None:
        """``ctx.expert_masked_mimic`` and ``ctx.expert_contact_goal``: the first slots, unmasked.

        What the expert's own control publishes at this state. Its config reveals every body and
        both halves of every slot (probabilities 1.0), so its masks are ``goal_valid`` broadcast.
        The window, deadlines, dwell channels and contact sets are this control's first slots:
        ``ContactGraph.next_goal_indices`` serves a prefix of the same window for fewer slots, and
        ``_init_expert_view`` asserted the schedule semantics agree. A manual goal passes through.
        """
        k = self._expert_view_steps
        num_envs = self.env.num_envs
        valid = self.goal_valid[:, :k].contiguous()
        body_masks = (
            valid.view(num_envs, k, 1, 1)
            .expand(num_envs, k, self._expert_view_num_bodies, 2)
            .reshape(num_envs, -1)
        )
        ctx.expert_masked_mimic = MaskedMimicContext(
            mimic=ctx.mimic,
            ref_pos=ref_pos[:, :k].contiguous(),
            ref_rot=ref_rot[:, :k].contiguous(),
            target_times=self.target_times[:, :k].contiguous(),
            time_offsets=self._time_offsets[:, :k].contiguous(),
            target_poses_masks=valid,
            target_bodies_masks=body_masks,
        )
        visible = valid.float()
        ctx.expert_contact_goal = ContactGoalContext(
            contact_spec=self._gathered["contact"][:, :k] * visible.unsqueeze(-1),
            orient_spec=torch.nn.functional.one_hot(
                self._gathered["orient"][:, :k], self.graph.num_orientations
            ).float() * visible.unsqueeze(-1),
            visible=visible,
            time_offsets=self._time_offsets[:, :k].contiguous(),
            dwell_features=self._dwell_features[:, :k].contiguous(),
            node_ids=node_ids[:, :k].contiguous(),
            reached=ctx.contact_goal.reached,
        )

    def _target_terms(self, ctx: EnvContext) -> Dict[str, Optional[Tensor]]:
        """Weight-0 diagnostics of the contact-target sidecar (all ``[E]``), or all None without one.

        Inside the commanded hold (slot 0 valid, clip time in its window):

        * ``required_support_met`` -- share of the hold's ``required_support`` ground zones (not
          masked) carrying more than the reached-goal threshold; ``required_support_gate`` marks the
          rows that have any;
        * ``pair_target_met`` -- share of the hold's configured body-body contacts (B6's critical
          pairs and the statics restorations, not masked), on the frames where the reference and the
          human both close them (the sidecar's per-frame mask), in sensed contact;
          ``pair_target_gate`` marks the rows with any such eligible pair (0 without pair sensing);
        * ``known_free_load_n`` -- terrain-filtered load on zones the human keeps off the floor in
          that hold.

        Rows without targets carry the targeted rows' mean, as the physics diagnostics do.
        """
        keys = ("required_support_met", "required_support_gate", "pair_target_met", "pair_target_gate",
                "known_free_load_n")
        if self._targets is None:
            return {k: None for k in keys}
        num_envs, device = self.env.num_envs, self.env.device
        zeros = torch.zeros(num_envs, device=device)
        if self._manual is not None or ctx.mimic is None:
            return {k: zeros for k in keys}
        t = self._targets
        mids = self.env.motion_manager.motion_ids
        now = self.env.motion_manager.motion_times
        g = self._gathered
        in_hold = self.goal_valid[:, 0] & (now >= g["t_start"][:, 0]) & (now <= g["t_end"][:, 0])
        seg = self.goal_index[:, 0].clamp(0, t.masked.shape[1] - 1)
        masked = t.masked[mids, seg]                                              # [E, P]
        ground = getattr(ctx.current, "rigid_body_ground_forces", None)
        maps = self._contact_maps
        pair_forces = getattr(ctx.current, "rigid_body_pair_contact_forces", None)
        if ground is None or (pair_forces is None and maps["pair_slot"].numel() > 0):
            return {k: zeros for k in keys}
        forces = compute_contact_slot_forces(
            ground, pair_forces, maps["ground_slot"], maps["ground_body"], maps["pair_slot"],
            maps["pair_body_a"], maps["pair_body_b"], maps["num_pairs"],
        )
        touching = forces > maps["thresholds"].unsqueeze(0)                       # [E, P]

        required = t.required_support[mids, seg] & ~masked & in_hold.unsqueeze(-1)
        required_count = required.sum(-1)
        required_rows = required_count > 0
        required_met = (touching & required).sum(-1).float() / required_count.clamp(min=1).float()

        eligible = (t.configured[mids, seg] & ~masked & t.body_pair.unsqueeze(0)
                    & t.frame_pairs(mids, now) & in_hold.unsqueeze(-1))
        if maps["num_body_body_slots"] == 0:
            eligible = torch.zeros_like(eligible)
        eligible_count = eligible.sum(-1)
        pair_rows = eligible_count > 0
        pair_met = (touching & eligible).sum(-1).float() / eligible_count.clamp(min=1).float()

        # the support term's pricing: terrain-filtered vertical load, pooled into its zones
        zone_load = ground[..., 2].clamp_min(0.0) @ self._support_zone_matrix.t().to(ground.dtype)
        free = (t.ground_free[mids, seg][:, self._targets_support_cols]
                & ~masked[:, self._ground_pair_ids] & in_hold.unsqueeze(-1))
        free_load = (zone_load * free.to(zone_load.dtype)).sum(-1)

        def fill(x, rows):
            if bool(rows.any()):
                return torch.where(rows, x, x[rows].mean())
            return torch.zeros_like(x)

        return dict(
            required_support_met=fill(required_met, required_rows),
            required_support_gate=required_rows.float(),
            pair_target_met=fill(pair_met, pair_rows),
            pair_target_gate=pair_rows.float(),
            known_free_load_n=free_load,
        )

    def _physics_terms(self, ctx: EnvContext) -> Dict[str, Optional[Tensor]]:
        """Fine-tune C's swing and lean terms plus diagnostics (all ``[E]``), or all None."""
        keys = ("swing_penalty", "swing_load_n", "swing_gate", "lean_penalty", "lean_margin",
                "lean_gate", "lean_error", "lean_error_valid", "slip_power", "pair_load_n")
        if self._physics is None:
            return {k: None for k in keys}
        num_envs = self.env.num_envs
        device = self.env.device
        zeros = torch.zeros(num_envs, device=device)
        if self._manual is not None or ctx.mimic is None:
            return {k: zeros for k in keys}
        t = self._physics
        mids = self.env.motion_manager.motion_ids
        now = self.env.motion_manager.motion_times
        g = self._gathered
        in_hold = self.goal_valid[:, 0] & (now >= g["t_start"][:, 0]) & (now <= g["t_end"][:, 0])
        excluded = self._physics_excluded[mids]
        cur = ctx.current
        pos, rot = cur.rigid_body_pos, cur.rigid_body_rot
        ground = getattr(cur, "rigid_body_ground_forces", None)

        # swing-gated unloaded-limb penalty
        swing_gate = ~in_hold & ~excluded
        charged = swing_charged_load(ground, t.swing_at(mids, now), self._physics_zone_matrix, swing_gate)
        if self._swing_ema is not None:
            swing_pen = self._swing_ema.price(charged, swing_gate, self._swing_load_ref_n)
        else:
            swing_pen = (charged / self._swing_load_ref_n).clamp(0.0, 1.0) * swing_gate.float()

        # commanded-support lean
        seg = self.goal_index[:, 0].clamp(0, t.seg_lean_gate.shape[1] - 1)
        lean_gate = in_hold & t.seg_lean_gate[mids, seg] & ~excluded
        slots = self._zone_ground_slot
        goal_ground = (g["contact"][:, 0][:, slots.clamp(min=0)] > 0.5) & (slots >= 0)
        points = patch_points(pos, rot, t, self._lean_body_ids)
        zones = self._lean_point_zone
        valid = goal_ground[:, zones.clamp(min=0)] & (zones >= 0)
        com = whole_body_com(pos, rot, t.body_mass, t.body_com_local)
        margin = polygon_margin(points[..., :2], valid, com[:, :2])
        lean_pen = lean_shortfall(margin, lean_gate, self._lean_min_margin, self._lean_scale)
        centroid = (points[..., :2] * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        error = (com[:, :2] - centroid - t.seg_cop_rel[mids, seg]).norm(dim=-1)
        error_valid = lean_gate & t.seg_cop_valid[mids, seg]

        # loaded slip power on feet and hands
        slip = zeros.clone()
        if ground is not None:
            fz = ground[..., 2].clamp_min(0.0)
            for bodies in self._slip_zone_bodies:
                speed = corner_slip_speed(pos, rot, cur.rigid_body_vel, cur.rigid_body_ang_vel, t, bodies)
                slip = slip + fz[:, bodies].sum(-1) * (speed - 0.05).clamp(min=0.0)

        # load through the commanded hold's leg-on-arm / leg-on-trunk pairs
        pair_load = zeros.clone()
        pair_rows = torch.zeros(num_envs, dtype=torch.bool, device=device)
        pair_forces = getattr(cur, "rigid_body_pair_contact_forces", None)
        if ground is not None and (pair_forces is not None or self._contact_maps["pair_slot"].numel() == 0):
            maps = self._contact_maps
            forces = compute_contact_slot_forces(
                ground, pair_forces, maps["ground_slot"], maps["ground_body"], maps["pair_slot"],
                maps["pair_body_a"], maps["pair_body_b"], maps["num_pairs"],
            )
            cons = t.seg_pair_consequential[mids, seg] & in_hold.unsqueeze(-1)
            pair_load = (forces * cons.float()).sum(-1)
            pair_rows = cons.any(-1)

        def fill(x, rows):
            # diagnostics are logged as a batch mean: fill rows the quantity is not defined on
            # with the defined rows' mean, so the logged number is that mean
            if bool(rows.any()):
                return torch.where(rows, x, x[rows].mean())
            return torch.zeros_like(x)

        return dict(
            swing_penalty=swing_pen,
            swing_load_n=charged,
            swing_gate=swing_gate.float(),
            lean_penalty=lean_pen,
            lean_margin=fill(margin, lean_gate),
            lean_gate=lean_gate.float(),
            lean_error=fill(error, error_valid),
            lean_error_valid=error_valid.float(),
            slip_power=slip,
            pair_load_n=fill(pair_load, pair_rows),
        )

    def _unwanted_support(self, ctx: EnvContext) -> Tuple[Tensor, Tensor, Tensor]:
        """``(penalty [E] in [0,1], charged load [E] N, gate [E] float)``.

        The gate is "inside the commanded hold segment": slot 0 valid and the clip
        time within its ``[t_start, t_end]`` -- with ``include_current_segment``
        that is exactly when slot 0 *is* the segment being played. Manual goals
        (probes, the viz panel) have no clip segment, so the gate is off there.
        """
        num_envs = self.env.num_envs
        if self._manual is not None or ctx.mimic is None:
            zeros = torch.zeros(num_envs, device=self.env.device)
            return zeros, zeros, zeros
        now = self.env.motion_manager.motion_times
        g = self._gathered
        in_hold = (
            self.goal_valid[:, 0]
            & (now >= g["t_start"][:, 0])
            & (now <= g["t_end"][:, 0])
        )
        goal_ground = g["contact"][:, 0][:, self._ground_pair_ids] > 0.5
        mids = self.env.motion_manager.motion_ids
        excluded = self._support_excluded_motion[mids]
        known_free = None
        if getattr(self, "_targets", None) is not None:
            # A release's sidecar: only zones the human keeps off the floor in this hold are
            # known negatives, and a masked contact leaves every target (TODO C1).
            t = self._targets
            seg = self.goal_index[:, 0].clamp(0, t.masked.shape[1] - 1)
            known_free = (t.ground_free[mids, seg][:, self._targets_support_cols]
                          & ~t.masked[mids, seg][:, self._ground_pair_ids])
        penalty, charged = unwanted_support(
            ground_forces=getattr(ctx.current, "rigid_body_ground_forces", None),
            ref_body_pos=ctx.mimic.ref_state.rigid_body_pos,
            goal_ground=goal_ground,
            in_hold=in_hold,
            zone_matrix=self._support_zone_matrix,
            clear_height=self._support_clear_height,
            load_ref_n=self._support_load_ref_n,
            excluded=excluded,
            known_free=known_free,
        )
        gate = (in_hold & ~excluded).float()
        if self._support_ema is not None:
            # Same gate, same saturation; only the load is averaged first. The raw
            # per-frame newtons stay the reported diagnostic (unwanted_support_n).
            penalty = self._support_ema.price(charged, gate, self._support_load_ref_n)
        return penalty, charged, gate

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
            interval=self.config.interval_schedule,
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
        hold_duration: Optional[Tensor] = None,
        deadline_floor: Optional[float] = None,
    ) -> None:
        """Drive the goal from an explicit query instead of the clip's schedule.

        This is what makes a trained student answerable: "reach *this* contact
        configuration, holding *this* pose, within *this* long", where the pose is
        named as (clip, time) because that is the only pose representation the
        motion library can serve.  Every argument is ``[num_envs, num_goal_steps]``.

        The two dwell channels a scheduled slot carries are ``duration =
        t_end - t_hold`` (how long the configuration lasts past the commanded
        frame; constant) and ``remaining = t_end - now`` (counts down; before the
        frame it is the deadline plus the duration). A manual goal has no
        segment, so the caller supplies both:

        * ``hold_seconds`` feeds ``remaining``: the seconds left until the
          commanded configuration ends, at issue time. It counts down here
          between issues and stops at 0. Without it the channel reads 0, i.e.
          "stay 0 s" -- the opposite of what a 12 s hold probe means.
        * ``hold_duration`` feeds ``duration`` and stays put. None reuses
          ``hold_seconds``, the pre-S1 behaviour: one value in both channels,
          which misstates the duration by 4.1 s at p50 and 9.7 s at p90 over
          every scheduled slot of release v3 (S0 of
          ``expert_revist/graph_growth_2026_10_03/PLAN.MD``). Kept so old probe
          results reproduce; drivers on training's semantics pass it.
        * ``deadline_floor``: the deadline (``time_offsets``) counts down by
          ``dt`` per step and stops here. None is ``min_lead_s``, the legacy
          schedule's floor (its nearest hold is always at least that far away);
          with ``include_current_segment`` a reached goal's deadline is 0 in
          training, so drivers on that semantics pass 0.0.

        Re-issue as the plan advances. Call :meth:`clear_manual_goal` to return
        to the clip schedule.

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
        for name, tensor in (("hold_seconds", hold_seconds), ("hold_duration", hold_duration)):
            if tensor is not None and tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")
        if deadline_floor is not None and float(deadline_floor) < 0.0:
            raise ValueError(f"deadline_floor must be >= 0, got {deadline_floor}")
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

        def seconds(tensor: Optional[Tensor]) -> Tensor:
            if tensor is None:
                return torch.zeros_like(time_offsets.to(device).float())
            return tensor.to(device).float().clone()

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
            # `total` is the duration channel and stays put; `remaining` counts
            # down between re-issues so the channel reads 12 -> 0 over a hold.
            "hold_total": seconds(hold_duration if hold_duration is not None else hold_seconds),
            "hold_remaining": seconds(hold_seconds),
            "deadline_floor": (
                float(deadline_floor) if deadline_floor is not None
                else float(self.config.min_lead_s)
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
        # segment. Its contact set: on a v2 graph the segment holding the goal's
        # pose (the side-specific hold, as its scheduled goal serves it), else
        # the node's (ContactGraph.manual_contact).
        contact, resolved = self.graph.manual_contact(
            node, self._goal_motion_ids, self.target_times
        )
        self._manual_resolved = resolved
        self._gathered = {
            "node": safe_node,
            "t_start": self.target_times,
            "t_end": self.target_times + manual["hold_total"],
            "t_hold": self.target_times,
            "contact": contact,
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
        if self._support_ema is not None:
            # A new episode starts its load average from zero.
            self._support_ema.reset(env_ids)
        if getattr(self, "_physics", None) is not None and self._swing_ema is not None:
            self._swing_ema.reset(env_ids)
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
        if self._support_ema is not None:
            self._support_ema.mark_step()
        if getattr(self, "_physics", None) is not None and self._swing_ema is not None:
            self._swing_ema.mark_step()
        if not self._initialized:
            return

        if self._manual is not None:
            # A manual goal is a deadline, not a clip position: count it down so
            # the "time to target" the policy sees means the same thing it did
            # during training.
            self._manual["offset"] = (self._manual["offset"] - self.env.dt).clamp(
                min=self._manual.get("deadline_floor", self.config.min_lead_s)
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
        support_penalty, support_load_n, support_gate = self._unwanted_support(ctx)
        physics = self._physics_terms(ctx)
        physics.update(self._target_terms(ctx))

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
            unwanted_support=support_penalty,
            unwanted_support_n=support_load_n,
            support_gate=support_gate,
            **physics,
        )
        if getattr(self, "_expert_view_steps", 0) > 0:
            self._populate_expert_view(ctx, ref_pos, ref_rot, node_ids)
