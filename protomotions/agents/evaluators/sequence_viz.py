# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-training stick-figure videos of goal sequences, logged to wandb.

Every ``viz_every`` epochs the agent drives a small set of scripted goal
sequences with the **deployable prior** (``forward_inference``), captures the
rigid-body positions from the simulator state, and renders each sequence as a
matplotlib 3D stick-figure video — no Isaac rendering, no viewport, works
headless. Videos land in ``<root_dir>/viz/epoch_xxxxx/`` and, when a wandb
logger is attached, under the ``viz/`` panel.

The rollout protocol is the two proven ones combined:

* goal issuing is ported from ``data/scripts/render_contact_goal_sequence.py``
  (all five slots filled, deadlines re-armed instead of parking at the 0.2 s
  floor, observations rebuilt after every switch, ``dones`` ignored);
* env disruption follows ``MimicEvaluator``: snapshot ``env.save_state()`` and
  the motion manager's ids/times before, restore both after, and let the agent
  set ``_skip_next_policy_update`` so the transition epoch is never trained on.

All sequences run **concurrently**: ``set_manual_goal`` takes full
``[num_envs, slots]`` tensors, so env ``e`` follows sequence ``e % S`` and the
whole panel costs one batched rollout (~30-60 s) plus CPU rendering.

Sequence sources, filled up to ``num_sequences``:

1. **Plan files** (the probe plans): resolved by configuration string, with a
   fallback that strips body-body pairs whose two zones are both ground
   contacts in the same string — exactly the load-path demotion rule — so a
   plan written against a pre-load-path graph still resolves. Unresolvable
   plans are skipped with a warning, never fatal.
2. **Pure holds** at the highest-dwell nodes: start *at* the held pose, goal =
   the same node for ``hold_seconds``. This is the hold-drift measurement of
   ``notes/Student_improvement_plan2.MD`` §11.2, on video.
3. **Edge round-trips** from the graph's own segment tables (src → dst → src),
   most-frequent transitions first.

Everything here is wrapped so a failure disables the feature and logs, rather
than killing a 20-hour training run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


@dataclass
class SequenceVizConfig:
    """Configuration for the in-training sequence visualization."""

    viz_every: Optional[int] = field(
        default=None,
        metadata={"help": "Render the sequence panel every N epochs. None/0 disables."},
    )
    num_sequences: int = field(
        default=10,
        metadata={"help": "Total sequences (plans + holds + edge round-trips)."},
    )
    plan_files: List[str] = field(
        default_factory=list,
        metadata={"help": "Goal-sequence plan JSONs (render_contact_goal_sequence schema)."},
    )
    num_hold_sequences: int = field(
        default=3,
        metadata={"help": "Pure hold-drift sequences at the highest-dwell nodes."},
    )
    hold_seconds: float = field(
        default=8.0,
        metadata={"help": "Dwell requested by each pure-hold sequence."},
    )
    max_seconds: float = field(
        default=20.0,
        metadata={"help": "Hard cap on any sequence's duration."},
    )
    log_scalars: bool = field(
        default=True,
        metadata={
            "help": "Also send the panel's per-sequence and per-goal scalars to "
            "the logger. False sends only the videos, which is what early "
            "training wants: one unseeded nucleus draw per sequence per epoch "
            "makes every one of these a coin flip (round 7_1 §5.2 -- v7_1's "
            "hold_probe_standing was good on 7 of 24 panels with no trend), so "
            "the charts cost dashboard space and invite exactly the "
            "single-draw reading round 7 §11.5 had to retract. They are still "
            "written to viz/epoch_*/summary.json either way, so nothing is "
            "lost for offline analysis."
        },
    )
    settle_steps: int = field(
        default=10,
        metadata={"help": "Policy steps after reset before the plans start."},
    )
    max_replicas: int = field(
        default=0,
        metadata={
            "help": "Cap on how many environments per sequence are SCORED. 0 = "
            "all of them. Every env already runs one of the sequences "
            "(`env_sequence = env_ids % num_seq`) and is already stepped, so "
            "scoring them all is free: at 1024 envs and 28 plans that is ~36 "
            "independent nucleus draws per plan instead of the single unseeded "
            "one round 7_1 §5.2 had to caveat. Videos are still rendered from "
            "replica 0 only."
        },
    )
    legacy_settle: bool = field(
        default=False,
        metadata={
            "help": "Reproduce the pre-fix panel protocol: settle under the "
            "CLIP schedule, install the manual goal afterwards, and never "
            "flush the held intent. Exists only as the control arm for that "
            "fix -- for a hold probe the clip schedule during settle is the "
            "very continuation the probe is testing against, and up to "
            "chunk_steps-1 steps of an intent chosen under it survived into "
            "the plan."
        },
    )
    dump_traces: bool = field(
        default=False,
        metadata={"help": "Also write pose_error_traces.npz ([T, S*R] goal-pose "
                          "error, frame times, sequence order) for offline "
                          "survival analysis."},
    )
    pose_arrive_m: float = field(
        default=0.15,
        metadata={"help": "Goal-pose error below which a goal counts as reached."},
    )
    pose_depart_m: float = field(
        default=0.30,
        metadata={
            "help": "Goal-pose error above which a reached goal counts as left "
            "again; the gap to `pose_arrive_m` is the hysteresis that keeps "
            "`time_held_s` from chattering."
        },
    )
    reissue_every_s: float = field(
        default=0.5,
        metadata={"help": "Seconds between re-arming the goal slots."},
    )
    hold_lead_mode: str = field(
        default="clamp",
        metadata={"help": "'clamp' = max(remaining, hold_lead_s) (shipped); "
                          "'park' substitutes hold_lead_s only after the reach "
                          "window expires, so a large lead does not inflate the "
                          "reach deadlines of a multi-goal plan."},
    )
    hold_lead_s: float = field(
        default=1.2,
        metadata={"help": "Deadline held during a goal's hold phase (median "
                          "hold-to-hold gap is ~1.4 s; the 0.2 s floor was "
                          "never sustained in training)."},
    )
    render_fps: int = field(
        default=15,
        metadata={"help": "Video frame rate; sim steps are subsampled to it."},
    )
    video_px: int = field(
        default=480,
        metadata={"help": "Video width in pixels."},
    )


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in protomotions/tests/test_sequence_viz.py)
# --------------------------------------------------------------------------- #
def strip_supported_pairs(config: str) -> str:
    """Drop body-body pairs whose two zones are both ground contacts.

    String-level version of the load-path demotion rule
    (``build_contact_graph_from_rollouts.demote_supported_pairs``), so a plan
    written against a pre-load-path graph resolves against a load-path one.
    """
    if "@" not in config:
        return config
    pairs_part, orient = config.rsplit("@", 1)
    pairs = pairs_part.split("|")
    grounded = {p[:-2] for p in pairs if p.endswith(":G")}
    kept = []
    for pair in pairs:
        if not pair.endswith(":G") and "+" in pair:
            a, b = pair.split("+", 1)
            if a in grounded and b in grounded:
                continue
        kept.append(pair)
    return "|".join(kept) + "@" + orient


def _round(value, digits: int = 3):
    """JSON-safe round: NaN and None both become null."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return None if np.isnan(number) else round(number, digits)


def resolve_config(graph, config: str) -> Optional[int]:
    """Node id for a configuration string, tolerant of the identity-rule change."""
    for candidate in (config, strip_supported_pairs(config)):
        try:
            return int(graph.node_id_for_key(candidate))
        except KeyError:
            continue
    return None


_ZONE_SHORT = {
    "L_FOOT": "LF", "R_FOOT": "RF", "L_SHANK": "LS", "R_SHANK": "RS",
    "L_THIGH": "LT", "R_THIGH": "RT", "PELVIS": "PV", "TRUNK": "TK",
    "HEAD": "HD", "L_UPPER_ARM": "LU", "R_UPPER_ARM": "RU",
    "L_FOREARM": "LA", "R_FOREARM": "RA", "L_HAND": "LH", "R_HAND": "RH",
}


def short_config(config: str, max_len: int = 34) -> str:
    """Compact human-readable form of a configuration string for titles."""
    if "@" not in config:
        return config[:max_len]
    pairs_part, orient = config.rsplit("@", 1)
    out = []
    for pair in pairs_part.split("|"):
        if pair.endswith(":G"):
            out.append(_ZONE_SHORT.get(pair[:-2], pair[:-2]))
        elif "+" in pair:
            a, b = pair.split("+", 1)
            out.append(f"{_ZONE_SHORT.get(a, a)}·{_ZONE_SHORT.get(b, b)}")
        else:
            out.append(pair)
    return (" ".join(out) + " @" + orient[:4])[:max_len]


@dataclass
class VizGoal:
    name: str
    node: int
    pose_motion: int
    pose_time: float
    reach_s: float
    hold_s: float


@dataclass
class VizSequence:
    name: str
    start_motion: int
    start_time: float
    goals: List[VizGoal]

    @property
    def ends(self) -> List[float]:
        out, elapsed = [], 0.0
        for goal in self.goals:
            elapsed += goal.reach_s + goal.hold_s
            out.append(elapsed)
        return out

    @property
    def total_s(self) -> float:
        return self.ends[-1] if self.goals else 0.0

    def active_index(self, t: float) -> int:
        for i, end in enumerate(self.ends):
            if t < end:
                return i
        return len(self.goals) - 1


def fill_goal_slots(
    sequences: List[VizSequence],
    env_sequence: Tensor,
    t: float,
    slots: int,
    hold_lead_s: float,
    device: torch.device,
    hold_lead_mode: str = "clamp",
) -> Dict[str, Tensor]:
    """Per-env goal-slot tensors for ``set_manual_goal`` at sequence time ``t``.

    Mirrors ``render_contact_goal_sequence.GoalDriver.issue``: slot ``k`` holds
    the ``k``-th upcoming goal, slot 0's deadline counts down to the end of its
    reach window and then parks at ``hold_lead_s`` instead of the 0.2 s floor.
    A sequence past its end keeps its final goal at ``hold_lead_s``.

    ``hold_lead_mode`` decides what a large ``hold_lead_s`` does to the REACH
    phase, and the distinction is not cosmetic — it confounded the first
    deadline sweep:

    * ``clamp`` (the shipped behaviour) is ``max(remaining, hold_lead_s)``, so
      raising the lead also inflates every reach window shorter than it. A
      5-goal plan with 2 s reaches then never sees a deadline below 5 s and
      stops meeting its own waypoints.
    * ``park`` leaves the reach countdown alone and only substitutes
      ``hold_lead_s`` once the reach window has expired — i.e. exactly during
      the hold. That is the arm that isolates "what does the deadline mean
      while I am being asked to stay?".
    """
    num_seq = len(sequences)
    seq_node = torch.full((num_seq, slots), -1, dtype=torch.long)
    seq_pose_motion = torch.zeros(num_seq, slots, dtype=torch.long)
    seq_pose_time = torch.zeros(num_seq, slots)
    seq_offset = torch.zeros(num_seq, slots)
    seq_visible = torch.zeros(num_seq, slots, dtype=torch.bool)

    for s, sequence in enumerate(sequences):
        start = sequence.active_index(t)
        ends = sequence.ends
        for slot, index in enumerate(
            range(start, min(start + slots, len(sequence.goals)))
        ):
            goal = sequence.goals[index]
            remaining = ends[index] - goal.hold_s - t
            seq_node[s, slot] = goal.node
            seq_pose_motion[s, slot] = goal.pose_motion
            seq_pose_time[s, slot] = goal.pose_time
            if hold_lead_mode == "clamp":
                deadline = max(remaining, hold_lead_s)
            else:
                deadline = remaining if remaining > 0.0 else hold_lead_s
            seq_offset[s, slot] = deadline
            seq_visible[s, slot] = True

    env_sequence = env_sequence.cpu()
    return {
        "node_ids": seq_node[env_sequence].to(device),
        "pose_motion_ids": seq_pose_motion[env_sequence].to(device),
        "pose_times": seq_pose_time[env_sequence].to(device),
        "time_offsets": seq_offset[env_sequence].to(device),
        "pose_visible": seq_visible[env_sequence].to(device),
        "contact_visible": seq_visible[env_sequence].to(device),
    }


def derive_node_dwell(graph) -> Dict[int, Tuple[float, int, float]]:
    """``node -> (total trusted dwell, motion of longest segment, its t_hold)``."""
    stats: Dict[int, Tuple[float, float, int, float]] = {}
    seg_count = graph.seg_count.cpu()
    seg_node = graph.seg_node.cpu()
    seg_start, seg_end = graph.seg_start.cpu(), graph.seg_end.cpu()
    seg_hold = graph.seg_hold.cpu()
    for motion in range(seg_node.shape[0]):
        for k in range(int(seg_count[motion])):
            node = int(seg_node[motion, k])
            duration = float(seg_end[motion, k] - seg_start[motion, k])
            dwell, longest, best_motion, best_hold = stats.get(
                node, (0.0, -1.0, -1, 0.0)
            )
            if duration > longest:
                longest, best_motion, best_hold = (
                    duration, motion, float(seg_hold[motion, k]),
                )
            stats[node] = (dwell + duration, longest, best_motion, best_hold)
    return {
        node: (dwell, best_motion, best_hold)
        for node, (dwell, _longest, best_motion, best_hold) in stats.items()
    }


def derive_edges(graph) -> List[Tuple[int, int, int, int, float, float]]:
    """``(src, dst, count, motion, t_hold_src, t_hold_dst)`` most frequent first.

    The runtime graph tensors carry no edge table, but consecutive trusted
    segments within a motion *are* the transitions the JSON edges were built
    from, so they are re-derived here the same way.
    """
    counts: Dict[Tuple[int, int], int] = {}
    occurrence: Dict[Tuple[int, int], Tuple[int, float, float]] = {}
    best_dwell: Dict[Tuple[int, int], float] = {}
    seg_count = graph.seg_count.cpu()
    seg_node, seg_hold = graph.seg_node.cpu(), graph.seg_hold.cpu()
    seg_start, seg_end = graph.seg_start.cpu(), graph.seg_end.cpu()
    for motion in range(seg_node.shape[0]):
        n = int(seg_count[motion])
        for k in range(n - 1):
            src, dst = int(seg_node[motion, k]), int(seg_node[motion, k + 1])
            if src == dst:
                continue
            key = (src, dst)
            counts[key] = counts.get(key, 0) + 1
            # Prefer the occurrence arriving at the longest destination hold:
            # a clip often attempts a hard pose several times, and a goal cut
            # from a 2 s touch-and-return attempt asks for a pose the
            # reference itself does not sustain.
            dst_dwell = float(seg_end[motion, k + 1] - seg_start[motion, k + 1])
            if dst_dwell > best_dwell.get(key, -1.0):
                best_dwell[key] = dst_dwell
                occurrence[key] = (
                    motion,
                    float(seg_hold[motion, k]),
                    float(seg_hold[motion, k + 1]),
                )
    ranked = sorted(counts, key=lambda k: -counts[k])
    return [(src, dst, counts[(src, dst)], *occurrence[(src, dst)]) for src, dst in ranked]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def skeleton_bones(
    pos_body_names: List[str], common_body_names: List[str], parent_indices: List[int]
) -> List[Tuple[int, int, str]]:
    """``(child_pos_idx, parent_pos_idx, color)`` per bone.

    ``pos_body_names`` is the order the captured positions are indexed in.
    The kinematic tree is stated in COMMON/MJCF order, so the two are joined
    by name, never by position — the documented body-order trap. Note that
    ``get_robot_state()`` returns bodies already converted to COMMON order
    (``get_bodies_state`` → ``convert_to_common``), so for positions captured
    from it the join is the identity, not a reorder onto the simulator's raw
    order.
    """
    pos_index = {name: i for i, name in enumerate(pos_body_names)}
    bones = []
    for child, parent in enumerate(parent_indices):
        if parent < 0:
            continue
        child_name, parent_name = common_body_names[child], common_body_names[parent]
        if child_name not in pos_index or parent_name not in pos_index:
            continue
        if child_name.startswith("L_"):
            color = "#d62728"
        elif child_name.startswith("R_"):
            color = "#1f77b4"
        else:
            color = "#444444"
        bones.append((pos_index[child_name], pos_index[parent_name], color))
    return bones


def render_stick_video(
    positions: np.ndarray,
    bones: List[Tuple[int, int, str]],
    titles: List[str],
    out_path: Path,
    fps: int,
    video_px: int,
    pelvis_index: int = 0,
    head_index: Optional[int] = None,
) -> None:
    """Encode ``positions [T, B, 3]`` as a 3D stick-figure mp4.

    The camera follows the figure — per-frame x/y recentring (EMA-smoothed)
    with a fixed metric window — so the body fills the frame instead of
    shrinking to the whole-trajectory bounds; the 0.5 m world-anchored
    gridlines sliding under it supply the travel cue a static camera gave.
    """
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    from matplotlib import pyplot as plt
    from matplotlib import ticker

    dpi = 100
    # h264 pads to 16-px macro blocks; hand imageio compliant dimensions
    # instead of letting it resize (the 480x360 -> 480x368 warning).
    width = max(round(video_px / 16), 8) * 16
    height = max(round(video_px * 0.75 / 16), 6) * 16
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    # 3D axes reserve large margins; oversize the axes rect so the figure
    # fills the canvas instead of floating in padding.
    ax.set_position((-0.12, -0.14, 1.24, 1.26))

    half_span = 1.05
    z_hi = max(2.0, float(positions[..., 2].max()) + 0.15)
    ax.set_zlim(0.0, z_hi)
    ax.set_box_aspect((1.0, 1.0, z_hi / (2 * half_span)))
    ax.view_init(elev=12.0, azim=-70.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.set_major_locator(ticker.MultipleLocator(0.5))

    lines = [
        ax.plot([], [], [], color=color, linewidth=3.0, solid_capstyle="round")[0]
        for _c, _p, color in bones
    ]
    joints_by_color: Dict[str, List[int]] = {}
    for child, _parent, color in bones:
        joints_by_color.setdefault(color, []).append(child)
    dots = {
        color: ax.plot([], [], [], "o", color=color, markersize=3.5)[0]
        for color in joints_by_color
    }
    head = (
        ax.plot([], [], [], "o", color="#222222", markersize=9.0)[0]
        if head_index is not None
        else None
    )
    trail, = ax.plot([], [], [], color="#999999", linewidth=0.8, alpha=0.7)
    title = fig.text(0.02, 0.99, "", fontsize=8, va="top", family="monospace")

    pelvis = positions[:, pelvis_index]
    center = positions[0].mean(axis=0)[:2]
    frames = []
    for t in range(positions.shape[0]):
        pose = positions[t]
        center = 0.8 * center + 0.2 * pose.mean(axis=0)[:2]
        ax.set_xlim(center[0] - half_span, center[0] + half_span)
        ax.set_ylim(center[1] - half_span, center[1] + half_span)
        for line, (child, parent, _color) in zip(lines, bones):
            seg = pose[[parent, child]]
            line.set_data(seg[:, 0], seg[:, 1])
            line.set_3d_properties(seg[:, 2])
        for color, joint_ids in joints_by_color.items():
            pts = pose[joint_ids]
            dots[color].set_data(pts[:, 0], pts[:, 1])
            dots[color].set_3d_properties(pts[:, 2])
        if head is not None:
            point = pose[head_index : head_index + 1]
            head.set_data(point[:, 0], point[:, 1])
            head.set_3d_properties(point[:, 2])
        trail.set_data(pelvis[: t + 1, 0], pelvis[: t + 1, 1])
        trail.set_3d_properties(np.zeros(t + 1))
        title.set_text(titles[t] if t < len(titles) else "")
        fig.canvas.draw()
        # buffer_rgba() is a view into the canvas's reused draw buffer: without
        # the copy every appended frame aliases the final draw and the encoded
        # video plays as one frozen frame.
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame[: frame.shape[0] // 16 * 16, : frame.shape[1] // 16 * 16])
    plt.close(fig)

    imageio.mimwrite(str(out_path), frames, fps=fps, quality=7)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class SequenceVizRunner:
    """Owns the sequence set and produces the per-epoch video panel."""

    def __init__(self, agent, config: SequenceVizConfig):
        self.agent = agent
        self.config = config
        self.env = agent.env
        self.device = agent.device

        self.control = None
        for component in self.env.control_manager.components.values():
            if hasattr(component, "set_manual_goal"):
                self.control = component
                break
        if self.control is None:
            raise RuntimeError("no control component exposes set_manual_goal")
        self.graph = self.control.graph
        self.slots = int(self.control.config.num_goal_steps)

        motion_lib = self.env.motion_lib
        files = getattr(motion_lib, "motion_files", None) or []
        self.motion_names = [Path(str(f)).stem for f in files]
        if len(self.motion_names) != motion_lib.num_motions():
            self.motion_names = [f"motion_{i}" for i in range(motion_lib.num_motions())]

        kin = self.env.robot_config.kinematic_info
        # get_robot_state() returns bodies already converted to COMMON order,
        # so captured positions are indexed by kinematic_info's own body list —
        # joining onto the simulator's raw order scrambles the skeleton.
        common_names = list(kin.body_names)
        self.bones = skeleton_bones(
            common_names, common_names, list(kin.parent_indices)
        )
        self.pelvis_index = common_names.index("Pelvis")
        self.head_index = (
            common_names.index("Head") if "Head" in common_names else None
        )

        self.sequences = self._build_sequences()
        if not self.sequences:
            raise RuntimeError("no visualization sequences could be built")
        log.info(
            "sequence viz: %d sequences every %s epochs: %s",
            len(self.sequences),
            config.viz_every,
            ", ".join(s.name for s in self.sequences),
        )

    # ------------------------------------------------------------------ #
    # Sequence construction
    # ------------------------------------------------------------------ #
    def _resolve_clip(self, needle: str) -> Optional[int]:
        matches = [
            i for i, n in enumerate(self.motion_names) if needle.lower() in n.lower()
        ]
        if len(matches) > 1:
            # Synthetic hold clips embed their source clip's stem, so any plan
            # naming a source clip becomes ambiguous once holds join the
            # corpus; the original clip is the one a plan means.
            originals = [
                i for i in matches if not self.motion_names[i].startswith("hold_")
            ]
            if len(originals) == 1:
                return originals[0]
        return matches[0] if len(matches) == 1 else None

    def _load_plan(self, path: str) -> Optional[VizSequence]:
        plan_path = Path(path)
        if not plan_path.is_file():
            log.warning("sequence viz: plan file missing: %s", path)
            return None
        plan = json.loads(plan_path.read_text())
        start_motion = self._resolve_clip(plan["start"]["clip"])
        if start_motion is None:
            log.warning("sequence viz: start clip of %s not in corpus", plan_path.name)
            return None
        goals = []
        for entry in plan["goals"]:
            config = entry.get("config")
            node = resolve_config(self.graph, config) if config else None
            pose_motion = self._resolve_clip(entry["pose_clip"])
            if node is None or pose_motion is None:
                log.warning(
                    "sequence viz: skipping plan %s (goal '%s' does not resolve "
                    "against this graph/corpus)",
                    plan_path.name, entry.get("name", "?"),
                )
                return None
            goals.append(
                VizGoal(
                    name=entry.get("name", "goal"),
                    node=node,
                    pose_motion=pose_motion,
                    pose_time=float(entry["pose_time"]),
                    reach_s=float(entry["reach_s"]),
                    hold_s=float(entry["hold_s"]),
                )
            )
        sequence = VizSequence(
            name=plan_path.stem,
            start_motion=start_motion,
            start_time=float(plan["start"].get("time", 0.0)),
            goals=goals,
        )
        return self._capped(sequence)

    def _capped(self, sequence: VizSequence) -> VizSequence:
        while sequence.goals and sequence.total_s > self.config.max_seconds:
            sequence.goals = sequence.goals[:-1]
        return sequence if sequence.goals else None

    def _hold_sequences(self, count: int, taken: set) -> List[VizSequence]:
        out = []
        dwell = derive_node_dwell(self.graph)
        for node, (_total, motion, t_hold) in sorted(
            dwell.items(), key=lambda kv: -kv[1][0]
        ):
            if len(out) >= count:
                break
            if motion < 0 or node in taken:
                continue
            taken.add(node)
            key = self.graph.node_keys[node]
            out.append(
                VizSequence(
                    name=f"hold_{short_config(key, 20).replace(' ', '_')}",
                    start_motion=motion,
                    start_time=t_hold,
                    goals=[
                        VizGoal(
                            name="hold",
                            node=node,
                            pose_motion=motion,
                            pose_time=t_hold,
                            reach_s=0.5,
                            hold_s=self.config.hold_seconds,
                        )
                    ],
                )
            )
        return out

    def _edge_sequences(self, count: int, taken: set) -> List[VizSequence]:
        out = []
        for src, dst, _n, motion, t_src, t_dst in derive_edges(self.graph):
            if len(out) >= count:
                break
            if (src, dst) in taken or (dst, src) in taken:
                continue
            taken.add((src, dst))
            reach = float(np.clip(t_dst - t_src, 1.0, 4.0))
            src_key = self.graph.node_keys[src]
            dst_key = self.graph.node_keys[dst]
            out.append(
                VizSequence(
                    name=(
                        f"edge_{short_config(src_key, 12)}_to_"
                        f"{short_config(dst_key, 12)}"
                    ).replace(" ", "_"),
                    start_motion=motion,
                    start_time=max(t_src - 0.3, 0.0),
                    goals=[
                        VizGoal("out", dst, motion, t_dst, reach, 1.5),
                        VizGoal("back", src, motion, t_src, reach, 1.5),
                    ],
                )
            )
        return out

    def _build_sequences(self) -> List[VizSequence]:
        sequences: List[VizSequence] = []
        for path in self.config.plan_files:
            plan = self._load_plan(path)
            if plan is not None:
                sequences.append(plan)
        taken_nodes: set = set()
        room = self.config.num_sequences - len(sequences)
        sequences.extend(
            self._hold_sequences(min(self.config.num_hold_sequences, room), taken_nodes)
        )
        taken_edges: set = set()
        room = self.config.num_sequences - len(sequences)
        sequences.extend(self._edge_sequences(room, taken_edges))
        # wandb keys double as file names: ASCII, unique, bounded length.
        seen: Dict[str, int] = {}
        for sequence in sequences:
            name = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in sequence.name
            )[:48].strip("_")
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            sequence.name = name
        return sequences[: self.config.num_sequences]

    # ------------------------------------------------------------------ #
    # The rollout
    # ------------------------------------------------------------------ #
    def _refresh_obs(self):
        env = self.env
        env._current_context = env._build_global_context(
            env.simulator.get_robot_state()
        )
        env.compute_observations(context=env._current_context)
        return self.agent.obs_dict_to_tensordict(
            self.agent.add_agent_info_to_obs(env.get_obs())
        )

    def _measured_ground_zones(self, state, limit: int) -> Tensor:
        control = self.control
        magnitude = state.rigid_body_contact_forces.norm(dim=-1)
        zone_force = torch.zeros(
            self.env.num_envs, len(control._ground_zone_names), device=self.device
        )
        zone_force.index_add_(
            1, control._ground_zone_rows, magnitude[:, control._ground_zone_cols]
        )
        return (zone_force[:limit] > control.config.ground_contact_threshold_n).cpu()

    @torch.no_grad()
    def run(
        self, epoch: int, out_dir: Optional[Path] = None
    ) -> Tuple[Dict[str, Path], Dict[str, float]]:
        env, agent = self.env, self.agent
        sequences = self.sequences[: max(env.num_envs, 1)]
        num_seq = len(sequences)
        dt = float(env.dt)
        env_ids = torch.arange(env.num_envs, device=self.device)
        env_sequence = env_ids % num_seq
        # Every env is already running one of the sequences and is already
        # being stepped; only the *scoring* used to stop at the first num_seq
        # rows. Keeping all of them turns each panel scalar from one unseeded
        # nucleus draw into a rate over `replicas` draws, for free.
        replicas = max(env.num_envs // num_seq, 1)
        max_replicas = int(getattr(self.config, "max_replicas", 0) or 0)
        if max_replicas:
            replicas = min(replicas, max_replicas)
        num_scored = num_seq * replicas

        snapshot = env.save_state()
        cached_motion_ids = env.motion_manager.motion_ids.clone()
        cached_motion_times = env.motion_manager.motion_times.clone()
        agent.eval()
        # Chunked-intent models (the FSQ student) hold their latent across
        # steps; bound here so the finally block can flush even on an early
        # failure.
        flush_intent = getattr(agent.model, "flush_held_intent", None)

        try:
            starts_motion = torch.tensor(
                [s.start_motion for s in sequences], dtype=torch.long
            )
            starts_time = torch.tensor(
                [s.start_time for s in sequences], dtype=torch.float32
            )
            env.motion_manager.motion_ids[env_ids] = starts_motion[
                env_sequence.cpu()
            ].to(self.device)
            env.motion_manager.motion_times[env_ids] = starts_time[
                env_sequence.cpu()
            ].to(self.device)
            obs, _ = env.reset(
                env_ids, sample_flat=True, disable_motion_resample=True
            )
            # Whatever the training rollout left held does not describe these
            # episodes.
            if flush_intent is not None:
                flush_intent()
            agent.pre_collect_step(0)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

            total_s = min(max(s.total_s for s in sequences), self.config.max_seconds)
            total_steps = int(round(total_s / dt))
            reissue_every = max(int(round(self.config.reissue_every_s / dt)), 1)
            render_every = max(int(round(1.0 / (dt * self.config.render_fps))), 1)

            # The plan's own first goal is installed BEFORE the settle steps.
            # Settling under the *clip* schedule instead was a confound with
            # teeth: for `hold_probe_standing`, whose start is Downward Dog -a
            # @0.2 s, that schedule is exactly the fold-into-downdog
            # continuation the probe is testing against, and the intent chosen
            # under it survived into the plan because nothing flushed at
            # install (the standalone renderer does flush; the panel did not).
            legacy = bool(getattr(self.config, "legacy_settle", False))
            if not legacy:
                self.control.set_manual_goal(
                    **fill_goal_slots(
                        sequences, env_sequence, 0.0, self.slots,
                        self.config.hold_lead_s, self.device,
                        getattr(self.config, "hold_lead_mode", "clamp"),
                    )
                )
                if flush_intent is not None:
                    flush_intent()
                obs_td = self._refresh_obs()

            for step in range(max(self.config.settle_steps, 0)):
                outputs = agent.model.forward_inference(obs_td)
                action = outputs.get("mean_action", outputs.get("action"))
                obs, *_ = env.step(action)
                agent.pre_collect_step(step + 1)
                obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

            # Re-arm so the deadline the plan starts on is the one t=0 means,
            # not one the settle steps have already counted down.
            self.control.set_manual_goal(
                **fill_goal_slots(
                    sequences, env_sequence, 0.0, self.slots,
                    self.config.hold_lead_s, self.device,
                    getattr(self.config, "hold_lead_mode", "clamp"),
                )
            )
            obs_td = self._refresh_obs()
            active_index = [s.active_index(0.0) for s in sequences]

            positions: List[Tensor] = []
            root_rots: List[Tensor] = []
            frame_times: List[float] = []
            zones: List[Tensor] = []
            for step in range(total_steps):
                t = step * dt
                if step > 0 and step % reissue_every == 0:
                    self.control.set_manual_goal(
                        **fill_goal_slots(
                            sequences, env_sequence, t, self.slots,
                            self.config.hold_lead_s, self.device,
                            getattr(self.config, "hold_lead_mode", "clamp"),
                        )
                    )
                    # Flush only where the plan actually ADVANCED a goal, not on
                    # the periodic deadline re-arms -- the renderer's rule
                    # (`GoalDriver.issue`), which the panel never had.
                    now = [s.active_index(t) for s in sequences]
                    changed = [i for i, (a, b) in enumerate(zip(active_index, now))
                               if a != b]
                    if changed and flush_intent is not None and not legacy:
                        rows = torch.nonzero(
                            torch.isin(
                                env_sequence,
                                torch.tensor(changed, device=self.device),
                            ),
                            as_tuple=True,
                        )[0]
                        flush_intent(rows)
                    active_index = now
                    obs_td = self._refresh_obs()

                outputs = agent.model.forward_inference(obs_td)
                action = outputs.get("mean_action", outputs.get("action"))
                # dones deliberately ignored: falls stay in frame, and resets
                # would teleport the figure mid-video.
                obs, *_ = env.step(action)
                agent.pre_collect_step(step + 1)
                obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

                if step % render_every == 0:
                    state = env.simulator.get_robot_state()
                    positions.append(state.rigid_body_pos[:num_scored].cpu().clone())
                    # The root rotation is what heading-normalises the achieved
                    # pose for the goal-pose-error scalar; captured here so the
                    # metric costs nothing beyond a [S, 4] slice per frame.
                    root_rots.append(
                        state.rigid_body_rot[:num_scored, self.pelvis_index]
                        .cpu().clone()
                    )
                    zones.append(self._measured_ground_zones(state, num_scored))
                    frame_times.append(t)
        finally:
            try:
                self.control.clear_manual_goal()
            finally:
                env.motion_manager.motion_ids = cached_motion_ids
                env.motion_manager.motion_times = cached_motion_times
                env.restore_state(snapshot)
                # Same reasoning as the panel entry: the intent held from the
                # panel's episodes does not describe the restored training
                # rollout.
                if flush_intent is not None:
                    flush_intent()

        return self._encode(
            epoch, sequences, positions, root_rots, zones, frame_times, out_dir,
            replicas=replicas,
        )

    # ------------------------------------------------------------------ #
    # Goal pose error
    # ------------------------------------------------------------------ #
    def _goal_pose_errors(
        self,
        sequences: List[VizSequence],
        positions: Tensor,
        root_rots: Tensor,
        frame_times: List[float],
    ) -> Optional[np.ndarray]:
        """``[T, S]`` distance to the pose each sequence is being commanded.

        ``final_goal_iou`` scores the contact half, which is identical for every
        member of a degenerate node -- so it cannot tell Warrior III from Lord
        of the Dance, and that substitution went unnoticed for four rounds
        (``notes/Student_v7_improvement_investigation.MD`` §5.5). This is the
        pose half: mean per-body distance over the conditionable bodies, each
        pelvis-relative and heading-normalised, which is exactly what
        ``data/scripts/score_probe_pose.py`` reports offline and what
        ``ContactGraphControl._goal_pose_error`` logs live.

        Returns ``None`` if anything about the corpus lookup does not line up;
        the caller then simply omits the scalar rather than losing the videos.
        """
        from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate

        body_ids = self.control.conditionable_body_ids.cpu()
        offsets, flat = [], []
        for sequence in sequences:
            offsets.append(len(flat))
            flat.extend(sequence.goals)
        if not flat:
            return None

        reference = self.env.motion_lib.get_motion_state(
            torch.tensor([g.pose_motion for g in flat], device=self.device),
            torch.tensor(
                [g.pose_time for g in flat], device=self.device, dtype=torch.float32
            ),
        )

        def normalise(pos: Tensor, root_rot: Tensor) -> Tensor:
            """``[N, len(body_ids), 3]`` in each row's own heading frame."""
            local = pos[:, body_ids] - pos[:, self.pelvis_index].unsqueeze(1)
            heading = calc_heading_quat_inv(root_rot, w_last=True)
            n, b = local.shape[0], local.shape[1]
            return quat_rotate(
                heading.unsqueeze(1).expand(-1, b, -1).reshape(-1, 4),
                local.reshape(-1, 3),
                w_last=True,
            ).view(n, b, 3)

        goal_local = normalise(
            reference.rigid_body_pos.cpu(),
            reference.rigid_body_rot[:, self.pelvis_index].cpu(),
        )

        num_frames, num_cols = positions.shape[0], positions.shape[1]
        num_seq = len(sequences)
        errors = np.full((num_frames, num_cols), np.nan, dtype=np.float32)
        for s, sequence in enumerate(sequences):
            # `env_sequence = env_ids % num_seq`, so every column congruent to
            # s is a replica of this sequence: one batched normalise for all of
            # them, and the per-frame active goal is shared, so the whole
            # sequence costs two tensor ops rather than T x R Python steps.
            cols = torch.arange(s, num_cols, num_seq)
            if cols.numel() == 0:
                continue
            reps = cols.numel()
            flat = normalise(
                positions[:, cols].reshape(num_frames * reps, *positions.shape[2:]),
                root_rots[:, cols].reshape(num_frames * reps, 4),
            ).view(num_frames, reps, -1, 3)
            goals = goal_local[
                torch.tensor(
                    [offsets[s] + sequence.active_index(t) for t in frame_times]
                )
            ]  # [T, nb, 3]
            errors[:, cols.numpy()] = (
                (flat - goals.unsqueeze(1)).norm(dim=-1).mean(dim=-1).numpy()
            )
        return errors

    # ------------------------------------------------------------------ #
    # Encoding + scoring
    # ------------------------------------------------------------------ #
    def _score_replica(
        self,
        sequence: "VizSequence",
        measured: np.ndarray,
        pose_err: Optional[np.ndarray],
        frame_times: List[float],
        node_contact,
        goal_zone_ids,
    ) -> Dict:
        """Every scalar this panel reports, for ONE rollout of one sequence.

        ``measured`` is ``[T, Z]`` boolean ground zones, ``pose_err`` ``[T]``
        metres to the active goal's pose (or None).
        """
        # getattr with defaults: a frozen config pickled before these fields
        # existed unpickles without them, and inference tools load exactly
        # such a config.
        arrive = float(getattr(self.config, "pose_arrive_m", 0.15))
        depart = float(getattr(self.config, "pose_depart_m", 0.30))
        wanted = {}
        iou = np.empty(len(frame_times), dtype=np.float64)
        for f, t in enumerate(frame_times):
            goal = sequence.goals[sequence.active_index(t)]
            want = wanted.get(goal.node)
            if want is None:
                want = (
                    node_contact[goal.node].index_select(0, goal_zone_ids) > 0.5
                ).numpy()
                wanted[goal.node] = want
            union = float(np.logical_or(measured[f], want).sum())
            iou[f] = (
                float(np.logical_and(measured[f], want).sum()) / union
                if union else 1.0
            )

        final_goal = sequence.goals[-1]
        final_want = (
            node_contact[final_goal.node].index_select(0, goal_zone_ids) > 0.5
        ).numpy()
        hold_from = sequence.total_s - final_goal.hold_s
        hold_frames = [
            f for f, t in enumerate(frame_times)
            if hold_from <= t < sequence.total_s
        ]
        if hold_frames:
            hold_iou = float(np.mean(iou[hold_frames]))
            reached = bool((measured[hold_frames[-1]] == final_want).all())
        else:
            hold_iou, reached = float("nan"), False

        per_goal = []
        for gi, goal in enumerate(sequence.goals):
            start = sequence.ends[gi] - goal.reach_s - goal.hold_s
            frames = [
                f for f, t in enumerate(frame_times)
                if start <= t < sequence.ends[gi]
            ]
            if not frames:
                continue
            hold_f = [
                f for f, t in enumerate(frame_times)
                if sequence.ends[gi] - goal.hold_s <= t < sequence.ends[gi]
            ]
            row = {
                "goal": goal.name,
                "best_iou": float(np.max(iou[frames])),
                "hold_iou": float(np.mean(iou[hold_f])) if hold_f else float("nan"),
                "best_pose_err": float("nan"),
                # `best_pose_err` is a MINIMUM over the window, so a policy that
                # touches the pose for one frame and leaves scores the same as
                # one that holds it for twelve seconds -- which is why
                # hold_probe_standing read 0.02-0.03 m at all 26 v9 panels while
                # actually holding on 7 of them. These three cannot be
                # saturated that way, and they are the per-goal twins of the
                # `hold_iou` the contact half has had since round 7_1.
                "hold_pose_err": float("nan"),
                "end_pose_err": float("nan"),
                "time_held_s": float("nan"),
            }
            if pose_err is not None:
                window = pose_err[frames]
                if not np.all(np.isnan(window)):
                    row["best_pose_err"] = float(np.nanmin(window))
                if hold_f:
                    held = pose_err[hold_f]
                    if not np.all(np.isnan(held)):
                        row["hold_pose_err"] = float(np.nanmean(held))
                        row["end_pose_err"] = float(held[-1])
                # Longest run, inside this goal's window, that starts at
                # `pose_arrive_m` and has not yet re-crossed `pose_depart_m`.
                best_run = 0.0
                run_start = None
                for f in frames:
                    e = pose_err[f]
                    if np.isnan(e):
                        continue
                    if run_start is None:
                        if e <= arrive:
                            run_start = frame_times[f]
                    elif e > depart:
                        best_run = max(best_run, frame_times[f] - run_start)
                        run_start = None
                if run_start is not None:
                    best_run = max(
                        best_run, frame_times[frames[-1]] - run_start
                    )
                row["time_held_s"] = float(best_run)
            per_goal.append(row)

        out = {
            "final_goal_iou": hold_iou,
            "reached_exact": reached,
            "final_goal_pose_err": float("nan"),
            "best_pose_err": float("nan"),
            "per_goal": per_goal,
        }
        if pose_err is not None and hold_frames:
            window = pose_err[hold_frames]
            if not np.all(np.isnan(window)):
                out["final_goal_pose_err"] = float(np.nanmean(window))
            if not np.all(np.isnan(pose_err)):
                out["best_pose_err"] = float(np.nanmin(pose_err))
        if per_goal:
            out["max_goal_best_iou"] = max(g["best_iou"] for g in per_goal)
            out["min_goal_best_iou"] = min(g["best_iou"] for g in per_goal)
            scored = [g for g in per_goal if not np.isnan(g["best_pose_err"])]
            if scored:
                hardest = max(scored, key=lambda g: g["best_pose_err"])
                out["worst_goal"] = hardest["goal"]
                out["worst_goal_pose_err"] = hardest["best_pose_err"]
            held = [g for g in per_goal if not np.isnan(g["hold_pose_err"])]
            if held:
                worst = max(held, key=lambda g: g["hold_pose_err"])
                out["worst_goal_hold_pose_err"] = worst["hold_pose_err"]
                out["worst_hold_goal"] = worst["goal"]
        return out

    def _encode(
        self,
        epoch: int,
        sequences: List[VizSequence],
        positions: List[Tensor],
        root_rots: List[Tensor],
        zones: List[Tensor],
        frame_times: List[float],
        out_dir: Optional[Path] = None,
        replicas: int = 1,
    ) -> Tuple[Dict[str, Path], Dict[str, float]]:
        if out_dir is None:
            out_dir = Path(self.agent.root_dir) / "viz" / f"epoch_{epoch:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        position_stack = torch.stack(positions)  # [T, S*R, B, 3]
        stacked = position_stack.numpy()
        zone_stack = torch.stack(zones).numpy()  # [T, S*R, Z]

        # Never at the cost of the videos: this scalar is an addition to the
        # panel, not a precondition for it.
        pose_errors = None
        try:
            if root_rots:
                pose_errors = self._goal_pose_errors(
                    sequences, position_stack, torch.stack(root_rots), frame_times
                )
        except Exception:
            log.exception("sequence viz: goal pose error unavailable this epoch")
        zone_names = list(self.control._ground_zone_names)
        goal_zone_ids = self.control._ground_pair_ids.cpu()
        node_contact = self.graph.node_contact.cpu()

        videos: Dict[str, Path] = {}
        scalars: Dict[str, float] = {}
        summary = []
        num_seq = len(sequences)
        num_cols = zone_stack.shape[1]
        arrive = float(getattr(self.config, "pose_arrive_m", 0.15))
        for s, sequence in enumerate(sequences):
            cols = list(range(s, num_cols, num_seq))
            draws = [
                self._score_replica(
                    sequence, zone_stack[:, c],
                    None if pose_errors is None else pose_errors[:, c],
                    frame_times, node_contact, goal_zone_ids,
                )
                for c in cols
            ]
            head = draws[0]  # the replica the video shows

            def agg(key, source=None):
                vals = [
                    d[key] for d in (source or draws)
                    if key in d and not (
                        isinstance(d[key], float) and np.isnan(d[key])
                    )
                ]
                return np.array(vals, dtype=np.float64) if vals else None

            # Per-sequence scalars keep their replica-0 meaning so a video and
            # its numbers still describe the same rollout; the RATES beside
            # them are what should be read as the measurement.
            scalars[f"viz/{sequence.name}/final_goal_iou"] = head["final_goal_iou"]
            scalars[f"viz/{sequence.name}/reached_exact"] = float(
                head["reached_exact"]
            )
            for key in ("max_goal_best_iou", "min_goal_best_iou",
                        "worst_goal_pose_err", "final_goal_pose_err",
                        "best_pose_err", "worst_goal_hold_pose_err"):
                if key in head and not np.isnan(head[key]):
                    scalars[f"viz/{sequence.name}/{key}"] = head[key]

            entry = {
                "sequence": sequence.name,
                "replicas": len(draws),
                "final_goal_iou": _round(head["final_goal_iou"]),
                "reached_exact": head["reached_exact"],
                "final_goal_pose_err": _round(head["final_goal_pose_err"]),
                "best_pose_err": _round(head["best_pose_err"]),
                "goals": [g.name for g in sequence.goals],
                "worst_goal": head.get("worst_goal"),
                "worst_goal_pose_err": _round(
                    head.get("worst_goal_pose_err", float("nan"))
                ),
                "per_goal": [
                    {k: (_round(v) if isinstance(v, float) else v)
                     for k, v in g.items()}
                    for g in head["per_goal"]
                ],
            }

            # Every replica's own hold-window error, so a rate at a different
            # `pose_arrive_m` can be recomputed offline. It matters: several
            # plans sit in a tight band just above 0.15 m (trip_lf_lh_rf_rh_upri
            # reads p10 0.165 / p90 0.173), where a rate is a threshold
            # artifact and the percentiles are the honest reading.
            entry["final_goal_pose_err_by_replica"] = [
                _round(d["final_goal_pose_err"]) for d in draws
            ]
            final_err = agg("final_goal_pose_err")
            if final_err is not None:
                entry["final_goal_pose_err_p10"] = _round(np.percentile(final_err, 10))
                entry["final_goal_pose_err_p50"] = _round(np.median(final_err))
                entry["final_goal_pose_err_p90"] = _round(np.percentile(final_err, 90))
                entry["hold_success_rate"] = _round(float((final_err < arrive).mean()))
                scalars[f"viz/{sequence.name}/hold_success_rate"] = (
                    entry["hold_success_rate"]
                )
                scalars[f"viz/{sequence.name}/final_goal_pose_err_p50"] = (
                    entry["final_goal_pose_err_p50"]
                )
            entry["reached_exact_rate"] = _round(
                float(np.mean([bool(d["reached_exact"]) for d in draws]))
            )
            scalars[f"viz/{sequence.name}/reached_exact_rate"] = (
                entry["reached_exact_rate"]
            )
            iou_all = agg("final_goal_iou")
            if iou_all is not None:
                entry["final_goal_iou_p50"] = _round(np.median(iou_all))

            # Per-goal aggregates: the column that says WHICH goal in a plan
            # the policy loses, over every replica rather than one draw.
            names = [g["goal"] for g in head["per_goal"]]
            per_goal_agg = []
            for gi, name in enumerate(names):
                rows = [d["per_goal"][gi] for d in draws
                        if gi < len(d["per_goal"])]
                def col(key):
                    vals = [r[key] for r in rows
                            if not (isinstance(r[key], float) and np.isnan(r[key]))]
                    return np.array(vals) if vals else None
                item = {"goal": name, "n": len(rows)}
                for key in ("best_iou", "hold_iou", "best_pose_err",
                            "hold_pose_err", "end_pose_err", "time_held_s"):
                    v = col(key)
                    if v is not None:
                        item[f"{key}_p50"] = _round(float(np.median(v)))
                v = col("hold_pose_err")
                if v is not None:
                    item["hold_rate"] = _round(float((v < arrive).mean()))
                v = col("best_pose_err")
                if v is not None:
                    item["reach_rate"] = _round(float((v < arrive).mean()))
                # Per-replica, so a goal's rate can be re-scored CONDITIONAL on
                # the replica having survived the previous goal. Without it a
                # chain's later goals read 0 whenever the policy fell at goal 1,
                # which is a sequential confound, not a property of that goal.
                item["hold_pose_err_by_replica"] = [
                    _round(r["hold_pose_err"]) for r in rows
                ]
                # ...and how long each replica actually stayed. Arrival is not
                # the problem on any probe measured so far (reach_rate 1.00,
                # best error ~0.03 m); the response variable is the survival
                # time, and it is what separates side plank (12.9 s of a 12.9 s
                # window) from standing (4.2 s).
                item["time_held_s_by_replica"] = [
                    _round(r["time_held_s"]) for r in rows
                ]
                per_goal_agg.append(item)
                # Replica-0 per-goal scalars, unchanged in meaning since round
                # 7_1, plus the rate that is the point of scoring replicas.
                key = f"viz/{sequence.name}/goal{gi}_{name}"
                row = head["per_goal"][gi]
                scalars[f"{key}/best_iou"] = row["best_iou"]
                for field_name in ("hold_iou", "best_pose_err", "hold_pose_err",
                                   "time_held_s"):
                    if not np.isnan(row[field_name]):
                        scalars[f"{key}/{field_name}"] = row[field_name]
                if "hold_rate" in item:
                    scalars[f"{key}/hold_rate"] = item["hold_rate"]
            entry["per_goal_agg"] = per_goal_agg
            if per_goal_agg:
                rates = [g["hold_rate"] for g in per_goal_agg if "hold_rate" in g]
                if rates:
                    entry["min_goal_hold_rate"] = _round(min(rates))
                    scalars[f"viz/{sequence.name}/min_goal_hold_rate"] = min(rates)

            titles = []
            for f, t in enumerate(frame_times):
                index = sequence.active_index(t)
                goal = sequence.goals[index]
                on = " ".join(
                    _ZONE_SHORT.get(zone_names[z], zone_names[z])
                    for z in np.nonzero(zone_stack[f, cols[0]])[0]
                ) or "-"
                done = " (done)" if t >= sequence.total_s else ""
                titles.append(
                    f"{sequence.name}  t={t:4.1f}s{done}\n"
                    f"goal[{index}] {goal.name}: "
                    f"{short_config(self.graph.node_keys[goal.node])} | on: {on}"
                )
            path = out_dir / f"{sequence.name}.mp4"
            render_stick_video(
                stacked[:, cols[0]], self.bones, titles, path,
                self.config.render_fps, self.config.video_px,
                pelvis_index=self.pelvis_index,
                head_index=self.head_index,
            )
            videos[f"viz/{sequence.name}"] = path
            summary.append(entry)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        if getattr(self.config, "dump_traces", False) and pose_errors is not None:
            # [T, S*R] goal-pose error and [T] frame times. Arrival is not the
            # failure mode on any probe measured so far -- every one of them
            # reaches its pose -- so the response variable is the survival
            # time, and that needs the trace, not a summary statistic.
            np.savez_compressed(
                out_dir / "pose_error_traces.npz",
                pose_errors=pose_errors,
                frame_times=np.asarray(frame_times, dtype=np.float32),
                sequences=np.asarray([s.name for s in sequences]),
                replicas=np.int32(replicas),
            )
        log.info(
            "sequence viz @ epoch %d: %d videos -> %s", epoch, len(videos), out_dir
        )
        return videos, scalars
