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
    settle_steps: int = field(
        default=10,
        metadata={"help": "Policy steps after reset before the plans start."},
    )
    reissue_every_s: float = field(
        default=0.5,
        metadata={"help": "Seconds between re-arming the goal slots."},
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
) -> Dict[str, Tensor]:
    """Per-env goal-slot tensors for ``set_manual_goal`` at sequence time ``t``.

    Mirrors ``render_contact_goal_sequence.GoalDriver.issue``: slot ``k`` holds
    the ``k``-th upcoming goal, slot 0's deadline counts down to the end of its
    reach window and then parks at ``hold_lead_s`` instead of the 0.2 s floor.
    A sequence past its end keeps its final goal at ``hold_lead_s``.
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
            seq_offset[s, slot] = max(remaining, hold_lead_s)
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

            for step in range(max(self.config.settle_steps, 0)):
                outputs = agent.model.forward_inference(obs_td)
                action = outputs.get("mean_action", outputs.get("action"))
                obs, *_ = env.step(action)
                agent.pre_collect_step(step + 1)
                obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

            total_s = min(max(s.total_s for s in sequences), self.config.max_seconds)
            total_steps = int(round(total_s / dt))
            reissue_every = max(int(round(self.config.reissue_every_s / dt)), 1)
            render_every = max(int(round(1.0 / (dt * self.config.render_fps))), 1)

            self.control.set_manual_goal(
                **fill_goal_slots(
                    sequences, env_sequence, 0.0, self.slots,
                    self.config.hold_lead_s, self.device,
                )
            )
            obs_td = self._refresh_obs()

            positions: List[Tensor] = []
            frame_times: List[float] = []
            zones: List[Tensor] = []
            for step in range(total_steps):
                t = step * dt
                if step > 0 and step % reissue_every == 0:
                    self.control.set_manual_goal(
                        **fill_goal_slots(
                            sequences, env_sequence, t, self.slots,
                            self.config.hold_lead_s, self.device,
                        )
                    )
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
                    positions.append(state.rigid_body_pos[:num_seq].cpu().clone())
                    zones.append(self._measured_ground_zones(state, num_seq))
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

        return self._encode(epoch, sequences, positions, zones, frame_times, out_dir)

    # ------------------------------------------------------------------ #
    # Encoding + scoring
    # ------------------------------------------------------------------ #
    def _encode(
        self,
        epoch: int,
        sequences: List[VizSequence],
        positions: List[Tensor],
        zones: List[Tensor],
        frame_times: List[float],
        out_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Path], Dict[str, float]]:
        if out_dir is None:
            out_dir = Path(self.agent.root_dir) / "viz" / f"epoch_{epoch:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        stacked = torch.stack(positions).numpy()  # [T, S, B, 3]
        zone_stack = torch.stack(zones).numpy()  # [T, S, Z]
        zone_names = list(self.control._ground_zone_names)
        goal_zone_ids = self.control._ground_pair_ids.cpu()
        node_contact = self.graph.node_contact.cpu()

        videos: Dict[str, Path] = {}
        scalars: Dict[str, float] = {}
        summary = []
        for s, sequence in enumerate(sequences):
            titles = []
            iou_per_frame = []
            final_goal = sequence.goals[-1]
            wanted = (
                node_contact[final_goal.node].index_select(0, goal_zone_ids) > 0.5
            ).numpy()
            for f, t in enumerate(frame_times):
                index = sequence.active_index(t)
                goal = sequence.goals[index]
                measured = zone_stack[f, s]
                on = " ".join(
                    _ZONE_SHORT.get(zone_names[z], zone_names[z])
                    for z in np.nonzero(measured)[0]
                ) or "-"
                goal_key = self.graph.node_keys[goal.node]
                done = " (done)" if t >= sequence.total_s else ""
                titles.append(
                    f"{sequence.name}  t={t:4.1f}s{done}\n"
                    f"goal[{index}] {goal.name}: {short_config(goal_key)} | on: {on}"
                )
                goal_wanted = (
                    node_contact[goal.node].index_select(0, goal_zone_ids) > 0.5
                ).numpy()
                union = float(np.logical_or(measured, goal_wanted).sum())
                iou_per_frame.append(
                    float(np.logical_and(measured, goal_wanted).sum()) / union
                    if union else 1.0
                )

            # Final goal's hold window score: the per-sequence trend line.
            hold_from = sequence.total_s - final_goal.hold_s
            hold_frames = [
                f for f, t in enumerate(frame_times)
                if hold_from <= t < sequence.total_s
            ]
            if hold_frames:
                hold_iou = float(np.mean([iou_per_frame[f] for f in hold_frames]))
                final_zones = zone_stack[hold_frames[-1], s]
                reached = bool((final_zones == wanted).all())
            else:
                hold_iou, reached = float("nan"), False
            scalars[f"viz/{sequence.name}/final_goal_iou"] = hold_iou
            scalars[f"viz/{sequence.name}/reached_exact"] = float(reached)

            path = out_dir / f"{sequence.name}.mp4"
            render_stick_video(
                stacked[:, s], self.bones, titles, path,
                self.config.render_fps, self.config.video_px,
                pelvis_index=self.pelvis_index,
                head_index=self.head_index,
            )
            videos[f"viz/{sequence.name}"] = path
            summary.append(
                {
                    "sequence": sequence.name,
                    "final_goal_iou": None if np.isnan(hold_iou) else round(hold_iou, 3),
                    "reached_exact": reached,
                    "goals": [g.name for g in sequence.goals],
                }
            )

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info(
            "sequence viz @ epoch %d: %d videos -> %s", epoch, len(videos), out_dir
        )
        return videos, scalars
