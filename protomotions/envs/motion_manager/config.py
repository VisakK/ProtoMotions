# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for motion manager components.

This module contains all configuration dataclasses for motion manager functionality,
co-located with the motion manager implementations in the same directory.
"""

from typing import Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class MotionManagerConfig:
    """Configuration for motion management."""

    _target_: str = "protomotions.envs.motion_manager.motion_manager.MotionManager"

    init_start_prob: float = field(
        default=0.2,
        metadata={
            "help": "Probability to sample an initial pose instead of random time. Helps prevent local-minima in AMP.",
            "min": 0.0,
            "max": 1.0,
        }
    )

    subset_method: Optional[Union[str, List[int]]] = field(
        default=None,
        metadata={
            "help": "Motion subset for evaluation: 'first', 'last', 'random', or list of motion IDs. None uses all motions.",
            "options": ["first", "last", "random"],
        }
    )

    exclude_motion_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Motion IDs to exclude from sampling. Useful for removing problematic motions.",
        }
    )

    exclude_motions_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to file with motion IDs to exclude (one per line). Can also be an expert training directory.",
        }
    )

    realign_motion_with_humanoid_on_each_step: bool = field(
        default=False,
        metadata={
            "help": "Realign motion with humanoid each step. Prevents tracking error accumulation for imperfect retargeting.",
        }
    )


@dataclass
class MimicMotionManagerConfig(MotionManagerConfig):
    """Configuration for mimic motion management."""

    _target_: str = (
        "protomotions.envs.motion_manager.mimic_motion_manager.MimicMotionManager"
    )

    resample_on_reset: bool = field(
        default=True,
        metadata={"help": "Whether to resample motion on environment reset."}
    )


@dataclass
class ContactGraphMotionManagerConfig(MimicMotionManagerConfig):
    """Mimic motion management whose start times are anchored to graph segments.

    Plain reference-state initialisation samples a clip time *uniformly*, which
    spends most episodes in the middle of whatever the clip happens to be doing.
    The interesting states for a contact-conditioned student are the make/break
    boundaries -- the frames just before the support set changes -- and those are
    exactly what the contact graph enumerates.
    """

    _target_: str = (
        "protomotions.envs.motion_manager.contact_graph_motion_manager."
        "ContactGraphMotionManager"
    )

    graph_file: str = field(
        default="",
        metadata={"help": "contact_graph.pt whose segments anchor the start times."},
    )
    segment_start_prob: float = field(
        default=0.6,
        metadata={
            "help": "Probability a reset starts at a graph segment entry rather "
            "than at the uniformly-sampled time. Applied after init_start_prob, "
            "so it overrides the t=0 starts as well.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    pre_roll_s: float = field(
        default=0.5,
        metadata={
            "help": "Start up to this many seconds BEFORE the chosen segment "
            "begins, sampled uniformly in [0, pre_roll_s]. A fixed offset would "
            "put every episode at the same phase relative to the transition; the "
            "spread covers the approach as well as the hold.",
            "min": 0.0,
        },
    )
    segment_weighting: str = field(
        default="uniform",
        metadata={
            "help": "How a segment is chosen within a clip. 'uniform' gives every "
            "make/break boundary equal weight (so transitions are over-sampled "
            "relative to their share of clip time, which is the point). 'dwell' "
            "weights by segment duration, recovering roughly uniform-in-time. "
            "'rare_node' weights by 1/sqrt(corpus segment count of the node the "
            "segment holds), which favours configurations the corpus visits least.",
            "options": ["uniform", "dwell", "rare_node"],
        },
    )
