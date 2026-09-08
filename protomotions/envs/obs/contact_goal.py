# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation compute kernels for contact-configuration goals.

Pure tensor functions.  Bind them with :class:`MdpComponent` against
``EnvContext.contact_goal`` -- see
:mod:`protomotions.envs.control.contact_graph_control` for what fills that view.

The goal block is laid out one row per goal step so the MaskedMimic prior can
reshape it into the same token sequence as the sparse target poses and encode
both halves of a goal in a single token:

    [ contact multi-hot (P) | orientation one-hot (O) | visible (1) ]

``visible`` is not redundant with an all-zero contact vector: a configuration
with no active contacts is a real thing (a flight phase, a jump-back), and the
policy has to be able to tell it apart from "this slot was not specified".
"""

from torch import Tensor
import torch


def compute_contact_goal_obs(
    contact_spec: Tensor,
    orient_spec: Tensor,
    visible: Tensor,
) -> Tensor:
    """Flatten the per-goal-step contact specification.

    Args:
        contact_spec: Multi-hot contact pairs [num_envs, steps, pairs], already
            zeroed on hidden slots.
        orient_spec: One-hot orientation bin [num_envs, steps, bins], likewise.
        visible: 1.0 where the contact half is revealed [num_envs, steps].

    Returns:
        [num_envs, steps * (pairs + bins + 1)].
    """
    block = torch.cat(
        [contact_spec, orient_spec, visible.unsqueeze(-1).to(contact_spec.dtype)],
        dim=-1,
    )
    return block.reshape(block.shape[0], -1)


def compute_contact_goal_masks(visible: Tensor) -> Tensor:
    """Per-goal-step visibility of the contact half, as float [num_envs, steps]."""
    return visible.float()


def compute_contact_goal_reached(reached: Tensor) -> Tensor:
    """Weight-0 diagnostic: ground-zone IoU against the nearest goal [num_envs]."""
    return reached


def compute_contact_goal_pose_error(pose_error: Tensor) -> Tensor:
    """Weight-0 diagnostic: metres to the commanded *pose* [num_envs].

    The pose counterpart of ``compute_contact_goal_reached``, which scores the
    contact half only and therefore reads 1.00 for every member of a degenerate
    node (standing, single-leg, four-point). Mean per-body distance in the
    student's own goal representation; see ``ContactGraphControl._goal_pose_error``
    for the exact definition and how unmeasured rows are handled.
    """
    return pose_error


def compute_contact_goal_pose_error_visible(pose_error_visible: Tensor) -> Tensor:
    """Weight-0 diagnostic: fraction of rows the pose error was measured on."""
    return pose_error_visible


def compute_contact_history_obs(history_features: Tensor) -> Tensor:
    """Flatten the measured contact-event history tokens.

    Args:
        history_features: [num_envs, events, features] from ContactEventTracker —
            binary contact/orientation channels plus [0, 1]-scaled times, invalid
            slots already zeroed. Deliberately NOT run through a running
            normalizer, for the same rare-pair reason as ``contact_goal_obs``.

    Returns:
        [num_envs, events * features].
    """
    return history_features.reshape(history_features.shape[0], -1)


def compute_contact_history_masks(history_valid: Tensor) -> Tensor:
    """Per-event-token validity as float [num_envs, events] (1 = attend)."""
    return history_valid.float()


def compute_contact_event_flag(event_commit: Tensor) -> Tensor:
    """1.0 on envs whose contact configuration committed a change this step.

    The ContactEventTracker's debounced make/break event as a per-step flag,
    [num_envs, 1]. The FSQ student reads it as an intent-refresh trigger; it is
    a per-step signal, not accumulated, and is 0.0 everywhere when the tracker
    is disabled.
    """
    return event_commit.reshape(event_commit.shape[0], 1).float()


__all__ = [
    "compute_contact_goal_obs",
    "compute_contact_goal_masks",
    "compute_contact_goal_reached",
    "compute_contact_goal_pose_error",
    "compute_contact_goal_pose_error_visible",
    "compute_contact_history_obs",
    "compute_contact_history_masks",
    "compute_contact_event_flag",
]
