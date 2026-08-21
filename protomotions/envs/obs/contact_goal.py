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


__all__ = [
    "compute_contact_goal_obs",
    "compute_contact_goal_masks",
    "compute_contact_goal_reached",
]
