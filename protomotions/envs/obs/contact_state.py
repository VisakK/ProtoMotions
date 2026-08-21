# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the robot is *actually* touching, in the contact graph's own vocabulary.

``contact_goal_obs`` tells the policy which contact pairs it is being asked to
realise.  Until now there was no counterpart telling it which ones it currently
has: ProtoMotions' per-body contact sensors filter against the terrain, so the
policy could feel its hand on the floor but not its shin on its own upper arm.
For poses whose identity *is* a body-body load path -- crow's shins on the upper
arms, koundinyasana's thigh on the elbow, firefly's thighs on the upper arms --
the goal was therefore specified in a language the state could not answer in.

With ``RobotConfig.contact_pair_bodies`` set the simulator reports the per-pair
force matrix, and this kernel pools it into the *same* pair vocabulary the graph
uses, so the goal block and the state block are elementwise comparable:

    slot k of contact_goal_obs  <->  slot k of contact_state_obs

The output is binary on purpose.  Magnitudes are already available at much finer
resolution through ``contact_obs_v1``; what is missing is the categorical fact of
which named pairs are loaded, and a threshold crossing is exactly that fact.
"""

from typing import Optional

import torch
from torch import Tensor


def compute_contact_state_obs(
    ground_forces: Optional[Tensor],
    pair_forces: Optional[Tensor],
    ground_slot: Tensor,
    ground_body: Tensor,
    pair_slot: Tensor,
    pair_body_a: Tensor,
    pair_body_b: Tensor,
    thresholds: Tensor,
    num_pairs: int,
) -> Tensor:
    """Binary contact state over the graph's contact-pair vocabulary.

    Args:
        ground_forces: Per-body force against the terrain, ``[E, B, 3]``, in the
            same (common) body order as ``ground_body`` indexes.
        pair_forces: Per-body-pair force, ``[E, B, P, 3]``. Required whenever
            ``pair_slot`` is non-empty; ``None`` there is a configuration error
            rather than an empty result, because an all-zero body-body half is
            indistinguishable from "nothing is touching".
        ground_slot: Output slot for each (zone, body) membership, ``[n_ground]``.
        ground_body: Body index for each of those memberships, ``[n_ground]``.
        pair_slot: Output slot for each (zone-pair, body, body) membership.
        pair_body_a: First body index of each such membership (axis 1).
        pair_body_b: Second body index (axis 2 -- indexes ``contact_pair_bodies``).
        thresholds: Newtons above which each slot counts as in contact, ``[P]``.
            Per-slot rather than one scalar because the ground and body-body
            halves are calibrated separately (the graph uses 3 % of body weight
            for ground and 2 % for body-body).
        num_pairs: Width of the output, ``P``.

    Returns:
        ``[E, P]`` float tensor of 0.0/1.0.
    """
    if ground_forces is None:
        raise ValueError(
            "contact_state_obs requires rigid_body_ground_forces; this backend "
            "reported None. Only simulators whose per-body contact sensors "
            "filter against the terrain provide it."
        )
    num_envs = ground_forces.shape[0]
    total = torch.zeros(
        num_envs, num_pairs, device=ground_forces.device, dtype=ground_forces.dtype
    )

    if ground_slot.numel():
        total.index_add_(
            1, ground_slot, ground_forces.norm(dim=-1)[:, ground_body]
        )

    if pair_slot.numel():
        if pair_forces is None:
            raise ValueError(
                "contact_state_obs was configured with body-body pairs but the "
                "simulator reported no rigid_body_pair_contact_forces. Set "
                "RobotConfig.contact_pair_bodies so the per-body contact sensors "
                "also filter against the robot's own bodies."
            )
        # A zone pair is read in BOTH directions and combined with max, not sum.
        # Newton's third law makes the two equal in magnitude when both sensors
        # report, so summing would silently double every body-body force -- and
        # taking one direction alone would read zero whenever PhysX happened to
        # populate only the other. ``pair_slot`` therefore addresses a [P, 2]
        # buffer as ``slot * 2 + direction``.
        magnitude = pair_forces.norm(dim=-1)
        directional = torch.zeros(
            num_envs, num_pairs * 2, device=total.device, dtype=total.dtype
        )
        directional.index_add_(
            1, pair_slot, magnitude[:, pair_body_a, pair_body_b]
        )
        total = total + directional.view(num_envs, num_pairs, 2).amax(dim=-1)

    return (total > thresholds).to(ground_forces.dtype)


__all__ = ["compute_contact_state_obs"]
