# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heading-free AMP discriminator features (graph-growth round, card E2).

``amp_features_v1`` replaces the stock ``historical_max_coords_obs`` as the
discriminator input of the goal-conditioned expert
(``expert_revist/graph_growth_2026_10_03/PLAN.MD`` §1.2). The stock observation
normalises every frame by ``calc_heading_quat_inv``, which projects the root's
local x axis and is ill-conditioned on the prone and inverted frames (6.1 % of
this corpus, ``heading_chart_audit.py``). Nothing here depends on the heading.

Per frame, 163 values on the 24-body skeleton:

* root height above the ground (1);
* gravity in the root frame (3);
* root linear and angular velocity in the root frame (3 + 3);
* every non-root body's rotation relative to its parent, as tan-norm 6D (23 x 6);
* head, hands and toes relative to the root, in the root frame (5 x 3).

The window is ``AMP_FEATURES_V1_STEPS`` frames back from "now": step ``k`` is
the state at ``t - k * dt`` with ``dt`` the control step. On the agent side that
is the state history buffer's index ``k`` (index 0 = now; ``EnvContext.historical``
is ``buffer[:, 1:]``, so step ``k`` is its ``k - 1``), exactly the convention of
the stock ``historical_max_coords_obs``. On the demonstration side it is the
motion library sampled at ``t - k * dt``. Both sides call the same per-frame
function, ``amp_frame_features_v1``; ``protomotions/tests/test_amp_features_v1.py``
holds them equal to <= 1e-5 on the release library.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
from torch import Tensor

from protomotions.envs.obs.utils import select_step_indices
from protomotions.utils import rotations

# 8 frames over 0.67 s at 30 Hz, dense at the start so control-rate jitter is visible.
AMP_FEATURES_V1_STEPS: Tuple[int, ...] = (1, 2, 3, 4, 6, 9, 14, 20)
AMP_FEATURES_V1_KEY_BODIES: Tuple[str, ...] = ("Head", "L_Hand", "R_Hand", "L_Toe", "R_Toe")


def amp_features_v1_params(body_names: Sequence[str], parent_indices: Sequence[int],
                           key_body_names: Sequence[str] = AMP_FEATURES_V1_KEY_BODIES) -> dict:
    """Static index lists for ``amp_frame_features_v1`` from the robot's kinematic tree.

    Plain int lists (not tensors) so the same dict serves the env component, the
    demonstration component and ``torch.compile`` without device bookkeeping.
    """
    body_names = list(body_names)
    child_ids = [j for j, p in enumerate(parent_indices) if p >= 0]
    if len(child_ids) != len(body_names) - 1 or parent_indices[0] != -1:
        raise ValueError(f"expected a single root at index 0, got parents {list(parent_indices)}")
    return {
        "child_ids": child_ids,
        "parent_ids": [int(parent_indices[j]) for j in child_ids],
        "key_body_ids": [body_names.index(name) for name in key_body_names],
    }


def amp_features_v1_dim(num_bodies: int, num_key_bodies: int = len(AMP_FEATURES_V1_KEY_BODIES)) -> int:
    """Per-frame feature width: 10 + 6 (B - 1) + 3 K (163 for SMPL's 24 bodies)."""
    return 10 + 6 * (num_bodies - 1) + 3 * num_key_bodies


def amp_frame_features_v1(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    ground_height: Tensor,
    child_ids: List[int],
    parent_ids: List[int],
    key_body_ids: List[int],
) -> Tensor:
    """Heading-free features of single frames.

    Args:
        body_pos / body_vel / body_ang_vel: ``[N, B, 3]`` world frame.
        body_rot: ``[N, B, 4]`` world rotations, xyzw.
        ground_height: ``[N]`` or ``[N, 1]`` terrain height under the root.
        child_ids / parent_ids: every non-root body and its parent.
        key_body_ids: bodies whose root-frame position is observed.

    Returns:
        ``[N, 10 + 6 * len(child_ids) + 3 * len(key_body_ids)]``.
    """
    n = body_pos.shape[0]
    root_pos = body_pos[:, 0]
    root_rot = body_rot[:, 0]

    height = root_pos[:, 2:3] - ground_height.reshape(n, 1)
    down = torch.zeros_like(root_pos)
    down[:, 2] = -1.0
    gravity = rotations.quat_rotate_inverse(root_rot, down, True)
    lin_vel = rotations.quat_rotate_inverse(root_rot, body_vel[:, 0], True)
    ang_vel = rotations.quat_rotate_inverse(root_rot, body_ang_vel[:, 0], True)

    child = body_rot[:, child_ids]
    parent = body_rot[:, parent_ids]
    local = rotations.quat_mul(rotations.quat_conjugate(parent, True), child, True)
    local_6d = rotations.quat_to_tan_norm(local.reshape(-1, 4), True).reshape(n, -1)

    k = len(key_body_ids)
    rel = (body_pos[:, key_body_ids] - root_pos.unsqueeze(1)).reshape(n * k, 3)
    root_rot_k = root_rot.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
    key_local = rotations.quat_rotate_inverse(root_rot_k, rel, True).reshape(n, 3 * k)

    return torch.cat([height, gravity, lin_vel, ang_vel, local_6d, key_local], dim=-1)


def compute_amp_features_v1_from_state(
    historical_rigid_body_pos: Tensor,
    historical_rigid_body_rot: Tensor,
    historical_rigid_body_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor,
    historical_ground_heights: Tensor,
    history_steps: Union[List[int], Tuple[int, ...]],
    child_ids: List[int],
    parent_ids: List[int],
    key_body_ids: List[int],
) -> Tensor:
    """Agent side: the window from the state history buffer (``EnvContext.historical``).

    Returns ``[N, len(history_steps) * F]``, frames in ``history_steps`` order.
    """
    steps = list(history_steps)
    n = historical_rigid_body_pos.shape[0]
    num_bodies = historical_rigid_body_pos.shape[2]
    s = len(steps)

    pos = select_step_indices(historical_rigid_body_pos, steps).reshape(n * s, num_bodies, 3)
    rot = select_step_indices(historical_rigid_body_rot, steps).reshape(n * s, num_bodies, 4)
    vel = select_step_indices(historical_rigid_body_vel, steps).reshape(n * s, num_bodies, 3)
    ang = select_step_indices(historical_rigid_body_ang_vel, steps).reshape(n * s, num_bodies, 3)
    ground = select_step_indices(historical_ground_heights, steps).reshape(n * s)

    frames = amp_frame_features_v1(pos, rot, vel, ang, ground, child_ids, parent_ids, key_body_ids)
    return frames.reshape(n, s * frames.shape[-1])


def compute_amp_features_v1_from_motion_lib(
    motion_lib,
    motion_ids: Tensor,
    motion_times: Tensor,
    dt: float,
    history_steps: Union[List[int], Tuple[int, ...]],
    child_ids: List[int],
    parent_ids: List[int],
    key_body_ids: List[int],
) -> Tensor:
    """Demonstration side: the same window read from the motion library at ``t - k * dt``.

    Times are clamped to ``[0, motion_length]`` exactly as the env's reference
    reset clamps them when it fills the history buffer; the demonstration
    sampler keeps ``t >= max(steps) * dt`` so no demonstration window is clamped.
    The floor is flat (ground height 0), as in the reference reset.
    """
    steps = torch.as_tensor(list(history_steps), device=motion_times.device, dtype=motion_times.dtype)
    n = motion_ids.shape[0]
    s = steps.numel()
    times = (motion_times.unsqueeze(1) - steps.unsqueeze(0) * dt).clamp(min=0.0)
    lengths = motion_lib.motion_lengths[motion_ids].to(device=times.device, dtype=times.dtype)
    times = torch.min(times, lengths.unsqueeze(1).expand(-1, s))
    ids = motion_ids.unsqueeze(1).expand(-1, s).reshape(-1)

    state = motion_lib.get_motion_state(ids, times.reshape(-1))
    ground = torch.zeros(n * s, device=state.rigid_body_pos.device, dtype=state.rigid_body_pos.dtype)
    frames = amp_frame_features_v1(state.rigid_body_pos, state.rigid_body_rot, state.rigid_body_vel,
                                   state.rigid_body_ang_vel, ground, child_ids, parent_ids, key_body_ids)
    return frames.reshape(n, s * frames.shape[-1])
