# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""amp_features_v1: agent/demonstration parity, heading invariance, window alignment (PLAN.MD card E2).

The parity test is the card's mandatory one: agent features computed from a state
history buffer filled from the motion library -- exactly as the env's reference
reset fills it (``BaseEnv._reset_state_history``) -- must equal the demonstration
features read from the library at ``t - k * dt`` to <= 1e-5.
"""

from pathlib import Path

import pytest
import torch

from protomotions.agents.amp.component import AMPTrainingComponent
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.amp_features import (
    AMP_FEATURES_V1_STEPS,
    amp_features_v1_dim,
    amp_features_v1_params,
    amp_frame_features_v1,
    compute_amp_features_v1_from_motion_lib,
    compute_amp_features_v1_from_state,
)
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.utils import rotations

REPO = Path(__file__).resolve().parents[2]
RELEASE_DIR = REPO / "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
RELEASE_PACKAGE = RELEASE_DIR / "motions.pt"

SMPL_BODIES = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
               "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
               "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
SMPL_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
CONTROL_DT = 1.0 / 30.0


@pytest.fixture(scope="module")
def params():
    return amp_features_v1_params(SMPL_BODIES, SMPL_PARENTS)


@pytest.fixture(scope="module")
def release_lib():
    if not RELEASE_PACKAGE.exists():
        pytest.skip(f"release package not present: {RELEASE_PACKAGE}")
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    return MotionLib(MotionLibConfig(motion_file=str(RELEASE_PACKAGE)), device="cpu")


def _random_frames(n, b=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, b, 3, generator=g)
    rot = torch.nn.functional.normalize(torch.randn(n, b, 4, generator=g), dim=-1)
    vel = torch.randn(n, b, 3, generator=g)
    ang = torch.randn(n, b, 3, generator=g)
    return pos, rot, vel, ang


def _history_from_reference_reset(lib, motion_ids, motion_times, dt, num_history_steps):
    """StateHistoryBuffer filled exactly as BaseEnv._reset_state_history fills it."""
    n = motion_ids.shape[0]
    size = num_history_steps + 1
    offsets = -dt * torch.arange(size)
    times = (motion_times.unsqueeze(1) + offsets.unsqueeze(0)).clamp(min=0.0)
    times = torch.min(times, lib.motion_lengths[motion_ids].unsqueeze(1).expand(-1, size))
    state = lib.get_motion_state(motion_ids.unsqueeze(1).expand(-1, size).reshape(-1), times.reshape(-1))
    nb = state.rigid_body_pos.shape[1]
    history = StateHistoryBuffer(num_envs=n, num_history_steps=num_history_steps, num_bodies=nb,
                                 num_dofs=state.dof_pos.shape[1], action_dim=state.dof_pos.shape[1],
                                 num_contact_bodies=nb, anchor_body_index=0, device=torch.device("cpu"))
    history.reset_from_states(
        env_ids=torch.arange(n),
        rigid_body_pos=state.rigid_body_pos.view(n, size, nb, 3),
        rigid_body_rot=state.rigid_body_rot.view(n, size, nb, 4),
        rigid_body_vel=state.rigid_body_vel.view(n, size, nb, 3),
        rigid_body_ang_vel=state.rigid_body_ang_vel.view(n, size, nb, 3),
        dof_pos=state.dof_pos.view(n, size, -1),
        dof_vel=state.dof_vel.view(n, size, -1),
        ground_heights=torch.zeros(n, size),
        body_contacts=torch.zeros(n, size, nb, dtype=torch.bool),
    )
    return history


def _agent_features(history, params, steps=AMP_FEATURES_V1_STEPS):
    return compute_amp_features_v1_from_state(
        historical_rigid_body_pos=history.historical_rigid_body_pos,
        historical_rigid_body_rot=history.historical_rigid_body_rot,
        historical_rigid_body_vel=history.historical_rigid_body_vel,
        historical_rigid_body_ang_vel=history.historical_rigid_body_ang_vel,
        historical_ground_heights=history.historical_ground_heights,
        history_steps=list(steps), **params)


def test_dimension_is_163_per_frame(params):
    pos, rot, vel, ang = _random_frames(5)
    f = amp_frame_features_v1(pos, rot, vel, ang, torch.zeros(5), **params)
    assert f.shape == (5, 163) == (5, amp_features_v1_dim(24))


def test_frame_features_are_heading_free(params):
    """A yaw about the world z axis (plus any XY translation) changes nothing."""
    pos, rot, vel, ang = _random_frames(64, seed=1)
    yaw = torch.rand(64) * 6.283
    q = rotations.quat_from_angle_axis(yaw, torch.tensor([[0.0, 0.0, 1.0]]).expand(64, 3), True)
    qb = q.unsqueeze(1).expand(64, 24, 4).reshape(-1, 4)

    def turn(v):
        return rotations.quat_rotate(qb, v.reshape(-1, 3), True).reshape(64, 24, 3)

    shift = torch.randn(64, 1, 3)
    shift[..., 2] = 0.0
    turned = (turn(pos) + shift,
              rotations.quat_mul(qb, rot.reshape(-1, 4), True).reshape(64, 24, 4),
              turn(vel), turn(ang))
    a = amp_frame_features_v1(pos, rot, vel, ang, torch.zeros(64), **params)
    b = amp_frame_features_v1(*turned, torch.zeros(64), **params)
    assert (a - b).abs().max() < 1e-5


def test_window_picks_buffer_index_k_for_step_k(params):
    """Step k of the window is buffer index k (= t - k dt), never index 0 (= now)."""
    n, steps = 3, [1, 2, 4]
    history = StateHistoryBuffer(num_envs=n, num_history_steps=4, num_bodies=24, num_dofs=69, action_dim=69,
                                 num_contact_bodies=24, anchor_body_index=0, device=torch.device("cpu"))
    frames = [_random_frames(n, seed=10 + j) for j in range(5)]
    for j, (p, r, v, a) in enumerate(frames):
        history.rigid_body_pos[:, j] = p
        history.rigid_body_rot[:, j] = r
        history.rigid_body_vel[:, j] = v
        history.rigid_body_ang_vel[:, j] = a
    got = _agent_features(history, params, steps).view(n, len(steps), -1)
    for i, k in enumerate(steps):
        p, r, v, a = frames[k]
        want = amp_frame_features_v1(p, r, v, a, torch.zeros(n), **params)
        assert torch.equal(got[:, i], want)


def test_agent_and_demonstration_features_agree_on_the_release(release_lib):
    """The card's mandatory parity test, on the release-v2 library, to <= 1e-5."""
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("smpl_yogi_v2")
    ki = robot.kinematic_info
    assert list(ki.body_names) == SMPL_BODIES and list(ki.parent_indices) == SMPL_PARENTS
    params = amp_features_v1_params(ki.body_names, ki.parent_indices)

    g = torch.Generator().manual_seed(7)
    n = 512
    ids = torch.randint(0, release_lib.num_motions(), (n,), generator=g)
    lo = max(AMP_FEATURES_V1_STEPS) * CONTROL_DT
    times = lo + torch.rand(n, generator=g) * (release_lib.motion_lengths[ids] - lo)
    history = _history_from_reference_reset(release_lib, ids, times, CONTROL_DT, max(AMP_FEATURES_V1_STEPS))

    agent = _agent_features(history, params)
    demo = compute_amp_features_v1_from_motion_lib(release_lib, ids, times, CONTROL_DT,
                                                   list(AMP_FEATURES_V1_STEPS), **params)
    assert agent.shape == demo.shape == (n, 8 * 163)
    assert torch.isfinite(agent).all()
    assert (agent - demo).abs().max().item() <= 1e-5

    # ... and a one-step misalignment is far outside that tolerance (the test can fail).
    shifted = compute_amp_features_v1_from_motion_lib(release_lib, ids, times + CONTROL_DT, CONTROL_DT,
                                                      list(AMP_FEATURES_V1_STEPS), **params)
    assert (agent - shifted).abs().max().item() > 1e-2


def test_demonstration_component_binds_through_the_amp_context(release_lib, params):
    """The MdpComponent resolves through AMPTrainingComponent._call_ref_obs_fn's runtime context."""
    comp = MdpComponent(compute_func=compute_amp_features_v1_from_motion_lib, dynamic_vars={},
                        static_params={"history_steps": list(AMP_FEATURES_V1_STEPS), **params})
    ids = torch.tensor([0, 1, 2])
    times = torch.tensor([1.0, 2.0, 3.0])
    context = {"motion_lib": release_lib, "motion_ids": ids, "motion_times": times, "dt": CONTROL_DT,
               "num_state_history_steps": 20, "contact_body_ids": None}
    out = AMPTrainingComponent._call_ref_obs_fn(comp.get_compute_func(), context, comp.get_params().copy())
    direct = compute_amp_features_v1_from_motion_lib(release_lib, ids, times, CONTROL_DT,
                                                     list(AMP_FEATURES_V1_STEPS), **params)
    assert torch.equal(out, direct)
