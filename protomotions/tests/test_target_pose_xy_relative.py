"""Location-free student goals keep pose geometry without prescribing floor XY."""

import math

import pytest
import torch

from protomotions.envs.obs.masked_mimic import compute_target_poses_only
from protomotions.envs.obs.target_poses import (
    build_max_coords_target_poses,
    build_sparse_target_poses,
)


def _states():
    generator = torch.Generator().manual_seed(314)
    current = {}
    future = {}
    for name, size in (("pos", 3), ("rot", 4), ("vel", 3), ("ang_vel", 3)):
        current[name] = torch.randn(2, 3, size, generator=generator)
        future[name] = torch.randn(2, 4, 3, size, generator=generator)
    # Nontrivial root heading/tilt and limb rotations exercise frame handling.
    current["rot"] = torch.nn.functional.normalize(current["rot"], dim=-1)
    future["rot"] = torch.nn.functional.normalize(future["rot"], dim=-1)
    return current, future


def _observe(kind, current, future, *, full=True, steps=None, masks=None, **kwargs):
    if kind == "dense":
        return build_max_coords_target_poses(
            current["pos"], current["rot"], current["vel"], current["ang_vel"],
            future["pos"], future["rot"], future["vel"], future["ang_vel"],
            with_velocities=True, w_last=True, future_steps=steps,
            with_relative=full, **kwargs,
        )
    # Deliberately exclude root and reverse body order: alignment must use the
    # original root before conditionable-body selection.
    body_ids = torch.tensor([2, 1])
    args = dict(
        current_state_body_pos=current["pos"],
        current_state_body_rot=current["rot"],
        masked_mimic_ref_pos=future["pos"],
        masked_mimic_ref_rot=future["rot"],
        conditionable_body_ids=body_ids,
        future_steps=steps,
        include_root_relative=full,
        **kwargs,
    )
    if kind == "sparse":
        return build_sparse_target_poses(w_last=True, **args)
    if masks is None:
        masks = torch.ones(future["pos"].shape[0], future["pos"].shape[1] * 4)
    return compute_target_poses_only(masked_mimic_target_bodies_masks=masks, **args)


@pytest.mark.parametrize("kind", ["dense", "sparse", "masked"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("steps, count", [(None, 4), (1, 1), ([1, 4], 2)])
def test_independent_floor_translations_are_removed_but_legacy_default_is_preserved(
    kind, full, steps, count,
):
    current, future = _states()
    current_shift = torch.tensor([[[2.3, -1.7, 0.0]], [[-4.2, 0.8, 0.0]]])
    # Each target frame and each environment has an independent XY translation.
    future_shift = torch.tensor([
        [[[0.2, 3.7, 0.0]], [[4.1, -3.2, 0.0]], [[-2.1, 1.4, 0.0]], [[1.7, 6.2, 0.0]]],
        [[[2.9, 1.6, 0.0]], [[-0.4, 2.5, 0.0]], [[5.4, 0.9, 0.0]], [[-3.2, -1.6, 0.0]]],
    ])
    shifted_current = {**current, "pos": current["pos"] + current_shift}
    shifted_future = {**future, "pos": future["pos"] + future_shift}
    options = dict(full=full, steps=steps)
    fixed = _observe(kind, current, future, root_relative_xy=True, **options)
    legacy = _observe(kind, current, future, **options)
    torch.testing.assert_close(
        legacy, _observe(kind, current, future, root_relative_xy=False, **options),
        rtol=0, atol=0,
    )
    for observed_current, observed_future in (
        (shifted_current, future),
        (current, shifted_future),
        (shifted_current, shifted_future),
    ):
        torch.testing.assert_close(
            fixed,
            _observe(kind, observed_current, observed_future, root_relative_xy=True, **options),
            rtol=2e-5, atol=3e-6,
        )
        assert not torch.allclose(
            legacy, _observe(kind, observed_current, observed_future, **options)
        )
    features_per_step = (72 if full else 45) if kind == "dense" else (48 if full else 24)
    assert fixed.shape == legacy.shape == (2, count * features_per_step)


@pytest.mark.parametrize("kind", ["dense", "sparse", "masked"])
@pytest.mark.parametrize("enabled", [False, True])
def test_shared_world_translation_remains_invariant(kind, enabled):
    current, future = _states()
    shift = torch.tensor([[[2.3, -1.7, 0.0]], [[-4.2, 0.8, 0.0]]])
    translated_current = {**current, "pos": current["pos"] + shift}
    translated_future = {**future, "pos": future["pos"] + shift[:, None]}
    torch.testing.assert_close(
        _observe(kind, current, future, root_relative_xy=enabled),
        _observe(kind, translated_current, translated_future, root_relative_xy=enabled),
        rtol=2e-5, atol=3e-6,
    )


@pytest.mark.parametrize("yaw", [0.0, 0.73, -1.9])
def test_pose_offsets_height_rotations_and_velocity_errors_survive(yaw):
    current, future = _states()
    current_offsets = torch.tensor([[0., 0., 0.], [.4, .1, -.3], [-.2, .8, .5]])
    target_offsets = torch.tensor([[0., 0., 0.], [.9, -.2, -.7], [.6, -.5, .2]])
    current["pos"] = (current_offsets + torch.tensor([5., -3., 1.4]))[None].expand(2, -1, -1)
    future["pos"] = (target_offsets + torch.tensor([12., 8., 2.1]))[None, None].expand(2, 4, -1, -1)
    current["rot"][:, 0] = torch.tensor([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)])
    world_to_heading = torch.tensor([
        [math.cos(yaw), math.sin(yaw), 0.],
        [-math.sin(yaw), math.cos(yaw), 0.],
        [0., 0., 1.],
    ])
    expected_root = (target_offsets + torch.tensor([0., 0., .7])) @ world_to_heading.T
    expected_delta = (target_offsets - current_offsets + torch.tensor([0., 0., .7])) @ world_to_heading.T

    dense = _observe("dense", current, future, root_relative_xy=True).view(2, 4, 72)
    legacy_dense = _observe("dense", current, future).view(2, 4, 72)
    torch.testing.assert_close(dense[:, :, :9], expected_root.flatten().expand(2, 4, -1))
    torch.testing.assert_close(dense[:, :, 9:18], expected_delta.flatten().expand(2, 4, -1))
    # Everything after positions is unchanged: relative/heading rotations and
    # true world-derivative velocity errors, including root translation speed.
    torch.testing.assert_close(dense[:, :, 18:], legacy_dense[:, :, 18:], rtol=0, atol=0)
    expected_vel = (future["vel"] - current["vel"][:, None]) @ world_to_heading.T
    expected_ang_vel = (future["ang_vel"] - current["ang_vel"][:, None]) @ world_to_heading.T
    torch.testing.assert_close(dense[:, :, 54:63], expected_vel.flatten(2))
    torch.testing.assert_close(dense[:, :, 63:72], expected_ang_vel.flatten(2))

    for kind in ("sparse", "masked"):
        sparse = _observe(kind, current, future, root_relative_xy=True).view(2, 4, 2, 24)
        legacy_sparse = _observe(kind, current, future).view(2, 4, 2, 24)
        torch.testing.assert_close(sparse[..., :3], expected_delta[[2, 1]].expand(2, 4, -1, -1))
        torch.testing.assert_close(sparse[..., 6:9], expected_root[[2, 1]].expand(2, 4, -1, -1))
        torch.testing.assert_close(sparse[..., 12:], legacy_sparse[..., 12:], rtol=0, atol=0)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("steps, indices", [(None, [0, 1, 2, 3]), (1, [0]), ([1, 4], [0, 3])])
def test_masked_targets_stay_zero_with_frame_selection(full, steps, indices):
    current, future = _states()
    masks = torch.tensor([
        [[0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
        [[1, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 1], [0, 0, 0, 0]],
    ], dtype=torch.float)
    width = 12 if full else 6
    obs = _observe(
        "masked", current, future, root_relative_xy=True, full=full,
        steps=steps, masks=masks.flatten(1),
    ).view(2, len(indices), 2, 2, width)
    visible = masks[:, indices].view(2, len(indices), 2, 2, 1).bool().expand_as(obs)
    assert torch.count_nonzero(obs[~visible]) == 0
    unmasked = _observe(
        "sparse", current, future, root_relative_xy=True, full=full, steps=steps,
    ).view_as(obs)
    torch.testing.assert_close(obs[visible], unmasked[visible], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["dense", "sparse", "masked"])
def test_xy_removal_preserves_inputs_and_pose_gradients(kind):
    current, future = _states()
    current["pos"].requires_grad_()
    future["pos"].requires_grad_()
    before_current = {key: value.detach().clone() for key, value in current.items()}
    before_future = {key: value.detach().clone() for key, value in future.items()}
    obs = _observe(kind, current, future, root_relative_xy=True)
    weights = torch.linspace(.1, 1., obs.numel()).reshape_as(obs)
    (obs * weights).sum().backward()
    for state, before in ((current, before_current), (future, before_future)):
        for key in state:
            torch.testing.assert_close(state[key].detach(), before[key], rtol=0, atol=0)
        gradient = state["pos"].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[..., 1:, :2]) > 0
        assert torch.count_nonzero(gradient[..., 2]) > 0
        # Uniform XY translation has zero gradient for the current pose and
        # independently for every future frame; articulated pose gradients live.
        torch.testing.assert_close(
            gradient[..., :2].sum(dim=-2),
            torch.zeros_like(gradient[..., :2].sum(dim=-2)),
            atol=3e-6, rtol=0,
        )
