# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry and balance maths behind the contact-physics analysis.

These are the pieces a wrong answer would silently corrupt every figure with:
the convex hull, the polygon area, the signed margin (which decides whether the
COM is inside the support polygon), the friction combine rule and the contact
manifold area.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data", "scripts"))

from contact_physics_support import (  # noqa: E402
    LIMB_GROUPS,
    combine_friction,
    convex_hull_2d,
    group_dof_indices,
    hull_vertex_owners,
    joint_of_dof,
    manifold_area,
    polygon_area,
    signed_margin,
    step_slices,
)

SMPL_DOFS = [
    f"{joint}_{axis}"
    for joint in (
        "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
        "Torso", "Spine", "Chest", "Neck", "Head",
        "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
        "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
    )
    for axis in "xyz"
]


def test_convex_hull_drops_interior_points():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.3, 0.7]], float)
    hull = convex_hull_2d(square)
    assert len(hull) == 4
    assert polygon_area(hull) == pytest.approx(1.0)


def test_convex_hull_is_counter_clockwise():
    hull = convex_hull_2d(np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float))
    signed = 0.5 * sum(
        hull[i, 0] * hull[(i + 1) % len(hull), 1] - hull[(i + 1) % len(hull), 0] * hull[i, 1]
        for i in range(len(hull))
    )
    assert signed > 0


def test_degenerate_hulls_have_no_area():
    assert polygon_area(convex_hull_2d(np.zeros((0, 2)))) == 0.0
    assert polygon_area(convex_hull_2d(np.array([[1.0, 2.0]]))) == 0.0
    collinear = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    assert polygon_area(convex_hull_2d(collinear)) == pytest.approx(0.0, abs=1e-12)


def test_signed_margin_sign_and_value():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    hull = convex_hull_2d(square)
    assert signed_margin(np.array([0.5, 0.5]), hull) == pytest.approx(0.5)
    assert signed_margin(np.array([0.1, 0.5]), hull) == pytest.approx(0.1)
    assert signed_margin(np.array([-0.25, 0.5]), hull) == pytest.approx(-0.25)
    # A single contact point or a line of them supports nothing statically.
    assert signed_margin(np.array([0.0, 0.0]), np.array([[0.3, 0.0]])) == pytest.approx(-0.3)
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert signed_margin(np.array([0.5, 0.2]), line) == pytest.approx(-0.2)


def test_cop_of_a_convex_manifold_is_always_inside_its_hull():
    """The property the whole COP/support-polygon reading depends on."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        points = rng.normal(size=(rng.integers(3, 9), 2))
        forces = rng.uniform(0.1, 100.0, size=len(points))
        cop = (points * forces[:, None]).sum(0) / forces.sum()
        assert signed_margin(cop, convex_hull_2d(points)) >= -1e-9


def test_manifold_area_of_a_flat_foot_patch():
    """Four coplanar contact points on the floor: the hull is the real patch."""
    points = np.array(
        [[0.0, 0.0, 0.0], [0.16, 0.0, 0.0], [0.16, 0.09, 0.0], [0.0, 0.09, 0.0]]
    )
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    assert manifold_area(points, normals) == pytest.approx(0.16 * 0.09, rel=1e-6)


def test_manifold_area_is_measured_in_the_contact_plane():
    """A patch on a wall must keep its area, not collapse under xy projection."""
    points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1], [0.0, 0.0, 0.1]]
    )
    normals = np.tile([1.0, 0.0, 0.0], (4, 1))
    assert manifold_area(points, normals) == pytest.approx(0.02, rel=1e-6)


def test_manifold_area_needs_three_points():
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    assert manifold_area(np.zeros((2, 3)), normals) == 0.0


def test_hull_vertex_owners_maps_back_to_bodies():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0]], float)
    owners = np.array([7, 3, 3, 9])
    hull = convex_hull_2d(points[:, :2])
    found = hull_vertex_owners(points, owners, hull)
    assert set(found) == {7, 3}
    assert 9 not in found  # the interior point owns no vertex


def test_friction_combine_matches_physx_rules():
    assert combine_friction(1.0, 0.5, "average") == pytest.approx(0.75)
    assert combine_friction(1.0, 0.5, "min") == pytest.approx(0.5)
    assert combine_friction(1.0, 0.5, "max") == pytest.approx(1.0)
    assert combine_friction(1.0, 0.5, "multiply") == pytest.approx(0.5)
    assert combine_friction(1.0, 0.5, "unknown-mode") == pytest.approx(0.75)


def test_step_slices_bucket_a_step_sorted_record_array():
    steps = np.array([0, 0, 2, 2, 2, 5])
    starts = step_slices(steps, 7)
    assert starts.tolist() == [0, 2, 2, 5, 5, 5, 6, 6]
    assert steps[starts[2]:starts[3]].tolist() == [2, 2, 2]
    assert steps[starts[1]:starts[2]].size == 0


def test_limb_groups_partition_every_smpl_dof():
    grouped = group_dof_indices(SMPL_DOFS)
    covered = sorted(d for entries in grouped.values() for _, idx in entries for d in idx)
    assert covered == list(range(len(SMPL_DOFS)))
    assert set(grouped) == set(LIMB_GROUPS)
    assert [j for j, _ in grouped["left_leg"]] == ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"]


def test_limb_groups_reject_an_unknown_joint():
    with pytest.raises(ValueError, match="do not partition"):
        group_dof_indices(SMPL_DOFS + ["Tail_x"])
