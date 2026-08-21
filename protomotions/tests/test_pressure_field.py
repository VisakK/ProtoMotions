# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The maths that puts a measured pressure mat and a simulated foot on one grid.

Everything here is a place where a wrong answer would look completely plausible:
a half-cell error in the mat's inverse map, a pressure/newton conversion off by
the cell area, a body-order remap that files a hand's load under a wrist.  The
resulting picture would still be two heatmaps with blobs in them.

The rendering side is deliberately not tested -- it is checked by eye, which is
what the figure is for.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data", "scripts"))

from pressure_field import (  # noqa: E402
    CELL_AREA_CM2,
    MatGrid,
    field_cop,
    field_total_n,
    frame_for_time,
    measured_field,
    on_mat_fraction,
    reference_hold,
    simulated_field,
    spread,
    weighted_cop,
)
from render_planning import (  # noqa: E402
    outcome_suffix,
    plan_rollout,
    select_motions,
)

CELL = 0.0127
REPO = os.path.join(os.path.dirname(__file__), "..", "..")
ARCHIVE_DIR = os.path.join(REPO, "data", "smpl", "yoga_pressure")


def make_archive(ny=110, nx=37, ex=(-1.0, 0.0), ey=(0.0, -1.0), origin=(0.25, 1.43)):
    """A Tier-0-shaped archive dict with a controllable basis."""
    return {
        "mat_shape": (ny, nx),
        "cell_size_m": CELL,
        "mat_ex": np.asarray(ex, dtype=np.float64),
        "mat_ey": np.asarray(ey, dtype=np.float64),
        "mat_origin_xy": np.asarray(origin, dtype=np.float64),
    }


def forward(grid: MatGrid, row: float, col: float) -> np.ndarray:
    """The archive's own cell-centre formula, reproduced independently."""
    return (grid.origin + (col + 0.5) * grid.cell * grid.ex
            + (row + 0.5) * grid.cell * grid.ey)


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def test_cell_centres_match_the_archive_formula():
    grid = MatGrid.from_archive(make_archive(), pad_cells=0)
    centres = grid.cell_centres()
    for row, col in ((0, 0), (7, 3), (109, 36)):
        assert np.allclose(centres[row, col], forward(grid, row, col))


def test_to_cell_inverts_the_forward_map():
    grid = MatGrid.from_archive(make_archive(), pad_cells=4)
    rows, cols = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]),
                             indexing="ij")
    got = grid.to_cell(grid.cell_centres().reshape(-1, 2))
    want = np.stack([rows.ravel(), cols.ravel()], axis=1) + 0.5
    assert np.abs(got - want).max() < 1e-9


def test_to_cell_is_exact_on_a_non_orthonormal_basis():
    """The mat axes are a marker fit, so ``ex . ey`` is ~1e-3, not 0.

    A dot-product inverse (``u = (p - o) . ex``) round-trips cell *centres*
    perfectly, so the obvious unit test passes while real points bin into the
    wrong cell.  This one uses off-centre points, which is what actually breaks.
    """
    skew = make_archive(ex=(-0.99997938, -0.00641785), ey=(0.00734822, -0.99997294))
    grid = MatGrid.from_archive(skew, pad_cells=0)

    probes = [(3.27, 11.6), (54.9, 18.2), (108.4, 35.1)]
    for row, col in probes:
        p = forward(grid, row - 0.5, col - 0.5)
        got = grid.to_cell(p[None])[0]
        assert np.abs(got - np.array([row, col])).max() < 1e-9

    # ...and confirm the naive inverse really is wrong, so this test is not
    # guarding against nothing.
    p = forward(grid, 108.4 - 0.5, 35.1 - 0.5)
    d = p - grid.origin
    naive = np.array([d @ grid.ey, d @ grid.ex]) / grid.cell
    assert np.abs(naive - np.array([108.4, 35.1])).max() > 0.05


def test_pad_keeps_the_sensed_rectangle_addressable():
    grid = MatGrid.from_archive(make_archive(), pad_cells=8)
    assert grid.shape == (110 + 16, 37 + 16)
    assert grid.sensed == (8, 8, 110, 37)
    unpadded = MatGrid.from_archive(make_archive(), pad_cells=0)
    padded_centres = grid.cell_centres()[grid.sensed_slice()]
    assert np.abs(padded_centres - unpadded.cell_centres()).max() < 1e-12


def test_sensed_polygon_is_closed_and_the_right_size():
    grid = MatGrid.from_archive(make_archive(), pad_cells=8)
    poly = grid.sensed_polygon()
    assert poly.shape == (5, 2)
    assert np.allclose(poly[0], poly[-1])
    side_a = np.linalg.norm(poly[1] - poly[0])
    side_b = np.linalg.norm(poly[2] - poly[1])
    assert pytest.approx(37 * CELL, abs=1e-9) == side_a
    assert pytest.approx(110 * CELL, abs=1e-9) == side_b


# --------------------------------------------------------------------------- #
# rasterising
# --------------------------------------------------------------------------- #
def test_measured_field_lands_inside_the_sensed_block():
    archive = make_archive()
    dense = np.zeros((110, 37), dtype=np.float32)
    dense[5, 6] = 4.0
    archive["dense"] = lambda t: dense
    grid = MatGrid.from_archive(archive, pad_cells=8)
    field = measured_field(archive, 0, grid)
    assert field[8 + 5, 8 + 6] == 4.0
    assert field.sum() == 4.0
    assert pytest.approx(4.0 * CELL_AREA_CM2) == field_total_n(field)


def test_simulated_field_conserves_force_and_bins_by_position():
    grid = MatGrid.from_archive(make_archive(), pad_cells=8)
    target = [(20, 11), (20, 11), (77, 30)]
    pts = np.array([forward(grid, r, c) for r, c in target])
    forces = np.array([30.0, 70.0, 200.0])
    field = simulated_field(pts, forces, grid)
    assert pytest.approx(300.0, rel=1e-9) == field_total_n(field)
    assert pytest.approx(100.0 / CELL_AREA_CM2) == field[20, 11]
    assert pytest.approx(200.0 / CELL_AREA_CM2) == field[77, 30]


def test_simulated_field_drops_points_off_the_grid_rather_than_wrapping():
    """Negative indices would wrap round to the far edge and invent a contact."""
    grid = MatGrid.from_archive(make_archive(), pad_cells=2)
    inside = forward(grid, 40, 20)
    outside = grid.origin - 5.0 * grid.ex - 5.0 * grid.ey
    field = simulated_field(np.stack([inside, outside]), np.array([10.0, 90.0]), grid)
    assert pytest.approx(10.0, rel=1e-9) == field_total_n(field)
    assert (field > 0).sum() == 1


def test_simulated_field_handles_no_contacts():
    grid = MatGrid.from_archive(make_archive(), pad_cells=2)
    assert simulated_field(np.zeros((0, 2)), np.zeros(0), grid).sum() == 0.0


# --------------------------------------------------------------------------- #
# blur, totals, COP
# --------------------------------------------------------------------------- #
def test_spread_is_a_no_op_at_zero_sigma_and_conserves_load():
    grid = MatGrid.from_archive(make_archive(), pad_cells=8)
    field = simulated_field(forward(grid, 55, 18)[None], np.array([500.0]), grid)
    assert np.array_equal(spread(field, 0.0, CELL), field)
    blurred = spread(field, 0.02, CELL)
    assert pytest.approx(500.0, rel=1e-3) == field_total_n(blurred)
    assert blurred.max() < field.max()          # the point really was spread


def test_spread_does_not_move_the_centre_of_pressure():
    """Blur is cosmetic, so it must not shift where the load appears to be."""
    grid = MatGrid.from_archive(make_archive(), pad_cells=10)
    pts = np.stack([forward(grid, 50, 14), forward(grid, 50, 24)])
    field = simulated_field(pts, np.array([400.0, 200.0]), grid)
    before = field_cop(field, grid)
    after = field_cop(spread(field, 0.02, CELL), grid)
    assert np.linalg.norm(before - after) < 1e-3


def test_field_cop_matches_the_force_weighted_point_cop():
    grid = MatGrid.from_archive(make_archive(), pad_cells=8)
    pts = np.stack([forward(grid, 30, 10), forward(grid, 60, 25)])
    forces = np.array([300.0, 100.0])
    field = simulated_field(pts, forces, grid)
    assert np.linalg.norm(field_cop(field, grid) - weighted_cop(pts, forces)) < 1e-9


def test_cop_is_none_when_nothing_is_loaded():
    grid = MatGrid.from_archive(make_archive(), pad_cells=2)
    assert field_cop(np.zeros(grid.shape), grid) is None
    assert weighted_cop(np.zeros((0, 2)), np.zeros(0)) is None
    assert weighted_cop(np.zeros((2, 2)), np.zeros(2)) is None


def test_on_mat_fraction_separates_sensed_from_pad():
    grid = MatGrid.from_archive(make_archive(), pad_cells=8)
    on = forward(grid, 50, 18)          # inside the sensed rectangle
    off = forward(grid, 50, 3)          # on the grid, but in the pad
    field = simulated_field(np.stack([on, off]), np.array([75.0, 25.0]), grid)
    assert pytest.approx(0.75) == on_mat_fraction(field, grid)
    assert pytest.approx(100.0, rel=1e-9) == field_total_n(field)


# --------------------------------------------------------------------------- #
# which frames are "the pose"
# --------------------------------------------------------------------------- #
def _ankles(pairs):
    return np.array(pairs, dtype=np.float64)


def test_reference_hold_feet_up_needs_both_ankles():
    z = _ankles([(0.05, 0.05), (0.40, 0.05), (0.40, 0.40), (0.05, 0.40)])
    mask, mode = reference_hold(z, "feet_up")
    assert mode == "feet_up"
    assert list(mask) == [False, False, True, False]


def test_reference_hold_single_leg_needs_exactly_one():
    z = _ankles([(0.05, 0.05), (0.40, 0.05), (0.40, 0.40), (0.05, 0.40)])
    mask, mode = reference_hold(z, "single_leg")
    assert mode == "single_leg"
    assert list(mask) == [False, True, False, True]


def test_reference_hold_auto_prefers_feet_up_so_hard_poses_are_unchanged():
    z = _ankles([(0.40, 0.40)] * 30 + [(0.40, 0.05)] * 30)
    mask, mode = reference_hold(z, "auto")
    assert mode == "feet_up"
    assert mask.sum() == 30


def test_reference_hold_auto_falls_back_for_a_standing_balance():
    """The regression this exists for: a pose that never lifts both feet.

    Measured over the 14 single-leg clips, the feet-up definition selects
    **0.0 %** of frames and single-leg selects 67 %. Without the fallback every
    hold-phase metric on those clips reads n/a and the most useful column in the
    review table is empty.
    """
    z = _ankles([(0.40, 0.05)] * 40 + [(0.05, 0.05)] * 20)
    mask, mode = reference_hold(z, "auto")
    assert mode == "single_leg"
    assert mask.sum() == 40


def test_reference_hold_auto_keeps_the_strict_answer_when_neither_qualifies():
    z = _ankles([(0.05, 0.05)] * 50)
    mask, mode = reference_hold(z, "auto")
    assert mode == "feet_up"
    assert mask.sum() == 0


def test_reference_hold_auto_ignores_a_handful_of_stray_frames():
    """A few both-up frames in a standing clip must not select the wrong mode."""
    z = _ankles([(0.40, 0.40)] * 5 + [(0.40, 0.05)] * 40 + [(0.05, 0.05)] * 15)
    _, mode = reference_hold(z, "auto")
    assert mode == "single_leg"


def test_reference_hold_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        reference_hold(_ankles([(0.4, 0.4)]), "sideways")


def test_frame_for_time_clamps_rather_than_wrapping():
    assert frame_for_time(0.0, 100) == 0
    assert frame_for_time(1.0, 100) == 60
    assert frame_for_time(1.0 / 60.0, 100) == 1
    assert frame_for_time(-5.0, 100) == 0
    assert frame_for_time(1e6, 100) == 99


# --------------------------------------------------------------------------- #
# against the real archive
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(ARCHIVE_DIR), reason="pressure archive absent")
@pytest.mark.parametrize("clip", [
    "220923_Crane_Crow_Pose_or_Bakasana_-a",
    "220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a",
])
def test_real_archive_reproduces_its_own_force_and_cop(clip):
    """The grid is only right if it recovers the numbers the vendor stored.

    ``total_force_n`` and ``cop_world`` were written by a different script from
    the raw field, so reproducing both from ``mat_origin/ex/ey`` alone checks the
    placement, the cell area and the inverse map at once.
    """
    from pressure_bodies import load_archive

    path = os.path.join(ARCHIVE_DIR, f"{clip}.npz")
    if not os.path.isfile(path):
        pytest.skip(f"{clip} not archived")
    archive = load_archive(path)
    grid = MatGrid.from_archive(archive, pad_cells=8)

    frames = np.linspace(0, archive["n_frames"] - 1, 15).astype(int)
    for t in frames:
        field = measured_field(archive, t, grid)
        stored = float(archive["total_force_n"][t])
        if stored < 1.0:
            continue
        assert field_total_n(field) == pytest.approx(stored, rel=1e-5)
        cop = field_cop(field, grid)
        assert np.linalg.norm(cop - archive["cop_world"][t]) < 1e-3   # 1 mm


@pytest.mark.skipif(not os.path.isdir(ARCHIVE_DIR), reason="pressure archive absent")
def test_real_mat_axes_are_not_orthonormal():
    """Guards the reason ``to_cell`` solves instead of projecting.

    If this ever starts failing the basis has been orthogonalised upstream and
    the 2x2 solve becomes redundant -- but until then, removing it would bin
    real contacts one cell out at the far end of the mat.
    """
    from pressure_bodies import load_archive

    path = os.path.join(ARCHIVE_DIR, "220923_Crane_Crow_Pose_or_Bakasana_-a.npz")
    if not os.path.isfile(path):
        pytest.skip("clip not archived")
    grid = MatGrid.from_archive(load_archive(path), pad_cells=0)
    dot, nx, ny = grid.orthonormality()
    assert abs(dot) > 1e-5
    assert nx == pytest.approx(1.0, abs=1e-6)
    assert ny == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# video planning
# --------------------------------------------------------------------------- #
DT = 1.0 / 30.0


def test_plan_rollout_records_the_whole_clip_by_default():
    """The regression that motivated all of this: a 12 s cap on a 36 s pose."""
    steps, target, replay = plan_rollout(35.72, DT, max_seconds=0.0, min_seconds=15.0)
    assert steps == pytest.approx(35.72 / DT, abs=1)
    assert target == pytest.approx(35.72)
    assert replay is False


def test_plan_rollout_honours_an_explicit_cap():
    steps, target, replay = plan_rollout(35.72, DT, max_seconds=20.0, min_seconds=15.0)
    assert target == pytest.approx(20.0)
    assert steps == 600
    assert replay is False


def test_plan_rollout_replays_a_clip_shorter_than_the_floor():
    steps, target, replay = plan_rollout(9.0, DT, max_seconds=0.0, min_seconds=15.0)
    assert target == pytest.approx(15.0)
    assert steps == 450
    assert replay is True


def test_plan_rollout_does_not_replay_a_clip_that_already_clears_the_floor():
    for length in (15.0, 16.15, 59.02):
        _, _, replay = plan_rollout(length, DT, 0.0, 15.0)
        assert replay is False


def test_plan_rollout_lets_an_explicit_cap_outrank_the_floor():
    """A cap the caller typed is a hard limit.

    The bug this guards: computing the floor first gave 15 s for
    ``--max-seconds 10``, i.e. half again as much video as asked for -- the same
    class of surprise as the 12 s default this function exists to remove.
    """
    steps, target, replay = plan_rollout(35.0, DT, max_seconds=10.0, min_seconds=15.0)
    assert target == pytest.approx(10.0)
    assert steps == 300
    assert replay is False


def test_plan_rollout_never_arms_a_replay_it_cannot_use():
    """``allow_replay`` must mean "the target exceeds the clip", nothing else.

    Armed wrongly, the renderer waits for a clip-end reset that will not come
    before the cap, so the flag is dead weight at best and confusing at worst.
    """
    # cap below the clip: no replay, even though the floor is higher than the cap
    _, target, replay = plan_rollout(9.0, DT, max_seconds=5.0, min_seconds=15.0)
    assert target == pytest.approx(5.0)
    assert replay is False
    # cap above the clip but below the floor: replay up to the cap only
    _, target, replay = plan_rollout(9.0, DT, max_seconds=12.0, min_seconds=15.0)
    assert target == pytest.approx(12.0)
    assert replay is True


def test_outcome_suffix_distinguishes_a_fall_from_a_short_clip():
    assert outcome_suffix(31.4, "full", None) == "_31.4s"
    assert outcome_suffix(16.1, "clipend", None) == "_16.1s_clipend"
    assert outcome_suffix(7.3, "fell", 6.3) == "_07.3s_FELL@06.3s"
    # zero padding keeps the names sorting by duration
    assert outcome_suffix(9.0, "full", None) < outcome_suffix(11.0, "full", None)


def test_select_motions_filters_by_name_after_ranking():
    names = ["crow_a", "handstand", "side_crow_b"]
    assert [e[0] for e in select_motions(names, None, None, None, ["crow"])] == [0, 2]
    assert [e[0] for e in select_motions(names, None, None, None, ["HAND"])] == [1]
    assert [e[0] for e in select_motions(names, None, None, [2, 0], None)] == [2, 0]
    assert len(select_motions(names, None, None, None, None)) == 3
