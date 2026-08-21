# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put a measured pressure mat and a simulated foot on the same grid.

The MOYO mat (``notes/Moyo_pressure_port.MD``) is a 110 x 37 array of 12.7 mm
cells reading N/cm2, and the Tier-0 archive stores its placement **in the
reference clip's frame**.  A rollout, meanwhile, reports contact points in world
coordinates at a terrain location that has nothing to do with the clip.  This
module is the bridge: one grid object that both sides rasterise onto, so the two
bird's-eye images are the same pixels of the same square metre and can be
subtracted.

Three things here are load-bearing and easy to get wrong:

* **The mat axes are not exactly orthonormal.**  ``mat_ex`` / ``mat_ey`` come from
  a marker fit, so ``ex . ey`` is ~1e-3 rather than 0.  Inverting the map with
  dot products (``u = (p - o) . ex``) instead of a 2x2 solve is wrong by a
  fraction of a cell at the far end of a 1.4 m mat.  :meth:`MatGrid.to_cell`
  solves.
* **A simulator has no pressure.**  PhysX reports a handful of contact *points*
  with a normal force each and no area at all, so a raw rasterisation is a set of
  delta spikes reading hundreds of N/cm2 against a mat whose peak is ~15.  The
  fix is to spread both sides with the *same* Gaussian -- flesh spreads load over
  roughly the 2 cm the attribution kernel already uses -- and to compute every
  *number* (totals, COP, shares) from the unsmoothed data.  Smoothing is for the
  eye only.
* **The pad is not decoration.**  Load that runs off the mat is the documented
  failure mode of this capture (57 of 170 clips), and a grid clipped to the
  sensing area would hide exactly the simulated contacts that explain a
  disagreement.  The grid is padded and the sensed rectangle drawn on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 12.7 mm cells -> 1.27 cm square -> 1.6129 cm2, the divisor that turns the
# archive's N/cm2 into newtons per cell.
CELL_AREA_CM2 = 1.6129
PRESSURE_FPS = 60.0


@dataclass
class MatGrid:
    """A regular grid of mat cells expressed in the reference clip's frame.

    ``origin`` is the outer corner of cell (0, 0); cell ``(i, j)`` spans
    ``[j, j+1) * cell`` along ``ex`` and ``[i, i+1) * cell`` along ``ey``, so its
    centre is at ``origin + (j + 0.5) cell ex + (i + 0.5) cell ey``.  This is
    exactly the convention ``pressure_bodies.load_archive`` uses.

    ``sensed`` is the sub-block that the physical mat actually measures, as
    ``(row0, col0, ny, nx)``; everything outside it is pad.
    """

    origin: np.ndarray            # (2,) clip-frame xy of the outer corner
    ex: np.ndarray                # (2,) column direction
    ey: np.ndarray                # (2,) row direction
    cell: float                   # metres
    shape: Tuple[int, int]        # (ny, nx) including pad
    sensed: Tuple[int, int, int, int]

    # -- construction --------------------------------------------------------
    @classmethod
    def from_archive(cls, archive: dict, pad_cells: int = 8) -> "MatGrid":
        ny, nx = archive["mat_shape"]
        cell = float(archive["cell_size_m"])
        ex = np.asarray(archive["mat_ex"], dtype=np.float64)
        ey = np.asarray(archive["mat_ey"], dtype=np.float64)
        origin = np.asarray(archive["mat_origin_xy"], dtype=np.float64)
        padded_origin = origin - pad_cells * cell * ex - pad_cells * cell * ey
        return cls(
            origin=padded_origin,
            ex=ex,
            ey=ey,
            cell=cell,
            shape=(int(ny) + 2 * pad_cells, int(nx) + 2 * pad_cells),
            sensed=(pad_cells, pad_cells, int(ny), int(nx)),
        )

    # -- geometry ------------------------------------------------------------
    def _basis(self) -> np.ndarray:
        """Columns are ex, ey -- the matrix that maps (u, v) metres to clip xy."""
        return np.stack([self.ex, self.ey], axis=1)

    def cell_centres(self) -> np.ndarray:
        """(ny, nx, 2) clip-frame centre of every cell."""
        ny, nx = self.shape
        jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
        return (
            self.origin[None, None, :]
            + ((jj + 0.5) * self.cell)[..., None] * self.ex
            + ((ii + 0.5) * self.cell)[..., None] * self.ey
        )

    def cell_corners(self) -> np.ndarray:
        """(ny+1, nx+1, 2) clip-frame corners, for ``pcolormesh``."""
        ny, nx = self.shape
        jj, ii = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
        return (
            self.origin[None, None, :]
            + (jj * self.cell)[..., None] * self.ex
            + (ii * self.cell)[..., None] * self.ey
        )

    def to_cell(self, pts_xy: np.ndarray) -> np.ndarray:
        """Clip-frame points (M, 2) -> fractional (row, col), via a 2x2 solve.

        Fractional, not integer, so callers can bin (``floor``) or interpolate.
        The solve rather than a pair of dot products is deliberate: ``ex`` and
        ``ey`` are a marker fit and are only orthonormal to ~1e-3.
        """
        pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
        uv = np.linalg.solve(self._basis(), (pts - self.origin[None, :]).T).T
        return np.stack([uv[:, 1] / self.cell, uv[:, 0] / self.cell], axis=1)

    def sensed_polygon(self) -> np.ndarray:
        """(5, 2) closed clip-frame outline of the physically sensed rectangle."""
        r0, c0, ny, nx = self.sensed
        corners = [(r0, c0), (r0, c0 + nx), (r0 + ny, c0 + nx), (r0 + ny, c0)]
        pts = [
            self.origin + j * self.cell * self.ex + i * self.cell * self.ey
            for i, j in corners
        ]
        return np.asarray(pts + [pts[0]])

    def sensed_slice(self):
        r0, c0, ny, nx = self.sensed
        return (slice(r0, r0 + ny), slice(c0, c0 + nx))

    def orthonormality(self) -> Tuple[float, float, float]:
        """``(ex.ey, |ex|, |ey|)`` -- the numbers that justify the 2x2 solve."""
        return (
            float(self.ex @ self.ey),
            float(np.linalg.norm(self.ex)),
            float(np.linalg.norm(self.ey)),
        )


# --------------------------------------------------------------------------- #
# rasterising each side
# --------------------------------------------------------------------------- #
def measured_field(archive: dict, frame: int, grid: MatGrid) -> np.ndarray:
    """The mat's own reading for one frame, placed on ``grid``. N/cm2."""
    out = np.zeros(grid.shape, dtype=np.float64)
    dense = archive["dense"](int(frame))
    out[grid.sensed_slice()] = dense
    return out


def simulated_field(
    pts_xy: np.ndarray, forces_n: np.ndarray, grid: MatGrid
) -> np.ndarray:
    """Ground contact points -> N/cm2 on ``grid``, by binning then dividing by area.

    No spreading here: the raw rasterisation is the honest one and is what every
    reported number is computed from.  :func:`spread` is applied afterwards, to
    both sides, purely so the images are comparable.
    """
    out = np.zeros(grid.shape, dtype=np.float64)
    if len(pts_xy) == 0:
        return out
    rc = grid.to_cell(pts_xy)
    rows = np.floor(rc[:, 0]).astype(int)
    cols = np.floor(rc[:, 1]).astype(int)
    inside = (
        (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    )
    if not inside.any():
        return out
    np.add.at(out, (rows[inside], cols[inside]), np.asarray(forces_n)[inside])
    return out / CELL_AREA_CM2


def spread(field: np.ndarray, sigma_m: float, cell: float) -> np.ndarray:
    """Gaussian blur in metres, applied identically to both sides.

    A rigid simulator concentrates a whole limb's load on 1-6 solver points; a
    real foot spreads it over a patch.  Blurring both fields by the scale the
    pressure attribution already assumes (2 cm, ``Moyo_pressure_port.MD`` 4.2)
    makes the two images comparable without pretending the simulator measured a
    patch it never had.
    """
    if sigma_m <= 0:
        return field
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(field, sigma=sigma_m / cell, mode="constant", cval=0.0)


# --------------------------------------------------------------------------- #
# scalars, always from the unsmoothed data
# --------------------------------------------------------------------------- #
def field_total_n(field_pressure: np.ndarray) -> float:
    """Total newtons in an N/cm2 field."""
    return float(field_pressure.sum() * CELL_AREA_CM2)


def field_cop(field_pressure: np.ndarray, grid: MatGrid) -> Optional[np.ndarray]:
    """Pressure-weighted centroid in clip-frame xy, or None if nothing is loaded."""
    total = field_pressure.sum()
    if total <= 0:
        return None
    centres = grid.cell_centres()
    return np.array(
        [
            float((field_pressure * centres[..., 0]).sum() / total),
            float((field_pressure * centres[..., 1]).sum() / total),
        ]
    )


def weighted_cop(pts_xy: np.ndarray, forces_n: np.ndarray) -> Optional[np.ndarray]:
    """Force-weighted centroid of raw contact points -- no rasterisation loss."""
    if len(pts_xy) == 0:
        return None
    w = np.asarray(forces_n, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        return None
    return (np.asarray(pts_xy, dtype=np.float64) * w[:, None]).sum(0) / total


def frame_for_time(t_seconds: float, n_frames: int, fps: float = PRESSURE_FPS) -> int:
    """Motion time -> measured-pressure frame index, clamped to the clip."""
    return int(np.clip(round(float(t_seconds) * fps), 0, n_frames - 1))


HOLD_ANKLE_M = 0.25
HOLD_MIN_FRAMES = 20


def reference_hold(ankle_z: np.ndarray, mode: str = "auto",
                   threshold: float = HOLD_ANKLE_M):
    """Which frames are 'the pose', from the reference's two ankle heights.

    ``ankle_z`` is ``[T, 2]``.  Two definitions, because the corpus contains two
    kinds of balance and one definition silently answers ``never`` for the other:

    * ``feet_up`` -- **both** ankles clear the floor.  Right for an inversion or
      an arm balance, and what every hard-29 number is computed on.
    * ``single_leg`` -- **exactly one** ankle clears it.  A standing balance
      never lifts both feet: measured over the 14 single-leg clips, ``feet_up``
      selects **0.0 %** of frames and ``single_leg`` selects 67 %.  Without this,
      every hold-phase metric on those clips reads ``n/a``.

    ``auto`` prefers ``feet_up`` and falls back, so the hard poses are unaffected.
    Returns ``(mask, mode_used)``.
    """
    z = np.asarray(ankle_z, dtype=np.float64)
    up = z > threshold
    both, exactly_one = up.all(1), (up.sum(1) == 1)
    if mode == "feet_up":
        return both, "feet_up"
    if mode == "single_leg":
        return exactly_one, "single_leg"
    if mode != "auto":
        raise ValueError(f"unknown hold mode {mode!r}")
    if both.sum() >= HOLD_MIN_FRAMES:
        return both, "feet_up"
    if exactly_one.sum() >= HOLD_MIN_FRAMES:
        return exactly_one, "single_leg"
    return both, "feet_up"          # neither: keep the strict answer, report n/a


def on_mat_fraction(field_pressure: np.ndarray, grid: MatGrid) -> float:
    """Share of a field's load that falls inside the sensed rectangle.

    Applied to the *simulated* field this is the number that says whether a
    disagreement is the policy doing something different or the policy simply
    standing where the mat could never have seen it.
    """
    total = field_pressure.sum()
    if total <= 0:
        return float("nan")
    return float(field_pressure[grid.sensed_slice()].sum() / total)
