# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hold extension v2 (BUILD_PLAN Step 9, BodyFix Step 5): duration variants that keep the measured pressure on
their real frames and mark it invalid on the inserted ones.

``make_hold_extended_clips.py`` (v1) bakes each extendable hold into its clip in several durations (0, 3 and
7 s): the exemplar frame is repeated after itself and the velocities are recomputed from the spliced
kinematics. It drops every measured pressure field, because a tiled pressure frame would be an invented
measurement, so the physics tables re-timed the x0 pressure port through ``splice_index`` and read the
exemplar's measurement again on every inserted frame.

v2 keeps v1's kinematics exactly (``extend_motion``: the same splice, the same velocity recomputation) and
carries the three measured channels of the gated port (``ground_reaction``, ``rigid_body_ground_forces``,
``ground_reaction_valid``):

* **real frames** keep the source frame's measurement, untouched;
* **inserted frames** carry zeros with every validity column 0. Validity is authoritative: 0 means "no
  measurement", never "zero load" (the relabeling design, section 16);
* the **d = 0 variant is the gated port, byte for byte** (copied, not re-saved).

The lineage of every variant -- the x0 source frame of each frame and whether it was inserted -- is returned
for ``lineage.npz``: the contact-target sidecar maps per-frame evidence through it.
"""

from __future__ import annotations

from typing import Iterable

import torch

from make_hold_extended_clips import (
    RECOMPUTED_FIELDS,
    extend_motion,
    insertion_plan,
    shifted_hold,
    splice_index,
    velocity_reproduction_error,
)

PRESSURE_FIELDS = ("ground_reaction", "rigid_body_ground_forces", "ground_reaction_valid")
MAX_VELOCITY_DRIFT = 0.02      # make_hold_extended_clips.py --max-velocity-drift (mean, m/s and rad/s)

__all__ = ["PRESSURE_FIELDS", "inserted_mask", "extend_motion_v2", "variant_stem", "variant_holds",
           "insertion_plan", "splice_index", "shifted_hold", "velocity_drift"]


def variant_stem(stem: str, variant_s: float) -> str:
    """``<stem>`` for d = 0, ``<stem>_x<d>s`` otherwise (v1's naming)."""
    return stem if int(round(variant_s)) == 0 else f"{stem}_x{int(round(variant_s))}s"


def inserted_mask(index: torch.Tensor) -> torch.Tensor:
    """``[T]`` True on the frames a splice inserted: those repeating the previous frame's source frame."""
    out = torch.zeros(index.shape[0], dtype=torch.bool)
    if index.shape[0] > 1:
        out[1:] = index[1:] == index[:-1]
    return out


def extend_motion_v2(motion: dict, plan: list[tuple[int, int]]) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """``(variant, source frame index [T'], inserted [T'])``: v1's kinematic splice plus the measured channels.

    Raises if the motion has only some of the pressure fields: the port carries all three or none.
    """
    present = [k for k in PRESSURE_FIELDS if motion.get(k) is not None]
    if present and len(present) != len(PRESSURE_FIELDS):
        raise ValueError(f"the motion carries only {present} of the measured channels {PRESSURE_FIELDS}")
    kinematic = {k: v for k, v in motion.items() if k not in PRESSURE_FIELDS}
    out = extend_motion(kinematic, plan)
    index = splice_index(int(motion["rigid_body_pos"].shape[0]), plan)
    inserted = inserted_mask(index)
    for key in present:
        value = motion[key].index_select(0, index).clone()
        value[inserted] = 0
        out[key] = value
    return out, index, inserted


def variant_holds(holds: Iterable[dict], plan: list[tuple[int, int]], fps: int) -> list[dict]:
    """The clip's holds re-timed for a variant (``shifted_hold``: every other field, ``hold_id`` included, kept)."""
    return [shifted_hold(h, plan, fps) for h in holds]


def velocity_drift(motion: dict) -> dict:
    """``{field: (mean, max)}`` of |stored - recomputed| velocities on an unmodified clip: the converter-convention
    guard v1 applies before extending (a mean above ``MAX_VELOCITY_DRIFT`` means the conventions changed)."""
    kinematic = {k: v for k, v in motion.items() if k not in PRESSURE_FIELDS}
    out = velocity_reproduction_error(kinematic)
    assert set(out) == set(RECOMPUTED_FIELDS)
    return out
