# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unwanted-support penalty: ground load on zones the commanded hold keeps free.

Design and offline calibration: ``expert_revist/contact_reward/README.MD``. Replaces the
contact-matching reward this project switched off (``notes/Contact_label_consequence.MD`` §8)
with the one-sided, force-based term that note recommends, plus a label gate::

    free_z = z not in the commanded hold's ground set
             and  min over z's bodies of the reference joint-centre height > clear_height
    p      = gate * clamp( sum_z free_z * sum_{b in z} max(F_gnd_z,b, 0) / load_ref_n, 0, 1 )

Properties, each one a failure an earlier contact term had:

* **One-sided.** It never asks for contact, so a support the SMPL fit floats (the
  headstand head, 13.5-14.2 cm) cannot pull the policy into the floor.
* **Force, not a flag.** ``F_gnd`` is the terrain-filtered column (``rigid_body_ground_forces``),
  continuous in newtons: there is no binary state to flick (the foot-hopping exploit), and
  body-body contact -- crow's foot-foot stack, the shins on the upper arms -- never counts.
* **Both conditions.** Reference height alone charges float-biased supports; the label alone
  charges the known float-bias mislabels (Plow's feet). Requiring both is what removed every false
  positive in the calibration on the ft_a policy (README §3.3).

* **Optionally time-averaged** (:class:`ChargedLoadEMA`, ``support_ema_tau_s > 0``). The clamp
  above is per frame, which prices *how often* a free zone is down rather than *how much* it
  carries. ft_b learned to exploit exactly that by tapping. Averaging the charged load before the
  clamp closes the loophole.

Pure tensor functions; ``ContactGraphControl`` supplies the goal tables and the gate, and owns the
average's state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def unwanted_support(
    ground_forces: Optional[Tensor],
    ref_body_pos: Tensor,
    goal_ground: Tensor,
    in_hold: Tensor,
    zone_matrix: Tensor,
    clear_height: float,
    load_ref_n: float,
    excluded: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Penalty in [0, 1] and the charged load in newtons, both ``[E]``.

    Args:
        ground_forces: Terrain-filtered contact force per body ``[E, B, 3]`` in the robot's
            common body order. ``None`` (a backend without the column) yields zeros.
        ref_body_pos: Reference body positions at the current clip time ``[E, B, 3]``, same
            body order; heights are read as ``z`` (flat terrain).
        goal_ground: ``[E, Z]`` bool, True where zone ``z`` is in the commanded hold's ground
            contact set.
        in_hold: ``[E]`` bool gate, True while the clip is inside the commanded hold segment.
        zone_matrix: ``[Z, B]`` 0/1 pooling of bodies into zones.
        clear_height: A zone counts as kept clear by the reference when its lowest member's
            joint centre is above this (m).
        load_ref_n: Charged load at which the penalty saturates (N).
        excluded: Optional ``[E]`` bool; True rows are never charged.
    """
    num_envs = ref_body_pos.shape[0]
    device = ref_body_pos.device
    if ground_forces is None:
        zeros = torch.zeros(num_envs, device=device)
        return zeros, zeros
    zm = zone_matrix.to(device=device, dtype=ground_forces.dtype)
    fz = ground_forces[..., 2].clamp_min(0.0)                        # [E, B]
    load = fz @ zm.t()                                                # [E, Z]
    member = zm > 0.5                                                 # [Z, B]
    ref_z = ref_body_pos[..., 2].unsqueeze(1)                         # [E, 1, B]
    inf = torch.full_like(ref_z, float("inf"))
    zone_min_z = torch.where(member.unsqueeze(0), ref_z, inf).amin(dim=-1)   # [E, Z]
    free = (~goal_ground.bool()) & (zone_min_z > clear_height)
    gate = in_hold.bool()
    if excluded is not None:
        gate = gate & ~excluded.bool()
    charged = (load * free.to(load.dtype)).sum(dim=-1) * gate.to(load.dtype)   # [E]
    penalty = (charged / float(load_ref_n)).clamp(0.0, 1.0)
    return penalty, charged


def ema_alpha(tau_s: float, dt: float) -> float:
    """Per-step weight of a causal exponential moving average with time constant ``tau_s``.

    ``tau_s <= 0`` gives 1.0, i.e. no averaging.
    """
    if tau_s <= 0.0:
        return 1.0
    return float(dt) / (float(tau_s) + float(dt))


class ChargedLoadEMA:
    """Time-averages the charged load before the clamp, so the penalty prices load, not duty cycle.

    The per-frame term, ``clamp(charged / load_ref_n, 0, 1)``, prices a foot's duty cycle. A foot
    resting at 250 N and one striking at 650 N both cost 1.0 while down and 0 while up. ft_b's
    Crow -b learned exactly that: it tapped at 3-4 Hz and kept 68-82 % of its resting support for a
    quarter of the penalty (``expert_revist/ft_b_support/report.MD`` §4).

    Averaging over ``tau_s`` first makes a tapping foot pay what a resting foot with the same mean
    load pays, while a held lift stays nearly free. On ft_b's own rollouts at 0.25 s (report §8):
    Crow -b's tapping 0.13 -> 0.56, the same as a resting foot; Side Crow -c's held lift 0.01 -> 0.04.
    The result is still one-sided and bounded, and it is gated exactly like the per-frame term.

    The average advances once per env step. ``mark_step()`` comes from the control's ``step()``,
    and the next ``price()`` consumes it, so a context rebuilt again in the same step (after resets,
    or by probe drivers) reuses the same average. ``reset()`` zeroes the rows of new episodes, which
    gives each episode's first hold a ramp of about ``tau_s``.
    """

    def __init__(self, num_envs: int, tau_s: float, dt: float, device=None) -> None:
        if tau_s <= 0.0:
            raise ValueError("ChargedLoadEMA needs tau_s > 0; tau_s == 0 is the per-frame term")
        self.alpha = ema_alpha(tau_s, dt)
        self.state = torch.zeros(num_envs, device=device)
        self._pending = False

    def mark_step(self) -> None:
        """Arm one update for the next ``price()``: called once per env step."""
        self._pending = True

    def reset(self, env_ids: Tensor) -> None:
        self.state[env_ids] = 0.0

    def price(self, charged: Tensor, gate: Tensor, load_ref_n: float) -> Tensor:
        """Penalty ``[E]`` in [0, 1] from the averaged charged load (N), times ``gate``.

        ``charged`` is the per-frame load from :func:`unwanted_support`, already zero outside
        the gate. The average therefore decays between holds, and multiplying by ``gate`` keeps
        the penalty inside them, as in the per-frame term.
        """
        if self._pending:
            self._pending = False
            self.state.lerp_(charged.to(self.state.dtype), self.alpha)
        return (self.state / float(load_ref_n)).clamp(0.0, 1.0) * gate.to(self.state.dtype)


def zone_pooling_matrix(zone_bodies: dict, zone_order: list, body_names: list) -> Tensor:
    """``[Z, B]`` 0/1 matrix, zones in ``zone_order``; every zone must resolve to a body."""
    index = {n: i for i, n in enumerate(body_names)}
    m = torch.zeros(len(zone_order), len(body_names))
    for z, zone in enumerate(zone_order):
        members = [index[b] for b in zone_bodies[zone] if b in index]
        if not members:
            raise ValueError(f"zone {zone} has no body in {body_names}")
        m[z, members] = 1.0
    return m
