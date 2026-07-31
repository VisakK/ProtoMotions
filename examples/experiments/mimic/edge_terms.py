# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extra edge terms: localised tracking error, terminal goal state, tail termination.

Motivation (measured, see ``notes/Skill_graph_handstand_lessons.MD`` §5.1-5.2).
Both composition failures are *localised* configuration errors that the standard
mimic objective barely sees:

* the kick-up slides its left palm 19.5 cm off the support point;
* the exit lands with the legs **crossed** (L/R ankle order reversed in 100 % of
  envs), 0.22 m off at the ankles.

``gt_rew = exp(-25 * MSE)`` averages squared error over **all 24 bodies**, so one
body wrong by 0.20 m contributes 0.04/24 to the MSE and costs
``exp(-0.042) ~ 4 %`` of the position reward -- essentially free. Meanwhile the
termination uses **max** over bodies at 0.5 m, which 0.22 m passes comfortably.
Nothing in between penalises "one or two bodies badly wrong", which is exactly
the failure mode. These three components fill that gap.
"""

from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent


# --------------------------------------------------------------------------- #
# 1. Localised tracking error: top-k instead of mean-over-24.
# --------------------------------------------------------------------------- #
def compute_topk_gt_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    k: int = 4,
    coefficient: float = -15.0,
) -> Tensor:
    """exp(coef * mean of the k largest squared per-body position errors).

    Top-k rather than max: max has gradient through a single body and switches
    discontinuously as which body is worst changes; k=4 is enough to cover a
    limb (ankle+toe on both sides, or wrist+hand) while staying localised.
    """
    sq = (ref_rigid_body_pos - current_rigid_body_pos).pow(2).sum(-1)   # [envs, bodies]
    topk = sq.topk(min(k, sq.shape[-1]), dim=-1)[0].mean(dim=-1)
    return torch.exp(coefficient * topk)


def topk_gt_rew_factory(k: int = 4, coefficient: float = -15.0,
                        weight: float = 0.3) -> MdpComponent:
    """Penalise a few badly-placed bodies, which the mean-over-24 gt_rew dilutes.

    Coefficient calibrated against measurement: for the trained exit edge the
    top-4 MSE is ~0.048 typical and ~0.146 at p99, so coef -15 keeps the term in
    a discriminative range (0.62 typical, 0.23 at p99) instead of saturating at 0
    the way coef -25 would.
    """
    return MdpComponent(
        compute_func=compute_topk_gt_rew,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params={"k": k, "coefficient": coefficient, "weight": weight},
    )


# --------------------------------------------------------------------------- #
# 1b. Localised VELOCITY tracking: top-k instead of mean-over-24.
# --------------------------------------------------------------------------- #
def compute_topk_gv_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    k: int = 4,
    coefficient: float = -2.0,
) -> Tensor:
    """exp(coef * mean of the k largest squared per-body velocity errors).

    The velocity twin of ``compute_topk_gt_rew``, aimed at the measured v2
    failure (``notes/Skill_graph_handstand_lessons_2.MD`` §3.1): peak body speed
    1.83x the reference while pelvis vz was only 1.12x -- a few limbs moving
    twice as fast as the human, invisible to the mean-over-24 ``gv_rew``.
    """
    sq = (ref_rigid_body_vel - current_rigid_body_vel).pow(2).sum(-1)  # [envs, bodies]
    topk = sq.topk(min(k, sq.shape[-1]), dim=-1)[0].mean(dim=-1)
    return torch.exp(coefficient * topk)


def topk_gv_rew_factory(k: int = 4, coefficient: float = -2.0,
                        weight: float = 0.3) -> MdpComponent:
    """Penalise a few badly-MOVING bodies, which the mean-over-24 gv_rew dilutes.

    The dilution arithmetic (why gv could not see the 1.83x failure): one limb
    0.9 m/s too fast gives per-body squared error 0.81; mean-over-24 = 0.034, so
    ``gv_rew = exp(-0.5 * 0.034) = 0.983`` -- at weight 0.1 that costs ~0.002
    reward, essentially free. Top-4: 0.81/4 = 0.2025, ``exp(-2 * 0.2025) = 0.67``
    -- at weight 0.3 that costs ~0.10, a ~60x stronger and correctly-localised
    gradient. Coefficient -2 (not -15 like the position twin) because velocity
    errors are in (m/s)^2 and typical good-tracking top-4 velocity MSE (~0.1)
    should sit mid-range (exp(-0.2) ~ 0.82), not saturate.
    """
    return MdpComponent(
        compute_func=compute_topk_gv_rew,
        dynamic_vars={
            "current_rigid_body_vel": EnvContext.current.rigid_body_vel,
            "ref_rigid_body_vel": EnvContext.mimic.ref_state.rigid_body_vel,
        },
        static_params={"k": k, "coefficient": coefficient, "weight": weight},
    )


# --------------------------------------------------------------------------- #
# 2. Terminal goal state: pose AND velocity against the target node's pose,
#    over the last tail_steps of the clip.
# --------------------------------------------------------------------------- #
class NodeTerminalGoal:
    """Reward for arriving at the target node's pose, at rest.

    The node reference has zero velocity by construction (frozen clip), so the
    velocity half is simply "come to rest" -- which is the anti-jerk objective.
    It is deliberately asymmetric in usefulness: near-stationary arrivals score
    ~1 regardless, while a fast, jerky arrival is punished hard.

    Only the node's *pose* is needed here, not its critic, so this is much
    cheaper than NodeValueJoin.
    """

    def __init__(self, node_motion_pt: str, clip_steps: int, tail_steps: int = 30,
                 pos_coef: float = -15.0, vel_coef: float = -10.0, topk: int = 4):
        self.node_motion_pt = node_motion_pt
        self.clip_steps = int(clip_steps)
        self.tail_steps = int(tail_steps)
        self.pos_coef = float(pos_coef)
        self.vel_coef = float(vel_coef)
        self.topk = int(topk)
        self._ready = False

    def _build(self, device):
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        ml = MotionLib(MotionLibConfig(motion_file=self.node_motion_pt), device=str(device))
        st = ml.get_motion_state(
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, device=device),
        )
        self.ref_pos = st.rigid_body_pos[0].clone().to(device)
        del ml
        self._ready = True

    @torch._dynamo.disable
    def __call__(
        self,
        body_pos: Tensor,
        body_vel: Tensor,
        respawn_root_offset: Tensor,
        progress_buf: Tensor,
    ) -> Tensor:
        if not self._ready:
            self._build(body_pos.device)
        n = body_pos.shape[0]

        # XY-only translation, matching what the env applies to its own reference
        xy = torch.zeros_like(respawn_root_offset)
        xy[:, :2] = respawn_root_offset[:, :2]
        ref = self.ref_pos.unsqueeze(0).expand(n, -1, -1) + xy.unsqueeze(1)

        sq = (ref - body_pos).pow(2).sum(-1)
        pos_term = torch.exp(
            self.pos_coef * sq.topk(min(self.topk, sq.shape[-1]), dim=-1)[0].mean(dim=-1)
        )
        vel_term = torch.exp(self.vel_coef * body_vel.pow(2).sum(-1).mean(dim=-1))

        # Linear ramp over the last tail_steps; zero before. Safe as a *terminal*
        # objective specifically because it carries the lowest weight -- it
        # refines an already-tracking policy rather than driving early learning
        # (a hard gate as the primary signal was measured to be unlearnable,
        # notes §7.2).
        start = self.clip_steps - self.tail_steps
        ramp = ((progress_buf.float() - start) / max(self.tail_steps, 1)).clamp(0.0, 1.0)
        return ramp * 0.5 * (pos_term + vel_term)


def terminal_goal_rew_factory(node_motion_pt: str, clip_steps: int,
                              tail_steps: int = 30, weight: float = 0.1,
                              pos_coef: float = -15.0,
                              vel_coef: float = -10.0) -> MdpComponent:
    fn = NodeTerminalGoal(node_motion_pt, clip_steps, tail_steps, pos_coef, vel_coef)
    return MdpComponent(
        compute_func=fn,
        dynamic_vars={
            "body_pos": EnvContext.current.rigid_body_pos,
            "body_vel": EnvContext.current.rigid_body_vel,
            "respawn_root_offset": EnvContext.respawn_root_offset,
            "progress_buf": EnvContext.progress_buf,
        },
        static_params={"weight": weight},
    )


# --------------------------------------------------------------------------- #
# 3. Tail termination: a tighter tracking bound, but only near the hand-off.
# --------------------------------------------------------------------------- #
def compute_tail_tracking_error(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    progress_buf: Tensor,
    threshold: float = 0.25,
    clip_steps: int = 159,
    tail_steps: int = 30,
) -> Tensor:
    """Terminate on max per-body error > threshold, but only in the clip's tail.

    A *uniform* tighter bound is not viable: measured on the trained exit edge,
    the max per-body error is 0.242 m mean / 0.505 m worst, peaking mid-clip
    where the legs come down fast, and **100 % of envs** exceed 0.25 m at some
    point. Tightening globally would terminate every episode. The bad landing is
    a terminal-phase problem, so the tighter bound is applied there only.
    """
    err = (ref_rigid_body_pos - current_rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
    in_tail = progress_buf >= (clip_steps - tail_steps)
    return in_tail & (err > threshold)


def tail_tracking_error_term_factory(threshold: float = 0.25, clip_steps: int = 159,
                                     tail_steps: int = 30) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_tail_tracking_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "progress_buf": EnvContext.progress_buf,
        },
        static_params={
            "threshold": threshold,
            "clip_steps": clip_steps,
            "tail_steps": tail_steps,
        },
    )
