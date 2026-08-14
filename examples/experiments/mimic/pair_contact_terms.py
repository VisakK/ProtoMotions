# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pair-aware contact reward terms (design: ``notes/Contact_balance_reward_design.MD`` §3.5).

Why this exists (measured, ``notes/Physics_insights.md``): the shipped
``compute_contact_match_rew`` scores each body as "touching *anything*", so it is
provably blind to partner identity. Three force-verified failures it cannot see:

* crow ``L_THIGH+TRUNK``: 61.1 % reference dwell, 0.0 % learned — in geometry
  AND measured force. The mirror contact (R_Hip<->Chest) carries 1167 N at 60 %
  duty under the same reward stack, so the target is demonstrably feasible.
* side crow ``L_THIGH+R_THIGH``: 0 % in the reference (deliberately
  loose-excluded by the annotator), learned 743 N peak / 37 % force duty — an
  invented load path.
* crow's bilateral thigh support split by side (L_Hip<->Chest 0 N while
  R_Hip<->Chest carries 1167 N).

Design constraints this implementation honours:

* **No pair identity exists at runtime.** Sensors report per-body net normal
  force only, and per-pair PhysX force is not plumbed at 4096 envs. Pair
  detection is therefore GEOMETRIC: closed-form surface gaps between the typed
  collision primitives (sphere/capsule only — the same kernels the annotation
  pipeline validated, ``data/scripts/contact_geometry.py``), computed from body
  poses every step.
* **Load factor is body-level, not pair-attributed.** ``F_bb = net - ground``
  says a body carries *some* body-body load, not that it presses the intended
  partner (L_Hip is already loaded by L_Shoulder, so for L_Hip<->Chest the
  factor is near-vacuous). The geometric gradient is the real payload; the
  offline per-pair force audit (``compare_learned_contact_configs.py``) remains
  the only pair-load truth.
* **Targets clamp at tangency.** The reference kinematics interpenetrate up to
  6.8 cm on defining pairs; PhysX will not reproduce that. ``d+ = max(d, 0)``
  and full credit already at ``gap <= gap_target`` (1.5 cm), deliberately
  gentler than tangency to bound the pose-fidelity trade
  (``notes/Contact_reward_v1.MD`` §5).
* **Per-clip, side-specific, time-indexed targets.** Masks are rasterized
  offline from ``data/smpl/yoga_contact_configs/`` segments into a
  ``reftargets.pt`` table (built by ``data/scripts/package_pair_targets.py``)
  and indexed by (motion_id, motion_time) — the policy cannot influence its
  gate. Never mirrored: the side-crow stack is entirely on the left arm.

Kernels are callable classes with lazily device-built state (precedent:
``edge_terms.NodeTerminalGoal``), ``@torch._dynamo.disable`` because of the
data-dependent table gather.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.utils.rotations import quat_rotate


# --------------------------------------------------------------------------- #
# Geometry: closed-form segment-segment surface gap.
#
# Inlined from data/scripts/contact_geometry.py (_seg_seg, Ericson 5.1.9) so the
# training path has no dependency on data/scripts. A sphere is represented as a
# degenerate segment (p0 == p1); the degenerate branches below make that exact.
# data/scripts/validate_pair_reward.py cross-checks this against the original.
# --------------------------------------------------------------------------- #
def _seg_seg_closest(p1: Tensor, q1: Tensor, p2: Tensor, q2: Tensor):
    """Closest points between segments p1q1 / p2q2; all [..., 3]."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = torch.where(
        denom > 1e-12,
        ((b * f - c * e) / denom.clamp_min(1e-12)).clamp(0.0, 1.0),
        torch.zeros_like(a),
    )
    t = (b * s + f) / e.clamp_min(1e-12)
    t_cl = t.clamp(0.0, 1.0)
    recompute = (t != t_cl) | (e <= 1e-12)
    s = torch.where(recompute, ((t_cl * b - c) / a.clamp_min(1e-12)).clamp(0.0, 1.0), s)
    return p1 + s.unsqueeze(-1) * d1, p2 + t_cl.unsqueeze(-1) * d2


def _pair_surface_gap(a0, a1, ra, b0, b1, rb):
    """Signed surface gap between two capsules/spheres given world core
    segments [E,3] and radii; negative = penetration."""
    c1, c2 = _seg_seg_closest(a0, a1, b0, b1)
    return (c2 - c1).norm(dim=-1) - ra - rb


class _RefTargets:
    """Device-resident view of a reftargets.pt table (shared loader)."""

    def __init__(self, path: str, device: torch.device):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"pair-reward reftargets file not found: {path} — build it with "
                "data/scripts/package_pair_targets.py"
            )
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.fps = float(d["fps"])
        self.body_names = list(d["body_names"])
        self.num_frames = d["num_frames"].to(device=device, dtype=torch.long)
        self.frame_starts = d["frame_starts"].to(device=device, dtype=torch.long)

        self.pair_names = list(d["pair_names"])
        self.pair_body_a = d["pair_body_a"].to(device=device, dtype=torch.long)
        self.pair_body_b = d["pair_body_b"].to(device=device, dtype=torch.long)
        self.pair_kappa = d["pair_kappa"].to(device=device, dtype=torch.float32)
        self.encourage_mask = d["encourage_mask"].to(device=device, dtype=torch.float32)

        self.forbid_names = list(d["forbid_names"])
        self.forbid_body_a = d["forbid_body_a"].to(device=device, dtype=torch.long)
        self.forbid_body_b = d["forbid_body_b"].to(device=device, dtype=torch.long)
        self.forbid_mask = d["forbid_mask"].to(device=device, dtype=torch.float32)

        self.geom_type = d["geom_type"].to(device=device, dtype=torch.long)
        self.geom_radius = d["geom_radius"].to(device=device, dtype=torch.float32)
        self.geom_p0 = d["geom_p0"].to(device=device, dtype=torch.float32)
        self.geom_p1 = d["geom_p1"].to(device=device, dtype=torch.float32)
        self._ids_checked = False

        assert self.encourage_mask.shape == (
            int(self.num_frames.sum()),
            len(self.pair_names),
        ), "encourage_mask shape does not match num_frames/pair_names"
        assert self.forbid_mask.shape[0] == int(self.num_frames.sum())
        assert float(self.encourage_mask.min()) >= 0.0
        assert float(self.encourage_mask.max()) <= 1.0 + 1e-6

        # Every body used in a body-body pair must carry a sphere/capsule geom.
        bb = torch.cat(
            [
                self.pair_body_a[self.pair_body_b >= 0],
                self.pair_body_b[self.pair_body_b >= 0],
                self.forbid_body_a,
                self.forbid_body_b,
            ]
        )
        assert bool((self.geom_type[bb] > 0).all()), (
            "a pair-reward body has no sphere/capsule geom in the reftargets "
            "table — only closed-form primitives are supported"
        )

        # Ground pairs ('X_HAND:G'): resolved to the hand+wrist body ids.
        name_to_id = {n: i for i, n in enumerate(self.body_names)}
        self.ground_pair_ids = self.pair_body_b < 0  # [P] bool
        self.ground_zone_bodies = {}
        for p, pname in enumerate(self.pair_names):
            if not bool(self.ground_pair_ids[p]):
                continue
            side = pname[0]  # 'L' or 'R'
            self.ground_zone_bodies[p] = torch.tensor(
                [name_to_id[f"{side}_Wrist"], name_to_id[f"{side}_Hand"]],
                device=device,
                dtype=torch.long,
            )

    def frame_lookup(self, motion_ids: Tensor, motion_times: Tensor) -> Tensor:
        """Global row index into the [sum_T, ...] tables for each env.

        Nearest-frame rounding, not floor: motion_times are multiples of the
        policy dt, and ``(k/fps)*fps`` can land at ``k - eps`` in float32 —
        floor would then be off by one at exact frame boundaries. The masks
        are 7-frame-smoothed, so either neighbour is acceptable; nearest is
        exact at boundaries.
        """
        if not self._ids_checked:
            # One-time defense in depth behind the experiment-level contract
            # assert: an out-of-range motion id would otherwise die as an
            # uninformative device-side gather assert.
            mx = int(motion_ids.max())
            assert mx < self.num_frames.shape[0], (
                f"motion_id {mx} out of range for the reftargets table "
                f"({self.num_frames.shape[0]} motions) — the loaded motion "
                "file does not match the pair-reward targets"
            )
            self._ids_checked = True
        local = torch.round(motion_times * self.fps).long()
        local = torch.minimum(
            local.clamp_min(0), self.num_frames[motion_ids] - 1
        )
        return self.frame_starts[motion_ids] + local

    def world_segments(self, body_pos: Tensor, body_rot: Tensor, body_ids: Tensor):
        """World-frame core segment endpoints for the given body ids.

        body_pos [E,24,3], body_rot [E,24,4] xyzw; body_ids [K].
        Returns (p0 [E,K,3], p1 [E,K,3], radius [K]).
        """
        E = body_pos.shape[0]
        K = body_ids.shape[0]
        pos = body_pos[:, body_ids]  # [E,K,3]
        rot = body_rot[:, body_ids]  # [E,K,4]
        flat_rot = rot.reshape(E * K, 4)
        p0 = quat_rotate(
            flat_rot, self.geom_p0[body_ids].unsqueeze(0).expand(E, K, 3).reshape(E * K, 3),
            w_last=True,
        ).reshape(E, K, 3) + pos
        p1 = quat_rotate(
            flat_rot, self.geom_p1[body_ids].unsqueeze(0).expand(E, K, 3).reshape(E * K, 3),
            w_last=True,
        ).reshape(E, K, 3) + pos
        return p0, p1, self.geom_radius[body_ids]


_REFTARGETS_CACHE: dict = {}


def _load_reftargets(path: str, device: torch.device) -> _RefTargets:
    key = (os.path.abspath(path), str(device))
    if key not in _REFTARGETS_CACHE:
        _REFTARGETS_CACHE[key] = _RefTargets(path, device)
    return _REFTARGETS_CACHE[key]


def _body_body_load(
    net_forces: Tensor, ground_forces: Optional[Tensor], load_ref_n: float
) -> Tensor:
    """Per-body body-body load fraction in [0,1]: ||net - ground|| / load_ref_n.

    Body-level only — see the module docstring's honesty note.
    """
    if ground_forces is None:
        raise RuntimeError(
            "rigid_body_ground_forces is not populated — the pair-aware reward "
            "needs the per-body ground force column (IsaacLab per-body contact "
            "sensors filtered against the terrain). Refusing to fall back "
            "silently."
        )
    bb = (net_forces - ground_forces).norm(dim=-1)  # [E,24]
    return (bb / load_ref_n).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Encourage: make the annotated pairs, near-tangent and preferably loaded.
# --------------------------------------------------------------------------- #
class PairEncourageReward:
    """r = sum_p kappa_p tau_p(t) phi(gap+) (0.5 + 0.5 load) / max(sum kappa tau, 0.1).

    phi(d) = exp(-(max(d - gap_target, 0)/gap_sigma)^2): full credit at
    <= gap_target (1.5 cm), so for L_THIGH+TRUNK the ask is ~1-2.5 cm of closure
    from the reference pose, not tangency. Ground pairs (hands) score their
    measured ground normal force directly. Bounded in [0,1] by construction;
    denominator floored at 0.1 so frames with no annotated pair contribute ~0
    rather than NaN (combine_rewards hard-asserts isfinite).
    """

    def __init__(
        self,
        reftargets_pt: str,
        gap_target: float = 0.015,
        gap_sigma: float = 0.03,
        load_ref_n: float = 30.0,
        ground_ref_n: float = 10.0,
    ):
        self.reftargets_pt = reftargets_pt
        self.gap_target = float(gap_target)
        self.gap_sigma = float(gap_sigma)
        self.load_ref_n = float(load_ref_n)
        self.ground_ref_n = float(ground_ref_n)
        self._ready = False

    def _build(self, device):
        self.rt = _load_reftargets(self.reftargets_pt, device)
        rt = self.rt
        self.bb_pair_idx = (~rt.ground_pair_ids).nonzero(as_tuple=True)[0]
        self.g_pair_idx = rt.ground_pair_ids.nonzero(as_tuple=True)[0]
        self.bb_a = rt.pair_body_a[self.bb_pair_idx]
        self.bb_b = rt.pair_body_b[self.bb_pair_idx]
        # Unique bodies involved in body-body pairs, and per-pair positions in it
        all_bodies = torch.cat([self.bb_a, self.bb_b])
        self.uniq_bodies, inverse = torch.unique(all_bodies, return_inverse=True)
        n = self.bb_a.shape[0]
        self.a_in_uniq = inverse[:n]
        self.b_in_uniq = inverse[n:]
        self._ready = True

    @torch._dynamo.disable
    def __call__(
        self,
        body_pos: Tensor,
        body_rot: Tensor,
        net_forces: Tensor,
        ground_forces: Tensor,
        motion_ids: Tensor,
        motion_times: Tensor,
    ) -> Tensor:
        if not self._ready:
            self._build(body_pos.device)
        rt = self.rt
        E = body_pos.shape[0]

        fl = rt.frame_lookup(motion_ids, motion_times)
        enc = rt.encourage_mask[fl]  # [E, P]
        kappa = rt.pair_kappa.unsqueeze(0)  # [1, P]

        r_pairs = torch.zeros_like(enc)  # [E, P]

        # Body-body pairs: geometric gap x load factor.
        if self.bb_pair_idx.numel() > 0:
            p0, p1, radii = rt.world_segments(body_pos, body_rot, self.uniq_bodies)
            gap = _pair_surface_gap(
                p0[:, self.a_in_uniq],
                p1[:, self.a_in_uniq],
                radii[self.a_in_uniq],
                p0[:, self.b_in_uniq],
                p1[:, self.b_in_uniq],
                radii[self.b_in_uniq],
            )  # [E, n_bb]
            gap_pos = gap.clamp_min(0.0)
            phi = torch.exp(
                -((gap_pos - self.gap_target).clamp_min(0.0) / self.gap_sigma) ** 2
            )
            load = _body_body_load(net_forces, ground_forces, self.load_ref_n)
            pair_load = torch.minimum(
                load[:, self.bb_a], load[:, self.bb_b]
            )  # [E, n_bb]
            r_pairs[:, self.bb_pair_idx] = phi * (0.5 + 0.5 * pair_load)

        # Ground pairs (hands): measured ground normal force, direct.
        for p in self.g_pair_idx.tolist():
            zone = rt.ground_zone_bodies[p]
            fz = ground_forces[:, zone, 2].clamp_min(0.0).sum(dim=-1)
            r_pairs[:, p] = (fz / self.ground_ref_n).clamp(0.0, 1.0)

        w = kappa * enc
        return (w * r_pairs).sum(dim=-1) / (w.sum(dim=-1)).clamp_min(0.1)


# --------------------------------------------------------------------------- #
# Forbid: penalize invented load paths, load-gated so incidental unloaded
# proximity is never punished (the annotator loose-excluded thigh-thigh
# precisely because it is chronically proximal in legitimate postures).
# --------------------------------------------------------------------------- #
class PairForbidPenalty:
    """pen = sum_f tau_f(t) psi(gap+) load_f / max(sum_f tau_f, 1); psi ramps
    over the last ``margin`` (2 cm) of approach. Positive output in [0,1];
    give it a negative weight."""

    def __init__(
        self,
        reftargets_pt: str,
        margin: float = 0.02,
        load_ref_n: float = 30.0,
    ):
        self.reftargets_pt = reftargets_pt
        self.margin = float(margin)
        self.load_ref_n = float(load_ref_n)
        self._ready = False

    def _build(self, device):
        self.rt = _load_reftargets(self.reftargets_pt, device)
        rt = self.rt
        all_bodies = torch.cat([rt.forbid_body_a, rt.forbid_body_b])
        self.uniq_bodies, inverse = torch.unique(all_bodies, return_inverse=True)
        n = rt.forbid_body_a.shape[0]
        self.a_in_uniq = inverse[:n]
        self.b_in_uniq = inverse[n:]
        self._ready = True

    @torch._dynamo.disable
    def __call__(
        self,
        body_pos: Tensor,
        body_rot: Tensor,
        net_forces: Tensor,
        ground_forces: Tensor,
        motion_ids: Tensor,
        motion_times: Tensor,
    ) -> Tensor:
        if not self._ready:
            self._build(body_pos.device)
        rt = self.rt
        if rt.forbid_body_a.numel() == 0:
            return torch.zeros(body_pos.shape[0], device=body_pos.device)

        fl = rt.frame_lookup(motion_ids, motion_times)
        forb = rt.forbid_mask[fl]  # [E, F]

        p0, p1, radii = rt.world_segments(body_pos, body_rot, self.uniq_bodies)
        gap = _pair_surface_gap(
            p0[:, self.a_in_uniq],
            p1[:, self.a_in_uniq],
            radii[self.a_in_uniq],
            p0[:, self.b_in_uniq],
            p1[:, self.b_in_uniq],
            radii[self.b_in_uniq],
        )  # [E, F]
        psi = (1.0 - gap.clamp_min(0.0) / self.margin).clamp(0.0, 1.0)
        load = _body_body_load(net_forces, ground_forces, self.load_ref_n)
        pair_load = torch.minimum(
            load[:, self.rt.forbid_body_a], load[:, self.rt.forbid_body_b]
        )
        return (forb * psi * pair_load).sum(dim=-1) / forb.sum(dim=-1).clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Weight-0 diagnostic: the raw surface gap of one named pair, so wandb shows
# e.g. env/raw_r/diag_gap_L_THIGH_TRUNK_mean converging (or not) toward 0.015.
# --------------------------------------------------------------------------- #
class PairGapDiagnostic:
    def __init__(self, reftargets_pt: str, pair_name: str):
        self.reftargets_pt = reftargets_pt
        self.pair_name = pair_name
        self._ready = False

    def _build(self, device):
        rt = _load_reftargets(self.reftargets_pt, device)
        names = rt.pair_names + rt.forbid_names
        a = torch.cat([rt.pair_body_a, rt.forbid_body_a])
        b = torch.cat([rt.pair_body_b, rt.forbid_body_b])
        idx = names.index(self.pair_name)
        self.body_a = int(a[idx])
        self.body_b = int(b[idx])
        assert self.body_b >= 0, "gap diagnostic only supports body-body pairs"
        self.rt = rt
        self._ready = True

    @torch._dynamo.disable
    def __call__(self, body_pos: Tensor, body_rot: Tensor) -> Tensor:
        if not self._ready:
            self._build(body_pos.device)
        ids = torch.tensor(
            [self.body_a, self.body_b], device=body_pos.device, dtype=torch.long
        )
        p0, p1, radii = self.rt.world_segments(body_pos, body_rot, ids)
        return _pair_surface_gap(
            p0[:, 0], p1[:, 0], radii[0], p0[:, 1], p1[:, 1], radii[1]
        )


# --------------------------------------------------------------------------- #
# Weight-0 diagnostic: total ground vertical force over body weight. The S2
# cross-check from the design made permanent: in any supported steady state
# this reads ~1.0; if the filtered force_matrix_w buffer ever silently dies
# (the documented PhysX failure mode), env/raw_r/diag_ground_fz_mean pins to 0
# in wandb instead of the pair reward quietly losing its load gate.
# --------------------------------------------------------------------------- #
def compute_ground_force_fraction(
    ground_forces: Tensor, total_weight_n: float = 74.0 * 9.81
) -> Tensor:
    if ground_forces is None:
        raise RuntimeError("rigid_body_ground_forces missing (see design §2.2)")
    return ground_forces[..., 2].clamp_min(0.0).sum(dim=-1) / total_weight_n


def ground_fz_diag_factory(total_weight_n: float = 74.0 * 9.81) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_ground_force_fraction,
        dynamic_vars={
            "ground_forces": EnvContext.current.rigid_body_ground_forces,
        },
        static_params={"weight": 0.0, "total_weight_n": total_weight_n},
    )


# --------------------------------------------------------------------------- #
# Factories.
# --------------------------------------------------------------------------- #
def _pair_dynamic_vars():
    return {
        "body_pos": EnvContext.current.rigid_body_pos,
        "body_rot": EnvContext.current.rigid_body_rot,
        "net_forces": EnvContext.current.rigid_body_contact_forces,
        "ground_forces": EnvContext.current.rigid_body_ground_forces,
        "motion_ids": EnvContext.mimic.motion_ids,
        "motion_times": EnvContext.mimic.motion_times,
    }


def pair_encourage_rew_factory(
    reftargets_pt: str,
    weight: float = 0.3,
    gap_target: float = 0.015,
    gap_sigma: float = 0.03,
    load_ref_n: float = 30.0,
    ground_ref_n: float = 10.0,
) -> MdpComponent:
    fn = PairEncourageReward(
        reftargets_pt, gap_target, gap_sigma, load_ref_n, ground_ref_n
    )
    return MdpComponent(
        compute_func=fn,
        dynamic_vars=_pair_dynamic_vars(),
        static_params={
            "weight": weight,
            "max_value": abs(weight),
            "zero_during_grace_period": True,
        },
    )


def pair_forbid_rew_factory(
    reftargets_pt: str,
    weight: float = -0.1,
    min_value: float = -0.3,
    margin: float = 0.02,
    load_ref_n: float = 30.0,
) -> MdpComponent:
    fn = PairForbidPenalty(reftargets_pt, margin, load_ref_n)
    return MdpComponent(
        compute_func=fn,
        dynamic_vars=_pair_dynamic_vars(),
        static_params={
            "weight": weight,
            "min_value": min_value,
            "zero_during_grace_period": True,
        },
    )


def pair_gap_diag_factory(reftargets_pt: str, pair_name: str) -> MdpComponent:
    fn = PairGapDiagnostic(reftargets_pt, pair_name)
    return MdpComponent(
        compute_func=fn,
        dynamic_vars={
            "body_pos": EnvContext.current.rigid_body_pos,
            "body_rot": EnvContext.current.rigid_body_rot,
        },
        static_params={"weight": 0.0},
    )
