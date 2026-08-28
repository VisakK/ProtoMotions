# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online tracker of measured contact-configuration events.

The contact graph gives the student its *future* as a sequence of contact
configurations; this tracker gives it its *past* in the same vocabulary. It
watches the measured per-pair contact vector (the same one ``contact_state_obs``
reports) plus the trunk-orientation bin, segments the stream into
maximal constant-configuration runs, and keeps a short per-env ring buffer of
the most recent completed segments alongside the currently-open one.

Why this exists (`notes/Student_v4_methodology.MD` §7.2 and the round-5
discussion): the dense pose history spans a fraction of a second, so a support
change that happened one second ago is invisible to the policy, and over the
corpus knowing just one previous configuration removes 67 % of the
next-configuration entropy (1.63 → 0.54 bits). The tracker is measured-only —
no reference, no graph lookup — so its output is deployable and identical at
training and inference.

Faithfulness to the graph builder, and the two deliberate differences:

* identity is (contact pair set, orientation bin), like a graph node, with the
  builder's sticky-argmax orientation binning (`extract_contact_configs.
  orientation_bins`, margin 0.15) ported to per-env torch;
* segmentation is **causal**: a new identity must persist ``min_dwell_s``
  before it replaces the open segment (the builder's debounce and
  ``absorb_short_runs`` look both directions in time, which an online tracker
  cannot);
* pair activation uses the runtime make-threshold only (no per-pair
  hysteresis); threshold chatter cannot commit events because the identity
  dwell requirement absorbs it.

All state is env-side rollout context: the *observations* built from it are
stored and replayed by the training loop, the tracker itself is never
replayed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from protomotions.utils.rotations import quat_rotate_inverse

# Same score order as extract_contact_configs.ORIENT_BINS:
# upright, inverted, prone, supine, side_l, side_r.
NUM_ORIENT_BINS = 6


def orientation_scores(root_rot_xyzw: Tensor) -> Tensor:
    """Per-bin scores from gravity expressed in the root frame, ``[E, 6]``."""
    down = torch.tensor(
        [0.0, 0.0, -1.0], device=root_rot_xyzw.device, dtype=root_rot_xyzw.dtype
    ).expand(root_rot_xyzw.shape[0], 3)
    g = quat_rotate_inverse(root_rot_xyzw, down, w_last=True)
    return torch.stack(
        [-g[:, 2], g[:, 2], g[:, 0], -g[:, 0], g[:, 1], -g[:, 1]], dim=1
    )


class ContactEventTracker:
    """Per-env ring buffer of debounced contact-configuration segments.

    Token layout produced by :meth:`features` (``num_events`` tokens):
    slot 0 is the currently-open segment, slots 1.. are the most recent
    completed segments, newest first. Per-token features::

        [ contact multi-hot (num_pairs) | orientation one-hot (6) |
          ended_ago | dwell | is_open | valid ]

    with the two times clamped to ``time_clip_s`` and scaled into [0, 1] —
    deterministic scaling, so the block needs no running normalizer and the
    binary channels stay binary.
    """

    def __init__(
        self,
        num_envs: int,
        num_pairs: int,
        num_events: int,
        min_dwell_s: float,
        time_clip_s: float,
        orient_margin: float,
        device: torch.device,
    ):
        if num_events < 1:
            raise ValueError(f"num_events must be >= 1, got {num_events}")
        if min_dwell_s < 0.0:
            raise ValueError(f"min_dwell_s must be >= 0, got {min_dwell_s}")
        self.num_pairs = int(num_pairs)
        self.num_events = int(num_events)
        self.min_dwell_s = float(min_dwell_s)
        self.time_clip_s = float(time_clip_s)
        self.orient_margin = float(orient_margin)
        self.device = device

        E, P = num_envs, self.num_pairs
        K = self.num_events - 1  # completed-segment slots

        self.initialized = torch.zeros(E, dtype=torch.bool, device=device)
        # Episode-local clock; history is deliberately episode-local (it starts
        # empty at reset instead of being backfilled from the reference, which
        # would leak clip identity into a measured channel).
        self.time = torch.zeros(E, device=device)
        self.bin_cur = torch.zeros(E, dtype=torch.long, device=device)

        self.open_contact = torch.zeros(E, P, dtype=torch.bool, device=device)
        self.open_orient = torch.zeros(E, dtype=torch.long, device=device)
        self.open_since = torch.zeros(E, device=device)

        self.cand_active = torch.zeros(E, dtype=torch.bool, device=device)
        self.cand_contact = torch.zeros(E, P, dtype=torch.bool, device=device)
        self.cand_orient = torch.zeros(E, dtype=torch.long, device=device)
        self.cand_since = torch.zeros(E, device=device)

        self.closed_contact = torch.zeros(E, K, P, dtype=torch.bool, device=device)
        self.closed_orient = torch.zeros(E, K, dtype=torch.long, device=device)
        self.closed_start = torch.zeros(E, K, device=device)
        self.closed_end = torch.zeros(E, K, device=device)
        self.closed_valid = torch.zeros(E, K, dtype=torch.bool, device=device)

        # True on rows whose open segment was closed by the most recent
        # update() — "the measured configuration just committed a change".
        # Consumers use it as an event trigger (the FSQ student refreshes its
        # held intent code on it); it is a per-step flag, not accumulated.
        self.last_commit = torch.zeros(E, dtype=torch.bool, device=device)

    @property
    def feature_size(self) -> int:
        return self.num_pairs + NUM_ORIENT_BINS + 4

    # ------------------------------------------------------------------ #
    def reset(self, env_ids: Tensor) -> None:
        """Clear the given envs; they re-initialize from the next observation."""
        if len(env_ids) == 0:
            return
        self.initialized[env_ids] = False
        self.time[env_ids] = 0.0
        self.cand_active[env_ids] = False
        self.closed_valid[env_ids] = False
        self.last_commit[env_ids] = False

    def reset_all(self) -> None:
        """Clear every env — for after an external rollout drove the sim
        (probe scripts, the viz panel): the history those rollouts accumulated
        does not describe the restored episodes."""
        self.reset(torch.arange(self.initialized.shape[0], device=self.device))

    # ------------------------------------------------------------------ #
    def initialize_fresh(self, contact: Tensor, root_rot_xyzw: Tensor) -> None:
        """Open the first segment on rows cleared by :meth:`reset`.

        Branch-free (a few elementwise ops when no row is fresh), so callers
        can run it on every observation build without a host sync. Does not
        advance time and never commits events — that is :meth:`update`'s job,
        which runs exactly once per env step.
        """
        contact = contact > 0.5 if contact.dtype != torch.bool else contact
        fresh = ~self.initialized
        top = orientation_scores(root_rot_xyzw).argmax(dim=-1)
        self.bin_cur = torch.where(fresh, top, self.bin_cur)
        self.open_contact = torch.where(fresh.unsqueeze(-1), contact, self.open_contact)
        self.open_orient = torch.where(fresh, top, self.open_orient)
        self.open_since = torch.where(fresh, self.time, self.open_since)
        self.cand_active = self.cand_active & ~fresh
        self.initialized = self.initialized | fresh

    def update(self, contact: Tensor, root_rot_xyzw: Tensor, dt: float) -> None:
        """Advance one control step with the measured contact vector.

        Args:
            contact: ``[E, num_pairs]`` bool/float measured pair contact.
            root_rot_xyzw: ``[E, 4]`` pelvis rotation, w-last.
            dt: Control step in seconds.
        """
        contact = contact > 0.5 if contact.dtype != torch.bool else contact
        scores = orientation_scores(root_rot_xyzw)
        top = scores.argmax(dim=-1)

        fresh = ~self.initialized
        # Seed the sticky-argmax incumbent on fresh rows, then apply the margin
        # rule everywhere (a fresh row's incumbent == top, so the rule is a
        # no-op there this step).
        self.bin_cur = torch.where(fresh, top, self.bin_cur)
        beats = scores.gather(1, top.unsqueeze(1)).squeeze(1) > (
            scores.gather(1, self.bin_cur.unsqueeze(1)).squeeze(1) + self.orient_margin
        )
        self.bin_cur = torch.where((top != self.bin_cur) & beats, top, self.bin_cur)
        bins = self.bin_cur

        # Fresh rows open their first segment at t=0 and do not advance time
        # this call; initialized rows advance first so `time` is the timestamp
        # of THIS observation.
        self.time = self.time + dt * self.initialized.float()
        self.open_contact = torch.where(
            fresh.unsqueeze(-1), contact, self.open_contact
        )
        self.open_orient = torch.where(fresh, bins, self.open_orient)
        self.open_since = torch.where(fresh, self.time, self.open_since)
        self.initialized = self.initialized | fresh

        same_open = (contact == self.open_contact).all(dim=-1) & (
            bins == self.open_orient
        )
        same_cand = (
            self.cand_active
            & (contact == self.cand_contact).all(dim=-1)
            & (bins == self.cand_orient)
        )

        changed = ~same_open
        commit = changed & same_cand & (
            self.time - self.cand_since >= self.min_dwell_s
        )
        new_cand = changed & ~same_cand
        self.last_commit = commit

        if self.closed_contact.shape[1] > 0:
            c1 = commit.unsqueeze(-1)
            c2 = commit.unsqueeze(-1).unsqueeze(-1)
            # The open segment ended when the committed candidate first
            # appeared, not when it survived the dwell check.
            self.closed_contact = torch.where(
                c2,
                torch.cat(
                    [self.open_contact.unsqueeze(1), self.closed_contact[:, :-1]],
                    dim=1,
                ),
                self.closed_contact,
            )
            self.closed_orient = torch.where(
                c1,
                torch.cat(
                    [self.open_orient.unsqueeze(1), self.closed_orient[:, :-1]], dim=1
                ),
                self.closed_orient,
            )
            self.closed_start = torch.where(
                c1,
                torch.cat(
                    [self.open_since.unsqueeze(1), self.closed_start[:, :-1]], dim=1
                ),
                self.closed_start,
            )
            self.closed_end = torch.where(
                c1,
                torch.cat(
                    [self.cand_since.unsqueeze(1), self.closed_end[:, :-1]], dim=1
                ),
                self.closed_end,
            )
            self.closed_valid = torch.where(
                c1,
                torch.cat(
                    [
                        torch.ones_like(commit).unsqueeze(1),
                        self.closed_valid[:, :-1],
                    ],
                    dim=1,
                ),
                self.closed_valid,
            )

        self.open_contact = torch.where(
            commit.unsqueeze(-1), self.cand_contact, self.open_contact
        )
        self.open_orient = torch.where(commit, self.cand_orient, self.open_orient)
        self.open_since = torch.where(commit, self.cand_since, self.open_since)

        self.cand_contact = torch.where(
            new_cand.unsqueeze(-1), contact, self.cand_contact
        )
        self.cand_orient = torch.where(new_cand, bins, self.cand_orient)
        self.cand_since = torch.where(new_cand, self.time, self.cand_since)
        # A candidate survives only while the stream keeps disagreeing with the
        # open segment; matching the open segment again clears it.
        self.cand_active = (self.cand_active & changed & ~commit) | new_cand

    # ------------------------------------------------------------------ #
    def features(self) -> tuple[Tensor, Tensor]:
        """``(features [E, num_events, F], valid [E, num_events])``.

        Invalid slots are all-zero, like a hidden goal slot, so downstream
        consumers can concatenate without re-masking.
        """
        E = self.initialized.shape[0]
        P, O, M = self.num_pairs, NUM_ORIENT_BINS, self.num_events
        out = torch.zeros(E, M, self.feature_size, device=self.device)
        valid = torch.zeros(E, M, dtype=torch.bool, device=self.device)

        clip = self.time_clip_s
        valid[:, 0] = self.initialized
        out[:, 0, :P] = self.open_contact.float()
        out[:, 0, P : P + O] = torch.nn.functional.one_hot(self.open_orient, O).float()
        out[:, 0, P + O] = 0.0  # ended_ago: still open
        out[:, 0, P + O + 1] = ((self.time - self.open_since).clamp(0.0, clip)) / clip
        out[:, 0, P + O + 2] = 1.0  # is_open
        out[:, 0, P + O + 3] = 1.0  # valid flag (zeroed below if not)

        if M > 1:
            valid[:, 1:] = self.closed_valid & self.initialized.unsqueeze(-1)
            out[:, 1:, :P] = self.closed_contact.float()
            out[:, 1:, P : P + O] = torch.nn.functional.one_hot(
                self.closed_orient, O
            ).float()
            now = self.time.unsqueeze(-1)
            out[:, 1:, P + O] = ((now - self.closed_end).clamp(0.0, clip)) / clip
            out[:, 1:, P + O + 1] = (
                (self.closed_end - self.closed_start).clamp(0.0, clip)
            ) / clip
            out[:, 1:, P + O + 2] = 0.0
            out[:, 1:, P + O + 3] = 1.0

        out = out * valid.unsqueeze(-1).float()
        return out, valid


__all__ = ["ContactEventTracker", "orientation_scores", "NUM_ORIENT_BINS"]
