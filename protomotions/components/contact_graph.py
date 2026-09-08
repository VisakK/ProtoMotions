# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime view of a contact-configuration graph.

A *contact configuration* is the set of contact pairs -- ``ZONE:G`` (ground) and
``ZONE_A+ZONE_B`` (body-body) -- that carry load, plus a coarse trunk-orientation
bin.  ``data/scripts/build_contact_graph_from_rollouts.py`` extracts one segment
per maximal run of a constant configuration from *simulated expert rollouts* (the
force PhysX reports, not geometric proximity) and records, for every segment, the
clip it came from and the time at which the pose is being *held*.

This class is the training-time half: it turns those per-clip segment lists into
padded tensors and answers one question on the GPU, every step, for every
environment --

    "given that env is playing clip ``m`` at time ``t``, what are the next ``K``
    contact configurations it will reach, and when?"

That answer is the student's goal.  The lookup is a pure function of
``(motion_id, motion_time)``, so it needs no reset bookkeeping and cannot drift
out of sync with the motion manager.

The tables are keyed to a packaged ``MotionLib``'s motion order.  Loading against
a different library is a silent mislabelling -- every goal would come from the
wrong clip -- so :meth:`validate_against_motion_lib` checks the names and is
called by the control component at construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor


class ContactGraph:
    """Padded per-motion contact-configuration segment tables.

    Attributes:
        motion_names: Clip stem per motion id, in the packaged library's order.
        pair_names: Contact-pair names; index into the multi-hot node vectors.
        orientation_names: Trunk-orientation bin names.
        node_keys: Canonical configuration string per node id.
        node_contact: ``[num_nodes, num_pairs]`` multi-hot contact vector. What a
            *manual* goal gathers (probe plans, ``query_contact_goal.py``, the
            viz panel), and the fallback for scheduled goals on graphs written
            before ``seg_contact`` existed.
        node_orient: ``[num_nodes]`` orientation-bin index per node.
        seg_node: ``[num_motions, max_segments]`` node id, ``-1`` where padded.
        seg_contact: ``[num_motions, max_segments, num_pairs]`` multi-hot contact
            target of each *scheduled* goal, or ``None`` on older graphs. Node
            identity can be coarser than the contact set a segment actually held
            (``--body-pair-identity none`` makes a node exactly ``(ground set,
            orientation)``), so the goal is served per segment: the coarsening
            buys edge consolidation without giving back the body-body goal
            channel round 2 added.
        seg_start / seg_end / seg_hold: ``[num_motions, max_segments]`` clip times,
            ``+inf`` where padded so ``searchsorted`` runs off the end cleanly.
        seg_count: ``[num_motions]`` number of real segments per motion.
    """

    def __init__(self, payload: dict, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.motion_names: List[str] = list(payload["motion_names"])
        self.pair_names: List[str] = list(payload["pair_names"])
        self.orientation_names: List[str] = list(payload["orientation_names"])
        self.node_keys: List[str] = list(payload["node_keys"])
        self.min_lead_s: float = float(payload.get("min_lead_s", 0.2))
        # Zone -> body names, as the graph was built. Absent on graphs written
        # before this was stored; consumers fall back to the extractor's own
        # definition in that case.
        self.zone_order: Optional[List[str]] = (
            list(payload["zone_order"]) if payload.get("zone_order") else None
        )
        self.zone_bodies: Optional[dict] = payload.get("zone_bodies")

        self.node_contact: Tensor = payload["node_contact"].to(self.device).float()
        self.node_orient: Tensor = payload["node_orient"].to(self.device).long()
        self.seg_node: Tensor = payload["seg_node"].to(self.device).long()
        seg_contact = payload.get("seg_contact")
        self.seg_contact: Optional[Tensor] = (
            seg_contact.to(self.device).float() if seg_contact is not None else None
        )
        self.seg_start: Tensor = payload["seg_start"].to(self.device).float()
        self.seg_end: Tensor = payload["seg_end"].to(self.device).float()
        self.seg_hold: Tensor = payload["seg_hold"].to(self.device).float()
        self.seg_count: Tensor = payload["seg_count"].to(self.device).long()

        if self.seg_contact is not None and self.seg_contact.shape[:2] != self.seg_node.shape:
            raise ValueError(
                f"seg_contact is {tuple(self.seg_contact.shape)} but seg_node is "
                f"{tuple(self.seg_node.shape)}: the graph tables disagree on the "
                "segment layout"
            )
        if self.seg_hold.shape[0] != len(self.motion_names):
            raise ValueError(
                f"contact graph has {self.seg_hold.shape[0]} motion rows but "
                f"{len(self.motion_names)} motion names"
            )
        # A node id of -1 marks padding; clamp so an indexed gather is always in
        # range and pair it with the validity mask the lookup returns.
        self.safe_seg_node = self.seg_node.clamp(min=0)
        # Row-wise sorted holds are what makes the batched searchsorted valid.
        # Padding is +inf, which sorts after every real hold and compares False
        # against itself, so the whole row can be checked as-is.
        if bool((self.seg_hold[:, 1:] < self.seg_hold[:, :-1]).any()):
            raise ValueError("contact graph segment hold times are not sorted per motion")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str | Path, device: torch.device | str = "cpu") -> "ContactGraph":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(payload, device=device)

    @property
    def num_pairs(self) -> int:
        return self.node_contact.shape[1]

    @property
    def num_orientations(self) -> int:
        return len(self.orientation_names)

    @property
    def num_nodes(self) -> int:
        return self.node_contact.shape[0]

    @property
    def goal_feature_size(self) -> int:
        """Per-goal-step feature width: contact multi-hot + orientation + validity."""
        return self.num_pairs + self.num_orientations + 1

    # ------------------------------------------------------------------ #
    def validate_against_motion_lib(self, motion_files: List[str]) -> None:
        """Refuse to run against a library the tables were not built for."""
        names = [Path(f).stem for f in motion_files]
        if names != self.motion_names:
            overlap = len(set(names) & set(self.motion_names))
            raise ValueError(
                f"contact graph was built for a different motion library: "
                f"{len(names)} clips in the library, {len(self.motion_names)} in the "
                f"graph, {overlap} names in common. Rebuild the graph with "
                f"--motion-file pointing at this library."
            )

    def coverage(self) -> Tuple[int, int]:
        """``(motions with at least one segment, total motions)``."""
        return int((self.seg_count > 0).sum()), len(self.motion_names)

    # ------------------------------------------------------------------ #
    def next_goal_indices(
        self,
        motion_ids: Tensor,
        motion_times: Tensor,
        num_steps: int,
        min_lead_s: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Segment indices of the next ``num_steps`` holds, and their validity.

        Args:
            motion_ids: ``[E]`` clip id per environment.
            motion_times: ``[E]`` current clip time per environment.
            num_steps: How many upcoming goals to return.
            min_lead_s: A hold closer than this is already being passed through
                and is skipped, so the nearest goal is always something the
                policy still has time to act on.

        Returns:
            ``(indices [E, num_steps], valid [E, num_steps])``.  Indices are
            clamped into range for motions that run out of segments; ``valid``
            is False there and for motions with no segments at all, so callers
            can zero the goal rather than repeat a stale one.
        """
        lead = self.min_lead_s if min_lead_s is None else min_lead_s
        holds = self.seg_hold.index_select(0, motion_ids)              # [E, S]
        counts = self.seg_count.index_select(0, motion_ids)            # [E]
        threshold = (motion_times + lead).unsqueeze(-1)                # [E, 1]
        first = torch.searchsorted(holds.contiguous(), threshold.contiguous())
        offsets = torch.arange(num_steps, device=motion_ids.device).unsqueeze(0)
        raw = first + offsets                                          # [E, K]
        valid = raw < counts.unsqueeze(-1)
        # Past the last hold there is nothing left to aim at; hold the final
        # segment so the pose query stays well defined and mark it invalid.
        last = (counts - 1).clamp(min=0).unsqueeze(-1).expand_as(raw)
        indices = torch.minimum(raw, last).clamp(min=0)
        valid = valid & (counts.unsqueeze(-1) > 0)
        return indices, valid

    def gather(self, motion_ids: Tensor, indices: Tensor) -> dict:
        """Segment data for ``[E, K]`` indices into each env's own motion row."""
        rows = motion_ids.unsqueeze(-1).expand_as(indices)
        node = self.safe_seg_node[rows, indices]
        # Per segment where the graph provides it (see `seg_contact`), per node
        # otherwise. The two agree exactly on every graph built with a
        # body-body identity rule of "all" or "load_path".
        contact = (
            self.seg_contact[rows, indices]
            if self.seg_contact is not None
            else self.node_contact[node]
        )
        return {
            "node": node,
            "t_start": self.seg_start[rows, indices],
            "t_end": self.seg_end[rows, indices],
            "t_hold": self.seg_hold[rows, indices],
            "contact": contact,
            "orient": self.node_orient[node],
        }

    def describe_node(self, node_id: int) -> str:
        if node_id < 0 or node_id >= self.num_nodes:
            return "<none>"
        return self.node_keys[node_id]

    # ------------------------------------------------------------------ #
    # Mapping simulated forces into this graph's pair vocabulary
    # ------------------------------------------------------------------ #
    def zone_definition(self) -> Tuple[List[str], dict]:
        """``(zone_order, zone -> body names)`` this graph was built with.

        Graphs written before the zone definition was stored alongside the
        tables fall back to the extractor's own definition, which is what they
        were built with.
        """
        if self.zone_order and self.zone_bodies:
            return list(self.zone_order), dict(self.zone_bodies)

        import sys

        scripts = str(Path(__file__).resolve().parents[2] / "data" / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from extract_contact_configs import ZONE_ORDER, ZONES  # noqa: E402

        return list(ZONE_ORDER), dict(ZONES)

    def contact_scatter_maps(
        self,
        body_names: List[str],
        ground_threshold_n: float,
        body_threshold_n: float,
        pair_body_names: Optional[List[str]] = None,
        device: torch.device | str | None = None,
    ) -> dict:
        """Index tensors pooling per-body forces into this graph's pair slots.

        The output feeds
        :func:`protomotions.envs.obs.contact_state.compute_contact_state_obs`,
        and the point of building it here rather than in the experiment file is
        that the *goal* vector and the *state* vector then share one vocabulary
        by construction: slot ``k`` means the same pair on both sides.

        Args:
            body_names: Body order of the force tensors' axis 1 (common order).
            ground_threshold_n: Newtons above which a pooled ground zone counts.
            body_threshold_n: Newtons above which a pooled zone pair counts.
            pair_body_names: Body order of the pair tensor's axis 2, i.e.
                ``RobotConfig.contact_pair_bodies``. ``None`` means body-body
                pairs are not sensed and their slots are left empty.
            device: Device for the returned tensors.

        Returns:
            ``dict`` of the kernel's static params, plus ``pair_slot_names`` for
            diagnostics and ``num_body_body_slots`` so a caller can assert the
            body-body half is actually wired.
        """
        device = torch.device(device or self.device)
        zone_order, zones = self.zone_definition()
        body_index = {name: i for i, name in enumerate(body_names)}
        pair_index = (
            {name: i for i, name in enumerate(pair_body_names)}
            if pair_body_names
            else {}
        )
        known_zones = set(zone_order)

        ground_slot, ground_body = [], []
        pair_slot, pair_a, pair_b = [], [], []
        thresholds = [float("inf")] * self.num_pairs
        body_body_slots = 0

        for slot, pair_name in enumerate(self.pair_names):
            if pair_name.endswith(":G"):
                zone = pair_name[:-2]
                if zone not in known_zones:
                    continue
                members = [b for b in zones[zone] if b in body_index]
                if not members:
                    continue
                thresholds[slot] = ground_threshold_n
                for body in members:
                    ground_slot.append(slot)
                    ground_body.append(body_index[body])
                continue

            if "+" not in pair_name or not pair_index:
                continue
            zone_a, zone_b = pair_name.split("+", 1)
            if zone_a not in known_zones or zone_b not in known_zones:
                continue
            members_a = [b for b in zones[zone_a] if b in body_index]
            members_b = [b for b in zones[zone_b] if b in body_index]
            # Both directions are recorded: PhysX fills the column of whichever
            # body's sensor saw the contact, and the kernel maxes over the two
            # rather than summing, so a slot addresses [slot * 2 + direction].
            usable_ab = [
                (body_index[a], pair_index[b])
                for a in members_a
                for b in members_b
                if b in pair_index
            ]
            usable_ba = [
                (body_index[b], pair_index[a])
                for b in members_b
                for a in members_a
                if a in pair_index
            ]
            if not usable_ab and not usable_ba:
                continue
            thresholds[slot] = body_threshold_n
            body_body_slots += 1
            for row, column in usable_ab:
                pair_slot.append(slot * 2)
                pair_a.append(row)
                pair_b.append(column)
            for row, column in usable_ba:
                pair_slot.append(slot * 2 + 1)
                pair_a.append(row)
                pair_b.append(column)

        def as_long(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        return {
            "ground_slot": as_long(ground_slot),
            "ground_body": as_long(ground_body),
            "pair_slot": as_long(pair_slot),
            "pair_body_a": as_long(pair_a),
            "pair_body_b": as_long(pair_b),
            # inf on any slot this robot cannot realise, so it can never fire.
            "thresholds": torch.tensor(thresholds, dtype=torch.float, device=device),
            "num_pairs": self.num_pairs,
            "pair_slot_names": list(self.pair_names),
            "num_body_body_slots": body_body_slots,
        }

    def node_id_for_key(self, key: str) -> int:
        """Node id for a canonical configuration string.

        e.g. ``"L_HAND:G|R_HAND:G@inverted"``.

        Node *ids* are an artifact of build order: rebuilding the graph on a
        different corpus renumbers every node, and low ids survive the renumbering
        while meaning something else entirely. A saved plan or script that names
        its goals by id therefore keeps running after a rebuild and silently asks
        for the wrong pose. The key does not move, so anything persisted to disk
        should name nodes this way.

        Raises:
            KeyError: with the closest keys by pair overlap, since a key that is
                one pair out is the overwhelmingly likely mistake.
        """
        try:
            return self.node_keys.index(key)
        except ValueError:
            pass

        def parse(text: str) -> tuple:
            body, _, orient = text.rpartition("@")
            pairs = frozenset(p for p in body.split("|") if p and p != "NONE")
            return pairs, orient

        want_pairs, want_orient = parse(key)
        scored = []
        for index, candidate in enumerate(self.node_keys):
            pairs, orient = parse(candidate)
            scored.append(
                (
                    len(want_pairs ^ pairs) + (0 if orient == want_orient else 1),
                    index,
                    candidate,
                )
            )
        scored.sort()
        near = "\n  ".join(f"[{i}] {k}" for _d, i, k in scored[:5])
        raise KeyError(
            f"no node with configuration '{key}' in this graph "
            f"({self.num_nodes} nodes). Closest:\n  {near}"
        )


__all__ = ["ContactGraph"]
