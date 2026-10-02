# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contact-target sidecar of a release (TODO C1; BUILD_PLAN Step 9, BodyFix Step 5).

The graph's goal vector is binary: a pair is commanded or it is not, and "not" is read by the unwanted-
support term as "keep it off the floor". A curated release knows more, per hold: each contact's role
(``required_support``, ``required_touch``, ``forbidden_support``, ``allowed``, ``incidental``, ...), which
configured contacts the gate masked, which body-body contacts the reviewer called critical (B6) and which
statics restored, the gate's flags, the statics loads, which ground zones the human is *known* to keep
free, and on which frames the reference actually closes each configured body-body pair.
``reference_curation.contact_targets_v2`` compiles that into ``contact_targets.pt`` in the graph's own
layout (``[motion, segment, pair]``, plus a ``[motion, frame, pair]`` table), and this class loads it,
validated against the graph and the library like ``PhysicsTables``: motion order, pair vocabulary, zone
order, segment layout, the per-segment hold ids, frame counts, fps, the commanded set (= the graph's
``seg_contact``) and the graph's sha256.

Nothing here is trained. ``ContactGraphControl`` reads it for weight-0 diagnostics and to restrict the
unwanted-support term to zones the human is known to keep free (the design's "explicit known-negative
mask before an uncertain release is used for training").
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

ROLE_NAMES = ("none", "required_support", "required_touch", "forbidden_support", "allowed", "incidental",
              "unresolved", "unspecified")


class ContactTargets:
    """``contact_targets.pt`` on the device, checked against the graph it was compiled for."""

    def __init__(self, path: str, graph, motion_names: List[str], device,
                 motion_num_frames: Optional[Tensor] = None, fps: Optional[float] = None,
                 graph_sha256: Optional[str] = None, plant_mjcf: Optional[str] = None):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("kind") != "contact_targets" or int(payload.get("version", 0)) != 1:
            raise ValueError(f"{path} is not a v1 contact-target sidecar")
        if plant_mjcf is not None:
            from protomotions.utils import plant_identity

            plant_identity.require(payload.get(plant_identity.KEY), plant_mjcf, f"contact targets {path}")
        if list(payload["role_names"]) != list(ROLE_NAMES):
            raise ValueError(f"{path}: role vocabulary {payload['role_names']} is not {ROLE_NAMES}")
        if list(payload["motion_names"]) != list(motion_names) or list(payload["motion_names"]) != list(
            graph.motion_names
        ):
            raise ValueError(f"{path} was compiled for another motion library")
        if list(payload["pair_names"]) != list(graph.pair_names):
            raise ValueError(f"{path} uses another pair vocabulary than the graph")
        zone_order, _ = graph.zone_definition()
        if list(payload["zone_order"]) != list(zone_order):
            raise ValueError(f"{path} zone order {payload['zone_order']} != graph {zone_order}")
        if tuple(payload["seg_role"].shape[:2]) != tuple(graph.seg_node.shape):
            raise ValueError(f"{path} was compiled for another segment layout")
        if graph.seg_hold_index is not None:
            if list(payload["hold_ids"]) != list(graph.hold_ids) or not torch.equal(
                payload["seg_hold_index"].long(), graph.seg_hold_index.cpu()
            ):
                raise ValueError(f"{path}: its hold ids per segment differ from the graph's")
        if graph.seg_contact is not None:
            slots = torch.arange(graph.seg_node.shape[1]).unsqueeze(0)
            live = (slots < graph.seg_count.cpu().unsqueeze(-1)).unsqueeze(-1)
            if not torch.equal(payload["seg_commanded"] & live, (graph.seg_contact.cpu() > 0.5) & live):
                raise ValueError(f"{path}: its commanded contacts differ from the graph's seg_contact")
        if graph_sha256 is not None and payload.get("graph_sha256") != graph_sha256:
            raise ValueError(f"{path} was compiled for graph sha256 {str(payload.get('graph_sha256'))[:12]}, "
                             f"not {graph_sha256[:12]}")
        if fps is not None and round(float(fps)) != int(payload["fps"]):
            raise ValueError(f"{path} was compiled at {payload['fps']} fps, the library runs at {fps}")
        if motion_num_frames is not None and not torch.equal(
            payload["frame_len"].long(), torch.as_tensor(motion_num_frames).long().cpu()
        ):
            raise ValueError(f"{path}: its frame tables' lengths differ from the library's frame counts")

        self.path = str(path)
        self.release_id = payload.get("release_id")
        self.fps = float(payload["fps"])
        self.zone_order = list(payload["zone_order"])
        self.pair_names = list(payload["pair_names"])
        self.flag_names = list(payload["flag_names"])
        role = payload["seg_role"].long()
        self.required_support = (role == ROLE_NAMES.index("required_support")).to(device)
        self.forbidden_support = (role == ROLE_NAMES.index("forbidden_support")).to(device)
        self.masked = payload["seg_masked"].bool().to(device)
        self.configured = payload["seg_configured"].bool().to(device)
        self.critical = payload["seg_critical"].bool().to(device)
        self.ground_free = payload["seg_ground_free"].bool().to(device)          # [M, S, Z]
        self.body_pair = torch.tensor(["+" in p for p in self.pair_names], dtype=torch.bool, device=device)
        self.frame_pair_slots = payload["frame_pair_slots"].long().to(device)     # [Q]
        self.frame_pair_ok = payload["frame_pair_ok"].bool().to(device)          # [M, T, Q]
        self.frame_len = payload["frame_len"].long().to(device)                  # [M]

    def frame_pairs(self, motion_ids: Tensor, motion_times: Tensor) -> Tensor:
        """``[E, P]`` True where the reference (and the human) close a configured body-body pair at the
        env's clip frame -- the per-frame eligibility of a pair target."""
        frame = torch.round(motion_times * self.fps).long()
        frame = torch.minimum(frame.clamp(min=0), self.frame_len[motion_ids] - 1)
        ok = self.frame_pair_ok[motion_ids, frame]                                # [E, Q]
        out = torch.zeros(motion_ids.shape[0], len(self.pair_names), dtype=torch.bool, device=ok.device)
        if self.frame_pair_slots.numel():
            out[:, self.frame_pair_slots] = ok
        return out
