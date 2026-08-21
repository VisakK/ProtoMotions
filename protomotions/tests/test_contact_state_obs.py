# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for binary body-body contact observation over the graph's vocabulary."""

from __future__ import annotations

import math

import pytest
import torch

from protomotions.components.contact_graph import ContactGraph
from protomotions.envs.obs.contact_state import compute_contact_state_obs

# A miniature but structurally faithful robot: two zones that pool more than one
# body (feet, trunk) and two that do not, so pooling is actually exercised.
BODY_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
]
ZONE_ORDER = ["L_FOOT", "R_FOOT", "L_THIGH", "L_UPPER_ARM", "L_HAND"]
ZONE_BODIES = {
    "L_FOOT": ["L_Ankle", "L_Toe"],
    "R_FOOT": ["R_Ankle", "R_Toe"],
    "L_THIGH": ["L_Hip"],
    "L_UPPER_ARM": ["L_Shoulder"],
    "L_HAND": ["L_Wrist", "L_Hand"],
}
PAIR_NAMES = [
    "L_FOOT:G",
    "R_FOOT:G",
    "L_HAND:G",
    "L_FOOT+R_FOOT",
    "L_THIGH+L_UPPER_ARM",
]
GROUND_N = 20.0
BODY_N = 15.0


def _graph() -> ContactGraph:
    num_pairs = len(PAIR_NAMES)
    payload = {
        "motion_names": ["clip"],
        "pair_names": list(PAIR_NAMES),
        "orientation_names": ["upright"],
        "node_keys": ["x@upright"],
        "node_contact": torch.zeros(1, num_pairs),
        "node_orient": torch.zeros(1, dtype=torch.long),
        "seg_node": torch.zeros(1, 1, dtype=torch.long),
        "seg_start": torch.zeros(1, 1),
        "seg_end": torch.ones(1, 1),
        "seg_hold": torch.full((1, 1), 0.5),
        "seg_count": torch.ones(1, dtype=torch.long),
        "zone_order": list(ZONE_ORDER),
        "zone_bodies": dict(ZONE_BODIES),
        "min_lead_s": 0.2,
    }
    return ContactGraph(payload)


def _maps(with_pairs: bool = True):
    return _graph().contact_scatter_maps(
        body_names=BODY_NAMES,
        ground_threshold_n=GROUND_N,
        body_threshold_n=BODY_N,
        pair_body_names=list(BODY_NAMES) if with_pairs else None,
    )


def _run(maps, ground, pairs=None):
    kwargs = {k: v for k, v in maps.items() if k not in
              ("pair_slot_names", "num_body_body_slots")}
    return compute_contact_state_obs(ground_forces=ground, pair_forces=pairs, **kwargs)


def _zero_ground(num_envs=1):
    return torch.zeros(num_envs, len(BODY_NAMES), 3)


def _zero_pairs(num_envs=1):
    return torch.zeros(num_envs, len(BODY_NAMES), len(BODY_NAMES), 3)


def test_slot_order_matches_the_graphs_pair_vocabulary():
    maps = _maps()
    assert maps["num_pairs"] == len(PAIR_NAMES)
    assert maps["pair_slot_names"] == PAIR_NAMES
    assert maps["num_body_body_slots"] == 2
    # Ground slots take the ground threshold, body-body slots the body one.
    assert torch.allclose(
        maps["thresholds"], torch.tensor([GROUND_N] * 3 + [BODY_N] * 2)
    )


def test_ground_zones_pool_their_member_bodies():
    maps = _maps()
    ground = _zero_ground()
    # Neither ankle nor toe alone clears 20 N, but the L_FOOT zone does.
    ground[0, BODY_NAMES.index("L_Ankle"), 2] = 12.0
    ground[0, BODY_NAMES.index("L_Toe"), 2] = 12.0
    out = _run(maps, ground, _zero_pairs())
    assert out[0, PAIR_NAMES.index("L_FOOT:G")] == 1.0
    assert out[0, PAIR_NAMES.index("R_FOOT:G")] == 0.0
    assert out[0, PAIR_NAMES.index("L_HAND:G")] == 0.0


def test_ground_contact_below_threshold_does_not_fire():
    maps = _maps()
    ground = _zero_ground()
    ground[0, BODY_NAMES.index("L_Ankle"), 2] = 19.0
    out = _run(maps, ground, _zero_pairs())
    assert out[0, PAIR_NAMES.index("L_FOOT:G")] == 0.0


def test_body_body_pair_fires_from_the_pair_force_matrix():
    maps = _maps()
    pairs = _zero_pairs()
    thigh = BODY_NAMES.index("L_Hip")
    upper_arm = BODY_NAMES.index("L_Shoulder")
    pairs[0, thigh, upper_arm, 1] = 40.0
    out = _run(maps, _zero_ground(), pairs)
    assert out[0, PAIR_NAMES.index("L_THIGH+L_UPPER_ARM")] == 1.0
    # and nothing else moved
    assert float(out.sum()) == 1.0


def test_one_populated_direction_is_enough():
    """PhysX fills the sensing body's column; which body that is varies."""
    maps = _maps()
    thigh = BODY_NAMES.index("L_Hip")
    upper_arm = BODY_NAMES.index("L_Shoulder")
    slot = PAIR_NAMES.index("L_THIGH+L_UPPER_ARM")

    forward = _zero_pairs()
    forward[0, thigh, upper_arm, 1] = 30.0
    backward = _zero_pairs()
    backward[0, upper_arm, thigh, 1] = 30.0

    assert _run(maps, _zero_ground(), forward)[0, slot] == 1.0
    assert _run(maps, _zero_ground(), backward)[0, slot] == 1.0


def test_both_directions_are_maxed_not_summed():
    """Newton's third law fills both columns equally; summing would double it."""
    maps = _maps()
    thigh = BODY_NAMES.index("L_Hip")
    upper_arm = BODY_NAMES.index("L_Shoulder")
    slot = PAIR_NAMES.index("L_THIGH+L_UPPER_ARM")

    pairs = _zero_pairs()
    # 10 N each way. Summed that is 20 N and would clear the 15 N threshold;
    # maxed it is 10 N and must not.
    pairs[0, thigh, upper_arm, 1] = 10.0
    pairs[0, upper_arm, thigh, 1] = -10.0
    assert _run(maps, _zero_ground(), pairs)[0, slot] == 0.0

    pairs[0, thigh, upper_arm, 1] = 16.0
    pairs[0, upper_arm, thigh, 1] = -16.0
    assert _run(maps, _zero_ground(), pairs)[0, slot] == 1.0


def test_pair_pooling_sums_within_a_zone_pair():
    maps = _maps()
    slot = PAIR_NAMES.index("L_FOOT+R_FOOT")
    pairs = _zero_pairs()
    # Ankle-on-ankle and toe-on-toe, neither alone above 15 N.
    pairs[0, BODY_NAMES.index("L_Ankle"), BODY_NAMES.index("R_Ankle"), 0] = 9.0
    pairs[0, BODY_NAMES.index("L_Toe"), BODY_NAMES.index("R_Toe"), 0] = 9.0
    assert _run(maps, _zero_ground(), pairs)[0, slot] == 1.0


def test_ground_only_configuration_leaves_body_body_slots_unreachable():
    maps = _maps(with_pairs=False)
    assert maps["num_body_body_slots"] == 0
    assert maps["pair_slot"].numel() == 0
    thresholds = maps["thresholds"]
    # +inf, not a large number: an unreachable slot must be excluded from any
    # IoU denominator rather than merely hard to trigger.
    assert math.isinf(float(thresholds[PAIR_NAMES.index("L_FOOT+R_FOOT")]))
    out = _run(maps, _zero_ground(), None)
    assert float(out.sum()) == 0.0


def test_missing_pair_forces_with_configured_pairs_raises():
    maps = _maps()
    with pytest.raises(ValueError, match="contact_pair_bodies"):
        _run(maps, _zero_ground(), None)


def test_missing_ground_forces_raises():
    maps = _maps()
    with pytest.raises(ValueError, match="rigid_body_ground_forces"):
        _run(maps, None, _zero_pairs())


def test_batched_environments_are_independent():
    maps = _maps()
    ground = _zero_ground(num_envs=3)
    pairs = _zero_pairs(num_envs=3)
    ground[1, BODY_NAMES.index("L_Toe"), 2] = 50.0
    pairs[2, BODY_NAMES.index("L_Hip"), BODY_NAMES.index("L_Shoulder"), 0] = 50.0
    out = _run(maps, ground, pairs)
    assert float(out[0].sum()) == 0.0
    assert out[1, PAIR_NAMES.index("L_FOOT:G")] == 1.0
    assert float(out[1].sum()) == 1.0
    assert out[2, PAIR_NAMES.index("L_THIGH+L_UPPER_ARM")] == 1.0
    assert float(out[2].sum()) == 1.0


def test_maps_against_the_real_graph_cover_the_whole_vocabulary():
    """The shipped 44-clip graph, against the real SMPL body list."""
    from pathlib import Path

    graph_file = Path("data/smpl/yoga_contact_graph_student44/contact_graph.pt")
    if not graph_file.is_file():
        pytest.skip("student44 graph not built")
    graph = ContactGraph.from_file(str(graph_file))
    zone_order, zones = graph.zone_definition()
    bodies = sorted({b for zone in zone_order for b in zones[zone]})
    maps = graph.contact_scatter_maps(
        body_names=bodies,
        ground_threshold_n=GROUND_N,
        body_threshold_n=BODY_N,
        pair_body_names=bodies,
    )
    # 104 = 15 ground + 89 body-body, and every one of them must be reachable
    # once the robot senses its own bodies -- a silently unreachable slot is a
    # goal the student can be given and can never satisfy.
    assert maps["num_pairs"] == 104
    assert maps["num_body_body_slots"] == 89
    assert torch.all(torch.isfinite(maps["thresholds"]))
