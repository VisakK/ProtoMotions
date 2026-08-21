# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the measured-pressure reward terms.

The last test is the one that matters: it replays a crow hold frame taken from
the measured MOYO reference and checks both terms give the answer that
``notes/Pressure_supervision_design.MD`` 4 measured offline.
"""

import pytest
import torch

from examples.experiments.mimic.pressure_terms import (
    COMMON_BODY_NAMES,
    SUPPORT_ZONES,
    VALID_ON_MAT,
    VALID_PER_BODY,
    build_zone_matrix,
    compute_pressure_gate_diag,
    compute_pressure_share_rew,
    compute_pressure_unloaded_rew,
)

B = len(COMMON_BODY_NAMES)
Z = len(SUPPORT_ZONES)
IDX = {n: i for i, n in enumerate(COMMON_BODY_NAMES)}
ZONE_ORDER = list(SUPPORT_ZONES)


def _forces(spec: dict, num_envs: int = 1) -> torch.Tensor:
    """[E, B, 3] with the given per-body vertical newtons."""
    f = torch.zeros(num_envs, B, 3)
    for name, newtons in spec.items():
        f[:, IDX[name], 2] = newtons
    return f


def _valid(*cols, num_envs: int = 1) -> torch.Tensor:
    return torch.tensor([list(cols)] * num_envs, dtype=torch.float32)


ZM = build_zone_matrix()


def test_zone_matrix_partitions_every_body_exactly_once():
    assert ZM.shape == (Z, B)
    assert torch.equal(ZM.sum(0), torch.ones(B))
    hands = ZM[ZONE_ORDER.index("HANDS")]
    assert hands[IDX["L_Wrist"]] == 1 and hands[IDX["L_Hand"]] == 1
    assert hands[IDX["L_Elbow"]] == 0  # forearm is its own zone


def test_zone_matrix_rejects_an_unpooled_body():
    zones = {k: v for k, v in SUPPORT_ZONES.items() if k != "HEAD"}
    with pytest.raises(ValueError, match="pooled"):
        build_zone_matrix(zones)


def test_zone_matrix_rejects_a_double_pooled_body():
    zones = dict(SUPPORT_ZONES)
    zones["HANDS"] = zones["HANDS"] + ["L_Elbow"]
    with pytest.raises(ValueError, match="pooled"):
        build_zone_matrix(zones)


def test_share_reward_is_one_when_the_distributions_match():
    sim = _forces({"L_Hand": 300.0, "R_Hand": 400.0})
    ref = _forces({"L_Hand": 150.0, "R_Hand": 200.0})  # half the load, same shares
    r = compute_pressure_share_rew(sim, ref, _valid(1.0, 1.0, 1.0), ZM)
    assert torch.allclose(r, torch.ones(1), atol=1e-6)


def test_share_reward_is_invariant_to_a_uniform_gain_error():
    """The measured caveat: the mat reads 0.74-0.83 BW on forearm holds."""
    sim = _forces({"L_Elbow": 350.0, "R_Elbow": 350.0, "L_Hand": 26.0})
    full = _forces({"L_Elbow": 260.0, "R_Elbow": 260.0, "L_Hand": 19.0})
    scaled = _forces({"L_Elbow": 202.8, "R_Elbow": 202.8, "L_Hand": 14.82})  # x0.78
    a = compute_pressure_share_rew(sim, full, _valid(1.0, 1.0, 1.0), ZM)
    b = compute_pressure_share_rew(sim, scaled, _valid(1.0, 1.0, 1.0), ZM)
    assert torch.allclose(a, b, atol=1e-5)


def test_share_reward_falls_when_load_moves_to_the_wrong_zone():
    ref = _forces({"L_Hand": 350.0, "R_Hand": 350.0})
    good = compute_pressure_share_rew(
        _forces({"L_Hand": 360.0, "R_Hand": 360.0}), ref, _valid(1.0, 1.0, 1.0), ZM)
    bad = compute_pressure_share_rew(
        _forces({"L_Hand": 260.0, "R_Hand": 260.0, "L_Toe": 200.0}),
        ref, _valid(1.0, 1.0, 1.0), ZM)
    assert float(good) > float(bad)
    # TV = 200/720 -> exp(-3 * 0.2778)
    assert float(bad) == pytest.approx(float(torch.exp(torch.tensor(-3 * 200 / 720))), abs=1e-4)


def test_share_reward_is_gated_by_the_on_mat_column_not_coverage():
    sim = _forces({"L_Hand": 700.0})
    ref = _forces({"L_Hand": 500.0})
    # coverage poor (0.5) but on-mat column good -> still supervised
    assert float(compute_pressure_share_rew(sim, ref, _valid(0.5, 0.5, 1.0), ZM)) == 1.0
    # on-mat column poor -> gated off regardless of coverage
    assert float(compute_pressure_share_rew(sim, ref, _valid(1.0, 1.0, 0.4), ZM)) == 0.0


def test_share_reward_is_zero_in_flight_on_either_side():
    airborne = _forces({"L_Hand": 1.0})
    grounded = _forces({"L_Hand": 700.0})
    v = _valid(1.0, 1.0, 1.0)
    assert float(compute_pressure_share_rew(airborne, grounded, v, ZM)) == 0.0
    assert float(compute_pressure_share_rew(grounded, airborne, v, ZM)) == 0.0


def test_unloaded_penalty_is_zero_when_the_policy_matches():
    sim = _forces({"L_Hand": 350.0, "R_Hand": 375.0})
    ref = _forces({"L_Hand": 300.0, "R_Hand": 330.0})
    p = compute_pressure_unloaded_rew(sim, ref, _valid(1.0, 1.0, 1.0), ZM)
    assert float(p) == pytest.approx(0.0, abs=1e-6)


REF_N = 0.1 * 74.0 * 9.81  # the term saturates at 10 % of body weight


def test_unloaded_penalty_charges_load_on_an_empty_zone_at_the_10pct_scale():
    """Normalising by FULL body weight made this ~30x too weak to bite: realistic
    violations are 5-45 N, so the term sat in the bottom 6 % of its range and the
    policy bought stability by leaning on unloaded limbs anyway (design §7.6)."""
    sim = _forces({"L_Hand": 500.0, "L_Toe": 25.0})
    ref = _forces({"L_Hand": 630.0})  # measurement: nothing on the feet
    p = compute_pressure_unloaded_rew(sim, ref, _valid(1.0, 1.0, 1.0), ZM)
    assert float(p) == pytest.approx(25.0 / REF_N, abs=1e-4)
    # the observed regression (25 N) must now cost a meaningful fraction of reward
    assert 0.3 < float(p) < 0.4


def test_unloaded_penalty_saturates_above_ten_percent_body_weight():
    sim = _forces({"L_Hand": 500.0, "L_Toe": 250.0})
    ref = _forces({"L_Hand": 630.0})
    p = compute_pressure_unloaded_rew(sim, ref, _valid(1.0, 1.0, 1.0), ZM)
    assert float(p) == pytest.approx(1.0, abs=1e-6)


def test_unloaded_penalty_is_one_sided():
    """Reference loads a zone the policy does not: no penalty (that is the
    share term's job). The penalty must never push load *onto* a body."""
    sim = _forces({"L_Hand": 700.0})
    ref = _forces({"L_Hand": 400.0, "L_Toe": 300.0})
    assert float(compute_pressure_unloaded_rew(sim, ref, _valid(1.0, 1.0, 1.0), ZM)) == 0.0


def test_unloaded_penalty_uses_the_per_body_gate_not_the_on_mat_gate():
    sim = _forces({"L_Hand": 500.0, "L_Toe": 250.0})
    ref = _forces({"L_Hand": 630.0})
    # per-body confidence low -> a FALSE zero could punish a real contact
    assert float(compute_pressure_unloaded_rew(sim, ref, _valid(1.0, 0.4, 1.0), ZM)) == 0.0
    assert float(compute_pressure_unloaded_rew(sim, ref, _valid(0.4, 1.0, 0.4), ZM)) > 0.0


def test_unloaded_penalty_is_zero_when_the_reference_measured_no_load():
    sim = _forces({"L_Toe": 400.0})
    ref = _forces({"L_Hand": 1.0})  # reference in flight / mat saw nothing
    assert float(compute_pressure_unloaded_rew(sim, ref, _valid(1.0, 1.0, 1.0), ZM)) == 0.0


def test_missing_on_mat_column_raises_rather_than_silently_degrading():
    sim, ref = _forces({"L_Hand": 700.0}), _forces({"L_Hand": 500.0})
    with pytest.raises(RuntimeError, match="add_onmat_gate_to_motions"):
        compute_pressure_share_rew(sim, ref, _valid(1.0, 1.0), ZM)


def test_terms_return_per_env_zeros_when_the_motion_has_no_measured_channel():
    """Shape matters: combine_rewards broadcasts, so a size-1 fallback would
    silently apply one env's value to every env."""
    sim = _forces({"L_Hand": 700.0}, num_envs=4)
    assert torch.equal(compute_pressure_share_rew(sim, None, None, ZM), torch.zeros(4))
    assert torch.equal(compute_pressure_unloaded_rew(sim, None, None, ZM), torch.zeros(4))


def test_gate_diagnostic_reports_the_requested_column():
    v = torch.tensor([[1.0, 1.0, 0.2], [1.0, 0.3, 1.0]])
    assert torch.equal(compute_pressure_gate_diag(v, VALID_ON_MAT), torch.tensor([0.0, 1.0]))
    assert torch.equal(compute_pressure_gate_diag(v, VALID_PER_BODY), torch.tensor([1.0, 0.0]))


def test_batched_envs_are_scored_independently():
    sim = torch.cat([_forces({"L_Hand": 700.0}), _forces({"L_Hand": 660.0, "L_Toe": 40.0})])
    ref = torch.cat([_forces({"L_Hand": 630.0}), _forces({"L_Hand": 630.0})])
    v = _valid(1.0, 1.0, 1.0, num_envs=2)
    r = compute_pressure_share_rew(sim, ref, v, ZM)
    p = compute_pressure_unloaded_rew(sim, ref, v, ZM)
    assert float(r[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(r[1]) < float(r[0])
    assert float(p[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(p[1]) == pytest.approx(40.0 / REF_N, abs=1e-4)


def test_crow_hold_reproduces_the_offline_calibration():
    """Regression against notes/Pressure_supervision_design.MD 4.

    Measured crow hold: the human puts ~627 N through the hands and 12 N
    elsewhere.  The @6250 checkpoint keeps a toe down carrying 27.3 % of body
    weight; the @15635 checkpoint carries 2.5 %.
    """
    ref = _forces({"L_Wrist": 300.0, "R_Wrist": 327.0, "L_Toe": 12.0})
    mg = 74.0 * 9.81
    v = _valid(1.0, 1.0, 1.0)

    bad = _forces({"L_Wrist": 250.0, "R_Wrist": 277.0, "R_Toe": 0.273 * mg})
    good = _forces({"L_Wrist": 330.0, "R_Wrist": 360.0, "L_Toe": 0.025 * mg})

    # At the 10 %-BW scale the @6250 crow (27.3 % BW on the feet) saturates the
    # penalty, and the @15635 crow (2.5 % BW) costs a quarter of it. Under the
    # old full-mg scale these read 0.273 and 0.025 — both negligible after the
    # -0.3 weight, which is exactly why the term failed to deter (design §7.6).
    assert float(compute_pressure_unloaded_rew(bad, ref, v, ZM)) == pytest.approx(1.0, abs=1e-6)
    assert float(compute_pressure_unloaded_rew(good, ref, v, ZM)) == pytest.approx(0.25, abs=0.02)
    # and the share term separates them in the same direction
    assert float(compute_pressure_share_rew(good, ref, v, ZM)) > \
        float(compute_pressure_share_rew(bad, ref, v, ZM)) + 0.4
