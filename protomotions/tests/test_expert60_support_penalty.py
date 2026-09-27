"""Tests for the unwanted-support penalty (``protomotions/envs/control/support_penalty.py``).

Design and calibration: ``expert_revist/contact_reward/README.MD``. Pure CPU.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control.support_penalty import (
    ChargedLoadEMA,
    ema_alpha,
    unwanted_support,
    zone_pooling_matrix,
)
from protomotions.tests.test_expert60_ft_a import _graph_with_near_start_hold

BODIES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]
ZONES = {"L_FOOT": ["L_Ankle", "L_Toe"], "R_FOOT": ["R_Ankle", "R_Toe"], "HEAD": ["Neck", "Head"],
         "L_HAND": ["L_Wrist", "L_Hand"], "R_HAND": ["R_Wrist", "R_Hand"]}
ORDER = list(ZONES)
B = {n: i for i, n in enumerate(BODIES)}
MG = 74.0 * 9.81
DT = 1.0 / 30.0


def _crow(num_envs=1, foot_load=150.0, toe_z=0.40, hands_in_goal=True):
    """A crow-like frame: hands on the floor, a foot the reference keeps up."""
    forces = torch.zeros(num_envs, len(BODIES), 3)
    forces[:, B["L_Hand"], 2] = 300.0
    forces[:, B["R_Hand"], 2] = 300.0
    forces[:, B["L_Toe"], 2] = foot_load
    ref = torch.full((num_envs, len(BODIES), 3), 0.6)
    ref[:, B["L_Hand"], 2] = ref[:, B["R_Hand"], 2] = 0.04
    ref[:, B["L_Wrist"], 2] = ref[:, B["R_Wrist"], 2] = 0.06
    ref[:, B["L_Toe"], 2] = toe_z
    ref[:, B["L_Ankle"], 2] = toe_z + 0.05
    goal = torch.zeros(num_envs, len(ORDER), dtype=torch.bool)
    goal[:, ORDER.index("L_HAND")] = hands_in_goal
    goal[:, ORDER.index("R_HAND")] = hands_in_goal
    return forces, ref, goal


def _call(forces, ref, goal, in_hold=True, excluded=None, clear=0.25, frac=0.1):
    n = ref.shape[0]
    return unwanted_support(
        ground_forces=forces, ref_body_pos=ref, goal_ground=goal,
        in_hold=torch.full((n,), in_hold, dtype=torch.bool),
        zone_matrix=zone_pooling_matrix(ZONES, ORDER, BODIES), clear_height=clear,
        load_ref_n=frac * MG, excluded=excluded)


def test_foot_down_in_a_hands_only_hold_is_charged_and_hands_are_not():
    pen, charged = _call(*_crow(foot_load=36.3))          # half the 0.1 mg saturation
    assert charged.item() == pytest.approx(36.3)          # the toe pools into L_FOOT; hands excluded
    assert pen.item() == pytest.approx(36.3 / (0.1 * MG))


def test_saturates_at_the_reference_load():
    pen, charged = _call(*_crow(foot_load=5000.0))
    assert pen.item() == 1.0 and charged.item() == pytest.approx(5000.0)


def test_zone_in_the_goal_ground_set_is_never_charged():
    forces, ref, goal = _crow()
    goal[:, ORDER.index("L_FOOT")] = True
    pen, charged = _call(forces, ref, goal)
    assert pen.item() == 0.0 and charged.item() == 0.0


def test_zone_the_reference_keeps_low_is_not_charged_min_over_members():
    # ankle high, toe low: the zone's lowest member decides (float-biased foot)
    pen, _ = _call(*_crow(toe_z=0.12))
    assert pen.item() == 0.0
    pen, _ = _call(*_crow(toe_z=0.24))
    assert pen.item() == 0.0            # still under the 0.25 m clearance
    pen, _ = _call(*_crow(toe_z=0.26))
    assert pen.item() > 0.0


def test_gate_exclusion_and_missing_column():
    forces, ref, goal = _crow(num_envs=2)
    pen, _ = _call(forces, ref, goal, in_hold=False)
    assert torch.all(pen == 0)
    pen, charged = _call(forces, ref, goal, excluded=torch.tensor([True, False]))
    assert pen[0] == 0 and charged[0] == 0 and pen[1] > 0
    pen, charged = _call(None, ref, goal)
    assert torch.all(pen == 0) and torch.all(charged == 0)


def test_negative_normal_force_is_noise_not_load():
    forces, ref, goal = _crow(foot_load=-500.0)
    pen, charged = _call(forces, ref, goal)
    assert pen.item() == 0.0 and charged.item() == 0.0


def test_zone_pooling_matrix_rejects_an_unresolvable_zone():
    with pytest.raises(ValueError):
        zone_pooling_matrix({"TAIL": ["Tail"]}, ["TAIL"], BODIES)


def test_goal_conditioned_experiment_wires_the_support_penalty(tmp_path):
    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.robot_configs.factory import robot_config

    _graph_with_near_start_hold(tmp_path)  # writes tmp_path/g.pt
    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)

    def build(argv):
        args = parser.parse_args(["--hold-graph-file", str(tmp_path / "g.pt")] + argv)
        args.motion_file = "unused.pt"
        args.scenes_file = None
        args.batch_size = 32
        args.training_max_steps = 256
        robot_cfg = robot_config("smpl_yogi")
        exp.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
        env_cfg = exp.env_config(robot_cfg, args)
        return env_cfg, exp.agent_config(robot_cfg, env_cfg, args)

    base_env, base_agent = build([])
    assert not any("support" in k for k in base_env.reward_components)   # fine-tune A's dict
    ctrl = base_env.control_components["contact_graph"]
    assert ctrl.support_clear_height == 0.25 and ctrl.support_load_ref_frac == 0.1
    assert ctrl.support_exclude_motions == []
    assert ctrl.support_ema_tau_s == 0.0          # ft_b's per-frame term unless asked

    env_cfg, agent_cfg = build([
        "--support-penalty-weight", "-0.3", "--support-clear-height", "0.3",
        "--support-exclude-motions", "Cockerel_Pose-b", "Scale_Pose_or_Tolasana_-a",
    ])
    rew = env_cfg.reward_components
    params = rew["unwanted_support_rew"].get_params()
    assert params["weight"] == -0.3 and params["zero_during_grace_period"] is True
    assert rew["diag_unwanted_support_n"].get_params()["weight"] == 0.0
    assert rew["diag_support_gate"].get_params()["weight"] == 0.0
    ctrl = env_cfg.control_components["contact_graph"]
    assert ctrl.support_clear_height == 0.3
    assert ctrl.support_exclude_motions == ["Cockerel_Pose-b", "Scale_Pose_or_Tolasana_-a"]
    # the observation contract a warm start loads into is untouched
    assert agent_cfg.model.actor.in_keys == base_agent.model.actor.in_keys
    assert set(env_cfg.observation_components) == set(base_env.observation_components)

    env_cfg, _ = build(["--support-penalty-weight", "-0.3", "--support-ema-tau", "0.25"])
    assert env_cfg.control_components["contact_graph"].support_ema_tau_s == 0.25

    env_cfg, _ = build(["--support-penalty-weight", "0"])   # the calibration smoke
    assert env_cfg.reward_components["unwanted_support_rew"].get_params()["weight"] == 0.0

    with pytest.raises(ValueError):
        build(["--support-penalty-weight", "0.3"])


# --- Time-averaged pricing (ft_b report §8) --------------------------------------------------- #

def _drive(ema, loads, gate=1.0):
    """One env step per load, as the control drives it; returns the per-step penalties."""
    out = []
    for load in loads:
        ema.mark_step()
        out.append(ema.price(torch.tensor([float(load)]), torch.tensor([gate]), 0.1 * MG).item())
    return out


def test_ema_alpha_and_the_per_frame_default():
    assert ema_alpha(0.0, DT) == 1.0
    assert ema_alpha(0.25, DT) == pytest.approx(DT / (0.25 + DT))
    with pytest.raises(ValueError):
        ChargedLoadEMA(1, 0.0, DT)          # tau 0 is the per-frame term, not an EMA


def test_ema_charges_a_tapping_foot_like_a_resting_one_and_a_lift_stays_cheap():
    # A 4 s crow hold three ways: resting on the foot at 250 N; striking at 650 N on 3 frames
    # in 8 (about the same mean load; ft_b's Crow -b); a held lift with a 400 N touch every 2 s.
    frames = int(4.0 / DT)
    rest = [250.0] * frames
    tap = [650.0 if i % 8 < 3 else 0.0 for i in range(frames)]
    lift = [400.0 if i % 60 == 0 else 0.0 for i in range(frames)]

    def per_frame(loads):
        return sum(min(x / (0.1 * MG), 1.0) for x in loads) / len(loads)

    assert per_frame(rest) == 1.0
    assert per_frame(tap) == pytest.approx(3 / 8, abs=0.01)      # the loophole: duty cycle
    steady = slice(int(0.5 / DT), None)                           # past the episode-start ramp
    mean = lambda xs: sum(xs) / len(xs)                           # noqa: E731
    assert min(_drive(ChargedLoadEMA(1, 0.25, DT), rest)[steady]) == 1.0
    assert mean(_drive(ChargedLoadEMA(1, 0.25, DT), tap)[steady]) > 0.95
    assert mean(_drive(ChargedLoadEMA(1, 0.25, DT), lift)[steady]) < 0.15


def test_ema_advances_once_per_step_and_resets_only_the_given_rows():
    ema = ChargedLoadEMA(2, 0.25, DT)
    charged, gate = torch.tensor([100.0, 100.0]), torch.ones(2)
    ema.mark_step()
    first = ema.price(charged, gate, 0.1 * MG)
    again = ema.price(charged, gate, 0.1 * MG)       # a second context build in the same step
    assert torch.equal(first, again)
    assert ema.state[0].item() == pytest.approx(ema.alpha * 100.0)
    ema.reset(torch.tensor([0]))
    assert ema.state[0].item() == 0.0
    assert ema.state[1].item() == pytest.approx(ema.alpha * 100.0)


def test_ema_penalty_is_gated_and_the_average_decays_between_holds():
    ema = ChargedLoadEMA(1, 0.25, DT)
    _drive(ema, [300.0] * 30)                       # a second of charged load in a hold
    held = ema.state.item()
    assert _drive(ema, [0.0] * 5, gate=0.0) == [0.0] * 5
    assert ema.state.item() < held


def _support_ctx(num_envs, foot_n):
    from protomotions.tests.test_contact_graph import SIM_BODY_NAMES

    forces = torch.zeros(num_envs, len(SIM_BODY_NAMES), 3)
    forces[:, SIM_BODY_NAMES.index("L_Toe"), 2] = foot_n
    ref = torch.full((num_envs, len(SIM_BODY_NAMES), 3), 0.5)   # every joint above 0.25 m
    return SimpleNamespace(
        current=SimpleNamespace(rigid_body_ground_forces=forces),
        mimic=SimpleNamespace(ref_state=SimpleNamespace(rigid_body_pos=ref)),
    )


@pytest.mark.parametrize("tau", [0.0, 0.25])
def test_control_prices_the_support_load_once_per_env_step(tmp_path, monkeypatch, tau):
    from protomotions.tests.test_contact_graph import _make_control

    control = _make_control(
        tmp_path, monkeypatch, include_current_segment=True, interval_schedule=True,
        support_ema_tau_s=tau,
    )
    # Motion 0 at t = 3 s is inside its "one_leg" segment [2, 5]: R_FOOT is the ground set,
    # so a load on the left foot is charged.
    control.env.motion_manager.motion_times[:] = 3.0
    control.reset(torch.arange(4))
    ctx = _support_ctx(4, foot_n=50.0)
    ref_n = 0.1 * MG
    control.step()
    pen, charged, gate = control._unwanted_support(ctx)
    assert torch.all(gate == 1.0) and torch.allclose(charged, torch.full((4,), 50.0))
    if tau == 0.0:
        assert control._support_ema is None            # exactly ft_b's per-frame term
        assert torch.allclose(pen, torch.full((4,), 50.0 / ref_n))
        return
    a = control._support_ema.alpha
    assert torch.allclose(pen, torch.full((4,), a * 50.0 / ref_n))
    again, _, _ = control._unwanted_support(ctx)       # the post-reset rebuild of the same step
    assert torch.equal(pen, again)
    control.step()
    pen2, _, _ = control._unwanted_support(ctx)
    assert torch.allclose(pen2, torch.full((4,), (a + a * (1 - a)) * 50.0 / ref_n))
    control.reset(torch.tensor([0]))                    # a new episode in env 0 only
    after, _, _ = control._unwanted_support(ctx)
    assert after[0].item() == 0.0 and torch.allclose(after[1:], pen2[1:])
