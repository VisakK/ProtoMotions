# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card E6 (graph-growth PLAN.MD): the lineage flag, per-lineage diagnostics, the AMP state on a warm start."""

import argparse
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.amp.goal_conditioned import (
    GoalConditionedAMP,
    GoalConditionedAMPAgentConfig,
    GoalConditionedAMPComponent,
    LineageRewardMeter,
    amp_training_state_keys,
    lineage_match,
    optimizer_step_count,
    parse_amp_lineage_rule,
    parse_amp_lineage_weights,
    warm_start_calibration,
)
from protomotions.agents.utils.normalization import RewardRunningMeanStd

STEMS = [
    "220923_Crane_Crow_Pose_or_Bakasana_-a",
    "220923_Crane_Crow_Pose_or_Bakasana_-a_x3s",
    "220923_Tree_Pose_or_Vrksasana_-a",
    "SYN_E1_press_high_s0_t6px",
    "SYN_E1_press_high_s0_t6px_x7s",
    "SYN_B1_jumpplank_high_s1_t6rpx",
]


# ---------------------------------------------------------------------- #
# Item 1: PATTERN=W
# ---------------------------------------------------------------------- #
def test_parser_keeps_the_order_given_and_reads_floats():
    rules = parse_amp_lineage_weights(["SYN_E1=0.25", "SYN_=0.5", "Crane_Crow=1", "Tree=0"])
    assert list(rules) == ["SYN_E1", "SYN_", "Crane_Crow", "Tree"]
    assert rules == {"SYN_E1": 0.25, "SYN_": 0.5, "Crane_Crow": 1.0, "Tree": 0.0}
    assert parse_amp_lineage_weights([]) == {} == parse_amp_lineage_weights(None)
    assert parse_amp_lineage_rule("SYN_=5e-1") == ("SYN_", 0.5)


@pytest.mark.parametrize("bad", ["SYN_", "SYN_=", "=0.5", "SYN_=abc", "SYN_=-1", "SYN_=-0.01", "SYN_=nan",
                                 "SYN_=inf", "A=B=0.5", " SYN_=0.5", "SYN_ =0.5", ""])
def test_parser_rejects_malformed_rules(bad):
    with pytest.raises(ValueError):
        parse_amp_lineage_rule(bad)
    with pytest.raises(ValueError):
        parse_amp_lineage_weights(["Tree=1", bad])


def test_parser_rejects_a_pattern_given_twice():
    with pytest.raises(ValueError, match="given twice"):
        parse_amp_lineage_weights(["SYN_=0.5", "SYN_=0.25"])


def test_the_default_config_has_no_rules():
    # G1's resolved config carries {} and the flag's default parses to {}: an unchanged config.
    field = {f.name: f for f in dataclasses.fields(GoalConditionedAMPAgentConfig)}["amp_lineage_weights"]
    assert field.default_factory() == {}


def _amp_parser():
    from examples.experiments.mimic import mlp_goal_conditioned_amp as exp

    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)
    return parser


def test_the_flag_parses_into_the_config_dict_and_fails_loudly():
    parser = _amp_parser()
    assert parser.parse_args([]).amp_lineage_weights == []
    args = parser.parse_args(["--amp-lineage-weights", "SYN_=0.5", "Crane_Crow=0.5",
                              "--amp-demo-exclude-motions", "Scorpion_pose_or_vrischikasana-b", "SYN_"])
    assert args.amp_lineage_weights == ["SYN_=0.5", "Crane_Crow=0.5"]          # strings: config.yaml round-trips
    assert parse_amp_lineage_weights(args.amp_lineage_weights) == {"SYN_": 0.5, "Crane_Crow": 0.5}
    assert args.amp_demo_exclude_motions == ["Scorpion_pose_or_vrischikasana-b", "SYN_"]
    for bad in ("SYN_", "SYN_=-1", "SYN_=x"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--amp-lineage-weights", bad])


# ---------------------------------------------------------------------- #
# Item 2: matching and per-group accumulation
# ---------------------------------------------------------------------- #
def test_lineage_match_first_rule_wins_in_the_order_given():
    weights, which = lineage_match(STEMS, {"SYN_": 0.5, "SYN_E1": 0.25})
    assert which == [-1, -1, -1, 0, 0, 0] and weights == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    weights, which = lineage_match(STEMS, {"SYN_E1": 0.25, "SYN_": 0.5})
    assert which == [-1, -1, -1, 0, 0, 1] and weights == [1.0, 1.0, 1.0, 0.25, 0.25, 0.5]
    weights, which = lineage_match(STEMS, {"Crane_Crow": 0.5})
    assert which == [0, 0, -1, -1, -1, -1] and weights == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert lineage_match(STEMS, {}) == ([1.0] * 6, [-1] * 6)


def test_meter_splits_unweighted_means_and_the_syn_share():
    meter = LineageRewardMeter("cpu")
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
    syn = torch.tensor([False, True, False, True])
    meter.record(raw, raw * torch.where(syn, 0.5, 1.0), syn)
    meter.record(torch.tensor([5.0, 6.0, 7.0, 8.0]), torch.tensor([5.0, 3.0, 7.0, 4.0]), syn)
    log = {k: float(v) for k, v in meter.pop_log().items()}
    assert log["amp/syn_sample_share"] == pytest.approx(0.5)
    assert log["amp/reward_mean_syn"] == pytest.approx((2 + 4 + 6 + 8) / 4)
    assert log["amp/reward_mean_syn_weighted"] == pytest.approx((1 + 2 + 3 + 4) / 4)
    assert log["amp/reward_mean_human"] == pytest.approx((1 + 3 + 5 + 7) / 4)
    assert meter.pop_log() == {}                                  # reset; an empty epoch logs nothing


def test_meter_leaves_out_a_group_without_samples():
    meter = LineageRewardMeter("cpu")
    meter.record(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 3.0]), torch.tensor([False, False]))
    log = meter.pop_log()
    assert set(log) == {"amp/syn_sample_share", "amp/reward_mean_human"}
    assert float(log["amp/syn_sample_share"]) == 0.0 and float(log["amp/reward_mean_human"]) == pytest.approx(2.0)
    meter.record(torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([True]))
    assert set(meter.pop_log()) == {"amp/syn_sample_share", "amp/reward_mean_syn", "amp/reward_mean_syn_weighted"}


class _FakeDiscriminator:
    """``discriminator(td)[out_key]`` -> logits; ``compute_disc_reward`` = identity."""

    def __init__(self, logits):
        self.logits = logits
        self.module = SimpleNamespace(config=SimpleNamespace(out_keys=["disc_logits"]),
                                      compute_disc_reward=lambda x: x)

    def __call__(self, td):
        return {"disc_logits": self.logits.unsqueeze(-1)}


class _Buffer:
    def __init__(self):
        self.data = {}

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()


def _component(rules, motion_ids, logits):
    comp = object.__new__(GoalConditionedAMPComponent)
    files = [f"/x/motions/{s}.motion" for s in STEMS]
    cfg = SimpleNamespace(amp_lineage_weights=rules, normalize_rewards=False,
                          amp_parameters=SimpleNamespace(discriminator_reward_threshold=0.0))
    comp.agent = SimpleNamespace(config=cfg, device="cpu", experience_buffer=_Buffer(),
                                 motion_lib=SimpleNamespace(motion_files=files),
                                 motion_manager=SimpleNamespace(motion_ids=motion_ids))
    comp._motion_amp_weight = comp._motion_is_syn = comp._lineage_meter = None
    comp._lineage_ready = False
    comp.use_disc_critic = False
    comp.num_cumulative_bad_transitions = torch.zeros(len(motion_ids), dtype=torch.int32)
    comp.discriminator = _FakeDiscriminator(logits)
    return comp


def test_rollout_step_weights_per_env_and_accumulates_by_the_env_motion(capsys):
    motion_ids = torch.tensor([0, 3, 2, 5])                       # Crane_Crow, SYN_E1, Tree, SYN_B1
    comp = _component({"SYN_": 0.5}, motion_ids, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    comp.setup_lineage()
    out = capsys.readouterr().out
    assert "[amp lineage] rule 'SYN_' (w 0.5) matched 3 of 6 motions: SYN_E1_press_high_s0_t6px, " in out
    assert "[amp lineage] 3 of 6 motions are 'syn' (matched); 3 are 'human' (w 1.0)" in out
    comp.record_rollout_step({}, None, torch.zeros(4, dtype=torch.bool), torch.tensor([], dtype=torch.long), {}, 0)
    stored = comp.agent.experience_buffer.data[("amp_rewards", 0)]
    assert stored.tolist() == [1.0, 1.0, 3.0, 2.0]                # the syn envs at x 0.5
    comp.agent.motion_manager.motion_ids = torch.tensor([3, 3, 3, 3])
    comp.discriminator.logits = torch.tensor([8.0, 8.0, 8.0, 8.0])
    comp.record_rollout_step({}, None, torch.zeros(4, dtype=torch.bool), torch.tensor([], dtype=torch.long), {}, 1)
    log = {k: float(v) for k, v in comp._lineage_meter.pop_log().items()}
    assert log["amp/syn_sample_share"] == pytest.approx(6 / 8)
    assert log["amp/reward_mean_syn"] == pytest.approx((2 + 4 + 8 * 4) / 6)
    assert log["amp/reward_mean_syn_weighted"] == pytest.approx((1 + 2 + 4 * 4) / 6)
    assert log["amp/reward_mean_human"] == pytest.approx((1 + 3) / 2)


def test_a_rule_matching_nothing_warns_and_no_rules_do_nothing(capsys):
    comp = _component({"Tree": 0.0, "Tree_Pose": 0.5, "Lotus": 0.5}, torch.tensor([2]), torch.tensor([1.0]))
    comp.setup_lineage()
    out = capsys.readouterr().out
    assert "rule 'Tree' (w 0) matched 1 of 6 motions: 220923_Tree_Pose_or_Vrksasana_-a" in out
    # the warning says why: shadowed by an earlier rule, or no stem contains the pattern at all
    assert ("[amp lineage] WARNING: rule 'Tree_Pose' (w 0.5) matched no motion (of 6; 1 stem(s) contain it, all "
            "taken by an earlier rule)") in out
    assert "[amp lineage] WARNING: rule 'Lotus' (w 0.5) matched no motion (of 6; no stem contains it)" in out
    off = _component({}, torch.tensor([0, 3]), torch.tensor([1.0, 2.0]))
    off.setup_lineage()
    assert capsys.readouterr().out == "" and off._lineage_meter is None and off._env_amp_weight() is None
    off.record_rollout_step({}, None, torch.zeros(2, dtype=torch.bool), torch.tensor([], dtype=torch.long), {}, 0)
    assert off.agent.experience_buffer.data[("amp_rewards", 0)].tolist() == [1.0, 2.0]     # unweighted, as G1


# ---------------------------------------------------------------------- #
# Item 3: the AMP state on a warm start
# ---------------------------------------------------------------------- #
def _stepped_adam(module, steps):
    opt = torch.optim.Adam(module.parameters(), lr=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        module(torch.ones(2, module.in_features)).sum().backward()
        opt.step()
    return opt


def _norm(count=1, var=1.0):
    norm = RewardRunningMeanStd(fabric=None, shape=(1,), gamma=0.99, device="cpu")
    norm.count.fill_(count)
    norm.var.fill_(var)
    return norm


def _checkpoint(with_amp=True, calibrated_target=0.5):
    torch.manual_seed(0)
    sd = {"actor_optimizer": _stepped_adam(torch.nn.Linear(3, 2), 7).state_dict(),
          "critic_optimizer": _stepped_adam(torch.nn.Linear(3, 1), 7).state_dict(),
          "running_reward_norm": _norm(1000, 4.0).state_dict()}
    if with_amp:
        sd["discriminator_optimizer"] = _stepped_adam(torch.nn.Linear(4, 1), 5).state_dict()
        sd["disc_critic_optimizer"] = _stepped_adam(torch.nn.Linear(5, 1), 5).state_dict()
        sd["running_amp_reward_norm"] = _norm(655360000, 155.7582).state_dict()
        sd["amp_weight_calibration"] = {"ratio_history": [[4998, 0.49], [4999, 0.53]],
                                        "calibrated_target": calibrated_target}
    return sd


def _agent(amp_w=0.5, cal_ratio=0.0, non_amp_checkpoint=False):
    agent = object.__new__(GoalConditionedAMP)
    agent.config = SimpleNamespace(
        normalize_rewards=True, adaptive_lr=SimpleNamespace(enabled=False),
        advantage_normalization=SimpleNamespace(enabled=False, use_ema=False),
        amp_reward_w_target=amp_w, amp_calibrate_style_ratio=cal_ratio)
    agent.running_reward_norm = _norm()
    agent.actor_optimizer = torch.optim.Adam(torch.nn.Linear(3, 2).parameters(), lr=1e-4)
    agent.critic_optimizer = torch.optim.Adam(torch.nn.Linear(3, 1).parameters(), lr=1e-4)
    agent._warm_start_from_non_amp_checkpoint = non_amp_checkpoint
    comp = object.__new__(GoalConditionedAMPComponent)
    comp.agent = agent
    comp.use_disc_critic = True
    comp.discriminator_optimizer = torch.optim.Adam(torch.nn.Linear(4, 1).parameters(), lr=1e-4)
    comp.disc_critic_optimizer = torch.optim.Adam(torch.nn.Linear(5, 1).parameters(), lr=1e-4)
    comp.running_reward_norm = _norm()
    comp.ratio_history, comp.calibrated_target = [], None
    agent.amp_component = comp
    return agent, comp


def test_warm_start_restores_the_amp_state_and_the_configured_weight_wins(capsys):
    agent, comp = _agent(amp_w=0.3)
    agent._load_optimization_state(_checkpoint(calibrated_target=0.5))
    out = capsys.readouterr().out
    assert optimizer_step_count(agent.actor_optimizer) == 7                      # PPO's part still runs
    assert optimizer_step_count(comp.discriminator_optimizer) == 5
    assert optimizer_step_count(comp.disc_critic_optimizer) == 5
    assert int(comp.running_reward_norm.count) == 655360000
    assert float(comp.running_reward_norm.var) == pytest.approx(155.7582)
    assert comp.ratio_history == []                                              # the old run's epoch clock
    assert comp.calibrated_target is None and comp.target_w() == pytest.approx(0.3)
    assert ("Warm start: restored AMP training state from checkpoint: discriminator optimizer (step 5, lr 0.0001), "
            "disc-critic optimizer (step 5, lr 0.0001), AMP reward normaliser (count 655360000, var 155.8)") in out
    assert ("Warm start: AMP weight target 0.3 (configured) replaces the checkpoint's calibrated target 0.5 "
            "(calibration off); 2 calibration ratio-history entries") in out


def test_warm_start_at_g1s_weight_keeps_it(capsys):
    agent, comp = _agent(amp_w=0.5)
    agent._load_optimization_state(_checkpoint(calibrated_target=0.5))
    assert comp.target_w() == pytest.approx(0.5) and comp.calibrated_target is None
    assert "AMP weight target 0.5 (configured; calibration off; equals the checkpoint's calibrated target)" \
        in capsys.readouterr().out


def test_warm_start_from_a_ppo_checkpoint_skips_the_amp_state(capsys):
    agent, comp = _agent()
    agent._load_optimization_state(_checkpoint(with_amp=False))
    out = capsys.readouterr().out
    assert optimizer_step_count(agent.actor_optimizer) == 7
    assert optimizer_step_count(comp.discriminator_optimizer) == 0
    assert int(comp.running_reward_norm.count) == 1
    assert "Warm start: no AMP training state loaded -- the checkpoint carries none" in out
    # a checkpoint without discriminator weights never loads AMP optimiser state, even if it has some
    agent, comp = _agent(non_amp_checkpoint=True)
    agent._load_optimization_state(_checkpoint(with_amp=True))
    assert optimizer_step_count(comp.discriminator_optimizer) == 0
    assert "the checkpoint has no discriminator (a PPO checkpoint)" in capsys.readouterr().out


def test_warm_start_with_part_of_the_amp_state_raises():
    agent, _ = _agent()
    sd = _checkpoint()
    del sd["disc_critic_optimizer"]
    with pytest.raises(KeyError, match="disc_critic_optimizer"):
        agent._load_optimization_state(sd)
    assert amp_training_state_keys(sd, use_disc_critic=False, normalize_rewards=True) == \
        (["discriminator_optimizer", "running_amp_reward_norm"], [])
    assert amp_training_state_keys({}, True, False) == ([], ["discriminator_optimizer", "disc_critic_optimizer"])


def test_calibration_rule():
    # off: the configured target is the weight, whatever the checkpoint says
    assert warm_start_calibration(0.5, 0.5, 0.0)[0] is None
    target, note = warm_start_calibration(0.5, 0.2, 0.0)
    assert target is None and "replaces the checkpoint's calibrated target 0.5" in note
    assert warm_start_calibration(None, 0.2, 0.0)[0] is None
    # on: a restored calibration is kept, frozen; without one the run calibrates itself
    assert warm_start_calibration(0.5, 0.1, 0.25) == (0.5, warm_start_calibration(0.5, 0.1, 0.25)[1])
    assert warm_start_calibration(None, 0.1, 0.25)[0] is None


def test_calibration_on_keeps_the_restored_target_frozen(capsys):
    agent, comp = _agent(amp_w=0.1, cal_ratio=0.25)
    agent._load_optimization_state(_checkpoint(calibrated_target=0.42))
    assert comp.calibrated_target == pytest.approx(0.42) and comp.ratio_history == []
    comp.agent.config.amp_reward_w_start_epoch = 0
    comp._maybe_calibrate(5)                                          # frozen: never recomputed
    assert comp.target_w() == pytest.approx(0.42)
    assert "the checkpoint's calibration, kept frozen" in capsys.readouterr().out


def test_the_demonstration_set_is_printed_without_syn_stems(capsys):
    comp = _component({"SYN_": 0.5}, torch.tensor([0]), torch.tensor([1.0]))
    stems = STEMS + ["220923_Scorpion_pose_or_vrischikasana-b", "220923_Short_clip_-a"]
    comp.agent.motion_lib = SimpleNamespace(motion_files=[f"/x/{s}.motion" for s in stems],
                                            motion_lengths=torch.tensor([9.0] * 7 + [0.5]))
    comp.agent.env = SimpleNamespace(simulator=SimpleNamespace(dt=1 / 30))
    comp.agent.config.demo_exclude_regex = r"_x\d+s$"
    comp.agent.config.demo_exclude_motions = ["Scorpion_pose_or_vrischikasana-b", "SYN_"]
    comp.agent.config.demo_min_time_steps = 20
    comp._demo_ids = None
    comp._setup_demo_sampler()
    out = capsys.readouterr().out
    assert ("[amp demos] 2 of 8 motions serve as demonstrations, 16.7 s usable (t >= 0.667 s); excluded 2 by "
            "regex '_x\\\\d+s$', 3 by name ['Scorpion_pose_or_vrischikasana-b', 'SYN_'], 1 too short") in out
    assert ("[amp demos] excluded by name: SYN_E1_press_high_s0_t6px, SYN_B1_jumpplank_high_s1_t6rpx, "
            "220923_Scorpion_pose_or_vrischikasana-b") in out
    assert "[amp demos] demonstrations: 220923_Crane_Crow_Pose_or_Bakasana_-a, 220923_Tree_Pose_or_Vrksasana_-a\n" in out
    assert "[amp demos] lineage-matched motions among the demonstrations: 0\n" in out
