# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FSQ-C MaskedMimic student (chunked discrete-intent codes).

Covers the properties the design leans on:

* chunk semantics — codes refresh on the first post-reset step, hold
  mid-chunk, refresh on counter expiry, on the contact-event flag, and on
  ``flush_held_intent``;
* the replay contract — a batch re-forwarded from stored (pre-step state +
  observations) reproduces the rollout's ``privileged_action`` and CE targets
  exactly, and the TensorDict a rollout step leaves behind carries the
  *pre-step* state while the module buffers carry the post-step state;
* gradient routing — the token CE reaches only the AR prior branch (targets
  detached), the imitation MSE reaches the encoder only through
  refresh-row codes (the straight-through estimator);
* the three-stream output contract — sampled ``action``, greedy
  ``mean_action`` (deterministic), teacher ``privileged_action``; and
  ``forward_inference`` deliberately emitting no ``mean_action`` so the probe
  drivers run the sampled stream;
* the tracker's ``last_commit`` flag and the experiment wiring (encoder
  future window + expert-contract pinning + event-flag observation).
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    MLPLayerConfig,
    MLPWithConcatConfig,
    ModuleContainerConfig,
)
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.supervised.fsq_masked_mimic_config import (
    FSQCEScheduleConfig,
    FSQIntentConfig,
    FSQMaskedMimicModelConfig,
)
from protomotions.agents.supervised.fsq_masked_mimic_model import (
    FSQ_MEAN_CODES_KEY,
    FSQ_PRIOR_CODES_KEY,
    FSQ_STEPS_LEFT_KEY,
    FSQ_TEACHER_CODES_KEY,
    FSQMaskedMimicModel,
)
from protomotions.envs.control.contact_event_tracker import ContactEventTracker

OBS = 12
ACT = 7
SCALARS = 8
CHUNK = 4
EVENT_KEY = "contact_event_flag"


def _mlp_container(in_keys, out_key, num_out, hidden=16):
    return ModuleContainerConfig(
        in_keys=list(in_keys),
        out_keys=[out_key],
        models=[
            MLPWithConcatConfig(
                in_keys=list(in_keys),
                out_keys=[out_key],
                normalize_obs=False,
                num_out=num_out,
                layers=[MLPLayerConfig(units=hidden, activation="relu")],
            )
        ],
    )


def make_model(chunk_steps=CHUNK, event_flag_key=EVENT_KEY, ce_schedule=None):
    ar = DiscreteAutoregressiveTransformerConfig(
        token_key="fsq_target_tokens_in",
        logits_key=LATENT_LOGITS_KEY,
        generated_tokens_key=LATENT_KEY,
        context_encoder=ModuleContainerConfig(
            in_keys=["transformer_out"],
            out_keys=["fsq_ar_context"],
            models=[
                MLPWithConcatConfig(
                    in_keys=["transformer_out"],
                    out_keys=["fsq_ar_context"],
                    normalize_obs=False,
                    num_out=32,
                    layers=[],
                )
            ],
        ),
        d_model=32,
        num_heads=2,
        num_layers=1,
        ff_size=64,
        dropout=0.0,
        num_tokens=0,
        vocab_size=0,
    )
    config = FSQMaskedMimicModelConfig(
        encoder=_mlp_container(["obs_priv"], "encoder_codes_raw", SCALARS),
        prior=_mlp_container(["obs_dep"], "transformer_out", 32),
        ar_head=ar,
        trunk=_mlp_container(["obs_dep", "vae_latent"], "actor_trunk_out", ACT),
        fsq=FSQIntentConfig(
            num_fsq_levels=5,
            num_fsq_scalars=SCALARS,
            fsq_scalars_per_prior_token=4,
            chunk_steps=chunk_steps,
            event_flag_key=event_flag_key,
            ce_schedule=ce_schedule
            or FSQCEScheduleConfig(start_epoch=0, end_epoch=0, end_ce_coeff=1.0),
        ),
    )
    return FSQMaskedMimicModel(config)


def make_obs(num_envs, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return TensorDict(
        {
            "obs_priv": torch.randn(num_envs, OBS, generator=generator),
            "obs_dep": torch.randn(num_envs, OBS, generator=generator),
            EVENT_KEY: torch.zeros(num_envs, 1),
        },
        batch_size=num_envs,
    )


@pytest.fixture
def model():
    model = make_model()
    model.reset_rollout_context(num_envs=6, device="cpu")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Chunk semantics
# --------------------------------------------------------------------------- #
def test_first_step_refreshes_and_td_keeps_prestep_state(model):
    with torch.no_grad():
        td = model(make_obs(6, seed=1))
    for key in ("action", "mean_action", "privileged_action", TARGET_LATENT_KEY):
        assert key in td.keys()
    # The TensorDict carries what the experience buffer must store: the
    # PRE-step state (zeros here); the module buffers hold the post-step state.
    assert td[FSQ_STEPS_LEFT_KEY].max() == 0
    assert getattr(model, FSQ_STEPS_LEFT_KEY).min() == CHUNK - 1
    assert getattr(model, FSQ_TEACHER_CODES_KEY).abs().sum() > 0


def test_codes_hold_mid_chunk_and_refresh_on_counter_expiry(model):
    with torch.no_grad():
        model(make_obs(6, seed=1))
    held = getattr(model, FSQ_TEACHER_CODES_KEY).clone()
    held_prior = getattr(model, FSQ_PRIOR_CODES_KEY).clone()
    held_mean = getattr(model, FSQ_MEAN_CODES_KEY).clone()

    # Different observations mid-chunk: every stream must hold.
    for step in range(CHUNK - 1):
        with torch.no_grad():
            model(make_obs(6, seed=2 + step))
    assert torch.equal(held, getattr(model, FSQ_TEACHER_CODES_KEY))
    assert torch.equal(held_prior, getattr(model, FSQ_PRIOR_CODES_KEY))
    assert torch.equal(held_mean, getattr(model, FSQ_MEAN_CODES_KEY))
    assert getattr(model, FSQ_STEPS_LEFT_KEY).max() == 0

    # Counter expired: the teacher stream re-encodes (obs differ, so the code
    # must move) and the clock re-arms.
    with torch.no_grad():
        model(make_obs(6, seed=99))
    assert not torch.equal(held, getattr(model, FSQ_TEACHER_CODES_KEY))
    assert getattr(model, FSQ_STEPS_LEFT_KEY).min() == CHUNK - 1


def test_event_flag_refreshes_only_the_flagged_rows(model):
    with torch.no_grad():
        model(make_obs(6, seed=1))
    obs = make_obs(6, seed=3)
    obs[EVENT_KEY][0] = 1.0
    with torch.no_grad():
        model(obs)
    steps = getattr(model, FSQ_STEPS_LEFT_KEY)
    assert steps[0] == CHUNK - 1  # refreshed by the event
    assert steps[1] == CHUNK - 2  # ordinary mid-chunk decrement


def test_event_key_none_disables_event_refresh():
    model = make_model(event_flag_key=None)
    model.reset_rollout_context(num_envs=2, device="cpu")
    model.eval()
    with torch.no_grad():
        model(make_obs(2, seed=1))
    obs = make_obs(2, seed=2)
    obs[EVENT_KEY][:] = 1.0
    with torch.no_grad():
        model(obs)
    assert getattr(model, FSQ_STEPS_LEFT_KEY).max() == CHUNK - 2


def test_reset_rollout_state_rows_force_refresh(model):
    with torch.no_grad():
        model(make_obs(6, seed=1))
    model.reset_rollout_context(env_ids=torch.tensor([2, 4]))
    steps = getattr(model, FSQ_STEPS_LEFT_KEY)
    assert steps[2] == 0 and steps[4] == 0 and steps[0] == CHUNK - 1
    assert getattr(model, FSQ_TEACHER_CODES_KEY)[2].abs().sum() == 0


def test_flush_held_intent_forces_full_refresh(model):
    with torch.no_grad():
        model(make_obs(6, seed=1))
    model.flush_held_intent()
    assert getattr(model, FSQ_STEPS_LEFT_KEY).max() == 0
    held = getattr(model, FSQ_TEACHER_CODES_KEY).clone()
    with torch.no_grad():
        model(make_obs(6, seed=41))
    assert not torch.equal(held, getattr(model, FSQ_TEACHER_CODES_KEY))


# --------------------------------------------------------------------------- #
# Replay contract
# --------------------------------------------------------------------------- #
def test_replay_reproduces_rollout_privileged_action_and_targets(model):
    with torch.no_grad():
        model(make_obs(6, seed=1))
        rollout_td = model(make_obs(6, seed=2))  # a mid-chunk step

    model.train()
    replay_td = rollout_td.clone()
    out = model(replay_td)
    assert torch.allclose(
        out["privileged_action"], rollout_td["privileged_action"], atol=1e-6
    )
    assert torch.equal(out[TARGET_LATENT_KEY], rollout_td[TARGET_LATENT_KEY])
    # Replay computes the teacher-forced logits the CE needs and skips the
    # generated streams entirely.
    assert LATENT_LOGITS_KEY in out.keys()


def test_replay_does_not_touch_module_buffers(model):
    with torch.no_grad():
        model(make_obs(6, seed=1))
        rollout_td = model(make_obs(6, seed=2))
    steps_before = getattr(model, FSQ_STEPS_LEFT_KEY).clone()
    codes_before = getattr(model, FSQ_TEACHER_CODES_KEY).clone()
    model.train()
    model(rollout_td.clone())
    assert torch.equal(steps_before, getattr(model, FSQ_STEPS_LEFT_KEY))
    assert torch.equal(codes_before, getattr(model, FSQ_TEACHER_CODES_KEY))


def test_experience_buffer_keys_cover_outputs_and_chunk_state(model):
    keys = model.experience_buffer_keys()
    for key in (
        "action",
        "mean_action",
        "privileged_action",
        TARGET_LATENT_KEY,
        FSQ_TEACHER_CODES_KEY,
        FSQ_PRIOR_CODES_KEY,
        FSQ_MEAN_CODES_KEY,
        FSQ_STEPS_LEFT_KEY,
    ):
        assert key in keys, key


# --------------------------------------------------------------------------- #
# Gradient routing and loss
# --------------------------------------------------------------------------- #
def test_ce_reaches_ar_branch_only(model):
    with torch.no_grad():
        model(make_obs(6, seed=1))
        rollout_td = model(make_obs(6, seed=2))
    model.train()
    out = model(rollout_td.clone())
    loss, logs = model.compute_model_loss(
        out, current_epoch=10, zero_loss=torch.tensor(0.0)
    )
    assert torch.isfinite(loss)
    loss.backward()
    ar_grads = [p.grad for p in model._ar_head.parameters() if p.grad is not None]
    assert ar_grads and any(g.abs().sum() > 0 for g in ar_grads)
    # Targets are detached: the CE must not reach the encoder.
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in model._encoder.parameters()
    )
    assert "model/fsq_ce_loss" in logs and "model/fsq_token_accuracy" in logs


def test_mse_reaches_encoder_through_refresh_rows_only(model):
    with torch.no_grad():
        first_td = model(make_obs(6, seed=1))       # all rows refresh
        mid_td = model(make_obs(6, seed=2))         # no row refreshes

    model.train()
    out = model(first_td.clone())
    out["privileged_action"].square().mean().backward()
    refresh_grads = sum(
        p.grad.abs().sum() for p in model._encoder.parameters() if p.grad is not None
    )
    assert refresh_grads > 0, "STE gradient missing on refresh rows"

    model.zero_grad()
    out = model(mid_td.clone())
    out["privileged_action"].square().mean().backward()
    hold_grads = [
        p.grad.abs().sum()
        for p in model._encoder.parameters()
        if p.grad is not None
    ]
    assert all(g == 0 for g in hold_grads), (
        "mid-chunk samples decode a stored constant; the encoder must not "
        "receive gradients from them"
    )


def test_ce_coefficient_schedule():
    model = make_model(
        ce_schedule=FSQCEScheduleConfig(
            init_ce_coeff=0.0, end_ce_coeff=1.0, start_epoch=100, end_epoch=300
        )
    )
    assert model._ce_coefficient(0) == 0.0
    assert model._ce_coefficient(200) == pytest.approx(0.5)
    assert model._ce_coefficient(1000) == 1.0


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def test_mean_action_is_greedy_deterministic(model):
    with torch.no_grad():
        a = model(make_obs(6, seed=7))["mean_action"]
    model.reset_rollout_context(env_ids=torch.arange(6))
    with torch.no_grad():
        b = model(make_obs(6, seed=7))["mean_action"]
    assert torch.allclose(a, b)


def test_forward_inference_emits_only_the_sampled_stream(model):
    with torch.no_grad():
        td = model.forward_inference(make_obs(6, seed=5))
    assert "action" in td.keys()
    # Deliberate: the evaluators prefer mean_action; the probe/viz drivers
    # must run the sampled stream, so inference does not offer the greedy one.
    assert "mean_action" not in td.keys()
    assert "privileged_action" not in td.keys()


def test_forward_inference_argmax_mode_is_deterministic():
    model = make_model()
    model.config.fsq.inference_argmax = True
    model.reset_rollout_context(num_envs=4, device="cpu")
    model.eval()
    with torch.no_grad():
        a = model.forward_inference(make_obs(4, seed=9))["action"]
    model.reset_rollout_context(env_ids=torch.arange(4))
    with torch.no_grad():
        b = model.forward_inference(make_obs(4, seed=9))["action"]
    assert torch.allclose(a, b)


def test_tokens_round_trip_through_codes(model):
    tokens = torch.tensor([[0, 1], [624, 313], [17, 400]])
    codes = model._tokens_to_codes(tokens)
    assert torch.equal(model._codes_to_tokens(codes), tokens)


# --------------------------------------------------------------------------- #
# Tracker commit flag
# --------------------------------------------------------------------------- #
UPRIGHT = torch.tensor([0.0, 0.0, 0.0, 1.0])
DT = 1.0 / 30.0


def _tracker(num_envs=2):
    return ContactEventTracker(
        num_envs=num_envs,
        num_pairs=4,
        num_events=3,
        min_dwell_s=0.3,
        time_clip_s=10.0,
        orient_margin=0.15,
        device=torch.device("cpu"),
    )


def _pairs(*on, num_envs=2):
    v = torch.zeros(num_envs, 4, dtype=torch.bool)
    for pair in on:
        v[:, pair] = True
    return v


def test_tracker_last_commit_fires_exactly_on_the_commit_step():
    tracker = _tracker()
    rot = UPRIGHT.expand(2, 4)
    for _ in range(10):
        tracker.update(_pairs(0), rot, DT)
        assert not tracker.last_commit.any()
    # New configuration must persist min_dwell_s (0.3 s = 9 steps at 30 Hz)
    # before it commits; the flag is up for that single step only.
    commits = []
    for _ in range(12):
        tracker.update(_pairs(1), rot, DT)
        commits.append(tracker.last_commit.clone())
    total = torch.stack(commits).sum(dim=0)
    assert (total == 1).all(), "exactly one commit per env expected"
    assert not tracker.last_commit.any() or commits[-1].any() is False


def test_tracker_last_commit_cleared_by_reset():
    tracker = _tracker()
    rot = UPRIGHT.expand(2, 4)
    for _ in range(10):
        tracker.update(_pairs(0), rot, DT)
    for _ in range(12):
        tracker.update(_pairs(1), rot, DT)
        if tracker.last_commit.any():
            break
    assert tracker.last_commit.any()
    tracker.reset(torch.tensor([0, 1]))
    assert not tracker.last_commit.any()


# --------------------------------------------------------------------------- #
# Experiment wiring
# --------------------------------------------------------------------------- #
def _fsq_experiment_args(tmp_path, **overrides):
    from protomotions.tests.test_contact_graph import _toy_graph_payload

    graph_path = tmp_path / "graph.pt"
    torch.save(_toy_graph_payload(), graph_path)
    values = {
        "motion_file": "motions.pt",
        "scenes_file": None,
        "batch_size": 32,
        "training_max_steps": 1024,
        "contact_graph_file": str(graph_path),
        "motion_expert_file": None,
        "expert_model_path": None,
        "expert_model_paths": None,
        "goal_pose_visible_prob": 0.75,
        "goal_contact_visible_prob": 0.85,
        "goal_full_pose_prob": 0.75,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fsq_experiment_wiring(tmp_path):
    from examples.experiments.masked_mimic import (
        contact_graph_fsq_transformer as module,
    )
    from protomotions.tests.test_contact_graph import _StubRobotConfig
    from protomotions.agents.supervised.fsq_masked_mimic_config import (
        FSQMaskedMimicModelConfig,
    )

    args = _fsq_experiment_args(tmp_path)
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    env_cfg = module.env_config(robot_cfg, args)

    # The shared future buffer carries the encoder window; offset 1 leads so
    # positional selection can reproduce the experts' single-step contract.
    ctrl = env_cfg.control_components["contact_graph"]
    assert ctrl.future_steps == module.ENCODER_FUTURE_STEPS
    assert module.ENCODER_FUTURE_STEPS[0] == 1
    # The chunk-refresh trigger observation exists.
    assert module.CONTACT_EVENT_FLAG_KEY in env_cfg.observation_components

    agent_cfg = module.agent_config(robot_cfg, env_cfg, args)
    assert isinstance(agent_cfg.model, FSQMaskedMimicModelConfig)
    assert agent_cfg.model.prior.out_keys == ["transformer_out"]
    assert agent_cfg.model.encoder.out_keys == ["encoder_codes_raw"]
    assert agent_cfg.model.fsq.event_flag_key == module.CONTACT_EVENT_FLAG_KEY
    # Fixed sampling is baked in (standing policy since v3), and the round-2
    # checkpoint lesson too.
    rules = agent_cfg.evaluator.motion_weights_rules
    assert rules.motion_weights_update_success_discount == 1.0
    assert rules.motion_weights_update_failure_discount == 1.0
    assert agent_cfg.save_epoch_checkpoint_every == 500
    # The model config builds a working model.
    model = FSQMaskedMimicModel(agent_cfg.model)
    assert model.tokenization.num_prior_tokens == 4
    assert model.tokenization.prior_token_vocab_size == 5**4


def test_fsq_experiment_rejects_window_not_starting_at_one(tmp_path):
    from examples.experiments.masked_mimic import (
        contact_graph_fsq_transformer as module,
    )
    from protomotions.tests.test_contact_graph import _StubRobotConfig

    args = _fsq_experiment_args(tmp_path, encoder_future_steps=[2, 4, 8])
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    with pytest.raises(ValueError, match="consecutive"):
        module.env_config(robot_cfg, args)
