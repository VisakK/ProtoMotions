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


def make_model(
    chunk_steps=CHUNK,
    event_flag_key=EVENT_KEY,
    ce_schedule=None,
    latent_ema_alpha=1.0,
    ce_refresh_rows_only=False,
    ar_head_lr=None,
    ar_head_weight_decay=None,
    encoder_lr=None,
    chunk_phase_to_trunk=False,
):
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
        trunk=_mlp_container(
            ["obs_dep", "vae_latent"]
            + (["fsq_chunk_phase"] if chunk_phase_to_trunk else []),
            "actor_trunk_out",
            ACT,
        ),
        fsq=FSQIntentConfig(
            num_fsq_levels=5,
            num_fsq_scalars=SCALARS,
            fsq_scalars_per_prior_token=4,
            chunk_steps=chunk_steps,
            event_flag_key=event_flag_key,
            latent_ema_alpha=latent_ema_alpha,
            chunk_phase_to_trunk=chunk_phase_to_trunk,
            ce_refresh_rows_only=ce_refresh_rows_only,
            ce_schedule=ce_schedule
            or FSQCEScheduleConfig(start_epoch=0, end_epoch=0, end_ce_coeff=1.0),
        ),
        ar_head_lr=ar_head_lr,
        ar_head_weight_decay=ar_head_weight_decay,
        encoder_lr=encoder_lr,
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


def test_ce_refresh_rows_only_matches_a_hand_masked_cross_entropy():
    """The masked CE must equal the CE of the refresh rows alone, not a reweight."""
    import torch.nn.functional as F

    model = make_model(ce_refresh_rows_only=True)
    model.reset_rollout_context(num_envs=6, device="cpu")
    model.eval()
    with torch.no_grad():
        model(make_obs(6, seed=1))
        obs = make_obs(6, seed=2)
        obs[EVENT_KEY][:2] = 1.0  # two refresh rows in an otherwise held batch
        rollout_td = model(obs)

    model.train()
    out = model(rollout_td.clone())
    _, logs = model.compute_model_loss(
        out, current_epoch=10, zero_loss=torch.tensor(0.0)
    )

    refresh = out["_fsq_refresh"].reshape(-1).bool()
    assert refresh.sum() == 2 and (~refresh).sum() == 4, "test batch is not mixed"
    logits, target = out[LATENT_LOGITS_KEY], out[TARGET_LATENT_KEY]
    rows = refresh.repeat_interleave(target.shape[-1])
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[rows],
        target.reshape(-1)[rows],
        label_smoothing=model.config.fsq.label_smoothing,
    )
    assert torch.allclose(logs["model/fsq_ce_loss"], expected, atol=1e-6)
    # And the hold rows are still reported, just not trained on.
    assert "model/fsq_ce_loss_hold" in logs


def test_ce_over_all_rows_is_the_default(model):
    """Unset, the CE covers every row -- v6's behaviour, unchanged."""
    import torch.nn.functional as F

    with torch.no_grad():
        model(make_obs(6, seed=1))
        rollout_td = model(make_obs(6, seed=2))
    model.train()
    out = model(rollout_td.clone())
    _, logs = model.compute_model_loss(
        out, current_epoch=10, zero_loss=torch.tensor(0.0)
    )
    logits, target = out[LATENT_LOGITS_KEY], out[TARGET_LATENT_KEY]
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        label_smoothing=model.config.fsq.label_smoothing,
    )
    assert torch.allclose(logs["model/fsq_ce_loss"], expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Latent smoothing (round 7)
# --------------------------------------------------------------------------- #
def test_smoothing_off_declares_no_extra_rollout_state(model):
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_TEACHER_LATENT_KEY,
    )

    assert FSQ_TEACHER_LATENT_KEY not in model.rollout_state_specs()


def test_smoothing_ramps_the_decoded_latent_toward_the_committed_code():
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_TEACHER_LATENT_KEY,
    )

    alpha = 0.5
    model = make_model(latent_ema_alpha=alpha)
    model.reset_rollout_context(num_envs=6, device="cpu")
    model.eval()
    assert FSQ_TEACHER_LATENT_KEY in model.rollout_state_specs()

    with torch.no_grad():
        td = model(make_obs(6, seed=1))  # all rows refresh from a zero latent
    codes = getattr(model, FSQ_TEACHER_CODES_KEY)
    latent = getattr(model, FSQ_TEACHER_LATENT_KEY)
    # First step off a zero-initialised ramp: exactly alpha of the way there.
    assert torch.allclose(latent, alpha * codes, atol=1e-6)

    with torch.no_grad():
        model(make_obs(6, seed=2))  # a hold step: same code, ramp continues
    held = getattr(model, FSQ_TEACHER_CODES_KEY)
    assert torch.equal(held, codes), "the code must not change mid-chunk"
    assert torch.allclose(
        getattr(model, FSQ_TEACHER_LATENT_KEY),
        alpha * codes + (1 - alpha) * (alpha * codes),
        atol=1e-6,
    )
    # The ramp is what the trunk saw, not the raw code.
    assert not torch.allclose(td[FSQ_TEACHER_LATENT_KEY], codes)


def test_smoothing_replays_exactly_and_keeps_the_gradient_routing():
    model = make_model(latent_ema_alpha=0.5)
    model.reset_rollout_context(num_envs=6, device="cpu")
    model.eval()
    with torch.no_grad():
        model(make_obs(6, seed=1))
        rollout_td = model(make_obs(6, seed=2))  # mid-chunk

    model.train()
    out = model(rollout_td.clone())
    assert torch.allclose(
        out["privileged_action"], rollout_td["privileged_action"], atol=1e-6
    )
    # A hold row decodes a ramp between two stored constants, so the encoder
    # still gets nothing from it -- the duty cycle is unchanged by smoothing.
    out["privileged_action"].square().mean().backward()
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in model._encoder.parameters()
    )


def test_flush_clears_the_smoothing_ramp():
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_TEACHER_LATENT_KEY,
    )

    model = make_model(latent_ema_alpha=0.5)
    model.reset_rollout_context(num_envs=6, device="cpu")
    model.eval()
    with torch.no_grad():
        model(make_obs(6, seed=1))
    assert getattr(model, FSQ_TEACHER_LATENT_KEY).abs().sum() > 0
    model.flush_held_intent()
    assert getattr(model, FSQ_TEACHER_LATENT_KEY).abs().sum() == 0
    assert getattr(model, FSQ_STEPS_LEFT_KEY).abs().sum() == 0


# --------------------------------------------------------------------------- #
# Optimizer parameter groups (round 7)
# --------------------------------------------------------------------------- #
def test_no_param_groups_unless_an_override_is_set(model):
    assert model.optimizer_param_groups() is None


def test_param_groups_partition_every_trainable_tensor_exactly_once():
    model = make_model(ar_head_lr=1e-4, ar_head_weight_decay=0.01, encoder_lr=5e-5)
    groups = model.optimizer_param_groups()

    seen = [p for group in groups for p in group["params"]]
    expected = [p for p in model.parameters() if p.requires_grad]
    assert len(seen) == len(expected)
    assert {id(p) for p in seen} == {id(p) for p in expected}

    ar_ids = {id(p) for p in model._ar_head.parameters()}
    encoder_ids = {id(p) for p in model._encoder.parameters()}
    by_lr = {group.get("lr"): group for group in groups}
    assert {id(p) for p in by_lr[1e-4]["params"]} == ar_ids
    assert by_lr[1e-4]["weight_decay"] == 0.01
    assert {id(p) for p in by_lr[5e-5]["params"]} == encoder_ids
    # The remainder (prior + trunk) inherits the optimizer's own lr.
    assert None in by_lr and by_lr[None]["params"]


def test_param_groups_are_accepted_by_a_real_optimizer():
    model = make_model(ar_head_lr=1e-4, ar_head_weight_decay=0.01, encoder_lr=5e-5)
    optimizer = torch.optim.AdamW(
        model.optimizer_param_groups(), lr=2e-5, weight_decay=0.0
    )
    assert [g["lr"] for g in optimizer.param_groups] == [2e-5, 1e-4, 5e-5]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.0, 0.01, 0.0]


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


def _v7_args(tmp_path, **overrides):
    """v6's argument set with the round-7 parser's own defaults applied."""
    from examples.experiments.masked_mimic import contact_graph_fsq_v7 as module

    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)
    values = vars(parser.parse_known_args([])[0])
    values.update(vars(_fsq_experiment_args(tmp_path)))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_v7_experiment_applies_every_tier1_change(tmp_path):
    from examples.experiments.masked_mimic import contact_graph_fsq_v7 as module
    from protomotions.tests.test_contact_graph import _StubRobotConfig

    args = _v7_args(tmp_path)
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    env_cfg = module.env_config(robot_cfg, args)

    # 5. the metric that never existed, at weight 0 alongside the contact one
    for key in ("diag_goal_pose_error", "diag_goal_pose_visible"):
        assert env_cfg.reward_components[key].static_params["weight"] == 0.0
    assert "diag_contact_goal_ground_iou" in env_cfg.reward_components

    agent_cfg = module.agent_config(robot_cfg, env_cfg, args)
    model = agent_cfg.model
    # 1. chunk-boundary smoothing, and 2. the loss that opposes jerk
    assert model.fsq.latent_ema_alpha == 0.5
    assert agent_cfg.action_rate_loss_coeff == 0.5
    # 3. the trunk sees the goal
    assert set(module.GOAL_TRUNK_KEYS).issubset(model.trunk.in_keys)
    head = [m for m in model.trunk.models if "actor_trunk_out" in m.out_keys][0]
    assert module.TRUNK_GOAL_POSES_NORM in head.in_keys
    assert "contact_goal_obs" in head.in_keys
    # ...through its own normalizers, not the encoder's shared key names
    assert "masked_mimic_target_poses_norm" not in head.in_keys
    # 4. CE on refresh rows, own learning rates, and AdamW so decay means
    #    something
    assert model.fsq.ce_refresh_rows_only is True
    assert model.optimizer._target_ == "torch.optim.AdamW"
    assert (model.ar_head_lr, model.ar_head_weight_decay) == (1e-4, 0.01)
    assert model.encoder_lr == 5e-5
    # Inference/DAgger sampling moves to the setting Tier 0 priced
    assert (model.fsq.temperature, model.fsq.top_p) == (0.7, 0.8)
    # Everything else is v6's, inherited rather than restated
    assert agent_cfg.save_epoch_checkpoint_every == 500
    assert model.fsq.chunk_steps == 8
    assert FSQMaskedMimicModel(model).tokenization.num_prior_tokens == 4


def test_v7_ablation_flags_fall_back_to_v6(tmp_path):
    from examples.experiments.masked_mimic import contact_graph_fsq_v7 as module
    from protomotions.tests.test_contact_graph import _StubRobotConfig

    args = _v7_args(
        tmp_path,
        goal_conditioned_trunk=False,
        latent_ema_alpha=1.0,
        ce_refresh_rows_only=False,
        action_rate_loss_coeff=0.0,
    )
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    env_cfg = module.env_config(robot_cfg, args)
    agent_cfg = module.agent_config(robot_cfg, env_cfg, args)

    model = agent_cfg.model
    assert not set(module.GOAL_TRUNK_KEYS) & set(model.trunk.in_keys)
    assert model.fsq.latent_ema_alpha == 1.0
    assert model.fsq.ce_refresh_rows_only is False
    assert agent_cfg.action_rate_loss_coeff == 0.0
    assert FSQMaskedMimicModel(model).rollout_state_specs().keys() == {
        FSQ_TEACHER_CODES_KEY,
        FSQ_PRIOR_CODES_KEY,
        FSQ_MEAN_CODES_KEY,
        FSQ_STEPS_LEFT_KEY,
    }


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


# --------------------------------------------------------------------------- #
# Round 8: chunk phase, selective smoothing, pose-only goal conditioning
# --------------------------------------------------------------------------- #
def test_chunk_phase_counts_steps_since_the_code_was_issued():
    """0 on a refresh, then 1..chunk_steps-1 across the held rows."""
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_CHUNK_PHASE_KEY,
        FSQ_PHASE_INDEX_KEY,
    )

    model = make_model(chunk_phase_to_trunk=True)
    model.reset_rollout_context(num_envs=3, device="cpu")
    model.eval()
    seen = []
    with torch.no_grad():
        for step in range(2 * CHUNK):
            out = model(make_obs(3, seed=step))
            seen.append(out[FSQ_PHASE_INDEX_KEY].clone())
            one_hot = out[FSQ_CHUNK_PHASE_KEY]
            assert one_hot.shape == (3, CHUNK)
            # A one-hot of exactly the index, so the trunk reads the same clock
            # the loss mask does.
            assert torch.equal(one_hot.argmax(dim=-1), out[FSQ_PHASE_INDEX_KEY].long())
            assert torch.equal(one_hot.sum(dim=-1), torch.ones(3))

    phases = torch.stack(seen)[:, 0].tolist()
    assert phases == [float(step % CHUNK) for step in range(2 * CHUNK)]


def test_chunk_phase_is_not_requested_from_the_environment():
    """The model writes it; asking the env for it would fail on the first step."""
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_CHUNK_PHASE_KEY,
    )

    model = make_model(chunk_phase_to_trunk=True)
    assert FSQ_CHUNK_PHASE_KEY in model._trunk.in_keys
    assert FSQ_CHUNK_PHASE_KEY not in model.in_keys
    assert FSQ_CHUNK_PHASE_KEY not in model.get_inference_in_keys()
    # the same exclusion the latent has always had, for the same reason
    assert "vae_latent" not in model.in_keys


def test_chunk_phase_replays_from_the_stored_clock():
    """A replayed batch reproduces the rollout's phase, like its refresh mask."""
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_PHASE_INDEX_KEY,
    )

    model = make_model(chunk_phase_to_trunk=True)
    model.reset_rollout_context(num_envs=4, device="cpu")
    model.eval()
    rollout = []
    with torch.no_grad():
        for step in range(CHUNK + 2):
            obs = make_obs(4, seed=step)
            out = model(obs)
            rollout.append((out.clone(), out[FSQ_PHASE_INDEX_KEY].clone()))

    for stored, phase in rollout:
        replayed = model(stored.clone())
        assert torch.equal(replayed[FSQ_PHASE_INDEX_KEY], phase)


def test_chunk_phase_off_by_default_and_absent_from_the_batch():
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_CHUNK_PHASE_KEY,
    )

    model = make_model()
    model.reset_rollout_context(num_envs=2, device="cpu")
    with torch.no_grad():
        out = model(make_obs(2))
    assert FSQ_CHUNK_PHASE_KEY not in out.keys()


def _v8_args(tmp_path, **overrides):
    from examples.experiments.masked_mimic import contact_graph_fsq_v8 as module

    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)
    values = vars(parser.parse_known_args([])[0])
    values.update(vars(_fsq_experiment_args(tmp_path)))
    values.update(overrides)
    return argparse.Namespace(**values)


def _v8_agent_config(tmp_path, **overrides):
    from examples.experiments.masked_mimic import contact_graph_fsq_v8 as module
    from protomotions.tests.test_contact_graph import _StubRobotConfig

    args = _v8_args(tmp_path, **overrides)
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    env_cfg = module.env_config(robot_cfg, args)
    return module, module.agent_config(robot_cfg, env_cfg, args)


def test_v8_experiment_applies_the_round8_changes(tmp_path):
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_CHUNK_PHASE_KEY,
    )

    module, agent_cfg = _v8_agent_config(tmp_path)
    model = agent_cfg.model

    # 1. selective smoothing: the ramp is gone, the rate loss stays but skips
    #    the one row whose increment spans a code change.
    assert model.fsq.latent_ema_alpha == 1.0
    assert agent_cfg.action_rate_loss_coeff == 0.5
    assert agent_cfg.action_rate_free_steps == 1
    # 2. the trunk gets a clock
    assert model.fsq.chunk_phase_to_trunk is True
    head = [m for m in model.trunk.models if "actor_trunk_out" in m.out_keys][0]
    assert FSQ_CHUNK_PHASE_KEY in model.trunk.in_keys
    assert FSQ_CHUNK_PHASE_KEY in head.in_keys
    # 3. the POSE half of the goal only -- the contact half stays the code's job
    assert module.TRUNK_GOAL_POSES_NORM in head.in_keys
    assert module.TRUNK_GOAL_TIMES_NORM in head.in_keys
    assert "masked_mimic_target_masks" in head.in_keys
    assert "contact_goal_obs" not in head.in_keys
    assert "contact_goal_obs" not in model.trunk.in_keys
    # exactly one normalizer per continuous goal block, not two
    for key in (module.TRUNK_GOAL_POSES_NORM, module.TRUNK_GOAL_TIMES_NORM):
        assert sum(1 for m in model.trunk.models if key in (m.out_keys or [])) == 1
    # v7/v6 inheritance is untouched
    assert model.fsq.ce_refresh_rows_only is True
    assert (model.ar_head_lr, model.encoder_lr) == (1e-4, 5e-5)
    assert (model.fsq.temperature, model.fsq.top_p) == (0.7, 0.8)
    assert agent_cfg.save_epoch_checkpoint_every == 500
    # and it still builds
    assert FSQMaskedMimicModel(model).tokenization.num_prior_tokens == 4


def test_v8_trunk_goal_full_reproduces_v7_and_none_reproduces_v6(tmp_path):
    module, full = _v8_agent_config(tmp_path, trunk_goal="full")
    head = [m for m in full.model.trunk.models if "actor_trunk_out" in m.out_keys][0]
    assert "contact_goal_obs" in head.in_keys

    _, none = _v8_agent_config(tmp_path, trunk_goal="none")
    assert not set(module.GOAL_TRUNK_KEYS) & set(none.model.trunk.in_keys)


def test_v8_ablation_flags_fall_back_to_v7(tmp_path):
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_CHUNK_PHASE_KEY,
    )

    _, agent_cfg = _v8_agent_config(
        tmp_path,
        trunk_goal="full",
        chunk_phase_to_trunk=False,
        action_rate_free_steps=0,
        latent_ema_alpha=0.5,
    )
    assert agent_cfg.model.fsq.latent_ema_alpha == 0.5
    assert agent_cfg.model.fsq.chunk_phase_to_trunk is False
    assert agent_cfg.action_rate_free_steps == 0
    assert FSQ_CHUNK_PHASE_KEY not in agent_cfg.model.trunk.in_keys


# --------------------------------------------------------------------------- #
# Round 9: a one-token code, and a gradient path to the deployable action
# --------------------------------------------------------------------------- #
def test_single_token_code_round_trips_and_forwards():
    """Lever 1: 4 scalars packed 4-per-token is one token over a 625 vocabulary."""
    from protomotions.agents.common.discrete_latent import FSQTokenization

    tok = FSQTokenization(
        num_fsq_levels=5, num_fsq_scalars=4, fsq_scalars_per_prior_token=4
    )
    assert (tok.num_prior_tokens, tok.prior_token_vocab_size) == (1, 625)
    indices = torch.randint(0, 5, (7, 4))
    assert torch.equal(
        tok.prior_tokens_to_fsq_indices(tok.fsq_indices_to_prior_tokens(indices)),
        indices,
    )


def test_model_runs_end_to_end_with_a_single_token_code(monkeypatch):
    """The AR head, the streams and the CE all survive num_tokens == 1."""
    import protomotions.tests.test_fsq_masked_mimic as module

    monkeypatch.setattr(module, "SCALARS", 4)
    model = module.make_model()
    model.reset_rollout_context(num_envs=6, device="cpu")
    model.eval()
    assert model.tokenization.num_prior_tokens == 1
    with torch.no_grad():
        out = model(module.make_obs(6, seed=0))
    assert out[TARGET_LATENT_KEY].shape == (6, 1)
    for key in ("action", "mean_action", "privileged_action"):
        assert out[key].shape == (6, ACT)
    replayed = model(out.clone())
    loss, logs = model.compute_model_loss(
        replayed, current_epoch=1000, zero_loss=torch.zeros(())
    )
    assert torch.isfinite(loss)
    # With one token, full-code match IS token accuracy -- which is the whole
    # point of lever 1 (round 9: the exposure gap closes ~15-20x).
    assert logs["model/fsq_full_match"] == pytest.approx(
        logs["model/fsq_token_accuracy"]
    )
    with torch.no_grad():
        assert model.forward_inference(module.make_obs(6, seed=1))["action"].shape == (
            6,
            ACT,
        )


def test_rollout_publishes_the_latent_the_sampled_stream_decoded():
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_USED_PRIOR_LATENT_KEY,
        PRIOR_ACTION_KEY,
    )

    model = make_model()
    model.reset_rollout_context(num_envs=5, device="cpu")
    model.eval()
    with torch.no_grad():
        out = model(make_obs(5))
    assert FSQ_USED_PRIOR_LATENT_KEY in out.keys()
    # It is what `action` was decoded from, so re-decoding reproduces it.
    with torch.no_grad():
        again = model._decode(out.clone(), out[FSQ_USED_PRIOR_LATENT_KEY])
    assert torch.allclose(again, out["action"], atol=1e-6)
    assert PRIOR_ACTION_KEY not in out.keys()  # rollout path does not need it


def test_replay_redecodes_the_prior_action_with_gradient():
    """Without this the deployable action can carry no loss: the replay path
    skips the generated streams entirely."""
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_USED_PRIOR_LATENT_KEY,
        PRIOR_ACTION_KEY,
    )

    model = make_model()
    model.reset_rollout_context(num_envs=5, device="cpu")
    model.train()
    with torch.no_grad():
        out = model(make_obs(5))
    replayed = model(out.clone())
    assert PRIOR_ACTION_KEY in replayed.keys()
    action = replayed[PRIOR_ACTION_KEY]
    assert action.requires_grad
    assert torch.allclose(action, out["action"], atol=1e-5)
    model.zero_grad()
    action.square().mean().backward()
    trunk_grad = torch.cat(
        [p.grad.reshape(-1) for _, p in model._trunk.named_parameters()
         if p.grad is not None]
    )
    assert trunk_grad.norm() > 0
    # ...and it must NOT reach the encoder: the stored latent is a constant.
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for _, p in model._encoder.named_parameters()
    )


def test_replay_skips_the_prior_action_when_the_latent_was_not_stored():
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_USED_PRIOR_LATENT_KEY,
        PRIOR_ACTION_KEY,
    )

    model = make_model()
    model.reset_rollout_context(num_envs=5, device="cpu")
    model.eval()
    with torch.no_grad():
        out = model(make_obs(5))
    batch = out.clone()
    del batch[FSQ_USED_PRIOR_LATENT_KEY]
    assert PRIOR_ACTION_KEY not in model(batch).keys()


def _v9_agent_config(tmp_path, **overrides):
    from examples.experiments.masked_mimic import contact_graph_fsq_v9 as module
    from protomotions.tests.test_contact_graph import _StubRobotConfig

    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)
    values = vars(parser.parse_known_args([])[0])
    values.update(vars(_fsq_experiment_args(tmp_path)))
    # _fsq_experiment_args carries v6's own FSQ defaults; the round-9 parser
    # defaults are what this experiment is about, so re-apply them.
    values.update({"fsq_scalars": 4, "fsq_scalars_per_token": 4})
    values.update(overrides)
    args = argparse.Namespace(**values)
    robot_cfg = _StubRobotConfig()
    module.configure_robot_and_simulator(robot_cfg, SimpleNamespace(), args)
    env_cfg = module.env_config(robot_cfg, args)
    return module, module.agent_config(robot_cfg, env_cfg, args)


def test_v9_branches_from_v6_and_applies_both_levers(tmp_path):
    module, cfg = _v9_agent_config(tmp_path)
    model = cfg.model
    # Lever 1: one token, 625 vocab.
    assert (model.fsq.num_fsq_scalars, model.fsq.fsq_scalars_per_prior_token) == (4, 4)
    built = FSQMaskedMimicModel(model)
    assert built.tokenization.num_prior_tokens == 1
    assert built.tokenization.prior_token_vocab_size == 625
    # Lever 2.
    assert cfg.dagger_action_loss_coeff == 0.1
    # ...and everything else is v6, NOT v7/v8.
    assert model.fsq.latent_ema_alpha == 1.0
    assert model.fsq.ce_refresh_rows_only is False
    assert model.fsq.chunk_phase_to_trunk is False
    assert (model.fsq.temperature, model.fsq.top_p) == (1.0, 0.9)
    assert (model.ar_head_lr, model.encoder_lr) == (None, None)
    assert model.optimizer._target_ == "torch.optim.Adam"
    assert getattr(cfg, "action_rate_loss_coeff", 0.0) == 0.0
    assert not any(
        key.startswith("masked_mimic_target") or key == "contact_goal_obs"
        for key in model.trunk.in_keys
    )


def test_v9_lever_flags_can_be_reverted_to_v6(tmp_path):
    _, cfg = _v9_agent_config(
        tmp_path, dagger_action_loss_coeff=0.0, fsq_scalars=16,
        fsq_scalars_per_token=4,
    )
    assert cfg.dagger_action_loss_coeff == 0.0
    assert FSQMaskedMimicModel(cfg.model).tokenization.num_prior_tokens == 4


def test_materialize_leaves_no_uninitialized_parameters_with_one_token(monkeypatch):
    """DDP refuses a model with lazy parameters left uninitialized.

    With a single prior token the AR head's token encoder is unreachable from
    the rollout path -- generation never supplies a prefix -- so the base
    one-pass materialization left it lazy. Regression test for that crash.
    """
    import protomotions.tests.test_fsq_masked_mimic as module

    monkeypatch.setattr(module, "SCALARS", 4)
    model = module.make_model()
    assert model.tokenization.num_prior_tokens == 1
    model.reset_rollout_context(num_envs=4, device="cpu")
    model.materialize(module.make_obs(4, seed=0))
    uninitialized = [
        name
        for name, p in model.named_parameters()
        if isinstance(p, torch.nn.parameter.UninitializedParameter)
    ]
    assert uninitialized == []


def test_materialize_still_covers_the_multi_token_case(model):
    model.materialize(make_obs(6, seed=0))
    assert [
        name
        for name, p in model.named_parameters()
        if isinstance(p, torch.nn.parameter.UninitializedParameter)
    ] == []


# --------------------------------------------------------------------------- #
# Round 9 diagnosis §5.1 / T0.6: inference-time intent hysteresis
# --------------------------------------------------------------------------- #
def _single_token_model(hysteresis, chunk_steps=2):
    """One AR token, so the held code's probability is one softmax lookup."""
    model = make_model(chunk_steps=chunk_steps)
    model.config.fsq.num_fsq_scalars = 4
    model.config.fsq.fsq_scalars_per_prior_token = 4
    model = FSQMaskedMimicModel(model.config)
    model.config.fsq.intent_hysteresis = hysteresis
    model.reset_rollout_context(num_envs=4, device="cpu")
    model.eval()
    return model


def test_intent_hysteresis_defaults_to_a_no_op():
    """0.0 must reproduce every trained round exactly."""
    assert FSQIntentConfig().intent_hysteresis == 0.0
    model = _single_token_model(0.0)
    torch.manual_seed(3)
    with torch.no_grad():
        codes = [
            model.forward_inference(make_obs(4, seed=s))["latent_mu"].clone()
            for s in range(6)
        ]
    # chunk_steps=2, so the code is re-drawn every other step; with no
    # hysteresis at least one of those redraws moves it.
    assert any(not torch.equal(codes[0], c) for c in codes[1:])


def test_intent_hysteresis_holds_the_committed_code_across_timer_refreshes():
    """A 12 s hold is 45 nucleus draws; keeping the held code unless the prior
    abandons it is what turns that into one decision."""
    model = _single_token_model(1e-9)
    torch.manual_seed(3)
    with torch.no_grad():
        first = model.forward_inference(make_obs(4, seed=0))["latent_mu"].clone()
        later = [
            model.forward_inference(make_obs(4, seed=s))["latent_mu"].clone()
            for s in range(1, 6)
        ]
    # tau below any achievable probability keeps the held code at every timer
    # refresh, so the intent committed on the first step survives.
    for codes in later:
        assert torch.equal(first, codes)

    # ...and an external goal change still gets through: flush drops the
    # "there is an intent to keep" flag, so the next refresh re-draws.
    model.flush_held_intent()
    torch.manual_seed(11)
    with torch.no_grad():
        after = model.forward_inference(make_obs(4, seed=42))["latent_mu"].clone()
    assert not torch.equal(first, after)


def test_intent_hysteresis_never_suppresses_an_event_refresh():
    """A committed contact change is a real reason to reconsider the intent."""
    model = _single_token_model(1e-9)
    torch.manual_seed(5)
    with torch.no_grad():
        first = model.forward_inference(make_obs(4, seed=0))["latent_mu"].clone()
        obs = make_obs(4, seed=7)
        obs[EVENT_KEY] = torch.ones(4, 1)
        evented = model.forward_inference(obs)["latent_mu"].clone()
    assert not torch.equal(first, evented)
