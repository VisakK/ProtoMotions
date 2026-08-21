# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for supervised rollout dispatch without live environments."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.supervision import SupervisionLossConfig
from protomotions.agents.base_agent import agent as base_agent_module
from protomotions.agents.supervised import agent as supervised_agent_module
from protomotions.agents.supervised.agent import (
    SupervisedAgent,
    compute_prior_rollout_mask,
)
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.utils.metering import TensorAverageMeterDict


class _StudentModel(nn.Module):
    out_keys = ["action", "target_tokens"]

    def __init__(self):
        super().__init__()
        self.call_modes = []

    def forward(self, obs_td):
        self.call_modes.append(self.training)
        obs_td["action"] = obs_td["obs"] + 1.0
        obs_td["target_tokens"] = obs_td["obs"].long() + 2
        return obs_td

    def collect_expert_rollout(self, obs_td):
        self.call_modes.append(self.training)
        obs_td["action"] = obs_td["obs"] + 10.0
        obs_td["mean_action"] = obs_td["action"]
        obs_td["target_tokens"] = obs_td["obs"].long() + 20
        return obs_td


class _PrivilegedStudentModel(_StudentModel):
    out_keys = ["action", "privileged_action", "target_tokens"]

    def forward(self, obs_td):
        self.call_modes.append(self.training)
        obs_td["action"] = obs_td["obs"] + 1.0
        obs_td["privileged_action"] = obs_td["obs"] + 50.0
        obs_td["target_tokens"] = obs_td["obs"].long() + 2
        return obs_td


class _ExternalExpert:
    in_keys = ["expert_obs"]

    def __call__(self, obs_td):
        obs_td["mean_action"] = obs_td["expert_obs"] + 100.0
        return obs_td


class _ActionOnlyExternalActor:
    in_keys = ["actor_obs"]

    def __init__(self):
        self.calls = []

    def __call__(self, obs_td):
        self.calls.append(list(obs_td.keys()))
        obs_td["action"] = obs_td["actor_obs"] + 200.0
        return obs_td


class _ActorCriticExternalExpert:
    in_keys = ["actor_obs", "critic_obs"]

    def __init__(self):
        self._actor = _ActionOnlyExternalActor()

    def __call__(self, obs_td):
        raise AssertionError("external expert collection should call the actor only")


class _CreatedActorCriticExpertActor(nn.Module):
    in_keys = ["obs"]

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.forward_training_modes = []

    def forward(self, tensordict):
        self.forward_training_modes.append(self.training)
        tensordict["mean_action"] = self.linear(tensordict["obs"])
        return tensordict


class _CreatedActorCriticExpertModel(nn.Module):
    in_keys = ["obs", "critic_obs"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._actor = _CreatedActorCriticExpertActor(config)
        self._critic = nn.Linear(1, 1, bias=False)
        self.obs_norm = supervised_agent_module.RunningMeanStd(
            fabric=None,
            shape=(1,),
            device="cpu",
        )
        self.reset_calls = []

    def reset_rollout_context(self, num_envs, device):
        self.reset_calls.append((num_envs, device))

    def forward(self, tensordict):
        raise AssertionError("external expert materialization should call actor only")


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(2.0)

    def optimization_module(self):
        return self


class _ExperienceBufferRecorder:
    def __init__(self):
        self.data = {}
        self.registered = []

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()

    def register_key(self, key, shape=(), dtype=None):
        self.registered.append((key, shape, dtype))


class _OptimizerRecorder:
    def __init__(self):
        self.zero_grad_calls = []
        self.steps = 0
        self.loaded_state = None

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls.append(set_to_none)

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.loaded_state = state


class _FabricRecorder:
    def __init__(self):
        self.backward_losses = []

    def backward(self, loss):
        self.backward_losses.append(loss.detach().clone())
        loss.backward()


class _CreatedStudentModel(nn.Module):
    skip_default_weight_init = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(1, 1, bias=False)


class _CreatedExpertModel(nn.Module):
    in_keys = ["obs"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(1, 1, bias=False)
        self.obs_norm = supervised_agent_module.RunningMeanStd(
            fabric=None,
            shape=(1,),
            device="cpu",
        )
        self.reset_calls = []
        self.forward_training_modes = []

    def reset_rollout_context(self, num_envs, device):
        self.reset_calls.append((num_envs, device))

    def forward(self, tensordict):
        self.forward_training_modes.append(self.training)
        tensordict["mean_action"] = self.linear(tensordict["obs"])
        return tensordict

    def materialize_from_state_dict(self, state_dict):
        pass


class _CheckpointLinear(nn.Linear):
    def materialize_from_state_dict(self, state_dict):
        pass


def _agent(rollout_actor, expert_model=None):
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        rollout_actor=rollout_actor,
        loss=SimpleNamespace(target_key="target_tokens"),
    )
    agent.device = torch.device("cpu")
    agent.model = _StudentModel()
    agent.expert_model = expert_model
    return agent


def _obs_td():
    return TensorDict(
        {
            "obs": torch.tensor([[1.0], [2.0]]),
            "expert_expert_obs": torch.tensor([[5.0], [6.0]]),
        },
        batch_size=2,
    )


def test_create_model_without_expert_marks_expert_model_none(monkeypatch):
    monkeypatch.setattr(
        supervised_agent_module,
        "get_class",
        lambda target: _CreatedStudentModel,
    )

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(_target_="student.Target"),
        expert_model_path=None,
    )

    model = SupervisedAgent.create_model(agent)

    assert isinstance(model, _CreatedStudentModel)
    assert agent.expert_model is None


def test_create_model_loads_external_expert_and_freezes_it(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "expert.pt"
    torch.save(
        {"model": _CreatedExpertModel(SimpleNamespace()).state_dict()},
        checkpoint_path,
    )

    def get_class(target):
        if target == "student.Target":
            return _CreatedStudentModel
        if target == "expert.Target":
            return _CreatedExpertModel
        raise AssertionError(f"Unexpected target: {target}")

    monkeypatch.setattr(supervised_agent_module, "get_class", get_class)
    monkeypatch.setattr(
        supervised_agent_module,
        "load_resolved_configs_from_checkpoint",
        lambda path: {
            "env": {"name": "expert-env"},
            "agent": SimpleNamespace(
                model=SimpleNamespace(_target_="expert.Target")
            ),
        },
    )

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(_target_="student.Target"),
        expert_model_path=str(checkpoint_path),
    )
    agent.device = torch.device("cpu")
    agent.num_envs = 3
    agent.fabric = object()
    agent.env = SimpleNamespace(
        get_obs=lambda: {"expert_obs": torch.ones(3, 1)}
    )
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(obs, batch_size=3)

    model = SupervisedAgent.create_model(agent)

    assert isinstance(model, _CreatedStudentModel)
    assert agent.expert_env_config == {"name": "expert-env"}
    assert isinstance(agent.expert_model, _CreatedExpertModel)
    assert agent.expert_model.reset_calls == [(3, torch.device("cpu"))]
    assert agent.expert_model.obs_norm.fabric is agent.fabric
    assert agent.expert_model.forward_training_modes == [False]
    assert agent.expert_model.training is False
    assert all(not param.requires_grad for param in agent.expert_model.parameters())


def test_create_model_materializes_external_expert_actor_only(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "expert_actor_critic.pt"
    torch.save(
        {"model": _CreatedActorCriticExpertModel(SimpleNamespace()).state_dict()},
        checkpoint_path,
    )

    def get_class(target):
        if target == "student.Target":
            return _CreatedStudentModel
        if target == "expert.ActorCritic":
            return _CreatedActorCriticExpertModel
        raise AssertionError(f"Unexpected target: {target}")

    monkeypatch.setattr(supervised_agent_module, "get_class", get_class)
    monkeypatch.setattr(
        supervised_agent_module,
        "load_resolved_configs_from_checkpoint",
        lambda path: {
            "env": {"name": "expert-env"},
            "agent": SimpleNamespace(
                model=SimpleNamespace(
                    _target_="expert.ActorCritic",
                    actor=SimpleNamespace(in_keys=["obs"]),
                )
            ),
        },
    )

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(_target_="student.Target"),
        expert_model_path=str(checkpoint_path),
    )
    agent.device = torch.device("cpu")
    agent.num_envs = 3
    agent.fabric = object()
    agent.env = SimpleNamespace(get_obs=lambda: {"expert_obs": torch.ones(3, 1)})
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(obs, batch_size=3)

    SupervisedAgent.create_model(agent)

    assert agent.expert_actor is agent.expert_model._actor
    assert agent.expert_model.reset_calls == [(3, torch.device("cpu"))]
    assert agent.expert_model._actor.forward_training_modes == [False]
    assert agent.expert_model.training is False
    assert all(not param.requires_grad for param in agent.expert_model.parameters())


def test_build_expert_obs_td_requires_prefixed_expert_observations():
    agent = _agent(RolloutActor.STUDENT)
    obs_td = _obs_td()

    expert_td = SupervisedAgent._build_expert_obs_td(
        agent,
        obs_td,
        expert_in_keys=["expert_obs"],
    )

    assert torch.equal(expert_td["expert_obs"], torch.tensor([[5.0], [6.0]]))

    with pytest.raises(KeyError) as error:
        SupervisedAgent._build_expert_obs_td(agent, obs_td, expert_in_keys=["missing"])

    assert "expert_missing" in str(error.value)


def test_collect_rollout_output_uses_student_or_model_owned_expert_by_enum():
    obs_td = _obs_td()

    student_agent = _agent(RolloutActor.STUDENT)
    student_out = SupervisedAgent._collect_rollout_output(student_agent, obs_td.clone())
    assert torch.equal(student_out["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(student_out["target_tokens"], torch.tensor([[3], [4]]))

    expert_agent = _agent(RolloutActor.EXPERT)
    expert_out = SupervisedAgent._collect_rollout_output(expert_agent, obs_td.clone())
    assert torch.equal(expert_out["action"], torch.tensor([[11.0], [12.0]]))
    assert torch.equal(expert_out["mean_action"], torch.tensor([[11.0], [12.0]]))
    assert torch.equal(expert_out["target_tokens"], torch.tensor([[21], [22]]))


def test_external_expert_labels_student_rollouts_and_can_drive_actions():
    agent = _agent(RolloutActor.EXPERT, expert_model=_ExternalExpert())

    output = SupervisedAgent._collect_rollout_output(agent, _obs_td())

    assert torch.equal(output["expert_actions"], torch.tensor([[105.0], [106.0]]))
    assert torch.equal(output["action"], output["expert_actions"])
    assert torch.equal(output["mean_action"], output["expert_actions"])
    assert torch.equal(output["target_tokens"], torch.tensor([[3], [4]]))


def test_external_expert_collection_uses_actor_only_inputs_and_action_fallback():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ActorCriticExternalExpert())
    obs_td = TensorDict(
        {
            "obs": torch.tensor([[1.0], [2.0]]),
            "expert_actor_obs": torch.tensor([[5.0], [6.0]]),
        },
        batch_size=2,
    )

    action = SupervisedAgent._collect_external_expert_action(agent, obs_td)

    assert torch.equal(action, torch.tensor([[205.0], [206.0]]))
    assert agent.expert_model._actor.calls == [["actor_obs"]]


def test_collect_rollout_step_records_configured_supervision_target_key():
    agent = _agent(RolloutActor.STUDENT)
    agent.model_output_keys = ["action"]
    agent.experience_buffer = _ExperienceBufferRecorder()

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=7)

    assert torch.equal(output["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("action", 7)],
        torch.tensor([[2.0], [3.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("target_tokens", 7)],
        torch.tensor([[3], [4]]),
    )


def test_registers_external_expert_actions_and_model_generated_target_key():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ExternalExpert())
    agent.env = SimpleNamespace(
        robot_config=SimpleNamespace(number_of_actions=3),
    )
    agent.experience_buffer = _ExperienceBufferRecorder()

    SupervisedAgent.register_algorithm_experience_buffer_keys(agent)
    SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(agent, _obs_td())

    assert agent.experience_buffer.registered == [
        ("expert_actions", (3,), None),
        ("target_tokens", torch.Size([1]), torch.int64),
    ]
    assert agent.model.call_modes == [False]


def test_register_target_key_short_circuits_existing_buffer_key():
    agent = _agent(RolloutActor.STUDENT)
    agent.experience_buffer = SimpleNamespace(target_tokens=torch.zeros(1))

    SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(agent, _obs_td())


def test_register_target_key_from_observation_when_buffer_key_is_missing():
    agent = _agent(RolloutActor.STUDENT)
    agent.config.loss.target_key = "target_tokens"
    agent.experience_buffer = _ExperienceBufferRecorder()
    obs_td = _obs_td()
    obs_td["target_tokens"] = torch.tensor([[7], [8]])

    SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(agent, obs_td)

    assert agent.experience_buffer.registered == [
        ("target_tokens", torch.Size([1]), torch.int64),
    ]


def test_register_target_key_reports_missing_model_output():
    agent = _agent(RolloutActor.STUDENT)
    agent.config.loss.target_key = "missing_target"
    agent.experience_buffer = _ExperienceBufferRecorder()

    with pytest.raises(KeyError, match="missing_target"):
        SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(agent, _obs_td())
    assert agent.model.call_modes == [False]


def test_collect_rollout_step_prefers_privileged_action_for_student_training():
    agent = _agent(RolloutActor.STUDENT)
    agent.model = _PrivilegedStudentModel()
    agent.model_output_keys = agent.model.out_keys
    agent.experience_buffer = _ExperienceBufferRecorder()

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=9)

    assert torch.equal(output["action"], torch.tensor([[51.0], [52.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("privileged_action", 9)],
        torch.tensor([[51.0], [52.0]]),
    )


def test_collect_rollout_step_stores_external_expert_actions_when_not_target_key():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ExternalExpert())
    agent.model_output_keys = ["action", "target_tokens"]
    agent.experience_buffer = _ExperienceBufferRecorder()

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=4)

    assert torch.equal(output["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("expert_actions", 4)],
        torch.tensor([[105.0], [106.0]]),
    )


def test_collect_rollout_step_reports_missing_required_target_key():
    agent = _agent(RolloutActor.STUDENT)
    agent.config.loss.target_key = "missing_target"
    agent.model_output_keys = ["action"]
    agent.experience_buffer = _ExperienceBufferRecorder()

    with pytest.raises(KeyError, match="missing_target"):
        SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=0)


def test_create_optimizers_prepares_model_with_fabric_setup(monkeypatch):
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(optimizer=SimpleNamespace(name="sgd")),
    )
    optimizer = _OptimizerRecorder()
    monkeypatch.setattr(
        supervised_agent_module,
        "instantiate_optimizer",
        lambda config, module: optimizer,
    )
    setup_calls = []
    agent._setup_model_optimizer = lambda module, optimizer: (
        setup_calls.append((module, optimizer))
        or (SimpleNamespace(module=module), optimizer)
    )
    model = _LinearModel()

    SupervisedAgent.create_optimizers(agent, model)

    assert agent.training_model.module is model
    assert setup_calls == [(model, optimizer)]
    assert agent.supervised_optimizer is optimizer


def test_perform_optimization_step_steps_optimizer_and_clips(monkeypatch):
    clipped = []

    def fake_clip(config, fabric, model, optimizer, model_name):
        clipped.append((model_name, model, optimizer))
        return {"model/grad_norm": torch.tensor(3.0)}

    monkeypatch.setattr(base_agent_module, "handle_model_grad_clipping", fake_clip)

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace()
    agent.fabric = _FabricRecorder()
    agent.training_model = _LinearModel()
    agent.supervised_optimizer = _OptimizerRecorder()
    agent.supervised_step = lambda batch_dict: (
        agent.training_model.linear.weight.sum(),
        {"supervised/loss": torch.tensor(1.0)},
    )

    log_dict = SupervisedAgent.perform_optimization_step(
        agent,
        {"action": torch.zeros(1, 1)},
        batch_idx=0,
    )

    assert agent.supervised_optimizer.zero_grad_calls == [True]
    assert agent.supervised_optimizer.steps == 1
    assert [name for name, _, _ in clipped] == ["model"]
    assert clipped[0][1] is agent.training_model
    assert clipped[0][2] is agent.supervised_optimizer
    assert torch.equal(log_dict["model/grad_norm"], torch.tensor(3.0))


def test_default_extra_loss_returns_zero_on_agent_device():
    agent = object.__new__(SupervisedAgent)
    agent.device = torch.device("cpu")

    loss, log_dict = SupervisedAgent.calculate_extra_loss(
        agent,
        batch_dict={},
        actions=torch.zeros(2, 1),
    )

    assert torch.equal(loss, torch.zeros(()))
    assert log_dict == {}


def test_training_load_restores_model_weights_and_optimizer():
    agent = object.__new__(SupervisedAgent)
    model = _CheckpointLinear(1, 1, bias=False)
    model.weight.data.zero_()

    agent.model = model
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent.supervised_optimizer = _OptimizerRecorder()
    agent.device = torch.device("cpu")

    SupervisedAgent.load_parameters(
        agent,
        {
            "model": {"weight": torch.tensor([[2.0]])},
            "epoch": 4,
            "supervised_optimizer": {"lr": 0.1},
        },
        load_training_state=True,
    )

    assert agent.current_epoch == 4
    assert torch.equal(agent.model.weight, torch.tensor([[2.0]]))
    assert agent.supervised_optimizer.loaded_state == {"lr": 0.1}


def _prior_mixture_agent(
    fraction,
    start_epoch=0,
    ramp_epochs=0,
    current_epoch=0,
    num_envs=2,
):
    agent = _agent(RolloutActor.STUDENT)
    agent.model = _PrivilegedStudentModel()
    agent.model_output_keys = agent.model.out_keys
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent.config.prior_rollout_fraction = fraction
    agent.config.prior_rollout_start_epoch = start_epoch
    agent.config.prior_rollout_ramp_epochs = ramp_epochs
    agent.current_epoch = current_epoch
    agent.num_envs = num_envs
    return agent


def test_compute_prior_rollout_mask_inactive_cases():
    device = torch.device("cpu")
    assert (
        compute_prior_rollout_mask(0.0, 0, 0, current_epoch=10, num_envs=8, device=device)
        is None
    )
    assert (
        compute_prior_rollout_mask(0.5, 5, 0, current_epoch=4, num_envs=8, device=device)
        is None
    )
    # Rounds to zero envs.
    assert (
        compute_prior_rollout_mask(0.05, 0, 0, current_epoch=0, num_envs=2, device=device)
        is None
    )


def test_compute_prior_rollout_mask_block_and_ramp():
    device = torch.device("cpu")

    mask = compute_prior_rollout_mask(
        0.5, 0, 0, current_epoch=0, num_envs=8, device=device
    )
    assert mask.dtype == torch.bool and mask.shape == (8,)
    assert mask[:4].all() and not mask[4:].any()

    # Linear ramp: fraction 0.5 over 10 epochs, 5 epochs past start -> 0.25.
    ramped = compute_prior_rollout_mask(
        0.5, 100, 10, current_epoch=104, num_envs=8, device=device
    )
    assert ramped.sum().item() == 2

    # Past the ramp the full fraction applies and stays there.
    full = compute_prior_rollout_mask(
        0.5, 100, 10, current_epoch=500, num_envs=8, device=device
    )
    assert full.sum().item() == 4


def test_collect_rollout_step_mixes_prior_action_for_masked_envs():
    agent = _prior_mixture_agent(fraction=0.5)

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=0)

    # Env 0 is prior-driven (obs + 1), env 1 keeps the privileged action
    # (obs + 50); the buffer still stores the unmixed privileged action.
    assert torch.equal(output["action"], torch.tensor([[2.0], [52.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("privileged_action", 0)],
        torch.tensor([[51.0], [52.0]]),
    )


def test_collect_rollout_step_prior_mixture_respects_start_epoch():
    agent = _prior_mixture_agent(fraction=0.5, start_epoch=5, current_epoch=4)

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=0)

    assert torch.equal(output["action"], torch.tensor([[51.0], [52.0]]))


def test_record_prior_rollout_diagnostics_splits_groups():
    agent = object.__new__(SupervisedAgent)
    agent.device = torch.device("cpu")
    agent.num_envs = 4
    agent.episode_env_tensors = TensorAverageMeterDict(device=agent.device)
    agent.current_lengths = torch.tensor([9.0, 3.0, 5.0, 7.0])

    prior_mask = torch.tensor([True, True, False, False])
    dones = torch.tensor([1.0, 0.0, 0.0, 1.0])
    extras = {
        "raw_r/gt_rew": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "terminate": torch.tensor([0.0, 0.0, 1.0, 1.0]),
    }

    SupervisedAgent._record_prior_rollout_diagnostics(agent, prior_mask, dones, extras)

    diag = agent.episode_env_tensors.mean_and_clear()
    assert diag["prior_rollout/frac"] == pytest.approx(0.5)
    assert diag["prior_rollout/gt_rew"] == pytest.approx(0.5)
    assert diag["prior_rollout/gt_rew_ctrl"] == pytest.approx(0.5)
    assert diag["prior_rollout/terminate"] == pytest.approx(0.0)
    assert diag["prior_rollout/terminate_ctrl"] == pytest.approx(1.0)

    # Only env 0 is a prior-driven done; its recorded length matches the +1
    # the base record step applies afterwards.
    lengths = agent._prior_rollout_length_meter.mean_and_clear()
    assert lengths["episode_length"] == pytest.approx(10.0)


def test_record_rollout_step_without_mixture_only_delegates(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_agent_module.BaseAgent,
        "record_rollout_step",
        lambda self, *args: calls.append(args),
    )

    agent = _agent(RolloutActor.STUDENT)
    args = (
        TensorDict({}, batch_size=2),
        torch.zeros(2, 1),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([], dtype=torch.long),
        {},
        0,
    )

    SupervisedAgent.record_rollout_step(agent, *args)

    assert len(calls) == 1
    assert not hasattr(agent, "_prior_rollout_length_meter")


def test_post_epoch_logging_injects_prior_episode_length(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        base_agent_module.BaseAgent,
        "post_epoch_logging",
        lambda self, training_log_dict: seen.update(training_log_dict),
    )

    agent = object.__new__(SupervisedAgent)
    agent.device = torch.device("cpu")
    agent._prior_rollout_length_meter = TensorAverageMeterDict(device=agent.device)
    agent._prior_rollout_length_meter.add({"episode_length": torch.tensor([250.0])})

    SupervisedAgent.post_epoch_logging(agent, {"epoch": 3})

    assert seen["epoch"] == 3
    assert seen["info/prior_rollout_episode_length"] == pytest.approx(250.0)


def test_training_load_accepts_previous_maskedmimic_optimizer_key():
    agent = object.__new__(SupervisedAgent)
    model = _CheckpointLinear(1, 1, bias=False)
    model.weight.data.zero_()

    agent.model = model
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent.supervised_optimizer = _OptimizerRecorder()
    agent.device = torch.device("cpu")

    SupervisedAgent.load_parameters(
        agent,
        {
            "model": {"weight": torch.tensor([[2.0]])},
            "epoch": 4,
            "maskedmimic_optimizer": {"lr": 0.2},
        },
        load_training_state=True,
    )

    assert agent.supervised_optimizer.loaded_state == {"lr": 0.2}
