# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervised rollout imitation agent.

This agent collects rollouts with the student policy, labels those states with
an expert policy, and optimizes a configured supervision loss. Algorithms such
as MaskedMimic are experiment/model configurations of this generic loop.
"""

import torch
from torch import Tensor
from tensordict import TensorDict
import logging

from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint
from protomotions.utils.hydra_replacement import get_class
from typing import Tuple, Dict, Optional
from pathlib import Path

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.common.common import weight_init_trainable
from protomotions.agents.common.supervision import compute_supervision_loss
from protomotions.agents.optimizer.factory import instantiate_optimizer
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.supervised.expert_utils import get_expert_actor_in_keys
from protomotions.agents.utils.metering import TensorAverageMeterDict
from protomotions.agents.utils.normalization import RunningMeanStd

log = logging.getLogger(__name__)

# Steps since the intent code was issued, published into the batch by
# ``FSQMaskedMimicModel`` (``FSQ_PHASE_INDEX_KEY``). Named here rather than
# imported so the generic supervised agent does not depend on one model module;
# a test asserts the two spellings agree.
FSQ_PHASE_INDEX_KEY = "_fsq_chunk_phase"
# Published by ``FSQMaskedMimicModel`` (same names there; a test asserts the
# spellings agree). The latent the sampled stream decoded, and the action the
# model re-decodes from it on replay.
FSQ_USED_PRIOR_LATENT_KEY = "fsq_used_prior_latent"
PRIOR_ACTION_KEY = "prior_action"
# Per-row flag: was this row driven by the deployable prior (the DAgger block)?
PRIOR_ROLLOUT_MASK_KEY = "prior_rollout_mask"

# The expert's action one control step ago, stored alongside its current one.
# The rate term needs both endpoints of the teacher's own increment; the
# `previous_actions` observation is the *applied* action, so on its own it
# cancels out of the difference and leaves the plain imitation MSE.
PREVIOUS_EXPERT_ACTIONS_KEY = "previous_expert_actions"


# Mirrors ``fsq_masked_mimic_model.LADDER_PRED_KEY``. Duplicated as a literal
# rather than imported because the model module imports agent-side helpers, and
# a string constant is not worth a circular import.
LADDER_PRED_KEY = "ladder_pred"
VAE_LATENT_KEY = "vae_latent"

# How often the (no-gradient) code-ablation diagnostic runs. It costs one extra
# trunk forward, so it is a periodic probe rather than a per-batch metric.
LADDER_ABLATION_EVERY = 25


def ladder_target_key(offset: int) -> str:
    """Buffer key holding the expert action ``offset`` control steps ahead."""
    return f"ladder_target_h{offset}"


def ladder_valid_key(offset: int) -> str:
    """Buffer key marking rows whose horizon stays inside one episode."""
    return f"ladder_valid_h{offset}"


def compute_prior_rollout_mask(
    fraction: float,
    start_epoch: int,
    ramp_epochs: int,
    current_epoch: int,
    num_envs: int,
    device: torch.device,
) -> Optional[Tensor]:
    """Boolean ``[num_envs]`` mask of envs stepped with the prior's action.

    The mask is a fixed leading block so an env stays prior-driven across an
    entire episode and its trajectory is genuinely prior-induced, and it is a
    pure function of the epoch so every call within one epoch agrees. Returns
    ``None`` when the mixture is inactive.
    """
    if fraction <= 0.0 or current_epoch < start_epoch:
        return None
    if ramp_epochs > 0:
        progress = min(1.0, (current_epoch - start_epoch + 1) / ramp_epochs)
        fraction = fraction * progress
    num_prior = int(round(fraction * num_envs))
    if num_prior <= 0:
        return None
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask[:num_prior] = True
    return mask


class SupervisedAgent(BaseAgent):
    """Student/expert rollout agent for supervised distillation.

    The agent collects TensorDict rollouts, writes configured model outputs into
    the experience buffer, and optimizes ``SupervisionLossConfig``. Models and
    experiment files define which keys are predictions and labels.
    """

    model: BaseModel
    # Set only when the teacher-rate term is on (see PREVIOUS_EXPERT_ACTIONS_KEY).
    _previous_expert_actions: Optional[Tensor] = None

    def create_model(self):
        model_cls = get_class(self.config.model._target_)
        model: BaseModel = model_cls(config=self.config.model)
        if not getattr(model, "skip_default_weight_init", False):
            model.apply(weight_init_trainable)

        # Optionally load a pre-trained expert model if provided.
        # Note: Expert observation components are loaded in the experiment file
        # and prefixed with "expert_" for use during distillation training.
        expert_model_path = self.config.expert_model_path
        if expert_model_path is not None:
            expert_model, expert_actor, expert_actor_in_keys = self._build_external_expert(
                expert_model_path
            )
            self.expert_model = expert_model
            self.expert_actor = expert_actor
            self.expert_actor_in_keys = expert_actor_in_keys
        else:
            self.expert_model = None
            self.expert_actor = None
            self.expert_actor_in_keys = []

        return model

    def _build_external_expert(self, expert_model_path):
        """Load one frozen external expert actor from a checkpoint.

        Factored out of :meth:`create_model` so a subclass can hold more than one
        expert (see :mod:`protomotions.agents.supervised.multi_expert`). Returns
        ``(expert_model, expert_actor, expert_actor_in_keys)`` with the model in
        eval mode and every parameter frozen.
        """
        log.info(f"Loading expert model from: {expert_model_path}")

        checkpoint_path = Path(expert_model_path)
        assert (
            checkpoint_path.exists()
        ), f"Could not find expert model at {checkpoint_path}"

        resolved_configs = load_resolved_configs_from_checkpoint(checkpoint_path)

        self.expert_env_config = resolved_configs["env"]
        expert_agent_config: PPOAgentConfig = resolved_configs["agent"]

        # Create the expert model
        ExpertModelConfig = get_class(expert_agent_config.model._target_)
        expert_model: BaseModel = ExpertModelConfig(config=expert_agent_config.model)

        # Move model to device BEFORE materializing lazy modules
        expert_model = expert_model.to(self.device)
        expert_model.reset_rollout_context(
            num_envs=self.num_envs,
            device=self.device,
        )

        # Once model is created, we pass fabric to the RunningMeanStd modules.
        # This allows the modules to internally handle distributed aggregation of normalization moments.
        def pass_fabric_to_running_mean_std(module):
            if isinstance(module, RunningMeanStd):
                module.fabric = self.fabric

        expert_model.apply(pass_fabric_to_running_mean_std)

        expert_actor = self._external_expert_module_from(expert_model)
        expert_actor_in_keys = get_expert_actor_in_keys(expert_agent_config)
        if not expert_actor_in_keys:
            expert_actor_in_keys = list(getattr(expert_actor, "in_keys", []))

        log.info("Materializing expert actor lazy modules...")
        # External experts are frozen inference modules. Only the actor is
        # needed to label actions; materializing the full actor-critic model
        # can require critic-only observations that the distillation env does
        # not provide.
        expert_model.eval()
        with torch.no_grad():
            dummy_obs = self.env.get_obs()
            # Build expert obs tensordict (strips "expert_" prefix from keys)
            dummy_obs_td = self.obs_dict_to_tensordict(dummy_obs)
            dummy_expert_obs_td = self._build_expert_obs_td(
                dummy_obs_td, expert_actor_in_keys
            )
            _ = expert_actor(dummy_expert_obs_td)

        # Load weights before any distributed wrapper changes module keys.
        pre_trained_expert = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        self._load_external_expert_state(
            expert_model,
            pre_trained_expert["model"],
        )
        for param in expert_model.parameters():
            param.requires_grad = False

        # Keep the external expert as a plain frozen module. The trainable
        # student is wrapped by create_optimizers(); the expert only labels
        # rollouts and does not need gradient synchronization.
        expert_model.eval()
        return expert_model, expert_actor, expert_actor_in_keys

    @staticmethod
    def _external_expert_module_from(expert_model):
        wrapped_module = getattr(expert_model, "module", None)
        if wrapped_module is not None and (
            callable(wrapped_module)
            or hasattr(wrapped_module, "_actor")
            or hasattr(wrapped_module, "actor")
        ):
            expert_model = wrapped_module
        return getattr(
            expert_model,
            "_actor",
            getattr(expert_model, "actor", expert_model),
        )

    def _external_expert_module(self):
        expert_actor = getattr(self, "expert_actor", None)
        if expert_actor is not None:
            return expert_actor
        return self._external_expert_module_from(self.expert_model)

    def _load_external_expert_state(self, expert_model, model_state_dict):
        expert_actor = self._external_expert_module_from(expert_model)
        for prefix in ("_actor.", "actor."):
            actor_state_dict = {
                key[len(prefix) :]: value
                for key, value in model_state_dict.items()
                if key.startswith(prefix)
            }
            if actor_state_dict:
                expert_actor.load_state_dict(actor_state_dict)
                return

        expert_model.load_state_dict(model_state_dict)

    def _build_expert_obs_td(
        self, obs_td: TensorDict, expert_in_keys: list
    ) -> TensorDict:
        """Build expert observation TensorDict by stripping 'expert_' prefix from keys.

        The experiment file adds expert observation components with "expert_" prefix
        (e.g., "expert_max_coords_obs"). This method maps those back to the keys
        the expert model expects (e.g., "max_coords_obs").

        Args:
            obs_td: Full observation TensorDict with both student and expert_* keys
            expert_in_keys: List of keys the expert model expects

        Returns:
            TensorDict with keys matching expert model's in_keys
        """
        expert_obs = {}
        for key in expert_in_keys:
            expert_key = f"expert_{key}"
            if expert_key in obs_td.keys():
                # Prefer prefixed expert observation
                expert_obs[key] = obs_td[expert_key]
            else:
                raise KeyError(
                    f"Expert model requires observation '{expert_key}' for "
                    f"expert input '{key}'. Available keys: {list(obs_td.keys())}"
                )
        return TensorDict(expert_obs, batch_size=obs_td.batch_size, device=self.device)

    def create_optimizers(self, model: BaseModel):
        # A model may split its branches into parameter groups with their own
        # learning rates (the FSQ student's AR prior and privileged encoder);
        # None keeps the single flat group over every trainable parameter.
        param_groups = getattr(model, "optimizer_param_groups", lambda: None)()
        optimizer = instantiate_optimizer(
            self.config.model.optimizer,
            model.optimization_module(),
            params=param_groups,
        )
        self.training_model, self.supervised_optimizer = self._setup_model_optimizer(
            model,
            optimizer,
        )

    # -----------------------------
    # Training Loop and Dataset Processing
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        if self.expert_model is not None:
            num_actions = self.env.robot_config.number_of_actions
            self.experience_buffer.register_key(
                "expert_actions",
                shape=(num_actions,),
            )
            if self._action_rate_loss_coeff > 0.0:
                self.experience_buffer.register_key(
                    PREVIOUS_EXPERT_ACTIONS_KEY,
                    shape=(num_actions,),
                )
                # Zeroed on episode reset, matching what the environment does
                # to its action history (`env.py`: "Zero actions for historical
                # reset"), so the two endpoints of both increments agree on the
                # first step of an episode.
                self._previous_expert_actions = torch.zeros(
                    self.num_envs, num_actions, device=self.device
                )
            if self._dagger_action_loss_coeff > 0.0:
                # The mask is stored per row because minibatches are shuffled:
                # `compute_prior_rollout_mask` is a function of the env index,
                # which does not survive the shuffle.
                self.experience_buffer.register_key(
                    PRIOR_ROLLOUT_MASK_KEY, shape=()
                )
            for offset in self._ladder_offsets[1:]:
                # Filled in `pre_process_dataset` by shifting `expert_actions`
                # along the time axis, which is why these are registered here
                # and written once per rollout rather than per step.
                self.experience_buffer.register_key(
                    ladder_target_key(offset), shape=(num_actions,)
                )
                self.experience_buffer.register_key(
                    ladder_valid_key(offset), shape=()
                )

    def _ladder_ablation_due(self) -> bool:
        """True on the first optimisation batch of every Nth epoch."""
        if len(self._ladder_offsets) <= 1:
            return False
        if int(self.current_epoch) % LADDER_ABLATION_EVERY != 0:
            return False
        already = getattr(self, "_ladder_ablation_epoch", None)
        if already == int(self.current_epoch):
            return False
        self._ladder_ablation_epoch = int(self.current_epoch)
        return True

    @property
    def _ladder_loss_coeff(self) -> float:
        return float(getattr(self.config, "ladder_loss_coeff", 0.0) or 0.0)

    @property
    def _ladder_offsets(self) -> Tuple[int, ...]:
        """Ladder rungs declared by the model, or ``(0,)`` when there are none.

        Read off the model config rather than the agent's, so the trunk width,
        the decode split and the auxiliary targets cannot disagree about how
        many rungs exist.
        """
        if self._ladder_loss_coeff <= 0.0:
            return (0,)
        fsq = getattr(getattr(self.config, "model", None), "fsq", None)
        return tuple(int(o) for o in getattr(fsq, "ladder_offsets", (0,)))

    @property
    def _action_rate_loss_coeff(self) -> float:
        # getattr twice: resolved configs written before this field existed do
        # not carry it, and unit tests build bare agents with no config at all.
        config = getattr(self, "config", None)
        return float(getattr(config, "action_rate_loss_coeff", 0.0) or 0.0)

    @property
    def _action_rate_free_steps(self) -> int:
        config = getattr(self, "config", None)
        return int(getattr(config, "action_rate_free_steps", 0) or 0)

    @property
    def _dagger_action_loss_coeff(self) -> float:
        config = getattr(self, "config", None)
        return float(getattr(config, "dagger_action_loss_coeff", 0.0) or 0.0)

    def register_algorithm_experience_buffer_keys_from_obs(self, obs_td: TensorDict):
        target_key = self.config.loss.target_key
        if hasattr(self.experience_buffer, target_key):
            return

        if target_key in obs_td.keys():
            value = obs_td[target_key]
        else:
            with self._eval_model_for_buffer_registration(), torch.no_grad():
                output_td = self._collect_rollout_output(obs_td.clone())
            if target_key not in output_td.keys():
                raise KeyError(
                    f"Supervised loss target_key '{target_key}' was not produced by "
                    f"the rollout output. Available keys: {list(output_td.keys())}"
                )
            value = output_td[target_key]

        self.experience_buffer.register_key(
            target_key,
            shape=value.shape[1:],
            dtype=value.dtype,
        )

    def _collect_external_expert_action(self, obs_td: TensorDict) -> torch.Tensor:
        expert_actor = self._external_expert_module()
        expert_in_keys = getattr(self, "expert_actor_in_keys", None)
        if not expert_in_keys:
            expert_in_keys = list(getattr(expert_actor, "in_keys", []))
        expert_obs_td = self._build_expert_obs_td(
            obs_td,
            expert_in_keys,
        )
        expert_output_td = expert_actor(expert_obs_td)
        if "mean_action" in expert_output_td.keys():
            return expert_output_td["mean_action"]
        if "action" in expert_output_td.keys():
            return expert_output_td["action"]
        raise KeyError(
            "External expert actor must produce either 'mean_action' or 'action'. "
            f"Available keys: {list(expert_output_td.keys())}"
        )

    def _collect_rollout_output(self, obs_td: TensorDict) -> TensorDict:
        rollout_actor = self.config.rollout_actor
        if rollout_actor not in (RolloutActor.STUDENT, RolloutActor.EXPERT):
            raise ValueError(f"Unsupported supervised rollout_actor: {rollout_actor}")

        has_external_expert = self.expert_model is not None
        if rollout_actor == RolloutActor.EXPERT and not has_external_expert:
            model_expert_rollout = getattr(
                self.model,
                "collect_expert_rollout",
                None,
            )
            if model_expert_rollout is None:
                raise ValueError(
                    "rollout_actor=EXPERT needs an expert source: set "
                    "expert_model_path for an external expert, or use a model "
                    "that defines collect_expert_rollout."
                )
            output_td = model_expert_rollout(obs_td)
        else:
            output_td = self.model(obs_td)

        if has_external_expert:
            expert_action = self._collect_external_expert_action(obs_td)
            output_td["expert_actions"] = expert_action
            if rollout_actor == RolloutActor.EXPERT:
                output_td["action"] = expert_action
                output_td["mean_action"] = expert_action

        return output_td

    def _prior_rollout_env_mask(self) -> Optional[Tensor]:
        """Envs stepped with the deployable prior instead of the privileged action.

        ``getattr`` defaults keep resolved configs from before these fields
        existed (and minimal test configs) on the pure privileged rollout.
        """
        fraction = float(getattr(self.config, "prior_rollout_fraction", 0.0) or 0.0)
        if fraction <= 0.0:
            return None
        mask = compute_prior_rollout_mask(
            fraction=fraction,
            start_epoch=int(getattr(self.config, "prior_rollout_start_epoch", 0) or 0),
            ramp_epochs=int(getattr(self.config, "prior_rollout_ramp_epochs", 0) or 0),
            current_epoch=self.current_epoch,
            num_envs=self.num_envs,
            device=self.device,
        )
        if mask is not None and not getattr(self, "_prior_rollout_announced", False):
            self._prior_rollout_announced = True
            log.info(
                "prior-DAgger rollout mixture active from epoch %d: %d/%d envs "
                "stepped with the deployable prior's action",
                self.current_epoch, int(mask.sum()), self.num_envs,
            )
        return mask

    def collect_rollout_step(self, obs_td: TensorDict, step):
        """Collect student action and expert label for the current state."""
        output_td = self._collect_rollout_output(obs_td)

        prior_mask = None
        if self.config.rollout_actor == RolloutActor.EXPERT:
            action = output_td["action"]
        elif "privileged_action" in output_td:
            action = output_td[
                "privileged_action"
            ]  # During training, we use the privileged action
            prior_mask = self._prior_rollout_env_mask()
            if prior_mask is not None:
                # DAgger-style mixture: a block of envs is stepped with the
                # deployable prior's action, so the states the deployed policy
                # actually induces receive expert labels.
                action = torch.where(
                    prior_mask.unsqueeze(-1), output_td["action"], action
                )
        else:
            action = output_td["action"]  # During evaluation, we use the action

        if "privileged_action" in output_td.keys() and "expert_actions" in output_td.keys():
            self._record_action_gap_diagnostics(output_td, prior_mask)

        # Store model outputs
        output_keys = list(
            dict.fromkeys(list(self.model_output_keys) + [self.config.loss.target_key])
        )
        if (
            self._dagger_action_loss_coeff > 0.0
            and FSQ_USED_PRIOR_LATENT_KEY in output_td.keys()
        ):
            # Registered lazily from the first rollout step, like the loss
            # target itself: only the model knows the latent's width.
            if not hasattr(self.experience_buffer, FSQ_USED_PRIOR_LATENT_KEY):
                self.experience_buffer.register_key(
                    FSQ_USED_PRIOR_LATENT_KEY,
                    shape=output_td[FSQ_USED_PRIOR_LATENT_KEY].shape[1:],
                )
            output_keys.append(FSQ_USED_PRIOR_LATENT_KEY)
        for key in output_keys:
            if key in output_td:
                self.experience_buffer.update_data(key, step, output_td[key])
            elif key not in obs_td.keys():
                raise KeyError(
                    f"Supervised rollout output did not contain required key '{key}'. "
                    f"Available keys: {list(output_td.keys())}"
                )

        if self.expert_model is not None and "expert_actions" not in output_keys:
            self.experience_buffer.update_data(
                "expert_actions", step, output_td["expert_actions"]
            )

        if self._dagger_action_loss_coeff > 0.0:
            mask = prior_mask
            if mask is None:
                mask = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.experience_buffer.update_data(
                PRIOR_ROLLOUT_MASK_KEY, step, mask.float()
            )

        if self._previous_expert_actions is not None:
            # Store the *previous* step's label with this step's row, then
            # advance -- so a row carries (a_expert(t), a_expert(t-1)) and the
            # teacher's own increment is available in a shuffled minibatch.
            self.experience_buffer.update_data(
                PREVIOUS_EXPERT_ACTIONS_KEY, step, self._previous_expert_actions
            )
            self._previous_expert_actions = output_td["expert_actions"].detach().clone()

        output_td["action"] = action
        return output_td

    def _record_action_gap_diagnostics(
        self, output_td: TensorDict, prior_mask: Optional[Tensor]
    ) -> None:
        """Deployable-vs-privileged action gap against the expert label.

        The posterior mean is a residual on the prior mean, so the behavioural
        gap between the deployable path and the privileged one *is* that
        residual expressed in action space — and nothing else in the log
        reports it (`notes/Student_v4_methodology.MD` §7.1). Logged under
        ``env/action_gap/*``; the ``_dagger``/``_ctrl`` split separates states
        the prior itself induced from expert-manifold states.
        """
        from protomotions.agents.common.latent import (
            LATENT_MU_KEY,
            PRIVILEGED_LATENT_MU_KEY,
        )

        meter = getattr(self, "episode_env_tensors", None)
        if meter is None:
            return

        with torch.no_grad():
            expert = output_td["expert_actions"]
            prior_gap = (output_td["action"] - expert).square().mean(dim=-1)
            privileged_gap = (
                (output_td["privileged_action"] - expert).square().mean(dim=-1)
            )
            diag = {
                "action_gap/prior_mse": prior_gap.mean(),
                "action_gap/privileged_mse": privileged_gap.mean(),
            }
            latent_mu = output_td.get(LATENT_MU_KEY, None)
            privileged_mu = output_td.get(PRIVILEGED_LATENT_MU_KEY, None)
            if latent_mu is not None and privileged_mu is not None:
                diag["action_gap/latent_residual_l2"] = (
                    (privileged_mu - latent_mu).norm(dim=-1).mean()
                )
            if prior_mask is not None:
                diag["action_gap/prior_mse_dagger"] = prior_gap[prior_mask].mean()
                ctrl_mask = ~prior_mask
                if ctrl_mask.any():
                    diag["action_gap/prior_mse_ctrl"] = prior_gap[ctrl_mask].mean()
        meter.add(diag)

    def record_rollout_step(
        self,
        next_obs_td: TensorDict,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        terminated: Tensor,
        done_indices: Tensor,
        extras: Dict,
        step: int,
    ):
        prior_mask = self._prior_rollout_env_mask()
        if prior_mask is not None:
            self._record_prior_rollout_diagnostics(prior_mask, dones, extras)
        if self._previous_expert_actions is not None and done_indices.numel() > 0:
            # The environment zeroes its action history on reset, so the first
            # step of a new episode must see a zero previous expert action too
            # or the teacher's increment would be measured across the boundary.
            self._previous_expert_actions[done_indices] = 0.0
        super().record_rollout_step(
            next_obs_td,
            actions,
            rewards,
            dones,
            terminated,
            done_indices,
            extras,
            step,
        )

    def _record_prior_rollout_diagnostics(
        self, prior_mask: Tensor, dones: Tensor, extras: Dict
    ) -> None:
        """Split rollout health metrics by rollout driver.

        The imitation loss and the privileged-path rewards cannot see a
        deployable-prior regression; the prior-driven env block can. Logged
        under ``env/prior_rollout/*`` with ``_ctrl`` for the privileged block.
        """
        ctrl_mask = ~prior_mask
        diag = {"prior_rollout/frac": prior_mask.float().mean()}
        for extras_key, name in (("raw_r/gt_rew", "gt_rew"), ("terminate", "terminate")):
            value = extras.get(extras_key)
            if not isinstance(value, torch.Tensor) or value.numel() != prior_mask.numel():
                continue
            value = value.float().flatten()
            diag[f"prior_rollout/{name}"] = value[prior_mask].mean()
            if ctrl_mask.any():
                diag[f"prior_rollout/{name}_ctrl"] = value[ctrl_mask].mean()
        self.episode_env_tensors.add(diag)

        prior_dones = dones.bool() & prior_mask
        if prior_dones.any():
            if not hasattr(self, "_prior_rollout_length_meter"):
                self._prior_rollout_length_meter = TensorAverageMeterDict(
                    device=self.device
                )
            # current_lengths is incremented by the base record step after this
            # hook runs, so +1 reproduces the value the base meters will see.
            self._prior_rollout_length_meter.add(
                {"episode_length": (self.current_lengths + 1)[prior_dones]}
            )

    def post_epoch_logging(self, training_log_dict: Dict):
        meter = getattr(self, "_prior_rollout_length_meter", None)
        if meter is not None:
            lengths = meter.mean_and_clear()
            if "episode_length" in lengths:
                training_log_dict["info/prior_rollout_episode_length"] = lengths[
                    "episode_length"
                ]
        self._maybe_run_sequence_viz()
        super().post_epoch_logging(training_log_dict)

    def _maybe_run_sequence_viz(self) -> None:
        """Render the stick-figure sequence panel when the epoch asks for it.

        Fully guarded: any failure disables the feature for the rest of the run
        rather than killing training, and the env is snapshot/restored inside
        the runner, so the next policy update is skipped exactly as after eval.
        """
        viz_config = getattr(getattr(self, "config", None), "sequence_viz", None)
        every = int(getattr(viz_config, "viz_every", 0) or 0) if viz_config else 0
        if (
            every <= 0
            or self.current_epoch == 0
            or self.current_epoch % every != 0
            or getattr(self, "_sequence_viz_disabled", False)
            or self.fabric.global_rank != 0
        ):
            return
        try:
            if not hasattr(self, "_sequence_viz_runner"):
                from protomotions.agents.evaluators.sequence_viz import (
                    SequenceVizRunner,
                )

                self._sequence_viz_runner = SequenceVizRunner(self, viz_config)
            # The runner disturbs every env; skip the next update even if the
            # rollout dies partway, exactly as the evaluator path does.
            self._skip_next_policy_update = True
            videos, scalars = self._sequence_viz_runner.run(self.current_epoch)
            self._log_sequence_viz(videos, scalars)
        except Exception:
            log.exception(
                "sequence viz failed at epoch %d; disabling it for the rest of "
                "the run",
                self.current_epoch,
            )
            self._sequence_viz_disabled = True

    def _log_sequence_viz(self, videos: Dict, scalars: Dict) -> None:
        wandb_logger = next(
            (
                logger
                for logger in getattr(self.fabric, "loggers", [])
                if type(logger).__name__ == "WandbLogger"
            ),
            None,
        )
        if wandb_logger is None:
            return
        import wandb

        payload: Dict = {
            key: wandb.Video(str(path), format="mp4")
            for key, path in videos.items()
        }
        # Gated at the logger rather than at the source: the scalars are still
        # computed and still written to viz/epoch_*/summary.json, so offline
        # analysis is unaffected -- only the dashboard clutter goes away.
        viz_config = getattr(getattr(self, "config", None), "sequence_viz", None)
        if getattr(viz_config, "log_scalars", True):
            payload.update(scalars)
        wandb_logger.experiment.log(payload, step=self.current_epoch)

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        # Update model
        iter_log_dict = {}
        loss, loss_dict = self.supervised_step(batch_dict)
        iter_log_dict.update(loss_dict)
        grad_clip_dict = self._step_optimizer(
            loss=loss,
            model=self.training_model,
            optimizer=self.supervised_optimizer,
            model_name="model",
        )
        iter_log_dict.update(grad_clip_dict)

        return iter_log_dict

    # -----------------------------
    # Model Forward Pass and Loss Computation
    # -----------------------------
    def pre_process_dataset(self):
        """Build the action ladder's far-horizon targets, once per rollout.

        The buffer holds ``expert_actions`` as ``[T, E, A]`` and ``make_dict``
        flattens it with ``swap_and_flatten01`` to row = env*T + step, so a
        target for horizon ``h`` is just the same tensor shifted ``h`` places
        along the time axis. Two things make the shift honest:

        * rows within ``h`` of the end of the rollout have no target and are
          masked out -- the buffer is a window, not an episode;
        * so are rows with an episode boundary inside their horizon. ``dones``
          marks the step at which the episode ended, so the reference at
          ``t + h`` belongs to a different episode (and a different clip) if
          any of ``dones[t : t+h]`` fired. At the run's measured termination
          rates this costs ~2 % of control rows and ~4 % of DAgger rows at
          h = 24.

        The far rungs are legitimate on-policy DAgger labels: the expert is
        queried at every visited state, so ``expert_actions[t+h]`` is what the
        teacher does h steps down the trajectory the student actually produced.
        """
        super().pre_process_dataset()
        offsets = self._ladder_offsets
        if len(offsets) <= 1 or self.expert_model is None:
            return
        expert = self.experience_buffer.expert_actions  # [T, E, A]
        dones = self.experience_buffer.dones.bool()  # [T, E]
        num_steps = expert.shape[0]
        for offset in offsets[1:]:
            target = torch.zeros_like(expert)
            valid = torch.zeros(dones.shape, dtype=expert.dtype, device=expert.device)
            usable = num_steps - offset
            if usable > 0:
                target[:usable] = expert[offset:]
                # No done in [t, t+offset): a cumulative sum over the window is
                # cheaper and clearer than a rolling any().
                cum = torch.cumsum(dones.to(expert.dtype), dim=0)
                # boundaries in [t, t+offset) == cum[t+offset-1] - cum[t-1]
                upper = cum[offset - 1 : offset - 1 + usable]
                lower = torch.cat(
                    [torch.zeros_like(cum[:1]), cum[: usable - 1]], dim=0
                )
                valid[:usable] = (upper - lower <= 0).to(expert.dtype)
            self.experience_buffer.batch_update_data(
                ladder_target_key(offset), target
            )
            self.experience_buffer.batch_update_data(ladder_valid_key(offset), valid)

    @property
    def _ladder_balance(self) -> bool:
        return bool(getattr(self.config, "ladder_balance_to_imitation", False))

    def calculate_ladder_loss(
        self, batch_dict, imitation_loss: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict]:
        """Auxiliary MSE between the trunk's far rungs and the expert ahead.

        Each rung is divided by its own target variance before the mean, so the
        rungs are commensurable and an unequal valid-row count across horizons
        cannot tilt the sum. Rung 0 is deliberately excluded -- it is already
        the imitation loss, and adding it twice would only rescale that term.

        Also logs ``code_ablation_gap_h*``: the rise in a rung's normalised MSE
        when ``vae_latent`` is rolled across the batch. That is the pre-
        registered mechanism metric -- a large gap at the far rungs against a
        near-zero gap at rung 0 is the signature that the ladder engaged and
        nothing else did.
        """
        zero = (torch.tensor(0.0, device=self.device), {})
        offsets = self._ladder_offsets
        if len(offsets) <= 1 or LADDER_PRED_KEY not in batch_dict.keys():
            return zero
        rungs = batch_dict[LADDER_PRED_KEY]  # [B, rungs, A]
        total = torch.tensor(0.0, device=self.device)
        charged = 0
        log: Dict[str, Tensor] = {}
        for index, offset in enumerate(offsets):
            if index == 0:
                continue
            tkey, vkey = ladder_target_key(offset), ladder_valid_key(offset)
            if tkey not in batch_dict.keys():
                continue
            target = batch_dict[tkey]
            valid = batch_dict[vkey].reshape(-1) > 0.5
            if not bool(valid.any()):
                continue
            pred = rungs[..., index, :]
            scale = target[valid].var(unbiased=False).clamp(min=1e-8)
            per_row = (pred[valid] - target[valid]).square().mean(dim=-1)
            rung_loss = per_row.mean() / scale
            total = total + rung_loss
            charged += 1
            log[f"ladder/mse_h{offset}"] = rung_loss.detach()
            log[f"ladder/valid_frac_h{offset}"] = valid.float().mean()
        if charged == 0:
            return zero
        total = total / charged
        log["ladder/mean_rung"] = total.detach()
        if self._ladder_balance and imitation_loss is not None:
            # Put the ladder on the imitation loss's own scale, so the
            # coefficient means "this fraction of the imitation term" rather
            # than an absolute number on an unrelated scale. Detached on both
            # sides: this is a weighting, not a path for the imitation loss to
            # be optimised through the ladder.
            ratio = imitation_loss.detach() / total.detach().clamp(min=1e-12)
            total = total * ratio
            log["ladder/balance_scale"] = ratio
        weighted = self._ladder_loss_coeff * total
        log["ladder/loss"] = weighted.detach()
        return (weighted, log)

    @torch.no_grad()
    def ladder_code_ablation(self, batch_dict) -> Dict:
        """How much each rung actually depends on the intent code.

        Re-decodes the batch with ``vae_latent`` rolled by one row and reports
        the rise in each rung's normalised MSE. Rung 0's ceiling is 2.03e-4 of
        target variance (the h=0 target is state-linear to R^2 = 0.999797), so
        a near-zero gap there is expected and is the control; the far rungs
        have 0.017-0.091 available. Diagnostic only -- no gradient.
        """
        offsets = self._ladder_offsets
        if len(offsets) <= 1 or LADDER_PRED_KEY not in batch_dict.keys():
            return {}
        model = self.model
        decode = getattr(model, "_decode", None)
        latent = batch_dict.get(VAE_LATENT_KEY)
        if decode is None or latent is None:
            return {}
        shuffled = TensorDict(
            {k: v for k, v in batch_dict.items()}, batch_size=batch_dict.batch_size
        )
        probe_key = "_ladder_ablation_pred"
        decode(shuffled, torch.roll(latent, 1, dims=0), False, ladder_key=probe_key)
        if probe_key not in shuffled.keys():
            return {}
        rolled = shuffled[probe_key]
        base = batch_dict[LADDER_PRED_KEY]
        log: Dict[str, Tensor] = {}
        for index, offset in enumerate(offsets):
            tkey, vkey = ladder_target_key(offset), ladder_valid_key(offset)
            if index == 0:
                target = batch_dict.get(self.config.loss.target_key)
                if target is None:
                    continue
                valid = torch.ones(
                    target.shape[0], dtype=torch.bool, device=target.device
                )
            else:
                if tkey not in batch_dict.keys():
                    continue
                target = batch_dict[tkey]
                valid = batch_dict[vkey].reshape(-1) > 0.5
            if not bool(valid.any()):
                continue
            scale = target[valid].var(unbiased=False).clamp(min=1e-8)
            keep = (
                (base[..., index, :][valid] - target[valid]).square().mean() / scale
            )
            drop = (
                (rolled[..., index, :][valid] - target[valid]).square().mean() / scale
            )
            log[f"model/code_ablation_gap_h{offset}"] = (drop - keep).detach()
        return log

    def supervised_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Compute supervised imitation loss from a rollout batch."""
        # Convert to TensorDict and run model forward
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.training_model(batch_td)

        supervised_loss, supervised_log_dict = compute_supervision_loss(
            batch_td,
            self.config.loss,
        )
        actions = (
            batch_td["privileged_action"]
            if "privileged_action" in batch_td.keys()
            else batch_td["action"]
        )

        extra_loss, extra_log_dict = self.calculate_extra_loss(batch_td, actions)

        dagger_loss, dagger_log_dict = self.calculate_dagger_action_loss(batch_td)
        extra_loss = extra_loss + dagger_loss
        extra_log_dict.update(dagger_log_dict)

        ladder_loss, ladder_log_dict = self.calculate_ladder_loss(
            batch_td, imitation_loss=supervised_loss
        )
        extra_loss = extra_loss + ladder_loss
        extra_log_dict.update(ladder_log_dict)
        if self._ladder_ablation_due():
            extra_log_dict.update(self.ladder_code_ablation(batch_td))

        model_loss, model_log_dict = self.model.compute_model_loss(
            batch_td,
            current_epoch=self.current_epoch,
            zero_loss=supervised_loss,
            log_prefix="model",
        )

        loss = supervised_loss + extra_loss + model_loss

        log_dict = {
            "supervised/loss": supervised_loss.detach(),
            "supervised/extra_loss": extra_loss.detach(),
            "supervised/model_loss": model_loss.detach(),
            "losses/supervised_loss": loss.detach(),
        }
        log_dict.update(supervised_log_dict)
        log_dict.update(model_log_dict)
        log_dict.update(extra_log_dict)

        return loss, log_dict

    def calculate_extra_loss(self, batch_dict, actions) -> Tuple[Tensor, Dict]:
        """Teacher-rate matching, when configured; zero otherwise.

        ``||(a_student - a_prev) - (a_expert - a_expert_prev)||^2``. Note that
        the two increments must be measured from *different* previous actions:
        referencing both to the applied ``previous_actions`` cancels it and
        leaves the plain imitation MSE, which is why the previous expert action
        is carried in the buffer. Written out, the term is
        ``||residual(t) - residual(t-1)||^2`` -- it costs nothing for a
        persistent offset and everything for a residual that jitters, which is
        the shape of the chunk-clock kick round 6 measured. It shares its
        optimum with the imitation loss (a perfect student has both endpoints
        right, so both terms vanish together), so it biases toward temporal
        consistency without moving the target.

        ``action_rate_free_steps`` exempts the first rows of each intent chunk.
        Round 7_1 §2.2: the term's *only* contact with a commitment is the
        refresh row, whose increment is the one that spans a code change --
        everywhere else it charges jitter, which is what it is for. Charging
        the refresh row too is why v7 removed the 3.75 Hz artifact and the
        commitment together (code gain 0.33 -> 0.07). Exempting it separates
        the two in time. Needs the FSQ model, which publishes the chunk phase;
        without that key the term is charged everywhere, as before.
        """
        zero = (torch.tensor(0.0, device=self.device), {})
        if self._action_rate_loss_coeff <= 0.0:
            return zero
        target_key = self.config.loss.target_key
        if not {
            PREVIOUS_EXPERT_ACTIONS_KEY,
            target_key,
            "previous_actions",
        }.issubset(set(batch_dict.keys())):
            return zero

        expert = batch_dict[target_key]
        # `previous_actions` is a flattened action history, most recent step
        # first (`select_step_indices` is 1-indexed and `rotate_and_update`
        # inserts at 0), so the leading action-dim block is step t-1 whatever
        # `history_steps` the experiment asked for.
        previous = batch_dict["previous_actions"][..., : actions.shape[-1]]
        student_rate = actions - previous
        expert_rate = expert - batch_dict[PREVIOUS_EXPERT_ACTIONS_KEY]
        per_row = (student_rate - expert_rate).square().mean(dim=-1)

        free_steps = self._action_rate_free_steps
        phase = batch_dict.get(FSQ_PHASE_INDEX_KEY) if free_steps > 0 else None
        charged = None
        if phase is not None:
            charged = phase.reshape(-1) >= free_steps
            # A batch with no charged row cannot happen at chunk_steps=8 (only
            # ~13 % of rows are refreshes) but would make the mean undefined,
            # so fall back rather than emit a NaN into the total loss.
            if not bool(charged.any()):
                charged = None
        rate_error = per_row.mean() if charged is None else per_row[charged].mean()
        log = {"supervised/action_rate": rate_error.detach()}
        if charged is not None:
            log["supervised/action_rate_charged_frac"] = charged.float().mean()
        return (self._action_rate_loss_coeff * rate_error, log)

    def calculate_dagger_action_loss(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Imitation MSE on the DEPLOYABLE action, over prior-driven rows only.

        The trunk is otherwise trained exclusively through ``privileged_action``
        -- decoded from the *encoder's* code -- while at deployment it decodes a
        code sampled from the AR prior, which agrees with the encoder's on only
        3-18 % of refreshes. So on the overwhelming majority of deployed steps
        the decoder is extrapolating to a code it has never received a gradient
        for. That is textbook VQ-with-prior exposure bias and it is what forces
        the decoder to be either over-sensitive (v6's jerk) or insensitive
        (v7/v8's lost commitment); every knob tried so far has moved along that
        trade rather than shifting it (``notes/Student_improvement_round7_1.MD``
        §3.2, §7c).

        Restricted to the DAgger block on purpose. There the sampled action was
        the one actually applied and the expert labelled the *resulting* state,
        so matching it is exactly the correction DAgger prescribes. On a
        privileged-driven row the same term would only be pressure for every
        code to decode to the one expert action -- i.e. mode collapse, the thing
        the discrete head exists to avoid.
        """
        zero = (torch.tensor(0.0, device=self.device), {})
        coeff = self._dagger_action_loss_coeff
        if coeff <= 0.0:
            return zero
        target_key = self.config.loss.target_key
        if not {PRIOR_ACTION_KEY, PRIOR_ROLLOUT_MASK_KEY, target_key}.issubset(
            set(batch_dict.keys())
        ):
            return zero

        mask = batch_dict[PRIOR_ROLLOUT_MASK_KEY].reshape(-1) > 0.5
        if not bool(mask.any()):
            # Before `prior_rollout_start_epoch` there are no prior-driven rows
            # at all; the term is simply inactive rather than undefined.
            return zero
        error = (
            (batch_dict[PRIOR_ACTION_KEY] - batch_dict[target_key])
            .square()
            .mean(dim=-1)[mask]
            .mean()
        )
        return (
            coeff * error,
            {
                "supervised/dagger_action_loss": error.detach(),
                "supervised/dagger_action_frac": mask.float().mean(),
            },
        )

    # -----------------------------
    # State Saving and Restoration
    # -----------------------------
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["supervised_optimizer"] = self.supervised_optimizer.state_dict()
        return state_dict

    def _load_training_state(self, state_dict):
        super()._load_training_state(state_dict)
        optimizer_state = state_dict.get(
            "supervised_optimizer",
            state_dict.get("maskedmimic_optimizer"),
        )
        if optimizer_state is None:
            raise KeyError("supervised_optimizer")
        self.supervised_optimizer.load_state_dict(optimizer_state)
