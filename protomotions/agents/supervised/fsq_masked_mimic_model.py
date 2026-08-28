# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FSQ-C MaskedMimic student: chunked discrete-intent codes for the supervised agent.

Round 5 measured the Gaussian student's posterior residual collapsing to ~0 —
at convergence the CVAE is deterministic BC, and whatever multimodality the
data has is averaged inside the shared prior path
(``notes/Student_improvement_round5.MD`` §3). This model replaces the Gaussian
bottleneck with the repo's FSQ machinery, in the *chunked* form
(``notes/Student_multimodal_head_investigation.MD`` §3.3): the latent is a
finite-scalar-quantized **intent code held for a chunk of control steps**, and
the deployable prior is a categorical autoregressive head whose sampling — not
regression — picks which continuation to commit to.

Three code streams share one per-env chunk clock and refresh together (counter
expiry, a committed contact event, or episode reset):

* **teacher** — ``quantize(encoder(privileged obs))``; decoded to
  ``privileged_action`` (drives ctrl envs, carries the imitation MSE through
  the straight-through estimator);
* **sampled** — nucleus-sampled AR tokens; decoded to ``action`` (drives the
  DAgger env block and, via ``forward_inference``, probes/queries — the
  stream whose sampling is the commitment mechanism);
* **greedy** — argmax AR tokens; decoded to ``mean_action`` (what the
  evaluators prefer, so ``eval/success_rate`` stays deterministic and
  comparable across runs).

Between refreshes the codes are constant while the trunk keeps running
per-step on live state: closed-loop stabilization inside a committed intent.

Replay contract: the chunk state (three code sets + the clock) is declared
rollout state, so the framework stores the *pre-step* values in the experience
buffer and re-forwarding a batch reproduces the rollout's refresh decisions
exactly. Mid-chunk samples therefore decode the *stored* held code (a
constant — the encoder deliberately receives gradients only through
refresh-step samples, which are the only states that ever issue codes at
deployment), while the trunk trains on every sample against the expert label —
which *is* the commitment contract, stated as a loss.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from protomotions.agents.base_agent.model import (
    BaseModel,
    ProtoMotionsTensorDictModule,
    RolloutStateSpec,
)
from protomotions.agents.common.autoregressive import (
    resolve_discrete_autoregressive_config,
)
from protomotions.agents.common.discrete_latent import FSQTokenization
from protomotions.agents.common.fsq import FiniteScalarQuantizer
from protomotions.agents.common.latent import (
    LATENT_LOGITS_KEY,
    LATENT_MU_KEY,
    PRIVILEGED_LATENT_MU_KEY,
    TARGET_LATENT_KEY,
    VAE_LATENT_KEY,
)
from protomotions.utils.hydra_replacement import get_class

# Per-env chunk state. Declared as rollout state so the framework stores the
# pre-step values in the experience buffer and replays them during
# optimization; the module-side buffers always hold the post-step values.
FSQ_TEACHER_CODES_KEY = "fsq_teacher_codes"
FSQ_PRIOR_CODES_KEY = "fsq_prior_codes"
FSQ_MEAN_CODES_KEY = "fsq_mean_codes"
FSQ_STEPS_LEFT_KEY = "fsq_steps_left"


class FSQMaskedMimicModel(BaseModel):
    """MaskedMimic student with a chunked FSQ intent bottleneck and AR prior."""

    def __init__(self, config):
        super().__init__(config)

        encoder_class = get_class(self.config.encoder._target_)
        self._encoder = encoder_class(config=self.config.encoder)
        prior_class = get_class(self.config.prior._target_)
        self._prior = prior_class(config=self.config.prior)
        trunk_class = get_class(self.config.trunk._target_)
        self._trunk = trunk_class(config=self.config.trunk)

        fsq = self.config.fsq
        self.quantizer = FiniteScalarQuantizer(
            fsq.num_fsq_levels, fsq.num_fsq_scalars
        )
        self.tokenization = FSQTokenization(
            num_fsq_levels=fsq.num_fsq_levels,
            num_fsq_scalars=fsq.num_fsq_scalars,
            fsq_scalars_per_prior_token=fsq.fsq_scalars_per_prior_token,
        )
        ar_config = resolve_discrete_autoregressive_config(
            self.config.ar_head,
            num_tokens=self.tokenization.num_prior_tokens,
            vocab_size=self.tokenization.prior_token_vocab_size,
        )
        self._ar_head = get_class(ar_config._target_)(config=ar_config)

        trunk_in_keys = [key for key in self._trunk.in_keys if key != VAE_LATENT_KEY]
        event_keys = [fsq.event_flag_key] if fsq.event_flag_key else []
        self.in_keys = list(
            dict.fromkeys(
                self._prior.in_keys
                + self._encoder.in_keys
                + trunk_in_keys
                + event_keys
            )
        )
        self.out_keys = [
            "action",
            "mean_action",
            "privileged_action",
            TARGET_LATENT_KEY,
        ]

    # ------------------------------------------------------------------ #
    # Rollout state
    # ------------------------------------------------------------------ #
    def rollout_state_specs(self) -> dict[str, RolloutStateSpec]:
        num_scalars = self.config.fsq.num_fsq_scalars
        return {
            **super().rollout_state_specs(),
            FSQ_TEACHER_CODES_KEY: RolloutStateSpec(shape=(num_scalars,)),
            FSQ_PRIOR_CODES_KEY: RolloutStateSpec(shape=(num_scalars,)),
            FSQ_MEAN_CODES_KEY: RolloutStateSpec(shape=(num_scalars,)),
            # Zero-init means every stream refreshes on the first post-reset
            # forward, which is also what makes replay of that step exact.
            FSQ_STEPS_LEFT_KEY: RolloutStateSpec(shape=()),
        }

    def flush_held_intent(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Force a code refresh on the next forward (all streams).

        For external drivers that change the goal mid-episode
        (``set_manual_goal`` in the probe/viz tools): without this the policy
        executes the stale intent for up to ``chunk_steps`` more steps.
        """
        state = getattr(self, FSQ_STEPS_LEFT_KEY, None)
        if not torch.is_tensor(state):
            return
        if env_ids is None:
            state.zero_()
        else:
            state[env_ids] = 0.0

    # ------------------------------------------------------------------ #
    # Shared pieces
    # ------------------------------------------------------------------ #
    def _forward_module(self, module, tensordict, log_internals):
        if isinstance(module, ProtoMotionsTensorDictModule):
            return module(tensordict, log_internals=log_internals)
        return module(tensordict)

    def _decode(
        self,
        tensordict: TensorDict,
        codes: torch.Tensor,
        log_internals: bool = False,
    ) -> torch.Tensor:
        tensordict[VAE_LATENT_KEY] = codes
        tensordict = self._forward_module(self._trunk, tensordict, log_internals)
        return tensordict[self._trunk.out_keys[0]]

    def _refresh_mask(self, tensordict: TensorDict) -> torch.Tensor:
        """Refresh decision from the pre-step clock and the contact-event flag.

        A pure function of tensors that are stored in the experience buffer
        (rollout state + observations), so the replayed forward reproduces the
        rollout's decision exactly.
        """
        steps_left = tensordict[FSQ_STEPS_LEFT_KEY].reshape(-1)
        refresh = steps_left < 0.5
        event_key = self.config.fsq.event_flag_key
        if event_key and event_key in tensordict.keys():
            refresh = refresh | (
                tensordict[event_key].reshape(refresh.shape[0]) > 0.5
            )
        return refresh

    def _teacher_codes(
        self, tensordict: TensorDict, refresh: torch.Tensor, log_internals: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(codes_used, codes_now): held teacher codes and this step's encode.

        On refresh rows ``codes_used`` is the fresh straight-through-quantized
        encoder output (gradients flow); on hold rows it is the stored held
        code (a constant).
        """
        tensordict = self._forward_module(self._encoder, tensordict, log_internals)
        raw = tensordict[self._encoder.out_keys[0]]
        codes_now = self.quantizer.quantize(raw)
        held = tensordict[FSQ_TEACHER_CODES_KEY]
        codes_used = torch.where(refresh.unsqueeze(-1), codes_now, held)
        return codes_used, codes_now

    def _codes_to_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        return self.tokenization.fsq_indices_to_prior_tokens(
            self.quantizer.codes_to_indices(codes)
        )

    def _tokens_to_codes(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.quantizer.indices_to_codes(
            self.tokenization.prior_tokens_to_fsq_indices(tokens)
        )

    @torch.no_grad()
    def _greedy_from_context(
        self, context: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Argmax token sequence (deterministic counterpart of nucleus sampling)."""
        generated = []
        for _ in range(num_tokens):
            prefix = torch.stack(generated, dim=1) if generated else None
            step_logits = self._ar_head.next_logits_from_context(
                context, token_indices=prefix
            )
            generated.append(step_logits.argmax(dim=-1))
        return torch.stack(generated, dim=1)

    @torch.no_grad()
    def _advance_stream(
        self,
        held: torch.Tensor,
        context: torch.Tensor,
        refresh: torch.Tensor,
        greedy: bool,
    ) -> torch.Tensor:
        """Return the post-step codes of one AR-generated stream.

        Generation runs only on the refresh rows — with ``chunk_steps`` = 8
        that is ~1/8 of envs per step plus event rows, so the sequential AR
        loop stays cheap.
        """
        new_codes = held.clone()
        rows = refresh.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return new_codes
        fsq = self.config.fsq
        row_context = context[rows]
        if greedy:
            tokens = self._greedy_from_context(
                row_context, self.tokenization.num_prior_tokens
            )
        else:
            tokens, _, _ = self._ar_head.generate_from_context(
                row_context,
                num_tokens=self.tokenization.num_prior_tokens,
                temperature=fsq.temperature,
                top_p=fsq.top_p,
            )
        new_codes[rows] = self._tokens_to_codes(tokens)
        return new_codes

    def _write_rollout_state(
        self, tensordict: TensorDict, updates: Dict[str, torch.Tensor]
    ) -> None:
        """Persist post-step chunk state into the module buffers.

        The TensorDict keeps the *pre-step* values ``read_rollout_state``
        injected — those are what the experience buffer stores and what a
        replayed forward must start from.
        """
        batch_size = tensordict.batch_size[0]
        for key, value in updates.items():
            state = getattr(self, key)
            state[:batch_size] = value.detach()

    # ------------------------------------------------------------------ #
    # Forward paths
    # ------------------------------------------------------------------ #
    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        # A batch replayed from the experience buffer already carries the
        # chunk state; a live rollout TensorDict does not. This is the same
        # convention read_rollout_state itself uses, checked before the call
        # so the two modes cannot be confused.
        replay = FSQ_TEACHER_CODES_KEY in tensordict.keys()
        self.read_rollout_state(tensordict)

        refresh = self._refresh_mask(tensordict)
        teacher_codes, codes_now = self._teacher_codes(
            tensordict, refresh, log_internals
        )
        tensordict["privileged_action"] = self._decode(
            tensordict, teacher_codes, log_internals
        )

        tensordict = self._forward_module(self._prior, tensordict, log_internals)
        context = self._ar_head.encode_context(tensordict)

        target_tokens = self._codes_to_tokens(teacher_codes.detach())
        tensordict[TARGET_LATENT_KEY] = target_tokens

        if replay:
            # Optimization pass: teacher-forced logits for the CE; the
            # generated streams are not in any loss and are skipped.
            logits = self._ar_head.forward_from_tokens(
                context, self.tokenization.one_hot_prior_tokens(target_tokens)
            )
            tensordict[LATENT_LOGITS_KEY] = logits
            tensordict["_fsq_codes_now"] = codes_now.detach()
            tensordict["_fsq_refresh"] = refresh
            return tensordict

        steps_left = tensordict[FSQ_STEPS_LEFT_KEY].reshape(-1)
        prior_codes = self._advance_stream(
            tensordict[FSQ_PRIOR_CODES_KEY], context, refresh, greedy=False
        )
        mean_codes = self._advance_stream(
            tensordict[FSQ_MEAN_CODES_KEY], context, refresh, greedy=True
        )

        tensordict["action"] = self._decode(tensordict, prior_codes, log_internals)
        tensordict["mean_action"] = self._decode(
            tensordict, mean_codes, log_internals
        )
        # Deployable-vs-teacher intent disagreement, in code space: keeps the
        # agent's env/action_gap/latent_residual_l2 diagnostic meaningful.
        tensordict[LATENT_MU_KEY] = prior_codes
        tensordict[PRIVILEGED_LATENT_MU_KEY] = teacher_codes.detach()

        self._write_rollout_state(
            tensordict,
            {
                FSQ_TEACHER_CODES_KEY: teacher_codes,
                FSQ_PRIOR_CODES_KEY: prior_codes,
                FSQ_MEAN_CODES_KEY: mean_codes,
                FSQ_STEPS_LEFT_KEY: torch.where(
                    refresh,
                    torch.full_like(steps_left, float(self.config.fsq.chunk_steps - 1)),
                    (steps_left - 1.0).clamp(min=0.0),
                ),
            },
        )
        return tensordict

    def forward_inference(self, tensordict: TensorDict) -> TensorDict:
        """Deployable path: sampled (or greedy, per config) intent + trunk.

        Emits only ``action`` — deliberately no ``mean_action``, so the probe
        and viz drivers (which prefer it) run the *sampled* stream and seeded
        attempt diversity is measurable.
        """
        self.read_rollout_state(tensordict)
        refresh = self._refresh_mask(tensordict)
        steps_left = tensordict[FSQ_STEPS_LEFT_KEY].reshape(-1)

        tensordict = self._forward_module(self._prior, tensordict, False)
        context = self._ar_head.encode_context(tensordict)
        prior_codes = self._advance_stream(
            tensordict[FSQ_PRIOR_CODES_KEY],
            context,
            refresh,
            greedy=self.config.fsq.inference_argmax,
        )
        tensordict["action"] = self._decode(tensordict, prior_codes, False)
        tensordict[LATENT_MU_KEY] = prior_codes

        self._write_rollout_state(
            tensordict,
            {
                FSQ_PRIOR_CODES_KEY: prior_codes,
                FSQ_STEPS_LEFT_KEY: torch.where(
                    refresh,
                    torch.full_like(steps_left, float(self.config.fsq.chunk_steps - 1)),
                    (steps_left - 1.0).clamp(min=0.0),
                ),
            },
        )
        return tensordict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys = [key for key in self._trunk.in_keys if key != VAE_LATENT_KEY]
        event_keys = (
            [self.config.fsq.event_flag_key] if self.config.fsq.event_flag_key else []
        )
        return list(
            dict.fromkeys(self._prior.in_keys + trunk_in_keys + event_keys)
        )

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    def _ce_coefficient(self, current_epoch: int) -> float:
        schedule = self.config.fsq.ce_schedule
        if schedule.end_epoch <= schedule.start_epoch:
            progress = 0.0 if current_epoch < schedule.start_epoch else 1.0
        else:
            progress = min(
                max(0, current_epoch - schedule.start_epoch)
                / (schedule.end_epoch - schedule.start_epoch),
                1,
            )
        return schedule.init_ce_coeff + progress * (
            schedule.end_ce_coeff - schedule.init_ce_coeff
        )

    def compute_model_loss(
        self,
        tensordict: Optional[TensorDict],
        current_epoch: int,
        zero_loss: torch.Tensor,
        log_prefix: str = "model",
    ) -> Tuple[torch.Tensor, Dict]:
        loss, log_dict = super().compute_model_loss(
            tensordict,
            current_epoch=current_epoch,
            zero_loss=zero_loss,
            log_prefix=log_prefix,
        )
        if tensordict is None or LATENT_LOGITS_KEY not in tensordict.keys():
            return loss, log_dict

        logits = tensordict[LATENT_LOGITS_KEY]
        target = tensordict[TARGET_LATENT_KEY]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            label_smoothing=self.config.fsq.label_smoothing,
        )
        ce_coeff = self._ce_coefficient(current_epoch)
        model_loss = ce * ce_coeff

        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            token_accuracy = (predicted == target).float().mean()
            full_match = (predicted == target).all(dim=-1).float().mean()
            perplexity = self.quantizer.calculate_perplexity(
                tensordict["_fsq_codes_now"]
            )
            refresh_frac = tensordict["_fsq_refresh"].float().mean()

        log_dict.update(
            {
                f"{log_prefix}/fsq_ce_loss": ce.detach(),
                f"{log_prefix}/fsq_ce_coeff": torch.tensor(
                    ce_coeff, device=ce.device
                ),
                f"{log_prefix}/fsq_token_accuracy": token_accuracy,
                f"{log_prefix}/fsq_full_match": full_match,
                f"{log_prefix}/fsq_code_perplexity": perplexity,
                f"{log_prefix}/fsq_refresh_frac": refresh_frac,
            }
        )
        return loss + model_loss, log_dict
