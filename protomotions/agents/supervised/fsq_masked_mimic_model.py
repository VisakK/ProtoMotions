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

Round 7 adds two options on top, both defaulting to the v6 behaviour so that
run stays reproducible from its own config:

* ``latent_ema_alpha`` < 1 ramps a committed code into the trunk instead of
  substituting it, because round 6's holds carried 2-8x v5's joint-acceleration
  RMS with the excess sitting on the chunk clock itself (3.75 Hz, present in
  the greedy stream too, so it is the refresh and not the sampling). The ramp
  is per-stream rollout state, so the imitation loss trains *through* it.
* ``ce_refresh_rows_only`` takes the token CE over refresh rows alone — the
  only states where the deployed prior is ever asked to choose.

See ``notes/Student_v7_improvement_investigation.MD`` §10.3 and §5.3.
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

# The smoothed counterpart of each code stream: what the trunk actually
# decodes when ``latent_ema_alpha`` < 1 (v7). Also rollout state, so the ramp
# replays exactly and the imitation loss trains through it.
FSQ_TEACHER_LATENT_KEY = "fsq_teacher_latent"
FSQ_PRIOR_LATENT_KEY = "fsq_prior_latent"
FSQ_MEAN_LATENT_KEY = "fsq_mean_latent"

# One-hot position of the current step inside its chunk, written by the model
# and consumed by the trunk when ``chunk_phase_to_trunk`` is on (round 7_1 §6.2).
# The trunk has no clock otherwise, so a held code can only mean "shift the
# action by a constant" -- never "over the next 0.27 s, do this".
FSQ_CHUNK_PHASE_KEY = "fsq_chunk_phase"
# Diagnostics/losses that need the same quantity as an index rather than a
# one-hot: the refresh mask and the chunk phase, both pure functions of stored
# rollout state, so a replayed forward reproduces them exactly.
FSQ_REFRESH_KEY = "_fsq_refresh"
FSQ_PHASE_INDEX_KEY = "_fsq_chunk_phase"

# The latent the SAMPLED stream actually decoded this step -- the code after
# generation and after the EMA ramp, i.e. exactly what the trunk consumed to
# produce ``action``. Published on the rollout path so the agent can store it,
# and consumed on replay to re-decode ``prior_action`` WITH gradient. The
# generated streams are otherwise skipped on replay, so without this the
# deployable action can carry no loss at all (round 7_1 §3.2).
FSQ_USED_PRIOR_LATENT_KEY = "fsq_used_prior_latent"
PRIOR_ACTION_KEY = "prior_action"
# Full [B, rungs, A] trunk output of the PRIVILEGED decode, when the action
# ladder is on. Written only by that one decode so the later prior/mean
# decodes cannot overwrite the tensor the auxiliary loss reads.
LADDER_PRED_KEY = "ladder_pred"

_LATENT_OF_CODES = {
    FSQ_TEACHER_CODES_KEY: FSQ_TEACHER_LATENT_KEY,
    FSQ_PRIOR_CODES_KEY: FSQ_PRIOR_LATENT_KEY,
    FSQ_MEAN_CODES_KEY: FSQ_MEAN_LATENT_KEY,
}


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
        # Action ladder. `ladder_offsets` is (0,) by default, which makes every
        # branch below a no-op and the trunk a plain 69-way action head.
        self._ladder_offsets = tuple(int(o) for o in getattr(fsq, "ladder_offsets", (0,)))
        if self._ladder_offsets[0] != 0:
            raise ValueError(
                "fsq.ladder_offsets must start with 0 -- rung 0 is the action "
                f"the simulator executes, got {self._ladder_offsets}"
            )
        if sorted(set(self._ladder_offsets)) != list(self._ladder_offsets):
            raise ValueError(
                f"fsq.ladder_offsets must be strictly increasing, got {self._ladder_offsets}"
            )
        self._num_rungs = len(self._ladder_offsets)

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

        trunk_in_keys = self._external_trunk_in_keys()
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
    # Optimization
    # ------------------------------------------------------------------ #
    def optimizer_param_groups(self):
        """Split the AR prior and the privileged encoder off at their own rates.

        Two branches learn on very different budgets from the shared trunk:
        the AR token prior is a small classifier whose in-repo counterpart is
        trained at AdamW 1e-4, and the encoder receives the straight-through
        gradient on refresh rows only (~13 % of samples), further scaled by the
        latent-smoothing alpha. Returns ``None`` — one flat group, exactly v6 —
        unless at least one override is configured.

        The partition is by module prefix and is asserted exhaustive, so a
        parameter added to the model later cannot silently fall out of the
        optimizer.
        """
        ar_lr = getattr(self.config, "ar_head_lr", None)
        ar_wd = getattr(self.config, "ar_head_weight_decay", None)
        encoder_lr = getattr(self.config, "encoder_lr", None)
        if ar_lr is None and ar_wd is None and encoder_lr is None:
            return None

        buckets: Dict[str, list] = {"base": [], "ar_head": [], "encoder": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("_ar_head."):
                buckets["ar_head"].append(param)
            elif name.startswith("_encoder."):
                buckets["encoder"].append(param)
            else:
                buckets["base"].append(param)

        total = sum(len(group) for group in buckets.values())
        expected = sum(1 for p in self.parameters() if p.requires_grad)
        if total != expected:
            raise RuntimeError(
                f"FSQ parameter-group split covered {total} of {expected} "
                "trainable tensors"
            )

        groups = [{"params": buckets["base"]}]
        ar_group: Dict = {"params": buckets["ar_head"]}
        if ar_lr is not None:
            ar_group["lr"] = float(ar_lr)
        if ar_wd is not None:
            ar_group["weight_decay"] = float(ar_wd)
        groups.append(ar_group)
        encoder_group: Dict = {"params": buckets["encoder"]}
        if encoder_lr is not None:
            encoder_group["lr"] = float(encoder_lr)
        groups.append(encoder_group)
        return [group for group in groups if group["params"]]

    # ------------------------------------------------------------------ #
    # Rollout state
    # ------------------------------------------------------------------ #
    def rollout_state_specs(self) -> dict[str, RolloutStateSpec]:
        num_scalars = self.config.fsq.num_fsq_scalars
        specs = {
            **super().rollout_state_specs(),
            FSQ_TEACHER_CODES_KEY: RolloutStateSpec(shape=(num_scalars,)),
            FSQ_PRIOR_CODES_KEY: RolloutStateSpec(shape=(num_scalars,)),
            FSQ_MEAN_CODES_KEY: RolloutStateSpec(shape=(num_scalars,)),
            # Zero-init means every stream refreshes on the first post-reset
            # forward, which is also what makes replay of that step exact.
            FSQ_STEPS_LEFT_KEY: RolloutStateSpec(shape=()),
        }
        if self._smoothing_on:
            for key in _LATENT_OF_CODES.values():
                specs[key] = RolloutStateSpec(shape=(num_scalars,))
        return specs

    @property
    def _smoothing_on(self) -> bool:
        return float(getattr(self.config.fsq, "latent_ema_alpha", 1.0)) < 1.0

    def _smoothed_latent(
        self, tensordict: TensorDict, codes_key: str, codes: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """``(latent_to_decode, state_key)`` for one stream.

        With smoothing off this is the code itself and there is no state to
        carry. With it on the trunk decodes an exponential ramp toward the
        committed code, started from the *stored pre-step* latent — so a
        replayed forward reproduces it from the buffer, the encoder's
        straight-through gradient still flows (scaled by ``alpha``) on refresh
        rows only, and hold rows stay constants.
        """
        if not self._smoothing_on:
            return codes, None
        state_key = _LATENT_OF_CODES[codes_key]
        alpha = float(self.config.fsq.latent_ema_alpha)
        previous = tensordict[state_key]
        return previous + alpha * (codes - previous), state_key

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
        # Clear the smoothing ramp too: a latent carried over from another
        # episode would fade the *previous* intent into the new one for the
        # first few steps, which is the very carryover this call exists to
        # remove. Zero is what an episode reset leaves.
        for key in _LATENT_OF_CODES.values():
            latent = getattr(self, key, None)
            if not torch.is_tensor(latent):
                continue
            if env_ids is None:
                latent.zero_()
            else:
                latent[env_ids] = 0.0
        # ...and the inference hysteresis' "there is an intent worth keeping"
        # flag, or the very next refresh would put the flushed intent back.
        valid = getattr(self, "_inference_intent_valid", None)
        if torch.is_tensor(valid):
            if env_ids is None:
                valid.zero_()
            else:
                valid[env_ids] = False

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
        ladder_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Decode a code to the executed action, splitting the ladder rungs off.

        With ``ladder_offsets == (0,)`` this is exactly the pre-v11 function.
        With more rungs the trunk emits ``rungs * num_actions``; the returned
        tensor is rung 0 -- byte-identically the action every caller already
        expected -- and the full ``[B, rungs, A]`` block is published under
        ``ladder_key`` when one is asked for. Only the privileged decode passes
        a key, so the three later decodes in one forward cannot overwrite the
        tensor the auxiliary loss reads.
        """
        tensordict[VAE_LATENT_KEY] = codes
        tensordict = self._forward_module(self._trunk, tensordict, log_internals)
        out = tensordict[self._trunk.out_keys[0]]
        if self._num_rungs == 1:
            return out
        if out.shape[-1] % self._num_rungs != 0:
            raise ValueError(
                f"trunk emits {out.shape[-1]} outputs, which is not divisible by "
                f"{self._num_rungs} ladder rungs -- set the trunk's num_out to "
                "len(ladder_offsets) * number_of_actions in the experiment file"
            )
        rungs = out.reshape(*out.shape[:-1], self._num_rungs, -1)
        if ladder_key is not None:
            tensordict[ladder_key] = rungs
        return rungs[..., 0, :]

    def _write_chunk_phase(
        self, tensordict: TensorDict, refresh: torch.Tensor
    ) -> torch.Tensor:
        """Steps since this chunk's code was issued, as an index and a one-hot.

        ``fsq_steps_left`` is the *pre-step* counter, so a refresh row carries
        0 (or whatever an event interrupted) and the row one step later carries
        ``chunk_steps - 1``.  Phase is therefore
        ``0`` on a refresh and ``chunk_steps - steps_left`` otherwise, which
        spans exactly ``[0, chunk_steps)`` -- an event-triggered refresh lands
        on 0 like any other, and the counter is reset to ``chunk_steps - 1``
        behind it either way.

        Both outputs are pure functions of stored rollout state plus the stored
        event flag, exactly like ``_refresh_mask``, so replay reproduces them.
        """
        chunk = int(self.config.fsq.chunk_steps)
        steps_left = tensordict[FSQ_STEPS_LEFT_KEY].reshape(-1)
        phase = torch.where(
            refresh, torch.zeros_like(steps_left), chunk - steps_left
        ).clamp(0, chunk - 1)
        tensordict[FSQ_REFRESH_KEY] = refresh
        tensordict[FSQ_PHASE_INDEX_KEY] = phase
        if getattr(self.config.fsq, "chunk_phase_to_trunk", False):
            tensordict[FSQ_CHUNK_PHASE_KEY] = F.one_hot(
                phase.long(), num_classes=chunk
            ).to(steps_left.dtype)
        return phase

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

    @torch.no_grad()
    def _advance_stream_hysteretic(
        self,
        tensordict: TensorDict,
        held: torch.Tensor,
        context: torch.Tensor,
        refresh: torch.Tensor,
        greedy: bool,
    ) -> torch.Tensor:
        """``_advance_stream`` with the held intent given right of way.

        Deployment only (``forward_inference``). At a *timer* refresh the code
        the policy is already executing is put back unless the prior has
        stopped believing in it -- ``p(held | context) < intent_hysteresis``.
        A refresh triggered by a committed contact event is never suppressed,
        and neither is the first refresh after a reset or a
        ``flush_held_intent`` (there is no intent to keep), which is what stops
        the term from silently defeating an external goal change.

        Nothing here is declared rollout state: the training forward never
        calls it, so the replay contract is untouched.
        """
        new_codes = self._advance_stream(held, context, refresh, greedy)
        tau = float(getattr(self.config.fsq, "intent_hysteresis", 0.0) or 0.0)
        rows_total = refresh.shape[0]
        valid = getattr(self, "_inference_intent_valid", None)
        if valid is None or valid.shape[0] < rows_total:
            valid = torch.zeros(rows_total, dtype=torch.bool, device=refresh.device)
            self._inference_intent_valid = valid
        if tau > 0.0 and self.tokenization.num_prior_tokens == 1:
            event_key = self.config.fsq.event_flag_key
            event = (
                tensordict[event_key].reshape(rows_total) > 0.5
                if event_key and event_key in tensordict.keys()
                else torch.zeros_like(refresh)
            )
            rows = (refresh & ~event & valid[:rows_total]).nonzero(as_tuple=True)[0]
            if rows.numel() > 0:
                logits = self._ar_head.next_logits_from_context(
                    context[rows], token_indices=None
                )
                probs = torch.softmax(logits.float(), dim=-1)
                held_tokens = self._codes_to_tokens(held[rows]).reshape(-1, 1)
                p_held = probs.gather(1, held_tokens).reshape(-1)
                keep = p_held >= tau
                if bool(keep.any()):
                    new_codes[rows[keep]] = held[rows[keep]]
        valid[:rows_total] = valid[:rows_total] | refresh
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
        self._write_chunk_phase(tensordict, refresh)
        teacher_codes, codes_now = self._teacher_codes(
            tensordict, refresh, log_internals
        )
        teacher_latent, teacher_latent_key = self._smoothed_latent(
            tensordict, FSQ_TEACHER_CODES_KEY, teacher_codes
        )
        tensordict["privileged_action"] = self._decode(
            tensordict, teacher_latent, log_internals, ladder_key=LADDER_PRED_KEY
        )

        tensordict = self._forward_module(self._prior, tensordict, log_internals)
        context = self._ar_head.encode_context(tensordict)

        target_tokens = self._codes_to_tokens(teacher_codes.detach())
        tensordict[TARGET_LATENT_KEY] = target_tokens

        if replay:
            # Optimization pass: teacher-forced logits for the CE. The generated
            # streams are skipped -- re-running the sequential AR sampler would
            # be wasted work and could not reproduce the rollout's draw anyway.
            # But if the rollout stored the latent the sampled stream *decoded*,
            # decoding it again here is one extra trunk forward and gives the
            # deployable action a gradient path, which is what the DAgger action
            # loss needs.
            if FSQ_USED_PRIOR_LATENT_KEY in tensordict.keys():
                tensordict[PRIOR_ACTION_KEY] = self._decode(
                    tensordict, tensordict[FSQ_USED_PRIOR_LATENT_KEY], log_internals
                )
            logits = self._ar_head.forward_from_tokens(
                context, self.tokenization.one_hot_prior_tokens(target_tokens)
            )
            tensordict[LATENT_LOGITS_KEY] = logits
            tensordict["_fsq_codes_now"] = codes_now.detach()
            return tensordict

        steps_left = tensordict[FSQ_STEPS_LEFT_KEY].reshape(-1)
        prior_codes = self._advance_stream(
            tensordict[FSQ_PRIOR_CODES_KEY], context, refresh, greedy=False
        )
        mean_codes = self._advance_stream(
            tensordict[FSQ_MEAN_CODES_KEY], context, refresh, greedy=True
        )
        prior_latent, prior_latent_key = self._smoothed_latent(
            tensordict, FSQ_PRIOR_CODES_KEY, prior_codes
        )
        mean_latent, mean_latent_key = self._smoothed_latent(
            tensordict, FSQ_MEAN_CODES_KEY, mean_codes
        )

        tensordict["action"] = self._decode(tensordict, prior_latent, log_internals)
        tensordict["mean_action"] = self._decode(
            tensordict, mean_latent, log_internals
        )
        # Deployable-vs-teacher intent disagreement, in code space: keeps the
        # agent's env/action_gap/latent_residual_l2 diagnostic meaningful. The
        # raw codes, not the smoothed latents — the question is which intent
        # was committed to, not how fast it was faded in.
        tensordict[LATENT_MU_KEY] = prior_codes
        tensordict[PRIVILEGED_LATENT_MU_KEY] = teacher_codes.detach()
        # What the trunk consumed for `action`, post-generation and post-ramp,
        # so replay reproduces it exactly at any latent_ema_alpha.
        tensordict[FSQ_USED_PRIOR_LATENT_KEY] = prior_latent.detach()

        updates = {
            FSQ_TEACHER_CODES_KEY: teacher_codes,
            FSQ_PRIOR_CODES_KEY: prior_codes,
            FSQ_MEAN_CODES_KEY: mean_codes,
            FSQ_STEPS_LEFT_KEY: torch.where(
                refresh,
                torch.full_like(steps_left, float(self.config.fsq.chunk_steps - 1)),
                (steps_left - 1.0).clamp(min=0.0),
            ),
        }
        for key, latent in (
            (teacher_latent_key, teacher_latent),
            (prior_latent_key, prior_latent),
            (mean_latent_key, mean_latent),
        ):
            if key is not None:
                updates[key] = latent
        self._write_rollout_state(tensordict, updates)
        return tensordict

    def materialize(self, tensordict: TensorDict) -> TensorDict:
        """Create every lazy parameter, on BOTH forward paths.

        The framework materializes ``LazyLinear``/``RunningMeanStd`` with one
        dummy forward before wrapping the model in DDP, and the base
        implementation runs only the rollout path. That is not enough here:
        the AR head's *token encoder* is reached from ``next_logits_from_context``
        only when a prefix exists, and from ``forward_from_tokens`` only on
        replay. With ``num_prior_tokens >= 2`` generation supplies a prefix for
        tokens 2..N and the module is materialized by accident; with a **single**
        token (round 9's small code) generation never passes one, so the encoder
        stayed uninitialized and DDP refused the model with "Modules with
        uninitialized parameters can't be used with DistributedDataParallel".

        Running the replay path as well fixes it for any token count, and also
        covers the replay-only ``prior_action`` decode. The replay branch
        returns before ``_write_rollout_state``, so this second pass leaves the
        chunk buffers exactly as the first pass set them.
        """
        rollout = super().materialize(tensordict)
        self(rollout.clone())
        return rollout

    def forward_inference(self, tensordict: TensorDict) -> TensorDict:
        """Deployable path: sampled (or greedy, per config) intent + trunk.

        Emits only ``action`` — deliberately no ``mean_action``, so the probe
        and viz drivers (which prefer it) run the *sampled* stream and seeded
        attempt diversity is measurable.
        """
        self.read_rollout_state(tensordict)
        refresh = self._refresh_mask(tensordict)
        self._write_chunk_phase(tensordict, refresh)
        steps_left = tensordict[FSQ_STEPS_LEFT_KEY].reshape(-1)

        tensordict = self._forward_module(self._prior, tensordict, False)
        context = self._ar_head.encode_context(tensordict)
        prior_codes = self._advance_stream_hysteretic(
            tensordict,
            tensordict[FSQ_PRIOR_CODES_KEY],
            context,
            refresh,
            greedy=self.config.fsq.inference_argmax,
        )
        prior_latent, prior_latent_key = self._smoothed_latent(
            tensordict, FSQ_PRIOR_CODES_KEY, prior_codes
        )
        tensordict["action"] = self._decode(tensordict, prior_latent, False)
        tensordict[LATENT_MU_KEY] = prior_codes

        updates = {
            FSQ_PRIOR_CODES_KEY: prior_codes,
            FSQ_STEPS_LEFT_KEY: torch.where(
                refresh,
                torch.full_like(steps_left, float(self.config.fsq.chunk_steps - 1)),
                (steps_left - 1.0).clamp(min=0.0),
            ),
        }
        if prior_latent_key is not None:
            updates[prior_latent_key] = prior_latent
        self._write_rollout_state(tensordict, updates)
        return tensordict

    def _external_trunk_in_keys(self) -> list:
        """Trunk inputs the *environment* has to provide.

        The latent and the chunk phase are produced by this model inside
        ``forward``, so they must not appear in ``in_keys`` -- that list is what
        selects observations, and asking the env for them would fail at the
        first step.
        """
        produced = {VAE_LATENT_KEY, FSQ_CHUNK_PHASE_KEY}
        return [key for key in self._trunk.in_keys if key not in produced]

    def get_inference_in_keys(self) -> list:
        trunk_in_keys = self._external_trunk_in_keys()
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
        refresh = tensordict[FSQ_REFRESH_KEY].reshape(-1).bool()

        # The trained CE either covers every row (v6) or only the refresh rows
        # — the states at which the deployed prior is actually asked to choose
        # a code. A batch with no refresh row at all cannot happen in a real
        # rollout (the counter guarantees ~1/chunk_steps of them) but would
        # make the masked CE undefined, so fall back rather than emit a NaN.
        rows = torch.ones_like(refresh)
        if self.config.fsq.ce_refresh_rows_only and bool(refresh.any()):
            rows = refresh
        token_rows = rows.repeat_interleave(
            target.shape[-1] if target.dim() > 1 else 1
        )
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1])[token_rows],
            target.reshape(-1)[token_rows],
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
            refresh_frac = refresh.float().mean()

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
        log_dict.update(self._split_ce_diagnostics(logits, target, refresh, log_prefix))
        return loss + model_loss, log_dict

    @torch.no_grad()
    def _split_ce_diagnostics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        refresh: torch.Tensor,
        log_prefix: str,
    ) -> Dict:
        """Report the token CE separately on refresh rows and hold rows.

        The aggregate ``fsq_ce_loss`` mixes two different questions. On a **refresh**
        row the target is the code the encoder issues *now*, and that is the
        only kind of state at which the deployed prior is ever asked to choose —
        so its CE is the honest measure of how much genuine ambiguity the
        conditioning leaves. On a **hold** row the target is the code issued up
        to ``chunk_steps - 1`` steps earlier, from a context that has since
        moved on; its CE carries stale-label noise that has nothing to do with
        multimodality, and hold rows are ~87 % of the batch at
        ``chunk_steps = 8``.

        Diagnostic only — the loss is unchanged. Read
        ``fsq_ce_loss_refresh`` (not the aggregate) as "effective continuations
        per decision"; a large ``fsq_ce_loss_hold`` above it is the lag, not
        mode content. See ``notes/Student_v7_improvement_investigation.MD`` §5.3.
        """
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_target = target.reshape(-1)
        tokens_per_row = target.shape[-1] if target.dim() > 1 else 1
        row_of_token = refresh.repeat_interleave(tokens_per_row)
        out = {}
        for name, mask in (("refresh", row_of_token), ("hold", ~row_of_token)):
            if not bool(mask.any()):
                continue
            sub_logits, sub_target = flat_logits[mask], flat_target[mask]
            out[f"{log_prefix}/fsq_ce_loss_{name}"] = F.cross_entropy(
                sub_logits, sub_target, label_smoothing=self.config.fsq.label_smoothing
            )
            out[f"{log_prefix}/fsq_token_accuracy_{name}"] = (
                sub_logits.argmax(dim=-1) == sub_target
            ).float().mean()
        return out
