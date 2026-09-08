# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for the FSQ-C MaskedMimic student (chunked discrete-intent codes)."""

from dataclasses import dataclass, field
from typing import Optional

from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    ModuleContainerConfig,
)
from protomotions.agents.common.latent import LATENT_KEY, LATENT_LOGITS_KEY


@dataclass
class FSQCEScheduleConfig:
    """Linear ramp of the token cross-entropy coefficient.

    The CE trains only the prior branch (context transformer + AR head); the
    imitation MSE trains only the encoder/trunk autoencoder — the parameter
    sets are disjoint, so the ramp exists to (a) let the code space stabilize
    before the AR head fits it and (b) keep early CE gradients from eating the
    shared global-norm clip budget while the autoencoder is still learning.
    """

    init_ce_coeff: float = field(
        default=0.0, metadata={"help": "CE coefficient before start_epoch.", "min": 0.0}
    )
    end_ce_coeff: float = field(
        default=1.0, metadata={"help": "CE coefficient after end_epoch.", "min": 0.0}
    )
    start_epoch: int = field(
        default=100, metadata={"help": "Epoch the ramp begins.", "min": 0}
    )
    end_epoch: int = field(
        default=600, metadata={"help": "Epoch the ramp ends.", "min": 0}
    )


@dataclass
class FSQIntentConfig:
    """The discrete-intent bottleneck and its chunked (held-code) semantics.

    The code is refreshed — re-encoded on the teacher path, re-generated on the
    prior paths — when any of these fire, in lockstep for all streams:

    * the per-env chunk counter expires (``chunk_steps`` control steps),
    * the measured contact-event tracker commits a new configuration segment
      (``event_flag_key`` observation, 1.0 on the commit step), or
    * an episode reset zeroes the counter (framework rollout-state reset).

    Between refreshes the code is held constant while the trunk keeps running
    per-step on live state: closed-loop stabilization inside a committed intent.
    """

    num_fsq_levels: int = field(
        default=5, metadata={"help": "Quantization levels per scalar (odd)."}
    )
    num_fsq_scalars: int = field(
        default=16, metadata={"help": "FSQ scalar code dimensions."}
    )
    fsq_scalars_per_prior_token: int = field(
        default=4,
        metadata={"help": "Scalars packed into one AR prior token (mixed radix)."},
    )
    chunk_steps: int = field(
        default=8,
        metadata={
            "help": "Control steps a code is held before a scheduled refresh "
            "(8 @ 30 Hz = 0.27 s, the corpus's stay-command median scale).",
            "min": 1,
        },
    )
    event_flag_key: Optional[str] = field(
        default="contact_event_flag",
        metadata={
            "help": "Observation key carrying the per-env 'contact segment "
            "committed this step' flag used as a refresh trigger. None "
            "disables event-triggered refresh (counter/reset only)."
        },
    )
    intent_hysteresis: float = field(
        default=0.0,
        metadata={
            "help": "INFERENCE ONLY. At a *timer* refresh, keep the code the "
            "policy is already holding unless the AR prior gives it less than "
            "this probability. 0.0 disables and reproduces every trained "
            "round exactly. Motivation (notes/Student_v9_round9_diagnosis.MD "
            "§5.1): the chunk clock re-draws the intent 3.75 times a second, "
            "so a 12 s commanded hold is 45 independent nucleus draws and "
            "survives only if none of them commits to a departure -- the "
            "measured per-draw error is ~10x higher at an ambiguous goal "
            "(standing) than at an unambiguous one (side plank). An event "
            "refresh is never suppressed: a committed contact change is a "
            "real reason to reconsider. Single-token codes only (v9's 4 "
            "scalars x 1 token); with several tokens the held code's joint "
            "probability is not one softmax and the term is ignored.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    latent_ema_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Exponential smoothing applied to the code *before* it "
            "reaches the trunk, per stream: latent <- latent + alpha * (code - "
            "latent). 1.0 hands the raw code straight through (v6). Lower "
            "values ramp a newly committed intent in over ~1/alpha steps, "
            "which is what removes the step change the round-6 measurements "
            "found sitting at the chunk clock (30/8 = 3.75 Hz) in both the "
            "sampled and the greedy stream — see "
            "notes/Student_v7_improvement_investigation.MD §10.3. The smoothed "
            "latent is declared rollout state, so the loss trains through the "
            "ramp and replay reproduces it exactly.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    chunk_phase_to_trunk: bool = field(
        default=False,
        metadata={
            "help": "Feed the trunk a one-hot of how many steps have passed "
            "since the intent code was issued (width = chunk_steps). The trunk "
            "otherwise has no clock, so a held code can only express a constant "
            "offset on the action -- never a short program. Round 7_1 §6.2. "
            "False reproduces v6/v7. Note this is the opposite call from the AR "
            "head, which is deliberately given no phase input (v7 §1.3): the "
            "head is only ever queried at a refresh, where the phase is "
            "constant, while the trunk runs at every phase."
        },
    )
    ce_refresh_rows_only: bool = field(
        default=False,
        metadata={
            "help": "Take the token cross-entropy over refresh rows only. "
            "Refresh states are the only ones at which the deployed prior is "
            "ever asked to choose a code; a hold row's target is the code "
            "issued up to chunk_steps-1 steps earlier, under a context that "
            "has since moved on, and those rows are ~87 % of a batch at "
            "chunk_steps=8. False reproduces v6's unmasked CE. The split "
            "diagnostics are logged either way."
        },
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Sampling temperature for prior tokens."}
    )
    top_p: float = field(
        default=0.9, metadata={"help": "Nucleus threshold for prior tokens."}
    )
    inference_argmax: bool = field(
        default=False,
        metadata={
            "help": "forward_inference (probes/viz/query) decodes greedy tokens "
            "instead of nucleus samples. The in-training evaluator is argmax "
            "regardless: it prefers 'mean_action', which is always greedy."
        },
    )
    label_smoothing: float = field(
        default=0.01, metadata={"help": "Label smoothing for the token CE."}
    )
    ce_schedule: FSQCEScheduleConfig = field(
        default_factory=FSQCEScheduleConfig,
        metadata={"help": "Token cross-entropy coefficient ramp."},
    )


@dataclass
class FSQMaskedMimicModelConfig(BaseModelConfig):
    """FSQ-C MaskedMimic student model configuration.

    Same three-container layout as the Gaussian student, with the bottleneck
    swapped: ``encoder`` ends in one raw-code head (width ``num_fsq_scalars``),
    ``prior`` ends at the transformer summary token (no mu/logvar heads), and
    ``ar_head`` is the categorical autoregressive prior over packed FSQ tokens
    conditioned on that summary. ``trunk`` is unchanged and consumes the codes
    through ``vae_latent``.
    """

    _target_: str = (
        "protomotions.agents.supervised.fsq_masked_mimic_model.FSQMaskedMimicModel"
    )

    encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Privileged encoder ending in the raw FSQ code head."},
    )
    prior: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Deployable context network ending at 'transformer_out'."},
    )
    ar_head: DiscreteAutoregressiveTransformerConfig = field(
        default_factory=lambda: DiscreteAutoregressiveTransformerConfig(
            token_key="fsq_target_tokens_in",
            logits_key=LATENT_LOGITS_KEY,
            generated_tokens_key=LATENT_KEY,
            num_tokens=0,
            vocab_size=0,
        ),
        metadata={
            "help": "Autoregressive token prior; num_tokens/vocab_size are "
            "resolved from the FSQ settings at model construction."
        },
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Code-to-action decoder trunk (reads 'vae_latent')."},
    )
    fsq: FSQIntentConfig = field(
        default_factory=FSQIntentConfig,
        metadata={"help": "FSQ bottleneck and chunk semantics."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for supervised training."},
    )
    ar_head_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for the AR token prior (context encoder + "
            "AR transformer), as its own optimizer parameter group. None "
            "shares the single rate above, which is what v6 did — and the "
            "in-repo GPC prior recipe uses AdamW 1e-4 for the same kind of "
            "head against this model's 2e-5 (round 6 §9.2 named the shared "
            "optimizer as the prime suspect for its slow convergence).",
            "min": 0.0,
        },
    )
    ar_head_weight_decay: Optional[float] = field(
        default=None,
        metadata={
            "help": "Weight decay for the AR-prior parameter group. Only "
            "meaningful with a decoupled-decay optimizer (torch.optim.AdamW); "
            "None inherits the shared setting.",
            "min": 0.0,
        },
    )
    encoder_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for the privileged encoder, as its own "
            "parameter group. The straight-through gradient reaches the "
            "encoder only on refresh rows — ~13 % of samples at chunk_steps=8, "
            "a 7.6x cut in its learning signal versus the per-step v5 "
            "posterior — and latent smoothing scales what does arrive by "
            "alpha. Raising this rate is the direct offset. None shares the "
            "rate above.",
            "min": 0.0,
        },
    )
