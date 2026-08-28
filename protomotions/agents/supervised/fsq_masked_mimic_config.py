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
