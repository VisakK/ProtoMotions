# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for generic supervised rollout training."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from protomotions.agents.base_agent.config import BaseAgentConfig, BaseModelConfig
from protomotions.agents.common.supervision import SupervisionLossConfig
from protomotions.agents.evaluators.sequence_viz import SequenceVizConfig


class RolloutActor(Enum):
    """Policy source used to step the environment during supervised rollout collection."""

    STUDENT = "student"
    EXPERT = "expert"

    @classmethod
    def from_str(cls, value: str) -> "RolloutActor":
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            valid = [member.value for member in cls]
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. Valid values are: {valid}"
            )


@dataclass
class SupervisedAgentConfig(BaseAgentConfig):
    """Generic supervised imitation agent configuration.

    Experiment files choose the rollout actor, optional external expert
    checkpoint, and supervised loss keys. The agent loop stays independent of
    the specific student model.
    """

    _target_: str = "protomotions.agents.supervised.agent.SupervisedAgent"

    model: BaseModelConfig = field(
        default_factory=BaseModelConfig,
        metadata={"help": "Model configuration."},
    )
    expert_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional checkpoint for an external expert policy."},
    )
    rollout_actor: RolloutActor = field(
        default=RolloutActor.STUDENT,
        metadata={
            "help": "Policy used for collecting rollout actions."
        },
    )
    prior_rollout_fraction: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of envs stepped with the deployable prior's action "
            "instead of the privileged action (DAgger-style rollout mixture, so "
            "the deployed policy's induced states receive expert labels). "
            "0 disables and reproduces the pure privileged rollout.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    prior_rollout_start_epoch: int = field(
        default=0,
        metadata={
            "help": "No envs are prior-driven before this epoch, so early "
            "training still bootstraps on the privileged action.",
            "min": 0,
        },
    )
    prior_rollout_ramp_epochs: int = field(
        default=0,
        metadata={
            "help": "Linearly ramp the prior-driven fraction from 0 to "
            "prior_rollout_fraction over this many epochs after the start "
            "epoch. 0 switches on at full fraction immediately.",
            "min": 0,
        },
    )
    loss: SupervisionLossConfig = field(
        default_factory=SupervisionLossConfig,
        metadata={"help": "Supervised loss over model outputs and labels."},
    )
    action_rate_loss_coeff: float = field(
        default=0.0,
        metadata={
            "help": "Weight of a teacher-rate-matching term added to the "
            "imitation loss: ||(a_student - a_prev) - (a_expert - "
            "a_expert_prev)||^2. Unlike a bare ||da||^2 smoothness penalty it "
            "costs nothing for a transition the expert also makes fast, so it "
            "opposes jitter without taxing a legitimate kick-up; and unlike "
            "the `action_smoothness` reward component it is actually "
            "differentiated (rewards never are, in a supervised agent). "
            "Requires an external expert -- it is what supplies the previous "
            "expert action. 0 disables and reproduces the loss exactly.",
            "min": 0.0,
        },
    )
    action_rate_free_steps: int = field(
        default=0,
        metadata={
            "help": "Steps at the START of each intent chunk on which the "
            "teacher-rate term is NOT charged. 0 charges every row (v7). 1 "
            "skips the refresh row itself -- the only row whose action "
            "increment spans a code change, and therefore the only row where "
            "the rate term opposes a commitment rather than jitter. Round 7_1 "
            "§2.2 measured that suppressing the chunk-clock artifact and "
            "suppressing commitment were the same operation; this separates "
            "them in time. Requires the FSQ model, which is what publishes the "
            "chunk phase.",
            "min": 0,
        },
    )
    dagger_action_loss_coeff: float = field(
        default=0.0,
        metadata={
            "help": "Weight of an imitation MSE on the DEPLOYABLE (sampled) "
            "action, restricted to the rows the prior itself drove. The trunk "
            "is otherwise trained only through `privileged_action`, i.e. only "
            "on codes the ENCODER produced -- so at deployment it decodes a "
            "code it has never had a gradient for on ~97 % of refreshes "
            "(round 7_1 §3.2, `fsq_full_match` 0.03-0.18). On a prior-driven "
            "row the sampled action was actually applied and the expert "
            "labelled the resulting state, which is exactly what DAgger asks "
            "the deployable policy to match; on a privileged-driven row it "
            "would just be pressure for every code to decode to one action. "
            "Keep it small -- it trades code diversity for off-code "
            "robustness. 0 disables it (every round before 9).",
            "min": 0.0,
        },
    )
    sequence_viz: Optional[SequenceVizConfig] = field(
        default=None,
        metadata={
            "help": "In-training stick-figure videos of goal sequences, "
            "rendered with matplotlib and logged to wandb. None disables."
        },
    )
