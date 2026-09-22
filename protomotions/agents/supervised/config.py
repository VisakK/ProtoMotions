# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for generic supervised rollout training."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from protomotions.agents.base_agent.config import BaseAgentConfig, BaseModelConfig
from protomotions.agents.common.supervision import SupervisionLossConfig


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
    ladder_loss_coeff: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the ACTION LADDER: an auxiliary MSE between the "
            "trunk's far rungs and the expert's action that many control steps "
            "ahead. 0 disables it and the trunk emits one action, which is "
            "every round before 11.\n\n"
            "The rungs themselves are declared by `model.fsq.ladder_offsets`; "
            "this is only their weight. Each rung is normalised by its own "
            "target variance before the mean, so an unequal valid-row count "
            "across horizons (a row is dropped when an episode boundary falls "
            "inside its horizon) cannot tilt the sum.\n\n"
            "Motivation, measured: the h=0 target is state-linear to "
            "R^2 = 0.999797, so there is 2.03e-4 of variance for the goal to "
            "explain at the horizon the imitation loss is evaluated on -- and "
            "at a bit-identical state the different-goal label excess at one "
            "step is +0.0023 rad with a 95 % CI that contains zero. At 24 "
            "steps it is +0.0633. Nothing in the goal channel can matter until "
            "the loss looks where the goal does.",
            "min": 0.0,
        },
    )
    ladder_balance_to_imitation: bool = field(
        default=False,
        metadata={
            "help": "Scale the action ladder so its magnitude is "
            "`ladder_loss_coeff` TIMES the imitation loss, rather than "
            "`ladder_loss_coeff` in absolute terms.\n\n"
            "Measured on the v11-A1 arm, this is not a nicety. Each rung is "
            "divided by its own target variance to make the rungs "
            "commensurable with each other -- but that leaves the ladder on a "
            "completely different scale from the raw imitation MSE, which is "
            "taken on normalised actions. At epoch 6,000 of A1 the imitation "
            "term was 3.735e-4 and the ladder term 1.228e-2: the ladder was "
            "**32.9x** the imitation loss, i.e. **97.05 %** of the gradient "
            "the trunk ever saw. The trunk spent the run optimising futures it "
            "never executes, its imitation MSE ended 1.30x v9's at the same "
            "epoch, and commanded-goal pose error regressed 0.0395 m "
            "(t = 2.93) against the matched baseline -- while the mechanism "
            "itself worked (code_ablation_gap ratio h15/h0 = 3.1x).\n\n"
            "Balancing also holds as training proceeds. A fixed coefficient "
            "drifts toward ladder dominance on its own, because the imitation "
            "MSE keeps descending its power law while the far rungs plateau -- "
            "the far horizons are irreducibly harder, which is the entire "
            "reason they were added.",
        },
    )
