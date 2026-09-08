# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 contact-graph student with a chunked FSQ intent bottleneck (FSQ-C).

This is :mod:`contact_graph_transformer` (the v5 recipe — same student44h
corpus, graph, experts, DAgger, evaluation) with the Gaussian VAE head swapped
for the design in ``notes/Student_multimodal_head_investigation.MD`` §3.3,
motivated by round 5's finding that the CVAE converges to deterministic BC and
mode-averages at commitment points:

**1. The latent is a finite-scalar-quantized intent code, held for a chunk.**
The privileged encoder ends in ``num_fsq_scalars`` raw scalars, straight-
through-quantized into a code that is **re-issued only when a chunk boundary
fires** — every ``chunk_steps`` control steps, on a committed contact event
(the ContactEventTracker's debounced make/break, surfaced as the new
``contact_event_flag`` observation), or at episode reset. Between refreshes
the code is constant while the trunk keeps running per-step on live state:
closed-loop stabilization inside a committed intent. So the encoder needs to
see *which continuation is coming*, not just the next frame — the dense
reference future is widened from one step to a strided ~0.5 s window
(``--encoder-future-steps``), encoder-only; the experts' deep-copied
``expert_mimic_target_poses`` is pinned to its original single step so their
frozen observation contract stays byte-identical.

**2. The deployable prior is a categorical autoregressive head.** The
15-token context transformer is unchanged but now ends at its summary token;
a small AR transformer over the packed FSQ tokens is trained with
cross-entropy against the teacher codes (coefficient ramped by
``--ce-start/end-epoch``; the CE and the imitation MSE train disjoint
parameter sets). At refresh points the prior *samples* (nucleus) — sampling,
not regression, picks which continuation to commit to. Three streams share
one chunk clock: teacher codes → ``privileged_action`` (ctrl envs + MSE),
sampled tokens → ``action`` (DAgger block, probes, queries), greedy tokens →
``mean_action`` (what the evaluators prefer, so ``eval/success_rate`` stays
deterministic and comparable with v4/v5).

Train (defaults reproduce the v5 recipe apart from the head)::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/masked_mimic/contact_graph_fsq_transformer.py \
      --experiment-name smpl_yogi_contact_graph_student_s2_v6_fsq \
      --motion-file data/smpl/yoga_yogi_student44h.pt \
      --contact-graph-file data/smpl/yoga_contact_graph_student44h/contact_graph.pt \
      --motion-expert-file data/smpl/yoga_yogi_student44h.experts.json \
      --expert-model-paths <easy128> <hard29> <singleleg14> \
      --num-envs 1024 --batch-size 8192 --headless True --use-wandb

or ``data/scripts/run_student_distill_v6.sh``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.supervised.multi_expert import (
    MultiExpertMaskedMimicAgentConfig,
)


def _load_base_module():
    """Load the sibling Gaussian-student experiment by file path.

    ``examples/experiments/masked_mimic`` is not a package, so the base module
    is loaded the same way the training harness loads experiment files. Its
    import has no side effects (constants and functions only).
    """
    base_path = Path(__file__).resolve().parent / "contact_graph_transformer.py"
    spec = importlib.util.spec_from_file_location(
        "contact_graph_transformer_base", base_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

# Strided dense-future window for the privileged encoder, in control steps at
# 30 Hz: 0.033 s .. 0.5 s. Step 1 MUST come first: the experts' deep-copied
# target-poses observation selects future-buffer *positions*, and pinning it
# to position 1 only reproduces their frame-offset-1 contract if offset 1 sits
# in position 1 (env_config asserts this). Four strided steps rather than six:
# each step is a full 573-float max-coords block, and the 6-step window's extra
# buffer + optimization activations tipped the 24 GB card into OOM at the v5
# batch size (first smoke); 4 steps still span the 0.5 s commitment window.
ENCODER_FUTURE_STEPS = [1, 5, 10, 15]

NUM_GOAL_STEPS = base.NUM_GOAL_STEPS
NUM_HISTORY_EVENT_TOKENS = base.NUM_HISTORY_EVENT_TOKENS
NUM_HISTORICAL_CONDITIONED_STEPS = base.NUM_HISTORICAL_CONDITIONED_STEPS
STATE_KEYS = base.STATE_KEYS
UNNORMALIZED_STATE_KEYS = base.UNNORMALIZED_STATE_KEYS

# The per-step "a contact segment just committed" flag, consumed directly by
# the FSQ model as a chunk-refresh trigger (never fed to a network container).
CONTACT_EVENT_FLAG_KEY = "contact_event_flag"


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Everything the base experiment takes, plus the FSQ-C knobs."""
    base.additional_experiment_arguments(parser)
    parser.add_argument(
        "--encoder-future-steps",
        type=int,
        nargs="+",
        default=list(ENCODER_FUTURE_STEPS),
        help="Frame offsets of the encoder's dense reference-future window "
             "(control steps; must start with 1 — see ENCODER_FUTURE_STEPS).",
    )
    parser.add_argument(
        "--fsq-levels", type=int, default=5,
        help="Quantization levels per FSQ scalar (odd).",
    )
    parser.add_argument(
        "--fsq-scalars", type=int, default=16,
        help="Number of FSQ scalar code dimensions.",
    )
    parser.add_argument(
        "--fsq-scalars-per-token", type=int, default=4,
        help="FSQ scalars packed into one AR prior token "
             "(vocab = levels ** this; must divide --fsq-scalars).",
    )
    parser.add_argument(
        "--chunk-steps", type=int, default=8,
        help="Control steps an intent code is held before a scheduled refresh "
             "(8 @ 30 Hz = 0.27 s).",
    )
    parser.add_argument(
        "--event-refresh",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=True,
        help="Also refresh the intent code when the contact-event tracker "
             "commits a new configuration segment.",
    )
    parser.add_argument(
        "--fsq-temperature", type=float, default=1.0,
        help="Sampling temperature for the AR prior tokens.",
    )
    parser.add_argument(
        "--fsq-top-p", type=float, default=0.9,
        help="Nucleus threshold for the AR prior tokens.",
    )
    parser.add_argument(
        "--fsq-inference-argmax",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=False,
        help="forward_inference (probes/viz/queries) decodes greedy tokens "
             "instead of nucleus samples. The evaluator is greedy regardless.",
    )
    parser.add_argument(
        "--ce-start-epoch", type=int, default=100,
        help="Epoch the token cross-entropy coefficient starts ramping 0 -> 1.",
    )
    parser.add_argument(
        "--ce-end-epoch", type=int, default=600,
        help="Epoch the token cross-entropy coefficient reaches 1.",
    )


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator
# The base override function resolves its own relative imports against the
# base module's file location, so re-exporting it directly is safe.
apply_inference_overrides = base.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """The base environment, with the encoder future window and event flag.

    Two mutations on top of :func:`base.env_config`:

    1. The shared mimic future buffer is widened from the experts' single step
       to ``--encoder-future-steps``. The student's own ``mimic_target_poses``
       (encoder-only) consumes the whole buffer automatically; the experts'
       deep-copied component is pinned to the *positions* of its original
       consecutive frame offsets, keeping the frozen observation contract
       byte-identical. Position selection is what makes the offset-1-first
       ordering of the window a hard requirement, asserted here.
    2. ``contact_event_flag`` joins the observations: the tracker's per-step
       committed-segment flag, read by the FSQ model as a refresh trigger.
    """
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs import compute_contact_event_flag

    cfg = base.env_config(robot_cfg, args)
    ctrl = cfg.control_components["contact_graph"]

    window = [int(s) for s in getattr(args, "encoder_future_steps", ENCODER_FUTURE_STEPS)]
    if len(window) == 0 or sorted(set(window)) != window:
        raise ValueError(
            f"--encoder-future-steps must be strictly increasing, got {window}"
        )

    # After base.env_config the control's future_steps is the experts' need:
    # an int N meaning consecutive offsets 1..N (the base raise logic compares
    # ints and would have failed loudly on anything else).
    expert_future = ctrl.future_steps
    if not isinstance(expert_future, int):
        raise ValueError(
            "expected the base experiment to leave an int future_steps "
            f"(the experts' consecutive-offset need), got {expert_future!r}"
        )
    expert_offsets = list(range(1, expert_future + 1))
    if window[: len(expert_offsets)] != expert_offsets:
        raise ValueError(
            f"--encoder-future-steps must start with the experts' consecutive "
            f"offsets {expert_offsets} (expert observations select future-buffer "
            f"positions, so the offsets must occupy the leading positions); "
            f"got {window}"
        )

    expert_component = cfg.observation_components.get("expert_mimic_target_poses")
    if expert_component is not None:
        # Pin the expert copy to the first `expert_future` buffer positions —
        # exactly the frames the expert was trained on. Without this it would
        # consume the whole widened buffer and change shape, and the frozen
        # actor would refuse to load against it.
        expert_component.static_params["future_steps"] = expert_future
    ctrl.future_steps = window

    cfg.observation_components[CONTACT_EVENT_FLAG_KEY] = MdpComponent(
        compute_func=compute_contact_event_flag,
        dynamic_vars={"event_commit": EnvContext.contact_goal.event_commit},
    )
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MultiExpertMaskedMimicAgentConfig:
    """The v5 agent with the FSQ-C model in place of the Gaussian VAE."""
    from protomotions.agents.common.config import (
        DiscreteAutoregressiveTransformerConfig,
        MLPLayerConfig,
        MLPWithConcatConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ModuleOperationReshapeConfig,
        ObsProcessorConfig,
        TransformerConfig,
    )
    from protomotions.agents.common.latent import LATENT_KEY, LATENT_LOGITS_KEY
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.supervised.fsq_masked_mimic_config import (
        FSQCEScheduleConfig,
        FSQIntentConfig,
        FSQMaskedMimicModelConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    transformer_token_size = 512
    transformer_encoder_widths = 256
    num_fsq_scalars = int(getattr(args, "fsq_scalars", 16))

    def state_normalizers():
        return [
            ObsProcessorConfig(
                in_keys=[key],
                out_keys=[f"{key}_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            )
            for key in STATE_KEYS
        ]

    state_norm_keys = [f"{key}_norm" for key in STATE_KEYS]

    # Privileged encoder: identical to the v5 encoder trunk, ending in one raw
    # FSQ-code head instead of the mu/logvar pair. Its dense reference future
    # is now the widened ~0.5 s window (env_config), which is what lets the
    # code carry *which continuation* rather than just the next frame.
    encoder_config = ModuleContainerConfig(
        in_keys=STATE_KEYS
        + UNNORMALIZED_STATE_KEYS
        + [
            "mimic_target_poses",
            "masked_mimic_target_poses",
            "masked_mimic_target_bodies_masks",
            "masked_mimic_target_times",
            "masked_mimic_target_poses_masks",
            "contact_goal_obs",
            "contact_history_obs",
        ],
        out_keys=["encoder_codes_raw"],
        models=[
            *state_normalizers(),
            ObsProcessorConfig(
                in_keys=["mimic_target_poses"],
                out_keys=["mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["masked_mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["masked_mimic_target_times_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=state_norm_keys
                + UNNORMALIZED_STATE_KEYS
                + [
                    "mimic_target_poses_norm",
                    "masked_mimic_target_poses_norm",
                    "masked_mimic_target_bodies_masks",
                    "masked_mimic_target_times_norm",
                    "masked_mimic_target_poses_masks",
                    "contact_goal_obs",
                    "contact_history_obs",
                ],
                out_keys=["encoder_trunk_out"],
                num_out=512,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(5)
                ],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_codes_raw"],
                num_out=num_fsq_scalars,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    # Deployable context: the v5 15-token transformer, unchanged, ending at
    # its summary token — the mu/logvar heads are gone; the AR head below is
    # what turns the summary into a (sampled) latent.
    prior_config = ModuleContainerConfig(
        in_keys=STATE_KEYS
        + UNNORMALIZED_STATE_KEYS
        + [
            "masked_mimic_target_poses",
            "masked_mimic_target_masks",
            "masked_mimic_target_times",
            "masked_mimic_target_poses_masks",
            "contact_goal_obs",
            "contact_history_obs",
            "contact_history_masks",
            "historical_pose_obs",
        ],
        out_keys=["transformer_out"],
        models=[
            *state_normalizers(),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["target_poses_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_GOAL_STEPS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_masks"],
                out_keys=["target_masks_seq"],
                normalize_obs=False,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_GOAL_STEPS, -1]
                    )
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["target_times_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_GOAL_STEPS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["contact_goal_obs"],
                out_keys=["contact_goal_seq"],
                normalize_obs=False,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_GOAL_STEPS, -1]
                    )
                ],
            ),
            ObsProcessorConfig(
                in_keys=["contact_history_obs"],
                out_keys=["contact_history_seq"],
                normalize_obs=False,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_HISTORY_EVENT_TOKENS, -1]
                    )
                ],
            ),
            ObsProcessorConfig(
                in_keys=["historical_pose_obs"],
                out_keys=["historical_pose_obs_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=state_norm_keys + UNNORMALIZED_STATE_KEYS,
                out_keys=["current_state_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[
                    MLPLayerConfig(units=transformer_encoder_widths, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "target_poses_seq",
                    "target_masks_seq",
                    "target_times_seq",
                    "contact_goal_seq",
                ],
                out_keys=["masked_mimic_target_poses_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[
                    MLPLayerConfig(units=transformer_encoder_widths, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_GOAL_STEPS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["historical_pose_obs_seq"],
                out_keys=["historical_pose_obs_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[
                    MLPLayerConfig(units=transformer_encoder_widths, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["contact_history_seq"],
                out_keys=["contact_history_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[
                    MLPLayerConfig(units=transformer_encoder_widths, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_HISTORY_EVENT_TOKENS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            TransformerConfig(
                in_keys=[
                    "current_state_token",
                    "masked_mimic_target_poses_token",
                    "historical_pose_obs_token",
                    "contact_history_token",
                    "masked_mimic_target_poses_masks",
                    "contact_history_masks",
                ],
                out_keys=["transformer_out"],
                transformer_token_size=transformer_token_size,
                latent_dim=transformer_token_size,
                input_and_mask_mapping={
                    "masked_mimic_target_poses_token": "masked_mimic_target_poses_masks",
                    "contact_history_token": "contact_history_masks",
                },
                output_activation="relu",
            ),
        ],
    )

    # Small categorical AR head over the packed FSQ tokens, conditioned on the
    # context transformer's summary. num_tokens/vocab_size are resolved by the
    # model from the FSQ settings.
    ar_head_config = DiscreteAutoregressiveTransformerConfig(
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
                    num_out=512,
                    layers=[MLPLayerConfig(units=512, activation="gelu")],
                )
            ],
        ),
        d_model=512,
        num_heads=4,
        num_layers=2,
        ff_size=1024,
        # Zero so the teacher-forced CE is identical in train and eval modes —
        # the context transformer upstream is the regularized part.
        dropout=0.0,
        activation="gelu",
        num_tokens=0,
        vocab_size=0,
    )

    trunk_config = ModuleContainerConfig(
        in_keys=STATE_KEYS
        + UNNORMALIZED_STATE_KEYS
        + ["previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            *state_normalizers(),
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["previous_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=state_norm_keys
                + UNNORMALIZED_STATE_KEYS
                + ["previous_actions_norm", "vae_latent"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(3)
                ],
            ),
        ],
    )

    fsq_config = FSQIntentConfig(
        num_fsq_levels=int(getattr(args, "fsq_levels", 5)),
        num_fsq_scalars=num_fsq_scalars,
        fsq_scalars_per_prior_token=int(getattr(args, "fsq_scalars_per_token", 4)),
        chunk_steps=int(getattr(args, "chunk_steps", 8)),
        event_flag_key=(
            CONTACT_EVENT_FLAG_KEY
            if getattr(args, "event_refresh", True)
            else None
        ),
        temperature=float(getattr(args, "fsq_temperature", 1.0)),
        top_p=float(getattr(args, "fsq_top_p", 0.9)),
        inference_argmax=bool(getattr(args, "fsq_inference_argmax", False)),
        ce_schedule=FSQCEScheduleConfig(
            start_epoch=int(getattr(args, "ce_start_epoch", 100)),
            end_epoch=int(getattr(args, "ce_end_epoch", 600)),
        ),
    )

    model_config = FSQMaskedMimicModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        ar_head=ar_head_config,
        trunk=trunk_config,
        fsq=fsq_config,
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = MimicEvaluatorConfig(
        evaluation_components={
            "gt_error": gt_error_factory(threshold=0.25),
            "gr_error": gr_error_factory(),
            "max_joint_error": max_joint_error_factory(),
        },
        # Fixed sampling — standing policy since the v3 controlled rerun proved
        # the adaptive curriculum caused the v2 collapse. Baked in here rather
        # than left to launch-script overrides.
        motion_weights_rules=MotionWeightsRulesConfig(
            motion_weights_update_success_discount=1.0,
            motion_weights_update_failure_discount=1.0,
        ),
        save_predicted_motion_lib_every=None,
    )

    viz_every = int(getattr(args, "viz_sequences_every", 0) or 0)
    sequence_viz = None
    if viz_every > 0:
        from protomotions.agents.evaluators.sequence_viz import SequenceVizConfig

        sequence_viz = SequenceVizConfig(
            viz_every=viz_every,
            plan_files=list(getattr(args, "viz_plan_files", []) or []),
            num_sequences=int(getattr(args, "viz_num_sequences", 10) or 10),
            max_seconds=float(getattr(args, "viz_max_seconds", 20.0) or 20.0),
            log_scalars=bool(getattr(args, "viz_log_scalars", True)),
        )

    expert_paths = base._expert_paths(args)
    return MultiExpertMaskedMimicAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        # Round-2 lesson: the eval optimum must actually reach disk.
        save_epoch_checkpoint_every=500,
        evaluator=evaluator_config,
        expert_model_path=None,
        expert_model_paths=expert_paths,
        motion_expert_file=getattr(args, "motion_expert_file", None),
        sequence_viz=sequence_viz,
    )
