# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for evaluators."""

from typing import List, Any, Dict, Optional, Union
from dataclasses import dataclass, field

from protomotions.envs.mdp_component import MdpComponent


@dataclass
class EvaluatorConfig:
    """Configuration for base evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    evaluation_components: Dict[str, MdpComponent] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of MdpComponent evaluation metrics for success/failure tracking."}
    )
    max_eval_steps: int = field(
        default=600,
        metadata={"help": "Maximum steps per evaluation episode.", "min": 1}
    )
    eval_metrics_every: Optional[int] = field(
        default=200,
        metadata={"help": "Evaluate metrics every N epochs. None = disabled.", "min": 1}
    )


@dataclass
class MotionWeightsRulesConfig:
    """Configuration for motion weights update rule."""

    motion_weights_update_success_discount: float = field(
        default=0.999,
        metadata={"help": "Discount factor for successful motion weights.", "min": 0.0, "max": 1.0}
    )
    motion_weights_update_failure_discount: float = field(
        default=0.999,
        metadata={"help": "Discount for failed motions. 0 = set weight straight to 1.", "min": 0.0, "max": 1.0}
    )
    min_motion_weight: Union[float, str] = field(
        default="1/num_motions",
        metadata={"help": "Minimum weight for any motion. '1/num_motions' or float value."}
    )


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    save_predicted_motion_lib_every: Optional[int] = field(
        default=3,
        metadata={"help": "Save pred_motion_lib every M evals. None = disabled.", "min": 1}
    )
    motion_weights_rules: MotionWeightsRulesConfig = field(
        default_factory=MotionWeightsRulesConfig,
        metadata={"help": "Rules for updating motion sampling weights."}
    )
    eval_action_ema_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "EMA smoothing factor for actions during evaluation only. "
                "Simulates deployment low-pass filtering. "
                "a_applied = alpha * a_policy + (1-alpha) * a_prev. "
                "None = disabled (raw actions). Typical values: 0.5-0.8."
                "Smaller alpha = more smoothing."
            ),
            "min": 0.0,
            "max": 1.0,
        }
    )


@dataclass
class HoldCurriculumConfig:
    """Uniform + performance curriculum driven by ``HoldCurriculumEvaluator``."""

    uniform_fraction: float = field(
        default=0.8,
        metadata={"help": "Share of sampling mass spread uniformly over motions.", "min": 0.0, "max": 1.0},
    )
    score_ema_keep: float = field(
        default=0.5,
        metadata={"help": "EMA weight on the previous per-motion score (smooths single-rollout noise).",
                  "min": 0.0, "max": 1.0},
    )
    priority_power: float = field(
        default=1.0, metadata={"help": "Prioritized mass ~ (1 - score) ** power + eps."}
    )
    priority_eps: float = field(
        default=1e-3, metadata={"help": "Floor added to every motion's priority."}
    )
    track_weight: float = field(
        default=0.5, metadata={"help": "score = w * p_track + (1 - w) * p_hold."}
    )
    track_fail_m: float = field(
        default=0.5, metadata={"help": "Max-body error counted as a tracking failure (training gate)."}
    )
    pose_threshold_m: float = field(
        default=0.15, metadata={"help": "6-body best-yaw distance under which a hold frame is attained."}
    )
    foot_down_z: float = field(
        default=0.08, metadata={"help": "A support zone is on the floor below this body-origin height."}
    )
    unloaded_ref_min_z: float = field(
        default=0.15,
        metadata={"help": "A zone must stay off the floor in a hold when the reference keeps it above this."},
    )
    event_dilate_frames: int = field(
        default=7,
        metadata={"help": "Event-aware family-hold metric (logged only): each support violation is "
                          "dilated this many frames either side, so a limb that touches down several "
                          "times a second never counts as lifted (expert_revist/ft_b_support/report.MD §4)."},
    )
    report_exclude_motions: List[str] = field(
        default_factory=list,
        metadata={"help": "Motion-name substrings left out of the *_penalised arm-balance metrics "
                          "and of the drag aggregates."},
    )
    drag_report_motions: List[str] = field(
        default_factory=list,
        metadata={"help": "Motion-name substrings logged individually as eval/drag/<name>_J and "
                          "pooled as eval/drag/top_J (the fine-tune C drag gate). Drag is computed "
                          "only when the contact-graph control has physics tables loaded."},
    )
    drag_load_n: float = field(
        default=50.0, metadata={"help": "A foot/hand zone drags above this ground load (N) ..."}
    )
    drag_slip_mps: float = field(
        default=0.10, metadata={"help": "... while its slowest bottom corner slides faster than this (m/s)."}
    )
    drag_mu: float = field(
        default=0.75, metadata={"help": "Friction coefficient converting load x slip into drag work (J)."}
    )
    # --- Support rule v2 (card E1, expert_revist/graph_growth_2026_10_03/PLAN.MD) --- #
    # Plain scalar defaults on purpose: frozen configs pickled before these fields existed resume
    # with the class defaults, and the evaluator reads them with getattr defaults as well.
    support_rule: str = field(
        default="v1",
        metadata={"help": "Score that drives the curriculum's sampling: 'v1' (body origin below "
                          "foot_down_z on zones the reference keeps above unloaded_ref_min_z) or 'v2' "
                          "(load on the release sidecar's known-free zones; needs the physics tables "
                          "and the sidecar). Every logged eval/perf* key and score_based.ckpt stay v1; "
                          "v2 is logged under eval/perf_v2/* whenever it can be computed."},
    )
    support_v2_load_frac_bw: float = field(
        default=0.03, metadata={"help": "v2: a known-free zone is loaded at this share of body weight."}
    )
    support_v2_min_share: float = field(
        default=0.2, metadata={"help": "v2: ... on this share of the hold window (load mask dilated by "
                                       "event_dilate_frames first) -> a support violation."}
    )
    support_v2_down_m: float = field(
        default=0.02, metadata={"help": "v2: a commanded zone is down when its lowest collider point is this low."}
    )
    support_v2_realised_share: float = field(
        default=0.9, metadata={"help": "v2: commanded supports are realised when every commanded zone is "
                                       "down on this share of the window."}
    )
    support_v2_tracked_share: float = field(
        default=0.9, metadata={"help": "v2: a hold counts (realised / substitution totals) when this share "
                                       "of its window is tracked."}
    )


@dataclass
class HoldCurriculumEvaluatorConfig(MimicEvaluatorConfig):
    """MimicEvaluator whose sampling update is a uniform + hold-performance mixture."""

    _target_: str = "protomotions.agents.evaluators.hold_curriculum_evaluator.HoldCurriculumEvaluator"
    hold_manifest: str = field(
        default="",
        metadata={"help": "holds_extended.yaml of the packaged motion file (holds, families, groups)."},
    )
    curriculum: HoldCurriculumConfig = field(default_factory=HoldCurriculumConfig)
