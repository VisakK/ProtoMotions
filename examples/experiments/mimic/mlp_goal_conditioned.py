# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Goal-conditioned Stage-1 expert: hold-graph goals, asymmetric actor/critic.

``expert_revist/expert_revisit.MD`` §3. Every yoga expert so far has been a
next-frame tracker (``MimicControlConfig.future_steps = 1``): its action is an
affine map of the reference pose 33 ms ahead (R² = 0.9998, v11), so its labels
carry no goal at any horizon and no student distilled from them could learn to
reach a *commanded* goal. This expert is built so that its action cannot be
computed without the goal:

**Actor** sees exactly what a deployed policy can see -- proprioception with
the contact-rich block (``max_coords_obs``, ``contact_obs_v1``,
``contact_proximity_obs``, ``contact_state_obs``), its previous action, a short
pose history, and the **goal**: the next ``NUM_GOAL_STEPS`` holds of the
hold graph, each as a full-body pose (``trackable_bodies_subset="all"``), its
contact configuration, its deadline and its dwell (``include_current_segment``
+ ``dwell_channels``, so inside a hold the command is "stay here for X more
seconds" and outside it "be there in X seconds"). No ``mimic_target_poses``.

**Critic** sees all of that plus the dense reference future
(``mimic_target_poses`` over ``CRITIC_FUTURE_STEPS``): asymmetric actor/critic.
Value estimation stays as easy as it is for the trackers while the actor is
forced through the goal. ``PPOModel`` takes the union of the two key sets and
the actor's ``in_keys`` are what an exported or distilled policy consumes, so
the privileged window never leaks.

**Reward** is the unchanged ``mlp_2x_no_contact_rew`` tracking stack. The
observation is sparse; the reward is not -- which is the difference from
sparse-goal RL. Stillness in the extended holds is paid for by ``gv``/``gav``
against a frozen reference. The goal-attainment quantities the runtime
already computes (contact IoU against the nearest goal, 6-body pose error to
the commanded hold) are logged at weight 0 as ``raw_r/diag_*``.

**Goals** come from ``data/scripts/build_hold_graph.py`` (nodes = named holds
from the kinematic manifest; the transitions are the gaps between them), on
the corpus ``make_hold_extended_clips.py`` writes -- every clip in three hold
duration variants, so the exit time is not predictable from the pose.

**Visualization.** The in-training stick-figure panel the students carried
(``SequenceViz``: scripted goal sequences driven through ``set_manual_goal``,
rendered with matplotlib, logged to wandb under ``viz/``) runs here too, every
``--viz-sequences-every`` epochs, on the expert's own acceptance probes
(``data/scripts/make_hold_graph_probe_plans.py``: fork tests from the standing
start, dwell tests at the hold) plus graph-derived holds. It queries
``PPOModel.forward_inference`` -- the deterministic ``mean_action`` -- so what
it shows is the deployed policy, not a sample.

Round 1 deliberately trains "the same way": each episode follows one clip and
its own schedule. The counterfactuals that make the goal causal come from the
corpus itself -- sixty clips share the standing start with sixty different
goals. Matched-state clip switching is the round-1.5 augmentation.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_goal_conditioned.py \
      --experiment-name smpl_yogi_expert60_goal_conditioned \
      --motion-file data/smpl/yoga_yogi_expert60.pt \
      --hold-graph-file data/smpl/yoga_hold_graph_expert60/contact_graph.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb \
      --overrides env.ref_respawn_offset=0.005

Smoke (no wandb)::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_goal_conditioned.py \
      --experiment-name expert60_gc_smoke --motion-file data/smpl/yoga_yogi_expert60.pt \
      --hold-graph-file data/smpl/yoga_hold_graph_expert60/contact_graph.pt \
      --num-envs 8 --batch-size 32 --training-max-steps 256 --headless True \
      --overrides env.ref_respawn_offset=0.005 env.contact_diagnostics_interval=1
"""

from __future__ import annotations

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig


DEFAULT_HOLD_GRAPH = "data/smpl/yoga_hold_graph_expert60/contact_graph.pt"

# Goal window: this hold and the next. Tier-0 measured the student acting on
# the whole five-slot window (a downdog in slot 1 moved a standing hold rate
# 0.176 -> 0.003); two slots is the smallest window that still says where the
# clip goes after the current hold.
NUM_GOAL_STEPS = 2

# Dense reference future for the CRITIC only (control steps at 30 Hz, 0.03-0.5 s).
CRITIC_FUTURE_STEPS = [1, 5, 10, 15]

# Short strided pose history for the actor (0.03, 0.27, 0.5 s), 1-based into the
# stored buffer (select_step_indices convention).
HISTORICAL_STEP_INDICES = [1, 8, 15]
TOTAL_STORED_HISTORICAL_STEPS = 15

# Same scaling constants the earlier experts were trained with.
CONTACT_OBS_V1_PARAMS = {
    "force_reference_n": 100.0,
    "force_clip_n": 5000.0,
    "force_rate_reference_n_per_s": 1000.0,
    "force_rate_clip_n_per_s": 50000.0,
    "friction_mu": 1.0,
    "friction_utilization_clip": 2.0,
    "velocity_reference_mps": 2.0,
    "contact_age_clip_s": 2.0,
    "air_age_clip_s": 2.0,
}

# Default panel: eight fork probes (standing start -> commanded family hold ->
# standing) and two dwell probes, from make_hold_graph_probe_plans.py; the
# remaining slots are filled with graph-derived holds at the highest-dwell
# nodes. Plans that do not resolve against the run's graph are skipped with a
# warning, never fatal.
DEFAULT_VIZ_PLANS = [
    "data/scripts/plans_expert60/fork_Warrior_II_Pose_or_Virabhadrasana_II.json",
    "data/scripts/plans_expert60/fork_Tree_Pose_or_Vrksasana.json",
    "data/scripts/plans_expert60/fork_Crane_Crow_Pose_or_Bakasana.json",
    "data/scripts/plans_expert60/fork_Handstand_pose_or_Adho_Mukha_Vrksasana.json",
    "data/scripts/plans_expert60/fork_Feathered_Peacock_Pose_or_Pincha_Mayuras.json",
    "data/scripts/plans_expert60/fork_Supported_Headstand_pose_or_Salamba_Sirs.json",
    "data/scripts/plans_expert60/fork_Side_Plank_Pose_or_Vasisthasana.json",
    "data/scripts/plans_expert60/fork_Downward_Facing_Dog_pose_or_Adho_Mukha_S.json",
    "data/scripts/plans_expert60/dwell_Warrior_II_Pose_or_Virabhadrasana_II_3s.json",
    "data/scripts/plans_expert60/dwell_Warrior_II_Pose_or_Virabhadrasana_II_10s.json",
]

# The deployable observation contract. Anything distilled from this expert
# receives exactly these keys.
ACTOR_KEYS = [
    "max_coords_obs",
    "contact_obs_v1",
    "contact_proximity_obs",
    "contact_state_obs",
    "previous_actions",
    "historical_pose_obs",
    "masked_mimic_target_poses",
    "masked_mimic_target_masks",
    "masked_mimic_target_times",
    "masked_mimic_target_poses_masks",
    "contact_goal_obs",
    "contact_goal_masks",
]
# Privileged, critic only.
CRITIC_ONLY_KEYS = ["mimic_target_poses"]
CRITIC_KEYS = ACTOR_KEYS + CRITIC_ONLY_KEYS


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hold-graph-file",
        type=str,
        default=DEFAULT_HOLD_GRAPH,
        help="contact_graph.pt from build_hold_graph.py, keyed to --motion-file.",
    )
    parser.add_argument(
        "--goal-bodies",
        type=str,
        default="all",
        choices=["all", "subset"],
        help="Bodies in the goal pose: every body ('all', the default -- the "
             "expert must reproduce the hold) or the robot config's 6-body "
             "trackable subset.",
    )
    parser.add_argument(
        "--sense-body-pair-contacts",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=True,
        help="Filter every contact sensor against the robot's own bodies so "
             "contact_state_obs can answer the body-body half of the goal "
             "(crow's shins on the upper arms). False senses ground only.",
    )
    parser.add_argument(
        "--segment-start-prob", type=float, default=0.6,
        help="Probability an episode starts just before a hold (ContactGraphMotionManager).",
    )
    parser.add_argument("--segment-pre-roll-s", type=float, default=0.5)
    parser.add_argument(
        "--segment-weighting", type=str, default="uniform",
        choices=["uniform", "dwell", "rare_node"],
    )
    parser.add_argument(
        "--critic-future-steps", type=int, nargs="+", default=list(CRITIC_FUTURE_STEPS),
        help="Frame offsets of the critic's dense reference window (must start at 1).",
    )
    parser.add_argument(
        "--viz-sequences-every", type=int, default=500,
        help="Render the stick-figure goal-sequence panel every N epochs, saved "
             "under results/<exp>/viz/ and logged to wandb. 0 disables.",
    )
    parser.add_argument(
        "--viz-plan-files", type=str, nargs="*", default=list(DEFAULT_VIZ_PLANS),
        help="Goal-sequence plan JSONs on the panel; the rest of the panel is "
             "graph-derived holds at the highest-dwell nodes and edge round trips.",
    )
    parser.add_argument(
        "--viz-num-sequences", type=int, default=12,
        help="Total sequences on the panel (plans first, then graph-derived).",
    )
    parser.add_argument(
        "--viz-max-seconds", type=float, default=20.0,
        help="Cap on a sequence's duration; longer plans lose goals from the end.",
    )
    parser.add_argument(
        "--viz-hold-seconds", type=float, default=8.0,
        help="Dwell requested by each graph-derived pure-hold sequence.",
    )
    parser.add_argument(
        "--viz-log-scalars",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=False,
        help="Also send the panel's per-sequence/per-goal scalars to wandb. Off "
             "by default (round 8): each is one draw per sequence per epoch; "
             "they are always written to viz/epoch_*/summary.json.",
    )


def terrain_config(args: argparse.Namespace):
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Sense every body (and, by default, every body pair); goal on every body."""
    updates = dict(
        contact_bodies="all",
        contact_observation_bodies="all",
        # Irrelevant here (no contact reward); set for the observation contract
        # the contact-rich experts established.
        contact_reward_bodies=["all_left_foot_bodies", "all_right_foot_bodies"],
        contact_pair_bodies=(
            "all" if getattr(args, "sense_body_pair_contacts", True) else None
        ),
    )
    if getattr(args, "goal_bodies", "all") == "all":
        updates["trackable_bodies_subset"] = "all"
    robot_cfg.update_fields(**updates)


def _contact_observation_body_ids(robot_cfg: RobotConfig) -> list:
    body_names = robot_cfg.contact_observation_bodies or []
    if not body_names:
        raise ValueError(
            "mlp_goal_conditioned requires contact observation bodies; "
            "configure_robot_and_simulator() must run first."
        )
    name_to_id = {
        name: body_id for body_id, name in enumerate(robot_cfg.kinematic_info.body_names)
    }
    return [name_to_id[name] for name in body_names]


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    import torch
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        contact_obs_v1_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        nearest_surface_obs_factory,
        pow_rew_factory,
        previous_actions_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.contact_graph_control import ContactGraphControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.motion_manager.config import ContactGraphMotionManagerConfig
    from protomotions.envs.obs import (
        compute_contact_goal_masks,
        compute_contact_goal_obs,
        compute_contact_goal_pose_error,
        compute_contact_goal_pose_error_visible,
        compute_contact_goal_reached,
        compute_contact_state_obs,
        compute_historical_poses_with_time,
        compute_target_masks_only,
        compute_target_poses_only,
        compute_target_time_offsets,
        to_float,
    )

    critic_future = [int(s) for s in getattr(args, "critic_future_steps", CRITIC_FUTURE_STEPS)]
    if not critic_future or critic_future[0] != 1 or sorted(set(critic_future)) != critic_future:
        raise ValueError(
            f"--critic-future-steps must be strictly increasing and start at 1, got {critic_future}"
        )
    graph_file = getattr(args, "hold_graph_file", DEFAULT_HOLD_GRAPH)

    control_components = {
        "contact_graph": ContactGraphControlConfig(
            num_masked_future_steps=NUM_GOAL_STEPS,
            num_goal_steps=NUM_GOAL_STEPS,
            future_steps=critic_future,
            bootstrap_on_episode_end=True,
            graph_file=graph_file,
            min_lead_s=0.2,
            # Round 1: the goal is always fully specified. Partial-spec
            # conditioning is a student-stage concern.
            pose_visible_prob=1.0,
            contact_visible_prob=1.0,
            full_pose_prob=1.0,
            require_first_goal_specified=True,
            num_history_events=0,
            # "Stay here for X more seconds" inside a hold, "be there in X
            # seconds" outside one (v10_1 semantics).
            dwell_channels=True,
            include_current_segment=True,
            force_max_conditioned_bodies_prob=0.0,
            force_small_num_conditioned_bodies_prob=0.0,
        ),
    }

    conditionable_body_ids = torch.tensor(
        [
            robot_cfg.kinematic_info.body_names.index(name)
            for name in robot_cfg.trackable_bodies_subset
        ],
        dtype=torch.long,
    )
    body_ids = _contact_observation_body_ids(robot_cfg)

    # The measured-contact block's vocabulary comes from the graph so slot k of
    # contact_state_obs and slot k of the goal's contact half name the same pair.
    graph = ContactGraph.from_file(graph_file, device="cpu")
    pair_bodies = list(getattr(robot_cfg, "contact_pair_bodies", None) or [])
    contact_state_params = graph.contact_scatter_maps(
        body_names=list(robot_cfg.kinematic_info.body_names),
        # 3 % / 2 % of the 74 kg body weight -- the make thresholds the shipped
        # graphs were annotated with.
        ground_threshold_n=0.03 * 74.0 * 9.81,
        body_threshold_n=0.02 * 74.0 * 9.81,
        pair_body_names=pair_bodies or None,
    )
    contact_state_params.pop("pair_slot_names")
    body_body_slots = contact_state_params.pop("num_body_body_slots")
    if pair_bodies and any("+" in name for name in graph.pair_names) and body_body_slots == 0:
        raise ValueError(
            "body-body contact sensing was requested and the graph names body-body "
            "pairs, but none of them mapped onto a sensed body."
        )

    observation_components = {
        # --- deployable ------------------------------------------------- #
        "max_coords_obs": max_coords_obs_factory(observe_contacts=False),
        "contact_obs_v1": contact_obs_v1_factory(body_ids=body_ids, **CONTACT_OBS_V1_PARAMS),
        "contact_proximity_obs": nearest_surface_obs_factory(body_ids=body_ids),
        "contact_state_obs": MdpComponent(
            compute_func=compute_contact_state_obs,
            dynamic_vars={
                "ground_forces": EnvContext.current.rigid_body_ground_forces,
                "pair_forces": EnvContext.current.rigid_body_pair_contact_forces,
            },
            static_params=contact_state_params,
        ),
        "previous_actions": previous_actions_factory(history_steps=1),
        "historical_pose_obs": MdpComponent(
            compute_func=compute_historical_poses_with_time,
            dynamic_vars={
                "historical_rigid_body_pos": EnvContext.historical.rigid_body_pos,
                "historical_rigid_body_rot": EnvContext.historical.rigid_body_rot,
                "historical_rigid_body_vel": EnvContext.historical.rigid_body_vel,
                "historical_rigid_body_ang_vel": EnvContext.historical.rigid_body_ang_vel,
                "historical_ground_heights": EnvContext.historical.ground_heights,
                "historical_body_contacts": EnvContext.historical.body_contacts,
                "dt": EnvContext.dt,
            },
            static_params={
                "history_steps": HISTORICAL_STEP_INDICES,
                "local_obs": True,
                "root_height_obs": True,
                "w_last": True,
            },
        ),
        # --- the goal: pose half ---------------------------------------- #
        "masked_mimic_target_poses": MdpComponent(
            compute_func=compute_target_poses_only,
            dynamic_vars={
                "current_state_body_pos": EnvContext.current.rigid_body_pos,
                "current_state_body_rot": EnvContext.current.rigid_body_rot,
                "masked_mimic_ref_pos": EnvContext.masked_mimic.ref_pos,
                "masked_mimic_ref_rot": EnvContext.masked_mimic.ref_rot,
                "masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks,
            },
            static_params={
                "conditionable_body_ids": conditionable_body_ids,
                "include_root_relative": True,
            },
        ),
        "masked_mimic_target_masks": MdpComponent(
            compute_func=compute_target_masks_only,
            dynamic_vars={
                "masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks,
            },
            static_params={"conditionable_body_ids": conditionable_body_ids},
        ),
        "masked_mimic_target_times": MdpComponent(
            compute_func=compute_target_time_offsets,
            dynamic_vars={"masked_mimic_time_offsets": EnvContext.masked_mimic.time_offsets},
        ),
        "masked_mimic_target_poses_masks": MdpComponent(
            compute_func=to_float,
            dynamic_vars={"x": EnvContext.masked_mimic.target_poses_masks},
        ),
        # --- the goal: contact half (+ deadline/dwell channels) ---------- #
        "contact_goal_obs": MdpComponent(
            compute_func=compute_contact_goal_obs,
            dynamic_vars={
                "contact_spec": EnvContext.contact_goal.contact_spec,
                "orient_spec": EnvContext.contact_goal.orient_spec,
                "visible": EnvContext.contact_goal.visible,
                "dwell_features": EnvContext.contact_goal.dwell_features,
            },
        ),
        "contact_goal_masks": MdpComponent(
            compute_func=compute_contact_goal_masks,
            dynamic_vars={"visible": EnvContext.contact_goal.visible},
        ),
        # --- privileged, CRITIC ONLY ------------------------------------ #
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
    }

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        # Goal attainment, logged at weight 0: how much of the nearest goal's
        # contact set is realised, and the 6-body distance to the commanded
        # pose (same arithmetic as data/scripts/score_probe_pose.py).
        "diag_contact_goal_iou": MdpComponent(
            compute_func=compute_contact_goal_reached,
            dynamic_vars={"reached": EnvContext.contact_goal.reached},
            static_params={"weight": 0.0},
        ),
        "diag_goal_pose_error": MdpComponent(
            compute_func=compute_contact_goal_pose_error,
            dynamic_vars={"pose_error": EnvContext.contact_goal.pose_error},
            static_params={"weight": 0.0},
        ),
        "diag_goal_pose_visible": MdpComponent(
            compute_func=compute_contact_goal_pose_error_visible,
            dynamic_vars={"pose_error_visible": EnvContext.contact_goal.pose_error_visible},
            static_params={"weight": 0.0},
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=TOTAL_STORED_HISTORICAL_STEPS,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=ContactGraphMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            graph_file=graph_file,
            segment_start_prob=getattr(args, "segment_start_prob", 0.6),
            pre_roll_s=getattr(args, "segment_pre_roll_s", 0.5),
            segment_weighting=getattr(args, "segment_weighting", "uniform"),
        ),
        contact_force_on_threshold_n=5.0,
        contact_force_off_threshold_n=2.0,
        contact_diagnostics_interval=100,
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """The 2x PPO trackers' networks and optimisers; asymmetric input contracts."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=list(ACTOR_KEYS),
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=list(ACTOR_KEYS),
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=list(CRITIC_KEYS),
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=2048, activation="relu") for _ in range(4)],
    )

    viz_every = int(getattr(args, "viz_sequences_every", 0) or 0)
    sequence_viz = None
    if viz_every > 0:
        from protomotions.agents.evaluators.sequence_viz import SequenceVizConfig

        sequence_viz = SequenceVizConfig(
            viz_every=viz_every,
            plan_files=list(getattr(args, "viz_plan_files", DEFAULT_VIZ_PLANS) or []),
            num_sequences=int(getattr(args, "viz_num_sequences", 12) or 12),
            hold_seconds=float(getattr(args, "viz_hold_seconds", 8.0) or 8.0),
            max_seconds=float(getattr(args, "viz_max_seconds", 20.0) or 20.0),
            log_scalars=bool(getattr(args, "viz_log_scalars", False)),
        )

    return PPOAgentConfig(
        sequence_viz=sequence_viz,
        model=PPOModelConfig(
            in_keys=list(CRITIC_KEYS),
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        save_epoch_checkpoint_every=1000,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Evaluation: no terminations, long episodes, start every clip at t = 0."""
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
