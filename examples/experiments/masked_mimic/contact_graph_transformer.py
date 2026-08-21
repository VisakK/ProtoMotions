# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 MaskedMimic student conditioned on contact configurations.

This is :mod:`examples.experiments.masked_mimic.transformer` with two changes,
both of which exist to answer one question -- *"what motion gets me to this
contact configuration, held in this pose?"*

**1. The goal comes from a contact graph, not a random future frame.**
``data/scripts/build_contact_graph_from_rollouts.py`` rolls each Stage-1 expert
out on its own clips and force-annotates the result: a contact pair is active
when PhysX reports load on it, so the graph records what the experts *actually
do*, not what geometric proximity suggests they might.  Nodes are contact
configurations, edges are the make/break transitions between them, and every node
and edge carries the clip and clip-time it came from.  ``ContactGraphControl``
turns that into the student's goal: the next few configurations the clip reaches,
each with the pose being held at that moment.

The contact half and the pose half of every goal are masked *independently*.  With
both revealed this is MaskedMimic with a better-chosen target frame; with only
the contact set revealed the student has to find a pose that realises it.  The two
are not redundant -- ``notes/Pressure_supervision_design.MD`` §3 shows a crow foot
9 cm off the floor carrying 636 N, so which contacts bear load is simply not a
function of the reference kinematics.

**2. Three experts, routed by clip.**  The yoga corpus is covered by three Stage-1
trackers on disjoint slices, and none is competent outside its own.
``MultiExpertSupervisedAgent`` labels each environment with the expert that owns
the clip it is playing, using the table
``data/scripts/package_student_corpus.py`` writes beside the packaged library.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/masked_mimic/contact_graph_transformer.py \
      --experiment-name smpl_yogi_contact_graph_student_s2 \
      --motion-file data/smpl/yoga_yogi_student171.pt \
      --contact-graph-file data/smpl/yoga_contact_graph/contact_graph.pt \
      --motion-expert-file data/smpl/yoga_yogi_student171.experts.json \
      --expert-model-paths results/smpl_yogi_easy128_contact_rich/last.ckpt \
                           results/smpl_yogi_hard29_pressure_ab_s1/last.ckpt \
                           results/smpl_yogi_singleleg14_pressure_ab_s1/last.ckpt \
      --num-envs 1024 --batch-size 8192 --headless True --use-wandb \
      --overrides env.ref_respawn_offset=0.005

See ``notes/Student_distill_experiment1.MD``.
"""

from __future__ import annotations

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.supervised.multi_expert import (
    MultiExpertMaskedMimicAgentConfig,
)


# One token per upcoming contact configuration. Kept equal to the control
# component's num_goal_steps -- the prior reshapes the goal block by this.
NUM_GOAL_STEPS = 5
TOTAL_STORED_HISTORICAL_STEPS = 5
NUM_HISTORICAL_CONDITIONED_STEPS = 5

DEFAULT_CONTACT_GRAPH = "data/smpl/yoga_contact_graph/contact_graph.pt"

# Same scaling constants the three Stage-1 experts were trained with. The student
# sees its own copy of the contact channels so it has the expert's sensory
# contract and the imitation loss is not floored by missing inputs.
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

STATE_KEYS = ["max_coords_obs", "contact_obs_v1", "contact_proximity_obs"]

# Binary, already in [0, 1], and deliberately NOT run through a running
# normaliser: most of the 104 contact pairs are active on a tiny fraction of
# frames, so normalising would divide a rare pair by a near-zero standard
# deviation and hand the network a clamped spike instead of a flag. Same
# reasoning as `contact_goal_obs`, which this block is the measured counterpart
# of -- slot k means the same pair in both.
UNNORMALIZED_STATE_KEYS = ["contact_state_obs"]


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Contact-graph and multi-expert CLI arguments."""
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Single expert checkpoint (prepended to --expert-model-paths).",
    )
    parser.add_argument(
        "--expert-model-paths",
        type=str,
        nargs="+",
        default=None,
        help="Expert checkpoints, in the order the routing table indexes them.",
    )
    parser.add_argument(
        "--motion-expert-file",
        type=str,
        default=None,
        help="JSON mapping motion id -> expert index (package_student_corpus.py).",
    )
    parser.add_argument(
        "--contact-graph-file",
        type=str,
        default=DEFAULT_CONTACT_GRAPH,
        help="contact_graph.pt built from expert rollouts.",
    )
    parser.add_argument(
        "--goal-pose-visible-prob",
        type=float,
        default=0.75,
        help="Probability a goal slot reveals the held pose.",
    )
    parser.add_argument(
        "--goal-contact-visible-prob",
        type=float,
        default=0.85,
        help="Probability a goal slot reveals the contact configuration.",
    )
    parser.add_argument(
        "--goal-full-pose-prob",
        type=float,
        default=0.75,
        help="Given a revealed pose, probability every body is revealed.",
    )
    parser.add_argument(
        "--segment-start-prob",
        type=float,
        default=0.6,
        help="Probability an episode starts at a contact-graph segment entry "
             "instead of a uniformly-sampled clip time. 0 restores plain RSI.",
    )
    parser.add_argument(
        "--segment-pre-roll-s",
        type=float,
        default=0.5,
        help="Anchored starts begin up to this long before the segment, drawn "
             "uniformly, so the approach is covered as well as the entry frame.",
    )
    parser.add_argument(
        "--segment-weighting",
        type=str,
        default="uniform",
        choices=["uniform", "dwell", "rare_node"],
        help="How a segment is drawn within a clip.",
    )
    parser.add_argument(
        "--sense-body-pair-contacts",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=True,
        help="Filter every per-body contact sensor against the robot's own "
             "bodies too, so the student observes body-body contact and the "
             "reached-goal diagnostic covers the whole pair vocabulary. "
             "False restores ground-only sensing (and the old cost).",
    )


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    """Sense every body, as all three experts were trained to.

    The expert observation components are deep-copied out of their frozen
    configs with hard-coded body ids covering all 24 bodies, so the environment
    has to be configured to sense all 24 or those ids index nothing.
    """
    robot_cfg.update_fields(
        contact_bodies="all",
        contact_observation_bodies="all",
        contact_reward_bodies=["all_left_foot_bodies", "all_right_foot_bodies"],
        # Body-body contact sensing. Every per-body contact sensor additionally
        # filters against all 24 bodies, so PhysX reports the per-pair force and
        # `contact_state_obs` can tell the student whether the shin-on-upper-arm
        # half of its goal is actually satisfied. Without this the body-body
        # pairs are specified in the goal and unanswerable from state, which is
        # limit 1 of notes/Student_distill_experiment1.MD §8.
        contact_pair_bodies=(
            "all"
            if getattr(args, "sense_body_pair_contacts", True)
            else None
        ),
    )


def _expert_paths(args: argparse.Namespace) -> list:
    paths = list(getattr(args, "expert_model_paths", None) or [])
    single = getattr(args, "expert_model_path", None)
    if single:
        paths = [single] + paths
    return paths


def _contact_observation_body_ids(robot_cfg: RobotConfig) -> list:
    body_names = robot_cfg.contact_observation_bodies or []
    if not body_names:
        raise ValueError(
            "contact_graph_transformer requires contact observation bodies; "
            "configure_robot_and_simulator() must run first."
        )
    name_to_id = {
        name: body_id
        for body_id, name in enumerate(robot_cfg.kinematic_info.body_names)
    }
    return [name_to_id[name] for name in body_names]


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Environment with contact-graph goals and the experts' sensory contract."""
    import torch
    from protomotions.envs.motion_manager.config import (
        ContactGraphMotionManagerConfig,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.control.contact_graph_control import (
        ContactGraphControlConfig,
    )
    from protomotions.envs.component_factories import (
        contact_obs_v1_factory,
        max_coords_obs_factory,
        nearest_surface_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        action_smoothness_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.obs import (
        compute_historical_poses_with_time,
        compute_target_poses_only,
        compute_target_masks_only,
        compute_target_time_offsets,
        compute_contact_goal_obs,
        compute_contact_goal_masks,
        compute_contact_goal_reached,
        compute_contact_state_obs,
        to_float,
    )
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "contact_graph": ContactGraphControlConfig(
            num_masked_future_steps=NUM_GOAL_STEPS,
            num_goal_steps=NUM_GOAL_STEPS,
            future_steps=1,  # raised below if an expert needs more
            bootstrap_on_episode_end=True,
            graph_file=args.contact_graph_file,
            min_lead_s=0.2,
            pose_visible_prob=args.goal_pose_visible_prob,
            contact_visible_prob=args.goal_contact_visible_prob,
            full_pose_prob=args.goal_full_pose_prob,
            require_first_goal_specified=True,
            # Inherited masked-mimic knobs; only used for the partial-pose case.
            force_max_conditioned_bodies_prob=0.1,
            force_small_num_conditioned_bodies_prob=0.1,
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

    # The measured-contact block's vocabulary comes from the GRAPH, not from a
    # copy of the zone definition kept here: that is what guarantees slot k of
    # contact_state_obs and slot k of contact_goal_obs name the same pair.
    graph = ContactGraph.from_file(args.contact_graph_file, device="cpu")
    pair_bodies = list(getattr(robot_cfg, "contact_pair_bodies", None) or [])
    contact_state_params = graph.contact_scatter_maps(
        body_names=list(robot_cfg.kinematic_info.body_names),
        # 3 % / 2 % of a 74 kg body weight, the same make thresholds
        # build_contact_graph_from_rollouts.py annotated the graph with, so the
        # student's state block and its goal call a contact the same thing.
        ground_threshold_n=0.03 * 74.0 * 9.81,
        body_threshold_n=0.02 * 74.0 * 9.81,
        pair_body_names=pair_bodies or None,
    )
    contact_state_params.pop("pair_slot_names")
    body_body_slots = contact_state_params.pop("num_body_body_slots")
    graph_names_pairs = any("+" in name for name in graph.pair_names)
    if pair_bodies and graph_names_pairs and body_body_slots == 0:
        raise ValueError(
            "body-body contact sensing was requested and the graph names "
            "body-body pairs, but none of them mapped onto a sensed body. "
            "contact_pair_bodies does not cover the zones the graph uses, so "
            "the student would be given goals it can never observe."
        )
    print(
        f"contact_state_obs: {graph.num_pairs} slots "
        f"({body_body_slots} body-body), sensing pairs: {bool(pair_bodies)}"
    )

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True, root_height_obs=True, observe_contacts=False
        ),
        "contact_obs_v1": contact_obs_v1_factory(
            body_ids=body_ids, **CONTACT_OBS_V1_PARAMS
        ),
        "contact_proximity_obs": nearest_surface_obs_factory(body_ids=body_ids),
        "previous_actions": previous_actions_factory(history_steps=1),
        # Dense reference future: privileged, encoder-only.
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True, with_relative=True
        ),
        # --- the pose half of each contact-graph goal -------------------- #
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
            dynamic_vars={
                "masked_mimic_time_offsets": EnvContext.masked_mimic.time_offsets,
            },
        ),
        "masked_mimic_target_poses_masks": MdpComponent(
            compute_func=to_float,
            dynamic_vars={"x": EnvContext.masked_mimic.target_poses_masks},
        ),
        "masked_mimic_target_bodies_masks": MdpComponent(
            compute_func=to_float,
            dynamic_vars={"x": EnvContext.masked_mimic.target_bodies_masks},
        ),
        # --- what the robot is actually touching, in the goal's vocabulary -- #
        # Slot k of this block is the same contact pair as slot k of each goal
        # step's contact half, so "am I there yet?" is an elementwise question.
        # The body-body half is the point: crow, firefly and eight-angle all have
        # ground contact "two hands" and are told apart only by which limb is
        # pressed against which.
        "contact_state_obs": MdpComponent(
            compute_func=compute_contact_state_obs,
            dynamic_vars={
                "ground_forces": EnvContext.current.rigid_body_ground_forces,
                "pair_forces": EnvContext.current.rigid_body_pair_contact_forces,
            },
            static_params=contact_state_params,
        ),
        # --- the contact half ------------------------------------------- #
        "contact_goal_obs": MdpComponent(
            compute_func=compute_contact_goal_obs,
            dynamic_vars={
                "contact_spec": EnvContext.contact_goal.contact_spec,
                "orient_spec": EnvContext.contact_goal.orient_spec,
                "visible": EnvContext.contact_goal.visible,
            },
        ),
        "contact_goal_masks": MdpComponent(
            compute_func=compute_contact_goal_masks,
            dynamic_vars={"visible": EnvContext.contact_goal.visible},
        ),
        # --- history ----------------------------------------------------- #
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
                "history_steps": TOTAL_STORED_HISTORICAL_STEPS,
                "local_obs": True,
                "root_height_obs": True,
                "w_last": True,
            },
        ),
    }

    expert_paths = _expert_paths(args)
    if expert_paths:
        from protomotions.agents.supervised.expert_utils import (
            get_expert_observation_components,
        )
        from protomotions.utils.config_utils import (
            load_resolved_configs_from_checkpoint,
        )

        # Every expert reads the same block, so the conflict check has to be
        # against the *student's* keys, not against a block an earlier expert
        # already contributed.
        student_obs_keys = list(observation_components.keys())
        reference_keys = None
        for path in expert_paths:
            expert_configs = load_resolved_configs_from_checkpoint(path)
            expert_env_config = expert_configs["env"]
            expert_agent_config = expert_configs["agent"]

            expert_history_steps = getattr(
                expert_env_config, "num_state_history_steps", 0
            )
            assert TOTAL_STORED_HISTORICAL_STEPS >= expert_history_steps, (
                f"Insufficient history: current={TOTAL_STORED_HISTORICAL_STEPS}, "
                f"{path} requires={expert_history_steps}"
            )

            if getattr(expert_env_config, "control_components", None):
                for ctrl_cfg in expert_env_config.control_components.values():
                    expert_num_future = getattr(ctrl_cfg, "future_steps", None)
                    if expert_num_future is not None:
                        cfg = control_components["contact_graph"]
                        if cfg.future_steps < expert_num_future:
                            cfg.future_steps = expert_num_future

            components = get_expert_observation_components(
                expert_env_config,
                expert_agent_config,
                existing_obs_keys=student_obs_keys,
            )
            if reference_keys is None:
                reference_keys = sorted(components.keys())
                observation_components.update(components)
            elif sorted(components.keys()) != reference_keys:
                raise ValueError(
                    "experts disagree on their observation contract:\n"
                    f"  first: {reference_keys}\n"
                    f"  {path}: {sorted(components.keys())}\n"
                    "The environment computes one expert_* block shared by all."
                )

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.25),
    }

    reward_components = {
        **mimic_tracking_rewards_factory(
            gt_weight=0.5, gr_weight=0.3, gt_coef=-100.0, gr_coef=-5.0
        ),
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        # Weight 0: how much of the nearest goal's *ground* contact set the
        # policy currently realises. Body-body pairs are not observable at
        # runtime (ProtoMotions' contact sensors are filtered against the
        # terrain only), so this is the ground subset and nothing more.
        "diag_contact_goal_ground_iou": MdpComponent(
            compute_func=compute_contact_goal_reached,
            dynamic_vars={"reached": EnvContext.contact_goal.reached},
            static_params={"weight": 0.0},
        ),
    }

    return EnvConfig(
        max_episode_length=1000,
        num_state_history_steps=TOTAL_STORED_HISTORICAL_STEPS,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        # Episode starts are anchored to the graph's own make/break boundaries
        # rather than sampled uniformly in the clip. The two probabilities
        # compose: init_start_prob is applied first by the base manager and then
        # segment_start_prob overrides it, so the effective mixture at the
        # defaults is 60 % anchored / 8 % at t=0 / 32 % uniform.
        motion_manager=ContactGraphMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            graph_file=args.contact_graph_file,
            segment_start_prob=getattr(args, "segment_start_prob", 0.6),
            pre_roll_s=getattr(args, "segment_pre_roll_s", 0.5),
            segment_weighting=getattr(args, "segment_weighting", "uniform"),
        ),
        # Same contact hysteresis the Stage-1 experts were trained with.
        contact_force_on_threshold_n=5.0,
        contact_force_off_threshold_n=2.0,
        contact_diagnostics_interval=100,
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MultiExpertMaskedMimicAgentConfig:
    from protomotions.agents.supervised.masked_mimic_config import (
        MaskedMimicModelConfig,
        MaskedMimicVAEConfig,
        VAENoiseType,
        KLDScheduleConfig,
    )
    from protomotions.agents.common.config import (
        ObsProcessorConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        MLPWithConcatConfig,
        TransformerConfig,
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    transformer_token_size = 512
    transformer_encoder_widths = 256
    vae_latent_dim = 64

    def state_normalizers(prefix: str = ""):
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

    # Privileged encoder: sees the dense reference future *and* the goal, and
    # produces the residual posterior the imitation loss is taken on.
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
        ],
        out_keys=["encoder_mu", "encoder_logvar"],
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
                out_keys=["encoder_mu"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_logvar"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    # Deployable prior: current state + history + the sparse contact-graph goals.
    # One token per goal carries both halves, so a goal specified only by its
    # contact set is still a token the transformer attends to.
    prior_config = ModuleContainerConfig(
        in_keys=STATE_KEYS
        + UNNORMALIZED_STATE_KEYS
        + [
            "masked_mimic_target_poses",
            "masked_mimic_target_masks",
            "masked_mimic_target_times",
            "masked_mimic_target_poses_masks",
            "contact_goal_obs",
            "historical_pose_obs",
        ],
        out_keys=["prior_mu", "prior_logvar"],
        models=[
            # The state token used to normalise inside its own MLP. It cannot
            # any more: contact_state_obs joins it and must NOT be normalised,
            # so the normalisation moves out to explicit processors and the MLP
            # takes the normalised keys plus the raw binary block.
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
            # Already multi-hot in [0, 1]; normalizing would blow up the rare
            # pairs (near-zero variance) for no benefit.
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
            TransformerConfig(
                in_keys=[
                    "current_state_token",
                    "masked_mimic_target_poses_token",
                    "historical_pose_obs_token",
                    "masked_mimic_target_poses_masks",
                ],
                out_keys=["transformer_out"],
                transformer_token_size=transformer_token_size,
                latent_dim=transformer_token_size,
                input_and_mask_mapping={
                    "masked_mimic_target_poses_token": "masked_mimic_target_poses_masks"
                },
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_mu"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_logvar"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
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

    model_config = MaskedMimicModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        trunk=trunk_config,
        vae=MaskedMimicVAEConfig(
            vae_latent_dim=vae_latent_dim,
            vae_noise_type=VAENoiseType.NORMAL,
            kld_schedule=KLDScheduleConfig(start_epoch=500, end_epoch=2000),
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = MimicEvaluatorConfig(
        evaluation_components={
            "gt_error": gt_error_factory(threshold=0.25),
            "gr_error": gr_error_factory(),
            "max_joint_error": max_joint_error_factory(),
        },
        # The Stage-1 easy-128 run accumulated 7.7 GB of predicted motion libs
        # over its lifetime. A distillation run is judged on its checkpoints and
        # on query_contact_goal.py; regenerating a rollout is cheap, so this is
        # off by default rather than quietly filling the disk overnight.
        save_predicted_motion_lib_every=None,
    )

    expert_paths = _expert_paths(args)
    return MultiExpertMaskedMimicAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        # ~250 MB per checkpoint; every 2000 epochs matches the Stage-1 runs and
        # keeps a long unattended run inside a sane disk budget.
        save_epoch_checkpoint_every=2000,
        evaluator=evaluator_config,
        expert_model_path=None,
        expert_model_paths=expert_paths,
        motion_expert_file=getattr(args, "motion_expert_file", None),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: MultiExpertMaskedMimicAgentConfig,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """Strip the experts at inference; the student runs on its prior alone."""
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../mimic/mlp.py"
    )
    apply_inference_overrides_fn(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )

    if agent_cfg is None:
        return

    paths = list(getattr(agent_cfg, "expert_model_paths", None) or [])
    if getattr(agent_cfg, "expert_model_path", None):
        paths = [agent_cfg.expert_model_path] + paths

    if paths and env_cfg is not None and getattr(env_cfg, "observation_components", None):
        from protomotions.agents.supervised.expert_utils import (
            get_expert_observation_keys,
        )
        from protomotions.utils.config_utils import (
            load_resolved_configs_from_checkpoint,
        )

        expert_configs = load_resolved_configs_from_checkpoint(paths[0])
        expert_obs_keys = get_expert_observation_keys(
            expert_configs["env"], expert_configs["agent"]
        )
        for key in expert_obs_keys:
            env_cfg.observation_components.pop(key, None)

    agent_cfg.expert_model_path = None
    agent_cfg.expert_model_paths = []
