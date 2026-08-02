# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base environment implementation for reinforcement learning.

This module provides the foundational environment class for all RL tasks. It integrates
the simulator, handles robot state management, computes observations and rewards, manages
episode resets, and coordinates with terrain and scene systems.

Key Classes:
    - BaseEnv: Core environment class that all tasks inherit from

Key Features:
    - Multi-simulator support (IsaacGym, IsaacLab, Genesis)
    - Terrain integration for complex ground surfaces
    - Scene management for object interaction
    - Motion library integration for reference motions
    - Modular observation components

## BaseEnv

| Member | Type | Why Kept |
|--------|------|----------|
| `config` | `EnvConfig` | Core config, used everywhere |
| `robot_config` | `RobotConfig` | Core config, used everywhere |
| `device` | `torch.device` | Required for tensor creation |
| `terrain` | `Terrain` | Core dependency for terrain queries |
| `scene_lib` | `SceneLib` | Core dependency for scene/object handling |
| `motion_lib` | `MotionLib` | Core dependency for reference motions |
| `simulator` | `Simulator` | Core dependency for physics |
| `num_envs` | `int` | Frequently accessed, avoiding repeated `simulator.num_envs` |
| `max_episode_length` | `int` | Mutable - modified by agent for curriculum learning |
| `dt` | `float` | Frequently accessed, avoiding repeated `simulator.dt` |
| `rew_buf` | `Tensor` | Mutable buffer - accumulates rewards each step |
| `reset_buf` | `Tensor` | Mutable buffer - tracks which envs need reset |
| `progress_buf` | `Tensor` | Mutable buffer - tracks episode progress |
| `terminate_buf` | `Tensor` | Mutable buffer - tracks terminations |
| `extras` | `dict` | Mutable - collects per-step logging data |
| `respawn_root_offset` | `Tensor` | Mutable state - tracks spawn position offsets |
| `skip_height_correction` | `bool` | Performance optimization flag (read-only after init) |
| `motion_manager` | `MotionManager` | Core component for motion sampling |
| `motion_manager_disable_resample` | `bool` | Mutable flag - controlled by evaluator |
| `terrain_obs_cb` | `TerrainObs` | Observation component |
| `scene_obs_cb` | `SceneObs` | Observation component |

"""

from functools import cached_property
import logging
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import torch
from torch import Tensor
from protomotions.utils.hydra_replacement import get_class

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    ObjectState,
    ResetState,
)
from protomotions.envs.terminations import check_max_length_term
from protomotions.envs.context_views import (
    EnvContext,
    CurrentStateView,
    HistoricalView,
    TerrainContext,
    SceneSurfaceContext,
    IsaacLabContactContext,
)
from protomotions.envs.obs.observation_noise import (
    NoisyObservations,
    apply_observation_noise,
    apply_reset_noise,
)
from protomotions.components.terrains.terrain import Terrain
from protomotions.envs.obs.scene_obs import SceneObs
from protomotions.envs.obs.terrain_obs import TerrainObs
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.control.manager import ControlManager

# Component infrastructure for MdpComponent-based configs
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.base_env.utils import (
    combine_rewards,
    combine_terminations,
)
from protomotions.components.pose_lib import build_body_ids_tensor

from protomotions.robot_configs.base import RobotConfig

if TYPE_CHECKING:
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.motion_lib import MotionLib


log = logging.getLogger(__name__)


class BaseEnv:
    """Base class for all reinforcement learning environments.

    Provides core functionality for robot simulation including:
    - Simulator integration (IsaacGym, IsaacLab, Genesis)
    - Terrain management
    - Scene and object handling
    - Motion library integration
    - Observation and reward computation
    - Episode management and resets

    Subclasses should implement task-specific reward functions and
    observation spaces by overriding compute_reward() and compute_observations().

    Attributes:
        simulator: The physics simulator instance.
        num_envs: Number of parallel environments.
        device: PyTorch device for computations.
        terrain: Terrain instance for complex ground surfaces.
        scene_lib: Library of object scenes for interaction tasks.
        motion_lib: Library of reference motions for imitation tasks.

    Example:
        >>> config = SteeringEnvConfig()
        >>> robot_config = G1Config()
        >>> env = Steering(config, robot_config, simulator_config, device)
        >>> obs, _ = env.reset()
        >>> next_obs, rewards, dones, info = env.step(action_dict)
    """

    def __init__(
        self,
        config: EnvConfig,
        robot_config: RobotConfig,
        device: torch.device,
        terrain: "Terrain",
        simulator: Simulator,
        scene_lib: "SceneLib",
        motion_lib: "MotionLib",
        *args,
        **kwargs,
    ):
        """Initialize BaseEnv.

        Args:
            config: Environment configuration
            robot_config: Robot configuration
            device: Device for computation
            terrain: Pre-created Terrain object (always provided, can be None for visualizers)
            simulator: Pre-created Simulator shell (not yet initialized, will be initialized by env)
            scene_lib: Pre-created SceneLib (always provided, empty if no scenes)
            motion_lib: Pre-created MotionLib (always provided, empty if no motions)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self.robot_config = robot_config
        self.device = device
        self.terrain = terrain
        self.scene_lib = scene_lib
        self.motion_lib = motion_lib
        self.simulator = simulator
        self.num_envs = simulator.num_envs

        self.max_episode_length = self.config.max_episode_length

        # Buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.respawn_root_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # Per-episode odometer corruption parameters.
        # Sampled once at episode reset; held constant within the episode.
        # Identity values (scale=1, yaw_bias=0) until first reset.
        self.odom_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.odom_yaw_cos_sin = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.odom_yaw_cos_sin[:, 0] = 1.0  # cos(0) = 1

        # Contact force tracking for impact penalty rewards
        # Initialized properly after simulator init when we know num_bodies
        self.prev_contact_force_magnitudes = None
        self.previous_contact_forces = None
        self.contact_active_state = None
        self.contact_age_steps = None
        self.contact_air_age_steps = None
        self.contact_temporal_valid = None
        # Separate temporal state for the opt-in IsaacLab-normal-force
        # observation.  It cannot share contact_active_state: the legacy
        # tracker deliberately ORs in backend raw flags, whereas this contract
        # uses force-only 5/2 N hysteresis.
        self.isaaclab_previous_normal_force_w = None
        self.isaaclab_previous_active = None
        self.isaaclab_contact_age_s = None
        self.isaaclab_air_age_s = None
        self.isaaclab_contact_temporal_valid = None
        self._isaaclab_contact_sensor_indices = None
        self._isaaclab_contact_body_ids = None
        self._physics_step_count = 0

        # Action buffers (current step only; previous actions come from state_history)
        num_actions = robot_config.number_of_actions
        self._current_raw_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )
        self._current_processed_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )

        # Global context cache - built once per step in post_physics_step
        # and reused by observations, rewards, and terminations
        self._current_context: Dict[str, Any] = None

        # Noisy observation cache - computed once in post_physics_step,
        # reused by both state_history and _build_global_context
        self._current_noisy_obs = None

        self.skip_height_correction = (
            self.config.skip_correct_terrain_height_on_flat and self.terrain.is_flat()
        )

        self.initialize_simulator()

    def initialize_simulator(self):
        """Initialize simulator with task-specific visualization markers.

        Called at the end of __init__ to finalize simulator setup after visualization
        markers have been created (potentially by child env class override).
        """
        self._validate_contact_tracking_config()

        if (
            hasattr(self.robot_config, "kinematic_info")
            and self.robot_config.kinematic_info is not None
        ):
            self.robot_config.kinematic_info.to(self.device)

        # Initialize contact force buffer now that we know num_bodies
        num_bodies = self.robot_config.kinematic_info.num_bodies
        self.prev_contact_force_magnitudes = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.float, device=self.device
        )
        self.previous_contact_forces = torch.zeros(
            self.num_envs,
            num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
        )
        self.contact_active_state = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.bool, device=self.device
        )
        self.contact_age_steps = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.long, device=self.device
        )
        self.contact_air_age_steps = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.long, device=self.device
        )
        self.contact_temporal_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._initialize_isaaclab_contact_buffers()

        if self.config.num_state_history_steps > 0:
            # Check if observation noise is configured - if so, allocate noisy buffers
            store_noisy = (
                self.simulator.config.domain_randomization is not None
                and self.simulator.config.domain_randomization.observation_noise
                is not None
                and self.simulator.config.domain_randomization.observation_noise.has_noise()
            )
            self.state_history = StateHistoryBuffer(
                num_envs=self.num_envs,
                num_history_steps=self.config.num_state_history_steps,
                num_bodies=num_bodies,
                num_dofs=self.robot_config.kinematic_info.num_dofs,
                action_dim=self.robot_config.number_of_actions,
                num_contact_bodies=len(self.contact_body_ids),
                anchor_body_index=self.robot_config.anchor_body_index,
                device=self.device,
                store_noisy=store_noisy,
            )
        else:
            self.state_history = None

        if (
            self.motion_lib.num_motions() > 0
            and self.config.ref_contact_smooth_window > 0
        ):
            self.motion_lib.smooth_contacts(self.config.ref_contact_smooth_window)

        self.dt = self.simulator.dt

        if self.motion_lib.num_motions() > 0:
            self._validate_motion_lib_compatibility()
            self.create_motion_manager()
        else:
            self.motion_manager = None

        self.terrain_obs_cb = TerrainObs(self.terrain.config, self)
        self.scene_obs_cb = SceneObs(self.config.scene_obs, self)

        self._key_bindings = self.simulator.user_interface.scope("env")
        self._key_bindings.register("R", "reset", "Reset all environments")
        self.control_manager = ControlManager(self.config.control_components, self)

        visualization_markers = self.create_visualization_markers(
            self.simulator.headless
        )
        self.simulator._initialize_with_markers(visualization_markers)

        # Component infrastructure for MdpComponent
        self._component_manager = ComponentManager(self.device)
        self._observation_buffer: Dict[str, Tensor] = {}

        # Seed the stateful contact tracker from the first available simulator
        # sample. The initial observation still reports temporal_valid=0; after
        # it is computed, this sample becomes the previous value for the next step.
        current_state = self.simulator.get_robot_state()
        isaaclab_contact_state = self._get_isaaclab_contact_sensor_state()
        self._validate_contact_observation_support(
            current_state, isaaclab_contact_state
        )
        self._update_contact_state(current_state)
        self._log_contact_observation_contract()

        # Initialize observations.
        self._current_context = self._build_global_context(
            current_state, isaaclab_contact_state
        )
        self._initialize_observations()
        self._finalize_contact_state(current_state)
        self._finalize_isaaclab_contact_state(isaaclab_contact_state)

    def _validate_contact_tracking_config(self) -> None:
        """Validate state-tracker and diagnostics parameters before simulation."""
        force_on = float(
            getattr(self.config, "contact_force_on_threshold_n", 5.0)
        )
        force_off = float(
            getattr(self.config, "contact_force_off_threshold_n", 2.0)
        )
        diagnostics_interval = int(
            getattr(self.config, "contact_diagnostics_interval", 0)
        )
        diagnostics_max_envs = int(
            getattr(self.config, "contact_diagnostics_max_envs", 256)
        )
        if (
            not math.isfinite(force_on)
            or not math.isfinite(force_off)
            or force_off < 0.0
            or force_on < force_off
        ):
            raise ValueError(
                "Contact hysteresis requires "
                "contact_force_on_threshold_n >= "
                "contact_force_off_threshold_n >= 0 and finite values; got "
                f"{force_on} and {force_off} N."
            )
        if diagnostics_interval < 0:
            raise ValueError("contact_diagnostics_interval must be >= 0")
        if diagnostics_max_envs < 1:
            raise ValueError("contact_diagnostics_max_envs must be >= 1")

    def _isaaclab_contact_components(self) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Any]]]:
        """Return aggregate and pair components independent of key prefixes.

        MaskedMimic copies expert components under ``expert_``-prefixed keys.
        Looking at the pure function identity (represented by its stable public
        name here to avoid an eager import cycle) keeps the simulator contract
        available to both Stage-1 and Stage-2 configurations.
        """
        core_names = {
            "compute_isaaclab_contact_obs_v1",
        }
        pair_names = {
            "compute_isaaclab_contact_pair_obs_v1",
        }
        core = []
        pair = []
        for key, component in self.config.observation_components.items():
            compute_func = getattr(component, "compute_func", None)
            name = getattr(compute_func, "__name__", "")
            if name in core_names:
                core.append((key, component))
            elif name in pair_names:
                pair.append((key, component))
        return core, pair

    @staticmethod
    def _component_body_ids(component: Any) -> List[int]:
        body_ids = component.static_params.get("body_ids")
        if isinstance(body_ids, Tensor):
            return [int(value) for value in body_ids.tolist()]
        return [int(value) for value in (body_ids or [])]

    def _initialize_isaaclab_contact_buffers(self) -> None:
        """Allocate opt-in temporal state without affecting legacy configs."""
        core_components, pair_components = self._isaaclab_contact_components()
        if not core_components and not pair_components:
            return
        if not core_components:
            raise ValueError(
                "isaaclab_contact_pair_obs_v1 requires an "
                "isaaclab_contact_obs_v1 component with the same body order"
            )

        body_ids = self._component_body_ids(core_components[0][1])
        if not body_ids:
            raise ValueError(
                "isaaclab_contact_obs_v1 requires at least one observation body"
            )
        on_threshold = core_components[0][1].static_params.get(
            "contact_on_threshold_n"
        )
        off_threshold = core_components[0][1].static_params.get(
            "contact_off_threshold_n"
        )
        for key, component in core_components[1:]:
            if self._component_body_ids(component) != body_ids:
                raise ValueError(
                    f"IsaacLab contact component '{key}' uses a different body order"
                )
            if (
                component.static_params.get("contact_on_threshold_n") != on_threshold
                or component.static_params.get("contact_off_threshold_n")
                != off_threshold
            ):
                raise ValueError(
                    "All IsaacLab aggregate contact components must share "
                    "hysteresis thresholds because they share temporal state"
                )
        for key, component in pair_components:
            if self._component_body_ids(component) != body_ids:
                raise ValueError(
                    f"IsaacLab pair contact component '{key}' must use the "
                    "aggregate component's body order"
                )

        self._isaaclab_contact_body_ids = torch.tensor(
            body_ids, dtype=torch.long, device=self.device
        )
        shape = (self.num_envs, len(body_ids))
        self.isaaclab_previous_normal_force_w = torch.zeros(
            *shape, 3, dtype=torch.float, device=self.device
        )
        self.isaaclab_previous_active = torch.zeros(
            *shape, dtype=torch.bool, device=self.device
        )
        self.isaaclab_contact_age_s = torch.zeros(
            *shape, dtype=torch.float, device=self.device
        )
        self.isaaclab_air_age_s = torch.zeros(
            *shape, dtype=torch.float, device=self.device
        )
        self.isaaclab_contact_temporal_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _validate_contact_observation_support(
        self,
        current_state: RobotState,
        isaaclab_contact_state: Optional[Any] = None,
    ) -> None:
        """Fail early when the configured backend cannot satisfy v1's contract."""
        components = self.config.observation_components
        core_components, pair_components = self._isaaclab_contact_components()
        if core_components or pair_components:
            self._validate_isaaclab_contact_observation_support(
                isaaclab_contact_state, core_components, pair_components
            )
        if "contact_obs_v1" not in components:
            return

        backend = getattr(self.simulator.config, "_target_", type(self.simulator).__name__)
        forces = current_state.rigid_body_contact_forces
        contacts = current_state.rigid_body_contacts
        expected_shape = (
            self.num_envs,
            self.robot_config.kinematic_info.num_bodies,
            3,
        )
        if forces is None:
            raise RuntimeError(
                f"contact_obs_v1 requires rigid_body_contact_forces, but backend "
                f"'{backend}' returned None."
            )
        if tuple(forces.shape) != expected_shape:
            raise RuntimeError(
                f"contact_obs_v1 expected rigid_body_contact_forces with shape "
                f"{expected_shape}, but backend '{backend}' returned "
                f"{tuple(forces.shape)}."
            )
        if contacts is None:
            raise RuntimeError(
                f"contact_obs_v1 requires rigid_body_contacts, but backend "
                f"'{backend}' returned None."
            )
        if self.contact_observation_body_ids.numel() == 0:
            raise ValueError(
                "contact_obs_v1 requires at least one "
                "contact_observation_bodies entry."
            )

        expected_ids = self.contact_observation_body_ids.tolist()
        contact_component_ids = components["contact_obs_v1"].static_params.get(
            "body_ids"
        )
        contact_component_ids_list = (
            contact_component_ids.tolist()
            if isinstance(contact_component_ids, Tensor)
            else list(contact_component_ids or [])
        )
        if contact_component_ids_list != expected_ids:
            raise ValueError(
                "contact_obs_v1 body_ids must match the resolved "
                "contact_observation_bodies order. "
                f"Expected {expected_ids}, got {contact_component_ids}."
            )
        if "contact_proximity_obs" in components:
            proximity_ids = components[
                "contact_proximity_obs"
            ].static_params.get("body_ids")
            proximity_ids_list = (
                proximity_ids.tolist()
                if isinstance(proximity_ids, Tensor)
                else list(proximity_ids or [])
            )
            if proximity_ids_list != expected_ids:
                raise ValueError(
                    "contact_proximity_obs body_ids must match the resolved "
                    "contact_observation_bodies order. "
                    f"Expected {expected_ids}, got {proximity_ids}."
                )

    def _get_isaaclab_contact_sensor_state(self) -> Optional[Any]:
        """Read the rich capability only when a configured component needs it."""
        core_components, pair_components = self._isaaclab_contact_components()
        if not core_components and not pair_components:
            return None
        getter = getattr(self.simulator, "get_contact_sensor_state", None)
        return None if getter is None else getter()

    def _validate_isaaclab_contact_observation_support(
        self,
        contact_state: Optional[Any],
        core_components: List[Tuple[str, Any]],
        pair_components: List[Tuple[str, Any]],
    ) -> None:
        """Validate backend capability, body mapping, history, and pair fields."""
        backend = getattr(
            self.simulator.config, "_target_", type(self.simulator).__name__
        )
        if contact_state is None:
            component_keys = [key for key, _ in core_components + pair_components]
            raise RuntimeError(
                f"IsaacLab contact observations {component_keys} require the "
                "rich contact-sensor capability, but backend "
                f"'{backend}' returned None. Use the IsaacLab backend and enable "
                "simulator.contact_sensor_observation."
            )
        if self._isaaclab_contact_body_ids is None:
            raise RuntimeError(
                "IsaacLab contact temporal buffers were not initialized before "
                "capability validation."
            )

        expected_body_ids = [
            int(value) for value in self._isaaclab_contact_body_ids.tolist()
        ]
        if expected_body_ids != self.contact_observation_body_ids.tolist():
            raise ValueError(
                "isaaclab_contact_obs_v1 body_ids must match the resolved "
                "contact_observation_bodies order. Expected "
                f"{self.contact_observation_body_ids.tolist()}, got "
                f"{expected_body_ids}."
            )

        common_sensor_ids = [
            int(value) for value in contact_state.common_body_indices.tolist()
        ]
        if len(set(common_sensor_ids)) != len(common_sensor_ids):
            raise RuntimeError(
                "IsaacLab contact capability reported duplicate common body indices"
            )
        sensor_index_by_common_id = {
            common_id: sensor_id
            for sensor_id, common_id in enumerate(common_sensor_ids)
        }
        missing = [
            body_id
            for body_id in expected_body_ids
            if body_id not in sensor_index_by_common_id
        ]
        if missing:
            missing_names = [
                self.robot_config.kinematic_info.body_names[body_id]
                for body_id in missing
            ]
            raise RuntimeError(
                "IsaacLab observation bodies lack configured contact sensors: "
                f"{missing_names}. Sensor bodies are {contact_state.body_names}."
            )
        self._isaaclab_contact_sensor_indices = torch.tensor(
            [sensor_index_by_common_id[body_id] for body_id in expected_body_ids],
            dtype=torch.long,
            device=self.device,
        )

        expected_history = int(self.simulator.decimation)
        actual_history = int(contact_state.normal_force_history_w.shape[1])
        if not contact_state.history_newest_first:
            raise RuntimeError(
                "IsaacLab contact observation requires newest-first force history"
            )
        if actual_history != expected_history:
            raise RuntimeError(
                "IsaacLab contact history must equal control decimation; got "
                f"H={actual_history}, decimation={expected_history}."
            )
        expected_envs = self.num_envs
        if contact_state.normal_force_w.shape[0] != expected_envs:
            raise RuntimeError(
                "IsaacLab contact capability returned the wrong environment axis: "
                f"{contact_state.normal_force_w.shape[0]} versus {expected_envs}."
            )

        if pair_components:
            required_pair_fields = (
                "filtered_normal_force_w",
                "filtered_normal_force_history_w",
                "filtered_normal_force_valid",
                "filtered_normal_force_history_valid",
                "friction_force_w",
                "friction_force_valid",
                "mean_contact_point_w",
                "mean_contact_point_valid",
                "pair_slot_valid",
            )
            missing_fields = [
                name
                for name in required_pair_fields
                if getattr(contact_state, name) is None
            ]
            if missing_fields:
                raise RuntimeError(
                    "isaaclab_contact_pair_obs_v1 requires filtered normal "
                    "force, true friction force, and mean contact point data; "
                    f"missing fields: {missing_fields}. Enable pair data, "
                    "track_friction_forces, and track_contact_points."
                )
            if contact_state.filter_metadata.num_filters < 1:
                raise RuntimeError(
                    "isaaclab_contact_pair_obs_v1 requires at least one "
                    "terrain or scene-object filter"
                )

    def _log_contact_observation_contract(self) -> None:
        """Log the versioned body ordering and observation dimensions once."""
        components = self.config.observation_components
        core_components, pair_components = self._isaaclab_contact_components()
        if core_components:
            from protomotions.envs.obs import isaaclab_contact_obs_v1_dim

            num_bodies = int(self._isaaclab_contact_body_ids.numel())
            state = self._get_isaaclab_contact_sensor_state()
            num_filters = state.filter_metadata.num_filters
            pair_dim = 15 * num_bodies * num_filters if pair_components else 0
            log.info(
                "IsaacLab contact observation bodies (common order): %s",
                [
                    self.robot_config.kinematic_info.body_names[body_id]
                    for body_id in self._isaaclab_contact_body_ids.tolist()
                ],
            )
            log.info(
                "IsaacLab contact sensor indices=%s common_indices=%s "
                "filters=%s",
                self._isaaclab_contact_sensor_indices.tolist(),
                state.common_body_indices.tolist(),
                state.filter_metadata.labels,
            )
            log.info(
                "IsaacLab contact contract: aggregate=isaaclab_contact_obs_v1 "
                "K=%d H=%d dim=%d; pair=isaaclab_contact_pair_obs_v1 "
                "enabled=%s F=%d dim=%d; normal-force semantics="
                "IsaacLab normal-only; filters=%s",
                num_bodies,
                state.normal_force_history_w.shape[1],
                isaaclab_contact_obs_v1_dim(num_bodies),
                bool(pair_components),
                num_filters,
                pair_dim,
                state.filter_metadata.labels,
            )

        if "contact_obs_v1" not in components:
            return

        from protomotions.envs.obs import contact_obs_v1_dim

        num_bodies = len(self.contact_observation_body_names)
        proximity_dim = (
            num_bodies * 3 if "contact_proximity_obs" in components else 0
        )
        log.info(
            "contact_obs_v1 bodies (kinematic order): %s",
            self.contact_observation_body_names,
        )
        log.info("Contact reward bodies: %s", self.contact_reward_body_names)
        log.info(
            "Contact observation contract: K=%d, contact_obs_v1=%d, "
            "contact_proximity_obs=%d",
            num_bodies,
            contact_obs_v1_dim(num_bodies),
            proximity_dim,
        )

    def _reset_contact_state(self, env_ids: Tensor) -> None:
        """Clear temporal contact state for a partial or full environment reset."""
        self.previous_contact_forces[env_ids] = 0.0
        self.contact_active_state[env_ids] = False
        self.contact_age_steps[env_ids] = 0
        self.contact_air_age_steps[env_ids] = 0
        self.contact_temporal_valid[env_ids] = False

    def _reset_isaaclab_contact_state(self, env_ids: Tensor) -> None:
        """Clear only selected environments in the rich-contact tracker."""
        if getattr(self, "isaaclab_previous_normal_force_w", None) is None:
            return
        self.isaaclab_previous_normal_force_w[env_ids] = 0.0
        self.isaaclab_previous_active[env_ids] = False
        self.isaaclab_contact_age_s[env_ids] = 0.0
        self.isaaclab_air_age_s[env_ids] = 0.0
        self.isaaclab_contact_temporal_valid[env_ids] = False

    def _select_isaaclab_contact_bodies(
        self, tensor: Optional[Tensor], body_axis: int
    ) -> Optional[Tensor]:
        if tensor is None:
            return None
        if self._isaaclab_contact_sensor_indices is None:
            raise RuntimeError(
                "IsaacLab contact sensor-to-observation mapping is unavailable"
            )
        return tensor.index_select(body_axis, self._isaaclab_contact_sensor_indices)

    def _build_isaaclab_contact_context(
        self, contact_state: Optional[Any]
    ) -> Optional[IsaacLabContactContext]:
        """Select one capability sample into the policy's deterministic K order."""
        if getattr(self, "isaaclab_previous_normal_force_w", None) is None:
            return None
        if contact_state is None:
            raise RuntimeError(
                "Configured IsaacLab contact observations have no sensor state"
            )
        if self._isaaclab_contact_sensor_indices is None:
            core, pair = self._isaaclab_contact_components()
            self._validate_isaaclab_contact_observation_support(
                contact_state, core, pair
            )

        normal_force_w = self._select_isaaclab_contact_bodies(
            contact_state.normal_force_w, 1
        )
        normal_force_valid = self._select_isaaclab_contact_bodies(
            contact_state.normal_force_valid, 1
        )
        normal_force_history_valid = self._select_isaaclab_contact_bodies(
            contact_state.normal_force_history_valid, 2
        )
        sensor_data_valid = (
            contact_state.sensor_data_valid
            & normal_force_valid.all(dim=1)
            & normal_force_history_valid.all(dim=(1, 2))
        )
        body_weight_n = contact_state.body_weight_n
        if body_weight_n is None:
            core_components, _ = self._isaaclab_contact_components()
            fallback = float(
                core_components[0][1].static_params.get(
                    "fallback_force_reference_n", 600.0
                )
            )
            body_weight_n = torch.full(
                (self.num_envs, 1),
                fallback,
                dtype=normal_force_w.dtype,
                device=self.device,
            )

        return IsaacLabContactContext(
            normal_force_w=normal_force_w,
            normal_force_history_w=self._select_isaaclab_contact_bodies(
                contact_state.normal_force_history_w, 2
            ),
            normal_force_valid=normal_force_valid,
            normal_force_history_valid=normal_force_history_valid,
            sensor_data_valid=sensor_data_valid,
            filtered_normal_force_w=self._select_isaaclab_contact_bodies(
                contact_state.filtered_normal_force_w, 1
            ),
            filtered_normal_force_history_w=(
                self._select_isaaclab_contact_bodies(
                    contact_state.filtered_normal_force_history_w, 2
                )
            ),
            filtered_normal_force_valid=self._select_isaaclab_contact_bodies(
                contact_state.filtered_normal_force_valid, 1
            ),
            filtered_normal_force_history_valid=(
                self._select_isaaclab_contact_bodies(
                    contact_state.filtered_normal_force_history_valid, 2
                )
            ),
            friction_force_w=self._select_isaaclab_contact_bodies(
                contact_state.friction_force_w, 1
            ),
            friction_force_valid=self._select_isaaclab_contact_bodies(
                contact_state.friction_force_valid, 1
            ),
            mean_contact_point_w=self._select_isaaclab_contact_bodies(
                contact_state.mean_contact_point_w, 1
            ),
            mean_contact_point_valid=self._select_isaaclab_contact_bodies(
                contact_state.mean_contact_point_valid, 1
            ),
            pair_slot_valid=self._select_isaaclab_contact_bodies(
                contact_state.pair_slot_valid, 1
            ),
            body_weight_n=body_weight_n,
            previous_normal_force_w=self.isaaclab_previous_normal_force_w,
            previous_active=self.isaaclab_previous_active,
            previous_contact_age_s=self.isaaclab_contact_age_s,
            previous_air_age_s=self.isaaclab_air_age_s,
            temporal_valid=self.isaaclab_contact_temporal_valid,
        )

    def _finalize_isaaclab_contact_state(
        self,
        contact_state: Optional[Any],
        env_ids: Optional[Tensor] = None,
    ) -> None:
        """Commit the current force-only state after observations consume it."""
        if (
            getattr(self, "isaaclab_previous_normal_force_w", None) is None
            or contact_state is None
        ):
            return
        from protomotions.envs.obs import update_isaaclab_contact_state_v1

        selected_force = self._select_isaaclab_contact_bodies(
            contact_state.normal_force_w, 1
        )
        valid = (
            contact_state.sensor_data_valid
            & self._select_isaaclab_contact_bodies(
                contact_state.normal_force_valid, 1
            ).all(dim=1)
            & self._select_isaaclab_contact_bodies(
                contact_state.normal_force_history_valid, 2
            ).all(dim=(1, 2))
        )
        if env_ids is not None:
            selected_force = selected_force[env_ids]
            valid = valid[env_ids]
            previous_active = self.isaaclab_previous_active[env_ids]
            previous_contact_age = self.isaaclab_contact_age_s[env_ids]
            previous_air_age = self.isaaclab_air_age_s[env_ids]
            temporal_valid = self.isaaclab_contact_temporal_valid[env_ids]
        else:
            previous_active = self.isaaclab_previous_active
            previous_contact_age = self.isaaclab_contact_age_s
            previous_air_age = self.isaaclab_air_age_s
            temporal_valid = self.isaaclab_contact_temporal_valid

        core_components, _ = self._isaaclab_contact_components()
        params = core_components[0][1].static_params
        active, contact_age, air_age, _, _ = update_isaaclab_contact_state_v1(
            normal_force_w=selected_force,
            previous_active=previous_active,
            previous_contact_age_s=previous_contact_age,
            previous_air_age_s=previous_air_age,
            temporal_valid=temporal_valid & valid,
            dt=float(self.dt),
            contact_on_threshold_n=float(params["contact_on_threshold_n"]),
            contact_off_threshold_n=float(params["contact_off_threshold_n"]),
        )

        if env_ids is None:
            scoped_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            destination_ids = valid.nonzero(as_tuple=True)[0]
            source_mask = valid
        else:
            scoped_ids = env_ids
            destination_ids = env_ids[valid]
            source_mask = valid
        invalid_ids = scoped_ids[~valid]
        if invalid_ids.numel() > 0:
            # Never bridge force deltas, hysteresis, transitions, or ages over
            # a reset/non-finite sensor interval.  The next valid sample is a
            # fresh first sample with temporal_valid=False.
            self._reset_isaaclab_contact_state(invalid_ids)
        if destination_ids.numel() == 0:
            return
        self.isaaclab_previous_normal_force_w[destination_ids] = selected_force[
            source_mask
        ]
        self.isaaclab_previous_active[destination_ids] = active[source_mask]
        self.isaaclab_contact_age_s[destination_ids] = contact_age[source_mask]
        self.isaaclab_air_age_s[destination_ids] = air_age[source_mask]
        self.isaaclab_contact_temporal_valid[destination_ids] = True

    @staticmethod
    def _clear_reset_contact_sample(
        current_state: RobotState,
        env_ids: Tensor,
    ) -> RobotState:
        """Mask contact buffers that may be stale immediately after teleport.

        Some simulator backends, including Isaac Lab, update articulation
        kinematics immediately when reset state is written but do not refresh
        contact sensors until the next physics step. Reset-return observations
        therefore use an explicit no-contact sample. The first post-physics
        sample exposes current contact force with ``temporal_valid=False`` and
        becomes the previous sample only after that observation is computed.

        The returned state is a clone so a backend-owned state object or tensor
        can never be modified through a returned view.
        """
        reset_observation_state = current_state.clone()
        if reset_observation_state.rigid_body_contact_forces is not None:
            reset_observation_state.rigid_body_contact_forces[env_ids] = 0.0
        if reset_observation_state.rigid_body_contacts is not None:
            reset_observation_state.rigid_body_contacts[env_ids] = False
        return reset_observation_state

    def _update_contact_state(
        self,
        current_state: RobotState,
        env_ids: Optional[Tensor] = None,
    ) -> None:
        """Update hysteresis and duration buffers from one current state sample."""
        if (
            current_state.rigid_body_contact_forces is None
            or current_state.rigid_body_contacts is None
        ):
            return

        from protomotions.envs.obs import update_contact_state

        if env_ids is None:
            raw_contacts = current_state.rigid_body_contacts
            contact_forces = current_state.rigid_body_contact_forces
            previous_active = self.contact_active_state
            previous_contact_age = self.contact_age_steps
            previous_air_age = self.contact_air_age_steps
        else:
            raw_contacts = current_state.rigid_body_contacts[env_ids]
            contact_forces = current_state.rigid_body_contact_forces[env_ids]
            previous_active = self.contact_active_state[env_ids]
            previous_contact_age = self.contact_age_steps[env_ids]
            previous_air_age = self.contact_air_age_steps[env_ids]

        active, contact_age, air_age = update_contact_state(
            raw_contacts=raw_contacts,
            contact_forces=contact_forces,
            previous_active=previous_active,
            previous_contact_age_steps=previous_contact_age,
            previous_air_age_steps=previous_air_age,
            force_on_threshold_n=float(
                getattr(self.config, "contact_force_on_threshold_n", 5.0)
            ),
            force_off_threshold_n=float(
                getattr(self.config, "contact_force_off_threshold_n", 2.0)
            ),
        )

        if env_ids is None:
            self.contact_active_state.copy_(active)
            self.contact_age_steps.copy_(contact_age)
            self.contact_air_age_steps.copy_(air_age)
        else:
            self.contact_active_state[env_ids] = active
            self.contact_age_steps[env_ids] = contact_age
            self.contact_air_age_steps[env_ids] = air_age

    def _finalize_contact_state(
        self,
        current_state: RobotState,
        env_ids: Optional[Tensor] = None,
    ) -> None:
        """Promote the current force to the previous sample after observation."""
        forces = current_state.rigid_body_contact_forces
        if forces is None:
            return
        if env_ids is None:
            self.previous_contact_forces.copy_(forces)
            self.contact_temporal_valid.fill_(True)
        else:
            self.previous_contact_forces[env_ids] = forces[env_ids]
            self.contact_temporal_valid[env_ids] = True

    def _record_contact_diagnostics(self, current_state: RobotState) -> None:
        """Record sampled aggregate statistics without changing policy inputs."""
        interval = int(getattr(self.config, "contact_diagnostics_interval", 0))
        components = self.config.observation_components
        if (
            interval <= 0
            or "contact_obs_v1" not in components
            or self._physics_step_count % interval != 0
        ):
            return

        num_sample_envs = min(
            self.num_envs,
            int(getattr(self.config, "contact_diagnostics_max_envs", 256)),
        )
        ids = self.contact_observation_body_ids
        forces = current_state.rigid_body_contact_forces[:num_sample_envs, ids]
        previous = self.previous_contact_forces[:num_sample_envs, ids]
        active = self.contact_active_state[:num_sample_envs, ids]
        valid = self.contact_temporal_valid[:num_sample_envs]
        force_rate = (forces - previous) / max(float(self.dt), 1e-6)
        force_rate = torch.where(
            valid[:, None, None], force_rate, torch.zeros_like(force_rate)
        )

        force_magnitude = torch.linalg.vector_norm(forces, dim=-1)
        force_rate_magnitude = torch.linalg.vector_norm(force_rate, dim=-1)
        upward = torch.clamp(forces[..., 2], min=0.0)
        horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)

        params = components["contact_obs_v1"].static_params
        force_clip = float(params.get("force_clip_n", 5000.0))
        force_rate_clip = float(
            params.get("force_rate_clip_n_per_s", 50000.0)
        )
        friction_mu = float(params.get("friction_mu", 1.0))
        friction_clip = float(params.get("friction_utilization_clip", 2.0))
        friction = horizontal / (friction_mu * upward + 1e-6)
        friction = torch.clamp(friction / friction_clip, 0.0, 1.0)

        self.extras.update(
            {
                "contact/active_body_fraction": active.float().mean(),
                "contact/any_selected_contact_fraction": active.any(dim=1)
                .float()
                .mean(),
                "contact/force_magnitude_mean": force_magnitude.mean(),
                "contact/force_magnitude_p95": torch.quantile(
                    force_magnitude, 0.95
                ),
                "contact/force_magnitude_p99": torch.quantile(
                    force_magnitude, 0.99
                ),
                "contact/force_rate_magnitude_p95": torch.quantile(
                    force_rate_magnitude, 0.95
                ),
                "contact/force_rate_magnitude_p99": torch.quantile(
                    force_rate_magnitude, 0.99
                ),
                "contact/max_contact_age_s": (
                    self.contact_age_steps[:num_sample_envs, ids].max()
                    * float(self.dt)
                ),
                "contact/force_component_clip_fraction": (
                    forces.abs() >= force_clip
                )
                .float()
                .mean(),
                "contact/force_rate_component_clip_fraction": (
                    force_rate.abs() >= force_rate_clip
                )
                .float()
                .mean(),
                "contact/friction_utilization_proxy_mean": friction.mean(),
                "contact/friction_utilization_proxy_max": friction.max(),
                "contact/num_observation_bodies": torch.tensor(
                    float(ids.numel()), device=self.device
                ),
            }
        )

    def _record_isaaclab_contact_diagnostics(
        self, contact_state: Optional[Any]
    ) -> None:
        """Record scale, clipping, lifecycle, and pair-validity diagnostics."""
        interval = int(getattr(self.config, "contact_diagnostics_interval", 0))
        core_components, pair_components = self._isaaclab_contact_components()
        if (
            interval <= 0
            or not core_components
            or contact_state is None
            or self._physics_step_count % interval != 0
        ):
            return

        num_sample_envs = min(
            self.num_envs,
            int(getattr(self.config, "contact_diagnostics_max_envs", 256)),
        )
        contact = self._build_isaaclab_contact_context(contact_state)
        force = contact.normal_force_w[:num_sample_envs]
        history = contact.normal_force_history_w[:num_sample_envs]
        weight = contact.body_weight_n[:num_sample_envs].clamp_min(1.0e-4)
        force_bw = force / weight[:, None, :]
        history_bw = history / weight[:, None, None, :]
        params = core_components[0][1].static_params
        force_clip = float(params.get("force_clip_bodyweights", 10.0))
        force_delta_clip = float(
            params.get("force_delta_clip_bodyweights", 10.0)
        )
        history_norm_bw = torch.linalg.vector_norm(history_bw, dim=-1)
        sensor_valid = contact.sensor_data_valid[:num_sample_envs]
        temporal_valid = (
            contact.temporal_valid[:num_sample_envs] & sensor_valid
        )
        previous_force = contact.previous_normal_force_w[:num_sample_envs]
        force_delta_bw = torch.where(
            temporal_valid[:, None, None],
            (force - previous_force) / weight[:, None, :],
            torch.zeros_like(force),
        )

        from protomotions.envs.obs import update_isaaclab_contact_state_v1

        active, contact_age, air_age, _, _ = update_isaaclab_contact_state_v1(
            normal_force_w=force,
            previous_active=contact.previous_active[:num_sample_envs],
            previous_contact_age_s=(
                contact.previous_contact_age_s[:num_sample_envs]
            ),
            previous_air_age_s=contact.previous_air_age_s[:num_sample_envs],
            temporal_valid=temporal_valid,
            dt=float(self.dt),
            contact_on_threshold_n=float(params["contact_on_threshold_n"]),
            contact_off_threshold_n=float(params["contact_off_threshold_n"]),
        )
        active &= sensor_valid[:, None]

        force_norm_bw = torch.linalg.vector_norm(force_bw, dim=-1)
        upward_support_bw = torch.clamp(force[..., 2], min=0.0).sum(
            dim=1
        ) / weight.squeeze(-1)
        upward = torch.clamp(force[..., 2], min=0.0)
        upward_total = upward.sum(dim=1, keepdim=True)
        load_fraction = upward / (upward_total + 1.0e-6)
        if force.shape[1] > 1:
            support_entropy = -(
                load_fraction
                * torch.log(torch.clamp(load_fraction, min=1.0e-6))
            ).sum(dim=1) / math.log(force.shape[1])
            support_entropy = torch.where(
                upward_total.squeeze(1) > 1.0e-6,
                support_entropy,
                torch.zeros_like(support_entropy),
            )
        else:
            support_entropy = torch.zeros_like(upward_support_bw)

        diagnostics = {
            "isaaclab_contact/sensor_data_valid_fraction": (
                sensor_valid.float().mean()
            ),
            "isaaclab_contact/body_weight_n_mean": weight.mean(),
            "isaaclab_contact/normal_force_bodyweights_mean": force_norm_bw.mean(),
            "isaaclab_contact/normal_force_bodyweights_max": force_norm_bw.max(),
            "isaaclab_contact/substep_peak_bodyweights_p95": torch.quantile(
                history_norm_bw.amax(dim=1), 0.95
            ),
            "isaaclab_contact/substep_peak_bodyweights_p99": torch.quantile(
                history_norm_bw.amax(dim=1), 0.99
            ),
            "isaaclab_contact/force_component_clip_fraction": (
                force_bw.abs() >= force_clip
            )
            .float()
            .mean(),
            "isaaclab_contact/force_delta_component_clip_fraction": (
                force_delta_bw.abs() >= force_delta_clip
            )
            .float()
            .mean(),
            "isaaclab_contact/active_body_count_mean": active.float()
            .sum(dim=1)
            .mean(),
            "isaaclab_contact/contact_age_s_mean": contact_age.mean(),
            "isaaclab_contact/air_age_s_mean": air_age.mean(),
            "isaaclab_contact/upward_support_bodyweights_mean": (
                upward_support_bw.mean()
            ),
            "isaaclab_contact/support_load_entropy_mean": support_entropy.mean(),
            "isaaclab_contact/aggregate_measurement_valid_fraction": (
                contact.normal_force_valid[:num_sample_envs].float().mean()
            ),
            "isaaclab_contact/history_measurement_valid_fraction": (
                contact.normal_force_history_valid[:num_sample_envs]
                .float()
                .mean()
            ),
            "isaaclab_contact/num_observation_bodies": torch.tensor(
                float(force.shape[1]), device=self.device
            ),
        }
        for body_idx, common_body_id in enumerate(
            self._isaaclab_contact_body_ids.tolist()
        ):
            body_name = self.robot_config.kinematic_info.body_names[common_body_id]
            diagnostics[
                f"isaaclab_contact/active_fraction_by_body/{body_name}"
            ] = active[:, body_idx].float().mean()
        if pair_components:
            slots = contact.pair_slot_valid[:num_sample_envs]
            points = contact.mean_contact_point_valid[:num_sample_envs]
            friction = contact.friction_force_valid[:num_sample_envs]
            filtered_valid = contact.filtered_normal_force_valid[
                :num_sample_envs
            ]
            pair_measurement_valid = (
                slots
                & sensor_valid[:, None, None]
                & filtered_valid
                & friction
            )
            pair_normal_norm = torch.linalg.vector_norm(
                contact.filtered_normal_force_w[:num_sample_envs], dim=-1
            )
            pair_friction_norm = torch.linalg.vector_norm(
                contact.friction_force_w[:num_sample_envs], dim=-1
            )
            pair_ratio = pair_friction_norm / (pair_normal_norm + 1.0e-4)
            valid_ratios = pair_ratio[pair_measurement_valid]
            if valid_ratios.numel() == 0:
                ratio_p50 = torch.zeros((), device=self.device)
                ratio_p95 = torch.zeros((), device=self.device)
            else:
                ratio_p50 = torch.quantile(valid_ratios, 0.50)
                ratio_p95 = torch.quantile(valid_ratios, 0.95)
            diagnostics.update(
                {
                    "isaaclab_contact/pair_slot_valid_fraction": (
                        slots.float().mean()
                    ),
                    "isaaclab_contact/mean_contact_point_valid_fraction": (
                        (points & slots).float().sum()
                        / slots.float().sum().clamp_min(1.0)
                    ),
                    "isaaclab_contact/friction_measurement_valid_fraction": (
                        (friction & slots).float().sum()
                        / slots.float().sum().clamp_min(1.0)
                    ),
                    "isaaclab_contact/friction_to_normal_ratio_p50": ratio_p50,
                    "isaaclab_contact/friction_to_normal_ratio_p95": ratio_p95,
                }
            )
        self.extras.update(diagnostics)

    def _validate_motion_lib_compatibility(self):
        """Validate that the motion file is compatible with the robot config."""
        ki = self.robot_config.kinematic_info
        expected_dofs = ki.num_dofs
        expected_bodies = ki.num_bodies

        sample_state = self.motion_lib.get_motion_state(
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.zeros(1, device=self.device),
        )
        motion_dofs = sample_state.dof_pos.shape[1]
        motion_bodies = sample_state.rigid_body_pos.shape[1]

        if motion_dofs != expected_dofs or motion_bodies != expected_bodies:
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"MOTION FILE / ROBOT MISMATCH\n"
                f"{'=' * 70}\n"
                f"Motion file has {motion_dofs} DOFs and {motion_bodies} bodies,\n"
                f"but robot '{type(self.robot_config).__name__}' expects "
                f"{expected_dofs} DOFs and {expected_bodies} bodies.\n\n"
                f"The motion file was likely generated for a different robot.\n"
                f"Make sure --motion-file matches the robot in your "
                f"checkpoint/config.\n"
                f"{'=' * 70}"
            )

    ###############################################################
    # Getters
    ###############################################################
    def is_simulation_running(self):
        """Check if the physics simulation is running.

        Returns:
            Boolean indicating simulation state
        """
        return self.simulator.is_simulation_running()

    def get_obs(self):
        """Gather observations from all components.

        Returns:
            Dictionary of observation tensors from humanoid, terrain, scene,
            and dynamic observation components
        """
        obs = {}
        terrain_obs = self.terrain_obs_cb.get_obs()
        obs.update(terrain_obs)
        if self.scene_lib.num_scenes() > 0 and self.config.scene_obs.enabled:
            scene_obs = self.scene_obs_cb.get_obs()
            obs.update(scene_obs)

        # Get dynamic observations
        dynamic_obs = {
            name: tensor.clone() for name, tensor in self._observation_buffer.items()
        }
        obs.update(dynamic_obs)

        return obs

    def get_action_size(self):
        """Get the dimensionality of the action space.

        Returns:
            Number of action dimensions
        """
        return self.simulator.num_act

    def consume_reset_request(self) -> bool:
        """Return and consume a user-interface reset request."""
        return self._key_bindings.reset.consume()

    ###############################################################
    # Component Processing
    ###############################################################
    def _initialize_observations(self):
        """Initialize observation buffers."""
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._process_observations(self.context, all_env_ids)

    def _process_observations(self, context: EnvContext, env_ids: Tensor):
        """Process observations using MdpComponent."""
        raw_obs = self._component_manager.execute_all(
            components=self.config.observation_components,
            ctx=context,
        )

        # Update observation buffer with results
        for name, obs_value in raw_obs.items():
            if name not in self._observation_buffer:
                self._observation_buffer[name] = torch.zeros(
                    self.num_envs,
                    obs_value.shape[-1],
                    dtype=obs_value.dtype,
                    device=self.device,
                )
            # MdpComponent always computes for all envs, update specified subset
            self._observation_buffer[name][env_ids] = obs_value[env_ids]

    def _process_rewards(
        self, context: EnvContext, grace_mask: Optional[Tensor] = None
    ):
        """Process rewards using MdpComponent."""
        raw_rewards = self._component_manager.execute_all(
            components=self.config.reward_components,
            ctx=context,
        )

        return combine_rewards(
            raw_rewards=raw_rewards,
            configs=self.config.reward_components,
            grace_mask=grace_mask,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _process_terminations(self, context: EnvContext):
        """Process terminations using MdpComponent."""
        raw_terms = self._component_manager.execute_all(
            components=self.config.termination_components,
            ctx=context,
        )

        return combine_terminations(
            raw_terms=raw_terms,
            configs=self.config.termination_components,
            num_envs=self.num_envs,
            device=self.device,
        )

    _action_config_device_ready: bool = False

    def _process_action(self, action: Tensor, context: EnvContext) -> Dict[str, Tensor]:
        """Process action using single action config dict.

        action_config is a single dict with "fn" key and parameters.
        """
        if self.config.action_config is None:
            return {"processed_action": action}

        # Lazy device migration on first call
        if not self._action_config_device_ready:
            for key, val in self.config.action_config.items():
                if isinstance(val, torch.Tensor):
                    self.config.action_config[key] = val.to(action.device)
            self._action_config_device_ready = True

        fn = self.config.action_config["fn"]
        # Extract all params except "fn"
        params = {k: v for k, v in self.config.action_config.items() if k != "fn"}
        params["action"] = action
        return fn(**params)

    ###############################################################
    # Cached Properties
    ###############################################################
    @cached_property
    def contact_body_ids(self) -> torch.Tensor:
        """Body indices for contact sensing."""
        return build_body_ids_tensor(
            self.robot_config.kinematic_info.body_names,
            self.robot_config.contact_bodies,
            self.device,
        )

    @cached_property
    def contact_observation_body_names(self) -> List[str]:
        """Resolved names exposed by contact observations."""
        body_names = getattr(
            self.robot_config, "contact_observation_bodies", None
        )
        if body_names is None:
            body_names = self.robot_config.contact_bodies
        return list(body_names or [])

    @cached_property
    def contact_observation_body_ids(self) -> torch.Tensor:
        """Body indices exposed by contact observations."""
        return build_body_ids_tensor(
            self.robot_config.kinematic_info.body_names,
            self.contact_observation_body_names,
            self.device,
        )

    @cached_property
    def contact_reward_body_names(self) -> List[str]:
        """Resolved names used by contact-matching rewards."""
        body_names = getattr(self.robot_config, "contact_reward_bodies", None)
        if body_names is None:
            body_names = self.robot_config.contact_bodies
        return list(body_names or [])

    @cached_property
    def contact_reward_body_ids(self) -> torch.Tensor:
        """Body indices used by contact-matching rewards."""
        return build_body_ids_tensor(
            self.robot_config.kinematic_info.body_names,
            self.contact_reward_body_names,
            self.device,
        )

    @cached_property
    def non_termination_contact_body_ids(self) -> torch.Tensor:
        """Body indices that don't trigger termination on contact."""
        body_names = self.robot_config.kinematic_info.body_names
        if self.robot_config.non_termination_contact_bodies == "all":
            return build_body_ids_tensor(body_names, body_names, self.device)
        else:
            return build_body_ids_tensor(
                body_names,
                self.robot_config.non_termination_contact_bodies,
                self.device,
            )

    @cached_property
    def default_reset_state(self) -> ResetState:
        """Default robot reset state from simulator."""
        return self.simulator.get_default_robot_reset_state()

    @cached_property
    def default_object_state(self) -> ObjectState:
        """Default object state (empty if no scenes)."""
        return self.scene_lib.get_default_object_state(self.device)

    def update_respawn_root_offset_by_env_ids(
        self,
        env_ids,
        ref_state: Optional[RobotState] = None,
        sample_flat: bool = False,
    ) -> torch.Tensor:
        """
        Samples a new starting position for the environment.
        And obtains the root translation offset relative to the reference state.

        This method considers both scene and terrain requirements.

        When a scene is required for obj interaction,
        the character is spawned relative to the scene's position.

        For environments without a scene, a random valid coordinate is sampled,
        and non-negative vertical offset is added based on terrain height.

        During co-training, scene groups use flat terrain, but during
        inference the resolved terrain may be complex (with negative heights
        that get normalised).  Height correction is applied to both scene
        and non-scene envs unless the terrain is entirely flat.

        """

        respawn_offset = torch.zeros((len(env_ids), 3), device=self.device)

        # Get boolean masks for scene vs non-scene envs
        scene_mask, non_scene_mask = self.get_scene_non_scene_mask(env_ids)

        if scene_mask.any():
            scene_pos = self.scene_lib.get_scene_positions(self.terrain, self.device)
            respawn_offset[scene_mask, :2] = scene_pos[env_ids[scene_mask], :2]

            # Scene envs also need terrain height correction — the object
            # playground is flat at height-field 0, but terrain normalisation
            # (shifting min height to z=0) can raise the playground above
            # world z=0.  Without correction the agent spawns underground.
            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[scene_mask, 2] = terrain_heights

        if non_scene_mask.any():
            num_non_scene = non_scene_mask.sum().item()
            respawn_position_xy = self.terrain.sample_valid_locations(
                num_envs=num_non_scene, sample_flat=sample_flat
            )

            if ref_state is None:
                ref_root = torch.zeros((num_non_scene, 2), device=self.device)
            else:
                ref_root = ref_state.root_pos[non_scene_mask, :2]
            respawn_offset[non_scene_mask, :2] = respawn_position_xy - ref_root

            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[non_scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        non_scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[non_scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[non_scene_mask, 2] = terrain_heights

        respawn_offset[:, 2] += self.config.ref_respawn_offset

        self.respawn_root_offset[env_ids] = respawn_offset

    def align_motion_with_humanoid(self, env_ids, root_pos):
        """Compute XY offset between humanoid spawn position and reference motion data.

        Args:
            env_ids: Environment indices to align
            root_pos: Desired root positions [len(env_ids), 3]
        """
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids],
            self.motion_manager.motion_times[env_ids],
        )

        self.respawn_root_offset[env_ids, :2] = (
            root_pos[:, :2] - ref_state.rigid_body_pos[:, 0, :2]
        )

    def get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        self, target_pos: Tensor, env_ids: Optional[Tensor] = None
    ) -> Tensor:
        """Compute spawn offset with terrain height correction for reference poses.

        Used by motion tracking tasks to correctly position reference poses in the environment,
        accounting for both XY spawn offset and terrain height.

        Args:
            target_pos: Reference body positions [num_envs, num_bodies, 3]
                       without spawning offset applied.
            env_ids: Environment indices [num_envs]. If None, uses all envs.

        Returns:
            Offset to add to target_pos [num_envs, num_bodies, 3].

        Note:
            - For XY offset: all bodies share the same respawn_root_offset
            - For Z offset: all bodies share the same offset computed from
              the body furthest below terrain
            - This preserves the rigid body structure during spawning
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        new_offset = torch.zeros_like(target_pos)
        new_offset[:, :, :2] = self.respawn_root_offset[env_ids, :2][:, None, :]

        if not self.skip_height_correction:
            target_pos_spawned = target_pos.clone() + new_offset
            z_offset = self.terrain.find_terrain_height_for_max_below_body(
                target_pos_spawned
            )
            new_offset[:, :, 2] = z_offset.unsqueeze(1)

        return new_offset

    def get_scene_non_scene_mask(self, env_ids):
        """
        Returns boolean masks indicating which envs require a scene and which don't.

        Args:
            env_ids: Environment IDs to check

        Returns:
            scene_mask: Boolean tensor (len(env_ids),) - True for scene envs
            non_scene_mask: Boolean tensor (len(env_ids),) - True for non-scene envs

        Note: For now assumes either all or none require a scene
        """
        num_envs = len(env_ids)
        if self.scene_lib.num_scenes() > 0:
            scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        else:
            scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        return scene_mask, non_scene_mask

    def get_markers_state(self):
        """Compute visualization marker positions for rendering.

        Returns:
            Dictionary mapping marker names to MarkerState objects
        """
        if self.simulator.headless:
            return {}

        markers_state = {}

        # Update terrain markers
        if self.config.show_terrain_markers:
            height_maps = self.terrain.get_height_maps(
                self.simulator.get_root_state(), None, return_all_dims=True
            ).view(self.num_envs, -1, 3)
            markers_state["terrain_markers"] = MarkerState(
                translation=height_maps,
                orientation=torch.zeros(
                    self.num_envs, height_maps.shape[1], 4, device=self.device
                ),
            )

        # Merge markers from control components
        control_markers_state = self.control_manager.get_markers_state()
        markers_state.update(control_markers_state)

        return markers_state

    ###############################################################
    # Environment step logic
    ###############################################################
    def step(self, action: Tensor):
        """Step the environment forward one timestep.

        Args:
            action: Raw action tensor from the policy [num_envs, num_actions]

        Returns:
            obs, rewards, dones, terminated, extras
        """
        self.extras = {}

        # Invalidate cached context - will be rebuilt after physics in post_physics_step
        self._current_context = None
        self._current_noisy_obs = None

        # Store current actions
        self._current_raw_action[:] = action

        # Process action
        action_dict = self._process_action(action, self.context)
        processed_action = action_dict["processed_action"]
        self._current_processed_action[:] = processed_action

        self.simulator.step(processed_action, markers_callback=self.get_markers_state)

        self.post_physics_step()

        if self.consume_reset_request():
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras

    def on_epoch_end(self, current_epoch: int):
        """Hook called at end of each training epoch. Override in subclasses if needed.

        Args:
            current_epoch: Current epoch number
        """
        pass

    def post_physics_step(self):
        """Update environment state after physics simulation step.

        Increments progress counter, updates motion manager, computes observations and rewards,
        checks for resets, and stores raw robot state in extras for logging.
        """
        self.progress_buf += 1
        self._physics_step_count += 1
        current_state = self.simulator.get_robot_state()
        isaaclab_contact_state = self._get_isaaclab_contact_sensor_state()
        self._update_contact_state(current_state)

        if self.state_history is not None:
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[:, 0]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[
                :, self.contact_body_ids
            ].bool()

            # Compute noisy versions if observation noise is configured and history stores noisy data
            noisy_kwargs = {}
            if self.state_history.store_noisy:
                obs_noise_cfg = (
                    self.simulator.config.domain_randomization.observation_noise
                )

                # Single source of truth: uniform noise via apply_observation_noise
                noisy = apply_observation_noise(
                    obs_noise_cfg=obs_noise_cfg,
                    robot_state=current_state,
                    anchor_idx=self.robot_config.anchor_body_index,
                    ground_heights=ground_heights,
                )
                self._current_noisy_obs = noisy

                # Extract noisy tensors for history buffer
                noisy_kwargs["noisy_rigid_body_pos"] = noisy.rigid_body_pos
                noisy_kwargs["noisy_rigid_body_rot"] = noisy.rigid_body_rot
                noisy_kwargs["noisy_rigid_body_vel"] = noisy.rigid_body_vel
                noisy_kwargs["noisy_rigid_body_ang_vel"] = noisy.rigid_body_ang_vel
                noisy_kwargs["noisy_dof_pos"] = noisy.dof_pos
                noisy_kwargs["noisy_dof_vel"] = noisy.dof_vel
                noisy_kwargs["noisy_ground_heights"] = noisy.ground_heights

            self.state_history.rotate_and_update(
                rigid_body_pos=current_state.rigid_body_pos,
                rigid_body_rot=current_state.rigid_body_rot,
                rigid_body_vel=current_state.rigid_body_vel,
                rigid_body_ang_vel=current_state.rigid_body_ang_vel,
                dof_pos=current_state.dof_pos,
                dof_vel=current_state.dof_vel,
                actions=self._current_raw_action,
                ground_heights=ground_heights,
                body_contacts=body_contacts,
                processed_actions=self._current_processed_action,
                **noisy_kwargs,
            )

        if self.motion_manager is not None and hasattr(
            self.motion_manager, "post_physics_step"
        ):
            self.motion_manager.post_physics_step()

        self.control_manager.step()

        if (
            self.motion_manager is not None
            and self.motion_manager.config.realign_motion_with_humanoid_on_each_step
        ):
            # When realign_motion_with_humanoid_on_each_step is True, we re-align before computing observations and rewards.
            # This ensures the robot only matches the local-pose with global orientation.
            self.align_motion_with_humanoid(
                torch.arange(self.num_envs, device=self.device, dtype=torch.long),
                self.simulator.get_root_state().root_pos,
            )

        # Build context once and reuse for observations, rewards, and terminations
        self._current_context = self._build_global_context(
            current_state, isaaclab_contact_state
        )

        self.compute_observations(context=self._current_context)
        self.compute_reward(context=self._current_context)
        self.reset_buf[:], self.terminate_buf[:] = self.check_resets_and_terminations(
            context=self._current_context
        )

        self.extras["terminate"] = self.terminate_buf

        rbs = current_state
        for k, _ in rbs.get_shape_mapping(flattened=True).items():
            self.extras[f"raw/{k}"] = rbs.flatten_bodies(k)

        self._record_contact_diagnostics(rbs)
        self._record_isaaclab_contact_diagnostics(isaaclab_contact_state)

        # Update previous contact forces for next step's impact penalty
        self.prev_contact_force_magnitudes[:] = torch.norm(
            rbs.rigid_body_contact_forces, dim=-1
        )
        self._finalize_contact_state(rbs)
        self._finalize_isaaclab_contact_state(isaaclab_contact_state)

    def user_reset(self):
        """Force environments to reset on next check (triggered by user input)."""
        self.progress_buf[:] = 100000000000

    def compute_observations(self, env_ids=None, context: EnvContext = None):
        """Compute observations for specified environments.

        Args:
            env_ids: Environment indices to update (None = all environments)
            context: Pre-built EnvContext from self.context property.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if context is None:
            raise ValueError("context is required - use self.context to build it")

        # Process dynamic observations
        self._process_observations(context, env_ids)

        self.terrain_obs_cb.compute_observations(env_ids)
        if self.scene_lib.num_scenes() > 0:
            self.scene_obs_cb.compute_observations(env_ids)

    def check_resets_and_terminations(self, context: EnvContext):
        """Check reset and termination conditions.

        Only handles max episode length directly. All other terminations
        (including height/fall termination) should be configured via:
        - termination_components (dynamic termination system)
        - control_components (task-specific terminations)

        Args:
            context: Pre-built context from self.context property.

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        max_length_reached = check_max_length_term(
            self.progress_buf, self.max_episode_length
        )
        reset_buf = max_length_reached.clone()
        terminated = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        comp_reset, comp_terminate = (
            self.control_manager.check_resets_and_terminations()
        )
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate

        # Process terminations
        comp_reset, comp_terminate, term_logging = self._process_terminations(context)
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate
        self.extras.update(term_logging)

        return reset_buf, terminated

    ###############################################################
    # Dynamic Reward System
    ###############################################################
    @property
    def context(self) -> EnvContext:
        """Get global context for observation/reward/termination evaluation.

        Returns cached context from _current_context if set (after post_physics_step),
        otherwise builds a fresh context.

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        if self._current_context is None:
            self._current_context = self._build_global_context()
        return self._current_context

    def _build_global_context(
        self,
        current_state: Optional[RobotState] = None,
        isaaclab_contact_state: Optional[Any] = None,
    ) -> EnvContext:
        """Build a fresh global context for observations, rewards, and terminations.

        Creates typed EnvContext with view wrappers around existing data structures.
        Controllers populate their task-specific views via populate_context().

        When observation noise is configured:
        - noisy views have noise applied
        - current views contain clean data

        When no observation noise is configured:
        - Both point to the same tensors (memory efficient)

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        if current_state is None:
            current_state = self.simulator.get_robot_state()
        if (
            isaaclab_contact_state is None
            and getattr(self, "isaaclab_previous_normal_force_w", None) is not None
        ):
            isaaclab_contact_state = self._get_isaaclab_contact_sensor_state()
        anchor_idx = self.robot_config.anchor_body_index

        ground_heights = self.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        ).squeeze(-1)

        body_contacts = current_state.rigid_body_contacts[
            :, self.contact_body_ids
        ].bool()

        # Contact force magnitudes for impact penalty rewards
        current_contact_force_magnitudes = torch.norm(
            current_state.rigid_body_contact_forces, dim=-1
        )

        # Use cached noisy obs from post_physics_step when available.
        # During init/reset the cache is None — use clean (no-noise) fallback.
        if self._current_noisy_obs is not None:
            noisy = self._current_noisy_obs
        else:
            noisy = apply_observation_noise(
                obs_noise_cfg=None,
                robot_state=current_state,
                anchor_idx=anchor_idx,
                ground_heights=ground_heights,
            )

        scene_surface_context = self._build_scene_surface_context()
        isaaclab_contact_context = self._build_isaaclab_contact_context(
            isaaclab_contact_state
        )

        # Build context with view wrappers
        ctx = EnvContext(
            # Core state views (wrap RobotState without copying)
            current=CurrentStateView(current_state, anchor_idx),
            noisy=CurrentStateView(noisy, anchor_idx),
            # Historical views (wrap StateHistoryBuffer without copying)
            historical=HistoricalView(self.state_history, use_noisy=False)
            if self.state_history
            else None,
            noisy_historical=HistoricalView(self.state_history, use_noisy=True)
            if self.state_history
            else None,
            # Actions (historical)
            current_processed_action=self._current_processed_action,
            previous_action=self.state_history.actions[:, 1]
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            previous_processed_action=self.state_history.processed_actions[:, 1]
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            # Environment state
            ground_heights=ground_heights,
            noisy_ground_heights=noisy.ground_heights,
            respawn_root_offset=self.respawn_root_offset,
            terrain=TerrainContext(
                self.terrain.height_points,
                self.terrain.height_samples,
            ),
            scene=scene_surface_context,
            isaaclab_contact=isaaclab_contact_context,
            body_contacts=body_contacts,
            current_contact_force_magnitudes=current_contact_force_magnitudes,
            prev_contact_force_magnitudes=self.prev_contact_force_magnitudes,
            previous_contact_forces=self.previous_contact_forces,
            contact_active_state=self.contact_active_state,
            contact_age_steps=self.contact_age_steps,
            contact_air_age_steps=self.contact_air_age_steps,
            contact_temporal_valid=self.contact_temporal_valid,
            dt=self.dt,
            progress_buf=self.progress_buf,
            # Contact tracking
            contact_body_ids=self.contact_body_ids,
            contact_observation_body_ids=self.contact_observation_body_ids,
            contact_reward_body_ids=self.contact_reward_body_ids,
            non_termination_contact_body_ids=self.non_termination_contact_body_ids,
            # Per-episode odometer corruption parameters
            odom_scale=self.odom_scale,
            odom_yaw_cos_sin=self.odom_yaw_cos_sin,
        )

        # Controllers populate their task-specific views
        self.control_manager.populate_context(ctx)

        return ctx

    def _build_scene_surface_context(self) -> SceneSurfaceContext:
        """Build scene-object surface tensors for component observations.

        Nearest-surface observations bind these fields unconditionally. Envs
        without object pointclouds receive empty tensors, which lets the compute
        kernel naturally fall back to terrain-only behavior.
        """
        has_object_pointclouds = (
            getattr(self.scene_lib, "_object_pointclouds", None) is not None
        )
        if self.scene_lib.num_objects_per_scene <= 0 or not has_object_pointclouds:
            object_pos = torch.zeros(self.num_envs, 0, 3, device=self.device)
            object_rot = torch.zeros(self.num_envs, 0, 4, device=self.device)
            neutral_pointclouds = torch.zeros(
                self.num_envs, 0, 0, 3, device=self.device
            )
            object_valid_mask = torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )
            return SceneSurfaceContext(
                object_pos=object_pos,
                object_rot=object_rot,
                neutral_pointclouds=neutral_pointclouds,
                object_valid_mask=object_valid_mask,
            )

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        object_state = self.simulator.get_object_root_state()
        return SceneSurfaceContext(
            object_pos=object_state.root_pos,
            object_rot=object_state.root_rot,
            neutral_pointclouds=self.scene_lib.get_scene_neutral_pointcloud(env_ids),
            object_valid_mask=self.scene_lib.get_per_object_valid_mask(env_ids),
        )

    def get_has_reset_grace(self):
        """Check if environments are in the grace period after reset.

        Grace period is useful for zeroing rewards that are unreliable immediately
        after reset (e.g., power consumption, contact changes).

        Returns:
            Boolean tensor indicating which environments are within reset_grace_period steps of last reset.
            Returns None if reset_grace_period is 0 or negative.
        """
        if self.config.reset_grace_period <= 0:
            return None
        return self.progress_buf <= self.config.reset_grace_period

    def compute_reward(self, context: EnvContext):
        """Compute base rewards using the dynamic reward component system.

        Args:
            context: Pre-built EnvContext from self.context property.

        Subclasses should override this to add task-specific rewards, calling super().compute_reward() first.
        """
        grace_mask = self.get_has_reset_grace()

        # Process rewards
        combined_reward, reward_logging = self._process_rewards(context, grace_mask)

        self.rew_buf[:] = combined_reward
        self.extras.update(reward_logging)
        self.extras["total_env_reward"] = combined_reward

    ###############################################################
    # Handle Resets
    ###############################################################
    def move_reset_robot_obj_states_to_respawn_position(
        self,
        env_ids,
        new_states: ResetState,
        new_object_states: ObjectState,
    ) -> Tuple[ResetState, ObjectState]:
        new_states.root_pos += self.respawn_root_offset[env_ids]
        if self.scene_lib.num_scenes() > 0:
            new_object_states.root_pos += self.respawn_root_offset[env_ids].unsqueeze(1)

        return new_states, new_object_states

    def compute_default_reset_state(
        self, env_ids, sample_flat: bool = False
    ) -> Tuple[ResetState, ObjectState]:
        """Reset environments to default state."""

        new_states = self.default_reset_state[env_ids].clone()
        new_object_states = self.default_object_state[env_ids].clone()

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=None,
            sample_flat=sample_flat,
        )

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids, new_states, new_object_states
        )

    def compute_ref_reset_state(
        self,
        env_ids,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        sample_flat: bool = False,
    ) -> Tuple[ResetState, ObjectState]:
        """Compute reset state from reference motion data.

        Args:
            env_ids: Environment indices to reset
            motion_ids: Motion IDs to use [len(env_ids)]
            motion_times: Start times for each motion [len(env_ids)]
            sample_flat: If True, spawn on flat terrain

        Returns:
            Tuple of (reset_state, object_reset_state)
        """

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)
        new_states = ResetState.from_robot_state(ref_state)

        new_object_states = self.scene_lib.get_scene_pose(
            env_ids, motion_times, respawn_offset=self.config.ref_object_respawn_offset
        )
        new_object_states.root_vel = torch.zeros_like(new_object_states.root_pos)
        new_object_states.root_ang_vel = torch.zeros_like(new_object_states.root_pos)

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=ref_state,
            sample_flat=sample_flat,
        )

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids, new_states, new_object_states
        )

    def reset(
        self,
        env_ids=None,
        sample_flat=False,
        force_default_mask=None,
        disable_motion_resample=False,
    ):
        """Reset environments and return observations.

        - auto if no motion_lib: reset from default state
        - auto if motion_lib exists: reset from reference motion
        - force_default_mask: optional boolean mask [len(env_ids)] to force specific envs
            ref_prob = 0.5
            mask = torch.bernoulli(torch.full((len(env_ids),), 1-ref_prob)).bool()
            env.reset(env_ids, force_default_mask=mask)

        Args:
            env_ids: Environment IDs to reset, or None to reset all
            sample_flat: If True, spawn on flat terrain (useful for evaluation)
            force_default_mask: Optional boolean mask [len(env_ids)] to force specific envs
                               to use default reset instead of reference motion reset.
                               Only used if motion_lib exists.
            disable_motion_resample: If True, skip resampling motions (use existing motion_ids/times).
                               Useful for evaluation when you want to replay specific motions.

        Returns:
            obs: Dictionary of observation tensors
            info: Dictionary containing reset metadata (currently empty)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if len(env_ids) == 0:
            return self.get_obs(), {}

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        # Start with default reset for all envs
        new_states, new_object_states = self.compute_default_reset_state(
            env_ids, sample_flat
        )

        # STEP 1: Reset motion manager and determine which envs need reference motion reset
        # This calls motion_manager.sample_motions() internally
        ref_env_ids, motion_ids, motion_times = self._get_ref_reset_envs(
            env_ids, force_default_mask, disable_motion_resample
        )

        # Overwrite ref envs with reference motion reset
        if len(ref_env_ids) > 0:
            ref_states, ref_object_states = self.compute_ref_reset_state(
                ref_env_ids, motion_ids, motion_times, sample_flat
            )

            ref_indices = torch.isin(env_ids, ref_env_ids).nonzero(as_tuple=True)[0]

            new_states[ref_indices] = ref_states
            new_object_states[ref_indices] = ref_object_states

        if self.robot_config.reset_noise is not None:
            apply_reset_noise(
                reset_state=new_states,
                config=self.robot_config.reset_noise,
                dof_limits_lower=self.robot_config.kinematic_info.dof_limits_lower,
                dof_limits_upper=self.robot_config.kinematic_info.dof_limits_upper,
            )

        self.simulator.reset_envs(new_states, new_object_states, env_ids)
        current_state = self.simulator.get_robot_state()
        isaaclab_contact_state = self._get_isaaclab_contact_sensor_state()
        self._reset_contact_state(env_ids)
        self._reset_isaaclab_contact_state(env_ids)
        current_state = self._clear_reset_contact_sample(current_state, env_ids)

        default_mask = ~torch.isin(env_ids, ref_env_ids)
        if self.state_history is not None:
            self._reset_state_history(
                env_ids,
                default_mask,
                ref_env_ids,
                motion_ids,
                motion_times,
                current_state=current_state,
            )

        # Reset control components after motion_manager has been reset
        self.control_manager.reset(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.terminate_buf[env_ids] = False
        self.prev_contact_force_magnitudes[env_ids] = 0.0
        self._current_raw_action[env_ids] = 0.0
        self._current_processed_action[env_ids] = 0.0

        # Resample per-episode odometer corruption parameters.
        # These remain constant within an episode and are used by
        # corrupted_xy_offset_factory when present in observation components.
        n = len(env_ids)
        self.odom_scale[env_ids] = torch.empty(n, device=self.device).uniform_(
            self.config.odom_scale_range[0], self.config.odom_scale_range[1]
        )
        yaw_bias = torch.empty(n, device=self.device).uniform_(
            -self.config.odom_yaw_range_deg, self.config.odom_yaw_range_deg
        ) * (3.14159265358979 / 180.0)
        self.odom_yaw_cos_sin[env_ids, 0] = torch.cos(yaw_bias)
        self.odom_yaw_cos_sin[env_ids, 1] = torch.sin(yaw_bias)

        # Update cached noisy obs for the reset envs with fresh noise
        if self._current_noisy_obs is not None:
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[env_ids, 0]
            ).squeeze(-1)
            obs_noise_cfg = self.simulator.config.domain_randomization.observation_noise
            noisy_subset = apply_observation_noise(
                obs_noise_cfg=obs_noise_cfg,
                robot_state=current_state,
                env_ids=env_ids,
                anchor_idx=self.robot_config.anchor_body_index,
                ground_heights=ground_heights,
            )
            self._current_noisy_obs.update_subset(env_ids, noisy_subset)

        # Recompute observations after reset to reflect new control component state
        # Invalidate and rebuild context since state changed
        self._current_context = None
        self._current_context = self._build_global_context(
            current_state, isaaclab_contact_state
        )
        self.compute_observations(env_ids, context=self._current_context)

        return self.get_obs(), {}

    def _get_ref_reset_envs(
        self, env_ids, force_default_mask, disable_motion_resample=False
    ):
        """Determine which envs should use reference motion reset and reset motion manager.

        This method is responsible for resetting the motion_manager by calling
        motion_manager.sample_motions(). Control components should be reset AFTER
        this method is called so they have access to fresh motion_ids and motion_times.

        Args:
            env_ids: Environment IDs to check
            force_default_mask: Boolean mask to force default reset
            disable_motion_resample: If True, use existing motion_ids/times instead of resampling

        Returns:
            ref_env_ids: Environments to reset with reference motion
            motion_ids: Motion IDs for ref resets (or None)
            motion_times: Motion times for ref resets (or None)
        """
        # No motions - no ref resets
        if self.motion_lib.num_motions() == 0:
            empty_ids = torch.tensor([], device=self.device, dtype=torch.long)
            return empty_ids, None, None

        if force_default_mask is not None:
            assert (
                len(force_default_mask) == len(env_ids)
            ), f"force_default_mask length {len(force_default_mask)} != env_ids length {len(env_ids)}"
            ref_env_ids = env_ids[~force_default_mask]
        else:
            ref_env_ids = env_ids

        if len(ref_env_ids) > 0:
            if not disable_motion_resample:
                self.motion_manager.sample_motions(ref_env_ids)
            motion_ids = self.motion_manager.motion_ids[ref_env_ids]
            motion_times = self.motion_manager.motion_times[ref_env_ids]
        else:
            motion_ids = None
            motion_times = None

        return ref_env_ids, motion_ids, motion_times

    def _reset_state_history(
        self,
        env_ids: Tensor,
        default_mask: Tensor,
        ref_env_ids: Tensor,
        motion_ids: Optional[Tensor],
        motion_times: Optional[Tensor],
        current_state: Optional[RobotState] = None,
    ):
        """Reset state history buffer for specified environments.

        For default reset: repeat current state across all history slots.
        For ref reset: query motion_lib at t-dt, t-2*dt, ... to get historical states.

        Args:
            env_ids: All environment indices being reset.
            default_mask: Boolean mask indicating which envs use default reset.
            ref_env_ids: Environment indices using reference motion reset.
            motion_ids: Motion IDs for ref envs (or None).
            motion_times: Motion times for ref envs (or None).
            current_state: Optional already-fetched post-reset simulator state.
        """
        default_env_ids = env_ids[default_mask]
        num_history_steps = self.state_history.num_history_steps
        # Buffer stores current + history, so total slots = num_history_steps + 1
        buffer_size = num_history_steps + 1

        # Default reset: repeat current simulator state to all buffer slots
        if len(default_env_ids) > 0:
            if current_state is None:
                current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[default_env_ids, 0]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[default_env_ids][
                :, self.contact_body_ids
            ].bool()
            self.state_history.reset_from_single_state(
                env_ids=default_env_ids,
                rigid_body_pos=current_state.rigid_body_pos[default_env_ids],
                rigid_body_rot=current_state.rigid_body_rot[default_env_ids],
                rigid_body_vel=current_state.rigid_body_vel[default_env_ids],
                rigid_body_ang_vel=current_state.rigid_body_ang_vel[default_env_ids],
                dof_pos=current_state.dof_pos[default_env_ids],
                dof_vel=current_state.dof_vel[default_env_ids],
                ground_heights=ground_heights,
                body_contacts=body_contacts,
            )

        # Reference reset: fill buffer with current state at index 0 and historical states at index 1+
        # This ensures historical_* properties (which return [:, 1:]) give exactly num_history_steps elements
        if len(ref_env_ids) > 0 and motion_ids is not None and motion_times is not None:
            # motion_ids shape: [len(ref_env_ids)]
            # motion_times shape: [len(ref_env_ids)]
            num_ref_envs = len(ref_env_ids)

            # Create time offsets: [0, -dt, -2*dt, ..., -N*dt] for buffer_size slots
            # Index 0 = current (t), Index 1..N = historical (t-dt, t-2*dt, ..., t-N*dt)
            time_offsets = -self.dt * torch.arange(buffer_size, device=self.device)

            # Expand for batch query: [num_ref_envs, buffer_size]
            expanded_motion_ids = motion_ids.unsqueeze(1).expand(-1, buffer_size)
            expanded_motion_times = motion_times.unsqueeze(1) + time_offsets.unsqueeze(
                0
            )

            # Clamp times to valid range
            motion_lengths = self.motion_lib.motion_lengths[motion_ids]
            expanded_motion_times = expanded_motion_times.clamp(min=0.0)
            expanded_motion_times = torch.min(
                expanded_motion_times,
                motion_lengths.unsqueeze(1).expand(-1, buffer_size),
            )

            # Flatten for motion_lib query
            flat_motion_ids = expanded_motion_ids.reshape(-1)
            flat_motion_times = expanded_motion_times.reshape(-1)

            # Query motion library
            historical_state = self.motion_lib.get_motion_state(
                flat_motion_ids, flat_motion_times
            )

            # Motion library data is recorded on flat terrain (height = 0)
            # Only simulator-based states need terrain height queries
            historical_ground_heights = torch.zeros(
                num_ref_envs, buffer_size, device=self.device
            )

            # Get contacts from motion library if available, otherwise zeros
            if historical_state.rigid_body_contacts is not None:
                flat_contacts = historical_state.rigid_body_contacts[
                    :, self.contact_body_ids
                ].bool()
                historical_body_contacts = flat_contacts.view(
                    num_ref_envs, buffer_size, -1
                )
            else:
                historical_body_contacts = torch.zeros(
                    num_ref_envs,
                    buffer_size,
                    len(self.contact_body_ids),
                    dtype=torch.bool,
                    device=self.device,
                )

            # Reshape back to [num_ref_envs, buffer_size, ...]
            self.state_history.reset_from_states(
                env_ids=ref_env_ids,
                rigid_body_pos=historical_state.rigid_body_pos.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_rot=historical_state.rigid_body_rot.view(
                    num_ref_envs, buffer_size, -1, 4
                ),
                rigid_body_vel=historical_state.rigid_body_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_ang_vel=historical_state.rigid_body_ang_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                dof_pos=historical_state.dof_pos.view(num_ref_envs, buffer_size, -1),
                dof_vel=historical_state.dof_vel.view(num_ref_envs, buffer_size, -1),
                ground_heights=historical_ground_heights,
                body_contacts=historical_body_contacts,
                actions=None,  # Zero actions for historical reset
            )

    ###############################################################
    # Motion and Visualization Helpers
    ###############################################################
    def create_motion_manager(self):
        """Instantiate motion manager from configuration."""
        MotionManagerClass = get_class(self.config.motion_manager._target_)

        fixed_motion_ids = None
        if self.scene_lib.num_scenes() > 0:
            humanoid_motion_ids = self.scene_lib.get_humanoid_motion_ids()
            if humanoid_motion_ids is not None:
                fixed_motion_ids = torch.tensor(
                    humanoid_motion_ids, dtype=torch.long, device=self.device
                )

        self.motion_manager = MotionManagerClass(
            config=self.config.motion_manager,
            num_envs=self.num_envs,
            env_dt=self.dt,
            device=self.device,
            motion_lib=self.motion_lib,
            fixed_motion_ids_per_env=fixed_motion_ids,
        )

    def create_visualization_markers(self, headless: bool):
        """Create visualization markers based on headless flag.

        Args:
            headless: If True, no markers are created (empty dict).
                      If False, creates markers according to config.

        Returns:
            Dict of visualization markers.
        """
        if headless:
            return {}

        visualization_markers = {}

        if self.config.show_terrain_markers:
            terrain_markers = []
            for _ in range(self.terrain.num_height_points):
                terrain_markers.append(MarkerConfig(size="small"))
            terrain_markers_cfg = VisualizationMarkerConfig(
                type="sphere", color=(0.008, 0.345, 0.224), markers=terrain_markers
            )
            visualization_markers["terrain_markers"] = terrain_markers_cfg

        # Merge markers from control components
        control_markers = self.control_manager.create_visualization_markers(headless)
        visualization_markers.update(control_markers)

        return visualization_markers

    def get_state_dict(self):
        """Get environment state for checkpointing.

        Returns:
            Dictionary containing motion manager state
        """
        if self.motion_manager is not None:
            return {"motion_manager": self.motion_manager.get_state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        """Load environment state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        if self.motion_manager is not None:
            self.motion_manager.load_state_dict(state_dict["motion_manager"])

    def get_task_id(self):
        """Get task identifier for logging and checkpointing.

        Returns:
            String identifier (motion file name or 'null')
        """
        if self.motion_manager is not None:
            return self.motion_lib.motion_file.split("/")[-1]
        return "null"

    @staticmethod
    def apply_motion_weights_to_scene_weights(
        save_dir: Optional[str], motion_file: Optional[str], device: torch.device
    ) -> Optional[list]:
        """Apply motion weights from checkpoint as scene weights for curriculum learning.

        Loads motion weights from a previous training checkpoint and uses them as
        scene replication weights, allowing over-sampling of scenes corresponding to
        failed motions in curriculum learning.

        IMPORTANT: Assumes 1:1 correspondence between scenes and motions,
        where scene[i].humanoid_motion_id == i.

        Args:
            save_dir: Directory where checkpoints are saved (or None)
            motion_file: Motion file path to identify checkpoint (or None)
            device: PyTorch device

        Returns:
            List of scene weights from motion training or None if not available
        """
        from pathlib import Path

        if not save_dir or not motion_file:
            return None

        try:
            evaluated_motions = motion_file.split("/")[-1]
            checkpoint_path = Path(save_dir) / f"env_{evaluated_motions}.ckpt"

            if not checkpoint_path.exists():
                return None

            print(f"Loading motion weights from checkpoint: {checkpoint_path}")
            checkpoint_data = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            if "motion_manager" not in checkpoint_data:
                print(
                    "No motion_manager found in checkpoint, using uniform scene weights."
                )
                return None

            motion_weights = checkpoint_data["motion_manager"]["motion_weights"]
            print(f"Applying {len(motion_weights)} motion weights as scene weights")
            print(
                "WARNING: Assumes 1:1 scene-to-motion correspondence (scene[i].humanoid_motion_id == i)"
            )
            return motion_weights.cpu().tolist()

        except Exception as e:
            print(f"Error applying motion weights to scene weights: {e}")
            return None

    def save_state(self) -> dict:
        """Save all mutable env state for later restoration.

        Snapshots the current state of the environment including robot state,
        simulator state, progress/reset/terminate buffers, and state history.
        This is useful for temporarily interrupting normal training to run
        evaluation episodes, then restoring to continue training from where
        it left off.

        Returns:
            Dictionary containing cloned copies of all mutable state tensors
        """
        snapshot = {
            "robot_state": self.simulator.get_robot_state(),
            "markers_state": self.get_markers_state(),
            "actions": self.simulator.get_current_actions(),
            "progress_buf": self.progress_buf.clone(),
            "reset_buf": self.reset_buf.clone(),
            "terminate_buf": self.terminate_buf.clone(),
            "respawn_root_offset": self.respawn_root_offset.clone(),
            "odom_scale": self.odom_scale.clone(),
            "odom_yaw_cos_sin": self.odom_yaw_cos_sin.clone(),
            "prev_contact_force_magnitudes": (
                self.prev_contact_force_magnitudes.clone()
            ),
            "previous_contact_forces": self.previous_contact_forces.clone(),
            "contact_active_state": self.contact_active_state.clone(),
            "contact_age_steps": self.contact_age_steps.clone(),
            "contact_air_age_steps": self.contact_air_age_steps.clone(),
            "contact_temporal_valid": self.contact_temporal_valid.clone(),
            "physics_step_count": self._physics_step_count,
        }
        if self.state_history is not None:
            snapshot["state_history"] = self.state_history.save_state()
        if getattr(self, "isaaclab_previous_normal_force_w", None) is not None:
            snapshot.update(
                {
                    "isaaclab_previous_normal_force_w": (
                        self.isaaclab_previous_normal_force_w.clone()
                    ),
                    "isaaclab_previous_active": (
                        self.isaaclab_previous_active.clone()
                    ),
                    "isaaclab_contact_age_s": self.isaaclab_contact_age_s.clone(),
                    "isaaclab_air_age_s": self.isaaclab_air_age_s.clone(),
                    "isaaclab_contact_temporal_valid": (
                        self.isaaclab_contact_temporal_valid.clone()
                    ),
                }
            )
        if self._current_noisy_obs is not None:
            from dataclasses import fields as dc_fields

            noisy = self._current_noisy_obs
            snapshot["_current_noisy_obs"] = NoisyObservations(
                **{f.name: getattr(noisy, f.name).clone() for f in dc_fields(noisy)}
            )
        if self.scene_lib.num_objects_per_scene > 0:
            snapshot["object_state"] = self.simulator.get_object_root_state()
        return snapshot

    def restore_state(self, snapshot: dict) -> None:
        """Restore env state from a previous save_state() snapshot.

        Restores all mutable state that was captured by save_state(),
        including robot positions/velocities, buffers, and state history.

        Args:
            snapshot: Dictionary from save_state() containing state tensors
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.reset_envs(
            snapshot["robot_state"], snapshot.get("object_state"), env_ids
        )

        if "state_history" in snapshot and self.state_history is not None:
            self.state_history.load_state(snapshot["state_history"])

        self.progress_buf.copy_(snapshot["progress_buf"])
        self.reset_buf.copy_(snapshot["reset_buf"])
        self.terminate_buf.copy_(snapshot["terminate_buf"])
        self.respawn_root_offset.copy_(snapshot["respawn_root_offset"])
        if "odom_scale" in snapshot:
            self.odom_scale.copy_(snapshot["odom_scale"])
            self.odom_yaw_cos_sin.copy_(snapshot["odom_yaw_cos_sin"])
        if "previous_contact_forces" in snapshot:
            self.prev_contact_force_magnitudes.copy_(
                snapshot["prev_contact_force_magnitudes"]
            )
            self.previous_contact_forces.copy_(
                snapshot["previous_contact_forces"]
            )
            self.contact_active_state.copy_(snapshot["contact_active_state"])
            self.contact_age_steps.copy_(snapshot["contact_age_steps"])
            self.contact_air_age_steps.copy_(snapshot["contact_air_age_steps"])
            self.contact_temporal_valid.copy_(
                snapshot["contact_temporal_valid"]
            )
            self._physics_step_count = snapshot.get(
                "physics_step_count", self._physics_step_count
            )
        else:
            # Backward compatibility for snapshots created before temporal
            # contact state existed. The first post-restore force rate is gated.
            self.prev_contact_force_magnitudes.zero_()
            self._reset_contact_state(env_ids)
            self._update_contact_state(self.simulator.get_robot_state())
        if getattr(self, "isaaclab_previous_normal_force_w", None) is not None:
            if "isaaclab_previous_normal_force_w" in snapshot:
                self.isaaclab_previous_normal_force_w.copy_(
                    snapshot["isaaclab_previous_normal_force_w"]
                )
                self.isaaclab_previous_active.copy_(
                    snapshot["isaaclab_previous_active"]
                )
                self.isaaclab_contact_age_s.copy_(
                    snapshot["isaaclab_contact_age_s"]
                )
                self.isaaclab_air_age_s.copy_(snapshot["isaaclab_air_age_s"])
                self.isaaclab_contact_temporal_valid.copy_(
                    snapshot["isaaclab_contact_temporal_valid"]
                )
            else:
                self._reset_isaaclab_contact_state(env_ids)
        self._current_noisy_obs = snapshot.get("_current_noisy_obs")
        self._current_context = None

        # IsaacGym needs an extra step after state restore to sync internal state
        if "isaacgym" in self.simulator.config._target_.lower():
            self.simulator.step(snapshot["actions"], markers_callback=None)

    def close(self) -> None:
        """Release control-component and env-owned UI handles, then close
        the simulator. Safe to call multiple times."""
        control_manager = getattr(self, "control_manager", None)
        if control_manager is not None:
            for component in control_manager.components.values():
                component.close()

        ui = getattr(self, "_key_bindings", None)
        if ui is not None:
            ui.unregister_all()
            self._key_bindings = None

        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            close = getattr(simulator, "close", None)
            if callable(close):
                close()
