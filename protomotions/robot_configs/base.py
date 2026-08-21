# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base robot configuration classes.

This module defines the configuration dataclasses for robot morphology, control
parameters, and simulator-specific settings. All robot configurations (SMPL, G1, etc.)
inherit from RobotConfig and specify their robot-specific parameters.

Key Classes:
    - RobotConfig: Main robot configuration
    - RobotAssetConfig: Robot asset and physics properties
    - RobotControlConfig: PD control parameters
    - ControlType: Enum for control modes

Key Features:
    - Multi-simulator support
    - Configurable PD gains and action scaling
    - Asset file management
    - Body name mappings for observations
    - Kinematic structure extraction
"""

from dataclasses import dataclass, field
from protomotions.components.pose_lib import ControlInfo, KinematicInfo
from protomotions.simulator.base_simulator.config import RobotNoiseConfig
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.mujoco.config import MujocoSimParams

from typing import List, Optional, Dict, Union
from enum import Enum
import os
import re
import torch


@dataclass
class SimulatorParams:
    isaacgym: IsaacGymSimParams = field(default_factory=IsaacGymSimParams)
    isaaclab: IsaacLabSimParams = field(default_factory=IsaacLabSimParams)
    genesis: GenesisSimParams = field(default_factory=GenesisSimParams)
    newton: NewtonSimParams = field(default_factory=NewtonSimParams)
    mujoco: MujocoSimParams = field(default_factory=MujocoSimParams)


@dataclass
class InitState:
    """Configuration for robot initial state."""

    pos: Optional[List[float]]  # [x, y, z] in meters


class ControlType(Enum):
    """Enum defining the available control types for the robot.

    BUILT_IN_PD: Built-in PD controller (e.g. Isaac Gym's internal PD controller)
    VELOCITY: Velocity control using custom PD controller
    TORQUE: Direct torque control
    PROPORTIONAL: Proportional control using custom PD controller
    """

    BUILT_IN_PD = "built_in_pd"
    TORQUE = "torque"
    PROPORTIONAL = "proportional"

    @classmethod
    def from_str(cls, value: str) -> "ControlType":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )


@dataclass
class RobotAssetConfig:
    """Configuration for robot asset properties."""

    # Optional fields with defaults
    asset_root: str = "protomotions/data/assets"
    self_collisions: bool = True

    # Optional fields
    asset_file_name: str = None
    usd_asset_file_name: str = None
    usd_bodies_root_prim_path: str = None
    replace_cylinder_with_capsule: Optional[bool] = None
    thickness: Optional[float] = None
    max_angular_velocity: Optional[float] = None
    max_linear_velocity: Optional[float] = None
    density: Optional[float] = None
    angular_damping: Optional[float] = None
    linear_damping: Optional[float] = None
    disable_gravity: Optional[bool] = None
    fix_base_link: Optional[bool] = None

    def __post_init__(self):
        """Validate that asset_file_name is set."""

        if not self.asset_file_name or not self.asset_file_name.endswith(".xml"):
            raise ValueError(
                f"RobotAssetConfig.asset_file_name ('{self.asset_file_name}') "
                f"must be a valid path to an .xml MJCF file to extract kinematic info. "
                "if you are using URDF, convert it to MJCF first"
            )


@dataclass
class ControlConfig:
    """Configuration for robot control parameters."""

    # Control info overrides for specific joints instead of the values from the MJCF asset
    override_control_info: Optional[Dict[str, ControlInfo]] = None

    # Can be "built_in_pd" or "proportional"/"velocity"/"torque" for Proportional, Velocity, Torque control
    control_type: ControlType = ControlType.BUILT_IN_PD

    # the positional limits used for rewards
    soft_pos_limit: float = 0.9

    # The following field is loaded post-init and populated from the MJCF asset
    # Note: Using Field(init=False) to exclude from __init__ signature
    control_info: Dict[str, ControlInfo] = field(init=False)

    def __post_init__(self):
        """Validate that override_control_info is a dictionary."""
        if self.override_control_info is not None:
            override_control_info = {}
            for key, value in self.override_control_info.items():
                if isinstance(value, dict):
                    override_control_info[key] = ControlInfo.from_dict(value)
                else:
                    override_control_info[key] = value
            self.override_control_info = override_control_info

    def initialize_control_info(self, asset: RobotAssetConfig):
        """Initialize control info from asset configuration."""
        if not hasattr(self, "control_info"):
            from protomotions.components.pose_lib import extract_control_info

            self.control_info = extract_control_info(
                mjcf_path=os.path.join(asset.asset_root, asset.asset_file_name),
                override_control_info=self.override_control_info,
            )


@dataclass
class RobotConfig:
    """Configuration for robot morphology and control parameters.

    Defines all robot-specific parameters including asset files, body names,
    control settings, and physical properties. Each robot (SMPL, G1, etc.)
    should subclass this and provide robot-specific values.

    Key configuration areas:

    - **Asset**: Robot mesh/URDF files and physical properties
    - **Body Names**: Named body parts for tracking and observations
    - **Control**: PD gains, action scales, and control modes
    - **Kinematic Info**: Joint structure extracted from MJCF/URDF
    - **Simulator Params**: Simulator-specific overrides

    Example::

        config = RobotConfig(
            asset=RobotAssetConfig(asset_file_name="robot.xml"),
            control=ControlConfig(control_type=ControlType.BUILT_IN_PD)
        )
    """

    # Required nested config (subclasses must override with their own default_factory)
    asset: RobotAssetConfig

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {}
    )

    default_root_height: float = 1
    default_dof_pos: Optional[Union[List[float], torch.Tensor, Dict[str, float]]] = (
        None  # Default joint positions for resets (if None, uses zeros).
        # Can be a Dict[str, float] mapping regex patterns to values (e.g.,
        # {".*_knee_joint": 0.669}). Patterns are resolved against dof_names
        # in __post_init__. Unmatched DOFs default to 0.0.
    )

    contact_bodies: Optional[Union[List[str], str]] = (
        None  # "all" means all bodies. Contact sensors are expensive to simulate, so we only spawn them when required.
    )
    contact_observation_bodies: Optional[Union[List[str], str]] = None
    """Bodies exposed by contact observations.

    ``None`` falls back to :attr:`contact_bodies`, preserving the legacy
    behavior in configurations which only define the simulator sensor set.
    """
    contact_reward_bodies: Optional[Union[List[str], str]] = None
    """Bodies used by contact-matching rewards.

    ``None`` falls back to :attr:`contact_bodies`, preserving the legacy
    reward body set.
    """
    contact_pair_bodies: Optional[Union[List[str], str]] = None
    """Bodies each contact sensor additionally filters *itself* against.

    ``None`` (the default) senses ground contact only, which is what every
    simulator backend has always reported: the per-body sensors filter against
    the terrain mesh, so a policy can feel its foot on the floor but not its
    shin on its own upper arm.  Naming bodies here adds one filter column per
    body to every sensor, which is what makes ``rigid_body_pair_contact_forces``
    non-``None``.

    This is off by default because it is not free -- each sensor goes from one
    filter to ``1 + len(contact_pair_bodies)`` and PhysX pays for the extra
    contact reduction every substep.  Must be a subset of
    :attr:`contact_bodies`, since only those bodies have sensors to filter.
    """
    trackable_bodies_subset: Union[List[str], str] = "all"

    non_termination_contact_bodies: Union[List[str], str] = "all"
    init_state: Optional[InitState] = None
    reset_noise: Optional[RobotNoiseConfig] = None
    mimic_small_marker_bodies: Optional[List[str]] = None

    # Anchor body for observations/rewards (None defaults to root body at index 0)
    anchor_body_name: Optional[str] = None

    # Optional with default
    contact_pairs_multiplier: int = 16
    control: ControlConfig = field(default_factory=ControlConfig)

    # The following fields are loaded post-init and populated from the MJCF asset
    # Note: Using Field(init=False) to exclude from __init__ signature
    kinematic_info: KinematicInfo = field(init=False)
    number_of_actions: int = field(init=False)
    anchor_body_index: int = field(init=False)
    _contact_observation_bodies_use_fallback: bool = field(
        init=False, repr=False, default=True
    )
    _contact_reward_bodies_use_fallback: bool = field(
        init=False, repr=False, default=True
    )

    # Dictionary of simulator-specific simulation parameters
    simulation_params: SimulatorParams = field(default_factory=SimulatorParams)

    def __post_init__(self):
        """Compute derived fields after initialization."""

        from protomotions.components.pose_lib import extract_kinematic_info

        self.kinematic_info = extract_kinematic_info(
            os.path.join(self.asset.asset_root, self.asset.asset_file_name)
        )

        # Resolve anchor_body_index from anchor_body_name
        if self.anchor_body_name is None:
            self.anchor_body_index = 0  # Default to root body
        else:
            if self.anchor_body_name not in self.kinematic_info.body_names:
                raise ValueError(
                    f"anchor_body_name '{self.anchor_body_name}' not found in body_names: "
                    f"{self.kinematic_info.body_names}"
                )
            self.anchor_body_index = self.kinematic_info.body_names.index(
                self.anchor_body_name
            )

        # Initialize control info in the control config
        self.control.initialize_control_info(self.asset)

        self.number_of_actions = self.kinematic_info.num_dofs

        # Initialize default_dof_pos: use provided values or zeros
        if self.default_dof_pos is None:
            # Default to zero joint positions
            self.default_dof_pos = torch.zeros(
                self.kinematic_info.num_dofs, dtype=torch.float32
            )
        elif isinstance(self.default_dof_pos, dict):
            # Resolve regex pattern dict to tensor
            dof_pos = torch.zeros(self.kinematic_info.num_dofs, dtype=torch.float32)
            for pattern, value in self.default_dof_pos.items():
                for i, name in enumerate(self.kinematic_info.dof_names):
                    if re.fullmatch(pattern, name):
                        dof_pos[i] = value
            self.default_dof_pos = dof_pos
        else:
            # Convert to tensor if needed and validate
            if not isinstance(self.default_dof_pos, torch.Tensor):
                self.default_dof_pos = torch.tensor(
                    self.default_dof_pos, dtype=torch.float32
                )
            assert (
                len(self.default_dof_pos) == self.kinematic_info.num_dofs
            ), f"default_dof_pos length {len(self.default_dof_pos)} != num_dofs {self.kinematic_info.num_dofs}"

        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self._contact_observation_bodies_use_fallback = (
            self.contact_observation_bodies is None
        )
        self._contact_reward_bodies_use_fallback = self.contact_reward_bodies is None
        self._resolve_contact_body_fields()
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )

        required_abstract_names = [
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "all_left_hand_bodies",
            "all_right_hand_bodies",
            "head_body_name",
            "torso_body_name",
        ]
        for name in required_abstract_names:
            assert (
                name in self.common_naming_to_robot_body_names.keys()
            ), f"RobotConfig.common_naming_to_robot_body_names must contain {name}"

    def update_fields(self, **kwargs):
        """Update robot config fields and reprocess derived fields.

        This is useful when you need to modify fields after initialization
        without manually calling __post_init__.

        Args:
            **kwargs: Fields to update on the robot config
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"RobotConfig has no field '{key}'")
            setattr(self, key, value)

        if "contact_observation_bodies" in kwargs:
            self._contact_observation_bodies_use_fallback = (
                kwargs["contact_observation_bodies"] is None
            )
        if "contact_reward_bodies" in kwargs:
            self._contact_reward_bodies_use_fallback = (
                kwargs["contact_reward_bodies"] is None
            )

        # Reprocess fields that depend on the updated values
        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self._resolve_contact_body_fields()
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )

    def _resolve_contact_body_fields(self):
        """Resolve, order, and validate all contact-related body selections."""

        self.contact_bodies = resolve_body_names_in_kinematic_order(
            self.contact_bodies,
            self,
            field_name="contact_bodies",
        )

        observation_selection = (
            self.contact_bodies
            if self._contact_observation_bodies_use_fallback
            else self.contact_observation_bodies
        )
        reward_selection = (
            self.contact_bodies
            if self._contact_reward_bodies_use_fallback
            else self.contact_reward_bodies
        )
        self.contact_observation_bodies = resolve_body_names_in_kinematic_order(
            observation_selection,
            self,
            field_name="contact_observation_bodies",
        )
        self.contact_reward_bodies = resolve_body_names_in_kinematic_order(
            reward_selection,
            self,
            field_name="contact_reward_bodies",
        )
        # None stays None: an empty list would still be a request for body-body
        # sensing, and it must stay distinguishable from "do not sense pairs".
        if self.contact_pair_bodies is not None:
            self.contact_pair_bodies = resolve_body_names_in_kinematic_order(
                self.contact_pair_bodies,
                self,
                field_name="contact_pair_bodies",
            )

        sensor_bodies = set(self.contact_bodies or [])
        for field_name, selected_bodies in (
            ("contact_observation_bodies", self.contact_observation_bodies),
            ("contact_reward_bodies", self.contact_reward_bodies),
            ("contact_pair_bodies", self.contact_pair_bodies),
        ):
            missing_sensor_bodies = [
                name for name in (selected_bodies or []) if name not in sensor_bodies
            ]
            if missing_sensor_bodies:
                raise ValueError(
                    f"{field_name} must be a subset of contact_bodies so the "
                    "simulator reports contact state for every selected body. "
                    f"Missing from contact_bodies: {missing_sensor_bodies}. "
                    f"Resolved contact_bodies: {self.contact_bodies or []}"
                )


def abstract_names_to_body_names(
    names: Union[List[str], str], robot_config: RobotConfig
):
    if names is None:
        return None

    if names == "all":
        return robot_config.kinematic_info.body_names

    if isinstance(names, list):
        parsed_names = []
        for name in names:
            parsed_names.extend(abstract_names_to_body_names(name, robot_config))
        return parsed_names

    if names == "root":
        return [robot_config.kinematic_info.body_names[0]]
    elif names in robot_config.common_naming_to_robot_body_names.keys():
        return robot_config.common_naming_to_robot_body_names[names]
    else:
        return [names]


def resolve_body_names_in_kinematic_order(
    names: Optional[Union[List[str], str]],
    robot_config: RobotConfig,
    *,
    field_name: str,
) -> Optional[List[str]]:
    """Resolve a body selection and return unique names in kinematic order.

    Abstract aliases are expanded with :func:`abstract_names_to_body_names`.
    Literal names are validated here so configuration errors are reported
    before simulator construction.
    """

    resolved_names = abstract_names_to_body_names(names, robot_config)
    if resolved_names is None:
        return None
    if isinstance(resolved_names, str):
        resolved_names = [resolved_names]

    all_body_names = robot_config.kinematic_info.body_names
    all_body_name_set = set(all_body_names)
    missing_names = list(
        dict.fromkeys(name for name in resolved_names if name not in all_body_name_set)
    )
    if missing_names:
        raise ValueError(
            f"{field_name} contains body names that are not present in the robot "
            f"kinematic model: {missing_names}. Available bodies: {all_body_names}"
        )

    selected_name_set = set(resolved_names)
    return [name for name in all_body_names if name in selected_name_set]
