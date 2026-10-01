# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MOYO yoga performer's own body as the training plant: plant v2 (BodyFix Steps 1-2).

Identical to :class:`SmplYogiRobotConfig` (joint names and axes, trackable and contact bodies, simulation
parameters, effort limits taken from the MJCF) except:

* **the asset** ``data/assets/smpl/smpl_yogi03596_v2.xml`` and its USD package, built by
  ``data/scripts/build_subject_plant_v2.py`` and ``build_plant_v2_usd.py``: her SMPL-X skeleton and masses
  (74 kg), the shipped collider primitives placed on her segments, a widened joint box (23 ranges) and wrist /
  hand torque limits 30 / 15 N m (``smpl_yogi03596_v2_limits.json``);
* **``default_root_height`` 0.975 m**: the rest pose's pelvis stands 0.9745 m above its lowest collider (feet
  flat), so a default reset starts 0.5 mm off the floor (the shipped robot's 0.95 m sat 4.4 cm above legs that
  reached 0.906 m);
* **the neck's PD gains** scaled by its joint-space inertia at rest, x2.32 (the head sphere now sits on the
  cranium, 19 cm above the neck joint), so the servo keeps the natural frequency (91 rad/s) and damping ratio it
  had on the shipped plant (``data/reference_curation/plant_v2/plant_v2.json`` -> ``pd_gains``); every other
  group moved less than 1.5x and keeps its gains.

Checkpoints trained on ``smpl_yogi`` are off-distribution here, and motions and physics tables must be built for
this plant: ``protomotions.utils.plant_identity`` refuses the others. Use with ``--robot-name smpl_yogi_v2``.
"""

from dataclasses import dataclass, field

from protomotions.robot_configs.base import (
    ControlConfig,
    ControlInfo,
    ControlType,
    RobotAssetConfig,
)
from protomotions.robot_configs.smpl_yogi import SmplYogiRobotConfig

NECK_INERTIA_RATIO = 2.3154          # plant_v2.json pd_gains.joint_space_inertia_rest.Neck.ratio


@dataclass
class SmplYogiV2RobotConfig(SmplYogiRobotConfig):
    default_root_height: float = 0.975

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                ".*_(Hip|Knee|Ankle)_.*": ControlInfo(
                    stiffness=800, damping=80, velocity_limit=100
                ),
                ".*_Toe_.*": ControlInfo(
                    stiffness=500, damping=50, velocity_limit=100
                ),
                "(Torso|Spine|Chest)_.*": ControlInfo(
                    stiffness=1000, damping=100, velocity_limit=100
                ),
                "Neck_.*": ControlInfo(
                    stiffness=round(500 * NECK_INERTIA_RATIO),
                    damping=round(50 * NECK_INERTIA_RATIO),
                    velocity_limit=100,
                ),
                "(Head|.*_Thorax|.*_Shoulder|.*_Elbow)_.*": ControlInfo(
                    stiffness=500, damping=50, velocity_limit=100
                ),
                ".*_(Wrist|Hand)_.*": ControlInfo(
                    stiffness=300, damping=30, velocity_limit=100
                ),
            },
        )
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            # kinematic_info (dof names + joint limits) is parsed from this MJCF at __post_init__ and matches the
            # USD spawned from smpl_yogi03596_v2_usd/ (build_plant_v2_usd.py checks the limits layer by layer).
            asset_root="data/assets",
            asset_file_name="smpl/smpl_yogi03596_v2.xml",
            usd_asset_file_name="smpl/smpl_yogi03596_v2_usd/smpl_yogi03596_v2_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Pelvis/",
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )
