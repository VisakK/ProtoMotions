# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMPL humanoid rescaled to the MOYO yoga subject ``yogi_03596`` (female).

Identical in every way to :class:`SmplRobotConfig` (the ``smpl_boxhands_lowtorque``
asset) **except the asset geometry**: bone lengths, collision geoms and masses were
rescaled to the subject's measured body shape by
``data/scripts/build_subject_skeleton.py`` --

    upper arm x1.194   forearm x1.013   thigh x0.977   shin x0.977
    total mass 54.5 kg -> 74 kg (subject ground-reaction weight)

Joint names, ranges, actuator force ranges and PD/armature values are unchanged,
so the parsed ``kinematic_info`` (dof names + joint limits) is byte-for-byte the
same as the boxhands asset and every downstream config/control path is compatible.

Use with ``--robot-name smpl_yogi``.  See ``notes/Bring_up.md``.
"""

from dataclasses import dataclass, field

from protomotions.robot_configs.base import (
    ControlConfig,
    ControlInfo,
    ControlType,
    RobotAssetConfig,
)
from protomotions.robot_configs.smpl import SmplRobotConfig


@dataclass
class SmplYogiRobotConfig(SmplRobotConfig):
    # PD gains and velocity limits are inherited from SmplRobotConfig unchanged.
    # The one deliberate difference is that ``effort_limit`` is NOT overridden, so
    # ``extract_kinematic_info`` keeps each joint's ``actuatorfrcrange`` from the
    # MJCF instead of the stock uniform 500 N*m.
    #
    # 500 N*m on every joint is not a torque limit, it is the absence of one: it
    # is 25x the MJCF's wrist value, 50x the hand's, and enough for a wrist to
    # hold this 74 kg body on a 0.68 m lever. Nothing binds, so the plant cannot
    # produce human-like effort distribution -- the policy is free to solve poses
    # with torques no human could generate. The MJCF's per-joint ranges
    # (hip 300, knee 200, ankle 100, shoulder 150, elbow 100, wrist 20, hand 10,
    # toe 20 N*m) are physiologically plausible for a 74 kg adult and are what
    # ``build_subject_skeleton.py`` preserved from the source asset.
    #
    # This changes the plant: policies trained before this are off-distribution
    # and must be retrained, not resumed. ``smpl`` (boxhands) is deliberately
    # left on the uniform 500 so existing stock-SMPL experiments are untouched;
    # its MJCF carries usable ranges too if the same fix is wanted there.
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
                "(Neck|Head|.*_Thorax|.*_Shoulder|.*_Elbow)_.*": ControlInfo(
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
            # Subject-specific SMPL asset (repo-root data/assets). kinematic_info
            # (dof names + joint limits) is parsed from this MJCF at __post_init__
            # and matches the USD spawned from smpl_yogi03596_lowtorque_usd/.
            asset_root="data/assets",
            asset_file_name="smpl/smpl_yogi03596_lowtorque.xml",
            usd_asset_file_name="smpl/smpl_yogi03596_lowtorque_usd/smpl_yogi03596_lowtorque_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Pelvis/",
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )
