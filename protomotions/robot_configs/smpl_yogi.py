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

from protomotions.robot_configs.base import RobotAssetConfig
from protomotions.robot_configs.smpl import SmplRobotConfig


@dataclass
class SmplYogiRobotConfig(SmplRobotConfig):
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
