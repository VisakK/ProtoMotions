# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the mass and actuation PhysX actually simulates, vs. the source MJCF.

The USD is the only source of truth for mass at runtime -- ProtoMotions passes
no ``mass_props`` when it spawns the robot articulation -- and the MJCF importer
has been observed to drop per-geom densities.  This boots the robot in Isaac Lab
and reads the masses and joint effort limits straight out of the PhysX view, so
the numbers are measured rather than inferred.

Both have silently disagreed with the asset before: the MJCF importer dropped
per-geom density (halving the mass), and the stock robot config overrode every
joint's ``actuatorfrcrange`` with a uniform 500 N*m -- 25x the MJCF's wrist value
and 50x its hand value, which makes the torque limit non-binding and human-like
effort distribution unreachable.

Usage::

    python usd_convert/check_simulated_mass.py --robot-name smpl_yogi \
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque_flat.xml
"""

from __future__ import annotations

import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--simulator", default="isaaclab")
    parser.add_argument(
        "--mjcf",
        default=None,
        help="Optional source MJCF to compare the simulated total against.",
    )
    parser.add_argument("--tolerance", type=float, default=0.01, help="Relative.")
    return parser


parser = create_parser()
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    from lightning.fabric import Fabric

    from protomotions.robot_configs.factory import robot_config as make_robot_config
    from protomotions.simulator.factory import simulator_config as make_simulator_config
    from protomotions.utils.fabric_config import FabricConfig

    fabric = Fabric(
        **FabricConfig(
            accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()
    simulation_app = AppLauncher({"headless": True, "device": str(fabric.device)}).app

    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.scene_lib import SceneLibConfig
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )
    from protomotions.utils.component_builder import build_all_components

    robot_config = make_robot_config(args.robot_name)
    simulator_config = make_simulator_config(
        simulator=args.simulator,
        robot_config=robot_config,
        headless=True,
        num_envs=1,
        experiment_name="check_simulated_mass",
    )
    terrain_config, simulator_config = convert_friction_for_simulator(
        TerrainConfig(), simulator_config
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=SceneLibConfig(),
        motion_lib_config=MotionLibConfig(),
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        simulation_app=simulation_app,
    )
    simulator = components["simulator"]
    # Normally Env drives this once it has built its markers; there is no Env here.
    simulator._initialize_with_markers(None)

    masses = simulator._robot.root_physx_view.get_masses()[0]
    body_names = simulator._robot.data.body_names
    total = float(masses.sum())

    efforts = simulator._robot.root_physx_view.get_dof_max_forces()[0]
    dof_names = simulator._robot.data.joint_names

    print()
    print(f"{'body':<14}{'kg':>9}")
    print("-" * 23)
    for name, mass in zip(body_names, masses.tolist()):
        print(f"{name:<14}{mass:9.3f}")
    print("-" * 23)
    print(f"{'SIMULATED':<14}{total:9.3f}")

    print()
    print(f"{'joint':<16}{'N*m':>9}")
    print("-" * 25)
    seen = {}
    for name, eff in zip(dof_names, efforts.tolist()):
        seen.setdefault(name.rsplit("_", 1)[0], eff)
    for name, eff in seen.items():
        print(f"{name:<16}{eff:9.1f}")
    if len(set(efforts.tolist())) == 1:
        print(
            f"\nWARNING: every joint shares one effort limit "
            f"({efforts[0]:.0f} N*m) -- the torque limit is almost certainly not "
            "modelling anything. Check override_control_info in the robot config."
        )

    status = 0
    if args.mjcf:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from apply_mjcf_masses_to_usd import read_mjcf_bodies

        expected = sum(b["mass"] for b in read_mjcf_bodies(Path(args.mjcf)).values())
        error = abs(total - expected) / expected
        print(f"{'MJCF':<14}{expected:9.3f}")
        verdict = "OK" if error <= args.tolerance else "MISMATCH"
        print(f"\n[{verdict}] simulated is {total / expected * 100:.2f}% of the MJCF mass")
        if verdict == "MISMATCH":
            print(
                "Restore the dropped densities:\n"
                "  python usd_convert/apply_mjcf_masses_to_usd.py "
                f"--mjcf {args.mjcf} --usd <usd package dir>"
            )
            status = 1

    if hasattr(simulator, "shutdown"):
        simulator.shutdown()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
