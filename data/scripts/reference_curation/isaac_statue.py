# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab statue (BUILD_PLAN Step 7's still-open list): does the MuJoCo witness's verdict transfer to the
training plant?

The same test as ``witness.run``, in the training simulator: the smpl_yogi articulation as training builds
it (``robot_configs.smpl_yogi``, the USD with the MJCF masses, PhysX's hard exp-map joint limits, the
terrain's friction averaged with the robot's 0.5, 120 Hz), put at rest in a reference pose, every PD target
at the pose, gains ``--stiffness-scale`` times training's (damping times its square root, as the witness),
the plant's torque limits, gravity on, no root assistance, for ``--duration`` seconds. One environment per
pose, all in one batch.

The feed-forward ``Kp^-1 tau_LP`` of the MuJoCo witness is left out on both sides: the LP's torques are per
MuJoCo XYZ hinge, and PhysX drives the exp-map coordinates, so the comparison uses the plain statue
(``witness.run(..., tau=None)``) against this one.

Verdict, as the witness's: it settles less than 5 cm in the first 0.5 s, then drifts less than 2 cm; its
contacts are judged geometrically at the end: a zone touching in the pose (lowest surface <= 1 cm) must
still be within 2 cm, and no zone beyond 2 cm may come within 1 cm.

Run it as its own process (IsaacLab launches before torch)::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python data/scripts/reference_curation/isaac_statue.py \\
        --poses poses.json --out result.json

``poses.json`` is ``[{"id", "motion", "frame"}]``.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--poses", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--stiffness-scale", type=float, default=10.0)
parser.add_argument("--duration", type=float, default=3.0)
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.WARNING)
SETTLE_S, SETTLE_M, DRIFT_M, TOUCH_M, BAND_M = 0.5, 0.05, 0.02, 0.01, 0.02


def main() -> int:
    from lightning.fabric import Fabric

    from protomotions.robot_configs.factory import robot_config as make_robot_config
    from protomotions.simulator.factory import simulator_config as make_simulator_config
    from protomotions.utils.fabric_config import FabricConfig

    fabric = Fabric(**FabricConfig(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]).as_kwargs())
    fabric.launch()
    simulation_app = AppLauncher({"headless": True, "device": str(fabric.device)}).app

    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.scene_lib import SceneLibConfig
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components
    from reference_curation import capture
    from extract_contact_configs import ZONE_ORDER

    poses = json.load(open(args.poses))
    robot_config = make_robot_config("smpl_yogi")
    k = args.stiffness_scale
    for info in robot_config.control.control_info.values():
        info.stiffness, info.damping = k * info.stiffness, math.sqrt(k) * info.damping
    simulator_config = make_simulator_config(simulator="isaaclab", robot_config=robot_config, headless=True,
                                             num_envs=len(poses), experiment_name="isaac_statue")
    terrain_config, simulator_config = convert_friction_for_simulator(TerrainConfig(), simulator_config)
    comps = build_all_components(terrain_config=terrain_config, scene_lib_config=SceneLibConfig(),
                                 motion_lib_config=MotionLibConfig(), simulator_config=simulator_config,
                                 robot_config=robot_config, device=fabric.device, save_dir=None,
                                 simulation_app=simulation_app)
    sim = comps["simulator"]
    sim._initialize_with_markers(None)
    robot = sim._robot
    dev = robot.data.body_pos_w.device
    sim_bodies = list(robot.data.body_names)
    sim_dofs = list(robot.data.joint_names)
    common_bodies = capture.skeleton().names
    common_dofs = [f"{b}_{a}" for b in common_bodies[1:] for a in "xyz"]
    body_perm = [sim_bodies.index(b) for b in common_bodies]
    dof_perm = [common_dofs.index(d) for d in sim_dofs]             # sim dof i <- common dof dof_perm[i]
    # The flat terrain's importer holds no env origins, and every env spawns at the same place (ProtoMotions
    # places robots at reset): the poses get their own 2.5 m grid on the flat map, or they collide.
    side = int(math.ceil(math.sqrt(len(poses))))
    grid = torch.tensor([[2.0 + 2.5 * (i % side), 2.0 + 2.5 * (i // side), 0.0] for i in range(len(poses))])
    origins = grid.to(dev)

    root, dofs, start_pos, start_rot = [], [], [], []
    for p in poses:
        m = torch.load(p["motion"], map_location="cpu", weights_only=False)
        f = int(p["frame"])
        pos, rot = m["rigid_body_pos"][f].double(), m["rigid_body_rot"][f].double()
        start_pos.append(pos.numpy())
        start_rot.append(rot.numpy())
        q = rot[0]
        root.append(torch.cat([pos[0], torch.stack([q[3], q[0], q[1], q[2]]), torch.zeros(6, dtype=torch.float64)]))
        dofs.append(m["dof_pos"][f].double()[dof_perm])
    root = torch.stack(root).float().to(dev)
    root[:, :2] += origins[:, :2]
    dofs = torch.stack(dofs).float().to(dev)
    ids_all = torch.arange(len(poses), device=dev)
    robot.write_root_state_to_sim(root, ids_all)
    robot.write_joint_state_to_sim(dofs, torch.zeros_like(dofs), None, ids_all)
    robot.set_joint_position_target(dofs, joint_ids=None, env_ids=ids_all)
    sim._scene.write_data_to_sim()
    dt = sim._sim.get_physics_dt()
    n, n_settle = int(round(args.duration / dt)), int(round(SETTLE_S / dt))

    def body_state():
        p = robot.data.body_pos_w[:, body_perm].clone()
        p[:, :, :2] -= origins[:, None, :2]
        q = robot.data.body_quat_w[:, body_perm][..., [1, 2, 3, 0]].clone()     # xyzw
        return p.double().cpu().numpy(), q.double().cpu().numpy()

    sim._scene.update(dt=dt)
    x0, _ = body_state()
    ground_start = float(np.min([capture.body_min_z(x0[e][None], start_rot[e][None]).min() for e in range(len(poses))]))
    x_settle = None
    for i in range(n):
        robot.set_joint_position_target(dofs, joint_ids=None, env_ids=ids_all)
        sim._scene.write_data_to_sim()
        sim._sim.step(render=False)
        sim._scene.update(dt=dt)
        if i + 1 == n_settle:
            x_settle, _ = body_state()
    x_end, q_end = body_state()
    out = []
    for e, p in enumerate(poses):
        z0 = capture.zone_min(capture.body_min_z(start_pos[e][None], start_rot[e][None]))[0]
        z1 = capture.zone_min(capture.body_min_z(x_end[e][None], q_end[e][None]))[0]
        settle = float(np.linalg.norm(x_settle[e] - x0[e], axis=-1).max())
        drift = float(np.linalg.norm(x_end[e] - x_settle[e], axis=-1).max())
        lifted = [ZONE_ORDER[i] for i in range(len(ZONE_ORDER)) if z0[i] <= TOUCH_M and z1[i] > BAND_M]
        landed = [ZONE_ORDER[i] for i in range(len(ZONE_ORDER)) if z0[i] > BAND_M and z1[i] <= TOUCH_M]
        out.append({**p, "settle_cm": round(100 * settle, 2), "drift_cm": round(100 * drift, 2),
                    "final_cm": round(100 * float(np.linalg.norm(x_end[e] - x0[e], axis=-1).max()), 2),
                    "lifted": lifted, "landed": landed,
                    "passed": bool(settle < SETTLE_M and drift < DRIFT_M and not (lifted or landed))})
    json.dump({"stiffness_scale": k, "duration_s": args.duration, "physics_dt": dt, "lowest_start_m": ground_start,
               "results": out},
              open(args.out, "w"), indent=1)
    print(f"isaac statue: {sum(r['passed'] for r in out)}/{len(out)} held -> {args.out}", flush=True)
    os._exit(0)   # simulation_app.close() can hang after the result is written (a known IsaacLab exit hang)


if __name__ == "__main__":
    sys.exit(main())
