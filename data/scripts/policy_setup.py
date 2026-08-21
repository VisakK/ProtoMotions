# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared checkpoint -> (env, agent) setup for offline policy analysis.

Mirrors :mod:`protomotions.inference_agent` exactly -- same frozen-config load,
same override precedence, same component build order -- so that anything these
tools measure matches what ``inference_agent.py`` would show interactively.  It
exists so ``eval_per_motion.py`` and ``render_policy_videos.py`` cannot drift
apart from each other or from inference.

Import order matters and is the caller's responsibility: the simulator package
has to be imported before ``torch``, so a script must run
``import_simulator_before_torch(args.simulator)`` at module level *before*
importing this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def add_common_args(parser) -> None:
    """Arguments shared by every rollout tool (names match inference_agent.py)."""
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--simulator", type=str, default="isaaclab")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--motion-file", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="config overrides, e.g. env.ref_respawn_offset=0.005",
    )


def build(args, app_launcher_cls: Optional[Any] = None) -> Dict[str, Any]:
    """Build fabric, components, env and agent from a checkpoint.

    Returns a dict with keys: fabric, env, agent, motion_lib, simulator, configs.
    """
    import torch
    from lightning.fabric import Fabric

    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class

    checkpoint = Path(args.checkpoint)
    resolved_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_path.exists(), f"Could not find resolved configs at {resolved_path}"

    log.info("Loading resolved configs from %s", resolved_path)
    cfg = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = cfg["robot"]
    simulator_config = cfg["simulator"]
    terrain_config = cfg.get("terrain")
    scene_lib_config = cfg["scene_lib"]
    motion_lib_config = cfg["motion_lib"]
    env_config = cfg["env"]
    agent_config = cfg["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test

        log.info("Switching simulator %s -> %s", current_simulator, args.simulator)
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    if args.num_envs is not None:
        simulator_config.num_envs = args.num_envs
    if args.motion_file is not None:
        motion_lib_config.motion_file = args.motion_file
    simulator_config.headless = args.headless

    if args.overrides:
        from protomotions.utils.config_utils import (
            apply_config_overrides,
            parse_cli_overrides,
        )

        cli_overrides = parse_cli_overrides(args.overrides)
        if cli_overrides:
            apply_config_overrides(
                cli_overrides, env_config, simulator_config, robot_config,
                agent_config, terrain_config, motion_lib_config, scene_lib_config,
            )
            if any(key.startswith("robot.") for key in cli_overrides):
                # apply_config_overrides assigns straight to the field, so an
                # abstract body selection ("all", "all_left_foot_bodies") is left
                # as the raw string and every consumer that iterates it gets
                # characters. update_fields() re-runs the resolver and is
                # idempotent on already-resolved values.
                robot_config.update_fields()

    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric: Fabric = Fabric(
        **FabricConfig(
            accelerator=accelerator, devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        assert app_launcher_cls is not None, "isaaclab needs the AppLauncher class"
        app = app_launcher_cls(
            {"headless": args.headless, "device": str(fabric.device)}
        )
        simulator_extra_params["simulation_app"] = app.app

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
        **simulator_extra_params,
    )

    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(args.checkpoint, load_env=False, load_training_state=False)

    return dict(
        fabric=fabric,
        env=env,
        agent=agent,
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
        configs=cfg,
    )


def motion_names(motion_lib) -> List[str]:
    """Clip name per motion id, falling back to an index label."""
    import os

    files = getattr(motion_lib, "motion_files", None)
    n = motion_lib.num_motions()
    if not files or len(files) != n:
        return [f"motion_{i:04d}" for i in range(n)]
    return [os.path.basename(f).replace(".motion", "") for f in files]
