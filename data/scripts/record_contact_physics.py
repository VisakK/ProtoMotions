# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the contact and actuation physics of a trained tracker, per physics substep.

Rolls a Stage-1 policy in closed loop (one environment per reference motion) and
dumps, for every *physics* step (not every policy step), the quantities needed to
reason about how the learned motion is actually supported:

* per contact **pair** normal force -- ``(body, ground)`` and ``(body, body)`` --
  from a PhysX ``RigidContactView`` created with one filter per robot body plus
  the terrain mesh,
* the raw **contact manifold**: every contact point PhysX reported, with its
  normal, its normal-force magnitude and its separation distance.  This is what
  makes a real centre of pressure, a real support polygon and a contact-patch
  area possible -- the aggregated ``net_forces_w`` the policy observes cannot
  give any of them,
* per-point **friction** (tangential) forces,
* joint torques, both IsaacLab's PD reconstruction (``applied_torque``) and
  PhysX's measured joint force (``get_dof_projected_joint_forces``),
* body poses / centres of mass / velocities and per-body mass, for the whole-body
  COM and for the m*(g + a) force-balance check.

Nothing here interprets the numbers; :mod:`data/scripts/plot_contact_physics.py`
does that.

Example::

    python data/scripts/record_contact_physics.py \
      --checkpoint results/smpl_yogi_balance_v2_contact_rich_rew_realtorque/last.ckpt \
      --motion-file data/smpl/yoga_yogi_balance_subset_v2_contacts \
      --motions 220923_Chair_Pose_or_Utkatasana_-b \
      --out-dir results/Contact_Physics_analysis
"""

from __future__ import annotations

import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="A single .motion file (a packaged .pt or a directory is accepted "
        "only when it holds exactly one clip). One clip per process -- see "
        "check_single_motion for why.",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--simulator", type=str, default="isaaclab")
    parser.add_argument(
        "--max-contact-points",
        type=int,
        default=16,
        help="PhysX contact-manifold points kept per (body, filter) pair. PhysX "
        "reports at most ~4-8 for a convex-vs-plane patch; 16 leaves headroom.",
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=0,
        help="Policy steps to run past the end of the longest clip.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Hard cap on policy steps (0 = run every clip to its end).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic mean action.",
    )
    parser.add_argument(
        "--usd-asset",
        type=str,
        default=None,
        help="Override robot.asset.usd_asset_file_name (e.g. to match a checkpoint's plant).",
    )
    return parser


parser = create_parser()
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
_logger = logging.getLogger("record_contact_physics")


class _Log:
    """Isaac's app launcher reconfigures python logging; print survives it."""

    @staticmethod
    def info(fmt, *fmt_args) -> None:
        print("record_contact_physics: " + (fmt % fmt_args if fmt_args else fmt), flush=True)
        _logger.debug(fmt, *fmt_args)


log = _Log()

GROUND_FILTER_PATH = "/World/ground/terrain/mesh"
GROUND_NAME = "ground"


# ---------------------------------------------------------------------------
# motion bookkeeping
# ---------------------------------------------------------------------------
def list_motion_stems(motion_file: str) -> list[str]:
    """Stems of the motions in ``motion_file``, in motion-id order."""
    path = Path(motion_file)
    if path.is_dir():
        return [p.stem for p in sorted(path.rglob("*.motion"))]
    if path.suffix == ".motion":
        return [path.stem]
    packaged = torch.load(motion_file, map_location="cpu", weights_only=False)
    return [Path(f).stem for f in packaged["motion_files"]]


def check_single_motion(motion_file: str) -> str:
    """Assert ``motion_file`` holds exactly one clip and return its stem.

    One clip per process, deliberately.  Two separate things only behave at
    ``num_envs = 1`` with a single-motion library:

    * PhysX resolves the per-body contact filters.  With more environments the
      ``env_*`` filter patterns expand to more prims than the sensor set can
      absorb; the view comes back with empty filter paths and every body-body
      column silently stays zero.
    * The motion manager samples ids from the subset at reset rather than
      pinning env *i* to clip *i*, and its clip ordering is its own -- it is not
      the sorted directory listing.  Handing it one clip removes the ambiguity.
    """
    stems = list_motion_stems(motion_file)
    if len(stems) != 1:
        raise SystemExit(
            f"--motion-file resolves to {len(stems)} motions; pass one .motion file "
            f"(record one clip per process). First few: {stems[:5]}"
        )
    return stems[0]


# ---------------------------------------------------------------------------
# PhysX pair-contact view
# ---------------------------------------------------------------------------
class PairContactView:
    """A ``RigidContactView`` over every robot body, filtered against the ground
    *and* every other robot body.

    The contact sensors ProtoMotions builds are per-body and filtered against the
    terrain only, with ``track_contact_points`` off, so they can report neither
    body-body forces (crow pose: knee on upper arm) nor contact points.  This
    view adds both without touching the trained configuration.
    """

    def __init__(self, sim_view, body_root_glob: str, body_names: list[str],
                 num_envs: int, max_points_per_pair: int):
        self.body_names = list(body_names)
        self.num_envs = num_envs
        num_bodies = len(body_names)
        pattern = f"{body_root_glob}({'|'.join(body_names)})"
        filters = [GROUND_FILTER_PATH] + [f"{body_root_glob}{b}" for b in body_names]
        self.view = sim_view.create_rigid_contact_view(
            pattern,
            filter_patterns=filters,
            max_contact_data_count=max_points_per_pair * num_bodies * num_envs,
        )
        self.filter_names = [GROUND_NAME] + list(body_names)

        sensor_paths = list(self.view.sensor_paths)
        if len(sensor_paths) != num_envs * num_bodies:
            raise RuntimeError(
                f"pair contact view matched {len(sensor_paths)} sensors, "
                f"expected {num_envs * num_bodies}"
            )
        if self.view.filter_count != len(filters):
            raise RuntimeError(
                f"pair contact view has filter_count={self.view.filter_count}, "
                f"expected {len(filters)} (1 ground + {num_bodies} bodies)"
            )
        # PhysX pairs each sensor with its own copy of every filter pattern. When
        # the pattern expands to a different number of prims than the sensor set
        # can absorb it silently hands back empty paths and every body-body
        # column stays zero, so refuse to record rather than log a lie.
        resolved = list(self.view.filter_paths)
        if resolved and isinstance(resolved[0], (list, tuple)):
            unresolved = [i for i, p in enumerate(resolved[0]) if not str(p)]
            if unresolved:
                missing = [filters[i] for i in unresolved[:3]]
                raise RuntimeError(
                    f"{len(unresolved)}/{len(filters)} contact filters did not resolve "
                    f"to a prim (e.g. {missing}). This happens with num_envs > 1; "
                    "record one motion per process."
                )

        # Row -> (env, body).  Resolved from the prim paths rather than assumed,
        # because PhysX does not promise env-major ordering.
        body_index = {b: i for i, b in enumerate(body_names)}
        row_env = np.zeros(len(sensor_paths), dtype=np.int64)
        row_body = np.zeros(len(sensor_paths), dtype=np.int64)
        for row, path in enumerate(sensor_paths):
            match = re.search(r"/env_(\d+)/", path)
            if match is None:
                raise RuntimeError(f"cannot parse env index from sensor path '{path}'")
            row_env[row] = int(match.group(1))
            leaf = path.rsplit("/", 1)[-1]
            if leaf not in body_index:
                raise RuntimeError(f"unexpected sensor leaf '{leaf}' in '{path}'")
            row_body[row] = body_index[leaf]
        self.row_env = row_env
        self.row_body = row_body
        # Flat scatter index: row -> env * num_bodies + body.
        self.row_to_slot = torch.as_tensor(row_env * num_bodies + row_body)
        self.num_bodies = num_bodies
        self.sensor_paths = sensor_paths
        try:
            self.filter_paths = list(self.view.filter_paths)
        except Exception:  # pragma: no cover - backend dependent
            self.filter_paths = []

    def describe(self) -> str:
        return (
            f"sensors={self.view.sensor_count} filters={self.view.filter_count} "
            f"max_contact_data={self.view.max_contact_data_count}\n"
            f"  first sensor paths: {self.sensor_paths[:3]}\n"
            f"  filter paths[:3]  : {self.filter_paths[:3]}"
        )


def _gather_pair_buffer(counts, starts, *columns):
    """Flatten PhysX's ``(count, start_index)`` per-pair contact buffers.

    Returns ``(rows, cols, gathered_columns)`` where ``rows``/``cols`` index the
    (sensor, filter) pair each contact point belongs to.
    """
    device = counts.device
    mask = counts > 0
    if not bool(mask.any()):
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, [c[:0] for c in columns]
    rows, cols = mask.nonzero(as_tuple=True)
    count = counts[rows, cols].to(torch.long)
    start = starts[rows, cols].to(torch.long)
    total = int(count.sum())
    pair_ids = torch.repeat_interleave(torch.arange(rows.numel(), device=device), count)
    block_starts = count.cumsum(0) - count
    deltas = torch.arange(total, device=device) - block_starts.repeat_interleave(count)
    flat = start[pair_ids] + deltas
    gathered = [c.index_select(0, flat) for c in columns]
    return rows[pair_ids], cols[pair_ids], gathered


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------
class SubstepRecorder:
    """Captures one sample per physics substep, hooked onto ``scene.update``."""

    def __init__(self, simulator, pair_view: PairContactView, num_envs: int):
        self.sim = simulator
        self.robot = simulator._robot
        self.pair_view = pair_view
        self.num_envs = num_envs
        self.num_bodies = pair_view.num_bodies
        self.num_filters = pair_view.view.filter_count
        self.device = simulator.device
        self.dt_phys = float(simulator._sim.get_physics_dt())
        self.armed = False
        self.slot = pair_view.row_to_slot.to(self.device)

        self.dense: dict[str, list] = {k: [] for k in (
            "pair_force_w",      # [Ts,E,B,F,3] normal force, world frame
            "net_force_w",       # [Ts,E,B,3]   net normal force per body
            "torque_applied",    # [Ts,E,D]     IsaacLab PD reconstruction
            "torque_measured",   # [Ts,E,D]     PhysX projected joint force
            "dof_pos",           # [Ts,E,D]
            "dof_vel",           # [Ts,E,D]
            "body_pos_w",        # [Ts,E,B,3]   link frame origin
            "body_quat_w",       # [Ts,E,B,4]   wxyz
            "body_com_pos_w",    # [Ts,E,B,3]
            "body_com_vel_w",    # [Ts,E,B,3]
        )}
        # Sparse contact-manifold records, one row per contact point.
        self.points: dict[str, list] = {k: [] for k in (
            "step", "env", "body", "filter", "pos", "normal", "force", "sep"
        )}
        self.friction: dict[str, list] = {k: [] for k in (
            "step", "env", "body", "filter", "pos", "force"
        )}
        self.num_substeps = 0

    def _scatter_rows(self, flat: torch.Tensor, trailing: tuple[int, ...]) -> torch.Tensor:
        """(sensor_rows, ...) -> (E, B, ...) using the resolved row mapping."""
        out = torch.zeros(
            (self.num_envs * self.num_bodies, *trailing),
            device=flat.device,
            dtype=flat.dtype,
        )
        out[self.slot] = flat.reshape(-1, *trailing)
        return out.reshape(self.num_envs, self.num_bodies, *trailing)

    def capture(self) -> None:
        if not self.armed:
            return
        step = self.num_substeps
        view = self.pair_view.view
        dt = self.dt_phys

        net = view.get_net_contact_forces(dt=dt)
        matrix = view.get_contact_force_matrix(dt=dt)
        self.dense["net_force_w"].append(
            self._scatter_rows(net, (3,)).to("cpu", non_blocking=False)
        )
        self.dense["pair_force_w"].append(
            self._scatter_rows(matrix, (self.num_filters, 3)).to("cpu")
        )

        forces, points, normals, seps, counts, starts = view.get_contact_data(dt=dt)
        counts = counts.view(-1, self.num_filters)
        starts = starts.view(-1, self.num_filters)
        rows, cols, (pt, nrm, frc, sep) = _gather_pair_buffer(
            counts, starts, points, normals, forces.view(-1, 1), seps.view(-1, 1)
        )
        if rows.numel():
            rows_np = rows.to("cpu").numpy()
            self.points["step"].append(np.full(rows_np.shape, step, dtype=np.int32))
            self.points["env"].append(self.pair_view.row_env[rows_np].astype(np.int16))
            self.points["body"].append(self.pair_view.row_body[rows_np].astype(np.int16))
            self.points["filter"].append(cols.to("cpu").numpy().astype(np.int16))
            self.points["pos"].append(pt.to("cpu").numpy().astype(np.float32))
            self.points["normal"].append(nrm.to("cpu").numpy().astype(np.float32))
            self.points["force"].append(frc.to("cpu").numpy().reshape(-1).astype(np.float32))
            self.points["sep"].append(sep.to("cpu").numpy().reshape(-1).astype(np.float32))

        fr_forces, fr_points, fr_counts, fr_starts = view.get_friction_data(dt=dt)
        fr_counts = fr_counts.view(-1, self.num_filters)
        fr_starts = fr_starts.view(-1, self.num_filters)
        rows, cols, (fpt, ffrc) = _gather_pair_buffer(
            fr_counts, fr_starts, fr_points, fr_forces
        )
        if rows.numel():
            rows_np = rows.to("cpu").numpy()
            self.friction["step"].append(np.full(rows_np.shape, step, dtype=np.int32))
            self.friction["env"].append(self.pair_view.row_env[rows_np].astype(np.int16))
            self.friction["body"].append(self.pair_view.row_body[rows_np].astype(np.int16))
            self.friction["filter"].append(cols.to("cpu").numpy().astype(np.int16))
            self.friction["pos"].append(fpt.to("cpu").numpy().astype(np.float32))
            self.friction["force"].append(ffrc.to("cpu").numpy().astype(np.float32))

        data = self.robot.data
        self.dense["torque_applied"].append(data.applied_torque.to("cpu"))
        try:
            measured = self.robot.root_physx_view.get_dof_projected_joint_forces()
            self.dense["torque_measured"].append(measured.to("cpu").clone())
        except Exception:  # pragma: no cover - backend dependent
            self.dense["torque_measured"].append(torch.zeros_like(data.applied_torque).cpu())
        self.dense["dof_pos"].append(data.joint_pos.to("cpu"))
        self.dense["dof_vel"].append(data.joint_vel.to("cpu"))
        self.dense["body_pos_w"].append(data.body_link_pos_w.to("cpu"))
        self.dense["body_quat_w"].append(data.body_link_quat_w.to("cpu"))
        self.dense["body_com_pos_w"].append(data.body_com_pos_w.to("cpu"))
        self.dense["body_com_vel_w"].append(data.body_com_lin_vel_w.to("cpu"))
        self.num_substeps += 1

    def stacked(self) -> dict[str, np.ndarray]:
        out = {k: torch.stack(v, dim=0).numpy() for k, v in self.dense.items() if v}
        for prefix, store in (("cp", self.points), ("fr", self.friction)):
            for key, chunks in store.items():
                name = f"{prefix}_{key}"
                if chunks:
                    out[name] = np.concatenate(chunks, axis=0)
                else:
                    shape = (0, 3) if key in ("pos", "normal") else (0,)
                    out[name] = np.zeros(shape, dtype=np.float32)
        return out


# ---------------------------------------------------------------------------
def _per_body_masses(robot) -> np.ndarray:
    for getter in (
        lambda: robot.root_physx_view.get_masses()[0],
        lambda: robot.data.default_mass[0],
    ):
        try:
            return np.asarray(getter().cpu().numpy(), dtype=np.float64)
        except Exception:  # pragma: no cover - backend dependent
            continue
    return np.zeros(0, dtype=np.float64)


def main() -> None:
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = Path(args.checkpoint)
    cfg_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert cfg_path.exists(), f"missing {cfg_path}"
    cfgs = torch.load(cfg_path, map_location="cpu", weights_only=False)

    robot_config = cfgs["robot"]
    simulator_config = cfgs["simulator"]
    terrain_config = cfgs.get("terrain")
    scene_lib_config = cfgs["scene_lib"]
    motion_lib_config = cfgs["motion_lib"]
    env_config = cfgs["env"]
    agent_config = cfgs["agent"]

    motion_lib_config.motion_file = args.motion_file
    if args.usd_asset:
        log.info("USD override: %s", args.usd_asset)
        robot_config.asset.usd_asset_file_name = args.usd_asset

    motion_name = check_single_motion(args.motion_file)
    motion_id = 0
    num_envs = 1
    log.info("recording %s", motion_name)

    simulator_config.num_envs = num_envs
    simulator_config.headless = True
    if getattr(simulator_config, "domain_randomization", None) is not None:
        simulator_config.domain_randomization.push = None
    robot_config.reset_noise = None

    env_config.motion_manager.subset_method = [motion_id]
    env_config.motion_manager.init_start_prob = 1.0
    env_config.max_episode_length = 1_000_000

    from lightning.fabric import Fabric  # noqa: E402
    from protomotions.utils.fabric_config import FabricConfig  # noqa: E402

    fabric = Fabric(
        **FabricConfig(
            accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()

    sim_extra = {}
    if args.simulator == "isaaclab":
        sim_extra["simulation_app"] = AppLauncher(
            {"headless": True, "device": str(fabric.device)}
        ).app

    from protomotions.simulator.base_simulator.utils import (  # noqa: E402
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402
    from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
    from protomotions.utils.component_builder import build_all_components  # noqa: E402
    from protomotions.utils.hydra_replacement import get_class  # noqa: E402

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **sim_extra,
    )

    env: BaseEnv = get_class(env_config._target_)(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    agent: BaseAgent = get_class(agent_config._target_)(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False, load_training_state=False)
    agent.eval()

    simulator = env.simulator
    body_names = list(robot_config.kinematic_info.body_names)
    dof_names = list(robot_config.kinematic_info.dof_names)
    sim_body_names = list(simulator._robot.body_names)
    sim_dof_names = list(simulator._robot.joint_names)

    body_root_glob = robot_config.asset.usd_bodies_root_prim_path.replace(".*", "*")
    sensor = next(iter(simulator._contact_sensor_map.values()))
    sim_view = sensor._physics_sim_view
    pair_view = PairContactView(
        sim_view,
        body_root_glob=body_root_glob,
        body_names=sim_body_names,
        num_envs=num_envs,
        max_points_per_pair=args.max_contact_points,
    )
    log.info("pair contact view: %s", pair_view.describe())

    recorder = SubstepRecorder(simulator, pair_view, num_envs)
    scene = simulator._scene
    original_update = scene.update

    def patched_update(dt):
        original_update(dt)
        recorder.capture()

    scene.update = patched_update

    ctrl: dict[str, list] = {k: [] for k in (
        "substep_index", "motion_time", "progress", "done", "terminated", "track_err"
    )}

    def ctrl_snapshot(done=None, terminated=None) -> None:
        manager = env.motion_manager
        state = simulator.get_robot_state()
        ref = env.motion_lib.get_motion_state(manager.motion_ids, manager.motion_times)
        ref_pos = ref.rigid_body_pos.clone()
        ref_pos = ref_pos + env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_pos
        )
        err = (ref_pos - state.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
        zeros = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        ctrl["substep_index"].append(np.full(num_envs, recorder.num_substeps, np.int32))
        ctrl["motion_time"].append(manager.motion_times.cpu().numpy())
        ctrl["progress"].append(env.progress_buf.cpu().numpy())
        ctrl["done"].append((zeros if done is None else done.bool()).cpu().numpy())
        ctrl["terminated"].append(
            (zeros if terminated is None else terminated.bool()).cpu().numpy()
        )
        ctrl["track_err"].append(err.detach().cpu().numpy())

    action_key = "action" if args.stochastic else "mean_action"
    obs, _ = env.reset()

    # The motion manager only samples ids on reset, so this has to come after it.
    playing = int(env.motion_manager.motion_ids[0])
    library = [Path(f).stem for f in env.motion_lib.motion_files]
    assert playing == motion_id and library[playing] == motion_name, (
        f"motion manager is playing clip {playing} ({library[playing]}), "
        f"not the requested {motion_id} ({motion_name})"
    )
    motion_length = float(env.motion_lib.motion_lengths[playing])
    dt_ctrl = float(env.dt)
    total_steps = int(np.ceil(motion_length / dt_ctrl)) + args.extra_steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    log.info(
        "clip length %.2f s -> %d policy steps (dt=%.5f, physics dt=%.5f)",
        motion_length,
        total_steps,
        dt_ctrl,
        recorder.dt_phys,
    )

    recorder.armed = True
    ctrl_snapshot()

    for step in range(total_steps):
        with torch.no_grad():
            td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            model_outs = agent.model(td)
            action = model_outs[action_key if action_key in model_outs else "action"]
        obs, _rew, dones, terminated, _extras = env.step(action)
        ctrl_snapshot(done=dones, terminated=terminated)
        if (step + 1) % 100 == 0:
            log.info("step %d/%d (%d substeps)", step + 1, total_steps, recorder.num_substeps)

    recorder.armed = False
    scene.update = original_update

    dense = recorder.stacked()
    ctrl_arrays = {k: np.stack(v, axis=0) for k, v in ctrl.items()}
    masses = _per_body_masses(simulator._robot)
    # The terrain's friction is only half the story: PhysX combines it with the
    # robot's own shape material (which nothing in the config sets, so it keeps
    # PhysX's 0.5 default), and the combine mode decides how.
    try:
        robot_material = np.asarray(
            simulator._robot.root_physx_view.get_material_properties()[0].cpu().numpy(),
            np.float64,
        )
    except Exception:  # pragma: no cover - backend dependent
        robot_material = np.zeros((0, 3))
    combine_mode = getattr(
        terrain_config.sim_config.combine_mode, "value", terrain_config.sim_config.combine_mode
    )
    log.info(
        "friction: terrain %.2f, robot shapes %s, combine '%s'",
        float(terrain_config.sim_config.static_friction),
        np.unique(robot_material[:, 0]).round(3).tolist() if len(robot_material) else "n/a",
        combine_mode,
    )
    # ProtoMotions spawns each env from the terrain's respawn grid rather than an
    # env-origin buffer, so anchor the BEV plots on each env's first pelvis xy.
    origin_xy = dense["body_pos_w"][0, :, 0, :2].copy()

    # Gains as the simulator actually has them; the config is only a fallback.
    control_info = robot_config.control.control_info

    def _from_config(field: str) -> np.ndarray:
        values = []
        for name in sim_dof_names:
            raw = getattr(control_info[name], field, None) if name in control_info else None
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(np.nan)
        return np.array(values, np.float64)

    def _from_sim(attr: str, field: str) -> np.ndarray:
        try:
            return np.asarray(
                getattr(simulator._robot.data, attr)[0].cpu().numpy(), np.float64
            )
        except Exception:  # pragma: no cover - backend dependent
            return _from_config(field)

    stiffness = _from_sim("joint_stiffness", "stiffness")
    damping = _from_sim("joint_damping", "damping")
    effort_limit = _from_config("effort_limit")
    try:
        effort_limit = np.asarray(
            simulator._robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy(),
            np.float64,
        )
    except Exception:  # pragma: no cover - backend dependent
        pass

    out_root = Path(args.out_dir)
    decimation = int(simulator.decimation)
    env_idx = 0
    motion_dir = out_root / motion_name
    motion_dir.mkdir(parents=True, exist_ok=True)

    # Truncate at the clip's end: the motion manager resets the track once the
    # clip runs out, and everything after that reset belongs to a new episode.
    done = ctrl_arrays["done"][:, env_idx]
    end_ctrl = int(np.argmax(done)) if done.any() else len(done) - 1
    n_ctrl = end_ctrl + 1
    n_sub = min(int(ctrl_arrays["substep_index"][end_ctrl, env_idx]), recorder.num_substeps)

    payload: dict[str, np.ndarray] = {}
    for key, arr in dense.items():
        if key.startswith(("cp_", "fr_")):
            continue
        payload[key] = arr[:n_sub, env_idx]
    for key, arr in ctrl_arrays.items():
        payload[f"ctrl_{key}"] = arr[:n_ctrl, env_idx]

    for prefix in ("cp", "fr"):
        sel = (dense[f"{prefix}_env"] == env_idx) & (dense[f"{prefix}_step"] < n_sub)
        for key in ("step", "body", "filter", "pos", "normal", "force", "sep"):
            full = f"{prefix}_{key}"
            if full in dense:
                payload[full] = dense[full][sel]

    np.savez_compressed(
        motion_dir / "rollout.npz",
        motion_name=np.array(motion_name),
        motion_id=np.array(motion_id),
        motion_length_s=np.array(motion_length),
        checkpoint=np.array(str(checkpoint)),
        body_names=np.array(sim_body_names),
        common_body_names=np.array(body_names),
        dof_names=np.array(sim_dof_names),
        common_dof_names=np.array(dof_names),
        filter_names=np.array(pair_view.filter_names),
        body_masses=masses,
        origin_xy=origin_xy[env_idx],
        dt_phys=np.array(recorder.dt_phys),
        dt_ctrl=np.array(dt_ctrl),
        decimation=np.array(decimation),
        effort_limit=effort_limit,
        stiffness=stiffness,
        damping=damping,
        gravity=np.array(
            simulator._sim.cfg.gravity
            if hasattr(simulator._sim, "cfg")
            else (0.0, 0.0, -9.81)
        ),
        static_friction=np.array(float(terrain_config.sim_config.static_friction)),
        dynamic_friction=np.array(float(terrain_config.sim_config.dynamic_friction)),
        friction_combine_mode=np.array(str(combine_mode)),
        robot_material=robot_material,
        contact_offset=np.array(float(simulator_config.sim.physx.contact_offset)),
        rest_offset=np.array(float(simulator_config.sim.physx.rest_offset)),
        **payload,
    )
    log.info(
        "wrote %s (%d substeps, %d policy steps, %d contact points)",
        motion_dir / "rollout.npz",
        n_sub,
        n_ctrl,
        int(payload["cp_step"].shape[0]),
    )

    if hasattr(simulator, "shutdown"):
        simulator.shutdown()


if __name__ == "__main__":
    main()
