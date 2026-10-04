"""Plant v2 in IsaacLab/PhysX -- the simulator the expert trains in -- as a batched rollout engine for sampling MPC.

Lane T generated its edges on a MuJoCo copy of plant v2 (``plant_mj``). That copy matches the masses, gains, torque
limits and friction, but not the actuation: MuJoCo chains three hinges per joint and its PD acts on per-body Euler
angles, while PhysX's spherical joints carry the exp-map coordinates, a hard box and a drive that acts on those
coordinates (PLAN.MD §1.3 "Sim-to-sim"). Measured here on the same exemplars at the same gains (2026-10-04,
open loop, PD targets at the pose; ``expert_revist/graph_growth_2026_10_03/t_edges_physx/README.MD``,
scripts and outputs in ``output/edge_synthesis/physx/plant_measurements/``):

=====================  ======================================  ======================================
hold                   MuJoCo (``plant_mj``)                   PhysX (this module)
=====================  ======================================  ======================================
Crow -a@651            root -2.4 cm at 0.17 s, falls at 0.9 s  root -10 cm at 0.2 s: the right elbow
                                                               gives way, saturated at 100 N m
Chaturanga -a@485      holds (root 0.265 -> 0.219 m)           root 0.265 -> 0.14 m, right elbow
                                                               0.8 rad off its target, saturated
Handstand -a@1193      holds 1.8 s                             root -10 cm at 0.5 s
=====================  ======================================  ======================================

The USD's joints are proper mirrors (identity frames, mirrored limits), so this is PhysX's drive, not an asset bug:
at large rotations (the chaturanga elbow sits at |v| = 1.94 rad) a PD on exp-map coordinates is not a
restoring spring, and a stiffened statue goes unstable rather than holding (gains and torque limits x100: the right
elbow still runs 1.3 rad in 67 ms). So nothing generated on the MuJoCo plant transfers as-is, and the gravity
feed-forward that lane T reads off a stiff MuJoCo statue has no PhysX counterpart. Sampling MPC needs no model:
it is run here, in the training plant itself.

The plant is built exactly as training builds it (``retarget_v2_physx.py``'s recipe): robot ``smpl_yogi_v2``
(BUILT_IN_PD at the training gains, the MJCF's torque limits, PhysX's hard exp-map box, every non-parent-child
body pair colliding), ``TerrainConfig()`` (flat; terrain 1.0 averaged with the robot's 0.5 -> 0.75), the robot's
IsaacLab sim params (120 Hz physics, decimation 4 -> 30 Hz control, TGS 4/4 iterations, contact offset 2 cm,
rest offset 0) and a contact sensor on every body (the ground force of a body = its sensor's terrain column, as
``IsaacLabSimulator._get_simulator_bodies_contact_buf`` reads it). Body-pair sensors (training's
``contact_pair_bodies``) are left out: sensors report, they do not act.

Only the stepping differs from ``IsaacLabSimulator._physics_step``, never the physics. Training writes the
targets before each of the 4 physics substeps through IsaacLab's per-actuator loop (69 actuator objects, 7.8 ms
per substep at 256 envs) and refreshes all 24 contact sensors after each (4.3 ms); the targets are held for the
whole control step anyway (zero-order hold), so ``control_step`` writes them to PhysX once and refreshes the
sensors only where a force is read. The cost is almost all fixed overhead (66 ms per control step at 256 envs,
74 ms at 2048 through the stock path), so samples are cheap: many seeds run side by side as *blocks* of envs.

Coordinates are training's COMMON order (``kinematic_info``: MJCF body and dof order; ``.motion`` order), dof =
exp-map, quaternions xyzw, positions env-local (each env sits at its own ``grid`` offset on the 100 m flat map:
envs never collide with each other, and spreading them keeps the broadphase small). Root velocities are the
root COM's (IsaacLab writes and reads COM velocity, so a state round-trips exactly).

A restored state continues like an unrestored one to PhysX's own noise floor (measured: 0.09-0.56 mm max body
deviation over 0.3 s between a restored copy and the continuous run, the same as between two restored copies).
That floor is the envs' different world positions (float32 rounding at 5-100 m, amplified by contact-rich
dynamics), not run-to-run nondeterminism: a whole MPPI run repeated with the same env count, grid and seeds
reproduced bit for bit (B1, 4 seeds, 2026-10-04).

Import order: IsaacLab must be imported before torch. A CLI using this module starts with
``AppLauncher = protomotions.utils.simulator_imports.import_simulator_before_torch("isaaclab")`` and calls
``launch`` before importing anything that imports torch-dependent IsaacLab modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import torch

ROBOT = "smpl_yogi_v2"
GRID_ORIGIN = (5.0, 5.0)
GRID_SPACING = 2.5


# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
def launch(app_launcher_cls, num_envs: int, headless: bool = True, experiment_name: str = "edge_mppi_physx"):
    """Start IsaacLab and build training's simulator for ``num_envs`` envs. Returns ``(simulation_app, sim,
    robot_config, components)``. ``app_launcher_cls`` is ``import_simulator_before_torch("isaaclab")``."""
    from lightning.fabric import Fabric

    from protomotions.robot_configs.factory import robot_config as make_robot_config
    from protomotions.simulator.factory import simulator_config as make_simulator_config
    from protomotions.utils.fabric_config import FabricConfig

    fabric = Fabric(**FabricConfig(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]).as_kwargs())
    fabric.launch()
    simulation_app = app_launcher_cls({"headless": headless, "device": str(fabric.device)}).app

    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.scene_lib import SceneLibConfig
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    robot_config = make_robot_config(ROBOT)
    robot_config.update_fields(contact_bodies="all")
    simulator_config = make_simulator_config(simulator="isaaclab", robot_config=robot_config, headless=headless,
                                             num_envs=num_envs, experiment_name=experiment_name)
    terrain_config, simulator_config = convert_friction_for_simulator(TerrainConfig(), simulator_config)
    comps = build_all_components(terrain_config=terrain_config, scene_lib_config=SceneLibConfig(),
                                 motion_lib_config=MotionLibConfig(), simulator_config=simulator_config,
                                 robot_config=robot_config, device=fabric.device, save_dir=None,
                                 simulation_app=simulation_app)
    sim = comps["simulator"]
    sim._initialize_with_markers(None)
    comps["terrain_config"] = terrain_config
    comps["simulator_config"] = simulator_config
    return simulation_app, sim, robot_config, comps


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #
@dataclass
class State:
    """Robot states, COMMON order, env-local root position, xyzw root quaternion, root COM velocities."""
    root_pos: torch.Tensor          # [n, 3]
    root_rot: torch.Tensor          # [n, 4]
    root_vel: torch.Tensor          # [n, 3]
    root_ang_vel: torch.Tensor      # [n, 3]
    dof_pos: torch.Tensor           # [n, 69]
    dof_vel: torch.Tensor           # [n, 69]

    def map(self, fn) -> "State":
        return State(*(fn(getattr(self, f.name)) for f in fields(self)))

    def __getitem__(self, idx) -> "State":
        return self.map(lambda t: t[idx])

    def repeat_interleave(self, n: int) -> "State":
        return self.map(lambda t: t.repeat_interleave(n, 0))

    def clone(self) -> "State":
        return self.map(lambda t: t.clone())

    def numpy(self) -> dict:
        return {f.name: getattr(self, f.name).detach().cpu().double().numpy() for f in fields(self)}

    @staticmethod
    def cat(states: list) -> "State":
        return State(*(torch.cat([getattr(s, f.name) for s in states], 0) for f in fields(State)))

    @staticmethod
    def at_rest(root_pos, root_rot_xyzw, dof_pos, device) -> "State":
        """A state at rest from ``[n, 3]``, ``[n, 4]`` xyzw, ``[n, 69]`` (numpy or torch)."""
        rp = torch.as_tensor(np.asarray(root_pos), dtype=torch.float32, device=device).reshape(-1, 3)
        rr = torch.as_tensor(np.asarray(root_rot_xyzw), dtype=torch.float32, device=device).reshape(-1, 4)
        dp = torch.as_tensor(np.asarray(dof_pos), dtype=torch.float32, device=device).reshape(-1, 69)
        z = torch.zeros_like(rp)
        return State(rp, rr / rr.norm(dim=-1, keepdim=True), z, z.clone(), dp, torch.zeros_like(dp))


def quat_xyzw_to_mat(q: torch.Tensor) -> torch.Tensor:
    """``[..., 4]`` xyzw (unit) -> ``[..., 3, 3]``."""
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    return torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
                        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
                        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], -1).reshape(q.shape[:-1] + (3, 3))


# --------------------------------------------------------------------------- #
# Rollout buffers
# --------------------------------------------------------------------------- #
@dataclass
class Frames:
    """Robot quantities at a sequence of instants: ``[M, T, ...]`` (env-local, COMMON order)."""
    pos: torch.Tensor               # [M, T, 24, 3] body (link) origins
    rot: torch.Tensor               # [M, T, 24, 4] xyzw
    vel: torch.Tensor               # [M, T, 24, 3] body COM linear velocity
    ang_vel: torch.Tensor           # [M, T, 24, 3]
    dof: torch.Tensor               # [M, T, 69]
    dof_vel: torch.Tensor           # [M, T, 69]
    ground: torch.Tensor            # [M, T, 24, 3] terrain contact force on each body (+z = floor pushing up)
    ctrl: torch.Tensor              # [M, T, 69] the PD targets in force over the step that ended at this instant

    @staticmethod
    def empty(m: int, t: int, device) -> "Frames":
        def z(*s):
            return torch.zeros((m, t) + s, device=device)
        return Frames(z(24, 3), z(24, 4), z(24, 3), z(24, 3), z(69), z(69), z(24, 3), z(69))

    def numpy(self) -> dict:
        return {f.name: getattr(self, f.name).detach().cpu().numpy() for f in fields(self)}


# --------------------------------------------------------------------------- #
# The plant
# --------------------------------------------------------------------------- #
class PhysXPlant:
    """Training's PhysX simulator as a batched rollout engine. ``num_envs`` = blocks x samples."""

    def __init__(self, sim, robot_config, spacing: float = GRID_SPACING):
        from extract_contact_configs import ZONE_ORDER, ZONES
        from protomotions.envs.action.action_functions import build_pd_action_offset_scale

        self.sim, self.robot_config = sim, robot_config
        self.robot = sim._robot
        self.device = sim.device
        self.N = sim.num_envs
        self.decimation = int(sim.decimation)
        self.dt_phys = float(sim._sim.get_physics_dt())
        self.dt_ctrl = float(sim.dt)
        ki = robot_config.kinematic_info
        self.body_names = list(ki.body_names)
        self.dof_names = list(ki.dof_names)
        conv = sim.data_conversion
        self.dof_to_sim = conv.dof_convert_to_sim          # sim[:, i] = common[:, dof_to_sim[i]]
        self.dof_to_common = conv.dof_convert_to_common    # common[:, j] = sim[:, dof_to_common[j]]
        self.body_to_common = conv.body_convert_to_common
        dev = self.device
        self.kp = sim._common_p_gains.clone()
        self.kd = sim._common_d_gains.clone()
        self.tau_lim = sim._torque_limits_common.clone()
        self.lower = ki.dof_limits_lower.to(dev).float()
        self.upper = ki.dof_limits_upper.to(dev).float()
        off, sc = build_pd_action_offset_scale(ki.hinge_axes_map, ki.dof_limits_lower, ki.dof_limits_upper, 1.0, dev)
        self.act_lo, self.act_hi = (off - sc).float(), (off + sc).float()   # what the policy's tanh can command
        # gains and limits as PhysX holds them must equal the robot config's (a mismatch = not training's plant)
        kp_sim = self.robot.data.joint_stiffness[0][self.dof_to_common]
        kd_sim = self.robot.data.joint_damping[0][self.dof_to_common]
        ef_sim = self.robot.data.joint_effort_limits[0][self.dof_to_common]
        for name, a, b in (("stiffness", kp_sim, self.kp), ("damping", kd_sim, self.kd), ("effort", ef_sim, self.tau_lim)):
            if not torch.allclose(a, b, rtol=1e-5, atol=1e-4):
                raise RuntimeError(f"PhysX {name} differs from the robot config: {a} vs {b}")
        # env grid
        side = int(math.ceil(math.sqrt(self.N)))
        i = torch.arange(self.N, device=dev)
        self.grid = torch.stack([GRID_ORIGIN[0] + spacing * (i % side), GRID_ORIGIN[1] + spacing * (i // side),
                                 torch.zeros_like(i)], -1).float()
        # contact sensors in COMMON body order (None where a body has none)
        self.sensors = [sim._contact_sensor_map.get(b) for b in self.body_names]
        if any(s is None for s in self.sensors):
            raise RuntimeError("every body needs a contact sensor (robot_config.contact_bodies='all')")
        sim._validate_contact_sensor_filters_once()
        # mass model: the MJCF as MuJoCo compiles it (PhysX's masses, COMs and inertias equal it to 1e-7,
        # data/reference_curation/plant_v2/physx_smpl_yogi_v2.json)
        from edge_synthesis import plant_mj as pm
        m = pm.build_model(sensors=False)
        if [m.body(k).name for k in range(1, m.nbody)] != self.body_names:
            raise RuntimeError("MJCF body order differs from kinematic_info's")
        self.mass = torch.as_tensor(m.body_mass[1:], dtype=torch.float32, device=dev)
        self.com_local = torch.as_tensor(m.body_ipos[1:], dtype=torch.float32, device=dev)
        iq = m.body_iquat[1:]                                                     # wxyz
        Ri = quat_xyzw_to_mat(torch.as_tensor(np.c_[iq[:, 1:], iq[:, :1]], dtype=torch.float32, device=dev))
        I_diag = torch.as_tensor(m.body_inertia[1:], dtype=torch.float32, device=dev)
        self.inertia_local = Ri @ torch.diag_embed(I_diag) @ Ri.transpose(-1, -2)  # [24, 3, 3] body frame
        masses_physx = self.robot.root_physx_view.get_masses()[0].to(dev)[self.body_to_common].float()
        if not torch.allclose(masses_physx, self.mass, rtol=1e-4):
            raise RuntimeError(f"PhysX masses differ from the MJCF: {masses_physx} vs {self.mass}")
        self.total_mass = float(self.mass.sum())
        self.weight_n = self.total_mass * 9.81
        self.zone_order = list(ZONE_ORDER)
        self.zone_bodies = [[self.body_names.index(b) for b in ZONES[z]] for z in ZONE_ORDER]
        zb = torch.zeros(len(ZONE_ORDER), 24, device=dev)
        for zi, bs in enumerate(self.zone_bodies):
            zb[zi, bs] = 1.0
        self.zone_matrix = zb                                                     # [15, 24]
        # every DOF's velocity and effort targets are zero for implicit drives; write them once
        self.robot.set_joint_velocity_target(torch.zeros_like(self.robot.data.joint_vel_target))
        self.robot.set_joint_effort_target(torch.zeros_like(self.robot.data.joint_effort_target))
        self.robot.write_data_to_sim()
        self.all_indices = self.robot._ALL_INDICES
        self.steps = 0

    # ------------------------------------------------------------------ states
    def write(self, state: State, env_ids: torch.Tensor | None = None) -> None:
        """Put ``state`` (one row per env in ``env_ids``) into the simulation, PD targets at its pose."""
        from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion

        env_ids = torch.arange(self.N, device=self.device) if env_ids is None else env_ids
        rs = ResetState(root_pos=state.root_pos.float() + self.grid[env_ids], root_rot=state.root_rot.float(),
                        root_vel=state.root_vel.float(), root_ang_vel=state.root_ang_vel.float(),
                        dof_pos=state.dof_pos.float(), dof_vel=state.dof_vel.float(),
                        state_conversion=StateConversion.COMMON)
        self.sim.reset_envs(rs, env_ids=env_ids)

    def read(self, env_ids: torch.Tensor | None = None) -> State:
        d = self.robot.data
        q = d.root_quat_w                                                          # wxyz
        st = State(d.root_pos_w - self.grid, torch.cat([q[:, 1:], q[:, :1]], -1), d.root_lin_vel_w.clone(),
                   d.root_ang_vel_w.clone(), d.joint_pos[:, self.dof_to_common].clone(),
                   d.joint_vel[:, self.dof_to_common].clone())
        return st if env_ids is None else st[env_ids]

    # ------------------------------------------------------------------ stepping
    def _write_targets(self, targets: torch.Tensor) -> None:
        t_sim = targets[:, self.dof_to_sim].contiguous()
        self.robot._data.joint_pos_target[:] = t_sim                  # keep IsaacLab's buffer coherent
        self.robot.root_physx_view.set_dof_position_targets(t_sim, self.all_indices)

    def _update_sensors(self, dt: float) -> None:
        for s in self.sensors:
            s.update(dt, force_recompute=True)

    def ground_forces(self) -> torch.Tensor:
        """``[N, 24, 3]`` terrain force on each body at the last physics step (COMMON order)."""
        return torch.stack([s.data.force_matrix_w[:, 0, 0, :] for s in self.sensors], 1)

    def _snapshot(self, out: Frames, k: int, rows: torch.Tensor | slice, targets: torch.Tensor, ground: bool) -> None:
        d = self.robot.data
        b = self.body_to_common
        out.pos[:, k] = (d.body_pos_w[:, b] - self.grid[:, None])[rows]
        qw = d.body_quat_w[:, b]
        out.rot[:, k] = torch.cat([qw[..., 1:], qw[..., :1]], -1)[rows]
        out.vel[:, k] = d.body_lin_vel_w[:, b][rows]
        out.ang_vel[:, k] = d.body_ang_vel_w[:, b][rows]
        out.dof[:, k] = d.joint_pos[:, self.dof_to_common][rows]
        out.dof_vel[:, k] = d.joint_vel[:, self.dof_to_common][rows]
        out.ctrl[:, k] = targets[rows]
        if ground:
            out.ground[:, k] = self.ground_forces()[rows]

    def control_step(self, targets: torch.Tensor, sub: Frames | None = None, sub_k: int = 0,
                     sub_rows: torch.Tensor | slice = slice(None)) -> None:
        """One 30 Hz control step: ``targets [N, 69]`` (COMMON) held over ``decimation`` physics substeps.
        With ``sub``, every substep of ``sub_rows`` is recorded into ``sub`` from index ``sub_k`` (sensors
        refreshed each substep); without, the sensors are refreshed once, after the last substep."""
        self._write_targets(targets)
        for k in range(self.decimation):
            self.sim._sim.step(render=False)
            self.robot.update(self.dt_phys)
            if sub is not None:
                self._update_sensors(self.dt_phys)
                self._snapshot(sub, sub_k + k, sub_rows, targets, ground=True)
        if sub is None:
            self._update_sensors(self.dt_ctrl)
        self.steps += 1

    def rollout(self, start: State, ctrl: torch.Tensor, sub_rows: torch.Tensor | None = None) -> tuple:
        """Every env from ``start`` (``[N]`` rows, or ``[B]`` rows repeated over B equal blocks), then
        ``ctrl [N, H, 69]`` PD targets (clamped to the policy's range). Returns ``(frames, sub)``: the state at each
        control-step boundary ``[N, H, ...]``, and -- for ``sub_rows`` -- every physics substep
        ``[len(sub_rows), H * decimation, ...]``."""
        n, h = ctrl.shape[:2]
        if n != self.N:
            raise ValueError(f"ctrl has {n} rows for {self.N} envs")
        if start.root_pos.shape[0] != self.N:
            start = start.repeat_interleave(self.N // start.root_pos.shape[0])
        self.write(start)
        ctrl = torch.maximum(torch.minimum(ctrl, self.act_hi), self.act_lo)
        frames = Frames.empty(self.N, h, self.device)
        sub = None if sub_rows is None else Frames.empty(len(sub_rows), h * self.decimation, self.device)
        for k in range(h):
            self.control_step(ctrl[:, k], sub, k * self.decimation, slice(None) if sub_rows is None else sub_rows)
            self._snapshot(frames, k, slice(None), ctrl[:, k], ground=True)   # sensors hold the last substep
        return frames, sub

    # ------------------------------------------------------------------ derived quantities
    def mass_quantities(self, f: Frames) -> dict:
        """``com [.., 3]``, ``com_vel [.., 3]``, ``ang_mom [.., 3]`` (about the COM), ``R [.., 24, 3, 3]``."""
        R = quat_xyzw_to_mat(f.rot)
        c = f.pos + (R @ self.com_local[:, :, None])[..., 0]                    # body COMs
        m = self.mass[:, None]
        com = (m * c).sum(-2) / self.total_mass
        com_vel = (m * f.vel).sum(-2) / self.total_mass
        rc = c - com[..., None, :]
        rv = f.vel - com_vel[..., None, :]
        L = (m * torch.cross(rc, rv, dim=-1)).sum(-2)
        Iw = R @ self.inertia_local @ R.transpose(-1, -2)
        L = L + (Iw @ f.ang_vel[..., None])[..., 0].sum(-2)
        return {"com": com, "com_vel": com_vel, "ang_mom": L, "R": R}

    def zone_forces(self, ground: torch.Tensor) -> torch.Tensor:
        """``[.., 24, 3]`` per-body ground forces -> ``[.., 15, 3]`` per zone (``ZONE_ORDER``)."""
        return torch.einsum("zb,...bk->...zk", self.zone_matrix, ground)

    def pd_torque(self, ctrl: torch.Tensor, dof: torch.Tensor, dof_vel: torch.Tensor, clip: bool = True) -> torch.Tensor:
        """The implicit drive's torque, as IsaacLab estimates it (``ImplicitActuator.compute``)."""
        tau = self.kp * (ctrl - dof) - self.kd * dof_vel
        return torch.maximum(torch.minimum(tau, self.tau_lim), -self.tau_lim) if clip else tau

    def box_excess(self, dof: torch.Tensor) -> torch.Tensor:
        """``[.., 69]`` rad past PhysX's exp-map box, each joint's better representative (``plant_mj.box_excess``)."""
        v = dof.reshape(dof.shape[:-1] + (23, 3))
        n = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        alt = v - 2 * math.pi * v / n
        lo, hi = self.lower.reshape(23, 3), self.upper.reshape(23, 3)

        def exc(x):
            return (lo - x).clamp(min=0) + (x - hi).clamp(min=0)
        ev, ea = exc(v), exc(alt)
        pick = (ea.sum(-1) < ev.sum(-1))[..., None]
        return torch.where(pick, ea, ev).reshape(dof.shape)
