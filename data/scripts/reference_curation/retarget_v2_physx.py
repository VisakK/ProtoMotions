# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Step 3 motions in PhysX (BodyFix Step 3, the check carried in from Step 2): training loads them on plant
v2, and the bodies they store are the bodies PhysX puts there.

Built exactly as training builds it (robot config -> ``build_all_components`` -> IsaacLab, headless, no policy):

1. **The library loads on ``smpl_yogi_v2``.** ``build_all_components`` builds a MotionLib from ``--motion-dir``
   (every ``.motion`` of the retarget) and its plant check (``require_motion_plant``) passes. The negative control:
   the same check on the shipped ftC library (no identity: plant v1) raises ``PlantMismatchError``.
2. **Stored ``rigid_body_pos`` = PhysX FK.** MotionLib serves the stored bodies, not FK, so a writer whose FK
   differed from PhysX's would hand the policy targets its body cannot take. Every clip's hold exemplars (labels
   v1.1) and ``--per-clip`` evenly spaced frames go through training's reset (``Simulator.reset_envs`` with the
   stored root and ``dof_pos``); PhysX's link poses (no step) must equal the stored ones to 1e-5 m (Step 2
   measured the writer's FK at 1.4e-6 m on 418 frames of the unedited fit).
3. **No launch at a reset.** The same frames, with the training scripts' 5 mm ``ref_respawn_offset`` and the PD
   targets at the pose, stepped for ``--duration`` s: Step 2's launch rule, the root overshooting the height that
   just clears the floor by more than 1 cm within 0.5 s, or rising faster than 0.25 m/s -- the latter only within
   the first ``EARLY_S`` (a depenetration launch is immediate: Step 2's thrown root moves up from the first step).
   Step 2 applied the rule to two-foot standing holds; on a single-leg pose the passive plant falls and bounces:
   Tree's 8 flags under the full window were the root moving up at 0.33-0.86 m/s at 0.47 s, after dropping
   83-85 cm (her unedited fit: 7 of 26 frames, up to 1.63 m/s).

Writes ``data/reference_curation/retarget_v2/<retarget_id>/physx.json`` (the retarget's record folder) and exits
non-zero on any failure. Run it as its own process (IsaacLab launches before torch)::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python data/scripts/reference_curation/retarget_v2_physx.py \\
        --motion-dir output/reference_curation/retarget_v2/<retarget_id>
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--motion-dir", required=True)
parser.add_argument("--robot", default="smpl_yogi_v2")
parser.add_argument("--per-clip", type=int, default=8, help="evenly spaced frames per clip, beside the exemplars")
parser.add_argument("--duration", type=float, default=0.5, help="reset test length (s)")
parser.add_argument("--out", default=None)
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCHEMA_VERSION = 1
MODULE = "reference_curation.retarget_v2_physx"
FK_TOL_M = 1e-5
RESPAWN_M = 0.005
WINDOW_S = 0.5
LAUNCH_RISE_M = 0.01
LAUNCH_VZ = 0.25
EARLY_S = 0.1
THROW_VZ = 1.0          # the positive control: the first N_CONTROL frames again, the root thrown up at this speed
N_CONTROL = 3


def _quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between unit quaternions (xyzw), ``2 atan2(|v|, |w|)`` of ``conj(a) b`` (Step 2's trap: ``2 acos``
    reads float32 rounding as ~0.05 deg)."""
    a, b = a.double(), b.double()
    w = (a * b).sum(-1)
    v = a[..., 3:4] * b[..., :3] - b[..., 3:4] * a[..., :3] - torch.cross(a[..., :3], b[..., :3], dim=-1)
    return 2 * torch.atan2(v.norm(dim=-1), w.abs())


def _reset(sim, root_pos, root_rot_xyzw, dof_pos, root_vel=None):
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion

    n, dev = root_pos.shape[0], sim.device
    z3 = torch.zeros(n, 3, device=dev)
    root_vel = z3.clone() if root_vel is None else root_vel.float().to(dev)
    state = ResetState(root_pos=root_pos.float().to(dev), root_rot=root_rot_xyzw.float().to(dev), root_vel=root_vel,
                       root_ang_vel=z3.clone(), dof_pos=dof_pos.float().to(dev),
                       dof_vel=torch.zeros_like(dof_pos, dtype=torch.float32, device=dev),
                       state_conversion=StateConversion.COMMON)
    sim.reset_envs(state, env_ids=torch.arange(n, device=dev))


def frames_of(motion_dir: Path, per_clip: int) -> dict:
    """Every clip's labels-v1.1 hold exemplars plus ``per_clip`` evenly spaced frames: the stored root, root
    quaternion, ``dof_pos`` and bodies."""
    from reference_curation import retarget_v2 as r2

    labels = r2.load_labels()
    holds = {stem: [int(h["frame_hold"]) for h in hs] for stem, hs in labels["clips"]}
    rows, pos, rot, dof = [], [], [], []
    for path in sorted(Path(motion_dir).glob("*.motion")):
        m = torch.load(path, map_location="cpu", weights_only=False)
        T = m["dof_pos"].shape[0]
        ex = [f for f in holds.get(path.stem, []) if f < T]
        fr = sorted(set(ex) | {int(round(k * (T - 1) / max(per_clip - 1, 1))) for k in range(per_clip)})
        for f in fr:
            rows.append({"stem": path.stem, "frame": f, "exemplar": f in ex})
        idx = torch.as_tensor(fr)
        pos.append(m["rigid_body_pos"][idx])
        rot.append(m["rigid_body_rot"][idx])
        dof.append(m["dof_pos"][idx])
    pos, rot, dof = torch.cat(pos), torch.cat(rot), torch.cat(dof)
    control = np.r_[np.zeros(len(rows), bool), np.ones(N_CONTROL, bool)]          # thrown copies of the first frames
    rows += [{**rows[i], "control": "thrown"} for i in range(N_CONTROL)]
    return {"rows": rows, "pos": torch.cat([pos, pos[:N_CONTROL]]), "rot": torch.cat([rot, rot[:N_CONTROL]]),
            "dof": torch.cat([dof, dof[:N_CONTROL]]), "control": control}


def fk_check(sim, frames: dict, names: list) -> dict:
    _reset(sim, frames["pos"][:, 0], frames["rot"][:, 0], frames["dof"])
    rs = sim.get_robot_state()
    got_p, got_q = rs.rigid_body_pos.double().cpu(), rs.rigid_body_rot.double().cpu()
    e = (got_p - frames["pos"].double()).norm(dim=-1)
    a = _quat_angle(got_q, frames["rot"])
    back = (rs.dof_pos.double().cpu() - frames["dof"].double()).abs()
    w = np.unravel_index(int(e.argmax()), tuple(e.shape))
    return {"frames": int((~torch.as_tensor(frames["control"])).sum()), "exemplars": sum(r["exemplar"] for r in frames["rows"] if "control" not in r),
            "clips": len({r["stem"] for r in frames["rows"]}),
            "pos_max_err_m": float(e.max()), "pos_p99_err_m": float(e.flatten().quantile(0.99)),
            "rot_max_err_deg": float(np.degrees(a.max())), "dof_readback_max_err_rad": float(back.max()),
            "worst": {**frames["rows"][w[0]], "body": names[w[1]]}}


def reset_test(sim, frames: dict, duration: float) -> dict:
    """Step 2's launch rule on every frame, reset at the stored pose + ``RESPAWN_M`` with the PD targets at it."""
    from reference_curation import fit_writer as fw

    n = sim.num_envs
    low0 = fw.body_lowest(frames["pos"].double().numpy(), frames["rot"].double().numpy()).min(1)
    ctrl = torch.as_tensor(frames["control"])
    root = frames["pos"][:, 0].clone()
    root[:, 2] += RESPAWN_M
    vel = torch.zeros(n, 3)
    vel[torch.as_tensor(frames["control"]), 2] = THROW_VZ
    _reset(sim, root, frames["rot"][:, 0], frames["dof"], vel)
    z0 = sim.get_robot_state().rigid_body_pos[:, 0, 2].double().cpu()
    dt = sim.dt if hasattr(sim, "dt") else sim._sim.get_physics_dt() * sim.decimation
    steps, win = int(round(duration / dt)), int(round(WINDOW_S / dt))
    targets = frames["dof"].float().to(sim.device)
    z, vz = [], []
    for _ in range(steps):
        sim.step(targets)
        st = sim.get_robot_state()
        z.append(st.rigid_body_pos[:, 0, 2].double().cpu())
        vz.append(st.rigid_body_vel[:, 0, 2].double().cpu())
    z, vz = torch.stack(z, 1), torch.stack(vz, 1)
    owed = torch.as_tensor(np.maximum(0.0, -(low0 + RESPAWN_M)))
    overshoot = (z[:, :win].max(1).values - z0 - owed).clamp(min=0)
    early = max(1, int(round(EARLY_S / dt)))
    vz_up = vz[:, :early].max(1).values.clamp(min=0)
    launch = (overshoot > LAUNCH_RISE_M) | (vz_up > LAUNCH_VZ)
    k = vz[:, :win].argmax(1)                                        # the step of the fastest upward root speed
    drop_before = torch.stack([z0[i] - z[i, :int(k[i]) + 1].min() for i in range(n)])
    bad = [{**frames["rows"][i], "overshoot_cm": round(100 * float(overshoot[i]), 3), "vz_up_m_s": round(float(vz_up[i]), 3),
            "at_s": round(float((k[i] + 1) * dt), 3), "root_drop_before_cm": round(100 * float(drop_before[i]), 2)}
           for i in np.nonzero((launch & ~ctrl).numpy())[0]]
    controls = {"replicas": int(ctrl.sum()), "seen_as_launch": int((launch & ctrl).sum()), "throw_m_s": THROW_VZ}
    launch = launch & ~ctrl
    late = vz[:, :win].max(1).values.clamp(min=0)
    return {"frames": int((~ctrl).sum()), "launched": int(launch.sum()), "launched_rows": bad[:20], "positive_control": controls,
            "thrown": int((overshoot > LAUNCH_RISE_M).sum()), "early_s": EARLY_S,
            "overshoot_max_cm": round(100 * float(overshoot[~ctrl].max()), 3),
            "root_vz_up_max_m_s": round(float(vz_up[~ctrl].max()), 4),
            "root_vz_up_max_window_m_s": round(float(late[~ctrl].max()), 4),
            "root_drop_at_window_end_max_cm": round(100 * float((z0 - z[:, win - 1])[~ctrl].max()), 2),
            "lowest_collider_at_reset_cm": {"min": round(100 * float(low0[~frames["control"]].min()), 3),
                                            "p50": round(100 * float(np.median(low0[~frames["control"]])), 3)}}


def failures(rec: dict) -> list[str]:
    bad = []
    if not rec["library"]["loaded"]:
        bad.append(f"the library did not load: {rec['library']['error']}")
    if not rec["library"]["v1_refused"]:
        bad.append("a plant v1 library was not refused")
    if rec["fk"]["pos_max_err_m"] > FK_TOL_M:
        bad.append(f"FK {rec['fk']['pos_max_err_m']:.2e} m at {rec['fk']['worst']}")
    if rec["reset"]["launched"]:
        bad.append(f"{rec['reset']['launched']} resets launched: {rec['reset']['launched_rows'][:5]}")
    c = rec["reset"]["positive_control"]
    if c["seen_as_launch"] != c["replicas"]:
        bad.append(f"positive control: a root thrown up at {THROW_VZ} m/s was not seen as a launch: {c}")
    return bad


def main() -> int:
    from lightning.fabric import Fabric

    from protomotions.robot_configs.factory import robot_config as make_robot_config
    from protomotions.simulator.factory import simulator_config as make_simulator_config
    from protomotions.utils.fabric_config import FabricConfig

    t0 = time.time()
    fabric = Fabric(**FabricConfig(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]).as_kwargs())
    fabric.launch()
    simulation_app = AppLauncher({"headless": True, "device": str(fabric.device)}).app

    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.scene_lib import SceneLibConfig
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils import plant_identity
    from protomotions.utils.component_builder import build_all_components, build_motion_lib_from_config, require_motion_plant
    from reference_curation import ids

    motion_dir = Path(args.motion_dir)
    motion_dir = motion_dir if motion_dir.is_absolute() else ids.REPO / motion_dir
    robot_config = make_robot_config(args.robot)
    frames = frames_of(motion_dir, args.per_clip)
    n = len(frames["rows"])
    simulator_config = make_simulator_config(simulator="isaaclab", robot_config=robot_config, headless=True,
                                             num_envs=n, experiment_name="retarget_v2_physx")
    terrain_config, simulator_config = convert_friction_for_simulator(TerrainConfig(), simulator_config)
    library = {"motion_dir": ids.display_path(motion_dir), "loaded": False, "error": None}
    mcfg = MotionLibConfig(motion_file=str(motion_dir))
    try:
        comps = build_all_components(terrain_config=terrain_config, scene_lib_config=SceneLibConfig(),
                                     motion_lib_config=mcfg, simulator_config=simulator_config,
                                     robot_config=robot_config, device=fabric.device, save_dir=None,
                                     simulation_app=simulation_app)
        ml = comps["motion_lib"]
        library.update(loaded=True, motions=int(ml.num_motions()), plant_sha256=ml.plant_sha256,
                       plant=plant_identity.name_of(ml.plant_sha256))
    except Exception as exc:  # noqa: BLE001
        library["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED library: {library['error']}", flush=True)
        os._exit(1)
    try:                                   # the negative control: the shipped (plant v1) library on plant v2
        v1cfg = MotionLibConfig(motion_file=str(ids.SHIPPED_DIR / "220923_Crane_Crow_Pose_or_Bakasana_-a.motion"))
        require_motion_plant(build_motion_lib_from_config(v1cfg, fabric.device), v1cfg, robot_config)
        library["v1_refused"] = False
    except plant_identity.PlantMismatchError as exc:
        library["v1_refused"] = True
        library["v1_error"] = str(exc)[:200]
    sim = comps["simulator"]
    sim._initialize_with_markers(None)
    names = list(robot_config.kinematic_info.body_names)
    rec = {"robot": args.robot, "plant": plant_identity.identity(plant_identity.robot_mjcf(robot_config)),
           "num_envs": n, "library": library, "fk": fk_check(sim, frames, names),
           "reset": reset_test(sim, frames, args.duration)}
    rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, sorted(motion_dir.glob("*.motion"))), **rec,
           "seconds": round(time.time() - t0, 1)}
    rec["failures"] = failures(rec)
    out = Path(args.out) if args.out else ids.DATA_ROOT / "retarget_v2" / motion_dir.name / "physx.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    fk, rs = rec["fk"], rec["reset"]
    print(f"retarget_v2 physx: library {library.get('motions')} motions on {library.get('plant')} (v1 refused: "
          f"{library['v1_refused']}); FK {fk['frames']} frames of {fk['clips']} clips <= {fk['pos_max_err_m']:.1e} m, "
          f"rot <= {fk['rot_max_err_deg']:.1e} deg; positive control {rs['positive_control']['seen_as_launch']}/"
          f"{rs['positive_control']['replicas']} seen; resets launched {rs['launched']}/{rs['frames']} (overshoot <= "
          f"{rs['overshoot_max_cm']} cm, vz <= {rs['root_vz_up_max_m_s']} m/s) -> {ids.display_path(out)}"
          + (f"; FAILED: {rec['failures']}" if rec["failures"] else ""), flush=True)
    os._exit(1 if rec["failures"] else 0)   # simulation_app.close() can hang (the known IsaacLab exit hang)


if __name__ == "__main__":
    sys.exit(main())
