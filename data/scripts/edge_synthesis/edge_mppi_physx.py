"""Sampling MPC on one edge of ``edges.json``, executed in IsaacLab/PhysX -- the expert's training simulator.

The PhysX counterpart of ``edge_mppi.py`` (cards T3/T4 on the MuJoCo plant): the same endpoints, timing, contact
schedule, sketch (``--keyframes land``, ``--via``) or T2 reference (``--reference``), the same cost terms and
weights (``costs_torch.EdgeCostTorch`` evaluates ``costs.EdgeCost``'s), the same MPPI settings -- but the rollouts
and the executed motion are PhysX's (``physx_plant``): its exp-map joints and drives, hard box, contacts and
torque limits. Several seeds run at once, one block of envs each (``mppi_torch``).

What does not carry over from the MuJoCo runs, and why:

* ``--feedforward`` (lane T's statue gravity feed-forward) has no PhysX counterpart: a stiffened PhysX statue goes
  unstable instead of holding (``physx_plant``'s module doc), so its torques are not holding torques. ``--warmup``
  replaces it: MPPI holds S for that long *before* the settle (with ``--warmup-iters`` updates per replan), so the
  offsets that hold S on this plant are found by the planner itself and carried into the edge; the warm-up is not
  part of the recorded motion.
* Every rollout starts from a restored PhysX state. A rerun with the same seeds *and the same number of seeds
  and samples* (the env grid) reproduces the run bit for bit; changing the block layout moves every env and gives
  a new draw.

Output per seed: ``output/edge_synthesis/physx/runs/<run>/{trajectory.npz, run.json}`` (the executed motion at the
physics rate, 120 Hz; ``run.json`` has ``edge_mppi.evaluate``'s metrics, computed the same way). The MuJoCo runs in
``output/edge_synthesis/runs/`` are left as they are.

CLI (GPU; one IsaacLab process -- at most two on the A5000, none beside an expert training run)::

    PYTHONPATH=.:data/scripts python -m edge_synthesis.edge_mppi_physx --edge B1 --keyframes land \\
        --seeds 0 1 2 --samples 512 --noise 0.08
    PYTHONPATH=.:data/scripts python -m edge_synthesis.edge_mppi_physx --hold-test crow --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--edge", default=None)
parser.add_argument("--timing", default="mid")
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
parser.add_argument("--samples", type=int, default=256)
parser.add_argument("--noise", type=float, default=0.05)
parser.add_argument("--horizon", type=int, default=24)
parser.add_argument("--knots", type=int, default=4)
parser.add_argument("--iters", type=int, default=2)
parser.add_argument("--lam", type=float, default=0.05)
parser.add_argument("--replan", type=int, default=3)
parser.add_argument("--interp", default="linear")
parser.add_argument("--settle", type=float, default=0.3)
parser.add_argument("--hold", type=float, default=2.0)
parser.add_argument("--warmup", type=float, default=0.0, help="s of MPPI holding S before the settle (not recorded)")
parser.add_argument("--warmup-iters", type=int, default=4)
parser.add_argument("--start-iters", type=int, default=0,
                    help="MPPI updates of the first plan from the start state before anything executes")
parser.add_argument("--start-sigma", type=float, default=3.0, help="noise multiple the start updates anneal from")
parser.add_argument("--weights", default="{}", help="JSON overrides of costs.EdgeWeights")
parser.add_argument("--tag", default="px")
parser.add_argument("--keyframes", default="none", choices=("none", "land"))
parser.add_argument("--reference", default=None, help="a quasistatic reference dir to track (T2's dynamic check)")
parser.add_argument("--via", default=None, help="touch down in this hold's exemplar (with --keyframes land)")
parser.add_argument("--start", default="auto", choices=("auto", "exemplar", "reference"),
                    help="initial state: S's exemplar, or the reference's first frame (lane T's choice; auto)")
parser.add_argument("--hold-test", default=None, choices=("crow", "handstand", "chaturanga", "plank", "tripod"),
                    help="T1's regression in PhysX: hold the pose 5 s (no edge)")
parser.add_argument("--hold-cost", default="auto", choices=("auto", "benchmark", "pose"),
                    help="benchmark: mppi.hold_cost (crow, handstand); pose: EdgeCost's terminal term (default for the rest)")
parser.add_argument("--hold-seconds", type=float, default=5.0)
parser.add_argument("--hold-weights", default="{}", help="JSON overrides of the hold cost's weights")
parser.add_argument("--out-root", default=None)
parser.add_argument("--print-log", action="store_true", help="print every replan (not --verbose: Kit reads that from argv)")
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

GOAL6 = ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")


def out_root() -> Path:
    from reference_curation import ids

    return Path(args.out_root) if args.out_root else ids.REPO / "output/edge_synthesis/physx/runs"


# --------------------------------------------------------------------------- #
# The sketch in PhysX's coordinates
# --------------------------------------------------------------------------- #
def expmap_grid(pm_plant, qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, dict]:
    """Hinge ``qpos [G, 76]`` (a sketch on the control grid) -> exp-map dof ``[G, 69]``: the box's representative of
    each joint (``retarget.nearest_representative``), then continuity along the grid (a joint whose representative
    flips by ~2 pi between neighbours takes the other one). Reports the largest step between neighbours."""
    from reference_curation import retarget as rt

    _, _, dof = pm_plant.expmap_from_qpos(qpos)
    dof = rt.nearest_representative(torch.as_tensor(dof), torch.as_tensor(lower), torch.as_tensor(upper)).numpy()
    v = dof.reshape(len(dof), 23, 3)
    flips = 0
    for i in range(1, len(v)):
        n = np.linalg.norm(v[i], axis=-1, keepdims=True)
        alt = v[i] - 2 * np.pi * v[i] / np.maximum(n, 1e-9)
        use = np.linalg.norm(alt - v[i - 1], axis=-1) < np.linalg.norm(v[i] - v[i - 1], axis=-1) - 1.0
        if use.any():
            v[i][use] = alt[use]
            flips += int(use.sum())
    dof = v.reshape(len(v), 69)
    step = np.abs(np.diff(dof, axis=0)).max() if len(dof) > 1 else 0.0
    return dof, {"representative_flips": flips, "max_step_rad": round(float(step), 4)}


def release_state(stem: str, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(root_pos, root_quat_xyzw, dof)`` of a release v2 frame (the stored float32, exactly)."""
    from edge_synthesis import plant_mj as pm

    m = pm.load_motion(stem)
    return (m["rigid_body_pos"][frame, 0].double().numpy(), m["rigid_body_rot"][frame, 0].double().numpy(),
            m["dof_pos"][frame].double().numpy())


# --------------------------------------------------------------------------- #
# Evaluation (edge_mppi.evaluate's metrics, on the PhysX motion)
# --------------------------------------------------------------------------- #
def evaluate(fr: dict, t: np.ndarray, info: dict) -> dict:
    """``fr``: one block's executed frames at the physics rate (numpy ``[T, ...]``), ``t [T]`` their times."""
    from scipy.spatial.transform import Rotation

    from edge_synthesis import costs as C
    from edge_synthesis import edge_mppi as EM
    from extract_contact_configs import ZONE_ORDER

    sched, ep, npc, dec = info["sched"], info["ep"], info["np_cost"], info["decimation"]
    names = info["body_names"]
    bi = {n: i for i, n in enumerate(names)}
    b_idx = np.arange(dec - 1, len(t), dec)                    # control-step boundaries
    tb = t[b_idx]
    pos = fr["pos"][b_idx].astype(np.float64)
    R = Rotation.from_quat(fr["rot"][b_idx].reshape(-1, 4)).as_matrix().reshape(len(b_idx), 24, 3, 3)
    g6 = [bi[b] for b in GOAL6]
    rel = pos[:, g6] - pos[:, :1]
    dst_rel = ep.dst_pos[g6] - ep.dst_pos[:1]
    d6 = EM.best_yaw_d6(rel, dst_rel[None])
    in_hold = tb >= sched.T + info["hold"] - 2.0 + 1e-9
    zf = np.einsum("zb,tbk->tzk", info["zone_matrix"], fr["ground"].astype(np.float64))     # [T, 15, 3] 120 Hz
    com_vel = info["com_vel_b"]
    low = C.zone_lowest(npc.sk, pos, R)[0]
    m = sched.masks(tb)
    fz_b = zf[b_idx, :, 2]
    viol = m["free"] & ((low < C.FREE_M) | (fz_b > C.LOAD_N))
    tau = info["kp"] * (fr["ctrl"] - fr["dof"]) - info["kd"] * fr["dof_vel"]
    sat = (np.abs(tau) >= 0.98 * info["tau_lim"]).mean(0)
    box = np.degrees(info["box_b"])
    p60 = fr["pos"][1::2].astype(np.float64)                   # 120 Hz -> 60 fps (the export rate)
    v60 = np.linalg.norm(np.diff(p60, axis=0), axis=-1) * 60.0
    a60 = np.linalg.norm(np.diff(p60, 2, axis=0), axis=-1) * 3600.0
    v99 = C.corpus_speed_p99()
    hands = [bi[b] for b in ("L_Hand", "R_Hand", "L_Wrist", "R_Wrist")]
    drift = np.linalg.norm(pos[:, hands, :2] - ep.src_pos[hands, :2], axis=-1)
    edge_win = (tb >= 0) & (tb <= sched.T + 1.0)
    up = R[:, 0, :, 2]
    dst_up = Rotation.from_quat(ep.dst_rot[0]).as_matrix()[:, 2]
    tilt = np.degrees(np.arccos(np.clip(up @ dst_up, -1, 1)))
    hold_win = tb >= sched.T
    held = bool(hold_win.any() and (np.abs(pos[hold_win, 0, 2] - info["dst_root_z"]) < 0.10).all()
                and (tilt[hold_win] < 30).all())
    dof_names = info["dof_names"]
    out = {
        "final_d6_m": round(float(d6[-1]), 4),
        "hold_d6_mean_m": round(float(d6[in_hold].mean()), 4) if in_hold.any() else None,
        "com_speed_end": round(float(np.linalg.norm(com_vel[-1])), 4),
        "landing_peak_bw": {z: round(float(zf[:, i, 2].max() / info["weight_n"]), 3) for i, z in enumerate(ZONE_ORDER)
                            if zf[:, i, 2].max() > C.LOAD_N},
        "hand_drift_cm": round(100 * float(drift[edge_win].max()), 2) if edge_win.any() else None,
        "free_violations": {z: int(viol[:, i].sum()) for i, z in enumerate(ZONE_ORDER) if viol[:, i].any()},
        "torque_saturated_share": {dof_names[j]: round(float(sat[j]), 3) for j in np.argsort(-sat)[:5] if sat[j] > 0},
        "box_excess_max_deg": round(float(box.max()), 2),
        "box_worst": dof_names[int(np.unravel_index(box.argmax(), box.shape)[1])],
        "speed_over_p99": {names[b]: round(float(v60[:, b].max() / v99[b]), 2) for b in range(24)
                           if v60[:, b].max() > v99[b]},
        "acc_over_100_frames": int((a60 > 100).any(-1).sum()),
        "held": held, "tilt_max_hold_deg": round(float(tilt[hold_win].max()), 2) if hold_win.any() else None,
        "d6_track": [round(float(x), 4) for x in d6[:: max(1, len(d6) // 40)]],
    }
    win = (tb >= 0) & (tb <= sched.T)
    if win.any():
        err = np.linalg.norm(pos[win] - info["ref_pos_b"][win], axis=-1)
        out["sketch_body_err_mean_cm"] = round(100 * float(err.mean()), 2)
        out["sketch_body_err_p95_cm"] = round(100 * float(np.percentile(err.mean(-1), 95)), 2)
        out["sketch_body_err_max_cm"] = round(100 * float(err.max()), 2)
    return out


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def build_problem(pm_plant, plant):
    """Lane T's edge set-up (``edge_mppi.run_edge``), unchanged, on the CPU MuJoCo plant (kinematics only)."""
    from edge_synthesis import edge_mppi as EM
    from edge_synthesis import sketch as SK

    spec = SK.load_edges()
    e = SK.edge(spec, args.edge)
    ep = SK.endpoints(pm_plant, e)
    reference = Path(args.reference) if args.reference else None
    timing = args.timing
    if reference is not None:
        rrec = json.load(open(reference / "record.json"))
        timing = rrec["durations_s"]
    durs = SK.timing(e, timing)
    sched = SK.schedule(e, durs)
    keyframes = None if args.keyframes == "none" else args.keyframes
    if reference is not None:
        sk = SK.RefSketch(pm_plant, reference, sched.T)
        ep.src_qpos = sk.qpos(np.array([-args.settle]))[0]
        ep.dst_qpos = sk.qpos(np.array([sched.T + args.hold]))[0]
        ep.src_pos, ep.dst_pos = pm_plant.fk(ep.src_qpos[None])[0][0], pm_plant.fk(ep.dst_qpos[None])[0][0]
        from scipy.spatial.transform import Rotation
        ep.dst_rot = Rotation.from_matrix(pm_plant.fk(ep.dst_qpos[None])[1][0]).as_quat()
        keyframes = None
    if keyframes == "land":
        makes = sorted({te for te, _, kind in sched.events if kind == "make" and te > 0})
        land_q = ep.dst_qpos if args.via is None else EM.via_qpos(pm_plant, ep, args.via)
        keyframes = [(makes[0], land_q)] if makes and makes[0] < sched.T else []
    if reference is None:
        kfs = [(0.0, ep.src_qpos)] + list(keyframes or []) + [(sched.T, ep.dst_qpos)]
        sk = SK.Sketch(pm_plant, kfs)
        sk.build_grid(t0=-args.settle - args.warmup - 1.0, t1=sched.T + args.hold + 2.0)
    return e, ep, durs, sched, sk, reference


def start_state(e: dict, ep, reference, sk, pm_plant, device):
    """S's exemplar (exact release values) or the reference's first frame (lane T's start for ``--reference``)."""
    from scipy.spatial.transform import Rotation

    from edge_synthesis.physx_plant import State

    mode = args.start if args.start != "auto" else ("reference" if reference is not None else "exemplar")
    if mode == "exemplar":
        rp, rq, dof = release_state(e["source"]["stem"], int(e["source"]["frame_hold"]))
        src = "release exemplar"
    else:
        z = np.load(Path(reference) / "reference.npz")
        k = int(np.argmin(np.abs(z["times"] - (-args.settle))))
        rp, rq, dof = z["root_pos"][k], Rotation.from_matrix(z["root_rot"][k]).as_quat(), z["dof"][k]
        src = f"reference frame {k} (t={float(z['times'][k]):.3f})"
    return State.at_rest(rp, rq, dof, device), mode, src


def main() -> int:
    from edge_synthesis import costs as C
    from edge_synthesis import costs_torch as CT
    from edge_synthesis import mppi_torch as MT
    from edge_synthesis import physx_plant as PP
    from edge_synthesis import plant_mj as pm
    from reference_curation import ids
    import edge_synthesis

    seeds = list(args.seeds)
    n_envs = len(seeds) * args.samples
    t_launch = time.time()
    app, sim, robot_config, comps = PP.launch(AppLauncher, n_envs, experiment_name="edge_mppi_physx")
    plant = PP.PhysXPlant(sim, robot_config)
    dev = plant.device
    launch_s = time.time() - t_launch
    pm_plant = pm.Plant(nthread=1, sensors=True)
    if pm_plant.body_names != plant.body_names:
        raise RuntimeError("MuJoCo and PhysX body orders differ")
    lower, upper = plant.lower.cpu().double().numpy(), plant.upper.cpu().double().numpy()
    cfg = MT.MPPIConfig(horizon=args.horizon, knots=args.knots, samples=args.samples, iters=args.iters, lam=args.lam,
                        noise=args.noise, replan=args.replan, interp=args.interp, seed=seeds[0])
    if args.hold_test:
        return hold_test(plant, pm_plant, cfg, seeds)
    e, ep, durs, sched, sk, reference = build_problem(pm_plant, plant)
    dt = plant.dt_ctrl
    n_steps = int(round((args.settle + sched.T + args.hold) / dt))
    n_steps = int(math.ceil(n_steps / args.replan) * args.replan)
    n_warm = int(math.ceil(round(args.warmup / dt) / args.replan) * args.replan)
    t_start = -args.settle - n_warm * dt
    grid = CT.Grid(t_start, dt, n_warm + n_steps + args.horizon + 2)
    q_grid = sk.qpos(grid.t)
    base_np, rep = expmap_grid(pm_plant, q_grid, lower, upper)
    base = torch.as_tensor(base_np, dtype=torch.float32, device=dev)
    weights = C.EdgeWeights(**json.loads(args.weights))
    cost = CT.EdgeCostTorch(pm_plant, sk, sched, ep.src_qpos, ep.dst_qpos, weights, grid, dev, args.horizon)

    def cost_fn(frames, j):
        view, _ = MT.make_view(plant, frames)
        return cost(view, j)

    s0, start_mode, start_src = start_state(e, ep, reference, sk, pm_plant, dev)
    state0 = PP.State.cat([s0] * len(seeds))
    ctl = MT.MPPITorch(plant, cost_fn, cfg, base, seeds)
    dst_up = torch.as_tensor(PP.quat_xyzw_to_mat(torch.as_tensor(ep.dst_rot[0], dtype=torch.float32))[:, 2],
                             device=dev)

    def monitor(state, b):
        R = PP.quat_xyzw_to_mat(state.root_rot[b])
        return {"root_z": round(float(state.root_pos[b, 2]), 4),
                "tilt_to_D_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(R[:, 2] @ dst_up))))), 2)}

    tic = time.perf_counter()
    start_log = ctl.optimise_start(state0, 0, args.start_iters, args.start_sigma) if args.start_iters else []
    if n_warm:
        # the warm-up: hold S (the base plan before the settle is S) with more updates per replan; not recorded
        iters = cfg.iters
        ctl.cfg.iters = args.warmup_iters
        warm = ctl.run(state0, n_warm, j0=0, monitor=monitor, verbose=args.print_log)
        ctl.cfg.iters = iters
        state0 = plant.read(ctl.first)
        warm_log = warm.log
    else:
        warm_log = []
    ex = ctl.run(state0, n_steps, j0=n_warm, monitor=monitor, verbose=args.print_log)
    wall = time.perf_counter() - tic
    frames = ex.frames()                                         # {k: [B, T, ...]} at the physics rate
    T_sub = frames["pos"].shape[1]
    t_sub = -args.settle + plant.dt_phys * (np.arange(T_sub) + 1)
    # derived quantities at the control-step boundaries, per block
    dec = plant.decimation
    b_idx = np.arange(dec - 1, T_sub, dec)
    ref_pos_b = pm_plant.fk(sk.qpos(t_sub[b_idx]))[0]
    common = {"sched": sched, "ep": ep, "np_cost": cost.np_cost, "decimation": dec, "body_names": plant.body_names,
              "dof_names": plant.dof_names, "hold": args.hold, "zone_matrix": plant.zone_matrix.cpu().numpy(),
              "kp": plant.kp.cpu().numpy(), "kd": plant.kd.cpu().numpy(), "tau_lim": plant.tau_lim.cpu().numpy(),
              "weight_n": plant.weight_n, "dst_root_z": float(ep.dst_qpos[2]), "ref_pos_b": ref_pos_b}
    out_dir_root = out_root()
    timing_name = Path(args.reference).name if args.reference else args.timing
    recs = []
    for b, seed in enumerate(seeds):
        fb = {k: v[b] for k, v in frames.items()}
        fr_t = PP.Frames(**{k: torch.as_tensor(v[:, b_idx], device=dev) for k, v in
                            {k: frames[k][b:b + 1] for k in frames}.items()})
        mq = plant.mass_quantities(fr_t)
        info = {**common, "com_vel_b": mq["com_vel"][0].cpu().double().numpy(),
                "box_b": plant.box_excess(fr_t.dof)[0].cpu().double().numpy()}
        metrics = evaluate(fb, t_sub, info)
        name = f"{args.edge}_{timing_name}_s{seed}" + (f"_{args.tag}" if args.tag else "")
        rec = {"provenance": edge_synthesis.provenance(), "backend": "physx", "edge": args.edge, "label": e["label"],
               "timing": timing_name if args.reference else args.timing, "durations_s": [round(d, 3) for d in durs],
               "via": args.via, "reference": None if reference is None else ids.display_path(reference),
               "feedforward": False, "warmup_s": round(n_warm * dt, 3), "warmup_iters": args.warmup_iters,
               "start_iters": args.start_iters, "start_sigma": args.start_sigma,
               "start": {"mode": start_mode, "source": start_src},
               "T": round(sched.T, 3), "seed": seed, "settle_s": args.settle, "hold_s": args.hold,
               "wall_s": round(wall, 1), "launch_s": round(launch_s, 1), "blocks": len(seeds),
               "mppi": asdict(cfg), "cost": cost.describe(), "hand_residual_m": ep.hand_residual_m,
               "anchor_yaw_deg": round(float(np.degrees(ep.yaw)), 2), "metrics": metrics,
               "base_plan": rep,
               "plant": {"simulator": "isaaclab", "robot": PP.ROBOT, "dt_phys": plant.dt_phys,
                         "decimation": dec, "num_envs": plant.N, "samples_per_block": args.samples,
                         "terrain_friction": [comps["terrain_config"].sim_config.static_friction,
                                              comps["terrain_config"].sim_config.dynamic_friction],
                         "physx": {k: getattr(comps["simulator_config"].sim.physx, k)
                                   for k in ("solver_type", "num_position_iterations", "num_velocity_iterations",
                                             "contact_offset", "rest_offset", "bounce_threshold_velocity")}},
               "exec_spread_mm_max": max(r["blocks"][b]["exec_spread_mm"] for r in ex.log),
               "log_tail": [r["blocks"][b] for r in ex.log[-3:]]}
        d = out_dir_root / name
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "trajectory.npz", **fb, t0=-args.settle, dt=plant.dt_phys, t=t_sub,
                            decimation=dec, body_names=np.array(plant.body_names),
                            dof_names=np.array(plant.dof_names), kp=common["kp"], kd=common["kd"],
                            tau_lim=common["tau_lim"], zone_matrix=common["zone_matrix"],
                            weight_n=plant.weight_n)
        log_b = [{"j": r["j"], "plan_s": r["plan_s"], **r["blocks"][b]} for r in ex.log]
        warm_b = [{"j": r["j"], **r["blocks"][b]} for r in warm_log]
        start_b = [{"k": r["k"], "sigma_scale": r["sigma_scale"], "cost_min": r["cost_min"][b],
                    "cost_nominal": r["cost_nominal"][b]} for r in start_log]
        (d / "run.json").write_text(json.dumps({**rec, "log": log_b, "warmup_log": warm_b, "start_log": start_b}, indent=1,
                                               default=float) + "\n")
        recs.append((name, metrics))
        print(json.dumps({"run": name, "wall_s": rec["wall_s"], **{k: v for k, v in metrics.items() if k != "d6_track"}},
                         default=float), flush=True)
    print("DONE", flush=True)
    os._exit(0)


def hold_test(plant, pm_plant, cfg, seeds) -> int:
    """T1's MPPI hold regression (``mppi.hold``), in PhysX: hold crow / the handstand for ``--hold-seconds``."""
    from edge_synthesis import costs_torch as CT
    from edge_synthesis import mppi as M
    from edge_synthesis import mppi_torch as MT
    from edge_synthesis import physx_plant as PP
    from edge_synthesis import plant_mj as pm

    poses = {**M.HOLD_POSES, "chaturanga": ("220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a", 485),
             "plank": ("220923_Plank_Pose_or_Kumbhakasana_-a", 541),
             "tripod": ("220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b", 1093)}
    stem, frame = poses[args.hold_test]
    rp, rq, dof = release_state(stem, frame)
    dev = plant.device
    s0 = PP.State.at_rest(rp, rq, dof, dev)
    qref = torch.as_tensor(dof, dtype=torch.float32, device=dev)
    upref = PP.quat_xyzw_to_mat(torch.as_tensor(rq, dtype=torch.float32, device=dev))[:, 2]
    zref = float(rp[2])
    kind = args.hold_cost if args.hold_cost != "auto" else ("benchmark" if args.hold_test in M.HOLD_POSES else "pose")
    if kind == "benchmark":
        hc = CT.HoldCostTorch(plant.body_names, qref, zref, upref, json.loads(args.hold_weights))
    else:
        m = pm.load_motion(stem)
        pos_ref = torch.as_tensor(m["rigid_body_pos"][frame].numpy(), dtype=torch.float32, device=dev)
        hc = CT.PoseHoldCostTorch(pos_ref, upref, zref)
        hc.w_report = {"terminal": hc.w.terminal, "torque": hc.w.torque, "box": hc.w.box}
    steps = int(math.ceil(round(args.hold_seconds / plant.dt_ctrl) / cfg.replan) * cfg.replan)
    base = qref[None].expand(steps + cfg.horizon + 2, -1)
    watch = [plant.body_names.index(b) for b in CT.HoldCostTorch.WATCH]

    def cost_fn(frames, j):
        view, _ = MT.make_view(plant, frames)
        return hc(view, j, frames.dof)

    def monitor(state, b):
        R = PP.quat_xyzw_to_mat(state.root_rot[b])
        return {"root_z": round(float(state.root_pos[b, 2]), 4),
                "tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(R[:, 2] @ upref))))), 2)}

    ctl = MT.MPPITorch(plant, cost_fn, cfg, base, seeds)
    tic = time.perf_counter()
    state0 = PP.State.cat([s0] * len(seeds))
    start_log = ctl.optimise_start(state0, 0, args.start_iters, args.start_sigma) if args.start_iters else []
    ex = ctl.run(state0, steps, monitor=monitor, verbose=args.print_log)
    wall = time.perf_counter() - tic
    fr = ex.frames()
    res = []
    for b, seed in enumerate(seeds):
        z = np.array([r["blocks"][b]["root_z"] for r in ex.log])
        tilt = np.array([r["blocks"][b]["tilt_deg"] for r in ex.log])
        watch_min = float(fr["pos"][b][:, watch, 2].min())
        held = bool((z > zref - 0.10).all() and tilt.max() < 30 and (watch_min > 0.06 or kind != "benchmark"))
        res.append({"seed": seed, "held": held, "root_z_drop_max_m": round(zref - float(z.min()), 4),
                    "tilt_max_deg": round(float(tilt.max()), 2), "watch_min_z": round(watch_min, 4),
                    "exec_spread_mm_max": max(r["blocks"][b]["exec_spread_mm"] for r in ex.log)})
    out = {"pose": args.hold_test, "seconds": args.hold_seconds, "wall_s": round(wall, 1),
           "plan_s_per_0.1s": round(float(np.mean([r["plan_s"] for r in ex.log])) * 0.1 / (cfg.replan * plant.dt_ctrl), 3),
           "mppi": asdict(cfg), "start_iters": args.start_iters, "start_sigma": args.start_sigma, "hold_cost": kind,
           "hold_weights": hc.w if kind == "benchmark" else hc.w_report,
           "num_envs": plant.N, "held": sum(r["held"] for r in res), "of": len(res), "runs": res,
           "start_cost_first_last": [start_log[0]["cost_min"], start_log[-1]["cost_min"]] if start_log else None,
           "root_z_trace": [[r["blocks"][b]["root_z"] for r in ex.log[::5]] for b in range(len(seeds))]}
    print("HOLD_TEST", json.dumps(out), flush=True)
    d = out_root().parent / "hold_tests"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.hold_test}_{args.tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    t_sub = plant.dt_phys * (np.arange(fr["pos"].shape[1]) + 1)
    np.savez_compressed(d / f"{args.hold_test}_{args.tag}.npz", **fr, t=t_sub, seeds=np.array(seeds),
                        kp=plant.kp.cpu().numpy(), kd=plant.kd.cpu().numpy(), tau_lim=plant.tau_lim.cpu().numpy(),
                        zone_matrix=plant.zone_matrix.cpu().numpy(), weight_n=plant.weight_n,
                        dof_names=np.array(plant.dof_names), body_names=np.array(plant.body_names),
                        log=np.array(json.dumps(ex.log)))
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
