"""Sampling MPC on one edge of ``edges.json`` (cards T3 and T4): S's exemplar -> D's, executed on plant v2 in MuJoCo.

The run: S's exemplar as the initial state (zero velocity), ``settle`` seconds holding S, the edge at a chosen
timing (``sketch.timing``), then ``hold`` seconds holding D. The planner tracks the hand-anchored sketch between
the endpoints (plus any intermediate keyframes) under ``costs.EdgeCost``. The executed motion is saved at the
physics rate with its PD targets and sensors; ``evaluate`` reports what T3's acceptance and T5's gate read:

* ``final_d6_m``: the 6-body pose error to D (pelvis-relative, best-fit yaw; ``census.py``'s metric) at the end,
  and its mean over the hold;
* ``com_speed_end``: the COM speed at the end;
* ``landing_peak_bw``: per zone, the peak normal load over the run in body weights;
* ``hand_drift_cm``: how far the planted hands moved horizontally from S's;
* ``free_violations``: control steps on which a known-free zone came within 2 cm of the floor or carried load
  outside its event windows;
* ``torque_saturated_share``: per joint, the share of physics steps at >= 98 % of the torque limit;
* ``box_excess_max_deg``, ``speed_over_p99``, ``acc_over_100``: the contract's kinematic checks;
* ``held``: the root within 10 cm of D's height and the pelvis within 30 deg of D's orientation over the hold.

CLI::

    PYTHONPATH=.:data/scripts python -m edge_synthesis.edge_mppi --edge B1 [--timing mid] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from extract_contact_configs import ZONE_ORDER
from edge_synthesis import costs as C
from edge_synthesis import gpu_guard
from edge_synthesis import mppi as M
from edge_synthesis import plant_mj as pm
from edge_synthesis import sketch as SK
from reference_curation import ids

OUT_ROOT = ids.REPO / "output/edge_synthesis/runs"
GOAL6 = ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")


def best_yaw_d6(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean distance of the 6 goal bodies (``[.., 6, 3]`` pelvis-relative) after the best yaw of ``a`` onto ``b``."""
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    th = np.arctan2(num, den)[..., None]
    c, s = np.cos(th), np.sin(th)
    x = c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] + c * a[..., 1]
    rot = np.stack([x, y, np.broadcast_to(a[..., 2], x.shape)], -1)
    return np.linalg.norm(rot - b, axis=-1).mean(-1)


def evaluate(plant: pm.Plant, arr: dict, sched: SK.Schedule, cost: C.EdgeCost, ep: SK.Endpoints, t0: float,
             hold: float) -> dict:
    sub = plant.sub
    qpos = arr["qpos"][sub - 1::sub]                         # control-step boundaries
    t = t0 + (np.arange(len(qpos)) + 1) / pm.CTRL_HZ
    pos, rot = plant.fk(qpos)
    bi = plant.body_index
    g6 = [bi[b] for b in GOAL6]
    rel = pos[:, g6] - pos[:, :1]
    dst_rel = ep.dst_pos[g6] - ep.dst_pos[:1]
    d6 = best_yaw_d6(rel, dst_rel[None])
    in_hold = t >= sched.T + hold - 2.0 + 1e-9
    sens = arr["sens"]
    f = plant.zone_forces(sens)                               # [steps, 15, 3]
    com_vel = plant.com_vel(sens[sub - 1::sub])
    low, _ = C.zone_lowest(cost.sk, pos, rot)
    m = sched.masks(t)
    fz = f[sub - 1::sub, :, 2]
    viol = (m["free"] & ((low < C.FREE_M) | (fz > C.LOAD_N)))
    # torque saturation at the physics rate
    tau = plant.kp * (arr["ctrl"] - arr["qpos"][:, 7:]) - plant.kd * arr["qvel"][:, 6:]
    sat = (np.abs(tau) >= 0.98 * plant.torque_limit).mean(0)
    _, _, dof = plant.expmap_from_qpos(qpos)
    box = np.degrees(plant.box_excess(dof))
    # kinematics at 60 fps (the export rate)
    p60, _ = plant.fk(arr["qpos"][3::4])
    v60 = np.linalg.norm(np.diff(p60, axis=0), axis=-1) * 60.0
    a60 = np.linalg.norm(np.diff(p60, 2, axis=0), axis=-1) * 3600.0
    v99 = C.corpus_speed_p99()
    hands = [bi[b] for b in ("L_Hand", "R_Hand", "L_Wrist", "R_Wrist")]
    drift = np.linalg.norm(pos[:, hands, :2] - ep.src_pos[hands, :2], axis=-1)
    edge_win = (t >= 0) & (t <= sched.T + 1.0)
    up = rot[:, 0, :, 2]
    dst_up = ep.dst_rot[0]
    from scipy.spatial.transform import Rotation
    dst_up = Rotation.from_quat(dst_up).as_matrix()[:, 2]
    tilt = np.degrees(np.arccos(np.clip(up @ dst_up, -1, 1)))
    hold_win = t >= sched.T
    held = bool(hold_win.any() and (np.abs(qpos[hold_win, 2] - ep.dst_qpos[2]) < 0.10).all() and
                (tilt[hold_win] < 30).all())
    return {
        "final_d6_m": round(float(d6[-1]), 4), "hold_d6_mean_m": round(float(d6[in_hold].mean()), 4) if in_hold.any() else None,
        "com_speed_end": round(float(np.linalg.norm(com_vel[-1])), 4),
        "landing_peak_bw": {z: round(float(f[:, i, 2].max() / plant.weight_n), 3) for i, z in enumerate(ZONE_ORDER)
                            if f[:, i, 2].max() > C.LOAD_N},
        "hand_drift_cm": round(100 * float(drift[edge_win].max()), 2) if edge_win.any() else None,
        "free_violations": {z: int(viol[:, i].sum()) for i, z in enumerate(ZONE_ORDER) if viol[:, i].any()},
        "torque_saturated_share": {plant.joint_names[j]: round(float(sat[j]), 3) for j in np.argsort(-sat)[:5] if sat[j] > 0},
        "box_excess_max_deg": round(float(box.max()), 2),
        "box_worst": plant.dof_names[int(np.unravel_index(box.argmax(), box.shape)[1])],
        "speed_over_p99": {plant.body_names[b]: round(float(v60[:, b].max() / v99[b]), 2) for b in range(24)
                           if v60[:, b].max() > v99[b]},
        "acc_over_100_frames": int((a60 > 100).any(-1).sum()),
        "held": held, "tilt_max_hold_deg": round(float(tilt[hold_win].max()), 2) if hold_win.any() else None,
        "d6_track": [round(float(x), 4) for x in d6[:: max(1, len(d6) // 40)]],
    }


FF_STEP_S = 0.1          # feed-forward offsets every this many seconds along the sketch
FF_SETTLE_S = 0.06       # a stiffened statue settles this long before its actuator torques are read
FF_GAIN = 10.0           # the statue's stiffness multiple (BUILD_PLAN Step 7: a statue buckles at training gains)
FF_MAX_RAD = 0.5


class FeedForward:
    """Gravity feed-forward for the PD targets: at every ``FF_STEP_S`` along the sketch, the sketch's pose as a
    stiffened statue (gains x``FF_GAIN``, damping x sqrt, torque limits unchanged) settles ``FF_SETTLE_S`` from rest;
    its actuator torques tau become target offsets ``tau / kp`` at the training gains, so a training-gain servo
    aimed at ``pose + offset`` produces the torque the pose needs. MPPI then samples around ``sketch + offset``
    instead of discovering the droop correction itself (the first press execution lagged 13 cm and toppled)."""

    def __init__(self, plant: pm.Plant, sk, t0: float, t1: float):
        stiff = pm.Plant(nthread=1, gain_scale=FF_GAIN)
        self.t = np.arange(t0, t1 + 1e-9, FF_STEP_S)
        off = np.zeros((len(self.t), plant.nu))
        import mujoco
        n = int(round(FF_SETTLE_S / plant.dt))
        for i, ti in enumerate(self.t):
            q = sk.qpos(np.array([ti]))[0]
            s0, d = stiff.make_state(q)
            for _ in range(n):
                mujoco.mj_step(stiff.model, d)
            tau = d.actuator_force.copy()
            off[i] = np.clip(tau / plant.kp, -FF_MAX_RAD, FF_MAX_RAD)
        self.off = off

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, float)
        out = np.empty(t.shape + (self.off.shape[1],))
        for j in range(self.off.shape[1]):
            out[..., j] = np.interp(t, self.t, self.off[:, j])
        return out


def via_qpos(plant: pm.Plant, ep: SK.Endpoints, hold_id: str) -> np.ndarray:
    """The exemplar of ``hold_id`` (an endpoint of some edge in edges.json) re-anchored on S's hands."""
    for e in SK.load_edges()["edges"]:
        for k in ("source", "destination"):
            if e[k]["hold_id"] == hold_id:
                vp, vr, _ = SK.exemplar(e[k])
                yaw, t = SK.hand_anchor_transform(ep.src_pos, vp, plant.body_index)
                vp2, vr2 = SK.apply_planar(vp, vr, yaw, t)
                return plant.qpos_from_bodies(vp2, vr2)
    raise KeyError(f"{hold_id} is not an endpoint in edges.json")


def run_edge(edge_id: str, timing: str | list = "mid", seed: int = 0, settle: float = 0.3, hold: float = 2.0,
             cfg: M.MPPIConfig | None = None, weights: C.EdgeWeights | None = None, keyframes=None,
             nthread: int = gpu_guard.POLITE_THREADS, out_dir: Path | None = None, verbose: bool = False,
             reference: Path | None = None, via: str | None = None, feedforward: bool = False) -> dict:
    spec = SK.load_edges()
    e = SK.edge(spec, edge_id)
    plant = pm.Plant(nthread=nthread)
    ep = SK.endpoints(plant, e)
    if reference is not None:
        # T2's dynamic check: track a certified quasi-static reference; its own timing, start and (IK-edited) end
        rrec = json.load(open(Path(reference) / "record.json"))
        timing = rrec["durations_s"]
    durs = SK.timing(e, timing)
    sched = SK.schedule(e, durs)
    if reference is not None:
        sk = SK.RefSketch(plant, Path(reference), sched.T)
        ep.src_qpos = sk.qpos(np.array([-settle]))[0]
        ep.dst_qpos = sk.qpos(np.array([sched.T + hold]))[0]
        ep.src_pos, ep.dst_pos = plant.fk(ep.src_qpos[None])[0][0], plant.fk(ep.dst_qpos[None])[0][0]
        from scipy.spatial.transform import Rotation
        ep.dst_rot = Rotation.from_matrix(plant.fk(ep.dst_qpos[None])[1][0]).as_quat()
        keyframes = None
    if keyframes == "land":
        # dynamic edges: the sketch reaches D at the first touchdown and holds it through the landing phase --
        # or, with ``via``, touches down in another hold's exemplar (re-anchored on S's hands) and moves to D after
        makes = sorted({te for te, _, kind in sched.events if kind == "make" and te > 0})
        land_q = ep.dst_qpos if via is None else via_qpos(plant, ep, via)
        keyframes = [(makes[0], land_q)] if makes and makes[0] < sched.T else []
    if reference is None:
        kfs = [(0.0, ep.src_qpos)] + list(keyframes or []) + [(sched.T, ep.dst_qpos)]
        sk = SK.Sketch(plant, kfs)
        sk.build_grid(t0=-settle - 1.0, t1=sched.T + hold + 2.0)
    cost = C.EdgeCost(plant, sk, sched, ep.src_qpos, ep.dst_qpos, weights)
    cfg = cfg or M.MPPIConfig(seed=seed)
    s0, _ = plant.make_state(ep.src_qpos)
    base_fn = sk.pd_targets
    if feedforward:
        ff = FeedForward(plant, sk, -settle - 0.2, sched.T + hold + 1.0)
        base_fn = lambda t: sk.pd_targets(t) + ff(t)          # noqa: E731
    ctl = M.MPPI(plant, cost, cfg, base_fn=base_fn)
    tic = time.perf_counter()
    traj = ctl.run(s0, settle + sched.T + hold, t0=-settle, verbose=verbose)
    wall = time.perf_counter() - tic
    arr = traj.arrays()
    metrics = evaluate(plant, arr, sched, cost, ep, -settle, hold)
    # mean body error to the tracked sketch / reference over the edge (T2's execution check reads this)
    qb = arr["qpos"][plant.sub - 1::plant.sub]
    tb = -settle + (np.arange(len(qb)) + 1) / pm.CTRL_HZ
    win = (tb >= 0) & (tb <= sched.T)
    pb, _ = plant.fk(qb[win])
    pr, _ = plant.fk(sk.qpos(tb[win]))
    err = np.linalg.norm(pb - pr, axis=-1)
    metrics["sketch_body_err_mean_cm"] = round(100 * float(err.mean()), 2)
    metrics["sketch_body_err_p95_cm"] = round(100 * float(np.percentile(err.mean(-1), 95)), 2)
    metrics["sketch_body_err_max_cm"] = round(100 * float(err.max()), 2)
    import edge_synthesis
    rec = {"provenance": edge_synthesis.provenance(), "edge": edge_id, "label": e["label"], "timing": timing,
           "durations_s": [round(d, 3) for d in durs],
           "via": via, "reference": None if reference is None else ids.display_path(reference),
           "feedforward": feedforward,
           "T": round(sched.T, 3), "seed": seed, "settle_s": settle, "hold_s": hold, "wall_s": round(wall, 1),
           "mppi": asdict(cfg), "cost": cost.describe(), "hand_residual_m": ep.hand_residual_m,
           "anchor_yaw_deg": round(float(np.degrees(ep.yaw)), 2), "metrics": metrics,
           "log_tail": traj.log[-3:], "plant": {"dt": plant.dt, "cone": plant.cone, "gain_scale": plant.gain_scale}}
    plant.close()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "trajectory.npz", qpos=arr["qpos"], qvel=arr["qvel"], ctrl=arr["ctrl"],
                            sens=arr["sens"], t0=-settle, dt=plant.dt)
        (out_dir / "run.json").write_text(json.dumps({**rec, "log": traj.log}, indent=1, default=float) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edge", required=True)
    ap.add_argument("--timing", default="mid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--interp", default="linear")
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=gpu_guard.POLITE_THREADS)
    ap.add_argument("--weights", default="{}", help="JSON overrides of costs.EdgeWeights")
    ap.add_argument("--tag", default="")
    ap.add_argument("--keyframes", default="none", choices=("none", "land"))
    ap.add_argument("--reference", type=Path, default=None, help="a quasistatic reference dir to track (T2 check)")
    ap.add_argument("--via", default=None, help="touch down in this hold's exemplar (with --keyframes land)")
    ap.add_argument("--feedforward", action="store_true", help="statue gravity feed-forward on the PD targets")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    print("cpu policy", gpu_guard.be_polite(), flush=True)
    cfg = M.MPPIConfig(horizon=args.horizon, samples=args.samples, iters=args.iters, noise=args.noise, seed=args.seed,
                       interp=args.interp)
    w = C.EdgeWeights(**json.loads(args.weights))
    timing_name = Path(args.reference).name if args.reference is not None else args.timing   # a reference has its own
    name = f"{args.edge}_{timing_name}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    rec = run_edge(args.edge, args.timing, args.seed, hold=args.hold, cfg=cfg, weights=w, nthread=args.threads,
                   keyframes=None if args.keyframes == "none" else args.keyframes, out_dir=OUT_ROOT / name,
                   verbose=args.verbose, reference=args.reference, via=args.via, feedforward=args.feedforward)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("log_tail", "cost", "mppi")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
