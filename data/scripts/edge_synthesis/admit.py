"""Card T5 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: the admission gate of synthesised edge clips.

Every check is pre-registered in the card and machine-only; this module computes each number on an exported
variant (``export_motion``: the ``.motion``, its JSON, the MuJoCo run it came from) and writes them all to
``expert_revist/graph_growth_2026_10_03/admitted.json``.

==================  ===================================================================================================
group               check (the card's)
==================  ===================================================================================================
contract            0 frames more than 1 deg past the exp-map box; floor >= -0.5 cm; 0 new body-pair overlaps > 1 mm
                    against the endpoints (closed braces may touch); no body acceleration > 100 m/s^2 outside a
                    declared impact window (``IMPACT_S`` around each executed touchdown)
statics             s* <= 1 on the quasi-static frames (the export projection's certification)
dynamics            on the MuJoCo execution: torque utilisation >= 0.98 on <= 5 % of physics steps per joint;
                    friction inside the cone (per zone, tangential / normal load <= 0.75 + ``CONE_TOL``)
naturalness         per-body peak speed and jerk <= the corpus p99 of that body (x0 release motions, 60 fps);
                    E5's frozen critic -- **pending** until card E5 exists
endpoints           final 6-body error to D <= 0.05 m; COM speed <= 0.05 m/s at the end; braces closed (<= 1 cm)
                    at a crow end
physx               stored ``rigid_body_pos`` = FK of the stored coordinates to 1e-5 m; the clip loads into a
                    ``MotionLib`` whose plant is smpl_yogi_v2's (CPU); 0 launches at reset in PhysX -- **pending**
                    (IsaacLab needs the GPU, which G1 holds)
==================  ===================================================================================================

A variant is ``admitted`` when every computed check passes, and stays ``provisional`` while a pending check is
pending: the gate never assumes a check it could not run. The advisory VLM packet (Pass-C style) is not run here.

CLI::

    PYTHONPATH=.:data/scripts python -m edge_synthesis.admit output/edge_synthesis/motions/SYN_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from extract_contact_configs import ZONE_ORDER
from edge_synthesis import costs as C
from edge_synthesis import gpu_guard
from edge_synthesis import plant_mj as pm
from edge_synthesis import sketch as SK
from reference_curation import ids

OUT = ids.REPO / "expert_revist/graph_growth_2026_10_03/admitted.json"
BOX_DEG = 1.0
FLOOR_CM = -0.5
OVERLAP_M = 0.001
ACC_MAX = 100.0
IMPACT_S = 0.1
TORQUE_UTIL = 0.98
TORQUE_SHARE = 0.05
MU = 0.75
CONE_TOL = 0.05
D6_MAX = 0.05
COM_SPEED_MAX = 0.05
BRACE_CM = 1.0


def _corpus():
    d = json.load(open(C.CORPUS_KIN))
    return np.asarray(d["speed60"]["p99"]), np.asarray(d["jerk60"]["p99"]), np.asarray(d["speed60"]["max"]), \
        np.asarray(d["jerk60"]["max"])


def execution(runrec: dict, z, plant) -> dict:
    """The executed motion's per-step series the dynamics checks read: ``t [T]``, zone forces ``[T, 15, 3]``,
    PD torques ``[T, nu]`` with their limits and joint names. MuJoCo runs: the physics steps of ``plant_mj`` (its
    sensors, its hinge torques). PhysX runs (``edge_mppi_physx``): the physics substeps PhysX recorded (its
    terrain-filtered body forces summed per zone, its exp-map drive torques ``kp (target - q) - kd q'``)."""
    if runrec.get("backend") == "physx":
        f = np.einsum("zb,tbk->tzk", z["zone_matrix"], z["ground"].astype(np.float64))
        tau = z["kp"] * (z["ctrl"] - z["dof"]) - z["kd"] * z["dof_vel"]
        return {"t": z["t"], "zone_forces": f, "tau": tau, "tau_lim": z["tau_lim"],
                "joint_names": [str(n) for n in z["dof_names"]], "weight_n": float(z["weight_n"])}
    sens = z["sens"]
    tz = float(z["t0"]) + float(z["dt"]) * (np.arange(len(sens)) + 1)
    tau = plant.kp * (z["ctrl"] - z["qpos"][:, 7:]) - plant.kd * z["qvel"][:, 6:]
    return {"t": tz, "zone_forces": plant.zone_forces(sens), "tau": tau, "tau_lim": plant.torque_limit,
            "joint_names": plant.joint_names, "weight_n": plant.weight_n}


def check_variant(js: Path) -> dict:
    from reference_curation import fit_writer as fw
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2
    from edge_synthesis import edge_mppi as EM
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig
    from protomotions.utils import plant_identity

    rec = json.load(open(js))
    mpath = ids.REPO / rec["motion"]
    mot = torch.load(mpath, map_location="cpu", weights_only=False)
    run = ids.REPO / rec["source_run"]
    runrec = json.load(open(run / "run.json"))
    z = np.load(run / "trajectory.npz")
    e = SK.edge(SK.load_edges(), rec["edge"])
    sched = SK.schedule(e, runrec["durations_s"])
    plant = pm.Plant(nthread=1)
    out = {"variant": rec["name"], "edge": rec["edge"], "backend": runrec.get("backend", "mujoco"),
           "motion": rec["motion"], "sha256": ids.sha256_file(mpath),
           "source_run": rec["source_run"], "durations_s": runrec["durations_s"], "via": runrec.get("via"),
           "reference": runrec.get("reference")}
    fps = int(mot["fps"])
    pos = mot["rigid_body_pos"].double().numpy()
    dof = mot["dof_pos"].double().numpy()
    T = pos.shape[0]
    t = rec["t_first"] + np.arange(T) / fps
    # ---------------- contract
    with rv2.on_plant("v2") as sk:
        exc = np.degrees(np.maximum(sk.lower.numpy() - dof, 0) + np.maximum(dof - sk.upper.numpy(), 0))
        from protomotions.utils.rotations import quaternion_to_matrix
        rot = quaternion_to_matrix(mot["rigid_body_rot"].double(), w_last=True)
        P = torch.as_tensor(pos)
        hgt = rt.candidate_heights(sk, rt.candidate_points(sk, P, rot)).numpy()
        nf, na, nb, gap = rt.near_body_pairs(sk, P, rot, -OVERLAP_M - 1e-4)
        ends = set()
        for f in (0, T - 1):
            _, ea, eb, _ = rt.near_body_pairs(sk, P[f:f + 1], rot[f:f + 1], -OVERLAP_M - 1e-4)
            ends |= {(int(a), int(b)) for a, b in zip(ea, eb)}
        from extract_contact_configs import ZONES
        zone_of = {sk.names.index(b): zz for zz, bs in ZONES.items() for b in bs}
        braces = [sched.config_at(float(x))[1] for x in t]
        new = [(int(f), sk.names[a], sk.names[b], round(100 * float(g), 2)) for f, a, b, g in zip(nf, na, nb, gap)
               if (int(a), int(b)) not in ends and f"{zone_of[int(a)]}+{zone_of[int(b)]}" not in braces[int(f)]
               and f"{zone_of[int(b)]}+{zone_of[int(a)]}" not in braces[int(f)]]
        acc = np.zeros(pos.shape[:2])
        acc[1:-1] = np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1) * fps ** 2
        # impact windows: around every executed touchdown (a zone's ground state switching on in the run)
        exe = execution(runrec, z, plant)
        f_all = exe["zone_forces"][..., 2]
        tz = exe["t"]
        on = f_all > C.LOAD_N
        touch = [tz[i] for zi in range(len(ZONE_ORDER)) for i in np.nonzero(on[1:, zi] & ~on[:-1, zi])[0] + 1]
        impact = np.zeros(T, bool)
        for tt in touch:
            impact |= np.abs(t - tt) <= IMPACT_S
        spikes = (acc > ACC_MAX).any(1) & ~impact
        brace_gap = {}
        if rec["edge"] in ("E3", "E4", "B2"):
            for br in sorted(sched.dst_braces):
                g_, _, _ = rt.zone_pair_gaps(sk, P, rot, *br.split("+"), np.arange(T - 10, T))
                brace_gap[br] = round(100 * float(g_.max()), 2)
    contract = {"box_frames_past_1deg": int((exc > BOX_DEG).any(1).sum()), "box_max_deg": round(float(exc.max()), 2),
                "floor_min_cm": round(100 * float(hgt.min()), 2), "new_overlaps_1mm": len(new),
                "new_overlap_examples": new[:5], "acc_over_100_outside_impacts": int(spikes.sum()),
                "acc_max": round(float(acc.max()), 1), "impact_windows_s": [round(float(x), 3) for x in touch]}
    contract["pass"] = (contract["box_frames_past_1deg"] == 0 and contract["floor_min_cm"] >= FLOOR_CM
                        and contract["new_overlaps_1mm"] == 0 and contract["acc_over_100_outside_impacts"] == 0)
    # ---------------- statics (the projection's certification)
    cert = (rec.get("projection") or {}).get("certification", {})
    st = cert.get("statics", {})
    statics = {"checked": st.get("checked"), "s_star_max": st.get("s_star_max"),
               "beyond": len(st.get("beyond", [])), "not_optimal": len(st.get("not_optimal", [])),
               "not_optimal_examples": st.get("not_optimal", [])[:3]}
    statics["pass"] = bool(st) and statics["beyond"] == 0 and statics["not_optimal"] == 0
    # ---------------- dynamics (the MuJoCo execution)
    sat = (np.abs(exe["tau"]) >= TORQUE_UTIL * exe["tau_lim"]).mean(0)
    fz = exe["zone_forces"]
    ratio = np.linalg.norm(fz[..., :2], axis=-1) / np.maximum(fz[..., 2], 1e-9)
    loaded = fz[..., 2] > 50.0
    cone = {zz: round(float(np.percentile(ratio[loaded[:, i], i], 99)), 3) for i, zz in enumerate(ZONE_ORDER)
            if loaded[:, i].sum() > 10}
    dynamics = {"torque_saturated_share_max": round(float(sat.max()), 3),
                "torque_saturated_joints": {exe["joint_names"][j]: round(float(sat[j]), 3)
                                            for j in np.nonzero(sat > TORQUE_SHARE)[0]},
                "friction_ratio_p99": cone}
    dynamics["pass"] = not dynamics["torque_saturated_joints"] and all(v <= MU + CONE_TOL for v in cone.values())
    # ---------------- naturalness
    v99, j99, vmax, jmax = _corpus()
    v = np.linalg.norm(np.diff(pos, axis=0), axis=-1) * fps
    j = np.linalg.norm(np.diff(pos, 3, axis=0), axis=-1) * fps ** 3
    names = list(mot.get("body_names", [])) or pm.Plant(nthread=1, sensors=False).body_names
    nat = {"speed_over_p99": {names[b]: round(float(v[:, b].max() / v99[b]), 2) for b in range(24) if v[:, b].max() > v99[b]},
           "jerk_over_p99": {names[b]: round(float(j[:, b].max() / j99[b]), 2) for b in range(24) if j[:, b].max() > j99[b]},
           "speed_over_corpus_max": {names[b]: round(float(v[:, b].max() / vmax[b]), 2) for b in range(24)
                                     if v[:, b].max() > vmax[b]},
           "critic": "pending (card E5)"}
    # an alternative reading for the user's decision (not the pre-registered rule): jerk outside the impact windows
    jt = t[1:-2] if len(t) > 3 else t[:0]
    out_imp = ~np.array([np.any(np.abs(jt_ - np.asarray(touch)) <= IMPACT_S) if touch else False for jt_ in jt], bool)
    jo = j[out_imp] if out_imp.any() else j[:0]
    nat["jerk_over_p99_outside_impacts"] = {names[b]: round(float(jo[:, b].max() / j99[b]), 2) for b in range(24)
                                            if len(jo) and jo[:, b].max() > j99[b]}
    nat["pass_p99"] = not nat["speed_over_p99"] and not nat["jerk_over_p99"]
    # ---------------- endpoints (the execution's final state, as edge_mppi evaluated it)
    m = runrec["metrics"]
    endpoints = {"final_d6_m": m["final_d6_m"], "com_speed_end": m["com_speed_end"], "brace_gap_end_cm": brace_gap}
    endpoints["pass"] = (m["final_d6_m"] <= D6_MAX and m["com_speed_end"] <= COM_SPEED_MAX
                         and all(g <= BRACE_CM for g in brace_gap.values()))
    # ---------------- physx (CPU part)
    rt_ = fw.round_trip(mot, "v2")
    lib = MotionLib(MotionLibConfig(motion_file=str(mpath)), device="cpu")
    try:
        plant_identity.require(lib.plant_sha256, plant_identity.mjcf_path("v2"), "synthetic clip")
        plant_ok = True
    except Exception as exc_:  # noqa: BLE001
        plant_ok = f"{type(exc_).__name__}: {exc_}"
    physx = {"round_trip_pos_m": rt_["pos_m"], "round_trip_ok": fw.round_trip_ok(rt_), "motionlib_cpu_plant_v2": plant_ok,
             "motionlib_frames": int(lib.motion_num_frames[0]), "launches_at_reset": "pending (IsaacLab, GPU)"}
    physx["pass_cpu"] = physx["round_trip_ok"] and plant_ok is True
    out.update(contract=contract, statics=statics, dynamics=dynamics, naturalness=nat, endpoints=endpoints, physx=physx)
    computed = [contract["pass"], statics["pass"], dynamics["pass"], nat["pass_p99"], endpoints["pass"], physx["pass_cpu"]]
    out["failed"] = [k for k, ok in zip(("contract", "statics", "dynamics", "naturalness_p99", "endpoints", "physx_cpu"),
                                        computed) if not ok]
    out["decision"] = "provisional" if all(computed) else "rejected"
    out["pending"] = ["naturalness critic (E5)", "PhysX: 0 launches at reset"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variants", nargs="+", type=Path, help="export_motion JSON records")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    print("cpu policy", gpu_guard.be_polite(), flush=True)
    torch.set_num_threads(1)
    rows = [check_variant(p) for p in args.variants]
    rec = {**ids.provenance(1, "edge_synthesis.admit", __file__, args.variants), "kind": "edge_admission",
           "thresholds": {"box_deg": BOX_DEG, "floor_cm": FLOOR_CM, "overlap_m": OVERLAP_M, "acc_max": ACC_MAX,
                          "impact_s": IMPACT_S, "torque_util": TORQUE_UTIL, "torque_share": TORQUE_SHARE, "mu": MU,
                          "cone_tol": CONE_TOL, "d6_max": D6_MAX, "com_speed_max": COM_SPEED_MAX, "brace_cm": BRACE_CM},
           "variants": rows,
           "summary": {r["variant"]: {"decision": r["decision"], "failed": r["failed"]} for r in rows}}
    args.out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    for r in rows:
        print(r["variant"], r["decision"], "failed:", r["failed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
