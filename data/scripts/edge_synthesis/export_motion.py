"""Card T4's export: an executed edge (``edge_mppi``'s ``trajectory.npz``) as a plant-v2 ``.motion``.

* every ``EXPORT_STEP``-th physics state (240 Hz -> 60 fps, the release's rate);
* the joint coordinates as PhysX's exp-map, each joint's representative chosen and clipped into the box by
  ``plant_mj.Plant.export_dof`` (``retarget.nearest_representative``; the record reports the largest clip -- an export
  that clips more than 1 deg fails T5's contract, it is not hidden);
* every field by ``fit_writer.write_motion`` (pose_lib FK in float32, geometric contacts, plant v2's ``plant_sha256``,
  without which MotionLib on ``smpl_yogi_v2`` refuses the clip);
* the pressure channels zero with validity 0 (``ground_reaction``, ``rigid_body_ground_forces``,
  ``ground_reaction_valid``): MotionLib packs those channels all-or-nothing, and a synthetic frame has no mat;
* a JSON record beside it: the run it came from, the clip, the round trip (stored bodies = FK of the stored coordinates).

CLI: ``PYTHONPATH=.:data/scripts python -m edge_synthesis.export_motion output/edge_synthesis/runs/<run> [--out <dir>]``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from edge_synthesis import plant_mj as pm
from reference_curation import ids

EXPORT_STEP = 4                  # 240 Hz physics -> 60 fps
OUT_ROOT = ids.REPO / "output/edge_synthesis/motions"


def export(run_dir: Path, out_dir: Path | None = None, t_from: float | None = None, t_to: float | None = None) -> dict:
    from reference_curation import fit_writer as fw

    run_dir = Path(run_dir)
    z = np.load(run_dir / "trajectory.npz")
    rec = json.load(open(run_dir / "run.json"))
    plant = pm.Plant(nthread=1, sensors=False)
    dt, t0 = float(z["dt"]), float(z["t0"])
    t = t0 + dt * (np.arange(len(z["qpos"])) + 1)
    idx = np.arange(EXPORT_STEP - 1, len(t), EXPORT_STEP)
    if t_from is not None:
        idx = idx[t[idx] >= t_from - 1e-9]
    if t_to is not None:
        idx = idx[t[idx] <= t_to + 1e-9]
    qpos = z["qpos"][idx]
    root_pos, root_q, dof, clip = plant.export_dof(qpos)
    fps = int(round(1.0 / (dt * EXPORT_STEP)))
    mot = fw.write_motion(root_pos, root_q, dof, fps, "v2")
    T = mot["dof_pos"].shape[0]
    mot["ground_reaction"] = torch.zeros(T, 3)
    mot["rigid_body_ground_forces"] = torch.zeros(T, 24, 3)
    mot["ground_reaction_valid"] = torch.zeros(T, 3)
    out_dir = Path(out_dir) if out_dir is not None else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"SYN_{rec['edge']}_{run_dir.name}"
    path = out_dir / f"{name}.motion"
    torch.save(mot, path)
    back = torch.load(path, map_location="cpu", weights_only=False)
    rt_ = fw.round_trip(back, "v2")
    out = {"motion": ids.display_path(path), "name": name, "frames": T, "fps": fps, "t_first": float(t[idx[0]]),
           "t_last": float(t[idx[-1]]), "source_run": ids.display_path(run_dir), "edge": rec["edge"],
           "durations_s": rec["durations_s"], "box_clip": clip, "round_trip": rt_,
           "round_trip_ok": fw.round_trip_ok(rt_), "sha256": ids.sha256_file(path),
           "plant_sha256": mot.get("plant_sha256"), "pressure": "zero, validity 0 (synthetic)"}
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--raw", action="store_true", help="clip into the box without the projection solve")
    args = ap.parse_args(argv)
    from edge_synthesis import gpu_guard
    print("cpu policy", gpu_guard.be_polite(), flush=True)
    torch.set_num_threads(1)
    if args.raw:
        out = export(args.run, args.out)
        print(json.dumps(out, indent=1, default=float))
        return 0 if out["round_trip_ok"] and out["box_clip"]["clip_max_deg"] <= 1.0 else 1
    out = export_projected(args.run, args.out)
    c = out["projection"]["certification"]
    print(json.dumps({k: v for k, v in out.items() if k != "projection"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in out["projection"].items() if k != "certification"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in c.items() if k != "statics"}, indent=1, default=float))
    print("statics s* max", c["statics"]["s_star_max"], "beyond", len(c["statics"]["beyond"]))
    return 0 if out["round_trip_ok"] else 1



# --------------------------------------------------------------------------- #
# The export projection: the executed motion, brought inside PhysX's box with its own contacts kept
# --------------------------------------------------------------------------- #
TOUCHDOWN_EXEMPT_S = 0.2       # a zone that lands or lifts is neither kept off nor planted this long around it


def executed_contacts(plant, z: dict, idx: np.ndarray, t: np.ndarray, sched) -> "object":
    """The executed motion's own contact configuration at the exported frames: from the departure on, a zone is
    planted where the floor carries more than ``costs.LOAD_N`` on it and its lowest point is within 2 cm; before it,
    S's scheduled configuration; braces and the balance frames come from the edge's schedule."""
    from edge_synthesis import costs as C
    from edge_synthesis import quasistatic as Q
    from extract_contact_configs import ZONE_ORDER
    from reference_curation import mosh_replay as mr

    sk = mr.skeleton_for(mr.V2_XML)
    sens = z["sens"]
    f = plant.zone_forces(sens)[idx, :, 2]                                  # [F, 15]
    pos, rot = plant.fk(z["qpos"][idx])
    low, _ = C.zone_lowest(sk, pos, rot)
    ground = (f > C.LOAD_N) & (low < 0.02)
    sc = Q.schedule_frames(sched, t[idx])
    # before the departure the body is in S's hold: its configuration is the schedule's (the force-based reading of
    # the first frames sees the plant settling onto the floor -- the first B1 export's frame 0 read 'infeasible')
    pre = t[idx] < 0.0
    ground[pre] = sc.ground[pre]
    exempt = np.zeros_like(ground)
    tt = t[idx]
    for zi in range(len(ZONE_ORDER)):
        ch = np.nonzero(np.diff(ground[:, zi].astype(int)) != 0)[0]
        for c in ch:
            exempt[np.abs(tt - tt[c]) <= TOUCHDOWN_EXEMPT_S, zi] = True
    return Q.Contacts(ground, sc.braces, sc.known, sc.balance, exempt)


def project(run_dir: Path, iters: int = 40) -> dict:
    """``{times, root_pos, root_rot, dof, report}``: the run's executed motion at 60 fps projected into PhysX's box
    by ``quasistatic``'s solver, anchored on the executed motion itself under its executed contacts."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2
    from edge_synthesis import quasistatic as Q
    from edge_synthesis import sketch as SK
    from scipy.spatial.transform import Rotation

    run_dir = Path(run_dir)
    z = dict(np.load(run_dir / "trajectory.npz"))
    rec = json.load(open(run_dir / "run.json"))
    plant = pm.Plant(nthread=1)
    dt, t0 = float(z["dt"]), float(z["t0"])
    t = t0 + dt * (np.arange(len(z["qpos"])) + 1)
    idx = np.arange(EXPORT_STEP - 1, len(t), EXPORT_STEP)
    e = SK.edge(SK.load_edges(), rec["edge"])
    sched = SK.schedule(e, rec["durations_s"])
    con = executed_contacts(plant, z, idx, t, sched)
    root_pos, root_q, dof = plant.expmap_from_qpos(z["qpos"][idx])
    R = Rotation.from_quat(root_q).as_matrix()
    with rv2.on_plant("v2") as sk:
        dof_rep = rt.nearest_representative(torch.as_tensor(dof), sk.lower, sk.upper).numpy()
        prob = Q.build_problem(sk, f"export_{run_dir.name}", t[idx], root_pos, R, dof_rep, con)
        x, rep = rv2.solve(prob, iters=iters)
        S = Q.Pose(root_pos[0], R[0], dof_rep[0])
        D = Q.Pose(root_pos[-1], R[-1], dof_rep[-1])
        cert, _, _ = Q.certify(sk, prob, x, con, t[idx], S, D, dst_braces=sched.dst_braces)
        rp, rr, dd = (v.numpy() for v in rt.unpack(prob, x))
        with torch.no_grad():
            p_before, _ = rt.fk(sk, torch.as_tensor(root_pos), torch.as_tensor(R), torch.as_tensor(dof_rep))
            p_after, _ = rt.fk(sk, torch.as_tensor(rp), torch.as_tensor(rr), torch.as_tensor(dd))
        edit = (p_after - p_before).norm(dim=-1)
    report = {"solver": {k: rep[k] for k in ("iterations", "seconds", "energy_start", "energy")},
              "edit_body_max_cm": round(100 * float(edit.max()), 2), "edit_body_p50_cm": round(100 * float(edit.median()), 2),
              "box_before_max_deg": round(float(np.degrees(plant.box_excess(dof_rep).max())), 2),
              "executed_ground_frames": {z_: int(con.ground[:, i].sum()) for i, z_ in enumerate(
                  __import__("extract_contact_configs").ZONE_ORDER) if con.ground[:, i].any()},
              "certification": cert}
    return {"times": t[idx], "root_pos": rp, "root_rot": rr, "dof": dd, "report": report, "rec": rec}


def export_projected(run_dir: Path, out_dir: Path | None = None, iters: int = 40) -> dict:
    from reference_curation import fit_writer as fw

    pr = project(run_dir, iters)
    mot = fw.write_motion(pr["root_pos"], pr["root_rot"], pr["dof"], 60, "v2")
    T = mot["dof_pos"].shape[0]
    mot["ground_reaction"] = torch.zeros(T, 3)
    mot["rigid_body_ground_forces"] = torch.zeros(T, 24, 3)
    mot["ground_reaction_valid"] = torch.zeros(T, 3)
    out_dir = Path(out_dir) if out_dir is not None else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"SYN_{pr['rec']['edge']}_{Path(run_dir).name}"
    path = out_dir / f"{name}.motion"
    torch.save(mot, path)
    back = torch.load(path, map_location="cpu", weights_only=False)
    rt_ = fw.round_trip(back, "v2")
    out = {"motion": ids.display_path(path), "name": name, "frames": T, "fps": 60, "t_first": float(pr["times"][0]),
           "t_last": float(pr["times"][-1]), "source_run": ids.display_path(run_dir), "edge": pr["rec"]["edge"],
           "durations_s": pr["rec"]["durations_s"], "projection": pr["report"], "round_trip": rt_,
           "round_trip_ok": fw.round_trip_ok(rt_), "sha256": ids.sha256_file(path),
           "plant_sha256": mot.get("plant_sha256"), "pressure": "zero, validity 0 (synthetic)"}
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    np.savez_compressed(out_dir / f"{name}.projected.npz", times=pr["times"], root_pos=pr["root_pos"],
                        root_rot=pr["root_rot"], dof=pr["dof"])
    return out


if __name__ == "__main__":
    sys.exit(main())
