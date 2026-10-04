"""Card T4's export for a PhysX run (``edge_mppi_physx``): the executed motion as a plant-v2 ``.motion``.

The MuJoCo export (``export_motion``) has to *project* its executions into PhysX's joint box, because MuJoCo's
hinges have no box (the jump-backs needed up to 18 cm of edits at the airborne feet). A PhysX execution is already
in the box -- PhysX enforces it as hard stops -- so this export changes nothing the simulator did, unless asked:

* the frames are every 2nd physics substep (120 Hz -> 60 fps, the release's rate; the same instants as the
  MuJoCo export's every 4th of 240 Hz), the root and the exp-map ``dof_pos`` exactly as PhysX reported them (each
  joint's representative chosen against the box, ``retarget.nearest_representative``);
* **certification** is lane T's, unchanged: ``quasistatic.build_problem`` over the motion under its *executed*
  contacts (a zone is planted where its PhysX ground load exceeds ``costs.LOAD_N`` and its lowest collider point is
  within 2 cm; before the departure, S's scheduled configuration), then ``quasistatic.certify`` -- statics on the
  quasi-static frames, the box, the floor, new body-pair overlaps against the endpoints, the COM margin, the braces;
* **de-penetration, not projection** (``--mode``): the stored frames are PhysX's with only the millimetres of
  penetration PhysX's contacts allow removed (``depenetrate``, the default: T2's solver with the floor, the 253-pair
  guard and the stay-near terms only). T4's full projection (``--mode full``) edited the first PhysX B1 run by up to
  15.8 cm (p50 1.3 cm) to satisfy the kinematic rows -- planted faces flat, no slide, known-free zones 2 cm up, the
  COM margin on balance frames -- that a dynamic execution does not meet frame by frame: it would replace what
  PhysX did. The raw frames of that run met the contract except one 3.2 mm toe-on-toe overlap in flight
  (``--mode raw`` stores them; the record always carries the raw frames' certification too);
* every field by ``fit_writer.write_motion`` (pose_lib FK in float32, geometric contacts, plant v2's
  ``plant_sha256``); the pressure channels zero with validity 0, as every synthetic clip.

Writes ``output/edge_synthesis/physx/motions/SYN_<edge>_<run>.{motion,json,projected.npz}``.

CLI: ``PYTHONPATH=.:data/scripts python -m edge_synthesis.export_physx output/edge_synthesis/physx/runs/<run>``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from reference_curation import ids

OUT_ROOT = ids.REPO / "output/edge_synthesis/physx/motions"
TOUCHDOWN_EXEMPT_S = 0.2


def frames_60(z: dict) -> np.ndarray:
    """Indices of the 60 fps frames in a 120 Hz PhysX trajectory (substeps 2, 4, ...: t0 + k / 60)."""
    return np.arange(1, len(z["t"]), 2)


def executed_contacts(z: dict, idx: np.ndarray, sched, sk) -> "object":
    """The motion's own contact configuration at the frames ``idx`` (``export_motion.executed_contacts``'s rule on
    PhysX's per-body terrain forces)."""
    from scipy.spatial.transform import Rotation

    from edge_synthesis import costs as C
    from edge_synthesis import quasistatic as Q
    from extract_contact_configs import ZONE_ORDER

    t = z["t"][idx]
    fz = np.einsum("zb,tb->tz", z["zone_matrix"], z["ground"][idx, :, 2].astype(np.float64))
    pos = z["pos"][idx].astype(np.float64)
    R = Rotation.from_quat(z["rot"][idx].reshape(-1, 4)).as_matrix().reshape(len(idx), 24, 3, 3)
    low, _ = C.zone_lowest(sk, pos, R)
    ground = (fz > C.LOAD_N) & (low < 0.02)
    sc = Q.schedule_frames(sched, t)
    pre = t < 0.0
    ground[pre] = sc.ground[pre]
    exempt = np.zeros_like(ground)
    for zi in range(len(ZONE_ORDER)):
        for c in np.nonzero(np.diff(ground[:, zi].astype(int)) != 0)[0]:
            exempt[np.abs(t - t[c]) <= TOUCHDOWN_EXEMPT_S, zi] = True
    return Q.Contacts(ground, sc.braces, sc.known, sc.balance, exempt)


# The de-penetration solve: T2's solver with only the floor, the 253-pair guard and the stay-near terms. The rows a
# kinematic reference needs (planted faces resting and flat, no slide, known-free zones up, balance, brace closure)
# are switched off: a PhysX execution is dynamic, and those rows would rewrite it (15.8 cm on the first B1 run).
DEPENETRATE = {"support": 0.0, "off": 0.0, "pair": 0.0, "pen_soft": 0.0, "balance": 0.0, "flat": 0.0, "vel": 0.0,
               "ends": 0.0}
MODES = ("depenetrate", "raw", "full")


def project(run_dir: Path, iters: int = 40, mode: str = "depenetrate") -> dict:
    """``{times, root_pos, root_rot, dof, report}`` of a PhysX run at 60 fps, certified under its executed contacts.
    ``mode``: ``depenetrate`` (only floor and pair penetrations removed, ``DEPENETRATE``), ``raw`` (no solve: the
    PhysX frames clipped 0.5 deg inside the box) or ``full`` (T4's projection, every row of the problem)."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "raw":
        iters = 0
    from scipy.spatial.transform import Rotation

    from edge_synthesis import quasistatic as Q
    from edge_synthesis import sketch as SK
    from reference_curation import mosh_replay as mr
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2
    from extract_contact_configs import ZONE_ORDER

    run_dir = Path(run_dir)
    z = dict(np.load(run_dir / "trajectory.npz"))
    rec = json.load(open(run_dir / "run.json"))
    if rec.get("backend") != "physx":
        raise ValueError(f"{run_dir} is not a PhysX run (use export_motion for MuJoCo runs)")
    idx = frames_60(z)
    t = z["t"][idx]
    e = SK.edge(SK.load_edges(), rec["edge"])
    sched = SK.schedule(e, rec["durations_s"])
    con = executed_contacts(z, idx, sched, mr.skeleton_for(mr.V2_XML))
    root_pos = z["pos"][idx, 0].astype(np.float64)
    R = Rotation.from_quat(z["rot"][idx, 0].astype(np.float64)).as_matrix()
    dof = z["dof"][idx].astype(np.float64)
    with rv2.on_plant("v2") as sk:
        dof_rep = rt.nearest_representative(torch.as_tensor(dof), sk.lower, sk.upper).numpy()
        box_before = np.degrees(np.maximum(sk.lower.numpy() - dof_rep, 0) + np.maximum(dof_rep - sk.upper.numpy(), 0))
        prob = Q.build_problem(sk, f"export_{run_dir.name}", t, root_pos, R, dof_rep, con)
        x0 = rv2.solve(prob, iters=0)[0]
        cert0, _, _ = Q.certify(sk, prob, x0, con, t, Q.Pose(root_pos[0], R[0], dof_rep[0]),
                                Q.Pose(root_pos[-1], R[-1], dof_rep[-1]), dst_braces=sched.dst_braces)
        if mode == "depenetrate":
            prob.weights.update(DEPENETRATE)
        x, rep = rv2.solve(prob, x0=x0, iters=iters)
        S = Q.Pose(root_pos[0], R[0], dof_rep[0])
        D = Q.Pose(root_pos[-1], R[-1], dof_rep[-1])
        cert, _, _ = Q.certify(sk, prob, x, con, t, S, D, dst_braces=sched.dst_braces)
        rp, rr, dd = (v.numpy() for v in rt.unpack(prob, x))
        with torch.no_grad():
            p_before, _ = rt.fk(sk, torch.as_tensor(root_pos), torch.as_tensor(R), torch.as_tensor(dof_rep))
            p_after, _ = rt.fk(sk, torch.as_tensor(rp), torch.as_tensor(rr), torch.as_tensor(dd))
        edit = (p_after - p_before).norm(dim=-1)
        # PhysX's own bodies against FK of PhysX's own coordinates (the exported frames before any edit)
        fk_vs_physx = float((p_before - torch.as_tensor(z["pos"][idx].astype(np.float64))).norm(dim=-1).max())
    raw = {k: v for k, v in cert0.items() if k != "statics"}
    raw["statics_s_star_max"] = cert0["statics"]["s_star_max"]
    report = {"solver": {k: rep[k] for k in ("iterations", "seconds", "energy_start", "energy")},
              "mode": mode, "projected": iters > 0, "raw_certification": raw,
              "edit_body_max_cm": round(100 * float(edit.max()), 2), "edit_body_p50_cm": round(100 * float(edit.median()), 2),
              "box_before_max_deg": round(float(box_before.max()), 2),
              "physx_bodies_vs_fk_m": fk_vs_physx,
              "executed_ground_frames": {zn: int(con.ground[:, i].sum()) for i, zn in enumerate(ZONE_ORDER)
                                         if con.ground[:, i].any()},
              "certification": cert}
    return {"times": t, "root_pos": rp, "root_rot": rr, "dof": dd, "report": report, "rec": rec}


def export(run_dir: Path, out_dir: Path | None = None, iters: int = 40, mode: str = "depenetrate") -> dict:
    from reference_curation import fit_writer as fw

    pr = project(run_dir, iters, mode)
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
           "t_last": float(pr["times"][-1]), "source_run": ids.display_path(Path(run_dir).resolve()),
           "backend": "physx", "edge": pr["rec"]["edge"], "durations_s": pr["rec"]["durations_s"],
           "projection": pr["report"], "round_trip": rt_, "round_trip_ok": fw.round_trip_ok(rt_),
           "sha256": ids.sha256_file(path), "plant_sha256": mot.get("plant_sha256"),
           "pressure": "zero, validity 0 (synthetic)"}
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    np.savez_compressed(out_dir / f"{name}.projected.npz", times=pr["times"], root_pos=pr["root_pos"],
                        root_rot=pr["root_rot"], dof=pr["dof"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mode", default="depenetrate", choices=MODES,
                    help="depenetrate (default): remove floor/pair penetrations only; raw: no solve; full: T4's projection")
    args = ap.parse_args(argv)
    from edge_synthesis import gpu_guard
    print("cpu policy", gpu_guard.be_polite(), flush=True)
    torch.set_num_threads(1)
    ok = True
    for run in args.runs:
        out = export(run, args.out, mode=args.mode)
        c = out["projection"]["certification"]
        print(json.dumps({k: v for k, v in out.items() if k not in ("projection", "round_trip")}, default=float))
        print(json.dumps({k: v for k, v in out["projection"].items() if k != "certification"}, default=float))
        print(json.dumps({k: v for k, v in c.items() if k != "statics"}, default=float))
        print("statics s* max", c["statics"]["s_star_max"], "beyond", len(c["statics"]["beyond"]), flush=True)
        ok &= out["round_trip_ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
