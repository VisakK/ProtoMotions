# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plant checks (BUILD_PLAN Step 7's still-open list, closed before Step 8 uses statics): does the
physics layer describe the *training* plant?

Step 7's statics and witness run in MuJoCo, with XYZ hinges; training runs in PhysX, with D6 joints.
Everything here is measured on the recorded PhysX rollouts (``results/Contact_Physics_analysis*/<clip>/
rollout.npz``: a trained tracker on the smpl_yogi plant, 74 kg, 120 Hz, with per-pair contact forces from a
``RigidContactView``), so no GPU is needed.

1. **Joint coordinates** (``joint_convention``). A 3-dof joint's PhysX position is the exponential map
   (rotation vector) of its local rotation: ``dof_pos`` equals ``rotvec(R_parent^T R_child)`` of the
   recorded body quaternions (w first) to float precision on every frame. The ``.motion`` files'
   ``dof_pos`` use the same map (``extract_qpos_from_transforms(..., "exp_map")``), so they *are* the
   plant's joint coordinates.
2. **Joint limits** (``limit_hardness``). The USD carries the MJCF ranges exactly and a ``physxLimit``
   stiffness of 300-500, which Step 7 read as soft limits. The rollouts say otherwise: no coordinate
   passes its range by more than about 1 deg, while the actuator pushes into the stop at its torque
   limit (the applied torque is absorbed; the measured joint force stays small). The limits act as a hard
   box on the exp-map coordinates.
3. **The references against that box** (``reference_violations``). In the plant's coordinates, most
   shipped exemplars ask for joint positions the plant cannot reach. The count in MuJoCo's XYZ
   decomposition (Step 7: 290 of 303) was the wrong coordinates but nearly the right answer.
4. **Contact forces against the gated LP** (``force_check``). On recorded frames, the LP (with the
   frame's full inverse dynamics, M qacc + C, from 120 Hz finite differences, and the joint stops as free
   unilateral torques) gives every PhysX contact a normal-load interval within the plant's torque limits;
   the measured load should fall inside it. Pair necessity (useful / redundant) is compared with whether
   PhysX loads the pair.

``data/reference_curation/plant/plant_v1.json`` and ``summary.md`` hold the numbers.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.plant
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import glob
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import static_hold_lp as S
from extract_contact_configs import ZONES
from reference_curation import ids, statics

MODULE = "reference_curation.plant"
SCHEMA_VERSION = 1
PLANT_DIR = ids.DATA_ROOT / "plant"
ROLLOUT_GLOB = "results/Contact_Physics_analysis*/*/rollout.npz"
LIMIT_REPORT_DEG = 2.0
CONTACT_N = 5.0                  # a PhysX contact carrying more than this is in the LP's contact set
FORCE_TOL_FRAC = 0.05            # the measured load may miss the LP interval by 5 % of body weight
STRIDE = 4                       # every 4th physics frame (the 30 Hz control rate)
PARENT = {b: S.BODY[p] for b, p in zip(S.BODY, S.PARENT) if p >= 0}
BODY_ZONE = {b: z for z, bodies in ZONES.items() for b in bodies}


def rollouts(pattern: str = ROLLOUT_GLOB) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(ids.REPO / pattern)))]


def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def common_order(z: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """``(pos [T,B,3], quat xyzw [T,B,4], perm)`` in COMMON (MJCF) order; the rollout is in simulator
    order with quaternions w first."""
    names = list(z["body_names"])
    perm = [names.index(b) for b in S.BODY]
    return z["body_pos_w"][:, perm].astype(np.float64), z["body_quat_w"][:, perm][..., [1, 2, 3, 0]].astype(np.float64), perm


# --------------------------------------------------------------------------- #
# 1-2. Joint coordinates and limits
# --------------------------------------------------------------------------- #
def joint_convention(z: dict) -> dict:
    """Max |dof_pos - rotvec(local rotation)| and the same against XYZ Euler angles (rad)."""
    pos, quat, _ = common_order(z)
    dn = list(z["dof_names"])
    err_rv, err_xyz = 0.0, 0.0
    for b in S.BODY[1:]:
        rl = Rotation.from_quat(quat[:, S.BODY.index(PARENT[b])]).inv() * Rotation.from_quat(quat[:, S.BODY.index(b)])
        val = z["dof_pos"][:, [dn.index(f"{b}_{a}") for a in "xyz"]].astype(np.float64)
        err_rv = max(err_rv, float(np.abs(rl.as_rotvec() - val).max()))
        err_xyz = max(err_xyz, float(np.abs(rl.as_euler("XYZ") - val).max()))
    return {"max_err_rotvec_rad": err_rv, "max_err_xyz_euler_rad": err_xyz}


def limit_hardness(z: dict) -> dict:
    """How far past its MJCF range any recorded joint coordinate goes, and the torques at the worst frame."""
    rng = np.degrees(S.M.jnt_range[1:])
    dn = list(z["dof_names"])
    idx = [dn.index(n) for n in S.JNAMES]
    d = np.degrees(z["dof_pos"][:, idx].astype(np.float64))
    v = np.maximum(d - rng[:, 1], 0) + np.maximum(rng[:, 0] - d, 0)
    j = int(v.max(0).argmax())
    f = int(v[:, j].argmax())
    at_stop = v > 0.25
    applied = np.abs(z["torque_applied"][:, idx])
    return {"frames": int(len(d)), "max_violation_deg": round(float(v.max()), 3), "worst_joint": S.JNAMES[j],
            "worst_torque_applied_nm": round(float(z["torque_applied"][f, idx[j]]), 1),
            "worst_torque_measured_nm": round(float(z["torque_measured"][f, idx[j]]), 1),
            "frames_past_0.5deg": int((v > 0.5).any(1).sum()), "frames_past_2deg": int((v > 2).any(1).sum()),
            "joint_frames_at_stop": int(at_stop.sum()),
            "at_stop_with_saturated_actuator": int((at_stop & (applied >= 0.99 * z["effort_limit"][idx])).sum())}


def reference_violations(labels: dict, motion_dir: Path = ids.SHIPPED_DIR) -> dict:
    """The shipped references against the plant's box: exemplars past a range by > ``LIMIT_REPORT_DEG``
    in the exp-map coordinates (the plant's) and in MuJoCo's XYZ decomposition (Step 7's count), and every
    frame of every clip in exp-map."""
    rng = np.degrees(S.M.jnt_range[1:])
    by_joint, worst = collections.Counter(), []
    holds = exp_holds = xyz_holds = frames = frames_past = 0
    for stem, hs in labels["clips"]:
        mot = torch.load(ids.motion_path(stem, motion_dir), map_location="cpu", weights_only=False)
        d = np.degrees(mot["dof_pos"].double().numpy())
        v = np.maximum(d - rng[:, 1], 0) + np.maximum(rng[:, 0] - d, 0)
        frames += len(d)
        frames_past += int((v > LIMIT_REPORT_DEG).any(1).sum())
        pos, rot = mot["rigid_body_pos"].double().numpy(), mot["rigid_body_rot"].double().numpy()
        for h in hs:
            f = int(h["frame_hold"])
            holds += 1
            if (v[f] > LIMIT_REPORT_DEG).any():
                exp_holds += 1
                by_joint.update(S.JNAMES[j] for j in np.nonzero(v[f] > LIMIT_REPORT_DEG)[0])
                worst.append((h["hold_id"], round(float(v[f].max()), 1), S.JNAMES[int(v[f].argmax())]))
            xyz_holds += bool(statics.limit_violations(_qpos(pos[f], rot[f])))
    worst.sort(key=lambda w: -w[1])
    return {"holds": holds, "exemplars_past_range_expmap": exp_holds, "exemplars_past_range_xyz": xyz_holds,
            "by_joint_expmap": dict(by_joint.most_common(12)), "worst": worst[:12],
            "frames": frames, "frames_past_range_expmap": frames_past}


def _qpos(pos, rot) -> np.ndarray:
    S.set_pose(pos, rot)
    return S.D.qpos.copy()


# --------------------------------------------------------------------------- #
# 4. Contact forces against the gated LP
# --------------------------------------------------------------------------- #
QVEL_MAX = 30.0          # rad/s: a larger finite difference is an Euler-branch flip, not motion; skip


def dynamic_bias(qs: list, dt: float) -> np.ndarray | None:
    """``M qacc + C(q, qvel) + g(q)`` at the middle of three consecutive qpos, or ``None`` on a branch flip."""
    v0, v1 = np.zeros(S.M.nv), np.zeros(S.M.nv)
    mujoco.mj_differentiatePos(S.M, v0, dt, qs[0], qs[1])
    mujoco.mj_differentiatePos(S.M, v1, dt, qs[1], qs[2])
    qvel, qacc = 0.5 * (v0 + v1), (v1 - v0) / dt
    if np.abs(qvel[6:]).max() > QVEL_MAX:
        return None
    S.D.qpos[:], S.D.qvel[:], S.D.qacc[:] = qs[1], qvel, qacc
    mujoco.mj_forward(S.M, S.D)
    S.D.qacc[:] = qacc
    out = np.zeros(S.M.nv)
    mujoco.mj_rne(S.M, S.D, 1, out)
    return out


def physx_contacts(z: dict, perm: list, f: int) -> tuple[dict, dict]:
    """``({zone: vertical ground load N}, {pair: |force| N})`` PhysX measured at physics frame ``f``."""
    from reference_curation import verdicts

    pf = z["pair_force_w"][f][perm].astype(np.float64)       # [B common, 25 filters, 3]
    fnames = list(z["filter_names"])
    ground = collections.Counter()
    for i, b in enumerate(S.BODY):
        ground[BODY_ZONE[b]] += float(pf[i, 0, 2])
    pairs = collections.Counter()
    for i, a in enumerate(S.BODY):
        for j, fb in enumerate(fnames[1:], start=1):
            za, zb = BODY_ZONE[a], BODY_ZONE[fb]
            key = next(("+".join(p) for p in verdicts.BODY_PAIRS if set(p) == {za, zb}), None)
            if key:
                pairs[key] += 0.5 * float(np.linalg.norm(pf[i, j]))    # each pair appears from both sides
    return ({k: v for k, v in ground.items() if v > CONTACT_N}, {k: v for k, v in pairs.items() if v > CONTACT_N})


def force_frame(pos3: np.ndarray, quat3: np.ndarray, dt: float, ground: dict, pairs: dict, weight_n: float) -> dict | None:
    """One frame: the LP intervals of every PhysX contact and whether the measured load falls inside."""
    qs = [_qpos(p, q) for p, q in zip(pos3, quat3)]
    bias = dynamic_bias(qs, dt)
    if bias is None:
        return None
    prob, contacts, _ = statics.build_problem(pos3[1], quat3[1], ground=sorted(ground), pairs=sorted(pairs), stops=True)
    prob.bias = bias
    measured = {**{f"{z}:G": n for z, n in ground.items()}, **pairs}
    rows, tol = [], FORCE_TOL_FRAC * weight_n
    st, sol = statics.solve_lp(prob, stops=True)
    feasible = st == "optimal" and sol["s"] <= statics.S_CAP
    cap = statics.S_CAP if feasible else (max(statics.S_CAP, sol["s"] * (1 + statics.PEAK_SLACK)) if st == "optimal" else None)
    for name, c in contacts.items():
        row = {"contact": name, "kind": c["kind"], "measured_n": round(measured[name], 1), "gated_in": c["loaded"]}
        if c["loaded"] and cap is not None:
            i = prob.names.index(name)
            s_lo, lo = statics.solve_lp(prob, stops=True, s_cap=cap, load_of=i, sense=1)
            s_hi, hi = statics.solve_lp(prob, stops=True, s_cap=cap, load_of=i, sense=-1)
            if s_lo == "optimal" and s_hi == "optimal":
                row.update(lo_n=round(float(lo["loads"][i]), 1), hi_n=round(float(hi["loads"][i]), 1))
                row["inside"] = bool(lo["loads"][i] - tol <= measured[name] <= hi["loads"][i] + tol)
        rows.append(row)
    return {"lp_status": st, "s_star": None if st != "optimal" else round(float(sol["s"]), 3),
            "within_limits": bool(feasible), "rows": rows}


def force_check(path: Path, stride: int = STRIDE) -> dict:
    z = _load(path)
    pos, quat, perm = common_order(z)
    dt = float(z["dt_phys"])
    weight = float(z["body_masses"].sum()) * 9.81
    frames = range(8, len(pos) - 1, stride)
    out, skipped = [], 0
    for f in frames:
        ground, pairs = physx_contacts(z, perm, f)
        r = force_frame(pos[f - 1:f + 2], quat[f - 1:f + 2], dt, ground, pairs, weight)
        if r is None:
            skipped += 1
            continue
        r["frame"] = f
        r["motion_time"] = None
        out.append(r)
    return {"rollout": ids.display_path(path), "clip": str(z["motion_name"]), "checkpoint": str(z["checkpoint"]),
            "frames": len(out), "skipped_branch_flips": skipped, "records": out}


def summarize_forces(results: list[dict]) -> dict:
    """Per contact kind: how often the measured load falls inside the LP interval, and the frames the LP
    cannot hold within the plant's limits."""
    by = collections.defaultdict(lambda: collections.Counter())
    lp = collections.Counter()
    miss = []
    for res in results:
        for r in res["records"]:
            lp["within_limits" if r["within_limits"] else ("beyond_limits" if r["lp_status"] == "optimal" else r["lp_status"])] += 1
            for row in r["rows"]:
                k = row["kind"]
                if not row["gated_in"]:
                    by[k]["gated_out"] += 1
                elif "inside" in row:
                    by[k]["inside" if row["inside"] else "outside"] += 1
                    if not row["inside"]:
                        miss.append((res["clip"], r["frame"], row["contact"], row["measured_n"], row["lo_n"], row["hi_n"]))
                else:
                    by[k]["no_interval"] += 1
    return {"lp": dict(lp), "contacts": {k: dict(v) for k, v in by.items()}, "outside_examples": miss[:20]}


def pair_loads_vs_necessity(results: list[dict], statics_dir: Path) -> list[dict]:
    """For every configured pair of Step 7's statics on a rollout clip's holds: its necessity (on the
    shipped reference) beside how often PhysX loads it (> ``CONTACT_N``) while the policy plays the hold's
    window, and the median load then."""
    rows = [json.loads(l) for l in open(Path(statics_dir) / "contacts.jsonl")]
    out = []
    for res in results:
        z = _load(ids.REPO / res["rollout"])
        _, _, perm = common_order(z)
        t = np.arange(len(z["body_pos_w"])) * float(z["dt_phys"])
        frame = np.rint(t * 60).astype(int)          # the rollout plays the clip in real time from 0
        for r in rows:
            stem, f_hold = ids.parse_hold_id(r["hold_id"])
            if stem != res["clip"] or r["kind"] != "pair":
                continue
            # the hold's window in x0 frames: the exemplar +- 0.5 s (the labels' windows are in holds.jsonl)
            sel = np.nonzero(np.abs(frame - f_hold) <= 30)[0]
            if not len(sel):
                continue
            loads = np.array([physx_contacts(z, perm, int(f))[1].get(r["contact"], 0.0) for f in sel[::4]])
            out.append({"clip": stem, "hold_id": r["hold_id"], "pair": r["contact"], "necessity": r["necessity"],
                        "basis": r["necessity_basis"], "physx_loaded_fraction": round(float((loads > CONTACT_N).mean()), 3),
                        "physx_median_n": round(float(np.median(loads)), 1)})
    return out


def _force_worker(path: str) -> dict:
    return force_check(Path(path))


def run(labels_dir: Path | None = None, workers: int = 1) -> dict:
    labels = statics.load_labels(labels_dir or statics.default_labels_dir())
    paths = rollouts()
    conv = {ids.display_path(p): joint_convention(_load(p)) for p in paths}
    limits = {ids.display_path(p): limit_hardness(_load(p)) for p in paths}
    refs = reference_violations(labels)
    if workers > 1:
        from reference_curation.human_mesh import _single_threaded_children
        with _single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            forces = list(ex.map(_force_worker, [str(p) for p in paths]))
    else:
        forces = [force_check(p) for p in paths]
    statics_dir = sorted((ids.DATA_ROOT / "statics").glob(f"{labels['id']}.statics_v1.*"))[-1]
    necessity = pair_loads_vs_necessity(forces, statics_dir)
    return {"joint_convention": conv, "limits": limits, "references": refs,
            "forces": summarize_forces(forces), "forces_by_rollout": {f["rollout"]: summarize_forces([f]) for f in forces},
            "pair_necessity_vs_physx": necessity, "labels_id": labels["id"], "statics_id": statics_dir.name,
            "rollouts": [ids.display_path(p) for p in paths]}


def summary_markdown(res: dict) -> str:
    conv = res["joint_convention"]
    lim = res["limits"]
    ref = res["references"]
    f = res["forces"]
    worst_conv = max(v["max_err_rotvec_rad"] for v in conv.values())
    worst_lim = max(v["max_violation_deg"] for v in lim.values())
    lines = ["# Plant checks (BUILD_PLAN Step 7's still-open list)", "",
             f"Rollouts: {len(res['rollouts'])} (`{ROLLOUT_GLOB}`).", "",
             "## Joint coordinates", "",
             f"PhysX's `dof_pos` equals the rotation vector of each local rotation to **{worst_conv:.1e} rad** on "
             f"every frame of every rollout (XYZ Euler angles: up to "
             f"{max(v['max_err_xyz_euler_rad'] for v in conv.values()):.2f} rad). The plant's joint coordinates are "
             "the exp-map, the `.motion` files' `dof_pos` convention.", "",
             "## Joint limits", "",
             f"Largest excursion past a range over {sum(v['frames'] for v in lim.values())} frames: **{worst_lim:.2f} deg**, "
             f"with {sum(v['at_stop_with_saturated_actuator'] for v in lim.values())} joint-frames at a stop with the "
             "actuator saturated against it. The limits are a hard box on the exp-map coordinates.", "",
             "| Rollout | Frames | Max past range (deg) | Worst joint | Applied / measured torque (N m) |", "|---|---|---|---|---|"]
    lines += [f"| `{k.split('/')[-2][:48]}` ({k.split('/')[1]}) | {v['frames']} | {v['max_violation_deg']} | {v['worst_joint']} | "
              f"{v['worst_torque_applied_nm']} / {v['worst_torque_measured_nm']} |" for k, v in lim.items()]
    lines += ["", "## The shipped references against the box", "",
              f"Exemplars past a range by > {LIMIT_REPORT_DEG:g} deg: **{ref['exemplars_past_range_expmap']} of {ref['holds']}** "
              f"in exp-map (the plant's coordinates), {ref['exemplars_past_range_xyz']} in MuJoCo's XYZ decomposition. "
              f"Frames: {ref['frames_past_range_expmap']} of {ref['frames']}.", "",
              "By joint (exemplars): " + ", ".join(f"{k} {v}" for k, v in ref["by_joint_expmap"].items()), "",
              "Worst: " + "; ".join(f"`{h}` {d} deg ({j})" for h, d, j in ref["worst"][:6]), "",
              "## Contact forces against the gated LP", "",
              f"LP on the recorded frames (full inverse dynamics, joint stops free): {f['lp']}.", ""]
    for kind, c in f["contacts"].items():
        n = c.get("inside", 0) + c.get("outside", 0)
        lines.append(f"- {kind}: measured load inside the LP interval (+-{100 * FORCE_TOL_FRAC:g} % body weight) on "
                     f"**{c.get('inside', 0)} of {n}** ({(c.get('inside', 0) / max(n, 1)):.3f}); gated out {c.get('gated_out', 0)}, "
                     f"no interval {c.get('no_interval', 0)}")
    nec = res["pair_necessity_vs_physx"]
    lines += ["", "## Step 7's pair necessity against PhysX's pair loads", "",
              "| Hold | Pair | Necessity (basis) | PhysX loaded | Median N |", "|---|---|---|---|---|"]
    lines += [f"| `{r['hold_id']}` | {r['pair']} | {r['necessity']} ({r['basis']}) | {r['physx_loaded_fraction']} | "
              f"{r['physx_median_n']} |" for r in nec]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, default=PLANT_DIR)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        res = run(args.labels, args.workers)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    worst_conv = max(v["max_err_rotvec_rad"] for v in res["joint_convention"].values())
    worst_lim = max(v["max_violation_deg"] for v in res["limits"].values())
    failures = []
    if worst_conv > 1e-4:
        failures.append(f"dof_pos is not the exp-map of the recorded rotations ({worst_conv:.2e} rad)")
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = [ids.REPO / r for r in res["rollouts"]]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), **res}
    (args.out / "plant_v1.json").write_text(json.dumps(record, indent=1) + "\n")
    (args.out / "summary.md").write_text(summary_markdown(res))
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    c = res["forces"]["contacts"]
    print(f"plant: exp-map joints (max err {worst_conv:.1e} rad), limits hard (max {worst_lim:.2f} deg past range); "
          f"references past range at {res['references']['exemplars_past_range_expmap']}/{res['references']['holds']} exemplars; "
          f"ground loads inside LP intervals {c.get('ground', {}).get('inside', 0)}/"
          f"{c.get('ground', {}).get('inside', 0) + c.get('ground', {}).get('outside', 0)}; "
          f"in {time.time() - start:.0f} s -> {ids.display_path(args.out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
