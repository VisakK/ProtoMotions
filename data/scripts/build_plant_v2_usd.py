# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plant v2 as a training asset (BodyFix Step 2, the USD half): convert, fix the articulation root, verify.

Turns ``data/assets/smpl/smpl_yogi03596_v2_flat.xml`` (``build_subject_plant_v2.py``) into the IsaacLab USD
package ``data/assets/smpl/smpl_yogi03596_v2_usd/`` and checks, without PhysX, that it is that MJCF; the
simulated side (masses, limits and FK as PhysX reports them) is ``reference_curation/plant_v2_physx.py``.

1. **Flatten check.** ``usd_convert/flatten_mjcf.py`` applied to ``smpl_yogi03596_v2.xml`` must compile to the
   same MuJoCo model as the builder's ``_v2_flat.xml`` (every compiled array but the name buffers). ``_v2.xml``
   itself compiles to it too except the motors' ``ctrlrange`` (its ``<default><motor>`` is not inlined, exactly
   as in the shipped pair; IsaacLab drops actuator ctrlrange): the flat twin the converter reads is the plant.
2. **Convert**: ``usd_convert/convert_robot_mjcf_to_usda.py``'s steps, run here with its own functions (flat
   check, strip ``<contact>``, the Isaac Sim importer, ``patch_usda``: the ``_cleaned`` suffix and
   ``over "worldBody" (active = false)``, then ``apply_mjcf_masses_to_usd.py`` to restore the MJCF densities).
   The wrapper cannot be run as it is: Isaac Sim saves every layer and then hangs in ``simulation_app.close()``
   (> 10 min, SIGTERM ignored), and the wrapper waits on it forever (under ``nohup`` it died with it; either
   way the post-processing never ran). So the importer runs under a watchdog: once it has printed its output
   line it gets 60 s to exit, then its process group is killed. The visual-mesh patch (another Isaac Sim app)
   is skipped because the MJCF has no mesh geom, which is checked. Log:
   ``output/reference_curation/plant_v2/usd_convert.log``.
3. **Articulation root.** The MJCF importer authors ``ArticulationRootAPI`` on an empty ``worldBody`` Xform as
   well as on the pelvis, and two roots make IsaacLab refuse the asset. Besides the ``.usda``'s deactivation,
   the API is removed from the physics layer itself (MimicKit's ``fix_humanoid_usd_articulation_root.py``
   rule, applied here to this package only), so the layer is right on its own; the composed stage must then
   hold exactly one active articulation root, on the pelvis rigid body.
4. **Checks against the MJCF** (every one of the 24 bodies and 23 joints):
   * each rigid body's density is the MJCF geom's, and none has an authored mass (density governs, so PhysX
     derives MuJoCo's mass, COM and inertia from the primitive);
   * each collision prim is its MJCF geom: position, orientation (box, capsule axis) and size, to 1e-3 mm /
     1e-3 deg (the card's tolerance; the importer reproduced the shipped plant to 1e-5 mm);
   * each body's rest transform is MuJoCo's rest pose, and each joint's frame is the MJCF body offset in its
     parent (``localPos0``; ``localPos1`` zero, both rotations identity), between parent and child;
   * each joint's ``limit:rot*`` is the MJCF range, the widened ones included, and each drive's ``maxForce``
     the MJCF ``actuatorfrcrange`` (wrist 30, hand 15 N m).
   * and the adoption verifier's ``adoption_risks/usd_collider_check.py`` (the card's check, untracked under
     ``output/``), re-run on this package: every collision prim within 1e-3 mm / 1e-3 deg.

Writes ``data/reference_curation/plant_v2/usd_v2.json`` (with the sha256 of every layer and of the MJCF it
checked) and exits non-zero if anything fails.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python data/scripts/build_plant_v2_usd.py [--skip-convert]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "data" / "scripts"), str(REPO), str(REPO / "usd_convert")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reference_curation import ids  # noqa: E402

SCHEMA_VERSION = 1
MODULE = "build_plant_v2_usd"
XML = REPO / "data/assets/smpl/smpl_yogi03596_v2.xml"
FLAT = REPO / "data/assets/smpl/smpl_yogi03596_v2_flat.xml"
USD_DIR = REPO / "data/assets/smpl/smpl_yogi03596_v2_usd"
STEM = FLAT.stem                                           # the converter names everything after its input
USDA = USD_DIR / f"{STEM}.usda"
LAYERS = {k: USD_DIR / "configuration" / f"{STEM}_{k}.usd" for k in ("base", "physics", "robot", "sensor")}
RECORD = ids.DATA_ROOT / "plant_v2" / "usd_v2.json"
LOG = ids.OUTPUT_ROOT / "plant_v2" / "usd_convert.log"
CONVERTER = REPO / "usd_convert/convert_robot_mjcf_to_usda.py"
IMPORTER = REPO / "usd_convert/convert_mjcf_to_usd.py"
IMPORTER_DONE = "Generated USD file:"      # printed by IMPORTER once MjcfConverter has saved every layer
IMPORTER_GRACE_S = 60.0                    # then it may take this long to exit before it is killed
IMPORTER_TIMEOUT_S = 1800.0
ADOPTION_CHECK = REPO / "output/reference_curation/subject_body_check/adoption_risks/usd_collider_check.py"
POS_TOL_MM = 1e-3
ROT_TOL_DEG = 1e-3
SIZE_TOL_MM = 1e-3
LIMIT_TOL_DEG = 1e-4
DENSITY_RTOL = 1e-6


# --------------------------------------------------------------------------- #
# 1. flatten
# --------------------------------------------------------------------------- #
def flatten_check() -> dict:
    """``flatten_mjcf`` of the main XML against the builder's flat twin, and the main XML against it."""
    import flatten_mjcf

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{XML.stem}_flattened.xml"
        flatten_mjcf.flatten_mjcf(str(XML), str(out), verify=False)
        diffs_flattener = flatten_mjcf.verify_models_match(str(out), str(FLAT))
    main = flatten_mjcf.verify_models_match(str(XML), str(FLAT))
    # The main XML's <default><motor ctrlrange="-1 1"> is not inlined by the flattener, exactly as in the shipped
    # pair (whose main and flat differ in the same two fields); IsaacLab drops actuator ctrlrange anyway.
    inherited = [d for d in main if not d.startswith(("actuator_ctrllimited", "actuator_ctrlrange"))]
    return {"flattened_main_vs_flat": diffs_flattener, "main_vs_flat": main, "main_vs_flat_beyond_motor_ctrlrange": inherited}


# --------------------------------------------------------------------------- #
# 2. convert
# --------------------------------------------------------------------------- #
def _run_importer(cmd: list[str], log) -> dict:
    """Run the Isaac Sim importer; once it has printed ``IMPORTER_DONE`` (every layer is saved by then) it gets
    ``IMPORTER_GRACE_S`` to exit before its process group is killed."""
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
    t0, t_done, exit_ = time.time(), None, "exited"
    while proc.poll() is None:
        time.sleep(2.0)
        if t_done is None and IMPORTER_DONE in LOG.read_text(errors="replace"):
            t_done = time.time()
        late = t_done is not None and time.time() - t_done > IMPORTER_GRACE_S
        if late or time.time() - t0 > IMPORTER_TIMEOUT_S:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            exit_ = "killed: hung on exit after writing" if late else "killed: timed out before writing"
    done = t_done is not None or IMPORTER_DONE in LOG.read_text(errors="replace")
    return {"returncode": proc.returncode, "exit": exit_, "reported_output": done,
            "seconds_to_output": round((t_done or time.time()) - t0, 1), "seconds": round(time.time() - t0, 1)}


def convert() -> dict:
    """``convert_robot_mjcf_to_usda.main``'s steps, in its order and with its own functions, the importer under a
    watchdog; the old package is replaced. Isaac Sim saves every layer and can then hang in
    ``simulation_app.close()`` (it did here: > 10 min idle on a futex after writing, and SIGTERM was ignored), and
    the wrapper, waiting on it, never reaches its post-processing (under ``nohup`` it died with it instead)."""
    import convert_robot_mjcf_to_usda as wrapper

    if USD_DIR.exists():
        shutil.rmtree(USD_DIR)
    USD_DIR.mkdir(parents=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = {"log": ids.display_path(LOG), "flat_issues": wrapper.verify_mjcf_is_flat(str(FLAT))}
    if rec["flat_issues"]:
        return {**rec, "returncode": 1}
    cleaned = FLAT.with_name(f"{STEM}_cleaned.xml")
    rec["stripped"] = wrapper.strip_mjcf(str(FLAT), str(cleaned))            # <contact>, <sensor>, <tendon>
    cmd = [sys.executable, wrapper.CONVERTER_SCRIPT, str(cleaned), str(USDA), "--make-instanceable", "--headless",
           "--kit_args", "--enable isaacsim.asset.importer.mjcf"]
    try:
        with open(LOG, "w") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            rec["importer"] = _run_importer(cmd, log)
            written = USDA.is_file() and all(p.is_file() for p in LAYERS.values())
            if not (rec["importer"]["reported_output"] and written):
                return {**rec, "returncode": 1, "seconds": round(time.time() - t0, 1)}
            # the wrapper's step 4 patches the base layer with the MJCF's <mesh> visuals (another Isaac Sim app);
            # this plant has none, which is checked rather than assumed
            meshes = [g for g in ET.parse(cleaned).getroot().iter("geom") if g.get("type") == "mesh"]
            if meshes:
                raise RuntimeError(f"{len(meshes)} mesh geoms: the visual-mesh patch would be needed")
            rec["visual_mesh_patch"] = "not needed: the MJCF has no mesh geoms"
            wrapper.patch_usda(str(USDA), STEM)                             # "_cleaned" suffix, worldBody override
            log.write("\n$ apply_mjcf_masses_to_usd\n")
            log.flush()
            masses = subprocess.run([sys.executable, wrapper.MASS_SCRIPT, "--mjcf", str(FLAT), "--usd",
                                     str(LAYERS["physics"]), "--no-backup"], cwd=str(REPO), stdout=log,
                                    stderr=subprocess.STDOUT)
            rec["masses_returncode"] = masses.returncode
    finally:
        if cleaned.exists():
            cleaned.unlink()
    rec["returncode"] = int(rec["masses_returncode"] != 0)
    rec["cleaned_xml_left_behind"] = cleaned.exists()
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------- #
# 3. articulation root
# --------------------------------------------------------------------------- #
def _layer_roots(layer: Path) -> list[str]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(layer))
    out = []
    for prim in stage.Traverse():
        meta = prim.GetMetadata("apiSchemas")
        applied = list(meta.GetAddedOrExplicitItems()) if meta else []
        if any("ArticulationRootAPI" in s for s in applied):
            out.append(str(prim.GetPath()))
    return out


def fix_articulation_root() -> dict:
    """Remove ``ArticulationRootAPI`` from every ``worldBody`` prim of the physics layer (an empty Xform: no
    child, no joint), keep the pelvis one; returns what was there and what was removed."""
    from pxr import Usd, UsdPhysics

    before = _layer_roots(LAYERS["physics"])
    stage = Usd.Stage.Open(str(LAYERS["physics"]))
    removed = []
    for path in before:
        prim = stage.GetPrimAtPath(path)
        if prim.GetName() == "worldBody":
            if list(prim.GetChildren()):
                raise RuntimeError(f"{path} has children; not removing its articulation root blindly")
            for s in ("PhysxArticulationAPI",):
                if s in [str(x) for x in (prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems())]:
                    prim.RemoveAppliedSchema(s)
            if prim.RemoveAPI(UsdPhysics.ArticulationRootAPI):
                removed.append(path)
    if removed:
        stage.GetRootLayer().Save()
    return {"roots_before": before, "removed": removed, "roots_after": _layer_roots(LAYERS["physics"])}


def composed_roots() -> dict:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(USDA))
    roots = [str(p.GetPath()) for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
             if p.HasAPI(UsdPhysics.ArticulationRootAPI) and p.IsActive()]
    dp = stage.GetDefaultPrim()
    wb = stage.GetPrimAtPath(dp.GetPath().AppendChild("worldBody")) if dp else None
    text = USDA.read_text()
    return {"default_prim": str(dp.GetPath()) if dp else None, "active_articulation_roots": roots,
            "usda_has_worldbody_override": 'over "worldBody"' in text,
            "usda_has_cleaned_suffix": "_cleaned" in text,
            "worldbody_active": bool(wb and wb.IsValid() and wb.IsActive())}


# --------------------------------------------------------------------------- #
# 4. checks against the MJCF
# --------------------------------------------------------------------------- #
def _mj():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(FLAT))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _local_mat(prim, stop_depth: int):
    """Transform of ``prim`` relative to its ancestor at path depth ``stop_depth`` (column vectors)."""
    from pxr import Gf, UsdGeom

    M = Gf.Matrix4d(1.0)
    p = prim
    while p and p.GetPath().pathString.count("/") > stop_depth:
        M = M * UsdGeom.Xformable(p).GetLocalTransformation()
        p = p.GetParent()
    return np.array(M).T


def check_physics_layer() -> dict:
    """Densities, colliders, body rest transforms, joint frames, limits and drive forces against the MJCF."""
    import mujoco
    from pxr import Usd, UsdPhysics

    from apply_mjcf_masses_to_usd import read_mjcf_bodies

    m, d = _mj()
    stage = Usd.Stage.Open(str(LAYERS["physics"]))
    root = stage.GetDefaultPrim()
    body_names = [m.body(i).name for i in range(1, m.nbody)]
    rigid = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    out = {"bodies": len(rigid), "missing_bodies": sorted(set(body_names) - set(rigid)),
           "extra_bodies": sorted(set(rigid) - set(body_names))}
    # densities
    spec = read_mjcf_bodies(FLAT)
    dens = {}
    for b in body_names:
        mass_api = UsdPhysics.MassAPI(rigid[b])
        rho, mass = mass_api.GetDensityAttr().Get(), mass_api.GetMassAttr().Get()
        dens[b] = {"usd_density": rho, "mjcf_density": spec[b]["density"], "usd_mass_attr": mass,
                   "rel_err": abs(rho - spec[b]["density"]) / spec[b]["density"]}
    out["density_max_rel_err"] = max(v["rel_err"] for v in dens.values())
    out["authored_masses"] = sorted(b for b, v in dens.items() if v["usd_mass_attr"])
    out["mjcf_total_mass_kg"] = float(sum(spec[b]["mass"] for b in body_names))
    # body rest transforms (relative to the root body Xform, i.e. pelvis at the origin)
    body_err = {}
    for i, b in enumerate(body_names, start=1):
        T = _local_mat(rigid[b], 2)
        body_err[b] = float(np.linalg.norm(T[:3, 3] - (d.xpos[i] - d.xpos[1])) * 1e3)
        body_err[b + ":rot_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    out["body_rest_pos_max_err_mm"] = max(v for k, v in body_err.items() if not k.endswith(":rot_deg"))
    out["body_rest_rot_max_err_deg"] = max(v for k, v in body_err.items() if k.endswith(":rot_deg"))
    # colliders
    kinds = {mujoco.mjtGeom.mjGEOM_SPHERE: "Sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "Capsule", mujoco.mjtGeom.mjGEOM_BOX: "Cube"}
    rows = []
    for g in range(m.ngeom):
        b = m.body(int(m.geom_bodyid[g])).name
        want = kinds[int(m.geom_type[g])]
        prims = [p for p in stage.Traverse() if p.GetPath().pathString.startswith(f"/collisions/{b}/") and p.GetTypeName() == want]
        if len(prims) != 1:
            rows.append({"body": b, "error": f"{len(prims)} {want} collision prims"})
            continue
        prim = prims[0]
        T = _local_mat(prim, 2)
        R_mj = np.zeros(9)
        mujoco.mju_quat2Mat(R_mj, m.geom_quat[g])
        R_mj = R_mj.reshape(3, 3)
        row = {"body": b, "type": want, "pos_err_mm": float(np.linalg.norm(T[:3, 3] - m.geom_pos[g]) * 1e3)}
        if want == "Cube":
            scale = np.linalg.norm(T[:3, :3], axis=0)
            Rn = T[:3, :3] / scale
            row["rot_err_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(Rn.T @ R_mj) - 1) / 2, -1, 1))))
            row["size_err_mm"] = float(np.abs(scale - m.geom_size[g][:3]).max() * 1e3)
        elif want == "Capsule":
            ax = prim.GetAttribute("axis").Get()
            k = "XYZ".index(ax)
            a = T[:3, k] / np.linalg.norm(T[:3, k])
            row["rot_err_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(a @ R_mj[:, 2]))))))
            half = prim.GetAttribute("height").Get() / 2 * float(np.linalg.norm(T[:3, k]))
            row["size_err_mm"] = float(max(abs(prim.GetAttribute("radius").Get() - m.geom_size[g][0]),
                                           abs(half - m.geom_size[g][1])) * 1e3)
        else:
            row["rot_err_deg"] = 0.0
            row["size_err_mm"] = float(abs(prim.GetAttribute("radius").Get() - m.geom_size[g][0]) * 1e3)
        rows.append(row)
    out["collider_errors"] = [r for r in rows if "error" in r]
    ok_rows = [r for r in rows if "error" not in r]
    out["collider_pos_max_err_mm"] = max(r["pos_err_mm"] for r in ok_rows)
    out["collider_rot_max_err_deg"] = max(r["rot_err_deg"] for r in ok_rows)
    out["collider_size_max_err_mm"] = max(r["size_err_mm"] for r in ok_rows)
    # joints: frames, limits, drives
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    jerr, lim_err, force_err, frame_err = [], 0.0, 0.0, 0.0
    for i, b in enumerate(body_names[1:], start=2):
        jp = joints.get(b)
        if jp is None:
            jerr.append(f"{b}: no joint")
            continue
        j = UsdPhysics.Joint(jp)
        parent = m.body(int(m.body_parentid[i])).name
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not (b0 and b1 and b0[0].name == parent and b1[0].name == b):
            jerr.append(f"{b}: joins {b0} -> {b1}, not {parent} -> {b}")
        lp0 = np.array(j.GetLocalPos0Attr().Get(), float)
        lp1 = np.array(j.GetLocalPos1Attr().Get(), float)
        q0, q1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
        frame_err = max(frame_err, float(np.abs(lp0 - m.body_pos[i]).max() * 1e3), float(np.abs(lp1).max() * 1e3))
        for q in (q0, q1):
            if abs(q.GetReal() - 1.0) > 1e-7:
                jerr.append(f"{b}: rotated joint frame {q}")
        for k, axis in enumerate("xyz"):
            jid = m.joint(f"{b}_{axis}").id
            lo, hi = np.degrees(m.jnt_range[jid])
            ulo = jp.GetAttribute(f"limit:rot{axis.upper()}:physics:low").Get()
            uhi = jp.GetAttribute(f"limit:rot{axis.upper()}:physics:high").Get()
            name = jp.GetAttribute(f"mjcf:rot{axis.upper()}:name").Get()
            if name != f"{b}_{axis}":
                jerr.append(f"{b}: rot{axis.upper()} is named {name}")
            lim_err = max(lim_err, abs(ulo - lo), abs(uhi - hi))
            mf = jp.GetAttribute(f"drive:rot{axis.upper()}:physics:maxForce").Get()
            force_err = max(force_err, abs(mf - float(m.jnt_actfrcrange[jid][1])))
    out.update(joints=len(joints), joint_errors=jerr, joint_frame_max_err_mm=frame_err, limit_max_err_deg=float(lim_err),
               drive_max_force_max_err=float(force_err))
    out["per_body_density"] = dens
    out["colliders"] = rows
    return out


def adoption_collider_check() -> dict:
    """The card's item 5: the adoption verifier's own ``usd_collider_check.py`` re-run on this package. Its two
    hard-coded paths (the shipped physics layer and flat MJCF) are swapped for v2's and its repo root pinned, in a
    temporary copy so its JSON (the shipped result, next to it) is not overwritten; nothing else changes."""
    if not ADOPTION_CHECK.is_file():
        return {"script": ids.display_path(ADOPTION_CHECK), "skipped": "not found"}
    src = ADOPTION_CHECK.read_text()
    subs = {"data/assets/smpl/smpl_yogi03596_lowtorque_usd/configuration/smpl_yogi03596_lowtorque_flat_physics.usd":
            ids.display_path(LAYERS["physics"]),
            "data/assets/smpl/smpl_yogi03596_lowtorque_flat.xml": ids.display_path(FLAT),
            "Path(__file__).resolve().parents[4]": f"Path({str(REPO)!r})"}
    for a, b in subs.items():
        if src.count(a) != 1:
            raise RuntimeError(f"{ADOPTION_CHECK} changed: {a!r} occurs {src.count(a)} times")
        src = src.replace(a, b)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / ADOPTION_CHECK.name
        script.write_text(src)
        res = subprocess.run([sys.executable, str(script)], cwd=str(REPO), capture_output=True, text=True)
        out = script.with_suffix(".json")
        worst = json.loads(out.read_text())["worst"] if out.is_file() else None
    return {"script": ids.display_path(ADOPTION_CHECK), "script_sha256": ids.sha256_file(ADOPTION_CHECK),
            "substituted": list(subs.values())[:2], "returncode": res.returncode, "worst": worst,
            "stderr_tail": res.stderr[-400:] if res.returncode else ""}


# --------------------------------------------------------------------------- #
def failures(rec: dict) -> list[str]:
    bad = []
    fl = rec["flatten"]
    if fl["flattened_main_vs_flat"] or fl["main_vs_flat_beyond_motor_ctrlrange"]:
        bad.append(f"flattening differs: {fl}")
    conv = rec.get("convert")
    if conv and (conv["returncode"] != 0 or conv.get("cleaned_xml_left_behind", True)):
        bad.append(f"converter: {conv}")
    r = rec["roots"]
    if len(r["composed"]["active_articulation_roots"]) != 1 or not r["composed"]["active_articulation_roots"][0].endswith("/Pelvis/Pelvis"):
        bad.append(f"articulation roots {r['composed']['active_articulation_roots']}")
    if any("worldBody" in p for p in r["layer"]["roots_after"]) or not r["composed"]["usda_has_worldbody_override"]:
        bad.append("worldBody still carries an articulation root / the .usda lost its override")
    c = rec["physics_layer"]
    if c["missing_bodies"] or c["extra_bodies"] or c["collider_errors"] or c["joint_errors"]:
        bad.append(f"structure: {c['missing_bodies']} {c['extra_bodies']} {c['collider_errors']} {c['joint_errors']}")
    if c["density_max_rel_err"] > DENSITY_RTOL or c["authored_masses"]:
        bad.append(f"densities off by {c['density_max_rel_err']:.2e} / authored masses {c['authored_masses']}")
    if c["collider_pos_max_err_mm"] > POS_TOL_MM or c["collider_rot_max_err_deg"] > ROT_TOL_DEG or c["collider_size_max_err_mm"] > SIZE_TOL_MM:
        bad.append("colliders differ from the MJCF geoms beyond 1e-3 mm / 1e-3 deg")
    if c["body_rest_pos_max_err_mm"] > POS_TOL_MM or c["joint_frame_max_err_mm"] > POS_TOL_MM or c["body_rest_rot_max_err_deg"] > ROT_TOL_DEG:
        bad.append("body or joint frames differ from the MJCF")
    if c["limit_max_err_deg"] > LIMIT_TOL_DEG or c["drive_max_force_max_err"] > 1e-4:
        bad.append(f"joint limits off by {c['limit_max_err_deg']} deg / drive forces by {c['drive_max_force_max_err']}")
    a = rec.get("adoption_collider_check", {})
    if "skipped" not in a:
        w = a.get("worst") or {}
        mm = [w.get(k, np.inf) for k in ("pos_err_mm", "box_extent_err_mm", "radius_err_mm")]
        deg = [w.get(k, np.inf) for k in ("box_rot_err_deg", "capsule_axis_err_deg")]
        if a.get("returncode") != 0 or max(mm) > POS_TOL_MM or max(deg) > ROT_TOL_DEG:
            bad.append(f"adoption collider check: {a}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-convert", action="store_true", help="check the existing package only")
    args = ap.parse_args(argv)
    t0 = time.time()
    rec = {"flatten": flatten_check()}
    if rec["flatten"]["flattened_main_vs_flat"] or rec["flatten"]["main_vs_flat_beyond_motor_ctrlrange"]:
        print(f"FAILED: flattening differs {rec['flatten']}")
        return 1
    if not args.skip_convert:
        rec["convert"] = convert()
        if rec["convert"]["returncode"] != 0:
            print(f"FAILED: converter exited {rec['convert']['returncode']}; see {rec['convert']['log']}")
            return 1
    rec["roots"] = {"layer": fix_articulation_root(), "composed": composed_roots()}
    rec["physics_layer"] = check_physics_layer()
    rec["adoption_collider_check"] = adoption_collider_check()
    rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [XML, FLAT, CONVERTER, IMPORTER]),
           "package": ids.display_path(USD_DIR), "usda": ids.display_path(USDA),
           "layer_sha256": {k: ids.sha256_file(p) for k, p in {"usda": USDA, **LAYERS}.items()},
           **rec, "seconds": round(time.time() - t0, 1)}
    if args.skip_convert and RECORD.exists():                # keep the conversion facts of the last full run
        old = json.loads(RECORD.read_text())
        if "convert" in old:
            rec["convert"] = old["convert"]
    rec["failures"] = failures(rec)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    c = rec["physics_layer"]
    print(f"plant v2 USD: {ids.display_path(USDA)}; roots {rec['roots']['composed']['active_articulation_roots']}; "
          f"{c['bodies']} bodies, {c['joints']} joints; colliders <= {c['collider_pos_max_err_mm']:.1e} mm / "
          f"{c['collider_rot_max_err_deg']:.1e} deg; limits <= {c['limit_max_err_deg']:.1e} deg; density rel err "
          f"{c['density_max_rel_err']:.1e}; {rec['seconds']} s" + (f"; FAILED: {rec['failures']}" if rec["failures"] else ""))
    return 1 if rec["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
