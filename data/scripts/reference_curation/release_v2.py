# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The training release on plant v2 (BUILD_PLAN Steps 9-10, BodyFix Step 5): one immutable set of training
artifacts, built from three things only -- gate v2's release manifest, the Step 3 references and plant v2.

Inputs, pinned by id and content (nothing reads "the newest folder")
-------------------------------------------------------------------
* gate v2 ``holds_repaired_ftC_posefix.gate_v2.b2f76f2606/release_holds.yaml`` (its sha256 is pinned here): labels
  v2's manifest without the 10 excluded holds, the commanded contacts (configured minus masks) and the gate block.
  It names everything else: labels v2 (annotations), the Step 3 retarget, the fit, plant v2;
* the Step 3 references (``retarget_v2``), each checked against the retarget record's sha256;
* their gated pressure ports (``pressure_v2``: the reference plus the mat's three channels), each checked
  against the pressure record's sha256 and to carry exactly the reference's kinematics.

The chain (``build_expert60_ftR.sh``'s, pointed at v2 everywhere)
------------------------------------------------------------------
1. **Hold extension v2** (``hold_extension_v2``): every clip in the 0 / 3 / 7 s variants, real-frame pressure kept,
   inserted frames invalid; ``holds_extended.yaml`` (every hold keeps its ``hold_id`` and gate block) and
   ``lineage.npz`` (each variant frame's x0 source frame and whether it was inserted).
2. **Package** (``package_motion_subset.py``): ``motions.pt``, verified frame by frame, the pressure included.
3. **Graph v2** (``build_hold_graph_v2.py``): keyed on hold ids; manual goals resolve the side-specific segment.
4. **Physics tables v2** (``build_physics_tables_v2.py``): column 1 for newtons, column 2 for shares, the
   attribution-visibility gate on "unloaded"; the plant's and the graph's sha256.
5. **Contact-target sidecar** (``contact_targets_v2``; TODO C1): roles, masks, flags, critical and restored pairs,
   statics loads, known-free ground zones, per-frame pair eligibility.
6. **Probe plans** (``make_hold_graph_probe_plans.py``) on the release's own graph, for the viz panel.

Then the **identity checks** (TODO C5, ``check``: the port of ``check_expert60_ftR.py``): package, graph, tables,
sidecar and manifest agree on hashes, motion order, frame counts, fps, body order, ``pair_names``, hold ids and the
plant, every derived artifact equals what its inputs determine, and the runtime loaders accept them; and the **swing
calibration re-check** (TODO C3, ``calibrate_swing``) on the new tables against the human evidence.

Outputs
-------
Heavy (untracked, like every corpus in ``data/smpl/``): ``data/smpl/reference_curation/<release_id>/``. The
record: ``data/reference_curation/releases/<release_id>.json`` (+ ``.md``), every artifact's sha256; it is written
last and only when every check passes, so a folder without a record is not a release. The release id hashes the
pinned inputs, the builders' sources and the parameters; a release is never rebuilt in place.

Run in a plant-v1 process (no ``REFERENCE_PLANT``): the human-side capture stores refuse plant v2, and plant v2 is
named explicitly everywhere it is meant.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.release_v2 --build
    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.release_v2 --check <release_id>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from extract_contact_configs import ORIENT_BINS, ZONE_ORDER, mjcf_body_names, zone_pairs
from reference_curation import capture_v4, contact_targets_v2 as ct, hold_extension_v2 as hx, ids, sources

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.release_v2"
SCHEMA_VERSION = 1
RELEASE_VERSION = "release_v2"
GATE_ID = "holds_repaired_ftC_posefix.gate_v2.b2f76f2606"
GATE_MANIFEST_SHA256 = "faa474ee65296108a11c786a8c8139ba4259dc8dc25da69b42dcf40fe25581bb"
PLANT = "v2"
VARIANTS = (0.0, 3.0, 7.0)
FPS = 60
CONTROL_HZ = 30            # the evaluator's unit (control steps); the full-clip window covers the longest motion
MIN_LEAD_S = 0.2
HEAVY_ROOT = REPO / "data/smpl/reference_curation"
RECORD_ROOT = ids.DATA_ROOT / "releases"
SCRIPTS = REPO / "data/scripts"
ARCHIVES = REPO / "data/smpl/yoga_pressure"
KINEMATIC = ("rigid_body_pos", "rigid_body_rot", "local_rigid_body_rot", "dof_pos", "rigid_body_contacts")
PACKED = {"gts": "rigid_body_pos", "grs": "rigid_body_rot", "lrs": "local_rigid_body_rot", "dps": "dof_pos",
          "dvs": "dof_vel", "gvs": "rigid_body_vel", "gavs": "rigid_body_ang_vel", "contacts": "rigid_body_contacts",
          "gnf": "rigid_body_ground_forces", "grc": "ground_reaction", "grw": "ground_reaction_valid"}
# The code whose output the release is: its sha256s are part of the release id.
BUILDERS = (Path(__file__), Path(hx.__file__), Path(ct.__file__), SCRIPTS / "make_hold_extended_clips.py",
            SCRIPTS / "package_motion_subset.py", SCRIPTS / "build_hold_graph_v2.py", SCRIPTS / "build_hold_graph.py",
            SCRIPTS / "build_physics_tables_v2.py", SCRIPTS / "build_physics_tables.py",
            SCRIPTS / "make_hold_graph_probe_plans.py", SCRIPTS / "extract_contact_configs.py",
            SCRIPTS / "contact_geometry.py")
# For the swing re-check (C3): the ftC and ftR tables' own statistics.
TABLE_BASELINES = {"ftC": REPO / "data/smpl/yoga_hold_graph_expert60_ftC/physics_tables.json",
                   "ftR": REPO / "data/smpl/yoga_hold_graph_expert60_ftR/physics_tables.json"}
ARTIFACTS = {  # record role -> file in the release folder
    "package": "motions.pt", "package_manifest": "motions.yaml", "graph": "contact_graph.pt",
    "graph_description": "contact_graph.json", "physics_tables": "physics_tables.pt",
    "physics_tables_stats": "physics_tables.json", "contact_targets": "contact_targets.pt",
    "contact_targets_summary": "contact_targets.json", "holds_extended": "holds_extended.yaml",
    "release_holds": "release_holds.yaml", "lineage": "lineage.npz",
}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def plant_mjcf() -> Path:
    return plant_identity.mjcf_path(PLANT)


def pinned_inputs(gate_id: str = GATE_ID) -> dict:
    """Every input of the release, resolved from the gate's manifest and checked against its pins."""
    gate_dir = ids.DATA_ROOT / "gate_v2" / gate_id
    manifest_path = gate_dir / "release_holds.yaml"
    sha = ids.sha256_file(manifest_path)
    if gate_id == GATE_ID and sha != GATE_MANIFEST_SHA256:
        raise ValueError(f"{ids.display_path(manifest_path)} has sha256 {sha[:12]}, pinned {GATE_MANIFEST_SHA256[:12]}")
    manifest = yaml.safe_load(open(manifest_path))
    lab = manifest["labels"]
    if lab.get("gate_id") != gate_id:
        raise ValueError(f"the manifest names gate {lab.get('gate_id')}, not {gate_id}")
    v2 = plant_identity.identity(PLANT)
    if manifest.get(plant_identity.KEY) != v2[plant_identity.KEY] or lab["plant"][plant_identity.KEY] != v2[plant_identity.KEY]:
        raise plant_identity.PlantMismatchError(f"gate {gate_id}'s manifest is not on plant v2")
    rid = lab["retarget_id"]
    retarget_path = ids.DATA_ROOT / "retarget_v2" / rid / "retarget.json"
    pressure_path = ids.DATA_ROOT / "pressure_v2" / rid / "pressure.json"
    return {
        "gate_id": gate_id, "gate_dir": gate_dir, "gate_record": gate_dir / "gate.json",
        "manifest_path": manifest_path, "manifest_sha256": sha, "manifest": manifest,
        "labels_id": lab["labels_id"], "labels_dir": ids.DATA_ROOT / "labels" / lab["labels_id"],
        "statics_id": lab["statics_id"], "retarget_id": rid, "fit_id": lab["fit_id"],
        "retarget_record_path": retarget_path, "retarget_record": json.loads(retarget_path.read_text()),
        "pressure_record_path": pressure_path, "pressure_record": json.loads(pressure_path.read_text()),
        "port_dir": ids.OUTPUT_ROOT / "pressure_v2" / rid / "gated",
        "plant": v2,
        "stems": [c["stem"] for c in manifest["clips"]],
    }


def parameters() -> dict:
    return {"variants_s": list(VARIANTS), "fps": FPS, "min_lead_s": MIN_LEAD_S,
            "known_free_share": ct.FREE_SHARE, "pair_close_m": ct.PAIR_CLOSE_M,
            "tables": "build_physics_tables_v2.py defaults (v1's thresholds)"}


def release_id(inp: dict) -> str:
    """``<labels stem>.release_v2.<hash>`` over the pinned inputs, the builders' sources and the parameters."""
    key = {
        "gate_id": inp["gate_id"], "manifest_sha256": inp["manifest_sha256"],
        "annotations": ids.sha256_file(inp["labels_dir"] / "annotations.jsonl"),
        "retarget_record": ids.sha256_file(inp["retarget_record_path"]),
        "pressure_record": ids.sha256_file(inp["pressure_record_path"]),
        "plant": inp["plant"][plant_identity.KEY],
        "builders": {ids.display_path(p): ids.sha256_file(p) for p in BUILDERS},
        "parameters": parameters(),
    }
    stem = inp["labels_id"].split(".labels_")[0]
    return f"{stem}.{RELEASE_VERSION}.{ids.sha256_json(key)[:10]}"


def port_path(inp: dict, stem: str) -> Path:
    return inp["port_dir"] / f"{stem}.motion"


def load(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def verify_sources(inp: dict) -> list[str]:
    """Every reference and gated port is the recorded one, on plant v2, with the manifest's frames; the port
    carries exactly the reference's kinematics plus the three measured channels."""
    problems = []
    sha = inp["plant"][plant_identity.KEY]
    motions, outputs = inp["retarget_record"]["motions"], inp["pressure_record"]["outputs"]
    for c in inp["manifest"]["clips"]:
        stem, src, port = c["stem"], Path(c["source"]), port_path(inp, c["stem"])
        if ids.sha256_file(src) != motions.get(stem):
            problems.append(f"{stem}: the reference is not the retarget record's")
            continue
        if ids.sha256_file(port) != outputs.get(f"gated/{stem}.motion"):
            problems.append(f"{stem}: the gated port is not the pressure record's")
            continue
        m, p = load(src), load(port)
        if m.get(plant_identity.KEY) != sha or p.get(plant_identity.KEY) != sha:
            problems.append(f"{stem}: not on plant v2")
        if int(m["fps"]) != FPS or int(c["fps"]) != FPS or m["rigid_body_pos"].shape[0] != int(c["num_frames"]):
            problems.append(f"{stem}: {m['rigid_body_pos'].shape[0]} frames at {m['fps']} fps, the manifest says "
                            f"{c['num_frames']} at {c['fps']}")
        for k, v in m.items():
            same = torch.equal(p[k], v) if torch.is_tensor(v) else p.get(k) == v
            if not same:
                problems.append(f"{stem}: the port's {k} differs from the reference's")
        missing = [k for k in hx.PRESSURE_FIELDS if p.get(k) is None]
        if missing or p["ground_reaction_valid"].shape[1] != 3:
            problems.append(f"{stem}: the port lacks {missing or 'validity column 2'}")
    return problems


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def extend_corpus(inp: dict, out: Path) -> dict:
    """Hold extension v2 into ``out``: ``motions/``, ``holds_extended.yaml``, ``lineage.npz``. Returns facts."""
    motions_dir = out / "motions"
    motions_dir.mkdir(parents=True)
    emitted, lineage = [], {}
    drift_max = {k: [0.0, 0.0] for k in ("dof_vel", "rigid_body_vel", "rigid_body_ang_vel")}
    for clip in inp["manifest"]["clips"]:
        stem = clip["stem"]
        port = port_path(inp, stem)
        motion = load(port)
        for k, (mean, peak) in hx.velocity_drift(motion).items():
            drift_max[k] = [max(drift_max[k][0], mean), max(drift_max[k][1], peak)]
            if mean > hx.MAX_VELOCITY_DRIFT:
                raise ValueError(f"{stem}: recomputed {k} drifts {mean:.4f} from the stored one: the converter's "
                                 "conventions changed")
        num_frames = int(motion["rigid_body_pos"].shape[0])
        for d in VARIANTS:
            plan = hx.insertion_plan(clip["holds"], int(round(d * FPS)))
            name = hx.variant_stem(stem, d)
            path = motions_dir / f"{name}.motion"
            if d == 0.0:
                shutil.copyfile(port, path)          # the d = 0 variant is the gated port, byte for byte
                index = torch.arange(num_frames)
                inserted = torch.zeros(num_frames, dtype=torch.bool)
            else:
                variant, index, inserted = hx.extend_motion_v2(motion, plan)
                torch.save(variant, path)
            n = int(index.shape[0])
            emitted.append({
                **{k: v for k, v in clip.items() if k not in ("holds", "source", "num_frames", "length_s")},
                "stem": name, "source_stem": stem, "source": clip["source"],
                "pressure_port": ids.display_path(port), "variant_s": float(d),
                "inserted_frames": int(sum(k for _, k in plan)), "num_frames": n, "length_s": round(n / FPS, 3),
                "sha256": ids.sha256_file(path),
                "holds": hx.variant_holds(clip["holds"], plan, FPS),
            })
            lineage[f"{name}.index"] = index.numpy().astype(np.int32)
            lineage[f"{name}.inserted"] = inserted.numpy()
    payload = {
        "version": 2, "builder": f"{MODULE} (hold_extension_v2)",
        "source_manifest": ids.display_path(inp["manifest_path"]), "source_manifest_sha256": inp["manifest_sha256"],
        "variants_s": [float(v) for v in VARIANTS],
        "pressure": "real frames: the gated port's measurement; inserted frames: zeros with every validity column 0",
        "velocity_drift_max": {k: {"mean": round(v[0], 5), "max": round(v[1], 5)} for k, v in drift_max.items()},
        "clips": emitted,
    }
    with open(out / "holds_extended.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=120)
    np.savez_compressed(out / "lineage.npz", **lineage)
    return {"motions": len(emitted), "frames": sum(e["num_frames"] for e in emitted),
            "inserted_frames": sum(e["inserted_frames"] for e in emitted), "velocity_drift_max": payload["velocity_drift_max"]}


def run(script: str, args: list, log) -> str:
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{SCRIPTS}")
    env.pop("REFERENCE_PLANT", None)
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    log.write(f"$ {' '.join(cmd)}\n[{time.time() - start:.0f} s, exit {proc.returncode}]\n{proc.stdout[-20000:]}\n"
              f"{proc.stderr[-20000:]}\n")
    log.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"{script} exited {proc.returncode}: {proc.stderr[-1500:]}")
    return proc.stdout


def build(gate_id: str = GATE_ID, force: bool = False) -> tuple[Path, dict]:
    """Build, check and record the release; refuse to touch an existing one."""
    capture_v4.require_v1_process()
    start = time.time()
    inp = pinned_inputs(gate_id)
    problems = verify_sources(inp)
    if problems:
        raise ValueError("inputs do not verify:\n  " + "\n  ".join(problems[:20]))
    rid = release_id(inp)
    out, record_path = HEAVY_ROOT / rid, RECORD_ROOT / f"{rid}.json"
    if record_path.exists():
        raise FileExistsError(f"release {rid} exists (its record is {ids.display_path(record_path)}); run --check")
    if out.exists():
        if not force:
            raise FileExistsError(f"{ids.display_path(out)} exists without a record (a failed build); pass --force "
                                  "to delete it and rebuild")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    facts = {"release_id": rid}
    shutil.copyfile(inp["manifest_path"], out / "release_holds.yaml")
    with open(out / "build.log", "w") as log:
        facts["extension"] = extend_corpus(inp, out)
        ext = yaml.safe_load(open(out / "holds_extended.yaml"))
        files = [str(out / "motions" / f"{c['stem']}.motion") for c in ext["clips"]]
        run("package_motion_subset.py", ["--force", "--out", out / "motions.pt", "--yaml", out / "motions.yaml", *files],
            log)
        run("build_hold_graph_v2.py", ["--manifest", out / "holds_extended.yaml", "--motion-file", out / "motions.pt",
                                       "--out-dir", out, "--min-lead-s", MIN_LEAD_S], log)
        run("build_physics_tables_v2.py", ["--extended-manifest", out / "holds_extended.yaml", "--graph",
                                           out / "contact_graph.pt", "--motion-dir", out / "motions", "--motion-file",
                                           out / "motions.pt", "--out", out / "physics_tables.pt", "--mjcf",
                                           plant_mjcf(), "--archive-dir", ARCHIVES, "--fps", FPS], log)
        payload, summary = ct.compile_targets(out, inp["labels_dir"], inp["manifest"], rid)
        torch.save(payload, out / "contact_targets.pt")
        (out / "contact_targets.json").write_text(json.dumps(summary, indent=1) + "\n")
        run("make_hold_graph_probe_plans.py", ["--graph", out / "contact_graph.json", "--out-dir", out / "plans"], log)
    passed, failed, check_facts = check(out, inp)
    facts.update(check_facts)
    if failed:
        for f in failed:
            print(f"FAILED  {f}", file=sys.stderr)
        raise RuntimeError(f"{len(failed)} identity checks failed; {ids.display_path(out)} is not a release")
    calibration = calibrate_swing(out, inp)
    rec = record(rid, inp, out, passed, facts, calibration, time.time() - start)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(rec, indent=1) + "\n")
    record_path.with_suffix(".md").write_text(summary_markdown(rec))
    return out, rec


# --------------------------------------------------------------------------- #
# Identity checks (TODO C5)
# --------------------------------------------------------------------------- #
class Checks:
    def __init__(self):
        self.passed, self.failed = [], []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        (self.passed if ok else self.failed).append(f"{name}{': ' + detail if detail else ''}")
        return bool(ok)


def check(out: Path, inp: dict) -> tuple[list, list, dict]:
    """The release's identity checks; ``(passed, failed, facts)``. Every derived artifact is re-derived or compared
    with what its inputs determine; nothing is trusted because a builder printed OK."""
    from build_hold_graph import node_key
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.envs.control.contact_targets import ContactTargets
    from protomotions.envs.control.physics_terms import PhysicsTables

    c, facts = Checks(), {}
    v2_sha = inp["plant"][plant_identity.KEY]
    manifest = inp["manifest"]
    x0 = {cl["stem"]: cl for cl in manifest["clips"]}

    # 1. the pinned manifest and its sources
    c("manifest pinned", ids.sha256_file(out / "release_holds.yaml") == inp["manifest_sha256"],
      f"{len(manifest['clips'])} clips, {sum(len(cl['holds']) for cl in manifest['clips'])} holds")
    problems = verify_sources(inp)
    c("sources are the retarget's references and the pressure record's ports", not problems, "; ".join(problems[:3]))

    # 2. hold extension v2
    ext = yaml.safe_load(open(out / "holds_extended.yaml"))
    names = [e["stem"] for e in ext["clips"]]
    expected = [hx.variant_stem(s, d) for s in x0 for d in VARIANTS]
    c("extended clips: every clip in every variant, in manifest order", names == expected, f"{len(names)} motions")
    lineage = np.load(out / "lineage.npz")
    bad_splice, bad_pressure, bad_holds, bad_x0 = [], [], [], []
    motions = {}
    for e in ext["clips"]:
        stem, src = e["stem"], e["source_stem"]
        path = out / "motions" / f"{stem}.motion"
        m = load(path)
        motions[stem] = m
        port = load(port_path(inp, src))
        plan = hx.insertion_plan(x0[src]["holds"], int(round(e["variant_s"] * FPS)))
        index = hx.splice_index(int(x0[src]["num_frames"]), plan)
        inserted = hx.inserted_mask(index)
        if (ids.sha256_file(path) != e["sha256"] or m.get(plant_identity.KEY) != v2_sha or int(m["fps"]) != FPS
                or len(index) != e["num_frames"] or not np.array_equal(lineage[f"{stem}.index"], index.numpy())
                or not np.array_equal(lineage[f"{stem}.inserted"], inserted.numpy())
                or any(not torch.equal(m[k], port[k][index]) for k in KINEMATIC)):
            bad_splice.append(stem)
        real = ~inserted
        for k in hx.PRESSURE_FIELDS:
            if not torch.equal(m[k][real], port[k][index][real]) or bool(m[k][inserted].abs().sum() > 0):
                bad_pressure.append(f"{stem}.{k}")
        if e["variant_s"] == 0.0 and ids.sha256_file(path) != ids.sha256_file(port_path(inp, src)):
            bad_x0.append(stem)
        if hx.variant_holds(x0[src]["holds"], plan, FPS) != e["holds"]:
            bad_holds.append(stem)
    c("every variant is its port spliced (kinematics, plant, fps, lineage)", not bad_splice, ", ".join(bad_splice[:3]))
    c("pressure: real frames are the port's measurement, inserted frames zero and invalid", not bad_pressure,
      ", ".join(bad_pressure[:3]))
    c("the d = 0 variants are the gated ports byte for byte", not bad_x0, ", ".join(bad_x0[:3]))
    c("every variant's holds are the manifest's, re-timed, hold ids kept", not bad_holds, ", ".join(bad_holds[:3]))

    # 3. package
    pk = load(out / "motions.pt")
    pk_names = [Path(f).stem for f in pk["motion_files"]]
    c("package order is the extended manifest's", pk_names == names)
    c("package frame counts, fps and plant", [int(n) for n in pk["motion_num_frames"]] == [e["num_frames"] for e in ext["clips"]]
      and bool(torch.allclose(pk["motion_dt"].double(), torch.full_like(pk["motion_dt"].double(), 1.0 / FPS)))
      and pk.get("plant_sha256") == v2_sha)
    starts, bad = pk["length_starts"].tolist(), []
    for i, stem in enumerate(names):
        sl = slice(starts[i], starts[i] + int(pk["motion_num_frames"][i]))
        for key, field in PACKED.items():
            if pk.get(key) is None or not torch.equal(pk[key][sl], motions[stem][field].to(pk[key].dtype)):
                bad.append(f"{stem}.{key}")
                break
    c("package equals the motions frame by frame, the measured channels included", not bad, ", ".join(bad[:3]))
    frames = torch.tensor([int(n) for n in pk["motion_num_frames"]])
    facts["longest_motion_s"] = round(float(frames.max()) / FPS, 3)
    del pk

    # 4. graph v2
    g = load(out / "contact_graph.pt")
    pairs = list(zone_pairs())
    c("graph v2 built from this package and manifest", int(g.get("graph_version", 1)) == 2
      and g["package_sha256"] == ids.sha256_file(out / "motions.pt")
      and g["manifest_sha256"] == ids.sha256_file(out / "holds_extended.yaml") and g.get("plant_sha256") == v2_sha)
    c("graph motion order, frames and fps are the package's", list(g["motion_names"]) == names
      and torch.equal(g["motion_num_frames"].long(), frames) and int(g["fps"]) == FPS)
    c("graph pair and orientation vocabularies", list(g["pair_names"]) == pairs and list(g["orientation_names"]) == list(ORIENT_BINS))
    pidx = {p: i for i, p in enumerate(pairs)}
    bad_seg, bad_a4 = [], []
    for m_, stem in enumerate(names):
        holds = sorted(ext["clips"][m_]["holds"], key=lambda h: float(h["t_hold"]))
        if int(g["seg_count"][m_]) != len(holds):
            bad_seg.append(stem)
            continue
        for k, h in enumerate(holds):
            want = torch.zeros(len(pairs))
            want[[pidx[p] for p in h["pairs"]]] = 1.0
            ground = sorted(p for p in h["pairs"] if p.endswith(":G"))
            node = int(g["seg_node"][m_, k])
            # times are stored as float32: compare with the float32 of the manifest's value, exactly
            times = torch.tensor([float(h["t_start"]), float(h["t_hold"]), float(h["t_end"])], dtype=torch.float32)
            stored = torch.stack([g["seg_start"][m_, k], g["seg_hold"][m_, k], g["seg_end"][m_, k]]).float()
            if (g["hold_ids"][int(g["seg_hold_index"][m_, k])] != h["hold_id"]
                    or not torch.equal(g["seg_contact"][m_, k], want)
                    or g["node_keys"][node] != node_key(h["name"], ground, h["orientation"])
                    or not torch.equal(stored, times)):
                bad_seg.append(f"{stem}@{k}")
            configured = {p for p in h["pairs_configured"] if p.endswith(":G")}
            if set(ground) != configured - set(h["gate"]["masks"]):
                bad_a4.append(h["hold_id"])
    c("graph segments are the manifest's holds (hold id, times, commanded contacts, node key)", not bad_seg,
      ", ".join(bad_seg[:3]))
    c("A4: every commanded ground set is the configured supports minus the gate's masks", not bad_a4,
      ", ".join(bad_a4[:3]))
    graph = ContactGraph(g)
    live = torch.arange(graph.seg_node.shape[1]).unsqueeze(0) < graph.seg_count.unsqueeze(-1)
    mids = torch.arange(len(names)).unsqueeze(-1).expand_as(live)[live]
    times, nodes = graph.seg_hold[live], graph.seg_node[live]
    contact, resolved = graph.manual_contact(nodes, mids, times)
    c("manual goals resolve the side-specific segment (every hold's pose -> its own contact set)",
      bool(resolved.all()) and torch.equal(contact, graph.seg_contact[live]), f"{int(live.sum())} segments")
    desc = json.loads((out / "contact_graph.json").read_text())
    # What v1's manual-goal rule would have served: the ground set plus the body-body pairs in at least half of
    # the node's segments, the duration variants voting once each (build_hold_graph.py, --goal-pair-vote 0.5).
    seg_contact = graph.seg_contact[live]
    body = torch.tensor(["+" in p for p in pairs])
    vote = torch.zeros(graph.num_nodes, len(pairs))
    count = torch.zeros(graph.num_nodes)
    vote.index_add_(0, nodes, seg_contact)
    count.index_add_(0, nodes, torch.ones(len(nodes)))
    v1_rule = torch.where(body, (vote / count.clamp(min=1).unsqueeze(-1) >= 0.5).float(), graph.node_contact)[nodes]
    facts["graph"] = {"nodes": graph.num_nodes, "edges": len(desc["edges"]), "segments": int(live.sum()),
                      "hold_ids": len(graph.hold_ids),
                      "nodes_with_pair_conflicts": sum(1 for n in desc["nodes"] if n["pair_conflicts"]),
                      "segments_v1_vote_served_another_set": int((v1_rule != seg_contact).any(-1).sum()),
                      "segments_node_consensus_differs": int((graph.node_contact[nodes] != seg_contact).any(-1).sum())}

    # 5. physics tables v2
    t = load(out / "physics_tables.pt")
    body_names = mjcf_body_names(str(plant_mjcf()))
    c("tables v2 built with this graph, package and manifest", int(t.get("version", 1)) == 2
      and t["graph_sha256"] == ids.sha256_file(out / "contact_graph.pt")
      and t["package_sha256"] == ids.sha256_file(out / "motions.pt")
      and t["manifest_sha256"] == ids.sha256_file(out / "holds_extended.yaml"))
    c("tables keyed to the package (names, frames, fps), the robot's body order and plant v2",
      list(t["motion_names"]) == names and torch.equal(t["swing_len"].long(), frames) and int(t["fps"]) == FPS
      and list(t["body_names"]) == body_names and t["plant_sha256"] == v2_sha)
    c("tables' pair names, zone order and segment layout are the graph's", list(t["pair_names"]) == pairs
      and list(t["zone_order"]) == list(ZONE_ORDER) and tuple(t["seg_pair_consequential"].shape) == (*g["seg_node"].shape, len(pairs)))
    stale = []
    for m_, stem in enumerate(names):
        ins = torch.from_numpy(lineage[f"{stem}.inserted"])
        src = t["swing_source"][m_, : len(ins)][ins]
        if bool((src >= 2).any()):
            stale.append(stem)
    c("no pressure-sourced swing label on an inserted frame", not stale, ", ".join(stale[:3]))
    facts["tables"] = {"swing_zone_frames": int(t["swing"].sum()), "lean_gated_segments": int(t["seg_lean_gate"].sum()),
                       "cop_valid_segments": int(t["seg_cop_valid"].sum()),
                       "share_valid_segments": int(t["seg_share_valid"].sum())}
    del t

    # 6. contact-target sidecar
    s = load(out / "contact_targets.pt")
    c("sidecar compiled for this graph, package and manifest", s["graph_sha256"] == ids.sha256_file(out / "contact_graph.pt")
      and s["package_sha256"] == ids.sha256_file(out / "motions.pt")
      and s["manifest_sha256"] == ids.sha256_file(out / "holds_extended.yaml") and s.get(plant_identity.KEY) == v2_sha)
    c("sidecar names, pairs, hold ids and frames are the graph's", list(s["motion_names"]) == names
      and list(s["pair_names"]) == pairs and list(s["hold_ids"]) == list(g["hold_ids"])
      and torch.equal(s["seg_hold_index"], g["seg_hold_index"]) and torch.equal(s["frame_len"], frames))
    L = live.unsqueeze(-1)
    commanded = s["seg_commanded"] & L
    c("sidecar: commanded = configured - masks = the graph's seg_contact",
      torch.equal(commanded, (s["seg_configured"] & ~s["seg_masked"]) & L)
      and torch.equal(commanded, (g["seg_contact"] > 0.5) & L) and not bool((s["seg_masked"] & ~s["seg_configured"]).any()))
    summary = json.loads((out / "contact_targets.json").read_text())
    anns = ct.read_annotations(inp["labels_dir"])
    released = [h["hold_id"] for cl in manifest["clips"] for h in cl["holds"]]
    crit = sum(1 for h in released for a in anns.get(h, {}).values()
               if a.get("critical") and a["target_role"] == "required_touch" and a["in_configuration"])
    restored = sum(1 for h in released for a in anns.get(h, {}).values() if a.get("restored"))
    masks = Counter(p for cl in manifest["clips"] for h in cl["holds"] for p in h["gate"]["masks"])
    c("sidecar counts equal the labels' and the gate's on the released holds",
      summary["critical"] == crit and summary["restored"] == restored and summary["masked"] == dict(masks),
      f"critical {crit}, restored {restored}, masks {dict(masks)}")
    facts["sidecar"] = {k: summary[k] for k in ("holds", "roles", "configured_body_body", "critical", "restored",
                                                 "masked", "flags", "pose_roles", "known_free_zone_holds",
                                                 "free_unknown", "frame_pair_partly_realised")}
    del s

    # 7. the runtime loaders accept them (the constructors training uses, on the CPU)
    try:
        rg = ContactGraph.from_file(out / "contact_graph.pt")
        rg.validate_against_motion_lib([f"{n}.motion" for n in names], motion_num_frames=frames, fps=FPS)
        rt = PhysicsTables(str(out / "physics_tables.pt"), names, body_names, "cpu", plant_mjcf=str(plant_mjcf()),
                           motion_num_frames=frames, fps=FPS)
        ok = rt.version == 2 and rt.pair_names == rg.pair_names and rt.graph_sha256 == ids.sha256_file(out / "contact_graph.pt")
        ContactTargets(str(out / "contact_targets.pt"), rg, names, "cpu", motion_num_frames=frames, fps=FPS,
                       graph_sha256=ids.sha256_file(out / "contact_graph.pt"), plant_mjcf=str(plant_mjcf()))
        c("runtime loaders accept the graph, tables and sidecar", ok)
    except Exception as exc:  # noqa: BLE001 -- a refusal is a failed check
        c("runtime loaders accept the graph, tables and sidecar", False, f"{type(exc).__name__}: {exc}")

    # 8. probe plans resolve against the release's graph
    plans = sorted((out / "plans").glob("*.json"))
    unresolved = []
    keys = set(g["node_keys"])
    for p in plans:
        plan = json.loads(p.read_text())
        if plan["start"]["clip"] not in names or any(goal["config"] not in keys or goal["pose_clip"] not in names
                                                      for goal in plan["goals"]):
            unresolved.append(p.name)
    c("probe plans resolve against the release's graph and library", bool(plans) and not unresolved,
      f"{len(plans)} plans; " + ", ".join(unresolved[:3]))
    facts["plans"] = len(plans)
    return c.passed, c.failed, facts


# --------------------------------------------------------------------------- #
# The swing term's calibration on the new tables (TODO C3)
# --------------------------------------------------------------------------- #
def calibrate_swing(out: Path, inp: dict) -> dict:
    """The swing labels against the human evidence, on the x0 motions (one per clip).

    The term charges the policy's ground load on a zone the reference swings. A correct label is a zone the human
    moves through the air or holds unloaded, so on a labelled zone-frame the human's own evidence should say
    *separated* (the mesh, markers or mat: ``sources.ground_source``) or, where the mat can see the zone (column 1
    and attribution-visible), carry under 14 N. The share of labelled zone-frames contradicting that is the label's
    false-positive rate; the human's measured load on labelled frames is what a perfect imitator would be charged.
    """
    t = load(out / "physics_tables.pt")
    stats = json.loads((out / "physics_tables.json").read_text())
    ext = yaml.safe_load(open(out / "holds_extended.yaml"))
    names = [e["stem"] for e in ext["clips"]]
    rows = {"velocity": Counter(), "pressure": Counter(), "both": Counter()}
    load_on_label = []
    from build_physics_tables import PRESSURE_ZONES
    from extract_contact_configs import ZONES

    zone_cols = {z: i for i, z in enumerate(ZONE_ORDER)}
    for m, stem in enumerate(names):
        if float(ext["clips"][m]["variant_s"]) != 0.0:
            continue
        T = int(ext["clips"][m]["num_frames"])
        src = t["swing_source"][m, :T].numpy()
        rec = capture_v4.load(stem, rebuild=False)
        state, _ = sources.ground_source(rec)
        col1 = np.nan_to_num(rec["mat_valid_body"], nan=0.0) >= 0.9
        visible = rec["attr_visible"]
        zone_load = np.nan_to_num(rec["mat_zone_load"], nan=0.0)
        for kind, code in (("velocity", 1), ("pressure", 2), ("both", 3)):
            sel = src == code
            rows[kind]["zone_frames"] += int(sel.sum())
            rows[kind]["human_separated"] += int((sel & (state == 0)).sum())
            rows[kind]["human_contact"] += int((sel & (state == 1)).sum())
            rows[kind]["human_unknown"] += int((sel & (state == -1)).sum())
            seen = sel & col1[:, None] & visible
            rows[kind]["mat_visible"] += int(seen.sum())
            rows[kind]["mat_over_14n"] += int((seen & (zone_load > 14.0)).sum())
            rows[kind]["mat_over_70n"] += int((seen & (zone_load > 70.0)).sum())
        labelled = src > 0
        for z in PRESSURE_ZONES:
            zi = zone_cols[z]
            seen = labelled[:, zi] & col1 & visible[:, zi]
            load_on_label.append(zone_load[seen, zi])
    loads = np.concatenate(load_on_label) if load_on_label else np.zeros(0)

    def rate(r):
        n = max(r["zone_frames"], 1)
        return {**dict(r), "human_contact_share": round(r["human_contact"] / n, 4),
                "human_separated_share": round(r["human_separated"] / n, 4),
                "mat_over_14n_share_of_visible": round(r["mat_over_14n"] / max(r["mat_visible"], 1), 4)}

    baselines = {}
    for name, path in TABLE_BASELINES.items():
        if path.exists():
            b = json.loads(path.read_text())
            baselines[name] = {k: b.get(k) for k in ("velocity_only", "pressure_only", "vetoed", "frames")}
    return {
        "rules": stats["rules"],
        "all_motions": {k: stats[k] for k in ("frames", "labels", "velocity_only", "pressure_only", "both", "vetoed",
                                              "labels_v1_rule", "pressure_only_v1_rule", "vetoed_v1_rule",
                                              "unloaded_invisible_frames")},
        "baselines_180_motions": baselines,
        "x0_against_human_evidence": {k: rate(v) for k, v in rows.items()},
        "human_load_on_labelled_feet_hands_n": {
            "frames": int(len(loads)),
            "p50": round(float(np.median(loads)), 2) if len(loads) else None,
            "p90": round(float(np.percentile(loads, 90)), 2) if len(loads) else None,
            "p99": round(float(np.percentile(loads, 99)), 2) if len(loads) else None,
            "max": round(float(loads.max()), 2) if len(loads) else None},
        "zones_note": f"pressure rule on {list(PRESSURE_ZONES)}; zone bodies {dict((z, ZONES[z]) for z in PRESSURE_ZONES)}",
    }


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
def record(rid: str, inp: dict, out: Path, passed: list, facts: dict, calibration: dict, seconds: float) -> dict:
    artifacts = {role: {"path": ids.display_path(out / name), "sha256": ids.sha256_file(out / name),
                        "bytes": (out / name).stat().st_size} for role, name in ARTIFACTS.items()}
    ext = yaml.safe_load(open(out / "holds_extended.yaml"))
    plans = {p.name: ids.sha256_file(p) for p in sorted((out / "plans").glob("*.json"))}
    inputs = [inp["manifest_path"], inp["gate_record"], inp["labels_dir"] / "annotations.jsonl",
              inp["labels_dir"] / "holds.yaml", inp["retarget_record_path"], inp["pressure_record_path"],
              plant_mjcf(), plant_identity.flat_path(PLANT)]
    groups = Counter(cl["group"] for cl in inp["manifest"]["clips"])
    longest = facts["longest_motion_s"]
    return {
        **ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs),
        "kind": "reference_release", "release_id": rid, "release_version": RELEASE_VERSION,
        "plant": inp["plant"], "robot": "smpl_yogi_v2",
        "gate_id": inp["gate_id"], "labels_id": inp["labels_id"], "statics_id": inp["statics_id"],
        "retarget_id": inp["retarget_id"], "fit_id": inp["fit_id"],
        "dir": ids.display_path(out), "parameters": parameters(),
        "builders": {ids.display_path(p): ids.sha256_file(p) for p in BUILDERS},
        "artifacts": artifacts, "plans": plans,
        "motions": {e["stem"]: e["sha256"] for e in ext["clips"]},
        "counts": {"clips": len(inp["manifest"]["clips"]), "groups": dict(groups),
                   "holds": sum(len(cl["holds"]) for cl in inp["manifest"]["clips"]),
                   "extendable_holds": sum(1 for cl in inp["manifest"]["clips"] for h in cl["holds"] if h.get("extend")),
                   **facts["extension"], "graph": facts["graph"], "tables": facts["tables"],
                   "sidecar": facts["sidecar"], "plans": facts["plans"]},
        "training": {"robot": "smpl_yogi_v2", "longest_motion_s": longest,
                     "eval_max_steps": int(math.ceil(longest * CONTROL_HZ)),
                     "note": "the full-clip evaluator must cover the longest motion: eval_max_steps >= this"},
        "checks": {"passed": passed, "failed": []},
        "calibration_c3": calibration,
        "seconds": round(seconds, 1),
    }


def summary_markdown(rec: dict) -> str:
    c = rec["counts"]
    cal = rec["calibration_c3"]
    a = cal["all_motions"]
    x = cal["x0_against_human_evidence"]
    lines = [f"# Release `{rec['release_id']}`", "",
             f"Generated by `{MODULE}` (BodyFix Step 5 / BUILD_PLAN Steps 9-10) from gate `{rec['gate_id']}` (labels "
             f"`{rec['labels_id']}`), the Step 3 references `{rec['retarget_id']}` and plant v2 "
             f"(`{rec['plant']['plant_sha256'][:12]}`). Heavy files: `{rec['dir']}/` (untracked).", "",
             "| | |", "|---|---|",
             f"| clips / holds / extendable | {c['clips']} / {c['holds']} / {c['extendable_holds']} ({c['groups']}) |",
             f"| motions / frames / inserted | {c['motions']} / {c['frames']} / {c['inserted_frames']} |",
             f"| graph | {c['graph']['nodes']} nodes, {c['graph']['edges']} edges, {c['graph']['segments']} segments, "
             f"{c['graph']['hold_ids']} hold ids; {c['graph']['nodes_with_pair_conflicts']} nodes with side-specific "
             f"pair conflicts; a manual goal at a hold's own pose would have been served another contact set by v1's "
             f"0.5-vote node rule on {c['graph']['segments_v1_vote_served_another_set']} segments (v2 serves each "
             f"segment's own; the node consensus differs from it on {c['graph']['segments_node_consensus_differs']}) |",
             f"| tables | {c['tables']} |",
             f"| sidecar | critical {c['sidecar']['critical']}, restored {c['sidecar']['restored']}, masks "
             f"{c['sidecar']['masked']}, flags {c['sidecar']['flags']}, known-free zone-holds "
             f"{c['sidecar']['known_free_zone_holds']} |",
             f"| training | robot `{rec['training']['robot']}`, longest motion {rec['training']['longest_motion_s']} s -> "
             f"eval_max_steps >= {rec['training']['eval_max_steps']} |",
             f"| probe plans | {c['plans']} (`plans/`, from the release's own graph) |",
             "", f"Identity checks: {len(rec['checks']['passed'])} passed, 0 failed.", "",
             "## Swing labels (TODO C3)", "",
             f"All {a['frames']} frames: {a['labels']} labelled zone-frames (v1's pressure rule on the same motions: "
             f"{a['labels_v1_rule']}); velocity-only {a['velocity_only']}, pressure-only {a['pressure_only']} (v1 rule "
             f"{a['pressure_only_v1_rule']}), both {a['both']}, vetoed {a['vetoed']} (v1 rule {a['vetoed_v1_rule']}); "
             f"{a['unloaded_invisible_frames']} v1 'unloaded' frames were invisible to the attribution.", "",
             "| x0 labels by source | zone-frames | human separated | human in contact | mat-visible | > 14 N on the mat |",
             "|---|---|---|---|---|---|"]
    for k, r in x.items():
        lines.append(f"| {k} | {r['zone_frames']} | {r['human_separated_share']:.3f} | {r['human_contact_share']:.3f} | "
                     f"{r['mat_visible']} | {r['mat_over_14n_share_of_visible']:.3f} |")
    h = cal["human_load_on_labelled_feet_hands_n"]
    lines += ["", f"The human's measured load on labelled feet/hands (mat-visible): p50 {h['p50']} N, p90 {h['p90']} N, "
              f"p99 {h['p99']} N, max {h['max']} N over {h['frames']} zone-frames.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="build, check and record the release")
    ap.add_argument("--check", metavar="RELEASE_ID", help="re-run the identity checks on a recorded release")
    ap.add_argument("--gate-id", default=GATE_ID)
    ap.add_argument("--force", action="store_true", help="delete a failed build's folder (one without a record)")
    ap.add_argument("--print-id", action="store_true", help="print the id the inputs and builders determine")
    args = ap.parse_args(argv)
    try:
        if args.print_id:
            print(release_id(pinned_inputs(args.gate_id)))
            return 0
        if args.build:
            out, rec = build(args.gate_id, force=args.force)
            c = rec["counts"]
            print(f"release {rec['release_id']}: {c['clips']} clips, {c['holds']} holds, {c['motions']} motions, "
                  f"{c['graph']['nodes']} nodes; {len(rec['checks']['passed'])} identity checks passed in "
                  f"{rec['seconds']:.0f} s -> {ids.display_path(RECORD_ROOT / (rec['release_id'] + '.json'))}")
            return 0
        if args.check:
            rec = json.loads((RECORD_ROOT / f"{args.check}.json").read_text())
            inp = pinned_inputs(rec["gate_id"])
            now = release_id(inp)
            if now != rec["release_id"]:
                # A frozen release stays valid; only a rebuild would differ. Its artifacts are checked below.
                print(f"note: today's inputs and builders determine {now}; this release was built by the recorded ones")
            out = REPO / rec["dir"]
            passed, failed, _ = check(out, inp)
            for role, a in rec["artifacts"].items():
                if ids.sha256_file(REPO / a["path"]) != a["sha256"]:
                    failed.append(f"artifact {role} changed since the record")
            for f in failed:
                print(f"FAILED  {f}", file=sys.stderr)
            print(f"release {rec['release_id']}: {len(passed)} identity checks passed, {len(failed)} failed")
            return 1 if failed else 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
