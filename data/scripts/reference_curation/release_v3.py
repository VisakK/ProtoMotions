# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card R3 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: release v3, release v2 plus the spliced synthetic
clips, built by a wrapper around v2's builders.

Inputs, pinned by id and content (they and the builders' sources make the release id)
--------------------------------------------------------------------------------------
* **release v2** ``holds_repaired_ftC_posefix.release_v2.a2dda5d2ac``, by its record's sha256 (``RELEASE_V2_RECORD_SHA256``);
  every v2 file read is checked against that record;
* **the synthetic clips** ``synthetic_v3.d034199f23`` (card R2, ``splice_v3``), by its record's sha256; every clip
  file (``.motion``, ``.holds.yaml``, ``.lineage.npz``, ``.json``) is checked against that record;
* **the drop list**, matched by exact stem (``apply_drop``; the synthetic stems contain each other, ``t6px`` <
  ``t6px12``), which must hold the user's pre-G3 drop (``USER_DROP``), and the **purpose** (``candidate``: every D1
  variant but the drop; ``final``: the candidate minus G2's failures);
* the D1 policy and the user's T6 decision, as the synthetic record carries them.

The chain (release v2's ``build``, ``release_v2.py:277-323``, wrapped)
---------------------------------------------------------------------
1. **Manifest** (``release_holds.yaml``): release v2's, its clips first, then the kept synthetic clips' manifest
   entries (``<stem>.holds.yaml``) in the synthetic record's order (edge, timing, seed, tag). v2's motion ids stay
   0-167; every ``SYN_`` hold id sorts after every ``220923_`` one, so v2's hold indices stay too.
2. **Sources.** Human clips: v2's ``verify_sources``. Synthetic clips: sha256 against the synthetic record.
3. **Extension.** v2's 168 ``.motion`` files are copied byte for byte (sha256 = v2's record) with their manifest
   entries and lineage arrays. Each synthetic clip is extended by ``hold_extension_v2`` exactly as v2's
   ``extend_corpus`` extends a human one: x0 is R2's file byte for byte, x3s and x7s insert at S (the clip's only
   ``extend`` hold), after ``velocity_drift``'s guard. ``lineage.npz`` keeps v2's arrays and adds, per synthetic
   variant, ``index`` and ``inserted`` (v2's meaning: the x0 frame, inserted or not) and R2's per-frame lineage read
   through ``index``: ``kind``, ``source``, ``source_frame``, ``variant_t`` and ``stems``.
4. **Package** (``package_motion_subset.py``): v2's 168 motions in v2's order, then the synthetic ones, weights by
   ``--weights``: 1.0 per human motion, 3/n per motion of an edge with n kept variants (D5: each edge weighs three
   human clips' x0/x3s/x7s).
5. **Graph** (``build_hold_graph_v2.py``, unchanged).
6. **Tables** (``physics_tables_v3``: ``build_physics_tables_v2.py`` with the synthetic clips on its no-pressure path).
7. **Sidecar** (``contact_targets_v3``: v2's ``compile_targets`` for the human motions, inherited rows for the
   synthetic ones).
8. **Plans**: ``make_hold_graph_probe_plans.py``, unchanged, then ``make_edge_probe_plans.py`` (R4a).
9. **Checks** (``check``), then ``calibrate_swing`` on the human x0 stems (it must equal v2's ``calibration_c3``),
   then the record (written last, only when everything passes).

Checks
------
* **Release v2 passes its own 26 checks** (``release_v2.check`` on v2) and its artifacts are its record's.
* **The human part is release v2, slice for slice**: manifest entries, motion files, lineage arrays, package
  entries (every per-motion value and per-frame slice), graph rows and node tables, table rows, sidecar rows and
  release v2's sidecar summary, bit for bit after padding.
* **Release v2's 26 checks, on the whole of v3** (each restated for synthetic lineage where v2's reads human
  evidence): sources, extension, package, graph (A4 over the inherited hold ids), tables, sidecar (its counts over
  the inherited hold ids), the runtime loaders, the plans.
* **Synthetic lineage**: R2's per-clip checks re-run on the clips as packaged (real frames against release v2 after
  the recorded transform, transition frames against the admitted variant's re-export, the windows' content, the
  holds), x3s / x7s = hold extension of x0, pressure zero and invalid on every frame.
* **Inherited labels**: every synthetic hold's copied fields and every sidecar row equal its ``inherits`` hold's.
* **Graph**: the node-key set is release v2's (no id moves); ``pair_names`` and ``orientation_names`` equal v2's in
  order; each kept edge's S -> D is present once per synthetic motion; no other edge is new.
* **Plant**: every motion, the package, the graph, the tables and the sidecar carry plant v2's sha256.
* **Plans**: v3's family plans equal v2's file for file; the edge plans are R4a's (the plan clip per edge by its
  rule) and resolve with SequenceViz's own loader within the launcher's 24 s cap.
* **Record**: ``training.eval_max_steps`` covers the longest motion.

Run in a plant-v1 process (no ``REFERENCE_PLANT``), single-threaded::

    PYTHONPATH=.:data/scripts OMP_NUM_THREADS=1 ../env_isaaclab/bin/python -m reference_curation.release_v3 --build
    PYTHONPATH=.:data/scripts OMP_NUM_THREADS=1 ../env_isaaclab/bin/python -m reference_curation.release_v3 --check <id>
    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.release_v3 --print-id [--drop ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from extract_contact_configs import ORIENT_BINS, ZONE_ORDER, mjcf_body_names, zone_pairs
from reference_curation import capture_v4, hold_extension_v2 as hx, ids
from reference_curation import contact_targets_v2 as ct
from reference_curation import contact_targets_v3 as ct3
from reference_curation import physics_tables_v3 as tables_v3
from reference_curation import release_v2 as R2
from reference_curation import splice_v3 as S3

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.release_v3"
SCHEMA_VERSION = 1
RELEASE_VERSION = "release_v3"
PLANT = "v2"
FPS = R2.FPS
VARIANTS = R2.VARIANTS
MIN_LEAD_S = R2.MIN_LEAD_S
CONTROL_HZ = R2.CONTROL_HZ
RELEASE_V2_ID = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
RELEASE_V2_RECORD_SHA256 = "10dc501f7213d1c26ab1811ce1a8fe5e6f2aeaf1d9ec3cace834d8605d79a3ff"
SYNTHETIC_ID = "synthetic_v3.d034199f23"
SYNTHETIC_RECORD_SHA256 = "20d6c3f71914dc1e4da4de5801e6a2d8c34706552e35c7208b3d0d25562ae058"
# The user's decision (2026-10-04, after R2): this clip lands knee down; every build drops it.
USER_DROP = ("SYN_E2_jumpback_mid_s3_t6rpx12",)
PURPOSES = ("candidate", "final")
EDGE_CLIPS = 3            # D5: each edge carries the sampling mass of three human clips (x0, x3s and x7s each)
EDGE_ORDER = S3.EDGE_ORDER
HEAVY_ROOT = R2.HEAVY_ROOT
RECORD_ROOT = R2.RECORD_ROOT
SCRIPTS = R2.SCRIPTS
ARCHIVES = R2.ARCHIVES
ARTIFACTS = R2.ARTIFACTS
SYN_LINEAGE = ("kind", "source", "source_frame", "variant_t")     # R2's per-frame lineage, read through index
VIZ_MAX_SECONDS = 24.0    # run_expert_graph_ft.sh: a panel plan longer than this loses goals
# The card's R4a picks under T6's selection (PLAN.MD R4a): the rule must reproduce them while all are kept.
CARD_PLAN_CLIPS = {"E1": "SYN_E1_press_high_s0_t6px", "E3": "SYN_E3_lower_high_s3_t6px12",
                   "B1": "SYN_B1_jumpplank_high_s1_t6rpx", "E2": "SYN_E2_jumpback_mid_s0_t6px12",
                   "E5": "SYN_E5_floatdown_mid_s0_t6rpx12"}
# Tables fields that name files (they differ between releases by construction).
TABLE_PATH_PARAMS = ("extended_manifest", "graph", "motion_dir", "motion_file", "out", "archive_dir", "mjcf")


# --------------------------------------------------------------------------- #
# Pure rules (unit-tested)
# --------------------------------------------------------------------------- #
def apply_drop(order: list[str], drop) -> list[str]:
    """``order`` without the stems in ``drop``, matched exactly (never by substring: ``..._t6px`` is a prefix of
    ``..._t6px12``). An unknown or repeated stem is refused."""
    drop = list(drop)
    unknown = [d for d in drop if d not in order]
    if unknown:
        raise ValueError(f"--drop names stems the synthetic record does not hold (exact match only): {unknown}")
    if len(set(drop)) != len(drop):
        raise ValueError(f"--drop repeats a stem: {sorted(s for s in drop if drop.count(s) > 1)}")
    return [s for s in order if s not in set(drop)]


def check_purpose(purpose: str, drop) -> None:
    """Both purposes drop the user's ``USER_DROP``; a final release drops at least one more (G2's failures)."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    missing = [s for s in USER_DROP if s not in drop]
    if missing:
        raise ValueError(f"the drop list must hold the user's pre-G3 drop {missing} (PLAN.MD, 2026-10-04)")
    if purpose == "final" and set(drop) == set(USER_DROP):
        raise ValueError("a final release drops G2's failures on top of the candidate's; with none, the candidate "
                         "is the final (PLAN.MD D3)")


def edge_weights(edges: dict[str, list[str]]) -> dict[str, float]:
    """``{edge: per-motion weight}``: 3/n for an edge with n kept variants (D5)."""
    return {e: EDGE_CLIPS / len(v) for e, v in edges.items() if v}


def motion_weights(clips: list[dict], weights: dict[str, float]) -> list[float]:
    """Per extended-manifest entry: 1.0 for a human motion, its edge's weight for a synthetic one."""
    return [weights[c["synthetic"]["edge"]] if c.get("synthetic") else 1.0 for c in clips]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def plant_mjcf() -> Path:
    return plant_identity.mjcf_path(PLANT)


def load(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def v2_record_path() -> Path:
    return RECORD_ROOT / f"{RELEASE_V2_ID}.json"


def synthetic_record_path() -> Path:
    return S3.RECORD_ROOT / f"{SYNTHETIC_ID}.json"


def pinned_inputs(drop=USER_DROP, purpose: str = "candidate") -> dict:
    """Every input, resolved and checked against its pin."""
    v2_path, syn_path = v2_record_path(), synthetic_record_path()
    for path, want in ((v2_path, RELEASE_V2_RECORD_SHA256), (syn_path, SYNTHETIC_RECORD_SHA256)):
        if ids.sha256_file(path) != want:
            raise ValueError(f"{ids.display_path(path)} has sha256 {ids.sha256_file(path)[:12]}, pinned {want[:12]}")
    v2 = json.loads(v2_path.read_text())
    syn = json.loads(syn_path.read_text())
    if v2["release_id"] != RELEASE_V2_ID or syn["synthetic_id"] != SYNTHETIC_ID:
        raise ValueError("the pinned records name other ids")
    if syn["release_v2"] != {"release_id": RELEASE_V2_ID, "record": ids.display_path(v2_path),
                             "sha256": RELEASE_V2_RECORD_SHA256}:
        raise ValueError(f"{SYNTHETIC_ID} was spliced from another release v2 record: {syn['release_v2']}")
    if syn["checks"]["failed"]:
        raise ValueError(f"{SYNTHETIC_ID}'s record lists failed checks")
    drop = sorted(drop)
    check_purpose(purpose, drop)
    kept = apply_drop(syn["order"], drop)
    edges_path = REPO / "expert_revist/graph_growth_2026_10_03/edges.json"
    if ids.sha256_file(edges_path) != syn["key"]["inputs"][ids.display_path(edges_path)]:
        raise ValueError("edges.json is not the one the synthetic clips were spliced against")
    entries = {}
    for stem in kept:
        hp = REPO / syn["clips"][stem]["holds"]["path"]
        if ids.sha256_file(hp) != syn["clips"][stem]["holds"]["sha256"]:
            raise ValueError(f"{ids.display_path(hp)} is not the synthetic record's (sha256)")
        entries[stem] = yaml.safe_load(open(hp))
    by_edge = {e: [s for s in kept if entries[s]["synthetic"]["edge"] == e] for e in EDGE_ORDER}
    return {
        "v2": v2, "v2_record_path": v2_path, "v2_record_sha256": RELEASE_V2_RECORD_SHA256, "v2_out": REPO / v2["dir"],
        "v2_inp": R2.pinned_inputs(v2["gate_id"]),
        "syn": syn, "syn_record_path": syn_path, "syn_record_sha256": SYNTHETIC_RECORD_SHA256,
        "drop": drop, "purpose": purpose, "kept": kept, "entries": entries,
        "by_edge": {e: v for e, v in by_edge.items() if v}, "edges_path": edges_path,
        "edges": json.loads(edges_path.read_text()), "plant": plant_identity.identity(PLANT),
    }


def builders() -> list[Path]:
    """The code whose output the release is: this wrapper, its three new modules, release v2's builders (the human
    part is their output) and the kernels that make the synthetic sidecar rows."""
    from reference_curation import fit_writer, human_mesh, mosh_replay, retarget, retarget_v2

    return [Path(__file__), Path(ct3.__file__), Path(tables_v3.__file__), SCRIPTS / "make_edge_probe_plans.py",
            *R2.BUILDERS, Path(capture_v4.__file__), Path(retarget.__file__), Path(retarget_v2.__file__),
            Path(fit_writer.__file__), Path(mosh_replay.__file__), Path(human_mesh.__file__)]


def parameters() -> dict:
    return {**R2.parameters(), "release_v2": RELEASE_V2_ID, "synthetic_id": SYNTHETIC_ID,
            "order": "release v2's motions in its order, then the kept synthetic clips in the synthetic record's "
                     "order (edge, timing, seed, tag), each as x0, x3s, x7s",
            "drop_match": "exact stem", "edge_clips": EDGE_CLIPS,
            "weights": "1.0 per human motion; 3/n per motion of an edge with n kept variants (D5)",
            "synthetic_extension": "hold_extension_v2 at S, the clip's only extend hold; x0 = R2's file byte for byte",
            "tables": "build_physics_tables_v2.py via physics_tables_v3 (synthetic clips: no-pressure path)",
            "sidecar": ct3.RULES, "plans": "make_hold_graph_probe_plans.py, then make_edge_probe_plans.py (R4a)"}


def id_key(inp: dict) -> dict:
    return {"schema": SCHEMA_VERSION, "version": RELEASE_VERSION,
            "release_v2": {"release_id": RELEASE_V2_ID, "record_sha256": inp["v2_record_sha256"]},
            "synthetic": {"synthetic_id": SYNTHETIC_ID, "record_sha256": inp["syn_record_sha256"]},
            "drop": list(inp["drop"]), "purpose": inp["purpose"],
            "d1_policy": inp["syn"]["d1_policy"], "t6_decision": inp["syn"]["t6_decision"],
            "builders": {ids.display_path(p): ids.sha256_file(p) for p in builders()},
            "parameters": parameters()}


def release_id(inp: dict) -> str:
    """``<labels stem>.release_v3.<sha256_json(id_key)[:10]>``."""
    stem = inp["v2"]["labels_id"].split(".labels_")[0]
    return f"{stem}.{RELEASE_VERSION}.{ids.sha256_json(id_key(inp))[:10]}"


def verify_synthetic(inp: dict) -> list[str]:
    """Every kept synthetic clip's four files are the synthetic record's; its ``.motion`` is on plant v2 at 60 fps
    with its manifest's frames, and its manifest names that file."""
    problems = []
    sha = inp["plant"][plant_identity.KEY]
    for stem in inp["kept"]:
        rec = inp["syn"]["clips"][stem]
        bad = [k for k in ("motion", "holds", "lineage", "json")
               if ids.sha256_file(REPO / rec[k]["path"]) != rec[k]["sha256"]]
        if bad:
            problems.append(f"{stem}: {bad} are not the synthetic record's")
            continue
        entry = inp["entries"][stem]
        if Path(entry["source"]).resolve() != (REPO / rec["motion"]["path"]).resolve():
            problems.append(f"{stem}: the manifest's source is not the recorded motion")
        m = load(REPO / rec["motion"]["path"])
        if m.get(plant_identity.KEY) != sha or int(m["fps"]) != FPS or int(m["dof_pos"].shape[0]) != entry["num_frames"]:
            problems.append(f"{stem}: not plant v2 / {FPS} fps / the manifest's {entry['num_frames']} frames")
    return problems


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def manifest_v3(inp: dict) -> dict:
    """Release v2's manifest, then the kept synthetic entries, with a ``release_v3`` block before the clips."""
    v2m = inp["v2_inp"]["manifest"]
    out = {k: v for k, v in v2m.items() if k != "clips"}
    out["release_v3"] = {"release_v2": RELEASE_V2_ID, "synthetic_id": SYNTHETIC_ID,
                         "synthetic_record_sha256": inp["syn_record_sha256"], "purpose": inp["purpose"],
                         "drop": list(inp["drop"]), "clips": f"release v2's {len(v2m['clips'])}, then "
                                                             f"{len(inp['kept'])} synthetic"}
    out["clips"] = list(v2m["clips"]) + [inp["entries"][s] for s in inp["kept"]]
    return out


def extend_corpus(inp: dict, out: Path) -> dict:
    """``motions/``, ``holds_extended.yaml`` and ``lineage.npz`` (module doc, step 3). Returns facts."""
    v2, v2_out = inp["v2"], inp["v2_out"]
    motions_dir = out / "motions"
    motions_dir.mkdir(parents=True)
    v2_ext_path = v2_out / "holds_extended.yaml"
    if ids.sha256_file(v2_ext_path) != v2["artifacts"]["holds_extended"]["sha256"]:
        raise ValueError("release v2's holds_extended.yaml is not its record's")
    v2_ext = yaml.safe_load(open(v2_ext_path))
    emitted = []
    for e in v2_ext["clips"]:
        src = v2_out / "motions" / f"{e['stem']}.motion"
        if ids.sha256_file(src) != v2["motions"][e["stem"]]:
            raise ValueError(f"{ids.display_path(src)} is not release v2's")
        shutil.copyfile(src, motions_dir / src.name)
        emitted.append(e)
    with np.load(v2_out / "lineage.npz") as z:
        lineage = {k: z[k] for k in z.files}
    drift_max = {k: [0.0, 0.0] for k in ("dof_vel", "rigid_body_vel", "rigid_body_ang_vel")}
    for stem in inp["kept"]:
        rec, entry = inp["syn"]["clips"][stem], inp["entries"][stem]
        src = REPO / rec["motion"]["path"]
        motion = load(src)
        for k, (mean, peak) in hx.velocity_drift(motion).items():
            drift_max[k] = [max(drift_max[k][0], mean), max(drift_max[k][1], peak)]
            if mean > hx.MAX_VELOCITY_DRIFT:
                raise ValueError(f"{stem}: recomputed {k} drifts {mean:.4f} from the stored one")
        with np.load(REPO / rec["lineage"]["path"]) as z:
            lin = {k: z[k] for k in z.files}
        n0 = int(motion["rigid_body_pos"].shape[0])
        for d in VARIANTS:
            plan = hx.insertion_plan(entry["holds"], int(round(d * FPS)))
            name = hx.variant_stem(stem, d)
            path = motions_dir / f"{name}.motion"
            if d == 0.0:
                shutil.copyfile(src, path)             # x0 is R2's file, byte for byte
                index, inserted = torch.arange(n0), torch.zeros(n0, dtype=torch.bool)
            else:
                variant, index, inserted = hx.extend_motion_v2(motion, plan)
                torch.save(variant, path)
            n = int(index.shape[0])
            emitted.append({
                **{k: v for k, v in entry.items() if k not in ("holds", "source", "num_frames", "length_s")},
                "stem": name, "source_stem": stem, "source": entry["source"], "pressure_port": None,
                "variant_s": float(d), "inserted_frames": int(sum(k for _, k in plan)), "num_frames": n,
                "length_s": round(n / FPS, 3), "sha256": ids.sha256_file(path),
                "holds": hx.variant_holds(entry["holds"], plan, FPS),
            })
            idx = index.numpy()
            lineage[f"{name}.index"] = idx.astype(np.int32)
            lineage[f"{name}.inserted"] = inserted.numpy()
            for k in SYN_LINEAGE:
                lineage[f"{name}.{k}"] = lin[k][idx]
            lineage[f"{name}.stems"] = lin["stems"]
    payload = {
        "version": v2_ext["version"], "builder": f"{MODULE} (hold_extension_v2)",
        "source_manifest": ids.display_path(out / "release_holds.yaml"),
        "source_manifest_sha256": ids.sha256_file(out / "release_holds.yaml"),
        "release_v2_extended_sha256": v2["artifacts"]["holds_extended"]["sha256"],
        "variants_s": [float(v) for v in VARIANTS],
        "pressure": v2_ext["pressure"] + "; synthetic clips: zeros with every validity column 0 on every frame",
        "velocity_drift_max": v2_ext["velocity_drift_max"],
        "synthetic_velocity_drift_max": {k: {"mean": round(v[0], 6), "max": round(v[1], 5)}
                                         for k, v in drift_max.items()},
        "clips": emitted,
    }
    with open(out / "holds_extended.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=120)
    np.savez_compressed(out / "lineage.npz", **lineage)
    syn = [e for e in emitted if e.get("synthetic")]
    return {"motions": len(emitted), "frames": sum(e["num_frames"] for e in emitted),
            "inserted_frames": sum(e["inserted_frames"] for e in emitted),
            "synthetic_motions": len(syn), "synthetic_frames": sum(e["num_frames"] for e in syn),
            "synthetic_inserted_frames": sum(e["inserted_frames"] for e in syn),
            "synthetic_velocity_drift_max": payload["synthetic_velocity_drift_max"]}


def run(args: list, log) -> str:
    """A builder in a fresh plant-v1 process (``python <script> ...`` or ``python -m <module> ...``)."""
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{SCRIPTS}", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    env.pop("REFERENCE_PLANT", None)
    cmd = [sys.executable, *map(str, args)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    log.write(f"$ {' '.join(cmd)}\n[{time.time() - start:.0f} s, exit {proc.returncode}]\n{proc.stdout[-20000:]}\n"
              f"{proc.stderr[-20000:]}\n")
    log.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, args[:2]))} exited {proc.returncode}: {proc.stderr[-1500:]}")
    return proc.stdout


def build(drop=USER_DROP, purpose: str = "candidate", force: bool = False, log_fn=print) -> tuple[Path, dict]:
    """Build, check and record the release; refuse to touch an existing one."""
    capture_v4.require_v1_process()
    start = time.time()
    inp = pinned_inputs(drop, purpose)
    problems = R2.verify_sources(inp["v2_inp"]) + verify_synthetic(inp)
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
    log_fn(f"building {rid} ({purpose}; {len(inp['kept'])} synthetic clips, drop {inp['drop']})")
    facts = {"release_id": rid}
    with open(out / "release_holds.yaml", "w") as f:
        yaml.safe_dump(manifest_v3(inp), f, sort_keys=False, width=120)
    with open(out / "build.log", "w") as log:
        facts["extension"] = extend_corpus(inp, out)
        log_fn(f"extension: {facts['extension']['motions']} motions [{time.time() - start:.0f} s]")
        ext = yaml.safe_load(open(out / "holds_extended.yaml"))
        files = [out / "motions" / f"{c['stem']}.motion" for c in ext["clips"]]
        weights = motion_weights(ext["clips"], edge_weights(inp["by_edge"]))
        run([SCRIPTS / "package_motion_subset.py", "--force", "--out", out / "motions.pt", "--yaml",
             out / "motions.yaml", *files, "--weights", *weights], log)
        log_fn(f"package [{time.time() - start:.0f} s]")
        run([SCRIPTS / "build_hold_graph_v2.py", "--manifest", out / "holds_extended.yaml", "--motion-file",
             out / "motions.pt", "--out-dir", out, "--min-lead-s", MIN_LEAD_S], log)
        log_fn(f"graph [{time.time() - start:.0f} s]")
        run(["-m", "reference_curation.physics_tables_v3", "--extended-manifest", out / "holds_extended.yaml",
             "--graph", out / "contact_graph.pt", "--motion-dir", out / "motions", "--motion-file", out / "motions.pt",
             "--out", out / "physics_tables.pt", "--mjcf", plant_mjcf(), "--archive-dir", ARCHIVES, "--fps", FPS], log)
        log_fn(f"tables [{time.time() - start:.0f} s]")
        payload, summary = ct3.compile_targets_v3(out, inp["v2_inp"]["labels_dir"], inp["v2_inp"]["manifest"], rid,
                                                  inp["edges"], len(inp["v2"]["motions"]))
        ct3.write(payload, summary, out)
        del payload
        log_fn(f"sidecar [{time.time() - start:.0f} s]")
        run([SCRIPTS / "make_hold_graph_probe_plans.py", "--graph", out / "contact_graph.json", "--out-dir",
             out / "plans"], log)
        run([SCRIPTS / "make_edge_probe_plans.py", "--graph", out / "contact_graph.json", "--manifest",
             out / "holds_extended.yaml", "--out-dir", out / "plans"], log)
        log_fn(f"plans [{time.time() - start:.0f} s]")
    passed, failed, check_facts = check(out, inp, log_fn=log_fn)
    facts.update(check_facts)
    if failed:
        for f in failed:
            print(f"FAILED  {f}", file=sys.stderr)
        raise RuntimeError(f"{len(failed)} checks failed; {ids.display_path(out)} is not a release")
    calibration = calibrate_swing(out, inp)
    bad = calibration_differs(calibration, inp["v2"]["calibration_c3"])
    if bad:
        raise RuntimeError(f"calibrate_swing on the human x0 stems differs from release v2's in {bad}")
    rec = record(rid, inp, out, passed, facts, calibration, time.time() - start)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(rec, indent=1) + "\n")
    record_path.with_suffix(".md").write_text(summary_markdown(rec))
    problems = verify_record(record_path)
    if problems:
        record_path.unlink()
        record_path.with_suffix(".md").unlink()
        raise RuntimeError(f"the written record does not load as the runtime reads it: {problems}")
    return out, rec


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _eq(a, b) -> bool:
    """Exact equality of tensors (dtype and shape included), arrays and plain values."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and (
            bool(torch.equal(a, b)) if not a.is_floating_point() else bool(torch.equal(a.nan_to_num(-1e30),
                                                                                        b.nan_to_num(-1e30))))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f")))
    return a == b


def rows_equal(v3: torch.Tensor, v2: torch.Tensor, n: int, pad) -> bool:
    """``v3[:n]`` equals ``v2`` "bit for bit after padding": on v2's extent exactly, and padding beyond it."""
    a = v3[:n]
    if a.dim() != v2.dim() or a.dtype != v2.dtype or a.shape[0] != v2.shape[0]:
        return False
    if any(x < y for x, y in zip(a.shape[1:], v2.shape[1:])):
        return False
    inner = a[tuple([slice(None)] + [slice(0, s) for s in v2.shape[1:]])]
    if not _eq(inner.contiguous(), v2):
        return False
    rest = a.clone()
    rest[tuple([slice(None)] + [slice(0, s) for s in v2.shape[1:]])] = pad
    return bool((rest.nan_to_num(-1e30) == torch.tensor(pad).nan_to_num(-1e30).to(rest.dtype)).all()) \
        if rest.is_floating_point() else bool((rest == pad).all())


def kept_by_variant(names: list[str]) -> dict:
    return {s: [hx.variant_stem(s, d) for d in VARIANTS] for s in names}


PACKAGE_MOTION_KEYS = ("length_starts", "motion_lengths", "motion_dt", "motion_num_frames", "motion_weights")
GRAPH_GLOBAL = ("node_keys", "node_names", "node_contact", "node_orient", "pair_names", "orientation_names", "zone_order",
                "zone_bodies", "min_lead_s", "graph_version", "manual_goal_rule", "fps")
GRAPH_ROWS = {"seg_node": -1, "seg_contact": 0.0, "seg_start": float("inf"), "seg_end": float("inf"),
              "seg_hold": float("inf"), "seg_hold_index": -1}
TABLE_GLOBAL = ("version", "fps", "zone_order", "zone_bodies", "plant", "plant_sha256", "swing_source_codes", "pair_names",
                "rules", "body_names", "body_mass", "body_com_local", "geom_type", "box_center", "box_half", "box_quat",
                "cap_a", "cap_b", "radius", "sph_center")
TABLE_ROWS = {"swing": False, "swing_source": 0, "seg_cop_rel": 0.0, "seg_cop_valid": False, "seg_com_rel": 0.0,
              "seg_zone_share": 0.0, "seg_share_valid": False, "seg_lean_gate": False, "seg_pair_consequential": False}
SIDECAR_GLOBAL = ("kind", "version", "labels_id", "gate_id", plant_identity.KEY, "fps", "pair_names", "zone_order",
                  "role_names", "necessity_names", "pose_role_names", "flag_names", "frame_pair_names", "frame_pair_slots")


def sidecar_pad(key: str, dtype: torch.dtype):
    """``contact_targets_v2``'s padding of a ``seg_*`` / ``frame_*`` array."""
    if key == "seg_pose_role":
        return -1
    if key.startswith("seg_lp"):
        return float("nan")
    return False if dtype == torch.bool else 0


def human_slice_package(pk: dict, pk2: dict, H: int) -> list[str]:
    """Release v2's package entries (every per-frame field on its frames, every per-motion value, the file stems and
    the plant) against v3's first ``H`` motions."""
    F2 = int(pk2["motion_num_frames"].sum())
    bad = [k for k in R2.PACKED if not torch.equal(pk[k][:F2], pk2[k])]
    bad += [k for k in PACKAGE_MOTION_KEYS if not torch.equal(pk[k][:H], pk2[k])]
    if [Path(f).stem for f in pk2["motion_files"]] != [Path(f).stem for f in pk["motion_files"][:H]]:
        bad.append("motion_files")
    if pk2.get("plant_sha256") != pk.get("plant_sha256"):
        bad.append("plant_sha256")
    return bad


def human_slice_graph(g: dict, g2: dict, H: int) -> list[str]:
    """Release v2's graph tables (node tables, vocabularies) and its motions' segment rows against v3's (after
    padding); v2's hold ids first, the synthetic ones after."""
    bad = [k for k in GRAPH_GLOBAL if not _eq(g[k], g2[k])]
    bad += [k for k, pad in GRAPH_ROWS.items() if not rows_equal(g[k], g2[k], H, pad)]
    bad += [k for k in ("seg_count", "motion_num_frames") if not torch.equal(g[k][:H], g2[k])]
    n = len(g2["hold_ids"])
    if list(g["hold_ids"][:n]) != list(g2["hold_ids"]) or not all(h.startswith("SYN_") for h in g["hold_ids"][n:]):
        bad.append("hold_ids")
    return bad


def human_slice_tables(t: dict, t2: dict, H: int) -> list[str]:
    """Release v2's table rows, body constants, rules and non-path parameters against v3's (after padding)."""
    bad = [k for k in TABLE_GLOBAL if not _eq(t[k], t2[k])]
    if {k: v for k, v in t["params"].items() if k not in TABLE_PATH_PARAMS} != {
            k: v for k, v in t2["params"].items() if k not in TABLE_PATH_PARAMS}:
        bad.append("params")
    bad += [k for k, pad in TABLE_ROWS.items() if not rows_equal(t[k], t2[k], H, pad)]
    bad += [] if torch.equal(t["swing_len"][:H], t2["swing_len"]) else ["swing_len"]
    return bad


def human_slice_sidecar(s: dict, s2: dict, H: int) -> list[str]:
    """Release v2's sidecar vocabularies, rules and rows against v3's (after padding)."""
    bad = [k for k in SIDECAR_GLOBAL if not _eq(s[k], s2[k])]
    bad += [] if {k: s["rules"].get(k) for k in s2["rules"]} == s2["rules"] else ["rules"]
    bad += [k for k in ct3.SEG_KEYS + ct3.FRAME_KEYS if not rows_equal(s[k], s2[k], H, sidecar_pad(k, s[k].dtype))]
    bad += [] if torch.equal(s["frame_len"][:H], s2["frame_len"]) else ["frame_len"]
    return bad


def inherited_row_problems(s: dict, g: dict, clips: list[dict], H: int) -> list[str]:
    """Hold ids of the synthetic segments whose sidecar row differs from their ``inherits`` hold's (that hold's x0
    motion's row)."""
    row_of = {}
    for m in range(H):
        if float(clips[m]["variant_s"]) == 0.0:
            for k in range(int(g["seg_count"][m])):
                row_of[g["hold_ids"][int(g["seg_hold_index"][m, k])]] = (m, k)
    by_id = {h["hold_id"]: h for e in clips[H:] for h in e["holds"]}
    bad = []
    for m in range(H, len(clips)):
        for k in range(int(g["seg_count"][m])):
            hid = g["hold_ids"][int(g["seg_hold_index"][m, k])]
            mh, kh = row_of[by_id[hid]["inherits"]]
            if any(not _eq(s[key][m, k], s[key][mh, kh]) for key in ct3.SEG_KEYS):
                bad.append(hid)
    return bad


def synthetic_variant_problems(m: dict, x0: dict, entry: dict, d: float, e: dict, lin_v: dict, lin0: dict,
                               sha: str) -> list[str]:
    """What is wrong with one synthetic duration variant ``m`` (its extended-manifest entry ``e``, its release lineage
    arrays ``lin_v``) against its x0 clip (R2's motion ``x0``, manifest entry ``entry`` and lineage ``lin0``):
    ``splice`` (kinematics = x0 at the splice index, plant, fps, index/inserted), ``pressure`` (zero, validity 0,
    every frame), ``holds`` (re-timed by hold extension) and ``lineage`` (R2's arrays read through the index)."""
    plan = hx.insertion_plan(entry["holds"], int(round(d * FPS)))
    index = hx.splice_index(int(entry["num_frames"]), plan)
    inserted = hx.inserted_mask(index)
    out = []
    if (m.get(plant_identity.KEY) != sha or int(m["fps"]) != FPS or len(index) != e["num_frames"]
            or not np.array_equal(lin_v["index"], index.numpy()) or not np.array_equal(lin_v["inserted"], inserted.numpy())
            or any(not torch.equal(m[k], x0[k][index]) for k in R2.KINEMATIC)):
        out.append("splice")
    if (any(float(m[k].abs().max()) != 0.0 for k in hx.PRESSURE_FIELDS)
            or tuple(m["ground_reaction_valid"].shape) != (len(index), 3)):
        out.append("pressure")
    if hx.variant_holds(entry["holds"], plan, FPS) != e["holds"] or e["source_stem"] != entry["stem"]:
        out.append("holds")
    idx = index.numpy()
    if (any(not np.array_equal(lin_v[k], lin0[k][idx], equal_nan=k == "variant_t") for k in SYN_LINEAGE)
            or not np.array_equal(lin_v["stems"], lin0["stems"])):
        out.append("lineage")
    return out


def check(out: Path, inp: dict, log_fn=print) -> tuple[list, list, dict]:
    """Every check of the release (module doc); ``(passed, failed, facts)``."""
    from build_hold_graph import node_key
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.envs.control.contact_targets import ContactTargets
    from protomotions.envs.control.physics_terms import PhysicsTables

    c, facts = R2.Checks(), {}
    t0 = time.time()
    v2, v2_out, v2_inp = inp["v2"], inp["v2_out"], inp["v2_inp"]
    sha = inp["plant"][plant_identity.KEY]
    H = len(v2["motions"])
    v2_names = list(v2["motions"])
    syn_names = [n for s in inp["kept"] for n in kept_by_variant([s])[s]]
    names_want = v2_names + syn_names

    # ---- 0. release v2, on its own ---------------------------------------------------------------------------- #
    p2, f2, _ = R2.check(v2_out, v2_inp)
    moved = [role for role, a in v2["artifacts"].items() if ids.sha256_file(REPO / a["path"]) != a["sha256"]]
    c("release v2 passes its own identity checks and its artifacts are its record's", not f2 and len(p2) == 26
      and not moved, f"{len(p2)} passed, failed {f2[:2]}, moved {moved}")
    log_fn(f"  checks: release v2 [{time.time() - t0:.0f} s]")

    # ---- 1. manifest and sources ------------------------------------------------------------------------------ #
    m3 = yaml.safe_load(open(out / "release_holds.yaml"))
    v2m = v2_inp["manifest"]
    c("manifest: release v2's (every field, its clips first), then the kept synthetic entries in the record's order",
      {k: v for k, v in m3.items() if k not in ("clips", "release_v3")} == {k: v for k, v in v2m.items() if k != "clips"}
      and m3["clips"][: len(v2m["clips"])] == v2m["clips"]
      and m3["clips"][len(v2m["clips"]):] == [inp["entries"][s] for s in inp["kept"]]
      and m3["release_v3"]["drop"] == list(inp["drop"]) and m3["release_v3"]["purpose"] == inp["purpose"],
      f"{len(v2m['clips'])} + {len(inp['kept'])} clips")
    problems = R2.verify_sources(v2_inp) + verify_synthetic(inp)
    c("sources: release v2's references and ports; the synthetic record's clip files", not problems,
      "; ".join(problems[:3]))

    # ---- 2. extension ----------------------------------------------------------------------------------------- #
    ext = yaml.safe_load(open(out / "holds_extended.yaml"))
    ext2 = yaml.safe_load(open(v2_out / "holds_extended.yaml"))
    names = [e["stem"] for e in ext["clips"]]
    ext_by = {e["stem"]: e for e in ext["clips"]}
    c("extended clips: release v2's motions in its order, then every kept synthetic clip as x0, x3s, x7s",
      names == names_want and ext["source_manifest_sha256"] == ids.sha256_file(out / "release_holds.yaml"),
      f"{len(names)} motions")
    c("human slice: the extended-manifest entries are release v2's", ext["clips"][:H] == ext2["clips"])
    bad = [s for s in v2_names if ids.sha256_file(out / "motions" / f"{s}.motion") != v2["motions"][s]]
    c("human slice: the motion files are release v2's, byte for byte", not bad, ", ".join(bad[:3]))
    lineage = np.load(out / "lineage.npz")
    with np.load(v2_out / "lineage.npz") as z2:
        bad = [k for k in z2.files if k not in lineage.files or not _eq(lineage[k], z2[k])]
        v2_keys = set(z2.files)
    c("human slice: lineage.npz keeps release v2's arrays", not bad, ", ".join(bad[:3]))
    motions, bad_splice, bad_pressure, bad_holds, bad_x0, bad_lineage = {}, [], [], [], [], []
    for stem in inp["kept"]:
        rec, entry = inp["syn"]["clips"][stem], inp["entries"][stem]
        x0 = load(REPO / rec["motion"]["path"])
        with np.load(REPO / rec["lineage"]["path"]) as z:
            lin = {k: z[k] for k in z.files}
        for d, name in zip(VARIANTS, kept_by_variant([stem])[stem]):
            path = out / "motions" / f"{name}.motion"
            m = load(path)
            motions[name] = m
            e = ext_by[name]
            failed = synthetic_variant_problems(m, x0, entry, d, e, {k: lineage[f"{name}.{k}"] for k in (
                "index", "inserted", "stems", *SYN_LINEAGE)}, lin, sha)
            if ids.sha256_file(path) != e["sha256"]:
                failed.append("splice")
            for kind, sink in (("splice", bad_splice), ("pressure", bad_pressure), ("holds", bad_holds),
                               ("lineage", bad_lineage)):
                if kind in failed:
                    sink.append(name)
            if d == 0.0 and ids.sha256_file(path) != rec["motion"]["sha256"]:
                bad_x0.append(name)
    extra = sorted(set(lineage.files) - v2_keys - {f"{n}.{k}" for n in syn_names
                                                   for k in ("index", "inserted", "stems", *SYN_LINEAGE)})
    c("every synthetic variant is its x0 spliced at S (kinematics, plant, fps, lineage index/inserted)",
      not bad_splice, ", ".join(bad_splice[:3]))
    c("pressure: zero with every validity column 0 on every frame of every synthetic motion", not bad_pressure,
      ", ".join(bad_pressure[:3]))
    c("the synthetic x0 motions are R2's files byte for byte", not bad_x0, ", ".join(bad_x0[:3]))
    c("every synthetic variant's holds are its clip's, re-timed, hold ids kept", not bad_holds, ", ".join(bad_holds[:3]))
    c("lineage.npz: each synthetic variant's R2 lineage read through its index, and nothing else added",
      not bad_lineage and not extra, ", ".join((bad_lineage + extra)[:3]))
    drift = {k: tuple(v.values()) for k, v in ext["synthetic_velocity_drift_max"].items()}
    c("hold extension's velocity guard on the synthetic clips (mean <= 0.02 per field)",
      all(mean <= hx.MAX_VELOCITY_DRIFT for mean, _ in drift.values()), str(drift))
    log_fn(f"  checks: manifest, sources, extension [{time.time() - t0:.0f} s]")

    # ---- 3. package -------------------------------------------------------------------------------------------- #
    pk = load(out / "motions.pt")
    pk2 = torch.load(v2_out / "motions.pt", map_location="cpu", weights_only=False, mmap=True)
    pk_names = [Path(f).stem for f in pk["motion_files"]]
    frames = torch.tensor([int(n) for n in pk["motion_num_frames"]])
    c("package order is the extended manifest's", pk_names == names)
    c("package frame counts, fps and plant", [int(n) for n in pk["motion_num_frames"]] == [e["num_frames"] for e in
                                                                                         ext["clips"]]
      and bool(torch.allclose(pk["motion_dt"].double(), torch.full_like(pk["motion_dt"].double(), 1.0 / FPS)))
      and pk.get("plant_sha256") == sha)
    starts, bad = pk["length_starts"].tolist(), []
    for i, stem in enumerate(names):
        if i < H:
            continue                                   # the human slice is compared with release v2's package below
        sl = slice(starts[i], starts[i] + int(pk["motion_num_frames"][i]))
        for key, field in R2.PACKED.items():
            if pk.get(key) is None or not torch.equal(pk[key][sl], motions[stem][field].to(pk[key].dtype)):
                bad.append(f"{stem}.{key}")
                break
    c("package equals the synthetic motions frame by frame, the pressure channels included", not bad,
      ", ".join(bad[:3]))
    bad = human_slice_package(pk, pk2, H)
    c("human slice: package entries are release v2's (every per-motion value and per-frame slice)", not bad,
      ", ".join(bad[:3]))
    weights = edge_weights(inp["by_edge"])
    want = torch.tensor(motion_weights(ext["clips"], weights), dtype=torch.float32)
    yml = yaml.safe_load(open(out / "motions.yaml"))["motions"]
    declared = torch.tensor([float(m["weight"]) for m in yml], dtype=torch.float32)
    c("package weights: 1.0 per human motion, 3/n per motion of an edge with n kept variants (D5), the yaml's",
      torch.equal(pk["motion_weights"], want) and torch.equal(declared, want)
      and [Path(m["file"]).name for m in yml] == [f"{n}.motion" for n in names],
      f"{ {e: round(w, 4) for e, w in weights.items()} }; edges {float(want[H:].sum()):.3f} of {float(want.sum()):.3f}")
    facts["longest_motion_s"] = round(float(frames.max()) / FPS, 3)
    facts["weights"] = {"edges": {e: {"variants": len(v), "per_motion": weights[e]} for e, v in inp["by_edge"].items()},
                        "uniform_share_edges": round(float(want[H:].sum() / want.sum()), 4)}
    del pk, pk2
    log_fn(f"  checks: package [{time.time() - t0:.0f} s]")

    # ---- 4. graph ---------------------------------------------------------------------------------------------- #
    g = load(out / "contact_graph.pt")
    g2 = load(v2_out / "contact_graph.pt")
    pairs = list(zone_pairs())
    c("graph v2 built from this package and manifest", int(g.get("graph_version", 1)) == 2
      and g["package_sha256"] == ids.sha256_file(out / "motions.pt")
      and g["manifest_sha256"] == ids.sha256_file(out / "holds_extended.yaml") and g.get("plant_sha256") == sha)
    c("graph motion order, frames and fps are the package's", list(g["motion_names"]) == names
      and torch.equal(g["motion_num_frames"].long(), frames) and int(g["fps"]) == FPS)
    c("graph pair and orientation vocabularies are the extractor's and release v2's, in order",
      list(g["pair_names"]) == pairs == list(g2["pair_names"])
      and list(g["orientation_names"]) == list(ORIENT_BINS) == list(g2["orientation_names"]))
    c("graph node-key set equals release v2's (no node id moves)", list(g["node_keys"]) == list(g2["node_keys"]),
      f"{len(g['node_keys'])} nodes")
    pidx = {p: i for i, p in enumerate(pairs)}
    v2_holds = {h["hold_id"]: h for cl in v2m["clips"] for h in cl["holds"]}
    bad_seg, bad_a4 = [], []
    for m_, stem in enumerate(names):
        holds = sorted(ext["clips"][m_]["holds"], key=lambda h: float(h["t_hold"]))
        if int(g["seg_count"][m_]) != len(holds):
            bad_seg.append(stem)
            continue
        for k, h in enumerate(holds):
            want_c = torch.zeros(len(pairs))
            want_c[[pidx[p] for p in h["pairs"]]] = 1.0
            ground = sorted(p for p in h["pairs"] if p.endswith(":G"))
            node = int(g["seg_node"][m_, k])
            t = torch.tensor([float(h["t_start"]), float(h["t_hold"]), float(h["t_end"])], dtype=torch.float32)
            stored = torch.stack([g["seg_start"][m_, k], g["seg_hold"][m_, k], g["seg_end"][m_, k]]).float()
            if (g["hold_ids"][int(g["seg_hold_index"][m_, k])] != h["hold_id"]
                    or not torch.equal(g["seg_contact"][m_, k], want_c)
                    or g["node_keys"][node] != node_key(h["name"], ground, h["orientation"])
                    or not torch.equal(stored, t)):
                bad_seg.append(f"{stem}@{k}")
            src = v2_holds[h.get("inherits", h["hold_id"])]          # A4 over the inherited hold ids
            configured = {p for p in src["pairs_configured"] if p.endswith(":G")}
            if set(ground) != configured - set(src["gate"]["masks"]):
                bad_a4.append(h["hold_id"])
    c("graph segments are the manifest's holds (hold id, times, commanded contacts, node key)", not bad_seg,
      ", ".join(bad_seg[:3]))
    c("A4: every commanded ground set is the configured supports minus the gate's masks (inherited ids)",
      not bad_a4, ", ".join(bad_a4[:3]))
    graph = ContactGraph(g)
    live = torch.arange(graph.seg_node.shape[1]).unsqueeze(0) < graph.seg_count.unsqueeze(-1)
    mids = torch.arange(len(names)).unsqueeze(-1).expand_as(live)[live]
    contact, resolved = graph.manual_contact(graph.seg_node[live], mids, graph.seg_hold[live])
    c("manual goals resolve the side-specific segment (every hold's pose -> its own contact set)",
      bool(resolved.all()) and torch.equal(contact, graph.seg_contact[live]), f"{int(live.sum())} segments")
    S2 = g2["seg_node"].shape[1]
    bad = human_slice_graph(g, g2, H)
    c("human slice: graph rows and node tables are release v2's (after padding); SYN_ hold ids sort after v2's",
      not bad, ", ".join(bad[:3]))
    desc = json.loads((out / "contact_graph.json").read_text())
    desc2 = json.loads((v2_out / "contact_graph.json").read_text())
    bad = [s for s in v2_names if desc["clips"][s] != desc2["clips"][s]]
    bad += [n2["key"] for n3, n2 in zip(desc["nodes"], desc2["nodes"])
            if (n3["key"], n3["goal_pairs"], sorted(n3["pair_conflicts"])) != (n2["key"], n2["goal_pairs"],
                                                                                  sorted(n2["pair_conflicts"]))]
    keys = list(g["node_keys"])
    e3 = {(keys[e["src"]], keys[e["dst"]]): e for e in desc["edges"]}
    e2 = {(keys[e["src"]], keys[e["dst"]]): e for e in desc2["edges"]}
    for key, e in e2.items():
        human_occ = [o for o in e3.get(key, {}).get("occurrences", []) if not o["motion"].startswith("SYN_")]
        if human_occ != e["occurrences"]:
            bad.append(f"edge {key}")
    c("human slice: the graph description's human clips, node keys and goal pairs, and every v2 edge's human "
      "occurrences are release v2's", not bad, ", ".join(map(str, bad[:3])))
    need, s_to_d = {}, set()
    for eid, stems in inp["by_edge"].items():
        spec = ct3.edge_spec(inp["edges"], eid)
        key = (spec["source"]["node_key"], spec["destination"]["node_key"])
        s_to_d.add(key)
        occ = [o for o in e3.get(key, {}).get("occurrences", []) if o["motion"].startswith("SYN_")]
        n_syn = sum(1 for o in occ if o["motion"] in syn_names)
        need[eid] = {"S": key[0], "D": key[1], "in_v2": key in e2, "synthetic_occurrences": n_syn,
                     "motions": len(VARIANTS) * len(stems), "pass": key in e3 and n_syn == len(VARIANTS) * len(stems)}
    new = sorted(set(e3) - set(e2))
    c("each kept edge's S -> D is present once per synthetic motion, and no other edge is new",
      all(x["pass"] for x in need.values()) and set(new) <= s_to_d,
      "; ".join(f"{k}: {x['synthetic_occurrences']}/{x['motions']}" for k, x in need.items()) + f"; new {len(new)}")
    seg_v2 = Counter(int(n) for n in g2["seg_count"])
    facts["graph"] = {"nodes": graph.num_nodes, "edges": len(desc["edges"]), "segments": int(live.sum()),
                      "hold_ids": len(graph.hold_ids), "max_segments": int(g["seg_node"].shape[1]),
                      "max_segments_v2": int(S2), "nodes_with_pair_conflicts": sum(1 for n in desc["nodes"]
                                                                                  if n["pair_conflicts"]),
                      "new_edges": [list(k) for k in new], "s_to_d": need, "v2_segment_counts": dict(seg_v2)}
    log_fn(f"  checks: graph [{time.time() - t0:.0f} s]")

    # ---- 5. tables --------------------------------------------------------------------------------------------- #
    t = load(out / "physics_tables.pt")
    t2 = load(v2_out / "physics_tables.pt")
    body_names = mjcf_body_names(str(plant_mjcf()))
    c("tables v2 built with this graph, package and manifest", int(t.get("version", 1)) == 2
      and t["graph_sha256"] == ids.sha256_file(out / "contact_graph.pt")
      and t["package_sha256"] == ids.sha256_file(out / "motions.pt")
      and t["manifest_sha256"] == ids.sha256_file(out / "holds_extended.yaml"))
    c("tables keyed to the package (names, frames, fps), the robot's body order and plant v2",
      list(t["motion_names"]) == names and torch.equal(t["swing_len"].long(), frames) and int(t["fps"]) == FPS
      and list(t["body_names"]) == body_names and t["plant_sha256"] == sha)
    c("tables' pair names, zone order and segment layout are the graph's", list(t["pair_names"]) == pairs
      and list(t["zone_order"]) == list(ZONE_ORDER)
      and tuple(t["seg_pair_consequential"].shape) == (*g["seg_node"].shape, len(pairs)))
    stale = []
    for m_, stem in enumerate(names):
        ins = torch.from_numpy(lineage[f"{stem}.inserted"])
        if bool((t["swing_source"][m_, : len(ins)][ins] >= 2).any()):
            stale.append(stem)
    c("no pressure-sourced swing label on an inserted frame", not stale, ", ".join(stale[:3]))
    syn_rows = slice(H, len(names))
    c("synthetic motions took the no-pressure path: velocity labels only, no COP or share signature",
      not bool((t["swing_source"][syn_rows] >= 2).any()) and not bool(t["seg_cop_valid"][syn_rows].any())
      and not bool(t["seg_share_valid"][syn_rows].any()) and not bool(t["seg_zone_share"][syn_rows].any())
      and json.loads((out / "physics_tables.json").read_text())["pressure_motions"] == H,
      f"{int(t['swing'][syn_rows].sum())} synthetic swing zone-frames")
    bad = human_slice_tables(t, t2, H)
    st, st2 = (json.loads((d / "physics_tables.json").read_text()) for d in (out, v2_out))
    bad += [s for s in v2_names if st["swing_share"].get(s) != st2["swing_share"].get(s)]
    c("human slice: table rows, body constants, rules and parameters are release v2's (after padding)", not bad,
      ", ".join(bad[:3]))
    facts["tables"] = {"swing_zone_frames": int(t["swing"].sum()), "lean_gated_segments": int(t["seg_lean_gate"].sum()),
                       "cop_valid_segments": int(t["seg_cop_valid"].sum()),
                       "share_valid_segments": int(t["seg_share_valid"].sum()),
                       "synthetic_swing_zone_frames": int(t["swing"][syn_rows].sum()),
                       "synthetic_velocity_labels": int((t["swing_source"][syn_rows] == 1).sum()),
                       "synthetic_lean_gated_segments": int(t["seg_lean_gate"][syn_rows].sum())}
    del t, t2
    log_fn(f"  checks: tables [{time.time() - t0:.0f} s]")

    # ---- 6. sidecar -------------------------------------------------------------------------------------------- #
    s = load(out / "contact_targets.pt")
    s2 = load(v2_out / "contact_targets.pt")
    c("sidecar compiled for this graph, package and manifest",
      s["graph_sha256"] == ids.sha256_file(out / "contact_graph.pt")
      and s["package_sha256"] == ids.sha256_file(out / "motions.pt")
      and s["manifest_sha256"] == ids.sha256_file(out / "holds_extended.yaml") and s.get(plant_identity.KEY) == sha)
    c("sidecar names, pairs, hold ids and frames are the graph's", list(s["motion_names"]) == names
      and list(s["pair_names"]) == pairs and list(s["hold_ids"]) == list(g["hold_ids"])
      and torch.equal(s["seg_hold_index"], g["seg_hold_index"]) and torch.equal(s["frame_len"], frames))
    L = live.unsqueeze(-1)
    commanded = s["seg_commanded"] & L
    c("sidecar: commanded = configured - masks = the graph's seg_contact",
      torch.equal(commanded, (s["seg_configured"] & ~s["seg_masked"]) & L)
      and torch.equal(commanded, (g["seg_contact"] > 0.5) & L) and not bool((s["seg_masked"] & ~s["seg_configured"]).any()))
    summary = json.loads((out / "contact_targets.json").read_text())
    anns = ct.read_annotations(v2_inp["labels_dir"])
    x0_holds = [h.get("inherits", h["hold_id"]) for e in ext["clips"] if float(e["variant_s"]) == 0.0
                for h in e["holds"]]
    crit = sum(1 for h in x0_holds for a in anns.get(h, {}).values()
               if a.get("critical") and a["target_role"] == "required_touch" and a["in_configuration"])
    restored = sum(1 for h in x0_holds for a in anns.get(h, {}).values() if a.get("restored"))
    masks = Counter(p for h in x0_holds for p in v2_holds[h]["gate"]["masks"])
    c("sidecar counts equal the labels' and the gate's over the x0 holds (synthetic: the inherited hold ids)",
      summary["critical"] == crit and summary["restored"] == restored and summary["masked"] == dict(masks),
      f"critical {crit}, restored {restored}, masks {dict(masks)}")
    v2_summary = json.loads((v2_out / "contact_targets.json").read_text())
    bad = human_slice_sidecar(s, s2, H)
    bad += [] if summary["human_x0"] == v2_summary else ["contact_targets.json human_x0"]
    c("human slice: sidecar rows, vocabularies and rules, and release v2's sidecar summary (after padding)", not bad,
      ", ".join(bad[:3]))
    del s2
    bad = inherited_row_problems(s, g, ext["clips"], H)
    c("inherited labels: every synthetic segment's sidecar row equals its inherits hold's", not bad,
      ", ".join(bad[:3]))
    bad = sidecar_frame_problems(out, s, ext, lineage, names, H, inp)
    c("sidecar frames of the synthetic motions: capture v4 on real frames, the planned brace and the clip's "
      "geometry on synthetic and blend frames, x3s/x7s through the index", not bad, "; ".join(bad[:3]))
    conflicts = summary["known_free_conflicts"]
    c("no synthetic frame's planned ground zone is known-free in the hold the schedule commands there",
      conflicts == 0 and all(not f["known_free_conflicts"] for f in summary["synthetic_frames"].values()),
      f"{conflicts} conflicts")
    gaps = [f["real_frame_gap_vs_capture_m"] for f in summary["synthetic_frames"].values()]
    facts["sidecar"] = {k: summary[k] for k in ("holds", "roles", "configured_body_body", "critical", "restored",
                                                 "masked", "flags", "pose_roles", "known_free_zone_holds",
                                                 "free_unknown", "frame_pair_partly_realised")}
    facts["sidecar"]["synthetic_x0"] = {k: summary["synthetic_x0"][k] for k in ("holds", "critical", "restored",
                                                                              "known_free_zone_holds",
                                                                              "frame_pair_partly_realised")}
    facts["sidecar"]["real_frame_gap_vs_capture_max_m"] = max(gaps) if gaps else None
    del s
    log_fn(f"  checks: sidecar [{time.time() - t0:.0f} s]")

    # ---- 7. the runtime loaders -------------------------------------------------------------------------------- #
    try:
        rg = ContactGraph.from_file(out / "contact_graph.pt")
        rg.validate_against_motion_lib([f"{n}.motion" for n in names], motion_num_frames=frames, fps=FPS)
        rt = PhysicsTables(str(out / "physics_tables.pt"), names, body_names, "cpu", plant_mjcf=str(plant_mjcf()),
                           motion_num_frames=frames, fps=FPS)
        ok = rt.version == 2 and rt.pair_names == rg.pair_names and rt.graph_sha256 == ids.sha256_file(
            out / "contact_graph.pt")
        ContactTargets(str(out / "contact_targets.pt"), rg, names, "cpu", motion_num_frames=frames, fps=FPS,
                       graph_sha256=ids.sha256_file(out / "contact_graph.pt"), plant_mjcf=str(plant_mjcf()))
        c("runtime loaders accept the graph, tables and sidecar", ok)
    except Exception as exc:  # noqa: BLE001 -- a refusal is a failed check
        c("runtime loaders accept the graph, tables and sidecar", False, f"{type(exc).__name__}: {exc}")
    log_fn(f"  checks: runtime loaders [{time.time() - t0:.0f} s]")

    # ---- 8. plans ---------------------------------------------------------------------------------------------- #
    plan_facts = check_plans(c, out, v2_out, names, set(g["node_keys"]), desc, ext, inp)
    facts["plans"] = plan_facts["count"]
    facts["edge_plans"] = plan_facts
    log_fn(f"  checks: plans [{time.time() - t0:.0f} s]")

    # ---- 9. synthetic lineage, inherited labels, plant, record ------------------------------------------------- #
    lin_facts = check_synthetic_lineage(c, inp, out)
    facts["synthetic_lineage"] = lin_facts
    plant_ok = (all(motions[n].get(plant_identity.KEY) == sha for n in syn_names)
                and all(ids.sha256_file(out / "motions" / f"{n}.motion") == v2["motions"][n] for n in v2_names)
                and v2["plant"][plant_identity.KEY] == sha)
    c("plant: every synthetic motion carries plant v2's sha256 and every human motion is release v2's file (on plant "
      "v2 by v2's own check); the package, graph, tables and sidecar are checked above", plant_ok)
    longest, steps = facts["longest_motion_s"], int(math.ceil(facts["longest_motion_s"] * CONTROL_HZ))
    syn_longest = max(e["length_s"] for e in ext["clips"][H:])
    c("record: training.eval_max_steps covers the longest motion (the synthetic ones included)",
      steps / CONTROL_HZ >= longest >= syn_longest, f"longest {longest} s (synthetic {syn_longest} s) -> {steps} steps")
    facts["longest_synthetic_motion_s"] = syn_longest
    log_fn(f"  checks: synthetic lineage, plant [{time.time() - t0:.0f} s]")
    return c.passed, c.failed, facts


def sidecar_frame_problems(out: Path, s: dict, ext: dict, lineage, names: list, H: int, inp: dict) -> list[str]:
    """Re-derive the synthetic motions' per-frame pair rows (``contact_targets_v3``'s rules) and compare."""
    from reference_curation import fit_writer as fw
    from reference_curation import human_mesh as hm
    from reference_curation.retarget_v2 import motion_state

    q_pairs = list(s["frame_pair_names"])
    q_cols = [hm.PAIR_NAMES.index(p) for p in q_pairs]
    qidx = {p: i for i, p in enumerate(q_pairs)}
    sk = fw.skeleton(PLANT)
    evidence, rows, problems = {}, {}, []
    by = {e["stem"]: e for e in ext["clips"]}
    for name in names[H:]:
        e = by[name]
        if float(e["variant_s"]) != 0.0:
            continue
        lin = ct3.lineage_of(lineage, name)
        stems = [str(x) for x in lin["stems"]]
        real = lin["kind"] == ct3.KIND_REAL
        mot = load(out / "motions" / f"{name}.motion")
        pos, rot = motion_state(mot)
        with torch.no_grad():
            gap = capture_v4.zone_pair_gaps(sk, torch.as_tensor(pos), torch.as_tensor(rot))[:, q_cols]
        ref, human = gap <= ct3.PAIR_CLOSE_M, np.zeros(gap.shape, bool)
        for f in np.nonzero(real)[0]:
            stem = stems[int(lin["source"][f])]
            if stem not in evidence:
                rec = capture_v4.load(stem, rebuild=False)
                evidence[stem] = (np.asarray(rec["avatar_pair_gap"], np.float64), np.asarray(rec["human_pair_state"]))
            g, h = evidence[stem]
            sf = int(lin["source_frame"][f])
            ref[f] = g[sf, q_cols] <= ct3.PAIR_CLOSE_M
            human[f] = h[sf, q_cols] == 1
        spec = ct3.edge_spec(inp["edges"], e["synthetic"]["edge"])
        phase = ct3.phase_at(np.nan_to_num(lin["variant_t"], nan=-1.0), e["synthetic"]["durations_s"])
        for f in np.nonzero(~real)[0]:
            _, braces = ct3.planned_config(spec, int(phase[f]))
            human[f, [qidx[b] for b in braces]] = True
        if not np.array_equal(np.isnan(lin["variant_t"]), real):
            problems.append(f"{name}: variant_t is not NaN exactly on the real frames")
        rows[name] = (ref, human)
    for m_, name in enumerate(names):
        if m_ < H:
            continue
        e = by[name]
        ref, human = rows[e["source_stem"]]
        index = lineage[f"{name}.index"]
        n = len(index)
        want = {"frame_pair_ref": ref[index], "frame_pair_human": human[index], "frame_pair_ok": ref[index] & human[index]}
        for k, v in want.items():
            if not np.array_equal(s[k][m_, :n].numpy(), v) or bool(s[k][m_, n:].any()):
                problems.append(f"{name}.{k}")
    return problems


def check_plans(c, out: Path, v2_out: Path, names: list, keys: set, desc: dict, ext: dict, inp: dict) -> dict:
    """v2's plan check on every plan, v3's family plans = v2's file for file, and R4a's edge plans."""
    import make_edge_probe_plans as mep

    plans = sorted((out / "plans").glob("*.json"))
    unresolved = []
    for p in plans:
        plan = json.loads(p.read_text())
        if plan["start"]["clip"] not in names or any(goal["config"] not in keys or goal["pose_clip"] not in names
                                                      for goal in plan["goals"]):
            unresolved.append(p.name)
    c("probe plans resolve against the release's graph and library", bool(plans) and not unresolved,
      f"{len(plans)} plans; " + ", ".join(unresolved[:3]))
    v2_plans = {p.name: p.read_bytes() for p in sorted((v2_out / "plans").glob("*.json"))}
    edge_plans, picks = mep.edge_plans(desc, ext["clips"])
    edge_names = {f"{n}.json" for n in edge_plans}
    v3_plans = {p.name: p.read_bytes() for p in plans}
    c("family plans: release v2's, file for file (no synthetic clip takes a family over)",
      {n: b for n, b in v3_plans.items() if n not in edge_names} == v2_plans, f"{len(v2_plans)} plans")
    want = {f"{n}.json": (json.dumps(p, indent=2)).encode() for n, p in edge_plans.items()}
    c("edge plans: edge_, nohijack_ and fork_edge_ per kept edge, R4a's (plan clip by its rule)",
      {n: v3_plans.get(n) for n in want} == want and len(edge_names) == 3 * len(inp["by_edge"])
      and set(picks) == set(inp["by_edge"]), ", ".join(f"{e} <- {s}" for e, s in picks.items()))
    card = {e: s for e, s in CARD_PLAN_CLIPS.items() if s in inp["kept"]}
    c("the plan-clip rule reproduces the card's R4a picks (where they are kept)",
      all(picks.get(e) == s for e, s in card.items()), f"{len(card)} of {len(CARD_PLAN_CLIPS)} card picks kept")
    # what the launcher does: SequenceViz's own loader, within the panel's cap
    from protomotions.agents.evaluators.sequence_viz import SequenceVizConfig, SequenceVizRunner
    from protomotions.components.contact_graph import ContactGraph

    panel = [out / "plans" / n for n in sorted(edge_names) if not n.startswith("fork_")]
    allp = [out / "plans" / n for n in sorted(edge_names)]
    runner = SequenceVizRunner.__new__(SequenceVizRunner)
    runner.config = SequenceVizConfig(viz_every=1, num_sequences=len(allp) + 2, plan_files=[str(p) for p in allp],
                                      max_seconds=VIZ_MAX_SECONDS)
    runner.graph = ContactGraph.from_file(out / "contact_graph.pt")
    runner.motion_names = names
    lengths, short = {}, []
    for p in allp:
        seq = runner._load_plan(str(p))
        n_goals = len(json.loads(p.read_text())["goals"])
        if seq is None or len(seq.goals) < n_goals:
            short.append(p.name)
        else:
            lengths[p.stem] = round(seq.total_s, 2)
    c(f"edge plans resolve with SequenceViz's loader and keep every goal within the panel's {VIZ_MAX_SECONDS:.0f} s",
      not short, ", ".join(short[:3]))
    return {"count": len(plans), "edge_plans": sorted(edge_names), "panel": [p.name for p in panel],
            "picks": picks, "total_s": lengths}


def check_synthetic_lineage(c, inp: dict, out: Path) -> dict:
    """R2's per-clip checks on the kept clips as the release holds them (x0 = R2's bytes): real frames against
    release v2 after the recorded transform, transition frames against the admitted variant's re-export (sha256),
    the motion file (FK, plant, pressure), the holds (inherited fields, exemplars) and the windows' content."""
    rel = S3.release_v2()
    key = inp["syn"]["key"]["variants"]
    names_chk = S3.name_checks(inp["kept"], list(rel["clips"]))
    kept_names = {hx.variant_stem(s, d) for s in inp["kept"] for d in VARIANTS}
    recorded = [x for x in inp["syn"]["name_collisions"] if x[0] in inp["kept"] and x[1] in kept_names]
    c("names: no synthetic stem contains a real stem or ends in _x<digits>s; the substring collisions are the "
      "synthetic record's among the kept clips", names_chk["pass"] and names_chk["name_collisions"] == recorded,
      f"{len(recorded)} collisions")
    cache, out_facts, failures, inherited = {}, {}, {}, {}
    for stem in inp["kept"]:
        rec = inp["syn"]["clips"][stem]
        entry = inp["entries"][stem]
        syn = entry["synthetic"]
        var_path = REPO / syn["motion"]
        if (ids.sha256_file(var_path) != key[stem]["motion"] or syn["sha256"] != key[stem]["motion"]
                or ids.sha256_file(REPO / syn["record"]) != key[stem]["record"]):
            failures[stem] = ["the admitted variant's re-export is not the one spliced (sha256)"]
            continue
        mot = load(out / "motions" / f"{stem}.motion")
        with np.load(REPO / rec["lineage"]["path"]) as z:
            lin = {k: z[k] for k in z.files}
        info = json.loads((REPO / rec["json"]["path"]).read_text())
        for s in (info["S"]["stem"], info["D"]["stem"]):
            if s not in cache:
                cache[s] = S3.release_motion(rel, s)
        sources = {info["S"]["stem"]: cache[info["S"]["stem"]], info["D"]["stem"]: cache[info["D"]["stem"]]}
        res = {"motion": S3.check_motion_file(mot), "real_frames": S3.check_real_frames(mot, lin, sources),
               "synthetic_frames": S3.check_synthetic_frames(mot, lin, load(var_path)),
               "holds": S3.check_holds(entry, lin, rel), "hold_content": S3.check_hold_content(entry, mot, lin, info, rel)}
        failures[stem] = [k for k, v in res.items() if not v["pass"] and k != "holds"]
        inherited[stem] = res["holds"]["problems"]
        out_facts[stem] = {"real_pos_m": res["real_frames"]["pos_m"], "real_dof_exact": res["real_frames"]["dof_exact"],
                           "real_vel": res["real_frames"]["vel"], "ang_vel_horizon_ties":
                               res["real_frames"]["ang_vel_horizon_ties"],
                           "synthetic_pos_m": res["synthetic_frames"]["pos_m"],
                           "synthetic_dof_exact": res["synthetic_frames"]["dof_exact"],
                           "fk_round_trip_m": res["motion"]["round_trip"]["pos_m"],
                           "windows_trimmed": res["hold_content"]["trimmed"]}
    bad = {k: v for k, v in failures.items() if v}
    c("synthetic lineage: real frames = release v2 after the transform, transition frames = the admitted variant, "
      "FK, plant, zero pressure, the windows' content (R2's checks)", not bad,
      "; ".join(f"{k}: {v}" for k, v in list(bad.items())[:3]))
    bad = {k: v for k, v in inherited.items() if v}
    c("inherited labels: every synthetic hold's copied fields (name, pairs, orientation, labels, gate, ...) equal its "
      "inherits hold's, and each frame_hold is that hold's own exemplar", not bad and len(inherited) == len(inp["kept"]),
      "; ".join(f"{k}: {v[:2]}" for k, v in list(bad.items())[:3]))
    return out_facts


def calibrate_swing(out: Path, inp: dict) -> dict:
    """Release v2's ``calibrate_swing`` on the human x0 stems: a view of the release whose manifest lists only the
    human motions (first in the tables), with the release's own tables. Adds the synthetic motions' label counts."""
    H = len(inp["v2"]["motions"])
    ext = yaml.safe_load(open(out / "holds_extended.yaml"))
    if any(c["stem"].startswith("SYN_") for c in ext["clips"][:H]):
        raise ValueError("the first release-v2-many motions are not the human ones")
    with tempfile.TemporaryDirectory(prefix="release_v3_calibration_") as tmp:
        tmp = Path(tmp)
        with open(tmp / "holds_extended.yaml", "w") as f:
            yaml.safe_dump({**ext, "clips": ext["clips"][:H]}, f, sort_keys=False, width=120)
        for name in ("physics_tables.pt", "physics_tables.json"):
            os.symlink(out / name, tmp / name)
        cal = R2.calibrate_swing(tmp, inp["v2_inp"])
    t = load(out / "physics_tables.pt")
    src, sw = t["swing_source"][H:], t["swing"][H:]
    cal["synthetic_motions"] = {"motions": len(ext["clips"]) - H, "labels": int(sw.sum()),
                                "velocity_only": int((src == 1).sum()), "pressure": int((src >= 2).sum())}
    cal["note"] = ("x0_against_human_evidence and human_load_on_labelled_feet_hands_n are over the human x0 stems "
                   "(release v2's); all_motions counts every v3 motion, the synthetic ones included")
    return cal


def calibration_differs(cal: dict, v2_cal: dict) -> list[str]:
    """The keys of release v2's ``calibration_c3`` that differ (``all_motions`` counts the synthetic motions too)."""
    return [k for k in v2_cal if k != "all_motions" and cal.get(k) != v2_cal[k]]


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
def record(rid: str, inp: dict, out: Path, passed: list, facts: dict, calibration: dict, seconds: float) -> dict:
    artifacts = {role: {"path": ids.display_path(out / name), "sha256": ids.sha256_file(out / name),
                        "bytes": (out / name).stat().st_size} for role, name in ARTIFACTS.items()}
    ext = yaml.safe_load(open(out / "holds_extended.yaml"))
    plans = {p.name: ids.sha256_file(p) for p in sorted((out / "plans").glob("*.json"))}
    v2, v2_inp = inp["v2"], inp["v2_inp"]
    inputs = [inp["v2_record_path"], inp["syn_record_path"], inp["edges_path"], v2_inp["manifest_path"],
              v2_inp["labels_dir"] / "annotations.jsonl", plant_mjcf(), plant_identity.flat_path(PLANT)]
    groups = Counter(c["group"] for c in inp["v2_inp"]["manifest"]["clips"]) + Counter(
        inp["entries"][s]["group"] for s in inp["kept"])
    longest = facts["longest_motion_s"]
    syn = inp["syn"]
    return {
        **ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs),
        "kind": "reference_release", "release_id": rid, "release_version": RELEASE_VERSION,
        "purpose": inp["purpose"], "drop": list(inp["drop"]), "variants": list(inp["kept"]),
        "plant": inp["plant"], "robot": "smpl_yogi_v2",
        "release_v2": {"release_id": RELEASE_V2_ID, "record": ids.display_path(inp["v2_record_path"]),
                       "sha256": inp["v2_record_sha256"]},
        "synthetic": {"synthetic_id": SYNTHETIC_ID, "record": ids.display_path(inp["syn_record_path"]),
                      "sha256": inp["syn_record_sha256"], "order": syn["order"], "kept": list(inp["kept"]),
                      "edges": inp["by_edge"], "name_collisions": [x for x in syn["name_collisions"]
                                                                   if x[0] in inp["kept"]],
                      "transition_contacts": {k: v for k, v in syn["transition_contacts"].items() if k in inp["kept"]},
                      "windows_trimmed": {k: v for k, v in syn["windows_trimmed"].items() if k in inp["kept"]},
                      "d1_policy": syn["d1_policy"], "t6_decision": syn["t6_decision"]},
        "gate_id": v2["gate_id"], "labels_id": v2["labels_id"], "statics_id": v2["statics_id"],
        "retarget_id": v2["retarget_id"], "fit_id": v2["fit_id"],
        "dir": ids.display_path(out), "key": id_key(inp), "parameters": parameters(),
        "builders": {ids.display_path(p): ids.sha256_file(p) for p in builders()},
        "check_modules": {ids.display_path(p): ids.sha256_file(p) for p in (Path(S3.__file__),)},
        "artifacts": artifacts, "plans": plans,
        "motions": {e["stem"]: e["sha256"] for e in ext["clips"]},
        "weights": facts["weights"],
        "counts": {"clips": len(v2_inp["manifest"]["clips"]) + len(inp["kept"]), "human_clips":
                   len(v2_inp["manifest"]["clips"]), "synthetic_clips": len(inp["kept"]), "groups": dict(groups),
                   "holds": sum(len(c["holds"]) for c in v2_inp["manifest"]["clips"]) + sum(
                       len(inp["entries"][s]["holds"]) for s in inp["kept"]),
                   **facts["extension"], "graph": facts["graph"], "tables": facts["tables"],
                   "sidecar": facts["sidecar"], "plans": facts["plans"], "edge_plans": facts["edge_plans"]},
        "synthetic_lineage": facts["synthetic_lineage"],
        "training": {"robot": "smpl_yogi_v2", "longest_motion_s": longest,
                     "eval_max_steps": int(math.ceil(longest * CONTROL_HZ)),
                     "note": "the full-clip evaluator must cover the longest motion: eval_max_steps >= this"},
        "checks": {"passed": passed, "failed": []},
        "calibration_c3": calibration,
        "seconds": round(seconds, 1),
    }


def verify_record(record_path: Path) -> list[str]:
    """What training reads of the record (``release_identity``, ``run_expert_graph_ft.sh``) loads and matches."""
    from protomotions.utils.release_identity import ROLES, ReleaseMismatchError, load_release, require_artifact

    problems = []
    try:
        rec = load_release(record_path)
        for role in ROLES:
            require_artifact(rec, role, REPO / rec["artifacts"][role]["path"])
        if not (REPO / rec["dir"] / "plans").is_dir() or rec.get("robot") != "smpl_yogi_v2":
            problems.append("dir/plans or robot")
        plant_identity.require(rec["plant"].get(plant_identity.KEY), plant_mjcf(), f"release {rec['release_id']}")
        pk = torch.load(REPO / rec["artifacts"]["package"]["path"], map_location="cpu", weights_only=False, mmap=True)
        stems = [Path(str(f)).name.rsplit(".motion", 1)[0] for f in pk["motion_files"]]
        if list(rec["motions"]) != stems:
            problems.append("motions != the package's motion_files")
        longest = float(pk["motion_num_frames"].max()) / FPS
        steps = rec["training"]["eval_max_steps"]
        if (not isinstance(steps, int) or steps / CONTROL_HZ < longest
                or rec["training"]["longest_motion_s"] != round(longest, 3)):
            problems.append(f"training.eval_max_steps {steps} does not cover the longest motion ({longest:.3f} s)")
    except (ReleaseMismatchError, plant_identity.PlantMismatchError, KeyError, OSError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    return problems


def summary_markdown(rec: dict) -> str:
    c = rec["counts"]
    s = rec["synthetic"]
    w = rec["weights"]
    cal = rec["calibration_c3"]
    lines = [f"# Release `{rec['release_id']}` ({rec['purpose']})", "",
             f"Generated by `{MODULE}` (card R3, `expert_revist/graph_growth_2026_10_03/PLAN.MD`): release v2 "
             f"`{rec['release_v2']['release_id']}` plus the spliced synthetic clips `{s['synthetic_id']}`, without "
             f"{rec['drop']}. Heavy files: `{rec['dir']}/` (untracked).", "",
             "| | |", "|---|---|",
             f"| clips | {c['clips']} ({c['human_clips']} human, {c['synthetic_clips']} synthetic: "
             + ", ".join(f"{e} x{len(v)}" for e, v in s["edges"].items()) + ") |",
             f"| motions / frames / inserted | {c['motions']} / {c['frames']} / {c['inserted_frames']} (synthetic "
             f"{c['synthetic_motions']} / {c['synthetic_frames']} / {c['synthetic_inserted_frames']}) |",
             "| weights | human 1.0; per edge 3/n: " + ", ".join(f"{e} {v['per_motion']:.4g}" for e, v in
                                                              w["edges"].items())
             + f"; the edges' share of the uniform mass {w['uniform_share_edges']} |",
             f"| graph | {c['graph']['nodes']} nodes (release v2's keys), {c['graph']['edges']} edges "
             f"({len(c['graph']['new_edges'])} new: the S -> D of the kept edges), {c['graph']['segments']} segments, "
             f"{c['graph']['hold_ids']} hold ids |",
             f"| tables | {c['tables']} |",
             f"| sidecar | critical {c['sidecar']['critical']}, restored {c['sidecar']['restored']}, masks "
             f"{c['sidecar']['masked']}, known-free zone-holds {c['sidecar']['known_free_zone_holds']}; real-frame pair "
             f"gaps vs capture v4 <= {c['sidecar']['real_frame_gap_vs_capture_max_m']:.2e} m |",
             f"| plans | {c['plans']} (release v2's {c['plans'] - len(c['edge_plans']['edge_plans'])}, plus "
             f"{len(c['edge_plans']['edge_plans'])} edge plans: " + ", ".join(f"{e} <- `{p}`" for e, p in
                                                                              c["edge_plans"]["picks"].items()) + ") |",
             f"| training | robot `{rec['training']['robot']}`, longest motion {rec['training']['longest_motion_s']} s "
             f"-> eval_max_steps >= {rec['training']['eval_max_steps']} |",
             "", f"Checks: {len(rec['checks']['passed'])} passed, 0 failed. `calibrate_swing` on the human x0 stems "
                 f"equals release v2's; the synthetic motions carry {cal['synthetic_motions']['labels']} swing "
                 "zone-frames, all from the velocity rule.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def check_recorded(rid: str, log_fn=print) -> tuple[list, list]:
    """Re-run every check on a recorded release, plus the record's own: artifacts and plans by sha256, what training
    reads of it, and the swing calibration."""
    rec = json.loads((RECORD_ROOT / f"{rid}.json").read_text())
    inp = pinned_inputs(rec["drop"], rec["purpose"])
    now = release_id(inp)
    if now != rec["release_id"]:
        log_fn(f"note: today's inputs and builders determine {now}; this release was built by the recorded ones")
    out = REPO / rec["dir"]
    passed, failed, _ = check(out, inp, log_fn=log_fn)
    for role, a in rec["artifacts"].items():
        (passed if ids.sha256_file(REPO / a["path"]) == a["sha256"] else failed).append(f"artifact {role} sha256")
    plans = {p.name: ids.sha256_file(p) for p in sorted((out / "plans").glob("*.json"))}
    (passed if plans == rec["plans"] else failed).append("plans are the record's")
    motions = {n: ids.sha256_file(out / "motions" / f"{n}.motion") for n in rec["motions"]}
    (passed if motions == rec["motions"] else failed).append("motions are the record's")
    problems = verify_record(RECORD_ROOT / f"{rid}.json")
    (failed if problems else passed).append(f"the record loads as training reads it{': ' + str(problems) if problems else ''}")
    cal = calibrate_swing(out, inp)
    (passed if cal == rec["calibration_c3"] else failed).append("calibrate_swing reproduces the record's")
    return passed, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="build, check and record the release")
    ap.add_argument("--check", metavar="RELEASE_ID", help="re-run every check on a recorded release")
    ap.add_argument("--print-id", action="store_true", help="print the id the inputs and builders determine")
    ap.add_argument("--drop", nargs="*", default=list(USER_DROP),
                    help=f"synthetic stems to leave out, exact match (default: the user's {list(USER_DROP)})")
    ap.add_argument("--purpose", choices=PURPOSES, default="candidate")
    ap.add_argument("--force", action="store_true", help="delete a failed build's folder (one without a record)")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.print_id:
            print(release_id(pinned_inputs(args.drop, args.purpose)))
            return 0
        if args.build:
            out, rec = build(args.drop, args.purpose, force=args.force)
            c = rec["counts"]
            print(f"release {rec['release_id']} ({rec['purpose']}): {c['clips']} clips, {c['motions']} motions, "
                  f"{c['graph']['nodes']} nodes, {c['plans']} plans; {len(rec['checks']['passed'])} checks passed in "
                  f"{rec['seconds']:.0f} s -> {ids.display_path(RECORD_ROOT / (rec['release_id'] + '.json'))}")
            return 0
        if args.check:
            passed, failed = check_recorded(args.check)
            for f in failed:
                print(f"FAILED  {f}", file=sys.stderr)
            print(f"release {args.check}: {len(passed)} checks passed, {len(failed)} failed")
            return 1 if failed else 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
