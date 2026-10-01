# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statics v2 (BodyFix Step 4, item 4): Step 7's gated LP and statue on plant v2 and the Step 3 references.

What a hold asks of the plant is unchanged: labels v1.1's configured contacts (the ground set, the
``required_touch`` pairs and the carried labels) at the hold's corrected exemplar (``statics.hold_request``). The LP
is Step 7's, unchanged (``statics.analyse``: ground points only within 2 cm, pairs only within 1 cm, typed
statuses, the strict min/max-load necessity test, the joint-stop diagnostic, the counterfactual when a contact the
human makes is not realised). What changes:

* **the plant**: plant v2 (her skeleton and masses, the widened joint box, wrist / hand torque 30 / 15 N m), set
  in this process by ``retarget_v2.on_plant()`` (``mosh_replay.use_plant``: ``static_hold_lp``'s model, statics'
  joint box). Its records carry plant v2's MJCF among their inputs, which ``ids.require_plant`` reads;
* **the references**: the ``retarget_v2`` motions (``capture_v4.RETARGET_ID``);
* **the statue** (``witness_v2``): plant v2's training gains, and TODO D4's contact rule (only the zones the LP
  loads must stay down). Step 7's stricter verdict is kept per hold (``witness.passed_strict``).

A hold's ``verdict`` is Step 7's: ``held`` (the statue passed), ``feasible`` (the LP holds it within the plant's
limits, the statue did not), ``beyond_plant``, ``infeasible``, ``support_not_realised`` or the solver's failure;
``holdable`` = held + feasible. ``write`` puts ``holds.jsonl``, ``contacts.jsonl`` (one row per configured contact:
the labels' role beside the gated and counterfactual necessity, ``statics.contact_rows``), ``statics.json`` and
``summary.md`` into ``data/reference_curation/statics_v2/<labels_id>.statics_v2.<hash>/`` (not ``statics/``, whose
records Step 2's tests expect to be plant v1's).

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.statics_v2 [--labels <dir>] [--workers 8]
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

from reference_curation import capture_v4, fit_writer as fw, ids, pressure_v2

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.statics_v2"
SCHEMA_VERSION = 1
STATICS_VERSION = "v2"
PLANT = "v2"
STATICS_DIR = ids.DATA_ROOT / "statics_v2"     # not statics/: Step 2's test refuses every record there on plant v2
LABELS_V11 = ids.DATA_ROOT / "labels" / "holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828"


def motion_dir(rid: str = capture_v4.RETARGET_ID) -> Path:
    return pressure_v2.RETARGET_ROOT / rid


def config() -> dict:
    """``statics.CONFIG`` with plant v2's model in place of the one statics bound at import."""
    import mujoco

    from reference_curation import statics

    flat = fw.plant_paths(PLANT)[1]
    mass = float(mujoco.MjModel.from_xml_path(str(flat)).body_mass.sum())
    return {**statics.CONFIG, "plant": ids.display_path(flat), "mass_kg": round(mass, 3)}


# --------------------------------------------------------------------------- #
# One hold, inside the plant context
# --------------------------------------------------------------------------- #
def audit_hold(stem: str, hold: dict, anns: list[dict], mdir: Path, run_witness: bool = True) -> dict:
    """``statics.audit_hold`` with ``witness_v2`` (plant v2's gains, the loaded-zone contact rule). Call inside
    ``retarget_v2.on_plant()``."""
    from reference_curation import statics, witness_v2

    req = statics.hold_request(anns)
    frame = int(hold["frame_hold"])
    pos, rot = statics.load_pose(stem, frame, mdir)
    gated = statics.analyse(pos, rot, req["ground"], req["pairs"], stops=True)
    missing = [n for n in req["closed"] if not gated["contacts"][n]["realised"]]
    cf = None
    if missing or gated["status"] != "optimal" or gated["beyond_plant"]:
        cf = statics.analyse(pos, rot, req["ground"], req["pairs"], closed=req["closed"])
        cf = cf if cf["counterfactual"] else None
    wit = None
    if run_witness and gated["status"] == "optimal" and not gated["beyond_plant"]:
        loaded = sorted(n for n, c in gated["contacts"].items()
                        if c["kind"] == "ground" and (c["load_n"] or 0.0) > statics.LOAD_EPS_N)
        wit = witness_v2.run(pos, rot, gated["tau"], [n for n in req["pairs"] if gated["contacts"][n]["realised"]],
                             loaded=loaded)
    status = gated["status"]
    verdict = ("held" if wit and wit["passed"] else "feasible") if status == "optimal" and not gated["beyond_plant"] \
        else ("beyond_plant" if status == "optimal" else status)
    return {"hold_id": hold["hold_id"], "stem": stem, "hold_name": hold["name"], "family_hold": bool(hold.get("extend")),
            "frame": frame, "t": hold["t_hold"], "request": req, "not_realised_human_contacts": missing,
            "verdict": verdict, "gated": statics._public(gated), "counterfactual": statics._public(cf), "witness": wit}


def _worker(job: tuple) -> tuple[list[dict], list[str]]:
    import torch

    from reference_curation import retarget_v2

    stem, holds, anns, mdir, run_witness = job
    torch.set_num_threads(1)
    out, failures = [], []
    with retarget_v2.on_plant(PLANT):
        for h in holds:
            try:
                out.append(audit_hold(stem, h, anns[h["hold_id"]], Path(mdir), run_witness))
            except Exception as exc:  # noqa: BLE001 -- report every broken hold, then fail
                failures.append(f"{h['hold_id']}: {type(exc).__name__}: {exc}")
    return out, failures


def audit_holds(labels: dict, stems: list[str], mdir: Path, workers: int = 8,
                run_witness: bool = True) -> tuple[list[dict], list[str]]:
    """Every hold of ``labels`` on ``stems``, in the labels' order, in spawned single-threaded workers."""
    from reference_curation import human_mesh as hm

    jobs = [(stem, holds, {h["hold_id"]: labels["anns"][h["hold_id"]] for h in holds}, str(mdir), run_witness)
            for stem, holds in labels["clips"] if stem in stems]
    if workers > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            parts = list(ex.map(_worker, jobs))
    else:
        parts = [_worker(j) for j in jobs]
    return [r for rs, _ in parts for r in rs], [f for _, fs in parts for f in fs]


# --------------------------------------------------------------------------- #
# Metrics and output
# --------------------------------------------------------------------------- #
def witness_metrics(records: list[dict]) -> dict:
    """What D4's loaded-zone rule changed, and the statue's motion verdict (settle and drift alone)."""
    w = [r for r in records if r["witness"]]
    flips = [r["hold_id"] for r in w if r["witness"]["passed"] and not r["witness"].get("passed_strict", True)]
    return {"run": len(w), "passed": sum(r["witness"]["passed"] for r in w),
            "passed_strict": sum(r["witness"].get("passed_strict", False) for r in w),
            "still": sum(r["witness"].get("still", False) for r in w), "passed_only_with_d4": flips}


def summary_markdown(sid: str, m: dict, wm: dict, records: list[dict], rows: list[dict], base: str) -> str:
    holdable = m["all"]["verdict"].get("held", 0) + m["all"]["verdict"].get("feasible", 0)
    head = [f"# Statics v2 `{sid}`", "",
            f"Generated by `{MODULE}` (BodyFix Step 4, item 4): Step 7's gated LP on plant v2 and the Step 3 references "
            f"(`{capture_v4.RETARGET_ID}`), with the plant-v2 statue (`witness_v2`: the `smpl_yogi_v2` gains, TODO D4's "
            "loaded-zone contact rule).", "",
            f"**LP-holdable within the plant's limits: {holdable} of {m['all']['holds']}** (held {m['all']['verdict'].get('held', 0)}, "
            f"feasible {m['all']['verdict'].get('feasible', 0)}). Statue: {wm['passed']} of {wm['run']} pass with D4's rule, "
            f"{wm['passed_strict']} with Step 7's, {wm['still']} still by motion alone (settle < 5 cm, drift < 2 cm); "
            f"{len(wm['passed_only_with_d4'])} pass only with D4.", ""]
    body = base.splitlines()[4:]   # statics.summary_markdown after its own title and generator line
    return "\n".join(head + body) + "\n"


def statics_id(labels: dict, records: list[dict], mdir: Path) -> str:
    from reference_curation import witness_v2

    stems = sorted({r["stem"] for r in records})
    key = {"schema": SCHEMA_VERSION, "config": config(), "witness": witness_v2.CONFIG, "labels": labels["id"],
           "plant": plant_identity.sha256(PLANT),
           "generators": {Path(f).name: ids.sha256_file(f) for f in generator_files()},
           "motions": {s: ids.sha256_file(ids.motion_path(s, mdir)) for s in stems},
           "holds": sorted(r["hold_id"] for r in records)}
    return f"{labels['id']}.statics_{STATICS_VERSION}.{ids.sha256_json(key)[:10]}"


def generator_files() -> list[Path]:
    import static_hold_lp as S
    import protomotions.robot_configs.smpl_yogi_v2 as yogi_v2

    from reference_curation import retarget_v2, statics, witness, witness_v2

    return [Path(__file__), Path(statics.__file__), Path(witness.__file__), Path(witness_v2.__file__), Path(S.__file__),
            Path(yogi_v2.__file__), Path(retarget_v2.__file__), *fw.plant_paths(PLANT)]


def write(labels: dict, records: list[dict], mdir: Path, out_root: Path = STATICS_DIR) -> Path:
    from reference_curation import statics, witness_v2

    rows = statics.contact_rows(records, labels)
    sid = statics_id(labels, records, mdir)
    out = Path(out_root) / sid
    out.mkdir(parents=True, exist_ok=True)
    head = {"schema_version": SCHEMA_VERSION, "statics_id": sid}
    (out / "holds.jsonl").write_text("".join(json.dumps({**head, **r}, allow_nan=False) + "\n" for r in records))
    (out / "contacts.jsonl").write_text("".join(json.dumps({**head, **x}, allow_nan=False) + "\n" for x in rows))
    m, wm = statics.metrics(records, rows), witness_metrics(records)
    inputs = [labels["dir"] / "holds.yaml", labels["dir"] / "annotations.jsonl", *fw.plant_paths(PLANT)]
    inputs += [ids.motion_path(s, mdir) for s in sorted({r["stem"] for r in records})]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "statics_id": sid, "labels_id": labels["id"],
              "plant": plant_identity.identity(PLANT), "retarget_id": mdir.name, "motion_dir": ids.display_path(mdir),
              "config": config(), "witness": witness_v2.CONFIG,
              "generators": {ids.display_path(p): ids.sha256_file(p) for p in generator_files()},
              "metrics": m, "witness_metrics": wm}
    (out / "statics.json").write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    (out / "summary.md").write_text(summary_markdown(sid, m, wm, records, rows, statics.summary_markdown(sid, m, records, rows)))
    return out


def load(statics_dir: Path) -> dict:
    """``{"record", "holds": {hold_id: row}, "contacts": {(hold_id, contact): row}}`` of a statics v2 folder,
    refused unless it was built on plant v2."""
    statics_dir = Path(statics_dir)
    record = json.loads((statics_dir / "statics.json").read_text())
    ids.require_plant(record, f"statics {statics_dir.name}", fw.plant_paths(PLANT)[0])
    holds = {json.loads(l)["hold_id"]: json.loads(l) for l in open(statics_dir / "holds.jsonl")}
    contacts = {(r["hold_id"], r["contact"]): r for r in map(json.loads, open(statics_dir / "contacts.jsonl"))}
    return {"record": record, "holds": holds, "contacts": contacts}


def main(argv: list[str] | None = None) -> int:
    from reference_curation import statics

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, default=LABELS_V11, help="a labels folder (holds.yaml + annotations.jsonl)")
    ap.add_argument("--retarget-id", default=capture_v4.RETARGET_ID)
    ap.add_argument("--stem", nargs="*")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--no-witness", action="store_true")
    ap.add_argument("--out-root", type=Path, default=STATICS_DIR)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        labels = statics.load_labels(args.labels)
        corpus, _ = fw.corpus()
        stems = [s for s in corpus if not args.stem or s in args.stem]
        mdir = motion_dir(args.retarget_id)
        records, failures = audit_holds(labels, stems, mdir, args.workers, not args.no_witness)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    failures += statics.check(records)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures or not records:
        print(f"statics_v2: {len(failures)} failures, {len(records)} holds; nothing written", file=sys.stderr)
        return 1
    out = write(labels, records, mdir, args.out_root)
    v = collections.Counter(r["verdict"] for r in records)
    wm = witness_metrics(records)
    print(f"statics_v2 {out.name}: {len(records)} holds, verdicts {dict(v)}; holdable {v['held'] + v['feasible']}; statue "
          f"{wm['passed']}/{wm['run']} (Step 7's rule {wm['passed_strict']}, still {wm['still']}) in "
          f"{time.time() - start:.0f} s -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
