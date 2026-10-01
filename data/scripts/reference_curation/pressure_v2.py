# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MOYO mat attribution re-run on plant v2 and the Step 3 references (BodyFix Step 4, item 1; TODO C3).

The per-body share of the measured mat load is decided by the reference's geometry: a cell's force goes to the
collision geoms near the floor above it (``attribute_pressure_to_bodies.py``). Every attributed field the capture
store v1 carries was measured on the shipped plant, posed by the *pressure port's* kinematics (the unrepaired
grounded conversion): ``mat_zone_load``, ``mat_unexplained``, validity columns 1 (coverage x explained) and 2 (on-mat
x explained), and ``attr_visible``. On plant v2 the references are her own motion on her own skeleton
(``retarget_v2``), so the attribution is re-run on them with plant v2's colliders, by the chain the card names,
unchanged, as subprocesses:

1. ``attribute_pressure_to_bodies.py --motion-dir <retarget_v2 motions> --mjcf <plant v2>`` (default kernel:
   sigma 2 cm, sigma_z 2 cm, z band 6 cm, max distance 6 cm);
2. ``add_pressure_to_motions.py --bodies-dir ...``: the pressure port, the retarget's ``.motion`` plus
   ``ground_reaction``, ``rigid_body_ground_forces`` and ``ground_reaction_valid`` (columns 0, 1);
3. ``add_onmat_gate_to_motions.py --mjcf <plant v2>``: column 2. **Every** clip is gated, not only the 35 the
   shipped gated port covers: the on-mat test is geometric and there is no reason to leave a clip without it.

Outputs (bulky, regenerable) under ``output/reference_curation/pressure_v2/<retarget_id>/``: ``archives/`` (links to
the corpus's Tier-0 archives), ``bodies/``, ``pressure/``, ``gated/``. The record ``data/reference_curation/
pressure_v2/<retarget_id>/pressure.json`` (+ ``summary.md``) holds the chain's provenance (every output's sha256)
and how well the mat agrees with the geometry, per clip and pooled, beside the same numbers for the shipped port
(``data/smpl/yoga_pressure_bodies``, plant v1, the unrepaired pose) and for ftR (plant v1, the Step 8 retarget):

* ``explained``: the share of the measured load that lands on some body (cells farther than 6 cm from every
  near-floor geom are left unassigned): on loaded frames, the median, and the share of frames at >= 0.9;
* ``cop_residual``: the distance between the measured COP and the COP of the attributed cells, p50 and p90.

Capture store v4 (``capture_v4``) reads the gated port.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.pressure_v2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from reference_curation import fit_writer as fw
from reference_curation import ids

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.pressure_v2"
SCHEMA_VERSION = 1
PLANT = "v2"
RETARGET_ID = "holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828.retarget_v2.e4278ada02"
RETARGET_ROOT = ids.OUTPUT_ROOT / "retarget_v2"
OUT_ROOT = ids.OUTPUT_ROOT / "pressure_v2"
RECORD_ROOT = ids.DATA_ROOT / "pressure_v2"
ARCHIVES = REPO / "data/smpl/yoga_pressure"
SCRIPTS = REPO / "data/scripts"
CHAIN = ("attribute_pressure_to_bodies.py", "add_pressure_to_motions.py", "add_onmat_gate_to_motions.py")
# The earlier attributions of the same mat, for comparison (plant v1).
BASELINES = {"shipped_port": REPO / "data/smpl/yoga_pressure_bodies",
             "ftR": REPO / "data/smpl/yoga_pressure_bodies_ftR"}
MIN_FORCE_N = 20.0          # attribute_pressure_to_bodies.py --min-force: a frame with less is not attributed
EXPLAINED_GOOD = 0.9


def paths(rid: str = RETARGET_ID, out_root: Path = OUT_ROOT) -> dict:
    out = Path(out_root) / rid
    return {"motions": RETARGET_ROOT / rid, "out": out, "archives": out / "archives", "bodies": out / "bodies",
            "pressure": out / "pressure", "gated": out / "gated"}


def gated_port(stem: str, rid: str = RETARGET_ID, out_root: Path = OUT_ROOT) -> Path:
    """The gated pressure port of a clip: the retarget's ``.motion`` with the three measured fields."""
    return paths(rid, out_root)["gated"] / f"{stem}.motion"


def _run(args: list, log) -> str:
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{SCRIPTS}")
    proc = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True, env=env)
    log.write(f"$ {' '.join(map(str, args))}\n{proc.stdout}\n{proc.stderr}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(args[0]).name} exited {proc.returncode}: {proc.stderr[-800:]}")
    return proc.stdout


def run_chain(stems: list[str], rid: str = RETARGET_ID, out_root: Path = OUT_ROOT, force: bool = False) -> dict:
    """Run the three scripts of the chain into ``out_root/<rid>/`` (refusing to overwrite unless ``force``)."""
    p = paths(rid, out_root)
    mjcf = fw.plant_paths(PLANT)[0]
    for d in ("bodies", "pressure", "gated"):
        if p[d].exists() and any(p[d].iterdir()) and not force:
            raise FileExistsError(f"{p[d]} exists; pass force to rebuild")
    if not p["motions"].is_dir():
        raise FileNotFoundError(f"no retarget motions at {p['motions']}")
    for d in ("archives", "bodies", "pressure", "gated"):
        if p[d].exists():
            for f in p[d].iterdir():
                f.unlink()
        p[d].mkdir(parents=True, exist_ok=True)
    missing = [s for s in stems if not (ARCHIVES / f"{s}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"no Tier-0 archive for {missing}")
    for s in stems:
        (p["archives"] / f"{s}.npz").symlink_to((ARCHIVES / f"{s}.npz").resolve())
    with open(p["out"] / "chain.log", "w") as log:
        _run([SCRIPTS / CHAIN[0], "--archive-dir", p["archives"], "--motion-dir", p["motions"], "--mjcf", mjcf,
              "--out-dir", p["bodies"]], log)
        _run([SCRIPTS / CHAIN[1], "--in-dir", p["motions"], "--archive-dir", ARCHIVES, "--bodies-dir", p["bodies"],
              "--out-dir", p["pressure"]], log)
        _run([SCRIPTS / CHAIN[2], "--in-dir", p["pressure"], "--archive-dir", ARCHIVES, "--out-dir", p["gated"],
              "--mjcf", mjcf, "--clips", *stems], log)
    return p


# --------------------------------------------------------------------------- #
# How well the mat agrees with the geometry
# --------------------------------------------------------------------------- #
def attribution_stats(bodies_npz: Path, archive_npz: Path) -> dict:
    """On loaded frames (total >= ``MIN_FORCE_N``): explained median and share >= 0.9, COP residual p50 / p90."""
    from pressure_bodies import load_archive

    b = np.load(bodies_npz)
    total = load_archive(archive_npz)["total_force_n"]
    loaded = np.nan_to_num(total, nan=0.0) >= MIN_FORCE_N
    e = np.asarray(b["explained"], float)[loaded]
    r = np.asarray(b["cop_residual_m"], float)[loaded]
    r = r[np.isfinite(r)]
    return {"loaded_frames": int(loaded.sum()),
            "explained_p50": round(float(np.median(e)), 4) if len(e) else None,
            "explained_ge_0_9": round(float((e >= EXPLAINED_GOOD).mean()), 4) if len(e) else None,
            "cop_residual_cm_p50": round(100 * float(np.median(r)), 3) if len(r) else None,
            "cop_residual_cm_p90": round(100 * float(np.percentile(r, 90)), 3) if len(r) else None,
            "_e": e, "_r": r}


def _pooled(rows: list[dict]) -> dict:
    e = np.concatenate([x["_e"] for x in rows]) if rows else np.zeros(0)
    r = np.concatenate([x["_r"] for x in rows]) if rows else np.zeros(0)
    return {"clips": len(rows), "loaded_frames": int(len(e)),
            "explained_p50": round(float(np.median(e)), 4), "explained_ge_0_9": round(float((e >= EXPLAINED_GOOD).mean()), 4),
            "cop_residual_cm_p50": round(100 * float(np.median(r)), 3),
            "cop_residual_cm_p90": round(100 * float(np.percentile(r, 90)), 3)}


def port_stats(port: Path) -> dict:
    """Validity columns of a gated port: medians and the shares of frames at >= 0.9."""
    import torch

    v = torch.load(port, map_location="cpu", weights_only=False)["ground_reaction_valid"].double().numpy()
    return {f"col{k}_ge_0_9": round(float((v[:, k] >= 0.9).mean()), 4) for k in range(v.shape[1])}


def measure(stems: list[str], p: dict) -> dict:
    per, pooled = {}, {}
    sets = {"v2": p["bodies"], **BASELINES}
    for name, d in sets.items():
        rows = []
        for s in stems:
            f = Path(d) / f"{s}.npz"
            if f.exists():
                st = attribution_stats(f, ARCHIVES / f"{s}.npz")
                rows.append(st)
                per.setdefault(s, {})[name] = {k: v for k, v in st.items() if not k.startswith("_")}
        pooled[name] = _pooled(rows)
    for s in stems:
        per[s]["v2_port"] = port_stats(p["gated"] / f"{s}.motion")
    onmat = json.loads((p["gated"] / "index.json").read_text())
    return {"per_clip": per, "pooled": pooled,
            "onmat_index": {c["clip"]: {k: round(c[k], 4) for k in ("on_mat_frac", "gate_cov_expl", "gate_onmat_expl")}
                            for c in onmat["clips"]}}


def check(stems: list[str], p: dict) -> list[str]:
    """The chain's contract: every clip attributed, ported and gated, on plant v2, with three validity columns."""
    import torch

    problems = []
    sha = plant_identity.sha256(PLANT)
    for s in stems:
        for d, suffix in (("bodies", ".npz"), ("pressure", ".motion"), ("gated", ".motion")):
            if not (p[d] / f"{s}{suffix}").exists():
                problems.append(f"{s}: no {d} output")
        g = p["gated"] / f"{s}.motion"
        if g.exists():
            m = torch.load(g, map_location="cpu", weights_only=False)
            if m.get(plant_identity.KEY) != sha:
                problems.append(f"{s}: gated port is not on plant v2")
            if tuple(m["ground_reaction_valid"].shape[1:]) != (3,):
                problems.append(f"{s}: ground_reaction_valid has {m['ground_reaction_valid'].shape[1]} columns")
            src = torch.load(p["motions"] / f"{s}.motion", map_location="cpu", weights_only=False)
            if not torch.equal(m["rigid_body_pos"], src["rigid_body_pos"]):
                problems.append(f"{s}: the port's kinematics differ from the retarget's")
    return problems


def record(rid: str, stems: list[str], dropped: dict, p: dict, m: dict, seconds: float) -> dict:
    outputs = {f"{d}/{f.name}": ids.sha256_file(f) for d in ("bodies", "gated")
               for f in sorted(p[d].iterdir()) if f.suffix in (".npz", ".motion")}
    inputs = [fw.plant_paths(PLANT)[0], *(SCRIPTS / c for c in CHAIN), SCRIPTS / "pressure_bodies.py"]
    inputs += [p["motions"] / f"{s}.motion" for s in stems] + [ARCHIVES / f"{s}.npz" for s in stems]
    return {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "retarget_id": rid,
            "plant": plant_identity.identity(PLANT), "out_dir": ids.display_path(p["out"]), "stems": stems,
            "dropped": dropped, "chain": list(CHAIN), "kernel": {"sigma_m": 0.02, "sigma_z_m": 0.02, "z_thresh_m": 0.06,
                                                              "max_dist_m": 0.06, "min_force_n": MIN_FORCE_N,
                                                              "near_ground_m": 0.15, "mat_margin_m": 0.02},
            "outputs": outputs, "seconds": round(seconds, 1), **m}


def summary_markdown(rec: dict) -> str:
    po = rec["pooled"]
    lines = [f"# Mat attribution on plant v2 `{rec['retarget_id']}`", "",
             f"Generated by `{MODULE}` (BodyFix Step 4, item 1): the chain `{' -> '.join(rec['chain'])}` run unchanged "
             "on the Step 3 references with plant v2's colliders, every clip gated. Loaded frames: mat total >= "
             f"{MIN_FORCE_N:g} N.", "",
             "| Attribution | Clips | Loaded frames | Explained p50 | Frames explained >= 0.9 | COP residual p50 / p90 (cm) |",
             "|---|---|---|---|---|---|"]
    names = {"shipped_port": "plant v1, the shipped port's pose (capture store v1)", "ftR": "plant v1, Step 8 retarget (ftR)",
             "v2": "**plant v2, Step 3 retarget (this record)**"}
    for k in ("shipped_port", "ftR", "v2"):
        x = po[k]
        lines.append(f"| {names[k]} | {x['clips']} | {x['loaded_frames']} | {x['explained_p50']} | "
                     f"{100 * x['explained_ge_0_9']:.1f} % | {x['cop_residual_cm_p50']} / {x['cop_residual_cm_p90']} |")
    worst = sorted(rec["per_clip"].items(), key=lambda kv: kv[1]["v2"]["explained_ge_0_9"])[:8]
    lines += ["", "Clips with the fewest well-explained loaded frames on plant v2 (share explained >= 0.9: v2 / ftR / "
              "shipped port):", ""]
    for s, v in worst:
        lines.append(f"- `{s}`: {v['v2']['explained_ge_0_9']} / {v.get('ftR', {}).get('explained_ge_0_9')} / "
                     f"{v.get('shipped_port', {}).get('explained_ge_0_9')}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--retarget-id", default=RETARGET_ID)
    ap.add_argument("--force", action="store_true", help="rebuild the chain's outputs")
    ap.add_argument("--measure-only", action="store_true", help="re-measure existing outputs")
    args = ap.parse_args(argv)
    start = time.time()
    try:
        stems, dropped = fw.corpus()
        p = paths(args.retarget_id)
        if not args.measure_only:
            run_chain(stems, args.retarget_id, force=args.force)
        failures = check(stems, p)
        m = measure(stems, p)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures:
        return 1
    rec = record(args.retarget_id, stems, dropped, p, m, time.time() - start)
    out = RECORD_ROOT / args.retarget_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "pressure.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "summary.md").write_text(summary_markdown(rec))
    po = rec["pooled"]
    print(f"pressure_v2 {args.retarget_id}: {len(stems)} clips attributed and gated on plant v2 in {rec['seconds']:.0f} s; "
          f"explained >= 0.9 on {100 * po['v2']['explained_ge_0_9']:.1f} % of loaded frames (ftR "
          f"{100 * po['ftR']['explained_ge_0_9']:.1f} %, shipped port {100 * po['shipped_port']['explained_ge_0_9']:.1f} %); "
          f"COP residual p90 {po['v2']['cop_residual_cm_p90']} cm (ftR {po['ftR']['cop_residual_cm_p90']}, shipped "
          f"{po['shipped_port']['cop_residual_cm_p90']}) -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
