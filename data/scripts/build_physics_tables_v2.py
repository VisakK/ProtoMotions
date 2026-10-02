# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics tables v2: fine-tune C's tables built on a release (BUILD_PLAN Step 9, BodyFix Step 5).

The same three things as ``build_physics_tables.py`` (v1), with v1's kernels and thresholds -- swing
labels ``[M, T, Z]``, per-segment load signatures, body constants -- and these changes:

1. **Each motion's own pressure frames.** Hold extension v2 (``reference_curation.hold_extension_v2``)
   carries the measured mat channels into every duration variant on its real frames and marks the
   inserted frames invalid (``ground_reaction_valid`` 0). A variant frame therefore reads its own
   measurement. v1 re-timed the x0 port through ``splice_index``, which repeated the exemplar's
   measurement over the inserted frames, and read it at ``pose_repair.from``'s kinematics.
2. **Column 1 for absolute thresholds, column 2 only for normalised quantities.** The pressure rule's
   "unloaded" (< ``--unloaded-n``) and the veto (> ``--loaded-n``) compare newtons, so they need the mat to
   have measured the whole load: column 1 (coverage x explained) >= 0.9. Column 2 (on-mat x explained)
   also admits a uniform gain deficit -- forearm holds read 0.74-0.83 body weight -- which leaves shares
   and the COP right and the newtons wrong. So column 2 gates the load shares, and with column 0 the COP,
   and nothing else. v1 OR-ed column 2 into the absolute rule; that is why ftR's pressure-only swing
   labels went from 18k to 51k frames (TODO C3).
3. **The attribution-visibility gate on "unloaded".** The attribution gives a cell's load only to geoms
   within 6 cm of the floor (``ATTR_BAND_M``) and over the mat, so outside that a zone reads 0 N whatever
   it carries. "Unloaded" counts only where the zone's lowest patch point is within the band and inside
   the mat's sensing rectangle (+ ``--mat-margin``; a 2x2 solve, because the mat axes are not
   orthonormal).
4. **Identity.** ``version`` 2; the sha256 of the graph, the package and the manifest; the graph's
   ``pair_names`` (``PhysicsTables`` refuses another graph's); fps checked on every motion and the swing
   rows' lengths equal to the motions'; ``swing_source`` (1 velocity, 2 pressure, 3 both) for the
   calibration re-check. The stats also count the labels v1's pressure rule would give on the same
   motions, so the effect of 2-3 is measured, not assumed.

    PYTHONPATH=.:data/scripts python data/scripts/build_physics_tables_v2.py \\
      --extended-manifest <release>/holds_extended.yaml --graph <release>/contact_graph.pt \\
      --motion-dir <release>/motions --motion-file <release>/motions.pt --out <release>/physics_tables.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_v2.xml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from build_physics_tables import (  # noqa: E402
    LEAN_SUPPORT,
    PRESSURE_ZONES,
    body_constants,
    consequential,
    median_filter,
    patch_points,
    zone_patch_speed,
)
from extract_contact_configs import ZONE_ORDER, ZONES, mjcf_body_names  # noqa: E402
from protomotions.utils import plant_identity  # noqa: E402
from protomotions.utils.rotations import quat_rotate  # noqa: E402

TABLES_VERSION = 2
ARCHIVES = REPO / "data/smpl/yoga_pressure"
ATTR_BAND_M = 0.06          # attribute_pressure_to_bodies.py: only geoms this close to the floor compete for load
VALID = 0.9                 # every validity column's gate
SWING_SOURCE = {"none": 0, "velocity": 1, "pressure": 2, "both": 3}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mat_frame(archive) -> dict:
    """The mat's sensing rectangle in the clip frame: origin, the 2x2 axis matrix and its extent (m)."""
    origin = np.asarray(archive["mat_origin_xy"], np.float64)
    axes = np.stack([np.asarray(archive["mat_ex"], np.float64), np.asarray(archive["mat_ey"], np.float64)], 1)
    n_rows, n_cols = [int(v) for v in archive["mat_shape"]]
    cell = float(archive["cell_size_m"])
    # mat_ex runs along columns, mat_ey along rows (moyo_pressure_io convention)
    return {"origin": origin, "inv": np.linalg.inv(axes), "len": np.array([n_cols * cell, n_rows * cell])}


def on_mat(xy: np.ndarray, mat: dict, margin_m: float) -> np.ndarray:
    """``[...]`` True where clip-frame points ``xy [..., 2]`` lie inside the sensing rectangle + margin."""
    uv = (xy - mat["origin"]) @ mat["inv"].T
    return ((uv >= -margin_m) & (uv <= mat["len"] + margin_m)).all(-1)


def zone_lowest_point(pos, rot, consts, bodies) -> tuple[np.ndarray, np.ndarray]:
    """``(z [T], xy [T, 2])`` of the zone's lowest patch point (the selection the speed kernel uses)."""
    world, _ = patch_points(pos, rot, consts, bodies)
    low = world[..., 2].argmin(dim=1)
    point = world[torch.arange(world.shape[0]), low]
    return point[:, 2].numpy().astype(np.float64), point[:, :2].numpy().astype(np.float64)


def pressure_fields(motion: dict):
    """``(forces_z [T, B], valid [T, 3], ground_reaction [T, 3])`` or None when the motion has no mat."""
    if motion.get("ground_reaction") is None:
        return None
    valid = motion["ground_reaction_valid"].double()
    if valid.shape[1] < 3:
        raise ValueError("ground_reaction_valid has fewer than 3 columns: run add_onmat_gate_to_motions.py first")
    return motion["rigid_body_ground_forces"][..., 2].double(), valid, motion["ground_reaction"].double()


def swing_rules(speed: np.ndarray, zmin: np.ndarray, visible: np.ndarray, load: np.ndarray | None,
                valid: np.ndarray | None, zone: str, args) -> dict:
    """Per-frame boolean pieces of one zone's swing label under the v2 rules, plus v1's for comparison."""
    T = len(speed)
    smooth = median_filter(speed, args.speed_filter)
    w = int(round(args.lift_window_s * args.fps))
    zpad = np.pad(zmin, (w, w), mode="edge")
    local_min = np.lib.stride_tricks.sliding_window_view(zpad, 2 * w + 1).min(-1)
    lifted = zmin - local_min > args.lift_m
    vel = (smooth > args.swing_speed) & lifted
    moving = smooth > args.moving_speed
    false = np.zeros(T, dtype=bool)
    out = {"vel": vel, "unl": false, "veto": false, "unl_v1": false, "veto_v1": false}
    if load is None:
        return out
    col0, col1, col2 = (valid[:, k] >= VALID for k in range(3))
    near = zmin < args.near_floor_m
    out["veto"] = col1 & (load > args.loaded_n)
    out["veto_v1"] = (col1 | col2) & (load > args.loaded_n)
    if zone in PRESSURE_ZONES:
        out["unl"] = col1 & visible & (load < args.unloaded_n) & near & moving
        out["unl_v1"] = (col1 | col2) & (load < args.unloaded_n) & near & moving
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extended-manifest", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--motion-file", required=True, help="the packaged library the graph was built from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive-dir", default=str(ARCHIVES), help="Tier-0 archives: the mat's geometry per clip")
    ap.add_argument("--mjcf", default=str(plant_identity.mjcf_path()),
                    help="the plant the tables are for; every motion must record it")
    ap.add_argument("--fps", type=int, default=60)
    # v1's thresholds, unchanged
    ap.add_argument("--swing-speed", type=float, default=0.25)
    ap.add_argument("--speed-filter", type=int, default=5)
    ap.add_argument("--lift-m", type=float, default=0.01)
    ap.add_argument("--lift-window-s", type=float, default=0.5)
    ap.add_argument("--moving-speed", type=float, default=0.05)
    ap.add_argument("--unloaded-n", type=float, default=14.0)
    ap.add_argument("--loaded-n", type=float, default=70.0)
    ap.add_argument("--near-floor-m", type=float, default=0.10)
    ap.add_argument("--mat-margin", type=float, default=0.02)
    args = ap.parse_args()

    plant = plant_identity.identity(args.mjcf)
    body_names = mjcf_body_names(str(plant_identity.mjcf_path(args.mjcf)))
    consts = body_constants(body_names, plant_identity.mjcf_path(args.mjcf))
    zone_bodies = {z: [body_names.index(b) for b in ZONES[z]] for z in ZONE_ORDER}
    graph_path, package_path, manifest_path = (Path(p).resolve() for p in (args.graph, args.motion_file,
                                                                             args.extended_manifest))
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    if int(graph.get("graph_version", 1)) < 2:
        raise ValueError(f"{graph_path} is not a v2 graph (build_hold_graph_v2.py)")
    package_sha = sha256(package_path)
    if graph.get("package_sha256") != package_sha:
        raise ValueError("the graph was built from another package than --motion-file")
    names = list(graph["motion_names"])
    pair_names = list(graph["pair_names"])
    if list(graph.get("zone_order") or ZONE_ORDER) != list(ZONE_ORDER):
        raise ValueError("graph zone order differs from the extractor's")
    if int(graph["fps"]) != args.fps:
        raise ValueError(f"the graph is at {graph['fps']} fps, --fps {args.fps}")
    ext = {c["stem"]: c for c in yaml.safe_load(open(manifest_path))["clips"]}
    if sorted(ext) != sorted(names):
        raise ValueError("the extended manifest and the graph name different motions")

    M, Z = len(names), len(ZONE_ORDER)
    S_max = graph["seg_node"].shape[1]
    lengths, swings, sources = [], [], []
    seg_cop_rel = torch.zeros(M, S_max, 2)
    seg_cop_valid = torch.zeros(M, S_max, dtype=torch.bool)
    seg_com_rel = torch.zeros(M, S_max, 2)
    seg_zone_share = torch.zeros(M, S_max, Z)
    seg_share_valid = torch.zeros(M, S_max, dtype=torch.bool)
    seg_lean_gate = torch.zeros(M, S_max, dtype=torch.bool)
    seg_pair_conseq = torch.zeros(M, S_max, len(pair_names), dtype=torch.bool)
    conseq_slots = torch.tensor([consequential(p) for p in pair_names])
    keys = ("velocity_only", "pressure_only", "both", "vetoed", "pressure_only_v1_rule", "vetoed_v1_rule",
            "labels_v1_rule", "labels")
    stats = dict(rules={"absolute_newtons": "column 1 >= 0.9",
                        "shares": "column 2 >= 0.9",
                        "cop": "column 0 or column 2 >= 0.9",
                        "unloaded": f"zone's lowest patch point within {ATTR_BAND_M} m of the floor and over the mat "
                                    f"(+{args.mat_margin} m)",
                        "inserted_frames": "ground_reaction_valid 0: no pressure label, no signature frame"},
                 swing_share={}, frames=0, pressure_motions=0, unloaded_invisible_frames=0,
                 **{k: 0 for k in keys}, per_zone={z: {k: 0 for k in keys} for z in ZONE_ORDER})
    mass, com_local = consts["body_mass"], consts["body_com_local"]
    for m, stem in enumerate(names):
        e = ext[stem]
        motion = torch.load(Path(args.motion_dir) / f"{stem}.motion", map_location="cpu", weights_only=False)
        plant_identity.require(motion.get(plant_identity.KEY), args.mjcf, f"{stem}.motion")
        if int(motion["fps"]) != args.fps:
            raise ValueError(f"{stem}: {motion['fps']} fps, the tables are at {args.fps}")
        pos, rot = motion["rigid_body_pos"].float(), motion["rigid_body_rot"].float()
        T = pos.shape[0]
        if T != int(e["num_frames"]) or T != int(graph["motion_num_frames"][m]):
            raise ValueError(f"{stem}: {T} frames, the manifest / graph disagree")
        lengths.append(T)
        pressure = pressure_fields(motion)
        mat = None
        if pressure is not None:
            stats["pressure_motions"] += 1
            archive = Path(args.archive_dir) / f"{e['source_stem']}.npz"
            if not archive.exists():
                raise FileNotFoundError(f"{stem}: carries pressure but its archive {archive} is missing")
            with np.load(archive, allow_pickle=True) as a:
                mat = mat_frame(a)
            fz, valid, gr = pressure
            valid_np = valid.numpy()
        swing = torch.zeros(T, Z, dtype=torch.bool)
        source = torch.zeros(T, Z, dtype=torch.int8)
        for z, zone in enumerate(ZONE_ORDER):
            speed, _ = zone_patch_speed(pos, rot, consts, zone_bodies[zone], args.fps)
            zmin, xy = zone_lowest_point(pos, rot, consts, zone_bodies[zone])
            load = visible = None
            if pressure is not None:
                load = fz[:, zone_bodies[zone]].sum(-1).numpy()
                visible = (zmin <= ATTR_BAND_M) & on_mat(xy, mat, args.mat_margin)
            r = swing_rules(speed.numpy().astype(np.float64), zmin, visible, load,
                            valid_np if pressure is not None else None, zone, args)
            label = (r["vel"] | r["unl"]) & ~r["veto"]
            label_v1 = (r["vel"] | r["unl_v1"]) & ~r["veto_v1"]
            swing[:, z] = torch.from_numpy(label)
            src = np.where(r["vel"] & r["unl"], 3, np.where(r["unl"], 2, np.where(r["vel"], 1, 0)))
            source[:, z] = torch.from_numpy((src * label).astype(np.int8))
            counts = {"velocity_only": int((r["vel"] & ~r["unl"] & ~r["veto"]).sum()),
                      "pressure_only": int((r["unl"] & ~r["vel"] & ~r["veto"]).sum()),
                      "both": int((r["unl"] & r["vel"] & ~r["veto"]).sum()),
                      "vetoed": int(((r["vel"] | r["unl"]) & r["veto"]).sum()),
                      "pressure_only_v1_rule": int((r["unl_v1"] & ~r["vel"] & ~r["veto_v1"]).sum()),
                      "vetoed_v1_rule": int(((r["vel"] | r["unl_v1"]) & r["veto_v1"]).sum()),
                      "labels_v1_rule": int(label_v1.sum()), "labels": int(label.sum())}
            for k, v in counts.items():
                stats[k] += v
                stats["per_zone"][zone][k] += v
            if pressure is not None and zone in PRESSURE_ZONES:
                stats["unloaded_invisible_frames"] += int((r["unl_v1"] & ~visible).sum())
        stats["frames"] += T
        swings.append(swing)
        sources.append(source)

        # ---- per-segment load signatures (variant frames; inserted frames carry validity 0) ---- #
        holds = {round(float(h["t_hold"]), 3): h for h in e["holds"]}
        for k in range(int(graph["seg_count"][m])):
            h = holds.get(round(float(graph["seg_hold"][m, k]), 3))
            if h is None:
                raise ValueError(f"{stem}: graph segment {k} is not in the manifest")
            ground = [p.split(":")[0] for p in h["pairs"] if p.endswith(":G")]
            gset = set(ground)
            seg_lean_gate[m, k] = bool(gset & {"L_HAND", "R_HAND"}) and gset <= LEAN_SUPPORT
            seg_pair_conseq[m, k] = (graph["seg_contact"][m, k] > 0.5) & conseq_slots
            fh = min(int(round(float(h["t_hold"]) * args.fps)), T - 1)
            bodies = [b for z in ground for b in zone_bodies[z]]
            if not bodies:
                continue
            pts, _ = patch_points(pos[fh:fh + 1], rot[fh:fh + 1], consts, bodies)
            centroid = pts[0, :, :2].mean(0)
            com = ((pos[fh] + quat_rotate(rot[fh], com_local, w_last=True)) * mass[:, None]).sum(0) / mass.sum()
            seg_com_rel[m, k] = com[:2] - centroid
            if pressure is None:
                continue
            f0 = int(round(float(h["t_start"]) * args.fps))
            f1 = min(int(round(float(h["t_end"]) * args.fps)), T - 1)
            window = torch.arange(f0, f1 + 1)
            v = valid[window]
            ok = window[(v[:, 0] >= VALID) | (v[:, 2] >= VALID)]
            if len(ok) >= 10:
                seg_cop_rel[m, k] = gr[ok, 1:3].float().median(dim=0).values - centroid
                seg_cop_valid[m, k] = True
            okb = window[v[:, 2] >= VALID]
            if len(okb) >= 10:
                zl = torch.stack([fz[okb][:, zone_bodies[z]].sum(-1) for z in ZONE_ORDER], -1)
                share = zl / zl.sum(-1, keepdim=True).clamp(min=1e-6)
                seg_zone_share[m, k] = share.median(dim=0).values.float()
                seg_share_valid[m, k] = True
        stats["swing_share"][stem] = {z: round(float(swing[:, i].float().mean()), 3)
                                      for i, z in enumerate(ZONE_ORDER) if swing[:, i].any()}

    T_max = max(lengths)
    swing_all = torch.zeros(M, T_max, Z, dtype=torch.bool)
    source_all = torch.zeros(M, T_max, Z, dtype=torch.int8)
    for m, (sw, so) in enumerate(zip(swings, sources)):
        swing_all[m, : sw.shape[0]] = sw
        source_all[m, : so.shape[0]] = so
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = dict(
        version=TABLES_VERSION, motion_names=names, fps=args.fps, zone_order=list(ZONE_ORDER),
        plant=plant, plant_sha256=plant[plant_identity.KEY],
        zone_bodies={z: list(ZONES[z]) for z in ZONE_ORDER},
        swing=swing_all, swing_len=torch.tensor(lengths), swing_source=source_all, swing_source_codes=SWING_SOURCE,
        seg_cop_rel=seg_cop_rel, seg_cop_valid=seg_cop_valid, seg_com_rel=seg_com_rel,
        seg_zone_share=seg_zone_share, seg_share_valid=seg_share_valid,
        seg_lean_gate=seg_lean_gate, seg_pair_consequential=seg_pair_conseq,
        pair_names=pair_names, params=params,
        graph_sha256=sha256(graph_path), package_sha256=package_sha, manifest_sha256=sha256(manifest_path),
        rules=stats["rules"], **consts,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {args.out}: {M} motions, T_max {T_max}, {Z} zones; pressure on {stats['pressure_motions']} motions; "
          f"swing labels {stats['labels']} (v1's pressure rule on the same motions: {stats['labels_v1_rule']}): "
          f"velocity-only {stats['velocity_only']}, pressure-only {stats['pressure_only']} "
          f"(v1 rule {stats['pressure_only_v1_rule']}), both {stats['both']}, vetoed {stats['vetoed']}; "
          f"lean-gated segments {int(seg_lean_gate.sum())}, COP-valid {int(seg_cop_valid.sum())}, "
          f"share-valid {int(seg_share_valid.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
