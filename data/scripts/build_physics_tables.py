# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tables for fine-tune C's physics terms (``protomotions/envs/control/physics_terms.py``).

Three things, keyed to a packaged motion library and its hold graph:

**Swing labels** ``swing [M, T, Z]``: at clip frame t, zone z of the reference carries no load
and must not be carried by the policy either. ``expert_revist/contact_balance_investigation``
§2: the Warrior II / Upward Plank drag is a limb the human *unloaded* (0 N on the mat) while
lifting it only 2.5-5 cm -- float-bias sized, invisible to tracking -- which the policy slides
along the floor under load instead. Two sources, OR-ed, with a pressure veto:

* velocity rule (every zone): the reference zone's contact patch moves faster than
  ``--swing-speed`` horizontally *and* has risen ``--lift-m`` above its own lowest height within
  ``--lift-window-s`` (a step lifts, a deliberate slide -- heels drawn in for Bridge -- does
  not; the float bias is a constant offset and cancels). The speed is the slowest material
  point of the patch (bottom-face corners of a box, the two ends of a capsule, the bottom of a
  sphere), so a pivot or a roll reads ~0 and only a translating patch counts; median-filtered.
* pressure rule (feet and hands, where the per-body channel is valid): the mat puts less than
  ``--unloaded-n`` on the zone while the reference keeps it within ``--near-floor-m`` and moves
  it faster than ``--moving-speed`` (a placed, still limb reading ~0 N is touching, not
  swinging -- pincha's hands, unloaded by the forearms).
* veto: the mat puts more than ``--loaded-n`` on the zone (valid frames only). This is what
  stops a reference foot that skates while loaded -- the SMPL fit's pivots -- from being
  labelled a swing.

Hold variants (``_x3s`` / ``_x7s``) are re-timed with ``make_hold_extended_clips.splice_index``;
pressure is read from the clip's MOYO port (``yoga_motions_proto_yogi_pressure[_gated]``) at the
original, unrepaired clip's frames (the pose repair changes no timing).

**Load signatures** per graph segment: the human COP relative to the centroid of the hold's
ground support (world XY; ``seg_cop_rel`` + ``seg_cop_valid``), the reference COM relative to
the same centroid on the sim's mass model (``seg_com_rel``), median per-zone load shares
(``seg_zone_share`` + ``seg_share_valid``, L/R split), the labelled leg-on-arm / leg-on-trunk
pairs (``seg_pair_consequential``), and the lean gate (``seg_lean_gate``: the ground set has a
hand and nothing but hands, forearms and head).

**Body constants** for the runtime kernels (COMMON body order): masses and COM offsets from the
MJCF geom densities (74.0 kg), and every body's single collision geom (box / capsule / sphere).

    PYTHONPATH=. python data/scripts/build_physics_tables.py \\
      --extended-manifest data/smpl/yoga_motions_proto_yogi_expert60_ftC/holds_extended.yaml \\
      --source-manifest data/smpl/expert60/holds_repaired_ftC_posefix.yaml \\
      --graph data/smpl/yoga_hold_graph_expert60_ftC/contact_graph.pt \\
      --motion-dir data/smpl/yoga_motions_proto_yogi_expert60_ftC \\
      --out data/smpl/yoga_hold_graph_expert60_ftC/physics_tables.pt
"""

from __future__ import annotations

import argparse
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

from contact_geometry import parse_typed_geoms  # noqa: E402
from extract_contact_configs import ZONE_ORDER, ZONES, mjcf_body_names  # noqa: E402
from make_hold_extended_clips import insertion_plan, splice_index  # noqa: E402
from protomotions.utils.rotations import quat_rotate, quat_rotate_inverse  # noqa: E402

MJCF = REPO / "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
PRESSURE = REPO / "data/smpl/yoga_motions_proto_yogi_pressure"
PRESSURE_GATED = REPO / "data/smpl/yoga_motions_proto_yogi_pressure_gated"
LEG = {"L_FOOT", "R_FOOT", "L_SHANK", "R_SHANK", "L_THIGH", "R_THIGH"}
ARM_OR_TRUNK = {"L_UPPER_ARM", "R_UPPER_ARM", "L_FOREARM", "R_FOREARM", "TRUNK"}
LEAN_SUPPORT = {"L_HAND", "R_HAND", "L_FOREARM", "R_FOREARM", "HEAD"}
PRESSURE_ZONES = ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND")
BOX_SIGNS = torch.tensor([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


def consequential(pair: str) -> bool:
    if ":G" in pair:
        return False
    a, b = pair.split("+")
    return (a in LEG and b in ARM_OR_TRUNK) or (b in LEG and a in ARM_OR_TRUNK)


# --------------------------------------------------------------------------- #
# Body constants
# --------------------------------------------------------------------------- #
def body_constants(body_names):
    typed = parse_typed_geoms(str(MJCF), body_names)
    B = len(body_names)
    c = dict(
        body_names=list(body_names),
        body_mass=torch.zeros(B), body_com_local=torch.zeros(B, 3),
        geom_type=torch.zeros(B, dtype=torch.long),  # 0 box, 1 capsule, 2 sphere
        box_center=torch.zeros(B, 3), box_half=torch.zeros(B, 3), box_quat=torch.zeros(B, 4),
        cap_a=torch.zeros(B, 3), cap_b=torch.zeros(B, 3), radius=torch.zeros(B),
        sph_center=torch.zeros(B, 3),
    )
    c["box_quat"][:, 3] = 1.0
    for i, b in enumerate(body_names):
        geoms = typed[b]
        assert len(geoms) == 1, f"{b}: expected one collision geom"
        g = geoms[0]
        c["body_mass"][i] = float(g["mass"])
        if g["type"] == "box":
            c["geom_type"][i] = 0
            c["box_center"][i] = torch.as_tensor(g["center"])
            c["box_half"][i] = torch.as_tensor(g["half"])
            c["box_quat"][i] = torch.as_tensor(g["quat"])
            c["body_com_local"][i] = torch.as_tensor(g["center"])
        elif g["type"] == "capsule":
            c["geom_type"][i] = 1
            c["cap_a"][i] = torch.as_tensor(g["seg"][0])
            c["cap_b"][i] = torch.as_tensor(g["seg"][1])
            c["radius"][i] = float(g["radius"])
            c["body_com_local"][i] = torch.as_tensor((g["seg"][0] + g["seg"][1]) / 2)
        else:
            c["geom_type"][i] = 2
            c["sph_center"][i] = torch.as_tensor(g["center"])
            c["radius"][i] = float(g["radius"])
            c["body_com_local"][i] = torch.as_tensor(g["center"])
    total = float(c["body_mass"].sum())
    assert abs(total - 74.0) < 0.05, f"MJCF mass model totals {total} kg, expected 74.0"
    return c


def patch_points(pos, rot, consts, bodies):
    """World candidate contact points of ``bodies``: ``[T, n, 3]`` and the body index of each.

    Box: its 4 lowest corners (the bottom face, whatever the tilt); capsule: both ends lowered
    by the radius; sphere: its bottom. The same selection ``physics_terms`` uses at runtime.
    """
    pts, owners = [], []
    for i in bodies:
        p, q = pos[:, i], rot[:, i]
        t = int(consts["geom_type"][i])
        if t == 0:
            local = consts["box_center"][i] + quat_rotate(
                consts["box_quat"][i].expand(8, 4), BOX_SIGNS * consts["box_half"][i], w_last=True)
            world = quat_rotate(q.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4),
                                local.expand(p.shape[0], 8, 3).reshape(-1, 3), w_last=True).view(-1, 8, 3) + p[:, None]
            low = world[..., 2].argsort(dim=1)[:, :4]
            world = torch.gather(world, 1, low.unsqueeze(-1).expand(-1, 4, 3))
            pts.append(world)
            owners += [i] * 4
        elif t == 1:
            for e in (consts["cap_a"][i], consts["cap_b"][i]):
                w = quat_rotate(q, e.expand(p.shape[0], 3), w_last=True) + p
                w = w.clone()
                w[:, 2] -= consts["radius"][i]
                pts.append(w[:, None])
                owners.append(i)
        else:
            w = quat_rotate(q, consts["sph_center"][i].expand(p.shape[0], 3), w_last=True) + p
            w = w.clone()
            w[:, 2] -= consts["radius"][i]
            pts.append(w[:, None])
            owners.append(i)
    return torch.cat(pts, dim=1), owners


def zone_patch_speed(pos, rot, consts, bodies, fps):
    """[T] slowest horizontal material speed over a zone's patch points (finite differences).

    Each point is carried in its body's frame from frame t to t+1, so a pivot or a roll about a
    patch point reads ~0 for that point.
    """
    world, owners = patch_points(pos, rot, consts, bodies)
    T, n = world.shape[:2]
    speeds = torch.full((T, n), float("inf"))
    for j, i in enumerate(owners):
        p, q = pos[:, i], rot[:, i]
        local = quat_rotate_inverse(q, world[:, j] - p, w_last=True)
        nq = torch.cat([q[1:], q[-1:]])
        npos = torch.cat([p[1:], p[-1:]])
        moved = quat_rotate(nq, local, w_last=True) + npos
        speeds[:, j] = (moved - world[:, j])[:, :2].norm(dim=-1) * fps
    return speeds.min(dim=1).values, world[..., 2].min(dim=1).values


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    h = k // 2
    xp = np.pad(x, (h, h), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=-1)


def load_pressure(stem, original_pos):
    """(per-zone load [T, Z] N, valid [T] per-body gate, cop [T, 2], cop_valid [T]) or None."""
    for path, extra_col in ((PRESSURE_GATED / f"{stem}.motion", 2), (PRESSURE / f"{stem}.motion", None)):
        if not path.exists():
            continue
        pm = torch.load(str(path), map_location="cpu", weights_only=False)
        if "ground_reaction" not in pm:
            continue
        if pm["rigid_body_pos"].shape != original_pos.shape or \
                float((pm["rigid_body_pos"] - original_pos).abs().max()) > 1e-4:
            raise ValueError(f"{stem}: pressure clip kinematics differ from the unrepaired source")
        v = pm["ground_reaction_valid"]
        body_valid = v[:, 1] >= 0.9
        cop_valid = v[:, 0] >= 0.9
        if extra_col is not None and v.shape[1] > extra_col:
            body_valid |= v[:, extra_col] >= 0.9
            cop_valid |= v[:, extra_col] >= 0.9
        return pm["rigid_body_ground_forces"][..., 2], body_valid, pm["ground_reaction"][:, 1:3], cop_valid
    return None


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extended-manifest", required=True)
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--swing-speed", type=float, default=0.25, help="m/s, slowest patch point")
    ap.add_argument("--speed-filter", type=int, default=5, help="median filter, frames")
    ap.add_argument("--lift-m", type=float, default=0.01,
                    help="velocity rule: the limb must rise this far above its own placement height")
    ap.add_argument("--lift-window-s", type=float, default=0.5)
    ap.add_argument("--moving-speed", type=float, default=0.05,
                    help="m/s: the pressure rule needs the limb to be moving at least this fast")
    ap.add_argument("--unloaded-n", type=float, default=14.0, help="~2 %% of the 700 N subject")
    ap.add_argument("--loaded-n", type=float, default=70.0, help="~10 %% of the subject: veto")
    ap.add_argument("--near-floor-m", type=float, default=0.10)
    args = ap.parse_args()

    body_names = mjcf_body_names(str(MJCF))
    consts = body_constants(body_names)
    zone_bodies = {z: [body_names.index(b) for b in ZONES[z]] for z in ZONE_ORDER}
    graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    names = list(graph["motion_names"])
    pair_names = list(graph["pair_names"])
    if list(graph.get("zone_order") or ZONE_ORDER) != list(ZONE_ORDER):
        raise ValueError("graph zone order differs from the extractor's")
    ext = {c["stem"]: c for c in yaml.safe_load(open(args.extended_manifest))["clips"]}
    src = {c["stem"]: c for c in yaml.safe_load(open(args.source_manifest))["clips"]}

    M, Z = len(names), len(ZONE_ORDER)
    S_max = graph["seg_node"].shape[1]
    lengths = []
    swings = []
    seg_cop_rel = torch.zeros(M, S_max, 2)
    seg_cop_valid = torch.zeros(M, S_max, dtype=torch.bool)
    seg_com_rel = torch.zeros(M, S_max, 2)
    seg_zone_share = torch.zeros(M, S_max, Z)
    seg_share_valid = torch.zeros(M, S_max, dtype=torch.bool)
    seg_lean_gate = torch.zeros(M, S_max, dtype=torch.bool)
    seg_pair_conseq = torch.zeros(M, S_max, len(pair_names), dtype=torch.bool)
    conseq_slots = torch.tensor([consequential(p) for p in pair_names])
    stats = dict(swing_share={}, velocity_only=0, pressure_only=0, vetoed=0, frames=0, pressure_clips=0)

    mass = consts["body_mass"]
    com_local = consts["body_com_local"]
    for m, stem in enumerate(names):
        e = ext[stem]
        s = src[e["source_stem"]]
        motion = torch.load(str(Path(args.motion_dir) / f"{stem}.motion"), map_location="cpu", weights_only=False)
        fps = int(motion["fps"])
        pos, rot = motion["rigid_body_pos"].float(), motion["rigid_body_rot"].float()
        T = pos.shape[0]
        lengths.append(T)
        # variant frame -> source frame
        original_path = s.get("pose_repair", {}).get("from", s["source"])
        original = torch.load(original_path, map_location="cpu", weights_only=False)
        insert = int(round(float(e["variant_s"]) * fps))
        index = splice_index(int(original["rigid_body_pos"].shape[0]), insertion_plan(s["holds"], insert))
        assert len(index) == T, f"{stem}: splice index {len(index)} != {T} frames"
        pressure = load_pressure(e["source_stem"], original["rigid_body_pos"])
        if pressure is not None:
            stats["pressure_clips"] += 1
        swing = torch.zeros(T, Z, dtype=torch.bool)
        for z, zone in enumerate(ZONE_ORDER):
            speed, zmin = zone_patch_speed(pos, rot, consts, zone_bodies[zone], fps)
            smooth = torch.from_numpy(median_filter(speed.numpy(), args.speed_filter))
            # a step lifts the limb off its own placement height, a slide does not: the lowest
            # point must sit above its minimum over +-lift_window_s (float bias is a constant
            # offset there and cancels)
            w = int(round(args.lift_window_s * fps))
            zn = zmin.numpy()
            zpad = np.pad(zn, (w, w), mode="edge")
            local_min = np.lib.stride_tricks.sliding_window_view(zpad, 2 * w + 1).min(-1)
            lifted = torch.from_numpy(zn - local_min > args.lift_m)
            vel = (smooth > args.swing_speed) & lifted
            moving = smooth > args.moving_speed
            unl = torch.zeros(T, dtype=torch.bool)
            veto = torch.zeros(T, dtype=torch.bool)
            if pressure is not None:
                load_b, body_valid, _, _ = pressure
                load = load_b[:, zone_bodies[zone]].sum(-1)[index]
                valid = body_valid[index]
                veto = valid & (load > args.loaded_n)
                if zone in PRESSURE_ZONES:
                    # a *moving* limb the mat says is unloaded: the start of a step or a hand
                    # re-placement. A placed, still limb that reads ~0 N is left alone (pincha's
                    # hands, which the forearms unload, are touching, not swinging).
                    unl = valid & (load < args.unloaded_n) & (zmin < args.near_floor_m) & moving
            swing[:, z] = (vel | unl) & ~veto
            stats["velocity_only"] += int((vel & ~unl & ~veto).sum())
            stats["pressure_only"] += int((unl & ~vel & ~veto).sum())
            stats["vetoed"] += int(((vel | unl) & veto).sum())
        stats["frames"] += T
        swings.append(swing)

        # ---- per-segment load signatures --------------------------------- #
        holds = {round(float(h["t_hold"]), 3): h for h in e["holds"]}
        for k in range(int(graph["seg_count"][m])):
            t_hold = round(float(graph["seg_hold"][m, k]), 3)
            h = holds.get(t_hold)
            if h is None:
                raise ValueError(f"{stem}: graph segment {k} (t_hold {t_hold}) not in the manifest")
            ground = [p.split(":")[0] for p in h["pairs"] if p.endswith(":G")]
            gset = set(ground)
            seg_lean_gate[m, k] = bool(gset & {"L_HAND", "R_HAND"}) and gset <= LEAN_SUPPORT
            seg_pair_conseq[m, k] = (graph["seg_contact"][m, k] > 0.5) & conseq_slots
            fh = min(int(round(float(h["t_hold"]) * fps)), T - 1)
            bodies = [b for z in ground for b in zone_bodies[z]]
            if not bodies:
                continue
            pts, _ = patch_points(pos[fh:fh + 1], rot[fh:fh + 1], consts, bodies)
            centroid = pts[0, :, :2].mean(0)
            com = ((pos[fh] + quat_rotate(rot[fh], com_local, w_last=True)) * mass[:, None]).sum(0) / mass.sum()
            seg_com_rel[m, k] = com[:2] - centroid
            if pressure is None:
                continue
            load_b, body_valid, cop, cop_valid = pressure
            f0 = int(round(float(h["t_start"]) * fps))
            f1 = min(int(round(float(h["t_end"]) * fps)), T - 1)
            src_frames = index[f0:f1 + 1].unique()
            ok = src_frames[cop_valid[src_frames]]
            if len(ok) >= 10:
                seg_cop_rel[m, k] = cop[ok].median(dim=0).values - centroid
                seg_cop_valid[m, k] = True
            okb = src_frames[body_valid[src_frames]]
            if len(okb) >= 10:
                zl = torch.stack([load_b[okb][:, zone_bodies[z]].sum(-1) for z in ZONE_ORDER], -1)
                share = zl / zl.sum(-1, keepdim=True).clamp(min=1e-6)
                seg_zone_share[m, k] = share.median(dim=0).values
                seg_share_valid[m, k] = True
        stats["swing_share"][stem] = {z: round(float(swing[:, i].float().mean()), 3)
                                      for i, z in enumerate(ZONE_ORDER) if swing[:, i].any()}

    T_max = max(lengths)
    swing_all = torch.zeros(M, T_max, Z, dtype=torch.bool)
    for m, sw in enumerate(swings):
        swing_all[m, : sw.shape[0]] = sw
    payload = dict(
        version=1, motion_names=names, fps=60, zone_order=list(ZONE_ORDER),
        zone_bodies={z: list(ZONES[z]) for z in ZONE_ORDER},
        swing=swing_all, swing_len=torch.tensor(lengths),
        seg_cop_rel=seg_cop_rel, seg_cop_valid=seg_cop_valid, seg_com_rel=seg_com_rel,
        seg_zone_share=seg_zone_share, seg_share_valid=seg_share_valid,
        seg_lean_gate=seg_lean_gate, seg_pair_consequential=seg_pair_conseq,
        pair_names=pair_names, params=vars(args), **consts,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report = Path(args.out).with_suffix(".json")
    report.write_text(json.dumps(stats, indent=1))
    print(f"wrote {args.out}: {M} motions, T_max {T_max}, {Z} zones; pressure on {stats['pressure_clips']} "
          f"motions; label frames velocity-only {stats['velocity_only']}, pressure-only "
          f"{stats['pressure_only']}, vetoed {stats['vetoed']}; lean-gated segments "
          f"{int(seg_lean_gate.sum())}, COP-valid segments {int(seg_cop_valid.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
