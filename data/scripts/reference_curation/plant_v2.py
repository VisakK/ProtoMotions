# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BodyFix Step 1 acceptance: plant v2 against the performer's own motion, without PhysX.

``data/scripts/build_subject_plant_v2.py`` builds the plant and checks what needs no motion (mass, dof names
and limits, rest FK, rest overlaps). This module drives it, and the shipped plant for the baselines, with the
female MoSh fit (``mosh_replay.mosh_kinematics``) over the 59 clips of the manifest that have a readable fit,
with the definitions of the subject-body check's replay study (``output/reference_curation/subject_body_check/
replay/replay_metrics.py``, whose numbers the card quotes), and writes the numbers into
``data/reference_curation/plant_v2/plant_v2.json`` under ``acceptance`` (with the sha256 of the XML pair
they were measured on) and per clip / per hold under ``output/reference_curation/plant_v2/``.

Configurations (plant, hand mode; ``g`` = shifted per frame so the lowest collider sits at +5 mm, the shipped
grounding rule):

  C1 / C1g   shipped plant, hand = "wrist" (the replay study's C1: the card's overlap baselines)
  S1 / S1g   shipped plant, hand = "fingers" (the same motion as V2 on the old body)
  V2 / V2g   plant v2, hand = "fingers"
  C3 / C3g   the prototype's hybrid (her skeleton, shipped colliders), hand = "wrist", only if its XML is on
             disk: C3g is the card's 261-float reference, so it checks this module's float definition

Measured:

* **labelled supports floating**: labels v1.1's configured observed ground supports (919 on the 59 clips),
  the window median of the zone's lowest collider point; > 2 cm floats. Crowns are the HEAD supports whose
  window is more than half head-blocked in the Step 8 lineage (27): the head within ``CROWN_M`` of the floor,
  ungrounded.
* **self-overlap**: pair-frames of the 253 colliding pairs overlapping by more than 1 cm, classified by the
  human's zone-pair state (capture store v2 ``human_pair_state``); the pair-frames the human keeps apart that
  V2 adds over C1 and over S1 are counted as "new".
* **statics** (``reference_curation.statics`` + ``witness``, unchanged, on plant v2 via ``mosh_replay.use_plant``):
  the 300 holds of labels v1.1 with a fit at ``frame_hold``, the labels' configured ground zones and pairs:
  ``v2_reground`` (lowest collider +5 mm), ``v2_reground0`` (0 mm), ``v2_raw``; ``shipped_reground`` (the 41
  baseline). LP-holdable = held + feasible.
* **wrist and hand torque**: for every hold whose gated LP is optimal, the joint that binds and the smallest
  factor on the wrist and hand limits together that brings s* to 1 with every other joint at its own limit
  (``torque_sweep``); build_subject_plant_v2's ``TORQUE_DECISIONS`` rests on it.
* **COM vs mat COP**: at every hold exemplar where the MOYO mat sees the whole body (``mat_valid_cov`` >= 0.9
  and ``mat_total`` >= 0.6 BW), the horizontal distance between the plant's COM (``static_hold_lp.set_pose``)
  and the mat COP, ungrounded.
* touch-frame floats and penetration per zone, slide (``build_physics_tables.zone_patch_speed``'s rule),
  acceleration spikes, and the joint box on the hold exemplars (v2's box against the shipped one).

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.plant_v2 [--workers 8]
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import json
import math
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

from reference_curation import ids
from reference_curation import mosh_replay as mr

MODULE = "reference_curation.plant_v2"
SCHEMA_VERSION = 1
RECORD = ids.DATA_ROOT / "plant_v2" / "plant_v2.json"
OUT_DIR = ids.OUTPUT_ROOT / "plant_v2"
STORE = ids.OUTPUT_ROOT / "capture"
LABELS_V1 = ids.DATA_ROOT / "labels/holds_repaired_ftC_posefix.labels_v1.249a920cea"
LABELS_V11 = ids.DATA_ROOT / "labels/holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828"
LINEAGE_DIR = ids.OUTPUT_ROOT / "retarget/holds_repaired_ftC_posefix.labels_v1.249a920cea.retarget_v1.22c19e569d"
HYBRID_XML = ids.OUTPUT_ROOT / "subject_body_check/replay/smpl_yogi03596_hybrid_protoskel_shippedgeoms.xml"

FLOAT_M = 0.02
FLOAT5_M = 0.05
PEN_M = -0.005
DEEP_M = 0.01
GROUND_CLEARANCE_M = 0.005
CROWN_M = 0.02
SLIDE_FAST = 0.05
SLIDE_ZONES = ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND")
MAT_COV_MIN, MAT_BW_MIN = 0.9, 0.6
BW_N = 74.0 * 9.81

# The card's baselines (BodyFix Step 1 acceptance).
BASELINE = {"overlap_pair_frames_C1": 80058, "overlap_human_separated_C1": 7746, "floats_grounded_C3g": 261,
            "floats_grounded_target": 200, "crowns": 27, "lp_holdable_shipped_same_motion": 41,
            "lp_holdable_prototype_grounded": 141, "lp_holdable_prototype_0cm": 190, "com_cop_p50_cm_max": 1.3}
WRIST_HAND = ("Wrist", "Hand")
SWEEP_MAX_K = 20.0


def plants() -> dict:
    out = {"shipped": (mr.SHIPPED_XML, mr.SHIPPED_FLAT), "v2": (mr.V2_XML, mr.V2_FLAT)}
    if HYBRID_XML.exists():
        out["hybrid"] = (HYBRID_XML, None)
    return out


def configs() -> dict:
    c = {"C1": ("shipped", "wrist"), "S1": ("shipped", "fingers"), "V2": ("v2", "fingers")}
    if HYBRID_XML.exists():
        c["C3"] = ("hybrid", "wrist")
    return c


def fit_stems() -> list[str]:
    from reference_curation import human_mesh as hm

    return [s for s in ids.manifest_stems() if hm.load_fit(s)[0] is not None]


# --------------------------------------------------------------------------- #
# Kernels (replay_metrics' definitions)
# --------------------------------------------------------------------------- #
def _patch_local(sk, body: int):
    from scipy.spatial.transform import Rotation

    g = sk.geoms[sk.names[body]][0]
    if g["type"] == "box":
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        rg = Rotation.from_quat(g["quat"]).as_matrix()
        return np.asarray(g["center"], float) + (rg @ (signs * np.asarray(g["half"], float)).T).T, 0.0, "box"
    if g["type"] == "capsule":
        return np.asarray(g["seg"], float), float(g["radius"]), "capsule"
    return np.asarray(g["center"], float)[None], float(g["radius"]), "sphere"


def zone_slide(sk, pos, rot, zone_bodies, fps):
    """``[T]`` slowest horizontal material speed (m/s) over the zone's patch points: a box's 4 lowest corners,
    capsule ends and sphere bottoms, each carried in its body frame from t to t+1
    (``build_physics_tables.zone_patch_speed``'s rule)."""
    T = pos.shape[0]
    speeds = []
    for b in zone_bodies:
        loc, r, kind = _patch_local(sk, b)
        w = pos[:, b, None] + np.einsum("tij,kj->tki", rot[:, b], loc)
        if kind == "box":
            low = np.argsort(w[..., 2], axis=1)[:, :4]
            w = np.take_along_axis(w, low[..., None], axis=1)
        else:
            w = w.copy()
            w[..., 2] -= r
        rel = w[:-1] - pos[:-1, b, None]
        lc = np.einsum("tji,tkj->tki", rot[:-1, b], rel)
        moved = np.einsum("tij,tkj->tki", rot[1:, b], lc) + pos[1:, b, None]
        speeds.append(np.linalg.norm((moved - w[:-1])[..., :2], axis=-1) * fps)
    s = np.concatenate(speeds, 1).min(1)
    return np.r_[s, s[-1:]] if T > 1 else np.zeros(T)


def _labels():
    from reference_curation import statics

    return statics.load_labels(LABELS_V1), statics.load_labels(LABELS_V11)


# --------------------------------------------------------------------------- #
# One clip: the replay
# --------------------------------------------------------------------------- #
def replay_clip(stem: str) -> dict:
    import torch

    from extract_contact_configs import ZONE_ORDER, ZONES
    from reference_curation import human_mesh as hm, retarget as rt

    t0 = time.time()
    ZI = {z: i for i, z in enumerate(ZONE_ORDER)}
    lab1, lab11 = _labels()
    anns11 = {h: rows for h, rows in lab11["anns"].items() if h.startswith(stem + "@")}
    v2s = np.load(STORE / "v2" / f"{stem}.npz") if (STORE / "v2" / f"{stem}.npz").exists() else None
    v3 = np.load(STORE / "v3" / f"{stem}.npz")
    state, hz = v3["human_floor_state"], v3["human_floor_z"].astype(np.float64)
    T = state.shape[0]
    lin = LINEAGE_DIR / f"{stem}.lineage.npz"
    blocked = np.load(lin)["head_blocked"] if lin.exists() else np.zeros(T, bool)

    P = plants()
    sks = {k: mr.skeleton_for(x) for k, (x, _) in P.items()}
    names = sks["shipped"].names
    for k, sk in sks.items():
        assert sk.names == names and sk.parents == sks["shipped"].parents, k
    kin_f = mr.mosh_kinematics(stem, hand="fingers")
    assert kin_f["dof"].shape[0] == T, (stem, kin_f["dof"].shape, T)
    kins = {"fingers": kin_f, "wrist": mr.with_wrist_hands(kin_f)}
    fps = kin_f["fps"]
    out = {"stem": stem, "T": int(T), "configs": {}, "head_blocked_frames": int(blocked.sum())}
    pcol = {frozenset(n.split("+")): k for k, n in enumerate(hm.PAIR_NAMES)}
    zmin, arrays = {}, {}
    for cfg, (plant, hand) in configs().items():
        sk = sks[plant]
        pos, rot = mr.fk_bodies(sk, kins[hand])
        if plant == "v2":
            err = float(np.linalg.norm(pos - mr.body_joint_targets(kin_f), axis=-1).max())
            out["v2_fk_vs_her_joints_max_m"] = err
        z = mr.zone_lowest(sk, pos, rot)
        zmin[cfg] = z
        zmin[cfg + "g"] = z - (z.min(1, keepdims=True) - GROUND_CLEARANCE_M)
        f, a, b, g = rt.near_body_pairs(sk, torch.as_tensor(pos), torch.as_tensor(rot), 0.0)
        deep = g < -DEEP_M
        cls, keys = collections.Counter(), {}
        for fi, i, j in zip(f[deep], a[deep], b[deep]):
            za, zb = rt.BODY_ZONE[names[i]], rt.BODY_ZONE[names[j]]
            if za == zb:
                c = "same_zone"
            elif frozenset((za, zb)) not in pcol or v2s is None:
                c = "adjacent_zones_no_reading"
            else:
                c = {1: "human_contact", 0: "human_separated", -1: "human_undecided"}[
                    int(v2s["human_pair_state"][fi, pcol[frozenset((za, zb))]])]
            cls[c] += 1
            keys[(int(fi), int(i), int(j))] = c
        slide = np.stack([zone_slide(sk, pos, rot, [names.index(x) for x in ZONES[zn]], fps) for zn in SLIDE_ZONES], 1)
        acc = np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1).max(1) * fps ** 2 if T > 2 else np.zeros(0)
        arrays[cfg] = {"keys": keys, "slide": slide}
        out["configs"][cfg] = {"overlap_deep": int(deep.sum()), "overlap_class": dict(cls),
                               "overlap_pairs": dict(collections.Counter(f"{names[i]}+{names[j]}" for i, j in zip(a[deep], b[deep]))),
                               "acc_spike_frames": int((acc > rt.SPIKE_ACC).sum())}
    # overlaps V2 adds on pairs the human keeps apart
    for base in ("C1", "S1"):
        new = [k for k, c in arrays["V2"]["keys"].items() if c == "human_separated" and k not in arrays[base]["keys"]]
        out["configs"]["V2"][f"new_human_separated_vs_{base}"] = len(new)
        out["configs"]["V2"][f"new_human_separated_vs_{base}_pairs"] = dict(collections.Counter(
            f"{names[i]}+{names[j]}" for _, i, j in new))

    touch_all = state == 1
    touch = touch_all.copy()
    touch[:, ZI["HEAD"]] &= ~blocked
    sep = state == 0
    for cfg, z in zmin.items():
        c = out["configs"].setdefault(cfg, {})
        c.update({"touch_zone": touch.sum(0).tolist(), "float2_zone": ((z > FLOAT_M) & touch).sum(0).tolist(),
                  "float5_zone": ((z > FLOAT5_M) & touch).sum(0).tolist(), "sep_zone": sep.sum(0).tolist(),
                  "phantom_zone": ((z <= FLOAT_M) & sep).sum(0).tolist(),
                  "pen_touch_zone": ((z < PEN_M) & touch_all).sum(0).tolist(),
                  "touch_all_zone": touch_all.sum(0).tolist(), "min_zone_m": z.min(0).tolist()})
    samples = {cfg: {zn: z[touch_all[:, ZI[zn]], ZI[zn]].astype(np.float32) for zn in ZONE_ORDER} for cfg, z in zmin.items()}
    samples["human"] = {zn: hz[touch_all[:, ZI[zn]], ZI[zn]].astype(np.float32) for zn in ZONE_ORDER}
    slide = {cfg: {zn: arrays[cfg]["slide"][touch_all[:, ZI[zn]], k].astype(np.float32) for k, zn in enumerate(SLIDE_ZONES)}
             for cfg in arrays}

    supports = []
    for hid, rows in anns11.items():
        for a in rows:
            if a["kind"] != "ground" or not a["in_configuration"] or a["source_state"] != "observed_contact":
                continue
            iv = a["interval"]
            f0, f1 = iv["start_frame"], iv["end_frame_exclusive"] - 1
            zn = a["zones"][0]
            row = {"hold_id": hid, "zone": zn, "f0": f0, "f1": f1,
                   "head_blocked": bool(zn == "HEAD" and blocked[f0:f1 + 1].mean() > 0.5),
                   "human_cm": round(100 * float(np.median(hz[f0:f1 + 1, ZI[zn]])), 2)}
            for cfg, z in zmin.items():
                row[cfg] = round(100 * float(np.median(z[f0:f1 + 1, ZI[zn]])), 2)
            supports.append(row)
    out["supports"] = supports
    out["seconds"] = round(time.time() - t0, 1)
    return {"result": out, "samples": samples, "slide": slide}


def _run_replay(stem):
    import torch

    torch.set_num_threads(1)
    try:
        return stem, replay_clip(stem), None
    except Exception as exc:  # noqa: BLE001
        import traceback

        return stem, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def aggregate_replay(parts: list[dict]) -> dict:
    from extract_contact_configs import ZONE_ORDER

    results = [p["result"] for p in parts]
    cfgs = list(results[0]["configs"])
    raw = [c for c in cfgs if c in configs()]          # a rigid grounding shift changes no overlap or slide
    agg = {"clips": len(results), "frames": int(sum(r["T"] for r in results)),
           "v2_fk_vs_her_joints_max_m": max(r.get("v2_fk_vs_her_joints_max_m", 0.0) for r in results)}
    S = [s for r in results for s in r["supports"]]
    crowns = [s for s in S if s["head_blocked"]]
    reach = [s for s in S if not s["head_blocked"]]
    sup = {"n": len(S), "crowns_n": len(crowns), "human_over2": sum(s["human_cm"] > 2.0 for s in S)}
    for cfg in cfgs:
        v = np.array([s[cfg] for s in S])
        sup[cfg] = {"over2": int((v > 2.0).sum()), "over5": int((v > 5.0).sum()), "below_minus2": int((v < -2.0).sum()),
                    "over2_reachable": sum(s[cfg] > 2.0 for s in reach), "over2_crowns": sum(s[cfg] > 2.0 for s in crowns),
                    "median_cm": round(float(np.median(v)), 2), "p90_cm": round(float(np.percentile(v, 90)), 2),
                    "abs_minus_human_le_2cm": int(sum(abs(s[cfg] - s["human_cm"]) <= 2.0 for s in S))}
        zc = collections.Counter()
        for s in S:
            zc[s["zone"]] += s[cfg] > 2.0
        sup[cfg]["over2_by_zone"] = dict(zc)
    agg["supports"] = sup
    agg["crowns"] = {"n": len(crowns), "rows": [{k: s[k] for k in ("hold_id", "human_cm", *[c for c in cfgs if c in s])}
                                                for s in crowns],
                     "within_crown_m": {cfg: int(sum(abs(s[cfg]) <= 100 * CROWN_M for s in crowns)) for cfg in cfgs},
                     "max_abs_cm": {cfg: round(float(max(abs(s[cfg]) for s in crowns)), 2) for cfg in cfgs},
                     "median_cm": {cfg: round(float(np.median([s[cfg] for s in crowns])), 2) for cfg in cfgs}}
    zf = {}
    for cfg in results[0]["configs"]:
        tot = {k: np.sum([r["configs"][cfg][k] for r in results], 0) for k in
               ("touch_zone", "float2_zone", "float5_zone", "sep_zone", "phantom_zone", "pen_touch_zone", "touch_all_zone")}
        zf[cfg] = {"float2_share": round(100 * float(tot["float2_zone"].sum() / tot["touch_zone"].sum()), 2),
                   "float5_share": round(100 * float(tot["float5_zone"].sum() / tot["touch_zone"].sum()), 2),
                   "phantom_share": round(100 * float(tot["phantom_zone"].sum() / tot["sep_zone"].sum()), 2),
                   "pen_touch_frames": int(tot["pen_touch_zone"].sum()), "touch_frames": int(tot["touch_all_zone"].sum()),
                   "per_zone": {zn: {"touch": int(tot["touch_all_zone"][i]),
                                     "float2_share": round(100 * float(tot["float2_zone"][i] / max(tot["touch_zone"][i], 1)), 2),
                                     "pen_touch_share": round(100 * float(tot["pen_touch_zone"][i] / max(tot["touch_all_zone"][i], 1)), 2),
                                     "touch_p50_cm": (round(100 * float(np.median(np.concatenate([p["samples"][cfg][zn] for p in parts]))), 2)
                                                      if tot["touch_all_zone"][i] else None),
                                     "min_cm": round(100 * float(min(r["configs"][cfg]["min_zone_m"][i] for r in results)), 2)}
                                for i, zn in enumerate(ZONE_ORDER)}}
    hum = np.concatenate([np.concatenate([p["samples"]["human"][zn] for p in parts]) for zn in ZONE_ORDER])
    agg["human_skin_touch"] = {"n": int(len(hum)), "below_minus_0p5cm": int((hum < PEN_M).sum()),
                               "per_zone_p50_cm": {zn: round(100 * float(np.median(np.concatenate([p["samples"]["human"][zn] for p in parts]))), 2)
                                                   if sum(len(p["samples"]["human"][zn]) for p in parts) else None
                                                   for zn in ZONE_ORDER}}
    agg["zone_frames"] = zf
    ov = {}
    for cfg in raw:
        oc, pp = collections.Counter(), collections.Counter()
        for r in results:
            oc.update(r["configs"][cfg]["overlap_class"])
            pp.update(r["configs"][cfg]["overlap_pairs"])
        ov[cfg] = {"deep_pair_frames": int(sum(r["configs"][cfg]["overlap_deep"] for r in results)),
                   "by_human_state": dict(oc), "top_pairs": dict(pp.most_common(12)),
                   "acc_spike_frames": int(sum(r["configs"][cfg]["acc_spike_frames"] for r in results))}
    for base in ("C1", "S1"):
        ov["V2"][f"new_human_separated_vs_{base}"] = int(sum(r["configs"]["V2"][f"new_human_separated_vs_{base}"] for r in results))
        c = collections.Counter()
        for r in results:
            c.update(r["configs"]["V2"][f"new_human_separated_vs_{base}_pairs"])
        ov["V2"][f"new_human_separated_vs_{base}_top_pairs"] = dict(c.most_common(10))
    agg["overlaps"] = ov
    sl = {}
    for cfg in raw:
        p90, fast, n = [], 0, 0
        for p in parts:
            v = np.concatenate([p["slide"][cfg][zn] for zn in SLIDE_ZONES])
            if len(v):
                p90.append(np.percentile(v, 90))
                fast += int((v > SLIDE_FAST).sum())
                n += len(v)
        sl[cfg] = {"clip_median_of_p90_cm_s": round(100 * float(np.median(p90)), 2), "share_gt_5cm_s": round(fast / n, 4)}
    agg["slide"] = sl
    return agg


# --------------------------------------------------------------------------- #
# Statics arms (the repo's statics + witness, unchanged, with the plant swapped in process)
# --------------------------------------------------------------------------- #
ARMS = {"shipped_reground": ("shipped", GROUND_CLEARANCE_M), "v2_raw": ("v2", None),
        "v2_reground": ("v2", GROUND_CLEARANCE_M), "v2_reground0": ("v2", 0.0)}


def _plant_ctx(plant: str):
    return mr.use_plant(mr.V2_XML, mr.V2_FLAT) if plant == "v2" else contextlib.nullcontext()


def arm_poses(plant: str, clearance, stem: str, frames: np.ndarray):
    xml = mr.V2_XML if plant == "v2" else mr.SHIPPED_XML
    sk = mr.skeleton_for(xml)
    kin = mr.mosh_kinematics(stem, hand="fingers", frames=frames, xml_path=xml)
    pos, rot = mr.fk_bodies(sk, kin)
    low = mr.body_lowest(sk, pos, rot)
    dz = np.zeros(len(frames)) if clearance is None else clearance - low.min(1)
    pos = pos.copy()
    pos[:, :, 2] += dz[:, None]
    return pos, rot, dz


def _audit(stem, hold, anns, pos, quat):
    from reference_curation import statics, witness

    req = statics.hold_request(anns)
    g = statics.analyse(pos, quat, req["ground"], req["pairs"], stops=True)
    wit = None
    if g["status"] == "optimal" and not g["beyond_plant"]:
        wit = witness.run(pos, quat, g["tau"], [n for n in req["pairs"] if g["contacts"][n]["realised"]])
    st = g["status"]
    verdict = ("held" if wit and wit["passed"] else "feasible") if st == "optimal" and not g["beyond_plant"] \
        else ("beyond_plant" if st == "optimal" else st)
    return req, g, wit, verdict


def torque_sweep(pos, quat, req) -> dict | None:
    """How far the wrist and hand torque limits bind this pose: the gated LP's s* at the plant's limits, the
    joint that binds, and the smallest factor ``k`` on the wrist and hand limits together (their ratio kept)
    that brings s* to 1 with every other joint at its own limit. ``None`` when a support is not realised or
    the LP is not optimal; ``k`` = 1 when s* <= 1 already, ``inf`` when even ``SWEEP_MAX_K`` does not (another
    joint binds)."""
    import static_hold_lp as S

    from reference_curation import statics

    prob, contacts, _ = statics.build_problem(pos, quat, req["ground"], req["pairs"])
    if any(c["kind"] == "ground" and not c["realised"] for c in contacts.values()):
        return None
    wh = np.array([any(k in n for k in WRIST_HAND) for n in S.JNAMES])
    base = prob.tau_max.copy()

    def solve(k):
        prob.tau_max = np.where(wh, base * k, base)
        return statics.solve_lp(prob)

    st, sol = solve(1.0)
    if st != "optimal":
        return None
    s0 = float(sol["s"])
    j = int(np.argmax(sol["util"]))
    rec = {"s_star": round(s0, 4), "binding_joint": S.JNAMES[j],
           "wrist_hand_limits_Nm": sorted({float(x) for x in base[wh]})}
    if s0 <= 1.0:
        rec["k_needed"] = 1.0
        return rec
    st, sol = solve(SWEEP_MAX_K)
    if st != "optimal" or sol["s"] > 1.0:
        rec["k_needed"] = math.inf
        rec["binding_joint_at_max_k"] = S.JNAMES[int(np.argmax(sol["util"]))] if st == "optimal" else None
        return rec
    lo, hi = 1.0, SWEEP_MAX_K
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        st, sol = solve(mid)
        lo, hi = (lo, mid) if st == "optimal" and sol["s"] <= 1.0 else (mid, hi)
    rec["k_needed"] = round(hi, 3)
    return rec


def statics_job(job):
    import torch
    from scipy.spatial.transform import Rotation

    from reference_curation import statics  # noqa: F401  (imported before the plant swap)

    torch.set_num_threads(1)
    arm, stem, holds, anns = job
    plant, clearance = ARMS[arm]
    out, fails = [], []
    try:
        with _plant_ctx(plant):
            import static_hold_lp as S

            assert abs(S.M.body_mass.sum() - 74.0) < 1e-3
            frames = np.array([int(h["frame_hold"]) for h in holds])
            pos, rot, dz = arm_poses(plant, clearance, stem, frames)
            for k, h in enumerate(holds):
                try:
                    quat = Rotation.from_matrix(rot[k]).as_quat()
                    req, g, wit, verdict = _audit(stem, h, anns[h["hold_id"]], pos[k], quat)
                    rec = {"arm": arm, "hold_id": h["hold_id"], "verdict": verdict, "s_star": g["s_star"],
                           "dz_cm": round(100 * float(dz[k]), 3), "top_joints": g.get("top_joints"),
                           "ground": {n: [c["height_cm"], c["realised"]] for n, c in g["contacts"].items() if c["kind"] == "ground"},
                           "witness_passed": None if wit is None else wit["passed"]}
                    if arm in ("v2_reground", "v2_raw", "v2_reground0", "shipped_reground"):
                        rec["torque_sweep"] = torque_sweep(pos[k], quat, req)
                    out.append(rec)
                except Exception as exc:  # noqa: BLE001
                    fails.append(f"{arm} {h['hold_id']}: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        fails.append(f"{arm} {stem}: {type(exc).__name__}: {exc}")
    return out, fails


def run_statics(workers: int, stems: list[str]) -> tuple[dict, list[dict], list[str]]:
    from reference_curation import human_mesh as hm, statics

    labels = statics.load_labels(LABELS_V11)
    have = set(stems)
    jobs = [(arm, stem, holds, {h["hold_id"]: labels["anns"][h["hold_id"]] for h in holds})
            for arm in ARMS for stem, holds in labels["clips"] if stem in have]
    if workers > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                min(workers, 8), mp_context=multiprocessing.get_context("spawn")) as ex:
            parts = list(ex.map(statics_job, jobs))
    else:
        parts = [statics_job(j) for j in jobs]
    recs = [r for rs, _ in parts for r in rs]
    fails = [f for _, fs in parts for f in fs]
    summ = {}
    for arm in ARMS:
        rs = [r for r in recs if r["arm"] == arm]
        c = collections.Counter(r["verdict"] for r in rs)
        summ[arm] = {"holds": len(rs), "verdict": dict(c), "lp_holdable": c["held"] + c["feasible"], "held": c["held"]}
        sw = [r["torque_sweep"] for r in rs if r.get("torque_sweep")]
        if sw:
            k = np.array([x["k_needed"] for x in sw])
            fin = k[np.isfinite(k) & (k > 1.0)]
            summ[arm]["torque_sweep"] = {
                "optimal_holds": len(sw), "within_limits": int((k == 1.0).sum()),
                "wrist_hand_limited": int(len(fin)), "other_joint_limited": int((~np.isfinite(k)).sum()),
                "k_needed_sorted": sorted(round(float(x), 3) for x in fin),
                "binding_joints_beyond_plant": dict(collections.Counter(x["binding_joint"] for x in sw if x["s_star"] > 1))}
    return summ, recs, fails


# --------------------------------------------------------------------------- #
# COM vs mat COP and the joint box at the exemplars
# --------------------------------------------------------------------------- #
def com_cop(stems: list[str]) -> tuple[dict, list[dict]]:
    import static_hold_lp as S
    from scipy.spatial.transform import Rotation

    from reference_curation import statics

    labels = statics.load_labels(LABELS_V11)
    rows = []
    have = set(stems)
    for stem, holds in labels["clips"]:
        if stem not in have:
            continue
        v1 = np.load(STORE / "v1" / f"{stem}.npz")
        frames = np.array([int(h["frame_hold"]) for h in holds])
        cop, tot, cov = v1["mat_cop"][frames], v1["mat_total"][frames], v1["mat_valid_cov"][frames]
        coms = {}
        for plant in ("shipped", "v2"):
            with _plant_ctx(plant):
                pos, rot, _ = arm_poses(plant, None, stem, frames)
                c = []
                for k in range(len(frames)):
                    S.set_pose(pos[k], Rotation.from_matrix(rot[k]).as_quat())
                    c.append(S.D.subtree_com[1][:2].copy())
                coms[plant] = np.array(c)
        for k, h in enumerate(holds):
            ok = bool(np.isfinite(cop[k]).all() and cov[k] >= MAT_COV_MIN and tot[k] >= MAT_BW_MIN * BW_N)
            rows.append({"hold_id": h["hold_id"], "mat_ok": ok,
                         **{f"err_cm_{p}": round(100 * float(np.linalg.norm(c[k] - cop[k])), 2) if ok else None
                            for p, c in coms.items()}})
    good = [r for r in rows if r["mat_ok"]]
    summ = {"holds": len(rows), "mat_ok": len(good), "gate": f"mat_valid_cov >= {MAT_COV_MIN} and mat_total >= {MAT_BW_MIN} BW"}
    for p in ("shipped", "v2"):
        e = np.array([r[f"err_cm_{p}"] for r in good])
        summ[p] = {"p25": round(float(np.percentile(e, 25)), 2), "p50": round(float(np.median(e)), 2),
                   "p75": round(float(np.percentile(e, 75)), 2), "p90": round(float(np.percentile(e, 90)), 2),
                   "share_le_2cm": round(float(np.mean(e <= 2)), 3)}
    return summ, rows


def joint_box_exemplars(stems: list[str]) -> dict:
    """Hold exemplars (labels v1.1) with a coordinate more than 2 deg past the box, female fit (fingers),
    nearest representative, on the shipped and the v2 box."""
    import torch

    from reference_curation import retarget as rt, statics

    labels = statics.load_labels(LABELS_V11)
    have = set(stems)
    sks = {"shipped": mr.skeleton_for(mr.SHIPPED_XML), "v2": mr.skeleton_for(mr.V2_XML)}
    out = {k: {"exemplars": 0, "past": 0, "by_coord": collections.Counter()} for k in sks}
    for stem, holds in labels["clips"]:
        if stem not in have:
            continue
        frames = np.array([int(h["frame_hold"]) for h in holds])
        dof = torch.as_tensor(mr.mosh_kinematics(stem, hand="fingers", frames=frames)["dof"])
        for k, sk in sks.items():
            rep = rt.nearest_representative(dof, sk.lower, sk.upper)
            exc = np.degrees(torch.maximum(sk.lower - rep, rep - sk.upper).clamp_min(0).numpy())
            coords = [f"{b}_{a}" for b in sk.names[1:] for a in "xyz"]
            for e in exc:
                out[k]["exemplars"] += 1
                out[k]["past"] += int((e > 2.0).any())
                out[k]["by_coord"].update(coords[j] for j in np.nonzero(e > 2.0)[0])
    return {k: {"exemplars": v["exemplars"], "past_2deg": v["past"], "by_coord": dict(v["by_coord"].most_common(10))}
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
def acceptance_failures(acc: dict) -> list[str]:
    """The card's acceptance tests on the measured numbers (the static ones are the builder's)."""
    bad = []
    ov = acc["replay"]["overlaps"]
    if ov["V2"]["deep_pair_frames"] > BASELINE["overlap_pair_frames_C1"]:
        bad.append(f"self-overlap pair-frames {ov['V2']['deep_pair_frames']} > C1 {BASELINE['overlap_pair_frames_C1']}")
    sep = ov["V2"]["by_human_state"].get("human_separated", 0)
    if sep > BASELINE["overlap_human_separated_C1"]:
        bad.append(f"overlaps on pairs the human keeps apart {sep} > C1 {BASELINE['overlap_human_separated_C1']}")
    cr = acc["replay"]["crowns"]
    if cr["n"] != BASELINE["crowns"] or cr["within_crown_m"]["V2"] != cr["n"]:
        bad.append(f"crowns within {CROWN_M * 100:.0f} cm: {cr['within_crown_m']['V2']}/{cr['n']}")
    fl = acc["replay"]["supports"]["V2g"]["over2"]
    if fl > BASELINE["floats_grounded_C3g"]:
        bad.append(f"grounded floats {fl} > {BASELINE['floats_grounded_C3g']}")
    if acc["com_cop"]["v2"]["p50"] > BASELINE["com_cop_p50_cm_max"]:
        bad.append(f"COM vs mat COP p50 {acc['com_cop']['v2']['p50']} cm > {BASELINE['com_cop_p50_cm_max']}")
    if acc["replay"]["v2_fk_vs_her_joints_max_m"] > 1e-5:
        bad.append(f"v2 FK misses her joints by {acc['replay']['v2_fk_vs_her_joints_max_m']:.1e} m")
    return bad


def _finite(o):
    """``o`` with every non-finite float replaced by ``None`` (strict JSON)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def measure(workers: int = 8) -> tuple[dict, dict]:
    """``(acceptance, rows)``: every acceptance number, and the per-clip / per-hold rows. Writes nothing."""
    from reference_curation import human_mesh as hm

    t0 = time.time()
    stems = fit_stems()
    if workers > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                min(workers, 8), mp_context=multiprocessing.get_context("spawn")) as ex:
            outs = list(ex.map(_run_replay, stems))
    else:
        outs = [_run_replay(s) for s in stems]
    fails = [f"{s}: {e}" for s, _, e in outs if e]
    parts = [p for _, p, e in outs if e is None]
    replay = aggregate_replay(parts)
    statics_summ, statics_recs, sfails = run_statics(workers, stems)
    cc, cc_rows = com_cop(stems)
    acc = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [mr.V2_XML, mr.V2_FLAT, mr.SHIPPED_XML, LABELS_V11 / "annotations.jsonl"]),
           "xml_sha256": ids.sha256_file(mr.V2_XML), "flat_sha256": ids.sha256_file(mr.V2_FLAT),
           "clips": len(stems), "baselines": BASELINE, "replay": replay, "statics": statics_summ, "com_cop": cc,
           "joint_box_exemplars": joint_box_exemplars(stems), "failures": fails + sfails}
    acc["acceptance_failures"] = acceptance_failures(acc)
    acc["seconds"] = round(time.time() - t0, 1)
    return acc, {"replay": [p["result"] for p in parts], "statics": statics_recs, "com_cop": cc_rows}


def write(acc: dict, rows: dict) -> Path:
    """The acceptance into plant_v2.json (refused if it was measured on another XML pair) and the rows under
    ``OUT_DIR``."""
    rec = json.loads(RECORD.read_text())
    if (rec["outputs"]["xml_sha256"], rec["outputs"]["flat_sha256"]) != (acc["xml_sha256"], acc["flat_sha256"]):
        raise RuntimeError("plant_v2.json describes another XML pair; rebuild the plant first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "replay_per_clip.jsonl", "w") as f:
        for r in rows["replay"]:
            f.write(json.dumps({k: v for k, v in r.items() if k != "supports"}, default=float) + "\n")
    with open(OUT_DIR / "replay_supports.jsonl", "w") as f:
        for r in rows["replay"]:
            for s in r["supports"]:
                f.write(json.dumps({"stem": r["stem"], **s}) + "\n")
    with open(OUT_DIR / "statics_holds.jsonl", "w") as f:
        for r in rows["statics"]:
            f.write(json.dumps(_finite(r), allow_nan=False) + "\n")
    (OUT_DIR / "com_cop_rows.json").write_text(json.dumps(rows["com_cop"], indent=0) + "\n")
    rec["acceptance"] = json.loads(json.dumps(_finite(acc), allow_nan=False, default=float))
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    return RECORD


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    acc, rows = measure(args.workers)
    write(acc, rows)
    r = acc["replay"]
    st = acc["statics"]
    bad = acc["acceptance_failures"] + acc["failures"]
    print(f"plant v2 acceptance: overlaps V2 {r['overlaps']['V2']['deep_pair_frames']} (C1 {r['overlaps']['C1']['deep_pair_frames']}), "
          f"human-separated {r['overlaps']['V2']['by_human_state'].get('human_separated', 0)}; crowns "
          f"{r['crowns']['within_crown_m']['V2']}/{r['crowns']['n']}; grounded floats {r['supports']['V2g']['over2']}/"
          f"{r['supports']['n']}; LP-holdable v2 {st['v2_reground']['lp_holdable']} (0 cm {st['v2_reground0']['lp_holdable']}, "
          f"shipped {st['shipped_reground']['lp_holdable']}); COM-COP p50 {acc['com_cop']['v2']['p50']} cm; {acc['seconds']} s"
          + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
