# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package per-pair reference contact targets for the pair-aware contact reward (T5).

S3 side-channel packaging (notes/Contact_balance_reward_design.MD section 2.3 / 3.5):
rasterize the cleaned contact-configuration segments (data/smpl/yoga_contact_configs/
<stem>.json) into per-frame per-pair encourage masks tau_p(t) for the crow-pair
corpus, compute the side-crow L_Hip<->R_Hip forbid mask from reference geometry,
and bundle the typed geom parameters the runtime distance kernels need.

Output: data/smpl/yoga_yogi_crow_pair_reftargets.pt -- a plain dict, schema is a
hard contract with the training-time reward kernels which gather rows by
(motion_id, motion_time):

    version        str
    motion_pt      str   (the motion package these targets are tied to)
    fps            60
    num_motions    2
    motion_names   list[str] (yaml order == motion_id order)
    num_frames     LongTensor [M]
    frame_starts   LongTensor [M]      (cumulative, frame_starts[0] == 0)
    pair_names     list[str] [P]
    pair_body_a    LongTensor [P]      (common-order body ids)
    pair_body_b    LongTensor [P]      (-1 for ground pairs)
    pair_kappa     FloatTensor [P]
    encourage_mask FloatTensor [sum(T), P]   (smoothed, in [0,1])
    forbid_names   list[str] [F]
    forbid_body_a  LongTensor [F]
    forbid_body_b  LongTensor [F]
    forbid_mask    FloatTensor [sum(T), F]   (smoothed, in [0,1])
    body_names     list[str] [24]      (COMMON order)
    geom_type      LongTensor [24]     (0 = box/none, 1 = sphere, 2 = capsule)
    geom_radius    FloatTensor [24]
    geom_p0        FloatTensor [24,3]  (body-local core segment endpoint; sphere p0==p1==center)
    geom_p1        FloatTensor [24,3]

Masks are rasterized from segments (skipping statically_inconsistent ones, painting
[start_frame, end_frame] inclusive) and then smoothed with a 7-frame moving average
with replicate padding, per motion -- the motion_lib.smooth_contacts convention.

Validation: per-pair rasterized dwell vs the geometric detector's per-frame
'active' series (compute_active_pairs) with frame-level Jaccard, dwell spot
checks against results/Contact_Physics_analysis_crow_pair_final, and the
side-crow reference L_Hip<->R_Hip gap percentiles for the forbid rule.
Report saved next to the output .pt.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
sys.path.insert(0, str(REPO))

from contact_geometry import geom_pair_distance, geom_to_world, parse_typed_geoms  # noqa: E402
from extract_contact_configs import compute_active_pairs, mjcf_body_names  # noqa: E402

MJCF = REPO / "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
CLIP_DIR = REPO / "data/smpl/yoga_yogi_balance_subset_v2_contacts"
JSON_DIR = REPO / "data/smpl/yoga_contact_configs"
YAML_PATH = REPO / "data/smpl/yoga_yogi_crow_pair_v2_contacts.yaml"
OUT_PATH = REPO / "data/smpl/yoga_yogi_crow_pair_reftargets.pt"
REPORT_PATH = REPO / "data/smpl/yoga_yogi_crow_pair_reftargets_report.md"
MOTION_PT = "data/smpl/yoga_yogi_crow_pair_v3_contacts.pt"

FPS = 60
SMOOTH_WINDOW = 7
# Forbid-mask rule: forbid applies where the REFERENCE gap exceeds this.  The
# training-time penalty ramp psi(d) = clamp(1 - d/0.02, 0, 1) fires only below
# 2 cm, and the side-crow reference L_Hip<->R_Hip gap never drops below 2.42 cm,
# so a policy exactly matching the reference sits at psi = 0 on every frame --
# the "reference is never penalized" invariant is protected by the ramp itself.
# 0.022 = 2 mm safety margin above the psi boundary (was 0.03, needlessly
# conservative: it masked off most of the hold).
FORBID_GAP_M = 0.022
PSI_ZERO_M = 0.02  # psi(d) support boundary, for the invariant assert below

# (pair_name, body_a, body_b (None = ground), kappa).  Global union over both
# clips; zone -> representative member from the measured closest members.
ENCOURAGE_PAIRS = [
    ("L_HAND:G", "L_Hand", None, 1.0),
    ("R_HAND:G", "R_Hand", None, 1.0),
    ("L_SHANK+L_UPPER_ARM", "L_Knee", "L_Shoulder", 1.0),
    ("R_SHANK+R_UPPER_ARM", "R_Knee", "R_Shoulder", 1.0),
    ("L_THIGH+L_UPPER_ARM", "L_Hip", "L_Shoulder", 1.0),
    ("L_THIGH+TRUNK", "L_Hip", "Chest", 1.0),
    ("R_THIGH+TRUNK", "R_Hip", "Chest", 1.0),
    ("R_THIGH+R_UPPER_ARM", "R_Hip", "R_Shoulder", 0.5),
    ("R_THIGH+L_UPPER_ARM", "R_Hip", "L_Shoulder", 1.0),
    ("R_THIGH+L_FOREARM", "R_Hip", "L_Elbow", 1.0),
    ("R_SHANK+L_FOREARM", "R_Knee", "L_Elbow", 1.0),
    ("L_SHANK+R_THIGH", "L_Knee", "R_Hip", 1.0),
    ("L_SHANK+R_SHANK", "L_Knee", "R_Knee", 1.0),
]
FORBID_PAIRS = [("L_THIGH+R_THIGH", "L_Hip", "R_Hip")]

# Spot-check expectations: (clip_idx, pair_name) -> reference dwell fraction
# from results/Contact_Physics_analysis_crow_pair_final/*/contact_config_match.md.
EXPECTED_DWELL = {
    (0, "L_HAND:G"): 0.700,
    (0, "R_HAND:G"): 0.705,
    (0, "L_SHANK+L_UPPER_ARM"): 0.574,
    (0, "R_SHANK+R_UPPER_ARM"): 0.580,
    (0, "L_THIGH+TRUNK"): 0.611,
    (0, "R_THIGH+TRUNK"): 0.604,
    (1, "R_THIGH+L_UPPER_ARM"): 0.497,
    (1, "R_THIGH+L_FOREARM"): 0.330,
}


def pair_name_variants(name):
    """The JSON stores zone pairs in ZONE_ORDER-canonical 'A+B' form; accept
    either order so a vocabulary/JSON ordering mismatch cannot silently zero a
    column."""
    variants = {name}
    if "+" in name:
        a, b = name.split("+")
        variants.add(f"{b}+{a}")
    return variants


def rasterize_pair(segments, pair_name, num_frames):
    """1.0 on frames covered by a non-statically_inconsistent segment whose
    contact list contains the pair; [start_frame, end_frame] inclusive."""
    mask = np.zeros(num_frames, dtype=np.float32)
    variants = pair_name_variants(pair_name)
    for seg in segments:
        if seg["statically_inconsistent"]:
            continue
        seg_pairs = {c["pair"] for c in seg["contacts"]}
        if variants & seg_pairs:
            mask[seg["start_frame"] : seg["end_frame"] + 1] = 1.0
    return mask


def smooth_columns(mask_np, window=SMOOTH_WINDOW):
    """Per-column moving average with replicate padding -- exactly the
    motion_lib.smooth_contacts convention (uniform conv1d kernel, replicate
    pad, clamp to [0,1]).  Applied per motion, never across boundaries."""
    x = torch.from_numpy(mask_np).float()  # [T, C]
    kernel = torch.ones(1, 1, window, dtype=torch.float32) / window
    pad = window // 2
    xc = x.t().unsqueeze(1)  # [C, 1, T]
    xc = torch.nn.functional.pad(xc, (pad, pad), mode="replicate")
    out = torch.nn.functional.conv1d(xc, kernel, padding=0).squeeze(1).t()
    return out.clamp(0.0, 1.0)


def build_geom_table(mjcf_path, body_names):
    typed = parse_typed_geoms(mjcf_path, body_names)
    B = len(body_names)
    geom_type = torch.zeros(B, dtype=torch.int64)
    geom_radius = torch.zeros(B, dtype=torch.float32)
    geom_p0 = torch.zeros(B, 3, dtype=torch.float32)
    geom_p1 = torch.zeros(B, 3, dtype=torch.float32)
    for i, name in enumerate(body_names):
        geoms = typed[name]
        assert len(geoms) == 1, f"{name}: expected exactly 1 collision geom, got {len(geoms)}"
        g = geoms[0]
        if g["type"] == "sphere":
            geom_type[i] = 1
            geom_radius[i] = float(g["radius"])
            c = torch.from_numpy(np.asarray(g["center"], dtype=np.float32))
            geom_p0[i] = c
            geom_p1[i] = c
        elif g["type"] == "capsule":
            geom_type[i] = 2
            geom_radius[i] = float(g["radius"])
            seg = torch.from_numpy(np.asarray(g["seg"], dtype=np.float32))
            geom_p0[i] = seg[0]
            geom_p1[i] = seg[1]
        else:  # box -> type 0 (box/none); radius 0, p0 == p1 == box center (informative only)
            geom_type[i] = 0
            geom_radius[i] = 0.0
            c = torch.from_numpy(np.asarray(g["center"], dtype=np.float32))
            geom_p0[i] = c
            geom_p1[i] = c
    return geom_type, geom_radius, geom_p0, geom_p1


def reference_pair_gap(motion, body_names, typed, body_a, body_b):
    """Signed surface gap series [T] between two bodies' geoms on the reference."""
    ia, ib = body_names.index(body_a), body_names.index(body_b)
    ga = geom_to_world(typed[body_a][0], motion["rigid_body_pos"][:, ia], motion["rigid_body_rot"][:, ia])
    gb = geom_to_world(typed[body_b][0], motion["rigid_body_pos"][:, ib], motion["rigid_body_rot"][:, ib])
    gap, _, _ = geom_pair_distance(ga, gb)
    return gap


def main():
    torch.set_num_threads(1)
    body_names = mjcf_body_names(MJCF)
    assert len(body_names) == 24, body_names
    typed = parse_typed_geoms(MJCF, body_names)

    with open(YAML_PATH) as f:
        motion_entries = yaml.safe_load(f)["motions"]
    stems = [Path(e["file"]).stem for e in motion_entries]
    assert len(stems) == 2, stems

    clips = []
    for stem in stems:
        motion = torch.load(str(CLIP_DIR / f"{stem}.motion"), map_location="cpu", weights_only=False)
        with open(JSON_DIR / f"{stem}.json") as f:
            ann = json.load(f)
        T = motion["rigid_body_pos"].shape[0]
        assert ann["num_frames"] == T, (stem, ann["num_frames"], T)
        assert int(motion.get("fps", FPS)) == FPS and ann["fps"] == FPS
        clips.append({"stem": stem, "motion": motion, "ann": ann, "T": T})

    num_frames = torch.tensor([c["T"] for c in clips], dtype=torch.int64)
    frame_starts = torch.tensor([0, clips[0]["T"]], dtype=torch.int64)
    total_T = int(num_frames.sum())
    P = len(ENCOURAGE_PAIRS)

    # Contract cross-check: the table is READ from the v2 sources (identical
    # kinematics, richer sidecars) but DECLARES itself tied to MOTION_PT (the
    # label-repaired v3 package). Assert the declared target actually agrees
    # in clip identity, order, and frame counts — otherwise a regenerated v3
    # package would silently disagree with the motion_id mapping shipped here.
    v3 = torch.load(str(REPO / MOTION_PT), map_location="cpu", weights_only=False)
    v3_names = [Path(str(f)).stem for f in v3["motion_files"]]
    v3_frames = [int(n) for n in v3["motion_num_frames"]]
    assert v3_names == stems and v3_frames == num_frames.tolist(), (
        f"declared contract target {MOTION_PT} disagrees with the packaging "
        f"sources: v3 clips {v3_names} frames {v3_frames} vs packaged "
        f"{stems} frames {num_frames.tolist()}"
    )

    # ------------------------------------------------------------------ #
    # Encourage masks: rasterize from JSON segments, then smooth per clip.
    # ------------------------------------------------------------------ #
    raw_encourage = []  # per clip [T, P] pre-smoothing (kept for validation)
    encourage_rows = []
    for c in clips:
        raw = np.stack(
            [rasterize_pair(c["ann"]["segments"], name, c["T"]) for name, *_ in ENCOURAGE_PAIRS],
            axis=1,
        )
        raw_encourage.append(raw)
        encourage_rows.append(smooth_columns(raw))
    encourage_mask = torch.cat(encourage_rows, dim=0)

    # ------------------------------------------------------------------ #
    # Forbid mask: side crow only, from reference geometry.  1.0 where the
    # reference L_Hip<->R_Hip surface gap exceeds FORBID_GAP_M -- frames where
    # the reference itself dips below are masked OFF so the reference is never
    # in violation of its own forbid.
    # ------------------------------------------------------------------ #
    forbid_stats = {}
    raw_forbid = []
    forbid_rows = []
    for ci, c in enumerate(clips):
        raw = np.zeros((c["T"], len(FORBID_PAIRS)), dtype=np.float32)
        if ci == 1:  # side crow
            # Hands-only phase (hands:G present, no foot:G) from the cleaned
            # segments -- the hold, where the invented thigh-thigh load path lives.
            hands_only = np.zeros(c["T"], dtype=bool)
            for seg in c["ann"]["segments"]:
                seg_pairs = {ct["pair"] for ct in seg["contacts"]}
                if ({"L_HAND:G", "R_HAND:G"} & seg_pairs
                        and not {"L_FOOT:G", "R_FOOT:G"} & seg_pairs):
                    hands_only[seg["start_frame"] : seg["end_frame"] + 1] = True
            for fi, (fname, ba, bb) in enumerate(FORBID_PAIRS):
                gap = reference_pair_gap(c["motion"], body_names, typed, ba, bb).numpy()
                assert gap.min() > PSI_ZERO_M, (
                    f"{fname}: reference gap dips to {gap.min():.4f} m, inside the "
                    f"psi(d) ramp ({PSI_ZERO_M} m) -- the mask rule alone no longer "
                    "protects the reference; revisit FORBID_GAP_M.")
                on = gap > FORBID_GAP_M
                raw[:, fi] = on.astype(np.float32)
                forbid_stats[fname] = {
                    "gap_min": float(gap.min()),
                    "gap_p5": float(np.percentile(gap, 5)),
                    "gap_p50": float(np.percentile(gap, 50)),
                    "frac_masked_off": float((~on).mean()),
                    "frac_on": float(on.mean()),
                    "frac_on_hold": float(on[hands_only].mean()),
                    "hold_frames": int(hands_only.sum()),
                }
        raw_forbid.append(raw)
        forbid_rows.append(smooth_columns(raw))
    forbid_mask = torch.cat(forbid_rows, dim=0)

    # ------------------------------------------------------------------ #
    # Geom table.
    # ------------------------------------------------------------------ #
    geom_type, geom_radius, geom_p0, geom_p1 = build_geom_table(MJCF, body_names)
    geom_expect = {
        "L_Hip": (2, 0.055), "R_Hip": (2, 0.055), "L_Knee": (2, 0.05), "R_Knee": (2, 0.05),
        "Chest": (1, 0.11), "L_Shoulder": (2, 0.045), "R_Shoulder": (2, 0.045),
        "L_Elbow": (2, 0.04), "R_Elbow": (2, 0.04),
    }
    for name, (gt, gr) in geom_expect.items():
        i = body_names.index(name)
        assert int(geom_type[i]) == gt and abs(float(geom_radius[i]) - gr) < 1e-4, (
            name, int(geom_type[i]), float(geom_radius[i]))

    body_index = {n: i for i, n in enumerate(body_names)}
    pair_names = [p[0] for p in ENCOURAGE_PAIRS]
    pair_body_a = torch.tensor([body_index[p[1]] for p in ENCOURAGE_PAIRS], dtype=torch.int64)
    pair_body_b = torch.tensor(
        [body_index[p[2]] if p[2] is not None else -1 for p in ENCOURAGE_PAIRS], dtype=torch.int64
    )
    pair_kappa = torch.tensor([p[3] for p in ENCOURAGE_PAIRS], dtype=torch.float32)
    forbid_names = [p[0] for p in FORBID_PAIRS]
    forbid_body_a = torch.tensor([body_index[p[1]] for p in FORBID_PAIRS], dtype=torch.int64)
    forbid_body_b = torch.tensor([body_index[p[2]] for p in FORBID_PAIRS], dtype=torch.int64)

    # ------------------------------------------------------------------ #
    # Final contract asserts.
    # ------------------------------------------------------------------ #
    assert encourage_mask.shape == (total_T, P)
    assert forbid_mask.shape == (total_T, len(FORBID_PAIRS))
    assert frame_starts[0] == 0 and frame_starts[1] == num_frames[0]
    assert int(frame_starts[-1] + num_frames[-1]) == total_T
    for t in (encourage_mask, forbid_mask):
        assert not torch.isnan(t).any()
        assert t.min() >= 0.0 and t.max() <= 1.0
    assert forbid_mask[: clips[0]["T"]].abs().sum() == 0.0, "crow rows of forbid_mask must be zero"

    out = {
        "version": f"{date.today().isoformat()} pair-aware contact reward reference targets v1.1 "
                   f"(crow pair, 13 encourage + 1 forbid @ ref-gap>{FORBID_GAP_M * 100:.1f}cm, "
                   f"7-frame smoothed)",
        "motion_pt": MOTION_PT,
        "fps": FPS,
        "num_motions": len(clips),
        "motion_names": stems,
        "num_frames": num_frames,
        "frame_starts": frame_starts,
        "pair_names": pair_names,
        "pair_body_a": pair_body_a,
        "pair_body_b": pair_body_b,
        "pair_kappa": pair_kappa,
        "encourage_mask": encourage_mask,
        "forbid_names": forbid_names,
        "forbid_body_a": forbid_body_a,
        "forbid_body_b": forbid_body_b,
        "forbid_mask": forbid_mask,
        "body_names": body_names,
        "geom_type": geom_type,
        "geom_radius": geom_radius,
        "geom_p0": geom_p0,
        "geom_p1": geom_p1,
    }
    torch.save(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH}")

    # ------------------------------------------------------------------ #
    # Validation vs the geometric detector's per-frame 'active' series.
    # ------------------------------------------------------------------ #
    lines = [
        "# yoga_yogi_crow_pair_reftargets.pt -- packaging report",
        "",
        f"- generated: {date.today().isoformat()}",
        f"- output: `{OUT_PATH.relative_to(REPO)}`",
        f"- motion package (contract tie): `{MOTION_PT}`",
        f"- clips (yaml order = motion_id): {stems}",
        f"- frames: {num_frames.tolist()} (frame_starts {frame_starts.tolist()}, total {total_T})",
        f"- smoothing: {SMOOTH_WINDOW}-frame moving average, replicate padding, per clip "
        f"(motion_lib.smooth_contacts convention)",
        "",
        "## 1. Rasterized dwell vs geometric detector ('active' series), pre-smoothing",
        "",
        "Jaccard is frame-level between the rasterized segment mask and "
        "compute_active_pairs' cleaned 'active' mask for the same zone pair. "
        "'-' = pair absent from both (all-zero in this clip).",
        "",
    ]
    print("\n=== validation: rasterized vs detector 'active' ===")
    low_jaccard = []
    for ci, c in enumerate(clips):
        resolved = compute_active_pairs(
            CLIP_DIR / f"{c['stem']}.motion", MJCF, c["ann"]["thresholds"]
        )
        active = resolved["active"]
        lines += [f"### motion_id {ci}: {c['stem']}", "",
                  "| pair | raster dwell | detector dwell | Jaccard |",
                  "|---|---|---|---|"]
        for pi, (name, *_rest) in enumerate(ENCOURAGE_PAIRS):
            raster = raw_encourage[ci][:, pi] > 0.5
            key = name if name in active else next(
                (v for v in pair_name_variants(name) if v in active), None)
            det = active[key].astype(bool) if key is not None else np.zeros(c["T"], dtype=bool)
            union = (raster | det).sum()
            inter = (raster & det).sum()
            if union == 0:
                jac_str, jac = "-", None
            else:
                jac = inter / union
                jac_str = f"{jac:.3f}"
                if jac < 0.9:
                    low_jaccard.append((ci, name, jac))
            row = (f"| `{name}` | {raster.mean() * 100:6.1f}% | {det.mean() * 100:6.1f}% "
                   f"| {jac_str} |")
            lines.append(row)
            print(f"  clip{ci} {name:24s} raster {raster.mean() * 100:5.1f}%  "
                  f"active {det.mean() * 100:5.1f}%  J={jac_str}")
        lines.append("")

    # ------------------------------------------------------------------ #
    # Spot checks vs contact_config_match.md reference dwell.
    # ------------------------------------------------------------------ #
    lines += ["## 2. Expected-dwell spot checks (vs contact_config_match.md reference column)",
              "", "| clip | pair | expected | rasterized | delta (pp) |", "|---|---|---|---|---|"]
    print("\n=== spot checks ===")
    for (ci, name), expected in EXPECTED_DWELL.items():
        pi = pair_names.index(name)
        got = float((raw_encourage[ci][:, pi] > 0.5).mean())
        delta = (got - expected) * 100
        lines.append(f"| {ci} | `{name}` | {expected * 100:.1f}% | {got * 100:.1f}% | {delta:+.1f} |")
        print(f"  clip{ci} {name:24s} expected {expected * 100:5.1f}%  got {got * 100:5.1f}%  ({delta:+.1f} pp)")

    # ------------------------------------------------------------------ #
    # Forbid stats.
    # ------------------------------------------------------------------ #
    lines += ["", "## 3. Forbid channel (side crow, reference L_Hip<->R_Hip surface gap)", "",
              f"Rule: forbid ON where the reference gap > {FORBID_GAP_M * 100:.1f} cm. The runtime "
              f"penalty ramp psi(d) = clamp(1 - d/{PSI_ZERO_M:.2f}, 0, 1) fires only below "
              f"{PSI_ZERO_M * 100:.0f} cm, so with the measured reference minimum above that, a "
              "policy matching the reference is at psi = 0 everywhere -- the invariant is carried "
              "by the ramp, and the mask threshold only needs a small safety margin above it "
              "(asserted at packaging time).", ""]
    print("\n=== forbid ===")
    for fname, st in forbid_stats.items():
        msg = (f"{fname}: gap min {st['gap_min'] * 100:.2f} cm, p5 {st['gap_p5'] * 100:.2f} cm, "
               f"p50 {st['gap_p50'] * 100:.2f} cm; ON {st['frac_on'] * 100:.1f}% of side-crow frames "
               f"({st['frac_masked_off'] * 100:.1f}% masked OFF by the {FORBID_GAP_M * 100:.1f} cm "
               f"rule); ON within the hands-only hold {st['frac_on_hold'] * 100:.1f}% "
               f"({st['hold_frames']} hold frames)")
        lines.append(f"- {msg}")
        print(f"  {msg}")
    lines += ["- crow (motion_id 0) rows: all zero", ""]

    if low_jaccard:
        lines += ["## Jaccard < 0.9 (investigate)", ""]
        for ci, name, jac in low_jaccard:
            lines.append(f"- clip {ci} `{name}`: J = {jac:.3f}")
        lines.append("")

    lines += ["## 4. Contract asserts", "",
              f"- encourage_mask {list(encourage_mask.shape)}, forbid_mask {list(forbid_mask.shape)}, "
              "all values in [0,1], no NaN",
              f"- frame_starts {frame_starts.tolist()} consistent with num_frames {num_frames.tolist()}",
              "- geom radii verified: L/R_Hip 0.055, L/R_Knee 0.05, Chest 0.11, "
              "L/R_Shoulder 0.045, L/R_Elbow 0.04",
              "- boxes (ankles/toes/wrists/hands) stored as geom_type 0, radius 0, p0=p1=box center",
              f"- forbid invariant: reference gap min stays above the psi(d) boundary "
              f"({PSI_ZERO_M * 100:.0f} cm) -- asserted", "",
              "## 5. Findings", "",
              "### Sub-1.0 Jaccards are exactly the statically_inconsistent segments (by design)", "",
              "The only Jaccard values below 1.000 are on side crow, and each deficit equals the "
              "frames of the skipped `statically_inconsistent` segments (312-314, 315-325, "
              "1104-1114) that contain the pair:", "",
              "- `L_HAND:G` in all 3 segments -> 25 frames = 1.84 pp (J=0.974)",
              "- `R_HAND:G` in 312-314 + 315-325 -> 14 frames = 1.03 pp (J=0.985)",
              "- `L_SHANK+R_SHANK` in 1104-1114 -> 11 frames = 0.81 pp (J=0.990)", "",
              "Nothing else differs: segments are the cleaned form of the same detector, and the "
              "raster matches the `active` series frame-for-frame everywhere else.", "",
              "### Forbid threshold history", "",
              "v1 used ref-gap > 3 cm, which masked the forbid OFF for 54.1% of side-crow frames "
              "including 87% of the hands-only hold (reference thigh-thigh gap p50 during the hold "
              "is 2.54 cm; the thighs are chronically 2.5-4.5 cm apart with legs together, cf. "
              "`BODY_LOOSE_EXCLUDE`). Since the runtime ramp psi(d) is zero above 2 cm and the "
              "reference never goes below 2.42 cm, the 3 cm rule was needlessly conservative; "
              "v1.1 lowered it to 2.2 cm (2 mm above the psi support), putting the forbid ON "
              "through the hold where the learned policy invents the 743 N thigh-thigh load path.", ""]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nwrote {REPORT_PATH}")
    return low_jaccard


if __name__ == "__main__":
    main()
