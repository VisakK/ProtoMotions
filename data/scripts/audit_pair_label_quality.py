# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit how solid each pair-reward target actually is in the REFERENCE clip.

The pair-aware reward (``examples/experiments/mimic/pair_contact_terms.py``) asks
the policy to close a set of body-body pairs to <= 1.5 cm. Those targets were
rasterized from the geometric contact-configuration annotation, which activates a
body-body pair through one of two tiers:

* **strict** -- surface gap < ``body_make`` = 2 cm (real, touching geometry);
* **loose** -- surface gap < ``body_loose_make`` = 4.5 cm, held still
  (< ``body_still_v`` = 0.06 m/s) for >= ``loose_dwell_s`` = 0.4 s. This tier
  exists to recover *compressed* body-body supports that the SMPL fit floats.

The loose tier is blanket-applied to every body-body pair except the single
hardcoded ``BODY_LOOSE_EXCLUDE = {L_THIGH, R_THIGH}``, whose own comment says the
pair is "chronically 2.5-4.5 cm apart whenever the legs are together". Any other
chronically-proximal pair in the same leg bundle gets latched at up to 4.5 cm
with no such protection.

This script measures, per pair, the *exact* reference surface gap (the same
closed-form kernels the annotation and the reward use) over the frames the reward
actually demands the pair, and classifies the target:

    REAL       p50 gap < 2 cm            -- strict-tier, genuine contact
    MARGINAL   2 cm <= p50 < 3.5 cm      -- loose-tier, plausible compression
    PROXIMITY  p50 >= 3.5 cm             -- loose-tier latch near the 4.5 cm edge

and reports what the trained policy actually did (from the rollout recorded by
``record_reward_terms.py``), so the closure the reward *drove* is visible.

Usage::

    PYTHONPATH=. python data/scripts/audit_pair_label_quality.py \
      --rollout-dir results/Reward_landscape \
      --out results/Reward_landscape/pair_label_audit.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
sys.path.insert(0, str(REPO))

from contact_geometry import geom_pair_distance, geom_to_world, parse_typed_geoms  # noqa: E402
from extract_contact_configs import (  # noqa: E402
    BODY_LOOSE_EXCLUDE,
    DEFAULT_THRESHOLDS,
    mjcf_body_names,
)
from package_pair_targets import ENCOURAGE_PAIRS, FORBID_PAIRS  # noqa: E402

MJCF = REPO / "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
CLIP_DIR = REPO / "data/smpl/yoga_yogi_crow_pair_v3_contacts"
STRICT_CM = DEFAULT_THRESHOLDS["body_make"] * 100
LOOSE_CM = DEFAULT_THRESHOLDS["body_loose_make"] * 100


def classify(p50_cm):
    if p50_cm < STRICT_CM:
        return "REAL"
    if p50_cm < 3.5:
        return "MARGINAL"
    return "PROXIMITY"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-dir", type=str, default="results/Reward_landscape")
    ap.add_argument(
        "--out", type=str, default="results/Reward_landscape/pair_label_audit.md"
    )
    a = ap.parse_args()

    body_names = mjcf_body_names(MJCF)
    typed = parse_typed_geoms(MJCF, body_names)
    rt = torch.load(
        REPO / "data/smpl/yoga_yogi_crow_pair_reftargets.pt",
        map_location="cpu",
        weights_only=False,
    )
    pair_names = list(rt["pair_names"])
    enc = rt["encourage_mask"].numpy()
    forb = rt["forbid_mask"].numpy()
    starts = rt["frame_starts"].tolist()
    nframes = rt["num_frames"].tolist()

    rollouts = {}
    for p in sorted(Path(a.rollout_dir).glob("*/rollout_rewards.npz")):
        d = np.load(p, allow_pickle=True)
        rollouts[str(d["motion_name"])] = d

    spec = [(n, ba, bb, k) for n, ba, bb, k in ENCOURAGE_PAIRS if bb is not None]
    spec += [(n, ba, bb, None) for n, ba, bb in FORBID_PAIRS]

    rows = []
    for ci, stem in enumerate(rt["motion_names"]):
        motion = torch.load(
            str(CLIP_DIR / f"{stem}.motion"), map_location="cpu", weights_only=False
        )
        s, T = starts[ci], nframes[ci]
        roll = rollouts.get(stem)

        for name, ba, bb, kappa in spec:
            is_forbid = kappa is None
            if is_forbid:
                mask = forb[s : s + T, 0] > 0.5
            else:
                mask = enc[s : s + T, pair_names.index(name)] > 0.5
            if not mask.any():
                continue

            ia, ib = body_names.index(ba), body_names.index(bb)
            ga = geom_to_world(
                typed[ba][0], motion["rigid_body_pos"][:, ia], motion["rigid_body_rot"][:, ia]
            )
            gb = geom_to_world(
                typed[bb][0], motion["rigid_body_pos"][:, ib], motion["rigid_body_rot"][:, ib]
            )
            gap = geom_pair_distance(ga, gb)[0].numpy() * 100.0
            sel = gap[mask]

            learned = None
            if roll is not None and not is_forbid:
                i = list(roll["pair_names"]).index(name)
                g = roll["pair_gap"][:, :, i] * 100
                m = roll["pair_gate"][:, :, i] > 0.5
                if m.any():
                    learned = float(np.nanmedian(g[m]))

            rows.append(
                {
                    "clip": "crow" if ci == 0 else "side",
                    "pair": name,
                    "bodies": f"{ba}+{bb}",
                    "kappa": kappa,
                    "forbid": is_forbid,
                    "dwell": float(mask.mean()),
                    "ref_min": float(sel.min()),
                    "ref_p50": float(np.median(sel)),
                    "ref_max": float(sel.max()),
                    "verdict": classify(float(np.median(sel))),
                    "learned_p50": learned,
                    "loose_excluded": frozenset(name.split("+")) in BODY_LOOSE_EXCLUDE,
                }
            )

    L = [
        "# Are the pair-reward targets real contacts in the reference?",
        "",
        f"Strict (touching) tier is a surface gap < **{STRICT_CM:.1f} cm**; the loose "
        f"tier latches anything still for 0.4 s within **{LOOSE_CM:.1f} cm**. Only "
        f"`{'/'.join(sorted(list(BODY_LOOSE_EXCLUDE)[0]))}` is excluded from the loose tier.",
        "",
        "Gaps are the exact closed-form surface distance between the MJCF collision "
        "primitives, measured over the frames the reward actually demands the pair.",
        "",
        "| clip | pair | k | dwell | ref gap min / **p50** / max (cm) | verdict | learned p50 (cm) | closure driven |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["clip"], -r["ref_p50"])):
        k = "forbid" if r["forbid"] else f"{r['kappa']:g}"
        lp = f"{r['learned_p50']:.1f}" if r["learned_p50"] is not None else "—"
        drive = (
            f"{r['ref_p50'] - r['learned_p50']:+.1f} cm"
            if r["learned_p50"] is not None
            else "—"
        )
        L.append(
            f"| {r['clip']} | `{r['pair']}` ({r['bodies']}) | {k} | "
            f"{100 * r['dwell']:.0f}% | {r['ref_min']:.1f} / **{r['ref_p50']:.1f}** / "
            f"{r['ref_max']:.1f} | **{r['verdict']}** | {lp} | {drive} |"
        )

    bad = [r for r in rows if r["verdict"] == "PROXIMITY" and not r["forbid"]]
    marg = [r for r in rows if r["verdict"] == "MARGINAL" and not r["forbid"]]
    L += [
        "",
        f"**{len(bad)} of {len([r for r in rows if not r['forbid']])} encourage targets "
        f"are PROXIMITY-tier** (reference p50 >= 3.5 cm, i.e. latched by the loose tier "
        f"near its {LOOSE_CM:.1f} cm edge), and {len(marg)} more are MARGINAL.",
        "",
    ]
    for r in bad:
        L.append(
            f"- `{r['pair']}` ({r['clip']}): reference surfaces never closer than "
            f"{r['ref_min']:.1f} cm, median {r['ref_p50']:.1f} cm — "
            f"{LOOSE_CM - r['ref_p50']:.1f} cm inside the loose threshold. The reward "
            f"asks for <= 1.5 cm; the policy delivered {r['learned_p50']:.1f} cm."
        )
    L.append("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
