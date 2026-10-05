# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card R2 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: the synthetic clip writer.

Each of the selected PhysX variants (``selected_physx.json``, the user's T6 reading: 29 variants of E1, E3, B1, E2
and E5) becomes one full clip in S's clip frame: a real lead-in from S's clip, the synthesised transition, and a real
lead-out from D's clip. R2 writes clips, holds and lineage; R3 packages them into release v3.

The spliced clip (``layout``; L lead-in frames, the variant's N frames, k_d = 17 its departure, k_a its arrival)
-----------------------------------------------------------------------------------------------------------------
==============  =================================  ======================================================  =========
part            spliced frames                     what                                                    kind
==============  =================================  ======================================================  =========
lead-in         0 .. L - 1                         S's clip from the ``frame_start`` of the last hold      real
                                                   before S to S's exemplar - 1
S's exemplar    L                                  S's clip, frame ``frame_hold``                          real
settle          L + 1 .. L + k_d                   variant frames 0 .. k_d - 1, blended from S's exemplar  blend
transition      L + 1 + k_d .. L + k_a             variant frames k_d .. k_a - 1                           synthetic
hold at D       L + 1 + k_a .. L + N               variant frames k_a .. N - 1, blended toward D'          blend
lead-out        L + 1 + N .. end                   D's clip from D's exemplar to the end of the first      real
                                                   hold after D (or the clip's end), placed as D'
==============  =================================  ======================================================  =========

* **The variant's clock** (``variant_clock``): frame k is at t = t_first + k / 60 with t_first = -0.3 + 1/60 (the
  PhysX run restored S's exemplar at t = -0.3 and held it for ``settle_s``), so the departure is k_d = 17 and the
  arrival the first frame with t >= T, k_a = 17 + ceil(60 T); the hold at D runs to the last frame.
* **The variant is in S's clip frame**: the run started from S's release exemplar. That is checked, not assumed:
  variant frame 0 must sit within ``START_ALL_CM`` of S's exemplar on every body, else the build stops.
* **D'** is D's exemplar placed by the generator's own hand-anchor transform (``sketch.hand_anchor_transform(S
  exemplar, D exemplar)``: a yaw about the vertical and an XY shift that put D's hand midpoint on S's and its hand
  line along S's). It moves the root only -- position, rotation (quaternions are ProtoMotions' xyzw) -- and leaves
  every joint coordinate as it is; the lead-out is D's clip under the same transform.
* **The S seam**, over the settle: settle frame j = ``blend(S's exemplar, variant frame j, w_j)``,
  w_j = smoothstep((j + 1) / k_d), so frame k_d - 1 is the variant's own. PhysX settles the restored pose in its
  first substeps (variant frame 0 sits up to 0.98 cm off the exemplar); unblended that would be a one-frame jump.
* **The D seam**, over the whole hold at D: variant frame k becomes ``blend(variant frame k, D', w_k)``,
  w_k = smoothstep((k - k_a + 1) / (N - k_a + 1)). w reaches 1 on the frame *after* the last variant frame, which
  is the lead-out's first frame (D' itself), so no frame repeats. (The card's ``/ (N - k_a)`` reaches 1 on the last
  variant frame and repeats D' there.) The quintic's flat ends still leave the last blend frames (and the first
  settle frame) within micrometres of the real frame next to them: a near-repeat (``NEAR_REPEAT_M``) is allowed
  only inside a hold window. The blend carries the residuals the user accepted (PLAN.MD top, T6): the E1/E3 palms'
  12 deg, the landings' 2.8-10 cm of planted-body offset, and E2/E5's low chaturanga.
* **Blend math** (``blend``): slerp of the root rotation and of every joint's local rotation, each exp-map
  coordinate the nearest representative inside the box (``retarget.nearest_representative``); the root is placed
  so the hands' midpoint (``L_Hand``, ``R_Hand``) follows the same blend between the two poses' midpoints
  (``quasistatic.anchor_trajectory``'s rule with a moving anchor).
* **Blend frames are checked** (``contract``) against ``quasistatic.certify``'s contract, computed directly
  (certify needs a solved problem and a schedule), per blend part against that part's own endpoints: no joint
  coordinate more than 1 deg past the box, every collider point >= ``FLOOR_CM`` above the floor, and no body pair
  overlapping deeper than ``OVERLAP_M`` that is not already overlapping in the part's endpoints (the settle: S's
  exemplar and the variant's first frame; the hold at D: D's exemplar and the variant's last frame); a brace of the
  part's hold (S's on the settle, D's on the hold at D) whose zones its exemplar holds within
  ``retarget_v2.PAIR_REACH_M`` may touch. A blend part with a failing frame goes through ``export_physx``'s
  de-penetration solve (``depenetrate``: ``DEPENETRATE`` weights, ``PIN_FRAMES`` neighbours pinned either side so
  the edit tapers to nothing at the real and synthetic frames); ``<stem>.json`` records every edit. The crow's feet
  rest together (L_Ankle / R_Ankle 1.0 mm deep), and slerping two leg chains between them drives the ankles up to
  9 mm into each other.
* **A hold's window holds its label** (``content_windows``): no synthetic or blend frame inside S's or D's window
  touches the floor (``rigid_body_contacts``, 1 cm) with a zone the hold does not label as ground. S's window ends
  before the first such frame of the settle (at the departure otherwise), D's starts after the last such frame of
  its blend (at the arrival otherwise). One clip needs it: SYN_E2_jumpback_mid_s3_t6rpx12's variant lands knee down
  and the blend toward D' lifts the knee only after 23 frames. Unplanned floor contacts on transition frames (the
  E2 jump-backs' left knee) are the variant's own content and are recorded, not edited.

Checks (``check_clip``, then ``check_motionlib`` and ``check_graph`` on the library; ``--check`` re-runs them all)
-----------------------------------------------------------------------------------------------------------------
* the FK round trip (``fit_writer.round_trip``, 1e-5 m), plant v2's identity, the pressure channels zero;
* real frames against their sources after the recorded transform: positions ``REAL_POS_M``, ``dof_pos`` exactly,
  velocities ``REAL_VEL`` where pose_lib's forward difference over f .. f + 3 reads only the source's own frames
  (an angular velocity off by more must equal another horizon's candidate whose magnitude ties the smallest:
  ``horizon_candidates``);
* transition frames equal the variant's (positions ``REAL_POS_M``, ``dof_pos`` exactly);
* the four junctions (planted bodies <= ``JUNCTION_CM`` frame to frame); the seam window (``ACC_WINDOW`` frames
  either side of a junction's two frames): no body's acceleration (admit's second difference) above the larger of
  its two sources' hold-window maxima + ``ACC_MARGIN``, read twice and both gated: the card's (the human sources,
  S's and D's hold windows, on every junction) and the spliced parts' own (S, the variant's settle and hold at D,
  D, per junction); no two consecutive frames identical, a near-repeat only inside a hold window; no exp-map flip;
* the D blend's planted-body travel and the seam offsets before and after blending (recorded, not gated);
* the transform against the generator's (``run.json``'s ``anchor_yaw_deg``) and on FK of a real frame;
* the holds (``holds_well_formed``, ``inherits``, the windows' content), hold extension's velocity guard and its
  insertion dry-run;
* ``MotionLib`` loads the clips with release v2's 168 motions on the CPU; ``build_hold_graph_v2.py`` on a small
  library (x0, and x0 + x3s + x7s) keeps v2's node keys and pair names and has each edge's S -> D.

Written per clip (``fit_writer.write_motion``: pose_lib FK in float32, velocities, geometric contacts and plant
v2's ``plant_sha256``; then the three pressure channels as zeros with validity 0, as T6's export): ``<stem>.motion``,
``<stem>.holds.yaml`` (the clip's manifest entry, which R3 merges), ``<stem>.lineage.npz`` and ``<stem>.json`` (the
frame indices, the seam offsets before and after blending, each planted body's travel in the D blend, the
transform, every check). The holds copy their source holds verbatim apart from ``hold_id`` (``<stem>@<frame_hold>``),
``inherits``, the frames and times, and ``extend`` (true on S only).

Names are ``SYN_<edge>_<kind>_<timing>_s<seed>_<tag>`` (R3 and R4a reference them). ``t6px`` is a prefix of
``t6px12`` and ``t6rpx`` of ``t6rpx12``, so one synthetic stem is a substring of another: every per-stem filter
downstream (R3's ``--drop``) must match stems exactly, never by substring (the record lists the collisions). The
runtime's motion filters do match by substring (``support_exclude_motions``, ``physics_exclude_motions``,
``report_exclude_motions``, ``drag_report_motions``), so a pattern naming one of the shorter stems also selects the
longer one.

Outputs: ``data/smpl/reference_curation/synthetic_v3/<synthetic id>/`` and the record
``data/reference_curation/synthetic_v3/<synthetic id>.json``, written last and only when every check passes. The id
hashes the inputs (by sha256), every variant's files, the builders' sources (everything whose code shapes the
written bytes), plant v2's MJCF and the parameters; an id is never rebuilt in place.

CLI (single-threaded, without ``REFERENCE_PLANT``)::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.splice_v3 --build
    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.splice_v3 --check <synthetic id>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from extract_contact_configs import ZONES
from reference_curation import fit_writer as fw
from reference_curation import ids

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.splice_v3"
SCHEMA_VERSION = 1
SPLICE_VERSION = "synthetic_v3"
PLANT = "v2"
FPS = 60
GG = REPO / "expert_revist/graph_growth_2026_10_03"
SELECTED = GG / "selected_physx.json"
ADMITTED = GG / "admitted_physx.json"
EDGES = GG / "edges.json"
RELEASE_V2_ID = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
RELEASE_V2_RECORD = ids.DATA_ROOT / "releases" / f"{RELEASE_V2_ID}.json"
HEAVY_ROOT = REPO / "data/smpl/reference_curation" / SPLICE_VERSION
RECORD_ROOT = ids.DATA_ROOT / SPLICE_VERSION
SCRIPTS = REPO / "data/scripts"

EDGE_ORDER = ("E1", "E3", "B1", "E2", "E5")      # the user's D1 decision's order (selected_physx.json)
EDGE_KIND = {"E1": "press", "E3": "lower", "B1": "jumpplank", "E2": "jumpback", "E5": "floatdown"}
TAGS = ("t6px", "t6px12", "t6rpx", "t6rpx12")
KIND = {"real": 0, "synthetic": 1, "blend": 2}
HANDS = ("L_Hand", "R_Hand")
NAMES = ("Pelvis L_Hip L_Knee L_Ankle L_Toe R_Hip R_Knee R_Ankle R_Toe Torso Spine Chest Neck Head L_Thorax "
         "L_Shoulder L_Elbow L_Wrist L_Hand R_Thorax R_Shoulder R_Elbow R_Wrist R_Hand").split()
BI = {n: i for i, n in enumerate(NAMES)}
# The holds' copied fields are everything but these (the card's "New" list, plus extend).
HOLD_NEW = ("hold_id", "inherits", "frame_start", "frame_hold", "frame_end", "t_start", "t_hold", "t_end",
            "duration_s", "extend")

START_ALL_CM = 1.0         # seams.START_ALL_CM: variant frame 0 against S's exemplar, every body (T6: <= 0.98 cm)
FLOOR_CM = -0.5            # quasistatic.certify's contract: every collider point
OVERLAP_M = 0.001          # ... no new body-pair overlap deeper than this
OVERLAP_TOL_M = 1e-4       # ... beyond the solver's tolerance on the guard's -1 mm floor (quasistatic.OVERLAP_TOL_M)
PAIR_REACH_M = 0.02        # retarget_v2.PAIR_REACH_M: a brace the exemplar holds within this may touch
PIN_FRAMES = 2             # a de-penetration solve pins this many frames either side of the blend frames
JUNCTION_CM = 0.5          # the four junctions: planted bodies move <= this from one frame to the next
ACC_WINDOW = fw.VELOCITY_MAX_HORIZON     # seam window: this many frames either side of a junction's two frames
ACC_MARGIN = 10.0          # m/s^2 over the larger of the two sources' hold-window maxima
REAL_POS_M = 1e-5          # real frames against their sources (positions), and the FK round trip
REAL_VEL = 1e-2            # m/s (and rad/s) away from the seams
HORIZON_TIE = 1e-3         # rad/s: an angular velocity off by more than REAL_VEL must equal another horizon's candidate
HORIZON_MAG_TIE = 1e-4     # rad/s: ... whose magnitude is within this of the smallest candidate's (a genuine tie)
NEAR_REPEAT_M = 1e-5       # a step below this on every body repeats a frame at the build's FK tolerance
BOX_DEG = 1.0              # quasistatic.certify's contract: no joint coordinate this far past the box
TRANSFORM_YAW_DEG = 0.01   # run.json rounds anchor_yaw_deg to 0.01 deg
EXTENSION_S = (3.0, 7.0)   # R3's hold-extension variants, dry-run here


# --------------------------------------------------------------------------- #
# Pure rules (unit-tested)
# --------------------------------------------------------------------------- #
def smoothstep(x) -> np.ndarray:
    """The quintic smoothstep of ``sketch``/``quasistatic`` (zero first and second derivative at 0 and 1)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def variant_clock(frames: int, t_first: float, T: float, settle_s: float, hold_s: float, fps: int = FPS) -> dict:
    """``{k_d, k_a, N, hold_frames}`` of a variant (module doc); asserts the card's three facts."""
    N = int(frames)
    if abs(t_first - (-settle_s + 1.0 / fps)) > 1e-6:
        raise ValueError(f"t_first {t_first} is not -settle_s + 1/fps = {-settle_s + 1.0 / fps}")
    t = t_first + np.arange(N) / fps
    k_d = int(round(settle_s * fps)) - 1
    if abs(t[k_d]) > 1e-6:
        raise ValueError(f"the departure frame {k_d} is at t = {t[k_d]}, not 0")
    k_a = k_d + int(math.ceil(fps * T - 1e-6))
    if not (t[k_a] >= T - 1e-6 and t[k_a - 1] < T - 1e-6):
        raise ValueError(f"the arrival frame {k_a} (t = {t[k_a]}) is not the first with t >= T = {T}")
    if t[-1] < T + hold_s - 1e-6:
        raise ValueError(f"the variant ends at t = {t[-1]}, before T + hold_s = {T + hold_s}")
    return {"k_d": k_d, "k_a": k_a, "N": N, "hold_frames": N - k_a}


def s_weights(k_d: int) -> np.ndarray:
    """``[k_d]`` the variant's weight on settle frame j: smoothstep((j + 1) / k_d) (1 on the last)."""
    return smoothstep((np.arange(k_d) + 1.0) / k_d)


def d_weights(k_a: int, N: int) -> np.ndarray:
    """``[N - k_a]`` D''s weight on variant frame k = k_a .. N - 1: smoothstep((k - k_a + 1) / (N - k_a + 1)), so it
    reaches 1 on the frame after the last (the lead-out's first, D' itself) and no frame repeats."""
    return smoothstep((np.arange(k_a, N) - k_a + 1.0) / (N - k_a + 1.0))


def layout(L: int, clock: dict, n_out: int) -> dict:
    """Spliced frame indices of every part (module doc; ranges inclusive)."""
    k_d, k_a, N = clock["k_d"], clock["k_a"], clock["N"]
    return {"lead_in": [0, L - 1], "s_exemplar": L, "settle": [L + 1, L + k_d], "departure": L + 1 + k_d,
            "transition": [L + 1 + k_d, L + k_a], "arrival": L + 1 + k_a, "d_hold": [L + 1 + k_a, L + N],
            "lead_out": [L + 1 + N, L + N + n_out], "num_frames": L + 1 + N + n_out,
            # the four junctions: (frame before, frame after)
            "junctions": {"lead_in|s_exemplar": [L - 1, L], "s_exemplar|settle": [L, L + 1],
                          "settle|transition": [L + k_d, L + 1 + k_d], "d_hold|lead_out": [L + N, L + 1 + N]}}


def synthetic_name(row: dict) -> str:
    """``SYN_<edge>_<kind>_<timing>_s<seed>_<tag>`` of a selection row (the tag is the variant name's last token)."""
    edge, recipe = row["edge"], row["recipe"]
    timing = recipe["timing"].split("_")[-1]
    tag = row["variant"].rsplit("_", 1)[-1]
    if tag not in TAGS or timing not in ("high", "mid"):
        raise ValueError(f"{row['variant']}: unexpected tag {tag!r} or timing {timing!r}")
    return f"SYN_{edge}_{EDGE_KIND[edge]}_{timing}_s{int(recipe['seed'])}_{tag}"


def order_key(name: str) -> tuple:
    """(edge, timing, seed, tag): R3's fixed order of the synthetic clips."""
    _, edge, _, timing, seed, tag = name.split("_")
    return EDGE_ORDER.index(edge), timing, int(seed[1:]), TAGS.index(tag)


def name_checks(names: list[str], real_stems: list[str], variants_s=(0.0,) + EXTENSION_S) -> dict:
    """The card's three asserts (no name contains a real stem, none ends in ``_x<digits>s``, all unique), the
    reverse (no real stem contains ``SYN_``), and every synthetic-vs-synthetic substring collision over the names
    R3's extension will emit (a stem inside another clip's name; never one of its own duration variants)."""
    from reference_curation import hold_extension_v2 as hx

    contains_real = sorted((n, r) for n in names for r in real_stems if r in n)
    x_suffix = sorted(n for n in names if re.search(r"_x\d+s$", n))
    dupes = sorted({n for n in names if names.count(n) > 1})
    real_syn = sorted(r for r in real_stems if "SYN_" in r)
    emitted = {hx.variant_stem(n, d): n for n in names for d in variants_s}
    collisions = sorted([a, b] for a in names for b, src in emitted.items() if src != a and a in b)
    return {"contains_real_stem": contains_real, "ends_x_digits_s": x_suffix, "duplicates": dupes,
            "real_stems_with_SYN_": real_syn, "unique": len(set(names)) == len(names), "count": len(names),
            "name_collisions": collisions,
            "pass": not contains_real and not x_suffix and not dupes and not real_syn}


def place(pos: np.ndarray, rot_xyzw: np.ndarray, yaw: float, t_xy) -> tuple[np.ndarray, np.ndarray]:
    """Positions ``[..., 3]`` and rotations ``[..., 4]`` (xyzw) rotated by ``yaw`` about z and shifted by ``t_xy``
    (``sketch.apply_planar``'s map, which ``hand_anchor_transform``'s (yaw, t) is defined for)."""
    Rz = Rotation.from_euler("z", yaw)
    p = Rz.apply(np.asarray(pos, float).reshape(-1, 3)).reshape(np.shape(pos))
    p[..., :2] += np.asarray(t_xy, float)
    q = (Rz * Rotation.from_quat(np.asarray(rot_xyzw, float).reshape(-1, 4))).as_quat().reshape(np.shape(rot_xyzw))
    return p, q


def rotate_vectors(v: np.ndarray, yaw: float) -> np.ndarray:
    """Velocities ``[..., 3]`` under the transform (a yaw; the shift does not move a velocity)."""
    return Rotation.from_euler("z", yaw).apply(np.asarray(v, float).reshape(-1, 3)).reshape(np.shape(v))


def holds_well_formed(holds: list[dict], num_frames: int, x0: bool = True) -> list[str]:
    """Problems with a clip's holds: frames inside the clip, start <= hold <= end, ordered by ``frame_hold`` and
    disjoint, times = round(frame / 60, 4), ``duration_s`` = round(frames / 60, 3), exactly one ``extend``, and (on
    the x0 clip; a duration variant keeps its x0 ids) ``hold_id`` = ``<stem>@<frame_hold>``."""
    out = []
    for h in holds:
        fs, fh, fe = h["frame_start"], h["frame_hold"], h["frame_end"]
        if not (0 <= fs <= fh <= fe < num_frames):
            out.append(f"{h['hold_id']}: frames {fs}/{fh}/{fe} not ordered inside {num_frames}")
        for k, f in (("t_start", fs), ("t_hold", fh), ("t_end", fe)):
            if h[k] != round(f / FPS, 4):
                out.append(f"{h['hold_id']}: {k} {h[k]} != round({f}/{FPS}, 4)")
        if h["duration_s"] != round((fe - fs + 1) / FPS, 3):
            out.append(f"{h['hold_id']}: duration_s {h['duration_s']}")
        if x0 and h["hold_id"] != f"{h['hold_id'].rsplit('@', 1)[0]}@{fh}":
            out.append(f"{h['hold_id']}: the id's frame is not frame_hold {fh}")
    for a, b in zip(holds, holds[1:]):
        if not (a["frame_hold"] < b["frame_hold"] and a["frame_end"] < b["frame_start"]):
            out.append(f"{a['hold_id']} and {b['hold_id']} are not ordered and disjoint")
    if sum(bool(h.get("extend")) for h in holds) != 1:
        out.append(f"{sum(bool(h.get('extend')) for h in holds)} holds extend (one, S, should)")
    return out


def copy_hold(src: dict, stem: str, frames: tuple[int, int, int], extend: bool) -> dict:
    """``src`` verbatim, with the card's new fields: ``hold_id``, ``inherits``, the frames and times, ``extend``."""
    fs, fh, fe = (int(f) for f in frames)
    out = {k: v for k, v in src.items() if k not in HOLD_NEW}
    out.update(hold_id=f"{stem}@{fh}", inherits=src["hold_id"], frame_start=fs, frame_hold=fh, frame_end=fe,
               t_start=round(fs / FPS, 4), t_hold=round(fh / FPS, 4), t_end=round(fe / FPS, 4),
               duration_s=round((fe - fs + 1) / FPS, 3), extend=bool(extend))
    return out


def splice_holds(stem: str, s_clip: dict, s_hold: dict, d_clip: dict, d_hold: dict, lay: dict, lead_in_start: int,
                 lead_out_end: int, s_end: int | None = None, d_start: int | None = None) -> list[dict]:
    """The clip's holds (module doc): the lead-in holds, S (its own start to the departure, or ``s_end``;
    ``frame_hold`` = L, ``extend``), D (the arrival, or ``d_start``, to its own end; ``frame_hold`` = the lead-out's
    first frame) and the lead-out holds, each inheriting its source hold."""
    L, out0 = lay["s_exemplar"], lay["lead_out"][0]
    sfh, dfh = int(s_hold["frame_hold"]), int(d_hold["frame_hold"])
    s_end = lay["departure"] if s_end is None else int(s_end)
    d_start = lay["arrival"] if d_start is None else int(d_start)
    if not (L <= s_end <= lay["departure"] and lay["arrival"] <= d_start <= out0):
        raise ValueError(f"S's window end {s_end} or D's start {d_start} is outside its part")
    out = []
    for h in sorted(s_clip["holds"], key=lambda h: h["frame_hold"]):
        if h["hold_id"] != s_hold["hold_id"] and h["frame_start"] >= lead_in_start and h["frame_end"] < sfh:
            out.append(copy_hold(h, stem, tuple(f - lead_in_start for f in
                                                (h["frame_start"], h["frame_hold"], h["frame_end"])), False))
    out.append(copy_hold(s_hold, stem, (int(s_hold["frame_start"]) - lead_in_start, L, s_end), True))
    out.append(copy_hold(d_hold, stem, (d_start, out0, int(d_hold["frame_end"]) - dfh + out0), False))
    for h in sorted(d_clip["holds"], key=lambda h: h["frame_hold"]):
        if h["hold_id"] != d_hold["hold_id"] and h["frame_start"] > dfh and h["frame_end"] <= lead_out_end:
            out.append(copy_hold(h, stem, tuple(f - dfh + out0 for f in
                                                (h["frame_start"], h["frame_hold"], h["frame_end"])), False))
    return out


def ground_zones(hold: dict) -> list[str]:
    """The zones a hold labels as ground (``pairs_ground``: ``<ZONE>:G``)."""
    return sorted(p[:-2] for p in hold.get("pairs_ground", []) if p.endswith(":G"))


def zone_contacts(contacts) -> dict[str, np.ndarray]:
    """``{zone: [T] bool}``: a zone touches the floor when any of its bodies does (``rigid_body_contacts [T, 24]``,
    ``fit_writer``'s 1 cm geometric rule)."""
    c = np.asarray(contacts, bool)
    return {z: c[:, [BI[b] for b in bs]].any(1) for z, bs in ZONES.items()}


def unlabelled(zc: dict, frames, labelled) -> dict[str, list[int]]:
    """``{zone: frames}`` of the zones outside ``labelled`` that touch the floor on ``frames``."""
    frames = np.asarray(frames, int)
    return {z: [int(f) for f in frames[c[frames]]] for z, c in zc.items() if z not in labelled and c[frames].any()}


def content_windows(zc: dict, lay: dict, s_ground, d_ground) -> dict:
    """S's window end and D's window start (module doc): no synthetic or blend frame inside either window may
    touch the floor with a zone the hold does not label. S ends before the first offending frame of the settle and
    the departure (at the departure otherwise); D starts after the last offending frame of its blend (at the
    arrival otherwise)."""
    L, dep, arr, last = lay["s_exemplar"], lay["departure"], lay["arrival"], lay["d_hold"][1]
    s_bad = unlabelled(zc, np.arange(L + 1, dep + 1), set(s_ground))
    d_bad = unlabelled(zc, np.arange(arr, last + 1), set(d_ground))
    s_first = min((f[0] for f in s_bad.values()), default=None)
    d_last = max((f[-1] for f in d_bad.values()), default=None)
    return {"s_end": dep if s_first is None else s_first - 1, "s_end_card": dep, "s_unlabelled": s_bad,
            "d_start": arr if d_last is None else d_last + 1, "d_start_card": arr, "d_unlabelled": d_bad}


def near_repeats(pos: np.ndarray, holds: list[dict]) -> dict:
    """Consecutive frames whose step is below ``NEAR_REPEAT_M`` on every body (``pos [T, 24, 3]``), and those
    outside every hold window (a hold may rest; nothing else may stall)."""
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(1)
    near = np.nonzero(step < NEAR_REPEAT_M)[0]
    inside = [any(h["frame_start"] <= f and f + 1 <= h["frame_end"] for h in holds) for f in near]
    return {"pairs": [[int(f), int(f + 1), float(step[f])] for f in near],
            "outside_holds": [[int(f), int(f + 1)] for f, ok in zip(near, inside) if not ok],
            "min_step_m": float(step.min()), "pass": all(inside)}


# --------------------------------------------------------------------------- #
# Poses: (root_pos [n,3], root_rot [n,4] xyzw, dof [n,69] exp-map), float64
# --------------------------------------------------------------------------- #
def frames_of(motion: dict, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The stored coordinates of ``idx``: the root body's position and rotation and ``dof_pos`` (float64 of the
    stored float32, so a real frame is written back exactly)."""
    idx = torch.as_tensor(np.atleast_1d(np.asarray(idx, int)))
    return (motion["rigid_body_pos"][idx, 0].double().numpy(), motion["rigid_body_rot"][idx, 0].double().numpy(),
            motion["dof_pos"][idx].double().numpy())


def fk(sk, root_pos, root_rot_xyzw, dof) -> tuple[np.ndarray, np.ndarray]:
    """``(pos [n,24,3], rot [n,24,3,3])``: ``retarget.fk`` in float64."""
    from reference_curation import retarget as rt

    R = Rotation.from_quat(np.asarray(root_rot_xyzw, float).reshape(-1, 4)).as_matrix()
    with torch.no_grad():
        p, r = rt.fk(sk, torch.as_tensor(np.asarray(root_pos, float).reshape(-1, 3)), torch.as_tensor(R),
                     torch.as_tensor(np.asarray(dof, float).reshape(-1, 69)))
    return p.numpy(), r.numpy()


def hands_mid(sk, root_pos, root_rot, dof) -> np.ndarray:
    p, _ = fk(sk, root_pos, root_rot, dof)
    return p[:, [BI[h] for h in HANDS]].mean(1)


def blend(sk, a: tuple, b: tuple, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame, pose ``a`` blended toward ``b`` by ``w`` (``b``'s weight): slerped root and local rotations,
    exp-map coordinates as the nearest representative inside the box, the root placed so the hands' midpoint
    follows the same blend of the two midpoints. A frame with w == 0 or 1 is that pose's own coordinates."""
    from reference_curation import retarget as rt

    w = np.asarray(w, float)
    n = len(w)
    ra, qa, da = (np.asarray(x, float).reshape(n, -1) for x in a)
    rb, qb, db = (np.asarray(x, float).reshape(n, -1) for x in b)
    Ra, Rb = Rotation.from_quat(qa), Rotation.from_quat(qb)
    root_rot = (Ra * Rotation.from_rotvec(w[:, None] * (Ra.inv() * Rb).as_rotvec())).as_quat()
    La, Lb = Rotation.from_rotvec(da.reshape(-1, 3)), Rotation.from_rotvec(db.reshape(-1, 3))
    rel = (La.inv() * Lb).as_rotvec().reshape(n, -1, 3)
    loc = La * Rotation.from_rotvec((w[:, None, None] * rel).reshape(-1, 3))
    dof = rt.nearest_representative(torch.as_tensor(loc.as_rotvec().reshape(n, -1)), sk.lower, sk.upper).numpy()
    mid = (1 - w)[:, None] * hands_mid(sk, ra, qa, da) + w[:, None] * hands_mid(sk, rb, qb, db)
    root_pos = mid - hands_mid(sk, np.zeros_like(ra), root_rot, dof)
    for i in np.nonzero(w <= 0.0)[0]:
        root_pos[i], root_rot[i], dof[i] = ra[i], qa[i], da[i]
    for i in np.nonzero(w >= 1.0)[0]:
        root_pos[i], root_rot[i], dof[i] = rb[i], qb[i], db[i]
    return root_pos, root_rot, dof


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def release_v2() -> dict:
    """Release v2's record, folder, holds manifest (by stem) and node keys; the manifest and the graph description
    are checked against the record's sha256s, the x0 motions against its ``motions`` as they are read."""
    rec = json.loads(RELEASE_V2_RECORD.read_text())
    out = REPO / rec["dir"]
    for role in ("release_holds", "graph_description"):
        art = rec["artifacts"][role]
        if ids.sha256_file(REPO / art["path"]) != art["sha256"]:
            raise ValueError(f"{art['path']} is not release v2's {role} (sha256)")
    manifest = yaml.safe_load(open(REPO / rec["artifacts"]["release_holds"]["path"]))
    graph = json.load(open(REPO / rec["artifacts"]["graph_description"]["path"]))
    return {"record": rec, "dir": out, "manifest": manifest, "clips": {c["stem"]: c for c in manifest["clips"]},
            "node_keys": [n["key"] for n in graph["nodes"]], "pair_names": graph["pair_names"]}


def release_motion(rel: dict, stem: str) -> dict:
    path = rel["dir"] / "motions" / f"{stem}.motion"
    if ids.sha256_file(path) != rel["record"]["motions"][stem]:
        raise ValueError(f"{ids.display_path(path)} is not release v2's")
    return load(path)


def inputs() -> dict:
    """The pinned inputs: the selection, the admission, edges.json, release v2 (by its record) and every variant's
    files (each checked against the sha256 the selection gives). The D1 policy and the user's T6 decision are the
    pinned files' own copies (the admission's ``d1_policy``, the selection's ``rule.t6_decision``)."""
    sel = json.loads(SELECTED.read_text())
    adm = json.loads(ADMITTED.read_text())
    if sel["inputs"].get(ids.display_path(ADMITTED)) != ids.sha256_file(ADMITTED):
        raise ValueError("selected_physx.json was not made from today's admitted_physx.json")
    rows = {r["variant"]: r for r in adm["variants"]}
    variants = []
    for row in sel["variants"]:
        motion = REPO / row["motion"]
        if ids.sha256_file(motion) != row["sha256"] or rows[row["variant"]]["sha256"] != row["sha256"]:
            raise ValueError(f"{row['variant']}: the motion is not the selection's (sha256)")
        variants.append({"row": row, "admission": rows[row["variant"]], "motion": motion,
                         "record": REPO / row["record"], "run": REPO / row["source_run"] / "run.json",
                         "name": synthetic_name(row)})
    variants.sort(key=lambda v: order_key(v["name"]))
    return {"selected": sel, "admitted": adm, "edges": json.loads(EDGES.read_text()), "variants": variants,
            "release": release_v2(), "d1_policy": adm.get("d1_policy"), "t6_decision": sel["rule"]["t6_decision"]}


def builders() -> list[Path]:
    """The code whose output the clips are; their sha256s are part of the id: the writer and FK (``fit_writer``,
    ``retarget``, ``mosh_replay``, pose_lib), the transform (``sketch``), the de-penetration solve and the contract
    that triggers it (``export_physx``, ``quasistatic``, ``retarget_v2``, ``extract_contact_configs``'s zones)."""
    import extract_contact_configs
    from protomotions.components import pose_lib

    from edge_synthesis import export_physx, quasistatic, sketch
    from reference_curation import mosh_replay, retarget, retarget_v2

    return [Path(__file__), Path(fw.__file__), Path(retarget.__file__), Path(mosh_replay.__file__),
            Path(sketch.__file__), Path(pose_lib.__file__), Path(retarget_v2.__file__), Path(quasistatic.__file__),
            Path(export_physx.__file__), Path(extract_contact_configs.__file__)]


def parameters() -> dict:
    return {"fps": FPS, "plant": PLANT, "plant_sha256": plant_identity.sha256(PLANT),
            "names": "SYN_<edge>_<kind>_<timing>_s<seed>_<tag>", "kinds": EDGE_KIND,
            "s_blend": "settle frame j = blend(S exemplar, variant j, w = smoothstep((j + 1) / k_d))",
            "d_blend": "variant frame k = blend(variant k, D', w = smoothstep((k - k_a + 1) / (N - k_a + 1)))",
            "placement": "root placed so the L_Hand/R_Hand midpoint follows the blend of the two midpoints",
            "transform": "sketch.hand_anchor_transform(S exemplar, D exemplar): yaw about z + XY on the root",
            "windows": "S's window ends before the first settle/departure frame touching with an unlabelled zone, D's "
                       "starts after the last such frame of its blend (rigid_body_contacts; card: departure/arrival)",
            "contract": "per blend part against its own endpoints (settle: S exemplar + variant frame 0, braces of S; "
                        "hold at D: D exemplar + variant last frame, braces of D); box, floor, new overlaps",
            "start_all_cm": START_ALL_CM, "floor_cm": FLOOR_CM, "overlap_m": OVERLAP_M, "overlap_tol_m": OVERLAP_TOL_M,
            "box_deg": BOX_DEG, "pair_reach_m": PAIR_REACH_M,
            "depenetrate": "export_physx.DEPENETRATE, 40 iterations, PIN_FRAMES pinned",
            "pin_frames": PIN_FRAMES, "junction_cm": JUNCTION_CM, "acc_window": ACC_WINDOW,
            "acc_margin": ACC_MARGIN,
            "acc_rule": "per body, a junction's window may not exceed max(two sources' hold-window maxima) "
                        "+ acc_margin; gated twice: card = S's and D's hold windows in their clips on every "
                        "junction; parts = S, the variant's settle and hold at D, D, per junction (maxima over "
                        "interior stencils)",
            "real_pos_m": REAL_POS_M, "real_vel": REAL_VEL, "horizon_tie": HORIZON_TIE,
            "horizon_mag_tie": HORIZON_MAG_TIE, "near_repeat_m": NEAR_REPEAT_M, "velocity_drift": 0.02,
            "extension_dry_run_s": list(EXTENSION_S)}


def synthetic_id(inp: dict) -> str:
    """``synthetic_v3.<sha256_json(key)[:10]>`` over the card's key: the selection, the admission, release v2's
    record and edges.json by sha256, the builders' sources and the parameters (plus every variant's files)."""
    return f"{SPLICE_VERSION}.{ids.sha256_json(id_key(inp))[:10]}"


def id_key(inp: dict) -> dict:
    return {"schema": SCHEMA_VERSION, "version": SPLICE_VERSION,
            "inputs": {ids.display_path(p): ids.sha256_file(p) for p in (SELECTED, ADMITTED, RELEASE_V2_RECORD, EDGES)},
            "variants": {v["name"]: {"motion": ids.sha256_file(v["motion"]), "record": ids.sha256_file(v["record"]),
                                     "run": ids.sha256_file(v["run"])} for v in inp["variants"]},
            "builders": {ids.display_path(p): ids.sha256_file(p) for p in builders()},
            "parameters": parameters()}


# --------------------------------------------------------------------------- #
# One clip
# --------------------------------------------------------------------------- #
def endpoint(rel: dict, ep: dict) -> tuple[dict, dict]:
    """``(clip, hold)`` of an edges.json endpoint in release v2's manifest (its exemplar frame checked)."""
    clip = rel["clips"][ep["stem"]]
    hold = next(h for h in clip["holds"] if h["hold_id"] == ep["hold_id"])
    if int(hold["frame_hold"]) != int(ep["frame_hold"]) or hold["name"] != ep["name"]:
        raise ValueError(f"{ep['hold_id']}: edges.json's exemplar {ep['frame_hold']} is not the release's")
    return clip, hold


def splice(sk, v: dict, inp: dict, cache: dict) -> dict:
    """Build one clip: ``{name, motion, holds, lineage, info}`` (nothing written)."""
    from edge_synthesis import sketch as SK

    rel, row, name = inp["release"], v["row"], v["name"]
    e = SK.edge(inp["edges"], row["edge"])
    s_clip, s_hold = endpoint(rel, e["source"])
    d_clip, d_hold = endpoint(rel, e["destination"])
    for stem in (s_clip["stem"], d_clip["stem"]):
        if stem not in cache:
            cache[stem] = release_motion(rel, stem)
    Sm, Dm = cache[s_clip["stem"]], cache[d_clip["stem"]]
    sfh, dfh = int(s_hold["frame_hold"]), int(d_hold["frame_hold"])
    var = load(v["motion"])
    rec = json.loads(v["record"].read_text())
    run = json.loads(v["run"].read_text())
    N = int(var["dof_pos"].shape[0])
    if rec["frames"] != N or int(var["fps"]) != FPS or rec["fps"] != FPS:
        raise ValueError(f"{name}: {N} frames at {var['fps']} fps, the export says {rec['frames']} at {rec['fps']}")
    clock = variant_clock(N, rec["t_first"], run["T"], run["settle_s"], run["hold_s"])
    k_d, k_a = clock["k_d"], clock["k_a"]
    # the variant is in S's clip frame: frame 0 within START_ALL_CM of S's exemplar, every body
    vpos = var["rigid_body_pos"].double().numpy()
    s_pos = Sm["rigid_body_pos"][sfh].double().numpy()
    start_cm = 100 * np.linalg.norm(vpos[0] - s_pos, axis=-1)
    if start_cm.max() > START_ALL_CM:
        raise ValueError(f"{name}: variant frame 0 is {start_cm.max():.2f} cm from S's exemplar "
                         f"({NAMES[int(start_cm.argmax())]}): not in S's clip frame; stopping")
    # D' and the lead-out: the generator's hand-anchor transform
    d_pos = Dm["rigid_body_pos"][dfh].double().numpy()
    yaw, t_xy = SK.hand_anchor_transform(s_pos, d_pos, BI)
    earlier = [h for h in s_clip["holds"] if h["frame_hold"] < sfh]
    lead_in_start = max(earlier, key=lambda h: h["frame_hold"])["frame_start"]
    later = [h for h in d_clip["holds"] if h["frame_hold"] > dfh]
    lead_out_end = min(later, key=lambda h: h["frame_hold"])["frame_end"] if later else int(d_clip["num_frames"]) - 1
    L = sfh - lead_in_start
    n_out = lead_out_end - dfh + 1
    lay = layout(L, clock, n_out)
    T = lay["num_frames"]
    # coordinates
    root_pos, root_rot, dof = np.empty((T, 3)), np.empty((T, 4)), np.empty((T, 69))
    src = np.full(T, -1, int)
    src_frame = np.full(T, -1, int)
    kind = np.zeros(T, int)
    seg = np.zeros(T, int)
    blend_w = np.zeros(T)
    blend_src = np.full(T, -1, int)
    blend_frame = np.full(T, -1, int)
    yaw_f = np.zeros(T)
    xy_f = np.zeros((T, 2))
    variant_t = np.full(T, np.nan)
    stems = [s_clip["stem"], d_clip["stem"], row["variant"]]
    S_, D_, V_ = 0, 1, 2
    segs = []

    def segment(nm, a, b, source, kd, y=0.0, xy=(0.0, 0.0)):
        segs.append({"name": nm, "start": int(a), "end": int(b), "source": source, "kind": kd, "yaw": float(y),
                     "xy": [float(xy[0]), float(xy[1])]})
        seg[a:b + 1] = len(segs) - 1

    # lead-in and S's exemplar: real, untransformed
    idx = np.arange(lead_in_start, sfh + 1)
    root_pos[:L + 1], root_rot[:L + 1], dof[:L + 1] = frames_of(Sm, idx)
    src[:L + 1], src_frame[:L + 1] = S_, idx
    segment("lead_in", 0, L - 1, S_, KIND["real"])
    segment("s_exemplar", L, L, S_, KIND["real"])
    # settle: S's exemplar -> variant frames 0 .. k_d - 1
    Sx = frames_of(Sm, [sfh])
    a = [np.repeat(x, k_d, 0) for x in Sx]
    b = frames_of(var, np.arange(k_d))
    ws = s_weights(k_d)
    sl = slice(L + 1, L + 1 + k_d)
    root_pos[sl], root_rot[sl], dof[sl] = blend(sk, a, b, ws)
    src[sl], src_frame[sl], kind[sl] = V_, np.arange(k_d), KIND["blend"]
    blend_w[sl], blend_src[sl], blend_frame[sl] = 1 - ws, S_, sfh
    segment("settle", L + 1, L + k_d, V_, KIND["blend"])
    # transition: the variant's own frames
    sl = slice(L + 1 + k_d, L + 1 + k_a)
    root_pos[sl], root_rot[sl], dof[sl] = frames_of(var, np.arange(k_d, k_a))
    src[sl], src_frame[sl], kind[sl] = V_, np.arange(k_d, k_a), KIND["synthetic"]
    segment("transition", L + 1 + k_d, L + k_a, V_, KIND["synthetic"])
    # hold at D: variant frames k_a .. N - 1 -> D'
    Dx = frames_of(Dm, [dfh])
    dp_root, dq_root = place(Dx[0], Dx[1], yaw, t_xy)
    Dp = (dp_root, dq_root, Dx[2])
    wd = d_weights(k_a, N)
    b = [np.repeat(x, N - k_a, 0) for x in Dp]
    a = frames_of(var, np.arange(k_a, N))
    sl = slice(L + 1 + k_a, L + 1 + N)
    root_pos[sl], root_rot[sl], dof[sl] = blend(sk, a, b, wd)
    src[sl], src_frame[sl], kind[sl] = V_, np.arange(k_a, N), KIND["blend"]
    blend_w[sl], blend_src[sl], blend_frame[sl] = wd, D_, dfh
    yaw_f[sl], xy_f[sl] = yaw, t_xy
    segment("d_hold", L + 1 + k_a, L + N, V_, KIND["blend"], yaw, t_xy)
    variant_t[L + 1:L + 1 + N] = rec["t_first"] + np.arange(N) / FPS
    # lead-out: D's clip under the transform
    idx = np.arange(dfh, lead_out_end + 1)
    p, q, d = frames_of(Dm, idx)
    sl = slice(L + 1 + N, T)
    root_pos[sl], root_rot[sl] = place(p, q, yaw, t_xy)
    dof[sl] = d
    src[sl], src_frame[sl] = D_, idx
    yaw_f[sl], xy_f[sl] = yaw, t_xy
    segment("lead_out", L + 1 + N, T - 1, D_, KIND["real"], yaw, t_xy)
    # the blend frames under certify's contract; a failing blend part is de-penetrated (export_physx's solve)
    depen = {}
    edited = np.zeros(T, bool)
    for part, (f0, f1), ground, hold in (("settle", lay["settle"], e["source"]["ground"], s_hold),
                                         ("d_hold", lay["d_hold"], e["destination"]["ground"], d_hold)):
        rules = part_rules(sk, part, Sm, sfh, Dm, dfh, var, s_hold, d_hold)
        fr = np.arange(f0, f1 + 1)
        P, Rm = fk(sk, root_pos[fr], root_rot[fr], dof[fr])
        before = contract(sk, torch.as_tensor(P), torch.as_tensor(Rm), dof[fr], rules)
        if not before["failing"]:
            continue
        braces = [x for x in hold.get("pairs_configured", []) if not x.endswith(":G")]
        rp, rq, dd, rep = depenetrate(root_pos, root_rot, dof, (f0, f1), list(ground), braces, f"{name}_{part}")
        P2, Rm2 = fk(sk, rp, rq, dd)
        after = contract(sk, torch.as_tensor(P2), torch.as_tensor(Rm2), dd, rules)
        edit = 100 * np.linalg.norm(P2 - P, axis=-1)
        i, j = np.unravel_index(int(edit.argmax()), edit.shape)
        root_pos[fr], root_rot[fr], dof[fr] = rp, rq, dd
        edited[fr] = True
        depen[part] = {"frames": [int(f0), int(f1)], "braces": braces, "ground": list(ground),
                       "failing_before": [int(fr[x]) for x in before["failing"]],
                       "new_before": [[int(fr[x[0]])] + x[1:] for x in before["new"]],
                       "floor_min_cm_before": round(float(before["floor_cm"].min()), 3),
                       "box_max_deg_before": round(float(before["box_deg"].max()), 3),
                       "failing_after": [int(fr[x]) for x in after["failing"]],
                       "floor_min_cm_after": round(float(after["floor_cm"].min()), 3),
                       "box_max_deg_after": round(float(after["box_deg"].max()), 3),
                       "edit_max_cm": round(float(edit.max()), 3), "edit_p50_cm": round(float(np.median(edit)), 4),
                       "edit_worst": [int(fr[i]), NAMES[j]],
                       "edit_first_last_frame_cm": [round(float(edit[0].max()), 4), round(float(edit[-1].max()), 4)],
                       "edit_failing_frames_max_cm": round(float(edit[before["failing"]].max()), 3),
                       "solver": rep}
    # the motion
    mot = fw.write_motion(root_pos, root_rot, dof, FPS, PLANT)
    mot["ground_reaction"] = torch.zeros(T, 3)
    mot["rigid_body_ground_forces"] = torch.zeros(T, 24, 3)
    mot["ground_reaction_valid"] = torch.zeros(T, 3)
    # the holds: S's and D's windows hold their labels (content_windows); unplanned contacts on transition frames
    # (no phase of the edge, nor S or D, puts that zone on the floor) are the variant's own, recorded
    zc = zone_contacts(mot["rigid_body_contacts"].numpy())
    win = content_windows(zc, lay, ground_zones(s_hold), ground_zones(d_hold))
    holds = splice_holds(name, s_clip, s_hold, d_clip, d_hold, lay, int(lead_in_start), int(lead_out_end),
                         s_end=win["s_end"], d_start=win["d_start"])
    planned = set(ground_zones(s_hold)) | set(ground_zones(d_hold)) | {z for p in e["phases"] for z in p["ground"]}
    tr = unlabelled(zc, np.arange(lay["transition"][0], lay["transition"][1] + 1), planned)
    transition_contacts = {z: {"frames": len(f), "first": f[0], "last": f[-1],
                               "planned_free": any(z in p.get("free", []) for p in e["phases"])} for z, f in tr.items()}
    lineage = {"stems": np.array(stems), "source": src.astype(np.int16), "source_frame": src_frame.astype(np.int32),
               "kind": kind.astype(np.int8), "segment": seg.astype(np.int16), "blend_w": blend_w,
               "blend_source": blend_src.astype(np.int16), "blend_source_frame": blend_frame.astype(np.int32),
               "yaw": yaw_f, "xy": xy_f, "variant_t": variant_t, "depenetrated": edited,
               "seg_name": np.array([s["name"] for s in segs]), "seg_start": np.array([s["start"] for s in segs]),
               "seg_end": np.array([s["end"] for s in segs]), "seg_source": np.array([s["source"] for s in segs]),
               "seg_kind": np.array([s["kind"] for s in segs]), "seg_yaw": np.array([s["yaw"] for s in segs]),
               "seg_xy": np.array([s["xy"] for s in segs])}
    info = {"variant": row["variant"], "edge": row["edge"], "label": row["label"], "recipe": row["recipe"],
            "clock": clock, "T": run["T"], "durations_s": run["durations_s"], "settle_s": run["settle_s"],
            "hold_s": run["hold_s"], "t_first": rec["t_first"], "layout": lay,
            "lead_in_source": [int(lead_in_start), sfh], "lead_out_source": [dfh, int(lead_out_end)],
            "S": {"stem": s_clip["stem"], "hold_id": s_hold["hold_id"], "frame_hold": sfh},
            "D": {"stem": d_clip["stem"], "hold_id": d_hold["hold_id"], "frame_hold": dfh},
            "transform": {"yaw_rad": float(yaw), "yaw_deg": float(np.degrees(yaw)), "t_xy": [float(x) for x in t_xy],
                          "run_anchor_yaw_deg": run["anchor_yaw_deg"]},
            "start_offset_cm": {"all_max": round(float(start_cm.max()), 3), "worst": NAMES[int(start_cm.argmax())]},
            "depenetration": depen, "windows": win, "transition_contacts": transition_contacts}
    return {"name": name, "motion": mot, "holds": holds, "lineage": lineage, "info": info,
            "sources": {"S": Sm, "D": Dm, "V": var}}


def clip_entry(c: dict, v: dict, inp: dict, motion_path: Path) -> dict:
    """The clip's manifest entry (``<stem>.holds.yaml``): the fields the builders read, the holds and the
    ``synthetic`` block (the variant, its admission row, the D1 policy and the user's T6 decision). ``source`` is
    absolute, as in release v2's manifest (``release_v2.verify_sources`` opens ``Path(c["source"])``)."""
    T = int(c["motion"]["dof_pos"].shape[0])
    i = c["info"]
    return {"stem": c["name"], "group": "edge", "family": f"SYN_{i['edge']}", "source": str(motion_path.resolve()),
            "fps": FPS, "num_frames": T, "length_s": round(T / FPS, 3), "capture_available": False,
            "synthetic": {
                "variant": i["variant"], "edge": i["edge"], "label": i["label"],
                "motion": ids.display_path(v["motion"]), "sha256": ids.sha256_file(v["motion"]),
                "record": ids.display_path(v["record"]), "record_sha256": ids.sha256_file(v["record"]),
                "source_run": v["row"]["source_run"], "run_sha256": ids.sha256_file(v["run"]),
                "recipe": v["row"]["recipe"], "T": i["T"], "durations_s": i["durations_s"], "clock": i["clock"],
                "S": i["S"], "D": i["D"], "transform": i["transform"],
                "admission": {"record": ids.display_path(ADMITTED), "sha256": ids.sha256_file(ADMITTED),
                              "row": v["admission"]},
                "selection": {"record": ids.display_path(SELECTED), "sha256": ids.sha256_file(SELECTED)},
                "d1_policy": inp["d1_policy"], "t6_decision": inp["t6_decision"]},
            "holds": c["holds"]}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def acc(pos: np.ndarray) -> np.ndarray:
    """``[T, B]`` admit's acceleration metric: |p[f+1] - 2 p[f] + p[f-1]| fps^2 (0 on the end frames)."""
    out = np.zeros(pos.shape[:2])
    out[1:-1] = np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1) * FPS ** 2
    return out


def window_acc_max(pos: np.ndarray, windows, interior: bool = True) -> np.ndarray:
    """``[B]`` the largest acceleration inside the windows ``[(a, b)]`` (inclusive): interior stencils only, or
    (``interior=False``) every frame of the window with its neighbours in the clip."""
    best = np.zeros(pos.shape[1])
    for a, b in windows:
        if interior and b - a >= 2:
            best = np.maximum(best, acc(pos[a:b + 1])[1:-1].max(0))
        elif not interior:
            best = np.maximum(best, acc(pos)[max(a, 1):min(b, len(pos) - 2) + 1].max(0))
    return best


def planted_bodies(zones) -> list[int]:
    return [BI[b] for z in zones for b in ZONES[z]]


def check_real_frames(mot: dict, lin: dict, sources: dict) -> dict:
    """Real frames against their sources after the recorded transform: positions (``REAL_POS_M``), rotations,
    ``dof_pos`` exactly, contacts, and the velocities (``REAL_VEL``) where pose_lib's forward difference over
    f .. f + 3 reads the source's own consecutive frames (elsewhere a seam is within 3 frames)."""
    stems = [str(s) for s in lin["stems"]]
    pos, rot = mot["rigid_body_pos"].double().numpy(), mot["rigid_body_rot"].double().numpy()
    T = pos.shape[0]
    real = np.nonzero(lin["kind"] == KIND["real"])[0]
    h = fw.VELOCITY_MAX_HORIZON
    worst = {"pos_m": 0.0, "rot_deg": 0.0, "vel": 0.0, "ang_vel": 0.0, "ang_vel_norm": 0.0, "dof_vel": 0.0,
             "local_rot": 0.0, "tie_magnitude_gap": 0.0}
    dof_exact, contacts_equal, compared, excluded, ties, unexplained = True, True, 0, 0, 0, []
    for s in np.unique(lin["source"][real]):
        f = real[lin["source"][real] == s]
        m = sources[stems[s]]
        sf = lin["source_frame"][f]
        yaw, xy = float(lin["yaw"][f[0]]), lin["xy"][f[0]]
        if not (np.allclose(lin["yaw"][f], yaw) and np.allclose(lin["xy"][f], xy)):
            raise ValueError("one source's real frames carry two transforms")
        p, q = place(m["rigid_body_pos"][sf].double().numpy(), m["rigid_body_rot"][sf].double().numpy(), yaw, xy)
        worst["pos_m"] = max(worst["pos_m"], float(np.abs(pos[f] - p).max()))
        dq = (Rotation.from_quat(rot[f].reshape(-1, 4)).inv() * Rotation.from_quat(q.reshape(-1, 4))).magnitude()
        worst["rot_deg"] = max(worst["rot_deg"], float(np.degrees(dq.max())))
        dof_exact &= bool(torch.equal(mot["dof_pos"][f], m["dof_pos"][sf]))
        contacts_equal &= bool(torch.equal(mot["rigid_body_contacts"][f], m["rigid_body_contacts"][sf]))
        worst["local_rot"] = max(worst["local_rot"], float((mot["local_rigid_body_rot"][f, 1:].double()
                                                            - m["local_rigid_body_rot"][sf, 1:].double()).abs().max()))
        ns = int(m["rigid_body_pos"].shape[0])
        ok = []
        for i, sfi in zip(f, sf):
            run_ok = all(i + k < T and lin["kind"][i + k] == KIND["real"] and lin["source"][i + k] == s
                         and lin["source_frame"][i + k] == sfi + k for k in range(1, h + 1) if sfi + k < ns)
            ends_together = all((i + k >= T) == (sfi + k >= ns) for k in range(1, h + 1))
            ok.append(run_ok and ends_together)
        ok = np.array(ok, bool)
        compared += int(ok.sum())
        excluded += int((~ok).sum())
        if ok.any():
            fo, so = f[ok], sf[ok]
            for key, field, rotate in (("vel", "rigid_body_vel", True), ("ang_vel", "rigid_body_ang_vel", True),
                                       ("dof_vel", "dof_vel", False)):
                ref = m[field][so].double().numpy()
                ref = rotate_vectors(ref, yaw) if rotate else ref
                diff = np.abs(mot[field][fo].double().numpy() - ref)
                worst[key] = max(worst[key], float(diff.max()))
                if key == "ang_vel":
                    worst["ang_vel_norm"] = max(worst["ang_vel_norm"], float(np.linalg.norm(
                        mot[field][fo].double().numpy() - ref, axis=-1).max()))
                    # an entry off by more is a tie: it equals another horizon's candidate whose magnitude is
                    # within HORIZON_MAG_TIE of the smallest (pose_lib's argmin flips under float32 round-off)
                    for i, b in zip(*np.nonzero(diff.max(-1) > REAL_VEL)):
                        cand = horizon_candidates(m["rigid_body_rot"][:, b].double().numpy(), int(so[i]), yaw)
                        got = mot[field][fo[i], b].double().numpy()
                        least = min(np.linalg.norm(x) for x in cand)
                        gaps = [np.linalg.norm(x) - least for x in cand if np.abs(got - x).max() <= HORIZON_TIE]
                        if gaps and min(gaps) <= HORIZON_MAG_TIE:
                            ties += 1
                            worst["tie_magnitude_gap"] = max(worst["tie_magnitude_gap"], float(min(gaps)))
                        else:
                            unexplained.append([int(fo[i]), NAMES[b], round(float(diff[i, b].max()), 4)])
    out = {k: (v if k in ("pos_m", "tie_magnitude_gap") else round(v, 6)) for k, v in worst.items()}
    out.update(dof_exact=dof_exact, contacts_equal=contacts_equal, real_frames=int(len(real)),
               velocity_compared=compared, velocity_excluded_near_seams=excluded, ang_vel_horizon_ties=ties,
               ang_vel_unexplained=unexplained)
    out["pass"] = (worst["pos_m"] <= REAL_POS_M and dof_exact and worst["vel"] <= REAL_VEL
                   and worst["dof_vel"] <= REAL_VEL and not unexplained)
    return out


def horizon_candidates(quat_xyzw: np.ndarray, f: int, yaw: float) -> list[np.ndarray]:
    """pose_lib's angular-velocity candidates of one body at frame ``f`` (horizons 1 .. 3: rotvec(q[f+h] q[f]^-1)
    fps / h), placed by ``yaw``. The stored value is the smallest-magnitude candidate, so where two nearly tie
    (|h1| and |h2| within ~1e-6 rad/s) float32 round-off picks either one."""
    out = []
    for h in range(1, fw.VELOCITY_MAX_HORIZON + 1):
        if f + h < len(quat_xyzw):
            w = (Rotation.from_quat(quat_xyzw[f + h]) * Rotation.from_quat(quat_xyzw[f]).inv()).as_rotvec() * FPS / h
            out.append(rotate_vectors(w, yaw))
    return out


def check_synthetic_frames(mot: dict, lin: dict, var: dict) -> dict:
    """Transition frames are the variant's own: positions to ``REAL_POS_M``, ``dof_pos`` exactly."""
    f = np.nonzero(lin["kind"] == KIND["synthetic"])[0]
    sf = lin["source_frame"][f]
    d = float((mot["rigid_body_pos"][f].double() - var["rigid_body_pos"][sf].double()).abs().max())
    exact = bool(torch.equal(mot["dof_pos"][f], var["dof_pos"][sf]))
    return {"frames": int(len(f)), "pos_m": d, "dof_exact": exact, "pass": d <= REAL_POS_M and exact}


def check_seams(c: dict) -> dict:
    """The junctions (planted bodies), the seam offsets before and after blending, the D blend's planted-body
    travel, the seam-window acceleration rule and repeated frames (module doc; PLAN.MD R2's acceptance)."""
    mot, info = c["motion"], c["info"]
    lay, clock = info["layout"], info["clock"]
    k_d, k_a, N = clock["k_d"], clock["k_a"], clock["N"]
    pos = mot["rigid_body_pos"].double().numpy()
    Sm, Dm, var = c["sources"]["S"], c["sources"]["D"], c["sources"]["V"]
    e_src, e_dst = c["edge"]["source"], c["edge"]["destination"]
    s_pl, d_pl = planted_bodies(e_src["ground"]), planted_bodies(e_dst["ground"])
    vpos = var["rigid_body_pos"].double().numpy()
    sfh, dfh = info["S"]["frame_hold"], info["D"]["frame_hold"]
    s_ex = Sm["rigid_body_pos"][sfh].double().numpy()
    yaw, t_xy = info["transform"]["yaw_rad"], info["transform"]["t_xy"]
    d_ex, _ = place(Dm["rigid_body_pos"][dfh].double().numpy(), Dm["rigid_body_rot"][dfh].double().numpy(), yaw, t_xy)
    out = {}
    # junctions
    jn = {}
    for nm, (a, b) in lay["junctions"].items():
        bodies = d_pl if nm == "d_hold|lead_out" else s_pl
        dd = 100 * np.linalg.norm(pos[b] - pos[a], axis=-1)
        jn[nm] = {"frames": [a, b], "planted_cm": round(float(dd[bodies].max()), 4),
                  "planted_worst": NAMES[bodies[int(dd[bodies].argmax())]], "all_cm": round(float(dd.max()), 4),
                  "pass": float(dd[bodies].max()) <= JUNCTION_CM}
    out["junctions"] = jn
    # seam offsets before and after blending
    def off(p, q, bodies):
        dd = 100 * np.linalg.norm(p - q, axis=-1)
        return {"all_cm": round(float(dd.max()), 3), "all_worst": NAMES[int(dd.argmax())],
                "planted_cm": round(float(dd[bodies].max()), 3),
                "planted_worst": NAMES[bodies[int(dd[bodies].argmax())]]}
    L = lay["s_exemplar"]
    out["seam_offsets"] = {
        "S": {"before": off(vpos[0], s_ex, s_pl), "after": off(pos[L + 1], pos[L], s_pl)},
        "D": {"before": off(vpos[N - 1], d_ex, d_pl), "after": off(pos[L + N], pos[L + 1 + N], d_pl)}}
    # the D blend's planted-body travel (horizontal; recorded, not gated: the user accepted it)
    a0, a1 = lay["arrival"], lay["lead_out"][0]
    trav = {}
    for b in d_pl:
        xy = pos[a0:a1 + 1, b, :2]
        step = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
        own = np.linalg.norm(np.diff(vpos[k_a:N, b, :2], axis=0), axis=-1)
        trav[NAMES[b]] = {"path_cm": round(100 * float(step.sum()), 3),
                          "net_cm": round(100 * float(np.linalg.norm(xy[-1] - xy[0])), 3),
                          "variant_own_path_cm": round(100 * float(own.sum()), 3),
                          "offset_carried_cm": round(100 * float(np.linalg.norm(d_ex[b, :2] - vpos[N - 1, b, :2])), 3),
                          "peak_cm_s": round(100 * FPS * float(step.max()), 2)}
    out["d_blend_planted_travel"] = trav
    # the seam-window acceleration rule, read twice and both gated: "card" takes the human sources (S's and D's
    # hold windows in their clips, the card's "7.9-8.4 m/s^2") on every junction; "parts" takes each junction's own
    # two parts (S, the variant's settle and hold at D "V", D). Maxima over interior stencils (the stricter);
    # "full" (the window's end frames' stencils too, which reach one frame outside) is recorded beside them.
    Spos, Dpos = Sm["rigid_body_pos"].double().numpy(), Dm["rigid_body_pos"].double().numpy()
    s_win = [(c["s_hold"]["frame_start"], c["s_hold"]["frame_end"])]
    d_win = [(c["d_hold"]["frame_start"], c["d_hold"]["frame_end"])]
    hold_max = {"S": window_acc_max(Spos, s_win), "D": window_acc_max(Dpos, d_win),
                "V": window_acc_max(vpos, [(0, k_d - 1), (k_a, N - 1)])}
    full_max = {"S": window_acc_max(Spos, s_win, interior=False), "D": window_acc_max(Dpos, d_win, interior=False)}
    sources_of = {"card": {nm: ("S", "D") for nm in lay["junctions"]},
                  "parts": {"lead_in|s_exemplar": ("S", "S"), "s_exemplar|settle": ("S", "V"),
                            "settle|transition": ("V", "V"), "d_hold|lead_out": ("V", "D")}}
    A = acc(pos)
    own = own_acc(c)
    out["acc_rule"] = {"hold_window_max": {k: round(float(v.max()), 3) for k, v in hold_max.items()},
                       "hold_window_max_full": {k: round(float(v.max()), 3) for k, v in full_max.items()}}
    for reading, srcs in sources_of.items():
        rule = {}
        for nm, (a, b) in lay["junctions"].items():
            s1, s2 = srcs[nm]
            thr = np.maximum(hold_max[s1], hold_max[s2]) + ACC_MARGIN
            fr = np.arange(max(a - ACC_WINDOW, 1), min(b + ACC_WINDOW, len(pos) - 2) + 1)
            over = A[fr] - thr[None]
            i, j = np.unravel_index(int(np.argmax(over)), over.shape)
            rule[nm] = {"sources": [s1, s2], "window": [int(fr[0]), int(fr[-1])],
                        "acc_max": round(float(A[fr].max()), 3), "worst_body": NAMES[int(np.argmax(A[fr].max(0)))],
                        "threshold_at_worst": round(float(thr[int(np.argmax(A[fr].max(0)))]), 3),
                        "worst_excess": round(float(over.max()), 3), "worst_excess_frame": int(fr[i]),
                        "worst_excess_body": NAMES[j], "own_acc_max": round(float(own[fr].max()), 3),
                        "pass": float(over.max()) <= 0.0}
        out["acc_rule"][reading] = {"junctions": rule, "pass": all(r["pass"] for r in rule.values())}
    out["acc_rule"]["pass"] = out["acc_rule"]["card"]["pass"] and out["acc_rule"]["parts"]["pass"]
    # repeated frames: no two consecutive spliced frames identical, anywhere
    dpos = np.abs(np.diff(pos, axis=0)).max(axis=(1, 2))
    ddof = np.abs(np.diff(mot["dof_pos"].double().numpy(), axis=0)).max(1)
    same = np.nonzero((dpos == 0) & (ddof == 0))[0]
    seam_frames = sorted({f for a, b in lay["junctions"].values() for f in (a, b)})
    seam_min = {nm: float(max(dpos[a], ddof[a])) for nm, (a, b) in lay["junctions"].items()}
    out["repeated_frames"] = {"identical_pairs": [[int(i), int(i + 1)] for i in same],
                              "seam_min_step": seam_min, "seam_frames": seam_frames,
                              "s_exemplar_to_settle_cm": round(100 * float(np.linalg.norm(
                                  pos[L + 1] - pos[L], axis=-1).max()), 6),
                              "d_hold_last_to_lead_out_cm": round(100 * float(np.linalg.norm(
                                  pos[L + N] - pos[L + 1 + N], axis=-1).max()), 6),
                              "pass": len(same) == 0}
    # representative flips: no exp-map coordinate jumps by more than pi between frames
    jumps = np.abs(np.diff(mot["dof_pos"].double().numpy(), axis=0))
    out["dof_step_max_rad"] = round(float(jumps.max()), 4)
    out["representative_flips"] = int((jumps > math.pi).sum())
    out["pass"] = (all(j["pass"] for j in jn.values()) and out["acc_rule"]["pass"] and out["repeated_frames"]["pass"]
                   and out["representative_flips"] == 0)
    return out


def own_acc(c: dict) -> np.ndarray:
    """``[T, B]`` every spliced frame's acceleration in its own source (real: the source clip's, placed; variant
    and blend frames: the variant's at that frame): what the frame accelerates without the splice."""
    lin = c["lineage"]
    T = len(lin["kind"])
    out = np.zeros((T, 24))
    srcs = [c["sources"]["S"], c["sources"]["D"], c["sources"]["V"]]
    cache = {}
    for i in range(T):
        s, f = int(lin["source"][i]), int(lin["source_frame"][i])
        if s not in cache:
            cache[s] = acc(srcs[s]["rigid_body_pos"].double().numpy())
        out[i] = cache[s][f]
    return out


def contract_rules(sk, endpoints: list[tuple[dict, int]], hold: dict) -> dict:
    """What a blend part's frames may overlap: the body pairs already deeper than ``OVERLAP_M`` in the part's
    endpoints (``[(motion, frame)]``: the exemplar the part blends with first, then the variant's frame at the
    part's far end), and the braces of the part's hold (its configured body-body pairs) whose zones the exemplar
    holds within ``PAIR_REACH_M``."""
    from protomotions.utils.rotations import quaternion_to_matrix

    from reference_curation import retarget as rt

    within = -OVERLAP_M - OVERLAP_TOL_M
    allowed, braces = set(), {}
    for m, fr in endpoints:
        p = m["rigid_body_pos"][fr:fr + 1].double()
        r = quaternion_to_matrix(m["rigid_body_rot"][fr:fr + 1].double(), w_last=True)
        _, ea, eb, _ = rt.near_body_pairs(sk, p, r, within)
        allowed |= {(int(x), int(y)) for x, y in zip(ea, eb)}
    m, fr = endpoints[0]
    p = m["rigid_body_pos"][fr:fr + 1].double()
    r = quaternion_to_matrix(m["rigid_body_rot"][fr:fr + 1].double(), w_last=True)
    for pair in hold.get("pairs_configured", []):
        if not pair.endswith(":G"):
            g, _, _ = rt.zone_pair_gaps(sk, p, r, *pair.split("+"))
            braces[pair] = round(100 * float(g.min()), 3)
    exempt = {tuple(sorted(pr.split("+"))) for pr, g in braces.items() if g / 100 <= PAIR_REACH_M}
    return {"allowed": allowed, "exempt": exempt, "braces_gap_cm": braces}


def part_rules(sk, part: str, Sm: dict, sfh: int, Dm: dict, dfh: int, var: dict, s_hold: dict, d_hold: dict) -> dict:
    """``contract_rules`` of one blend part: the settle against S's exemplar and the variant's first frame with S's
    braces, the hold at D against D's exemplar and the variant's last frame with D's braces (certify's endpoints and
    closed braces, part by part)."""
    if part == "settle":
        return contract_rules(sk, [(Sm, sfh), (var, 0)], s_hold)
    return contract_rules(sk, [(Dm, dfh), (var, int(var["dof_pos"].shape[0]) - 1)], d_hold)


def contract(sk, P: torch.Tensor, R: torch.Tensor, dof, rules: dict) -> dict:
    """``quasistatic.certify``'s box, floor and overlap rules on poses ``P [n,24,3]``, ``R [n,24,3,3]`` (float64)
    and their joint coordinates ``dof [n,69]``: ``box_deg [n]`` (each frame's largest excess past the joint box),
    ``floor_cm [n]`` (each frame's lowest collider point), ``new`` ([local frame, body, body, gap cm] of every pair
    deeper than ``OVERLAP_M`` that ``rules`` do not allow), ``brace_depth_cm`` and ``failing`` (local frames)."""
    from reference_curation import retarget as rt

    dof = np.asarray(dof, float).reshape(-1, 69)
    lo, hi = sk.lower.numpy(), sk.upper.numpy()
    box = np.degrees(np.maximum(lo - dof, 0) + np.maximum(dof - hi, 0)).max(1)
    with torch.no_grad():
        hgt = rt.candidate_heights(sk, rt.candidate_points(sk, P, R)).numpy()
    nf, na, nb, gap = rt.near_body_pairs(sk, P, R, -OVERLAP_M - OVERLAP_TOL_M)
    zone_of = {BI[b]: z for z, bs in ZONES.items() for b in bs}
    new, brace_depth = [], {}
    for fi, a, b, g in zip(nf, na, nb, gap):
        key = tuple(sorted((zone_of[int(a)], zone_of[int(b)])))
        if key in rules["exempt"]:
            nm = "+".join(key)
            brace_depth[nm] = min(brace_depth.get(nm, 0.0), round(100 * float(g), 3))
        elif (int(a), int(b)) not in rules["allowed"]:
            new.append([int(fi), NAMES[int(a)], NAMES[int(b)], round(100 * float(g), 3)])
    floor_cm = 100 * hgt.min(1)
    failing = sorted({x[0] for x in new} | {int(i) for i in np.nonzero(floor_cm < FLOOR_CM)[0]}
                     | {int(i) for i in np.nonzero(box > BOX_DEG)[0]})
    return {"box_deg": box, "floor_cm": floor_cm, "new": new, "brace_depth_cm": brace_depth, "failing": failing}


def _install_freeze():
    """Wrap ``retarget_v2.bounds`` in this process so a problem carrying ``plan["splice_v3_freeze"]`` has those
    frames bounds-pinned (lo = hi); every other problem is unchanged. ``retarget_v2.py`` stays untouched (its source
    hash is part of the retarget record), as ``quasistatic`` and ``exact`` wrap it."""
    from reference_curation import retarget_v2 as rv2

    if getattr(rv2.bounds, "_splice_v3", False):
        return
    orig = rv2.bounds

    def bounds(prob):
        lo, hi = orig(prob)
        fz = prob.plan.get("splice_v3_freeze")
        if fz is None:
            return lo, hi
        m = torch.as_tensor(fz["mask"])
        lo, hi = lo.clone(), hi.clone()
        lo[m], hi[m] = fz["x"][m], fz["x"][m]
        return lo, hi

    bounds._splice_v3 = True
    rv2.bounds = bounds


def depenetrate(root_pos, root_rot, dof, part: tuple[int, int], ground: list, braces: list, name: str,
                iters: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """``export_physx``'s de-penetration solve on the blend frames ``part`` (inclusive) of a clip's coordinates
    (``[T, 3]``, ``[T, 4]`` xyzw, ``[T, 69]``): ``quasistatic.build_problem`` under the given planted zones and closed
    braces, then ``retarget_v2.solve`` with ``export_physx.DEPENETRATE`` (only the floor, the 253-pair guard and the
    stay-near terms). The problem also holds ``PIN_FRAMES`` neighbouring frames either side, bounds-pinned, so the
    edit tapers to nothing at the real or synthetic frames next to the blend. Returns the part's ``(root_pos,
    root_rot xyzw, dof, solver report)``."""
    from extract_contact_configs import ZONE_ORDER

    from edge_synthesis import export_physx as XP
    from edge_synthesis import quasistatic as Q
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    a, b = part
    fr = np.arange(a - PIN_FRAMES, b + PIN_FRAMES + 1)
    n = len(fr)
    gr = np.zeros((n, len(ZONE_ORDER)), bool)
    gr[:, [ZONE_ORDER.index(z) for z in ground]] = True
    con = Q.Contacts(gr, [set(braces) for _ in range(n)], np.zeros(len(ZONE_ORDER), bool), np.zeros(n, bool))
    R = Rotation.from_quat(np.asarray(root_rot[fr], float)).as_matrix()
    _install_freeze()
    with rv2.on_plant(PLANT) as skp:
        dof_rep = rt.nearest_representative(torch.as_tensor(np.asarray(dof[fr], float)), skp.lower, skp.upper).numpy()
        prob = Q.build_problem(skp, name, fr / FPS, np.asarray(root_pos[fr], float), R, dof_rep, con)
        x0 = rv2.solve(prob, iters=0)[0]
        mask = np.zeros(n, bool)
        mask[:PIN_FRAMES] = mask[-PIN_FRAMES:] = True
        exact = torch.cat([torch.zeros(n, 6, dtype=torch.float64), torch.as_tensor(dof_rep)], 1)   # unpack = as given
        x0[torch.as_tensor(mask)] = exact[torch.as_tensor(mask)]
        prob.plan["splice_v3_freeze"] = {"mask": mask, "x": exact}
        prob.weights.update(XP.DEPENETRATE)
        x, rep = rv2.solve(prob, x0=x0, iters=iters)
        rp, rr, dd = (v.numpy() for v in rt.unpack(prob, x))
    keep = slice(PIN_FRAMES, n - PIN_FRAMES)
    # no wall-clock seconds: <stem>.json is part of the build and must reproduce byte for byte
    return (rp[keep], Rotation.from_matrix(rr[keep]).as_quat(), dd[keep],
            {k: rep[k] for k in ("iterations", "energy_start", "energy")})


def blend_frame_checks(sk, c: dict) -> dict:
    """``contract`` on each blend part of the written clip (``part_rules`` from its sources)."""
    from protomotions.utils.rotations import quaternion_to_matrix

    mot, lin, info = c["motion"], c["lineage"], c["info"]
    out = {"depenetrated": info.get("depenetration", {})}
    for part in ("settle", "d_hold"):
        f0, f1 = info["layout"][part]
        f = np.arange(f0, f1 + 1)
        if not (lin["kind"][f] == KIND["blend"]).all():
            raise ValueError(f"{part} frames {f0}..{f1} are not all blend frames")
        rules = part_rules(sk, part, c["sources"]["S"], info["S"]["frame_hold"], c["sources"]["D"],
                           info["D"]["frame_hold"], c["sources"]["V"], c["s_hold"], c["d_hold"])
        P = mot["rigid_body_pos"][f].double()
        R = quaternion_to_matrix(mot["rigid_body_rot"][f].double(), w_last=True)
        k = contract(sk, P, R, mot["dof_pos"][f].double().numpy(), rules)
        worst = int(np.argmin(k["floor_cm"]))
        out[part] = {"frames": int(len(f)), "box_max_deg": round(float(k["box_deg"].max()), 4),
                     "floor_min_cm": round(float(k["floor_cm"].min()), 3), "floor_worst_frame": int(f[worst]),
                     "new_overlaps_1mm": len(k["new"]),
                     "new_overlap_examples": [[int(f[x[0]])] + x[1:] for x in k["new"][:8]],
                     "source_pairs_allowed": len(rules["allowed"]), "braces_gap_cm": rules["braces_gap_cm"],
                     "braces_exempt": sorted("+".join(e) for e in rules["exempt"]),
                     "brace_depth_cm": k["brace_depth_cm"], "failing_frames": [int(f[i]) for i in k["failing"]],
                     "pass": not k["failing"]}
    out["frames"] = out["settle"]["frames"] + out["d_hold"]["frames"]
    out["pass"] = out["settle"]["pass"] and out["d_hold"]["pass"]
    return out


def check_transform(c: dict) -> dict:
    """The hand-anchor transform against the generator's: run.json's ``anchor_yaw_deg``, and the yaw a planar
    Procrustes fit of D's raw exemplar onto the variant's last frame finds (all 24 bodies), with the variant's last
    frame's knuckles against D'."""
    info = c["info"]
    vpos = c["sources"]["V"]["rigid_body_pos"][-1].double().numpy()
    Dm = c["sources"]["D"]
    dfh = info["D"]["frame_hold"]
    d_raw = Dm["rigid_body_pos"][dfh].double().numpy()
    tr = info["transform"]
    d_ex, _ = place(d_raw, Dm["rigid_body_rot"][dfh].double().numpy(), tr["yaw_rad"], tr["t_xy"])
    a, b = vpos[:, :2] - vpos[:, :2].mean(0), d_raw[:, :2] - d_raw[:, :2].mean(0)
    fit = math.degrees(math.atan2(float((b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]).sum()), float((a * b).sum())))
    knuckles = {h: round(100 * float(np.linalg.norm(vpos[BI[h]] - d_ex[BI[h]])), 3) for h in HANDS}
    out = {"yaw_deg": round(tr["yaw_deg"], 4), "run_anchor_yaw_deg": tr["run_anchor_yaw_deg"],
           "yaw_vs_run_deg": round(abs(tr["yaw_deg"] - tr["run_anchor_yaw_deg"]), 4),
           "procrustes_last_frame_yaw_deg": round(fit, 3), "procrustes_minus_yaw_deg": round(fit - tr["yaw_deg"], 3),
           "last_frame_knuckles_to_dprime_cm": knuckles}
    out["pass"] = out["yaw_vs_run_deg"] <= TRANSFORM_YAW_DEG / 2 + 1e-9
    return out


def check_root_transform_fk(sk, c: dict) -> dict:
    """The transform on a real frame: FK of the transformed root with unchanged ``dof_pos`` equals the rigidly
    transformed source bodies (float64), and the lead-out's first frame is D' (float32 FK)."""
    info = c["info"]
    Dm = c["sources"]["D"]
    dfh = info["D"]["frame_hold"]
    tr = info["transform"]
    r, q, d = frames_of(Dm, [dfh])
    p0, _ = fk(sk, r, q, d)
    rp, rq = place(r, q, tr["yaw_rad"], tr["t_xy"])
    p1, _ = fk(sk, rp, rq, d)
    want, _ = place(p0[0], np.tile(q, (24, 1)), tr["yaw_rad"], tr["t_xy"])
    first = info["layout"]["lead_out"][0]
    stored = c["motion"]["rigid_body_pos"][first].double().numpy()
    want32, _ = place(Dm["rigid_body_pos"][dfh].double().numpy(), Dm["rigid_body_rot"][dfh].double().numpy(),
                      tr["yaw_rad"], tr["t_xy"])
    out = {"fk_float64_m": float(np.abs(p1[0] - want).max()), "lead_out_first_m": float(np.abs(stored - want32).max())}
    out["pass"] = max(out.values()) <= REAL_POS_M
    return out


def check_holds(entry: dict, lin: dict, rel: dict, mot: dict | None = None, sources: dict | None = None) -> dict:
    """Well formed (``holds_well_formed``); every ``inherits`` names a release v2 hold whose name, pairs and
    orientation equal the copy's, every copied field equals it, and each ``frame_hold`` is a real frame of that
    hold's own clip at its own exemplar. With the motions, also each hold's mean body speed at ``frame_hold`` in
    the clip and in its source (recorded: the copied ``speed_at_hold`` is the proposer's smoothed value there)."""
    problems = holds_well_formed(entry["holds"], entry["num_frames"])
    by_id = {h["hold_id"]: (c["stem"], h) for c in rel["manifest"]["clips"] for h in c["holds"]}
    stems = [str(s) for s in lin["stems"]]
    speed = {}
    for h in entry["holds"]:
        if h["inherits"] not in by_id:
            problems.append(f"{h['hold_id']}: inherits an unknown hold {h['inherits']}")
            continue
        stem, src = by_id[h["inherits"]]
        for k in set(src) | set(h):
            if k not in HOLD_NEW and src.get(k) != h.get(k):
                problems.append(f"{h['hold_id']}: {k} differs from {h['inherits']}'s")
        f = h["frame_hold"]
        if not (lin["kind"][f] == KIND["real"] and stems[lin["source"][f]] == stem
                and int(lin["source_frame"][f]) == int(src["frame_hold"])):
            problems.append(f"{h['hold_id']}: frame_hold {f} is not {stem}'s exemplar {src['frame_hold']}")
        if mot is not None and stem in sources:
            spl = float(mot["rigid_body_vel"][f].norm(dim=-1).mean())
            own = float(sources[stem]["rigid_body_vel"][int(src["frame_hold"])].norm(dim=-1).mean())
            speed[h["hold_id"]] = {"clip_m_s": round(spl, 4), "source_m_s": round(own, 4),
                                   "speed_at_hold": src.get("speed_at_hold")}
    ext = [h for h in entry["holds"] if h.get("extend")]
    return {"holds": len(entry["holds"]), "extend": [h["hold_id"] for h in ext], "problems": problems,
            "speed_at_frame_hold": speed, "pass": not problems}


def check_hold_content(entry: dict, mot: dict, lin: dict, info: dict, rel: dict) -> dict:
    """Every hold's window holds its label: no synthetic or blend frame inside it touches the floor with a zone
    the hold does not label as ground (``content_windows``). S's and D's windows are the card's (S from its own
    start to the departure, D from the arrival to its own end) trimmed by exactly the rule: recomputed from the
    written contacts, they equal the recorded ``windows`` and the holds. Recorded beside it: each labelled zone's
    share of the non-real frames it touches on, and unlabelled contacts on the windows' real frames (the source's)."""
    lay, win = info["layout"], info["windows"]
    by_id = {h["hold_id"]: h for c in rel["manifest"]["clips"] for h in c["holds"]}
    zc = zone_contacts(mot["rigid_body_contacts"].numpy())
    s_src, d_src = by_id[info["S"]["hold_id"]], by_id[info["D"]["hold_id"]]
    again = content_windows(zc, lay, ground_zones(s_src), ground_zones(d_src))
    s = next(h for h in entry["holds"] if h["inherits"] == info["S"]["hold_id"] and h.get("extend"))
    d = next(h for h in entry["holds"] if h["inherits"] == info["D"]["hold_id"] and h["frame_start"] > s["frame_end"])
    out0, lead_in_start, dfh = lay["lead_out"][0], info["lead_in_source"][0], info["lead_out_source"][0]
    problems = []
    if (again["s_end"], again["d_start"]) != (win["s_end"], win["d_start"]):
        problems.append(f"the windows recompute to {again['s_end']}/{again['d_start']}, not the recorded "
                        f"{win['s_end']}/{win['d_start']}")
    if (s["frame_start"], s["frame_hold"], s["frame_end"]) != (s_src["frame_start"] - lead_in_start,
                                                                lay["s_exemplar"], win["s_end"]):
        problems.append(f"S's window {s['frame_start']}..{s['frame_end']} is not the layout's")
    if (d["frame_start"], d["frame_hold"], d["frame_end"]) != (win["d_start"], out0, d_src["frame_end"] - dfh + out0):
        problems.append(f"D's window {d['frame_start']}..{d['frame_end']} is not the layout's")
    holds = {}
    for h in entry["holds"]:
        fr = np.arange(h["frame_start"], h["frame_end"] + 1)
        nonreal = fr[lin["kind"][fr] != KIND["real"]]
        real = fr[lin["kind"][fr] == KIND["real"]]
        lab = ground_zones(h)
        bad = unlabelled(zc, nonreal, set(lab))
        if bad:
            problems.append(f"{h['hold_id']}: unlabelled floor contact on non-real frames {bad}")
        holds[h["hold_id"]] = {
            "nonreal_frames": int(len(nonreal)), "unlabelled_nonreal": bad,
            "labelled_share_nonreal": {z: round(float(zc[z][nonreal].mean()), 4) for z in lab} if len(nonreal) else {},
            "unlabelled_real": {z: len(f) for z, f in unlabelled(zc, real, set(lab)).items()}}
    return {"windows": {k: win[k] for k in ("s_end", "s_end_card", "d_start", "d_start_card")},
            "trimmed": [k for k, c in (("S", win["s_end"] != win["s_end_card"]),
                                        ("D", win["d_start"] != win["d_start_card"])) if c],
            "holds": holds, "problems": problems, "pass": not problems}


def check_extension(mot: dict, entry: dict) -> dict:
    """R3's extension step, dry-run: ``hold_extension_v2``'s velocity guard on the spliced clip, then its
    insertion (``insertion_plan`` at S, the only ``extend`` hold), ``extend_motion_v2`` and ``variant_holds`` for
    every duration variant: the inserted frames repeat S's exemplar and the re-timed holds stay well formed."""
    from reference_curation import hold_extension_v2 as hx

    drift = {k: (round(m, 6), round(x, 6)) for k, (m, x) in hx.velocity_drift(mot).items()}
    out = {"velocity_drift_mean_max": drift, "variants": {}}
    ok = all(m <= hx.MAX_VELOCITY_DRIFT for m, _ in drift.values())
    s = next(h for h in entry["holds"] if h.get("extend"))
    T = int(mot["dof_pos"].shape[0])
    for d in EXTENSION_S:
        n = int(round(d * FPS))
        plan = hx.insertion_plan(entry["holds"], n)
        variant, index, inserted = hx.extend_motion_v2(mot, plan)
        holds = hx.variant_holds(entry["holds"], plan, FPS)
        Tv = int(variant["dof_pos"].shape[0])
        rep = bool((index[inserted] == s["frame_hold"]).all())
        pressure_zero = all(float(variant[k].abs().max()) == 0.0 for k in hx.PRESSURE_FIELDS)
        probs = holds_well_formed(holds, Tv, x0=False)
        sv = next(h for h in holds if h.get("extend"))
        row = {"stem": hx.variant_stem(entry["stem"], d), "plan": plan, "frames": Tv, "inserted": int(inserted.sum()),
               "repeats_s_exemplar": rep, "pressure_zero": pressure_zero,
               "s_window": [sv["frame_start"], sv["frame_end"]],
               "length_s": round(Tv / FPS, 3), "hold_problems": probs}
        row["pass"] = (plan == [(s["frame_hold"], n)] and Tv == T + n and rep and pressure_zero and not probs
                       and sv["frame_end"] - sv["frame_start"] == s["frame_end"] - s["frame_start"] + n)
        out["variants"][row["stem"]] = row
        ok &= row["pass"]
    out["pass"] = bool(ok)
    return out


def check_motion_file(mot: dict) -> dict:
    """Plant v2's identity, 60 fps, every field finite, the pressure channels zero with validity 0 (``fit_writer
    .round_trip``: stored bodies = FK of the stored coordinates to ``REAL_POS_M``)."""
    rt_ = fw.round_trip(mot, PLANT)
    pressure = {k: float(mot[k].abs().max()) for k in ("ground_reaction", "rigid_body_ground_forces",
                                                         "ground_reaction_valid")}
    out = {"round_trip": rt_, "fps": int(mot["fps"]), "pressure_max": pressure,
           "plant_sha256": mot.get(plant_identity.KEY)}
    out["pass"] = (fw.round_trip_ok(rt_) and out["fps"] == FPS and max(pressure.values()) == 0.0
                   and out["plant_sha256"] == plant_identity.sha256(PLANT))
    return out


def check_clip(sk, c: dict, entry: dict, rel: dict) -> dict:
    """Every per-clip acceptance item of the card on a spliced clip (in memory or reloaded)."""
    lin = c["lineage"]
    sources = {str(lin["stems"][0]): c["sources"]["S"], str(lin["stems"][1]): c["sources"]["D"]}
    out = {"motion": check_motion_file(c["motion"]), "real_frames": check_real_frames(c["motion"], lin, sources),
           "synthetic_frames": check_synthetic_frames(c["motion"], c["lineage"], c["sources"]["V"]),
           "seams": check_seams(c), "blend_frames": blend_frame_checks(sk, c), "transform": check_transform(c),
           "transform_fk": check_root_transform_fk(sk, c),
           "holds": check_holds(entry, lin, rel, c["motion"], sources),
           "hold_content": check_hold_content(entry, c["motion"], lin, c["info"], rel),
           "near_repeats": near_repeats(c["motion"]["rigid_body_pos"].double().numpy(), entry["holds"]),
           "extension": check_extension(c["motion"], entry)}
    out["failed"] = [k for k, v in out.items() if not v["pass"]]
    return out


# --------------------------------------------------------------------------- #
# Library-level checks: MotionLib on the CPU, the graph dry run
# --------------------------------------------------------------------------- #
def check_motionlib(paths: list[Path], rel: dict) -> dict:
    """``MotionLib`` (CPU, admit.py's pattern) loads the spliced clips with release v2's 168 motions, on plant v2."""
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    human = [rel["dir"] / "motions" / f"{s}.motion" for s in rel["record"]["motions"]]
    with tempfile.TemporaryDirectory() as tmp:
        y = Path(tmp) / "library.yaml"
        y.write_text("motions:\n" + "".join(f"  - file: {p.resolve()}\n    weight: 1.0\n" for p in human + list(paths)))
        lib = MotionLib(MotionLibConfig(motion_file=str(y)), device="cpu")
    try:
        plant_identity.require(lib.plant_sha256, plant_identity.mjcf_path(PLANT), "synthetic_v3 + release v2")
        plant_ok = True
    except Exception as exc:  # noqa: BLE001
        plant_ok = f"{type(exc).__name__}: {exc}"
    frames = [int(n) for n in lib.motion_num_frames]
    want = [int(load(p)["dof_pos"].shape[0]) for p in paths]
    out = {"motions": int(len(lib.motion_num_frames)), "human": len(human), "synthetic": len(paths),
           "plant_v2": plant_ok, "synthetic_frames_match": frames[len(human):] == want,
           "pressure_channel_packed": getattr(lib, "grc", None) is not None}
    out["pass"] = out["motions"] == len(human) + len(paths) and plant_ok is True and out["synthetic_frames_match"]
    return out


def endpoint_stems(edges: dict, edge_ids) -> list[str]:
    out = []
    for e in edges["edges"]:
        if e["id"] in edge_ids:
            for ep in (e["source"], e["destination"]):
                if ep["stem"] not in out:
                    out.append(ep["stem"])
    return out


def check_graph(entries: list[dict], paths: list[Path], inp: dict, extended: bool = False) -> dict:
    """``build_hold_graph_v2.py`` (unchanged, its runtime round trip included) into a temporary folder on a small
    library packaged by ``package_motion_subset.py``: the edges' endpoint clips' x0 motions plus the spliced clips
    (with ``extended``, also every spliced clip's ``EXTENSION_S`` variants, made as release v2's ``extend_corpus``
    makes them: R3's library). Every synthetic node key must be one of release v2's, and each edge's S -> D must
    appear once per spliced motion."""
    from edge_synthesis import sketch as SK
    from reference_curation import hold_extension_v2 as hx

    rel = inp["release"]
    edge_ids = sorted({e["synthetic"]["edge"] for e in entries})
    real = endpoint_stems(inp["edges"], edge_ids)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        files = [rel["dir"] / "motions" / f"{s}.motion" for s in real] + list(paths)
        clips = [rel["clips"][s] for s in real] + list(entries)
        if extended:
            for entry, path in zip(entries, paths):
                mot = load(path)
                for d in EXTENSION_S:
                    plan = hx.insertion_plan(entry["holds"], int(round(d * FPS)))
                    variant, index, _ = hx.extend_motion_v2(mot, plan)
                    name = hx.variant_stem(entry["stem"], d)
                    torch.save(variant, tmp / f"{name}.motion")
                    files.append(tmp / f"{name}.motion")
                    n = int(index.shape[0])
                    clips.append({**{k: v for k, v in entry.items() if k not in ("holds", "num_frames", "length_s")},
                                  "stem": name, "source_stem": entry["stem"], "variant_s": float(d), "num_frames": n,
                                  "length_s": round(n / FPS, 3), "holds": hx.variant_holds(entry["holds"], plan, FPS)})
        env = dict(os.environ, PYTHONPATH=f"{REPO}:{SCRIPTS}")
        env.pop("REFERENCE_PLANT", None)
        (tmp / "holds.yaml").write_text(yaml.safe_dump({"version": 2, "clips": clips}, sort_keys=False, width=120))
        for script, args in (("package_motion_subset.py", ["--force", "--out", tmp / "motions.pt", "--yaml",
                                                           tmp / "motions.yaml", *files]),
                             ("build_hold_graph_v2.py", ["--manifest", tmp / "holds.yaml", "--motion-file",
                                                         tmp / "motions.pt", "--out-dir", tmp, "--min-lead-s", 0.2])):
            proc = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], cwd=REPO,
                                  capture_output=True, text=True, env=env)
            if proc.returncode != 0:
                return {"pass": False, "error": f"{script} exited {proc.returncode}: {proc.stderr[-1500:]}"}
        g = json.load(open(tmp / "contact_graph.json"))
    keys = [n["key"] for n in g["nodes"]]
    syn = sorted({s["config"] for stem, cl in g["clips"].items() if stem.startswith("SYN_") for s in cl["segments"]})
    edges = {(keys[x["src"]], keys[x["dst"]]): x for x in g["edges"]}
    per_clip = 1 + (len(EXTENSION_S) if extended else 0)
    need = {}
    for eid in edge_ids:
        e = SK.edge(inp["edges"], eid)
        sk_, dk_ = e["source"]["node_key"], e["destination"]["node_key"]
        occ = edges.get((sk_, dk_))
        n_syn = sum(1 for o in (occ or {}).get("occurrences", []) if o["motion"].startswith("SYN_"))
        want = per_clip * sum(1 for x in entries if x["synthetic"]["edge"] == eid)
        need[eid] = {"S": sk_, "D": dk_, "present": occ is not None, "synthetic_occurrences": n_syn, "motions": want,
                     "pass": occ is not None and n_syn == want}
    out = {"library": {"real_x0": real, "synthetic_motions": len(files) - len(real), "extended": extended},
           "nodes": len(keys), "edges": len(g["edges"]), "hold_ids": len(g["hold_ids"]), "synthetic_keys": syn,
           "keys_not_in_v2": sorted(set(syn) - set(rel["node_keys"])),
           "all_keys_not_in_v2": sorted(set(keys) - set(rel["node_keys"])),
           "pair_names_equal_v2": g["pair_names"] == rel["pair_names"], "edges_s_to_d": need}
    out["pass"] = (not out["keys_not_in_v2"] and not out["all_keys_not_in_v2"] and out["pair_names_equal_v2"]
                   and all(x["pass"] for x in need.values()))
    return out


# --------------------------------------------------------------------------- #
# Build, record, check
# --------------------------------------------------------------------------- #
def paths_of(out: Path, name: str) -> dict:
    return {k: out / f"{name}.{ext}" for k, ext in (("motion", "motion"), ("holds", "holds.yaml"),
                                                       ("lineage", "lineage.npz"), ("json", "json"))}


def require_v1_process() -> None:
    if os.environ.get("REFERENCE_PLANT") or ids.PLANT != "v1":
        raise RuntimeError("run splice_v3 without REFERENCE_PLANT (plant v2 is named explicitly where it is meant)")


def build(force: bool = False, log=print) -> tuple[Path, dict]:
    """Write every clip, check them all, and write the record last (only when every check passes)."""
    require_v1_process()
    start = time.time()
    inp = inputs()
    sid = synthetic_id(inp)
    out, record_path = HEAVY_ROOT / sid, RECORD_ROOT / f"{sid}.json"
    if record_path.exists():
        raise FileExistsError(f"{sid} exists (its record is {ids.display_path(record_path)}); run --check")
    if out.exists():
        if not force:
            raise FileExistsError(f"{ids.display_path(out)} exists without a record (a failed build); pass --force "
                                  "to delete it and rebuild")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rel = inp["release"]
    names = [v["name"] for v in inp["variants"]]
    names_chk = name_checks(names, list(rel["clips"]))
    if not names_chk["pass"]:
        raise ValueError(f"the names fail the card's asserts: {names_chk}")
    sk = fw.skeleton(PLANT)
    cache, clips, entries, paths = {}, {}, [], []
    for v in inp["variants"]:
        t0 = time.time()
        c = splice(sk, v, inp, cache)
        p = paths_of(out, c["name"])
        torch.save(c["motion"], p["motion"])
        entry = clip_entry(c, v, inp, p["motion"])
        with open(p["holds"], "w") as f:
            yaml.safe_dump(entry, f, sort_keys=False, width=120)
        np.savez_compressed(p["lineage"], **c["lineage"])
        c = reload(c, p)
        attach_endpoints(c, inp)
        chk = check_clip(sk, c, entry, rel)
        info = {**c["info"], "checks": chk}           # no wall-clock seconds: the file reproduces byte for byte
        p["json"].write_text(json.dumps(info, indent=1, default=_json) + "\n")
        clips[c["name"]] = {"variant": v["row"]["variant"], "edge": v["row"]["edge"], "frames": entry["num_frames"],
                            "length_s": entry["length_s"], "failed": chk["failed"],
                            "windows_trimmed": chk["hold_content"]["trimmed"],
                            "transition_contacts": {z: x["frames"]
                                                    for z, x in c["info"]["transition_contacts"].items()},
                            **{k: {"path": ids.display_path(pp), "sha256": ids.sha256_file(pp)} for k, pp in p.items()}}
        entries.append(entry)
        paths.append(p["motion"])
        log(f"{c['name']}: {entry['num_frames']} frames ({entry['length_s']} s), failed {chk['failed']} "
            f"[{time.time() - t0:.1f} s]")
    lib = check_motionlib(paths, rel)
    log(f"MotionLib: {lib}")
    graph = {"x0": check_graph(entries, paths, inp), "extended": check_graph(entries, paths, inp, extended=True)}
    log(f"graph dry run: x0 {graph['x0']['pass']}, extended {graph['extended']['pass']}")
    failed = [f"{n}: {c['failed']}" for n, c in clips.items() if c["failed"]]
    failed += [] if lib["pass"] else [f"motionlib: {lib}"]
    failed += [f"graph {k}: {g}" for k, g in graph.items() if not g["pass"]]
    rec = record(sid, inp, out, clips, names_chk, lib, graph, failed, time.time() - start)
    if failed:
        (out / "failed_record.json").write_text(json.dumps(rec, indent=1, default=_json) + "\n")
        raise RuntimeError(f"{len(failed)} checks failed; {ids.display_path(out)} is not a build:\n  "
                           + "\n  ".join(failed[:20]))
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(rec, indent=1, default=_json) + "\n")
    return out, rec


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def reload(c: dict, p: dict) -> dict:
    """The clip as written: the ``.motion`` and ``lineage.npz`` read back from disk."""
    c = dict(c)
    c["motion"] = load(p["motion"])
    with np.load(p["lineage"]) as z:
        c["lineage"] = {k: z[k] for k in z.files}
    return c


def attach_endpoints(c: dict, inp: dict) -> None:
    from edge_synthesis import sketch as SK

    e = SK.edge(inp["edges"], c["info"]["edge"])
    c["edge"] = e
    _, c["s_hold"] = endpoint(inp["release"], e["source"])
    _, c["d_hold"] = endpoint(inp["release"], e["destination"])


def record(sid: str, inp: dict, out: Path, clips: dict, names_chk: dict, lib: dict, graph: dict, failed: list,
           seconds: float) -> dict:
    var_inputs = [p for v in inp["variants"] for p in (v["motion"], v["record"], v["run"])]
    by_edge = defaultdict(list)
    for n, c in clips.items():
        by_edge[c["edge"]].append(n)
    rel = inp["release"]
    return {
        **ids.provenance(SCHEMA_VERSION, MODULE, __file__, [SELECTED, ADMITTED, RELEASE_V2_RECORD, EDGES,
                                                             rel["dir"] / "release_holds.yaml", *var_inputs]),
        "kind": "synthetic_clips", "synthetic_id": sid, "version": SPLICE_VERSION, "dir": ids.display_path(out),
        "release_v2": {"release_id": RELEASE_V2_ID, "record": ids.display_path(RELEASE_V2_RECORD),
                       "sha256": ids.sha256_file(RELEASE_V2_RECORD)},
        "plant": plant_identity.identity(PLANT), "key": id_key(inp), "parameters": parameters(),
        "builders": {ids.display_path(p): ids.sha256_file(p) for p in builders()},
        "order": list(clips), "edges": dict(by_edge), "clips": clips,
        "names": names_chk, "name_collisions": names_chk["name_collisions"],
        "name_note": ("t6px is a prefix of t6px12 and t6rpx of t6rpx12, so a synthetic stem can be a substring of "
                      "another clip's name (name_collisions): every per-stem filter downstream (R3's --drop, any "
                      "drop list) must match stems exactly, never by substring. The runtime's motion filters match "
                      "by substring (contact_graph_control: support_exclude_motions, physics_exclude_motions; "
                      "hold_curriculum_evaluator: report_exclude_motions, drag_report_motions), so a pattern naming "
                      "a shorter stem there also selects the longer one and its _x3s/_x7s"),
        "windows_trimmed": {n: c["windows_trimmed"] for n, c in clips.items() if c["windows_trimmed"]},
        "transition_contacts": {n: c["transition_contacts"] for n, c in clips.items() if c["transition_contacts"]},
        "transition_contacts_note": ("zones touching the floor (rigid_body_contacts) on transition frames that no "
                                     "phase of the edge, nor S or D, puts on the floor: the variant's own content "
                                     "(admission's free_violations), recorded, not edited"),
        "d1_policy": inp["d1_policy"], "t6_decision": inp["t6_decision"],
        "motionlib": lib, "graph_dry_run": graph,
        "lineage_format": {
            "stems": "[S stem, D stem, variant name]; source indexes it",
            "source/source_frame": "per frame: the frame's source and its frame there (variant frames for settle, "
                                   "transition and hold at D)",
            "kind": KIND, "segment": "index into seg_* (lead_in, s_exemplar, settle, transition, d_hold, lead_out)",
            "blend_w/blend_source/blend_source_frame": (
                "on blend frames: the real exemplar's weight and frame (S's exemplar on the settle, D's on the hold "
                "at D); 0/-1 elsewhere"),
            "yaw/xy": "per frame: the transform placing the frame's real source (or blend partner) in the clip: "
                      "rotate by yaw about z, then shift by xy (root only; dof_pos unchanged)",
            "variant_t": "t on the variant's clock (NaN on real frames)",
            "depenetrated": "per frame: edited by the de-penetration solve (blend frames only; <stem>.json: how much)"},
        "checks": {"failed": failed, "clips": {n: c["failed"] for n, c in clips.items()}},
        "seconds": round(seconds, 1),
    }


def check(sid: str, log=print) -> tuple[list, list]:
    """Re-run every check on a recorded build: file sha256s against the record, the recorded inputs (the four
    pinned files and every variant's motion, export record and run.json, by the sha256s in the record's ``key``)
    still on disk, then the per-clip and library checks on the files as written. Each variant is resolved from its
    clip's own ``synthetic`` block, never from today's selection. ``(passed, failed)``."""
    require_v1_process()
    rec = json.loads((RECORD_ROOT / f"{sid}.json").read_text())
    passed, failed = [], []
    try:
        now = synthetic_id(inputs())
        if now != sid:
            log(f"note: today's inputs and builders determine {now}; {sid} was built by the recorded ones")
    except Exception as exc:  # noqa: BLE001
        log(f"note: today's inputs no longer resolve ({type(exc).__name__}: {exc}); checking the recorded ones")
    # the recorded inputs the checks read (release v2's record, edges.json) must be on disk unchanged; the selection
    # and the admission are not read here (each clip carries its admission row), so a change there is only noted
    key = rec["key"]
    read = {ids.display_path(RELEASE_V2_RECORD), ids.display_path(EDGES)}
    changed = [p for p, sha in key["inputs"].items() if not (REPO / p).exists() or ids.sha256_file(REPO / p) != sha]
    if set(changed) - read:
        log(f"note: {sorted(set(changed) - read)} changed since {sid} was built (not read by the checks)")
    bad = sorted(set(changed) & read)
    (failed if bad else passed).append(f"recorded inputs the checks read{f': {bad} changed' if bad else ''}")
    for name, c in rec["clips"].items():
        for k in ("motion", "holds", "lineage", "json"):
            ok = ids.sha256_file(REPO / c[k]["path"]) == c[k]["sha256"]
            (passed if ok else failed).append(f"{name}.{k} sha256")
    rel = release_v2()
    inp = {"release": rel, "edges": json.loads(EDGES.read_text())}
    names_chk = name_checks(list(rec["clips"]), list(rel["clips"]))
    (passed if names_chk["pass"] and names_chk["name_collisions"] == rec["name_collisions"] else failed).append("names")
    sk = fw.skeleton(PLANT)
    entries, paths = [], []
    cache = {}
    for name, c in rec["clips"].items():
        p = {k: REPO / c[k]["path"] for k in ("motion", "holds", "lineage", "json")}
        entry = yaml.safe_load(open(p["holds"]))
        syn, want = entry["synthetic"], key["variants"].get(name)
        files = {"motion": REPO / syn["motion"], "record": REPO / syn["record"],
                 "run": REPO / syn["source_run"] / "run.json"}
        moved = [k for k, f in files.items() if want is None or not f.exists() or ids.sha256_file(f) != want[k]]
        if moved:
            failed.append(f"{name}: the recorded variant's {moved} changed or is missing; its clip checks are skipped")
            continue
        passed.append(f"{name}: the recorded variant's files")
        info = json.loads(p["json"].read_text())
        cl = {"name": name, "info": {k: info[k] for k in info if k not in ("checks", "seconds")}}
        for stem in (info["S"]["stem"], info["D"]["stem"]):
            if stem not in cache:
                cache[stem] = release_motion(rel, stem)
        cl["sources"] = {"S": cache[info["S"]["stem"]], "D": cache[info["D"]["stem"]], "V": load(files["motion"])}
        cl = reload(cl, p)
        attach_endpoints(cl, inp)
        chk = check_clip(sk, cl, entry, rel)
        for k, r in chk.items():
            if k != "failed":
                (passed if r["pass"] else failed).append(f"{name}: {k}")
        entries.append(entry)
        paths.append(p["motion"])
    lib = check_motionlib(paths, rel)
    (passed if lib["pass"] else failed).append(f"motionlib: {lib['motions']} motions, plant v2 {lib['plant_v2']}")
    for ext in (False, True):
        graph = check_graph(entries, paths, inp, extended=ext)
        (passed if graph["pass"] else failed).append(
            f"graph dry run ({'x0 + x3s + x7s' if ext else 'x0'}): keys not in v2 {graph.get('keys_not_in_v2')}, "
            f"S->D {[(k, x['pass']) for k, x in graph.get('edges_s_to_d', {}).items()]}")
    return passed, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="splice, check and record every selected variant")
    ap.add_argument("--check", metavar="SYNTHETIC_ID", help="re-run every check on a recorded build")
    ap.add_argument("--print-id", action="store_true", help="print the id the inputs and builders determine")
    ap.add_argument("--force", action="store_true", help="delete a failed build's folder (one without a record)")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.print_id:
            print(synthetic_id(inputs()))
            return 0
        if args.build:
            out, rec = build(force=args.force)
            print(f"{rec['synthetic_id']}: {len(rec['clips'])} clips "
                  f"({', '.join(f'{k} x{len(v)}' for k, v in rec['edges'].items())}) in {rec['seconds']:.0f} s -> "
                  f"{ids.display_path(RECORD_ROOT / (rec['synthetic_id'] + '.json'))}")
            return 0
        if args.check:
            passed, failed = check(args.check)
            for f in failed:
                print(f"FAILED  {f}", file=sys.stderr)
            print(f"{args.check}: {len(passed)} checks passed, {len(failed)} failed")
            return 1 if failed else 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
