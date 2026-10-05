# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contact-target sidecar of release v3 (card R3 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``).

Release v3 is release v2 plus the spliced synthetic clips (``splice_v3``), appended after v2's 168 motions. The
sidecar keeps ``contact_targets_v2``'s format (the runtime's ``ContactTargets`` loads it unchanged) and is compiled in
two parts:

* **Human motions: v2's ``compile_targets``, unchanged,** run on a view of the release that holds only the human
  motions' graph rows (``human_view``). Their rows are therefore what v2's compiler makes of them, and release v3's
  checks compare them with release v2's sidecar bit for bit.
* **Synthetic motions** have no labels, no gate rows and no capture of their own: every synthetic hold *inherits* a
  release v2 hold (``inherits``), and the synthetic frames are the generator's.

  - **Per segment**, every array is copied from the inherited hold's row (its x0 motion's): commanded, configured,
    masked, roles, critical, restored, necessity, the LP loads, the known-free zones and their shares, the gate's
    flags and the pose role. The inherited hold's annotations, gate block and human evidence are the synthetic
    hold's, since its name, contacts and labels are copied verbatim.
  - **Per frame** (``frame_pair_*``, the configured body-body pairs ``Q``), on the x0 clip; the x3s / x7s rows are
    the x0 rows read through the variant's ``index`` (an inserted frame takes S's exemplar, a real frame), as v2 maps
    a human variant to its x0 frames.

    ======================  ==============================================  ==========================================
    frame kind (lineage)    "ref" (the reference closes the pair)           "human" (the pair is in contact)
    ======================  ==============================================  ==========================================
    real (lead-in, S's      capture v4's ``avatar_pair_gap`` <= 1 cm at     capture v4's ``human_pair_state`` == 1 at
    exemplar, lead-out)     the source stem's ``source_frame``              the same frame
    synthetic, blend        the clip's own geometry: ``capture_v4           the generator's planned brace for the
    (settle, transition,    .zone_pair_gaps`` (the kernel capture v4         frame's phase (``planned_config``: S's
    hold at D)              used) <= 1 cm                                   braces before the departure, the phase's
                                                                            in the transition, D's from the arrival)
    ======================  ==============================================  ==========================================

    A real frame's pair gaps do not change under the yaw/XY transform that placed it (a rigid motion), so the source
    stem's evidence is the clip's; ``compile`` measures the geometric gaps on the real frames too and reports how far
    they are from capture v4's (``real_frame_gap_vs_capture_m``).

The **planned contacts** come from ``edges.json``'s phases with the variant's ``durations_s``, on the variant's clock
(lineage ``variant_t``), exactly as the generator's ``sketch.Schedule.phase_at`` / ``config_at`` read them (mirrored
here, so this module does not import MuJoCo; a test checks the two agree). ``known_free_conflicts`` checks that no
synthetic or blend frame's planned ground zone is a known-free zone of the hold the schedule commands at that frame:
the hold whose window holds it (S or D), or outside every window the next hold (``include_current_segment``'s slot
0).
"""

from __future__ import annotations

import collections
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from extract_contact_configs import ZONE_ORDER
from reference_curation import capture_v4, contact_targets_v2 as ct, ids

PAIR_CLOSE_M = ct.PAIR_CLOSE_M
KIND_REAL = 0                     # splice_v3.KIND: 0 real, 1 synthetic, 2 blend
SEG_KEYS = ("seg_commanded", "seg_configured", "seg_masked", "seg_role", "seg_critical", "seg_restored",
            "seg_necessity", "seg_lp_load_n", "seg_lp_load_min_n", "seg_lp_load_max_n", "seg_ground_free",
            "seg_ground_free_share", "seg_flags", "seg_pose_role")
FRAME_KEYS = ("frame_pair_ok", "frame_pair_ref", "frame_pair_human")
# Graph payload fields indexed by motion first: the human view slices them.
GRAPH_MOTION_AXIS = ("seg_node", "seg_contact", "seg_start", "seg_end", "seg_hold", "seg_count", "seg_hold_index",
                     "motion_num_frames")
RULES = {
    "synthetic_segments": "every array copied from the inherited hold's row (the inherits hold's x0 motion)",
    "synthetic_frames": (f"real frames: capture v4 at the source stem's source_frame (avatar_pair_gap <= "
                         f"{PAIR_CLOSE_M} m, human_pair_state == 1); synthetic and blend frames: ref = the clip's "
                         f"geometry (capture_v4.zone_pair_gaps) <= {PAIR_CLOSE_M} m, human = the generator's planned "
                         "brace for the frame's phase (edges.json, variant_t, durations_s; sketch.Schedule's rule); "
                         "x3s/x7s rows through the variant's index"),
}


# --------------------------------------------------------------------------- #
# The generator's schedule (sketch.Schedule.phase_at / config_at), pure
# --------------------------------------------------------------------------- #
def edge_spec(edges: dict, edge_id: str) -> dict:
    for e in edges["edges"]:
        if e["id"] == edge_id:
            return e
    raise KeyError(edge_id)


def phase_at(t, durations) -> np.ndarray:
    """``sketch.Schedule.phase_at``: -1 before the departure, P after the arrival (t >= T), else the phase index."""
    t = np.asarray(t, float)
    bounds = np.r_[0.0, np.cumsum(np.asarray(durations, float))]
    p = np.searchsorted(bounds, t, side="right") - 1
    return np.where(t < 0, -1, np.where(t >= bounds[-1], len(durations), p))


def planned_config(e: dict, phase: int) -> tuple[list[str], list[str]]:
    """``(ground zones, braces)`` the generator plans in ``phase`` (``sketch.Schedule.config_at``): S's before the
    departure, D's after the arrival, the phase's in between."""
    if phase < 0:
        return sorted(e["source"]["ground"]), sorted(b["pair"] for b in e["source"]["braces"])
    if phase >= len(e["phases"]):
        return sorted(e["destination"]["ground"]), sorted(b["pair"] for b in e["destination"]["braces"])
    p = e["phases"][phase]
    return sorted(p["ground"]), sorted(p["braces"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def lineage_of(lineage, name: str) -> dict:
    """A synthetic variant's arrays in the release's ``lineage.npz`` (``release_v3.extend_corpus``)."""
    return {k: lineage[f"{name}.{k}"] for k in ("index", "inserted", "kind", "source", "source_frame", "variant_t",
                                                 "stems")}


def human_view(release_dir: Path, tmp: Path, graph: dict, n_human: int) -> Path:
    """A release folder holding only the first ``n_human`` motions' graph rows (the human ones; release v3 puts
    them first), with the release's own manifest and lineage linked in: what ``ct.compile_targets`` reads."""
    view = dict(graph)
    view["motion_names"] = list(graph["motion_names"][:n_human])
    for key in GRAPH_MOTION_AXIS:
        view[key] = graph[key][:n_human].clone()
    torch.save(view, tmp / "contact_graph.pt")
    for name in ("holds_extended.yaml", "lineage.npz"):
        os.symlink(Path(release_dir).resolve() / name, tmp / name)
    return tmp


def known_free_conflicts(holds: list[dict], kind: np.ndarray, phase: np.ndarray, e: dict, free_rows: dict) -> list:
    """Frames whose planned ground zone is known-free in the hold the schedule commands there: the hold whose window
    holds the frame, or (outside every window) the next hold. ``free_rows`` maps a hold id to its known-free zones.
    Real frames are the source's and are not planned."""
    out = []
    ordered = sorted(holds, key=lambda h: h["frame_start"])
    for f in np.nonzero(kind != KIND_REAL)[0]:
        f = int(f)
        hold = next((h for h in ordered if h["frame_start"] <= f <= h["frame_end"]), None)
        if hold is None:
            hold = next((h for h in ordered if h["frame_start"] > f), None)
        if hold is None:
            continue
        ground, _ = planned_config(e, int(phase[f]))
        bad = sorted(set(ground) & set(free_rows[hold["hold_id"]]))
        if bad:
            out.append([f, hold["hold_id"], bad])
    return out


# --------------------------------------------------------------------------- #
# Compile
# --------------------------------------------------------------------------- #
def compile_targets_v3(release_dir: Path, labels_dir: Path, x0_manifest: dict, release_id: str, edges: dict,
                       n_human: int) -> tuple[dict, dict]:
    """``(payload, summary)`` of release v3's ``contact_targets.pt``. ``x0_manifest`` is release v2's manifest (the
    human holds' labels), ``edges`` the pinned ``edges.json``, ``n_human`` the number of human motions (first in the
    graph)."""
    from reference_curation import fit_writer as fw
    from reference_curation import human_mesh as hm
    from reference_curation.retarget_v2 import motion_state

    capture_v4.require_v1_process()
    release_dir = Path(release_dir)
    graph = torch.load(release_dir / "contact_graph.pt", map_location="cpu", weights_only=False)
    names = list(graph["motion_names"])
    if any(n.startswith("SYN_") for n in names[:n_human]) or not all(n.startswith("SYN_") for n in names[n_human:]):
        raise ValueError("release v3 puts the human motions first and the SYN_ motions after them")

    # 1. the human motions: v2's compiler on the human view
    with tempfile.TemporaryDirectory(prefix="release_v3_human_view_") as tmp:
        human, human_summary = ct.compile_targets(human_view(release_dir, Path(tmp), graph, n_human), labels_dir,
                                                  x0_manifest, release_id)
    pairs = list(graph["pair_names"])
    pidx = {p: i for i, p in enumerate(pairs)}
    q_pairs = list(human["frame_pair_names"])
    qidx = {p: i for i, p in enumerate(q_pairs)}

    # 2. the synthetic motions
    ext = yaml.safe_load(open(release_dir / "holds_extended.yaml"))
    ext_clips = {c["stem"]: c for c in ext["clips"]}
    holds = ct.segment_holds(ext_clips, graph)
    lineage = np.load(release_dir / "lineage.npz")
    anns = ct.read_annotations(labels_dir)
    # the inherited rows: hold id -> (motion, segment) of its x0 motion
    row_of = {}
    for m in range(n_human):
        if float(ext_clips[names[m]]["variant_s"]) == 0.0:
            for k, h in enumerate(holds[m]):
                row_of[h["hold_id"]] = (m, k)
    M = len(names)
    lengths = [int(n) for n in graph["motion_num_frames"]]
    T = max(lengths)
    seg = {k: torch.zeros((M, *human[k].shape[1:]), dtype=human[k].dtype) for k in SEG_KEYS}
    for k in SEG_KEYS:
        seg[k][:] = -1 if k == "seg_pose_role" else (float("nan") if human[k].is_floating_point()
                                                     and k.startswith("seg_lp") else 0)
        seg[k][:n_human] = human[k]
    frame = {k: torch.zeros(M, T, len(q_pairs), dtype=torch.bool) for k in FRAME_KEYS}
    for k in FRAME_KEYS:
        frame[k][:n_human, : human[k].shape[1]] = human[k]

    sk = fw.skeleton("v2")
    q_cols = [hm.PAIR_NAMES.index(p) for p in q_pairs]
    evidence = {}
    x0_rows, facts = {}, {}
    free_rows = {hid: [z for z, f in zip(ZONE_ORDER, seg["seg_ground_free"][m, k].tolist()) if f]
                 for hid, (m, k) in row_of.items()}
    for m in range(n_human, M):
        name, entry = names[m], ext_clips[names[m]]
        # per segment: the inherited hold's row
        for k, h in enumerate(holds[m]):
            if h.get("inherits") not in row_of:
                raise ValueError(f"{h['hold_id']}: inherits {h.get('inherits')}, which no human x0 segment holds")
            mh, kh = row_of[h["inherits"]]
            for key in SEG_KEYS:
                seg[key][m, k] = seg[key][mh, kh]
            free_rows[h["hold_id"]] = free_rows[h["inherits"]]
        src = entry["source_stem"]
        if float(entry["variant_s"]) != 0.0:
            continue
        # per frame, on the x0 clip
        lin = lineage_of(lineage, name)
        stems = [str(s) for s in lin["stems"]]
        syn = entry["synthetic"]
        e = edge_spec(edges, syn["edge"])
        n = lengths[m]
        if len(lin["kind"]) != n:
            raise ValueError(f"{name}: lineage has {len(lin['kind'])} frames, the library {n}")
        phase = phase_at(np.nan_to_num(lin["variant_t"], nan=-1.0), syn["durations_s"])
        mot = torch.load(release_dir / "motions" / f"{name}.motion", map_location="cpu", weights_only=False)
        pos, rot = motion_state(mot)
        with torch.no_grad():
            gap = capture_v4.zone_pair_gaps(sk, torch.as_tensor(pos), torch.as_tensor(rot))[:, q_cols]
        ref = gap <= PAIR_CLOSE_M
        human_state = np.zeros((n, len(q_pairs)), bool)
        real = lin["kind"] == KIND_REAL
        gap_diff = 0.0
        for s_idx in np.unique(lin["source"][real]):
            stem = stems[int(s_idx)]
            if stem not in evidence:
                rec = capture_v4.load(stem, rebuild=False)
                if list(rec.meta["pair_order"]) != list(hm.PAIR_NAMES):
                    raise ValueError(f"{stem}: capture v4's pair order is not human_mesh.PAIR_NAMES")
                evidence[stem] = {"pair_human": np.asarray(rec["human_pair_state"]),
                                  "pair_gap": np.asarray(rec["avatar_pair_gap"], np.float64)}
            ev = evidence[stem]
            f = np.nonzero(real & (lin["source"] == s_idx))[0]
            sf = lin["source_frame"][f]
            captured = ev["pair_gap"][sf][:, q_cols]
            gap_diff = max(gap_diff, float(np.abs(np.minimum(gap[f], 0.1) - np.minimum(captured, 0.1)).max()))
            ref[f] = captured <= PAIR_CLOSE_M
            human_state[f] = ev["pair_human"][sf][:, q_cols] == 1
        planned_braces = {}
        for f in np.nonzero(~real)[0]:
            p = int(phase[f])
            if p not in planned_braces:
                _, braces = planned_config(e, p)
                missing = [b for b in braces if b not in qidx]
                if missing:
                    raise ValueError(f"{name}: planned braces {missing} are not configured anywhere (no frame column)")
                planned_braces[p] = [qidx[b] for b in braces]
            human_state[f, planned_braces[p]] = True
        x0_rows[name] = {"ref": ref, "human": human_state}
        conflicts = known_free_conflicts(holds[m], lin["kind"], phase, e, free_rows)
        facts[name] = {"frames": n, "nonreal_frames": int((~real).sum()),
                       "phases": {str(p): int((phase[~real] == p).sum()) for p in sorted(set(phase[~real].tolist()))},
                       "real_frame_gap_vs_capture_m": gap_diff,
                       "nonreal_ref_closed": {q_pairs[q]: int(ref[~real, q].sum()) for q in range(len(q_pairs))
                                              if ref[~real, q].any()},
                       "nonreal_planned": {q_pairs[q]: int(human_state[~real, q].sum()) for q in range(len(q_pairs))
                                           if human_state[~real, q].any()},
                       "known_free_conflicts": conflicts}
    for m in range(n_human, M):
        name, entry = names[m], ext_clips[names[m]]
        src = entry["source_stem"]
        rows = x0_rows[src]
        index = lineage_of(lineage, name)["index"]
        if len(index) != lengths[m]:
            raise ValueError(f"{name}: lineage has {len(index)} frames, the library {lengths[m]}")
        ref, human_state = rows["ref"][index], rows["human"][index]
        frame["frame_pair_ref"][m, : lengths[m]] = torch.from_numpy(ref)
        frame["frame_pair_human"][m, : lengths[m]] = torch.from_numpy(human_state)
        frame["frame_pair_ok"][m, : lengths[m]] = torch.from_numpy(ref & human_state)

    payload = {**{k: v for k, v in human.items() if k not in SEG_KEYS + FRAME_KEYS},
               **seg, **frame,
               "release_id": release_id, "graph_sha256": ids.sha256_file(release_dir / "contact_graph.pt"),
               "package_sha256": graph["package_sha256"], "manifest_sha256": graph["manifest_sha256"],
               "motion_names": names, "hold_ids": list(graph["hold_ids"]),
               "seg_hold_index": graph["seg_hold_index"].clone(),
               "frame_len": torch.tensor(lengths, dtype=torch.long),
               "rules": {**human["rules"], **RULES}}
    summary = summarise_v3(payload, holds, names, ext_clips, anns, pidx, human_summary, n_human, facts)
    return payload, summary


def summarise_v3(payload: dict, holds: list, names: list, ext_clips: dict, anns: dict, pidx: dict,
                 human_summary: dict, n_human: int, facts: dict) -> dict:
    """v2's summary over every x0 motion (the synthetic holds counted with their inherited annotations), plus the
    human motions' own (v2's compiler on them: release v2's ``contact_targets.json``) and the synthetic ones'."""
    outside = {"all": collections.Counter(), "synthetic": collections.Counter()}
    free_unknown = dict(human_summary["free_unknown"])
    for m, stem in enumerate(names):
        if float(ext_clips[stem]["variant_s"]) != 0.0:
            continue
        for h in holds[m]:
            hid = h.get("inherits", h["hold_id"]) if m >= n_human else h["hold_id"]
            for contact, a in anns.get(hid, {}).items():
                if contact not in pidx:
                    outside["all"][(contact, a["target_role"])] += 1
                    if m >= n_human:
                        outside["synthetic"][(contact, a["target_role"])] += 1
            if m >= n_human and hid in human_summary["free_unknown"]:
                free_unknown[h["hold_id"]] = human_summary["free_unknown"][hid]
    everything = ct.summarise(payload, holds, names, ext_clips, free_unknown, outside["all"])
    syn_clips = {s: (c if s.startswith("SYN_") else {**c, "variant_s": -1.0}) for s, c in ext_clips.items()}
    synthetic = ct.summarise(payload, holds, names, syn_clips,
                             {h: v for h, v in free_unknown.items() if h.startswith("SYN_")}, outside["synthetic"])
    return {**everything, "human_x0": human_summary, "synthetic_x0": synthetic, "synthetic_frames": facts,
            "known_free_conflicts": sum(len(f["known_free_conflicts"]) for f in facts.values())}


def write(payload: dict, summary: dict, release_dir: Path) -> None:
    torch.save(payload, Path(release_dir) / "contact_targets.pt")
    (Path(release_dir) / "contact_targets.json").write_text(json.dumps(summary, indent=1, default=_json) + "\n")


def _json(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
