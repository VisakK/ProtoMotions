# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contact-target sidecar of a release (TODO C1; BUILD_PLAN Step 9, BodyFix Step 5).

The hold graph's goal vector is binary and the unwanted-support term reads its complement as "keep it off
the floor". A curated release knows more. This compiles, in the graph's own layout (``[motion, segment,
pair]`` and ``[motion, frame, pair]``), what labels v2 and gate v2 decided per hold, plus the per-frame and
known-negative evidence the runtime needs (``protomotions/envs/control/contact_targets.py`` loads it):

==========================  ==========  =================================================================
array                       shape       meaning
==========================  ==========  =================================================================
``seg_commanded``           [M,S,P]     the commanded contacts: configured minus the gate's masks (= the
                                        graph's ``seg_contact``)
``seg_configured``          [M,S,P]     labels v2's configured contacts
``seg_masked``              [M,S,P]     the gate's masks; a masked contact leaves every target
``seg_role``                [M,S,P]     each annotated contact's ``target_role`` (``ROLE_NAMES`` codes; 0 none)
``seg_critical``            [M,S,P]     B6's critical body-body contacts (the reviewer's self-consistency)
``seg_restored``            [M,S,P]     contacts the statics restoration (``labels_v2_restore``) put back
``seg_necessity``           [M,S,P]     statics v2's necessity of each configured contact (codes)
``seg_lp_load_n`` (+min/max) [M,S,P]    statics v2's representative load and interval (N; NaN where none)
``seg_ground_free``         [M,S,Z]     ground zones the human is *known* to keep free over the hold window
``seg_ground_free_share``   [M,S,Z]     the share of window frames the human evidence calls them separated
``seg_flags``               [M,S,F]     the gate's flags (``flag_names``); not release decisions
``seg_pose_role``           [M,S]       labels v2's ``pose_role`` (codes)
``frame_pair_ok``           [M,T,Q]     the reference closes configured body-body pair q (surface gap <= 1 cm)
                                        **and** the human's mesh is in contact, at the frame's x0 source
                                        frame: the per-frame eligibility of a pair target (TODO C1: the
                                        ``critical_partly_realised`` hold's pair is closed on 47 % of her
                                        contact frames)
``frame_pair_ref/_human``   [M,T,Q]     its two halves, for analysis
==========================  ==========  =================================================================

Rules, each measured on the release before it was fixed:

* **Known free.** A ground zone is a known negative of a hold when it is not configured and the human
  evidence (``sources.ground_source`` on capture store v4: the mesh, then the markers, then the mat) calls it
  separated on >= ``FREE_SHARE`` of the window's frames. On gate v2's 270 holds 3,233 of the 3,236 non-configured
  zone-holds are separated on every frame; the other 3 are never seen touching but are undecided on 9-79 % of
  their frames, and stay unknown (never charged). Configured zones are in contact on 100 % of their frames.
* **Per-frame pairs.** The reference's gap is capture store v4's ``avatar_pair_gap`` (plant v2, the Step 3
  references, ``retarget`` kernels); 1 cm is the closure band Step 3's pair requests and Step 4's statics use.
  Variant frames map to x0 frames through ``lineage.npz``; an inserted frame takes its exemplar's state.

Human-side stores are read the way ``capture_v4`` requires: in a plant-v1 process (no ``REFERENCE_PLANT=v2``),
plant v2 named explicitly.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from extract_contact_configs import ZONE_ORDER
from reference_curation import capture_v4, gate_v2, human_mesh as hm, ids, sources

REPO = ids.REPO
VERSION = 1
FREE_SHARE = 0.95
PAIR_CLOSE_M = 0.01
ROLE_NAMES = ("none", "required_support", "required_touch", "forbidden_support", "allowed", "incidental",
              "unresolved", "unspecified")
NECESSITY_NAMES = ("none", "required", "useful", "redundant", "undecided")
POSE_ROLE_NAMES = ("family", "standing", "variant", "preparation", "undecided")
FLAG_NAMES = tuple(gate_v2.RULES["flag"])


def read_annotations(labels_dir: Path) -> dict:
    """``{hold_id: {contact: annotation}}`` of a labels v2 folder."""
    anns: dict = collections.defaultdict(dict)
    for line in open(Path(labels_dir) / "annotations.jsonl"):
        a = json.loads(line)
        anns[a["hold_id"]][a["contact"]] = a
    return dict(anns)


def segment_holds(ext_clips: dict, graph: dict) -> list[list[dict]]:
    """Per motion (graph order), its holds in segment order, each checked against the graph's hold id."""
    out = []
    for m, stem in enumerate(graph["motion_names"]):
        holds = sorted(ext_clips[stem]["holds"], key=lambda h: float(h["t_hold"]))
        if len(holds) != int(graph["seg_count"][m]):
            raise ValueError(f"{stem}: {len(holds)} holds, the graph has {int(graph['seg_count'][m])} segments")
        for k, h in enumerate(holds):
            if graph["hold_ids"][int(graph["seg_hold_index"][m, k])] != h["hold_id"]:
                raise ValueError(f"{stem} segment {k}: the graph's hold id is not the manifest's {h['hold_id']}")
        out.append(holds)
    return out


def known_free(state: np.ndarray, hold: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(free [Z], separated share [Z])`` of an x0 hold from the human ground state ``[T, Z]`` (1/0/-1)."""
    window = state[int(hold["frame_start"]): int(hold["frame_end"]) + 1]
    share = (window == 0).mean(0)
    configured = {p[:-2] for p in hold["pairs_configured"] if p.endswith(":G")}
    free = np.array([z not in configured for z in ZONE_ORDER]) & (share >= FREE_SHARE)
    return free, share


def compile_targets(release_dir: Path, labels_dir: Path, x0_manifest: dict, release_id: str) -> tuple[dict, dict]:
    """``(payload, summary)`` of ``contact_targets.pt`` for the release in ``release_dir`` (graph, extended
    manifest and lineage already built)."""
    capture_v4.require_v1_process()
    graph = torch.load(release_dir / "contact_graph.pt", map_location="cpu", weights_only=False)
    ext = yaml.safe_load(open(release_dir / "holds_extended.yaml"))
    ext_clips = {c["stem"]: c for c in ext["clips"]}
    lineage = np.load(release_dir / "lineage.npz")
    anns = read_annotations(labels_dir)
    x0 = {c["stem"]: c for c in x0_manifest["clips"]}
    x0_holds = {h["hold_id"]: h for c in x0_manifest["clips"] for h in c["holds"]}

    names, pairs = list(graph["motion_names"]), list(graph["pair_names"])
    pidx = {p: i for i, p in enumerate(pairs)}
    M, S, P, Z = len(names), graph["seg_node"].shape[1], len(pairs), len(ZONE_ORDER)
    holds = segment_holds(ext_clips, graph)
    shape = (M, S, P)
    seg = {k: torch.zeros(shape, dtype=torch.bool) for k in ("commanded", "configured", "masked", "critical", "restored")}
    seg_role = torch.zeros(shape, dtype=torch.int8)
    seg_necessity = torch.zeros(shape, dtype=torch.int8)
    lp = {k: torch.full(shape, float("nan")) for k in ("load_n", "load_min_n", "load_max_n")}
    seg_free = torch.zeros(M, S, Z, dtype=torch.bool)
    seg_free_share = torch.zeros(M, S, Z)
    seg_flags = torch.zeros(M, S, len(FLAG_NAMES), dtype=torch.bool)
    seg_pose_role = torch.full((M, S), -1, dtype=torch.int8)
    outside_vocab = collections.Counter()

    # human-side and plant-v2 evidence per x0 clip (capture store v4; never rebuilt here)
    evidence = {}
    for stem in x0:
        rec = capture_v4.load(stem, rebuild=False)
        state, _ = sources.ground_source(rec)
        evidence[stem] = {"ground": state, "pair_human": np.asarray(rec["human_pair_state"]),
                          "pair_gap": np.asarray(rec["avatar_pair_gap"], np.float64)}
        if list(rec.meta["pair_order"]) != list(hm.PAIR_NAMES):
            raise ValueError(f"{stem}: capture v4's pair order is not human_mesh.PAIR_NAMES")
    free_unknown = {}
    for m, stem in enumerate(names):
        for k, h in enumerate(holds[m]):
            hid = h["hold_id"]
            for key, field in (("commanded", "pairs"), ("configured", "pairs_configured")):
                seg[key][m, k, [pidx[p] for p in h[field]]] = True
            seg["masked"][m, k, [pidx[p] for p in h["gate"]["masks"]]] = True
            is_x0 = float(ext_clips[stem]["variant_s"]) == 0.0
            for contact, a in anns.get(hid, {}).items():
                if contact not in pidx:
                    if a["in_configuration"]:
                        raise ValueError(f"{hid}: configured contact {contact} is outside the graph's vocabulary")
                    outside_vocab[(contact, a["target_role"])] += int(is_x0)
                    continue
                i = pidx[contact]
                seg_role[m, k, i] = ROLE_NAMES.index(a["target_role"])
                seg["critical"][m, k, i] = bool(a.get("critical")) and a["target_role"] == "required_touch" and \
                    bool(a["in_configuration"])
                seg["restored"][m, k, i] = bool(a.get("restored"))
                st = a.get("statics") or {}
                if st.get("necessity"):
                    seg_necessity[m, k, i] = NECESSITY_NAMES.index(st["necessity"])
                for key in lp:
                    if st.get(key) is not None:
                        lp[key][m, k, i] = float(st[key])
            for f in h["gate"]["flags"]:
                seg_flags[m, k, FLAG_NAMES.index(f)] = True
            seg_pose_role[m, k] = POSE_ROLE_NAMES.index(h["labels"]["pose_role"])
            src = ext_clips[stem]["source_stem"]
            free, share = known_free(evidence[src]["ground"], x0_holds[hid])
            seg_free[m, k] = torch.from_numpy(free)
            seg_free_share[m, k] = torch.from_numpy(share.astype(np.float32))
            unknown = [z for z, f, s in zip(ZONE_ORDER, free, share)
                       if not f and f"{z}:G" not in h["pairs_configured"]]
            if unknown:
                free_unknown[hid] = {z: round(float(share[ZONE_ORDER.index(z)]), 3) for z in unknown}

    # per-frame eligibility of the configured body-body pairs
    q_pairs = sorted({pairs[i] for i in torch.nonzero(seg["configured"].any(0).any(0)).flatten().tolist()
                      if "+" in pairs[i]})
    q_cols = [hm.PAIR_NAMES.index(p) for p in q_pairs]
    lengths = [int(n) for n in graph["motion_num_frames"]]
    T = max(lengths)
    frame = {k: torch.zeros(M, T, len(q_pairs), dtype=torch.bool) for k in ("ok", "ref", "human")}
    for m, stem in enumerate(names):
        index = lineage[f"{stem}.index"]
        if len(index) != lengths[m]:
            raise ValueError(f"{stem}: lineage has {len(index)} frames, the library {lengths[m]}")
        ev = evidence[ext_clips[stem]["source_stem"]]
        ref = ev["pair_gap"][index][:, q_cols] <= PAIR_CLOSE_M
        human = ev["pair_human"][index][:, q_cols] == 1
        frame["ref"][m, : lengths[m]] = torch.from_numpy(ref)
        frame["human"][m, : lengths[m]] = torch.from_numpy(human)
        frame["ok"][m, : lengths[m]] = torch.from_numpy(ref & human)

    from protomotions.utils import plant_identity

    payload = {
        "kind": "contact_targets", "version": VERSION, "release_id": release_id,
        "labels_id": x0_manifest["labels"]["labels_id"], "gate_id": x0_manifest["labels"]["gate_id"],
        plant_identity.KEY: graph.get("plant_sha256"), "graph_sha256": ids.sha256_file(release_dir / "contact_graph.pt"),
        "package_sha256": graph["package_sha256"], "manifest_sha256": graph["manifest_sha256"],
        "fps": int(graph["fps"]), "motion_names": names, "pair_names": pairs, "zone_order": list(ZONE_ORDER),
        "hold_ids": list(graph["hold_ids"]), "seg_hold_index": graph["seg_hold_index"].clone(),
        "role_names": list(ROLE_NAMES), "necessity_names": list(NECESSITY_NAMES),
        "pose_role_names": list(POSE_ROLE_NAMES), "flag_names": list(FLAG_NAMES),
        "seg_commanded": seg["commanded"], "seg_configured": seg["configured"], "seg_masked": seg["masked"],
        "seg_role": seg_role, "seg_critical": seg["critical"], "seg_restored": seg["restored"],
        "seg_necessity": seg_necessity, "seg_lp_load_n": lp["load_n"], "seg_lp_load_min_n": lp["load_min_n"],
        "seg_lp_load_max_n": lp["load_max_n"], "seg_ground_free": seg_free, "seg_ground_free_share": seg_free_share,
        "seg_flags": seg_flags, "seg_pose_role": seg_pose_role,
        "frame_pair_names": q_pairs, "frame_pair_slots": torch.tensor([pidx[p] for p in q_pairs], dtype=torch.long),
        "frame_pair_ok": frame["ok"], "frame_pair_ref": frame["ref"], "frame_pair_human": frame["human"],
        "frame_len": torch.tensor(lengths, dtype=torch.long),
        "rules": {"known_free": f"not configured and the human evidence (sources.ground_source) separated on >= "
                                f"{FREE_SHARE} of the hold window's frames",
                  "frame_pair_ok": f"reference surface gap <= {PAIR_CLOSE_M} m (capture v4 avatar_pair_gap) and the "
                                   "human mesh in contact (human_pair_state == 1), at the x0 source frame"},
    }
    summary = summarise(payload, holds, names, ext_clips, free_unknown, outside_vocab)
    return payload, summary


def summarise(payload: dict, holds: list, names: list, ext_clips: dict, free_unknown: dict,
              outside_vocab: collections.Counter) -> dict:
    """Counts over the x0 motions (one per hold; the variants repeat them) and the per-frame coverage."""
    x0 = [m for m, s in enumerate(names) if float(ext_clips[s]["variant_s"]) == 0.0]
    pairs = payload["pair_names"]
    ground = torch.tensor([p.endswith(":G") for p in pairs])
    live = torch.zeros(payload["seg_role"].shape[:2], dtype=torch.bool)
    for m in x0:
        live[m, : len(holds[m])] = True
    L = live.unsqueeze(-1)
    role = payload["seg_role"].long()
    roles = {name: {"ground": int(((role == i) & L & ground).sum()), "pair": int(((role == i) & L & ~ground).sum())}
             for i, name in enumerate(ROLE_NAMES) if i}
    cfg_bb = payload["seg_configured"] & L & ~ground
    partly = {}
    q_slots = payload["frame_pair_slots"].tolist()
    for m in x0:
        for k, h in enumerate(holds[m]):
            f0, f1 = int(h["frame_start"]), int(h["frame_end"])
            for q, slot in enumerate(q_slots):
                if not bool(payload["seg_configured"][m, k, slot]):
                    continue
                human = payload["frame_pair_human"][m, f0:f1 + 1, q]
                ref = payload["frame_pair_ref"][m, f0:f1 + 1, q]
                share = float((human & ref).sum()) / max(int(human.sum()), 1)
                if share < gate_v2.PARTLY_REALISED:
                    partly[f"{h['hold_id']}:{pairs[slot]}"] = round(share, 3)
    return {
        "holds": int(live.sum()),
        "roles": roles,
        "configured_body_body": int(cfg_bb.sum()),
        "critical": int((payload["seg_critical"] & L).sum()),
        "restored": int((payload["seg_restored"] & L).sum()),
        "masked": {pairs[i]: int(payload["seg_masked"][..., i][live].sum()) for i in range(len(pairs))
                   if bool(payload["seg_masked"][..., i][live].any())},
        "flags": {f: int(payload["seg_flags"][..., i][live].sum()) for i, f in enumerate(FLAG_NAMES)},
        "pose_roles": {r: int((payload["seg_pose_role"][live] == i).sum()) for i, r in enumerate(POSE_ROLE_NAMES)},
        "known_free_zone_holds": int(payload["seg_ground_free"][live].sum()),
        "free_unknown": free_unknown,
        "frame_pairs": list(payload["frame_pair_names"]),
        "frame_pair_partly_realised": partly,
        "annotations_outside_the_pair_vocabulary": {f"{c} ({r})": n for (c, r), n in sorted(outside_vocab.items())},
    }
