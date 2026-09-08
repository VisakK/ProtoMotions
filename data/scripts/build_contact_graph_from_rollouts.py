# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a contact-configuration graph from *simulated* expert rollouts.

``notes/Contact_config_def.MD`` builds the same kind of graph from **geometric
proximity** on the kinematic reference clips, and documents at length why that is
hard: the SMPL fit floats genuinely load-bearing parts by 2-16 cm, so no fixed
surface-distance threshold both catches a kneeling shank and rejects firefly's
deliberately hovering feet.  A whole static-support model exists to recover from
that.

This module answers the question a different way.  A trained Stage-1 tracker
reproduces the pose *in a physics simulator*, where "is this contact carrying
load" is not inferred at all -- PhysX reports it.  ``record_pressure_rollout.py``
already writes, per physics substep,

    ground_fz [Ts, B]      per-body vertical force against the terrain
    bb_force  [Ts, B, B]   magnitude of every body-body contact pair force

so a contact is *active* when the measured normal force crosses a threshold.  No
geometry, no float bias, no support model.  What the graph then describes is what
the **expert policies actually do**, which is precisely the behaviour the Stage-2
student is distilled from.

Zones, the adjacency mask and the orientation bins are taken unchanged from
``extract_contact_configs`` so the two graphs are directly comparable
(``--compare-geometric``).

Outputs (``--out-dir``):

* ``contact_graph.json`` -- nodes, edges and per-clip segment tables, with the
  motion each segment and each transition came from;
* ``contact_graph.pt``   -- the same thing as padded tensors keyed to a packaged
  ``MotionLib`` order, which is what :class:`protomotions.components.contact_graph.ContactGraph`
  loads at training time.

Usage::

    PYTHONPATH=. python data/scripts/build_contact_graph_from_rollouts.py \
      --motion-file data/smpl/yoga_yogi_student171.pt \
      --rollout-dir results/easy128_contact_rollouts \
      --rollout-dir results/hard29_pressure_rollouts \
      --rollout-dir results/singleleg14_pressure_rollouts \
      --out-dir data/smpl/yoga_contact_graph
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_contact_configs import (  # noqa: E402
    ADJACENT,
    ORIENT_BINS,
    ZONE_ORDER,
    ZONES,
    clean_active,
    config_string,
    orientation_bins,
    stabilize_bins,
    zone_pairs,
)

GRAVITY = 9.81


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in protomotions/tests/test_contact_graph.py)
# --------------------------------------------------------------------------- #
def pair_index() -> tuple[list[str], list[tuple[str, str] | tuple[str, None]]]:
    """Ordered pair names and their (zone_a, zone_b) decomposition.

    ``zone_b`` is ``None`` for a ground pair.  Order matches
    ``extract_contact_configs.zone_pairs()`` exactly, so a multi-hot vector built
    here indexes the same way as the geometric extractor's pair list.
    """
    names = zone_pairs()
    decomposed: list = []
    for name in names:
        if name.endswith(":G"):
            decomposed.append((name[:-2], None))
        else:
            a, b = name.split("+")
            decomposed.append((a, b))
    return names, decomposed


def zone_sim_indices(sim_body_names: list[str]) -> dict[str, list[int]]:
    """Map each zone to the *simulator* body indices it pools.

    Resolved by name.  ``pair_force_w`` and friends are indexed in simulator body
    order, which is **not** the COMMON/MJCF order the ``.motion`` fields use --
    ``notes/Pressure_supervision_design.MD`` §4 records a draft of that note
    reporting torso contact that does not exist because of exactly this.
    """
    lookup = {name: i for i, name in enumerate(sim_body_names)}
    out: dict[str, list[int]] = {}
    for zone in ZONE_ORDER:
        members = [lookup[b] for b in ZONES[zone] if b in lookup]
        if len(members) != len(ZONES[zone]):
            missing = [b for b in ZONES[zone] if b not in lookup]
            raise ValueError(f"zone {zone} bodies missing from the rollout: {missing}")
        out[zone] = members
    return out


def substeps_to_policy_steps(
    values: np.ndarray, substep_index: np.ndarray, reduce: str = "mean"
) -> np.ndarray:
    """Reduce a per-substep signal to one sample per policy step.

    ``record_pressure_rollout`` snapshots ``substep_index`` *before* stepping, so
    entry ``i >= 1`` closes the window opened by entry ``i - 1``: the substeps
    belonging to policy step ``i`` are ``[substep_index[i-1], substep_index[i])``.
    Entry 0 is the pre-roll snapshot and has no substeps behind it.

    ``reduce="mean"`` is right for forces, which fluctuate substep to substep and
    whose average over the control interval is what "is this contact loaded"
    should be asked of. ``reduce="last"`` is right for **orientations**: averaging
    quaternions component-wise is only approximately a rotation, is not
    renormalised, and silently degenerates if the backend flips a sign between
    substeps. The last substep is the state at the instant ``ctrl_motion_time[i]``
    was read, so it is both exact and the correctly aligned one.

    Returns an array of length ``len(substep_index) - 1`` aligned with
    ``ctrl_*[1:]``.
    """
    if substep_index.ndim != 1 or len(substep_index) < 2:
        raise ValueError("substep_index must have at least two entries")
    if reduce not in ("mean", "last"):
        raise ValueError(f"reduce must be 'mean' or 'last', got {reduce!r}")
    n_steps = len(substep_index) - 1
    out = np.zeros((n_steps, *values.shape[1:]), dtype=np.float64)
    total = values.shape[0]
    for i in range(n_steps):
        lo = min(int(substep_index[i]), max(total - 1, 0))
        hi = min(int(substep_index[i + 1]), total)
        if hi > lo:
            out[i] = values[lo:hi].mean(axis=0) if reduce == "mean" else values[hi - 1]
        elif total:
            # A window past the end of the recorded substeps (a truncated dump).
            # Repeating the last sample is wrong by a frame; writing zeros would
            # invent a contact-free frame, which the segmenter would believe.
            out[i] = values[lo]
    return out


def pair_forces_from_bodies(
    ground_fz: np.ndarray,
    bb_force: np.ndarray,
    zone_idx: dict[str, list[int]],
    decomposed: list,
) -> np.ndarray:
    """Per-frame force magnitude for every contact pair, in ``pair_index()`` order.

    Ground pairs sum the (signed) vertical terrain force over the zone's bodies
    and clamp at zero -- PhysX reports the reaction pushing the body up, so a
    negative entry is numerical noise rather than the floor pulling.

    Body-body pairs sum the symmetrised pair magnitudes over the zone block.  The
    matrix is symmetrised first (``0.5 * (M + Mᵀ)``) because PhysX measures the
    same interaction once at each of the two sensors; adding both rows unmodified
    would double count.
    """
    symmetric = 0.5 * (bb_force + np.swapaxes(bb_force, -1, -2))
    ground_pos = np.clip(ground_fz, 0.0, None)
    n_frames = ground_fz.shape[0]
    out = np.zeros((n_frames, len(decomposed)), dtype=np.float64)
    for p, (zone_a, zone_b) in enumerate(decomposed):
        rows = zone_idx[zone_a]
        if zone_b is None:
            out[:, p] = ground_pos[:, rows].sum(axis=1)
        else:
            cols = zone_idx[zone_b]
            out[:, p] = symmetric[:, rows][:, :, cols].sum(axis=(1, 2))
    return out


BODY_PAIR_IDENTITY_RULES = ("all", "load_path", "none")


def demote_supported_pairs(
    active: np.ndarray, decomposed: list, rule: str = "load_path"
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``active`` into identity pairs and demoted (secondary) pairs.

    ``rule`` selects how much of the body-body half enters **node identity**
    (i.e. is allowed to split segments and create nodes).  Demoted pairs are
    still measured, still stored per segment, and still reach the goal vector --
    they simply stop fragmenting the graph.

    ``"all"``
        Nothing is demoted; every body-body pair splits nodes.  The
        pre-round-3 rule, kept for ablation.

    ``"load_path"``
        A body-body pair is demoted on every frame where **both** of its zones
        have their own ground contact: each member is independently supported,
        so the mutual press is internal to the support set rather than a
        distinct support topology.  This is a topological rule, not a force
        floor, because force does not discriminate -- measured on the student44
        graph, the incidental ``L_FOOT+R_FOOT`` press while standing reaches
        40 % of body weight (median 18 %), well above genuine load paths like
        Eagle's hooked foot.  The cases that must survive all do: crow's shins
        on upper arms (arms not grounded), tree's foot on the standing thigh
        (thigh not grounded), Eagle's hook (the hooked foot not grounded),
        Tolasana's pressed thighs (neither grounded).

    ``"none"``
        **No** body-body pair enters identity: a node is exactly ``(ground
        support set, orientation bin)``.  Round 7_1's measurement is what
        motivates it -- on the student44h graph 51 of 102 nodes are singletons
        and 89 of the 104 pairs are body-body, so a marginal press fragments a
        pose into nodes no transition is ever observed between.  Firefly's hold
        is split across nodes 52/53/59 by a shank that rests on an arm for part
        of it, and its own frozen ``hold_`` clip -- byte-identical in pose to
        the node-52 segment, 0.00 m apart in the goal representation -- is
        labelled node 59.  Under this rule the body-body set moves to the
        segment, where it belongs: it describes *this* execution of the pose
        rather than defining a separate configuration.

    Returns ``(identity, demoted)`` boolean arrays of ``active``'s shape;
    ``identity | demoted == active`` and the two are disjoint.
    """
    if rule not in BODY_PAIR_IDENTITY_RULES:
        raise ValueError(
            f"body-pair identity rule must be one of {BODY_PAIR_IDENTITY_RULES}, "
            f"got {rule!r}"
        )
    if rule == "all":
        return active, np.zeros_like(active)

    is_body_pair = np.array([b is not None for _, b in decomposed])
    if rule == "none":
        demoted = active & is_body_pair[None, :]
        return active & ~demoted, demoted

    ground_pair_of = {
        zone_a: p for p, (zone_a, zone_b) in enumerate(decomposed) if zone_b is None
    }
    demoted = np.zeros_like(active)
    for p, (zone_a, zone_b) in enumerate(decomposed):
        if zone_b is None:
            continue
        both_grounded = (
            active[:, ground_pair_of[zone_a]] & active[:, ground_pair_of[zone_b]]
        )
        demoted[:, p] = active[:, p] & both_grounded
    return active & ~demoted, demoted


def force_hysteresis(force: np.ndarray, make_n: float, break_n: float) -> np.ndarray:
    """Latch a contact on above ``make_n``, release it below ``break_n``.

    Mirror image of ``extract_contact_configs.hysteresis_active``, which latches
    on a *shrinking* gap; here the observable grows when the contact forms.
    """
    if np.any(np.asarray(make_n) < np.asarray(break_n)):
        raise ValueError("make_n must be >= break_n for a force hysteresis")
    active = np.zeros(force.shape, dtype=bool)
    on = np.zeros(force.shape[1:], dtype=bool) if force.ndim > 1 else False
    for t in range(force.shape[0]):
        on = np.where(on, force[t] > break_n, force[t] > make_n)
        active[t] = on
    return active


def segment_runs(config_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of a constant config id as ``(start, end_inclusive, id)``."""
    runs: list[tuple[int, int, int]] = []
    if len(config_ids) == 0:
        return runs
    start = 0
    for t in range(1, len(config_ids) + 1):
        if t == len(config_ids) or config_ids[t] != config_ids[start]:
            runs.append((start, t - 1, int(config_ids[start])))
            start = t
    return runs


def absorb_short_runs(config_ids: np.ndarray, min_frames: int) -> np.ndarray:
    """Absorb runs shorter than ``min_frames`` into their neighbours.

    Keeps a total tiling of the clip: a dropped run is *reassigned*, never left
    as a hole, so segment lookup at any time always lands in a segment.  The
    absorbed run takes the id of whichever adjacent long run is longer, which
    avoids biasing every transient toward the past.
    """
    out = np.asarray(config_ids).copy()
    if min_frames <= 1 or len(out) == 0:
        return out
    changed = True
    while changed:
        changed = False
        runs = segment_runs(out)
        if len(runs) <= 1:
            break
        lengths = [end - start + 1 for start, end, _ in runs]
        order = np.argsort(lengths, kind="stable")
        for r in order:
            if lengths[r] >= min_frames:
                break
            start, end, _ = runs[r]
            left = runs[r - 1] if r > 0 else None
            right = runs[r + 1] if r + 1 < len(runs) else None
            if left is None and right is None:
                continue
            if left is None:
                donor = right
            elif right is None:
                donor = left
            else:
                len_l = left[1] - left[0] + 1
                len_r = right[1] - right[0] + 1
                donor = left if len_l >= len_r else right
            out[start : end + 1] = donor[2]
            changed = True
            break
    return out


def hold_frame(start: int, end: int, speed: np.ndarray, window: int = 1) -> int:
    """Most-static *window* in the middle 60 % of a segment -- the "held" pose.

    The boundaries of a segment are the make/break events; the pose worth handing
    a student as a goal is the one being *held*, so the search is restricted away
    from the transitions.

    Picking it needs care, and the reason is measurable. Inside a hold the
    body-COM speed is near-zero and nearly *flat*, so a bare ``argmin`` picks
    essentially at random: re-recording one clip with the same policy left every
    segment boundary identical to the frame and still moved the chosen hold by
    3.8 s, because PhysX is not deterministic across runs. Two things fix that,
    and both are needed --

    * ``window`` averages the speed over ~half a second, so the search is over
      the most-static *stretch* rather than the single luckiest frame;
    * among frames within 10 % of the minimum -- which inside a hold is most of
      them -- the one nearest the segment's centre wins, which is stable under
      noise and is the sensible reading of "held" anyway.
    """
    span = end - start
    lo = start + int(round(0.2 * span))
    hi = start + int(round(0.8 * span))
    if hi <= lo:
        return start + span // 2
    values = speed[lo : hi + 1]
    if window > 1 and len(values) >= window:
        values = np.convolve(values, np.ones(window) / window, mode="same")
    lowest = float(values.min())
    tolerance = lowest + 0.1 * (float(values.max()) - lowest) + 1e-9
    candidates = np.nonzero(values <= tolerance)[0]
    centre = 0.5 * (len(values) - 1)
    return lo + int(candidates[np.argmin(np.abs(candidates - centre))])


# --------------------------------------------------------------------------- #
# Per-clip pass
# --------------------------------------------------------------------------- #
def annotate_clip(
    path: Path, args, decomposed: list, motion_lib=None, motion_id: int | None = None
) -> dict:
    """Force-annotate one recorded rollout into contact segments.

    ``motion_lib``/``motion_id`` are optional but strongly recommended: with them
    the *hold* frame of each segment is chosen from the **reference** kinematics
    rather than from the rollout. Segment boundaries have to come from the
    rollout -- that is the whole point, they are where the measured load changes
    -- but the pose handed to a student comes from the reference, and choosing
    its timestamp from the reference makes it deterministic. Without this, two
    recordings of the same policy on the same clip agree on every boundary to the
    frame and still disagree on the hold by seconds, because PhysX is not
    deterministic across runs and a hold's speed profile is nearly flat.
    """
    z = np.load(path, allow_pickle=True)
    sim_body_names = [str(s) for s in z["body_names"]]
    zone_idx = zone_sim_indices(sim_body_names)

    substep_index = np.asarray(z["ctrl_substep_index"], dtype=np.int64)
    motion_time = np.asarray(z["ctrl_motion_time"], dtype=np.float64)[1:]
    track_err = np.asarray(z["ctrl_track_err"], dtype=np.float64)[1:]

    ground = substeps_to_policy_steps(np.asarray(z["ground_fz"]), substep_index)
    pairwise = substeps_to_policy_steps(np.asarray(z["bb_force"]), substep_index)
    quat_wxyz = substeps_to_policy_steps(
        np.asarray(z["body_quat_w"]), substep_index, reduce="last"
    )
    com_vel = substeps_to_policy_steps(np.asarray(z["body_com_vel_w"]), substep_index)

    n_steps = min(len(motion_time), ground.shape[0])
    motion_time, track_err = motion_time[:n_steps], track_err[:n_steps]
    ground, pairwise = ground[:n_steps], pairwise[:n_steps]
    quat_wxyz, com_vel = quat_wxyz[:n_steps], com_vel[:n_steps]

    body_mass = float(np.asarray(z["body_masses"]).sum())
    weight_n = body_mass * GRAVITY

    forces = pair_forces_from_bodies(ground, pairwise, zone_idx, decomposed)
    is_ground = np.array([b is None for _, b in decomposed])
    make = np.where(is_ground, args.ground_make_bw, args.body_make_bw) * weight_n
    brk = np.where(is_ground, args.ground_break_bw, args.body_break_bw) * weight_n
    active = force_hysteresis(forces, make, brk)

    dt = float(z["dt_ctrl"])
    fps = 1.0 / dt
    merge_n = max(int(round(args.merge_s * fps)), 1)
    dwell_n = max(int(round(args.dwell_s * fps)), 1)
    for p in range(active.shape[1]):
        active[:, p] = clean_active(active[:, p], merge_n, dwell_n)

    rule = getattr(args, "body_pair_identity", "load_path")
    identity, demoted = demote_supported_pairs(active, decomposed, rule=rule)
    # What the GOAL is allowed to name, independent of the identity rule: the
    # load-path definition, always. Coarsening identity is meant to stop a
    # marginal press *fragmenting* a pose into singleton nodes -- it is not a
    # decision to stop asking for firefly's thighs-on-upper-arms. Under
    # `load_path` this is exactly `identity`, which is what makes the whole
    # option a byte-level no-op for the graphs already trained on; under `all`
    # the union keeps identity's own extra pairs.
    # Body-body half only: a ground pair is identity under every rule, so the
    # goal's ground set must be exactly the node's. Without this restriction a
    # segment that absorbed a short neighbouring run (`absorb_short_runs`) can
    # pick up that run's ground pair -- measured on 4 of 411 segments, and it
    # would make the option a non-no-op for the rule the trained graphs use.
    is_body_pair = np.array([b is not None for _, b in decomposed])
    goal_active = (
        identity
        | (demote_supported_pairs(active, decomposed, "load_path")[0] & is_body_pair[None, :])
    )

    pelvis = sim_body_names.index("Pelvis")
    quat_xyzw = torch.from_numpy(quat_wxyz[:, pelvis][:, [1, 2, 3, 0]].astype(np.float32))
    raw_bins, _ = orientation_bins(quat_xyzw)
    bins = stabilize_bins(raw_bins, max(int(round(args.reorient_s * fps)), 1))

    # Config id per frame: the active *identity* pair set plus the orientation
    # bin. Demoted pairs stay measured but do not split segments or nodes.
    keys: dict[tuple, int] = {}
    ids = np.zeros(n_steps, dtype=np.int64)
    for t in range(n_steps):
        key = (tuple(np.nonzero(identity[t])[0].tolist()), int(bins[t]))
        ids[t] = keys.setdefault(key, len(keys))
    ids = absorb_short_runs(ids, max(int(round(args.min_node_dwell_s * fps)), 1))

    speed = np.linalg.norm(com_vel, axis=-1).mean(axis=-1)
    hold_speed, hold_speed_source = speed, "rollout"
    if motion_lib is not None and motion_id is not None:
        reference = motion_lib.get_motion_state(
            torch.full((n_steps,), int(motion_id), dtype=torch.long),
            torch.as_tensor(motion_time, dtype=torch.float32),
        )
        hold_speed = (
            reference.rigid_body_vel.norm(dim=-1).mean(dim=-1).cpu().numpy()
        )
        hold_speed_source = "reference"
    good = track_err <= args.track_err_tol
    hold_window = max(int(round(args.hold_smooth_s * fps)), 1)

    # Synthetic frozen-reference clips (make_hold_motions.py) hold one pose for
    # the whole clip, so their speed profile is exactly flat and the most-static
    # search would tie-break to the centre -- turning half the clip into a
    # zeroed-goal tail. Every timestamp is equally "the held pose" there, and a
    # late hold frame is what makes the clip a long sustained stay-command.
    prefix = getattr(args, "synthetic_hold_prefix", None)
    synthetic_hold = bool(prefix) and path.parent.name.startswith(prefix)

    inverse = {v: k for k, v in keys.items()}
    segments = []
    for start, end, cid in segment_runs(ids):
        pair_ids, obin = inverse[cid]
        if synthetic_hold:
            held = start + int(round(0.9 * (end - start)))
        else:
            held = hold_frame(start, end, hold_speed, window=hold_window)
        good_fraction = float(good[start : end + 1].mean())
        # Demoted pairs present on most of the segment are kept as secondary
        # attributes: real measured contact, deliberately not node identity.
        secondary_dwell = demoted[start : end + 1].mean(axis=0)
        secondary_ids = [
            int(p) for p in np.nonzero(secondary_dwell >= args.secondary_dwell_frac)[0]
            if p not in pair_ids
        ]
        # What the GOAL names for this segment: the identity set, plus every
        # goal-eligible pair this execution actually held for a majority of the
        # segment. Node identity coarsens under --body-pair-identity none; the
        # goal must not, or the coarsening would give back round 2's body-body
        # goal channel.
        goal_dwell = goal_active[start : end + 1].mean(axis=0)
        goal_ids = sorted(
            set(pair_ids)
            | {
                int(p)
                for p in np.nonzero(goal_dwell >= args.secondary_dwell_frac)[0]
                if is_body_pair[p]
            }
        )
        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "t_start": float(motion_time[start]),
                "t_end": float(motion_time[end]),
                "t_hold": float(motion_time[held]),
                "duration_s": float(motion_time[end] - motion_time[start] + dt),
                "pairs": [decomposed_name(p, decomposed) for p in pair_ids],
                "pair_ids": list(pair_ids),
                "ground_pairs": [
                    decomposed_name(p, decomposed) for p in pair_ids if not is_body_pair[p]
                ],
                "ground_pair_ids": [int(p) for p in pair_ids if not is_body_pair[p]],
                "secondary_pairs": [
                    decomposed_name(p, decomposed) for p in secondary_ids
                ],
                "secondary_pair_ids": secondary_ids,
                "goal_pairs": [decomposed_name(p, decomposed) for p in goal_ids],
                "goal_pair_ids": goal_ids,
                "orientation_bin": ORIENT_BINS[obin],
                "orientation_id": int(obin),
                "good_fraction": good_fraction,
                "trusted": good_fraction >= args.min_good_fraction,
                "mean_track_err": float(track_err[start : end + 1].mean()),
                "hold_speed": float(speed[held]),
                "mean_pair_force_n": {
                    decomposed_name(p, decomposed): float(
                        forces[start : end + 1, p].mean()
                    )
                    for p in sorted(set(goal_ids) | set(secondary_ids))
                },
            }
        )

    return {
        "clip": path.parent.name,
        "num_steps": n_steps,
        "dt": dt,
        "weight_n": weight_n,
        "hold_speed_source": hold_speed_source,
        "motion_length_s": float(z["motion_length_s"]),
        "good_fraction": float(good.mean()),
        "max_track_err": float(track_err.max()) if n_steps else float("nan"),
        "segments": segments,
    }


def decomposed_name(index: int, decomposed: list) -> str:
    zone_a, zone_b = decomposed[index]
    return f"{zone_a}:G" if zone_b is None else f"{zone_a}+{zone_b}"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def node_identity_pairs(segment: dict, node_identity: str) -> tuple[list, list]:
    """``(pair names, pair ids)`` that define this segment's NODE.

    ``"segment"`` keys a node by the same set that split the segment -- what
    every graph before round 7_1 did.  ``"ground"`` keys it by the ground
    support set alone, so a body-body pair still splits segments (and still
    reaches the goal through ``seg_contact``) but stops creating a node.

    The distinction is the whole point: on the student44h graph, keying by the
    full set makes 51 of 102 nodes singletons and 158 of 198 edges count-1,
    and firefly's hold is three nodes because a shank rests on an arm for part
    of it.  Merging the *segments* instead would be the blunt version of this
    and costs goals -- measured, 411 -> 344 goals and the body-body half of the
    goal vector halved (86 -> 43 goals naming one), because a pair characteristic
    of a sub-phase falls below the majority threshold of the merged whole.
    """
    if node_identity == "segment":
        return list(segment["pairs"]), list(segment["pair_ids"])
    if node_identity == "ground":
        return (
            list(segment.get("ground_pairs", segment["pairs"])),
            list(segment.get("ground_pair_ids", segment["pair_ids"])),
        )
    raise ValueError(f"unknown node identity {node_identity!r}")


def build_graph(
    clip_records: dict[str, dict],
    motion_names: list[str],
    pair_names: list[str],
    node_pair_dwell_frac: float = 0.5,
    node_identity: str = "segment",
):
    """Assemble node/edge tables in packaged-MotionLib motion-id order."""
    node_ids: dict[str, int] = {}
    node_rows: list[dict] = []
    per_motion: list[list[dict]] = [[] for _ in motion_names]
    edges: dict[tuple[int, int], dict] = {}

    for motion_id, name in enumerate(motion_names):
        record = clip_records.get(name)
        if record is None:
            continue
        previous_node = None
        for segment in record["segments"]:
            key_pairs, key_pair_ids = node_identity_pairs(segment, node_identity)
            key = config_string(key_pairs, segment["orientation_bin"])
            if key not in node_ids:
                node_ids[key] = len(node_rows)
                node_rows.append(
                    {
                        "key": key,
                        "pairs": sorted(key_pairs),
                        "pair_ids": sorted(key_pair_ids),
                        "orientation_bin": segment["orientation_bin"],
                        "orientation_id": segment["orientation_id"],
                        "total_dwell_s": 0.0,
                        "num_segments": 0,
                        "motions": [],
                        # Dwell-weighted occupancy of every pair the node's
                        # segments name in their goal vector. Body-body pairs
                        # can be per-segment (see --body-pair-identity), so a
                        # node-level vector -- which is what a *manual* goal
                        # gathers -- has to be a majority, not a union: a pair
                        # held on one 0.5 s pass out of 30 s does not describe
                        # the node.
                        "goal_pair_dwell_s": {},
                    }
                )
            node = node_ids[key]
            row = node_rows[node]
            if segment["trusted"]:
                row["total_dwell_s"] += segment["duration_s"]
                row["num_segments"] += 1
                if name not in row["motions"]:
                    row["motions"].append(name)
                for pair_id in segment.get("goal_pair_ids", segment["pair_ids"]):
                    row["goal_pair_dwell_s"][int(pair_id)] = (
                        row["goal_pair_dwell_s"].get(int(pair_id), 0.0)
                        + segment["duration_s"]
                    )
            per_motion[motion_id].append({**segment, "node": node, "config": key})

            if not segment["trusted"]:
                # A stretch the expert did not track breaks the chain. Joining
                # across it would record "A -> B" for a transition that was
                # really "A -> fell over -> B", which is not a transition the
                # student can be asked to reproduce.
                previous_node = None
                continue

            if previous_node is not None and previous_node[0] != node:
                edge = edges.setdefault(
                    (previous_node[0], node),
                    {"src": previous_node[0], "dst": node, "count": 0, "occurrences": []},
                )
                edge["count"] += 1
                edge["occurrences"].append(
                    {
                        "motion_id": motion_id,
                        "motion": name,
                        "t": segment["t_start"],
                        # Both endpoints in clip time: the student is asked to go
                        # *from* a held pose *to* the next held pose, and an edge
                        # without both cannot be replayed.
                        "t_hold_src": previous_node[1],
                        "t_hold_dst": segment["t_hold"],
                    }
                )
            previous_node = (node, segment["t_hold"])

    for row in node_rows:
        dwell = row["total_dwell_s"]
        row["goal_pair_ids"] = sorted(
            pair_id
            for pair_id, held in row["goal_pair_dwell_s"].items()
            if dwell <= 0.0 or held / dwell >= node_pair_dwell_frac
        )
        row["goal_pairs"] = [pair_names[i] for i in row["goal_pair_ids"]]
        # The raw occupancy is what makes the majority threshold auditable
        # later; keep it rounded rather than dropping it.
        row["goal_pair_dwell_frac"] = {
            pair_names[i]: round(held / dwell, 3)
            for i, held in sorted(row["goal_pair_dwell_s"].items())
            if dwell > 0.0
        }
        del row["goal_pair_dwell_s"]

    return node_ids, node_rows, per_motion, edges


def pack_tensors(node_rows, per_motion, pair_names, motion_names, min_lead_s):
    """Padded per-motion segment tables for GPU lookup at training time."""
    num_nodes = len(node_rows)
    num_pairs = len(pair_names)
    node_contact = torch.zeros(num_nodes, num_pairs, dtype=torch.float32)
    node_orient = torch.zeros(num_nodes, dtype=torch.long)
    for i, row in enumerate(node_rows):
        # Identity pairs always; plus whatever body-body pairs the node holds
        # for a majority of its dwell (build_graph). With the pre-round-7_1
        # rules these coincide, so the table is unchanged for those graphs.
        node_contact[i, row["pair_ids"]] = 1.0
        node_contact[i, row.get("goal_pair_ids", [])] = 1.0
        node_orient[i] = row["orientation_id"]

    kept = [[s for s in segments if s["trusted"]] for segments in per_motion]
    max_segments = max((len(s) for s in kept), default=0)
    max_segments = max(max_segments, 1)
    num_motions = len(motion_names)

    seg_node = torch.full((num_motions, max_segments), -1, dtype=torch.long)
    seg_start = torch.full((num_motions, max_segments), float("inf"))
    seg_end = torch.full((num_motions, max_segments), float("inf"))
    seg_hold = torch.full((num_motions, max_segments), float("inf"))
    seg_count = torch.zeros(num_motions, dtype=torch.long)
    # The contact target of a *scheduled* goal, per segment rather than per
    # node. Node identity may coarsen (--body-pair-identity none) so that a
    # pose stops being fragmented into singleton nodes no edge is ever observed
    # between; the goal must keep naming what this execution actually held, or
    # the coarsening would silently give back round 2's body-body goal channel.
    # ~1 MB at this corpus size (80 x 30 x 104 floats).
    seg_contact = torch.zeros(num_motions, max_segments, num_pairs, dtype=torch.float32)

    for motion_id, segments in enumerate(kept):
        segments = sorted(segments, key=lambda s: s["t_hold"])
        seg_count[motion_id] = len(segments)
        for k, segment in enumerate(segments):
            seg_node[motion_id, k] = segment["node"]
            seg_start[motion_id, k] = segment["t_start"]
            seg_end[motion_id, k] = segment["t_end"]
            seg_hold[motion_id, k] = segment["t_hold"]
            seg_contact[motion_id, k, segment.get("goal_pair_ids", segment["pair_ids"])] = 1.0

    return {
        "motion_names": motion_names,
        "pair_names": pair_names,
        "orientation_names": list(ORIENT_BINS),
        # Self-describing: the runtime pools per-body contact forces into these
        # zones, and reading the definition off the graph means it can never
        # drift from the one the graph was built with (and means nothing under
        # protomotions/ has to import from data/scripts/).
        "zone_order": list(ZONE_ORDER),
        "zone_bodies": {zone: list(ZONES[zone]) for zone in ZONE_ORDER},
        "node_keys": [row["key"] for row in node_rows],
        "node_contact": node_contact,
        "node_orient": node_orient,
        "seg_node": seg_node,
        "seg_contact": seg_contact,
        "seg_start": seg_start,
        "seg_end": seg_end,
        "seg_hold": seg_hold,
        "seg_count": seg_count,
        "min_lead_s": float(min_lead_s),
    }


# --------------------------------------------------------------------------- #
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=str, required=True,
                        help="packaged .pt whose motion order the tables are keyed to")
    parser.add_argument("--rollout-dir", type=str, action="append", required=True,
                        help="directory of <clip>/pressure_rollout.npz (repeatable)")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--ground-make-bw", type=float, default=0.03,
                        help="ground contact latches on above this fraction of body weight")
    parser.add_argument("--ground-break-bw", type=float, default=0.015)
    parser.add_argument("--body-make-bw", type=float, default=0.02,
                        help="body-body contact latches on above this fraction of body weight")
    parser.add_argument("--body-break-bw", type=float, default=0.01)
    parser.add_argument("--merge-s", type=float, default=0.15,
                        help="active runs separated by a shorter break are merged")
    parser.add_argument("--dwell-s", type=float, default=0.20,
                        help="active runs shorter than this are dropped")
    parser.add_argument("--reorient-s", type=float, default=0.30)
    parser.add_argument("--min-node-dwell-s", type=float, default=0.50,
                        help="configuration runs shorter than this are absorbed into a neighbour")
    parser.add_argument("--hold-smooth-s", type=float, default=0.50,
                        help="window used to find the most-static stretch of a segment; "
                             "a bare per-frame argmin is not stable across rollouts")
    parser.add_argument("--hold-from-rollout", action="store_true",
                        help="choose hold frames from the rollout's own dynamics instead "
                             "of the reference. Not recommended: PhysX is nondeterministic "
                             "across runs and a hold's speed profile is nearly flat, so the "
                             "chosen frame moves by seconds between two recordings of the "
                             "same policy on the same clip.")
    parser.add_argument("--synthetic-hold-prefix", type=str, default="hold_",
                        help="clips whose name starts with this are frozen-reference "
                             "hold clips (make_hold_motions.py); their hold frame is "
                             "placed at 90%% of each segment instead of searched, "
                             "because a frozen clip's speed profile is exactly flat. "
                             "Pass an empty string to disable.")
    parser.add_argument("--body-pair-identity", choices=BODY_PAIR_IDENTITY_RULES,
                        default="load_path",
                        help="how much of the body-body half enters NODE IDENTITY. "
                             "'all' = every body-body pair splits nodes (pre-round-3). "
                             "'load_path' (default) demotes only pairs whose two zones "
                             "are both independently grounded. 'none' demotes all of "
                             "them: a node is exactly (ground support set, orientation). "
                             "Demoted pairs are still measured, still stored per segment, "
                             "and still reach the goal through seg_contact -- they only "
                             "stop fragmenting the graph.")
    parser.add_argument("--keep-supported-pairs", action="store_true",
                        help="deprecated alias for --body-pair-identity all.")
    parser.add_argument("--node-identity", choices=("segment", "ground"),
                        default="segment",
                        help="what defines a NODE. 'segment' (default) keys it by the "
                             "same pair set that split the segment -- every graph before "
                             "round 7_1. 'ground' keys it by the ground support set plus "
                             "the orientation bin, so a body-body pair still splits "
                             "segments and still reaches the goal, but stops creating a "
                             "node. Prefer this over --body-pair-identity none, which "
                             "merges the SEGMENTS too and costs goals.")
    parser.add_argument("--secondary-dwell-frac", type=float, default=0.5,
                        help="a demoted pair is recorded on a segment (and enters that "
                             "segment's goal vector) when it is active for at least this "
                             "fraction of the segment.")
    parser.add_argument("--node-pair-dwell-frac", type=float, default=0.5,
                        help="a pair enters the NODE-level contact vector -- what a "
                             "manual/probe goal gathers -- when the node's segments hold "
                             "it for at least this fraction of the node's dwell. A "
                             "majority rather than a union, so a pair held on one short "
                             "pass does not end up describing the whole node.")
    parser.add_argument("--track-err-tol", type=float, default=0.50,
                        help="max body tracking error (m) for a frame to be trusted")
    parser.add_argument("--min-good-fraction", type=float, default=0.60,
                        help="fraction of trusted frames a segment needs to enter the graph")
    parser.add_argument("--min-lead-s", type=float, default=0.20,
                        help="stored default: how far ahead a goal must be to be selectable")
    parser.add_argument("--compare-geometric", type=str, default=None,
                        help="directory of the geometric extractor's per-clip JSON")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.keep_supported_pairs:
        args.body_pair_identity = "all"
    pair_names, decomposed = pair_index()
    print(f"body-pair segmentation rule: {args.body_pair_identity}   "
          f"node identity: {args.node_identity}")
    print(f"{len(pair_names)} contact pairs "
          f"({len(ZONE_ORDER)} ground + {len(pair_names) - len(ZONE_ORDER)} body-body, "
          f"{len(ADJACENT)} adjacent zone pairs masked)")

    packaged = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    motion_names = [Path(f).stem for f in packaged["motion_files"]]
    print(f"{len(motion_names)} motions in {args.motion_file}")

    # Either recorder's output is accepted: record_pressure_rollout.py writes a
    # superset (it also stores the ground contact manifold), and
    # record_contact_graph_rollouts.py writes exactly these channels much faster.
    files: dict[str, Path] = {}
    for directory in args.rollout_dir:
        for stem in ("contact_rollout.npz", "pressure_rollout.npz"):
            for npz in sorted(Path(directory).glob(f"*/{stem}")):
                files[npz.parent.name] = npz
    print(f"{len(files)} recorded rollouts found in {len(args.rollout_dir)} directories")

    missing = [n for n in motion_names if n not in files]
    if missing:
        print(f"WARNING: {len(missing)} motions have no rollout and will carry no "
              f"contact goals, e.g. {missing[:3]}")

    # The reference library, for choosing each segment's hold frame
    # deterministically (see annotate_clip).
    motion_lib = None
    if not args.hold_from_rollout:
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        print("loading the reference library for deterministic hold frames...")
        motion_lib = MotionLib(
            config=MotionLibConfig(motion_file=args.motion_file), device="cpu"
        )

    records: dict[str, dict] = {}
    for i, name in enumerate(motion_names):
        if name not in files:
            continue
        records[name] = annotate_clip(
            files[name], args, decomposed, motion_lib=motion_lib, motion_id=i
        )
        if (i + 1) % 20 == 0 or i + 1 == len(motion_names):
            print(f"  annotated {len(records)}/{len(motion_names)}")

    node_ids, node_rows, per_motion, edges = build_graph(
        records, motion_names, pair_names,
        node_pair_dwell_frac=args.node_pair_dwell_frac,
        node_identity=args.node_identity,
    )
    tensors = pack_tensors(node_rows, per_motion, pair_names, motion_names, args.min_lead_s)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, out_dir / "contact_graph.pt")

    graph_json = {
        "motion_file": args.motion_file,
        "thresholds": {
            k: getattr(args, k)
            for k in (
                "ground_make_bw", "ground_break_bw", "body_make_bw", "body_break_bw",
                "merge_s", "dwell_s", "reorient_s", "min_node_dwell_s",
                "track_err_tol", "min_good_fraction",
            )
        },
        "supported_pair_rule": {
            "all": "keep",
            "load_path": "demote_both_grounded",
            "none": "demote_all_body_pairs",
        }[args.body_pair_identity],
        "body_pair_identity": args.body_pair_identity,
        "node_identity": args.node_identity,
        "node_pair_dwell_frac": args.node_pair_dwell_frac,
        "secondary_dwell_frac": args.secondary_dwell_frac,
        "pair_names": pair_names,
        "orientation_names": list(ORIENT_BINS),
        "nodes": node_rows,
        "edges": sorted(edges.values(), key=lambda e: -e["count"]),
        "clips": {
            name: {
                "motion_id": motion_names.index(name),
                "good_fraction": records[name]["good_fraction"],
                "max_track_err": records[name]["max_track_err"],
                "segments": per_motion[motion_names.index(name)],
            }
            for name in records
        },
    }
    (out_dir / "contact_graph.json").write_text(json.dumps(graph_json, indent=1))

    singletons = sum(1 for r in node_rows if r["num_segments"] == 1)
    counts = [e["count"] for e in edges.values()]
    once = sum(1 for c in counts if c == 1)
    print(f"\nnode/edge shape:  {len(node_rows)} nodes ({singletons} singletons)   "
          f"{len(edges)} edges ({once} seen once, "
          f"{sum(1 for c in counts if c >= 6)} seen 6+ times)")

    trusted_counts = [int(c) for c in tensors["seg_count"]]
    recorded = [c for c in trusted_counts if c > 0]
    print(f"\nnodes {len(node_rows)}   edges {len(edges)}   "
          f"trusted segments {sum(trusted_counts)} over {len(recorded)} clips "
          f"(median {int(np.median(recorded)) if recorded else 0} per clip, "
          f"max {max(trusted_counts)}); {len(trusted_counts) - len(recorded)} clips have none")
    dwell = sorted(node_rows, key=lambda r: -r["total_dwell_s"])[:15]
    print("\ntop nodes by dwell:")
    for row in dwell:
        print(f"  {row['total_dwell_s']:8.1f}s  {row['num_segments']:4d} seg  "
              f"{len(row['motions']):3d} clips  {row['key'][:110]}")
    top_edges = sorted(edges.values(), key=lambda e: -e["count"])[:10]
    print("\ntop transitions:")
    for edge in top_edges:
        print(f"  x{edge['count']:3d}  {node_rows[edge['src']]['key'][:50]}"
              f"  ->  {node_rows[edge['dst']]['key'][:50]}")

    weak = [(n, r["good_fraction"]) for n, r in records.items() if r["good_fraction"] < 0.9]
    if weak:
        print(f"\n{len(weak)} clips with tracking failures "
              f"(< 90 % of frames within {args.track_err_tol} m):")
        for name, fraction in sorted(weak, key=lambda x: x[1])[:15]:
            print(f"  {100 * fraction:5.1f}%  {name}")

    if args.compare_geometric:
        compare_geometric(records, Path(args.compare_geometric))

    print(f"\nwrote {out_dir/'contact_graph.pt'} and {out_dir/'contact_graph.json'}")
    return 0


def compare_geometric(records: dict[str, dict], geometric_dir: Path) -> None:
    """Report agreement with the geometric extractor's dominant configuration.

    The two are not expected to match everywhere -- that is the point of building
    this one -- but on clips where the reference geometry is unambiguous they
    should, and a systematic disagreement is worth seeing before training on it.
    """
    print("\ndominant configuration, force-annotated vs geometric:")
    agree = total = 0
    for name, record in sorted(records.items()):
        path = geometric_dir / f"{name}.json"
        if not path.is_file():
            continue
        geometric = json.loads(path.read_text())
        by_dwell: dict[str, float] = defaultdict(float)
        for segment in geometric["segments"]:
            by_dwell[segment["config"]] += segment["duration_s"]
        if not by_dwell:
            continue
        geometric_top = max(by_dwell.items(), key=lambda kv: kv[1])[0]
        mine: dict[str, float] = defaultdict(float)
        for segment in record["segments"]:
            if segment["trusted"]:
                key = config_string(segment["pairs"], segment["orientation_bin"])
                mine[key] += segment["duration_s"]
        if not mine:
            continue
        my_top = max(mine.items(), key=lambda kv: kv[1])[0]
        total += 1
        same = my_top == geometric_top
        agree += int(same)
        if not same:
            print(f"  {name[:52]:52s}\n     force: {my_top[:100]}\n     geom : {geometric_top[:100]}")
    if total:
        print(f"  -> identical dominant configuration on {agree}/{total} clips "
              f"({100 * agree / total:.0f} %)")


if __name__ == "__main__":
    raise SystemExit(main())
