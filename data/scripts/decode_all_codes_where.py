# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where does the code space GO, and what does the prior think of the codes that hold?

Post-processes ``decode_all_codes.py``'s ``rollouts.npz``.  CPU only.

Two questions the hold/leave counts cannot answer on their own:

1. **Where do the departing codes go?**  Each terminal pose is named by its
   nearest trusted hold pose in the corpus -- the same nearest-node-member
   arithmetic ``score_probe_pose.py`` uses -- so "the decoder's marginal
   behaviour at a standing state is to fold toward four-point" becomes a ranked
   table with counts instead of an impression from the montage.

2. **Is the holding region low-ranked or invisible?**  The AR prior's full
   625-way distribution at the commanded state is captured by the rollout script.
   Joining it onto the per-code outcome gives the prior's total probability mass
   on the codes that hold, the best holding code's rank, and whether that code
   survives the deployed ``top_p`` nucleus at all.  A holding set that carries,
   say, 0.3 % of the mass is a different engineering problem from one the
   sampler cannot reach.

Usage::

    PYTHONPATH=. python data/scripts/decode_all_codes_where.py \\
      --npz output/decode_all_codes/last/rollouts.npz \\
      --graph data/smpl/yoga_contact_graph_student44h/contact_graph.json \\
      --motion-file data/smpl/yoga_yogi_student44h.pt \\
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml \\
      --summary output/decode_all_codes/last/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_pose_separation import CONDITIONABLE, GoalPoses, mean_body_distance  # noqa: E402
from decode_all_codes_report import normalize  # noqa: E402


def corpus_hold_poses(graph_json: str, poses: GoalPoses, body_ids, min_dwell_s=0.0):
    """Every trusted hold pose in the graph, as ``(label, node_key, [B,3])``."""
    graph = json.loads(Path(graph_json).read_text())
    names = [Path(f).stem for f in graph["motion_names"]] if "motion_names" in graph else None
    out = []
    for clip, record in graph["clips"].items():
        motion = None
        if names is not None and clip in names:
            motion = names.index(clip)
        else:
            hits = [i for i, n in enumerate(poses_names(poses)) if n == clip]
            motion = hits[0] if hits else None
        if motion is None:
            continue
        for seg in record["segments"]:
            if not seg.get("trusted", False):
                continue
            if seg.get("duration_s", 0.0) < min_dwell_s:
                continue
            out.append(
                (f"{clip}@{seg['t_hold']:.2f}", seg["config"],
                 poses.at(motion, float(seg["t_hold"]), body_ids))
            )
    return out


def poses_names(poses: GoalPoses):
    return getattr(poses, "_names_cache", None) or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--npz", required=True)
    parser.add_argument("--graph", required=True, help="contact_graph.json")
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--summary", required=True,
                        help="summary.json from the same run, for the state list")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="the deployed nucleus threshold, for the reachability test")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    data = np.load(args.npz)
    summary = json.loads(Path(args.summary).read_text())
    poses = GoalPoses(args.motion_file, args.mjcf)
    poses._names_cache = [Path(f).stem for f in
                          __import__("torch").load(args.motion_file, map_location="cpu",
                                                   weights_only=False)["motion_files"]]
    body_ids = poses.body_ids(CONDITIONABLE)
    bank = corpus_hold_poses(args.graph, poses, body_ids)
    bank_labels = [b[0] for b in bank]
    bank_configs = [b[1] for b in bank]
    bank_poses = np.stack([b[2] for b in bank])          # [M, 6, 3]
    print(f"corpus bank: {len(bank)} trusted hold poses")

    positions = data["positions"]
    root_rot = data["root_rot"]
    pose_err = data["pose_err"]
    state_of_env = data["state_of_env"]
    slot_of_env = data["slot_of_env"]
    num_codes = int(summary["num_codes"])
    controls = int(summary["controls"])
    threshold = float(summary["hold_threshold_m"])
    logits = data["prior_logits_step0"] if "prior_logits_step0" in data else None
    code_token = data["code_token"] if "code_token" in data else None

    payload = {"states": []}
    for s, state in enumerate(summary["states"]):
        sel = np.nonzero(state_of_env == s)[0]
        slots = slot_of_env[sel]
        code_envs = sel[slots < num_codes]
        order = np.argsort(slots[slots < num_codes])
        code_envs = code_envs[order]
        samp = sel[(slots >= num_codes) & (slots < num_codes + controls)]
        greedy = sel[slots >= num_codes + controls]

        terminal = normalize(positions[-1, code_envs], root_rot[-1, code_envs])
        terminal = terminal[:, body_ids]                 # [codes, 6, 3]
        dist = np.linalg.norm(
            terminal[:, None, :, :] - bank_poses[None], axis=-1
        ).mean(axis=-1)                                  # [codes, M]
        nearest = dist.argmin(axis=1)
        nearest_d = dist.min(axis=1)
        held = pose_err[:, code_envs].max(axis=0) <= threshold

        counts = {}
        for i, j in enumerate(nearest):
            key = (bank_labels[j], bank_configs[j])
            entry = counts.setdefault(key, dict(n=0, held=0, d=[]))
            entry["n"] += 1
            entry["held"] += int(held[i])
            entry["d"].append(float(nearest_d[i]))
        ranked = sorted(counts.items(), key=lambda kv: -kv[1]["n"])

        print(f"\n=== {state['plan']}  ({state['node_key']}) ===")
        print(f"held {int(held.sum())}/{num_codes} codes; "
              f"sampled {state['sampled_control']['held_rate']:.2f}, "
              f"greedy {state['greedy_control']['held_rate']:.2f}")
        print(f"{'where the terminal pose lands':<58}{'codes':>7}{'held':>6}{'d(m)':>8}")
        for (label, config), entry in ranked[: args.top]:
            print(f"{label[:56]:<58}{entry['n']:>7}{entry['held']:>6}"
                  f"{np.median(entry['d']):>8.3f}")

        record = dict(
            plan=state["plan"], node_key=state["node_key"],
            held=int(held.sum()),
            destinations=[
                dict(label=label, config=config, codes=entry["n"],
                     held=entry["held"], median_distance_m=float(np.median(entry["d"])))
                for (label, config), entry in ranked
            ],
        )

        if logits is not None and logits.size and code_token is not None:
            row = logits[int(code_envs[0])] if logits.ndim == 2 else logits
            probs = np.exp(row - row.max())
            probs = probs / probs.sum()
            code_probs = probs[code_token]
            hold_mass = float(code_probs[held].sum())
            rank = np.argsort(-code_probs)
            rank_of = np.empty_like(rank)
            rank_of[rank] = np.arange(len(rank))
            best_hold_rank = int(rank_of[held].min()) if held.any() else -1
            # nucleus membership at the deployed top_p
            sorted_p = code_probs[rank]
            keep = np.cumsum(sorted_p) <= args.top_p
            keep[0] = True
            in_nucleus = np.zeros(len(code_probs), dtype=bool)
            in_nucleus[rank[keep]] = True
            record["prior"] = dict(
                entropy_bits=float(-(probs * np.log2(np.clip(probs, 1e-12, None))).sum()),
                mass_on_holding_codes=hold_mass,
                best_holding_code_rank=best_hold_rank,
                holding_codes_in_top_p=int((held & in_nucleus).sum()),
                nucleus_size=int(in_nucleus.sum()),
                argmax_code_holds=bool(held[int(rank[0])]),
                argmax_prob=float(code_probs[rank[0]]),
            )
            print(f"prior at this state: entropy {record['prior']['entropy_bits']:.2f} bits, "
                  f"nucleus {record['prior']['nucleus_size']}/{num_codes} codes")
            print(f"  probability mass on the {int(held.sum())} holding codes: "
                  f"{hold_mass:.4f}   best holding code ranked #{best_hold_rank + 1}")
            print(f"  holding codes inside top-p={args.top_p}: "
                  f"{record['prior']['holding_codes_in_top_p']}"
                  f"   argmax code holds: {record['prior']['argmax_code_holds']}")
        payload["states"].append(record)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
