#!/usr/bin/env python3
"""Card E3 acceptance: a seeded offline replay of the real ContactGraphMotionManager (CPU).

Builds the manager exactly as a run does -- release v2's packed library (``MotionLib`` on the
CPU) and its contact graph -- and resets every env many times:

1. **Configured shares.** At G3's setting (``segment_start_prob`` 0.4, ``segment_end_prob`` 0.2,
   pre-roll 0-0.5 s, ``init_start_prob`` 0.2), the drawn shares of arrival / departure / t = 0 /
   uniform starts against the shares the configuration implies on this graph (every clip has
   segments and an eligible departure, so 0.40 / 0.20 / 0.08 / 0.32). Pass: within 1 % (relative).
2. **Departure landings.** Every departure start lies in ``[t_end - 0.5 s, t_end]`` of an eligible
   segment of its clip, none in a clip's final segment window, none on (or after) a clip's last
   frame ``length - env_dt``.
3. **The logged path.** Per-step reset batches through ``pop_step_logs`` (what ``BaseEnv`` puts
   in the extras) into the agent's own meter (``TensorAverageMeterDict``, float16 storage, as
   ``record_rollout_step`` feeds it): every step must carry the same keys in the same order, and
   ``env/anchor/<kind>_resets / env/anchor/resets`` -- the two logged epoch means -- must equal
   the pooled share.
4. **Flags off = before E3.** G1's frozen motion-manager config (p_start 0.6, no
   ``segment_end_prob`` field) through the pre-E3 manager (from git) and the new one on the same
   seeds: identical motion ids, times and RNG state after every call.

Usage (repo root)::

    OMP_NUM_THREADS=1 ../env_isaaclab/bin/python \
        expert_revist/graph_growth_2026_10_03/e3_sampling/replay_departure_anchoring.py

Writes ``replay_departure_anchoring.json`` beside this file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.envs.motion_manager.config import ContactGraphMotionManagerConfig  # noqa: E402
from protomotions.envs.motion_manager.contact_graph_motion_manager import (  # noqa: E402
    KIND_ARRIVAL,
    KIND_DEPARTURE,
    KIND_T0,
    KIND_UNIFORM,
    START_KINDS,
    ContactGraphMotionManager,
)

RELEASE = REPO / "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
G1_RESOLVED = REPO / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/resolved_configs.pt"
PRE_E3_COMMIT = "5fdbd50f115d2e8da666c683a475d69c3ac43045"
ENV_DT = 4.0 / 120.0          # G1: IsaacLab 120 Hz, decimation 4
NUM_ENVS = 4096


def build(config, motion_lib, cls=ContactGraphMotionManager):
    return cls(config, NUM_ENVS, ENV_DT, torch.device("cpu"), motion_lib)


def expected_shares(manager, p_start, p_end, p_init):
    """Shares the configuration implies on this graph, given the clip-sampling weights."""
    w = manager.motion_weights / manager.motion_weights.sum()
    arrive = (manager.seg_count > 0).float()
    depart = manager._departure_eligible.any(dim=-1).float()
    anchored = p_start * arrive + p_end * depart
    return {
        "arrival": float((w * p_start * arrive).sum()),
        "departure": float((w * p_end * depart).sum()),
        "t0": float((w * (1.0 - anchored) * p_init).sum()),
        "uniform": float((w * (1.0 - anchored) * (1.0 - p_init)).sum()),
    }


def replay_shares(manager, rounds, seed):
    """``rounds`` resets of every env; returns counts per kind and the departure landing checks."""
    torch.manual_seed(seed)
    counts = torch.zeros(len(START_KINDS), dtype=torch.long)
    envs = torch.arange(NUM_ENVS)
    leads, end_gaps, outside, in_last, on_last_frame, segs_hit = [], [], 0, 0, 0, torch.zeros_like(
        manager._departure_eligible, dtype=torch.long)
    for _ in range(rounds):
        manager.sample_motions(envs)
        kind = manager.start_kind
        counts += torch.bincount(kind, minlength=len(START_KINDS))
        dep = kind == KIND_DEPARTURE
        if not bool(dep.any()):
            continue
        mids = manager.motion_ids[dep]
        t = manager.motion_times[dep]
        ends = manager.seg_end[mids]                                   # [n, S], +inf padded
        eligible = manager._departure_eligible[mids]
        lead = ends - t.unsqueeze(-1)                                  # >= 0 before the end
        inside = eligible & (lead >= -1e-5) & (lead <= 0.5 + 1e-5)
        hit = inside.any(dim=-1)
        outside += int((~hit).sum())
        first = inside.float().argmax(dim=-1)
        leads.append(lead.gather(-1, first.unsqueeze(-1)).squeeze(-1)[hit])
        segs_hit.index_put_((mids[hit], first[hit]), torch.ones_like(first[hit]), accumulate=True)
        last = (manager.seg_count[mids] - 1).clamp(min=0)
        last_end = ends.gather(-1, last.unsqueeze(-1)).squeeze(-1)
        in_last += int(((last_end - t) <= 0.5 + 1e-5).sum())
        length = manager._motion_lengths[mids]
        end_gaps.append(length - t)
        on_last_frame += int((t >= length - ENV_DT).sum())
    leads = torch.cat(leads)
    end_gaps = torch.cat(end_gaps)
    eligible_total = int(manager._departure_eligible.sum())
    return counts, {
        "departures": int(counts[KIND_DEPARTURE]),
        "inside_t_end_minus_0.5_to_t_end_of_an_eligible_segment": int(counts[KIND_DEPARTURE]) - outside,
        "outside_any_window": outside,
        "inside_a_clips_final_segment_window": in_last,
        "at_or_after_last_frame_length_minus_env_dt": on_last_frame,
        "lead_before_t_end_s": {"min": float(leads.min()), "p50": float(leads.median()),
                                "max": float(leads.max()), "mean": float(leads.mean())},
        "min_distance_to_clip_end_s": float(end_gaps.min()),
        "eligible_segments_drawn": int((segs_hit > 0).sum()),
        "eligible_segments": eligible_total,
        "draws_per_eligible_segment": {"min": int(segs_hit[manager._departure_eligible].min()),
                                       "max": int(segs_hit[manager._departure_eligible].max())},
    }


def replay_logged(manager, steps, seed, mean_resets=14.0):
    """Per-step reset batches (Poisson-sized, like a 4,096-env rollout) through pop_step_logs
    into the agent's meter, as one epoch of ``steps`` steps."""
    from protomotions.agents.utils.metering import TensorAverageMeterDict

    torch.manual_seed(seed)
    picks = torch.Generator().manual_seed(seed + 1)
    pooled = torch.zeros(len(START_KINDS))
    meter = TensorAverageMeterDict()      # BaseAgent.episode_env_tensors (float16 storage)
    manager.pop_step_logs()
    key_orders, empty_steps = set(), 0
    for _ in range(steps):
        n = int(torch.poisson(torch.tensor(mean_resets), generator=picks))
        envs = torch.randperm(NUM_ENVS, generator=picks)[:n]
        if n:
            manager.sample_motions(envs)
            pooled += torch.bincount(manager.start_kind[envs], minlength=len(START_KINDS)).float()
        else:
            empty_steps += 1
        logs = manager.pop_step_logs()
        key_orders.add(tuple(logs))
        # record_rollout_step's scalar branch: extras[key].float().flatten()
        meter.add({key: value.float().flatten() for key, value in logs.items()})
    means = meter.mean()
    total = float(pooled.sum())
    logged = {k: float(means[f"anchor/{k}_resets"]) / float(means["anchor/resets"]) for k in START_KINDS}
    exact = {k: float(pooled[i]) / total for i, k in enumerate(START_KINDS)}
    return {
        "steps": steps,
        "steps_without_resets": empty_steps,
        "distinct_key_orders_over_steps": len(key_orders),
        "keys": list(next(iter(key_orders))),
        "env/anchor/resets (mean resets per step)": float(means["anchor/resets"]),
        "env/anchor/<kind>_resets / env/anchor/resets": logged,
        "pooled_share": exact,
        "max_abs_diff_logged_vs_pooled": max(abs(logged[k] - exact[k]) for k in START_KINDS),
    }


def module_at(commit, rel_path, name):
    source = subprocess.run(["git", "show", f"{commit}:{rel_path}"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replay_flags_off(motion_lib, calls, seed):
    """G1's frozen config, pre-E3 manager vs the new one, identical seeds and env batches."""
    config = torch.load(G1_RESOLVED, map_location="cpu", weights_only=False)["env"].motion_manager
    old = module_at(PRE_E3_COMMIT, "protomotions/envs/motion_manager/contact_graph_motion_manager.py",
                    "cgmm_pre_e3")
    ref = build(config, motion_lib, old.ContactGraphMotionManager)
    new = build(config, motion_lib)
    picks = torch.Generator().manual_seed(seed)
    mismatched_calls, resets = 0, 0
    for step in range(calls):
        envs = (torch.arange(NUM_ENVS) if step == 0
                else torch.nonzero(torch.rand(NUM_ENVS, generator=picks) < 0.01).flatten())
        torch.manual_seed(seed + step)
        ref.sample_motions(envs)
        ref_rng = torch.get_rng_state()
        torch.manual_seed(seed + step)
        new.sample_motions(envs)
        same = (torch.equal(new.motion_ids, ref.motion_ids) and torch.equal(new.motion_times, ref.motion_times)
                and torch.equal(torch.get_rng_state(), ref_rng))
        mismatched_calls += int(not same)
        resets += len(envs)
    return {
        "config": {"segment_start_prob": config.segment_start_prob,
                   "segment_end_prob_in_pickle": "segment_end_prob" in config.__dict__,
                   "segment_end_prob_read": config.segment_end_prob,
                   "init_start_prob": config.init_start_prob, "pre_roll_s": config.pre_roll_s,
                   "segment_weighting": config.segment_weighting},
        "pre_e3_commit": PRE_E3_COMMIT,
        "calls": calls, "resets": resets, "calls_differing": mismatched_calls,
        "arrival_share_new_code": float((new.start_kind == KIND_ARRIVAL).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rounds", type=int, default=1000, help="full 4,096-env resets for the shares")
    parser.add_argument("--steps", type=int, default=20000, help="per-step batches for the logged path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()

    motion_lib = MotionLib(MotionLibConfig(motion_file=str(RELEASE / "motions.pt")), device="cpu")
    p_start, p_end, p_init, pre_roll = 0.4, 0.2, 0.2, 0.5
    config = ContactGraphMotionManagerConfig(
        init_start_prob=p_init, resample_on_reset=True, graph_file=str(RELEASE / "contact_graph.pt"),
        segment_start_prob=p_start, segment_end_prob=p_end, pre_roll_s=pre_roll, segment_weighting="uniform",
    )
    manager = build(config, motion_lib)
    expected = expected_shares(manager, p_start, p_end, p_init)
    counts, landing = replay_shares(manager, args.rounds, args.seed)
    total = int(counts.sum())
    shares = {}
    for k, name in enumerate(START_KINDS):
        observed = int(counts[k]) / total
        rel = abs(observed - expected[name]) / expected[name]
        shares[name] = {"expected": expected[name], "observed": observed, "count": int(counts[k]),
                        "abs_dev": observed - expected[name], "rel_dev": rel, "within_1pct": rel <= 0.01}
    # ~14 resets per step is a 4,096-env rank; 0.5 is a small rank where most steps reset nothing.
    logged_runs = {f"mean_{m:g}_resets_per_step": replay_logged(build(config, motion_lib), args.steps,
                                                                args.seed + 7, mean_resets=m)
                   for m in (14.0, 0.5)}
    flags_off = replay_flags_off(motion_lib, 200, args.seed + 1000)

    graph = {
        "motions": int(manager.seg_count.numel()),
        "segments": int(manager.seg_count.sum()),
        "departure_eligible_segments": int(manager._departure_eligible.sum()),
        "motions_with_an_eligible_departure": int(manager._departure_eligible.any(dim=-1).sum()),
        "excluded_segments_ending_within_env_dt_of_clip_end": int(manager.seg_count.sum())
        - int(manager._departure_eligible.sum()),
    }
    passed = (all(s["within_1pct"] for s in shares.values())
              and landing["outside_any_window"] == 0 and landing["inside_a_clips_final_segment_window"] == 0
              and landing["at_or_after_last_frame_length_minus_env_dt"] == 0
              and flags_off["calls_differing"] == 0
              and all(r["distinct_key_orders_over_steps"] == 1 and r["max_abs_diff_logged_vs_pooled"] < 1e-6
                      for r in logged_runs.values()))
    result = {
        "card": "E3 (graph_growth_2026_10_03/PLAN.MD)",
        "release": str(RELEASE.relative_to(REPO)),
        "config": {"segment_start_prob": p_start, "segment_end_prob": p_end, "init_start_prob": p_init,
                   "pre_roll_s": pre_roll, "segment_weighting": "uniform", "num_envs": NUM_ENVS,
                   "env_dt": ENV_DT, "rounds": args.rounds, "seed": args.seed},
        "graph": graph,
        "resets": total,
        "shares": shares,
        "departure_landing": landing,
        "logged_path": logged_runs,
        "flags_off_vs_pre_e3": flags_off,
        "pass": passed,
        "seconds": round(time.time() - started, 1),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for name, s in shares.items():
        print(f"{name:9s} expected {s['expected']:.4f} observed {s['observed']:.4f} "
              f"rel dev {100 * s['rel_dev']:.2f} %")
    print(f"departures {landing['departures']}: outside a window {landing['outside_any_window']}, "
          f"in a final window {landing['inside_a_clips_final_segment_window']}, "
          f"on the last frame {landing['at_or_after_last_frame_length_minus_env_dt']}")
    for name, logged in logged_runs.items():
        print(f"logged path ({name}): {logged['distinct_key_orders_over_steps']} key order over "
              f"{logged['steps']} steps ({logged['steps_without_resets']} without a reset); "
              f"logged vs pooled share max |diff| {logged['max_abs_diff_logged_vs_pooled']:.2e}")
    print(f"flags off vs pre-E3: {flags_off['calls_differing']} of {flags_off['calls']} calls differ")
    print(f"PASS={passed}  -> {args.out}")


if __name__ == "__main__":
    main()
