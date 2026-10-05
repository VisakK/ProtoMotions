"""G1 (the AMP warm-start fine-tune) against its pre-registered gates, and every evaluation's trajectory.

Reads the run's full TensorBoard dump and e15500's band (``expert_revist/expert56_v2_e15500/data``). Writes
``data/tb_eval_scalars.json`` (every ``eval/*`` series), ``data/tb_training_binned.json`` (training and AMP
diagnostics per 500-epoch bin) and ``data/gates.json``; prints the tables of README §2.

    ../env_isaaclab/bin/python expert_revist/run1_gap_analysis/dump_tb_scalars.py \\
        results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/lightning_logs/version_0/events.out.tfevents.* \\
        output/renderings/expert56_v2_amp_g1_e5000/tb_scalars.json
    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g1_amp_eval/g1_gates.py \\
        output/renderings/expert56_v2_amp_g1_e5000/tb_scalars.json
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E15500 = HERE.parents[1] / "expert56_v2_e15500/data"
LAST3 = (4000, 4500, 5000)

# (tag, short name, gate kind, threshold). Gates are PLAN.MD's G1 card: per group mean >= e15500 band mean - 0.035,
# jerk < 72.2 (improvement < 68.6), high-jerk < 1.30 %, drag <= 39.6 J, substitutions (force rule) <= 6.
ROWS = [
    ("eval/perf_group/single_leg_score", "single-leg", "ge", 0.948),
    ("eval/perf_group/inversion_score", "inversion", "ge", 0.843),
    ("eval/perf_group/arm_balance_score", "arm balance", "ge", 0.939),
    ("eval/perf_group/connective_score", "connective", "ge", 0.920),
    ("eval/normalized_jerk_mean", "jerk", "lt", 72.2),
    ("eval/high_jerk_frame_percentage_mean", "high-jerk %", "lt", 1.30),
    ("eval/drag/all_J", "drag J", "le", 39.6),
    ("eval/perf/substitution_holds_v2_x0", "subst. v2 x0", "le", 6),
    ("eval/success_rate", "success", None, None),
    ("eval/perf/score", "perf score", None, None),
    ("eval/perf/track", "track", None, None),
    ("eval/perf/hold", "hold", None, None),
    ("eval/perf/family_hold", "family hold", None, None),
    ("eval/perf/worst10_score", "worst-10", None, None),
    ("eval/gt_error/mean", "gt err m", None, None),
    ("eval/perf/support_realised_v2_x0", "realised v2 x0", None, None),
    ("eval/perf/holds_tracked_v2_x0", "tracked v2 x0", None, None),
    ("eval/perf/support_violation", "viol. v1", None, None),
    ("eval/perf/support_violation_v2", "viol. v2", None, None),
    ("eval/perf_v2/score", "perf score v2", None, None),
    ("eval/action_delta_mean_deg", "action delta deg", None, None),
    ("eval/action_rate_mean_rad_s", "action rate", None, None),
    ("eval/drag/top_J", "drag top J", None, None),
    ("eval/drag_group/single_leg_J", "drag single-leg", None, None),
    ("eval/drag_group/inversion_J", "drag inversion", None, None),
    ("eval/drag_group/arm_balance_J", "drag arm bal.", None, None),
    ("eval/drag_group/connective_J", "drag connective", None, None),
    ("eval/curriculum/ess", "curric. ESS", None, None),
]
TRAINING = ("info/episode_length", "env/terminate_mean", "env/raw_r/gt_rew_mean", "env/raw_r/gr_rew_mean",
            "env/raw_r/unwanted_support_rew_mean", "env/raw_r/swing_penalty_rew_mean",
            "env/raw_r/lean_penalty_rew_mean", "env/raw_r/diag_pair_target_met_mean",
            "env/raw_r/diag_known_free_load_n_mean", "env/raw_r/diag_swing_load_n_mean",
            "env/raw_r/diag_lean_error_m_mean", "actor/clip_frac", "actor/approx_kl", "critic/explained_variance",
            "amp/reward_w", "amp/reward_w_target", "discriminator/agent_acc", "discriminator/pos_acc",
            "discriminator/replay_acc", "rewards/unnormalized_amp_rewards", "rewards/amp_rewards",
            "advantages/disc_raw_std", "advantages/disc_std", "disc_critic/explained_variance",
            "discriminator/expert_logit_mean", "discriminator/agent_logit_mean", "times/last_epoch_seconds")
BIN = 500


def verdict(kind: str | None, thr, value: float) -> str:
    if kind is None:
        return ""
    ok = {"ge": value >= thr, "lt": value < thr, "le": value <= thr}[kind]
    return "pass" if ok else "FAIL"


def main(full: str) -> None:
    tb = json.load(open(full))
    evals = {t: v for t, v in tb.items() if t.startswith("eval/")}
    (HERE / "data/tb_eval_scalars.json").write_text(json.dumps(evals) + "\n")
    binned = {}
    for t in TRAINING:
        bins: dict[int, list[float]] = {}
        for step, value in tb.get(t, []):
            bins.setdefault(int(step) // BIN * BIN, []).append(value)
        binned[t] = {str(k): st.mean(v) for k, v in sorted(bins.items())}
    (HERE / "data/tb_training_binned.json").write_text(json.dumps({"bin_epochs": BIN, "tags": binned}, indent=1) + "\n")

    e15 = json.load(open(E15500 / "tb_eval_scalars.json"))
    band = json.load(open(E15500 / "eval_band_13000_15500.json"))
    epochs = [int(a) for a, _ in evals["eval/success_rate"]]
    score = dict((int(a), b) for a, b in evals["eval/perf/score"])
    best = max(score, key=score.get)
    print(f"evaluations: {epochs}")
    print(f"score_based.ckpt = epoch {best} (eval/perf/score {score[best]:.4f}; v1 score, the checkpoint rule)\n")

    out = {"epochs": epochs, "score_based_epoch": best, "rows": []}
    head = f"{'metric':18s} {'e15500':>8s} {'band':>14s} | " + " ".join(f"{e:>7d}" for e in epochs)
    print(head + f" | {'last3':>7s} gate")
    for tag, name, kind, thr in ROWS:
        series = dict((int(a), b) for a, b in evals.get(tag, []))
        if not series:
            continue
        e15_series = dict((int(a), b) for a, b in e15.get(tag, []))
        base = e15_series.get(15500)
        b = band.get(tag)
        last3 = [series[e] for e in LAST3 if e in series]
        mean3 = sum(last3) / len(last3) if last3 else float("nan")
        v = verdict(kind, thr, mean3)
        band_s = f"{b['band_mean']:.3f}+-{b['band_sd']:.3f}" if b else ""
        base_s = f"{base:8.3f}" if base is not None else f"{'-':>8s}"
        print(f"{name:18s} {base_s} {band_s:>14s} | " + " ".join(f"{series.get(e, float('nan')):7.3f}" for e in epochs)
              + f" | {mean3:7.3f} {v}")
        out["rows"].append(dict(tag=tag, name=name, gate=kind, threshold=thr, e15500=base,
                                band=b, series=series, last3=last3, last3_mean=mean3, verdict=v))
    (HERE / "data/gates.json").write_text(json.dumps(out, indent=1) + "\n")

    print("\ntraining / AMP diagnostics, 500-epoch bins")
    keys = sorted({k for t in TRAINING for k in binned[t]}, key=int)
    print(f"{'tag':44s} " + " ".join(f"{int(k):>7d}" for k in keys))
    for t in TRAINING:
        print(f"{t:44s} " + " ".join(f"{binned[t].get(k, float('nan')):7.4g}" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1])
