# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""eval_gate: pre-registered, self-invalidating experiment verdicts.

Motivation (``notes/Skill_graph_handstand_lessons_2.MD``): the v2 exit scored
composition 1.000 and was reported "fixed" while moving its limbs at 1.83x the
human's speed. The metric was blind and the reader (an LLM agent) aggregated
what it saw. This script removes every judgment call from the reporting path:

* Success criteria live in a JSON committed **before** training starts
  (``data/scripts/criteria/``). The report records whether that ordering holds
  (criteria mtime vs checkpoint mtime); post-hoc criteria cap the verdict.
* Every gate's pass/fail and the top-level verdict are computed **here, by
  code** -- the reading agent only relays them.
* Calibration is enforced: a zero-action rollout through the same pipeline
  must FAIL tracking, all artifacts must carry seeds and checkpoint SHA256s,
  and the composition sweep must cover the registered switch window.
  If calibration fails, the verdict is INVALID -- a broken pipeline cannot PASS.
* Composition gates are necessary, never sufficient: quality gates failing on
  top of passing composition gates yields MIXED, the verdict the v2 misreport
  would have received.
* The best verdict code can produce is NO_REGISTERED_FAILURE. PASS exists only
  behind ``--finalize approve``, which requires an interactive terminal and a
  typed phrase -- visual inspection has caught two failures every aggregate
  metric missed, and this keeps the human in the loop unforgeable-by-default.

No simulation here: this consumes artifacts from ``probe_quality.py`` and
``eval_composition.py``.

Usage::

    python data/scripts/eval_gate.py \
        --criteria data/scripts/criteria/edge_kickup_v3.json \
        --checkpoint results/edge_kickup_v3/final.ckpt \
        --quality results/edge_kickup_v3/quality.json \
        --quality-zero results/edge_kickup_v3/quality_zero.json \
        --composition results/edge_kickup_v3/composition.json \
        --baseline-quality results/edge_kickup_join/quality.json \
        --out results/edge_kickup_v3/eval_report.json

    # after watching the rollouts:
    python data/scripts/eval_gate.py --finalize approve --out results/edge_kickup_v3/eval_report.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

STANDING_CAVEAT = (
    "NO_REGISTERED_FAILURE means no pre-registered failure mode fired. It is not "
    "proof of naturalness or correctness: metrics are blind to failure modes nobody "
    "has registered yet. Human review of rendered rollouts is required before PASS."
)
FINALIZE_PHRASE = "I reviewed the rollouts"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, what, problems):
    if not path:
        problems.append(f"missing artifact: {what} (not provided)")
        return None
    if not os.path.exists(path):
        problems.append(f"missing artifact: {what} ({path} does not exist)")
        return None
    try:
        return json.load(open(path))
    except Exception as e:  # noqa: BLE001
        problems.append(f"unreadable artifact: {what} ({path}: {e})")
        return None


def finalize(args):
    report = json.load(open(args.out))
    if not sys.stdin.isatty():
        sys.exit("--finalize requires an interactive terminal: the sign-off must come "
                 "from a human, not from a script or an agent piping input.")
    print(f"Experiment: {report.get('experiment')}  verdict: {report.get('verdict')}")
    print(f"To sign off, type exactly: {FINALIZE_PHRASE}")
    typed = input("> ").strip()
    if typed != FINALIZE_PHRASE:
        sys.exit("phrase mismatch -- not finalized")
    report["human_review"] = {
        "status": "approved" if args.finalize == "approve" else "rejected",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tty": os.ttyname(sys.stdin.fileno()) if sys.stdin.isatty() else None,
    }
    if args.finalize == "approve" and report["verdict"] == "NO_REGISTERED_FAILURE":
        report["verdict"] = "PASS"
    elif args.finalize == "reject":
        report["verdict"] = "FAIL"
        report.setdefault("caveats", []).append("human review rejected the rollouts")
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"finalized: verdict={report['verdict']} -> {args.out}")


def gate_value(g, quality, composition, window, problems):
    """Resolve a gate's measured value from the artifacts. None = unresolvable."""
    src = g["source"]
    if src == "quality":
        if quality is None:
            return None
        v = quality.get("metrics", {}).get(g["metric"])
        if v is None:
            problems.append(f"gate {g['id']}: metric '{g['metric']}' absent from quality artifact")
        return v
    if src == "composition":
        if composition is None:
            return None
        rows = {r["switch_step"]: r for r in composition.get("sweep", [])}
        missing = [k for k in window if k not in rows]
        if missing:
            problems.append(f"gate {g['id']}: switch steps {missing} missing from sweep "
                            f"(window not covered -- INVALID)")
            return None
        vals = [rows[k].get(g["metric"]) for k in window]
        if any(v is None for v in vals):
            problems.append(f"gate {g['id']}: metric '{g['metric']}' null somewhere in window")
            return None
        agg = g.get("agg", "min_over_window")
        return {"min_over_window": min, "max_over_window": max,
                "mean_over_window": lambda x: sum(x) / len(x)}[agg](vals)
    problems.append(f"gate {g['id']}: unknown source '{src}'")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria")
    ap.add_argument("--checkpoint")
    ap.add_argument("--quality")
    ap.add_argument("--quality-zero", dest="quality_zero")
    ap.add_argument("--composition")
    ap.add_argument("--baseline-quality", dest="baseline_quality")
    ap.add_argument("--out", required=True)
    ap.add_argument("--finalize", choices=["approve", "reject"])
    args = ap.parse_args()

    if args.finalize:
        finalize(args)
        return

    if not (args.criteria and args.checkpoint):
        sys.exit("need --criteria and --checkpoint (or --finalize)")

    problems, caveats = [], [STANDING_CAVEAT]
    criteria = load_json(args.criteria, "criteria", problems)
    quality = load_json(args.quality, "quality", problems)
    quality_zero = load_json(args.quality_zero, "quality_zero (calibration)", problems)
    composition = load_json(args.composition, "composition", problems)
    baseline_q = json.load(open(args.baseline_quality)) if (
        args.baseline_quality and os.path.exists(args.baseline_quality)) else None

    ck_sha = sha256(args.checkpoint) if os.path.exists(args.checkpoint) else None
    if ck_sha is None:
        problems.append(f"checkpoint {args.checkpoint} does not exist")

    # --- pre-registration check: criteria must predate the checkpoint ---
    pre_registered = None
    if criteria is not None and os.path.exists(args.checkpoint):
        pre_registered = os.path.getmtime(args.criteria) <= os.path.getmtime(args.checkpoint)
        if not pre_registered:
            caveats.append("criteria file is NEWER than the checkpoint -- NOT pre-registered; "
                           "verdict capped at MIXED")

    # --- calibration: the pipeline must be able to fail ---
    calibration = {"pass": False, "checks": []}
    if quality_zero is not None:
        zerr = quality_zero.get("metrics", {}).get("track_err_mean")
        ok = quality_zero.get("action_mode") == "zero" and zerr is not None and zerr > 0.5
        calibration["checks"].append(
            {"check": "zero_action_must_fail_tracking", "track_err_mean": zerr, "pass": bool(ok)})
        if not ok:
            problems.append("calibration: zero-action rollout did NOT fail tracking "
                            f"(track_err_mean={zerr}) -- the pipeline cannot distinguish failure")
    if quality is not None:
        for field in ("seed", "checkpoint_sha256"):
            if quality.get(field) is None:
                problems.append(f"calibration: quality artifact missing '{field}'")
        if ck_sha and quality.get("checkpoint_sha256") not in (None, ck_sha):
            problems.append("calibration: quality artifact was produced from a DIFFERENT "
                            "checkpoint than --checkpoint (sha mismatch)")
        if quality.get("action_mode") != "policy":
            problems.append("calibration: --quality artifact is not an action_mode=policy run")
    if composition is not None:
        min_envs = (criteria or {}).get("min_num_envs", {}).get("composition", 0)
        if composition.get("num_envs", 0) < min_envs:
            problems.append(f"calibration: composition ran {composition.get('num_envs')} envs "
                            f"< registered minimum {min_envs}")
        if composition.get("checkpoint_sha256") not in (None, ck_sha):
            problems.append("calibration: composition artifact sha mismatch vs --checkpoint")
        if composition.get("seed") is None:
            caveats.append("composition artifact carries no seed (legacy script?) -- "
                           "rerun with --seed for reproducibility")
    calibration["pass"] = not any(p.startswith("calibration") or p.startswith("missing")
                                  or p.startswith("unreadable") for p in problems)

    # --- gates ---
    gates_out = []
    window = (criteria or {}).get("switch_window", [])
    n_task_fail = n_quality_fail = n_unresolved_binding = 0
    for g in (criteria or {}).get("gates", []):
        entry = {k: g.get(k) for k in ("id", "source", "metric", "agg", "op", "binding")}
        # threshold: absolute, or relative to the previous-best baseline artifact
        thr = g.get("threshold")
        if g.get("baseline_metric"):
            bv = (baseline_q or {}).get("metrics", {}).get(g["baseline_metric"])
            if bv is None:
                entry.update({"status": "unresolved",
                              "note": "baseline artifact/metric missing"})
                gates_out.append(entry)
                if g.get("binding"):
                    n_unresolved_binding += 1
                    caveats.append(f"binding gate {g['id']} unresolved: baseline missing")
                continue
            thr = bv * g.get("baseline_factor", 1.0)
            entry["baseline_value"] = bv
        entry["threshold"] = thr
        v = gate_value(g, quality, composition, window, problems)
        entry["value"] = v
        if v is None or thr is None:
            entry["status"] = "unresolved"
            gates_out.append(entry)
            if g.get("binding"):
                n_unresolved_binding += 1
                caveats.append(f"binding gate {g['id']} unresolved: value unavailable")
            continue
        ok = v <= thr if g["op"] == "<=" else v >= thr
        entry.update({"status": "evaluated", "pass": bool(ok)})
        gates_out.append(entry)
        if not ok:
            caveats.append(f"gate {g['id']} FAILED: {g['metric']} = {v:.4g} "
                           f"(required {g['op']} {thr:.4g}{', binding' if g.get('binding') else ', non-binding'})")
            if g.get("binding"):
                if g["source"] == "composition":
                    n_task_fail += 1
                else:
                    n_quality_fail += 1

    # --- verdict, computed here and only here ---
    invalid = (criteria is None or not calibration["pass"]
               or any("INVALID" in p for p in problems))
    if invalid:
        verdict = "INVALID"
    elif n_task_fail > 0:
        verdict = "FAIL"
    elif n_quality_fail > 0 or n_unresolved_binding > 0:
        verdict = "MIXED"
    else:
        verdict = "NO_REGISTERED_FAILURE"
    if verdict == "NO_REGISTERED_FAILURE" and pre_registered is False:
        verdict = "MIXED"

    report = {
        "schema": "eval_report/v1",
        "experiment": (criteria or {}).get("experiment"),
        "verdict": verdict,
        "verdict_semantics": {
            "INVALID": "pipeline/calibration broken -- no claim can be made",
            "FAIL": "a binding composition gate failed",
            "MIXED": "composition passed but a binding quality gate failed or was "
                     "unresolved (this is the verdict the v2 'fixed' misreport deserved)",
            "NO_REGISTERED_FAILURE": "all binding gates passed; awaiting human review",
            "PASS": "all binding gates passed AND a human signed off interactively",
        },
        "pre_registered": pre_registered,
        "criteria": {"path": args.criteria,
                     "sha256": sha256(args.criteria) if criteria is not None else None,
                     "registered_utc": (criteria or {}).get("registered_utc")},
        "checkpoint": {"path": args.checkpoint, "sha256": ck_sha,
                       "epoch": (quality or {}).get("checkpoint_epoch")},
        "artifacts": {name: ({"path": p, "sha256": sha256(p)} if p and os.path.exists(p) else None)
                      for name, p in [("quality", args.quality),
                                      ("quality_zero", args.quality_zero),
                                      ("composition", args.composition),
                                      ("baseline_quality", args.baseline_quality)]},
        "calibration": calibration,
        "problems": problems,
        "gates": gates_out,
        "caveats": caveats,
        "human_review": {"status": "pending"},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in
                      ("experiment", "verdict", "pre_registered", "problems", "caveats")}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
