#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run G3 of expert_revist/graph_growth_2026_10_03/PLAN.MD (Step 3): fine-tune G1's AMP expert on a release that holds
# the synthesised edges (card E6's launcher; G1's run_expert_amp_ft.sh stays G1's and unchanged).
#
# Copied from run_expert_amp_ft.sh, argument for argument, except:
#   * NO DEFAULTS for RELEASE, EXPERIMENT or CHECKPOINT (G1's launcher derives them from the release hash; on v2 that
#     resumes G1, on v3 it looks for a checkpoint that does not exist). All three are required.
#   * It REFUSES an existing results/${EXPERIMENT}/last.ckpt (the name actually used, _smoke/_timing included) unless
#     RESUME=1, before any side effect: a relaunch under a used name would RESUME that run and silently ignore every
#     flag and library change. It also refuses a name a live train_agent.py is writing to.
#   * A WARM START with the PPO *and* AMP optimisation state (--warm-start-optimization-state; card E6 item 3): the
#     discriminator/disc-critic optimisers, the AMP reward normaliser and the weight calibration load too. RESUME=1
#     passes neither --checkpoint nor that flag: train_agent restores the run's own config.yaml/resolved_configs.pt
#     and loads its last.ckpt in full, ignoring everything else on the command line. RESUME=1 accepts only a run this
#     launcher started (its experiment file, the lineage flag in its config.yaml, this release's library).
#   * AMP at G1's weight from epoch 0: --amp-reward-w 0.5, start = full = 0, calibration off (G1 paid w = 0 for
#     200 epochs, then calibrated to 0.5). GP 10, discriminator batch 4,096, as G1.
#   * The style reward on synthetic clips x 0.5 (--amp-lineage-weights SYN_=0.5, PLAN.MD D4), and synthetic clips are
#     never demonstrations (--amp-demo-exclude-motions Scorpion_pose_or_vrischikasana-b SYN_; the flag REPLACES its
#     default list, so Scorpion -b is repeated). On a release with SYN_ clips it refuses a malformed rule, a rule that
#     matches nothing, a SYN_ clip no rule weights, and an exclusion list that lets a SYN_ x0 clip or Scorpion -b
#     into the demonstrations.
#   * Card E3's sampling: arrival 0.4, departure 0.2, pre-roll 0.5 s, the package motion prior, frozen actor/critic
#     observation normalisers (passed on every launch: the freeze is not in the state dict).
#   * The panel: G1's 10 plans plus every edge_*.json and nohijack_*.json in the release's plans/ (R4a; none on v2),
#     --viz-num-sequences = their count + 2. Every plan is resolved offline by SequenceViz's own loader against the
#     release's library and graph, in every mode; one that does not resolve, or loses a goal to --viz-max-seconds,
#     fails the launch (the panel itself would swap it for a graph-derived sequence with only a warning).
#   * EVAL_MAX_STEPS 2,250, raised to the record's training.eval_max_steps if that is larger.
#   * TIMING=1 keeps the real EVAL_MAX_STEPS: its one evaluation (the loaded checkpoint's, after epoch 1) then
#     allocates the real [motions x steps] evaluation buffers, so the peak memory it shows includes them.
#
#   RELEASE=... EXPERIMENT=... CHECKPOINT=... bash data/scripts/run_expert_graph_ft.sh             # G3: 6,000 epochs, wandb
#   ... MAX_EPOCHS=4500 bash data/scripts/run_expert_graph_ft.sh                                   # G3b from epoch_1500
#   ... SMOKE=1 bash data/scripts/run_expert_graph_ft.sh          # 256 envs x SMOKE_EPOCHS (4), eval/save every 2, no wandb
#   ... TIMING=1 bash data/scripts/run_expert_graph_ft.sh         # 4096 envs x TIMING_EPOCHS (50), no wandb, one real eval
#   ... CONFIG_ONLY=1 bash data/scripts/run_expert_graph_ft.sh    # write resolved configs and exit (no simulation)
#   ... RESUME=1 bash data/scripts/run_expert_graph_ft.sh         # continue results/${EXPERIMENT} (its own frozen config)
#   ... DRY_RUN=1 bash data/scripts/run_expert_graph_ft.sh        # every check, then print the command and exit 0
#
# The release-v2 smoke (no SYN_ clips there; a real stem exercises the lineage path):
#   RELEASE=holds_repaired_ftC_posefix.release_v2.a2dda5d2ac EXPERIMENT=zz_e6_smoke_v2 \
#   CHECKPOINT=results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/epoch_5000.ckpt AMP_LINEAGE=Crane_Crow=0.5 \
#   SMOKE=1 SMOKE_EPOCHS=10 bash data/scripts/run_expert_graph_ft.sh
#
# Grep the log for (card E6): "Warm start: restored AMP training state" (or "Warm start: no AMP training state
# loaded"), "Warm start: AMP weight target", "[amp lineage]", "[amp demos]".
# Watch (PLAN.MD Step 3 runbook): the human groups against the floors, eval/perf_group/edge_*, amp/reward_mean_syn
# against amp/reward_mean_human, amp/syn_sample_share, advantages/style_to_task_std, discriminator/{pos,agent}_acc,
# eval/curriculum/ess, the anchoring shares, times/last_epoch_seconds.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${PY:-python}
read -r -a PY_CMD <<< "${PY}"

die() { echo "run_expert_graph_ft.sh: $*" >&2; exit 1; }

SMOKE=${SMOKE:-0}
TIMING=${TIMING:-0}
CONFIG_ONLY=${CONFIG_ONLY:-0}
RESUME=${RESUME:-0}
DRY_RUN=${DRY_RUN:-0}
WANDB=${WANDB:-1}
for v in SMOKE TIMING CONFIG_ONLY RESUME DRY_RUN WANDB; do
  [[ "${!v}" =~ ^[01]$ ]] || die "${v} must be 0 or 1, got '${!v}'"
done
RELEASE=${RELEASE:-}
EXPERIMENT=${EXPERIMENT:-}
CHECKPOINT=${CHECKPOINT:-}
[[ -n "${RELEASE}" ]] || die "RELEASE is required (e.g. holds_repaired_ftC_posefix.release_v3.<id>); there is no default"
[[ -n "${EXPERIMENT}" ]] || die "EXPERIMENT is required (a new name under results/); there is no default"
[[ -n "${CHECKPOINT}" ]] || die "CHECKPOINT is required (G3: results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/epoch_5000.ckpt); there is no default"
NAME_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'
[[ "${EXPERIMENT}" =~ ${NAME_RE} ]] || die "EXPERIMENT '${EXPERIMENT}' must be a plain directory name"
[[ "${RELEASE}" =~ ${NAME_RE} ]] || die "RELEASE '${RELEASE}' must be a release id (a record in data/reference_curation/releases/)"
[[ "${SMOKE}" == "1" && "${TIMING}" == "1" ]] && die "SMOKE=1 and TIMING=1 are exclusive"
if [[ "${SMOKE}" == "1" ]]; then
  EXPERIMENT=${EXPERIMENT}_smoke
elif [[ "${TIMING}" == "1" ]]; then
  EXPERIMENT=${EXPERIMENT}_timing
fi

# --- Refusals: on the name actually used, before anything is read or written ---------------------------------- #
SAVE_DIR=results/${EXPERIMENT}
if [[ "${RESUME}" == "1" ]]; then
  [[ "${CONFIG_ONLY}" == "1" ]] && die "CONFIG_ONLY=1 with RESUME=1 would overwrite ${SAVE_DIR}'s frozen configs"
  [ -f "${SAVE_DIR}/last.ckpt" ] || die "RESUME=1 but ${SAVE_DIR}/last.ckpt does not exist (RESUME=1 passes no \
--checkpoint, so train_agent would start a fresh run from scratch)"
elif [ -e "${SAVE_DIR}/last.ckpt" ] || [ -L "${SAVE_DIR}/last.ckpt" ]; then
  die "${SAVE_DIR}/last.ckpt exists: train_agent would RESUME that run and ignore every flag and library change. \
Pick a new EXPERIMENT (a warm start), pass RESUME=1 to continue it, or move the directory away."
fi
LIVE=$(pgrep -af '[t]rain_agent\.py' || true)
if [[ -n "${LIVE}" ]] && grep -qF -e " --experiment-name ${EXPERIMENT} " -e " --experiment-name=${EXPERIMENT} " \
    <<< "$(sed 's/$/ /' <<< "${LIVE}")"; then
  die "a live train_agent.py is writing to ${SAVE_DIR}; refusing to launch into it"
fi
RECORD=data/reference_curation/releases/${RELEASE}.json
[ -f "${RECORD}" ] || die "missing release record: ${RECORD}"
[ -f "${CHECKPOINT}" ] || die "missing warm-start checkpoint: ${CHECKPOINT}"

EXPERIMENT_PATH=examples/experiments/mimic/mlp_goal_conditioned_amp.py
EVAL_MAX_STEPS_GIVEN=${EVAL_MAX_STEPS:+1}
NUM_ENVS=${NUM_ENVS:-4096}
BATCH_SIZE=${BATCH_SIZE:-16384}
UNIFORM_FRACTION=${UNIFORM_FRACTION:-0.8}
EVAL_EVERY=${EVAL_EVERY:-500}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-2250}       # raised below to the record's training.eval_max_steps if that is larger
SAVE_EVERY=${SAVE_EVERY:-500}
MAX_EPOCHS=${MAX_EPOCHS:-6000}               # ~19 h; the 1,500-epoch decision (G2) sits inside it
VIZ_EVERY=${VIZ_EVERY:-500}
VIZ_MAX_SECONDS=24.0
EXTRA=${EXTRA:-}
SUPPORT_RULE=${SUPPORT_RULE:-v2}
SUPPORT_WEIGHT=${SUPPORT_WEIGHT:--0.3}
SWING_WEIGHT=${SWING_WEIGHT:--0.3}
LEAN_WEIGHT=${LEAN_WEIGHT:--0.3}
AMP_W=${AMP_W:-0.5}                          # G1's calibrated weight
AMP_START=${AMP_START:-0}
AMP_FULL=${AMP_FULL:-0}
AMP_CAL_RATIO=${AMP_CAL_RATIO:-0}            # calibration off: AMP_W is the weight
AMP_DISC_BATCH=${AMP_DISC_BATCH:-4096}
AMP_GRAD_PENALTY=${AMP_GRAD_PENALTY:-10.0}
G1_DEMO_EXCLUDE="Scorpion_pose_or_vrischikasana-b"                           # the flag's default list (G1's)
AMP_LINEAGE=${AMP_LINEAGE-SYN_=0.5}                                          # unset -> default; '' -> no rules
AMP_DEMO_EXCLUDE=${AMP_DEMO_EXCLUDE-${G1_DEMO_EXCLUDE} SYN_}                  # replaces the flag's default list
SEGMENT_START_PROB=${SEGMENT_START_PROB:-0.4}
SEGMENT_END_PROB=${SEGMENT_END_PROB:-0.2}
SEGMENT_PRE_ROLL_S=${SEGMENT_PRE_ROLL_S:-0.5}
MOTION_PRIOR=${MOTION_PRIOR:-package}
FREEZE_OBS_NORMALIZERS=${FREEZE_OBS_NORMALIZERS:-True}
DRAG_REPORT=${DRAG_REPORT:-"Plow_Pose_or_Halasana_-b Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a \
Upward_Plank_Pose_or_Purvottanasana_-a Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a \
Low_Lunge_pose_or_Anjaneyasana_-a Peacock_Pose_or_Mayurasana_-a Plow_Pose_or_Halasana_-a \
Warrior_II_Pose_or_Virabhadrasana_II_-a"}
# G1's 10 panel plans, by file name in the release's plans/ (the edge_/nohijack_ ones are globbed below)
G1_PLANS=(
  fork_Warrior_II_Pose_or_Virabhadrasana_II.json
  fork_Tree_Pose_or_Vrksasana.json
  fork_Crane_Crow_Pose_or_Bakasana.json
  fork_Handstand_pose_or_Adho_Mukha_Vrksasana.json
  fork_Feathered_Peacock_Pose_or_Pincha_Mayuras.json
  fork_Supported_Headstand_pose_or_Salamba_Sirs.json
  fork_Side_Plank_Pose_or_Vasisthasana.json
  fork_Downward_Facing_Dog_pose_or_Adho_Mukha_S.json
  dwell_Warrior_II_Pose_or_Virabhadrasana_II_3s.json
  dwell_Warrior_II_Pose_or_Virabhadrasana_II_10s.json
)

if [[ "${SMOKE}" == "1" ]]; then
  NUM_ENVS=256; BATCH_SIZE=1024; EVAL_EVERY=2; EVAL_MAX_STEPS=150; SAVE_EVERY=2; VIZ_EVERY=0; WANDB=0
  MAX_EPOCHS=${SMOKE_EPOCHS:-4}; AMP_DISC_BATCH=1024
elif [[ "${TIMING}" == "1" ]]; then
  # EVAL_MAX_STEPS stays real: the loaded checkpoint's evaluation is the memory peak checklist item 6 reads.
  EVAL_EVERY=100000; SAVE_EVERY=100000; VIZ_EVERY=0; WANDB=0
  MAX_EPOCHS=${TIMING_EPOCHS:-50}
fi
for v in NUM_ENVS BATCH_SIZE EVAL_EVERY EVAL_MAX_STEPS SAVE_EVERY MAX_EPOCHS; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "${v} must be a positive integer, got '${!v}'"
done
[[ "${VIZ_EVERY}" =~ ^[0-9]+$ ]] || die "VIZ_EVERY must be a non-negative integer, got '${VIZ_EVERY}'"
MAX_STEPS=$((MAX_EPOCHS * NUM_ENVS * 32))
read -r -a LINEAGE_ARGS <<< "${AMP_LINEAGE}"
read -r -a DEMO_EXCLUDE_ARGS <<< "${AMP_DEMO_EXCLUDE}"

# --- The release, read-only: artifacts, the library's own stems, lineage, demonstrations, panel plans ---------- #
# Stems come from the package's motion_files (what the AMP component and the panel see at runtime), never from a
# record key; the lineage rules go through the agent's own parser and matcher; plans through SequenceViz's loader.
RUN_CONFIG=""
if [[ "${RESUME}" == "1" ]]; then RUN_CONFIG=${SAVE_DIR}/config.yaml; fi
REC_VARS=$("${PY_CMD[@]}" - "${RECORD}" "${RUN_CONFIG}" "${AMP_LINEAGE}" "${AMP_DEMO_EXCLUDE}" "${G1_DEMO_EXCLUDE}" \
  "${EXPERIMENT_PATH}" "${VIZ_MAX_SECONDS}" "${G1_PLANS[@]}" <<'PYEOF'
import contextlib, glob, json, logging, math, os, re, shlex, sys
from pathlib import Path

record, run_config, lineage, demo_exclude, g1_exclude, experiment_path, viz_max_s = sys.argv[1:8]
g1_plans = sys.argv[8:]
real_stdout = sys.stdout


def die(msg):
    print(f"run_expert_graph_ft.sh: {msg}", file=sys.stderr)
    sys.exit(1)


out = {}
with contextlib.redirect_stdout(sys.stderr):        # stdout carries only the assignments below
    logging.disable(logging.INFO)                   # import chatter; SequenceViz's warnings still pass
    rec = json.load(open(record))
    a = rec["artifacts"]
    out.update(MOTIONS=a["package"]["path"], GRAPH=a["graph"]["path"], PHYSICS=a["physics_tables"]["path"],
               TARGETS=a["contact_targets"]["path"], HOLD_MANIFEST=a["holds_extended"]["path"],
               PLANS_DIR=f"{rec['dir']}/plans", ROBOT=rec["robot"], PURPOSE=rec.get("purpose") or "n/a")
    for k in ("MOTIONS", "GRAPH", "PHYSICS", "TARGETS", "HOLD_MANIFEST"):
        if not os.path.isfile(out[k]):
            die(f"missing: {out[k]}")
    v = rec["training"]["eval_max_steps"]
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
        die(f"{record}: training.eval_max_steps {v!r} is not a positive number")
    out["MIN_EVAL_STEPS"] = int(math.ceil(v))

    # The library's stems, exactly as GoalConditionedAMPComponent._stems and SequenceViz read them.
    import torch
    try:
        package = torch.load(out["MOTIONS"], map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:                            # a non-zip (legacy) file cannot be mapped
        package = torch.load(out["MOTIONS"], map_location="cpu", weights_only=False)
    stems = [Path(str(f)).name.rsplit(".motion", 1)[0] for f in (package.get("motion_files") or [])]
    del package
    if not stems:
        die(f"{out['MOTIONS']} lists no motion_files: nothing to check the lineage or demonstrations against")
    listed = rec.get("motions")
    if isinstance(listed, (dict, list)) and listed and set(map(str, listed)) != set(stems):
        die(f"{record}'s motions ({len(listed)}) are not the package's {len(stems)} motion_files")
    x = re.compile(r"_x\d+s$")
    synthetic = [s for s in stems if s.startswith("SYN_")]

    # Variant list for the echo: a record key when R3 provides one, else the SYN_ x0 stems.
    variants = rec.get("variants")
    if not variants and isinstance(rec.get("synthetic"), dict):
        variants = rec["synthetic"].get("variants")
    if isinstance(variants, dict):
        variants = list(variants)
    elif isinstance(variants, list):
        variants = [str(i.get("name") or i.get("stem") or i) if isinstance(i, dict) else str(i) for i in variants]
    else:
        variants = None
    if not variants:
        variants = [s for s in synthetic if not x.search(s)]
    out["N_VARIANTS"], out["VARIANTS"] = len(variants), (", ".join(variants) if variants else "n/a")

    # Demonstrations as GoalConditionedAMP chooses them (before its minimum-length filter).
    excl = demo_exclude.split()
    demos = [s for s in stems if not x.search(s) and not any(p in s for p in excl)]
    out["N_MOTIONS"], out["N_SYN"], out["N_DEMOS"] = len(stems), len(synthetic), len(demos)
    leaked = [s for s in demos if s.startswith("SYN_")]
    if leaked:
        die(f"AMP_DEMO_EXCLUDE='{demo_exclude}' would make synthetic clips demonstrations: {' '.join(leaked)}")
    readmitted = [s for s in demos if any(p in s for p in g1_exclude.split())]
    if readmitted:
        die(f"AMP_DEMO_EXCLUDE='{demo_exclude}' drops G1's exclusion [{g1_exclude}] (the flag replaces its default "
            f"list), so these become demonstrations again: {' '.join(readmitted)}")

    # Lineage rules: the agent's own parser and first-match-wins matcher.
    from protomotions.agents.amp.goal_conditioned import lineage_match, parse_amp_lineage_weights
    try:
        rules = parse_amp_lineage_weights(lineage.split())
    except ValueError as exc:
        die(f"AMP_LINEAGE='{lineage}': {exc}")
    _, which = lineage_match(stems, rules)
    lines, dead = [], []
    for k, (pat, w) in enumerate(rules.items()):
        n = sum(j == k for j in which)
        lines.append(f"{pat}={w:g} -> {n} of {len(stems)} motions" + ("" if n else " (WARNING: none)"))
        if not n:
            dead.append(pat)
    out["LINEAGE_REPORT"] = "; ".join(lines) if lines else "none (every motion weighs 1.0)"
    if synthetic:
        if dead:
            die(f"AMP_LINEAGE='{lineage}': rule(s) {dead} match no motion (first match wins) on a release with "
                f"{len(synthetic)} SYN_ motions; a typo would train them at style weight 1.0")
        unweighted = [s for s, j in zip(stems, which) if s.startswith("SYN_") and j < 0]
        if unweighted:
            die(f"AMP_LINEAGE='{lineage}' leaves {len(unweighted)} SYN_ motions at style weight 1.0 (D4 weights them; "
                f"pass SYN_=1.0 for an unweighted arm): {' '.join(unweighted[:6])}")

    # RESUME: only a run this launcher started, on this release's library.
    if run_config:
        if not os.path.isfile(run_config):
            die(f"RESUME=1 but {run_config} does not exist")
        cfg = json.load(open(run_config))
        save_dir = os.path.dirname(run_config)
        if cfg.get("experiment_path") != experiment_path:
            die(f"RESUME=1: {save_dir} was launched with {cfg.get('experiment_path')}, not {experiment_path}; "
                f"this launcher resumes only its own runs")
        if "amp_lineage_weights" not in cfg:
            die(f"RESUME=1: {save_dir}/config.yaml has no amp_lineage_weights, so this launcher did not start it "
                f"(G1 trains on v2's library too)")
        if cfg.get("motion_file") != out["MOTIONS"]:
            die(f"RESUME=1: {save_dir} trains on {cfg.get('motion_file')}, not {Path(record).stem}'s "
                f"{out['MOTIONS']}; a resume keeps its own library (and crashes across libraries). Warm-start a new "
                f"EXPERIMENT instead.")

    # Panel plans: G1's 10 plus the release's edge_/nohijack_ plans, each resolved by SequenceViz's own loader.
    plans_dir = out["PLANS_DIR"]
    g1 = [f"{plans_dir}/{name}" for name in g1_plans]
    edge = sorted(glob.glob(f"{glob.escape(plans_dir)}/edge_*.json"))
    nohijack = sorted(glob.glob(f"{glob.escape(plans_dir)}/nohijack_*.json"))
    plans = g1 + edge + nohijack
    from protomotions.agents.evaluators.sequence_viz import SequenceVizConfig, SequenceVizRunner
    from protomotions.components.contact_graph import ContactGraph
    warnings = []

    class _Collect(logging.Handler):
        def emit(self, record_):
            warnings.append(record_.getMessage())

    logging.getLogger("protomotions.agents.evaluators.sequence_viz").addHandler(_Collect())
    runner = SequenceVizRunner.__new__(SequenceVizRunner)      # the loader only, no env
    runner.config = SequenceVizConfig(viz_every=1, num_sequences=len(plans) + 2, plan_files=plans,
                                      max_seconds=float(viz_max_s))
    runner.graph = ContactGraph.from_file(out["GRAPH"])
    runner.motion_names = stems
    for path in plans:
        if not os.path.isfile(path):
            die(f"missing panel plan: {path}")
        del warnings[:]
        sequence = runner._load_plan(path)
        if sequence is None:
            die(f"panel plan {path} does not resolve against {out['MOTIONS']} / {out['GRAPH']}: "
                f"{'; '.join(warnings) or 'all goals capped away'}")
        n_goals = len(json.load(open(path))["goals"])
        if len(sequence.goals) < n_goals:
            die(f"panel plan {path}: --viz-max-seconds {viz_max_s} keeps {len(sequence.goals)} of its {n_goals} goals")
    out["N_G1_PLANS"], out["N_EDGE_PLANS"], out["N_NOHIJACK_PLANS"] = len(g1), len(edge), len(nohijack)

for k, v in out.items():
    print(f"{k}={shlex.quote(str(v))}", file=real_stdout)
print("VIZ_PLANS=(" + " ".join(shlex.quote(p) for p in plans) + ")", file=real_stdout)
PYEOF
)
eval "${REC_VARS}"   # assigned first, so a failing reader stops the launcher (set -e)
if [[ "${SMOKE}" != "1" ]]; then
  if (( EVAL_MAX_STEPS < MIN_EVAL_STEPS )); then
    if [[ -n "${EVAL_MAX_STEPS_GIVEN}" ]]; then
      die "EVAL_MAX_STEPS ${EVAL_MAX_STEPS} does not cover the longest motion (${MIN_EVAL_STEPS} steps)"
    fi
    EVAL_MAX_STEPS=${MIN_EVAL_STEPS}
  fi
fi
VIZ_NUM_SEQUENCES=$(( ${#VIZ_PLANS[@]} + 2 ))
read -r -a DRAG_REPORT_ARGS <<< "${DRAG_REPORT}"
read -r -a EXTRA_ARGS <<< "${EXTRA}"

CMD=("${PY_CMD[@]}" protomotions/train_agent.py --robot-name "${ROBOT}" --simulator isaaclab
  --experiment-path "${EXPERIMENT_PATH}"
  --experiment-name "${EXPERIMENT}")
if [[ "${RESUME}" != "1" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}" --warm-start-optimization-state)
fi
CMD+=(--motion-file "${MOTIONS}"
  --hold-graph-file "${GRAPH}"
  --release-record "${RECORD}" --contact-targets "${TARGETS}"
  --goal-bodies all --sense-body-pair-contacts True
  --segment-start-prob "${SEGMENT_START_PROB}" --segment-end-prob "${SEGMENT_END_PROB}"
  --segment-pre-roll-s "${SEGMENT_PRE_ROLL_S}" --motion-prior "${MOTION_PRIOR}"
  --freeze-obs-normalizers "${FREEZE_OBS_NORMALIZERS}"
  --critic-future-steps 1 5 10 15
  --interval-schedule True
  --curriculum mixture --uniform-fraction "${UNIFORM_FRACTION}" --hold-manifest "${HOLD_MANIFEST}")
if [[ -n "${SUPPORT_RULE}" ]]; then CMD+=(--support-rule "${SUPPORT_RULE}"); fi
CMD+=(--eval-every "${EVAL_EVERY}" --eval-max-steps "${EVAL_MAX_STEPS}" --save-every "${SAVE_EVERY}"
  --viz-sequences-every "${VIZ_EVERY}" --viz-num-sequences "${VIZ_NUM_SEQUENCES}" --viz-max-seconds "${VIZ_MAX_SECONDS}"
  --viz-log-scalars False --viz-plan-files "${VIZ_PLANS[@]}"
  --support-penalty-weight "${SUPPORT_WEIGHT}" --support-clear-height 0.25
  --support-load-ref-frac 0.1 --support-ema-tau 0.25
  --physics-tables "${PHYSICS}"
  --swing-penalty-weight "${SWING_WEIGHT}" --swing-ema-tau 0.1
  --lean-penalty-weight "${LEAN_WEIGHT}" --lean-min-margin 0.03 --lean-scale 0.10
  --drag-report-motions "${DRAG_REPORT_ARGS[@]}"
  --amp-reward-w "${AMP_W}" --amp-w-start-epoch "${AMP_START}" --amp-w-full-epoch "${AMP_FULL}"
  --amp-calibrate-style-ratio "${AMP_CAL_RATIO}"
  --amp-disc-batch-size "${AMP_DISC_BATCH}" --amp-grad-penalty "${AMP_GRAD_PENALTY}"
  --amp-lineage-weights "${LINEAGE_ARGS[@]}"
  --amp-demo-exclude-motions "${DEMO_EXCLUDE_ARGS[@]}"
  --num-envs "${NUM_ENVS}" --batch-size "${BATCH_SIZE}" --training-max-steps "${MAX_STEPS}"
  --headless True)
if [[ "${WANDB}" != "0" ]]; then CMD+=(--use-wandb); fi
if [[ "${CONFIG_ONLY}" == "1" ]]; then CMD+=(--create-config-only); fi
CMD+=(--overrides env.ref_respawn_offset=0.005 "${EXTRA_ARGS[@]}")

MODE="fine-tune"
[[ "${SMOKE}" == "1" ]] && MODE="SMOKE (${MAX_EPOCHS} epochs)"
[[ "${TIMING}" == "1" ]] && MODE="TIMING (${MAX_EPOCHS} epochs)"
[[ "${CONFIG_ONLY}" == "1" ]] && MODE="${MODE}, CONFIG_ONLY"
echo "=== G3 graph ${MODE} on release ${RELEASE} -> results/${EXPERIMENT}"
echo "    release        : purpose ${PURPOSE}; ${N_VARIANTS} synthetic variants: ${VARIANTS}"
if [[ "${RESUME}" == "1" ]]; then
  echo "    RESUME         : ${SAVE_DIR}/last.ckpt with the run's own config.yaml + resolved_configs.pt (every flag below is ignored)"
else
  echo "    warm start     : ${CHECKPOINT} (+ PPO and AMP optimisation state)"
fi
echo "    experiment     : ${EXPERIMENT_PATH}"
echo "    robot / corpus : ${ROBOT} | ${MOTIONS} (${N_MOTIONS} motions, ${N_SYN} SYN_)"
echo "    terms          : support ${SUPPORT_WEIGHT} (EMA 0.25 s), swing ${SWING_WEIGHT} (0.1 s), lean ${LEAN_WEIGHT}; ref_respawn_offset 0.005"
echo "    AMP            : w ${AMP_W} from epoch ${AMP_START} (full by ${AMP_FULL}); calibration ratio ${AMP_CAL_RATIO} (0 = off); disc batch ${AMP_DISC_BATCH}, GP ${AMP_GRAD_PENALTY}"
echo "    lineage        : ${LINEAGE_REPORT}"
echo "    demonstrations : x0 clips without [${AMP_DEMO_EXCLUDE}] -> ${N_DEMOS} motions (before the length filter); SYN_ among them: 0"
echo "    sampling       : arrival ${SEGMENT_START_PROB}, departure ${SEGMENT_END_PROB}, pre-roll ${SEGMENT_PRE_ROLL_S} s; mixture ${UNIFORM_FRACTION} uniform, motion prior ${MOTION_PRIOR}; support rule ${SUPPORT_RULE:-v1 (flag not passed)}"
echo "    normalisers    : actor/critic frozen ${FREEZE_OBS_NORMALIZERS} (discriminator free)"
echo "    panel          : ${#VIZ_PLANS[@]} plans (${N_G1_PLANS} G1, ${N_EDGE_PLANS} edge, ${N_NOHIJACK_PLANS} no-hijack; all resolve) -> --viz-num-sequences ${VIZ_NUM_SEQUENCES}, every ${VIZ_EVERY}"
echo "    envs / batch   : ${NUM_ENVS} / ${BATCH_SIZE}; ${MAX_EPOCHS} epochs; eval every ${EVAL_EVERY} over ${EVAL_MAX_STEPS} steps (record: >= ${MIN_EVAL_STEPS}); save every ${SAVE_EVERY}"
echo "    command        : $(printf '%q ' "${CMD[@]}")"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: nothing launched."
  exit 0
fi

exec "${CMD[@]}"
