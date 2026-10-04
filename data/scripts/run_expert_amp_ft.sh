#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run G1 of expert_revist/graph_growth_2026_10_03/PLAN.MD: the release-v2 expert, warm-started from epoch 15,500,
# fine-tuned with an AMP style reward (card E2's experiment, examples/experiments/mimic/mlp_goal_conditioned_amp.py).
#
# Everything else is run_expert_release_v2.sh's run, argument for argument: the same release record (every artifact
# checked by sha256), the same reward stack, curriculum, evaluator and viz panel. What changes:
#   * a WARM START (--checkpoint epoch_15500.ckpt --warm-start-optimization-state, a new experiment name): actor,
#     critic, normalisers, PPO optimisers, advantage EMA and the task-reward normaliser load; the discriminator
#     networks start random; counters, evaluator state and motion weights start fresh. Never a resume: a resume
#     re-reads the old run's frozen configs and would ignore all of this.
#   * AMP: style weight 0 for epochs 0-199 (the discriminator trains against the loaded policy for free), then a
#     linear ramp to its target at epoch 500; no discriminator termination; x0-only demonstrations. The target is
#     CALIBRATED at epoch 200 (PLAN.MD §1.2 "by advantage scale, not by guess"): the weight at which the style
#     advantage std is AMP_CAL_RATIO (0.25, inside the plan's 0.2-0.35) of the task advantage std, from the median
#     ratio of epochs 100-199, clamped to [0.05, 0.5]. The 50-epoch timing run measured the ratio at w = 0.1 as
#     0.07-0.08, i.e. the plan's starting guess sits far below its own band. AMP_CAL_RATIO=0 uses AMP_W as given.
#     Gradient penalty 10: the timing run's agent accuracy passed 0.9 by epoch ~11 (PLAN.MD: "raise to 10 if agent
#     accuracy stays above 0.9"); the offline check separates the e15500 rollouts from the reference at AUROC 1.0.
#   * --support-rule v2 (card E1): the curriculum's sampling scores support substitutions; every eval/perf* key
#     stays v1 so the numbers compare with e15500's band.
#
#   bash data/scripts/run_expert_amp_ft.sh                # G1: 5,000 epochs (~14 h), wandb
#   CONTROL=1 bash data/scripts/run_expert_amp_ft.sh      # G1's matched control: the same continuation, no AMP
#   SMOKE=1 bash data/scripts/run_expert_amp_ft.sh        # 256 envs x 4 epochs, AMP on from epoch 1, no wandb
#   TIMING=1 bash data/scripts/run_expert_amp_ft.sh       # 4096 envs x 50 epochs, no wandb, near-free evals
#   CONFIG_ONLY=1 bash data/scripts/run_expert_amp_ft.sh  # write resolved configs and exit (no simulation)
#
# Watch (PLAN.MD G1): eval/perf_group/*_score against e15500's band (accept >= 0.948 / 0.843 / 0.939 / 0.920),
# eval jerk/drag, eval/perf/substitution_holds_v2*, discriminator/{pos_acc,agent_acc} in 0.55-0.85,
# rewards/unnormalized_amp_rewards in 0.3-1.0, advantages/style_to_task_std in 0.2-0.35 once w is on
# (advantages/style_to_task_std_at_target predicts it during the w = 0 phase), amp/reward_w{,_target}, amp/parity_*.
# The first evaluation runs right after epoch 0 (a loaded checkpoint always evaluates); before the ramp it should
# reproduce e15500's group scores within 0.01.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${PY:-python}

SMOKE=${SMOKE:-0}
TIMING=${TIMING:-0}
CONTROL=${CONTROL:-0}
CONFIG_ONLY=${CONFIG_ONLY:-0}
RELEASE=${RELEASE:-holds_repaired_ftC_posefix.release_v2.a2dda5d2ac}
RECORD=data/reference_curation/releases/${RELEASE}.json
[ -f "${RECORD}" ] || { echo "missing release record: ${RECORD}" >&2; exit 1; }
SHORT=${RELEASE##*.}
CHECKPOINT=${CHECKPOINT:-results/smpl_yogi_v2_expert56_${SHORT}/epoch_15500.ckpt}
[ -f "${CHECKPOINT}" ] || { echo "missing warm-start checkpoint: ${CHECKPOINT}" >&2; exit 1; }
if [[ "${CONTROL}" == "1" ]]; then
  EXPERIMENT=${EXPERIMENT:-smpl_yogi_v2_expert56_ctrl_ft_${SHORT}}
  EXPERIMENT_PATH=examples/experiments/mimic/mlp_goal_conditioned.py
else
  EXPERIMENT=${EXPERIMENT:-smpl_yogi_v2_expert56_amp_ft_${SHORT}}
  EXPERIMENT_PATH=examples/experiments/mimic/mlp_goal_conditioned_amp.py
fi
NUM_ENVS=${NUM_ENVS:-4096}
BATCH_SIZE=${BATCH_SIZE:-16384}
UNIFORM_FRACTION=${UNIFORM_FRACTION:-0.8}
EVAL_EVERY=${EVAL_EVERY:-500}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-2250}       # >= the record's training.eval_max_steps (the longest motion)
SAVE_EVERY=${SAVE_EVERY:-500}
MAX_EPOCHS=${MAX_EPOCHS:-5000}               # ~14 h; checkpoints every 500 epochs, ~0.7 GB each
VIZ_EVERY=${VIZ_EVERY:-500}
WANDB=${WANDB:-1}
EXTRA=${EXTRA:-}
SUPPORT_RULE=${SUPPORT_RULE:-v2}
SUPPORT_WEIGHT=${SUPPORT_WEIGHT:--0.3}
SWING_WEIGHT=${SWING_WEIGHT:--0.3}
LEAN_WEIGHT=${LEAN_WEIGHT:--0.3}
AMP_W=${AMP_W:-0.1}
AMP_START=${AMP_START:-200}
AMP_FULL=${AMP_FULL:-500}
AMP_DISC_BATCH=${AMP_DISC_BATCH:-4096}
AMP_GRAD_PENALTY=${AMP_GRAD_PENALTY:-10.0}
AMP_CAL_RATIO=${AMP_CAL_RATIO:-0.25}
AMP_CAL_WINDOW=${AMP_CAL_WINDOW:-100}
DRAG_REPORT=${DRAG_REPORT:-"Plow_Pose_or_Halasana_-b Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a \
Upward_Plank_Pose_or_Purvottanasana_-a Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a \
Low_Lunge_pose_or_Anjaneyasana_-a Peacock_Pose_or_Mayurasana_-a Plow_Pose_or_Halasana_-a \
Warrior_II_Pose_or_Virabhadrasana_II_-a"}

if [[ "${SMOKE}" == "1" ]]; then
  EXPERIMENT=${EXPERIMENT}_smoke
  NUM_ENVS=256; BATCH_SIZE=1024; EVAL_EVERY=2; EVAL_MAX_STEPS=150; SAVE_EVERY=2; MAX_EPOCHS=4; VIZ_EVERY=0; WANDB=0
  AMP_START=2; AMP_FULL=3; AMP_DISC_BATCH=1024; AMP_CAL_WINDOW=2   # calibrate + exercise the style reward in 4 epochs
elif [[ "${TIMING}" == "1" ]]; then
  EXPERIMENT=${EXPERIMENT}_timing
  EVAL_EVERY=100000; EVAL_MAX_STEPS=150; SAVE_EVERY=100000; MAX_EPOCHS=50; VIZ_EVERY=0; WANDB=0
fi
MAX_STEPS=$((MAX_EPOCHS * NUM_ENVS * 32))

eval "$(${PY} - "${RECORD}" <<'PYEOF'
import json, sys
rec = json.load(open(sys.argv[1]))
a = rec["artifacts"]
print(f"MOTIONS={a['package']['path']}")
print(f"GRAPH={a['graph']['path']}")
print(f"PHYSICS={a['physics_tables']['path']}")
print(f"TARGETS={a['contact_targets']['path']}")
print(f"HOLD_MANIFEST={a['holds_extended']['path']}")
print(f"PLANS_DIR={rec['dir']}/plans")
print(f"MIN_EVAL_STEPS={rec['training']['eval_max_steps']}")
print(f"ROBOT={rec['robot']}")
PYEOF
)"
if (( EVAL_MAX_STEPS < MIN_EVAL_STEPS )) && [[ "${SMOKE}" != "1" && "${TIMING}" != "1" ]]; then
  echo "EVAL_MAX_STEPS ${EVAL_MAX_STEPS} does not cover the longest motion (${MIN_EVAL_STEPS} steps)" >&2; exit 1
fi
P=${PLANS_DIR}
VIZ_PLANS=(
  "$P/fork_Warrior_II_Pose_or_Virabhadrasana_II.json"
  "$P/fork_Tree_Pose_or_Vrksasana.json"
  "$P/fork_Crane_Crow_Pose_or_Bakasana.json"
  "$P/fork_Handstand_pose_or_Adho_Mukha_Vrksasana.json"
  "$P/fork_Feathered_Peacock_Pose_or_Pincha_Mayuras.json"
  "$P/fork_Supported_Headstand_pose_or_Salamba_Sirs.json"
  "$P/fork_Side_Plank_Pose_or_Vasisthasana.json"
  "$P/fork_Downward_Facing_Dog_pose_or_Adho_Mukha_S.json"
  "$P/dwell_Warrior_II_Pose_or_Virabhadrasana_II_3s.json"
  "$P/dwell_Warrior_II_Pose_or_Virabhadrasana_II_10s.json"
)
for f in "${MOTIONS}" "${GRAPH}" "${PHYSICS}" "${TARGETS}" "${HOLD_MANIFEST}" "${VIZ_PLANS[@]}"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
if [ -e "results/${EXPERIMENT}/last.ckpt" ]; then
  echo "NOTE: results/${EXPERIMENT}/last.ckpt exists -> train_agent will RESUME that run (the warm start is ignored)."
fi
WANDB_FLAG=""
if [[ "${WANDB}" != "0" ]]; then WANDB_FLAG="--use-wandb"; fi
SUPPORT_RULE_FLAG=""
if [[ -n "${SUPPORT_RULE}" ]]; then SUPPORT_RULE_FLAG="--support-rule ${SUPPORT_RULE}"; fi
AMP_FLAGS=""
if [[ "${CONTROL}" != "1" ]]; then
  AMP_FLAGS="--amp-reward-w ${AMP_W} --amp-w-start-epoch ${AMP_START} --amp-w-full-epoch ${AMP_FULL} \
--amp-disc-batch-size ${AMP_DISC_BATCH} --amp-grad-penalty ${AMP_GRAD_PENALTY} \
--amp-calibrate-style-ratio ${AMP_CAL_RATIO} --amp-calibrate-window ${AMP_CAL_WINDOW}"
fi
CONFIG_ONLY_FLAG=""
if [[ "${CONFIG_ONLY}" == "1" ]]; then CONFIG_ONLY_FLAG="--create-config-only"; fi

echo "=== G1 $([[ "${CONTROL}" == "1" ]] && echo "CONTROL (no AMP)" || echo "AMP fine-tune") on release ${RELEASE} -> results/${EXPERIMENT}"
echo "    warm start     : ${CHECKPOINT} (+ optimisation state)"
echo "    experiment     : ${EXPERIMENT_PATH}"
echo "    robot / corpus : ${ROBOT} | ${MOTIONS}"
echo "    terms          : support ${SUPPORT_WEIGHT} (EMA 0.25 s), swing ${SWING_WEIGHT} (0.1 s), lean ${LEAN_WEIGHT}"
if [[ "${CONTROL}" != "1" ]]; then
  echo "    AMP            : w 0 until epoch ${AMP_START}, ramp to target by ${AMP_FULL}; target calibrated to style/task std ratio ${AMP_CAL_RATIO} (0 = fixed ${AMP_W}); disc batch ${AMP_DISC_BATCH}, GP ${AMP_GRAD_PENALTY}"
fi
echo "    curriculum     : mixture ${UNIFORM_FRACTION} uniform; support rule ${SUPPORT_RULE:-v1 (flag not passed)}"
echo "    envs / batch   : ${NUM_ENVS} / ${BATCH_SIZE}; ${MAX_EPOCHS} epochs; eval every ${EVAL_EVERY} over ${EVAL_MAX_STEPS} steps"

# shellcheck disable=SC2086
${PY} protomotions/train_agent.py --robot-name "${ROBOT}" --simulator isaaclab \
  --experiment-path "${EXPERIMENT_PATH}" \
  --experiment-name "${EXPERIMENT}" \
  --checkpoint "${CHECKPOINT}" --warm-start-optimization-state \
  --motion-file "${MOTIONS}" \
  --hold-graph-file "${GRAPH}" \
  --release-record "${RECORD}" --contact-targets "${TARGETS}" \
  --goal-bodies all --sense-body-pair-contacts True \
  --segment-start-prob 0.6 --critic-future-steps 1 5 10 15 \
  --interval-schedule True \
  --curriculum mixture --uniform-fraction "${UNIFORM_FRACTION}" --hold-manifest "${HOLD_MANIFEST}" \
  ${SUPPORT_RULE_FLAG} \
  --eval-every "${EVAL_EVERY}" --eval-max-steps "${EVAL_MAX_STEPS}" --save-every "${SAVE_EVERY}" \
  --viz-sequences-every "${VIZ_EVERY}" --viz-num-sequences 12 --viz-max-seconds 24.0 \
  --viz-log-scalars False --viz-plan-files "${VIZ_PLANS[@]}" \
  --support-penalty-weight "${SUPPORT_WEIGHT}" --support-clear-height 0.25 \
  --support-load-ref-frac 0.1 --support-ema-tau 0.25 \
  --physics-tables "${PHYSICS}" \
  --swing-penalty-weight "${SWING_WEIGHT}" --swing-ema-tau 0.1 \
  --lean-penalty-weight "${LEAN_WEIGHT}" --lean-min-margin 0.03 --lean-scale 0.10 \
  --drag-report-motions ${DRAG_REPORT} \
  ${AMP_FLAGS} \
  --num-envs "${NUM_ENVS}" --batch-size "${BATCH_SIZE}" --training-max-steps "${MAX_STEPS}" \
  --headless True ${WANDB_FLAG} ${CONFIG_ONLY_FLAG} \
  --overrides env.ref_respawn_offset=0.005 ${EXTRA}
