#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The release-v3 student (cards S1/S2 of expert_revist/graph_growth_2026_10_03/PLAN.MD): v9's FSQ recipe distilled
# from the goal-conditioned expert G3 (epoch_3420.ckpt) on release v3, with the expert view, the teacher's schedule
# semantics and the identity asserts of examples/experiments/masked_mimic/contact_graph_fsq_release_v3.py.
#
# S2's recipe (PLAN.MD card S2): one-token code (FSQ 4 scalars, 4 per token), dagger_action_loss_coeff 0.1,
# prior_rollout_fraction 0.25 from epoch 1,000 (ramp 500), fixed sampling, a checkpoint every 500, root-relative
# student targets, NO pushes (the teacher was never trained under them; Lane S's note). The panel runs S1's route
# plans and the edge plans on training's timing.
#
#   bash data/scripts/run_student_distill_release_v3.sh                 # S2 (wandb)
#   CONFIG_ONLY=1 bash data/scripts/run_student_distill_release_v3.sh   # resolved configs only (CPU, no simulation)
#   SMOKE=1 bash data/scripts/run_student_distill_release_v3.sh         # 256 envs x SMOKE_EPOCHS (4), no wandb
#   DRY_RUN=1 bash data/scripts/run_student_distill_release_v3.sh       # print the command
#   RESUME=1 bash data/scripts/run_student_distill_release_v3.sh        # continue results/${EXPERIMENT}
#
# It refuses an existing results/${EXPERIMENT}/last.ckpt unless RESUME=1 (a relaunch would resume and ignore every
# flag), and a name a live train_agent.py is writing to.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${PY:-python}
read -r -a PY_CMD <<< "${PY}"
die() { echo "run_student_distill_release_v3.sh: $*" >&2; exit 1; }

SMOKE=${SMOKE:-0}; CONFIG_ONLY=${CONFIG_ONLY:-0}; RESUME=${RESUME:-0}; DRY_RUN=${DRY_RUN:-0}; WANDB=${WANDB:-1}
for v in SMOKE CONFIG_ONLY RESUME DRY_RUN WANDB; do
  [[ "${!v}" =~ ^[01]$ ]] || die "${v} must be 0 or 1, got '${!v}'"
done

RELEASE=${RELEASE:-holds_repaired_ftC_posefix.release_v3.2f132f4299}
EXPERT=${EXPERT:-results/smpl_yogi_v2_expert56_g3_2f132f4299/epoch_3420.ckpt}
EXPERIMENT=${EXPERIMENT:-smpl_yogi_v2_student_release_v3_g3e3420}
ROUTING=${ROUTING:-data/smpl/student_release_v3/${RELEASE}.g3_e3420.experts.json}
[[ "${SMOKE}" == "1" ]] && EXPERIMENT=${EXPERIMENT}_smoke
SAVE_DIR=results/${EXPERIMENT}

RECORD=data/reference_curation/releases/${RELEASE}.json
[ -f "${RECORD}" ] || die "missing release record: ${RECORD}"
[ -f "${EXPERT}" ] || die "missing expert checkpoint: ${EXPERT}"
[ -f "$(dirname "${EXPERT}")/resolved_configs.pt" ] || die "missing the expert's resolved_configs.pt beside ${EXPERT}"
[ -f "${ROUTING}" ] || die "missing routing table ${ROUTING} (data/scripts/make_single_expert_routing.py)"
MOTIONS=$("${PY_CMD[@]}" -c "import json,sys; print(json.load(open(sys.argv[1]))['artifacts']['package']['path'])" "${RECORD}")
[ -f "${MOTIONS}" ] || die "missing the release package ${MOTIONS}"

if [[ "${RESUME}" == "1" ]]; then
  [[ "${CONFIG_ONLY}" == "1" ]] && die "CONFIG_ONLY=1 with RESUME=1 would overwrite ${SAVE_DIR}'s frozen configs"
  [ -f "${SAVE_DIR}/last.ckpt" ] || die "RESUME=1 but ${SAVE_DIR}/last.ckpt does not exist"
elif [ -e "${SAVE_DIR}/last.ckpt" ] || [ -L "${SAVE_DIR}/last.ckpt" ]; then
  die "${SAVE_DIR}/last.ckpt exists: train_agent would RESUME it and ignore every flag. Pick a new EXPERIMENT, \
pass RESUME=1, or move the directory away."
fi
LIVE=$(pgrep -af '[t]rain_agent\.py' || true)
if [[ -n "${LIVE}" ]] && grep -qF -e " --experiment-name ${EXPERIMENT} " <<< "$(sed 's/$/ /' <<< "${LIVE}")"; then
  die "a live train_agent.py is writing to ${SAVE_DIR}"
fi

NUM_ENVS=${NUM_ENVS:-1024}
BATCH_SIZE=${BATCH_SIZE:-8192}
MAX_EPOCHS=${MAX_EPOCHS:-8000}
DAGGER_FRACTION=${DAGGER_FRACTION:-0.25}
DAGGER_START=${DAGGER_START:-1000}
DAGGER_RAMP=${DAGGER_RAMP:-500}
DAGGER_ACTION_COEFF=${DAGGER_ACTION_COEFF:-0.1}
FSQ_SCALARS=${FSQ_SCALARS:-4}
FSQ_PER_TOKEN=${FSQ_PER_TOKEN:-4}
SEGMENT_START_PROB=${SEGMENT_START_PROB:-0.6}
VIZ_EVERY=${VIZ_EVERY:-500}
EXTRA=${EXTRA:-}
if [[ "${SMOKE}" == "1" ]]; then
  NUM_ENVS=256; BATCH_SIZE=2048; MAX_EPOCHS=${SMOKE_EPOCHS:-4}; VIZ_EVERY=0; WANDB=0
  DAGGER_START=0; DAGGER_RAMP=0          # the smoke exercises the prior-driven block and its loss too
fi
for v in NUM_ENVS BATCH_SIZE MAX_EPOCHS; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "${v} must be a positive integer, got '${!v}'"
done
MAX_STEPS=$((MAX_EPOCHS * NUM_ENVS * 32))
read -r -a EXTRA_ARGS <<< "${EXTRA}"

CMD=("${PY_CMD[@]}" protomotions/train_agent.py --robot-name smpl_yogi_v2 --simulator isaaclab
  --experiment-path examples/experiments/masked_mimic/contact_graph_fsq_release_v3.py
  --experiment-name "${EXPERIMENT}"
  --motion-file "${MOTIONS}"
  --expert-model-path "${EXPERT}"
  --motion-expert-file "${ROUTING}"
  --segment-start-prob "${SEGMENT_START_PROB}"
  --sense-body-pair-contacts True
  --fsq-scalars "${FSQ_SCALARS}" --fsq-scalars-per-token "${FSQ_PER_TOKEN}"
  --dagger-action-loss-coeff "${DAGGER_ACTION_COEFF}"
  --student-root-relative-xy True
  --viz-sequences-every "${VIZ_EVERY}" --viz-log-scalars False
  --num-envs "${NUM_ENVS}" --batch-size "${BATCH_SIZE}" --training-max-steps "${MAX_STEPS}"
  --headless True)
[[ "${WANDB}" == "1" ]] && CMD+=(--use-wandb)
[[ "${CONFIG_ONLY}" == "1" ]] && CMD+=(--create-config-only)
CMD+=(--overrides env.ref_respawn_offset=0.005
  agent.prior_rollout_fraction="${DAGGER_FRACTION}"
  agent.prior_rollout_start_epoch="${DAGGER_START}"
  agent.prior_rollout_ramp_epochs="${DAGGER_RAMP}"
  "${EXTRA_ARGS[@]}")

echo "=== release-v3 student -> ${SAVE_DIR}"
echo "    teacher : ${EXPERT} (one expert; routing ${ROUTING})"
echo "    release : ${RELEASE} (${MOTIONS})"
echo "    recipe  : v9 -- FSQ ${FSQ_SCALARS}/${FSQ_PER_TOKEN}, dagger-action ${DAGGER_ACTION_COEFF}, DAgger ${DAGGER_FRACTION} from ${DAGGER_START} (ramp ${DAGGER_RAMP}); ${NUM_ENVS} envs / ${BATCH_SIZE}; ${MAX_EPOCHS} epochs"
echo "    command : $(printf '%q ' "${CMD[@]}")"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: nothing launched."
  exit 0
fi
exec "${CMD[@]}"
