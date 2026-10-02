#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The first expert on plant v2, trained fresh on a curation release (BodyFix Step 5 = BUILD_PLAN Steps 9-10).
#
# Everything the run loads comes from ONE release record (reference_curation.release_v2): the packaged motions,
# graph v2, physics tables v2, the contact-target sidecar, the evaluator's hold manifest and the viz panel's
# probe plans. The run names the record (--release-record), so every artifact is checked against it by sha256 at
# startup and the run refuses to start on a mismatch (protomotions/utils/release_identity.py).
#
# Fresh: no checkpoint, fresh normaliser statistics, robot smpl_yogi_v2, a new experiment identity that names the
# release. The reward is fine-tune C's stack, unchanged in kind (expert_revist/ft_c/README.MD): the 2x tracking
# terms, the averaged unwanted-support term (-0.3, 0.25 s EMA; with the sidecar it charges only zones the human is
# known to keep free), the swing term (-0.3, 0.1 s) and the lean term (-0.3). The curriculum (80 % uniform) and
# the full-clip hold-aware evaluator are fine-tune A's. New in this run, all at WEIGHT 0: the sidecar's target
# diagnostics (diag_required_support_*, diag_pair_target_*, diag_known_free_load_n).
#
#   bash data/scripts/run_expert_release_v2.sh                 # the run
#   SMOKE=1 bash data/scripts/run_expert_release_v2.sh         # Step 9 exit: 256 envs, 4 epochs, every term weight 0
#
# Watch: eval/perf_group/{single_leg,inversion}_score against ft_c's 0.998 / 0.998 on ftC (BodyFix Step 5's
# acceptance), eval/perf_group/*, eval/drag/*, env/raw_r/diag_* (raw, never trained), and the viz/ panel.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${PY:-python}

SMOKE=${SMOKE:-0}
RELEASE=${RELEASE:-holds_repaired_ftC_posefix.release_v2.a2dda5d2ac}
RECORD=data/reference_curation/releases/${RELEASE}.json
[ -f "${RECORD}" ] || { echo "missing release record: ${RECORD}" >&2; exit 1; }
SHORT=${RELEASE##*.}
EXPERIMENT=${EXPERIMENT:-smpl_yogi_v2_expert56_${SHORT}}
NUM_ENVS=${NUM_ENVS:-4096}
BATCH_SIZE=${BATCH_SIZE:-16384}
UNIFORM_FRACTION=${UNIFORM_FRACTION:-0.8}
EVAL_EVERY=${EVAL_EVERY:-500}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-2250}       # >= the record's training.eval_max_steps (the longest motion)
SAVE_EVERY=${SAVE_EVERY:-500}
MAX_EPOCHS=${MAX_EPOCHS:-30000}              # ~3 days; checkpoints every 500 epochs, ~0.64 GB each
VIZ_EVERY=${VIZ_EVERY:-500}
WANDB=${WANDB:-1}
EXTRA=${EXTRA:-}
SUPPORT_WEIGHT=${SUPPORT_WEIGHT:--0.3}
SWING_WEIGHT=${SWING_WEIGHT:--0.3}
LEAN_WEIGHT=${LEAN_WEIGHT:--0.3}
# ft_c's drag gate set, minus Firefly -b (dropped from the corpus, BodyFix Step 3)
DRAG_REPORT=${DRAG_REPORT:-"Plow_Pose_or_Halasana_-b Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a \
Upward_Plank_Pose_or_Purvottanasana_-a Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a \
Low_Lunge_pose_or_Anjaneyasana_-a Peacock_Pose_or_Mayurasana_-a Plow_Pose_or_Halasana_-a \
Warrior_II_Pose_or_Virabhadrasana_II_-a"}

if [[ "${SMOKE}" == "1" ]]; then
  EXPERIMENT=${EXPERIMENT}_smoke
  NUM_ENVS=256; BATCH_SIZE=1024; EVAL_EVERY=2; EVAL_MAX_STEPS=150; SAVE_EVERY=2; MAX_EPOCHS=4; VIZ_EVERY=0; WANDB=0
  SUPPORT_WEIGHT=0; SWING_WEIGHT=0; LEAN_WEIGHT=0
fi
MAX_STEPS=$((MAX_EPOCHS * NUM_ENVS * 32))

# Every path from the record; the run re-checks each one by sha256.
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
if (( EVAL_MAX_STEPS < MIN_EVAL_STEPS )) && [[ "${SMOKE}" != "1" ]]; then
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
  echo "NOTE: results/${EXPERIMENT}/last.ckpt exists -> train_agent will RESUME that run."
fi
WANDB_FLAG=""
if [[ "${WANDB}" != "0" ]]; then WANDB_FLAG="--use-wandb"; fi

echo "=== fresh expert on release ${RELEASE} -> results/${EXPERIMENT}"
echo "    robot / corpus : ${ROBOT} | ${MOTIONS}"
echo "    graph / tables : ${GRAPH} | ${PHYSICS}"
echo "    sidecar        : ${TARGETS} (weight-0 diagnostics; known-free support gate)"
echo "    terms          : support ${SUPPORT_WEIGHT} (EMA 0.25 s), swing ${SWING_WEIGHT} (0.1 s), lean ${LEAN_WEIGHT}"
echo "    envs / batch   : ${NUM_ENVS} / ${BATCH_SIZE}; ${MAX_EPOCHS} epochs; eval every ${EVAL_EVERY} over ${EVAL_MAX_STEPS} steps"

# shellcheck disable=SC2086
${PY} protomotions/train_agent.py --robot-name "${ROBOT}" --simulator isaaclab \
  --experiment-path examples/experiments/mimic/mlp_goal_conditioned.py \
  --experiment-name "${EXPERIMENT}" \
  --motion-file "${MOTIONS}" \
  --hold-graph-file "${GRAPH}" \
  --release-record "${RECORD}" --contact-targets "${TARGETS}" \
  --goal-bodies all --sense-body-pair-contacts True \
  --segment-start-prob 0.6 --critic-future-steps 1 5 10 15 \
  --interval-schedule True \
  --curriculum mixture --uniform-fraction "${UNIFORM_FRACTION}" --hold-manifest "${HOLD_MANIFEST}" \
  --eval-every "${EVAL_EVERY}" --eval-max-steps "${EVAL_MAX_STEPS}" --save-every "${SAVE_EVERY}" \
  --viz-sequences-every "${VIZ_EVERY}" --viz-num-sequences 12 --viz-max-seconds 24.0 \
  --viz-log-scalars False --viz-plan-files "${VIZ_PLANS[@]}" \
  --support-penalty-weight "${SUPPORT_WEIGHT}" --support-clear-height 0.25 \
  --support-load-ref-frac 0.1 --support-ema-tau 0.25 \
  --physics-tables "${PHYSICS}" \
  --swing-penalty-weight "${SWING_WEIGHT}" --swing-ema-tau 0.1 \
  --lean-penalty-weight "${LEAN_WEIGHT}" --lean-min-margin 0.03 --lean-scale 0.10 \
  --drag-report-motions ${DRAG_REPORT} \
  --num-envs "${NUM_ENVS}" --batch-size "${BATCH_SIZE}" --training-max-steps "${MAX_STEPS}" \
  --headless True ${WANDB_FLAG} \
  --overrides env.ref_respawn_offset=0.005 ${EXTRA}
