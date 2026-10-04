#!/bin/bash
# Lane T's edge recipes executed in PhysX (edge_mppi_physx; card T7, t_edges_physx/README.MD), 4 seeds per
# (edge, timing) in one IsaacLab process. The per-recipe settings are lane T's MuJoCo ones (t_edges/README.MD and
# its queue scripts); set NOISE to run every recipe at one noise instead (pass 2 of T7: NOISE=0.12 TAG=px12).
#
#   SAMPLES=512 REPLAN=3 ITERS=2 START_ITERS=16 TAG=px bash data/scripts/edge_synthesis/run_physx_recipes.sh
#   EDGES="E1h B1m" TAG=px bash data/scripts/edge_synthesis/run_physx_recipes.sh      # a subset
#
# GPU: at most two IsaacLab processes on the A5000, none beside an expert training run (PLAN.MD §2 Operations).
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.:data/scripts
L=output/edge_synthesis/physx/logs
Q=output/edge_synthesis/quasistatic
mkdir -p $L
TAG=${TAG:-px}
SEEDS=${SEEDS:-"0 1 2 3"}
COMMON="--seeds $SEEDS --samples ${SAMPLES:-512} --replan ${REPLAN:-3} --iters ${ITERS:-2} --start-iters ${START_ITERS:-16} --tag $TAG"
M="python -m edge_synthesis.edge_mppi_physx $COMMON"
TRACK='{"track": 20, "track_end": 20}'
n() { echo "${NOISE:-$1}"; }          # lane T's noise for the recipe, unless NOISE overrides it
run() { local log=$1; shift; echo "$(date +%T) start $log"; "$@" > $L/${TAG}_$log.log 2>&1; echo "$(date +%T) end $log rc=$?"; }
for e in ${EDGES:-E1h E1m E3h E3m E2 E5 B1m B1h E4h E4m B2}; do
  case $e in
    E1h) run E1_high $M --edge E1 --reference $Q/E1_high --noise $(n 0.05) --weights "$TRACK" ;;
    E1m) run E1_mid  $M --edge E1 --reference $Q/E1_mid  --noise $(n 0.05) --weights "$TRACK" ;;
    E3h) run E3_high $M --edge E3 --reference $Q/E3_high --noise $(n 0.05) --weights "$TRACK" ;;
    E3m) run E3_mid  $M --edge E3 --reference $Q/E3_mid  --noise $(n 0.05) --weights "$TRACK" ;;
    E4h) run E4_high $M --edge E4 --reference $Q/E4_high --noise $(n 0.05) --weights "$TRACK" ;;
    E4m) run E4_mid  $M --edge E4 --reference $Q/E4_mid  --noise $(n 0.05) --weights "$TRACK" ;;
    B2)  run B2_mid  $M --edge B2 --reference $Q/B2_mid  --noise $(n 0.05) --weights "$TRACK" ;;
    E2)  run E2_mid  $M --edge E2 --keyframes land --noise $(n 0.08) --weights '{"anchor": 8, "box": 2, "flat": 0}' ;;
    E5)  run E5_mid  $M --edge E5 --keyframes land --via 220923_Plank_Pose_or_Kumbhakasana_-a@541 --noise $(n 0.08) ;;
    B1m) run B1_mid  $M --edge B1 --keyframes land --noise $(n 0.08) ;;
    B1h) run B1_high $M --edge B1 --keyframes land --timing high --noise $(n 0.08) ;;
  esac
done
echo "$(date +%T) QUEUE DONE"
