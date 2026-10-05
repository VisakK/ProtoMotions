#!/bin/bash
# Card T6 (PLAN.MD): the D1 selection's recipes re-executed in PhysX on the endpoint-exact references
# (edge_synthesis.exact), from S's exemplar, 5 seeds per recipe in one IsaacLab process each; then the export
# (de-penetration), the PhysX reset check, admission with the seam group and the D1 policy, and the selection.
#
#   bash data/scripts/edge_synthesis/run_t6.sh                       # every stage
#   STAGES="mppi" RECIPES="B1m B1m12" bash data/scripts/edge_synthesis/run_t6.sh
#
# Stages: refs (CPU) | mppi (GPU) | export (CPU) | check (GPU) | admit (CPU) | select (CPU).
# GPU: at most two IsaacLab processes on the A5000 (PARALLEL=2), none beside an expert training run (PLAN.MD §2).
# Outputs: references output/edge_synthesis/exact/, runs output/edge_synthesis/physx/runs_t6/, motions
# output/edge_synthesis/physx/motions_t6/, the admission record expert_revist/graph_growth_2026_10_03/admitted_physx.json
# and the selection selected_physx.json (the pre-T6 records are kept as *_pre_t6.json).
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.:data/scripts
PLAN=expert_revist/graph_growth_2026_10_03
X=output/edge_synthesis/exact
R=${R:-output/edge_synthesis/physx/runs_t6}
MOT=${MOT:-output/edge_synthesis/physx/motions_t6}
L=${L:-output/edge_synthesis/physx/logs_t6}
mkdir -p $L
STAGES=${STAGES:-"refs mppi export check admit select"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
PARALLEL=${PARALLEL:-2}
COMMON="--start exemplar --seeds $SEEDS --samples ${SAMPLES:-512} --replan 3 --iters 2 --start-iters 16 --out-root $R"
M="python -m edge_synthesis.edge_mppi_physx $COMMON"
TRACK='{"track": 20, "track_end": 20}'
# landing edges: the card's landing aim (costs.EdgeWeights.cone: D's feet, 1.5 cm tolerance). E2's lane-T override
# (anchor 8, box 2, flat 0) is dropped: the card keeps the hands planted, and under it the pre-T6 E2 hands slid
# 1.8-3.4 cm; flat 100 is the setting its T6 probe ran (runs_t6test/E2_*_t6cone).
LAND=${LAND:-'{"cone": 10}'}
LAND_E2=${LAND_E2:-'{"cone": 10, "flat": 100}'}
# the landing recipes exactly as recorded, only the reference changed (no cone; E2's lane-T override):
#   LAND='{}' LAND_E2='{"anchor": 8, "box": 2, "flat": 0}' R=output/edge_synthesis/physx/runs_t6_nocone \
#   MOT=output/edge_synthesis/physx/motions_t6_nocone L=output/edge_synthesis/physx/logs_t6_nocone \
#   STAGES="mppi export" RECIPES="B1h B1h12 B1m B1m12 E2m E2m12 E5m12" bash data/scripts/edge_synthesis/run_t6.sh
recipe() {   # name -> the edge_mppi_physx arguments of the D1 recipe (selected_physx.json, pre-T6, "recipe")
  case $1 in
    E1h)   echo "--edge E1 --reference $X/E1_high --noise 0.05 --weights '$TRACK' --tag t6px" ;;
    E1m)   echo "--edge E1 --reference $X/E1_mid  --noise 0.05 --weights '$TRACK' --tag t6px" ;;
    E3h)   echo "--edge E3 --reference $X/E3_high --noise 0.05 --weights '$TRACK' --tag t6px" ;;
    E3h12) echo "--edge E3 --reference $X/E3_high --noise 0.12 --weights '$TRACK' --tag t6px12" ;;
    B1h)   echo "--edge B1 --reference $X/B1_high --noise 0.08 --weights '$LAND' --tag t6px" ;;
    B1h12) echo "--edge B1 --reference $X/B1_high --noise 0.12 --weights '$LAND' --tag t6px12" ;;
    B1m)   echo "--edge B1 --reference $X/B1_mid  --noise 0.08 --weights '$LAND' --tag t6px" ;;
    B1m12) echo "--edge B1 --reference $X/B1_mid  --noise 0.12 --weights '$LAND' --tag t6px12" ;;
    E2m)   echo "--edge E2 --reference $X/E2_mid  --noise 0.08 --weights '$LAND_E2' --tag t6px" ;;
    E2m12) echo "--edge E2 --reference $X/E2_mid  --noise 0.12 --weights '$LAND_E2' --tag t6px12" ;;
    E5m12) echo "--edge E5 --reference $X/E5_mid  --noise 0.12 --weights '$LAND' --tag t6px12" ;;
  esac
}
RECIPES=${RECIPES:-"E1h E1m E3h E3h12 B1h B1h12 B1m B1m12 E2m E2m12 E5m12"}

for stage in $STAGES; do
  echo "$(date +%T) stage $stage"
  case $stage in
    refs)
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -m edge_synthesis.exact --d1 > $L/refs.log 2>&1 || { echo "refs FAILED (see $L/refs.log)"; exit 1; } ;;
    mppi)
      n=0
      for r in $RECIPES; do
        args=$(recipe $r)
        [ -z "$args" ] && { echo "unknown recipe $r"; exit 1; }
        ( echo "$(date +%T) start $r"; eval "$M $args --print-log" > $L/mppi_$r.log 2>&1; echo "$(date +%T) end $r rc=$?" ) &
        n=$((n + 1))
        if [ $n -ge $PARALLEL ]; then wait -n; n=$((n - 1)); fi
      done
      wait ;;
    export)
      for tag in t6px t6px12; do SKIP_EXISTING=1 RUNS=$R OUT=$MOT TAG=$tag bash data/scripts/edge_synthesis/export_physx_batch.sh; done ;;
    check)
      python data/scripts/reference_curation/retarget_v2_physx.py --motion-dir $MOT --per-clip 20 \
        --out output/edge_synthesis/physx/physx_launch_check_t6.json > $L/physx_launch_check.log 2>&1 ;;
    admit)
      [ -f $PLAN/admitted_physx_pre_t6.json ] || cp $PLAN/admitted_physx.json $PLAN/admitted_physx_pre_t6.json
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m edge_synthesis.admit $MOT/SYN_*.json \
        --physx-launch-check output/edge_synthesis/physx/physx_launch_check_t6.json --policy d1 \
        --note "Card T6: the D1 recipes re-executed in PhysX on the endpoint-exact references (edge_synthesis.exact), from S's exemplar, 5 seeds each" \
        --out $PLAN/admitted_physx.json > $L/admit.log 2>&1 ;;
    select)
      [ -f $PLAN/selected_physx_pre_t6.json ] || cp $PLAN/selected_physx.json $PLAN/selected_physx_pre_t6.json
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m edge_synthesis.select_variants \
        --admitted $PLAN/admitted_physx.json --out $PLAN/selected_physx.json > $L/select.log 2>&1
      cat $L/select.log ;;
  esac
done
echo "$(date +%T) T6 DONE"
