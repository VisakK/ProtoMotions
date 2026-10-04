#!/bin/bash
# Export every PhysX run whose name ends in _${TAG} (export_physx, de-penetration), four single-threaded workers,
# one E-core each (PLAN.MD §2 Operations), then nothing else: admit.py and select_variants.py are separate steps.
#
#   TAG=px bash data/scripts/edge_synthesis/export_physx_batch.sh
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:data/scripts
L=output/edge_synthesis/physx/logs
mkdir -p $L
TAG=${TAG:?set TAG}
runs=( $(ls -d output/edge_synthesis/physx/runs/*_${TAG} | sort) )
for w in 0 1 2 3; do
  part=()
  for i in "${!runs[@]}"; do [ $((i % 4)) -eq $w ] && part+=("${runs[$i]}"); done
  [ ${#part[@]} -gt 0 ] && taskset -c $((16 + w)) python -m edge_synthesis.export_physx "${part[@]}" > $L/export_${TAG}_w$w.log 2>&1 &
done
wait
echo "EXPORT DONE ${TAG}"
