#!/usr/bin/env bash
set -euo pipefail

precision_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$precision_dir"
export MAX_JOBS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 -u run_e0_uniform_storage.py \
  --models resnet50,resnet101,resnet152 \
  --ell-grid 12,16,20,24,28,32 \
  --calibration-samples 512 --test-samples 0 \
  --device cuda --batch-size 16 --workers 2 \
  --integer-backend gemm \
  --arithmetic-mode ring --overflow-mode wrap \
  --results-dir results/e0_uniform_ring_imagenette

python3 summarize_uniform_storage.py \
  --results-dir results/e0_uniform_ring_imagenette
