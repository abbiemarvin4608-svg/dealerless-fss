#!/usr/bin/env bash
set -euo pipefail
# Run from any directory; output must be a NEW directory.
repro=${1:?Usage: run.sh REPRODUCTION_DIRECTORY NEW_OUTPUT_DIRECTORY}
output=${2:?Usage: run.sh REPRODUCTION_DIRECTORY NEW_OUTPUT_DIRECTORY}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repro=$(realpath "$repro")
output=$(realpath -m "$output")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
quant_python="$repro/seedot-venv/bin/python"
float_python="$repro/venv/bin/python"
"$quant_python" "$script_dir/prepare.py" --reproduction "$repro" --output "$output"
"$quant_python" -u "$script_dir/quantize.py" --run "$output" > "$output/quantization.log" 2>&1
"$float_python" -u "$script_dir/evaluate_fp32.py" --reproduction "$repro" --run "$output" > "$output/evaluate_fp32.log" 2>&1
"$quant_python" -u "$script_dir/evaluate_fixed.py" --reproduction "$repro" --run "$output" > "$output/evaluate_fixed.log" 2>&1
"$quant_python" "$script_dir/export_contract.py" --run "$output"
"$quant_python" "$script_dir/finalize.py" --reproduction "$repro" --run "$output"
