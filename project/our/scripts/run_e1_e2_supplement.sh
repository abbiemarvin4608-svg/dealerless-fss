#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BENCH="${BUILD_DIR}/test/benchmark/nseg_sweep"
RESULTS_DIR="${PROJECT_DIR}/results"
OUT_CSV="${RESULTS_DIR}/e1_e2_sweep.csv"
RUNS=3
BASE_PORT=35000

run_one() {
    local ell="$1" nseg="$2" port="$3"
    echo "  [ell=${ell} nseg=${nseg}]"
    "${BENCH}" r=2 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e12s_srv.log 2>&1 &
    local srv=$!
    sleep 0.5
    "${BENCH}" r=3 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e12s_cli.log 2>&1 &
    local cli=$!
    wait "${srv}" "${cli}" 2>/dev/null || true
    grep -E "^${ell},${nseg}," /tmp/e12s_srv.log >> "${OUT_CSV}" || echo "    (no output)"
}

port="${BASE_PORT}"
for ell in 28 32; do
    for nseg in 2 4 8; do
        run_one "${ell}" "${nseg}" "${port}"
        port=$((port + 1))
    done
done

echo "Done. Updated ${OUT_CSV}"
