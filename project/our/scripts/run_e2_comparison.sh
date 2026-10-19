#!/usr/bin/env bash
# E2 focused comparison: Does ℓ=20 monolithic outperform ℓ=32 segmented?
#
# Configurations:
#   1. ℓ=20, nseg=1 (monolithic, "E0 shows this precision is sufficient")
#   2. ℓ=32, nseg=2 (segmented, "standard 32-bit precision")
#   3. ℓ=24, nseg=1 (monolithic boundary, expected to be slower)
#   4. ℓ=28, nseg=1 (monolithic infeasible, should timeout/OOM)
#
# This directly answers: if precision allows ℓ=20, is monolithic FSS better?

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
RUNS="${1:-10}"

BENCH="${BUILD_DIR}/test/benchmark/nseg_sweep"
RESULTS_DIR="${PROJECT_DIR}/results"
OUT_CSV="${RESULTS_DIR}/e2_comparison.csv"

if [[ ! -x "${BENCH}" ]]; then
    echo "ERROR: benchmark not built at ${BENCH}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "ell,nseg,offline_time_ms,online_time_ms,offline_comm_bytes,online_comm_bytes" > "${OUT_CSV}"

run_one() {
    local ell="$1" nseg="$2" port="$3" timeout_sec="${4:-60}"

    echo "  [ell=${ell}, nseg=${nseg}] Starting..."

    # Run with timeout to catch infeasible configurations
    timeout "${timeout_sec}s" "${BENCH}" r=2 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e2_server.log 2>&1 &
    local srv=$!
    sleep 0.3
    timeout "${timeout_sec}s" "${BENCH}" r=3 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e2_client.log 2>&1 &
    local cli=$!

    if wait "${srv}" && wait "${cli}"; then
        if grep -q "CORRECTNESS FAILURE" /tmp/e2_server.log; then
            echo "    ✗ CORRECTNESS FAILURE"
            return 1
        fi
        grep -E "^${ell},${nseg}," /tmp/e2_server.log >> "${OUT_CSV}"
        echo "    ✓ Complete"
    else
        echo "    ✗ TIMEOUT or OOM (infeasible)"
        echo "${ell},${nseg},TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "${OUT_CSV}"
    fi
}

BASE_PORT=35200
port="${BASE_PORT}"

echo
echo "============================================================"
echo "E2 Comparison: Monolithic ℓ=20 vs Segmented ℓ=32"
echo "============================================================"
echo

echo "[1/5] ℓ=20, nseg=1 (monolithic, E0-sufficient precision)"
run_one 20 1 $((port++)) 60

echo
echo "[2/5] ℓ=32, nseg=2 (segmented, standard 32-bit)"
run_one 32 2 $((port++)) 60

echo
echo "[3/5] ℓ=24, nseg=1 (monolithic boundary, expect slowdown)"
run_one 24 1 $((port++)) 60

echo
echo "[4/5] ℓ=28, nseg=1 (monolithic infeasible, expect timeout)"
run_one 28 1 $((port++)) 60

echo
echo "[5/5] ℓ=32, nseg=4 (over-segmented, suboptimal)"
run_one 32 4 $((port++)) 60

echo
echo "============================================================"
echo "Results:"
echo "============================================================"
cat "${OUT_CSV}"

echo
echo "Saved to: ${OUT_CSV}"
