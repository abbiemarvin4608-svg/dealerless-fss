#!/usr/bin/env bash
# E1/E2: Segmentation sweep benchmark driver.
#
# E1: For fixed ell=32, sweep nseg to find the optimal segmentation count nseg*.
# E2: For each nseg, find the feasibility frontier (largest ell that completes).
#
# Runs the two-party nseg_sweep benchmark locally (both parties on localhost),
# collecting CSV rows into results/e1_e2_sweep.csv.
#
# Usage:
#   ./run_e1_e2.sh [BUILD_DIR] [RUNS]
#
# Prereq: build the project first (cmake --build build --target nseg_sweep).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_DIR}/build}"
RUNS="${2:-10}"

BENCH="${BUILD_DIR}/test/benchmark/nseg_sweep"
RESULTS_DIR="${PROJECT_DIR}/results"
OUT_CSV="${RESULTS_DIR}/e1_e2_sweep.csv"

if [[ ! -x "${BENCH}" ]]; then
    echo "ERROR: benchmark not built at ${BENCH}" >&2
    echo "Build it first: cmake --build '${BUILD_DIR}' --target nseg_sweep" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# Sweep configuration.
ELL_VALUES=(16 20 24 28 32)
NSEG_VALUES=(1 2 4 8 16)
BASE_PORT=34100

echo "ell,nseg,offline_time_ms,online_time_ms,offline_comm_bytes,online_comm_bytes" > "${OUT_CSV}"

run_one() {
    local ell="$1" nseg="$2" port="$3"
    # nseg cannot exceed ell (each segment needs at least 1 bit).
    if (( nseg > ell )); then
        return 0
    fi

    # SERVER (r=2) waits; start first in background.
    "${BENCH}" r=2 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e12_server.log 2>&1 &
    local srv=$!
    sleep 0.3
    "${BENCH}" r=3 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e12_client.log 2>&1 &
    local cli=$!

    wait "${srv}" || true
    wait "${cli}" || true

    if grep -q "CORRECTNESS FAILURE" /tmp/e12_server.log; then
        echo "  [ell=${ell} nseg=${nseg}] CORRECTNESS FAILURE" >&2
    fi
    # Append the data row (skip the header line the benchmark also prints).
    grep -E "^${ell},${nseg}," /tmp/e12_server.log >> "${OUT_CSV}" || \
        echo "  [ell=${ell} nseg=${nseg}] no output row" >&2
}

port="${BASE_PORT}"
for ell in "${ELL_VALUES[@]}"; do
    for nseg in "${NSEG_VALUES[@]}"; do
        echo "Running ell=${ell}, nseg=${nseg}..."
        run_one "${ell}" "${nseg}" "${port}"
        port=$((port + 1))
    done
done

echo
echo "Done. Results written to ${OUT_CSV}"
cat "${OUT_CSV}"
