#!/usr/bin/env bash
# Compare monolithic vs segmented at ℓ=20
#
# Question: At ℓ=20 (sufficient precision per E0), does segmentation
# improve performance, or is monolithic faster?

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
RUNS="${1:-10}"

BENCH="${BUILD_DIR}/test/benchmark/nseg_sweep"
RESULTS_DIR="${PROJECT_DIR}/results"
OUT_CSV="${RESULTS_DIR}/e2_ell20_comparison.csv"

if [[ ! -x "${BENCH}" ]]; then
    echo "ERROR: benchmark not built at ${BENCH}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "ell,nseg,offline_time_ms,online_time_ms,offline_comm_bytes,online_comm_bytes" > "${OUT_CSV}"

run_one() {
    local ell="$1" nseg="$2" port="$3"

    echo "  [ell=${ell}, nseg=${nseg}] Running..."

    "${BENCH}" r=2 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e2_server.log 2>&1 &
    local srv=$!
    sleep 0.3
    "${BENCH}" r=3 p="${port}" ell="${ell}" nseg="${nseg}" runs="${RUNS}" > /tmp/e2_client.log 2>&1 &
    local cli=$!

    wait "${srv}" && wait "${cli}"

    if grep -q "CORRECTNESS FAILURE" /tmp/e2_server.log; then
        echo "    ✗ CORRECTNESS FAILURE"
        return 1
    fi
    grep -E "^${ell},${nseg}," /tmp/e2_server.log >> "${OUT_CSV}"
    echo "    ✓ Complete"
}

BASE_PORT=36000
port="${BASE_PORT}"
ell=20

echo
echo "============================================================"
echo "ℓ=20: Monolithic (nseg=1) vs Segmented (nseg=2,4,8)"
echo "============================================================"
echo "Question: Does segmentation speed up generation at ℓ=20?"
echo

for nseg in 1 2 4 8; do
    echo "[nseg=${nseg}]"
    run_one "${ell}" "${nseg}" $((port++))
    echo
done

echo "============================================================"
echo "Results:"
echo "============================================================"
cat "${OUT_CSV}"

echo
echo
echo "Analysis:"
echo "--------"
awk -F',' 'NR>1 {
    printf "nseg=%2d: offline=%7.2f ms, online=%7.2f ms, online_comm=%d bytes\n",
           $2, $3, $4, $6
}' "${OUT_CSV}"

echo
echo "Saved to: ${OUT_CSV}"
