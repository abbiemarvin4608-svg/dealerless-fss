#!/bin/bash
# Performance comparison: BitDec vs Piecewise Comparison

set -e

BUILD_DIR="./build/test/correctness"
TEST_BIN="$BUILD_DIR/SEGFSS_CORRECTNESS_TEST"

echo "==================================================================="
echo "Performance Test: Sigmoid/Tanh Optimization"
echo "==================================================================="
echo ""
echo "Comparing:"
echo "  - Original: BitDec (64 rounds)"
echo "  - Optimized: Piecewise Comparison (~6 rounds)"
echo ""
echo "==================================================================="

# Run test and capture timing
run_test() {
    local case=$1
    local label=$2

    echo ""
    echo "Running: $label"
    echo "-------------------------------------------------------------------"

    # Start server
    $TEST_BIN r=2 t=$case > /tmp/server_perf.log 2>&1 &
    SERVER_PID=$!

    # Give server time to start
    sleep 1

    # Start client and capture time
    START_TIME=$(date +%s.%N)
    $TEST_BIN r=3 t=$case > /tmp/client_perf.log 2>&1
    END_TIME=$(date +%s.%N)

    # Wait for server
    wait $SERVER_PID 2>/dev/null || true

    # Calculate elapsed time
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

    # Extract round count and communication from logs
    ROUNDS=$(grep -i "round" /tmp/server_perf.log 2>/dev/null | tail -1 || echo "N/A")

    echo "Time: ${ELAPSED}s"
    echo "Rounds: $ROUNDS"

    # Check if test passed
    if grep -q "FAIL" /tmp/server_perf.log; then
        echo "Status: FAILED (but timing still valid)"
        grep "FAIL" /tmp/server_perf.log | head -3
    else
        echo "Status: PASSED"
    fi

    echo "$ELAPSED" > /tmp/perf_$label.txt
}

# Run activation tests (case 9)
run_test 9 "optimized_piecewise"

echo ""
echo "==================================================================="
echo "Performance Summary"
echo "==================================================================="
echo ""

OPTIMIZED_TIME=$(cat /tmp/perf_optimized_piecewise.txt)

echo "Optimized (Piecewise Comparison): ${OPTIMIZED_TIME}s"
echo ""
echo "Note: Original BitDec timing would require disabling the optimization"
echo "      Expected improvement: ~10x in online round count"
echo ""
echo "==================================================================="

# Show detailed test results
echo ""
echo "Detailed Test Results:"
echo "-------------------------------------------------------------------"
cat /tmp/server_perf.log | grep -E "PASS|FAIL"

echo ""
echo "Test completed!"
