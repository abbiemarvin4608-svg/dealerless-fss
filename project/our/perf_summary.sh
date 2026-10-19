#!/bin/bash
# Simple performance verification

echo "====================================================================="
echo "Sigmoid/Tanh Optimization Summary"
echo "====================================================================="
echo ""

# Run test once to get actual metrics
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=2 t=9 > /tmp/srv.log 2>&1 &
sleep 1
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=3 t=9 > /tmp/cli.log 2>&1
wait

echo "Test Results:"
echo "---------------------------------------------------------------------"
grep -E "PASS|FAIL" /tmp/srv.log | head -12

echo ""
echo "Summary:"
PASS_COUNT=$(grep -c "PASS" /tmp/srv.log)
FAIL_COUNT=$(grep -c "FAIL" /tmp/srv.log)
echo "  Passed: $PASS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo ""
echo "====================================================================="
