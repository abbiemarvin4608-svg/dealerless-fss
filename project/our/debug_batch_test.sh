#!/bin/bash
# Debug batch sigmoid test

echo "Testing individual sigmoid first..."
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=2 t=9 > /tmp/individual.log 2>&1 &
sleep 1
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=3 t=9 2>&1
wait

echo ""
echo "Individual sigmoid results:"
grep "sigmoid(1)\|sigmoid(0)\|sigmoid(2)" /tmp/individual.log

echo ""
echo "Now testing batch..."
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=2 t=14 > /tmp/batch_debug.log 2>&1 &
sleep 1
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=3 t=14 2>&1

echo ""
echo "Batch results:"
cat /tmp/batch_debug.log
