# Correlated-DPF Benchmark

This benchmark was moved from the `test/benchmark` tree of the sibling
`dealerless-FSS_public` checkout so the paper project owns its DPF control-word
experiments. Its backend implementations use the paper-owned dFSS sources in
`project/our/src/dfss`.

The copied benchmark source is distributed under the Apache License 2.0; see
`LICENSE`. Enable it with `-DBUILD_DFSS_EXT_BENCHMARKS=ON`. Run the
`build/test/benchmark/DFSS_EXT_BENCH` binary on both parties with
`bench=cw_generation`, `bin=N`, and `protocol=beaver|figure10`. Recorded
results are in `project/our/results/cw_generation_results.md`.
