# SDCF-10 primitive cost scan

`manifest.py` declares candidates and per-party resource limits before any row
is executed. `runner.py` runs real two-party localhost processes sequentially,
retains commands, exit codes and both raw logs, and leaves measurements `null`
for every incomplete configuration. A peer allocation failure terminates the
other party promptly; an unclassified signal is a failure, not inferred OOM.

Build and run from the repository root:

```sh
cmake --build project/our/build --target primitive_cost_sweep -j2
SDCF_SWEEP_TEST_BINARY=project/our/build/test/benchmark/primitive_cost_sweep python3 -m unittest discover -s project/our/tools/primitive_cost_sweep -v
python3 project/our/tools/primitive_cost_sweep/manifest.py --output /tmp/sweep-manifest.json
flock /tmp/sdcf-measurement.lock python3 project/our/tools/primitive_cost_sweep/runner.py --binary project/our/build/test/benchmark/primitive_cost_sweep --output /tmp/sweep
```

`SDCF_SWEEP_TEST_BINARY` enables the real two-party DCF and conversion output
verification regression. Without it, this executable-dependent test is skipped.

Full-tree means the current SCI-compatible adapter's batched cGGM tree with
`nseg=1`; segmented uses the same implementation with multiple trees and a
balanced Boolean scan. This is a DPF-based DCF: it returns both strict-less
and equality shares. It does not benchmark the historical `exactTruncate`
adapter, whose `shift <= 24` limit is unrelated to this independent DCF API.
The 32-bit conversion supplement uses unsigned `truncate_and_reduce` from
the 64-bit input ring to the 32-bit output ring with `shift=32`.

The persistent worker-pool recheck in `benchmark_pool.py` instead uses signed
`div_pow2` with a 32-bit input ring and `shift=14`. Its primary low-borrow
comparison domain is 14 bits (`comparison_bits=14`); the 31-bit sign and
32-bit wrap comparison domains are recorded separately in
`active_comparisons`. Input ring width and comparison domain width are
different metadata fields.

The executable accepts `cggm-threads=N` and records `cggm_thread_limit` in
each party result. Zero retains the hardware default; positive values bound
local hashing workers. The protocol socket remains on the calling thread.
The historical acceptance scan predates batched parallel online evaluation.
For the separate 16-bit online recheck, including explicit one/four-worker
rows and ten fresh queries per row, use:

```sh
python3 project/our/tools/primitive_cost_sweep/recheck_online.py --binary project/our/build/test/benchmark/primitive_cost_sweep --output /tmp/online-recheck
```

Each row has three fresh runs in one session. Setup is connection/base-OT
initialization. Offline generates masks/B2A, correlated keys and Boolean
triples. Online consumes those materials. Reconstruction for correctness
occurs after the online counters and timers stop. The direct total measures
offline start through online finish. First-query cost adds measured session
setup; setup alone is reusable. No one-use material is reused across runs.

The output retains every party sample. Latencies use the median of the
aligned per-run party maximum; communication sums both parties' sends.
The current scanner checks reconstructed synthetic outputs after stopping
online timers and counters; only pass/fail is retained. The original SDCF-10
acceptance scanner relied on independent SDCF-02/04 correctness evidence.
`Peer::rounds`
is reported as transport I/O events, not logical rounds.
Online logical scan layers and adapter logical rounds are recorded; offline
logical rounds are not separable and stay `null`. Nested timings overlap:
comparison preprocessing contains key generation and scan preprocessing;
comparison online contains tree evaluation and scan. They must not be added
to their child timings. Local arithmetic correction and output-ring
conversion are not independently timed and stay `null`; counts and masked
bit conversion time remain available. DCF-only rows have zero adapter work.

The 1.5 GiB limit is `RLIMIT_AS` (virtual address space), not an RSS cap.
Peak RSS is separately observed per party. The limit is enforced around
real allocation attempts; no memory extrapolation is used to label a row.
