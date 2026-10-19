# Allocator Environment And Address-Space Diagnosis

The second cohort, `formal-budget-fixed/`, retains nine qualified rows and a
failed `parallel-a2-full-bitdec` row. Queries 0--8 completed; query9 failed
while preparing the final TanH worker with `std::bad_alloc`. Party2 VmPeak
was 1606959104 bytes, leaving 3653632 bytes below the 1610612736-byte cap.
The failure means the next allocation could not be satisfied within the cap,
not that a recorded VmPeak exceeded it.

The material, DCF working sets and generated/consumed balances match the old
qualified row. Party2 address space increased by 100 MiB per query in the new
run and about 50 MiB in the old run. Four full-LUT calls retain 6400 * 1024 * 4
bytes each, or 100 MiB per query. Preparation occurs on the main thread and
consumption/release occurs on several execution threads. The explicit owners
release their materials; allocator arena retention or fragmentation explains
the observed pattern without establishing an ownership leak.

A separate ten-query diagnostic used the same runtime and protocol workload,
with these launch environment values:

```text
MALLOC_ARENA_MAX=2
MALLOC_TRIM_THRESHOLD_=1
MALLOC_MMAP_THRESHOLD_=131072
```

Both parties completed all ten queries. Sampled VmPeak was
665567232/676253696 bytes; party2's later prefix address-space observation
remained 592916480 bytes. The result is retained at
`/tmp/paper-folded-allocator-trim-full-a2-20261013/result.json`.
The launch values come from the recorded diagnostic invocation; the runtime's
`commands.json` records its own environment updates and does not independently
attest these inherited values. All three values changed together, so this
diagnostic does not attribute the improvement to one parameter. Its timing
samples are excluded from formal statistics.

The current `frozen-plan-allocator-clean.json` freezes the same three values.
`benchmark.py` removes inherited `MALLOC_*` and `GLIBC_TUNABLES`, then applies
them to every row and supplementary group. The first instrumented controller's
startup environment (`/proc/648998/environ`) was observed to contain exactly
these allocator entries, with no `GLIBC_TUNABLES` or `LD_PRELOAD`; the parent
benchmark (`/proc/648954/environ`) had none of these entries. These are startup
observations for the named processes, not per-query environment attestations.

The `formal-allocator-fixed/` attempt preceding environment cleaning was
interrupted, and its original plan and raw evidence remain available. Its
samples, both older cohorts and the isolated diagnostic are excluded from
`formal-allocator-clean/` statistics. The current cohort reruns every row.

Allocator settings alter allocation, resource use and elapsed time; they do
not alter the protocol or integer mapping. Every comparison must use the same
settings. Aggressive trimming can add overhead, which remains in the measured
costs rather than being subtracted or treated as an algorithmic speedup.
