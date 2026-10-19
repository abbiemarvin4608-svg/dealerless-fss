# Persistent-pool matched measurements

For the 2026-10-10 runtime optimization baseline, pass
`--profile runtime-opt-01 --phase after` with the selected primitive/runtime
binaries and frozen fixture. This profile fixes the approved CPU masks,
30/10 DCF repetitions, five complete-conversion repetitions, ten single-step
repetitions, zero warmup, 120/600-second limits, and fixed-4 runtime rows.
It includes all four backends and cGGM limits 1/4 where applicable. The older
`pool-recheck` profile remains available for the historical pool experiment.
The profile writes the manifest before running, and `--plan-only` performs no
protocol calls. Do not build or run other workloads during the measurements;
the measurement lock coordinates runners, not compilers.

Each primitive row probes the complete Peer/SCI port set (base plus offsets
0, 3, 50, 100) below the system's ephemeral source-port range. The manifest
contains planned ports; `execution.json` and `commands.json` retain the actual
selected ports and commands. Probing reduces local collisions but does not
silently rerun a failed measurement. Socket bind errors remain explicit
failure reasons with null measurements.

The conversion bridge and runtime must be rebuilt together when their C ABI
metrics change. Preserve the matching shared bridge alongside any baseline
runtime, because the executable otherwise follows its build-time rpath.
`CONVERSION_DETAIL` now records material loading, query preparation, local
evaluation, scan, rounding, B2A, release, serialized bytes consumed, key and
gate counts, and actual local evaluation participants. Requested worker limits
and domain segment widths are recorded by `CONVERSION_ONLINE`. These nested
call timings diagnose work; the directly measured online/query/session
intervals remain the latency results. Missing native detail stays absent and
must not be interpreted as zero. The driver retains both default-state and
nonzero-state diagnostics separately from secure timing rows.

`benchmark_pool.py` freezes its manifest before execution, holds the existing
measurement lock, and runs all backends sequentially. The before phase uses a
copied executable from the implementation before the pool change; the after
phase uses the new primitive executable and a newly built unified runtime.
Keep both binary identities and both manifests with the results.

The default resource allocation gives the server CPUs `0,2,4,10`, the client
CPUs `6,8,12,16`, and the relay CPU `14`. These are disjoint physical cores on
the current machine. All rows use these same allocations, including the SCI
and Beaver controls. The script verifies available CPUs and physical core
separation. Local cGGM workers vary between one and four; SCI protocol workers
remain one. `OMP_NUM_THREADS=1` is recorded, although explicit SCI OpenMP
`num_threads` clauses can override that setting. This allocation needs a fresh
SCI baseline; historical results pinned each party to a single core.

The first matched pilot used a minimum of1024 estimated hashes per local
participant. With count128 and16-bit comparisons, a four-worker ceiling still
raised evaluation time from35.3 to87.5microseconds for nseg2 and from37.1
to95.6microseconds for nseg4. These pilot measurements motivated a minimum
of4096 estimated hashes/expansions and32 contiguous segment tasks per
participant. The count128 walks now run on the caller; count12800 walks use
four participants for both nseg2 and nseg4 when the configured ceiling is
four. The pool still starts three resident workers during setup. Preserve
the pilot separately and rerun the full matched matrix for final results.

From the repository root, after preserving the old primitive executable:

```sh
python3 project/our/tools/primitive_cost_sweep/benchmark_pool.py \
  --phase before \
  --old-binary /tmp/sdcf-prepool-96581cd/primitive_cost_sweep \
  --old-source 96581cd \
  --large-batch \
  --output project/our/results/segmented_dpf_dcf_truncation/10/pool_before
```

Rebuild the shared bridge and primitive executable after the implementation,
then compile a new runtime from the frozen graph:

```sh
cmake --build project/our/build \
  --target segfss_sci_bridge primitive_cost_sweep -j2
python3 project/our/tools/sci_conversion/build.py \
  --out /tmp/sdcf-pool-runtime \
  --generated project/our/results/ticket07_hybrid_full/sources/sirnn_fixed.cpp \
  --bridge project/our/build/libsegfss_sci_bridge.so --jobs 2
python3 project/our/tools/primitive_cost_sweep/benchmark_pool.py \
  --phase after \
  --new-binary project/our/build/test/benchmark/primitive_cost_sweep \
  --new-source <implementation-commit> \
  --runtime-binary /tmp/sdcf-pool-runtime/sdcf-runtime \
  --fixture project/our/results/ticket07_hybrid_full/runtime_backup/fixture \
  --large-batch \
  --output project/our/results/segmented_dpf_dcf_truncation/10/pool_after
```

The output directory must be new. `--plan-only` writes the full matrix and
commands without running measurements. The defaults are:

| Workload | Configurations | Fresh queries per row |
| --- | --- | ---: |
| 16-bit DCF, count128, before and after | nseg2/4, local workers1/4 | 30 |
| Optional16-bit DCF, count12800, before and after | nseg2/4, local workers1/4 | 10 |
| After: signed32-bit div_pow2, shift14, count128 | nseg policy2/4, local workers1/4 | 5 |
| After: one recurrent step, Batch128 | SCI and SCI_Beaver, local worker setting4 | 5 |
| After: one recurrent step, Batch128 | SCI_ours and SCI_Beaver_segmented, nseg policy2/4, workers1/4 | 5 |

Every query consumes fresh offline materials. There is no warmup. The link is
unshaped localhost TCP, and each party has a1.5GiB address-space cap. Primitive
rows have a120-second timeout; runtime rows have a600-second timeout.
`SEGFSS_*`, `SDCF_*`, and inherited connector offsets are removed from child
environments; each runner then applies its explicit experiment settings.

For runtime rows, `max_segments` is a policy. Each active low/sign/wrap
comparison uses `min(max_segments, comparison_bits)` segments; a zero-width
domain has no comparison. Exact widths and actual segment counts are in the
`CONVERSION_OFFLINE` and `CONVERSION_ONLINE` logs. Batch128 recurrent vectors
contain12800 elements. Under nseg2, a32-bit comparison creates25600 depth16
keys together; its tree buffers exceed the declared memory budget. The pool
change does not add offline tiling. Actual failed attempts retain OOM or
timeout status and null completed measurements. The count128 primitive and
conversion rows supply complete nseg2 measurements when they pass.

`summary.json` retains every status. Primitive output verification reconstructs
the synthetic DCF/conversion outputs after the timed intervals. Online and
offline primitive times are medians of aligned per-query party maxima; local
evaluation and scan also retain each aligned value, min/max, and standard
deviation. Runtime records preserve setup, fresh offline, online, output, and
direct session totals. Communication sums both parties' sends once. Pool
creation belongs to setup, so it must not be omitted from the primitive setup
interval or deferred into online evaluation.

The after phase also runs independent default and nonzero-hidden diagnostics with one
query per completed backend/nseg at four local workers. It compares all15
graph-level tensors to native SCI through `compare_diagnostics.py`; these
diagnostic runs are excluded from timing summaries. Failed secure rows skip
diagnostics to avoid repeating the same allocation failure. Diagnostic results
and the hidden artifact identity are retained separately.

This is primitive, conversion, and single-recurrent-step evidence. SDCF-09
remains deferred; these measurements do not establish full Google-30 graph
correctness, Top-1 quality, or an end-to-end inference speedup.
