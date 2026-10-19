# Runtime Optimization Execution

Started 2026-10-10 (Asia/Shanghai), authorized by the user's implement-spec request.

- Integration branch: integration/sci-fss-runtime-optimization.
- Implementation/review base: 64ad3de.
- Scope: ten approved tickets; existing primitive, full conversion, Beaver and Batch=128 single-step interfaces. Historical SDCF-09 remains deferred.
- Pre-existing main-worktree change: extra indentation before compareDomain in sci_truncation.cpp. Preserve this unrelated edit through integration.
- Baseline binaries preserved in /tmp/runtime-opt-baseline. The runtime must use the saved shared bridge via LD_LIBRARY_PATH; copying only the executable does not freeze the baseline.
- Formal measurements use /tmp/sdcf-measurement.lock and run sequentially without concurrent compilation/heavy tests. Ticket 01 baseline is complete in project/our/results/runtime_optimization_20261010/baseline, with the preserved port-failure replacement in baseline-port-recheck.

## Work Graph

| Ticket | State | Branch / Evidence |
| --- | --- | --- |
| 01 | resolved | code 19446ea, port fix 30e359b; integrated through 4c38e10; baseline and diagnostics passed |
| 02 | resolved | 60c6b3b + 5112d24 + da1a421; fusion/pair/packed byte transport, 64 primitive rows, 11 conversion/material rows, real wire-tail and network checks retained |
| 03 | resolved | 9f2fffb + 5f5ff97 + fcac230; continuous layout and bounded same-query prefetch; diagnostics, bridge lifecycle and ten-query ablations complete |
| 04 | resolved | real two-party rings/layouts/alias/tile tests: 278824 integers; kernel controls and default/nonzero diagnostics complete; total-latency regressions retained |
| 05 | resolved | wide_confirm: six ten-query completions and two unpartitioned OOM rows; fixed2 tile8 Batch128 timeout retained; three policy conversions and depth-cap-8 timestep complete |
| 06 | resolved | independent online/offline AES; 64 primitive rows and seven kernel/AES ten-query rows, all numerical diagnostics passed; direct-query regressions retained |
| 07 | resolved | 3e87d4c; eight ten-query activation rows and 16 default/nonzero diagnostics complete; setup/RSS increase reported |
| 08 | resolved | explicit Peer/context ownership, sequential/isolated/parallel graph modes and MatAdd batching; 16 timing rows, 36 diagnostics, participant/context regressions complete |
| 09 | resolved | Beaver tiles 1/8/32/128 and bounded independent Beaver/FSS overlap; 11 ten-query rows, both-state diagnostics and all-tile oracle complete; historical receive telemetry limitation retained |
| 10 | resolved | final2 provides two matching shared-scheduling cohorts, eight formal rows, 16 diagnostics, and the Chinese Batch=128 report; retained raw audit and two-axis reviews are linked in the report |

## Shared Context

- Spec and approved tickets are adjacent to this file.
- Build exploration: /tmp/runtime-opt-build-notes.md.
- Parallel protocol exploration: /tmp/runtime-opt-parallel-notes.md.
- Material / batch-view coordination: /tmp/runtime-opt-material-notes.md.
- Current roles: runtime_ticket04 assembles completed evidence, runtime_ticket05 validates 09, runtime_merger validates 08 and owns review fixes/integration. Root owns final artifact, corrected wide resource rows, final activation controls, and report integration.
- Implementers use independent worktrees, TDD through the already approved behavior interfaces, then merge integration before handing off to a merger.
- Existing generated graph and frozen model/input fixture are referenced from the main worktree; they are ignored artifacts and do not appear in fresh worktrees.

## Verification

Baseline: six Batch=128 single-step rows, ten fresh queries each, all completed. Four backends in default and nonzero hidden state matched native SCI at 15 tensors / 172928 integers. All eight DCF rows passed; one conversion launch encountered a port conflict, retained as failed with a separate successful same-shape five-query replacement. Optimization tickets 01–10 are resolved with evidence and explicit limitations in each Answer. Final2 supplies the matched comparison; the report and retained raw audit record the final evidence boundary.

The two-axis review at 64ad3de...1043284 found four Spec issues and one telemetry issue. A single review implementer fixed request-level participant limits, aggregate prefetch budgets, same-layer MatAdd batching, and serial/overlap telemetry. Corrected wide-domain formal rows use ten fresh queries and preserve earlier one-query probes. Two independent conversion contexts passed real concurrent two-party tests for segmented/full-tree paths (144 literal integers each). The pool4/request1-or2/AES0-or1 regression passed. Fixed2/tile8 Batch128 timed out at its declared 600-second limit; completed costs remain null. Historical SDCF-09 remains deferred.

Cleanup audit found substantive uncommitted Ticket 02 work in its original
worktree. The patch is preserved at /tmp/runtime-opt-02-uncommitted.patch.
The single review implementer selectively integrated direct byte transport
and independent real wire-count/tail assertions in da1a421; obsolete grouping
prototypes were superseded by the integrated paired-extension plan. Existing
frozen artifacts and samples are unchanged. New builds use separate byte-core
and byte-runtime directories under the same measurement lock. The final table
will use the revised artifact after exact integer and communication checks.

Ticket closure evidence review: 02–09 retain their frozen raw cohorts and numerical diagnostics; newer byte transport results are a separate cohort. The retained-raw audit in `project/our/results/runtime_optimization_20261010/FINAL_RAW_AUDIT.json` independently verifies final phase logs, maximum-party distributions, sum-of-sends traffic, fresh material/call coverage, actual participant caps, and wide-domain tile/hash/memory accounting. Fixed-2 complete Batch128 remains a declared-budget timeout; historical SDCF-09 remains deferred.
