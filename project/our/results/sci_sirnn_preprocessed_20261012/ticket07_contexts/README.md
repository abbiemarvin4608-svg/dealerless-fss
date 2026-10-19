# Ticket 07: isolated native preprocessing contexts

All 16 final batch-128 single-step diagnostic rows passed: four backends, two
execution cohorts, and default/nonzero hidden states. Each row ran two fresh
queries. Every query's 15 tensors and 172928 integers match both its freshly
executed SCI control and the frozen native-SCI reference. These diagnostic
rows establish correctness and resource eligibility; ticket 08 measures formal
ten-query performance. SDCF-09's 99-step graph remains deferred.

The four live native protocol contexts have stable ownership. Sequential
four-worker activations use slots 0,1,2,3. Concurrent two-worker branches assign
Sigmoid to slots 0,1 and TanH to slots 2,3, independently of completion order.
All three prepared backends physically prepare their OT and comparison
bit-triple inventories before loading each query's inputs. Their complete
public call plans, COT width vectors and one-use stock balances agree across
both parties. Native and bridge generation epochs remain unchanged throughout
input, ready, online and output, including the material-ready/input gap. SCI's
control rows retain their actual online generation behavior.

Private overlays now enforce and observe the requested team size at all six
native SCI protocol team sites: two LinearOT sites, one auxiliary-protocol
site, and three comparison sites. Both OT directions still execute. Each
activation worker uses one participant; four sequential workers total four,
and two concurrent branches peak at two plus two. Eigen uses one thread.
The four-core cap also governs CGGM and material-generation workers. Shared
structured logging emits complete records atomically during parallel stock
consumption. Communication includes forward, reverse and additional channels;
overlapped graph groups report measured wall time rather than summed task time.

Each party ran on four fixed CPUs, with a 1610612736-byte RLIMIT_AS. Server
affinity was 0,2,4,10 and client affinity was 6,8,12,16. All diagnostic rows
completed without timeout or OOM. Peak RSS in KiB was:

| Hidden state | Cohort | Backend | Server | Client |
|---|---|---|---:|---:|
| default | a4-sequential | SCI | 370224 | 229540 |
| default | a4-sequential | SCI_preprocessed | 829128 | 527936 |
| default | a4-sequential | SCI_ours | 808020 | 669464 |
| default | a4-sequential | SCI_Beaver_segmented | 478632 | 297300 |
| default | a2-parallel | SCI | 376900 | 235776 |
| default | a2-parallel | SCI_preprocessed | 872696 | 536920 |
| default | a2-parallel | SCI_ours | 888140 | 753632 |
| default | a2-parallel | SCI_Beaver_segmented | 540160 | 318044 |
| nonzero | a4-sequential | SCI | 373788 | 232908 |
| nonzero | a4-sequential | SCI_preprocessed | 827112 | 528156 |
| nonzero | a4-sequential | SCI_ours | 807856 | 669680 |
| nonzero | a4-sequential | SCI_Beaver_segmented | 478504 | 308372 |
| nonzero | a2-parallel | SCI | 374192 | 241964 |
| nonzero | a2-parallel | SCI_preprocessed | 879664 | 534188 |
| nonzero | a2-parallel | SCI_ours | 877332 | 753320 |
| nonzero | a2-parallel | SCI_Beaver_segmented | 538140 | 318144 |

The following native stock counts are per party, per query; they are identical
for both hidden states. All four contexts are covered, remaining stock is zero,
and padding is discarded rather than reused. Triple padding is zero. These
counts describe native residual protocols; segmented FSS keys and arithmetic
Beaver stocks retain their separate recorded inventories.

| Cohort | Backend | OT calls | Consumed OTs | OT padding discarded | Consumed bit-triples |
|---|---|---:|---:|---:|---:|
| a4-sequential | SCI_preprocessed | 559 | 6237216 | 765408 | 1902336 |
| a4-sequential | SCI_ours | 444 | 4529056 | 553056 | 773632 |
| a4-sequential | SCI_Beaver_segmented | 416 | 3136000 | 413184 | 422400 |
| a2-parallel | SCI_preprocessed | 351 | 6237216 | 721888 | 1902336 |
| a2-parallel | SCI_ours | 236 | 4529056 | 512608 | 773632 |
| a2-parallel | SCI_Beaver_segmented | 208 | 3136000 | 372736 | 422400 |

The real two-party primitive harness additionally covers the uneven count-501
tail, two consecutive queries, reversed branch execution order and parallel
execution. The red run observed eight actual participants before the quota
fix. The green run observes four while preserving every output integer. Wrong
context and wrong direction calls both reject with inventory exhaustion on
both parties, exit code 2 and generation delta zero. See
[primitive evidence](primitives/README.md).

The complete frozen execution plan, commands, configs, schedules, raw party
logs, collected records and verification result are under [diagnostics](diagnostics).
Its [integer verification](diagnostics/integer_verification.json) records every
row's equality, resource and stock checks. The build source archive preserves
61 source/overlay/runner snapshots and the exact compiler commands. The
primitive archive preserves all 27 local and overlay source dependencies.

Frozen runtime SHA-256:
`32f796a4bf50217e03fe523107e62aedef59b4691f73b176d801b5eb2026aeee`.
Frozen bridge SHA-256:
`b9d753d1df474f23c2a32484dc4bd3e65a242d04478caf5c10518ec07d476f0a`.
The native source state is commit `7f38a26`; later ticket 07 changes only
verification and evidence capture. No runtime rebuild occurred between the
final diagnostic matrix and ticket 08's formal plan.

Verification command, run from the repository root:

```sh
python3 project/our/tools/segmented_dcf_runtime/verify_preprocessed_contexts.py verify \
  --directory /tmp/sci-preprocessed-07-final-diagnostics \
  --reference-dir /home/jiang/dealerless-lunwen/project/our/results/runtime_optimization_20261010/baseline/diagnostic
```

The verifier completed with `{"status": "passed", "rows": 16}`. The targeted
Python suite passed 41 tests with one compiled-fixture skip; the real existing
arithmetic COT harness passed, and all three real context/material tests passed.
