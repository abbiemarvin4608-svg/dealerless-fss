# SCI+SiRNN preprocessing implementation

Date: 2026-10-12 (Asia/Shanghai)

Integration branch: `integration/sci-sirnn-preprocessed`

Starting commit: `b577554`

## Ticket frontier

| Ticket | State | Implementation branch | Evidence |
| --- | --- | --- | --- |
| 01 | closed | implement/sirnn-prep-01 | [Real GOT logs](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket01_got/README.md); two real-party test methods and three collector checks passed |
| 02 | closed | ticket/sci-preprocessed-02 | [Real LUT/GOT logs](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket02_got/REPORT_CN.md) |
| 03 | closed | ticket03/prepared-arithmetic-cot | [Real arithmetic COT logs](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket03_cot/README.md) |
| 04 | closed | implement/sirnn-prep-04-triples | [Real comparison/conversion logs](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket04_conversion/README.md) |
| 05 | closed | ticket/sci-preprocessed-05 | [Native activation logs](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket05_activation/REPORT_CN.md) |
| 06 | closed | ticket06/shared-preprocessed-step | [Four-backend single-step evidence](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket06_step/README.md) |
| 07 | closed | ticket07/contextquota | [Final context diagnostic matrix](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket07_contexts/README.md) |
| 08 | closed | ticket08/fair-audit-report | [Chinese report](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/REPORT.zh.md); [24-row eligibility](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/ELIGIBILITY.json) |

The user-approved test interfaces are the real two-party primitives and
the existing FastGRNN single-timestep runtime at batch=128. Each protocol
change must demonstrate actual input-independent material generation,
input-dependent online adaptation, fresh one-use consumption and measured
phase accounting. Partial coverage does not qualify as SCI_preprocessed.

Ticket 01 integrated commits `2a9c328`, `018002a`, `481b896`, `b7c474c`.
Two fresh queries each requested and consumed 12 logical GOTs, generated
256 SIMD rows, and discarded 244 padding rows. The receiver sent three
packed correction bytes and one synchronization byte online per query.
Seven invalid lifecycle modes failed explicitly. The existing default SCI
batch=128 single-step diagnostic matched all 15 tensor rows / 172928 integer
values. These primitive checks establish correctness and accounting; they
are not formal preprocessing performance rows.

Tickets 02/03 integrated implementations `608b1c1` and `fb90152`. Native
radix LUT and both bit-matrix layouts reconstruct the same integers as
default SCI. Native B2A, signed matrix/Hadamard multiplication and mixed-width
arithmetic ring checks likewise agree with default. Per-query random pads
are physically completed before input; native generation is absent online.
After merging both branches, all ten real GOT/LUT/COT test methods and three
material collector checks passed. Compilation ran serially with the shared
build lock. Both native GOT and native COT overlay/copy paths remain active.
Canonical evidence records actual adaptation communication, material balances
and strict failures. These remain primitive diagnostics, not the final
batch=128 performance comparison.

Ticket 04 integrated branch tip `4eb73a2`. Four real planner/triple test
methods passed; 42 output vectors / 546 integers matched default SCI.
Each conversion query generated and consumed 1840 standard/correlated safe
bit-triples before input, with zero native generation in input/online/output.
Public plans cover native OT and triple consumption; incomplete, mismatched,
reused or unsafe material is rejected. Five affected collector checks passed
after the merge. Whole-step and activation coverage was subsequently completed
by tickets 05–08.

Ticket 05 integrated implementation `8b276ef`. Both real activation tests
passed at counts 13 and 12800, each using two fresh queries in default and
prepared modes. Sigmoid and TanH integer outputs agree exactly. At count
12800 the complete native activations consume 3136000 OTs and 422400 safe
bit-triples per query, physically generated before the material-ready barrier;
native generation remains zero after input. Scratch plans are identical for
zero and 65535 inputs without communication, material generation or PRG
advancement. Diagnostic openings are confined to output accounting. These are
activation diagnostics; tickets 06–08 subsequently completed the full
four-backend single-step comparison.

Ticket 06 integrated implementation `95d97b4` and qualification `18059c1`.
Eight real batch=128 diagnostic rows cover all four backends under default and
frozen nonzero hidden states. Each row runs two fresh queries, matching all
15 tensors / 172928 integers per query with the freshly executed SCI control.
Prepared rows verify complete public OT/triple plans, zero planning side
effects, one-use balanced material and no native/bridge generation after
prepare. Remaining native calls in both ours backends share the same policy;
replaced calls and Beaver/FSS's offline OT are not duplicated into online stock.
The missing-triple-owner regression fails explicitly. Canonical evidence
includes source snapshots, exact build/launch commands, phase and resource
logs. These are correctness runs, not ticket 08's ten-query performance rows.

Ticket 07 integrated final tip `5409fb0`. All sixteen final batch=128
diagnostic rows passed under a4-sequential/a2-parallel, both hidden states and
all four backends. Each row ran two fresh queries with 15 tensors / 172928
integers per query matching both a fresh SCI control and the frozen reference.
Four native contexts have stable slots, exact public call plans and one-use
inventories; all prepared input/ready/online/output generation deltas remain
zero. Six SCI protocol-team sites enforce observed participants totaling four
per party, alongside single-thread Eigen and the shared material/CGGM cap.
All rows met four fixed CPUs and 1.5 GiB RLIMIT_AS without OOM or timeout.
The uneven-count-501 harness verifies reversed branch ordering, fresh queries
and wrong-context/direction rejection, with red/green participant evidence.
The final verifier reports sixteen rows passed; the targeted Python suite has
41 passes and one compiled-fixture skip. Frozen binaries remain unchanged
between this matrix and the formal ticket 08 plan. Implementer worktrees were
removed after final review; ticket 08 archives their runner sources so its
audit does not depend on implementer worktrees.

Ticket 08 integrated implementation `4a6aa63` and completed all eight ten-fresh-query formal rows after the frozen
sixteen-row diagnostic matrix. The archived audit qualifies all 24 rows, checks
raw log recollection, full strict native/bridge material coverage and matched
resources, and independently aggregates aligned party maxima and actual sends.
Beaver+segmented FSS versus SCI_preprocessed is 1.870x / 1.451x faster online
(a4 / a2), with 58.56% less online communication; direct query totals rise
220.83% / 262.38%. Conversion-only SCI_ours online medians increase 7.88% /
20.09%, with overlapping a4 observed ranges. Five focused report tests pass.
Raw phase samples, direct query/first/session costs, complete session bytes,
commands, source identities, two-party logs and the Chinese report are archived.
Formal measurements ran serially under the frozen per-party CPU and memory
limits. All tickets are closed. The [Chinese report](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/REPORT.zh.md),
[24-row eligibility evidence](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/ELIGIBILITY.json)
and [focused validation](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/VALIDATION.md)
preserve the final reviewable result. SDCF-09 remains deferred.

Final independent Standards and Spec reviews of
`b5775543bfaa6ac1249ca04317cb869206460835...af739fc457a08d58f80e7f020514cf5f38dcb050`
both report zero findings requiring a change. Their separate reports are
archived in the [final integration review](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/FINAL_REVIEW.md).
No runtime or benchmark changes were required by this review.

All three remaining implementer worktrees have been cleaned up with non-force
removal. The canonical archived audit passed again after removal:
`complete=true, eligible=true, formal=8, diagnostic=16`. Frozen binaries,
fixtures and source dependencies remain available; no result samples changed.

The existing changes in the previous runtime-optimization index and
`sci_truncation.cpp` predate this implementation and are not part of these
ticket commits.

## Actual memory-ready follow-up

The user requested fresh measurements with both backends' query materials
physically resident before input, under the same 1.5 GiB per-party budget.
The unchanged frozen runtime supports prepare-time bounded prefetch. The
follow-up uses prefetch_bytes=536870912 within that process limit and requires
complete inventory coverage rather than allowing a partial prefix.

All 16 diagnostic rows and eight ten-query formal rows qualify. Every FSS
conversion has a resident hit and zero online file-loading time. Seven focused
qualification tests pass; the final independent review reports no findings.
Complete SCI+ours is 2.423x / 1.848x faster online than SCI+SiRNN preprocessed
(a4 / a2), with 58.56% less online communication. Direct total query time is
6.853792 s / 7.144087 s versus 2.092429 s / 2.004052 s; these measurements
retain actual offline generation and loading. Conversion-only ablation online
medians are 569.308 ms / 537.953 ms versus 553.881 ms / 437.472 ms, with sample
ranges overlapping the baseline in both cohorts.

The [follow-up report](../../project/our/results/sci_sirnn_resident_20261012/REPORT.zh.md),
[eligibility](../../project/our/results/sci_sirnn_resident_20261012/ELIGIBILITY.json)
and [review](../../project/our/results/sci_sirnn_resident_20261012/INDEPENDENT_REVIEW.md)
preserve fresh measurements, not deductions from historical loading intervals.
All eight tickets remain closed; SDCF-09 remains deferred.
