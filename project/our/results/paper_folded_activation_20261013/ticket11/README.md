# Ticket11 frozen complete-activation measurements

The final execution uses the persistent common build in `final-build/`:
`runtime/sdcf-runtime`, `core/libsegfss_sci_bridge.so`, and `fixture/`. The sole
final plan/output are `frozen-plan-final.json` and `formal-final/`. Its summary
and analysis are `complete`: ten qualified rows, eighteen complete groups,
180 fresh queries, eleven complete named comparisons and zero measurement gaps.
Supplementary sampling ended at `exhausted_budget`; both LUT-construction
comparisons remain online `unresolved` after their three-group cap. Tickets 11
and 12 are closed with those unresolved results retained. See the
[final performance report](formal-final/REPORT.md) and
[integrated Chinese report](../ticket12/REPORT.zh.md).

The current plan freezes exactly three passed qualifications:
`qualification-reuse-final.json`,
`final-build/qualification-budget/qualification.json`, and
`final-build/qualification-folded/qualification.json`. The first binds explicit
historical qualification reuse to the rebuilt identity; the latter two record
new default/nonzero SCI small-table DPF diagnostics and four default BitDec
folded diagnostics covering both LUT families at sequential-a1 and parallel-a2.
`requalify_rebuild.py` rechecks the retained execution and source proofs before
writing the reuse record. `final-build/build-input-reuse.json` identifies 62
runtime inputs with only the bridge output identity changed, and 146 matching
bridge source/normalized compile/link inputs. Historical runtime compile
commands were not retained; this source comparison does not claim exact
historical runtime command equality or the existence of the lost `/tmp` builds.
Build commands, current source identities and fixture provenance are retained
under `final-build/`; build/reproduction instructions are in
[ticket12/README.zh.md](../ticket12/README.zh.md).

The original `formal/` cohort retains nine qualified rows and a failed SCI
small-table DPF control. The outdated metadata reservation was repaired; see
[BUDGET_FIX.md](BUDGET_FIX.md). `formal-budget-fixed/` then retained nine
qualified rows and an address-space allocation failure in
`parallel-a2-full-bitdec` at query9. These failed cohorts and their plans remain
historical evidence. The isolated allocator diagnostic and
`formal-allocator-fixed/` are also excluded from current statistics.

`formal-allocator-clean/` is an invalid historical cohort for final inference.
Its original `/tmp` build vanished; supplementary logs preserve the resulting
missing-binary failures. Its manifest was subsequently rewritten to name a
different persistent runtime/fixture while keeping an incompatible claimed
binary identity, and failed group statuses were reset to `not_started`.
The original manifest, failed-group summary backup, changed manifest and logs
are retained and indexed in [historical-cohort-audit.json](historical-cohort-audit.json).
Those rewritten identities and status changes do not restore qualification.

Two subsequent attempts are archived with explicit exclusion records:
`formal-final-invalid-analyzer-mismatch-20261013/` completed its first row,
then stopped because `analyze.py` changed after freezing;
`formal-final-invalid-qualification-selection-20261013/` was intentionally
stopped after its plan selected only the old `ticket09/step-final`
qualification. Their plans, manifests and raw execution evidence remain
available. The qualification omission is corrected in `frozen-plan-final.json`.
Every sample in the current cohort is fresh; no prior attempt supplies timing
samples or supplementary groups.

`benchmark.py freeze` records one common runtime/bridge build, fixture, passed
diagnostic artifact identities, commands, the ten-row order, eleven named
parent/candidate comparisons and the existing BCa method. It does not build or
benchmark. `benchmark.py run` reuses that build and executes rows sequentially
under `/tmp/sdcf-measurement.lock`; every declared row is recorded before the
first run. Timeout, OOM, interrupted and unstarted rows remain visible. No
smaller-batch replacement or historical sample enters this cohort.

The eight BitDec/control rows are both LUT families at sequential-a1 and
parallel-a2, full-LUT parallel-a1 branch attribution, and three parallel-a2
controls: `SCI_preprocessed`, `SCI_Beaver_segmented`, and
`SCI_Beaver_segmented_DPF_LUT`. Two full-LUT segmented-sign candidates match
the sequential-a1 and parallel-a2 BitDec parents. All rows use batch128,
steps1, ten fresh queries, warmup0, default hidden state, a 600-second runtime
budget, fixed-4 outer segmentation, four pinned CPUs per party, a 1.5 GiB
address-space limit and a 512 MiB inclusive prefetch budget. Correctness
qualification artifacts separately cover default and nonzero hidden states.

```bash
python3 project/our/results/paper_folded_activation_20261013/ticket11/benchmark.py freeze \
  --binary project/our/results/paper_folded_activation_20261013/ticket11/final-build/runtime/sdcf-runtime \
  --bridge project/our/results/paper_folded_activation_20261013/ticket11/final-build/core/libsegfss_sci_bridge.so \
  --fixture project/our/results/paper_folded_activation_20261013/ticket11/final-build/fixture \
  --qualification project/our/results/paper_folded_activation_20261013/ticket11/qualification-reuse-final.json \
  --qualification project/our/results/paper_folded_activation_20261013/ticket11/final-build/qualification-budget/qualification.json \
  --qualification project/our/results/paper_folded_activation_20261013/ticket11/final-build/qualification-folded/qualification.json \
  --output project/our/results/paper_folded_activation_20261013/ticket11/frozen-plan-final.json
python3 project/our/results/paper_folded_activation_20261013/ticket11/benchmark.py run \
  --plan project/our/results/paper_folded_activation_20261013/ticket11/frozen-plan-final.json \
  --output project/our/results/paper_folded_activation_20261013/ticket11/formal-final
python3 project/our/results/paper_folded_activation_20261013/ticket11/analyze.py \
  --directory project/our/results/paper_folded_activation_20261013/ticket11/formal-final
python3 -m unittest discover -s project/our/results/paper_folded_activation_20261013/ticket11 -p 'test_*.py'
```

The displayed plan and output already exist; use fresh output names for a
reproduction. Do not run formal measurements concurrently with builds, other measurements or
quality computation. Existing output/plan paths are never overwritten. After
all ten initial rows finish and qualify, the runner selects the first frozen
comparison requiring extra evidence: unresolved, non-regressed two-stage step
admission, or unresolved complete Sigmoid/TanH online or LUT online event
windows in LUT-construction comparisons. It declares both fresh groups in
`summary.json` before executing candidate then baseline (the reverse of the
comparison's parent order), then analyzes again. Each group has ten fresh
queries. Each row has at most three groups, including shared controls. A failed
group stops supplementary sampling; a capped unresolved comparison reports
`blocked_by_group_budget` and remains experimental. Every selection and its
trigger metrics is retained in `extra_group_decisions`. A group is never
discarded or replaced.

The instrumented runtime controller runs under `/usr/bin/prlimit --as=1610612736`
as well as the two party limits, bounding post-party trace analysis. After each
group exits, passive raw traces are archived with gzip level 1, compared byte
for byte against their decompressed copy, and checked for a complete footer
before removing the raw copy. Failed partial traces retain explicit footer
status. Archive time and external controller resources are recorded separately
from protocol latency.

The explicit command wrapper `instrumented_run.py` executes the existing
runtime unchanged and requests its single post-party trace analysis with
`ticket11_lut_union` scopes for the online LUT stage union. All original phase
and scope validation remains active. The union uses the existing complete
dependency DAG and selected-send-flight depth, preserving external dependencies
and parallel worker ownership; it never sums worker rounds. The bounded-memory
public `scope_groups` API reuses the analyzer's scope inventory. Offline LUT stages
retain their exact per-call scope records without an additional aggregate DAG
pass. Optional `analyze.py --replay-rounds` independently replays raw traces
after execution; it is intentionally not the default for large offline traces.

`analysis.json` schema1 contains all ten declared rows in `rows`, keyed by ID,
and all named parent links in `comparisons`. Each qualified row includes raw
query evidence under `data`, elapsed/work distributions, network projections,
checks and trace identities. `groups[]` retains every declared group's complete
data, qualification checks, trace identities and execution metadata. Pooled
`data.queries` adds the original `group` number without renumbering query IDs;
its scalar session metadata describes group 1, and `data.sessions` retains all
sessions. `accounting` aggregates setup/table/model/first/session costs across
groups and conservative memory maxima. `cost_distributions` includes phase and
activation bytes, causal rounds and wall/work samples. Missing, failed or unqualified rows have a reason;
their comparisons are unavailable. `REPORT.md` is the concise human view,
and `ELIGIBILITY.json` preserves explicit row qualification. `REPORT.md` is in
Chinese and includes session accounting, complete query/activation timing, LUT
opening/product/event windows, phase communication models, every activation
stage and comparison verdicts. Ticket12 consumed `formal-final/analysis.json`
with `frozen-plan-final.json`; its final `summary.json` and `REPORT.zh.md` are
`complete`.

Wall comparisons use the same-query maximum of complete party intervals.
Stage work sums are attribution. LUT online event windows use each party's
earliest LUT-open start and latest local-product end across owned workers,
including scheduling gaps; they are distinct from summed work. Full Sigmoid,
TanH and whole-step online intervals are measured directly. Per-kind isolated
offline activation wall is unavailable; the all-folded preparation timer is
direct. Public-table preparation is a subset of setup and is never added to
setup or first-query time a second time. `/usr/bin/time` controller records
include post-party trace analysis and are separate from party resource limits.

The LAN/WAN formula is `8 * both_party_sent_bytes / bandwidth + RTT * rounds`,
with LAN 1 Gbit/s / 0.5 ms RTT and WAN 400 Mbit/s / 4 ms RTT. It models
communication only. It neither adds localhost compute nor labels relay
measurements as paper hardware. Wall BCa ratios are baseline/candidate; the LUT
construction orientation is segmented/full. The existing method draws
independently within measurement groups, using 10000 draws and seed20261013,
then computes a ratio of pooled medians. Network-model comparisons retain the
existing deterministic rule on pooled medians: lower cost is improved, higher
cost is regressed, and equal cost is unchanged. They do not have BCa intervals.
LUT construction changes both tree segmentation and payload/B2A construction;
SCI method controls also change the numeric activation algorithm. Current
two-byte q10 indices use 102400 bytes across both activations/both parties;
the hypothetical 64000-byte tightly packed representation is unimplemented.

The completed row/group record is `formal-final/summary.json`.
The plan uses the same sanitized allocator environment for every initial and
supplementary group: inherited `MALLOC_*` and `GLIBC_TUNABLES` are removed, then
`MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=1`, and
`MALLOC_MMAP_THRESHOLD_=131072` are set. Four reverse-order supplementary
decisions produced eight extra groups, all complete; no historical attempt
contributed samples. Archived failures and interrupted attempts remain visible.

The matched segmented/full10 step-online ratios are 0.9931 (95% BCa
[0.9635, 1.0430]) at sequential-a1 and 0.9883 ([0.9124, 1.0299]) at
parallel-a2, both unresolved. Step-offline ratios are 1.5151
([1.4513, 1.5558]) and 1.5560 ([1.4792, 1.6033]), favoring full10.
Parallel-a2 segmented DPF with BitDec takes 0.207982 s step-online, including
0.089768 s complete Sigmoid and 0.089408 s complete TanH, plus 11.731217 s
step-offline. Its three SCI online ratios in ticket12 are descriptive medians;
the frozen SCI BCa comparisons use full10 as candidate and do not establish
segmented-vs-SCI statistical admission. The integrated recommendation is
`no_qualified_new_default`; each configuration keeps its named qualification.
