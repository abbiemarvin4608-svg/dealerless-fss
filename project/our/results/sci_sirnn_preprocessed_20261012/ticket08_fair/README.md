# Ticket 08: batch=128 fair preprocessing benchmark

All 8 formal rows and all 16 default/nonzero-hidden diagnostic rows qualify.
Saved actual child argv/environment match both the recorded configuration and
the declared runner flags; original fixture hashes are checked from disk.
The comparison uses the same final native runtime and bridge as ticket 07,
with actual query-independent random OT and safe bit-triple preparation in
all three prepared backends. Input-dependent corrections stay online.
The default SCI row retains online native generation as a control.
SDCF-09's 99-step graph, classifier and Top-1 validation remain deferred.

[REPORT.zh.md](REPORT.zh.md) gives the Chinese results, all 24 row qualifications,
phase latency/communication, setup, direct query/first-request/session costs,
RSS and the six within-cohort comparisons. [ELIGIBILITY.json](ELIGIBILITY.json)
contains every eligibility check; [COMPARISON.json](COMPARISON.json) preserves
all aligned query samples, min/max, population standard deviation and ratios.

The final ten-query measurement plan is [formal-plan.json](formal-plan.json).
[formal/](formal) stores its completed serial execution manifest, every declared
and actual command, environment/config/schedule, raw party/runner logs and
collected result. Warmup is zero. [diagnostic-plan.json](diagnostic-plan.json)
and [diagnostics/](diagnostics) preserve the sixteen two-query integer runs.
Each query's 15 tensors / 172928 integers match both a fresh SCI control and
the frozen reference. The [reference/](reference) copy holds the two original
SCI reference party logs and frozen hidden fixture, with original bytes and
hashes in [reference_sources.json](reference_sources.json). These references
are correctness evidence, never historical performance baselines.

The final build/source archive is
[../ticket07_contexts/build](../ticket07_contexts/build); it contains the
exact -O3 compile/link commands, generated driver, all private OT/comparison
headers and locally compiled native/bridge sources. Its source state is
`7f38a26`; subsequent ticket 07 changes affect verification/evidence only.
Frozen native runtime SHA-256 is
`32f796a4bf50217e03fe523107e62aedef59b4691f73b176d801b5eb2026aeee`;
bridge SHA-256 is
`b9d753d1df474f23c2a32484dc4bd3e65a242d04478caf5c10518ec07d476f0a`.
Both diagnostic and formal manifests/configs repeat these identities.

The native runtime's driver physically calls prepare, seal_prepare, sync and
material_ready before beginning input and reading query/hidden files.
The logs record MATERIAL_READY before the input interval completes; the
archived driver establishes the input-start/read ordering. The strict collector
also verifies zero native and bridge generation throughout input, ready,
online and output, including the material-ready/input gap. Selection correction,
message/correlation adaptation, comparison AND opening/correction, material
reads/waits and completed-key evaluation remain online. Public ScalarMul stays
local with no multiplication triples. OT padding is discarded per query,
never reused; every useful OT/triple stock is consumed once and balanced.

Session time is the measured FINISHED interval, taking the slower party.
Session communication sums actual counter differences for application_io,
setup, each query_total and teardown. These intervals cover the session without
overlap; first_query and individual query phases are not counted again.
Both parties' total sends match the opposite receives. Communication sums bytes,
whereas phase latency medians are never added to estimate direct total time.
[Independent arithmetic review](independent_math_review.md) agrees with every
main median and all eight session byte totals. The
[independent archive review](independent_archive_review.md) also checks all
24 rows and demonstrates rejection of mismatched actual execution contracts.

The original declaration paths are kept as historical evidence. In particular,
commands refer to the former ticket 07 worktree runner; the complete frozen
Python runner sources are copied under [runner/](runner), with hashes in
[runner_sources.json](runner_sources.json). The audit below imports this
archived runner and reference, so removing the implementer worktrees does not
remove either audit dependency. It still verifies the frozen native binaries
at their original `/tmp/sci-preprocessed-final-runtime` and
`/tmp/sci-preprocessed-final-core` locations, the generated model source and
shared external SCI source on this measurement host. It does not execute the
protocol, build code or modify the recorded experiment declarations.

Run from the repository root:

```sh
python3 project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/audit_preprocessed.py \
  --results project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair \
  --contract project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/runner/segmented_dcf_runtime/run.py \
  --reference-dir project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/reference
python3 -m unittest discover \
  -s project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair \
  -p test_preprocessed_report.py
```

The audit exits zero with `complete=true, eligible=true, formal=8, diagnostic=16`.
The focused qualification/report tests pass five tests. Their worked examples
check actual argv/environment disagreement, disabled preprocessing, missing
nonzero qualification, aligned party maxima/direct totals and complete
non-overlapping session communication.
See [VALIDATION.md](VALIDATION.md) and saved command output for the observed
red/green result. No protocol implementation or new formal sample was changed
while this report was generated.

The [final integration review](FINAL_REVIEW.md) records the independent
Standards and Spec reviews of the complete implementation. Both axes report
zero findings requiring a change.
