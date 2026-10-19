# Unified SDCF correctness regression

Run from the repository root:

```sh
python3 project/our/tools/segmented_dcf_regression/run.py \
  --output project/our/results/segmented_dpf_dcf_truncation/04/run
```

The entry point configures/builds the three existing correctness executables
with at most two build workers, then runs the following existing suites:

| Suite | Execution | Coverage |
| --- | --- | --- |
| `oracle_literals` | Local fixed literals | Separate SCI/floor examples, 64-bit boundaries, invalid contracts |
| `native_oracle` | Real SCI two-party cross-check | Frozen native truncation/extension APIs versus the common oracle |
| `comparison_local` | Explicitly labelled local 2PC simulation | All threshold/query/nseg combinations in 1..8 bits |
| `comparison_socket` | Real two-party keygen/evaluation/scan | Full-tree, segmented and legacy full-tree comparison |
| `adapter_socket` | Real two-party arithmetic reconstruction | SCI-compatible and independent floor modes, full-tree/segmented, conversion and material ownership |

`report.json` contains the backend/mode coverage matrix, actual value and
configuration counts, suite exit codes, first mismatch, and explicit
`failed`, `timeout`, `oom`, `unavailable` or `unsupported` rows. Uncompleted
rows have `values: null`; neither a missing binary nor a resource failure
counts as a passing zero-cost case. SCI's native signed 64-bit division by
`2^63` stays `unsupported`, while its independently specified oracle literal
and the adapter boundary case are checked. Passing tested cases with that
known gap yield `passed_with_unsupported`, not a claim of complete native
coverage.

`identity.json` records the exact command, checkout commit, tracked changes,
SCI revision, reference-source status and build path. A dirty source diff,
when present, is saved as `source.patch`. Each suite saves exact commands,
raw process logs and execution metadata. Native SCI additionally preserves
its original compiler command, party logs, sampled vectors and summary.

`--timeout` (default 240 seconds) is per suite and includes native cross-check
compilation. `--memory-gib` (default 1.5 GiB) caps each suite process's address
space; it is not a peak-RSS measurement. Processes and their descendant
process groups are stopped on timeout or peer failure. Workloads execute
sequentially. The comparison socket suite already includes a 12,800-row
batch; SDCF-03 retains the independently measured large adapter batch.

`--suites` selects a subset, reported as `partial`, and `--no-build` reuses
binaries only when the CMake cache belongs to this checkout. A new output
directory is required to avoid silently mixing separate executions.

Run the independent execution-state tests with:

```sh
python3 -m unittest discover -s project/our/tools/segmented_dcf_regression -p test_runner.py
```

Those tests exercise actual process failure, timeout, allocation failure, and
missing executables; they do not simulate protocol correctness.
