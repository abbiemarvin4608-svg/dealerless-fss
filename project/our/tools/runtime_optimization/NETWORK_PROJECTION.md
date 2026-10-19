# Round-aware communication-time projection

`network_projection.py` turns qualified, freshly measured offline/online byte
counters and causal RTT rounds into the requested transmission-plus-propagation
model. It does not measure LAN/WAN latency or infer local computation from
localhost wall-clock time.

```sh
python3 project/our/tools/runtime_optimization/network_projection.py \
  --round-evidence path/to/ROUND_ELIGIBILITY.json \
  --output-directory path/to/new-results

python3 -m unittest discover \
  -s project/our/tools/runtime_optimization -p test_network_projection.py
```

The CLI emits `NETWORK_MODEL.json` and `NETWORK_MODEL.zh.md`. Invalid or
unqualified evidence returns exit code 1 before writing model outputs.

## Input Contract

The root object requires `schema: 1`, `eligible: true`,
`cohort: "a2-parallel"`, and
`convention: "two-party exchange; causal send-flight clocks"`.
Exactly two unique rows must represent `SCI_preprocessed` and
`SCI_Beaver_segmented`, each with `eligible: true` and exact query coverage
`0..9`.

Each query has `phases.offline` and `phases.online`. Both phases require:

- `rounds`: a nonnegative integer from qualified causal trace evidence.
- `sent_bytes`: two nonnegative integer party counters, in party order.
- `received_bytes`: the two matching receive counters, in the same order.
- `trace_coverage: true`: all phase communication is covered by the trace.

The send counters must match the opposite parties' receive counters. The
upstream qualification owns round reconstruction, protocol source provenance,
fresh input correctness, material residency, resource limits, and actual argv
and environment checks. This calculator validates their declared contract and
does not replace those checks.

## Model

| Environment | Bandwidth (bit/s) | RTT (seconds) |
| --- | ---: | ---: |
| LAN | 1,000,000,000 | 0.0005 |
| WAN | 400,000,000 | 0.004 |

Bandwidth is decimal; MiB remains 2^20 bytes. Both parties' sent bytes are
summed, with each message counted once. The requested model treats their
aggregate volume as serialized over one bandwidth:

```text
transmission_seconds = 8 * sum(sent_bytes) / bandwidth_bits_per_second
propagation_seconds = rtt_seconds * rounds
communication_seconds = transmission_seconds + propagation_seconds
```

Each query's components are added before taking the median of ten fresh
queries. The JSON preserves exact per-query counters and modeled components,
as well as medians, ranges, and population standard deviations. Offline plus
online is also summed per query before summarization. Online model speedup
is baseline median divided by complete-method median. No localhost timing
is added, and setup, input/ready, output, and teardown are excluded from
these offline/online projections.

The tests use a literal worked example, enforce qualification and complete
coverage, reject fractional/negative counters and mismatched communication,
check per-query-before-median aggregation, exercise the executable CLI,
and verify zero-cost and rejection behavior.
