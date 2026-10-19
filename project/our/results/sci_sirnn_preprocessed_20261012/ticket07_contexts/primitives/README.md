# Ticket 07 native context and material harness

Three real two-party tests passed in 25.917 seconds using ticket 07's private
SCI quota overlays and shared structured logger. The compiled manifest records
legacy_observation_sites=0: final runs use actual runtime instrumentation.

## Positive Runs

The native sigmoid and tanh chains use count 501, two fresh queries and the
same boundary-heavy input shares. A newly executed single-worker default row
provides the integer reference. Four-worker default/prepared rows and two-worker
isolated/parallel default/prepared rows match all four output vectors exactly.

Four workers own chunks 126,125,125,125 at offsets 0,126,251,376 and slots
0,1,2,3. Each worker observes one OT participant; the aggregate observed bound
is four. Two-worker branches own chunks 251,250 at offsets 0,251. Sigmoid uses
slots 0,1 and tanh uses slots 2,3; parallel branch peaks are two plus two.
Isolated execution changes branch order while preserving the same ownership.

Every prepared row checks complete one-use OT plans, full COT width vectors,
comparison bit-triple inventories, balanced consumption, and equal public plans
across both parties. A traversal over two distinct public scratch patterns
must produce the same plan without communication, native/bridge generation,
or PRG-block changes. Actual query loading follows material-ready. Input,
ready, online and output generation deltas are zero for both native and bridge
materials. All structured protocol records parse successfully during concurrent
stock consumption.

## Negative Runs

Only context 0's straight B2A COT stream is prepared. Calling actual B2A on
context 1, or selecting context 0's reversed OT pointer for B2A, fails with
arithmetic COT inventory exhaustion. Both parties exit 2; the generation epoch
does not change and no fallback stock is generated.

## Red Evidence

The sibling primitives_red/default_a4_sequential_501 directory records the
meaningful pre-fix failure from base 675c2e9. Observation-only instrumentation
measured the unchanged SCI comparison/multiplexer fixed-size teams: each of
four workers requested one participant but actually used two, producing an
observed bound of eight. Integer output was already equal to the reference;
the failing assertion was the actual participant quota. The harness retains
observation-only instrumentation solely for uninstrumented legacy sources.

## Execution

Every party uses four fixed CPUs and a 1610612736-byte address-space limit:
server 0,2,4,10 and client 6,8,12,16. Compilation and protocol runs acquire
/tmp/sci-preprocessed-build.lock. Per-row result.json preserves process exits,
commands and the five relevant worker environment settings. Protocol logs
include the observed groups, stock identities and peak RSS. The build folder
preserves the compiled private sources, overlays, source hashes, binary hash
and compiler command.

```sh
env SDCF_CONTEXT_EVIDENCE_DIR=/absolute/new-evidence \
  python3 -m unittest test_native_context_materials
```

Run from project/our/tools/segmented_dcf_runtime. This is ownership/correctness
evidence; formal ten-query performance remains ticket 08. SDCF-09 remains
deferred.
