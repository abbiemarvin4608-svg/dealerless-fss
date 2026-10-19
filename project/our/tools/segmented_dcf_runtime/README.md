# SDCF fixed-batch runtime

This runner executes the frozen SCI mixed-width Google-30 graph. It uses
the existing `google30_hybrid/build_full.py` parser and preserves every
numerical call argument except the independent batch dimension. MatMul
operates on `batch x K` matrices; elementwise calls operate on the entire
batch. Native SCI Sigmoid/TanH remain unchanged. No full heldout evaluation
or Top-1 claim is made here; SDCF-09 is deferred by user instruction.

The default binary implements `SCI`. Requests for other registered names
(`SCI_ours`, `SCI_Beaver`, `SCI_Beaver_full`, `SCI_Beaver_segmented`) produce
`unavailable`, null measurements, and no automatic fallback. Subsequent
tickets implement these backends through the adapter interface below.

## Build and run

```sh
python3 tools/segmented_dcf_runtime/build.py \
  --out /tmp/sdcf-native-build \
  --generated /path/to/frozen/sirnn_fixed.cpp
python3 tools/segmented_dcf_runtime/run.py \
  --binary /tmp/sdcf-native-build/sdcf-runtime \
  --fixture /tmp/ticket07-fixture-v2 \
  --output /tmp/sdcf-native-step --backend SCI --batch 128 --steps 1
```

`--steps 99` executes all recurrence steps, classifier and ArgMax. The
same 128 fixed records are used for each query; setup is reused, and each
backend must generate fresh one-use preprocessing for every query.
`--queries 3 --warmup 0 --cpus 10,12,14 --cggm-threads 1 --timeout 3600 --memory-gib 1.5`
are the default frozen resource/measurement settings. Backends run in
separate sequential invocations, never concurrently during comparison.
The runner records compiler, binary/source identities, CPU model, exact
commands, environment overrides and configuration before execution.

The default uses one CPU and one local cGGM worker per party. To enable
four local workers, explicitly assign at least four available CPUs to
each party, for example:

```sh
python3 tools/segmented_dcf_runtime/run.py \
  --binary /tmp/sdcf-native-build/sdcf-runtime \
  --fixture /tmp/ticket07-fixture-v2 --output /tmp/sdcf-parallel-step \
  --backend SCI_Beaver_segmented --steps 1 \
  --cggm-threads 4 --server-cpus 0,1,2,3 --client-cpus 4,5,6,7 --cpus 0,4,8
```

CPU lists accept comma-separated IDs available in the current process's
affinity. The third `--cpus` value remains the network relay CPU. SCI
protocol threads and `OMP_NUM_THREADS` remain one. The runner injects
`SEGFSS_CGGM_THREADS` after removing inherited benchmark overrides and
records the local worker limit, protocol thread count, requested affinity
and observed party affinity separately. `run_matrix.py` forwards the same
options. Cost comparisons require matching worker limits and actual
affinities; historical metadata is interpreted as one local worker with
one CPU per party. Existing single-thread results remain single-thread
evidence and cannot establish a parallel speedup.

Network profiles are `localhost`, `sirnn_lan` (local payload relay at
0.5 ms RTT and 1000 Mbps), and `wan` (local payload relay at
4 ms RTT and 400 Mbps). Historical artifacts retain their recorded profiles.
The joint optimization experiment uses unshaped localhost measurements and
separately models these LAN/WAN links from actual bytes and causal rounds.
These relays do not reproduce the original paper
hardware or an Internet route. Both SCI and added bridge sockets must use
the documented reserved port offsets and connection redirection.

To write both communication models during a localhost run, pass
`--round-trace --network localhost --network-projection both`.
`--network-projection lan` or `wan` selects one model. These options use the
qualified trace and do not shape protocol traffic. For each fresh query's
offline and online phases, the runner computes:

```text
transmission_seconds = 8 * sum(party_sent_bytes) / bandwidth_bits_per_second
propagation_seconds = logical_rounds * rtt_seconds
communication_seconds = transmission_seconds + propagation_seconds
```

LAN uses `1000000000 bit/s` and `0.0005 s RTT`; WAN uses
`400000000 bit/s` and `0.004 s RTT`. The byte total counts both parties'
sends once. Rounds use the causal critical path, including fork/join
dependencies across parallel channels. `result.json` writes the per-query
components and phase medians to
`measurements.network_projection.networks.LAN/WAN`; each
`measurements.queries[*].offline/online` also contains `logical_rounds`,
`party_sent_bytes`, `party_received_bytes`, and `network_projection.LAN/WAN`.
Warmup queries are excluded from model summaries. The
`offline_plus_online_seconds_median` field sums those two phases per query
before taking the median. Wall-clock measurements remain separate from
the communication model. Without a qualified round trace, the projection
has `eligible=false` and a reason.

## Stages and evidence

Per-party `METRIC` records physically separate application I/O, setup,
per-query offline preparation, input loading, ready barrier, online
computation, output reconstruction and teardown. Online begins only after
private inputs and hidden state are loaded and confirmed by the barrier.
Native SCI online OT remains online. Its offline callback does no work;
the tiny measured empty interval remains visible instead of being inferred
by subtraction. `total_seconds` is measured directly, including reporting.

Each phase retains per-party time, sent/received bytes and send/recv call
counts. Summary wall time is the maximum party value; communication is
the sum of the parties' sends, counted once. With `--round-trace`, qualified
causal rounds are attached to each atomic query phase after trace analysis;
unavailable dependency counts remain null. Socket calls are not relabeled
as rounds. RSS is per process, in KiB. The runtime checks
that the adapter's offline-generation epoch does not change online.

`schedule.json` records every graph-level layer's public arguments,
dependencies, tensor width, shape and batching group. `COVERAGE` records
prove each query consumed every scheduled graph call. Internal native
SCI conversions and activation operations are opaque here; a replacement
backend must log its actual subcall coverage separately. Final classifier
sign extension and rowwise ArgMax use native SCI and must be retained or
explicitly added to a replacement's preprocessing schedule.

Timeout, allocation failure and unavailable backends retain status and
null completed measurements. A failed run is not a zero-time result.
Secure one-step workloads reveal only the final hidden-state output to
the client; 99-step workloads reveal predictions. Intermediate values
appear only in diagnostic runs and never constitute performance evidence.

## Adapter API

`backend_types.h` defines `Layer` with index, step, operation name,
function, complete original source call, all numeric arguments, input
tensor names, output tensor, shape, output width and element count. Its
first numeric argument is the actual batch. The entire ordered schedule
is passed to `setup` and `prepare`, enabling input-independent generation
or a disk-backed material stream. Per-domain segment counts must be
validated explicitly; `SDCF_MAX_SEGMENTS` is a public policy parameter,
not permission to silently repair an illegal `nseg` request.

Implement the following functions in namespace `sdcf_backend`:

```cpp
std::vector<std::string> supported_backends();
void validate(const std::string& backend);
void setup(const std::string& backend, int sci_party,
           const std::vector<sdcf_runtime::Layer>& schedule);
void prepare(int query, const std::vector<sdcf_runtime::Layer>& schedule);
void enter_online(int query);
void layer_begin(const sdcf_runtime::Layer& layer);
void layer_end(const sdcf_runtime::Layer& layer);
void leave_online(int query);
void finish();
sdcf_runtime::Counters counters();
```

`setup` runs after SCI initialization. `prepare` must finish all fresh
material generation before input/ready/online. `leave_online` must reject
unconsumed or mismatched materials. Counters include only additional
bridge sockets: SCI-side Beaver uses the already instrumented SCI
channels. `offline_generation_epoch` must increase for every material
generation operation. Do not use thread-local current-layer state: SCI
launches a joined worker thread even with one protocol thread.

Build extensions with `--adapter /path/backend_adapter.h`, repeated
`--extra-source`, repeated `--extra-library`, and
`--override SCI/relative.cpp=/path/private-patched.cpp`. SCI source
overrides are checked against the compiled source list. The original
SCI source tree is never changed. Graph-level hooks can preserve native
activation internals while logging those calls as remaining SCI work.

## Nonzero-state diagnostics and tests

```sh
python3 tools/segmented_dcf_runtime/prepare_hidden.py \
  --output /tmp/nonzero-hidden.i16 --batch 128
python3 tools/segmented_dcf_runtime/run.py \
  --binary /tmp/sdcf-native-build/sdcf-runtime \
  --fixture /tmp/ticket07-fixture-v2 --output /tmp/sdcf-nonzero \
  --steps 1 --queries 1 --diagnostic --hidden /tmp/nonzero-hidden.i16
SDCF_TEST_BINARY=/tmp/sdcf-native-build/sdcf-runtime \
  python3 -m unittest discover -s tools/segmented_dcf_runtime -p 'test_*.py'
```

The public tests cover independent batch grouping, recurrence dependency
preservation, phase aggregation and fail-closed backend availability.
Compiled runtime and live two-party logs remain the execution evidence.
# Native Activation Workers

`--activation-workers 1|2|3|4` selects resident, independent SCI Math/IO/OT
contexts for Sigmoid and TanH. The default is 1. The graph keeps one SCI protocol
thread; activation workers use contiguous chunks with a minimum target of 128
elements. Slot 0 borrows the original native context and executes on the caller.
Additional contexts and worker startup are included in setup. All worker traffic
is included in the existing atomic SCI phase counters.

Each non-primary slot owns ports at `base + 300 + 150 * (slot - 1)`, plus SCI's
50 and 100 offsets. The runner reserves and forwards those ports on shaped links.
An initial public handshake rejects differing context/worker counts.

The build uses private SCI source overlays. Both directions of LinearOT's
explicit OpenMP regions execute in a two-iteration loop: a single activation
worker retains two OT participants, while multiple activation workers use one
OT participant each. Eigen internal parallelism is disabled. The maximum
activation and cGGM configurations are four participants, including the caller.
No SCI installation files are modified.

Use `ACTIVATION_DETAIL` for activation wall time, total worker work time and
actual chunk participants, and `ACTIVATION_WORKER` for per-slot ranges, native
conversion calls and sent bytes. These replace the old native activation totals
printed by `finalize`; those legacy counters do not cover the replacement
dispatch. The run result preserves the worker records in `parties.json` and
validates complete, nonoverlapping coverage for every query/layer identity.

`TaskScope`, `context(slot)` and `run_worker_group` expose bounded ownership to
the subsequent graph scheduler without changing SCI's global protocol pointers.
Concurrent users must own disjoint slot ranges. Native activation conversion
dispatch uses a thread-local scope, so it cannot consume graph FSS materials.

`--graph-mode sequential|isolated|parallel` selects the single-step graph
schedule. Isolated and parallel schedules allocate two branches, each with two
owned native protocol contexts and at most two activation or cGGM participants.
Parallel execution joins the projection chains before the shared linear result
and joins the activation chains before subsequent elementwise work.

`--graph-openings independent|batched` evaluates homogeneous opening batching
separately from the channel schedule. The batched candidate concatenates the
ready gate and candidate biases in MatAdd rows 5 and 7, performs one signed
16-bit shift-3 conversion, and maps the two output ranges back to their original
layers. Projection extension contracts differ and remain separate. Both modes
retain all 15 tensor and task identities; `GRAPH_OPENING_GROUP` records the
two-layer mapping and conversion consumption belongs to layer 5.

`--prefetch-bytes` is a total party budget. Graph conversion contexts divide it
equally, assigning any remainder to lower context indices. Their explicit
allocation records and resident-prefix records are retained in `parties.json`.

The private NetIO overlay also records received bytes per protocol connection.
`BEAVER_OFFLINE` reports each matrix's owned sent/received traffic and cross-term
batch count. The collector checks these bytes against the opposite party's
matching matrix. Builds without that receive counter report `null`; historical
unmarked receive fields are retained in raw logs and normalized to unavailable
in the collected records. Phase communication always uses the measured channel
totals and counts each party's sends once.

`verify_native_workers.py` runs the existing single-step diagnostic for all four
backends with activation 1 and 4, using default/nonzero hidden states and two
fresh queries per row. It compares every reconstructed tensor to the frozen
native SCI reference. Diagnostic phases are not performance evidence.

## Complete folded activation schedules

`SCI_Beaver_segmented_PaperLUT --folded-activations both` replaces both native
SCI activation bodies with the complete folded Q12 Sigmoid/TanH protocol.
`--lut-key-family full_q_arithmetic|segmented_boolean_tensor_b2a` selects the
unsegmented 10-bit or segmented `(5,5)` 1024-entry LUT while retaining the same
online kernel. The explicit schedules are graph sequential/activation workers
1, graph parallel/activation workers 1, and graph parallel/activation workers
2. The latter uses four disjoint activation owners. Each owner holds a thread
local cGGM cap of one and performs its openings on its own thread. Cap-two
owners may use one sender helper while their caller receives. The local cGGM
executor remains a serialized shared pool, so input-independent generation is
reported as serial slot-major preparation.

`ACTIVATION_WORKER` includes actual observed hashing and transport participant
peaks; `ACTIVATION_DETAIL.wall_seconds` encloses the full worker join.
`FOLDED_PREPARE_GROUP.wall_seconds` directly encloses preparation of all folded
calls. Per-kind offline call/stage timers are work attribution, since this
shared envelope also includes work of the other activation. No isolated-kind
offline elapsed time is inferred from these sums.

## Offline optimization options

The PaperLUT runner defaults to `--lut-key-family full_q_arithmetic`. This
constructs the complete 10-bit arithmetic selector directly, replacing the
`segmented_boolean_tensor_b2a` construction's `(5,5)` Boolean tensor and leaf
B2A conversion. Both families retain the same resident selector layout and
online lookup kernel. To reproduce the segmented construction, select
`--lut-key-family segmented_boolean_tensor_b2a` explicitly.

The additional options are independent and are recorded in the run config
and both parties' environments:

| CLI option | Environment | Default and behavior |
| --- | --- | --- |
| `--fuse-sign-wrap` | `SEGFSS_FUSE_SIGN_WRAP=1` | Off. For supported signed conversions, derive the wrap correction from the existing lower-bit sign comparison instead of generating a separate full-width wrap DCF. The online correction adds a Boolean AND dependency, so compare both offline cost and online rounds. |
| `--bit-block-mode direct_cot` | `SEGFSS_BIT_BLOCK_MODE=direct_cot` | `legacy`. Select direct correlated OT for the secret bit-times-block products used in cGGM key generation. `direct_ot` remains available as the chosen-message OT control. |
| `--native-ring-cot` | `SEGFSS_NATIVE_RING_COT=1` | Off. Use native arithmetic correlated OT for dFSS ring correlations, including mask B2A and arithmetic LUT payload adaptation. |
| `--cache-lut-leaves` | `SEGFSS_CACHE_LUT_LEAVES=1` | Off. In full arithmetic LUT construction, retain each current tile's expanded leaves until payload correction has finished, then reuse them to materialize the selector. This removes duplicate local tree expansion; it does not itself reduce protocol messages. |
| `--folded-reuse-input-sign` | `SEGFSS_FOLDED_REUSE_SIGN=1` | Off. Pass the signed16 input extension's secret XOR sign share to the folded core, omitting duplicate sign comparison and its preprocessing inventory. |
| `--folded-clip-bits 22` | `SEGFSS_FOLDED_CLIP_BITS=22` | 32. Narrow only the native signed16/Q7 clip predicate; BitXA, magnitude, LUT and output arithmetic stay in ring32. |
| `--lut-skip-zero` | `SEGFSS_LUT_SKIP_ZERO=1` | Off. Scan public nonzero coefficient intervals with the original 1024-entry cyclic alignment. The current f12/q10 table has 533 nonzero entries. |
| `--lut-simd` | `SEGFSS_LUT_SIMD=1` | Off. Dispatch to AVX2 low-32-bit multiply/add when supported, with a scalar fallback and exact modular output. |

The leaf cache is released before the next tile. Its actual capacity is
included in working-memory accounting. With `--generation-budget-bytes`,
the cache plan reserves all final selectors, preprocessing scratch and
`--generation-reserve-bytes`, then reduces the requested LUT tile when needed.
A budget that cannot fit the required preprocessing is rejected before IO.
`LUT_LEAF_CACHE` records the selected tile, cache bytes, local expansion/hash
counts and budget estimates. `--generation-budget-bytes 0` disables the
budget-driven reduction; cache storage still stays within one LUT tile.

For example, enable all options on a runtime built with the PaperLUT bridge:

```sh
python3 tools/segmented_dcf_runtime/run.py \
  --binary /tmp/sdcf-paper-build/sdcf-runtime \
  --fixture /tmp/ticket07-fixture-v2 --output /tmp/sdcf-offline-options \
  --backend SCI_Beaver_segmented_PaperLUT --batch 128 --steps 1 \
  --folded-activations both --lut-key-family full_q_arithmetic \
  --fuse-sign-wrap --bit-block-mode direct_cot --native-ring-cot \
  --cache-lut-leaves --memory-gib 1.5 \
  --generation-budget-bytes 1610612736 \
  --generation-reserve-bytes 536870912 \
  --queries 3 --warmup 0 --round-trace --network-projection both
```

Use fresh output directories and match public shapes, resource limits and
tile settings when comparing individual options. Every query still generates
fresh masks, keys and triples before private input arrives, and consumes each
material exactly once. The cache reuses a tree only within that material's
offline construction; it never reuses material across queries.

The online options are frozen during query preparation and preserve the
existing native integer mapping. They support both LUT families and both sign
backends. Reused sign shares remain secret and are bound to the same owning
channel, query and generation epoch. The 22-bit predicate is valid for the
native signed16/Q7 contract; generic activation inputs retain their original
comparison width. With BitDec, sign reuse removes 31 carry layers and clipping
uses 21 rather than 31; the full graph depth comes from the measured trace.

`METRIC.process_cpu_seconds` records cumulative process user+system CPU across
all threads using `CLOCK_PROCESS_CPUTIME_ID`. Phase aggregation reports its
maximum across parties and its two-party sum separately from wall time. CPU
time includes instrumentation and system calls and can exceed wall time when
threads compute concurrently. LAN/WAN projections use communication bytes and
causal rounds and are reported separately from these localhost measurements.

Passive round traces have nested actor-owned semantic scopes with immutable
`label, query, layer, slot, call, kind, stage` fields and an occurrence index
for repeated stage tiles. Forked actors inherit their owner's scope stack.
`rounds.json.scopes` matches opposite-party scope identities, checks paired
bytes, and counts selected-scope send flights on the complete phase dependency
DAG. External causal edges and native fork/join dependencies are retained.
The original whole-phase rounds remain separately qualified; scoped rounds
or worker timers are never summed to describe a parallel activation or step.
