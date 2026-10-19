# Native arithmetic COT preprocessing

Ticket 03 supplies a private SCI overlay for ordinary `send_cot` / `recv_cot`
and mixed-width `send_batched_cot` / `recv_batched_cot`. Shared SCI installation
files are not changed. Default mode retains the original algorithms.

## Public Preparation Interface

Every SplitIKNP context owns `arithmetic_stock`. During the shared lifecycle's
`Preparing` phase, register the context and plan its full native call order in
the common ledger. Then call:

```cpp
ot->prepare_arithmetic_cot({{count, widths, messages_per_ot, batched}});
```

The public call descriptor is `sdcf_arithmetic::Call`. Ordinary COT has a single
width, one message per OT, and `batched=false`; its ledger protocol is
`iknp_cot`. Batched COT uses `batched=true`, the **complete ordered** width
vector, and SCI's messages-per-OT value; its protocol is `iknp_batched_cot`.
The ledger's message width is the maximum width. Widths lie in 1..64. Count
must be positive and divisible by the number of width groups, matching SCI's
equal-size grouping contract.

One preparation invocation may contain one or many descriptors. Repeated
invocations in the same query append fresh calls in order, allowing replay
interleaved with other material families on the same context. Preparation in a
new query rejects unconsumed previous-query stock and then starts fresh stock.
Actual online calls check count, full width vector, lane count, ordinary versus
batched mode, query identity, and shared lifecycle before consuming once.
No online extension or refill is permitted in strict mode.

Each call physically runs native IKNP extension against independently random
receiver choices. CRH hashes and every lane's final random pad are completed
before the material-ready barrier. Both final pads are retained by the sender;
the selected pad and random choice are retained by the receiver. Public SIMD
extension padding is generated and charged offline, then discarded separately
from effective OT consumption.

## Arithmetic Relation And Online Cost

For each OT/lane in its native `Z/(2^width)` ring, let sender pads be `p0,p1`
and receiver's offline random choice be `r0`. After the true choice `r` arrives,
the receiver sends `d=r XOR r0`. The sender sets `s=p_d` and sends
`c=correlation+s-p_(1-d)`. The receiver returns its random pad plus `c` exactly
when `r=1`. Therefore the outputs satisfy
`receiver = sender + r * correlation (mod 2^width)`.

Choices and correlations never enter preparation. The receiver's correction
uses `ceil(count/8)` bytes. The sender transmits a single packed arithmetic
correction per lane, occupying
`ceil(sum_group(group_count * width * messages_per_ot)/8)` bytes. Both are
charged online through the actual native NetIO counters. Batched lanes share
the choice correction; different width groups preserve their native rings.

Native COT calls during `Preparing`, such as Beaver's own offline material
generation, retain SCI's original path. They do not consume query-online COT
stock. Their actual extension and communication remain charged to offline.

## Live Verification

`python3 -m unittest test_prepared_cot -v` uses real two-party SCI B2A, signed
LinearOT matrix multiplication, and Hadamard multiplication (the MulCir
primitive), with original callers recompiled against the overlay. Each party
has at most four affinity CPUs and 1.5 GiB `RLIMIT_AS`. Heavy compilation takes
the shared `/tmp/sci-preprocessed-build.lock`.

The tests cover two fresh queries, both directions, 1/13/17/33/64-bit rings,
mixed groups, count-257 SIMD tails, non-byte aligned payloads, five-lane
messages spanning multiple hashes, append preparation, exact contract
rejection, exhausted/unconsumed stock, unfinished random material and blocked
online extension. Reconstructed native integers agree with worked examples
and default SCI. Diagnostic openings are in matching measured phases for both
modes. For nine 13-bit B2A conversions, the measured online sends are 16 bytes
for party 1 and 3 bytes for party 2, including the one-byte phase sync.

These are primitive diagnostics, not a completed FastGRNN preprocessing
baseline. Comparison and graph-level conversion preparation belong to later
tickets. Set `SDCF_COT_EVIDENCE_DIR` to preserve party logs and exit summaries.
