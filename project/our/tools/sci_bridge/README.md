# SCI private MSB bridge

The ABI preserves `AuxProtocols::MSB` semantics: additive ring shares in, XOR
shares of the reconstructed sign bit out. All calls use two parties without a
trusted dealer. The shared library hides its `party`, `peer` and OT symbols to
coexist with SCI in the same process. SCI roles 1/2 map to dFSS roles 2/3.

Build (from repository root):

```sh
cmake -S project/our -B /tmp/sci-msb-build \
  -DDEALERLESS_FSS_ROOT=/home/jiang/dealerless-lunwen/project/dealerless-FSS_public \
  -DEZPC_ROOT=/home/jiang/EzPC -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/sci-msb-build --target SCI_SEGFSS_MSB_TEST -j 3
python3 project/our/tools/sci_bridge/run_tests.py \
  --binary /tmp/sci-msb-build/SCI_SEGFSS_MSB_TEST \
  --output /tmp/sci-msb-check --port 35400
```

Each party uses persistent sockets and base OT state; every MSB call generates
fresh mask bits, keys and scan triples. A batch of 100 is a single vectorized
protocol call. Local GGM hashing runs on one worker. Directional OT exchanges
retain the underlying library's communication workers. Do not call a bridge
context concurrently; this version supports one context per process.

For an **isolated copy** of SCI's `BuildingBlocks/aux-protocols.cpp`, include
`sci_hook.h` and insert this at the beginning of `AuxProtocols::MSB`:

```cpp
if (sci_segmented::msb(party, x, msb_x, size, bw_x)) return;
```

Compile with include paths `project/our/src`, `project/our/tools/sci_bridge`;
link `libsegfss_sci_bridge.so` and give its directory as runtime rpath. Set the
following identically at both parties:

```sh
export SEGFSS_SCI_ENABLE=1 SEGFSS_SCI_WIDTH=16 SEGFSS_SCI_NSEG=4
export SEGFSS_SCI_PORT=35420 SEGFSS_SCI_HOST=127.0.0.1
```

The bridge port family must not overlap SCI's port family. Optional
`SEGFSS_SCI_TRACE` selects a **different file per party**; it contains only public
width/count/segment metadata. Use `sci_segmented::stats()` for phase totals and
`finish()` for explicit shutdown. Width is the MSB input ring: width=16 uses
15 comparison bits; nseg splits those 15 bits. The hook leaves other widths on
SCI and preserves all activations, rescaling and output reconstruction.

Statistics include fresh material construction in offline totals, masked
opening/evaluation/scan in online totals, and the extra context/base OT setup
separately. These are components for the complete executor's accounting;
`rounds` is the inherited Peer/NetIO communication-counter sum, **not a measured
critical-path round count**, and must not be relabelled as one. Online dependent
protocol rounds are one masked opening plus `ceil(log2(nseg))` scan levels.

The standalone correctness executable reconstructs outputs only in its named
oracle tests; its final private run uses locally generated cryptographic input
shares and opens neither input nor output. This primitive result is not a
completed Google-30 executor or an end-to-end speed claim.
