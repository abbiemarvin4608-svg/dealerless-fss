# Native SCI Matrix Beaver Backend

This module implements the SCI-compatible Beaver backend for SDCF-07/08. It
replaces MatMul and MulCir with dealerless Beaver triples while retaining SCI
input sign extension, floor-and-reduce, final division/left shift and output
reduction. Google's public ScalarMul (`zeta/nu`) remains a local multiplication
and is not charged as an OT operation. The `prepare_scalarmul`/
`multiply_scalarmul` API provides a separate private-scalar Beaver path for
the SDCF-08 control and tests.

## Material and Execution

`prepare_matmul` samples independent private shares of U and V in the
accumulator ring L and obtains additive shares of both cross products using
SCI's real two-party `LinearOT::matmul_cross_terms`. Local U_i*V_i completes W.
This is a matrix triple, with no trusted dealer. The call uses L/L/L ring
widths and bypasses the native higher-level matrix API's implicit truncation.
Offline 32-row tiles cap temporary allocations while retaining one U/V/W
matrix triple. V is shared across the row tiles of that single triple.

`multiply_matmul` validates the ring/shape, consumes the material, opens the
whole concatenated E/F vector, and computes W+E*V+U*F+E*F (the last term on
one party). It returns the raw accumulator-ring result. Material is move-only;
shape mismatch is rejected before use and repeated consumption is rejected.
The adapter also consumes/poisons a key if a live conversion fails. Materials
are freed after the query when the next preparation begins, or at finish.

Two independent, opt-in runtime controls preserve the original kernel as a
matched reference. `--beaver-algebra` computes the server share as
`W + E*(V+F) + U*F`, reducing its local matrix products from three to two;
the client still computes `W + E*V + U*F`. `--beaver-mapped-layout` maps
row-major uint64 buffers directly into single-participant Eigen products,
including the local offline product. Both operate in the unchanged integer
ring and leave the E/F opening and fresh-material contract intact.

The core API accepts `MatmulOptions` during preparation and an optional
caller-owned `MatmulWorkspace` during multiplication. The output overload
reuses caller storage and permits output aliases with operands or material;
the workspace must be disjoint from operands and output. The serial runtime
reuses one workspace and output buffer across layers and queries.

Matrix records include `local_kernel_seconds`, `buffer_seconds`,
`kernel_compute_seconds`, and `kernel_layout_copy_seconds`. The private SCI
overlay measures its original Eigen input/output copies, including matrix
allocation, separately from multiplication. Calls linked to an uninstrumented
SCI archive report the unavailable subintervals as `null`. These nested
intervals are localization evidence, not additional query wall time. The mapped
path has no explicit layout copies and keeps Eigen internal parallelism off.

The original private weights remain ALICE-owned with BOB's share zero; ALICE
locally sign-extends them. Shared activations use a real SCI sign-extension
protocol online. For K terms the ring is `bwA+bwB+ceil(log2(K))`. After Beaver
accumulation, native SCI floor-reduces by `ceil(log2(K))` into `bwA+bwB`; the
original caller then performs the frozen final div_pow2 or left shift.

The runtime prepares fresh materials for every query before the input/ready
barrier and rejects generation during online through the common epoch check.
`BEAVER_OFFLINE` and `BEAVER_ONLINE` lines report actual shapes, rings, row
tiles, E/F sizes, consumption and native conversion coverage. One E/F opening
is one logical exchange; the native conversions' total logical rounds remain
unmeasured and are not inferred from socket calls.

`adapter.cpp` keeps the current layer in process-global state, because native
SCI invokes a joined worker thread even at one application thread. The runtime
currently uses one application thread per party. SCI's internal two-direction
OT region still uses two OpenMP workers, as in the native control.

## Build and Reproduce

```sh
python3 tools/sci_beaver/build.py \
  --out /tmp/sdcf-07-beaver-build \
  --generated /path/to/frozen/sirnn_fixed.cpp
python3 tools/sci_beaver/run_tests.py \
  --output results/segmented_dpf_dcf_truncation/07/primitives
python3 tools/segmented_dcf_runtime/run.py \
  --binary /tmp/sdcf-07-beaver-build/sdcf-runtime \
  --fixture /tmp/ticket07-fixture-v2 \
  --output results/segmented_dpf_dcf_truncation/07/beaver_secure \
  --backend SCI_Beaver --batch 128 --steps 1 --queries 3
```

`build.py` changes a checked build-tree copy of `library_fixed.cpp`, compiles
all SCI sources against the runtime's metered NetIO header, and includes the
new source/header identities in the runtime manifest. No external SCI source
or prebuilt archive is modified. The same binary accepts `--backend SCI` and
uses the original matrix implementation as the control.

For intermediate correctness, add `--diagnostic --queries 1 --hidden <file>`
to each backend run and compare party-2 logs with `compare_diagnostics.py`.
Diagnostics reveal tensors only for testing and are separate from secure
timing. Complete reproducible commands, coverage and result files are in
`results/segmented_dpf_dcf_truncation/07/`.

To attribute the local matrix changes, run otherwise identical secure rows
with neither flag, `--beaver-algebra`, `--beaver-mapped-layout`, and both flags.
The runner forwards their values explicitly after clearing inherited SDCF
configuration. The real two-party regression checks all four combinations
against independent integer reconstruction and native SCI conversion, including
non-square Batch=128 shapes and output aliases in 8/32/64-bit rings.

## Extension Points

The core `beaver.h` API is independent of the graph coordinator. Subsequent
backends can reuse it with native SCI packs. `adapter.h` implements the common
SDCF backend lifecycle, and its `matmul`/`mulcir` hooks own the mixed-width
conversion order. Any segmented conversion replacement must preserve that
order and use the same prepared query/layer context. Public zeta/nu are not
secret multiplication triples and remain distinct from MulCir and
private-scalar material.
