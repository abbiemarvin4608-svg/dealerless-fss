# Native Comparison and Complete Conversion Evidence

Command, from `project/our/tools/segmented_dcf_runtime`:

```sh
SDCF_TRIPLE_EVIDENCE_DIR=/home/jiang/dealerless-lunwen-impl04/project/our/results/sci_sirnn_preprocessed_20261012/ticket04_conversion python3 -m unittest test_prepared_triples.py
```

Result: 3 test methods passed in 12.404 seconds. Compilation uses
`/tmp/sci-preprocessed-build.lock`; each real party has four allowed CPUs and
1536 MiB `RLIMIT_AS`. All success modes run two fresh, one-use queries.

After merging the latest unified build and the nested offline ordinary-GOT
fix, `python3 -m unittest test_native_planning.py test_prepared_triples.py`
passed all four test methods in 18.115 seconds with the same evidence-directory
environment variable. The retained logs are from this final run. An inactive
registered triple context is also planned on each query and emits no empty
inventory records.

The conversion suite checks 20 real native protocol calls with 21 observed
output vectors per query (comparison-with-equality yields two vectors):
greater/less comparison, comparison-with-equality, equality, wrap, MSB, B2A,
signed/unsigned truncation, signed/unsigned division by a power of two,
truncate-and-reduce, truncate-reduce-extend, truncation with supplied MSB, and
signed/zero extension to 27- and 64-bit rings. Signed truncation and division
are also exercised in the 64-bit ring. Input vectors have 13 elements, so
comparison padding to multiples of eight is exercised.

Default and prepared SCI match exactly on 42 output vectors / 546 integers.
The independent native integer checks include zero, positive/negative values,
negative remainders, sharing wrap/low-bit borrow, signed 13-bit boundaries,
`INT64_MIN`, and `INT64_MAX`. Native `truncate` rounds toward negative infinity;
native `div_pow2` rounds toward zero. Original protocol implementations and
their output rings are retained.

For each conversion-suite query, SCI safely generates and consumes 1840
bit-triples: 21 standard `_16KKOT_to_4OT` calls and seven correlated `_8KKOT`
calls. Both parties reach the material-ready barrier before input. Every
input/online/output metric reports a zero native generation epoch delta, and
per-protocol `CONVERSION_MATERIAL_COVERAGE` records account for all consumed
OT/triple stock. The second comparison output reuses the same protocol result.

The direct material test reconstructs standard and correlated triples in
packed and unpacked layouts, including correlated offsets 3 and 8. Each query
generates/consumes 86 triples across four calls. Triple balances have no
remaining material or material discarded after generation; native comparison
padding is part of the public protocol count. Extension SIMD padding is
recorded separately by the OT inventory.

Both parties reject unsafe generation during planning and actual prepare,
wrong count contracts, repeated consumption, incomplete consumption, and
attempted online material generation. Each rejected mode retains its two
party logs and exit-code record. The shared collector independently accepts
the planned/generated/consumed/balance triple evidence for both successful
prepared modes and both parties.

Diagnostic openings are included in online communication for both modes.
These primitive checks establish correctness and coverage; single-step
batch-128 measurements are reported by the integration benchmark separately.
