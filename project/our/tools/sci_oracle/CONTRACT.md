# SCI Numeric Contract

Frozen reference: EzPC `f24bf3e022dde3493cf0ffe3ebd92dd73936b734`, SCI
`src/defines.h` with `DIV_RESCALING` defined. The relevant files are unmodified
relative to that commit. The cross-check compiles the frozen `truncation.cpp`
and `value-extension.cpp` directly, linking the existing SCI static libraries;
its complete compiler command is saved in `results/.../01/build_command.json`.
`DIV_RESCALING` selects the rescaling calls in `library_fixed.cpp`; it does not
change the meaning of `Truncation::truncate_and_reduce` in LinearOT or Math.

## Public Oracle

Include `sci_numeric_oracle.h`; call
`segfss::reference::evaluate(residue, Contract{input_bits, output_bits, shift,
call, signed_arithmetic, numeric_mode})`. All values are reconstructed integers
modulo `2^input_bits`. This is a plaintext correctness oracle, never a function
to apply separately to secret shares.

`NumericMode::sci_compatible` is the default. Call site and mode are separate:

| Call | Positive shift | Output |
| --- | --- | --- |
| `SciCall::div_pow2` | Signed division toward zero; unsigned floor | SCI preserves input ring, followed by explicit signed/zero extension or local reduction to `output_bits` |
| `SciCall::truncate` | Signed floor; unsigned floor | Same explicit output conversion |
| `SciCall::truncate_and_reduce` | Floor, modulo shrinking ring | Exactly `input_bits-shift` bits |
| `SciCall::convert` | Rejected | Shift zero converts ring; negative shift multiplies by `2^(-shift)` before destination-ring reduction |

Positive truncation requires `0 < shift < input_bits`; zero is identity before
output conversion. Ring widths are 1..64. Left shifts are -63..0 and never
construct DCF material. Invalid contracts throw `std::invalid_argument` before
evaluation. Arbitrary output widths are an adapter composition, not an extra
argument of the native SCI truncation API. A negative input with nonzero low
bits adds one after floor only at `div_pow2` calls. There is no saturation.

`NumericMode::exact_floor_diagnostic` always floors into the shrinking ring,
requires `output_bits=input_bits-shift`, and remains an independent mode.
For example, input `253` in 8 bits is -3: SCI division by 2 is `255` in 8 bits;
floor diagnostic is `126` in 7 bits. Diagnostic results must not enter an
SCI-compatible performance row.

## Share Corrections

For unsigned shares `a,b` in a k-bit ring, let `L=2^s`,
`w=floor((a+b)/2^k)`, and `carry=floor(((a mod L)+(b mod L))/L)`.
Reconstructed unsigned floor is
`(a>>s)+(b>>s)+carry-w*2^(k-s)`. Reducing into `k-s` bits cancels the full-ring
wrap term; keeping or extending the ring does not. Signed floor additionally
subtracts `sign(x)*2^(k-s)`. Toward-zero adds
`sign(x) AND ((x mod L)!=0)`. SCI implements equivalent corrections through
sign-bit mapping, wrap comparison, equality and B2A. Sign extension of a
k-bit shared residue into m bits requires correcting the additive wrap and
subtracting `sign(x)*2^k`; merely widening each share is incorrect.

The masked-input equivalent used by truncation adapters is
`floor(x_unsigned/L) = floor(z/L)-floor(r/L)-[z_low<r_low]
+[z<r]*2^(k-s)` for `z=(x+r) mod 2^k`. Its full-ring wrap term disappears only
in the shrinking output ring. Signed correction and toward-zero correction
still follow the call-site contract.

## Frozen Call Sites

All graph tensors use two's-complement rings and `real=signed(integer)*2^-frac`.
The independent mixed-width graph and historical narrow graph differ in W2,
U2 and Bg; they must never share a performance comparison row accidentally.

| Call site | Input bits / frac | Product / accumulator bits | Ordered rescaling | Output bits / frac |
| --- | --- | --- | --- | --- |
| MatMul W1, K=32 | X 16/11, W1 8/5 | 24 / 29 | `truncate_and_reduce(5)` then `div_pow2(1)` then reduce | 16/10 |
| MatMul W2, K=16, independent | 16/10, W2 16/12 | 32 / 36 | floor-reduce 4, then div 11, then reduce | 16/7 |
| MatMul W2, historical | 16/10, W2 8/4 | 24 / 28 | floor-reduce 4, then div 3, then reduce | 16/7 |
| MatMul U1, K=100 | H 16/16, U1 8/3 | 24 / 31 | floor-reduce 7, then div 7, then reduce | 8/5 |
| MatMul U2, K=35, independent | 8/5, U2 16/11 | 24 / 30 | floor-reduce 6, then shift zero, then reduce | 16/10 |
| MatMul U2, historical | 8/5, U2 8/3 | 16 / 22 | floor-reduce 6, then left shift 6 | 16/8 |
| MatMul FC, K=100 | H 16/16, FC 8/2 | 24 / 31 | floor-reduce 7, then div 9, then reduce | 8/2 |
| MulCir gate * H | 16/14, 16/16 | 32 / 32 | div 14, then reduce | 16/16 |
| ScalarMul zeta * (1-gate) | public 16/17, 16/14 | 32 / 32 | sign-extend B to 32, local product, div 14, reduce | 16/17 |
| MulCir adjusted gate * candidate | 16/17, 16/7 | 32 / 32 | div 7, then reduce | 16/17 |
| MatAdd input paths | destination scale aligned per operand | temporary ring 16, classifier 8 | sign-extend each operand; div per operand; add/subtract; div demote; reduce | graph destination |
| Independent recurrent sum | W2 16/7, U2 16/10 | none / 16 | div U2 by 3 before add | 16/7 |
| Historical recurrent sum | W2 16/7, U2 16/8 | none / 16 | div U2 by 1 before add | 16/7 |
| Independent gate bias | 16/7, Bg 16/10 | none / 16 | div Bg by 3 before add | 16/7 |
| Historical gate bias | 16/7, Bg 8/2 | none / 16 | div input by 5; add; left shift 5 | 16/7 |
| Candidate bias | 16/7, Bh 16/10 | none / 16 | div Bh by 3 before add | 16/7 |
| Public nu broadcast | 16/17, public nu 16/23 | none / 16 | div nu by 6 before add | 16/17 |
| Hidden-state sum | 16/16, 16/17 | none / 16 | div second operand by 1 before add | 16/16 |
| Classifier bias | 8/2, FCbias 8/5 | none / 8 | div bias by 3 before add | 8/2 |
| ArgMax input | 8/2 | none | `XTProtocol::s_extend(8,16)` | 16/2 |
| Sigmoid, TanH | signed 16/7 | SCI Math internal variable rings | Preserve `Math/math-functions.cpp` operations; inner `truncate_and_reduce` stays floor | 16/14, 16/7 |

MatMul's first stage comes from `LinearOT/linear-ot.cpp` matrix multiplication
with `accumulate=true`: `extra_bits=ceil(log2(K))`. The second comes from
`library_fixed.cpp::MatMul_thread`, with `shift=shrA+shrB+demote+H1-depth` after
public scale factors have been converted into exponents. `bwTemp` declared by
the generated call does not replace the actual `bwA+bwB+depth` accumulator.
`MulCir_thread`, `ScalarMul`, `MatAdd`, `AddOrSubCir`, and `AdjustScaleShr` use
the explicit DIV_RESCALING call chain. ScalarMul's A is public in this graph.

## Boundary and Evidence

Native signed `div_pow2` at `shift=input_bits-1` requests a zero-bit
`compare_with_eq`. At this revision its one-entry message allocation is sent
through two-message IKNP, so direct execution is not a valid reference there.
`nativeDirectSupported` identifies this case. For widths below 64 the test
sign-extends through SCI into width+1, invokes SCI division, then reduces;
the oracle agrees. The 64/63 case has literal boundary tests but is explicitly
`native_direct_unsupported`; no direct SCI match is claimed.

Run:

```sh
g++ -std=c++17 -fsanitize=undefined -Iproject/our/src project/our/test/correctness/test_sci_numeric_oracle.cpp -o /tmp/sdcf-oracle-unit
/tmp/sdcf-oracle-unit
python3 project/our/tools/sci_oracle/run_crosscheck.py --output project/our/results/segmented_dpf_dcf_truncation/01
```

The SCI run exhausts residues and valid shifts for 1..8 bits, checks graph and
scan widths through 64 bits with fixed random and boundary vectors, and uses
three additive share decompositions per input to exercise wrap correction.
It also checks the seven actual MatMul floor/division/left-shift chains.
Reconstruction happens only in this diagnostic harness, on party 2. Raw party
logs, sampled cross-checked vectors, command, exit codes and coverage are saved
alongside the report. This is primitive numeric evidence, not an end-to-end
model-quality or performance result.
