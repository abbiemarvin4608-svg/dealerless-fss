// Scalar semantics of the physical-register ring emulator (E0-R).
//
// A linear-layer register is a uint64 masked to exactly L bits after every
// product and every addition, so nothing wider than L bits can reach an
// output.  The int128 "exact" accumulator kept next to it is a diagnostic
// side channel: it only feeds the overflow / bit-length statistics.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define RING_HD __host__ __device__ __forceinline__
#else
#define RING_HD inline
#endif

namespace ring {

RING_HD uint64_t mask_bits(int bits) {
  return bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
}

// Two's-complement value of the low `bits` bits of x.
RING_HD int64_t to_signed(uint64_t x, int bits) {
  const uint64_t sign = uint64_t{1} << (bits - 1);
  return static_cast<int64_t>(((x & mask_bits(bits)) ^ sign) - sign);
}

// floor(v / 2^s) without relying on the sign behaviour of >>.
RING_HD int64_t floor_shift(int64_t v, int s) {
  if (s == 0) return v;
  return v >= 0 ? v >> s : -(((-(v + 1)) >> s) + 1);
}

// Length of the shortest two's-complement encoding of v.
RING_HD int signed_bits(__int128 v) {
  unsigned __int128 m = v < 0 ? static_cast<unsigned __int128>(-(v + 1))
                              : static_cast<unsigned __int128>(v);
  int bits = 1;
  while (m) {
    ++bits;
    m >>= 1;
  }
  return bits;
}

// One multiply-accumulate in the L-bit register.  Casting a signed code to
// uint64 and masking is its sign extension into Z_{2^L} (valid for ell <= L).
RING_HD uint64_t mac(uint64_t acc, int64_t x, int64_t w, uint64_t mask) {
  return (acc + ((static_cast<uint64_t>(x) * static_cast<uint64_t>(w)) & mask)) & mask;
}

// Truncate-reduce: y = floor(signed_L(acc) / 2^s) mod 2^ell as a signed
// representative.  `exact` is used only for the statistics.
RING_HD int64_t finish(uint64_t acc, __int128 exact, int s, int ell, int L,
                       int64_t& overflow, int64_t& wraps, int& max_bits) {
  const int64_t q = floor_shift(to_signed(acc, L), s);
  const int64_t y = to_signed(static_cast<uint64_t>(q), ell);
  wraps += (y != q);
  const int bits = signed_bits(exact);
  overflow += (bits > L);
  max_bits = bits > max_bits ? bits : max_bits;
  return y;
}

}  // namespace ring
