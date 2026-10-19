// Explicit ell-bit ring arithmetic, or checked wide arithmetic for legacy control.
#pragma once
#include <cstdint>
#if defined(__CUDACC__)
#define UNIFORM_HD __host__ __device__ __forceinline__
#else
#define UNIFORM_HD inline
#endif
namespace uniform_fixed {
UNIFORM_HD int signed_bits(__int128 value) {
  unsigned __int128 magnitude=value<0?static_cast<unsigned __int128>(-(value+1)):static_cast<unsigned __int128>(value);
  const uint64_t high=static_cast<uint64_t>(magnitude>>64),low=static_cast<uint64_t>(magnitude);
#ifdef __CUDA_ARCH__
  return high?129-__clzll(high):(low?65-__clzll(low):1);
#else
  return high?129-__builtin_clzll(high):(low?65-__builtin_clzll(low):1);
#endif
}
UNIFORM_HD __int128 floor_div(__int128 value,int64_t divisor) {
  return value>=0?value/divisor:-((-(value+1))/divisor)-1;
}
UNIFORM_HD __int128 floor_shift(__int128 value,int s) {
  if(!s)return value;
  return value>=0?value>>s:-(((-(value+1))>>s)+1);
}
UNIFORM_HD bool outside_ring(__int128 value,int ell) {
  const __int128 half=static_cast<__int128>(1)<<(ell-1);
  return value < -half || value >= half;
}
// Defined unsigned masking; negative operands cannot cause signed C++ overflow.
UNIFORM_HD int64_t wrap_signed(__int128 value,int ell) {
  const uint64_t modulus=uint64_t{1}<<ell;
  const uint64_t residue=static_cast<uint64_t>(static_cast<unsigned __int128>(value))&(modulus-1);
  return residue >= (modulus>>1)?static_cast<int64_t>(residue)-static_cast<int64_t>(modulus):static_cast<int64_t>(residue);
}
template<typename Accumulator> struct PrefixTrace {
  Accumulator low=0,high=0;
  UNIFORM_HD void observe(Accumulator value) { low=value<low?value:low;high=value>high?value:high; }
};
struct Trace {
  __int128 low=0,high=0;
  bool wrapped=false;
  UNIFORM_HD void observe(__int128 value) { low=value<low?value:low;high=value>high?value:high; }
  UNIFORM_HD int bits()const { const int a=signed_bits(low),b=signed_bits(high);return a>b?a:b; }
};
UNIFORM_HD int64_t reduce_ring(__int128 value,int ell,Trace& trace) {
  trace.wrapped=trace.wrapped||outside_ring(value,ell);
  const int64_t reduced=wrap_signed(value,ell);
  // Physical carrier width is excluded: the logical register only sees residue.
  trace.observe(reduced);
  return reduced;
}
UNIFORM_HD __int128 mac(__int128 acc,__int128 product,int ell,bool ring,Trace& trace) {
  if(ring) {
    const int64_t term=reduce_ring(product,ell,trace);
    return reduce_ring(acc+term,ell,trace);
  }
  trace.observe(product);acc+=product;trace.observe(acc);return acc;
}
UNIFORM_HD int64_t writeback(__int128 result,int ell,bool wrap,bool& affected) {
  affected=affected||outside_ring(result,ell);
  if(wrap)return wrap_signed(result,ell);
  const int64_t qmin=-(int64_t{1}<<(ell-1)),qmax=(int64_t{1}<<(ell-1))-1;
  return result<qmin?qmin:(result>qmax?qmax:static_cast<int64_t>(result));
}
// stats=[affected output count,logical register overflow output count,
//        max logical register bits,max pre-writeback output bits].
// In legacy wide+saturate these retain the historical four definitions.
UNIFORM_HD int64_t finish(__int128 unscaled,int s,int64_t bias_q,int ell,int register_bits,
    Trace& trace,int64_t& clips,int64_t& overflows,int& register_peak,int& storage_peak,
    bool ring=true,bool wrap=true) {
  const __int128 lifted=static_cast<__int128>(bias_q)*(int64_t{1}<<s);
  __int128 biased;
  if(ring) {
    const int64_t bias=reduce_ring(lifted,ell,trace);
    biased=reduce_ring(unscaled+bias,ell,trace);
  } else { trace.observe(lifted);biased=unscaled+lifted;trace.observe(biased); }
  // The signed ell-bit residue, never discarded high product bits, is shifted.
  const __int128 result=floor_shift(biased,s);
  const int bits=trace.bits(),stored_bits=signed_bits(result);
  overflows+=bits>register_bits;
  register_peak=bits>register_peak?bits:register_peak;
  storage_peak=stored_bits>storage_peak?stored_bits:storage_peak;
  bool affected=ring&&trace.wrapped;
  const int64_t encoded=writeback(result,ell,wrap,affected);clips+=affected;return encoded;
}
UNIFORM_HD int64_t finish_average(__int128 sum,int64_t count,int ell,int register_bits,
    Trace& trace,int64_t& clips,int64_t& overflows,int& register_peak,int& storage_peak,
    bool ring=true,bool wrap=true) {
  const __int128 result=floor_div(sum,count);
  const int bits=trace.bits(),stored_bits=signed_bits(result);
  overflows+=bits>register_bits;
  register_peak=bits>register_peak?bits:register_peak;
  storage_peak=stored_bits>storage_peak?stored_bits:storage_peak;
  bool affected=ring&&trace.wrapped;
  const int64_t encoded=writeback(result,ell,wrap,affected);clips+=affected;return encoded;
}
}
