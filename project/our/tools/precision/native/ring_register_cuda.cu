// CUDA kernels for the physical-register ring emulator (E0-R).
//
// Deliberately free of torch headers: nvcc on torch/extension.h needs more
// memory than the development machine has.  Tensor checks and dispatch live
// in ring_register.cpp; these launchers take raw contiguous pointers.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "ring_register_cuda.h"
#include "ring_register_math.h"

namespace {

constexpr int kThreads = 256;

int blocks_for(int64_t elements) {
  return static_cast<int>(std::min<int64_t>((elements + kThreads - 1) / kThreads, 4096));
}

// Warp-reduce the per-thread statistics; one atomic per warp and counter.
__device__ void publish(int64_t* stats, int64_t overflow, int64_t wraps, int max_bits) {
  long long o = overflow, w = wraps;
  for (int offset = 16; offset > 0; offset >>= 1) {
    o += __shfl_down_sync(0xffffffffu, o, offset);
    w += __shfl_down_sync(0xffffffffu, w, offset);
    max_bits = max(max_bits, __shfl_down_sync(0xffffffffu, max_bits, offset));
  }
  if ((threadIdx.x & 31) == 0) {
    auto* counters = reinterpret_cast<unsigned long long*>(stats);
    if (o) atomicAdd(counters, static_cast<unsigned long long>(o));
    if (w) atomicAdd(counters + 1, static_cast<unsigned long long>(w));
    atomicMax(counters + 2, static_cast<unsigned long long>(max_bits));
  }
}

__global__ void conv2d_kernel(const int64_t* __restrict__ x, const int64_t* __restrict__ w,
                              const int64_t* __restrict__ b, int64_t* __restrict__ y,
                              int64_t* __restrict__ stats, int n, int cin, int hin, int win,
                              int cout, int kh, int kw, int hout, int wout, int stride,
                              int pad, int ell, int s, int L) {
  const uint64_t mask = ring::mask_bits(L);
  const int64_t total = static_cast<int64_t>(n) * cout * hout * wout;
  int64_t overflow = 0, wraps = 0;
  int max_bits = 0;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t t = index;
    const int ow = t % wout; t /= wout;
    const int oh = t % hout; t /= hout;
    const int oc = t % cout;
    const int batch = static_cast<int>(t / cout);
    uint64_t acc = 0;
    __int128 exact = 0;
    for (int ic = 0; ic < cin; ++ic) {
      for (int r = 0; r < kh; ++r) {
        const int ih = oh * stride - pad + r;
        if (ih < 0 || ih >= hin) continue;  // zero padding contributes 0
        const int64_t* xrow = x + ((static_cast<int64_t>(batch) * cin + ic) * hin + ih) * win;
        const int64_t* wrow = w + ((static_cast<int64_t>(oc) * cin + ic) * kh + r) * kw;
        for (int q = 0; q < kw; ++q) {
          const int iw = ow * stride - pad + q;
          if (iw < 0 || iw >= win) continue;
          acc = ring::mac(acc, xrow[iw], wrow[q], mask);
          exact += static_cast<__int128>(xrow[iw]) * wrow[q];
        }
      }
    }
    if (b != nullptr) {
      acc = (acc + static_cast<uint64_t>(b[oc])) & mask;
      exact += b[oc];
    }
    y[index] = ring::finish(acc, exact, s, ell, L, overflow, wraps, max_bits);
  }
  publish(stats, overflow, wraps, max_bits);
}

__global__ void linear_kernel(const int64_t* __restrict__ x, const int64_t* __restrict__ w,
                              const int64_t* __restrict__ b, int64_t* __restrict__ y,
                              int64_t* __restrict__ stats, int rows, int k, int m, int ell,
                              int s, int L) {
  const uint64_t mask = ring::mask_bits(L);
  const int64_t total = static_cast<int64_t>(rows) * m;
  int64_t overflow = 0, wraps = 0;
  int max_bits = 0;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t* xrow = x + (index / m) * k;
    const int64_t* wrow = w + (index % m) * k;
    uint64_t acc = 0;
    __int128 exact = 0;
    for (int t = 0; t < k; ++t) {
      acc = ring::mac(acc, xrow[t], wrow[t], mask);
      exact += static_cast<__int128>(xrow[t]) * wrow[t];
    }
    if (b != nullptr) {
      acc = (acc + static_cast<uint64_t>(b[index % m])) & mask;
      exact += b[index % m];
    }
    y[index] = ring::finish(acc, exact, s, ell, L, overflow, wraps, max_bits);
  }
  publish(stats, overflow, wraps, max_bits);
}

__global__ void channel_affine_kernel(const int64_t* __restrict__ x,
                                      const int64_t* __restrict__ mult,
                                      const int64_t* __restrict__ offset,
                                      int64_t* __restrict__ y, int64_t* __restrict__ stats,
                                      int64_t total, int64_t channels, int64_t inner,
                                      int ell, int s, int L) {
  const uint64_t mask = ring::mask_bits(L);
  int64_t overflow = 0, wraps = 0;
  int max_bits = 0;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = (index / inner) % channels;
    const uint64_t acc =
        (ring::mac(0, x[index], mult[c], mask) + static_cast<uint64_t>(offset[c])) & mask;
    const __int128 exact = static_cast<__int128>(x[index]) * mult[c] + offset[c];
    y[index] = ring::finish(acc, exact, s, ell, L, overflow, wraps, max_bits);
  }
  publish(stats, overflow, wraps, max_bits);
}

}  // namespace

namespace ring_cuda {

cudaError_t conv2d(const int64_t* x, const int64_t* w, const int64_t* b, int64_t* y,
                   int64_t* stats, int n, int cin, int hin, int win, int cout, int kh, int kw,
                   int hout, int wout, int stride, int pad, int ell, int s, int L,
                   cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(n) * cout * hout * wout;
  conv2d_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
      x, w, b, y, stats, n, cin, hin, win, cout, kh, kw, hout, wout, stride, pad, ell, s, L);
  return cudaGetLastError();
}

cudaError_t linear(const int64_t* x, const int64_t* w, const int64_t* b, int64_t* y,
                   int64_t* stats, int rows, int k, int m, int ell, int s, int L,
                   cudaStream_t stream) {
  linear_kernel<<<blocks_for(static_cast<int64_t>(rows) * m), kThreads, 0, stream>>>(
      x, w, b, y, stats, rows, k, m, ell, s, L);
  return cudaGetLastError();
}

cudaError_t channel_affine(const int64_t* x, const int64_t* mult, const int64_t* offset,
                           int64_t* y, int64_t* stats, int64_t total, int64_t channels,
                           int64_t inner, int ell, int s, int L, cudaStream_t stream) {
  channel_affine_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
      x, mult, offset, y, stats, total, channels, inner, ell, s, L);
  return cudaGetLastError();
}

}  // namespace ring_cuda
