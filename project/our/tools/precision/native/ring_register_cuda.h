// Raw-pointer launchers for the E0-R CUDA kernels (see ring_register_cuda.cu).
// All pointers are device pointers to contiguous int64 codes; `b` may be
// null; `stats` points to three zeroed int64 counters.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ring_cuda {

cudaError_t conv2d(const int64_t* x, const int64_t* w, const int64_t* b, int64_t* y,
                   int64_t* stats, int n, int cin, int hin, int win, int cout, int kh, int kw,
                   int hout, int wout, int stride, int pad, int ell, int s, int L,
                   cudaStream_t stream);

cudaError_t linear(const int64_t* x, const int64_t* w, const int64_t* b, int64_t* y,
                   int64_t* stats, int rows, int k, int m, int ell, int s, int L,
                   cudaStream_t stream);

cudaError_t channel_affine(const int64_t* x, const int64_t* mult, const int64_t* offset,
                           int64_t* y, int64_t* stats, int64_t total, int64_t channels,
                           int64_t inner, int ell, int s, int L, cudaStream_t stream);

}  // namespace ring_cuda
