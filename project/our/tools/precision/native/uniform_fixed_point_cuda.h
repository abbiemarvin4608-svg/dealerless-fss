#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace uniform_cuda {
cudaError_t conv2d(const int64_t*, const int64_t*, const int64_t*, int64_t*, int64_t*,
                  int n, int cin, int hin, int win, int cout, int kh, int kw,
                  int hout, int wout, int sh, int sw, int ph, int pw,
                  int dh, int dw, int ell, int s, int register_bits, bool fast64, bool ring, bool wrap, cudaStream_t);
cudaError_t linear(const int64_t*, const int64_t*, const int64_t*, int64_t*, int64_t*,
                  int, int, int, int, int, int, bool, bool, bool, cudaStream_t);
cudaError_t channel_affine(const int64_t*, const int64_t*, const int64_t*, int64_t*,
                          int64_t*, int64_t, int64_t, int64_t, int, int, int, bool, bool, cudaStream_t);
cudaError_t average(const int64_t*, int64_t*, int64_t*, int64_t, int64_t,
                    int, int, bool, bool, cudaStream_t);
cudaError_t combine_limb_dots(const int32_t*, const int64_t*, const int64_t*,
                             int64_t*, int64_t*, int64_t, int64_t, int64_t,
                             int, int, int, int, bool, bool, cudaStream_t);
}
