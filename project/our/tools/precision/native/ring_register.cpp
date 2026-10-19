// Tensor checks, CPU reference kernels and device dispatch for E0-R.
#include <torch/extension.h>

#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ring_register_math.h"

#ifdef RING_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "ring_register_cuda.h"
#endif

namespace {

void check_codes(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == torch::kInt64, name, " must hold int64 codes");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// True when the non-empty tensors live on CUDA; raises on mixed devices.
bool on_cuda(std::initializer_list<const torch::Tensor*> tensors) {
  const bool first = (*tensors.begin())->is_cuda();
  for (const auto* t : tensors) {
    TORCH_CHECK(t->numel() == 0 || t->is_cuda() == first,
                "ring kernels require all tensors on one device");
  }
#ifndef RING_WITH_CUDA
  TORCH_CHECK(!first, "ring kernels were built without CUDA");
#endif
  return first;
}

const int64_t* optional_ptr(const torch::Tensor& t) {
  return t.numel() ? t.data_ptr<int64_t>() : nullptr;
}

struct CpuStats {
  int64_t overflow = 0, wraps = 0;
  int max_bits = 0;
  torch::Tensor tensor() const {
    return torch::tensor(std::vector<int64_t>{overflow, wraps, max_bits}, torch::kInt64);
  }
};

#ifdef RING_WITH_CUDA
void check_launch(cudaError_t err) {
  TORCH_CHECK(err == cudaSuccess, "ring CUDA kernel failed: ", cudaGetErrorString(err));
}
#endif

}  // namespace

using RingResult = std::tuple<torch::Tensor, torch::Tensor>;

RingResult linear(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
                  int ell, int s, int L) {
  check_codes(x, "x");
  check_codes(w, "weight");
  check_codes(b, "bias");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1),
              "linear expects x [N, K] and weight [M, K]");
  TORCH_CHECK(b.numel() == 0 || b.numel() == w.size(0), "bias must be empty or [M]");
  const int64_t rows = x.size(0), k = x.size(1), m = w.size(0);
  auto y = torch::empty({rows, m}, x.options());
  if (on_cuda({&x, &w, &b})) {
#ifdef RING_WITH_CUDA
    const c10::cuda::CUDAGuard guard(x.device());
    auto stats = torch::zeros({3}, x.options());
    if (y.numel()) {
      check_launch(ring_cuda::linear(
          x.data_ptr<int64_t>(), w.data_ptr<int64_t>(), optional_ptr(b), y.data_ptr<int64_t>(),
          stats.data_ptr<int64_t>(), static_cast<int>(rows), static_cast<int>(k),
          static_cast<int>(m), ell, s, L, at::cuda::getCurrentCUDAStream()));
    }
    return {y, stats};
#endif
  }
  const auto* xp = x.data_ptr<int64_t>();
  const auto* wp = w.data_ptr<int64_t>();
  const int64_t* bp = optional_ptr(b);
  auto* yp = y.data_ptr<int64_t>();
  const uint64_t mask = ring::mask_bits(L);
  CpuStats st;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t j = 0; j < m; ++j) {
      uint64_t acc = 0;
      __int128 exact = 0;
      for (int64_t t = 0; t < k; ++t) {
        acc = ring::mac(acc, xp[r * k + t], wp[j * k + t], mask);
        exact += static_cast<__int128>(xp[r * k + t]) * wp[j * k + t];
      }
      if (bp != nullptr) {
        acc = (acc + static_cast<uint64_t>(bp[j])) & mask;
        exact += bp[j];
      }
      yp[r * m + j] = ring::finish(acc, exact, s, ell, L, st.overflow, st.wraps, st.max_bits);
    }
  }
  return {y, st.tensor()};
}

RingResult conv2d(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
                  int stride, int pad, int ell, int s, int L) {
  check_codes(x, "x");
  check_codes(w, "weight");
  check_codes(b, "bias");
  TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && x.size(1) == w.size(1),
              "conv2d expects x [N, C, H, W] and weight [O, C, kh, kw]");
  TORCH_CHECK(b.numel() == 0 || b.numel() == w.size(0), "bias must be empty or [O]");
  TORCH_CHECK(stride >= 1 && pad >= 0, "conv2d needs stride >= 1 and padding >= 0");
  const int n = x.size(0), cin = x.size(1), hin = x.size(2), win = x.size(3);
  const int cout = w.size(0), kh = w.size(2), kw = w.size(3);
  const int hout = (hin + 2 * pad - kh) / stride + 1;
  const int wout = (win + 2 * pad - kw) / stride + 1;
  TORCH_CHECK(hout > 0 && wout > 0, "conv2d kernel larger than padded input");
  auto y = torch::empty({n, cout, hout, wout}, x.options());
  if (on_cuda({&x, &w, &b})) {
#ifdef RING_WITH_CUDA
    const c10::cuda::CUDAGuard guard(x.device());
    auto stats = torch::zeros({3}, x.options());
    if (y.numel()) {
      check_launch(ring_cuda::conv2d(
          x.data_ptr<int64_t>(), w.data_ptr<int64_t>(), optional_ptr(b), y.data_ptr<int64_t>(),
          stats.data_ptr<int64_t>(), n, cin, hin, win, cout, kh, kw, hout, wout, stride, pad,
          ell, s, L, at::cuda::getCurrentCUDAStream()));
    }
    return {y, stats};
#endif
  }
  const auto* xp = x.data_ptr<int64_t>();
  const auto* wp = w.data_ptr<int64_t>();
  const int64_t* bp = optional_ptr(b);
  auto* yp = y.data_ptr<int64_t>();
  const uint64_t mask = ring::mask_bits(L);
  CpuStats st;
  const int64_t total = y.numel();
  for (int64_t index = 0; index < total; ++index) {
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
        const int64_t* xrow = xp + ((static_cast<int64_t>(batch) * cin + ic) * hin + ih) * win;
        const int64_t* wrow = wp + ((static_cast<int64_t>(oc) * cin + ic) * kh + r) * kw;
        for (int q = 0; q < kw; ++q) {
          const int iw = ow * stride - pad + q;
          if (iw < 0 || iw >= win) continue;
          acc = ring::mac(acc, xrow[iw], wrow[q], mask);
          exact += static_cast<__int128>(xrow[iw]) * wrow[q];
        }
      }
    }
    if (bp != nullptr) {
      acc = (acc + static_cast<uint64_t>(bp[oc])) & mask;
      exact += bp[oc];
    }
    yp[index] = ring::finish(acc, exact, s, ell, L, st.overflow, st.wraps, st.max_bits);
  }
  return {y, st.tensor()};
}

RingResult channel_affine(const torch::Tensor& x, const torch::Tensor& mult,
                          const torch::Tensor& offset, int ell, int s, int L) {
  check_codes(x, "x");
  check_codes(mult, "mult");
  check_codes(offset, "offset");
  TORCH_CHECK(x.dim() >= 2, "channel_affine expects x [N, C, ...]");
  const int64_t channels = x.size(1);
  TORCH_CHECK(mult.numel() == channels && offset.numel() == channels,
              "mult and offset must hold one code per channel");
  const int64_t total = x.numel();
  const int64_t outer = x.size(0) * channels;
  const int64_t inner = outer ? total / outer : 1;
  auto y = torch::empty_like(x);
  if (on_cuda({&x, &mult, &offset})) {
#ifdef RING_WITH_CUDA
    const c10::cuda::CUDAGuard guard(x.device());
    auto stats = torch::zeros({3}, x.options());
    if (total) {
      check_launch(ring_cuda::channel_affine(
          x.data_ptr<int64_t>(), mult.data_ptr<int64_t>(), offset.data_ptr<int64_t>(),
          y.data_ptr<int64_t>(), stats.data_ptr<int64_t>(), total, channels, inner, ell, s, L,
          at::cuda::getCurrentCUDAStream()));
    }
    return {y, stats};
#endif
  }
  const auto* xp = x.data_ptr<int64_t>();
  const auto* mp = mult.data_ptr<int64_t>();
  const auto* op = offset.data_ptr<int64_t>();
  auto* yp = y.data_ptr<int64_t>();
  const uint64_t mask = ring::mask_bits(L);
  CpuStats st;
  for (int64_t index = 0; index < total; ++index) {
    const int64_t c = (index / inner) % channels;
    const uint64_t acc =
        (ring::mac(0, xp[index], mp[c], mask) + static_cast<uint64_t>(op[c])) & mask;
    const __int128 exact = static_cast<__int128>(xp[index]) * mp[c] + op[c];
    yp[index] = ring::finish(acc, exact, s, ell, L, st.overflow, st.wraps, st.max_bits);
  }
  return {y, st.tensor()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear, "TR_s(x W^T + b) in an L-bit register");
  m.def("conv2d", &conv2d, "zero-padded conv2d in an L-bit register");
  m.def("channel_affine", &channel_affine, "per-channel TR_s(x * mult + offset)");
}
