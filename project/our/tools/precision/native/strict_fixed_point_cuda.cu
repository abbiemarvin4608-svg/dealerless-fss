#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

__device__ __forceinline__ int64_t floor_shift(int64_t value, int bits) {
  if (bits == 0) {
    return value;
  }
  const int64_t scale = int64_t{1} << bits;
  return value >= 0 ? value >> bits : -(((-value) + scale - 1) >> bits);
}

__device__ __forceinline__ int64_t clamp_count(
    int64_t value, int64_t qmin, int64_t qmax, uint64_t& count) {
  if (value < qmin) {
    ++count;
    return qmin;
  }
  if (value > qmax) {
    ++count;
    return qmax;
  }
  return value;
}

__global__ void strict_conv2d_kernel(
    const int64_t* __restrict__ input,
    const int64_t* __restrict__ weight,
    const int64_t* __restrict__ bias,
    int64_t* __restrict__ output,
    int64_t* __restrict__ stats,
    int n, int cin, int hin, int win, int cout, int kh, int kw,
    int hout, int wout, int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int frac_bits, int64_t qmin,
    int64_t qmax, bool has_bias) {
  const int64_t total = static_cast<int64_t>(n) * cout * hout * wout;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t t = index;
    const int ow = t % wout; t /= wout;
    const int oh = t % hout; t /= hout;
    const int oc = t % cout; t /= cout;
    const int batch = static_cast<int>(t);
    int64_t acc = 0;
    uint64_t mul_sat = 0;
    uint64_t add_sat = 0;
    for (int ic = 0; ic < cin; ++ic) {
      for (int r = 0; r < kh; ++r) {
        const int ih = oh * stride_h - pad_h + r * dilation_h;
        for (int c = 0; c < kw; ++c) {
          const int iw = ow * stride_w - pad_w + c * dilation_w;
          int64_t x = 0;
          if (ih >= 0 && ih < hin && iw >= 0 && iw < win) {
            x = input[((static_cast<int64_t>(batch) * cin + ic) * hin + ih) * win + iw];
          }
          const int64_t w = weight[((static_cast<int64_t>(oc) * cin + ic) * kh + r) * kw + c];
          int64_t product = floor_shift(x * w, frac_bits);
          product = clamp_count(product, qmin, qmax, mul_sat);
          acc = clamp_count(acc + product, qmin, qmax, add_sat);
        }
      }
    }
    if (has_bias) {
      acc = clamp_count(acc + bias[oc], qmin, qmax, add_sat);
    }
    output[index] = acc;
    if (mul_sat != 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(stats),
                static_cast<unsigned long long>(mul_sat));
    }
    if (add_sat != 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(stats + 1),
                static_cast<unsigned long long>(add_sat));
    }
  }
}

__global__ void strict_linear_kernel(
    const int64_t* __restrict__ input,
    const int64_t* __restrict__ weight,
    const int64_t* __restrict__ bias,
    int64_t* __restrict__ output,
    int64_t* __restrict__ stats,
    int rows, int in_features, int out_features, int frac_bits,
    int64_t qmin, int64_t qmax, bool has_bias) {
  const int64_t total = static_cast<int64_t>(rows) * out_features;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int row = index / out_features;
    const int col = index % out_features;
    int64_t acc = 0;
    uint64_t mul_sat = 0;
    uint64_t add_sat = 0;
    for (int k = 0; k < in_features; ++k) {
      int64_t product = floor_shift(
          input[static_cast<int64_t>(row) * in_features + k] *
              weight[static_cast<int64_t>(col) * in_features + k],
          frac_bits);
      product = clamp_count(product, qmin, qmax, mul_sat);
      acc = clamp_count(acc + product, qmin, qmax, add_sat);
    }
    if (has_bias) {
      acc = clamp_count(acc + bias[col], qmin, qmax, add_sat);
    }
    output[index] = acc;
    if (mul_sat != 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(stats),
                static_cast<unsigned long long>(mul_sat));
    }
    if (add_sat != 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(stats + 1),
                static_cast<unsigned long long>(add_sat));
    }
  }
}

__global__ void strict_max_pool2d_kernel(
    const int64_t* __restrict__ input, int64_t* __restrict__ output,
    int n, int channels, int hin, int win, int hout, int wout,
    int kh, int kw, int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int64_t qmin) {
  const int64_t total = static_cast<int64_t>(n) * channels * hout * wout;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t t = index;
    const int ow = t % wout; t /= wout;
    const int oh = t % hout; t /= hout;
    const int ch = t % channels; t /= channels;
    const int batch = static_cast<int>(t);
    int64_t best = qmin;
    for (int r = 0; r < kh; ++r) {
      const int ih = oh * stride_h - pad_h + r * dilation_h;
      for (int c = 0; c < kw; ++c) {
        const int iw = ow * stride_w - pad_w + c * dilation_w;
        if (ih >= 0 && ih < hin && iw >= 0 && iw < win) {
          const int64_t value = input[((static_cast<int64_t>(batch) * channels + ch) * hin + ih) * win + iw];
          best = value > best ? value : best;
        }
      }
    }
    output[index] = best;
  }
}

__global__ void strict_global_avg_pool2d_kernel(
    const int64_t* __restrict__ input, int64_t* __restrict__ output,
    int64_t* __restrict__ stats, int n, int channels, int hin, int win,
    int64_t qmin, int64_t qmax) {
  const int64_t total = static_cast<int64_t>(n) * channels;
  const int count = hin * win;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int batch = index / channels;
    const int ch = index % channels;
    int64_t acc = 0;
    uint64_t add_sat = 0;
    for (int i = 0; i < count; ++i) {
      const int h = i / win;
      const int w = i % win;
      const int64_t value = input[((batch * channels + ch) * hin + h) * win + w];
      acc = clamp_count(acc + value, qmin, qmax, add_sat);
    }
    // Inputs to the ResNet global average pool are non-negative after ReLU,
    // but retain mathematical floor division for a general signed input.
    output[index] = acc >= 0 ? acc / count : -(((-acc) + count - 1) / count);
    if (add_sat != 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(stats + 1),
                static_cast<unsigned long long>(add_sat));
    }
  }
}

int blocks_for(int64_t elements) {
  constexpr int threads = 256;
  return static_cast<int>(std::min<int64_t>((elements + threads - 1) / threads, 65535));
}

void check_common(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "expected a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, "expected torch.int64 codes");
  TORCH_CHECK(tensor.is_contiguous(), "expected contiguous tensor");
}

}  // namespace

std::vector<torch::Tensor> strict_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t frac_bits,
    int64_t qmin, int64_t qmax) {
  check_common(input);
  check_common(weight);
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "conv2d expects NCHW tensors");
  TORCH_CHECK(input.size(1) == weight.size(1), "input/weight channel mismatch");
  const bool has_bias = bias.numel() != 0;
  if (has_bias) check_common(bias);
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t hout = (input.size(2) + 2 * pad_h - dilation_h * (weight.size(2) - 1) - 1) / stride_h + 1;
  const int64_t wout = (input.size(3) + 2 * pad_w - dilation_w * (weight.size(3) - 1) - 1) / stride_w + 1;
  auto output = torch::empty({input.size(0), weight.size(0), hout, wout}, input.options());
  auto stats = torch::zeros({2}, input.options());
  const int64_t total = output.numel();
  strict_conv2d_kernel<<<blocks_for(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<int64_t>(), weight.data_ptr<int64_t>(),
      has_bias ? bias.data_ptr<int64_t>() : nullptr, output.data_ptr<int64_t>(),
      stats.data_ptr<int64_t>(), input.size(0), input.size(1), input.size(2),
      input.size(3), weight.size(0), weight.size(2), weight.size(3), hout,
      wout, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
      frac_bits, qmin, qmax, has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, stats};
}

std::vector<torch::Tensor> strict_linear_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t frac_bits, int64_t qmin, int64_t qmax) {
  check_common(input);
  check_common(weight);
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "linear expects matrices");
  TORCH_CHECK(input.size(1) == weight.size(1), "input/weight feature mismatch");
  const bool has_bias = bias.numel() != 0;
  if (has_bias) check_common(bias);
  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty({input.size(0), weight.size(0)}, input.options());
  auto stats = torch::zeros({2}, input.options());
  strict_linear_kernel<<<blocks_for(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<int64_t>(), weight.data_ptr<int64_t>(),
      has_bias ? bias.data_ptr<int64_t>() : nullptr, output.data_ptr<int64_t>(),
      stats.data_ptr<int64_t>(), input.size(0), input.size(1), weight.size(0),
      frac_bits, qmin, qmax, has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, stats};
}

torch::Tensor strict_max_pool2d_cuda(
    torch::Tensor input, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t qmin) {
  check_common(input);
  TORCH_CHECK(input.dim() == 4, "max_pool2d expects NCHW input");
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t hout = (input.size(2) + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  const int64_t wout = (input.size(3) + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
  auto output = torch::empty({input.size(0), input.size(1), hout, wout}, input.options());
  strict_max_pool2d_kernel<<<blocks_for(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<int64_t>(), output.data_ptr<int64_t>(), input.size(0),
      input.size(1), input.size(2), input.size(3), hout, wout, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, qmin);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> strict_global_avg_pool2d_cuda(
    torch::Tensor input, int64_t qmin, int64_t qmax) {
  check_common(input);
  TORCH_CHECK(input.dim() == 4, "global_avg_pool2d expects NCHW input");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty({input.size(0), input.size(1), 1, 1}, input.options());
  auto stats = torch::zeros({2}, input.options());
  strict_global_avg_pool2d_kernel<<<blocks_for(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<int64_t>(), output.data_ptr<int64_t>(), stats.data_ptr<int64_t>(),
      input.size(0), input.size(1), input.size(2), input.size(3), qmin, qmax);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, stats};
}
