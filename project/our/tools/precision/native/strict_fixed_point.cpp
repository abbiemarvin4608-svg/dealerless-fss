#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> strict_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t frac_bits,
    int64_t qmin, int64_t qmax);

std::vector<torch::Tensor> strict_linear_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t frac_bits, int64_t qmin, int64_t qmax);

torch::Tensor strict_max_pool2d_cuda(
    torch::Tensor input, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t qmin);

std::vector<torch::Tensor> strict_global_avg_pool2d_cuda(
    torch::Tensor input, int64_t qmin, int64_t qmax);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &strict_conv2d_cuda,
        "Strict Q(ell,s) convolution (CUDA)");
  m.def("linear", &strict_linear_cuda,
        "Strict Q(ell,s) linear layer (CUDA)");
  m.def("max_pool2d", &strict_max_pool2d_cuda,
        "Integer max-pool (CUDA)");
  m.def("global_avg_pool2d", &strict_global_avg_pool2d_cuda,
        "Strict Q(ell,s) global average pool (CUDA)");
}
