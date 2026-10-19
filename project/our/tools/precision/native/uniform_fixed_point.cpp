#include <torch/extension.h>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <vector>
#include "uniform_fixed_point_math.h"
#ifdef UNIFORM_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "uniform_fixed_point_cuda.h"
#endif

namespace {
void check_codes(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == torch::kInt64 && t.is_contiguous(),
              "uniform kernels require contiguous int64 codes");
}
bool cuda_device(std::initializer_list<const torch::Tensor*> tensors) {
  const auto device = (*tensors.begin())->device();
  for (const auto* t : tensors) {
    check_codes(*t);
    TORCH_CHECK(t->numel() == 0 || t->device() == device, "mixed tensor devices");
  }
#ifndef UNIFORM_WITH_CUDA
  TORCH_CHECK(!device.is_cuda(), "uniform kernels built without CUDA");
#endif
  return device.is_cuda();
}
void check_format(int ell, int s, int register_bits, uint64_t terms) {
  TORCH_CHECK(ell >= 2 && ell <= 32 && s >= 0 && s < ell,
              "require 2 <= ell <= 32 and 0 <= s < ell");
  TORCH_CHECK(register_bits >= 2 && register_bits <= 128,
              "register_bits must be in [2,128]");
  // All operands in the converted model are ell-bit storage codes. With
  // ell<=32 and K<=2^32-1, products, every prefix, and the lifted bias
  // need at most 2*ell+32+1<=97 bits, strictly below int128's signed range.
  TORCH_CHECK(terms <= uint64_t{0xffffffff}, "dot product exceeds safe int128 term bound");
}
const int64_t* pointer(const torch::Tensor& t) {
  return t.numel() ? t.data_ptr<int64_t>() : nullptr;
}
struct Stats {
  int64_t clips = 0, overflows = 0;
  int register_peak = 1, storage_peak = 1;
  torch::Tensor tensor() const {
    return torch::tensor(std::vector<int64_t>{clips, overflows, register_peak, storage_peak},
                         torch::kInt64);
  }
};
#ifdef UNIFORM_WITH_CUDA
void checked_launch(cudaError_t result) {
  TORCH_CHECK(result == cudaSuccess, "uniform CUDA kernel: ", cudaGetErrorString(result));
}
#endif
}
using Result = std::tuple<torch::Tensor, torch::Tensor>;

Result linear(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
              int ell, int s, int register_bits, bool fast64, bool ring, bool wrap) {
  TORCH_CHECK(!ring || wrap, "ring arithmetic requires wrap overflow mode");
  const bool cuda = cuda_device({&x, &w, &b});
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1),
              "linear expects [N,K], [M,K]");
  TORCH_CHECK(b.numel() == 0 || b.numel() == w.size(0), "invalid bias shape");
  const int64_t rows = x.size(0), k = x.size(1), m = w.size(0);
  check_format(ell, s, register_bits, k);
  auto y = torch::empty({rows, m}, x.options());
  if (cuda) {
#ifdef UNIFORM_WITH_CUDA
    const c10::cuda::CUDAGuard guard(x.device());
    auto stats = torch::zeros({4}, x.options());
    if (y.numel()) checked_launch(uniform_cuda::linear(
        pointer(x), pointer(w), pointer(b), y.data_ptr<int64_t>(), stats.data_ptr<int64_t>(),
        rows, k, m, ell, s, register_bits, fast64, ring, wrap, at::cuda::getCurrentCUDAStream()));
    return {y, stats};
#endif
  }
  const auto *xp = pointer(x), *wp = pointer(w), *bp = pointer(b);
  auto* yp = y.data_ptr<int64_t>();
  Stats st;
  for (int64_t r = 0; r < rows; ++r) for (int64_t j = 0; j < m; ++j) {
    __int128 acc = 0;
    uniform_fixed::Trace trace;
    for (int64_t t = 0; t < k; ++t) {
      const __int128 product = static_cast<__int128>(xp[r*k+t]) * wp[j*k+t];
      acc = uniform_fixed::mac(acc, product, ell, ring, trace);
    }
    yp[r*m+j] = uniform_fixed::finish(acc, s, bp ? bp[j] : 0, ell, register_bits,
        trace, st.clips, st.overflows, st.register_peak, st.storage_peak, ring, wrap);
  }
  return {y, st.tensor()};
}

Result conv2d(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
              int sh, int sw, int ph, int pw, int dh, int dw,
              int ell, int s, int register_bits, bool fast64, bool ring, bool wrap) {
  TORCH_CHECK(!ring || wrap, "ring arithmetic requires wrap overflow mode");
  const bool cuda = cuda_device({&x, &w, &b});
  TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && x.size(1) == w.size(1),
              "conv expects [N,C,H,W], [O,C,kh,kw]");
  TORCH_CHECK(b.numel() == 0 || b.numel() == w.size(0), "invalid bias shape");
  TORCH_CHECK(sh > 0 && sw > 0 && ph >= 0 && pw >= 0 && dh > 0 && dw > 0,
              "invalid convolution dimensions");
  const int n=x.size(0), cin=x.size(1), hin=x.size(2), win=x.size(3);
  const int cout=w.size(0), kh=w.size(2), kw=w.size(3);
  const int hout=(hin+2*ph-dh*(kh-1)-1)/sh+1, wout=(win+2*pw-dw*(kw-1)-1)/sw+1;
  TORCH_CHECK(hout > 0 && wout > 0, "kernel larger than padded input");
  check_format(ell, s, register_bits, static_cast<uint64_t>(cin)*kh*kw);
  auto y = torch::empty({n,cout,hout,wout}, x.options());
  if (cuda) {
#ifdef UNIFORM_WITH_CUDA
    const c10::cuda::CUDAGuard guard(x.device());
    auto stats = torch::zeros({4}, x.options());
    if (y.numel()) checked_launch(uniform_cuda::conv2d(
        pointer(x), pointer(w), pointer(b), y.data_ptr<int64_t>(), stats.data_ptr<int64_t>(),
        n,cin,hin,win,cout,kh,kw,hout,wout,sh,sw,ph,pw,dh,dw,ell,s,register_bits,fast64,ring,wrap,
        at::cuda::getCurrentCUDAStream()));
    return {y, stats};
#endif
  }
  const auto *xp=pointer(x), *wp=pointer(w), *bp=pointer(b);
  auto* yp=y.data_ptr<int64_t>();
  Stats st;
  for (int64_t index=0; index<y.numel(); ++index) {
    int64_t t=index;
    const int ow=t%wout; t/=wout;
    const int oh=t%hout; t/=hout;
    const int oc=t%cout, batch=t/cout;
    __int128 acc=0;
    uniform_fixed::Trace trace;
    for (int ic=0; ic<cin; ++ic) for (int r=0; r<kh; ++r) {
      const int ih=oh*sh-ph+r*dh;
      if (ih<0 || ih>=hin) continue;
      for (int c=0; c<kw; ++c) {
        const int iw=ow*sw-pw+c*dw;
        if (iw<0 || iw>=win) continue;
        const int64_t xv=xp[((static_cast<int64_t>(batch)*cin+ic)*hin+ih)*win+iw];
        const int64_t wv=wp[((static_cast<int64_t>(oc)*cin+ic)*kh+r)*kw+c];
        const __int128 product=static_cast<__int128>(xv)*wv;
        acc=uniform_fixed::mac(acc,product,ell,ring,trace);
      }
    }
    yp[index]=uniform_fixed::finish(acc,s,bp?bp[oc]:0,ell,register_bits,trace,
        st.clips,st.overflows,st.register_peak,st.storage_peak,ring,wrap);
  }
  return {y,st.tensor()};
}

Result channel_affine(const torch::Tensor& x, const torch::Tensor& mult,
                      const torch::Tensor& offset, int ell, int s, int register_bits, bool ring, bool wrap) {
  TORCH_CHECK(!ring || wrap, "ring arithmetic requires wrap overflow mode");
  const bool cuda=cuda_device({&x,&mult,&offset});
  TORCH_CHECK(x.dim()>=2, "affine expects [N,C,...]");
  const int64_t channels=x.size(1), total=x.numel();
  TORCH_CHECK(mult.numel()==channels && offset.numel()==channels,"invalid affine shape");
  check_format(ell,s,register_bits,1);
  const int64_t inner=x.size(0)*channels ? total/(x.size(0)*channels) : 1;
  auto y=torch::empty_like(x);
  if (cuda) {
#ifdef UNIFORM_WITH_CUDA
    const c10::cuda::CUDAGuard guard(x.device());
    auto stats=torch::zeros({4},x.options());
    if(total) checked_launch(uniform_cuda::channel_affine(pointer(x),pointer(mult),pointer(offset),
        y.data_ptr<int64_t>(),stats.data_ptr<int64_t>(),total,channels,inner,ell,s,register_bits,ring,wrap,
        at::cuda::getCurrentCUDAStream()));
    return {y,stats};
#endif
  }
  const auto *xp=pointer(x), *mp=pointer(mult), *op=pointer(offset);
  auto* yp=y.data_ptr<int64_t>();
  Stats st;
  for(int64_t i=0;i<total;++i) {
    const int64_t c=(i/inner)%channels;
    const __int128 product=static_cast<__int128>(xp[i])*mp[c];
    uniform_fixed::Trace trace;
    const __int128 reduced=ring?uniform_fixed::reduce_ring(product,ell,trace):product;
    if(!ring) trace.observe(product);
    yp[i]=uniform_fixed::finish(reduced,s,op[c],ell,register_bits,trace,
        st.clips,st.overflows,st.register_peak,st.storage_peak,ring,wrap);
  }
  return {y,st.tensor()};
}

Result average(const torch::Tensor& x, int ell, int register_bits, bool ring, bool wrap) {
  TORCH_CHECK(!ring || wrap, "ring arithmetic requires wrap overflow mode");
  const bool cuda=cuda_device({&x});
  TORCH_CHECK(x.dim()==4,"global average expects [N,C,H,W]");
  const int64_t rows=x.size(0)*x.size(1), count=x.size(2)*x.size(3);
  TORCH_CHECK(count>0,"empty pooling window");
  check_format(ell,0,register_bits,count);
  auto y=torch::empty({x.size(0),x.size(1),1,1},x.options());
  if(cuda) {
#ifdef UNIFORM_WITH_CUDA
    const c10::cuda::CUDAGuard guard(x.device());
    auto stats=torch::zeros({4},x.options());
    if(rows) checked_launch(uniform_cuda::average(pointer(x),y.data_ptr<int64_t>(),
        stats.data_ptr<int64_t>(),rows,count,ell,register_bits,ring,wrap,at::cuda::getCurrentCUDAStream()));
    return {y,stats};
#endif
  }
  const auto* xp=pointer(x); auto* yp=y.data_ptr<int64_t>(); Stats st;
  for(int64_t row=0;row<rows;++row) {
    __int128 sum=0; uniform_fixed::Trace trace;
    for(int64_t k=0;k<count;++k) {
      if(ring) sum=uniform_fixed::reduce_ring(sum+xp[row*count+k],ell,trace);
      else {sum+=xp[row*count+k]; trace.observe(sum);}
    }
    yp[row]=uniform_fixed::finish_average(sum,count,ell,register_bits,trace,
        st.clips,st.overflows,st.register_peak,st.storage_peak,ring,wrap);
  }
  return {y,st.tensor()};
}

Result combine_limb_dots(const torch::Tensor& dots,const torch::Tensor& shifts,
                        const torch::Tensor& bias,int ell,int s,int register_bits,
                        int proof_bits, bool ring, bool wrap) {
  TORCH_CHECK(!ring || wrap, "ring arithmetic requires wrap overflow mode");
  TORCH_CHECK(dots.scalar_type()==torch::kInt32 && dots.is_contiguous() && dots.dim()==3,
              "limb dots must be contiguous int32 [L,M,N]");
  check_codes(shifts); check_codes(bias);
  const int64_t limbs=dots.size(0), rows=dots.size(1), columns=dots.size(2);
  TORCH_CHECK(limbs>0 && shifts.numel()==limbs && shifts.dim()==1,"invalid limb shifts");
  TORCH_CHECK(bias.numel()==0 || (bias.dim()==1 && bias.numel()==columns),"invalid limb bias");
  TORCH_CHECK(shifts.device()==dots.device() &&
              (bias.numel()==0 || bias.device()==dots.device()),"mixed limb tensor devices");
  check_format(ell,s,register_bits,limbs);
  TORCH_CHECK(proof_bits>=1 && proof_bits<=128,"invalid accumulator proof width");
  TORCH_CHECK(ring || proof_bits<=register_bits,"register overflow: proven prefix bound exceeds register_bits");
  const int64_t smallest=shifts.min().item<int64_t>(), largest=shifts.max().item<int64_t>();
  // int32 dot * 2^62 uses <=94 bits; <=2^32 terms need <=126 bits.
  // Thus all limb products and prefixes stay strictly inside signed int128.
  TORCH_CHECK(smallest>=0 && largest<=62,"limb shifts must lie in [0,62]");
  auto y=torch::empty({rows,columns},bias.options().device(dots.device()));
  if(dots.is_cuda()) {
#ifdef UNIFORM_WITH_CUDA
    const c10::cuda::CUDAGuard guard(dots.device());
    auto stats=torch::zeros({4},y.options());
    if(y.numel()) checked_launch(uniform_cuda::combine_limb_dots(
        dots.data_ptr<int32_t>(),shifts.data_ptr<int64_t>(),pointer(bias),
        y.data_ptr<int64_t>(),stats.data_ptr<int64_t>(),limbs,rows,columns,
        ell,s,register_bits,proof_bits,ring,wrap,at::cuda::getCurrentCUDAStream()));
    return {y,stats};
#else
    TORCH_CHECK(false,"uniform kernels built without CUDA");
#endif
  }
  const auto* dp=dots.data_ptr<int32_t>(); const auto* sp=pointer(shifts);
  const auto* bp=pointer(bias); auto* yp=y.data_ptr<int64_t>(); Stats st;
  st.register_peak=ring?1:proof_bits;
  const int64_t size=rows*columns;
  for(int64_t i=0;i<size;++i) {
    __int128 acc=0; uniform_fixed::Trace trace;
    for(int64_t limb=0;limb<limbs;++limb) {
      const __int128 term=static_cast<__int128>(dp[limb*size+i])*(static_cast<__int128>(1)<<sp[limb]);
      acc=uniform_fixed::mac(acc,term,ell,ring,trace);
    }
    yp[i]=uniform_fixed::finish(acc,s,bp?bp[i%columns]:0,ell,register_bits,trace,
                              st.clips,st.overflows,st.register_peak,st.storage_peak,ring,wrap);
  }
  return {y,st.tensor()};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  namespace py=pybind11;
  m.def("linear",&linear,py::arg("x"),py::arg("weight"),py::arg("bias"),
        py::arg("ell"),py::arg("s"),py::arg("register_bits"),py::arg("fast64")=false,
        py::arg("ring")=true,py::arg("wrap")=true);
  m.def("conv2d",&conv2d,py::arg("x"),py::arg("weight"),py::arg("bias"),
        py::arg("sh"),py::arg("sw"),py::arg("ph"),py::arg("pw"),py::arg("dh"),py::arg("dw"),
        py::arg("ell"),py::arg("s"),py::arg("register_bits"),py::arg("fast64")=false,
        py::arg("ring")=true,py::arg("wrap")=true);
  m.def("channel_affine",&channel_affine,py::arg("x"),py::arg("mult"),py::arg("offset"),
        py::arg("ell"),py::arg("s"),py::arg("register_bits"),py::arg("ring")=true,py::arg("wrap")=true);
  m.def("average",&average,py::arg("x"),py::arg("ell"),py::arg("register_bits"),
        py::arg("ring")=true,py::arg("wrap")=true);
  m.def("combine_limb_dots",&combine_limb_dots,py::arg("dots"),py::arg("shifts"),py::arg("bias"),
        py::arg("ell"),py::arg("s"),py::arg("register_bits"),py::arg("proof_bits"),
        py::arg("ring")=true,py::arg("wrap")=true);
}
