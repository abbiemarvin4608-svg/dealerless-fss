// Torch-free CUDA implementation: explicit ell-bit residues or legacy checked wide arithmetic.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include "uniform_fixed_point_math.h"
#include "uniform_fixed_point_cuda.h"

namespace {
constexpr int kThreads=256;
int blocks(int64_t count) { return static_cast<int>(std::min<int64_t>((count+255)/256,4096)); }

__device__ void publish(int64_t* stats,int64_t clips,int64_t overflows,int reg,int storage) {
  long long c=clips,o=overflows;
  for(int delta=16;delta;delta>>=1) {
    c+=__shfl_down_sync(0xffffffffu,c,delta);
    o+=__shfl_down_sync(0xffffffffu,o,delta);
    reg=max(reg,__shfl_down_sync(0xffffffffu,reg,delta));
    storage=max(storage,__shfl_down_sync(0xffffffffu,storage,delta));
  }
  if((threadIdx.x&31)==0) {
    auto* p=reinterpret_cast<unsigned long long*>(stats);
    if(c) atomicAdd(p,static_cast<unsigned long long>(c));
    if(o) atomicAdd(p+1,static_cast<unsigned long long>(o));
    atomicMax(p+2,static_cast<unsigned long long>(reg));
    atomicMax(p+3,static_cast<unsigned long long>(storage));
  }
}

template <typename Accumulator>
__global__ void conv_kernel(const int64_t* __restrict__ x,const int64_t* __restrict__ w,
    const int64_t* __restrict__ b,int64_t* __restrict__ y,int64_t* stats,
    int n,int cin,int hin,int win,int cout,int kh,int kw,int hout,int wout,
    int sh,int sw,int ph,int pw,int dh,int dw,int ell,int s,int register_bits,bool ring,bool wrap) {
  const int64_t total=static_cast<int64_t>(n)*cout*hout*wout;
  int64_t clips=0,overflows=0; int reg=1,storage=1;
  for(int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      index<total;index+=static_cast<int64_t>(blockDim.x)*gridDim.x) {
    int64_t t=index; const int ow=t%wout; t/=wout;
    const int oh=t%hout; t/=hout; const int oc=t%cout,batch=t/cout;
    Accumulator acc=0; uniform_fixed::Trace trace;
    for(int ic=0;ic<cin;++ic) for(int r=0;r<kh;++r) {
      const int ih=oh*sh-ph+r*dh;
      if(ih<0||ih>=hin) continue;
      const int64_t* xrow=x+((static_cast<int64_t>(batch)*cin+ic)*hin+ih)*win;
      const int64_t* wrow=w+((static_cast<int64_t>(oc)*cin+ic)*kh+r)*kw;
      for(int q=0;q<kw;++q) {
        const int iw=ow*sw-pw+q*dw;
        if(iw<0||iw>=win) continue;
        if(ring) acc=static_cast<Accumulator>(uniform_fixed::mac(acc,
            static_cast<__int128>(xrow[iw])*wrow[q],ell,true,trace));
        else {
          const Accumulator product=static_cast<Accumulator>(xrow[iw])*wrow[q];
          trace.observe(product);acc+=product;trace.observe(acc);
        }
      }
    }
    y[index]=uniform_fixed::finish(acc,s,b?b[oc]:0,ell,register_bits,trace,
                                 clips,overflows,reg,storage,ring,wrap);
  }
  publish(stats,clips,overflows,reg,storage);
}

template <typename Accumulator>
__global__ void linear_kernel(const int64_t* __restrict__ x,const int64_t* __restrict__ w,
    const int64_t* __restrict__ b,int64_t* __restrict__ y,int64_t* stats,
    int rows,int k,int m,int ell,int s,int register_bits,bool ring,bool wrap) {
  const int64_t total=static_cast<int64_t>(rows)*m;
  int64_t clips=0,overflows=0; int reg=1,storage=1;
  for(int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      index<total;index+=static_cast<int64_t>(blockDim.x)*gridDim.x) {
    const int64_t* xr=x+(index/m)*k;
    const int64_t* wr=w+(index%m)*k;
    Accumulator acc=0; uniform_fixed::Trace trace;
    for(int i=0;i<k;++i) {
      if(ring) acc=static_cast<Accumulator>(uniform_fixed::mac(acc,
          static_cast<__int128>(xr[i])*wr[i],ell,true,trace));
      else {
        const Accumulator product=static_cast<Accumulator>(xr[i])*wr[i];
        trace.observe(product);acc+=product;trace.observe(acc);
      }
    }
    y[index]=uniform_fixed::finish(acc,s,b?b[index%m]:0,ell,register_bits,trace,
                                 clips,overflows,reg,storage,ring,wrap);
  }
  publish(stats,clips,overflows,reg,storage);
}

__global__ void affine_kernel(const int64_t* __restrict__ x,const int64_t* __restrict__ mult,
    const int64_t* __restrict__ offset,int64_t* __restrict__ y,int64_t* stats,
    int64_t total,int64_t channels,int64_t inner,int ell,int s,int register_bits,bool ring,bool wrap) {
  int64_t clips=0,overflows=0; int reg=1,storage=1;
  for(int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      index<total;index+=static_cast<int64_t>(blockDim.x)*gridDim.x) {
    const int64_t channel=(index/inner)%channels;
    const __int128 product=static_cast<__int128>(x[index])*mult[channel];
    uniform_fixed::Trace trace;
    const __int128 reduced=ring?uniform_fixed::reduce_ring(product,ell,trace):product;
    if(!ring)trace.observe(product);
    y[index]=uniform_fixed::finish(reduced,s,offset[channel],ell,register_bits,trace,
                                 clips,overflows,reg,storage,ring,wrap);
  }
  publish(stats,clips,overflows,reg,storage);
}

__global__ void average_kernel(const int64_t* __restrict__ x,int64_t* __restrict__ y,
    int64_t* stats,int64_t rows,int64_t count,int ell,int register_bits,bool ring,bool wrap) {
  int64_t clips=0,overflows=0; int reg=1,storage=1;
  for(int64_t row=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      row<rows;row+=static_cast<int64_t>(blockDim.x)*gridDim.x) {
    __int128 sum=0; uniform_fixed::Trace trace;
    for(int64_t i=0;i<count;++i) {
      if(ring)sum=uniform_fixed::reduce_ring(sum+x[row*count+i],ell,trace);
      else {sum+=x[row*count+i];trace.observe(sum);}
    }
    y[row]=uniform_fixed::finish_average(sum,count,ell,register_bits,trace,
                                         clips,overflows,reg,storage,ring,wrap);
  }
  publish(stats,clips,overflows,reg,storage);
}

__global__ void combine_kernel(const int32_t* __restrict__ dots,const int64_t* __restrict__ shifts,
    const int64_t* __restrict__ bias,int64_t* __restrict__ y,int64_t* stats,
    int64_t limbs,int64_t rows,int64_t columns,int ell,int s,int register_bits,int proof_bits,bool ring,bool wrap) {
  const int64_t total=rows*columns;
  int64_t clips=0,overflows=0; int reg=ring?1:proof_bits,storage=1;
  for(int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      i<total;i+=static_cast<int64_t>(blockDim.x)*gridDim.x) {
    __int128 acc=0; uniform_fixed::Trace trace;
    for(int64_t limb=0;limb<limbs;++limb) {
      const __int128 term=static_cast<__int128>(dots[limb*total+i])*
                          (static_cast<__int128>(1)<<shifts[limb]);
      acc=uniform_fixed::mac(acc,term,ell,ring,trace);
    }
    y[i]=uniform_fixed::finish(acc,s,bias?bias[i%columns]:0,ell,register_bits,trace,
                              clips,overflows,reg,storage,ring,wrap);
  }
  publish(stats,clips,overflows,reg,storage);
}
}
namespace uniform_cuda {
cudaError_t conv2d(const int64_t* x,const int64_t* w,const int64_t* b,int64_t* y,int64_t* st,
    int n,int cin,int hin,int win,int cout,int kh,int kw,int hout,int wout,
    int sh,int sw,int ph,int pw,int dh,int dw,int ell,int s,int reg,bool fast64,bool ring,bool wrap,cudaStream_t stream) {
  if(fast64) conv_kernel<int64_t><<<blocks(static_cast<int64_t>(n)*cout*hout*wout),kThreads,0,stream>>>(
      x,w,b,y,st,n,cin,hin,win,cout,kh,kw,hout,wout,sh,sw,ph,pw,dh,dw,ell,s,reg,ring,wrap);
  else conv_kernel<__int128><<<blocks(static_cast<int64_t>(n)*cout*hout*wout),kThreads,0,stream>>>(
      x,w,b,y,st,n,cin,hin,win,cout,kh,kw,hout,wout,sh,sw,ph,pw,dh,dw,ell,s,reg,ring,wrap);
  return cudaGetLastError();
}
cudaError_t linear(const int64_t* x,const int64_t* w,const int64_t* b,int64_t* y,int64_t* st,
    int rows,int k,int m,int ell,int s,int reg,bool fast64,bool ring,bool wrap,cudaStream_t stream) {
  if(fast64) linear_kernel<int64_t><<<blocks(static_cast<int64_t>(rows)*m),kThreads,0,stream>>>(x,w,b,y,st,rows,k,m,ell,s,reg,ring,wrap);
  else linear_kernel<__int128><<<blocks(static_cast<int64_t>(rows)*m),kThreads,0,stream>>>(x,w,b,y,st,rows,k,m,ell,s,reg,ring,wrap);
  return cudaGetLastError();
}
cudaError_t channel_affine(const int64_t* x,const int64_t* mult,const int64_t* offset,
    int64_t* y,int64_t* st,int64_t total,int64_t channels,int64_t inner,
    int ell,int s,int reg,bool ring,bool wrap,cudaStream_t stream) {
  affine_kernel<<<blocks(total),kThreads,0,stream>>>(x,mult,offset,y,st,total,channels,inner,ell,s,reg,ring,wrap);
  return cudaGetLastError();
}
cudaError_t average(const int64_t* x,int64_t* y,int64_t* st,int64_t rows,int64_t count,
    int ell,int reg,bool ring,bool wrap,cudaStream_t stream) {
  average_kernel<<<blocks(rows),kThreads,0,stream>>>(x,y,st,rows,count,ell,reg,ring,wrap);
  return cudaGetLastError();
}
cudaError_t combine_limb_dots(const int32_t* dots,const int64_t* shifts,const int64_t* bias,
    int64_t* y,int64_t* st,int64_t limbs,int64_t rows,int64_t columns,
    int ell,int s,int reg,int proof_bits,bool ring,bool wrap,cudaStream_t stream) {
  combine_kernel<<<blocks(rows*columns),kThreads,0,stream>>>(dots,shifts,bias,y,st,
      limbs,rows,columns,ell,s,reg,proof_bits,ring,wrap);
  return cudaGetLastError();
}
}
