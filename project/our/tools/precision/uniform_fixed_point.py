"""Uniform Q(ell,s) storage with wide integer accumulation by default.

Inputs, parameters, and stored outputs use ell-bit saturating encodings.
Conv/FC retain exact products and accumulation in checked wide registers,
then floor-rescale once before writing back to ell bits. Pooling uses a wide
sum before division. BN folding and entry encoding are offline floating-point
work; converted network operators are integers.

Explicit arithmetic_mode='ring', overflow_mode='wrap' retains the strict
modulo-2**ell control: products, MAC prefixes, and pooling sums wrap before
signed rescaling. This module is a plaintext numerical simulator, not a
secure sharing-conversion or wider-ring MPC protocol.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_NATIVE = None


def _native_ops():
    global _NATIVE
    if _NATIVE is None:
        from torch.utils.cpp_extension import CUDA_HOME, load
        os.environ.setdefault("MAX_JOBS", "1")
        directory = Path(__file__).resolve().parent / "native"
        sources = [str(directory / "uniform_fixed_point.cpp")]
        flags = ["-O3"]
        if CUDA_HOME is not None and torch.cuda.is_available():
            sources.append(str(directory / "uniform_fixed_point_cuda.cu"))
            flags.append("-DUNIFORM_WITH_CUDA")
        _NATIVE = load(
            name="e0_uniform_fixed_point_ops_v2_ring", sources=sources,
            extra_cflags=flags, extra_cuda_cflags=["-O3"], verbose=False,
        )
    return _NATIVE


def _pair(value):
    return tuple(int(v) for v in value) if isinstance(value, tuple) else (int(value), int(value))


def _bits_tensor(codes: torch.Tensor) -> torch.Tensor:
    """Exact signed width for codes in the proven int64-safe tensor operations."""
    if not codes.numel():
        return torch.ones((), dtype=torch.int64, device=codes.device)
    magnitude = torch.maximum(codes.max(), -codes.min() - 1)
    powers = torch.tensor([1 << i for i in range(63)], device=codes.device)
    return 1 + (magnitude >= powers).sum()


class UniformFixedPoint:
    def __init__(self, ell: int, s: int, track_stats: bool = True, register_bits: int = 128,
                 integer_backend: str = "scalar", arithmetic_mode: str = "wide",
                 overflow_mode: str = "saturate"):
        if not isinstance(ell, int) or not isinstance(s, int) or not 2 <= ell <= 32 or not 0 <= s < ell:
            raise ValueError("require integer 2 <= ell <= 32 and 0 <= s < ell")
        if not isinstance(register_bits, int) or not 2 <= register_bits <= 128:
            raise ValueError("register_bits must be an integer in [2,128]")
        if integer_backend not in {"scalar", "gemm"}:
            raise ValueError("integer_backend must be 'scalar' or 'gemm'")
        if arithmetic_mode not in {"ring", "wide"} or overflow_mode not in {"wrap", "saturate"}:
            raise ValueError("invalid arithmetic_mode or overflow_mode")
        if arithmetic_mode == "ring" and overflow_mode != "wrap":
            raise ValueError("ring arithmetic requires wrap overflow semantics")
        self.arithmetic_mode, self.overflow_mode = arithmetic_mode, overflow_mode
        self.ring, self.wrap = arithmetic_mode == "ring", overflow_mode == "wrap"
        self.ell, self.s = ell, s
        self.register_bits = ell if self.ring else register_bits
        self.integer_backend = integer_backend
        self.scale = 1 << s
        self.modulus, self.sign_bit = 1 << ell, 1 << (ell-1)
        self.qmin, self.qmax = -(1 << (ell - 1)), (1 << (ell - 1)) - 1
        self.track_stats = track_stats
        # An exact fast implementation is permitted only after an absolute-sum
        # bound proves every raw product and every prefix fits signed int64.
        self.allow_int64_fastpath = True
        self.offline_clipped = self.offline_total = 0
        self.offline_max_storage_bits = 1
        self.reset_runtime_stats()

    def reset_runtime_stats(self):
        self.input_clipped = self.input_total = 0
        self.input_max_storage_bits = 1
        self._native_stats = []
        self.activation_total = 0
        self.int64_fastpath_calls = self.int128_path_calls = 0
        self.gemm_calls = 0

    def fast64_safe(self, codes, terms, weight_bound, bias_bound):
        if not codes.is_cuda or not self.allow_int64_fastpath:
            self.int128_path_calls += 1
            return False
        if self.ring:
            # Each canonical product fits int64 for ell<=32; each subsequent
            # add is reduced immediately, so no unrestricted prefix survives.
            self.int64_fastpath_calls += 1
            return True
        input_bound = int(codes.abs().amax().item()) if codes.numel() else 0
        bound = int(terms)*input_bound*int(weight_bound) + (int(bias_bound) << self.s)
        safe = bound <= (1 << 63)-1
        if safe:
            self.int64_fastpath_calls += 1
        else:
            self.int128_path_calls += 1
        return safe

    def reduce_signed(self, codes):
        """Canonical signed representative, without overflow-prone half addition."""
        if torch.is_floating_point(codes):
            residue = torch.remainder(codes, self.modulus).to(torch.int64)
        else:
            residue = codes.to(torch.int64) & (self.modulus-1)
        return torch.where(residue >= self.sign_bit, residue-self.modulus, residue)

    def encode(self, values: torch.Tensor, kind="input") -> torch.Tensor:
        if torch.is_floating_point(values):
            if not bool(torch.isfinite(values).all()):
                raise ValueError("cannot encode non-finite values")
            scaled = torch.floor(values.detach().to(torch.float64) * self.scale)
            if not bool(torch.isfinite(scaled).all()):
                raise OverflowError("scaled entry code is not finite")
        else:
            scaled = values.detach().to(torch.int64)
        mask = (scaled < self.qmin) | (scaled > self.qmax)
        encoded = (self.reduce_signed(scaled) if self.wrap else
                   scaled.clamp(self.qmin, self.qmax).to(torch.int64))
        if self.track_stats:
            clipped = int(mask.sum().item())
            if scaled.numel():
                observed = encoded if self.ring else scaled
                lo, hi = int(observed.min().item()), int(observed.max().item())
                width = 1 + max(hi, -lo - 1, 0).bit_length()
            else:
                width = 1
            if kind == "parameter":
                self.offline_clipped += clipped
                self.offline_total += scaled.numel()
                self.offline_max_storage_bits = max(self.offline_max_storage_bits, width)
            else:
                self.input_clipped += clipped
                self.input_total += scaled.numel()
                self.input_max_storage_bits = max(self.input_max_storage_bits, width)
        return encoded

    def decode(self, codes):
        return codes.to(torch.float64) / self.scale

    def record_native(self, stats, total):
        if int(stats[1].item()):
            raise OverflowError("uniform temporary register overflow in a product or accumulator prefix")
        if self.track_stats:
            self._native_stats.append(stats)
            self.activation_total += int(total)

    def _writeback(self, exact, register_values=None):
        # Residual adds and comparisons involve at most ell+1 <= 33 bits.
        values = exact if register_values is None else register_values
        clips = ((exact < self.qmin) | (exact > self.qmax)).sum()
        encoded = self.reduce_signed(exact) if self.wrap else exact.clamp(self.qmin, self.qmax).to(torch.int64)
        reg_bits = _bits_tensor(encoded if self.ring else values)
        storage_bits = _bits_tensor(encoded if self.ring else exact)
        stats = torch.stack([clips, (reg_bits > self.register_bits).to(torch.int64),
                             reg_bits, storage_bits])
        self.record_native(stats, exact.numel())
        return encoded

    def add(self, left, right):
        if self.ring:
            left, right = self.reduce_signed(left), self.reduce_signed(right)
        return self._writeback(left.to(torch.int64) + right.to(torch.int64))

    def stats(self):
        clips = overflows = 0
        register_peak = storage_peak = 1
        devices = {}
        for item in self._native_stats:
            devices.setdefault(item.device, []).append(item)
        for group in devices.values():
            values = torch.stack(group)
            sums = values[:, :2].sum(0).tolist()
            peaks = values[:, 2:].amax(0).tolist()
            clips += int(sums[0]); overflows += int(sums[1])
            register_peak = max(register_peak, int(peaks[0]))
            storage_peak = max(storage_peak, int(peaks[1]))
        if overflows:
            raise OverflowError("uniform temporary register overflow")
        result = {
            "offline_clipped": self.offline_clipped, "offline_total": self.offline_total,
            "input_clipped": self.input_clipped, "input_total": self.input_total,
            "activation_clipped": clips, "activation_total": self.activation_total,
            "offline_clipping_rate_pct": 100.0*self.offline_clipped/self.offline_total if self.offline_total else 0.0,
            "input_clipping_rate_pct": 100.0*self.input_clipped/self.input_total if self.input_total else 0.0,
            "activation_clipping_rate_pct": 100.0*clips/self.activation_total if self.activation_total else 0.0,
            "register_overflows": overflows, "max_register_bits": register_peak,
            "max_storage_bits": max(storage_peak, self.offline_max_storage_bits, self.input_max_storage_bits),
            "offline_max_storage_bits": self.offline_max_storage_bits,
            "input_max_storage_bits": self.input_max_storage_bits,
            "activation_max_storage_bits": storage_peak,
            "register_bits": self.register_bits,
            "int64_fastpath_calls": self.int64_fastpath_calls,
            "int128_path_calls": self.int128_path_calls,
            "gemm_calls": self.gemm_calls,
            "max_register_bits_kind": ("logical_ring_width" if self.ring else
                                       "proven_upper_bound" if self.gemm_calls else "observed_prefix"),
            "requested_integer_backend": self.integer_backend,
            "effective_integer_backend": "gemm" if self.gemm_calls else "scalar",
        }
        if self.wrap:
            result.update(offline_wrapped=self.offline_clipped, input_wrapped=self.input_clipped,
                          activation_wrapped=clips)
            for kind in ("offline", "input", "activation"):
                result[f"{kind}_wrapping_rate_pct"] = result[f"{kind}_clipping_rate_pct"]
        # Preserve the historical stats shape for explicit wide+saturate tests.
        if self.ring or self.wrap:
            result.update(arithmetic_mode=self.arithmetic_mode, overflow_mode=self.overflow_mode,
                          activation_overflow_event_scope=("limb_reconstruction_or_scalar_mac" if self.gemm_calls
                                                           else "scalar_mac"))
        return result


class UniformConv2d(nn.Module):
    def __init__(self, module, fp):
        super().__init__()
        if module.groups != 1 or module.padding_mode != "zeros":
            raise NotImplementedError("uniform convolution requires groups=1, zero padding")
        self.fp = fp
        self.stride, self.padding, self.dilation = _pair(module.stride), _pair(module.padding), _pair(module.dilation)
        self.register_buffer("weight_q", fp.encode(module.weight, "parameter"))
        self.register_buffer("bias_q", fp.encode(module.bias, "parameter") if module.bias is not None
                             else torch.empty(0, dtype=torch.int64, device=module.weight.device))
        self.terms = module.weight.shape[1]*module.weight.shape[2]*module.weight.shape[3]
        self.weight_bound = int(self.weight_q.abs().amax().item())
        self.bias_bound = int(self.bias_q.abs().amax().item()) if self.bias_q.numel() else 0
        self.gemm_weights = None
        if fp.integer_backend == "gemm":
            from uniform_integer_gemm import CachedIntegerGemmWeights
            self.gemm_weights = CachedIntegerGemmWeights(self.weight_q)

    def forward(self, codes):
        if self.fp.integer_backend == "gemm" and codes.is_cuda:
            from uniform_integer_gemm import conv2d_gemm
            output, stats = conv2d_gemm(codes, self.gemm_weights, self.bias_q,
                self.stride, self.padding, self.dilation, self.fp.ell, self.fp.s,
                self.fp.register_bits, _native_ops().combine_limb_dots,
                arithmetic_mode=self.fp.arithmetic_mode, overflow_mode=self.fp.overflow_mode)
            self.fp.gemm_calls += 1
            self.fp.record_native(stats, output.numel())
            return output
        fast = self.fp.fast64_safe(codes, self.terms, self.weight_bound, self.bias_bound)
        output, stats = _native_ops().conv2d(codes.contiguous(), self.weight_q.contiguous(),
            self.bias_q.contiguous(), *self.stride, *self.padding, *self.dilation,
            self.fp.ell, self.fp.s, self.fp.register_bits, fast, self.fp.ring, self.fp.wrap)
        self.fp.record_native(stats, output.numel())
        return output


class UniformLinear(nn.Module):
    def __init__(self, module, fp):
        super().__init__()
        self.fp = fp
        self.register_buffer("weight_q", fp.encode(module.weight, "parameter"))
        self.register_buffer("bias_q", fp.encode(module.bias, "parameter") if module.bias is not None
                             else torch.empty(0, dtype=torch.int64, device=module.weight.device))
        self.terms = module.weight.shape[1]
        self.weight_bound = int(self.weight_q.abs().amax().item())
        self.bias_bound = int(self.bias_q.abs().amax().item()) if self.bias_q.numel() else 0
        self.gemm_weights = None
        if fp.integer_backend == "gemm":
            from uniform_integer_gemm import CachedIntegerGemmWeights
            self.gemm_weights = CachedIntegerGemmWeights(self.weight_q)

    def forward(self, codes):
        if self.fp.integer_backend == "gemm" and codes.is_cuda:
            from uniform_integer_gemm import linear_gemm
            output, stats = linear_gemm(codes, self.gemm_weights, self.bias_q,
                self.fp.ell, self.fp.s, self.fp.register_bits, _native_ops().combine_limb_dots,
                arithmetic_mode=self.fp.arithmetic_mode, overflow_mode=self.fp.overflow_mode)
            self.fp.gemm_calls += 1
            self.fp.record_native(stats, output.numel())
            return output
        fast = self.fp.fast64_safe(codes, self.terms, self.weight_bound, self.bias_bound)
        output, stats = _native_ops().linear(codes.contiguous(), self.weight_q.contiguous(),
            self.bias_q.contiguous(), self.fp.ell, self.fp.s, self.fp.register_bits, fast,
            self.fp.ring, self.fp.wrap)
        self.fp.record_native(stats, output.numel())
        return output


class UniformBatchNorm2d(nn.Module):
    def __init__(self, module, fp):
        super().__init__()
        if not module.track_running_stats:
            raise ValueError("uniform BN requires running statistics")
        self.fp = fp
        var, mean = module.running_var.detach().double(), module.running_mean.detach().double()
        gamma = module.weight.detach().double() if module.affine else torch.ones_like(var)
        beta = module.bias.detach().double() if module.affine else torch.zeros_like(var)
        multiplier = gamma / torch.sqrt(var + module.eps)
        self.register_buffer("multiplier_q", fp.encode(multiplier, "parameter"))
        self.register_buffer("offset_q", fp.encode(beta - mean*multiplier, "parameter"))

    def forward(self, codes):
        output, stats = _native_ops().channel_affine(codes.contiguous(), self.multiplier_q.contiguous(),
            self.offset_q.contiguous(), self.fp.ell, self.fp.s, self.fp.register_bits,
            self.fp.ring, self.fp.wrap)
        self.fp.record_native(stats, output.numel())
        return output


class UniformReLU(nn.Module):
    def forward(self, codes):
        return codes.clamp_min(0)


class UniformMaxPool2d(nn.Module):
    def __init__(self, module, fp=None):
        super().__init__()
        if module.ceil_mode or _pair(module.dilation) != (1, 1):
            raise NotImplementedError("uniform maxpool requires floor mode, dilation=1")
        self.kernel, self.stride, self.padding = _pair(module.kernel_size), _pair(module.stride or module.kernel_size), _pair(module.padding)
        self.padding_code = fp.qmin if fp is not None else -(1 << 31)

    def forward(self, codes):
        ph, pw = self.padding
        padded = F.pad(codes, (pw, pw, ph, ph), value=self.padding_code)
        return padded.unfold(2, self.kernel[0], self.stride[0]).unfold(3, self.kernel[1], self.stride[1]).amax(dim=(-2,-1))


class UniformAdaptiveAvgPool2d(nn.Module):
    def __init__(self, module, fp):
        super().__init__()
        if module.output_size not in (1, (1,1)):
            raise NotImplementedError("uniform pooling supports only global average")
        self.fp = fp

    def forward(self, codes):
        output, stats = _native_ops().average(codes.contiguous(), self.fp.ell, self.fp.register_bits,
                                             self.fp.ring, self.fp.wrap)
        self.fp.record_native(stats, output.numel())
        return output


class UniformBlock(nn.Module):
    def __init__(self, block, fp):
        super().__init__()
        self.fp = fp
        depth = 3 if hasattr(block, "conv3") else 2
        self.convs = nn.ModuleList(UniformConv2d(getattr(block,f"conv{i}"),fp) for i in range(1,depth+1))
        self.bns = nn.ModuleList(UniformBatchNorm2d(getattr(block,f"bn{i}"),fp) for i in range(1,depth+1))
        self.relu = UniformReLU()
        self.downsample = None
        if block.downsample is not None:
            self.downsample = nn.Sequential(UniformConv2d(block.downsample[0],fp),UniformBatchNorm2d(block.downsample[1],fp))

    def forward(self,codes):
        identity = codes if self.downsample is None else self.downsample(codes)
        result = codes
        for i,(conv,bn) in enumerate(zip(self.convs,self.bns)):
            result = bn(conv(result))
            if i+1 < len(self.convs): result = self.relu(result)
        return self.relu(self.fp.add(result,identity))


class UniformResNet(nn.Module):
    def __init__(self, model, fp):
        super().__init__()
        self.fp = fp
        self.conv1,self.bn1 = UniformConv2d(model.conv1,fp),UniformBatchNorm2d(model.bn1,fp)
        self.relu,self.maxpool = UniformReLU(),UniformMaxPool2d(model.maxpool,fp)
        self.blocks = nn.Sequential(*(UniformBlock(block,fp) for stage in range(1,5)
                                    for block in getattr(model,f"layer{stage}")))
        self.avgpool,self.fc = UniformAdaptiveAvgPool2d(model.avgpool,fp),UniformLinear(model.fc,fp)

    def forward(self,inputs):
        codes = self.fp.encode(inputs,"input")
        codes = self.maxpool(self.relu(self.bn1(self.conv1(codes))))
        codes = self.avgpool(self.blocks(codes))
        return self.fc(codes.flatten(1))


def convert_resnet_to_uniform(model, ell, s, track_stats=True, register_bits=128,
                              integer_backend="scalar", arithmetic_mode="wide", overflow_mode="saturate"):
    return UniformResNet(model,UniformFixedPoint(ell,s,track_stats,register_bits,integer_backend,
                                                arithmetic_mode,overflow_mode)).eval()
