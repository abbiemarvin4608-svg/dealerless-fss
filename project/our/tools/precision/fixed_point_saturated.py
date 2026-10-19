"""Strict saturating integer fixed-point inference for E0-Q-fixed.

Every persistent value is a signed ``ell``-bit integer code at scale ``2**s``.
An elementary multiplication may use a host ``2*ell``-bit temporary solely to
form the product; it is immediately floor-rescaled and saturated back to the
agreed Q(ell,s) format. Dot products then perform this fixed-point multiply and
an ell-bit saturating add for every term. There is no widened semantic
accumulator and no floating-point network operator.

The implementation has a literal CPU reference path and a CUDA kernel with the
same ordered multiply-then-add semantics. Floating point is used only to
encode the input/model constants and by the separate FP32 reference model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_NATIVE = None


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _native_ops():
    """Compile/load the exact CUDA kernels lazily on first CUDA inference."""
    global _NATIVE
    if _NATIVE is None:
        from torch.utils.cpp_extension import load

        native = Path(__file__).resolve().parent / "native"
        _NATIVE = load(
            name="e0_q_fixed_strict_ops_v1",
            sources=[
                str(native / "strict_fixed_point.cpp"),
                str(native / "strict_fixed_point_cuda.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _NATIVE


class SaturatingFixedPoint:
    """Strict Q(ell,s) arithmetic with operation-level saturation statistics."""

    def __init__(self, ell: int, s: int, track_stats: bool = True):
        if not isinstance(ell, int) or not isinstance(s, int):
            raise TypeError("ell and s must be integers")
        if ell < 2 or ell > 32 or s < 0 or s >= ell:
            raise ValueError("require 2 <= ell <= 32 and 0 <= s < ell")
        self.ell, self.s = ell, s
        self.scale = 1 << s
        self.qmin = -(1 << (ell - 1))
        self.qmax = (1 << (ell - 1)) - 1
        self.min_value = self.qmin / self.scale
        self.max_value = self.qmax / self.scale
        self.track_stats = track_stats
        self.offline_clipped = 0
        self.offline_total = 0
        self.reset_runtime_stats()

    def reset_runtime_stats(self) -> None:
        self.input_clipped = 0
        self.input_total = 0
        self._stat_tensors: Dict[str, list[torch.Tensor]] = {
            "mul": [], "add": []
        }
        self._stat_totals = {"mul": 0, "add": 0}

    def _record_mask(self, kind: str, clipped: torch.Tensor) -> None:
        if not self.track_stats:
            return
        self._stat_tensors[kind].append(clipped.to(torch.int64).sum())
        self._stat_totals[kind] += clipped.numel()

    def record_native(self, stats: torch.Tensor, mul_total: int,
                      add_total: int) -> None:
        if not self.track_stats:
            return
        self._stat_tensors["mul"].append(stats[0])
        self._stat_tensors["add"].append(stats[1])
        self._stat_totals["mul"] += int(mul_total)
        self._stat_totals["add"] += int(add_total)

    @staticmethod
    def _sum_scalars(values: list[torch.Tensor]) -> int:
        if not values:
            return 0
        by_device: Dict[str, list[torch.Tensor]] = {}
        for value in values:
            by_device.setdefault(str(value.device), []).append(value)
        total = 0
        for group in by_device.values():
            total += int(torch.stack(group).sum().item())
        return total

    def stats(self) -> dict:
        mul_sat = self._sum_scalars(self._stat_tensors["mul"])
        add_sat = self._sum_scalars(self._stat_tensors["add"])
        mul_total = self._stat_totals["mul"]
        add_total = self._stat_totals["add"]
        return {
            "offline_clipped": self.offline_clipped,
            "offline_total": self.offline_total,
            "offline_clipping_rate_pct": (
                100.0 * self.offline_clipped / self.offline_total
                if self.offline_total else 0.0
            ),
            "input_clipped": self.input_clipped,
            "input_total": self.input_total,
            "input_clipping_rate_pct": (
                100.0 * self.input_clipped / self.input_total
                if self.input_total else 0.0
            ),
            "mul_saturations": mul_sat,
            "mul_operations": mul_total,
            "mul_saturation_rate_pct": 100.0 * mul_sat / mul_total if mul_total else 0.0,
            "add_saturations": add_sat,
            "add_operations": add_total,
            "add_saturation_rate_pct": 100.0 * add_sat / add_total if add_total else 0.0,
        }

    def encode(self, x: torch.Tensor, kind: str = "input") -> torch.Tensor:
        """Encode a float once; all subsequent network operations use codes."""
        if not torch.is_floating_point(x):
            return x.to(torch.int64).clamp(self.qmin, self.qmax)
        scaled = torch.floor(x.to(torch.float64) * self.scale)
        clipped = (scaled < self.qmin) | (scaled > self.qmax)
        if self.track_stats:
            count = int(clipped.sum().item())
            if kind == "parameter":
                self.offline_clipped += count
                self.offline_total += scaled.numel()
            else:
                self.input_clipped += count
                self.input_total += scaled.numel()
        return scaled.clamp(self.qmin, self.qmax).to(torch.int64)

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        return q.to(torch.float64) / self.scale

    def clamp(self, q: torch.Tensor, operation: str | None = None) -> torch.Tensor:
        if operation is not None:
            self._record_mask(operation, (q < self.qmin) | (q > self.qmax))
        return q.clamp(self.qmin, self.qmax).to(torch.int64)

    def add(self, *qs: torch.Tensor) -> torch.Tensor:
        if not qs:
            raise ValueError("add requires at least one tensor")
        out = qs[0].to(torch.int64)
        for q in qs[1:]:
            out = self.clamp(out + q.to(torch.int64), "add")
        return out

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Q multiply: 2ell-bit host product, floor rescale, immediate clamp."""
        product = a.to(torch.int64) * b.to(torch.int64)
        if self.s:
            product = torch.div(product, self.scale, rounding_mode="floor")
        return self.clamp(product, "mul")


def _strict_conv2d_cpu(x_q: torch.Tensor, weight_q: torch.Tensor,
                       bias_q: torch.Tensor, stride, padding, dilation,
                       fp: SaturatingFixedPoint) -> torch.Tensor:
    """Literal vectorized-over-output CPU reference; intended for tests."""
    if dilation != (1, 1):
        raise NotImplementedError("CPU reference currently requires dilation=1")
    sh, sw = stride
    ph, pw = padding
    kh, kw = weight_q.shape[-2:]
    padded = F.pad(x_q, (pw, pw, ph, ph))
    patches = padded.unfold(2, kh, sh).unfold(3, kw, sw)
    n, cin, hout, wout, _, _ = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, cin * kh * kw)
    weights = weight_q.reshape(weight_q.shape[0], -1)
    acc = torch.zeros((patches.shape[0], weights.shape[0]), dtype=torch.int64)
    for k in range(patches.shape[1]):
        term = fp.mul(patches[:, k:k + 1], weights[:, k].view(1, -1))
        acc = fp.add(acc, term)
    if bias_q.numel():
        acc = fp.add(acc, bias_q.view(1, -1))
    return acc.reshape(n, hout, wout, weights.shape[0]).permute(0, 3, 1, 2).contiguous()


class SaturatingConv2d(nn.Module):
    def __init__(self, module: nn.Conv2d, fp: SaturatingFixedPoint):
        super().__init__()
        if module.groups != 1:
            raise NotImplementedError("E0-Q-fixed Conv2d currently expects groups=1")
        if module.padding_mode != "zeros":
            raise NotImplementedError("E0-Q-fixed expects zero padding")
        self.fp = fp
        self.stride = _pair(module.stride)
        self.padding = _pair(module.padding)
        self.dilation = _pair(module.dilation)
        self.kernel_size = _pair(module.kernel_size)
        self.register_buffer("weight_q", fp.encode(module.weight.detach(), "parameter"))
        bias = (fp.encode(module.bias.detach(), "parameter")
                if module.bias is not None else torch.empty(0, dtype=torch.int64))
        self.register_buffer("bias_q", bias)

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        x_q = x_q.to(torch.int64).contiguous()
        if not x_q.is_cuda:
            return _strict_conv2d_cpu(
                x_q, self.weight_q, self.bias_q, self.stride, self.padding,
                self.dilation, self.fp
            )
        output, stats = _native_ops().conv2d(
            x_q, self.weight_q.contiguous(), self.bias_q.contiguous(),
            *self.stride, *self.padding, *self.dilation,
            self.fp.s, self.fp.qmin, self.fp.qmax,
        )
        product_count = (output.numel() * self.weight_q.shape[1] *
                         self.kernel_size[0] * self.kernel_size[1])
        add_count = product_count + (output.numel() if self.bias_q.numel() else 0)
        self.fp.record_native(stats, product_count, add_count)
        return output


class SaturatingLinear(nn.Module):
    def __init__(self, module: nn.Linear, fp: SaturatingFixedPoint):
        super().__init__()
        self.fp = fp
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.register_buffer("weight_q", fp.encode(module.weight.detach(), "parameter"))
        bias = (fp.encode(module.bias.detach(), "parameter")
                if module.bias is not None else torch.empty(0, dtype=torch.int64))
        self.register_buffer("bias_q", bias)

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        x_q = x_q.to(torch.int64).contiguous()
        if not x_q.is_cuda:
            acc = torch.zeros((*x_q.shape[:-1], self.out_features), dtype=torch.int64)
            for k in range(self.in_features):
                term = self.fp.mul(
                    x_q[..., k:k + 1], self.weight_q[:, k].view(1, -1)
                )
                acc = self.fp.add(acc, term)
            if self.bias_q.numel():
                acc = self.fp.add(acc, self.bias_q)
            return acc
        output, stats = _native_ops().linear(
            x_q, self.weight_q.contiguous(), self.bias_q.contiguous(),
            self.fp.s, self.fp.qmin, self.fp.qmax,
        )
        product_count = output.numel() * self.in_features
        add_count = product_count + (output.numel() if self.bias_q.numel() else 0)
        self.fp.record_native(stats, product_count, add_count)
        return output


class SaturatingBatchNorm2d(nn.Module):
    """Evaluation BN encoded as one strict-Q affine map."""

    def __init__(self, module: nn.BatchNorm2d, fp: SaturatingFixedPoint):
        super().__init__()
        self.fp = fp
        if not module.track_running_stats:
            raise ValueError("E0-Q-fixed requires BatchNorm running statistics")
        gamma = module.weight.detach() if module.affine else torch.ones(module.num_features)
        beta = module.bias.detach() if module.affine else torch.zeros(module.num_features)
        # BN parameters are public offline constants. The online affine map
        # below contains only fixed-point multiply and add operations.
        multiplier = gamma / torch.sqrt(module.running_var.detach() + module.eps)
        offset = beta - module.running_mean.detach() * multiplier
        self.register_buffer("multiplier_q", fp.encode(multiplier, "parameter"))
        self.register_buffer("offset_q", fp.encode(offset, "parameter"))
        self.num_features = module.num_features

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        shape = (1, self.num_features) + (1,) * (x_q.dim() - 2)
        return self.fp.add(
            self.fp.mul(x_q, self.multiplier_q.view(shape)),
            self.offset_q.view(shape),
        )


class SaturatingReLU(nn.Module):
    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(x_q.to(torch.int64), 0)


class SaturatingMaxPool2d(nn.Module):
    def __init__(self, module: nn.MaxPool2d, fp: SaturatingFixedPoint):
        super().__init__()
        self.fp = fp
        self.kernel_size = _pair(module.kernel_size)
        self.stride = _pair(module.stride or module.kernel_size)
        self.padding = _pair(module.padding)
        self.dilation = _pair(module.dilation)
        self.ceil_mode = bool(module.ceil_mode)
        if self.ceil_mode:
            raise NotImplementedError("strict CUDA max-pool does not support ceil_mode")

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        x_q = x_q.to(torch.int64).contiguous()
        if not x_q.is_cuda:
            return F.max_pool2d(
                x_q, self.kernel_size, self.stride, self.padding,
                self.dilation, self.ceil_mode,
            )
        return _native_ops().max_pool2d(
            x_q, *self.kernel_size, *self.stride, *self.padding,
            *self.dilation, self.fp.qmin,
        )


class SaturatingAdaptiveAvgPool2d(nn.Module):
    def __init__(self, module: nn.AdaptiveAvgPool2d, fp: SaturatingFixedPoint):
        super().__init__()
        if module.output_size not in (1, (1, 1)):
            raise NotImplementedError("only global average pooling is supported")
        self.fp = fp

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        x_q = x_q.to(torch.int64).contiguous()
        if not x_q.is_cuda:
            flat = x_q.flatten(-2)
            acc = torch.zeros_like(flat[..., 0])
            for i in range(flat.shape[-1]):
                acc = self.fp.add(acc, flat[..., i])
            return torch.div(
                acc, flat.shape[-1], rounding_mode="floor"
            ).unsqueeze(-1).unsqueeze(-1)
        output, stats = _native_ops().global_avg_pool2d(
            x_q, self.fp.qmin, self.fp.qmax
        )
        self.fp.record_native(stats, 0, output.numel() * x_q.shape[-2] * x_q.shape[-1])
        return output


class SaturatingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block: nn.Module, fp: SaturatingFixedPoint):
        super().__init__()
        self.fp = fp
        self.conv1 = SaturatingConv2d(block.conv1, fp)
        self.bn1 = SaturatingBatchNorm2d(block.bn1, fp)
        self.relu = SaturatingReLU()
        self.conv2 = SaturatingConv2d(block.conv2, fp)
        self.bn2 = SaturatingBatchNorm2d(block.bn2, fp)
        self.downsample = None
        if block.downsample is not None:
            self.downsample = nn.Sequential(
                SaturatingConv2d(block.downsample[0], fp),
                SaturatingBatchNorm2d(block.downsample[1], fp),
            )

    def forward(self, x_q):
        identity = x_q if self.downsample is None else self.downsample(x_q)
        out = self.relu(self.bn1(self.conv1(x_q)))
        out = self.bn2(self.conv2(out))
        return self.relu(self.fp.add(out, identity))


class SaturatingBottleneck(nn.Module):
    expansion = 4

    def __init__(self, block: nn.Module, fp: SaturatingFixedPoint):
        super().__init__()
        self.fp = fp
        self.conv1 = SaturatingConv2d(block.conv1, fp)
        self.bn1 = SaturatingBatchNorm2d(block.bn1, fp)
        self.conv2 = SaturatingConv2d(block.conv2, fp)
        self.bn2 = SaturatingBatchNorm2d(block.bn2, fp)
        self.conv3 = SaturatingConv2d(block.conv3, fp)
        self.bn3 = SaturatingBatchNorm2d(block.bn3, fp)
        self.relu = SaturatingReLU()
        self.downsample = None
        if block.downsample is not None:
            self.downsample = nn.Sequential(
                SaturatingConv2d(block.downsample[0], fp),
                SaturatingBatchNorm2d(block.downsample[1], fp),
            )

    def forward(self, x_q):
        identity = x_q if self.downsample is None else self.downsample(x_q)
        out = self.relu(self.bn1(self.conv1(x_q)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(self.fp.add(out, identity))


class SaturatingResNet(nn.Module):
    """Strict integer-domain forward for torchvision ResNet-50/101/152."""

    def __init__(self, model: nn.Module, fp: SaturatingFixedPoint):
        super().__init__()
        self.fp = fp
        self.conv1 = SaturatingConv2d(model.conv1, fp)
        self.bn1 = SaturatingBatchNorm2d(model.bn1, fp)
        self.relu = SaturatingReLU()
        self.maxpool = SaturatingMaxPool2d(model.maxpool, fp)
        cls = (SaturatingBottleneck
               if model.layer1[0].__class__.__name__ == "Bottleneck"
               else SaturatingBasicBlock)
        self.layer1 = nn.Sequential(*(cls(block, fp) for block in model.layer1))
        self.layer2 = nn.Sequential(*(cls(block, fp) for block in model.layer2))
        self.layer3 = nn.Sequential(*(cls(block, fp) for block in model.layer3))
        self.layer4 = nn.Sequential(*(cls(block, fp) for block in model.layer4))
        self.avgpool = SaturatingAdaptiveAvgPool2d(model.avgpool, fp)
        self.fc = SaturatingLinear(model.fc, fp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.fp.encode(x, "input")
        x_q = self.relu(self.bn1(self.conv1(x_q)))
        x_q = self.maxpool(x_q)
        x_q = self.layer1(x_q)
        x_q = self.layer2(x_q)
        x_q = self.layer3(x_q)
        x_q = self.layer4(x_q)
        x_q = self.avgpool(x_q)
        return self.fc(torch.flatten(x_q, 1))


def convert_resnet_to_saturating(model: nn.Module, ell: int, s: int,
                                 track_stats: bool = True) -> SaturatingResNet:
    """Build a strict-Q, non-wrapping copy without floating runtime modules."""
    fp = SaturatingFixedPoint(ell, s, track_stats=track_stats)
    return SaturatingResNet(model.cpu().eval(), fp).eval()
