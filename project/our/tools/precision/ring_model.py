"""E0-R network modules: ResNet inference with every value in a finite ring.

Storage tensors hold ell-bit two's-complement codes at scale 2^s.  Linear
layers (convolution, fully connected, evaluation BatchNorm) run in an L-bit
register through :mod:`ring_register`; everything else is ring arithmetic on
storage codes or a comparison:

* encoding: the plaintext owner rounds to nearest (offline, so free) and the
  code wraps into Z_{2^ell}; online truncations stay floor (``ring_register``);
* residual add: addition modulo 2^ell;
* global average pool: sum modulo 2^ell, then exact floor division by the
  public window size (CrypTFlow2's division by a public constant);
* ReLU / max pool: comparisons on the signed representatives.

Weights and BN multipliers are encoded at 2^s, biases and BN offsets at
2^(2s) so they join the accumulator before the single truncation.  Every op
records ``overflow`` (exact register value wider than L), ``wraps`` (storage
value reduced modulo 2^ell) and the exact ``max_bits``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ring_register import RingSpec, ring_channel_affine, ring_conv2d, ring_linear

# Padding positions are public, so MPC max pool leaves them out of the
# comparisons; a pad below every storage code (ell <= 32) models that.
_PAD_MIN = -(1 << 62)
_POWERS: dict[torch.device, torch.Tensor] = {}


def wrap(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed representative of ``x mod 2^bits`` (bits <= 32, |x| < 2^62)."""
    half = 1 << (bits - 1)
    return ((x + half) & ((1 << bits) - 1)) - half


def _signed_bits(x: torch.Tensor) -> torch.Tensor:
    """Two's-complement width of the widest element, kept on the device."""
    if x.numel() == 0:
        return torch.ones((), dtype=torch.int64, device=x.device)
    powers = _POWERS.get(x.device)
    if powers is None:
        powers = _POWERS[x.device] = torch.tensor([1 << k for k in range(63)],
                                                  device=x.device)
    magnitude = torch.maximum(x.max(), -x.min() - 1)  # v < 0 needs as many as -1 - v
    return 1 + (magnitude >= powers).sum()

def _storage_stats(exact: torch.Tensor, wrapped: torch.Tensor) -> torch.Tensor:
    wraps = (exact != wrapped).sum()
    return torch.stack([torch.zeros_like(wraps), wraps, _signed_bits(exact)])


def _exact_codes(value: torch.Tensor, scale_bits: int) -> torch.Tensor:
    """Round a plaintext constant to nearest at scale 2^scale_bits."""
    scaled = torch.round(value.detach().to(torch.float64) * 2.0 ** scale_bits)
    if scaled.numel() and float(scaled.abs().max()) >= 2.0 ** 62:
        raise OverflowError("fixed-point code does not fit the int64 emulator")
    return scaled.to(torch.int64)


def _square(value, what: str) -> int:
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if pair[0] != pair[1]:
        raise NotImplementedError(f"E0-R expects square {what}")
    return int(pair[0])


class RingContext:
    """Ring widths shared by one converted network plus per-op statistics.

    Codes depend only on ``s``, so one conversion can be re-run under any
    ``(ell, L)`` with the same ``s`` via :meth:`use`.
    """

    def __init__(self, spec: RingSpec):
        self.s = spec.s
        self._stats: dict[str, tuple[str, torch.Tensor]] = {}
        self.use(spec)

    def use(self, spec: RingSpec) -> None:
        if spec.s != self.s:
            raise ValueError("codes were encoded for a different fractional width")
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self._stats = {}

    def record(self, label: str, kind: str, stats: torch.Tensor) -> None:
        """Accumulate ``[overflow, wraps, max_bits]`` without a host sync."""
        entry = self._stats.get(label)
        if entry is None:
            self._stats[label] = (kind, stats.clone())
            return
        acc = entry[1]
        acc[:2] += stats[:2]
        acc[2:] = torch.maximum(acc[2:], stats[2:])

    def layer(self, label: str) -> dict:
        kind, stats = self._stats[label]
        overflow, wraps, bits = stats.tolist()
        return {"kind": kind, "overflow": overflow, "wraps": wraps, "max_bits": bits}

    def layers(self) -> list[dict]:
        return [{"layer": label, **self.layer(label)} for label in self._stats]

    def totals(self) -> dict:
        rows = self.layers()

        def peak(kind):
            return max((r["max_bits"] for r in rows if r["kind"] == kind), default=0)

        return {"register_overflows": sum(r["overflow"] for r in rows),
                "storage_wraps": sum(r["wraps"] for r in rows),
                "max_register_bits": peak("register"),
                "max_storage_bits": peak("storage")}

class RingEncode(nn.Module):
    """Client input: round to nearest at 2^s, then reduce into Z_{2^ell}."""

    def __init__(self, ctx: RingContext, label: str):
        super().__init__()
        self.ctx, self.label = ctx, label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exact = _exact_codes(x, self.ctx.s)
        y = wrap(exact, self.ctx.spec.ell)
        self.ctx.record(self.label, "storage", _storage_stats(exact, y))
        return y


class RingConv2d(nn.Module):
    def __init__(self, module: nn.Conv2d, ctx: RingContext, label: str):
        super().__init__()
        if module.groups != 1 or module.padding_mode != "zeros":
            raise NotImplementedError("E0-R expects groups=1 zero-padded convolutions")
        if _square(module.dilation, "dilation") != 1:
            raise NotImplementedError("E0-R expects dilation 1")
        self.stride = _square(module.stride, "stride")
        self.padding = _square(module.padding, "padding")
        self.ctx, self.label = ctx, label
        self.register_buffer("weight_q", _exact_codes(module.weight, ctx.s))
        bias = (_exact_codes(module.bias, 2 * ctx.s) if module.bias is not None
                else torch.empty(0, dtype=torch.int64))
        self.register_buffer("bias_q", bias)

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        y, stats = ring_conv2d(x_q, self.weight_q, self.bias_q, self.ctx.spec,
                               stride=self.stride, padding=self.padding)
        self.ctx.record(self.label, "register", stats)
        return y


class RingLinear(nn.Module):
    def __init__(self, module: nn.Linear, ctx: RingContext, label: str):
        super().__init__()
        self.ctx, self.label = ctx, label
        self.register_buffer("weight_q", _exact_codes(module.weight, ctx.s))
        bias = (_exact_codes(module.bias, 2 * ctx.s) if module.bias is not None
                else torch.empty(0, dtype=torch.int64))
        self.register_buffer("bias_q", bias)

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        y, stats = ring_linear(x_q, self.weight_q, self.bias_q, self.ctx.spec)
        self.ctx.record(self.label, "register", stats)
        return y


class RingBatchNorm2d(nn.Module):
    """Evaluation BN as one register affine: TR_s(x * mult + offset)."""

    def __init__(self, module: nn.BatchNorm2d, ctx: RingContext, label: str):
        super().__init__()
        if not module.track_running_stats:
            raise ValueError("E0-R needs BatchNorm running statistics")
        var = module.running_var.detach().double()
        mean = module.running_mean.detach().double()
        gamma = module.weight.detach().double() if module.affine else torch.ones_like(var)
        beta = module.bias.detach().double() if module.affine else torch.zeros_like(var)
        mult = gamma / torch.sqrt(var + module.eps)
        self.register_buffer("mult_q", _exact_codes(mult, ctx.s))
        self.register_buffer("offset_q", _exact_codes(beta - mean * mult, 2 * ctx.s))
        self.ctx, self.label = ctx, label

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        y, stats = ring_channel_affine(x_q, self.mult_q, self.offset_q, self.ctx.spec)
        self.ctx.record(self.label, "register", stats)
        return y

class RingReLU(nn.Module):
    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(x_q, 0)


class RingMaxPool2d(nn.Module):
    def __init__(self, module: nn.MaxPool2d):
        super().__init__()
        if module.ceil_mode or _square(module.dilation, "dilation") != 1:
            raise NotImplementedError("E0-R max pool expects floor mode and dilation 1")
        self.kernel = _square(module.kernel_size, "kernel")
        self.stride = _square(module.stride or module.kernel_size, "stride")
        self.padding = _square(module.padding, "padding")

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        p, k, st = self.padding, self.kernel, self.stride
        x = F.pad(x_q, (p, p, p, p), value=_PAD_MIN)
        return x.unfold(2, k, st).unfold(3, k, st).amax(dim=(-2, -1))


class RingAdd(nn.Module):
    """Residual addition modulo 2^ell."""

    def __init__(self, ctx: RingContext, label: str):
        super().__init__()
        self.ctx, self.label = ctx, label

    def forward(self, a_q: torch.Tensor, b_q: torch.Tensor) -> torch.Tensor:
        exact = a_q + b_q
        y = wrap(exact, self.ctx.spec.ell)
        self.ctx.record(self.label, "storage", _storage_stats(exact, y))
        return y


class RingAvgPool(nn.Module):
    """Global average pool: ring sum, then exact floor division by the count."""

    def __init__(self, ctx: RingContext, label: str):
        super().__init__()
        self.ctx, self.label = ctx, label

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        exact = x_q.sum(dim=(-2, -1), keepdim=True)
        total = wrap(exact, self.ctx.spec.ell)
        self.ctx.record(self.label, "storage", _storage_stats(exact, total))
        count = x_q.shape[-2] * x_q.shape[-1]
        return torch.div(total, count, rounding_mode="floor")

class RingBlock(nn.Module):
    """torchvision BasicBlock / Bottleneck with ring layers."""

    def __init__(self, block: nn.Module, ctx: RingContext, label: str):
        super().__init__()
        depth = 3 if hasattr(block, "conv3") else 2
        self.convs = nn.ModuleList(
            RingConv2d(getattr(block, f"conv{i}"), ctx, f"{label}.conv{i}")
            for i in range(1, depth + 1))
        self.bns = nn.ModuleList(
            RingBatchNorm2d(getattr(block, f"bn{i}"), ctx, f"{label}.bn{i}")
            for i in range(1, depth + 1))
        self.relu = RingReLU()
        self.downsample = None
        if block.downsample is not None:
            self.downsample = nn.Sequential(
                RingConv2d(block.downsample[0], ctx, f"{label}.downsample.conv"),
                RingBatchNorm2d(block.downsample[1], ctx, f"{label}.downsample.bn"))
        self.add = RingAdd(ctx, f"{label}.add")

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        identity = x_q if self.downsample is None else self.downsample(x_q)
        out = x_q
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            out = bn(conv(out))
            if i + 1 < len(self.convs):
                out = self.relu(out)
        return self.relu(self.add(out, identity))


class RingResNet(nn.Module):
    """torchvision ResNet evaluated in rings; returns logit codes at 2^s."""

    def __init__(self, model: nn.Module, ctx: RingContext):
        super().__init__()
        if model.avgpool.output_size not in (1, (1, 1)):
            raise NotImplementedError("E0-R expects global average pooling")
        self.ctx = ctx
        self.encode = RingEncode(ctx, "input")
        self.conv1 = RingConv2d(model.conv1, ctx, "conv1")
        self.bn1 = RingBatchNorm2d(model.bn1, ctx, "bn1")
        self.relu = RingReLU()
        self.maxpool = RingMaxPool2d(model.maxpool)
        self.blocks = nn.Sequential(*(
            RingBlock(block, ctx, f"layer{stage}.{index}")
            for stage in range(1, 5)
            for index, block in enumerate(getattr(model, f"layer{stage}"))))
        self.avgpool = RingAvgPool(ctx, "avgpool")
        self.fc = RingLinear(model.fc, ctx, "fc")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.maxpool(self.relu(self.bn1(self.conv1(self.encode(x)))))
        x_q = self.avgpool(self.blocks(x_q))
        return self.fc(torch.flatten(x_q, 1))


@torch.no_grad()
def convert_resnet(model: nn.Module, spec: RingSpec) -> RingResNet:
    """Encode a float ResNet once for ``spec.s``; switch widths via ``ctx.use``."""
    return RingResNet(model.eval(), RingContext(spec)).eval()
