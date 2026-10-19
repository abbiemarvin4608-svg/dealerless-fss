"""Isolate signed fixed-point precision at every ResNet ReLU boundary.

The FP32 control quantizes only ReLU inputs/outputs.  The integer control
accepts codes in a common base Q(ell,s), converts them using integer shifts
and floor division, applies the same activation format, and returns base
codes.  All activation formats use floor encoding and signed saturation.
Neither control changes the surrounding convolution or accumulator policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivationPrecisionStats:
    """Accumulate device counters without synchronizing after each ReLU."""

    _counter_names = (
        "relu_lower_clipped", "relu_upper_clipped",
        "relu_positive_rounded", "relu_positive_zeroed",
    )

    def __init__(self, activation_bits: int, activation_s: int,
                 track_stats: bool = True):
        self.activation_bits = activation_bits
        self.activation_s = activation_s
        self.track_stats = track_stats
        self.module_names: list[str] = []
        self.reset()

    def reset(self):
        self.input_total = 0
        self.call_count = 0
        self.module_call_counts: dict[str, int] = {}
        self._device_counts: dict[torch.device, torch.Tensor] = {}

    def record(self, name: str, total: int, lower: torch.Tensor,
               upper: torch.Tensor, rounded: torch.Tensor,
               zero_changes: torch.Tensor):
        if not self.track_stats:
            return
        self.input_total += total
        self.call_count += 1
        self.module_call_counts[name] = self.module_call_counts.get(name, 0) + 1
        counts = torch.stack([mask.sum(dtype=torch.int64) for mask in
                              (lower, upper, rounded, zero_changes)]).detach()
        previous = self._device_counts.get(counts.device)
        self._device_counts[counts.device] = counts if previous is None else previous + counts

    def snapshot(self):
        # One small transfer per device, independent of the network depth.
        totals = [0] * len(self._counter_names)
        for values in self._device_counts.values():
            for index, value in enumerate(values.tolist()):
                totals[index] += int(value)
        result = dict(zip(self._counter_names, totals))
        result.update(
            relu_input_total=self.input_total,
            relu_output_total=self.input_total,
            relu_input_clipped=totals[0] + totals[1],
            relu_calls=self.call_count,
            relu_module_count=len(self.module_names),
            relu_module_call_counts=dict(self.module_call_counts),
            relu_bits=self.activation_bits,
            relu_s=self.activation_s,
            relu_overflow_mode="saturate",
            relu_rounding_mode="floor",
            relu_track_stats=self.track_stats,
        )
        total = self.input_total
        result["relu_input_clipping_pct"] = 100.0 * result["relu_input_clipped"] / total if total else 0.0
        result["relu_lower_clipping_pct"] = 100.0 * totals[0] / total if total else 0.0
        result["relu_upper_clipping_pct"] = 100.0 * totals[1] / total if total else 0.0
        result["relu_positive_rounding_pct"] = 100.0 * totals[2] / total if total else 0.0
        result["relu_positive_zeroing_pct"] = 100.0 * totals[3] / total if total else 0.0
        return result


class ActivationPrecisionReLU(nn.Module):
    """Quantize a signed pre-ReLU input and its output to Q(bits,s).

    With no base format the input/output are floating tensors.  With a base
    format they are int64 codes: all conversions stay in integer arithmetic.
    The signed 16-bit format therefore has maximum code 32767, even though
    the ReLU output is nonnegative.  The output is converted back to its
    surrounding representation, without an additional floating computation.
    """

    def __init__(self, activation_bits: int, activation_s: int,
                 base_ell: int | None = None, base_s: int | None = None,
                 stats: ActivationPrecisionStats | None = None,
                 name: str = "relu"):
        super().__init__()
        if (not isinstance(activation_bits, int) or not isinstance(activation_s, int)
                or not 2 <= activation_bits <= 32 or not 0 <= activation_s < activation_bits):
            raise ValueError("require 2 <= activation_bits <= 32 and 0 <= activation_s < activation_bits")
        if (base_ell is None) != (base_s is None):
            raise ValueError("base_ell and base_s must be supplied together")
        if base_ell is not None:
            if (not isinstance(base_ell, int) or not isinstance(base_s, int)
                    or not 2 <= base_ell <= 32 or not 0 <= base_s < base_ell):
                raise ValueError("require 2 <= base_ell <= 32 and 0 <= base_s < base_ell")
        self.activation_bits, self.activation_s = activation_bits, activation_s
        self.base_ell, self.base_s = base_ell, base_s
        self.qmin, self.qmax = -(1 << (activation_bits - 1)), (1 << (activation_bits - 1)) - 1
        self.scale = 1 << activation_s
        self.stats = stats or ActivationPrecisionStats(activation_bits, activation_s)
        self.name = name

    @staticmethod
    def _rescale_integer(codes: torch.Tensor, source_s: int, target_s: int):
        difference = target_s - source_s
        if difference > 0:
            return codes * (1 << difference)
        if difference < 0:
            return torch.div(codes, 1 << -difference, rounding_mode="floor")
        return codes

    def forward(self, inputs: torch.Tensor):
        if self.base_ell is None:
            if not torch.is_floating_point(inputs):
                raise TypeError("floating ReLU control requires floating inputs")
            scaled = inputs * self.scale
            raw_codes = torch.floor(scaled)
            # Clamp before integer conversion so very large finite FP32
            # values cannot overflow the int64 implementation container.
            activation_codes = raw_codes.clamp(self.qmin, self.qmax).to(torch.int64)
            output_codes = activation_codes.clamp_min(0)
            if self.stats.track_stats:
                self.stats.record(self.name, inputs.numel(), raw_codes < self.qmin,
                                  raw_codes > self.qmax,
                                  (inputs > 0) & (scaled != raw_codes),
                                  (inputs > 0) & (output_codes == 0))
            return output_codes.to(dtype=inputs.dtype) / self.scale

        if inputs.dtype != torch.int64:
            raise TypeError("fixed ReLU control requires int64 base codes")
        raw_codes = self._rescale_integer(inputs, self.base_s, self.activation_s)
        activation_codes = raw_codes.clamp(self.qmin, self.qmax)
        output_codes = activation_codes.clamp_min(0)
        result = self._rescale_integer(output_codes, self.activation_s, self.base_s)
        if self.stats.track_stats:
            if self.activation_s < self.base_s:
                rounded = (inputs > 0) & (torch.remainder(inputs, 1 << (self.base_s - self.activation_s)) != 0)
            else:
                rounded = torch.zeros_like(inputs, dtype=torch.bool)
            self.stats.record(self.name, inputs.numel(), raw_codes < self.qmin,
                              raw_codes > self.qmax, rounded,
                              (inputs > 0) & (result == 0))
        # Saturation at the activation boundary cannot increase a canonical
        # positive base code; this exact return conversion needs no extra clip.
        return result


def inject_relu_quantizers(model: nn.Module, activation_bits: int = 16,
                           activation_s: int = 12,
                           base_ell: int | None = None,
                           base_s: int | None = None,
                           track_stats: bool = True):
    """Replace or reconfigure every torchvision/UniformResNet ReLU in place.

    Returns ``(replacements, stats)``.  ``replacements`` maps module paths to
    replacement modules; ``stats.reset()`` and ``stats.snapshot()`` control
    their common statistics.  A Bottleneck's shared ReLU remains one module
    called three times, so invocation counts exceed the number of modules.
    Explicit base_ell/base_s select the integer path.  Reinjection replaces
    previous quantizers and installs a fresh collector, so one converted
    network can be reused across an activation-format scan.
    """
    from uniform_fixed_point import UniformReLU

    stats = ActivationPrecisionStats(activation_bits, activation_s, track_stats)
    # Construct first to validate arguments even if the model has no ReLU.
    ActivationPrecisionReLU(activation_bits, activation_s, base_ell, base_s, stats)
    replacements: dict[str, ActivationPrecisionReLU] = {}
    original_to_replacement: dict[int, ActivationPrecisionReLU] = {}

    def visit(parent: nn.Module, prefix: str):
        # _modules preserves alias names; named_children deduplicates aliases.
        for child_name, child in list(parent._modules.items()):
            if child is None:
                continue
            path = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (nn.ReLU, UniformReLU, ActivationPrecisionReLU)):
                integer_child = isinstance(child, UniformReLU) or (
                    isinstance(child, ActivationPrecisionReLU) and child.base_ell is not None)
                if integer_child and base_ell is None:
                    raise ValueError("integer ReLU requires an explicit integer base format")
                replacement = original_to_replacement.get(id(child))
                if replacement is None:
                    replacement = ActivationPrecisionReLU(
                        activation_bits, activation_s, base_ell, base_s, stats, path)
                    original_to_replacement[id(child)] = replacement
                    stats.module_names.append(path)
                parent._modules[child_name] = replacement
                replacements[path] = replacement
            else:
                visit(child, path)

    visit(model, "")
    if not replacements:
        raise ValueError("model contains no nn.ReLU or UniformReLU modules")
    return replacements, stats
