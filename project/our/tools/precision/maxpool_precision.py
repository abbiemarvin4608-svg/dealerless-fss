"""Restrict integer MaxPool inputs and outputs to signed Q(bits,s).

The surrounding network retains its base format and accumulator policy.
Conversion uses integer floor scaling and signed saturation. The maximum is
selected among the converted codes, with the activation format's minimum as
the padding sentinel, before an exact lift back to the surrounding format.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPoolPrecisionStats:
    """Device counters for the MaxPool precision boundaries.

    Input rounding and clipping count conversion before the comparison.
    The selected output is already an activation-format code: its repeated
    output-format conversion makes no change, so output rounding and clipping
    are zero. The output is then lifted to base codes using integer scaling.
    """

    _counter_names = (
        "maxpool_input_lower_clipped", "maxpool_input_upper_clipped",
        "maxpool_input_rounded",
    )

    def __init__(self, activation_bits: int, activation_s: int,
                 track_stats: bool = True):
        self.activation_bits = activation_bits
        self.activation_s = activation_s
        self.track_stats = track_stats
        self.module_names: list[str] = []
        self.reset()

    def reset(self):
        self.input_total = self.output_total = self.call_count = 0
        self.module_call_counts: dict[str, int] = {}
        self._device_counts: dict[torch.device, torch.Tensor] = {}

    def record(self, name: str, input_total: int, output_total: int,
               lower: torch.Tensor, upper: torch.Tensor, rounded: torch.Tensor):
        if not self.track_stats:
            return
        self.input_total += input_total
        self.output_total += output_total
        self.call_count += 1
        self.module_call_counts[name] = self.module_call_counts.get(name, 0) + 1
        counts = torch.stack([mask.sum(dtype=torch.int64) for mask in
                              (lower, upper, rounded)]).detach()
        previous = self._device_counts.get(counts.device)
        self._device_counts[counts.device] = counts if previous is None else previous + counts

    def snapshot(self):
        totals = [0] * len(self._counter_names)
        for values in self._device_counts.values():
            for index, value in enumerate(values.tolist()):
                totals[index] += int(value)
        result = dict(zip(self._counter_names, totals))
        result.update(
            maxpool_calls=self.call_count,
            maxpool_module_count=len(self.module_names),
            maxpool_module_call_counts=dict(self.module_call_counts),
            maxpool_input_total=self.input_total,
            maxpool_output_total=self.output_total,
            maxpool_input_clipped=totals[0] + totals[1],
            maxpool_output_clipped=0,
            maxpool_output_rounded=0,
            maxpool_bits=self.activation_bits,
            maxpool_s=self.activation_s,
            maxpool_overflow_mode="saturate",
            maxpool_rounding_mode="floor",
            maxpool_track_stats=self.track_stats,
            maxpool_output_counter_scope="selected_quantized_code_before_base_lift",
            maxpool_comparison_scope="signed_activation_codes",
        )
        total = self.input_total
        for suffix in ("clipped", "lower_clipped", "upper_clipped", "rounded"):
            result[f"maxpool_input_{suffix}_pct"] = (
                100.0 * result[f"maxpool_input_{suffix}"] / total if total else 0.0)
        result["maxpool_output_clipped_pct"] = 0.0
        result["maxpool_output_rounded_pct"] = 0.0
        return result


class ActivationPrecisionMaxPool2d(nn.Module):
    """Compare only signed activation-format codes, returning base int64 codes."""

    def __init__(self, module: nn.Module, activation_bits: int = 16,
                 activation_s: int = 6, base_ell: int = 24, base_s: int = 15,
                 stats: MaxPoolPrecisionStats | None = None, name: str = "maxpool"):
        super().__init__()
        from activation_precision import ActivationPrecisionReLU
        from uniform_fixed_point import UniformMaxPool2d

        # Reuse the established format validation and integer rescale helper.
        ActivationPrecisionReLU(activation_bits, activation_s, base_ell, base_s,
                                stats=None)
        if isinstance(module, (UniformMaxPool2d, ActivationPrecisionMaxPool2d)):
            self.kernel, self.stride, self.padding = module.kernel, module.stride, module.padding
        else:
            raise TypeError("MaxPool precision control requires UniformMaxPool2d")
        self.activation_bits, self.activation_s = activation_bits, activation_s
        self.base_ell, self.base_s = base_ell, base_s
        self.qmin, self.qmax = -(1 << (activation_bits - 1)), (1 << (activation_bits - 1)) - 1
        self.padding_code = self.qmin
        self.stats = stats or MaxPoolPrecisionStats(activation_bits, activation_s)
        self.name = name

    @staticmethod
    def _rescale_integer(codes: torch.Tensor, source_s: int, target_s: int):
        from activation_precision import ActivationPrecisionReLU
        return ActivationPrecisionReLU._rescale_integer(codes, source_s, target_s)

    def forward(self, inputs: torch.Tensor):
        if inputs.dtype != torch.int64:
            raise TypeError("fixed MaxPool control requires int64 base codes")
        if inputs.ndim != 4:
            raise ValueError("fixed MaxPool control requires NCHW inputs")
        raw_codes = self._rescale_integer(inputs, self.base_s, self.activation_s)
        activation_codes = raw_codes.clamp(self.qmin, self.qmax)
        ph, pw = self.padding
        padded = F.pad(activation_codes, (pw, pw, ph, ph), value=self.padding_code)
        output_codes = padded.unfold(2, self.kernel[0], self.stride[0]).unfold(
            3, self.kernel[1], self.stride[1]).amax(dim=(-2, -1))
        result = self._rescale_integer(output_codes, self.activation_s, self.base_s)
        if self.stats.track_stats:
            if self.activation_s < self.base_s:
                rounded = torch.remainder(inputs, 1 << (self.base_s - self.activation_s)) != 0
            else:
                rounded = torch.zeros_like(inputs, dtype=torch.bool)
            self.stats.record(self.name, inputs.numel(), result.numel(),
                              raw_codes < self.qmin, raw_codes > self.qmax, rounded)
        return result


def inject_maxpool_quantizers(model: nn.Module, activation_bits: int = 16,
                             activation_s: int = 6, base_ell: int = 24,
                             base_s: int = 15, track_stats: bool = True):
    """Replace all UniformMaxPool2d modules in place, preserving aliases.

    Returns a map of module paths to replacement modules and their shared
    collector. Reinjection inherits kernel/stride/padding and starts fresh
    statistics. No other module or surrounding fixed-point policy is changed.
    """
    from activation_precision import ActivationPrecisionReLU
    from uniform_fixed_point import UniformMaxPool2d

    ActivationPrecisionReLU(activation_bits, activation_s, base_ell, base_s)
    stats = MaxPoolPrecisionStats(activation_bits, activation_s, track_stats)
    replacements: dict[str, ActivationPrecisionMaxPool2d] = {}
    original_to_replacement: dict[int, ActivationPrecisionMaxPool2d] = {}

    def visit(parent: nn.Module, prefix: str):
        for child_name, child in list(parent._modules.items()):
            if child is None:
                continue
            path = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (UniformMaxPool2d, ActivationPrecisionMaxPool2d)):
                replacement = original_to_replacement.get(id(child))
                if replacement is None:
                    replacement = ActivationPrecisionMaxPool2d(
                        child, activation_bits, activation_s, base_ell, base_s, stats, path)
                    original_to_replacement[id(child)] = replacement
                    stats.module_names.append(path)
                parent._modules[child_name] = replacement
                replacements[path] = replacement
            else:
                visit(child, path)

    visit(model, "")
    if not replacements:
        raise ValueError("model contains no UniformMaxPool2d modules")
    return replacements, stats
