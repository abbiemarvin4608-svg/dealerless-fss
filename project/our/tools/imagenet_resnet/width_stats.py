"""Exact signed-ring boundary counts, independent of modular residues.

Each dot event refers to the complete mathematical dot before truncation.
It does not count individual products, MAC prefixes, or limb prefixes.
"""
from __future__ import annotations

import numpy as np
import torch


SCOPES = {
    "parameter_encoding": "rounded folded weight and bias codes, once per model",
    "input_encoding": "rounded input codes, once per evaluated image",
    "dot_before_truncation": "complete signed integer dot outputs before modulo and shift; excludes bias",
    "bias_add": "truncated signed ring dot plus same-scale bias, before modulo",
    "residual_add": "sum of two signed ring inputs, before modulo",
    "avgpool_sum_before_division": "sum of 49 signed ring values, before modulo and division",
}


class RingOverflowStats:
    def __init__(self, bits):
        self.bits = bits
        self.counts = {}

    def add(self, name, affected, total):
        previous, denominator = self.counts.get(name, (0, 0))
        self.counts[name] = (previous + affected, denominator + int(total))

    def observe(self, name, values):
        half = 1 << (self.bits - 1)
        self.add(name, ((values < -half) | (values >= half)).sum(dtype=torch.int64), values.numel())

    def observe_encoding(self, name, values, frac):
        raw = torch.round(values.detach().double() * (1 << frac))
        self.observe(name, raw)

    def wrap_combiner(self, combine):
        def measured(dots, shifts, bias, ell, scale, register_bits, proof_bits, ring, wrap):
            if bias.numel() or not ring or not wrap or ell != self.bits:
                raise ValueError("dot statistics require an unbiased dot in the selected ring")
            # The native wide combiner reconstructs exact int128 sums. With
            # s=0 and no bias, status[0] counts final dot range violations.
            temporary, status = combine(dots, shifts, bias, ell, 0, 128, proof_bits, False, True)
            del temporary
            self.add("dot_before_truncation", status[0], dots.shape[1] * dots.shape[2])
            return combine(dots, shifts, bias, ell, scale, register_bits, proof_bits, ring, wrap)
        return measured

    def observe_cpu_dot(self, left, right):
        bound = left.shape[1] * int(left.abs().max()) * int(right.abs().max())
        if bound <= (1 << 63) - 1:
            self.observe("dot_before_truncation", left @ right)
            return
        exact = left.numpy().astype(object) @ right.numpy().astype(object)
        half = 1 << (self.bits - 1)
        self.add("dot_before_truncation", int(np.count_nonzero((exact < -half) | (exact >= half))), exact.size)

    def snapshot(self, reset=False):
        result = {}
        for name, (affected, total) in self.counts.items():
            affected = int(affected)
            result[name] = {"wrapped": affected, "total": total,
                            "rate_pct": 100.0 * affected / total if total else 0.0,
                            "scope": SCOPES[name], "measurement": "exact"}
        if reset:
            self.counts.clear()
        return result


def merge_counts(target, incoming):
    for name, row in incoming.items():
        previous = target.setdefault(name, dict(row, wrapped=0, total=0, rate_pct=0.0))
        previous["wrapped"] += row["wrapped"]
        previous["total"] += row["total"]
        previous["rate_pct"] = 100.0 * previous["wrapped"] / previous["total"] if previous["total"] else 0.0
