"""Physical-register ring emulator (E0-R) for fixed-point MPC inference.

Storage values are ell-bit two's-complement codes (signed representatives of
Z_{2^ell}) at scale 2^s.  A linear layer lifts its inputs into a register of
exactly L bits, multiplies and accumulates modulo 2^L -- the register is
masked to L bits after every product and every addition -- then
floor-truncates by s bits and reduces into Z_{2^ell}.  Nothing wider than L
bits influences an output.

For diagnostics only, the native kernels also keep the exact (unreduced)
accumulator in an int128.  It never feeds an output; it yields the per-call
statistics ``[register_overflows, storage_wraps, max_exact_bits]``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_NATIVE = None


def _native():
    """Compile/load the register kernels lazily."""
    global _NATIVE
    if _NATIVE is None:
        from torch.utils.cpp_extension import load

        # One compiler at a time: the development box has ~2 GB of free RAM.
        os.environ.setdefault("MAX_JOBS", "1")
        native = Path(__file__).resolve().parent / "native"
        sources = [str(native / "ring_register.cpp")]
        cflags = ["-O3"]
        if torch.cuda.is_available():
            sources.append(str(native / "ring_register_cuda.cu"))
            cflags.append("-DRING_WITH_CUDA")
        _NATIVE = load(
            name="e0_ring_register_ops_v1",
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _NATIVE


class RingSpec:
    """Storage ring Z_{2^ell} at scale 2^s with an L-bit linear register."""

    def __init__(self, ell: int, s: int, L: int | None = None):
        L = ell if L is None else L
        if not all(isinstance(value, int) for value in (ell, s, L)):
            raise TypeError("ell, s and L must be integers")
        # ell <= 32 keeps storage arithmetic exact in int64 and matches the
        # 32-bit FSS comparison domain; wider rings only appear as registers.
        if not 2 <= ell <= 32:
            raise ValueError("storage width ell must lie in [2, 32]")
        if not ell <= L <= 64:
            raise ValueError("register width L must satisfy ell <= L <= 64")
        if not 0 <= s < ell:
            raise ValueError("fractional bits s must satisfy 0 <= s < ell")
        self.ell, self.s, self.L = ell, s, L


def ring_linear(x_q: torch.Tensor, weight_q: torch.Tensor, bias_q: torch.Tensor,
                spec: RingSpec):
    """``TR_s(x W^T + b)`` in an L-bit register; returns ``(y, stats)``.

    ``bias_q`` is at scale 2^(2s) (it joins the accumulator before the
    truncation); pass an empty tensor for no bias.
    """
    return _native().linear(x_q.contiguous(), weight_q.contiguous(),
                            bias_q.contiguous(), spec.ell, spec.s, spec.L)


def ring_conv2d(x_q: torch.Tensor, weight_q: torch.Tensor, bias_q: torch.Tensor,
                spec: RingSpec, stride: int = 1, padding: int = 0):
    """Zero-padded 2-D convolution with the ``ring_linear`` register rules."""
    return _native().conv2d(x_q.contiguous(), weight_q.contiguous(),
                            bias_q.contiguous(), int(stride), int(padding),
                            spec.ell, spec.s, spec.L)


def ring_channel_affine(x_q: torch.Tensor, mult_q: torch.Tensor,
                        offset_q: torch.Tensor, spec: RingSpec):
    """Per-channel ``TR_s(x * mult + offset)`` (evaluation BatchNorm).

    ``mult_q`` is at scale 2^s and ``offset_q`` at scale 2^(2s).
    """
    return _native().channel_affine(x_q.contiguous(), mult_q.contiguous(),
                                    offset_q.contiguous(), spec.ell, spec.s,
                                    spec.L)
