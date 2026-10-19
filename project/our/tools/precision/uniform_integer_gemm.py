"""Exact integer GEMMs for uniform fixed-point conv/linear operators.

Signed codes use base-128 digits. All low digits are nonnegative 7-bit values;
the highest digit is signed. CUDA int8 GEMMs produce exact int32 limb dots,
whose weighted terms are reconstructed in checked wide integer registers by
default, then floor-rescaled once and saturated to ell-bit stored codes.
Explicit ring/wrap mode reduces weighted terms and accumulation modulo
2**ell before signed rescaling; discarded high bits never reach output.
The native callback always receives the selected ring and wrap flags.

Weight digits are cached buffers. Input digits use the least number required
by their actual signed range. Register bounds cover both original MAC prefixes
and every limb-reconstruction prefix; they are mathematical upper bounds,
not observed prefix peaks. No floating-point operation occurs here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_DIGIT_BITS = 7
_DIGIT_MASK = 127
_MAX_DOT_BYTES = 256 * 1024 * 1024
_MAX_INT32_K = ((1 << 31) - 1) // (128 * 128)


def _pair(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _bounds(codes):
    if codes.dtype != torch.int64 or codes.numel() == 0:
        raise ValueError("integer GEMM requires nonempty int64 code tensors")
    lo, hi = torch.aminmax(codes)
    low, high = torch.stack((lo, hi)).tolist()
    return int(low), int(high)


def _limb_count(low, high):
    width = 1 + max(high, -low - 1, 0).bit_length()
    if width > 32:
        raise ValueError("integer GEMM supports at most signed 32-bit stored codes")
    return max(1, (width + _DIGIT_BITS - 1) // _DIGIT_BITS)


def _weighted_digit_abs_bound(low, high, count):
    highest_shift = _DIGIT_BITS * (count - 1)
    highest = max(abs(low >> highest_shift), abs(high >> highest_shift))
    # Each lower digit is <=127, whose weighted geometric sum is 2^shift-1.
    return (1 << highest_shift) - 1 + (highest << highest_shift)


def _pack_left(matrix, count):
    rows, columns = matrix.shape
    padded_rows = max(32, (rows + 15) // 16 * 16)
    padded_columns = (columns + 31) // 32 * 32
    limbs = []
    for index in range(count):
        shifted = matrix >> (_DIGIT_BITS * index)
        digit = shifted if index == count - 1 else shifted & _DIGIT_MASK
        packed = digit.to(torch.int8).contiguous()
        if padded_rows != rows or padded_columns != columns:
            packed = F.pad(packed, (0, padded_columns - columns, 0, padded_rows - rows))
        limbs.append(packed)
    return limbs


class CachedIntegerGemmWeights(nn.Module):
    """Cache packed [K,N] weight digits as movable module buffers."""

    def __init__(self, weight_q):
        super().__init__()
        if weight_q.dtype != torch.int64 or weight_q.dim() not in (2, 4):
            raise ValueError("cached weights require int64 [N,K] or [N,C,kh,kw]")
        self.weight_shape = tuple(weight_q.shape)
        self.low, self.high = _bounds(weight_q)
        self.limb_count = _limb_count(self.low, self.high)
        self.abs_bound = max(abs(self.low), abs(self.high))
        self.weighted_digit_abs_bound = _weighted_digit_abs_bound(self.low, self.high, self.limb_count)
        matrix = weight_q.flatten(1).T
        self.k, self.n = matrix.shape
        if self.k > _MAX_INT32_K:
            raise ValueError("reduction is too long to prove each int32 limb dot safe")
        self.padded_k = (self.k + 31) // 32 * 32
        self.padded_n = (self.n + 7) // 8 * 8
        for index in range(self.limb_count):
            shifted = matrix >> (_DIGIT_BITS * index)
            digit = shifted if index == self.limb_count - 1 else shifted & _DIGIT_MASK
            packed = digit.to(torch.int8).contiguous()
            if self.padded_k != self.k or self.padded_n != self.n:
                packed = F.pad(packed, (0, self.padded_n - self.n, 0, self.padded_k - self.k))
            self.register_buffer("limb_%d" % index, packed)
        for input_count in range(1, 6):
            shifts = [_DIGIT_BITS * (i + j) for i in range(input_count) for j in range(self.limb_count)]
            self.register_buffer("shifts_%d" % input_count,
                                 torch.tensor(shifts, dtype=torch.int64, device=weight_q.device))

    @property
    def limbs(self):
        return [getattr(self, "limb_%d" % index) for index in range(self.limb_count)]


def integer_im2col(codes, kernel, stride, padding, dilation):
    """NCHW integer tensor -> [N*Hout*Wout,C*kh*kw] integer matrix."""
    kh, kw = _pair(kernel)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    if min(kh, kw, sh, sw, dh, dw) <= 0 or min(ph, pw) < 0:
        raise ValueError("invalid convolution geometry")
    padded = F.pad(codes, (pw, pw, ph, ph)) if ph or pw else codes
    patches = padded.unfold(2, dh * (kh - 1) + 1, sh).unfold(3, dw * (kw - 1) + 1, sw)
    patches = patches[..., ::dh, ::dw]
    batch, channels, oh, ow, _, _ = patches.shape
    matrix = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch * oh * ow, channels * kh * kw)
    return matrix, (batch, oh, ow)


def _gemm_matrix(matrix, weights, bias_q, ell, s, register_bits,
                 combine_callback, input_bounds=None, max_dot_bytes=_MAX_DOT_BYTES,
                 arithmetic_mode="wide", overflow_mode="saturate"):
    ring, wrap = arithmetic_mode == "ring", overflow_mode == "wrap"
    if arithmetic_mode not in {"ring", "wide"} or overflow_mode not in {"wrap", "saturate"} or (ring and not wrap):
        raise ValueError("invalid integer GEMM arithmetic policy")
    if matrix.dtype != torch.int64 or matrix.dim() != 2 or matrix.shape[1] != weights.k:
        raise ValueError("input integer matrix does not match the cached weights")
    if bias_q.dtype != torch.int64 or bias_q.numel() not in (0, weights.n):
        raise ValueError("bias must contain same-scale int64 output-channel codes")
    if not 2 <= ell <= 32 or not 0 <= s < ell or not 2 <= register_bits <= 128:
        raise ValueError("invalid uniform fixed-point format")
    wlimbs = weights.limbs
    if matrix.device != wlimbs[0].device or matrix.device != bias_q.device:
        raise ValueError("input, cached weight limbs and bias must share a device")
    low, high = _bounds(matrix) if input_bounds is None else input_bounds
    input_count = _limb_count(low, high)
    input_abs = max(abs(low), abs(high))
    bias_abs = int(bias_q.abs().amax().item()) if bias_q.numel() else 0
    lifted_bias_bound = bias_abs << s
    original_mac_bound = weights.k * input_abs * weights.abs_bound + lifted_bias_bound
    reassembly_bound = (weights.k * _weighted_digit_abs_bound(low, high, input_count)
                        * weights.weighted_digit_abs_bound + lifted_bias_bound)
    proof_bits = 1 + max(original_mac_bound, reassembly_bound).bit_length()
    # Limb kernels' exact int32 dots remain proven safe independently. Their
    # physical carrier bound cannot widen the logical ring accumulator.
    if proof_bits > (128 if ring else register_bits):
        raise OverflowError("register overflow: absolute bounds cannot prove requested temporary width")
    pairs = input_count * weights.limb_count
    bytes_per_row = pairs * weights.padded_n * 4
    chunk_rows = max(32, int(max_dot_bytes) // max(1, bytes_per_row) // 16 * 16)
    shifts = getattr(weights, "shifts_%d" % input_count)
    outputs, stats = [], []
    for start in range(0, matrix.shape[0], chunk_rows):
        current = matrix[start:start + chunk_rows]
        alimbs = _pack_left(current, input_count)
        dots = torch.empty((pairs, alimbs[0].shape[0], weights.padded_n),
                           dtype=torch.int32, device=matrix.device)
        pair = 0
        for left in alimbs:
            for right in wlimbs:
                torch._int_mm(left, right, out=dots[pair])
                pair += 1
        # Exclude padded rows/channels from bias application and observed
        # writeback statistics. Full chunks normally need no extra copy.
        actual = dots[:, :current.shape[0], :weights.n].contiguous()
        output, status = combine_callback(actual, shifts, bias_q.contiguous(),
                                          ell, s, register_bits, proof_bits, ring, wrap)
        outputs.append(output)
        stats.append(status)
    all_stats = torch.stack(stats)
    combined_stats = torch.cat((all_stats[:, :2].sum(0), all_stats[:, 2:].amax(0)))
    return torch.cat(outputs, dim=0), combined_stats


def conv2d_gemm(codes, packed_weights, bias_q, stride, padding, dilation,
                ell, s, register_bits, combine_callback,
                arithmetic_mode="wide", overflow_mode="saturate"):
    """Exact uniform conv, returning NCHW int64 output and four native stats."""
    if codes.dim() != 4 or len(packed_weights.weight_shape) != 4:
        raise ValueError("conv2d GEMM expects NCHW input and convolution weights")
    _, channels, kh, kw = packed_weights.weight_shape
    if codes.shape[1] != channels:
        raise ValueError("convolution input channels do not match cached weights")
    bounds = _bounds(codes)
    matrix, (batch, oh, ow) = integer_im2col(codes, (kh, kw), stride, padding, dilation)
    output, stats = _gemm_matrix(matrix, packed_weights, bias_q, ell, s,
                                 register_bits, combine_callback, input_bounds=bounds,
                                 arithmetic_mode=arithmetic_mode, overflow_mode=overflow_mode)
    output = output.reshape(batch, oh, ow, packed_weights.n).permute(0, 3, 1, 2).contiguous()
    return output, stats


def linear_gemm(codes, packed_weights, bias_q, ell, s, register_bits, combine_callback,
                arithmetic_mode="wide", overflow_mode="saturate"):
    """Exact uniform linear operator, preserving leading input dimensions."""
    if codes.dim() < 2 or len(packed_weights.weight_shape) != 2:
        raise ValueError("linear GEMM requires at least rank-two input and matrix weights")
    matrix = codes.reshape(-1, codes.shape[-1])
    output, stats = _gemm_matrix(matrix, packed_weights, bias_q, ell, s,
                                 register_bits, combine_callback,
                                 arithmetic_mode=arithmetic_mode, overflow_mode=overflow_mode)
    return output.reshape(*codes.shape[:-1], packed_weights.n), stats
