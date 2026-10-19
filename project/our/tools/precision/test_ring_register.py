"""Tests for the physical-register ring emulator (E0-R).

A value lives in the storage ring Z_{2^ell} at scale 2^s; a linear layer runs
in a register of exactly L bits (ell <= L <= 64).  Expected values are derived
by hand (see comments) or by the literal big-integer oracle below, which masks
the register to L bits after every product and every addition.
"""

import itertools
import random

import pytest
import torch

import ring_register  # later-cycle functions are looked up at call time
from ring_register import RingSpec, ring_linear

NO_BIAS = torch.empty(0, dtype=torch.int64)


def oracle_linear(x_rows, w_rows, bias, s, ell, L):
    """Literal L-bit register MACs, then floor-truncate and reduce to ell."""
    mask = (1 << L) - 1
    out = []
    for x_row in x_rows:
        row = []
        for j, w_row in enumerate(w_rows):
            acc = 0
            for x, w in zip(x_row, w_row):
                acc = (acc + (((x & mask) * (w & mask)) & mask)) & mask
            if bias:
                acc = (acc + (bias[j] & mask)) & mask
            signed = acc - (1 << L) if acc >> (L - 1) else acc
            q = (signed >> s) & ((1 << ell) - 1)  # Python >> is floor
            row.append(q - (1 << ell) if q >> (ell - 1) else q)
        out.append(row)
    return out


def min_signed_bits(value):
    """Smallest L with value in [-2^(L-1), 2^(L-1)), by direct search."""
    return next(L for L in range(1, 256)
                if -(1 << (L - 1)) <= value < (1 << (L - 1)))


def oracle_stats(x_rows, w_rows, bias, s, ell, L):
    overflow = wraps = max_bits = 0
    for x_row in x_rows:
        for j, w_row in enumerate(w_rows):
            exact = sum(x * w for x, w in zip(x_row, w_row))
            exact += bias[j] if bias else 0
            bits = min_signed_bits(exact)
            max_bits = max(max_bits, bits)
            overflow += bits > L
            reg = exact % (1 << L)
            signed = reg - (1 << L) if reg >> (L - 1) else reg
            wraps += min_signed_bits(signed >> s) > ell
    return [overflow, wraps, max_bits]


def run_linear(x, w, b, spec):
    y, stats = ring_linear(torch.tensor(x, dtype=torch.int64),
                           torch.tensor(w, dtype=torch.int64),
                           torch.tensor(b, dtype=torch.int64) if b else NO_BIAS,
                           spec)
    return y.tolist(), stats.tolist()


def test_q32_16_product_wraps_to_zero_in_a_32_bit_register():
    # 2.0 * 1.0 in Q(32,16): P = 2^17 * 2^16 = 2^33 == 0 (mod 2^32).
    y, (overflow, wraps, bits) = run_linear([[2 << 16]], [[1 << 16]], None,
                                            RingSpec(ell=32, s=16, L=32))
    assert y == [[0]]
    assert overflow == 1
    assert bits == 35  # 2^33 needs 34 magnitude bits plus a sign bit


def test_same_product_is_exact_in_a_48_bit_register():
    y, (overflow, wraps, bits) = run_linear([[2 << 16]], [[1 << 16]], None,
                                            RingSpec(ell=32, s=16, L=48))
    assert y == [[2 << 16]]
    assert (overflow, wraps, bits) == (0, 0, 35)


def test_partial_sum_wrap_cancels_when_final_sum_fits():
    # 8-bit register: 100 + 100 wraps to -56, then -56 - 100 wraps back to
    # 100.  A saturating register would give 127 - 100 = 27.
    y, (overflow, _, _) = run_linear([[100, 100, -100]], [[1, 1, 1]], None,
                                     RingSpec(ell=8, s=0, L=8))
    assert y == [[100]]
    assert overflow == 0  # only the final sum's magnitude matters


def test_final_sum_wraps_modulo_register_width():
    # 100 + 100 = 200 == -56 (mod 2^8).
    y, (overflow, _, bits) = run_linear([[100, 100]], [[1, 1]], None,
                                        RingSpec(ell=8, s=0, L=8))
    assert y == [[-56]]
    assert (overflow, bits) == (1, 9)


def test_truncation_floors_negative_accumulators():
    # floor(-7 / 4) = -2; truncation toward zero would give -1.
    y, _ = run_linear([[-7]], [[1]], None, RingSpec(ell=16, s=2, L=16))
    assert y == [[-2]]


def test_truncate_reduce_wraps_into_the_storage_ring():
    # P = 100 * 32 = 3200 fits a 16-bit register; floor(3200 / 16) = 200,
    # which is -56 in Z_{2^8}.
    y, (overflow, wraps, _) = run_linear([[100]], [[32]], None,
                                         RingSpec(ell=8, s=4, L=16))
    assert y == [[-56]]
    assert (overflow, wraps) == (0, 1)


def test_bias_is_added_at_double_scale_before_truncation():
    # P = 1 * 8 + 8 = 16 -> floor(16 / 16) = 1.  Adding the bias after the
    # truncation instead would give floor(8 / 16) + 8 = 8.
    y, _ = run_linear([[1]], [[8]], [8], RingSpec(ell=16, s=4, L=32))
    assert y == [[1]]


@pytest.mark.parametrize("ell,s,L", [(12, 5, 12), (20, 8, 20), (32, 10, 32),
                                     (32, 10, 42), (32, 16, 48), (32, 12, 64)])
@pytest.mark.parametrize("magnitude", ["full_ring", "realistic"])
def test_linear_matches_literal_register_oracle(ell, s, L, magnitude):
    rng = random.Random(1000 * L + 10 * s + len(magnitude))
    if magnitude == "full_ring":
        x_bound, w_bound = 1 << (ell - 1), 1 << (L - 1)
    else:
        x_bound, w_bound = 1 << min(s + 3, ell - 1), 1 << min(s + 1, L - 1)
    x = [[rng.randrange(-x_bound, x_bound) for _ in range(37)] for _ in range(3)]
    w = [[rng.randrange(-w_bound, w_bound) for _ in range(37)] for _ in range(5)]
    b = [rng.randrange(-w_bound, w_bound) for _ in range(5)]
    y, stats = run_linear(x, w, b, RingSpec(ell=ell, s=s, L=L))
    assert y == oracle_linear(x, w, b, s, ell, L)
    assert stats == oracle_stats(x, w, b, s, ell, L)


def test_spec_rejects_register_narrower_than_storage():
    with pytest.raises(ValueError):
        RingSpec(ell=32, s=10, L=31)


def im2col(x, k, stride, padding):
    """Zero-padded k*k patches: rows in (n, oh, ow), columns in (c, r, q)."""
    n_, c_, h_, w_ = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
    ho = (h_ + 2 * padding - k) // stride + 1
    wo = (w_ + 2 * padding - k) // stride + 1
    rows = []
    for n, oh, ow in itertools.product(range(n_), range(ho), range(wo)):
        row = []
        for c, r, q in itertools.product(range(c_), range(k), range(k)):
            ih, iw = oh * stride - padding + r, ow * stride - padding + q
            row.append(x[n][c][ih][iw] if 0 <= ih < h_ and 0 <= iw < w_ else 0)
        rows.append(row)
    return rows


def test_conv_reduces_the_window_sum_modulo_the_register():
    # 1 + 2 + 3 + 4 = 10 = 0b1010, which is -6 in a 4-bit register.
    y, stats = ring_register.ring_conv2d(
        torch.tensor([[[[1, 2], [3, 4]]]]), torch.ones((1, 1, 2, 2), dtype=torch.int64),
        NO_BIAS, RingSpec(ell=4, s=0, L=4))
    assert y.tolist() == [[[[-6]]]]
    assert stats.tolist() == [1, 0, 5]


@pytest.mark.parametrize("ell,s,L", [(12, 5, 12), (32, 10, 32), (32, 10, 42),
                                     (32, 12, 64)])
def test_conv_matches_literal_register_oracle(ell, s, L):
    rng = random.Random(7 * L + s)
    xb, wb = 1 << min(s + 3, ell - 1), 1 << min(s + 1, L - 1)
    x = [[[[rng.randrange(-xb, xb) for _ in range(5)] for _ in range(5)]
          for _ in range(3)] for _ in range(2)]
    w = [[[[rng.randrange(-wb, wb) for _ in range(3)] for _ in range(3)]
          for _ in range(3)] for _ in range(4)]
    b = [rng.randrange(-wb, wb) for _ in range(4)]
    y, stats = ring_register.ring_conv2d(
        torch.tensor(x), torch.tensor(w), torch.tensor(b),
        RingSpec(ell=ell, s=s, L=L), stride=2, padding=1)
    rows = im2col(x, 3, stride=2, padding=1)
    w_rows = [[v for channel in o for line in channel for v in line] for o in w]
    assert y.shape == (2, 4, 3, 3)
    assert y.permute(0, 2, 3, 1).reshape(-1, 4).tolist() == \
        oracle_linear(rows, w_rows, b, s, ell, L)
    assert stats.tolist() == oracle_stats(rows, w_rows, b, s, ell, L)


def test_channel_affine_truncates_after_adding_the_offset():
    # Q(8,2): 0.75 * 1.5 + 0.25 -> P = 3*6 + 4 = 22, floor(22 / 4) = 5 (1.25).
    y, stats = ring_register.ring_channel_affine(
        torch.tensor([[[[3]]]]), torch.tensor([6]), torch.tensor([4]),
        RingSpec(ell=8, s=2, L=8))
    assert y.tolist() == [[[[5]]]]
    assert stats.tolist() == [0, 0, 6]


@pytest.mark.parametrize("ell,s,L", [(12, 5, 12), (32, 10, 32), (32, 16, 48)])
def test_channel_affine_matches_literal_register_oracle(ell, s, L):
    rng = random.Random(11 * L + s)
    xb, mb = 1 << min(s + 3, ell - 1), 1 << min(s + 2, L - 1)
    ob = 1 << min(2 * s + 2, L - 1)
    x = [[[[rng.randrange(-xb, xb) for _ in range(3)] for _ in range(2)]
          for _ in range(4)] for _ in range(2)]
    mult = [rng.randrange(-mb, mb) for _ in range(4)]
    offset = [rng.randrange(-ob, ob) for _ in range(4)]
    y, stats = ring_register.ring_channel_affine(
        torch.tensor(x), torch.tensor(mult), torch.tensor(offset),
        RingSpec(ell=ell, s=s, L=L))
    want, counts = [], [0, 0, 0]
    for n, c, h, q in itertools.product(range(2), range(4), range(2), range(3)):
        args = ([[x[n][c][h][q]]], [[mult[c]]], [offset[c]], s, ell, L)
        want.append(oracle_linear(*args)[0][0])
        overflow, wraps, bits = oracle_stats(*args)
        counts = [counts[0] + overflow, counts[1] + wraps, max(counts[2], bits)]
    assert y.flatten().tolist() == want
    assert stats.tolist() == counts


CUDA_CASES = [(12, 5, 12, "realistic"), (32, 10, 32, "realistic"),
              (32, 10, 42, "realistic"), (32, 12, 64, "full_ring")]
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def random_codes(shape, bits, generator):
    bound = 1 << min(bits, 62)
    return torch.randint(-bound, bound, shape, generator=generator, dtype=torch.int64)


def assert_same_on_cuda(fn, *tensors, **kwargs):
    y_cpu, st_cpu = fn(*tensors, **kwargs)
    y_gpu, st_gpu = fn(*(t.cuda() if torch.is_tensor(t) else t for t in tensors),
                       **kwargs)
    assert y_gpu.is_cuda
    assert torch.equal(y_gpu.cpu(), y_cpu)
    assert st_gpu.cpu().tolist() == st_cpu.tolist()
    return st_cpu.tolist()


@cuda_only
@pytest.mark.parametrize("ell,s,L,magnitude", CUDA_CASES)
@pytest.mark.parametrize("k,stride,padding", [(3, 1, 1), (3, 2, 1), (1, 2, 0), (7, 2, 3)])
def test_cuda_conv_is_bit_identical_to_cpu(ell, s, L, magnitude, k, stride, padding):
    g = torch.Generator().manual_seed(L * 100 + s * 10 + k)
    xbits, wbits = ((ell - 1, L - 1) if magnitude == "full_ring"
                    else (min(s + 4, ell - 1), s + 1))
    x = random_codes((3, 8, 13, 13), xbits, g)
    w = random_codes((10, 8, k, k), wbits, g)
    b = random_codes((10,), wbits, g)
    spec = RingSpec(ell=ell, s=s, L=L)
    stats = assert_same_on_cuda(ring_register.ring_conv2d, x, w, b, spec,
                                stride=stride, padding=padding)
    assert stats[2] > 0


@cuda_only
@pytest.mark.parametrize("ell,s,L,magnitude", CUDA_CASES)
def test_cuda_linear_and_affine_are_bit_identical_to_cpu(ell, s, L, magnitude):
    g = torch.Generator().manual_seed(L * 10 + s)
    xbits, wbits = ((ell - 1, L - 1) if magnitude == "full_ring"
                    else (min(s + 4, ell - 1), s + 1))
    spec = RingSpec(ell=ell, s=s, L=L)
    assert_same_on_cuda(ring_linear, random_codes((33, 300), xbits, g),
                        random_codes((17, 300), wbits, g),
                        random_codes((17,), wbits, g), spec)
    assert_same_on_cuda(ring_register.ring_channel_affine,
                        random_codes((5, 6, 9, 9), xbits, g),
                        random_codes((6,), wbits, g),
                        random_codes((6,), min(2 * s + 2, L - 1), g), spec)
