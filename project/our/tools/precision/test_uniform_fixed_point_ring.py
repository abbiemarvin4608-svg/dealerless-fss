"""Literal modulo-2**ell oracles for the explicit uniform-ring control.

These tests deliberately force multiplication, MAC-prefix, bias-lift, input,
and pooling overflows.  They distinguish strict ring arithmetic before
rescaling from both saturation and a wide-MAC/output-wrap implementation.
The test oracles use Python integers and mask every product and addition.
"""
import copy
import itertools
import random

import pytest
import torch
import torch.nn as nn

from uniform_fixed_point import (
    UniformAdaptiveAvgPool2d, UniformBatchNorm2d, UniformConv2d,
    UniformFixedPoint as _UniformFixedPoint, UniformLinear, UniformMaxPool2d,
    _native_ops as _raw_native_ops, convert_resnet_to_uniform as _convert_resnet_to_uniform,
)


def UniformFixedPoint(*args, **kwargs):
    kwargs.setdefault("arithmetic_mode", "ring")
    kwargs.setdefault("overflow_mode", "wrap")
    return _UniformFixedPoint(*args, **kwargs)


def convert_resnet_to_uniform(*args, **kwargs):
    kwargs.setdefault("arithmetic_mode", "ring")
    kwargs.setdefault("overflow_mode", "wrap")
    return _convert_resnet_to_uniform(*args, **kwargs)


class _RingNative:
    def __getattr__(self, name):
        function = getattr(_raw_native_ops(), name)

        def ring_call(*args, **kwargs):
            kwargs.setdefault("ring", True)
            kwargs.setdefault("wrap", True)
            return function(*args, **kwargs)

        return ring_call


def _native_ops():
    return _RingNative()


def signed_ring(value, ell):
    residue = int(value) & ((1 << ell) - 1)
    return residue - (1 << ell) if residue & (1 << (ell - 1)) else residue


def ring_mac(row, weight, bias, ell, s):
    acc = 0
    for left, right in zip(row, weight):
        product = signed_ring(signed_ring(left, ell) * signed_ring(right, ell), ell)
        acc = signed_ring(acc + product, ell)
    lifted_bias = signed_ring(signed_ring(bias, ell) << s, ell)
    acc = signed_ring(acc + lifted_bias, ell)
    return signed_ring(acc // (1 << s), ell)


def linear_oracle(x, w, b, ell, s):
    bias = b.tolist() if b.numel() else [0] * w.shape[0]
    return torch.tensor([[ring_mac(row, weight, offset, ell, s)
                          for weight, offset in zip(w.tolist(), bias)]
                         for row in x.tolist()], dtype=torch.int64)


def assert_ring_tensor(codes, ell):
    assert codes.dtype == torch.int64
    assert bool((codes >= -(1 << (ell - 1))).all())
    assert bool((codes < (1 << (ell - 1))).all())


def assert_ring_stats(stats, ell):
    # Wider diagnostic arithmetic never changes the logical register width.
    assert int(stats[1]) == 0
    assert int(stats[2]) <= ell


def test_default_wide_storage_and_explicit_ring_control_are_distinct():
    default = _UniformFixedPoint(4, 2)
    assert default.arithmetic_mode == "wide"
    assert default.overflow_mode == "saturate"
    assert default.register_bits == 128
    assert default.encode(torch.tensor([8], dtype=torch.int64)).item() == 7
    fp = UniformFixedPoint(4, 2)
    assert fp.arithmetic_mode == "ring"
    assert fp.overflow_mode == "wrap"
    assert fp.register_bits == 4
    assert fp.encode(torch.tensor([8], dtype=torch.int64)).item() == -8


@pytest.mark.parametrize("ell,s", [(4, 2), (16, 12), (32, 20)])
def test_encode_reduces_parameters_and_inputs_instead_of_saturating(ell, s):
    fp = UniformFixedPoint(ell, s)
    period, half = 1 << ell, 1 << (ell - 1)
    raw = torch.tensor([-3*period-1, -half-1, -half, -1, 0, half-1, half,
                        3*period+1], dtype=torch.int64)
    expected = [signed_ring(int(code), ell) for code in raw]
    for kind in ("input", "parameter"):
        assert fp.encode(raw, kind).tolist() == expected
        values = raw.to(torch.float64) / (1 << s)
        encoded = fp.encode(values, kind)
        assert encoded.tolist() == expected
        assert_ring_tensor(encoded, ell)
    # The same finite range is enforced when statistics are disabled.
    quiet = UniformFixedPoint(ell, s, track_stats=False)
    assert quiet.encode(raw).tolist() == expected


def test_float_encoding_floors_before_ring_reduction():
    fp = UniformFixedPoint(4, 2)
    values = torch.tensor([-4.1, -2.1, -.1, .3, 2., 4.1], dtype=torch.float64)
    assert fp.encode(values).tolist() == [-1, 7, -1, 1, -8, 0]


def test_writeback_and_residual_add_wrap_positive_and_negative_overflow():
    fp = UniformFixedPoint(4, 0)
    assert fp._writeback(torch.tensor([15, -17, 32, -32])).tolist() == [-1, -1, 0, 0]
    assert fp.add(torch.tensor([6, -6]), torch.tensor([6, -6])).tolist() == [-4, 4]
    stats = fp.stats()
    assert stats["register_overflows"] == 0
    assert stats["max_register_bits"] <= 4
    assert stats["activation_wrapped"] >= 2


def test_product_is_reduced_before_rescaling_not_only_at_output():
    source = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        source.weight.fill_(7/4)
    fp = UniformFixedPoint(4, 2)
    layer = UniformLinear(source, fp)
    result = layer(torch.tensor([[7]], dtype=torch.int64))
    # 49 mod 16 = 1; signed floor(1/4) = 0.  Wide MAC then output
    # wrap would produce wrap4(floor(49/4)) = -4, which must fail.
    assert result.item() == 0
    assert signed_ring((7*7)//4, 4) == -4
    assert fp.stats()["activation_wrapped"] == 1


def test_lifted_bias_wraps_before_truncation_and_is_stored_at_same_scale():
    source = nn.Linear(1, 1).double()
    with torch.no_grad():
        source.weight.zero_()
        source.bias.fill_(7/4)
    fp = UniformFixedPoint(4, 2)
    layer = UniformLinear(source, fp)
    assert layer.bias_q.tolist() == [7]
    # Lift7<<2=28 => -4 modulo16 => -1 after floor(/4).
    assert layer(torch.tensor([[0]], dtype=torch.int64)).item() == -1
    assert fp.stats()["activation_wrapped"] == 1


def test_prefix_wrap_is_recorded_even_when_the_final_sum_cancels():
    source = nn.Linear(3, 1, bias=False).double()
    with torch.no_grad():
        source.weight.fill_(1)
    fp = UniformFixedPoint(4, 0)
    result = UniformLinear(source, fp)(torch.tensor([[6, 6, -6]], dtype=torch.int64))
    assert result.item() == 6
    assert fp.stats()["activation_wrapped"] == 1
    assert fp.stats()["max_register_bits"] <= 4


@pytest.mark.parametrize("ell,s", [(4, 0), (4, 2), (16, 12), (24, 15), (32, 20)])
def test_native_linear_matches_literal_per_product_and_add_ring_oracle(ell, s):
    rng = random.Random(ell*100+s)
    half = 1 << (ell-1)
    x = torch.tensor([[rng.randrange(-half, half) for _ in range(39)]
                      for _ in range(3)], dtype=torch.int64)
    w = torch.tensor([[rng.randrange(-half, half) for _ in range(39)]
                      for _ in range(5)], dtype=torch.int64)
    b = torch.tensor([rng.randrange(-half, half) for _ in range(5)], dtype=torch.int64)
    result, stats = _native_ops().linear(x, w, b, ell, s, ell)
    assert torch.equal(result, linear_oracle(x, w, b, ell, s))
    assert_ring_tensor(result, ell)
    assert_ring_stats(stats, ell)


def test_native_kernel_masks_noncanonical_int64_inputs_into_the_ring():
    x = torch.tensor([[23, -25, 1000]], dtype=torch.int64)
    w = torch.tensor([[35, 1001, -1001]], dtype=torch.int64)
    b = torch.tensor([1000], dtype=torch.int64)
    result, stats = _native_ops().linear(x, w, b, 4, 2, 4)
    assert torch.equal(result, linear_oracle(x, w, b, 4, 2))
    assert_ring_tensor(result, 4)
    assert_ring_stats(stats, 4)


def test_channel_affine_reduces_product_and_offset_lift_before_rescale():
    x = torch.tensor([[[[7, -7]]]], dtype=torch.int64)
    mult = torch.tensor([3], dtype=torch.int64)
    bias = torch.tensor([7], dtype=torch.int64)
    result, stats = _native_ops().channel_affine(x, mult, bias, 4, 2, 4)
    expected = [ring_mac([v], [3], 7, 4, 2) for v in (7, -7)]
    assert result.flatten().tolist() == expected
    assert_ring_stats(stats, 4)


def test_average_pool_sum_wraps_before_division():
    fp = UniformFixedPoint(4, 0)
    pool = UniformAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d(1), fp)
    # Sum24 => signed_ring24=-8; floor(-8/4)=-2. Wide sum would be6.
    assert pool(torch.full((1, 1, 2, 2), 6, dtype=torch.int64)).item() == -2
    assert fp.stats()["activation_wrapped"] == 1


def test_average_pool_tracks_wrapped_prefix_even_when_sum_is_zero():
    fp = UniformFixedPoint(4, 0)
    pool = UniformAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d(1), fp)
    result = pool(torch.tensor([[[[6, 6], [-6, -6]]]], dtype=torch.int64))
    assert result.item() == 0
    assert fp.stats()["activation_wrapped"] == 1


def conv_oracle(x, w, b, stride, padding, dilation, ell, s):
    n, cin, h, width = x.shape
    cout, _, kh, kw = w.shape
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    ho = (h+2*ph-dh*(kh-1)-1)//sh+1
    wo = (width+2*pw-dw*(kw-1)-1)//sw+1
    result = torch.empty((n, cout, ho, wo), dtype=torch.int64)
    for batch, oc, oh, ow in itertools.product(range(n), range(cout), range(ho), range(wo)):
        row, weight = [], []
        for ic, r, col in itertools.product(range(cin), range(kh), range(kw)):
            ih, iw = oh*sh-ph+r*dh, ow*sw-pw+col*dw
            if 0 <= ih < h and 0 <= iw < width:
                row.append(int(x[batch, ic, ih, iw]))
                weight.append(int(w[oc, ic, r, col]))
        result[batch, oc, oh, ow] = ring_mac(row, weight, int(b[oc]), ell, s)
    return result


@pytest.mark.parametrize("ell,s", [(4, 2), (32, 12)])
def test_conv_geometry_and_ring_mac_match_independent_oracle(ell, s):
    generator = torch.Generator().manual_seed(ell+s)
    half = 1 << (ell-1)
    x = torch.randint(-half, half, (2, 2, 5, 6), generator=generator, dtype=torch.int64)
    w = torch.randint(-half, half, (3, 2, 2, 2), generator=generator, dtype=torch.int64)
    b = torch.randint(-half, half, (3,), generator=generator, dtype=torch.int64)
    geometry = ((2, 1), (1, 0), (1, 2))
    result, stats = _native_ops().conv2d(x, w, b, *geometry[0], *geometry[1], *geometry[2], ell, s, ell)
    assert torch.equal(result, conv_oracle(x, w, b, *geometry, ell, s))
    assert_ring_stats(stats, ell)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("ell,s", [(4, 2), (32, 20)])
def test_native_cpu_cuda_and_fast64_ring_results_and_stats_match(ell, s):
    generator = torch.Generator().manual_seed(719+ell)
    half = 1 << (ell-1)
    x = torch.randint(-half, half, (3, 39), generator=generator, dtype=torch.int64)
    w = torch.randint(-half, half, (5, 39), generator=generator, dtype=torch.int64)
    b = torch.randint(-half, half, (5,), generator=generator, dtype=torch.int64)
    cpu, sc = _native_ops().linear(x, w, b, ell, s, ell)
    for fast in (False, True):
        cuda, sg = _native_ops().linear(x.cuda(), w.cuda(), b.cuda(), ell, s, ell, fast)
        assert torch.equal(cpu, cuda.cpu())
        assert torch.equal(sc, sg.cpu())


def limb_case():
    generator = torch.Generator().manual_seed(991)
    x = torch.randint(-(1 << 31), 1 << 31, (3, 37), generator=generator, dtype=torch.int64)
    w = torch.randint(-(1 << 31), 1 << 31, (4, 37), generator=generator, dtype=torch.int64)
    b = torch.tensor([11, -13, 17, -19], dtype=torch.int64)
    xd = [(x >> (7*i)) & 127 for i in range(4)] + [x >> 28]
    wd = [(w >> (7*i)) & 127 for i in range(4)] + [w >> 28]
    dots = torch.stack([(a @ c.t()).to(torch.int32) for a in xd for c in wd])
    shifts = torch.tensor([7*(i+j) for i in range(5) for j in range(5)], dtype=torch.int64)
    proof_bits = 1+(37*int(x.abs().max())*int(w.abs().max())+(int(b.abs().max()) << 20)).bit_length()
    assert proof_bits > 64
    return x, w, b, dots, shifts, proof_bits


def test_limb_combiner_discards_high_bits_before_rescale_even_beyond_int64():
    x, w, b, dots, shifts, proof_bits = limb_case()
    result, stats = _native_ops().combine_limb_dots(dots, shifts, b, 32, 20, 32, proof_bits)
    assert torch.equal(result, linear_oracle(x, w, b, 32, 20))
    assert_ring_stats(stats, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_limb_combiner_cpu_cuda_modulo_results_and_stats_match():
    _, _, b, dots, shifts, proof_bits = limb_case()
    cpu, sc = _native_ops().combine_limb_dots(dots, shifts, b, 32, 20, 32, proof_bits)
    cuda, sg = _native_ops().combine_limb_dots(dots.cuda(), shifts.cuda(), b.cuda(), 32, 20, 32, proof_bits)
    assert torch.equal(cpu, cuda.cpu())
    assert torch.equal(sc, sg.cpu())


def tiny_resnet():
    from torchvision.models.resnet import BasicBlock, ResNet
    torch.manual_seed(41)
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10).eval()


def check_network_boundaries(model, inputs, ell):
    # Hooks inspect every converted operator output, including block/residual
    # outputs; all stored parameter buffers are checked separately.
    def check(_module, _args, output):
        if torch.is_tensor(output):
            assert_ring_tensor(output, ell)
    handles = [module.register_forward_hook(check) for module in model.modules()]
    try:
        with torch.no_grad():
            return model(inputs)
    finally:
        for handle in handles:
            handle.remove()


def test_complete_small_network_keeps_all_parameter_and_output_codes_in_ring():
    ell, s = 4, 2
    model = convert_resnet_to_uniform(tiny_resnet(), ell, s)
    assert not list(model.parameters())
    for buffer in model.buffers():
        assert_ring_tensor(buffer, ell)
    # Values exceed entry range by many periods; every boundary must still be
    # canonical ell-bit code regardless of the int64 container.
    inputs = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(2))*100
    result = check_network_boundaries(model, inputs, ell)
    assert_ring_tensor(result, ell)
    stats = model.fp.stats()
    assert stats["input_wrapped"] > 0
    assert stats["activation_wrapped"] > 0
    assert stats["register_overflows"] == 0
    assert stats["max_register_bits"] <= ell


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("ell,s", [(4, 2), (16, 12), (32, 20)])
def test_full_small_network_cpu_cuda_scalar_gemm_strict_ring_parity(ell, s):
    source = tiny_resnet()
    cpu = convert_resnet_to_uniform(source, ell, s)
    scalar = convert_resnet_to_uniform(copy.deepcopy(source), ell, s).cuda()
    gemm = convert_resnet_to_uniform(copy.deepcopy(source), ell, s, integer_backend="gemm").cuda()
    inputs = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(2))*100
    reference = check_network_boundaries(cpu, inputs, ell)
    scalar_result = check_network_boundaries(scalar, inputs.cuda(), ell).cpu()
    gemm_result = check_network_boundaries(gemm, inputs.cuda(), ell).cpu()
    assert torch.equal(reference, scalar_result)
    assert torch.equal(reference, gemm_result)
    assert gemm.fp.stats()["gemm_calls"] > 0
    for fixed in (cpu, scalar, gemm):
        for module in fixed.modules():
            # Packed int8 limbs and shift metadata are backend scratch
            # representations; original model constants are these Q codes.
            for name, buffer in module.named_buffers(recurse=False):
                if name.endswith("_q"):
                    assert_ring_tensor(buffer, ell)
        assert fixed.fp.stats()["register_overflows"] == 0
        assert fixed.fp.stats()["max_register_bits"] <= ell
