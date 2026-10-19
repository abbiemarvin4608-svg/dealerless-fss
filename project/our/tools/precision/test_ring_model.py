"""Model-level E0-R tests: storage ops, layer modules and ResNet conversion."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet

from ring_model import (RingAdd, RingAvgPool, RingBatchNorm2d, RingContext, RingConv2d,
                        RingEncode, RingLinear, RingMaxPool2d, convert_resnet, wrap)
from ring_register import RingSpec


def test_wrap_maps_integers_to_signed_ring_representatives():
    x = torch.tensor([127, 128, -129, 300, -128])
    assert wrap(x, 8).tolist() == [127, -128, 127, 44, -128]


def test_encode_rounds_to_nearest_and_wraps_into_the_storage_ring():
    ctx = RingContext(RingSpec(ell=8, s=4))
    q = RingEncode(ctx, "input")(torch.tensor([0.5, 0.97, -0.9, -8.0, 8.0, 9.0]))
    # 15.52 -> 16 and -14.4 -> -14 (floor would give 15 and -15); 128, 144 wrap.
    assert q.tolist() == [8, 16, -14, -128, -128, -112]
    assert ctx.layer("input") == {"kind": "storage", "overflow": 0, "wraps": 2, "max_bits": 9}


def test_residual_add_wraps_in_the_storage_ring():
    ctx = RingContext(RingSpec(ell=8, s=4))
    y = RingAdd(ctx, "add")(torch.tensor([100, -100, 50]), torch.tensor([100, -100, 20]))
    assert y.tolist() == [-56, 56, 70]
    assert ctx.layer("add") == {"kind": "storage", "overflow": 0, "wraps": 2, "max_bits": 9}


@pytest.mark.parametrize("value,bits", [(0, 1), (-1, 1), (1, 2), (127, 8), (128, 9),
                                        (-128, 8), (-129, 9)])
def test_max_bits_is_the_exact_twos_complement_width(value, bits):
    ctx = RingContext(RingSpec(ell=16, s=4))
    RingAdd(ctx, "add")(torch.tensor([value]), torch.tensor([0]))
    assert ctx.layer("add")["max_bits"] == bits


def test_stats_accumulate_across_batches_and_reset_on_a_new_spec():
    ctx = RingContext(RingSpec(ell=8, s=4))
    add = RingAdd(ctx, "add")
    add(torch.tensor([100]), torch.tensor([100]))  # 200 wraps, 9 bits
    add(torch.tensor([1000]), torch.tensor([0]))   # wraps, 11 bits
    add(torch.tensor([1]), torch.tensor([1]))      # fits, 3 bits
    assert ctx.layer("add") == {"kind": "storage", "overflow": 0, "wraps": 2, "max_bits": 11}
    ctx.use(RingSpec(ell=12, s=4))
    assert ctx.layers() == []
    assert add(torch.tensor([100]), torch.tensor([100])).tolist() == [200]


def test_global_avgpool_floor_divides_the_wrapped_ring_sum():
    ctx = RingContext(RingSpec(ell=8, s=4))
    x = torch.tensor([[[[1, 2], [3, 5]], [[-1, -2], [-3, -5]], [[100, 100], [100, 100]]]])
    y = RingAvgPool(ctx, "avgpool")(x)
    # 11 // 4 = 2, -11 // 4 = -3, and 400 wraps to -112 before -112 // 4 = -28.
    assert y.shape == (1, 3, 1, 1)
    assert y.flatten().tolist() == [2, -3, -28]
    assert ctx.layer("avgpool") == {"kind": "storage", "overflow": 0, "wraps": 1,
                                    "max_bits": 10}


def test_maxpool_ignores_padding_even_for_negative_windows():
    pool = RingMaxPool2d(nn.MaxPool2d(3, stride=2, padding=1))
    negative = torch.full((1, 1, 3, 3), -5, dtype=torch.int64)
    assert pool(negative).tolist() == [[[[-5, -5], [-5, -5]]]]
    ramp = torch.arange(-4, 5).view(1, 1, 3, 3)
    expected = F.max_pool2d(ramp.double(), 3, stride=2, padding=1).long()
    assert torch.equal(pool(ramp), expected)


def dyadic(shape, low, high, s, generator):
    """Random multiples of 2^-s, so the fixed-point reference is exact."""
    return torch.randint(low, high, shape, generator=generator).double() / 2 ** s


def test_conv_module_is_exact_on_dyadic_values_with_bias_at_double_scale():
    g = torch.Generator().manual_seed(0)
    spec = RingSpec(ell=16, s=4, L=32)
    ctx = RingContext(spec)
    conv = nn.Conv2d(3, 5, 3, stride=2, padding=1, bias=True).double()
    with torch.no_grad():
        conv.weight.copy_(dyadic(conv.weight.shape, -24, 25, 4, g))
        conv.bias.copy_(dyadic(conv.bias.shape, -300, 301, 8, g))
    x = dyadic((2, 3, 7, 7), -40, 41, 4, g)
    y = RingConv2d(conv, ctx, "conv")((x * 16).long())
    assert torch.equal(y, torch.floor(conv(x) * 16).long())
    stats = ctx.layer("conv")
    assert stats["kind"] == "register" and stats["overflow"] == stats["wraps"] == 0


def test_linear_module_is_exact_on_dyadic_values_with_bias_at_double_scale():
    g = torch.Generator().manual_seed(1)
    ctx = RingContext(RingSpec(ell=16, s=4, L=32))
    fc = nn.Linear(20, 6).double()
    with torch.no_grad():
        fc.weight.copy_(dyadic(fc.weight.shape, -24, 25, 4, g))
        fc.bias.copy_(dyadic(fc.bias.shape, -300, 301, 8, g))
    x = dyadic((3, 20), -40, 41, 4, g)
    y = RingLinear(fc, ctx, "fc")((x * 16).long())
    assert torch.equal(y, torch.floor(fc(x) * 16).long())


def test_batchnorm_is_one_channel_affine_with_offset_at_double_scale():
    ctx = RingContext(RingSpec(ell=16, s=4, L=32))
    bn = nn.BatchNorm2d(2, eps=0.0).eval()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([0.75, -0.5]))
        bn.bias.copy_(torch.tensor([0.5, 2.0]))
        bn.running_mean.copy_(torch.tensor([0.3125, 0.0]))
        bn.running_var.copy_(torch.tensor([0.25, 1.0]))
    # mult = [1.5, -0.5] (scale 2^4), offset = [0.03125, 2.0] (scale 2^8):
    # (-40 * 24 + 8) / 16 = -59.5 floors to -60.
    x = torch.tensor([-2.5, 1.0, -0.5, 2.0]).view(1, 2, 2, 1)
    y = RingBatchNorm2d(bn, ctx, "bn")((x * 16).long())
    assert y.flatten().tolist() == [-60, 24, 36, 16]


def tiny_resnet(seed=0):
    """One bottleneck per stage with non-trivial BN statistics."""
    torch.manual_seed(seed)
    model = ResNet(Bottleneck, [1, 1, 1, 1], num_classes=10)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.2, 0.2)
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 1.5)
    return model.eval()


def run_ring(model, spec, x):
    ring = convert_resnet(model, spec)
    return ring(x), ring.ctx


def test_converted_resnet_tracks_float_logits_without_wraps():
    model, x = tiny_resnet(), torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    codes, ctx = run_ring(model, RingSpec(ell=32, s=12, L=64), x)
    totals = ctx.totals()
    assert totals["register_overflows"] == totals["storage_wraps"] == 0
    with torch.no_grad():
        expected = model(x).double()
    assert torch.allclose(codes.double() / 2 ** 12, expected, rtol=0, atol=0.02)
    assert torch.equal(codes.argmax(1), expected.argmax(1))


@pytest.mark.parametrize("family,s", [("uniform", 6), ("two_ring", 16)])
def test_register_at_max_exact_bits_is_bit_identical_to_64_bits(family, s):
    model = tiny_resnet(1)
    x = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    wide, ctx = run_ring(model, RingSpec(ell=32, s=s, L=64), x)
    totals = ctx.totals()
    assert totals["register_overflows"] == totals["storage_wraps"] == 0
    if family == "uniform":
        bits = max(totals["max_register_bits"], totals["max_storage_bits"])
        assert bits <= 32
        tight, below = RingSpec(ell=bits, s=s), RingSpec(ell=bits - 1, s=s)
    else:
        bits = totals["max_register_bits"]
        assert bits > 33
        tight, below = RingSpec(ell=32, s=s, L=bits), RingSpec(ell=32, s=s, L=bits - 1)
    same, tight_ctx = run_ring(model, tight, x)
    assert torch.equal(same, wide)
    assert tight_ctx.totals() == totals
    _, below_ctx = run_ring(model, below, x)
    below_totals = below_ctx.totals()
    assert below_totals["register_overflows"] + below_totals["storage_wraps"] > 0


def test_context_rejects_a_spec_with_different_fractional_bits():
    ring = convert_resnet(tiny_resnet(), RingSpec(ell=32, s=12, L=64))
    with pytest.raises(ValueError):
        ring.ctx.use(RingSpec(ell=32, s=10, L=64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_converted_resnet_is_bit_identical_on_cuda():
    model = tiny_resnet(2)
    x = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(2))
    # Registers peak at 24 bits here, so L = 23 overflows; L > ell + s = 22 lets
    # truncate-reduce wrap too (a uniform ring, ell == L, never can).
    spec = RingSpec(ell=12, s=10, L=23)
    cpu_codes, cpu_ctx = run_ring(model, spec, x)
    totals = cpu_ctx.totals()
    assert totals["register_overflows"] > 0 and totals["storage_wraps"] > 0
    ring = convert_resnet(model, spec).cuda()
    gpu_codes = ring(x.cuda())
    assert torch.equal(gpu_codes.cpu(), cpu_codes)
    assert ring.ctx.layers() == cpu_ctx.layers()
