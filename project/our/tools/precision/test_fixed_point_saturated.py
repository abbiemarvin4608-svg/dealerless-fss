"""Correctness tests for strict E0-Q-fixed arithmetic."""

import pytest
import torch
import torch.nn as nn

from fixed_point_saturated import (
    SaturatingAdaptiveAvgPool2d,
    SaturatingConv2d,
    SaturatingFixedPoint,
    SaturatingLinear,
    SaturatingMaxPool2d,
    convert_resnet_to_saturating,
)


def test_encode_is_floor_and_saturating():
    fp = SaturatingFixedPoint(4, 1)
    encoded = fp.encode(torch.tensor([-5.0, -0.1, 0.5, 9.0]))
    assert torch.equal(encoded, torch.tensor([-8, -1, 1, 7]))


def test_multiply_floor_rescales_and_immediately_saturates():
    fp = SaturatingFixedPoint(4, 2)
    # floor((-3 * 3) / 4) = -3.
    assert fp.mul(torch.tensor([-3]), torch.tensor([3])).item() == -3
    # floor((7 * 7) / 4) = 12, then saturation to qmax=7.
    assert fp.mul(torch.tensor([7]), torch.tensor([7])).item() == 7


def test_addition_saturates_each_step_instead_of_using_wide_accumulator():
    fp = SaturatingFixedPoint(4, 0)
    # Strict ordered Q additions: sat(6+6)=7, then sat(7-6)=1.
    strict = fp.add(torch.tensor([6]), torch.tensor([6]), torch.tensor([-6]))
    assert strict.item() == 1
    assert strict.item() != 6  # unclipped wide sum would be 6


def test_linear_uses_strict_q_mac_order():
    fp = SaturatingFixedPoint(4, 0)
    base = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        base.weight.fill_(1.0)
    layer = SaturatingLinear(base, fp)
    output = layer(torch.tensor([[6, 6, -6]], dtype=torch.int64))
    assert output.item() == 1


def test_conv_uses_strict_q_mac_order():
    fp = SaturatingFixedPoint(4, 0)
    base = nn.Conv2d(1, 1, (1, 3), bias=False)
    with torch.no_grad():
        base.weight.fill_(1.0)
    layer = SaturatingConv2d(base, fp)
    output = layer(torch.tensor([[[[6, 6, -6]]]], dtype=torch.int64))
    assert output.item() == 1


def test_global_average_pool_has_ell_bit_accumulator():
    fp = SaturatingFixedPoint(4, 0)
    layer = SaturatingAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d((1, 1)), fp)
    x = torch.tensor([[[[6, 6], [-6, -6]]]], dtype=torch.int64)
    # Ordered sum: 6 -> 7 -> 1 -> -5; floor(-5/4)=-2.
    assert layer(x).item() == -2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_kernels_match_literal_cpu_reference():
    torch.manual_seed(7)
    conv = nn.Conv2d(2, 3, 3, stride=2, padding=1, bias=True)
    linear = nn.Linear(7, 4, bias=True)
    max_pool = nn.MaxPool2d(3, stride=2, padding=1)
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    fp_cpu = SaturatingFixedPoint(8, 3)
    fp_cuda = SaturatingFixedPoint(8, 3)
    conv_cpu = SaturatingConv2d(conv, fp_cpu)
    conv_cuda = SaturatingConv2d(conv, fp_cuda).cuda()
    x = torch.randint(-100, 101, (2, 2, 7, 7), dtype=torch.int64)
    assert torch.equal(conv_cpu(x), conv_cuda(x.cuda()).cpu())

    lin_cpu = SaturatingLinear(linear, fp_cpu)
    lin_cuda = SaturatingLinear(linear, fp_cuda).cuda()
    z = torch.randint(-100, 101, (5, 7), dtype=torch.int64)
    assert torch.equal(lin_cpu(z), lin_cuda(z.cuda()).cpu())

    max_cpu = SaturatingMaxPool2d(max_pool, fp_cpu)
    max_cuda = SaturatingMaxPool2d(max_pool, fp_cuda).cuda()
    assert torch.equal(max_cpu(x), max_cuda(x.cuda()).cpu())

    avg_cpu = SaturatingAdaptiveAvgPool2d(avg_pool, fp_cpu)
    avg_cuda = SaturatingAdaptiveAvgPool2d(avg_pool, fp_cuda).cuda()
    assert torch.equal(avg_cpu(x), avg_cuda(x.cuda()).cpu())


def test_converted_resnet_has_no_floating_runtime_operator():
    from torchvision.models import resnet18

    fixed = convert_resnet_to_saturating(resnet18(weights=None).eval(), 16, 8)
    forbidden = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.MaxPool2d,
                 nn.AdaptiveAvgPool2d)
    assert not any(isinstance(module, forbidden) for module in fixed.modules())
    assert all(parameter.numel() == 0 for parameter in fixed.parameters())
    assert all(buffer.dtype == torch.int64 for buffer in fixed.buffers())
