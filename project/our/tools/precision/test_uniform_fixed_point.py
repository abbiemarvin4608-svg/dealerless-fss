"""Historical wide/saturating control; independent integer oracles.

The main implementation defaults to wide/saturating arithmetic. Every call
in this file also explicitly selects this policy so these tests verify the
same numerical rules independently of future default changes.
"""
import copy

import pytest
import torch
import torch.nn as nn

from uniform_fixed_point import (
    UniformAdaptiveAvgPool2d, UniformBatchNorm2d, UniformConv2d,
    UniformFixedPoint as _UniformFixedPoint, UniformLinear, UniformMaxPool2d,
    _native_ops as _raw_native_ops, convert_resnet_to_uniform as _convert_resnet_to_uniform,
)


def UniformFixedPoint(*args, **kwargs):
    kwargs.setdefault("arithmetic_mode", "wide")
    kwargs.setdefault("overflow_mode", "saturate")
    return _UniformFixedPoint(*args, **kwargs)


def convert_resnet_to_uniform(*args, **kwargs):
    kwargs.setdefault("arithmetic_mode", "wide")
    kwargs.setdefault("overflow_mode", "saturate")
    return _convert_resnet_to_uniform(*args, **kwargs)


class _LegacyNative:
    def __getattr__(self, name):
        function = getattr(_raw_native_ops(), name)

        def legacy_call(*args, **kwargs):
            kwargs.setdefault("ring", False)
            kwargs.setdefault("wrap", False)
            return function(*args, **kwargs)

        return legacy_call


def _native_ops():
    return _LegacyNative()


def test_all_constants_share_floor_encoding_and_finite_storage_range():
    fp = UniformFixedPoint(4, 2)
    assert fp.encode(torch.tensor([-3., -.1, .3, 3.]), "parameter").tolist() == [-8,-1,1,7]
    source = nn.Linear(2, 1).double()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[.3,-.1]])); source.bias.fill_(.3)
    layer = UniformLinear(source, fp)
    assert layer.weight_q.tolist() == [[1,-1]]
    assert layer.bias_q.tolist() == [1]  # Scale 2^s, never 2^(2s).
    assert fp.stats()["offline_clipped"] == 2
    assert fp.stats()["max_storage_bits"] == 5


def test_dot_product_rescales_once_instead_of_flooring_each_term():
    source = nn.Linear(2, 1, bias=False).double()
    with torch.no_grad(): source.weight.fill_(.75)
    fp = UniformFixedPoint(8,2)
    layer = UniformLinear(source,fp)
    assert layer(torch.tensor([[1,1]],dtype=torch.int64)).item() == 1  # floor(6/4), not 0+0.


def test_cancellation_survives_wide_mac_then_one_storage_saturation():
    source = nn.Linear(3,1,bias=False).double()
    with torch.no_grad(): source.weight.fill_(1)
    fp = UniformFixedPoint(4,0)
    result = UniformLinear(source,fp)(torch.tensor([[6,6,-6]],dtype=torch.int64))
    assert result.item() == 6  # Strict per-term saturation returns 1.
    stats = fp.stats()
    assert stats["activation_clipped"] == 0
    assert stats["max_register_bits"] == 5
    assert stats["max_storage_bits"] == 4


def test_negative_floor_and_same_scale_bias_match_python_integer_oracle():
    x = torch.tensor([[3,-5,7],[-7,2,-1]],dtype=torch.int64)
    w = torch.tensor([[3,5,-7],[-2,1,4]],dtype=torch.int64)
    bias = torch.tensor([2,-3],dtype=torch.int64)
    output,_ = _native_ops().linear(x,w,bias,8,3,128)
    expected = [[max(-128,min(127,sum(int(a)*int(b) for a,b in zip(row,weight))//8+int(offset)))
                 for weight,offset in zip(w,bias)] for row in x]
    assert output.tolist() == expected


def test_bn_affine_offset_is_quantized_at_the_same_scale():
    bn = nn.BatchNorm2d(2,eps=0).double().eval()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([.75,-.5])); bn.bias.copy_(torch.tensor([.5,2]))
        bn.running_mean.copy_(torch.tensor([.3125,0])); bn.running_var.copy_(torch.tensor([.25,1]))
    fp = UniformFixedPoint(16,4)
    layer = UniformBatchNorm2d(bn,fp)
    assert layer.multiplier_q.tolist() == [24,-8]
    assert layer.offset_q.tolist() == [0,32]  # floor(.03125*16)=0, not an extra fractional scale.
    x = torch.tensor([-40,16,-8,32]).view(1,2,2,1)
    assert layer(x).flatten().tolist() == [-60,24,36,16]


def test_average_pool_uses_wide_sum_before_division():
    fp = UniformFixedPoint(4,0)
    pool = UniformAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d(1),fp)
    values = torch.tensor([[[[6,6],[-6,-6]],[[6,6],[6,6]],[[-1,-2],[-3,-5]]]],dtype=torch.int64)
    assert pool(values).flatten().tolist() == [0,6,-3]
    assert fp.stats()["max_register_bits"] == 6
    assert fp.stats()["activation_clipped"] == 0


def test_residual_add_saturates_only_its_output():
    fp = UniformFixedPoint(4,0)
    assert fp.add(torch.tensor([6,-6]),torch.tensor([6,-6])).tolist() == [7,-8]
    assert fp.stats()["activation_clipped"] == 2
    assert fp.stats()["max_storage_bits"] == 5


def test_prefix_overflow_is_detected_even_when_final_sum_cancels():
    peak = (1 << 31)-1
    x = torch.tensor([[peak,peak,peak,-peak,-peak,-peak]],dtype=torch.int64)
    w = torch.full((1,6),peak,dtype=torch.int64)
    empty = torch.empty(0,dtype=torch.int64)
    result,stats = _native_ops().linear(x,w,empty,32,0,128)
    assert result.item() == 0 and stats[1].item() == 0 and stats[2].item() == 65
    _,narrow_stats = _native_ops().linear(x,w,empty,32,0,64)
    assert narrow_stats[1].item() == 1
    with pytest.raises(OverflowError,match="prefix"):
        UniformFixedPoint(32,0,register_bits=64).record_native(narrow_stats,1)


def test_stats_cover_unclipped_storage_values_and_reset_preserves_parameters():
    fp = UniformFixedPoint(4,1)
    fp.encode(torch.tensor([9.]),"parameter")
    fp.encode(torch.tensor([-9.]),"input")
    fp.add(torch.tensor([7]),torch.tensor([7]))
    assert fp.stats()["max_storage_bits"] == 6
    assert fp.stats()["activation_max_storage_bits"] == 5
    fp.reset_runtime_stats()
    stats = fp.stats()
    assert stats["offline_clipped"] == 1 and stats["input_clipped"] == 0
    assert stats["activation_total"] == 0 and stats["max_storage_bits"] == 6


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_ops_equal_cpu_oracles_including_wide_prefix_statistics():
    generator = torch.Generator().manual_seed(7)
    x = torch.randint(-40,41,(2,2,5,6),generator=generator,dtype=torch.int64)
    w = torch.randint(-30,31,(3,2,3,2),generator=generator,dtype=torch.int64)
    b = torch.tensor([3,-5,11],dtype=torch.int64)
    args = (2,1,1,0,1,2,8,3,128)
    cpu,st_cpu = _native_ops().conv2d(x,w,b,*args)
    cuda,st_cuda = _native_ops().conv2d(x.cuda(),w.cuda(),b.cuda(),*args)
    assert torch.equal(cpu,cuda.cpu()) and torch.equal(st_cpu,st_cuda.cpu())
    fast,st_fast = _native_ops().conv2d(x.cuda(),w.cuda(),b.cuda(),*args,True)
    assert torch.equal(cpu,fast.cpu()) and torch.equal(st_cpu,st_fast.cpu())
    peak = (1<<31)-1
    xlin = torch.tensor([[peak,peak,peak,-peak,-peak,-peak]],dtype=torch.int64)
    wlin = torch.full((1,6),peak,dtype=torch.int64)
    for register in (64,128):
        a,sa = _native_ops().linear(xlin,wlin,torch.empty(0,dtype=torch.int64),32,0,register)
        b,sb = _native_ops().linear(xlin.cuda(),wlin.cuda(),torch.empty(0,dtype=torch.int64,device="cuda"),32,0,register)
        assert torch.equal(a,b.cpu()) and torch.equal(sa,sb.cpu())
    cpu,sc = _native_ops().average(x,8,128)
    cuda,sg = _native_ops().average(x.cuda(),8,128)
    assert torch.equal(cpu,cuda.cpu()) and torch.equal(sc,sg.cpu())


def _tiny_resnet():
    from torchvision.models.resnet import BasicBlock,ResNet
    torch.manual_seed(3)
    model = ResNet(BasicBlock,[1,1,1,1],num_classes=10).eval()
    return model


def test_full_network_has_uniform_integer_buffers_and_tracks_fp32_at_high_precision():
    source = _tiny_resnet()
    fixed = convert_resnet_to_uniform(source,32,20)
    assert not list(fixed.parameters())
    assert all(buffer.dtype == torch.int64 and (buffer >= fixed.fp.qmin).all() and
               (buffer <= fixed.fp.qmax).all() for buffer in fixed.buffers())
    inputs = torch.randn(1,3,16,16,generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        codes = fixed(inputs)
        reference = source(inputs).double()
    assert codes.dtype == torch.int64
    assert torch.allclose(fixed.fp.decode(codes),reference,rtol=0,atol=.002)
    assert fixed.fp.stats()["register_overflows"] == 0
    assert fixed.fp.stats()["activation_clipped"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_full_network_cpu_cuda_parity():
    source = _tiny_resnet()
    cpu = convert_resnet_to_uniform(source,16,8)
    cuda = convert_resnet_to_uniform(copy.deepcopy(source),16,8).cuda()
    inputs = torch.randn(1,3,16,16,generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a,b = cpu(inputs),cuda(inputs.cuda()).cpu()
    assert torch.equal(a,b)
    cpu_stats,cuda_stats = cpu.fp.stats(),cuda.fp.stats()
    for key in ("int64_fastpath_calls","int128_path_calls"):
        cpu_stats.pop(key);cuda_stats.pop(key)
    assert cpu_stats == cuda_stats
    cuda.fp.reset_runtime_stats()
    cuda.fp.allow_int64_fastpath = False
    with torch.no_grad(): wide = cuda(inputs.cuda()).cpu()
    assert torch.equal(a,wide)
    pool = UniformMaxPool2d(nn.MaxPool2d(3,2,1))
    negative = torch.full((1,1,3,3),-5,dtype=torch.int64)
    assert torch.equal(pool(negative),pool(negative.cuda()).cpu())


def _five_limb_case():
    generator = torch.Generator().manual_seed(91)
    x = torch.randint(-(1<<31),(1<<31),(6,20),generator=generator,dtype=torch.int64)
    w = torch.randint(-(1<<31),(1<<31),(4,20),generator=generator,dtype=torch.int64)
    xd = [(x >> (7*i)) & 127 for i in range(4)] + [x >> 28]
    wd = [(w >> (7*i)) & 127 for i in range(4)] + [w >> 28]
    dots = torch.stack([(a @ b.t()).to(torch.int32) for a in xd for b in wd])
    shifts = torch.tensor([7*(i+j) for i in range(5) for j in range(5)],dtype=torch.int64)
    bias = torch.tensor([11,-13,17,-19],dtype=torch.int64)
    raw = [[sum(int(a)*int(b) for a,b in zip(row,weight)) for weight in w] for row in x]
    assert any(abs(value) >= 1<<63 for row in raw for value in row)
    expected = [[max(-(1<<31),min((1<<31)-1,value//(1<<30)+int(offset)))
                 for value,offset in zip(row,bias)] for row in raw]
    # Deliberately conservative, proving every raw-dot and limb-reassembly prefix.
    raw_bound = 20*int(x.abs().max())*int(w.abs().max())+(int(bias.abs().max())<<30)
    recombine_bound = max(sum(abs(int(dots[t,i,j]))<<int(shifts[t]) for t in range(25))
                          +(abs(int(bias[j]))<<30) for i in range(6) for j in range(4))
    proof_bits = 1+max(raw_bound,recombine_bound).bit_length()
    return dots,shifts,bias,torch.tensor(expected,dtype=torch.int64),proof_bits


def test_limb_combiner_matches_python_big_integers_for_beyond_int64_raw_dots():
    dots,shifts,bias,expected,proof_bits = _five_limb_case()
    result,stats = _native_ops().combine_limb_dots(dots,shifts,bias,32,30,128,proof_bits)
    assert torch.equal(result,expected)
    assert stats[1].item() == 0 and stats[2].item() >= proof_bits > 64
    with pytest.raises(RuntimeError,match="register overflow"):
        _native_ops().combine_limb_dots(dots,shifts,bias,32,30,64,proof_bits)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_limb_combiner_cpu_cuda_codes_and_statistics_are_identical():
    dots,shifts,bias,expected,proof_bits = _five_limb_case()
    cpu,sc = _native_ops().combine_limb_dots(dots,shifts,bias,32,30,128,proof_bits)
    cuda,sg = _native_ops().combine_limb_dots(dots.cuda(),shifts.cuda(),bias.cuda(),32,30,128,proof_bits)
    assert torch.equal(cuda.cpu(),expected) and torch.equal(cpu,cuda.cpu())
    assert torch.equal(sc,sg.cpu())


def test_limb_combiner_checks_recombination_prefix_despite_final_cancellation():
    dots = torch.tensor([[[1]],[[1]],[[-1]],[[-1]]],dtype=torch.int32)
    shifts = torch.full((4,),62,dtype=torch.int64)
    empty = torch.empty(0,dtype=torch.int64)
    result,stats = _native_ops().combine_limb_dots(dots,shifts,empty,32,0,128,65)
    assert result.item() == 0 and stats[2].item() == 65
    # A false caller proof does not hide the actually observed 65-bit prefix.
    _,stats = _native_ops().combine_limb_dots(dots,shifts,empty,32,0,64,64)
    assert stats[1].item() == 1


def test_requested_gemm_on_cpu_uses_the_exact_scalar_reference():
    fp = UniformFixedPoint(16,8,integer_backend="gemm")
    source = nn.Linear(3,2).double()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[.5,-.25,1.],[-1.,.125,.75]]))
        source.bias.copy_(torch.tensor([.25,-.5]))
    layer = UniformLinear(source,fp)
    result = layer(torch.tensor([[123,-57,89]],dtype=torch.int64))
    reference,_ = _native_ops().linear(torch.tensor([[123,-57,89]],dtype=torch.int64),
                                      layer.weight_q,layer.bias_q,16,8,128)
    assert torch.equal(result,reference)
    assert fp.stats()["effective_integer_backend"] == "scalar"
    assert fp.stats()["max_register_bits_kind"] == "observed_prefix"


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("ell,s",[(16,8),(32,20)])
def test_full_network_gemm_and_scalar_match_with_uniform_stored_parameters(ell,s):
    source = _tiny_resnet()
    scalar = convert_resnet_to_uniform(source,ell,s).cuda()
    gemm = convert_resnet_to_uniform(copy.deepcopy(source),ell,s,integer_backend="gemm").cuda()
    inputs = torch.randn(2,3,16,16,generator=torch.Generator().manual_seed(12)).cuda()
    with torch.no_grad():
        reference,result = scalar(inputs),gemm(inputs)
    assert torch.equal(reference,result)
    ss,gs = scalar.fp.stats(),gemm.fp.stats()
    for key in ("offline_clipped","offline_total","input_clipped","input_total",
                "activation_clipped","activation_total","max_storage_bits","register_overflows"):
        assert ss[key] == gs[key]
    assert gs["gemm_calls"] > 0 and gs["effective_integer_backend"] == "gemm"
    assert gs["max_register_bits_kind"] == "proven_upper_bound"
    assert gs["max_register_bits"] >= ss["max_register_bits"]
