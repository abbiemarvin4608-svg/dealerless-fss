"""Independent floor/saturation oracles for the ReLU-only precision ablation."""
import copy
import math

import pytest
import torch
import torch.nn as nn

from activation_precision import (
    ActivationPrecisionReLU, ActivationPrecisionStats, inject_relu_quantizers,
)
from uniform_fixed_point import UniformReLU


def test_float_control_quantizes_signed_input_and_signed_sixteen_bit_output():
    layer = ActivationPrecisionReLU(16, 2)
    inputs = torch.tensor([-10000., -1.25, -.01, 0., .0625, .125, 8191.75, 8192.])
    original = inputs.clone()
    output = layer(inputs)
    expected = [max(0, max(-32768, min(32767, math.floor(float(x) * 4)))) / 4
                for x in inputs]
    assert output.tolist() == expected
    assert output.dtype == inputs.dtype
    assert torch.equal(inputs, original)
    stats = layer.stats.snapshot()
    assert stats["relu_input_total"] == 8
    assert stats["relu_lower_clipped"] == 1
    assert stats["relu_upper_clipped"] == 1
    assert stats["relu_positive_rounded"] == 2
    assert stats["relu_positive_zeroed"] == 2
    assert stats["relu_calls"] == 1
    assert output[-1].item() == 32767 / 4  # Does not reuse the sign bit.


def test_float_extreme_values_saturate_before_cast_without_integer_overflow():
    layer = ActivationPrecisionReLU(16, 15)
    result = layer(torch.tensor([-1.e30, 1.e30]))
    assert result.tolist() == [0., 32767 / 32768]
    assert layer.stats.snapshot()["relu_input_clipped"] == 2


def test_integer_control_uses_signed_floor_saturation_and_exact_return_scaling():
    layer = ActivationPrecisionReLU(4, 1, base_ell=16, base_s=3)
    inputs = torch.tensor([-32768, -33, -32, -1, 0, 1, 3, 4, 7, 28, 31, 32, 32767], dtype=torch.int64)
    original = inputs.clone()
    expected = [max(0, max(-8, min(7, int(x) // 4))) * 4 for x in inputs]
    assert layer(inputs).tolist() == expected
    assert torch.equal(inputs, original)
    stats = layer.stats.snapshot()
    assert stats["relu_lower_clipped"] == 2
    assert stats["relu_upper_clipped"] == 2
    assert stats["relu_positive_rounded"] == 5
    assert stats["relu_positive_zeroed"] == 2


@pytest.mark.parametrize("activation_s", range(16))
def test_integer_base_lattice_survives_exactly_when_no_activation_clipping(activation_s):
    factor = 1 << (15 - activation_s)
    # Every code is on the coarser activation lattice and fits Q(16,s).
    inputs = torch.tensor([-32768, -7, -1, 0, 1, 7, 32767], dtype=torch.int64) * factor
    layer = ActivationPrecisionReLU(16, activation_s, base_ell=32, base_s=15)
    output = layer(inputs)
    assert torch.equal(output, inputs.clamp_min(0))
    assert layer.stats.snapshot()["relu_input_clipped"] == 0
    assert layer.stats.snapshot()["relu_positive_rounded"] == 0


@pytest.mark.parametrize("activation_s", [0, 3, 12, 15])
def test_signed_limits_and_neighbors_match_python_floor_oracle(activation_s):
    factor = 1 << (15 - activation_s)
    inputs = torch.tensor([-32768 * factor - 1, -32768 * factor,
                           32767 * factor, 32768 * factor - 1, 32768 * factor], dtype=torch.int64)
    layer = ActivationPrecisionReLU(16, activation_s, base_ell=32, base_s=15)
    expected = [max(0, max(-32768, min(32767, int(x) // factor))) * factor for x in inputs]
    assert layer(inputs).tolist() == expected
    assert layer.stats.snapshot()["relu_lower_clipped"] == 1
    assert layer.stats.snapshot()["relu_upper_clipped"] == 1


def test_integer_upscale_has_no_float_roundtrip_even_for_large_base_codes():
    layer = ActivationPrecisionReLU(32, 5, base_ell=32, base_s=0)
    inputs = torch.tensor([-2147483648, -1, 0, 1, 67108863, 67108864, 2147483647], dtype=torch.int64)
    expected = [max(0, max(-(1 << 31), min((1 << 31) - 1, int(x) * 32))) // 32
                for x in inputs]
    assert layer(inputs).tolist() == expected
    assert layer(inputs).dtype == torch.int64


def test_return_to_coarser_base_counts_positive_values_that_become_zero():
    layer = ActivationPrecisionReLU(16, 15, base_ell=24, base_s=0)
    assert layer(torch.tensor([0, 1], dtype=torch.int64)).tolist() == [0, 0]
    assert layer.stats.snapshot()["relu_positive_zeroed"] == 1


def test_shared_stats_reset_and_aliased_module_preserve_invocation_count():
    class SharedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.alias = self.relu

        def forward(self, inputs):
            return self.relu(self.alias(self.relu(inputs)))

    model = SharedModel()
    replacements, stats = inject_relu_quantizers(model, 16, 12)
    assert set(replacements) == {"relu", "alias"}
    assert model.relu is model.alias
    inputs = torch.tensor([-1., 0., 1.])
    assert torch.equal(model(inputs), inputs.clamp_min(0))
    assert stats.snapshot()["relu_calls"] == 3
    assert stats.snapshot()["relu_module_count"] == 1
    assert stats.snapshot()["relu_input_total"] == 9
    stats.reset()
    assert stats.snapshot()["relu_calls"] == 0
    assert stats.snapshot()["relu_input_total"] == 0
    assert stats.snapshot()["relu_input_clipped"] == 0


def test_uniform_relu_replacement_requires_explicit_integer_base_format():
    model = nn.Sequential(UniformReLU())
    with pytest.raises(ValueError, match="explicit integer base"):
        inject_relu_quantizers(model)
    replacements, stats = inject_relu_quantizers(model, 16, 12, base_ell=24, base_s=15)
    assert len(replacements) == 1
    assert model(torch.tensor([-1, 1, 8], dtype=torch.int64)).tolist() == [0, 0, 8]
    assert stats.snapshot()["relu_calls"] == 1


def test_reinjection_reuses_integer_network_and_installs_fresh_stats():
    model = nn.Sequential(UniformReLU())
    first_modules, first_stats = inject_relu_quantizers(
        model, 16, 12, base_ell=24, base_s=15)
    inputs = torch.tensor([-8, 8, 32768], dtype=torch.int64)
    assert model(inputs).tolist() == [0, 8, 32768]
    before = first_stats.snapshot()
    second_modules, second_stats = inject_relu_quantizers(
        model, 16, 0, base_ell=24, base_s=15)
    assert second_modules["0"] is not first_modules["0"]
    assert second_stats is not first_stats
    assert second_stats.snapshot()["relu_calls"] == 0
    assert model(inputs).tolist() == [0, 0, 32768]
    assert first_stats.snapshot() == before
    assert second_stats.snapshot()["relu_calls"] == 1
    assert second_stats.snapshot()["relu_module_call_counts"] == {"0": 1}
    assert second_stats.snapshot()["relu_s"] == 0


def test_reinjection_preserves_aliases_and_float_path():
    model = nn.Module()
    model.a = nn.ReLU()
    model.b = model.a
    inject_relu_quantizers(model, 16, 12, track_stats=False)
    modules, stats = inject_relu_quantizers(model, 16, 1)
    assert modules["a"] is modules["b"]
    assert model.a(torch.tensor([.25, .75])).tolist() == [0., .5]
    assert stats.snapshot()["relu_calls"] == 1


@pytest.mark.parametrize("architecture,blocks,calls", [("resnet50", 16, 49), ("resnet152", 50, 151)])
def test_torchvision_all_bottleneck_relu_calls_are_replaced(architecture, blocks, calls):
    import torchvision.models as models
    model = getattr(models, architecture)(weights=None).eval()
    replacements, stats = inject_relu_quantizers(model, 16, 12)
    assert len(replacements) == blocks + 1
    assert not any(isinstance(module, nn.ReLU) for module in model.modules())
    # Exercise the actual forward graph; a small image and a single CPU
    # thread keep this check inexpensive without downloading weights.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 32, 32))
    finally:
        torch.set_num_threads(previous_threads)
    assert output.shape == (1, 1000)
    assert stats.snapshot()["relu_calls"] == calls
    assert stats.snapshot()["relu_module_count"] == blocks + 1
    assert stats.snapshot()["relu_input_total"] > calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("activation_s", [0, 12, 15])
def test_cuda_integer_conversion_and_device_stats_match_cpu(activation_s):
    inputs = torch.tensor([-8388608, -1, 0, 1, 7, 8, 1000, 32767, 8388607], dtype=torch.int64)
    cpu = ActivationPrecisionReLU(16, activation_s, base_ell=24, base_s=15)
    cuda = ActivationPrecisionReLU(16, activation_s, base_ell=24, base_s=15)
    expected = cpu(inputs)
    actual = cuda(inputs.cuda())
    assert torch.equal(actual.cpu(), expected)
    assert cuda.stats.snapshot() == cpu.stats.snapshot()


def test_float_replacement_preserves_unmodified_modules_and_baseline_on_exact_values():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(inplace=True), nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2)); model[0].bias.zero_()
        model[2].weight.fill_(1); model[2].bias.zero_()
    reference = copy.deepcopy(model)
    original_linears = model[0], model[2]
    inject_relu_quantizers(model, 16, 12)
    inputs = torch.tensor([[-1., 2.], [1., 3.]])
    assert torch.equal(model(inputs), reference(inputs))
    assert model[0] is original_linears[0] and model[2] is original_linears[1]


def test_disabled_statistics_do_not_change_outputs():
    model = nn.Sequential(nn.ReLU())
    _, stats = inject_relu_quantizers(model, 16, 12, track_stats=False)
    assert model(torch.tensor([-.1, .1])).tolist() == [0., math.floor(.1 * 4096) / 4096]
    assert stats.snapshot()["relu_calls"] == 0


@pytest.mark.parametrize("kwargs", [{"activation_bits": 1, "activation_s": 0},
                                    {"activation_bits": 16, "activation_s": 16},
                                    {"activation_bits": 16, "activation_s": 12, "base_ell": 24}])
def test_invalid_formats_rejected(kwargs):
    with pytest.raises(ValueError):
        ActivationPrecisionReLU(**kwargs)


def test_wrong_tensor_representation_rejected():
    with pytest.raises(TypeError, match="floating"):
        ActivationPrecisionReLU(16, 12)(torch.tensor([1]))
    with pytest.raises(TypeError, match="int64"):
        ActivationPrecisionReLU(16, 12, base_ell=24, base_s=15)(torch.tensor([1.]))
