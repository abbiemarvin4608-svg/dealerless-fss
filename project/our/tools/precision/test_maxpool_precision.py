"""Independent integer oracles for the signed MaxPool precision boundary."""
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from activation_precision import ActivationPrecisionReLU
from maxpool_precision import ActivationPrecisionMaxPool2d, inject_maxpool_quantizers
from uniform_fixed_point import UniformFixedPoint, UniformMaxPool2d, UniformReLU


def _pool(kernel=3, stride=2, padding=1):
    return UniformMaxPool2d(nn.MaxPool2d(kernel, stride, padding), UniformFixedPoint(24, 15))


def _literal_oracle(inputs, kernel, stride, padding, bits, activation_s, base_s):
    """Python scalar quantization and window maxima, independent of tensor path."""
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    def to_activation(value):
        difference = activation_s - base_s
        value = int(value) * (1 << difference) if difference >= 0 else int(value) // (1 << -difference)
        return min(qmax, max(qmin, value))
    def to_base(value):
        difference = base_s - activation_s
        return value * (1 << difference) if difference >= 0 else value // (1 << -difference)
    n, c, height, width = inputs.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    output = torch.empty((n, c, (height + 2 * ph - kh) // sh + 1,
                          (width + 2 * pw - kw) // sw + 1), dtype=torch.int64)
    for batch in range(n):
        for channel in range(c):
            for row in range(output.shape[2]):
                for col in range(output.shape[3]):
                    values = []
                    for dy in range(kh):
                        for dx in range(kw):
                            y, x = row * sh - ph + dy, col * sw - pw + dx
                            values.append(to_activation(inputs[batch, channel, y, x])
                                          if 0 <= y < height and 0 <= x < width else qmin)
                    output[batch, channel, row, col] = to_base(max(values))
    return output


def test_signed_floor_saturation_and_negative_padding_match_scalar_oracle():
    layer = ActivationPrecisionMaxPool2d(_pool(3, 1, 1), 4, 1, 24, 3)
    inputs = torch.tensor([[[[-33, -1, 0], [1, 3, 4], [28, 31, 32]]]], dtype=torch.int64)
    original = inputs.clone()
    expected = _literal_oracle(inputs, (3, 3), (1, 1), (1, 1), 4, 1, 3)
    assert torch.equal(layer(inputs), expected)
    assert torch.equal(inputs, original)
    assert layer.padding_code == -8
    stats = layer.stats.snapshot()
    assert stats["maxpool_calls"] == 1
    assert stats["maxpool_input_total"] == stats["maxpool_output_total"] == 9
    assert stats["maxpool_input_lower_clipped"] == 1
    assert stats["maxpool_input_upper_clipped"] == 1
    assert stats["maxpool_input_rounded"] == 5
    assert stats["maxpool_output_clipped"] == stats["maxpool_output_rounded"] == 0


def test_padding_cannot_replace_negative_valid_values_with_zero():
    inputs = torch.full((1, 1, 2, 2), -1025, dtype=torch.int64)
    layer = ActivationPrecisionMaxPool2d(_pool(3, 1, 1), 16, 6, 24, 15)
    assert layer(inputs).tolist() == [[[[-1536, -1536], [-1536, -1536]]]]


@pytest.mark.parametrize("activation_s", [0, 6, 12, 15])
def test_quantization_commutes_with_maxpool_including_signed_padding(activation_s):
    # This checks the independent monotonic-conversion equivalence without
    # introducing a second MaxPool operation in the experiment's forward path.
    torch.manual_seed(12)
    inputs = torch.randint(-(1 << 23), 1 << 23, (2, 3, 7, 9), dtype=torch.int64)
    pool = _pool((3, 3), (2, 2), (1, 1))
    layer = ActivationPrecisionMaxPool2d(pool, 16, activation_s, 24, 15)
    raw_max = pool(inputs)
    expected_codes = torch.div(raw_max, 1 << (15 - activation_s), rounding_mode="floor").clamp(-32768, 32767)
    assert torch.equal(layer(inputs), expected_codes * (1 << (15 - activation_s)))


def test_relu_then_same_format_maxpool_is_exactly_idempotent():
    torch.manual_seed(91)
    inputs = torch.randint(-(1 << 23), 1 << 23, (2, 3, 7, 9), dtype=torch.int64)
    relu = ActivationPrecisionReLU(16, 6, base_ell=24, base_s=15)
    original_pool = _pool()
    limited_pool = ActivationPrecisionMaxPool2d(original_pool, 16, 6, 24, 15)
    relu_output = relu(inputs)
    assert torch.equal(limited_pool(relu_output), original_pool(relu_output))
    stats = limited_pool.stats.snapshot()
    assert stats["maxpool_input_rounded"] == stats["maxpool_input_clipped"] == 0


def test_q16_s6_does_not_clip_canonical_q24_s15_inputs():
    inputs = torch.tensor([[[[-8388608, -1], [0, 8388607]]]], dtype=torch.int64)
    layer = ActivationPrecisionMaxPool2d(_pool(1, 1, 0), 16, 6, 24, 15)
    assert layer(inputs).tolist() == [[[[-8388608, -512], [0, 8388096]]]]
    assert layer.stats.snapshot()["maxpool_input_clipped"] == 0


def test_tuple_kernel_stride_padding_and_upscale_format_match_scalar_oracle():
    inputs = torch.tensor([[[[-2, -1, 0, 1], [3, 4, 5, 6], [7, 8, 9, 10]]]], dtype=torch.int64)
    layer = ActivationPrecisionMaxPool2d(_pool((2, 3), (1, 2), (0, 1)), 4, 2, 24, 0)
    expected = _literal_oracle(inputs, (2, 3), (1, 2), (0, 1), 4, 2, 0)
    assert torch.equal(layer(inputs), expected)


def test_injection_aliases_reset_reinjection_and_preserves_other_modules():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = nn.Identity()
            self.relu = UniformReLU()
            self.pool = _pool(1, 1, 0)
            self.alias = self.pool

        def forward(self, inputs):
            return self.alias(self.pool(self.relu(self.identity(inputs))))

    model = Model()
    identity, relu = model.identity, model.relu
    replacements, collector = inject_maxpool_quantizers(model)
    assert set(replacements) == {"pool", "alias"}
    assert model.pool is model.alias
    assert model.identity is identity and model.relu is relu
    inputs = torch.tensor([[[[-1, 513], [1024, 1535]]]], dtype=torch.int64)
    assert model(inputs).tolist() == [[[[0, 512], [1024, 1024]]]]
    stats = collector.snapshot()
    assert stats["maxpool_module_count"] == 1
    assert stats["maxpool_calls"] == 2
    assert stats["maxpool_module_call_counts"] == {"pool": 2}
    assert stats["maxpool_input_total"] == stats["maxpool_output_total"] == 8
    first_pool, first_stats = model.pool, collector.snapshot()
    replacements, collector = inject_maxpool_quantizers(model, activation_s=12)
    assert model.pool is model.alias and model.pool is not first_pool
    assert model.pool.kernel == first_pool.kernel
    assert model.pool.stride == first_pool.stride and model.pool.padding == first_pool.padding
    assert collector.snapshot()["maxpool_calls"] == 0
    model(inputs)
    assert collector.snapshot()["maxpool_calls"] == 2
    assert first_pool.stats.snapshot() == first_stats
    collector.reset()
    assert collector.snapshot()["maxpool_calls"] == 0
    assert collector.snapshot()["maxpool_input_total"] == 0


def test_disabled_statistics_do_not_change_output():
    first = nn.Sequential(_pool())
    second = copy.deepcopy(first)
    _, enabled = inject_maxpool_quantizers(first, track_stats=True)
    _, disabled = inject_maxpool_quantizers(second, track_stats=False)
    inputs = torch.arange(49, dtype=torch.int64).reshape(1, 1, 7, 7) * 513 - 10
    assert torch.equal(first(inputs), second(inputs))
    assert enabled.snapshot()["maxpool_calls"] == 1
    assert disabled.snapshot()["maxpool_calls"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("activation_s", [0, 6, 12, 15])
def test_cuda_output_and_device_stats_match_cpu(activation_s):
    inputs = torch.tensor([[[[-8388608, -32769, -513, -1],
                             [0, 1, 513, 32768], [8388607, 3, -3, 1024]]]], dtype=torch.int64)
    cpu = ActivationPrecisionMaxPool2d(_pool(3, 1, 1), 16, activation_s, 24, 15)
    cuda = ActivationPrecisionMaxPool2d(_pool(3, 1, 1), 16, activation_s, 24, 15)
    assert torch.equal(cpu(inputs), cuda(inputs.cuda()).cpu())
    assert cpu.stats.snapshot() == cuda.stats.snapshot()


@pytest.mark.parametrize("kwargs", [dict(activation_bits=1), dict(activation_s=16),
                                    dict(base_ell=1), dict(base_s=24)])
def test_invalid_formats_rejected(kwargs):
    with pytest.raises(ValueError):
        inject_maxpool_quantizers(nn.Sequential(_pool()), **kwargs)


def test_wrong_representation_and_missing_pool_rejected():
    layer = ActivationPrecisionMaxPool2d(_pool())
    with pytest.raises(TypeError, match="int64"):
        layer(torch.zeros(1, 1, 3, 3))
    with pytest.raises(ValueError, match="NCHW"):
        layer(torch.zeros(3, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="no UniformMaxPool"):
        inject_maxpool_quantizers(nn.Sequential(nn.ReLU()))
    with pytest.raises(TypeError, match="UniformMaxPool"):
        ActivationPrecisionMaxPool2d(nn.MaxPool2d(3))
