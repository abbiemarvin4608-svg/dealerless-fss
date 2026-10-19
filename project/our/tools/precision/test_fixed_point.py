"""
Test suite for fixed-point emulation module.

Following TDD methodology: each test was written before the production code,
watched to fail, then implementation was added to make it pass.
"""

import pytest
import torch
import torch.nn as nn
from fixed_point import (
    FixedPointConfig,
    FixedPointBatchNorm,
    FixedPointLinear,
    FixedPointConv2d,
    FixedPointReLU,
    FixedPointAdaptiveAvgPool2d,
    convert_model_to_fixed_point,
)


class TestFixedPointConfig:
    """Test the configuration and quantization primitives."""

    def test_quantize_basic_values(self):
        """Quantize uses the paper's floor encoding, not round-to-nearest."""
        config = FixedPointConfig(total_bits=32, frac_bits=16)

        # 0.5 = 32768/65536 (exactly representable)
        x = torch.tensor([0.5])
        x_q = config.quantize(x)
        assert torch.isclose(x_q, torch.tensor([0.5], dtype=torch.float64), atol=1e-5)

        # Positive and negative non-grid values both use mathematical floor.
        x = torch.tensor([0.3])
        x_q = config.quantize(x)
        assert x_q.item() == torch.floor(x.double() * config.scale).item() / config.scale

        low_precision = FixedPointConfig(total_bits=8, frac_bits=4)
        assert low_precision.quantize(torch.tensor([-0.1])).item() == -0.125

    def test_quantize_wraps_in_ring(self):
        """Out-of-range values wrap in Z_(2^ell); they never saturate."""
        config = FixedPointConfig(total_bits=16, frac_bits=8)
        # floor(1000 * 256) mod 2^16 = 59392, decoded as -6144 / 256.
        assert config.quantize(torch.tensor([1000.0])).item() == -24.0

        small = FixedPointConfig(total_bits=8, frac_bits=4)
        # 7.5 + 1.0 = 8.5 wraps from residue 136 to signed residue -120.
        wrapped = small.add(torch.tensor([7.5]), torch.tensor([1.0]))
        assert wrapped.item() == -7.5

    def test_signed_floor_truncation(self):
        """Negative products use floor division, matching signedTruncate."""
        config = FixedPointConfig(
            total_bits=8, frac_bits=4, truncation_mode="exact")
        result = config.multiply(torch.tensor([-0.3]), torch.tensor([0.5]))
        # enc(-0.3)=-5, enc(0.5)=8; floor((-5*8)/16)=-3 -> -3/16.
        assert result.item() == -0.1875

    def test_average_pool_sums_in_the_ring_before_public_division(self):
        """Pooling must not decode/sum signed representatives outside the ring."""
        config = FixedPointConfig(total_bits=8, frac_bits=2)
        pool = FixedPointAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d((1, 1)), config)
        # Each 20.0 encodes to residue 80.  Four residues sum to 320, which
        # wraps to 64 before division by four and decoding (result 4.0).
        # Summing the signed representatives first would incorrectly return
        # the unwrapped average 20.0.
        x = torch.full((1, 1, 2, 2), 20.0)
        assert pool(x).item() == 4.0

    def test_relu_uses_ring_sign_bit(self):
        """ReLU follows DReLU's two's-complement sign test, including wrap."""
        config = FixedPointConfig(total_bits=8, frac_bits=4)
        relu = FixedPointReLU(config)
        x = torch.tensor([[1.0, -1.0, 7.5, 8.0]], dtype=torch.float64)
        # 8.0 encodes to residue 128, whose ring sign bit is set, so it is
        # treated as negative in Z_2^8 rather than being clamped as a float.
        expected = torch.tensor([[1.0, 0.0, 7.5, 0.0]], dtype=torch.float64)
        assert torch.equal(relu(x), expected)


class TestFixedPointLinear:
    """Test fixed-point linear layer with correct truncation semantics."""

    def test_multiplication_truncation(self):
        """
        REGRESSION TEST for bug where truncate() was applied in float domain.

        This test would have caught the original bug where:
          0.5 * 0.5 = 0.25 (float domain)
          truncate(0.25) incorrectly gave 0.0

        Correct behavior (integer domain):
          (0.5 * 2^16) * (0.5 * 2^16) = 32768 * 32768 = 1073741824
          floor(1073741824 / 2^16) = 16384
          16384 / 2^16 = 0.25  ✓
        """
        config = FixedPointConfig(total_bits=32, frac_bits=16)

        # Create 1x1 linear: y = 0.5 * x
        lin = FixedPointLinear(1, 1, config, bias=False)
        lin.linear.weight.data = torch.tensor([[0.5]])

        x = torch.tensor([[0.5]])
        y = lin(x)

        # Expected: 0.5 * 0.5 = 0.25 (with small quantization error)
        assert torch.isclose(y, torch.tensor([[0.25]], dtype=torch.float64), atol=1e-4), \
            f"Expected ~0.25, got {y.item():.6f}"

    def test_identity_linear_preserves_values(self):
        """Identity weight matrix should preserve input (within quantization error)."""
        config = FixedPointConfig(total_bits=32, frac_bits=16)

        lin = FixedPointLinear(4, 4, config, bias=False)
        lin.linear.weight.data = torch.eye(4)

        x = torch.tensor([[0.1, 0.25, -0.5, 0.75]])
        y = lin(x)

        # Should be close to input
        assert torch.allclose(y, x.double(), atol=1e-4), \
            f"Identity failed: input {x}, output {y}"

    def test_integer_conv_matches_hand_computation(self):
        """Conv2d uses integer accumulation followed by one signed truncation."""
        config = FixedPointConfig(total_bits=8, frac_bits=4)
        conv = FixedPointConv2d(1, 1, 2, config, bias=False)
        conv.conv.weight.data.fill_(0.5)
        x = torch.tensor([[[[0.25, -0.25], [0.5, -0.5]]]])
        # Encoded dot product: (4 - 4 + 8 - 8) * 8 = 0.
        assert conv(x).item() == 0.0


class TestModelConversion:
    """Test conversion of full models to fixed-point."""

    def test_convert_mlp_replaces_all_layers(self):
        """Converting MLP should replace Linear and ReLU with fixed-point equivalents."""
        from models import MLP

        config = FixedPointConfig(total_bits=32, frac_bits=16)
        model = MLP()
        fp_model = convert_model_to_fixed_point(model, config)

        # Check fc1, fc2, fc3 are FixedPointLinear
        assert isinstance(fp_model.fc1, FixedPointLinear), \
            f"fc1 should be FixedPointLinear, got {type(fp_model.fc1)}"
        assert isinstance(fp_model.fc2, FixedPointLinear)
        assert isinstance(fp_model.fc3, FixedPointLinear)

        # Check relu layers are FixedPointReLU
        assert isinstance(fp_model.relu1, FixedPointReLU)
        assert isinstance(fp_model.relu2, FixedPointReLU)

    def test_convert_resnet_folds_batchnorm(self):
        """BatchNorm must not remain a floating-point layer in the ring graph."""
        from models import ResNet20

        config = FixedPointConfig(total_bits=24, frac_bits=12)
        fp_model = convert_model_to_fixed_point(ResNet20().eval(), config)
        assert isinstance(fp_model.bn1, FixedPointBatchNorm)
        assert isinstance(fp_model.layer1[0].bn1, FixedPointBatchNorm)

    def test_convert_lstm_replaces_all_layers(self):
        """
        NEW FEATURE TEST (TDD): Converting LSTM should handle Embedding and LSTM layers.

        This test WILL FAIL initially because FixedPointEmbedding and FixedPointLSTM
        don't exist yet. This proves we need them.
        """
        from models import SimpleLSTM

        from fixed_point import FixedPointEmbedding, FixedPointLSTM

        config = FixedPointConfig(total_bits=32, frac_bits=16)
        model = SimpleLSTM(vocab_size=100, embed_dim=32, hidden_dim=32)
        fp_model = convert_model_to_fixed_point(model, config)

        assert isinstance(fp_model.embedding, FixedPointEmbedding), \
            f"embedding should be FixedPointEmbedding, got {type(fp_model.embedding)}"
        assert isinstance(fp_model.lstm, FixedPointLSTM), \
            f"lstm should be FixedPointLSTM, got {type(fp_model.lstm)}"
        assert isinstance(fp_model.fc, FixedPointLinear), \
            f"fc should be FixedPointLinear, got {type(fp_model.fc)}"

    def test_lstm_forward_matches_float_closely(self):
        """
        Fixed-point LSTM output should be close to float LSTM at high bitwidth.

        At ℓ=32, s=16, the fixed-point emulation should track float32 closely
        (small quantization error, not divergence).
        """
        from models import SimpleLSTM

        config = FixedPointConfig(total_bits=32, frac_bits=16)
        model = SimpleLSTM(vocab_size=100, embed_dim=32, hidden_dim=32)
        model.eval()

        # Fixed token sequence
        x = torch.randint(0, 100, (2, 10))

        with torch.no_grad():
            float_out = model(x)
            fp_model = convert_model_to_fixed_point(model, config)
            fp_model.eval()
            fp_out = fp_model(x)

        # At 32-bit, outputs should be close (logits within reasonable tolerance)
        max_err = (float_out - fp_out).abs().max().item()
        assert max_err < 0.5, \
            f"Fixed-point LSTM diverged from float: max_err={max_err:.4f}"

    def test_conversion_does_not_mutate_original(self):
        """
        REGRESSION TEST for bug where convert_model_to_fixed_point mutated
        the input model in place, causing second conversion to nest wrappers.

        The bug manifested as:
          First call: nn.Linear -> FixedPointLinear (correct)
          Second call: FixedPointLinear.linear -> FixedPointLinear (wrong!)

        Fix: deepcopy before conversion in precision_sweep().
        """
        from models import MLP

        config = FixedPointConfig(total_bits=32, frac_bits=16)
        original_model = MLP()

        # First conversion
        fp1 = convert_model_to_fixed_point(original_model, config)
        assert isinstance(fp1.fc1, FixedPointLinear)

        # Original should still be nn.Linear (not mutated)
        assert isinstance(original_model.fc1, nn.Linear), \
            "Original model was mutated! convert_model_to_fixed_point is destructive."

        # Second conversion of original should still work
        fp2 = convert_model_to_fixed_point(original_model, config)
        assert isinstance(fp2.fc1, FixedPointLinear)
        # Should not have nested: FixedPointLinear.linear should be nn.Linear
        assert isinstance(fp2.fc1.linear, nn.Linear), \
            "Nested conversion! fc1.linear should be nn.Linear, not FixedPointLinear"


class TestRegressionBugs:
    """
    Regression tests for bugs already fixed.

    These tests pass now, but document the bugs and prevent reintroduction.
    """

    def test_truncate_zero_bug_fixed(self):
        """
        Documents the bug where truncate(0.25) returned 0.0 instead of 0.25.

        Root cause: truncate() applied in float domain instead of integer domain.
        """
        config = FixedPointConfig(total_bits=32, frac_bits=16)

        result = config.multiply(torch.tensor([0.5]), torch.tensor([0.5]))

        assert torch.isclose(result, torch.tensor([0.25], dtype=torch.float64), atol=1e-5), \
            "Truncation bug reintroduced!"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
