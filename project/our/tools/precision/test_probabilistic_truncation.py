import torch

from fixed_point import FixedPointConfig


def test_default_truncation_is_exact_signed_floor():
    config = FixedPointConfig(total_bits=8, frac_bits=2)
    accumulator = torch.tensor([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4])
    expected = torch.remainder(
        torch.div(accumulator, 4, rounding_mode="floor"), 256)
    assert config.truncation_mode == "exact"
    assert torch.equal(config.truncate_accumulator(accumulator), expected)


def test_probabilistic_truncation_is_stochastic_with_one_ulp_error():
    config = FixedPointConfig(
        total_bits=8,
        frac_bits=2,
        truncation_mode="probabilistic",
        seed=12345,
    )
    # The low two bits are 1, so stochastic truncation rounds up with
    # probability 1/4.  A vector lets us observe the Bernoulli source without
    # making the model-level tests flaky.
    accumulator = torch.full((4096,), 1, dtype=torch.int64)
    output = config.truncate_accumulator(accumulator)
    assert torch.all((output == 0) | (output == 1))
    rounded_up = int((output == 1).sum().item())
    assert 700 <= rounded_up <= 1300


def test_exact_mode_remains_available_as_a_reference_variant():
    config = FixedPointConfig(
        total_bits=8,
        frac_bits=2,
        truncation_mode="exact",
    )
    accumulator = torch.tensor([-1, 1, 2, 3, 4], dtype=torch.int64)
    assert torch.equal(
        config.truncate_accumulator(accumulator),
        torch.tensor([255, 0, 0, 0, 1], dtype=torch.int64),
    )


def test_probabilistic_mode_handles_negative_residues():
    config = FixedPointConfig(
        total_bits=8,
        frac_bits=2,
        truncation_mode="probabilistic",
        seed=7,
    )
    accumulator = torch.tensor([-7, -6, -5, -4, -3, -2, -1], dtype=torch.int64)
    signed_output = config.to_signed(config.truncate_accumulator(accumulator))
    floor_output = torch.div(accumulator, 4, rounding_mode="floor")
    # The omitted low-bit correction is a Bernoulli carry, including for
    # negative two's-complement values.  Every result is floor or floor+1.
    assert torch.all(
        (signed_output == floor_output)
        | (signed_output == floor_output + 1)
    )


def test_probabilistic_mode_is_reproducible_with_an_injected_seed():
    accumulator = torch.tensor([1, 2, 3, -1, -2, -3], dtype=torch.int64)
    first = FixedPointConfig(8, 2, truncation_mode="probabilistic", seed=91)
    second = FixedPointConfig(8, 2, truncation_mode="probabilistic", seed=91)
    assert torch.equal(
        first.truncate_accumulator(accumulator),
        second.truncate_accumulator(accumulator),
    )
