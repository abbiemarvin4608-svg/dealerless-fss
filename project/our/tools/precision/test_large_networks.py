#!/usr/bin/env python3
"""
Test whether larger networks (ResNet50/101/152) require ℓ > 20 bits.

Strategy:
1. Load torchvision pretrained models (no ImageNet needed for activation range)
2. Measure activation ranges with random inputs (representative of real data stats)
3. If max_activation >> 8, then ℓ=20 (4 int bits) is insufficient
4. Run precision sweep to find the actual cliff

Usage: python3 test_large_networks.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from fixed_point import FixedPointConfig, convert_model_to_fixed_point


def measure_activation_ranges(model, input_shape=(3, 224, 224), num_samples=100, device='cpu'):
    """
    Measure activation ranges across model layers.

    Uses random inputs with ImageNet-like statistics (mean=0, std=1 after normalization).
    """
    print(f"  Measuring activation ranges on {num_samples} random samples...")

    activations = {}
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            if name not in activations:
                activations[name] = {'min': float('inf'), 'max': float('-inf'), 'count': 0}
            if isinstance(output, tuple):
                output = output[0]
            activations[name]['min'] = min(activations[name]['min'], output.min().item())
            activations[name]['max'] = max(activations[name]['max'], output.max().item())
            activations[name]['count'] += 1
        return fn

    # Register hooks on key layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            # Random input with ImageNet statistics
            x = torch.randn(1, *input_shape, device=device)
            model(x)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return activations


def analyze_activation_ranges(act_ranges, model_name):
    """Analyze activation ranges and determine minimum integer bits needed."""
    print(f"\n{'Layer':<50} {'Min':>10} {'Max':>10} {'Range':>10}")
    print("-" * 82)

    max_abs = 0
    critical_layers = []

    for name, stats in sorted(act_ranges.items()):
        min_val = stats['min']
        max_val = stats['max']
        range_val = max_val - min_val
        abs_max = max(abs(min_val), abs(max_val))

        # Print top layers by absolute magnitude
        if abs_max > 5:  # Threshold for interesting layers
            print(f"{name:<50} {min_val:>10.2f} {max_val:>10.2f} {range_val:>10.2f}")
            max_abs = max(max_abs, abs_max)
            critical_layers.append((name, abs_max))

    # Compute required integer bits
    import math
    if max_abs > 0:
        # Need to represent [-max_abs, max_abs] in 2's complement
        # 2^(int_bits-1) must be > max_abs
        min_int_bits = math.ceil(math.log2(max_abs)) + 1  # +1 for sign bit
    else:
        min_int_bits = 1

    print("\n" + "="*82)
    print(f"{model_name} Summary:")
    print(f"  Max absolute activation: {max_abs:.2f}")
    print(f"  Minimum integer bits required: {min_int_bits}")
    print(f"  With s=16 fractional bits: ℓ_min = {min_int_bits + 16}")
    print(f"  Current ℓ=20 (4 int bits) gives range: [-8, 8)")

    if max_abs > 8:
        print(f"  → CONCLUSION: ℓ=20 INSUFFICIENT (activations exceed ±8)")
        print(f"  → Recommend ℓ ≥ {min_int_bits + 16} for faithful representation")
    else:
        print(f"  → ℓ=20 may be sufficient (activations within ±8)")

    print("="*82)

    return max_abs, min_int_bits


def precision_sweep_imagenet_model(model, model_name, bitwidths, frac_bits=16, num_samples=100, device='cpu'):
    """
    Sweep bitwidths on a pretrained ImageNet model.

    Uses random inputs (no real ImageNet needed). Measures whether the model
    produces consistent outputs across bitwidths.
    """
    print(f"\n{model_name}: Precision sweep with random inputs")
    print(f"  (Output consistency test, not accuracy)")

    # Generate test inputs
    inputs = [torch.randn(1, 3, 224, 224, device=device) for _ in range(num_samples)]

    # Get float32 baseline outputs
    model.eval()
    float_outputs = []
    with torch.no_grad():
        for x in inputs:
            out = model(x)
            float_outputs.append(out)

    results = []

    for ell in bitwidths:
        print(f"  Testing ℓ={ell}, s={frac_bits}...")
        config = FixedPointConfig(total_bits=ell, frac_bits=frac_bits)
        fp_model = convert_model_to_fixed_point(model, config)
        fp_model.to(device)
        fp_model.eval()

        # Measure output divergence from float
        total_rel_error = 0.0
        total_top1_match = 0

        with torch.no_grad():
            for i, x in enumerate(inputs):
                fp_out = fp_model(x)
                float_out = float_outputs[i]

                # Relative error in logits
                rel_error = torch.abs(fp_out - float_out) / (torch.abs(float_out) + 1e-8)
                total_rel_error += rel_error.mean().item()

                # Top-1 agreement
                if fp_out.argmax() == float_out.argmax():
                    total_top1_match += 1

        avg_rel_error = total_rel_error / num_samples
        top1_agreement = 100.0 * total_top1_match / num_samples

        results.append({
            'model': model_name,
            'ell': ell,
            'int_bits': ell - frac_bits,
            'frac_bits': frac_bits,
            'mean_rel_error': avg_rel_error,
            'top1_agreement_pct': top1_agreement
        })

        print(f"    Mean relative error: {avg_rel_error:.4f}, Top-1 agreement: {top1_agreement:.1f}%")

    return results


def main():
    device = 'cpu'
    bitwidths = [16, 20, 24, 28, 32]
    frac_bits = 16

    print("="*82)
    print("Large Network Precision Study")
    print("="*82)
    print("Goal: Determine if ResNet50/101/152 require ℓ > 20 bits")
    print("Strategy: Measure activation ranges → compare to ℓ=20's ±8 range")
    print("="*82)

    models_to_test = [
        ('ResNet50', models.resnet50(pretrained=True)),
        ('ResNet101', models.resnet101(pretrained=True)),
        ('ResNet152', models.resnet152(pretrained=True)),
    ]

    all_results = []

    for model_name, model in models_to_test:
        print(f"\n{'='*82}")
        print(f"Testing: {model_name}")
        print(f"{'='*82}")

        model.to(device)
        model.eval()

        # Step 1: Measure activation ranges
        act_ranges = measure_activation_ranges(model, num_samples=50, device=device)
        max_abs, min_int_bits = analyze_activation_ranges(act_ranges, model_name)

        # Step 2: Precision sweep if activation range is interesting
        if max_abs > 5:  # Worth testing
            sweep_results = precision_sweep_imagenet_model(
                model, model_name, bitwidths, frac_bits, num_samples=50, device=device
            )
            all_results.extend(sweep_results)

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        output_path = Path('results/e0_large_networks.csv')
        df.to_csv(output_path, index=False)
        print(f"\n{'='*82}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*82}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
