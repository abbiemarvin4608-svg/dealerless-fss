#!/usr/bin/env python3
"""
Diagnostic script to understand the ℓ=20 precision cliff.

Measures:
1. Activation ranges across ResNet20 layers to understand integer-bit requirements
2. 2D sweep across (integer_bits, fractional_bits) to decouple the two dimensions
3. Identify whether the cliff is driven by integer overflow or fractional precision

Usage: python3 diagnose_precision.py
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path

from models import ResNet20
from fixed_point import FixedPointConfig, convert_model_to_fixed_point


def measure_activation_ranges(model, device='cpu', num_batches=10):
    """Measure min/max activation values across layers."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    activations = {}
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            if name not in activations:
                activations[name] = {'min': float('inf'), 'max': float('-inf')}
            if isinstance(output, tuple):
                output = output[0]
            activations[name]['min'] = min(activations[name]['min'], output.min().item())
            activations[name]['max'] = max(activations[name]['max'], output.max().item())
        return fn

    # Register hooks on key layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            model(inputs)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return activations


def sweep_2d_precision(model, int_bits_range, frac_bits_range, device='cpu'):
    """Sweep across (integer_bits, fractional_bits) grid."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    results = []

    for int_bits in int_bits_range:
        for frac_bits in frac_bits_range:
            total_bits = int_bits + frac_bits
            if total_bits < 8 or total_bits > 64:
                continue

            print(f"Testing int_bits={int_bits}, frac_bits={frac_bits} (ℓ={total_bits})...")
            config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
            fp_model = convert_model_to_fixed_point(model, config)
            fp_model.to(device)
            fp_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = fp_model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            accuracy = 100.0 * correct / total
            results.append({
                'int_bits': int_bits,
                'frac_bits': frac_bits,
                'total_bits': total_bits,
                'accuracy': accuracy
            })
            print(f"  → Accuracy: {accuracy:.2f}%")

    return results


def main():
    device = 'cpu'

    # Load trained ResNet20
    print("Loading ResNet20...")
    resnet_path = Path('models/resnet20_cifar10.pth')
    model = ResNet20().to(device)
    model.load_state_dict(torch.load(resnet_path, map_location=device))
    model.eval()

    # 1. Measure activation ranges
    print("\n" + "="*70)
    print("STEP 1: Measuring activation ranges")
    print("="*70)
    act_ranges = measure_activation_ranges(model, device, num_batches=10)

    print("\nActivation ranges (float32 model):")
    print(f"{'Layer':<40} {'Min':>12} {'Max':>12} {'Range':>12}")
    print("-" * 78)
    for name, stats in sorted(act_ranges.items()):
        range_val = stats['max'] - stats['min']
        print(f"{name:<40} {stats['min']:>12.4f} {stats['max']:>12.4f} {range_val:>12.4f}")

    # Find max absolute value to determine integer bit requirement
    max_abs = max(max(abs(s['min']), abs(s['max'])) for s in act_ranges.values())
    min_int_bits = int(max_abs).bit_length() + 1  # +1 for sign bit
    print(f"\n→ Max absolute activation: {max_abs:.4f}")
    print(f"→ Minimum integer bits needed: {min_int_bits} (to represent range [{-2**(min_int_bits-1)}, {2**(min_int_bits-1)}))")

    # 2. 2D sweep
    print("\n" + "="*70)
    print("STEP 2: 2D precision sweep")
    print("="*70)
    print("Sweeping (integer_bits, fractional_bits) to decouple the two dimensions\n")

    # Test critical region around the cliff
    int_bits_range = [2, 3, 4, 5, 6, 8]
    frac_bits_range = [12, 14, 16, 18, 20]

    results = sweep_2d_precision(model, int_bits_range, frac_bits_range, device)

    # Display as table
    print("\n" + "="*70)
    print("Results: Accuracy (%) for (int_bits, frac_bits) grid")
    print("="*70)
    label = 'int_bits \\ frac_bits'
    print(f"{label:<20}", end='')
    for fb in frac_bits_range:
        print(f"{fb:>8}", end='')
    print(f"{'ℓ=total':>12}")
    print("-" * 90)

    for ib in int_bits_range:
        print(f"{ib:<20}", end='')
        for fb in frac_bits_range:
            match = [r for r in results if r['int_bits'] == ib and r['frac_bits'] == fb]
            if match:
                acc = match[0]['accuracy']
                ell = match[0]['total_bits']
                print(f"{acc:>7.1f}%", end='')
            else:
                print(f"{'---':>8}", end='')
        # Show total bits for this row (using middle frac_bits value)
        mid_fb = frac_bits_range[len(frac_bits_range)//2]
        print(f"  ({ib}+{mid_fb}={ib+mid_fb})")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"The original E0 experiment fixed s={16} and varied ℓ.")
    print(f"With s=16, the cliff at ℓ=20 corresponds to int_bits={20-16}=4.")
    print(f"\nThis 2D sweep shows whether the cliff is driven by:")
    print(f"  - Integer overflow (vertical axis): need ≥4 integer bits")
    print(f"  - Fractional precision (horizontal axis): need ≥16 fractional bits")
    print(f"\nIf accuracy is high across the frac_bits={16} column for int_bits≥4,")
    print(f"then the cliff is purely integer-driven, NOT a precision requirement.")


if __name__ == '__main__':
    main()
