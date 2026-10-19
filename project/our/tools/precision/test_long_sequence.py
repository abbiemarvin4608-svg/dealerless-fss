#!/usr/bin/env python3
"""
Test whether long-sequence RNN requires ℓ > 20 bits.

The original E0 used seq_len=20. This script tests seq_len ∈ {50, 100, 200, 400}
to stress the LSTM's cell-state accumulation across many timesteps, where truncation
errors compound.

Hypothesis: With seq_len=20, ℓ=20 works. With seq_len=200+, truncation error
accumulation will require ℓ≥24 or higher to maintain accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from models import SimpleLSTM
from fixed_point import FixedPointConfig, convert_model_to_fixed_point


def generate_sequence_sum_task(num_samples, seq_len, vocab_size=100, seed=0):
    """
    Generate synthetic sequence-classification dataset.

    Task: label = 1 iff sum of token values exceeds median sum.
    Forces LSTM to accumulate across timesteps in cell state.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, vocab_size, (num_samples, seq_len), generator=g)
    sums = X.sum(dim=1)
    threshold = sums.median()
    y = (sums > threshold).long()
    return X, y


def evaluate_lstm(model, seq_len, device='cpu', vocab_size=100, num_samples=2000):
    """Evaluate LSTM on synthetic task with given sequence length."""
    X, y = generate_sequence_sum_task(num_samples, seq_len, vocab_size, seed=9999)
    test_loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    device = 'cpu'
    vocab_size = 100
    embed_dim = 32
    hidden_dim = 32

    # Load trained LSTM (trained on seq_len=20)
    print("Loading trained SimpleLSTM...")
    lstm_path = Path('models/lstm_synthetic.pth')
    model = SimpleLSTM(vocab_size=vocab_size, embed_dim=embed_dim,
                       hidden_dim=hidden_dim, num_classes=2).to(device)
    model.load_state_dict(torch.load(lstm_path, map_location=device))
    model.eval()

    seq_lengths = [20, 50, 100, 200, 400]
    bitwidths = [12, 16, 20, 24, 28, 32]
    frac_bits = 16

    print("\n" + "="*80)
    print("Long-Sequence LSTM Precision Test")
    print("="*80)
    print(f"Vocab size: {vocab_size}, Embed dim: {embed_dim}, Hidden dim: {hidden_dim}")
    print(f"Fractional bits fixed at s={frac_bits}")
    print(f"Testing sequence lengths: {seq_lengths}")
    print(f"Testing bitwidths ℓ: {bitwidths}")
    print("="*80)

    results = []

    # Baseline: float32 across all sequence lengths
    print("\nFloat32 baseline:")
    for seq_len in seq_lengths:
        acc = evaluate_lstm(model, seq_len, device, vocab_size)
        results.append({
            'seq_len': seq_len,
            'ell': 'float32',
            'int_bits': '-',
            'frac_bits': '-',
            'accuracy': acc
        })
        print(f"  seq_len={seq_len:3d}: {acc:.2f}%")

    # Fixed-point sweep
    print("\nFixed-point sweep:")
    for seq_len in seq_lengths:
        print(f"\nSequence length = {seq_len}")
        for ell in bitwidths:
            int_bits = ell - frac_bits
            config = FixedPointConfig(total_bits=ell, frac_bits=frac_bits)
            fp_model = convert_model_to_fixed_point(model, config)
            fp_model.to(device)

            acc = evaluate_lstm(fp_model, seq_len, device, vocab_size)
            results.append({
                'seq_len': seq_len,
                'ell': ell,
                'int_bits': int_bits,
                'frac_bits': frac_bits,
                'accuracy': acc
            })
            print(f"  ℓ={ell:2d} (int={int_bits}, s={frac_bits}): {acc:.2f}%")

    # Summary table
    print("\n" + "="*80)
    print("Summary: Accuracy (%) for (seq_len, ℓ) grid")
    print("="*80)

    # Header
    header = 'seq_len \\ ℓ'
    print(f"{header:<12}", end='')
    for ell in ['float32'] + bitwidths:
        print(f"{str(ell):>8}", end='')
    print()
    print("-" * 90)

    # Rows
    for seq_len in seq_lengths:
        print(f"{seq_len:<12}", end='')
        for ell in ['float32'] + bitwidths:
            match = [r for r in results if r['seq_len'] == seq_len and r['ell'] == ell]
            if match:
                acc = match[0]['accuracy']
                print(f"{acc:>7.1f}%", end='')
            else:
                print(f"{'---':>8}", end='')
        print()

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print("Expected behavior:")
    print("  - If ℓ=20 remains sufficient across all seq_len → truncation error doesn't")
    print("    accumulate significantly (LSTM gates may reset error)")
    print("  - If accuracy drops at longer seq_len for ℓ=20 but recovers at ℓ≥24 →")
    print("    this proves longer sequences require higher precision")
    print()
    print("Key question: Does the cliff move upward (require ℓ>20) as seq_len increases?")
    print("="*80)

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    output_path = Path('results/e0_long_sequence.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
