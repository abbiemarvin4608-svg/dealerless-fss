#!/usr/bin/env python3
"""
E0: Precision study for dealerless FSS paper.

Trains (or loads) MLP/ResNet20/SimpleLSTM models, then sweeps bitwidth
ℓ ∈ {12, 16, 20, 24, 28, 32} with s=16 fractional bits to demonstrate
the 20-bit precision cliff where accuracy degrades sharply.

Output:
  - results/e0_precision.csv: (model, ell, accuracy) rows
  - results/e0_precision.png: accuracy vs bitwidth curves

Usage:
  python run_e0.py [--train] [--epochs N]

  --train: train from scratch (default: use pretrained if available)
  --epochs N: training epochs (default: MLP=10, ResNet=40, LSTM=5)
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import MLP, ResNet20, SimpleLSTM
from fixed_point import FixedPointConfig, convert_model_to_fixed_point
from results_io import merge_results


def train_mnist_mlp(epochs=10, device='cpu'):
    """Train MLP on MNIST."""
    print(f"Training MLP on MNIST for {epochs} epochs...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model


def train_cifar_resnet(epochs=40, device='cpu'):
    """Train ResNet20 on CIFAR-10."""
    print(f"Training ResNet20 on CIFAR-10 for {epochs} epochs...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    return model


def evaluate_mnist(model, device='cpu'):
    """Evaluate model on MNIST test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_cifar(model, device='cpu'):
    """Evaluate model on CIFAR-10 test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

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


# ============================================================================
# LSTM on a synthetic sequence-classification task
# ============================================================================
# We use a synthetic "sum-threshold" task instead of a downloaded corpus
# (IMDB etc.) for speed and full reproducibility. Each example is a length-L
# sequence of integer tokens in [0, vocab_size). The label is 1 if the sum of
# the tokens exceeds the median sum, else 0. This forces the LSTM to ACCUMULATE
# token values across timesteps in its cell state, which is exactly the
# recurrent-accumulation behaviour that stresses fixed-point precision.

# Fixed hyperparameters for the synthetic task (shared by train + eval so the
# fixed-point sweep evaluates on the same held-out test set).
_SEQ_LEN = 20
_VOCAB_SIZE = 100
_N_TRAIN = 8000
_N_TEST = 2000


def generate_sequence_sum_task(num_samples, seq_len=_SEQ_LEN, vocab_size=_VOCAB_SIZE, seed=0):
    """
    Generate a synthetic sequence-classification dataset.

    Task: label = 1 iff the sum of the token values in the sequence exceeds the
    median sum across the batch (so classes are ~balanced). The signal lives in
    the accumulated sum, which an LSTM must track through its recurrence.

    Args:
        num_samples: number of sequences to generate.
        seq_len: length of each sequence.
        vocab_size: token vocabulary size (tokens in [0, vocab_size)).
        seed: RNG seed for reproducibility.

    Returns:
        (X, y): X is (num_samples, seq_len) long tensor of tokens,
                y is (num_samples,) long tensor of binary labels.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, vocab_size, (num_samples, seq_len), generator=g)
    sums = X.sum(dim=1)
    threshold = sums.median()
    y = (sums > threshold).long()
    return X, y


def train_synthetic_lstm(vocab_size=_VOCAB_SIZE, embed_dim=32, hidden_dim=32,
                         epochs=5, device='cpu'):
    """Train SimpleLSTM on the synthetic sum-threshold sequence task."""
    print(f"Training SimpleLSTM on synthetic sequence task for {epochs} epochs...")
    X, y = generate_sequence_sum_task(_N_TRAIN, _SEQ_LEN, vocab_size, seed=42)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

    model = SimpleLSTM(vocab_size=vocab_size, embed_dim=embed_dim,
                       hidden_dim=hidden_dim, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, train_acc={train_acc:.2f}%")

    return model


def evaluate_synthetic_lstm(model, device='cpu', vocab_size=_VOCAB_SIZE):
    """Evaluate a (float or fixed-point) LSTM on the synthetic test set."""
    X, y = generate_sequence_sum_task(_N_TEST, _SEQ_LEN, vocab_size, seed=1234)
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


def precision_sweep(model_name, float_model, eval_fn, bitwidths, frac_bits=16, device='cpu'):
    """Sweep bitwidths and measure accuracy degradation."""
    results = []

    # Baseline: float32 accuracy
    print(f"\n{model_name}: Evaluating float32 baseline...")
    float_acc = eval_fn(float_model, device)
    print(f"  Float32 accuracy: {float_acc:.2f}%")
    results.append({'model': model_name, 'ell': 'float32', 'accuracy': float_acc})

    # Fixed-point sweep
    for ell in bitwidths:
        print(f"\n{model_name}: Evaluating ℓ={ell}, s={frac_bits}...")
        config = FixedPointConfig(total_bits=ell, frac_bits=frac_bits)
        # convert_model_to_fixed_point is non-destructive (deep-copies internally)
        fp_model = convert_model_to_fixed_point(float_model, config)
        fp_model.to(device)

        acc = eval_fn(fp_model, device)
        print(f"  ℓ={ell}: accuracy={acc:.2f}%")
        results.append({'model': model_name, 'ell': ell, 'accuracy': acc})

    return results


def main():
    parser = argparse.ArgumentParser(description='E0: Precision study')
    parser.add_argument('--train', action='store_true', help='Train from scratch')
    parser.add_argument('--epochs-mlp', type=int, default=10, help='MLP training epochs')
    parser.add_argument('--epochs-resnet', type=int, default=40, help='ResNet training epochs')
    parser.add_argument('--epochs-lstm', type=int, default=8, help='LSTM training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--models', type=str, default='mlp,resnet',
                        help='Comma-separated models to run: mlp,resnet,lstm')
    args = parser.parse_args()

    selected = {m.strip().lower() for m in args.models.split(',') if m.strip()}

    device = args.device
    bitwidths = [12, 16, 20, 24, 28, 32]
    frac_bits = 16

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    all_results = []

    # ========================================================================
    # MLP on MNIST
    # ========================================================================
    if 'mlp' in selected:
        mlp_path = models_dir / 'mlp_mnist.pth'
        if args.train or not mlp_path.exists():
            mlp_model = train_mnist_mlp(epochs=args.epochs_mlp, device=device)
            torch.save(mlp_model.state_dict(), mlp_path)
            print(f"Saved trained MLP to {mlp_path}")
        else:
            print(f"Loading pretrained MLP from {mlp_path}")
            mlp_model = MLP().to(device)
            mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))

        mlp_results = precision_sweep('MLP', mlp_model, evaluate_mnist, bitwidths, frac_bits, device)
        all_results.extend(mlp_results)

    # ========================================================================
    # ResNet20 on CIFAR-10
    # ========================================================================
    if 'resnet' in selected:
        resnet_path = models_dir / 'resnet20_cifar10.pth'
        if args.train or not resnet_path.exists():
            resnet_model = train_cifar_resnet(epochs=args.epochs_resnet, device=device)
            torch.save(resnet_model.state_dict(), resnet_path)
            print(f"Saved trained ResNet20 to {resnet_path}")
        else:
            print(f"Loading pretrained ResNet20 from {resnet_path}")
            resnet_model = ResNet20().to(device)
            resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))

        resnet_results = precision_sweep('ResNet20', resnet_model, evaluate_cifar, bitwidths, frac_bits, device)
        all_results.extend(resnet_results)

    # ========================================================================
    # SimpleLSTM on synthetic sequence-classification task
    # ========================================================================
    if 'lstm' in selected:
        lstm_path = models_dir / 'lstm_synthetic.pth'
        if args.train or not lstm_path.exists():
            lstm_model = train_synthetic_lstm(
                vocab_size=100, embed_dim=32, hidden_dim=32,
                epochs=args.epochs_lstm, device=device)
            torch.save(lstm_model.state_dict(), lstm_path)
            print(f"Saved trained LSTM to {lstm_path}")
        else:
            print(f"Loading pretrained LSTM from {lstm_path}")
            lstm_model = SimpleLSTM(vocab_size=100, embed_dim=32, hidden_dim=32).to(device)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))

        lstm_results = precision_sweep('LSTM', lstm_model, evaluate_synthetic_lstm, bitwidths, frac_bits, device)
        all_results.extend(lstm_results)

    # ========================================================================
    # Save results (MERGE mode: preserves other models' data)
    # ========================================================================
    csv_path = results_dir / 'e0_precision.csv'
    merge_results(csv_path, all_results)
    print(f"\nResults merged into {csv_path}")

    # Display current state
    df = pd.read_csv(csv_path)
    print(df.to_string(index=False))

    # ========================================================================
    # Plot
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in df['model'].unique():
        if model_name in ['MLP', 'ResNet20', 'LSTM']:
            model_df = df[df['model'] == model_name]
            # Exclude float32 from plot (for numeric x-axis)
            model_df_numeric = model_df[model_df['ell'] != 'float32'].copy()
            if len(model_df_numeric) > 0:
                model_df_numeric['ell'] = model_df_numeric['ell'].astype(int)
                ax.plot(model_df_numeric['ell'], model_df_numeric['accuracy'],
                        marker='o', label=model_name, linewidth=2)

    ax.axvline(x=20, color='red', linestyle='--', linewidth=1, alpha=0.7, label='ℓ=20 (precision cliff)')
    ax.set_xlabel('Bitwidth ℓ (fractional bits s=16)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('E0: Fixed-Point Precision vs Bitwidth', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = results_dir / 'e0_precision.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
