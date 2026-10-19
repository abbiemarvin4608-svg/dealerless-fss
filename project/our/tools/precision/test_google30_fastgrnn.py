#!/usr/bin/env python3
"""
P0-Google30: FastGRNN precision test on Google Speech Commands (30-class).

Uses the official EdgeML FastGRNN implementation to reproduce SiRNN's Google-30
benchmark. Tests whether real speech data requires ℓ > 20 bits for RNN models.

Model: FastGRNN (from Microsoft EdgeML)
Dataset: Speech Commands v0.02, 30-class setup (LABELMAP30)
Precision sweep: ℓ ∈ {16, 20, 24, 28, 32} with s=16 fractional bits
"""

import wave
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from python_speech_features import fbank

try:
    from edgeml_pytorch.graph.rnn import FastGRNNCell
except ImportError:  # Optional dependency; the local baseline has no EdgeML dependency.
    FastGRNNCell = None
from fixed_point import FixedPointConfig, FixedPointLinear


# Google-30 label map (matches EdgeML's process_google.py LABELMAP30)
LABELMAP30 = {
    '_background_noise_': 0, 'bed': 1, 'bird': 2,
    'cat': 3, 'dog': 4, 'down': 5, 'eight': 6,
    'five': 7, 'four': 8, 'go': 9, 'happy': 10,
    'house': 11, 'left': 12, 'marvin': 13, 'nine': 14,
    'no': 15, 'off': 16, 'on': 17, 'one': 18,
    'right': 19, 'seven': 20, 'sheila': 21, 'six': 22,
    'stop': 23, 'three': 24, 'tree': 25, 'two': 26,
    'up': 27, 'wow': 28, 'yes': 29, 'zero': 30
}
NUM_CLASSES = 31


def extract_fbank_features(wav_path, n_filt=32):
    """Extract log-mel filterbank features (matches EdgeML preprocessing)."""
    with wave.open(str(wav_path), 'rb') as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # Log-mel filterbank
    fbank_feat, _ = fbank(audio, samplerate=rate, nfilt=n_filt)
    return fbank_feat  # (T, n_filt)


class GoogleSpeech30Dataset(Dataset):
    """Google Speech Commands 30-class dataset."""

    def __init__(self, root, split='train'):
        self.root = Path(root)
        self.split = split

        # Load official splits
        val_files = self._load_split_file('validation_list.txt')
        test_files = self._load_split_file('testing_list.txt')

        samples = []

        # Collect all words
        for word_dir in self.root.iterdir():
            if not word_dir.is_dir():
                continue

            word = word_dir.name
            if word not in LABELMAP30:
                continue  # Skip words not in 30-class setup

            label = LABELMAP30[word]

            for wav in word_dir.glob('*.wav'):
                rel_path = f"{word}/{wav.name}"

                if rel_path in test_files:
                    if split == 'test':
                        samples.append((wav, label))
                elif rel_path in val_files:
                    if split == 'val':
                        samples.append((wav, label))
                else:
                    if split == 'train':
                        samples.append((wav, label))

        self.samples = samples
        print(f"  {split}: {len(samples)} samples")

    def _load_split_file(self, fname):
        fpath = self.root / fname
        if not fpath.exists():
            return set()
        return set(fpath.read_text().strip().split('\n'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feats = extract_fbank_features(path, n_filt=32)
        x = torch.from_numpy(feats).float()
        return x, label


def collate_pad(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs])
    max_len = lengths.max().item()

    xs_padded = torch.zeros(len(xs), max_len, xs[0].size(1))
    for i, x in enumerate(xs):
        xs_padded[i, :x.size(0)] = x

    ys = torch.tensor(ys, dtype=torch.long)
    return xs_padded, ys, lengths


class FastGRNNClassifier(nn.Module):
    """FastGRNN-based classifier for Google-30."""

    def __init__(self, input_size=32, hidden_size=128, num_classes=31):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = FastGRNNCell(
            input_size, hidden_size,
            gate_nonlinearity='sigmoid',
            update_nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Run FastGRNN cell over sequence
        for t in range(seq_len):
            h = self.rnn_cell(x[:, t, :], h)

        # Use final hidden state for classification
        logits = self.fc(h)
        return logits


class FixedPointFastGRNNCell(nn.Module):
    """
    Fixed-point FastGRNN cell with MPC truncation semantics.

    Re-implements the official FastGRNNCell recurrence in fixed point:
      z     = sigmoid(x@W + h@U + bias_gate)
      c     = tanh(x@W + h@U + bias_update)
      h_new = z*h + (sigmoid(zeta)*(1-z) + sigmoid(nu)) * c

    Every matmul and elementwise product truncates the low s bits, modeling the
    truncation that follows each multiplication in a real MPC protocol. This
    mirrors FixedPointLSTM in fixed_point.py.
    """

    def __init__(self, cell, config):
        super().__init__()
        self.cell = cell
        self.config = config

    def _mm_trunc(self, a, b):
        """Fixed-point matmul with integer-domain truncation."""
        cfg = self.config
        a_int = torch.round(cfg.quantize(a) * cfg.scale)
        b_int = torch.round(cfg.quantize(b) * cfg.scale)
        y_int = a_int @ b_int
        return torch.floor(y_int / cfg.scale) / cfg.scale

    def forward(self, input, state):
        cfg = self.config
        cell = self.cell

        # Input->gate and hidden->gate matmuls (weights already quantized)
        wComp = self._mm_trunc(input, cell.W)
        uComp = self._mm_trunc(state, cell.U)
        pre_comp = wComp + uComp

        # Gate/update nonlinearities (LUT/piecewise-poly in real MPC;
        # here computed in float then re-quantized, as in FixedPointLSTM)
        z = cfg.quantize(torch.sigmoid(pre_comp + cell.bias_gate))
        c = cfg.quantize(torch.tanh(pre_comp + cell.bias_update))

        # h_new = z*h + (sigmoid(zeta)*(1-z) + sigmoid(nu)) * c
        # Each elementwise product truncates.
        zeta = cfg.quantize(torch.sigmoid(cell.zeta))
        nu = cfg.quantize(torch.sigmoid(cell.nu))

        z_h = cfg.truncate_elementwise(z, state)
        one_minus_z = cfg.quantize(1.0 - z)
        zeta_term = cfg.truncate_elementwise(zeta.expand_as(one_minus_z), one_minus_z)
        gate_c = cfg.quantize(zeta_term + nu)
        gc_c = cfg.truncate_elementwise(gate_c, c)

        return cfg.quantize(z_h + gc_c)


def convert_fastgrnn_to_fixed_point(model, config):
    """Convert FastGRNN model to fixed-point (non-destructive)."""
    import copy
    fp_model = copy.deepcopy(model)

    # Quantize all weights to the fixed-point grid
    for param in fp_model.rnn_cell.parameters():
        param.data = config.quantize(param.data)
    for param in fp_model.fc.parameters():
        param.data = config.quantize(param.data)

    # Wrap the cell so the recurrence runs with truncation
    fp_model.rnn_cell = FixedPointFastGRNNCell(fp_model.rnn_cell, config)

    # Final linear layer also needs fixed-point truncation
    fp_fc = FixedPointLinear(
        fp_model.fc.in_features, fp_model.fc.out_features,
        config, bias=fp_model.fc.bias is not None)
    fp_fc.linear.weight.data = fp_model.fc.weight.data.clone()
    if fp_model.fc.bias is not None:
        fp_fc.linear.bias.data = fp_model.fc.bias.data.clone()
    fp_model.fc = fp_fc

    return fp_model


def train_model(train_loader, val_loader, device='cpu', epochs=30):
    """Train FastGRNN classifier."""
    if FastGRNNCell is None:
        raise RuntimeError(
            "EdgeML is required for the official Google-30 training path; "
            "use Baselines/models.py for the dependency-free reference model."
        )
    model = FastGRNNClassifier(input_size=32, hidden_size=128, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"  Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xs, ys, lengths in train_loader:
            xs, ys = xs.to(device), ys.to(device)

            optimizer.zero_grad()
            outputs = model(xs, lengths)
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += ys.size(0)
            correct += predicted.eq(ys).sum().item()

        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        train_acc = 100.0 * correct / total

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}: train_acc={train_acc:.1f}%, val_acc={val_acc:.2f}%, loss={total_loss/len(train_loader):.4f}")

    return model


def evaluate_model(model, loader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xs, ys, lengths in loader:
            xs, ys = xs.to(device), ys.to(device)
            outputs = model(xs, lengths)
            _, predicted = outputs.max(1)
            total += ys.size(0)
            correct += predicted.eq(ys).sum().item()

    return 100.0 * correct / total


def precision_sweep(model, test_loader, bitwidths, device='cpu'):
    """Sweep fixed-point bitwidths."""
    results = []

    # Float32 baseline
    print("  Evaluating float32 baseline...")
    float_acc = evaluate_model(model, test_loader, device)
    print(f"    Float32: {float_acc:.2f}%")
    results.append({'model': 'FastGRNN', 'ell': 32, 'accuracy_pct': float_acc})

    # Fixed-point sweep
    for ell in bitwidths:
        print(f"  ℓ={ell} (int={ell-16}, s=16): ", end='', flush=True)
        config = FixedPointConfig(total_bits=ell, frac_bits=16)
        fp_model = convert_fastgrnn_to_fixed_point(model, config)
        fp_model.to(device)

        acc = evaluate_model(fp_model, test_loader, device)
        print(f"{acc:.2f}%")
        results.append({'model': 'FastGRNN', 'ell': ell, 'accuracy_pct': acc})

    return results


def main():
    import pandas as pd

    if FastGRNNCell is None:
        raise SystemExit(
            "EdgeML is not installed. Install it to run the official "
            "Google-30 training path, or use Baselines.run_baselines."
        )

    data_root = Path('data/SpeechCommands')

    print("="*80)
    print("P0-Google30: FastGRNN Precision Test on Speech Commands")
    print("="*80)
    print("Dataset: Google Speech Commands v0.02 (30-class)")
    print("Model: FastGRNN (official EdgeML implementation)")
    print("Goal: Reproduce SiRNN Google-30 benchmark with precision sweep")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load datasets
    print("\nLoading Google Speech Commands (30-class)...")
    train_ds = GoogleSpeech30Dataset(data_root, split='train')
    val_ds = GoogleSpeech30Dataset(data_root, split='val')
    test_ds = GoogleSpeech30Dataset(data_root, split='test')

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              collate_fn=collate_pad, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            collate_fn=collate_pad, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=collate_pad, num_workers=0)

    # Train
    print("\nTraining FastGRNN model...")
    model = train_model(train_loader, val_loader, device=device, epochs=30)

    # Test
    print("\nEvaluating on test set...")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Float32 test accuracy: {test_acc:.2f}%")

    # Precision sweep
    print("\nPrecision sweep...")
    bitwidths = [16, 20, 24, 28]
    results = precision_sweep(model, test_loader, bitwidths, device)

    # Save results
    df = pd.DataFrame(results)
    output_path = Path('results/e0_google30_fastgrnn.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
