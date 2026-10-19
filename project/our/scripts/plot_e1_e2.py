#!/usr/bin/env python3
"""
Plot E1/E2 results: segmentation sweep analysis.

Reads results/e1_e2_sweep.csv and generates:
  - E1 plot: offline/online cost vs nseg (fixed ell=32)
  - E2 plot: feasibility frontier (max ell per nseg)

Usage:
  python plot_e1_e2.py [--csv results/e1_e2_sweep.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_e1_nseg_sweep(df, output_dir):
    """E1: For ell=32, show offline/online cost vs nseg."""
    df_e1 = df[df['ell'] == 32].copy()
    if len(df_e1) == 0:
        print("No ell=32 data for E1 plot", file=sys.stderr)
        return

    df_e1 = df_e1.sort_values('nseg')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time
    ax1.plot(df_e1['nseg'], df_e1['offline_time_ms'], marker='o', label='Offline', linewidth=2)
    ax1.plot(df_e1['nseg'], df_e1['online_time_ms'], marker='s', label='Online', linewidth=2)
    ax1.set_xlabel('Number of segments (nseg)', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_title('E1: Computation Cost vs Segmentation (ℓ=32)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    # Communication
    ax2.plot(df_e1['nseg'], df_e1['offline_comm_bytes'], marker='o', label='Offline', linewidth=2)
    ax2.plot(df_e1['nseg'], df_e1['online_comm_bytes'], marker='s', label='Online', linewidth=2)
    ax2.set_xlabel('Number of segments (nseg)', fontsize=11)
    ax2.set_ylabel('Communication (bytes)', fontsize=11)
    ax2.set_title('E1: Communication Cost vs Segmentation (ℓ=32)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    fig.tight_layout()
    out_path = output_dir / 'e1_nseg_sweep.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"E1 plot saved to {out_path}")


def plot_e2_feasibility(df, output_dir):
    """E2: For each nseg, show cost scaling with ell (feasibility frontier)."""
    nseg_values = sorted(df['nseg'].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for nseg in nseg_values:
        df_nseg = df[df['nseg'] == nseg].sort_values('ell')
        if len(df_nseg) == 0:
            continue

        # Time
        ax1.plot(df_nseg['ell'], df_nseg['offline_time_ms'] + df_nseg['online_time_ms'],
                 marker='o', label=f'nseg={nseg}', linewidth=2)

        # Communication
        ax2.plot(df_nseg['ell'], df_nseg['offline_comm_bytes'] + df_nseg['online_comm_bytes'],
                 marker='o', label=f'nseg={nseg}', linewidth=2)

    ax1.set_xlabel('Bitwidth (ℓ)', fontsize=11)
    ax1.set_ylabel('Total Time (ms)', fontsize=11)
    ax1.set_title('E2: Scalability with Bitwidth', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Bitwidth (ℓ)', fontsize=11)
    ax2.set_ylabel('Total Communication (bytes)', fontsize=11)
    ax2.set_title('E2: Communication Scaling with Bitwidth', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / 'e2_feasibility.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"E2 plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot E1/E2 segmentation sweep results')
    parser.add_argument('--csv', type=str, default='results/e1_e2_sweep.csv',
                        help='Path to sweep CSV (default: results/e1_e2_sweep.csv)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points from {csv_path}")
    print(df.to_string(index=False))

    output_dir = csv_path.parent
    output_dir.mkdir(exist_ok=True)

    plot_e1_nseg_sweep(df, output_dir)
    plot_e2_feasibility(df, output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
