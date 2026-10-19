#!/usr/bin/env python3
"""Plot E3 phase breakdown: offline vs online cost decomposition across nseg."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read E3 phase breakdown data
df = pd.read_csv('results/e3_phase_breakdown.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Offline phase breakdown (stacked bar) ---
offline_phases = ['split_ms', 'cot_ms', 'expand_ms', 'cw_ms', 'leaf_ms',
                  'scan_triples_ms']
offline_labels = ['BitDec+Slice', 'COT', 'Expand', 'CW Exchange', 'Leaf',
                  'Scan triples']
offline_data = df[offline_phases].values.T

x = np.arange(len(df))
colors_offline = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b']

bottom = np.zeros(len(df))
for i, (phase, label) in enumerate(zip(offline_data, offline_labels)):
    ax1.bar(x, phase, bottom=bottom, label=label, color=colors_offline[i])
    bottom += phase

ax1.set_xlabel('nseg', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.set_title('E3: Offline Phase Breakdown (ℓ=32)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['nseg'])
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# --- Right: Offline vs Online total (line plot) ---
ax2.plot(df['nseg'], df['offline_ms'], marker='o', linewidth=2, label='Offline', color='#1f77b4')
ax2.plot(df['nseg'], df['online_ms'], marker='s', linewidth=2, label='Online (prefix scan)', color='#d62728')
ax2.plot(df['nseg'], df['offline_ms'] + df['online_ms'], marker='^', linewidth=2,
         label='Total', color='#2ca02c', linestyle='--')

ax2.set_xlabel('nseg', fontsize=12)
ax2.set_ylabel('Time (ms)', fontsize=12)
ax2.set_title('E3: Offline vs Online Cost (ℓ=32)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xticks(df['nseg'])

plt.tight_layout()
plt.savefig('results/e3_phase_breakdown.png', dpi=150, bbox_inches='tight')
print(f"Saved: results/e3_phase_breakdown.png")

# Print key findings
print("\n=== E3 Key Findings ===")
print(f"BitDec (split) dominates offline: {df['split_ms'].mean():.1f}ms avg (nseg-independent)")
def value_at(nseg, column):
    rows = df.loc[df['nseg'] == nseg, column]
    return float(rows.iloc[0]) if not rows.empty else None

expand_low = value_at(2, 'expand_ms')
expand_high = value_at(8, 'expand_ms')
if expand_low is not None and expand_high is not None:
    print(f"Expand scales exponentially with depth: nseg=2→{expand_low:.2f}ms, nseg=8→{expand_high:.2f}ms")
online_low = value_at(2, 'online_ms')
online_high = value_at(16, 'online_ms')
if online_low is not None and online_high is not None:
    print(f"Online scan grows linearly: nseg=2→{online_low:.1f}ms, nseg=16→{online_high:.1f}ms")
total = df['offline_ms'] + df['online_ms']
best = df.loc[total.idxmin(), 'nseg']
print(f"Total-latency nseg*={int(best)} for ℓ=32: {total.min():.1f}ms; "
      "nseg=2 remains the online/communication choice")
