#!/usr/bin/env python3
"""Generate E4 ablation data + plot from the deterministic hash-count model.

Validated against test_ablation.cpp measured counts:
  - Half-Tree cGGM expansion: 2^(d-1) - 1 hashes (1/internal node)
  - Standard GGM expansion:   2 * (2^(d-1) - 1) hashes (2/internal node)
  - Prefix-parity eval (Grotto): d + 1 hashes  [O(log N)]
  - Full-domain eval:           ~1.5*2^d - 1 hashes  [O(N)]
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rows = []
for d in range(4, 21):
    N = 2 ** d
    internal = (2 ** (d - 1)) - 1
    ht_expand = internal
    std_expand = 2 * internal
    prefix_eval = d + 1
    # full-domain: internal expansion (2^(d-1)-1) + all 2^d leaf hashes
    full_eval = internal + N
    rows.append({
        'depth': d,
        'N': N,
        'ht_expand_hashes': ht_expand,
        'std_expand_hashes': std_expand,
        'expand_ratio': std_expand / ht_expand,
        'prefix_eval_hashes': prefix_eval,
        'full_eval_hashes': full_eval,
        'eval_ratio': full_eval / prefix_eval,
    })

df = pd.DataFrame(rows)
df.to_csv('results/e4_ablation.csv', index=False)
print("Saved: results/e4_ablation.csv")
print(df.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (i) Expansion: Half-Tree vs standard GGM
ax1.plot(df['depth'], df['ht_expand_hashes'], marker='o', linewidth=2,
         label='Half-Tree cGGM (1 hash/node)', color='#2ca02c')
ax1.plot(df['depth'], df['std_expand_hashes'], marker='s', linewidth=2,
         label='Standard GGM (2 hashes/node)', color='#d62728')
ax1.set_xlabel('Segment depth (bits)', fontsize=12)
ax1.set_ylabel('Expansion hash calls', fontsize=12)
ax1.set_title('E4(i): cGGM Expansion Cost (2× savings)', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# (ii) Eval: prefix-parity vs full-domain
ax2.plot(df['depth'], df['prefix_eval_hashes'], marker='o', linewidth=2,
         label='Grotto prefix-parity  O(log N)', color='#2ca02c')
ax2.plot(df['depth'], df['full_eval_hashes'], marker='s', linewidth=2,
         label='Full-domain expansion  O(N)', color='#d62728')
ax2.set_xlabel('Segment depth (bits)', fontsize=12)
ax2.set_ylabel('Evaluation hash calls', fontsize=12)
ax2.set_title('E4(ii): Online Eval Cost (O(log N) vs O(N))', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/e4_ablation.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/e4_ablation.png")

print("\n=== E4 Key Findings ===")
print(f"(i)  Half-Tree halves expansion cost: {df['expand_ratio'].iloc[0]:.1f}x fewer hashes (constant across depth)")
print(f"(ii) Grotto eval at depth=16: {df[df.depth==16]['prefix_eval_hashes'].iloc[0]} hashes vs "
      f"{df[df.depth==16]['full_eval_hashes'].iloc[0]} full-domain "
      f"({df[df.depth==16]['eval_ratio'].iloc[0]:.0f}x speedup)")
