# Comparison output demand and leaf hashing

`SEGFSS_PRUNE_EQUALITY` and `SEGFSS_PRUNE_LEAF_HASHES` are independent,
disabled-by-default matched ablations. They do not change keys, segmentation,
integer contracts or activation protocols. The online path retains compact
keys and evaluates PRG paths; it does not compile an outer comparison to a LUT.

For adjacent high/low prefixes, the balanced scan uses
`L = L_left XOR (e_left AND L_right)` and
`e = e_left AND e_right`. Every merge needs the left equality for `L`.
The second AND and the right subtree's equality are needed only when the
parent equality is requested. An odd final prefix is carried unchanged.
Demand propagates backwards from the root before material generation.
For L-only scans with 1–8 segments, the planned AND counts per comparison
are `0, 1, 3, 4, 7, 8, 10, 11`; full `(L,e)` scans retain `2*(nseg-1)`.
The same plan controls resident triple capacity, offline generation and
online consumption. Online cannot upgrade an L-only inventory to `(L,e)`.

Signed toward-zero `div_pow2` needs low-part equality to distinguish a
nonzero discarded part. Floor truncation and sign/wrap comparisons require
only L. Fused comparisons preserve each domain's own demand while sharing
the independent layer opening. Pruning decreases payload and triple count;
it does not remove the remaining balanced tree's dependency layers.

At the cGGM leaf, public final query direction 0 needs the right sibling's
control bit for L and the selected left leaf for e. Direction 1 contributes
no final sibling to L and needs only the selected right leaf for e. With
leaf pruning enabled, the hash counts for directions 0/1 are respectively
`2/1` for `(L,e)` and `1/0` for L-only. The direction is from the already
opened masked query. Required local shares agree with the original evaluator
on identical keys in scalar, bounded AES batch and compact-key paths.

Conversion spool version 4 stores output demand and both pruning flags;
version 3 material is rejected rather than interpreted with a different
inventory. `CONVERSION_DETAIL` reports consumed AND gates, actual internal
and leaf hashes, skipped leaf hashes, omitted e outputs and both flags.

`comparison_pruning_benchmark` and `bench_comparison_pruning.py` measure a
signed 32-bit `div_pow2` by 2^14 with depth-cap-8 segments. Each party has
four disjoint logical CPUs and RLIMIT_AS 1.5 GiB. Fresh complete material
is in resident vectors before query shares are constructed. The four flag
combinations run in alternating order, with raw causal socket traces and
the independent SCI integer oracle. Offline/online wall, wire bytes,
rounds, scan triples and hash work are reported separately. LAN uses
1 Gbps, 0.5 ms RTT; WAN uses 400 Mbps, 4 ms RTT, with
`wire_bytes * 8 / bandwidth_bps + rounds * RTT`.
This experiment measures a complete conversion operator; the ticket 13
FastGRNN measurements determine whole-method acceptance.
