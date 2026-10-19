# Ticket 03 Arithmetic COT Evidence

The four live two-party tests passed on 2026-10-12. The retained logs cover
prepared and default SCI, two fresh queries per successful mode, and six
intentional failures with exit code 2 on both parties.

Reproduce from the runtime tools directory:

```sh
SDCF_COT_EVIDENCE_DIR=/absolute/path/to/evidence python3 -m unittest test_prepared_cot -v
```

The primitive harness compiles original SCI AuxProtocols and LinearOT callers
against a private header overlay. Each process has at most four affinity CPUs
and 1.5 GiB `RLIMIT_AS`; compilation serializes through the shared build lock.
These small diagnostics do not claim batch=128 FastGRNN results.

| Mode | Checked behavior | Offline generation epoch/query | Online generation epoch/query |
| --- | --- | ---: | ---: |
| prepared | Nine 13-bit native B2A conversions | 1 | 0 |
| default | Same native B2A and integer outputs | 0 | 1 |
| offline_cot | Independent native offline COT followed by prepared B2A | 2 | 0 |
| multiplication_prepared | Native signed matrix and Hadamard multiplication, both directions | 4 | 0 |
| multiplication_default | Same native signed integer examples | 0 | 4 |
| edges_prepared | Both directions, 1/17/33/64-bit rings, 257-count SIMD tail, five lanes, append replay | 8 | 0 |
| edges_default | Same arithmetic ring relations | 0 | 8 |

The nine B2A conversions send 15 sender correction bytes and 2 receiver choice
correction bytes online, plus one sync byte per party. Material balances show
9 effective OTs, 256 generated, 9 consumed, 247 public padding discarded.
The signed multiplication diagnostic sends 47 bytes per party online including
sync, consumes 49 effective OTs in each direction, and generates 512 per
direction. Boundary diagnostics consume 322 per direction, generate 1280,
and separately discard 958 public padding OTs per direction.

Failure directories `width_mismatch`, `group_mismatch`, `repeat`, `incomplete`,
`unfinished_random`, and `online_extension` retain the exact rejection reason.
The edges fixture's diagnostic share openings are included in its online
measurements in both modes. B2A and multiplication share openings occur in
output measurements, after the online boundary. This evidence covers COT;
comparison bit-triples and whole-step preprocessing are later tickets.
