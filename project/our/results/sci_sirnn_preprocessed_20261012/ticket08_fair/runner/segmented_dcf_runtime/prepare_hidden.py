#!/usr/bin/env python3
"""Prepare a client-owned nonzero hidden-state fixture for operator regression."""
import argparse
import json
from pathlib import Path
import struct


def prepare(output, batch):
    boundary = [-32768, -16385, -257, -1, 0, 1, 255, 16383, 32767]
    values = [boundary[(sample + column) % len(boundary)] for sample in range(batch) for column in range(100)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(struct.pack("<" + "h" * len(values), *values))
    output.with_suffix(".json").write_text(json.dumps({"owner": "client", "batch": batch,
        "hidden_width": 100, "dtype": "little-endian signed int16", "boundary_values": boundary,
        "purpose": "independent nonzero-state diagnostic; not quality evaluation"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.batch <= 256:
        parser.error("batch must be 1..256")
    prepare(args.output, args.batch)
