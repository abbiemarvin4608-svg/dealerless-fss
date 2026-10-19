#!/usr/bin/env python3
"""Write a fixed signed-int16 hidden state for Batch=128 operator diagnostics."""
import argparse
from pathlib import Path
import struct


def write_fixture(path):
    edges = (-32768, -32767, -16385, -129, -3, -1, 0, 1, 3, 127, 129, 16383, 32767)
    values = [edges[index % len(edges)] for index in range(128 * 100)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<" + "h" * len(values), *values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_fixture(parser.parse_args().output)
