#!/usr/bin/env python3
"""Generate a diagnostic CNN and independent Python-bigint tensor oracle."""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def generate(directory, bits, frac):
    directory.mkdir(parents=True, exist_ok=True)
    mask, half = (1 << bits) - 1, 1 << (bits - 1)
    generator = random.Random(701 + bits)

    def signed(value):
        value &= mask
        return value - (1 << bits) if value >= half else value

    def random_codes(count):
        return [generator.randrange(-half, half) & mask for _ in range(count)]

    x = np.array(random_codes(3 * 14 * 14), dtype="<u8").reshape(1, 3, 14, 14)
    weight = np.array(random_codes(4 * 3 * 3 * 3), dtype="<u8").reshape(4, 3, 3, 3)
    bias = np.array([half // 3, half // 2, -half // 3 & mask, half - 11], dtype="<u8")
    fc_weight = np.array(random_codes(5 * 4), dtype="<u8").reshape(5, 4)
    fc_bias = np.array(random_codes(5), dtype="<u8")
    conv = np.empty((1, 4, 14, 14), dtype="<u8")
    for co in range(4):
        for h in range(14):
            for w in range(14):
                total = 0
                for ci in range(3):
                    for kh in range(3):
                        for kw in range(3):
                            ih, iw = h + kh - 1, w + kw - 1
                            if 0 <= ih < 14 and 0 <= iw < 14:
                                total += int(x[0, ci, ih, iw]) * int(weight[co, ci, kh, kw])
                conv[0, co, h, w] = ((signed(total) // (1 << frac)) + int(bias[co])) & mask
    relu = np.array([max(0, signed(int(v))) for v in conv.flat], dtype="<u8").reshape(conv.shape)
    pooled = np.empty((1, 4, 7, 7), dtype="<u8")
    for co in range(4):
        for h in range(7):
            for w in range(7):
                pooled[0, co, h, w] = max(signed(int(v)) for v in relu[0, co, h * 2:h * 2 + 2, w * 2:w * 2 + 2].flat) & mask
    added = np.array([(int(v) * 2) & mask for v in pooled.flat], dtype="<u8").reshape(pooled.shape)
    averaged = np.array([signed(sum(int(v) for v in added[0, co].flat)) // 49 & mask
                         for co in range(4)], dtype="<u8").reshape(1, 4, 1, 1)
    fc = np.array([(signed(sum(int(averaged[0, ci, 0, 0]) * int(fc_weight[co, ci])
                              for ci in range(4))) // (1 << frac) + int(fc_bias[co])) & mask
                   for co in range(5)], dtype="<u8").reshape(1, 5)
    tensors = dict(input=x, conv=conv, relu=relu, maxpool=pooled, add=added, avgpool=averaged, fc=fc)
    nodes = [
        dict(name="conv", op="conv2d", inputs=["input"], output="conv", weight_offset=0,
             weight_shape=[4, 3, 3, 3], bias_offset=108, bias_count=4, kernel=3, stride=1, padding=1),
        dict(name="relu", op="relu", inputs=["conv"], output="relu"),
        dict(name="maxpool", op="maxpool", inputs=["relu"], output="maxpool", kernel=2, stride=2, padding=0),
        dict(name="add", op="add", inputs=["maxpool", "maxpool"], output="add"),
        dict(name="avgpool", op="avgpool", inputs=["add"], output="avgpool", kernel=7, stride=1, padding=0),
        dict(name="fc", op="linear", inputs=["avgpool"], output="fc", weight_offset=112,
             weight_shape=[5, 4], bias_offset=132, bias_count=5),
    ]
    for node in nodes:
        node["shape"] = list(tensors[node["output"]].shape)
    graph = dict(model="diagnostic_contract_w%d" % bits, batch=1, bits=bits, frac=frac,
                 input="input", output="fc", tensors={key: list(value.shape) for key, value in tensors.items()},
                 nodes=nodes, evidence_scope="diagnostic")
    (directory / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    with (directory / "parameters.bin").open("wb") as stream:
        for value in (weight, bias, fc_weight, fc_bias):
            stream.write(value.tobytes())
    audit = directory / "audit"
    audit.mkdir(exist_ok=True)
    index = dict(model=graph["model"], bits=bits, frac=frac, dtype="uint64LE_ring_residues",
                 evidence_scope="diagnostic", oracle="independent Python bigint", tensors={})
    for position, (name, value) in enumerate(tensors.items()):
        filename = "input.bin" if name == "input" else "%04d.bin" % position
        (audit / filename).write_bytes(value.tobytes())
        index["tensors"][name] = dict(file=filename, shape=list(value.shape), elements=value.size)
    (audit / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for bits, frac in ((8, 0), (16, 4), (24, 8), (32, 12)):
        graph = generate(args.output / ("w%d" % bits), bits, frac)
        print(json.dumps({"model": graph["model"], "nodes": len(graph["nodes"])}))
