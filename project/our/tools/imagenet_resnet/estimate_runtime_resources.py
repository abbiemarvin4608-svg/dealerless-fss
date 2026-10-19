#!/usr/bin/env python3
"""Derive ring32/f12 CNN inventory sizes from the public graph, without protocols."""
import argparse
import json
import math
from pathlib import Path


def divided_up(value, divisor):
    return (value + divisor - 1) // divisor


def scan_gates(segments):
    levels = []
    active = segments
    while True:
        levels.append([1] * active)
        if active == 1:
            break
        active = divided_up(active, 2)
    levels[-1][0] = 0
    for upper in range(len(levels) - 1, 0, -1):
        lower = levels[upper - 1]
        for parent, demand in enumerate(levels[upper]):
            left = 2 * parent
            if left + 1 < len(lower):
                lower[left], lower[left + 1] = 1, demand
            else:
                lower[left] = demand
    return sum(len(levels[i]) // 2 + sum(levels[i + 1][:len(levels[i]) // 2])
               for i in range(len(levels) - 1))


def comparison_row_bytes(bits):
    segments = divided_up(bits, 8)
    return segments * 48 + (bits - segments) * 16 + 3 * scan_gates(segments)


def estimate(graph, tile=64):
    if graph["bits"] != 32 or graph["frac"] != 12 or tile < 1:
        raise ValueError("estimate requires ring32/f12, optimized depth-cap-8, and positive tile")
    sizes = {key: math.prod(shape) for key, shape in graph["tensors"].items()}
    uses = {key: 0 for key in sizes}
    for node in graph["nodes"]:
        for key in node["inputs"]:
            uses[key] += 1
    uses[graph["output"]] += 1
    live = {graph["input"]: sizes[graph["input"]] * 8}
    totals = dict(scalar_macs=0, matrix_calls=0, linear_cot_instances=0,
                  linear_cot_calls=0, beaver_spool_bytes_per_party=0,
                  nonlinear_spool_bytes_per_party=0, conversion_spool_bytes_per_party=0,
                  primitive_calls=0, truncation_calls=0, live_tensors_plus_output_peak_bytes=0,
                  max_tensor_bytes=max(sizes.values()) * 8, max_matrix_a_bytes=0,
                  max_matrix_b_bytes=0, max_matrix_product_bytes=0,
                  plan_matrix_zero_fill_total_bytes=0)

    def primitive(kind, count, width=32):
        calls = divided_up(count, 4096)
        totals["primitive_calls"] += calls
        if kind == "msb":
            # CnnCall, seven vector lengths and two flags; x86_64 key records.
            value = calls * 76 + count * (9 + comparison_row_bytes(width - 1))
        elif kind == "select":
            value = calls * 12 + count * 25
        else:
            value = calls * 36 + count * 3
        totals["nonlinear_spool_bytes_per_party"] += value

    for node in graph["nodes"]:
        count = sizes[node["output"]]
        totals["live_tensors_plus_output_peak_bytes"] = max(
            totals["live_tensors_plus_output_peak_bytes"], sum(live.values()) + count * 8)
        live[node["output"]] = count * 8
        op = node["op"]
        if op in ("conv2d", "linear"):
            outputs = node["weight_shape"][0]
            inner = math.prod(node["weight_shape"]) // outputs
            positions = math.prod(node["shape"][2:]) if op == "conv2d" else 1
            for _ in range(node["shape"][0]):
                for position in range(0, positions, tile):
                    cols = min(tile, positions - position)
                    row_tile = min(outputs, max(1, (1 << 20) // (inner * cols)))
                    totals["plan_matrix_zero_fill_total_bytes"] += inner * cols * 8
                    for row in range(0, outputs, row_tile):
                        rows = min(row_tile, outputs - row)
                        macs = rows * inner * cols
                        cot_instances = inner * cols * 32
                        cot_batch = min(cot_instances,
                                        divided_up(1 << 18, macs) * inner * cols)
                        totals["scalar_macs"] += macs
                        totals["matrix_calls"] += 1
                        totals["linear_cot_instances"] += cot_instances
                        totals["linear_cot_calls"] += divided_up(cot_instances, cot_batch)
                        matrix_words = rows * inner + inner * cols + rows * cols
                        totals["beaver_spool_bytes_per_party"] += 24 + matrix_words * 8
                        totals["plan_matrix_zero_fill_total_bytes"] += (rows * inner + rows * cols) * 8
                        totals["max_matrix_a_bytes"] = max(totals["max_matrix_a_bytes"], rows * inner * 8)
                        totals["max_matrix_b_bytes"] = max(totals["max_matrix_b_bytes"], inner * cols * 8)
                        totals["max_matrix_product_bytes"] = max(totals["max_matrix_product_bytes"], rows * cols * 8)
            calls = divided_up(count, 4096)
            totals["truncation_calls"] += calls
            # 32->32 floor shift12, fused sign/wrap: domains12/31 and two B2A bits.
            # Per row: input/high/sign masks17, rounding3, B2A masks18, DCF259+636.
            # Per call: material metadata224 + comparison metadata408+424.
            totals["conversion_spool_bytes_per_party"] += count * 933 + calls * 1056
        elif op == "relu":
            primitive("msb", count)
            primitive("select", count)
        elif op == "maxpool":
            for _ in range(node["kernel"] ** 2 - 1):
                primitive("msb", count)
                primitive("select", count)
        elif op == "avgpool":
            shape = graph["tensors"][node["inputs"][0]]
            divisor = shape[2] * shape[3]
            primitive("msb", count)
            primitive("and", count)
            primitive("and", count)
            small_bits = 8
            while (1 << (small_bits - 1)) <= 4 * divisor:
                small_bits += 8
            half_remainder, ring_remainder = (1 << 31) % divisor, (1 << 32) % divisor
            thresholds = (half_remainder, half_remainder + divisor,
                          half_remainder + ring_remainder - divisor,
                          half_remainder + ring_remainder,
                          half_remainder + ring_remainder + divisor)
            for threshold in thresholds:
                if 0 < threshold <= 2 * divisor - 2:
                    primitive("msb", count, small_bits)
                    primitive("select", count)
            primitive("select", count)
        elif op != "add":
            raise ValueError("unsupported public operator: " + op)
        for key in node["inputs"]:
            uses[key] -= 1
            if uses[key] == 0 and key != node["output"]:
                del live[key]

    # Both private spools coexist. Linear COT alone gives a strict lower bound.
    totals["native_linear_spool_bytes_both_parties_lower_bound"] = (
        totals["scalar_macs"] * 32 * 24 + totals["linear_cot_instances"] +
        totals["linear_cot_calls"] * 48)
    totals["native_linear_spool_bytes_party1_lower_bound"] = (
        totals["scalar_macs"] * 32 * 16 + totals["linear_cot_calls"] * 24)
    totals["native_linear_spool_bytes_party2_lower_bound"] = (
        totals["scalar_macs"] * 32 * 8 + totals["linear_cot_instances"] +
        totals["linear_cot_calls"] * 24)
    totals["ours_spool_bytes_per_party"] = sum(totals[key] for key in (
        "beaver_spool_bytes_per_party", "nonlinear_spool_bytes_per_party",
        "conversion_spool_bytes_per_party"))
    totals["ours_spool_bytes_both_parties"] = totals["ours_spool_bytes_per_party"] * 2
    return dict(status="public_shape_estimate_not_measurement", model=graph["model"],
                batch=graph["batch"], bits=32, frac=12, tile=tile,
                optimization_profile="optimized", abi="x86_64_current_serialization",
                abi_sizes=dict(CggmKeyRecord=48, CnnCall=12,
                               SegmentedDcfMetrics=320, SciTruncationMetrics=88),
                native_scope="linear COT only; excludes nonlinear inventory and other files",
                ours_scope="all three current private temporary spools; excludes logs and filesystem metadata",
                **totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = estimate(json.loads(args.graph.read_text()), args.tile)
    payload = json.dumps(value, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
