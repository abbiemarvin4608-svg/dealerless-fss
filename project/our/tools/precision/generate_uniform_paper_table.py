#!/usr/bin/env python3
"""Generate the E0-U paper table only from complete formal CSV evidence.

The script refuses partial scans, diagnostic rows, changed arithmetic, or a
validation-selected s. It independently recomputes the training argmax for
each (model, ell), then writes one booktabs table to paper/tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


MODELS = ("resnet50", "resnet101", "resnet152")
MODEL_LABELS = {"resnet50": "ResNet-50", "resnet101": "ResNet-101", "resnet152": "ResNet-152"}
ELL_GRID = (12, 16, 20, 24, 28, 32)
ARITHMETIC = "uniform_ell_bit_storage_wide_integer_dot_then_floor_rescale"
RING_ARITHMETIC = "uniform_ell_bit_ring_mac_then_signed_floor_rescale"


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        fail(f"{source}: missing columns {missing}")


def assert_exact(frame: pd.DataFrame, column: str, value, source: Path) -> None:
    if not frame[column].eq(value).all():
        fail(f"{source}: {column} must be exactly {value!r}")


def check_grid(frame: pd.DataFrame, keys: set[tuple], columns: tuple[str, ...], source: Path) -> None:
    if frame.duplicated(list(columns)).any():
        fail(f"{source}: duplicate keys in {columns}")
    actual = set(frame[list(columns)].itertuples(index=False, name=None))
    if actual != keys or len(frame) != len(keys):
        fail(f"{source}: expected {len(keys)} complete rows, got {len(frame)}")


def verify_sources(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    candidate_path = results_dir / "e0_uniform_storage_candidates.csv"
    final_path = results_dir / "e0_uniform_storage_final.csv"
    if not candidate_path.is_file() or not final_path.is_file():
        fail(f"formal E0-U CSVs are incomplete under {results_dir}")
    candidates, final = pd.read_csv(candidate_path), pd.read_csv(final_path)
    arithmetic_values = set(final.get("arithmetic", []))
    if len(arithmetic_values) != 1 or not arithmetic_values <= {ARITHMETIC, RING_ARITHMETIC}:
        fail("table sources have an unsupported or mixed arithmetic identity")
    arithmetic = next(iter(arithmetic_values))
    ring = arithmetic == RING_ARITHMETIC
    require_columns(candidates, {"model", "split", "ell", "s", "samples", "status",
                                "evidence_scope", "arithmetic", "top1_correct", "top1_pct",
                                "register_bits", "register_overflows"}, candidate_path)
    require_columns(final, {"model", "split", "ell", "s", "samples", "status",
                            "evidence_scope", "arithmetic", "top1_correct", "top1_pct",
                            "fp32_top1_pct", "delta_pp", "prediction_agreement_pct",
                            "register_bits", "register_overflows"}, final_path)
    candidate_keys = {(model, ell, s) for model in MODELS for ell in ELL_GRID for s in range(ell)}
    final_keys = {(model, ell) for model in MODELS for ell in ELL_GRID}
    check_grid(candidates, candidate_keys, ("model", "ell", "s"), candidate_path)
    check_grid(final, final_keys, ("model", "ell"), final_path)
    for frame, source, split, samples in ((candidates, candidate_path, "calibration", 512),
                                          (final, final_path, "validation", 3925)):
        assert_exact(frame, "split", split, source)
        assert_exact(frame, "samples", samples, source)
        assert_exact(frame, "status", "completed", source)
        assert_exact(frame, "evidence_scope", "formal", source)
        assert_exact(frame, "arithmetic", arithmetic, source)
        expected_register = frame.ell if ring else 128
        if (frame.register_bits != expected_register).any() or (frame.register_overflows != 0).any():
            fail(f"{source}: inconsistent logical registers or arithmetic error")
        if ring:
            assert_exact(frame, "evaluation", "measured", source)
            assert_exact(frame, "max_register_bits_kind", "logical_ring_width", source)
            if (frame.max_register_bits > frame.ell).any():
                fail(f"{source}: logical ring value exceeds ell")
        for row in frame.itertuples(index=False):
            if not math.isfinite(float(row.top1_correct)) or not (0 <= row.top1_correct <= samples) \
                    or row.top1_correct != int(row.top1_correct):
                fail(f"{source}: invalid top1_correct")
            if not math.isfinite(float(row.top1_pct)) \
                    or abs(float(row.top1_pct) - 100 * int(row.top1_correct) / samples) > 1e-8:
                fail(f"{source}: top1_pct disagrees with top1_correct")
    if ((final.s < 0) | (final.s >= final.ell) | (final.s % 1 != 0)).any():
        fail(f"{final_path}: validation-selected s outside legal range")
    for column in ("fp32_top1_pct", "prediction_agreement_pct"):
        if not all(math.isfinite(float(value)) and 0 <= float(value) <= 100 for value in final[column]):
            fail(f"{final_path}: invalid {column}")
    if not all(math.isfinite(float(value)) for value in final.delta_pp):
        fail(f"{final_path}: invalid delta_pp")

    metadata = {}
    for model in MODELS:
        path = results_dir / f"uniform_storage_{model}_metadata.json"
        if not path.is_file():
            fail(f"missing metadata: {path}")
        metadata[model] = json.loads(path.read_text())
        item = metadata[model]
        if item.get("model") != model or item.get("arithmetic") != arithmetic \
                or item.get("evidence_scope") != "formal" or item.get("calibration_samples") != 512 \
                or item.get("validation_samples") != 3925 or item.get("register_bits") != ("ell" if ring else 128) \
                or item.get("classifier_classes") != 1000 or item.get("dataset") != "imagenette2-160" \
                or item.get("weights") != f"{MODEL_LABELS[model].replace('-', '')}_Weights.IMAGENET1K_V1":
            fail(f"{path}: metadata does not describe formal E0-U")
        policy = item.get("fp32_arithmetic", {})
        if policy.get("matmul_allow_tf32") is not False \
                or policy.get("cudnn_allow_tf32") is not False \
                or policy.get("float32_matmul_precision") != "highest":
            fail(f"{path}: FP32 TF32-off policy is missing")

    # Recompute from all training candidates, without importing runner selection.
    for model in MODELS:
        if final.loc[final.model == model, "fp32_top1_pct"].nunique() != 1:
            fail(f"{model}: FP32 reference varies across widths")
        for ell in ELL_GRID:
            train = candidates[(candidates.model == model) & (candidates.ell == ell)]
            selected = final[(final.model == model) & (final.ell == ell)]
            best = train.sort_values(["top1_correct", "s"], ascending=[False, True]).iloc[0]
            value = int(selected.iloc[0].s)
            if value != int(best.s):
                fail(f"{model} ell={ell}: final s={value} differs from training argmax s={int(best.s)}")
            row = selected.iloc[0]
            if abs(float(row.delta_pp) - (float(row.top1_pct) - float(row.fp32_top1_pct))) > 1e-8:
                fail(f"{model} ell={ell}: delta_pp is inconsistent")
    return candidates, final, metadata


def latex_num(value: float) -> str:
    return f"{float(value):.3f}"


def render_table(final: pd.DataFrame, development_fixture: bool = False) -> str:
    row_end = r"\\"
    ring = final.arithmetic.eq(RING_ARITHMETIC).all()
    caption = (
        (r"Strict ring ResNet inference (E0-U-R): parameters, inputs, raw products, "
         r"dot-product accumulation, and pooling sums are reduced modulo $2^\ell$. "
         r"Each dot product is interpreted as a signed $\ell$-bit value before one "
         r"floor rescaling; stored tensors use uniform $Q(\ell,s)$. " if ring else
         r"Historical saturation control (E0-U-S): ResNet inference with uniform "
         r"$Q(\ell,s)$ storage, 128-bit integer accumulation, one floor rescaling "
         r"per dot product, and saturating writeback. ") +
        r"For each $\ell$, $s$ maximizes Top-1 on 512 class-balanced "
        r"training images, with ties choosing smaller $s$; evaluation uses all 3,925 "
        r"ImageNette validation images. $\Delta$ is Top-1 minus FP32 in percentage points, "
        r"and agreement is the percentage of predictions matching FP32. "
        r"The FP32 reference disables CUDA matmul and cuDNN TF32."
    )
    if development_fixture:
        caption = "Development fixture. " + caption
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:e0-uniform-ring}" if ring else "\\label{tab:e0-uniform}",
        "\\begingroup",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        f"Model & $\\ell$ & $s$ & \\shortstack{{Top-1\\\\(\\%)}} & "
        f"\\shortstack{{$\\Delta$\\\\(pp)}} & \\shortstack{{Agreement\\\\(\\%)}} {row_end}",
        "\\midrule",
    ]
    for model_index, model in enumerate(MODELS):
        rows = final[final.model == model].sort_values("ell")
        for row_index, row in enumerate(rows.itertuples(index=False)):
            model_cell = MODEL_LABELS[model] if row_index == 0 else ""
            lines.append(f"{model_cell} & {int(row.ell)} & {int(row.s)} & "
                         f"{latex_num(row.top1_pct)} & {float(row.delta_pp):+.3f} & "
                         f"{latex_num(row.prediction_agreement_pct)} {row_end}")
        if model_index != len(MODELS) - 1:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\endgroup", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "e0_uniform_ring_imagenette")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[4] / "paper" / "tables" / "e0_uniform_ring.tex")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--development-fixture", action="store_true",
                        help="mark development tables; fixture source and output must both be under /tmp")
    args = parser.parse_args()
    if args.development_fixture:
        for path in (args.results_dir, args.output):
            if Path("/tmp") not in path.resolve().parents:
                parser.error("development fixture inputs and output must be under /tmp")
    try:
        _, final, _ = verify_sources(args.results_dir)
    except (RuntimeError, ValueError, pd.errors.EmptyDataError) as error:
        parser.exit(2, f"Evidence incomplete or invalid: {error}\n")
    print("Formal evidence verified: 396 calibration candidates, 18 selected validation rows, independent training argmax.")
    if args.check_only:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(render_table(final, args.development_fixture))
    os.replace(temporary, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
