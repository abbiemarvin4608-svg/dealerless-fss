#!/usr/bin/env python3
"""Render the prescribed E0-U CSV evidence using CPU-only matplotlib.

Visual contract: three independent ell-by-s training-calibration heatmaps,
training-selected validation Top-1 / FP32 agreement curves, and a historical
strict-MAC Top-1 comparison. No metric, experiment, or validation configuration
is selected here. Rendering requires all 396 formal calibration candidates
and 18 completed full-validation rows; incomplete evidence produces no figures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("resnet50", "resnet101", "resnet152")
MODEL_LABELS = {"resnet50": "ResNet-50", "resnet101": "ResNet-101", "resnet152": "ResNet-152"}
ELL_GRID = (12, 16, 20, 24, 28, 32)
MODEL_COLORS = {"resnet50": "#0072B2", "resnet101": "#D55E00", "resnet152": "#009E73"}
UNIFORM_COLOR = "#0072B2"
STRICT_COLOR = "#D55E00"
BASELINE_COLOR = "#666666"
ARITHMETIC = "uniform_ell_bit_storage_wide_integer_dot_then_floor_rescale"
RING_ARITHMETIC = "uniform_ell_bit_ring_mac_then_signed_floor_rescale"
STRICT_ARITHMETIC = "strict_q_mul_then_ell_bit_saturating_add"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "e0_uniform_storage_imagenette"
DEFAULT_STRICT = Path(__file__).resolve().parent / "results" / "e0_q_fixed_strict_imagenette_all" / "e0_q_fixed_final.csv"


def require(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source.name}: missing columns {sorted(missing)}")


def assert_values(frame: pd.DataFrame, column: str, expected, source: Path) -> None:
    if not frame[column].eq(expected).all():
        raise ValueError(f"{source.name}: every {column} must be {expected!r}")


def assert_percentage(frame: pd.DataFrame, column: str, source: Path) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not (np.isfinite(values).all() and values.between(0, 100).all()):
        raise ValueError(f"{source.name}: {column} must contain finite percentages in [0,100]")


def assert_keys(frame: pd.DataFrame, expected: set[tuple], columns: tuple[str, ...], source: Path) -> None:
    if frame.duplicated(list(columns)).any():
        raise ValueError(f"{source.name}: duplicate configuration keys")
    actual = set(frame[list(columns)].itertuples(index=False, name=None))
    if actual != expected or len(frame) != len(expected):
        raise ValueError(f"{source.name}: incomplete grid; expected {len(expected)} rows, got {len(frame)}")


def load_evidence(results_dir: Path, strict_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_path = results_dir / "e0_uniform_storage_candidates.csv"
    final_path = results_dir / "e0_uniform_storage_final.csv"
    candidates = pd.read_csv(candidate_path)
    final = pd.read_csv(final_path)
    strict = pd.read_csv(strict_path)
    arithmetic_values = set(final.get("arithmetic", []))
    if len(arithmetic_values) != 1 or not arithmetic_values <= {ARITHMETIC, RING_ARITHMETIC}:
        raise ValueError("mixed or unsupported arithmetic identity")
    arithmetic = next(iter(arithmetic_values))
    common = {"model", "split", "ell", "s", "samples", "status", "arithmetic"}
    require(candidates, common | {"top1_pct", "evidence_scope"}, candidate_path)
    require(final, common | {"top1_pct", "fp32_top1_pct", "prediction_agreement_pct", "evidence_scope"}, final_path)
    require(strict, common | {"accuracy_pct", "float_accuracy_pct"}, strict_path)

    cal_keys = {(model, ell, s) for model in MODELS for ell in ELL_GRID for s in range(ell)}
    val_keys = {(model, ell) for model in MODELS for ell in ELL_GRID}
    assert_keys(candidates, cal_keys, ("model", "ell", "s"), candidate_path)
    assert_keys(final, val_keys, ("model", "ell"), final_path)
    assert_keys(strict, val_keys, ("model", "ell"), strict_path)
    for frame, source, split, samples in (
            (candidates, candidate_path, "calibration", 512),
            (final, final_path, "validation", 3925)):
        assert_values(frame, "split", split, source)
        assert_values(frame, "samples", samples, source)
        assert_values(frame, "status", "completed", source)
        assert_values(frame, "evidence_scope", "formal", source)
        assert_values(frame, "arithmetic", arithmetic, source)
        assert_percentage(frame, "top1_pct", source)
    assert_values(strict, "split", "validation", strict_path)
    assert_values(strict, "samples", 3925, strict_path)
    assert_values(strict, "status", "ok", strict_path)
    assert_values(strict, "arithmetic", STRICT_ARITHMETIC, strict_path)
    assert_percentage(final, "fp32_top1_pct", final_path)
    assert_percentage(final, "prediction_agreement_pct", final_path)
    assert_percentage(strict, "accuracy_pct", strict_path)
    assert_percentage(strict, "float_accuracy_pct", strict_path)
    for frame, source in ((final, final_path), (strict, strict_path)):
        if not ((frame.s >= 0) & (frame.s < frame.ell) & (frame.s % 1 == 0)).all():
            raise ValueError(f"{source.name}: selected s is not legal")
    for model in MODELS:
        if final.loc[final.model == model, "fp32_top1_pct"].nunique() != 1:
            raise ValueError(f"{final_path.name}: inconsistent FP32 baseline for {model}")
    return (candidates.sort_values(["model", "ell", "s"]),
            final.sort_values(["model", "ell"]), strict.sort_values(["model", "ell"]))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save_figure(fig, stem: str, output_dir: Path, development_fixture: bool) -> list[str]:
    if development_fixture:
        fig.text(.5, .995, "Development fixture", ha="center", va="top",
                 color="#9C2A00", fontsize=10, fontweight="bold")
    names = []
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"metadata": {"Title": stem, "Creator": "plot_uniform_storage.py"}} if extension == "pdf" else {}
        fig.savefig(path, dpi=300, bbox_inches="tight", **kwargs)
        names.append(path.name)
    plt.close(fig)
    return names


def heatmap(model: str, candidates: pd.DataFrame, final: pd.DataFrame):
    matrix = np.full((len(ELL_GRID), max(ELL_GRID)), np.nan)
    for row in candidates[candidates.model == model].itertuples(index=False):
        matrix[ELL_GRID.index(int(row.ell)), int(row.s)] = row.top1_pct
    fig, ax = plt.subplots(figsize=(9.0, 3.3))
    fig.subplots_adjust(bottom=.22, left=.085, right=.88, top=.86)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E6E6E6")
    values = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=100,
                       aspect="auto", interpolation="nearest")
    for row in final[final.model == model].itertuples(index=False):
        y = ELL_GRID.index(int(row.ell))
        ax.scatter(int(row.s), y, s=74, facecolors="none", edgecolors="black", linewidths=3)
        ax.scatter(int(row.s), y, s=74, facecolors="none", edgecolors="white", linewidths=1.4)
    ax.set(yticks=range(len(ELL_GRID)), yticklabels=ELL_GRID,
           xticks=(0, 4, 8, 12, 16, 20, 24, 28, 31),
           xlabel="Fractional width, s", ylabel="Storage width, ℓ (bits)",
           title=f"{MODEL_LABELS[model]} calibration")
    ax.set_xticks(np.arange(-.5, max(ELL_GRID), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ELL_GRID), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.4, alpha=.5)
    ax.tick_params(which="minor", length=0)
    colorbar_ax = fig.add_axes([.91, .22, .018, .64])
    fig.colorbar(values, cax=colorbar_ax, label="Top-1 (%)")
    fig.text(.5, .035, "Circles: training-selected s. Gray: invalid s ≥ ℓ.", ha="center", fontsize=8)
    return fig


def format_line_axes(ax, label: str) -> None:
    ax.set(xlim=(11, 33), ylim=(-5, 105), xticks=ELL_GRID,
           yticks=(0, 20, 40, 60, 80, 100), ylabel=label)
    ax.grid(axis="y", color="#D9D9D9", linewidth=.5)
    ax.set_axisbelow(True)


def selected_labels(ax, rows: pd.DataFrame, y_column: str, color: str,
                    upper: bool = True, x_offset: int = 0) -> None:
    for row in rows.itertuples(index=False):
        value = float(getattr(row, y_column))
        offset = 8 if upper else (-14 if value >= 8 else 19)
        horizontal = x_offset
        if x_offset and int(row.ell) == ELL_GRID[0]:
            horizontal += 14
        elif x_offset and int(row.ell) == ELL_GRID[-1]:
            horizontal -= 14
        ax.annotate(f"s={int(row.s)}", (int(row.ell), value), xytext=(horizontal, offset),
                    textcoords="offset points", ha="center", color=color, fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .15})


def validation_curves(final: pd.DataFrame):
    label = "Strict ring" if final.arithmetic.eq(RING_ARITHMETIC).all() else "Wide MAC + saturation"
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 8.1), sharex=True)
    fig.subplots_adjust(hspace=.4, wspace=.3, top=.95, bottom=.09)
    for index, model in enumerate(MODELS):
        rows = final[final.model == model]
        color = MODEL_COLORS[model]
        left, right = axes[index]
        left.plot(rows.ell, rows.top1_pct, "o-", color=color, linewidth=1.5,
                  markersize=4, label=label)
        left.axhline(rows.fp32_top1_pct.iloc[0], color=BASELINE_COLOR, linestyle="--",
                     linewidth=1.2, label="Current FP32")
        selected_labels(left, rows, "top1_pct", color)
        right.plot(rows.ell, rows.prediction_agreement_pct, "s-", color=color,
                   linewidth=1.5, markersize=4)
        format_line_axes(left, "Top-1 (%)")
        format_line_axes(right, "FP32 agreement (%)")
        left.set_title(MODEL_LABELS[model])
        right.set_title(MODEL_LABELS[model])
        left.legend(loc="lower right", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Storage width, ℓ (bits)")
    fig.text(.5, .025, "Training-selected s; full validation (n = 3,925).", ha="center", fontsize=8)
    return fig


def strict_comparison(final: pd.DataFrame, strict: pd.DataFrame):
    label = "Strict ring" if final.arithmetic.eq(RING_ARITHMETIC).all() else "Wide MAC + saturation"
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    fig.subplots_adjust(hspace=.38, top=.95, bottom=.1)
    for ax, model in zip(axes, MODELS):
        uniform_rows = final[final.model == model]
        strict_rows = strict[strict.model == model]
        ax.plot(uniform_rows.ell, uniform_rows.top1_pct, "o-", color=UNIFORM_COLOR,
                linewidth=1.6, markersize=4, label=label)
        ax.plot(strict_rows.ell, strict_rows.accuracy_pct, "s--", color=STRICT_COLOR,
                linewidth=1.4, markersize=4, label="Historical ordered saturation")
        selected_labels(ax, uniform_rows, "top1_pct", UNIFORM_COLOR, x_offset=-14)
        selected_labels(ax, strict_rows, "accuracy_pct", STRICT_COLOR, upper=False, x_offset=14)
        format_line_axes(ax, "Top-1 (%)")
        ax.set_title(MODEL_LABELS[model])
        ax.legend(loc="lower right", frameon=False)
    axes[-1].set_xlabel("Storage width, ℓ (bits)")
    fig.text(.5, .025, "Each curve uses its own training-selected s; full validation (n = 3,925).",
             ha="center", fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--strict-final", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--output-dir", type=Path,
                        help="defaults to <results-dir>/figures")
    parser.add_argument("--check-only", action="store_true", help="validate evidence without writing figures")
    parser.add_argument("--development-fixture", action="store_true",
                        help="watermark development plots; all fixture paths must be under /tmp")
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir / "figures"
    if args.development_fixture:
        for path in (args.results_dir, args.strict_final, output_dir):
            if Path("/tmp") not in path.resolve().parents:
                parser.error("development fixture inputs and outputs must be under /tmp")
    try:
        candidates, final, strict = load_evidence(args.results_dir, args.strict_final)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as error:
        parser.exit(2, f"Evidence incomplete or invalid: {error}\n")
    print("Evidence complete: 396 calibration candidates, 18 selected full-validation rows, 18 historical strict rows.")
    if args.check_only:
        return
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for model in MODELS:
        figures += save_figure(heatmap(model, candidates, final),
                              f"calibration_heatmap_{model}", output_dir, args.development_fixture)
    figures += save_figure(validation_curves(final), "validation_selected_curves",
                          output_dir, args.development_fixture)
    figures += save_figure(strict_comparison(final, strict), "strict_top1_comparison",
                          output_dir, args.development_fixture)
    sources = {"calibration_candidates": args.results_dir / "e0_uniform_storage_candidates.csv",
               "selected_validation": args.results_dir / "e0_uniform_storage_final.csv",
               "historical_strict_validation": args.strict_final}
    manifest = {
        "script": "plot_uniform_storage.py",
        "arithmetic": str(final.arithmetic.iloc[0]),
        "evidence_scope": "development_fixture" if args.development_fixture else "formal",
        "models": list(MODELS), "ell_grid": list(ELL_GRID),
        "calibration_samples": 512, "validation_samples": 3925,
        "source_files": {key: os.path.relpath(path.resolve(), output_dir.resolve()) for key, path in sources.items()},
        "source_rows": {"calibration_candidates": len(candidates), "selected_validation": len(final),
                        "historical_strict_validation": len(strict)},
        "figures": figures,
        "palette": {"heatmap": "viridis [0,100]", "model_lines": MODEL_COLORS,
                    "uniform": UNIFORM_COLOR, "strict": STRICT_COLOR},
        "selection": "selected s is read from the supplied validation rows, whose runner selects only on training calibration",
        "fp32": "validation curve figure uses only current fp32_top1_pct; strict comparison omits FP32 baselines",
        "historical_fp32_policy": "TF32 policy was not recorded; historical and current FP32 baselines are not merged",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(figures)} figure files and figure_manifest.json to {output_dir}")


if __name__ == "__main__":
    main()
