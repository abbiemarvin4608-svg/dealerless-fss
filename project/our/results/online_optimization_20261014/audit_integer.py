#!/usr/bin/env python3
"""Use the independent integer oracle with explicit SCI online-generation policy."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[3] / "project/our/tools/segmented_dcf_runtime"
sys.path.insert(0, str(RUNTIME))
from audit_folded_step import audit


def check(directory, fixture, activation_mode):
    result = json.loads((directory / "result.json").read_text())
    report = audit(directory, fixture, fixture, activation_mode)
    report["original_protocol_checks"] = dict(report["protocol"]["checks"])
    config = result["config"]
    if config["backend"] == "SCI":
        parties = json.loads((directory / "parties.json").read_text())
        expected_queries = set(range(config["queries"] + config.get("warmup_queries", 0)))
        policy = config.get("native_preprocessing_effective") is False and activation_mode == "sci"
        evidence = []
        for party in parties:
            online = [row for row in party["phases"] if row["phase"] == "online"]
            ready = party.get("native_material_ready", [])
            policy &= {row["query"] for row in online} == expected_queries
            policy &= {row["query"] for row in ready} == expected_queries
            policy &= all(row.get("strict") is False for row in ready)
            policy &= all(row.get("native_generation_epoch_delta", 0) > 0 for row in online)
            policy &= not any(party.get(name) for name in ("native_material_prepared", "native_triple_prepared",
                                                           "folded_activation_prepared", "dpf_lut_prepared"))
            evidence.append({"party": party["party"], "online_generation": [
                {key: value for key, value in row.items() if key == "query" or "generation_epoch_delta" in key}
                for row in online], "material_ready": ready})
        # Zero generation is required only by the prepared variants. Preserve
        # that original failed/not-applicable value instead of relabeling it.
        report["protocol"]["checks"].pop("recorded_online_generation_zero")
        report["protocol"]["checks"]["unprocessed_generation_matches_policy"] = bool(policy)
        report["online_generation_policy"] = {"mode": "native generation inside online", "evidence": evidence,
            "zero_online_generation": "not applicable to unprocessed SCI"}
        report["stock_applicability"] = {"prepared_inventory": "N/A", "one_use_stock_verification": "N/A",
            "reason": "SCI generates protocol material inside online; no prepared query stock is requested",
            "original_recorded_inventory_balanced": report["original_protocol_checks"]["recorded_inventory_balanced"]}
        report["status"] = "passed" if all(report["protocol"]["checks"].values()) and \
            not any(row["mismatches"] for row in report["rows"]) else "failed"
    else:
        report["online_generation_policy"] = {"mode": "all query material before input", "zero_online_generation": "required"}
        report["stock_applicability"] = {"prepared_inventory": "required", "one_use_stock_verification": "required"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--activation-mode", choices=("folded", "sci"), required=True)
    args = parser.parse_args()
    try:
        report = check(args.directory, args.fixture, args.activation_mode)
    except (ValueError, KeyError, OSError) as error:
        report = {"status": "failed", "error": str(error)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "mismatches": sum(row["mismatches"] for row in report.get("rows", [])),
                      "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
