#!/usr/bin/env python3
"""Bind retained qualification evidence to the persistent rebuilt runtime."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

import requalify as reuse

HERE = Path(__file__).resolve().parent
FINAL = HERE / "final-build"


def verify_proof(record):
    path = Path(record["path"]).resolve()
    reuse.require(reuse.digest(path) == record["sha256"], "evidence changed: " + str(path))
    return path


def source_hashes(entries):
    return {key: value["sha256"] for key, value in entries.items()}


def current_bridge_inputs(core):
    dependency_files, inputs = reuse.bridge_dependencies(core)
    entries = {reuse.normalize(path, Path("/not-a-runtime"), core):
               dict(path=str(path), sha256=reuse.digest(path)) for path in sorted(inputs)}
    for path in sorted(core.rglob("flags.make")) + sorted(core.rglob("link.txt")):
        if any(part in reuse.LIBRARY_TARGETS for part in path.parts):
            key = "bridge-config/" + path.relative_to(core).as_posix()
            entries[key] = dict(path=str(path),
                sha256=reuse.sha(path.read_text().replace(str(core), "{bridge_build}").encode()),
                raw_sha256=reuse.digest(path))
    return dependency_files, entries


def verify_rebuild_inputs(args):
    prior = reuse.read(args.prior_reuse)
    reuse.require(prior.get("schema") == 1 and prior.get("status") == "passed" and
                  prior.get("eligible") is True, "prior qualification reuse did not pass")
    recorded = reuse.read(args.rebuild_input_proof)
    reuse.require(verify_proof(recorded["old_qualification_reuse_proof"]) ==
                  args.prior_reuse.resolve(), "rebuild input proof references another prior record")
    old_config_path = verify_proof(recorded["old_config_proof"])
    old_config = reuse.read(old_config_path)
    old_manifest = old_config["runtime_manifest"]
    reuse.require(old_manifest == recorded["old_runtime_manifest"],
                  "retained execution runtime manifest differs from the rebuild comparison")
    old_identity = prior["current_build"]
    reuse.require(old_config["binary"] == old_identity["binary"]["path"] and
                  old_config["binary_sha256"] == old_manifest["binary_sha256"] ==
                  old_identity["binary"]["sha256"], "recorded prior binary identities differ")
    old_runtime = Path(old_config["binary"]).parent
    bridges = [Path(path) for path in old_manifest["sources"]
               if Path(path).name == "libsegfss_sci_bridge.so"]
    reuse.require(len(bridges) == 1, "retained runtime must identify one bridge")
    old_bridge = bridges[0]
    reuse.require(dict(path=str(old_bridge), sha256=old_manifest["sources"][str(old_bridge)]) ==
                  old_identity["bridge"], "recorded prior bridge identities differ")
    old_inputs = {reuse.normalize(path, old_runtime, old_bridge.parent):
                  dict(path=path, sha256=value) for path, value in old_manifest["sources"].items()}
    reuse.require(len(old_inputs) == 62 and old_inputs == recorded["old_normalized_runtime_inputs"] ==
                  old_identity["normalized_sources"], "retained runtime input inventory differs")

    new, _, _, core = reuse.build_identity(args.new_manifest)
    reuse.require(new == recorded["current_build"], "current build differs from the rebuild proof")
    old_metadata = {key: old_manifest.get(key) for key in new["metadata"]}
    reuse.require(old_metadata == old_identity["metadata"] == new["metadata"],
                  "generated source, compiler, or numeric settings changed")
    metadata_comparison = {key: dict(before=old_manifest.get(key), after=new["metadata"][key], match=True)
                           for key in new["metadata"]}
    reuse.require(metadata_comparison == recorded["runtime_metadata_comparison"],
                  "rebuild metadata comparison differs")
    reuse.require(len(new["normalized_sources"]) == 62, "current runtime input inventory differs")
    changes = reuse.differences(old_inputs, new["normalized_sources"],
                               {"bridge/libsegfss_sci_bridge.so"})
    reuse.require(len(changes) == 1 and changes[0]["source"] == "bridge/libsegfss_sci_bridge.so" and
                  changes[0]["allowed"] is True and changes == recorded["runtime_manifest_changes"],
                  "rebuild must change only the bridge output identity among the 62 runtime inputs")

    bridge_path = verify_proof(recorded["bridge_input_comparison"])
    bridge_record = reuse.read(bridge_path)
    reuse.require(Path(bridge_record["old_evidence_path"]).resolve() == args.prior_reuse.resolve() and
                  bridge_record["old_evidence_sha256"] == reuse.digest(args.prior_reuse),
                  "bridge comparison references another prior record")
    baseline_inputs = prior["bridge_source_snapshot"]["current"]["inputs"]
    dependency_files, inputs = current_bridge_inputs(core)
    reuse.require(len(baseline_inputs) == len(inputs) == 146 and
                  bridge_record["old_inputs"] == baseline_inputs and
                  bridge_record["new_inputs"] == inputs and
                  source_hashes(baseline_inputs) == source_hashes(inputs),
                  "one or more of the 146 recorded bridge inputs changed")
    reuse.require(bridge_record["new_dependency_files"] == dependency_files and
                  bridge_record["new_bridge"] == new["bridge"] and
                  bridge_record["differences"] == [] and
                  bridge_record["all_recorded_inputs_match"] is True,
                  "bridge comparison or dependency inventory differs")
    for item in dependency_files:
        verify_proof(item)

    tools_path = verify_proof(recorded["build_tool_inputs"])
    for entry in reuse.read(tools_path)["files"].values():
        verify_proof(entry)
    fixture_path = verify_proof(recorded["fixture_provenance"])
    fixtures = reuse.read(fixture_path)
    for name, expected in fixtures["fixture_files"].items():
        reuse.require(reuse.digest(fixture_path.parent / "fixture" / name) == expected,
                      "rebuilt fixture changed: " + name)
    reuse.require(all(fixtures["fixture_files"].get(name) == value for name, value in
                      old_config["fixture_files"].items()), "rebuilt fixture differs from retained execution")
    reuse.require(reuse.reviewed_patch(prior["old_source_commit"]) == prior["reviewed_source_repairs"],
                  "the reviewed budget repair or its Git baseline changed")

    historical = reuse.historical_proofs([item["path"] for item in prior["historical_qualifications"]])
    reuse.require(historical == prior["historical_qualifications"],
                  "historical qualifications or original execution hashes changed")
    reuse.validate_historical_coverage(historical)
    return prior, recorded, new, historical, dict(
        retained_config=reuse.proof(old_config_path), runtime_inputs=62,
        only_runtime_change="bridge/libsegfss_sci_bridge.so", bridge_inputs=146,
        bridge_inputs_identical=True, historical_qualifications=len(historical),
        historical_execution_hashes_identical=True,
        bridge_comparison=reuse.proof(bridge_path), build_tool_inputs=reuse.proof(tools_path),
        fixture_provenance=reuse.proof(fixture_path))


def build_report(args):
    prior, recorded, current, historical, checks = verify_rebuild_inputs(args)
    diagnostics = reuse.new_diagnostics(args.dpf_qualification, args.folded_qualification, current)
    executions = [execution for item in historical for execution in item["original_executions"]]
    reuse.require(bool(executions), "historical execution identity proofs are required")
    fixtures = executions[0]["original_fixture_files"]
    reuse.require(bool(fixtures) and fixtures == prior["fixture_files"] and all(
        row["original_fixture_files"] == fixtures for row in executions), "historical fixture identities differ")
    hidden_hashes = {row["original_hidden_sha256"] for row in executions if row["original_hidden_sha256"]}
    for row in diagnostics["rows"]:
        config = row["result"]["config"]
        reuse.require(config.get("fixture_files") == fixtures, "current diagnostic fixture differs")
        if config["hidden_file"]:
            reuse.require(config.get("hidden_sha256") in hidden_hashes,
                          "current diagnostic nonzero state differs")
    limitations = list(dict.fromkeys(prior["limitations"] + recorded["limitations"] + [
        "Prior /tmp binaries and build directories are absent; retained identity records supply historical provenance",
        "The rebuilt runtime has a new binary identity; current selective diagnostics bind this build while unchanged source qualifications are explicitly reused",
        "Historical runtime build_commands.json was not retained; exact historical runtime command equality is not asserted"]))
    return dict(schema=1, status="passed", eligible=True,
        recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Qualification reuse for a persistent rebuild with identical recorded source inputs; no formal timing samples are reused",
        original_build=prior["original_build"], prior_qualified_build=prior["current_build"],
        current_build=current, old_source_commit=prior["old_source_commit"],
        prior_qualification_reuse=reuse.proof(args.prior_reuse),
        rebuild_input_proof=reuse.proof(args.rebuild_input_proof), rebuild_input_checks=checks,
        allowed_changes=dict(runtime_inputs=["bridge/libsegfss_sci_bridge.so"], repository_sources=[],
            meaning="Rebuild and relocate the reviewed budget-fixed sources; only recorded bridge output identity differs"),
        reviewed_source_repairs=prior["reviewed_source_repairs"],
        runtime_manifest_changes=recorded["runtime_manifest_changes"],
        bridge_source_snapshot=prior["bridge_source_snapshot"],
        historical_qualifications=historical, current_diagnostic_proofs=diagnostics,
        fixture_files=fixtures, reused_coverage=prior["reused_coverage"],
        current_coverage=prior["current_coverage"], limitations=limitations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-reuse", type=Path, default=HERE / "qualification-reuse.json")
    parser.add_argument("--rebuild-input-proof", type=Path, default=FINAL / "build-input-reuse.json")
    parser.add_argument("--new-manifest", type=Path, default=FINAL / "runtime/runtime_manifest.json")
    parser.add_argument("--dpf-qualification", type=Path, default=FINAL / "qualification-budget/qualification.json")
    parser.add_argument("--folded-qualification", type=Path, default=FINAL / "qualification-folded/qualification.json")
    parser.add_argument("--output", type=Path, default=HERE / "qualification-reuse-final.json")
    parser.add_argument("--check-inputs", action="store_true", help="Verify retained sources and provenance without reading new diagnostics or writing a report")
    args = parser.parse_args()
    if not args.check_inputs:
        reuse.require(not args.output.exists(), "qualification reuse report must not overwrite an existing artifact")
    try:
        if args.check_inputs:
            _, _, _, _, checks = verify_rebuild_inputs(args)
            print(json.dumps(dict(status="checked", checks=checks), sort_keys=True))
            return 0
        report = build_report(args)
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(json.dumps(dict(status="failed", reason=str(error))))
        return 1
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
