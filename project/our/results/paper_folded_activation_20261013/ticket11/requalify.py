#!/usr/bin/env python3
"""Record limited qualification reuse after the reviewed DPF budget repair."""
import argparse
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
COHORT = HERE.parent
MEMORY = 1536 << 20
DPF = "SCI_Beaver_segmented_DPF_LUT"
FOLDED = "SCI_Beaver_segmented_PaperLUT"
NATIVE = "project/our/tools/segmented_dcf_runtime/native_dpf_lut.cpp"
BRIDGE_CPP = "project/our/src/sci_dpf_bridge.cpp"
BRIDGE_HEADER = "project/our/src/sci_dpf_bridge.h"
SOURCE_CHANGES = {NATIVE, BRIDGE_CPP, BRIDGE_HEADER}
MANIFEST_CHANGES = {"repo/" + NATIVE, "runtime/sci_dpf_bridge.h",
                    "bridge/libsegfss_sci_bridge.so"}
LIBRARY_TARGETS = {"segfss_sci_bridge.dir", "segfss.dir", "dfss.dir", "dfss_common.dir"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def proof(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_blob(commit, relative):
    return subprocess.check_output(["git", "show", commit + ":" + relative], cwd=str(REPO))


def normalize(path, runtime, bridge):
    path = Path(path).resolve()
    for label, root in (("runtime", runtime), ("bridge", bridge), ("repo", REPO)):
        try:
            return label + "/" + path.relative_to(root).as_posix()
        except ValueError:
            continue
    return "external/" + path.as_posix()


def build_identity(manifest_path, old=False):
    manifest_path = Path(manifest_path).resolve()
    data = read(manifest_path)
    runtime = manifest_path.parent
    binary = runtime / "sdcf-runtime"
    bridges = [Path(path).resolve() for path in data["sources"]
               if Path(path).name == "libsegfss_sci_bridge.so"]
    require(len(bridges) == 1, "runtime manifest must identify exactly one bridge")
    bridge = bridges[0]
    require(digest(binary) == data["binary_sha256"], "retained binary differs from its manifest")
    require(digest(bridge) == data["sources"][str(bridge)], "bridge differs from its manifest")
    if not old:
        for path, expected in data["sources"].items():
            require(digest(path) == expected, "current runtime source changed: " + path)
    entries = {}
    for path, value in data["sources"].items():
        key = normalize(path, runtime, bridge.parent)
        require(key not in entries, "normalized source identity collision: " + key)
        entries[key] = dict(path=path, sha256=value)
    return dict(manifest=proof(manifest_path), binary=proof(binary), bridge=proof(bridge),
                metadata={key: data.get(key) for key in
                          ("generated_sha256", "numeric_mode", "DIV_RESCALING", "compiler")},
                normalized_sources=entries), data, runtime, bridge.parent


def differences(before, after, allowed):
    rows = []
    for key in sorted(set(before) | set(after)):
        left, right = before.get(key), after.get(key)
        if (left or {}).get("sha256") != (right or {}).get("sha256"):
            rows.append(dict(source=key, before=left, after=right,
                             allowed=key in allowed and left is not None and right is not None))
    return rows


def reviewed_patch(commit):
    replacements = {
        BRIDGE_CPP: [("extern \"C\" {\n", "extern \"C\" {\n"
            "uint64_t segfss_sci_dpf_material_metadata_bytes(void) {\n"
            "    return segfss::DpfPublicLutMaterial{}.residentBytes();\n}\n")],
        BRIDGE_HEADER: [("int segfss_sci_dpf_prepare(",
            "/* Retained material metadata, excluding the packed selection vector. */\n"
            "uint64_t segfss_sci_dpf_material_metadata_bytes(void);\n"
            "int segfss_sci_dpf_prepare(")],
        NATIVE: [
            ("#pragma weak segfss_sci_dpf_create\n",
             "#pragma weak segfss_sci_dpf_create\n#pragma weak segfss_sci_dpf_material_metadata_bytes\n"),
            ("void check(int result)",
             "uint64_t material_bytes(int index_bits,int output_bits) {\n"
             "    if(!segfss_sci_dpf_material_metadata_bytes)\n"
             "        throw std::runtime_error(\"DPF native material layout bridge is not linked\");\n"
             "    return (uint64_t(1)<<index_bits)*((output_bits+7)/8)+\n"
             "        segfss_sci_dpf_material_metadata_bytes();\n}\nvoid check(int result)"),
            ("sigmoid_count*(2*(256*2+256)+(64*2+256))",
             "sigmoid_count*(2*material_bytes(8,16)+material_bytes(6,15))"),
            ("tanh_count*(2*(256*2+256)+(8*2+256))",
             "tanh_count*(2*material_bytes(8,9)+material_bytes(3,9))"),
            ("((uint64_t(1)<<c.index_bits)*((std::max(c.output_bits[0],c.output_bits[1])+7)/8)+256)",
             "material_bytes(c.index_bits,std::max(c.output_bits[0],c.output_bits[1]))"),
        ],
    }
    rows = []
    for relative, edits in replacements.items():
        baseline = git_blob(commit, relative).decode()
        expected = baseline
        for before, after in edits:
            require(expected.count(before) == 1, "reviewed baseline replacement is ambiguous: " + relative)
            expected = expected.replace(before, after, 1)
        actual = (REPO / relative).read_text()
        require(actual == expected, "source differs from the exact reviewed budget repair: " + relative)
        rows.append(dict(source=relative, baseline_git_sha256=sha(baseline.encode()),
            current_sha256=sha(actual.encode()), exact_reviewed_repair=True,
            diff="".join(difflib.unified_diff(baseline.splitlines(True), actual.splitlines(True),
                                            fromfile=commit + "/" + relative, tofile="current/" + relative))))
    return rows


def bridge_dependencies(root):
    files, dependencies = [], set()
    targets = set()
    for path in sorted(root.rglob("*.o.d")):
        owned = [part for part in path.parts if part in LIBRARY_TARGETS]
        if not owned:
            continue
        targets.update(owned)
        files.append(proof(path))
        body = path.read_text().replace("\\\n", " ").split(":", 1)[1]
        for token in shlex.split(body):
            item = Path(token)
            item = (root / item).resolve() if not item.is_absolute() else item.resolve()
            if not item.as_posix().startswith(("/usr/", "/lib/")):
                dependencies.add(item)
    require(targets == LIBRARY_TARGETS, "bridge dependency graph is missing a linked library target")
    dependencies.update((REPO / "project/our/CMakeLists.txt",
                         REPO / "project/our/tools/sci_bridge/exports.map"))
    return files, dependencies


def bridge_snapshot(old_root, new_root, commit):
    inventories = []
    observations = []
    for old, root in ((True, old_root), (False, new_root)):
        files, inputs = bridge_dependencies(root)
        entries = {}
        for path in sorted(inputs):
            key = normalize(path, Path("/not-a-runtime-build"), root)
            if old and key.startswith("repo/"):
                content = git_blob(commit, key[5:])
                origin = "reconstructed_git_baseline"
            else:
                content = path.read_bytes()
                origin = "retained_build_overlay" if key.startswith("bridge/") else "current_source_snapshot"
            if key.startswith("external/"):
                origin = "current_external_dependency; historical source hash was not recorded"
            entries[key] = dict(path=str(path), sha256=sha(content), origin=origin)
        for path in sorted(root.rglob("flags.make")) + sorted(root.rglob("link.txt")):
            if any(part in LIBRARY_TARGETS for part in path.parts):
                key = "bridge-config/" + path.relative_to(root).as_posix()
                normalized = path.read_text().replace(str(root), "{bridge_build}")
                entries[key] = dict(proof(path), sha256=sha(normalized.encode()),
                                    raw_sha256=digest(path), origin="normalized_recorded_compile_or_link_command")
        inventories.append(entries)
        observations.append(dict(dependency_files=files, inputs=entries))
    changes = differences(*inventories, {"repo/" + BRIDGE_CPP, "repo/" + BRIDGE_HEADER})
    return dict(baseline=observations[0], current=observations[1], changes=changes,
        provenance="Old bridge source hashes were absent from the runtime manifest. Repository baseline hashes are reconstructed from the named Git commit, not retroactively asserted as recorded old build inputs. Retained overlay files and compile/link commands are compared directly. External dependency source hashes are observations of the current installation."), inventories


def passing_checks(value, label):
    require(isinstance(value, dict) and bool(value) and all(item is True for item in value.values()),
            "nonempty boolean audit checks must all pass: " + label)


def diagnostic_result(path, identity, backend):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "result.json"
    data = read(path)
    config = data["config"]
    require(data["status"] == "complete" and data["party_exit_codes"] == [0, 0], "new diagnostic is incomplete")
    require(config["binary_sha256"] == identity["binary"]["sha256"], "new diagnostic has another binary identity")
    require(config["runtime_manifest"]["sources"].get(identity["bridge"]["path"]) == identity["bridge"]["sha256"],
            "new diagnostic has another bridge identity")
    require(config["backend"] == backend and config["batch"] == 128 and config["steps"] == 1 and
            config["start"] == 0 and config["queries"] == 2 and config["warmup_queries"] == 0 and
            config["diagnostic"] is True, "new diagnostic workload is not batch128 / two fresh single steps")
    require(data["address_space_hard_limit_bytes_per_party"] == MEMORY and
            config["memory_limit_bytes_per_party"] == MEMORY, "new diagnostic memory limit differs")
    return dict(proof(path), config=config), path.parent


def integer_proof(path, mode):
    data = read(path)
    require(data.get("status") == "passed" and data.get("activation_mode") == mode and
            data.get("batch") == 128 and data.get("queries") == 2, "integer audit has another scope or failed")
    require(len(data["rows"]) == 2 and all(row["integer_values"] == 172928 and row["tensors"] == 15 and
            row["mismatches"] == 0 for row in data["rows"]), "integer audit is incomplete or differs")
    passing_checks(data["protocol"]["checks"], str(path))
    return dict(proof(path), activation_mode=mode, rows=data["rows"], protocol_checks=data["protocol"]["checks"])


def new_diagnostics(dpf_path, folded_path, identity):
    dpf, folded = read(dpf_path), read(folded_path)
    require(dpf.get("status") == folded.get("status") == "passed", "new selective qualifications must pass")
    require(dpf.get("binary_sha256") == identity["binary"]["sha256"] and
            dpf.get("bridge_sha256") == identity["bridge"]["sha256"], "DPF qualification identities differ")
    rows, states, folded_cases = [], [], []
    for row in dpf["rows"]:
        require(row.get("status") == "passed", "DPF diagnostic row failed")
        passing_checks(row.get("checks"), "DPF row")
        require({"integer", "material_residency", "joint_prefix_budget", "resources",
                 "online_prefix_ready", "actual_limit", "participant_limit"} <= set(row["checks"]),
                "DPF qualification is missing required resource/material checks")
        require(len(row.get("parties", [])) == 2 and {party["party"] for party in row["parties"]} == {1, 2},
                "DPF qualification is missing per-party evidence")
        for party in row["parties"]:
            require(len(party["resident_bytes_per_query"]) == 2 and all(
                0 < value <= party["reserve_bytes"] for value in party["resident_bytes_per_query"]),
                "DPF resident evidence is incomplete or exceeds its reservation")
            require(all(0 < party[field] <= MEMORY for field in ("vm_peak_bytes", "rss_peak_bytes")),
                    "DPF party resources exceed the declared limit")
        result, directory = diagnostic_result(row["result_path"], identity, DPF)
        config = result["config"]
        require(config["graph_mode"] == "parallel" and config["activation_workers_per_party"] == 2 and
                config["dpf_activations"] == "both", "DPF diagnostic schedule or activation selection differs")
        state = "nonzero" if config["hidden_file"] else "default"
        states.append(state)
        audit = Path(row["integer_audit_path"])
        rows.append(dict(backend=DPF, state=state, result=result,
                         integer_audit=integer_proof(audit, "sci"), resource_checks=row["checks"],
                         parties=row["parties"]))
    require(sorted(states) == ["default", "nonzero"], "DPF diagnostics must cover default and nonzero state once each")
    for row in folded["rows"]:
        require(row.get("integer_status") == row.get("residency_status") == "passed", "folded diagnostic row failed")
        result, directory = diagnostic_result(Path(folded_path).parent / row["id"], identity, FOLDED)
        config = result["config"]
        require(config["folded_sign_backend"] == "bitdec" and config["folded_activations"] == "both" and
                not config["hidden_file"], "folded selective diagnostic must use BitDec and default state")
        case = (config["graph_mode"], config["activation_workers_per_party"], config["lut_key_family"])
        folded_cases.append(case)
        residency_path = directory / "residency-audit.json"
        residency = read(residency_path)
        require(residency.get("status") == "passed" and len(residency.get("parties", [])) == 2,
                "folded residency audit failed or has missing parties")
        passing_checks(residency["checks"], "folded residency")
        for party in residency["parties"]:
            passing_checks(party["checks"], "folded party residency")
            require(all(0 < party["resources"][field] <= MEMORY for field in
                        ("conservative_vm_peak_bytes", "conservative_rss_peak_bytes")), "folded resources exceed limit")
        rows.append(dict(backend=FOLDED, case=case, result=result,
            integer_audit=integer_proof(directory / "integer-audit.json", "folded"),
            residency_audit=dict(proof(residency_path), checks=residency["checks"], parties=residency["parties"])))
    expected = [(graph, workers, family) for graph, workers in (("sequential", 1), ("parallel", 2))
                for family in ("full_q_arithmetic", "segmented_boolean_tensor_b2a")]
    require(sorted(folded_cases) == sorted(expected), "four folded construction/schedule cases are required")
    return dict(dpf_qualification=proof(dpf_path), folded_qualification=proof(folded_path), rows=rows)


def historical_proofs(paths):
    rows = []
    for path in paths:
        path = Path(path).resolve()
        data = read(path)
        require(data.get("status") == "passed" or data.get("eligible") is True, "historical qualification did not pass")
        executions = []
        for row in data.get("rows", []):
            result_path = path.parent / row.get("id", "") / "result.json"
            if result_path.exists():
                result = read(result_path)
                require(result["status"] == "complete" and result["party_exit_codes"] == [0, 0], "historical row incomplete")
                config = result["config"]
                executions.append(dict(result=proof(result_path), original_binary=config["binary"],
                    original_binary_sha256=config["binary_sha256"],
                    original_runtime_manifest=config.get("runtime_manifest"),
                    original_fixture_files=config.get("fixture_files"),
                    original_hidden_sha256=config.get("hidden_sha256"),
                    configuration={key: config.get(key) for key in ("backend", "folded_sign_backend", "lut_key_family",
                                   "graph_mode", "activation_workers_per_party", "hidden_file", "batch", "queries")}))
        rows.append(dict(proof(path), original_summary=data, original_executions=executions,
                         identity_binding="original execution results" if executions else
                         "historical cohort association; original summary did not record executable hashes"))
    return rows


def validate_historical_coverage(proofs):
    native, contexts, steps = set(), set(), set()
    for item in proofs:
        data = item["original_summary"]
        if "checked_inputs" in data:
            require(data["checked_inputs"] == {"sigmoid": 65536, "tanh": 65536} and all(
                data.get(field) is True for field in ("random_sharing", "zero_online_generation",
                    "zero_online_load", "no_native_sci_activation_inventory")),
                    "historical native domain or lifecycle evidence is incomplete")
            native.add((data["lut_family"], data["sign_backend"]))
        for row in data.get("rows", []):
            if "mode" in row:
                require(row.get("status") == "passed" and row.get("party_exit_codes") == [0, 0],
                        "historical explicit-context execution failed")
                contexts.add(row["mode"])
            if "integer_status" in row:
                require(row["integer_status"] == row.get("residency_status") == "passed" and
                        row.get("integer_values_per_query") == [172928, 172928] and
                        row.get("integer_mismatches") == [0, 0], "historical step integer/residency scope differs")
        for execution in item["original_executions"]:
            config = execution["configuration"]
            if config["backend"] == FOLDED:
                steps.add((config["folded_sign_backend"], config["lut_key_family"], config["graph_mode"],
                           config["activation_workers_per_party"], bool(config["hidden_file"])))
    expected = {(sign, family, graph, workers, nonzero) for sign in ("bitdec", "segmented_fss")
        for family in ("full_q_arithmetic", "segmented_boolean_tensor_b2a")
        for graph, workers in (("sequential", 1), ("parallel", 2)) for nonzero in (False, True)}
    require(native == {(sign, family) for sign in (0, 1) for family in (0, 1)},
            "historical native evidence must cover both sign backends and both LUT families")
    require({"private-sign-contexts", "private-sign-segmented-contexts"} <= contexts,
            "historical explicit-context coverage is incomplete")
    require(expected <= steps, "historical folded step/state/sign/construction coverage is incomplete")


def build_report(args):
    commit = subprocess.check_output(["git", "rev-parse", args.old_source_ref + "^{commit}"], cwd=str(REPO), text=True).strip()
    old, old_data, old_runtime, old_bridge = build_identity(args.old_manifest, old=True)
    new, new_data, new_runtime, new_bridge = build_identity(args.new_manifest)
    repair = reviewed_patch(commit)
    require(old["metadata"] == new["metadata"], "numeric source, compiler, or numeric build settings changed")
    require(old_data["sources"].get(str(REPO / NATIVE)) == next(row["baseline_git_sha256"] for row in repair
            if row["source"] == NATIVE), "named Git baseline differs from recorded old DPF runtime source")
    runtime_changes = differences(old["normalized_sources"], new["normalized_sources"], MANIFEST_CHANGES)
    require(runtime_changes and all(row["allowed"] for row in runtime_changes), "unapproved runtime input change")
    snapshots, _ = bridge_snapshot(old_bridge, new_bridge, commit)
    require(all(row["allowed"] for row in snapshots["changes"]), "unapproved bridge input change")
    historical = args.historical_qualification or [COHORT / "ticket08/batch128-owner/qualification.json",
        COHORT / "ticket09/step-final/qualification.json", COHORT / "ticket09/contexts-final/qualification.json"] + [
        COHORT / ("ticket09/native-final/sign%d/%d/qualification.json" % (sign, family))
        for sign in (0, 1) for family in (0, 1)]
    historical = historical_proofs(historical)
    validate_historical_coverage(historical)
    diagnostics = new_diagnostics(args.dpf_qualification, args.folded_qualification, new)
    executions = [execution for item in historical for execution in item["original_executions"]]
    require(bool(executions), "historical step execution proofs are required")
    fixtures = executions[0]["original_fixture_files"]
    require(bool(fixtures) and all(row["original_fixture_files"] == fixtures for row in executions),
            "historical fixture identities differ")
    hidden_hashes = {row["original_hidden_sha256"] for row in executions if row["original_hidden_sha256"]}
    for row in diagnostics["rows"]:
        config = row["result"]["config"]
        require(config.get("fixture_files") == fixtures, "new diagnostic fixture differs from historical qualification")
        if config["hidden_file"]:
            require(config.get("hidden_sha256") in hidden_hashes,
                    "new diagnostic nonzero state differs from historical qualification")
    return dict(schema=1, status="passed", eligible=True,
        recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Qualification reuse after DPF resident metadata budget repair; no formal timing samples are reused",
        original_build=old, current_build=new, old_source_commit=commit,
        allowed_changes=dict(runtime_inputs=sorted(MANIFEST_CHANGES), repository_sources=sorted(SOURCE_CHANGES),
            meaning="Exact metadata accessor addition and use of actual retained material metadata in DPF reservation/precheck; protocol and folded paths are unchanged"),
        reviewed_source_repairs=repair, runtime_manifest_changes=runtime_changes,
        bridge_source_snapshot=snapshots, historical_qualifications=historical,
        current_diagnostic_proofs=diagnostics, fixture_files=fixtures,
        reused_coverage=["both LUT families and both sign backends over all 65536 native inputs",
            "explicit nondefault contexts, ownership/shape/stock rejection", "default and nonzero complete folded steps",
            "unchanged folded scheduling, inventory and resource paths"],
        current_coverage=["DPF default and nonzero batch128 single steps, two fresh queries each",
            "BitDec default batch128 single steps, both LUT families with sequential-a1 and parallel-a2, two fresh queries each"],
        limitations=["Historical bridge repository sources are reconstructed from Git because their hashes were absent from the original runtime manifest",
            "External bridge dependency headers have current snapshots, not historical build source attestations",
            "Reused historical qualifications retain their original binary identities and are not claimed as new executions",
            "Formal ten-row performance evidence must be regenerated under the current frozen common build"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, default=Path("/tmp/paper-folded-delivery-runtime/runtime_manifest.json"))
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--old-source-ref", default="HEAD")
    parser.add_argument("--historical-qualification", type=Path, action="append")
    parser.add_argument("--dpf-qualification", type=Path, required=True)
    parser.add_argument("--folded-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "qualification reuse report must not overwrite an existing artifact")
    try:
        report = build_report(args)
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        report = dict(schema=1, status="failed", eligible=False, reason=str(error))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output), reason=report.get("reason"))))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
