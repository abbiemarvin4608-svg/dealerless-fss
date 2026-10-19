"""Actual resident SCI workers own fresh material under an observed four-core limit."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import tempfile
import unittest

from build import native_io_overlay, native_ot_overlays, native_overlays
from native_got_overlay import native_got_overlay
from native_planning_overlay import native_planning_overlays
from native_triple_overlay import native_triple_overlay
from run import records, validate_native_materials, validate_native_public_plan, validate_native_triples

HERE = Path(__file__).resolve().parent
SCI = Path("/home/jiang/EzPC/SCI")


def observe_legacy_teams(out, overrides):
    # Observe unchanged fixed-size teams so an uninstrumented quota cannot pass.
    aux = Path(overrides.get("BuildingBlocks/aux-protocols.cpp", SCI / "src/BuildingBlocks/aux-protocols.cpp"))
    text = aux.read_text()
    observed_sites = 0
    if "sdcf_native::observe_ot_team" not in text:
        text = text.replace("#pragma omp parallel num_threads(2)",
            "auto* context_observer = sdcf_native::team_observer();\n#pragma omp parallel num_threads(2)")
        text = text.replace("if (omp_get_thread_num() == 1)",
            "sdcf_native::observe_ot_team(context_observer, omp_get_num_threads());\n    if (omp_get_thread_num() == 1)")
        path = out / "aux_observed.cpp"
        path.write_text('#include "native_workers.h"\n' + text)
        overrides["BuildingBlocks/aux-protocols.cpp"] = str(path)
        observed_sites += 1
    for name in ("millionaire.h", "millionaire_with_equality.h", "equality.h"):
        path = out / "include_override/Millionaire" / name
        text = path.read_text() if path.exists() else (SCI / "src/Millionaire" / name).read_text()
        if "sdcf_native::observe_ot_team" in text:
            continue
        needle = "#pragma omp parallel num_threads(2)"
        if text.count(needle) != 1:
            raise AssertionError("comparison observer source changed: " + name)
        text = text.replace(needle, "auto* context_observer = sdcf_native::team_observer();\n" + needle)
        text = text.replace("if (omp_get_thread_num() == 1)",
            "sdcf_native::observe_ot_team(context_observer, omp_get_num_threads());\n    if (omp_get_thread_num() == 1)")
        path.write_text('#include "native_workers.h"\n' + text)
        observed_sites += 1
    return observed_sites


class NativeContextMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="native-context-material-")
        out = Path(cls.directory.name)
        native_ot_overlays(out, SCI)
        native_io_overlay(out, SCI)
        native_got_overlay(out)
        native_planning_overlays(out, SCI)
        native_triple_overlay(out, SCI)
        overrides = {}
        native_overlays(out, SCI, overrides)
        legacy_observation_sites = observe_legacy_teams(out, overrides)
        for name in ("native_dpf_lut.h", "native_preprocessing.h", "native_planning.h", "native_plan_replay.h", "native_query_materials.h",
                     "prepared_got.h", "prepared_cot.h", "prepared_triples.h", "measurement.h", "io_counters.h",
                     "backend_types.h", "backend_adapter.h", "native_workers.h", "matrix_metrics.h",
                     "test_native_context_materials.cpp", "runtime_log.h", "native_workers.cpp", "round_trace.h", "round_trace.cpp"):
            if (HERE / name).exists():
                (out / name).write_bytes((HERE / name).read_bytes())
        cls.binary = out / "context-test"
        command = ["g++", "-O3", "-DNDEBUG", "-std=c++17", "-maes", "-msse4.1", "-mavx2", "-mrdseed",
                   "-pthread", "-fopenmp", "-DSCI_OT=1", "-DEIGEN_DONT_PARALLELIZE", "-Wno-deprecated-declarations",
                   "-I" + str(out), "-I" + str(out / "include_override"), "-I" + str(SCI / "src"),
                   "-I" + str(SCI / "extern/eigen"), str(out / "test_native_context_materials.cpp"),
                   str(out / "native_workers.cpp"), str(out / "round_trace.cpp")]
        command += [str(overrides.get(name, SCI / "src" / name)) for name in
                    ("Math/math-functions.cpp", "LinearOT/linear-ot.cpp", "BuildingBlocks/aux-protocols.cpp",
                     "BuildingBlocks/truncation.cpp", "BuildingBlocks/value-extension.cpp")]
        command += ["-Wl,--start-group", str(SCI / "build/install/lib/libSCI-OT.a"),
                    str(SCI / "build/install/lib/libSCI-GC.a"), "-Wl,--end-group", "-lssl", "-lcrypto", "-lgmp",
                    "-o", str(cls.binary)]
        with open("/tmp/sci-preprocessed-build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        if os.environ.get("SDCF_CONTEXT_EVIDENCE_DIR"):
            evidence = Path(os.environ["SDCF_CONTEXT_EVIDENCE_DIR"]) / "build"
            evidence.mkdir(parents=True, exist_ok=True)
            paths = [out / "test_native_context_materials.cpp", out / "native_workers.cpp", out / "round_trace.cpp",
                     Path(overrides["LinearOT/linear-ot.cpp"]),
                     Path(overrides["BuildingBlocks/aux-protocols.cpp"])]
            paths += sorted(out.glob("*.h"))
            paths += list((out / "include_override").rglob("*.h"))
            hashes = {}
            for path in paths:
                relative = path.relative_to(out)
                target = evidence / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            harness = HERE / "test_native_context_materials.py"
            (evidence / harness.name).write_bytes(harness.read_bytes())
            hashes[harness.name] = hashlib.sha256(harness.read_bytes()).hexdigest()
            (evidence / "manifest.json").write_text(json.dumps(dict(command=command,
                binary_sha256=hashlib.sha256(cls.binary.read_bytes()).hexdigest(), source_sha256=hashes,
                legacy_observation_sites=legacy_observation_sites), indent=2) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_parties(self, mode, workers, graph_mode, count=501):
        for port in range(28731, 46000, 173):
            sockets = []
            try:
                for offset in (0, 50, 100, 300, 350, 400, 450, 500, 550, 600, 650, 700):
                    sock = socket.socket(); sockets.append(sock); sock.bind(("127.0.0.1", port + offset))
                break
            except OSError:
                pass
            finally:
                for sock in sockets:
                    sock.close()
        def limits(role):
            def apply():
                os.sched_setaffinity(0, (0, 2, 4, 10) if role == 1 else (6, 8, 12, 16))
                resource.setrlimit(resource.RLIMIT_AS, (1536 << 20, 1536 << 20))
            return apply
        environment = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE",
            SDCF_ACTIVATION_WORKERS=str(workers), SDCF_GRAPH_MODE=graph_mode,
            SDCF_GRAPH_OPENINGS="independent", SDCF_OFFLINE_OVERLAP="0",
            SEGFSS_CGGM_THREADS="2" if graph_mode != "sequential" else "4")
        commands = [[str(self.binary), str(role), str(port), mode, str(count)] for role in (1, 2)]
        outputs = [tempfile.TemporaryFile(mode="w+t") for _ in (1, 2)]
        children = []
        with open("/tmp/sci-preprocessed-build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                children = [subprocess.Popen(commands[role - 1], stdout=outputs[role - 1], stderr=subprocess.STDOUT,
                    text=True, preexec_fn=limits(role), env=environment) for role in (1, 2)]
                for child in children:
                    child.wait(timeout=120)
                logs = []
                for output in outputs:
                    output.seek(0); logs.append(output.read())
                exits = [child.returncode for child in children]
                if os.environ.get("SDCF_CONTEXT_EVIDENCE_DIR"):
                    evidence = Path(os.environ["SDCF_CONTEXT_EVIDENCE_DIR"]) / ("%s_a%d_%s_%d" % (mode, workers, graph_mode, count))
                    evidence.mkdir(parents=True, exist_ok=True)
                    for role, log in enumerate(logs, 1):
                        (evidence / ("party%d.log" % role)).write_text(log)
                    (evidence / "result.json").write_text(json.dumps(dict(mode=mode, workers=workers, graph_mode=graph_mode,
                        count=count, party_exit_codes=exits, commands=commands,
                        environment={name: environment[name] for name in ("OMP_NUM_THREADS", "OMP_DYNAMIC",
                            "SDCF_ACTIVATION_WORKERS", "SDCF_GRAPH_MODE", "SEGFSS_CGGM_THREADS")},
                        per_party_address_space_bytes=1536 << 20, cpu_affinity=[[0, 2, 4, 10], [6, 8, 12, 16]]), indent=2) + "\n")
                return exits, logs
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill(); child.wait()
                for output in outputs:
                    output.close()

    def check_worker_case(self, mode, workers, graph_mode, expected_outputs=None):
        exits, logs = self.run_parties(mode, workers, graph_mode)
        self.assertEqual(exits, [0, 0], [log[-6000:] for log in logs])
        outputs = records(logs[1], "ACTIVATION_OUTPUT")
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(len(row["values"]) == 501 for row in outputs))
        if expected_outputs is not None:
            self.assertEqual(outputs, expected_outputs)
        plans = []
        for log in logs:
            worker_rows = records(log, "ACTIVATION_WORKER")
            self.assertEqual(len(worker_rows), 4 * workers)
            for query in range(2):
                for name, index in (("Sigmoid", 0), ("TanH", 1)):
                    rows = sorted((row for row in worker_rows if row["query"] == query and row["function"] == name),
                                  key=lambda row: row["begin"])
                    chunks = [501] if workers == 1 else [126, 125, 125, 125] if workers == 4 else [251, 250]
                    slots = list(range(workers)) if graph_mode == "sequential" or index == 0 else [2, 3]
                    self.assertEqual([row["count"] for row in rows], chunks)
                    self.assertEqual([row["slot"] for row in rows], slots)
                    self.assertEqual([row["begin"] for row in rows], [sum(chunks[:i]) for i in range(workers)])
                    self.assertTrue(all(row["ot_participants"] == (2 if workers == 1 else 1) for row in rows), rows)
            self.assertTrue(all(row["compute_participants"] <= 4 for row in records(log, "ACTIVATION_DETAIL")))
            if graph_mode == "parallel":
                groups = records(log, "GRAPH_GROUP")
                self.assertEqual(len(groups), 2)
                self.assertTrue(all(row["observed_compute_upper_bound"] <= 4 for row in groups))
            if mode == "prepared":
                ready = records(log, "MATERIAL_READY")
                consumed = records(log, "NATIVE_MATERIAL_CONSUMED")
                valid, _ = validate_native_materials(ready, records(log, "NATIVE_MATERIAL_PREPARED"), consumed,
                    records(log, "NATIVE_MATERIAL_BALANCE"), records(log, "METRIC"), 2)
                self.assertTrue(valid)
                plan = validate_native_public_plan(records(log, "NATIVE_PUBLIC_PLAN"), consumed, 2)
                plans.append(plan)
                triple_valid, _ = validate_native_triples(records(log, "NATIVE_PUBLIC_TRIPLE_PLAN"),
                    records(log, "NATIVE_TRIPLE_PREPARED"), records(log, "NATIVE_TRIPLE_CONSUMED"),
                    records(log, "NATIVE_TRIPLE_BALANCE"), ready, 2)
                self.assertTrue(triple_valid)
                self.assertEqual({row["context"] for row in consumed}, set(range(4)) if graph_mode != "sequential" else set(range(workers)))
                self.assertTrue(all(row["native_generation_epoch_delta"] == row["bridge_generation_epoch_delta"] == 0
                    for row in records(log, "METRIC") if row["phase"] in ("input", "ready", "online", "output")))
                lines = log.splitlines()
                for query in range(2):
                    ready_index = next(i for i, line in enumerate(lines) if line.startswith("MATERIAL_READY ") and json.loads(line.split(" ", 1)[1])["query"] == query)
                    input_index = next(i for i, line in enumerate(lines) if line.startswith("QUERY_INPUT_LOADED ") and json.loads(line.split(" ", 1)[1])["query"] == query)
                    self.assertLess(ready_index, input_index)
        if plans:
            self.assertEqual(plans[0], plans[1])
        return outputs

    def test_a4_uneven_chunks_observe_actual_quota_and_consume_fresh_stock(self):
        reference = self.check_worker_case("default", 1, "sequential")
        self.check_worker_case("default", 4, "sequential", reference)
        self.check_worker_case("prepared", 4, "sequential", reference)

    def test_a2_owned_graph_branches_preserve_integers_and_material_under_order_change(self):
        reference = self.check_worker_case("default", 1, "sequential")
        for graph_mode in ("isolated", "parallel"):
            self.check_worker_case("default", 2, graph_mode, reference)
            self.check_worker_case("prepared", 2, graph_mode, reference)

    def test_real_b2a_rejects_other_context_and_direction_without_generation(self):
        for mode in ("wrong_context", "wrong_direction"):
            exits, logs = self.run_parties(mode, 1, "sequential", 13)
            self.assertEqual(exits, [2, 2], [log[-6000:] for log in logs])
            for log in logs:
                rejection = records(log, "CONTEXT_REJECTION")
                self.assertEqual(rejection, [dict(case=mode, native_generation_epoch_delta=0)])
                self.assertIn("inventory exhausted", log)
                self.assertNotIn("unowned B2A material was accepted", log)


if __name__ == "__main__":
    unittest.main()
