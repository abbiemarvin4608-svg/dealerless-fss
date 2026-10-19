"""Real two-party SCI comparisons and complete integer conversions."""
import fcntl
import json
import os
import resource
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

from build import native_io_overlay, native_ot_overlays
from native_triple_overlay import native_triple_overlay
from native_planning_overlay import native_planning_overlays
from native_got_overlay import native_got_overlay

HERE = Path(__file__).resolve().parent
SCI = Path("/home/jiang/EzPC/SCI")


class PreparedTripleConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="prepared-conversion-test-")
        out = Path(cls.directory.name)
        native_ot_overlays(out, SCI)
        native_io_overlay(out, SCI)
        native_planning_overlays(out, SCI)
        native_got_overlay(out)
        native_triple_overlay(out, SCI)
        for name in ("native_preprocessing.h", "native_planning.h", "native_plan_replay.h",
                     "prepared_got.h", "prepared_cot.h", "prepared_triples.h", "measurement.h",
                     "io_counters.h", "backend_types.h", "backend_adapter.h", "test_prepared_triples.cpp"):
            (out / name).write_bytes((HERE / name).read_bytes())
        cls.binary = out / "conversion-test"
        flags = ["g++", "-O2", "-std=c++17", "-maes", "-msse4.1", "-mavx2", "-mrdseed", "-pthread", "-fopenmp", "-DSCI_OT=1",
                 "-Wno-deprecated-declarations", "-I" + str(out), "-I" + str(out / "include_override"),
                 "-I" + str(SCI / "src"), "-I" + str(SCI / "extern/eigen")]
        command = flags + [str(out / "test_prepared_triples.cpp"),
                           str(SCI / "src/BuildingBlocks/aux-protocols.cpp"),
                           str(SCI / "src/BuildingBlocks/value-extension.cpp"),
                           str(SCI / "src/BuildingBlocks/truncation.cpp"),
                           "-Wl,--start-group", str(SCI / "build/install/lib/libSCI-OT.a"),
                           str(SCI / "build/install/lib/libSCI-BuildingBlocks.a"),
                           str(SCI / "build/install/lib/libSCI-GC.a"), "-Wl,--end-group",
                           "-lssl", "-lcrypto", "-lgmp", "-o", str(cls.binary)]
        with open("/tmp/sci-preprocessed-build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_parties(self, mode):
        for port in range(25023, 45000, 179):
            sockets = []
            try:
                for offset in (0, 50, 100):
                    sock = socket.socket()
                    sockets.append(sock)
                    sock.bind(("127.0.0.1", port + offset))
                break
            except OSError:
                pass
            finally:
                for sock in sockets:
                    sock.close()
        cpus = sorted(os.sched_getaffinity(0))
        def limits(role):
            def apply():
                os.sched_setaffinity(0, cpus[(role - 1) * 4:role * 4] or cpus[:4])
                resource.setrlimit(resource.RLIMIT_AS, (1536 << 20, 1536 << 20))
            return apply
        output_files = [tempfile.TemporaryFile(mode="w+t") for _ in (1, 2)]
        children = [subprocess.Popen([str(self.binary), str(role), str(port), mode],
                                    stdout=output_files[role - 1], stderr=subprocess.STDOUT, text=True,
                                    preexec_fn=limits(role)) for role in (1, 2)]
        try:
            for child in children:
                child.wait(timeout=90)
            logs = []
            for output in output_files:
                output.seek(0)
                logs.append(output.read())
            if os.environ.get("SDCF_TRIPLE_EVIDENCE_DIR"):
                evidence = Path(os.environ["SDCF_TRIPLE_EVIDENCE_DIR"]) / mode
                evidence.mkdir(parents=True, exist_ok=True)
                for role, log in enumerate(logs, 1):
                    (evidence / ("party%d.log" % role)).write_text(log)
                (evidence / "result.json").write_text(json.dumps({"mode": mode,
                    "party_exit_codes": [child.returncode for child in children]}, indent=2) + "\n")
            return [child.returncode for child in children], logs
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            for output in output_files:
                output.close()

    @staticmethod
    def rows(log, prefix):
        return [json.loads(line.split(" ", 1)[1]) for line in log.splitlines() if line.startswith(prefix + " ")]

    def assert_prepared(self, logs):
        for log in logs:
            metrics = self.rows(log, "METRIC")
            self.assertTrue(all(row["native_generation_epoch_delta"] == 0 for row in metrics
                                if row["phase"] in ("input", "online", "output")), log)
            self.assertTrue(all(row["native_generation_epoch_delta"] > 0 for row in metrics
                                if row["phase"] == "offline"), log)
            self.assertLess(log.index("MATERIAL_READY "), log.index('"phase":"input"'))
            balances = self.rows(log, "NATIVE_TRIPLE_BALANCE")
            self.assertEqual(len(balances), 2, log)
            self.assertTrue(all(row["requested"] == row["generated"] == row["consumed"] and
                                row["remaining"] == row["padding_discarded"] == 0 for row in balances), log)

    def test_safe_standard_and_correlated_material_is_ready_before_input(self):
        exits, logs = self.run_parties("triples_prepared")
        self.assertEqual(exits, [0, 0], logs)
        self.assert_prepared(logs)
        for log in logs:
            self.assertEqual(len(self.rows(log, "TRIPLE_RESULT")), 2)
            self.assertEqual(len(self.rows(log, "NATIVE_TRIPLE_CONSUMED")), 8)

    def test_complete_conversions_match_default_integer_contract_at_edges(self):
        reference = None
        for mode in ("conversions_default", "conversions_prepared"):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [0, 0], logs)
            values = self.rows(logs[1], "CONVERSION_RESULT")
            self.assertEqual(len(values), 42, logs)
            if reference is None:
                reference = values
            else:
                self.assertEqual(values, reference)
                self.assert_prepared(logs)
                for log in logs:
                    prepared = self.rows(log, "NATIVE_TRIPLE_PREPARED")
                    self.assertTrue(all(row["standard_calls"] > 0 and row["correlated_calls"] > 0 for row in prepared))
                    plans = self.rows(log, "PLAN_RESULT")
                    self.assertEqual(len(plans), 2)
                    self.assertTrue(all(row["secret_independent"] for row in plans))
                    coverage = self.rows(log, "CONVERSION_MATERIAL_COVERAGE")
                    self.assertEqual(len(coverage), 42)
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 0 for row in coverage))
                    self.assertTrue(all(row["native_ot_consumed"] > 0 or row["bit_triples_consumed"] > 0
                                        for row in coverage if row["protocol"] != "compare_equality"))

    def test_triple_inventory_fails_closed(self):
        for mode, reason in (("unsafe", "unsafe bit-triple generation method"),
                             ("unsafe_prepare", "unsafe bit-triple generation method"),
                             ("mismatch", "bit-triple material contract mismatch"),
                             ("repeat", "bit-triple inventory exhausted"),
                             ("incomplete", "unconsumed bit-triple material"),
                             ("online_generation", "material generation after prepare")):
            with self.subTest(mode=mode):
                exits, logs = self.run_parties(mode)
                self.assertEqual(exits, [2, 2], logs)
                self.assertTrue(all(reason in log for log in logs), logs)


if __name__ == "__main__":
    unittest.main()
