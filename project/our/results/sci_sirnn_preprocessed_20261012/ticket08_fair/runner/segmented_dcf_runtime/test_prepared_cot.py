"""Real native arithmetic-COT callers with pre-input random material."""
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

HERE = Path(__file__).resolve().parent
SCI = Path("/home/jiang/EzPC/SCI")


class PreparedCOTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="prepared-cot-test-")
        out = Path(cls.directory.name)
        native_ot_overlays(out, SCI)
        native_io_overlay(out, SCI)
        for name in ("native_preprocessing.h", "prepared_cot.h", "measurement.h", "io_counters.h",
                     "backend_types.h", "backend_adapter.h", "test_prepared_cot.cpp"):
            (out / name).write_bytes((HERE / name).read_bytes())
        cls.binary = out / "cot-test"
        flags = ["g++", "-O2", "-std=c++17", "-maes", "-msse4.1", "-mavx2", "-mrdseed", "-pthread", "-fopenmp", "-DSCI_OT=1",
                 "-Wno-deprecated-declarations", "-I" + str(out), "-I" + str(out / "include_override"),
                 "-I" + str(SCI / "src"), "-I" + str(SCI / "extern/eigen")]
        command = flags + [str(out / "test_prepared_cot.cpp"), str(SCI / "src/BuildingBlocks/aux-protocols.cpp"),
                           str(SCI / "src/LinearOT/linear-ot.cpp"),
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
        for port in range(25011, 45000, 173):
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
                chosen = cpus[(role - 1) * 4:role * 4] or cpus[:4]
                os.sched_setaffinity(0, chosen)
                resource.setrlimit(resource.RLIMIT_AS, (int(1.5 * (1 << 30)), int(1.5 * (1 << 30))))
            return apply
        children = [subprocess.Popen([str(self.binary), str(role), str(port), mode],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    preexec_fn=limits(role))
                    for role in (1, 2)]
        try:
            logs = [child.communicate(timeout=45)[0] for child in children]
            if os.environ.get("SDCF_COT_EVIDENCE_DIR"):
                evidence = Path(os.environ["SDCF_COT_EVIDENCE_DIR"]) / mode
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

    def test_native_b2a_prepares_before_input_and_keeps_corrections_online(self):
        for mode in ("prepared", "default", "offline_cot"):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [0, 0], logs)
            for party_index, log in enumerate(logs):
                self.assertEqual(log.count('"correct":true'), 2)
                metrics = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
                           if line.startswith("METRIC ")]
                if mode != "default":
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 0
                                        for row in metrics if row["phase"] == "online"))
                    self.assertTrue(all(row["sent_bytes"] == (16 if party_index == 0 else 3)
                                        for row in metrics if row["phase"] == "online"))
                    self.assertTrue(all(row["native_generation_epoch_delta"] == (2 if mode == "offline_cot" else 1)
                                        for row in metrics if row["phase"] == "offline"))
                    self.assertLess(log.index("MATERIAL_READY "), log.index('"phase":"input"'))
                else:
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 1
                                        for row in metrics if row["phase"] == "online"))


    def test_native_matmul_and_mulcir_use_mixed_width_cot_in_both_directions(self):
        for mode in ("multiplication_prepared", "multiplication_default"):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [0, 0], logs)
            for log in logs:
                self.assertEqual(log.count('"correct":true'), 2)
                if mode == "multiplication_prepared":
                    consumed = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
                                if line.startswith("NATIVE_MATERIAL_CONSUMED ")]
                    self.assertEqual(len(consumed), 8)
                    self.assertTrue(all(row["widths"] == [7, 6, 5, 4, 3, 2, 1] for row in consumed))
                    metrics = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
                               if line.startswith("METRIC ")]
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 0
                                        for row in metrics if row["phase"] == "online"))
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 4
                                        for row in metrics if row["phase"] == "offline"))
                    self.assertTrue(all(row["sent_bytes"] == 47
                                        for row in metrics if row["phase"] == "online"))


    def test_arithmetic_ring_boundaries_mixed_width_tails_and_long_lanes(self):
        for mode in ("edges_prepared", "edges_default"):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [0, 0], logs)
            for log in logs:
                self.assertEqual(log.count('"correct":true'), 2)
                if mode == "edges_prepared":
                    balances = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
                                if line.startswith("NATIVE_MATERIAL_BALANCE ")]
                    self.assertEqual(len(balances), 4)
                    self.assertTrue(all((row["requested"], row["generated"], row["consumed"], row["padding_discarded"])
                                        == (322, 1280, 322, 958) for row in balances))
                    metrics = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
                               if line.startswith("METRIC ")]
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 0
                                        for row in metrics if row["phase"] == "online"))

    def test_arithmetic_stock_fails_closed(self):
        for mode, reason in (("width_mismatch", "arithmetic COT material contract mismatch"),
                             ("group_mismatch", "arithmetic COT material contract mismatch"),
                             ("repeat", "arithmetic COT inventory exhausted"),
                             ("incomplete", "unconsumed native material"),
                             ("unfinished_random", "native random OT outputs incomplete"),
                             ("online_extension", "material generation after prepare")):
            with self.subTest(mode=mode):
                exits, logs = self.run_parties(mode)
                self.assertEqual(exits, [2, 2], logs)
                self.assertTrue(all(reason in log for log in logs), logs)


if __name__ == "__main__":
    unittest.main()
