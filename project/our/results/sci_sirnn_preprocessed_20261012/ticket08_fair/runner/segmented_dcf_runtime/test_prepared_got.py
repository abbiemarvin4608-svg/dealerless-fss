"""Real two-party GOT preparation, selection correction, and material lifetime."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

from build import native_io_overlay, native_ot_overlays
from run import records, validate_native_materials


HERE = Path(__file__).resolve().parent
SCI = Path("/home/jiang/EzPC/SCI")


class PreparedGOTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="prepared-got-test-")
        out = Path(cls.directory.name)
        native_ot_overlays(out, SCI)
        native_io_overlay(out, SCI)
        for name in ("native_preprocessing.h", "measurement.h", "io_counters.h", "backend_types.h", "backend_adapter.h"):
            (out / name).write_bytes((HERE / name).read_bytes())
        cls.binary = out / "got-test"
        (out / "test_prepared_got.cpp").write_bytes((HERE / "test_prepared_got.cpp").read_bytes())
        command = ["g++", "-O2", "-std=c++17", "-maes", "-msse4.1", "-mavx2", "-mrdseed", "-pthread",
                   "-Wno-deprecated-declarations", "-I" + str(out), "-I" + str(out / "include_override"),
                   "-I" + str(SCI / "src"), str(out / "test_prepared_got.cpp"),
                   str(SCI / "build/install/lib/libSCI-OT.a"), "-lssl", "-lcrypto", "-lgmp", "-o", str(cls.binary)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_parties(self, mode):
        for port in range(25000, 45000, 173):
            sockets = []
            try:
                for offset in (0, 50, 100):
                    sock = socket.socket(); sockets.append(sock)
                    sock.bind(("127.0.0.1", port + offset))
                break
            except OSError:
                pass
            finally:
                for sock in sockets: sock.close()
        children = [subprocess.Popen([str(self.binary), str(role), str(port), mode],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for role in (1, 2)]
        try:
            logs = [child.communicate(timeout=40)[0] for child in children]
            if os.environ.get("SDCF_GOT_EVIDENCE_DIR"):
                evidence = Path(os.environ["SDCF_GOT_EVIDENCE_DIR"]) / mode
                evidence.mkdir(parents=True, exist_ok=True)
                for role, log in enumerate(logs, 1):
                    (evidence / ("party%d.log" % role)).write_text(log)
                (evidence / "result.json").write_text(json.dumps({"mode": mode,
                    "party_exit_codes": [child.returncode for child in children],
                    "commands": [[str(self.binary), str(role), str(port), mode] for role in (1, 2)]}, indent=2) + "\n")
            return [child.returncode for child in children], logs
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill(); child.wait()

    def test_fresh_prepared_ot_matches_default_and_charges_one_bit_correction(self):
        prepared_exits, prepared = self.run_parties("prepared")
        default_exits, default = self.run_parties("default")
        self.assertEqual(prepared_exits, [0, 0], prepared)
        self.assertEqual(default_exits, [0, 0], default)
        for logs in (prepared, default):
            for log in logs:
                self.assertEqual(log.count('"correct":true'), 2)
        for log in prepared:
            verified, _ = validate_native_materials(records(log, "MATERIAL_READY"),
                records(log, "NATIVE_MATERIAL_PREPARED"), records(log, "NATIVE_MATERIAL_CONSUMED"),
                records(log, "NATIVE_MATERIAL_BALANCE"), records(log, "METRIC"), 2)
            self.assertTrue(verified)
            material = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
                        if line.startswith("NATIVE_MATERIAL_BALANCE ")]
            self.assertEqual([(row["requested"], row["generated"], row["consumed"], row["padding_discarded"])
                              for row in material], [(12, 256, 12, 244), (12, 256, 12, 244)])
            phases = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines() if line.startswith("METRIC ")]
            for row in phases:
                if row["phase"] == "online":
                    self.assertEqual(row["native_generation_epoch_delta"], 0)
                if row["phase"] == "offline":
                    self.assertEqual(row["native_generation_epoch_delta"], 1)
            self.assertLess(log.index("MATERIAL_READY "), log.index('"phase":"input"'))
        online = [[json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
                   if line.startswith("METRIC ") and '"phase":"online"' in line] for log in prepared]
        self.assertEqual([row["sent_bytes"] for row in online[1]], [4, 4])

    def test_material_errors_and_post_ready_generation_fail_closed(self):
        for mode, reason in (("wrong_contract", "material contract mismatch"),
                             ("incomplete", "unconsumed native material"),
                             ("repeat", "material inventory exhausted"),
                             ("unfinished_random_ot", "native random OT outputs incomplete"),
                             ("extend_during_input", "material generation after prepare"),
                             ("extend_after_ready", "material generation after prepare"),
                             ("extend_during_output", "material generation after prepare")):
            with self.subTest(mode=mode):
                exits, logs = self.run_parties(mode)
                self.assertEqual(exits, [2, 2], logs)
                self.assertTrue(all(reason in log for log in logs), logs)


if __name__ == "__main__":
    unittest.main()
