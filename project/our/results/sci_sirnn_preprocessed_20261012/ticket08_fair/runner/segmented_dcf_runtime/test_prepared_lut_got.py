"""Native two-party LUT and bit-matrix GOT with input-independent preparation."""
import json
import os
from pathlib import Path
import socket
import resource
import subprocess
import tempfile
import unittest

from build import native_io_overlay, native_ot_overlays
from native_got_overlay import native_got_overlay
from run import records, validate_native_materials

HERE = Path(__file__).resolve().parent
SCI = Path("/home/jiang/EzPC/SCI")


class PreparedLUTGOTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="prepared-lut-got-")
        out = Path(cls.directory.name)
        native_ot_overlays(out, SCI)
        native_got_overlay(out)
        native_io_overlay(out, SCI)
        for name in ("native_preprocessing.h", "prepared_got.h", "measurement.h", "io_counters.h", "backend_types.h", "backend_adapter.h"):
            (out / name).write_bytes((HERE / name).read_bytes())
        cls.binary = out / "lut-got-test"
        (out / "test_prepared_lut_got.cpp").write_bytes((HERE / "test_prepared_lut_got.cpp").read_bytes())
        command = ["g++", "-O2", "-std=c++17", "-maes", "-msse4.1", "-mavx2", "-mrdseed", "-pthread", "-fopenmp",
                   "-Wno-deprecated-declarations", "-I" + str(out), "-I" + str(out / "include_override"),
                   "-I" + str(SCI / "src"), "-I" + str(SCI / "extern/eigen"), str(out / "test_prepared_lut_got.cpp"),
                   str(SCI / "src/BuildingBlocks/aux-protocols.cpp"), str(SCI / "src/LinearOT/linear-ot.cpp"),
                   str(SCI / "src/BuildingBlocks/truncation.cpp"), str(SCI / "src/BuildingBlocks/value-extension.cpp"),
                   str(SCI / "build/install/lib/libSCI-OT.a"), str(SCI / "build/install/lib/libSCI-GC.a"),
                   "-lssl", "-lcrypto", "-lgmp", "-o", str(cls.binary)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode: raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_parties(self, mode):
        for port in range(27000, 45000, 173):
            sockets = []
            try:
                for offset in (0, 50, 100):
                    sock = socket.socket(); sockets.append(sock); sock.bind(("127.0.0.1", port + offset))
                break
            except OSError: pass
            finally:
                for sock in sockets: sock.close()
        def resources(role):
            def configure():
                allowed = sorted(os.sched_getaffinity(0))
                selected = (0, 2, 4, 10) if role == 1 else (6, 8, 12, 16)
                selected = [cpu for cpu in selected if cpu in allowed]
                os.sched_setaffinity(0, selected or allowed[:4])
                resource.setrlimit(resource.RLIMIT_AS, (1536 << 20, 1536 << 20))
            return configure
        children = [subprocess.Popen([str(self.binary), str(role), str(port), mode],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    preexec_fn=resources(role), env=dict(os.environ, OMP_NUM_THREADS="2"))
                    for role in (1, 2)]
        try:
            logs = [child.communicate(timeout=60)[0] for child in children]
            if os.environ.get("SDCF_LUT_GOT_EVIDENCE_DIR"):
                evidence = Path(os.environ["SDCF_LUT_GOT_EVIDENCE_DIR"]) / mode
                evidence.mkdir(parents=True, exist_ok=True)
                for role, log in enumerate(logs, 1):
                    (evidence / ("party%d.log" % role)).write_text(log)
                (evidence / "result.json").write_text(json.dumps({"mode": mode,
                    "party_exit_codes": [child.returncode for child in children]}, indent=2) + "\n")
            return [child.returncode for child in children], logs
        finally:
            for child in children:
                if child.poll() is None: child.kill(); child.wait()

    def test_real_lut_matches_default_with_fresh_pre_input_kkot(self):
        for mode in ("prepared", "default"):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [0, 0], logs)
            for role, log in enumerate(logs, 1):
                self.assertEqual(len(records(log, "LUT_GOT_RESULT")), 2)
                if mode == "prepared":
                    valid, _ = validate_native_materials(records(log, "MATERIAL_READY"),
                        records(log, "NATIVE_MATERIAL_PREPARED"), records(log, "NATIVE_MATERIAL_CONSUMED"),
                        records(log, "NATIVE_MATERIAL_BALANCE"), records(log, "METRIC"), 2)
                    self.assertTrue(valid)
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 0
                        for row in records(log, "METRIC") if row["phase"] == "online"))
                    self.assertEqual([row["choice_correction_bits"] for row in records(log, "NATIVE_MATERIAL_CONSUMED")], [33, 33])
                    self.assertEqual([row["sent_bytes"] for row in records(log, "METRIC") if row["phase"] == "online"],
                                     [145, 145] if role == 1 else [6, 6])

    def test_real_bit_matrices_and_long_mixed_groups_match_default(self):
        for mode in ("batched_prepared", "batched_default"):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [0, 0], logs)
            for log in logs:
                self.assertEqual(len(records(log, "BATCHED_GOT_RESULT")), 2)
                if mode == "batched_prepared":
                    valid, _ = validate_native_materials(records(log, "MATERIAL_READY"),
                        records(log, "NATIVE_MATERIAL_PREPARED"), records(log, "NATIVE_MATERIAL_CONSUMED"),
                        records(log, "NATIVE_MATERIAL_BALANCE"), records(log, "METRIC"), 2)
                    self.assertTrue(valid)
                    contracts = records(log, "NATIVE_MATERIAL_CONSUMED")
                    self.assertEqual(len(contracts), 16)
                    self.assertEqual({(row["message_bits"], row["messages_per_ot"]) for row in contracts},
                                     {(13, 11), (17, 11), (64, 3), (5, 1)})
                    self.assertTrue(all(row["native_generation_epoch_delta"] == 0
                        for row in records(log, "METRIC") if row["phase"] == "online"))

    def test_all_lut_radices_and_byte_word_widths(self):
        for mode in ("lut_all_prepared", "lut_all_default"):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [0, 0], logs)
            for log in logs:
                self.assertEqual(len(records(log, "LUT_GOT_RESULT")), 2)
                if mode == "lut_all_prepared":
                    contracts = records(log, "NATIVE_MATERIAL_CONSUMED")
                    self.assertEqual(len(contracts), 32)
                    self.assertEqual({row["messages_per_ot"] for row in contracts},
                                     {2, 4, 8, 16, 32, 64, 128, 256})
                    self.assertEqual({row["message_bits"] for row in contracts}, {1, 7, 8, 13, 64})
                    valid, _ = validate_native_materials(records(log, "MATERIAL_READY"),
                        records(log, "NATIVE_MATERIAL_PREPARED"), records(log, "NATIVE_MATERIAL_CONSUMED"),
                        records(log, "NATIVE_MATERIAL_BALANCE"), records(log, "METRIC"), 2)
                    self.assertTrue(valid)

    def test_batched_contract_and_repeat_consumption_fail_closed(self):
        for mode, expected in (("batched_wrong_contract", "material contract mismatch"),
                               ("batched_repeat", "material inventory exhausted")):
            exits, logs = self.run_parties(mode)
            self.assertEqual(exits, [2, 2], logs)
            self.assertTrue(all(expected in log for log in logs), logs)
if __name__ == "__main__": unittest.main()
