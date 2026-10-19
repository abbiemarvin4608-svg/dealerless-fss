"""Check execution-state reporting independently of cryptographic correctness."""

from pathlib import Path
import sys
import tempfile
import unittest

from run import execute


class RunnerStatusTests(unittest.TestCase):
    def run_process(self, source, timeout=5, memory_gib=None):
        with tempfile.TemporaryDirectory() as directory:
            return execute([[sys.executable, "-c", source]], Path(directory), timeout, memory_gib)

    def test_first_mismatch_overrides_zero_exit(self):
        report = self.run_process("print('MISMATCH input=253 actual=0 expected=126')")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["first_mismatch"], "MISMATCH input=253 actual=0 expected=126")

    def test_timeout_remains_timeout(self):
        report = self.run_process("import time; time.sleep(10)", timeout=0.05)
        self.assertEqual(report["status"], "timeout")
        self.assertIsNone(report["first_mismatch"])

    def test_actual_allocation_failure_is_oom(self):
        report = self.run_process("x = bytearray(128 * 1024 * 1024)", memory_gib=0.0625)
        self.assertEqual(report["status"], "oom")
        self.assertNotEqual(report["exit_codes"], [0])

    def test_missing_binary_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            report = execute([[str(Path(directory) / "missing")]], Path(directory), 5)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["exit_codes"], [])


if __name__ == "__main__":
    unittest.main()
