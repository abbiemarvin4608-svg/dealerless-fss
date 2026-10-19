"""Native DPF reservations must cover the actual material layout."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
OUR = HERE.parents[1]
UPSTREAM = OUR.parent / "dealerless-FSS_public" / "src"
SCI = Path(os.environ.get("SCI_ROOT", "/home/jiang/EzPC/SCI"))


class NativeDpfBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="native-dpf-budget-")
        cls.binary = Path(cls.directory.name) / "budget-test"
        command = ["g++", "-O1", "-std=c++17", "-pthread", "-maes", "-msse4.1", "-mavx2",
                   "-fopenmp", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                   "-Wno-deprecated-declarations", "-I" + str(HERE), "-I" + str(OUR / "src"),
                   "-I" + str(OUR / "src/dfss"), "-I" + str(UPSTREAM),
                   "-I" + str(UPSTREAM / "deps/cryptoTools"), "-I" + str(SCI / "src"),
                   "-I" + str(SCI / "extern/eigen"), str(HERE / "test_native_dpf_budget.cpp"),
                   str(HERE / "native_dpf_lut.cpp"), str(OUR / "src/dpf_lut.cpp"),
                   str(UPSTREAM / "deps/cryptoTools/cryptoTools/Common/Defines.cpp"),
                   "-o", str(cls.binary)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            cls.directory.cleanup()
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def check(self, kind):
        for mode in ("sigmoid", "tanh", "combined"):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.binary), mode, kind], text=True,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_planned_bytes_match_actual_materials(self):
        self.check("planned")

    def test_reserve_covers_batch128_materials(self):
        self.check("reserve")

    def test_batch128_prepare_remains_within_reserve(self):
        self.check("prepare")


if __name__ == "__main__":
    unittest.main()
