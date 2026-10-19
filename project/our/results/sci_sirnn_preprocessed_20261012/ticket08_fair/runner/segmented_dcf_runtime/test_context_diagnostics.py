"""Frozen final diagnostic commands at the shared runner boundary."""
from pathlib import Path
import unittest

import verify_preprocessed_contexts as verification


class ContextDiagnosticPlanTests(unittest.TestCase):
    def test_all_sixteen_rows_share_resources_and_one_binary(self):
        rows = verification.diagnostic_rows(Path("/tmp/frozen/sdcf-runtime"),
            Path("/tmp/fixture"), Path("/tmp/hidden.i16"))
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({row["id"] for row in rows}), 16)
        for row in rows:
            command = row["command"]
            for flag, value in (("--binary", "/tmp/frozen/sdcf-runtime"),
                                ("--queries", "2"), ("--batch", "128"), ("--steps", "1"),
                                ("--memory-gib", "1.5"), ("--server-cpus", "0,2,4,10"),
                                ("--client-cpus", "6,8,12,16")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--diagnostic", command)
            self.assertEqual("--native-preprocessing" in command,
                             row["backend"] != "SCI")
            self.assertEqual("--hidden" in command, row["hidden"] == "nonzero")
            self.assertEqual(command[command.index("--activation-workers") + 1],
                             "4" if row["cohort"] == "a4-sequential" else "2")


if __name__ == "__main__":
    unittest.main()
