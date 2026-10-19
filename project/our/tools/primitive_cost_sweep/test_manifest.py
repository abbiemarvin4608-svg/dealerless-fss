import unittest

from manifest import build_manifest, validate_manifest


class ManifestTests(unittest.TestCase):
    def test_freezes_71_rows_and_required_families(self):
        rows = build_manifest()
        self.assertEqual(len(rows), 71)
        self.assertEqual({row["family"] for row in rows}, {"A", "B", "C", "D"})
        self.assertEqual({row["family"]: sum(r["family"] == row["family"] for r in rows)
                          for row in rows}, {"A": 31, "B": 25, "C": 12, "D": 3})
        validate_manifest(rows)

    def test_primary_and_native_domains_are_explicit(self):
        rows = build_manifest()
        primary = [row for row in rows if row["family"] == "A"]
        self.assertEqual({row["comparison_bits"] for row in primary}, {16, 20, 24, 32})
        self.assertTrue(all(row["comparison_bits"] == row["bits"] for row in primary))
        native = [row for row in rows if row["family"] == "B"]
        self.assertEqual({row["comparison_bits"] for row in native},
                         {1, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 23, 31})
        self.assertTrue(all(row["nseg"] in {1, min(4, row["comparison_bits"])} for row in native))
        self.assertTrue(all(row["nseg"] <= row["comparison_bits"] for row in rows if row["kind"] == "dcf"))

    def test_failure_rows_keep_null_measurements(self):
        row = build_manifest()[0]
        row.update(status="timeout", measurements=None, reason="wall timeout")
        validate_manifest([row])
        row["status"] = "complete"
        with self.assertRaisesRegex(ValueError, "measurements"):
            validate_manifest([row])


if __name__ == "__main__":
    unittest.main()
