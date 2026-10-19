"""Malformed owned schedules must not qualify as matched parallel evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_folded_parallel import audit

EVIDENCE = Path(__file__).resolve().parents[2] / "results/paper_folded_activation_20261013/ticket08/batch128-owner"


class FoldedParallelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="folded-parallel-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        original = EVIDENCE / "parallel-a2-full-default"
        self.result = json.loads((original / "result.json").read_text())
        self.parties = json.loads((original / "parties.json").read_text())
        self.logs = [(original / ("party%d.log" % role)).read_text() for role in (1, 2)]

    def check(self):
        (self.root / "result.json").write_text(json.dumps(self.result))
        (self.root / "parties.json").write_text(json.dumps(self.parties))
        for role, log in enumerate(self.logs, 1):
            (self.root / ("party%d.log" % role)).write_text(log)
        return audit(self.root)

    def test_actual_three_schedules_two_families_and_both_batches_qualify(self):
        for batch in (1, 128):
            directory = EVIDENCE.parent / ("batch%d-owner" % batch)
            qualification = json.loads((directory / "qualification.json").read_text())
            self.assertEqual(qualification["status"], "passed")
            self.assertEqual(len(qualification["rows"]), 12)
            for row in qualification["rows"]:
                self.assertEqual(audit(directory / row["id"])["status"], "passed", row["id"])

    def test_missing_owned_prefix_and_wrong_joint_budget_are_rejected(self):
        original = self.logs[0]
        lines = original.splitlines()
        index = next(index for index, line in enumerate(lines) if line.startswith("CONVERSION_PREFETCH "))
        self.logs[0] = "\n".join(lines[:index] + lines[index + 1:]) + "\n"
        self.assertFalse(self.check()["parties"][0]["checks"]["complete_outer_prefixes"])
        self.logs[0] = original
        self.parties[0]["conversion_prefetch_budgets"][0]["context_budget_bytes"] += 1
        self.assertFalse(self.check()["parties"][0]["checks"]["joint_budget_accounted"])

    def test_observed_transport_oversubscription_is_rejected(self):
        self.parties[0]["activation_workers"][0]["observed_transport_participants"] = 2
        report = self.check()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["parties"][0]["checks"]["compute_budget_bounded"])

    def test_selected_sign_backend_requires_its_own_reservation(self):
        self.result["config"]["folded_sign_backend"] = "segmented_fss"
        self.result["config"]["folded_sign_segments"] = 4
        report = self.check()
        self.assertTrue(report["checks"]["explicit_schedule"])
        self.assertFalse(report["parties"][0]["checks"]["joint_budget_accounted"])
        self.result["config"]["folded_sign_segments"] = 2
        self.assertFalse(self.check()["checks"]["explicit_schedule"])

    def test_missing_whole_activation_boundary_is_rejected(self):
        rounds = self.result["round_measurements"]
        rounds["scopes"] = [row for row in rounds["scopes"] if not (
            row["label"] == "activation" and row["phase"] == "online" and row["query"] == 0 and row["kind"] == 0)]
        report = self.check()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["scope_traffic_qualified"])


if __name__ == "__main__":
    unittest.main()
