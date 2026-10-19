"""Evidence accounting regressions; no protocol is launched."""

import copy
import unittest

import analyze_rnn_widths as rnn
import prepare_primitive_ablation as primitive


class RnnEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = rnn.read_json(rnn.DEFAULT_SOURCE / "config.json")
        cls.parties = rnn.read_json(rnn.DEFAULT_SOURCE / "parties.json")

    def test_logical_instances_count_the_two_party_protocol_once(self):
        result = rnn.analyze(self.config, self.parties)
        totals = {row["comparison_bits"]: row["logical_comparisons_per_query"] for row in result["domains"]}
        self.assertEqual(totals, {23: 6528, 31: 51200})
        self.assertEqual(sum(totals.values()), 57728)
        self.assertEqual(len(result["operators"]), 19)

    def test_shrinking_ring_is_not_misclassified_as_wide_sign(self):
        result = rnn.analyze(self.config, self.parties)
        row = next(row for row in result["operators"] if row["operator"] == "MatMul_02" and row["slot"] == 1)
        self.assertEqual((row["input_bits"], row["output_bits"], row["shift"]), (36, 32, 4))
        self.assertEqual((row["low_comparison_bits"], row["sign_comparison_bits"]), (4, 0))
        self.assertFalse(row["contains_domain_gt20"])

    def test_partial_online_evidence_is_rejected(self):
        parties = copy.deepcopy(self.parties)
        parties[1]["conversion_details"].pop()
        with self.assertRaisesRegex(ValueError, "identities differ"):
            rnn.analyze(self.config, parties)

    def test_duplicate_detail_cannot_silently_overwrite_a_call(self):
        parties = copy.deepcopy(self.parties)
        parties[0]["conversion_details"].append(parties[0]["conversion_details"][0])
        with self.assertRaisesRegex(ValueError, "duplicate conversion"):
            rnn.analyze(self.config, parties)

    def test_work_fraction_is_not_divided_by_whole_stage_wall(self):
        parties = copy.deepcopy(self.parties)
        for party in parties:
            for phase in party["phases"]:
                if phase["phase"] == "online":
                    phase["seconds"] *= 100
        before = rnn.analyze(self.config, self.parties)
        after = rnn.analyze(self.config, parties)
        self.assertEqual([row["high_domain_conversion_work_share_percent"] for row in before["conversion_work"]],
                         [row["high_domain_conversion_work_share_percent"] for row in after["conversion_work"]])
        self.assertTrue(all(row["pure_high_domain_dcf_ms"] is None for row in after["conversion_work"]))


class PrimitiveEvidence(unittest.TestCase):
    def test_reuse_preserves_batch_and_internal_domain(self):
        rows, config = primitive.reused_rows(primitive.DEFAULT_SOURCE)
        self.assertEqual({row["batch"] for row in rows}, {100})
        self.assertEqual({row["comparison_bits"] for row in rows if row["input_bits"] in (24, 32)}, {23, 31})
        self.assertNotIn(28, {row["input_bits"] for row in rows})

    def test_all_five_repeats_count_but_warmup_does_not(self):
        common = dict(network={"name": "lan_model"}, backend="ours", input_bits=28, nseg=4)
        records = [dict(common, tag="warmup", status="complete", total_ms=10000)]
        records += [dict(common, tag="repeat%d" % index, status="complete", total_ms=index)
                    for index in range(1, 6)]
        summary = primitive.aggregate_missing(records, 5)[0]
        self.assertEqual(summary["completed"], 5)
        self.assertEqual(summary["total_ms"]["median"], 3)

    def test_resource_failure_does_not_create_timing_data(self):
        records = [dict(network={"name": "lan_model"}, backend="ours", input_bits=28,
                        nseg=1, tag="repeat%d" % index, status="not_run") for index in range(1, 6)]
        summary = primitive.aggregate_missing(records, 5)[0]
        self.assertEqual((summary["completed"], summary["unavailable"]), (0, 5))
        self.assertIsNone(summary["total_ms"])


if __name__ == "__main__":
    unittest.main()
