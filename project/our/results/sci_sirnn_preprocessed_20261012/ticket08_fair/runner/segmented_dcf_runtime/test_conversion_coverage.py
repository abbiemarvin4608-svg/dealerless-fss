"""Conversion evidence contracts through the public two-party log collector."""
import copy
import json
import unittest

from run import collect, validate_graph_openings, validate_prefetch_budgets, normalize_beaver_offline, validate_beaver_traffic


class ConversionCoverageTests(unittest.TestCase):
    schedule = [dict(index=0, function="MatMul", output_elements=6),
                dict(index=1, function="MulCir", output_elements=6)]

    def evidence(self):
        calls = [dict(query=0, index=index, slot=0, count=count, input_bits=16,
                      output_bits=32, shift=0, call=3, numeric_mode="sci_compatible",
                      comparison_backend="segmented_dcf", sample_groups=1)
                 for index, count in ((0, 4), (1, 12))]
        group = dict(query=0, index=1, slot=0, input_groups=2, group_count=6,
                     mapping="lhs_then_rhs", contract="homogeneous_signed_extension")
        prep = [dict(ordinal=i, fresh=True, dealerless=True) for i in (0, 1)]
        return [[("CONVERSION_OFFLINE", copy.deepcopy(row)) for row in calls] +
                [("CONVERSION_ONLINE", copy.deepcopy(row)) for row in reversed(calls)] +
                [("CONVERSION_GROUP", copy.deepcopy(group))] +
                [("CONVERSION_PREPROCESS_DETAIL", copy.deepcopy(row)) for row in prep]
                for _ in (1, 2)]

    def logs(self, evidence):
        logs = []
        for role, rows in enumerate(evidence, 1):
            rows = list(rows)
            rows += [("METRIC", dict(phase=phase, query=query, seconds=1,
                                    sent_bytes=0, received_bytes=0))
                     for phase, query in (("setup", -1), ("teardown", -1),
                                          ("offline", 0), ("input", 0), ("ready", 0),
                                          ("online", 0), ("output", 0))]
            rows += [("COVERAGE", dict(query=0, index=layer["index"],
                                     elements=layer["output_elements"], sample_groups=1))
                     for layer in self.schedule]
            rows += [("FINISHED", dict(party=role, status="complete", total_seconds=9,
                                      peak_rss_kib=100))]
            logs.append("\n".join(prefix + " " + json.dumps(row) for prefix, row in rows))
        return logs

    def collect(self, evidence, backend="SCI_Beaver_segmented"):
        return collect(self.logs(evidence), 1, self.schedule, False, backend=backend)

    def test_preserves_offline_evidence_and_accepts_unordered_completion(self):
        evidence = self.evidence()
        evidence[0].append(("BEAVER_OFFLINE", dict(query=0, index=0, fresh=True,
                                                  kernel_layout_copy_seconds=0.0002)))
        evidence[1] = list(reversed(evidence[1]))
        parties, _ = self.collect(evidence)
        self.assertEqual(len(parties[0]["conversion_offline_calls"]), 2)
        self.assertEqual(len(parties[0]["conversion_preprocess_details"]), 2)
        self.assertEqual(parties[0]["beaver_offline_calls"][0]["kernel_layout_copy_seconds"], 0.0002)

    def test_rejects_missing_or_duplicate_consumption(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                evidence = self.evidence()
                online = next(row for row in evidence[0] if row[0] == "CONVERSION_ONLINE")
                if duplicate:
                    evidence[0].append(copy.deepcopy(online))
                else:
                    evidence[0].remove(online)
                with self.assertRaisesRegex(ValueError, "conversion"):
                    self.collect(evidence)

    def test_rejects_mismatched_numerical_or_group_contract(self):
        for prefix, field, value in (("CONVERSION_ONLINE", "output_bits", 31),
                                     ("CONVERSION_GROUP", "group_count", 5),
                                     ("CONVERSION_GROUP", "contract", "heterogeneous_extension")):
            with self.subTest(prefix=prefix, field=field):
                evidence = self.evidence()
                next(row for kind, row in evidence[0] if kind == prefix)[field] = value
                with self.assertRaisesRegex(ValueError, "conversion"):
                    self.collect(evidence)

    def test_rejects_missing_or_duplicate_group_and_incomplete_fresh_preprocessing(self):
        for prefix, duplicate in (("CONVERSION_GROUP", False), ("CONVERSION_GROUP", True),
                                  ("CONVERSION_PREPROCESS_DETAIL", False)):
            with self.subTest(prefix=prefix, duplicate=duplicate):
                evidence = self.evidence()
                record = next(row for row in evidence[0] if row[0] == prefix)
                if duplicate:
                    evidence[0].append(copy.deepcopy(record))
                else:
                    evidence[0].remove(record)
                with self.assertRaisesRegex(ValueError, "conversion"):
                    self.collect(evidence)

    def test_rejects_contracts_which_differ_between_parties(self):
        evidence = self.evidence()
        for kind, row in evidence[1]:
            if kind in ("CONVERSION_OFFLINE", "CONVERSION_ONLINE") and row["index"] == 0:
                row["count"] = 5
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.collect(evidence)

    def test_rejects_missing_enabled_conversion_or_unscheduled_identity(self):
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.collect([[], []])
        evidence = self.evidence()
        for kind, row in evidence[0]:
            if kind in ("CONVERSION_OFFLINE", "CONVERSION_ONLINE") and row["index"] == 0:
                row["index"] = 99
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.collect(evidence)

    def test_rejects_coordinated_omission_from_the_public_call_plan(self):
        self.schedule = [dict(index=0, function="MatMul", output_elements=6, shape=[2, 1, 3],
                              source_call="MatMul(1, 1, 3, 1, 1, 0, 0, 1, 16, 16, 32, 16, A, B, C, tmp);"),
                         dict(index=1, function="MulCir", output_elements=6, shape=[2, 3],
                              source_call="MulCir(1, 3, 1, 1, 1, 16, 16, 32, 16, A, B, C);")]
        evidence = self.evidence()
        for party in evidence:
            for kind, row in party:
                if kind in ("CONVERSION_OFFLINE", "CONVERSION_ONLINE") and row["index"] == 0:
                    row["count"] = 2
        self.collect(evidence)
        omitted = [[(kind, row) for kind, row in party
                    if row.get("index") != 0 and not (kind == "CONVERSION_PREPROCESS_DETAIL" and row["ordinal"] == 0)]
                   for party in evidence]
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.collect(omitted)

    def test_allows_archived_rows_without_preprocessing_details_and_native_rows(self):
        evidence = [[row for row in party if row[0] != "CONVERSION_PREPROCESS_DETAIL"]
                    for party in self.evidence()]
        self.collect(evidence)
        self.collect([[], []], backend="SCI")

    def test_batched_matadd_mapping_requires_both_layer_identities(self):
        schedule = [dict(index=i, function="MatAdd", output_elements=12800) for i in (5, 7)]
        row = dict(query=0, index=5, indices=[5, 7], count_per_group=12800, input_groups=2,
                   mapping="gate_bias_then_candidate_bias", contract="homogeneous_signed_truncation")
        validate_graph_openings([row], schedule, 1, True)
        for field, value in (("indices", [5, 5]), ("count_per_group", 25600), ("query", 1)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "graph opening"):
                validate_graph_openings([dict(row, **{field: value})], schedule, 1, True)
        with self.assertRaisesRegex(ValueError, "graph opening"):
            validate_graph_openings([], schedule, 1, True)

    def test_prefetch_budget_is_shared_by_contexts_and_resident_bytes_stay_bounded(self):
        budgets = [dict(context=i, contexts=2, context_budget_bytes=5 if i == 0 else 4,
                        party_budget_bytes=9) for i in (0, 1)]
        prefixes = [dict(budget_bytes=5, resident_bytes=3), dict(budget_bytes=4, resident_bytes=4)]
        validate_prefetch_budgets(budgets, prefixes, 2, 1)
        with self.assertRaisesRegex(ValueError, "prefetch"):
            validate_prefetch_budgets([dict(row, context_budget_bytes=9) for row in budgets], prefixes, 2, 1)
        with self.assertRaisesRegex(ValueError, "prefetch"):
            validate_prefetch_budgets(budgets, [dict(budget_bytes=4, resident_bytes=5)], 2, 1)

    def test_beaver_owner_traffic_matches_remote_sends_and_rejects_inferred_zero(self):
        def row(sent, received):
            return dict(query=0, index=0, sent_bytes=sent, received_bytes=received,
                        receive_instrumented=True, tiles=4, cross_term_batches=8)
        parties = [dict(beaver_offline_calls=[row(70, 90)]), dict(beaver_offline_calls=[row(90, 70)])]
        validate_beaver_traffic(parties)
        broken = copy.deepcopy(parties)
        broken[0]["beaver_offline_calls"][0]["received_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "Beaver per-owner"):
            validate_beaver_traffic(broken)
        archived = dict(query=0, index=0, sent_bytes=70, received_bytes=0)
        normalized = normalize_beaver_offline([archived])[0]
        self.assertIsNone(normalized["received_bytes"])
        self.assertFalse(normalized["receive_instrumented"])
        self.assertEqual(archived["received_bytes"], 0)
        validate_beaver_traffic([dict(beaver_offline_calls=[normalized])] * 2)


if __name__ == "__main__":
    unittest.main()
