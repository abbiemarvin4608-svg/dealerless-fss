import unittest
from comparison import compare_predictions, summarize


def record(backend="SCI", status="complete"):
    config = {"backend": backend, "batch": 128, "steps": 99, "start": 0,
              "network": {"name": "localhost"}, "diagnostic": False,
              "cpus": [10, 12, 14], "queries": 3, "warmup_queries": 0,
              "max_segments": 4, "application_threads_per_party": 1,
              "fixture_files": {"server_model.i16": "model", "client_input.i16": "input"},
              "sample_ids": ["sample"], "binary_sha256": "binary",
              "runtime_manifest": {"generated_sha256": "graph"}}
    phase = lambda seconds: {"seconds": seconds, "sent_bytes": 100,
                             "send_calls": None, "recv_calls": None, "logical_rounds": None}
    return {"config": config, "status": status, "reason": None,
            "measurements": None if status != "complete" else
            {"setup": phase(0.5), "first_query": phase(7), "total_seconds": 18,
             "peak_rss_kib_per_party": [1024, 2048],
             "queries": [{"offline": phase(offline), "online": phase(online),
                          "output": phase(0.01), "input": phase(0.02), "ready": phase(0.03),
                          "query_total": phase(total)}
                         for offline, online, total in [(1, 4, 5), (2, 3, 6), (3, 5, 7)]]}}


class ComparisonTests(unittest.TestCase):
    def test_fixed_batch_output_check_locates_first_different_prediction(self):
        expected = "PRED 0 10 7\nPRED 0 11 3\nPRED 1 10 7\nPRED 1 11 3\n"
        actual = "PRED 0 10 7\nPRED 0 11 8\nPRED 1 10 7\nPRED 1 11 3\n"
        result = compare_predictions(expected, actual, batch=2, start=10, queries=2)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["matched"], 3)
        self.assertEqual(result["first_difference"], {"query": 0, "sample": 11, "SCI": 3, "candidate": 8})
        result = compare_predictions(expected, expected, batch=2, start=10, queries=2)
        self.assertEqual(result["status"], "passed")
        with self.assertRaisesRegex(ValueError, "prediction coverage"):
            compare_predictions(expected, "PRED 0 10 7\n", batch=2, start=10, queries=2)

    def test_uses_measured_session_total_and_keeps_unfinished_costs_null(self):
        rows = summarize([("SCI/result.json", record()),
                          ("full/result.json", record("SCI_Beaver_full", "oom"))])
        self.assertEqual(rows[0]["offline_seconds"]["median"], 2)
        self.assertEqual(rows[0]["online_seconds"]["median"], 4)
        self.assertEqual(rows[0]["session_total_seconds"], 18)
        self.assertEqual(rows[0]["session_seconds_per_query"], 6)
        self.assertEqual(rows[0]["first_query_seconds"], 7)
        self.assertIsNone(rows[0]["online_io"]["send_calls"]["median"])
        self.assertIsNone(rows[1]["online_seconds"])
        self.assertIsNone(rows[1]["session_total_seconds"])
        self.assertEqual(rows[1]["status"], "oom")

    def test_rejects_comparing_different_private_artifacts(self):
        changed = record("SCI_Beaver")
        changed["config"]["fixture_files"]["server_model.i16"] = "other-model"
        with self.assertRaisesRegex(ValueError, "different comparison contract"):
            summarize([("SCI/result.json", record()), ("Beaver/result.json", changed)])

    def test_rejects_diagnostic_and_duplicate_backend_rows(self):
        changed = record()
        changed["config"]["diagnostic"] = True
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            summarize([("diagnostic/result.json", changed)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([("a/result.json", record()), ("b/result.json", record())])

    def test_legacy_single_worker_contract_matches_explicit_defaults(self):
        explicit = record("SCI_Beaver")
        explicit["config"].update(cggm_worker_limit_per_party=1,
                                  sci_protocol_threads_per_party=1,
                                  cpu_affinity_per_party=[[10], [12]])
        self.assertEqual(len(summarize([("old", record()), ("new", explicit)])), 2)

    def test_rejects_different_worker_limits_and_party_affinities(self):
        for update in ({"cggm_worker_limit_per_party": 4},
                       {"sci_protocol_threads_per_party": 2},
                       {"cpu_affinity_per_party": [[10, 11], [12, 13]]}):
            changed = record("SCI_Beaver")
            changed["config"].update(update)
            with self.assertRaisesRegex(ValueError, "different comparison contract"):
                summarize([("old", record()), ("changed", changed)])


if __name__ == "__main__":
    unittest.main()
