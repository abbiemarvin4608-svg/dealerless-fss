"""Public dependency schedules and phase-result contracts."""
import unittest
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from run import (attach_network_measurements, attach_round_measurements,
                 project_network_costs, project_network_costs_set)
from runtime import aggregate_phases, dependency_schedule, validate_fixture


class ScheduleTests(unittest.TestCase):
    def test_batch_rows_and_recurrent_dependencies_remain_explicit(self):
        schema = [
            {"operator": "MatMul_01", "function": "MatMul", "output": "tmp19bw16",
             "source_call": "MatMul(1, 32, 16, 1, 2, 5, 0, 1, 16, 8, 32, 16, tmp15bw16, W1, tmp19bw16, tmp18bw32);"},
            {"operator": "MatAdd_04", "function": "MatAdd", "output": "tmp11bw16",
             "source_call": "MatAdd(1, 100, 1, 2, 1, 1, 16, 16, 16, 16, tmp11bw16, tmp19bw16, tmp11bw16);"},
        ]
        schedule = dependency_schedule(schema, 128, 99, classifier=False)
        self.assertEqual(len(schedule), 198)
        self.assertEqual(schedule[0]["shape"], [128, 32, 16])
        self.assertEqual(schedule[0]["sample_groups"], 1)
        self.assertEqual(schedule[3]["dependencies"], [1, 2])
        self.assertEqual(schedule[-1]["step"], 98)
        self.assertEqual(schedule[-1]["output_elements"], 12800)


class MeasurementTests(unittest.TestCase):
    def test_reports_direct_query_and_first_query_totals(self):
        parties = []
        for party, query_total, first_total in [(1, 7, 10), (2, 8, 9)]:
            phases = [{"phase": phase, "query": query,
                       "seconds": seconds, "sent_bytes": 0, "received_bytes": 0,
                       "send_calls": 0, "recv_calls": 0, "logical_rounds": None}
                      for phase, query, seconds in
                      [("setup", -1, 2), ("teardown", -1, 1),
                       ("offline", 0, 3), ("input", 0, 0.1), ("ready", 0, 0.1),
                       ("online", 0, 4), ("output", 0, 0.1),
                       ("query_total", 0, query_total), ("first_query", -1, first_total)]]
            parties.append({"party": party, "status": "complete", "total_seconds": 11,
                            "peak_rss_kib": 100, "phases": phases})
        result = aggregate_phases(parties, 1)
        self.assertEqual(result["queries"][0]["query_total"]["seconds"], 8)
        self.assertEqual(result["first_query"]["seconds"], 10)
        self.assertEqual(result["query_total_seconds"],
                         {"median": 8, "min": 8, "max": 8, "pstdev": 0, "values": [8]})
        self.assertEqual(result["session_total_seconds"], 11)
        self.assertEqual(result["online_seconds"]["values"], [4])

    def test_uses_measured_total_and_sums_only_sends(self):
        parties = []
        for party, online, total, sent, received in [(1, 5, 12, 10, 20), (2, 4, 11, 20, 10)]:
            phases = [{"phase": phase, "query": 0 if phase in ("offline", "input", "ready", "online", "output") else -1,
                       "seconds": online if phase == "online" else 1,
                       "sent_bytes": sent, "received_bytes": received,
                       "send_calls": 2, "recv_calls": 3, "logical_rounds": None}
                      for phase in ("setup", "offline", "input", "ready", "online", "output", "teardown")]
            parties.append({"party": party, "status": "complete", "total_seconds": total,
                            "peak_rss_kib": 100, "phases": phases})
        result = aggregate_phases(parties, 1)
        self.assertEqual(result["total_seconds"], 12)
        self.assertEqual(result["queries"][0]["online"]["seconds"], 5)
        self.assertEqual(result["queries"][0]["online"]["sent_bytes"], 30)
        self.assertIsNone(result["queries"][0]["online"]["logical_rounds"])
        parties[1]["phases"] = [row for row in parties[1]["phases"] if row["phase"] != "offline"]
        with self.assertRaisesRegex(ValueError, "offline"):
            aggregate_phases(parties, 1)

    def test_rejects_graph_and_model_from_different_artifacts(self):
        contract = {"generated_sha256": "graph-a", "server_model_sha256": "weights-a", "client_input_sha256": "inputs-a"}
        with self.assertRaisesRegex(ValueError, "generated graph"):
            validate_fixture(contract, {"generated_sha256": "graph-b"},
                             {"server_model.i16": "weights-a", "client_input.i16": "inputs-a"})

    def test_network_projection_uses_send_sum_and_causal_rounds(self):
        phases = []
        for query in range(3):
            phases.extend([
                {"query": query, "phase": "offline", "party_sent_bytes": [100 + query, 200],
                 "logical_rounds": 2 + query},
                {"query": query, "phase": "online", "party_sent_bytes": [10, 20 + query],
                 "logical_rounds": 4},
            ])
        result = project_network_costs(
            {"status": "passed", "phases": phases}, (0.5, 1000), query_start=1, query_count=2)
        self.assertTrue(result["eligible"])
        self.assertEqual({"offline", "online"}, set(result["phases"]))
        self.assertEqual([row["query"] for row in result["phases"]["offline"]], [1, 2])
        self.assertEqual([row["query"] for row in result["queries"]], [1, 2])
        offline = result["queries"][0]["phases"]["offline"]
        self.assertEqual(offline["bytes"], 301)
        self.assertAlmostEqual(offline["transmission_seconds"], 8 * 301 / 1_000_000_000)
        self.assertAlmostEqual(offline["propagation_seconds"], 0.0015)
        self.assertAlmostEqual(offline["communication_seconds"],
                               8 * 301 / 1_000_000_000 + 0.0015)

    def test_network_projection_rejects_localhost_without_link_parameters(self):
        result = project_network_costs(
            {"status": "passed", "phases": []}, (None, None))
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "localhost has no declared bandwidth or RTT")

    def test_round_and_network_fields_are_attached_to_query_phases(self):
        rounds = {"status": "passed", "phases": [
            {"query": 0, "phase": "offline", "logical_rounds": 3,
             "send_flight_depths": [2, 3], "party_sent_bytes": [100, 200],
             "party_received_bytes": [200, 100]},
            {"query": 0, "phase": "online", "logical_rounds": 4,
             "send_flight_depths": [4, 4], "party_sent_bytes": [10, 20],
             "party_received_bytes": [20, 10]},
        ]}
        measurements = {"queries": [{
            "query": 0,
            "offline": {"per_party": [{}, {}]},
            "online": {"per_party": [{}, {}]},
        }]}
        attach_round_measurements(measurements, rounds)
        projection = project_network_costs_set(rounds, "both")
        attach_network_measurements(measurements, projection)
        offline = measurements["queries"][0]["offline"]
        self.assertEqual(offline["logical_rounds"], 3)
        self.assertEqual(offline["party_sent_bytes"], [100, 200])
        self.assertEqual(offline["per_party"][1]["round_send_flight_depth"], 3)
        self.assertEqual(set(offline["network_projection"]), {"LAN", "WAN"})
        self.assertAlmostEqual(offline["network_projection"]["LAN"]["propagation_seconds"], 0.0015)


@unittest.skipUnless(os.environ.get("SDCF_TEST_BINARY"), "set SDCF_TEST_BINARY for compiled runtime integration")
class RuntimeIntegrationTests(unittest.TestCase):
    def test_unavailable_configuration_keeps_measurements_null(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            command = [sys.executable, str(Path(__file__).with_name("run.py")),
                       "--binary", os.environ["SDCF_TEST_BINARY"], "--backend", "SCI",
                       "--fixture", "/tmp/ticket07-fixture-v2", "--output", str(output), "--steps", "99",
                       "--graph-openings", "batched"]
            child = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(child.returncode, 3, child.stderr)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["measurements"])


if __name__ == "__main__":
    unittest.main()
