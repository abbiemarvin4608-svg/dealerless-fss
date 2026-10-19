import json
import os
import signal
import socket
import tempfile
import unittest
from pathlib import Path

from runner import aggregate_parties, classify_status, command_for, run_pair
from manifest import build_manifest


class RunnerTests(unittest.TestCase):
    def test_command_contains_frozen_budget_and_three_runs(self):
        row = build_manifest()[0]
        command = command_for(Path("/bin/echo"), row, 32000, 2)
        self.assertIn("runs=3", command)
        self.assertIn("timeout=30", command)

    def test_status_classifier_preserves_failure_kinds(self):
        self.assertEqual(classify_status(0, "RESULT status=complete", ""), "complete")
        self.assertEqual(classify_status(3, "", "UNSUPPORTED unsupported"), "unsupported")
        self.assertEqual(classify_status(-signal.SIGSEGV, "", ""), "failed")
        self.assertEqual(classify_status(137, "", "OOM out of memory"), "oom")
        self.assertEqual(classify_status(2, "", "error"), "failed")

    def test_aggregation_uses_both_senders_and_aligned_run_maxima(self):
        parties = [
            {"samples": [{"offline_seconds": 2, "online_seconds": 1,
                           "total_seconds": 3, "offline_sent_bytes": 10,
                           "online_sent_bytes": 5}], "setup_seconds": 1},
            {"samples": [{"offline_seconds": 3, "online_seconds": 2,
                           "total_seconds": 5, "offline_sent_bytes": 11,
                           "online_sent_bytes": 6}], "setup_seconds": 2},
        ]
        result = aggregate_parties(parties)
        self.assertEqual(result["offline_seconds"], 3)
        self.assertEqual(result["online_seconds"], 2)
        self.assertEqual(result["total_seconds"], 5)
        self.assertEqual(result["offline_bytes"], 21)
        self.assertEqual(result["online_bytes"], 11)
        self.assertEqual(result["setup_seconds"], 2)

    def test_peer_failure_is_preserved_without_waiting_for_wall_timeout(self):
        row = build_manifest()[0]
        with tempfile.TemporaryDirectory() as directory:
            result = run_pair(row, Path(directory), Path("/bin/sh"), timeout=2,
                              executable_args=["-c", "echo OOM >&2; exit 137"])
            self.assertEqual(result["status"], "oom")
            self.assertIsNone(result["measurements"])

    def test_bind_failure_preserves_diagnostic_and_null_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_pair(build_manifest()[0], Path(directory), Path("/bin/sh"), timeout=2,
                              executable_args=["-c", "echo 'error: bind: Address already in use' >&2; exit 1"])
            self.assertEqual(result["status"], "failed")
            self.assertIn("Address already in use", result["reason"])
            self.assertIsNone(result["measurements"])

    def test_pair_reserves_auxiliary_ports_below_ephemeral_range(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as directory:
            for requested_port in range(22000, 22100):
                try:
                    listener.bind(("0.0.0.0", requested_port + 100))
                    break
                except OSError:
                    continue
            else:
                self.skipTest("no free low auxiliary port for collision check")
            run_pair(build_manifest()[0], Path(directory), Path("/bin/echo"), timeout=2,
                     port=requested_port)
            execution = json.loads((Path(directory) / "execution.json").read_text())
            selected = execution["selected_port"]
            self.assertNotEqual(selected, requested_port)
            self.assertLess(selected + 100, int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0]))
            self.assertTrue(all("port=" + str(selected) in command for command in execution["commands"]))

    def test_pair_runner_keeps_null_measurements_on_timeout(self):
        row = build_manifest()[0]
        with tempfile.TemporaryDirectory() as directory:
            result = run_pair(row, Path(directory), Path("/bin/sh"), timeout=0.05,
                              executable_args=["-c", "sleep 1"])
            self.assertEqual(result["status"], "timeout")
            self.assertIsNone(result["measurements"])
            self.assertTrue((Path(directory) / "party1.stdout.log").exists())
            self.assertTrue((Path(directory) / "party2.stderr.log").exists())
            json.loads((Path(directory) / "execution.json").read_text())


@unittest.skipUnless(os.environ.get("SDCF_SWEEP_TEST_BINARY"),
                     "set SDCF_SWEEP_TEST_BINARY to run the real two-party verifier regression")
class RealOutputVerificationTests(unittest.TestCase):
    def test_parallel_fresh_queries_preserve_outputs_and_party_affinities(self):
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 8:
            self.skipTest("eight available CPUs are required for disjoint party affinities")
        affinities = [available[:4], available[4:8]]
        executable = Path(os.environ["SDCF_SWEEP_TEST_BINARY"]).resolve()
        row = dict(build_manifest()[0], id="parallel-affinity-verification",
                   bits=16, comparison_bits=16, backend="segmented", nseg=4,
                   count=128, runs=3, cggm_threads=4)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("SDCF_", "SEGFSS_"))}
        env["OMP_NUM_THREADS"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            result = run_pair(row, Path(directory), executable, timeout=30,
                              port=port, party_cpus=affinities, env=env)
            self.assertEqual(result["status"], "complete", result.get("reason"))
            execution = json.loads((Path(directory) / "execution.json").read_text())
            self.assertEqual(execution["observed_cpu_affinity_per_party"], affinities)
            for party_result in result["measurements"]["parties"]:
                self.assertTrue(party_result["correctness_passed"])
                self.assertEqual(len(party_result["samples"]), 3)
                for sample in party_result["samples"]:
                    self.assertGreaterEqual(sample["eval_participants"], 1)
                    self.assertLessEqual(sample["eval_participants"], 4)
                    self.assertGreaterEqual(sample["query_prepare_seconds"], 0)
                    self.assertGreaterEqual(sample["local_evaluation_seconds"], 0)

    def test_dcf_and_conversion_outputs_are_verified(self):
        executable = Path(os.environ["SDCF_SWEEP_TEST_BINARY"]).resolve()
        dcf = dict(build_manifest()[0], id="verification-dcf", bits=4,
                   comparison_bits=4, backend="segmented", nseg=2, count=4,
                   runs=1, cggm_threads=1)
        conversion = dict(dcf, id="verification-conversion", kind="conversion",
                          family="C", bits=8, comparison_bits=3, shift=3,
                          operation="div_pow2", signed=False,
                          numeric_mode="sci_compatible")
        # DCF row zero has query == threshold; conversion row zero has
        # input 127 and expected output 15. An exchange that zeros the local
        # shares instead of contacting the peer fails both verifiers.
        for row in (dcf, conversion):
            with self.subTest(kind=row["kind"]), tempfile.TemporaryDirectory() as directory:
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                result = run_pair(row, Path(directory), executable, timeout=30, port=port)
                self.assertEqual(result["status"], "complete", result.get("reason"))
                self.assertIsNotNone(result["measurements"])
                parties = result["measurements"]["parties"]
                self.assertEqual(len(parties), 2)
                for party_result in parties:
                    self.assertTrue(party_result["correctness_passed"])
                    self.assertEqual(len(party_result["samples"]), 1)


if __name__ == "__main__":
    unittest.main()
