import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import round_trace_stream as streaming
from round_trace_stream import analyze_pair
from round_trace import analyze_pair as original_analyze
from test_round_trace import Trace


class StreamingRoundTraceTest(unittest.TestCase):
    def pair(self):
        traces = Trace(1), Trace(2)
        for trace in traces:
            trace.begin()
        return traces

    def replay(self, traces, compressed=False, **options):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for party, trace in enumerate(traces, 1):
                path = Path(directory) / ("party%d.jsonl%s" % (party, ".gz" if compressed else ""))
                opener = gzip.open if compressed else open
                with opener(path, "wt") as stream:
                    for row in trace.events:
                        stream.write(json.dumps(row) + "\n")
                paths.append(path)
            actual = analyze_pair(*paths, **options)
            expected = original_analyze(*paths, expected_phases=options.get("expected_phases"))
        self.assertEqual(actual["status"], expected["status"], (actual, expected))
        if actual["status"] == "passed":
            self.assertEqual(actual["events"], expected["events"])
            self.assertEqual(actual["phases"], expected["phases"])
        return actual, expected

    def test_independent_actors_can_resolve_receives_before_later_local_sends(self):
        left, right = self.pair()
        for trace, incoming, outgoing in ((left, 42000, 42001), (right, 42001, 42000)):
            fork = trace.event("fork")
            trace.event("start", actor=2, dependency=fork)
            trace.io("recv", actor=2, channel=incoming)
            end = trace.event("end", actor=2)
            trace.io("send", channel=outgoing)
            trace.event("join", dependency=end)
            trace.end()
        actual, _ = self.replay((left, right))
        self.assertEqual(actual["status"], "passed", actual)
        self.assertEqual(actual["phases"][0]["send_flight_depths"], [1, 1])

    def test_chunk_boundaries_and_multiple_phases_in_gzip(self):
        left, right = self.pair()
        for length in (3, 5, 7):
            left.io("send", length)
        right.io("recv", 9)
        right.io("recv", 6)
        right.io("send", 15)
        left.io("recv", 15)
        for trace in (left, right):
            trace.event("phase_end")
            trace.event("phase_begin", query=1)
            trace.io("send")
            trace.events[-1]["query"] = 1
            trace.io("recv")
            trace.events[-1]["query"] = 1
            trace.event("phase_end", query=1)
            trace.events.append(dict(type="complete", events=trace.next_id))
        actual, _ = self.replay((left, right), compressed=True)
        self.assertEqual(actual["status"], "passed", actual)
        self.assertEqual([row["logical_rounds"] for row in actual["phases"]], [1, 1])

    def test_scope_identity_is_checked_without_individual_scope_replays(self):
        traces = self.pair()
        for trace in traces:
            trace.event("scope_begin", scope=1, parent_scope=0, semantic=dict(
                label="activation", query=0, layer=6, slot=-1, call=-1, kind=0, stage=-1))
            trace.io("send")
            trace.events[-1]["scope"] = 1
            trace.io("recv")
            trace.events[-1]["scope"] = 1
            trace.event("scope_end", scope=1)
            trace.end()
        actual, _ = self.replay(traces)
        self.assertEqual(actual["scopes"], [])
        self.assertIn("scope_rounds_omitted", actual)
        actual, expected = self.replay(traces, include_scopes=True)
        self.assertEqual(actual["scopes"], expected["scopes"])
        traces[1].events[2]["semantic"]["layer"] = 7
        actual, _ = self.replay(traces)
        self.assertEqual(actual["status"], "failed")

    def test_incomplete_footer_offset_and_metric_corruption_are_rejected(self):
        original = self.pair()
        for trace in original:
            trace.io("send")
            trace.io("recv")
            trace.end()
        for corruption in ("footer", "count", "offset", "after_footer", "actor", "phase"):
            with self.subTest(corruption=corruption):
                traces = copy.deepcopy(original)
                rows = traces[0].events
                if corruption == "footer":
                    rows.pop()
                elif corruption == "count":
                    rows[-1]["events"] += 1
                elif corruption == "offset":
                    rows[2]["offset"] += 1
                elif corruption == "after_footer":
                    rows.append(dict(type="complete", events=traces[0].next_id))
                elif corruption == "actor":
                    rows[2]["actor"] = 99
                else:
                    rows[2]["phase"] = "input"
                actual, _ = self.replay(traces)
                self.assertEqual(actual["status"], "failed")
                self.assertIsNone(actual["logical_rounds"])
        metric = [dict(query=0, phase="online", sent_bytes=9, received_bytes=8)]
        actual, _ = self.replay(original, expected_phases=[metric, metric])
        self.assertEqual(actual["status"], "failed")

    def test_communication_cycle_is_rejected(self):
        traces = self.pair()
        for trace in traces:
            trace.io("recv")
            trace.io("send")
            trace.end()
        actual, _ = self.replay(traces)
        self.assertEqual(actual["status"], "failed")
        self.assertIn("cycle", actual["reason"])

    def test_actor_metadata_survives_cache_eviction(self):
        traces = self.pair()
        for trace in traces:
            for actor in range(2, 2100):
                fork = trace.event("fork")
                trace.event("start", actor=actor, dependency=fork)
                trace.io("send", actor=actor)
                trace.io("recv", actor=actor)
                end = trace.event("end", actor=actor)
                trace.event("join", dependency=end)
            trace.end()
        actual, _ = self.replay(traces)
        self.assertEqual(actual["status"], "passed", actual)
        self.assertEqual(actual["phases"][0]["logical_rounds"], 2098)

    def test_qualified_dispatch_uses_immutable_digest_checked_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.write_bytes(b"qualified-analyzer")
            with mock.patch.object(streaming, "__file__", str(root / "round_trace_stream.py")), \
                    mock.patch.dict(os.environ, SEGFSS_ROUNDS_NATIVE_BINARY="", SEGFSS_ROUNDS_REBUILD=""):
                identity = streaming.qualify_native(candidate, {"status": "passed"})
                installed, note = streaming._native_binary()
                self.assertIsNone(note)
                self.assertEqual(streaming._binary_digest(installed), identity["binary_sha256"])
                candidate.write_bytes(b"next-unqualified-candidate")
                self.assertEqual(installed.read_bytes(), b"qualified-analyzer")
                installed.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    streaming._native_binary()

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_single_large_phase_under_96_mib_address_space_limit(self):
        iterations = 40000
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for party in (1, 2):
                path = Path(directory) / ("party%d.jsonl" % party)
                paths.append(path)
                with path.open("w") as stream:
                    stream.write(json.dumps(dict(type="meta", version=1, party=party)) + "\n")
                    identifier = 0

                    def emit(kind, **fields):
                        nonlocal identifier
                        identifier += 1
                        row = dict(type=kind, id=identifier, actor=1, prev=identifier - 1,
                                   query=0, phase="online")
                        row.update(fields)
                        stream.write(json.dumps(row) + "\n")

                    emit("phase_begin")
                    # The first receive must traverse a long remote actor
                    # chain, forcing the DFS stack and record cache to spill.
                    if party == 2:
                        for _ in range(100000):
                            emit("fork")
                    for offset in range(iterations):
                        emit("send", channel=42000, offset=offset, length=1)
                        emit("recv", channel=42000, offset=offset, length=1)
                    emit("phase_end")
                    stream.write(json.dumps(dict(type="complete", events=identifier)) + "\n")
            script = """
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, (96 * 1024 ** 2, 96 * 1024 ** 2))
sys.path.insert(0, sys.argv[1])
from round_trace_stream import analyze_pair
result = analyze_pair(sys.argv[2], sys.argv[3], use_native=False)
assert result['status'] == 'passed', result
assert result['phases'][0]['send_flight_depths'] == [40000, 40000], result
"""
            result = subprocess.run([sys.executable, "-B", "-c", script, str(Path(__file__).resolve().parent),
                                     *map(str, paths)], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
