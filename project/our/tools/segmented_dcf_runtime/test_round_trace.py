import json
import gzip
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_trace import analyze_pair


class Trace:
    def __init__(self, party):
        self.party = party
        self.events = [{"type": "meta", "version": 1, "party": party}]
        self.previous = {}
        self.offsets = {}
        self.next_id = 0

    def event(self, kind, actor=1, **fields):
        self.next_id += 1
        record = dict(type=kind, id=self.next_id, actor=actor,
                      prev=self.previous.get(actor, 0), phase="online", query=0)
        record.update(fields)
        self.events.append(record)
        self.previous[actor] = self.next_id
        return self.next_id

    def io(self, direction, length=8, actor=1, channel=42000):
        key = direction, channel
        offset = self.offsets.get(key, 0)
        self.offsets[key] = offset + length
        return self.event(direction, actor, channel=channel, offset=offset, length=length)

    def begin(self):
        self.event("phase_begin")

    def end(self):
        self.event("phase_end")
        self.events.append({"type": "complete", "events": self.next_id})


class RoundTraceTest(unittest.TestCase):
    def full_fixture_scope_depth(self, traces, scope):
        # This fixture alternates each selected send/receive, so each selected
        # send is a flight. An independent full-DAG longest path is its oracle.
        nodes = {(party, row["id"]): row for party, trace in enumerate(traces)
                 for row in trace.events if "id" in row}
        sends = {(party, row["channel"], row["offset"], row["length"]): key
                 for key, row in nodes.items() for party in (key[0],) if row["type"] == "send"}
        memo = {}

        def depth(key):
            if key in memo:
                return memo[key]
            row, party = nodes[key], key[0]
            dependencies = [(party, row[field]) for field in ("prev", "dependency") if row.get(field)]
            if row["type"] == "recv":
                dependencies.append(sends[1 - party, row["channel"], row["offset"], row["length"]])
            values = [depth(dependency) for dependency in dependencies]
            value = [max((item[index] for item in values), default=0) for index in (0, 1)]
            if row["type"] == "send" and row.get("scope", 0) == scope:
                value[party] += 1
            memo[key] = value
            return value

        return max(max(depth(key)) for key in nodes)

    def analyze(self, left, right, expected=None, scope_groups=None):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, trace in enumerate((left, right)):
                path = Path(directory) / str(index)
                path.write_text("".join(json.dumps(event) + "\n" for event in trace.events))
                paths.append(path)
            return analyze_pair(*paths, expected_phases=expected, scope_groups=scope_groups)

    def pair(self):
        traces = Trace(1), Trace(2)
        for trace in traces:
            trace.begin()
        return traces

    def finish(self, left, right):
        left.end()
        right.end()
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "passed", result)
        return result["phases"][0]["logical_rounds"]

    def test_symmetric_opening_is_one_round(self):
        left, right = self.pair()
        for trace in (left, right):
            trace.io("send")
            trace.io("recv")
        self.assertEqual(self.finish(left, right), 1)

    def test_request_response_and_chunks_are_one_round(self):
        left, right = self.pair()
        left.io("send", 3)
        left.io("send", 5)
        right.io("recv", 8)
        right.io("send", 8)
        left.io("recv", 3)
        left.io("recv", 5)
        self.assertEqual(self.finish(left, right), 1)

    def test_cross_channel_dependencies_are_counted(self):
        left, right = self.pair()
        for channel in (42000, 42001):
            left.io("send", channel=channel)
            right.io("recv", channel=channel)
            right.io("send", channel=channel)
            left.io("recv", channel=channel)
        self.assertEqual(self.finish(left, right), 2)

    def test_parallel_branches_join_use_maximum(self):
        left, right = self.pair()
        for trace in (left, right):
            fork = trace.event("fork")
            for actor, count in ((2, 2), (3, 3)):
                trace.event("start", actor, dependency=fork)
                for _ in range(count):
                    trace.io("send", actor=actor, channel=42000 + actor)
                    trace.io("recv", actor=actor, channel=42000 + actor)
                end = trace.event("end", actor)
                trace.event("join", dependency=end)
            trace.io("send", channel=42009)
            trace.io("recv", channel=42009)
        self.assertEqual(self.finish(left, right), 4)

    def test_activation_scope_inherits_children_and_preserves_external_dependencies(self):
        left, right = self.pair()
        for trace in (left, right):
            outside = trace.event("fork", scope=0)
            trace.event("scope_begin", scope=1, parent_scope=0,
                        semantic=dict(label="activation", query=0, layer=6,
                                      slot=-1, call=-1, kind=0, stage=-1))
            inside = trace.event("fork", scope=1)
            trace.event("start", 2, dependency=inside, scope=1)
            for _ in range(2):
                trace.io("send", actor=2, channel=42002)
                trace.events[-1]["scope"] = 1
                trace.io("recv", actor=2, channel=42002)
                trace.events[-1]["scope"] = 1
            inside_end = trace.event("end", 2, scope=1)
            trace.event("start", 3, dependency=outside, scope=0)
            for _ in range(4):
                trace.io("send", actor=3, channel=42003)
                trace.io("recv", actor=3, channel=42003)
            outside_end = trace.event("end", 3, scope=0)
            trace.event("join", dependency=inside_end, scope=1)
            trace.io("send", channel=42001)
            trace.events[-1]["scope"] = 1
            trace.io("recv", channel=42001)
            trace.events[-1]["scope"] = 1
            trace.event("scope_end", scope=1)
            trace.event("join", dependency=outside_end, scope=0)
            trace.io("send", channel=42009)
            trace.io("recv", channel=42009)
            trace.end()
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["phases"][0]["logical_rounds"], 5)
        scope = result["scopes"][0]
        self.assertEqual(scope["logical_rounds"], 3)
        self.assertEqual(scope["logical_rounds"], self.full_fixture_scope_depth((left, right), 1))
        self.assertEqual(scope["party_sent_bytes"], [24, 24])
        self.assertEqual(scope["party_received_bytes"], [24, 24])

    def test_scope_missing_end_or_cross_party_identity_is_rejected(self):
        left, right = self.pair()
        for trace in (left, right):
            trace.event("scope_begin", scope=1, parent_scope=0,
                        semantic=dict(label="activation", query=0, layer=6,
                                      slot=-1, call=-1, kind=0, stage=-1))
            trace.event("scope_end", scope=1)
            trace.end()
        right.events[2]["semantic"]["kind"] = 1
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "failed")
        right.events[2]["semantic"]["kind"] = 0
        right.events[3]["type"] = "join"
        right.events[3]["dependency"] = 0
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "failed")

    def test_scope_union_keeps_dependencies_through_unselected_stages(self):
        left, right = self.pair()
        for trace in (left, right):
            for identifier, stage, kind in ((1, 8, 0), (2, 7, 0), (3, 9, 0), (4, 10, 1)):
                trace.event("scope_begin", scope=identifier, parent_scope=0,
                            semantic=dict(label="activation_stage", query=0, layer=6,
                                          slot=-1, call=identifier, kind=kind, stage=stage))
                trace.io("send")
                trace.events[-1]["scope"] = identifier
                trace.io("recv")
                trace.events[-1]["scope"] = identifier
                trace.event("scope_end", scope=identifier)
            trace.end()
        result = self.analyze(left, right, scope_groups=[dict(
            label="lut_union", phase="online", source_label="activation_stage",
            stages=list(range(8, 16)), group_by=["kind"])])
        self.assertEqual(result["status"], "passed", result)
        grouped = {row["kind"]: row for row in result["scopes"] if row["label"] == "lut_union"}
        self.assertEqual(grouped[0]["send_flight_depths"], [2, 2])
        self.assertEqual(grouped[0]["party_sent_bytes"], [16, 16])
        self.assertEqual(grouped[1]["send_flight_depths"], [1, 1])
        self.assertEqual(grouped[1]["party_received_bytes"], [8, 8])

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_phase_replay_fits_bounded_memory(self):
        # Streaming fixture construction avoids counting the test's own event
        # dictionaries against the analyzer's independently enforced budget.
        iterations = 30000
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
                                   query=0, phase="online", scope=1)
                        row.update(fields)
                        stream.write(json.dumps(row) + "\n")

                    emit("phase_begin", scope=0)
                    emit("scope_begin", parent_scope=0, semantic=dict(
                        label="activation", query=0, layer=6, slot=-1, call=-1, kind=0, stage=-1))
                    for offset in range(iterations):
                        emit("send", channel=42000, offset=offset, length=1)
                        emit("recv", channel=42000, offset=offset, length=1)
                    emit("scope_end")
                    emit("phase_end", scope=0)
                    stream.write(json.dumps(dict(type="complete", events=identifier)) + "\n")
            script = """
import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 ** 2, 128 * 1024 ** 2))
sys.path.insert(0, sys.argv[1])
from round_trace import analyze_pair
result = analyze_pair(sys.argv[2], sys.argv[3])
assert result['status'] == 'passed', result
assert result['phases'][0]['send_flight_depths'] == [30000, 30000], result
assert result['scopes'][0]['send_flight_depths'] == [30000, 30000], result
print(json.dumps({'peak_rss_kib': int(next(line for line in open('/proc/self/status')
                                        if line.startswith('VmHWM:')).split()[1])}))
"""
            replay = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve().parent),
                                     *map(str, paths)], capture_output=True, text=True, timeout=30)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)

    def test_send_only_child_join_does_not_start_another_flight(self):
        left, right = self.pair()
        fork = left.event("fork")
        left.event("start", 2, dependency=fork)
        left.io("send", actor=2)
        end = left.event("end", 2)
        left.event("join", dependency=end)
        left.io("send", channel=42001)
        right.io("recv")
        right.io("recv", channel=42001)
        self.assertEqual(self.finish(left, right), 1)

    def test_concurrent_receive_join_starts_another_flight(self):
        left, right = self.pair()
        fork = left.event("fork")
        left.event("start", 2, dependency=fork)
        left.io("send", actor=2)
        end = left.event("end", 2)
        right.io("send", channel=42001)
        left.io("recv", channel=42001)
        left.event("join", dependency=end)
        left.io("send", channel=42002)
        right.io("recv")
        right.io("recv", channel=42002)
        self.assertEqual(self.finish(left, right), 2)

    def test_missing_completion_and_unmatched_bytes_fail_with_null_rounds(self):
        left, right = self.pair()
        left.io("send", 8)
        left.end()
        right.end()
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["logical_rounds"])
        right.events.pop()
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "failed")

    def test_byte_conservation_is_required(self):
        left, right = self.pair()
        for trace in (left, right):
            trace.io("send")
            trace.io("recv")
            trace.end()
        expected = [{"phases": [{"phase": "online", "query": 0,
                                 "sent_bytes": 9, "received_bytes": 8}]},
                    {"phases": [{"phase": "online", "query": 0,
                                 "sent_bytes": 8, "received_bytes": 8}]}]
        result = self.analyze(left, right, expected)
        self.assertEqual(result["status"], "failed")
        self.assertIn("bytes", result["reason"])

    def test_missing_zero_byte_phase_is_rejected(self):
        left, right = self.pair()
        left.end()
        right.end()
        rows = [{"phase": "online", "query": 0, "sent_bytes": 0, "received_bytes": 0},
                {"phase": "input", "query": 0, "sent_bytes": 0, "received_bytes": 0}]
        result = self.analyze(left, right, [{"phases": rows}, {"phases": rows}])
        self.assertEqual(result["status"], "failed")
        self.assertIn("phase set", result["reason"])

    def test_child_without_join_is_rejected(self):
        left, right = self.pair()
        for trace in (left, right):
            fork = trace.event("fork")
            trace.event("start", 2, dependency=fork)
            trace.io("send", actor=2)
            trace.io("recv", actor=2)
            trace.event("end", 2)
            trace.end()
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "failed")
        self.assertIn("fork/join", result["reason"])

    def test_phase_events_must_be_inside_boundaries(self):
        left, right = self.pair()
        for trace in (left, right):
            trace.io("send")
            trace.io("recv")
            trace.end()
            trace.events[2], trace.events[1] = trace.events[1], trace.events[2]
            trace.events[1]["id"], trace.events[2]["id"] = 1, 2
        result = self.analyze(left, right)
        self.assertEqual(result["status"], "failed")

    def test_gzip_and_multiple_phase_offsets(self):
        left, right = self.pair()
        for trace in (left, right):
            trace.io("send")
            trace.io("recv")
            trace.event("phase_end")
            trace.event("phase_begin", query=1)
            trace.io("send")
            trace.events[-1]["query"] = 1
            trace.io("recv")
            trace.events[-1]["query"] = 1
            trace.event("phase_end", query=1)
            trace.events.append({"type": "complete", "events": trace.next_id})
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, trace in enumerate((left, right)):
                path = Path(directory) / (str(index) + ".jsonl.gz")
                with gzip.open(path, "wt") as stream:
                    stream.write("".join(json.dumps(row) + "\n" for row in trace.events))
                paths.append(path)
            result = analyze_pair(*paths)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual([phase["logical_rounds"] for phase in result["phases"]], [1, 1])

    def test_cpp_actor_trace_end_to_end(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "trace-test"
            subprocess.run(["g++", "-std=c++17", "-pthread", "-I" + str(here),
                            str(here / "test_round_trace.cpp"), str(here / "round_trace.cpp"),
                            "-o", str(binary)], check=True, capture_output=True, text=True)
            paths = []
            for party in (1, 2):
                path = Path(directory) / ("party%d.jsonl" % party)
                environment = dict(os.environ, SDCF_ROUND_TRACE=str(path))
                if party == 1:
                    environment["TEST_ROUND_CHILD"] = "1"
                else:
                    environment.pop("TEST_ROUND_CHILD", None)
                subprocess.run([str(binary), str(party)], env=environment, check=True)
                paths.append(path)
            result = analyze_pair(*paths)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["phases"][0]["logical_rounds"], 1)

    def test_cpp_semantic_scope_inherits_fork_and_reports_stage_without_resetting_phase(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "trace-test"
            subprocess.run(["g++", "-std=c++17", "-pthread", "-I" + str(here),
                            str(here / "test_round_trace.cpp"), str(here / "round_trace.cpp"),
                            "-o", str(binary)], check=True, capture_output=True, text=True)
            paths = []
            for party in (1, 2):
                path = Path(directory) / ("party%d.jsonl" % party)
                environment = dict(os.environ, SDCF_ROUND_TRACE=str(path), TEST_ROUND_SCOPES="1")
                subprocess.run([str(binary), str(party)], env=environment, check=True)
                paths.append(path)
            result = analyze_pair(*paths)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["phases"][0]["logical_rounds"], 3)
        scoped = {row["label"]: row for row in result["scopes"]}
        self.assertEqual(scoped["activation"]["logical_rounds"], 2)
        self.assertEqual(scoped["activation"]["party_sent_bytes"], [16, 16])
        self.assertEqual(scoped["activation_stage"]["logical_rounds"], 1)
        self.assertEqual(scoped["activation_stage"]["party_sent_bytes"], [8, 8])
        self.assertEqual(scoped["activation_stage"]["stage"], 3)
        self.assertEqual(scoped["activation_stage"]["layer"], 6)


if __name__ == "__main__":
    unittest.main()
