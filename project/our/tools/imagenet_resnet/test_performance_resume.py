"""Verify interrupted trace recovery cannot rerun or replace protocol evidence."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import run_performance as runner


class ResumeTests(unittest.TestCase):
    def test_full_measurement_requires_guard_before_writing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            from argparse import Namespace
            results = Path(temporary) / "results"
            with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "real enclosing memory guard"):
                runner.run(Namespace(results=results))
            self.assertFalse(results.exists())

    def test_cost_rows_use_separate_evidence_directories(self):
        original = runner.guarded_baseline_rows(False)
        costs = runner.guarded_baseline_rows(False, "off")
        self.assertEqual([row[:3] for row in original], [row[:3] for row in costs])
        self.assertFalse({row[3] for row in original} & {row[3] for row in costs})

    def test_guard_rejects_stale_boot_and_changed_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            group = Path(temporary) / "fixture"
            group.mkdir()
            report = Path(temporary) / "guard.json"
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            value = dict(status="running", cgroup=str(group), memory_limit_bytes=2048 << 20,
                         memory_and_swap_limit_bytes=2560 << 20, started_epoch=1, boot_id=boot)
            runner.write_json(report, value)
            (group / "memory.limit_in_bytes").write_text(str(2048 << 20))
            (group / "memory.memsw.limit_in_bytes").write_text(str(2560 << 20))
            real_read = Path.read_text

            def read_text(path, *args, **kwargs):
                return "5:memory:/fixture\n" if str(path) == "/proc/self/cgroup" else real_read(path, *args, **kwargs)

            with patch.dict("os.environ", dict(IMAGENET_MEMORY_GUARD_REPORT=str(report)), clear=True), \
                    patch.object(Path, "read_text", read_text):
                self.assertEqual(runner.require_memory_guard()["memory_limit_bytes"], 2048 << 20)
                runner.write_json(report, dict(value, boot_id="previous-boot"))
                with self.assertRaisesRegex(ValueError, "current boot"):
                    runner.require_memory_guard()
                runner.write_json(report, value)
                (group / "memory.limit_in_bytes").write_text(str(4096 << 20))
                with self.assertRaisesRegex(ValueError, "limits"):
                    runner.require_memory_guard()

    def test_baseline_refresh_requires_enclosing_guard(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "real enclosing memory guard"):
                runner.guarded_baseline_rows(True)
        rows = runner.guarded_baseline_rows(False)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row[1] == "SCI_preprocessed" for row in rows), 2)

    def test_guard_context_records_runtime_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "guard.json"
            value = dict(cgroup="/fixture", memory_limit_bytes=2048 << 20,
                         memory_and_swap_limit_bytes=2560 << 20, started_epoch=1, boot_id="boot")
            runner.write_json(report, value)
            with patch.dict("os.environ", dict(IMAGENET_MEMORY_GUARD_REPORT=str(report)), clear=True):
                self.assertEqual(runner.guard_context(), dict(value, report=str(report)))

    def fixture(self, root):
        directory, oracle = root / "run", root / "oracle"
        directory.mkdir()
        oracle.mkdir()
        identity = dict(model="resume_fixture", batch=1, backend="SCI", bits=32, frac=12,
                        queries=1, diagnostic=False, binary_sha256="binary")
        graph = dict(nodes=[dict(name="fc", op="linear")])
        status = dict(identity, status="running")
        for name in ("PLAN.json", "STATUS.json"):
            runner.write_json(directory / name, status)
        zero = dict(seconds=0., process_cpu_seconds=0., sent_bytes=0, received_bytes=0)
        rounds = dict(status="passed", analyzer="native", phases=[],
                      analyzer_binary_sha256=runner.read_json(runner.HERE / ".round_trace_native/qualified.json")["binary_sha256"])
        for query, phase in [(-1, "setup")] + [(0, name) for name in ("offline", "input", "online", "output")]:
            rounds["phases"].append(dict(query=query, phase=phase, logical_rounds=0,
                                         party_sent_bytes=[0, 0], party_received_bytes=[0, 0]))
        for role in (1, 2):
            party = dict(status="complete", role=role, model=identity["model"], backend="SCI", bits=32, frac=12,
                         method=dict(diagnostic=False), setup=zero,
                         queries=[dict(query=0, phases={name: zero for name in ("offline", "input", "online", "output")},
                                       output_codes=[0] * 1000)],
                         layers=[dict(query=0, layer=0, name="fc", op="linear")])
            runner.write_json(directory / ("party%d.json" % role), party)
            (directory / ("party%d.trace.jsonl" % role)).write_text("original protocol trace")
        runner.write_json(oracle / "index.json", dict(model=identity["model"], bits=32, frac=12,
            tensors=dict(fc=dict(file="fc.bin", elements=1000, shape=[1, 1000]))))
        (oracle / "fc.bin").write_bytes(struct.pack("<1000Q", *([0] * 1000)))
        return directory, identity, graph, oracle, rounds

    def test_recovers_trace_and_full_output_then_skips_qualified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, identity, graph, oracle, rounds = self.fixture(Path(temporary))
            original = (directory / "party1.trace.jsonl").read_bytes()
            calls = []

            def analyze(*args):
                calls.append(args)
                return rounds

            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(runner.qualify_existing(directory, identity, graph, oracle, analyze),
                                 "recovered_completed_protocol")
                self.assertEqual(runner.qualify_existing(directory, identity, graph, oracle, analyze),
                                 "skipped_qualified")
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0][2]), 2)
            self.assertEqual(len(calls[0][2][0]), 5)
            self.assertEqual((directory / "party1.trace.jsonl").read_bytes(), original)
            self.assertEqual(runner.read_json(directory / "OUTPUT_AUDIT.json")["values_checked"], 1000)
            self.assertEqual(runner.read_json(directory / "STATUS.json")["status"], "complete")
            self.assertEqual(len(list((directory / "recovery_history").glob("STATUS.*.json"))), 1)

    def test_changed_identity_and_partial_protocol_are_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, identity, graph, oracle, rounds = self.fixture(Path(temporary))
            original = (directory / "STATUS.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.qualify_existing(directory, dict(identity, binary_sha256="different"), graph, oracle)
            self.assertEqual((directory / "STATUS.json").read_bytes(), original)
            party = runner.read_json(directory / "party2.json")
            party["layers"] = []
            runner.write_json(directory / "party2.json", party)
            with self.assertRaisesRegex(ValueError, "full-graph"):
                runner.qualify_existing(directory, identity, graph, oracle)
            self.assertFalse((directory / "RECOVERY.json").exists())
            self.assertEqual((directory / "STATUS.json").read_bytes(), original)

    def test_wrong_output_never_becomes_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, identity, graph, oracle, rounds = self.fixture(Path(temporary))
            party = runner.read_json(directory / "party2.json")
            party["queries"][0]["output_codes"][17] = 1
            runner.write_json(directory / "party2.json", party)
            with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "1000-logit"):
                runner.qualify_existing(directory, identity, graph, oracle, lambda *_: rounds)
            self.assertEqual(runner.read_json(directory / "STATUS.json")["status"], "recovery_failed")
            self.assertEqual(runner.read_json(directory / "OUTPUT_AUDIT.json")["mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
