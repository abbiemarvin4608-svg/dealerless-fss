"""Contract and causal-union tests; no binaries or benchmark workloads run."""
import gzip
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark
sys.path.insert(0, str(benchmark.RUNTIME))
import round_trace
import scope_rounds
import analyze
from test_round_trace import Trace


class FrozenPlanTests(unittest.TestCase):
    def frozen_plan(self, directory):
        build, fixture = directory / "build", directory / "fixture"
        build.mkdir()
        fixture.mkdir()
        for name in ("sdcf-runtime", "libsegfss_sci_bridge.so", "runtime_manifest.json", "schema.json"):
            (build / name).write_text(name)
        for name in ("contract.json", "server_model.i16", "client_input.i16"):
            (fixture / name).write_text(name)
        proof = directory / "qualification.json"
        benchmark.write(proof, dict(status="passed"))
        description = dict(backends=[row["backend"] for row in benchmark.declarations()])
        with patch.object(benchmark.subprocess, "check_output", return_value=json.dumps(description)):
            return benchmark.freeze(build / "sdcf-runtime", build / "libsegfss_sci_bridge.so", fixture, [proof])

    def test_analyzer_rejects_missing_frozen_binary_before_reading_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self.frozen_plan(directory)
            benchmark.write(directory / "manifest.json", plan)
            benchmark.write(directory / "frozen-plan.json", plan)
            Path(plan["binary"]).unlink()
            with self.assertRaisesRegex(ValueError, "frozen input.*missing"):
                analyze.analyze(directory)

    def test_analyzer_rejects_manifest_different_from_frozen_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self.frozen_plan(directory)
            benchmark.write(directory / "frozen-plan.json", plan)
            changed = copy.deepcopy(plan)
            changed["frozen_at"] = "changed after execution"
            benchmark.write(directory / "manifest.json", changed)
            with self.assertRaisesRegex(ValueError, "manifest differs from.*frozen plan"):
                analyze.analyze(directory)

    def declared_summary(self, plan):
        summary = dict(extra_group_decisions=[], rows=[dict(id=row["id"], status="complete",
            groups=[dict(benchmark.group_record(row["id"], 1), status="complete")]) for row in plan["rows"]])
        pair = dict(plan["comparisons"][0], extra_group_reasons=["step_admission"])
        order = benchmark.declare_extra_pair(pair, summary)
        order[0][1].update(status="failed", reason="controller memory cap")
        order[0][0].update(status="failed", reason="controller memory cap")
        return summary, order

    def test_analyzer_rejects_deleted_declared_failed_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self.frozen_plan(directory)
            summary, order = self.declared_summary(plan)
            order[0][0]["groups"].pop()
            order[0][0]["status"] = "complete"
            for name, value in (("manifest.json", plan), ("frozen-plan.json", plan), ("summary.json", summary)):
                benchmark.write(directory / name, value)
            with self.assertRaisesRegex(ValueError, "fresh-group inventory"):
                analyze.analyze(directory)

    def test_analyzer_rejects_deleted_decision_with_group_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self.frozen_plan(directory)
            summary, order = self.declared_summary(plan)
            summary["extra_group_decisions"].pop()
            for identity in (plan["comparisons"][0]["candidate"], plan["comparisons"][0]["baseline"]):
                row = next(row for row in summary["rows"] if row["id"] == identity)
                row["groups"].pop()
                row["status"] = "complete"
            # Keep artifacts from the deleted pair so a summary rewrite cannot hide execution evidence.
            candidate = plan["comparisons"][0]["candidate"]
            (directory / candidate / "group2").mkdir(parents=True)
            (directory / (candidate + ".group2.runner.log")).write_text("partial")
            for name, value in (("manifest.json", plan), ("frozen-plan.json", plan), ("summary.json", summary)):
                benchmark.write(directory / name, value)
            with self.assertRaisesRegex(ValueError, "undeclared (group directory|runner artifact)"):
                analyze.analyze(directory)

    def test_verification_rejects_removed_binary_identity_and_stale_binary_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self.frozen_plan(directory)
            removed = copy.deepcopy(plan)
            del removed["file_identities"][plan["binary"]]
            with self.assertRaisesRegex(ValueError, "frozen input inventory"):
                benchmark.verify(removed)
            stale = copy.deepcopy(plan)
            stale["binary_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "binary identity"):
                benchmark.verify(stale)

    def test_verification_keeps_frozen_statistics_and_sample_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.frozen_plan(Path(temporary))
            for key, value in (("seed", 0), ("interval", .99), ("method", "modified")):
                changed = copy.deepcopy(plan)
                changed["bootstrap"][key] = value
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, "statistical contract"):
                    benchmark.verify(changed)
            for key, value in (("query_count", 2), ("warmup_queries", 1), ("timeout_seconds", 60)):
                changed = copy.deepcopy(plan)
                changed[key] = value
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, "query contract"):
                    benchmark.verify(changed)

    def test_ten_fixed_rows_and_matched_sign_parents(self):
        rows = benchmark.declarations()
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row["id"] for row in rows}), 10)
        self.assertEqual(sum(row["role"] == "sign_candidate" for row in rows), 2)
        for row in rows:
            command = benchmark.command(row, Path("/binary"), Path("/fixture"))
            self.assertNotIn("--diagnostic", command)
            self.assertEqual(command[command.index("--queries") + 1], "10")
            self.assertEqual(command[command.index("--warmup") + 1], "0")
            self.assertEqual(command[command.index("--timeout") + 1], "600")
            self.assertEqual(command[command.index("--segment-policy") + 1], "fixed-4")
        for pair in benchmark.comparisons():
            if pair["kind"] != "sign_candidate":
                continue
            before, after = (next(row for row in rows if row["id"] == pair[key]) for key in ("baseline", "candidate"))
            for key in ("backend", "family", "graph_mode", "activation_workers"):
                self.assertEqual(before[key], after[key])

    def test_missing_rows_remain_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = dict(rows=benchmark.declarations(), comparisons=benchmark.comparisons(),
                bootstrap={}, encoding={}, communication_formula="", interpretation={})
            benchmark.write(directory / "manifest.json", plan)
            benchmark.write(directory / "frozen-plan.json", plan)
            with patch.object(benchmark, "verify"):
                result = analyze.analyze(directory)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["rows"]), 10)
            self.assertTrue(all(row["status"] == "not_started" for row in result["rows"].values()))
            self.assertTrue(all(pair["status"] == "unavailable" for pair in result["comparisons"]))

    def test_lut_event_window_is_party_span_not_worker_sum(self):
        def call(begin, end):
            stages = [dict(begin_us=begin, end_us=end) for _ in range(18)]
            return dict(query=0, kind=0, online=stages)
        parties = [dict(folded_activation_consumed=[call(100, 300), call(200, 500)]),
                   dict(folded_activation_consumed=[call(1000, 1300), call(1100, 1500)])]
        result = analyze.window(parties, 0, 0, (14, 15))
        self.assertEqual(result["per_party_seconds"], [.0004, .0005])
        self.assertEqual(result["seconds"], .0005)
        del parties[0]["folded_activation_consumed"][0]["online"][14]["begin_us"]
        self.assertIsNone(analyze.window(parties, 0, 0, (14, 15)))


class ScopeUnionTests(unittest.TestCase):
    def scoped_exchange(self, trace, scope, stage, length, actor=1):
        trace.event("scope_begin", actor, scope=scope, parent_scope=0,
            semantic=dict(label="activation_stage", query=0, layer=6, slot=0, call=actor, kind=0, stage=stage))
        for direction in ("send", "recv"):
            trace.io(direction, length, actor=actor, channel=42000 + actor)
            trace.events[-1]["scope"] = scope
        trace.event("scope_end", actor, scope=scope)

    def result(self, traces):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / ("party%d.jsonl" % index) for index in (1, 2)]
            for path, trace in zip(paths, traces):
                path.write_text("".join(json.dumps(event) + "\n" for event in trace.events))
            with scope_rounds.with_lut_unions():
                result = round_trace.analyze_pair(*paths)
            self.assertEqual(result["status"], "passed", result)
            return result

    def parallel(self, serial=False):
        traces = Trace(1), Trace(2)
        for trace in traces:
            trace.begin()
            fork = trace.event("fork")
            ends = []
            for actor, length in ((2, 5), (3, 7)):
                if serial and actor == 3:
                    trace.event("join", dependency=ends.pop())
                    fork = trace.event("fork")
                trace.event("start", actor, dependency=fork)
                self.scoped_exchange(trace, actor, 14, length, actor)
                ends.append(trace.event("end", actor))
            for end in ends:
                trace.event("join", dependency=end)
            trace.end()
        return traces

    def test_parallel_scopes_use_union_dag_and_exclude_external_bytes(self):
        result = self.result(self.parallel())
        union = next(row for row in result["scopes"] if row["label"] == "ticket11_lut_union")
        self.assertEqual(union["logical_rounds"], 1)
        self.assertEqual(union["party_sent_bytes"], [12, 12])
        self.assertEqual(union["party_received_bytes"], [12, 12])

    def test_selected_dependency_serializes_selected_flights(self):
        result = self.result(self.parallel(serial=True))
        union = next(row for row in result["scopes"] if row["label"] == "ticket11_lut_union")
        self.assertEqual(union["logical_rounds"], 2)

    def test_union_bounds_and_excluded_stage_keep_external_dependencies(self):
        traces = Trace(1), Trace(2)
        for trace in traces:
            trace.begin()
            for _ in range(4):
                trace.io("send")
                trace.io("recv")
            self.scoped_exchange(trace, 1, 8, 5)
            self.scoped_exchange(trace, 2, 16, 19)
            self.scoped_exchange(trace, 3, 15, 7)
            for _ in range(3):
                trace.io("send")
                trace.io("recv")
            trace.end()
        result = self.result(traces)
        union = next(row for row in result["scopes"] if row["label"] == "ticket11_lut_union")
        self.assertEqual(union["logical_rounds"], 2)
        self.assertEqual(union["party_sent_bytes"], [12, 12])
        self.assertEqual(result["phases"][0]["logical_rounds"], 10)

    def test_wrapper_restores_public_analyzer_after_error(self):
        original = round_trace.analyze_pair
        with self.assertRaises(RuntimeError):
            with scope_rounds.with_lut_unions():
                self.assertIsNot(round_trace.analyze_pair, original)
                raise RuntimeError("interrupted")
        self.assertIs(round_trace.analyze_pair, original)


class FreshGroupTests(unittest.TestCase):
    def group(self, number, value):
        query = dict(query=0, step={phase: dict(wall_seconds=value, sent_bytes=10, causal_rounds=1,
            network={net: dict(total_seconds=1.) for net in benchmark.NETWORKS}) for phase in analyze.PHASES},
            activations={name: dict(online=dict(wall_seconds=value, sent_bytes=10, causal_rounds=1,
                network={net: dict(total_seconds=1.) for net in benchmark.NETWORKS})) for name in analyze.KINDS})
        return dict(group=number, path="row/group%d" % number, data=dict(queries=[dict(query, query=index) for index in range(10)],
            setup=dict(seconds=value, sent_bytes=10), model_prepare=dict(seconds=.1, sent_bytes=0),
            first_query=dict(seconds=2 * value, sent_bytes=20), session_total_seconds=11 * value,
            public_table_prepare=None, peak_rss_kib_per_party=[1, 2],
            process_resource_peaks=[dict(vm_peak_bytes=4096, rss_peak_bytes=1024), dict(vm_peak_bytes=8192, rss_peak_bytes=2048)]))

    def row(self, values):
        groups = [self.group(index, value) for index, value in enumerate(values, 1)]
        return dict(status="qualified", groups=groups, data=analyze.pooled_data(groups))

    def test_samples_keep_all_queries_and_group_strata(self):
        row = self.row([2., 4.])
        samples = analyze.sample_groups(row, analyze.metric_samples)
        self.assertEqual(samples["step/online"], [[2.] * 10, [4.] * 10])
        self.assertEqual([query["group"] for query in row["data"]["queries"]], [1] * 10 + [2] * 10)
        value = analyze.ratio(tuple(tuple(group) for group in samples["step/online"]), ((1.,) * 10, (2.,) * 10))
        self.assertEqual(value["baseline_group_sizes"], [10, 10])
        self.assertEqual(value["candidate_group_sizes"], [10, 10])
        self.assertEqual(value["speedup"], 2.)

    def test_analyzer_pools_two_groups_and_preserves_later_failed_declarations(self):
        declaration, companion = benchmark.declarations()[5:7]
        identity, baseline = declaration["id"], companion["id"]
        pair = dict(id="pair", kind="method_comparison", baseline=baseline, candidate=identity,
                    extra_group_reasons=["step_admission"])
        plan = dict(rows=[declaration, companion], comparisons=[pair], bootstrap={}, encoding={},
                    communication_formula="", interpretation={})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            benchmark.write(directory / "manifest.json", plan)
            benchmark.write(directory / "frozen-plan.json", plan)
            summary = dict(extra_group_decisions=[], rows=[dict(id=name, status="complete",
                groups=[benchmark.group_record(name, 1)]) for name in (identity, baseline)])
            order = benchmark.declare_extra_pair(pair, summary)
            groups = summary["rows"][0]["groups"]
            for group in groups + summary["rows"][1]["groups"]:
                group["status"] = "complete"
            benchmark.write(directory / "summary.json", summary)
            qualified = [dict(status="qualified", data=self.group(number, value)["data"])
                         for number, value in ((1, 2.), (2, 4.), (1, 3.), (2, 5.))]
            with patch.object(benchmark, "verify"), patch.object(analyze, "qualify", side_effect=qualified):
                report = analyze.analyze(directory)
            row = report["rows"][identity]
            self.assertEqual(row["distributions"]["step/online"]["median"], 3.)
            self.assertEqual(row["cost_distributions"]["step/online"]["sent_bytes"]["samples"], 20)
            self.assertEqual(row["accounting"]["setup"]["wall_seconds"]["median"], 3.)
            self.assertEqual(len(row["groups"]), 2)
            self.assertIn("Setup (s)", analyze.markdown(report))
            groups[1].update(status="failed", reason="controller memory cap")
            benchmark.declare_extra_pair(pair, summary)
            summary["rows"][0].update(status="failed", groups=groups)
            benchmark.write(directory / "summary.json", summary)
            with patch.object(benchmark, "verify"), patch.object(analyze, "qualify", return_value=qualified[0]):
                failed = analyze.analyze(directory)["rows"][identity]
            self.assertEqual([group["status"] for group in failed["groups"]], ["qualified", "failed", "not_started"])
            self.assertNotIn("distributions", failed)
            self.assertIn("memory cap", failed["reason"])

    def test_equal_network_is_unchanged_and_budget_stops_unresolved(self):
        rows = dict(before=self.row([1.] * 3), after=self.row([1.] * 3))
        pair = dict(id="pair", baseline="before", candidate="after", kind="lut_construction")
        result = analyze.compare(pair, rows)
        self.assertEqual(result["network"]["step/online/LAN"]["verdict"], "unchanged")
        self.assertEqual(result["default_verdict"], "unresolved")
        self.assertEqual(result["additional_groups_status"], "blocked_by_group_budget")
        self.assertFalse(result["additional_groups_needed"])

    def test_group_inventory_allows_declared_partial_or_failed_pair(self):
        baseline, candidate = benchmark.declarations()[:2]
        pair = dict(id="pair", kind="lut_construction", baseline=baseline["id"], candidate=candidate["id"])
        plan = dict(rows=[baseline, candidate], comparisons=[pair], maximum_groups_per_row=3)
        summary = dict(extra_group_decisions=[dict(comparison="pair", reasons=["step_admission"],
            order=[candidate["id"] + "/group2", baseline["id"] + "/group2"])], rows=[])
        for declaration, status in ((baseline, "failed"), (candidate, "complete")):
            groups = [benchmark.group_record(declaration["id"], 1),
                      benchmark.group_record(declaration["id"], 2, "pair")]
            groups[0]["status"] = "complete"
            groups[1]["status"] = "not_started" if status == "failed" else "complete"
            summary["rows"].append(dict(id=declaration["id"], status=status, groups=groups))
        with tempfile.TemporaryDirectory() as temporary:
            executions = analyze.execution_groups(Path(temporary), summary, plan)
            self.assertEqual(executions[baseline["id"]]["groups"][1]["status"], "not_started")
            self.assertEqual(executions[candidate["id"]]["groups"][1]["status"], "complete")

    def test_matched_extras_are_declared_reverse_order_with_shared_row_cap(self):
        summary = dict(extra_group_decisions=[], rows=[dict(id=name, groups=[benchmark.group_record(name, 1)])
                                                     for name in ("before", "after")])
        pair = dict(id="pair", baseline="before", candidate="after", extra_group_reasons=["step_admission"],
                    additional_groups_needed=True)
        plan = dict(maximum_groups_per_row=3)
        self.assertIs(benchmark.next_extra_pair(dict(comparisons=[pair]), summary, plan), pair)
        order = benchmark.declare_extra_pair(pair, summary)
        self.assertEqual([row["id"] for row, _ in order], ["after", "before"])
        self.assertEqual([group["group"] for _, group in order], [2, 2])
        benchmark.declare_extra_pair(pair, summary)
        self.assertIsNone(benchmark.next_extra_pair(dict(comparisons=[pair]), summary, plan))
        with self.assertRaisesRegex(ValueError, "budget"):
            benchmark.declare_extra_pair(pair, summary)

    def test_archive_checks_exact_content_and_footer_before_deleting_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "party1.roundtrace.jsonl"
            content = '{"type":"meta","version":1,"party":1}\n{"type":"complete","events":0}\n'
            raw.write_text(content)
            record = benchmark.archive_trace(raw, require_complete=True)
            self.assertTrue(record["complete_footer"])
            self.assertFalse(raw.exists())
            with gzip.open(record["path"], "rt") as stream:
                self.assertEqual(stream.read(), content)
            incomplete = Path(temporary) / "party2.roundtrace.jsonl"
            incomplete.write_text('{"type":"meta","version":1,"party":2}\n')
            with self.assertRaisesRegex(ValueError, "footer"):
                benchmark.archive_trace(incomplete, require_complete=True)
            self.assertTrue(incomplete.exists())

    def test_runner_completes_initial_cohort_before_bounded_matched_extras(self):
        declarations = benchmark.declarations()
        for row in declarations:
            row["command"] = ["--output", "{output}"]
        plan = dict(rows=declarations, maximum_groups_per_row=3)
        before, after = (row["id"] for row in declarations[:2])
        calls = []

        def launch(command, **unused):
            self.assertIn("/usr/bin/prlimit", command)
            self.assertIn("--as=%d" % benchmark.MEMORY, command)
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            benchmark.write(output / "result.json", dict(status="complete"))
            calls.append(output)
            return Mock(wait=Mock(return_value=0))

        def inspect(directory):
            summary = benchmark.read(Path(directory) / "summary.json")
            counts = {row["id"]: len(row["groups"]) for row in summary["rows"]}
            pending = counts[before] < 3
            return dict(status="incomplete" if pending else "complete", sampling_status="pending_extra_groups" if pending else "exhausted_budget",
                rows={row["id"]: dict(status="qualified") for row in summary["rows"]},
                comparisons=[dict(id="pair", baseline=before, candidate=after, additional_groups_needed=pending,
                                  extra_group_reasons=["step_admission"])])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "formal"
            with patch.object(benchmark, "verify"), patch.object(benchmark.fcntl, "flock"), \
                    patch.object(benchmark.subprocess, "Popen", side_effect=launch), \
                    patch.object(analyze, "analyze", side_effect=inspect), patch.object(analyze, "save_report"):
                summary = benchmark.run_plan(plan, output)
            self.assertEqual([path.relative_to(output).as_posix() for path in calls[:10]],
                             [row["id"] for row in declarations])
            self.assertEqual([path.relative_to(output).as_posix() for path in calls[10:]],
                             [after + "/group2", before + "/group2", after + "/group3", before + "/group3"])
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(summary["extra_group_decisions"]), 2)
            self.assertTrue(all(len(row["groups"]) <= 3 for row in summary["rows"]))


if __name__ == "__main__":
    unittest.main()
