"""Complete preprocessing qualification at the public two-party collector."""
import json
import unittest

import run
from run import collect
import test_native_material_coverage as coverage


class PreprocessingQualificationTests(unittest.TestCase):
    schedule = coverage.NativeMaterialCoverageTests.schedule

    def logs(self):
        result = []
        for log in coverage.NativeMaterialCoverageTests().triple_logs():
            purity = 'NATIVE_PLAN_PURITY ' + json.dumps(dict(query=0,
                separate_public_buffers=True, sent_bytes=0, received_bytes=0,
                generation_delta=0, bridge_generation_delta=0, prg_blocks_delta=0))
            log = log.replace("MATERIAL_READY ", purity + "\nMATERIAL_READY ")
            result.append(log + '\nNATIVE_PUBLIC_PLAN ' + json.dumps(dict(query=0,
                context=0, direction="straight", call=0, protocol="iknp_got", count=9,
                message_bits=13, messages_per_ot=1, widths=[])))
        for party in range(2):
            self.change(result, "NATIVE_MATERIAL_CONSUMED", lambda row: dict(row, widths=[]), party)
            self.change(result, "METRIC", lambda row: dict(row, bridge_generation_epoch_delta=0), party)
        return result

    def change(self, logs, prefix, mutate, party=0):
        changed = []
        for line in logs[party].splitlines():
            kind, payload = line.split(" ", 1)
            row = json.loads(payload)
            if kind == prefix:
                row = mutate(row)
            if row is not None:
                changed.append(kind + " " + json.dumps(row))
        logs[party] = "\n".join(changed)
        return logs

    def test_requested_mode_must_match_each_ready_record(self):
        with self.assertRaisesRegex(ValueError, "requested native preprocessing"):
            collect(self.logs(), 1, self.schedule, False, native_preprocessing=False)
        logs = self.logs()
        for party in range(2):
            logs[party] = "\n".join(line for line in logs[party].splitlines()
                if not line.startswith(("NATIVE_", "MATERIAL_READY ")))
        with self.assertRaisesRegex(ValueError, "requested native preprocessing"):
            collect(logs, 1, self.schedule, False, native_preprocessing=True)

    def test_complete_public_ot_plan_must_match_exact_consumed_widths(self):
        logs = self.logs()
        for party in range(2):
            for prefix in ("NATIVE_PUBLIC_PLAN", "NATIVE_MATERIAL_CONSUMED"):
                self.change(logs, prefix, lambda row: dict(row,
                    protocol="iknp_batched_cot", widths=[7, 13, 13]), party)
        parties, _ = collect(logs, 1, self.schedule, False, native_preprocessing=True)
        self.assertTrue(parties[0]["native_public_plan_verified"])
        for field, value in (("widths", [8, 13, 13]), ("call", 1)):
            with self.subTest(field=field):
                changed = self.change(logs[:], "NATIVE_MATERIAL_CONSUMED",
                    lambda row: dict(row, **{field: value}))
                with self.assertRaisesRegex(ValueError, "native public OT plan"):
                    collect(changed, 1, self.schedule, False, native_preprocessing=True)
        changed = self.change(logs[:], "NATIVE_PUBLIC_PLAN", lambda row: None)
        with self.assertRaisesRegex(ValueError, "native public OT plan"):
            collect(changed, 1, self.schedule, False, native_preprocessing=True)

    def test_planning_purity_must_be_complete_before_ready(self):
        parties, _ = collect(self.logs(), 1, self.schedule, False, native_preprocessing=True)
        self.assertTrue(parties[0]["native_planning_verified"])
        for field, value in (("generation_delta", 1), ("bridge_generation_delta", 1),
                             ("prg_blocks_delta", 1), ("sent_bytes", 1),
                             ("separate_public_buffers", False), (None, None)):
            with self.subTest(field=field):
                logs = self.change(self.logs(), "NATIVE_PLAN_PURITY",
                    lambda row: None if field is None else dict(row, **{field: value}))
                with self.assertRaisesRegex(ValueError, "native planning purity"):
                    collect(logs, 1, self.schedule, False, native_preprocessing=True)
        logs = self.logs()
        lines = logs[0].splitlines()
        purity = next(line for line in lines if line.startswith("NATIVE_PLAN_PURITY "))
        logs[0] = "\n".join(line for line in lines if line != purity) + "\n" + purity
        with self.assertRaisesRegex(ValueError, "native planning purity"):
            collect(logs, 1, self.schedule, False, native_preprocessing=True)

    def test_complete_preprocessing_requires_triples_and_zero_bridge_epochs(self):
        logs = self.logs()
        logs[0] = "\n".join(line for line in logs[0].splitlines()
            if not line.startswith(("NATIVE_PUBLIC_TRIPLE_PLAN ", "NATIVE_TRIPLE_")))
        with self.assertRaisesRegex(ValueError, "complete native preprocessing"):
            collect(logs, 1, self.schedule, False, native_preprocessing=True)
        for phase in ("input", "ready", "online", "output"):
            for value in (None, 1):
                with self.subTest(phase=phase, value=value):
                    def mutate(row):
                        if row.get("phase") == phase:
                            if value is None:
                                row.pop("bridge_generation_epoch_delta")
                            else:
                                row["bridge_generation_epoch_delta"] = value
                        return row
                    logs = self.change(self.logs(), "METRIC", mutate)
                    with self.assertRaisesRegex(ValueError, "bridge material generated or unmetered"):
                        collect(logs, 1, self.schedule, False, native_preprocessing=True)


class PreprocessingModeTests(unittest.TestCase):
    def test_native_mode_preserves_control_and_enforces_single_step(self):
        self.assertFalse(run.native_preprocessing_mode("SCI", False, 99))
        self.assertTrue(run.native_preprocessing_mode("SCI_preprocessed", False, 1))
        for backend in ("SCI_ours", "SCI_Beaver_segmented"):
            self.assertFalse(run.native_preprocessing_mode(backend, False, 1))
            self.assertTrue(run.native_preprocessing_mode(backend, True, 1))
        with self.assertRaisesRegex(ValueError, "SCI control"):
            run.native_preprocessing_mode("SCI", True, 1)
        for backend in ("SCI_preprocessed", "SCI_ours", "SCI_Beaver_segmented"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "single-step"):
                run.native_preprocessing_mode(backend, True, 99)


if __name__ == "__main__":
    unittest.main()
