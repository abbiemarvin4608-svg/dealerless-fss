"""Native preprocessing evidence through the two-party log collector."""
import copy
import json
import unittest

from run import collect


class NativeMaterialCoverageTests(unittest.TestCase):
    schedule = [dict(index=0, function="ScalarMul", output_elements=9)]

    def logs(self):
        logs = []
        for party in (1, 2):
            rows = [("METRIC", dict(phase="setup", query=-1, seconds=1,
                                    sent_bytes=0, received_bytes=0))]
            rows += [("NATIVE_MATERIAL_PREPARED", dict(query=0, context=0,
                      direction="straight", requested=9, generated=128)),
                     ("MATERIAL_READY", dict(query=0, strict=True,
                      native_generation_epoch=1))]
            for phase in ("offline", "input", "ready"):
                rows.append(("METRIC", dict(phase=phase, query=0, seconds=1,
                    sent_bytes=0, received_bytes=0,
                    native_generation_epoch_delta=1 if phase == "offline" else 0)))
            rows += [("NATIVE_MATERIAL_CONSUMED", dict(query=0, context=0,
                      direction="straight", call=0, protocol="iknp_got", count=9,
                      message_bits=13, messages_per_ot=1, choice_correction_bits=9)),
                     ("NATIVE_MATERIAL_BALANCE", dict(query=0, context=0,
                      direction="straight", requested=9, generated=128,
                      consumed=9, padding_discarded=119, remaining=0))]
            for phase in ("online", "output"):
                rows.append(("METRIC", dict(phase=phase, query=0, seconds=1,
                    sent_bytes=0, received_bytes=0, native_generation_epoch_delta=0)))
            rows += [("COVERAGE", dict(query=0, index=0, elements=9, sample_groups=1)),
                     ("METRIC", dict(phase="teardown", query=-1, seconds=1,
                      sent_bytes=0, received_bytes=0)),
                     ("FINISHED", dict(party=party, status="complete", total_seconds=8,
                      peak_rss_kib=100))]
            logs.append("\n".join(prefix + " " + json.dumps(row) for prefix, row in rows))
        return logs

    def test_preserves_verified_generation_and_one_use_consumption(self):
        parties, _ = collect(self.logs(), 1, self.schedule, False, backend="SCI")
        self.assertEqual(parties[0]["native_material_prepared"][0]["generated"], 128)
        self.assertEqual(parties[0]["native_material_consumed"][0]["count"], 9)
        self.assertTrue(parties[0]["native_preprocessing_verified"])

    def test_rejects_post_prepare_generation_and_incomplete_inventory(self):
        for prefix, field, value in (("METRIC", "native_generation_epoch_delta", 1),
                                     ("NATIVE_MATERIAL_BALANCE", "consumed", 8),
                                     ("NATIVE_MATERIAL_PREPARED", "generated", 8)):
            with self.subTest(prefix=prefix, field=field):
                logs = self.logs()
                rows = []
                for line in logs[0].splitlines():
                    kind, payload = line.split(" ", 1)
                    row = json.loads(payload)
                    if kind == prefix and (kind != "METRIC" or row["phase"] == "input"):
                        row[field] = value
                    rows.append(kind + " " + json.dumps(row))
                logs[0] = "\n".join(rows)
                with self.assertRaisesRegex(ValueError, "native material"):
                    collect(logs, 1, self.schedule, False, backend="SCI")

    def test_rejects_different_contracts_between_parties(self):
        logs = self.logs()
        logs[1] = logs[1].replace('"message_bits": 13', '"message_bits": 12')
        with self.assertRaisesRegex(ValueError, "native material contracts"):
            collect(logs, 1, self.schedule, False, backend="SCI")

    def triple_logs(self):
        result = []
        for log in self.logs():
            rows = [
                ("NATIVE_PUBLIC_TRIPLE_PLAN", dict(query=0, context=0, call=0,
                    count=16, method=3, packed=True, offset=8)),
                ("NATIVE_TRIPLE_PREPARED", dict(query=0, context=0, requested=16,
                    generated=16, standard_calls=0, correlated_calls=1)),
                ("NATIVE_TRIPLE_CONSUMED", dict(query=0, context=0, call=0,
                    count=16, method=3, packed=True, offset=8)),
                ("NATIVE_TRIPLE_BALANCE", dict(query=0, context=0, requested=16,
                    generated=16, consumed=16, remaining=0, padding_discarded=0)),
            ]
            result.append(log + "\n" + "\n".join(prefix + " " + json.dumps(row) for prefix, row in rows))
        return result

    def test_verifies_safe_triples_against_the_complete_public_plan(self):
        parties, _ = collect(self.triple_logs(), 1, self.schedule, False, backend="SCI")
        self.assertTrue(parties[0]["native_triples_verified"])
        self.assertEqual(parties[0]["native_triple_prepared"][0]["generated"], 16)

    def test_rejects_missing_incomplete_unsafe_or_reassigned_triples(self):
        for prefix, field, value in (("NATIVE_TRIPLE_CONSUMED", "offset", 1),
                                    ("NATIVE_TRIPLE_PREPARED", "generated", 8),
                                    ("NATIVE_PUBLIC_TRIPLE_PLAN", "method", 0),
                                    ("NATIVE_TRIPLE_CONSUMED", "context", 1),
                                    ("NATIVE_TRIPLE_CONSUMED", None, None)):
            with self.subTest(prefix=prefix, field=field):
                logs = self.triple_logs()
                changed = []
                for line in logs[0].splitlines():
                    kind, payload = line.split(" ", 1)
                    if kind == prefix:
                        if field is None: continue
                        row = json.loads(payload); row[field] = value
                        line = kind + " " + json.dumps(row)
                    changed.append(line)
                logs[0] = "\n".join(changed)
                with self.assertRaisesRegex(ValueError, "native bit-triple"):
                    collect(logs, 1, self.schedule, False, backend="SCI")


if __name__ == "__main__":
    unittest.main()
