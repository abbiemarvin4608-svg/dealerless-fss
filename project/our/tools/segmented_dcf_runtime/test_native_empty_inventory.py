"""Empty SCI inventories require an explicit, complete public-plan declaration."""
import unittest

from run import (validate_native_materials, validate_native_public_plan,
                 validate_native_triples, validate_native_plan_summary)


class NativeEmptyInventoryTests(unittest.TestCase):
    def setUp(self):
        self.ready = [dict(query=q, strict=True) for q in range(2)]
        self.phases = [dict(query=q, phase=p, native_generation_epoch_delta=0)
                       for q in range(2) for p in ('input', 'ready', 'online', 'output')]
        self.summary = [dict(query=q, registered_ot_streams=2,
                             registered_triple_contexts=1, ot_calls=0, triple_calls=0)
                        for q in range(2)]

    def test_explicit_empty_plan_and_strict_lifecycle_verify(self):
        validate_native_plan_summary(self.summary, [], [], 2)
        self.assertEqual(validate_native_materials(self.ready, [], [], [], self.phases,
                                                   2, empty_verified=True), (True, []))
        self.assertEqual(validate_native_public_plan([], [], 2, empty_verified=True), [])
        self.assertEqual(validate_native_triples([], [], [], [], self.ready,
                                                 2, empty_verified=True), (True, []))

    def test_missing_duplicate_or_nonzero_declarations_rejected(self):
        for rows in ([], self.summary[:1], self.summary * 2,
                     [dict(self.summary[0], ot_calls=1), self.summary[1]],
                     [dict(self.summary[0], registered_ot_streams=0), self.summary[1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_native_plan_summary(rows, [], [], 2)

    def test_empty_declaration_cannot_cover_consumption_or_online_generation(self):
        used = [dict(query=0, context=0, direction='straight', call=0, count=1)]
        with self.assertRaises(ValueError):
            validate_native_materials(self.ready, [], used, [], self.phases, 2,
                                      empty_verified=True)
        self.phases[0]['native_generation_epoch_delta'] = 1
        with self.assertRaises(ValueError):
            validate_native_materials(self.ready, [], [], [], self.phases, 2,
                                      empty_verified=True)


if __name__ == '__main__':
    unittest.main()
