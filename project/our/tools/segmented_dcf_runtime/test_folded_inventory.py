"""Folded activation results require complete resident, one-use material."""
import copy
import json
import unittest

from run import aggregate_folded_activations, native_preprocessing_mode, validate_folded_config, validate_folded_inventory
from runtime import BACKENDS


class FoldedInventoryTests(unittest.TestCase):
    def inventory(self):
        plan = [dict(query=0, layer=4, slot=0, call=0, kind=0, count=13,
                     input_bits=16, output_bits=16, input_scale=7, output_scale=14,
                     lut_family=0, sign_backend=0, sign_segments=0, table_version=1),
                dict(query=0, layer=8, slot=0, call=1, kind=1, count=13,
                     input_bits=16, output_bits=16, input_scale=7, output_scale=7,
                     lut_family=0, sign_backend=0, sign_segments=0, table_version=1)]
        metrics = dict(offline=[dict(elapsed_us=1, sent_bytes=0, received_bytes=0)
                               for _ in range(18)],
                       online=[dict(elapsed_us=1, sent_bytes=0, received_bytes=0)
                              for _ in range(18)],
                       offline_wall_us=23, online_wall_us=29, resident_bytes=13 * 1024 * 4,
                       peak_working_bytes=256, online_generated=0, online_loaded_bytes=0)
        prepared = [dict(row, **copy.deepcopy(metrics), requests=2, calls=2,
                         generation_epoch=1, pre_input=True) for row in plan]
        consumed = [dict(row, **copy.deepcopy(metrics), sent_bytes=0, received_bytes=0,
                         native_ot_consumed=0, bit_triples_consumed=0) for row in plan]
        balances = [dict(query=0, generated=2, consumed=2, remaining=0,
                         generated_elements=26, consumed_elements=26,
                         online_generation_epoch_delta=0, online_generated=0, online_loaded_bytes=0)]
        phases = [dict(query=0, phase=phase, folded_generation_epoch_delta=0)
                  for phase in ("input", "ready", "online", "output")]
        return [plan, prepared, consumed, balances, phases]

    def test_complete_native_activation_inventory_is_accepted(self):
        self.assertTrue(validate_folded_inventory(*self.inventory(), 1, True))

    def test_explicit_private_sign_inventory_matches_its_owned_configuration(self):
        rows = self.inventory()
        for group in rows[:3]:
            for row in group:
                row.update(sign_backend=1, sign_segments=4)
        self.assertTrue(validate_folded_inventory(*rows, 1, True))
        config = [dict(slot=0, sigmoid="folded_sigma_f12_q10", tanh="folded_sigma_f12_q10",
                       lut_family=0, sign_backend=1, sign_segments=4, index_bits=10,
                       table_entries=1024, table_version=1, port_offset=2000,
                       selection_storage="u32_ring_shares", lut_segments=1, lut_segment_widths=[10])]
        schedule = [dict(index=4, function="Sigmoid", output_elements=13),
                    dict(index=8, function="TanH", output_elements=13)]
        self.assertEqual(validate_folded_config(config, rows[0], schedule, 1, True,
            expected_sign_backend="segmented_fss", expected_sign_segments=4)['sign_segments'], 4)
        for field, value in (("sign_backend", 0), ("sign_segments", 0), ("sign_segments", 32)):
            changed = copy.deepcopy(rows)
            changed[0][0][field] = value
            with self.assertRaises(ValueError):
                validate_folded_inventory(*changed, 1, True)
        with self.assertRaises(ValueError):
            validate_folded_config(config, rows[0], schedule, 1, True, expected_sign_backend="bitdec")
        config[0]['sign_segments'] = 2
        with self.assertRaises(ValueError):
            validate_folded_config(config, rows[0], schedule, 1, True,
                expected_sign_backend="segmented_fss", expected_sign_segments=4)

    def test_paper_backend_always_requires_single_step_preprocessing(self):
        self.assertIn("SCI_Beaver_segmented_PaperLUT", BACKENDS)
        self.assertTrue(native_preprocessing_mode("SCI_Beaver_segmented_PaperLUT", False, 1))
        with self.assertRaises(ValueError):
            native_preprocessing_mode("SCI_Beaver_segmented_PaperLUT", False, 99)

    def test_preparation_after_ready_and_consumption_before_ready_are_rejected(self):
        rows = self.inventory()
        plan = ["FOLDED_ACTIVATION_PLAN " + json.dumps(row) for row in rows[0]]
        prepared = ["FOLDED_ACTIVATION_PREPARED " + json.dumps(row) for row in rows[1]]
        consumed = ["FOLDED_ACTIVATION_CONSUMED " + json.dumps(row) for row in rows[2]]
        ready = ['MATERIAL_READY {"query":0,"strict":true}']
        self.assertTrue(validate_folded_inventory(*rows, 1, True,
            log="\n".join(plan + prepared + ready + consumed)))
        for lines in (plan + ready + prepared + consumed, plan + prepared + consumed + ready):
            with self.assertRaises(ValueError):
                validate_folded_inventory(*rows, 1, True, log="\n".join(lines))

    def test_selected_activation_must_cover_the_public_schedule(self):
        config = [dict(slot=0, sigmoid="folded_sigma_f12_q10", tanh="folded_sigma_f12_q10",
                       lut_family=0, sign_backend=0, index_bits=10, table_entries=1024,
                       table_version=1, port_offset=2000, selection_storage="u32_ring_shares",
                       lut_segments=1, lut_segment_widths=[10])]
        schedule = [dict(index=4, function="Sigmoid", output_elements=13),
                    dict(index=8, function="TanH", output_elements=13)]
        plan = self.inventory()[0]
        self.assertIsNotNone(validate_folded_config(config, plan, schedule, 1, True,
            expected_selection="both", expected_family="full_q_arithmetic"))
        for field, value in (("table_entries", 512), ("sigmoid", "native_sci"), ("lut_family", 1)):
            changed = copy.deepcopy(config)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                validate_folded_config(changed, plan, schedule, 1, True,
                    expected_selection="both", expected_family="full_q_arithmetic")
        with self.assertRaises(ValueError):
            validate_folded_config(config, plan[:1], schedule, 1, True)

    def test_lut_segmentation_must_match_the_selected_constructor(self):
        schedule = [dict(index=4, function="Sigmoid", output_elements=13),
                    dict(index=8, function="TanH", output_elements=13)]
        for family, name, segments, widths in ((0, "full_q_arithmetic", 1, [10]),
                (1, "segmented_boolean_tensor_b2a", 2, [5, 5])):
            config = [dict(slot=0, sigmoid="folded_sigma_f12_q10", tanh="folded_sigma_f12_q10",
                           lut_family=family, sign_backend=0, index_bits=10, table_entries=1024,
                           table_version=1, port_offset=2000, selection_storage="u32_ring_shares",
                           lut_segments=segments, lut_segment_widths=widths)]
            plan = [dict(row, lut_family=family) for row in self.inventory()[0]]
            contract = validate_folded_config(config, plan, schedule, 1, True,
                expected_selection="both", expected_family=name)
            self.assertEqual(contract["lut_segments"], segments)
            self.assertEqual(contract["lut_segment_widths"], widths)
            for field, value in (("lut_segments", 4), ("lut_segment_widths", [4, 3, 3])):
                changed = copy.deepcopy(config)
                changed[0][field] = value
                with self.assertRaises(ValueError):
                    validate_folded_config(changed, plan, schedule, 1, True)
                del changed[0][field]
                with self.assertRaises(ValueError):
                    validate_folded_config(changed, plan, schedule, 1, True)

    def test_complete_folded_activation_must_not_consume_native_sci_inventory(self):
        for field in ("native_ot_consumed", "bit_triples_consumed"):
            changed = self.inventory()
            changed[2][0][field] = 1
            with self.assertRaises(ValueError):
                validate_folded_inventory(*changed, 1, True)
            del changed[2][0][field]
            with self.assertRaises(ValueError):
                validate_folded_inventory(*changed, 1, True)

    def test_stage_aggregation_counts_each_message_once_and_preserves_worker_work(self):
        rows = self.inventory()
        parties = [dict(folded_activation_prepared=copy.deepcopy(rows[1]),
                        folded_activation_consumed=copy.deepcopy(rows[2])) for _ in range(2)]
        for party, sent, received, work in ((parties[0], 5, 7, 12), (parties[1], 7, 5, 17)):
            call = party["folded_activation_consumed"][0]
            call.update(sent_bytes=sent, received_bytes=received)
            call["online"][3] = dict(elapsed_us=work, sent_bytes=sent, received_bytes=received)
        actual = aggregate_folded_activations(parties, 1)
        stage = actual[0]["sigmoid"]["online"]["stages"]["sign"]
        self.assertEqual(stage["sent_bytes"], 12)
        self.assertEqual(stage["work_us_per_party"], [12, 17])
        self.assertEqual(actual[0]["sigmoid"]["online"]["sent_bytes"], 12)
        self.assertEqual(actual[0]["sigmoid"]["calls"][0]["online_wall_us"], 29)
        parties[1]["folded_activation_consumed"][0]["received_bytes"] = 6
        with self.assertRaises(ValueError):
            aggregate_folded_activations(parties, 1)

    def test_next_query_requires_new_preparation_and_zero_online_generation(self):
        rows = self.inventory()
        for index in range(len(rows)):
            rows[index] += [dict(copy.deepcopy(row), query=1) for row in rows[index]]
        with self.assertRaises(ValueError):
            validate_folded_inventory(*rows, 2, True)
        for row in rows[1]:
            if row["query"] == 1:
                row["generation_epoch"] = 2
        self.assertTrue(validate_folded_inventory(*rows, 2, True))
        for field, value in (("online_generated", 1), ("online_loaded_bytes", 4)):
            changed = copy.deepcopy(rows)
            changed[2][-1][field] = value
            with self.assertRaises(ValueError):
                validate_folded_inventory(*changed, 2, True)

    def test_offline_complete_bytes_must_cover_stage_traffic(self):
        rows=self.inventory()
        parties=[dict(folded_activation_prepared=copy.deepcopy(rows[1]),
                      folded_activation_consumed=copy.deepcopy(rows[2])) for _ in range(2)]
        for party,sent,received in ((parties[0],11,17),(parties[1],17,11)):
            call=party['folded_activation_prepared'][0]
            call.update(sent_bytes=sent,received_bytes=received)
            call['offline'][0]=dict(elapsed_us=3,sent_bytes=sent,received_bytes=received)
        actual=aggregate_folded_activations(parties,1)
        self.assertEqual(actual[0]['sigmoid']['offline']['sent_bytes'],28)
        parties[0]['folded_activation_prepared'][0]['sent_bytes']=12
        with self.assertRaises(ValueError):aggregate_folded_activations(parties,1)


if __name__ == "__main__":
    unittest.main()
