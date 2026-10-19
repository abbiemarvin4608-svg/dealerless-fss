"""Semantic traffic contracts at the public paired-results boundary."""
import copy
import unittest

import run


class ScopedMeasurementTest(unittest.TestCase):
    def test_complete_activation_and_stage_bytes_are_checked_against_native_records(self):
        stages = [dict(elapsed_us=0, sent_bytes=0, received_bytes=0) for _ in range(18)]
        stages[3] = dict(elapsed_us=20, sent_bytes=8, received_bytes=8)
        call = dict(query=0, layer=6, slot=0, call=0, kind=0, online=stages,
                    sent_bytes=8, received_bytes=8)
        party = dict(native_worker_config=[dict(worker_local_cggm_cap=True)],
                     activations=[dict(query=0, index=6)],
                     activation_workers=[dict(query=0, index=6, sent_bytes=0)],
                     folded_activation_prepared=[], folded_activation_consumed=[call],
                     folded_prepare_groups=[], dpf_lut_consumed=[])
        base = dict(query=0, layer=6, slot=0, call=0, kind=0, stage=-1,
                    instance=0, phase="online", logical_rounds=1,
                    party_sent_bytes=[8, 8], party_received_bytes=[8, 8])
        scopes = [dict(base, label="activation", slot=-1, call=-1),
                  dict(base, label="folded_call"),
                  dict(base, label="activation_stage", stage=3)]
        rounds = dict(status="passed", scopes=scopes)
        run.validate_scope_measurements(rounds, [party, copy.deepcopy(party)])
        altered = copy.deepcopy(rounds)
        altered["scopes"][2]["party_sent_bytes"] = [16, 16]
        with self.assertRaisesRegex(ValueError, "stage traffic"):
            run.validate_scope_measurements(altered, [party, copy.deepcopy(party)])
        altered = copy.deepcopy(rounds)
        altered["scopes"].pop(0)
        with self.assertRaisesRegex(ValueError, "activation scope coverage"):
            run.validate_scope_measurements(altered, [party, copy.deepcopy(party)])


if __name__ == "__main__":
    unittest.main()
