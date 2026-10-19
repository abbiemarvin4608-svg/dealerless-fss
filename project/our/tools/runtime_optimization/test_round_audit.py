import copy
import json
from pathlib import Path
import unittest

from audit_rounds import CONVENTION, traced_actual_contract


class RoundAuditContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path('/home/jiang/dealerless-lunwen/project/our/results/sci_sirnn_resident_20261012')
        cls.row_id = 'SCI_preprocessed-a2-parallel'
        row = root / 'formal' / cls.row_id
        cls.config = json.loads((row / 'config.json').read_text())
        cls.actual = json.loads((row / 'commands.json').read_text())

    def prepared(self):
        config, actual = copy.deepcopy(self.config), copy.deepcopy(self.actual)
        config.update(round_trace=True, round_convention=CONVENTION)
        for role, environment in enumerate(actual['environment'], 1):
            environment['SDCF_ROUND_TRACE'] = str(Path(environment['TMPDIR']).parent /
                                                  ('party%d.roundtrace.jsonl' % role))
        return config, actual

    def test_each_party_must_write_its_own_trace_in_the_executed_row(self):
        config, actual = self.prepared()
        self.assertTrue(traced_actual_contract(config, actual, self.row_id))
        for value in ('/tmp/another-row/party1.roundtrace.jsonl',
                      actual['environment'][1]['SDCF_ROUND_TRACE'], None):
            altered = copy.deepcopy(actual)
            if value is None:
                altered['environment'][0].pop('SDCF_ROUND_TRACE')
            else:
                altered['environment'][0]['SDCF_ROUND_TRACE'] = value
            self.assertFalse(traced_actual_contract(config, altered, self.row_id))

    def test_old_untraced_rows_and_changed_native_policy_are_rejected(self):
        self.assertFalse(traced_actual_contract(self.config, self.actual, self.row_id))
        config, actual = self.prepared()
        actual['environment'][0]['SDCF_NATIVE_PREPROCESSING'] = '0'
        self.assertFalse(traced_actual_contract(config, actual, self.row_id))

    def test_added_trace_field_does_not_hide_command_or_resource_changes(self):
        config, actual = self.prepared()
        for key, value in (('SEGFSS_PREFETCH_BYTES', '0'), ('OMP_NUM_THREADS', '4'),
                           ('SDCF_ACTIVATION_WORKERS', '4')):
            altered = copy.deepcopy(actual)
            altered['environment'][0][key] = value
            self.assertFalse(traced_actual_contract(config, altered, self.row_id))
        altered = copy.deepcopy(actual)
        altered['commands'][0][2] = '0,2'
        self.assertFalse(traced_actual_contract(config, altered, self.row_id))


if __name__ == '__main__':
    unittest.main()
