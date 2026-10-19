"""Public diagnostic audit fixtures with independently worked graph values."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_folded_step import audit


class FoldedStepAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='folded-step-audit-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = self.root / 'fixture'
        self.fixture.mkdir()
        self.artifact = self.root / 'artifact'
        (self.artifact / 'numeric_artifact').mkdir(parents=True)
        shapes = {'W1': (32, 16), 'W2': (16, 100), 'U1': (100, 35),
                  'U2': (35, 100), 'Bg': (1, 100), 'Bh': (1, 100),
                  'FC': (100, 30), 'FCbias': (1, 30)}
        model, header = [], []
        for name, (rows, cols) in shapes.items():
            value = 8 if name == 'Bg' else -16 if name == 'Bh' else 0
            values = [value] * (rows * cols)
            model.extend(values)
            header.append('int %s_temp[%d][%d] = {%s};' %
                          (name, rows, cols, ','.join(map(str, values))))
        np.asarray(model, dtype='<i2').tofile(self.fixture / 'server_model.i16')
        (self.artifact / 'numeric_artifact/model_fixed.h').write_text('\n'.join(header))
        np.zeros((2, 99, 32), dtype='<i2').tofile(self.fixture / 'client_input.i16')
        self.hidden = self.root / 'hidden.i16'
        np.full((1, 100), 128, dtype='<i2').tofile(self.hidden)
        self.directory = self.root / 'row'
        self.directory.mkdir()
        self.config = dict(diagnostic=True, steps=1, batch=1, queries=2, start=1,
                           hidden_file=str(self.hidden))
        self.values = {'tmp19': (16, 0), 'tmp21': (100, 0), 'tmp23': (35, 0),
                       'tmp25': (100, 0), 'tmp26': (100, 0), 'tmp27': (100, 1),
                       'tmp28': (100, 8192), 'tmp29': (100, -2), 'tmp30': (100, -2),
                       'tmp31': (100, 64), 'tmp33': (100, 8192), 'tmp34': (100, 8327),
                       'tmp35': (100, 8733), 'tmp36': (100, -136), 'H': (100, -4)}
        self.write_row()

    def write_row(self):
        result = dict(status='complete', party_exit_codes=[0, 0], config=self.config)
        (self.directory / 'result.json').write_text(json.dumps(result))
        logs = [[], []]
        for query in range(self.config['queries'] + self.config.get('warmup_queries', 0)):
            ready = 'METRIC ' + json.dumps(dict(phase='ready', query=query,
                                               sent_bytes=1, received_bytes=1))
            for log in logs:
                log.append(ready)
            for name, (cols, value) in self.values.items():
                bits = 8 if name == 'tmp23' else 16
                logs[1].append('TENSOR 0 %s %s %d 1 %d %s' %
                               (name, name, bits, cols, ' '.join([str(value)] * cols)))
        for role, log in enumerate(logs, 1):
            (self.directory / ('party%d.log' % role)).write_text('\n'.join(log) + '\n')

    def test_uses_actual_hidden_and_final_hidden_semantic_name(self):
        report = audit(self.directory, self.artifact, self.fixture, 'folded')
        self.assertEqual(report['status'], 'passed')
        self.assertEqual([row['integer_values'] for row in report['rows']], [1351, 1351])

    def test_reference_uses_model_loaded_by_the_runtime(self):
        (self.artifact / 'numeric_artifact/model_fixed.h').write_text('stale header')
        report = audit(self.directory, self.artifact, self.fixture, 'folded')
        self.assertEqual(report['status'], 'passed')

    def test_wrong_activation_mapping_reports_all_tensor_differences(self):
        self.values['tmp28'] = (100, 8196)
        self.write_row()
        report = audit(self.directory, self.artifact, self.fixture, 'folded')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([row['tensors_detail']['tmp28']['mismatches'] for row in report['rows']],
                         [100, 100])
        self.assertEqual([row['final_hidden_mismatches'] for row in report['rows']], [0, 0])

    def test_checks_all_actual_query_ids_including_warmup(self):
        self.config.update(queries=1, warmup_queries=1)
        self.write_row()
        report = audit(self.directory, self.artifact, self.fixture, 'folded')
        self.assertEqual([row['query'] for row in report['rows']], [0, 1])

    def test_reports_unbalanced_two_party_traffic(self):
        path = self.directory / 'party1.log'
        path.write_text(path.read_text().replace('"received_bytes": 1', '"received_bytes": 2'))
        report = audit(self.directory, self.artifact, self.fixture, 'folded')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['protocol']['checks']['phase_sent_received_balance'])

    def test_accepts_only_fully_accounted_native_padding(self):
        for padding in (1092, None, 1091):
            with self.subTest(padding=padding):
                self.write_row()
                balance = dict(query=0, generated=1792, consumed=700, remaining=0)
                if padding is not None:
                    balance['padding_discarded'] = padding
                for role in (1, 2):
                    path = self.directory / ('party%d.log' % role)
                    path.write_text(path.read_text() + 'NATIVE_MATERIAL_BALANCE ' +
                                    json.dumps(balance) + '\n')
                report = audit(self.directory, self.artifact, self.fixture, 'folded')
                self.assertEqual(report['status'], 'passed' if padding == 1092 else 'failed')

    def test_paper_backend_requires_complete_activation_lifecycle(self):
        self.config.update(backend='SCI_Beaver_segmented_PaperLUT', folded_activations='both')
        self.write_row()
        report = audit(self.directory, self.artifact, self.fixture, 'folded')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['protocol']['checks']['folded_lifecycle_complete'])


if __name__ == '__main__':
    unittest.main()
