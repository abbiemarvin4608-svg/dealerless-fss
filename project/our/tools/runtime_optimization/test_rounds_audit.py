"""Archive support and fresh-control checks for qualified round measurements."""
import gzip
from pathlib import Path
import tempfile
import unittest

import audit_preprocessed as base
import audit_rounds as audit


class RoundsAuditTests(unittest.TestCase):
    def test_archived_traces_use_gzip_when_raw_files_are_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory/'party1.roundtrace.jsonl.gz'
            with gzip.open(path, 'wt') as stream:
                stream.write('{"type":"meta","party":1,"version":1}\n')
            self.assertEqual(audit.trace_path(directory, 1), path)
            raw = directory/'party1.roundtrace.jsonl'
            raw.write_text('raw evidence\n')
            self.assertEqual(audit.trace_path(directory, 1), raw)
            with self.assertRaises(FileNotFoundError):
                audit.trace_path(directory, 2)

    def test_prepared_default_and_nonzero_diagnostics_equal_fresh_sci_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            declarations = []
            for state in ('default', 'nonzero'):
                frozen = base.RESULTS/'reference'/state/'SCI-n4-t4/party2.log'
                tensors = [line for line in frozen.read_text().splitlines() if line.startswith('TENSOR ')]
                for backend in ('SCI', 'SCI_preprocessed', 'SCI_Beaver_segmented'):
                    row_id = state+'-a2-parallel-'+backend
                    declarations.append(dict(id=row_id, hidden=state, backend=backend, cohort='a2-parallel'))
                    output = directory/row_id
                    output.mkdir()
                    (output/'party1.log').write_text('no revealed tensors\n')
                    (output/'party2.log').write_text('\n'.join(tensors*2)+'\n')
            checked = audit.fresh_integer_checks(directory, declarations)
            self.assertTrue(all(checked.values()), checked)
            changed = directory/'nonzero-a2-parallel-SCI_Beaver_segmented/party2.log'
            changed.write_text(changed.read_text().replace('TENSOR ', 'MISSING ', 1))
            checked = audit.fresh_integer_checks(directory, declarations)
            self.assertFalse(checked['nonzero-a2-parallel-SCI_Beaver_segmented'])
            self.assertTrue(checked['default-a2-parallel-SCI_Beaver_segmented'])

    def test_default_sci_zero_communication_offline_phase_has_zero_rounds(self):
        rounds = {(query, phase): dict(logical_rounds=0 if phase == 'offline' else 7,
                    party_sent_bytes=[0, 0] if phase == 'offline' else [8, 8])
                  for query in range(2) for phase in ('offline', 'online')}
        self.assertTrue(audit.phase_round_coverage(rounds, 2))
        rounds[(0, 'online')]['logical_rounds'] = 0
        self.assertFalse(audit.phase_round_coverage(rounds, 2))
        rounds[(0, 'online')]['logical_rounds'] = -1
        self.assertFalse(audit.phase_round_coverage(rounds, 2))
        del rounds[(0, 'online')]
        self.assertFalse(audit.phase_round_coverage(rounds, 2))


if __name__ == '__main__':
    unittest.main()
