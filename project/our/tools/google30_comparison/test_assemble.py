import json
from pathlib import Path
import tempfile
import unittest

from assemble import assemble


class AssemblyTests(unittest.TestCase):
    def test_single_repeat_cannot_make_top_level_ranking_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'runs'/'lan'/'SCI_repeat1'
            run.mkdir(parents=True)
            (run/'result.json').write_text(json.dumps(dict(status='complete', backend='SCI',
                track='same_graph', network={'name':'lan'}, run_kind='repeat',
                total_seconds=10, link_bytes=100, query_count=1,
                config_id='native', comparison_group_id='same-resources',
                quality_backend_identity='secure-id')))
            (root/'quality.json').write_text(json.dumps(dict(samples=6798, loss_pp=.4,
                backend_identities={'SCI':'secure-id'})))
            (root/'comparison.json').write_text(json.dumps(dict(complete=True,
                compared_secure_samples=6798, full_secure_agreement=True, full_oracle_match=True)))
            rows = assemble(root, root/'quality.json', root, root/'table')
            self.assertTrue(rows[0]['eligible'])
            summary = json.loads((root/'table'/'summary.json').read_text())
            self.assertEqual(summary['ranking_status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
