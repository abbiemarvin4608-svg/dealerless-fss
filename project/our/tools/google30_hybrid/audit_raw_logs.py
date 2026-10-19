#!/usr/bin/env python3
"""Cross-check ledger predictions and MSB metadata against completed two-party logs."""
import argparse
import json
from pathlib import Path
import re


def audit(runs):
    output = {}
    for backend in ('SCI', 'SCI_ours'):
        root = Path(runs) / backend
        rows = [json.loads(line) for line in (root / 'predictions.jsonl').read_text().splitlines()]
        batches = {}
        calls = elements = keys = 0
        for row in rows:
            batches.setdefault(row['run'], []).append(row)
        for name, batch_rows in batches.items():
            directory = root / name
            execution = json.loads((directory / 'execution.json').read_text())
            if (execution['exit_codes'] != [0, 0] or execution['backend'] != backend
                    or execution['diagnostic'] or not execution['fresh_process_and_material']):
                raise ValueError('batch execution not a completed secure run: ' + name)
            client = (directory / 'party2.log').read_text()
            server = (directory / 'party1.log').read_text()
            if 'TENSOR ' in client or 'TENSOR ' in server:
                raise ValueError('secure mode revealed intermediate tensors: ' + name)
            if re.search(r'^PRED ', server, re.M):
                raise ValueError('server log disclosed client-only output: ' + name)
            pairs = [(int(i), int(p)) for i, p in re.findall(r'^PRED (\d+) (\d+)$', client, re.M)]
            prediction = dict(pairs)
            if len(prediction) != len(pairs) or len(pairs) != len(batch_rows):
                raise ValueError('prediction coverage differs: ' + name)
            if any(prediction.get(row['sample_index']) != row['prediction'] for row in batch_rows):
                raise ValueError('logged prediction differs from ledger: ' + name)
            if backend == 'SCI_ours':
                trace = [json.loads(line) for line in (directory / 'party1.jsonl').read_text().splitlines()]
                other = [json.loads(line) for line in (directory / 'party2.jsonl').read_text().splitlines()]
                if trace != other or len(trace) != 297:
                    raise ValueError('MSB dispatch count or parties differ: ' + name)
                if any(t['input_bits'] != 16 or t['comparison_bits'] != 15 or t['nseg'] != 4
                       or not t['fresh_material'] for t in trace):
                    raise ValueError('MSB numerical contract differs: ' + name)
                if sum(t['count'] for t in trace) != len(batch_rows) * 29700:
                    raise ValueError('MSB element coverage differs: ' + name)
                calls += len(trace)
                elements += sum(t['count'] for t in trace)
                keys += sum(t['keys_generated'] for t in trace)
        output[backend] = dict(secure_samples=len(rows), successful_batches=len(batches),
            logged_predictions_equal_ledger=True, intermediates_revealed=False,
            segmented_calls=calls, segmented_elements=elements, fresh_segment_keys=keys)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.runs)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
