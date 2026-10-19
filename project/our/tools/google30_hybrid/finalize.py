#!/usr/bin/env python3
"""Publish complete secure quality only after rechecking both ledgers against the oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full import digest, validate_ledger


def quality_from_ledgers(paths, contract, labels, fp32, oracle):
    labels, fp32, oracle = (np.asarray(x) for x in (labels, fp32, oracle))
    size = contract['samples']
    if any(x.shape != (size,) for x in (labels, fp32, oracle)):
        raise ValueError('reference length differs from contract')
    if labels.tolist() != contract['labels']:
        raise ValueError('reference labels differ from contract')
    ledgers, identities, oracle_mismatches, metadata_mismatches = {}, {}, {}, {}
    for backend in ('SCI', 'SCI_ours'):
        rows = [json.loads(line) for line in Path(paths[backend]).read_text().splitlines()]
        identities[backend] = rows[0]['identity'] if rows else None
        validate_ledger(rows, contract['sample_ids'], identities[backend])
        for row in rows:
            if row['contract'] != contract['identity'] or row.get('diagnostic') is not False:
                raise ValueError('requires same-contract non-diagnostic ledger')
            if row.get('backend') != backend or row.get('label') != int(labels[row['sample_index']]):
                raise ValueError('backend or label binding differs')
        by_id = {row['sample_index']: row for row in rows}
        ledgers[backend] = by_id
        oracle_mismatches[backend] = [i for i, row in by_id.items() if row['prediction'] != oracle[i]]
        metadata_mismatches[backend] = [i for i, row in by_id.items()
            if row.get('oracle_prediction') != oracle[i]
            or row.get('oracle_match') != bool(row['prediction'] == oracle[i])]
    shared = sorted(set(ledgers['SCI']) & set(ledgers['SCI_ours']))
    differences = [i for i in shared if ledgers['SCI'][i]['prediction'] != ledgers['SCI_ours'][i]['prediction']]
    coverage = all(sorted(rows) == list(range(size)) for rows in ledgers.values())
    oracle_equal = coverage and not any(oracle_mismatches.values()) and not any(metadata_mismatches.values())
    complete = oracle_equal and not differences
    result = dict(schema_version='google30-secure-quality-v1', contract=contract['identity'],
        backend_identities=identities, required_samples=size, samples=len(shared),
        backend_samples={key: len(rows) for key, rows in ledgers.items()},
        coverage_complete=coverage, complete=complete, full_secure_evaluation=coverage,
        full_secure_agreement=coverage and not differences, full_oracle_match=oracle_equal,
        oracle_mismatches=oracle_mismatches, oracle_metadata_mismatches=metadata_mismatches,
        prediction_differences=differences, fp32_top1_pct=float(np.mean(fp32 == labels) * 100),
        fixed_top1_pct=None, loss_pp=None, prediction_agreement_pct=None,
        quality_gate_pass=False, quality_limit_pp=1.0, quality_source='full_two_party_secure_ledgers')
    if complete:
        prediction = np.array([ledgers['SCI'][i]['prediction'] for i in range(size)])
        result['fixed_top1_pct'] = float(np.mean(prediction == labels) * 100)
        result['loss_pp'] = result['fp32_top1_pct'] - result['fixed_top1_pct']
        result['prediction_agreement_pct'] = float(np.mean(prediction == fp32) * 100)
        result['quality_gate_pass'] = result['loss_pp'] <= 1.0
        result['fp32_correct_fixed_wrong'] = int(np.sum((fp32 == labels) & (prediction != labels)))
        result['fixed_correct_fp32_wrong'] = int(np.sum((prediction == labels) & (fp32 != labels)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--SCI', type=Path, required=True)
    parser.add_argument('--hybrid', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    unsigned = {key: value for key, value in contract.items() if key != 'identity'}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != contract['identity']:
        raise ValueError('contract identity differs')
    paths = {'SCI': args.SCI, 'SCI_ours': args.hybrid}
    references = {'labels': args.quality / 'labels.npy', 'fp32': args.quality / 'fp32_predictions.npy',
                  'oracle': args.oracle, 'integer_logits': args.oracle.parent / 'logits.npy',
                  'fp32_logits': args.quality / 'fp32_logits.npy',
                  'numeric_contract': args.quality / 'numeric_contract.json'}
    result = quality_from_ledgers(paths, contract, *(np.load(references[key]) for key in ('labels', 'fp32', 'oracle')))
    result['evidence'] = {key: {'path': str(path.resolve()), 'sha256': digest(path)}
                          for key, path in dict(paths, **references).items()}
    result['numerical_contract'] = {key: contract[key] for key in
        ('generated_sha256', 'model_sha256', 'client_input_sha256', 'server_model_sha256', 'manifest_sha256')}
    # FP32 numerical error is evaluated with the exact same integer oracle,
    # separately from the class-only secure output and accuracy evidence.
    integer_logits = np.load(args.oracle.parent / 'logits.npy')
    floating_logits = np.load(args.quality / 'fp32_logits.npy')
    numeric = json.loads((args.quality / 'numeric_contract.json').read_text())
    operations = numeric['operators']
    argmax = next(op for op in operations if op['operation'] == 'ArgMax')
    scale = argmax['tensors']['A']['scale']
    if integer_logits.shape != floating_logits.shape or integer_logits.shape != (contract['samples'], 30):
        raise ValueError('logit shape differs from full evaluation')
    if not np.array_equal(integer_logits.argmax(axis=1), np.load(args.oracle)):
        raise ValueError('integer logits and oracle predictions differ')
    error = integer_logits.astype(np.float64) * (2.0 ** scale) - floating_logits
    result['logit_error'] = dict(reference='TensorFlow FP32 pre-softmax logits',
        comparison='SCI integer oracle pre-ArgMax logits * 2**scale', scale=scale,
        source='plaintext_integer_oracle; full secure predictions independently matched, secure logits are not revealed',
        units='dimensionless pre-softmax logits; no fitted rescaling', elements=error.size,
        mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error ** 2))),
        p95_abs=float(np.percentile(np.abs(error), 95)), max_abs=float(np.abs(error).max()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
