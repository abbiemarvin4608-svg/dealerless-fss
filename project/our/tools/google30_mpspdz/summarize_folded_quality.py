#!/usr/bin/env python3
"""Compare complete frozen valid predictions from the plaintext integer oracle."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from folded_semantics import table_version


def compare_predictions(labels, fp32, sci, folded, logits):
    labels, fp32, sci, folded = [np.asarray(values) for values in (labels, fp32, sci, folded)]
    if any(values.shape != (6798,) for values in (labels, fp32, sci, folded)):
        raise ValueError('quality qualification requires all 6798 frozen valid samples')
    for values in (labels, fp32, sci, folded):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= 30)):
            raise ValueError('predictions and labels must be integer Google-30 class IDs')
    logits = np.asarray(logits)
    if (logits.shape != (6798, 30) or not np.issubdtype(logits.dtype, np.integer)
            or np.any((logits < -128) | (logits > 127))):
        raise ValueError('folded logits must be the complete signed-int8 frozen graph outputs')
    if not np.array_equal(logits.argmax(axis=1), folded):
        raise ValueError('folded predictions do not match classification-head logits')
    counts = {name: int(np.count_nonzero(values == labels))
              for name, values in [('fp32', fp32), ('sci', sci), ('folded', folded)]}
    accuracy = {name: count * 100 / len(labels) for name, count in counts.items()}
    loss = (counts['fp32'] - counts['folded']) * 100 / len(labels)
    return {
        'execution_mode': 'plaintext_full_sequence_quality_only', 'secure_samples': 0,
        'samples': len(labels), 'full_holdout': True, 'sequence_steps': 99, 'classes': 30,
        'fp32_correct': counts['fp32'], 'fp32_top1_pct': accuracy['fp32'],
        'sci_correct': counts['sci'], 'sci_top1_pct': accuracy['sci'],
        'folded_correct': counts['folded'], 'folded_top1_pct': accuracy['folded'],
        'folded_loss_pp': loss,
        'sci_loss_pp': (counts['fp32'] - counts['sci']) * 100 / len(labels),
        'folded_sci_prediction_agreement_pct': float(np.mean(folded == sci) * 100),
        'folded_fp32_prediction_agreement_pct': float(np.mean(folded == fp32) * 100),
        'quality_loss_threshold_pp': 1.0, 'model_quality_qualified': loss <= 1.0,
        'folded_sci_prediction_disagreements': int(np.count_nonzero(folded != sci)),
        'sci_correct_folded_wrong': int(np.count_nonzero((sci == labels) & (folded != labels))),
        'folded_correct_sci_wrong': int(np.count_nonzero((folded == labels) & (sci != labels))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--input-csv', required=True, type=Path)
    parser.add_argument('--sci-directory', required=True, type=Path)
    parser.add_argument('--folded-directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    policy = json.loads((args.artifact / 'freeze_policy.json').read_text())
    if policy['evaluation_split'] != 'valid' or policy['final_loss_threshold_pp'] != 1.0:
        raise ValueError('expected the frozen official valid and its 1.0pp loss threshold')
    expected_csv = (args.artifact / 'heldout_input/X.csv').resolve()
    if args.input_csv.resolve() != expected_csv:
        raise ValueError('quality report requires the frozen normalized valid CSV')
    summaries = {}
    for mode, directory in [('sci', args.sci_directory), ('folded', args.folded_directory)]:
        summary = json.loads((directory / 'summary.json').read_text())
        if (summary['activation_mode'] != mode or summary['samples'] != 6798
                or not summary['full_holdout'] or summary['secure_samples'] != 0
                or summary['numeric_graph'] != 'frozen_google30_independent'
                or summary['initial_hidden'] != 'zero'):
            raise ValueError('expected complete matching plaintext integer graph results')
        summaries[mode] = summary
    if summaries['folded']['table_version'] != table_version():
        raise ValueError('folded result does not use the frozen correction table')
    labels = np.load(args.artifact / 'labels.npy')
    fp32 = np.load(args.artifact / 'fp32_predictions.npy')
    sci = np.load(args.sci_directory / 'predictions.npy')
    folded = np.load(args.folded_directory / 'predictions.npy')
    sci_logits = np.load(args.sci_directory / 'logits.npy')
    folded_logits = np.load(args.folded_directory / 'logits.npy')
    if sci_logits.shape != (6798, 30) or not np.array_equal(sci_logits.argmax(axis=1), sci):
        raise ValueError('SCI classification-head logits do not match its predictions')
    report = compare_predictions(labels, fp32, sci, folded, folded_logits)
    manifest_path = args.artifact / 'manifests/valid_samples.jsonl'
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if len(manifest) != 6798 or [row['label'] for row in manifest] != labels.tolist():
        raise ValueError('valid sample manifest does not match frozen label order')
    with args.input_csv.open() as stream:
        if sum(1 for _ in stream) != 6798:
            raise ValueError('valid feature CSV must contain exactly 6798 rows')
    report.update({
        'activation_mode': 'folded', 'numeric_graph': 'frozen_google30_independent',
        'table_version': table_version(), 'internal_fractional_bits': 12, 'index_bits': 10,
        'checkpoint_rule': policy['checkpoint_rule'], 'source_model': policy['source_model'],
        'frozen_policy_version': policy['version'], 'frozen_at_utc': policy['frozen_at_utc'],
        'normalization': policy['normalization'], 'quantization_rule': policy['quantization_rule'],
        'quality_tuning_performed': False,
        'lut_construction_applicability': 'shared by realizations proven to implement this same integer mapping',
        'sources': {'artifact': str(args.artifact.resolve()), 'input_csv': str(expected_csv),
                    'sample_manifest': str(manifest_path.resolve()),
                    'sci_results': str(args.sci_directory),
                    'folded_results': str(args.folded_directory)},
        'logit_encoding': {'bits': 8, 'fractional_bits': 2},
        'folded_sci_logits_max_integer_delta': int(np.abs(folded_logits - sci_logits).max()),
        'folded_sci_logits_mean_absolute_integer_delta': float(np.abs(folded_logits - sci_logits).mean()),
    })
    args.output.mkdir(parents=True, exist_ok=True)
    columns = ['sample_id', 'wav_path', 'class', 'label', 'fp32_prediction',
               'sci_prediction', 'folded_prediction', 'folded_matches_sci']
    rows = [[index, row['path'], row['class'], int(labels[index]), int(fp32[index]),
             int(sci[index]), int(folded[index]), int(folded[index] == sci[index])]
            for index, row in enumerate(manifest)]
    for filename, selected in [('predictions.csv', rows),
                               ('sci_folded_disagreements.csv', [row for row in rows if not row[-1]])]:
        with (args.output / filename).open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(columns)
            writer.writerows(selected)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
