#!/usr/bin/env python3
"""Freeze training-only Google-30 quantization, then evaluate the official valid split."""
import argparse
import json
from pathlib import Path
import sys


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def audit_splits(data):
    records = {split: [json.loads(line) for line in (data / (split + '_samples.jsonl')).read_text().splitlines()]
               for split in ('train', 'test', 'valid')}
    paths = {split: {r['path'] for r in rows} for split, rows in records.items()}
    speakers = {split: {Path(p).name.split('_nohash_')[0] for p in items}
                for split, items in paths.items()}
    for split in records:
        if len(paths[split]) != len(records[split]):
            raise ValueError('duplicate sample in ' + split)
    for a, b in [('train', 'test'), ('train', 'valid'), ('test', 'valid')]:
        if paths[a] & paths[b]:
            raise ValueError('sample overlap: ' + a + '/' + b)
        if speakers[a] & speakers[b]:
            raise ValueError('speaker overlap: ' + a + '/' + b)
    return {'samples': {s: len(r) for s, r in records.items()},
            'speakers': {s: len(r) for s, r in speakers.items()},
            'sample_overlap': 0, 'speaker_overlap': 0}


def summarize(float_logits, fixed_logits, labels):
    import numpy as np
    reference = np.asarray(float_logits, dtype=np.float64)
    actual = np.asarray(fixed_logits, dtype=np.float64)
    labels = np.asarray(labels)
    if reference.shape != actual.shape or reference.ndim != 2 or labels.shape != (len(reference),) or not len(reference):
        raise ValueError('Logits/labels must have matching nonempty sample axes')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite logits')
    ref_prediction, prediction = reference.argmax(1), actual.argmax(1)
    ref_correct, correct = ref_prediction == labels, prediction == labels
    fp32_accuracy, fixed_accuracy = 100.0 * ref_correct.mean(), 100.0 * correct.mean()
    loss = float(fp32_accuracy - fixed_accuracy)
    delta = actual - reference
    return {'samples': len(labels), 'fp32_correct': int(ref_correct.sum()),
            'fixed_correct': int(correct.sum()), 'fp32_top1_pct': float(fp32_accuracy),
            'fixed_top1_pct': float(fixed_accuracy), 'loss_pp': loss,
            'prediction_agreement_pct': float(100.0 * (ref_prediction == prediction).mean()),
            'fp32_correct_fixed_wrong': int((ref_correct & ~correct).sum()),
            'fixed_correct_fp32_wrong': int((correct & ~ref_correct).sum()),
            'quality_threshold_pp': 1.0, 'passes_1pp': loss <= 1.0, 'passes_0_5pp': loss <= 0.5,
            'logit_error': {'reference': 'TensorFlow FP32 pre-softmax logits',
                            'comparison': 'dequantized fixed pre-ArgMax logits, integer * 2**scale',
                            'units': 'dimensionless pre-softmax logit values; no fitted rescaling',
                            'aggregation': 'all samples and all classes equally weighted',
                            'elements': int(delta.size), 'mae': float(np.abs(delta).mean()),
                            'rmse': float(np.sqrt(np.square(delta).mean())),
                            'max_abs': float(np.abs(delta).max()),
                            'p95_abs': float(np.percentile(np.abs(delta), 95))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit-splits')
    audit.add_argument('--data', type=Path, required=True)
    summary = commands.add_parser('summarize')
    for name in ['float-logits', 'fixed-logits', 'labels', 'output']:
        summary.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'audit-splits':
        print(json.dumps(audit_splits(args.data), indent=2))
    elif args.command == 'summarize':
        import numpy as np
        report = summarize(np.load(str(args.float_logits)), np.load(str(args.fixed_logits)), np.load(str(args.labels)))
        write_json(args.output, report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        sys.exit(str(error))
