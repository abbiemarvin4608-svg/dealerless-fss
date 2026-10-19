#!/usr/bin/env python3
"""Join frozen sample IDs, predictions, and errors into the reviewable artifact."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import numpy as np
from quality import summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reproduction', type=Path, required=True)
    args = parser.parse_args()
    root, repro = args.run.resolve(), args.reproduction.resolve()
    fp32, fixed = [np.load(str(root / name)) for name in ['fp32_logits.npy', 'fixed_logits.npy']]
    labels = np.load(str(root / 'labels.npy'))
    ids = [json.loads(line) for line in (root / 'manifests/valid_samples.jsonl').read_text().splitlines()]
    if [item['label'] for item in ids] != labels.tolist():
        raise ValueError('Manifest and labels differ')
    fp32_pred, fixed_pred = fp32.argmax(1), fixed.argmax(1)
    for name, expected in [('fp32_predictions.npy', fp32_pred), ('fixed_predictions.npy', fixed_pred)]:
        if not np.array_equal(np.load(str(root / name)), expected):
            raise ValueError('Prediction vector disagrees with logits: ' + name)
    report = summarize(fp32, fixed, labels)
    report.update(evaluation_split='official valid', execution_mode='plaintext_quality_only',
                  numeric_contract='numeric_contract.json', frozen_configuration='frozen_quantization.json',
                  final_evaluation_completed_utc=datetime.now(timezone.utc).isoformat(),
                  selection_independent=True, previously_observed_posthoc=True,
                  secure_quality_status='not evaluated; SeeDot FLOATEXP is not bitwise SCI activation evidence')
    write_json(root / 'quality.json', report)
    with (root / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['row', 'sample', 'label', 'fp32_prediction', 'fixed_prediction', 'fp32_correct', 'fixed_correct', 'agreement', 'logit_mae', 'logit_max_abs'])
        delta = np.abs(fixed.astype(np.float64) - fp32.astype(np.float64))
        for i, item in enumerate(ids):
            writer.writerow([i, item['path'], labels[i], fp32_pred[i], fixed_pred[i],
                             int(fp32_pred[i] == labels[i]), int(fixed_pred[i] == labels[i]),
                             int(fp32_pred[i] == fixed_pred[i]), float(delta[i].mean()), float(delta[i].max())])
    provenance = root / 'provenance'
    provenance.mkdir(exist_ok=True)
    for name in ['train_google30_official.py', 'evaluate_google30_official.py', 'prepare_seedot.py', 'process_google30_streaming.py']:
        shutil.copy2(str(repro / 'scripts' / name), str(provenance / ('historical_' + name)))
    shutil.copy2(str(repro.parent / 'EdgeML-sirnn/tf/edgeml_tf/trainer/fastTrainer.py'), str(provenance / 'historical_fastTrainer.py'))
    shutil.copy2(str(root / 'SeeDot/temp/Predictor/output/fixed/prediction-info-testing.txt'), str(provenance / 'calibration_prediction_info.txt'))
    shutil.copy2(str(root / 'SeeDot/temp/Predictor/output/fixed/stats-testing.txt'), str(provenance / 'calibration_stats.txt'))
    env = {'python_quantizer': platform.python_version(), 'numpy_quantizer': np.__version__,
           'compiler': subprocess.check_output(['g++', '--version'], universal_newlines=True).splitlines()[0],
           'worker_limit': 4, 'blas_threads': 1,
           'timing_note': 'Quality evaluation only; no timings in this artifact are secure performance measurements.'}
    command = [str(repro / 'venv/bin/python'), '-c', 'import json, numpy, tensorflow; print(json.dumps({"tensorflow": tensorflow.__version__, "numpy": numpy.__version__}))']
    env['fp32_environment'] = json.loads(subprocess.check_output(command, universal_newlines=True))
    write_json(root / 'environment.json', env)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
