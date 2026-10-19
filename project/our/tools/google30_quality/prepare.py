#!/usr/bin/env python3
"""Prepare an isolated, training-only SeeDot workspace and freeze the policy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from quality import audit_splits, write_json

PARAMS = {'W1': (32, 16), 'W2': (16, 100), 'U1': (100, 35), 'U2': (35, 100),
          'Bg': (1, 100), 'Bh': (1, 100), 'FC': (100, 30), 'FCbias': (30,),
          'zeta': (1, 1), 'nu': (1, 1), 'mean': (3168,), 'std': (3168,)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repro, out = args.reproduction.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Use a new output directory; frozen runs are not overwritten')
    data = repro / 'data/Google-30-processed'
    source_model = repro / 'models/Google-30-FastGRNN-r16-u35-h100'
    source = repro.parent / 'EdgeML-sirnn/tools/SeeDot'
    audit = audit_splits(data)
    out.mkdir(parents=True)
    manifests = out / 'manifests'
    manifests.mkdir()
    for split in ['train', 'test', 'valid']:
        shutil.copy2(str(data / (split + '_samples.jsonl')), str(manifests))
    shutil.copy2(str(data / 'preprocessing.json'), str(manifests))
    fp32 = out / 'fp32_model'
    fp32.mkdir()
    for name, shape in PARAMS.items():
        value = np.load(str(source_model / (name + '.npy')), allow_pickle=False)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Unexpected parameter ' + name)
        shutil.copy2(str(source_model / (name + '.npy')), str(fp32))
    for name in ['training_config.json', 'training_history.json', 'command.txt']:
        shutil.copy2(str(source_model / name), str(fp32))
    train = np.load(str(data / 'train.npy'), mmap_mode='r')
    labels = np.asarray(train[:, 0], dtype=int)
    first = [int(np.flatnonzero(labels == label)[0]) for label in range(30)]
    remainder = np.setdiff1d(np.arange(len(train)), first)
    selected = np.sort(np.r_[first, np.random.RandomState(42).choice(remainder, 512 - 30, replace=False)])
    records = [json.loads(line) for line in (manifests / 'train_samples.jsonl').read_text().splitlines()]
    with (manifests / 'calibration_samples.jsonl').open('w') as stream:
        for index in selected:
            stream.write(json.dumps(dict(records[index], train_row=int(index))) + '\n')
    mean, std = [np.load(str(fp32 / (name + '.npy'))) for name in ['mean', 'std']]
    calibration = np.array(train[selected], dtype=np.float64)
    calibration[:, 1:] = (calibration[:, 1:] - mean) / std
    policy = {
        'version': 'google30-independent-quality-v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_rule': 'Reuse the sole completed official 300-epoch reproduction checkpoint by provenance; do not select among checkpoints using valid.',
        'historical_selection': 'FastTrainer compressed-phase maximum testAcc over official test; official valid is absent from training and fixer inputs.',
        'holdout': 'official valid: 6798 rows; evaluated posthoc historically, never supplied to saved-model or quantization selection in audited scripts',
        'evaluation_split': 'valid', 'calibration_split': 'train', 'calibration_rows': 512,
        'calibration_indices': selected.tolist(), 'seed': 42, 'dataset_audit': audit,
        'quantization_rule': 'One upstream SeeDot DDS/VBW search on 512 training rows: red_disagree, base16, candidate8/16/32, maxScaleRange [0,-16], 3 demotion offsets (9 for X), no iterative VBW.',
        'selection_loss_budget_pp': 1.0, 'final_loss_threshold_pp': 1.0, 'sensitivity_threshold_pp': 0.5,
        'retry_rule': 'No tuning after valid results; a failed gate remains a reported failure.',
        'system_track_budget': 'All later backend-specific quantizers use this checkpoint, calibration manifest, preset budget and rule; frozen graph track reuses the exported graph.',
        'normalization': 'saved float64 mean/std from full official train only; FP32 casts normalized input to float32; fixed CSV uses upstream %.6f then truncation toward zero',
        'input_bounds': [float(calibration[:, 1:].min()), float(calibration[:, 1:].max())],
        'final_evaluation_features_read_during_preparation': False,
        'source_edge_commit': subprocess.check_output(['git', '-C', str(source.parent.parent), 'rev-parse', 'HEAD'], universal_newlines=True).strip(),
        'source_reproduction': str(repro), 'source_model': str(source_model),
    }
    write_json(out / 'freeze_policy.json', policy)
    write_json(out / 'split_audit.json', audit)
    work = out / 'SeeDot'
    shutil.copytree(str(source), str(work), ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'temp', 'datasets', 'model', 'EzPC', 'm3dump', 'arduinodump'))
    dataset = work / 'datasets/fastgrnn/Google-30'
    model = work / 'model/fastgrnn/Google-30'
    dataset.mkdir(parents=True)
    model.mkdir(parents=True)
    for name in PARAMS:
        if name not in ['mean', 'std']:
            shutil.copy2(str(fp32 / (name + '.npy')), str(model))
    # BOTH names are training calibration copies. Upstream fixer and final smoke
    # therefore have no held-out features or labels to consult.
    for name in ['train.npy', 'test.npy']:
        np.save(str(dataset / name), calibration)
    cpp = work / 'seedot/Predictor/main.cpp'
    text = cpp.read_text()
    marker = '\t\t\t\tthreads.push_back(thread(launchThread, features_size, features_int_copy, features_intV_copy, features_float_copy, counter, vector_float_res.back(), vector_int_res.back(), vector_int_resV.back()));'
    if text.count(marker) != 1:
        raise ValueError('Unexpected upstream worker launch')
    text = text.replace(marker, marker + '\n\t\t\t\tif (threads.size() >= 4) { for (auto &worker : threads) worker.join(); threads.clear(); }')
    cpp.write_text(text)
    config = work / 'seedot/config.py'
    text = config.read_text().replace('permittedClassificationAccuracyLoss = 2.0', 'permittedClassificationAccuracyLoss = 1.0')
    config.write_text(text)
    subprocess.run([sys.executable, str(work / 'fixSeeDotInput.py'), '--seedot_file', str(work / 'seedot/compiler/input/fastgrnn-g30.sd'),
                    '--model_dir', str(model), '--dataset_dir', str(dataset), '-n', '1'], cwd=str(work), check=True)
    print('FROZEN ' + str(out / 'freeze_policy.json'))

if __name__ == '__main__':
    main()
