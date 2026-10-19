#!/usr/bin/env python3
"""Recompile and execute the frozen historical SeeDot program; never requantize."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True, help='Historical reproduction root containing immutable input CSV')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', default='all', help='all or comma separated zero-based row indices')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.archive / 'SeeDot/temp/Predictor'
    data = args.runtime / 'SeeDot/temp/Predictor/input'
    labels = [int(float(r[0])) for r in csv.reader((data / 'Y.csv').open())]
    selected = list(range(len(labels))) if args.rows == 'all' else [int(x) for x in args.rows.split(',')]
    if len(set(selected)) != len(selected) or any(x < 0 or x >= len(labels) for x in selected):
        raise ValueError('rows must be distinct valid indices')
    info = (source / 'output/fixed/prediction-info-testing.txt').read_text()
    if int(re.search(r'#test points\s*=\s*(\d+)', info)[1]) != len(labels):
        raise ValueError('archived reference and input row counts differ')
    if 'logProgramOutput = true' not in (source / 'datatypes.h').read_text():
        raise ValueError('reference error logging must be enabled')
    reference = labels.copy()
    errors = list(re.finditer(r'^Main: Incorrect prediction for input (\d+) element 0\. Predicted (\d+) Expected (\d+)$', info, re.M))
    if len(errors) != len(labels) - int(re.search(r'Correct predictions\s*=\s*(\d+)', info)[1]):
        raise ValueError('incomplete archived error records')
    for m in errors:
        row = int(m[1]) - 1
        if labels[row] != int(m[3]):
            raise ValueError('reference label mismatch')
        reference[row] = int(m[2])
    build = out / 'build'
    build.mkdir(exist_ok=True)
    for name in ['seedot_fixed.cpp', 'datatypes.h', 'predictors.h', 'profile.h', 'library_fixed.h', 'model_fixed.h', 'vars_fixed.h']:
        shutil.copy2(source / name, build / name)
    shutil.copy2(HERE / 'replay_main.cpp', build / 'replay_main.cpp')
    command = ['g++', '-O3', '-std=c++11', '-fwrapv', 'replay_main.cpp', 'seedot_fixed.cpp', '-o', 'replay']
    result = subprocess.run(command, cwd=build, text=True, capture_output=True)
    (out / 'compile.log').write_text(result.stdout + result.stderr)
    result.check_returncode()
    row_file = build / 'rows.txt'
    row_file.write_text('\n'.join(map(str, sorted(selected))) + '\n')
    executed = subprocess.run([str(build / 'replay'), str(data / 'X.csv'), str(row_file)], check=True, text=True, capture_output=True)
    replay = {int(r[0]): int(r[1]) for r in csv.reader(executed.stdout.splitlines())}
    if set(replay) != set(selected):
        raise ValueError('replay omitted a requested row')
    secure = {int(r['test_row_index']): r for r in csv.DictReader((args.archive / 'secure_samples.csv').open())}
    rows = []
    for row in selected:
        prior = secure.get(row)
        rows.append(dict(row=row, label=labels[row], prediction=replay[row], archived_seedot=reference[row],
                         matches_archived_seedot=replay[row] == reference[row],
                         historical_sci_prediction=int(prior['sirnn_prediction']) if prior else '',
                         matches_historical_sci=replay[row] == int(prior['sirnn_prediction']) if prior else '',
                         wav_path=prior['wav_path'] if prior else ''))
    with (out / 'predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    secure_rows = [r for r in rows if r['historical_sci_prediction'] != '']
    with (out / 'historical_secure_differences.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(secure_rows)
    summary = dict(samples=len(rows), correct=sum(r['prediction'] == r['label'] for r in rows),
                   mismatches_to_archived_seedot=sum(not r['matches_archived_seedot'] for r in rows),
                   historical_sci_samples_compared=len(secure_rows), historical_sci_mismatches=sum(not r['matches_historical_sci'] for r in secure_rows),
                   sequence_steps=99, classifier_classes=30, execution='compiled historical SeeDot reference, plaintext',
                   historical_secure_scope='30 historical selected samples; not full-test secure accuracy',
                   reference_nonlinearity='FLOATEXP: float exp/tanh followed by integer cast; distinct from SCI integer approximation',
                   quality_scope='historical replay, not independent final quality', compiler=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0], compile_command=command)
    dump(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))
    if summary['mismatches_to_archived_seedot'] or summary['historical_sci_mismatches']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
