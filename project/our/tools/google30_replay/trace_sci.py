#!/usr/bin/env python3
"""Instrument isolated SCI sources, execute two real parties, collect public call metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
SOURCES = ['library_fixed.cpp', 'BuildingBlocks/aux-protocols.cpp',
           'BuildingBlocks/truncation.cpp', 'BuildingBlocks/value-extension.cpp',
           'Math/math-functions.cpp', 'LinearOT/linear-ot.cpp']

def inject(text, signature, code):
    match = re.search(signature + r'[^{}]*\{', text, re.S)
    if not match:
        raise ValueError('unrecognized frozen SCI source signature: ' + signature)
    return text[:match.end()] + '\n' + code + '\n' + text[match.end():]

def instrument(src):
    patches = {
        'BuildingBlocks/aux-protocols.cpp': [
            (r'void AuxProtocols::MSB\(', 'MSB', 'size', 'bw_x', '1'),
            (r'void AuxProtocols::wrap_computation\(', 'wrap', 'size', 'bw_x', '1'),
            (r'void AuxProtocols::lookup_table\(', 'OT_LUT', 'size', 'bw_x', 'bw_y'),
            (r'void AuxProtocols::MSB_to_Wrap\(', 'MSB_to_Wrap', 'size', 'bw_x', '1'),
        ],
        'Millionaire/millionaire.h': [
            (r'void compare\(', 'compare', 'num_cmps', 'bitlength', '1', '(greater_than ? 1 : 0) + (equality ? 2 : 0)'),
        ],
        'Millionaire/equality.h': [(r'void check_equality\(', 'equality', 'num_eqs', 'bitlength', '1')],
        'Millionaire/millionaire_with_equality.h': [
            (r'void compare_with_eq\(', 'compare_with_eq', 'num_cmps', 'bitlength', '2', 'greater_than ? 1 : 0'),
        ],
        'BuildingBlocks/truncation.cpp': [
            (r'void Truncation::div_pow2\(', 'div_pow2', 'dim', 'bw', 'bw', 'shift'),
            (r'void Truncation::truncate\(', 'truncate', 'dim', 'bw', 'bw', 'shift'),
            (r'void Truncation::truncate_and_reduce\(', 'truncate_and_reduce', 'dim', 'bw', 'bw - shift', 'shift'),
        ],
    }
    for filename, items in patches.items():
        path = src / filename
        text = path.read_text()
        for signature, kind, count, bits_in, bits_out, *variant in items:
            code = 'replay_trace::Scope trace_scope("%s"); replay_trace::event("%s", %s, %s, %s, %s);' % (kind, kind, count, bits_in, bits_out, variant[0] if variant else '0')
            text = inject(text, signature, code)
        path.write_text(text)
    path = src / 'Math/math-functions.cpp'
    text = path.read_text()
    for name in ['sigmoid', 'tanh', 'div', 'lookup_table_exp', 'reciprocal_approximation']:
        text = inject(text, r'void MathFunctions::' + name + r'\(', 'replay_trace::Scope trace_scope("%s");' % name)
    path.write_text(text)

def summarize(path):
    groups = {}
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as f:
        for row in csv.reader(f):
            if len(row) != 8:
                raise ValueError('trace requires eight public metadata columns')
            op, step, stack, kind, count, bits_in, bits_out, variant = row
            key = (op, stack, kind, int(bits_in), int(bits_out), int(variant))
            group = groups.setdefault(key, dict(operator=op, stack=stack, primitive=kind,
                      input_bits=int(bits_in), output_bits=int(bits_out), variant=int(variant),
                      invocations=0, scalar_instances=0, steps=set()))
            group['invocations'] += 1
            group['scalar_instances'] += int(count)
            group['steps'].add(int(step))
    return [dict(g, steps=sorted(g['steps'])) for _, g in sorted(groups.items())]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--sci', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--skip-build', action='store_true')
    p.add_argument('--source-snapshot', type=Path, help='Frozen sci_src.tar.gz emitted by freeze_contract.py')
    args = p.parse_args()
    if args.row < 0 or args.threads < 1 or args.jobs < 1:
        p.error('row must be nonnegative and threads/jobs positive')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    build = out / 'build'; build.mkdir(exist_ok=True)
    src = build / 'sci_src'
    network = build / 'network'
    if not args.skip_build:
        if args.source_snapshot:
            extracted = build / 'frozen_sources'
            with tarfile.open(args.source_snapshot) as archive:
                archive.extractall(extracted)
            shutil.copytree(extracted / 'src', src, dirs_exist_ok=True)
        else:
            shutil.copytree(args.sci / 'src', src, dirs_exist_ok=True)
        shutil.copytree(args.archive / 'FastGRNN/Google-30', network, dirs_exist_ok=True)
        shutil.copy2(HERE / 'trace.h', build / 'trace.h')
        instrument(src)
        generated = network / 'sirnn_fixed.cpp'
        text = generated.read_text()
        text = re.sub(r'(for \(uint32_t i =[^\n]+\{)', r'\1\n replay_trace::step = i;', text)
        # Only generated calls, in their public execution order. Worker threads are joined before the next call.
        pattern = re.compile(r'^(\s*)((?:MatMul|MatAdd|MatSubBroadCastA|MatAddBroadCastB|Sigmoid|TanH|MulCir|ScalarMul|ArgMax|AdjustScaleShl)\([^\n]+\);)', re.M)
        counts = {}
        def label(m):
            function = m[2].split('(')[0]; counts[function] = counts.get(function, 0) + 1
            name = '%s_%02d' % (function, counts[function])
            reset = 'replay_trace::step = 99; ' if name == 'MatMul_05' else ''
            return m[1] + reset + 'replay_trace::op = "' + name + '";\n' + m[1] + m[2]
        text = pattern.sub(label, text)
        text = text.replace('\txt->s_extend((int32_t)30,', '\treplay_trace::op = \"ArgMaxInputSignExtend_01\";\n\txt->s_extend((int32_t)30,')
        generated.write_text(text)
        flags = ['g++', '-O3', '-std=c++17', '-pthread', '-maes', '-msse4.1', '-mavx2', '-mrdseed',
                 '-fopenmp', '-DSCI_OT=1', '-Wno-deprecated-declarations', '-include', str(build / 'trace.h'),
                 '-I' + str(network), '-I' + str(src), '-I' + str(args.sci / 'extern/eigen')]
        paths = [src / name for name in SOURCES] + [network / 'main.cpp', generated]
        commands = []
        objects = []
        for i, path in enumerate(paths):
            obj = build / ('object%d.o' % i); objects.append(str(obj))
            commands.append(flags + ['-c', str(path), '-o', str(obj)])
        def compile_one(item):
            i, cmd = item
            result = subprocess.run(cmd, text=True, capture_output=True)
            (out / ('compile_%d.log' % i)).write_text(result.stdout + result.stderr)
            result.check_returncode()
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            list(executor.map(compile_one, enumerate(commands)))
        libs = args.sci / 'build/install/lib'
        command = flags + objects + ['-o', str(build / 'google30-traced'), '-Wl,--start-group']
        command += [str(libs / ('libSCI-' + name + '.a')) for name in ['OT', 'LinearOT', 'GC', 'Math', 'BuildingBlocks', 'FloatingPoint']]
        command += ['-Wl,--end-group', '-lssl', '-lcrypto', '-lgmp']
        result = subprocess.run(command, text=True, capture_output=True)
        (out / 'link.log').write_text(result.stdout + result.stderr); result.check_returncode()
        (out / 'build_commands.json').write_text(json.dumps(commands + [command], indent=2) + '\n')
    input_dir = out / 'input'; input_dir.mkdir(exist_ok=True)
    for name in ['X.csv', 'Y.csv']:
        with (args.runtime / 'SeeDot/temp/Predictor/input' / name).open() as source:
            line = next((line for i, line in enumerate(source) if i == args.row), None)
        if line is None: raise ValueError('sample row missing')
        (input_dir / name).write_text(line)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    def reap_party(process):
        if process.poll() is None:
            process.kill()
        process.wait()

    with ExitStack() as parties:
        processes = []
        for party in [1, 2]:
            env = dict(os.environ, GOOGLE30_TRACE=str(out / ('party%d_trace.csv' % party)))
            command = [str(build / 'google30-traced'), 'r=%d' % party, 'port=%d' % port,
                       'nt=%d' % args.threads, 'inputDir=' + str(input_dir) + '/']
            log = parties.enter_context((out / ('party%d.log' % party)).open('w'))
            process = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
            parties.callback(reap_party, process)
            processes.append(process)
            if party == 1: time.sleep(0.3)
        for process in processes:
            if process.wait(timeout=600): raise RuntimeError('SCI process failed; inspect party log')
    traces = []
    for party in [1, 2]:
        path = out / ('party%d_trace.csv' % party)
        groups = summarize(path); traces.append(groups)
        with path.open('rb') as source, gzip.open(str(path) + '.gz', 'wb') as target:
            shutil.copyfileobj(source, target)
        path.unlink()
    if traces[0] != traces[1]:
        raise ValueError('public call traces disagree across parties')
    (out / 'primitive_counts.json').write_text(json.dumps(traces[0], indent=2) + '\n')
    flat = [{k:v for k,v in g.items() if k != 'steps'} for g in traces[0]]
    with (out / 'primitive_counts.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    predictions = []
    for party in [1, 2]:
        log = (out / ('party%d.log' % party)).read_text()
        predictions.append(re.findall(r'Predicted Label:\s*(-?\d+)\s*,\s*Actual Label:\s*(-?\d+)', log))
    historical = {int(r['test_row_index']): int(r['sirnn_prediction'])
                  for r in csv.DictReader((args.archive / 'secure_samples.csv').open())}
    if len(predictions[1]) != 1:
        raise ValueError('expected exactly one client output')
    if args.row in historical and int(predictions[1][0][0]) != historical[args.row]:
        raise ValueError('instrumented prediction differs from historical SCI')
    for op in ['Sigmoid_01', 'TanH_01']:
        steps = set(step for g in traces[0] if g['operator'] == op for step in g['steps'])
        if steps != set(range(99)):
            raise ValueError('trace does not cover all 99 steps: ' + op)
    summary = dict(row=args.row, threads=args.threads, parties_succeeded=2, predictions=predictions,
                   public_traces_equal=True, metric_scope='instrumented correctness/call-count run; timings are not performance evidence',
                   counts_scope='one full 99-step inference including classification; one party, never summed over both',
                   primitives={kind:dict(invocations=sum(g['invocations'] for g in traces[0] if g['primitive']==kind),
                                         scalar_instances=sum(g['scalar_instances'] for g in traces[0] if g['primitive']==kind),
                                         input_bits=sorted(set(g['input_bits'] for g in traces[0] if g['primitive']==kind)))
                               for kind in sorted(set(g['primitive'] for g in traces[0]))})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); print(json.dumps(summary))

if __name__ == '__main__': main()
