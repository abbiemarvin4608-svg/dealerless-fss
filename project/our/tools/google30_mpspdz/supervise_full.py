#!/usr/bin/env python3
"""Run resumable semi2k evaluation slices until the declared target is covered.

Each slice invokes :func:`evaluate` with a finite wall-clock budget.  The
ledger, evaluation manifest, and summary are atomically updated after every
completed batch, so an operator may stop this process between slices without
losing secure predictions or accidentally turning partial coverage into a
quality result.
"""
import argparse
import json
import time
from pathlib import Path

from evaluate_secure import evaluate


def supervise(args):
    output = Path(args.output).resolve()
    state_path = output / 'supervisor_state.json'
    output.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'slices': 0, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'slice_budget_s': args.slice_seconds, 'pause_s': args.pause_seconds}
    if state_path.exists():
        state.update(json.loads(state_path.read_text()))
    state['status'] = 'running'
    slice_count = 0
    while args.max_slices is None or slice_count < args.max_slices:
        if args.stop_file and Path(args.stop_file).exists():
            state['status'] = 'paused_stop_file'
            break
        before = 0
        summary_path = output / 'summary.json'
        if summary_path.exists():
            before = json.loads(summary_path.read_text()).get('secure_samples', 0)
        summary = evaluate(args.artifact, args.input_csv, output, args.spdz, args.oracle_dir,
                           args.batch, args.queries, args.limit, args.timeout, args.memory_mib,
                           tuple(args.cpus), args.slice_seconds, args.stop_file)
        state['slices'] += 1
        slice_count += 1
        state['last_summary'] = summary
        state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        if summary['complete']:
            state['status'] = 'complete'
            state_path.write_text(json.dumps(state, indent=2) + '\n')
            return summary
        if summary.get('mismatches'):
            state['status'] = 'mismatch_stopped'
            state_path.write_text(json.dumps(state, indent=2) + '\n')
            return summary
        if summary.get('secure_samples', 0) <= before:
            state['status'] = 'unavailable_no_progress'
            state_path.write_text(json.dumps(state, indent=2) + '\n')
            return summary
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        if args.pause_seconds:
            time.sleep(args.pause_seconds)
    state['status'] = state.get('status', 'slice_limit') if state.get('status') != 'running' else 'slice_limit'
    state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    return json.loads((output / 'summary.json').read_text()) if (output / 'summary.json').exists() else state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--input-csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spdz', type=Path, default=Path('/home/jiang/MP-SPDZ'))
    p.add_argument('--oracle-dir', type=Path, required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--queries', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--memory-mib', type=int, default=1536)
    p.add_argument('--cpus', default='6,8')
    p.add_argument('--slice-seconds', type=int, default=1200)
    p.add_argument('--pause-seconds', type=int, default=0)
    p.add_argument('--max-slices', type=int)
    p.add_argument('--stop-file', type=Path)
    a = p.parse_args()
    a.cpus = [int(x) for x in a.cpus.split(',')]
    if len(a.cpus) != 2:
        p.error('--cpus needs two CPU IDs')
    print(json.dumps(supervise(a), indent=2))
