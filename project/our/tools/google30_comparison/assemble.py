#!/usr/bin/env python3
"""Assemble immutable two-backend measurement records into typed rows.

The assembler never fills missing performance or quality values.  It emits
partial rows with an explicit ineligibility reason so later completion of the
full secure ledger can be added without changing the table schema.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
try:
    from .metrics import combine
    from .schema import result_row, summarize_rows
except ImportError:
    sys.path.insert(0, str(HERE))
    from metrics import combine
    from schema import result_row, summarize_rows


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _quality(path):
    value = _read(path)
    if value is None:
        return None
    return value


def _comparison_for(result, comparison_root):
    explicit = result.get('comparison_artifact')
    if explicit:
        value = _read(explicit)
        if value is not None:
            return value, explicit
    path = Path(comparison_root) if comparison_root else None
    if path:
        candidates = [path / str(result.get('backend', '')) / 'comparison.json',
                      path / 'comparison.json', path / 'summary.json']
        for candidate in candidates:
            value = _read(candidate)
            if value is not None:
                return value, str(candidate)
    return result.get('comparison'), None


def _result_dirs(root):
    root = Path(root)
    return sorted(root.glob('runs/*/*/result.json')) + sorted(root.glob('*/*/result.json'))


def assemble(root, quality_path=None, comparison_root=None, output=None):
    root = Path(root)
    output = Path(output or root)
    quality = _quality(quality_path) if quality_path else None
    rows = []
    seen = set()
    for result_path in _result_dirs(root):
        if result_path in seen:
            continue
        seen.add(result_path)
        run = _read(result_path) or {}
        run['artifact'] = str(result_path.resolve())
        if 'network' not in run:
            run['network'] = {'name': result_path.parent.parent.name}
        if 'run_kind' not in run:
            run['run_kind'] = result_path.parent.name
        # If a complete run has raw per-party files, rebuild its communication
        # aggregate rather than trusting a hand-edited summary.
        party_paths = [result_path.parent / ('party%d_metrics.json' % role) for role in (1, 2)]
        if (run.get('status') == 'complete' and all(path.exists() for path in party_paths)
                and 'fss_link_bytes' not in run):
            run.update(combine([_read(path) for path in party_paths]))
        comparison, comparison_path = _comparison_for(run, comparison_root)
        row = result_row(run, quality, comparison,
                         backend=run.get('backend', result_path.parent.parent.name),
                         track=run.get('track', 'same_graph'),
                         config_id=run.get('config_id'))
        row['comparison_artifact'] = comparison_path
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get('backend')), str(row.get('track')),
                               str(row.get('network')), str(row.get('run_kind')),
                               str(row.get('measurement_artifact'))))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rows.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
    profiles = summarize_rows(rows)
    comparisons = []
    for track, network in sorted({(r['track'], r['network']) for r in rows}):
        pair = [profiles.get('%s/%s/%s' % (backend, track, network), {})
                for backend in ('SCI', 'SCI_ours')]
        if not all(profile.get('rankable') for profile in pair):
            continue
        groups = [profile['configuration_groups'][0]['comparison_group_id'] for profile in pair]
        if groups[0] == groups[1]:
            comparisons.append(dict(track=track, network=network, comparison_group_id=groups[0]))
    (output / 'summary.json').write_text(json.dumps({
        'schema_version': 'google30-comparison-v1',
        'required_samples': 6798,
        'quality_gate': 'relative FP32 Top-1 loss <= 1.0 pp plus full prediction agreement',
        'rows': len(rows),
        'profiles': profiles,
        'rankable_comparisons': comparisons,
        'ranking_status': 'available' if comparisons else 'unavailable',
        'ranking_reason': None if comparisons else
                          'both backends need three distinct eligible repeats with the same resource and graph binding',
    }, indent=2, sort_keys=True) + '\n')
    if rows:
        with (output / 'table.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='measurement root containing runs/<network>/<tag>/result.json')
    parser.add_argument('--quality', type=Path)
    parser.add_argument('--comparison-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = assemble(args.root, args.quality, args.comparison_root, args.output)
    print(json.dumps({'rows': len(rows), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
