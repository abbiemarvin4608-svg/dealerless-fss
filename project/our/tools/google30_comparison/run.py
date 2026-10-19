#!/usr/bin/env python3
"""Run the two frozen backends through one shared all-channel relay.

This runner is deliberately protocol-agnostic: command templates are supplied
by the build manifest, while both parties use the same ports, CPU policy,
query count and timeout.  It records failed/censored runs and does not turn
those into zero-valued measurements.
"""
import argparse
import hashlib
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
try:
    from .metrics import combine
except ImportError:
    sys.path.insert(0, str(HERE))
    from metrics import combine


CHANNEL_OFFSETS = (0, 3, 20, 23, 50, 70, 100, 120, 200)


def stop(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def reserve_ports():
    for base in range(18000, 48000, 137):
        sockets = []
        try:
            for offset in CHANNEL_OFFSETS + tuple(1000 + x for x in CHANNEL_OFFSETS):
                sock = socket.socket()
                sock.bind(('127.0.0.1', base + offset))
                sockets.append(sock)
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError('could not reserve shared relay port range')


def _format_command(template, role, base, metrics, owned, start, count, queries):
    values = {'role': role, 'port': base, 'metrics': str(metrics), 'owned': str(owned),
              'start': start, 'count': count, 'queries': queries}
    return [str(part).format(**values) for part in template]


def _bridge_from_log(path):
    matches = re.findall(
        r'BRIDGE\s+calls=(\d+)\s+elements=(\d+)\s+segment_keys=(\d+)\s+'
        r'setup_ms=([0-9.eE+-]+)\s+offline_ms=([0-9.eE+-]+)\s+online_ms=([0-9.eE+-]+)\s+'
        r'sent_bytes=(\d+)\s+received_bytes=(\d+)', path.read_text())
    if not matches:
        return {}
    calls, elements, keys, setup, offline, online, sent, received = matches[-1]
    return {'calls': int(calls), 'elements': int(elements), 'segment_keys': int(keys),
            'setup_ms': float(setup), 'offline_ms': float(offline), 'online_ms': float(online),
            'sent_bytes': int(sent), 'received_bytes': int(received),
            'setup_sent_bytes': None, 'setup_received_bytes': None,
            'offline_sent_bytes': None, 'offline_received_bytes': None,
            'online_sent_bytes': None, 'online_received_bytes': None}


def _execution_from_log(path):
    matches = re.findall(r'EXECUTION\s+setup_seconds=([0-9.eE+-]+)\s+'
                         r'compute_and_output_seconds=([0-9.eE+-]+)\s+'
                         r'total_seconds=([0-9.eE+-]+)\s+peak_rss_kib=(\d+)',
                         path.read_text())
    if not matches:
        return None
    setup, compute, total, rss = matches[-1]
    return {'setup_seconds': float(setup), 'compute_seconds': float(compute),
            'total_seconds': float(total), 'peak_rss_kib': int(rss)}


def _digest(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _backend_identity(config):
    binary = None
    commands = config.get('commands', {})
    if commands.get('1'):
        binary = commands['1'][0]
    manifest_path = config.get('runtime_manifest')
    if manifest_path is None and binary:
        manifest_path = str(Path(binary).resolve().parent / 'runtime_manifest.json')
    manifest = None
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text())
    payload = {
        'backend': config.get('backend'), 'track': config.get('track', 'same_graph'),
        'nseg': config.get('nseg'), 'binary_sha256': _digest(binary),
        'runtime_source_identity': (manifest or {}).get('source_identity'),
        'generated_sha256': (manifest or {}).get('generated_sha256'),
        'contract_sha256': _digest(config.get('contract_path')),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _prediction_lines(path):
    return [(int(index), int(prediction)) for index, prediction in re.findall(
        r'^PRED\s+(\d+)\s+(-?\d+)$', path.read_text(), re.MULTILINE)]


def _verify_predictions(log, config, start, count, queries):
    observed = _prediction_lines(log)
    expected_count = count * queries
    if len(observed) != expected_count:
        raise ValueError('prediction count mismatch: expected %d, got %d' %
                         (expected_count, len(observed)))
    expected_indices = list(range(start, start + count)) * queries
    if [index for index, _ in observed] != expected_indices:
        raise ValueError('prediction indices are not one ordered block per query')
    oracle_path = config.get('oracle_predictions') or config.get('oracle_path')
    expected = None
    if oracle_path:
        try:
            import numpy as np
            expected = np.load(oracle_path, allow_pickle=False).astype(int).tolist()
        except Exception as error:
            raise ValueError('cannot load oracle predictions: %s' % error)
    elif config.get('expected_predictions') is not None:
        expected = list(config['expected_predictions'])
    elif config.get('expected_prediction') is not None:
        expected = [int(config['expected_prediction'])] * (start + count)
    if expected is None:
        raise ValueError('prediction oracle is required for a complete run')
    for query in range(queries):
        for offset, (_, prediction) in enumerate(observed[query * count:(query + 1) * count]):
            index = start + offset
            if index >= len(expected) or prediction != int(expected[index]):
                raise ValueError('oracle prediction mismatch at sample %d' % index)
    return {'oracle_path': str(oracle_path) if oracle_path else None,
            'oracle_sha256': _digest(oracle_path) if oracle_path else None,
            'samples': count, 'queries': queries, 'full_match': True}


def run_pair(config, output, network, run_kind='repeat'):
    """Execute one two-party run and save a self-contained ``result.json``."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = reserve_ports()
    mappings = ','.join('%d:%d' % (base + 1000 + offset, base + offset)
                        for offset in CHANNEL_OFFSETS)
    relay = [sys.executable, str(HERE.parent / 'google30_benchmark' / 'link_proxy.py'),
             '--ports', mappings, '--rtt-ms', str(network['rtt_ms']),
             '--mbps', str(network['mbps'])]
    if len(config.get('cpus', [])) >= 3 and config['cpus'][2] is not None:
        relay = ['taskset', '-c', str(config['cpus'][2])] + relay
    cpus = config.get('cpus', [0, 2, 4])
    timeout = int(network.get('timeout_seconds', config.get('timeout_seconds', 7200)))
    processes = []
    commands = []
    started = time.monotonic()
    status, reason = 'complete', None
    with ExitStack() as stack:
        relay_log = stack.enter_context((output / 'relay.log').open('w'))
        relay_proc = subprocess.Popen(relay, stdout=subprocess.PIPE, stderr=relay_log,
                                      text=True)
        stack.callback(stop, relay_proc)
        if relay_proc.stdout.readline().strip() != 'READY':
            raise RuntimeError('relay did not start')
        relay_proc.stdout.close()
        for role in (1, 2):
            metrics = output / ('party%d_metrics.json' % role)
            owned = Path(config['server_model'] if role == 1 else config['client_input'])
            template = config['commands'][str(role)]
            query_count = int(network.get('queries', config.get('queries', 1)))
            cmd = _format_command(template, role, base, metrics, owned,
                                   config.get('start', 0), config.get('count', 1),
                                   query_count)
            cpu = cpus[role - 1]
            if cpu is not None:
                cmd = ['taskset', '-c', str(cpu)] + cmd
            commands.append(cmd)
            env = dict(os.environ)
            env.update(config.get('environment', {}))
            env['GOOGLE30_METRICS'] = str(metrics)
            # The SCI source reads SWEEP_CONNECT_OFFSET in the client-side
            # connector shim. Keep the historical variable for old binaries,
            # but set the actual one used by the current bridge.
            env['GOOGLE30_PROXY_OFFSET'] = str(config.get('proxy_offset', 1000))
            env['SWEEP_CONNECT_OFFSET'] = str(config.get('proxy_offset', 1000))
            # The FSS bridge opens its own TCP socket.  Its client connection
            # must use the same all-channel port offset as SCI.
            env['SEGFSS_SCI_PORT'] = str(base + 20)
            env.setdefault('SEGFSS_SCI_HOST', '127.0.0.1')
            offset_library = config.get('connect_offset_library')
            if offset_library:
                env['LD_PRELOAD'] = str(offset_library) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
            log = stack.enter_context((output / ('party%d.log' % role)).open('w'))
            def limit():
                resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
            processes.append(subprocess.Popen(cmd, env=env, stdout=log,
                                              stderr=subprocess.STDOUT,
                                              preexec_fn=limit))
        deadline = time.monotonic() + timeout
        for process in processes:
            try:
                code = process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                status, reason = 'censored', 'predeclared wall timeout %ss' % timeout
                break
            if code:
                status, reason = 'failed', 'party exit %d' % code
                break
        if status != 'complete':
            for process in processes:
                stop(process)
        exits = [process.poll() for process in processes]
    query_count = int(network.get('queries', config.get('queries', 1)))
    backend_identity, identity_payload = _backend_identity(config)
    result = {'schema_version': 'google30-comparison-run-v1', 'status': status,
              'reason': reason, 'run_kind': run_kind, 'backend': config['backend'],
              'track': config.get('track', 'same_graph'), 'config_id': config.get('config_id'),
              'network': network, 'queries': query_count, 'query_count': query_count,
              'batch': config.get('count', 1), 'start': config.get('start', 0),
              'nseg': config.get('nseg'), 'comparison_signature': config.get('comparison_signature'),
              'comparison_group_id': config.get('comparison_group_id', config.get('comparison_signature')),
              'backend_identity': backend_identity,
              'backend_identity_payload': identity_payload,
              'quality_backend_identity': config.get('quality_backend_identity'),
              'observed_wall_seconds': time.monotonic() - started,
              'commands': commands, 'relay_command': relay, 'party_exit_codes': exits}
    if status == 'complete':
        metrics = [_read_json(output / ('party%d_metrics.json' % role)) for role in (1, 2)]
        for role, metric in zip((1, 2), metrics):
            bridge = _bridge_from_log(output / ('party%d.log' % role))
            if bridge:
                metric['bridge'] = bridge
            execution = _execution_from_log(output / ('party%d.log' % role))
            if execution:
                metric.setdefault('total_seconds', execution['total_seconds'])
                metric.setdefault('peak_rss_kib', execution['peak_rss_kib'])
        result['prediction_verification'] = _verify_predictions(
            output / 'party2.log', config, config.get('start', 0),
            config.get('count', 1), query_count)
        result.update(combine(metrics))
    else:
        result.update({'total_seconds': None, 'link_bytes': None,
                       'dependent_rounds': None, 'offline_seconds': None,
                       'online_seconds': None, 'peak_rss_kib_per_party': None})
    (output / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def _read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--network', action='append')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = _read_json(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [p for p in config['networks'] if not args.network or p['name'] in args.network]
    all_results = []
    for network in selected:
        repeats = int(network.get('repeats', config.get('repeats', 3)))
        tags = ['smoke'] if args.smoke else ['warmup'] + ['repeat%d' % i for i in range(1, repeats + 1)]
        for tag in tags:
            query_count = int(network.get('queries', config.get('queries', 1)))
            kind = 'session' if query_count > 1 else ('warmup' if tag == 'warmup' else 'repeat')
            all_results.append(run_pair(config, args.output / network['name'] / tag,
                                        network, kind))
    (args.output / 'manifest.json').write_text(json.dumps({'config': str(args.config),
        'results': all_results}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
