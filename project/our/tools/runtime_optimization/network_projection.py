#!/usr/bin/env python3
"""Project communication time from qualified phase bytes and causal RTT rounds."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


NETWORKS = {
    'LAN': dict(bandwidth_bits_per_second=1000000000, rtt_seconds=0.0005),
    'WAN': dict(bandwidth_bits_per_second=400000000, rtt_seconds=0.004),
}
BACKENDS = ('SCI_preprocessed', 'SCI_Beaver_segmented')
PAPER_LUT_BACKENDS = ('SCI_preprocessed', 'SCI_Beaver_segmented_PaperLUT')
PHASES = ('offline', 'online')
CONVENTION = 'two-party exchange; causal send-flight clocks'
LABELS = {'SCI_preprocessed': 'SCI+SiRNN 预处理',
          'SCI_Beaver_segmented': 'SCI+ours 完整方法',
          'SCI_Beaver_segmented_PaperLUT': 'SCI+ours Segmented DPF PaperLUT'}


def distribution(values):
    return dict(values=values, median=statistics.median(values), min=min(values),
                max=max(values), pstdev=statistics.pstdev(values))


def validate(evidence, backends=BACKENDS):
    if (type(evidence.get('schema')) is not int or evidence.get('schema') != 1 or evidence.get('eligible') is not True
            or evidence.get('cohort') != 'a2-parallel' or evidence.get('convention') != CONVENTION):
        raise ValueError('requires qualified schema 1 a2-parallel causal round evidence')
    rows = evidence.get('rows', [])
    if len(rows) != 2 or {row.get('backend') for row in rows} != set(backends):
        raise ValueError('requires exactly one row for each preprocessing backend')
    if len({row.get('id') for row in rows}) != 2:
        raise ValueError('row ids must be unique')
    for row in rows:
        queries = row.get('queries', [])
        if row.get('eligible') is not True:
            raise ValueError('unqualified row: ' + str(row.get('id')))
        if (len(queries) != 10 or any(type(query.get('query')) is not int for query in queries)
                or {query.get('query') for query in queries} != set(range(10))):
            raise ValueError('requires exact query coverage 0..9: ' + str(row.get('id')))
        for query in queries:
            for phase in PHASES:
                record = query.get('phases', {}).get(phase, {})
                if record.get('trace_coverage') is not True:
                    raise ValueError('missing trace coverage: %s query %s %s' % (
                        row.get('id'), query.get('query'), phase))
                if type(record.get('rounds')) is not int or record['rounds'] < 0:
                    raise ValueError('rounds must be nonnegative integers')
                for field in ('sent_bytes', 'received_bytes'):
                    values = record.get(field)
                    if (not isinstance(values, list) or len(values) != 2
                            or any(type(value) is not int or value < 0 for value in values)):
                        raise ValueError(field + ' must contain two nonnegative integer counters')
                if record['sent_bytes'] != list(reversed(record['received_bytes'])):
                    raise ValueError('party send/receive byte counters do not match')


def project(evidence, backends=BACKENDS):
    """Compute each query's complete network sum before taking any median."""
    validate(evidence, backends=backends)
    result = dict(schema=1, kind='communication-time-model', eligible=True,
                  cohort=evidence['cohort'], round_convention=evidence['convention'],
                  backends=list(backends),
                  networks=NETWORKS, local_computation_included=False, measured_lan_wan=False,
                  transmission_formula='8 * sum(sent_bytes) / bandwidth_bits_per_second',
                  propagation_formula='rtt_seconds * rounds',
                  aggregation='Per-query component sums, then median across 10 fresh queries',
                  rows=[], comparisons=[])
    for row in sorted(evidence['rows'], key=lambda item: backends.index(item['backend'])):
        output = dict(id=row['id'], backend=row['backend'], label=LABELS[row['backend']],
                      queries=[], summary={})
        for query in sorted(row['queries'], key=lambda item: item['query']):
            item = dict(query=query['query'], phases={}, offline_plus_online_seconds={})
            for phase in PHASES:
                source = query['phases'][phase]
                projected = dict(rounds=source['rounds'], bytes=sum(source['sent_bytes']),
                                 sent_bytes=source['sent_bytes'], received_bytes=source['received_bytes'],
                                 networks={})
                for name, network in NETWORKS.items():
                    transmission = 8 * projected['bytes'] / network['bandwidth_bits_per_second']
                    propagation = source['rounds'] * network['rtt_seconds']
                    projected['networks'][name] = dict(transmission_seconds=transmission,
                        propagation_seconds=propagation, communication_seconds=transmission+propagation)
                item['phases'][phase] = projected
            for name in NETWORKS:
                item['offline_plus_online_seconds'][name] = sum(
                    item['phases'][phase]['networks'][name]['communication_seconds'] for phase in PHASES)
            output['queries'].append(item)
        for phase in PHASES:
            phases = [query['phases'][phase] for query in output['queries']]
            summary = {field: distribution([item[field] for item in phases]) for field in ('bytes', 'rounds')}
            summary['networks'] = {name: {field: distribution([item['networks'][name][field] for item in phases])
                for field in ('transmission_seconds', 'propagation_seconds', 'communication_seconds')}
                for name in NETWORKS}
            output['summary'][phase] = summary
        output['summary']['offline_plus_online_seconds'] = {name: distribution([
            query['offline_plus_online_seconds'][name] for query in output['queries']]) for name in NETWORKS}
        result['rows'].append(output)
    by_backend = {row['backend']: row for row in result['rows']}
    baseline, ours = (by_backend[backend] for backend in backends)
    for name in NETWORKS:
        comparison = dict(network=name, baseline=backends[0], comparison=backends[1], phases={})
        for phase in PHASES:
            left, right = (row['summary'][phase]['networks'][name]['communication_seconds']['median']
                           for row in (baseline, ours))
            comparison['phases'][phase] = dict(baseline_seconds=left, comparison_seconds=right,
                speedup=left/right if right else None, ratio=right/left if left else None,
                reduction=1-right/left if left else None)
        result['comparisons'].append(comparison)
    return result


def chinese_report(result):
    lines = ['# a2 并行：实测通信轮数与 LAN/WAN 通信时间模型', '',
        '输入为 batch=128、FastGRNN 单时间步的两方阶段通信与轮数记录。'
        '比较方法按结果中的 baseline/comparison backend 解释。', '',
        '以下为通信时间模型，LAN/WAN 没有实际网络整形测量。'
        '不加入本地计算、材料生成或加载时间，也不把 localhost 实测墙钟时间当作纯计算加上去。', '',
        '| 环境 | 带宽 bit/s | RTT s |', '| --- | ---: | ---: |']
    lines += ['| %s | %d | %.4f |' % (name, parameters['bandwidth_bits_per_second'],
        parameters['rtt_seconds']) for name, parameters in result['networks'].items()]
    lines += ['', '```text', '传输时间（s） = 8 × 通信字节数 / 带宽（bit/s）',
        '传播时间（s） = RTT（s） × 轮数', '通信时间（s） = 传输时间 + 传播时间', '```', '',
        '通信字节数为双方发送字节之和，每条消息计一次；按给定模型在单一带宽上串行传输。'
        '一轮按两方一次互发交换计，轮数来自因果通信记录，a2 并行通道使用整体关键路径，'
        '不将各分支轮数相加。每查询先计算完整通信时间，再对十个新鲜查询取中位数；'
        '原始分布和逐查询各项保存在 NETWORK_MODEL.json。setup、input/ready、output 和 teardown 不计入这两个阶段。', '',
        '| 方法 | 阶段 | 通信量 bytes 中位数 | 通信量 MiB | 轮数中位数 [min,max] |',
        '| --- | --- | ---: | ---: | --- |']
    for row in result['rows']:
        for phase in PHASES:
            summary = row['summary'][phase]
            rounds = summary['rounds']
            lines += ['| %s | %s | %.0f | %.6f | %g [%g,%g] |' % (row['label'], phase,
                summary['bytes']['median'], summary['bytes']['median']/2**20,
                rounds['median'], rounds['min'], rounds['max'])]
    for name in result['networks']:
        lines += ['', '## '+name, '',
            '| 方法 | 阶段 | 传输 s | 传播 s | 通信时间 s [min,max] |',
            '| --- | --- | ---: | ---: | --- |']
        for row in result['rows']:
            for phase in PHASES:
                summary = row['summary'][phase]['networks'][name]
                total = summary['communication_seconds']
                lines += ['| %s | %s | %.9f | %.9f | %.9f [%.9f,%.9f] |' % (
                    row['label'], phase, summary['transmission_seconds']['median'],
                    summary['propagation_seconds']['median'], total['median'], total['min'], total['max'])]
        comparison = next(row for row in result['comparisons'] if row['network'] == name)
        online = comparison['phases']['online']
        if online['speedup'] is not None and online['reduction'] is not None:
            lines += ['', '在线通信模型加速比 = %.9f / %.9f = %.6fx；在线通信时间变化 %+.6f%%。' % (
                online['baseline_seconds'], online['comparison_seconds'], online['speedup'], -100*online['reduction'])]
        else:
            lines += ['', '在线通信模型存在零成本阶段，在线加速比不可定义。']
        lines += ['', '| 方法 | 离线+在线通信模型 s（逐查询求和后中位数） |', '| --- | ---: |']
        lines += ['| %s | %.9f |' % (row['label'], row['summary']['offline_plus_online_seconds'][name]['median'])
                  for row in result['rows']]
    lines += ['', '加速比仅指传输与传播两项之和，不能解释为实测 LAN/WAN 总查询时延。'
        '每方 1.5 GiB 预算与输入前实际材料驻留由本轮资格检查验证。SDCF-09 继续延期。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round-evidence', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--backends', default=','.join(BACKENDS),
                        help='exactly two backend ids, in baseline/comparison order')
    args = parser.parse_args()
    backends = tuple(value.strip() for value in args.backends.split(',') if value.strip())
    if len(backends) != 2 or any(value not in LABELS for value in backends) or len(set(backends)) != 2:
        print('network projection rejected: --backends must name two supported distinct backends', file=sys.stderr)
        return 1
    try:
        raw = args.round_evidence.read_bytes()
        result = project(json.loads(raw), backends=backends)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('network projection rejected: '+str(error), file=sys.stderr)
        return 1
    result['source'] = dict(round_evidence=str(args.round_evidence.resolve()),
                            sha256=hashlib.sha256(raw).hexdigest())
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory/'NETWORK_MODEL.json').write_text(
        json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    (args.output_directory/'NETWORK_MODEL.zh.md').write_text(chinese_report(result), encoding='utf-8')
    print(json.dumps(dict(eligible=True, rows=len(result['rows']), queries=10,
                         networks=list(result['networks']), output=str(args.output_directory))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
