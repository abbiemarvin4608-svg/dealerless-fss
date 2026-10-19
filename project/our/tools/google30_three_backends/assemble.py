#!/usr/bin/env python3
"""Join measured, quality-qualified backends without filling missing secure evidence."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google30_comparison.schema import rankable_repeat_group

BACKENDS = ('SCI', 'SCI_ours', 'MP-SPDZ semi2k')


def _distribution(values):
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values), mean=statistics.mean(values),
                stdev=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values))


def make_table(sci_rows, secure_quality, mp_quality, mp_rows):
    rows, comparisons = [], []
    networks = sorted({row['network'] for row in sci_rows + mp_rows}) or ['lan', 'wan_2ms']
    for network in networks:
        for backend in BACKENDS:
            is_mp = backend == 'MP-SPDZ semi2k'
            quality = mp_quality if is_mp else secure_quality
            observed = [row for row in (mp_rows if is_mp else sci_rows)
                if row.get('backend') == backend and row.get('network') == network]
            full = (quality.get('complete') is True and
                    quality.get('secure_samples' if is_mp else 'samples') == 6798)
            if not is_mp:
                full = (full and quality.get('full_secure_evaluation') is True
                        and quality.get('full_oracle_match') is True
                        and quality.get('full_secure_agreement') is True)
            else:
                full = full and quality.get('integer_predictions_equal') is True and not quality.get('mismatches')
            loss = quality.get('full_loss_pp' if is_mp else 'loss_pp') if full else None
            quality_pass = full and loss is not None and loss <= 1.0
            candidates = [row for row in observed
                if row.get('track') == 'same_graph' and row.get('run_kind') == 'repeat'
                and row.get('query_count') == 1 and row.get('status') == 'complete'
                and row.get('eligible') is True]
            # Separate configurations never become a synthetic three-repeat group.
            groups = {row.get('comparison_group_id') for row in candidates}
            group = next(iter(groups)) if len(groups) == 1 else None
            measurement_pass = rankable_repeat_group(candidates)
            eligible = quality_pass and measurement_pass
            reasons = []
            if not full:
                reasons.append('full 6798-sample secure prediction/oracle validation incomplete')
            elif not quality_pass:
                reasons.append('relative FP32 Top-1 loss exceeds 1.0 pp or is absent')
            if not measurement_pass:
                reasons.append('three eligible N=1 repeats of one configuration on this network are unavailable')
            values = candidates if eligible else []
            stats = {field: _distribution([row[field] for row in values if row.get(field) is not None])
                for field in ('total_seconds', 'first_query_seconds', 'link_bytes',
                              'offline_seconds', 'online_seconds')}
            common = dict(backend=backend, network=network, status='available' if eligible else 'unavailable',
                eligible=eligible, reasons=reasons, threat_model='two-party semi-honest, no trusted dealer',
                private_inputs='server matrices/bias; client input sequence; original zeta/nu public',
                output='final class revealed only to client',
                secure_samples=quality.get('secure_samples' if is_mp else 'samples', 0),
                top1_pct=quality.get('full_top1_pct' if is_mp else 'fixed_top1_pct') if full else None,
                fp32_top1_pct=quality.get('full_fp32_top1_pct' if is_mp else 'fp32_top1_pct'), loss_pp=loss,
                prediction_agreement_pct=quality.get('full_prediction_agreement_pct' if is_mp else 'prediction_agreement_pct') if full else None,
                quality_pass=quality_pass, numerical_error=secure_quality.get('logit_error') if full else None,
                repeated_runs=len(values), total_median_seconds=stats['total_seconds']['median'] if values else None,
                link_median_bytes=stats['link_bytes']['median'] if values else None,
                statistics=stats, offline_seconds=stats['offline_seconds'], online_seconds=stats['online_seconds'],
                phase_note='Full offline/online separation unavailable for interleaved live OT; total includes all costs.',
                dependent_rounds=None, rounds_note='No critical-path counter; party I/O counts are not dependency depth.',
                peak_rss_kib_per_party=[max((row[field] for row in values if row.get(field) is not None), default=None)
                    for field in ('peak_rss_kib_party1', 'peak_rss_kib_party2')],
                comparison_group_id=group,
                measurement_artifacts=[row.get('measurement_artifact') for row in candidates],
                observed_runs=[{key: row.get(key) for key in
                    ('measurement_artifact', 'config_id', 'run_kind', 'query_count', 'status',
                     'reason', 'ineligibility_reasons', 'timeout_seconds', 'memory_limit_mib')}
                    for row in observed],
                persistent_sessions=[{key: row.get(key) for key in
                    ('measurement_artifact', 'query_count', 'first_query_seconds', 'second_query_seconds',
                     'session_seconds_per_query', 'total_seconds', 'link_bytes', 'status')}
                    for row in observed
                    if (row.get('query_count') or 0) > 1 and row.get('run_kind') != 'warmup'],
                contract=secure_quality.get('contract') if full and not is_mp else quality.get('binding'))
            for track in ('same_graph', 'system_candidates'):
                rows.append(dict(common, track=track,
                    configuration_note='frozen independent02 SCI integer graph; nseg4 for SCI_ours' if track == 'same_graph'
                    else 'same tested native configuration; no separate quantization search or globally optimal configuration claim'))
        selected = {row['backend']: row for row in rows if row['network'] == network and row['track'] == 'same_graph'}
        if (selected['SCI']['eligible'] and selected['SCI_ours']['eligible']
                and selected['SCI']['comparison_group_id'] == selected['SCI_ours']['comparison_group_id']):
            comparisons.append(dict(network=network, track='same_graph',
                ours_over_sci_latency_ratio=selected['SCI_ours']['total_median_seconds']/selected['SCI']['total_median_seconds'],
                ours_over_sci_communication_ratio=selected['SCI_ours']['link_median_bytes']/selected['SCI']['link_median_bytes'],
                scope='same graph, N=1, same documented measurement configuration'))
    return dict(schema_version='google30-three-backends-v1', required_secure_samples=6798,
        quality_loss_limit_pp=1.0, rows=rows, two_backend_comparisons=comparisons,
        three_backend_ranking_available=all(row['eligible'] for row in rows)
            and all(len({row['comparison_group_id'] for row in rows if row['network'] == network}) == 1
                    for network in networks),
        context=[
            dict(backend='ABY-64', status='context_only', reason='two-party semi-honest is not itself dealer-assisted; historical 64-bit numerical/control configuration differs'),
            dict(backend='dealer-assisted FSS', status='context_only', reason='trusted preprocessing assumption differs from dealerless main table'),
            dict(backend='CrypTFlow2', status='no_independent_Google30_RNN_result', reason='shares SCI implementation with SiRNN; do not duplicate a SCI result as an independent RNN baseline')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sci-rows', type=Path, required=True)
    parser.add_argument('--secure-quality', type=Path, required=True)
    parser.add_argument('--mp-quality', type=Path, required=True)
    parser.add_argument('--mp-rows', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text())
    result = make_table(load(args.sci_rows), load(args.secure_quality), load(args.mp_quality),
                        load(args.mp_rows) if args.mp_rows else [])
    result['sources'] = {key: str(path.resolve()) for key, path in vars(args).items() if path is not None and key != 'output'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    fields = ['track','network','backend','status','secure_samples','top1_pct','loss_pp',
              'prediction_agreement_pct','total_median_seconds','link_median_bytes','repeated_runs','eligible','reasons']
    with (args.output/'table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['rows'])
    def number(value, digits=3):
        return 'unavailable' if value is None else f'{value:.{digits}f}'
    lines = ['# Google-30 三后端证据表', '',
        '固定质量门槛：相对同一FP32模型的Top-1损失≤1.0 pp；agreement另报。', '',
        '| 轨道 | 网络模型 | 后端 | 安全样本 | Top-1 % | loss pp | agreement % | 首查询总延迟中位数(s) | 通信(B) | 状态 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in result['rows']:
        lines.append('| ' + ' | '.join([
            row['track'], row['network'], row['backend'], str(row['secure_samples'])+'/6798',
            number(row['top1_pct'], 4), number(row['loss_pp'], 4), number(row['prediction_agreement_pct'], 4),
            number(row['total_median_seconds']), number(row['link_median_bytes'], 0), row['status']]) + ' |')
    lines += ['', '`same_graph` 固定原生整数图。`system_candidates` 使用各后端当前已验证配置；本轮配置相同，两个视图共享运行证据，未另做量化搜索，不声称全局最佳。', '',
        'offline/online不可分离项和真实依赖轮数保留null；双方发送量之和仅计算一次。逐次分布、峰值RSS、N=2会话及日志索引见summary.json。', '',
        '完整三后端排名：' + ('可用' if result['three_backend_ranking_available'] else 'unavailable') + '。']
    for comparison in result['two_backend_comparisons']:
        lines.append(f"\n{comparison['network']}同图对照：SCI+ours/SCI总延迟比 {comparison['ours_over_sci_latency_ratio']:.4f}，通信比 {comparison['ours_over_sci_communication_ratio']:.4f}。")
    for backend in BACKENDS:
        reasons = sorted({reason for row in result['rows'] if row['backend'] == backend for reason in row['reasons']})
        if reasons:
            lines.append('\n'+backend+'：'+'; '.join(reasons)+'.')
    (args.output/'TABLE_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': len(result['rows']), 'three_backend_ranking_available': result['three_backend_ranking_available']}))


if __name__ == '__main__':
    main()
