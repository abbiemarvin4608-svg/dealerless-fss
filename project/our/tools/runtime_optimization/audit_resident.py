#!/usr/bin/env python3
"""Qualify fresh SCI/FSS measurements with all query material resident before input."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import audit_preprocessed as base


PREFETCH_BYTES = 536870912
MEMORY_BYTES = 1610612736
FROZEN_BINARY = '32f796a4bf50217e03fe523107e62aedef59b4691f73b176d801b5eb2026aeee'
FROZEN_BRIDGE = 'b9d753d1df474f23c2a32484dc4bd3e65a242d04478caf5c10518ec07d476f0a'
RESULTS = base.REPO / 'project/our/results/sci_sirnn_resident_20261012'
LABELS = {'SCI': 'SCI+SiRNN 默认', 'SCI_preprocessed': 'SCI+SiRNN 预处理',
          'SCI_ours': '仅转换替换（消融）', 'SCI_Beaver_segmented': 'SCI+ours（完整方法）'}


def resident_party_checks(log, party, config, queries):
    """Require the complete prepared prefix, rather than a successful partial prefetch."""
    uses_fss = config['backend'] in ('SCI_ours', 'SCI_Beaver_segmented')
    contexts = config['graph_branches']
    budgets = party['conversion_prefetch_budgets']
    details = party['conversion_details']
    calls = party['conversion_calls']
    ready, prefixes = {}, defaultdict(list)
    query = 0
    for position, line in enumerate(log.splitlines()):
        if line.startswith('CONVERSION_PREFETCH '):
            prefixes[query].append((position, json.loads(line.split(' ', 1)[1])))
        elif line.startswith('MATERIAL_READY '):
            row = json.loads(line.split(' ', 1)[1])
            ready.setdefault(row['query'], []).append(position)
        elif line.startswith('METRIC '):
            row = json.loads(line.split(' ', 1)[1])
            if row['phase'] == 'query_total':
                query = row['query'] + 1
    allocations = {row.get('context'): row.get('context_budget_bytes') for row in budgets}
    checks = dict(
        uniform_resident_policy=config.get('prefetch_bytes') == PREFETCH_BYTES
            and config.get('prefetch_mode') == 'prepared_prefix'
            and config.get('memory_limit_bytes_per_party') == MEMORY_BYTES,
        exact_conversion_detail_coverage=base.exact_coverage(details,
            set(base.identities(calls, ('query', 'index', 'slot'))), ('query', 'index', 'slot')),
        no_online_file_loading=all(type(row.get('prefetched')) is int and row['prefetched'] == 1
            and row.get('material_load_ms') == 0 and row.get('prefetch_wait_ms') == 0
            and math.isfinite(row.get('material_preload_ms', float('nan')))
            and row['material_preload_ms'] >= 0 for row in details),
        no_protocol_work_hidden_in_input_ready=all(
            row['sent_bytes'] == row['received_bytes'] == (1 if row['phase'] == 'ready' else 0)
            for row in party['phases'] if row['phase'] in ('input', 'ready')),
        no_generation_after_prepare=(config['backend'] == 'SCI' or all(
            row.get('native_generation_epoch_delta') == row.get('bridge_generation_epoch_delta') == 0
            for row in party['phases'] if row['phase'] in ('input', 'ready', 'online', 'output'))),
        prefix_before_input=True, complete_context_prefixes=True, bounded_resident_contexts=True,
        resident_inventory_consumed_in_order=True,
    )
    evidence = dict(queries=[], budget_bytes=PREFETCH_BYTES, memory_limit_bytes=MEMORY_BYTES,
                    inventory='FSS prepared prefix' if uses_fss else 'native in-memory stocks')
    if not uses_fss:
        checks['no_unrequested_fss_inventory'] = not budgets and not prefixes and not details and not calls
        return checks, evidence
    checks['exact_context_budget_allocation'] = (len(budgets) == contexts
        and set(allocations) == set(range(contexts))
        and all(row.get('contexts') == contexts and row.get('party_budget_bytes') == PREFETCH_BYTES
            and type(row.get('context_budget_bytes')) is int and row['context_budget_bytes'] > 0
            for row in budgets) and sum(allocations.values()) == PREFETCH_BYTES)
    checks['exact_query_prefix_coverage'] = set(prefixes) == set(range(queries))
    by_identity = {(row['query'], row['index'], row['slot']): row for row in calls}
    for query in range(queries):
        grouped = defaultdict(list)
        for row in details:
            if row['query'] == query:
                identity = (row['query'], row['index'], row['slot'])
                grouped[by_identity[identity]['context']].append(row)
        prepared = prefixes.get(query, [])
        positions = ready.get(query, [])
        checks['prefix_before_input'] &= (len(positions) == 1
            and all(position < positions[0] for position, _ in prepared))
        expected = Counter((len(rows), allocations.get(context), rows[0].get('resident_bytes'))
                           for context, rows in grouped.items())
        actual = Counter((row.get('calls'), row.get('budget_bytes'), row.get('resident_bytes'))
                         for _, row in prepared)
        expected_calls = 21 if config['backend'] == 'SCI_Beaver_segmented' else 19
        checks['complete_context_prefixes'] &= (len(prepared) == contexts
            and set(grouped) == set(range(contexts)) and expected == actual
            and sum(len(rows) for rows in grouped.values()) == expected_calls)
        checks['bounded_resident_contexts'] &= all(
            row.get('mode') == 'prepared_prefix' and row.get('charged_phase') == 'offline'
            and type(row.get('resident_bytes')) is int and 0 < row['resident_bytes'] <= row['budget_bytes']
            and type(row.get('address_space_bytes')) is int and 0 < row['address_space_bytes'] < MEMORY_BYTES
            and type(row.get('workspace_reserve_bytes')) is int and row['workspace_reserve_bytes'] > 0
            for _, row in prepared)
        for context, rows in grouped.items():
            live = [row.get('resident_bytes') for row in rows]
            checks['resident_inventory_consumed_in_order'] &= (all(type(value) is int
                and 0 < value <= allocations.get(context, 0) for value in live)
                and all(left > right for left, right in zip(live, live[1:]))
                and all(row.get('prefetch_budget_bytes') == allocations.get(context) for row in rows))
        evidence['queries'].append(dict(query=query, calls=sum(len(rows) for rows in grouped.values()),
            contexts=len(prepared), resident_bytes=sum(row.get('resident_bytes', 0) for _, row in prepared),
            preloaded_ms=sum(row.get('material_preload_ms', 0) for rows in grouped.values() for row in rows)))
    return checks, evidence


def audit_row(directory, declaration, diagnostic, contract, reference_dir):
    # Keep the original evidence checks; only replace its intentionally frozen
    # prefetch-disabled policy with this experiment's declared resident policy.
    row, config = base.audit_row(directory, declaration, diagnostic, contract, reference_dir)
    if config is None:
        return row, config
    row['checks']['frozen_optimizer_defaults'] = (config.get('beaver_mapped_layout') is False
        and config.get('aes_offline') is False and config.get('prefetch_bytes') == PREFETCH_BYTES
        and config.get('prefetch_mode') == 'prepared_prefix')
    row['checks']['declared_prefetch_budget_matches'] = base.arg(
        declaration['command'], '--prefetch-bytes') == str(PREFETCH_BYTES)
    row['checks']['same_frozen_binary'] = config['binary_sha256'] == FROZEN_BINARY
    output = directory / declaration['id']
    parties = base.read(output / 'parties.json')
    row['evidence']['resident_materials_per_party'] = []
    for role, (party, party_audit) in enumerate(zip(parties, row['parties']), 1):
        checks, evidence = resident_party_checks((output / ('party%d.log' % role)).read_text(),
            party, config, 2 if diagnostic else 10)
        party_audit['checks'].update(checks)
        party_audit['eligible'] = all(party_audit['checks'].values())
        row['evidence']['resident_materials_per_party'].append(evidence)
    row['eligible'] = all(row['checks'].values()) and all(party['eligible'] for party in row['parties'])
    if not row['eligible']:
        row['measurements'] = None
    return row, config


def audit_matrix(formal_dir, diagnostic_dir, reference_dir, contract_path):
    contract = base.load_contract_module(contract_path)
    report = dict(scope='Batch128 single FastGRNN timestep; material resident before input; SDCF-09 deferred',
        formal=[], diagnostic=[], cohorts=[], artifact_checks={}, contract_checker=str(contract_path))
    configurations, declarations, manifests = [], {}, {}
    for kind, directory, count in (('diagnostic', diagnostic_dir, 16), ('formal', formal_dir, 8)):
        manifest = base.read(directory / 'manifest.json')
        manifests[kind] = manifest
        rows = manifest.get('command_rows', manifest['rows'])
        declarations[kind] = rows
        expected = {(backend, cohort, hidden) for backend in base.BACKENDS
            for cohort in ('a4-sequential', 'a2-parallel')
            for hidden in (('default', 'nonzero') if kind == 'diagnostic' else (None,))}
        actual = [(row['backend'], row['cohort'], row.get('hidden')) for row in rows]
        summary = base.read(directory / 'summary.json')['rows']
        report['artifact_checks'][kind] = dict(
            exact_declaration_set=len(rows) == count and len(set(actual)) == count and set(actual) == expected,
            unique_ids=len({row['id'] for row in rows}) == count,
            complete_summary=len(summary) == count and all(row['status'] == 'complete' for row in summary),
            exact_summary_ids={row['id'] for row in summary} == {row['id'] for row in rows},
            no_extra_output_rows={path.name for path in directory.iterdir() if path.is_dir()} == {row['id'] for row in rows},
            actual_binary_frozen=base.sha(Path(manifest['binary'])) == manifest['binary_sha256'] == FROZEN_BINARY,
            actual_bridge_frozen=base.sha(Path(manifest['bridge'])) == manifest['bridge_sha256'] == FROZEN_BRIDGE)
        for declaration in rows:
            try:
                row, config = audit_row(directory, declaration, kind == 'diagnostic', contract, reference_dir)
                if config:
                    sources = config['runtime_manifest']['sources']
                    bridges = [(path, value) for path, value in sources.items() if Path(path).name == 'libsegfss_sci_bridge.so']
                    row['checks']['frozen_manifest_identity'] = (config['binary'] == manifest['binary']
                        and config['binary_sha256'] == manifest['binary_sha256']
                        and bridges == [(manifest['bridge'], manifest['bridge_sha256'])])
                    row['checks']['frozen_generated_source'] = base.sha(Path(config['runtime_manifest']['generated'])) == config['runtime_manifest']['generated_sha256']
                    row['checks']['hidden_state_matches_declaration'] = (config['hidden_file'] == base.arg(declaration['command'], '--hidden')
                        and ((config['hidden_file'] is None) if declaration.get('hidden') != 'nonzero' else
                            base.sha(Path(config['hidden_file'])) == '96d0e2436ca32b5fd5d333ba75866540a2eb03f8e7ee0a2f2fd0a5fcbf31f400'))
                    row['eligible'] &= all(row['checks'].values())
                    configurations.append(config)
            except Exception as error:
                row = dict(id=declaration['id'], backend=declaration['backend'], cohort=declaration['cohort'],
                    hidden=declaration.get('hidden'), status='invalid_evidence', reason=str(error), eligible=False, measurements=None)
            report[kind].append(row)
    for kind in ('formal', 'diagnostic'):
        audits = {row['id']: row for row in report[kind]}
        grouped = defaultdict(list)
        for declaration in declarations[kind]:
            grouped[(declaration['cohort'], declaration.get('hidden'))].append((declaration, audits[declaration['id']]))
        for (cohort, hidden), rows in grouped.items():
            configs = [row['evidence']['shared_configuration'] for _, row in rows if 'evidence' in row]
            checks = dict(four_backends=len(rows) == 4 and {d['backend'] for d, _ in rows} == base.BACKENDS,
                shared_command_and_environment=all(base.normalized_command(d['command']) == base.normalized_command(rows[0][0]['command'])
                    and d.get('environment') == rows[0][0].get('environment') for d, _ in rows),
                shared_observed_configuration=len(configs) == 4 and all(config == configs[0] for config in configs),
                all_rows_qualified=all(row['eligible'] for _, row in rows))
            report['cohorts'].append(dict(kind=kind, cohort=cohort, hidden=hidden, checks=checks, eligible=all(checks.values())))
    for row in report['formal']:
        row.setdefault('checks', {})
        row['checks']['matching_default_and_nonzero_diagnostics'] = base.matching_diagnostics(row, report['diagnostic'])
        formal = next(d for d in declarations['formal'] if d['id'] == row['id'])
        matches = [d for d in declarations['diagnostic'] if d['backend'] == row['backend'] and d['cohort'] == row['cohort']]
        row['checks']['formal_diagnostic_command_contract'] = len(matches) == 2 and all(
            base.diagnostic_command(d['command']) == base.diagnostic_command(formal['command'])
            and d.get('environment') == formal.get('environment') for d in matches)
        row['eligible'] &= all(row['checks'].values())
    artifacts = report['artifact_checks']
    artifacts['same_declared_binary_and_bridge'] = all(manifests['formal'][key] == manifests['diagnostic'][key]
        for key in ('binary', 'binary_sha256', 'bridge', 'bridge_sha256'))
    artifacts['single_runtime_manifest'] = len(configurations) == 24 and all(
        config['runtime_manifest'] == configurations[0]['runtime_manifest'] for config in configurations)
    if configurations:
        report['public_scalar_source'] = base.public_scalar_source(Path(configurations[0]['binary']))
        artifacts['public_scalar_contract'] = all(report['public_scalar_source'][key] is True for key in (
            'native_scalar_contract_matches_original', 'beaver_scalar_contract_matches_original'))
    report['complete'] = len(report['formal']) == 8 and len(report['diagnostic']) == 16 and all(
        row['status'] == 'complete' for row in report['formal'] + report['diagnostic'])
    report['eligible'] = report['complete'] and all(row['eligible'] for row in report['formal'] + report['diagnostic']) \
        and all(cohort['eligible'] for cohort in report['cohorts']) and all(
            all(value.values()) if isinstance(value, dict) else value for value in artifacts.values())
    for row in report['formal'] + report['diagnostic']:
        if not row['eligible']:
            row['measurements'] = None
    return report


def chinese_report(report, comparison):
    lines = ['# Batch=128：查询材料实际驻留内存后的重新测量', '',
        '四个后端使用原来相同的冻结二进制、桥接库、模型和输入。每方四核、1.5 GiB RLIMIT_AS；512 MiB 是该限制内的 FSS 驻留预算，不是额外内存。全部行统一声明此预算，两个纯 SCI 后端不分配 FSS 库存。', '',
        '预处理 SCI 的随机 OT、比较 bit-triple，以及 ours 的剩余原生库存均在输入前就绪于内存。FSS 在 offline 实际读取、反序列化全部查询材料，并在 MATERIAL_READY 前报告完整库存；全部在线转换必须命中库存、material_load_ms=0。在线输入相关修正、打开、密钥评价、扫描、释放仍照实计量，没有事后扣除任何计时。', '',
        'SCI_ours 是仅转换替换的消融组；SCI_Beaver_segmented 才是完整 SCI+ours。SDCF-09 的 99 步整图继续延期。', '',
        '时间为同一查询双方最大墙钟时间，再取 10 个新鲜查询的中位数；warmup=0。直接 query total、首请求和会话总时间独立测量，阶段中位数不相加。通信为双方发送字节之和，MiB=2^20 bytes。嵌套并行转换计时不相加作为查询墙钟。']
    if not report['eligible']:
        lines += ['', '**资格未通过：不报告加速结论或不完整查询的完成成本。**', '',
                  '| 行 | 状态 | 合格 |', '| --- | --- | --- |']
        lines += ['| %s | %s | %s |' % (row['id'], row['status'], row['eligible'])
                  for row in report['formal'] + report['diagnostic']]
        return '\n'.join(lines) + '\n'
    lines += ['', '16 组默认/非零隐状态诊断均通过完整整数核验；每查询 15 张量、172928 个整数。八个正式行共 80 个新鲜查询，关闭中间张量揭示。24 行完整库存、实际 argv/environment、生成 epoch、资源限制及原始日志重聚合均通过资格检查。']
    for cohort in ('a4-sequential', 'a2-parallel'):
        lines += ['', '## ' + cohort, '',
            '| 后端 | offline s | online s [min,max] | 直接 query total s | online MiB | offline MiB | RSS 双方 MiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        rows = [row for row in report['formal'] if row['cohort'] == cohort]
        for row in rows:
            measured = row['measurements']; online = measured['online_seconds']
            lines += ['| %s | %.6f | %.6f [%.6f,%.6f] | %.6f | %.6f | %.6f | %.2f / %.2f |' % (
                LABELS[row['backend']], measured['offline_seconds']['median'], online['median'], online['min'], online['max'],
                measured['query_total_seconds']['median'], measured['online_bytes']['median']/2**20,
                measured['offline_bytes']['median']/2**20, *[value/1024 for value in measured['peak_rss_kib_per_party']])]
        for item in comparison['comparisons']:
            if item['cohort'] != cohort:
                continue
            change = item['changes']
            lines += ['', '%s → %s：在线倍数 %.3fx，在线时延变化 %+.2f%%，在线通信变化 %+.2f%%；直接查询总时延变化 %+.2f%%。' % (
                LABELS[item['baseline']], LABELS[item['comparison']], change['online_speedup'],
                (change['online_seconds']['ratio']-1)*100, (change['online_bytes']['ratio']-1)*100,
                (change['query_total_seconds']['ratio']-1)*100)]
            if change['online_observed_ranges_overlap']:
                lines += ['在线样本范围重叠，本轮中位数差异不足以说明稳定排序；未做显著性检验。']
        lines += ['', '| 后端 | input / ready / output s | setup / 首请求 / session s | 直接查询 / session MiB |',
                  '| --- | --- | --- | --- |']
        for row in rows:
            measured = row['measurements']
            lines += ['| %s | %s | %s | %.6f / %.6f |' % (LABELS[row['backend']],
                ' / '.join('%.6f' % measured[phase+'_seconds']['median'] for phase in ('input', 'ready', 'output')),
                ' / '.join('%.6f' % measured[key] for key in ('setup_seconds', 'first_query_seconds', 'session_total_seconds')),
                measured['query_total_bytes']['median']/2**20, measured['session_total_bytes']/2**20)]
        lines += ['', '| FSS 配置 | 每方每查询转换数 | 驻留材料双方 MiB（首查询） |', '| --- | ---: | --- |']
        for row in rows:
            if row['backend'] in ('SCI_ours', 'SCI_Beaver_segmented'):
                inventories = row['evidence']['resident_materials_per_party']
                lines += ['| %s | %d | %s |' % (LABELS[row['backend']], inventories[0]['queries'][0]['calls'],
                    ' / '.join('%.2f' % (inventory['queries'][0]['resident_bytes']/2**20) for inventory in inventories))]
    lines += ['', '材料驻留加载是真实离线工作，离线通信、加载时间及查询总成本均保留。SCI 库存的在线内存访问与 ours 已驻留材料的在线内存访问都仍包含在测量中。input 无通信，ready 每方发送/接收各一个同步字节；全部 input/ready/online/output 原生及 bridge 生成 epoch 增量为零（默认 SCI 控制除外）。会话通信使用 application_io、setup、query_total、teardown 的互不重叠计数，双方发送/接收交叉核验。', '',
        'ELIGIBILITY.json 保存完整资格与驻留库存证据；COMPARISON.json 保存全部逐查询分布及六组比较。formal/、diagnostics/、reference/、runner/ 保存原始运行、冻结整数参考与运行器源码。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--formal', type=Path)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--reference-dir', type=Path, default=base.RESULTS/'reference')
    parser.add_argument('--contract', type=Path, default=base.REPO/'project/our/tools/segmented_dcf_runtime/run.py')
    args = parser.parse_args()
    report = audit_matrix(args.formal or args.results/'formal', args.diagnostics or args.results/'diagnostics',
                          args.reference_dir, args.contract)
    comparison = base.comparisons(report)
    args.results.mkdir(parents=True, exist_ok=True)
    for name, value in (('ELIGIBILITY.json', report), ('COMPARISON.json', comparison)):
        (args.results/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    (args.results/'REPORT.zh.md').write_text(chinese_report(report, comparison))
    print(json.dumps(dict(complete=report['complete'], eligible=report['eligible'],
        formal=sum(row['eligible'] for row in report['formal']), diagnostic=sum(row['eligible'] for row in report['diagnostic']))))
    return 0 if report['eligible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
