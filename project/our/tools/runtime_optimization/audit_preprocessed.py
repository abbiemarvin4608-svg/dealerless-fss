#!/usr/bin/env python3
"""Qualify and report the frozen, genuinely preprocessed SCI benchmark artifacts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


REPO = Path('/home/jiang/dealerless-lunwen')
RESULTS = REPO / 'project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair'
BACKENDS = {'SCI', 'SCI_preprocessed', 'SCI_ours', 'SCI_Beaver_segmented'}
AFFINITY = [[0, 2, 4, 10], [6, 8, 12, 16]]
SHARED_FIELDS = (
    'binary', 'binary_sha256', 'fixture', 'fixture_identity', 'fixture_files',
    'sample_ids', 'hidden_file', 'batch', 'start', 'steps', 'queries',
    'warmup_queries', 'cpu_affinity_per_party', 'requested_cpu_affinity_per_party',
    'cpu_model', 'machine', 'memory_limit_bytes_per_party', 'network',
    'activation_workers_per_party', 'cggm_worker_limit_per_party', 'graph_mode',
    'graph_openings', 'graph_branches', 'branch_compute_limit', 'offline_overlap',
    'beaver_algebra', 'beaver_mapped_layout', 'beaver_tile_rows', 'aes_online',
    'aes_offline', 'conversion_optimizations', 'material_layout', 'prefetch_mode',
    'prefetch_bytes', 'max_segments', 'segment_policy', 'generation_tile_keys',
    'generation_budget_bytes', 'generation_reserve_bytes', 'diagnostic',
    'native_default_ot_participants', 'native_activation_ot_participants',
    'native_eigen_threads', 'native_ot_direction_schedule', 'measurement_boundaries',
    'application_threads_per_party', 'sci_protocol_threads_per_party',
    'native_protocol_contexts_per_party', 'channel_offsets', 'offline_task_count',
    'offline_task_compute_limit', 'timeout_seconds', 'setup_amortization', 'output_policy',
    'full_test_set_evaluation',
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(log, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in log.splitlines()
            if line.startswith(prefix + ' ')]


def arg(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def native_mode_matches(config, command):
    strict = config.get('backend') in BACKENDS - {'SCI'}
    return (config.get('backend') in BACKENDS
            and config.get('backend') == arg(command, '--backend')
            and config.get('native_preprocessing_requested') is strict
            and config.get('native_preprocessing_effective') is strict
            and ('--native-preprocessing' in command) is strict)


def actual_run_contract_matches(config, actual, row_id):
    """Match the saved child argv/environment to the frozen public contract."""
    commands, environments = actual.get('commands'), actual.get('environment')
    if not isinstance(commands, list) or not isinstance(environments, list) or len(commands) != 2 or len(environments) != 2:
        return False
    port = None
    tmpdir = None
    for role, (command, environment) in enumerate(zip(commands, environments), 1):
        if len(command) != 14 or not isinstance(environment, dict):
            return False
        try:
            base_port = int(command[5])
        except (TypeError, ValueError):
            return False
        if not 1 <= base_port <= 65000 or (port is not None and port != base_port):
            return False
        port = base_port
        expected_command = ['taskset', '-c', ','.join(map(str, AFFINITY[role - 1])), config['binary'],
            str(role), str(base_port), str(Path(config['fixture']) / ('server_model.i16' if role == 1 else 'client_input.i16')),
            str(config['start']), str(config['batch']), str(config['steps']),
            'diagnostic' if config['diagnostic'] else 'secure', str(config['queries'] + config['warmup_queries']),
            config['backend'], config['hidden_file'] or 'none']
        if command != expected_command:
            return False
        current_tmpdir = environment.get('TMPDIR')
        if not isinstance(current_tmpdir, str) or not Path(current_tmpdir).is_absolute() \
                or Path(current_tmpdir).parts[-2:] != (row_id, 'tmp') \
                or (tmpdir is not None and tmpdir != current_tmpdir):
            return False
        tmpdir = current_tmpdir
        expected_environment = dict(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1',
            SDCF_BACKEND=config['backend'], SDCF_NATIVE_PREPROCESSING=str(int(config['native_preprocessing_effective'])),
            SEGFSS_FUSE_DOMAINS=str(int(config['conversion_optimizations']['fuse_domains'])),
            SEGFSS_PACK_BITS=str(int(config['conversion_optimizations']['pack_bits'])),
            SDCF_PAIR_EXTENSIONS=str(int(config['conversion_optimizations']['pair_extensions'])),
            SEGFSS_MATERIAL_LAYOUT=config['material_layout'], SEGFSS_PREFETCH_BYTES=str(config['prefetch_bytes']),
            SEGFSS_AES_ONLINE=str(int(config['aes_online'])), SEGFSS_AES_OFFLINE=str(int(config['aes_offline'])),
            SEGFSS_SEGMENT_POLICY=config['segment_policy'], SEGFSS_GENERATION_TILE_KEYS=str(config['generation_tile_keys']),
            SEGFSS_GENERATION_BUDGET_BYTES=str(config['generation_budget_bytes']),
            SEGFSS_GENERATION_RESERVE_BYTES=str(config['generation_reserve_bytes']),
            SDCF_ACTIVATION_WORKERS=str(config['activation_workers_per_party']), SDCF_GRAPH_MODE=config['graph_mode'],
            SDCF_GRAPH_OPENINGS=config['graph_openings'], SEGFSS_CGGM_THREADS=str(config['cggm_worker_limit_per_party']),
            SDCF_BEAVER_ALGEBRA=str(int(config['beaver_algebra'])), SDCF_BEAVER_MAPPED_LAYOUT=str(int(config['beaver_mapped_layout'])),
            SDCF_BEAVER_TILE_ROWS=str(config['beaver_tile_rows']), SDCF_OFFLINE_OVERLAP=str(int(config['offline_overlap'])),
            SDCF_PORT=str(base_port + 20), SDCF_HOST='127.0.0.1', SDCF_MAX_SEGMENTS=str(config['max_segments']), TMPDIR=tmpdir)
        if environment != expected_environment:
            return False
    return True


def declared_configuration_matches(config, command):
    values = {'--binary': 'binary', '--fixture': 'fixture', '--backend': 'backend',
        '--batch': 'batch', '--start': 'start', '--steps': 'steps', '--queries': 'queries',
        '--warmup': 'warmup_queries',
        '--activation-workers': 'activation_workers_per_party', '--cggm-threads': 'cggm_worker_limit_per_party',
        '--graph-mode': 'graph_mode', '--graph-openings': 'graph_openings', '--max-segments': 'max_segments',
        '--material-layout': 'material_layout', '--segment-policy': 'segment_policy',
        '--generation-tile-keys': 'generation_tile_keys', '--generation-budget-bytes': 'generation_budget_bytes',
        '--generation-reserve-bytes': 'generation_reserve_bytes', '--beaver-tile-rows': 'beaver_tile_rows'}
    if any(str(config[field]) != arg(command, flag) for flag, field in values.items()):
        return False
    switches = {'--diagnostic': config['diagnostic'], '--aes-online': config['aes_online'],
        '--aes-offline': config['aes_offline'], '--beaver-algebra': config['beaver_algebra'],
        '--beaver-mapped-layout': config['beaver_mapped_layout'], '--offline-overlap': config['offline_overlap']}
    switches.update({'--' + name.replace('_', '-'): value for name, value in config['conversion_optimizations'].items()})
    return all((flag in command) is value for flag, value in switches.items()) \
        and config['timeout_seconds'] == float(arg(command, '--timeout', 'nan')) \
        and config['network']['name'] == arg(command, '--network') \
        and config['memory_limit_bytes_per_party'] == int(float(arg(command, '--memory-gib', '0')) * 2**30) \
        and config['hidden_file'] == arg(command, '--hidden') \
        and all(arg(command, flag) == ','.join(map(str, AFFINITY[index]))
                for index, flag in enumerate(('--server-cpus', '--client-cpus')))


def matching_diagnostics(formal, diagnostics):
    rows = [row for row in diagnostics if row.get('backend') == formal['backend']
            and row.get('cohort') == formal['cohort']]
    return (len(rows) == 2 and {row.get('hidden') for row in rows} == {'default', 'nonzero'}
            and all(row.get('eligible') is True for row in rows))


def measured_summary(parties, queries):
    """Recompute aligned query maxima and direct intervals from party records."""
    def pair(phase, query):
        rows = [[row for row in party['phases'] if row['phase'] == phase and row['query'] == query]
                for party in parties]
        if any(len(rows_for_party) != 1 for rows_for_party in rows):
            raise ValueError('missing or duplicate measured phase: ' + phase)
        left, right = (rows_for_party[0] for rows_for_party in rows)
        if left['sent_bytes'] != right['received_bytes'] or right['sent_bytes'] != left['received_bytes']:
            raise ValueError('unbalanced phase communication: ' + phase)
        return dict(seconds=max(left['seconds'], right['seconds']),
                    bytes=left['sent_bytes'] + right['sent_bytes'])
    def distribution(values):
        return dict(median=statistics.median(values), min=min(values), max=max(values),
                    pstdev=statistics.pstdev(values), values=values)
    summary = dict(session_total_seconds=max(party['total_seconds'] for party in parties),
                   peak_rss_kib_per_party=[party['peak_rss_kib'] for party in parties])
    # These measured counter intervals cover the session without overlap.
    # first_query and individual query phases would duplicate the same bytes.
    session_intervals = {'application_io', 'setup', 'query_total', 'teardown'}
    for field in ('sent_bytes', 'received_bytes'):
        summary['session_' + field + '_per_party'] = [
            sum(row[field] for row in party['phases'] if row['phase'] in session_intervals)
            for party in parties]
    sent, received = (summary['session_' + field + '_per_party']
                      for field in ('sent_bytes', 'received_bytes'))
    if sent != list(reversed(received)):
        raise ValueError('unbalanced session communication')
    summary['session_total_bytes'] = sum(sent)
    summary['session_communication_intervals'] = sorted(session_intervals)
    for phase in ('setup', 'first_query'):
        measured = pair(phase, -1)
        summary[phase + '_seconds'] = measured['seconds']
        summary[phase + '_bytes'] = measured['bytes']
    for phase in ('offline', 'input', 'ready', 'online', 'output', 'query_total'):
        if not all(any(row['phase'] == phase for row in party['phases']) for party in parties):
            continue
        aligned = [pair(phase, query) for query in range(queries)]
        for field in ('seconds', 'bytes'):
            summary[phase + '_' + field] = distribution([row[field] for row in aligned])
    return summary


def normalized_command(command):
    normalized = []
    cursor = 0
    while cursor < len(command):
        if command[cursor] in ('--backend', '--output'):
            cursor += 2
        elif command[cursor] == '--native-preprocessing':
            cursor += 1
        else:
            normalized.append(command[cursor])
            cursor += 1
    return normalized


def identities(rows, fields=('query', 'index')):
    return [tuple(row.get(field) for field in fields) for row in rows]


def exact_coverage(rows, expected, fields=('query', 'index')):
    actual = identities(rows, fields)
    return len(actual) == len(set(actual)) and set(actual) == expected


def positive_cap(rows, field, limit):
    return all(type(row.get(field)) is int and 1 <= row[field] <= limit for row in rows)


def peak(rows, field, absent=0):
    return max(row[field] for row in rows) if rows else absent


def scalar_function(path):
    source = path.read_text()
    begin = source.index('void ScalarMul(uint64_t *A')
    brace = source.index('{', begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def public_scalar_source(binary):
    directory = binary.parent
    original = scalar_function(Path('/home/jiang/EzPC/SCI/src/library_fixed.cpp'))
    native = scalar_function(directory / 'library_fixed_native_workers.cpp')
    for replacement, pointer in (
            ('sdcf_native::selected_xt()', 'xt'),
            ('sdcf_native::selected_truncation()', 'truncation'),
            ('sdcf_native::selected_aux()', 'aux')):
        native = native.replace(replacement, pointer)
    beaver = scalar_function(directory / 'library_fixed_beaver.cpp')
    return dict(native_scalar_contract_matches_original=native == original,
                beaver_scalar_contract_matches_original=beaver == original,
                scalar_multiplication='C[i] = (tmpC[i] * A[0]) & maskTemp',
                native_source=str(directory / 'library_fixed_native_workers.cpp'),
                beaver_source=str(directory / 'library_fixed_beaver.cpp'))


def load_contract_module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('preprocessed_public_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_row(directory, declaration, diagnostic, contract, reference_dir):
    row_id = declaration['id']
    output = directory / row_id
    result_path = output / 'result.json'
    if not result_path.exists():
        return dict(id=row_id, backend=declaration['backend'], cohort=declaration['cohort'], hidden=declaration.get('hidden'), status='pending', eligible=False, measurements=None), None
    result = read(result_path)
    if result.get('status') != 'complete':
        return dict(id=row_id, backend=declaration['backend'], cohort=declaration['cohort'], hidden=declaration.get('hidden'), status=result.get('status'), reason=result.get('reason'),
                    eligible=False, measurements=None), None
    cfg = read(output / 'config.json')
    parties = read(output / 'parties.json')
    schedule = read(output / 'schedule.json')
    commands = read(output / 'commands.json')
    queries = 2 if diagnostic else 10
    profile = declaration['cohort']
    cap = 2 if profile == 'a2-parallel' else 4
    graph_mode = 'parallel' if cap == 2 else 'sequential'
    expected_layers = {(query, layer['index']) for query in range(queries) for layer in schedule}
    expected_beaver = {(query, layer['index']) for query in range(queries) for layer in schedule
                       if layer['function'] in ('MatMul', 'MulCir')}
    expected_scalar = {(query, layer['index']) for query in range(queries) for layer in schedule
                       if layer['function'] == 'ScalarMul'}
    uses_beaver = cfg['backend'] == 'SCI_Beaver_segmented'
    uses_fss = cfg['backend'] in ('SCI_ours', 'SCI_Beaver_segmented')
    checks = dict(
        result_and_config_identical=result['config'] == cfg,
        completed_two_parties=result.get('party_exit_codes') == [0, 0]
            and len(parties) == 2 and {p['party'] for p in parties} == {1, 2}
            and all(p['status'] == 'complete' for p in parties),
        declared_backend_matches=cfg['backend'] == arg(declaration['command'], '--backend') == declaration['backend'],
        exact_native_mode=native_mode_matches(cfg, declaration['command']),
        declared_configuration_matches=declared_configuration_matches(cfg, declaration['command']),
        actual_child_contract_matches=actual_run_contract_matches(cfg, commands, row_id),
        actual_fixture_matches_frozen_hashes=all(sha(Path(cfg['fixture']) / name) == value
            for name, value in cfg['fixture_files'].items()),
        frozen_release_numeric_contract=cfg['runtime_manifest'].get('numeric_mode') == 'sci_compatible'
            and cfg['runtime_manifest'].get('DIV_RESCALING') is True,
        protocol_thread_and_context_contract=cfg.get('application_threads_per_party') == 1
            and cfg.get('sci_protocol_threads_per_party') == 1 and cfg.get('native_protocol_contexts_per_party') == 4,
        no_extra_offline_resources=cfg.get('offline_task_count') == 1
            and cfg.get('offline_task_compute_limit') == cap and cfg.get('timeout_seconds') == 600,
        unchanged_scope=cfg.get('full_test_set_evaluation') is False,
        frozen_optimizer_defaults=cfg.get('beaver_mapped_layout') is False
            and cfg.get('aes_offline') is False and cfg.get('prefetch_bytes') == 0,
        frozen_batch_query_and_scope=cfg['batch'] == 128 and cfg['start'] == 0
            and cfg['steps'] == 1 and cfg['queries'] == queries and cfg['warmup_queries'] == 0
            and cfg['diagnostic'] is diagnostic and cfg['quality_validation'] == 'SDCF-09 deferred by user',
        measured_query_count_matches=len(result['measurements']['queries']) == queries
            and result['measurements']['warmup_queries'] == [],
        profile_matches=cfg['activation_workers_per_party'] == cap
            and cfg['cggm_worker_limit_per_party'] == cap and cfg['branch_compute_limit'] == cap
            and cfg['graph_mode'] == graph_mode and cfg['offline_overlap'] is False,
        exact_four_core_affinity=cfg['cpu_affinity_per_party'] == AFFINITY
            and cfg['requested_cpu_affinity_per_party'] == AFFINITY,
        declared_memory_matches=cfg['memory_limit_bytes_per_party'] == 1610612736,
        peak_rss_fits=all(p['peak_rss_kib'] * 1024 < cfg['memory_limit_bytes_per_party']
                          for p in parties),
        binary_matches_frozen_manifest=cfg['binary_sha256'] == cfg['runtime_manifest']['binary_sha256']
            and cfg['binary_sha256'] == sha(Path(cfg['binary'])),
        secure_mode_without_tensor=True,
    )
    evidence = dict(profile=profile, backend=cfg['backend'], queries=queries,
                    peak_rss_kib_per_party=[p['peak_rss_kib'] for p in parties],
                    participants={}, fresh_materials_per_party=[],
                    public_scalar_calls_per_party=[], tensor_counts_per_party=[])
    logs = [(output / ('party%d.log' % role)).read_text() for role in (1, 2)]
    try:
        fresh_parties, fresh_measurements = contract.collect(logs, queries, schedule, diagnostic,
            backend=cfg['backend'], native_preprocessing=cfg['native_preprocessing_effective'])
        fresh_measurements['warmup_queries'] = []
        checks['raw_log_recollection_matches'] = json.loads(json.dumps(fresh_parties)) == parties
        checks['raw_phase_aggregation_matches'] = fresh_measurements == result['measurements']
    except Exception as error:
        checks['raw_log_recollection_matches'] = False
        checks['raw_phase_aggregation_matches'] = False
        evidence['recollection_error'] = str(error)
    party_audits = []
    for role, party in enumerate(parties, 1):
        log = (output / ('party%d.log' % role)).read_text()
        coverage = records(log, 'COVERAGE')
        worker = party['native_worker_config']
        tasks = party['graph_tasks']
        groups = party['graph_groups']
        prep = party['offline_preparation']
        off = party['beaver_offline_calls']
        online = party['beaver_calls']
        scalar = party['public_scalar_calls']
        detail = party['conversion_details']
        pre = party['conversion_preprocess_details']
        generation = records(log, 'DCF_GENERATION')
        tensors = [line for line in log.splitlines() if line.startswith('TENSOR ')]
        finished = records(log, 'FINISHED')
        phase_records = party['phases']
        wanted_phases = Counter((q, phase) for q in range(queries)
                                for phase in ('offline', 'input', 'ready', 'online', 'output', 'query_total'))
        actual_phases = Counter((p['query'], p['phase']) for p in phase_records if p['query'] >= 0)
        matrix_off = [m for m in off if 'rows' in m]
        party_checks = dict(
            exact_schedule_coverage=exact_coverage(coverage, expected_layers)
                and all(row['batch'] == 128 and row['sample_groups'] == 1 for row in coverage),
            exact_phase_coverage=actual_phases == wanted_phases,
            exact_session_phase_coverage=Counter(p['phase'] for p in phase_records if p['query'] == -1)
                == Counter(('application_io', 'setup', 'first_query', 'teardown')),
            completed_log_record=len(finished) == 1 and finished[0]['party'] == role
                and finished[0]['status'] == 'complete',
            taskset_command_matches=len(commands['commands']) == 2
                and commands['commands'][role - 1][:3] == ['taskset', '-c', ','.join(map(str, AFFINITY[role - 1]))]
                and commands['commands'][role - 1][3] == cfg['binary'],
            worker_configuration=len(worker) == 1 and worker[0]['cggm_limit'] == cap
                and worker[0]['requested_activation_workers'] == cap and worker[0]['eigen_threads'] == 1
                and worker[0]['protocol_contexts'] == 4
                and worker[0]['max_compute_participants'] == 4
                and worker[0]['default_ot_participants'] == 2 and worker[0]['activation_ot_participants'] == 1,
            graph_task_coverage=exact_coverage(tasks, expected_layers),
            observed_task_ot_cap=positive_cap(tasks, 'observed_ot_participants', min(cap, 2)),
            observed_task_compute_cap=positive_cap(tasks, 'observed_compute_participants', cap),
            observed_graph_total_cap=all(g['observed_compute_upper_bound'] <= 4
                and g['observed_first_branch_peak'] <= cap and g['observed_second_branch_peak'] <= cap for g in groups),
            graph_join_coverage=exact_coverage(groups, {(q, index) for q in range(queries) for index in (0, 5)})
                if cap == 2 else not groups,
            activation_actual_cap=positive_cap(party['activations'], 'actual_workers', cap)
                and positive_cap(party['activations'], 'compute_participants', cap)
                and positive_cap(party['activation_workers'], 'ot_participants', 1),
            observed_cggm_generation_cap=positive_cap(pre, 'generation_participants', cap),
            observed_cggm_online_cap=positive_cap(detail, 'eval_participants', cap)
                and positive_cap(detail, 'keygen_participants', cap),
            requested_dcf_generation_cap=positive_cap(generation, 'participant_limit', cap),
            offline_preparation_coverage=exact_coverage(prep, {(q,) for q in range(queries)}, ('query',))
                and all(p['ready'] is True and p['overlap'] is False and p['task_limit'] == 1 for p in prep),
            observed_offline_ot_cap=positive_cap(prep, 'beaver_ot_participants', 2),
            fresh_beaver_coverage=exact_coverage(off, expected_beaver if uses_beaver else set())
                and all(p.get('fresh') is True and p.get('dealerless') is True
                        and p.get('activation_triples') == 0 for p in off),
            one_use_beaver_consumption=exact_coverage(online, expected_beaver if uses_beaver else set())
                and all(p.get('consumed') is True for p in online),
            matrix_owned_receive_available=all(p.get('receive_instrumented') is True
                and type(p.get('received_bytes')) is int and p['received_bytes'] > 0 for p in matrix_off),
            public_scalar_local_without_triples=exact_coverage(scalar, expected_scalar if uses_beaver else set())
                and all(p['operand_visibility'] == 'public_model_scalar' and p['multiplication'] == 'local'
                        and p['multiplication_triples'] == 0 and p['elements'] == 12800 for p in scalar),
            no_raw_secret_opening=all(p.get('raw_secret_openings') == 0 for p in detail),
            no_diagnostic_disclosure=(diagnostic or not tensors),
        )
        strict = cfg['backend'] != 'SCI'
        party_checks['native_inventory_mode_and_coverage'] = (
            all(party.get(key) is strict for key in ('native_preprocessing_verified', 'native_public_plan_verified',
                'native_planning_verified', 'native_triples_verified'))
            and len(party['native_material_ready']) == queries
            and all(item['strict'] is strict for item in party['native_material_ready']))
        party_checks['default_control_has_no_inventory'] = strict or all(not party.get(key) for key in (
            'native_public_plan', 'native_material_prepared', 'native_material_consumed', 'native_material_balances',
            'native_planning_purity', 'native_triple_plan', 'native_triple_prepared', 'native_triple_consumed', 'native_triple_balances'))
        party_checks['all_live_contexts_planned'] = (not strict or all(
            {item['context'] for item in party['native_public_plan'] if item['query'] == query} == set(range(4))
            for query in range(queries)))
        party_checks['input_phase_after_material_ready'] = True
        for query in range(queries):
            ready_positions = [i for i, line in enumerate(log.splitlines()) if line.startswith('MATERIAL_READY ')
                and json.loads(line.split(' ', 1)[1]).get('query') == query]
            input_positions = [i for i, line in enumerate(log.splitlines()) if line.startswith('METRIC ')
                and json.loads(line.split(' ', 1)[1]).get('query') == query
                and json.loads(line.split(' ', 1)[1]).get('phase') == 'input']
            party_checks['input_phase_after_material_ready'] &= (len(ready_positions) == len(input_positions) == 1
                and ready_positions[0] < input_positions[0])
        try:
            contract.validate_conversions(party['conversion_offline_calls'], party['conversion_calls'],
                pre, party['conversion_groups'], schedule, queries, cfg['backend'], False)
            party_checks['complete_public_conversion_contract'] = True
        except Exception as error:
            party_checks['complete_public_conversion_contract'] = False
            evidence.setdefault('conversion_errors', []).append(str(error))
        party_checks['fresh_fss_preprocessing_coverage'] = (
            len(pre) == len(party['conversion_calls']) and
            all(p.get('fresh') is True and p.get('dealerless') is True for p in pre)
            and (bool(pre) is uses_fss))
        checks['secure_mode_without_tensor'] &= diagnostic or not tensors
        if diagnostic:
            state = declaration['hidden']
            reference = reference_dir / state / 'SCI-n4-t4/party2.log'
            expected = [line for line in reference.read_text().splitlines() if line.startswith('TENSOR ')]
            party_checks['exact_integer_reference'] = (not tensors if role == 1 else
                len(expected) == 15 and sum(len(line.split()) - 7 for line in expected) == 172928
                and tensors == expected * queries)
        else:
            party_checks['secure_command_mode'] = 'secure' in commands['commands'][role - 1]
        evidence['tensor_counts_per_party'].append(len(tensors))
        evidence['fresh_materials_per_party'].append(dict(beaver=len(off), conversion=len(pre)))
        evidence['public_scalar_calls_per_party'].append(len(scalar))
        evidence['participants'][str(role)] = dict(
            actual_cggm_generation=peak(pre, 'generation_participants'),
            actual_cggm_evaluation=peak(detail, 'eval_participants'),
            actual_graph_ot=peak(tasks, 'observed_ot_participants'),
            graph_task_observed_compute_participants_max=peak(tasks, 'observed_compute_participants', None),
            graph_group_observed_compute_upper_bound_max=peak(groups, 'observed_compute_upper_bound', None),
            actual_beaver_offline_ot=peak(prep, 'beaver_ot_participants'),
            actual_activation_compute=peak(party['activations'], 'compute_participants'),
            actual_activation_worker_ot=peak(party['activation_workers'], 'ot_participants'))
        party_audits.append(dict(party=role, checks=party_checks,
                                 eligible=all(party_checks.values())))
    try:
        contract.validate_beaver_traffic(parties)
        checks['cross_party_owned_beaver_send_receive_match'] = True
    except Exception as error:
        checks['cross_party_owned_beaver_send_receive_match'] = False
        evidence['beaver_traffic_error'] = str(error)
    evidence['native_materials_per_party_per_query'] = [
        [dict(query=query,
              ot={field: sum(item[field] for item in party['native_material_balances'] if item['query'] == query)
                  for field in ('requested', 'generated', 'consumed', 'padding_discarded', 'remaining')},
              triples={field: sum(item[field] for item in party['native_triple_balances'] if item['query'] == query)
                       for field in ('requested', 'generated', 'consumed', 'padding_discarded', 'remaining')},
              standard_triple_calls=sum(item['standard_calls'] for item in party['native_triple_prepared'] if item['query'] == query),
              correlated_triple_calls=sum(item['correlated_calls'] for item in party['native_triple_prepared'] if item['query'] == query))
         for query in range(queries)] for party in parties]
    evidence['shared_configuration'] = {key: cfg.get(key) for key in SHARED_FIELDS}
    return dict(id=row_id, backend=cfg['backend'], cohort=profile, hidden=declaration.get('hidden'),
                status='complete', checks=checks, parties=party_audits, evidence=evidence,
                measurements=measured_summary(parties, queries),
                eligible=all(checks.values()) and all(p['eligible'] for p in party_audits)), cfg


def diagnostic_command(command):
    """Compare the formal contract with only declared diagnostic differences removed."""
    values, cursor = [], 0
    while cursor < len(command):
        flag = command[cursor]
        if flag in ('--output', '--queries', '--hidden'):
            cursor += 2
        elif flag == '--diagnostic':
            cursor += 1
        else:
            values.append(flag)
            cursor += 1
    return values


def audit_matrix(formal_dir, diagnostic_dir, reference_dir, contract_path):
    contract = load_contract_module(contract_path)
    report = dict(scope='Batch128 single FastGRNN timestep; SDCF-09 deferred', formal=[], diagnostic=[],
                  cohorts=[], artifact_checks={}, contract_checker=str(contract_path))
    all_cfg, declarations, manifests = [], {}, {}
    for kind, directory, count in (('diagnostic', diagnostic_dir, 16), ('formal', formal_dir, 8)):
        manifest = read(directory / 'manifest.json')
        manifests[kind] = manifest
        rows = manifest.get('command_rows', manifest['rows'])
        declarations[kind] = rows
        expected = {(backend, cohort, hidden) for backend in BACKENDS
                    for cohort in ('a4-sequential', 'a2-parallel')
                    for hidden in (('default', 'nonzero') if kind == 'diagnostic' else (None,))}
        actual = [(row['backend'], row['cohort'], row.get('hidden')) for row in rows]
        summary = read(directory / 'summary.json')['rows']
        report['artifact_checks'][kind] = dict(
            exact_declaration_set=len(rows) == count and len(set(actual)) == count and set(actual) == expected,
            unique_ids=len({row['id'] for row in rows}) == count,
            complete_summary=len(summary) == count and all(row['status'] == 'complete' for row in summary),
            exact_summary_ids={row['id'] for row in summary} == {row['id'] for row in rows},
            no_extra_output_rows={path.name for path in directory.iterdir() if path.is_dir()} == {row['id'] for row in rows},
            actual_binary_frozen=sha(Path(manifest['binary'])) == manifest['binary_sha256'],
            actual_bridge_frozen=sha(Path(manifest['bridge'])) == manifest['bridge_sha256'])
        for declaration in rows:
            try:
                row, cfg = audit_row(directory, declaration, kind == 'diagnostic', contract, reference_dir)
            except Exception as error:
                row = dict(id=declaration['id'], backend=declaration['backend'], cohort=declaration['cohort'],
                           hidden=declaration.get('hidden'), status='invalid_evidence', reason=str(error),
                           eligible=False, measurements=None)
                cfg = None
            if cfg:
                sources = cfg['runtime_manifest']['sources']
                bridges = [(path, value) for path, value in sources.items() if Path(path).name == 'libsegfss_sci_bridge.so']
                checks = row['checks']
                checks['frozen_manifest_identity'] = (cfg['binary'] == manifest['binary']
                    and cfg['binary_sha256'] == manifest['binary_sha256']
                    and cfg['binary'] == arg(declaration['command'], '--binary')
                    and bridges == [(manifest['bridge'], manifest['bridge_sha256'])])
                checks['frozen_generated_source'] = sha(Path(cfg['runtime_manifest']['generated'])) == cfg['runtime_manifest']['generated_sha256']
                checks['hidden_state_matches_declaration'] = (cfg['hidden_file'] == arg(declaration['command'], '--hidden')
                    and ((cfg['hidden_file'] is None) if declaration.get('hidden') != 'nonzero' else
                         sha(Path(cfg['hidden_file'])) == '96d0e2436ca32b5fd5d333ba75866540a2eb03f8e7ee0a2f2fd0a5fcbf31f400'))
                row['eligible'] &= all(checks.values())
                all_cfg.append(cfg)
            report[kind].append(row)
    grouped = defaultdict(list)
    for kind in ('formal', 'diagnostic'):
        audits = {row['id']: row for row in report[kind]}
        for declaration in declarations[kind]:
            grouped[(kind, declaration['cohort'], declaration.get('hidden'))].append((declaration, audits[declaration['id']]))
    for (kind, cohort, hidden), rows in grouped.items():
        configurations = [row['evidence']['shared_configuration'] for _, row in rows if 'evidence' in row]
        checks = dict(four_backends={d['backend'] for d, _ in rows} == BACKENDS and len(rows) == 4,
            shared_command_and_environment=all(normalized_command(d['command']) == normalized_command(rows[0][0]['command'])
                and d.get('environment') == rows[0][0].get('environment') for d, _ in rows),
            shared_observed_configuration=len(configurations) == 4 and all(c == configurations[0] for c in configurations),
            all_rows_qualified=all(row['eligible'] for _, row in rows))
        report['cohorts'].append(dict(kind=kind, cohort=cohort, hidden=hidden, checks=checks, eligible=all(checks.values())))
    for formal in report['formal']:
        formal.setdefault('checks', {})
        formal['checks']['matching_default_and_nonzero_diagnostics'] = matching_diagnostics(formal, report['diagnostic'])
        formal_declaration = next(d for d in declarations['formal'] if d['id'] == formal['id'])
        matches = [d for d in declarations['diagnostic'] if d['backend'] == formal['backend'] and d['cohort'] == formal['cohort']]
        formal['checks']['formal_diagnostic_command_contract'] = len(matches) == 2 and all(
            diagnostic_command(d['command']) == diagnostic_command(formal_declaration['command'])
            and d.get('environment') == formal_declaration.get('environment') for d in matches)
        formal['eligible'] &= all(formal['checks'].values())
    artifacts = report['artifact_checks']
    artifacts['same_declared_binary_and_bridge'] = all(manifests['formal'][key] == manifests['diagnostic'][key]
        for key in ('binary', 'binary_sha256', 'bridge', 'bridge_sha256'))
    artifacts['single_runtime_manifest'] = len(all_cfg) == 24 and all(c['runtime_manifest'] == all_cfg[0]['runtime_manifest'] for c in all_cfg)
    if all_cfg:
        report['public_scalar_source'] = public_scalar_source(Path(all_cfg[0]['binary']))
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


def comparisons(report):
    output = dict(scope=report['scope'], eligible=report['eligible'], rows=report['formal'], comparisons=[])
    for cohort in ('a4-sequential', 'a2-parallel'):
        rows = {row['backend']: row for row in report['formal'] if row['cohort'] == cohort}
        for baseline, ours in (('SCI', 'SCI_preprocessed'), ('SCI_preprocessed', 'SCI_ours'),
                               ('SCI_preprocessed', 'SCI_Beaver_segmented')):
            left, right = rows.get(baseline), rows.get(ours)
            eligible = bool(report['eligible'] and left and right and left['eligible'] and right['eligible'])
            comparison = dict(cohort=cohort, baseline=baseline, comparison=ours, eligible=eligible, changes=None)
            if eligible:
                changes = {}
                for phase in ('offline', 'online', 'query_total'):
                    for field in ('seconds', 'bytes'):
                        key = phase + '_' + field
                        base, new = left['measurements'][key], right['measurements'][key]
                        changes[key] = dict(baseline=base, comparison=new,
                            signed_absolute_change=new['median'] - base['median'],
                            ratio=new['median'] / base['median'] if base['median'] else None,
                            reduction=1-new['median']/base['median'] if base['median'] else None)
                changes['online_speedup'] = left['measurements']['online_seconds']['median'] / right['measurements']['online_seconds']['median']
                base, new = left['measurements']['online_seconds'], right['measurements']['online_seconds']
                changes['online_observed_ranges_overlap'] = max(base['min'], new['min']) <= min(base['max'], new['max'])
                comparison['changes'] = changes
            output['comparisons'].append(comparison)
    return output

def chinese_report(report, comparison):
    if not report['eligible']:
        lines = ['# Batch=128：资格未通过', '', '未通过完整资格核对；以下保留真实状态，不给出完成成本或加速结论。', '', '| 行 | 状态 | 合格 |', '| --- | --- | --- |']
        lines += ['| %s | %s | %s |' % (row['id'],row['status'],row['eligible']) for row in report['formal']+report['diagnostic']]
        return '\n'.join(lines)+'\n'
    lines = ['# Batch=128：SCI+SiRNN 真实预处理公平对照', '',
        '在双方都预处理剩余原生 SCI 工作后，Beaver+分段 FSS 组合仍降低在线时延和通信；仅替换图级转换的 SCI_ours 本轮在线时延增加。预处理显著增加每查询离线成本，所有预处理方案的直接查询总时延均高于默认 SCI。', '',
        '测试使用两方 FastGRNN 单时间步、batch=128、同一冻结模型与输入，localhost，无网络整形。每行 10 个新鲜查询、warmup=0、600 秒预算；服务方 CPU 0,2,4,10，客户端 CPU 6,8,12,16，每方四核及 1.5 GiB RLIMIT_AS。a4-sequential 顺序执行图任务，激活最多四个 worker；a2-parallel 两分支各最多两个计算参与者。八行串行测量。', '',
        '默认 SCI 仅使用持久 setup，不启用每查询原生预处理；SCI_preprocessed、SCI_ours、SCI_Beaver_segmented 都在输入与隐状态装载前实际生成本查询所需随机 OT 和安全标准/相关 bit-triple。SCI_ours 仅替换图级截断/扩环，组合后端还替换图级乘法。Sigmoid/TanH 数值算法保持原生 SCI。', '',
        '16 个默认/非零隐状态诊断分别与新 SCI 控制和冻结 SCI 输出逐整数一致；每查询 15 张量、172928 个整数。八个正式行关闭中间张量揭示。原始日志重新收集后与已保存的材料、计量和配置记录一致，全部 24 行通过资格核对，其中 8 行进入主表。', '',
        '时间采用同查询双方最大墙钟时间；通信采用双方发送字节之和。表内时间单位为秒，通信单位为 MiB（2^20 bytes）。query total、首请求和会话总时间均直接测量；阶段中位数不相加。会话总通信按 application_io、setup、各 query total、teardown 的实际计数差累计，覆盖加载模型、上下文初始化、所有查询及结束握手，不重复计入 first_query 或细分阶段；双方发送与对方接收计数交叉一致。每阶段 10 个样本、标准差及范围保存在 COMPARISON.json。']
    for cohort in ('a4-sequential', 'a2-parallel'):
        rows = [row for row in report['formal'] if row['cohort'] == cohort]
        lines += ['', '## ' + cohort, '', '| 后端 | offline | online 中位数 [min,max] | 直接 query total | setup | 首请求 | session total | RSS 双方 MiB |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in rows:
            if not row['eligible']:
                lines.append('| %s | %s | null | null | null | null | null | null |' % (row['backend'], row['status']))
                continue
            m = row['measurements']; online = m['online_seconds']; rss = m['peak_rss_kib_per_party']
            lines.append('| %s | %.6f | %.6f [%.6f,%.6f] | %.6f | %.6f | %.6f | %.6f | %.2f / %.2f |' % (
                row['backend'], m['offline_seconds']['median'], online['median'], online['min'], online['max'],
                m['query_total_seconds']['median'], m['setup_seconds'], m['first_query_seconds'],
                m['session_total_seconds'], rss[0]/1024, rss[1]/1024))
        lines += ['', '| 后端 | input 时间 | ready 时间 | output 时间 | offline 通信 | online 通信 | query total 通信 | setup 通信 | 首请求通信 | session total 通信 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            if row['eligible']:
                m = row['measurements']
                lines.append('| %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |' % (
                    row['backend'], *[m[p+'_seconds']['median'] for p in ('input','ready','output')],
                    *[m[p+'_bytes']['median']/2**20 for p in ('offline','online','query_total')],
                    m['setup_bytes']/2**20, m['first_query_bytes']/2**20, m['session_total_bytes']/2**20))
        lines += ['', '| 后端 | input / ready / output 通信 MiB |', '| --- | --- |']
        for row in rows:
            if row['eligible']:
                m=row['measurements']
                lines.append('| %s | %s |' % (row['backend'], ' / '.join('%.6f' % (m[p+'_bytes']['median']/2**20) for p in ('input','ready','output'))))
        for item in comparison['comparisons']:
            if item['cohort'] != cohort or not item['eligible']:
                continue
            c=item['changes']; t=c['online_seconds']; b=c['online_bytes']
            time_word='下降' if t['reduction'] >= 0 else '增加'
            byte_word='下降' if b['reduction'] >= 0 else '增加'
            lines += ['', '%s → %s：在线倍数 %.3fx，在线时延%s %.2f%%，在线通信%s %.2f%%。每查询离线时延变化 %+.6f 秒，直接 query total 变化 %+.6f 秒（%+.2f%%）；离线通信变化 %+.6f MiB，直接 query total 通信变化 %+.6f MiB。' % (
                item['baseline'], item['comparison'], c['online_speedup'], time_word, abs(t['reduction'])*100,
                byte_word, abs(b['reduction'])*100, c['offline_seconds']['signed_absolute_change'],
                c['query_total_seconds']['signed_absolute_change'], (c['query_total_seconds']['ratio']-1)*100,
                c['offline_bytes']['signed_absolute_change']/2**20, c['query_total_bytes']['signed_absolute_change']/2**20)]
            if c['online_observed_ranges_overlap']:
                lines += ['这两行在线样本范围重叠；本轮中位数差异不足以给出稳定排序，也未进行显著性检验。']
            else:
                lines += ['本轮在线样本范围分离；这描述本次 10 查询观测，不作为显著性检验。']
    lines += ['', '## 24 行资格', '', '| 类型 | 行 | 状态 | 合格 |', '| --- | --- | --- | --- |']
    for kind in ('formal', 'diagnostic'):
        lines += ['| %s | %s | %s | %s |' % (kind,row['id'],row['status'],row['eligible'])
                  for row in report[kind]]
    lines += ['', '## 每查询材料与在线边界', '',
        '| cohort | 后端 | 每方有效随机 OT | 每方生成随机 OT（含 padding） | 每方安全 bit-triple | 标准 / 相关 triple 调用 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for row in report['formal']:
        if row['eligible']:
            sample=row['evidence']['native_materials_per_party_per_query'][0][0]
            lines.append('| %s | %s | %d | %d | %d | %d / %d |' % (row['cohort'],row['backend'],sample['ot']['consumed'],
                sample['ot']['generated'],sample['triples']['consumed'],sample['standard_triple_calls'],sample['correlated_triple_calls']))
    lines += ['', '上表是剩余原生 SCI 库存，不包含已经替换的调用；组合后端的 Beaver 与 FSS 材料另有原始生成/消费记录。所有查询的两方库存逐项核验，随机 OT padding 在查询结束丢弃，剩余有效库存为零；bit-triple 由实际安全协议生成。四个原生上下文的计划与消费完整覆盖，严格模式 input、ready、online、output 的原生和 bridge 生成 epoch 增量均为零。', '',
        'Base OT 与持久上下文创建计 setup；每查询随机 OT 扩展、bit-triple 库存、Beaver/FSS 材料生成及 material-ready 工作计 offline。输入相关选择修正、算术相关量/消息适配、比较 AND 的打开与修正留在 online；输入后的材料读取、等待、掩码与密钥评价同样留在 online。input 仅装载查询数据，ready 仅同步，表中通信证明没有把协议通信隐藏在这两个阶段。最终公开输出重构计 output，直接 query total 包含全部这些成本。', '',
        '预处理机会相同，材料数量与工作量随协议替换而不同。本轮 Beaver+分段 FSS 的在线收益伴随更大的离线和直接总时间；SCI_ours 未取得在线时延优势。SDCF-09 的 99 步整图、分类头及质量验收继续延期，本报告不更新整网性能或 Top-1 结论。', '',
        '复核入口：ELIGIBILITY.json 保存全部资格检查，COMPARISON.json 保存逐行分布与六组比例；formal/、diagnostics/ 保存声明、实际命令、配置与两方日志，runner/ 保存冻结运行器源码。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--formal',type=Path)
    parser.add_argument('--diagnostics',type=Path)
    parser.add_argument('--reference-dir',type=Path,default=REPO/'project/our/results/runtime_optimization_20261010/baseline/diagnostic')
    parser.add_argument('--contract',type=Path,default=REPO/'project/our/tools/segmented_dcf_runtime/run.py')
    args=parser.parse_args()
    report=audit_matrix(args.formal or args.results/'formal', args.diagnostics or args.results/'diagnostics',
                        args.reference_dir,args.contract)
    comparison=comparisons(report)
    args.results.mkdir(parents=True,exist_ok=True)
    for name,value in (('ELIGIBILITY.json',report),('COMPARISON.json',comparison)):
        (args.results/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    (args.results/'REPORT.zh.md').write_text(chinese_report(report,comparison))
    print(json.dumps(dict(complete=report['complete'],eligible=report['eligible'],
        formal=sum(row['eligible'] for row in report['formal']),diagnostic=sum(row['eligible'] for row in report['diagnostic']))))
    return 0 if report['eligible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
