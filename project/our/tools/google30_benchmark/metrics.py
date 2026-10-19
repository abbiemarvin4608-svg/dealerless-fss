"""Public result aggregation: bytes are both parties' sends, never sends+receives."""
import statistics


def summarize_parties(parties):
    if len(parties) != 2 or {p['party'] for p in parties} != {1, 2}:
        raise ValueError('two distinct successful party measurements required')
    parties = sorted(parties, key=lambda p: p['party'])
    phases = set(parties[0]['phases'])
    if phases != set(parties[1]['phases']):
        raise ValueError('party phase boundaries disagree')
    merged = {}
    for name in sorted(phases):
        values = [p['phases'][name] for p in parties]
        if any(values[i]['sent_bytes'] != values[1-i]['received_bytes'] for i in [0, 1]):
            raise ValueError('unbalanced link byte counts at ' + name)
        merged[name] = dict(seconds=max(v['seconds'] for v in values),
                           sent_bytes=sum(v['sent_bytes'] for v in values))
    total = max(p['total_seconds'] for p in parties)
    query_lists = [p.get('queries') for p in parties]
    observed_queries = None
    if any(q is not None for q in query_lists) and (not all(query_lists) or len(query_lists[0]) != len(query_lists[1])):
        raise ValueError('party query counts disagree')
    if all(query_lists):
        observed_queries = len(query_lists[0])
        if any(len(p.get('queries', [])) != observed_queries for p in parties):
            raise ValueError('party query counts disagree')
        for idx in range(observed_queries):
            if any(query_lists[i][idx]['sent_bytes'] != query_lists[1-i][idx]['received_bytes'] for i in [0,1]):
                raise ValueError('unbalanced query bytes')
            for key in ('seconds','sent_bytes','received_bytes'):
                if query_lists[0][idx][key] < 0 or query_lists[1][idx][key] < 0:
                    raise ValueError('negative query metric')
    first_query = total if not observed_queries or observed_queries == 1 else max(
        p['phases'].get('application_io',{}).get('seconds',0) +
        p['phases'].get('initialization_base_ot',{}).get('seconds',0) + p['queries'][0]['seconds'] for p in parties)
    link_bytes = sum(x['sent_bytes'] for x in merged.values())
    return dict(phases=merged, first_query_seconds=first_query, total_seconds=total,
        link_bytes=link_bytes,
        online_seconds=merged.get('online_compute', {}).get('seconds'),
        online_scope='compute only; includes inseparable OT-extension/preprocessing and conversions; I/O shown separately',
        offline_seconds=None, offline_status='not_separable',
        offline_reason='SCI generates OT-extension material inside online operations; base OT separately measured',
        dependent_rounds=None, rounds_status='unavailable',
        rounds_reason='library direction transitions / I/O calls are not protocol dependency rounds',
        io_calls=sum(v['send_calls']+v['recv_calls'] for p in parties for v in p['phases'].values()),
        peak_rss_kib_per_party=[p['peak_rss_kib'] for p in parties],
        amortization=dict(queries=observed_queries or 1, rule='fresh process and base OT for every measured query; no material reused' if not observed_queries else 'base OT/connection initialized once; every query uses fresh one-use extension material',
            initialization_excluded_seconds=max(p['total_seconds']-p['phases'].get('initialization_base_ot',{}).get('seconds',0) for p in parties),
            measured_session_seconds_per_query=(total / observed_queries if observed_queries else None),
            measured_session_bytes_per_query=(link_bytes / observed_queries if observed_queries else None),
            measured_second_query_seconds=(max(p['queries'][1]['seconds'] for p in parties) if observed_queries and observed_queries >= 2 else None),
            status='observed persistent session' if observed_queries else 'component subtraction only, not an observed persistent-session query latency'),
        batch=1, throughput_queries_per_second=(observed_queries/total if observed_queries else 1/total),
        throughput_scope='sequential queries / complete session including initialization; batch=1; no batched executor measured')


def distribution(values):
    return dict(n=len(values), median=statistics.median(values), mean=statistics.mean(values),
        stdev=statistics.stdev(values) if len(values)>1 else 0,
        minimum=min(values), maximum=max(values))
