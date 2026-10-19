"""Strict two-party aggregation for the Google-30 comparison.

SCI socket counters and segmented-FSS bridge counters are kept separate. The
reported link total is their union (both parties' sends, exactly once), and
unknown protocol quantities remain ``None`` rather than being inferred.
"""
import statistics


def quality_eligibility(quality, comparison):
    """Return the predeclared accuracy gate used by comparison tables."""
    reasons = []
    if quality.get('samples') != 6798:
        reasons.append('quality does not cover all 6798 fixed valid samples')
    if (not comparison.get('complete') or
            not comparison.get('full_secure_agreement') or
            comparison.get('compared_secure_samples') != 6798):
        reasons.append('full 6798 actual secure backend agreement is pending')
    if comparison.get('full_oracle_match') is False:
        reasons.append('secure output differs from the frozen integer oracle')
    loss = quality.get('loss_pp')
    if loss is None or loss > 1:
        reasons.append('FP32 Top-1 loss exceeds 1.0 pp or is unknown')
    return {'eligible': not reasons, 'reasons': reasons}


def _bridge_value(bridge, prefix, key):
    value = bridge.get(prefix + '_' + key)
    if value is None and isinstance(bridge.get(prefix), dict):
        value = bridge[prefix].get(key)
    return value


def _bridge_components(parties):
    """Validate mirrored bridge counters and return FSS union bytes."""
    result = {}
    for prefix in ('setup', 'offline', 'online'):
        sent = [_bridge_value(p.get('bridge', {}), prefix, 'sent_bytes') for p in parties]
        received = [_bridge_value(p.get('bridge', {}), prefix, 'received_bytes') for p in parties]
        if any(v is None for v in sent + received):
            continue
        if sent[0] != received[1] or sent[1] != received[0]:
            raise ValueError('unbalanced bridge ' + prefix + ' bytes')
        result[prefix + '_link_bytes'] = sent[0] + sent[1]
        times = [_bridge_value(p.get('bridge', {}), prefix, 'ms') for p in parties]
        if all(v is not None for v in times):
            result[prefix + '_seconds'] = max(times) / 1000.0
    # Older bridge logs only expose cumulative counters.  They are still
    # safe to include once, but their phase split remains unavailable.
    if not any(key.endswith('_link_bytes') for key in result):
        sent = [p.get('bridge', {}).get('sent_bytes') for p in parties]
        received = [p.get('bridge', {}).get('received_bytes') for p in parties]
        if all(v is not None for v in sent + received):
            if sent[0] != received[1] or sent[1] != received[0]:
                raise ValueError('unbalanced cumulative bridge bytes')
            result['cumulative_link_bytes'] = sent[0] + sent[1]
    result['available'] = all(prefix + '_link_bytes' in result
                              for prefix in ('setup', 'offline', 'online'))
    if result['available']:
        cumulative_sent = [p.get('bridge', {}).get('sent_bytes') for p in parties]
        cumulative_received = [p.get('bridge', {}).get('received_bytes') for p in parties]
        if all(v is not None for v in cumulative_sent + cumulative_received):
            phase_sent = [sum(_bridge_value(p.get('bridge', {}), prefix, 'sent_bytes')
                              for prefix in ('setup', 'offline', 'online')) for p in parties]
            phase_received = [sum(_bridge_value(p.get('bridge', {}), prefix, 'received_bytes')
                                  for prefix in ('setup', 'offline', 'online')) for p in parties]
            if phase_sent != cumulative_sent or phase_received != cumulative_received:
                raise ValueError('bridge phase counters do not match cumulative counters')
    result['link_bytes'] = sum(value for key, value in result.items()
                               if key.endswith('_link_bytes') and key != 'cumulative_link_bytes')
    result['link_bytes'] += result.get('cumulative_link_bytes', 0)
    return result


def combine(parties):
    """Merge two party records and enforce mirrored communication counts."""
    if len(parties) != 2 or {p['party'] for p in parties} != {1, 2}:
        raise ValueError('requires two successful distinct parties')
    parties = sorted(parties, key=lambda p: p['party'])
    phases = set(parties[0]['phases'])
    if phases != set(parties[1]['phases']):
        raise ValueError('phase boundaries differ')
    merged = {}
    for name in sorted(phases):
        left, right = [p['phases'][name] for p in parties]
        if (left['sent_bytes'] != right['received_bytes'] or
                right['sent_bytes'] != left['received_bytes']):
            raise ValueError('unbalanced all-channel bytes: ' + name)
        merged[name] = {
            'seconds': max(left['seconds'], right['seconds']),
            'sent_bytes': left['sent_bytes'] + right['sent_bytes'],
            'link_bytes': left['sent_bytes'] + right['sent_bytes'],
            'send_calls': left.get('send_calls', 0) + right.get('send_calls', 0),
            'recv_calls': left.get('recv_calls', 0) + right.get('recv_calls', 0),
        }
    query_lists = [p.get('queries', []) for p in parties]
    if not query_lists[0] or len(query_lists[0]) != len(query_lists[1]):
        raise ValueError('missing query observations')
    observations = []
    for index, (left, right) in enumerate(zip(*query_lists)):
        if (left['sent_bytes'] != right['received_bytes'] or
                right['sent_bytes'] != left['received_bytes']):
            raise ValueError('unbalanced query bytes at index ' + str(index))
        row = {
            'seconds': max(left['seconds'], right['seconds']),
            'link_bytes': left['sent_bytes'] + right['sent_bytes'],
            'sci_link_bytes': left['sent_bytes'] + right['sent_bytes'],
            'send_calls': left.get('send_calls', 0) + right.get('send_calls', 0),
            'recv_calls': left.get('recv_calls', 0) + right.get('recv_calls', 0),
        }
        if 'fss_link_bytes' in left and 'fss_link_bytes' in right:
            if left['fss_link_bytes'] != right.get('fss_received_bytes'):
                raise ValueError('unbalanced query FSS bytes at index ' + str(index))
            row['fss_link_bytes'] = left['fss_link_bytes'] + right['fss_link_bytes']
            row['link_bytes'] += row['fss_link_bytes']
        observations.append(row)
    bridge_components = _bridge_components(parties)
    sci_link = sum(value['link_bytes'] for value in merged.values())
    link = sci_link + bridge_components['link_bytes']
    total = max(p['total_seconds'] for p in parties)
    first = max(
        sum(p['phases'].get(name, {}).get('seconds', 0)
            for name in ('application_io', 'initialization_base_ot')) +
        p['queries'][0]['seconds'] for p in parties)
    return {
        'total_seconds': total,
        'link_bytes': link,
        'sci_link_bytes': sci_link,
        'fss_link_bytes': bridge_components['link_bytes'],
        'phases': merged,
        'queries': observations,
        'query_count': len(observations),
        'observed_first_request_seconds': observations[0]['seconds'],
        'observed_second_request_seconds': observations[1]['seconds'] if len(observations) > 1 else None,
        'first_query_with_initialization_seconds': first,
        'amortized_session_seconds_per_query': total / len(observations),
        'amortized_session_bytes_per_query': link / len(observations),
        'offline_seconds': None,
        'offline_reason': 'SCI preprocessing is inseparable inside compute; FSS subset measured separately',
        'online_seconds': None,
        'online_reason': 'request wall time contains lazy preprocessing; no separately scheduled full offline pass',
        'fss_components_included_in_total': bridge_components,
        'query_fss_bytes_observed': all('fss_link_bytes' in row for row in observations),
        'dependent_rounds': None,
        'dependent_rounds_reason': 'socket direction switches are not protocol dependency rounds',
        'peak_rss_kib_per_party': [p['peak_rss_kib'] for p in parties],
        'communication_rule': 'sum both parties sends over SCI plus bridge channels once; never add receives or re-add FSS component bytes',
    }


def distribution(values):
    if not values:
        return None
    return {
        'n': len(values),
        'median': statistics.median(values),
        'mean': statistics.mean(values),
        'min': min(values),
        'max': max(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
    }
