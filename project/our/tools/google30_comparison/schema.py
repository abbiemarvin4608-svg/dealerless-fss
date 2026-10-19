"""Typed result rows for the two-backend Google-30 comparison.

No row is eligible for a performance ranking until it is tied to a complete
6798-sample secure evaluation and the independent quality gate.  This module
also accepts partial run artifacts so an interrupted experiment remains
visible as an explicitly ineligible observation.
"""
try:
    from .metrics import quality_eligibility, distribution
except ImportError:  # direct unittest discovery
    from metrics import quality_eligibility, distribution

REQUIRED_SAMPLES = 6798
QUALITY_LIMIT_PP = 1.0


def _repeat_groups(rows):
    groups = {}
    for row in rows:
        if (not row.get('eligible') or row.get('status') != 'complete' or
                row.get('run_kind') != 'repeat' or row.get('query_count') != 1):
            continue
        config_id, group_id = row.get('config_id'), row.get('comparison_group_id')
        if not config_id or not group_id:
            continue
        if not row.get('measurement_artifact') or row.get('total_seconds') is None or row.get('link_bytes') is None:
            continue
        groups.setdefault((config_id, group_id), []).append(row)
    return groups


def rankable_repeat_group(rows):
    """Require all supplied rows to be one auditable three-repeat q1 group."""
    groups = _repeat_groups(rows)
    if len(groups) != 1:
        return False
    group = next(iter(groups.values()))
    return (len(group) == len(rows) and len(group) >= 3 and
            len({row['measurement_artifact'] for row in group}) == len(group))


def normalize_comparison(value):
    """Normalize ticket-07 ledgers and pairwise comparison JSON."""
    value = value or {}
    complete = bool(value.get('complete', value.get('full_secure_evaluation', False)))
    compared = value.get('compared_secure_samples')
    if compared is None:
        compared = value.get('completed_samples', value.get('SCI_secure_samples', 0))
    agreement = value.get('full_secure_agreement')
    if agreement is None and 'prediction_differences' in value:
        agreement = complete and not value.get('prediction_differences')
    oracle = value.get('full_oracle_match')
    mismatches = value.get('oracle_mismatches', [])
    mismatch_free = (not mismatches if not isinstance(mismatches, dict)
                     else all(not entries for entries in mismatches.values()))
    if oracle is None and 'oracle_mismatches' in value:
        oracle = complete and mismatch_free
    return {
        'complete': complete,
        'full_secure_agreement': bool(agreement) if agreement is not None else False,
        'compared_secure_samples': int(compared or 0),
        'full_oracle_match': bool(oracle) if oracle is not None else False,
        'oracle_mismatches': mismatches,
        'prediction_differences': value.get('prediction_differences', []),
    }


def quality_fields(quality):
    quality = quality or {}
    logits = quality.get('logit_error', {}) or {}
    return {
        'quality_samples': quality.get('samples'),
        'fp32_top1_pct': quality.get('fp32_top1_pct'),
        'fixed_top1_pct': quality.get('fixed_top1_pct'),
        'loss_pp': quality.get('loss_pp'),
        'prediction_agreement_pct': quality.get('prediction_agreement_pct'),
        'logit_mae': logits.get('mae'),
        'logit_rmse': logits.get('rmse'),
        'logit_p95_abs': logits.get('p95_abs'),
        'logit_max_abs': logits.get('max_abs'),
        'logit_units': logits.get('units'),
        'quality_source': quality.get('quality_source'),
        'quality_backend_identities': quality.get('backend_identities'),
        'quality_numerical_contract': quality.get('numerical_contract'),
    }


def result_row(run, quality=None, comparison=None, *, backend=None, track=None,
               network=None, config_id=None):
    """Create one stable table row from an aggregate run result."""
    backend_name = backend or run.get('backend')
    comparison = normalize_comparison(comparison or run.get('comparison'))
    q = quality_fields(quality or run.get('quality'))
    gate = quality_eligibility(
        {'samples': q['quality_samples'], 'loss_pp': q['loss_pp']}, comparison)
    reasons = list(gate['reasons'])
    if run.get('status') != 'complete':
        reasons.insert(0, 'performance run status is ' + str(run.get('status')))
    if run.get('run_kind') == 'warmup':
        reasons.insert(0, 'warmup is excluded from performance ranking')
    identities = q.get('quality_backend_identities') or {}
    quality_identity = run.get('quality_backend_identity')
    if identities and backend_name not in identities:
        reasons.append('quality artifact has no identity for backend ' + str(backend_name))
    elif identities and not quality_identity:
        reasons.append('performance run is missing secure quality identity binding')
    elif identities and quality_identity != identities.get(backend_name):
        reasons.append('secure quality identity does not match backend ' + str(backend_name))
    return {
        'schema_version': 'google30-comparison-row-v1',
        'backend': backend_name,
        'track': track or run.get('track', 'same_graph'),
        'network': network or run.get('network', {}).get('name', run.get('network')),
        'config_id': config_id or run.get('config_id'),
        'comparison_signature': run.get('comparison_signature'),
        'comparison_group_id': run.get('comparison_group_id', run.get('comparison_signature')),
        'backend_identity': run.get('backend_identity'),
        'quality_backend_identity': run.get('quality_backend_identity'),
        'status': run.get('status', 'unknown'),
        'run_kind': run.get('run_kind', run.get('tag')),
        'eligible': bool(gate['eligible'] and not reasons and
                         run.get('status') == 'complete' and
                         run.get('run_kind') != 'warmup'),
        'ineligibility_reasons': reasons,
        **q,
        'secure_required_samples': REQUIRED_SAMPLES,
        'secure_compared_samples': comparison['compared_secure_samples'],
        'secure_complete': comparison['complete'],
        'full_secure_agreement': comparison['full_secure_agreement'],
        'full_oracle_match': comparison['full_oracle_match'],
        'secure_prediction_differences': comparison['prediction_differences'],
        'secure_oracle_mismatches': comparison['oracle_mismatches'],
        'total_seconds': run.get('total_seconds'),
        'first_query_seconds': run.get('first_query_with_initialization_seconds', run.get('first_query_seconds')),
        'second_query_seconds': run.get('observed_second_request_seconds', run.get('second_query_seconds')),
        'session_seconds_per_query': run.get('amortized_session_seconds_per_query'),
        'link_bytes': run.get('link_bytes'),
        'sci_link_bytes': run.get('sci_link_bytes'),
        'fss_link_bytes': run.get('fss_link_bytes'),
        'offline_seconds': run.get('offline_seconds'),
        'online_seconds': run.get('online_seconds'),
        'dependent_rounds': run.get('dependent_rounds'),
        'peak_rss_kib_party1': (run.get('peak_rss_kib_per_party') or [None, None])[0],
        'peak_rss_kib_party2': (run.get('peak_rss_kib_per_party') or [None, None])[1],
        'query_count': run.get('query_count', run.get('queries')),
        'measurement_artifact': run.get('artifact'),
        'comparison_artifact': run.get('comparison_artifact'),
    }


def summarize_rows(rows):
    """Summarize only eligible repeat rows; keep unavailable status explicit."""
    result = {}
    keys = sorted({(row['backend'], row['track'], row['network']) for row in rows})
    for backend, track, network in keys:
        selected = [row for row in rows if
                    (row['backend'], row['track'], row['network']) == (backend, track, network)]
        eligible = [row for row in selected if row['eligible'] and row['run_kind'] == 'repeat'
                    and row.get('query_count') == 1]
        groups = _repeat_groups(eligible)
        complete_groups = [group for group in groups.values() if rankable_repeat_group(group)]
        rankable = rankable_repeat_group(eligible)
        rankable_rows = eligible if rankable else []
        result['%s/%s/%s' % (backend, track, network)] = {
            'backend': backend, 'track': track, 'network': network,
            'runs': len(selected), 'eligible_repeats': len(eligible),
            'complete_repeat_groups': len(complete_groups),
            'configuration_groups': [dict(config_id=key[0], comparison_group_id=key[1],
                rankable=rankable_repeat_group(group), runs=len(group),
                latency_seconds=distribution([r['total_seconds'] for r in group]),
                link_bytes=distribution([r['link_bytes'] for r in group]))
                for key, group in groups.items()],
            'latency_seconds': distribution([r['total_seconds'] for r in rankable_rows
                                             if r['total_seconds'] is not None]),
            'link_bytes': distribution([r['link_bytes'] for r in rankable_rows
                                        if r['link_bytes'] is not None]),
            'rankable': rankable,
            'rankability_reason': None if rankable else
                                  'requires three distinct eligible repeats of exactly one configuration; see groups',
        }
    return result
