"""Complete prefixes must remain complete across fresh query boundaries."""
import copy
import json
import unittest

from audit_folded_residency import party_checks


class FoldedResidencyTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(batch=128, prefetch_bytes=536870912,
                           memory_limit_bytes_per_party=1610612736,
                           prefetch_mode='prepared_prefix')
        self.party = dict(conversion_calls=[], conversion_offline_calls=[],
                          conversion_preprocess_details=[], conversion_details=[])
        reserve = 25600 * 8192 + (8 << 20)
        budget = self.config['prefetch_bytes'] - reserve
        self.party['conversion_prefetch_budgets'] = [dict(context=0, contexts=1,
            context_budget_bytes=budget, party_budget_bytes=budget,
            configured_party_budget_bytes=536870912, dpf_reserved_bytes=reserve)]
        self.lines = [('FOLDED_ACTIVATION_RESERVE', dict(resident_budget_bytes=reserve,
            sigmoid_count=12800, tanh_count=12800, per_party_address_space_bytes=1610612736,
            includes_allocator_headroom=True))]
        for query in range(2):
            self.lines += [('CONVERSION_PREFETCH', dict(mode='prepared_prefix', calls=2,
                resident_bytes=100, budget_bytes=budget, workspace_reserve_bytes=1000,
                address_space_bytes=100000, charged_phase='offline'))]
            for index in range(2):
                identity = dict(query=query, index=index, slot=0)
                for name in ('conversion_calls', 'conversion_offline_calls'):
                    self.party[name].append(dict(identity, context=0))
                generated = dict(ordinal=index, fresh=True, dealerless=True)
                self.party['conversion_preprocess_details'].append(generated)
                self.lines.append(('CONVERSION_PREPROCESS_DETAIL', generated))
                self.party['conversion_details'].append(dict(identity, prefetched=1,
                    material_load_ms=0, prefetch_wait_ms=0, material_preload_ms=0.01,
                    resident_bytes=100 - index * 50, prefetch_budget_bytes=budget))
            self.lines += [('MATERIAL_READY', dict(query=query)),
                           ('METRIC', dict(query=query, phase='query_total'))]

    def check(self):
        log = '\n'.join(prefix + ' ' + json.dumps(row) for prefix, row in self.lines)
        return party_checks(log, self.party, self.config, 2)[0]

    def test_two_fresh_complete_prefixes_pass(self):
        self.assertTrue(all(self.check().values()))

    def test_partial_or_missing_prefix_is_not_successful_residency(self):
        self.lines[1][1]['calls'] = 1
        self.assertFalse(self.check()['complete_context_prefixes'])
        self.lines.pop(1)
        self.assertFalse(self.check()['complete_context_prefixes'])

    def test_online_load_wait_and_missing_details_fail(self):
        for field in ('material_load_ms', 'prefetch_wait_ms', 'prefetched'):
            party = copy.deepcopy(self.party)
            self.party['conversion_details'][0][field] = 0.5
            self.assertFalse(self.check()['no_online_load_or_wait'])
            self.party = party
        self.party['conversion_details'].pop()
        self.assertFalse(self.check()['exact_conversion_coverage'])

    def test_late_prefix_reused_query_and_resource_overflow_fail(self):
        boundary = next(i for i, (kind, _) in enumerate(self.lines) if kind == 'MATERIAL_READY')
        self.lines[1], self.lines[boundary] = self.lines[boundary], self.lines[1]
        self.assertFalse(self.check()['prefix_before_input'])
        self.lines[1], self.lines[boundary] = self.lines[boundary], self.lines[1]
        next(row for kind, row in self.lines if kind == 'METRIC')['query'] = 1
        self.assertFalse(self.check()['complete_context_prefixes'])
        self.lines[1][1]['address_space_bytes'] = 1610612737
        self.assertFalse(self.check()['prefix_resources_bounded'])

    def test_joint_folded_reserve_is_charged_to_outer_budget(self):
        self.party['conversion_prefetch_budgets'][0]['dpf_reserved_bytes'] = 0
        self.assertFalse(self.check()['joint_budget_accounted'])

    def test_private_sign_reserve_reduces_the_outer_prefix_allocation(self):
        self.config.update(folded_sign_backend='segmented_fss', folded_sign_segments=4)
        self.assertFalse(self.check()['joint_budget_accounted'])
        reserve = 25600 * (8192 + 1024 + 256 * 4) + (8 << 20)
        budget = self.config['prefetch_bytes'] - reserve
        self.lines[0][1]['resident_budget_bytes'] = reserve
        allocation = self.party['conversion_prefetch_budgets'][0]
        allocation.update(context_budget_bytes=budget, party_budget_bytes=budget, dpf_reserved_bytes=reserve)
        for kind, row in self.lines:
            if kind == 'CONVERSION_PREFETCH':
                row['budget_bytes'] = budget
        for row in self.party['conversion_details']:
            row['prefetch_budget_bytes'] = budget
        self.assertTrue(all(self.check().values()))

    def test_preprocessing_ordinals_are_fresh_and_before_ready(self):
        self.lines[2][1]['ordinal'] = 1
        self.assertFalse(self.check()['exact_conversion_coverage'])
        self.lines[2][1]['ordinal'] = 0
        boundary = next(i for i, (kind, _) in enumerate(self.lines) if kind == 'MATERIAL_READY')
        self.lines[2], self.lines[boundary] = self.lines[boundary], self.lines[2]
        self.assertFalse(self.check()['prefix_before_input'])


if __name__ == '__main__':
    unittest.main()
