#!/usr/bin/env python3
"""Freeze public folded references and crosscheck the existing plaintext graph."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

import folded_semantics as folded
import sci_semantics
import vector_oracle as vector


def mismatch_count(actual, expected):
    return int(np.count_nonzero(np.asarray(actual) != np.asarray(expected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--fixture-directory', required=True, type=Path)
    parser.add_argument('--encoded-input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    legacy_path = Path(__file__).resolve().parents[1] / 'google30_hybrid/verify_activation_outputs.py'
    spec = importlib.util.spec_from_file_location('legacy_folded_oracle', legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    table = folded.correction_table()
    table_120 = folded.correction_table(120)
    table_checks = {
        'entries': len(table),
        'precision_80_vs_120_mismatches': mismatch_count(table, table_120),
        'legacy_double_table_mismatches': mismatch_count(table, legacy.DELTA),
        'version': folded.table_version(),
    }
    (args.output / 'correction_table.json').write_text(json.dumps({**table_checks, 'values': table}, indent=2) + '\n')

    inputs = np.arange(-32768, 32768, dtype=np.int64)
    mappings = {'input_q7': inputs.astype(np.int16)}
    checks = {}
    for kind, name in [('sigmoid', 'Sigmoid'), ('tanh', 'TanH')]:
        expected = np.asarray([legacy.oracle(name, int(x)) for x in inputs], dtype=np.int64)
        actual = vector.activation(inputs, kind, activation_mode='folded')
        scalar = np.asarray([folded.activation(int(x), kind) for x in inputs], dtype=np.int64)
        checks[kind] = {
            'inputs': len(inputs),
            'legacy_reference_mismatches': mismatch_count(actual, expected),
            'independent_scalar_mismatches': mismatch_count(actual, scalar),
            'range': [int(actual.min()), int(actual.max())],
        }
        mappings[kind] = actual.astype(np.int16)
    np.savez_compressed(args.output / 'native_mappings.npz', **mappings)

    q12_inputs = [-2097152, -65537, -65536, -65535, -32769, -32768, -32767,
                  -65, -64, -63, -1, 0, 1, 63, 64, 65, 32767, 32768, 32769,
                  65535, 65536, 65537, 2097088]
    q12_rows = []
    for x in q12_inputs:
        c = min(abs(x), 65535)
        q12_rows.append({'input_q12': x, 'nonnegative': int(x >= 0), 'abs': abs(x),
                         'below_clip': int(abs(x) < 65536), 'clipped': c,
                         'index': c // 64, 'correction': table[c // 64],
                         'sigmoid_q12': folded.sigma_q12(x),
                         'legacy_sigmoid_q12': legacy.sigma_q12(x)})
    (args.output / 'q12_boundaries.json').write_text(json.dumps(q12_rows, indent=2) + '\n')

    header = args.artifact / 'numeric_artifact/model_fixed.h'
    model = vector.load(header)
    scalar_model = {name: value.tolist() for name, value in model.items()}
    fixtures = json.loads((args.fixture_directory / 'fixture/fixtures.json').read_text())
    fixture_x = np.asarray([row['x'] for row in fixtures['inputs']], dtype=np.int64)
    fixture_h = np.asarray([row['hidden'] for row in fixtures['inputs']], dtype=np.int64)
    secure_sci = json.loads((args.fixture_directory / 'SCI_integer_outputs.json').read_text())
    encoded = np.memmap(args.encoded_input, dtype='<i2', mode='r').reshape(-1, 99, 32)
    batched_x = np.asarray(encoded[:128, 0], dtype=np.int64)
    states = {
        'zero': np.zeros((128, 100), dtype=np.int64),
        'nonzero': ((np.arange(128 * 100).reshape(128, 100) * 37 + 11) % 257) - 128,
    }
    graph_checks = {}
    for mode in vector.ACTIVATION_MODES:
        rows = []
        _, trace = vector.step(fixture_x, fixture_h, model, trace=True, activation_mode=mode)
        fixture_mismatches = 0
        secure_mismatches = 0
        for case in range(len(fixture_x)):
            expected = sci_semantics.step(fixture_x[case].tolist(), fixture_h[case].tolist(),
                                          scalar_model, variant='independent', activation_mode=mode)
            fixture_mismatches += sum(mismatch_count(trace[name][case], value) for name, value in expected.items())
            if mode == 'sci':
                secure_mismatches += sum(mismatch_count(trace[name][case], secure_sci[str(case)][name]) for name in trace)
        (args.output / ('fixture-step-' + mode + '.json')).write_text(json.dumps({
            str(case): {name: values[case].tolist() for name, values in trace.items()}
            for case in range(len(fixture_x))
        }) + '\n')
        for state_name, hidden in states.items():
            _, values = vector.step(batched_x, hidden, model, trace=True, activation_mode=mode)
            mismatches = 0
            for case in range(128):
                expected = sci_semantics.step(batched_x[case].tolist(), hidden[case].tolist(),
                                              scalar_model, variant='independent', activation_mode=mode)
                mismatches += sum(mismatch_count(values[name][case], value) for name, value in expected.items())
            _, baseline = vector.step(batched_x, hidden, model, trace=True, activation_mode='sci')
            pre_names = ('tmp19', 'tmp21', 'tmp23', 'tmp25', 'tmp26', 'tmp27', 'tmp29')
            pre_mismatches = sum(mismatch_count(values[name], baseline[name]) for name in pre_names)
            np.savez_compressed(args.output / ('batch128-step-' + mode + '-' + state_name + '.npz'), **values)
            rows.append({'initial_hidden': state_name, 'samples': 128, 'semantic_tensors': len(values),
                         'integer_values': sum(value.size for value in values.values()),
                         'scalar_vector_mismatches': mismatches,
                         'preactivation_sci_mismatches': pre_mismatches})
        predictions, logits, traces = vector.infer(encoded[:3], model, trace=True, activation_mode=mode)
        full_mismatches = 0
        head_mismatches = 0
        for case in range(3):
            h = [0] * 100
            for t in range(99):
                expected = sci_semantics.step(encoded[case, t].tolist(), h, scalar_model,
                                              variant='independent', activation_mode=mode)
                full_mismatches += sum(mismatch_count(traces[t][name][case], value) for name, value in expected.items())
                h = expected['H']
            fc = sci_semantics.matmul(h, scalar_model['FC'], 16, 8, 8, 16)
            expected_logits = [sci_semantics.signed(value + sci_semantics.trunc0(bias, 3), 8)
                               for value, bias in zip(fc, scalar_model['FCbias'][0])]
            head_mismatches += mismatch_count(logits[case], expected_logits)
            head_mismatches += int(predictions[case] != np.argmax(expected_logits))
        graph_checks[mode] = {
            'prior_three_fixtures_scalar_vector_mismatches': fixture_mismatches,
            'prior_three_fixtures_actual_sci_mismatches': secure_mismatches if mode == 'sci' else None,
            'batch128': rows,
            'full_sequence_smoke_samples': 3, 'full_sequence_steps': 99,
            'full_sequence_tensor_mismatches': full_mismatches,
            'classification_head_mismatches': head_mismatches,
            'predictions': predictions.tolist(), 'logits': logits.tolist(),
        }

    policy = json.loads((args.artifact / 'freeze_policy.json').read_text())
    metadata = {
        'artifact': str(args.artifact.resolve()), 'source_model': policy['source_model'],
        'checkpoint_rule': policy['checkpoint_rule'],
        'frozen_fp32_weights': str((args.artifact / 'fp32_model').resolve()),
        'policy_version': policy['version'], 'frozen_at_utc': policy['frozen_at_utc'],
        'graph': str((args.artifact / 'numeric_artifact/predict.ezpc').resolve()),
        'model_header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
        'normalization': policy['normalization'],
        'normalization_mean': str((args.artifact / 'fp32_model/mean.npy').resolve()),
        'normalization_std': str((args.artifact / 'fp32_model/std.npy').resolve()),
        'quantization_configuration': str((args.artifact / 'frozen_quantization.json').resolve()),
        'validation_input_csv': str((args.artifact / 'heldout_input/X.csv').resolve()),
        'validation_manifest': str((args.artifact / 'manifests/valid_samples.jsonl').resolve()),
        'validation_samples': policy['dataset_audit']['samples']['valid'],
        'quality_loss_threshold_pp': policy['final_loss_threshold_pp'],
        'input_encoding': 'upstream six decimal CSV then int(float(value)*2048), toward zero',
        'secure_input_fixture': str(args.encoded_input.resolve()),
        'scope': 'numeric truth only; full6798 quality evaluation belongs to ticket10',
        'ring_contract': {'input': 'signed16->32, then local x32', 'index': '32->26->10',
                          'tanh_output': 'signed floor32->27->16', 'sigmoid_output': 'Q12 x4 then32->16'},
    }
    (args.output / 'frozen_sources.json').write_text(json.dumps(metadata, indent=2) + '\n')
    numeric_counts = [table_checks['precision_80_vs_120_mismatches'], table_checks['legacy_double_table_mismatches']]
    numeric_counts += [row[key] for row in checks.values() for key in ('legacy_reference_mismatches', 'independent_scalar_mismatches')]
    numeric_counts += [int(row['sigmoid_q12'] != row['legacy_sigmoid_q12']) for row in q12_rows]
    for mode, row in graph_checks.items():
        numeric_counts.extend([row['prior_three_fixtures_scalar_vector_mismatches'], row['full_sequence_tensor_mismatches'], row['classification_head_mismatches']])
        if mode == 'sci': numeric_counts.append(row['prior_three_fixtures_actual_sci_mismatches'])
        for batch_row in row['batch128']:
            numeric_counts.extend([batch_row['scalar_vector_mismatches'], batch_row['preactivation_sci_mismatches']])
    report = {'passed': not any(numeric_counts), 'table': table_checks, 'native_mappings': checks,
              'q12_boundary_cases': len(q12_rows), 'graphs': graph_checks,
              'secure_samples': 0, 'full_holdout_evaluated': False}
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'table_version': table_checks['version'],
                      'native_inputs_per_activation': len(inputs), 'graph_modes': list(graph_checks)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
