#!/usr/bin/env python3
"""Export named numeric arguments from this run's generated EzPC graph."""
import argparse
import json
from pathlib import Path
import re
from quality import write_json

ADD = 'I J shrA shrB shrC demote bwA bwB bwTemp bwC A B C'.split()
BINARY = 'I J shrA shrB demote bwA bwB bwTemp bwC A B C'.split()
SIGNATURES = {
    'MatMul': 'I K J shrA shrB H1 H2 demote bwA bwB bwTemp bwC A B C T'.split(),
    'MatAdd': ADD, 'MatSubBroadCastA': ADD, 'MatAddBroadCastB': ADD,
    'Sigmoid': 'I J scale_in scale_out bwA bwB A B'.split(),
    'TanH': 'I J scale_in scale_out bwA bwB A B'.split(),
    'MulCir': BINARY, 'ScalarMul': BINARY,
    'ArgMax': 'I J bwA bwIndex A Index'.split(),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    frozen = json.loads((root / 'frozen_quantization.json').read_text())
    code = (root / 'numeric_artifact/predict.ezpc').read_text()
    cpp = (root / 'numeric_artifact/seedot_fixed.cpp').read_text()
    contract = json.loads((root / 'numeric_contract.json').read_text())
    types = {name: int(bits) for bits, name in re.findall(r'\bint(8|16|32)_t\s+(\w+)\s*(?:\[|;)', cpp)}
    records = []
    in_recurrence = False
    for line in code.splitlines():
        if line == '\tfor i=[0:99] {':
            in_recurrence = True
        elif in_recurrence and line == '\t};':
            in_recurrence = False
        match = re.fullmatch(r'\s*(\w+)\(([^;]+)\);', line)
        if match is None or match.group(1) not in SIGNATURES:
            continue
        name = match.group(1)
        args = [a.strip() for a in match.group(2).split(',')]
        if len(args) != len(SIGNATURES[name]):
            raise ValueError('Unexpected ' + name + ' signature')
        values = {key: int(value) if re.fullmatch(r'-?\d+', value) else value
                  for key, value in zip(SIGNATURES[name], args)}
        tensors = {}
        for key in ['A', 'B', 'C', 'T', 'Index']:
            if key in values and isinstance(values[key], str):
                tensor = values[key]
                base = re.sub(r'bw\d+$', '', tensor)
                tensors[key] = {'tensor': tensor, 'scale': frozen['all_scales'].get(base),
                                'bits': values.get('bw' + key, values.get('bwTemp') if key == 'T' else None)}
        record = {'ordinal': len(records) + 1, 'operation': name,
                  'calls_per_inference': 99 if in_recurrence else 1,
                  'named_arguments': values, 'tensors': tensors}
        if name == 'MatMul':
            record.update(product_bits=values['bwTemp'], accumulator_bits=values['bwTemp'], accumulation_length=values['K'],
                          rounding='C++ integer division toward zero at each reduction branch for H1 levels; H2 further unscaled reduction levels; final divisions shrA, shrB, demote sequentially')
        elif name in ['MulCir', 'ScalarMul']:
            record.update(product_bits=values['bwTemp'], accumulator_bits=None,
                          rounding='product then sequential integer divisions shrA, shrB, demote toward zero')
        elif name in ['Sigmoid', 'TanH']:
            record.update(elements_per_inference=record['calls_per_inference'] * values['I'] * values['J'],
                          rounding='SeeDot FLOATEXP float math then cast toward zero to bwB',
                          function_input_fraction_bits=(values['scale_in'].bit_length() - 1),
                          function_output_fraction_bits=(values['scale_out'].bit_length() - 1))
        elif name == 'ArgMax':
            record.update(comparison_input_bits=values['bwA'], comparisons_per_inference=values['I'] * values['J'] - 1,
                          tie_rule='first maximum; strict greater update',
                          sci_port_note='must sign-extend 8-bit logits before SCI difference-based comparison; 8-bit index reconstruction')
        else:
            record.update(accumulator_bits=values['bwTemp'],
                          rounding='divide A/shrA and B/shrB toward zero, divide each by shrC, add/subtract, divide by demote')
        records.append(record)
    if len(records) != 18:
        raise ValueError('Expected 15 recurrent and 3 classifier call sites, got %d' % len(records))
    public_constants = {}
    for name, tensor in [('zeta', 'tmp9bw16'), ('nu', 'tmp10bw16'), ('one', 'tmp32')]:
        assignments = re.findall(r'^\s*' + tensor + r'\s*=\s*(-?\d+)L;', code, re.M)
        if not assignments:
            raise ValueError('Missing compiled scalar constant: ' + name)
        public_constants[name] = int(assignments[-1])
    contract.update(operators=records, cpp_declared_storage_bits=types,
                    timesteps=99, recurrent_call_sites=15, classifier_call_sites=3,
                    narrowing='SATURATE is disabled in this generated artifact. Native g++ signed casts used; this is not the CNN saturated-writeback contract.',
                    secure_backend_contract_status='unvalidated; exported EzPC interface fixes widths/scales but SCI approximations and truncations need ticket07 validation',
                    actual_fss_invocations=0,
                    comparison_domain_note='16-bit activation I/O is not evidence of 16-bit internal SCI comparison domains; this artifact runs no SCI/FSS backend',
                    model_privacy_boundary={
                        'scope':'declared EzPC inputs and embedded constants; plaintext quality evaluation does not validate a secure backend',
                        'ownership':'SERVER input declarations cover matrix weights and biases; CLIENT supplies X; learned zeta/nu are public compiled constants',
                        'server_private_parameters':['W1','W2','U1','U2','Bg','Bh','FC','FCbias'],
                        'public_model_scalars':{name:public_constants[name] for name in ['zeta','nu']},
                        'public_program_constants':{'one':public_constants['one']},
                        'backend_requirement':'preserve this public-parameter boundary in same-graph comparisons; hiding these scalars requires a separate secure implementation and new cost accounting'})
    write_json(root / 'numeric_contract.json', contract)
    print('Exported %d operator sites from the new graph' % len(records))

if __name__ == '__main__':
    main()
