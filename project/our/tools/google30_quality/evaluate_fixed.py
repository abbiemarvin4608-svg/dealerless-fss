#!/usr/bin/env python3
"""Run the exact frozen SeeDot C++ graph, additionally capturing pre-ArgMax logits."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from quality import write_json

RUNNER = r'''
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "datatypes.h"
#include "predictors.h"
int64_t quality_logits[30];
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream source(argv[1]);
    std::ofstream target(argv[2]);
    if (!source || !target) return 3;
    std::vector<MYINT> values(3168);
    std::vector<MYINT*> pointers(3168);
    for (int i=0; i<3168; ++i) pointers[i] = &values[i];
    std::string line;
    while (std::getline(source, line)) {
        std::istringstream row(line);
        std::string value;
        for (int i=0; i<3168; ++i) {
            if (!std::getline(row, value, ',')) return 4;
            const double encoded = std::trunc(std::ldexp(std::stod(value), -scaleForX));
            if (encoded < -32768 || encoded > 32767) return 5;
            values[i] = static_cast<MYINT>(encoded);
        }
        if (std::getline(row, value, ',')) return 6;
        int32_t prediction = -1;
        seedotFixed(pointers.data(), &prediction);
        target << prediction;
        for (auto logit : quality_logits) target << ',' << logit;
        target << '\n';
    }
    return target.good() ? 0 : 7;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reproduction', type=Path, required=True)
    args = parser.parse_args()
    root, repro = args.run.resolve(), args.reproduction.resolve()
    frozen = json.loads((root / 'frozen_quantization.json').read_text())
    artifact = root / 'numeric_artifact'
    code = (artifact / 'seedot_fixed.cpp').read_text()
    match = re.search(r'ArgMax<[^>]+>\(&(?P<tensor>\w+)\[0\]\[0\], 1, 30, &\w+\);', code)
    if match is None or len(re.findall(r'\bArgMax<', code)) != 1:
        raise ValueError('Expected one final 30-class ArgMax')
    tensor = match.group('tensor')
    base_name = re.sub(r'_\d+$', '', tensor)
    scale = int(frozen['all_scales'][base_name])
    instrumented = code.replace(
        'using namespace std;', 'extern int64_t quality_logits[30];\nusing namespace std;', 1)
    capture = 'for (int c=0; c<30; ++c) quality_logits[c] = ' + tensor + '[0][c];\n\t\t'
    instrumented = instrumented.replace(match.group(0), capture + match.group(0), 1)
    build = root / 'fixed_evaluator'
    build.mkdir(exist_ok=True)
    (build / 'instrumented.cpp').write_text(instrumented)
    (build / 'runner.cpp').write_text(RUNNER)
    command = ['g++', '-std=c++11', '-O3', '-Wno-narrowing', '-I', str(artifact),
               str(build / 'runner.cpp'), str(build / 'instrumented.cpp'), str(artifact / 'library_fixed.cpp'),
               '-o', str(build / 'evaluate')]
    subprocess.run(command, check=True)
    # First establish that capture/runner preserves the selected Predictor's
    # 512-row calibration predictions. This check is independent of valid.
    calibration_csv = root / 'SeeDot/temp/Predictor/input/X.csv'
    subprocess.run([str(build / 'evaluate'), str(calibration_csv), str(build / 'calibration_logits.csv')], check=True)
    calibration_result = np.loadtxt(str(build / 'calibration_logits.csv'), delimiter=',', dtype=np.int64)
    labels_cal = np.loadtxt(str(root / 'SeeDot/temp/Predictor/input/Y.csv'), delimiter=',', dtype=np.int64)
    expected_cal = labels_cal.copy()
    log = (root / 'SeeDot/temp/Predictor/output/fixed/prediction-info-testing.txt').read_text()
    for row, prediction in re.findall(r'Main: Incorrect prediction for input (\d+) element 0\. Predicted (-?\d+) Expected \d+', log):
        expected_cal[int(row) - 1] = int(prediction)
    if not np.array_equal(expected_cal, calibration_result[:, 0]):
        raise ValueError('Evaluator differs from uninstrumented frozen SeeDot Predictor')
    model = root / 'fp32_model'
    mean, std = [np.load(str(model / (name + '.npy'))) for name in ['mean', 'std']]
    raw = np.load(str(repro / 'data/Google-30-processed/valid.npy'), mmap_mode='r')
    input_dir = root / 'heldout_input'
    input_dir.mkdir(exist_ok=True)
    with (input_dir / 'X.csv').open('w') as stream:
        for start in range(0, len(raw), 128):
            values = (raw[start:start + 128, 1:] - mean) / std
            np.savetxt(stream, values, fmt='%.6f', delimiter=',')
    np.savetxt(str(input_dir / 'Y.csv'), raw[:, 0], fmt='%d')
    subprocess.run([str(build / 'evaluate'), str(input_dir / 'X.csv'), str(build / 'valid_logits.csv')], check=True)
    result = np.loadtxt(str(build / 'valid_logits.csv'), dtype=np.int64, delimiter=',')
    if result.shape != (len(raw), 31):
        raise ValueError('Incomplete fixed output')
    if not np.array_equal(result[:, 0], result[:, 1:].argmax(1)):
        raise ValueError('ArgMax output differs from captured logits')
    np.save(str(root / 'fixed_predictions.npy'), result[:, 0])
    np.save(str(root / 'fixed_logits_integer.npy'), result[:, 1:])
    np.save(str(root / 'fixed_logits.npy'), np.ldexp(result[:, 1:].astype(np.float64), scale))
    contract = {
        'version': 'google30-independent-quality-v1', 'graph': 'numeric_artifact/predict.ezpc',
        'x86_graph': 'numeric_artifact/seedot_fixed.cpp', 'weights': 'numeric_artifact/model_fixed.h',
        'tensor_scales_and_demotions': 'frozen_quantization.json',
        'real_value_rule': 'signed_integer * 2**scale; negative scale is fractional precision',
        'input': {'bits': 16, 'scale': frozen['scale_for_x'], 'csv_decimals': 6, 'rounding': 'truncate toward zero'},
        'logits': {'tensor': base_name, 'cpp_tensor': tensor, 'scale': scale,
                   'bits': int(re.search(r'_(\d+)$', tensor).group(1))},
        'overflow_and_arithmetic': 'unmodified upstream generated C++ and library_fixed.h; explicit operation templates and rescaling constants preserved',
        'activation_reference': 'SeeDot x86 FLOATEXP implementation; NOT SCI intermediate integer equality evidence',
        'fixed_evaluator_calibration_agreement': {'samples': len(expected_cal), 'matching': len(expected_cal)},
        'execution_mode': 'plaintext_quality_only', 'secure_execution': False,
    }
    write_json(root / 'numeric_contract.json', contract)
    print('Fixed complete: %d samples, logit scale=%d' % (len(raw), scale), flush=True)

if __name__ == '__main__':
    main()
