#!/usr/bin/env python3
"""Run SeeDot only on frozen training calibration and archive its selected graph."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from quality import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    if (root / 'frozen_quantization.json').exists():
        raise ValueError('Quantization is already frozen; use a new run for a new policy')
    policy = json.loads((root / 'freeze_policy.json').read_text())
    work = root / 'SeeDot'
    os.chdir(str(work))
    sys.path.insert(0, str(work))
    import seedot.main as main_module
    original = main_module.Main.run
    def captured_run(obj):
        status = original(obj)
        if not status:
            raise RuntimeError('SeeDot quantization failed')
        write_json(root / 'frozen_quantization.json', {
            'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'policy_version': policy['version'],
            'scale_for_x': obj.scaleForX, 'all_scales': obj.allScales,
            'variable_to_bitwidth_map': obj.variableToBitwidthMap,
            'demoted_variables': obj.demotedVarsList, 'demoted_offsets': obj.demotedVarsOffsets,
            'calibration_float_accuracy_pct': obj.flAccuracy,
            'calibration_final_metrics': obj.testingAccuracy,
            'scale_search_metrics': obj.accuracy, 'variable_demote_details': obj.varDemoteDetails,
            'evaluation_read': False,
        })
        return status
    main_module.Main.run = captured_run
    spec = importlib.util.spec_from_file_location('driver', str(work / 'SeeDot-dev.py'))
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    sys.argv = ['SeeDot-dev.py', '-a', 'fastgrnn', '-e', 'fixed', '-d', 'Google-30', '-m', 'red_disagree', '-t', 'EzPC', '-n', '1', '-l', 'info']
    obj = driver.MainDriver()
    obj.parseArgs()
    obj.run()
    artifact = root / 'numeric_artifact'
    artifact.mkdir()
    predictor = work / 'temp/Predictor'
    for path in predictor.iterdir():
        if path.suffix in ['.h', '.cpp'] or path.name == 'Makefile':
            shutil.copy2(str(path), str(artifact))
    shutil.copy2(str(work / 'EzPC/predict.ezpc'), str(artifact))
    shutil.copy2(str(work / 'model/fastgrnn/Google-30/input.sd'), str(artifact))
    shutil.copytree(str(work / 'EzPC/output'), str(artifact / 'compiler_logs'))
    print('FROZEN GRAPH ' + str(artifact))

if __name__ == '__main__':
    main()
