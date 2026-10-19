#!/usr/bin/env python3
"""Inventory the migration inputs; never run experiments or change Git/files elsewhere."""
import csv
from datetime import datetime, timezone
from email.parser import Parser
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUR = REPO / 'project/our'
REPRO = Path('/home/jiang/EzPC/SIRNN/reproduction')
SKIP = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'mpc_venv',
        'seedot-venv', 'preprocess-venv', '.round_trace_native', 'data', 'models', 'results',
        'node_modules', 'CMakeFiles', 'site-packages', 'temp', 'SeeDot', 'external_patches'}
SUFFIXES = {'.py', '.sh', '.c', '.cpp', '.cc', '.cu', '.h', '.hpp', '.in',
            '.cmake', '.md', '.txt', '.json', '.map', '.patch', '.tex', '.bib',
            '.sty', '.cls', '.bst', '.natvis'}
OPTIONAL_PATHS = {
    'project/our/tools/precision/test_speech_commands.py',
    'project/our/tools/precision/test_deep_lstm.py',
    'project/our/tools/precision/test_lstm_e0.py',
    'project/our/tools/precision/audio_features.py',
    'project/our/tools/precision/fixed_point.py',
    'project/our/tools/precision/run_e0_uniform_storage.py',
    'project/our/tools/precision/run_e0_ring.py',
    'project/our/tools/precision/run_e0_q_all.py',
    'project/our/tools/precision/run_relu_precision_ablation.py',
    'project/our/tools/precision/run_relu_maxpool_precision.py',
    'project/our/tools/precision/test_run_e0_ring.py',
    'project/our/tools/precision/test_run_e0_uniform_storage.py',
    'project/our/tools/precision/test_run_relu_precision_ablation.py',
    'project/our/tools/precision/test_run_relu_maxpool_precision.py',
    'project/our/tools/precision/test_sigma_activation_lut.py',
    'project/our/tools/precision/validate_uniform_ring_network.py',
}


def command(args, cwd=REPO, binary=False):
    result = subprocess.run(args, cwd=cwd, capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode().strip()


def write_tsv(name, fields, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def allocated_bytes(path):
    if not path.exists():
        return ''
    return int(command(['du', '-s', '-B1', '--', str(path)]).split()[0])


def repo_path(text):
    path = Path(text)
    return path if path.is_absolute() else REPO / path


def sources(root):
    if root.is_file():
        yield root
        return
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = sorted(name for name in children
                             if name not in SKIP and not name.startswith('build')
                             and not name.endswith('-build') and name != 'final-build')
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_symlink() or name == '.git' or '.bak.' in name:
                continue
            if path.suffix in SUFFIXES or name in {'LICENSE', 'NOTICE', 'COPYING'} or not path.suffix:
                # Extensionless vendor headers are source too; executables are not.
                with path.open('rb') as stream:
                    if stream.read(4) == b'\x7fELF':
                        continue
                yield path


def main():
    tracked = set(command(['git', 'ls-files', '-z'], binary=True).decode().split('\0')) - {''}
    modified = set(command(['git', 'diff', 'HEAD', '--name-only', '-z'], binary=True).decode().split('\0')) - {''}
    selected = {}
    roots = [('project/our/src', 'ours_core'), ('project/our/test', 'protocol_tests'),
             ('project/our/scripts', 'project_scripts'), ('project/our/docs', 'project_docs'),
             ('project/dealerless-FSS_public', 'reference_dfss'), ('paper', 'paper_sources')]
    roots += [('project/our/tools/' + child.name, 'tools/' + child.name)
              for child in sorted((OUR / 'tools').iterdir()) if child.is_dir()]
    # The historical baseline is inventoried, but omitted from the recommended source whitelist.
    roots += [('project/our/Baselines', 'legacy_baseline_optional')]
    experiments = ['online_optimization_20261014', 'offline_optimization_20261014',
                   'paper_folded_activation_20261013/ticket11', 'paper_folded_activation_20261014',
                   'precision_motivation_final', 'imagenet_resnet_20261015',
                   'sci_sirnn_preprocessed_20261012']
    roots += [('project/our/results/' + name, 'experiment/' + name) for name in experiments]
    roots += [('.scratch/sci-sirnn-preprocessed-baseline', 'preprocessing_spec')]
    roots += [('.scratch/sci-fss-runtime-optimization', 'runtime_optimization_spec'),
              ('.scratch/paper-sigmoid-tanh-lut-replacement', 'folded_activation_spec'),
              ('.scratch/multiwidth-dealerless-inference', 'multiwidth_experiment_spec')]
    for root, component in roots:
        for path in sources(REPO / root):
            relative = path.relative_to(REPO).as_posix()
            # Result logs, per-query JSON, and metrics belong in evidence, not the source package.
            if component.startswith('experiment/') and path.suffix not in {'.py', '.sh', '.md'}:
                continue
            # Do not recursively include generated graph copies from historical result trees.
            if component.startswith('experiment/') and any(part in {'sources', 'qualification',
                    'qualification-invalid-build-hooks', 'checks', 'formal', 'performance',
                    'history', 'secure_validation', 'contract_fixtures', 'conv1_pilot',
                    'round_analyzer_validation', 'width_scan', 'missing_width_runs'}
                    for part in path.relative_to(REPO / root).parts[:-1]):
                continue
            if path.parent == HERE and path.name not in {'README.zh.md', 'inventory.py'}:
                continue
            selected[relative] = component
    for path in OUR.iterdir():
        if path.is_file() and path.suffix in SUFFIXES:
            selected[path.relative_to(REPO).as_posix()] = 'project_top_level'
    for path in REPO.iterdir():
        if path.is_file() and path.suffix in {'.md', '.tex', '.bib'}:
            selected[path.relative_to(REPO).as_posix()] = 'repository_documentation'
    explicit = [
        'project/our/results/online_optimization_20261014/build/build_runtime.py',
        'project/our/results/ticket07_hybrid_full/sources/sirnn_fixed.cpp',
        'project/our/results/ticket07_hybrid_full/sources/model_fixed.h',
        'project/our/results/paper_folded_activation_20261013/ticket11/frozen-plan-final.json',
    ]
    for name in explicit:
        if not (REPO / name).is_file():
            raise FileNotFoundError(name)
        selected[name] = 'required_frozen_source_or_plan'
    untracked = [name for name in selected if name not in tracked]
    ignored = set()
    if untracked:
        result = subprocess.run(['git', 'check-ignore', '-z', '--stdin'], cwd=REPO,
                                input=('\0'.join(untracked) + '\0').encode(), capture_output=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode())
        ignored = set(result.stdout.decode().split('\0')) - {''}
    rows = []
    for name, component in sorted(selected.items()):
        state = ('tracked_modified' if name in modified else 'tracked_clean') if name in tracked else (
            'ignored_untracked' if name in ignored else 'untracked')
        recommendation = ('optional_archive' if component == 'legacy_baseline_optional'
                          or name in OPTIONAL_PATHS else 'github_source')
        rows.append(dict(path=name, component=component, bytes=(REPO / name).stat().st_size,
                         git_state=state, recommended=recommendation))
    write_tsv('source_inventory.tsv', ['path', 'component', 'bytes', 'git_state', 'recommended'], rows)
    recommended = [row['path'] for row in rows if row['recommended'] == 'github_source']
    migration_docs = [
        'README.zh.md', 'inventory.py', 'source_inventory.tsv', 'github_sources.list',
        'git_missing_sources.list', 'required_inputs.tsv', 'external_sources.tsv',
        'external_revisions.json', 'environment.observed.tsv', 'cleanup_candidates.tsv',
        'inventory_summary.json',
    ]
    recommended.extend('project/our/docs/server_migration_20261016/' + name
                       for name in migration_docs)
    recommended = sorted(set(recommended))
    (HERE / 'github_sources.list').write_text('\n'.join(recommended) + '\n')
    missing = [name for name in recommended if name not in tracked]
    (HERE / 'git_missing_sources.list').write_text('\n'.join(missing) + '\n')

    inputs = [
        ('rnn_frozen_step', 'project/our/results/paper_folded_activation_20261013/ticket11/final-build/fixture', 'required_input', '当前batch128输入、模型和整数契约'),
        ('rnn_frozen_step', 'project/our/results/sci_sirnn_resident_20261012/reference', 'required_input', '非零hidden i16及来源'),
        ('rnn_frozen_step', 'project/our/results/ticket07_hybrid_full/sources', 'required_source', '冻结生成图和model header，源码白名单也收录'),
        ('rnn_frozen_step', 'project/our/results/online_optimization_20261014/build/runtime/runtime_manifest.json', 'build_provenance', '重建来源身份，不是可执行程序'),
        ('rnn_frozen_step', 'project/our/results/online_optimization_20261014/build/runtime/build_commands.json', 'build_provenance', '历史编译命令'),
        ('rnn_evidence', 'project/our/results/online_optimization_20261014/formal', 'evidence', '四行10query；含大轨迹，可冷热分离'),
        ('rnn_evidence', 'project/our/results/online_optimization_20261014/qualification', 'evidence', '零/非零hidden资格'),
        ('rnn_evidence', 'project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair', 'evidence', '历史正式原生预处理对比'),
        ('cnn_accuracy', 'project/our/results/imagenet_resnet_20261015/data/val', 'dataset_external_transfer', '完整50000张验证图'),
        ('cnn_accuracy', 'project/our/results/imagenet_resnet_20261015/data/MANIFEST.json', 'dataset_metadata', '来源与验证清单'),
        ('cnn_accuracy', 'project/our/results/imagenet_resnet_20261015/data/class_to_idx.json', 'dataset_metadata', '类别映射'),
        ('cnn_accuracy', 'project/our/results/imagenet_resnet_20261015/data/labels.csv', 'dataset_metadata', '标签'),
        ('cnn_accuracy', 'project/our/results/imagenet_resnet_20261015/data/ILSVRC2012_validation_ground_truth.txt', 'dataset_metadata', '官方标签'),
        ('cnn_accuracy', 'project/our/results/imagenet_resnet_20261015/data/meta.mat', 'dataset_metadata', '官方类别元数据'),
        ('cnn_accuracy', 'project/our/results/precision_motivation_final/data', 'dataset_external_transfer', '2000张训练校准图及抽取记录'),
        ('cnn_accuracy', 'project/our/results/precision_motivation_final/experiment1/calibration_manifest.json', 'required_input', '冻结训练校准清单'),
        ('cnn_accuracy', '/home/jiang/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth', 'weights_external_transfer', 'IMAGENET1K_V1，不能换V2'),
        ('cnn_accuracy', '/home/jiang/.cache/torch/hub/checkpoints/resnet152-394f9c45.pth', 'weights_external_transfer', 'IMAGENET1K_V1'),
        ('cnn_resume', 'project/our/results/precision_motivation_final/experiment1/width_scan', 'checkpoint_and_evidence', '配置、断点、证书和已完成验证'),
        ('cnn_resume', 'project/our/results/imagenet_resnet_20261015/accuracy', 'checkpoint_and_evidence', '已完成32/12完整验证基线'),
        ('cnn_secure', 'project/our/results/imagenet_resnet_20261015/models/resnet50_w32_f12', 'required_input_external_transfer', '固定图、parameters和audit'),
        ('cnn_secure', 'project/our/results/imagenet_resnet_20261015/models/resnet152_w32_f12', 'required_input_external_transfer', '固定图、parameters和audit'),
        ('cnn_secure', 'project/our/results/imagenet_resnet_20261015/contract_fixtures/w32', 'qualification_input', '小图逐层整数资格'),
        ('cnn_secure', 'project/our/results/imagenet_resnet_20261015/final_migration/build/build.json', 'build_provenance', '当前CNN编译命令及身份'),
        ('motivation_evidence', 'project/our/results/precision_motivation_final/experiment2', 'evidence', '23/31位转换算子与开销表'),
        ('motivation_evidence', 'project/our/results/precision_motivation_final/primitive_ablation', 'evidence', '复用及缺失位宽原语补测，build可重建'),
        ('motivation_evidence', 'project/our/results/ticket10_width_sweep', 'evidence', '已有位宽扫描和不分段OOM证据'),
        ('rnn_full_reproduction', '/home/jiang/EzPC/SIRNN/reproduction/models/Google-30-FastGRNN-r16-u35-h100', 'required_input', '训练权重、mean/std、配置、历史'),
        ('rnn_full_reproduction', '/home/jiang/EzPC/SIRNN/reproduction/data/Google-30-processed', 'dataset_external_transfer', '完整NPY、样本清单和特征来源'),
        ('rnn_full_reproduction', '/home/jiang/EzPC/SIRNN/reproduction/data/Google-30-v0.01', 'dataset_optional_if_frozen', '原始音频；仅冻结单步不需要'),
        ('rnn_full_reproduction', '/home/jiang/EzPC/SIRNN/reproduction/data/speech_commands_v0.01.tar.gz', 'source_archive', '保留一个来源副本'),
        ('rnn_full_reproduction', '/home/jiang/EzPC/SIRNN/reproduction/SeeDot/temp/Predictor/input/X.csv', 'generated_input', '完整SeeDot测试输入'),
        ('rnn_full_reproduction', '/home/jiang/EzPC/SIRNN/reproduction/SeeDot/temp/Predictor/input/Y.csv', 'generated_input', '完整SeeDot测试标签'),
    ]
    input_rows = [dict(task=task, path=name, exists=repo_path(name).exists(),
                       allocated_bytes=allocated_bytes(repo_path(name)), category=category, note=note)
                  for task, name, category, note in inputs]
    write_tsv('required_inputs.tsv', ['task', 'path', 'exists', 'allocated_bytes', 'category', 'note'], input_rows)

    cleanup = [
        ('obsolete_download', 'project/our/tools/precision/data/imagenet1k/archives', '完整验证集在新results；确认不再续传旧源'),
        ('obsolete_download', 'project/our/results/imagenet_resnet_20261015/data/ILSVRC2012_img_val.tar.parts', '未用官方源分块；新验证集已完成'),
        ('obsolete_download', '/home/jiang/EzPC/SIRNN/reproduction/data/speech_commands_v0.01.tar.gz.parts', '完整tar和解压数据均在后可清理'),
        ('omit_transfer_keep_origin', 'project/our/results/imagenet_resnet_20261015/data/kaggle_titericz_imagenet1k_val_v1.zip', '已有val；至少保留一份来源归档'),
        ('rebuild_environment', 'project/our/tools/precision/.venv', '先保存版本，新服务器重建；本机继续运行仍需要'),
        ('rebuild_environment', '/home/jiang/EzPC/SIRNN/reproduction/venv', 'TF1训练环境，保存train_environment后重建'),
        ('rebuild_environment', '/home/jiang/EzPC/SIRNN/reproduction/seedot-venv', 'SeeDot环境，使用lock重建'),
        ('rebuild_cache', 'project/our/tools/imagenet_resnet/.round_trace_native', '统计器缓存可重编；保留已生成ROUNDS证据'),
        ('rebuild_cache', 'project/our/tools/imagenet_resnet/__pycache__', 'Python缓存'),
        ('rebuild_cache', 'project/our/tools/imagenet_resnet/.pytest_cache', 'pytest缓存'),
        ('rebuild_cache', '/home/jiang/.cache/torch_extensions', '新服务器CUDA扩展重编'),
        ('diagnostic_export', 'project/our/results/precision_motivation_final/experiment1/r152_w28_f27_diagnostic/evaluation/export/parameters.bin', '仅内存诊断；保留metrics、guard和报告'),
        ('diagnostic_export', 'project/our/results/precision_motivation_final/experiment1/r152_w28_f27_released_diagnostic/evaluation/export/parameters.bin', '仅内存诊断；保留metrics、guard和报告'),
        ('cold_archive_evidence', 'project/our/results/imagenet_resnet_20261015/performance/resnet50_SCI_q1', '正式原始轨迹，非缓存；保留party计量、输出audit和COST_MEASUREMENT'),
        ('cold_archive_evidence', 'project/our/results/imagenet_resnet_20261015/history/pre_osrng/performance/resnet50_SCI_q1', '另一随机源运行，非重复；保留历史说明与证据'),
        ('cold_archive_partial_only', 'project/our/results/paper_folded_activation_20261013', '不能整删：ticket11计划、final-build/fixture和脚本仍直接依赖'),
        ('cold_archive_partial_only', 'project/our/results/paper_folded_activation_20261014', '历史轨迹与重测；保留报告、运行脚本和身份'),
        ('cold_archive_partial_only', 'project/our/results/offline_optimization_20261014', '大轨迹可归档；保留优化前后证据、脚本和构建身份'),
        ('cold_archive_partial_only', 'project/our/results/online_optimization_20261014', '最新入口不能删；仅大轨迹冷存，保留资格与报告'),
        ('rebuild_after_qualification', 'project/our/results/online_optimization_20261014/build', '含手写build_runtime.py必须单独保留；重建成功后再清对象'),
        ('rebuild_after_qualification', 'project/our/results/imagenet_resnet_20261015/final_migration/build', '旧二进制/库不跨机器混用；先保留build.json'),
        ('legacy_optional_archive', 'project/our/Baselines', '早期明文/smoke框架；不用于当前真实安全测试'),
        ('legacy_optional_archive', 'project/our/tools/precision/data/imagenette2-160', '不参与完整ImageNet；旧精度复现仍需'),
        ('legacy_optional_archive', 'project/our/tools/precision/models/lstm_synthetic.pth', '旧合成LSTM模型；不是Google30正式模型'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_speech_commands.py', '旧Speech Commands v0.02流程；已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_deep_lstm.py', '旧深层LSTM实验；已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_lstm_e0.py', '旧LSTM E0实验；已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/audio_features.py', '旧音频特征处理辅助代码；已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/fixed_point.py', '旧LSTM定点实现；已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/run_e0_uniform_storage.py', '旧E0精度脚本；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/run_e0_ring.py', '旧E0精度脚本；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/run_e0_q_all.py', '旧E0精度脚本；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/run_relu_precision_ablation.py', '旧精度消融脚本；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/run_relu_maxpool_precision.py', '旧精度脚本；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_run_e0_ring.py', '旧精度测试；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_run_e0_uniform_storage.py', '旧精度测试；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_run_relu_precision_ablation.py', '旧精度测试；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_run_relu_maxpool_precision.py', '旧精度测试；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/test_sigma_activation_lut.py', '旧精度测试；不参与当前CNN，已从GitHub源码白名单排除'),
        ('legacy_optional_archive', 'project/our/tools/precision/validate_uniform_ring_network.py', '旧统一环网路由测试；当前CNN有独立精确实现'),
    ]
    write_tsv('cleanup_candidates.tsv', ['category', 'path', 'exists', 'allocated_bytes', 'condition'],
              [dict(category=category, path=name, exists=repo_path(name).exists(),
                    allocated_bytes=allocated_bytes(repo_path(name)), condition=note)
               for category, name, note in cleanup])

    external = []
    external_paths = list(sources(REPRO / 'scripts'))
    if REPRO.exists():
        external_paths += [path for path in REPRO.iterdir() if path.is_file() and
                          (path.suffix in {'.md', '.sh'} or path.name.startswith('requirements'))]
    external_paths += [REPRO / 'logs/train_environment.txt']
    external_paths += list((REPRO / 'models/Google-30-FastGRNN-r16-u35-h100').glob('*.json'))
    external_paths += [REPRO / 'models/Google-30-FastGRNN-r16-u35-h100/command.txt']
    for path in sorted(set(external_paths)):
        if path.is_file():
            external.append(dict(path=str(path), bytes=path.stat().st_size,
                restore_relative_to_reproduction=path.relative_to(REPRO).as_posix(),
                note='工作区外本地复现源码/配置；当前本仓库push不会包含'))
    for name in ('SCIConfig.local.patch', 'SEAL.max_degree.local.patch'):
        path = HERE / 'external_patches' / name
        if path.is_file():
            external.append(dict(path=str(path), bytes=path.stat().st_size,
                restore_relative_to_reproduction='external_patches/' + name,
                note='机器专用EzPC/SEAL本地差异快照；不在GitHub源码白名单，迁移前需人工检查'))
    write_tsv('external_sources.tsv', ['path', 'bytes', 'restore_relative_to_reproduction', 'note'], external)
    vcs = []
    for name in ['/home/jiang/EzPC', '/home/jiang/EzPC/SIRNN/EdgeML',
                 '/home/jiang/EzPC/SIRNN/EdgeML-sirnn', '/home/jiang/external-baselines/SiRNN',
                 str(REPO / 'project/dealerless-FSS_public'), '/home/jiang/EzPC/SCI/extern/SEAL']:
        path = Path(name)
        if (path / '.git').exists():
            vcs.append(dict(path=name, head=command(['git', '-C', name, 'rev-parse', 'HEAD']),
                            tracked_status=command(['git', '-C', name, 'status', '--short', '--untracked-files=no']).splitlines()))
    write_json('external_revisions.json', vcs)
    patch_dir = HERE / 'external_patches'
    patch_dir.mkdir(exist_ok=True)
    for directory, relative, output in [('/home/jiang/EzPC', 'SCI/cmake/SCIConfig.cmake', 'SCIConfig.local.patch'),
            ('/home/jiang/EzPC/SCI/extern/SEAL', 'native/src/seal/util/defines.h', 'SEAL.max_degree.local.patch')]:
        if Path(directory).exists():
            result = subprocess.run(['git', '-C', directory, 'diff', '--', relative],
                                    capture_output=True)
            if result.returncode == 0:
                (patch_dir / output).write_bytes(result.stdout)

    env_rows = []
    environments = [('cnn_precision', OUR / 'tools/precision/.venv'), ('google30_tf1_train', REPRO / 'venv'),
                    ('google30_seedot', REPRO / 'seedot-venv'), ('google30_preprocess', REPRO / 'preprocess-venv')]
    for identity, path in environments:
        config = path / 'pyvenv.cfg'
        if config.exists():
            env_rows.append(dict(environment=identity, package='__pyvenv_cfg__', version=config.read_text().strip().replace('\n', '; '), source=str(config)))
        for meta in sorted(path.glob('lib/python*/site-packages/*.dist-info/METADATA')):
            record = Parser().parsestr(meta.read_text(errors='replace'), headersonly=True)
            env_rows.append(dict(environment=identity, package=record.get('Name', ''),
                                 version=record.get('Version', ''), source=str(meta)))
    # The preprocess venv uses system-site-packages; record its explicit reproducibility requirements too.
    for path in [REPRO / 'requirements_preprocess.txt', REPRO / 'requirements_seedot.lock.txt']:
        if path.exists():
            for line in path.read_text().splitlines():
                if '==' in line and not line.lstrip().startswith('#'):
                    name, version = line.split('==', 1)
                    env_rows.append(dict(environment='declared_' + path.stem, package=name,
                                         version=version, source=str(path)))
    write_tsv('environment.observed.tsv', ['environment', 'package', 'version', 'source'], env_rows)
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), repo=str(REPO),
                   head=command(['git', 'rev-parse', 'HEAD']), source_files=len(rows),
                   source_bytes=sum(row['bytes'] for row in rows), github_source_files=len(recommended),
                   missing_github_source_files=len(missing), modified_files_in_inventory_scope=sum(
                       row['git_state'] == 'tracked_modified' for row in rows), external_source_files=len(external),
                   missing_required_input_paths=[row['path'] for row in input_rows if not row['exists']],
                   size_semantics='du -s -B1 allocated bytes; overlapping directories are not additive',
                   scope='curated source/experiment roots, required frozen sources and inputs; not a full backup',
                   git_local_exclude=(REPO / '.git/info/exclude').read_text().splitlines())
    write_json('inventory_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
