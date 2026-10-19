#!/usr/bin/env python3
"""Build the integer CNN runner against isolated SCI and existing ours sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUR = HERE.parents[1]
RUNTIME = HERE.parent / "segmented_dcf_runtime"


def function(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def trace_parallel(source):
    marker = "#pragma omp parallel num_threads(2)"
    cursor = 0
    while True:
        start = source.find(marker, cursor)
        if start < 0:
            break
        opening = source.index("{", start)
        depth, end = 1, opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        body = source[opening + 1:end - 1].replace("omp_get_thread_num()", "cnn_direction")
        replacement = ("{ sdcf_round::DirectionGroup cnn_group(2);\n"
                       "#pragma omp parallel for num_threads(2)\n"
                       "for (int cnn_direction=0; cnn_direction<2; ++cnn_direction) {\n"
                       "sdcf_round::DirectionScope cnn_scope(cnn_group, cnn_direction);\n" + body +
                       "\n} cnn_group.join(); }")
        source = source[:start] + replacement + source[end:]
        cursor = start + len(replacement)
    return '#include "round_trace.h"\n' + source


def native_prepared_overlays(out, sci, module):
    """Reuse the RNN random-OT/triple protocols with bounded disk storage."""
    module.native_ot_overlays(out, sci)
    module.native_planning_overlays(out, sci)
    module.native_triple_overlay(out, sci)
    module.native_got_overlay(out)
    for name in ("native_preprocessing.h", "native_planning.h", "native_plan_replay.h",
                 "native_query_materials.h", "prepared_got.h", "prepared_cot.h", "prepared_triples.h"):
        shutil.copyfile(RUNTIME / name, out / name)
    shutil.copyfile(HERE / "cnn_native_spool.h", out / "cnn_native_spool.h")

    def patch(name, changes):
        path = out / name
        text = path.read_text()
        for before, after in changes:
            if before not in text:
                raise ValueError("CNN native spool insertion changed: " + before)
            text = text.replace(before, after)
        path.write_text(text)

    patch("prepared_got.h", [
        ('#include "native_preprocessing.h"', '#include "native_preprocessing.h"\n#include "cnn_native_spool.h"'),
        ('queue.push_back(std::move(material));', 'cnn_spool::save(ot, 1, material.choices, material.masks, material.words);\n    queue.push_back(std::move(material));'),
        ('found->second.pop_front();\n    return result;', 'found->second.pop_front();\n    cnn_spool::load(ot, 1, result.choices, result.masks, result.words);\n    return result;'),
        ('call.protocol != "iknp_batched_got" ||', '(call.protocol != "iknp_batched_got" && call.protocol != "iknp_got") ||'),
        ('int count, int bits, int group) {', 'int count, int bits, int group, const char* protocol = "iknp_batched_got") {'),
        ('consume(ot, "iknp_batched_got", count, bits, group);', 'consume(ot, protocol, count, bits, group);'),
        ('take_got(ot, "iknp_batched_got", count, bits, group)', 'take_got(ot, protocol, count, bits, group)'),
    ])
    patch("prepared_cot.h", [
        ('#include "native_preprocessing.h"', '#include "native_preprocessing.h"\n#include "cnn_native_spool.h"'),
        ('stock.calls.push_back(std::move(pads));', 'cnn_spool::save(&stock, 2, pads.zero, pads.one, pads.random_choices);\n        stock.calls.push_back(std::move(pads));'),
        ('return pads;\n}', 'cnn_spool::load(&stock, 2, pads.zero, pads.one, pads.random_choices);\n    return pads;\n}'),
        ('pads.zero.clear(); pads.one.clear(); ++stock.cursor;',
         'cnn_spool::release(pads.zero); cnn_spool::release(pads.one); ++stock.cursor;'),
        ('pads.zero.clear(); pads.random_choices.clear(); ++stock.cursor;',
         'cnn_spool::release(pads.zero); cnn_spool::release(pads.random_choices); ++stock.cursor;'),
    ])
    patch("prepared_triples.h", [
        ('#include "native_preprocessing.h"', '#include "native_preprocessing.h"\n#include "cnn_native_spool.h"'),
        ('inventory.stock.push_back(std::move(material));', 'cnn_spool::save(generator->otpack, 3, material.a, material.b, material.c);\n        inventory.stock.push_back(std::move(material));'),
        ('std::memcpy(a, material.a.data(), bytes(call));', 'cnn_spool::load(pack, 3, material.a, material.b, material.c);\n    std::memcpy(a, material.a.data(), bytes(call));'),
    ])
    patch("native_plan_replay.h", [
        ('int ordinary = 0;\n        for (const auto& call : calls) if (call.protocol == "iknp_got") ordinary += call.count;', ''),
        ('ot->set_precomp_batch_size(ordinary);\n        if (ordinary) ot->preprocess();', 'ot->set_precomp_batch_size(0);'),
        ('if (call.protocol == "iknp_batched_got") {', 'if (call.protocol == "iknp_batched_got" || call.protocol == "iknp_got") {'),
    ])
    path = out / "include_override/OT/split-iknp.h"
    text = path.read_text()
    for kind in ("uint8_t", "uint64_t"):
        send = "void send_impl(" + kind + " **data, int length, int l) {"
        recv = "void recv_impl(" + kind + " *data, uint8_t *b, int length, int l) {"
        send_hook = '''
    if (sdcf_preprocess::strict && sdcf_preprocess::phase == sdcf_preprocess::Phase::Online) {
      std::vector<uint64_t> flat(size_t(length) * 2);
      for (int i=0;i<length;++i) { flat[2*i]=data[i][0]; flat[2*i+1]=data[i][1]; }
      sdcf_preprocess::prepared_batched_got_send(this, flat.data(), length, l, 1, "iknp_got"); return;
    }'''
        recv_hook = '''
    if (sdcf_preprocess::strict && sdcf_preprocess::phase == sdcf_preprocess::Phase::Online) {
      std::vector<uint64_t> flat(length);
      sdcf_preprocess::prepared_batched_got_recv(this, flat.data(), b, length, l, 1, "iknp_got");
      for (int i=0;i<length;++i) data[i]=flat[i]; return;
    }'''
        text = text.replace(send, send + send_hook).replace(recv, recv + recv_hook)
    path.write_text(text)


def build(out, sci, core, jobs):
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location("cnn_runtime_build", RUNTIME / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.native_io_overlay(out, sci)
    native_prepared_overlays(out, sci, module)
    overlay = out / "include_override"
    for relative in ("Millionaire/millionaire.h", "Millionaire/millionaire_with_equality.h", "Millionaire/equality.h"):
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(trace_parallel((sci / "src" / relative).read_text()))
    native_sources = []
    for relative in ("LinearOT/linear-ot.cpp", "BuildingBlocks/aux-protocols.cpp",
                     "BuildingBlocks/truncation.cpp", "BuildingBlocks/value-extension.cpp",
                     "GC/emp-tool.cpp", "GC/halfgate_gen.cpp", "GC/halfgate_eva.cpp"):
        source = (sci / "src" / relative).read_text()
        if relative == "LinearOT/linear-ot.cpp":
            source = source.replace("#define MAX_NUM_OT (1 << 24)", "#define MAX_NUM_OT (1 << 18)")
        target = out / (relative.replace("/", "_") + ".cpp")
        target.write_text(trace_parallel(source))
        native_sources.append(target)
    original = (sci / "src/functionalities_uniform.h").read_text()
    helpers = [function(original, "signedIntType getAnyRingSignedVal("), function(original, "signedIntType div_floor(")]
    average = function(original, "void funcAvgPoolTwoPowerRing(")
    average = average.replace("void funcAvgPoolTwoPowerRing(", "void cnn_original_avg(")
    avg_source = out / "native_average.cpp"
    avg_source.write_text('#include "globals.h"\n#include <cmath>\n#include <memory>\n'
        'int32_t bitlength=32; uint64_t prime_mod=uint64_t(1)<<32;\n' + "\n".join(helpers) + "\n" + average +
        '\nvoid cnn_native_avg(int role,sci::IOPack* io,sci::OTPack* ot,ReLUProtocol<uint64_t>* relu,int bits,uint64_t* x,uint64_t* y,int n,uint64_t d) {\n'
        'bitlength=bits; prime_mod=uint64_t(1)<<bits;\n'
        'static std::unique_ptr<sci::IKNP<sci::NetIO>> native_ot;\n'
        'static std::unique_ptr<sci::KKOT<sci::NetIO>> native_kkot;\n'
        'if (!native_ot) { native_ot=std::make_unique<sci::IKNP<sci::NetIO>>(io->io); native_kkot=std::make_unique<sci::KKOT<sci::NetIO>>(io->io); }\n'
        'sci::PRG128 prg; cnn_original_avg(role,io,ot,native_ot.get(),native_kkot.get(),relu,&prg,n,x,y,d);\n}\n')
    flags = ['g++', '-O3', '-std=c++17', '-pthread', '-maes', '-msse4.1', '-mavx2', '-mrdseed', '-fopenmp',
             '-ffunction-sections', '-fdata-sections', '-Wno-deprecated-declarations', '-DSCI_OT=1',
             '-DEMP_USE_RANDOM_DEVICE=1',
             '-DEIGEN_DONT_PARALLELIZE', '-DUSE_EIGEN', '-I' + str(out), '-I' + str(overlay),
             '-I' + str(RUNTIME), '-I' + str(OUR / 'src'), '-I' + str(sci / 'src'), '-I' + str(sci / 'extern/eigen')]
    sources = [HERE / 'secure_driver.cpp', HERE.parent / 'sci_beaver/beaver.cpp', RUNTIME / 'round_trace.cpp', avg_source] + native_sources
    commands = []
    objects = []
    for index, source in enumerate(sources):
        obj = out / ('native_%d.o' % index); objects.append(obj)
        commands.append(flags + ['-c', str(source), '-o', str(obj)])
    bridge_flags = ['g++', '-O3', '-std=c++17', '-fPIC', '-pthread', '-maes', '-msse4.1', '-mavx2', '-fopenmp',
                    '-DEMP_USE_RANDOM_DEVICE=1', '-Wno-deprecated-declarations', '-I' + str(OUR / 'src'),
                    '-I' + str(core / 'dealerless-fss/fss-overlay-src'),
                    '-I' + str(core / 'dealerless-fss/fss-overlay-src/deps/cryptoTools'),
                    '-I' + str(sci / 'src'), '-I' + str(OUR.parent / 'dealerless-FSS_public'), '-I' + str(sci / 'extern/eigen')]
    bridge_objects = []
    for index, source in enumerate((HERE / 'cnn_bridge.cpp', OUR / 'src/sci_conversion_bridge.cpp')):
        obj = out / ('bridge_%d.o' % index); bridge_objects.append(obj)
        commands.append(bridge_flags + ['-c', str(source), '-o', str(obj)])

    def compile_one(item):
        index, command = item
        source = Path(command[command.index('-c') + 1])
        dependencies = (list(overlay.rglob('*.h')) + list(out.glob('*.h')) +
                        list(HERE.glob('*.h')) + list(RUNTIME.glob('*.h')) + list((OUR / 'src').glob('*.h')))
        if source.name == 'cnn_bridge.cpp':
            dependencies += [OUR / 'src/sci_msb_bridge.cpp']
        checksum = hashlib.sha256(json.dumps(command).encode() + source.read_bytes() +
            b''.join(path.read_bytes() for path in sorted(dependencies))).hexdigest()
        stamp = out / ('compile_%d.sha256' % index)
        if stamp.exists() and stamp.read_text() == checksum and Path(command[-1]).exists():
            return
        completed = subprocess.run(command, text=True, capture_output=True)
        (out / ('compile_%d.log' % index)).write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError('compile failed, see ' + str(out / ('compile_%d.log' % index)))
        stamp.write_text(checksum)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(compile_one, enumerate(commands)))
    library = out / 'libcnn_ours.so'
    export = out / 'bridge.exports'
    export.write_text('{ global: segfss_sci_*; local: *; };\n')
    bridge_link = ['g++', '-shared', '-Wl,-Bsymbolic', '-Wl,--version-script=' + str(export),
        '-o', str(library)] + list(map(str, bridge_objects)) + [str(core / 'libsegfss.a'),
        str(core / 'dealerless-fss/libdfss.a'), str(core / 'dealerless-fss/libdfss_common.a'),
        '-lssl', '-lcrypto', '-lgmp', '-pthread', '-fopenmp']
    subprocess.run(bridge_link, check=True)
    archives = [sci / ('build/install/lib/libSCI-' + name + '.a') for name in ('OT', 'LinearOT', 'GC', 'Math', 'BuildingBlocks', 'FloatingPoint')]
    binary = out / 'resnet_secure'
    link = ['g++', '-o', str(binary)] + list(map(str, objects)) + ['-L' + str(out), '-lcnn_ours',
        '-Wl,-rpath,' + str(out), '-Wl,--gc-sections', '-Wl,--start-group'] + list(map(str, archives)) + [
        '-Wl,--end-group', '-lssl', '-lcrypto', '-lgmp', '-pthread', '-fopenmp']
    subprocess.run(link, check=True)
    (out / 'build.json').write_text(json.dumps({'binary': str(binary), 'core': str(core), 'sci': str(sci),
        'commands': commands + [bridge_link, link], 'native_ot_payload_bound': 1 << 18,
        'sci_default_seed_source': 'std::random_device (EMP_USE_RANDOM_DEVICE=1)'}, indent=2) + '\n')
    print(binary)
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=OUR / 'results/imagenet_resnet_20261015/build')
    parser.add_argument('--sci', type=Path, default=Path('/home/jiang/EzPC/SCI'))
    parser.add_argument('--core', type=Path, default=OUR / 'results/online_optimization_20261014/build/core')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    build(args.build.resolve(), args.sci.resolve(), args.core.resolve(), args.jobs)


if __name__ == '__main__':
    main()
