#!/usr/bin/env python3
"""Build a metered native mixed-width graph from the frozen generated calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from native_cot_overlay import native_cot_overlays
from native_planning_overlay import native_planning_overlays
from native_triple_overlay import native_triple_overlay

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "google30_hybrid"))
from build_full import generate as original_generate
from runtime import call_arguments
from native_got_overlay import native_got_overlay

SOURCES = ["library_fixed.cpp", "library_fixed_uniform.cpp", "globals.cpp",
           "cleartext_library_fixed.cpp", "LinearOT/linear-ot.cpp", "GC/emp-tool.cpp",
           "GC/halfgate_gen.cpp", "GC/halfgate_eva.cpp", "BuildingBlocks/aux-protocols.cpp",
           "BuildingBlocks/truncation.cpp", "BuildingBlocks/value-extension.cpp",
           "Math/math-functions.cpp"]
LIBS = ["OT", "LinearOT", "GC", "Math", "BuildingBlocks", "FloatingPoint"]

# Scheduling uses fixed independent branches.  Freeze the complete original
# numeric and tensor contract before emitting any fork, including the tail.
FROZEN_RECURRENT_CALLS = (
    "MatMul(1,32,16,1,2,5,0,1,16,8,32,16,tmp15bw16,W1,tmp19bw16,tmp18bw32);",
    "MatMul(1,16,100,32,64,4,0,1,16,16,32,16,tmp19bw16,W2,tmp21bw16,tmp20bw32);",
    "MatMul(1,100,35,1,1,6,1,256,16,8,32,8,tmp11bw16,U1,tmp23bw8,tmp22bw32);",
    "MatMul(1,35,100,1,1,6,0,1,8,16,32,16,tmp23bw8,U2,tmp25bw16,tmp24bw32);",
    "MatAdd(1,100,1,8,1,1,16,16,16,16,tmp21bw16,tmp25bw16,tmp26bw16);",
    "MatAdd(1,100,1,8,1,1,16,16,16,16,tmp26bw16,Bg,tmp27bw16);",
    "Sigmoid(1,100,128,16384,16,16,tmp27bw16,tmp28bw16);",
    "MatAdd(1,100,1,8,1,1,16,16,16,16,tmp26bw16,Bh,tmp29bw16);",
    "TanH(1,100,128,128,16,16,tmp29bw16,tmp30bw16);",
    "MulCir(1,100,128,128,1,16,16,32,16,tmp28bw16,tmp11bw16,tmp31bw16);",
    "MatSubBroadCastA(1,100,1,1,1,1,16,16,16,16,tmp32,tmp28bw16,tmp33bw16);",
    "ScalarMul(1,100,128,128,1,16,16,32,16,tmp9bw16,tmp33bw16,tmp34bw16);",
    "MatAddBroadCastB(1,100,1,64,1,1,16,16,16,16,tmp34bw16,tmp10bw16,tmp35bw16);",
    "MulCir(1,100,8,16,1,16,16,32,16,tmp35bw16,tmp30bw16,tmp36bw16);",
    "MatAdd(1,100,1,2,1,1,16,16,16,16,tmp31bw16,tmp36bw16,tmp11bw16);",
)


def validate_frozen_recurrent_graph(schema):
    if len(schema) < len(FROZEN_RECURRENT_CALLS):
        raise ValueError("fixed graph requires the complete 15-call recurrent contract")
    for index, (row, source_call) in enumerate(zip(schema, FROZEN_RECURRENT_CALLS)):
        function = source_call.split("(", 1)[0]
        expected = call_arguments({"source_call": source_call})
        output = expected[-2 if function == "MatMul" else -1]
        bits = int(re.search(r"bw(\d+)$", output)[1])
        if (row["function"] != function or call_arguments(row) != expected or
                row["output"] != output or row["bits"] != bits):
            raise ValueError("fixed recurrent graph contract changed at call " + str(index))


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def generate(source):
    # The existing parser owns the graph. We change only the public batch
    # dimension and attach scheduling/measurement to the original calls.
    _, schema = original_generate(source)
    validate_frozen_recurrent_graph(schema)
    names = {name for row in schema for name in re.findall(r"\btmp\d+bw\d+\b", row["source_call"])} - {"tmp9bw16", "tmp10bw16"}
    widths = {name: 100 for name in names}
    widths.update(tmp15bw16=32, tmp19bw16=16, tmp23bw8=35, tmp39bw8=30, tmp40bw8=30)
    declarations = "\n".join("vector<int64_t> storage_%s(batch * %d, 0); int64_t* %s = storage_%s.data();" %
                              (name, widths[name], name, name) for name in sorted(names))
    fills = "\n".join("fill(storage_%s.begin(), storage_%s.end(), 0);" % (name, name) for name in sorted(names))
    planning_shapes = {name: "batch * %d" % widths[name] for name in names}
    planning_shapes.update(W1="512", W2="1600", U1="3500", U2="3500", Bg="batch * 100", Bh="batch * 100",
                           FC="3000", FCbias="batch * 30")
    planning_shadows = "\n".join("vector<int64_t> public_%s(planning ? %s : 0, 0); "
        "int64_t* %s = planning ? public_%s.data() : %s.data();" %
        (name, shape, name, name, name + "_storage" if name not in names else "storage_" + name)
        for name, shape in sorted(planning_shapes.items()))

    def descriptor(row):
        args = call_arguments(row)
        prefix = next(index for index, value in enumerate(args) if not re.fullmatch(r"-?\d+", value))
        numeric = ["batch"] + args[1:prefix]
        source_call = json.dumps(row["source_call"])
        # Bg and Bh are fixed model constants.  The runner can select the
        # prepared ring value at query construction time; keeping this as a
        # public schedule expression makes the same binary usable for the
        # legacy and prepared controls without fabricating a second graph.
        if row["function"] == "MatAdd" and len(args) > 11 and args[11] in ("Bg", "Bh"):
            numeric[3] = "(model_bias_precomputed ? 1 : 8)"
            prepared = list(args)
            prepared[3] = "1"
            source_call = "(model_bias_precomputed ? %s : %s)" % (
                json.dumps("MatAdd(" + ", ".join(prepared) + ");"), source_call)
        output_index = len(args) - (2 if row["function"] == "MatMul" else 1)
        inputs = [value for value in args[prefix:output_index] if value not in ("tmp32", "tmp9bw16", "tmp10bw16")]
        shape = ["batch"] + args[1:3 if row["function"] == "MatMul" else 2]
        return "schedule.push_back({int(schedule.size()), t, %s, %s, %s, %s, {%s}, {%s}, {%s}, %d, batch * %d});" % (
            json.dumps(row["operator"]), json.dumps(row["function"]), source_call,
            json.dumps(row["output"]), ",".join(numeric), ",".join(json.dumps(value) for value in inputs),
            ",".join(shape), row["bits"], widths[row["output"]])

    def call(row, index, scheduled, invocation=None):
        args = call_arguments(row)
        args[0] = "batch"
        if row["function"] == "MatAdd" and len(args) > 11 and args[11] in ("Bg", "Bh"):
            args[3] = "(model_bias_precomputed ? 1 : 8)"
        descriptor = "schedule.at(t * 15 + %d)" % index if scheduled else "schedule.at(cursor++)"
        return "\n".join(["{ const auto& layer = %s;" % descriptor,
            "const int branch = sdcf_native::graph_branch(layer.index);",
            "sdcf_native::TaskScope native_task(query, layer.index, branch * 2, sdcf_native::isolated_graph() ? 2 : 0);",
            "const auto layer_start = sdcf_measure::Clock::now();",
            "sdcf_backend::layer_begin(layer);",
            invocation if invocation is not None else row["function"] + "(" + ", ".join(args) + ");",
            "sdcf_backend::layer_end(layer);",
            "sdcf_native::log(std::string(\"GRAPH_TASK {\\\"query\\\":\") + std::to_string(query) +",
            "  \",\\\"index\\\":\" + std::to_string(layer.index) + \",\\\"branch\\\":\" + std::to_string(branch) +",
            "  \",\\\"first_protocol_slot\\\":\" + std::to_string(branch * 2) +",
            "  \",\\\"requested_ot_participants\\\":2,\\\"observed_ot_participants\\\":\" + std::to_string(native_task.observed_ot_participants()) +",
            "  \",\\\"observed_compute_participants\\\":\" + std::to_string(native_task.observed_compute_participants()) +",
            "  \",\\\"branch_compute_limit\\\":\" + std::to_string(sdcf_native::isolated_graph() ? 2 : 4) +",
            "  \",\\\"seconds\\\":\" + std::to_string(std::chrono::duration<double>(sdcf_measure::Clock::now() - layer_start).count()) + \"}\\n\"); }"])

    def inspection(row):
        return 'inspect("%s", "%s", %s, batch, %d, %d);' % (
            row["operator"], row["tensor"], row["output"], widths[row["output"]], row["bits"])

    def calls(rows):
        result = []
        for index, row in enumerate(rows):
            result += [call(row, index, False), inspection(row)]
        return "\n".join(result)

    expected = ["MatMul", "MatMul", "MatMul", "MatMul", "MatAdd", "MatAdd", "Sigmoid", "MatAdd", "TanH"]
    if [row["function"] for row in schema[:9]] != expected:
        raise ValueError("fixed graph branch operator schedule changed")
    add_a, add_b = (call_arguments(schema[i]) for i in (5, 7))
    if add_a[:10] != add_b[:10] or add_a[10] != add_b[10] or add_a[1] != "100":
        raise ValueError("graph opening batch requires homogeneous MatAdd contracts and shared input")
    if add_a[1:10] != ["100", "1", "8", "1", "1", "16", "16", "16", "16"] or (add_a[11], add_b[11]) != ("Bg", "Bh"):
        raise ValueError("model bias preparation requires the frozen owned 16-bit /8 graph")
    grouped_args = ["2 * batch"] + add_a[1:10] + ["grouped_input.data()", "grouped_bias.data()", "grouped_output.data()"]
    grouped_args[3] = "(model_bias_precomputed ? 1 : 8)"
    grouped_add = call(schema[5], 5, True, "\n".join([
        "const size_t grouped_count = size_t(batch) * 100;",
        "vector<int64_t> grouped_input(2 * grouped_count), grouped_bias(2 * grouped_count), grouped_output(2 * grouped_count);",
        "std::copy_n(%s, grouped_count, grouped_input.begin());" % add_a[10],
        "std::copy_n(%s, grouped_count, grouped_input.begin() + grouped_count);" % add_a[10],
        "std::copy_n(%s, grouped_count, grouped_bias.begin());" % add_a[11],
        "std::copy_n(%s, grouped_count, grouped_bias.begin() + grouped_count);" % add_b[11],
        "MatAdd(" + ", ".join(grouped_args) + ");",
        "std::copy_n(grouped_output.begin(), grouped_count, %s);" % add_a[12],
        "std::copy_n(grouped_output.begin() + grouped_count, grouped_count, %s);" % add_b[12],
        'sdcf_native::LogLine() << "GRAPH_OPENING_GROUP {\\\"query\\\":" << query << ",\\\"index\\\":" << layer.index',
        '  << ",\\\"indices\\\":[" << layer.index << "," << layer.index + 2 << "],\\\"count_per_group\\\":" << grouped_count',
        '  << ",\\\"input_groups\\\":2,\\\"mapping\\\":\\\"gate_bias_then_candidate_bias\\\",\\\"contract\\\":\\\"homogeneous_signed_truncation\\\"}\\n";']))
    grouped_completion = call(schema[7], 7, True, "")
    recurrent = ["if (sdcf_native::graph_mode() == \"parallel\") {"]
    for first, second in (([0, 1], [2, 3]), ([5, 6], [7, 8])):
        if first[0] == 5:
            recurrent += ["if (sdcf_native::batched_graph_openings()) {", grouped_add, grouped_completion,
                "{ sdcf_native::TaskScope graph_group(query, t * 15 + 5, 0, 2);",
                "sdcf_native::run_task_pair([&] {", call(schema[6], 6, True),
                "}, [&] {", call(schema[8], 8, True), "}); }", "} else {"]
        recurrent.append("{ sdcf_native::TaskScope graph_group(query, t * 15 + %d, 0, 2);\n" % first[0] +
                         "sdcf_native::run_task_pair([&] {\n" + "\n".join(call(schema[i], i, True) for i in first) +
                         "\n}, [&] {\n" + "\n".join(call(schema[i], i, True) for i in second) + "\n}); }")
        if first[0] == 0:
            recurrent += [inspection(row) for row in schema[:4]] + [call(schema[4], 4, True), inspection(schema[4])]
        else:
            recurrent.append("}")
            recurrent += [inspection(row) for row in schema[5:9]]
    recurrent += ["if (sdcf_native::tail_parallel()) {",
        "{ sdcf_native::TaskScope graph_group(query, t * 15 + 9, 0, 2);",
        "sdcf_native::run_task_pair([&] {", call(schema[9], 9, True),
        "}, [&] {"]
    recurrent += [call(schema[i], i, True) for i in range(10, 14)]
    recurrent += ["}); }", "} else {"]
    recurrent += [call(schema[i], i, True) for i in range(9, 14)]
    recurrent += ["}"] + [inspection(row) for row in schema[9:14]]
    recurrent += [call(schema[14], 14, True), inspection(schema[14])]
    recurrent += ["cursor += 15;", "} else if (sdcf_native::batched_graph_openings()) {"]
    for i in range(5):
        recurrent += [call(schema[i], i, True), inspection(schema[i])]
    recurrent += [grouped_add, grouped_completion, call(schema[6], 6, True), call(schema[8], 8, True)]
    recurrent += [inspection(row) for row in schema[5:9]]
    for i in range(9, 15):
        recurrent += [call(schema[i], i, True), inspection(schema[i])]
    recurrent += ["cursor += 15;", "} else {", calls(schema[:15]), "}"]

    text = (HERE / "driver.cpp.in").read_text()
    for token, value in {"DECLARATIONS": declarations, "FILL_GENERATED": fills, "PLANNING_SHADOWS": planning_shadows,
                         "RECURRENT_SCHEDULE": "\n".join(descriptor(row) for row in schema[:15]),
                         "CLASSIFIER_SCHEDULE": "\n".join(descriptor(row) for row in schema[15:]),
                         "RECURRENT_CALLS": "\n".join(recurrent), "CLASSIFIER_CALLS": calls(schema[15:])}.items():
        text = text.replace(token, value)
    return text, schema


def native_ot_overlays(out, sci):
    """Meter actual extension and enforce the native per-query material contract."""
    shutil.copyfile(HERE / "runtime_log.h", out / "runtime_log.h")
    target = out / "include_override/OT"
    target.mkdir(parents=True, exist_ok=True)
    for name, receiver in (("split-iknp.h", "bool *r"), ("split-kkot.h", "const uint8_t *r")):
        text = (sci / "src/OT" / name).read_text()
        text = text.replace('#include "split-utils.h"', '#include "OT/split-utils.h"')
        include = '#include "OT/ot.h"'
        text = text.replace(include, include + '\n#include "native_preprocessing.h"', 1)
        for signature in ("void send_pre(int length) {", "void recv_pre(" + receiver + ", int length) {"):
            if text.count(signature) != 1:
                raise ValueError("SCI extension entry changed: " + name + " " + signature)
            text = text.replace(signature, signature + '\n    sdcf_preprocess::generation_begin(this);', 1)
        tail = "    this->block_size = old_block_size;"
        if text.count(tail) != 2:
            raise ValueError("SCI extension completion changed: " + name)
        text = text.replace(tail, "    sdcf_preprocess::generation_complete(this, length);\n" + tail)
        preprocess_tail = "    counter = 0;"
        if text.count(preprocess_tail) != 1:
            raise ValueError("SCI random OT completion changed: " + name)
        text = text.replace(preprocess_tail, "    sdcf_preprocess::random_ot_complete(this, counter);\n" + preprocess_tail)
        if name == "split-iknp.h":
            for type_, role in (("uint64_t", "send"), ("uint64_t", "recv"),
                                ("uint8_t", "send"), ("uint8_t", "recv")):
                signature = ("void send_impl(" + type_ + " **data, int length, int l) {" if role == "send" else
                             "void recv_impl(" + type_ + " *data, uint8_t *b, int length, int l) {")
                if text.count(signature) != 1:
                    raise ValueError("SCI GOT wrapper changed: " + signature)
                direct = ("send_pre(length); got_send_post<" + type_ + ">(data, length);" if role == "send" else
                          "recv_pre((bool*)b, length); got_recv_post<" + type_ + ">(data, b, length);")
                preparation = ('\n    if (sdcf_preprocess::strict && sdcf_preprocess::phase == sdcf_preprocess::Phase::Preparing) {'
                               '\n      this->l = l; ' + direct + ' return;\n    }')
                text = text.replace(signature, signature + preparation + '\n    sdcf_preprocess::consume(this, "iknp_got", length, l);', 1)
        (target / name).write_text(text)
    shutil.copyfile(HERE / "prepared_cot.h", out / "prepared_cot.h")
    native_cot_overlays(out, sci)


def native_io_overlay(out, sci):
    net_dir = out / "include_override/utils"
    net_dir.mkdir(parents=True, exist_ok=True)
    net = (sci / "src/utils/net_io_channel.h").read_text()
    for needle, replacement in [
        ('#include "utils/io_channel.h"', '#include "utils/io_channel.h"\n#include <stdexcept>\n#include "io_counters.h"\n#include "round_trace.h"'),
        ("  LastCall last_call = LastCall::None;", "  LastCall last_call = LastCall::None;\n  uint64_t recv_counter = 0;"),
        ("  void send_data(const void *data, int len) {", "  void send_data(const void *data, int len) {\n    if (sdcf_round_trace_io) sdcf_round_trace_io(0, this->port, counter, len);\n    sdcf_measure::sent += len; ++sdcf_measure::sends;"),
        ("  void recv_data(void *data, int len) {", "  void recv_data(void *data, int len) {\n    const auto sdcf_receive_offset = recv_counter;\n    recv_counter += len;\n    sdcf_measure::received += len; ++sdcf_measure::recvs;"),
        ("      addr = string(address);", '      addr = string(address);\n      if (const char* offset = std::getenv("GOOGLE30_PROXY_OFFSET")) port += std::atoi(offset);'),
    ]:
        if net.count(needle) != 1:
            raise ValueError("SCI NetIO source changed: " + needle)
        net = net.replace(needle, replacement, 1)
    # fread/fwrite return zero on EOF or transport error.  The original SCI
    # loop retries zero forever, so a truncated framed packed opening could
    # hang instead of reporting the malformed message.
    if net.count("if (res >= 0)") != 2:
        raise ValueError("SCI NetIO progress checks changed")
    net = net.replace("if (res >= 0)", "if (res > 0)")
    for operation in ("send", "receive"):
        needle = 'fprintf(stderr, "error: net_send_data %d\\n", res);'
        if needle not in net:
            raise ValueError("SCI NetIO zero-progress error point changed")
        net = net.replace(needle, 'throw std::runtime_error("SCI NetIO ' + operation + ' failed or reached EOF");', 1)
    end = net.index("\n  }\n};", net.index("  void recv_data(void *data, int len) {"))
    net = net[:end] + "\n    if (sdcf_round_trace_io) sdcf_round_trace_io(1, this->port, sdcf_receive_offset, len);" + net[end:]
    (net_dir / "net_io_channel.h").write_text(net)
    return net_dir / "net_io_channel.h"


def trace_direction_groups(text):
    """Keep the actual OpenMP direction schedule in the dependency trace."""
    marker = "auto* sdcf_observer = sdcf_native::team_observer();\n#pragma omp parallel for"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return text
        opening = text.index("{", text.index("for (int sdcf_direction", start))
        depth, closing = 1, opening + 1
        while depth:
            depth += (text[closing] == "{") - (text[closing] == "}")
            closing += 1
        replacement = ("{\n  sdcf_round::DirectionGroup sdcf_round_team(sdcf_native::ot_participants());\n" +
                       text[start:opening + 1] +
                       "\n  sdcf_round::DirectionScope sdcf_round_direction(sdcf_round_team, sdcf_direction);" +
                       text[opening + 1:closing] + "\n  sdcf_round_team.join();\n}")
        text = text[:start] + replacement + text[closing:]
        cursor = start + len(replacement)


def native_overlays(out, sci, overrides):
    relative = "Math/math-functions.cpp"
    math = Path(overrides.get(relative, sci / "src" / relative)).read_text()
    for name, is_tanh in (("sigmoid", "false"), ("tanh", "true")):
        signature = re.compile(r"void MathFunctions::" + name + r"\([^{};]*\) \{")
        math, count = signature.subn(lambda match: match[0] +
            "\n  sdcf_dpf::ActivationScope sdcf_activation(" + is_tanh + ");", math)
        if count != 1:
            raise ValueError("SCI activation DPF scope point changed: " + name)
    reciprocal_start = math.index("  if (party == ALICE) {", math.index("void MathFunctions::reciprocal_approximation"))
    reciprocal_end = math.index("  for (int i = 0; i < dim; i++) {\n    tmp_1[i] = dn[i] & s_min_m_mask;", reciprocal_start)
    math = (math[:reciprocal_start] +
            "  if (!sdcf_dpf::lookup_reciprocal || !sdcf_dpf::lookup_reciprocal(this, dim, tmp_2, c0, c1, m, lookup_A0, lookup_A1)) {\n" +
            math[reciprocal_start:reciprocal_end] + "  }\n" + math[reciprocal_end:])
    exp_start = math.index("  if (party == ALICE) {", math.index("void MathFunctions::lookup_table_exp"))
    exp_end = math.index("\n  uint8_t *zero_shares = new uint8_t[dim];", exp_start)
    math = (math[:exp_start] +
            "  if (!sdcf_dpf::lookup_exp || !sdcf_dpf::lookup_exp(this, dim, x_digits, digits_exp, num_digits, digit_size, last_digit_size, s_x, s_y, lookup_neg_exp)) {\n" +
            math[exp_start:exp_end] + "  }\n" + math[exp_end:])
    target = out / "math_functions_native_dpf.cpp"
    target.write_text('#include "native_dpf_lut.h"\n' + math)
    overrides[relative] = str(target)
    library = Path(overrides.get("library_fixed.cpp", sci / "src/library_fixed.cpp")).read_text()
    include_end = library.index("\n", library.rindex("#include"))
    library = library[:include_end] + "\n#undef LOG_LAYERWISE\n" + library[include_end:]
    for name, getter in (("mult", "mult"), ("truncation", "truncation"), ("aux", "aux"), ("xt", "xt"), ("math", "math")):
        library = re.sub(r"\b" + name + r"Arr\[tid\]->", "sdcf_native::selected_" + getter + "(tid)->", library)
        library = re.sub(r"\b" + name + r"->", "sdcf_native::selected_" + getter + "()->", library)
    for function, pattern in (
        ("MulCir", r"threads\[i\] = std::thread\((MulCir_thread,[\s\S]*?)\);"),
        ("MatMul", r"threads\[i\] =\s*std::thread\((MatMul_thread,[\s\S]*?)\);")):
        def direct(match):
            args = match[1]
            name, arguments = args.split(",", 1)
            return "if (lnum_threads == 1) " + name + "(" + arguments.strip() + ");\n    else " + match[0]
        library, replaced = re.subn(pattern, direct, library)
        if replaced != 1:
            raise ValueError("SCI direct single-worker replacement changed: " + function)
    library = library.replace("threads[i].join();", "if (threads[i].joinable()) threads[i].join();")
    library = library.replace("std::thread", "sdcf_round::Thread")
    for function in ("Sigmoid", "TanH"):
        signature = re.compile(r"void " + function + r"\(int64_t I, int64_t J,[^{};]*uint64_t \*A, uint64_t \*B\) \{")
        call = ('\n  sdcf_native::activation(%s, A, B, I * J, bwA, bwB, '
                'int(log2(scale_in)), int(log2(scale_out)));\n  return;') % ("true" if function == "TanH" else "false")
        library, count = signature.subn(lambda match: match[0] + call, library)
        if count != 1:
            raise ValueError("SCI activation replacement point changed: " + function)
    target = out / "library_fixed_native_workers.cpp"
    target.write_text('#include "native_workers.h"\n' + library)
    overrides["library_fixed.cpp"] = str(target)

    linear = Path(overrides.get("LinearOT/linear-ot.cpp", sci / "src/LinearOT/linear-ot.cpp")).read_text()
    linear, count = re.subn(r"#pragma omp parallel num_threads\(2\)\n(\s*)\{",
        r"auto* sdcf_observer = sdcf_native::team_observer();\n#pragma omp parallel for schedule(static, 1) num_threads(sdcf_native::ot_participants())\n\1for (int sdcf_direction = 0; sdcf_direction < 2; ++sdcf_direction) {\n  sdcf_native::observe_ot_team(sdcf_observer, omp_get_num_threads());", linear)
    if count != 2 or linear.count("omp_get_thread_num()") != 4:
        raise ValueError("SCI OT direction replacement points changed")
    linear = linear.replace("omp_get_thread_num()", "sdcf_direction")
    linear = trace_direction_groups(linear)
    for needle, replacement in [
        ("                            uint64_t *inB, uint64_t *outC) {\n  Eigen::Matrix",
         "                            uint64_t *inB, uint64_t *outC) {\n"
         "  const auto sdcf_copy_begin = sdcf_local_matrix::Clock::now();\n  Eigen::Matrix"),
        ("  eigen_C = eigen_A * eigen_B;",
         "  const auto sdcf_compute_begin = sdcf_local_matrix::Clock::now();\n"
         "  sdcf_local_matrix::metrics.layout_copy_seconds += sdcf_local_matrix::seconds(sdcf_copy_begin, sdcf_compute_begin);\n"
         "  eigen_C = eigen_A * eigen_B;\n"
         "  const auto sdcf_compute_end = sdcf_local_matrix::Clock::now();\n"
         "  sdcf_local_matrix::metrics.compute_seconds += sdcf_local_matrix::seconds(sdcf_compute_begin, sdcf_compute_end);"),
        ("      Arr2DIdxRowM(outC, dim1, dim3, i, j) = eigen_C(i, j);\n    }\n  }\n}",
         "      Arr2DIdxRowM(outC, dim1, dim3, i, j) = eigen_C(i, j);\n    }\n  }\n"
         "  sdcf_local_matrix::metrics.layout_copy_seconds += sdcf_local_matrix::seconds(sdcf_compute_end, sdcf_local_matrix::Clock::now());\n"
         "  ++sdcf_local_matrix::metrics.calls;\n}"),
    ]:
        if linear.count(needle) != 1:
            raise ValueError("SCI Eigen matrix timing point changed")
        linear = linear.replace(needle, replacement, 1)
    target = out / "linear_ot_native_workers.cpp"
    target.write_text('#include "native_workers.h"\n#include "matrix_metrics.h"\n' + linear)
    overrides["LinearOT/linear-ot.cpp"] = str(target)

    def cap_direction_team(text, source):
        text, count = re.subn(r"#pragma omp parallel num_threads\(2\)\n(\s*)\{",
            r"auto* sdcf_observer = sdcf_native::team_observer();\n#pragma omp parallel for schedule(static, 1) num_threads(sdcf_native::ot_participants())\n\1for (int sdcf_direction = 0; sdcf_direction < 2; ++sdcf_direction) {\n  sdcf_native::observe_ot_team(sdcf_observer, omp_get_num_threads());", text)
        if count != 1 or text.count("omp_get_thread_num()") != 1:
            raise ValueError("SCI protocol direction replacement point changed: " + source)
        return '#include "native_workers.h"\n' + trace_direction_groups(text.replace("omp_get_thread_num()", "sdcf_direction"))

    relative = "BuildingBlocks/aux-protocols.cpp"
    aux = Path(overrides.get(relative, sci / "src" / relative)).read_text()
    target = out / "aux_protocols_native_workers.cpp"
    target.write_text(cap_direction_team(aux, relative))
    overrides[relative] = str(target)
    for name in ("millionaire.h", "millionaire_with_equality.h", "equality.h"):
        relative = "Millionaire/" + name
        target = out / "include_override" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(cap_direction_team((sci / "src" / relative).read_text(), relative))


def build(out, sci, generated, jobs=2, adapter=None, extra_sources=(), extra_libraries=(), overrides=(), compiler_defines=()):
    out, sci, generated = (Path(value).resolve() for value in (out, sci, generated))
    out.mkdir(parents=True, exist_ok=True)
    driver, schema = generate(generated.read_text())
    (out / "driver.cpp").write_text(driver)
    (out / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    for name in ("backend_types.h", "measurement.h", "io_counters.h", "runtime_log.h", "round_trace.h", "native_workers.h", "matrix_metrics.h", "native_preprocessing.h", "prepared_got.h",
                 "native_planning.h", "native_plan_replay.h", "native_query_materials.h", "prepared_triples.h", "native_dpf_lut.h", "native_folded.h"):
        shutil.copyfile(HERE / name, out / name)
    for name in ("sci_dpf_bridge.h", "sci_msb_bridge.h", "sci_folded_bridge.h"):
        shutil.copyfile(HERE.parent.parent / "src" / name, out / name)
    shutil.copyfile(adapter or HERE / "backend_adapter.h", out / "backend_adapter.h")
    net_header = native_io_overlay(out, sci)
    native_ot_overlays(out, sci)
    native_planning_overlays(out, sci)
    native_triple_overlay(out, sci)
    native_got_overlay(out)
    overrides = dict(pair.split("=", 1) for pair in overrides)
    unknown = set(overrides) - set(SOURCES)
    if unknown:
        raise ValueError("unknown SCI source overrides: " + repr(unknown))
    native_overlays(out, sci, overrides)
    flags = ["g++", "-O3", "-std=c++17", "-pthread", "-maes", "-msse4.1", "-mavx2", "-mrdseed", "-fopenmp",
             "-DSCI_OT=1", "-DEIGEN_DONT_PARALLELIZE", "-Wno-deprecated-declarations", "-I" + str(out), "-I" + str(out / "include_override"),
             "-I" + str(sci / "src"), "-I" + str(sci / "extern/eigen"), "-I" + str(HERE.parents[1] / "src")]
    flags += ["-D" + value for value in compiler_defines]
    sources = [out / "driver.cpp"] + [Path(overrides[name]).resolve() if name in overrides else sci / "src" / name for name in SOURCES]
    sources += [Path(source).resolve() for source in extra_sources]
    sources.append(HERE.parents[1] / "src/model_bias.cpp")
    sources.append(HERE / "native_workers.cpp")
    sources.append(HERE / "native_dpf_lut.cpp")
    sources.append(HERE / "native_folded.cpp")
    sources.append(HERE / "round_trace.cpp")
    if adapter:
        flags.append("-I" + str(Path(adapter).resolve().parent))
    objects = [out / ("object_%02d.o" % index) for index in range(len(sources))]
    commands = [flags + ["-c", str(source), "-o", str(obj)] for source, obj in zip(sources, objects)]

    def compile_one(item):
        index, command = item
        result = subprocess.run(command, text=True, capture_output=True)
        (out / ("compile_%02d.log" % index)).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("compile failed: " + str(out / ("compile_%02d.log" % index)))

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(compile_one, enumerate(commands)))
    archives = [sci / "build/install/lib" / ("libSCI-" + name + ".a") for name in LIBS]
    libraries = [str(Path(path).resolve()) for path in extra_libraries]
    command = flags + [str(obj) for obj in objects] + libraries
    command += ["-Wl,-rpath," + str(Path(path).parent) for path in libraries]
    command += ["-Wl,--start-group"] + [str(path) for path in archives]
    command += ["-Wl,--end-group", "-lssl", "-lcrypto", "-lgmp", "-o", str(out / "sdcf-runtime")]
    result = subprocess.run(command, text=True, capture_output=True)
    (out / "link.log").write_text(result.stdout + result.stderr)
    (out / "build_commands.json").write_text(json.dumps({"compile": commands, "link": command}, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("link failed: " + str(out / "link.log"))
    versioned = sources + archives + [generated, HERE / "driver.cpp.in", out / "backend_adapter.h",
                                     HERE.parents[1] / "src/model_bias.h",
                                     out / "backend_types.h", out / "measurement.h", out / "io_counters.h", out / "runtime_log.h", out / "round_trace.h", out / "native_workers.h", out / "matrix_metrics.h",
                                     net_header, out / "native_preprocessing.h", out / "prepared_cot.h", out / "prepared_got.h",
                                     out / "native_planning.h", out / "native_plan_replay.h", out / "native_query_materials.h", out / "include_override/utils/prg.h",
                                     out / "prepared_triples.h", out / "include_override/Millionaire/bit-triple-generator.h",
                                     out / "native_dpf_lut.h", out / "native_folded.h", out / "sci_dpf_bridge.h", out / "sci_folded_bridge.h",
                                     out / "include_override/OT/split-iknp.h", out / "include_override/OT/split-kkot.h",
                                     sci / "src/defines.h"] + [out / "include_override/Millionaire" / name
                                     for name in ("millionaire.h", "millionaire_with_equality.h", "equality.h")] + [Path(path) for path in libraries]
    manifest = {"generated": str(generated), "generated_sha256": digest(generated),
                "binary_sha256": digest(out / "sdcf-runtime"), "numeric_mode": "sci_compatible",
                "DIV_RESCALING": bool(re.search(r"^#define DIV_RESCALING\b", (sci / "src/defines.h").read_text(), re.M)),
                "sources": {str(path): digest(path) for path in versioned},
                "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                "ldd": subprocess.check_output(["ldd", str(out / "sdcf-runtime")], text=True)}
    (out / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out / "sdcf-runtime"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sci", type=Path, default=Path("/home/jiang/EzPC/SCI"))
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--extra-source", action="append", default=[])
    parser.add_argument("--extra-library", action="append", default=[])
    parser.add_argument("--override", action="append", default=[], help="SCI-relative.cpp=/absolute/private-copy.cpp")
    args = parser.parse_args()
    print(build(args.out, args.sci, args.generated, args.jobs, args.adapter,
                args.extra_source, args.extra_library, args.override))
