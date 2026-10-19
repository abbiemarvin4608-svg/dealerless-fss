#include "cnn_bridge.h"
#include "sci_msb_bridge.h"
#include "sci_conversion_bridge.h"
#include "LinearOT/linear-ot.h"
#include "BuildingBlocks/truncation.h"
#include "NonLinear/relu-ring.h"
#include "../sci_beaver/beaver.h"
#include "../segmented_dcf_runtime/io_counters.h"
#include "../segmented_dcf_runtime/round_trace.h"
#include "native_query_materials.h"
#include "cnn_native_spool.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <regex>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

using boost::property_tree::ptree;
using Words = std::vector<uint64_t>;
using Bytes = std::vector<uint8_t>;
using Shape = std::vector<int>;
using Clock = std::chrono::steady_clock;
extern void cnn_native_avg(int, sci::IOPack*, sci::OTPack*, ReLUProtocol<uint64_t>*,
    int, uint64_t*, uint64_t*, int, uint64_t);

namespace {
uint64_t mask(int bits) { return bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1; }
double cpu() { timespec t{}; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
double seconds(Clock::time_point from) { return std::chrono::duration<double>(Clock::now() - from).count(); }
size_t elements(const Shape& shape) {
    size_t n = 1;
    for (int d : shape) { if (d <= 0 || n > SIZE_MAX / size_t(d)) throw std::runtime_error("invalid tensor shape"); n *= d; }
    return n;
}
Shape shapeOf(const ptree& node) { Shape result; for (auto& x : node) result.push_back(x.second.get_value<int>()); return result; }
Words readWords(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open integer codes: " + path);
    auto n = stream.tellg(); if (n < 0 || n % 8) throw std::runtime_error("uint64 file size is not a multiple of eight");
    Words out(size_t(n) / 8); stream.seekg(0);
    stream.read(reinterpret_cast<char*>(out.data()), n); if (!stream) throw std::runtime_error("integer code read failed");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    for (auto& x : out) x = __builtin_bswap64(x);
#endif
    return out;
}
ptree wordArray(const Words& values) { ptree out; for (auto x : values) { ptree v; v.put_value(x); out.push_back({"", v}); } return out; }
void writeTypedJson(std::ostream& stream, const ptree& tree, const std::string& key = "") {
    if (tree.empty()) {
        const auto value = tree.get_value<std::string>();
        static const std::regex number("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?");
        if (value.empty() && (key == "queries" || key == "layers" || key == "shape" || key == "output_codes")) stream << "[]";
        else if (value == "true" || value == "false" || std::regex_match(value, number)) stream << value;
        else stream << '"' << boost::property_tree::json_parser::create_escapes(value) << '"';
        return;
    }
    const bool array = tree.begin()->first.empty(); stream << (array ? '[' : '{'); bool first = true;
    for (const auto& item : tree) {
        if (!first) stream << ','; first = false;
        if (!array) stream << '"' << boost::property_tree::json_parser::create_escapes(item.first) << "\":";
        writeTypedJson(stream, item.second, item.first);
    }
    stream << (array ? ']' : '}');
}
template<class T> void spoolWrite(FILE* file, const std::vector<T>& v) {
    uint64_t n = v.size();
    if (std::fwrite(&n, sizeof(n), 1, file) != 1 || (n && std::fwrite(v.data(), sizeof(T), n, file) != n))
        throw std::runtime_error("Beaver spool write failed");
}
template<class T> std::vector<T> spoolRead(FILE* file, size_t expected) {
    uint64_t n = 0; if (std::fread(&n, sizeof(n), 1, file) != 1 || n != expected) throw std::runtime_error("Beaver spool shape mismatch");
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, file) != n) throw std::runtime_error("Beaver spool read failed"); return v;
}
struct Node {
    std::string name, op, output;
    std::vector<std::string> inputs;
    Shape shape, weight_shape;
    uint64_t weight_offset = 0, bias_offset = 0;
    int bias_count = 0, kernel = 1, stride = 1, padding = 0;
};
struct MatCall { int rows, inner, cols, bits; };
struct Options {
    std::string backend = "SCI", graph, parameters, input, output, audit_dir, optimization_profile = "optimized";
    int role = 0, port = 32000, queries = 1, tile = 64;
    bool diagnostic = false, plan_only = false;
    uint64_t spool_budget = 0, native_spool_budget = 0;
    double offline_budget = 0;
};
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--diagnostic") { o.diagnostic = true; continue; }
        if (key == "--plan-only") { o.plan_only = true; continue; }
        if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
        std::string value = argv[++i];
        if (key == "--backend") o.backend = value;
        else if (key == "--graph") o.graph = value;
        else if (key == "--parameters") o.parameters = value;
        else if (key == "--input") o.input = value;
        else if (key == "--audit-dir") { o.audit_dir = value; o.diagnostic = true; }
        else if (key == "--output") o.output = value;
        else if (key == "--role") o.role = std::stoi(value);
        else if (key == "--port") o.port = std::stoi(value);
        else if (key == "--queries") o.queries = std::stoi(value);
        else if (key == "--tile") o.tile = std::stoi(value);
        else if (key == "--spool-budget") o.spool_budget = std::stoull(value);
        else if (key == "--native-spool-budget") o.native_spool_budget = std::stoull(value);
        else if (key == "--optimization-profile") o.optimization_profile = value;
        else if (key == "--offline-budget-seconds") o.offline_budget = std::stod(value);
        else throw std::runtime_error("unknown option " + key);
    }
    if ((o.role != 1 && o.role != 2) || o.graph.empty() || o.output.empty() || o.queries < 1 || o.tile < 1 || o.tile > 4096)
        throw std::runtime_error("require role 1/2, --graph, --output, positive queries and tile 1..4096");
    if (o.backend != "SCI" && o.backend != "SCI_preprocessed" && o.backend != "SCI_ours" && o.backend != "SCI_ours_beaver") throw std::runtime_error("unknown backend");
    if (o.optimization_profile != "optimized" && o.optimization_profile != "baseline") throw std::runtime_error("unknown optimization profile");
    return o;
}
struct Runtime {
    Options options;
    ptree graph, result, query_results, layer_results, diagnostic_manifest;
    std::string input_name, output_name;
    int bits = 32, frac = 12, query = 0;
    std::map<std::string, Shape> shapes;
    std::vector<Node> nodes;
    Words parameters;
    std::unique_ptr<sci::IOPack> io;
    std::unique_ptr<sci::OTPack> ot;
    std::unique_ptr<LinearOT> linear;
    std::unique_ptr<AuxProtocols> aux;
    std::unique_ptr<Truncation> trunc;
    std::unique_ptr<ReLURingProtocol<uint64_t>> relu_native;
    void* cnn = nullptr;
    void* conversion = nullptr;
    FILE* beaver_file = nullptr;
    uint64_t beaver_bytes = 0, conversion_material_bytes = 0;
    double native_plan_seconds = 0;
    std::vector<CnnCall> primitive_calls;
    std::vector<segfss_sci_conversion_call> conversion_calls;
    std::vector<MatCall> matrix_calls;
    size_t primitive_cursor = 0, conversion_cursor = 0, matrix_cursor = 0;
    bool planning = true, native_planning = false;
    sdcf_beaver::MatmulWorkspace beaver_workspace;
    bool ours() const { return options.backend == "SCI_ours" || options.backend == "SCI_ours_beaver"; }
    bool nativePrepared() const { return options.backend == "SCI_preprocessed"; }
    bool beaver() const { return options.backend == "SCI_ours_beaver"; }
    ~Runtime() {
        cnn_spool::discard();
        if (beaver_file) std::fclose(beaver_file);
        if (conversion) segfss_sci_conversion_destroy(conversion);
        if (cnn) segfss_sci_cnn_destroy(cnn);
    }
    void load() {
        const bool optimized = options.optimization_profile == "optimized";
        for (const char* flag : {"SEGFSS_FUSE_DOMAINS", "SEGFSS_PACK_BITS", "SEGFSS_PACK_RING",
                "SEGFSS_AES_ONLINE", "SEGFSS_AES_OFFLINE", "SEGFSS_PRUNE_EQUALITY",
                "SEGFSS_PRUNE_LEAF_HASHES", "SEGFSS_MERGE_LEAF_CW", "SEGFSS_REUSE_GENERATION_WORKSPACE",
                "SEGFSS_FUSE_SIGN_WRAP", "SEGFSS_NATIVE_RING_COT"})
            setenv(flag, optimized ? "1" : "0", 1);
        setenv("SEGFSS_BIT_BLOCK_MODE", optimized ? "direct_cot" : "direct_ot", 1);
        setenv("SEGFSS_MATERIAL_LAYOUT", optimized ? "contiguous" : "legacy", 1);
        setenv("SEGFSS_SEGMENT_POLICY", optimized ? "depth-cap-8" : "fixed-4", 1);
        setenv("SEGFSS_CGGM_THREADS", "1", 0);
        setenv("SEGFSS_GENERATION_TILE_KEYS", "256", 0);
        setenv("SEGFSS_GENERATION_BUDGET_BYTES", "268435456", 0);
        setenv("SEGFSS_GENERATION_RESERVE_BYTES", "67108864", 0);
        setenv("SEGFSS_PREFETCH_BYTES", "8388608", 0);
        setenv("SDCF_NATIVE_PREPROCESSING", nativePrepared() ? "1" : "0", 1);
        boost::property_tree::read_json(options.graph, graph);
        bits = graph.get<int>("bits"); frac = graph.get<int>("frac");
        if ((bits != 8 && bits != 16 && bits != 24 && bits != 32) || frac < 0 || frac >= bits) throw std::runtime_error("unsupported fixed point contract");
        input_name = graph.get<std::string>("input"); output_name = graph.get<std::string>("output");
        for (auto& value : graph.get_child("tensors")) shapes[value.first] = shapeOf(value.second);
        for (auto& value : graph.get_child("nodes")) {
            auto& p = value.second; Node n;
            n.name = p.get<std::string>("name"); n.op = p.get<std::string>("op"); n.output = p.get<std::string>("output");
            for (auto& x : p.get_child("inputs")) n.inputs.push_back(x.second.get_value<std::string>());
            n.shape = shapeOf(p.get_child("shape")); shapes[n.output] = n.shape;
            if (auto s = p.get_child_optional("weight_shape")) n.weight_shape = shapeOf(*s);
            n.weight_offset = p.get<uint64_t>("weight_offset", 0); n.bias_offset = p.get<uint64_t>("bias_offset", 0);
            n.bias_count = p.get<int>("bias_count", 0); n.kernel = p.get<int>("kernel", 1);
            n.stride = p.get<int>("stride", 1); n.padding = p.get<int>("padding", 0);
            if (n.inputs.empty()) throw std::runtime_error("node lacks an input"); nodes.push_back(n);
        }
        if (options.role == 1) {
            if (options.parameters.empty()) throw std::runtime_error("server requires --parameters");
            parameters = readWords(options.parameters);
            for (auto& n : nodes) if (n.op == "conv2d" || n.op == "linear") {
                if (n.weight_offset + elements(n.weight_shape) > parameters.size() || n.bias_offset + n.bias_count > parameters.size())
                    throw std::runtime_error("parameter offset exceeds file");
            }
            for (auto& x : parameters) x &= mask(bits);
        }
        result.put("status", "running"); result.put("role", options.role); result.put("backend", options.backend);
        result.put("model", graph.get<std::string>("model")); result.put("bits", bits); result.put("frac", frac);
        result.put("method.linear", beaver() ? "existing_sdcf_beaver_algebra_mapped" : "native_SCI_LinearOT_Alice_has_A");
        result.put("method.nonlinear", ours() ? "existing_privateMsb_BitXA_sciTruncate" : "native_SCI_AuxProtocols_Truncation");
        result.put("method.prepared", beaver() || nativePrepared());
        result.put("method.preparation", nativePrepared() ? "native_random_OT_and_bit_triples_before_input_private_disk_spool" :
            (beaver() ? "all_material_before_input_private_disk_spool" : (ours() ? "nonlinear_prepared_native_linear_online" : "native_material_generated_online")));
        result.put("method.avgpool", "shared_exact_public_division_MSB_AND_selection");
        result.put("method.optimization_profile", options.optimization_profile);
        result.put("method.activation_specific_optimizations", "not_applicable_no_sigmoid_tanh_or_LUT");
        result.put("method.rnn_graph_parallelism", "not_applicable_sequential_CNN_graph");
        result.put("method.conv_layout", "OIHW_x_im2col_NCHW_position_tiles");
        result.put("method.linear_accumulator", "same_ring_modulo_then_signed_floor_shift");
        result.put("method.maxpool", "nonnegative_ReLU_precondition_padding_zero");
        result.put("method.native_max_ot_payload", 1 << 18);
        result.put("method.matrix_max_products_per_call", 1 << 20);
        result.put("method.diagnostic", options.diagnostic);
    }
    struct Snapshot { Clock::time_point time; double process_cpu; uint64_t sent, received; };
    Snapshot snapshot() {
        uint64_t sent = sdcf_measure::sent.load(), received = sdcf_measure::received.load(), bytes = 0, s = 0, r = 0;
        if (cnn) { segfss_sci_cnn_counters(cnn, &s, &r, &bytes); sent += s; received += r; }
        if (conversion) { segfss_sci_conversion_counters c{}; segfss_sci_conversion_get_counters(conversion, &c); sent += c.sent_bytes; received += c.received_bytes; }
        return {Clock::now(), cpu(), sent, received};
    }
    ptree delta(const Snapshot& before) {
        auto after = snapshot(); ptree p; p.put("seconds", std::chrono::duration<double>(after.time - before.time).count());
        p.put("process_cpu_seconds", after.process_cpu - before.process_cpu); p.put("sent_bytes", after.sent - before.sent);
        p.put("received_bytes", after.received - before.received); return p;
    }
    void setup() {
        sdcf_round::initialize(options.role);
        segfss_sci_set_round_trace_callbacks(sdcf_round_trace_io, sdcf_round_actor_fork,
            sdcf_round_actor_enter, sdcf_round_actor_leave, sdcf_round_actor_join);
        segfss_sci_set_round_scope_callbacks(sdcf_round_scope_begin, sdcf_round_scope_end);
        sdcf_round::phase_begin("setup", -1); auto before = snapshot();
        io = std::make_unique<sci::IOPack>(options.role, options.port, "127.0.0.1");
        ot = std::make_unique<sci::OTPack>(io.get(), options.role);
        linear = std::make_unique<LinearOT>(options.role, io.get(), ot.get());
        aux = std::make_unique<AuxProtocols>(options.role, io.get(), ot.get());
        trunc = std::make_unique<Truncation>(options.role, io.get(), ot.get());
        relu_native = std::make_unique<ReLURingProtocol<uint64_t>>(options.role, 0, io.get(), bits, 4, ot.get());
        if (nativePrepared()) sdcf_planning::register_pack(ot.get(), 0);
        if (ours()) {
            cnn = segfss_sci_cnn_create(options.role, "127.0.0.1", options.port + 20);
            if (!cnn) throw std::runtime_error(segfss_sci_cnn_error());
            conversion = segfss_sci_conversion_create(options.role, "127.0.0.1", options.port + 40, 0, 4);
            if (!conversion) throw std::runtime_error(segfss_sci_conversion_error());
        }
        result.add_child("setup", delta(before)); sdcf_round::phase_end("setup", -1);
    }
    Bytes msb(const Words& x, int width) {
        Bytes out(x.size());
        for (size_t base = 0; base < x.size(); base += 4096) {
            int n = std::min<size_t>(4096, x.size() - base); CnnCall call{1, n, width};
            if (planning) { primitive_calls.push_back(call); continue; }
            if (ours()) {
                if (segfss_sci_cnn_online(cnn, &call, x.data() + base, nullptr, nullptr, nullptr, out.data() + base)) throw std::runtime_error(segfss_sci_cnn_error());
                ++primitive_cursor;
            } else aux->MSB(const_cast<uint64_t*>(x.data() + base), out.data() + base, n, width);
        }
        return out;
    }
    Bytes conjunction(const Bytes& a, const Bytes& b) {
        Bytes out(a.size());
        for (size_t base = 0; base < a.size(); base += 4096) {
            int n = std::min<size_t>(4096, a.size() - base); CnnCall call{3, n, bits};
            if (planning) { primitive_calls.push_back(call); continue; }
            if (ours()) {
                if (segfss_sci_cnn_online(cnn, &call, nullptr, a.data() + base, b.data() + base, nullptr, out.data() + base)) throw std::runtime_error(segfss_sci_cnn_error());
                ++primitive_cursor;
            } else aux->AND(const_cast<uint8_t*>(a.data() + base), const_cast<uint8_t*>(b.data() + base), out.data() + base, n);
        }
        return out;
    }
    Words select(const Bytes& selection, const Words& x, int width) {
        Words out(x.size());
        for (size_t base = 0; base < x.size(); base += 4096) {
            int n = std::min<size_t>(4096, x.size() - base); CnnCall call{2, n, width};
            if (planning) { primitive_calls.push_back(call); continue; }
            if (ours()) {
                if (segfss_sci_cnn_online(cnn, &call, x.data() + base, selection.data() + base, nullptr, out.data() + base, nullptr)) throw std::runtime_error(segfss_sci_cnn_error());
                ++primitive_cursor;
            } else aux->multiplexer(const_cast<uint8_t*>(selection.data() + base), const_cast<uint64_t*>(x.data() + base), out.data() + base, n, width, width);
        }
        return out;
    }
    Words shift(const Words& x) {
        if (!frac) return x;
        Words out(x.size());
        for (size_t base = 0; base < x.size(); base += 4096) {
            int n = std::min<size_t>(4096, x.size() - base);
            segfss_sci_conversion_call call{n, bits, bits, frac, 1, 1};
            if (planning) { conversion_calls.push_back(call); continue; }
            if (ours()) {
                segfss_sci_conversion_metrics metrics{};
                if (segfss_sci_conversion_online(conversion, &call, x.data() + base, out.data() + base, &metrics)) throw std::runtime_error(segfss_sci_conversion_error());
                conversion_material_bytes += metrics.material_bytes;
                ++conversion_cursor;
            } else trunc->truncate(n, const_cast<uint64_t*>(x.data() + base), out.data() + base, frac, bits, true);
        }
        for (auto& v : out) v &= mask(bits); return out;
    }
    Words relu(const Words& x) {
        auto signs = msb(x, bits); if (options.role == 1) for (auto& s : signs) s ^= 1;
        return select(signs, x, bits);
    }
    Words matmul(const Words& a, const Words& b, int rows, int inner, int cols) {
        if (planning) { matrix_calls.push_back({rows, inner, cols, bits}); return Words(size_t(rows) * cols); }
        Words out(size_t(rows) * cols);
        if (beaver()) {
            sdcf_beaver::MatmulMaterial material; material.rows = rows; material.inner = inner; material.cols = cols; material.bits = bits;
            material.options = {true, true};
            material.u = spoolRead<uint64_t>(beaver_file, size_t(rows) * inner);
            material.v = spoolRead<uint64_t>(beaver_file, size_t(inner) * cols);
            material.w = spoolRead<uint64_t>(beaver_file, size_t(rows) * cols);
            sdcf_beaver::multiply_matmul(options.role, *io, *linear, a, b, material, out,
                options.optimization_profile == "optimized" ? &beaver_workspace : nullptr); ++matrix_cursor;
        } else {
            linear->matmul_cross_terms(rows, inner, cols, const_cast<uint64_t*>(a.data()), const_cast<uint64_t*>(b.data()), out.data(), bits, bits, bits, true, MultMode::Alice_has_A);
            if (options.role == 1 && !native_planning) {
                Words local(out.size()); linear->matmul_cleartext(rows, inner, cols, const_cast<uint64_t*>(a.data()), const_cast<uint64_t*>(b.data()), local.data(), true);
                for (size_t i = 0; i < out.size(); ++i) out[i] += local[i];
            }
        }
        for (auto& x : out) x &= mask(bits); return out;
    }
    // Exact floor division of the signed ring representative by a positive public divisor.
    Words publicDivide(const Words& input, uint64_t d) {
        const uint64_t ring = uint64_t(1) << bits, half = ring >> 1, ring_mask = ring - 1;
        Words z = input, low(input.size()), r(input.size()), quotient(input.size());
        Bytes ah(input.size()), bh(input.size()), t(input.size());
        for (size_t i = 0; i < z.size(); ++i) {
            if (options.role == 1) z[i] = (z[i] + half) & ring_mask;
            low[i] = z[i] & (half - 1); r[i] = z[i] % d; quotient[i] = z[i] / d;
            const auto high = (z[i] >> (bits - 1)) & 1;
            (options.role == 1 ? ah[i] : bh[i]) = high; t[i] = high;
        }
        auto carry = msb(low, bits); auto both = conjunction(ah, bh); auto tc = conjunction(t, carry);
        Bytes wrap(input.size()); for (size_t i = 0; i < wrap.size(); ++i) wrap[i] = both[i] ^ tc[i];
        int small_bits = 8; while ((uint64_t(1) << (small_bits - 1)) <= 4 * d) small_bits += 8;
        auto greaterEqual = [&](int64_t threshold) {
            if (threshold <= 0 || threshold > int64_t(2 * d - 2)) return Words(input.size(), options.role == 1 && threshold <= 0 ? 1 : 0);
            Words difference = r; if (options.role == 1) for (auto& v : difference) v = (v - uint64_t(threshold)) & mask(small_bits);
            auto sign = msb(difference, small_bits); if (options.role == 1) for (auto& v : sign) v ^= 1;
            Words one(input.size(), options.role == 1 ? 1 : 0); return select(sign, one, bits);
        };
        const int64_t rh = half % d, rr = ring % d;
        auto a0 = greaterEqual(rh), a1 = greaterEqual(rh + d);
        auto b0 = greaterEqual(rh + rr - int64_t(d)), b1 = greaterEqual(rh + rr), b2 = greaterEqual(rh + rr + d);
        Words k0(input.size()), difference(input.size());
        for (size_t i = 0; i < k0.size(); ++i) {
            k0[i] = (a0[i] + a1[i] - (options.role == 1 ? 1 : 0)) & ring_mask;
            const uint64_t k1 = b0[i] + b1[i] + b2[i] - (options.role == 1 ? 2 : 0);
            difference[i] = (k1 - k0[i] - (options.role == 1 ? ring / d : 0)) & ring_mask;
        }
        auto correction = select(wrap, difference, bits);
        for (size_t i = 0; i < quotient.size(); ++i)
            quotient[i] = (quotient[i] + k0[i] + correction[i] - (options.role == 1 ? half / d : 0)) & ring_mask;
        return quotient;
    }
    Words executeNode(const Node& n, const std::map<std::string, Words>& tensors) {
        const auto& input = tensors.at(n.inputs.at(0)); const auto& inshape = shapes.at(n.inputs.at(0));
        if (n.op == "relu") return relu(input);
        if (n.op == "add") {
            const auto& other = tensors.at(n.inputs.at(1)); if (input.size() != other.size()) throw std::runtime_error("ADD shape mismatch");
            Words out = input; for (size_t i = 0; i < out.size(); ++i) out[i] = (out[i] + other[i]) & mask(bits); return out;
        }
        if (n.op == "avgpool") {
            if (inshape.size() != 4 || n.shape.size() < 2) throw std::runtime_error("AvgPool requires NCHW global pooling");
            const int spatial = inshape[2] * inshape[3]; Words sum(size_t(inshape[0]) * inshape[1]);
            if (elements(n.shape) != sum.size()) throw std::runtime_error("only global AvgPool is supported");
            if (!planning && !native_planning) for (size_t c = 0; c < sum.size(); ++c) for (int p = 0; p < spatial; ++p) sum[c] = (sum[c] + input[c * spatial + p]) & mask(bits);
            return publicDivide(sum, spatial);
        }
        if (n.op == "maxpool") {
            if (inshape.size() != 4 || n.shape.size() != 4) throw std::runtime_error("MaxPool requires NCHW");
            bool nonnegative = false;
            for (const auto& preceding : nodes) if (preceding.output == n.inputs[0]) nonnegative = preceding.op == "relu";
            if (!nonnegative) throw std::runtime_error("MaxPool requires an immediate ReLU producer for bounded signed differences");
            const int channels = inshape[1], ih = inshape[2], iw = inshape[3], oh = n.shape[2], ow = n.shape[3];
            Words out(elements(n.shape));
            for (int k = 0; k < n.kernel * n.kernel; ++k) {
                Words candidate(out.size());
                if (!planning && !native_planning) for (int b = 0; b < inshape[0]; ++b) for (int c = 0; c < channels; ++c) for (int h = 0; h < oh; ++h) for (int w = 0; w < ow; ++w) {
                    int y = h * n.stride - n.padding + k / n.kernel, x = w * n.stride - n.padding + k % n.kernel;
                    if (y >= 0 && y < ih && x >= 0 && x < iw) candidate[((b * channels + c) * oh + h) * ow + w] = input[((b * channels + c) * ih + y) * iw + x];
                }
                if (!k) { out = std::move(candidate); continue; }
                Words diff(out.size()); for (size_t i = 0; i < out.size(); ++i) diff[i] = (candidate[i] - out[i]) & mask(bits);
                auto positive = relu(diff); for (size_t i = 0; i < out.size(); ++i) out[i] = (out[i] + positive[i]) & mask(bits);
            }
            return out;
        }
        if (n.op != "conv2d" && n.op != "linear") throw std::runtime_error("unsupported operator " + n.op);
        const bool conv = n.op == "conv2d";
        int outputs = n.weight_shape.at(0), inner = int(elements(n.weight_shape) / outputs);
        const int batches = inshape.at(0);
        const int positions = conv ? n.shape.at(2) * n.shape.at(3) : 1;
        if (!conv && input.size() != size_t(batches) * inner) throw std::runtime_error("linear flattened dimension mismatch");
        Words out(elements(n.shape));
        for (int batch = 0; batch < batches; ++batch) for (int p0 = 0; p0 < positions; p0 += options.tile) {
            int cols = std::min(options.tile, positions - p0);
            int row_tile = std::min(outputs, std::max(1, int((uint64_t(1) << 20) / (uint64_t(inner) * cols))));
            Words b(size_t(inner) * cols);
            if (!planning && !native_planning) for (int k = 0; k < inner; ++k) for (int j = 0; j < cols; ++j) {
                if (!conv) b[k * cols + j] = input[size_t(batch) * inner + k];
                else {
                    int channel = k / (n.kernel * n.kernel), ky = (k / n.kernel) % n.kernel, kx = k % n.kernel;
                    int y = ((p0 + j) / n.shape[3]) * n.stride - n.padding + ky;
                    int x = ((p0 + j) % n.shape[3]) * n.stride - n.padding + kx;
                    if (y >= 0 && y < inshape[2] && x >= 0 && x < inshape[3]) b[k * cols + j] = input[((batch * inshape[1] + channel) * inshape[2] + y) * inshape[3] + x];
                }
            }
            for (int row = 0; row < outputs; row += row_tile) {
                const int rows = std::min(row_tile, outputs - row); Words a(size_t(rows) * inner);
                if (!planning && !native_planning && options.role == 1) std::copy_n(parameters.begin() + n.weight_offset + size_t(row) * inner, a.size(), a.begin());
                auto product = matmul(a, b, rows, inner, cols);
                for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) out[(size_t(batch) * outputs + row + i) * positions + p0 + j] = product[i * cols + j];
            }
        }
        out = shift(out);
        if (!planning && !native_planning && options.role == 1 && n.bias_count) {
            if (n.bias_count != outputs) throw std::runtime_error("bias dimension mismatch");
            for (int batch = 0; batch < batches; ++batch) for (int c = 0; c < outputs; ++c) for (int p = 0; p < positions; ++p)
                out[(size_t(batch) * outputs + c) * positions + p] = (out[(size_t(batch) * outputs + c) * positions + p] + parameters[n.bias_offset + c]) & mask(bits);
        }
        return out;
    }
    Words evaluate(Words input, ptree* diagnostics = nullptr) {
        std::map<std::string, Words> tensors; tensors[input_name] = std::move(input);
        std::map<std::string, int> uses; for (auto& n : nodes) for (auto& name : n.inputs) ++uses[name]; ++uses[output_name];
        int layer = 0;
        for (auto& n : nodes) {
            auto before = snapshot();
            { sdcf_round::SemanticScope scope(planning ? "folded_prepare" : "folded_call", query, layer);
              tensors[n.output] = executeNode(n, tensors); }
            if (tensors[n.output].size() != elements(n.shape)) throw std::runtime_error("node output shape mismatch: " + n.name);
            if (!planning && !native_planning) {
                auto record = delta(before); record.put("name", n.name); record.put("op", n.op); record.put("query", query); record.put("layer", layer);
                record.put("elements", tensors[n.output].size()); layer_results.push_back({"", record});
                std::fprintf(stderr, "CNN_LAYER_COMPLETE query=%d layer=%d name=%s seconds=%.6f\n", query, layer, n.name.c_str(), record.get<double>("seconds"));
                if (diagnostics) {
                    std::string directory = options.audit_dir.empty() ? options.output + ".audit" : options.audit_dir;
                    std::filesystem::create_directories(directory);
                    const auto path = std::filesystem::absolute(std::filesystem::path(directory) /
                        ("query" + std::to_string(query) + "_node" + std::to_string(layer) + ".bin"));
                    std::ofstream file(path, std::ios::binary);
                    const auto& values = tensors[n.output];
                    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(uint64_t));
                    if (!file) throw std::runtime_error("diagnostic share write failed");
                    ptree item; item.put("path", path.string()); item.put("bits", bits); item.put("layout", "NCHW");
                    item.put("sharing", "additive_share"); ptree shape;
                    for (auto dimension : n.shape) { ptree value; value.put_value(dimension); shape.push_back({"", value}); }
                    item.add_child("shape", shape); diagnostics->push_back({n.output, item});
                }
            }
            for (auto& name : n.inputs) if (--uses[name] == 0 && name != n.output) tensors.erase(name);
            ++layer;
        }
        return std::move(tensors.at(output_name));
    }
    void plan() {
        planning = true; evaluate(Words(elements(shapes.at(input_name)))); planning = false;
        result.put("plan.primitive_calls", primitive_calls.size()); result.put("plan.truncation_calls", conversion_calls.size()); result.put("plan.matrix_calls", matrix_calls.size());
    }
    void prepare() {
        if (nativePrepared()) {
            sdcf_preprocess::begin_prepare(query, true);
            const auto plan_start = Clock::now();
            const auto epoch = sdcf_preprocess::generation_epoch.load();
            const auto sent = sdcf_measure::sent.load(), received = sdcf_measure::received.load();
            sdcf_planning::begin(); native_planning = true;
            evaluate(Words(elements(shapes.at(input_name))));
            native_planning = false;
            const auto plan = sdcf_planning::end();
            if (epoch != sdcf_preprocess::generation_epoch.load() || sent != sdcf_measure::sent.load() || received != sdcf_measure::received.load())
                throw std::runtime_error("CNN public native planning changed protocol state");
            sdcf_planning::compare_plan(io->io, options.role, plan);
            native_plan_seconds = seconds(plan_start);
            cnn_spool::begin(options.native_spool_budget ? options.native_spool_budget : options.spool_budget);
            TripleGenerator generator(options.role, io.get(), ot.get());
            sdcf_planning::prepare_query_materials(query, {&generator});
            sdcf_preprocess::seal_prepare(query); cnn_spool::seal();
            sdcf_preprocess::material_ready(query);
            return;
        }
        if (!ours()) return;
        const auto start = Clock::now();
        if (segfss_sci_cnn_prepare(cnn, primitive_calls.data(), primitive_calls.size(), options.spool_budget)) throw std::runtime_error(segfss_sci_cnn_error());
        if (segfss_sci_conversion_prepare(conversion, conversion_calls.data(), conversion_calls.size())) throw std::runtime_error(segfss_sci_conversion_error());
        if (beaver()) {
            beaver_file = std::tmpfile(); if (!beaver_file) throw std::runtime_error("cannot create Beaver material spool");
            size_t prepared = 0;
            for (auto& call : matrix_calls) {
                if (options.offline_budget && seconds(start) > options.offline_budget) throw std::runtime_error("preparation exceeded --offline-budget-seconds");
                auto material = sdcf_beaver::prepare_matmul(options.role, *linear, call.rows, call.inner, call.cols, call.bits, 16, {true, true});
                spoolWrite(beaver_file, material.u); spoolWrite(beaver_file, material.v); spoolWrite(beaver_file, material.w);
                auto pos = std::ftell(beaver_file); if (pos < 0) throw std::runtime_error("Beaver spool accounting failed"); beaver_bytes = pos;
                if (options.spool_budget && beaver_bytes > options.spool_budget) throw std::runtime_error("Beaver material exceeds --spool-budget");
                if (++prepared % 100 == 0 || prepared == matrix_calls.size()) std::fprintf(stderr,
                    "CNN_BEAVER_PREPARED calls=%zu/%zu seconds=%.6f spool_bytes=%llu\n", prepared, matrix_calls.size(), seconds(start), static_cast<unsigned long long>(beaver_bytes));
            }
            if (std::fflush(beaver_file) || std::fseek(beaver_file, 0, SEEK_SET)) throw std::runtime_error("Beaver spool rewind failed");
        }
    }
    Words input() {
        Words input(elements(shapes.at(input_name)));
        if (options.role == 2) {
            if (!options.input.empty()) { input = readWords(options.input); if (input.size() != elements(shapes.at(input_name))) throw std::runtime_error("input code dimensions differ"); }
            else { uint32_t state = 12345; for (auto& x : input) { state = state * 1664525u + 1013904223u; x = (uint64_t(int64_t((state >> 16) % 8193) - 4096)) & mask(bits); } }
        }
        for (auto& x : input) x &= mask(bits);
        return input;
    }
    Words reveal(const Words& shares) {
        Words remote(shares.size()), out = shares;
        if (options.role == 1) io->io->send_data(shares.data(), shares.size() * 8);
        else { io->io->recv_data(remote.data(), remote.size() * 8); for (size_t i = 0; i < out.size(); ++i) out[i] = (out[i] + remote[i]) & mask(bits); }
        return out;
    }
    void save() {
        result.put_child("queries", query_results); result.put_child("layers", layer_results);
        std::ofstream stream(options.output);
        writeTypedJson(stream, result); stream << '\n'; if (!stream) throw std::runtime_error("result JSON write failed");
    }
    void run() {
        auto started = Clock::now(); load(); result.put("pre_setup.load_seconds", seconds(started));
        started = Clock::now(); plan(); result.put("pre_setup.plan_seconds", seconds(started));
        if (options.plan_only) { result.put("status", "planned"); save(); return; }
        setup(); save();
        for (query = 0; query < options.queries; ++query) {
            ptree q, phases; q.put("query", query);
            conversion_material_bytes = 0; beaver_bytes = 0; native_plan_seconds = 0;
            sdcf_round::phase_begin("offline", query); auto before = snapshot(); prepare(); phases.add_child("offline", delta(before)); sdcf_round::phase_end("offline", query);
            if (nativePrepared()) sdcf_preprocess::begin_input(query);
            sdcf_round::phase_begin("input", query); before = snapshot(); auto x = input();
            // Exchange a readiness token without revealing the client input.
            uint8_t token = 1, remote = 0;
            if (options.role == 1) { io->io->send_data(&token, 1); io->io->recv_data(&remote, 1); }
            else { io->io->recv_data(&remote, 1); io->io->send_data(&token, 1); }
            phases.add_child("input", delta(before)); sdcf_round::phase_end("input", query);
            sdcf_round::phase_begin("online", query); before = snapshot(); ptree diagnostic_shares;
            if (nativePrepared()) sdcf_preprocess::enter_online(query);
            const auto native_epoch = sdcf_preprocess::generation_epoch.load();
            auto output = evaluate(std::move(x), options.diagnostic ? &diagnostic_shares : nullptr);
            q.put("native_generation_epoch_delta", sdcf_preprocess::generation_epoch.load() - native_epoch);
            if (nativePrepared()) {
                if (sdcf_preprocess::generation_epoch.load() != native_epoch) throw std::runtime_error("SCI_preprocessed generated material online");
                sdcf_triples::balance(query); sdcf_preprocess::leave_online(query);
                q.put("native_spool_bytes", cnn_spool::bytes);
                q.put("native_spool_records", cnn_spool::written);
                q.put("native_spool_consumed", cnn_spool::loaded);
                cnn_spool::finish();
            }
            if (ours()) {
                uint64_t sent = 0, received = 0, bytes = 0;
                if (segfss_sci_cnn_counters(cnn, &sent, &received, &bytes)) throw std::runtime_error(segfss_sci_cnn_error());
                q.put("ours_nonlinear_spool_bytes", bytes);
                q.put("conversion_material_bytes", conversion_material_bytes);
                if (segfss_sci_cnn_finish(cnn)) throw std::runtime_error(segfss_sci_cnn_error());
                if (segfss_sci_conversion_finish_query(conversion)) throw std::runtime_error(segfss_sci_conversion_error());
                if (beaver() && matrix_cursor != matrix_calls.size()) throw std::runtime_error("Beaver inventory was not consumed");
            }
            phases.add_child("online", delta(before)); sdcf_round::phase_end("online", query);
            sdcf_round::phase_begin("output", query); before = snapshot(); auto clear = reveal(output);
            if (options.role == 2) q.add_child("output_codes", wordArray(clear));
            if (options.diagnostic) q.add_child("diagnostic_manifest", diagnostic_shares);
            phases.add_child("output", delta(before)); sdcf_round::phase_end("output", query);
            if (nativePrepared()) sdcf_preprocess::finish_query(query);
            q.add_child("phases", phases); q.put("beaver_spool_bytes", beaver_bytes); query_results.push_back({"", q});
            query_results.back().second.put("native_public_plan_seconds", native_plan_seconds);
            if (beaver_file) { std::fclose(beaver_file); beaver_file = nullptr; }
            primitive_cursor = conversion_cursor = matrix_cursor = 0;
            save();
        }
        result.put("status", "complete"); save(); sdcf_round::finish();
    }
};
}
int main(int argc, char** argv) {
    Runtime runtime;
    try { runtime.options = parse(argc, argv); runtime.run(); return 0; }
    catch (const std::exception& e) {
        std::cerr << "CNN_RUNTIME_ERROR: " << e.what() << std::endl;
        runtime.result.put("status", "failed"); runtime.result.put("error", e.what());
        if (!runtime.options.output.empty()) { try { runtime.save(); } catch (...) {} }
        try { sdcf_round::finish(); } catch (...) {} return 1;
    }
}
