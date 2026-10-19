#include "benchmarks.h"
#include "common.h"
#include "report.h"

#include "fss/internal/correlated_ggm.h"
#include "mpc/secure_ops.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ProductInputs {
    std::vector<u8> bit_shares;
    std::vector<block> block_shares;
    std::vector<block> expected;
};

ProductInputs make_product_inputs(int count, int run) {
    ProductInputs inputs;
    inputs.bit_shares.resize(count);
    inputs.block_shares.resize(count);
    inputs.expected.resize(count);
    for (int i = 0; i < count; ++i) {
        const u8 bit = static_cast<u8>((i + run) & 1);
        const u8 server_bit = static_cast<u8>((i * 3 + run / 2) & 1);
        const block value = osuCrypto::toBlock(
            0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(i + 31 * run),
            0xd1b54a32d192ed03ULL + static_cast<uint64_t>(17 * i + run));
        const block server_share = osuCrypto::toBlock(
            0x243f6a8885a308d3ULL + static_cast<uint64_t>(i),
            0x13198a2e03707344ULL ^ static_cast<uint64_t>(run));
        inputs.bit_shares[i] =
            party == SERVER ? server_bit : static_cast<u8>(bit ^ server_bit);
        inputs.block_shares[i] =
            party == SERVER ? server_share : (value ^ server_share);
        inputs.expected[i] = bit ? value : ZeroBlock;
    }
    return inputs;
}

uint64_t target_from_bits(int count, int run) {
    uint64_t target = 0;
    for (int i = 0; i < count; ++i) {
        target = (target << 1) | static_cast<uint64_t>((i + run) & 1);
    }
    return target;
}

std::vector<uint64_t> check_points(int bits, uint64_t target) {
    const uint64_t domain = uint64_t(1) << bits;
    std::vector<uint64_t> points;
    if (bits <= 12) {
        points.reserve(static_cast<size_t>(domain));
        for (uint64_t x = 0; x < domain; ++x) {
            points.push_back(x);
        }
        return points;
    }
    points = {0, target, target ^ 1,
              target ^ (uint64_t(1) << (bits - 1)), domain - 1};
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    return points;
}

bool check_tree(const dfss::internal::CorrelatedTreeMaterial& material,
                int bits, int run) {
    DPFKeyPack key;
    key.Bin = bits;
    key.Bout = 1;
    key.groupSize = 1;
    key.prefixBits = bits;
    key.suffixBits = 0;
    key.vectorSize = 1;
    key.k = material.scw;
    key.v = material.tau;
    key.hash_key = material.hash_key;

    const uint64_t target = target_from_bits(bits, run);
    const std::vector<uint64_t> points = check_points(bits, target);
    std::vector<block> leaves(points.size());
    std::vector<u8> opened_bits(bits);
    for (size_t i = 0; i < points.size(); ++i) {
        for (int j = 0; j < bits; ++j) {
            opened_bits[j] =
                static_cast<u8>((points[i] >> (bits - 1 - j)) & 1);
        }
        leaves[i] = dfss::internal::evalCorrelatedDPFLeaf(
            opened_bits.data(), key);
    }
    reconstruct(static_cast<int32_t>(leaves.size()), leaves.data());
    bool ok = true;
    for (size_t i = 0; i < points.size(); ++i) {
        const bool is_target = points[i] == target;
        const bool leaf_lsb = (_mm_cvtsi128_si64x(leaves[i]) & 1) != 0;
        const bool leaf_zero =
            std::memcmp(&leaves[i], &ZeroBlock, sizeof(block)) == 0;
        if (leaf_lsb != is_target || (!is_target && !leaf_zero)) {
            ok = false;
        }
    }
    if (!ok) {
        std::cerr << "cw_generation leaf mismatch: party=" << party
                  << " Bin=" << bits << " target=" << target << " values=";
        for (size_t i = 0; i < points.size(); ++i) {
            std::cerr << points[i] << ':'
                      << (_mm_cvtsi128_si64x(leaves[i]) & 1) << '/'
                      << (std::memcmp(&leaves[i], &ZeroBlock,
                                      sizeof(block)) == 0)
                      << ' ';
        }
        std::cerr << '\n';
    }
    return ok;
}

template <typename Fn>
PhaseMetric measure_verified_ot_offline(Fn&& fn) {
    const uint64_t io_sent_before = peer->getIOPack()->get_comm();
    const uint64_t io_rounds_before =
        peer->getIOPack()->io->num_rounds +
        peer->getIOPack()->io_rev->num_rounds;
    const PhaseMetric measured = measure_offline_phase(std::forward<Fn>(fn));
    const uint64_t io_sent_delta =
        peer->getIOPack()->get_comm() - io_sent_before;
    const uint64_t io_rounds_delta =
        peer->getIOPack()->io->num_rounds +
        peer->getIOPack()->io_rev->num_rounds - io_rounds_before;
    if (measured.sent != io_sent_delta ||
        measured.peer_rounds != io_rounds_delta) {
        throw std::runtime_error("OT sent-byte or round counter mismatch");
    }
    return measured;
}

template <typename Fn>
PhaseMetric measure_tree_online(Fn&& fn) {
    const uint64_t ot_sent_before = peer->getIOPack()->get_comm();
    const PhaseMetric measured = measure_online_phase(std::forward<Fn>(fn));
    if (peer->getIOPack()->get_comm() != ot_sent_before ||
        measured.sent != measured.received) {
        throw std::runtime_error(
            "tree online used OT or has asymmetric peer byte counters");
    }
    return measured;
}

void consume_tree_shaped(DpfBitBlockMulMaterial& material,
                         const ProductInputs& inputs,
                         std::vector<block>& output) {
    const int count = static_cast<int>(inputs.bit_shares.size());
    if (count > 1) {
        consumeDpfBitBlockMulMaterial(
            party, material, inputs.bit_shares.data(),
            inputs.block_shares.data(), output.data(), count - 1, peer);
    }
    consumeDpfBitBlockMulMaterial(
        party, material, inputs.bit_shares.data() + count - 1,
        inputs.block_shares.data() + count - 1, output.data() + count - 1,
        1, peer);
}

bool check_products(std::vector<block>& output, const ProductInputs& inputs) {
    reconstruct(static_cast<int32_t>(output.size()), output.data());
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::memcmp(&output[i], &inputs.expected[i], sizeof(block)) != 0) {
            return false;
        }
    }
    return true;
}

void emit_material_row(const BenchConfig& config, const char* phase,
                       const PhaseMetric& metric, bool ok) {
    Row row;
    row.group = "cw_generation";
    row.protocol = "beaver_ot_triple";
    row.phase = phase;
    row.Bin = config.Bin;
    row.Bout = 128;
    row.repeat = config.repeat;
    row.evaluatedPoints = config.Bin;
    row.metric = metric;
    row.status = ok ? "ok" : "correctness_failed";
    row.notes =
        "scope=bit_block_product;products_per_repeat=Bin;"
        "consume_batches=Bin-1+1;backend_warmup_excluded=1;" +
        correctness_note(config, "products_checked");
    emit_row(row);
}

void emit_cot_material_row(const BenchConfig& config, const char* phase,
                           const PhaseMetric& metric, bool ok) {
    Row row;
    row.group = "cw_generation";
    row.protocol = "figure10_cot_batch";
    row.phase = phase;
    row.Bin = config.Bin;
    row.Bout = 128;
    row.repeat = config.repeat;
    row.evaluatedPoints = config.Bin;
    row.metric = metric;
    row.status = ok ? "ok" : "correctness_failed";
    row.notes =
        "scope=batch_cot;correlations_per_repeat=Bin;"
        "one_send_and_one_receive_batch;session_setup_and_warmup_excluded=1;" +
        correctness_note(config, "cot_material_checked");
    emit_row(row);
}

void emit_tree_row(const BenchConfig& config, const char* protocol,
                   const char* phase, const PhaseMetric& metric, bool ok) {
    Row row;
    row.group = "cw_generation";
    row.protocol = protocol;
    row.phase = phase;
    row.Bin = config.Bin;
    row.Bout = 1;
    row.repeat = config.repeat;
    row.evaluatedPoints = 1;
    row.metric = metric;
    row.status = ok ? "ok" : "correctness_failed";
    row.notes =
        std::string("scope=full_tree;correlations_per_repeat=Bin;") +
        (config.Bin <= 12 ? "full_domain_check;" : "sampled_point_check;") +
        "session_setup_and_warmup_excluded=1;" +
        correctness_note(config, "leaf_block_checked");
    emit_row(row);
}

void run_beaver_material_bench(const BenchConfig& config) {
    const int count = config.Bin;

    // Complete one unmeasured call so backend setup and its base OTs do not
    // enter either measured phase.
    peer->reset_ot_precompute();
    {
        ProductInputs warm_inputs = make_product_inputs(count, -1);
        DpfBitBlockMulMaterial warm =
            prepareDpfBitBlockMulMaterial(party, count, peer);
        std::vector<block> warm_output(count);
        consume_tree_shaped(warm, warm_inputs, warm_output);
        if (config.checkCorrectness &&
            !check_products(warm_output, warm_inputs)) {
            throw std::runtime_error("Beaver warmup product check failed");
        }
    }

    PhaseMetric offline;
    PhaseMetric online;
    bool ok = true;
    for (int run = 0; run < config.repeat; ++run) {
        ProductInputs inputs = make_product_inputs(count, run);
        DpfBitBlockMulMaterial material;
        const PhaseMetric offline_run = measure_verified_ot_offline([&] {
            material = prepareDpfBitBlockMulMaterial(party, count, peer);
        });
        add_metric(offline, offline_run);

        if (phase_runs_online(config)) {
            std::vector<block> output(count);
            const PhaseMetric online_run = measure_online_phase([&] {
                consume_tree_shaped(material, inputs, output);
            });
            const int batches = count > 1 ? 2 : 1;
            const uint64_t bytes_per_party =
                static_cast<uint64_t>(count) * (sizeof(u8) + sizeof(block));
            if (online_run.sent != bytes_per_party ||
                online_run.received != bytes_per_party ||
                online_run.peer_rounds != static_cast<uint64_t>(4 * batches) ||
                online_run.reconstruct_rounds != 2 * batches) {
                throw std::runtime_error(
                    "Beaver consume sent-byte or round counter mismatch");
            }
            add_metric(online, online_run);
            if (config.checkCorrectness) {
                ok = check_products(output, inputs) && ok;
            }
        }
    }

    emit_material_row(config, "offline", offline, ok);
    if (phase_runs_online(config)) {
        emit_material_row(config, "online", online, ok);
    }
}

void run_beaver_tree_bench(const BenchConfig& config) {
    const int bits = config.Bin;
    peer->reset_ot_precompute();
    {
        ProductInputs warm_inputs = make_product_inputs(bits, -1);
        DpfBitBlockMulMaterial prepared =
            prepareDpfBitBlockMulMaterial(party, bits, peer);
        auto warm = dfss::internal::generateCorrelatedTreeBeaverPrepared(
            party, bits, 1, warm_inputs.bit_shares.data(),
            "cw_generation Beaver warmup", false, prepared);
        if (config.checkCorrectness && !check_tree(warm, bits, -1)) {
            throw std::runtime_error("Beaver tree warmup check failed");
        }
    }

    PhaseMetric offline;
    PhaseMetric online;
    bool ok = true;
    for (int run = 0; run < config.repeat; ++run) {
        ProductInputs inputs = make_product_inputs(bits, run);
        DpfBitBlockMulMaterial prepared;
        add_metric(offline, measure_verified_ot_offline([&] {
            prepared = prepareDpfBitBlockMulMaterial(party, bits, peer);
        }));
        if (phase_runs_online(config)) {
            dfss::internal::CorrelatedTreeMaterial tree;
            add_metric(online, measure_tree_online([&] {
                tree = dfss::internal::generateCorrelatedTreeBeaverPrepared(
                    party, bits, 1, inputs.bit_shares.data(),
                    "cw_generation Beaver", false, prepared);
            }));
            if (config.checkCorrectness) {
                ok = check_tree(tree, bits, run) && ok;
            }
        }
    }
    emit_tree_row(config, "beaver_prepared_tree", "offline", offline, ok);
    if (phase_runs_online(config)) {
        emit_tree_row(config, "beaver_prepared_tree", "online", online, ok);
    }
}

void run_figure10_tree_bench(const BenchConfig& config) {
    const int bits = config.Bin;
    auto session = dfss::internal::initializeFigure10CotSession(party, peer);
    peer->reset_ot_precompute();
    {
        ProductInputs warm_inputs = make_product_inputs(bits, -1);
        auto prepared = dfss::internal::prepareFigure10CotMaterial(
            party, bits, session, peer);
        auto warm = dfss::internal::generateCorrelatedTreeFigure10(
            party, bits, 1, warm_inputs.bit_shares.data(),
            "cw_generation Figure 10 warmup", false, session, prepared);
        if (config.checkCorrectness && !check_tree(warm, bits, -1)) {
            throw std::runtime_error("Figure 10 tree warmup check failed");
        }
    }

    PhaseMetric offline;
    PhaseMetric online;
    bool ok = true;
    for (int run = 0; run < config.repeat; ++run) {
        ProductInputs inputs = make_product_inputs(bits, run);
        dfss::internal::Figure10CotMaterial prepared;
        add_metric(offline, measure_verified_ot_offline([&] {
            prepared = dfss::internal::prepareFigure10CotMaterial(
                party, bits, session, peer);
        }));
        if (phase_runs_online(config)) {
            dfss::internal::CorrelatedTreeMaterial tree;
            add_metric(online, measure_tree_online([&] {
                tree = dfss::internal::generateCorrelatedTreeFigure10(
                    party, bits, 1, inputs.bit_shares.data(),
                    "cw_generation Figure 10", false, session, prepared);
            }));
            if (config.checkCorrectness) {
                ok = check_tree(tree, bits, run) && ok;
            }
        }
    }
    emit_tree_row(config, "figure10_parallel_cot_tree", "offline", offline,
                  ok);
    // This row exposes the batch COT primitive directly, so it can be
    // compared with beaver_ot_triple without including tree expansion.
    emit_cot_material_row(config, "offline", offline, ok);
    if (phase_runs_online(config)) {
        emit_tree_row(config, "figure10_parallel_cot_tree", "online", online,
                      ok);
    }
}

}  // namespace

void run_cw_generation_bench(const BenchConfig& config) {
    if (config.Bin < 1 || config.Bin > 24) {
        throw std::invalid_argument("cw_generation requires 1 <= bin <= 24");
    }
    if (config.protocol == "beaver") {
        run_beaver_material_bench(config);
        run_beaver_tree_bench(config);
        return;
    }
    if (config.protocol == "figure10") {
        run_figure10_tree_bench(config);
        return;
    }
    throw std::invalid_argument(
        "cw_generation requires protocol=beaver or protocol=figure10");
}
