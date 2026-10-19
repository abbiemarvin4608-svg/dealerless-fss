#include "cases.h"
#include "activation.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

namespace {

constexpr int kBits = 32;
constexpr int kFracBits = segfss::kSigmaDefaultFracBits;
constexpr int kQ = segfss::kSigmaDefaultQ;
constexpr int64_t kScale = int64_t(1) << kFracBits;
constexpr int64_t kSigmoidBound = 16 * kScale;
constexpr int64_t kTanhBound = 8 * kScale;
constexpr int64_t kIndexStep = int64_t(1) << (kFracBits - kQ + 4);

struct CommunicationCounters {
    uint64_t rounds;
    uint64_t sent;
    uint64_t received;
};

CommunicationCounters counters() {
    return {peer->rounds, peer->bytesSent, peer->bytesReceived};
}

void check_no_communication(ResultLog& log, const std::string& name,
                            CommunicationCounters before) {
    const auto after = counters();
    log.check_local(name + " no communication",
                    after.rounds == before.rounds &&
                        after.sent == before.sent &&
                        after.received == before.received,
                    1);
}

uint64_t check_batch(ResultLog& log, bool is_tanh,
                     const std::vector<int64_t>& values,
                     std::mt19937_64& generator) {
    const std::string name = std::string(is_tanh ? "tanh" : "sigmoid") +
                             " batch=" + std::to_string(values.size());
    auto keys = is_tanh
        ? segfss::tanhBatchOffline(party, static_cast<int>(values.size()),
                                   kBits, kFracBits, kQ, peer)
        : segfss::sigmaBatchOffline(party, static_cast<int>(values.size()),
                                    kBits, kFracBits, kQ, peer);
    const char* dpf_env = std::getenv("SEGFSS_DPF_LUT");
    const bool use_dpf = dpf_env != nullptr && dpf_env[0] == '1';
    bool materialized = true;
    for (const auto& key : keys) {
        materialized = materialized && !key.use_piecewise_compare &&
            (use_dpf
                 ? (key.use_dpf_lut &&
                    key.dpf_lut.selection_share.size() == (std::size_t(1) << kQ) &&
                    !key.dpf_lut.consumed && !key.dpf_lut.released)
                 : (!key.use_dpf_lut &&
                    key.segmented_lut.output_share_table.size() ==
                        (std::size_t(1) << kQ) &&
                    !key.segmented_lut.consumed));
    }
    log.check_local(name + " 1024 offline output shares", materialized, 1);

    std::vector<GroupElement> inputs;
    std::vector<uint64_t> expected;
    for (int64_t encoded : values) {
        const uint64_t value = twos_from_signed(encoded, kBits);
        uint64_t server_share;
        do {
            server_share = reduce_to_bits(generator(), kBits);
        } while (server_share == 0 || server_share == value);
        inputs.push_back(split_share(value, kBits, server_share));
        const auto plain = is_tanh
            ? segfss::tanhPlain(GroupElement(value, kBits), kFracBits, kQ)
            : segfss::sigmaPlain(GroupElement(value, kBits), kFracBits, kQ);
        expected.push_back(plain.value);
    }

    const auto online = [is_tanh](int role, const std::vector<GroupElement>& input,
                                 std::vector<segfss::SigmaActivationKey>& material,
                                 Peer* channel) {
        return is_tanh ? segfss::tanhBatchOnline(role, input, material, channel)
                       : segfss::sigmaBatchOnline(role, input, material, channel);
    };
    const auto before = counters();
    auto outputs = online(party, inputs, keys, peer);
    const uint64_t online_rounds = peer->rounds - before.rounds;
    log.check_local(name + " communicates", online_rounds > 0, 1);

    // Record the protocol counters before correctness reconstructs its output.
    log.check_vector(name + " exact fixed-point boundaries", outputs.data(),
                     expected, kBits);
    bool consumed = true;
    for (const auto& key : keys) {
        consumed = consumed &&
            (use_dpf ? key.dpf_lut.consumed : key.segmented_lut.consumed);
    }
    log.check_local(name + " consumes every key", consumed, 1);

    auto reuse_before = counters();
    bool rejected_reuse = false;
    try {
        (void)online(party, inputs, keys, peer);
    } catch (const std::logic_error&) {
        rejected_reuse = true;
    }
    log.check_local(name + " rejects key reuse", rejected_reuse, 1);
    check_no_communication(log, name + " reuse", reuse_before);

    if (keys.size() > 1) {
        // A fresh first key must not be consumed before a used later key is
        // rejected. This catches validation inside a per-element online loop.
        segfss::freeSigmaActivationKey(keys.front());
        keys.front() = is_tanh
            ? segfss::tanhOffline(party, kBits, kFracBits, kQ, peer)
            : segfss::sigmaOffline(party, kBits, kFracBits, kQ, peer);
        reuse_before = counters();
        rejected_reuse = false;
        try {
            (void)online(party, inputs, keys, peer);
        } catch (const std::logic_error&) {
            rejected_reuse = true;
        }
        log.check_local(name + " rejects a later used key", rejected_reuse, 1);
        check_no_communication(log, name + " later-key rejection", reuse_before);
        const char* dpf_env = std::getenv("SEGFSS_DPF_LUT");
        const bool use_dpf = dpf_env != nullptr && dpf_env[0] == '1';
        log.check_local(name + " poisons fresh first key after batch rejection",
                        use_dpf ? keys.front().dpf_lut.consumed
                                : keys.front().segmented_lut.consumed,
                        1);
    }
    for (auto& key : keys) {
        segfss::freeSigmaActivationKey(key);
    }
    return online_rounds;
}

void check_failure_consumes_material(ResultLog& log) {
    auto keys = segfss::sigmaBatchOffline(party, 2, kBits, kFracBits,
                                         kQ, peer);
    std::vector<GroupElement> inputs{
        split_share(twos_from_signed(-kScale, kBits), kBits, 0x12345678),
        split_share(twos_from_signed(kScale, kBits), kBits, 0x87654321),
    };
    // Preflight rejects the malformed later BitXA material before any network
    // operation.  Entering the online API still invalidates the whole batch,
    // so neither a fresh prefix nor the failed suffix can be retried.
    keys.back().relu_mul.bitwidth = kBits - 1;
    const auto before = counters();
    bool rejected = false;
    try {
        (void)segfss::sigmaBatchOnline(party, inputs, keys, peer);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    const auto after = counters();
    log.check_local("failed activation rejects before communication",
                    rejected && after.rounds == before.rounds &&
                        after.sent == before.sent &&
                        after.received == before.received, 1);
    const char* dpf_env = std::getenv("SEGFSS_DPF_LUT");
    const bool use_dpf = dpf_env != nullptr && dpf_env[0] == '1';
    // The batch-level guard invalidates both LUT families, including DPF
    // material that would otherwise be consumed only at masked-index open.
    const bool consumed_after_failure = use_dpf
        ? (keys.front().dpf_lut.consumed && keys.back().dpf_lut.consumed)
        : (keys.front().segmented_lut.consumed &&
           keys.back().segmented_lut.consumed);
    log.check_local("failed activation material state", consumed_after_failure, 1);

    rejected = false;
    const auto retry_before = counters();
    try {
        (void)segfss::sigmaBatchOnline(party, inputs, keys, peer);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    log.check_local("failed activation rejects retry", rejected, 1);
    check_no_communication(log, "failed activation retry", retry_before);
    for (auto& key : keys) {
        segfss::freeSigmaActivationKey(key);
    }
}

void check_dpf_batch_failure_consumes_material(ResultLog& log) {
    // Exercise the public DPF API directly: malformed suffix metadata must
    // poison the complete resident batch before any masked index is opened.
    constexpr int q = 3;
    PublicLUTData table;
    table.Bin = q;
    table.Bout = kBits;
    for (uint64_t index = 0; index < (uint64_t(1) << q); ++index) {
        table.values.emplace_back(3 * index + 1, kBits);
    }
    auto materials = segfss::dpfPublicLutBatchOffline(
        party, table, 2, 100, peer);
    std::vector<GroupElement> inputs{
        split_share(1, q, 2),
        split_share(3, q, 5),
    };
    materials.back().request_id += 1;
    const auto before = counters();
    bool rejected = false;
    try {
        (void)segfss::dpfPublicLutBatchOnline(
            party, inputs, materials, peer, 100, table);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    const auto after = counters();
    log.check_local("DPF malformed batch rejects before communication",
                    rejected && after.rounds == before.rounds &&
                        after.sent == before.sent &&
                        after.received == before.received,
                    1);
    const bool consumed = materials.size() == 2 &&
        materials[0].consumed && materials[1].consumed;
    log.check_local("DPF malformed batch consumes every material", consumed, 1);

    rejected = false;
    const auto retry_before = counters();
    try {
        (void)segfss::dpfPublicLutBatchOnline(
            party, inputs, materials, peer, 100, table);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    log.check_local("DPF malformed batch rejects retry", rejected, 1);
    check_no_communication(log, "DPF malformed batch retry", retry_before);
    for (auto& material : materials) {
        segfss::releaseDpfPublicLutMaterial(material);
    }
}

}  // namespace

void check_activation_batch(ResultLog& log) {
    std::mt19937_64 generator(0xBA7C4A17ULL);
    const uint64_t sigmoid_single_rounds = check_batch(log, false, {0}, generator);
    const uint64_t sigmoid_batch_rounds = check_batch(
        log, false,
        {-2 * kSigmoidBound, -kSigmoidBound - 1, -kSigmoidBound,
         -kSigmoidBound + 1, -kScale,
         -kIndexStep - 1, -kIndexStep, -kIndexStep + 1, -1, 0, 1,
         kIndexStep - 1, kIndexStep, kIndexStep + 1, kScale,
         kSigmoidBound - 1, kSigmoidBound, kSigmoidBound + 1,
         2 * kSigmoidBound},
        generator);
    log.check_local("sigmoid online peer rounds independent of batch size",
                    sigmoid_batch_rounds, sigmoid_single_rounds);

    const uint64_t tanh_single_rounds = check_batch(log, true, {0}, generator);
    const uint64_t tanh_batch_rounds = check_batch(
        log, true,
        {-2 * kTanhBound, -kTanhBound - 1, -kTanhBound,
         -kTanhBound + 1, -kScale,
         -kIndexStep / 2 - 1, -kIndexStep / 2, -kIndexStep / 2 + 1,
         -1, 0, 1,
         kIndexStep / 2 - 1, kIndexStep / 2, kIndexStep / 2 + 1, kScale,
         kTanhBound - 1, kTanhBound, kTanhBound + 1, 2 * kTanhBound},
        generator);
    log.check_local("tanh online peer rounds independent of batch size",
                    tanh_batch_rounds, tanh_single_rounds);

    check_failure_consumes_material(log);
    check_dpf_batch_failure_consumes_material(log);

    bool rejected_piecewise_batch = false;
    const auto disabled_before = counters();
    try {
        (void)segfss::batchPiecewiseCompareOnline(party, {}, {}, peer);
    } catch (const std::logic_error&) {
        rejected_piecewise_batch = true;
    }
    log.check_local("unmasked batch piecewise comparison disabled", rejected_piecewise_batch, 1);
    check_no_communication(log, "disabled piecewise batch", disabled_before);

    std::vector<segfss::SigmaActivationKey> legacy_keys(1);
    legacy_keys[0].bits = kBits;
    legacy_keys[0].use_piecewise_compare = true;
    bool rejected_legacy_batch = false;
    const auto legacy_before = counters();
    try {
        (void)segfss::sigmaBatchOnline(party, {GroupElement(0, kBits)},
                                     legacy_keys, peer);
    } catch (const std::logic_error&) {
        rejected_legacy_batch = true;
    }
    log.check_local("batch SIGMA rejects unmasked legacy mode", rejected_legacy_batch, 1);
    check_no_communication(log, "disabled legacy batch", legacy_before);
}
