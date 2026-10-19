#include "activation.h"
#include "folded_native.h"
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int party_instance = 0, party = 0, num_threads = 1, port = 32217;
int32_t bitlength = 32;
std::string address = "127.0.0.1";
int num_argmax = 1000, length = 1;
uint8_t choice_bit = 0;
bool verbose = false;
Peer *client = nullptr, *server = nullptr, *peer = nullptr;
Dealer* dealer = nullptr;

namespace {
void checkAutomaticComparisonContracts() {
    segfss::ActivationContext context(party, peer, 87);
    const GroupElement narrow_left(0x1234, 16);
    const std::vector<GroupElement> narrow{
        party == SERVER ? narrow_left : GroupElement(0, 16) - narrow_left};
    segfss::prepareSigmaActivation(context, 1, 16, 8, 6);
    auto output = segfss::sigmaBatchOnline(context, narrow);
    peer->open_additive(output.data(), 1, 16);
    if (output.front().value != 128 || context.clip_bits != 0)
        throw std::runtime_error("generic activation changed the automatic clip-width contract");
    const GroupElement wide_left(0xD15EA5E0ULL, 32);
    const std::vector<GroupElement> wide{
        party == SERVER ? wide_left : GroupElement(0, 32) - wide_left};
    auto prepare = [&] {
        segfss::DpfLutOptions options;
        segfss::DpfLutBinding binding;
        binding.context_id = context.identity;
        binding.query_id = context.generation_epoch;
        binding.table_version = 1;
        segfss::prepareSigmaActivation(context, 1, 32, 12, 10, options, binding);
        if (context.clip_bits != 0 || context.clip_segments != 0 ||
            context.materials.front().clip_bits != 32)
            throw std::runtime_error("paper preparation persisted automatic comparison settings");
    };
    // The same context can switch from a generic 16-bit activation to the
    // paper ring32 API without resetting any comparison options.
    prepare();
    output = segfss::sigmaBatchOnline(context, wide);
    peer->open_additive(output.data(), 1, 32);
    if (output.front().value != 2048)
        throw std::runtime_error("generic-to-paper activation changed zero's integer output");
    context.sign_backend = segfss::ActivationSignBackend::segmented_fss;
    for (int segments : {4, 2}) {
        context.sign_segments = segments;
        prepare();
        if (context.sign_material.material.nseg != segments || context.clip_material.material.nseg != segments)
            throw std::runtime_error("automatic clip segments did not follow the fresh query's sign segments");
        output = segfss::sigmaBatchOnline(context, wide);
        peer->open_additive(output.data(), 1, 32);
        if (output.front().value != 2048)
            throw std::runtime_error("changing automatic comparison segments changed integer output");
    }
}

void checkNativeClipReduction() {
    constexpr uint64_t clip_mask = (uint64_t(1) << 22) - 1;
    for (int64_t x = -32768; x <= 32767; ++x) {
        for (int scale : {32, 64}) {
            const int64_t u = x * scale;
            const int64_t z = (u < 0 ? -u : u) - 65536;
            const uint64_t left = GroupElement(0xE3141234ULL + (x + 32768) * 7919, 32).value;
            const uint64_t right = GroupElement(static_cast<uint64_t>(z) - left, 32).value;
            const uint64_t reduced = ((left & clip_mask) + (right & clip_mask)) & clip_mask;
            if (bool(reduced >> 21) != (z < 0))
                throw std::runtime_error("signed22 clip predicate differs on the native signed16 domain");
        }
    }
}

void checkNativeSignClip() {
    checkNativeClipReduction();
    const std::vector<int64_t> clear{-32768, -2049, -2048, -2047, -1025, -1024,
        -1023, -2, -1, 0, 1, 2, 1023, 1024, 1025, 2047, 2048, 2049, 32767};
    // Literal native outputs from the independent Decimal f12/q10 mapping.
    // Both clip boundaries, the ring32 abs of -32768*64, zero, and an odd
    // batch tail are covered without using the secure table builder as truth.
    const std::vector<int64_t> sigmoid{0, 0, 0, 0, 4, 4, 4, 8128, 8192,
        8192, 8192, 8256, 16380, 16380, 16380, 16384, 16384, 16384, 16384};
    const std::vector<int64_t> tanh{-128, -128, -128, -128, -128, -128, -128,
        -2, -1, 0, 1, 2, 128, 128, 128, 128, 128, 128, 128};
    std::vector<uint64_t> native_shares;
    for (std::size_t i = 0; i < clear.size(); ++i) {
        const GroupElement value(static_cast<uint64_t>(clear[i]), 16);
        const GroupElement left(0xD15E + 7919 * i, 16);
        native_shares.push_back((party == SERVER ? left : value - left).value);
    }
    segfss::ActivationContext context(party, peer, 81);
    segfss::SciTruncationWorkspace workspace;
    for (bool private_sign : {false, true}) for (bool segmented_lut : {false, true})
    for (bool reuse : {false, true}) for (int clip_bits : {32, 22}) {
        context.sign_backend = private_sign ? segfss::ActivationSignBackend::segmented_fss
                                            : segfss::ActivationSignBackend::bitdec;
        context.sign_segments = private_sign ? 4 : 0;
        context.clip_segments = private_sign ? 4 : 0;
        context.native_signed16_input = true;
        context.reuse_input_sign = reuse;
        context.clip_bits = clip_bits;
        segfss::DpfLutOptions options;
        options.tile_size = 5;
        if (segmented_lut) options.family = segfss::DpfLutKeyFamily::segmented_boolean_tensor_b2a;
        auto prepare = [&] {
            segfss::DpfLutBinding binding;
            binding.context_id = context.identity;
            binding.query_id = context.generation_epoch;
            binding.table_version = 1;
            binding.family = options.family;
            segfss::prepareSigmaActivation(context, clear.size(), 32, 12, 10, options, binding);
        };
        auto signs = [&] {
            segfss::ActivationInputSignShares result;
            result.owner_peer = peer;
            result.binding = context.lut_binding;
            result.generation_epoch = context.generation_epoch;
            return result;
        };
        for (bool is_tanh : {false, true}) {
            auto extension = segfss::folded::prepareInputExtension(party, clear.size(), peer,
                segfss::SegmentPolicy::fixed_4, nullptr, nullptr);
            auto output_scale = is_tanh ? segfss::folded::prepareTanhOutput(party,
                clear.size(), peer, segfss::SegmentPolicy::fixed_4, nullptr, nullptr)
                : segfss::SciTruncationMaterial{};
            prepare();
            for (const auto& key : context.materials) {
                const std::size_t sign_count = private_sign || reuse ? 0 : 62;
                const std::size_t clip_count = private_sign ? 0 : 2 * (clip_bits - 1);
                if (key.sign_and.size() != sign_count || key.clip_and.size() != clip_count)
                    throw std::runtime_error("native optimization prepared incorrect Boolean inventory");
            }
            if (private_sign && (context.sign_material.material.count != (reuse ? 0 : int(clear.size())) ||
                context.clip_material.material.width != clip_bits))
                throw std::runtime_error("native optimization prepared incorrect private MSB inventory");
            auto input_signs = signs();
            auto q12 = segfss::folded::nativeInputToQ12(party, native_shares.data(),
                clear.size(), extension, peer, &workspace, nullptr, reuse ? &input_signs.shares : nullptr);
            if (is_tanh) q12 = segfss::folded::tanhCoreInput(q12);
            auto result = segfss::sigmaBatchOnline(context, q12, nullptr, reuse ? &input_signs : nullptr);
            if (is_tanh) result = segfss::folded::tanhNativeOutput(party, result, output_scale,
                peer, &workspace, nullptr);
            else for (auto& value : result) value = GroupElement(value.value * 4, 16);
            peer->open_additive(result.data(), result.size(), 16);
            const auto& expected = is_tanh ? tanh : sigmoid;
            for (std::size_t i = 0; i < result.size(); ++i)
                if (result[i].value != GroupElement(static_cast<uint64_t>(expected[i]), 16).value)
                    throw std::runtime_error("native sign/clip optimization differs from literal integer truth");
            for (const auto& key : context.materials)
                if (key.sign_and.cursor != key.sign_and.size() || key.clip_and.cursor != key.clip_and.size())
                    throw std::runtime_error("native sign/clip optimization left unused Boolean inventory");
            if (reuse && !input_signs.consumed)
                throw std::runtime_error("native input signs were not consumed exactly once");
            if (private_sign && (!context.clip_material.material.consumed ||
                (!reuse && !context.sign_material.material.consumed)))
                throw std::runtime_error("native sign/clip optimization left unused private MSB inventory");
        }
        if (reuse) {
            // Stale query/epoch, foreign owner, and missing signs must fail
            // before any I/O. A rejected call consumes the prepared query.
            for (int mutation = 0; mutation < 4; ++mutation) {
                prepare();
                auto input_signs = signs();
                input_signs.shares.resize(clear.size());
                if (mutation == 0) ++input_signs.binding.query_id;
                if (mutation == 1) ++input_signs.generation_epoch;
                if (mutation == 2) input_signs.owner_peer = nullptr;
                const auto sent = peer->bytesSent, received = peer->bytesReceived;
                bool rejected = false;
                try { (void)segfss::sigmaBatchOnline(context,
                    std::vector<GroupElement>(clear.size(), GroupElement(0, 32)), nullptr,
                    mutation == 3 ? nullptr : &input_signs); }
                catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected || context.prepared || sent != peer->bytesSent || received != peer->bytesReceived)
                    throw std::runtime_error("native sign binding mismatch communicated or retained the query");
            }
        }
    }
    // Narrowing is an explicit native-input contract, never inferred from
    // generic ring32 shares. Reject the unsupported 21-bit TanH predicate.
    for (int invalid_clip : {21, 22}) {
        context.sign_backend = segfss::ActivationSignBackend::bitdec;
        context.sign_segments = context.clip_segments = 0;
        context.native_signed16_input = false;
        context.reuse_input_sign = false;
        context.clip_bits = invalid_clip;
        segfss::DpfLutOptions options;
        segfss::DpfLutBinding binding;
        binding.context_id = context.identity;
        binding.table_version = 1;
        const auto sent = peer->bytesSent, received = peer->bytesReceived;
        bool rejected = false;
        try { segfss::prepareSigmaActivation(context, clear.size(), 32, 12, 10, options, binding); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || sent != peer->bytesSent || received != peer->bytesReceived)
            throw std::runtime_error("generic or unsupported clip narrowing was accepted");
    }
    std::cout << "NATIVE_SIGN_CLIP boundaries=19 native_clip_domain=65536 sigmoid,tanh "
              << "reuse=0,1 clip=32,22 sign=bitdec,segmented lut=full,segmented status=passed\n";
}

void checkContexts(bool paper = false, bool segmented = false, bool private_sign = false) {
    constexpr int bits = 32;
    const std::vector<int64_t> encoded{-65537, -65536, -65535, -4096,
                                     -64, -1, 0, 1, 64, 4096, 65536};
    // Known integer values of the paper's fixed public table, including an
    // odd batch tail and both sides of the half-open clipping boundary.
    const std::vector<int64_t> sigmoid_expected{0, 0, 0, 1101, 2032,
                                               2048, 2048, 2048, 2064, 2995, 4096};
    const std::vector<int64_t> tanh_expected{-4096, -4096, -4096, -3120, -64,
                                            0, 0, 0, 64, 3120, 4096};
    setenv("SEGFSS_DPF_LUT", "1", 1);
    std::vector<GroupElement> shares;
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        const GroupElement clear(static_cast<uint64_t>(encoded[i]), bits);
        const GroupElement left(0xD15EA5E0ULL + 7919 * i, bits);
        shares.push_back(party == SERVER ? left : clear - left);
    }
    auto check = [&](std::vector<GroupElement> output, Peer* channel,
                     const std::vector<int64_t>& expected) {
        channel->open_additive(output.data(), static_cast<int>(output.size()), bits);
        for (std::size_t i = 0; i < output.size(); ++i)
            if (output[i].value != GroupElement(static_cast<uint64_t>(expected[i]), bits).value)
                throw std::runtime_error("complete activation differs from literal oracle");
    };
    for (bool tanh : {false, true}) {
        auto keys = segfss::sigmaBatchOffline(party, shares.size(), bits, 12, 10, peer);
        check(tanh ? segfss::tanhBatchOnline(party, shares, keys, peer)
                   : segfss::sigmaBatchOnline(party, shares, keys, peer),
              peer, tanh ? tanh_expected : sigmoid_expected);
        for (auto& key : keys) segfss::freeSigmaActivationKey(key);
    }
    std::unique_ptr<Peer> first(party == SERVER ? waitForPeer(port + 20, party)
                                               : new Peer(address, port + 20, party));
    std::unique_ptr<Peer> second(party == SERVER ? waitForPeer(port + 40, party)
                                                : new Peer(address, port + 40, party));
    segfss::ActivationContext one(party, first.get(), 1);
    segfss::ActivationContext two(party, second.get(), 2);
    auto prepare = [&](segfss::ActivationContext& context) {
        if (!paper) {
            segfss::prepareSigmaActivation(context, shares.size(), bits, 12, 10);
            return;
        }
        segfss::DpfLutOptions options;
        context.sign_backend = private_sign ? segfss::ActivationSignBackend::segmented_fss
                                            : segfss::ActivationSignBackend::bitdec;
        context.sign_segments = private_sign ? 4 : 0;
        options.tile_size = 5;
        if (segmented) options.family = segfss::DpfLutKeyFamily::segmented_boolean_tensor_b2a;
        segfss::DpfLutBinding binding;
        binding.context_id = context.identity;
        binding.query_id = context.generation_epoch;
        binding.operator_id = context.identity - 1;
        binding.worker_id = context.identity;
        binding.table_version = 1;
        binding.family = options.family;
        const auto* previous_table = context.public_table.get();
        segfss::prepareSigmaActivation(context, shares.size(), bits, 12, 10,
                                       options, binding, &context.lut_metrics);
        if (!context.public_table ||
            (previous_table && context.public_table.get() != previous_table))
            throw std::runtime_error("paper activation did not retain its immutable public table");
    };
    const auto default_sent = peer->bytesSent, default_received = peer->bytesReceived;
    const auto default_rounds = numRounds;
    for (int query = 0; query < 2; ++query) {
        prepare(one);
        prepare(two);
        const auto products_one = first->bit_block_ledger.products;
        const auto products_two = second->bit_block_ledger.products;
        const auto generated_one = first->preprocessing_ledger;
        const auto generated_two = second->preprocessing_ledger;
        auto run_one = std::async(std::launch::async, [&] {
            check(segfss::sigmaBatchOnline(one, shares), first.get(), sigmoid_expected);
        });
        auto run_two = std::async(std::launch::async, [&] {
            check(segfss::tanhBatchOnline(two, shares), second.get(), tanh_expected);
        });
        run_one.get(); run_two.get();
        if (first->bit_block_ledger.products != products_one ||
            second->bit_block_ledger.products != products_two ||
            first->preprocessing_ledger.boolean_triples != generated_one.boolean_triples ||
            second->preprocessing_ledger.boolean_triples != generated_two.boolean_triples ||
            first->preprocessing_ledger.arithmetic_triples != generated_one.arithmetic_triples ||
            second->preprocessing_ledger.arithmetic_triples != generated_two.arithmetic_triples ||
            first->preprocessing_ledger.b2a_values != generated_one.b2a_values ||
            second->preprocessing_ledger.b2a_values != generated_two.b2a_values ||
            one.statistics.consumed != shares.size() * (query + 1) ||
            two.statistics.consumed != shares.size() * (query + 1))
            throw std::runtime_error("online generated material or context consumption differs");
        for (const auto* context : {&one, &two})
            for (const auto& key : context->materials)
                if (key.sign_and.cursor != (private_sign ? 0 : 62) ||
                    key.clip_and.cursor != (private_sign ? 0 : 62) ||
                    (private_sign && (!key.sign_and.a.empty() || !key.clip_and.a.empty())) ||
                    !key.relu_mul.consumed || !key.clip_select_mul.consumed ||
                    !key.branch_select_mul.consumed || !key.trunc_key.consumed ||
                    !key.dpf_lut.consumed)
                    throw std::runtime_error("complete activation did not consume resident material");
        if (private_sign)
            for (const auto* context : {&one, &two})
                if (!context->sign_material.material.consumed || !context->clip_material.material.consumed ||
                    context->sign_material.material.scan.cursor != 6 * shares.size() ||
                    context->clip_material.material.scan.cursor != 6 * shares.size())
                    throw std::runtime_error("complete activation did not consume independent MSB stocks");
        const auto sent = first->bytesSent, received = first->bytesReceived;
        bool rejected = false;
        try { (void)segfss::sigmaBatchOnline(one, shares); }
        catch (const std::logic_error&) { rejected = true; }
        if (!rejected || sent != first->bytesSent || received != first->bytesReceived)
            throw std::runtime_error("consumed context material must reject before I/O");
    }
    prepare(one);
    const auto sent = first->bytesSent, received = first->bytesReceived;
    bool rejected = false;
    try { prepare(one); }
    catch (const std::logic_error&) { rejected = true; }
    if (!rejected || sent != first->bytesSent || received != first->bytesReceived)
        throw std::runtime_error("preparation replaced an unconsumed inventory");
    auto wrong_shape = shares;
    wrong_shape.back() = GroupElement(0, 16);
    rejected = false;
    try { (void)segfss::sigmaBatchOnline(one, wrong_shape); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || sent != first->bytesSent || received != first->bytesReceived ||
        one.prepared || one.statistics.failed_batches != 1)
        throw std::runtime_error("shape failure must consume inventory before I/O");
    if (paper) {
        prepare(one);
        ++one.lut_binding.query_id;
        const auto binding_sent = first->bytesSent, binding_received = first->bytesReceived;
        rejected = false;
        try { (void)segfss::sigmaBatchOnline(one, shares); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || binding_sent != first->bytesSent ||
            binding_received != first->bytesReceived || one.prepared)
            throw std::runtime_error("paper binding failure must consume inventory before I/O");
        for (const auto& key : one.materials)
            if (!key.dpf_lut.consumed)
                throw std::runtime_error("paper binding failure left usable LUT inventory");
    }
    if (private_sign) {
        for (int mutation = 0; mutation < 7; ++mutation) {
            prepare(one);
            auto& stock = one.clip_material;
            switch (mutation) {
            case 0: stock.material.keys.back().depth = 0; break;
            case 1: stock.material.owner_peer = second.get(); break;
            case 2: ++stock.binding.query_id; break;
            case 3: ++stock.generation_epoch; break;
            case 4: std::swap(one.sign_material, one.clip_material); break;
            case 5: stock.material.keys.pop_back(); break;
            case 6: stock.material.scan.a.pop_back(); break;
            }
            const auto before_sent = first->bytesSent, before_received = first->bytesReceived;
            rejected = false;
            try { (void)segfss::sigmaBatchOnline(one, shares); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected || before_sent != first->bytesSent || before_received != first->bytesReceived ||
                one.prepared || !one.sign_material.material.consumed || !one.clip_material.material.consumed)
                throw std::runtime_error("private MSB corruption was not rejected before the first opening");
        }
        prepare(one);
        auto moved = std::move(one.clip_material.material);
        const auto before_sent = first->bytesSent, before_received = first->bytesReceived;
        rejected = false;
        try { (void)segfss::sigmaBatchOnline(one, shares); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || before_sent != first->bytesSent || before_received != first->bytesReceived || one.prepared)
            throw std::runtime_error("moved-from private MSB stock was accepted");
        rejected = false;
        try { (void)segfss::privateMsbOffline(party == SERVER ? CLIENT : SERVER, 1, bits, 4, first.get()); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || before_sent != first->bytesSent || before_received != first->bytesReceived)
            throw std::runtime_error("private MSB generation accepted a foreign role before I/O");
    }
    if (peer->bytesSent != default_sent || peer->bytesReceived != default_received ||
        numRounds != default_rounds || first->bytesSent == 0 || second->bytesSent == 0)
        throw std::runtime_error("non-default activation touched default context statistics");
    const std::vector<int64_t> truncate_expected{-1025, -1024, -1024, -64,
                                                -1, -1, 0, 0, 1, 64, 1024};
    auto truncation = segfss::exactTruncateBatchOffline(
        party, shares.size(), bits, 6, first.get());
    std::vector<segfss::ExactTruncationKey*> pointers;
    for (auto& key : truncation) pointers.push_back(&key);
    auto truncated = segfss::exactTruncateBatch(party, shares, pointers, first.get());
    first->open_additive(truncated.data(), truncated.size(), bits - 6);
    for (std::size_t i = 0; i < truncated.size(); ++i) {
        if (truncated[i].value != GroupElement(
                static_cast<uint64_t>(truncate_expected[i]), bits - 6).value)
            throw std::runtime_error("batched exact truncation differs from floor literals");
        segfss::freeExactTruncationKey(truncation[i]);
    }
    auto foreign = segfss::sigmaBatchOffline(party, 1, bits, 12, 10, first.get());
    const auto foreign_sent = second->bytesSent, foreign_received = second->bytesReceived;
    rejected = false;
    try { (void)segfss::sigmaBatchOnline(party, {shares.front()}, foreign, second.get()); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || foreign_sent != second->bytesSent ||
        foreign_received != second->bytesReceived || !foreign.front().dpf_lut.consumed)
        throw std::runtime_error("foreign context inventory was accepted or emitted messages");
    segfss::freeSigmaActivationKey(foreign.front());
    for (const auto* context : {&one, &two})
        std::cout << "ACTIVATION_CONTEXT_LEDGER identity=" << context->identity
                  << " epochs=" << context->generation_epoch
                  << " generated=" << context->statistics.generated
                  << " consumed=" << context->statistics.consumed
                  << " failed_batches=" << context->statistics.failed_batches
                  << " offline_sent=" << context->statistics.offline_sent
                  << " offline_received=" << context->statistics.offline_received
                  << " online_sent=" << context->statistics.online_sent
                  << " online_received=" << context->statistics.online_received << '\n';
    std::cout << "ACTIVATION_CONTEXTS contexts=2 queries=2 bits=32 frac=12 q=10 "
              << "elements=11 concurrent=1 default_untouched=1 paper=" << paper
              << " segmented=" << segmented << " status=passed\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) return 2;
    party = std::stoi(argv[1]); port = std::stoi(argv[2]);
    peer = party == SERVER ? waitForPeer(port) : new Peer(address, port);
    if (argc == 4 && std::string(argv[3]) == "native-sign-clip") {
        checkAutomaticComparisonContracts();
        checkNativeSignClip();
        delete peer;
        return 0;
    }
    if (argc == 4 && (std::string(argv[3]) == "contexts" ||
                      std::string(argv[3]) == "paper-contexts" ||
                      std::string(argv[3]) == "segmented-contexts" ||
                      std::string(argv[3]) == "private-sign-contexts" ||
                      std::string(argv[3]) == "private-sign-segmented-contexts")) {
        checkContexts(std::string(argv[3]) != "contexts",
                       std::string(argv[3]) == "segmented-contexts" ||
                       std::string(argv[3]) == "private-sign-segmented-contexts",
                       std::string(argv[3]).find("private-sign") == 0);
        delete peer;
        return 0;
    }
    for (bool tanh : {false, true}) for (bool first_dpf : {true, false}) {
        std::vector<segfss::SigmaActivationKey> keys;
        for (bool dpf : {first_dpf, !first_dpf}) {
            setenv("SEGFSS_DPF_LUT", dpf ? "1" : "0", 1);
            keys.push_back(segfss::sigmaOffline(party, 16, 4, 5, peer));
        }
        const auto sent = peer->bytesSent, received = peer->bytesReceived;
        bool rejected = false;
        try {
            const std::vector<GroupElement> inputs{GroupElement(0,16), GroupElement(0,16)};
            if (tanh) segfss::tanhBatchOnline(party, inputs, keys, peer);
            else segfss::sigmaBatchOnline(party, inputs, keys, peer);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || sent != peer->bytesSent || received != peer->bytesReceived)
            throw std::runtime_error("heterogeneous activation backends must reject before I/O");
        for (auto& key : keys) {
            if (!key.segmented_lut.consumed || !key.dpf_lut.consumed)
                throw std::runtime_error("heterogeneous activation invalidation is incomplete");
            segfss::freeSigmaActivationKey(key);
        }
    }
    delete peer;
    std::cout << "ACTIVATION_BATCH_CONTRACT sigmoid,tanh native-first,dpf-first reject_before_IO consume_all status=passed\n";
}
