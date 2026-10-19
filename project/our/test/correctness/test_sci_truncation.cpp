#include "sci_truncation.h"
#include "mpc/api.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

int party_instance = 0;
int party = 0;
int32_t bitlength = 32;
int num_threads = 1;
int port = 32000;
std::string address = "127.0.0.1";
int num_argmax = 1000;
uint8_t choice_bit = 0;
bool verbose = false;
int length = 1;
Peer* client = nullptr;
Peer* server = nullptr;
Dealer* dealer = nullptr;
Peer* peer = nullptr;

namespace {
using segfss::DcfBackend;
using segfss::reference::Contract;
using segfss::reference::NumericMode;
using segfss::reference::SciCall;
std::size_t tested_values = 0, tested_configs = 0;
std::map<std::pair<DcfBackend, NumericMode>, std::pair<std::size_t, std::size_t>> coverage;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void invalid(Function&& function, const std::string& name) {
    bool caught = false;
    try { function(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, name + " must reject invalid input");
}

struct ScopedEnvironment {
    const char* name;
    bool present;
    std::string value;
    ScopedEnvironment(const char* variable, const char* setting) : name(variable),
        present(std::getenv(variable) != nullptr), value(present ? std::getenv(variable) : "") {
        setenv(name, setting, 1);
    }
    ~ScopedEnvironment() {
        if (present) setenv(name, value.c_str(), 1);
        else unsetenv(name);
    }
};

void configurationTests() {
    ScopedEnvironment fusion("SEGFSS_FUSE_SIGN_WRAP", "0");
    static_assert(!std::is_copy_constructible<segfss::SciTruncationMaterial>::value,
                  "truncation material must have unique ownership");
    static_assert(!std::is_copy_assignable<segfss::SciTruncationMaterial>::value,
                  "truncation material cannot be copied by assignment");
    const Contract contract{8, 8, 1};
    const auto config = segfss::makeSciTruncationConfig(
        contract, DcfBackend::segmented, 4);
    require(config.low_shift_nseg == 1 && config.sign_nseg == 4 && config.wrap_nseg == 4,
            "comparison domains must use legal explicit segment counts");
    for (int nseg : {-1, 0, 2}) {
        auto invalid_config = config;
        invalid_config.low_shift_nseg = nseg;
        invalid([&] { segfss::sciTruncationDomains(invalid_config); }, "illegal explicit nseg");
    }
    auto full = segfss::makeSciTruncationConfig(contract, DcfBackend::full_tree, 4);
    require(full.low_shift_nseg == 1 && full.sign_nseg == 1 && full.wrap_nseg == 1,
            "full-tree must use one tree per comparison domain");
    full.sign_nseg = 2;
    invalid([&] { segfss::sciTruncationDomains(full); }, "segmented config labelled full-tree");
    const auto local = segfss::makeSciTruncationConfig({8, 8, -3, SciCall::convert}, DcfBackend::segmented, 4);
    const auto domains = segfss::sciTruncationDomains(local);
    require(domains.local_only && domains.low_shift_bits == 0 && domains.sign_bits == 0 &&
            domains.wrap_bits == 0, "local conversion must not construct comparison domains");
    invalid([&] { segfss::makeSciTruncationConfig(contract, DcfBackend::segmented, 0); }, "zero policy limit");
    invalid([&] { segfss::makeSciTruncationConfig({8, 8, 1, SciCall::convert}, DcfBackend::segmented, 4); },
            "positive conversion shift");
    using segfss::SegmentPolicy;
    const std::vector<int> widths{1, 8, 9, 16, 17, 24, 25, 31, 32};
    const std::vector<int> expected{1, 1, 2, 2, 3, 3, 4, 4, 4};
    for (std::size_t i = 0; i < widths.size(); ++i) {
        require(segfss::segmentCount(widths[i], SegmentPolicy::depth_cap_8) == expected[i],
                "depth-cap-8 must use the minimum public segment count");
        const segfss::SegmentLayout layout(widths[i], expected[i]);
        require(layout.maxWidth() <= 8 && layout.shift.back() == 0,
                "depth-cap-8 must cover contiguous MSB-first segments");
    }
    require(segfss::segmentCount(0, SegmentPolicy::depth_cap_8) == 0,
            "zero-bit domain must be inactive");
    const auto depth_cap = segfss::makeSciTruncationConfig(
        {32, 32, 14}, DcfBackend::segmented, SegmentPolicy::depth_cap_8);
    require(depth_cap.low_shift_nseg == 2 && depth_cap.sign_nseg == 4 && depth_cap.wrap_nseg == 4,
            "signed32 shift14 must choose low14/sign31/wrap32 independently");
    auto policy_mismatch = depth_cap;
    policy_mismatch.segment_policy = SegmentPolicy::fixed_4;
    require(!segfss::sameSciTruncationConfig(depth_cap, policy_mismatch),
            "material contract must include policy even when segment counts agree");
    invalid([&] { segfss::sciTruncationDomains(policy_mismatch); }, "policy/count mismatch");
    setenv("SEGFSS_FUSE_SIGN_WRAP", "1", 1);
    const auto fused = segfss::makeSciTruncationConfig(
        {32, 32, 14}, DcfBackend::segmented, SegmentPolicy::depth_cap_8);
    const auto fused_domains = segfss::sciTruncationDomains(fused);
    require(fused.fuse_sign_wrap && fused.wrap_nseg == 0 && fused_domains.wrap_bits == 0 &&
            fused_domains.fused_sign_wrap && fused_domains.correction_count == 3,
            "fusion removes the wrap comparison and one correction bit");
    auto flag_mismatch = fused;
    flag_mismatch.fuse_sign_wrap = false;
    require(!segfss::sameSciTruncationConfig(fused, flag_mismatch),
            "material contract must bind the sign/wrap fusion flag");
    for (const Contract fallback : {Contract{32, 32, 14, SciCall::truncate, false},
            Contract{32, 18, 14, SciCall::truncate, true}, Contract{64, 64, 0, SciCall::convert, true}}) {
        const auto config = segfss::makeSciTruncationConfig(fallback, DcfBackend::segmented, 4);
        require(config.fuse_sign_wrap && !segfss::sciTruncationDomains(config).fused_sign_wrap,
                "unsigned, nonextending and local contracts use their original paths");
    }
    // Independent truth table for the fused integer correction, before any
    // sharing, DCF, or ring arithmetic is involved.
    for (int a : {0, 1}) for (int l : {0, 1}) for (int b : {0, 1}) {
        const int sign = a ^ l ^ b;
        const int wrap = b ? (a & l) : (a | l);
        const int t = (a ^ b) & (l ^ b);
        require(wrap - sign == (1 - 2 * b) * t, "sign/wrap fusion truth table");
    }
}

void ringPackingTests() {
    // Exercise widths that cross byte boundaries as well as the exact 64-bit
    // path.  Values are intentionally wider than the selected ring so the
    // masking contract is checked too.
    for (const int width : {1, 2, 7, 8, 9, 13, 16, 31, 32, 63, 64}) {
        const std::vector<uint64_t> values{
            0, 1, 0x55aa55aa55aa55aaULL, UINT64_MAX,
            0x0123456789abcdefULL, 17};
        const auto payload = segfss::packRingValues(values, width);
        const auto unpacked = segfss::unpackRingValues(payload, values.size(), width);
        const uint64_t mask = segfss::reference::mask(width);
        require(unpacked.size() == values.size(), "packed ring count");
        for (std::size_t index = 0; index < values.size(); ++index)
            require(unpacked[index] == (values[index] & mask), "packed ring value");
        require(payload.size() == (values.size() * static_cast<std::size_t>(width) + 7) / 8,
                "packed ring payload size");
    }
    invalid([] { segfss::packRingValues({1}, 0); }, "zero-width ring packing");
    invalid([] { segfss::unpackRingValues({0}, 1, 0); }, "zero-width ring unpacking");
    invalid([] { segfss::unpackRingValues({0x80}, 1, 1); }, "nonzero ring tail padding");
    invalid([] { segfss::unpackRingValues({}, 1, 8); }, "truncated ring payload");
}

std::vector<GroupElement> shareInputs(const std::vector<uint64_t>& values, int bits) {
    std::vector<GroupElement> inputs;
    const uint64_t mask = segfss::reference::mask(bits);
    for (std::size_t row = 0; row < values.size(); ++row) {
        const uint64_t server_share = (row * 0x9e3779b97f4a7c15ULL + 0x8d67341ULL) & mask;
        inputs.emplace_back(party == SERVER ? server_share : (values[row] - server_share) & mask, bits);
    }
    return inputs;
}

std::vector<uint64_t> openOutputs(const std::vector<GroupElement>& outputs, int bits) {
    std::vector<uint64_t> own(outputs.size()), remote(outputs.size());
    for (std::size_t row = 0; row < outputs.size(); ++row) own[row] = outputs[row].value;
    if (party == SERVER) {
        peer->send_u64(own.data(), own.size());
        peer->recv_u64(remote.data(), remote.size());
    } else {
        peer->recv_u64(remote.data(), remote.size());
        peer->send_u64(own.data(), own.size());
    }
    const uint64_t mask = segfss::reference::mask(bits);
    for (std::size_t row = 0; row < own.size(); ++row) own[row] = (own[row] + remote[row]) & mask;
    return own;
}

void checkOutputs(const std::vector<GroupElement>& outputs, const std::vector<uint64_t>& values,
                  const segfss::SciTruncationConfig& config) {
    const auto& contract = config.contract;
    require(outputs.size() == values.size(), "output count");
    const auto opened = openOutputs(outputs, contract.output_bits);
    for (std::size_t row = 0; row < values.size(); ++row) {
        const auto expected = segfss::reference::evaluate(values[row], contract);
        require(outputs[row].bitsize == contract.output_bits && opened[row] == expected,
                "oracle mismatch k=" + std::to_string(contract.input_bits) +
                " m=" + std::to_string(contract.output_bits) + " s=" + std::to_string(contract.shift) +
                " call=" + std::to_string(static_cast<int>(contract.call)) +
                " signed=" + std::to_string(contract.signed_arithmetic) +
                " mode=" + std::to_string(static_cast<int>(contract.numeric_mode)) +
                " input=" + std::to_string(values[row]) + " expected=" + std::to_string(expected) +
                " actual=" + std::to_string(opened[row]));
    }
    tested_values += values.size();
    ++tested_configs;
    auto& row = coverage[{config.backend, contract.numeric_mode}];
    ++row.first;
    row.second += values.size();
    if (contract.input_bits == 8 && contract.output_bits == 7 && contract.shift == 1 &&
        contract.numeric_mode == NumericMode::exact_floor_diagnostic && values == std::vector<uint64_t>{253})
        require(opened.front() == 126, "independent floor literal: -3 / 2 floors to -2, residue 126 in 7 bits");
}

void verifyBatch(const Contract& contract, DcfBackend backend, int max_segments,
                 const std::vector<uint64_t>& values, bool report = false) {
    const auto config = segfss::makeSciTruncationConfig(contract, backend, max_segments);
    const auto domains = segfss::sciTruncationDomains(config);
    const auto inputs = shareInputs(values, contract.input_bits);
    auto material = segfss::sciTruncateOfflineBatch(party, values.size(), config, peer);
    const uint64_t sent_before = peer->bytesSent, received_before = peer->bytesReceived;
    const auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer);
    if (domains.local_only) {
        require(peer->bytesSent == sent_before && peer->bytesReceived == received_before &&
                material.metrics.masked_input_values == 0 && material.metrics.online_logical_rounds == 0,
                "local conversion must not communicate");
    } else {
        require(material.metrics.masked_input_values == values.size() &&
                material.metrics.masked_conversion_bits == values.size() * domains.correction_count,
                "only masked inputs and masked correction bits are opened by the adapter");
    }
    checkOutputs(outputs, values, config);
    invalid([&] { segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer); }, "repeated consumption");
    if (party == SERVER && report) {
        std::cout << "socket k=" << contract.input_bits << " m=" << contract.output_bits
                  << " shift=" << contract.shift << " call=" << static_cast<int>(contract.call)
                  << " signed=" << contract.signed_arithmetic << " mode=" << static_cast<int>(contract.numeric_mode)
                  << " backend=" << (backend == DcfBackend::full_tree ? "full_tree" : "segmented")
                  << " count=" << values.size() << " low_shift=" << domains.low_shift_bits << '/' << config.low_shift_nseg
                  << " sign=" << domains.sign_bits << '/' << config.sign_nseg
                  << " wrap=" << domains.wrap_bits << '/' << config.wrap_nseg
                  << " masked_inputs=" << material.metrics.masked_input_values
                  << " masked_bits=" << material.metrics.masked_conversion_bits
                  << " logical_online_rounds=" << material.metrics.online_logical_rounds;
        for (const auto domain : {std::pair<const char*, const segfss::SegmentedDcfMaterial*>{"low_depths", &material.low_shift},
                                  {"sign_depths", &material.sign}, {"wrap_depths", &material.wrap}}) {
            std::cout << ' ' << domain.first << '=';
            if (domain.second->count == 0) std::cout << "none";
            else for (int width : domain.second->layout.width) std::cout << width << ',';
        }
        std::cout << " PASS\n";
    }
    segfss::releaseSciTruncationMaterial(material);
    segfss::releaseSciTruncationMaterial(material);
    invalid([&] { segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer); }, "released material");
}

void exhaustiveSmall() {
    for (int bits = 1; bits <= 6; ++bits) {
        std::vector<uint64_t> values;
        for (int share_variant = 0; share_variant < 2; ++share_variant)
            for (uint64_t value = 0; value < (uint64_t(1) << bits); ++value) values.push_back(value);
        for (int shift = 0; shift < bits; ++shift) {
            for (bool signed_arithmetic : {false, true}) {
                for (SciCall call : {SciCall::div_pow2, SciCall::truncate, SciCall::truncate_and_reduce}) {
                    const std::vector<int> widths = call == SciCall::truncate_and_reduce ?
                        std::vector<int>{bits - shift} : std::vector<int>{1, bits - shift, bits, bits + 2};
                    for (int output_bits : widths) {
                        Contract contract{bits, output_bits, shift, call, signed_arithmetic};
                        verifyBatch(contract, DcfBackend::full_tree, 1, values);
                        verifyBatch(contract, DcfBackend::segmented, 3, values);
                    }
                }
                verifyBatch({bits, bits - shift, shift, SciCall::div_pow2, signed_arithmetic,
                    NumericMode::exact_floor_diagnostic}, DcfBackend::segmented, 3, values);
            }
        }
        for (int shift : {0, -1, -3, -63}) {
            for (bool signed_arithmetic : {false, true}) {
                for (int output_bits : {1, bits, bits + 3}) {
                    Contract contract{bits, output_bits, shift, SciCall::convert, signed_arithmetic};
                    verifyBatch(contract, DcfBackend::full_tree, 1, values);
                    verifyBatch(contract, DcfBackend::segmented, 3, values);
                }
            }
        }
        if (party == SERVER) std::cout << "socket exhaustive bits=" << bits
                                      << " configurations=" << tested_configs << " values=" << tested_values << " PASS\n";
    }
}

void boundaryTests() {
    for (int bits : {8, 16, 20, 24, 32, 64}) {
        const uint64_t mask = segfss::reference::mask(bits), half = uint64_t(1) << (bits - 1);
        const std::vector<uint64_t> values{0, 1, 2, 3, half - 1, half, half + 1, mask - 2, mask - 1, mask};
        for (int shift : {1, bits / 2, bits - 1}) {
            for (bool signed_arithmetic : {false, true}) {
                for (int output_bits : {1, bits - shift, bits, 64}) {
                    verifyBatch({bits, output_bits, shift, SciCall::div_pow2, signed_arithmetic},
                                DcfBackend::segmented, 16, values, true);
                    verifyBatch({bits, output_bits, shift, SciCall::truncate, signed_arithmetic},
                                DcfBackend::segmented, 16, values);
                }
            }
        }
        for (int shift : {0, -1, -63}) {
            for (bool signed_arithmetic : {false, true})
                verifyBatch({bits, 64, shift, SciCall::convert, signed_arithmetic},
                            DcfBackend::segmented, 16, values, true);
        }
    }
    // Same nondivisible negative input, two independently named numeric modes.
    verifyBatch({8, 8, 1}, DcfBackend::segmented, 4, {253}, true);
    verifyBatch({8, 7, 1, SciCall::div_pow2, true, NumericMode::exact_floor_diagnostic},
                DcfBackend::segmented, 4, {253}, true);
    verifyBatch({8, 7, 1, SciCall::div_pow2, true, NumericMode::exact_floor_diagnostic},
                DcfBackend::full_tree, 1, {253}, true);
}

void ownershipTests() {
    const auto config = segfss::makeSciTruncationConfig({8, 12, 1}, DcfBackend::segmented, 4);
    auto original = segfss::sciTruncateOffline(party, config, peer);
    auto moved = std::move(original);
    const auto input = shareInputs({253}, 8).front();
    invalid([&] { segfss::sciTruncateOnline(party, input, config, original, peer); }, "moved-from material");
    invalid([&] { segfss::sciTruncateOnline(party, GroupElement(0, 7), config, moved, peer); }, "input ring mismatch");
    invalid([&] { segfss::sciTruncateOnlineBatch(party, {}, config, moved, peer); }, "input count mismatch");
    auto mismatch = config;
    mismatch.contract.call = SciCall::truncate;
    invalid([&] { segfss::sciTruncateOnline(party, input, mismatch, moved, peer); }, "call-site mismatch");
    mismatch = config;
    mismatch.contract.output_bits = 13;
    invalid([&] { segfss::sciTruncateOnline(party, input, mismatch, moved, peer); }, "output ring mismatch");
    mismatch = config;
    mismatch.contract.shift = 2;
    mismatch.low_shift_nseg = 2;
    invalid([&] { segfss::sciTruncateOnline(party, input, mismatch, moved, peer); }, "shift mismatch");
    checkOutputs({segfss::sciTruncateOnline(party, input, config, moved, peer)}, {253}, config);
    verifyBatch(config.contract, DcfBackend::segmented, 4, {253}, true);

    auto first = segfss::sciTruncateOffline(party, config, peer);
    auto second = segfss::sciTruncateOfflineBatch(party, 2, config, peer);
    const auto inputs = shareInputs({253, 0, 128}, 8);
    invalid([&] { segfss::sciTruncateOnlineBatch(party, {inputs[0], inputs[1]}, config,
                    std::vector<segfss::SciTruncationMaterial*>{&first, &first}, peer); }, "duplicate batch material");
    require(!first.consumed, "validation failure must occur before consuming valid donor material");
    segfss::SciTruncationMetrics metrics;
    const auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, config,
        std::vector<segfss::SciTruncationMaterial*>{&first, &second}, peer, &metrics);
    checkOutputs(outputs, {253, 0, 128}, config);
    require(first.consumed && second.consumed && metrics.masked_input_values == 3,
            "combined batch must consume every donor and open one complete input batch");
    invalid([&] { segfss::sciTruncateOnline(party, input, config, first, peer); }, "group donor reused");
    auto unused = segfss::sciTruncateOffline(party, config, peer);
    segfss::releaseSciTruncationMaterial(unused);
    invalid([&] { segfss::sciTruncateOnline(party, input, config, unused, peer); }, "release before use");
    const auto diagnostic = segfss::makeSciTruncationConfig(
        {8, 7, 1, SciCall::div_pow2, true, NumericMode::exact_floor_diagnostic}, DcfBackend::segmented, 4);
    auto diagnostic_material = segfss::sciTruncateOffline(party, diagnostic, peer);
    auto sci = diagnostic;
    sci.contract.numeric_mode = NumericMode::sci_compatible;
    sci.sign_nseg = 4;
    invalid([&] { segfss::sciTruncateOnline(party, input, sci, diagnostic_material, peer); }, "numeric mode mismatch");
    checkOutputs({segfss::sciTruncateOnline(party, input, diagnostic, diagnostic_material, peer)}, {253}, diagnostic);
    if (party == SERVER) std::cout << "ownership_single_batch_group_contract_modes PASS\n";
}

void largeBatch() {
    std::vector<uint64_t> values(12800);
    for (std::size_t row = 0; row < values.size(); ++row)
        values[row] = (row * 0x9e3779b9ULL + (row % 2 == 0 ? 1 : 0)) & UINT32_MAX;
    verifyBatch({32, 16, 14}, DcfBackend::segmented, 4, values, true);
}

void communicationTests() {
    setenv("SEGFSS_PACK_RING", "1", 1);
    for (int fused : {0, 1}) for (int packed : {0, 1}) {
        setenv("SEGFSS_FUSE_DOMAINS", fused ? "1" : "0", 1);
        setenv("SEGFSS_PACK_BITS", packed ? "1" : "0", 1);
        for (int count : {1, 7, 8, 9}) {
            const auto config = segfss::makeSciTruncationConfig({32, 32, 14}, DcfBackend::segmented, 4);
            std::vector<uint64_t> values(count);
            for (int i = 0; i < count; ++i) values[i] = i % 2 ? UINT32_MAX - i : (1ULL << 31) + i;
            auto material = segfss::sciTruncateOfflineBatch(party, count, config, peer);
            const auto inputs = shareInputs(values, 32);
            const auto sent = peer->bytesSent, received = peer->bytesReceived;
            const auto io_calls = peer->rounds;
            const auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer);
            require(material.metrics.online_logical_rounds == (fused ? 5 : 9), "three-domain signed conversion communication depth");
            require(peer->rounds - io_calls == 2 * (fused ? 5 : 9) + 1,
                    "framed masked opening adds one local receive call, no new send flight");
            const auto bytes = [packed](uint64_t bits) { return packed ? (bits + 7) / 8 : bits; };
            const uint64_t scan = fused ? bytes(24 * count) + bytes(12 * count) :
                3 * (bytes(8 * count) + bytes(4 * count));
            const uint64_t expected = segfss::kRingPackingHeaderBytes + 4 * count +
                scan + bytes(2 * count) + bytes(4 * count);
            require(peer->bytesSent - sent == expected && peer->bytesReceived - received == expected,
                    "actual conversion wire payload including partial packed bytes");
            checkOutputs(outputs, values, config);
            const auto extend = segfss::makeSciTruncationConfig(
                {8, 16, 0, SciCall::convert, false}, DcfBackend::segmented, 4);
            std::vector<uint64_t> narrow(count);
            for (int i = 0; i < count; ++i) narrow[i] = (31 * i + 127) & 255;
            auto extension = segfss::sciTruncateOfflineBatch(party, count, extend, peer);
            const auto narrow_inputs = shareInputs(narrow, 8);
            const auto extension_sent = peer->bytesSent;
            auto extended = segfss::sciTruncateOnlineBatch(party, narrow_inputs, extend, extension, peer);
            require(peer->bytesSent - extension_sent == segfss::kRingPackingHeaderBytes + count +
                    bytes(8 * count) + bytes(4 * count) + bytes(count),
                    "Boolean corrections of lengths 1,7,8,9 preserve tail bits and exact payload");
            checkOutputs(extended, narrow, extend);
        }
    }
    setenv("SEGFSS_FUSE_DOMAINS", "1", 1);
    setenv("SEGFSS_PACK_BITS", "1", 1);
    auto mixed = segfss::makeSciTruncationConfig({16, 16, 7}, DcfBackend::segmented, 4);
    mixed.low_shift_nseg = 2; mixed.sign_nseg = 3;
    const std::vector<uint64_t> values{0, 1, 127, 128, 32767, 32768, 65535};
    auto material = segfss::sciTruncateOfflineBatch(party, values.size(), mixed, peer);
    auto inputs = shareInputs(values, 16);
    const auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, mixed, material, peer);
    checkOutputs(outputs, values, mixed);
    require(material.metrics.online_logical_rounds == 5, "mixed 2/3/4-segment scan uses maximum depth");
    verifyBatch({1, 8, 0, SciCall::convert}, DcfBackend::segmented, 4, {0, 1});
    verifyBatch({8, 8, -2, SciCall::convert}, DcfBackend::segmented, 4, {0, 1, 255});
    unsetenv("SEGFSS_FUSE_DOMAINS"); unsetenv("SEGFSS_PACK_BITS");
    unsetenv("SEGFSS_PACK_RING");
}

void policyTests() {
    setenv("SEGFSS_MATERIAL_LAYOUT", "contiguous", 1);
    setenv("SEGFSS_FUSE_DOMAINS", "1", 1);
    setenv("SEGFSS_GENERATION_TILE_KEYS", "7", 1);
    setenv("SEGFSS_GENERATION_BUDGET_BYTES", "268435456", 1);
    setenv("SEGFSS_GENERATION_RESERVE_BYTES", "1048576", 1);
    for (auto policy : {segfss::SegmentPolicy::fixed_2, segfss::SegmentPolicy::fixed_4,
                        segfss::SegmentPolicy::depth_cap_8}) {
        const auto config = segfss::makeSciTruncationConfig({32, 32, 14}, DcfBackend::segmented, policy);
        std::vector<uint64_t> values(128);
        const uint64_t literals[]{0, 1, 16383, 16384, 2147483647, 2147483648, 4294950913, 4294967295};
        for (std::size_t row = 0; row < values.size(); ++row) values[row] = literals[row % 8];
        auto material = segfss::sciTruncateOfflineBatch(party, values.size(), config, peer);
        auto mismatch = config;
        mismatch.segment_policy = segfss::SegmentPolicy::explicit_counts;
        const auto inputs = shareInputs(values, 32);
        invalid([&] { segfss::sciTruncateOnlineBatch(party, inputs, mismatch, material, peer); },
                "serialized policy contract mismatch");
        require(!material.consumed, "policy mismatch must fail before material consumption");
        const auto output = segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer);
        checkOutputs(output, values, config);
        require(material.metrics.online_logical_rounds == (policy == segfss::SegmentPolicy::fixed_2 ? 4 : 5),
                "fused conversion depth follows the deepest comparison domain");
        segfss::releaseSciTruncationMaterial(material);
    }
    unsetenv("SEGFSS_MATERIAL_LAYOUT"); unsetenv("SEGFSS_FUSE_DOMAINS");
    unsetenv("SEGFSS_GENERATION_TILE_KEYS"); unsetenv("SEGFSS_GENERATION_BUDGET_BYTES");
    unsetenv("SEGFSS_GENERATION_RESERVE_BYTES");
}

void demandPruningTests() {
    setenv("SEGFSS_AES_ONLINE", "1", 1);
    setenv("SEGFSS_PACK_BITS", "1", 1);
    for (bool compact : {false, true}) for (bool equality : {false, true})
        for (bool leaf : {false, true}) for (bool fused : {false, true}) {
        setenv("SEGFSS_MATERIAL_LAYOUT", compact ? "contiguous" : "legacy", 1);
        setenv("SEGFSS_PRUNE_EQUALITY", equality ? "1" : "0", 1);
        setenv("SEGFSS_PRUNE_LEAF_HASHES", leaf ? "1" : "0", 1);
        setenv("SEGFSS_FUSE_DOMAINS", fused ? "1" : "0", 1);
        const std::vector<uint64_t> values{0, 1, 7, 8, 127, 128, 129, 248, 249, 254, 255};
        for (int nseg : {1, 2, 3, 4}) for (const Contract contract : {
                Contract{8, 8, 3, SciCall::truncate, true},
                Contract{8, 8, 3, SciCall::div_pow2, true},
                Contract{8, 5, 3, SciCall::truncate_and_reduce, true},
                Contract{8, 16, 0, SciCall::convert, true},
                Contract{8, 16, 0, SciCall::convert, false}}) {
            const auto config = segfss::makeSciTruncationConfig(contract, DcfBackend::segmented, nseg);
            const auto domains = segfss::sciTruncationDomains(config);
            auto material = segfss::sciTruncateOfflineBatch(party, values.size(), config, peer);
            require(!material.low_shift.count || material.low_shift.need_e == domains.toward_zero,
                    "only signed toward-zero div requires low-domain equality");
            for (const auto* comparison : {&material.low_shift, &material.sign, &material.wrap})
                if (comparison->count)
                    require(comparison->scan.size() == values.size() * segfss::planDcfScan(
                        comparison->nseg, comparison->need_e, equality).and_gates_per_row,
                        "conversion inventory matches public operator output demand");
            const auto inputs = shareInputs(values, contract.input_bits);
            const auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer);
            checkOutputs(outputs, values, config);
        }
    }
    for (const char* flag : {"SEGFSS_AES_ONLINE", "SEGFSS_PACK_BITS", "SEGFSS_MATERIAL_LAYOUT",
            "SEGFSS_PRUNE_EQUALITY", "SEGFSS_PRUNE_LEAF_HASHES", "SEGFSS_FUSE_DOMAINS"}) unsetenv(flag);
    if (party == SERVER) std::cout << "conversion_demand_pruning modes=32 nseg=1,2,3,4 contracts=5 boundary_values=11 PASS\n";
}

std::vector<GroupElement> randomizedShares(const std::vector<uint64_t>& values, int bits, uint64_t seed) {
    std::mt19937_64 random(seed);
    std::vector<GroupElement> shares;
    const uint64_t mask = segfss::reference::mask(bits);
    for (auto value : values) {
        const uint64_t first = random() & mask;
        shares.emplace_back(party == SERVER ? first : (value - first) & mask, bits);
    }
    return shares;
}

void signWrapFusionTests() {
    ScopedEnvironment fusion("SEGFSS_FUSE_SIGN_WRAP", "0");
    ScopedEnvironment domains_flag("SEGFSS_FUSE_DOMAINS", "1");
    ScopedEnvironment packed("SEGFSS_PACK_BITS", "1");
    ScopedEnvironment ring_packed("SEGFSS_PACK_RING", "1");
    ScopedEnvironment prune("SEGFSS_PRUNE_EQUALITY", "1");
    ScopedEnvironment leaves("SEGFSS_PRUNE_LEAF_HASHES", "1");
    ScopedEnvironment aes("SEGFSS_AES_ONLINE", "1");
    std::vector<Contract> contracts;
    for (const int bits : {1, 8, 32, 64}) {
        const int wide = std::min(64, bits + 8);
        for (const NumericMode mode : {NumericMode::sci_compatible, NumericMode::exact_floor_diagnostic}) {
            if (mode == NumericMode::exact_floor_diagnostic) {
                // This existing diagnostic contract requires exactly the
                // shrinking output ring, so sign/wrap fusion stays inactive.
                contracts.push_back({bits, bits, 0, SciCall::convert, true, mode});
                if (bits > 1) {
                    const int shift = bits == 32 ? 14 : bits / 2;
                    for (const SciCall call : {SciCall::div_pow2, SciCall::truncate, SciCall::truncate_and_reduce})
                        for (const bool signed_arithmetic : {false, true})
                            contracts.push_back({bits, bits - shift, shift, call, signed_arithmetic, mode});
                }
                continue;
            }
            contracts.push_back({bits, wide, 0, SciCall::convert, true, mode});
            contracts.push_back({bits, wide, 0, SciCall::convert, false, mode});
            if (bits > 1) {
                const int shift = bits == 32 ? 14 : bits / 2;
                contracts.push_back({bits, bits, shift, SciCall::div_pow2, true, mode});
                contracts.push_back({bits, wide, shift, SciCall::truncate, true, mode});
                contracts.push_back({bits, bits, shift, SciCall::div_pow2, false, mode});
                contracts.push_back({bits, bits - shift, shift, SciCall::truncate_and_reduce, true, mode});
                contracts.push_back({bits, 1, bits - 1, SciCall::div_pow2, true, mode});
                contracts.push_back({bits, wide, -1, SciCall::convert, true, mode});
            }
        }
    }
    std::size_t fused_configs = 0;
    for (std::size_t case_index = 0; case_index < contracts.size(); ++case_index) {
        const auto contract = contracts[case_index];
        const uint64_t mask = segfss::reference::mask(contract.input_bits);
        const uint64_t half = uint64_t(1) << (contract.input_bits - 1);
        std::vector<uint64_t> values{0, 1, half - 1, half, half + 1, mask - 1, mask};
        std::mt19937_64 random(0x51f732ULL + case_index);
        for (int i = 0; i < 5; ++i) values.push_back(random() & mask);
        for (auto& value : values) value &= mask;
        const auto inputs = randomizedShares(values, contract.input_bits, 0xac764dULL + case_index);
        for (int enabled : {0, 1}) {
            setenv("SEGFSS_FUSE_SIGN_WRAP", enabled ? "1" : "0", 1);
            // Exercise both independent and grouped scans, and both compact
            // and owning key layouts without multiplying the case count.
            setenv("SEGFSS_FUSE_DOMAINS", case_index % 2 ? "1" : "0", 1);
            ScopedEnvironment compact("SEGFSS_MATERIAL_LAYOUT", case_index % 2 ? "contiguous" : "legacy");
            const auto config = segfss::makeSciTruncationConfig(contract,
                DcfBackend::segmented, segfss::SegmentPolicy::depth_cap_8);
            const auto domains = segfss::sciTruncationDomains(config);
            const auto b2a_before = peer->preprocessing_ledger.b2a_values;
            auto material = segfss::sciTruncateOfflineBatch(party, values.size(), config, peer);
            const std::size_t post_gates = values.size() * (domains.toward_zero + domains.fused_sign_wrap);
            require(material.rounding.size() == post_gates && material.rounding.cursor == 0,
                    "joint sign/wrap and rounding inventory has exactly one gate per requested product");
            require(material.conversion_mask_bits.size() == values.size() * domains.correction_count,
                    "fusion correction mask inventory matches its reduced correction count");
            require(peer->preprocessing_ledger.b2a_values - b2a_before ==
                    (domains.local_only ? 0 : values.size() * (contract.input_bits + domains.correction_count)),
                    "offline B2A ledger includes only the requested input and correction masks");
            if (domains.fused_sign_wrap) {
                ++fused_configs;
                require(material.wrap.count == 0 && material.wrap.keyCount() == 0 && material.wrap.scan.size() == 0,
                        "fused signed extension generates no wrap DCF");
            }
            const auto sent = peer->bytesSent, received = peer->bytesReceived;
            // Both stale expected configs and a freshly rebuilt opposite
            // config must fail before consuming stock or starting network I/O.
            setenv("SEGFSS_FUSE_SIGN_WRAP", enabled ? "0" : "1", 1);
            invalid([&] { segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer); },
                    "runtime fusion flag mismatch");
            const auto mismatch = segfss::makeSciTruncationConfig(contract,
                DcfBackend::segmented, segfss::SegmentPolicy::depth_cap_8);
            invalid([&] { segfss::sciTruncateOnlineBatch(party, inputs, mismatch, material, peer); },
                    "stored fusion flag mismatch");
            invalid([&] { segfss::sciTruncateOfflineBatch(party, values.size(), config, peer); },
                    "offline fusion flag mismatch");
            require(!material.consumed && material.rounding.cursor == 0 &&
                    peer->bytesSent == sent && peer->bytesReceived == received,
                    "flag mismatch does not consume inventory or communicate");
            setenv("SEGFSS_FUSE_SIGN_WRAP", enabled ? "1" : "0", 1);
            const auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer);
            std::size_t layers = 0;
            for (const auto* comparison : {&material.low_shift, &material.sign, &material.wrap}) {
                if (case_index % 2) layers = std::max(layers, comparison->metrics.online_scan_layers);
                else layers += comparison->metrics.online_scan_layers;
            }
            require(material.metrics.online_logical_rounds ==
                    (domains.local_only ? 0 : 2 + layers + (post_gates > 0)),
                    "joint products add exactly one online round, including fusion without rounding");
            require(material.rounding.cursor == post_gates && material.metrics.masked_conversion_bits ==
                    (domains.local_only ? 0 : values.size() * domains.correction_count),
                    "joint product inventory is consumed once and conversion accounting is exact");
            checkOutputs(outputs, values, config);
            invalid([&] { segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer); },
                    "fused material repeated consumption");
            segfss::releaseSciTruncationMaterial(material);
            require(material.rounding.size() == 0, "release clears joint product inventory");
        }
    }
    // Keep the full-tree backend covered on small domains, where construction
    // is bounded, and exercise row-major stock concatenation across donors.
    setenv("SEGFSS_FUSE_SIGN_WRAP", "1", 1);
    verifyBatch({8, 16, 3}, DcfBackend::full_tree, 1, {0, 1, 127, 128, 129, 253, 255});
    ownershipTests();
    if (party == SERVER) std::cout << "sign_wrap_fusion configs=" << contracts.size() * 2
        << " fused=" << fused_configs << " random_sharing=1 flag_binding=1 joint_and_round=1 PASS\n";
}
void auxiliarySignTests() {
    const std::vector<uint64_t> values{32768, 32769, 65535, 0, 1, 32766, 32767};
    for (const char* fusion : {"0", "1"}) {
        ScopedEnvironment environment("SEGFSS_FUSE_SIGN_WRAP", fusion);
        const auto config = segfss::makeSciTruncationConfig(
            {16, 32, 0, SciCall::convert, true}, DcfBackend::segmented, 4);
        const auto inputs = shareInputs(values, 16);
        auto baseline = segfss::sciTruncateOfflineBatch(party, values.size(), config, peer);
        auto auxiliary = segfss::sciTruncateOfflineBatch(party, values.size(), config, peer);
        auto sent = peer->bytesSent, received = peer->bytesReceived;
        const auto baseline_outputs = segfss::sciTruncateOnlineBatch(party, inputs, config, baseline, peer);
        const auto baseline_sent = peer->bytesSent - sent, baseline_received = peer->bytesReceived - received;
        sent = peer->bytesSent; received = peer->bytesReceived;
        std::vector<u8> signs;
        const auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, config, auxiliary,
            peer, nullptr, &signs);
        require(signs.size() == values.size() && peer->bytesSent - sent == baseline_sent &&
            peer->bytesReceived - received == baseline_received &&
            auxiliary.metrics.online_logical_rounds == baseline.metrics.online_logical_rounds,
            "extracting the existing sign adds no online communication or rounds");
        checkOutputs(baseline_outputs, values, config);
        checkOutputs(outputs, values, config);
        // These are public synthetic test inputs, so reconstruct only here
        // to verify the returned XOR signs independently of the extension.
        peer->open_xor(signs.data(), signs.size());
        for (std::size_t i = 0; i < signs.size(); ++i)
            require(signs[i] == (values[i] >> 15), "auxiliary XOR sign differs at a signed16 boundary");
    }
    const auto local = segfss::makeSciTruncationConfig(
        {16, 16, 0, SciCall::convert, true}, DcfBackend::segmented, 4);
    auto material = segfss::sciTruncateOfflineBatch(party, values.size(), local, peer);
    const auto inputs = shareInputs(values, 16);
    const auto sent = peer->bytesSent, received = peer->bytesReceived;
    std::vector<u8> signs;
    invalid([&] { (void)segfss::sciTruncateOnlineBatch(party, inputs, local, material,
        peer, nullptr, &signs); }, "auxiliary sign for a contract without a sign comparison");
    require(!material.consumed && sent == peer->bytesSent && received == peer->bytesReceived,
        "unsupported auxiliary sign request rejects before consumption or I/O");
    if (party == SERVER) std::cout << "AUXILIARY_INPUT_SIGN boundaries=7 fusion=0,1 extra_bytes=0 extra_rounds=0 PASS\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        configurationTests();
        ringPackingTests();
        if (argc == 1) {
            std::cout << "SCI truncation configuration: PASS\n";
            return 0;
        }
        require(argc == 3 || argc == 4, "usage: SCI_TRUNCATION_TEST [role port [large]]");
        party = std::stoi(argv[1]);
        port = std::stoi(argv[2]);
        require(party == SERVER || party == CLIENT, "role must be 2 or 3");
        segfss::setCggmThreadLimit(1);
        peer = party == SERVER ? waitForPeer(port) : new Peer(address, port);
        if (argc == 4) {
            const std::string mode(argv[3]);
            require(mode == "large" || mode == "communication" || mode == "policy" || mode == "demand" ||
                    mode == "fusion" || mode == "sign-aux", "unknown test mode");
            if (mode == "communication") communicationTests();
            else if (mode == "fusion") signWrapFusionTests();
            else if (mode == "sign-aux") auxiliarySignTests();
            else if (mode == "demand") demandPruningTests();
            else if (mode == "policy") policyTests(); else largeBatch();
        } else {
            exhaustiveSmall();
            boundaryTests();
            ownershipTests();
        }
        delete peer;
        peer = nullptr;
        for (const auto& row : coverage) {
            std::cout << "COVERAGE backend=" << (row.first.first == DcfBackend::full_tree ? "full_tree" : "segmented")
                      << " numeric_mode=" << (row.first.second == NumericMode::sci_compatible ? "sci_compatible" : "exact_floor_diagnostic")
                      << " configurations=" << row.second.first << " values=" << row.second.second << '\n';
        }
        std::cout << "SCI truncation: PASS configurations=" << tested_configs << " values=" << tested_values << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SCI truncation: FAIL " << error.what() << '\n';
        return 1;
    }
}
