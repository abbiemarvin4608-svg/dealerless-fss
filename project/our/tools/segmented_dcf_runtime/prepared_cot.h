#ifndef SDCF_PREPARED_COT_H
#define SDCF_PREPARED_COT_H
#include "native_preprocessing.h"
#include "OT/ot.h"
#include <algorithm>
#include <cstring>
#include <memory>

namespace sdcf_arithmetic {
struct Call {
    int count;
    std::vector<int> widths;
    int messages_per_ot = 1;
    bool batched = false;
};
struct Pads {
    Call call;
    std::vector<uint64_t> zero, one;
    std::vector<uint8_t> random_choices;
};
struct Stock {
    int query = -1;
    size_t cursor = 0;
    std::vector<Pads> calls;
};
inline uint64_t mask(int width) {
    return width == 64 ? UINT64_MAX : (uint64_t(1) << width) - 1;
}
inline int width_at(const Call& call, int ot_index) {
    return call.widths[size_t(ot_index) / (call.count / call.widths.size())];
}
inline size_t payload_bytes(const Call& call) {
    size_t bits = 0;
    for (int width : call.widths) bits += size_t(call.count / call.widths.size()) * call.messages_per_ot * width;
    return (bits + 7) / 8;
}
inline uint64_t read_bits(const uint8_t* bytes, size_t position, int width) {
    const int offset = position % 8, count = (offset + width + 7) / 8;
    bytes += position / 8;
    uint64_t value = 0;
    std::memcpy(&value, bytes, std::min(count, 8));
    value >>= offset;
    if (count == 9) value |= uint64_t(bytes[8]) << (64 - offset);
    return value & mask(width);
}
inline void write_bits(uint8_t* bytes, size_t position, uint64_t value, int width) {
    const int offset = position % 8, count = (offset + width + 7) / 8;
    bytes += position / 8;
    value &= mask(width);
    bytes[0] |= uint8_t(value << offset);
    for (int i = 1; i < count; ++i) bytes[i] |= uint8_t(value >> (i * 8 - offset));
}
inline void validate(const Call& call) {
    if (call.count < 1 || call.messages_per_ot < 1 || call.widths.empty() ||
        call.count % call.widths.size() != 0 ||
        (!call.batched && (call.widths.size() != 1 || call.messages_per_ot != 1)))
        throw std::runtime_error("invalid arithmetic COT contract");
    for (int width : call.widths)
        if (width < 1 || width > 64) throw std::runtime_error("invalid arithmetic COT width");
}
template<class OT>
void prepare(OT& ot, Stock& stock, const std::vector<Call>& calls) {
    sdcf_preprocess::generation_begin(&ot);
    if (stock.query != sdcf_preprocess::current_query) {
        if (stock.cursor != stock.calls.size())
            throw std::runtime_error("unconsumed arithmetic COT inventory");
        stock.calls.clear(); stock.cursor = 0; stock.query = sdcf_preprocess::current_query;
    }
    for (const Call& call : calls) {
        validate(call);
        Pads pads{call, std::vector<uint64_t>(size_t(call.count) * call.messages_per_ot), {}, {}};
        if (ot.party == sci::ALICE) {
            pads.one.resize(pads.zero.size());
            ot.send_pre(call.count);
        } else {
            auto choices = std::make_unique<bool[]>(call.count);
            ot.prg.random_bool(choices.get(), call.count);
            pads.random_choices.resize(call.count);
            for (int i = 0; i < call.count; ++i) pads.random_choices[i] = choices[i];
            ot.recv_pre(choices.get(), call.count);
        }
        const int max_hashes = (*std::max_element(call.widths.begin(), call.widths.end()) * call.messages_per_ot + 127) / 128;
        std::unique_ptr<sci::block128[]> hashes(new sci::block128[2 * max_hashes]);
        for (int i = 0; i < call.count; ++i) {
            const int width = width_at(call, i);
            const int hashes_per_choice = (width * call.messages_per_ot + 127) / 128;
            const sci::block128 seed = ot.party == sci::ALICE ? ot.qT[i] : ot.tT[i];
            for (int h = 0; h < hashes_per_choice; ++h) {
                hashes[h] = call.batched ? sci::xorBlocks(seed, sci::toBlock(h)) : seed;
                if (ot.party == sci::ALICE)
                    hashes[hashes_per_choice + h] = sci::xorBlocks(hashes[h], ot.block_s);
            }
            ot.crh.Hn(hashes.get(), hashes.get(), hashes_per_choice * (ot.party == sci::ALICE ? 2 : 1));
            for (int lane = 0; lane < call.messages_per_ot; ++lane) {
                const size_t index = size_t(i) * call.messages_per_ot + lane;
                pads.zero[index] = read_bits(reinterpret_cast<const uint8_t*>(hashes.get()), size_t(lane) * width, width);
                if (ot.party == sci::ALICE)
                    pads.one[index] = read_bits(reinterpret_cast<const uint8_t*>(hashes.get() + hashes_per_choice),
                                                size_t(lane) * width, width);
            }
        }
        if (ot.party == sci::ALICE) { delete[] ot.qT; ot.qT = nullptr; }
        else { delete[] ot.tT; ot.tT = nullptr; }
        sdcf_preprocess::random_ot_complete(&ot, call.count);
        stock.calls.push_back(std::move(pads));
    }
}
inline Pads& consume(Stock& stock, const Call& call) {
    validate(call);
    if (stock.query != sdcf_preprocess::current_query || stock.cursor >= stock.calls.size())
        throw std::runtime_error("arithmetic COT inventory exhausted or stale");
    Pads& pads = stock.calls[stock.cursor];
    if (pads.call.count != call.count || pads.call.widths != call.widths ||
        pads.call.messages_per_ot != call.messages_per_ot || pads.call.batched != call.batched)
        throw std::runtime_error("arithmetic COT material contract mismatch");
    return pads;
}
template<class OT>
void send(OT& ot, Stock& stock, uint64_t* output, const uint64_t* correlations, const Call& call) {
    auto& pads = consume(stock, call);
    sdcf_preprocess::consume(&ot, call.batched ? "iknp_batched_cot" : "iknp_cot", call.count,
                            *std::max_element(call.widths.begin(), call.widths.end()), call.messages_per_ot, 0, call.widths);
    std::vector<uint8_t> correction((call.count + 7) / 8);
    ot.io->recv_data(correction.data(), int(correction.size()));
    std::vector<uint8_t> messages(payload_bytes(call), 0);
    size_t position = 0;
    for (int i = 0; i < call.count; ++i) {
        const int d = (correction[i / 8] >> (i % 8)) & 1;
        const int width = width_at(call, i);
        for (int lane = 0; lane < call.messages_per_ot; ++lane) {
            const size_t index = size_t(i) * call.messages_per_ot + lane;
            output[index] = d ? pads.one[index] : pads.zero[index];
            const uint64_t value = (correlations[index] + output[index] - (d ? pads.zero[index] : pads.one[index])) & mask(width);
            write_bits(messages.data(), position, value, width);
            position += width;
        }
    }
    ot.io->send_data(messages.data(), int(messages.size()));
    pads.zero.clear(); pads.one.clear(); ++stock.cursor;
}
template<class OT>
void receive(OT& ot, Stock& stock, uint64_t* output, const bool* choices, const Call& call) {
    auto& pads = consume(stock, call);
    sdcf_preprocess::consume(&ot, call.batched ? "iknp_batched_cot" : "iknp_cot", call.count,
                            *std::max_element(call.widths.begin(), call.widths.end()), call.messages_per_ot, 0, call.widths);
    std::vector<uint8_t> correction((call.count + 7) / 8, 0);
    for (int i = 0; i < call.count; ++i)
        correction[i / 8] |= uint8_t(choices[i] ^ bool(pads.random_choices[i])) << (i % 8);
    ot.io->send_data(correction.data(), int(correction.size()));
    std::vector<uint8_t> messages(payload_bytes(call));
    ot.io->recv_data(messages.data(), int(messages.size()));
    size_t position = 0;
    for (int i = 0; i < call.count; ++i) {
        const int width = width_at(call, i);
        for (int lane = 0; lane < call.messages_per_ot; ++lane) {
            const size_t index = size_t(i) * call.messages_per_ot + lane;
            const uint64_t value = read_bits(messages.data(), position, width);
            output[index] = (pads.zero[index] + (choices[i] ? value : 0)) & mask(width);
            position += width;
        }
    }
    pads.zero.clear(); pads.random_choices.clear(); ++stock.cursor;
}
}
#endif
