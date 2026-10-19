#ifndef SDCF_PREPARED_GOT_H
#define SDCF_PREPARED_GOT_H
#include "native_preprocessing.h"
#include "utils/ccrf.h"
#include <algorithm>
#include <cstring>
#include <deque>

namespace sdcf_preprocess {
struct PreparedGOT {
    MaterialCall call;
    std::vector<uint8_t> choices, masks;
    std::vector<uint64_t> words;
};
inline std::mutex got_mutex;
inline std::map<const void*, std::deque<PreparedGOT>> got_material;

inline int selection_bits(int radix) {
    int bits = 0;
    while ((1 << bits) < radix) ++bits;
    return bits;
}
inline void put_bits(uint8_t* data, size_t offset, uint64_t value, int bits) {
    while (bits) {
        const int used = int(offset % 8), take = std::min(bits, 8 - used);
        data[offset / 8] |= uint8_t((value & ((1U << take) - 1)) << used);
        offset += take; value >>= take; bits -= take;
    }
}
inline uint64_t get_bits(const uint8_t* data, size_t offset, int bits) {
    uint64_t value = 0;
    int shift = 0;
    while (bits) {
        const int used = int(offset % 8), take = std::min(bits, 8 - used);
        value |= uint64_t((data[offset / 8] >> used) & ((1U << take) - 1)) << shift;
        offset += take; shift += take; bits -= take;
    }
    return value;
}
inline void store_got(const void* ot, PreparedGOT&& material) {
    std::lock_guard<std::mutex> lock(got_mutex);
    auto& queue = got_material[ot];
    if (!queue.empty() && queue.front().call.query != material.call.query)
        throw std::runtime_error("unconsumed GOT material from previous query");
    queue.push_back(std::move(material));
}
inline PreparedGOT take_got(const void* ot, const std::string& protocol,
                            int count, int bits, int grouping) {
    std::lock_guard<std::mutex> lock(got_mutex);
    auto found = got_material.find(ot);
    if (found == got_material.end() || found->second.empty())
        throw std::runtime_error("prepared GOT inventory exhausted");
    const auto& call = found->second.front().call;
    if (call.query != current_query || call.protocol != protocol || call.count != count ||
        call.message_bits != bits || call.messages_per_ot != grouping)
        throw std::runtime_error("prepared GOT contract mismatch");
    PreparedGOT result = std::move(found->second.front());
    found->second.pop_front();
    return result;
}

template <class OT> void prepare_kkot(OT* ot, const MaterialCall& call) {
    require_query(call.query);
    generation_begin(ot);
    if (call.protocol != "kkot_got" || call.messages_per_ot != ot->N)
        throw std::runtime_error("invalid prepared KKOT contract");
    PreparedGOT material;
    material.call = call;
    material.choices.resize(call.count);
    if (ot->party == sci::ALICE) ot->send_pre(call.count);
    else {
        ot->prg.random_data(material.choices.data(), call.count);
        for (auto& choice : material.choices) choice &= uint8_t(ot->N - 1);
        ot->recv_pre(material.choices.data(), call.count);
    }
    const int branches = ot->party == sci::ALICE ? ot->N : 1;
    material.words.resize(size_t(call.count) * branches);
    const int chunk = 256;
    auto* keys = new (std::align_val_t(32)) sci::block256[chunk * branches];
    auto* hashes = new sci::block128[chunk * branches];
    for (int start = 0; start < call.count; start += chunk) {
        const int count = std::min(chunk, call.count - start);
        for (int i = 0; i < count; ++i) for (int branch = 0; branch < branches; ++branch)
            keys[i * branches + branch] = ot->party == sci::ALICE ?
                sci::xorBlocks(ot->qT[start + i], ot->c_AND_s[branch]) : ot->tT[start + i];
        sci::CCRF(hashes, keys, count * branches);
        for (int i = 0; i < count * branches; ++i)
            material.words[size_t(start) * branches + i] = uint64_t(_mm_extract_epi64(hashes[i], 0));
    }
    delete[] keys; delete[] hashes;
    if (ot->party == sci::ALICE) { delete[] ot->qT; ot->qT = nullptr; }
    else { delete[] ot->tT; ot->tT = nullptr; }
    random_ot_complete(ot, call.count);
    store_got(ot, std::move(material));
}

template <class OT> void prepare_batched_got(OT* ot, const MaterialCall& call) {
    require_query(call.query);
    generation_begin(ot);
    if (call.protocol != "iknp_batched_got" || call.message_bits < 1 ||
        call.message_bits > 64 || call.messages_per_ot < 1)
        throw std::runtime_error("invalid prepared batched GOT contract");
    PreparedGOT material;
    material.call = call;
    material.choices.resize(call.count);
    if (ot->party == sci::ALICE) ot->send_pre(call.count);
    else {
        ot->prg.random_data(material.choices.data(), call.count);
        for (auto& choice : material.choices) choice &= 1;
        ot->recv_pre(reinterpret_cast<bool*>(material.choices.data()), call.count);
    }
    const int branches = ot->party == sci::ALICE ? 2 : 1;
    const size_t bit_count = size_t(call.message_bits) * call.messages_per_ot;
    const size_t bytes = (bit_count + 7) / 8;
    const int hashes_per_ot = int((bit_count + 127) / 128);
    material.masks.resize(size_t(call.count) * branches * bytes);
    const int chunk = std::max(1, 256 / (branches * hashes_per_ot));
    auto* hashes = new sci::block128[size_t(chunk) * branches * hashes_per_ot];
    for (int start = 0; start < call.count; start += chunk) {
        const int count = std::min(chunk, call.count - start);
        for (int i = 0; i < count; ++i) for (int branch = 0; branch < branches; ++branch)
            for (int h = 0; h < hashes_per_ot; ++h) {
                auto seed = ot->party == sci::ALICE ? ot->qT[start + i] : ot->tT[start + i];
                if (branch) seed = sci::xorBlocks(seed, ot->block_s);
                hashes[(i * branches + branch) * hashes_per_ot + h] = sci::xorBlocks(seed, sci::toBlock(h));
            }
        ot->crh.Hn(hashes, hashes, count * branches * hashes_per_ot);
        for (int i = 0; i < count; ++i) for (int branch = 0; branch < branches; ++branch)
            std::memcpy(material.masks.data() + (size_t(start + i) * branches + branch) * bytes,
                        hashes + (i * branches + branch) * hashes_per_ot, bytes);
    }
    delete[] hashes;
    if (ot->party == sci::ALICE) { delete[] ot->qT; ot->qT = nullptr; }
    else { delete[] ot->tT; ot->tT = nullptr; }
    random_ot_complete(ot, call.count);
    store_got(ot, std::move(material));
}

template <class OT> bool prepared_batched_got_send(OT* ot, uint64_t* data,
                                                  int count, int bits, int group) {
    if (!strict || phase == Phase::Preparing) return false;
    consume(ot, "iknp_batched_got", count, bits, group);
    auto material = take_got(ot, "iknp_batched_got", count, bits, group);
    std::vector<uint8_t> correction((size_t(count) + 7) / 8);
    ot->io->recv_data(correction.data(), int(correction.size()));
    const size_t bytes = (size_t(bits) * group + 7) / 8;
    std::vector<uint8_t> wire0(size_t(count) * bytes), wire1(size_t(count) * bytes);
    for (int i = 0; i < count; ++i) {
        const int adjust = int(get_bits(correction.data(), i, 1));
        for (int h = 0; h < group; ++h) {
            put_bits(wire0.data() + size_t(i) * bytes, size_t(h) * bits, data[(i * group + h) * 2], bits);
            put_bits(wire1.data() + size_t(i) * bytes, size_t(h) * bits, data[(i * group + h) * 2 + 1], bits);
        }
        for (size_t b = 0; b < bytes; ++b) {
            wire0[size_t(i) * bytes + b] ^= material.masks[(size_t(i) * 2 + adjust) * bytes + b];
            wire1[size_t(i) * bytes + b] ^= material.masks[(size_t(i) * 2 + (1 ^ adjust)) * bytes + b];
        }
    }
    ot->io->send_data(wire0.data(), int(wire0.size()));
    ot->io->send_data(wire1.data(), int(wire1.size()));
    return true;
}

template <class OT> bool prepared_batched_got_recv(OT* ot, uint64_t* data,
                                                  const uint8_t* choices, int count, int bits, int group) {
    if (!strict || phase == Phase::Preparing) return false;
    consume(ot, "iknp_batched_got", count, bits, group);
    auto material = take_got(ot, "iknp_batched_got", count, bits, group);
    std::vector<uint8_t> correction((size_t(count) + 7) / 8);
    for (int i = 0; i < count; ++i)
        put_bits(correction.data(), i, material.choices[i] ^ (choices[i] & 1), 1);
    ot->io->send_data(correction.data(), int(correction.size()));
    const size_t bytes = (size_t(bits) * group + 7) / 8;
    std::vector<uint8_t> wire0(size_t(count) * bytes), wire1(size_t(count) * bytes);
    ot->io->recv_data(wire0.data(), int(wire0.size()));
    ot->io->recv_data(wire1.data(), int(wire1.size()));
    for (int i = 0; i < count; ++i) {
        auto* selected = (choices[i] & 1 ? wire1.data() : wire0.data()) + size_t(i) * bytes;
        for (size_t b = 0; b < bytes; ++b) selected[b] ^= material.masks[size_t(i) * bytes + b];
        for (int h = 0; h < group; ++h)
            data[i * group + h] = get_bits(selected, size_t(h) * bits, bits);
    }
    return true;
}

template <class OT, class T> bool prepared_kkot_send(OT* ot, T** data, int count, int bits) {
    if (!strict || phase == Phase::Preparing) return false;
    consume(ot, "kkot_got", count, bits, ot->N, uint64_t(count) * selection_bits(ot->N));
    auto material = take_got(ot, "kkot_got", count, bits, ot->N);
    const int choice_bits = selection_bits(ot->N);
    std::vector<uint8_t> correction((size_t(count) * choice_bits + 7) / 8);
    ot->io->recv_data(correction.data(), int(correction.size()));
    const size_t wire_bytes = ((size_t(count) * ot->N * bits + sizeof(T) * 8 - 1) /
                               (sizeof(T) * 8)) * sizeof(T);
    std::vector<uint8_t> wire(wire_bytes);
    for (int i = 0; i < count; ++i) {
        const int adjust = int(get_bits(correction.data(), size_t(i) * choice_bits, choice_bits));
        for (int k = 0; k < ot->N; ++k)
            put_bits(wire.data(), (size_t(i) * ot->N + k) * bits,
                uint64_t(data[i][k]) ^ material.words[size_t(i) * ot->N + ((k + adjust) & (ot->N - 1))], bits);
    }
    ot->io->send_data(wire.data(), int(wire.size()));
    return true;
}
template <class OT, class T> bool prepared_kkot_recv(OT* ot, T* data, const uint8_t* choices,
                                                   int count, int bits) {
    if (!strict || phase == Phase::Preparing) return false;
    consume(ot, "kkot_got", count, bits, ot->N, uint64_t(count) * selection_bits(ot->N));
    auto material = take_got(ot, "kkot_got", count, bits, ot->N);
    const int choice_bits = selection_bits(ot->N);
    std::vector<uint8_t> correction((size_t(count) * choice_bits + 7) / 8);
    for (int i = 0; i < count; ++i)
        put_bits(correction.data(), size_t(i) * choice_bits,
                 uint8_t(material.choices[i] - choices[i]) & (ot->N - 1), choice_bits);
    ot->io->send_data(correction.data(), int(correction.size()));
    const size_t wire_bytes = ((size_t(count) * ot->N * bits + sizeof(T) * 8 - 1) /
                               (sizeof(T) * 8)) * sizeof(T);
    std::vector<uint8_t> wire(wire_bytes);
    ot->io->recv_data(wire.data(), int(wire.size()));
    const uint64_t mask = bits == 64 ? ~0ULL : ((1ULL << bits) - 1);
    for (int i = 0; i < count; ++i)
        data[i] = T((get_bits(wire.data(), (size_t(i) * ot->N + choices[i]) * bits, bits) ^ material.words[i]) & mask);
    return true;
}
}
#endif
