/*
 * Description: Refer to README.md
 * Author: Pengzhi Xing
 * Email: p.xing@std.uestc.edu.cn
 * Last Modified: 2024-12-02
 * License: Apache-2.0 License
 * Copyright (c) 2024 Pengzhi Xing
 * Usage:
 * Example:
 *
 * Change Log:
 * 2024-12-02 - Initial version of the authentication module
 */

#include "mpc/secure_ops.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

// server is 2 and client is 3

namespace {

using PackedBits = std::vector<uint64_t>;

int packed_word_count(int bits) {
    return (bits + 63) / 64;
}

uint64_t packed_last_mask(int bits) {
    const int rem = bits % 64;
    return rem == 0 ? ~uint64_t(0) : ((uint64_t(1) << rem) - 1);
}

void mask_packed_bits(PackedBits& value, int bits) {
    if (!value.empty()) {
        value.back() &= packed_last_mask(bits);
    }
}

u8 packed_bit(const PackedBits& value, int index) {
    return static_cast<u8>((value[index / 64] >> (index % 64)) & 1);
}

void set_packed_bit(PackedBits& value, int index, u8 bit) {
    const uint64_t mask = uint64_t(1) << (index % 64);
    if (bit & 1) {
        value[index / 64] |= mask;
    } else {
        value[index / 64] &= ~mask;
    }
}

PackedBits random_packed_bits(osuCrypto::PRNG& rng, int bits) {
    PackedBits output(packed_word_count(bits), 0);
    for (uint64_t& word : output) {
        word = rng.get<uint64_t>();
    }
    mask_packed_bits(output, bits);
    return output;
}

uint64_t pack_msb_bits_as_index(const u8* bits, int bit_count) {
    uint64_t value = 0;
    for (int i = 0; i < bit_count; i++) {
        value = (value << 1) | static_cast<uint64_t>(bits[i] & 1);
    }
    return value;
}

block select_block(u8 bit, const block& value) {
    return (bit & 1) ? value : ZeroBlock;
}

enum class BitBlockMode { legacy, block_ot, direct_ot_legacy, direct_ot, direct_cot };

// The default remains the historical two-COT path.  The environment switch
// is intentionally local to this primitive so callers (including cGGM/DPF)
// do not need a second API or a mismatched material lifecycle.
BitBlockMode bit_block_mode() {
    const char* value = std::getenv("SEGFSS_BIT_BLOCK_MODE");
    if (value == nullptr || std::strcmp(value, "legacy") == 0 ||
        std::strcmp(value, "") == 0) {
        return BitBlockMode::legacy;
    }
    if (std::strcmp(value, "block_ot") == 0) return BitBlockMode::block_ot;
    if (std::strcmp(value, "direct_ot_legacy") == 0) return BitBlockMode::direct_ot_legacy;
    if (std::strcmp(value, "direct_ot") == 0) return BitBlockMode::direct_ot;
    if (std::strcmp(value, "direct_cot") == 0) return BitBlockMode::direct_cot;
    throw std::invalid_argument(
        "SEGFSS_BIT_BLOCK_MODE must be legacy, block_ot, direct_ot_legacy, direct_ot, or direct_cot");
}

// The direct-only ablation preserves two 64-bit chosen-message OT encodings
// per direction. Both limbs use the same normalized choice and fresh mask.
void general_limb_ot(int party_id, const u8* choice, const block* delta,
                     const block* masks, block* received, int32_t size,
                     Peer* player) {
    std::vector<std::array<uint64_t, 2>> low(size), high(size);
    std::vector<uint64_t*> low_ptrs(size), high_ptrs(size);
    std::vector<uint64_t> low_recv(size), high_recv(size);
    std::vector<u8> normalized(size);
    for (int i = 0; i < size; ++i) {
        alignas(16) uint64_t m[2], d[2];
        std::memcpy(m, &masks[i], sizeof(block));
        std::memcpy(d, &delta[i], sizeof(block));
        low[i] = {m[0], m[0] ^ d[0]};
        high[i] = {m[1], m[1] ^ d[1]};
        low_ptrs[i] = low[i].data(); high_ptrs[i] = high[i].data();
        normalized[i] = choice[i] & 1;
    }
    auto send = [&](bool reversed) {
        player->send_ot_u64(reversed, low_ptrs.data(), size, 64);
        player->send_ot_u64(reversed, high_ptrs.data(), size, 64);
    };
    auto receive = [&](bool reversed) {
        player->recv_ot_u64(reversed, low_recv.data(), normalized.data(), size, 64);
        player->recv_ot_u64(reversed, high_recv.data(), normalized.data(), size, 64);
    };
    if (party_id == SERVER) { send(false); receive(true); }
    else if (party_id == CLIENT) { receive(false); send(true); }
    else throw std::invalid_argument("general_limb_ot unsupported party");
    for (int i = 0; i < size; ++i) {
        const uint64_t limbs[2] = {low_recv[i], high_recv[i]};
        std::memcpy(&received[i], limbs, sizeof(block));
    }
}

void general_block_ot(int party_id, const u8* choice,
                      const block* delta, block* sent_mask,
                      block* received, int32_t size, Peer* player) {
    std::vector<block> send0(size), send1(size);
    for (int i = 0; i < size; ++i) {
        send0[i] = sent_mask[i];
        send1[i] = sent_mask[i] ^ delta[i];
    }
    if (party_id == SERVER) {
        player->send_ot_block(false, send0.data(), send1.data(), size);
        player->recv_ot_block(true, received, choice, size);
    } else if (party_id == CLIENT) {
        player->recv_ot_block(false, received, choice, size);
        player->send_ot_block(true, send0.data(), send1.data(), size);
    } else {
        throw std::invalid_argument("general_block_ot unsupported party");
    }
}

// TripleGenerator uses KKOT directly, bypassing Peer::send/recv wrappers.
// Account for the bytes and I/O rounds emitted by the 1-out-of-8 KKOT used
// for Boolean Beaver triples so phase counters include the complete protocol.
struct KkotCounter {
  uint64_t bytes_sent = 0;
  uint64_t bytes_received = 0;
  uint64_t rounds = 0;
};

KkotCounter snapshot_kkot(Peer* player) {
    const auto* io = player->getOTPack()->kkot[2]->io;
    return {io->counter, io->recv_counter, io->num_rounds};
}

void account_kkot(Peer* player, const KkotCounter& before,
                  int num_triples) {
    const KkotCounter after = snapshot_kkot(player);
    (void)num_triples;
    player->bytesSent += after.bytes_sent - before.bytes_sent;
    player->bytesReceived += after.bytes_received - before.bytes_received;
    player->rounds += after.rounds - before.rounds;
}

}  // namespace

[[deprecated("Incorrect realization due to misuse of additive shares and xor shares generated by OT!")]]
void _multiplexer(int party_id, uint8_t *sel, block *dataA, block *output,
                 int32_t size, Peer* player){
    // This mux returns block data with b * x
    block localTemp[size];
    block localShares[size];
    block OTRes[size];
    block AnotherOTRes[size];
    for (int i = 0; i < size; i++){
        localTemp[i] = (*sel == 0 ? ZeroBlock: dataA[i]);
        localShares[i] = dataA[i] - localTemp[i] * 2;
    }
    switch(party_id){
        case 2:{
            for (int i=0; i < size; i++){
                player->send_cot(localShares[i], OTRes + i * sizeof(block), 1);
                player->recv_cot(AnotherOTRes + i * sizeof(block), 1, (bool*)sel);
            }
            break;
        }
        case 3:{
            for (int i=0; i < size; i++){
                player->recv_cot(OTRes + i * sizeof(block), 1, (bool*)sel);
                player->send_cot(localShares[i], AnotherOTRes + i * sizeof(block), 1);
            }
            break;
        }
        default:
            std::cout<<"Unsupported party!"<<std::endl;
    }
    for (int i = 0; i < size; i++){
        output[i] = localTemp[i] + OTRes[i] + AnotherOTRes[i];
    }
}

void multiplexer(int party_id, uint8_t *sel, const block *dataA, block *output,
                  int32_t size, Peer* player){
    // This mux returns block data with b * x
    std::vector<block> localTemp(size);
    std::vector<block> OTRes(size);
    std::vector<block> AnotherOTRes(size);
    std::vector<uint8_t> choice(size);
    for (int i = 0; i < size; i++){
        localTemp[i] = (*sel == 0 ? ZeroBlock: dataA[i]);
        choice[i] = static_cast<uint8_t>(*sel != 0);
    }
    switch(party_id){
        case 2:{
            player->send_cot(dataA, OTRes.data(), size, false);
            player->recv_cot(AnotherOTRes.data(), size, choice.data(), true);
            break;
        }
        case 3:{
            player->recv_cot(OTRes.data(), size, choice.data(), false);
            player->send_cot(dataA, AnotherOTRes.data(), size, true);
            break;
        }
        default:
            std::cout<<"Unsupported party!"<<std::endl;
    }
    for (int i = 0; i < size; i++){
        output[i] = localTemp[i] ^ OTRes[i] ^ AnotherOTRes[i];
    }
}

void multiplexer(int party_id, const std::vector<uint8_t>& sel,
                 const block *dataA, block *output, int32_t size,
                 Peer* player){
    if (static_cast<int32_t>(sel.size()) < size) {
        throw std::invalid_argument("block multiplexer selection vector is too small");
    }

    std::vector<block> localTemp(size);
    std::vector<block> OTRes(size);
    std::vector<block> AnotherOTRes(size);
    std::vector<uint8_t> choice(size);
    for (int i = 0; i < size; i++){
        localTemp[i] = (sel[i] == 0 ? ZeroBlock : dataA[i]);
        choice[i] = static_cast<uint8_t>(sel[i] != 0);
    }

    switch(party_id){
        case 2:{
            player->send_cot(dataA, OTRes.data(), size, false);
            player->recv_cot(AnotherOTRes.data(), size, choice.data(), true);
            break;
        }
        case 3:{
            player->recv_cot(OTRes.data(), size, choice.data(), false);
            player->send_cot(dataA, AnotherOTRes.data(), size, true);
            break;
        }
        default:
            std::cout<<"Unsupported party!"<<std::endl;
    }
    for (int i = 0; i < size; i++){
        output[i] = localTemp[i] ^ OTRes[i] ^ AnotherOTRes[i];
    }
}

void multiplexer(int party_id, uint8_t *sel, const uint64_t *dataA, uint64_t *output,
                 int32_t size, int32_t bw_x, int32_t bw_y, Peer* player){
    // This mux returns block data with b * x
    uint64_t localTemp[size];
    uint64_t localShares[size];
    uint64_t OTRes[size];
    uint64_t x_mask = (1ULL << bw_x) - 1;
    uint64_t y_mask = (1ULL << bw_y) - 1;
    for (int i = 0; i < size; i++){
        localTemp[i] = (*sel == 0 ? (uint64_t)0: dataA[i]);
        localShares[i] = (dataA[i] - localTemp[i] * 2) & x_mask;
    }
    // OT wrapper :
    // void send_cot(uint64_t *data0, uint64_t *corr, int length, int l)
    // void recv_cot(uint64_t *data, bool *b, int length, int l)
    switch(party_id){
        case 2:{
            for (int i=0; i < size; i++){
                player->send_cot(localShares[i], OTRes + i * sizeof(uint64_t), 1);
            }
            break;
        }
        case 3:{
            for (int i=0; i < size; i++){
                player->recv_cot(OTRes + i * sizeof(uint64_t), 1, sel);
            }
            break;
        }
        default:
            std::cout<<"Unsupported party!"<<std::endl;
    }
    for (int i = 0; i < size; i++){
        output[i] = (localTemp[i] + OTRes[i]) & y_mask;
    }
}


void multiplexer(int party_id, uint8_t *sel, const GroupElement *dataA, GroupElement *output,
                 int32_t size, Peer* player){
    GroupElement localTemp[size];
    GroupElement localShares[size];
    GroupElement OTRes[size];
    GroupElement AnotherOTRes[size];
    for (int i = 0; i < size; i++){
        localTemp[i] = (*sel == 0 ? GroupElement(0, dataA[i].bitsize): dataA[i]);
        localShares[i] = dataA[i] - localTemp[i] * 2;
        OTRes[i].bitsize = output[i].bitsize;
        AnotherOTRes[i].bitsize = output[i].bitsize;
    }
    // OT wrapper :
    // void send_cot(uint64_t *data0, uint64_t *corr, int length, int l)
    // void recv_cot(uint64_t *data, bool *b, int length, int l)
    switch(party_id){
        case 2:{
            for (int i=0; i < size; i++){
                player->send_cot(&(localShares[i]), &OTRes[i], 1, false);
                player->recv_cot(&AnotherOTRes[i], 1, sel, true);
            }
            break;
        }
        case 3:{
            for (int i=0; i < size; i++){
                player->recv_cot(&OTRes[i], 1, sel, false);
                player->send_cot(&(localShares[i]), &AnotherOTRes[i], 1, true);
            }
            break;
        }
        default:
            std::cout<<"Unsupported party! Current: "<< party_id <<std::endl;
    }
    for (int i = 0; i < size; i++){
        switch (party_id){
            case 2:{
                output[i] = localTemp[i] - OTRes[i] + AnotherOTRes[i];
                break;
            }
            case 3:{
                output[i] = localTemp[i] + OTRes[i] - AnotherOTRes[i];
            }
        }
    }
}

void multiplexer(int party_id, const std::vector<uint8_t>& sel,
                 const GroupElement *dataA, GroupElement *output,
                 int32_t size, Peer* player){
    if (static_cast<int32_t>(sel.size()) < size) {
        throw std::invalid_argument("multiplexer selection vector is too small");
    }

    std::vector<GroupElement> localTemp(size);
    std::vector<GroupElement> localShares(size);
    std::vector<GroupElement> OTRes(size);
    std::vector<GroupElement> AnotherOTRes(size);
    std::vector<uint8_t> choice(size);
    bool uniformBits = true;
    for (int i = 0; i < size; i++){
        localTemp[i] = (sel[i] == 0 ? GroupElement(0, dataA[i].bitsize): dataA[i]);
        localShares[i] = dataA[i] - localTemp[i] * 2;
        OTRes[i].bitsize = output[i].bitsize;
        AnotherOTRes[i].bitsize = output[i].bitsize;
        choice[i] = static_cast<uint8_t>(sel[i] != 0);
        if (i > 0 &&
            (localShares[i].bitsize != localShares[0].bitsize ||
             output[i].bitsize != output[0].bitsize)) {
            uniformBits = false;
        }
    }

    switch(party_id){
        case 2:{
            if (uniformBits) {
                // GroupElement COT supports vector correlations with one ring
                // size; fall back below only for mixed-width legacy callers.
                player->send_cot(localShares.data(), OTRes.data(), size,
                                 false);
                player->recv_cot(AnotherOTRes.data(), size, choice.data(),
                                 true);
            } else {
                for (int i=0; i < size; i++){
                    player->send_cot(&(localShares[i]), &OTRes[i], 1, false);
                    player->recv_cot(&AnotherOTRes[i], 1, &choice[i], true);
                }
            }
            break;
        }
        case 3:{
            if (uniformBits) {
                player->recv_cot(OTRes.data(), size, choice.data(), false);
                player->send_cot(localShares.data(), AnotherOTRes.data(), size,
                                 true);
            } else {
                for (int i=0; i < size; i++){
                    player->recv_cot(&OTRes[i], 1, &choice[i], false);
                    player->send_cot(&(localShares[i]), &AnotherOTRes[i], 1,
                                     true);
                }
            }
            break;
        }
        default:
            std::cout<<"Unsupported party! Current: "<< party_id <<std::endl;
    }

    for (int i = 0; i < size; i++){
        switch (party_id){
            case 2:{
                output[i] = localTemp[i] - OTRes[i] + AnotherOTRes[i];
                break;
            }
            case 3:{
                output[i] = localTemp[i] + OTRes[i] - AnotherOTRes[i];
            }
        }
    }
}

void multiplexer2(int party_id, uint8_t *sel, const uint64_t *dataA, const uint64_t *dataB, uint64_t *output,
                  int32_t size, int32_t bw_x, int32_t bw_y, Peer* player){
    // MUX2 was composed by:
    // AuxProtocols::multiplexer(input, m0=0, m1=x1)
    // plus
    // AuxProtocols::multiplexer(input^1, m0=0, m1=x0)

    // An update method:
    // if we need a and b, we call MUX @ a-b, and correct with b
    // [bit] [a] [b]
    // -> [a-b]
    // MUX ([bit], [a-b]) -> [output]
    // [output] + [b] -> [real_output]

    uint64_t x_mask = (1ULL << bw_x) - 1;
    uint64_t y_mask = (1ULL << bw_y) - 1;

    uint64_t a_minus_b[size];

    for (int i=0; i<size; i++){
        a_minus_b[i] = (dataA[i] - dataB[i]) & x_mask;
    }

    uint8_t real_sel = *sel ^((u8)party_id-2);

    multiplexer(party_id, &real_sel, a_minus_b, output, size, bw_x, bw_y, player);

    for (int i=0; i<size; i++){
        output[i] += dataB[i];
        output[i] = output[i] & y_mask;
    }
}

void multiplexer2(int party_id, uint8_t* control_bit, const osuCrypto::block* dataA, const osuCrypto::block* dataB,
                  osuCrypto::block* output, int32_t size, Peer* player){

    // if choice bit is 0, it should choose dataA, else dataB

    osuCrypto::block a_minus_b[size];

    for (int i=0; i<size; i++){
        a_minus_b[i] = dataA[i] xor dataB[i];
    }

    uint8_t real_sel = *control_bit ^ ((u8)(party_id-2));
    multiplexer(party_id, &real_sel, a_minus_b, output, size, player);

    for (int i=0; i<size; i++){
        output[i] = output[i] xor dataB[i];
    }
}

void multiplexer2(int party_id, const std::vector<uint8_t>& control_bit,
                  const osuCrypto::block* dataA,
                  const osuCrypto::block* dataB,
                  osuCrypto::block* output, int32_t size, Peer* player){
    if (static_cast<int32_t>(control_bit.size()) < size) {
        throw std::invalid_argument("block multiplexer2 selection vector is too small");
    }

    std::vector<osuCrypto::block> a_minus_b(size);
    std::vector<uint8_t> real_sel(size);
    for (int i=0; i<size; i++){
        a_minus_b[i] = dataA[i] xor dataB[i];
        real_sel[i] = control_bit[i] ^ static_cast<u8>(party_id - 2);
    }

    multiplexer(party_id, real_sel, a_minus_b.data(), output, size, player);

    for (int i=0; i<size; i++){
        output[i] = output[i] xor dataB[i];
    }
}

osuCrypto::block multiplexer2(int party_id, uint8_t control_bit,
                              const osuCrypto::block& dataA,
                              const osuCrypto::block& dataB,
                              Peer* player) {
    osuCrypto::block output;
    multiplexer2(party_id, &control_bit, &dataA, &dataB, &output, 1, player);
    return output;
}

void bitBlockMultiply(int party_id, const u8* bit_share,
                      const osuCrypto::block* block_share,
                      osuCrypto::block* output, int32_t size, Peer* player) {
    if (size <= 0) {
        return;
    }
    if (!player || (party_id != SERVER && party_id != CLIENT))
        throw std::invalid_argument("bitBlockMultiply requires a valid party and peer");
    const BitBlockMode mode = bit_block_mode();
    auto record_success = [&] {
        auto& ledger = player->bit_block_ledger;
        ++ledger.calls; ledger.products += size;
        const bool direct = mode == BitBlockMode::direct_ot || mode == BitBlockMode::direct_ot_legacy ||
                            mode == BitBlockMode::direct_cot;
        if (direct) ledger.direct_products += size;
        else {
            ledger.temporary_triples_generated += size;
            ledger.temporary_triples_consumed += size;
            ledger.ef_openings += 2;
        }
        if (mode == BitBlockMode::legacy || mode == BitBlockMode::direct_ot_legacy)
            ledger.general_ot_64 += uint64_t(4) * size;
        else if (mode == BitBlockMode::direct_cot) ledger.native_cot_128 += uint64_t(2) * size;
        else ledger.general_ot_128 += uint64_t(2) * size;
    };
    if (mode == BitBlockMode::direct_ot || mode == BitBlockMode::direct_ot_legacy ||
        mode == BitBlockMode::direct_cot) {
        // For each party p, OT sends (r_p, r_p XOR b_p) and the peer selects
        // with its local a_p.  If q_p is the selected peer message, then
        // z_p = r_p XOR q_p XOR (a_p ? b_p : 0), and XOR reconstruction is
        // (a0 XOR a1) * (b0 XOR b1).  The input shares remain secret because
        // chosen-message OT hides the receiver choice and masks the sender's
        // two messages with fresh r_p.
        std::vector<block> masks(size), received(size);
        std::vector<u8> choices(size);
        auto& direct_rng = player->randomSource();
        for (int i = 0; i < size; ++i) {
            // Native COT supplies its own fresh random sender base.
            if (mode != BitBlockMode::direct_cot) masks[i] = direct_rng.get<block>();
            choices[i] = static_cast<u8>(bit_share[i] & 1);
        }
        if (mode == BitBlockMode::direct_cot) {
            // Each direction has a separate IKNP instance. The vector-delta
            // overload supports independent block shares within one batch.
            if (party_id == SERVER) {
                player->send_cot_block(false, masks.data(), block_share, size);
                player->recv_cot_block(true, received.data(), choices.data(), size);
            } else {
                player->recv_cot_block(false, received.data(), choices.data(), size);
                player->send_cot_block(true, masks.data(), block_share, size);
            }
        } else if (mode == BitBlockMode::direct_ot_legacy)
            general_limb_ot(party_id, choices.data(), block_share, masks.data(), received.data(), size, player);
        else general_block_ot(party_id, choices.data(), block_share, masks.data(),
                              received.data(), size, player);
        for (int i = 0; i < size; ++i) {
            output[i] = masks[i] ^ received[i] ^
                        select_block(choices[i], block_share[i]);
        }
        record_success();
        return;
    }

    auto& rng = player->randomSource();
    std::vector<u8> a(size);
    std::vector<block> b(size);
    std::vector<block> sent_cross(size);
    std::vector<block> recv_cross(size);
    std::vector<u8> choice(size);
    std::vector<u8> e(size);
    std::vector<block> f(size);
    std::vector<block> c(size);

    for (int i = 0; i < size; i++) {
        a[i] = static_cast<u8>(rng.get<uint8_t>() & 1);
        b[i] = rng.get<block>();
        choice[i] = a[i];
    }

    if (mode == BitBlockMode::block_ot) {
        std::vector<block> masks(size);
        for (int i = 0; i < size; ++i) masks[i] = rng.get<block>();
        general_block_ot(party_id, choice.data(), b.data(), masks.data(),
                         sent_cross.data(), size, player);
        for (int i = 0; i < size; ++i) {
            c[i] = select_block(a[i], b[i]) ^ masks[i] ^ sent_cross[i];
            e[i] = static_cast<u8>((bit_share[i] ^ a[i]) & 1);
            f[i] = block_share[i] ^ b[i];
        }
        player->open_xor(e.data(), size);
        player->open_xor(f.data(), size);
        for (int i = 0; i < size; ++i) {
            output[i] = c[i] ^ select_block(e[i], b[i]) ^
                        select_block(a[i], f[i]);
            if (party_id == SERVER) {
                output[i] = output[i] ^ select_block(e[i], f[i]);
            }
        }
        record_success();
        return;
    }

    switch (party_id) {
        case SERVER: {
            player->send_cot(b.data(), sent_cross.data(), size, false);
            player->recv_cot(recv_cross.data(), size, choice.data(), true);
            break;
        }
        case CLIENT: {
            player->recv_cot(recv_cross.data(), size, choice.data(), false);
            player->send_cot(b.data(), sent_cross.data(), size, true);
            break;
        }
        default:
            throw std::invalid_argument("bitBlockMultiply unsupported party");
    }

    for (int i = 0; i < size; i++) {
        c[i] = select_block(a[i], b[i]) ^ sent_cross[i] ^ recv_cross[i];
        e[i] = static_cast<u8>((bit_share[i] ^ a[i]) & 1);
        f[i] = block_share[i] ^ b[i];
    }

    player->open_xor(e.data(), size);
    player->open_xor(f.data(), size);

    for (int i = 0; i < size; i++) {
        output[i] = c[i] ^ select_block(e[i], b[i]) ^
                    select_block(a[i], f[i]);
        if (party_id == SERVER) {
            output[i] = output[i] ^ select_block(e[i], f[i]);
        }
    }
    record_success();
}

DpfBitBlockMulMaterial prepareDpfBitBlockMulMaterial(int party_id,
                                                     int32_t size,
                                                     Peer* player) {
    DpfBitBlockMulMaterial material;
    if (size <= 0) {
        return material;
    }

    auto rng = secure_prng();
    material.a.resize(size);
    material.b.resize(size);
    material.c.resize(size);

    std::vector<block> send0(size);
    std::vector<block> send1(size);
    std::vector<block> recv(size);
    std::vector<uint8_t> choice(size);

    for (int i = 0; i < size; i++) {
        material.a[i] = static_cast<u8>(rng.get<uint8_t>() & 1);
        choice[i] = material.a[i];
        send0[i] = rng.get<block>();
        send1[i] = rng.get<block>();
        material.b[i] = send0[i] ^ send1[i];
    }

    switch (party_id) {
        case SERVER: {
            player->send_ot_block(false, send0.data(), send1.data(), size);
            player->recv_ot_block(true, recv.data(), choice.data(), size);
            break;
        }
        case CLIENT: {
            player->recv_ot_block(false, recv.data(), choice.data(), size);
            player->send_ot_block(true, send0.data(), send1.data(), size);
            break;
        }
        default:
            throw std::invalid_argument(
                "prepareDpfBitBlockMulMaterial unsupported party");
    }

    for (int i = 0; i < size; i++) {
        material.c[i] = recv[i] ^ send0[i] ^
                        select_block(material.a[i], material.b[i]);
    }
    return material;
}

void consumeDpfBitBlockMulMaterial(int party_id,
                                   DpfBitBlockMulMaterial& material,
                                   const u8* bit_share,
                                   const osuCrypto::block* block_share,
                                   osuCrypto::block* output,
                                   int32_t size, Peer* player) {
    if (size <= 0) {
        return;
    }
    if (material.cursor < 0 ||
        material.cursor + size > static_cast<int32_t>(material.a.size())) {
        throw std::invalid_argument(
            "consumeDpfBitBlockMulMaterial exceeds prepared material");
    }

    std::vector<u8> phi(size);
    std::vector<block> theta(size);
    for (int i = 0; i < size; i++) {
        const int idx = material.cursor + i;
        phi[i] = static_cast<u8>((bit_share[i] ^ material.a[idx]) & 1);
        theta[i] = block_share[i] ^ material.b[idx];
    }

    reconstruct(size, phi.data());
    reconstruct(size, theta.data());

    for (int i = 0; i < size; i++) {
        const int idx = material.cursor + i;
        output[i] = material.c[idx] ^
                    select_block(phi[i], material.b[idx]) ^
                    select_block(material.a[idx], theta[i]);
        if (party_id == SERVER) {
            output[i] = output[i] ^ select_block(phi[i], theta[i]);
        }
    }
    material.cursor += size;
}

void multiplexer2(int party_id, uint8_t *control_bit, const GroupElement* dataA, const GroupElement* dataB,
                  GroupElement* output, int32_t size, Peer* player){
    if (size == 1) {
        GroupElement a_minus_b = dataA[0] - dataB[0];
        uint8_t real_sel = control_bit[0] ^ static_cast<u8>(party_id - 2);
        multiplexer(party_id, &real_sel, &a_minus_b, output, 1, player);
        output[0] = output[0] + dataB[0];
        return;
    }

    std::vector<GroupElement> a_minus_b(size);
    std::vector<uint8_t> real_sel(size);

    for (int i=0; i<size; i++){
        a_minus_b[i] = dataA[i] - dataB[i];
        real_sel[i] = control_bit[i] ^((u8)party_id-2);
    }

    multiplexer(party_id, real_sel, a_minus_b.data(), output, size, player);

    for (int i=0; i<size; i++){
        output[i] = output[i] + dataB[i];
    }
}

GroupElement multiplexer2(int party_id, uint8_t control_bit,
                          const GroupElement& dataA,
                          const GroupElement& dataB,
                          Peer* player) {
    GroupElement output(0, dataA.bitsize);
    multiplexer2(party_id, &control_bit, &dataA, &dataB, &output, 1, player);
    return output;
}

void insecure_multiplexer2(int party_id, uint8_t *control_bit, const GroupElement* dataA, const GroupElement* dataB,
                  GroupElement* output, int32_t size, Peer* player){
    switch (party_id) {
        case 2:{
            player->send_u8(*control_bit);
            for (int i = 0; i < size; i++){
                output[i] = GroupElement(8888, output->bitsize);
                player->send_u64(dataA->value);
                player->send_u64(dataB->value);
            }
            break;
        }
        case 3:{
            u8 _ctrl = player->recv_u8();
            _ctrl = *control_bit ^ _ctrl;
            uint64_t recvA[size];
            uint64_t recvB[size];
            for (int i = 0; i < size; i++){
                recvA[i] = player->recv_u64();
                recvB[i] = player->recv_u64();
            }
            if (_ctrl == (u8)0){
                for (int i = 0; i < size; i++){
                    output[i] = dataA[i] - 8888 + recvA[i];
                }
            }else{
                for (int i = 0; i < size; i++){
                    output[i] = dataB[i] - 8888 + recvB[i];
                }
            }
            break;
        }
    }
}

void insecure_multiplexer(int party_id, uint8_t *control_bit, const GroupElement* dataA,
                          GroupElement* output, int32_t size, Peer* player){
    switch (party_id) {
        case 2:{
            for (int i = 0; i < size; i++){
                player->send_u8(control_bit[i]);
                output[i] = GroupElement(8888, output[i].bitsize);
                player->send_u64(dataA->value);
            }
            break;
        }
        case 3:{
            u8 _ctrl[size];
            uint64_t recvA[size];
            for (int i = 0; i < size; i++){
                _ctrl[i] = player->recv_u8();
                _ctrl[i] = control_bit[i] ^ _ctrl[i];
                recvA[i] = player->recv_u64();
            }
            for (int i = 0; i < size; i++) {
                if (_ctrl[i] == (u8) 1) {
                    output[i] = dataA[i] - 8888 + recvA[i];
                }else{
                    output[i] = GroupElement(-8888, output[i].bitsize);
                }
            }
            break;
        }
    }
}

u8 and_wrapper(int party_id, u8 dataA, u8 dataB, Peer* player){
    // first of all, we should generate triplets
    // initialize bit triples
    Triple and_triple(2);
    // generate triples
    const KkotCounter kkot_before = snapshot_kkot(player);
    TripleGenerator generator(party_id-1, player->getIOPack(), player->getOTPack());
    generator.generate(party_id-1, and_triple.ai, and_triple.bi, and_triple.ci,
                       and_triple.num_triples, _8KKOT);
    account_kkot(player, kkot_before, and_triple.num_triples);


    // construct e and f in ABY
    u8 ef[2];
    u8 e = dataA xor and_triple.ai[0];
    u8 f = dataB xor and_triple.bi[0];
    ef[0] = e;
    ef[1] = f;


    // send them to another party
    u8 _ef[2];
    player->send_u8(ef, 2);
    player->recv_u8(_ef, 2);
    u8 _e = _ef[0];
    u8 _f = _ef[1];

    e = e xor _e;
    f = f xor _f;

    // return results
    // here we deduct the party id to 0 and 1
    u8 res = ((u8)(party_id - 2) * e * f) xor (f * dataA) xor (e * dataB) xor and_triple.ci[0];
    return res;
}

void and_wrapper(int party_id, const u8* dataA, const u8* dataB, u8* output, int size, Peer* player){
    // first of all, we should generate triplets
    // initialize bit triples
    Triple and_triple(size + (size % 2) * 1);
    // generate triples
    const KkotCounter kkot_before = snapshot_kkot(player);
    TripleGenerator generator(party_id-1, player->getIOPack(), player->getOTPack());
    generator.generate(party_id-1, and_triple.ai, and_triple.bi, and_triple.ci,
                       and_triple.num_triples, _8KKOT);
    account_kkot(player, kkot_before, and_triple.num_triples);


    // construct e and f in ABY
    u8 e[size];
    u8 f[size];
    u8 ef[2 * size];
    for (int i = 0; i < size; i++){
        e[i] = dataA[i] xor and_triple.ai[i];
        f[i] = dataB[i] xor and_triple.bi[i];
        ef[i] = e[i];
        ef[i + size] = f[i];
    }

    // send them to another party
    u8 _e[size];
    u8 _f[size];
    u8 _ef[2 * size];
    player->send_u8(ef, 2 * size);
    player->recv_u8(_ef, 2 * size);

    for (int i = 0; i < size; i++){
        _e[i] = _ef[i];
        _f[i] = _ef[i + size];
        e[i] = e[i] xor _e[i];
        f[i] = f[i] xor _f[i];
        output[i] = ((u8)(party_id - 2) * e[i] * f[i]) xor (f[i] * dataA[i]) xor (e[i] * dataB[i]) xor and_triple.ci[i];
    }
    return;
}

[[deprecated("Incorrect realization due to misuse of additive shares and xor shares generated by OT!")]]
u8 _and_wrapper(int party_id, u8 data, Peer* player){
    // Single input AND gate is realized by one single call to COT
    uint64_t out;
    switch (party_id){
        case 2:{
            player->send_cot((uint64_t)data, &out, 1);
            break;
        }
        case 3:{
            player->recv_cot(&out, 1, &data);
            break;
        }
        default:{
            std::cout<<"Unsupported party! Current: "<< party_id <<std::endl;
        }
    }
    return (u8)out;
}

u8 and_wrapper(int party_id, u8 data, Peer* player){
    // Realized by call to shared-style input wrapper
    u8 out;

    switch (party_id){
        case 2: {
            out = and_wrapper(party_id, data, (u8) 0, player);
            break;
        }
        case 3: {
            out = and_wrapper(party_id, (u8) 0, data, player);
            break;
        }
        default:{
            std::cout<<"Unsupported party! Current: "<< party_id <<std::endl;
        }
    }
    return out;
}

u8 or_wrapper(int party_id, u8 dataA, u8 dataB, Peer* player){
    // An OR operation is realized by:
    // X or B = 1 xor [(1 xor X) and (1 xor B)]
    u8 one_shares = (u8)(party_id - 2);
    u8 out = one_shares;
    dataA = dataA xor one_shares;
    dataB = dataB xor one_shares;
    out = out xor (and_wrapper(party_id, dataA, dataB, player));
    return out;
}

void or_wrapper(int party_id, const u8* dataA, const u8* dataB, u8* output, int size, Peer* player){
    // An OR operation is realized by:
    // X or B = 1 xor [(1 xor X) and (1 xor B)]
    u8 one_shares = (u8)(party_id - 2);
    u8 _A[size];
    u8 _B[size];
    u8 and_output[size];
    for (int i = 0; i < size; i++){
        output[i] = one_shares;
        _A[i] = dataA[i] xor one_shares;
        _B[i] = dataB[i] xor one_shares;
        and_wrapper(party_id, _A, _B, and_output, size, player);
        output[i] = output[i] xor and_output[i];
    }
    return;
}

u8 or_wrapper(int party_id, u8 data, Peer* player){
    u8 out;
    switch (party_id) {
        case 2:{
            out = or_wrapper(party_id, data, (u8)0, player);
            break;
        }
        case 3:{
            out = or_wrapper(party_id, (u8)0, data, player);
            break;
        }
        default:{
            std::cout<<"Unsupported party! Current: "<< party_id <<std::endl;
        }
    }
    return out;
}

u8 cmp_2bit(int party_id, u8 a, u8 b, Peer* player){
    // The inputs are not shares!
    // a is high order bits
    u8 out;

    u8 notAnotB = (1 xor a) and (1 xor b);
    u8 notAB = (1 xor a) and b;
    u8 AnotB = a and (1 xor b);
    u8 AB = a and b;

    switch (party_id) {
        case 2:{
            u8 out_1 = and_wrapper(party_id, notAnotB, player);
            u8 out_2 = and_wrapper(party_id, notAB, player);
            u8 out_3 = and_wrapper(party_id, AnotB, player);
            u8 out_4 = and_wrapper(party_id, AB, player);
            u8 out_5 = or_wrapper(party_id, out_1, out_2, player);
            u8 out_6 = or_wrapper(party_id, out_5, out_3, player);
            out = or_wrapper(party_id, out_6, out_4, player);
            break;
        }
        case 3:{
            u8 out_1 = and_wrapper(party_id, notAB, player);
            u8 out_2 = and_wrapper(party_id, AnotB, player);
            u8 out_3 = and_wrapper(party_id, AB, player);
            u8 out_4 = and_wrapper(party_id, notAnotB, player);
            u8 out_5 = or_wrapper(party_id, out_1, out_2, player);
            u8 out_6 = or_wrapper(party_id, out_5, out_3, player);
            out = or_wrapper(party_id, out_6, out_4, player);
            break;
        }
    }

    return out;
}

u8 cmp_2bit_opt(int party_id, u8 a, u8 b, Peer* player){
    // a is the higher bits
    u8 out = 0;
    u8 and_share_0, and_share_1;
    and_share_0 = a;
    switch (party_id) {
        case 2:{
            and_share_1 = a xor b xor 1;
            break;
        }
        case 3:{
            out = b;
            and_share_1 = a xor b;
        }
    }
    out = out xor and_wrapper(party_id, and_share_0, and_share_1, player);
    return out;
}

void cmp_2bit_opt(int party_id, const u8* a, const u8* b, u8* output, int size, Peer* player){
    u8 and_share_0[size], and_share_1[size];
    for (int i = 0; i < size; i++){
        and_share_0[i] = a[i];
        switch (party_id) {
            case 2:{
                output[i] = 0;
                and_share_1[i] = a[i] xor b[i] xor 1;
                break;
            }
            case 3:{
                output[i] = b[i];
                and_share_1[i] = a[i] xor b[i];
            }
        }
    }
    u8 and_output[size];
    and_wrapper(party_id, and_share_0, and_share_1, and_output, size, player);
    for (int i = 0; i < size; i++){
        output[i] = output[i] xor and_output[i];
    }
    return;
}

u8 check_bit_overflow(int party_id, u8 x_share, u8 r_prev_share, Peer* player){
    u8 front = and_wrapper(party_id, x_share, player);
    u8 rare = and_wrapper(party_id, x_share, r_prev_share, player);
    return or_wrapper(party_id, front, rare, player);
}

std::vector<u8> BitDec(int party_id, GroupElement input, int bits,
                       Peer* player) {
    std::vector<u8> output(bits);
    u8 carry_share = 0;
    // Convert additive shares to MSB-first Boolean shares of the represented
    // value. DPF traversal consumes the path from MSB to LSB.
    for (int i = 0; i < bits; i++) {
        const u8 x_share = input[bits - i - 1];
        output[bits - i - 1] = x_share ^ carry_share;

        u8 and_left[2] = {0, x_share};
        u8 and_right[2] = {0, carry_share};
        if (party_id == SERVER) {
            and_left[0] = x_share;
        } else if (party_id == CLIENT) {
            and_right[0] = x_share;
        } else {
            throw std::invalid_argument("BitDec unsupported party");
        }
        u8 and_output[2] = {0, 0};
        and_wrapper(party_id, and_left, and_right, and_output, 2, player);
        carry_share = and_output[0] ^ and_output[1];
    }
    return output;
}

void booleanOneHotFromBits(int party_id, const u8* eta_bits, int bits,
                           u8* output, Peer* player) {
    if (bits < 0 || bits >= 31) {
        throw std::invalid_argument(
            "booleanOneHotFromBits requires 0 <= bits < 31");
    }
    const int length = 1 << bits;
    const int word_count = packed_word_count(length);
    const uint64_t eta_share = pack_msb_bits_as_index(eta_bits, bits);
    auto rng = secure_prng();

    if (party_id == SERVER) {
        // SERVER plays the ROT sender and later derives the row/column parity
        // shares R and C from the selected matrix rows.
        std::vector<PackedBits> r0(length, PackedBits(word_count, 0));
        std::vector<PackedBits> r1(length, PackedBits(word_count, 0));
        std::vector<PackedBits> m(length, PackedBits(word_count, 0));
        for (int i = 0; i < length; i++) {
            PackedBits delta = random_packed_bits(rng, length);
            for (int word = 0; word < word_count; word++) {
                block r0_block;
                player->send_cot(osuCrypto::toBlock(0, delta[word]),
                                 &r0_block, 1, false);
                r0[i][word] =
                    static_cast<uint64_t>(_mm_cvtsi128_si64x(r0_block));
                r1[i][word] = r0[i][word] ^ delta[word];
            }
            mask_packed_bits(r0[i], length);
            mask_packed_bits(r1[i], length);
        }

        std::vector<u8> v(length);
        player->recv_u8(v.data(), length);
        for (int i = 0; i < length; i++) {
            m[i] = (v[i] & 1) ? r1[i] : r0[i];
        }

        PackedBits r_out(word_count, 0);
        PackedBits c_out(word_count, 0);
        for (int i = 0; i < length; i++) {
            u8 r_bit = 0;
            u8 c_bit = 0;
            for (int j = 0; j < length; j++) {
                r_bit ^= packed_bit(m[j ^ i], i);
                c_bit ^= packed_bit(m[i ^ j], j);
            }
            set_packed_bit(r_out, i, r_bit);
            set_packed_bit(c_out, i, c_bit);
        }

        PackedBits s = c_out;
        set_packed_bit(s, static_cast<int>(eta_share),
                       packed_bit(s, static_cast<int>(eta_share)) ^ 1);
        // Send S = e_{eta_0} xor C. CLIENT shifts it by eta_1 and combines it
        // with W.
        player->send_u64(s.data(), word_count);
        for (int i = 0; i < length; i++) {
            output[i] = packed_bit(r_out, i);
        }
        return;
    }

    if (party_id == CLIENT) {
        // CLIENT receives one ROT row per index, masks its index share into v,
        // then computes the W correction vector.
        std::vector<u8> beta(length);
        std::vector<PackedBits> m(length, PackedBits(word_count, 0));
        for (int i = 0; i < length; i++) {
            beta[i] = static_cast<u8>(rng.get<uint8_t>() & 1);
            for (int word = 0; word < word_count; word++) {
                block received;
                bool choice = beta[i] != 0;
                player->recv_cot(&received, 1, &choice, false);
                m[i][word] =
                    static_cast<uint64_t>(_mm_cvtsi128_si64x(received));
            }
            mask_packed_bits(m[i], length);
        }

        std::vector<u8> v(length);
        for (int i = 0; i < length; i++) {
            v[i] = beta[i] ^ static_cast<u8>(i == eta_share);
        }
        player->send_u8(v.data(), length);

        PackedBits w_out(word_count, 0);
        for (int i = 0; i < length; i++) {
            const uint64_t theta = eta_share ^ static_cast<uint64_t>(i);
            u8 w_bit = 0;
            for (int j = 0; j < length; j++) {
                if (static_cast<uint64_t>(j) != theta) {
                    w_bit ^= packed_bit(m[j ^ i], i);
                }
                if (j != i) {
                    w_bit ^= packed_bit(m[theta ^ static_cast<uint64_t>(j)], j);
                }
            }
            set_packed_bit(w_out, i, w_bit);
        }

        PackedBits s(word_count, 0);
        player->recv_u64(s.data(), word_count);
        mask_packed_bits(s, length);
        for (int i = 0; i < length; i++) {
            output[i] = packed_bit(s, i ^ static_cast<int>(eta_share)) ^
                        packed_bit(w_out, i);
        }
        return;
    }

    throw std::invalid_argument("booleanOneHotFromBits unsupported party");
}

GroupElement cross_term_gen(int party_id, const GroupElement* input, bool hold_arithmetic, Peer* player){
    // This function uses COT for cross term generation
    // TODO: Add batched operation
    GroupElement cot_output(0, input->bitsize);
    GroupElement output = GroupElement(0, input->bitsize);
    if(hold_arithmetic){
        // Act as sender, send 2^i * a
        for (int i = 0; i < input->bitsize; i++){
            uint64_t constant = 1ULL << i;
            GroupElement data_to_send = *input * constant;
            player->send_cot(&data_to_send, &cot_output, 1, false);
            output = output - cot_output;
        }
    }
    else {
        for (int i = 0; i < input->bitsize; i++) {
            // Act as receiver, use lower bit as the choice bit
            u8 choice_bit = (*input)[input->bitsize - 1 - i];
            player->recv_cot(&cot_output, 1, &choice_bit, false);
            output = output + cot_output;
        }
    }
    return output;
}

namespace {

// Gilboa cross terms with one full-width COT per (element, bit), index
// i * bitsize + j.  Used for batches that fit the precomputed OT pool, where
// the pooled IKNP columns are amortized across calls.
void cross_term_gen_pooled(const GroupElement* input, GroupElement* output,
                           bool hold_arithmetic, int size, Peer* player){
    int uniform_bitsize = input[0].bitsize;
    int iteration_size = input[0].bitsize * size;

    if (hold_arithmetic){
        std::vector<GroupElement> cot_output(iteration_size);
        for (int i = 0; i < iteration_size; i++){
            cot_output[i].bitsize = input[0].bitsize;
        }
        uint64_t constant;
        std::vector<GroupElement> data_to_send(iteration_size);
        for (int i = 0; i < size; i++){
            for (int j = 0; j < input[i].bitsize; j++){
                constant = 1ULL << j;
                data_to_send[i * uniform_bitsize + j] = input[i] * constant;
                data_to_send[i * uniform_bitsize + j].bitsize = input[0].bitsize;
            }
        }
        // Call one round COT
        player->send_cot(data_to_send.data(), cot_output.data(), iteration_size, false);
        for (int i = 0; i < size; i++){
            for (int j = 0; j < uniform_bitsize; j++){
                output[i] = output[i] - cot_output[j + i * uniform_bitsize];
            }
        }
    }else{
        std::vector<GroupElement> cot_output(iteration_size);
        for (int i = 0; i < iteration_size; i++){
            cot_output[i].bitsize = input[0].bitsize;
        }
        std::vector<u8> choice_bit(iteration_size);
        for (int i = 0; i < size; i++){
            for(int j = 0; j < input[i].bitsize; j++){
                choice_bit[i * uniform_bitsize + j] = input[i][uniform_bitsize - 1 - j];
            }
        }
        // Call One round COT
        player->recv_cot(cot_output.data(), iteration_size, choice_bit.data(), false);
        for (int i = 0; i < size; i++){
            for (int j = 0; j < uniform_bitsize; j++){
                output[i] = output[i] + cot_output[j + i * uniform_bitsize];
            }
        }
    }
}

// Length-reduced Gilboa cross terms (SCI matmul_cross_terms pattern).  The
// j-th product bit is scaled by 2^j, so its COT only needs to be correct mod
// 2^(bitsize - j): the correlation is x mod 2^(bitsize - j) and the result is
// shifted left by j.  OTs are bit-major (index j * size + i) so every group
// shares one message length.  The sender then transmits
// sum_j ceil((bitsize - j) / 8) bytes per element instead of two bitsize-wide
// messages per OT.
void cross_term_gen_length_reduced(const GroupElement* input, GroupElement* output,
                                   bool hold_arithmetic, int size, Peer* player){
    const int bitsize = input[0].bitsize;
    const int num_ot = bitsize * size;
    std::vector<int> msg_len(bitsize);
    for (int j = 0; j < bitsize; j++){
        msg_len[j] = bitsize - j;
    }
    std::vector<uint64_t> cot(num_ot);
    if (hold_arithmetic){
        std::vector<uint64_t> corr(num_ot);
        for (int j = 0; j < bitsize; j++){
            const uint64_t low_mask = GroupElement(~uint64_t(0), bitsize - j).value;
            for (int i = 0; i < size; i++){
                corr[j * size + i] = input[i].value & low_mask;
            }
        }
        player->send_batched_cot_u64(false, cot.data(), corr.data(), msg_len, num_ot);
    } else {
        std::unique_ptr<bool[]> choice(new bool[num_ot]);
        for (int j = 0; j < bitsize; j++){
            for (int i = 0; i < size; i++){
                choice[j * size + i] = ((input[i].value >> j) & 1) != 0;
            }
        }
        player->recv_batched_cot_u64(false, cot.data(), choice.get(), msg_len, num_ot);
    }
    // Wrap-around in uint64_t is harmless: GroupElement reduces mod 2^bitsize.
    std::vector<uint64_t> acc(size, 0);
    for (int j = 0; j < bitsize; j++){
        for (int i = 0; i < size; i++){
            acc[i] += cot[j * size + i] << j;
        }
    }
    for (int i = 0; i < size; i++){
        const GroupElement term(acc[i], bitsize);
        output[i] = hold_arithmetic ? output[i] - term : output[i] + term;
    }
}

}  // namespace

void cross_term_gen(int party_id, const GroupElement* input, GroupElement* output, bool hold_arithmetic, int size, Peer* player){
    // This is the batched operation of cross term generation.
    // We assume that only one party holds the arithmetic with uniform bit length.
    (void)party_id;
    if (size <= 0){
        return;
    }
    // Both parties see the same public (bitsize, size) and pool size, so they
    // pick the same path.  Batches within the pool reuse its precomputed IKNP
    // columns; larger ones pay the same IKNP columns either way, so the
    // length-reduced COT only shrinks the sender payload.
    const int num_ot = input[0].bitsize * size;
    if (num_ot <= player->ot_precompute_batch_size()){
        cross_term_gen_pooled(input, output, hold_arithmetic, size, player);
    } else {
        cross_term_gen_length_reduced(input, output, hold_arithmetic, size, player);
    }
}

void beaver_mult_offline(int party_id, GroupElement* a, GroupElement* b, GroupElement* c, Peer* player, int size){
    // This is the offline phase of beaver multiplication
    // Here we have to invoke OT for generating cross-term c
    // we also have to call rpg
    // This function is optimized with batch operation
    auto& rng = player->randomSource();
    std::vector<GroupElement> OTshareC(2 * size);
    std::vector<GroupElement> OTinputAB(2 * size);
    std::vector<GroupElement> localShareC(size);
    for (int i = 0; i < size; i++) {
        a[i] = random_ge_from_prng(rng, a[i].bitsize);
        b[i] = random_ge_from_prng(rng, b[i].bitsize);
        auto a_value = a[i].value;
        auto b_value = b[i].value;

        // Invoke OT for cross term generation
        localShareC[i] = GroupElement(a_value * b_value, c[i].bitsize);
        OTshareC[2 * i] = GroupElement(0, c[i].bitsize);
        OTshareC[2 * i + 1] = GroupElement(0, c[i].bitsize);
        if (party_id == 2){
            OTinputAB[i] = a[i];
            OTinputAB[i].bitsize = a[i].bitsize;
            OTinputAB[i + size] = b[i];
            OTinputAB[i + size].bitsize = b[i].bitsize;
        }else{
            OTinputAB[i] = b[i];
            OTinputAB[i].bitsize = b[i].bitsize;
            OTinputAB[i + size] = a[i];
            OTinputAB[i + size].bitsize = a[i].bitsize;
        }
    }
    // Then we have to call 2 COTs
    switch (party_id){
        case 2:{
            cross_term_gen(party_id, OTinputAB.data(), OTshareC.data(), true, 2 * size, player);
            break;
        }
        case 3:{
            cross_term_gen(party_id, OTinputAB.data(), OTshareC.data(), false, 2 * size, player);
            break;
        }
        default:{
            std::cout<<"Unsupported party! Current: "<< party_id <<std::endl;
        }
    }
    for (int i = 0; i < size; i++){
        c[i] = localShareC[i] + OTshareC[i] + OTshareC[i + size];
    }
    player->preprocessing_ledger.arithmetic_triples += size;
}

void beaver_mult_online(int party_id, GroupElement input0, GroupElement input1,
                        GroupElement a, GroupElement b, GroupElement c,
                        GroupElement* output, Peer* player){
    *output = beaver_mult_online(party_id, input0, input1, a, b, c, player);
}

GroupElement beaver_mult_online(int party_id, GroupElement input0, GroupElement input1,
                                GroupElement a, GroupElement b, GroupElement c,
                                Peer* player){
    GroupElement input0_new(0, input0.bitsize);
    GroupElement input1_new(0, input1.bitsize);
    input0_new = input0 - a;
    input1_new = input1 - b;
    player->open_additive(&input0_new, 1, input0_new.bitsize);
    player->open_additive(&input1_new, 1, input1_new.bitsize);
    // Assume party_id \in {2, 3}
    return c + b * input0_new + a * input1_new + input1_new * input0_new * (party_id - 2);
}

void beaver_mult_online(int party_id, GroupElement* input0, GroupElement* input1,
                        const GroupElement* a, const GroupElement* b, const GroupElement* c,
                        GroupElement* output, int size, Peer* player){
    // This is the batched output of beaver multiplication
    for (int i = 0; i < size; i++){
        input0[i] = input0[i] - a[i];
        input1[i] = input1[i] - b[i];
    }
    // Here we call constant time reconstruction
    player->open_additive(input0, size, input0[0].bitsize);
    player->open_additive(input1, size, input1[0].bitsize);

    for (int i = 0; i < size; i++){
        output[i] = c[i] + b[i] * input0[i] + a[i] * input1[i] + input1[i] * input0[i] * (party_id - 2);
    }
}

void B2A(int party_id, u8* x, GroupElement* y, int size, int bw_y, Peer* player){
    // follows the implementation from CryptFlow EzPC SCI
    assert(bw_y <= 64 && bw_y >= 1);
    player->preprocessing_ledger.b2a_values += size;
    if (bw_y == 1) {
        for (int i = 0; i < size; i++) {
            y[i].value = uint64_t(x[i]) & 1;
        }
        return;
    }
    uint64_t mask = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

    if ((party_id - 2) == 0) {
        std::vector<GroupElement> corr_data(size);
        for (int i = 0; i < size; i++) {
            corr_data[i].bitsize = bw_y;
            corr_data[i].value = (-2 * uint64_t(x[i])) & mask;
            mod(corr_data[i]);
        }
        player->send_cot(corr_data.data(), y, size, false);

        for (int i = 0; i < size; i++) {
            y[i].value = (uint64_t(x[i]) - y[i].value) & mask;
            mod(y[i]);
        }
    } else { // party == sci::BOB
        player->recv_cot(y, size, x, false);

        for (int i = 0; i < size; i++) {
            y[i].value = (uint64_t(x[i]) + y[i].value) & mask;
            mod(y[i]);
        }
    }
}

GroupElement B2A(int party_id, u8 x, int bw_y, Peer* player) {
    GroupElement y(0, bw_y);
    B2A(party_id, &x, &y, 1, bw_y, player);
    return y;
}
