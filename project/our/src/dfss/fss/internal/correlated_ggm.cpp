#include "fss/internal/correlated_ggm.h"

#include <stdexcept>
#include <string>
#include <array>
#include <cstring>

#include <cryptoTools/Crypto/AES.h>

#include "mpc/api.h"
#include "fss/internal/ggm.h"
#include "fss/internal/payload_conversion.h"
#include "mpc/secure_ops.h"

namespace dfss::internal {

namespace {

BooleanElement blockLsb(const osuCrypto::block& input) {
    return _mm_cvtsi128_si64x(input) & 1;
}

template <typename KeyT>
osuCrypto::block evalLeaf(const BooleanElement* opened_bits, const KeyT& key,
                          BooleanElement* controlBitOut) {
    osuCrypto::block levelNode = key.k[0];
    BooleanElement controlBit = blockLsb(levelNode);
    for (int i = 0; i < key.Bin - 1; i++) {
        const BooleanElement direction =
            static_cast<BooleanElement>(opened_bits[i] & 1);
        const osuCrypto::block current = levelNode;
        const osuCrypto::block hashed = correlatedHash(current, key.hash_key);
        levelNode = hashed ^
                    (direction ? current : osuCrypto::ZeroBlock) ^
                    (controlBit ? key.k[i + 1] : osuCrypto::ZeroBlock);
        controlBit = blockLsb(levelNode);
    }

    const BooleanElement finalDirection =
        static_cast<BooleanElement>(opened_bits[key.Bin - 1] & 1);
    const osuCrypto::block q =
        correlatedHash(levelNode ^
                (finalDirection ? osuCrypto::OneBlock
                                : osuCrypto::ZeroBlock), key.hash_key);
    const osuCrypto::block finalCw =
        setBlockLsb(key.k[key.Bin],
                    key.v[2 * (key.Bin - 1) + finalDirection]);
    levelNode = q ^ (controlBit ? finalCw : osuCrypto::ZeroBlock);
    controlBit = blockLsb(levelNode);
    if (controlBitOut != nullptr) {
        *controlBitOut = controlBit;
    }
    return levelNode;
}

}  // namespace

namespace {

constexpr int kMaxFullTreeBits = 24;

void ensureSupportedFullTreeBits(const int bits, const char* caller) {
    if (bits < 0 || bits > kMaxFullTreeBits) {
        throw std::invalid_argument(
            std::string(caller) + " requires 0 <= bit length <= " +
            std::to_string(kMaxFullTreeBits) +
            " in the current full-tree implementation");
    }
}

void ensureParty(int party_id) {
    if (party_id != SERVER && party_id != CLIENT) {
        throw std::invalid_argument("Figure 10 COT requires SERVER or CLIENT");
    }
}

}  // namespace

osuCrypto::block correlatedHash(const osuCrypto::block& input,
                                const osuCrypto::block& hash_key) {
    return ccrHash(input ^ hash_key);
}

Figure10CotSession initializeFigure10CotSession(int party_id, Peer* player) {
    ensureParty(party_id);
    if (player == nullptr) {
        throw std::invalid_argument("Figure 10 COT requires a peer");
    }
    const int b = party_id - SERVER;
    auto rng = secure_prng();
    Figure10CotSession session;
    session.delta_prime = rng.get<osuCrypto::block>();
    session.leaf_delta = rng.get<osuCrypto::block>();
    const BooleanElement own_lsb = blockLsb(session.delta_prime);
    player->send_u8(own_lsb);
    const BooleanElement remote_lsb = player->recv_u8();
    const BooleanElement local_correction = remote_lsb ^ b;
    session.delta_share = session.delta_prime ^
        (local_correction ? osuCrypto::OneBlock : osuCrypto::ZeroBlock);
    session.receiver_delta_correction =
        (own_lsb ^ (1 - b)) ? osuCrypto::OneBlock : osuCrypto::ZeroBlock;

    std::array<osuCrypto::block, 2> local_seed{
        rng.get<osuCrypto::block>(), rng.get<osuCrypto::block>()};
    std::array<osuCrypto::block, 2> remote_seed;
    player->send_block(local_seed.data(), static_cast<int>(local_seed.size()));
    player->recv_block(remote_seed.data(), static_cast<int>(remote_seed.size()));
    session.hash_key = local_seed[0] ^ remote_seed[0];
    session.root_mask_key = local_seed[1] ^ remote_seed[1];
    session.initialized = true;
    return session;
}

Figure10CotMaterial prepareFigure10CotMaterial(
    int party_id, int Bin, const Figure10CotSession& session, Peer* player) {
    ensureParty(party_id);
    ensureSupportedFullTreeBits(Bin, "prepareFigure10CotMaterial");
    if (Bin <= 0 || !session.initialized || player == nullptr) {
        throw std::invalid_argument("Figure 10 COT session or depth is invalid");
    }
    Figure10CotMaterial material;
    material.sender_keys.resize(Bin);
    material.receiver_values.resize(Bin);
    material.random_choices.resize(Bin);
    auto rng = secure_prng();
    for (int i = 0; i < Bin; ++i) {
        material.random_choices[i] = rng.get<uint8_t>() & 1;
    }

    std::vector<osuCrypto::block> offsets(Bin, session.delta_prime);
    offsets[Bin - 1] = session.leaf_delta;
    auto send = [&](bool reversed) {
        player->send_cot_block(reversed, material.sender_keys.data(),
                               offsets.data(), Bin);
    };
    auto recv = [&](bool reversed) {
        player->recv_cot_block(reversed, material.receiver_values.data(),
                               material.random_choices.data(), Bin);
    };
    if (party_id == SERVER) {
        send(false);
        recv(true);
    } else {
        recv(false);
        send(true);
    }
    return material;
}

namespace {

CorrelatedTreeMaterial generateCorrelatedTreeImpl(
    int party_id, int Bin, int Bout, const BooleanElement* bits,
    const char* caller, bool computeArithmeticSums,
    DpfBitBlockMulMaterial* prepared_beaver,
    Figure10CotSession* cot_session, Figure10CotMaterial* prepared_cot) {
    ensureSupportedFullTreeBits(Bin, caller);
    if (Bin <= 0) {
        throw std::invalid_argument(std::string(caller) + " requires Bin > 0");
    }
    if (bits == nullptr) {
        throw std::invalid_argument(std::string(caller) +
                                    " requires target bits");
    }

    ensureParty(party_id);
    const int party_bit = party_id - SERVER;
    auto rng = secure_prng();

    const bool use_cot = cot_session != nullptr;
    if (use_cot && (prepared_cot == nullptr || !cot_session->initialized ||
                    prepared_cot->consumed ||
                    static_cast<int>(prepared_cot->sender_keys.size()) != Bin ||
                    static_cast<int>(prepared_cot->receiver_values.size()) != Bin ||
                    static_cast<int>(prepared_cot->random_choices.size()) != Bin)) {
        throw std::invalid_argument("Figure 10 COT material is invalid");
    }
    const osuCrypto::block hash_key =
        use_cot ? cot_session->hash_key : osuCrypto::ZeroBlock;
    osuCrypto::block root;
    if (use_cot) {
        osuCrypto::AES mask_prf(cot_session->root_mask_key);
        const osuCrypto::block w = mask_prf.ecbEncBlock(
            osuCrypto::toBlock(0, cot_session->tree_counter++));
        root = cot_session->delta_share ^ w;
    } else {
        root = rng.get<osuCrypto::block>();
        root = setBlockLsb(root, static_cast<osuCrypto::u8>(party_bit));
    }
    const osuCrypto::block delta_share =
        use_cot ? cot_session->delta_share : root;

    const size_t leafCapacity = size_t(1) << Bin;
    const size_t halfCapacity = (leafCapacity > 1) ? (leafCapacity / 2) : 1;
    auto largeLevelNodes =
        std::make_unique<osuCrypto::block[]>(leafCapacity);
    auto smallLevelNodes =
        std::make_unique<osuCrypto::block[]>(halfCapacity);
    osuCrypto::block* levelNodes =
        (Bin % 2 == 0) ? largeLevelNodes.get() : smallLevelNodes.get();
    osuCrypto::block* nextLevelNodes =
        (Bin % 2 == 0) ? smallLevelNodes.get() : largeLevelNodes.get();
    levelNodes[0] = root;

    auto tau = makeKeyArray<BooleanElement>(Bin * 2);
    auto scw = makeKeyArray<osuCrypto::block>(Bin + 1);
    scw[0] = root;
    for (int i = 0; i < Bin * 2; i++) {
        tau[i] = 0;
    }

    std::vector<BooleanElement> real_idx(Bin);
    for (int i = 0; i < Bin; i++) {
        real_idx[i] = static_cast<BooleanElement>(bits[i] & 1);
    }

    DpfBitBlockMulMaterial local_beaver;
    DpfBitBlockMulMaterial* beaver = prepared_beaver;
    if (!use_cot && beaver == nullptr && !computeArithmeticSums) {
        local_beaver = prepareDpfBitBlockMulMaterial(party_id, Bin, peer);
        beaver = &local_beaver;
    }

    std::vector<osuCrypto::block> cot_sender_keys;
    std::vector<osuCrypto::block> cot_receiver_values;
    if (use_cot) {
        const int packed_size = (Bin + 7) / 8;
        std::vector<BooleanElement> local_g(packed_size, 0);
        std::vector<BooleanElement> remote_g(packed_size, 0);
        for (int i = 0; i < Bin; ++i) {
            const BooleanElement g =
                real_idx[i] ^ prepared_cot->random_choices[i];
            local_g[i / 8] |= static_cast<BooleanElement>(g << (i % 8));
        }
        peer->send_u8(local_g.data(), packed_size);
        peer->recv_u8(remote_g.data(), packed_size);
        cot_sender_keys.resize(Bin);
        cot_receiver_values.resize(Bin);
        for (int i = 0; i < Bin; ++i) {
            const BooleanElement remote_choice =
                (remote_g[i / 8] >> (i % 8)) & 1;
            const osuCrypto::block sender_delta =
                i == Bin - 1 ? cot_session->leaf_delta : delta_share;
            cot_sender_keys[i] = prepared_cot->sender_keys[i] ^
                (remote_choice ? sender_delta : osuCrypto::ZeroBlock);
            cot_receiver_values[i] = prepared_cot->receiver_values[i];
            if (i < Bin - 1 && prepared_cot->random_choices[i]) {
                cot_receiver_values[i] = cot_receiver_values[i] ^
                    cot_session->receiver_delta_correction;
            }
        }
        prepared_cot->consumed = true;
    }

    std::vector<osuCrypto::block> delta_selected(Bin > 1 ? Bin - 1 : 0);
    if (!use_cot && Bin > 1) {
        std::vector<BooleanElement> alpha_bar_shares(Bin - 1);
        std::vector<osuCrypto::block> delta_choices(Bin - 1, delta_share);
        for (int i = 0; i < Bin - 1; i++) {
            alpha_bar_shares[i] =
                real_idx[i] ^ static_cast<BooleanElement>(1 - party_bit);
        }
        if (beaver != nullptr) {
            consumeDpfBitBlockMulMaterial(party_id, *beaver,
                                          alpha_bar_shares.data(),
                                          delta_choices.data(),
                                          delta_selected.data(), Bin - 1,
                                          peer);
        } else {
            bitBlockMultiply(party_id, alpha_bar_shares.data(),
                             delta_choices.data(), delta_selected.data(),
                             Bin - 1, peer);
        }
    }

    for (int i = 0; i < Bin - 1; i++) {
        const size_t levelSize = size_t(1) << i;

        osuCrypto::block levelHashXor = osuCrypto::ZeroBlock;
        for (size_t j = 0; j < levelSize; j++) {
            levelHashXor =
                levelHashXor ^ correlatedHash(levelNodes[j], hash_key);
        }

        osuCrypto::block layer_cw = levelHashXor;
        if (use_cot) {
            const BooleanElement alpha_bar = real_idx[i] ^ 1;
            layer_cw = layer_cw ^
                (alpha_bar ? delta_share : osuCrypto::ZeroBlock) ^
                cot_sender_keys[i] ^ cot_receiver_values[i];
        } else {
            layer_cw = layer_cw ^ delta_selected[i];
        }
        reconstruct(&layer_cw);
        scw[i + 1] = layer_cw;

        for (size_t j = 0; j < levelSize; j++) {
            const osuCrypto::block current = levelNodes[j];
            const BooleanElement controlBit = blockLsb(current);
            const osuCrypto::block hashed = correlatedHash(current, hash_key);
            const osuCrypto::block correction =
                controlBit ? layer_cw : osuCrypto::ZeroBlock;
            nextLevelNodes[2 * j] = hashed ^ correction;
            nextLevelNodes[2 * j + 1] = hashed ^ current ^ correction;
        }
        std::swap(levelNodes, nextLevelNodes);
    }

    const size_t penultimateSize = size_t(1) << (Bin - 1);
    const BooleanElement alpha_bar_last =
        real_idx[Bin - 1] ^ static_cast<BooleanElement>(1 - party_bit);
    osuCrypto::block high0 = osuCrypto::ZeroBlock;
    osuCrypto::block high1 = osuCrypto::ZeroBlock;
    BooleanElement low0 = 0;
    BooleanElement low1 = 0;
    for (size_t j = 0; j < penultimateSize; j++) {
        const osuCrypto::block q0 =
            correlatedHash(levelNodes[j], hash_key);
        const osuCrypto::block q1 =
            correlatedHash(levelNodes[j] ^ osuCrypto::OneBlock, hash_key);
        high0 = high0 ^ clearBlockLsb(q0);
        high1 = high1 ^ clearBlockLsb(q1);
        low0 ^= blockLsb(q0);
        low1 ^= blockLsb(q1);
    }

    osuCrypto::block hcw = osuCrypto::ZeroBlock;
    if (use_cot) {
        const osuCrypto::block mu = rng.get<osuCrypto::block>();
        const osuCrypto::block k = cot_sender_keys[Bin - 1];
        const osuCrypto::block m = cot_receiver_values[Bin - 1];
        auto high_hash = [&](const osuCrypto::block& input) {
            return clearBlockLsb(correlatedHash(input, hash_key));
        };
        const osuCrypto::block d = high_hash(mu ^ k) ^
            high_hash(mu ^ k ^ cot_session->leaf_delta) ^ high0 ^ high1;
        const osuCrypto::block local_messages[2] = {mu, d};
        osuCrypto::block remote_messages[2];
        peer->send_block(local_messages, 2);
        peer->recv_block(remote_messages, 2);
        const osuCrypto::block selected_high =
            real_idx[Bin - 1] ? high0 : high1;
        hcw = selected_high ^ high_hash(mu ^ k) ^
            high_hash(remote_messages[0] ^ m) ^
            (real_idx[Bin - 1] ? remote_messages[1]
                               : osuCrypto::ZeroBlock);
    } else {
        const osuCrypto::block high_diff = high0 ^ high1;
        osuCrypto::block high_selected = osuCrypto::ZeroBlock;
        if (beaver != nullptr) {
            consumeDpfBitBlockMulMaterial(party_id, *beaver,
                                          &alpha_bar_last, &high_diff,
                                          &high_selected, 1, peer);
        } else {
            bitBlockMultiply(party_id, &alpha_bar_last, &high_diff,
                             &high_selected, 1, peer);
        }
        hcw = high0 ^ high_selected;
    }
    BooleanElement lcw0 =
        low0 ^ real_idx[Bin - 1] ^ static_cast<BooleanElement>(party_bit);
    BooleanElement lcw1 = low1 ^ real_idx[Bin - 1];
    BooleanElement lcw_bits[2] = {lcw0, lcw1};
    reconstruct(&hcw, lcw_bits, 2);
    lcw0 = lcw_bits[0];
    lcw1 = lcw_bits[1];
    hcw = clearBlockLsb(hcw);
    scw[Bin] = hcw;
    tau[2 * (Bin - 1)] = lcw0;
    tau[2 * (Bin - 1) + 1] = lcw1;

    const osuCrypto::block leafCw0 = setBlockLsb(hcw, lcw0);
    const osuCrypto::block leafCw1 = setBlockLsb(hcw, lcw1);
    for (size_t j = 0; j < penultimateSize; j++) {
        const osuCrypto::block current = levelNodes[j];
        const BooleanElement controlBit = blockLsb(current);
        const osuCrypto::block q0 = correlatedHash(current, hash_key);
        const osuCrypto::block q1 =
            correlatedHash(current ^ osuCrypto::OneBlock, hash_key);
        nextLevelNodes[2 * j] =
            q0 ^ (controlBit ? leafCw0 : osuCrypto::ZeroBlock);
        nextLevelNodes[2 * j + 1] =
            q1 ^ (controlBit ? leafCw1 : osuCrypto::ZeroBlock);
    }
    std::swap(levelNodes, nextLevelNodes);

    osuCrypto::block leafXor = osuCrypto::ZeroBlock;
    GroupElement convertedSum(0, Bout);
    uint64_t controlBitSum = 0;
    for (size_t i = 0; i < leafCapacity; i++) {
        leafXor = leafXor ^ levelNodes[i];
        if (computeArithmeticSums) {
            convertedSum =
                convertedSum + convertBlockToGroup(Bout, levelNodes[i]);
        }
        controlBitSum += static_cast<uint64_t>(blockLsb(levelNodes[i]));
    }

    CorrelatedTreeMaterial material;
    material.Bin = Bin;
    material.Bout = Bout;
    material.scw = scw;
    material.tau = tau;
    material.leaf_xor = leafXor;
    material.converted_sum = convertedSum;
    material.control_bit_sum = controlBitSum;
    material.hash_key = hash_key;
    return material;
}

}  // namespace

CorrelatedTreeMaterial generateCorrelatedTree(
    int party_id, int Bin, int Bout, const BooleanElement* bits,
    const char* caller, bool computeArithmeticSums) {
    return generateCorrelatedTreeImpl(party_id, Bin, Bout, bits, caller,
                                      computeArithmeticSums, nullptr,
                                      nullptr, nullptr);
}

CorrelatedTreeMaterial generateCorrelatedTreeBeaverPrepared(
    int party_id, int Bin, int Bout, const BooleanElement* bits,
    const char* caller, bool computeArithmeticSums,
    DpfBitBlockMulMaterial& prepared) {
    if (static_cast<int>(prepared.a.size()) != Bin || prepared.cursor != 0) {
        throw std::invalid_argument("Beaver DPF material is invalid");
    }
    return generateCorrelatedTreeImpl(party_id, Bin, Bout, bits, caller,
                                      computeArithmeticSums, &prepared,
                                      nullptr, nullptr);
}

CorrelatedTreeMaterial generateCorrelatedTreeFigure10(
    int party_id, int Bin, int Bout, const BooleanElement* bits,
    const char* caller, bool computeArithmeticSums,
    Figure10CotSession& session, Figure10CotMaterial& prepared) {
    return generateCorrelatedTreeImpl(party_id, Bin, Bout, bits, caller,
                                      computeArithmeticSums, nullptr,
                                      &session, &prepared);
}

osuCrypto::block evalCorrelatedDPFLeaf(const BooleanElement* opened_bits,
                                       const DPFKeyPack& key,
                                       BooleanElement* controlBitOut) {
    return evalLeaf(opened_bits, key, controlBitOut);
}

osuCrypto::block evalCorrelatedDPFLeaf(
    const BooleanElement* opened_bits, const BooleanDPFKeyPack& key,
    BooleanElement* controlBitOut) {
    return evalLeaf(opened_bits, key, controlBitOut);
}

}  // namespace dfss::internal
