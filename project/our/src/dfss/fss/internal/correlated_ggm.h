#pragma once

#include <vector>

#include "commons/types.h"
#include "commons/group_element.h"
#include "commons/keypack.h"
#include "mpc/secure_ops.h"

// Correlated-GGM helpers for correlated DPF variants will be moved here from
// legacy DPF code during the staged refactor.
namespace dfss::internal {

struct CorrelatedTreeMaterial {
    int Bin = 0;
    int Bout = 0;
    KeyArray<osuCrypto::block> scw;
    KeyArray<BooleanElement> tau;
    osuCrypto::block leaf_xor = osuCrypto::ZeroBlock;
    GroupElement converted_sum;
    uint64_t control_bit_sum = 0;
    osuCrypto::block hash_key = osuCrypto::ZeroBlock;
};

struct Figure10CotSession {
    osuCrypto::block delta_prime = osuCrypto::ZeroBlock;
    osuCrypto::block delta_share = osuCrypto::ZeroBlock;
    osuCrypto::block leaf_delta = osuCrypto::ZeroBlock;
    osuCrypto::block hash_key = osuCrypto::ZeroBlock;
    osuCrypto::block root_mask_key = osuCrypto::ZeroBlock;
    osuCrypto::block receiver_delta_correction = osuCrypto::ZeroBlock;
    uint64_t tree_counter = 0;
    bool initialized = false;
};

struct Figure10CotMaterial {
    std::vector<osuCrypto::block> sender_keys;
    std::vector<osuCrypto::block> receiver_values;
    std::vector<BooleanElement> random_choices;
    bool consumed = false;
};

Figure10CotSession initializeFigure10CotSession(int party_id, Peer* player);
Figure10CotMaterial prepareFigure10CotMaterial(
    int party_id, int Bin, const Figure10CotSession& session, Peer* player);

CorrelatedTreeMaterial generateCorrelatedTreeBeaverPrepared(
    int party_id, int Bin, int Bout, const BooleanElement* bits,
    const char* caller, bool computeArithmeticSums,
    DpfBitBlockMulMaterial& prepared);

CorrelatedTreeMaterial generateCorrelatedTreeFigure10(
    int party_id, int Bin, int Bout, const BooleanElement* bits,
    const char* caller, bool computeArithmeticSums,
    Figure10CotSession& session, Figure10CotMaterial& prepared);

osuCrypto::block correlatedHash(const osuCrypto::block& input,
                                const osuCrypto::block& hash_key);

CorrelatedTreeMaterial generateCorrelatedTree(
    int party_id, int Bin, int Bout, const BooleanElement* bits,
    const char* caller, bool computeArithmeticSums);

osuCrypto::block evalCorrelatedDPFLeaf(const BooleanElement* opened_bits,
                                       const DPFKeyPack& key,
                                       BooleanElement* controlBitOut = nullptr);

osuCrypto::block evalCorrelatedDPFLeaf(
    const BooleanElement* opened_bits, const BooleanDPFKeyPack& key,
    BooleanElement* controlBitOut = nullptr);

}  // namespace dfss::internal
