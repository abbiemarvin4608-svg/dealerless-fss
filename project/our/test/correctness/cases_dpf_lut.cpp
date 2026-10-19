#include "cases.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "dpf_lut.h"
#include "mpc/api.h"

namespace {

using segfss::DpfPublicLutMaterial;

uint64_t maskForBits(int bits) {
    return bits == 64 ? UINT64_MAX : ((uint64_t(1) << bits) - 1);
}

PublicLUTData makeTable(int q, int output_bits) {
    PublicLUTData table;
    table.Bin = q;
    table.Bout = output_bits;
    table.values.reserve(size_t(1) << q);
    for (uint64_t i = 0; i < (uint64_t(1) << q); ++i) {
        const uint64_t value = (0x31u * i + 7u * i * i + 3u) &
                               maskForBits(output_bits);
        table.values.emplace_back(value, output_bits);
    }
    return table;
}

void check_lut_shape(ResultLog& log, int q, int output_bits,
                     std::mt19937_64& rng) {
    const std::size_t count = (std::size_t(1) << q) + 3;
    const auto table = makeTable(q, output_bits);
    const auto before = peer->rounds;
    auto materials = segfss::dpfPublicLutBatchOffline(
        party, table, count, static_cast<uint64_t>(q) << 32, peer);
    log.check_local("DPF LUT q=" + std::to_string(q) + " material count",
                    materials.size(), count);
    bool resident = true;
    for (const auto& material : materials) {
        resident = resident && !material.consumed && !material.released &&
            material.selection_share.size() == (size_t(1) << q) &&
            material.selection_share.bitsize() == output_bits &&
            material.selection_share.capacityBytes() ==
                material.selection_share.size() * static_cast<std::size_t>((output_bits + 7) / 8);
    }
    log.check_local("DPF LUT q=" + std::to_string(q) + " resident before input",
                    resident, 1);
    // Preparation owns all input-independent work.  The online call below is
    // the first place where the private index shares are supplied.
    const auto after_offline = peer->rounds;
    std::vector<GroupElement> inputs;
    std::vector<uint64_t> expected;
    inputs.reserve(count);
    expected.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const uint64_t index = i & maskForBits(q);
        const uint64_t local = rng() & maskForBits(q);
        inputs.push_back(split_share(index, q, local));
        expected.push_back(table.values[index].value);
    }
    auto outputs = segfss::dpfPublicLutBatchOnline(
        party, inputs, materials, peer, static_cast<uint64_t>(q) << 32,
        table);
    log.check_local("DPF LUT q=" + std::to_string(q) + " opened",
                    peer->rounds > after_offline, 1);
    log.check_vector("DPF LUT q=" + std::to_string(q) + " values",
                     outputs.data(), expected, output_bits);
    bool consumed = true;
    for (const auto& material : materials) consumed = consumed && material.consumed;
    log.check_local("DPF LUT q=" + std::to_string(q) + " one-use consume",
                    consumed, 1);
    const auto reuse_before = peer->rounds;
    bool rejected = false;
    try {
        (void)segfss::dpfPublicLutBatchOnline(
            party, inputs, materials, peer, static_cast<uint64_t>(q) << 32,
            table);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    log.check_local("DPF LUT q=" + std::to_string(q) + " rejects reuse",
                    rejected && peer->rounds == reuse_before, 1);
    log.check_local("DPF LUT q=" + std::to_string(q) + " offline not online",
                    after_offline >= before, 1);
    for (auto& material : materials)
        segfss::releaseDpfPublicLutMaterial(material);
}

}  // namespace

void check_dpf_public_lut(ResultLog& log) {
    std::mt19937_64 rng(0xD0F5A11ULL);
    check_lut_shape(log, 3, 7, rng);
    check_lut_shape(log, 6, 10, rng);
    check_lut_shape(log, 8, 16, rng);
}
