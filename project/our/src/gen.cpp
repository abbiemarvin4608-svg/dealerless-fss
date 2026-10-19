#include "gen.h"
#include "runtime_options.h"

#include <future>
#include <limits>
#include <unordered_set>
#include <stdexcept>
#include <cryptoTools/Common/Defines.h>

#include "Millionaire/bit-triple-generator.h"

#include "cggm.h"
#include "mpc/secure_ops.h"

// Π_Gen using the from-scratch correlated-GGM primitive (replaces the broken
// reference library's keyGenCorrelatedDPFBit which suffers from the SIOF bug
// in setBlockLsb). Now using segfss::cggmGen2pc + cggmEval.

namespace segfss {

namespace {

struct KkotCounter {
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t rounds = 0;
};

KkotCounter snapshotKkot(Peer* player) {
    const auto* io = player->getOTPack()->kkot[2]->io;
    return {io->counter, io->recv_counter, io->num_rounds};
}

void accountKkot(Peer* player, const KkotCounter& before) {
    const KkotCounter after = snapshotKkot(player);
    // TripleGenerator talks to KKOT directly, bypassing Peer wrappers.  NetIO
    // therefore exposes separate send/receive deltas that must both be
    // included in the phase communication total.
    player->bytesSent += after.bytes_sent - before.bytes_sent;
    player->bytesReceived += after.bytes_received - before.bytes_received;
    player->rounds += after.rounds - before.rounds;
}

}  // namespace

BooleanAndPreprocessing booleanAndOffline(int party_id, int gate_count,
                                          Peer* player) {
    if (gate_count < 0) {
        throw std::invalid_argument(
            "booleanAndOffline: gate count must be non-negative");
    }
    if (gate_count == 0) {
        return {};
    }
    if (player == nullptr) {
        throw std::invalid_argument("booleanAndOffline: peer is null");
    }

    // _8KKOT emits two triples per iteration and requires an even request.
    const int generated_count = gate_count + (gate_count & 1);
    Triple triples(generated_count);
    const KkotCounter before = snapshotKkot(player);
    TripleGenerator generator(party_id - 1, player->getIOPack(),
                               player->getOTPack());
    generator.generate(party_id - 1, triples.ai, triples.bi, triples.ci,
                       generated_count, _8KKOT);
    accountKkot(player, before);
    player->preprocessing_ledger.boolean_triples += generated_count;

    BooleanAndPreprocessing result;
    result.a.assign(triples.ai, triples.ai + gate_count);
    result.b.assign(triples.bi, triples.bi + gate_count);
    result.c.assign(triples.ci, triples.ci + gate_count);
    return result;
}

void openBooleanBits(int party_id, std::vector<u8>& values, Peer* player,
                     BooleanOpenWorkspace* workspace) {
    if (values.empty()) return;
    if (!player || (party_id != SERVER && party_id != CLIENT))
        throw std::invalid_argument("Boolean opening: invalid role or peer");
    BooleanOpenWorkspace local;
    auto& scratch = workspace ? *workspace : local;
    const bool packed = runtimeFlag("SEGFSS_PACK_BITS");
    const auto bytes = packed ? (values.size() + 7) / 8 : values.size();
    scratch.packed.assign(bytes, 0);
    scratch.remote_packed.resize(bytes);
    for (std::size_t bit = 0; bit < values.size(); ++bit) {
        if (values[bit] > 1) throw std::invalid_argument("Boolean opening: non-bit share");
        if (packed) scratch.packed[bit / 8] |= values[bit] << (bit % 8);
        else scratch.packed[bit] = values[bit];
    }
    if (party_id == SERVER) {
        player->send_u8(scratch.packed.data(), bytes);
        player->recv_u8(scratch.remote_packed.data(), bytes);
    } else {
        player->recv_u8(scratch.remote_packed.data(), bytes);
        player->send_u8(scratch.packed.data(), bytes);
    }
    for (std::size_t bit = 0; bit < values.size(); ++bit)
        values[bit] ^= packed ? (scratch.remote_packed[bit / 8] >> (bit % 8)) & 1 : scratch.remote_packed[bit];
}

void booleanAndOnline(int party_id, const u8* dataA, const u8* dataB,
                      u8* output, int size,
                      BooleanAndPreprocessing& preprocessing, Peer* player) {
    booleanAndOnlineBatch(party_id, {{dataA, dataB, output, size, &preprocessing}}, player);
}

void booleanAndOnlineBatch(int party_id, const std::vector<BooleanAndBatchView>& groups,
                          Peer* player, BooleanAndWorkspace* workspace) {
    std::size_t count = 0;
    std::unordered_set<BooleanAndPreprocessing*> seen;
    for (const auto& group : groups) {
        if (group.count < 0 || !group.material || !seen.insert(group.material).second ||
            (group.count && (!group.a || !group.b || !group.output)) ||
            group.material->b.size() != group.material->size() || group.material->c.size() != group.material->size() ||
            group.material->cursor > group.material->size() ||
            std::size_t(group.count) > group.material->size() - group.material->cursor)
            throw std::invalid_argument("booleanAndOnlineBatch: invalid input or material");
        count += group.count;
    }
    if (!count) return;
    if (count > std::size_t(std::numeric_limits<int>::max() / 2))
        throw std::invalid_argument("booleanAndOnlineBatch: message too large");
    BooleanAndWorkspace local;
    auto& scratch = workspace ? *workspace : local;
    scratch.opened.resize(2 * count);
    std::size_t index = 0;
    for (const auto& group : groups) {
        const auto offset = group.material->cursor;
        for (int i = 0; i < group.count; ++i) {
            scratch.opened[index + i] = group.a[i] ^ group.material->a[offset + i];
            scratch.opened[count + index + i] = group.b[i] ^ group.material->b[offset + i];
        }
        index += group.count;
    }
    // Consume every material before the single network opening, including failures.
    for (const auto& group : groups) group.material->cursor += group.count;
    openBooleanBits(party_id, scratch.opened, player, &scratch.wire);
    index = 0;
    for (const auto& group : groups) {
        const auto offset = group.material->cursor - group.count;
        for (int i = 0; i < group.count; ++i) {
            const u8 e = scratch.opened[index + i], f = scratch.opened[count + index + i];
            group.output[i] = static_cast<u8>(party_id - 2) * e * f ^
                (f * group.a[i]) ^ (e * group.b[i]) ^ group.material->c[offset + i];
        }
        index += group.count;
    }
}

SegmentKeys genSegmentKeys(int party_id, const SplitResult& split_result,
                           PhaseTimer* timer) {
    SegmentKeys result;

    if (split_result.segment_bits.size() !=
        static_cast<std::size_t>(split_result.layout.nseg)) {
        throw std::invalid_argument(
            "genSegmentKeys: segment count mismatch");
    }

    for (int j = 0; j < split_result.layout.nseg; j++) {
        const std::vector<u8>& segment_bits = split_result.segment_bits[j];
        const int width = split_result.layout.width[j];

        if (static_cast<int>(segment_bits.size()) != width) {
            throw std::invalid_argument(
                "genSegmentKeys: segment bit count mismatch");
        }
    }

    // Execute all independent trees level-by-level.  The batch primitive keeps
    // network I/O in one thread and combines every segment's COT/opening for a
    // protocol layer, so communication depth depends on max(width), not on the
    // sum of segment widths.
    result.keys =
        cggmGen2pcBatch(party_id, split_result.segment_bits, timer);

    return result;
}

SegmentEvalResult evalSegment(int party_id, const CggmKey& key,
                               const std::vector<u8>& xhat_seg_j) {
    // Delegate to cggmEval (Grotto prefix-parity walk). Returns {L_j, e_j}.
    (void)party_id;  // cggmEval doesn't need party_id (key already encodes it)
    CggmEval ev = cggmEval(key, xhat_seg_j);
    SegmentEvalResult result;
    result.L_j = ev.L;
    result.e_j = ev.e;
    return result;
}

std::vector<SegmentEvalResult> evalAllSegments(
    int party_id, const SegmentKeys& keys,
    const std::vector<std::vector<u8>>& xhat_segments) {

    const size_t nseg = keys.keys.size();
    if (xhat_segments.size() != nseg) {
        throw std::invalid_argument(
            "evalAllSegments: segment count mismatch");
    }

    std::vector<SegmentEvalResult> results(nseg);
    if (nseg == 1) {
        results[0] = evalSegment(party_id, keys.keys[0], xhat_segments[0]);
        return results;
    }

    // Evaluation is local after key generation, so independent segment walks
    // can safely occupy separate workers without touching the shared Peer.
    std::vector<std::future<void>> tasks;
    tasks.reserve(nseg);
    for (size_t j = 0; j < nseg; j++) {
        tasks.emplace_back(std::async(std::launch::async, [&, j]() {
            results[j] =
                evalSegment(party_id, keys.keys[j], xhat_segments[j]);
        }));
    }
    for (auto& task : tasks) {
        task.get();
    }

    return results;
}

u8 inferComparison(int party_id,
                   const std::vector<SegmentEvalResult>& segment_results,
                   Peer* player) {
    const int nseg = static_cast<int>(segment_results.size());
    if (nseg == 0) {
        throw std::invalid_argument("inferComparison: empty segment list");
    }

    // Use the associative comparison-summary operator
    //   (L_a,e_a) o (L_b,e_b) = (L_a XOR (e_a AND L_b), e_a AND e_b).
    std::vector<SegmentEvalResult> level = segment_results;
    while (level.size() > 1) {
        const std::size_t pair_count = level.size() / 2;
        std::vector<u8> lhs(2 * pair_count);
        std::vector<u8> rhs(2 * pair_count);
        std::vector<u8> products(2 * pair_count);
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const SegmentEvalResult& left = level[2 * pair];
            const SegmentEvalResult& right = level[2 * pair + 1];
            lhs[2 * pair] = left.e_j;
            rhs[2 * pair] = right.e_j;
            lhs[2 * pair + 1] = left.e_j;
            rhs[2 * pair + 1] = right.L_j;
        }
        and_wrapper(party_id, lhs.data(), rhs.data(), products.data(),
                    static_cast<int>(products.size()), player);

        std::vector<SegmentEvalResult> next;
        next.reserve((level.size() + 1) / 2);
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            SegmentEvalResult combined;
            combined.e_j = products[2 * pair];
            combined.L_j = level[2 * pair].L_j ^ products[2 * pair + 1];
            next.push_back(combined);
        }
        if ((level.size() & 1U) != 0) {
            next.push_back(level.back());
        }
        level = std::move(next);
    }
    return level[0].L_j;
}

u8 inferComparisonPreprocessed(
    int party_id, const std::vector<SegmentEvalResult>& segment_results,
    BooleanAndPreprocessing& preprocessing, Peer* player) {
    const int nseg = static_cast<int>(segment_results.size());
    if (nseg == 0) {
        throw std::invalid_argument("inferComparison: empty segment list");
    }

    // Use the associative comparison-summary operator
    //   (L_a,e_a) ⊗ (L_b,e_b)
    //     = (L_a XOR (e_a AND L_b), e_a AND e_b).
    // Its L component expands to the paper's
    //   XOR_j ((AND_{i<j} e_i) AND L_j),
    // while a balanced reduction is a work-efficient parallel prefix scan.
    // Both ANDs for every pair in a level are independent and are issued as
    // one batch, giving ceil(log2(nseg)) dependent secure-AND rounds and
    // exactly 2*(nseg-1) AND gates.
    std::vector<SegmentEvalResult> level = segment_results;
    while (level.size() > 1) {
        const std::size_t pair_count = level.size() / 2;
        std::vector<u8> lhs(2 * pair_count);
        std::vector<u8> rhs(2 * pair_count);
        std::vector<u8> products(2 * pair_count);

        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const SegmentEvalResult& left = level[2 * pair];
            const SegmentEvalResult& right = level[2 * pair + 1];
            lhs[2 * pair] = left.e_j;
            rhs[2 * pair] = right.e_j;
            lhs[2 * pair + 1] = left.e_j;
            rhs[2 * pair + 1] = right.L_j;
        }

        booleanAndOnline(party_id, lhs.data(), rhs.data(), products.data(),
                         static_cast<int>(products.size()), preprocessing,
                         player);

        std::vector<SegmentEvalResult> next;
        next.reserve((level.size() + 1) / 2);
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            SegmentEvalResult combined;
            combined.e_j = products[2 * pair];
            combined.L_j =
                level[2 * pair].L_j ^ products[2 * pair + 1];
            next.push_back(combined);
        }
        if ((level.size() & 1U) != 0) {
            next.push_back(level.back());
        }
        level = std::move(next);
    }

    return level[0].L_j;
}

}  // namespace segfss
