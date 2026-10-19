#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>

#include "instrumentation.h"
class Peer;

// From-scratch rewrite of the Correlated-GGM (Half-Tree) primitive for the
// dealerless segment-FSS protocol. See paper Alg. 2 (Pi_PREP) and Alg. 3
// (Pi_Gen). This header exposes:
//   * cggmGenDealer -- a *plaintext* Half-Tree cGGM key generator that emits
//     both parties' Boolean-leaf keys (shared public correction words, distinct
//     root shares). It is the reconstructed-domain reference the dealerless 2PC
//     protocol must reproduce; it is also directly usable in tests via XOR
//     reconstruction of the two keys.
//   * cggmEval -- the Grotto prefix-parity local walk (no communication),
//     returning the comparison share L and equality share e for one segment.
//
// Node layout: an osuCrypto::block; the control bit t is lsb(block); the seed
// occupies the remaining bits. The global offset Delta has lsb(Delta)=1.
namespace segfss {

// Bound local hashing participants (including the caller); zero selects the
// hardware default, and explicit limits are capped by hardware concurrency.
// Configuration eagerly starts reusable workers, so account for it as setup.
// Each participant needs at least 32 contiguous tasks and an estimated 4096
// hashes/expansions. Short count-128 walks therefore stay on the caller, while
// larger vector batches can use the full configured pool.
// Concurrent local batch calls and reconfiguration are safe; each executor
// runs one batch at a time. This does not make socket-based APIs concurrent.
void setCggmThreadLimit(std::size_t threads);
std::size_t cggmThreadLimit();
// Run local range work on the shared executor; work_items controls its
// minimum-useful-batch threshold and callbacks receive [begin, end).
std::size_t cggmParallelForRanges(
    std::size_t count, std::size_t work_items, std::size_t participant_limit,
    const std::function<void(std::size_t, std::size_t)>& function);
// Zero means no caller scope; transport adapters can honor cap-one ownership.
std::size_t cggmCallerThreadLimit();
void observeCallerTransportParticipants(std::size_t participants);

// Restrict this caller and its nested local batches without reconfiguring the
// shared executor. Independent activation owners may each hold a cap-one scope.
class CggmThreadScope {
public:
    explicit CggmThreadScope(std::size_t participants);
    ~CggmThreadScope();
    CggmThreadScope(const CggmThreadScope&) = delete;
    CggmThreadScope& operator=(const CggmThreadScope&) = delete;
    std::size_t effectiveLimit() const;
    std::size_t observedParticipants() const { return observed_; }
    std::size_t observedTransportParticipants() const { return transport_observed_; }
private:
    friend class CggmObserver;
    CggmThreadScope* previous_ = nullptr;
    std::size_t cap_ = 1, observed_ = 0, transport_observed_ = 0;
};

// Boolean-leaf cGGM key for a single segment of depth `depth` (= segment width).
// `internal_cw[i]` is CW_{i+1} for level i in [0, depth-2] (levels 1..depth-1).
// The leaf level (level depth) uses hcw (high bits) + lcw[sigma] (control bit)
// with leaf correction word for direction sigma = setBlockLsb(hcw, lcw[sigma]).
struct CggmKey {
    int depth = 0;
    osuCrypto::block root = osuCrypto::ZeroBlock;
    std::vector<osuCrypto::block> internal_cw;  // size depth-1 (empty if depth==1)
    osuCrypto::block hcw = osuCrypto::ZeroBlock;
    osuCrypto::u8 lcw[2] = {0, 0};
};

// Offsets survive owning-batch moves and appends. Views are valid only until
// the batch is modified; no view owns or extends one-use key lifetime.
struct CggmKeyRecord {
    osuCrypto::block root = osuCrypto::ZeroBlock;
    osuCrypto::block hcw = osuCrypto::ZeroBlock;
    uint64_t cw_offset = 0;
    int depth = 0;
    osuCrypto::u8 lcw[2] = {0, 0};
};
struct CggmKeyView {
    int depth;
    osuCrypto::block root, hcw;
    const osuCrypto::block* internal_cw;
    osuCrypto::u8 lcw[2];
};
struct CggmKeyBatch {
    std::vector<CggmKeyRecord> records;
    std::vector<osuCrypto::block> correction_words;
    void append(const std::vector<CggmKey>& keys);
    void append(const CggmKeyBatch& keys);
    CggmKeyView view(std::size_t index) const;
    std::size_t size() const { return records.size(); }
    std::size_t capacityBytes() const;
};

// Result of one segment walk: XOR-reconstruction of L over both parties yields
// 1{xhat < alpha} for this segment; XOR-reconstruction of e yields
// 1{xhat == alpha}.
struct CggmEval {
    osuCrypto::u8 L = 0;
    osuCrypto::u8 e = 0;
};

// Scratch counts only additional bounded AES work buffers across participants;
// existing tree/output storage is accounted separately by the material owner.
struct CggmGenerationStats {
    std::size_t internal_hashes = 0;
    std::size_t leaf_hashes = 0;
    std::size_t participants = 0;
    std::size_t scratch_peak_bytes = 0;
    bool batched_aes = false;
    std::size_t workspace_capacity_bytes = 0;
    std::size_t workspace_capacity_growths = 0;
    std::size_t leaf_cw_openings = 0;
    uint64_t bit_block_products = 0, temporary_triples = 0, temporary_triples_consumed = 0;
    uint64_t general_ot_64 = 0, general_ot_128 = 0, ef_openings = 0;
    uint64_t native_cot_128 = 0;
    bool reusable_workspace = false, merged_leaf_cw = false;
};

// Keep generation buffers across same-shape tiles, then destroy before input.
// Capacity stays charged for its lifetime; destruction does not assert AS return.
struct CggmGenerationWorkspace {
    std::vector<osuCrypto::block> nodes, hashes;
    std::vector<std::size_t> node_offsets, hash_offsets, counts;
    std::size_t capacityBytes() const noexcept;
};

struct CggmEvaluationStats {
    std::size_t participants = 0;
    std::size_t internal_hashes = 0;
    std::size_t leaf_hashes = 0;
    std::size_t scratch_peak_bytes = 0;
    bool batched_aes = false;
};

// Plaintext Half-Tree cGGM dealer. Given the secret target `alpha_bits`
// (MS bit first, length `depth`), produces the two parties' keys. The keys
// share all public correction words and differ only in `root`. `delta` must
// satisfy lsb(delta)=1; if delta==ZeroBlock a random valid delta is sampled.
std::pair<CggmKey, CggmKey> cggmGenDealer(
    const std::vector<osuCrypto::u8>& alpha_bits,
    osuCrypto::block delta = osuCrypto::ZeroBlock);

// Grotto prefix-parity walk on one party's key. `xhat_bits` is MS bit first,
// length key.depth.
CggmEval cggmEval(const CggmKey& key,
                  const std::vector<osuCrypto::u8>& xhat_bits,
                  bool need_e = true, bool prune_leaf_hashes = false);

// Batched local walks.  Each key/input pair is evaluated independently; no
// network operation is performed by this API.  The result order matches the
// input order, and callers may use it to evaluate all local segment trees in
// parallel before entering a protocol-level scan.
std::vector<CggmEval> cggmEvalBatch(
    const std::vector<CggmKey>& keys,
    const std::vector<std::vector<osuCrypto::u8>>& xhat_bits,
    CggmEvaluationStats* stats = nullptr, bool need_e = true,
    bool prune_leaf_hashes = false);
std::vector<CggmEval> cggmEvalBatch(
    const std::vector<CggmKey>& keys,
    const std::vector<std::vector<osuCrypto::u8>>& xhat_bits,
    const std::vector<osuCrypto::u8>& need_e,
    CggmEvaluationStats* stats = nullptr, bool prune_leaf_hashes = false);

// Row-major segment keys read bits directly from public queries. Output
// capacity belongs to the caller and can be reused for subsequent queries.
void cggmEvalBatch(const CggmKeyBatch& keys,
    const std::vector<uint64_t>& queries, const std::vector<int>& shifts,
    int segments, std::vector<CggmEval>& output,
    CggmEvaluationStats* stats = nullptr, bool need_e = true,
    bool prune_leaf_hashes = false);
void cggmEvalBatch(const CggmKeyBatch& keys,
    const std::vector<uint64_t>& queries, const std::vector<int>& shifts,
    int segments, std::vector<CggmEval>& output,
    const std::vector<osuCrypto::u8>& need_e,
    CggmEvaluationStats* stats = nullptr, bool prune_leaf_hashes = false);

// --- Dealerless 2PC layer (paper Alg. 2 Pi_PREP + Alg. 3 Pi_Gen) ---
//
// In the real protocol each party P_b holds only its own XOR-shares
// (<alpha_i>_b, <Delta'>_b) and interacts over the network; F_COT / F_Rand /
// reconstruct are ideal functionalities. cggmGen2pcSim runs BOTH parties'
// local Pi_PREP + Pi_Gen computations in a single process, emulating F_COT
// (fresh key + correlated MAC), F_Rand (shared public coins), and reconstruct
// (local XOR) directly. The two emitted keys are exactly the dealerless
// protocol's per-party outputs: they share every public correction word and
// carry independent root shares whose XOR equals Delta. Reconstructing the two
// keys via cggmEval must yield the same DPF/Grotto behaviour as cggmGenDealer.
//
// `alpha0_bits` XOR `alpha1_bits` (bitwise, MS bit first, length depth) is the
// secret target alpha. `delta0`, `delta1` are the two parties' offset shares
// Delta'_0, Delta'_1 (arbitrary blocks); Pi_PREP fixes lsb of the reconstructed
// Delta to 1 by folding the lsb of Delta'_{1-b} into P_b's share. `prng`
// supplies all protocol randomness (public coins W_i and leaf masks mu_b) so
// tests can seed it for determinism.
std::pair<CggmKey, CggmKey> cggmGen2pcSim(
    const std::vector<osuCrypto::u8>& alpha0_bits,
    const std::vector<osuCrypto::u8>& alpha1_bits,
    osuCrypto::block delta0,
    osuCrypto::block delta1,
    osuCrypto::PRNG& prng);

// Real socket-based 2PC key generation (paper Alg. 2 Pi_PREP + Alg. 3 Pi_Gen).
// Party party_id (SERVER=2, CLIENT=3) holds Boolean XOR-shares alpha_bits
// (MS bit first, length depth) of the secret target alpha, interacts with the
// other party over the global socket `peer`, and returns its own CggmKey.
// The two parties' keys share all public correction words; their roots are
// independent shares root_b = <Delta>_b XOR W. F_COT implemented via the
// reference's bitBlockMultiply; F_Rand via secure_prng(); reconstruct via the
// global peer socket. Requires global `peer` (Peer*) to be set.
// Optional timer enables E3 phase breakdown: cot, expand, cw_exchange, leaf.
CggmKey cggmGen2pc(int party_id, const std::vector<osuCrypto::u8>& alpha_bits,
                   PhaseTimer* timer = nullptr);

// Batched real 2PC generation for independent segment trees.  All segments
// execute each protocol stage together: one batched multiplication for the
// non-leaf correlations, one batched correction-word opening per tree level,
// and batched leaf multiplication/opening.  Consequently the communication
// depth is governed by the deepest segment, rather than by the sum of segment
// depths. The returned key order matches alpha_segments. An explicit player
// owns all OT/opening state; nullptr retains the legacy global-peer default.
std::vector<CggmKey> cggmGen2pcBatch(
    int party_id,
    const std::vector<std::vector<osuCrypto::u8>>& alpha_segments,
    PhaseTimer* timer = nullptr,
    CggmGenerationStats* stats = nullptr,
    Peer* player = nullptr,
    std::size_t participant_limit = 0,
    CggmGenerationWorkspace* workspace = nullptr);

}  // namespace segfss
