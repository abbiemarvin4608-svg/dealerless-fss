#include "cggm.h"
#include "runtime_options.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#include <cryptoTools/Crypto/PRNG.h>

#include "fss/internal/ggm.h"  // dfss::internal::ccrHash
#include "mpc/api.h"           // SERVER, CLIENT, peer, reconstruct
#include "mpc/secure_ops.h"    // secure_prng, bitBlockMultiply

// From-scratch Half-Tree correlated-GGM primitive.
//
// Node = osuCrypto::block; control bit t = lsb(block); Delta has lsb=1.
// cGGM child expansion (identical in dealer-gen and eval):
//   H = ccrHash(node); corr = t ? CW : 0;
//   left  = H ^ corr;   right = H ^ node ^ corr.
// Leaf level uses independent hashes q0 = ccrHash(node),
//   q1 = ccrHash(node ^ OneBlock), corrected by setBlockLsb(hcw, lcw[sigma]).
//
// Dealer correction words (reconstructed domain), from the DPF invariant
// (on-path node XOR = Delta with control-XOR 1; off-path XOR = 0):
//   CW_i        = H(a) ^ H(b) ^ (alpha_i == 0 ? Delta : 0)
//   hcw,lcw[lose]  chosen so the lose leaf reconstructs to 0
//   lcw[keep]      chosen so the alpha leaf control bit reconstructs to 1
// where (a,b) are the on-path parent's two shares and lose = 1 - alpha_last.
namespace segfss {

thread_local CggmThreadScope* active_cggm_scope = nullptr;
class CggmObserver {
public:
    static void record(std::size_t participants) {
        for (auto* scope = active_cggm_scope; scope; scope = scope->previous_)
            scope->observed_ = std::max(scope->observed_, participants);
    }
    static void recordTransport(std::size_t participants) {
        for (auto* scope = active_cggm_scope; scope; scope = scope->previous_)
            scope->transport_observed_ = std::max(scope->transport_observed_, participants);
    }
};

namespace {

// A batch owns the executor until every local task has finished. Resizing the
// configured executor can therefore replace it without invalidating an active
// call; the old executor joins its threads after its last batch releases it.
class SegmentWorkerPool {
 public:
    explicit SegmentWorkerPool(std::size_t capacity) : capacity_(capacity) {
        workers_.reserve(capacity_ - 1);
        for (std::size_t id = 1; id < capacity_; ++id) {
            workers_.emplace_back(new Worker);
        }
        try {
            for (std::size_t id = 1; id < capacity_; ++id) {
                workers_[id - 1]->thread = std::thread([this, id] { workerLoop(id); });
            }
            // Explicit configuration includes thread startup, rather than
            // deferring that cost to the first online batch.
            std::unique_lock<std::mutex> lock(mutex_);
            completed_.wait(lock, [&] { return ready_workers_ == workers_.size(); });
        } catch (...) {
            stop();
            throw;
        }
    }

    ~SegmentWorkerPool() { stop(); }

    std::size_t capacity() const { return capacity_; }

    template <typename Function>
    void run(std::size_t count, std::size_t participants, Function& function) {
        // Concurrent local batches are safe, but each pool schedules one batch
        // at a time. Protocol randomness and socket state remain in the caller.
        std::unique_lock<std::mutex> batch_lock(batch_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context_ = &function;
            invoke_ = [](void* context, std::size_t begin, std::size_t end) {
                auto& task = *static_cast<Function*>(context);
                task(begin, end);
            };
            count_ = count;
            participants_ = participants;
            remaining_ = participants - 1;
            error_ = nullptr;
            ++epoch_;
        }
        // Only wake workers with a substantial contiguous chunk. The caller
        // executes chunk zero while these resident workers execute the rest.
        for (std::size_t id = 1; id < participants; ++id) {
            workers_[id - 1]->available.notify_one();
        }
        std::exception_ptr caller_error;
        try {
            const auto range = chunk(0, count, participants);
            function(range.first, range.second);
        } catch (...) {
            caller_error = std::current_exception();
        }
        std::unique_lock<std::mutex> lock(mutex_);
        completed_.wait(lock, [&] { return remaining_ == 0; });
        const auto error = caller_error ? caller_error : error_;
        context_ = nullptr;
        invoke_ = nullptr;
        if (error) std::rethrow_exception(error);
    }

 private:
    struct Worker {
        std::condition_variable available;
        std::thread thread;
    };

    static std::pair<std::size_t, std::size_t> chunk(
        std::size_t id, std::size_t count, std::size_t participants) {
        const std::size_t width = count / participants;
        const std::size_t remainder = count % participants;
        const std::size_t begin = id * width + std::min(id, remainder);
        return {begin, begin + width + (id < remainder)};
    }

    void workerLoop(std::size_t id) {
        std::size_t observed_epoch = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        ++ready_workers_;
        completed_.notify_one();
        for (;;) {
            workers_[id - 1]->available.wait(lock, [&] {
                return stopping_ || (epoch_ != observed_epoch && id < participants_);
            });
            if (stopping_) return;
            observed_epoch = epoch_;
            const auto range = chunk(id, count_, participants_);
            auto invoke = invoke_;
            void* context = context_;
            lock.unlock();
            std::exception_ptr error;
            try { invoke(context, range.first, range.second); }
            catch (...) { error = std::current_exception(); }
            lock.lock();
            if (error && !error_) error_ = error;
            if (--remaining_ == 0) completed_.notify_one();
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        for (auto& worker : workers_) worker->available.notify_one();
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) worker->thread.join();
        }
    }

    const std::size_t capacity_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::mutex batch_mutex_;
    std::mutex mutex_;
    std::condition_variable completed_;
    bool stopping_ = false;
    std::size_t ready_workers_ = 0;
    std::size_t epoch_ = 0;
    std::size_t participants_ = 0;
    std::size_t remaining_ = 0;
    std::size_t count_ = 0;
    void* context_ = nullptr;
    void (*invoke_)(void*, std::size_t, std::size_t) = nullptr;
    std::exception_ptr error_;
};

class SegmentExecutor {
 public:
    void configure(std::size_t threads) {
        std::shared_ptr<SegmentWorkerPool> previous;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t capacity = resolvedLimit(threads);
            if (capacity == limit_ && (capacity == 1 || pool_)) return;
            auto replacement = capacity > 1 ? std::make_shared<SegmentWorkerPool>(capacity)
                                            : nullptr;
            previous = std::move(pool_);
            pool_ = std::move(replacement);
            limit_ = capacity;
        }
        // An idle old pool is joined outside the configuration lock. An active
        // batch retains shared ownership and completes before that pool joins.
    }

    std::shared_ptr<SegmentWorkerPool> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (limit_ == 1) return nullptr;
        if (!pool_) pool_ = std::make_shared<SegmentWorkerPool>(limit_);
        return pool_;
    }

    std::size_t limit() {
        std::lock_guard<std::mutex> lock(mutex_);
        return limit_;
    }

 private:
    static std::size_t resolvedLimit(std::size_t threads) {
        const std::size_t hardware = std::max(1u, std::thread::hardware_concurrency());
        return threads ? std::min(threads, hardware) : hardware;
    }
    std::mutex mutex_;
    std::size_t limit_ = resolvedLimit(0);
    std::shared_ptr<SegmentWorkerPool> pool_;
};

SegmentExecutor& segmentExecutor() {
    static SegmentExecutor executor;
    return executor;
}

inline osuCrypto::u8 blockLsb(const osuCrypto::block& b) {
    return static_cast<osuCrypto::u8>(_mm_cvtsi128_si64x(b) & 1);
}

// Self-contained lsb setters. We deliberately do NOT reuse the reference
// library's setBlockLsb/clearBlockLsb: those XOR against a
// namespace-global mask initialized from another translation unit's global
// (osuCrypto::OneBlock), which hits the static-initialization-order fiasco and
// silently becomes a no-op on the lsb. We build the mask at runtime instead.
inline osuCrypto::block setBlockLsb(const osuCrypto::block& input,
                                    osuCrypto::u8 bit) {
    const osuCrypto::block one = _mm_set_epi64x(0, 1);
    if (blockLsb(input) != (bit & 1)) {
        return _mm_xor_si128(input, one);
    }
    return input;
}

inline osuCrypto::block clearBlockLsb(const osuCrypto::block& input) {
    return setBlockLsb(input, 0);
}

// Segment-local hashing and expansion never touches the network.  Run those
// independent operations concurrently while keeping every socket operation in
// the caller thread as one explicitly batched protocol message.
template <typename Function>
std::size_t parallelForSegmentRanges(std::size_t count, std::size_t work_items,
                         Function function, std::size_t participant_limit = 0) {
    // Each participant gets at least 32 adjacent segment tasks and an
    // estimated 4096 hashes/expansions. The matched pilot showed that waking
    // resident workers still cost more than these short count-128 walks saved.
    // Keep those walks on the caller while large vector batches and offline
    // levels retain enough work per participant to use the configured pool.
    const std::size_t useful_participants = std::min(count / 32, work_items / 4096);
    if (active_cggm_scope) {
        const auto cap = active_cggm_scope->effectiveLimit();
        participant_limit = participant_limit ? std::min(participant_limit, cap) : cap;
    }
    if (useful_participants >= 2 && participant_limit != 1) {
        auto pool = segmentExecutor().acquire();
        if (pool) {
            const auto participants = std::min({pool->capacity(), useful_participants,
                participant_limit ? participant_limit : pool->capacity()});
            pool->run(count, participants, function);
            CggmObserver::record(participants);
            return participants;
        }
    }
    function(0, count);
    CggmObserver::record(count ? 1 : 0);
    return count ? 1 : 0;
}

template <typename Function>
std::size_t parallelForSegments(std::size_t count, std::size_t work_items,
                               Function function, std::size_t participant_limit = 0) {
    return parallelForSegmentRanges(count, work_items, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) function(index);
    }, participant_limit);
}

void ccrHashBatch(const osuCrypto::block* inputs, std::size_t count, osuCrypto::block* hashes) {
    osuCrypto::mAesFixedKey.ecbEncBlocks(inputs, count, hashes);
    for (std::size_t index = 0; index < count; ++index) hashes[index] ^= inputs[index];
}

constexpr std::size_t aes_lanes = 256;
struct AesWalkScratch {
    std::array<CggmKeyView, aes_lanes> keys;
    std::array<osuCrypto::block, aes_lanes> nodes;
    std::array<osuCrypto::block, 2 * aes_lanes> inputs, hashes;
    std::array<std::size_t, aes_lanes> offsets;
    std::array<osuCrypto::u8, aes_lanes> less;
};

// Each resident participant advances its own bounded block of independent
// paths. Completed shallow paths keep their original output positions.
template <typename KeyAt, typename DirectionAt, typename NeedEqualityAt>
std::size_t evaluateBatchedAes(std::size_t count, std::size_t work,
                              KeyAt key_at, DirectionAt direction_at,
                              std::vector<CggmEval>& output, NeedEqualityAt need_e_at,
                              bool prune_leaf_hashes) {
    return parallelForSegmentRanges(count, work, [&](std::size_t begin, std::size_t end) {
        AesWalkScratch scratch;
        for (std::size_t base = begin; base < end; base += aes_lanes) {
            const std::size_t lanes = std::min(aes_lanes, end - base);
            int max_depth = 0;
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                scratch.keys[lane] = key_at(base + lane);
                scratch.nodes[lane] = scratch.keys[lane].root;
                scratch.less[lane] = 0;
                max_depth = std::max(max_depth, scratch.keys[lane].depth);
            }
            for (int level = 0; level < max_depth; ++level) {
                std::size_t hash_count = 0;
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    const auto& key = scratch.keys[lane];
                    if (level >= key.depth) continue;
                    scratch.offsets[lane] = hash_count;
                    const bool leaf = level == key.depth - 1;
                    const bool direction = direction_at(base + lane, level, key.depth);
                    const bool need_e = need_e_at(base + lane);
                    if (!leaf || !prune_leaf_hashes || (need_e && !direction))
                        scratch.inputs[hash_count++] = scratch.nodes[lane];
                    if (leaf && (!prune_leaf_hashes || !direction || need_e))
                        scratch.inputs[hash_count++] = scratch.nodes[lane] ^ osuCrypto::OneBlock;
                }
                if (hash_count) ccrHashBatch(scratch.inputs.data(), hash_count, scratch.hashes.data());
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    const auto& key = scratch.keys[lane];
                    if (level >= key.depth) continue;
                    const auto node = scratch.nodes[lane];
                    const auto offset = scratch.offsets[lane];
                    const auto control = blockLsb(node);
                    const bool direction = direction_at(base + lane, level, key.depth);
                    if (level < key.depth - 1) {
                        const auto left = scratch.hashes[offset] ^
                            (control ? key.internal_cw[level] : osuCrypto::ZeroBlock);
                        const auto right = left ^ node;
                        if (!direction) scratch.less[lane] ^= blockLsb(right);
                        scratch.nodes[lane] = direction ? right : left;
                    } else {
                        const bool need_e = need_e_at(base + lane);
                        const bool have_left = !prune_leaf_hashes || (need_e && !direction);
                        const bool have_right = !prune_leaf_hashes || !direction || need_e;
                        const auto left = have_left ? scratch.hashes[offset] ^
                            (control ? setBlockLsb(key.hcw, key.lcw[0]) : osuCrypto::ZeroBlock) : osuCrypto::ZeroBlock;
                        const auto right = have_right ? scratch.hashes[offset + have_left] ^
                            (control ? setBlockLsb(key.hcw, key.lcw[1]) : osuCrypto::ZeroBlock) : osuCrypto::ZeroBlock;
                        if (!direction) scratch.less[lane] ^= blockLsb(right);
                        output[base + lane] = {scratch.less[lane], need_e ? blockLsb(direction ? right : left) : 0};
                    }
                }
            }
        }
    });
}

std::size_t leafHashCount(bool direction, bool need_e, bool prune_leaf_hashes) {
    if (!prune_leaf_hashes) return 2;
    return direction ? std::size_t(need_e) : 1 + std::size_t(need_e);
}

struct AesLeafScratch {
    std::array<osuCrypto::block, 2 * aes_lanes> inputs, hashes;
};

std::array<osuCrypto::block, 2> hashLeafSums(const osuCrypto::block* nodes,
                                          std::size_t count, bool batched) {
    std::array<osuCrypto::block, 2> sums{osuCrypto::ZeroBlock, osuCrypto::ZeroBlock};
    if (!batched) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto node = nodes[i];
            sums[0] ^= dfss::internal::ccrHash(node);
            sums[1] ^= dfss::internal::ccrHash(node ^ osuCrypto::OneBlock);
        }
        return sums;
    }
    AesLeafScratch scratch;
    for (std::size_t base = 0; base < count; base += aes_lanes) {
        const auto lanes = std::min(aes_lanes, count - base);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            scratch.inputs[2 * lane] = nodes[base + lane];
            scratch.inputs[2 * lane + 1] = nodes[base + lane] ^ osuCrypto::OneBlock;
        }
        ccrHashBatch(scratch.inputs.data(), 2 * lanes, scratch.hashes.data());
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            sums[0] ^= scratch.hashes[2 * lane];
            sums[1] ^= scratch.hashes[2 * lane + 1];
        }
    }
    return sums;
}

}  // namespace

CggmThreadScope::CggmThreadScope(std::size_t participants) : previous_(active_cggm_scope) {
    if (!participants) throw std::invalid_argument("cGGM caller scope requires a positive participant cap");
    cap_ = previous_ ? std::min(participants, previous_->cap_) : participants;
    active_cggm_scope = this;
}
CggmThreadScope::~CggmThreadScope() { active_cggm_scope = previous_; }
std::size_t CggmThreadScope::effectiveLimit() const { return std::min(cap_, cggmThreadLimit()); }
std::size_t cggmParallelForRanges(
    std::size_t count, std::size_t work_items, std::size_t participant_limit,
    const std::function<void(std::size_t, std::size_t)>& function) {
    if (!function) throw std::invalid_argument("cGGM range callback is empty");
    return parallelForSegmentRanges(count, work_items, function, participant_limit);
}
std::size_t cggmCallerThreadLimit() {
    return active_cggm_scope ? active_cggm_scope->effectiveLimit() : 0;
}
void observeCallerTransportParticipants(std::size_t participants) { CggmObserver::recordTransport(participants); }

void CggmKeyBatch::append(const std::vector<CggmKey>& keys) {
    std::size_t words = 0;
    for (const auto& key : keys) {
        if (key.depth < 1 || key.depth > 64 ||
            key.internal_cw.size() != static_cast<std::size_t>(key.depth - 1))
            throw std::invalid_argument("compact cGGM: malformed key");
        words += key.internal_cw.size();
    }
    records.reserve(records.size() + keys.size());
    correction_words.reserve(correction_words.size() + words);
    for (const auto& key : keys) {
        CggmKeyRecord record;
        record.root = key.root; record.hcw = key.hcw;
        record.depth = key.depth; record.cw_offset = correction_words.size();
        record.lcw[0] = key.lcw[0]; record.lcw[1] = key.lcw[1];
        records.push_back(record);
        correction_words.insert(correction_words.end(), key.internal_cw.begin(), key.internal_cw.end());
    }
}

void CggmKeyBatch::append(const CggmKeyBatch& keys) {
    if (&keys == this) throw std::invalid_argument("compact cGGM: cannot append itself");
    const auto offset = correction_words.size();
    records.reserve(records.size() + keys.size());
    correction_words.insert(correction_words.end(), keys.correction_words.begin(), keys.correction_words.end());
    for (auto record : keys.records) { record.cw_offset += offset; records.push_back(record); }
}

CggmKeyView CggmKeyBatch::view(std::size_t index) const {
    const auto& key = records.at(index);
    if (key.depth < 1 || key.depth > 64 || key.cw_offset > correction_words.size() ||
        static_cast<std::size_t>(key.depth - 1) > correction_words.size() - key.cw_offset)
        throw std::invalid_argument("compact cGGM: correction-word range");
    return {key.depth, key.root, key.hcw,
        key.depth == 1 ? nullptr : correction_words.data() + key.cw_offset,
        {key.lcw[0], key.lcw[1]}};
}

std::size_t CggmKeyBatch::capacityBytes() const {
    return records.capacity() * sizeof(CggmKeyRecord) + correction_words.capacity() * sizeof(osuCrypto::block);
}

void cggmEvalBatch(const CggmKeyBatch& keys,
    const std::vector<uint64_t>& queries, const std::vector<int>& shifts,
    int segments, std::vector<CggmEval>& output, CggmEvaluationStats* stats, bool need_e,
    bool prune_leaf_hashes) {
    cggmEvalBatch(keys, queries, shifts, segments, output,
                 std::vector<osuCrypto::u8>(keys.size(), need_e), stats, prune_leaf_hashes);
}

void cggmEvalBatch(const CggmKeyBatch& keys,
    const std::vector<uint64_t>& queries, const std::vector<int>& shifts,
    int segments, std::vector<CggmEval>& output,
    const std::vector<osuCrypto::u8>& need_e, CggmEvaluationStats* stats, bool prune_leaf_hashes) {
    if (segments < 1 || shifts.size() != static_cast<std::size_t>(segments) ||
        keys.size() != queries.size() * segments || need_e.size() != keys.size() ||
        std::any_of(need_e.begin(), need_e.end(), [](osuCrypto::u8 bit) { return bit > 1; }))
        throw std::invalid_argument("compact cGGM: query shape");
    std::size_t work = 0;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        const auto key = keys.view(index);
        const int shift = shifts[index % segments];
        if (shift < 0 || shift > 64 - key.depth)
            throw std::invalid_argument("compact cGGM: public shift");
        work += key.depth + 1;
    }
    output.resize(keys.size());
    const bool batched = runtimeFlag("SEGFSS_AES_ONLINE");
    const auto participants = batched ? evaluateBatchedAes(keys.size(), work,
        [&](std::size_t index) { return keys.view(index); },
        [&](std::size_t index, int level, int depth) {
            return (queries[index / segments] >> (shifts[index % segments] + depth - 1 - level)) & 1;
        }, output, [&](std::size_t index) { return bool(need_e[index]); }, prune_leaf_hashes) :
        parallelForSegments(keys.size(), work, [&](std::size_t index) {
        const auto key = keys.view(index);
        const uint64_t query = queries[index / segments];
        const int shift = shifts[index % segments];
        std::vector<osuCrypto::u8> bits(key.depth);
        for (int level = 0; level < key.depth; ++level)
            bits[level] = (query >> (shift + key.depth - 1 - level)) & 1;
        CggmKey expanded;
        expanded.depth = key.depth;
        expanded.root = key.root;
        expanded.hcw = key.hcw;
        expanded.lcw[0] = key.lcw[0];
        expanded.lcw[1] = key.lcw[1];
        if (key.depth > 1)
            expanded.internal_cw.assign(key.internal_cw, key.internal_cw + key.depth - 1);
        output[index] = cggmEval(expanded, bits, need_e[index], prune_leaf_hashes);
    });
    if (stats) {
        std::size_t leaves = 0;
        for (std::size_t index = 0; index < keys.size(); ++index)
            leaves += leafHashCount((queries[index / segments] >> shifts[index % segments]) & 1U,
                                    need_e[index], prune_leaf_hashes);
        stats->participants = participants;
        stats->internal_hashes = work - 2 * keys.size();
        stats->leaf_hashes = leaves;
        stats->scratch_peak_bytes = batched ? participants * sizeof(AesWalkScratch) : 0;
        stats->batched_aes = batched;
    }
}

std::pair<CggmKey, CggmKey> cggmGenDealer(
    const std::vector<osuCrypto::u8>& alpha_bits, osuCrypto::block delta) {
    const int depth = static_cast<int>(alpha_bits.size());
    if (depth <= 0) {
        throw std::invalid_argument("cggmGenDealer: depth must be > 0");
    }

    // Delta with lsb = 1 (sample if not supplied).
    if (std::memcmp(&delta, &osuCrypto::ZeroBlock, sizeof(delta)) == 0) {
        osuCrypto::PRNG prng(osuCrypto::sysRandomSeed());
        delta = prng.get<osuCrypto::block>();
    }
    delta = setBlockLsb(delta, 1);

    // Root shares: root0 random, root1 = root0 ^ Delta  (=> XOR = Delta).
    // Derive root0 deterministically from delta so a dealer invoked in two
    // separate processes with the same (alpha, delta) yields the same key pair
    // (the real 2PC protocol instead draws the root from <Delta> XOR W).
    osuCrypto::PRNG prng(delta);
    osuCrypto::block root0 = prng.get<osuCrypto::block>();
    osuCrypto::block root1 = root0 ^ delta;

    CggmKey k0;
    CggmKey k1;
    k0.depth = depth;
    k1.depth = depth;
    k0.root = root0;
    k1.root = root1;
    k0.internal_cw.assign(depth > 0 ? depth - 1 : 0, osuCrypto::ZeroBlock);
    k1.internal_cw.assign(depth > 0 ? depth - 1 : 0, osuCrypto::ZeroBlock);

    // Walk the on-path chain, emitting CW_i for internal levels 1..depth-1.
    osuCrypto::block a = root0;  // on-path parent, party 0
    osuCrypto::block b = root1;  // on-path parent, party 1
    for (int i = 1; i <= depth - 1; i++) {
        const osuCrypto::u8 alpha_i = alpha_bits[i - 1] & 1;  // MS bit first
        const osuCrypto::block Ha = dfss::internal::ccrHash(a);
        const osuCrypto::block Hb = dfss::internal::ccrHash(b);
        const osuCrypto::block cw =
            Ha ^ Hb ^ (alpha_i == 0 ? delta : osuCrypto::ZeroBlock);
        k0.internal_cw[i - 1] = cw;
        k1.internal_cw[i - 1] = cw;

        const osuCrypto::u8 t0 = blockLsb(a);
        const osuCrypto::u8 t1 = blockLsb(b);
        const osuCrypto::block c0 = t0 ? cw : osuCrypto::ZeroBlock;
        const osuCrypto::block c1 = t1 ? cw : osuCrypto::ZeroBlock;
        if (alpha_i == 0) {  // descend left
            a = Ha ^ c0;
            b = Hb ^ c1;
        } else {  // descend right
            a = Ha ^ a ^ c0;
            b = Hb ^ b ^ c1;
        }
    }

    // Leaf level. (a,b) is the on-path penultimate node (XOR = Delta).
    const osuCrypto::u8 alpha_last = alpha_bits[depth - 1] & 1;
    const osuCrypto::u8 keep = alpha_last;
    const osuCrypto::u8 lose = static_cast<osuCrypto::u8>(1 - alpha_last);

    auto qhash = [](const osuCrypto::block& node, osuCrypto::u8 sigma) {
        return dfss::internal::ccrHash(
            sigma ? (node ^ osuCrypto::OneBlock) : node);
    };

    // lose leaf reconstructs to 0  -> leafCw[lose] = q_lose(a) ^ q_lose(b).
    const osuCrypto::block lose_xor = qhash(a, lose) ^ qhash(b, lose);
    const osuCrypto::block hcw = clearBlockLsb(lose_xor);
    osuCrypto::u8 lcw[2] = {0, 0};
    lcw[lose] = blockLsb(lose_xor);
    // keep (alpha) leaf control-bit reconstructs to 1.
    lcw[keep] = static_cast<osuCrypto::u8>(
        1 ^ blockLsb(qhash(a, keep)) ^ blockLsb(qhash(b, keep)));

    k0.hcw = hcw;
    k1.hcw = hcw;
    k0.lcw[0] = lcw[0];
    k0.lcw[1] = lcw[1];
    k1.lcw[0] = lcw[0];
    k1.lcw[1] = lcw[1];

    return {k0, k1};
}

CggmEval cggmEval(const CggmKey& key,
                  const std::vector<osuCrypto::u8>& xhat_bits, bool need_e, bool prune_leaf_hashes) {
    const int depth = key.depth;
    if (static_cast<int>(xhat_bits.size()) != depth) {
        throw std::invalid_argument("cggmEval: xhat_bits size != depth");
    }

    osuCrypto::block levelNode = key.root;
    osuCrypto::u8 t = blockLsb(levelNode);
    osuCrypto::u8 L = 0;

    // Internal levels 1..depth-1.
    for (int i = 0; i < depth - 1; i++) {
        const osuCrypto::u8 dir = xhat_bits[i] & 1;
        const osuCrypto::block H = dfss::internal::ccrHash(levelNode);
        const osuCrypto::block corr =
            t ? key.internal_cw[i] : osuCrypto::ZeroBlock;
        const osuCrypto::block left = H ^ corr;
        const osuCrypto::block right = H ^ levelNode ^ corr;
        if (dir == 0) {
            L ^= blockLsb(right);  // right-sibling tag
            levelNode = left;
        } else {
            levelNode = right;
        }
        t = blockLsb(levelNode);
    }

    // Leaf level (level depth): independent hashes.
    const osuCrypto::u8 finalDir = xhat_bits[depth - 1] & 1;
    // For an L-only request the selected right leaf is irrelevant when the
    // public final direction is 1.  Avoid both leaf hashes in that case.
    if (prune_leaf_hashes && !need_e && finalDir) return {L, 0};
    if (prune_leaf_hashes && !need_e && !finalDir) {
        const auto q1 = dfss::internal::ccrHash(levelNode ^ osuCrypto::OneBlock);
        const auto rightLeaf = q1 ^ (t ? setBlockLsb(key.hcw, key.lcw[1]) : osuCrypto::ZeroBlock);
        return {static_cast<osuCrypto::u8>(L ^ blockLsb(rightLeaf)), 0};
    }
    if (prune_leaf_hashes && finalDir) {
        const auto q1 = dfss::internal::ccrHash(levelNode ^ osuCrypto::OneBlock);
        const auto rightLeaf = q1 ^ (t ? setBlockLsb(key.hcw, key.lcw[1]) : osuCrypto::ZeroBlock);
        return {L, blockLsb(rightLeaf)};
    }
    const auto q0 = dfss::internal::ccrHash(levelNode);
    const auto leftLeaf = q0 ^ (t ? setBlockLsb(key.hcw, key.lcw[0]) : osuCrypto::ZeroBlock);
    if (!finalDir) {
        const auto q1 = dfss::internal::ccrHash(levelNode ^ osuCrypto::OneBlock);
        const auto rightLeaf = q1 ^ (t ? setBlockLsb(key.hcw, key.lcw[1]) : osuCrypto::ZeroBlock);
        L ^= blockLsb(rightLeaf);
        return {L, need_e ? blockLsb(leftLeaf) : 0};
    }
    const auto q1 = dfss::internal::ccrHash(levelNode ^ osuCrypto::OneBlock);
    const auto rightLeaf = q1 ^ (t ? setBlockLsb(key.hcw, key.lcw[1]) : osuCrypto::ZeroBlock);
    return {L, need_e ? blockLsb(rightLeaf) : 0};
}

std::vector<CggmEval> cggmEvalBatch(
    const std::vector<CggmKey>& keys,
    const std::vector<std::vector<osuCrypto::u8>>& xhat_bits,
    CggmEvaluationStats* stats, bool need_e, bool prune_leaf_hashes) {
    return cggmEvalBatch(keys, xhat_bits, std::vector<osuCrypto::u8>(keys.size(), need_e),
                        stats, prune_leaf_hashes);
}

std::vector<CggmEval> cggmEvalBatch(
    const std::vector<CggmKey>& keys,
    const std::vector<std::vector<osuCrypto::u8>>& xhat_bits,
    const std::vector<osuCrypto::u8>& need_e, CggmEvaluationStats* stats, bool prune_leaf_hashes) {
    if (keys.size() != xhat_bits.size() || need_e.size() != keys.size() ||
        std::any_of(need_e.begin(), need_e.end(), [](osuCrypto::u8 bit) { return bit > 1; })) {
        throw std::invalid_argument(
            "cggmEvalBatch: key and input batch sizes differ");
    }
    std::size_t work_items = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].depth <= 0 ||
            keys[i].internal_cw.size() !=
                static_cast<std::size_t>(keys[i].depth - 1) ||
            xhat_bits[i].size() != static_cast<std::size_t>(keys[i].depth)) {
            throw std::invalid_argument(
                "cggmEvalBatch: malformed key or input bits");
        }
        // The leaf hashes both directions, hence a depth-d walk hashes d+1
        // blocks rather than d. This count controls local scheduling only.
        work_items += static_cast<std::size_t>(keys[i].depth) + 1;
    }

    std::vector<CggmEval> output(keys.size());
    // Every invocation only reads its key and public input and writes one
    // disjoint output slot, so segment walks can run concurrently.  The
    // caller performs all communication after this function returns.
    const bool batched = runtimeFlag("SEGFSS_AES_ONLINE");
    const auto participants = batched ? evaluateBatchedAes(keys.size(), work_items,
        [&](std::size_t index) {
            const auto& key = keys[index];
            return CggmKeyView{key.depth, key.root, key.hcw, key.internal_cw.data(), {key.lcw[0], key.lcw[1]}};
        }, [&](std::size_t index, int level, int) { return xhat_bits[index][level] & 1; }, output,
           [&](std::size_t index) { return bool(need_e[index]); }, prune_leaf_hashes) :
        parallelForSegments(keys.size(), work_items, [&](std::size_t i) {
        output[i] = cggmEval(keys[i], xhat_bits[i], need_e[i], prune_leaf_hashes);
    });
    if (stats) {
        std::size_t leaves = 0;
        for (std::size_t index = 0; index < keys.size(); ++index)
            leaves += leafHashCount(xhat_bits[index].back(), need_e[index], prune_leaf_hashes);
        stats->participants = participants;
        stats->internal_hashes = work_items - 2 * keys.size();
        stats->leaf_hashes = leaves;
        stats->scratch_peak_bytes = batched ? participants * sizeof(AesWalkScratch) : 0;
        stats->batched_aes = batched;
    }
    return output;
}

std::pair<CggmKey, CggmKey> cggmGen2pcSim(
    const std::vector<osuCrypto::u8>& alpha0_bits,
    const std::vector<osuCrypto::u8>& alpha1_bits, osuCrypto::block delta0,
    osuCrypto::block delta1, osuCrypto::PRNG& prng) {
    const int depth = static_cast<int>(alpha0_bits.size());
    if (depth <= 0 || static_cast<int>(alpha1_bits.size()) != depth) {
        throw std::invalid_argument(
            "cggmGen2pcSim: alpha shares must have equal positive depth");
    }

    // --- Paper Algorithm 2: Pi_PREP (parallel-COT preprocessing) ---
    //
    // Step 1: combine offset shares Delta'_0, Delta'_1 into a shared Delta whose
    // *reconstructed* lsb is 1. The parties exchange the single bit lsb(Delta'_b)
    // (public: Delta's lsb is 1 by construction, so this leaks nothing about the
    // high bits). We keep P_0's share and flip P_1's lsb so the XOR lsb is 1:
    //   D0 = Delta'_0 ;  D1 = Delta'_1 with lsb := lsb(Delta'_0) ^ 1.
    // Then lsb(D0 ^ D1) = lsb0 ^ (lsb0 ^ 1) = 1, and the high bits are the random
    // XOR of the two shares. D_b is party b's offset share <Delta>_b.
    const osuCrypto::u8 lsb0 = blockLsb(delta0);
    const osuCrypto::block D0 = delta0;
    const osuCrypto::block D1 = setBlockLsb(delta1, lsb0 ^ 1);
    const osuCrypto::block Delta = D0 ^ D1;  // reconstructed offset, lsb=1

    // Step 2: generate ℓ COT correlations via F_COT (one per alpha bit). In the
    // real protocol, party b is the receiver with choice bit <alpha_i>_b and gets
    // back M[<alpha_i>_b]; party 1-b is the sender with offset D_{1-b} and holds
    // both keys K[0], K[1]. The output satisfies the IT-MAC correlation:
    //   M[x_b] = K[x_b] XOR x_b * D_{1-b}.
    // We simulate F_COT by sampling fresh keys and computing the MACs directly.
    std::vector<osuCrypto::block> K0(depth);  // P_0's keys (sender for P_1's choice)
    std::vector<osuCrypto::block> K1(depth);  // P_1's keys (sender for P_0's choice)
    std::vector<osuCrypto::block> M0(depth);  // P_0's MACs (receiver with choice alpha0)
    std::vector<osuCrypto::block> M1(depth);  // P_1's MACs (receiver with choice alpha1)
    for (int i = 0; i < depth; i++) {
        const osuCrypto::u8 a0 = alpha0_bits[i] & 1;
        const osuCrypto::u8 a1 = alpha1_bits[i] & 1;
        // P_1 is sender for P_0's choice a0; P_0 receives M0[i] = K1[a0] ^ a0*D1.
        K1[i] = prng.get<osuCrypto::block>();
        M0[i] = K1[i] ^ (a0 ? D1 : osuCrypto::ZeroBlock);
        // P_0 is sender for P_1's choice a1; P_1 receives M1[i] = K0[a1] ^ a1*D0.
        K0[i] = prng.get<osuCrypto::block>();
        M1[i] = K0[i] ^ (a1 ? D0 : osuCrypto::ZeroBlock);
    }

    // --- Paper Algorithm 3: Pi_Gen (correlated-GGM key generation) ---
    //
    // Line 1: sample public coin W and set root shares <Delta>_b XOR W. Both
    // parties see the same W; the root XOR reconstructs to Delta (the two W
    // cancel). Root control-bit t_b = lsb(root_b).
    const osuCrypto::block W = prng.get<osuCrypto::block>();
    CggmKey k0, k1;
    k0.depth = depth;
    k1.depth = depth;
    k0.root = D0 ^ W;
    k1.root = D1 ^ W;
    k0.internal_cw.assign(depth > 0 ? depth - 1 : 0, osuCrypto::ZeroBlock);
    k1.internal_cw.assign(depth > 0 ? depth - 1 : 0, osuCrypto::ZeroBlock);

    // Internal levels 1..depth-1 (indices i=0..depth-2 in the code). Each party
    // expands all 2^i nodes at level i to 2^(i+1) nodes at level i+1, computes
    // its CW share for level i+1, sends it to the other party, and reconstructs
    // the public CW_{i+1} = CWshare_0 XOR CWshare_1. Then both descend.
    std::vector<osuCrypto::block> nodes0(1, k0.root);
    std::vector<osuCrypto::block> nodes1(1, k1.root);

    for (int i = 0; i < depth - 1; i++) {
        const int level = i + 1;  // paper level (root=0, first internal=1)
        const int idx = i;        // alpha bit index for this level (MS bit first)
        const osuCrypto::u8 a0 = alpha0_bits[idx] & 1;
        const osuCrypto::u8 a1 = alpha1_bits[idx] & 1;
        const osuCrypto::u8 abar0 = a0 ^ 1;  // literal per-share complement
        const osuCrypto::u8 abar1 = a1 ^ 1;

        // Line 3: each party computes its CW share = hash-sum XOR alpha-term XOR
        // COT-term, where the alpha-term is abar_b * D_b and the COT-term is
        // K[a_{1-b}] XOR M[a_b]. The hash-sum is the XOR of H_S over all nodes at
        // level i (off-path nodes are identical across parties and cancel when
        // reconstructed; the on-path node's hash contributes H(node0)^H(node1)).
        osuCrypto::block hsum0 = osuCrypto::ZeroBlock;
        osuCrypto::block hsum1 = osuCrypto::ZeroBlock;
        for (const auto& nd : nodes0) {
            hsum0 ^= dfss::internal::ccrHash(nd);
        }
        for (const auto& nd : nodes1) {
            hsum1 ^= dfss::internal::ccrHash(nd);
        }
        const osuCrypto::block cw_share0 =
            hsum0 ^ (abar0 ? D0 : osuCrypto::ZeroBlock) ^ K0[idx] ^ M0[idx];
        const osuCrypto::block cw_share1 =
            hsum1 ^ (abar1 ? D1 : osuCrypto::ZeroBlock) ^ K1[idx] ^ M1[idx];
        const osuCrypto::block cw = cw_share0 ^ cw_share1;
        k0.internal_cw[i] = cw;
        k1.internal_cw[i] = cw;

        // Line 4-5: expand children. For each parent (s||t), compute left child
        // (s'||t') := H_S(s||t) XOR t*CW, right child := left XOR (s||t).
        std::vector<osuCrypto::block> next0, next1;
        next0.reserve(nodes0.size() * 2);
        next1.reserve(nodes1.size() * 2);
        for (const auto& parent : nodes0) {
            const osuCrypto::u8 t = blockLsb(parent);
            const osuCrypto::block H = dfss::internal::ccrHash(parent);
            const osuCrypto::block corr = t ? cw : osuCrypto::ZeroBlock;
            const osuCrypto::block left = H ^ corr;
            const osuCrypto::block right = left ^ parent;
            next0.push_back(left);
            next0.push_back(right);
        }
        for (const auto& parent : nodes1) {
            const osuCrypto::u8 t = blockLsb(parent);
            const osuCrypto::block H = dfss::internal::ccrHash(parent);
            const osuCrypto::block corr = t ? cw : osuCrypto::ZeroBlock;
            const osuCrypto::block left = H ^ corr;
            const osuCrypto::block right = left ^ parent;
            next1.push_back(left);
            next1.push_back(right);
        }
        nodes0 = std::move(next0);
        nodes1 = std::move(next1);
    }

    // --- Leaf stage (Alg. 3 lines 6-9), the mu/d masking trick ---
    //
    // nodes0/nodes1 now hold the 2^(depth-1) penultimate nodes. H'_S = high
    // (lambda-1) bits of ccrHash (lsb cleared). All HCW/H'/XH terms live in the
    // high-bit domain (lsb=0); the leaf control bit is carried separately by LCW.
    auto Hhi = [](const osuCrypto::block& x) {
        return clearBlockLsb(dfss::internal::ccrHash(x));
    };
    const int lidx = depth - 1;  // alpha bit index for the leaf level
    const osuCrypto::u8 y0 = alpha0_bits[lidx] & 1;
    const osuCrypto::u8 y1 = alpha1_bits[lidx] & 1;

    // Line 6: XH_b[sigma] (high bits) and XL_b[sigma] (last bit), summed over all
    // penultimate nodes. node ^ sigma flips the lsb when sigma=1.
    osuCrypto::block XHsum0[2] = {osuCrypto::ZeroBlock, osuCrypto::ZeroBlock};
    osuCrypto::block XHsum1[2] = {osuCrypto::ZeroBlock, osuCrypto::ZeroBlock};
    for (const auto& nd : nodes0) {
        XHsum0[0] ^= dfss::internal::ccrHash(nd);
        XHsum0[1] ^= dfss::internal::ccrHash(nd ^ osuCrypto::OneBlock);
    }
    for (const auto& nd : nodes1) {
        XHsum1[0] ^= dfss::internal::ccrHash(nd);
        XHsum1[1] ^= dfss::internal::ccrHash(nd ^ osuCrypto::OneBlock);
    }
    osuCrypto::block XH0[2] = {clearBlockLsb(XHsum0[0]), clearBlockLsb(XHsum0[1])};
    osuCrypto::block XH1[2] = {clearBlockLsb(XHsum1[0]), clearBlockLsb(XHsum1[1])};
    osuCrypto::u8 XL0[2] = {blockLsb(XHsum0[0]), blockLsb(XHsum0[1])};
    osuCrypto::u8 XL1[2] = {blockLsb(XHsum1[0]), blockLsb(XHsum1[1])};

    // COT material for the leaf level (lidx): P_0 holds K[<a_d>_1]=K0[lidx] and
    // M[<a_d>_0]=M0[lidx]; P_1 holds K[<a_d>_0]=K1[lidx] and M[<a_d>_1]=M1[lidx].
    const osuCrypto::block KB0 = K0[lidx];  // K[<a_d>_1], held by P_0
    const osuCrypto::block MA0 = M0[lidx];  // M[<a_d>_0], held by P_0
    const osuCrypto::block KB1 = K1[lidx];  // K[<a_d>_0], held by P_1
    const osuCrypto::block MA1 = M1[lidx];  // M[<a_d>_1], held by P_1

    // Line 7: sample masks mu_b, compute d_b, reshare (mu_b, d_b).
    const osuCrypto::block mu0 = prng.get<osuCrypto::block>();
    const osuCrypto::block mu1 = prng.get<osuCrypto::block>();
    const osuCrypto::block d0v =
        Hhi(mu0 ^ KB0) ^ Hhi(mu0 ^ KB0 ^ D0) ^ XH0[0] ^ XH0[1];
    const osuCrypto::block d1v =
        Hhi(mu1 ^ KB1) ^ Hhi(mu1 ^ KB1 ^ D1) ^ XH1[0] ^ XH1[1];

    // Line 8: HCW shares (per-share literal complement y_b^1 selects XH index).
    const osuCrypto::block hcw0 =
        XH0[y0 ^ 1] ^ Hhi(mu0 ^ KB0) ^ Hhi(mu1 ^ MA0) ^
        (y0 ? d1v : osuCrypto::ZeroBlock);
    const osuCrypto::block hcw1 =
        XH1[y1 ^ 1] ^ Hhi(mu1 ^ KB1) ^ Hhi(mu0 ^ MA1) ^
        (y1 ? d0v : osuCrypto::ZeroBlock);
    const osuCrypto::block hcw = hcw0 ^ hcw1;

    // Line 8b: LCW[sigma] shares; b*{sigma=0} adds 1 to P_1's sigma=0 share.
    const osuCrypto::u8 lcw0_share0 =
        static_cast<osuCrypto::u8>(XL0[0] ^ y0);        // b=0, {sigma=0}*0
    const osuCrypto::u8 lcw0_share1 =
        static_cast<osuCrypto::u8>(XL1[0] ^ y1 ^ 1);    // b=1, {sigma=0}*1
    const osuCrypto::u8 lcw1_share0 =
        static_cast<osuCrypto::u8>(XL0[1] ^ y0);        // b=0
    const osuCrypto::u8 lcw1_share1 =
        static_cast<osuCrypto::u8>(XL1[1] ^ y1);        // b=1
    const osuCrypto::u8 lcw0 =
        static_cast<osuCrypto::u8>(lcw0_share0 ^ lcw0_share1);
    const osuCrypto::u8 lcw1 =
        static_cast<osuCrypto::u8>(lcw1_share0 ^ lcw1_share1);

    // Line 9: assemble the shared leaf correction word into both keys.
    k0.hcw = hcw;
    k1.hcw = hcw;
    k0.lcw[0] = lcw0;
    k0.lcw[1] = lcw1;
    k1.lcw[0] = lcw0;
    k1.lcw[1] = lcw1;

    return {k0, k1};
}

void setCggmThreadLimit(std::size_t threads) { segmentExecutor().configure(threads); }
std::size_t cggmThreadLimit() { return segmentExecutor().limit(); }

std::size_t CggmGenerationWorkspace::capacityBytes() const noexcept {
    return (nodes.capacity() + hashes.capacity()) * sizeof(osuCrypto::block) +
        (node_offsets.capacity() + hash_offsets.capacity() + counts.capacity()) * sizeof(std::size_t);
}

CggmKey cggmGen2pc(int party_id,
                   const std::vector<osuCrypto::u8>& alpha_bits,
                   PhaseTimer* timer) {
    std::vector<std::vector<osuCrypto::u8>> batch{alpha_bits};
    return cggmGen2pcBatch(party_id, batch, timer).front();
}

std::vector<CggmKey> cggmGen2pcBatch(
    int party_id,
    const std::vector<std::vector<osuCrypto::u8>>& alpha_segments,
    PhaseTimer* timer,
    CggmGenerationStats* stats, Peer* player, std::size_t participant_limit,
    CggmGenerationWorkspace* workspace) {
    if (!player) player = peer;
    if (party_id != SERVER && party_id != CLIENT) {
        throw std::invalid_argument(
            "cggmGen2pcBatch: party_id must be SERVER or CLIENT");
    }
    if (player == nullptr) {
        throw std::invalid_argument(
            "cggmGen2pcBatch: global peer must be set");
    }
    if (alpha_segments.empty()) {
        throw std::invalid_argument(
            "cggmGen2pcBatch: segment list must not be empty");
    }

    const std::size_t nseg = alpha_segments.size();
    const bool batched = runtimeFlag("SEGFSS_AES_OFFLINE");
    const bool merged_cw = runtimeFlag("SEGFSS_MERGE_LEAF_CW");
    CggmGenerationWorkspace owned_workspace;
    if (!workspace && runtimeFlag("SEGFSS_REUSE_GENERATION_WORKSPACE")) workspace = &owned_workspace;
    const auto product_before = player->bit_block_ledger;
    if (stats != nullptr) {
        *stats = {};
    }
    const int b = party_id - SERVER;  // 0 for SERVER, 1 for CLIENT
    int max_depth = 0;
    int internal_product_count = 0;
    std::vector<int> depths(nseg);
    std::vector<int> internal_offsets(nseg);
    for (std::size_t j = 0; j < nseg; ++j) {
        depths[j] = static_cast<int>(alpha_segments[j].size());
        if (depths[j] <= 0) {
            throw std::invalid_argument(
                "cggmGen2pcBatch: every segment depth must be > 0");
        }
        internal_offsets[j] = internal_product_count;
        internal_product_count += depths[j] - 1;
        max_depth = std::max(max_depth, depths[j]);
    }

    // Root generation (reference correlated_ggm.cpp:82-86): independent random root
    // per party with LSB set to party bit, so root_0 ^ root_1 = Δ has lsb=1.
    // The root serves as both the tree seed AND the delta share for multiplication.
    auto& local_rng = player->randomSource();
    std::vector<CggmKey> keys(nseg);
    std::vector<std::vector<osuCrypto::block>> nodes(workspace ? 0 : nseg);
    std::vector<std::vector<osuCrypto::block>> node_hashes(workspace ? 0 : nseg);
    std::size_t capacity_growths = 0;
    if (workspace) {
        std::size_t total_nodes = 0, total_hashes = 0;
        for (auto depth : depths) {
            if (depth >= int(8 * sizeof(std::size_t)))
                throw std::invalid_argument("cGGM expanded workspace depth overflows size_t");
            const auto leaves = std::size_t(1) << (depth - 1);
            if (leaves > (SIZE_MAX / sizeof(osuCrypto::block)) - total_nodes)
                throw std::invalid_argument("cGGM expanded workspace size overflows");
            total_nodes += leaves; total_hashes += leaves / 2;
        }
        auto resize = [&](auto& buffer, std::size_t size) {
            if (buffer.capacity() < size) ++capacity_growths;
            buffer.resize(size);
        };
        resize(workspace->nodes, total_nodes); resize(workspace->hashes, total_hashes);
        resize(workspace->node_offsets, nseg); resize(workspace->hash_offsets, nseg); resize(workspace->counts, nseg);
        std::size_t node_offset = 0, hash_offset = 0;
        for (std::size_t j = 0; j < nseg; ++j) {
            workspace->node_offsets[j] = node_offset; workspace->hash_offsets[j] = hash_offset;
            workspace->counts[j] = 1;
            const auto leaves = std::size_t(1) << (depths[j] - 1);
            node_offset += leaves; hash_offset += leaves / 2;
        }
    }
    auto node_count = [&](std::size_t j) { return workspace ? workspace->counts[j] : nodes[j].size(); };
    auto tree = [&](std::size_t j) { return workspace ? workspace->nodes.data() + workspace->node_offsets[j] : nodes[j].data(); };
    auto hashes_for = [&](std::size_t j) { return workspace ? workspace->hashes.data() + workspace->hash_offsets[j] : node_hashes[j].data(); };
    std::size_t internal_hashes = 0, participants = 1;
    for (std::size_t j = 0; j < nseg; ++j) {
        osuCrypto::block root = local_rng.get<osuCrypto::block>();
        root = setBlockLsb(root, static_cast<osuCrypto::u8>(b));
        keys[j].depth = depths[j];
        keys[j].root = root;
        keys[j].internal_cw.assign(depths[j] - 1,
                                   osuCrypto::ZeroBlock);
        if (workspace) tree(j)[0] = root;
        else nodes[j].push_back(root);
    }

    // Compute every segment's shares of ᾱ_i * Δ in one batched COT.  A
    // single cggmGen2pc call per segment would serialize this protocol stage.
    if (timer) timer->start("cot");
    std::vector<osuCrypto::u8> alpha_bar_b(internal_product_count);
    std::vector<osuCrypto::block> delta_b_vec(internal_product_count);
    std::vector<osuCrypto::block> alpha_bar_delta(internal_product_count);
    for (std::size_t j = 0; j < nseg; ++j) {
        for (int i = 0; i < depths[j] - 1; ++i) {
            const int flat = internal_offsets[j] + i;
            alpha_bar_b[flat] =
                (alpha_segments[j][i] & 1) ^ (1 - b);
            delta_b_vec[flat] = keys[j].root;
        }
    }
    if (internal_product_count > 0) {
        bitBlockMultiply(party_id, alpha_bar_b.data(), delta_b_vec.data(),
                         alpha_bar_delta.data(), internal_product_count, player);
    }
    if (timer) timer->stop("cot");

    // Internal levels execute in lockstep.  At level i every active segment
    // computes locally in parallel, followed by one batched CW opening.
    std::vector<std::size_t> active;
    std::vector<osuCrypto::block> cw_shares;
    active.reserve(nseg);
    cw_shares.reserve(nseg);
    for (int i = 0; i < max_depth - 1; ++i) {
        active.clear();
        std::size_t active_nodes = 0;
        for (std::size_t j = 0; j < nseg; ++j) {
            if (i < depths[j] - 1) {
                active.push_back(j);
                active_nodes += node_count(j);
            }
        }

        if (timer) timer->start("expand");
        cw_shares.resize(active.size());
        internal_hashes += active_nodes;
        participants = std::max(participants, parallelForSegments(active.size(), active_nodes, [&](std::size_t pos) {
            const std::size_t j = active[pos];
            osuCrypto::block hsum = osuCrypto::ZeroBlock;
            const auto count = node_count(j);
            if (!workspace) node_hashes[j].resize(count);
            auto* hashes = hashes_for(j);
            const auto* parents = tree(j);
            if (batched) ccrHashBatch(parents, count, hashes);
            for (std::size_t k = 0; k < count; ++k) {
                if (!batched) hashes[k] = dfss::internal::ccrHash(parents[k]);
                hsum ^= hashes[k];
            }
            cw_shares[pos] =
                hsum ^ alpha_bar_delta[internal_offsets[j] + i];
        }, participant_limit));
        if (timer) timer->stop("expand");

        if (timer) timer->start("cw_exchange");
        player->open_xor(cw_shares.data(), static_cast<int32_t>(cw_shares.size()));
        if (timer) timer->stop("cw_exchange");

        if (timer) timer->start("expand");
        participants = std::max(participants, parallelForSegments(active.size(), active_nodes, [&](std::size_t pos) {
            const std::size_t j = active[pos];
            const osuCrypto::block cw = cw_shares[pos];
            keys[j].internal_cw[i] = cw;

            const std::size_t parent_count = node_count(j);
            if (workspace) workspace->counts[j] = parent_count * 2;
            else nodes[j].resize(parent_count * 2);
            auto* descendants = tree(j);
            const auto* hashes = hashes_for(j);
            for (std::size_t k = parent_count; k-- > 0;) {
                const osuCrypto::block parent = descendants[k];
                const osuCrypto::u8 t = blockLsb(parent);
                const osuCrypto::block corr =
                    t ? cw : osuCrypto::ZeroBlock;
                const osuCrypto::block left = hashes[k] ^ corr;
                descendants[2 * k] = left;
                descendants[2 * k + 1] = left ^ parent;
            }
        }, participant_limit));
        if (timer) timer->stop("expand");
    }

    // Leaf computations are also segment-local, followed by one batched
    // multiplication and two batched openings (high blocks and low bits).
    if (timer) timer->start("leaf");
    std::vector<std::array<osuCrypto::block, 2>> xh(nseg);
    std::vector<std::array<osuCrypto::u8, 2>> xl(nseg);
    std::vector<osuCrypto::u8> leaf_choices(nseg);
    std::vector<osuCrypto::block> high_diffs(nseg);
    std::size_t leaf_nodes = 0;
    for (std::size_t j = 0; j < nseg; ++j) leaf_nodes += node_count(j);
    const auto leaf_participants = parallelForSegments(nseg, 2 * leaf_nodes, [&](std::size_t j) {
        const auto sums = hashLeafSums(tree(j), node_count(j), batched);
        xh[j][0] = clearBlockLsb(sums[0]);
        xh[j][1] = clearBlockLsb(sums[1]);
        xl[j][0] = blockLsb(sums[0]);
        xl[j][1] = blockLsb(sums[1]);

        const osuCrypto::u8 y_b =
            alpha_segments[j][depths[j] - 1] & 1;
        leaf_choices[j] = y_b ^ (1 - b);
        high_diffs[j] = xh[j][0] ^ xh[j][1];
    }, participant_limit);
    participants = std::max(participants, leaf_participants);

    std::vector<osuCrypto::block> high_selected_shares(nseg);
    bitBlockMultiply(party_id, leaf_choices.data(), high_diffs.data(),
                     high_selected_shares.data(),
                     static_cast<int32_t>(nseg), player);

    std::vector<osuCrypto::block> hcw(nseg);
    std::vector<osuCrypto::u8> lcw(2 * nseg);
    for (std::size_t j = 0; j < nseg; ++j) {
        hcw[j] = xh[j][0] ^ high_selected_shares[j];
        const osuCrypto::u8 y_b =
            alpha_segments[j][depths[j] - 1] & 1;
        lcw[2 * j] = xl[j][0] ^ y_b ^ static_cast<osuCrypto::u8>(b);
        lcw[2 * j + 1] = xl[j][1] ^ y_b;
    }
    if (merged_cw) player->open_xor_mixed(hcw.data(), static_cast<int32_t>(nseg),
                                        lcw.data(), static_cast<int32_t>(lcw.size()));
    else {
        player->open_xor(hcw.data(), static_cast<int32_t>(nseg));
        player->open_xor(lcw.data(), static_cast<int32_t>(lcw.size()));
    }

    for (std::size_t j = 0; j < nseg; ++j) {
        keys[j].hcw = clearBlockLsb(hcw[j]);
        keys[j].lcw[0] = lcw[2 * j] & 1;
        keys[j].lcw[1] = lcw[2 * j + 1] & 1;
    }
    if (timer) timer->stop("leaf");

    if (stats != nullptr) {
        *stats = {internal_hashes, 2 * leaf_nodes, participants,
                  batched ? leaf_participants * sizeof(AesLeafScratch) : 0, batched};
        if (workspace) stats->workspace_capacity_bytes = workspace->capacityBytes();
        else for (std::size_t j = 0; j < nseg; ++j)
            stats->workspace_capacity_bytes += (nodes[j].capacity() + node_hashes[j].capacity()) * sizeof(osuCrypto::block) +
                2 * sizeof(std::vector<osuCrypto::block>);
        stats->workspace_capacity_growths = workspace ? capacity_growths : 1;
        stats->leaf_cw_openings = merged_cw ? 1 : 2;
        stats->reusable_workspace = workspace != nullptr;
        stats->merged_leaf_cw = merged_cw;
        const auto& product_after = player->bit_block_ledger;
        stats->bit_block_products = product_after.products - product_before.products;
        stats->temporary_triples = product_after.temporary_triples_generated - product_before.temporary_triples_generated;
        stats->temporary_triples_consumed = product_after.temporary_triples_consumed - product_before.temporary_triples_consumed;
        stats->general_ot_64 = product_after.general_ot_64 - product_before.general_ot_64;
        stats->general_ot_128 = product_after.general_ot_128 - product_before.general_ot_128;
        stats->native_cot_128 = product_after.native_cot_128 - product_before.native_cot_128;
        stats->ef_openings = product_after.ef_openings - product_before.ef_openings;
    }

    return keys;
}

}  // namespace segfss
