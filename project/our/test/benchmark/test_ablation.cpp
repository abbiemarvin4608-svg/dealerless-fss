// E4 ablation TDD tests.
//
// Two ablation axes:
//   (i)  Half-Tree cGGM vs standard GGM  -> expansion hash-count difference
//   (ii) Grotto prefix-parity vs full-domain eval -> eval hash-count difference
//
// Correctness invariant: the full-domain evaluator must reconstruct the SAME
// comparison bit L=1{xhat<alpha} and equality bit e=1{xhat==alpha} as the
// prefix-parity evaluator. Only the cost (hash calls) differs.

#include "ablation.h"
#include "cggm.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "commons/types.h"
#include "mpc/comms.h"

using namespace segfss;
using namespace sci;

// Process-wide runtime globals required by the linked EzPC/SCI layer. This is
// a single-process plaintext test (dealer keys, no sockets), so these are only
// needed to satisfy the linker; no network runtime is initialized.
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

class ResultLog {
public:
    void check(const std::string& name, bool ok, const std::string& detail = "") {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
        if (!detail.empty()) std::cout << " " << detail;
        std::cout << "\n";
        failures_ += ok ? 0 : 1;
    }
    int failures() const { return failures_; }
private:
    int failures_ = 0;
};

// MS-first bit vector of `value` over `depth` bits.
static std::vector<osuCrypto::u8> bits_of(uint64_t value, int depth) {
    std::vector<osuCrypto::u8> out(depth);
    for (int i = 0; i < depth; i++) {
        out[i] = static_cast<osuCrypto::u8>((value >> (depth - 1 - i)) & 1);
    }
    return out;
}

// (ii) Correctness: full-domain eval reconstructs same L,e as prefix-parity,
// exhaustively over all (alpha, xhat) for small depth.
void test_full_domain_matches_prefix(ResultLog& log) {
    for (int depth = 1; depth <= 5; depth++) {
        const uint64_t N = 1ULL << depth;
        bool all_ok = true;
        for (uint64_t alpha = 0; alpha < N; alpha++) {
            auto [k0, k1] = cggmGenDealer(bits_of(alpha, depth));
            for (uint64_t x = 0; x < N; x++) {
                auto xb = bits_of(x, depth);
                CggmEval p0 = cggmEval(k0, xb);
                CggmEval p1 = cggmEval(k1, xb);
                CggmEval f0 = cggmEvalFullDomain(k0, xb, nullptr);
                CggmEval f1 = cggmEvalFullDomain(k1, xb, nullptr);

                const osuCrypto::u8 L_prefix = p0.L ^ p1.L;
                const osuCrypto::u8 e_prefix = p0.e ^ p1.e;
                const osuCrypto::u8 L_full = f0.L ^ f1.L;
                const osuCrypto::u8 e_full = f0.e ^ f1.e;

                if (L_full != L_prefix || e_full != e_prefix) {
                    all_ok = false;
                }
                // Also cross-check against ground truth.
                const osuCrypto::u8 L_truth = (x < alpha) ? 1 : 0;
                const osuCrypto::u8 e_truth = (x == alpha) ? 1 : 0;
                if (L_prefix != L_truth || e_prefix != e_truth) {
                    all_ok = false;
                }
            }
        }
        log.check("full_domain_matches_prefix depth=" + std::to_string(depth),
                  all_ok);
    }
}

// (ii) Cost: prefix-parity is O(log N) = depth+1 hashes; full-domain is O(N).
void test_eval_hash_counts(ResultLog& log) {
    for (int depth = 4; depth <= 10; depth += 2) {
        auto [k0, k1] = cggmGenDealer(bits_of(3, depth));
        auto xb = bits_of(1, depth);

        uint64_t prefix_hashes = 0;
        cggmEvalPrefixCounted(k0, xb, &prefix_hashes);

        uint64_t full_hashes = 0;
        cggmEvalFullDomain(k0, xb, &full_hashes);

        const uint64_t N = 1ULL << depth;
        // Prefix parity: depth-1 internal + 2 leaf hashes = depth+1.
        const bool prefix_ok = (prefix_hashes == static_cast<uint64_t>(depth + 1));
        // Full-domain expands all leaves: hashes >= N (order N).
        const bool full_ok = (full_hashes >= N);

        log.check("prefix_hashes==depth+1 depth=" + std::to_string(depth),
                  prefix_ok,
                  "got=" + std::to_string(prefix_hashes) +
                  " expected=" + std::to_string(depth + 1));
        log.check("full_hashes>=N depth=" + std::to_string(depth),
                  full_ok,
                  "got=" + std::to_string(full_hashes) +
                  " N=" + std::to_string(N));
    }
}

// (i) Half-Tree cGGM uses fewer expansion hashes than standard GGM.
void test_expansion_hash_counts(ResultLog& log) {
    for (int depth = 4; depth <= 12; depth += 2) {
        ExpansionCost cost = countExpansion(depth);
        // Half-Tree derives the right child by XOR (1 hash/node on internal
        // levels); standard GGM uses 2 hashes/node. Half-Tree must be cheaper.
        const bool ht_cheaper = cost.half_tree_hashes < cost.standard_ggm_hashes;
        log.check("half_tree<standard depth=" + std::to_string(depth),
                  ht_cheaper,
                  "HT=" + std::to_string(cost.half_tree_hashes) +
                  " STD=" + std::to_string(cost.standard_ggm_hashes));
    }
}

struct Case {
    int id;
    const char* name;
    void (*run)(ResultLog&);
};

Case cases[] = {
    {1, "full_domain_matches_prefix", test_full_domain_matches_prefix},
    {2, "eval_hash_counts", test_eval_hash_counts},
    {3, "expansion_hash_counts", test_expansion_hash_counts},
};

int main() {
    ResultLog log;
    const int n = sizeof(cases) / sizeof(cases[0]);
    std::cout << "=== E4 Ablation Tests ===\n";
    for (int i = 0; i < n; i++) {
        std::cout << "\nTest " << cases[i].id << ": " << cases[i].name << "\n";
        cases[i].run(log);
    }
    std::cout << "\n=== Summary ===\nTotal failures: " << log.failures() << "\n";
    return log.failures();
}
