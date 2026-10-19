#pragma once
#include <cstdint>

struct segfss_sci_conversion_call {
    int count, input_bits, output_bits, shift, call, signed_arithmetic;
};
struct segfss_sci_conversion_counters {
    uint64_t sent_bytes, received_bytes, generation_epoch;
};
struct segfss_sci_conversion_metrics {
    double mask_ms, comparison_preprocessing_ms, rounding_preprocessing_ms;
    double masked_open_ms, comparison_online_ms, rounding_online_ms, b2a_online_ms;
    uint64_t masked_values, masked_bits, logical_rounds;
    double material_load_ms, query_prepare_ms, local_evaluation_ms, scan_ms, release_ms, total_online_ms;
    uint64_t material_bytes, comparison_keys, scan_and_gates, eval_participants;
    double material_preload_ms, prefetch_wait_ms;
    uint64_t contiguous_material, prefetched, resident_bytes, prefetch_budget_bytes;
    uint64_t keygen_internal_hashes, keygen_leaf_hashes, keygen_participants, keygen_aes_scratch_bytes;
    uint64_t eval_internal_hashes, eval_leaf_hashes, eval_aes_scratch_bytes, aes_online, aes_offline;
    uint64_t pruned_e_outputs, skipped_leaf_hashes, prune_equality, prune_leaf_hashes;
};
extern "C" {
void* segfss_sci_conversion_create(int sci_party, const char* host, int port,
                                  int full_tree, int max_segments);
int segfss_sci_conversion_prepare(void*, const segfss_sci_conversion_call*, int calls);
int segfss_sci_conversion_set_generation_resident(void*, uint64_t bytes);
int segfss_sci_conversion_set_prefetch_budget(void*, uint64_t bytes);
int segfss_sci_conversion_online(void*, const segfss_sci_conversion_call*,
                                const uint64_t*, uint64_t*, segfss_sci_conversion_metrics*);
int segfss_sci_conversion_finish_query(void*);
int segfss_sci_conversion_get_counters(void*, segfss_sci_conversion_counters*);
void segfss_sci_conversion_destroy(void*);
const char* segfss_sci_conversion_error();
}
