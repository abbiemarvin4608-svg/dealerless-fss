#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Public call identity and immutable SCI integer table(s).  Exponent digits
 * use per-table groups; reciprocal coefficients use two outputs per input. */
typedef struct segfss_sci_dpf_call {
    int query, layer, slot, call, family, count, index_bits, tables;
    int output_bits[2];
    const uint64_t* values[2];
    int grouped_tables;
} segfss_sci_dpf_call;
typedef struct segfss_sci_dpf_counters {
    uint64_t sent_bytes, received_bytes, generation_epoch, resident_bytes;
    uint64_t generated_requests, consumed_requests;
    uint64_t mask_conversion_us, keygen_us, eval_all_us, offline_us;
    uint64_t online_open_us, inner_product_us, eval_points, online_open_batches;
} segfss_sci_dpf_counters;
void* segfss_sci_dpf_create(int sci_party, const char* host, int port);
/* Retained material metadata, excluding the packed selection vector. */
uint64_t segfss_sci_dpf_material_metadata_bytes(void);
int segfss_sci_dpf_prepare(void*, const segfss_sci_dpf_call*, int calls);
int segfss_sci_dpf_online(void*, const segfss_sci_dpf_call*, const uint64_t*, uint64_t*, uint64_t*);
int segfss_sci_dpf_finish_query(void*);
int segfss_sci_dpf_counters_get(void*, segfss_sci_dpf_counters*);
void segfss_sci_dpf_destroy(void*);
const char* segfss_sci_dpf_error(void);
#ifdef __cplusplus
}
#endif
