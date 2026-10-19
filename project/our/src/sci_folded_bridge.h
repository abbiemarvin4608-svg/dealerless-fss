#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
enum segfss_folded_kind { SEGFSS_FOLDED_SIGMOID = 0, SEGFSS_FOLDED_TANH = 1 };
enum segfss_folded_lut { SEGFSS_LUT_FULL_Q_ARITHMETIC = 0, SEGFSS_LUT_SEGMENTED_BOOLEAN_TENSOR_B2A = 1 };
enum segfss_folded_sign { SEGFSS_SIGN_BITDEC = 0, SEGFSS_SIGN_SEGMENTED_FSS = 1 };
#define SEGFSS_FOLDED_STAGE_COUNT 18
typedef struct segfss_sci_folded_call {
    int query, layer, slot, call, kind, count;
    int input_bits, output_bits, input_scale, output_scale;
    int lut_family, sign_backend, sign_segments;
    uint64_t table_version;
} segfss_sci_folded_call;
typedef struct segfss_sci_folded_stage {
    uint64_t elapsed_us, sent_bytes, received_bytes, begin_us, end_us;
} segfss_sci_folded_stage;
typedef struct segfss_sci_folded_metrics {
    segfss_sci_folded_stage offline[SEGFSS_FOLDED_STAGE_COUNT], online[SEGFSS_FOLDED_STAGE_COUNT];
    uint64_t offline_wall_us, online_wall_us, resident_bytes, peak_working_bytes;
    uint64_t online_generated, online_loaded_bytes;
} segfss_sci_folded_metrics;
typedef struct segfss_sci_folded_counters {
    uint64_t sent_bytes, received_bytes, generation_epoch, resident_bytes;
    uint64_t generated_requests, consumed_requests, generated_elements, consumed_elements;
    uint64_t online_generated, online_loaded_bytes;
    uint64_t public_table_prepare_us, public_table_resident_bytes;
    uint64_t public_table_prepare_sent_bytes, public_table_prepare_received_bytes;
} segfss_sci_folded_counters;
void* segfss_sci_folded_create(int sci_party, const char* host, int port, uint64_t context_identity);
int segfss_sci_folded_prepare(void*, const segfss_sci_folded_call*, int count);
// Preparation snapshots SEGFSS_FOLDED_REUSE_SIGN=0|1 (default 0) and
// SEGFSS_FOLDED_CLIP_BITS=32|22 (default 32). Both preserve the frozen native
// signed16/Q7 mapping and are supported with either sign/LUT backend.
int segfss_sci_folded_prepared_metrics(void*, int call, segfss_sci_folded_metrics*);
int segfss_sci_folded_online(void*, const segfss_sci_folded_call*, const uint64_t*, uint64_t*, segfss_sci_folded_metrics*);
int segfss_sci_folded_finish_query(void*);
int segfss_sci_folded_counters_get(void*, segfss_sci_folded_counters*);
void segfss_sci_folded_destroy(void*);
const char* segfss_sci_folded_error(void);
#ifdef __cplusplus
}
#endif
