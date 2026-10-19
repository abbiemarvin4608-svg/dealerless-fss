#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// ABI v3: SCI party 1/2. Each context owns its channel and is called serially;
// independent contexts may be used concurrently on separate calling threads.
// Legacy FSS activation entrypoints still use the first/default context.
// Persistent sockets/base OTs; fresh one-use masks, keys and triples per call.
typedef void (*segfss_sci_round_io_callback)(int direction, int channel, uint64_t offset, uint64_t length);
typedef uint64_t (*segfss_sci_round_fork_callback)(void);
typedef uint64_t (*segfss_sci_round_actor_callback)(uint64_t actor);
typedef void (*segfss_sci_round_join_callback)(uint64_t child);
void segfss_sci_set_round_trace_callbacks(segfss_sci_round_io_callback io,
    segfss_sci_round_fork_callback fork_actor, segfss_sci_round_actor_callback enter_actor,
    segfss_sci_round_actor_callback leave_actor, segfss_sci_round_join_callback join_actor);
typedef uint64_t (*segfss_sci_round_scope_begin_callback)(const char* label, int query,
    int layer, int slot, int call, int kind, int stage);
typedef void (*segfss_sci_round_scope_end_callback)(uint64_t scope);
void segfss_sci_set_round_scope_callbacks(segfss_sci_round_scope_begin_callback begin,
    segfss_sci_round_scope_end_callback end);
// Caller-thread scopes nest by minimum and never change the process pool.
void* segfss_sci_cggm_scope_begin(int participants);
int segfss_sci_cggm_scope_stats(void* scope, int* effective_limit, int* observed_peak, int* transport_peak);
int segfss_sci_cggm_scope_end(void* scope);
typedef struct segfss_sci_stats {
    uint64_t calls, elements, segment_keys;
    double setup_ms, offline_ms, online_ms;
    double mask_conversion_ms,keygen_ms,scan_preprocessing_ms;
    double masked_input_open_ms,evaluation_ms,scan_ms;
    uint64_t sent_bytes, received_bytes, rounds;
    uint64_t offline_sent_bytes, offline_received_bytes, offline_rounds;
    uint64_t online_sent_bytes, online_received_bytes, online_rounds;
    uint64_t setup_sent_bytes, setup_received_bytes, setup_rounds;
    double activation_offline_ms, activation_online_ms;
    uint64_t activation_calls, activation_elements, activation_keys;
    double activation_lut_online_ms;
} segfss_sci_stats;
// SEGFSS_CGGM_THREADS optionally bounds local cGGM hashing workers.  It must
// be a positive integer; unset defaults to one worker.  Socket operations
// always remain on the calling thread.
void* segfss_sci_create(int sci_party, const char* address, int port);
// Internal C++ bridge sharing: borrowed Peer owned by this context.
void* segfss_sci_get_peer(void* context);
// in_i are additive shares mod 2^width; out_i are XOR shares of sign bit.
// width 2..64; nseg 1..width-1; never opens an input or an output.
int segfss_sci_msb(void* context, const uint64_t* in_i, uint8_t* out_i,
                   int count, int width, int nseg);
// Legacy API: prepare one fresh material object for every MSB call observed in
// the previous query.  New code should use segfss_sci_precompute_schedule,
// which does not require a query containing secret inputs and spills each
// material object to a temporary file.
int segfss_sci_precompute_recorded(void* context);
// Prepare `calls` fresh materials for a public, fixed call shape.  This API
// never reads `in_i` and can therefore be run before the input-dependent
// online query.  Materials are serialized one at a time to a temporary file;
// the following calls to segfss_sci_msb consume them in order.
int segfss_sci_precompute_schedule(void* context, int count, int width,
                                   int nseg, int calls);
// Validate that all scheduled materials have been consumed.  The destroy
// operation remains void for ABI compatibility; callers should invoke this
// first when using a scheduled run.
int segfss_sci_finish(void* context);
int segfss_sci_get_stats(void* context, segfss_sci_stats* out);
// Prepare a fixed public activation schedule. kind alternates sigmoid/tanh
// per recurrent step; elements_per_call is the per-sample hidden width, so
// each activation call consumes batch * elements_per_call fresh keys.
// Materials are generated before private inputs arrive.
int segfss_sci_precompute_activations(void* context, int batch, int steps,
                                      int elements_per_call, int bits);
// kind=0 sigmoid, kind=1 tanh. Inputs are additive shares mod 2^bits in Q7.
// Sigmoid returns Q14 shares mod 2^bits; tanh returns Q7 shares mod
// 2^(bits-5), after exact truncation. The internal contract is f=12/q=10.
int segfss_sci_activation(void* context, int kind, const uint64_t* input,
                          uint64_t* output, int count, int bits);
const char* segfss_sci_error(void);
void segfss_sci_destroy(void* context);
#ifdef __cplusplus
}
#endif
