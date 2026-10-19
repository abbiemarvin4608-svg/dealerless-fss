#pragma once
#include <cstdint>
struct CnnCall { int kind, count, bits; };
extern "C" {
void* segfss_sci_cnn_create(int role, const char* host, int port);
int segfss_sci_cnn_prepare(void*, const CnnCall*, int count, uint64_t budget);
int segfss_sci_cnn_online(void*, const CnnCall*, const uint64_t*, const uint8_t*, const uint8_t*, uint64_t*, uint8_t*);
int segfss_sci_cnn_counters(void*, uint64_t* sent, uint64_t* received, uint64_t* spool_bytes);
int segfss_sci_cnn_finish(void*);
void segfss_sci_cnn_destroy(void*);
const char* segfss_sci_cnn_error();
}
