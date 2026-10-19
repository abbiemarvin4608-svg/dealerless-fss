#pragma once

#include "common/test_common.h"

#include "segment.h"

// Correctness cases for our segment-decomposition protocol.
void check_split(ResultLog& log);
void check_gen(ResultLog& log);
void check_cggm(ResultLog& log);
void check_cggm_2pc(ResultLog& log);
void check_correlated_dcf(ResultLog& log);
void check_infer(ResultLog& log);
void check_bitxa(ResultLog& log);
void check_bit_block_modes(ResultLog& log);
void check_relu(ResultLog& log);
void check_activation(ResultLog& log);
void check_activation_batch(ResultLog& log);
void check_dpf_public_lut(ResultLog& log);
void check_exact_truncation(ResultLog& log);
void check_exact_mul_truncation(ResultLog& log);
void check_beaver(ResultLog& log);
void check_matmul(ResultLog& log);
