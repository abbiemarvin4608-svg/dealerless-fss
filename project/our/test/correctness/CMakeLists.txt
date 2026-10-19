cmake_minimum_required(VERSION 3.13)

# Correctness driver for our segment-decomposition protocol.
# Follows the reference dealerless-FSS_public two-party runtime pattern:
# both parties run the same binary, r=2 server / r=3 client.

add_executable(SEGFSS_CORRECTNESS_TEST
    correctness.cpp
    cases_split.cpp
    cases_gen.cpp
    cases_cggm.cpp
    cases_cggm_2pc.cpp
    cases_correlated_dcf.cpp
    cases_infer.cpp
    cases_bitxa.cpp
    cases_relu.cpp
    cases_activation.cpp
    cases_batch.cpp
    cases_truncation.cpp
    cases_mul_truncation.cpp
    cases_beaver.cpp
    cases_matmul.cpp)
target_sources(SEGFSS_CORRECTNESS_TEST PRIVATE cases_dpf_lut.cpp)
target_compile_features(SEGFSS_CORRECTNESS_TEST PRIVATE cxx_std_17)
target_include_directories(SEGFSS_CORRECTNESS_TEST PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/../../src")
target_link_libraries(SEGFSS_CORRECTNESS_TEST PRIVATE segfss)

add_executable(SCI_NUMERIC_ORACLE_TEST test_sci_numeric_oracle.cpp)
target_compile_features(SCI_NUMERIC_ORACLE_TEST PRIVATE cxx_std_17)
target_include_directories(SCI_NUMERIC_ORACLE_TEST PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../src")
add_test(NAME sci_numeric_oracle COMMAND SCI_NUMERIC_ORACLE_TEST)

add_executable(SCI_MODEL_BIAS_TEST test_model_bias.cpp)
target_compile_features(SCI_MODEL_BIAS_TEST PRIVATE cxx_std_17)
target_include_directories(SCI_MODEL_BIAS_TEST PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../src")
target_link_libraries(SCI_MODEL_BIAS_TEST PRIVATE segfss)
add_test(NAME sci_model_bias COMMAND SCI_MODEL_BIAS_TEST)

add_executable(SEGMENTED_DCF_TEST test_segmented_dcf.cpp)
target_link_libraries(SEGMENTED_DCF_TEST PRIVATE segfss)
add_test(NAME segmented_dcf_local COMMAND SEGMENTED_DCF_TEST)

add_executable(SCI_TRUNCATION_TEST test_sci_truncation.cpp)
target_link_libraries(SCI_TRUNCATION_TEST PRIVATE segfss)
add_test(NAME sci_truncation_config COMMAND SCI_TRUNCATION_TEST)

add_executable(RING_PACKING_WIRE_TEST test_ring_packing_wire.cpp)
find_package(Threads REQUIRED)
target_compile_features(RING_PACKING_WIRE_TEST PRIVATE cxx_std_17)
target_include_directories(RING_PACKING_WIRE_TEST PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../src")
target_link_libraries(RING_PACKING_WIRE_TEST PRIVATE Threads::Threads)
add_test(NAME packed_ring_wire COMMAND RING_PACKING_WIRE_TEST)

add_executable(ACTIVATION_BATCH_CONTRACT_TEST test_activation_batch_contract.cpp)
target_sources(ACTIVATION_BATCH_CONTRACT_TEST PRIVATE
    ../../src/folded_native.cpp ../../src/folded_tanh_native.cpp)
target_link_libraries(ACTIVATION_BATCH_CONTRACT_TEST PRIVATE segfss)

add_executable(DPF_LUT_BATCH_TEST test_dpf_lut_batch.cpp)
target_compile_features(DPF_LUT_BATCH_TEST PRIVATE cxx_std_17)
target_include_directories(DPF_LUT_BATCH_TEST PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../src")
target_link_libraries(DPF_LUT_BATCH_TEST PRIVATE segfss)
