# Benchmark harness for E1–E4 experiments.

# E1/E2: Segmentation sweep benchmark
add_executable(nseg_sweep
    nseg_sweep.cpp)
target_link_libraries(nseg_sweep PRIVATE segfss)
target_compile_features(nseg_sweep PRIVATE cxx_std_17)

# SDCF-10: real two-party primitive cost sweep.  The Python harness starts one
# server/client pair per frozen row and retains raw logs for every status.
add_executable(primitive_cost_sweep
    primitive_cost_sweep.cpp)
target_link_libraries(primitive_cost_sweep PRIVATE segfss)
target_compile_features(primitive_cost_sweep PRIVATE cxx_std_17)

# Four independently configurable comparison pruning paths at the public
# two-party conversion seam, with the same causal trace as native runs.
add_executable(comparison_pruning_benchmark
    comparison_pruning_benchmark.cpp
    ../../tools/segmented_dcf_runtime/round_trace.cpp)
target_link_libraries(comparison_pruning_benchmark PRIVATE segfss)
target_compile_features(comparison_pruning_benchmark PRIVATE cxx_std_17)

# Matched four-way immediate product encoding/protocol and independent leaf-CW
# / generation-workspace ablations at the actual offline cGGM generation seam.
add_executable(cggm_generation_benchmark
    cggm_generation_benchmark.cpp
    ../../tools/segmented_dcf_runtime/round_trace.cpp)
target_link_libraries(cggm_generation_benchmark PRIVATE segfss)
target_compile_features(cggm_generation_benchmark PRIVATE cxx_std_17)

# One-use SIGMA folded-LUT activation cost (offline key generation vs online
# evaluation), measured at the same secret-share boundary as nseg_sweep.
add_executable(activation_benchmark
    activation_benchmark.cpp)
target_link_libraries(activation_benchmark PRIVATE segfss)
target_compile_features(activation_benchmark PRIVATE cxx_std_17)

add_executable(relu_batch_benchmark
    relu_batch_benchmark.cpp)
target_link_libraries(relu_batch_benchmark PRIVATE segfss)
target_compile_features(relu_batch_benchmark PRIVATE cxx_std_17)

# E3: Phase breakdown (instrumented version of nseg_sweep)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/phase_breakdown.cpp")
    add_executable(phase_breakdown
        phase_breakdown.cpp)
    target_link_libraries(phase_breakdown PRIVATE segfss)
    target_compile_features(phase_breakdown PRIVATE cxx_std_17)
endif()

# E4: Ablation study (compile-time switches for Half-Tree vs GGM, Grotto vs full-domain)
# Uses the same nseg_sweep driver but with recompiled segfss under different config flags.
# To run: cmake -DUSE_STANDARD_GGM=ON .. && make && ./nseg_sweep ...

# Instrumentation unit tests
add_executable(test_instrumentation
    test_instrumentation.cpp
    ../../src/instrumentation.cpp)
target_include_directories(test_instrumentation PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../src")
target_compile_features(test_instrumentation PRIVATE cxx_std_17)
add_test(NAME instrumentation COMMAND test_instrumentation)

# E4 ablation unit tests (single-process, plaintext dealer keys)
add_executable(test_ablation
    test_ablation.cpp)
target_include_directories(test_ablation PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../src")
target_link_libraries(test_ablation PRIVATE segfss)
target_compile_features(test_ablation PRIVATE cxx_std_17)
add_test(NAME ablation COMMAND test_ablation)

# Correlated-DPF control-word benchmark (migrated from dealerless-FSS_public,
# Apache-2.0; see dfss_ext/LICENSE). Enable with -DBUILD_DFSS_EXT_BENCHMARKS=ON.
if(BUILD_DFSS_EXT_BENCHMARKS)
    set(_dfss_ext_dir "${CMAKE_CURRENT_LIST_DIR}/dfss_ext")
    add_executable(DFSS_EXT_BENCH
        dfss_ext/benchmark.cpp
        dfss_ext/helper/cli.cpp
        dfss_ext/helper/common.cpp
        dfss_ext/helper/report.cpp
        dfss_ext/protocol/buildingblocks.cpp
        dfss_ext/protocol/dpf.cpp
        dfss_ext/protocol/et.cpp
        dfss_ext/microbench/payload_conversion.cpp
        dfss_ext/microbench/cw_generation.cpp)
    target_include_directories(DFSS_EXT_BENCH PRIVATE
        "${_dfss_ext_dir}"
        "${_dfss_ext_dir}/helper"
        "${_dfss_ext_dir}/protocol")
    target_link_libraries(DFSS_EXT_BENCH PRIVATE dfss)
endif()
