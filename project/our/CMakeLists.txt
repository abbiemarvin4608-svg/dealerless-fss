cmake_minimum_required(VERSION 3.13)
project(DealerlessSegmentFSS LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Dealerless 32-bit segment-decomposition FSS (our protocol).
#
# Builds against the reference dealerless-FSS_public library (target `dfss`),
# which itself builds against an EzPC checkout. Layout on this machine:
#   project/our                     <- this project
#   project/dealerless-FSS_public   <- reference library (sibling)
#   /home/jiang/EzPC                <- EzPC checkout (NOT a sibling)
#
# Configure with:
#   cmake -S . -B build \
#       -DDEALERLESS_FSS_ROOT=../dealerless-FSS_public \
#       -DEZPC_ROOT=/home/jiang/EzPC \
#       -DCMAKE_BUILD_TYPE=Release
# ---------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Locate the reference library.
get_filename_component(_default_ref "${CMAKE_CURRENT_LIST_DIR}/../dealerless-FSS_public" ABSOLUTE)
set(DEALERLESS_FSS_ROOT "${_default_ref}" CACHE PATH
    "Path to the dealerless-FSS_public reference checkout")

if(NOT EXISTS "${DEALERLESS_FSS_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "dealerless-FSS_public not found at '${DEALERLESS_FSS_ROOT}'. "
        "Set -DDEALERLESS_FSS_ROOT=/path/to/dealerless-FSS_public.")
endif()

# Default the EzPC root the reference library needs.
if(NOT EZPC_ROOT)
    set(EZPC_ROOT "/home/jiang/EzPC" CACHE PATH
        "Path to the EzPC checkout used by dealerless-FSS_public")
endif()

# Pull in the reference library without building its tests.
set(DEALERLESS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(_dealerless_fss_build_dir "${CMAKE_CURRENT_BINARY_DIR}/dealerless-fss")
add_subdirectory("${DEALERLESS_FSS_ROOT}" "${_dealerless_fss_build_dir}")

# Paper-owned dFSS sources live in src/dfss (mirroring the reference src/
# layout). The reference library is copied into its build overlay during
# configuration; replace only the files changed by our paper experiments after
# that copy, so the public checkout stays an unmodified dependency.
set(_paper_dfss_src "${CMAKE_CURRENT_LIST_DIR}/src/dfss")
set(_paper_dfss_overlay "${_dealerless_fss_build_dir}/fss-overlay-src")
set(_paper_dfss_overrides
    commons/keypack.h
    deps/OT/split-iknp.h
    fss/dpf.cpp
    fss/dpf.h
    fss/internal/correlated_ggm.cpp
    fss/internal/correlated_ggm.h
    fss/internal/ggm.cpp
    mpc/comms.cpp
    mpc/comms.h
    utils/net_io_channel.h
    utils/round_trace.h
    mpc/secure_ops.cpp)
foreach(_paper_dfss_file IN LISTS _paper_dfss_overrides)
    if(_paper_dfss_file STREQUAL "mpc/comms.cpp")
        set(_paper_dfss_destination "comms.cpp")
    elseif(_paper_dfss_file STREQUAL "deps/OT/split-iknp.h")
        set(_paper_dfss_destination "OT/split-iknp.h")
    else()
        set(_paper_dfss_destination "${_paper_dfss_file}")
    endif()
    get_filename_component(_paper_dfss_destination_dir
        "${_paper_dfss_destination}" DIRECTORY)
    if(_paper_dfss_destination_dir)
        file(MAKE_DIRECTORY
            "${_paper_dfss_overlay}/${_paper_dfss_destination_dir}")
    endif()
    configure_file(
        "${_paper_dfss_src}/${_paper_dfss_file}"
        "${_paper_dfss_overlay}/${_paper_dfss_destination}"
        COPYONLY)
endforeach()

# ---------------------------------------------------------------------------
# Our protocol library.
# ---------------------------------------------------------------------------
add_library(segfss STATIC
    src/segment.cpp
    src/segmented_dcf.cpp
    src/sci_truncation.cpp
    src/gen.cpp
    src/cggm.cpp
    src/linear.cpp
    src/activation.cpp
    src/activation_batch.cpp
    src/dpf_lut.cpp
    src/exact_truncation.cpp
    src/relu.cpp
    src/instrumentation.cpp
    src/ablation.cpp)
target_sources(segfss PRIVATE src/model_bias.cpp)
target_compile_features(segfss PUBLIC cxx_std_17)
target_include_directories(segfss PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/src")
target_link_libraries(segfss PUBLIC dfss)

# ---------------------------------------------------------------------------
# Tests (two-party executables that follow the reference runtime pattern).
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(test/correctness)

# ---------------------------------------------------------------------------
# Benchmarks (E1/E2 segmentation sweep, E3 phase breakdown, optional
# correlated-DPF control-word benchmark in test/benchmark/dfss_ext).
# ---------------------------------------------------------------------------
option(BUILD_DFSS_EXT_BENCHMARKS
    "Build the migrated correlated-DPF control-word benchmark" OFF)
add_subdirectory(test/benchmark)

# Hidden runtime globals permit SCI (party 1/2) and dFSS (party 2/3) to coexist.
set_target_properties(segfss PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_sources(segfss PRIVATE src/private_msb.cpp)
add_library(segfss_sci_bridge SHARED src/sci_msb_bridge.cpp src/sci_conversion_bridge.cpp src/sci_dpf_bridge.cpp
    src/folded_native.cpp src/folded_tanh_native.cpp src/sci_folded_bridge.cpp)
target_link_libraries(segfss_sci_bridge PRIVATE segfss)
target_link_options(segfss_sci_bridge PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/tools/sci_bridge/exports.map"
    "-Wl,-Bsymbolic")
# Patch only a build-tree copy to exercise the actual SCI MSB dispatch seam.
file(READ "${EZPC_ROOT}/SCI/src/BuildingBlocks/aux-protocols.cpp" _sci_aux_source)
string(REPLACE "  assert(bw_x <= 64);\n  int32_t shift = bw_x - 1;"
    "  if (sci_segmented::msb(party, x, msb_x, size, bw_x)) return;\n  assert(bw_x <= 64);\n  int32_t shift = bw_x - 1;"
    _sci_aux_source "${_sci_aux_source}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sci_aux_msb_test.cpp"
    "#include \"sci_hook.h\"\n${_sci_aux_source}")
add_executable(SCI_SEGFSS_MSB_TEST tools/sci_bridge/test_msb.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/sci_aux_msb_test.cpp")
target_include_directories(SCI_SEGFSS_MSB_TEST PRIVATE src tools/sci_bridge)
target_link_libraries(SCI_SEGFSS_MSB_TEST PRIVATE segfss_sci_bridge dealerless_sci_headers)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(SCI_SEGFSS_MSB_TEST PRIVATE OpenMP::OpenMP_CXX)
endif()
# Oracle exercises only AuxProtocols::MSB; discard unrelated GC/truncation APIs.
target_compile_options(SCI_SEGFSS_MSB_TEST PRIVATE -ffunction-sections -fdata-sections)
target_link_options(SCI_SEGFSS_MSB_TEST PRIVATE -Wl,--gc-sections)

# Small regression for input-independent FSS schedule spooling.  It uses the
# bridge directly so it can validate serialization, shape/exhaustion checks,
# and reconstructed MSB outputs without running a complete network.
add_executable(SCI_SEGFSS_PRECOMPUTE_TEST tools/sci_bridge/test_precompute.cpp)
target_include_directories(SCI_SEGFSS_PRECOMPUTE_TEST PRIVATE src tools/sci_bridge)
target_link_libraries(SCI_SEGFSS_PRECOMPUTE_TEST PRIVATE segfss_sci_bridge dealerless_sci_headers)
add_test(NAME sci_cggm_thread_config COMMAND SCI_SEGFSS_PRECOMPUTE_TEST)

add_executable(SCI_SEGFSS_ROUND_TRACE_TEST tools/sci_bridge/test_round_trace.cpp)
target_include_directories(SCI_SEGFSS_ROUND_TRACE_TEST PRIVATE src tools/sci_bridge)
target_link_libraries(SCI_SEGFSS_ROUND_TRACE_TEST PRIVATE segfss_sci_bridge dealerless_sci_headers)

add_executable(SCI_DPF_ROUND_TRACE_TEST tools/sci_bridge/test_dpf_round_trace.cpp)
target_include_directories(SCI_DPF_ROUND_TRACE_TEST PRIVATE src)
target_link_libraries(SCI_DPF_ROUND_TRACE_TEST PRIVATE segfss_sci_bridge dealerless_sci_headers)

add_executable(SCI_SEGFSS_ROUND_ACTOR_TEST tools/sci_bridge/test_round_actor.cpp)
target_compile_features(SCI_SEGFSS_ROUND_ACTOR_TEST PRIVATE cxx_std_17)
target_include_directories(SCI_SEGFSS_ROUND_ACTOR_TEST PRIVATE src)
add_test(NAME sci_round_trace_actor COMMAND SCI_SEGFSS_ROUND_ACTOR_TEST)

add_executable(SCI_SEGFSS_CONVERSION_TEST tools/sci_conversion/test_bridge.cpp)
target_include_directories(SCI_SEGFSS_CONVERSION_TEST PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(SCI_SEGFSS_CONVERSION_TEST PRIVATE segfss_sci_bridge dealerless_sci_headers Threads::Threads)
