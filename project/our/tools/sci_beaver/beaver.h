#pragma once
#include "LinearOT/linear-ot.h"
#include <cstdint>
#include <vector>

namespace sdcf_beaver {

struct MatmulOptions {
    bool algebra=false;
    bool mapped_layout=false;
};

struct MatmulTiming {
    double local_kernel_seconds=0;
    double compute_seconds=0;
    double layout_copy_seconds=0;
    double buffer_seconds=0;
    uint64_t local_products=0;
    bool copy_instrumented=false;
};

struct MatmulWorkspace {
    std::vector<uint64_t> opened, remote, product, combined_rhs, accumulation;
};

struct MatmulGenerationEstimate {
    uint64_t working_bytes=0,resident_bytes=0;
};
MatmulGenerationEstimate estimate_matmul_generation(int rows,int inner,int cols,int bits,int tile_rows);

struct MatmulMaterial {
    int rows=0, inner=0, cols=0, bits=0;
    std::vector<uint64_t> u, v, w;
    uint64_t offline_tiles=0, opened_values=0;
    int offline_tile_rows=32,offline_tail_rows=0;
    uint64_t offline_working_bytes=0,offline_resident_bytes=0;
    uint64_t offline_sent_bytes=0,offline_received_bytes=0,offline_io_rounds=0;
    bool offline_receive_instrumented=false;
    MatmulOptions options;
    MatmulTiming offline_local, online_local;
    bool consumed=false;
    MatmulMaterial() = default;
    MatmulMaterial(const MatmulMaterial&) = delete;
    MatmulMaterial& operator=(const MatmulMaterial&) = delete;
    MatmulMaterial(MatmulMaterial&&) = default;
    MatmulMaterial& operator=(MatmulMaterial&&) = default;
};

// A vector Beaver triple for element-wise (MulCir) multiplication.  The
// shares are held in the product ring; each query consumes the material once.
struct MulCirMaterial {
    int elements=0, bits=0;
    std::vector<uint64_t> u, v, w;
    uint64_t opened_values=0;
    bool consumed=false;
    MulCirMaterial() = default;
    MulCirMaterial(const MulCirMaterial&) = delete;
    MulCirMaterial& operator=(const MulCirMaterial&) = delete;
    MulCirMaterial(MulCirMaterial&&) = default;
    MulCirMaterial& operator=(MulCirMaterial&&) = default;
};

// A Beaver triple whose first operand is one private scalar and whose second
// operand is a private vector.  The scalar share is repeated only while
// generating/using the vector triple; it is opened once per query.
struct ScalarMulMaterial {
    int elements=0, bits=0;
    uint64_t u=0;
    std::vector<uint64_t> v, w;
    uint64_t opened_values=0;
    bool consumed=false;
    ScalarMulMaterial() = default;
    ScalarMulMaterial(const ScalarMulMaterial&) = delete;
    ScalarMulMaterial& operator=(const ScalarMulMaterial&) = delete;
    ScalarMulMaterial(ScalarMulMaterial&&) = default;
    ScalarMulMaterial& operator=(ScalarMulMaterial&&) = default;
};

int accumulation_bits(int inner);
MatmulMaterial prepare_matmul(int role, LinearOT& linear, int rows, int inner,
                             int cols, int bits, int offline_tile_rows=32,
                             MatmulOptions options={});
std::vector<uint64_t> multiply_matmul(int role, sci::IOPack& io, LinearOT& linear,
    const std::vector<uint64_t>& lhs, const std::vector<uint64_t>& rhs,
    MatmulMaterial& material, MatmulWorkspace* workspace=nullptr);
void multiply_matmul(int role, sci::IOPack& io, LinearOT& linear,
    const std::vector<uint64_t>& lhs, const std::vector<uint64_t>& rhs,
    MatmulMaterial& material, std::vector<uint64_t>& output,
    MatmulWorkspace* workspace=nullptr);

MulCirMaterial prepare_mulcir(int role, LinearOT& linear, int elements, int bits);
std::vector<uint64_t> multiply_mulcir(int role, sci::IOPack& io, LinearOT& linear,
    const std::vector<uint64_t>& lhs, const std::vector<uint64_t>& rhs,
    MulCirMaterial& material);

ScalarMulMaterial prepare_scalarmul(int role, LinearOT& linear, int elements, int bits);
std::vector<uint64_t> multiply_scalarmul(int role, sci::IOPack& io, LinearOT& linear,
    const std::vector<uint64_t>& scalar, const std::vector<uint64_t>& rhs,
    ScalarMulMaterial& material);

}
