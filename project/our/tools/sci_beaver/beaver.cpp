#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif
#include "beaver.h"
#include "../../src/sci_numeric_oracle.h"
#include "../../src/ring_packing.h"
#include "../segmented_dcf_runtime/matrix_metrics.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <limits>
#include <stdexcept>

namespace sdcf_beaver {
namespace {
using segfss::reference::mask;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
void local_matmul(LinearOT& linear,int rows,int inner,int cols,
                  const uint64_t* lhs,const uint64_t* rhs,uint64_t* output,
                  bool mapped_layout,MatmulTiming& timing) {
    const auto start=Clock::now();
    const auto before=sdcf_local_matrix::metrics;
    if(mapped_layout) {
        using Matrix=Eigen::Matrix<uint64_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        const Eigen::Map<const Matrix> a(lhs,rows,inner),b(rhs,inner,cols);
        Eigen::Map<Matrix> c(output,rows,cols);
        // The caller owns disjoint scratch, so Eigen may write directly.
        c.noalias()=a*b;
    } else {
        linear.matmul_cleartext(rows,inner,cols,const_cast<uint64_t*>(lhs),
                                const_cast<uint64_t*>(rhs),output,true);
    }
    const auto kernel_seconds=elapsed(start);
    timing.local_kernel_seconds+=kernel_seconds;
    if(mapped_layout) {
        timing.copy_instrumented=true;
        timing.compute_seconds+=kernel_seconds;
    } else if(sdcf_local_matrix::metrics.calls!=before.calls) {
        timing.copy_instrumented=true;
        timing.compute_seconds+=sdcf_local_matrix::metrics.compute_seconds-before.compute_seconds;
        timing.layout_copy_seconds+=sdcf_local_matrix::metrics.layout_copy_seconds-before.layout_copy_seconds;
    }
    ++timing.local_products;
}
void validate_role(int role, const LinearOT& linear) {
    if ((role!=sci::ALICE && role!=sci::BOB) || role!=linear.party)
        throw std::invalid_argument("Beaver requires the matching SCI role");
}
void validate_shape(int rows,int inner,int cols,int bits) {
    mask(bits);
    const auto limit=uint64_t(std::numeric_limits<int>::max())/64;
    if(rows<1 || inner<1 || cols<1 || uint64_t(inner)*cols>limit ||
       uint64_t(rows)>limit/(uint64_t(inner)*cols))
        throw std::invalid_argument("matrix shape exceeds bounded SCI arithmetic");
}
void validate_vector(int elements, int bits) {
    mask(bits);
    if (elements < 1 || uint64_t(elements) > uint64_t(std::numeric_limits<int>::max()) / 8)
        throw std::invalid_argument("vector shape exceeds bounded SCI arithmetic");
}

template<class IO> auto received_bytes(const IO& io, int) -> decltype(io.recv_counter) {
    return io.recv_counter;
}
template<class IO> uint64_t received_bytes(const IO&, long) { return 0; }
template<class IO> auto receives_instrumented(const IO& io, int) -> decltype(io.recv_counter, bool()) {
    return true;
}
template<class IO> bool receives_instrumented(const IO&, long) { return false; }
uint64_t received_bytes(const sci::IOPack& io) {
    return received_bytes(*io.io,0)+received_bytes(*io.io_rev,0)+received_bytes(*io.io_GC,0);
}

void open_values(int role, sci::IOPack& io, const std::vector<uint64_t>& local,
                 uint64_t ring_mask, std::vector<uint64_t>& opened,
                 std::vector<uint64_t>* workspace_remote = nullptr) {
    opened = local;
    std::vector<uint64_t> local_remote;
    auto& remote = workspace_remote ? *workspace_remote : local_remote;
    remote.resize(local.size());
    const int bits = int(__builtin_popcountll(ring_mask));
    const bool packed = [] {
        const char* value = std::getenv("SEGFSS_PACK_RING");
        return value && std::string(value) == "1";
    }();
    if (packed && bits < 1)
        throw std::invalid_argument("invalid Beaver opening ring width");
    if (packed) {
        remote = segfss::exchangeRingValues(role == sci::ALICE, local, bits,
            [&](const uint8_t* data, int size) { io.io->send_data(data, size); },
            [&](uint8_t* data, int size) { io.io->recv_data(data, size); });
    } else {
        const auto bytes = local.size() * sizeof(uint64_t);
        if (role == sci::ALICE) {
            io.io->send_data(local.data(), bytes);
            io.io->recv_data(remote.data(), bytes);
        } else {
            io.io->recv_data(remote.data(), bytes);
            io.io->send_data(local.data(), bytes);
        }
    }
    for (size_t i = 0; i < opened.size(); ++i)
        opened[i] = (opened[i] + remote[i]) & ring_mask;
}
}

int accumulation_bits(int inner) {
    if(inner<1) throw std::invalid_argument("matrix inner dimension must be positive");
    int depth=0;
    for(unsigned value=unsigned(inner)-1;value;value>>=1) ++depth;
    return depth;
}

MatmulGenerationEstimate estimate_matmul_generation(int rows,int inner,int cols,int bits,int tile_rows) {
    validate_shape(rows,inner,cols,bits);
    if(tile_rows<1) throw std::invalid_argument("offline row tile must be positive");
    const uint64_t tile=std::min(rows,tile_rows);
    MatmulGenerationEstimate estimate;
    estimate.resident_bytes=(uint64_t(rows)*inner+uint64_t(inner)*cols+uint64_t(rows)*cols)*sizeof(uint64_t);
    for(bool row_batching:{false,true}) {
        const uint64_t dim=row_batching?tile*inner:uint64_t(inner)*cols;
        const uint64_t messages=row_batching?cols:tile;
        const uint64_t payload=dim*messages;
        // Match SCI's public MAX_NUM_OT batching, with one extra dimension
        // batch for float-ceil slack and bounded COT/packing scratch.
        const uint64_t max_ot=(((uint64_t(1)<<24)+payload-1)/payload+1)*dim;
        const uint64_t batch=std::min(dim*bits,max_ot);
        const uint64_t working=batch*messages*7*sizeof(uint64_t)+batch*32+
            (tile*inner+uint64_t(inner)*cols+tile*cols)*2*sizeof(uint64_t)+(2ULL<<20);
        estimate.working_bytes=std::max(estimate.working_bytes,working);
    }
    return estimate;
}

MatmulMaterial prepare_matmul(int role,LinearOT& linear,int rows,int inner,
                             int cols,int bits,int tile_rows,MatmulOptions options) {
    validate_role(role,linear);
    validate_shape(rows,inner,cols,bits);
    if(tile_rows<1) throw std::invalid_argument("offline row tile must be positive");
    const auto before_sent=linear.iopack->io->counter+linear.iopack->io_rev->counter+linear.iopack->io_GC->counter;
    const auto before_received=received_bytes(*linear.iopack);
    const auto before_rounds=linear.iopack->io->num_rounds+linear.iopack->io_rev->num_rounds+linear.iopack->io_GC->num_rounds;
    const int public_plan[]{rows,inner,cols,bits,tile_rows};
    int remote_plan[5]{};
    if(role==sci::ALICE) {
        linear.iopack->io->send_data(public_plan,sizeof(public_plan));
        linear.iopack->io->recv_data(remote_plan,sizeof(remote_plan));
    } else {
        linear.iopack->io->recv_data(remote_plan,sizeof(remote_plan));
        linear.iopack->io->send_data(public_plan,sizeof(public_plan));
    }
    if(!std::equal(std::begin(public_plan),std::end(public_plan),std::begin(remote_plan)))
        throw std::invalid_argument("Beaver public matrix/tile plan differs between parties");
    MatmulMaterial m;
    m.rows=rows;m.inner=inner;m.cols=cols;m.bits=bits;m.options=options;
    const auto estimate=estimate_matmul_generation(rows,inner,cols,bits,tile_rows);
    m.offline_tile_rows=tile_rows;m.offline_tail_rows=(rows-1)%tile_rows+1;
    m.offline_working_bytes=estimate.working_bytes;m.offline_resident_bytes=estimate.resident_bytes;
    m.offline_receive_instrumented=receives_instrumented(*linear.iopack->io,0) &&
        receives_instrumented(*linear.iopack->io_rev,0) && receives_instrumented(*linear.iopack->io_GC,0);
    m.u.resize(size_t(rows)*inner);m.v.resize(size_t(inner)*cols);m.w.resize(size_t(rows)*cols);
    sci::PRG128 rng;
    rng.random_data(m.u.data(),m.u.size()*sizeof(uint64_t));
    rng.random_data(m.v.data(),m.v.size()*sizeof(uint64_t));
    const auto ring_mask=mask(bits);
    for(auto& x:m.u) x&=ring_mask;
    for(auto& x:m.v) x&=ring_mask;
    std::vector<uint64_t> cross(size_t(std::min(rows,tile_rows))*cols);
    for(int start=0;start<rows;start+=tile_rows) {
        const int count=std::min(tile_rows,rows-start);
        linear.matmul_cross_terms(count,inner,cols,m.u.data()+size_t(start)*inner,
                                  m.v.data(),cross.data(),bits,bits,bits,true,MultMode::None);
        local_matmul(linear,count,inner,cols,m.u.data()+size_t(start)*inner,
                     m.v.data(),m.w.data()+size_t(start)*cols,
                     options.mapped_layout,m.offline_local);
        const auto buffer_start=Clock::now();
        for(int i=0;i<count*cols;++i) m.w[size_t(start)*cols+i]=(m.w[size_t(start)*cols+i]+cross[i])&ring_mask;
        m.offline_local.buffer_seconds+=elapsed(buffer_start);
        ++m.offline_tiles;
    }
    m.offline_sent_bytes=linear.iopack->io->counter+linear.iopack->io_rev->counter+linear.iopack->io_GC->counter-before_sent;
    m.offline_received_bytes=received_bytes(*linear.iopack)-before_received;
    m.offline_io_rounds=linear.iopack->io->num_rounds+linear.iopack->io_rev->num_rounds+linear.iopack->io_GC->num_rounds-before_rounds;
    return m;
}

std::vector<uint64_t> multiply_matmul(int role,sci::IOPack& io,LinearOT& linear,
    const std::vector<uint64_t>& lhs,const std::vector<uint64_t>& rhs,MatmulMaterial& m,
    MatmulWorkspace* workspace) {
    std::vector<uint64_t> result;
    multiply_matmul(role,io,linear,lhs,rhs,m,result,workspace);
    return result;
}

void multiply_matmul(int role,sci::IOPack& io,LinearOT& linear,
    const std::vector<uint64_t>& lhs,const std::vector<uint64_t>& rhs,MatmulMaterial& m,
    std::vector<uint64_t>& output,MatmulWorkspace* workspace) {
    validate_role(role,linear);
    if(&io!=linear.iopack) throw std::invalid_argument("Beaver I/O context mismatch");
    if(m.consumed) throw std::logic_error("matrix Beaver material already consumed");
    validate_shape(m.rows,m.inner,m.cols,m.bits);
    if(lhs.size()!=size_t(m.rows)*m.inner || rhs.size()!=size_t(m.inner)*m.cols ||
       m.u.size()!=lhs.size() || m.v.size()!=rhs.size() || m.w.size()!=size_t(m.rows)*m.cols)
        throw std::invalid_argument("matrix Beaver input/material shape mismatch");
    const auto ring_mask=mask(m.bits);
    for(auto x:lhs) if((x&ring_mask)!=x) throw std::invalid_argument("lhs not encoded in material ring");
    for(auto x:rhs) if((x&ring_mask)!=x) throw std::invalid_argument("rhs not encoded in material ring");
    MatmulWorkspace local_workspace;
    auto& scratch=workspace?*workspace:local_workspace;
    if(&output==&scratch.opened || &output==&scratch.remote ||
       &output==&scratch.product || &output==&scratch.combined_rhs || &output==&scratch.accumulation ||
       &lhs==&scratch.opened || &lhs==&scratch.remote ||
       &lhs==&scratch.product || &lhs==&scratch.combined_rhs || &lhs==&scratch.accumulation ||
       &rhs==&scratch.opened || &rhs==&scratch.remote ||
       &rhs==&scratch.product || &rhs==&scratch.combined_rhs || &rhs==&scratch.accumulation)
        throw std::invalid_argument("matrix operands/output must not alias workspace");
    auto buffer_start=Clock::now();
    auto& opened=scratch.opened;
    auto& remote=scratch.remote;
    opened.resize(lhs.size()+rhs.size());remote.resize(opened.size());
    for(size_t i=0;i<lhs.size();++i) opened[i]=(lhs[i]-m.u[i])&ring_mask;
    for(size_t i=0;i<rhs.size();++i) opened[lhs.size()+i]=(rhs[i]-m.v[i])&ring_mask;
    const size_t lhs_size=lhs.size();
    m.online_local.buffer_seconds+=elapsed(buffer_start);
    m.consumed=true;
    open_values(role, io, opened, ring_mask, opened, &remote);
    buffer_start=Clock::now();
    m.opened_values=opened.size();
    auto& tmp=scratch.product;
    tmp.resize(m.w.size());
    const uint64_t* right=m.v.data();
    if(role==sci::ALICE && m.options.algebra) {
        scratch.combined_rhs.resize(m.v.size());
        for(size_t i=0;i<m.v.size();++i)
            scratch.combined_rhs[i]=(m.v[i]+opened[lhs_size+i])&ring_mask;
        right=scratch.combined_rhs.data();
    }
    m.online_local.buffer_seconds+=elapsed(buffer_start);
    local_matmul(linear,m.rows,m.inner,m.cols,opened.data(),right,tmp.data(),
                 m.options.mapped_layout,m.online_local);
    buffer_start=Clock::now();
    auto& result=scratch.accumulation;
    result=m.w;
    for(size_t i=0;i<result.size();++i) result[i]+=tmp[i];
    m.online_local.buffer_seconds+=elapsed(buffer_start);
    local_matmul(linear,m.rows,m.inner,m.cols,m.u.data(),opened.data()+lhs_size,tmp.data(),
                 m.options.mapped_layout,m.online_local);
    buffer_start=Clock::now();
    for(size_t i=0;i<result.size();++i) result[i]+=tmp[i];
    m.online_local.buffer_seconds+=elapsed(buffer_start);
    if(role==sci::ALICE && !m.options.algebra) {
        local_matmul(linear,m.rows,m.inner,m.cols,opened.data(),opened.data()+lhs_size,tmp.data(),
                     m.options.mapped_layout,m.online_local);
        buffer_start=Clock::now();
        for(size_t i=0;i<result.size();++i) result[i]+=tmp[i];
        m.online_local.buffer_seconds+=elapsed(buffer_start);
    }
    buffer_start=Clock::now();
    for(auto& x:result) x&=ring_mask;
    // Replace caller storage only after all reads, including material aliases.
    output.swap(result);
    m.online_local.buffer_seconds+=elapsed(buffer_start);
}

MulCirMaterial prepare_mulcir(int role, LinearOT& linear, int elements, int bits) {
    validate_role(role, linear);
    validate_vector(elements, bits);
    MulCirMaterial m;
    m.elements = elements;
    m.bits = bits;
    m.u.resize(elements);
    m.v.resize(elements);
    m.w.resize(elements);
    sci::PRG128 rng;
    rng.random_data(m.u.data(), m.u.size() * sizeof(uint64_t));
    rng.random_data(m.v.data(), m.v.size() * sizeof(uint64_t));
    const auto ring_mask = mask(bits);
    for (auto& x : m.u) x &= ring_mask;
    for (auto& x : m.v) x &= ring_mask;
    std::vector<uint64_t> cross(elements);
    linear.hadamard_cross_terms(elements, m.u.data(), m.v.data(), cross.data(),
                                bits, bits, bits, MultMode::None);
    linear.hadamard_cleartext(elements, m.u.data(), m.v.data(), m.w.data());
    for (int i = 0; i < elements; ++i) m.w[i] = (m.w[i] + cross[i]) & ring_mask;
    return m;
}

std::vector<uint64_t> multiply_mulcir(int role, sci::IOPack& io, LinearOT& linear,
    const std::vector<uint64_t>& lhs, const std::vector<uint64_t>& rhs,
    MulCirMaterial& m) {
    validate_role(role, linear);
    if (&io != linear.iopack) throw std::invalid_argument("Beaver I/O context mismatch");
    if (m.consumed) throw std::logic_error("MulCir Beaver material already consumed");
    validate_vector(m.elements, m.bits);
    if (lhs.size() != size_t(m.elements) || rhs.size() != size_t(m.elements) ||
        m.u.size() != lhs.size() || m.v.size() != rhs.size() || m.w.size() != lhs.size())
        throw std::invalid_argument("MulCir Beaver input/material shape mismatch");
    const auto ring_mask = mask(m.bits);
    for (auto x : lhs) if ((x & ring_mask) != x)
        throw std::invalid_argument("MulCir lhs not encoded in material ring");
    for (auto x : rhs) if ((x & ring_mask) != x)
        throw std::invalid_argument("MulCir rhs not encoded in material ring");
    std::vector<uint64_t> local(lhs.size() + rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) local[i] = (lhs[i] - m.u[i]) & ring_mask;
    for (size_t i = 0; i < rhs.size(); ++i) local[lhs.size() + i] = (rhs[i] - m.v[i]) & ring_mask;
    m.consumed = true;
    std::vector<uint64_t> opened;
    open_values(role, io, local, ring_mask, opened);
    m.opened_values = opened.size();
    std::vector<uint64_t> result = m.w;
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (result[i] + opened[i] * m.v[i] + m.u[i] * opened[lhs.size() + i]) & ring_mask;
        if (role == sci::ALICE)
            result[i] = (result[i] + opened[i] * opened[lhs.size() + i]) & ring_mask;
    }
    return result;
}

ScalarMulMaterial prepare_scalarmul(int role, LinearOT& linear, int elements, int bits) {
    validate_role(role, linear);
    validate_vector(elements, bits);
    ScalarMulMaterial m;
    m.elements = elements;
    m.bits = bits;
    m.v.resize(elements);
    m.w.resize(elements);
    sci::PRG128 rng;
    rng.random_data(&m.u, sizeof(m.u));
    rng.random_data(m.v.data(), m.v.size() * sizeof(uint64_t));
    const auto ring_mask = mask(bits);
    m.u &= ring_mask;
    for (auto& x : m.v) x &= ring_mask;
    std::vector<uint64_t> repeated(elements, m.u), cross(elements);
    linear.hadamard_cross_terms(elements, repeated.data(), m.v.data(), cross.data(),
                                bits, bits, bits, MultMode::None);
    linear.hadamard_cleartext(elements, repeated.data(), m.v.data(), m.w.data());
    for (int i = 0; i < elements; ++i) m.w[i] = (m.w[i] + cross[i]) & ring_mask;
    return m;
}

std::vector<uint64_t> multiply_scalarmul(int role, sci::IOPack& io, LinearOT& linear,
    const std::vector<uint64_t>& scalar, const std::vector<uint64_t>& rhs,
    ScalarMulMaterial& m) {
    validate_role(role, linear);
    if (&io != linear.iopack) throw std::invalid_argument("Beaver I/O context mismatch");
    if (m.consumed) throw std::logic_error("ScalarMul Beaver material already consumed");
    validate_vector(m.elements, m.bits);
    if (scalar.size() != 1 || rhs.size() != size_t(m.elements) ||
        m.v.size() != rhs.size() || m.w.size() != rhs.size())
        throw std::invalid_argument("ScalarMul Beaver input/material shape mismatch");
    const auto ring_mask = mask(m.bits);
    if ((scalar[0] & ring_mask) != scalar[0])
        throw std::invalid_argument("ScalarMul scalar not encoded in material ring");
    for (auto x : rhs) if ((x & ring_mask) != x)
        throw std::invalid_argument("ScalarMul rhs not encoded in material ring");
    std::vector<uint64_t> local(1 + rhs.size());
    local[0] = (scalar[0] - m.u) & ring_mask;
    for (size_t i = 0; i < rhs.size(); ++i) local[i + 1] = (rhs[i] - m.v[i]) & ring_mask;
    m.consumed = true;
    std::vector<uint64_t> opened;
    open_values(role, io, local, ring_mask, opened);
    m.opened_values = opened.size();
    std::vector<uint64_t> result = m.w;
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (result[i] + opened[0] * m.v[i] + m.u * opened[i + 1]) & ring_mask;
        if (role == sci::ALICE)
            result[i] = (result[i] + opened[0] * opened[i + 1]) & ring_mask;
    }
    return result;
}

}
