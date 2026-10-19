#include "private_msb.h"
#include "segmented_dcf.h"
#include "runtime_options.h"
#include "mpc/secure_ops.h"
#include <cryptoTools/Crypto/PRNG.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <chrono>

namespace segfss {
using MsbClock=std::chrono::steady_clock;
static double elapsedMs(MsbClock::time_point start){return std::chrono::duration<double,std::milli>(MsbClock::now()-start).count();}
std::size_t MsbMaterial::capacityBytes() const {
    std::size_t result=mask_shares.capacity()*sizeof(uint64_t)+mask_sign_shares.capacity()+
        keys.capacity()*sizeof(CggmKey)+scan.a.capacity()+scan.b.capacity()+scan.c.capacity();
    for(const auto& key:keys)result+=key.internal_cw.capacity()*sizeof(osuCrypto::block);
    return result;
}
void validatePrivateMsbMaterial(int role,const MsbMaterial& material,
    int count,int width,int nseg,Peer* player) {
    if((role!=SERVER&&role!=CLIENT)||!player||player->role()!=role||material.owner_peer!=player||
        count<1||width<2||width>64||nseg<1||nseg>=width||count>std::numeric_limits<int>::max()/(2*width)||
        material.consumed||material.count!=count||material.width!=width||material.nseg!=nseg||
        material.mask_shares.size()!=size_t(count)||material.mask_sign_shares.size()!=size_t(count)||
        material.keys.size()!=size_t(count)*nseg||material.scan.cursor!=0)
        throw std::invalid_argument("private MSB material owner, shape or one-use contract mismatch");
    const size_t gates=size_t(2)*count*(nseg-1);
    if(material.scan.a.size()!=gates||material.scan.b.size()!=gates||material.scan.c.size()!=gates)
        throw std::invalid_argument("private MSB scan inventory is incomplete");
    const SegmentLayout layout(width-1,nseg);
    for(size_t index=0;index<material.keys.size();++index) {
        const auto& key=material.keys[index];
        const int depth=layout.width[index%nseg];
        if(key.depth!=depth||key.internal_cw.size()!=size_t(depth-1))
            throw std::invalid_argument("private MSB segment key shape mismatch");
    }
}
MsbMaterial privateMsbOffline(int role,int count,int width,int nseg,Peer* player) {
    if((role!=SERVER && role!=CLIENT)||!player||player->role()!=role||count<1||width<2||width>64||nseg<1||nseg>=width)
        throw std::invalid_argument("privateMsbOffline: invalid role/count/width/nseg/peer");
    if(count>std::numeric_limits<int>::max()/(2*width))
        throw std::invalid_argument("privateMsbOffline: batch too large");
    MsbMaterial material;
    material.owner_peer=player;
    material.width=width;material.count=count;material.nseg=nseg;
    auto phase_start=MsbClock::now();
    SegmentLayout layout(width-1,nseg);
    osuCrypto::PRNG rng(osuCrypto::sysRandomSeed());
    std::vector<u8> bits(count*width);
    rng.get(bits.data(),bits.size());
    for(auto& bit:bits)bit&=1;
    std::vector<GroupElement> arithmetic(bits.size(),GroupElement(0,width));
    // Sampling the mask as XOR bits avoids decomposing any private input.
    // B2A is COT-based, batched, and generates arithmetic mask shares offline.
    B2A(role,bits.data(),arithmetic.data(),bits.size(),width,player);
    material.mask_shares.resize(count);
    material.mask_sign_shares.resize(count);
    std::vector<std::vector<u8>> thresholds(count*nseg);
    const uint64_t mask=width==64?UINT64_MAX:((uint64_t(1)<<width)-1);
    for(int row=0;row<count;++row) {
        uint64_t sum=0;
        for(int bit=0;bit<width;++bit)sum+=arithmetic[row*width+bit].value<<bit;
        material.mask_shares[row]=sum&mask;
        material.mask_sign_shares[row]=bits[row*width+width-1];
        for(int seg=0;seg<nseg;++seg) {
            auto& threshold=thresholds[row*nseg+seg];
            for(int k=layout.width[seg]-1;k>=0;--k)
                threshold.push_back(bits[row*width+layout.shift[seg]+k]);
        }
    }
    material.mask_conversion_ms=elapsedMs(phase_start);phase_start=MsbClock::now();
    auto options=dcfGenerationOptions();
    options.resident_bytes+=material.mask_shares.capacity()*sizeof(uint64_t)+material.mask_sign_shares.capacity();
    const auto plan=planDcfGeneration(count,layout,options);
    const bool reuse_workspace=runtimeFlag("SEGFSS_REUSE_GENERATION_WORKSPACE");
    CggmGenerationWorkspace workspace;
    material.keys.reserve(thresholds.size());
    for(const auto& tile:plan.tiles) {
        std::vector<std::vector<u8>> inputs(thresholds.begin()+tile.begin,
                                          thresholds.begin()+tile.begin+tile.key_count);
        CggmGenerationStats stats;
        auto keys=cggmGen2pcBatch(role,inputs,nullptr,&stats,player,plan.participant_limit,
                                 reuse_workspace?&workspace:nullptr);
        for(auto& key:keys)material.keys.push_back(std::move(key));
        material.generation_workspace_capacity_bytes=std::max<uint64_t>(
            material.generation_workspace_capacity_bytes,stats.workspace_capacity_bytes);
        material.generation_peak_working_bytes=std::max<uint64_t>(
            material.generation_peak_working_bytes,tile.working_bytes+stats.scratch_peak_bytes);
    }
    material.generation_tiles=plan.tiles.size();
    material.keygen_ms=elapsedMs(phase_start);phase_start=MsbClock::now();
    material.scan=booleanAndOffline(role,2*count*(nseg-1),player);
    material.scan_preprocessing_ms=elapsedMs(phase_start);
    return material;
}

void privateMsbOnline(int role,const uint64_t* shares,u8* output,
                      MsbMaterial& material,Peer* player) {
    try {
        if(!shares||!output)throw std::invalid_argument("privateMsbOnline: invalid input/output");
        validatePrivateMsbMaterial(role,material,material.count,material.width,material.nseg,player);
    } catch(...) {material.consumed=true;throw;}
    material.consumed=true;
    const int count=material.count,width=material.width,nseg=material.nseg;
    const uint64_t mask=width==64?UINT64_MAX:((uint64_t(1)<<width)-1);
    auto phase_start=MsbClock::now();
    std::vector<uint64_t> opened(count),remote(count);
    for(int row=0;row<count;++row)opened[row]=(shares[row]+material.mask_shares[row])&mask;
    // Role-ordered exchange handles batches larger than the socket buffer.
    if(role==SERVER){player->send_u64(opened.data(),count);player->recv_u64(remote.data(),count);}
    else {player->recv_u64(remote.data(),count);player->send_u64(opened.data(),count);}
    for(int row=0;row<count;++row)opened[row]=(opened[row]+remote[row])&mask;
    material.masked_input_open_ms=elapsedMs(phase_start);phase_start=MsbClock::now();
    SegmentLayout layout(width-1,nseg);
    std::vector<std::vector<u8>> public_bits(count*nseg);
    for(int row=0;row<count;++row)for(int seg=0;seg<nseg;++seg) {
        auto& bits=public_bits[row*nseg+seg];
        bits.reserve(layout.width[seg]);
        for(int k=layout.width[seg]-1;k>=0;--k)
            bits.push_back((opened[row]>>(layout.shift[seg]+k))&1);
    }
    auto level=cggmEvalBatch(material.keys,public_bits);
    material.evaluation_ms=elapsedMs(phase_start);phase_start=MsbClock::now();
    // Vectorize every dependent scan level across all elements and segments.
    int active=nseg;
    while(active>1) {
        const int pairs=active/2,next_count=(active+1)/2;
        std::vector<u8> a(2*count*pairs),b(a.size()),c(a.size());
        for(int row=0;row<count;++row)for(int p=0;p<pairs;++p) {
            auto left=level[row*active+2*p],right=level[row*active+2*p+1];
            int i=2*(row*pairs+p);
            a[i]=left.e;b[i]=right.e;a[i+1]=left.e;b[i+1]=right.L;
        }
        booleanAndOnline(role,a.data(),b.data(),c.data(),c.size(),material.scan,player);
        std::vector<CggmEval> next(count*next_count);
        for(int row=0;row<count;++row) {
            for(int p=0;p<pairs;++p) {
                int i=2*(row*pairs+p);
                next[row*next_count+p]={static_cast<u8>(level[row*active+2*p].L^c[i+1]),c[i]};
            }
            if(active&1)next[row*next_count+next_count-1]=level[row*active+active-1];
        }
        level=std::move(next);active=next_count;
    }
    material.scan_ms=elapsedMs(phase_start);
    for(int row=0;row<count;++row)
        output[row]=level[row].L^material.mask_sign_shares[row]^
                    (role==SERVER?static_cast<u8>(opened[row]>>(width-1)):0);
}
}
