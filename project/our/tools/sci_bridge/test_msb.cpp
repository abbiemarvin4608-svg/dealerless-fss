// Public seam: SCI additive ring shares -> XOR MSB shares.
#include "sci_msb_bridge.h"
#include "sci_hook.h"
#include "BuildingBlocks/aux-protocols.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int role=std::atoi(argv[1]), port=std::atoi(argv[2]), fss_port=port+20;
    const int exhaustive=std::atoi(argv[3]);
    sci::IOPack io(role, port, "127.0.0.1");
    sci::OTPack ot(&io,role);
    AuxProtocols original(role,&io,&ot);
    void* bridge=segfss_sci_create(role,"127.0.0.1",fss_port);
    if (!bridge) { std::cerr<<segfss_sci_error()<<'\n';return 2; }
    sci_segmented::context()=bridge;
    unsetenv("SEGFSS_SCI_ENABLE");
    int failed=0,total=0;
    for (int width: {2,9,16}) {
        const uint64_t modulus=1ULL<<width, mask=modulus-1;
        std::vector<uint64_t> values;
        if (width<16 && exhaustive) for(uint64_t x=0;x<modulus;++x) values.push_back(x);
        else {
            values={0,1,2,modulus/2-1,modulus/2,modulus/2+1,mask-1,mask};
            for(int b=1;b<width;++b) for(int d=-1;d<=1;++d) values.push_back(((1ULL<<b)+d)&mask);
            for(int k=0;k<55;++k) values.push_back((k*4051+17)&mask);
            // 9->16 sign extension: -256, -255, -1 and nonnegative 255.
            if(width==16)for(uint64_t x:{65280ULL,65281ULL,65535ULL,255ULL})values.push_back(x);
        }
        std::vector<uint64_t> shares(values.size());
        for(size_t i=0;i<values.size();++i) {
            const uint64_t a=(i*173+mask-2)&mask;
            shares[i]=role==1?a:(values[i]-a)&mask;
        }
        std::vector<uint8_t> oracle(values.size()),result(values.size()),remote(values.size());
        original.MSB(shares.data(),oracle.data(),values.size(),width);
        if(role==1) { io.io->send_data(oracle.data(),oracle.size());io.io->recv_data(remote.data(),remote.size()); }
        else { io.io->recv_data(remote.data(),remote.size());io.io->send_data(oracle.data(),oracle.size()); }
        for(size_t i=0;i<oracle.size();++i) oracle[i]^=remote[i];
        std::vector<int> counts={1,2,4,8,width-1};
        std::sort(counts.begin(),counts.end());counts.erase(std::unique(counts.begin(),counts.end()),counts.end());
        for(int nseg:counts) {
            if(nseg>width-1) continue;
            setenv("SEGFSS_SCI_ENABLE","1",1);
            setenv("SEGFSS_SCI_WIDTH",std::to_string(width).c_str(),1);
            setenv("SEGFSS_SCI_NSEG",std::to_string(nseg).c_str(),1);
            original.MSB(shares.data(),result.data(),values.size(),width);
            unsetenv("SEGFSS_SCI_ENABLE");
            if(role==1) { io.io->send_data(result.data(),result.size());io.io->recv_data(remote.data(),remote.size()); }
            else { io.io->recv_data(remote.data(),remote.size());io.io->send_data(result.data(),result.size()); }
            for(size_t i=0;i<result.size();++i) {
                uint8_t got=result[i]^remote[i],expected=values[i]>>(width-1);
                if(got!=expected || got!=oracle[i]) {
                    ++failed;std::cerr<<"mismatch width="<<width<<" nseg="<<nseg<<" x="<<values[i]<<" got="<<int(got)<<" SCI="<<int(oracle[i])<<" expected="<<int(expected)<<'\n';
                }
                ++total;
            }
            std::cout<<"CHECK role="<<role<<" input_bits="<<width<<" comparison_bits="<<width-1<<" nseg="<<nseg<<" cases="<<values.size()<<" failed="<<failed<<'\n';
        }
    }
    // A true private-input run: the peer never receives either local value.
    // Only its two random mask contributions and masked z are protocol inputs.
    std::vector<uint64_t> private_shares(100);
    sci::PRG128 randomness;
    randomness.random_data(private_shares.data(),private_shares.size()*sizeof(uint64_t));
    for(auto& x:private_shares) x&=65535;
    std::vector<uint8_t> secret_output(100);
    if(segfss_sci_msb(bridge,private_shares.data(),secret_output.data(),100,16,4)!=0)return 2;
    segfss_sci_stats stats{};
    segfss_sci_get_stats(bridge,&stats);
    std::cout<<"PRIVATE_RUN input_bits=16 nseg=4 elements=100 opened_results=0\n";
    std::cout<<"STATS calls="<<stats.calls<<" elements="<<stats.elements<<" keys="<<stats.segment_keys<<" offline_ms="<<stats.offline_ms<<" online_ms="<<stats.online_ms<<" sent_bytes="<<stats.sent_bytes<<" received_bytes="<<stats.received_bytes<<'\n';
    sci_segmented::finish();
    std::cout<<"RESULT role="<<role<<" cases="<<total<<" mismatches="<<failed<<'\n';
    return failed?1:0;
}
