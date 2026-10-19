// Public seam: additive ring shares -> XOR MSB shares; performance reveals neither.
#include "sci_msb_bridge.h"
#include "BuildingBlocks/aux-protocols.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sys/resource.h>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
int main(int argc,char**argv){
 if(argc!=8)return 2;
 int role=std::atoi(argv[1]),port=std::atoi(argv[2]),width=std::atoi(argv[4]),nseg=std::atoi(argv[5]),count=std::atoi(argv[6]);
 std::string backend=argv[3];bool check=std::string(argv[7])=="check";
 if(role<1||role>2||width<2||width>32||nseg<1||nseg>=width||count<1||(backend!="sci"&&backend!="ours"))return 2;
 uint64_t mask=(1ULL<<width)-1;std::vector<uint64_t> shares(count),values(count);
 if(check){
   for(int i=0;i<count;++i){uint64_t x=(uint64_t(i)*4051+17)&mask;
     if(i<8){uint64_t edge[]={0,1,2,mask,mask/2,mask/2+1,mask-1,mask};x=edge[i];}
     if(i>=8&&i<8+3*(width-1)){int j=i-8;x=((1ULL<<(1+j/3))+(j%3)-1)&mask;}
     values[i]=x;uint64_t a=(uint64_t(i)*173+mask-2)&mask;shares[i]=role==1?a:(x-a)&mask;}
 }else{
   // Only this party reads its private share file. Same files used for paired backends.
   const char* input=std::getenv("SWEEP_PRIVATE_SHARE_FILE");if(!input)return 2;
   std::ifstream file(input,std::ios::binary);file.read(reinterpret_cast<char*>(shares.data()),count*sizeof(uint64_t));
   if(file.gcount()!=count*int(sizeof(uint64_t)))return 2;for(auto& x:shares)x&=mask;
 }
 std::vector<uint8_t> output(count);double setup_ms=0,offline_ms=0,online_ms=0;uint64_t sent=0,received=0,setup_sent=0,setup_recv=0,off_sent=0,off_recv=0,on_sent=0,on_recv=0;int mismatches=0;auto total_start=Clock::now();
 segfss_sci_stats stats{};
 sci::IOPack* io=nullptr;sci::OTPack* ot=nullptr;AuxProtocols* aux=nullptr;void* bridge=nullptr;
 auto setup_start=Clock::now();
 if(backend=="sci"){
  io=new sci::IOPack(role,port,"127.0.0.1");ot=new sci::OTPack(io,role);aux=new AuxProtocols(role,io,ot);setup_ms=ms(setup_start);
  auto reads=[&](){return io->io->sweep_received+io->io_rev->sweep_received+io->io_GC->sweep_received;};
  setup_sent=io->get_comm();setup_recv=reads();auto begin=Clock::now();aux->MSB(shares.data(),output.data(),count,width);online_ms=ms(begin);sent=io->get_comm();received=reads();on_sent=sent-setup_sent;on_recv=received-setup_recv;
 }else{
  bridge=segfss_sci_create(role,"127.0.0.1",port);if(!bridge){std::cerr<<segfss_sci_error()<<'\n';return 3;}setup_ms=ms(setup_start);
  if(segfss_sci_msb(bridge,shares.data(),output.data(),count,width,nseg)){std::cerr<<segfss_sci_error()<<'\n';return 4;}
  segfss_sci_get_stats(bridge,&stats);offline_ms=stats.offline_ms;online_ms=stats.online_ms;sent=stats.sent_bytes;received=stats.received_bytes;
  setup_sent=stats.setup_sent_bytes;setup_recv=stats.setup_received_bytes;off_sent=stats.offline_sent_bytes;off_recv=stats.offline_received_bytes;on_sent=stats.online_sent_bytes;on_recv=stats.online_received_bytes;
 }
 double total_ms=ms(total_start);
 if(check){
   // Separate correctness-only channel; its explicit output reveal is outside performance.
   sci::NetIO reveal(role==1?nullptr:"127.0.0.1",port+200,true,true);std::vector<uint8_t> remote(count);
   if(role==1){reveal.send_data(output.data(),count);reveal.flush();reveal.recv_data(remote.data(),count);}else{reveal.recv_data(remote.data(),count);reveal.send_data(output.data(),count);reveal.flush();}
   for(int i=0;i<count;++i)if((output[i]^remote[i])!=(values[i]>>(width-1)))++mismatches;
 }
 struct rusage u;getrusage(RUSAGE_SELF,&u);const char* path=std::getenv("SWEEP_METRICS");if(!path)return 2;std::ofstream out(path);out.precision(12);
 out<<"{\"keygen_ms\":"<<stats.keygen_ms<<",\"mask_conversion_ms\":"<<stats.mask_conversion_ms<<",\"scan_preprocessing_ms\":"<<stats.scan_preprocessing_ms<<",\"masked_input_open_ms\":"<<stats.masked_input_open_ms<<",\"evaluation_ms\":"<<stats.evaluation_ms<<",\"scan_ms\":"<<stats.scan_ms<<",\"party\":"<<role<<",\"backend\":\""<<backend<<"\",\"input_bits\":"<<width<<",\"comparison_bits\":"<<width-1<<",\"nseg\":"<<nseg<<",\"count\":"<<count<<",\"completed_calls\":1,\"setup_ms\":"<<setup_ms<<",\"offline_ms\":"<<(backend=="sci"?"null":std::to_string(offline_ms))<<",\"online_ms\":"<<online_ms<<",\"total_ms\":"<<total_ms<<",\"sent_bytes\":"<<sent<<",\"received_bytes\":"<<received<<",\"setup_sent_bytes\":"<<setup_sent<<",\"setup_received_bytes\":"<<setup_recv<<",\"offline_sent_bytes\":"<<off_sent<<",\"offline_received_bytes\":"<<off_recv<<",\"online_sent_bytes\":"<<on_sent<<",\"online_received_bytes\":"<<on_recv<<",\"peak_rss_kib\":"<<u.ru_maxrss<<",\"opened_outputs\":"<<(check?count:0)<<",\"opened_inputs\":0,\"mismatches\":"<<mismatches<<",\"dependent_rounds\":null}\n";
 if(bridge)segfss_sci_destroy(bridge);delete aux;delete ot;delete io;return mismatches?5:0;
}
