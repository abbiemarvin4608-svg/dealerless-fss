#include "Math/math-functions.h"
#include "native_workers.h"
#include "native_query_materials.h"
#include "measurement.h"
#include <array>
#include <sys/resource.h>

namespace {
const std::array<int64_t,25> clear_values{{
    0,1,-1,127,-128,128,-129,1023,-1024,32767,-32768,32766,-32767,
    255,-256,256,-257,4095,-4096,4096,-4097,16383,-16384,8191,-8192}};

struct Suite {
    int role,query,count;
    std::vector<uint64_t> input,sigmoid_output,tanh_output;

    void load(int public_pattern=-1) {
        input.resize(count);
        for(int i=0;i<count;++i) {
            if(public_pattern>=0) input[i]=public_pattern?65535:0;
            else {
                const uint64_t clear=uint64_t(clear_values[(i+query*7)%clear_values.size()])&65535;
                const uint64_t alice=(0xfedcba98ULL+i*1987+query*337)&65535;
                input[i]=role==1?alice:(clear-alice)&65535;
            }
        }
        if(public_pattern<0)
            std::cout<<"QUERY_INPUT_LOADED {\"query\":"<<query<<",\"count\":"<<count<<"}\n";
    }

    void run() {
        sigmoid_output.assign(count,0);tanh_output.assign(count,0);
        auto sigmoid=[&] {
            sdcf_native::TaskScope task(query,0,0,sdcf_native::isolated_graph()?2:0);
            sdcf_native::activation(false,input.data(),sigmoid_output.data(),count,16,16,7,14);
        };
        auto tanh=[&] {
            sdcf_native::TaskScope task(query,1,sdcf_native::isolated_graph()?2:0,
                                        sdcf_native::isolated_graph()?2:0);
            sdcf_native::activation(true,input.data(),tanh_output.data(),count,16,16,7,7);
        };
        sdcf_native::TaskScope graph(query,0,0,sdcf_native::isolated_graph()?2:0);
        sdcf_native::run_task_pair(sigmoid,tanh);
    }

    void print(sci::IOPack& io) {
        for(const auto& item:{std::make_pair(std::string("sigmoid"),&sigmoid_output),
                             std::make_pair(std::string("tanh"),&tanh_output)}) {
            std::vector<uint64_t> remote(count);
            if(role==1) io.io->send_data(item.second->data(),count*sizeof(uint64_t));
            else {
                io.io->recv_data(remote.data(),count*sizeof(uint64_t));
                std::cout<<"ACTIVATION_OUTPUT {\"query\":"<<query<<",\"activation\":\""
                         <<item.first<<"\",\"values\":[";
                for(int i=0;i<count;++i) {
                    if(i) std::cout<<',';
                    std::cout<<((remote[i]+item.second->at(i))&65535);
                }
                std::cout<<"]}\n";
            }
        }
    }
};

void b2a(int slot,int count) {
    std::vector<uint8_t> input(count,0);
    std::vector<uint64_t> output(count,0);
    sdcf_native::context(slot).aux->B2A(input.data(),output.data(),count,16);
}
}

int main(int argc,char** argv) {
    bool initialized=false;
    try {
        if(argc!=5) throw std::runtime_error("role port mode count");
        const int role=std::stoi(argv[1]),port=std::stoi(argv[2]),count=std::stoi(argv[4]);
        const std::string mode=argv[3];
        const bool negative=mode=="wrong_context" || mode=="wrong_direction";
        const bool prepared=mode!="default";
        const auto setup=sdcf_measure::point();
        sci::IOPack io(role,port);
        sci::OTPack ot(&io,role);
        MathFunctions math(role,&io,&ot);
        sdcf_native::setup(role,port,"127.0.0.1",math,negative?2:1);
        initialized=true;
        std::vector<TripleGenerator*> generators;
        for(int slot=0;slot<sdcf_native::context_count();++slot) {
            auto& worker=sdcf_native::context(slot);
            sdcf_planning::register_pack(worker.otpack,slot);
            generators.push_back(worker.aux->mill->triple_gen);
        }
        sdcf_measure::report("setup",-1,setup);
        for(int query=0;query<2;++query) {
            const auto offline=sdcf_measure::point();
            sdcf_preprocess::begin_prepare(query,prepared);
            if(prepared) {
                std::string plans[2];
                for(int pattern=0;pattern<2;++pattern) {
                    const auto before=sdcf_measure::point();
                    const auto prg=sdcf_planning::prg_blocks.load();
                    sdcf_planning::begin();
                    if(negative) b2a(0,count);
                    else {
                        Suite scratch{role,query,count,{},{},{}};
                        scratch.load(pattern);scratch.run();
                    }
                    plans[pattern]=sdcf_planning::end();
                    const auto after=sdcf_measure::point();
                    if(before.sent!=after.sent || before.received!=after.received ||
                       before.native_epoch!=after.native_epoch || before.offline_epoch!=after.offline_epoch ||
                       prg!=sdcf_planning::prg_blocks.load())
                        throw std::runtime_error("worker public planning changed protocol state");
                }
                if(plans[0]!=plans[1]) throw std::runtime_error("worker values changed public plan");
                sdcf_planning::compare_plan(io.io,role,plans[0]);
                std::cout<<"NATIVE_PLAN_PURITY {\"query\":"<<query
                    <<",\"separate_public_buffers\":true,\"sent_bytes\":0,\"received_bytes\":0,\"generation_delta\":0,\"bridge_generation_delta\":0,\"prg_blocks_delta\":0}\n";
                sdcf_planning::prepare_query_materials(query,generators);
            }
            sdcf_preprocess::seal_prepare(query);
            io.io->sync();sdcf_preprocess::material_ready(query);
            sdcf_measure::report("offline",query,offline);
            const auto input=sdcf_measure::point();
            sdcf_preprocess::begin_input(query);
            Suite suite{role,query,count,{},{},{}};
            suite.load();sdcf_measure::report("input",query,input);
            const auto ready=sdcf_measure::point();
            io.io->sync();sdcf_measure::report("ready",query,ready);
            const auto online=sdcf_measure::point();
            sdcf_preprocess::enter_online(query);
            if(negative) {
                const auto epoch=sdcf_preprocess::generation_epoch.load();
                try {
                    if(mode=="wrong_context") b2a(1,count);
                    else {
                        auto* original=ot.iknp_straight;
                        ot.iknp_straight=ot.iknp_reversed;
                        try {b2a(0,count);} catch(...) {ot.iknp_straight=original;throw;}
                        ot.iknp_straight=original;
                    }
                } catch(const std::exception& error) {
                    std::cout<<"CONTEXT_REJECTION {\"case\":\""<<mode
                        <<"\",\"native_generation_epoch_delta\":"
                        <<sdcf_preprocess::generation_epoch.load()-epoch<<"}\n";
                    throw;
                }
                throw std::runtime_error("unowned B2A material was accepted");
            }
            suite.run();
            sdcf_triples::balance(query);sdcf_preprocess::leave_online(query);
            io.io->sync();sdcf_measure::report("online",query,online);
            const auto output=sdcf_measure::point();
            suite.print(io);
            io.io->sync();sdcf_preprocess::finish_query(query);
            sdcf_measure::report("output",query,output);
        }
        rusage usage{};getrusage(RUSAGE_SELF,&usage);
        std::cout<<"CONTEXT_RESOURCES {\"max_rss_kib\":"<<usage.ru_maxrss<<"}\n";
        sdcf_native::finish();
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"CONTEXT_ERROR "<<error.what()<<'\n';
        if(initialized) sdcf_native::finish();
        return 2;
    }
}
