#include "Math/math-functions.h"
#include "native_query_materials.h"
#include "measurement.h"
#include "native_workers.h"
#include "native_dpf_lut.h"
#include "native_folded.h"
#include <array>
#include <fstream>
#include <openssl/rand.h>
#include <sys/resource.h>

namespace {
const std::array<int64_t, 25> clear_values{{
    0,1,-1,127,-128,128,-129,1023,-1024,32767,-32768,32766,-32767,
    255,-256,256,-257,4095,-4096,4096,-4097,16383,-16384,8191,-8192}};

struct ActivationSuite {
    MathFunctions& math;
    int role, query, count;
    std::vector<uint64_t> input, sigmoid_output, tanh_output;

    void load(int pattern = -1, sci::IOPack* io = nullptr) {
        input.resize(count);
        std::vector<uint64_t> random_share(count);
        if(pattern<0) {
            if(!io)throw std::runtime_error("activation input requires sharing channel");
            if(role==1) {
                if(RAND_bytes(reinterpret_cast<unsigned char*>(random_share.data()),count*sizeof(uint64_t))!=1)
                    throw std::runtime_error("activation random sharing failed");
                io->io->send_data(random_share.data(),count*sizeof(uint64_t));
            } else io->io->recv_data(random_share.data(),count*sizeof(uint64_t));
        }
        for (int i = 0; i < count; ++i) {
            if (pattern >= 0) input[i] = pattern ? 65535 : 0;
            else {
                const uint64_t clear = uint64_t((std::getenv("SDCF_ACTIVATION_EXHAUSTIVE") ? ((query * count + i) & 65535) : clear_values[(i + query * 7) % clear_values.size()])) & 65535;
                const uint64_t alice = random_share[i] & 65535;
                input[i] = role == 1 ? alice : (clear - alice) & 65535;
            }
        }
    }

    void run() {
        sigmoid_output.assign(count,0); tanh_output.assign(count,0);
        auto activation = [&](const std::string& name, auto invoke) {
            const auto epoch = sdcf_preprocess::generation_epoch.load();
            const auto triple_epoch = sdcf_triples::generation_epoch.load();
            uint64_t ot_before = 0, triple_before = 0;
            if (!sdcf_planning::active()) {
                for (const auto& item : sdcf_preprocess::inventories) ot_before += item.second.consumed;
                for (const auto& item : sdcf_triples::inventories) triple_before += item.second.consumed;
            }
            invoke();
            if (!sdcf_planning::active()) {
                uint64_t ot_after = 0, triple_after = 0;
                for (const auto& item : sdcf_preprocess::inventories) ot_after += item.second.consumed;
                for (const auto& item : sdcf_triples::inventories) triple_after += item.second.consumed;
                std::cout << "ACTIVATION_MATERIAL_COVERAGE {\"query\":" << query
                    << ",\"activation\":\"" << name << "\",\"count\":" << count
                    << ",\"native_ot_consumed\":" << ot_after - ot_before
                    << ",\"bit_triples_consumed\":" << triple_after - triple_before
                    << ",\"native_generation_epoch_delta\":" << sdcf_preprocess::generation_epoch.load() - epoch
                    << ",\"triple_generation_epoch_delta\":" << sdcf_triples::generation_epoch.load() - triple_epoch << "}\n";
            }
        };
        activation("sigmoid", [&] {
            sdcf_native::TaskScope task(query,0);
            sdcf_native::activation(false,input.data(),sigmoid_output.data(),count,16,16,7,14);
        });
        activation("tanh", [&] {
            sdcf_native::TaskScope task(query,1);
            sdcf_native::activation(true,input.data(),tanh_output.data(),count,16,16,7,7);
        });
    }

    void print(sci::IOPack& io) {
        for (const auto& item : {std::make_pair(std::string("sigmoid"), &sigmoid_output),
                                 std::make_pair(std::string("tanh"), &tanh_output)}) {
            std::vector<uint64_t> remote(count);
            if (role == 1) io.io->send_data(item.second->data(), count * sizeof(uint64_t));
            else {
                io.io->recv_data(remote.data(), count * sizeof(uint64_t));
                std::cout << "ACTIVATION_OUTPUT {\"query\":" << query
                    << ",\"activation\":\"" << item.first << "\",\"values\":[";
                for (int i = 0; i < count; ++i) {
                    if (i) std::cout << ',';
                    std::cout << ((remote[i] + item.second->at(i)) & 65535);
                }
                std::cout << "]}\n";
            }
        }
    }
};
}

int main(int argc, char** argv) {
    try {
        const int role = std::stoi(argv[1]), port = std::stoi(argv[2]), count = std::stoi(argv[4]);
        const std::string mode = argv[3];
        const bool prepared = mode != "default";
        setenv("SDCF_NATIVE_PREPROCESSING",prepared?"1":"0",1);
        setenv("SDCF_DPF_SIGMOID",(mode=="dpf" || mode=="dpf_sigmoid")?"1":"0",1);
        setenv("SDCF_DPF_TANH",(mode=="dpf" || mode=="dpf_tanh")?"1":"0",1);
        setenv("SDCF_FOLDED_SIGMOID",(mode=="folded" || mode=="folded_sigmoid")?"1":"0",1);
        setenv("SDCF_FOLDED_TANH",(mode=="folded" || mode=="folded_tanh")?"1":"0",1);
        const int queries=argc>5?std::stoi(argv[5]):2;
        const auto setup = sdcf_measure::point();
        sci::IOPack io(role,port);
        sci::OTPack ot(&io,role);
        MathFunctions math(role,&io,&ot);
        TripleGenerator triples(role,&io,&ot);
        sdcf_native::setup(role,port,"127.0.0.1",math);
        sdcf_dpf::reserve_schedule(count,count);
        sdcf_folded::reserve_schedule(count,count);
        sdcf_planning::register_pack(&ot,0);
        sdcf_measure::report("setup",-1,setup);
        for (int query = 0; query < queries; ++query) {
            const auto offline = sdcf_measure::point();
            sdcf_preprocess::begin_prepare(query,prepared);
            if (prepared) {
                std::string traces[2];
                for (int pattern = 0; pattern < 2; ++pattern) {
                    ActivationSuite scratch{math,role,query,count,{},{},{}};
                    const auto before = sdcf_measure::point();
                    const auto prg_before = sdcf_planning::prg_blocks.load();
                    const auto triple_before = sdcf_triples::generation_epoch.load();
                    sdcf_planning::begin(); scratch.load(pattern); scratch.run(); traces[pattern] = sdcf_planning::end();
                    const auto after = sdcf_measure::point();
                    if (before.sent != after.sent || before.received != after.received ||
                        before.native_epoch != after.native_epoch || before.dpf_epoch != after.dpf_epoch ||
                        before.folded_epoch != after.folded_epoch ||
                        prg_before != sdcf_planning::prg_blocks.load() ||
                        triple_before != sdcf_triples::generation_epoch.load())
                        throw std::runtime_error("activation public planning changed protocol state");
                }
                if (traces[0] != traces[1]) throw std::runtime_error("activation values changed public material plan");
                sdcf_planning::compare_plan(io.io,role,traces[0]);
                std::cout << "ACTIVATION_PLAN {\"query\":" << query
                    << ",\"count\":" << count << ",\"scratch_patterns\":2,\"side_effects\":0,\"secret_independent\":true}\n";
                sdcf_planning::prepare_query_materials(query,mode == "missing_owner" ? std::vector<TripleGenerator*>{} : std::vector<TripleGenerator*>{&triples});
            }
            sdcf_preprocess::seal_prepare(query);
            io.io->sync(); sdcf_preprocess::material_ready(query);
            sdcf_measure::report("offline",query,offline);
            const auto input = sdcf_measure::point();
            sdcf_preprocess::begin_input(query);
            ActivationSuite suite{math,role,query,count,{},{},{}};
            suite.load(-1,&io);
            sdcf_measure::report("input",query,input);
            const auto ready = sdcf_measure::point();
            io.io->sync(); sdcf_measure::report("ready",query,ready);
            const auto online = sdcf_measure::point();
            sdcf_preprocess::enter_online(query);
            suite.run();
            sdcf_triples::balance(query);
            sdcf_dpf::balance(query);
            sdcf_folded::balance(query);
            sdcf_preprocess::leave_online(query);
            io.io->sync(); sdcf_measure::report("online",query,online);
            const auto output = sdcf_measure::point();
            suite.print(io);
            io.io->sync(); sdcf_preprocess::finish_query(query);
            sdcf_measure::report("output",query,output);
        }
        sdcf_native::finish();
        rusage usage{}; getrusage(RUSAGE_SELF,&usage);
        uint64_t vm_peak=0,vm_size=0;
        std::ifstream status("/proc/self/status");std::string key,rest;uint64_t value;
        while(status>>key) {
            if(key=="VmPeak:" || key=="VmSize:") {
                status>>value;
                if(key=="VmPeak:")vm_peak=value*1024;else vm_size=value*1024;
            }
            std::getline(status,rest);
        }
        std::cout << "ACTIVATION_RESOURCES {\"max_rss_kib\":" << usage.ru_maxrss
            << ",\"vm_peak_bytes\":" << vm_peak << ",\"vm_size_bytes\":" << vm_size << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ACTIVATION_ERROR " << error.what() << '\n'; return 2;
    }
}
