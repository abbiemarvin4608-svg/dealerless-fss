#include "Math/math-functions.h"
#include "native_query_materials.h"
#include "measurement.h"
#include <array>
#include <sys/resource.h>

namespace {
const std::array<int64_t, 25> clear_values{{
    0,1,-1,127,-128,128,-129,1023,-1024,32767,-32768,32766,-32767,
    255,-256,256,-257,4095,-4096,4096,-4097,16383,-16384,8191,-8192}};

struct ActivationSuite {
    MathFunctions& math;
    int role, query, count;
    std::vector<uint64_t> input, sigmoid_output, tanh_output;

    void load(int pattern = -1) {
        input.resize(count);
        for (int i = 0; i < count; ++i) {
            if (pattern >= 0) input[i] = pattern ? 65535 : 0;
            else {
                const uint64_t clear = uint64_t(clear_values[(i + query * 7) % clear_values.size()]) & 65535;
                const uint64_t alice = (0xfedcba98ULL + i * 1987 + query * 337) & 65535;
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
        activation("sigmoid", [&] { math.sigmoid(count,input.data(),sigmoid_output.data(),16,16,7,14); });
        activation("tanh", [&] { math.tanh(count,input.data(),tanh_output.data(),16,16,7,7); });
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
        const auto setup = sdcf_measure::point();
        sci::IOPack io(role,port);
        sci::OTPack ot(&io,role);
        MathFunctions math(role,&io,&ot);
        TripleGenerator triples(role,&io,&ot);
        sdcf_planning::register_pack(&ot,0);
        sdcf_measure::report("setup",-1,setup);
        for (int query = 0; query < 2; ++query) {
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
                        before.native_epoch != after.native_epoch ||
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
            suite.load();
            sdcf_measure::report("input",query,input);
            const auto ready = sdcf_measure::point();
            io.io->sync(); sdcf_measure::report("ready",query,ready);
            const auto online = sdcf_measure::point();
            sdcf_preprocess::enter_online(query);
            suite.run();
            sdcf_triples::balance(query);
            sdcf_preprocess::leave_online(query);
            io.io->sync(); sdcf_measure::report("online",query,online);
            const auto output = sdcf_measure::point();
            suite.print(io);
            io.io->sync(); sdcf_preprocess::finish_query(query);
            sdcf_measure::report("output",query,output);
        }
        rusage usage{}; getrusage(RUSAGE_SELF,&usage);
        std::cout << "ACTIVATION_RESOURCES {\"max_rss_kib\":" << usage.ru_maxrss << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ACTIVATION_ERROR " << error.what() << '\n'; return 2;
    }
}
