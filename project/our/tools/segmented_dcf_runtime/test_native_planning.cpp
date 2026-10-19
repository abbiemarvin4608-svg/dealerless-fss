#include "BuildingBlocks/aux-protocols.h"
#include "native_plan_replay.h"
#include "io_counters.h"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const int role = std::stoi(argv[1]), port = std::stoi(argv[2]);
        sci::IOPack io(role, port);
        sci::OTPack ot(&io, role);
        AuxProtocols aux(role, &io, &ot);
        sdcf_planning::register_pack(&ot, 0);
        for (int query = 0; query < 2; ++query) {
            sdcf_preprocess::begin_prepare(query, true);
            std::string traces[2];
            for (int pattern = 0; pattern < 2; ++pattern) {
                uint8_t scratch[9];
                uint64_t output[9];
                std::fill_n(scratch, 9, pattern);
                const auto epoch = sdcf_preprocess::generation_epoch.load();
                const auto sent = sdcf_measure::sent.load(), received = sdcf_measure::received.load();
                const auto counter = ot.iknp_straight->prg.counter;
                const auto random_blocks = sdcf_planning::prg_blocks.load();
                sdcf_planning::begin();
                aux.B2A(scratch, output, 9, 13);
                uint64_t scratch_messages[2][2] = {}, scratch_answers[2];
                uint64_t* scratch_pointers[2] = {scratch_messages[0], scratch_messages[1]};
                uint8_t scratch_choices[2] = {uint8_t(pattern), uint8_t(pattern)};
                if (role == 1) ot.iknp_straight->send(scratch_pointers, 2, 7);
                else ot.iknp_straight->recv(scratch_answers, scratch_choices, 2, 7);
                io.io->sync();
                traces[pattern] = sdcf_planning::end();
                if (sent != sdcf_measure::sent.load() || received != sdcf_measure::received.load() ||
                    epoch != sdcf_preprocess::generation_epoch.load() || counter != ot.iknp_straight->prg.counter ||
                    random_blocks != sdcf_planning::prg_blocks.load())
                    throw std::runtime_error("public planning changed protocol state");
            }
            if (traces[0] != traces[1]) throw std::runtime_error("private values changed public plan");
            sdcf_planning::compare_plan(io.io, role, traces[0]);
            sdcf_planning::prepare_ots(query);
            {
                sdcf_preprocess::GenerationPurposeScope purpose;
                uint64_t messages[2][2] = {{7, 19}, {8, 21}};
                uint64_t* pointers[2] = {messages[0], messages[1]};
                uint8_t choices[2] = {1, 0}; uint64_t answer[2];
                if (role == 1) ot.iknp_straight->send(pointers, 2, 7);
                else {
                    ot.iknp_straight->recv(answer, choices, 2, 7);
                    if (answer[0] != 19 || answer[1] != 8) throw std::runtime_error("nested offline GOT changed result");
                }
            }
            sdcf_preprocess::seal_prepare(query);
            io.io->sync(); sdcf_preprocess::material_ready(query);
            sdcf_preprocess::begin_input(query);
            uint8_t bits[9]; uint64_t result[9];
            for (int i = 0; i < 9; ++i) bits[i] = role == 1 ? uint8_t((i + query) % 2) : uint8_t(i % 3 == 0);
            sdcf_preprocess::enter_online(query);
            const auto epoch = sdcf_preprocess::generation_epoch.load();
            aux.B2A(bits, result, 9, 13);
            uint64_t messages[2][2] = {{7, 19}, {8, 21}}, answers[2];
            uint64_t* pointers[2] = {messages[0], messages[1]};
            uint8_t choices[2] = {1, 0};
            if (role == 1) ot.iknp_straight->send(pointers, 2, 7);
            else {
                ot.iknp_straight->recv(answers, choices, 2, 7);
                if (answers[0] != 19 || answers[1] != 8) throw std::runtime_error("nested protocol consumed query GOT stock");
            }
            sdcf_preprocess::leave_online(query);
            if (epoch != sdcf_preprocess::generation_epoch.load()) throw std::runtime_error("online extension");
            uint64_t remote[9];
            if (role == 1) io.io->send_data(result, sizeof(result));
            else {
                io.io->recv_data(remote, sizeof(remote));
                for (int i = 0; i < 9; ++i)
                    if (((remote[i] + result[i]) & 8191) != uint64_t(((i + query) % 2) ^ (i % 3 == 0)))
                        throw std::runtime_error("planned B2A changed result");
            }
            io.io->sync(); sdcf_preprocess::finish_query(query);
            std::cout << "PUBLIC_PLAN {\"query\":" << query << ",\"correct\":true,\"side_effects\":0}\n";
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << "PLANNING_ERROR " << error.what() << '\n'; return 2; }
}
