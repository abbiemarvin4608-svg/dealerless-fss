#include "OT/ot_pack.h"
#include "native_preprocessing.h"
#include "measurement.h"
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const int role = std::stoi(argv[1]), port = std::stoi(argv[2]);
        const std::string mode = argc > 3 ? argv[3] : "prepared";
        auto setup_start = sdcf_measure::point();
        sci::IOPack io(role, port);
        sci::OTPack ot(&io, role);
        sdcf_measure::report("setup", -1, setup_start);
        for (int query = 0; query < 2; ++query) {
            auto offline = sdcf_measure::point();
            const bool prepared = mode != "default";
            sdcf_preprocess::begin_prepare(query, prepared);
            if (prepared) {
                sdcf_preprocess::register_context(ot.iknp_straight, 0, "straight");
                sdcf_preprocess::plan(ot.iknp_straight,
                    {{query, 0, "iknp_got", 9, 13, 1}, {query, 1, "iknp_got", 3, 7, 1}});
                ot.iknp_straight->set_precomp_batch_size(12);
                if (mode == "unfinished_random_ot") {
                    bool random_choices[12] = {};
                    if (role == 1) ot.iknp_straight->send_pre(12);
                    else ot.iknp_straight->recv_pre(random_choices, 12);
                } else ot.iknp_straight->preprocess();
            }
            sdcf_preprocess::seal_prepare(query);
            io.io->sync();
            sdcf_preprocess::material_ready(query);
            sdcf_measure::report("offline", query, offline);
            auto input = sdcf_measure::point();
            sdcf_preprocess::begin_input(query);
            if (mode == "extend_during_input") ot.iknp_straight->send_pre(1);
            std::array<std::array<uint64_t, 2>, 9> messages{};
            std::array<uint64_t*, 9> pointers{};
            std::array<uint8_t, 9> choices{};
            std::array<uint64_t, 9> results{};
            for (int i = 0; i < 9; ++i) {
                messages[i] = {{uint64_t(100 + query * 20 + i), uint64_t(300 + query * 20 + i)}};
                pointers[i] = messages[i].data(); choices[i] = uint8_t((i + query) % 2);
            }
            sdcf_measure::report("input", query, input);
            auto ready = sdcf_measure::point();
            io.io->sync();
            sdcf_measure::report("ready", query, ready);
            auto online = sdcf_measure::point();
            sdcf_preprocess::enter_online(query);
            if (mode == "extend_after_ready") ot.iknp_straight->send_pre(1);
            if (mode == "wrong_contract") {
                if (role == 1) ot.iknp_straight->send(pointers.data(), 9, 12);
                else ot.iknp_straight->recv(results.data(), choices.data(), 9, 12);
            } else {
                if (role == 1) ot.iknp_straight->send(pointers.data(), 9, 13);
                else ot.iknp_straight->recv(results.data(), choices.data(), 9, 13);
                if (role == 2) for (int i = 0; i < 9; ++i)
                    if (results[i] != messages[i][choices[i]]) throw std::runtime_error("wrong selected message");
                if (mode != "incomplete") {
                    std::array<std::array<uint8_t, 2>, 3> small{{{{7, 17}}, {{8, 18}}, {{9, 19}}}};
                    std::array<uint8_t*, 3> small_ptr{{small[0].data(), small[1].data(), small[2].data()}};
                    std::array<uint8_t, 3> select{{1, 0, 1}}, answer{};
                    if (role == 1) ot.iknp_straight->send(small_ptr.data(), 3, 7);
                    else ot.iknp_straight->recv(answer.data(), select.data(), 3, 7);
                    if (role == 2 && answer != std::array<uint8_t, 3>{{17, 8, 19}})
                        throw std::runtime_error("wrong short selected message");
                }
                if (mode == "repeat") {
                    if (role == 1) ot.iknp_straight->send(pointers.data(), 9, 13);
                    else ot.iknp_straight->recv(results.data(), choices.data(), 9, 13);
                }
            }
            sdcf_preprocess::leave_online(query);
            io.io->sync();
            sdcf_measure::report("online", query, online);
            auto output = sdcf_measure::point();
            if (mode == "extend_during_output") ot.iknp_straight->send_pre(1);
            io.io->sync();
            sdcf_preprocess::finish_query(query);
            sdcf_measure::report("output", query, output);
            std::cout << "GOT_SELECTED {\"query\":" << query << ",\"correct\":true}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GOT_ERROR " << error.what() << '\n';
        return 2;
    }
}
