#include "BuildingBlocks/aux-protocols.h"
#include "LinearOT/linear-ot.h"
#include "measurement.h"
#include "prepared_cot.h"
#include <array>
#include <iostream>
#include <stdexcept>

void arithmetic_edges(sci::SplitIKNP<sci::NetIO>* direction, int query) {
    const std::vector<sdcf_arithmetic::Call> calls{{13, {64}, 1, false}, {1, {1}, 1, false},
        {257, {33}, 1, false}, {51, {64,17,1}, 5, true}};
    for (const auto& call : calls) {
        auto choices = std::make_unique<bool[]>(call.count);
        std::vector<uint64_t> correlations(size_t(call.count) * call.messages_per_ot), output(correlations.size());
        for (int i = 0; i < call.count; ++i) {
            choices[i] = (i + query) % 2;
            for (int lane = 0; lane < call.messages_per_ot; ++lane) {
                const size_t index = size_t(i) * call.messages_per_ot + lane;
                correlations[index] = (index % 3 == 0 ? UINT64_MAX : index % 3 == 1 ? 0 : uint64_t(1) << 63);
            }
        }
        if (direction->party == sci::ALICE) {
            if (call.batched) direction->send_batched_cot(output.data(), correlations.data(), call.widths,
                                                         call.count, call.messages_per_ot);
            else direction->send_cot(output.data(), correlations.data(), call.count, call.widths[0]);
        } else {
            if (call.batched) direction->recv_batched_cot(output.data(), choices.get(), call.widths,
                                                         call.count, call.messages_per_ot);
            else direction->recv_cot(output.data(), choices.get(), call.count, call.widths[0]);
        }
        // This diagnostic opening is charged to online in both compared modes.
        if (direction->party == sci::ALICE) direction->io->send_data(output.data(), int(output.size() * 8));
        else {
            std::vector<uint64_t> sender(output.size());
            direction->io->recv_data(sender.data(), int(sender.size() * 8));
            for (int i = 0; i < call.count; ++i)
                for (int lane = 0; lane < call.messages_per_ot; ++lane) {
                    const size_t index = size_t(i) * call.messages_per_ot + lane;
                    const uint64_t expected = sender[index] + (choices[i] ? correlations[index] : 0);
                    if (output[index] != (expected & sdcf_arithmetic::mask(sdcf_arithmetic::width_at(call, i))))
                        throw std::runtime_error("arithmetic COT ring relation mismatch");
                }
        }
    }
}

int main(int argc, char** argv) {
    try {
        const int role = std::stoi(argv[1]), port = std::stoi(argv[2]);
        const std::string mode = argc > 3 ? argv[3] : "prepared";
        auto setup = sdcf_measure::point();
        sci::IOPack io(role, port);
        sci::OTPack ot(&io, role);
        AuxProtocols aux(role, &io, &ot);
        LinearOT mult(role, &io, &ot);
        sdcf_measure::report("setup", -1, setup);
        for (int query = 0; query < 2; ++query) {
            auto offline = sdcf_measure::point();
            const bool multiplication = mode.find("multiplication") == 0;
            const bool edges = mode.find("edges") == 0;
            const bool mismatch = mode == "width_mismatch" || mode == "group_mismatch";
            const bool prepared = mode != "default" && mode != "multiplication_default" && mode != "edges_default";
            sdcf_preprocess::begin_prepare(query, prepared);
            if (mode == "offline_cot") {
                uint64_t output[9]{}, random_correlation[9]{11,23,6,2,73,15,21,9,4};
                bool random_choice[9]{};
                if (role == 1) ot.iknp_straight->send_cot(output, random_correlation, 9, 13);
                else ot.iknp_straight->recv_cot(output, random_choice, 9, 13);
            }
            if (prepared) {
                sdcf_preprocess::register_context(ot.iknp_straight, 0, "straight");
                if (edges) {
                    sdcf_preprocess::register_context(ot.iknp_reversed, 0, "reversed");
                    for (auto* direction : {ot.iknp_straight, ot.iknp_reversed}) {
                        sdcf_preprocess::plan(direction, {{query, 0, "iknp_cot", 13, 64, 1},
                            {query, 1, "iknp_cot", 1, 1, 1}, {query, 2, "iknp_cot", 257, 33, 1},
                            {query, 3, "iknp_batched_cot", 51, 64, 5}});
                        for (const auto& call : std::vector<sdcf_arithmetic::Call>{{13, {64}, 1, false},
                            {1, {1}, 1, false}, {257, {33}, 1, false}, {51, {64,17,1}, 5, true}})
                            direction->prepare_arithmetic_cot({call});
                    }
                } else if (mismatch) {
                    sdcf_preprocess::plan(ot.iknp_straight, {{query, 0, "iknp_batched_cot", 6, 64, 3}});
                    ot.iknp_straight->prepare_arithmetic_cot({{6, {64,17,1}, 3, true}});
                } else if (multiplication) {
                    sdcf_preprocess::register_context(ot.iknp_reversed, 0, "reversed");
                    sdcf_preprocess::plan(ot.iknp_straight,
                        {{query, 0, "iknp_batched_cot", 14, 7, 3}, {query, 1, "iknp_batched_cot", 35, 7, 1}});
                    sdcf_preprocess::plan(ot.iknp_reversed,
                        {{query, 0, "iknp_batched_cot", 14, 7, 3}, {query, 1, "iknp_batched_cot", 35, 7, 1}});
                    const std::vector<sdcf_arithmetic::Call> calls{{14, {7,6,5,4,3,2,1}, 3, true},
                                                               {35, {7,6,5,4,3,2,1}, 1, true}};
                    ot.iknp_straight->prepare_arithmetic_cot(calls);
                    ot.iknp_reversed->prepare_arithmetic_cot(calls);
                } else {
                    sdcf_preprocess::plan(ot.iknp_straight, {{query, 0, "iknp_cot", 9, 13, 1}});
                    if (mode == "unfinished_random") {
                        bool random_choices[9]{};
                        if (role == 1) ot.iknp_straight->send_pre(9);
                        else ot.iknp_straight->recv_pre(random_choices, 9);
                    } else ot.iknp_straight->prepare_arithmetic_cot({{9, {13}, 1, false}});
                }
            }
            sdcf_preprocess::seal_prepare(query);
            io.io->sync();
            sdcf_preprocess::material_ready(query);
            sdcf_measure::report("offline", query, offline);
            auto input = sdcf_measure::point();
            sdcf_preprocess::begin_input(query);
            std::array<uint8_t, 9> bits{};
            std::array<uint64_t, 9> output{};
            for (int i = 0; i < 9; ++i) bits[i] = uint8_t((i + query + role / 2) % 2);
            std::array<uint64_t, 2> matrix_a{{role == 1 ? 37u : 89u, role == 1 ? 16u : 115u}};
            std::array<uint64_t, 3> matrix_b{{role == 1 ? 42u : 90u, role == 1 ? 3u : 124u,
                                           role == 1 ? 13u : 122u}};
            std::array<uint64_t, 5> hadamard_a{{role == 1 ? 5u : 121u, role == 1 ? 10u : 54u,
                role == 1 ? 10u : 53u, role == 1 ? 8u : 121u, role == 1 ? 14u : 111u}};
            std::array<uint64_t, 5> hadamard_b{{role == 1 ? 4u : 123u, role == 1 ? 0u : 2u,
                role == 1 ? 9u : 116u, role == 1 ? 2u : 125u, role == 1 ? 4u : 125u}};
            std::array<uint64_t, 6> matrix_output{};
            std::array<uint64_t, 5> hadamard_output{};
            sdcf_measure::report("input", query, input);
            io.io->sync();
            auto online = sdcf_measure::point();
            sdcf_preprocess::enter_online(query);
            if (mode == "online_extension") ot.iknp_straight->send_pre(9);
            if (edges) {
                arithmetic_edges(ot.iknp_straight, query);
                arithmetic_edges(ot.iknp_reversed, query);
            } else if (mismatch) {
                uint64_t data[18]{}, corr[18]{};
                bool choices[6]{};
                std::vector<int> widths = mode == "width_mismatch" ? std::vector<int>{64,18,1} : std::vector<int>{64,17,1};
                const int lanes = mode == "group_mismatch" ? 2 : 3;
                if (role == 1) ot.iknp_straight->send_batched_cot(data, corr, widths, 6, lanes);
                else ot.iknp_straight->recv_batched_cot(data, choices, widths, 6, lanes);
            } else if (multiplication) {
                mult.matrix_multiplication(2, 1, 3, matrix_a.data(), matrix_b.data(), matrix_output.data(),
                                           7, 7, 7, true, true, true);
                mult.hadamard_product(5, hadamard_a.data(), hadamard_b.data(), hadamard_output.data(),
                                     7, 7, 7, true, true);
            } else if (mode != "incomplete") {
                aux.B2A(bits.data(), output.data(), 9, 13);
                if (mode == "repeat") aux.B2A(bits.data(), output.data(), 9, 13);
            }
            sdcf_preprocess::leave_online(query);
            io.io->sync();
            sdcf_measure::report("online", query, online);
            auto reveal = sdcf_measure::point();
            if (edges) {
            } else if (multiplication) {
                if (role == 1) {
                    io.io->send_data(matrix_output.data(), sizeof(matrix_output));
                    io.io->send_data(hadamard_output.data(), sizeof(hadamard_output));
                } else {
                    std::array<uint64_t, 6> other_matrix{};
                    std::array<uint64_t, 5> other_hadamard{};
                    io.io->recv_data(other_matrix.data(), sizeof(other_matrix));
                    io.io->recv_data(other_hadamard.data(), sizeof(other_hadamard));
                    const std::array<uint64_t, 6> expected_matrix{{120,2,114,12,125,21}};
                    const std::array<uint64_t, 5> expected_hadamard{{2,0,67,127,125}};
                    for (int i = 0; i < 6; ++i)
                        if (((other_matrix[i] + matrix_output[i]) & 127) != expected_matrix[i])
                            throw std::runtime_error("MatMul signed integer reconstruction mismatch");
                    for (int i = 0; i < 5; ++i)
                        if (((other_hadamard[i] + hadamard_output[i]) & 127) != expected_hadamard[i])
                            throw std::runtime_error("MulCir signed integer reconstruction mismatch");
                }
            } else if (role == 1) io.io->send_data(output.data(), sizeof(output));
            else {
                std::array<uint64_t, 9> other{};
                io.io->recv_data(other.data(), sizeof(other));
                for (int i = 0; i < 9; ++i)
                    if (((other[i] + output[i]) & 8191) != 1)
                        throw std::runtime_error("B2A reconstruction differs from XOR");
            }
            io.io->sync();
            sdcf_preprocess::finish_query(query);
            sdcf_measure::report("output", query, reveal);
            std::cout << "COT_RESULT {\"query\":" << query << ",\"correct\":true}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "COT_ERROR " << error.what() << '\n';
        return 2;
    }
}
