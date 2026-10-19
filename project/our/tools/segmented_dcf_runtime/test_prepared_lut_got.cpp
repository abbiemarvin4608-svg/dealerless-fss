#include "OT/ot_pack.h"
#include "BuildingBlocks/aux-protocols.h"
#include "LinearOT/linear-ot.h"
#include "prepared_got.h"
#include "measurement.h"
#include <iostream>
#include <array>
#include <vector>

template <typename T> void check_lut(AuxProtocols& aux, int role, int query,
                                     int radix_bits, int bits, int count) {
    const int radix = 1 << radix_bits;
    const uint64_t mask = bits == 64 ? ~0ULL : ((1ULL << bits) - 1);
    std::vector<std::vector<T>> table(count, std::vector<T>(radix));
    std::vector<T*> pointers(count);
    std::vector<T> choices(count), output(count);
    for (int i = 0; i < count; ++i) {
        pointers[i] = table[i].data(); choices[i] = (i * 7 + 31 + query * 13) & (radix - 1);
        for (int j = 0; j < radix; ++j)
            table[i][j] = T((0xfedcba9876540000ULL + query * 201 + i * 37 + j * 17) & mask);
    }
    aux.lookup_table<T>(role == 1 ? pointers.data() : nullptr,
        role == 2 ? choices.data() : nullptr, role == 2 ? output.data() : nullptr, count, radix_bits, bits);
    if (role == 2) for (int i = 0; i < count; ++i)
        if (output[i] != T((0xfedcba9876540000ULL + query * 201 + i * 37 + uint64_t(choices[i]) * 17) & mask))
            throw std::runtime_error("incorrect radix/width lookup result");
}

int main(int argc, char** argv) {
    try {
        const int role = std::stoi(argv[1]), port = std::stoi(argv[2]);
        const std::string mode = argc > 3 ? argv[3] : "prepared";
        sci::IOPack io(role, port);
        sci::OTPack ot(&io, role);
        AuxProtocols aux(role, &io, &ot);
        LinearOT linear(role, &io, &ot);
        const bool batched = mode.find("batched") != std::string::npos;
        const bool all_lut = mode.find("lut_all") != std::string::npos;
        const bool prepared = mode.find("default") == std::string::npos;
        for (int query = 0; query < 2; ++query) {
            const auto offline = sdcf_measure::point();
            sdcf_preprocess::begin_prepare(query, prepared);
            if (prepared && batched) {
                for (auto* selected : {ot.iknp_straight, ot.iknp_reversed}) {
                    sdcf_preprocess::register_context(selected, 0, selected == ot.iknp_straight ? "straight" : "reversed");
                    std::vector<sdcf_preprocess::MaterialCall> calls{
                        {query, 0, "iknp_batched_got", 9, 13, 11},
                        {query, 1, "iknp_batched_got", 9, 17, 11},
                        {query, 2, "iknp_batched_got", 7, 64, 3},
                        {query, 3, "iknp_batched_got", 5, 5, 1}};
                    sdcf_preprocess::plan(selected, calls);
                    for (const auto& call : calls) sdcf_preprocess::prepare_batched_got(selected, call);
                }
            } else if (prepared && all_lut) {
                for (int r = 1; r <= 8; ++r) {
                    auto* selected = ot.kkot[r - 1];
                    sdcf_preprocess::register_context(selected, 0, "kkot_" + std::to_string(1 << r));
                    const int word_bits = r == 8 ? 64 : 13, byte_bits = r % 3 == 0 ? 1 : r % 3 == 1 ? 7 : 8;
                    std::vector<sdcf_preprocess::MaterialCall> calls{
                        {query, 0, "kkot_got", 11, word_bits, 1 << r},
                        {query, 1, "kkot_got", 5, byte_bits, 1 << r}};
                    sdcf_preprocess::plan(selected, calls);
                    for (const auto& call : calls) sdcf_preprocess::prepare_kkot(selected, call);
                }
            } else if (prepared) {
                auto* selected = ot.kkot[2];
                sdcf_preprocess::register_context(selected, 0, "kkot_8");
                sdcf_preprocess::MaterialCall call{query, 0, "kkot_got", 11, 13, 8};
                sdcf_preprocess::plan(selected, {call});
                sdcf_preprocess::prepare_kkot(selected, call);
            }
            sdcf_preprocess::seal_prepare(query);
            io.io->sync();
            sdcf_preprocess::material_ready(query);
            sdcf_measure::report("offline", query, offline);
            const auto input = sdcf_measure::point();
            sdcf_preprocess::begin_input(query);
            std::vector<std::vector<uint64_t>> table(11, std::vector<uint64_t>(8));
            std::vector<uint64_t*> rows(11);
            std::vector<uint64_t> choices(11), output(11);
            for (int i = 0; i < 11; ++i) {
                rows[i] = table[i].data(); choices[i] = (i * 3 + query) % 8;
                for (int j = 0; j < 8; ++j) table[i][j] = 1000 + query * 100 + i * 8 + j;
            }
            sdcf_measure::report("input", query, input);
            const auto ready = sdcf_measure::point();
            io.io->sync(); sdcf_measure::report("ready", query, ready);
            const auto online = sdcf_measure::point();
            sdcf_preprocess::enter_online(query);
            if (batched) {
                std::vector<uint64_t> bitA(9), valueB(33), product(33), peer(33);
                const uint64_t clear_bits[9] = {1, 0, 1, 0, 1, 0, 1, 1, 0};
                for (int i = 0; i < 9; ++i) bitA[i] = role == 1 ? uint64_t(i % 2) : uint64_t(i % 2) ^ clear_bits[i];
                for (int i = 0; i < 33; ++i) {
                    const uint64_t clear = 20 + 20 * (i / 11) + i % 11 + 17 * query;
                    valueB[i] = role == 1 ? 7 + i : clear - (7 + i);
                }
                linear.matmul_multiplexer(3, 3, 11, bitA.data(), valueB.data(), product.data(),
                                           1, 13, 13, true, MultMode::None);
                io.io->send_data(product.data(), 33 * sizeof(uint64_t));
                io.io->recv_data(peer.data(), 33 * sizeof(uint64_t));
                for (int i = 0; i < 33; ++i) {
                    const int row = i / 11, col = i % 11;
                    const uint64_t expected = row == 0 ? 80 + 2 * col + 34 * query :
                        row == 1 ? 40 + col + 17 * query : 60 + 2 * col + 34 * query;
                    if (((product[i] + peer[i]) & 8191) != expected)
                        throw std::runtime_error("incorrect row-batched bit multiplication");
                }
                std::vector<uint64_t> valueA(33), bitB(9);
                for (int i = 0; i < 33; ++i) {
                    const uint64_t clear = 20 + 20 * (i % 3) + i / 3 + 17 * query;
                    valueA[i] = role == 1 ? 3 + i : clear - (3 + i);
                }
                for (int i = 0; i < 9; ++i) bitB[i] = role == 1 ? uint64_t(i % 2) : uint64_t(i % 2) ^ clear_bits[i];
                linear.matmul_multiplexer(11, 3, 3, valueA.data(), bitB.data(), product.data(),
                                           17, 1, 17, true, MultMode::None);
                io.io->send_data(product.data(), 33 * sizeof(uint64_t));
                io.io->recv_data(peer.data(), 33 * sizeof(uint64_t));
                for (int i = 0; i < 33; ++i) {
                    const int row = i / 3, col = i % 3;
                    const uint64_t expected = col == 0 ? 80 + 2 * row + 34 * query :
                        col == 1 ? 100 + 2 * row + 34 * query : 20 + row + 17 * query;
                    if (((product[i] + peer[i]) & 131071) != expected)
                        throw std::runtime_error("incorrect column-batched bit multiplication");
                }
                for (auto* selected : {ot.iknp_straight, ot.iknp_reversed}) {
                    for (const auto& request : {std::array<int, 3>{7, 64, 3}, std::array<int, 3>{5, 5, 1}}) {
                        const int count = request[0], bits = request[1],
                                  group = request[2] + (mode == "batched_wrong_contract" && bits == 64 ? 1 : 0);
                        std::vector<uint64_t> messages(2 * count * group), received(count * group);
                        std::vector<uint8_t> selections(count);
                        const uint64_t mask = bits == 64 ? ~0ULL : ((1ULL << bits) - 1);
                        for (int i = 0; i < count; ++i) {
                            selections[i] = (i + query) % 2;
                            for (int h = 0; h < group; ++h) for (int b = 0; b < 2; ++b)
                                messages[(i * group + h) * 2 + b] =
                                    (0xfedcba9876540000ULL + query * 999 + i * 117 + h * 13 + b * 391) & mask;
                        }
                        if (selected->party == 1) selected->send_batched_got(messages.data(), count, bits, group);
                        else {
                            selected->recv_batched_got(received.data(), selections.data(), count, bits, group);
                            for (int i = 0; i < count; ++i) for (int h = 0; h < group; ++h)
                                if (received[i * group + h] != messages[(i * group + h) * 2 + selections[i]])
                                    throw std::runtime_error("incorrect long grouped message");
                        }
                    }
                }
                if (mode == "batched_repeat") {
                    std::array<uint64_t, 10> repeated{};
                    std::array<uint8_t, 5> selected{};
                    if (role == 1) ot.iknp_straight->send_batched_got(repeated.data(), 5, 5, 1);
                    else ot.iknp_straight->recv_batched_got(repeated.data(), selected.data(), 5, 5, 1);
                }
            } else if (all_lut) {
                for (int r = 1; r <= 8; ++r) {
                    check_lut<uint64_t>(aux, role, query, r, r == 8 ? 64 : 13, 11);
                    check_lut<uint8_t>(aux, role, query, r, r % 3 == 0 ? 1 : r % 3 == 1 ? 7 : 8, 5);
                }
            } else aux.lookup_table<uint64_t>(role == 1 ? rows.data() : nullptr,
                role == 2 ? choices.data() : nullptr, role == 2 ? output.data() : nullptr, 11, 3, 13);
            if (!batched && !all_lut && role == 2) for (int i = 0; i < 11; ++i)
                if (output[i] != 1000 + query * 100 + i * 8 + choices[i])
                    throw std::runtime_error("incorrect lookup result");
            sdcf_preprocess::leave_online(query);
            io.io->sync(); sdcf_measure::report("online", query, online);
            const auto output_start = sdcf_measure::point();
            sdcf_preprocess::finish_query(query);
            sdcf_measure::report("output", query, output_start);
            std::cout << (batched ? "BATCHED_GOT_RESULT " : "LUT_GOT_RESULT ")
                << "{\"query\":" << query << ",\"correct\":true}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LUT_GOT_ERROR " << error.what() << '\n'; return 2;
    }
}
