#include "sci_numeric_oracle.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace segfss::reference;

void check(uint64_t input, const Contract& contract, uint64_t expected) {
    if (evaluate(input, contract) != expected)
        throw std::runtime_error("numeric oracle literal mismatch");
}

int main() {
    check(253, {8, 8, 1}, 255);  // -3 / 2 = -1
    check(253, {8, 8, 1, SciCall::truncate}, 254);
    check(253, {8, 7, 1, SciCall::truncate_and_reduce}, 126);
    check(253, {8, 16, 1}, 65535);
    check(128, {8, 16, 0, SciCall::convert}, 65408);
    check(253, {8, 16, -2, SciCall::convert}, 65524);
    check(127, {8, 8, -2, SciCall::convert}, 252);
    check(253, {8, 7, 1, SciCall::div_pow2, true,
                NumericMode::exact_floor_diagnostic}, 126);
    check(253, {8, 16, 1, SciCall::div_pow2, false}, 126);
    check(128, {8, 8, 7}, 255);
    check(129, {8, 8, 7}, 0);
    check(0x8000000000000000ULL, {64, 64, 63}, UINT64_MAX);
    check(0x8000000000000001ULL, {64, 64, 63}, 0);
    check(UINT64_MAX, {64, 64, 1}, 0);
    check(UINT64_MAX, {64, 64, 1, SciCall::truncate}, UINT64_MAX);
    check(0x8000000000000000ULL, {64, 64, 0}, 0x8000000000000000ULL);
    const Contract invalid[] = {
        {0, 8, 1}, {65, 8, 1}, {8, 0, 1}, {8, 65, 1},
        {8, 8, 8}, {8, 8, -1}, {8, 8, 1, SciCall::convert},
        {8, 8, -64, SciCall::convert},
        {8, 8, 1, SciCall::truncate_and_reduce},
        {8, 8, 1, SciCall::div_pow2, true, NumericMode::exact_floor_diagnostic}
    };
    for (const auto& contract : invalid) {
        bool rejected = false;
        try { evaluate(1, contract); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid contract accepted");
    }
    std::cout << "PASS literal vectors=16 invalid contracts=10\n";
}
