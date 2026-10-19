#include "sci_numeric_oracle.h"
#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/value-extension.h"
#include "defines.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <array>
#include <vector>

using namespace segfss::reference;

std::vector<uint64_t> values(int bits) {
    std::vector<uint64_t> result;
    if (bits <= 8) {
        for (uint64_t x = 0; x <= mask(bits); ++x) result.push_back(x);
    } else {
        const auto sign = uint64_t{1} << (bits - 1);
        result = {0, 1, 2, 3, sign - 2, sign - 1, sign, sign + 1,
                  sign + 2, mask(bits) - 2, mask(bits) - 1, mask(bits)};
        std::mt19937_64 rng(20261009 + bits);
        for (int i = 0; i < 64; ++i) result.push_back(rng() & mask(bits));
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
#ifndef DIV_RESCALING
    throw std::runtime_error("frozen contract requires DIV_RESCALING");
#endif
    const int role = std::stoi(argv[1]);
    sci::IOPack io(role, std::stoi(argv[2]), "127.0.0.1");
    sci::OTPack ot(&io, role);
    Truncation trunc(role, &io, &ot);
    XTProtocol xt(role, &io, &ot);
    std::ofstream vectors;
    if (role == 2) {
        vectors.open(argv[3]);
        vectors << "input_bits,output_bits,shift,call,signed,input,SCI,oracle,path\n";
    }
    uint64_t checked = 0, groups = 0, widened = 0, skipped = 0;
    auto cross = [&](const Contract& c) {
        if (!nativeDirectSupported(c) && c.input_bits == 64) {
            ++skipped;
            std::cout << "UNSUPPORTED native signed div_pow2 bits=64 shift=63\n";
            return true;
        }
        const auto clear = values(c.input_bits);
        std::vector<uint64_t> input(clear.size() * 3), output(input.size()), other(input.size());
        std::mt19937_64 rng(0xdecafbad + c.input_bits);
        for (size_t i = 0; i < input.size(); ++i) {
            const uint64_t share = (i % 3 == 0 ? 0 : i % 3 == 1 ? mask(c.input_bits) : rng()) & mask(c.input_bits);
            input[i] = role == 1 ? share : (clear[i / 3] - share) & mask(c.input_bits);
        }
        int native_bits = c.input_bits;
        const bool wide = !nativeDirectSupported(c);
        if (wide) {
            xt.s_extend(input.size(), input.data(), output.data(), c.input_bits, c.input_bits + 1);
            input.swap(output);
            native_bits++;
            ++widened;
        }
        int result_bits = native_bits;
        if (c.call == SciCall::convert) {
            if (c.output_bits > native_bits) {
                if (c.signed_arithmetic) xt.s_extend(input.size(), input.data(), output.data(), native_bits, c.output_bits);
                else xt.z_extend(input.size(), input.data(), output.data(), native_bits, c.output_bits);
            } else output = input;
            for (auto& x : output) x = (x << -c.shift) & mask(c.output_bits);
            result_bits = c.output_bits;
        } else if (c.call == SciCall::div_pow2) {
            trunc.div_pow2(input.size(), input.data(), output.data(), c.shift, native_bits, c.signed_arithmetic);
        } else if (c.call == SciCall::truncate) {
            trunc.truncate(input.size(), input.data(), output.data(), c.shift, native_bits, c.signed_arithmetic);
        } else {
            trunc.truncate_and_reduce(input.size(), input.data(), output.data(), c.shift, native_bits);
            result_bits -= c.shift;
        }
        if (c.output_bits > result_bits) {
            input.swap(output);
            if (c.signed_arithmetic) xt.s_extend(input.size(), input.data(), output.data(), result_bits, c.output_bits);
            else xt.z_extend(input.size(), input.data(), output.data(), result_bits, c.output_bits);
        }
        if (role == 1) io.io->send_data(output.data(), output.size() * sizeof(uint64_t));
        else {
            io.io->recv_data(other.data(), other.size() * sizeof(uint64_t));
            for (size_t i = 0; i < output.size(); ++i) {
                const auto actual = (output[i] + other[i]) & mask(c.output_bits);
                const auto expected = evaluate(clear[i / 3], c);
                if (i < 36) vectors << c.input_bits << ',' << c.output_bits << ',' << c.shift << ',' << int(c.call) << ',' << c.signed_arithmetic << ',' << clear[i / 3] << ',' << actual << ',' << expected << ',' << (wide ? "widened" : "direct") << '\n';
                if (actual != expected) {
                    std::cerr << "MISMATCH bits=" << c.input_bits << " out=" << c.output_bits << " shift=" << c.shift << " call=" << int(c.call) << " signed=" << c.signed_arithmetic << " input=" << clear[i / 3] << " actual=" << actual << " expected=" << expected << '\n';
                    return false;
                }
            }
        }
        checked += output.size();
        ++groups;
        std::cout << "GROUP bits=" << c.input_bits << " out=" << c.output_bits << " shift=" << c.shift << " call=" << int(c.call) << " signed=" << c.signed_arithmetic << " values=" << output.size() << " path=" << (wide ? "widened" : "direct") << '\n';
        return true;
    };
    for (int bits : {1,2,3,4,5,6,7,8,12,14,16,20,24,25,28,29,30,31,32,33,36,38,40,63,64}) {
        std::set<int> shifts{0, 1, bits / 2, bits - 1};
        if (bits <= 8) for (int s = 0; s < bits; ++s) shifts.insert(s);
        for (int shift : shifts) {
            if (shift >= bits) continue;
            for (bool sign : {false, true}) {
                for (auto call : {SciCall::div_pow2, SciCall::truncate}) {
                    if (!cross({bits, bits, shift, call, sign})) return 1;
                }
            }
            if (!cross({bits, bits - shift, shift, SciCall::truncate_and_reduce})) return 1;
        }
        for (int out : std::set<int>{1, bits, 64})
            for (int shift : {0, -1, -5})
                for (bool sign : {false, true})
                    if (!cross({bits, out, shift, SciCall::convert, sign})) return 1;
    }
    for (const auto& shape : std::vector<std::array<int, 4>>{
             {29,5,1,16}, {28,4,3,16}, {36,4,11,16},
             {31,7,7,8}, {22,6,-6,16}, {30,6,0,16}, {31,7,9,8}}) {
        const int bits = shape[0], first = shape[1], second = shape[2], out = shape[3];
        const auto clear = values(bits);
        std::vector<uint64_t> a(clear.size()), b(clear.size()), c(clear.size()), remote(clear.size());
        std::mt19937_64 rng(147 + bits);
        for (size_t i = 0; i < clear.size(); ++i) {
            const auto share = rng() & mask(bits);
            a[i] = role == 1 ? share : (clear[i] - share) & mask(bits);
        }
        trunc.truncate_and_reduce(a.size(), a.data(), b.data(), first, bits);
        const Contract stage1{bits, bits-first, first, SciCall::truncate_and_reduce};
        const Contract stage2{bits-first, out, second, second > 0 ? SciCall::div_pow2 : SciCall::convert};
        if (second > 0) trunc.div_pow2(b.size(), b.data(), c.data(), second, bits-first, true);
        else for (size_t i = 0; i < b.size(); ++i) c[i] = b[i] << -second;
        if (role == 1) io.io->send_data(c.data(), c.size()*sizeof(uint64_t));
        else {
            io.io->recv_data(remote.data(), remote.size()*sizeof(uint64_t));
            for (size_t i = 0; i < clear.size(); ++i) {
                const auto expected = evaluate(evaluate(clear[i], stage1), stage2);
                if (((c[i]+remote[i])&mask(out)) != expected)
                    throw std::runtime_error("SCI MatMul floor/division chain differs from oracle");
            }
        }
        checked += clear.size();
        ++groups;
        std::cout << "CHAIN accumulator_bits=" << bits << " first_floor_shift=" << first << " second_shift=" << second << " output_bits=" << out << " values=" << clear.size() << " status=passed\n";
    }
    std::cout << "SUMMARY checked=" << checked << " groups=" << groups << " widened_groups=" << widened << " native_direct_unsupported=" << skipped << " status=passed\n";
}
