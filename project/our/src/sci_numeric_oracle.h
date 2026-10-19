#pragma once

#include <cstdint>
#include <stdexcept>

namespace segfss::reference {

enum class NumericMode { sci_compatible, exact_floor_diagnostic };
enum class SciCall { div_pow2, truncate, truncate_and_reduce, convert };

struct Contract {
    int input_bits;
    int output_bits;
    int shift;
    SciCall call = SciCall::div_pow2;
    bool signed_arithmetic = true;
    NumericMode numeric_mode = NumericMode::sci_compatible;
};

inline uint64_t mask(int bits) {
    if (bits < 1 || bits > 64) throw std::invalid_argument("ring bits must be in [1,64]");
    return bits == 64 ? UINT64_MAX : (uint64_t{1} << bits) - 1;
}

inline __int128 signedValue(uint64_t value, int bits) {
    value &= mask(bits);
    return static_cast<__int128>(value) -
        ((value >> (bits - 1)) ? (static_cast<__int128>(1) << bits) : 0);
}

inline void validate(const Contract& c) {
    mask(c.input_bits);
    mask(c.output_bits);
    if (c.numeric_mode != NumericMode::sci_compatible &&
        c.numeric_mode != NumericMode::exact_floor_diagnostic)
        throw std::invalid_argument("unknown numeric mode");
    if (c.call != SciCall::div_pow2 && c.call != SciCall::truncate &&
        c.call != SciCall::truncate_and_reduce && c.call != SciCall::convert)
        throw std::invalid_argument("unknown SCI call");
    if (c.call == SciCall::convert) {
        if (c.shift > 0 || c.shift < -63)
            throw std::invalid_argument("conversion requires shift in [-63,0]");
    } else if (c.shift < 0 || c.shift >= c.input_bits) {
        throw std::invalid_argument("truncation requires 0 <= shift < input_bits");
    }
    if (c.call == SciCall::truncate_and_reduce &&
        c.output_bits != c.input_bits - c.shift)
        throw std::invalid_argument("truncate_and_reduce output ring must shrink by shift");
    if (c.numeric_mode == NumericMode::exact_floor_diagnostic &&
        (c.shift < 0 || c.output_bits != c.input_bits - c.shift))
        throw std::invalid_argument("floor diagnostic is an independent shrinking-ring contract");
}

// Plaintext reference only: input is the reconstructed residue, never a share.
inline uint64_t evaluate(uint64_t input, const Contract& c) {
    validate(c);
    input &= mask(c.input_bits);
    __int128 value = c.signed_arithmetic ? signedValue(input, c.input_bits)
                                       : static_cast<__int128>(input);
    if (c.shift < 0) {
        // Multiplication avoids implementation-defined shifting of a negative value.
        value *= static_cast<__int128>(1) << -c.shift;
    } else if (c.shift > 0) {
        const __int128 divisor = static_cast<__int128>(1) << c.shift;
        if (value >= 0) {
            value /= divisor;
        } else {
            const __int128 magnitude = -value;
            const bool floor = c.numeric_mode == NumericMode::exact_floor_diagnostic ||
                               c.call != SciCall::div_pow2;
            value = -(magnitude / divisor) - (floor && magnitude % divisor != 0);
        }
    }
    return static_cast<uint64_t>(value) & mask(c.output_bits);
}

// SCI's direct signed division has a zero-width internal comparison at this edge.
inline bool nativeDirectSupported(const Contract& c) {
    validate(c);
    return !(c.call == SciCall::div_pow2 && c.signed_arithmetic &&
             c.shift > 0 && c.shift == c.input_bits - 1);
}

}  // namespace segfss::reference
