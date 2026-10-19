#include "model_bias.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    assert(segfss::model::divideBiasBy8(0) == 0);
    assert(segfss::model::divideBiasBy8(7) == 0);
    assert(segfss::model::divideBiasBy8(8) == 1);
    assert(segfss::model::divideBiasBy8(-7) == 0);
    assert(segfss::model::divideBiasBy8(-8) == -1);
    assert(segfss::model::divideBiasBy8(-9) == -1);
    assert(segfss::model::encodeBias(65535) == 8191);
    // Model constants are reduced in the 16-bit ring.  They are not clipped
    // to the host int16 range before encoding.
    assert(segfss::model::encodeBias(8LL * 32768) == 32768);
    assert(segfss::model::encodeBias(-8LL * 32769) == 32767);
    assert(segfss::model::divideBiasBy8(8LL * 32768) == -32768);
    assert(segfss::model::divideBiasBy8(-8LL * 32769) == 32767);

    std::vector<int64_t> g0(100), g1(100);
    for (int i = 0; i < 100; ++i) {
        g0[i] = i * 17 - 900;
        g1[i] = 1200 - i * 19;
    }
    const auto material = segfss::model::prepareExactBias(g0, g1, 42);
    assert(material.validFor(42));
    assert(!material.validFor(43));
    assert(material.residentBytes() == 400);
    const auto expanded = segfss::model::broadcastBias(material.gate0, 128);
    assert(expanded.size() == 12800);
    for (std::size_t b = 0; b < 128; ++b) {
        for (std::size_t i = 0; i < 100; ++i) {
            assert(expanded[b * 100 + i] == material.gate0[i]);
        }
    }
    std::cout << "model bias checks: PASS\n";
    return 0;
}
