#include "cases.h"

#include "relu.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_relu_batch(ResultLog& log, int bitwidth,
                      const std::vector<int64_t>& signed_values) {
    const std::size_t size = signed_values.size();
    std::vector<GroupElement> x(size);
    std::vector<u8> drelu(size);
    std::vector<uint64_t> expected(size);
    for (std::size_t i = 0; i < size; ++i) {
        const uint64_t raw = bitwidth == 64
            ? static_cast<uint64_t>(signed_values[i])
            : twos_from_signed(signed_values[i], bitwidth);
        const uint64_t server_x =
            reduce_to_bits(0x5A5A5A5AULL + 17 * i, bitwidth);
        x[i] = split_share(raw, bitwidth, server_x);

        const u8 public_drelu =
            static_cast<u8>(signed_values[i] >= 0 ? 1 : 0);
        const u8 server_drelu = static_cast<u8>((i + 1) & 1U);
        drelu[i] = party == SERVER
            ? server_drelu
            : static_cast<u8>(public_drelu ^ server_drelu);
        expected[i] = signed_values[i] >= 0 ? raw : 0;
    }

    // All multiplication and Boolean-mask conversion work belongs offline.
    std::vector<segfss::ReluPreprocessing> preprocessing =
        segfss::reluOffline(party, static_cast<int>(size), bitwidth, peer);

    const int32_t rounds_before = numRounds;
    const uint64_t io_events_before = peer->rounds;
    const uint64_t bytes_before = peer->bytesSent + peer->bytesReceived;
    std::vector<GroupElement> output =
        segfss::reluOnline(party, x, drelu, preprocessing, peer);
    const int32_t online_rounds = numRounds - rounds_before;
    const uint64_t online_io_events = peer->rounds - io_events_before;
    const uint64_t online_bytes =
        peer->bytesSent + peer->bytesReceived - bytes_before;

    const int ring_bytes = bitwidth <= 8 ? 1 : bitwidth <= 16 ? 2
                                                   : bitwidth <= 32 ? 4 : 8;
    const uint64_t expected_bytes =
        2 * (size * ring_bytes + (size + 7) / 8);
    log.check_local("BitXA online logical rounds bw=" +
                        std::to_string(bitwidth),
                    static_cast<uint64_t>(online_rounds), 1);
    log.check_local("BitXA online peer I/O events bw=" +
                        std::to_string(bitwidth),
                    online_io_events, 2);
    log.check_local("BitXA online communication bw=" +
                        std::to_string(bitwidth),
                    online_bytes, expected_bytes);
    log.check_vector("BitXA ReLU bw=" + std::to_string(bitwidth),
                     output.data(), expected, bitwidth);

    bool rejected_reuse = false;
    try {
        (void)segfss::reluOnline(party, x, drelu, preprocessing, peer);
    } catch (const std::logic_error&) {
        rejected_reuse = true;
    }
    log.check_local("BitXA rejects mask reuse bw=" +
                        std::to_string(bitwidth),
                    rejected_reuse ? 1 : 0, 1);
}

}  // namespace

void check_relu(ResultLog& log) {
    check_relu_batch(log, 8, {-128, -17, -1, 0, 1, 42, 126, 127});
    check_relu_batch(log, 16,
                     {-32768, -1024, -1, 0, 1, 255, 4096, 32767});
    check_relu_batch(log, 32,
                     {-2147483647LL - 1, -65537, -1, 0,
                      1, 65536, 123456789, 2147483647, -42});
    check_relu_batch(log, 64,
                     {std::numeric_limits<int64_t>::min(), -1, 0, 1,
                      std::numeric_limits<int64_t>::max()});
}
