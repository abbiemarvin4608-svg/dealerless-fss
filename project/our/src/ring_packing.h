#pragma once

#include "sci_numeric_oracle.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace segfss {

// Packed openings carry their public shape in a fixed header.  The header is
// prepended to the payload and sent in the same transport write; receivers
// read and validate it before consuming the payload.  This prevents a
// malformed shape from silently consuming bytes from the following message.
constexpr std::size_t kRingPackingHeaderBytes = 16;
using RingPackingHeader = std::array<uint8_t, kRingPackingHeaderBytes>;

inline RingPackingHeader makeRingPackingHeader(std::size_t count, int width) {
    if (width < 1 || width > 64)
        throw std::invalid_argument("ring packing: invalid header shape");
    RingPackingHeader header{};
    header[0] = 'R'; header[1] = 'P'; header[2] = 1; header[3] = uint8_t(width);
    uint64_t value = static_cast<uint64_t>(count);
    for (int byte = 0; byte < 8; ++byte) {
        header[4 + byte] = static_cast<uint8_t>(value);
        value >>= 8;
    }
    return header;
}

inline std::pair<std::size_t, int> parseRingPackingHeader(const uint8_t* bytes,
                                                           std::size_t size) {
    if (!bytes || size != kRingPackingHeaderBytes || bytes[0] != 'R' ||
        bytes[1] != 'P' || bytes[2] != 1 || bytes[3] < 1 || bytes[3] > 64)
        throw std::invalid_argument("ring packing: invalid header");
    uint64_t count = 0;
    for (int byte = 0; byte < 8; ++byte)
        count |= uint64_t(bytes[4 + byte]) << (8 * byte);
    for (std::size_t byte = 12; byte < kRingPackingHeaderBytes; ++byte)
        if (bytes[byte]) throw std::invalid_argument("ring packing: nonzero reserved header");
    if (count > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("ring packing: header count overflow");
    return {static_cast<std::size_t>(count), int(bytes[3])};
}

// LSB-first, with zero tail padding; count and width belong to the public shape.
inline std::vector<uint8_t> packRingValues(const std::vector<uint64_t>& values, int width) {
    if (width < 1 || width > 64) throw std::invalid_argument("ring packing: width must be in [1,64]");
    if (values.size() > (std::numeric_limits<std::size_t>::max() - 7) / static_cast<std::size_t>(width))
        throw std::invalid_argument("ring packing: payload overflow");
    const std::size_t bits = values.size() * static_cast<std::size_t>(width);
    std::vector<uint8_t> payload((bits + 7) / 8, 0);
    const uint64_t mask = reference::mask(width);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const uint64_t value = values[index] & mask;
        for (int bit = 0; bit < width;) {
            const std::size_t position = index * static_cast<std::size_t>(width) + bit;
            const int offset = int(position % 8);
            const int count = std::min(width - bit, 8 - offset);
            payload[position / 8] |= static_cast<uint8_t>(((value >> bit) & ((1U << count) - 1)) << offset);
            bit += count;
        }
    }
    return payload;
}

inline std::vector<uint8_t> packRingMessage(const std::vector<uint64_t>& values,
                                            int width) {
    const auto payload = packRingValues(values, width);
    if (payload.size() > std::numeric_limits<std::size_t>::max() - kRingPackingHeaderBytes)
        throw std::invalid_argument("ring packing: message size overflow");
    std::vector<uint8_t> message(kRingPackingHeaderBytes + payload.size());
    const auto header = makeRingPackingHeader(values.size(), width);
    std::copy(header.begin(), header.end(), message.begin());
    std::copy(payload.begin(), payload.end(), message.begin() + kRingPackingHeaderBytes);
    return message;
}

inline std::vector<uint64_t> unpackRingValues(const std::vector<uint8_t>& payload,
                                       std::size_t count, int width) {
    if (width < 1 || width > 64) throw std::invalid_argument("ring packing: width must be in [1,64]");
    if (count > (std::numeric_limits<std::size_t>::max() - 7) / static_cast<std::size_t>(width))
        throw std::invalid_argument("ring packing: payload overflow");
    const std::size_t bits = count * static_cast<std::size_t>(width);
    if (payload.size() != (bits + 7) / 8) throw std::invalid_argument("ring packing: shape mismatch");
    std::vector<uint64_t> values(count, 0);
    for (std::size_t index = 0; index < count; ++index)
        for (int bit = 0; bit < width;) {
            const std::size_t position = index * static_cast<std::size_t>(width) + bit;
            const int offset = int(position % 8);
            const int count = std::min(width - bit, 8 - offset);
            values[index] |= static_cast<uint64_t>((payload[position / 8] >> offset) & ((1U << count) - 1)) << bit;
            bit += count;
        }
    if (!payload.empty() && (bits & 7) && (payload.back() >> (bits & 7)))
        throw std::invalid_argument("ring packing: nonzero tail padding");
    return values;
}

template<class Receive>
inline std::vector<uint64_t> receiveRingMessage(std::size_t count, int width,
                                                Receive&& receive) {
    const auto expected = makeRingPackingHeader(count, width);
    RingPackingHeader header{};
    receive(header.data(), static_cast<int>(header.size()));
    const auto shape = parseRingPackingHeader(header.data(), header.size());
    if (header != expected || shape.first != count || shape.second != width)
        throw std::invalid_argument("ring packing: remote public shape mismatch");
    if (count > (std::numeric_limits<std::size_t>::max() - 7) / static_cast<std::size_t>(width))
        throw std::invalid_argument("ring packing: payload overflow");
    const auto payload_size = (count * static_cast<std::size_t>(width) + 7) / 8;
    if (payload_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) - kRingPackingHeaderBytes)
        throw std::invalid_argument("ring packing: message exceeds transport size");
    std::vector<uint8_t> payload(payload_size);
    receive(payload.data(), static_cast<int>(payload.size()));
    return unpackRingValues(payload, count, width);
}

template<class Send, class Receive>
inline std::vector<uint64_t> exchangeRingValues(bool send_first,
    const std::vector<uint64_t>& local, int width, Send&& send, Receive&& receive) {
    const auto message = packRingMessage(local, width);
    if (message.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("ring packing: message exceeds transport size");
    std::vector<uint64_t> remote;
    if (send_first) {
        send(message.data(), static_cast<int>(message.size()));
        remote = receiveRingMessage(local.size(), width, receive);
    } else {
        remote = receiveRingMessage(local.size(), width, receive);
        send(message.data(), static_cast<int>(message.size()));
    }
    return remote;
}

}  // namespace segfss
