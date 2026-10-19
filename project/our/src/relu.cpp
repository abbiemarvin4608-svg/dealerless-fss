#include "relu.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <sys/socket.h>

#include "mpc/api.h"
#include "mpc/secure_ops.h"

namespace segfss {
namespace {

bool sendAll(int socket, const u8* data, std::size_t size) {
    std::size_t sent = 0;
    while (sent < size) {
        const ssize_t result =
            ::send(socket, data + sent, size - sent, 0);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(result);
    }
    return true;
}

bool recvAll(int socket, u8* data, std::size_t size) {
    std::size_t received = 0;
    while (received < size) {
        const ssize_t result =
            ::recv(socket, data + received, size - received, 0);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            return false;
        }
        received += static_cast<std::size_t>(result);
    }
    return true;
}

int ringByteWidth(int bitwidth) {
    if (bitwidth <= 8) return 1;
    if (bitwidth <= 16) return 2;
    if (bitwidth <= 32) return 4;
    return 8;
}

void setPackedBit(std::vector<u8>& buffer, std::size_t offset,
                  std::size_t bit_index, u8 bit) {
    if ((bit & 1U) != 0) {
        buffer[offset + bit_index / 8] |=
            static_cast<u8>(1U << (bit_index % 8));
    }
}

u8 getPackedBit(const std::vector<u8>& buffer, std::size_t offset,
                std::size_t bit_index) {
    return static_cast<u8>(
        (buffer[offset + bit_index / 8] >> (bit_index % 8)) & 1U);
}

// Open an additive ring value and a Boolean bit per item in one network
// exchange.  Ring shares are added and Boolean shares are XORed after receipt.
void openMaskedBitXA(const std::vector<GroupElement>& masked_x,
                     const std::vector<u8>& masked_y,
                     std::vector<GroupElement>& opened_x,
                     std::vector<u8>& opened_y, Peer* player) {
    if (masked_x.empty() || masked_x.size() != masked_y.size()) {
        throw std::invalid_argument(
            "openMaskedBitXA requires equal, non-empty inputs");
    }

    const std::size_t size = masked_x.size();
    const int bitwidth = masked_x[0].bitsize;
    const std::size_t ring_bytes = ringByteWidth(bitwidth);
    const std::size_t bit_bytes = (size + 7) / 8;
    const std::size_t bit_offset = size * ring_bytes;
    const std::size_t message_bytes = bit_offset + bit_bytes;
    std::vector<u8> send_buffer(message_bytes, 0);
    std::vector<u8> recv_buffer(message_bytes, 0);

    for (std::size_t i = 0; i < size; ++i) {
        if (masked_x[i].bitsize != bitwidth) {
            throw std::invalid_argument(
                "openMaskedBitXA requires a uniform ring bitwidth");
        }
        std::memcpy(send_buffer.data() + i * ring_bytes,
                    &masked_x[i].value, ring_bytes);
        setPackedBit(send_buffer, bit_offset, i, masked_y[i]);
    }

    const auto start = std::chrono::steady_clock::now();
    bool send_ok = true;
    bool recv_ok = true;
    if (player->useFile) {
        player->file.write(
            reinterpret_cast<const char*>(send_buffer.data()),
            static_cast<std::streamsize>(message_bytes));
        player->file.read(reinterpret_cast<char*>(recv_buffer.data()),
                          static_cast<std::streamsize>(message_bytes));
        send_ok = static_cast<bool>(player->file);
        recv_ok = send_ok;
    } else {
        const auto send_values = [&] {
            player->trace_raw_send(message_bytes);
            send_ok = sendAll(player->sendsocket, send_buffer.data(),
                              message_bytes);
        };
        const auto receive_values = [&] {
            recv_ok = recvAll(player->recvsocket, recv_buffer.data(),
                              message_bytes);
            if (recv_ok) player->trace_raw_received(message_bytes);
        };
        const bool owned = dfss_round::callbacks.caller_limit && dfss_round::callbacks.caller_limit() == 1;
        if (dfss_round::callbacks.transport_participants) dfss_round::callbacks.transport_participants(owned ? 1 : 2);
        if (owned) {
            if (player->role() == SERVER) { send_values(); receive_values(); }
            else { receive_values(); send_values(); }
        } else {
            const auto fork = dfss_round::fork();
            uint64_t send_tail = 0, receive_tail = 0;
            std::thread send_thread([&] {
                dfss_round::ActorScope actor(fork);
                send_values(); send_tail = actor.finish();
            });
            {
                dfss_round::ActorScope actor(fork);
                receive_values(); receive_tail = actor.finish();
            }
            send_thread.join();
            dfss_round::join(send_tail);
            dfss_round::join(receive_tail);
        }
    }
    if (!send_ok || !recv_ok) {
        throw std::runtime_error("BitXA combined opening failed");
    }

    opened_x.resize(size);
    opened_y.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
        uint64_t remote_x = 0;
        std::memcpy(&remote_x,
                    recv_buffer.data() + i * ring_bytes, ring_bytes);
        opened_x[i] = masked_x[i] + GroupElement(remote_x, bitwidth);
        opened_y[i] = static_cast<u8>(
            (masked_y[i] ^ getPackedBit(recv_buffer, bit_offset, i)) & 1U);
    }

    player->bytesSent += message_bytes;
    player->bytesReceived += message_bytes;
    player->rounds += 2;  // one send and one receive in the same round
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    ++player->additive_openings;
    player->opening_microseconds += elapsed;
    if (player == peer) {
        ++numRounds;
        reconstructMicroseconds += elapsed;
    }
}

void validateParty(int party_id) {
    if (party_id != SERVER && party_id != CLIENT) {
        throw std::invalid_argument(
            "BitXA party_id must be SERVER or CLIENT");
    }
}

}  // namespace

std::vector<BitXAPreprocessing> bitxaOffline(
    int party_id, int size, int bitwidth, Peer* player) {
    validateParty(party_id);
    if (player == nullptr) {
        throw std::invalid_argument("bitxaOffline requires a peer");
    }
    if (size <= 0) {
        throw std::invalid_argument("bitxaOffline size must be positive");
    }
    if (bitwidth < 1 || bitwidth > 64) {
        throw std::invalid_argument(
            "bitxaOffline bitwidth must be in [1,64]");
    }

    auto& rng = player->randomSource();
    std::vector<BitXAPreprocessing> preprocessing(size);
    std::vector<GroupElement> e(size, GroupElement(0, bitwidth));
    std::vector<GroupElement> f(size, GroupElement(0, bitwidth));
    for (int i = 0; i < size; ++i) {
        BitXAPreprocessing& prep = preprocessing[i];
        prep.bitwidth = bitwidth;
        prep.delta_y_bool = static_cast<u8>(rng.get<u8>() & 1U);
        prep.delta_x = random_ge_from_prng(rng, bitwidth);
        prep.delta_y_arith = GroupElement(0, bitwidth);
        prep.delta_z = GroupElement(0, bitwidth);

        // Algorithm 1, Offline 1(1): e is held by P0 and f by P1.
        if (party_id == SERVER) {
            e[i] = GroupElement(prep.delta_y_bool, bitwidth);
        } else {
            f[i] = GroupElement(prep.delta_y_bool, bitwidth);
        }
    }

    // Offline 1(2): ef = e*f, batched across all mixed products.
    std::vector<GroupElement> triple_a(size, GroupElement(0, bitwidth));
    std::vector<GroupElement> triple_b(size, GroupElement(0, bitwidth));
    std::vector<GroupElement> triple_c(size, GroupElement(0, bitwidth));
    std::vector<GroupElement> ef(size, GroupElement(0, bitwidth));
    beaver_mult_offline(party_id, triple_a.data(), triple_b.data(),
                        triple_c.data(), player, size);
    std::vector<GroupElement> e_for_mul = e;
    std::vector<GroupElement> f_for_mul = f;
    beaver_mult_online(party_id, e_for_mul.data(), f_for_mul.data(),
                       triple_a.data(), triple_b.data(), triple_c.data(),
                       ef.data(), size, player);

    // Offline 1(3): arithmetic sharing of delta_y = delta_y0 XOR delta_y1.
    for (int i = 0; i < size; ++i) {
        preprocessing[i].delta_y_arith = e[i] + f[i] - ef[i] * 2;
    }

    // Offline 2: delta_z = delta_x * delta_y, again fully batched.
    std::fill(triple_a.begin(), triple_a.end(), GroupElement(0, bitwidth));
    std::fill(triple_b.begin(), triple_b.end(), GroupElement(0, bitwidth));
    std::fill(triple_c.begin(), triple_c.end(), GroupElement(0, bitwidth));
    beaver_mult_offline(party_id, triple_a.data(), triple_b.data(),
                        triple_c.data(), player, size);

    std::vector<GroupElement> delta_x(size);
    std::vector<GroupElement> delta_y(size);
    std::vector<GroupElement> delta_z(size, GroupElement(0, bitwidth));
    for (int i = 0; i < size; ++i) {
        delta_x[i] = preprocessing[i].delta_x;
        delta_y[i] = preprocessing[i].delta_y_arith;
    }
    beaver_mult_online(party_id, delta_x.data(), delta_y.data(),
                       triple_a.data(), triple_b.data(), triple_c.data(),
                       delta_z.data(), size, player);
    for (int i = 0; i < size; ++i) {
        preprocessing[i].delta_z = delta_z[i];
    }
    return preprocessing;
}

BitXAPreprocessing bitxaOffline(int party_id, int bitwidth, Peer* player) {
    return bitxaOffline(party_id, 1, bitwidth, player).front();
}

std::vector<GroupElement> bitxaOnline(
    int party_id, const std::vector<GroupElement>& x,
    const std::vector<u8>& y, std::vector<BitXAPreprocessing>& preprocessing,
    Peer* player) {
    validateParty(party_id);
    if (player == nullptr) {
        throw std::invalid_argument("bitxaOnline requires a peer");
    }
    if (x.empty() || x.size() != y.size() ||
        x.size() != preprocessing.size()) {
        throw std::invalid_argument(
            "bitxaOnline requires equal, non-empty batches");
    }

    const int bitwidth = x[0].bitsize;
    std::vector<GroupElement> masked_x(x.size());
    std::vector<u8> masked_y(y.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        BitXAPreprocessing& prep = preprocessing[i];
        if (prep.consumed) {
            throw std::logic_error(
                "bitxaOnline preprocessing cannot be reused");
        }
        if (x[i].bitsize != bitwidth || prep.bitwidth != bitwidth ||
            prep.delta_x.bitsize != bitwidth ||
            prep.delta_y_arith.bitsize != bitwidth ||
            prep.delta_z.bitsize != bitwidth) {
            throw std::invalid_argument("bitxaOnline bitwidth mismatch");
        }
        masked_x[i] = x[i] + prep.delta_x;
        masked_y[i] = static_cast<u8>(
            (y[i] ^ prep.delta_y_bool) & 1U);
    }

    // Online 1-2: one combined opening round for every gate in the batch.
    std::vector<GroupElement> delta_x;
    std::vector<u8> delta_y;
    openMaskedBitXA(masked_x, masked_y, delta_x, delta_y, player);

    // Online 3: expand (Delta_x-delta_x) *
    // (Delta_y + delta_y - 2*Delta_y*delta_y) share-wise.
    std::vector<GroupElement> output(x.size(), GroupElement(0, bitwidth));
    const uint64_t party_bit = static_cast<uint64_t>(party_id - SERVER);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const BitXAPreprocessing& prep = preprocessing[i];
        const uint64_t public_y = static_cast<uint64_t>(delta_y[i] & 1U);

        GroupElement share = prep.delta_y_arith * delta_x[i] - prep.delta_z;
        if (public_y != 0) {
            share = share + delta_x[i] * party_bit;
            share = share - (prep.delta_y_arith * delta_x[i]) * 2;
            share = share - prep.delta_x;
            share = share + prep.delta_z * 2;
        }
        output[i] = share;
        preprocessing[i].consumed = true;
    }
    return output;
}

GroupElement bitxaOnline(int party_id, GroupElement x, u8 y,
                         BitXAPreprocessing& preprocessing, Peer* player) {
    std::vector<GroupElement> x_batch{x};
    std::vector<u8> y_batch{y};
    std::vector<BitXAPreprocessing> prep_batch{preprocessing};
    GroupElement output =
        bitxaOnline(party_id, x_batch, y_batch, prep_batch, player).front();
    preprocessing = prep_batch.front();
    return output;
}

std::vector<ReluPreprocessing> reluOffline(
    int party_id, int size, int bitwidth, Peer* player) {
    return bitxaOffline(party_id, size, bitwidth, player);
}

ReluPreprocessing reluOffline(int party_id, int bitwidth, Peer* player) {
    return bitxaOffline(party_id, bitwidth, player);
}

std::vector<GroupElement> reluOnline(
    int party_id, const std::vector<GroupElement>& x,
    const std::vector<u8>& drelu,
    std::vector<ReluPreprocessing>& preprocessing, Peer* player) {
    return bitxaOnline(party_id, x, drelu, preprocessing, player);
}

GroupElement reluOnline(int party_id, GroupElement x, u8 drelu,
                        ReluPreprocessing& preprocessing, Peer* player) {
    return bitxaOnline(party_id, x, drelu, preprocessing, player);
}

}  // namespace segfss
