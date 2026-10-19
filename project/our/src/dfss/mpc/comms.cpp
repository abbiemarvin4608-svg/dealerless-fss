/*
Original Authors: Deepak Kumaraswamy, Kanav Gupta
Modified by: Pengzhi Xing
Copyright:
Original Copyright (c) 2022 Microsoft Research
Copyright (c) 2024 Pengzhi Xing
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "mpc/comms.h"
#include "mpc/api.h"

#include <array>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <chrono>
#include <thread>

dfss_round::Callbacks dfss_round::callbacks;

namespace {
constexpr int kDefaultOtPrecomputeBatchSize = 128;

uint64_t bit_mask(int bitlen) {
    return bitlen >= 64 ? ~uint64_t(0) : ((uint64_t(1) << bitlen) - 1);
}

bool native_ring_cot() {
    const char* value = std::getenv("SEGFSS_NATIVE_RING_COT");
    if (!value || std::strcmp(value, "") == 0 || std::strcmp(value, "0") == 0)
        return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("SEGFSS_NATIVE_RING_COT must be 0 or 1");
}

void split_block(const osuCrypto::block& input, uint64_t* lo,
                 uint64_t* hi) {
    uint64_t limbs[2];
    std::memcpy(limbs, &input, sizeof(limbs));
    *lo = limbs[0];
    *hi = limbs[1];
}

osuCrypto::block make_block_from_limbs(uint64_t lo, uint64_t hi) {
    return osuCrypto::toBlock(hi, lo);
}

template <typename T>
T loadFromBytes(const char* buf) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    std::memcpy(&value, buf, sizeof(T));
    return value;
}
}  // namespace

void Peer::open_additive(GroupElement* shares, int size, int bits) {
    if (size < 0 || bits < 1 || bits > 64 || (size && !shares) ||
        (role_ != SERVER && role_ != CLIENT))
        throw std::invalid_argument("additive opening: invalid role or shape");
    if (!size) return;
    for (int i = 0; i < size; ++i)
        if (shares[i].bitsize != bits)
            throw std::invalid_argument("additive opening: heterogeneous ring");
    const auto started = std::chrono::steady_clock::now();
    std::vector<uint64_t> remote(size);
    const bool owned = dfss_round::callbacks.caller_limit && dfss_round::callbacks.caller_limit() == 1;
    if (dfss_round::callbacks.transport_participants) dfss_round::callbacks.transport_participants(owned ? 1 : 2);
    if (owned) {
        if (role_ == SERVER) {
            send_batched_input(shares, size, bits);
            recv_batched_input(remote.data(), size, bits);
        } else {
            recv_batched_input(remote.data(), size, bits);
            send_batched_input(shares, size, bits);
        }
    } else {
        const auto parent = dfss_round::fork();
        uint64_t send_tail = 0, receive_tail = 0;
        std::thread sender([&] {
            dfss_round::ActorScope actor(parent);
            send_batched_input(shares, size, bits);
            send_tail = actor.finish();
        });
        {
            dfss_round::ActorScope actor(parent);
            recv_batched_input(remote.data(), size, bits);
            receive_tail = actor.finish();
        }
        sender.join();
        dfss_round::join(send_tail);
        dfss_round::join(receive_tail);
    }
    for (int i = 0; i < size; ++i) shares[i] = shares[i] + remote[i];
    ++additive_openings;
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();
    opening_microseconds += elapsed;
    // Compatibility counters belong exclusively to the default serial peer.
    if (this == peer) {
        ++numRounds;
        reconstructMicroseconds += elapsed;
    }
}


Peer::Peer(std::string ip, int port) : Peer(ip, port, party) {}
Peer::Peer(std::string ip, int port, int role) : role_(role) {
    std::cerr << "trying to connect with server...";
    {
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr(ip.c_str());
        while (1) {
            recvsocket = socket(AF_INET, SOCK_STREAM, 0);

            if (connect(recvsocket, (struct sockaddr *)&addr,
                        sizeof(struct sockaddr)) == 0) {
                break;
            }

            ::close(recvsocket);
            usleep(1000);
        }
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    sleep(1);
    {
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port+3);
        addr.sin_addr.s_addr = inet_addr(ip.c_str());
        while(1){
            sendsocket = socket(AF_INET, SOCK_STREAM, 0);
            if (sendsocket < 0) {
                perror("socket");
                exit(1);
            }
            if (connect(sendsocket, (struct sockaddr *) &addr, sizeof(addr)) == 0) {
                break;
            }
            ::close(sendsocket);
            usleep(1000);
        }
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    iopack_ = std::make_unique<sci::IOPack>(role - 1, port);
    otpack = std::make_unique<sci::OTPack>(iopack_.get(), role - 1);
    block_ot = std::make_unique<sci::IKNP<sci::NetIO>>(iopack_->io);
    block_ot_reversed = std::make_unique<sci::IKNP<sci::NetIO>>(iopack_->io_rev);
    if (!MillInstance){
        // Need to reconfigure bitlen when calling
        MillInstance = std::make_unique<MillionaireProtocol>(role - 1, iopack_.get(), otpack.get());
    }
    configure_ot_precompute(kDefaultOtPrecomputeBatchSize);
    std::cerr << "connected" << "\n";
}

void Peer::close() {
    if (useFile) {
        file.close();
    }
    else {
        ::close(sendsocket);
        ::close(recvsocket);
    }
}

Peer* waitForPeer(int port) {
    return waitForPeer(port, party);
}
Peer* waitForPeer(int port, int role) {
    int sendsocket, recvsocket;
    std::cerr << "waiting for connection from client...";
    
    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY);       /* set our address to any interface */
        serv.sin_port = htons(port); /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,
                    sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) < 0) {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0) {
            perror("error: listen");
            exit(1);
        }
        sendsocket = accept(mysocket, (struct sockaddr *)&dest, &socksize);
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }

    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY);       /* set our address to any interface */
        serv.sin_port = htons(port+3); /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,
                    sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) < 0) {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0) {
            perror("error: listen");
            exit(1);
        }
        recvsocket = accept(mysocket, (struct sockaddr *)&dest, &socksize);
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }
    
    std::cerr << "connected" << std::endl;
    return new Peer(sendsocket, recvsocket, role, port);
}

void Peer::open_xor(u8* shares, int size) {
    std::vector<u8> remote(size);
    if (role_ == SERVER) { send_u8(shares, size); recv_u8(remote.data(), size); }
    else { recv_u8(remote.data(), size); send_u8(shares, size); }
    for (int i = 0; i < size; ++i) shares[i] ^= remote[i];
}
void Peer::open_xor(osuCrypto::block* shares, int size) {
    std::vector<osuCrypto::block> remote(size);
    if (role_ == SERVER) { send_block(shares, size); recv_block(remote.data(), size); }
    else { recv_block(remote.data(), size); send_block(shares, size); }
    for (int i = 0; i < size; ++i) shares[i] ^= remote[i];
}

void Peer::open_xor_mixed(osuCrypto::block* blocks, int block_count,
                          u8* bytes, int byte_count) {
    if (block_count < 0 || byte_count < 0 ||
        block_count > (INT32_MAX - byte_count) / int(sizeof(osuCrypto::block)))
        throw std::invalid_argument("mixed XOR opening payload exceeds int length");
    const int block_bytes = block_count * int(sizeof(osuCrypto::block));
    std::vector<u8> packed(block_bytes + byte_count);
    if (block_bytes) std::memcpy(packed.data(), blocks, block_bytes);
    if (byte_count) std::memcpy(packed.data() + block_bytes, bytes, byte_count);
    open_xor(packed.data(), static_cast<int>(packed.size()));
    if (block_bytes) std::memcpy(blocks, packed.data(), block_bytes);
    if (byte_count) std::memcpy(bytes, packed.data() + block_bytes, byte_count);
}


void Peer::send_ge(const GroupElement &g, int bw) {
    trace_raw_send(bw > 32 ? 8 : bw > 16 ? 4 : bw > 8 ? 2 : 1);
    if (bw > 32) {
        char *buf = (char *)(&g.value);
        if (useFile) {
            this->file.write(buf, 8);
        } else {
            send(sendsocket, buf, 8, 0);
        }
        bytesSent += 8;
    }
    else if (bw > 16) {
        char *buf = (char *)(&g.value);
        if (useFile) {
            this->file.write(buf, 4);
        } else {
            send(sendsocket, buf, 4, 0);
        }
        bytesSent += 4;
    }
    else if (bw > 8) {
        char *buf = (char *)(&g.value);
        if (useFile) {
            this->file.write(buf, 2);
        } else {
            send(sendsocket, buf, 2, 0);
        }
        bytesSent += 2;
    }
    else {
        char *buf = (char *)(&g.value);
        if (useFile) {
            this->file.write(buf, 1);
        } else {
            send(sendsocket, buf, 1, 0);
        }
        bytesSent += 1;
    }
    rounds++;
}

void Peer::send_block(const osuCrypto::block &b) {
    trace_raw_send(sizeof(osuCrypto::block));
    char *buf = (char *)(&b);
    if (useFile) {
        this->file.write(buf, sizeof(osuCrypto::block));
    } else {
        send(sendsocket, buf, sizeof(osuCrypto::block), 0);
    }
    bytesSent += sizeof(osuCrypto::block);
    rounds++;
}

void Peer::send_block(const osuCrypto::block* b, int size) {
    trace_raw_send(sizeof(osuCrypto::block) * size);
    char *buf = (char *)(b);
    if (useFile) {
        this->file.write(buf, sizeof(osuCrypto::block) * size);
    } else {
        send(sendsocket, buf, sizeof(osuCrypto::block) * size, 0);
    }
    bytesSent += sizeof(osuCrypto::block) * size;
    rounds++;
}

osuCrypto::block Peer::recv_block(){
    char buf[sizeof(osuCrypto::block)];
    if (useFile) {
        this->file.read(buf, sizeof(osuCrypto::block));
    } else {
        recv(recvsocket, buf, sizeof(osuCrypto::block), MSG_WAITALL);
    }
    osuCrypto::block b = loadFromBytes<osuCrypto::block>(buf);
    bytesReceived += sizeof(osuCrypto::block);
    trace_raw_received(sizeof(osuCrypto::block));
    rounds++;
    return b;
}

void Peer::recv_block(osuCrypto::block* output, int size) {
    if (useFile) {
        this->file.read((char *)output, sizeof(osuCrypto::block) * size);
    } else {
        recv(recvsocket, (char *)output, sizeof(osuCrypto::block) * size,
             MSG_WAITALL);
    }
    bytesReceived += sizeof(osuCrypto::block) * size;
    trace_raw_received(sizeof(osuCrypto::block) * size);
    rounds++;
}

void Peer::send_u8(const u8 &b){
    trace_raw_send(sizeof(u8));
    char *buf = (char*)(&b);
    if (useFile) {
        this->file.write(buf, sizeof(u8));
    }
    else{
        send(sendsocket, buf, sizeof(u8), 0);
    }
    bytesSent += sizeof(u8);
    rounds++;
}

void Peer::send_u8(const u8* b, int size){
    if (size < 0 || (size && !b)) throw std::invalid_argument("invalid byte send");
    trace_raw_send(size);
    if (useFile) {
        file.write(reinterpret_cast<const char*>(b), size);
        if (!file) throw std::runtime_error("byte file write failed");
        bytesSent += size;
    } else {
        int done = 0;
        while (done < size) {
            const auto sent = send(sendsocket, b + done, size - done, MSG_NOSIGNAL);
            if (sent < 0 && errno == EINTR) continue;
            if (sent <= 0) throw std::runtime_error("byte socket send failed");
            done += sent; bytesSent += sent;
        }
    }
    ++rounds;
}

void Peer::recv_u8(u8* output, int size){
    if (size < 0 || (size && !output)) throw std::invalid_argument("invalid byte receive");
    if (useFile) {
        file.read(reinterpret_cast<char*>(output), size);
        if (!file) throw std::runtime_error("byte file read failed");
        bytesReceived += size;
    } else {
        int done = 0;
        while (done < size) {
            const auto received = recv(recvsocket, output + done, size - done, 0);
            if (received < 0 && errno == EINTR) continue;
            if (received <= 0) throw std::runtime_error("byte socket receive failed");
            done += received; bytesReceived += received;
        }
    }
    trace_raw_received(size);
    ++rounds;
}

u8 Peer::recv_u8(){
    char buf[sizeof(u8)];
    if (useFile) {
        this->file.read(buf, sizeof(u8));
    } else {
        recv(recvsocket, buf, sizeof(u8), MSG_WAITALL);
    }
    u8 b = loadFromBytes<u8>(buf);
    bytesReceived += sizeof(u8);
    trace_raw_received(sizeof(u8));
    rounds++;
    return b;
}

void Peer::send_u64(const uint64_t* input, int size) {
    std::vector<GroupElement> ge_list(size);

    for (int i = 0; i < size; i++){
        ge_list[i].value = input[i];
        ge_list[i].bitsize = 64;
    }
    send_batched_input(ge_list.data(), size, 64);
}

void Peer::recv_u64(uint64_t* output, int size){
    std::vector<uint64_t> ge_list(size);
    recv_batched_input(ge_list.data(), size, 64);
    for (int i = 0; i < size; i++){
        output[i] = ge_list[i];
    }
}

void Peer::send_u64(const uint64_t &b){
    trace_raw_send(sizeof(uint64_t));
    char *buf = (char*)(&b);
    if (useFile) {
        this->file.write(buf, sizeof(uint64_t));
    }
    else{
        send(sendsocket, buf, sizeof(uint64_t), 0);
    }
    bytesSent += sizeof(uint64_t);
    rounds++;
}

uint64_t Peer::recv_u64(){
    char buf[sizeof(uint64_t)];
    if (useFile) {
        this->file.read(buf, sizeof(uint64_t));
    } else {
        recv(recvsocket, buf, sizeof(uint64_t), MSG_WAITALL);
    }
    uint64_t b = loadFromBytes<uint64_t>(buf);
    bytesReceived += sizeof(uint64_t);
    trace_raw_received(sizeof(uint64_t));
    rounds++;
    return b;
}

void Peer::send_ge(const GroupElement &g){
    send_u64(g.value);
    send_u64(g.bitsize);
}

void Peer::send_mask(const GroupElement &g) {
    send_ge(g, 64);
}

GroupElement Peer::recv_ge(){
    GroupElement tmp;
    uint64_t value = recv_u64();
    uint64_t bitsize = recv_u64();
    tmp.value = value;
    tmp.bitsize = bitsize;
    return tmp;
}

void Peer::send_input(const GroupElement &g) {
    send_ge(g, 64);
}

void Peer::send_batched_input(const GroupElement *g, int size, int bw)
{
    trace_raw_send(uint64_t(size) * (bw > 32 ? 8 : bw > 16 ? 4 : bw > 8 ? 2 : 1));
    if (bw > 32) {
        std::vector<uint64_t> temp(size);
        for (int i = 0; i < size; i++) {
            temp[i] = g[i].value;
        }
        char *buf = (char *)(temp.data());
        if (useFile) {
            this->file.write(buf, 8*size);
        } else {
            send(sendsocket, buf, 8*size, 0);
        }
        bytesSent += 8*size;
    }
    else if (bw > 16) {
        std::vector<uint32_t> temp(size);
        for (int i = 0; i < size; i++) {
            temp[i] = (uint32_t)g[i].value;
        }
        char *buf = (char *)(temp.data());
        if (useFile) {
            this->file.write(buf, 4*size);
        } else {
            send(sendsocket, buf, 4*size, 0);
        }
        bytesSent += 4*size;
    }
    else if (bw > 8) {
        std::vector<uint16_t> temp(size);
        for (int i = 0; i < size; i++) {
            temp[i] = (uint16_t)g[i].value;
        }
        char *buf = (char *)(temp.data());
        if (useFile) {
            this->file.write(buf, 2*size);
        } else {
            send(sendsocket, buf, 2*size, 0);
        }
        bytesSent += 2*size;
    }
    else {
        std::vector<uint8_t> temp(size);
        for (int i = 0; i < size; i++) {
            temp[i] = (uint8_t)g[i].value;
        }
        char *buf = (char *)(temp.data());
        if (useFile) {
            this->file.write(buf, size);
        } else {
            send(sendsocket, buf, size, 0);
        }
        bytesSent += size;
    }
    rounds++;
}

void Peer::recv_batched_input(uint64_t *g, int size, int bw)
{
    if (bw > 32) {
        if (useFile) {
            this->file.read((char *)g, 8*size);
        } else {
            recv(recvsocket, (char *)g, 8*size, MSG_WAITALL);
        }
        bytesReceived += 8*size;
    }
    else if (bw > 16) {
        std::vector<uint32_t> tmp(size);
        if (useFile) {
            this->file.read((char *)tmp.data(), 4*size);
        } else {
            recv(recvsocket, (char *)tmp.data(), 4*size, MSG_WAITALL);
        }
        for (int i = 0; i < size; i++) {
            g[i] = tmp[i];
        }
        bytesReceived += 4*size;
    }
    else if (bw > 8) {
        std::vector<uint16_t> tmp(size);
        if (useFile) {
            this->file.read((char *)tmp.data(), 2*size);
        } else {
            recv(recvsocket, (char *)tmp.data(), 2*size, MSG_WAITALL);
        }
        for (int i = 0; i < size; i++) {
            g[i] = tmp[i];
        }
        bytesReceived += 2*size;
    }
    else {
        std::vector<uint8_t> tmp(size);
        if (useFile) {
            this->file.read((char *)tmp.data(), size);
        } else {
            recv(recvsocket, (char *)tmp.data(), size, MSG_WAITALL);
        }
        for (int i = 0; i < size; i++) {
            g[i] = tmp[i];
        }
        bytesReceived += size;
    }
    trace_raw_received(uint64_t(size) * (bw > 32 ? 8 : bw > 16 ? 4 : bw > 8 ? 2 : 1));
    rounds++;
}

GroupElement Peer::recv_input() {
    char buf[8];
    if (useFile) {
        std::cerr << "Can't recv from peer in file mode\n";
        exit(1);
    } else {
        recv(recvsocket, buf, 8, MSG_WAITALL);
    }
    GroupElement g(loadFromBytes<uint64_t>(buf), bitlength);
    bytesReceived += 8;
    trace_raw_received(8);
    rounds++;
    return g;
}

void Peer::configure_ot_precompute(int batch_size) {
    ot_precompute_batch_size_ = batch_size;
    otpack->iknp_straight->set_precomp_batch_size(batch_size);
    otpack->iknp_reversed->set_precomp_batch_size(batch_size);
}

void Peer::reset_ot_precompute() {
    configure_ot_precompute(ot_precompute_batch_size_);
}

void Peer::send_ot_u64(bool using_aux_iknp, uint64_t** msgs, int length,
                       int bitlen) {
    if (length <= 0) {
        return;
    }
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    uint64_t pre_rounds = ot->io->num_rounds;
    ot->send(msgs, length, bitlen);
    rounds += (ot->io->num_rounds - pre_rounds);
    bytesSent += (ot->io->counter - pre_comm);
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::recv_ot_u64(bool using_aux_iknp, uint64_t* output,
                       uint8_t* choice, int length, int bitlen) {
    if (length <= 0) {
        return;
    }
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    uint64_t pre_rounds = ot->io->num_rounds;
    ot->recv(output, choice, length, bitlen);
    rounds += (ot->io->num_rounds - pre_rounds);
    bytesSent += (ot->io->counter - pre_comm);
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::send_batched_cot_u64(bool using_aux_iknp, uint64_t* data0,
                                uint64_t* corr, const std::vector<int>& msg_len,
                                int num_ot) {
    if (num_ot <= 0) {
        return;
    }
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    uint64_t pre_rounds = ot->io->num_rounds;
    ot->send_batched_cot(data0, corr, msg_len, num_ot);
    rounds += (ot->io->num_rounds - pre_rounds);
    bytesSent += (ot->io->counter - pre_comm);
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::recv_batched_cot_u64(bool using_aux_iknp, uint64_t* output,
                                bool* choice, const std::vector<int>& msg_len,
                                int num_ot) {
    if (num_ot <= 0) {
        return;
    }
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    uint64_t pre_rounds = ot->io->num_rounds;
    ot->recv_batched_cot(output, choice, msg_len, num_ot);
    rounds += (ot->io->num_rounds - pre_rounds);
    bytesSent += (ot->io->counter - pre_comm);
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::send_cot(uint64_t data, uint64_t* output, int length) {
    if (length <= 0) {
        return;
    }
    const uint64_t mask = bit_mask(64);
    auto rng = secure_prng();
    std::vector<std::array<uint64_t, 2>> msgs(length);
    std::vector<uint64_t*> msg_ptrs(length);
    for (int i = 0; i < length; i++) {
        output[i] = rng.get<uint64_t>() & mask;
        msgs[i][0] = output[i];
        msgs[i][1] = (output[i] + data) & mask;
        msg_ptrs[i] = msgs[i].data();
    }
    send_ot_u64(false, msg_ptrs.data(), length, 64);
}

void Peer::send_cot(const GroupElement* data, GroupElement* output, int length,
                    bool using_aux_iknp) {
    if (length <= 0) {
        return;
    }
    const int bitlen = data[0].bitsize;
    const uint64_t mask = bit_mask(bitlen);
    if (native_ring_cot()) {
        if (bitlen < 1 || bitlen > 64)
            throw std::invalid_argument("native ring COT width must be in [1,64]");
        std::vector<uint64_t> correlations(length), sender_base(length);
        for (int i = 0; i < length; ++i) {
            if (data[i].bitsize != bitlen)
                throw std::invalid_argument("native ring COT requires a uniform width");
            correlations[i] = data[i].value & mask;
        }
        // The extension creates a fresh random base r; the peer receives
        // r + choice * correlation modulo 2^bitlen. No caller mask is reused.
        send_batched_cot_u64(using_aux_iknp, sender_base.data(), correlations.data(),
                             {bitlen}, length);
        for (int i = 0; i < length; ++i)
            output[i] = GroupElement(sender_base[i] & mask, bitlen);
        preprocessing_ledger.native_ring_cot_values += length;
        return;
    }
    auto rng = secure_prng();
    std::vector<std::array<uint64_t, 2>> msgs(length);
    std::vector<uint64_t*> msg_ptrs(length);
    for (int i = 0; i < length; i++) {
        output[i].bitsize = bitlen;
        output[i].value = rng.get<uint64_t>() & mask;
        msgs[i][0] = output[i].value;
        msgs[i][1] = (output[i].value + data[i].value) & mask;
        msg_ptrs[i] = msgs[i].data();
    }
    send_ot_u64(using_aux_iknp, msg_ptrs.data(), length, bitlen);
}

void Peer::recv_cot(uint64_t* recv_arr, int size, uint8_t* sel) {
    recv_ot_u64(false, recv_arr, sel, size, 64);
}

void Peer::recv_cot(GroupElement* recv_arr, int size, uint8_t* sel,
                    bool using_aux_iknp) {
    if (size <= 0) {
        return;
    }
    const int bitlen = recv_arr[0].bitsize;
    std::vector<uint64_t> recv_value_arr(size);
    if (native_ring_cot()) {
        if (bitlen < 1 || bitlen > 64)
            throw std::invalid_argument("native ring COT width must be in [1,64]");
        auto choices = std::make_unique<bool[]>(size);
        for (int i = 0; i < size; ++i) {
            if (recv_arr[i].bitsize != bitlen)
                throw std::invalid_argument("native ring COT requires a uniform width");
            choices[i] = (sel[i] & 1U) != 0;
        }
        recv_batched_cot_u64(using_aux_iknp, recv_value_arr.data(), choices.get(),
                             {bitlen}, size);
        const uint64_t mask = bit_mask(bitlen);
        for (int i = 0; i < size; ++i)
            recv_arr[i] = GroupElement(recv_value_arr[i] & mask, bitlen);
        preprocessing_ledger.native_ring_cot_values += size;
        return;
    }
    recv_ot_u64(using_aux_iknp, recv_value_arr.data(), sel, size, bitlen);
    for (int i = 0; i < size; i++) {
        recv_arr[i].value = recv_value_arr[i];
        recv_arr[i].bitsize = bitlen;
    }
}

void Peer::send_cot(osuCrypto::block input, osuCrypto::block* output,
                    int size, bool using_aux_iknp) {
    std::vector<osuCrypto::block> corr(size, input);
    send_cot(corr.data(), output, size, using_aux_iknp);
}

void Peer::send_cot(const osuCrypto::block* input, osuCrypto::block* output,
                    int size, bool using_aux_iknp) {
    if (size <= 0) {
        return;
    }
    auto rng = secure_prng();
    std::vector<std::array<uint64_t, 2>> low_msgs(size);
    std::vector<std::array<uint64_t, 2>> high_msgs(size);
    std::vector<uint64_t*> low_ptrs(size);
    std::vector<uint64_t*> high_ptrs(size);
    for (int i = 0; i < size; i++) {
        const uint64_t low = rng.get<uint64_t>();
        const uint64_t high = rng.get<uint64_t>();
        uint64_t corr_low;
        uint64_t corr_high;
        split_block(input[i], &corr_low, &corr_high);
        output[i] = make_block_from_limbs(low, high);
        low_msgs[i][0] = low;
        low_msgs[i][1] = low ^ corr_low;
        high_msgs[i][0] = high;
        high_msgs[i][1] = high ^ corr_high;
        low_ptrs[i] = low_msgs[i].data();
        high_ptrs[i] = high_msgs[i].data();
    }
    send_ot_u64(using_aux_iknp, low_ptrs.data(), size, 64);
    send_ot_u64(using_aux_iknp, high_ptrs.data(), size, 64);
}

void Peer::recv_cot(osuCrypto::block* recv_arr, int size, bool* sel,
                    bool using_aux_iknp) {
    if (size <= 0) {
        return;
    }
    std::vector<uint8_t> choice(size);
    for (int i = 0; i < size; i++) {
        choice[i] = static_cast<uint8_t>(sel[i] != 0);
    }
    recv_cot(recv_arr, size, choice.data(), using_aux_iknp);
}

void Peer::recv_cot(osuCrypto::block* recv_arr, int size, uint8_t* sel,
                    bool using_aux_iknp) {
    if (size <= 0) {
        return;
    }
    std::vector<uint8_t> choice(size);
    std::vector<uint64_t> low(size);
    std::vector<uint64_t> high(size);
    for (int i = 0; i < size; i++) {
        choice[i] = static_cast<uint8_t>(sel[i] != 0);
    }
    recv_ot_u64(using_aux_iknp, low.data(), choice.data(), size, 64);
    recv_ot_u64(using_aux_iknp, high.data(), choice.data(), size, 64);
    for (int i = 0; i < size; i++) {
        recv_arr[i] = make_block_from_limbs(low[i], high[i]);
    }
}

void Peer::send_ot_block(bool using_aux_iknp, const osuCrypto::block* msg0,
                         const osuCrypto::block* msg1, int length) {
    if (length <= 0) {
        return;
    }
    static_assert(sizeof(osuCrypto::block) == sizeof(sci::block128),
                  "SCI and osuCrypto block sizes must match");
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    uint64_t pre_rounds = ot->io->num_rounds;
    ot->send(reinterpret_cast<const sci::block128*>(msg0),
             reinterpret_cast<const sci::block128*>(msg1), length);
    rounds += (ot->io->num_rounds - pre_rounds);
    bytesSent += (ot->io->counter - pre_comm);
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::recv_ot_block(bool using_aux_iknp, osuCrypto::block* output,
                         const uint8_t* choice, int length) {
    if (length <= 0) {
        return;
    }
    static_assert(sizeof(osuCrypto::block) == sizeof(sci::block128),
                  "SCI and osuCrypto block sizes must match");
    auto bool_choice = std::make_unique<bool[]>(length);
    for (int i = 0; i < length; i++) {
        bool_choice[i] = choice[i] != 0;
    }
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    uint64_t pre_rounds = ot->io->num_rounds;
    ot->recv(reinterpret_cast<sci::block128*>(output), bool_choice.get(),
             length);
    rounds += (ot->io->num_rounds - pre_rounds);
    bytesSent += (ot->io->counter - pre_comm);
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::send_cot_block(bool using_aux_iknp, osuCrypto::block* data0,
                          osuCrypto::block delta, int length) {
    if (length <= 0) {
        return;
    }
    std::vector<osuCrypto::block> deltas(length, delta);
    send_cot_block(using_aux_iknp, data0, deltas.data(), length);
}

void Peer::send_cot_block(bool using_aux_iknp, osuCrypto::block* data0,
                          const osuCrypto::block* delta, int length) {
    if (length <= 0) {
        return;
    }
    static_assert(sizeof(osuCrypto::block) == sizeof(sci::block128),
                  "SCI and osuCrypto block sizes must match");
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    const uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    const uint64_t pre_rounds = ot->io->num_rounds;
    ot->send_cot(reinterpret_cast<sci::block128*>(data0),
                 reinterpret_cast<const sci::block128*>(delta), length);
    const std::lock_guard<std::mutex> lock(ot_counter_mutex_);
    rounds += ot->io->num_rounds - pre_rounds;
    bytesSent += ot->io->counter - pre_comm;
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::recv_cot_block(bool using_aux_iknp, osuCrypto::block* output,
                          const uint8_t* choice, int length) {
    if (length <= 0) {
        return;
    }
    static_assert(sizeof(osuCrypto::block) == sizeof(sci::block128),
                  "SCI and osuCrypto block sizes must match");
    auto bool_choice = std::make_unique<bool[]>(length);
    for (int i = 0; i < length; ++i) {
        bool_choice[i] = choice[i] != 0;
    }
    auto* ot = using_aux_iknp ? otpack->iknp_reversed
                              : otpack->iknp_straight;
    const uint64_t pre_comm = ot->io->counter;
    const uint64_t pre_recv = ot->io->recv_counter;
    const uint64_t pre_rounds = ot->io->num_rounds;
    ot->recv_cot(reinterpret_cast<sci::block128*>(output),
                 bool_choice.get(), length);
    const std::lock_guard<std::mutex> lock(ot_counter_mutex_);
    rounds += ot->io->num_rounds - pre_rounds;
    bytesSent += ot->io->counter - pre_comm;
    bytesReceived += ot->io->recv_counter - pre_recv;
}

void Peer::mill(uint8_t *res, uint64_t *data, int num_cmps, int bitlength,
          bool greater_than, bool equality, int radix_base){
    MillInstance->configure(bitlength, radix_base);
    uint64_t pre_com_kkot = MillInstance->otpack->kkot[MillInstance->beta - 1]->io->counter;
    uint64_t pre_round_kkot = MillInstance->otpack->kkot[MillInstance->beta - 1]->io->num_rounds;
    uint64_t pre_com_iknp_s = MillInstance->otpack->iknp_straight->io->counter;
    uint64_t pre_round_iknp_s = MillInstance->otpack->iknp_straight->io->num_rounds;
    MillInstance->compare(res, data, num_cmps, bitlength, greater_than, equality, radix_base);
    uint64_t delta_com_kkot = MillInstance->otpack->kkot[MillInstance->beta - 1]->io->counter - pre_com_kkot;
    uint64_t delta_round_kkot = MillInstance->otpack->kkot[MillInstance->beta - 1]->io->num_rounds - pre_round_kkot;
    uint64_t delta_com_iknp_s = MillInstance->otpack->iknp_straight->io->counter - pre_com_iknp_s;
    uint64_t delta_round_iknp_s = MillInstance->otpack->iknp_straight->io->num_rounds - pre_round_iknp_s;
    rounds += (delta_round_iknp_s + delta_round_kkot);
    bytesSent += (delta_com_iknp_s + delta_com_kkot);
}

void Peer::mill(uint8_t *res, const uint64_t *dataA, const uint64_t* dataB, int num_cmps, int bitlength,
          bool greater_than, bool equality, int radix_base){
    // Send a-b to another one
    std::vector<uint64_t> dataC(num_cmps);
    uint64_t mask = 1ULL << bitlength;
    for (int i = 0; i < num_cmps; i++){
        dataC[i] = (dataA[i] - dataB[i]) % mask;
    }
    if (role_ == SERVER){
        send_u64(dataC.data(), num_cmps);
        for (int i = 0; i < num_cmps; i++){
            dataC[i] = 0;
        }
    }else{
        std::vector<uint64_t> data_tmp(num_cmps);
        recv_u64(data_tmp.data(), num_cmps);
        for (int i = 0; i < num_cmps; i++){
            dataC[i] = (dataC[i] + data_tmp[i]) % mask;
        }
    }
    mill(res, dataC.data(), num_cmps, bitlength, greater_than, equality, radix_base);
}

void Peer::mill(uint8_t *res, const GroupElement *data, int num_cmps,
          bool greater_than, bool equality, int radix_base){
    std::vector<uint64_t> u64data(num_cmps);
    int bitlen = data->bitsize;
    for (int i = 0; i < num_cmps; i++){
        u64data[i] = data[i].value;
    }
    mill(res, u64data.data(), num_cmps, bitlen, greater_than, equality, radix_base);
}

void Peer::mill(uint8_t *res, const GroupElement *dataA, const GroupElement* dataB, int num_cmps,
                bool greater_than, bool equality, int radix_base){
    std::vector<uint64_t> u64dataA(num_cmps);
    std::vector<uint64_t> u64dataB(num_cmps);
    int bitlen = dataA->bitsize;
    for (int i = 0; i < num_cmps; i++){
        u64dataA[i] = dataA[i].value;
        u64dataB[i] = dataB[i].value;
    }
    mill(res, u64dataA.data(), u64dataB.data(), num_cmps, bitlen, greater_than, equality, radix_base);
}

Dealer::Dealer(std::string ip, int port) {
    this->consocket = socket(AF_INET, SOCK_STREAM, 0);
    if (consocket < 0) {
        perror("socket");
        exit(1);
    }
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip.c_str());
    if (connect(consocket, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("connect");
        exit(1);
    }
}

void Dealer::close() {
    if (useFile) {
        file.close();
    }
    else {
        ::close(consocket);
    }
}

GroupElement Dealer::recv_mask() {
    char buf[8];
    if (useFile) {
        this->file.read(buf, 8);
    } else {
        recv(consocket, buf, 8, MSG_WAITALL);
    }
    GroupElement g(loadFromBytes<uint64_t>(buf), bitlength);
    bytesReceived += 8;
    return g;
}

osuCrypto::block Dealer::recv_block() {
    char buf[sizeof(osuCrypto::block)];
    if (useFile) {
        this->file.read(buf, sizeof(osuCrypto::block));
    } else {
        recv(consocket, buf, sizeof(osuCrypto::block), MSG_WAITALL);
    }
    osuCrypto::block b = loadFromBytes<osuCrypto::block>(buf);
    bytesReceived += sizeof(osuCrypto::block);
    return b;
}

GroupElement Dealer::recv_ge(int bl) {
    if (bl > 32) {
        char buf[8];
        if (useFile) {
            this->file.read(buf, 8);
        } else {
            recv(consocket, buf, 8, MSG_WAITALL);
        }
        GroupElement g(loadFromBytes<uint64_t>(buf), bl);
        bytesReceived += 8;
        return g;
    }
    else if (bl > 16) {
        char buf[4];
        if (useFile) {
            this->file.read(buf, 4);
        } else {
            recv(consocket, buf, 4, MSG_WAITALL);
        }
        GroupElement g(loadFromBytes<uint32_t>(buf), bl);
        bytesReceived += 4;
        return g;
    }
    else if (bl > 8) {
        char buf[2];
        if (useFile) {
            this->file.read(buf, 2);
        } else {
            recv(consocket, buf, 2, MSG_WAITALL);
        }
        GroupElement g(loadFromBytes<uint16_t>(buf), bl);
        bytesReceived += 2;
        return g;
    }
    else {
        char buf[1];
        if (useFile) {
            this->file.read(buf, 1);
        } else {
            recv(consocket, buf, 1, MSG_WAITALL);
        }
        GroupElement g(loadFromBytes<uint8_t>(buf), bl);
        bytesReceived += 1;
        return g;
    }
}

void Peer::sync() {
    char buf[1] = {1};
    trace_raw_send(1);
    send(sendsocket, buf, 1, 0);
    recv(recvsocket, buf, 1, MSG_WAITALL);
    trace_raw_received(1);
    bytesReceived += 1;
    bytesSent += 1;
    always_assert(buf[0] == 1);
}
