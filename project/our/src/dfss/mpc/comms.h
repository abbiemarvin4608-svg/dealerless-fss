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

#pragma once

#include <string>
#include <iostream>
#include <memory>
#include <mutex>
#include <cstdint>
#include <vector>
#include "commons/group_element.h"
#include "commons/keypack.h"
#include "OT/emp-ot.h"
#include "OT/ot_pack.h"
#include <cryptoTools/Common/Defines.h>
#include "Millionaire/bit-triple-generator.h"
#include "Millionaire/millionaire.h"

#include "OT/split-iknp.h"
#include "OT/iknp.h"
#include "utils/round_trace.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>

#define DEALER 1
#define SERVER 2
#define CLIENT 3

extern int party;
extern int port;

// Logical instantaneous products, independent of extension padding and of
// explicit prepare/consume inventories. Network totals remain Peer counters.
struct BitBlockLedger {
    uint64_t calls = 0, products = 0;
    uint64_t temporary_triples_generated = 0, temporary_triples_consumed = 0;
    uint64_t direct_products = 0;
    uint64_t general_ot_64 = 0, general_ot_128 = 0;
    uint64_t native_cot_128 = 0;
    uint64_t ef_openings = 0;
};

struct PreprocessingLedger {
    uint64_t boolean_triples = 0;
    uint64_t arithmetic_triples = 0;
    uint64_t b2a_values = 0;
    // Endpoint calls: sender and receiver each count the same correlations.
    uint64_t native_ring_cot_values = 0;
};

class Peer {
public:
    int sendsocket, recvsocket;
    bool useFile = false;
    std::fstream file;
    uint64_t bytesSent = 0;
    uint64_t bytesReceived = 0;
    uint64_t rounds = 0;
    BitBlockLedger bit_block_ledger;
    PreprocessingLedger preprocessing_ledger;
    uint64_t additive_openings = 0;
    uint64_t opening_microseconds = 0;

    osuCrypto::PRNG& randomSource() noexcept { return random_source_; }

    sci::IOPack* getIOPack() noexcept {
        return iopack_.get();
    }

    sci::OTPack* getOTPack() noexcept {
        return otpack.get();
    }

private:
    int role_ = 0;
    osuCrypto::PRNG random_source_{osuCrypto::sysRandomSeed()};
    std::unique_ptr<sci::IOPack> iopack_;
    std::unique_ptr<sci::OTPack> otpack;
    std::unique_ptr<sci::IKNP<sci::NetIO>> block_ot;
    std::unique_ptr<sci::IKNP<sci::NetIO>> block_ot_reversed;
    std::unique_ptr<MillionaireProtocol> MillInstance;
    int ot_precompute_batch_size_ = 0;
    std::mutex ot_counter_mutex_;
    uint64_t raw_sent_offset_ = 0, raw_received_offset_ = 0;

public:
    void trace_raw_send(uint64_t length) {
        if (useFile) return;
        dfss_round::io(0, sendsocket, role_ == SERVER, raw_sent_offset_, length, 200000);
        raw_sent_offset_ += length;
    }
    void trace_raw_received(uint64_t length) {
        if (useFile) return;
        dfss_round::io(1, recvsocket, role_ == SERVER, raw_received_offset_, length, 200000);
        raw_received_offset_ += length;
    }
    Peer(std::string ip, int port);
    Peer(std::string ip, int port, int role);
    Peer(int sendsocket, int recvsocket) : Peer(sendsocket, recvsocket, party, port) {}
    Peer(int sendsocket, int recvsocket, int role, int network_port) : role_(role) {
        this->sendsocket = sendsocket;
        this->recvsocket = recvsocket;
        // Here we change party number from 2S 3C to 1S, 2C
        this->iopack_ = std::make_unique<sci::IOPack>(role - 1, network_port);
        this->otpack = std::make_unique<sci::OTPack>(this->iopack_.get(), role - 1);
        this->block_ot = std::make_unique<sci::IKNP<sci::NetIO>>(this->iopack_->io);
        this->block_ot_reversed = std::make_unique<sci::IKNP<sci::NetIO>>(this->iopack_->io_rev);
        if (!this->MillInstance){
            this->MillInstance = std::make_unique<MillionaireProtocol>(role - 1, iopack_.get(), otpack.get());
        }
        configure_ot_precompute(128);
    }
    Peer(std::string filename) {
        this->useFile = true;
        this->file.open(filename, std::ios::out | std::ios::binary);
    }

    void close();
    int role() const noexcept { return role_; }
    void open_additive(GroupElement* shares, int size, int bits);
    void open_xor(u8* shares, int size);
    void open_xor(osuCrypto::block* shares, int size);
    // Independent leaf high-block and low-byte CWs share one opening.
    void open_xor_mixed(osuCrypto::block* blocks, int block_count,
                        u8* bytes, int byte_count);

    void send_ge(const GroupElement &g, int bw);


    void send_block(const osuCrypto::block &b);

    void send_block(const osuCrypto::block* b, int size);

    void recv_block(osuCrypto::block* output, int size);

    void send_u8(const u8&);

    void send_u8(const u8* b, int size);

    void recv_u8(u8* output, int size);

    void send_u64(const uint64_t &b);

    void send_u64(const uint64_t* input, int size);

    void recv_u64(uint64_t* output, int size);

    uint64_t recv_u64();

    osuCrypto::block recv_block();

    u8 recv_u8();

    void send_ge(const GroupElement &g);

    GroupElement recv_ge();

    void send_mask(const GroupElement &g);

    void send_input(const GroupElement &g);

    void send_batched_input(const GroupElement *g, int size, int bw);

    void sync();

    GroupElement recv_input();

    void recv_batched_input(uint64_t *g, int size, int bw);

    void send_cot(osuCrypto::block, osuCrypto::block*, int, bool using_aux_iknp = false);

    void send_cot(const osuCrypto::block*, osuCrypto::block*, int, bool using_aux_iknp = false);

    void recv_cot(osuCrypto::block* recv_arr, int size, bool* sel, bool using_aux_iknp = false);

    void recv_cot(osuCrypto::block* recv_arr, int size, uint8_t* sel,
                  bool using_aux_iknp = false);

    void send_ot_block(bool using_aux_iknp, const osuCrypto::block* msg0,
                       const osuCrypto::block* msg1, int length);

    void recv_ot_block(bool using_aux_iknp, osuCrypto::block* output,
                       const uint8_t* choice, int length);

    void send_cot_block(bool using_aux_iknp, osuCrypto::block* data0,
                        osuCrypto::block delta, int length);

    void send_cot_block(bool using_aux_iknp, osuCrypto::block* data0,
                        const osuCrypto::block* delta, int length);

    void recv_cot_block(bool using_aux_iknp, osuCrypto::block* output,
                        const uint8_t* choice, int length);

    void send_cot(uint64_t, uint64_t*, int);

    void recv_cot(uint64_t* recv_arr, int size, uint8_t* sel);

    void send_cot(const GroupElement* data, GroupElement* output, int length, bool using_aux_iknp);

    void recv_cot(GroupElement* recv_arr, int size, uint8_t* sel, bool using_aux_iknp);

    void configure_ot_precompute(int batch_size);

    void reset_ot_precompute();

    int ot_precompute_batch_size() const noexcept {
        return ot_precompute_batch_size_;
    }

    // Length-reduced arithmetic COT (SCI send/recv_batched_cot). The num_ot
    // OTs are grouped into msg_len.size() equal consecutive groups; group g
    // carries msg_len[g]-bit messages. Sender learns data0 and the receiver
    // gets data0 + b * corr, both mod 2^msg_len[g].
    void send_batched_cot_u64(bool using_aux_iknp, uint64_t* data0,
                              uint64_t* corr, const std::vector<int>& msg_len,
                              int num_ot);

    void recv_batched_cot_u64(bool using_aux_iknp, uint64_t* output, bool* choice,
                              const std::vector<int>& msg_len, int num_ot);

    void send_ot_u64(bool using_aux_iknp, uint64_t** msgs, int length, int bitlen);

    void recv_ot_u64(bool using_aux_iknp, uint64_t* output, uint8_t* choice,
                     int length, int bitlen);

    void mill(uint8_t *res, uint64_t *data, int num_cmps, int bitlength,
              bool greater_than = true, bool equality = false, int radix_base = MILL_PARAM);

    void mill(uint8_t *res, const uint64_t *dataA, const uint64_t* dataB, int num_cmps, int bitlength,
              bool greater_than = true, bool equality = false, int radix_base = MILL_PARAM);

    void mill(uint8_t *res, const GroupElement *data, int num_cmps,
              bool greater_than = true, bool equality = false, int radix_base = MILL_PARAM);

    void mill(uint8_t *res, const GroupElement* dataA, const GroupElement* dataB, int num_cmps,
              bool greater_than = true, bool equality = false, int radix_base = MILL_PARAM);

};

Peer* waitForPeer(int port);
Peer* waitForPeer(int port, int role);

class Dealer {
public:
    int consocket;
    bool useFile = false;
    std::fstream file;
    uint64_t bytesSent = 0;
    uint64_t bytesReceived = 0;

    Dealer(std::string ip, int port);

    Dealer(std::string filename) {
        this->useFile = true;
        this->file.open(filename, std::ios::in | std::ios::binary);
    }

    void close();

    GroupElement recv_mask();

    osuCrypto::block recv_block();

    GroupElement recv_ge(int bw);

};

extern Dealer *dealer;
extern Peer *server;
extern Peer *client;
extern Peer *peer;
