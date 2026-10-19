#include "cases.h"

#include "ArgMapping.h"

// Command-line entry point for our segment-FSS correctness cases.
// Mirrors the reference dealerless-FSS_public runtime pattern.

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

using namespace sci;

// Process-wide runtime globals expected by the imported EzPC/FSS layer.
int party_instance = 0;
int party = 0;
int32_t bitlength = 32;
int num_threads = 1;
int port = 32000;
std::string address = "127.0.0.1";
int num_argmax = 1000;
uint8_t choice_bit = 0;
bool verbose = false;
int length = 1;
Peer* client = nullptr;
Peer* server = nullptr;
Dealer* dealer = nullptr;
Peer* peer = nullptr;

namespace {

struct Case {
    int id;
    const char* name;
    void (*run)(ResultLog&);
};

const Case kCases[] = {
    {1, "segment split", check_split},
    {2, "segment gen", check_gen},
    {3, "cggm primitive", check_cggm},
    {4, "cggm dealerless 2pc", check_cggm_2pc},
    {5, "cross-segment infer", check_infer},
    {6, "one-round BitXA ReLU", check_relu},
    {7, "correlated DPF DCF full domain", check_correlated_dcf},
    {8, "BitXA full-domain mixed multiplication", check_bitxa},
    {9, "SIGMA folded q=10 activations", check_activation},
    {10, "exact DPF truncation", check_exact_truncation},
    {11, "exact DPF multiply-truncate", check_exact_mul_truncation},
    {12, "Beaver triple offline (Gilboa COT)", check_beaver},
    {13, "batched secure matrix multiplication", check_matmul},
    {14, "batch sigmoid activations", check_activation_batch},
    {15, "bit x block OT modes", check_bit_block_modes},
    {16, "DPF public LUT lifecycle", check_dpf_public_lut},
};

std::string case_help() {
    std::ostringstream out;
    out << "Case: all = 0";
    for (const auto& c : kCases) {
        out << "; " << c.name << " = " << c.id;
    }
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    int test_case = 0;
    ArgMapping amap;
    amap.arg("r", party, "Role of party: SERVER = 2; CLIENT = 3");
    amap.arg("p", port, "Port Number");
    const std::string help = case_help();
    amap.arg("t", test_case, help.c_str());
    amap.parse(argc, argv);

    if (party == CLIENT) {
        server = new Peer(address, port);
        peer = server;
    } else if (party == SERVER) {
        client = waitForPeer(port);
        peer = client;
    } else {
        std::cerr << "Pass r=2 for SERVER or r=3 for CLIENT.\n";
        return 2;
    }

    ResultLog log;
    for (const auto& c : kCases) {
        if (test_case == 0 || test_case == c.id) {
            c.run(log);
        }
    }

    if (party == SERVER) {
        std::cout << "Correctness checks: "
                  << (log.failures() == 0 ? "PASS" : "FAIL")
                  << " (" << log.failures() << " failed)\n";
    }
    return log.failures() == 0 ? 0 : 1;
}
