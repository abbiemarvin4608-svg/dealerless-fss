#include "round_trace.h"
#include <sys/socket.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int party = std::atoi(argv[1]);
    sdcf_round::initialize(party);
    sdcf_round::phase_begin("online", 0);
    if (std::getenv("TEST_ROUND_SCOPES")) {
        sdcf_round_trace_io(0, 42009, 0, 8);
        sdcf_round_trace_io(1, 42009, 0, 8);
        {
            sdcf_round::SemanticScope activation("activation", 0, 6, -1, -1, 0);
            {
                sdcf_round::SemanticScope stage("activation_stage", -1, -1, -1, -1, -1, 3);
                sdcf_round::Thread scoped_child([] {
                    sdcf_round_trace_io(0, 42001, 0, 8);
                    sdcf_round_trace_io(1, 42001, 0, 8);
                });
                scoped_child.join();
            }
            sdcf_round_trace_io(0, 42002, 0, 8);
            sdcf_round_trace_io(1, 42002, 0, 8);
        }
        sdcf_round::phase_end("online", 0);
        sdcf_round::finish();
        return 0;
    }
    sdcf_round::Thread child([] {
        if (std::getenv("TEST_ROUND_CHILD"))
            sdcf_round_trace_io(0, 42000, 0, 3);
    });
    child.join();
    if (party == 1) {
        sdcf_round_trace_io(0, 42000, 3, 5);
        sdcf_round_trace_io(1, 42000, 0, 8);
    } else {
        sdcf_round_trace_io(1, 42000, 0, 8);
        sdcf_round_trace_io(0, 42000, 0, 8);
    }
    sdcf_round::phase_end("online", 0);
    sdcf_round::finish();
    return 0;
}
