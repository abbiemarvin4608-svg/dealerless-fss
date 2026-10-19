#include "sci_msb_bridge.h"
#include "BuildingBlocks/aux-protocols.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Small two-party regression for the input-independent scheduled API.  It
// exercises serialization/deserialization, shape checking, exhaustion, and
// reconstructed MSB correctness without running the full Google-30 graph.
namespace {

constexpr int kCount = 16;
constexpr int kWidth = 16;
constexpr int kSegments = 4;
constexpr int kCallsPerSchedule = 2;
constexpr uint64_t kMask = (uint64_t(1) << kWidth) - 1;

// Include the signed transition points explicitly.  The values 32768 and
// 65535 are the ring encodings of -32768 and -1, respectively, so this list
// exercises both sides of the two's-complement sign boundary.
std::vector<uint64_t> testValues(int round, int call) {
    const uint64_t fixed[] = {
        0, 1, 32767, 32768, 65535, 32769, 65534, 2,
    };
    std::vector<uint64_t> values;
    values.reserve(kCount);
    for (uint64_t value : fixed) values.push_back(value);

    // Deterministic pseudo-random values alternate positive and negative
    // signed representatives.  No randomness is needed for this regression;
    // changing the schedule round/call changes the input pattern while
    // keeping both parties in lockstep.
    uint32_t state = 0x9e3779b9u ^ uint32_t(round * 0x45d9f3bu) ^
                     uint32_t(call * 0x27d4eb2du);
    while (static_cast<int>(values.size()) < kCount) {
        state = state * 1664525u + 1013904223u;
        const uint16_t magnitude = static_cast<uint16_t>(state & 0x7fffu);
        const int16_t signed_value =
            (values.size() & 1) ? -static_cast<int16_t>(magnitude)
                                : static_cast<int16_t>(magnitude);
        values.push_back(static_cast<uint16_t>(signed_value));
    }
    for (auto& value : values) value &= kMask;
    return values;
}

int checkCall(void* bridge, sci::IOPack& io, int role,
              const std::vector<uint64_t>& values, int width, int nseg) {
    const int count = static_cast<int>(values.size());
    std::vector<uint64_t> shares(count);
    for (int i = 0; i < count; ++i) {
        const uint64_t local =
            (uint64_t(i * 7919 + 17) ^ uint64_t(i * i * 31 + 5)) & kMask;
        shares[i] = role == 1 ? local : (values[i] - local) & kMask;
    }

    std::vector<uint8_t> output(count), remote(count);
    if (segfss_sci_msb(bridge, shares.data(), output.data(), count, width,
                       nseg) != 0) {
        std::cerr << segfss_sci_error() << '\n';
        return 1;
    }
    if (role == 1) {
        io.io->send_data(output.data(), output.size());
        io.io->recv_data(remote.data(), remote.size());
    } else {
        io.io->recv_data(remote.data(), remote.size());
        io.io->send_data(output.data(), output.size());
    }
    int failed = 0;
    for (int i = 0; i < count; ++i) {
        const uint8_t expected =
            static_cast<uint8_t>((values[i] >> (width - 1)) & 1);
        if ((output[i] ^ remote[i]) != expected) ++failed;
    }
    return failed;
}

bool expectFailure(int result, const char* operation) {
    if (result == 0) {
        std::cerr << operation << " unexpectedly succeeded\n";
        return false;
    }
    return true;
}

int checkThreadConfig() {
    const std::string overflow = std::to_string(std::numeric_limits<std::size_t>::max()) + "0";
    const std::vector<std::string> invalid_values{
        "", "0", "-1", "+4", " 4", "4 ", "4x", "1.5", overflow};
    for (const auto& value : invalid_values) {
        setenv("SEGFSS_CGGM_THREADS", value.c_str(), 1);
        // Invalid configuration must fail before waiting for a peer socket.
        void* bridge = segfss_sci_create(1, "127.0.0.1", 32000);
        if (bridge != nullptr || std::string(segfss_sci_error()) !=
                "SEGFSS_CGGM_THREADS must be a positive integer") {
            std::cerr << "invalid local cGGM thread configuration was accepted\n";
            return 1;
        }
    }
    unsetenv("SEGFSS_CGGM_THREADS");
    std::cout << "CGGM_THREAD_CONFIG invalid_values=" << invalid_values.size() << " PASS\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if(argc==1)return checkThreadConfig();
    if(argc!=3)return 2;
    const int role=std::atoi(argv[1]);
    const int port=std::atoi(argv[2]);
    sci::IOPack io(role,port,"127.0.0.1");
    sci::OTPack ot(&io,role);
    void* bridge=segfss_sci_create(role,"127.0.0.1",port+20);
    if(!bridge){std::cerr<<segfss_sci_error()<<'\n';return 3;}
    int failed=0;
    // Invalid public shapes must be rejected without consuming a schedule or
    // touching the peer socket.
    failed += !expectFailure(
        segfss_sci_precompute_schedule(bridge,0,kWidth,kSegments,
                                        kCallsPerSchedule),
        "zero-count schedule");
    failed += !expectFailure(
        segfss_sci_precompute_schedule(bridge,kCount,1,kSegments,
                                        kCallsPerSchedule),
        "one-bit schedule");
    failed += !expectFailure(
        segfss_sci_precompute_schedule(bridge,kCount,kWidth,kWidth,
                                        kCallsPerSchedule),
        "nseg-width schedule");

    if(segfss_sci_precompute_schedule(bridge,kCount,kWidth,kSegments,
                                      kCallsPerSchedule)!=0) {
        std::cerr<<segfss_sci_error()<<'\n';segfss_sci_destroy(bridge);return 4;
    }

    // A schedule cannot be replaced while any of its fresh materials remain
    // unconsumed.  finish() reports the same invariant.
    failed += !expectFailure(segfss_sci_finish(bridge),
                             "finish before schedule consumption");
    failed += !expectFailure(
        segfss_sci_precompute_schedule(bridge,kCount,kWidth,kSegments,
                                        kCallsPerSchedule),
        "replace unconsumed schedule");

    // A shape mismatch must not consume a scheduled material.  The following
    // correctly shaped call therefore still consumes the first entry.
    std::vector<uint8_t> wrong(kCount - 1);
    std::vector<uint64_t> first_values = testValues(0, 0);
    std::vector<uint64_t> first_shares(kCount);
    for (int i = 0; i < kCount; ++i) {
        const uint64_t local = (uint64_t(i * 7919 + 17) ^
                                uint64_t(i * i * 31 + 5)) & kMask;
        first_shares[i] = role == 1 ? local : (first_values[i] - local) & kMask;
    }
    failed += !expectFailure(
        segfss_sci_msb(bridge,first_shares.data(),wrong.data(),kCount - 1,
                       kWidth,kSegments),
        "wrong-count scheduled call");
    failed += checkCall(bridge,io,role,first_values,kWidth,kSegments);
    failed += checkCall(bridge,io,role,testValues(0,1),kWidth,kSegments);

    // Exhaustion is reported after both scheduled entries have been consumed.
    std::vector<uint8_t> output(kCount);
    failed += !expectFailure(
        segfss_sci_msb(bridge,first_shares.data(),output.data(),kCount,
                       kWidth,kSegments),
        "scheduled-call exhaustion");
    if(segfss_sci_finish(bridge)!=0) {
        std::cerr << segfss_sci_error() << '\n';
        failed++;
    }

    // A consumed schedule can be replaced on the same context.  The second
    // round uses a different boundary/pseudo-random input pattern and fresh
    // material, then checks exhaustion again.
    if(segfss_sci_precompute_schedule(bridge,kCount,kWidth,kSegments,
                                      kCallsPerSchedule)!=0) {
        std::cerr<<segfss_sci_error()<<'\n';failed++;
    } else {
        failed += checkCall(bridge,io,role,testValues(1,0),kWidth,kSegments);
        failed += checkCall(bridge,io,role,testValues(1,1),kWidth,kSegments);
        failed += !expectFailure(
            segfss_sci_msb(bridge,first_shares.data(),output.data(),kCount,
                           kWidth,kSegments),
            "second scheduled-call exhaustion");
        if(segfss_sci_finish(bridge)!=0) {
            std::cerr << segfss_sci_error() << '\n';
            failed++;
        }
    }

    segfss_sci_stats stats{};segfss_sci_get_stats(bridge,&stats);
    if(stats.calls!=2*kCallsPerSchedule ||
       stats.elements!=uint64_t(2*kCallsPerSchedule*kCount) ||
       stats.segment_keys!=uint64_t(2*kCallsPerSchedule*kCount*kSegments)) {
        std::cerr << "unexpected schedule statistics\n";
        failed++;
    }
    if(segfss_sci_finish(bridge)!=0)failed++;
    segfss_sci_destroy(bridge);
    std::cout<<"SCHEDULE_TEST role="<<role<<" calls="<<stats.calls
             <<" offline_ms="<<stats.offline_ms<<" online_ms="<<stats.online_ms
             <<" failures="<<failed<<'\n';
    return failed?1:0;
}
