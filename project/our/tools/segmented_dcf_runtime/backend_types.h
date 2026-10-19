#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sdcf_runtime {
struct Layer {
    int index = 0, step = 0;
    std::string operator_name, function, source_call, output_tensor;
    std::vector<int64_t> numeric_arguments;
    std::vector<std::string> input_tensor_names;
    std::vector<int> shape;
    int output_bits = 0, count = 0;
};

// Only additional channels belong here. SCI channels are metered separately.
struct Counters {
    uint64_t sent_bytes = 0, received_bytes = 0, send_calls = 0, recv_calls = 0;
    uint64_t offline_generation_epoch = 0, logical_rounds = 0;
    bool logical_rounds_known = false;
    bool io_calls_known = true;
};
}
