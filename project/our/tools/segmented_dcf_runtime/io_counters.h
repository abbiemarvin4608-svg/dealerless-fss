#pragma once
#include <atomic>
#include <cstdint>
namespace sdcf_measure {
inline std::atomic<uint64_t> sent{0}, received{0}, sends{0}, recvs{0};
}
