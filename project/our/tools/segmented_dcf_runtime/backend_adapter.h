#pragma once
#include "backend_types.h"
#include <stdexcept>

namespace sdcf_backend {
inline std::vector<std::string> supported_backends() { return {"SCI", "SCI_preprocessed"}; }
inline void validate(const std::string& name) {
    if (name != "SCI" && name != "SCI_preprocessed") throw std::runtime_error("unavailable backend: " + name);
}
inline void setup(const std::string& name, int,
                  const std::vector<sdcf_runtime::Layer>&) { validate(name); }
inline void prepare(int, const std::vector<sdcf_runtime::Layer>&) {}
inline void enter_online(int) {}
inline void layer_begin(const sdcf_runtime::Layer&) {}
inline void layer_end(const sdcf_runtime::Layer&) {}
inline void leave_online(int) {}
inline void finish() {}
inline sdcf_runtime::Counters counters() { return {}; }
}
