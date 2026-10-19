#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace segfss {
inline bool runtimeFlag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
}
