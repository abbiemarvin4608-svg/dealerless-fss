#ifndef SDCF_RUNTIME_LOG_H
#define SDCF_RUNTIME_LOG_H
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace sdcf_log {
inline std::mutex mutex;
inline void emit(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << text;
}
class Line : public std::ostringstream {
public:
    ~Line() { emit(str()); }
};
}
#endif
