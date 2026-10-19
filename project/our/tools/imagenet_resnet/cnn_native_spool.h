#ifndef IMAGENET_CNN_NATIVE_SPOOL_H
#define IMAGENET_CNN_NATIVE_SPOOL_H
#include <cstdio>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

// Only storage changes: native random OT pads and triples keep their original
// public call contracts and are reconstructed immediately before consumption.
namespace cnn_spool {
struct Record { long offset; int kind; };
inline FILE* file = nullptr;
inline std::mutex mutex;
inline std::map<const void*, std::deque<Record>> records;
inline uint64_t bytes = 0, budget = 0, written = 0, loaded = 0;
inline bool sealed = false;
template<class T> void release(std::vector<T>& values) {
    std::vector<T>().swap(values);
}
template<class T> void write(const T& value) {
    if (std::fwrite(&value, sizeof(T), 1, file) != 1)
        throw std::runtime_error("native spool write failed");
}
template<class T> T read() {
    T value{};
    if (std::fread(&value, sizeof(T), 1, file) != 1)
        throw std::runtime_error("native spool read failed");
    return value;
}
template<class T> void writeVector(std::vector<T>& values) {
    write(uint64_t(values.size()));
    if (!values.empty() && std::fwrite(values.data(), sizeof(T), values.size(), file) != values.size())
        throw std::runtime_error("native spool vector write failed");
    release(values);
}
template<class T> void readVector(std::vector<T>& values) {
    const auto count = read<uint64_t>();
    if (count > bytes / sizeof(T)) throw std::runtime_error("native spool vector exceeds file");
    values.resize(count);
    if (count && std::fread(values.data(), sizeof(T), count, file) != count)
        throw std::runtime_error("native spool vector read failed");
}
inline void begin(uint64_t limit) {
    std::lock_guard<std::mutex> lock(mutex);
    if (file) throw std::runtime_error("native spool previous query unfinished");
    file = std::tmpfile();
    if (!file) throw std::runtime_error("cannot create private native spool");
    records.clear(); bytes = written = loaded = 0; budget = limit; sealed = false;
}
template<class... V> void save(const void* owner, int kind, V&... vectors) {
    if (!file) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (sealed) throw std::runtime_error("native spool generation after input");
    const auto offset = std::ftell(file);
    if (offset < 0) throw std::runtime_error("native spool position failed");
    (writeVector(vectors), ...);
    const auto end = std::ftell(file);
    if (end < 0) throw std::runtime_error("native spool size failed");
    bytes = uint64_t(end);
    if (budget && bytes > budget) throw std::runtime_error("native spool budget exceeded");
    records[owner].push_back({offset, kind}); ++written;
}
template<class... V> void load(const void* owner, int kind, V&... vectors) {
    if (!file) return;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = records.find(owner);
    if (!sealed || found == records.end() || found->second.empty() || found->second.front().kind != kind)
        throw std::runtime_error("native spool inventory mismatch");
    if (std::fseek(file, found->second.front().offset, SEEK_SET))
        throw std::runtime_error("native spool seek failed");
    (readVector(vectors), ...);
    found->second.pop_front(); ++loaded;
}
inline void seal() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!file || sealed || std::fflush(file)) throw std::runtime_error("native spool seal failed");
    sealed = true;
}
inline void finish() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!file || !sealed || loaded != written) throw std::runtime_error("native spool inventory unconsumed");
    for (const auto& item : records) if (!item.second.empty()) throw std::runtime_error("native spool inventory remains");
    std::fclose(file); file = nullptr; records.clear();
}
inline void discard() {
    std::lock_guard<std::mutex> lock(mutex);
    if (file) std::fclose(file);
    file = nullptr; records.clear();
}
}
#endif
