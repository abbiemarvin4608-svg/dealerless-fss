#pragma once
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
namespace replay_trace {
inline std::string op = "initialization";
inline int step = -1;
inline std::mutex mutex;
inline thread_local std::string stack;
struct Scope {
  std::string previous;
  explicit Scope(const char *name) : previous(stack) { stack += std::string("/") + name; }
  ~Scope() { stack = previous; }
};
inline void event(const char *kind, int count, int input_bits, int output_bits, int variant = 0) {
  const char *path = std::getenv("GOOGLE30_TRACE");
  if (!path) return;
  std::lock_guard<std::mutex> lock(mutex);
  static std::ofstream stream(path);
  stream << op << ',' << step << ',' << stack << ',' << kind << ',' << count
         << ',' << input_bits << ',' << output_bits << ',' << variant << '\n';
}
}
