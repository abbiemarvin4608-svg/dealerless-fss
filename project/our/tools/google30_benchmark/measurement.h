#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>
#include <sys/resource.h>

namespace google30_measure {
inline std::atomic<uint64_t> sent{0}, received{0}, sends{0}, recvs{0};
using Clock = std::chrono::steady_clock;
struct Point { Clock::time_point time; uint64_t sent, received, sends, recvs; };
inline Point point() { return {Clock::now(), sent.load(), received.load(), sends.load(), recvs.load()}; }
struct Phase { double seconds; uint64_t sent, received, sends, recvs; };
inline std::map<std::string, Phase> phases;
inline Point start, last;
inline std::string current;
inline int role;
inline Point query_start;
inline std::vector<Phase> queries;
inline void begin(int r) { role=r; start=last=point(); current="application_io"; }
inline void phase(const char* name) {
  auto now=point();
  if(!current.empty()) {
    auto &p=phases[current];
    p.seconds += std::chrono::duration<double>(now.time-last.time).count();
    p.sent += now.sent-last.sent; p.received += now.received-last.received;
    p.sends += now.sends-last.sends; p.recvs += now.recvs-last.recvs;
  }
  current=name; last=now;
}
inline void query_begin() { phase("input_sharing"); query_start=point(); }
inline void query_end() {
  auto now=point(); queries.push_back({std::chrono::duration<double>(now.time-query_start.time).count(),
    now.sent-query_start.sent,now.received-query_start.received,now.sends-query_start.sends,now.recvs-query_start.recvs});
}
inline void finish() {
  phase(""); auto end=point(); struct rusage usage; getrusage(RUSAGE_SELF,&usage);
  const char *path=std::getenv("GOOGLE30_METRICS");
  if(!path) throw std::runtime_error("GOOGLE30_METRICS is required");
  std::ofstream f(path); f.precision(12);
  f << "{\"party\":" << role << ",\"total_seconds\":" << std::chrono::duration<double>(end.time-start.time).count()
    << ",\"peak_rss_kib\":" << usage.ru_maxrss << ",\"phases\":{";
  bool first=true;
  for(auto &entry:phases) { if(!first) f<<','; first=false; auto &p=entry.second;
    f << '\"' << entry.first << "\":{\"seconds\":" << p.seconds << ",\"sent_bytes\":" << p.sent
      << ",\"received_bytes\":" << p.received << ",\"send_calls\":" << p.sends << ",\"recv_calls\":" << p.recvs << '}'; }
  f << "},\"queries\":["; first=true;
  for(auto &p:queries) { if(!first) f<<','; first=false;
    f << "{\"seconds\":" << p.seconds << ",\"sent_bytes\":" << p.sent << ",\"received_bytes\":" << p.received
      << ",\"send_calls\":" << p.sends << ",\"recv_calls\":" << p.recvs << '}'; }
  f << "]}\n";
}
}
