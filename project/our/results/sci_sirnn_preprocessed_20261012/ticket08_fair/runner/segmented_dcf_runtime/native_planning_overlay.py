"""Intercept public shapes before native protocol state or random material changes."""


def replace(text, needle, value, count=1):
    if text.count(needle) != count:
        raise ValueError("SCI public-planning hook changed: " + needle)
    return text.replace(needle, value)


def native_planning_overlays(out, sci):
    for name in ("split-iknp.h", "split-kkot.h"):
        target = out / "include_override/OT" / name
        text = target.read_text()
        text = replace(text, '#include "native_preprocessing.h"', '#include "native_preprocessing.h"\n#include "native_planning.h"')
        for type_ in ("uint8_t", "uint64_t"):
            for role in ("send", "recv"):
                signature = ("void send_impl(" + type_ + " **data, int length, int l) {" if role == "send" else
                             "void recv_impl(" + type_ + " *data, uint8_t *b, int length, int l) {")
                protocol, group = ("iknp_got", "1") if name == "split-iknp.h" else ("kkot_got", "N")
                zero = "" if role == "send" else "std::fill_n(data, length, " + type_ + "(0)); "
                hook = '\n    if (sdcf_planning::record_ot(this, "' + protocol + '", length, l, ' + group + ')) { ' + zero + 'return; }'
                text = replace(text, signature, signature + hook)
        if name == "split-iknp.h":
            signatures = [
                ("  void send_batched_got(uint64_t *data, int num_ot, int l,\n                        int msgs_per_ot = 1) {", "iknp_batched_got", "num_ot", "l", "msgs_per_ot", "{}", ""),
                ("  void recv_batched_got(uint64_t *data, const uint8_t *r, int num_ot, int l,\n                        int msgs_per_ot = 1) {", "iknp_batched_got", "num_ot", "l", "msgs_per_ot", "{}", "std::fill_n(data, size_t(num_ot) * msgs_per_ot, uint64_t(0)); "),
                ("  void send_cot(uint64_t *data0, uint64_t *corr, int length, int l) {", "iknp_cot", "length", "l", "1", "{l}", "std::fill_n(data0, length, uint64_t(0)); "),
                ("  void recv_cot(uint64_t *data, bool *b, int length, int l) {", "iknp_cot", "length", "l", "1", "{l}", "std::fill_n(data, length, uint64_t(0)); "),
                ("  void send_batched_cot(uint64_t *data0, uint64_t *corr,\n                        std::vector<int> msg_len, int num_ot,\n                        int msgs_per_ot = 1) {", "iknp_batched_cot", "num_ot", "*std::max_element(msg_len.begin(), msg_len.end())", "msgs_per_ot", "msg_len", "std::fill_n(data0, size_t(num_ot) * msgs_per_ot, uint64_t(0)); "),
                ("  void recv_batched_cot(uint64_t *data, bool *b, std::vector<int> msg_len,\n                        int num_ot, int msgs_per_ot = 1) {", "iknp_batched_cot", "num_ot", "*std::max_element(msg_len.begin(), msg_len.end())", "msgs_per_ot", "msg_len", "std::fill_n(data, size_t(num_ot) * msgs_per_ot, uint64_t(0)); "),
            ]
            for signature, protocol, count, bits, group, widths, zero in signatures:
                hook = '\n    if (sdcf_planning::record_ot(this, "' + protocol + '", ' + ', '.join((count, bits, group, widths)) + ')) { ' + zero + 'return; }'
                text = replace(text, signature, signature + hook)
        target.write_text(text)

    target = out / "include_override/utils/prg.h"
    text = (sci / "src/utils/prg.h").read_text()
    text = replace(text, '#include "utils/aes-ni.h"', '#include "native_planning.h"\n#include <cstring>\n#include "utils/aes-ni.h"')
    for type_ in ("128", "256"):
        signature = "  PRG" + type_ + "(const void *seed = nullptr, int id = 0) {"
        hook = '\n    if (sdcf_planning::active() && seed == nullptr) { alignas(32) unsigned char zero[32] = {}; reseed(zero, id); return; }'
        text = replace(text, signature, signature + hook)
    for signature in ("  void random_data(void *data, int nbytes) {", "  void random_data_unaligned(void *data, int nbytes) {"):
        text = replace(text, signature, signature + '\n    if (sdcf_planning::active()) { std::memset(data, 0, nbytes); return; }', 2)
    for type_, size in (("128", 16), ("256", 32)):
        signature = "  void random_block(block" + type_ + " *data, int nblocks = 1) {"
        hook = '\n    if (sdcf_planning::active()) { std::memset(data, 0, size_t(nblocks) * ' + str(size) + '); return; }\n    sdcf_planning::prg_blocks += uint64_t(nblocks) * ' + str(size // 16) + ';'
        text = replace(text, signature, signature + hook, 2)
    target.write_text(text)

    target = out / "include_override/utils/net_io_channel.h"
    text = target.read_text()
    text = replace(text, '#include "io_counters.h"', '#include "io_counters.h"\n#include "native_planning.h"\n#include <cstring>')
    for signature, hook in (
        ("  void send_data(const void *data, int len) {", "\n    if (sdcf_planning::active()) return;"),
        ("  void recv_data(void *data, int len) {", "\n    if (sdcf_planning::active()) { std::memset(data, 0, len); return; }"),
        ("  void flush() {", "\n    if (sdcf_planning::active()) return;"),
    ):
        text = replace(text, signature, signature + hook)
    target.write_text(text)
