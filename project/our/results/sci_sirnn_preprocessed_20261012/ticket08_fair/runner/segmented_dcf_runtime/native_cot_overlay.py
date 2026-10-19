"""Keep SCI's algorithms and adapt arithmetic COT from finalized random OT."""


def native_cot_overlays(out, sci):
    target = out / "include_override/OT/split-iknp.h"
    text = target.read_text()
    text = text.replace('#include "native_preprocessing.h"',
                        '#include "native_preprocessing.h"\n#include "prepared_cot.h"', 1)
    anchor = "  // General OT sender with message length > 64"
    if text.count(anchor) != 1:
        raise ValueError("SCI COT member insertion point changed")
    members = '''  sdcf_arithmetic::Stock arithmetic_stock;
  void prepare_arithmetic_cot(const std::vector<sdcf_arithmetic::Call>& calls) {
    sdcf_arithmetic::prepare(*this, arithmetic_stock, calls);
  }

'''
    text = text.replace(anchor, members + anchor, 1)
    for role in ("send", "recv"):
        signature = ("  void send_cot(uint64_t *data0, uint64_t *corr, int length, int l) {" if role == "send" else
                     "  void recv_cot(uint64_t *data, bool *b, int length, int l) {")
        if text.count(signature) != 1:
            raise ValueError("SCI ordinary COT entry changed")
        call = ("sdcf_arithmetic::send(*this, arithmetic_stock, data0, corr, {length, {l}, 1, false});" if role == "send" else
                "sdcf_arithmetic::receive(*this, arithmetic_stock, data, b, {length, {l}, 1, false});")
        text = text.replace(signature, signature + "\n    if (sdcf_preprocess::strict && sdcf_preprocess::phase != sdcf_preprocess::Phase::Preparing) {\n      " + call + "\n      return;\n    }", 1)
    for role in ("send", "recv"):
        signature = ("  void send_batched_cot(uint64_t *data0, uint64_t *corr,\n"
                     "                        std::vector<int> msg_len, int num_ot,\n"
                     "                        int msgs_per_ot = 1) {" if role == "send" else
                     "  void recv_batched_cot(uint64_t *data, bool *b, std::vector<int> msg_len,\n"
                     "                        int num_ot, int msgs_per_ot = 1) {")
        if text.count(signature) != 1:
            raise ValueError("SCI batched COT entry changed")
        call = ("sdcf_arithmetic::send(*this, arithmetic_stock, data0, corr, {num_ot, msg_len, msgs_per_ot, true});"
                if role == "send" else
                "sdcf_arithmetic::receive(*this, arithmetic_stock, data, b, {num_ot, msg_len, msgs_per_ot, true});")
        text = text.replace(signature, signature + "\n    if (sdcf_preprocess::strict && sdcf_preprocess::phase != sdcf_preprocess::Phase::Preparing) {\n      " + call + "\n      return;\n    }", 1)
    target.write_text(text)
