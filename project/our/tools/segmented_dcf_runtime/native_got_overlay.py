"""Split native KKOT and grouped GOT at their public SCI entry points."""
from pathlib import Path


def native_got_overlay(out):
    target = Path(out) / "include_override/OT"
    path = target / "split-iknp.h"
    text = path.read_text()
    text = text.replace('#include "native_preprocessing.h"',
                        '#include "native_preprocessing.h"\n#include "prepared_got.h"', 1)
    for role in ("send", "recv"):
        signature = ("void send_batched_got(uint64_t *data, int num_ot, int l,\n"
                     "                        int msgs_per_ot = 1) {" if role == "send" else
                     "void recv_batched_got(uint64_t *data, const uint8_t *r, int num_ot, int l,\n"
                     "                        int msgs_per_ot = 1) {")
        hook = ("prepared_batched_got_send(this, data, num_ot, l, msgs_per_ot)" if role == "send" else
                "prepared_batched_got_recv(this, data, r, num_ot, l, msgs_per_ot)")
        if text.count(signature) != 1: raise ValueError("SCI batched GOT entry changed: " + signature)
        text = text.replace(signature, signature + "\n    if (sdcf_preprocess::" + hook + ") return;", 1)
    path.write_text(text)
    path = target / "split-kkot.h"
    text = path.read_text()
    text = text.replace('#include "native_preprocessing.h"',
                        '#include "native_preprocessing.h"\n#include "prepared_got.h"', 1)
    for type_, role in (("uint64_t", "send"), ("uint64_t", "recv"),
                        ("uint8_t", "send"), ("uint8_t", "recv")):
        signature = ("void send_impl(" + type_ + " **data, int length, int l) {" if role == "send" else
                     "void recv_impl(" + type_ + " *data, uint8_t *b, int length, int l) {")
        hook = ("prepared_kkot_send(this, data, length, l)" if role == "send" else
                "prepared_kkot_recv(this, data, b, length, l)")
        if text.count(signature) != 1: raise ValueError("SCI KKOT wrapper changed: " + signature)
        text = text.replace(signature, signature + "\n    if (sdcf_preprocess::" + hook + ") return;", 1)
    path.write_text(text)
