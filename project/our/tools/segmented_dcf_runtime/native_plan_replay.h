#ifndef SDCF_NATIVE_PLAN_REPLAY_H
#define SDCF_NATIVE_PLAN_REPLAY_H
#include "native_planning.h"
#include "prepared_got.h"
#include "prepared_cot.h"

namespace sdcf_planning {
template<class OT> void register_iknp(OT* ot, int context, const std::string& direction) {
    register_ot(ot, context, direction, [ot](int query, const std::vector<OTCall>& calls) {
        int ordinary = 0;
        for (const auto& call : calls) if (call.protocol == "iknp_got") ordinary += call.count;
        ot->set_precomp_batch_size(ordinary);
        if (ordinary) ot->preprocess();
        for (size_t index = 0; index < calls.size(); ++index) {
            const auto& call = calls[index];
            if (call.protocol == "iknp_batched_got") {
                sdcf_preprocess::prepare_batched_got(ot, {query, int(index), call.protocol, call.count, call.bits, call.group});
            } else if (call.protocol == "iknp_cot" || call.protocol == "iknp_batched_cot") {
                ot->prepare_arithmetic_cot({{call.count, call.widths, call.group, call.protocol == "iknp_batched_cot"}});
            } else if (call.protocol != "iknp_got") throw std::runtime_error("unsupported IKNP public request");
        }
    });
}
template<class OT> void register_kkot(OT* ot, int context, const std::string& direction) {
    register_ot(ot, context, direction, [ot](int query, const std::vector<OTCall>& calls) {
        for (size_t index = 0; index < calls.size(); ++index) {
            const auto& call = calls[index];
            if (call.protocol != "kkot_got") throw std::runtime_error("unsupported KKOT public request");
            sdcf_preprocess::prepare_kkot(ot, {query, int(index), call.protocol, call.count, call.bits, call.group});
        }
    });
}
template<class Pack> void register_pack(Pack* pack, int context) {
    register_iknp(pack->iknp_straight, context, "straight");
    register_iknp(pack->iknp_reversed, context, "reversed");
    for (int index = 0; index < 8; ++index)
        register_kkot(pack->kkot[index], context, "kkot_" + std::to_string(1 << (index + 1)));
    register_triples(pack, context);
}
}
#endif
