#ifndef SDCF_NATIVE_QUERY_MATERIALS_H
#define SDCF_NATIVE_QUERY_MATERIALS_H
#include "native_plan_replay.h"
#include "prepared_triples.h"
#include "Millionaire/bit-triple-generator.h"

namespace sdcf_planning {
inline void prepare_query_materials(int query, std::vector<TripleGenerator*> generators) {
    sdcf_preprocess::require_query(query);
    if (active() || !sdcf_preprocess::strict ||
        sdcf_preprocess::phase != sdcf_preprocess::Phase::Preparing)
        throw std::runtime_error("native query replay outside prepare");
    std::sort(generators.begin(),generators.end(),[](const auto* a,const auto* b) {
        return context(a->otpack) < context(b->otpack);
    });
    for (size_t index = 1; index < generators.size(); ++index)
        if (context(generators[index - 1]->otpack) == context(generators[index]->otpack))
            throw std::runtime_error("duplicate native triple replay context");
    for (const auto& item : triple_streams) {
        if (item.second.calls.empty()) continue;
        if (std::none_of(generators.begin(),generators.end(),[&](const auto* generator) {
            return generator->otpack == item.first;
        })) throw std::runtime_error("missing native triple replay context");
    }
    prepare_ots(query);
    for (auto* generator : generators) {
        std::vector<sdcf_triples::Call> calls;
        for (const auto& call : triple_calls(generator->otpack))
            calls.push_back({call.count,call.method,call.packed,call.offset});
        sdcf_triples::plan(generator->otpack,query,context(generator->otpack),calls);
        sdcf_triples::prepare(generator);
    }
    sdcf_triples::seal(query);
    if (sdcf_dpf::prepare_query) sdcf_dpf::prepare_query(query);
    if (sdcf_folded::prepare_query) sdcf_folded::prepare_query(query);
}
}
#endif
