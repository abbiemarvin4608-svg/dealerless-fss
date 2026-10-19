#include "sci_dpf_bridge.h"
#include "sci_msb_bridge.h"
#include "dpf_lut.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
struct Entry {
    segfss_sci_dpf_call shape{};
    uint64_t id = 0;
    std::vector<PublicLUTData> tables;
    std::vector<segfss::DpfPublicLutMaterial> material;
};
struct Context {
    void* network = nullptr;
    Peer* connection = nullptr;
    int party = 0;
    uint64_t epoch = 0, next_id = 0, generated = 0, consumed = 0;
    uint64_t setup_sent = 0, setup_received = 0;
    size_t cursor = 0;
    bool poisoned = false;
    segfss::DpfLutBatchMetrics metrics;
    std::vector<Entry> entries;
    ~Context() { if(network) segfss_sci_destroy(network); }
};
thread_local std::string error;
uint64_t ring_mask(int bits) { return (uint64_t(1) << bits) - 1; }
void validate(const segfss_sci_dpf_call& call) {
    if(call.query < 0 || call.slot < 0 || call.call < 0 || call.count < 1 ||
       call.index_bits < 1 || call.index_bits > 8 || call.tables < 1 || call.tables > 2 ||
       (call.grouped_tables && call.count % call.tables))
        throw std::runtime_error("invalid public SCI DPF LUT call");
    for(int t=0;t<call.tables;++t)
        if(call.output_bits[t] < 1 || call.output_bits[t] > 62 || !call.values[t])
            throw std::runtime_error("invalid public SCI DPF LUT table");
}
bool same(const segfss_sci_dpf_call& a,const segfss_sci_dpf_call& b) {
    if(a.query!=b.query || a.layer!=b.layer || a.slot!=b.slot || a.call!=b.call ||
       a.family!=b.family || a.count!=b.count || a.index_bits!=b.index_bits ||
       a.tables!=b.tables || a.grouped_tables!=b.grouped_tables) return false;
    for(int t=0;t<a.tables;++t) if(a.output_bits[t]!=b.output_bits[t]) return false;
    return true;
}
}
extern "C" {
uint64_t segfss_sci_dpf_material_metadata_bytes(void) {
    return segfss::DpfPublicLutMaterial{}.residentBytes();
}
void* segfss_sci_dpf_create(int role,const char* host,int port) {
    try {
        auto context=std::make_unique<Context>();
        context->network=segfss_sci_create(role,host,port);
        if(!context->network) throw std::runtime_error(segfss_sci_error());
        context->connection=static_cast<Peer*>(segfss_sci_get_peer(context->network));
        segfss_sci_stats stats{};
        if(segfss_sci_get_stats(context->network,&stats))
            throw std::runtime_error(segfss_sci_error());
        // Peer records later OT calls as channel-counter deltas. Preserve only
        // the construction traffic absent from Peer, rather than adding live
        // IOPack counters and counting lookup traffic twice.
        context->setup_sent=stats.setup_sent_bytes-context->connection->bytesSent;
        context->setup_received=stats.setup_received_bytes-context->connection->bytesReceived;
        context->party=role+1;
        return context.release();
    } catch(const std::exception& ex) { error=ex.what();return nullptr; }
}
int segfss_sci_dpf_prepare(void* ptr,const segfss_sci_dpf_call* calls,int count) {
    auto* ctx=static_cast<Context*>(ptr);
    try {
        if(!ctx || count<0 || (count && !calls) || ctx->poisoned || ctx->cursor!=ctx->entries.size())
            throw std::runtime_error("SCI DPF LUT previous inventory unfinished");
        ctx->entries.clear(); ctx->cursor=0;ctx->generated=ctx->consumed=0;
        ctx->metrics={};
        ++ctx->epoch;
        for(int c=0;c<count;++c) {
            validate(calls[c]);
            Entry item;item.shape=calls[c];item.id=ctx->next_id;
            ctx->next_id+=uint64_t(item.shape.count);
            if(ctx->next_id<item.id) throw std::runtime_error("SCI DPF LUT request overflow");
            for(int t=0;t<item.shape.tables;++t) {
                PublicLUTData table;table.Bin=item.shape.index_bits;table.Bout=item.shape.output_bits[t];
                for(uint64_t i=0;i<(uint64_t(1)<<table.Bin);++i)
                    table.values.emplace_back(item.shape.values[t][i]&ring_mask(table.Bout),table.Bout);
                item.tables.push_back(std::move(table));
            }
            const auto max_table=std::max_element(item.tables.begin(),item.tables.end(),
                [](const auto& a,const auto& b){return a.Bout<b.Bout;});
            segfss::DpfLutBatchMetrics metric;
            item.material=segfss::dpfPublicLutBatchOffline(ctx->party,*max_table,
                item.shape.count,item.id,ctx->connection,ctx->epoch,&metric);
            ctx->metrics.mask_conversion_us+=metric.mask_conversion_us;
            ctx->metrics.keygen_us+=metric.keygen_us;
            ctx->metrics.eval_all_us+=metric.eval_all_us;
            ctx->metrics.offline_us+=metric.offline_us;
            ctx->metrics.eval_points+=metric.eval_points;
            ctx->generated+=uint64_t(item.shape.count);
            ctx->entries.push_back(std::move(item));
        }
        return 0;
    } catch(const std::exception& ex) {if(ctx)ctx->poisoned=true;error=ex.what();return -1;}
}
int segfss_sci_dpf_online(void* ptr,const segfss_sci_dpf_call* call,
        const uint64_t* input,uint64_t* output0,uint64_t* output1) {
    auto* ctx=static_cast<Context*>(ptr);
    try {
        if(!ctx || !call || !input || !output0 || ctx->poisoned || ctx->cursor>=ctx->entries.size())
            throw std::runtime_error("SCI DPF LUT online inventory unavailable");
        validate(*call);
        auto& item=ctx->entries[ctx->cursor];
        if(!same(item.shape,*call) || (!call->grouped_tables && call->tables==2 && !output1))
            throw std::runtime_error("SCI DPF LUT online identity mismatch");
        ++ctx->cursor;ctx->consumed+=uint64_t(call->count);
        std::vector<GroupElement> inputs;inputs.reserve(call->count);
        for(int i=0;i<call->count;++i) inputs.emplace_back(input[i]&ring_mask(call->index_bits),call->index_bits);
        std::vector<const PublicLUTData*> tables;
        for(const auto& table:item.tables) tables.push_back(&table);
        segfss::DpfLutBatchMetrics metric;
        if(call->grouped_tables) {
            const int group=call->count/call->tables;
            std::vector<const PublicLUTData*> selected;
            for(int i=0;i<call->count;++i) selected.push_back(tables.at(i/group));
            auto outputs=segfss::dpfPublicLutBatchOnlinePerInput(ctx->party,inputs,item.material,
                ctx->connection,item.id,selected,&metric);
            for(int i=0;i<call->count;++i) output0[i]=outputs.at(i).value;
        } else {
            auto outputs=segfss::dpfPublicLutBatchOnlineMany(ctx->party,inputs,item.material,
                ctx->connection,item.id,tables,&metric);
            for(int i=0;i<call->count;++i) {
                output0[i]=outputs.at(0).at(i).value;
                if(call->tables==2) output1[i]=outputs.at(1).at(i).value;
            }
        }
        ctx->metrics.online_open_us+=metric.online_open_us;
        ctx->metrics.inner_product_us+=metric.inner_product_us;
        ctx->metrics.online_open_batches+=metric.online_open_batches;
        for(auto& material:item.material) segfss::releaseDpfPublicLutMaterial(material);
        return 0;
    } catch(const std::exception& ex) {if(ctx)ctx->poisoned=true;error=ex.what();return -1;}
}
int segfss_sci_dpf_finish_query(void* ptr) {
    auto* ctx=static_cast<Context*>(ptr);
    if(!ctx || ctx->poisoned || ctx->cursor!=ctx->entries.size() || ctx->generated!=ctx->consumed) {
        if(ctx)ctx->poisoned=true;error="SCI DPF LUT unbalanced inventory";return -1;
    }
    return 0;
}
int segfss_sci_dpf_counters_get(void* ptr,segfss_sci_dpf_counters* result) {
    auto* ctx=static_cast<Context*>(ptr);if(!ctx || !result)return -1;
    *result={ctx->setup_sent+ctx->connection->bytesSent,ctx->setup_received+ctx->connection->bytesReceived,
             ctx->epoch,0,ctx->generated,ctx->consumed};
    result->mask_conversion_us=ctx->metrics.mask_conversion_us;
    result->keygen_us=ctx->metrics.keygen_us;
    result->eval_all_us=ctx->metrics.eval_all_us;
    result->offline_us=ctx->metrics.offline_us;
    result->online_open_us=ctx->metrics.online_open_us;
    result->inner_product_us=ctx->metrics.inner_product_us;
    result->eval_points=ctx->metrics.eval_points;
    result->online_open_batches=ctx->metrics.online_open_batches;
    for(const auto& entry:ctx->entries) {
        for(const auto& table:entry.tables) result->resident_bytes+=table.values.capacity()*sizeof(GroupElement);
        for(const auto& material:entry.material) result->resident_bytes+=material.residentBytes();
    }
    return 0;
}
void segfss_sci_dpf_destroy(void* ptr) {delete static_cast<Context*>(ptr);}
const char* segfss_sci_dpf_error(void) {return error.c_str();}
}
