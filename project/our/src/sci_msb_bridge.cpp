#include "sci_msb_bridge.h"
#include "private_msb.h"
#include "activation.h"
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <type_traits>

// Hidden by the export map: never interpose SCI's party=1/2 global.
int party=0,port=0,party_instance=0,num_threads=1,num_argmax=1000,length=1;
int32_t bitlength=32;
uint8_t choice_bit=0;
bool verbose=false;
std::string address="127.0.0.1";
Peer *peer=nullptr,*client=nullptr,*server=nullptr;
Dealer* dealer=nullptr;
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
struct ScheduleEntry { int count=0, width=0, nseg=0; };
struct Context {
    std::unique_ptr<Peer> connection;
    int role = 0;
    segfss_sci_stats stats{};
    std::ofstream trace;
    // Legacy query-recorded mode.  It is retained for old callers, but the
    // scheduled mode below is the path used by the Batch=128 phase runs.
    std::vector<ScheduleEntry> schedule;
    std::vector<segfss::MsbMaterial> legacy_prepared;
    size_t legacy_cursor=0;
    bool legacy_prepared_mode=false;

    // Input-independent scheduled materials are serialized one record at a
    // time.  Keeping only the current record in memory avoids retaining the
    // 297 Batch=128 FSS materials (which exceed the one-GiB runner limit).
    FILE* prepared_file=nullptr;
    ScheduleEntry prepared_shape{};
    size_t prepared_calls=0;
    size_t prepared_cursor=0;
    bool prepared_poisoned=false;
    struct ActivationEntry {
        segfss::SigmaActivationKey key;
        segfss::ExactTruncationKey tanh_output_trunc;
        bool tanh=false;
    };
    std::vector<ActivationEntry> activation_material;
    size_t activation_cursor=0;
    bool activation_poisoned=false;
};
thread_local std::string last_error;
thread_local Context* active=nullptr;
std::mutex contexts_mutex;
std::unordered_set<Context*> contexts;
struct ContextScope {
    Context* previous;
    explicit ContextScope(void* ptr) : previous(active) {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        auto* requested=static_cast<Context*>(ptr);
        active=contexts.count(requested) ? requested : nullptr;
    }
    ~ContextScope() { active = previous; }
};
struct Counters { uint64_t sent,received,rounds; };
Counters snapshot(){const auto& peer = active->connection; return {peer->bytesSent,peer->bytesReceived,peer->rounds};}

std::size_t cggmThreadLimit() {
    const char* value = std::getenv("SEGFSS_CGGM_THREADS");
    if (!value) return 1;
    std::size_t threads = 0;
    const char* end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, threads);
    if (parsed.ec != std::errc{} || parsed.ptr != end || threads == 0) {
        throw std::invalid_argument("SEGFSS_CGGM_THREADS must be a positive integer");
    }
    return threads;
}

void refreshTotals(Context& ctx) {
    ctx.stats.sent_bytes = ctx.stats.setup_sent_bytes +
        ctx.stats.offline_sent_bytes + ctx.stats.online_sent_bytes;
    ctx.stats.received_bytes = ctx.stats.setup_received_bytes +
        ctx.stats.offline_received_bytes + ctx.stats.online_received_bytes;
    ctx.stats.rounds = ctx.stats.setup_rounds + ctx.stats.offline_rounds +
        ctx.stats.online_rounds;
}

void accountOffline(Context& ctx, const Counters& before,
                    const Counters& after, double elapsed_ms) {
    ctx.stats.offline_ms += elapsed_ms;
    ctx.stats.offline_sent_bytes += after.sent-before.sent;
    ctx.stats.offline_received_bytes += after.received-before.received;
    ctx.stats.offline_rounds += after.rounds-before.rounds;
    refreshTotals(ctx);
}

void accountOnline(Context& ctx, const Counters& before,
                   const Counters& after, double elapsed_ms) {
    ctx.stats.online_ms += elapsed_ms;
    ctx.stats.online_sent_bytes += after.sent-before.sent;
    ctx.stats.online_received_bytes += after.received-before.received;
    ctx.stats.online_rounds += after.rounds-before.rounds;
    refreshTotals(ctx);
}

template <typename T>
void writePod(FILE* file, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "serialized scalar must be trivially copyable");
    if(std::fwrite(&value, sizeof(T), 1, file) != 1)
        throw std::runtime_error("FSS material temporary write failed");
}

template <typename T>
T readPod(FILE* file) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "serialized scalar must be trivially copyable");
    T value{};
    if(std::fread(&value, sizeof(T), 1, file) != 1)
        throw std::runtime_error("FSS material temporary read failed");
    return value;
}

void writeBlock(FILE* file, const osuCrypto::block& block) {
    alignas(16) uint8_t bytes[sizeof(osuCrypto::block)];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes), block);
    if(std::fwrite(bytes, sizeof(bytes), 1, file) != 1)
        throw std::runtime_error("FSS block temporary write failed");
}

osuCrypto::block readBlock(FILE* file) {
    alignas(16) uint8_t bytes[sizeof(osuCrypto::block)];
    if(std::fread(bytes, sizeof(bytes), 1, file) != 1)
        throw std::runtime_error("FSS block temporary read failed");
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes));
}

void writeBytes(FILE* file, const std::vector<uint8_t>& values) {
    const uint64_t size=values.size(); writePod(file,size);
    if(size && std::fwrite(values.data(), sizeof(uint8_t), size, file)!=size)
        throw std::runtime_error("FSS byte-vector temporary write failed");
}

std::vector<uint8_t> readBytes(FILE* file, uint64_t expected,
                               const char* name) {
    const uint64_t size=readPod<uint64_t>(file);
    if(size!=expected)throw std::runtime_error(std::string("FSS material ")+name+" length mismatch");
    std::vector<uint8_t> values(static_cast<size_t>(size));
    if(size && std::fread(values.data(), sizeof(uint8_t), size, file)!=size)
        throw std::runtime_error("FSS byte-vector temporary read failed");
    return values;
}

void writeKey(FILE* file, const segfss::CggmKey& key) {
    writePod(file, static_cast<int32_t>(key.depth));
    writeBlock(file,key.root);
    writePod(file, static_cast<uint64_t>(key.internal_cw.size()));
    for(const auto& block:key.internal_cw)writeBlock(file,block);
    writeBlock(file,key.hcw);
    writePod(file,key.lcw[0]); writePod(file,key.lcw[1]);
}

segfss::CggmKey readKey(FILE* file, int expected_depth) {
    segfss::CggmKey key;
    key.depth=readPod<int32_t>(file);
    if(key.depth!=expected_depth)
        throw std::runtime_error("FSS material key depth mismatch");
    key.root=readBlock(file);
    const uint64_t internal=readPod<uint64_t>(file);
    if(internal!=static_cast<uint64_t>(std::max(0,key.depth-1)))
        throw std::runtime_error("FSS material internal correction-word length mismatch");
    key.internal_cw.resize(static_cast<size_t>(internal));
    for(auto& block:key.internal_cw)block=readBlock(file);
    key.hcw=readBlock(file);
    key.lcw[0]=readPod<uint8_t>(file); key.lcw[1]=readPod<uint8_t>(file);
    if((key.lcw[0]&~uint8_t(1))||(key.lcw[1]&~uint8_t(1)))
        throw std::runtime_error("FSS material leaf correction bit is not binary");
    return key;
}

void writeMaterial(FILE* file, const segfss::MsbMaterial& material) {
    writePod(file,static_cast<int32_t>(material.width));
    writePod(file,static_cast<int32_t>(material.count));
    writePod(file,static_cast<int32_t>(material.nseg));
    writePod(file,static_cast<uint64_t>(material.mask_shares.size()));
    for(uint64_t value:material.mask_shares)writePod(file,value);
    writeBytes(file,material.mask_sign_shares);
    writePod(file,static_cast<uint64_t>(material.keys.size()));
    for(const auto& key:material.keys)writeKey(file,key);
    writeBytes(file,material.scan.a);
    writeBytes(file,material.scan.b);
    writeBytes(file,material.scan.c);
}

segfss::MsbMaterial readMaterial(FILE* file, const ScheduleEntry& shape) {
    segfss::MsbMaterial material;
    material.width=readPod<int32_t>(file);
    material.count=readPod<int32_t>(file);
    material.nseg=readPod<int32_t>(file);
    if(material.width!=shape.width||material.count!=shape.count||material.nseg!=shape.nseg)
        throw std::runtime_error("FSS material schedule shape mismatch");
    const uint64_t expected_count=static_cast<uint64_t>(shape.count);
    const uint64_t expected_keys=expected_count*static_cast<uint64_t>(shape.nseg);
    const uint64_t mask_count=readPod<uint64_t>(file);
    if(mask_count!=expected_count)throw std::runtime_error("FSS mask length mismatch");
    material.mask_shares.resize(static_cast<size_t>(mask_count));
    for(auto& value:material.mask_shares)value=readPod<uint64_t>(file);
    material.mask_sign_shares=readBytes(file,expected_count,"mask-sign");
    const uint64_t key_count=readPod<uint64_t>(file);
    if(key_count!=expected_keys)throw std::runtime_error("FSS key count mismatch");
    material.keys.reserve(static_cast<size_t>(key_count));
    segfss::SegmentLayout layout(shape.width-1,shape.nseg);
    for(uint64_t i=0;i<key_count;++i)
        material.keys.emplace_back(readKey(file,layout.width[static_cast<size_t>(i)%shape.nseg]));
    const uint64_t expected_scan=2*expected_count*static_cast<uint64_t>(shape.nseg-1);
    material.scan.a=readBytes(file,expected_scan,"scan-a");
    material.scan.b=readBytes(file,expected_scan,"scan-b");
    material.scan.c=readBytes(file,expected_scan,"scan-c");
    material.scan.cursor=0;
    return material;
}

void accountOfflineMaterial(Context& ctx, const segfss::MsbMaterial& material) {
    ctx.stats.mask_conversion_ms += material.mask_conversion_ms;
    ctx.stats.keygen_ms += material.keygen_ms;
    ctx.stats.scan_preprocessing_ms += material.scan_preprocessing_ms;
}

void accountOnlineMaterial(Context& ctx, const segfss::MsbMaterial& material) {
    ctx.stats.masked_input_open_ms += material.masked_input_open_ms;
    ctx.stats.evaluation_ms += material.evaluation_ms;
    ctx.stats.scan_ms += material.scan_ms;
}
}
extern "C" {
void* segfss_sci_cggm_scope_begin(int participants) {
    try {
        if (participants < 1) throw std::invalid_argument("invalid caller cGGM participant cap");
        return new segfss::CggmThreadScope(participants);
    } catch (const std::exception& ex) { last_error = ex.what(); return nullptr; }
}
int segfss_sci_cggm_scope_stats(void* pointer, int* limit, int* peak, int* transport) {
    if (!pointer || !limit || !peak || !transport) return -1;
    const auto* scope = static_cast<segfss::CggmThreadScope*>(pointer);
    *limit = int(scope->effectiveLimit()); *peak = int(scope->observedParticipants());
    *transport = int(scope->observedTransportParticipants());
    return 0;
}
int segfss_sci_cggm_scope_end(void* pointer) {
    if (!pointer) return -1;
    auto* scope = static_cast<segfss::CggmThreadScope*>(pointer);
    const auto peak = std::max(scope->observedParticipants(), scope->observedTransportParticipants());
    delete scope;
    return int(peak);
}
void segfss_sci_set_round_trace_callbacks(segfss_sci_round_io_callback io,
    segfss_sci_round_fork_callback fork_actor, segfss_sci_round_actor_callback enter_actor,
    segfss_sci_round_actor_callback leave_actor, segfss_sci_round_join_callback join_actor) {
    dfss_round::callbacks.io = io; dfss_round::callbacks.fork = fork_actor;
    dfss_round::callbacks.enter = enter_actor; dfss_round::callbacks.leave = leave_actor;
    dfss_round::callbacks.join = join_actor;
}
void segfss_sci_set_round_scope_callbacks(segfss_sci_round_scope_begin_callback begin,
    segfss_sci_round_scope_end_callback end) {
    dfss_round::callbacks.scope_begin = begin; dfss_round::callbacks.scope_end = end;
}
void* segfss_sci_create(int sci_party,const char* host,int network_port) {
    try {
        if(sci_party<1||sci_party>2||network_port<1024||network_port>65400||!host)
            throw std::invalid_argument("bridge requires SCI role 1/2, valid host/port");
        auto ctx=std::make_unique<Context>();
        ctx->role=sci_party+1;
        auto start=Clock::now();
        segfss::setCggmThreadLimit(cggmThreadLimit());
        dfss_round::callbacks.caller_limit = segfss::cggmCallerThreadLimit;
        dfss_round::callbacks.transport_participants = segfss::observeCallerTransportParticipants;
        ctx->connection.reset(ctx->role==SERVER ? waitForPeer(network_port,ctx->role) : new Peer(host,network_port,ctx->role));
        auto* peer=ctx->connection.get();
        // Older activation APIs retain one default connection; conversion and
        // MSB use the explicit connection owned by their handle.
        if (!::peer) { ::peer=ctx->connection.get(); ::party=ctx->role; ::port=network_port; }
        ctx->stats.setup_ms=ms(start);
        // Include base-OT traffic before wrapper accounting begins.
        const auto* io=peer->getIOPack();
        ctx->stats.setup_sent_bytes=io->io->counter+io->io_rev->counter+io->io_GC->counter+peer->bytesSent;
        ctx->stats.setup_received_bytes=io->io->recv_counter+io->io_rev->recv_counter+io->io_GC->recv_counter+peer->bytesReceived;
        ctx->stats.setup_rounds=io->io->num_rounds+io->io_rev->num_rounds+io->io_GC->num_rounds+peer->rounds;
        if(const char* filename=std::getenv("SEGFSS_SCI_TRACE")) {
            ctx->trace.open(filename,std::ios::app);
            if(!ctx->trace)throw std::runtime_error("cannot open public metadata trace");
        }
        std::lock_guard<std::mutex> lock(contexts_mutex);
        contexts.insert(ctx.get());
        return ctx.release();
    } catch(const std::exception& e){last_error=e.what();return nullptr;}
}
int segfss_sci_msb(void* ptr,const uint64_t* input,uint8_t* output,int count,int width,int nseg) {
    ContextScope scope(ptr);
    const int party=active ? active->role : 0;
    Peer* peer=active ? active->connection.get() : nullptr;
    bool material_loaded=false;
    try {
        if(ptr!=active||!active||!input||!output)throw std::invalid_argument("invalid bridge context/input/output");
        auto& s=active->stats;
        segfss::MsbMaterial material;
        Counters before=snapshot();
        const bool scheduled=active->prepared_file!=nullptr;
        if(active->prepared_poisoned)
            throw std::runtime_error("prepared FSS schedule is poisoned after an online failure");
        auto online_start=Clock::now();
        if(scheduled) {
            if(active->prepared_cursor>=active->prepared_calls)
                throw std::runtime_error("prepared FSS schedule exhausted");
            const auto& expected=active->prepared_shape;
            if(expected.count!=count||expected.width!=width||expected.nseg!=nseg)
                throw std::runtime_error("prepared FSS call shape differs from scheduled shape");
            // Loading is part of the input-dependent online window.  Only the
            // current record is materialized in RAM.
            material_loaded=true;
            material=readMaterial(active->prepared_file,expected);
            material.owner_peer=peer;
            // The file cursor has advanced; consume the schedule slot before
            // any network activity so a failed online call cannot be retried
            // against a mismatched material record.
            ++active->prepared_cursor;
        } else if(active->legacy_prepared_mode) {
            if(active->legacy_cursor>=active->legacy_prepared.size())
                throw std::runtime_error("prepared FSS schedule exhausted");
            const auto& expected=active->schedule[active->legacy_cursor];
            if(expected.count!=count||expected.width!=width||expected.nseg!=nseg)
                throw std::runtime_error("prepared FSS call differs from recorded schedule");
            material=std::move(active->legacy_prepared[active->legacy_cursor++]);
        } else {
            auto start=Clock::now();
            material=segfss::privateMsbOffline(party,count,width,nseg,peer);
            accountOffline(*active,before,snapshot(),ms(start));
            accountOfflineMaterial(*active,material);
            active->schedule.push_back({count,width,nseg});
        }
        auto mid=snapshot();
        if(!scheduled)online_start=Clock::now();
        segfss::privateMsbOnline(party,input,output,material,peer);
        auto after=snapshot();
        accountOnline(*active,mid,after,ms(online_start));
        accountOnlineMaterial(*active,material);
        s.calls++;s.elements+=count;s.segment_keys+=uint64_t(count)*nseg;
        refreshTotals(*active);
        if(active->trace)active->trace<<"{\"primitive\":\"segmented_private_msb\",\"input_bits\":"<<width<<",\"comparison_bits\":"<<width-1<<",\"nseg\":"<<nseg<<",\"count\":"<<count<<",\"keys_generated\":"<<count*nseg<<",\"fresh_material\":true,\"opened_original_inputs\":0}"<<std::endl;
        return 0;
    }catch(const std::exception& e){
        if(active&&material_loaded)active->prepared_poisoned=true;
        last_error=e.what();return -1;
    }
}
int segfss_sci_precompute_recorded(void* ptr) {
    ContextScope scope(ptr);
    const int party=active ? active->role : 0;
    Peer* peer=active ? active->connection.get() : nullptr;
    try {
        if(ptr!=active||!active)throw std::invalid_argument("invalid bridge context");
        if(active->prepared_file||active->legacy_prepared_mode)throw std::runtime_error("FSS schedule already prepared");
        if(active->schedule.empty())throw std::runtime_error("no FSS call schedule recorded");
        active->legacy_prepared.clear();
        active->legacy_prepared.reserve(active->schedule.size());
        active->legacy_cursor=0;
        for(const auto& call:active->schedule) {
            const auto before=snapshot();
            const auto start=Clock::now();
            auto material=segfss::privateMsbOffline(party,call.count,call.width,call.nseg,peer);
            const auto after=snapshot();
            accountOffline(*active,before,after,ms(start));
            accountOfflineMaterial(*active,material);
            active->legacy_prepared.emplace_back(std::move(material));
        }
        active->legacy_prepared_mode=true;
        return 0;
    } catch(const std::exception& e){last_error=e.what();return -1;}
}
int segfss_sci_precompute_schedule(void* ptr,int count,int width,int nseg,int calls) {
    ContextScope scope(ptr);
    const int party=active ? active->role : 0;
    Peer* peer=active ? active->connection.get() : nullptr;
    try {
        if(ptr!=active||!active)throw std::invalid_argument("invalid bridge context");
        if(active->prepared_poisoned||active->legacy_prepared_mode)
            throw std::runtime_error("cannot start scheduled FSS after failure or legacy replay");
        if(active->prepared_file) {
            if(active->prepared_cursor!=active->prepared_calls)
                throw std::runtime_error("previous FSS schedule not fully consumed");
            std::fclose(active->prepared_file);
            active->prepared_file=nullptr;
        } else if(active->stats.calls!=0) {
            throw std::runtime_error("cannot switch a lazy FSS session to scheduled mode");
        }
        if(count<1||width<2||width>64||nseg<1||nseg>=width||calls<1)
            throw std::invalid_argument("invalid FSS schedule shape");
        active->prepared_file=std::tmpfile();
        if(!active->prepared_file)throw std::runtime_error("cannot create FSS material temporary file");
        active->prepared_shape={count,width,nseg};
        active->prepared_calls=static_cast<size_t>(calls);
        active->prepared_cursor=0;
        active->prepared_poisoned=false;
        try {
            for(int index=0;index<calls;++index) {
                const auto before=snapshot();
                const auto start=Clock::now();
                auto material=segfss::privateMsbOffline(party,count,width,nseg,peer);
                accountOfflineMaterial(*active,material);
                writeMaterial(active->prepared_file,material);
                accountOffline(*active,before,snapshot(),ms(start));
            }
            const auto flush_start=Clock::now();
            if(std::fflush(active->prepared_file)!=0||std::fseek(active->prepared_file,0,SEEK_SET)!=0)
                throw std::runtime_error("cannot rewind FSS material temporary file");
            active->stats.offline_ms+=ms(flush_start);
        } catch(...) {
            std::fclose(active->prepared_file);active->prepared_file=nullptr;
            active->prepared_calls=active->prepared_cursor=0;
            throw;
        }
        return 0;
    } catch(const std::exception& e){last_error=e.what();return -1;}
}
int segfss_sci_finish(void* ptr) {
    ContextScope scope(ptr);
    if(ptr!=active||!active){last_error="invalid bridge context";return -1;}
    if(active->prepared_poisoned || active->activation_poisoned) {
        last_error="prepared FSS schedule is poisoned after an online failure";return -1;
    }
    if(active->prepared_file&&active->prepared_cursor!=active->prepared_calls) {
        last_error="prepared FSS schedule not fully consumed";return -1;
    }
    if(active->legacy_prepared_mode&&active->legacy_cursor!=active->legacy_prepared.size()) {
        last_error="recorded FSS schedule not fully consumed";return -1;
    }
    if(!active->activation_material.empty()&&
       active->activation_cursor!=active->activation_material.size()) {
        last_error="activation schedule not fully consumed";return -1;
    }
    return 0;
}
int segfss_sci_get_stats(void* ptr,segfss_sci_stats* output){ContextScope scope(ptr);if(!active||!output)return -1;*output=active->stats;return 0;}
void* segfss_sci_get_peer(void* ptr){ContextScope scope(ptr);return active ? active->connection.get() : nullptr;}
int segfss_sci_precompute_activations(void* ptr,int batch,int steps,
                                      int elements_per_call,int bits) {
    ContextScope scope(ptr);
    const int party=active ? active->role : 0;
    Peer* peer=active ? active->connection.get() : nullptr;
    try {
        if(ptr!=active||!active)throw std::invalid_argument("invalid bridge context");
        if(batch<1||steps<1||elements_per_call<1||bits<2||bits>62)
            throw std::invalid_argument("invalid activation schedule shape");
        if(peer!=::peer)throw std::invalid_argument("legacy FSS activation requires the default context");
        if(active->activation_poisoned)
            throw std::runtime_error("activation schedule is poisoned after an online failure");
        const int64_t count64=int64_t(batch)*elements_per_call;
        if(count64>std::numeric_limits<int>::max())
            throw std::invalid_argument("activation call shape exceeds supported size");
        const int count=static_cast<int>(count64);
        if(!active->activation_material.empty()) {
            if(active->activation_cursor!=active->activation_material.size())
                throw std::runtime_error("activation schedule already prepared");
            for(auto& entry:active->activation_material) {
                segfss::freeSigmaActivationKey(entry.key);
                segfss::freeExactTruncationKey(entry.tanh_output_trunc);
            }
            active->activation_material.clear();
            active->activation_cursor=0;
        }
        const auto before=snapshot();
        const auto start=Clock::now();
        active->activation_material.reserve(static_cast<size_t>(steps)*2*
                                            static_cast<size_t>(count));
        for(int step=0;step<steps;++step) {
            for(int kind=0;kind<2;++kind) {
                auto keys=segfss::sigmaBatchOffline(party,count,bits,12,10,peer);
                for(int element=0;element<count;++element) {
                    Context::ActivationEntry entry;
                    entry.tanh=(kind==1);
                    entry.key = std::move(keys[static_cast<size_t>(element)]);
                    if(entry.tanh)
                        entry.tanh_output_trunc = segfss::exactTruncateOffline(
                            party,bits,5,peer);
                    active->activation_material.emplace_back(std::move(entry));
                    ++active->stats.activation_keys;
                }
            }
        }
        const auto elapsed=ms(start);
        accountOffline(*active,before,snapshot(),elapsed);
        active->stats.activation_offline_ms += elapsed;
        return 0;
    } catch(const std::exception& e){last_error=e.what();return -1;}
}
int segfss_sci_activation(void* ptr,int kind,const uint64_t* input,
                          uint64_t* output,int count,int bits) {
    ContextScope scope(ptr);
    const int party=active ? active->role : 0;
    Peer* peer=active ? active->connection.get() : nullptr;
    bool material_loaded=false;
    try {
        if(ptr!=active||!active||!input||!output||count<1||bits<2||bits>62)
            throw std::invalid_argument("invalid activation input");
        if(peer!=::peer)throw std::invalid_argument("legacy FSS activation requires the default context");
        if(kind!=0&&kind!=1)throw std::invalid_argument("invalid activation kind");
        if(active->activation_poisoned)
            throw std::runtime_error("activation schedule is poisoned after an online failure");
        if(active->activation_cursor+static_cast<size_t>(count)>
           active->activation_material.size())
            throw std::runtime_error("activation schedule exhausted");
        const size_t offset=active->activation_cursor;
        for(int i=0;i<count;++i) {
            const auto& entry=active->activation_material[offset+static_cast<size_t>(i)];
            if(entry.tanh!=(kind==1)||entry.key.bits!=bits)
                throw std::runtime_error("activation schedule kind or width mismatch");
        }
        const auto before=snapshot();
        const auto start=Clock::now();
        const uint64_t ring_mask=bits==64?UINT64_MAX:((uint64_t(1)<<bits)-1);
        std::vector<GroupElement> inputs;
        std::vector<segfss::SigmaActivationKey> keys;
        inputs.reserve(static_cast<size_t>(count));
        keys.reserve(static_cast<size_t>(count));
        for(int i=0;i<count;++i) {
            auto& entry=active->activation_material[offset+static_cast<size_t>(i)];
            GroupElement x(input[i]&ring_mask,bits);
            // Google-30 exposes Q7 activations.  Convert the share to the
            // internal Q12 contract locally before secure evaluation.
            inputs.push_back(x*uint64_t(1u<<5));
            keys.push_back(std::move(entry.key));
        }
        material_loaded=true;
        active->activation_cursor+=static_cast<size_t>(count);
        std::vector<GroupElement> outputs;
        try {
            outputs=kind==1
                ? segfss::tanhBatchOnline(party,inputs,keys,peer)
                : segfss::sigmaBatchOnline(party,inputs,keys,peer);
            if(kind==1) {
                std::vector<segfss::ExactTruncationKey*> truncations;
                truncations.reserve(static_cast<size_t>(count));
                for(int i=0;i<count;++i)
                    truncations.push_back(&active->activation_material[offset+static_cast<size_t>(i)].tanh_output_trunc);
                outputs=segfss::exactTruncateBatch(party,outputs,truncations,peer);
            } else {
                for(auto& y:outputs)y=y*uint64_t(1u<<2);
            }
        } catch(...) {
            for(int i=0;i<count;++i)
                active->activation_material[offset+static_cast<size_t>(i)].key=std::move(keys[static_cast<size_t>(i)]);
            throw;
        }
        for(int i=0;i<count;++i) {
            output[i]=outputs[static_cast<size_t>(i)].value&ring_mask;
            active->stats.activation_lut_online_ms+=keys[static_cast<size_t>(i)].segmented_lut.online_ms;
            active->activation_material[offset+static_cast<size_t>(i)].key=std::move(keys[static_cast<size_t>(i)]);
        }
        active->stats.activation_elements+=static_cast<uint64_t>(count);
        active->stats.activation_calls++;
        const auto after=snapshot();
        const auto elapsed=ms(start);
        accountOnline(*active,before,after,elapsed);
        active->stats.activation_online_ms += elapsed;
        return 0;
    } catch(const std::exception& e){
        if(active&&material_loaded)active->activation_poisoned=true;
        last_error=e.what();return -1;
    }
}
const char* segfss_sci_error(){return last_error.c_str();}
void segfss_sci_destroy(void* ptr) {
    ContextScope scope(ptr);
    if(!active)return;
    for(auto& entry:active->activation_material) {
        segfss::freeSigmaActivationKey(entry.key);
        segfss::freeExactTruncationKey(entry.tanh_output_trunc);
    }
    if(active->prepared_file)std::fclose(active->prepared_file);
    if(::peer==active->connection.get())::peer=nullptr;
    active->connection->close();
    { std::lock_guard<std::mutex> lock(contexts_mutex); contexts.erase(active); }
    delete active;
    active=nullptr;
}
}
