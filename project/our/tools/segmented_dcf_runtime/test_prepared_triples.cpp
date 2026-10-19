#include "BuildingBlocks/aux-protocols.h"
#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/value-extension.h"
#include "native_plan_replay.h"
#include "prepared_triples.h"
#include "measurement.h"
#include <array>
#include <limits>

namespace {
constexpr int count = 13;
uint64_t ring_mask(int bits) { return bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1; }
int64_t signed_value(uint64_t value, int bits) {
    if (bits == 64) return int64_t(value);
    return int64_t(value & ring_mask(bits)) - ((value >> (bits - 1)) ? int64_t(uint64_t(1) << bits) : 0);
}
uint64_t clear_value(int index, int bits, int query) {
    const std::array<int64_t, count> small{{0,1,-1,31,-31,32,-32,33,-33,4095,-4096,2047,-2048}};
    if (bits == 64 && index == 9) return uint64_t(INT64_MAX);
    if (bits == 64 && index == 10) return uint64_t(INT64_MIN);
    return uint64_t(small[(index + query) % count]) & ring_mask(bits);
}
uint64_t share_value(int index, int bits, int query, int role) {
    const uint64_t mask = ring_mask(bits);
    const std::array<uint64_t, 5> sharing{{0,UINT64_MAX,1,0x123456789abcdef0ULL,0x8000000000000000ULL}};
    const uint64_t alice = sharing[(index + query) % sharing.size()] & mask;
    return role == 1 ? alice : (clear_value(index, bits, query) - alice) & mask;
}

struct Suite {
    int role, query;
    sci::IOPack& io;
    AuxProtocols aux;
    Truncation trunc;
    XTProtocol xt;
    MillionaireWithEquality mill_eq;
    Equality eq;
    std::vector<std::pair<std::string, std::vector<uint64_t>>> results;
    uint64_t previous_ots = 0, previous_triples = 0, previous_generation = 0;
    Suite(int role, int query, sci::IOPack& io, sci::OTPack& ot)
        : role(role), query(query), io(io), aux(role,&io,&ot), trunc(role,&io,&ot),
          xt(role,&io,&ot), mill_eq(role,&io,&ot), eq(role,&io,&ot) {}

    void open(const std::string& name, const std::vector<uint64_t>& output,
              const std::vector<uint64_t>& expected, int bits, bool xor_share = false) {
        if (sdcf_planning::active()) return;
        if (sdcf_preprocess::strict) {
            uint64_t ot_count = 0, triple_count = 0;
            for (const auto& item : sdcf_preprocess::inventories) ot_count += item.second.consumed;
            for (const auto& item : sdcf_triples::inventories) triple_count += item.second.consumed;
            const uint64_t epoch = sdcf_preprocess::generation_epoch.load();
            std::cout << "CONVERSION_MATERIAL_COVERAGE {\"query\":" << query
                << ",\"protocol\":\"" << name << "\",\"native_ot_consumed\":" << ot_count-previous_ots
                << ",\"bit_triples_consumed\":" << triple_count-previous_triples
                << ",\"native_generation_epoch_delta\":" << epoch-previous_generation << "}\n";
            previous_ots = ot_count; previous_triples = triple_count; previous_generation = epoch;
        }
        std::vector<uint64_t> other(count), clear(count);
        if (role == 1) io.io->send_data(output.data(), count * sizeof(uint64_t));
        else {
            io.io->recv_data(other.data(), count * sizeof(uint64_t));
            for (int index = 0; index < count; ++index) {
                clear[index] = (xor_share ? other[index] ^ output[index] : other[index] + output[index]) & ring_mask(bits);
                if (clear[index] != expected[index])
                    throw std::runtime_error(name + " native integer contract mismatch at " + std::to_string(index)
                        + " expected " + std::to_string(expected[index]) + " actual " + std::to_string(clear[index]));
            }
            results.emplace_back(name, clear);
        }
    }

    void run(int scratch_variant = -1) {
        if (!sdcf_planning::active()) {
            previous_ots = previous_triples = 0;
            previous_generation = sdcf_preprocess::generation_epoch.load();
        }
        std::vector<uint64_t> in(count), out(count), expected(count), cmp_data(count);
        std::vector<uint8_t> bits(count), eq_bits(count), msb(count);
        auto fill = [&](int width) {
            for (int i = 0; i < count; ++i) {
                in[i] = scratch_variant < 0 ? share_value(i,width,query,role) :
                    (scratch_variant ? ring_mask(width) : 0);
                expected[i] = clear_value(i,width,query);
                msb[i] = role == 1 ? uint8_t(expected[i] >> (width-1)) : 0;
            }
            std::fill(out.begin(), out.end(), 0);
        };
        for (bool greater : {true,false}) {
            for (int i = 0; i < count; ++i) {
                const uint64_t left = clear_value(i,13,query), right = clear_value(i,13,query+1);
                cmp_data[i] = scratch_variant < 0 ? (role == 1 ? left : right) : scratch_variant;
                expected[i] = greater ? left > right : left < right;
            }
            aux.mill->compare(bits.data(),cmp_data.data(),count,13,greater);
            for (int i = 0; i < count; ++i) out[i] = bits[i];
            open(greater ? "compare_gt" : "compare_lt",out,expected,1,true);
        }
        for (int i = 0; i < count; ++i) {
            const uint64_t left = clear_value(i,13,query), right = i % 3 == 0 ? left : clear_value(i,13,query+1);
            cmp_data[i] = scratch_variant < 0 ? (role == 1 ? left : right) : scratch_variant;
            expected[i] = left > right;
        }
        mill_eq.compare_with_eq(bits.data(),eq_bits.data(),cmp_data.data(),count,13);
        for (int i = 0; i < count; ++i) out[i] = bits[i];
        open("compare_with_eq",out,expected,1,true);
        for (int i = 0; i < count; ++i) {
            expected[i] = i % 3 == 0;
            out[i] = eq_bits[i];
        }
        open("compare_equality",out,expected,1,true);
        eq.check_equality(bits.data(),cmp_data.data(),count,13);
        for (int i = 0; i < count; ++i) out[i] = bits[i];
        open("equality",out,expected,1,true);
        fill(13);
        aux.wrap_computation(in.data(),bits.data(),count,13);
        for (int i = 0; i < count; ++i) {
            out[i] = bits[i];
            const uint64_t alice = share_value(i,13,query,1), bob = share_value(i,13,query,2);
            expected[i] = alice + bob > ring_mask(13);
        }
        open("wrap",out,expected,1,true);
        fill(13);
        aux.MSB(in.data(),bits.data(),count,13);
        for (int i = 0; i < count; ++i) { out[i] = bits[i]; expected[i] >>= 12; }
        open("msb",out,expected,1,true);
        for (int i = 0; i < count; ++i) {
            bits[i] = (i + query + role) & 1;
            expected[i] = 1;
        }
        aux.B2A(bits.data(),out.data(),count,13);
        open("b2a",out,expected,13);
        auto divided = [&](int width, bool towards_zero, int output_width, bool signed_arithmetic) {
            for (int i = 0; i < count; ++i) {
                const uint64_t clear = clear_value(i,width,query);
                if (signed_arithmetic) {
                    const int64_t value = signed_value(clear,width);
                    int64_t result = value / 32;
                    if (!towards_zero && value % 32 < 0) --result;
                    expected[i] = uint64_t(result) & ring_mask(output_width);
                } else expected[i] = (clear >> 5) & ring_mask(output_width);
            }
        };
        fill(13); trunc.truncate(count,in.data(),out.data(),5,13,true);
        divided(13,false,13,true); open("truncate_signed",out,expected,13);
        fill(13); trunc.div_pow2(count,in.data(),out.data(),5,13,true);
        divided(13,true,13,true); open("div_towards_zero",out,expected,13);
        fill(13); trunc.truncate_and_reduce(count,in.data(),out.data(),5,13);
        divided(13,false,8,false); open("truncate_reduce",out,expected,8);
        fill(13); trunc.truncate_red_then_ext(count,in.data(),out.data(),5,13,true);
        divided(13,false,13,true); open("truncate_reduce_extend",out,expected,13);
        fill(13); xt.s_extend(count,in.data(),out.data(),13,27);
        for (int i = 0; i < count; ++i) expected[i] = uint64_t(signed_value(expected[i],13)) & ring_mask(27);
        open("sign_extend",out,expected,27);
        fill(13); xt.z_extend(count,in.data(),out.data(),13,27);
        open("zero_extend",out,expected,27);
        fill(13); trunc.truncate(count,in.data(),out.data(),5,13,false);
        divided(13,false,13,false); open("truncate_unsigned",out,expected,13);
        fill(13); trunc.div_pow2(count,in.data(),out.data(),5,13,false);
        divided(13,false,13,false); open("div_unsigned",out,expected,13);
        fill(13); trunc.truncate(count,in.data(),out.data(),5,13,true,msb.data());
        divided(13,false,13,true); open("truncate_known_msb",out,expected,13);
        fill(64); trunc.div_pow2(count,in.data(),out.data(),5,64,true);
        divided(64,true,64,true); open("div64",out,expected,64);
        fill(64); trunc.truncate(count,in.data(),out.data(),5,64,true);
        divided(64,false,64,true); open("truncate64",out,expected,64);
        fill(13); xt.s_extend(count,in.data(),out.data(),13,64);
        for (int i = 0; i < count; ++i) expected[i] = uint64_t(signed_value(expected[i],13));
        open("sign_extend64",out,expected,64);
        fill(13); xt.z_extend(count,in.data(),out.data(),13,64);
        open("zero_extend64",out,expected,64);
    }
    void print() {
        for (const auto& result : results) {
            std::cout << "CONVERSION_RESULT {\"query\":" << query << ",\"protocol\":\"" << result.first << "\",\"values\":[";
            for (size_t i = 0; i < result.second.size(); ++i) { if (i) std::cout << ','; std::cout << result.second[i]; }
            std::cout << "]}\n";
        }
    }
};

void triple_checks(TripleGenerator& generator, int role, sci::IOPack& io) {
    const std::vector<sdcf_triples::Call> calls{{14,2,false,1},{16,2,true,1},{24,3,false,3},{32,3,true,8}};
    for (const auto& call : calls) {
        std::vector<uint8_t> a(sdcf_triples::bytes(call)),b(a.size()),c(a.size());
        generator.generate(role,a.data(),b.data(),c.data(),call.count,TripleGenMethod(call.method),call.packed,call.offset);
        if (sdcf_planning::active()) continue;
        if (role == 1) { io.io->send_data(a.data(),a.size()); io.io->send_data(b.data(),b.size()); io.io->send_data(c.data(),c.size()); }
        else {
            std::vector<uint8_t> other_a(a.size()),other_b(a.size()),other_c(a.size());
            io.io->recv_data(other_a.data(),a.size()); io.io->recv_data(other_b.data(),b.size()); io.io->recv_data(other_c.data(),c.size());
            for (int i = 0; i < call.count; ++i) {
                const int pos = call.packed ? i / 8 : i, shift = call.packed ? i % 8 : 0;
                const uint8_t clear_a = ((a[pos]^other_a[pos])>>shift)&1;
                const uint8_t clear_b = ((b[pos]^other_b[pos])>>shift)&1;
                const uint8_t clear_c = ((c[pos]^other_c[pos])>>shift)&1;
                if ((clear_a & clear_b) != clear_c) throw std::runtime_error("native bit-triple reconstruction mismatch");
                if (call.method == 3 && i % (2*call.offset) < call.offset) {
                    const int paired = i+call.offset, paired_pos = call.packed ? paired/8 : paired;
                    const int paired_shift = call.packed ? paired%8 : 0;
                    if (clear_a != (((a[paired_pos]^other_a[paired_pos])>>paired_shift)&1))
                        throw std::runtime_error("native correlated bit-triple offset mismatch");
                }
            }
        }
    }
}
}

int main(int argc, char** argv) {
    try {
        const int role = std::stoi(argv[1]),port = std::stoi(argv[2]);
        const std::string mode = argv[3];
        auto setup = sdcf_measure::point();
        sci::IOPack io(role,port);
        sci::OTPack ot(&io,role);
        TripleGenerator generator(role,&io,&ot);
        sdcf_planning::register_pack(&ot,0);
        int inactive_context = 0;
        sdcf_planning::register_triples(&inactive_context,1);
        sdcf_measure::report("setup",-1,setup);
        const bool conversions = mode.find("conversions") == 0;
        const bool prepared = mode != "conversions_default";
        for (int query = 0; query < 2; ++query) {
            Suite suite(role,query,io,ot);
            auto offline = sdcf_measure::point();
            sdcf_preprocess::begin_prepare(query,prepared);
            if (prepared) {
                auto make_plan = [&](int variant) {
                    sdcf_planning::begin();
                    if (conversions) suite.run(variant);
                    else if (mode == "triples_prepared") triple_checks(generator,role,io);
                    else {
                        std::array<uint8_t,16> a{},b{},c{};
                        generator.generate(role,a.data(),b.data(),c.data(),16,mode=="unsafe" ? Ideal : _16KKOT_to_4OT);
                    }
                    return sdcf_planning::end();
                };
                const auto before = sdcf_measure::point();
                const std::string first = make_plan(0),second = make_plan(1);
                const auto after = sdcf_measure::point();
                if (first != second || before.native_epoch != after.native_epoch ||
                    before.sent != after.sent || before.received != after.received)
                    throw std::runtime_error("public plan changed randomness or communication");
                sdcf_planning::compare_plan(io.io,role,second);
                std::cout << "PLAN_RESULT {\"query\":" << query << ",\"secret_independent\":true}\n";
                sdcf_planning::prepare_ots(query);
                std::vector<sdcf_triples::Call> calls;
                for (const auto& call : sdcf_planning::triple_calls(&ot))
                    calls.push_back({call.count,call.method,call.packed,call.offset});
                sdcf_triples::plan(&ot,query,0,calls);
                sdcf_triples::plan(&inactive_context,query,1,{});
                sdcf_triples::prepare(&generator);
                if (mode == "unsafe_prepare") {
                    std::array<uint8_t,16> a{},b{},c{};
                    generator.generate(role,a.data(),b.data(),c.data(),16,Ideal);
                }
                sdcf_triples::seal(query);
            }
            sdcf_preprocess::seal_prepare(query);
            io.io->sync();
            sdcf_preprocess::material_ready(query);
            sdcf_measure::report("offline",query,offline);
            auto input = sdcf_measure::point();
            sdcf_preprocess::begin_input(query);
            sdcf_measure::report("input",query,input);
            io.io->sync();
            auto online = sdcf_measure::point();
            sdcf_preprocess::enter_online(query);
            if (conversions) suite.run();
            else if (mode == "triples_prepared") triple_checks(generator,role,io);
            else if (mode == "online_generation") sdcf_preprocess::generation_begin(&ot);
            else if (mode != "incomplete") {
                std::array<uint8_t,16> a{},b{},c{};
                generator.generate(role,a.data(),b.data(),c.data(),mode=="mismatch" ? 14 : 16,_16KKOT_to_4OT);
                if (mode == "repeat") generator.generate(role,a.data(),b.data(),c.data(),16,_16KKOT_to_4OT);
            }
            sdcf_triples::balance(query);
            sdcf_preprocess::leave_online(query);
            io.io->sync();
            sdcf_measure::report("online",query,online);
            auto output = sdcf_measure::point();
            suite.print();
            sdcf_preprocess::finish_query(query);
            sdcf_measure::report("output",query,output);
            if (!conversions) std::cout << "TRIPLE_RESULT {\"query\":" << query << ",\"correct\":true}\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "TRIPLE_ERROR " << error.what() << '\n';
        return 2;
    }
}
