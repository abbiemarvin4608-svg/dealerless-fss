#include "beaver.h"
#include "BuildingBlocks/value-extension.h"
#include "BuildingBlocks/truncation.h"
#include "../../src/sci_numeric_oracle.h"
#include "../../src/ring_packing.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <random>

using segfss::reference::mask;
using segfss::reference::signedValue;

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) return 2;
    const int role = std::stoi(argv[1]);
    sci::IOPack io(role, std::stoi(argv[2]), "127.0.0.1");
    sci::OTPack ot(&io, role);
    LinearOT linear(role, &io, &ot);
    if (argc == 4 && std::string(argv[3]) == "packed") {
        setenv("SEGFSS_PACK_RING", "1", 1);
        for (int bits : {1, 7, 9, 13, 32, 64}) {
            auto material = sdcf_beaver::prepare_matmul(role, linear, 1, 3, 1, bits, 1);
            std::vector<uint64_t> lhs = role == 1 ?
                std::vector<uint64_t>{0x123, 0x1abc, 0x77} : std::vector<uint64_t>(3);
            std::vector<uint64_t> rhs = role == 1 ?
                std::vector<uint64_t>{3, 5, 7} : std::vector<uint64_t>(3);
            for (auto& value : lhs) value &= mask(bits);
            for (auto& value : rhs) value &= mask(bits);
            const auto before = io.io->counter;
            auto result = sdcf_beaver::multiply_matmul(role, io, linear, lhs, rhs, material);
            const auto sent = io.io->counter - before;
            if (sent != segfss::kRingPackingHeaderBytes + (6 * bits + 7) / 8)
                throw std::runtime_error("packed Beaver opening byte count mismatch");
            uint64_t remote = 0;
            if (role == 1) io.io->send_data(result.data(), sizeof(uint64_t));
            else {
                io.io->recv_data(&remote, sizeof(remote));
                if (((result.front() + remote) & mask(bits)) != (uint64_t(35926) & mask(bits)))
                    throw std::runtime_error("packed Beaver literal product mismatch");
            }
            std::cout << "PACKED_MATMUL ring=" << bits << " sent_bytes=" << sent
                      << " status=passed\n";
            auto element_material = sdcf_beaver::prepare_mulcir(role, linear, 3, bits);
            const auto element_before = io.io->counter;
            auto products = sdcf_beaver::multiply_mulcir(role, io, linear, lhs, rhs, element_material);
            const auto element_sent = io.io->counter - element_before;
            if (element_sent != segfss::kRingPackingHeaderBytes + (6 * bits + 7) / 8)
                throw std::runtime_error("packed element opening byte count mismatch");
            std::vector<uint64_t> remote_products(products.size());
            if (role == 1) io.io->send_data(products.data(), products.size() * sizeof(uint64_t));
            else {
                io.io->recv_data(remote_products.data(), remote_products.size() * sizeof(uint64_t));
                const uint64_t expected[] = {873, 34220, 833};
                for (std::size_t i = 0; i < products.size(); ++i)
                    if (((products[i] + remote_products[i]) & mask(bits)) != (expected[i] & mask(bits)))
                        throw std::runtime_error("packed element literal product mismatch");
            }
            std::cout << "PACKED_MULCIR ring=" << bits << " sent_bytes=" << element_sent
                      << " status=passed\n";
        }
        return 0;
    }
    uint64_t values_checked = 0;
    auto sample_value=[](size_t i, int width, std::mt19937_64& rng) {
        const auto sign=uint64_t{1}<<(width-1);
        const uint64_t edges[]={0,1,mask(width),sign,sign-1,sign+1,3,mask(width)-2};
        return (i%3 ? rng() : edges[(i/3)%8]) & mask(width);
    };
    for (int tile : {1, 8, 32, 128}) for (int bits : {8, 32, 64}) {
        const int rows=9,inner=3,cols=4;
        auto material=sdcf_beaver::prepare_matmul(role,linear,rows,inner,cols,bits,tile);
        if(material.offline_tile_rows!=tile || material.offline_tiles!=uint64_t((rows+tile-1)/tile) ||
           material.offline_tail_rows!=(rows-1)%tile+1 || !material.offline_working_bytes ||
           material.offline_resident_bytes!=(uint64_t(rows)*inner+inner*cols+rows*cols)*sizeof(uint64_t))
            throw std::runtime_error("public row tile accounting differs from the matrix material");
        std::vector<uint64_t> clear_a(rows*inner),clear_b(inner*cols),a(clear_a.size()),b(clear_b.size());
        for(size_t i=0;i<a.size();++i) { clear_a[i]=i%2?mask(bits):i+1;a[i]=role==1?clear_a[i]:0; }
        for(size_t i=0;i<b.size();++i) { clear_b[i]=i%3?i+2:mask(bits)-1;b[i]=role==1?clear_b[i]:0; }
        auto got=sdcf_beaver::multiply_matmul(role,io,linear,a,b,material);
        std::vector<uint64_t> remote(got.size());
        if(role==1) io.io->send_data(got.data(),got.size()*sizeof(uint64_t));
        else {
            io.io->recv_data(remote.data(),remote.size()*sizeof(uint64_t));
            for(int row=0;row<rows;++row) for(int col=0;col<cols;++col) {
                unsigned __int128 expected=0;
                for(int k=0;k<inner;++k) expected+=static_cast<unsigned __int128>(clear_a[row*inner+k])*clear_b[k*cols+col];
                if(((got[row*cols+col]+remote[row*cols+col])&mask(bits))!=(uint64_t(expected)&mask(bits)))
                    throw std::runtime_error("row tile product differs from the independent ring oracle");
            }
        }
        values_checked+=got.size();
    }
    if(argc==4) {
        std::cout<<"TILE_TEST tiles=1,8,32,128 rings=8,32,64 values="<<values_checked<<" status=passed\n";
        return 0;
    }
    // Reconstruct literal wraparound products through the same two-party API,
    // including output storage which initially aliases either operand.
    for (bool algebra : {false, true}) for (bool mapped_layout : {false, true}) {
        sdcf_beaver::MatmulOptions options{algebra, mapped_layout};
        sdcf_beaver::MatmulWorkspace workspace;
        for (int bits : {8, 32, 64}) for (int alias : {0, 1, 2, 3, 4, 5}) {
            auto material=sdcf_beaver::prepare_matmul(role,linear,2,2,3,bits,32,options);
            std::vector<uint64_t> a{mask(bits),2,3,mask(bits)-1};
            std::vector<uint64_t> b{4,mask(bits),6,7,8,mask(bits)-2};
            if(role==2) { std::fill(a.begin(),a.end(),0);std::fill(b.begin(),b.end(),0); }
            std::vector<uint64_t> output;
            auto& target=alias==1?a:alias==2?b:alias==3?material.u:
                         alias==4?material.v:alias==5?material.w:output;
            sdcf_beaver::multiply_matmul(role,io,linear,a,b,material,target,&workspace);
            std::vector<uint64_t> remote(target.size());
            if(role==1) io.io->send_data(target.data(),target.size()*sizeof(uint64_t));
            else {
                io.io->recv_data(remote.data(),remote.size()*sizeof(uint64_t));
                const uint64_t expected[]={10,17,mask(bits)-11,mask(bits)-1,mask(bits)-18,24};
                for(size_t i=0;i<target.size();++i)
                    if(((target[i]+remote[i])&mask(bits))!=(expected[i]&mask(bits)))
                        throw std::runtime_error("configured matrix product or operand alias disagrees with literal ring oracle");
            }
            if(material.opened_values!=10 || !material.consumed)
                throw std::runtime_error("configured matrix opening/consumption contract failed");
            values_checked+=target.size();
        }
        std::cout<<"MATMUL_CONFIG_TEST algebra="<<algebra<<" mapped_layout="<<mapped_layout
                 <<" rings=8,32,64 aliases=none,lhs,rhs,u,v,w status=passed\n";
    }
    for (int tile : {1,8,32,128})
    for (const auto& options : std::vector<sdcf_beaver::MatmulOptions>{
            {false,false},{true,false},{false,true},{true,true}})
    for (const auto& shape : std::vector<std::array<int, 6>>{
            {2,3,4,4,3,2}, {4,2,3,1,2,0}, {2,1,2,32,32,1},
            {128,32,16,16,8,1}, {128,16,100,16,16,11},
            {128,100,35,16,8,7}, {128,35,100,8,16,0},
            {128,100,30,16,8,9}}) {
        if(tile!=32 && (options.algebra || options.mapped_layout)) continue;
        const int rows=shape[0], inner=shape[1], cols=shape[2], a_bits=shape[3], b_bits=shape[4], shift=shape[5];
        const int depth=sdcf_beaver::accumulation_bits(inner), bits=a_bits+b_bits+depth;
        auto material=sdcf_beaver::prepare_matmul(role, linear, rows, inner, cols, bits,tile,options);
        std::mt19937_64 rng(20261009+inner+cols);
        std::vector<uint64_t> a(rows*inner), b(inner*cols), clear_a(a.size()), clear_b(b.size()), aw(a.size()), bw(b.size());
        auto input_value=[&](size_t i, int width) {
            const auto sign=uint64_t{1}<<(width-1);
            const uint64_t edges[]={0,1,mask(width),sign,sign-1,sign+1,3,mask(width)-2};
            return (i%3 ? rng() : edges[(i/3)%8]) & mask(width);
        };
        for (size_t i=0;i<a.size();++i) {
            clear_a[i]=input_value(i,a_bits);
            const auto share=rng()&mask(a_bits);
            a[i]=role==1?share:(clear_a[i]-share)&mask(a_bits);
        }
        for (size_t i=0;i<b.size();++i) {
            clear_b[i]=input_value(i,b_bits);
            b[i]=role==1?clear_b[i]:0;
            bw[i]=role==1?uint64_t(signedValue(clear_b[i],b_bits))&mask(bits):0;
        }
        linear.xt->s_extend(a.size(),a.data(),aw.data(),a_bits,bits);
        bool mismatch_refused=false;
        try { sdcf_beaver::multiply_matmul(role,io,linear,std::vector<uint64_t>{},bw,material); }
        catch(const std::invalid_argument&) { mismatch_refused=true; }
        if(!mismatch_refused || material.consumed) throw std::runtime_error("matrix shape preflight failed");
        auto got=sdcf_beaver::multiply_matmul(role,io,linear,aw,bw,material);
        bool refused=false;
        try { sdcf_beaver::multiply_matmul(role,io,linear,aw,bw,material); }
        catch(const std::logic_error&) { refused=true; }
        if(!refused) throw std::runtime_error("consumed matrix material accepted");
        std::vector<uint64_t> remote(got.size()), actual(got.size()), native(got.size()), final(got.size()), native_final(got.size());
        linear.matrix_multiplication(rows,inner,cols,a.data(),b.data(),native.data(),a_bits,b_bits,a_bits+b_bits,true,true,true,MultMode::Alice_has_B);
        linear.trunc->truncate_and_reduce(got.size(),got.data(),actual.data(),depth,bits);
        linear.trunc->div_pow2(actual.size(),actual.data(),final.data(),shift,a_bits+b_bits,true);
        linear.trunc->div_pow2(native.size(),native.data(),native_final.data(),shift,a_bits+b_bits,true);
        for(auto* vector : {&got,&actual,&native,&final,&native_final}) {
            if(role==1) io.io->send_data(vector->data(),vector->size()*sizeof(uint64_t));
            else {
                io.io->recv_data(remote.data(),remote.size()*sizeof(uint64_t));
                for(size_t i=0;i<vector->size();++i) (*vector)[i]+=remote[i];
            }
        }
        if(role==2) for(int i=0;i<rows;++i) for(int j=0;j<cols;++j) {
            __int128 sum=0;
            for(int k=0;k<inner;++k) sum+=signedValue(clear_a[i*inner+k],a_bits)*signedValue(clear_b[k*cols+j],b_bits);
            const auto expected=uint64_t(sum)&mask(bits);
            const int index=i*cols+j;
            if((got[index]&mask(bits))!=expected ||
               (actual[index]&mask(a_bits+b_bits))!=(native[index]&mask(a_bits+b_bits)) ||
               (final[index]&mask(a_bits+b_bits))!=(native_final[index]&mask(a_bits+b_bits)))
                throw std::runtime_error("Beaver matrix disagrees with raw integer or native SCI");
        }
        values_checked+=got.size();
        std::cout<<"MATMUL_TEST rows="<<rows<<" inner="<<inner<<" cols="<<cols<<" ring="<<bits
                 <<" algebra="<<options.algebra<<" mapped_layout="<<options.mapped_layout
                 <<" tile_rows="<<tile
                 <<" offline_tiles="<<material.offline_tiles<<" opened_values="<<material.opened_values<<" status=passed\n";
    }

    // MulCir uses the same mixed-width product ring as native SCI, but opens
    // batched E/F vectors once and consumes a separate vector triple.
    for (const auto& shape : std::vector<std::array<int, 6>>{
            {1,16,16,16,14,16}, {4,32,8,16,7,8}, {128,100,16,16,7,16}, {8,8,32,32,31,32}}) {
        const int rows=shape[0], cols=shape[1], a_bits=shape[2], b_bits=shape[3], shift=shape[4], out_bits=shape[5];
        const int elements=rows*cols, bits=a_bits+b_bits;
        auto material=sdcf_beaver::prepare_mulcir(role, linear, elements, bits);
        std::mt19937_64 rng(20261017+elements);
        std::vector<uint64_t> clear_a(elements),clear_b(elements),a(elements),b(elements);
        for (int i=0;i<elements;++i) {
            clear_a[i]=sample_value(i,a_bits,rng); clear_b[i]=sample_value(i+3,b_bits,rng);
            const auto sa=rng()&mask(a_bits), sb=rng()&mask(b_bits);
            a[i]=role==1?sa:(clear_a[i]-sa)&mask(a_bits);
            b[i]=role==1?sb:(clear_b[i]-sb)&mask(b_bits);
        }
        std::vector<uint64_t> aw(elements),bw(elements);
        linear.xt->s_extend(elements,a.data(),aw.data(),a_bits,bits);
        linear.xt->s_extend(elements,b.data(),bw.data(),b_bits,bits);
        auto got=sdcf_beaver::multiply_mulcir(role,io,linear,aw,bw,material);
        bool refused=false;
        try { sdcf_beaver::multiply_mulcir(role,io,linear,aw,bw,material); }
        catch(const std::logic_error&) { refused=true; }
        if(!refused || material.opened_values!=uint64_t(2*elements))
            throw std::runtime_error("MulCir material consumption failed");
        std::vector<uint64_t> native(elements),remote(got.size()),final(elements),native_final(elements);
        linear.hadamard_product(elements,a.data(),b.data(),native.data(),a_bits,b_bits,bits,true,true,MultMode::None);
        linear.trunc->div_pow2(elements,got.data(),final.data(),shift,bits,true);
        linear.trunc->div_pow2(elements,native.data(),native_final.data(),shift,bits,true);
        for(auto* vector : {&got,&native,&final,&native_final}) {
            if(role==1) io.io->send_data(vector->data(),vector->size()*sizeof(uint64_t));
            else {
                io.io->recv_data(remote.data(),remote.size()*sizeof(uint64_t));
                for(size_t i=0;i<vector->size();++i) (*vector)[i]=((*vector)[i]+remote[i])&mask(bits);
            }
        }
        if(role==2) for(int i=0;i<elements;++i) {
            const auto expected=uint64_t(__int128(signedValue(clear_a[i],a_bits))*signedValue(clear_b[i],b_bits))&mask(bits);
            if(got[i]!=expected || native[i]!=expected ||
               (final[i]&mask(out_bits))!=(native_final[i]&mask(out_bits)))
                throw std::runtime_error("Beaver MulCir disagrees with integer or native SCI conversion");
        }
        values_checked+=got.size();
        std::cout<<"MULCIR_TEST elements="<<elements<<" ring="<<bits<<" shift="<<shift
                 <<" output_bits="<<out_bits<<" opened_values="<<material.opened_values<<" multiplication_triples="<<elements<<" status=passed\n";
    }

    // Private ScalarMul is tested separately from Google's public zeta/nu
    // ScalarMul calls; this path consumes a scalar Beaver triple and opens one
    // scalar E plus a vector F.
    for (const auto& shape : std::vector<std::array<int, 6>>{
            {1,16,16,16,14,16}, {4,32,8,16,7,8}, {128,100,16,16,14,16}, {8,8,32,32,31,32}}) {
        const int elements=shape[0]*shape[1], scalar_bits=shape[2], rhs_bits=shape[3], bits=scalar_bits+rhs_bits;
        const int shift=shape[4],out_bits=shape[5];
        auto material=sdcf_beaver::prepare_scalarmul(role, linear, elements, bits);
        std::mt19937_64 rng(20261029+elements);
        const auto clear_scalar=mask(scalar_bits)-2;
        const auto scalar_share=rng()&mask(scalar_bits);
        std::vector<uint64_t> clear_rhs(elements),rhs(elements),scalar(1);
        scalar[0]=role==1?scalar_share:(clear_scalar-scalar_share)&mask(scalar_bits);
        for(int i=0;i<elements;++i) {
            clear_rhs[i]=sample_value(i+7,rhs_bits,rng);
            const auto sr=rng()&mask(rhs_bits);
            rhs[i]=role==1?sr:(clear_rhs[i]-sr)&mask(rhs_bits);
        }
        std::vector<uint64_t> scalar_wide(1),rhs_wide(elements);
        linear.xt->s_extend(1,scalar.data(),scalar_wide.data(),scalar_bits,bits);
        linear.xt->s_extend(elements,rhs.data(),rhs_wide.data(),rhs_bits,bits);
        auto got=sdcf_beaver::multiply_scalarmul(role,io,linear,scalar_wide,rhs_wide,material);
        bool refused=false;
        try { sdcf_beaver::multiply_scalarmul(role,io,linear,scalar_wide,rhs_wide,material); }
        catch(const std::logic_error&) { refused=true; }
        if(!refused || material.opened_values!=uint64_t(elements+1))
            throw std::runtime_error("ScalarMul material consumption failed");
        std::vector<uint64_t> repeated(elements,scalar[0]),native(elements),final(elements),native_final(elements),remote(got.size());
        linear.hadamard_product(elements,repeated.data(),rhs.data(),native.data(),scalar_bits,rhs_bits,bits,true,true,MultMode::None);
        linear.trunc->div_pow2(elements,got.data(),final.data(),shift,bits,true);
        linear.trunc->div_pow2(elements,native.data(),native_final.data(),shift,bits,true);
        for(auto* vector:{&got,&native,&final,&native_final}) {
            if(role==1) io.io->send_data(vector->data(),vector->size()*sizeof(uint64_t));
            else {
                io.io->recv_data(remote.data(),remote.size()*sizeof(uint64_t));
                for(size_t i=0;i<vector->size();++i) (*vector)[i]+=remote[i];
            }
        }
        if(role==2) for(int i=0;i<elements;++i) {
            const auto expected=uint64_t(__int128(signedValue(clear_scalar,scalar_bits))*signedValue(clear_rhs[i],rhs_bits))&mask(bits);
            if((got[i]&mask(bits))!=expected || (native[i]&mask(bits))!=expected ||
               (final[i]&mask(out_bits))!=(native_final[i]&mask(out_bits)))
                throw std::runtime_error("Beaver private ScalarMul disagrees with integer or native SCI conversion");
        }
        values_checked+=got.size();
        std::cout<<"SCALARMUL_TEST elements="<<elements<<" ring="<<bits<<" shift="<<shift
                 <<" output_bits="<<out_bits<<" opened_values="<<material.opened_values
                 <<" multiplication_scalar_vector_materials=1 activation_triples=0 status=passed\n";
    }
    std::cout<<"SUMMARY values="<<values_checked<<" status=passed\n";
}
