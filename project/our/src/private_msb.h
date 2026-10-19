#pragma once
#include "gen.h"
#include <cstdint>
#include <vector>
namespace segfss {
// These move-only materials cannot accidentally be copied and reused.
struct MsbMaterial {
    Peer* owner_peer=nullptr;
    int width=0, count=0, nseg=0;
    std::vector<uint64_t> mask_shares;
    std::vector<u8> mask_sign_shares;
    std::vector<CggmKey> keys;
    BooleanAndPreprocessing scan;
    bool consumed=false;
    double mask_conversion_ms=0,keygen_ms=0,scan_preprocessing_ms=0;
    double masked_input_open_ms=0,evaluation_ms=0,scan_ms=0;
    uint64_t generation_tiles=0, generation_peak_working_bytes=0;
    uint64_t generation_workspace_capacity_bytes=0;
    MsbMaterial()=default;
    MsbMaterial(const MsbMaterial&)=delete;
    MsbMaterial& operator=(const MsbMaterial&)=delete;
    MsbMaterial(MsbMaterial&&)=default;
    MsbMaterial& operator=(MsbMaterial&&)=default;
    std::size_t capacityBytes() const;
};
MsbMaterial privateMsbOffline(int role,int count,int width,int nseg,Peer* player);
void validatePrivateMsbMaterial(int role,const MsbMaterial& material,
    int count,int width,int nseg,Peer* player);
void privateMsbOnline(int role,const uint64_t* shares,u8* output,
                      MsbMaterial& material,Peer* player);
}
