#!/usr/bin/env python3
"""Build the SDCF runtime with Beaver multiplication and native SCI conversions."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"segmented_dcf_runtime"))
from build import build,digest


def build_beaver(out,sci,generated,jobs=2,extra_sources=(),extra_libraries=(),conversion_overrides=(),compiler_defines=()):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    original=(Path(sci)/"src/library_fixed.cpp").read_text()
    needle="""  multArr[tid]->matrix_multiplication(I, K, J, A, B, C, bwA, bwB, bwA + bwB,
                                      true, true, true, mode);"""
    if original.count(needle)!=1:raise ValueError("SCI MatMul_thread replacement point changed")
    replacement="""  if (!sdcf_beaver::matmul(tid, A, B, C, I, K, J, bwA, bwB)) {
"""+needle+"\n  }"
    mulcir_needle="""  multArr[tid]->hadamard_product(dim, A, B, C, bwA, bwB, bwA + bwB, true, true,
                                 MultMode::None);"""
    if original.count(mulcir_needle)!=1:raise ValueError("SCI MulCir_thread replacement point changed")
    mulcir_replacement="""  if (!sdcf_beaver::mulcir(tid, A, B, C, dim, bwA, bwB, bwTemp, bwC,
                             (int64_t(1) << shrA), (int64_t(1) << shrB),
                             (int64_t(1) << demote))) {
"""+mulcir_needle+"\n  }"
    patched=out/"library_fixed_beaver.cpp"
    text=original.replace(needle,replacement,1).replace(mulcir_needle,mulcir_replacement,1)
    patched.write_text('#include "adapter.h"\n'+text)
    binary=build(out,sci,generated,jobs,HERE/"adapter.h",
                 [HERE/"beaver.cpp",HERE/"adapter.cpp"]+list(extra_sources),extra_libraries,
                 ["library_fixed.cpp="+str(patched)]+list(conversion_overrides),compiler_defines)
    manifest_path=out/"runtime_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    for header in [HERE/"beaver.h",HERE/"adapter.h",HERE.parents[1]/"src/sci_numeric_oracle.h"]:
        manifest["sources"][str(header)]=digest(header)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    return binary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--sci",type=Path,default=Path("/home/jiang/EzPC/SCI"))
    p.add_argument("--generated",type=Path,required=True)
    p.add_argument("--jobs",type=int,default=2)
    a=p.parse_args();print(build_beaver(a.out,a.sci,a.generated,a.jobs))
