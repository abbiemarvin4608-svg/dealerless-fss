#!/usr/bin/env python3
"""Build one native graph binary with SCI/Beaver/full-tree/segmented controls."""
import argparse
import importlib.util
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("beaver_builder", HERE.parent / "sci_beaver/build.py")
beaver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beaver)


def patch_conversions(out, sci):
    result = []
    hooks = {
        "BuildingBlocks/value-extension.cpp": {
            "s_extend": "sdcf_conversion::convert(party,inA,outB,dim,bwA,bwB,0,3,true)",
            "z_extend": "sdcf_conversion::convert(party,inA,outB,dim,bwA,bwB,0,3,false)",
        },
        "BuildingBlocks/truncation.cpp": {
            "div_pow2": "sdcf_conversion::convert(party,inA,outB,dim,bw,bw,shift,0,signed_arithmetic)",
            "truncate": "sdcf_conversion::convert(party,inA,outB,dim,bw,bw,shift,1,signed_arithmetic)",
            "truncate_and_reduce": "sdcf_conversion::convert(party,inA,outB,dim,bw,bw-shift,shift,2,true)",
        },
    }
    for relative, functions in hooks.items():
        text = (sci / "src" / relative).read_text()
        for function, expression in functions.items():
            owner = "XTProtocol" if "extension" in relative else "Truncation"
            needle = "void " + owner + "::" + function + "("
            if text.count(needle) != 1:
                raise ValueError("SCI conversion replacement point changed: " + needle)
            index = text.index("{", text.index(needle)) + 1
            text = text[:index] + "\n  if (" + expression + ") return;" + text[index:]
        target = out / Path(relative).name
        target.write_text('#include "' + str(HERE / "hook.h") + '"\n' + text)
        result.append(relative + "=" + str(target))
    return result


def build(out, sci, generated, bridge, jobs=2):
    out, sci = Path(out).resolve(), Path(sci).resolve()
    if not re.search(r"^#define DIV_RESCALING\b", (sci / "src/defines.h").read_text(), re.M):
        raise ValueError("frozen conversion plan requires SCI DIV_RESCALING=1")
    out.mkdir(parents=True, exist_ok=True)
    overrides = patch_conversions(out, sci)
    return beaver.build_beaver(out, sci, generated, jobs,
        extra_sources=[HERE / "hook.cpp"], extra_libraries=[Path(bridge).resolve()],
        conversion_overrides=overrides, compiler_defines=["SDCF_CONVERSION_RUNTIME=1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sci", type=Path, default=Path("/home/jiang/EzPC/SCI"))
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    print(build(args.out, args.sci, args.generated, args.bridge, args.jobs))
