#!/usr/bin/env python3
"""Build this experiment's runtime against its new shared bridge."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
spec = importlib.util.spec_from_file_location(
    "online_conversion_builder", REPO / "project/our/tools/sci_conversion/build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

if __name__ == "__main__":
    print(builder.build(
        HERE / "runtime", Path("/home/jiang/EzPC/SCI"),
        REPO / "project/our/results/ticket07_hybrid_full/sources/sirnn_fixed.cpp",
        HERE / "core/libsegfss_sci_bridge.so", jobs=4,
    ))
