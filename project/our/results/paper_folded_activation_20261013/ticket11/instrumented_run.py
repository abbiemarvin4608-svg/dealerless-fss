#!/usr/bin/env python3
"""Add LUT union scopes during the runtime's existing post-party trace analysis."""
import runpy
import sys

from benchmark import RUNTIME

sys.path.insert(0, str(RUNTIME))
from scope_rounds import with_lut_unions

if __name__ == "__main__":
    with with_lut_unions():
        runpy.run_path(str(RUNTIME / "run.py"), run_name="__main__")
