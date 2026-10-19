"""Independent integer reference for the paper's f12/q10 correction LUT.

The table is calculated with Decimal rather than protocol code or its double-
precision table builder. All signed divisions below round toward minus infinity.
"""
from decimal import Decimal, ROUND_FLOOR, localcontext
from functools import lru_cache
import hashlib
import struct


TABLE_PRECISION = 80
TABLE_ENTRIES = 1024


@lru_cache(maxsize=None)
def correction_table(precision=TABLE_PRECISION):
    with localcontext() as context:
        context.prec = precision
        return tuple(
            int((Decimal(4096) / (1 + (Decimal(j) / 64).exp()))
                .to_integral_value(rounding=ROUND_FLOOR))
            for j in range(TABLE_ENTRIES)
        )


def table_version():
    encoded = struct.pack("<1024H", *correction_table())
    return "folded_sigma_f12_q10-u16le-sha256:" + hashlib.sha256(encoded).hexdigest()


def sigma_q12(x):
    x = int(x)
    magnitude = abs(x)
    correction = correction_table()[min(magnitude, 65535) // 64]
    return 4096 - correction if x >= 0 else correction


def activation(x_q7, kind):
    x_q7 = int(x_q7)
    if not -32768 <= x_q7 <= 32767:
        raise ValueError("folded native input must be signed int16/Q7")
    if kind == "sigmoid":
        return 4 * sigma_q12(32 * x_q7)
    if kind == "tanh":
        return (2 * sigma_q12(64 * x_q7) - 4096) // 32
    raise ValueError("activation kind must be sigmoid or tanh")
