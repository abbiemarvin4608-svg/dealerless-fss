"""Request LUT unions from the bounded-memory causal trace analyzer."""
from contextlib import contextmanager

import round_trace

LUT_UNION = dict(label="ticket11_lut_union", phase="online", source_label="activation_stage",
                 stages=list(range(8, 16)), group_by=["kind"])


@contextmanager
def with_lut_unions():
    original = round_trace.analyze_pair

    def augmented(server, client, expected_phases=None, scope_groups=None):
        groups = list(scope_groups or ())
        if LUT_UNION not in groups:
            groups.append(LUT_UNION)
        return original(server, client, expected_phases=expected_phases, scope_groups=groups)

    round_trace.analyze_pair = augmented
    try:
        yield
    finally:
        round_trace.analyze_pair = original


def replay(server, client, parties):
    result = round_trace.analyze_pair(server, client, expected_phases=parties, scope_groups=[LUT_UNION])
    if result["status"] != "passed":
        raise ValueError("raw trace replay failed: " + result.get("reason", "unknown"))
    extra = [row for row in result["scopes"] if row["label"] == "ticket11_lut_union"]
    result["scopes"] = [row for row in result["scopes"] if row["label"] != "ticket11_lut_union"]
    return result, extra
