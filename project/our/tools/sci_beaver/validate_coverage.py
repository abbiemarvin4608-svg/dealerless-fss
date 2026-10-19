#!/usr/bin/env python3
"""Validate per-query multiplication material and public-scalar coverage."""
import argparse
import json
from pathlib import Path


def records(path, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in path.read_text().splitlines()
            if line.startswith(prefix + " ")]


def validate(directory):
    result = json.loads((directory / "result.json").read_text())
    config = result["config"]
    if result["status"] != "complete" or config["diagnostic"]:
        raise ValueError("coverage requires a completed secure run")
    schedule = json.loads((directory / "schedule.json").read_text())
    secret = {row["index"]: row for row in schedule if row["function"] in ("MatMul", "MulCir")}
    public = {row["index"]: row for row in schedule if row["function"] == "ScalarMul"}
    parties = []
    for role in (1, 2):
        path = directory / f"party{role}.log"
        offline = records(path, "BEAVER_OFFLINE")
        online = records(path, "BEAVER_ONLINE")
        scalars = records(path, "PUBLIC_SCALARMUL")
        summaries = []
        for query in range(config["queries"] + config["warmup_queries"]):
            prep = {row["index"]: row for row in offline if row["query"] == query}
            used = {row["index"]: row for row in online if row["query"] == query}
            local = {row["index"]: row for row in scalars if row["query"] == query}
            if set(prep) != set(secret) or set(used) != set(secret) or set(local) != set(public):
                raise ValueError("scheduled multiplication coverage differs")
            if sum(row["query"] == query for row in offline) != len(secret) or sum(
                    row["query"] == query for row in online) != len(secret):
                raise ValueError("duplicate multiplication material or consumption")
            matrix_triples = vector_triples = opened_values = 0
            for index, row in prep.items():
                executed = used[index]
                if not row["fresh"] or not row["dealerless"] or row["activation_triples"] != 0:
                    raise ValueError("multiplication material accounting includes another protocol")
                if not executed["consumed"] or executed["opening_groups"] != 1:
                    raise ValueError("multiplication material was not consumed once")
                if row["ring_bits"] != executed["ring_bits"]:
                    raise ValueError("multiplication ring differs between stages")
                if secret[index]["function"] == "MatMul":
                    expected = (row["rows"] * row["inner"], row["inner"] * row["cols"])
                    matrix_triples += row["triples"]
                else:
                    expected = (row["elements"], row["elements"])
                    vector_triples += row["triples"]
                if (executed["E_values"], executed["F_values"]) != expected:
                    raise ValueError("E/F opening dimensions differ from material")
                opened_values += sum(expected)
            for row in local.values():
                if row["multiplication"] != "local" or row["multiplication_triples"] != 0:
                    raise ValueError("public model scalar incorrectly charged as private multiplication")
            summaries.append({"query": query, "fresh_matrix_triples": matrix_triples,
                              "fresh_elementwise_triples": vector_triples,
                              "private_scalar_materials": 0, "public_scalar_calls": len(local),
                              "public_scalar_multiplication_triples": 0,
                              "E_F_opened_values": opened_values,
                              "batched_opening_groups": len(used)})
        parties.append(summaries)
    if parties[0] != parties[1]:
        raise ValueError("party multiplication coverage differs")
    return {"status": "passed", "queries": parties[0],
            "scope": "multiplication materials only; native activation OT remains online and is not counted as multiplication preprocessing",
            "binary_sha256": config["binary_sha256"], "fixture_identity": config["fixture_identity"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.run)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
