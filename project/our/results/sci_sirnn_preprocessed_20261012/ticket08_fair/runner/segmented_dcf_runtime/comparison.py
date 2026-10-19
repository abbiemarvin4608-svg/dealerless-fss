#!/usr/bin/env python3
"""Summarize matching secure workloads, including failed backend attempts."""
import argparse
import json
from pathlib import Path
import statistics


def compare_predictions(native, candidate, batch, start, queries):
    def parse(log):
        values = {}
        for line in log.splitlines():
            if line.startswith("PRED "):
                _, query, sample, prediction = line.split()
                key = (int(query), int(sample))
                if key in values:
                    raise ValueError("duplicate prediction coverage")
                values[key] = int(prediction)
        expected = {(query, sample) for query in range(queries) for sample in range(start, start + batch)}
        if set(values) != expected:
            raise ValueError("incomplete prediction coverage")
        return values
    reference, actual = parse(native), parse(candidate)
    differences = [(key, reference[key], actual[key]) for key in sorted(reference) if reference[key] != actual[key]]
    first = differences[0] if differences else None
    return {"status": "mismatch" if differences else "passed", "total": len(reference),
            "matched": len(reference) - len(differences),
            "first_difference": {"query": first[0][0], "sample": first[0][1], "SCI": first[1], "candidate": first[2]} if first else None,
            "scope": "same fixed batch; no full evaluation-set or Top-1 evidence"}


def distribution(values):
    if not values or any(value is None for value in values):
        return {"median": None, "min": None, "max": None, "pstdev": None, "values": values}
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "pstdev": statistics.pstdev(values), "values": values}


def comparison_contract(config):
    contract = {key: config.get(key) for key in
                ("batch", "steps", "start", "network", "cpus", "queries", "warmup_queries",
                 "application_threads_per_party", "fixture_files", "sample_ids", "binary_sha256",
                 "hidden_sha256", "machine", "cpu_model")}
    contract["generated_sha256"] = config.get("runtime_manifest", {}).get("generated_sha256")
    contract["cggm_worker_limit_per_party"] = config.get("cggm_worker_limit_per_party", 1)
    contract["sci_protocol_threads_per_party"] = config.get(
        "sci_protocol_threads_per_party", config.get("application_threads_per_party", 1))
    cpus = config.get("cpus")
    legacy_affinities = [[cpus[0]], [cpus[1]]] if cpus and len(cpus) >= 2 else None
    contract["cpu_affinity_per_party"] = config.get("cpu_affinity_per_party", legacy_affinities)
    return contract


def summarize(records):
    rows, contracts, seen = [], {}, set()
    for source, result in records:
        config = result["config"]
        if config.get("diagnostic"):
            raise ValueError("diagnostic runs cannot enter secure cost comparison")
        group = (config["network"]["name"], config["steps"])
        key = group + (config["backend"],)
        if key in seen:
            raise ValueError("duplicate backend workload: " + repr(key))
        seen.add(key)
        complete = result["status"] == "complete"
        contract = comparison_contract(config)
        if complete:
            if group in contracts and contracts[group] != contract:
                raise ValueError("different comparison contract within " + repr(group))
            contracts[group] = contract
        row = {"backend": config["backend"], "network": config["network"],
               "batch": config["batch"], "steps": config["steps"],
               "max_segments": config["max_segments"], "source": str(source),
               "cggm_worker_limit_per_party": contract["cggm_worker_limit_per_party"],
               "sci_protocol_threads_per_party": contract["sci_protocol_threads_per_party"],
               "cpu_affinity_per_party": contract["cpu_affinity_per_party"],
               "status": result["status"], "reason": result.get("reason"),
               "fixed_batch_outputs": result.get("fixed_batch_outputs"),
               "full_evaluation_quality": "SDCF-09 deferred by user",
               "quality_qualified_ranking": False,
               "setup_seconds": None, "offline_seconds": None, "online_seconds": None,
               "query_total_seconds": None, "first_query_seconds": None,
               "session_total_seconds": None, "session_seconds_per_query": None,
               "peak_rss_kib_per_party": None, "offline_io": None, "online_io": None,
               "queries": 0}
        if complete:
            measured = result["measurements"]
            queries = measured["queries"]
            if len(queries) != config["queries"]:
                raise ValueError("measured query count differs from frozen config")
            row.update(queries=len(queries), setup_seconds=measured["setup"]["seconds"],
                       first_query_seconds=measured.get("first_query", {}).get("seconds"),
                       session_total_seconds=measured["total_seconds"],
                       session_seconds_per_query=measured["total_seconds"] / len(queries),
                       peak_rss_kib_per_party=measured["peak_rss_kib_per_party"])
            for phase in ("offline", "online", "query_total"):
                row[phase + "_seconds"] = distribution([query.get(phase, {}).get("seconds") for query in queries])
            for phase in ("offline", "online"):
                row[phase + "_io"] = {field: distribution([query[phase].get(field) for query in queries])
                                      for field in ("sent_bytes", "send_calls", "recv_calls", "logical_rounds")}
        rows.append(row)
    return rows


def markdown(rows):
    lines = ["# Batch=128 secure workload costs", "",
             "SDCF-09 is deferred. These are fixed-batch performance measurements; "
             "full evaluation-set correctness and Top-1 qualification are pending.", "",
             "Offline and online are medians of the maximum party duration for each fresh query. "
             "Session time is measured directly, then divided by the number of queries for amortization. "
             "First-query time is measured from application startup through the first output. "
             "Communication sums both parties' sends once. Missing measurements remain unavailable.", "",
             "| Network | Steps | Backend | Status | Setup s | Offline s/query | Online s/query | First query s | Session s/query | Offline MB | Online MB |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def value(number):
        return "N/A" if number is None else "%.6f" % number
    for row in rows:
        offline = row["offline_seconds"]["median"] if row["offline_seconds"] else None
        online = row["online_seconds"]["median"] if row["online_seconds"] else None
        comm = [row[phase + "_io"]["sent_bytes"]["median"] / 1e6 if row[phase + "_io"] else None
                for phase in ("offline", "online")]
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" %
                     (row["network"]["name"], row["steps"], row["backend"], row["status"],
                      value(row["setup_seconds"]), value(offline), value(online),
                      value(row["first_query_seconds"]), value(row["session_seconds_per_query"]),
                      value(comm[0]), value(comm[1])))
    lines += ["", "## Evidence", ""]
    for row in rows:
        lines.append("- `%s`, %d steps, `%s`: [%s](%s)%s" %
                     (row["network"]["name"], row["steps"], row["backend"], row["status"],
                      row["source"], "; " + row["reason"] if row["reason"] else ""))
    lines += ["", "Each result preserves commands, phase logs, per-party memory and I/O, "
              "query values, and source/fixture identities. The JSON summary includes min/max "
              "and population standard deviation. LAN/WAN profiles are local payload relays, "
              "not original-paper hardware or live Internet measurements.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize([(str(path.resolve()), json.loads(path.read_text())) for path in args.result])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(markdown(rows))


if __name__ == "__main__":
    main()
