"""Public graph schedule and measured two-party phase aggregation."""
import re
import statistics


BACKENDS = ("SCI", "SCI_preprocessed", "SCI_ours", "SCI_Beaver", "SCI_Beaver_full", "SCI_Beaver_segmented", "SCI_Beaver_segmented_DPF_LUT", "SCI_Beaver_segmented_PaperLUT")


def distribution(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "pstdev": statistics.pstdev(values), "values": values}


def summarize_queries(result):
    """Summarize already aligned party maxima, excluding any removed warmup."""
    for phase in ("offline", "input", "ready", "online", "output", "query_total"):
        values = [query[phase]["seconds"] for query in result["queries"] if phase in query]
        result[phase + "_seconds"] = distribution(values) if values else None
        result[phase + "_seconds_median"] = statistics.median(values) if values else None
    result["session_total_seconds"] = result["total_seconds"]


def validate_fixture(contract, manifest, files):
    if contract.get("generated_sha256") != manifest.get("generated_sha256") or not contract.get("generated_sha256"):
        raise ValueError("generated graph differs from the prepared model/input contract")
    for name, key in (("server_model.i16", "server_model_sha256"), ("client_input.i16", "client_input_sha256")):
        if contract.get(key) != files.get(name) or not contract.get(key):
            raise ValueError("prepared artifact mismatch: " + name)


def call_arguments(row):
    return [value.strip() for value in row["source_call"].split("(", 1)[1][:-2].split(",")]


def dependency_schedule(schema, batch, steps, classifier=True, model_bias_precomputed=False):
    if batch < 1 or batch > 256 or steps not in (1, 99):
        raise ValueError("batch must be 1..256 and steps must be 1 or 99")
    recurrent = [row for row in schema if "FC," not in row["source_call"] and "FCbias," not in row["source_call"]]
    head = [row for row in schema if row not in recurrent]
    schedule, producers = [], {}
    for step, rows in [(index, recurrent) for index in range(steps)] + ([(steps, head)] if classifier else []):
        producers.pop("tmp15bw16", None)
        for row in rows:
            args = call_arguments(row)
            source_call = row["source_call"]
            if model_bias_precomputed and row["function"] == "MatAdd" and args[11] in ("Bg", "Bh"):
                if args[1:10] != ["100", "1", "8", "1", "1", "16", "16", "16", "16"]:
                    raise ValueError("model bias preparation requires the frozen signed 16-bit /8 contract")
                args[3] = "1"
                source_call = "MatAdd(" + ", ".join(args) + ");"
            output_index = len(args) - (2 if row["function"] == "MatMul" else 1)
            inputs = [value for value in args[:output_index] if re.fullmatch(r"[A-Za-z_]\w*", value)]
            dependencies = sorted({producers[value] for value in inputs if value in producers})
            shape = [batch] + [int(value) for value in args[1:3 if row["function"] == "MatMul" else 2]]
            index = len(schedule)
            schedule.append({
                "index": index, "step": step, "operator": row["operator"],
                "function": row["function"], "shape": shape,
                "sample_groups": 1, "output_elements": batch * shape[-1],
                "output_bits": int(re.search(r"bw(\d+)$", row["output"])[1]),
                "source_call": source_call, "inputs": inputs,
                "output": row["output"], "dependencies": dependencies,
                "dependency_layer": 1 + max((schedule[parent]["dependency_layer"] for parent in dependencies), default=-1),
            })
            producers[row["output"]] = index
    return schedule


def aggregate_phases(parties, queries):
    if [party.get("party") for party in parties] != [1, 2] or any(party.get("status") != "complete" for party in parties):
        raise ValueError("requires two completed ordered parties")

    def pair(phase, query):
        rows = [[row for row in party["phases"] if row["phase"] == phase and row["query"] == query] for party in parties]
        if any(len(row) != 1 for row in rows):
            raise ValueError("missing or duplicate phase " + phase)
        left, right = (row[0] for row in rows)
        if left["sent_bytes"] != right["received_bytes"] or right["sent_bytes"] != left["received_bytes"]:
            raise ValueError("unbalanced communication in phase " + phase)
        rounds = [row.get("logical_rounds") for row in (left, right)]
        def sum_known(key):
            values = [row.get(key) for row in (left, right)]
            return None if None in values else sum(values)
        cpu = [row.get("process_cpu_seconds") for row in (left, right)]
        return {"seconds": max(left["seconds"], right["seconds"]),
                "process_cpu_seconds_max": None if None in cpu else max(cpu),
                "process_cpu_seconds_total": sum_known("process_cpu_seconds"),
                "sent_bytes": left["sent_bytes"] + right["sent_bytes"],
                "send_calls": sum_known("send_calls"),
                "recv_calls": sum_known("recv_calls"),
                "logical_rounds": None if None in rounds else max(rounds),
                "per_party": [left, right]}

    result = {"total_seconds": max(party["total_seconds"] for party in parties),
              "total_seconds_per_party": [party["total_seconds"] for party in parties],
              "peak_rss_kib_per_party": [party["peak_rss_kib"] for party in parties],
              "setup": pair("setup", -1), "teardown": pair("teardown", -1),
              "queries": [{phase: pair(phase, query) for phase in ("offline", "input", "ready", "online", "output")} for query in range(queries)],
              "wall_rule": "maximum party wall time; total measured directly",
              "communication_rule": "sum both parties' sends once",
              "logical_rounds_reason": "null where native protocol dependency rounds are not instrumented"}
    if any(any(row["phase"] == "model_prepare" for row in party["phases"]) for party in parties):
        result["model_prepare"] = pair("model_prepare", -1)
    preparations = [party.get("model_bias_preparation", []) for party in parties]
    if any(preparations):
        result["model_bias_preparation"] = preparations
    for phase in ("query_total", "first_query"):
        present = [any(row["phase"] == phase for row in party["phases"]) for party in parties]
        if any(present):
            if phase == "first_query":
                result[phase] = pair(phase, -1)
            else:
                for query, record in enumerate(result["queries"]):
                    record[phase] = pair(phase, query)
    summarize_queries(result)
    return result
