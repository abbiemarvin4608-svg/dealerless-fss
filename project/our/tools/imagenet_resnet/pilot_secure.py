#!/usr/bin/env python3
"""Measure the unmodified first ResNet convolution as diagnostic evidence."""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    graph = json.loads((source / "graph.json").read_text())
    node = graph["nodes"][0]
    if node["op"] != "conv2d" or graph["tensors"][graph["input"]] != [1, 3, 224, 224]:
        raise ValueError("pilot requires the full 224x224 ResNet conv1")
    graph["model"] += "_conv1_diagnostic"
    graph["output"] = node["output"]
    graph["nodes"] = [node]
    graph["tensors"] = {name: graph["tensors"][name] for name in [graph["input"], node["output"]]}
    graph["evidence_scope"] = "diagnostic_first_layer_only"
    write_json(output / "graph.json", graph)
    oracle = json.loads((source / "audit/index.json").read_text())
    oracle["model"] = graph["model"]
    oracle["tensors"] = {name: oracle["tensors"][name] for name in graph["tensors"]}
    for record in oracle["tensors"].values():
        record["file"] = str((source / "audit" / record["file"]).resolve())
    (output / "oracle").mkdir()
    write_json(output / "oracle/index.json", oracle)
    return graph


def run(args):
    model = args.model.resolve()
    output = args.output.resolve()
    graph = prepare(model, output)
    node = graph["nodes"][0]
    outputs = 1
    for dimension in node["shape"]:
        outputs *= dimension
    inner = 1
    for dimension in node["weight_shape"][1:]:
        inner *= dimension
    rows = []
    for backend in ("SCI", "SCI_ours_beaver"):
        directory = output / backend
        resource_path = output / (backend + ".resource.txt")
        command = ["/usr/bin/time", "-v", "-o", str(resource_path), "python3", str(HERE / "run_secure.py"),
            "--binary", str(args.binary.resolve()), "--graph", str(output / "graph.json"),
            "--parameters", str(model / "parameters.bin"), "--input", str(model / "audit/input.bin"),
            "--backend", backend, "--queries", "1", "--diagnostic", "--output", str(directory),
            "--timeout", str(args.timeout), "--port", str(args.port)]
        subprocess.run(command, check=True)
        subprocess.run(["python3", str(HERE / "audit_secure.py"), "--directory", str(directory),
            "--oracle", str(output / "oracle")], check=True)
        parties = [json.loads((directory / ("party%d.json" % role)).read_text()) for role in (1, 2)]
        resource = {}
        for line in resource_path.read_text().splitlines():
            if ": " in line:
                key, value = line.strip().split(": ", 1)
                resource[key] = value
        phases = {}
        for phase in ("offline", "online", "input", "output"):
            metrics = [party["queries"][0]["phases"][phase] for party in parties]
            phases[phase] = dict(seconds=max(item["seconds"] for item in metrics),
                process_cpu_seconds=sum(item["process_cpu_seconds"] for item in metrics),
                sent_bytes=sum(item["sent_bytes"] for item in metrics),
                received_bytes=sum(item["received_bytes"] for item in metrics))
        rows.append(dict(backend=backend, status="complete", phases=phases,
            peak_process_rss_kib=int(resource.get("Maximum resident set size (kbytes)", 0)),
            rss_scope="maximum individual process including diagnostic output capture; not pair sum",
            beaver_spool_bytes_each_party=[party["queries"][0]["beaver_spool_bytes"] for party in parties],
            trace_bytes=[(directory / ("party%d.trace.jsonl" % role)).stat().st_size for role in (1, 2)],
            round_trace=json.loads((directory / "ROUNDS.json").read_text()),
            integer_audit=json.loads((directory / "INTEGER_AUDIT.json").read_text())))
        report = dict(status="running" if backend == "SCI" else "complete", evidence_scope="diagnostic_first_layer_only",
            graph=str(output / "graph.json"), multiply_accumulate_count=outputs * inner,
            input_shape=graph["tensors"][graph["input"]], output_shape=node["shape"], rows=rows)
        write_json(output / "PILOT.json", report)
        print(json.dumps({"backend": backend, "phases": phases, "peak_process_rss_kib": rows[-1]["peak_process_rss_kib"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=42000)
    parser.add_argument("--timeout", type=int, default=1800)
    run(parser.parse_args())
