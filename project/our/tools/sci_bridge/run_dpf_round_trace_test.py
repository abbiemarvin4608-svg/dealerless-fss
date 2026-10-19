#!/usr/bin/env python3
"""Check real two-party DPF setup and cumulative LUT traffic against callbacks."""
import argparse
import json
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for port in range(33000, 54000, 137):
        sockets = []
        try:
            for offset in (0, 50, 100):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port + offset))
            break
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    else:
        raise RuntimeError("no free two-party port range")
    children, logs, commands = [], [], []
    try:
        for role in (1, 2):
            command = [str(args.binary.resolve()), str(role), str(port)]
            commands.append(command)
            log = (args.output / ("party%d.log" % role)).open("w")
            logs.append(log)
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        exits = [child.wait(timeout=180) for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
        for child in children:
            child.wait()
        for log in logs:
            log.close()
    rows = []
    for role in (1, 2):
        rows.append([json.loads(line.split(" ", 1)[1])
            for line in (args.output / ("party%d.log" % role)).read_text().splitlines()
            if line.startswith("DPF_COUNTER_TRACE ")])
    expected = [("setup", -1)] + [(phase, query) for query in (0, 1)
        for phase in ("offline", "online", "snapshot")]
    matched = all([(row["phase"], row["query"]) for row in party] == expected for party in rows) and all(
        alice["phase"] == bob["phase"] and alice["query"] == bob["query"] and
        alice["sent_bytes"] == bob["received_bytes"] and
        alice["received_bytes"] == bob["sent_bytes"] and all(
            row["sent_bytes"] == row["trace_sent_bytes"] and
            row["received_bytes"] == row["trace_received_bytes"] for row in (alice, bob))
        for alice, bob in zip(*rows))
    report = {"status": "passed" if exits == [0, 0] and matched else "failed",
        "scope": "creation-time base OT, two fresh LUT queries, cumulative byte-matched callbacks, repeated snapshots",
        "commands": commands, "party_exit_codes": exits, "cross_party_bytes_matched": matched,
        "snapshots": rows}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
