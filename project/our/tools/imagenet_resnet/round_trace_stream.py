#!/usr/bin/env python3
"""Analyze phase send flights with bounded event memory and temporary disk.

The default validates semantic scopes but omits their individual round counts.
``include_scopes=True`` uses the original in-memory analyzer for small fixtures.
"""
import argparse
import collections
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segmented_dcf_runtime"))
from round_trace import (ATOMIC_PHASES, DEFINITION, KINDS, SEND, RECV, FORK, START,
                         END, JOIN, PHASE_BEGIN, PHASE_END, SCOPE_BEGIN, SCOPE_END,
                         _expected_rows)

MISSING = (1 << 64) - 1
SEMANTIC_FIELDS = ("label", "query", "layer", "slot", "call", "kind", "stage")


class _Pages:
    def __init__(self, capacity=128, size=65536):
        self.capacity, self.size = capacity, size
        self.rows = collections.OrderedDict()
        self.stores = set()

    def get(self, store, page):
        key = store, page
        row = self.rows.pop(key, None)
        if row is None:
            if len(self.rows) == self.capacity:
                (old_store, old_page), old = self.rows.popitem(last=False)
                self.flush(old_store, old_page, old)
            store.file.seek(page * self.size)
            data = store.file.read(self.size)
            row = [bytearray(data + bytes(self.size - len(data))), False]
        self.rows[key] = row
        return row

    def flush(self, store, page, row):
        if row[1]:
            store.file.seek(page * self.size)
            store.file.write(row[0])
            row[1] = False

    def close_store(self, store):
        for key in [key for key in self.rows if key[0] is store]:
            del self.rows[key]


class _Records:
    def __init__(self, pages, fields, directory):
        self.pages = pages
        self.record = struct.Struct("<" + "Q" * fields)
        # A page contains an integral number of records.
        self.per_page = pages.size // self.record.size
        self.file = tempfile.TemporaryFile(dir=directory)
        self.count = 0
        pages.stores.add(self)

    def get(self, index):
        if not 0 <= index < self.count:
            raise ValueError("missing local dependency")
        page, slot = divmod(index, self.per_page)
        row = self.pages.get(self, page)
        return self.record.unpack_from(row[0], slot * self.record.size)

    def put(self, index, value):
        page, slot = divmod(index, self.per_page)
        row = self.pages.get(self, page)
        self.record.pack_into(row[0], slot * self.record.size, *value)
        row[1] = True

    def append(self, value):
        self.put(self.count, value)
        self.count += 1

    def close(self):
        self.pages.close_store(self)
        self.pages.stores.discard(self)
        self.file.close()


class _Metadata:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("CREATE TABLE kv (bucket TEXT, key TEXT, value TEXT, PRIMARY KEY(bucket,key)) WITHOUT ROWID")
        self.db.execute("CREATE TABLE identities (identity TEXT PRIMARY KEY, parent TEXT) WITHOUT ROWID")
        self.cache = collections.OrderedDict()

    def get(self, bucket, key, default=None):
        key = bucket, str(key)
        row = self.cache.pop(key, None)
        if row is None:
            stored = self.db.execute("SELECT value FROM kv WHERE bucket=? AND key=?", key).fetchone()
            if stored is None:
                return default
            row = [json.loads(stored[0]), False]
        self.cache[key] = row
        self.trim()
        return row[0]

    def put(self, bucket, key, value):
        key = bucket, str(key)
        self.cache.pop(key, None)
        self.cache[key] = [value, True]
        self.trim()

    def trim(self):
        while len(self.cache) > 2048:
            key, (value, dirty) = self.cache.popitem(last=False)
            if dirty:
                self.db.execute("INSERT OR REPLACE INTO kv VALUES(?,?,?)", (*key, json.dumps(value)))

    def reset_phase(self):
        for key in [key for key in self.cache if key[0] != "actor"]:
            del self.cache[key]
        self.db.execute("DELETE FROM kv WHERE bucket != 'actor'")
        self.db.execute("DELETE FROM identities")


class _Phase:
    def __init__(self, first, key, pages, directory):
        self.first, self.key = first, key
        self.events = _Records(pages, 7, directory)
        self.states = _Records(pages, 4, directory)
        self.flows = {}
        self.pages, self.directory = pages, directory
        self.sent, self.received = 0, 0
        self.children = self.ends = self.joins = self.open_scopes = 0

    def flow(self, kind, channel):
        key = kind, channel
        if key not in self.flows:
            self.flows[key] = _Records(self.pages, 3, self.directory)
        return self.flows[key]

    def close(self):
        self.events.close()
        self.states.close()
        for flow in self.flows.values():
            flow.close()


class _Reader:
    def __init__(self, path, party, pages, directory):
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        self.stream = opener(path, "rt")
        try:
            if json.loads(next(self.stream, "null")) != {"type": "meta", "version": 1, "party": party}:
                raise ValueError("invalid trace metadata for party %d" % party)
        except BaseException:
            self.stream.close()
            raise
        self.party, self.pages, self.directory = party, pages, directory
        self.meta = _Metadata(Path(directory) / ("party%d.sqlite" % party))
        self.count, self.phase_number, self.complete = 0, 0, False
        self.frontiers = {}

    def next_phase(self):
        phase = None
        for line in self.stream:
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "complete":
                if self.complete or phase is not None or event.get("events") != self.count:
                    raise ValueError("trace footer event count or phase mismatch")
                self.complete = True
                continue
            if self.complete:
                raise ValueError("events follow the trace completion footer")
            self.count += 1
            if event.get("id") != self.count:
                raise ValueError("trace event sequence mismatch")
            if kind not in KINDS:
                raise ValueError("unknown trace event type")
            for name in ("actor", "prev", "dependency", "channel", "offset", "length", "scope"):
                value = event.get(name, 0)
                if not isinstance(value, int) or not 0 <= value < 1 << 64:
                    raise ValueError("invalid trace %s" % name)
            actor = event.get("actor", 0)
            if actor < 1 or type(event.get("scope", 0)) is not int:
                raise ValueError("invalid trace actor or semantic scope identifier")
            if not isinstance(event.get("query"), int) or not isinstance(event.get("phase"), str):
                raise ValueError("invalid trace phase")
            key = event["query"], event["phase"]
            if kind == "phase_begin":
                if actor != 1 or phase is not None or event["phase"] not in ATOMIC_PHASES:
                    raise ValueError("invalid phase begin boundary")
                self.phase_number += 1
                self.meta.reset_phase()
                phase = _Phase(self.count, key, self.pages, self.directory)
            if phase is None or phase.key != key:
                raise ValueError("event lies outside its phase boundaries")
            self.append(phase, event)
            if kind == "phase_end":
                if actor != 1:
                    raise ValueError("phase end must belong to the root actor")
                if phase.children != phase.ends or phase.ends != phase.joins:
                    raise ValueError("missing phase end or child fork/join completion")
                if phase.open_scopes:
                    raise ValueError("semantic scope is missing its end boundary")
                return phase
        if phase is not None or not self.complete:
            raise ValueError("trace did not complete for party %d" % self.party)
        return None

    def append(self, phase, event):
        kind, actor = KINDS[event["type"]], event["actor"]
        identifier, previous, dependency = self.count, event.get("prev", 0), event.get("dependency", 0)
        # actor record: last id, phase, ended, owner, current scope, joined.
        state = self.meta.get("actor", actor, [0, 0, False, 0, 0, False])
        if previous != state[0]:
            raise ValueError("actor predecessor is not consecutive")
        if kind == PHASE_BEGIN:
            state = [state[0], self.phase_number, False, 0, 0, False]
        elif kind == START:
            if state[1] == self.phase_number or previous:
                raise ValueError("actor started twice or has a prior lifetime")
            source = dependency - phase.first
            if not 0 <= source < phase.events.count:
                raise ValueError("actor start has no matching fork")
            fork = phase.events.get(source)
            if fork[0] != FORK:
                raise ValueError("actor start has no matching fork")
            state = [0, self.phase_number, False, fork[1], fork[4], False]
            phase.children += 1
        elif state[1] != self.phase_number:
            raise ValueError("communication actor has no traced fork/start")
        if state[2]:
            raise ValueError("actor used after completion")
        current, scope = state[4], event.get("scope", 0)
        if kind == SCOPE_BEGIN:
            semantic = event.get("semantic", {})
            if (not scope or self.meta.get("scope", scope) is not None or
                    event.get("parent_scope") != current or set(semantic) != set(SEMANTIC_FIELDS) or
                    semantic.get("label") not in ("activation", "folded_call", "activation_stage", "folded_prepare") or
                    any(type(semantic[field]) is not int for field in SEMANTIC_FIELDS[1:]) or
                    semantic["query"] != phase.key[0]):
                raise ValueError("invalid or duplicated semantic scope boundary")
            base = json.dumps([semantic[field] for field in SEMANTIC_FIELDS])
            occurrence = self.meta.get("occurrence", base, 0)
            self.meta.put("occurrence", base, occurrence + 1)
            identity = json.dumps([semantic[field] for field in SEMANTIC_FIELDS] + [occurrence])
            parent = self.meta.get("scope", current)[3] if current else None
            self.meta.put("scope", scope, [actor, current, True, identity])
            self.meta.db.execute("INSERT INTO identities VALUES(?,?)", (identity, parent))
            state[4] = current = scope
            phase.open_scopes += 1
        elif scope != current:
            raise ValueError("semantic scope does not follow its actor or inherited fork")
        cursor = current
        while cursor:
            row = self.meta.get("scope", cursor)
            if row is None or not row[2]:
                raise ValueError("actor used a semantic scope after its end boundary")
            if kind == END and row[0] == actor:
                raise ValueError("actor completed with an unfinished semantic scope")
            cursor = row[1]
        if kind == SCOPE_END:
            row = self.meta.get("scope", scope)
            if row is None or row[0] != actor or not row[2]:
                raise ValueError("unowned or duplicated semantic scope end")
            row[2] = False
            self.meta.put("scope", scope, row)
            state[4] = row[1]
            phase.open_scopes -= 1
        elif kind == PHASE_END and current:
            raise ValueError("phase completed with an unfinished semantic scope")
        if kind == END:
            if actor == 1 or not state[3]:
                raise ValueError("invalid child actor completion")
            state[2] = True
            phase.ends += 1
        elif kind == JOIN:
            source = dependency - phase.first
            if not 0 <= source < phase.events.count:
                raise ValueError("join has no matching owned child completion")
            end = phase.events.get(source)
            child = self.meta.get("actor", end[1])
            if end[0] != END or child is None or child[3] != actor or child[5]:
                raise ValueError("join has no matching owned child completion")
            child[5] = True
            self.meta.put("actor", end[1], child)
            phase.joins += 1
        state[0] = identifier
        self.meta.put("actor", actor, state)
        local = []
        for value in (previous, dependency if dependency != previous else 0):
            if not value or (value < phase.first and value == previous):
                local.append(MISSING)
            elif value < phase.first:
                raise ValueError("missing local dependency")
            else:
                local.append(value - phase.first)
        phase.events.append((kind, actor, *local, scope, 0, 0))
        if kind in (SEND, RECV):
            channel, offset, length = event["channel"], event["offset"], event["length"]
            if offset != self.frontiers.get((kind, channel), 0):
                raise ValueError("communication stream has a gap or overlap")
            if length <= 0 or offset + length >= 1 << 64:
                raise ValueError("invalid communication byte interval")
            self.frontiers[kind, channel] = offset + length
            phase.flow(kind, channel).append((offset, offset + length, phase.events.count - 1))
            if kind == SEND:
                phase.sent += length
            else:
                phase.received += length

    def close(self):
        self.stream.close()
        self.meta.db.close()


def _match(parties):
    for party, phase in enumerate(parties):
        for (kind, channel), flow in phase.flows.items():
            peer = parties[1 - party].flows.get((RECV if kind == SEND else SEND, channel))
            if peer is None:
                raise ValueError("unmatched send or receive bytes on channel %s" % channel)
            if kind == SEND:
                continue
            if flow.get(0)[0] != peer.get(0)[0] or flow.get(flow.count - 1)[1] != peer.get(peer.count - 1)[1]:
                raise ValueError("unmatched communication byte totals")
            first = 0
            for index in range(flow.count):
                lower, upper, node = flow.get(index)
                while peer.get(first)[1] <= lower:
                    first += 1
                last = first
                while last < peer.count and peer.get(last)[0] < upper:
                    last += 1
                row = list(phase.events.get(node))
                # Scope inheritance has been validated; reuse its slot for the
                # receive channel during clock replay.
                row[4], row[5], row[6] = channel, first, last
                phase.events.put(node, row)


def _depths(parties, pages, directory):
    stack = _Records(pages, 3, directory)
    maximum = [0, 0]
    for phase in parties:
        phase.states.count = phase.events.count

    def dependency(party, row, cursor):
        if cursor < 2:
            value = row[2 + cursor]
            return None if value == MISSING else (party, value)
        if row[0] != RECV or row[5] + cursor - 2 >= row[6]:
            return None
        flow = parties[1 - party].flows[SEND, row[4]]
        return 1 - party, flow.get(row[5] + cursor - 2)[2]

    try:
        for party, phase in enumerate(parties):
            for node in range(phase.events.count):
                if phase.states.get(node)[0] == 2:
                    continue
                phase.states.put(node, (1, 0, 0, 0))
                stack.append((party, node, 0))
                while stack.count:
                    actor_party, current, cursor = stack.get(stack.count - 1)
                    active = parties[actor_party]
                    row = active.events.get(current)
                    limit = 2 + (row[6] - row[5] if row[0] == RECV else 0)
                    if cursor < limit:
                        source = dependency(actor_party, row, cursor)
                        stack.put(stack.count - 1, (actor_party, current, cursor + 1))
                        if source is None:
                            continue
                        status = parties[source[0]].states.get(source[1])[0]
                        if status == 1:
                            raise ValueError("communication dependency cycle")
                        if status == 0:
                            parties[source[0]].states.put(source[1], (1, 0, 0, 0))
                            stack.append((*source, 0))
                        continue
                    clock0 = clock1 = 0
                    local = []
                    for position in range(limit):
                        source = dependency(actor_party, row, position)
                        if source is None:
                            continue
                        _, left, right, last = parties[source[0]].states.get(source[1])
                        clock0, clock1 = max(clock0, left), max(clock1, right)
                        if source[0] == actor_party:
                            local.append((left, right, last))
                    frontier = [value for value in local if not any(
                        other[0] >= value[0] and other[1] >= value[1] and other[:2] != value[:2]
                        for other in local)]
                    last = bool(frontier) and all(value[2] for value in frontier)
                    if row[0] == PHASE_BEGIN:
                        clock0 = clock1 = 0
                        last = False
                    elif row[0] == SEND:
                        if not last:
                            if actor_party:
                                clock1 += 1
                            else:
                                clock0 += 1
                        last = True
                    elif row[0] == RECV:
                        last = False
                    active.states.put(current, (2, clock0, clock1, int(last)))
                    maximum[0], maximum[1] = max(maximum[0], clock0), max(maximum[1], clock1)
                    stack.count -= 1
    finally:
        stack.close()
    return maximum


def analyze_pair(server_trace, client_trace, expected_phases=None, scope_groups=None,
                 include_scopes=False, temp_directory=None, use_native=True):
    """Match the original phase clock semantics without retaining a phase DAG."""
    try:
        result = None
        note = None
        if use_native and os.environ.get("SEGFSS_ROUNDS_PYTHON") != "1" and not any(
                Path(path).suffix == ".gz" for path in (server_trace, client_trace)):
            binary, note = _native_binary()
            if binary:
                result = _native_pair(binary, server_trace, client_trace, expected_phases, temp_directory)
        if result is None:
            result = _analyze_pair(server_trace, client_trace, expected_phases, temp_directory)
            result["analyzer"] = "python"
            result["analyzer_binary_sha256"] = None
            if note:
                result["native_unavailable"] = note
        if result["status"] != "passed":
            return result
        if include_scopes or scope_groups:
            from round_trace import analyze_pair as original_analyze
            scoped = original_analyze(server_trace, client_trace, expected_phases, scope_groups)
            if scoped["status"] != "passed" or scoped["phases"] != result["phases"]:
                raise ValueError("original scope replay differs from streaming phase analysis")
            result["scopes"] = scoped["scopes"]
            del result["scope_rounds_omitted"]
            result["scope_rounds_computed"] = True
        else:
            result["scope_rounds_computed"] = False
        return result
    except (OSError, EOFError, ValueError, KeyError, TypeError, sqlite3.Error, struct.error) as error:
        return {"status": "failed", "reason": str(error), "definition": DEFINITION,
                "logical_rounds": None, "phases": []}


def _native_binary():
    candidate = os.environ.get("SEGFSS_ROUNDS_NATIVE_BINARY")
    if candidate:
        binary = Path(candidate).resolve()
        if not binary.is_file():
            raise ValueError("explicit native analyzer does not exist: " + str(binary))
        return binary, None
    if os.environ.get("SEGFSS_ROUNDS_REBUILD") == "1":
        return _build_native_binary()
    root = Path(__file__).resolve().parent / ".round_trace_native"
    pointer = root / "qualified.json"
    if not pointer.is_file():
        return None, "no qualified native analyzer is recorded"
    identity = json.loads(pointer.read_text())
    binary = root / identity["binary"]
    if not binary.is_file() or _binary_digest(binary) != identity["binary_sha256"]:
        raise ValueError("qualified native analyzer is missing or its SHA256 differs")
    return binary, None


def _binary_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qualify_native(binary, validation):
    """Install an immutable analyzer after its regression and pilot checks."""
    root = Path(__file__).resolve().parent / ".round_trace_native"
    digest = _binary_digest(binary)
    destination = root / "qualified-binaries" / digest / "analyzer"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_name("analyzer.%d.tmp" % os.getpid())
        try:
            shutil.copy2(binary, temporary)
            if _binary_digest(temporary) != digest:
                raise ValueError("native analyzer changed while it was being qualified")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    if _binary_digest(destination) != digest:
        raise ValueError("qualified analyzer cache contents differ from its digest")
    identity = {"binary": str(destination.relative_to(root)), "binary_sha256": digest,
                "qualified_at_utc": datetime.now(timezone.utc).isoformat(), "validation": validation}
    pointer = root / ("qualified.%d.tmp" % os.getpid())
    pointer.write_text(json.dumps(identity, indent=2) + "\n")
    pointer.replace(root / "qualified.json")
    return identity


def _build_native_binary():
    here = Path(__file__).resolve().parent
    source = here / "round_trace_native.cpp"
    header = here / "vendor/nlohmann/json.hpp"
    if not source.is_file() or not header.is_file():
        return None, "native source or JSON header is unavailable"
    compiler = shutil.which("g++")
    if compiler is None:
        return None, "g++ is unavailable"
    flags = ["-O3", "-std=c++17", "-pthread", "-I" + str(here / "vendor")]
    libraries = ["-lsqlite3"]
    identity = hashlib.sha256()
    for path in [source, *sorted((here / "vendor/nlohmann").rglob("*.hpp"))]:
        identity.update(path.read_bytes())
    identity.update(json.dumps([compiler, flags, libraries]).encode())
    build = here / ".round_trace_native" / identity.hexdigest()[:20]
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "analyzer"
    with (build / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            temporary = build / ("analyzer.%d.tmp" % os.getpid())
            try:
                result = subprocess.run([compiler, *flags, str(source), *libraries, "-o", str(temporary)],
                                        capture_output=True, text=True, timeout=180)
                (build / "build.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    return None, "native compilation failed; see " + str(build / "build.log")
                temporary.replace(binary)
            except subprocess.TimeoutExpired:
                return None, "native compilation exceeded 180 seconds"
            finally:
                if temporary.exists():
                    temporary.unlink()
    return binary, None


def _native_pair(binary, server_trace, client_trace, expected_phases, temp_directory):
    binary_sha256 = _binary_digest(binary)
    with tempfile.TemporaryDirectory(prefix="resnet-rounds-native-", dir=temp_directory) as directory:
        output = Path(directory) / "rounds.json"
        command = [str(binary), "--server", str(Path(server_trace).resolve()),
                   "--client", str(Path(client_trace).resolve()), "--output", str(output),
                   "--temp-directory", directory]
        run = subprocess.run(command, capture_output=True, text=True)
        if not output.is_file():
            raise ValueError("native analyzer did not write a result: " + run.stderr[-2000:])
        result = json.loads(output.read_text())
        result["analyzer"] = "native"
        result["analyzer_binary_sha256"] = binary_sha256
        if result.get("status") != "passed":
            return result
        if run.returncode or result.get("definition") != DEFINITION:
            raise ValueError("native analyzer exit status or round definition mismatch")
        expected = [_expected_rows(party) for party in expected_phases] if expected_phases else None
        observed = set()
        for row in result["phases"]:
            key = row["query"], row["phase"]
            if key in observed or key[1] not in ATOMIC_PHASES:
                raise ValueError("parties have different or duplicated phase sets")
            observed.add(key)
            sent, received, depths = row["party_sent_bytes"], row["party_received_bytes"], row["send_flight_depths"]
            if (len(sent) != 2 or len(received) != 2 or sent != received[::-1] or
                    len(depths) != 2 or row["logical_rounds"] != max(depths)):
                raise ValueError("native analyzer phase bytes or clock report is invalid")
            if expected:
                for party in (0, 1):
                    metric = expected[party].get(key)
                    if metric is None:
                        raise ValueError("trace phase set differs from phase metrics")
                    if metric["sent_bytes"] != sent[party] or metric["received_bytes"] != received[party]:
                        raise ValueError("trace bytes differ from phase metric")
        if expected and (observed != set(expected[0]) or observed != set(expected[1])):
            raise ValueError("trace phase set differs from phase metrics")
        return result


def _analyze_pair(server_trace, client_trace, expected_phases, temp_directory):
    expected = [_expected_rows(party) for party in expected_phases] if expected_phases else None
    reports, observed, count = [], set(), 0
    with tempfile.TemporaryDirectory(prefix="resnet-rounds-", dir=temp_directory) as directory:
        pages = _Pages()
        readers = []
        try:
            for party, path in enumerate((server_trace, client_trace), 1):
                readers.append(_Reader(path, party, pages, directory))
            while True:
                parties = [reader.next_phase() for reader in readers]
                if parties == [None, None]:
                    break
                if None in parties:
                    raise ValueError("parties have different phase sets")
                key = parties[0].key
                if key != parties[1].key or key in observed:
                    raise ValueError("parties have different or duplicated phase sets")
                observed.add(key)
                for party in (0, 1):
                    if expected:
                        row = expected[party].get(key)
                        if row is None:
                            raise ValueError("trace phase set differs from phase metrics")
                        if row["sent_bytes"] != parties[party].sent or row["received_bytes"] != parties[party].received:
                            raise ValueError("trace bytes differ from phase metric")
                inventories = [reader.meta.db.execute("SELECT identity,parent FROM identities ORDER BY identity")
                               for reader in readers]
                while True:
                    rows = [cursor.fetchone() for cursor in inventories]
                    if rows[0] != rows[1]:
                        raise ValueError("semantic scope identities or parent identities differ between parties")
                    if rows[0] is None:
                        break
                _match(parties)
                depth = _depths(parties, pages, directory)
                reports.append(dict(query=key[0], phase=key[1], logical_rounds=max(depth),
                                    send_flight_depths=depth, party_sent_bytes=[phase.sent for phase in parties],
                                    party_received_bytes=[phase.received for phase in parties]))
                count += sum(phase.events.count for phase in parties)
                for phase in parties:
                    phase.close()
            if expected and (observed != set(expected[0]) or observed != set(expected[1])):
                raise ValueError("trace phase set differs from phase metrics")
        finally:
            for store in list(pages.stores):
                store.close()
            for reader in readers:
                reader.close()
    return {"status": "passed", "definition": DEFINITION, "events": count,
            "phases": reports, "scopes": [],
            "scope_rounds_omitted": "phase-only bounded-memory analysis"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--include-scopes", action="store_true")
    args = parser.parse_args()
    result = analyze_pair(args.server, args.client, include_scopes=args.include_scopes,
                          temp_directory=args.temp_directory)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
