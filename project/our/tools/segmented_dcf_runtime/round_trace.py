#!/usr/bin/env python3
"""Count paired send flights on the traced protocol dependency path."""
import argparse
import collections
import gzip
import json
from array import array
from pathlib import Path


DEFINITION = ("two-party causal send flights: coalesce consecutive local sends; "
              "match receive byte intervals; preserve fork/join dependencies; "
              "rounds=max(A send-flight depth, B send-flight depth)")
ATOMIC_PHASES = {"application_io", "model_prepare", "setup", "offline", "input", "ready", "online", "output", "teardown"}
KINDS = {kind: index for index, kind in enumerate(("send", "recv", "fork", "start", "end", "join",
                                                  "phase_begin", "phase_end", "scope_begin", "scope_end"))}
SEND, RECV, FORK, START, END, JOIN, PHASE_BEGIN, PHASE_END, SCOPE_BEGIN, SCOPE_END = range(10)
NONE = 2 ** 32 - 1


class _Phase:
    """Store one phase without retaining JSON dictionaries or repeated strings."""

    def __init__(self, event):
        self.first_id = event["id"]
        self.key = event["query"], event["phase"]
        self.kind = bytearray()
        self.fields = {name: array("Q") for name in
                       ("actor", "prev", "dependency", "channel", "offset", "length", "scope")}
        self.semantic, self.parent_scope = {}, {}

    def __len__(self):
        return len(self.kind)

    def append(self, event):
        index = len(self)
        self.kind.append(KINDS[event["type"]])
        for name, values in self.fields.items():
            value = event.get(name, 0)
            if not isinstance(value, int) or value < 0 or value >= 2 ** 64:
                raise ValueError("invalid trace %s" % name)
            if name == "scope" and type(value) is not int:
                raise ValueError("invalid semantic scope identifier")
            values.append(value)
        if event["type"] == "scope_begin":
            self.semantic[index] = event.get("semantic", {})
            self.parent_scope[index] = event.get("parent_scope")


def _phase_stream(path, party):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        if json.loads(next(stream, "null")) != {"type": "meta", "version": 1, "party": party}:
            raise ValueError("invalid trace metadata for party %d" % party)
        count, chunk, active_phase = 0, None, None
        predecessors, frontiers = {}, {}
        complete = False
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "complete":
                if complete or chunk is not None or event.get("events") != count:
                    raise ValueError("trace footer event count or phase mismatch")
                complete = True
                continue
            if complete:
                raise ValueError("events follow the trace completion footer")
            count += 1
            if event.get("id") != count:
                raise ValueError("trace event sequence mismatch")
            if kind not in KINDS:
                raise ValueError("unknown trace event type")
            actor = event.get("actor")
            if not isinstance(actor, int) or actor < 1:
                raise ValueError("invalid trace actor")
            if not isinstance(event.get("query"), int) or not isinstance(event.get("phase"), str):
                raise ValueError("invalid trace phase")
            if event.get("prev", 0) != predecessors.get(actor, 0):
                raise ValueError("actor predecessor is not consecutive")
            predecessors[actor] = count
            phase = event["query"], event["phase"]
            if kind == "phase_begin":
                if actor != 1 or active_phase is not None or event["phase"] not in ATOMIC_PHASES:
                    raise ValueError("invalid phase begin boundary")
                active_phase = phase
                chunk = _Phase(event)
            if active_phase != phase:
                raise ValueError("event lies outside its phase boundaries")
            if kind in ("send", "recv"):
                key = kind, event["channel"]
                if event["offset"] != frontiers.get(key, 0):
                    raise ValueError("communication stream has a gap or overlap")
                if not isinstance(event["length"], int) or event["length"] <= 0:
                    raise ValueError("invalid communication byte interval")
                frontiers[key] = event["offset"] + event["length"]
            chunk.append(event)
            if kind == "phase_end":
                if actor != 1:
                    raise ValueError("phase end must belong to the root actor")
                active_phase = None
                yield chunk
                chunk = None
        if not complete:
            raise ValueError("trace did not complete for party %d" % party)


def _expected_rows(party):
    rows = party.get("phases", party) if isinstance(party, dict) else party
    if isinstance(rows, dict):
        rows = list(rows.values())
    return {(row["query"], row["phase"]): row for row in rows if row["phase"] in ATOMIC_PHASES}


def analyze_pair(server_trace, client_trace, expected_phases=None, scope_groups=None):
    """Return qualified rounds, optionally also for unions of semantic scopes.

    Each scope_groups entry selects source_label, optional stages/phase, and
    group_by semantic fields, then assigns its label to the resulting reports.
    """
    try:
        return _analyze_pair(server_trace, client_trace, expected_phases, scope_groups)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {"status": "failed", "reason": str(error), "definition": DEFINITION,
                "logical_rounds": None, "phases": []}


def _analyze_pair(server_trace, client_trace, expected_phases, scope_groups=None):
    streams = [_phase_stream(server_trace, 1), _phase_stream(client_trace, 2)]
    expected = [_expected_rows(party) for party in expected_phases] if expected_phases else None
    reports, scopes, count, observed = [], [], 0, set()
    while True:
        parties = [next(stream, None) for stream in streams]
        if parties == [None, None]:
            break
        if None in parties:
            raise ValueError("parties have different phase sets")
        key = parties[0].key
        if key != parties[1].key or key in observed:
            raise ValueError("parties have different or duplicated phase sets")
        observed.add(key)
        if expected and (key not in expected[0] or key not in expected[1]):
            raise ValueError("trace phase set differs from phase metrics")
        phase_expected = [[expected[party][key]] for party in (0, 1)] if expected else None
        result = _analyze_phase(parties, phase_expected, scope_groups)
        reports.extend(result["phases"])
        scopes.extend(result["scopes"])
        count += result["events"]
        del parties, result
    if expected and (observed != set(expected[0]) or observed != set(expected[1])):
        raise ValueError("trace phase set differs from phase metrics")
    return {"status": "passed", "definition": DEFINITION, "events": count,
            "phases": reports, "scopes": scopes}


class _Graph:
    """Flat adjacency lists keep memory proportional to compact events and edges."""

    def __init__(self, parties):
        self.parties = parties
        self.split = len(parties[0])
        self.size = self.split + len(parties[1])
        if self.size >= NONE:
            raise ValueError("trace phase has too many events")
        self.kind = parties[0].kind + parties[1].kind
        self.dep_head = array("I", [NONE]) * self.size
        self.succ_head = array("I", [NONE]) * self.size
        self.dep_target, self.dep_next = array("I"), array("I")
        self.succ_target, self.succ_next = array("I"), array("I")
        self.waiting = array("I", [0]) * self.size

    def add_edge(self, node, dependency):
        edge = len(self.dep_target)
        if edge >= NONE:
            raise ValueError("trace phase has too many dependencies")
        self.dep_target.append(dependency)
        self.dep_next.append(self.dep_head[node])
        self.dep_head[node] = edge
        self.succ_target.append(node)
        self.succ_next.append(self.succ_head[dependency])
        self.succ_head[dependency] = edge
        self.waiting[node] += 1

    def dependencies(self, node):
        edge = self.dep_head[node]
        while edge != NONE:
            yield self.dep_target[edge]
            edge = self.dep_next[edge]

    def field(self, node, name):
        party = int(node >= self.split)
        return self.parties[party].fields[name][node - (self.split if party else 0)]


def _advance(graph, node, depths, last, selected=None):
    party = int(node >= graph.split)
    clock0, clock1, local = 0, 0, []
    for dependency in graph.dependencies(node):
        left, right = depths[0][dependency], depths[1][dependency]
        clock0, clock1 = max(clock0, left), max(clock1, right)
        if int(dependency >= graph.split) == party:
            local.append((left, right, last[dependency]))
    frontier = [state for state in local if not any(
        other[0] >= state[0] and other[1] >= state[1] and other[:2] != state[:2]
        for other in local)]
    last_send = bool(frontier) and all(state[2] for state in frontier)
    kind = graph.kind[node]
    if kind == PHASE_BEGIN:
        clock0, clock1, last_send = 0, 0, False
    elif kind == SEND:
        if selected is None or selected[node]:
            if not last_send:
                if party:
                    clock1 += 1
                else:
                    clock0 += 1
            last_send = True
        else:
            last_send = False
    elif kind == RECV:
        last_send = False
    depths[0][node], depths[1][node], last[node] = clock0, clock1, last_send


def _analyze_phase(parties, expected_phases, scope_groups=None):
    graph = _Graph(parties)
    flows = collections.defaultdict(lambda: array("I"))
    sent, received = [0, 0], [0, 0]
    for party, events in enumerate(parties):
        base = graph.split if party else 0
        fields = events.fields
        actor_last, actor_started = {1: fields["prev"][0]}, {1}
        ended_actors, starts, ends, joins = set(), {}, {}, {}
        for index, kind in enumerate(events.kind):
            node, identifier, actor = base + index, events.first_id + index, fields["actor"][index]
            dependency = fields["dependency"][index]
            if actor in ended_actors:
                raise ValueError("actor used after completion")
            if kind == START:
                if actor in actor_started:
                    raise ValueError("actor started twice")
                actor_started.add(actor)
                source = dependency - events.first_id
                if not 0 <= source < index or events.kind[source] != FORK:
                    raise ValueError("actor start has no matching fork")
                starts[actor] = fields["actor"][source]
            if actor not in actor_started:
                raise ValueError("communication actor has no traced fork/start")
            previous = fields["prev"][index]
            if previous != actor_last.get(actor, 0):
                raise ValueError("actor predecessor is not consecutive")
            actor_last[actor] = identifier
            for value in (previous, dependency if dependency != previous else 0):
                if not value:
                    continue
                if value < events.first_id and value == previous:
                    continue
                if not events.first_id <= value < events.first_id + len(events):
                    raise ValueError("missing local dependency")
                graph.add_edge(node, base + value - events.first_id)
            if kind == END:
                if actor == 1 or actor not in starts:
                    raise ValueError("invalid child actor completion")
                ends[identifier] = actor
                ended_actors.add(actor)
            elif kind == JOIN:
                child = ends.get(dependency)
                if child is None or starts[child] != actor or dependency in joins:
                    raise ValueError("join has no matching owned child completion")
                joins[dependency] = actor
            if kind in (SEND, RECV):
                length = fields["length"][index]
                if not length:
                    raise ValueError("invalid communication byte interval")
                (sent if kind == SEND else received)[party] += length
                flows[party, kind, fields["channel"][index]].append(node)
        if len(starts) != len(ends) or set(ends) != set(joins):
            raise ValueError("missing phase end or child fork/join completion")
    for (party, kind, channel), intervals in flows.items():
        if kind == RECV:
            _match_flow(flows.get((1 - party, SEND, channel), ()), intervals, graph)
        elif (1 - party, RECV, channel) not in flows:
            raise ValueError("unmatched send bytes on channel %s" % channel)
    del flows
    depths = [array("Q", [0]) * graph.size for _ in (0, 1)]
    last = bytearray(graph.size)
    order = array("I", (node for node, count in enumerate(graph.waiting) if count == 0))
    cursor = 0
    while cursor < len(order):
        node = order[cursor]
        _advance(graph, node, depths, last)
        cursor += 1
        edge = graph.succ_head[node]
        while edge != NONE:
            successor = graph.succ_target[edge]
            graph.waiting[successor] -= 1
            if not graph.waiting[successor]:
                order.append(successor)
            edge = graph.succ_next[edge]
    if cursor != graph.size:
        raise ValueError("communication dependency cycle")
    depth = [max(values, default=0) for values in depths]
    del depths, last, graph.succ_head, graph.succ_target, graph.succ_next, graph.waiting
    expected = [_expected_rows(party) for party in expected_phases] if expected_phases else None
    query, phase = parties[0].key
    if expected:
        for party in (0, 1):
            row = expected[party].get((query, phase))
            if row is None or row["sent_bytes"] != sent[party] or row["received_bytes"] != received[party]:
                raise ValueError("trace bytes differ from phase metric: %s %s party %d" % (query, phase, party + 1))
    reports = [dict(query=query, phase=phase, logical_rounds=max(depth), send_flight_depths=depth,
                    party_sent_bytes=sent, party_received_bytes=received)]
    scopes = _analyze_scopes(parties, graph, order, scope_groups)
    return {"status": "passed", "definition": DEFINITION, "events": graph.size,
            "phases": reports, "scopes": scopes}


def _scope_inventory(events, base=0):
    scopes, active, forks, occurrences = {}, {1: ()}, {}, collections.Counter()
    fields = ("label", "query", "layer", "slot", "call", "kind", "stage")
    for index, kind in enumerate(events.kind):
        actor = events.fields["actor"][index]
        if kind == START:
            active[actor] = forks[events.fields["dependency"][index]]
        stack = active[actor]
        current = stack[-1] if stack else 0
        identifier = events.fields["scope"][index]
        if kind == SCOPE_BEGIN:
            semantic = events.semantic[index]
            if (identifier == 0 or identifier in scopes or events.parent_scope[index] != current or
                    set(semantic) != set(fields) or semantic.get("label") not in
                    ("activation", "folded_call", "activation_stage", "folded_prepare") or
                    any(type(semantic[field]) is not int for field in fields[1:]) or
                    semantic["query"] != events.key[0]):
                raise ValueError("invalid or duplicated semantic scope boundary")
            base_identity = tuple(semantic[field] for field in fields)
            instance = occurrences[base_identity]
            occurrences[base_identity] += 1
            scopes[identifier] = dict(identity=base_identity + (instance,), semantic=semantic, owner=actor,
                                      instance=instance, parent=current, begin=base + index, end=None, keys=array("I"))
            stack = stack + (identifier,)
            active[actor] = stack
        elif identifier != current:
            raise ValueError("semantic scope does not follow its actor or inherited fork")
        for scope in stack:
            if scopes[scope]["end"] is not None:
                raise ValueError("actor used a semantic scope after its end boundary")
            scopes[scope]["keys"].append(base + index)
        if kind == FORK:
            forks[events.first_id + index] = stack
        elif kind == SCOPE_END:
            if not stack or scopes[identifier]["owner"] != actor or scopes[identifier]["end"] is not None:
                raise ValueError("unowned or duplicated semantic scope end")
            scopes[identifier]["end"] = base + index
            active[actor] = stack[:-1]
        elif kind == END:
            if any(scopes[scope]["owner"] == actor for scope in stack):
                raise ValueError("actor completed with an unfinished semantic scope")
            del active[actor]
        elif kind == PHASE_END and stack:
            raise ValueError("phase completed with an unfinished semantic scope")
    if any(row["end"] is None for row in scopes.values()):
        raise ValueError("semantic scope is missing its end boundary")
    return {row["identity"]: dict(row, parent_identity=scopes[row["parent"]]["identity"]
                                  if row["parent"] else None) for row in scopes.values()}


def _scope_depth(graph, order, bounds, selected):
    # External actors in the boundary window can carry dependencies between
    # selected sends, even though external sends have zero weight.
    depths = [array("Q", [0]) * graph.size for _ in (0, 1)]
    last = bytearray(graph.size)
    result = [0, 0]
    for node in order:
        lower, upper = bounds[int(node >= graph.split)]
        if lower <= node <= upper:
            _advance(graph, node, depths, last, selected)
            if selected[node]:
                result[0] = max(result[0], depths[0][node])
                result[1] = max(result[1], depths[1][node])
    return result


def _selected_report(graph, order, pairs):
    selected = bytearray(graph.size)
    bounds = [[graph.size, -1], [graph.size, -1]]
    for pair in pairs:
        for party, row in enumerate(pair):
            bounds[party][0] = min(bounds[party][0], row["begin"])
            bounds[party][1] = max(bounds[party][1], row["end"])
            for node in row["keys"]:
                selected[node] = 1
    sent, received, actors = [0, 0], [0, 0], [set(), set()]
    for party, (lower, upper) in enumerate(bounds):
        for node in range(lower, upper + 1):
            if not selected[node]:
                continue
            actors[party].add(graph.field(node, "actor"))
            kind = graph.kind[node]
            if kind == SEND:
                sent[party] += graph.field(node, "length")
            elif kind == RECV:
                received[party] += graph.field(node, "length")
                if any(int(dependency >= graph.split) != party and not selected[dependency]
                       for dependency in graph.dependencies(node)):
                    raise ValueError("semantic scope receive is owned by an external send")
    if sent != received[::-1]:
        raise ValueError("semantic scope bytes differ between parties")
    depth = _scope_depth(graph, order, bounds, selected)
    return dict(logical_rounds=max(depth), actors_per_party=[len(value) for value in actors],
                send_flight_depths=depth, party_sent_bytes=sent, party_received_bytes=received,
                definition="selected-scope send flights on the complete phase dependency DAG")


def _analyze_scopes(parties, graph, order, scope_groups=None):
    inventories = [_scope_inventory(events, graph.split if party else 0)
                   for party, events in enumerate(parties)]
    if set(inventories[0]) != set(inventories[1]):
        raise ValueError("semantic scope identities differ between parties")
    reports, pairs = [], []
    for identity in sorted(inventories[0]):
        pair = [inventory[identity] for inventory in inventories]
        if pair[0]["parent_identity"] != pair[1]["parent_identity"]:
            raise ValueError("semantic scope parent identities differ between parties")
        pairs.append(pair)
        reports.append(dict(pair[0]["semantic"], phase=parties[0].key[1], instance=pair[0]["instance"],
                            **_selected_report(graph, order, [pair])))
    for specification in scope_groups or ():
        if specification.get("phase", parties[0].key[1]) != parties[0].key[1]:
            continue
        groups = collections.defaultdict(list)
        group_by = specification.get("group_by", ())
        for pair in pairs:
            semantic = pair[0]["semantic"]
            if semantic["label"] != specification["source_label"]:
                continue
            if "stages" in specification and semantic["stage"] not in specification["stages"]:
                continue
            groups[tuple(semantic[field] for field in group_by)].append(pair)
        for identity, selected_pairs in sorted(groups.items()):
            report = dict(zip(group_by, identity))
            report.update(label=specification["label"], query=parties[0].key[0], phase=parties[0].key[1])
            report.update(_selected_report(graph, order, selected_pairs))
            reports.append(report)
    return reports


def _match_flow(sends, receives, graph):
    if not sends:
        raise ValueError("unmatched receive bytes")

    def start(node):
        return graph.field(node, "offset")

    def end(node):
        return start(node) + graph.field(node, "length")

    if start(sends[0]) != start(receives[0]) or end(sends[-1]) != end(receives[-1]):
        raise ValueError("unmatched communication byte totals")
    index = 0
    for receive in receives:
        lower, upper = start(receive), end(receive)
        while end(sends[index]) <= lower:
            index += 1
        cursor = index
        while cursor < len(sends) and start(sends[cursor]) < upper:
            graph.add_edge(receive, sends[cursor])
            cursor += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_pair(args.server, args.client)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["status"] == "passed" else 2)
