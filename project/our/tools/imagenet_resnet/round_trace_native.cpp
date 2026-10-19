// Phase-only counterpart of round_trace_stream.py. Event storage stays on disk.
#include "vendor/nlohmann/json.hpp"
#include <sqlite3.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
using U64 = uint64_t;
constexpr U64 MISSING = std::numeric_limits<U64>::max();
constexpr size_t PAGE_SIZE = 65536;
constexpr size_t PAGE_CAPACITY = 128;
constexpr const char* DEFINITION =
    "two-party causal send flights: coalesce consecutive local sends; "
    "match receive byte intervals; preserve fork/join dependencies; "
    "rounds=max(A send-flight depth, B send-flight depth)";
enum Kind : U64 { SEND, RECV, FORK, START, END, JOIN, PHASE_BEGIN, PHASE_END, SCOPE_BEGIN, SCOPE_END };

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void check(bool condition, const std::string& message) { if (!condition) fail(message); }
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

class TempDirectory {
public:
    std::filesystem::path path;
    explicit TempDirectory(const std::string& parent) {
        auto base = parent.empty() ? std::filesystem::temp_directory_path() : std::filesystem::path(parent);
        std::string pattern = (base / "resnet-native-rounds-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end()); writable.push_back(0);
        char* result = mkdtemp(writable.data());
        check(result, "cannot create native trace temporary directory"); path = result;
    }
    ~TempDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

class Pages {
    struct Key {
        int file;
        U64 page;
        bool operator==(const Key& other) const { return file == other.file && page == other.page; }
    };
    struct Hash {
        size_t operator()(const Key& key) const {
            return std::hash<U64>{}(key.page ^ (U64(key.file) * 0x9e3779b97f4a7c15ULL));
        }
    };
    struct Page { Key key; bool dirty = false; std::array<unsigned char, PAGE_SIZE> data; };
    std::list<Page> rows;
    std::unordered_map<Key, std::list<Page>::iterator, Hash> index;

    static void flush(Page& row) {
        if (!row.dirty) return;
        size_t done = 0;
        while (done < PAGE_SIZE) {
            ssize_t count = pwrite(row.key.file, row.data.data() + done, PAGE_SIZE - done,
                                   off_t(row.key.page * PAGE_SIZE + done));
            if (count < 0 && errno == EINTR) continue;
            check(count > 0, "temporary trace record write failed"); done += size_t(count);
        }
        row.dirty = false;
    }
public:
    Pages() { index.reserve(PAGE_CAPACITY * 2); }
    unsigned char* get(int file, U64 page, bool dirty) {
        check(page <= U64(std::numeric_limits<off_t>::max()) / PAGE_SIZE, "trace temporary file offset overflow");
        Key key{file, page};
        auto found = index.find(key);
        std::list<Page>::iterator row;
        if (found != index.end()) {
            row = found->second;
        } else {
            if (rows.size() == PAGE_CAPACITY) {
                row = rows.begin(); flush(*row); index.erase(row->key);
            } else {
                rows.emplace_back(); row = std::prev(rows.end());
            }
            row->key = key; row->dirty = false;
            size_t done = 0;
            while (done < PAGE_SIZE) {
                ssize_t count = pread(file, row->data.data() + done, PAGE_SIZE - done,
                                      off_t(page * PAGE_SIZE + done));
                if (count < 0 && errno == EINTR) continue;
                check(count >= 0, "temporary trace record read failed");
                if (!count) break;
                done += size_t(count);
            }
            std::memset(row->data.data() + done, 0, PAGE_SIZE - done);
            index.emplace(key, row);
        }
        rows.splice(rows.end(), rows, row);
        row->dirty = row->dirty || dirty;
        return row->data.data();
    }
    void close(int file) {
        for (auto row = rows.begin(); row != rows.end();) {
            if (row->key.file == file) { index.erase(row->key); row = rows.erase(row); }
            else ++row;
        }
    }
};

template<size_t N> class Records {
    Pages& pages;
    int file = -1;
    static constexpr U64 PER_PAGE = PAGE_SIZE / (N * sizeof(U64));
public:
    U64 count = 0;
    Records(Pages& cache, const std::filesystem::path& directory) : pages(cache) {
        std::string pattern = (directory / "records-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end()); writable.push_back(0);
        file = mkstemp(writable.data()); check(file >= 0, "cannot create native trace record store");
        unlink(writable.data());
    }
    Records(const Records&) = delete;
    Records& operator=(const Records&) = delete;
    ~Records() { if (file >= 0) { pages.close(file); ::close(file); } }
    std::array<U64, N> get(U64 item) {
        check(item < count, "missing local dependency");
        auto* data = pages.get(file, item / PER_PAGE, false) + (item % PER_PAGE) * N * sizeof(U64);
        std::array<U64, N> result; std::memcpy(result.data(), data, N * sizeof(U64)); return result;
    }
    void put(U64 item, const std::array<U64, N>& value) {
        auto* data = pages.get(file, item / PER_PAGE, true) + (item % PER_PAGE) * N * sizeof(U64);
        std::memcpy(data, value.data(), N * sizeof(U64));
    }
    void append(const std::array<U64, N>& value) { put(count, value); ++count; }
};

class Database {
    sqlite3* db = nullptr;
    sqlite3_stmt *select = nullptr, *insert = nullptr;
public:
    explicit Database(const std::filesystem::path& path) {
        check(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "cannot open trace metadata database");
        exec("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-2048; PRAGMA temp_store=FILE;"
             "CREATE TABLE kv(bucket INTEGER,key BLOB,value BLOB,PRIMARY KEY(bucket,key)) WITHOUT ROWID;"
             "CREATE TABLE occurrences(identity TEXT PRIMARY KEY,count INTEGER) WITHOUT ROWID;"
             "CREATE TABLE identities(identity TEXT PRIMARY KEY,parent TEXT) WITHOUT ROWID; BEGIN;");
        select = prepare("SELECT value FROM kv WHERE bucket=? AND key=?");
        insert = prepare("INSERT OR REPLACE INTO kv VALUES(?,?,?)");
    }
    ~Database() { sqlite3_finalize(select); sqlite3_finalize(insert); if (db) sqlite3_close(db); }
    void exec(const char* sql) {
        char* error = nullptr;
        int result = sqlite3_exec(db, sql, nullptr, nullptr, &error);
        std::string message = error ? error : "trace metadata SQL failed"; sqlite3_free(error);
        check(result == SQLITE_OK, message);
    }
    sqlite3_stmt* prepare(const char* sql) {
        sqlite3_stmt* result = nullptr;
        check(sqlite3_prepare_v2(db, sql, -1, &result, nullptr) == SQLITE_OK, sqlite3_errmsg(db)); return result;
    }
    void step_done(sqlite3_stmt* statement) {
        int result = sqlite3_step(statement); check(result == SQLITE_DONE, sqlite3_errmsg(db));
    }
    bool get(int bucket, U64 key, std::string& value) {
        sqlite3_reset(select); sqlite3_clear_bindings(select);
        sqlite3_bind_int(select, 1, bucket); sqlite3_bind_blob(select, 2, &key, sizeof(key), SQLITE_TRANSIENT);
        int result = sqlite3_step(select);
        if (result == SQLITE_DONE) return false;
        check(result == SQLITE_ROW, sqlite3_errmsg(db));
        const auto* blob = static_cast<const char*>(sqlite3_column_blob(select, 0));
        value.assign(blob, sqlite3_column_bytes(select, 0)); return true;
    }
    void put(int bucket, U64 key, const std::string& value) {
        sqlite3_reset(insert); sqlite3_clear_bindings(insert);
        sqlite3_bind_int(insert, 1, bucket); sqlite3_bind_blob(insert, 2, &key, sizeof(key), SQLITE_TRANSIENT);
        sqlite3_bind_blob(insert, 3, value.data(), int(value.size()), SQLITE_TRANSIENT); step_done(insert);
    }
    U64 occurrence(const std::string& identity) {
        auto* query = prepare("SELECT count FROM occurrences WHERE identity=?");
        sqlite3_bind_text(query, 1, identity.c_str(), int(identity.size()), SQLITE_TRANSIENT);
        int result = sqlite3_step(query); check(result == SQLITE_ROW || result == SQLITE_DONE, sqlite3_errmsg(db));
        U64 count = result == SQLITE_ROW ? U64(sqlite3_column_int64(query, 0)) : 0;
        sqlite3_finalize(query);
        auto* write = prepare("INSERT OR REPLACE INTO occurrences VALUES(?,?)");
        sqlite3_bind_text(write, 1, identity.c_str(), int(identity.size()), SQLITE_TRANSIENT);
        sqlite3_bind_int64(write, 2, sqlite3_int64(count + 1)); step_done(write); sqlite3_finalize(write); return count;
    }
    void identity(const std::string& identity, const std::string* parent) {
        auto* write = prepare("INSERT INTO identities VALUES(?,?)");
        sqlite3_bind_text(write, 1, identity.c_str(), int(identity.size()), SQLITE_TRANSIENT);
        if (parent) sqlite3_bind_text(write, 2, parent->c_str(), int(parent->size()), SQLITE_TRANSIENT);
        else sqlite3_bind_null(write, 2);
        step_done(write); sqlite3_finalize(write);
    }
};

using Actor = std::array<U64, 6>; // Last event, phase, ended, owner, scope, joined.
struct Scope { U64 owner = 0, parent = 0, open = 0; std::string identity; };
std::string encode(const Actor& value) { return std::string(reinterpret_cast<const char*>(value.data()), sizeof(value)); }
Actor decode_actor(const std::string& bytes) {
    check(bytes.size() == sizeof(Actor), "invalid actor metadata record"); Actor result;
    std::memcpy(result.data(), bytes.data(), sizeof(result)); return result;
}
std::string encode(const Scope& value) {
    std::array<U64, 3> prefix{value.owner, value.parent, value.open};
    return std::string(reinterpret_cast<const char*>(prefix.data()), sizeof(prefix)) + value.identity;
}
Scope decode_scope(const std::string& bytes) {
    check(bytes.size() >= sizeof(U64) * 3, "invalid scope metadata record"); std::array<U64, 3> prefix;
    std::memcpy(prefix.data(), bytes.data(), sizeof(prefix));
    return {prefix[0], prefix[1], prefix[2], bytes.substr(sizeof(prefix))};
}

template<class T> class MetadataCache {
    struct Entry { T value; bool dirty; typename std::list<U64>::iterator position; };
    Database& db;
    int bucket;
    std::list<U64> order;
    std::unordered_map<U64, Entry> entries;
    T (*decode)(const std::string&);
    void trim() {
        if (entries.size() <= 2048) return;
        U64 key = order.front(); auto found = entries.find(key);
        if (found->second.dirty) db.put(bucket, key, encode(found->second.value));
        order.pop_front(); entries.erase(found);
    }
public:
    MetadataCache(Database& database, int kind, T (*decoder)(const std::string&)) : db(database), bucket(kind), decode(decoder) {
        entries.reserve(4096);
    }
    bool get(U64 key, T& value) {
        auto found = entries.find(key);
        if (found == entries.end()) {
            std::string bytes; if (!db.get(bucket, key, bytes)) return false;
            order.push_back(key);
            entries.emplace(key, Entry{decode(bytes), false, std::prev(order.end())}); trim(); found = entries.find(key);
        } else order.splice(order.end(), order, found->second.position);
        value = found->second.value; return true;
    }
    void put(U64 key, const T& value) {
        auto found = entries.find(key);
        if (found == entries.end()) {
            order.push_back(key); entries.emplace(key, Entry{value, true, std::prev(order.end())}); trim();
        } else {
            found->second.value = value; found->second.dirty = true;
            order.splice(order.end(), order, found->second.position);
        }
    }
    void clear() { entries.clear(); order.clear(); }
};

struct Metadata {
    Database db;
    MetadataCache<Actor> actors;
    MetadataCache<Scope> scopes;
    explicit Metadata(const std::filesystem::path& path) : db(path), actors(db, 1, decode_actor), scopes(db, 2, decode_scope) {}
    void reset_phase() {
        scopes.clear(); db.exec("DELETE FROM kv WHERE bucket=2; DELETE FROM occurrences; DELETE FROM identities;");
    }
};

struct Phase {
    U64 first;
    Json query;
    std::string name;
    Records<7> events;
    Records<4> states;
    std::map<std::pair<U64, U64>, std::unique_ptr<Records<3>>> flows;
    Pages& pages;
    const std::filesystem::path& directory;
    U64 sent = 0, received = 0, children = 0, ends = 0, joins = 0, open_scopes = 0;
    Phase(U64 identifier, const Json& q, std::string phase, Pages& cache, const std::filesystem::path& path)
        : first(identifier), query(q), name(std::move(phase)), events(cache, path), states(cache, path), pages(cache), directory(path) {}
    Records<3>& flow(U64 kind, U64 channel) {
        auto key = std::make_pair(kind, channel); auto found = flows.find(key);
        if (found == flows.end()) found = flows.emplace(key, std::make_unique<Records<3>>(pages, directory)).first;
        return *found->second;
    }
};

bool integer(const Json& value) { return value.is_number_integer() || value.is_number_unsigned(); }
U64 number(const Json& event, const char* name, bool required = false) {
    auto found = event.find(name);
    if (found == event.end()) { check(!required, std::string("missing trace ") + name); return 0; }
    check(integer(*found), std::string("invalid trace ") + name);
    check(!found->is_number_integer() || found->is_number_unsigned() || found->get<int64_t>() >= 0,
          std::string("invalid trace ") + name);
    return found->get<U64>();
}

enum Field { ID, ACTOR, PREV, DEPENDENCY, CHANNEL, OFFSET, LENGTH, SCOPE, FOOTER_EVENTS,
             TYPE, PHASE, QUERY, OTHER };
constexpr std::array<const char*, 9> NUMBER_NAMES{"id", "actor", "prev", "dependency", "channel", "offset", "length", "scope", "events"};
struct Event {
    std::array<U64, 9> numbers{};
    unsigned present = 0, invalid = 0;
    std::string type, phase;
    Json query;
    bool type_valid = false, phase_valid = false, query_valid = false;
    U64 number(Field field, bool required = false) const {
        unsigned mask = 1u << field;
        if ((invalid & mask) || (required && !(present & mask))) fail(std::string("invalid trace ") + NUMBER_NAMES[field]);
        return numbers[field];
    }
};

// SAX avoids allocating a JSON object for every communication event. Unknown
// members still pass through the JSON parser; scope metadata uses the DOM below.
class EventSax : public nlohmann::json_sax<Json> {
    Field field = OTHER;
    size_t depth = 0;
    bool root_object = false;
    void wrong() {
        if (field <= FOOTER_EVENTS) { event.present |= 1u << field; event.invalid |= 1u << field; }
        else if (field == TYPE) event.type_valid = false;
        else if (field == PHASE) event.phase_valid = false;
        else if (field == QUERY) event.query_valid = false;
    }
    bool numeric(U64 value, bool nonnegative, const Json& query) {
        if (depth != 1) return true;
        if (field <= FOOTER_EVENTS) {
            unsigned mask = 1u << field; event.present |= mask;
            if (nonnegative) { event.numbers[field] = value; event.invalid &= ~mask; }
            else event.invalid |= mask;
        } else if (field == QUERY) { event.query = query; event.query_valid = true; }
        else if (field != OTHER) wrong();
        return true;
    }
public:
    Event event;
    std::string error;
    bool null() override { if (depth == 1) wrong(); return true; }
    bool boolean(bool) override { if (depth == 1) wrong(); return true; }
    bool number_integer(number_integer_t value) override { return numeric(U64(value), value >= 0, Json(value)); }
    bool number_unsigned(number_unsigned_t value) override { return numeric(value, true, Json(value)); }
    bool number_float(number_float_t, const string_t&) override { if (depth == 1) wrong(); return true; }
    bool string(string_t& value) override {
        if (depth != 1) return true;
        if (field == TYPE) { event.type = value; event.type_valid = true; }
        else if (field == PHASE) { event.phase = value; event.phase_valid = true; }
        else if (field != OTHER) wrong();
        return true;
    }
    bool binary(binary_t&) override { if (depth == 1) wrong(); return true; }
    bool start_object(std::size_t) override {
        if (!depth) root_object = true; else if (depth == 1) wrong();
        ++depth; return true;
    }
    bool key(string_t& value) override {
        if (depth != 1) return true;
        field = OTHER;
        switch (value.size()) {
        case 2: if (value == "id") field = ID; break;
        case 4: if (value == "type") field = TYPE; else if (value == "prev") field = PREV; break;
        case 5:
            if (value == "actor") field = ACTOR;
            else if (value == "phase") field = PHASE;
            else if (value == "query") field = QUERY;
            else if (value == "scope") field = SCOPE;
            break;
        case 6:
            if (value == "offset") field = OFFSET;
            else if (value == "length") field = LENGTH;
            else if (value == "events") field = FOOTER_EVENTS;
            break;
        case 7: if (value == "channel") field = CHANNEL; break;
        case 10: if (value == "dependency") field = DEPENDENCY; break;
        }
        return true;
    }
    bool end_object() override { --depth; return true; }
    bool start_array(std::size_t) override { if (depth == 1) wrong(); ++depth; return true; }
    bool end_array() override { --depth; return true; }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& exception) override {
        error = exception.what(); return false;
    }
    bool valid_root() const { return root_object && depth == 0; }
};

Event parse_event(const std::string& line) {
    EventSax parser;
    bool valid = Json::sax_parse(line, &parser);
    check(valid, parser.error.empty() ? "invalid trace JSON" : parser.error);
    check(parser.valid_root(), "trace event must be a JSON object"); return std::move(parser.event);
}

class Reader {
    std::ifstream stream;
    int party;
    Pages& pages;
    const std::filesystem::path& directory;
    std::map<std::pair<U64, U64>, U64> frontiers;
    U64 count = 0, phase_number = 0;
    bool complete = false;
    static U64 kind(const std::string& type) {
        static const std::map<std::string, U64> kinds{{"send", SEND}, {"recv", RECV}, {"fork", FORK},
            {"start", START}, {"end", END}, {"join", JOIN}, {"phase_begin", PHASE_BEGIN},
            {"phase_end", PHASE_END}, {"scope_begin", SCOPE_BEGIN}, {"scope_end", SCOPE_END}};
        auto found = kinds.find(type);
        check(found != kinds.end(), "unknown trace event type"); return found->second;
    }
    void append(Phase& phase, const Event& data, U64 type, const std::string& line) {
        U64 actor = data.numbers[ACTOR], previous = data.numbers[PREV], dependency = data.numbers[DEPENDENCY];
        Actor state{}; meta.actors.get(actor, state);
        check(previous == state[0], "actor predecessor is not consecutive");
        if (type == PHASE_BEGIN) state = {state[0], phase_number, 0, 0, 0, 0};
        else if (type == START) {
            check(state[1] != phase_number && !previous, "actor started twice or has a prior lifetime");
            check(dependency >= phase.first && dependency - phase.first < phase.events.count, "actor start has no matching fork");
            auto fork = phase.events.get(dependency - phase.first);
            check(fork[0] == FORK, "actor start has no matching fork");
            state = {0, phase_number, 0, fork[1], fork[4], 0}; ++phase.children;
        } else check(state[1] == phase_number, "communication actor has no traced fork/start");
        check(!state[2], "actor used after completion");
        U64 current = state[4], scope = data.numbers[SCOPE];
        if (type == SCOPE_BEGIN) {
            Json event = Json::parse(line);
            static const std::array<const char*, 7> fields{"label", "query", "layer", "slot", "call", "kind", "stage"};
            auto found = event.find("semantic");
            check(found != event.end() && found->is_object() && found->size() == fields.size(), "invalid or duplicated semantic scope boundary");
            const auto& semantic = *found;
            Scope existing;
            check(scope && !meta.scopes.get(scope, existing) && event.contains("parent_scope") &&
                  integer(event["parent_scope"]) && number(event, "parent_scope") == current,
                  "invalid or duplicated semantic scope boundary");
            for (const char* field : fields) check(semantic.contains(field), "invalid or duplicated semantic scope boundary");
            check(semantic["label"].is_string(), "invalid or duplicated semantic scope boundary");
            std::string label = semantic["label"].get<std::string>();
            check(label == "activation" || label == "folded_call" || label == "activation_stage" || label == "folded_prepare",
                  "invalid or duplicated semantic scope boundary");
            for (size_t index = 1; index < fields.size(); ++index)
                check(integer(semantic[fields[index]]), "invalid or duplicated semantic scope boundary");
            check(semantic["query"] == phase.query, "invalid or duplicated semantic scope boundary");
            Json identity = Json::array(); for (const char* field : fields) identity.push_back(semantic[field]);
            U64 occurrence = meta.db.occurrence(identity.dump()); identity.push_back(occurrence);
            Scope parent; std::string* parent_identity = nullptr;
            if (current) { check(meta.scopes.get(current, parent), "missing parent semantic scope"); parent_identity = &parent.identity; }
            Scope value{actor, current, 1, identity.dump()}; meta.scopes.put(scope, value);
            meta.db.identity(value.identity, parent_identity); state[4] = current = scope; ++phase.open_scopes;
        } else check(scope == current, "semantic scope does not follow its actor or inherited fork");
        for (U64 cursor = current; cursor;) {
            Scope value; check(meta.scopes.get(cursor, value) && value.open, "actor used a semantic scope after its end boundary");
            check(type != END || value.owner != actor, "actor completed with an unfinished semantic scope"); cursor = value.parent;
        }
        if (type == SCOPE_END) {
            Scope value; check(meta.scopes.get(scope, value) && value.owner == actor && value.open, "unowned or duplicated semantic scope end");
            value.open = 0; meta.scopes.put(scope, value); state[4] = value.parent; --phase.open_scopes;
        } else if (type == PHASE_END) check(!current, "phase completed with an unfinished semantic scope");
        if (type == END) {
            check(actor != 1 && state[3], "invalid child actor completion"); state[2] = 1; ++phase.ends;
        } else if (type == JOIN) {
            check(dependency >= phase.first && dependency - phase.first < phase.events.count, "join has no matching owned child completion");
            auto end = phase.events.get(dependency - phase.first); Actor child{};
            check(end[0] == END && meta.actors.get(end[1], child) && child[3] == actor && !child[5],
                  "join has no matching owned child completion");
            child[5] = 1; meta.actors.put(end[1], child); ++phase.joins;
        }
        state[0] = count; meta.actors.put(actor, state);
        std::array<U64, 2> local{};
        std::array<U64, 2> values{previous, dependency != previous ? dependency : 0};
        for (size_t index = 0; index < 2; ++index) {
            U64 value = values[index];
            if (!value || (value < phase.first && value == previous)) local[index] = MISSING;
            else { check(value >= phase.first, "missing local dependency"); local[index] = value - phase.first; }
        }
        phase.events.append({type, actor, local[0], local[1], scope, 0, 0});
        if (type == SEND || type == RECV) {
            U64 channel = data.number(CHANNEL, true), offset = data.number(OFFSET, true), length = data.number(LENGTH, true);
            auto key = std::make_pair(type, channel);
            check(offset == frontiers[key], "communication stream has a gap or overlap");
            check(length && length <= MISSING - offset, "invalid communication byte interval");
            frontiers[key] = offset + length; phase.flow(type, channel).append({offset, offset + length, phase.events.count - 1});
            U64& total = type == SEND ? phase.sent : phase.received;
            check(total <= MISSING - length, "trace phase byte total overflow"); total += length;
        }
    }
public:
    Metadata meta;
    Reader(const std::string& path, int role, Pages& cache, const std::filesystem::path& temp)
        : stream(path), party(role), pages(cache), directory(temp), meta(temp / ("party" + std::to_string(role) + ".sqlite")) {
        check(bool(stream), "cannot open trace input: " + path); std::string line;
        check(bool(std::getline(stream, line)), "invalid trace metadata for party " + std::to_string(party));
        check(Json::parse(line) == Json{{"type", "meta"}, {"version", 1}, {"party", party}},
              "invalid trace metadata for party " + std::to_string(party));
    }
    std::unique_ptr<Phase> next_phase() {
        std::unique_ptr<Phase> phase; std::string line;
        while (std::getline(stream, line)) {
            if (line.find_first_not_of(" \r\n\t") == std::string::npos) continue;
            Event event = parse_event(line);
            if (event.type_valid && event.type == "complete") {
                check(!complete && !phase && event.number(FOOTER_EVENTS, true) == count, "trace footer event count or phase mismatch");
                complete = true; continue;
            }
            check(!complete, "events follow the trace completion footer"); ++count;
            check(event.number(ID, true) == count, "trace event sequence mismatch");
            check(event.type_valid, "unknown trace event type"); U64 type = kind(event.type);
            for (Field field : {ACTOR, PREV, DEPENDENCY, CHANNEL, OFFSET, LENGTH, SCOPE}) event.number(field);
            U64 actor = event.numbers[ACTOR]; check(actor >= 1, "invalid trace actor or semantic scope identifier");
            check(event.query_valid && event.phase_valid, "invalid trace phase");
            const std::string& name = event.phase;
            if (type == PHASE_BEGIN) {
                static const std::set<std::string> phases{"application_io", "model_prepare", "setup", "offline", "input", "ready", "online", "output", "teardown"};
                check(actor == 1 && !phase && phases.count(name), "invalid phase begin boundary");
                ++phase_number; meta.reset_phase(); phase = std::make_unique<Phase>(count, event.query, name, pages, directory);
            }
            check(phase && phase->query == event.query && phase->name == name, "event lies outside its phase boundaries");
            append(*phase, event, type, line);
            if (type == PHASE_END) {
                check(actor == 1, "phase end must belong to the root actor");
                check(phase->children == phase->ends && phase->ends == phase->joins, "missing phase end or child fork/join completion");
                check(!phase->open_scopes, "semantic scope is missing its end boundary"); return phase;
            }
        }
        check(!stream.bad(), "trace input read failed");
        check(!phase && complete, "trace did not complete for party " + std::to_string(party)); return nullptr;
    }
};

void compare_identities(Reader& left, Reader& right) {
    auto* a = left.meta.db.prepare("SELECT identity,parent FROM identities ORDER BY identity");
    auto* b = right.meta.db.prepare("SELECT identity,parent FROM identities ORDER BY identity");
    try {
        while (true) {
            int x = sqlite3_step(a), y = sqlite3_step(b);
            check((x == SQLITE_ROW || x == SQLITE_DONE) && (y == SQLITE_ROW || y == SQLITE_DONE), "semantic identity database read failed");
            check(x == y, "semantic scope identities or parent identities differ between parties");
            if (x == SQLITE_DONE) break;
            for (int column = 0; column < 2; ++column) {
                bool nx = sqlite3_column_type(a, column) == SQLITE_NULL, ny = sqlite3_column_type(b, column) == SQLITE_NULL;
                check(nx == ny, "semantic scope identities or parent identities differ between parties");
                if (!nx) {
                    const char* sx = reinterpret_cast<const char*>(sqlite3_column_text(a, column));
                    const char* sy = reinterpret_cast<const char*>(sqlite3_column_text(b, column));
                    check(std::strcmp(sx, sy) == 0, "semantic scope identities or parent identities differ between parties");
                }
            }
        }
    } catch (...) { sqlite3_finalize(a); sqlite3_finalize(b); throw; }
    sqlite3_finalize(a); sqlite3_finalize(b);
}

void match(std::array<Phase*, 2> parties) {
    for (size_t party = 0; party < 2; ++party) {
        Phase& phase = *parties[party];
        for (auto& item : phase.flows) {
            U64 kind = item.first.first, channel = item.first.second;
            auto found = parties[1 - party]->flows.find({kind == SEND ? RECV : SEND, channel});
            check(found != parties[1 - party]->flows.end(), "unmatched send or receive bytes on channel " + std::to_string(channel));
            if (kind == SEND) continue;
            auto& flow = *item.second; auto& peer = *found->second;
            check(flow.get(0)[0] == peer.get(0)[0] && flow.get(flow.count - 1)[1] == peer.get(peer.count - 1)[1], "unmatched communication byte totals");
            U64 first = 0;
            for (U64 index = 0; index < flow.count; ++index) {
                auto interval = flow.get(index);
                while (first < peer.count && peer.get(first)[1] <= interval[0]) ++first;
                check(first < peer.count, "unmatched communication byte interval");
                U64 last = first;
                while (last < peer.count && peer.get(last)[0] < interval[1]) ++last;
                auto row = phase.events.get(interval[2]); row[4] = channel; row[5] = first; row[6] = last;
                phase.events.put(interval[2], row);
            }
        }
    }
}

std::array<U64, 2> depths(std::array<Phase*, 2> parties, Pages& pages, const std::filesystem::path& directory) {
    Records<3> stack(pages, directory); std::array<U64, 2> maximum{};
    for (auto* phase : parties) phase->states.count = phase->events.count;
    auto dependency = [&](size_t party, const std::array<U64, 7>& row, U64 cursor) -> std::pair<size_t, U64> {
        if (cursor < 2) return {party, row[2 + cursor]};
        if (row[0] != RECV || row[5] + cursor - 2 >= row[6]) return {party, MISSING};
        auto found = parties[1 - party]->flows.find({SEND, row[4]});
        check(found != parties[1 - party]->flows.end(), "missing receive source stream");
        return {1 - party, found->second->get(row[5] + cursor - 2)[2]};
    };
    for (size_t party = 0; party < 2; ++party) {
        Phase& phase = *parties[party];
        for (U64 node = 0; node < phase.events.count; ++node) {
            if (phase.states.get(node)[0] == 2) continue;
            phase.states.put(node, {1, 0, 0, 0}); stack.append({party, node, 0});
            while (stack.count) {
                auto top = stack.get(stack.count - 1); size_t actor_party = top[0];
                U64 current = top[1], cursor = top[2]; Phase& active = *parties[actor_party];
                auto row = active.events.get(current); U64 limit = 2 + (row[0] == RECV ? row[6] - row[5] : 0);
                if (cursor < limit) {
                    auto source = dependency(actor_party, row, cursor); ++top[2]; stack.put(stack.count - 1, top);
                    if (source.second == MISSING) continue;
                    auto state = parties[source.first]->states.get(source.second);
                    check(state[0] != 1, "communication dependency cycle");
                    if (!state[0]) { parties[source.first]->states.put(source.second, {1, 0, 0, 0}); stack.append({source.first, source.second, 0}); }
                    continue;
                }
                U64 clock0 = 0, clock1 = 0; size_t local_count = 0; std::array<std::array<U64, 4>, 2> local{};
                for (U64 position = 0; position < limit; ++position) {
                    auto source = dependency(actor_party, row, position); if (source.second == MISSING) continue;
                    auto state = parties[source.first]->states.get(source.second);
                    clock0 = std::max(clock0, state[1]); clock1 = std::max(clock1, state[2]);
                    if (source.first == actor_party) { check(local_count < 2, "invalid local dependency count"); local[local_count++] = state; }
                }
                bool last = local_count != 0;
                for (size_t i = 0; i < local_count; ++i) {
                    bool dominated = false;
                    for (size_t j = 0; j < local_count; ++j)
                        if (local[j][1] >= local[i][1] && local[j][2] >= local[i][2] &&
                            (local[j][1] != local[i][1] || local[j][2] != local[i][2])) dominated = true;
                    if (!dominated) last = last && bool(local[i][3]);
                }
                if (row[0] == PHASE_BEGIN) { clock0 = clock1 = 0; last = false; }
                else if (row[0] == SEND) { if (!last) { if (actor_party) ++clock1; else ++clock0; } last = true; }
                else if (row[0] == RECV) last = false;
                active.states.put(current, {2, clock0, clock1, U64(last)});
                maximum[0] = std::max(maximum[0], clock0); maximum[1] = std::max(maximum[1], clock1); --stack.count;
            }
        }
    }
    return maximum;
}

Json analyze(const std::string& server, const std::string& client, const std::string& parent) {
    TempDirectory temp(parent); Pages pages; Reader first(server, 1, pages, temp.path), second(client, 2, pages, temp.path);
    Json reports = Json::array(); std::set<std::pair<std::string, std::string>> observed; U64 count = 0;
    while (true) {
        auto start = std::chrono::steady_clock::now();
        auto left = first.next_phase(), right = second.next_phase();
        if (!left && !right) break;
        check(left && right, "parties have different phase sets");
        auto key = std::make_pair(left->query.dump(), left->name);
        check(left->query == right->query && left->name == right->name && observed.insert(key).second,
              "parties have different or duplicated phase sets");
        compare_identities(first, second); std::array<Phase*, 2> pair{left.get(), right.get()};
        auto parsed = std::chrono::steady_clock::now();
        match(pair);
        auto matched = std::chrono::steady_clock::now();
        auto depth = depths(pair, pages, temp.path);
        auto replayed = std::chrono::steady_clock::now();
        reports.push_back(Json{{"query", left->query}, {"phase", left->name}, {"logical_rounds", std::max(depth[0], depth[1])},
            {"send_flight_depths", depth}, {"party_sent_bytes", Json::array({left->sent, right->sent})},
            {"party_received_bytes", Json::array({left->received, right->received})}});
        count += left->events.count + right->events.count;
        std::cerr << "NATIVE_TRACE_PHASE query=" << left->query << " phase=" << left->name
                  << " events=" << left->events.count + right->events.count << " rounds=" << std::max(depth[0], depth[1])
                  << " read_validate_seconds=" << std::chrono::duration<double>(parsed - start).count()
                  << " match_seconds=" << std::chrono::duration<double>(matched - parsed).count()
                  << " depths_seconds=" << std::chrono::duration<double>(replayed - matched).count() << '\n';
    }
    return Json{{"status", "passed"}, {"definition", DEFINITION}, {"events", count}, {"phases", reports},
                {"scopes", Json::array()}, {"scope_rounds_omitted", "phase-only bounded-memory analysis"}};
}
}

int main(int argc, char** argv) {
    std::string server, client, output, directory; Json result;
    try {
        for (int index = 1; index < argc; ++index) {
            std::string option = argv[index]; check(index + 1 < argc, "missing command line option value");
            std::string value = argv[++index];
            if (option == "--server" || option == "--party1") server = value;
            else if (option == "--client" || option == "--party2") client = value;
            else if (option == "--output") output = value;
            else if (option == "--temp-directory") directory = value;
            else fail("unknown command line option: " + option);
        }
        check(!server.empty() && !client.empty(), "--server and --client are required");
        result = analyze(server, client, directory);
    } catch (const std::exception& error) {
        result = Json{{"status", "failed"}, {"reason", error.what()}, {"definition", DEFINITION},
                      {"logical_rounds", nullptr}, {"phases", Json::array()}};
    }
    std::string text = result.dump(2) + "\n";
    if (!output.empty()) {
        std::ofstream file(output); file << text;
        if (!file) { std::cerr << "cannot write analyzer output: " << output << '\n'; return 2; }
    }
    std::cout << text; return result["status"] == "passed" ? 0 : 2;
}
