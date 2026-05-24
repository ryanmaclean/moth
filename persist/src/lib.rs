//! File-backed `harness::SessionStore` with an append-only operation log
//! plus periodic snapshots.
//!
//! Two files per session at `<root>/<key>.snapshot.json` (a JSON document
//! `{version,messages:[...]}` mirroring the old format) and
//! `<root>/<key>.log.jsonl` (one NDJSON record per `append` call). A `save`
//! is a single `write_all` of `{"messages":[...]}` + newline against an
//! `OpenOptions::append(true)` handle — O(delta), not O(history).
//!
//! `load` reads the snapshot then drains the log, materialising the full
//! `Vec<ChatMessage>` in order.
//!
//! Auto-snapshot: when the log grows beyond `SNAPSHOT_RECORDS` lines or
//! `SNAPSHOT_BYTES` bytes the next `append` first rewrites the snapshot
//! and truncates the log so `load` stays bounded.
//!
//! Legacy migration: if a `<key>.json` from the pre-log layout is present
//! but `<key>.snapshot.json` is not, the old file is renamed in place on
//! the first `load` (or any path resolution) — opaque to callers.
//!
//! Keys may come from URL paths, so we reject anything that could escape
//! `root` or hide a file: no `/`, `\`, `..`, no leading `.`, capped at 200
//! chars. Otherwise opaque.
//!
//! Corrupt log line: we return `StoreError` describing the line number.
//! A corrupt store should be loud, not silently truncated. The existing
//! snapshot-parse-error behaviour is preserved verbatim.
//!
//! Deps: `harness` (the trait), `anthropic::json` (escape + parse).

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anthropic::json::{self, Json};
use harness::{ChatMessage, ContentBlock, Role, SessionStore, StoreError};

const WIRE_VERSION: u32 = 1;
const MAX_KEY_LEN: usize = 200;

/// Auto-snapshot threshold (line count). 256 was picked because it covers
/// a few hundred turns — well past the chatty CLI sessions we see in
/// practice — without dwarfing a typical snapshot file.
const SNAPSHOT_RECORDS: u64 = 256;

/// Auto-snapshot threshold (byte count). 1 MiB. Mostly there to defend
/// against pathological turns (multi-MiB tool outputs) blowing the log up
/// before the record-count threshold trips.
const SNAPSHOT_BYTES: u64 = 1024 * 1024;

pub struct FileStore {
    root: PathBuf,
    /// Serialises append + snapshot mutations across the whole store. One
    /// mutex for all keys is the right pick: appends are a single
    /// `write_all` and contention is bounded by the number of concurrent
    /// sessions, which is small in every deployment we have.
    write_lock: Mutex<()>,
}

impl FileStore {
    /// Construct a store without touching the filesystem. Use this when the
    /// caller has already ensured `root` exists.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), write_lock: Mutex::new(()) }
    }

    /// Construct a store, creating `root` (and parents) if missing.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root, write_lock: Mutex::new(()) })
    }

    fn snapshot_path(&self, key: &str) -> Result<PathBuf, StoreError> {
        validate_key(key)?;
        Ok(self.root.join(format!("{key}.snapshot.json")))
    }

    fn log_path(&self, key: &str) -> Result<PathBuf, StoreError> {
        validate_key(key)?;
        Ok(self.root.join(format!("{key}.log.jsonl")))
    }

    fn legacy_path(&self, key: &str) -> PathBuf {
        // No validate_key — callers already validated via snapshot_path.
        self.root.join(format!("{key}.json"))
    }

    /// One-time, transparent migration of pre-log `<key>.json` files into
    /// `<key>.snapshot.json`. Idempotent: if the snapshot already exists
    /// we leave any stray legacy file alone (treat as duplicate, snapshot
    /// wins).
    fn migrate_legacy(&self, key: &str) -> Result<(), StoreError> {
        let snap = self.snapshot_path(key)?;
        if snap.exists() {
            return Ok(());
        }
        let legacy = self.legacy_path(key);
        match fs::rename(&legacy, &snap) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => {
                Err(StoreError(format!("migrate {} -> {}: {e}", legacy.display(), snap.display())))
            }
        }
    }

    /// Decide whether the log for `key` has crossed an auto-snapshot
    /// threshold. Best-effort: a stat failure means "no, don't snapshot",
    /// because failing the append over a metadata read would be silly.
    fn should_auto_snapshot(&self, log: &Path) -> bool {
        let Ok(meta) = fs::metadata(log) else {
            return false;
        };
        if meta.len() >= SNAPSHOT_BYTES {
            return true;
        }
        let Ok(bytes) = fs::read(log) else {
            return false;
        };
        // Count newlines — fast, no parsing, matches our line-oriented log.
        // bytemuck-free: `iter().filter().count()` is fine.
        let lines = bytes.iter().filter(|b| **b == b'\n').count() as u64;
        lines >= SNAPSHOT_RECORDS
    }

    /// Read the full materialised history for `key` (snapshot ++ log).
    /// Returns `Ok(None)` only if neither snapshot nor log exists.
    fn read_full(&self, key: &str) -> Result<Option<Vec<ChatMessage>>, StoreError> {
        self.migrate_legacy(key)?;
        let snap = self.snapshot_path(key)?;
        let log = self.log_path(key)?;

        let mut history: Vec<ChatMessage> = Vec::new();
        let mut saw_any = false;

        match fs::read(&snap) {
            Ok(bytes) => {
                saw_any = true;
                history.extend(parse_snapshot(&bytes, &snap)?);
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(StoreError(format!("read {}: {e}", snap.display()))),
        }

        match fs::read(&log) {
            Ok(bytes) => {
                saw_any = true;
                apply_log(&bytes, &log, &mut history)?;
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(StoreError(format!("read {}: {e}", log.display()))),
        }

        if !saw_any {
            return Ok(None);
        }
        Ok(Some(history))
    }

    /// Compact: rewrite `<key>.snapshot.json` from `history`, truncate the
    /// log. Atomic-per-file via tmp+rename for the snapshot; the log gets
    /// removed last so a crash between leaves us with snapshot+stale-log,
    /// which `load` still resolves correctly (snapshot wins, log replays
    /// on top — duplicate entries possible in that rare crash window).
    /// Callers must hold `self.write_lock` while invoking.
    fn compact(&self, key: &str, history: &[ChatMessage]) -> Result<(), StoreError> {
        let snap = self.snapshot_path(key)?;
        let log = self.log_path(key)?;
        let tmp = snap.with_extension("json.tmp");
        let body = serialize_snapshot(history);
        let _ = fs::remove_file(&tmp);
        fs::write(&tmp, body.as_bytes())
            .map_err(|e| StoreError(format!("write {}: {e}", tmp.display())))?;
        fs::rename(&tmp, &snap).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            StoreError(format!("rename {} -> {}: {e}", tmp.display(), snap.display()))
        })?;
        // Best-effort log truncation — a leftover log just means slower
        // loads until the next compact succeeds; not a correctness issue
        // because `apply_log` on top of a fresh snapshot would no-op on
        // the (now-stale) records … wait, it wouldn't, those records get
        // re-applied. So we must remove or truncate the log here.
        match fs::remove_file(&log) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StoreError(format!("remove {}: {e}", log.display()))),
        }
    }
}

fn validate_key(key: &str) -> Result<(), StoreError> {
    if key.is_empty() {
        return Err(StoreError("empty key".into()));
    }
    if key.len() > MAX_KEY_LEN {
        return Err(StoreError(format!("key too long ({} > {MAX_KEY_LEN})", key.len())));
    }
    if key.starts_with('.') {
        return Err(StoreError(format!("key may not start with '.': {key:?}")));
    }
    if key.contains('/') || key.contains('\\') {
        return Err(StoreError(format!("key may not contain separators: {key:?}")));
    }
    if key.contains("..") {
        return Err(StoreError(format!("key may not contain '..': {key:?}")));
    }
    // No control chars or NUL. Everything else is opaque.
    if key.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return Err(StoreError(format!("key contains control byte: {key:?}")));
    }
    Ok(())
}

impl SessionStore for FileStore {
    fn load(&self, key: &str) -> Result<Option<Vec<ChatMessage>>, StoreError> {
        // Validate before touching the filesystem so a bad key fails fast
        // and matches the legacy contract (rejected_key tests etc.).
        validate_key(key)?;
        self.read_full(key)
    }

    fn append(&self, key: &str, new_messages: &[ChatMessage]) -> Result<(), StoreError> {
        validate_key(key)?;
        if new_messages.is_empty() {
            return Ok(());
        }

        let _guard = self.write_lock.lock().expect("persist write lock poisoned");

        let log = self.log_path(key)?;

        // Auto-snapshot trigger: if the log already exceeds the threshold,
        // compact it before adding more. Read full state (which the helper
        // does anyway) and compact. After this the log is gone and the
        // snapshot reflects current truth.
        if self.should_auto_snapshot(&log) {
            // `read_full` returns `None` only if neither file exists — but
            // we already saw the log, so it must be `Some`.
            let history = self.read_full(key)?.unwrap_or_default();
            self.compact(key, &history)?;
        }

        // Append a single NDJSON line.
        let mut line = String::new();
        serialize_log_record(&mut line, new_messages);
        line.push('\n');

        // create+append: create the log on first call, append thereafter.
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .map_err(|e| StoreError(format!("open {}: {e}", log.display())))?;
        f.write_all(line.as_bytes())
            .map_err(|e| StoreError(format!("append {}: {e}", log.display())))?;
        // No fsync — best-effort durability matches the prior FileStore.
        Ok(())
    }

    fn snapshot(&self, key: &str, history: &[ChatMessage]) -> Result<(), StoreError> {
        validate_key(key)?;
        let _guard = self.write_lock.lock().expect("persist write lock poisoned");
        self.compact(key, history)
    }
}

// ---- snapshot serialisation (compatible with the v1 file format) -------

fn serialize_snapshot(history: &[ChatMessage]) -> String {
    let mut s = String::new();
    s.push_str("{\"version\":");
    s.push_str(&WIRE_VERSION.to_string());
    s.push_str(",\"messages\":[");
    for (i, m) in history.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        serialize_message(&mut s, m);
    }
    s.push_str("]}");
    s
}

fn parse_snapshot(bytes: &[u8], path: &Path) -> Result<Vec<ChatMessage>, StoreError> {
    let v = json::parse(bytes).map_err(|e| StoreError(format!("parse {}: {e}", path.display())))?;
    let version = v
        .get("version")
        .ok_or_else(|| StoreError(format!("{}: missing version", path.display())))?;
    match version {
        Json::Num(n) if n == &WIRE_VERSION.to_string() => {}
        other => {
            return Err(StoreError(format!("{}: unsupported version {:?}", path.display(), other)));
        }
    }
    let messages = match v.get("messages") {
        Some(Json::Arr(a)) => a,
        _ => {
            return Err(StoreError(format!("{}: messages not an array", path.display())));
        }
    };
    let mut out = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        out.push(
            parse_message(m)
                .map_err(|e| StoreError(format!("{}: message[{i}]: {e}", path.display())))?,
        );
    }
    Ok(out)
}

// ---- append-log records ------------------------------------------------

/// Wire shape: `{"messages":[<message>, ...]}` per line. We don't bother
/// with `op` discriminators in v1 — there's only one op (append) and the
/// snapshot file carries its own header. Adding an op tag later is
/// trivially backward-compatible because unknown keys are ignored.
fn serialize_log_record(out: &mut String, msgs: &[ChatMessage]) {
    out.push_str("{\"messages\":[");
    for (i, m) in msgs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        serialize_message(out, m);
    }
    out.push_str("]}");
}

fn apply_log(bytes: &[u8], path: &Path, history: &mut Vec<ChatMessage>) -> Result<(), StoreError> {
    for (idx, raw) in bytes.split(|b| *b == b'\n').enumerate() {
        // Trim CR for tolerant Windows-edit handling.
        let line = match raw.last() {
            Some(b'\r') => &raw[..raw.len() - 1],
            _ => raw,
        };
        if line.is_empty() {
            continue;
        }
        let v = json::parse(line)
            .map_err(|e| StoreError(format!("parse {} line {}: {e}", path.display(), idx + 1)))?;
        let arr = match v.get("messages") {
            Some(Json::Arr(a)) => a,
            _ => {
                return Err(StoreError(format!(
                    "{} line {}: missing messages array",
                    path.display(),
                    idx + 1
                )));
            }
        };
        for (mi, m) in arr.iter().enumerate() {
            let parsed = parse_message(m).map_err(|e| {
                StoreError(format!("{} line {} message[{mi}]: {e}", path.display(), idx + 1))
            })?;
            history.push(parsed);
        }
    }
    Ok(())
}

// ---- shared message + block (de)serialisation --------------------------

fn serialize_message(s: &mut String, m: &ChatMessage) {
    s.push_str("{\"role\":\"");
    s.push_str(match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    });
    s.push_str("\",\"content\":[");
    for (i, b) in m.content.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        serialize_block(s, b);
    }
    s.push_str("]}");
}

fn serialize_block(s: &mut String, b: &ContentBlock) {
    match b {
        ContentBlock::Text(t) => {
            s.push_str(r#"{"type":"text","text":""#);
            json::escape_into(s, t);
            s.push_str("\"}");
        }
        ContentBlock::ToolUse { id, name, input } => {
            s.push_str(r#"{"type":"tool_use","id":""#);
            json::escape_into(s, id);
            s.push_str(r#"","name":""#);
            json::escape_into(s, name);
            s.push_str(r#"","input":"#);
            // input is a raw JSON value, spliced verbatim. If it's empty,
            // emit a `{}` placeholder so the resulting file is valid JSON.
            if input.trim().is_empty() {
                s.push_str("{}");
            } else {
                s.push_str(input);
            }
            s.push('}');
        }
        ContentBlock::ToolResult { tool_use_id, content, is_error } => {
            s.push_str(r#"{"type":"tool_result","tool_use_id":""#);
            json::escape_into(s, tool_use_id);
            s.push_str(r#"","content":""#);
            json::escape_into(s, content);
            s.push_str(r#"","is_error":"#);
            s.push_str(if *is_error { "true" } else { "false" });
            s.push('}');
        }
    }
}

fn parse_message(v: &Json) -> Result<ChatMessage, String> {
    let role = v.get("role").and_then(Json::as_str).ok_or_else(|| "missing role".to_string())?;
    let role = match role {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        other => return Err(format!("unknown role {other:?}")),
    };
    let content = match v.get("content") {
        Some(Json::Arr(a)) => a,
        _ => return Err("content not an array".into()),
    };
    let mut blocks = Vec::with_capacity(content.len());
    for (i, b) in content.iter().enumerate() {
        blocks.push(parse_block(b).map_err(|e| format!("block[{i}]: {e}"))?);
    }
    Ok(ChatMessage { role, content: blocks })
}

fn parse_block(v: &Json) -> Result<ContentBlock, String> {
    let ty = v.get("type").and_then(Json::as_str).ok_or_else(|| "missing type".to_string())?;
    match ty {
        "text" => {
            let text = v
                .get("text")
                .and_then(Json::as_str)
                .ok_or_else(|| "text: missing text".to_string())?;
            Ok(ContentBlock::Text(text.into()))
        }
        "tool_use" => {
            let id = v
                .get("id")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_use: missing id".to_string())?;
            let name = v
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_use: missing name".to_string())?;
            let input = v.get("input").ok_or_else(|| "tool_use: missing input".to_string())?;
            Ok(ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input: json_to_string(input).into(),
            })
        }
        "tool_result" => {
            let tool_use_id = v
                .get("tool_use_id")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_result: missing tool_use_id".to_string())?;
            let content = v
                .get("content")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_result: missing content".to_string())?;
            let is_error = match v.get("is_error") {
                Some(Json::Bool(b)) => *b,
                None => false,
                _ => return Err("tool_result: is_error not bool".into()),
            };
            Ok(ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error,
            })
        }
        other => Err(format!("unknown block type {other:?}")),
    }
}

/// Re-serialize a parsed `Json` node into its canonical string form. We need
/// this for `ToolUse::input`, which is stored as a raw JSON value in the
/// file and reconstructed as a string in `ContentBlock::ToolUse::input`.
fn json_to_string(v: &Json) -> String {
    let mut s = String::new();
    write_json(&mut s, v);
    s
}

fn write_json(s: &mut String, v: &Json) {
    match v {
        Json::Null => s.push_str("null"),
        Json::Bool(true) => s.push_str("true"),
        Json::Bool(false) => s.push_str("false"),
        Json::Num(n) => s.push_str(n),
        Json::Str(t) => {
            s.push('"');
            json::escape_into(s, t);
            s.push('"');
        }
        Json::Arr(items) => {
            s.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                write_json(s, item);
            }
            s.push(']');
        }
        Json::Obj(kv) => {
            s.push('{');
            for (i, (k, val)) in kv.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push('"');
                json::escape_into(s, k);
                s.push_str("\":");
                write_json(s, val);
            }
            s.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Per-test unique scratch dir under temp_dir(). EPOCH nanos + atomic
    // counter give us isolation without pulling in a rand crate.
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch() -> PathBuf {
        let n =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        let c = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("persist-test-{n}-{c}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(p: &PathBuf) {
        let _ = fs::remove_dir_all(p);
    }

    #[test]
    fn round_trip_text_only() {
        let root = scratch();
        let store = FileStore::new(&root);
        store
            .append("s1", &[ChatMessage::user("hi"), ChatMessage::assistant("hello there")])
            .unwrap();
        let loaded = store.load("s1").unwrap().expect("Some");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].role, Role::User);
        assert!(matches!(&loaded[0].content[0], ContentBlock::Text(t) if &**t == "hi"));
        assert_eq!(loaded[1].role, Role::Assistant);
        assert!(matches!(&loaded[1].content[0], ContentBlock::Text(t) if &**t == "hello there"));
        cleanup(&root);
    }

    #[test]
    fn round_trip_tool_use_and_result() {
        let root = scratch();
        let store = FileStore::new(&root);
        let history = vec![
            ChatMessage::user("list /tmp"),
            ChatMessage {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text("ok, running".into()),
                    ContentBlock::ToolUse {
                        id: "tu_1".into(),
                        name: "bash".into(),
                        input: r#"{"command":"ls /tmp"}"#.into(),
                    },
                ],
            },
            ChatMessage {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: "file1\nfile2".into(),
                    is_error: false,
                }],
            },
        ];
        store.append("sess", &history).unwrap();
        let loaded = store.load("sess").unwrap().expect("Some");
        assert_eq!(loaded.len(), 3);
        match &loaded[1].content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(&**id, "tu_1");
                assert_eq!(&**name, "bash");
                assert_eq!(&**input, r#"{"command":"ls /tmp"}"#);
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        match &loaded[2].content[0] {
            ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                assert_eq!(&**tool_use_id, "tu_1");
                assert_eq!(&**content, "file1\nfile2");
                assert!(!is_error);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        cleanup(&root);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let root = scratch();
        let store = FileStore::new(&root);
        assert!(store.load("does-not-exist").unwrap().is_none());
        cleanup(&root);
    }

    #[test]
    fn key_rejects_slash() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.append("a/b", &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains("separators"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_backslash() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.append("a\\b", &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains("separators"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_dotdot() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.append("..", &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains("..") || err.0.contains('.'), "got: {}", err.0);
        let err = store.append("foo..bar", &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains(".."), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_leading_dot() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.append(".hidden", &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains("'.'"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_too_long() {
        let root = scratch();
        let store = FileStore::new(&root);
        let long = "a".repeat(MAX_KEY_LEN + 1);
        let err = store.append(&long, &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains("too long"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_empty() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.append("", &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains("empty"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_control_chars() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.append("a\nb", &[ChatMessage::user("x")]).unwrap_err();
        assert!(err.0.contains("control"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn multiple_sessions_dont_collide() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.append("a", &[ChatMessage::user("alpha")]).unwrap();
        store.append("b", &[ChatMessage::user("beta")]).unwrap();
        store.append("c", &[ChatMessage::user("gamma")]).unwrap();
        let a = store.load("a").unwrap().unwrap();
        let b = store.load("b").unwrap().unwrap();
        let c = store.load("c").unwrap().unwrap();
        assert!(matches!(&a[0].content[0], ContentBlock::Text(t) if &**t == "alpha"));
        assert!(matches!(&b[0].content[0], ContentBlock::Text(t) if &**t == "beta"));
        assert!(matches!(&c[0].content[0], ContentBlock::Text(t) if &**t == "gamma"));
        // Cross-check: a's log path is distinct from b's.
        assert!(root.join("a.log.jsonl").exists());
        assert!(root.join("b.log.jsonl").exists());
        cleanup(&root);
    }

    #[test]
    fn corrupt_garbage_snapshot_returns_err() {
        let root = scratch();
        let store = FileStore::new(&root);
        fs::write(root.join("bad.snapshot.json"), b"not json at all").unwrap();
        let err = store.load("bad").unwrap_err();
        assert!(err.0.contains("parse"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn corrupt_log_line_returns_err() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.append("c", &[ChatMessage::user("ok")]).unwrap();
        // Splice a junk line into the log.
        let log = root.join("c.log.jsonl");
        let mut body = fs::read(&log).unwrap();
        body.extend_from_slice(b"NOT JSON\n");
        fs::write(&log, body).unwrap();
        let err = store.load("c").unwrap_err();
        assert!(err.0.contains("line 2"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn corrupt_unknown_version_returns_err() {
        let root = scratch();
        let store = FileStore::new(&root);
        fs::write(root.join("v.snapshot.json"), br#"{"version":99,"messages":[]}"#).unwrap();
        let err = store.load("v").unwrap_err();
        assert!(err.0.contains("version"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn rejected_key_does_not_clobber_existing_log() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.append("keep", &[ChatMessage::user("original")]).unwrap();
        let before = fs::read(root.join("keep.log.jsonl")).unwrap();
        let err = store.append("../keep", &[ChatMessage::user("evil")]).unwrap_err();
        assert!(err.0.contains("..") || err.0.contains("separators"));
        let after = fs::read(root.join("keep.log.jsonl")).unwrap();
        assert_eq!(before, after);
        cleanup(&root);
    }

    #[test]
    fn open_creates_root() {
        let n =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("persist-open-{n}/nested"));
        assert!(!dir.exists());
        let store = FileStore::open(&dir).unwrap();
        assert!(dir.exists());
        store.append("k", &[ChatMessage::user("hi")]).unwrap();
        cleanup(&dir.parent().unwrap().to_path_buf());
    }

    #[test]
    fn appending_extends_history() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.append("s", &[ChatMessage::user("first")]).unwrap();
        store.append("s", &[ChatMessage::user("second"), ChatMessage::assistant("ack")]).unwrap();
        let loaded = store.load("s").unwrap().unwrap();
        assert_eq!(loaded.len(), 3);
        assert!(matches!(&loaded[0].content[0], ContentBlock::Text(t) if &**t == "first"));
        assert!(matches!(&loaded[1].content[0], ContentBlock::Text(t) if &**t == "second"));
        assert!(matches!(&loaded[2].content[0], ContentBlock::Text(t) if &**t == "ack"));
        cleanup(&root);
    }

    #[test]
    fn escapes_in_text_round_trip() {
        let root = scratch();
        let store = FileStore::new(&root);
        let weird = "quotes \"x\" \\ slash \n newline \t tab \x01 ctrl é";
        store.append("esc", &[ChatMessage::user(weird)]).unwrap();
        let loaded = store.load("esc").unwrap().unwrap();
        if let ContentBlock::Text(t) = &loaded[0].content[0] {
            assert_eq!(&**t, weird);
        } else {
            panic!("expected text block");
        }
        // The embedded newline in `weird` must have been escaped — the log
        // should still be exactly one record. We sanity-check by counting
        // lines.
        let log_bytes = fs::read(root.join("esc.log.jsonl")).unwrap();
        let lines = log_bytes.iter().filter(|b| **b == b'\n').count();
        assert_eq!(lines, 1, "embedded newline must be escaped, not literal");
        cleanup(&root);
    }

    #[test]
    fn empty_append_is_noop() {
        let root = scratch();
        let store = FileStore::new(&root);
        // Appending nothing should not create a log file.
        store.append("e", &[]).unwrap();
        assert!(!root.join("e.log.jsonl").exists());
        // Load still returns None since there's nothing at all on disk.
        assert!(store.load("e").unwrap().is_none());
        cleanup(&root);
    }

    #[test]
    fn tool_use_with_nested_input_round_trips() {
        let root = scratch();
        let store = FileStore::new(&root);
        let history = vec![ChatMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "tu_x".into(),
                name: "edit".into(),
                input: r#"{"path":"/tmp/f","edits":[{"old":"a","new":"b"}],"flag":true}"#.into(),
            }],
        }];
        store.append("nest", &history).unwrap();
        let loaded = store.load("nest").unwrap().unwrap();
        match &loaded[0].content[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert_eq!(
                    &**input,
                    r#"{"path":"/tmp/f","edits":[{"old":"a","new":"b"}],"flag":true}"#
                );
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        cleanup(&root);
    }

    #[test]
    fn tool_result_is_error_round_trips() {
        let root = scratch();
        let store = FileStore::new(&root);
        let history = vec![ChatMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_e".into(),
                content: "boom".into(),
                is_error: true,
            }],
        }];
        store.append("err", &history).unwrap();
        let loaded = store.load("err").unwrap().unwrap();
        match &loaded[0].content[0] {
            ContentBlock::ToolResult { is_error, content, .. } => {
                assert!(*is_error);
                assert_eq!(&**content, "boom");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        cleanup(&root);
    }

    #[test]
    fn explicit_snapshot_replaces_log() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.append("s", &[ChatMessage::user("a")]).unwrap();
        store.append("s", &[ChatMessage::user("b")]).unwrap();
        // Now compact with a canonical view.
        let canonical = vec![ChatMessage::assistant("compacted-summary")];
        store.snapshot("s", &canonical).unwrap();
        // Snapshot file exists, log file removed.
        assert!(root.join("s.snapshot.json").exists());
        assert!(!root.join("s.log.jsonl").exists());
        let loaded = store.load("s").unwrap().unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(
            matches!(&loaded[0].content[0], ContentBlock::Text(t) if &**t == "compacted-summary")
        );
        cleanup(&root);
    }

    #[test]
    fn auto_snapshot_triggers_after_threshold() {
        let root = scratch();
        let store = FileStore::new(&root);
        // Drive past the record threshold with cheap entries.
        for _ in 0..(SNAPSHOT_RECORDS + 4) {
            store.append("a", &[ChatMessage::user("x")]).unwrap();
        }
        // The next append should have flipped the log: it must be small
        // (one record) because everything before got rolled into the
        // snapshot.
        let log_lines =
            fs::read(root.join("a.log.jsonl")).unwrap().iter().filter(|b| **b == b'\n').count();
        assert!(
            (log_lines as u64) < SNAPSHOT_RECORDS,
            "log should have shrunk after auto-snapshot, saw {log_lines} lines"
        );
        // History still round-trips with the full count.
        let loaded = store.load("a").unwrap().unwrap();
        assert_eq!(loaded.len() as u64, SNAPSHOT_RECORDS + 4);
        cleanup(&root);
    }

    #[test]
    fn legacy_json_migrated_to_snapshot() {
        let root = scratch();
        let store = FileStore::new(&root);
        // Synthesise an old-style <key>.json file by writing the v1 wire
        // shape directly.
        let legacy_body = r#"{"version":1,"messages":[{"role":"user","content":[{"type":"text","text":"old"}]}]}"#;
        fs::write(root.join("old.json"), legacy_body).unwrap();
        let loaded = store.load("old").unwrap().unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(matches!(&loaded[0].content[0], ContentBlock::Text(t) if &**t == "old"));
        // Migration was actually performed.
        assert!(!root.join("old.json").exists());
        assert!(root.join("old.snapshot.json").exists());
        // Subsequent appends layer on top of the migrated snapshot.
        store.append("old", &[ChatMessage::assistant("new")]).unwrap();
        let loaded = store.load("old").unwrap().unwrap();
        assert_eq!(loaded.len(), 2);
        cleanup(&root);
    }

    #[test]
    fn concurrent_appends_from_threads_all_recorded() {
        let root = scratch();
        let store = Arc::new(FileStore::new(&root));
        let workers = 8usize;
        let per_worker = 32usize;
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let s = store.clone();
            handles.push(thread::spawn(move || {
                for i in 0..per_worker {
                    s.append("c", &[ChatMessage::user(format!("w{w}-i{i}"))]).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let loaded = store.load("c").unwrap().unwrap();
        assert_eq!(loaded.len(), workers * per_worker);
        // Every text payload appears exactly once.
        let mut seen = std::collections::HashSet::new();
        for m in &loaded {
            if let ContentBlock::Text(t) = &m.content[0] {
                assert!(seen.insert(t.clone()), "duplicate {t}");
            }
        }
        assert_eq!(seen.len(), workers * per_worker);
        cleanup(&root);
    }

    #[test]
    fn snapshot_then_load_with_empty_log() {
        // A `snapshot` call without prior `append` still writes the
        // snapshot file and `load` returns Some(history).
        let root = scratch();
        let store = FileStore::new(&root);
        let history = vec![ChatMessage::user("only-snapshot")];
        store.snapshot("s", &history).unwrap();
        assert!(root.join("s.snapshot.json").exists());
        assert!(!root.join("s.log.jsonl").exists());
        let loaded = store.load("s").unwrap().unwrap();
        assert_eq!(loaded.len(), 1);
        cleanup(&root);
    }
}
