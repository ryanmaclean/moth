//! File-backed `harness::SessionStore`.
//!
//! One JSON file per session at `<root>/<key>.json`. Serialization shape
//! mirrors the Anthropic content-block wire format already used by the
//! `anthropic` crate (`type:text|tool_use|tool_result`), so a session file
//! reads like a transcript anyone familiar with the API can eyeball.
//!
//! Writes are atomic: render the full file to `<root>/<key>.json.tmp`, then
//! `rename()` over the final path. `std::fs::rename` is atomic on the same
//! filesystem on Linux + macOS.
//!
//! Keys may come from URL paths (`/agents/chat/<id>`), so we reject anything
//! that could escape `root` or hide a file: no `/`, `\`, `..`, no leading
//! `.`, and capped at 200 chars. Otherwise opaque.
//!
//! Deps: `harness` (the trait), `anthropic::json` (escape + parse). Nothing
//! else.

use std::fs;
use std::io;
use std::path::PathBuf;

use anthropic::json::{self, Json};
use harness::{ChatMessage, ContentBlock, Role, SessionStore, StoreError};

const WIRE_VERSION: u32 = 1;
const MAX_KEY_LEN: usize = 200;

pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    /// Construct a store without touching the filesystem. Use this when the
    /// caller has already ensured `root` exists.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Construct a store, creating `root` (and parents) if missing.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn path_for(&self, key: &str) -> Result<PathBuf, StoreError> {
        validate_key(key)?;
        Ok(self.root.join(format!("{key}.json")))
    }
}

fn validate_key(key: &str) -> Result<(), StoreError> {
    if key.is_empty() {
        return Err(StoreError("empty key".into()));
    }
    if key.len() > MAX_KEY_LEN {
        return Err(StoreError(format!(
            "key too long ({} > {MAX_KEY_LEN})",
            key.len()
        )));
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
        let path = self.path_for(key)?;
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(StoreError(format!("read {}: {e}", path.display()))),
        };
        let v = json::parse(&bytes)
            .map_err(|e| StoreError(format!("parse {}: {e}", path.display())))?;
        let version = v
            .get("version")
            .ok_or_else(|| StoreError(format!("{}: missing version", path.display())))?;
        match version {
            Json::Num(n) if n == &WIRE_VERSION.to_string() => {}
            other => {
                return Err(StoreError(format!(
                    "{}: unsupported version {:?}",
                    path.display(),
                    other
                )));
            }
        }
        let messages = match v.get("messages") {
            Some(Json::Arr(a)) => a,
            _ => {
                return Err(StoreError(format!(
                    "{}: messages not an array",
                    path.display()
                )));
            }
        };
        let mut out = Vec::with_capacity(messages.len());
        for (i, m) in messages.iter().enumerate() {
            out.push(parse_message(m).map_err(|e| {
                StoreError(format!("{}: message[{i}]: {e}", path.display()))
            })?);
        }
        Ok(Some(out))
    }

    fn save(&self, key: &str, history: &[ChatMessage]) -> Result<(), StoreError> {
        let path = self.path_for(key)?;
        let tmp = path.with_extension("json.tmp");
        let body = serialize(history);
        // Best-effort cleanup of a stale tmp from a prior crash. We don't care
        // if it isn't there.
        let _ = fs::remove_file(&tmp);
        fs::write(&tmp, body.as_bytes())
            .map_err(|e| StoreError(format!("write {}: {e}", tmp.display())))?;
        fs::rename(&tmp, &path).map_err(|e| {
            // Best-effort: remove the tmp so we don't leave litter on failure.
            let _ = fs::remove_file(&tmp);
            StoreError(format!(
                "rename {} -> {}: {e}",
                tmp.display(),
                path.display()
            ))
        })?;
        Ok(())
    }
}

fn serialize(history: &[ChatMessage]) -> String {
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
    let role = v
        .get("role")
        .and_then(Json::as_str)
        .ok_or_else(|| "missing role".to_string())?;
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
    let ty = v
        .get("type")
        .and_then(Json::as_str)
        .ok_or_else(|| "missing type".to_string())?;
    match ty {
        "text" => {
            let text = v
                .get("text")
                .and_then(Json::as_str)
                .ok_or_else(|| "text: missing text".to_string())?
                .to_string();
            Ok(ContentBlock::Text(text))
        }
        "tool_use" => {
            let id = v
                .get("id")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_use: missing id".to_string())?
                .to_string();
            let name = v
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_use: missing name".to_string())?
                .to_string();
            let input = v
                .get("input")
                .ok_or_else(|| "tool_use: missing input".to_string())?;
            Ok(ContentBlock::ToolUse { id, name, input: json_to_string(input) })
        }
        "tool_result" => {
            let tool_use_id = v
                .get("tool_use_id")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_result: missing tool_use_id".to_string())?
                .to_string();
            let content = v
                .get("content")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool_result: missing content".to_string())?
                .to_string();
            let is_error = match v.get("is_error") {
                Some(Json::Bool(b)) => *b,
                None => false,
                _ => return Err("tool_result: is_error not bool".into()),
            };
            Ok(ContentBlock::ToolResult { tool_use_id, content, is_error })
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Per-test unique scratch dir under temp_dir(). EPOCH nanos + atomic
    // counter give us isolation without pulling in a rand crate.
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch() -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
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
        let history = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello there"),
        ];
        store.save("s1", &history).unwrap();
        let loaded = store.load("s1").unwrap().expect("Some");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].role, Role::User);
        assert!(matches!(&loaded[0].content[0], ContentBlock::Text(t) if t == "hi"));
        assert_eq!(loaded[1].role, Role::Assistant);
        assert!(
            matches!(&loaded[1].content[0], ContentBlock::Text(t) if t == "hello there")
        );
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
        store.save("sess", &history).unwrap();
        let loaded = store.load("sess").unwrap().expect("Some");
        assert_eq!(loaded.len(), 3);
        match &loaded[1].content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "bash");
                assert_eq!(input, r#"{"command":"ls /tmp"}"#);
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        match &loaded[2].content[0] {
            ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                assert_eq!(tool_use_id, "tu_1");
                assert_eq!(content, "file1\nfile2");
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
        let err = store.save("a/b", &[]).unwrap_err();
        assert!(err.0.contains("separators"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_backslash() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.save("a\\b", &[]).unwrap_err();
        assert!(err.0.contains("separators"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_dotdot() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.save("..", &[]).unwrap_err();
        assert!(err.0.contains("..") || err.0.contains('.'), "got: {}", err.0);
        let err = store.save("foo..bar", &[]).unwrap_err();
        assert!(err.0.contains(".."), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_leading_dot() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.save(".hidden", &[]).unwrap_err();
        assert!(err.0.contains("'.'"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_too_long() {
        let root = scratch();
        let store = FileStore::new(&root);
        let long = "a".repeat(MAX_KEY_LEN + 1);
        let err = store.save(&long, &[]).unwrap_err();
        assert!(err.0.contains("too long"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_empty() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.save("", &[]).unwrap_err();
        assert!(err.0.contains("empty"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn key_rejects_control_chars() {
        let root = scratch();
        let store = FileStore::new(&root);
        let err = store.save("a\nb", &[]).unwrap_err();
        assert!(err.0.contains("control"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn multiple_sessions_dont_collide() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.save("a", &[ChatMessage::user("alpha")]).unwrap();
        store.save("b", &[ChatMessage::user("beta")]).unwrap();
        store.save("c", &[ChatMessage::user("gamma")]).unwrap();
        let a = store.load("a").unwrap().unwrap();
        let b = store.load("b").unwrap().unwrap();
        let c = store.load("c").unwrap().unwrap();
        assert!(matches!(&a[0].content[0], ContentBlock::Text(t) if t == "alpha"));
        assert!(matches!(&b[0].content[0], ContentBlock::Text(t) if t == "beta"));
        assert!(matches!(&c[0].content[0], ContentBlock::Text(t) if t == "gamma"));
        cleanup(&root);
    }

    #[test]
    fn corrupt_garbage_returns_err() {
        let root = scratch();
        let store = FileStore::new(&root);
        fs::write(root.join("bad.json"), b"not json at all").unwrap();
        let err = store.load("bad").unwrap_err();
        assert!(err.0.contains("parse"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn corrupt_truncated_returns_err() {
        let root = scratch();
        let store = FileStore::new(&root);
        store
            .save("trunc", &[ChatMessage::user("hello world")])
            .unwrap();
        let path = root.join("trunc.json");
        let full = fs::read(&path).unwrap();
        fs::write(&path, &full[..full.len() / 2]).unwrap();
        let err = store.load("trunc").unwrap_err();
        assert!(err.0.contains("parse"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn corrupt_unknown_version_returns_err() {
        let root = scratch();
        let store = FileStore::new(&root);
        fs::write(
            root.join("v.json"),
            br#"{"version":99,"messages":[]}"#,
        )
        .unwrap();
        let err = store.load("v").unwrap_err();
        assert!(err.0.contains("version"), "got: {}", err.0);
        cleanup(&root);
    }

    #[test]
    fn save_does_not_leave_tmp_file() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.save("clean", &[ChatMessage::user("x")]).unwrap();
        let tmp = root.join("clean.json.tmp");
        assert!(!tmp.exists(), "stray tmp left behind: {}", tmp.display());
        assert!(root.join("clean.json").exists());
        cleanup(&root);
    }

    #[test]
    fn rejected_key_does_not_clobber_existing_file() {
        // A save with an invalid key must fail BEFORE touching anything, so
        // an existing well-formed file on disk stays intact.
        let root = scratch();
        let store = FileStore::new(&root);
        store
            .save("keep", &[ChatMessage::user("original")])
            .unwrap();
        let before = fs::read(root.join("keep.json")).unwrap();
        // Try a doomed save with a bad key — must not clobber "keep".
        let err = store.save("../keep", &[ChatMessage::user("evil")]).unwrap_err();
        assert!(err.0.contains("..") || err.0.contains("separators"));
        let after = fs::read(root.join("keep.json")).unwrap();
        assert_eq!(before, after);
        cleanup(&root);
    }

    #[test]
    fn open_creates_root() {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("persist-open-{n}/nested"));
        assert!(!dir.exists());
        let store = FileStore::open(&dir).unwrap();
        assert!(dir.exists());
        store.save("k", &[ChatMessage::user("hi")]).unwrap();
        cleanup(&dir.parent().unwrap().to_path_buf());
    }

    #[test]
    fn overwrite_replaces_history() {
        let root = scratch();
        let store = FileStore::new(&root);
        store
            .save("s", &[ChatMessage::user("first")])
            .unwrap();
        store
            .save(
                "s",
                &[ChatMessage::user("second"), ChatMessage::assistant("ack")],
            )
            .unwrap();
        let loaded = store.load("s").unwrap().unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(matches!(&loaded[0].content[0], ContentBlock::Text(t) if t == "second"));
        cleanup(&root);
    }

    #[test]
    fn escapes_in_text_round_trip() {
        let root = scratch();
        let store = FileStore::new(&root);
        let weird = "quotes \"x\" \\ slash \n newline \t tab \x01 ctrl é";
        store
            .save("esc", &[ChatMessage::user(weird)])
            .unwrap();
        let loaded = store.load("esc").unwrap().unwrap();
        if let ContentBlock::Text(t) = &loaded[0].content[0] {
            assert_eq!(t, weird);
        } else {
            panic!("expected text block");
        }
        cleanup(&root);
    }

    #[test]
    fn empty_history_round_trips() {
        let root = scratch();
        let store = FileStore::new(&root);
        store.save("e", &[]).unwrap();
        let loaded = store.load("e").unwrap().unwrap();
        assert!(loaded.is_empty());
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
                input: r#"{"path":"/tmp/f","edits":[{"old":"a","new":"b"}],"flag":true}"#
                    .into(),
            }],
        }];
        store.save("nest", &history).unwrap();
        let loaded = store.load("nest").unwrap().unwrap();
        match &loaded[0].content[0] {
            ContentBlock::ToolUse { input, .. } => {
                // Exact byte equality: we re-serialize the parse tree, so
                // ordering of object keys is preserved by our by-Vec model.
                assert_eq!(
                    input,
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
        store.save("err", &history).unwrap();
        let loaded = store.load("err").unwrap().unwrap();
        match &loaded[0].content[0] {
            ContentBlock::ToolResult { is_error, content, .. } => {
                assert!(*is_error);
                assert_eq!(content, "boom");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        cleanup(&root);
    }
}
