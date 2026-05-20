//! Filesystem tools: `read_file`, `write_file`, `edit_file`.
//!
//! Each tool implements `harness::Tool`. Input is raw JSON (the model's
//! tool-call payload) parsed via `anthropic::json`. An optional `root`
//! sandbox-root is enforced per tool: with `Some(root)` the agent can't
//! escape the directory; with `None`, file ops are unrestricted (trusted
//! CLI runs). `..` components are always refused; absolute paths are
//! refused when a root is set.

use std::path::{Component, Path, PathBuf};

use anthropic::json::{Json, parse};
use harness::{Tool, ToolCtx, ToolError};

const MAX_BYTES: u64 = 1024 * 1024; // 1 MiB
const DEFAULT_LINE_LIMIT: usize = 2000;
const BINARY_SNIFF: usize = 8 * 1024;

/// Read a file's contents as numbered (`cat -n`-style) lines.
pub struct ReadTool {
    pub root: Option<PathBuf>,
}

/// Write content to a file, replacing existing content and creating parents.
pub struct WriteTool {
    pub root: Option<PathBuf>,
}

/// Replace exactly one occurrence of `old_text` in a file with `new_text`.
pub struct EditTool {
    pub root: Option<PathBuf>,
}

impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file's contents. Returns the file as a string. Errors if the file is not UTF-8 or exceeds 1 MiB."
    }

    fn input_schema(&self) -> &str {
        r#"{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"number","description":"1-based starting line."},"limit":{"type":"number","description":"Maximum number of lines to return."}},"required":["path"]}"#
    }

    fn call(&self, input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
        let v = parse_input(input)?;
        let path = require_str(&v, "path")?;
        let offset = optional_usize(&v, "offset")?;
        let limit = optional_usize(&v, "limit")?;
        let full = resolve(self.root.as_deref(), &path)?;

        let meta = std::fs::metadata(&full).map_err(io_err)?;
        if meta.len() > MAX_BYTES {
            return Err(ToolError(format!(
                "file too large: {} bytes (max {})",
                meta.len(),
                MAX_BYTES
            )));
        }

        let bytes = std::fs::read(&full).map_err(io_err)?;
        let sniff_len = bytes.len().min(BINARY_SNIFF);
        if bytes[..sniff_len].contains(&0) {
            return Err(ToolError("file appears binary".into()));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| ToolError(format!("not UTF-8: {e}")))?;

        Ok(format_numbered(text, offset, limit))
    }
}

impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file, creating it if needed. Replaces existing content."
    }

    fn input_schema(&self) -> &str {
        r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}"#
    }

    fn call(&self, input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
        let v = parse_input(input)?;
        let path = require_str(&v, "path")?;
        let content = require_str(&v, "content")?;
        let full = resolve(self.root.as_deref(), &path)?;

        if let Some(parent) = full.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }
        std::fs::write(&full, content.as_bytes()).map_err(io_err)?;
        Ok(format!("wrote {} bytes to {}", content.len(), full.display()))
    }
}

impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace exact text in a file. Errors if old_text doesn't appear, or appears more than once."
    }

    fn input_schema(&self) -> &str {
        r#"{"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"]}"#
    }

    fn call(&self, input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
        let v = parse_input(input)?;
        let path = require_str(&v, "path")?;
        let old_text = require_str(&v, "old_text")?;
        let new_text = require_str(&v, "new_text")?;
        let full = resolve(self.root.as_deref(), &path)?;

        let bytes = std::fs::read(&full).map_err(io_err)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| ToolError(format!("not UTF-8: {e}")))?;

        let count = text.matches(&old_text).count();
        if count == 0 {
            return Err(ToolError("old_text not found".into()));
        }
        if count > 1 {
            return Err(ToolError(format!(
                "old_text appears {count} times; refusing ambiguous edit"
            )));
        }
        let replaced = text.replacen(&old_text, &new_text, 1);
        std::fs::write(&full, replaced.as_bytes()).map_err(io_err)?;
        Ok(format!(
            "edited {}: {} chars replaced",
            full.display(),
            old_text.len()
        ))
    }
}

// ---- helpers ----

fn parse_input(input: &str) -> Result<Json, ToolError> {
    parse(input.as_bytes()).map_err(|e| ToolError(format!("invalid JSON: {e}")))
}

fn require_str(v: &Json, key: &str) -> Result<String, ToolError> {
    match v.get(key) {
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(_) => Err(ToolError(format!("field '{key}' must be a string"))),
        None => Err(ToolError(format!("missing field '{key}'"))),
    }
}

fn optional_usize(v: &Json, key: &str) -> Result<Option<usize>, ToolError> {
    match v.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Num(n)) => n
            .parse::<usize>()
            .map(Some)
            .map_err(|_| ToolError(format!("field '{key}' is not a non-negative integer"))),
        Some(_) => Err(ToolError(format!("field '{key}' must be a number"))),
    }
}

fn io_err(e: std::io::Error) -> ToolError {
    ToolError(e.to_string())
}

/// Resolve `path` under optional sandbox `root`. Refuses `..` components
/// always; refuses absolute paths when a root is set; ensures the
/// canonicalised result (when both the resolved path and root exist)
/// stays inside the root.
fn resolve(root: Option<&Path>, path: &str) -> Result<PathBuf, ToolError> {
    let p = Path::new(path);
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(ToolError(format!("path traversal not allowed: {path}")));
    }

    let Some(root) = root else {
        return Ok(p.to_path_buf());
    };

    if p.is_absolute() {
        return Err(ToolError(format!(
            "absolute paths not allowed under sandbox root: {path}"
        )));
    }

    let joined = root.join(p);
    // If both ends canonicalise, double-check that the resolved path
    // really sits under the root — guards against symlinks escaping.
    if let (Ok(rc), Ok(jc)) = (root.canonicalize(), joined.canonicalize())
        && !jc.starts_with(&rc)
    {
        return Err(ToolError(format!(
            "resolved path escapes sandbox root: {path}"
        )));
    }
    Ok(joined)
}

fn format_numbered(text: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    let start = offset.unwrap_or(1).max(1);
    let max = limit.unwrap_or(DEFAULT_LINE_LIMIT);

    let total = text.lines().count();
    let mut out = String::with_capacity(text.len() + 16);
    let mut last_emitted = 0usize;
    for (idx, line) in text.lines().enumerate() {
        let lineno = idx + 1;
        if lineno < start {
            continue;
        }
        if lineno >= start + max {
            break;
        }
        use std::fmt::Write;
        let _ = writeln!(&mut out, "{lineno}\t{line}");
        last_emitted = lineno;
    }
    if last_emitted > 0 && last_emitted < total {
        use std::fmt::Write;
        let _ = writeln!(&mut out, "[truncated at line {last_emitted}]");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use actor::{ActorRef, Spawned, spawn};
    use harness::{Instance, InstanceMsg, MockSandbox, Sandbox};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Lightweight unique-name allocator. We use temp_dir() + a counter
    // seeded from the clock to avoid collisions across parallel test runs.
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fstools-{label}-{}-{nanos}-{n}",
            std::process::id()
        ));
        p
    }

    fn cleanup(p: &Path) {
        let _ = std::fs::remove_dir_all(p);
        let _ = std::fs::remove_file(p);
    }

    // Each test needs a ToolCtx, even though file tools don't use it.
    // We spin up a tiny instance whose mailbox we never touch.
    struct Harness {
        inst: Option<Spawned<InstanceMsg>>,
    }

    impl Harness {
        fn new() -> Self {
            let sb: Box<dyn Sandbox> = Box::new(MockSandbox::new(vec![]));
            let inst = spawn(Instance::new("t", sb));
            Harness { inst: Some(inst) }
        }

        fn addr(&self) -> &ActorRef<InstanceMsg> {
            &self.inst.as_ref().unwrap().addr
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            if let Some(s) = self.inst.take() {
                let _ = s.join();
            }
        }
    }

    fn json_escape(s: &str) -> String {
        let mut out = String::new();
        anthropic::json::escape_into(&mut out, s);
        out
    }

    // ---- ReadTool ----

    #[test]
    fn read_happy_path() {
        let dir = unique("read-happy");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.txt");
        std::fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = ReadTool { root: None };
        let input = format!(r#"{{"path":"{}"}}"#, json_escape(&path.to_string_lossy()));
        let out = tool.call(&input, &ctx).unwrap();
        assert_eq!(out, "1\talpha\n2\tbeta\n3\tgamma\n");
        cleanup(&dir);
    }

    #[test]
    fn read_offset_and_limit() {
        let dir = unique("read-window");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.txt");
        std::fs::write(&path, "a\nb\nc\nd\ne\n").unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = ReadTool { root: None };
        let input = format!(
            r#"{{"path":"{}","offset":2,"limit":2}}"#,
            json_escape(&path.to_string_lossy())
        );
        let out = tool.call(&input, &ctx).unwrap();
        assert!(out.contains("2\tb"));
        assert!(out.contains("3\tc"));
        assert!(!out.contains("1\ta"));
        assert!(!out.contains("4\td"));
        assert!(out.contains("[truncated at line 3]"));
        cleanup(&dir);
    }

    #[test]
    fn read_missing_path_field() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let err = ReadTool { root: None }
            .call(r#"{"offset":1}"#, &ctx)
            .unwrap_err();
        assert!(err.0.contains("missing field 'path'"), "got: {}", err.0);
    }

    #[test]
    fn read_invalid_json() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let err = ReadTool { root: None }.call("not json", &ctx).unwrap_err();
        assert!(err.0.contains("invalid JSON"), "got: {}", err.0);
    }

    #[test]
    fn read_path_traversal_with_root() {
        let dir = unique("read-traversal");
        std::fs::create_dir_all(&dir).unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = ReadTool { root: Some(dir.clone()) };
        let err = tool
            .call(r#"{"path":"../etc/passwd"}"#, &ctx)
            .unwrap_err();
        assert!(
            err.0.contains("path traversal not allowed"),
            "got: {}",
            err.0
        );
        cleanup(&dir);
    }

    #[test]
    fn read_absolute_path_with_root() {
        let dir = unique("read-abs");
        std::fs::create_dir_all(&dir).unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = ReadTool { root: Some(dir.clone()) };
        let err = tool
            .call(r#"{"path":"/etc/passwd"}"#, &ctx)
            .unwrap_err();
        assert!(
            err.0.contains("absolute paths not allowed"),
            "got: {}",
            err.0
        );
        cleanup(&dir);
    }

    #[test]
    fn read_nonexistent_file() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let p = unique("read-nope").join("never.txt");
        let input = format!(r#"{{"path":"{}"}}"#, json_escape(&p.to_string_lossy()));
        let err = ReadTool { root: None }.call(&input, &ctx).unwrap_err();
        // io error message varies by platform; just check it's non-empty.
        assert!(!err.0.is_empty());
    }

    #[test]
    fn read_binary_file_refused() {
        let dir = unique("read-binary");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.bin");
        std::fs::write(&path, b"hello\x00world").unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let input = format!(r#"{{"path":"{}"}}"#, json_escape(&path.to_string_lossy()));
        let err = ReadTool { root: None }.call(&input, &ctx).unwrap_err();
        assert_eq!(err.0, "file appears binary");
        cleanup(&dir);
    }

    #[test]
    fn read_too_large_file_refused() {
        let dir = unique("read-big");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big.txt");
        // Just over 1 MiB of ASCII 'a's.
        let big = vec![b'a'; (MAX_BYTES as usize) + 16];
        std::fs::write(&path, &big).unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let input = format!(r#"{{"path":"{}"}}"#, json_escape(&path.to_string_lossy()));
        let err = ReadTool { root: None }.call(&input, &ctx).unwrap_err();
        assert!(err.0.contains("file too large"), "got: {}", err.0);
        cleanup(&dir);
    }

    #[test]
    fn read_truncates_at_default_line_limit() {
        let dir = unique("read-trunc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many.txt");
        let mut s = String::new();
        for i in 0..(DEFAULT_LINE_LIMIT + 50) {
            s.push_str(&format!("L{i}\n"));
        }
        std::fs::write(&path, &s).unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let input = format!(r#"{{"path":"{}"}}"#, json_escape(&path.to_string_lossy()));
        let out = ReadTool { root: None }.call(&input, &ctx).unwrap();
        assert!(
            out.contains(&format!("[truncated at line {DEFAULT_LINE_LIMIT}]")),
            "no truncation marker"
        );
        cleanup(&dir);
    }

    // ---- WriteTool ----

    #[test]
    fn write_happy_path() {
        let dir = unique("write-happy");
        let path = dir.join("out.txt");
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = WriteTool { root: None };
        let input = format!(
            r#"{{"path":"{}","content":"hello"}}"#,
            json_escape(&path.to_string_lossy())
        );
        let out = tool.call(&input, &ctx).unwrap();
        assert!(out.starts_with("wrote 5 bytes to "));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
        cleanup(&dir);
    }

    #[test]
    fn write_creates_parent_dirs() {
        let dir = unique("write-parents");
        let path = dir.join("nested").join("deeper").join("x.txt");
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = WriteTool { root: None };
        let input = format!(
            r#"{{"path":"{}","content":"ok"}}"#,
            json_escape(&path.to_string_lossy())
        );
        tool.call(&input, &ctx).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "ok");
        cleanup(&dir);
    }

    #[test]
    fn write_missing_path_field() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let err = WriteTool { root: None }
            .call(r#"{"content":"x"}"#, &ctx)
            .unwrap_err();
        assert!(err.0.contains("missing field 'path'"));
    }

    #[test]
    fn write_path_traversal_with_root() {
        let dir = unique("write-traversal");
        std::fs::create_dir_all(&dir).unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let err = WriteTool { root: Some(dir.clone()) }
            .call(r#"{"path":"../evil","content":"x"}"#, &ctx)
            .unwrap_err();
        assert!(err.0.contains("path traversal not allowed"));
        cleanup(&dir);
    }

    #[test]
    fn write_nonexistent_path_returns_io_error() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        // Use a parent path whose parent literally cannot be created
        // because it lives under a non-directory file.
        let dir = unique("write-noparent");
        std::fs::create_dir_all(&dir).unwrap();
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, b"i am a file").unwrap();
        let path = blocker.join("under_a_file").join("nope.txt");
        let input = format!(
            r#"{{"path":"{}","content":"x"}}"#,
            json_escape(&path.to_string_lossy())
        );
        let err = WriteTool { root: None }.call(&input, &ctx).unwrap_err();
        assert!(!err.0.is_empty());
        cleanup(&dir);
    }

    // ---- EditTool ----

    #[test]
    fn edit_exactly_one_occurrence_succeeds() {
        let dir = unique("edit-one");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("e.txt");
        std::fs::write(&path, "hello world\n").unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = EditTool { root: None };
        let input = format!(
            r#"{{"path":"{}","old_text":"world","new_text":"there"}}"#,
            json_escape(&path.to_string_lossy())
        );
        let out = tool.call(&input, &ctx).unwrap();
        assert!(out.contains("5 chars replaced"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello there\n");
        cleanup(&dir);
    }

    #[test]
    fn edit_zero_occurrences_errors() {
        let dir = unique("edit-zero");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("e.txt");
        std::fs::write(&path, "hello world\n").unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let input = format!(
            r#"{{"path":"{}","old_text":"absent","new_text":"x"}}"#,
            json_escape(&path.to_string_lossy())
        );
        let err = EditTool { root: None }.call(&input, &ctx).unwrap_err();
        assert!(err.0.contains("not found"));
        cleanup(&dir);
    }

    #[test]
    fn edit_multiple_occurrences_errors() {
        let dir = unique("edit-many");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("e.txt");
        std::fs::write(&path, "x x x").unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let input = format!(
            r#"{{"path":"{}","old_text":"x","new_text":"y"}}"#,
            json_escape(&path.to_string_lossy())
        );
        let err = EditTool { root: None }.call(&input, &ctx).unwrap_err();
        assert!(err.0.contains("3 times"), "got: {}", err.0);
        cleanup(&dir);
    }

    #[test]
    fn edit_missing_path_field() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let err = EditTool { root: None }
            .call(r#"{"old_text":"a","new_text":"b"}"#, &ctx)
            .unwrap_err();
        assert!(err.0.contains("missing field 'path'"));
    }

    #[test]
    fn edit_nonexistent_file_errors() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let p = unique("edit-nope").join("never.txt");
        let input = format!(
            r#"{{"path":"{}","old_text":"x","new_text":"y"}}"#,
            json_escape(&p.to_string_lossy())
        );
        let err = EditTool { root: None }.call(&input, &ctx).unwrap_err();
        assert!(!err.0.is_empty());
    }

    #[test]
    fn edit_path_traversal_with_root() {
        let dir = unique("edit-traversal");
        std::fs::create_dir_all(&dir).unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let err = EditTool { root: Some(dir.clone()) }
            .call(
                r#"{"path":"../etc/passwd","old_text":"a","new_text":"b"}"#,
                &ctx,
            )
            .unwrap_err();
        assert!(err.0.contains("path traversal not allowed"));
        cleanup(&dir);
    }

    // ---- definitions / metadata ----

    #[test]
    fn tool_definitions_round_trip() {
        let r = ReadTool { root: None };
        let w = WriteTool { root: None };
        let e = EditTool { root: None };
        assert_eq!(r.name(), "read_file");
        assert_eq!(w.name(), "write_file");
        assert_eq!(e.name(), "edit_file");
        for s in [r.input_schema(), w.input_schema(), e.input_schema()] {
            assert!(s.contains("\"required\""), "schema missing required: {s}");
        }
    }
}
