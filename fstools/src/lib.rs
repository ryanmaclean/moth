//! Filesystem tools: `read_file`, `write_file`, `edit_file`.
//!
//! Each tool implements `harness::Tool`. Input is raw JSON (the model's
//! tool-call payload) parsed via `anthropic::json`. An optional `root`
//! sandbox-root is enforced per tool: with `Some(root)` the agent can't
//! escape the directory; with `None`, file ops are unrestricted (trusted
//! CLI runs). `..` components are always refused; absolute paths are
//! refused when a root is set.
//!
//! ## Symlink safety
//!
//! When a `root` is configured, every component of the resolved path is
//! validated against symlinks: any symlink encountered along the path —
//! whether at an intermediate directory or at the leaf — causes the
//! operation to be refused, *even if the symlink points back inside the
//! root*. This strict policy is easier to reason about than tracking
//! "safe" vs "unsafe" symlinks; agents that legitimately need to follow
//! a symlink should resolve it themselves and pass the resolved path.
//!
//! For `WriteTool` creating new files (or intermediate directories), each
//! parent component is verified before being created via `mkdir`, and the
//! final leaf is written with `create_new`-style semantics that refuse to
//! overwrite a pre-existing symlink.
//!
//! ## Atomic writes
//!
//! `WriteTool` (and `EditTool`) stage their output in a sibling temp
//! file and `rename(2)` it over the destination, so a crash mid-write
//! leaves either the old contents or the new — never a truncated file.
//! The temp open uses `O_NOFOLLOW | O_EXCL | O_CREAT` to refuse a
//! pre-planted symlink at the staging path. See `atomic_write` for the
//! full rationale.

use std::path::{Component, Path, PathBuf};
#[cfg(unix)]
use std::sync::atomic::{AtomicU64, Ordering};

use anthropic::json::{Json, parse};
use harness::{Tool, ToolCtx, ToolError};

/// Monotonic counter used to disambiguate temp filenames produced by
/// `atomic_write` within a single process. Combined with the PID this
/// keeps temp paths unique across threads racing on the same target.
#[cfg(unix)]
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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
        let sandboxed = self.root.is_some();
        let full = resolve_existing(self.root.as_deref(), &path)?;

        let meta = std::fs::symlink_metadata(&full).map_err(io_err)?;
        if sandboxed && meta.file_type().is_symlink() {
            return Err(ToolError(format!("refusing to follow symlink: {path}")));
        }
        if sandboxed && hard_linked(&meta) {
            return Err(ToolError(format!("refusing hard-linked file (st_nlink > 1): {path}")));
        }
        // When sandboxed `resolve_existing` already vetted every
        // component, including the leaf, so by here the file cannot be
        // a symlink. The check above is belt-and-braces against a TOCTOU
        // race where the leaf was swapped after the walk.
        let meta = if meta.file_type().is_symlink() {
            // Rootless: follow the symlink to report size of the target
            // (matches historical behaviour, where std::fs::metadata
            // followed symlinks transparently).
            std::fs::metadata(&full).map_err(io_err)?
        } else {
            meta
        };
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
        let text = std::str::from_utf8(&bytes).map_err(|e| ToolError(format!("not UTF-8: {e}")))?;

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
        let sandboxed = self.root.is_some();
        let full = resolve_for_write(self.root.as_deref(), &path)?;

        if sandboxed {
            // Refuse to write through a pre-existing symlink at the
            // leaf. `symlink_metadata` does not follow symlinks, so a
            // planted `<root>/innocent -> /etc/passwd` is detected
            // here. Also refuse if the leaf has hard-link count > 1
            // — best-effort hard-link defence; doesn't catch every
            // case (an inode could acquire a second link between this
            // check and the write) but defeats the obvious vector.
            //
            // This check is belt-and-braces with the atomic-write path
            // (which itself opens with `O_NOFOLLOW` on the temp file
            // and `rename(2)`s over the destination): if a symlink is
            // planted between this check and the rename, the rename
            // simply replaces the symlink — it does NOT write through
            // it — so the target file is still safe.
            match std::fs::symlink_metadata(&full) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    return Err(ToolError(format!("refusing to write through symlink: {path}")));
                }
                Ok(meta) if hard_linked(&meta) => {
                    return Err(ToolError(format!(
                        "refusing to write through hard-linked file (st_nlink > 1): {path}"
                    )));
                }
                Ok(_) | Err(_) => {}
            }
        }
        atomic_write(&full, content.as_bytes())?;
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
        let sandboxed = self.root.is_some();
        let full = resolve_existing(self.root.as_deref(), &path)?;

        if sandboxed {
            let meta = std::fs::symlink_metadata(&full).map_err(io_err)?;
            if meta.file_type().is_symlink() {
                return Err(ToolError(format!("refusing to edit through symlink: {path}")));
            }
            if hard_linked(&meta) {
                return Err(ToolError(format!(
                    "refusing to edit hard-linked file (st_nlink > 1): {path}"
                )));
            }
        }
        let bytes = std::fs::read(&full).map_err(io_err)?;
        let text = std::str::from_utf8(&bytes).map_err(|e| ToolError(format!("not UTF-8: {e}")))?;

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
        atomic_write(&full, replaced.as_bytes())?;
        Ok(format!("edited {}: {} chars replaced", full.display(), old_text.len()))
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

/// Best-effort hard-link detection. A file with `nlink > 1` has another
/// directory entry somewhere — possibly outside our sandbox root — and
/// modifying it modifies the linked content as well. Path-level resolution
/// can't catch this because hard links share the same inode without
/// crossing any symlink boundary. Returns false on non-Unix where nlink
/// isn't available through std (we don't bring in `libc` just for this).
#[cfg(unix)]
fn hard_linked(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    meta.is_file() && meta.nlink() > 1
}

#[cfg(not(unix))]
fn hard_linked(_meta: &std::fs::Metadata) -> bool {
    false
}

/// Atomically replace the contents of `dest` with `content`.
///
/// Writes to a sibling temp file (`<dest>.<pid>.<counter>.tmp`) then
/// `rename(2)`s over the target. The rename is atomic on POSIX when the
/// two paths live on the same filesystem (which they do, by construction
/// — same parent directory), so readers always observe either the old
/// contents or the new, never a half-written file.
///
/// Defence-in-depth: the temp file is opened with `O_NOFOLLOW` and
/// `O_CREAT|O_EXCL` so that, even on the unlikely chance an attacker has
/// pre-planted a symlink at the exact temp path (PID + atomic counter),
/// the open fails rather than writing through it. The unique temp name
/// also keeps concurrent writers from clobbering each other's staging
/// files; whichever rename wins last is what the file ends up holding.
///
/// We `sync_data` the temp file before rename for durability, and
/// best-effort `sync_all` the parent dir so a power-cut after rename
/// still leaves the dirent committed. Failures of the directory fsync
/// are deliberately ignored — some filesystems (tmpfs) don't implement
/// it, and the data is already on disk by that point.
#[cfg(unix)]
fn atomic_write(dest: &Path, content: &[u8]) -> Result<(), ToolError> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    let file_name = dest
        .file_name()
        .ok_or_else(|| ToolError(format!("invalid write destination: {}", dest.display())))?;
    let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut tmp_name = std::ffi::OsString::from(file_name);
    tmp_name.push(format!(".{}.{}.tmp", std::process::id(), counter));
    let tmp = parent.join(&tmp_name);

    // O_NOFOLLOW guards the final path component: if `tmp` somehow
    // already exists as a symlink, the open refuses to follow it.
    // `create_new(true)` (=> O_CREAT|O_EXCL) makes a pre-existing tmp
    // path a hard error regardless.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_NOFOLLOW)
        .mode(0o600)
        .open(&tmp)
        .map_err(io_err)?;

    // Write + fsync the data, then drop the handle so rename(2) doesn't
    // race with any pending buffered writes inside std.
    let write_res = file.write_all(content).and_then(|()| file.sync_data()).map_err(io_err);
    drop(file);
    if let Err(e) = write_res {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(io_err(e));
    }

    // Best-effort: sync the parent directory so the renamed dirent is
    // durable. Failure here doesn't roll back the write (the data is
    // already visible to readers); just swallow.
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Non-Unix fallback: best-effort tmp-then-rename without O_NOFOLLOW
/// (a Unix-specific flag). Atomicity still holds on Windows because
/// `fs::rename` maps to `MoveFileExW` with REPLACE_EXISTING.
#[cfg(not(unix))]
fn atomic_write(dest: &Path, content: &[u8]) -> Result<(), ToolError> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    let file_name = dest
        .file_name()
        .ok_or_else(|| ToolError(format!("invalid write destination: {}", dest.display())))?;
    let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut tmp_name = std::ffi::OsString::from(file_name);
    tmp_name.push(format!(".{}.{}.tmp", std::process::id(), counter));
    let tmp = parent.join(&tmp_name);

    let mut file =
        std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(io_err)?;
    let write_res = file.write_all(content).and_then(|()| file.sync_data()).map_err(io_err);
    drop(file);
    if let Err(e) = write_res {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(io_err(e));
    }
    Ok(())
}

/// Split a relative path into its `Normal` components, rejecting any
/// `..`, `.`, prefix (drive letters on Windows) or root-dir components.
/// Returns the components as owned `PathBuf`s — easier to feed back into
/// `join` than `Component<'_>`.
fn user_components(path: &str) -> Result<Vec<PathBuf>, ToolError> {
    let p = Path::new(path);
    let mut out = Vec::new();
    for c in p.components() {
        match c {
            Component::Normal(part) => out.push(PathBuf::from(part)),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(ToolError(format!("path traversal not allowed: {path}")));
            }
            Component::RootDir | Component::Prefix(_) => {
                // The caller has already separately rejected absolute
                // paths when a root is set; this guards the
                // root-less branch from somehow producing an absolute.
                return Err(ToolError(format!(
                    "absolute paths not allowed under sandbox root: {path}"
                )));
            }
        }
    }
    Ok(out)
}

/// Resolve `path` under optional sandbox `root` for an operation that
/// requires the file to already exist (Read, Edit). Every intermediate
/// component, plus the leaf, must be a non-symlink that resolves inside
/// the canonical root.
fn resolve_existing(root: Option<&Path>, path: &str) -> Result<PathBuf, ToolError> {
    let p = Path::new(path);
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(ToolError(format!("path traversal not allowed: {path}")));
    }

    let Some(root) = root else {
        return Ok(p.to_path_buf());
    };

    if p.is_absolute() {
        return Err(ToolError(format!("absolute paths not allowed under sandbox root: {path}")));
    }

    let comps = user_components(path)?;
    let canonical_root =
        root.canonicalize().map_err(|e| ToolError(format!("sandbox root unreadable: {e}")))?;

    let mut cur = canonical_root.clone();
    for part in &comps {
        cur.push(part);
        // `symlink_metadata` does NOT follow symlinks, so we detect a
        // symlink at this component before walking through it.
        let meta = std::fs::symlink_metadata(&cur).map_err(io_err)?;
        if meta.file_type().is_symlink() {
            return Err(ToolError(format!(
                "refusing to traverse symlink at component {:?} of {path}",
                part.display()
            )));
        }
        // Belt and braces: even without symlinks, re-canonicalising the
        // running prefix must stay inside the canonical root. This also
        // catches obscure filesystems (bind mounts, hard-link tricks)
        // where the lexical join wandered out of bounds.
        let canon =
            cur.canonicalize().map_err(|e| ToolError(format!("cannot resolve path: {e}")))?;
        if !canon.starts_with(&canonical_root) {
            return Err(ToolError(format!("resolved path escapes sandbox root: {path}")));
        }
    }
    Ok(cur)
}

/// Resolve `path` under optional sandbox `root` for a write that may
/// need to create the leaf (and intermediate directories). Walks the
/// existing prefix using the same symlink checks as `resolve_existing`,
/// then creates any missing intermediate directories one at a time via
/// `mkdir` — never `create_dir_all` on a joined path, since that would
/// happily follow an attacker-planted symlink.
fn resolve_for_write(root: Option<&Path>, path: &str) -> Result<PathBuf, ToolError> {
    let p = Path::new(path);
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(ToolError(format!("path traversal not allowed: {path}")));
    }

    let Some(root) = root else {
        // No sandbox: preserve historical behaviour (create_dir_all on
        // the joined parent). Caller is trusted.
        if let Some(parent) = p.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }
        return Ok(p.to_path_buf());
    };

    if p.is_absolute() {
        return Err(ToolError(format!("absolute paths not allowed under sandbox root: {path}")));
    }

    let comps = user_components(path)?;
    let canonical_root =
        root.canonicalize().map_err(|e| ToolError(format!("sandbox root unreadable: {e}")))?;

    if comps.is_empty() {
        return Err(ToolError(format!("empty path: {path}")));
    }

    let last_idx = comps.len() - 1;
    let mut cur = canonical_root.clone();
    for (i, part) in comps.iter().enumerate() {
        cur.push(part);
        let is_leaf = i == last_idx;
        match std::fs::symlink_metadata(&cur) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    return Err(ToolError(format!(
                        "refusing to traverse symlink at component {:?} of {path}",
                        part.display()
                    )));
                }
                // Verify the running prefix stays under the canonical
                // root after symlink-free resolution.
                let canon = cur
                    .canonicalize()
                    .map_err(|e| ToolError(format!("cannot resolve path: {e}")))?;
                if !canon.starts_with(&canonical_root) {
                    return Err(ToolError(format!("resolved path escapes sandbox root: {path}")));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // This component does not exist. If it's an
                // intermediate, create it. If it's the leaf, leave it
                // — the caller (`atomic_write`) will create it. In
                // either case the parent has already been verified.
                if !is_leaf {
                    std::fs::create_dir(&cur).map_err(io_err)?;
                    // Re-validate: the freshly created dir should
                    // canonicalize inside the root.
                    let canon = cur
                        .canonicalize()
                        .map_err(|e| ToolError(format!("cannot resolve path: {e}")))?;
                    if !canon.starts_with(&canonical_root) {
                        return Err(ToolError(format!(
                            "resolved path escapes sandbox root: {path}"
                        )));
                    }
                }
            }
            Err(e) => return Err(io_err(e)),
        }
    }
    Ok(cur)
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
        let nanos =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!("fstools-{label}-{}-{nanos}-{n}", std::process::id()));
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
        let err = ReadTool { root: None }.call(r#"{"offset":1}"#, &ctx).unwrap_err();
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
        let err = tool.call(r#"{"path":"../etc/passwd"}"#, &ctx).unwrap_err();
        assert!(err.0.contains("path traversal not allowed"), "got: {}", err.0);
        cleanup(&dir);
    }

    #[test]
    fn read_absolute_path_with_root() {
        let dir = unique("read-abs");
        std::fs::create_dir_all(&dir).unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = ReadTool { root: Some(dir.clone()) };
        let err = tool.call(r#"{"path":"/etc/passwd"}"#, &ctx).unwrap_err();
        assert!(err.0.contains("absolute paths not allowed"), "got: {}", err.0);
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
        let input =
            format!(r#"{{"path":"{}","content":"hello"}}"#, json_escape(&path.to_string_lossy()));
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
        let input =
            format!(r#"{{"path":"{}","content":"ok"}}"#, json_escape(&path.to_string_lossy()));
        tool.call(&input, &ctx).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "ok");
        cleanup(&dir);
    }

    #[test]
    fn write_missing_path_field() {
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let err = WriteTool { root: None }.call(r#"{"content":"x"}"#, &ctx).unwrap_err();
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
    fn write_atomically_replaces_existing() {
        // Sanity: an existing file's contents are wholly replaced by
        // the new write — the rename-over semantics hand the agent a
        // fresh inode whose contents == the staged payload.
        let dir = unique("write-atomic-replace");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.txt");
        std::fs::write(&path, "initial").unwrap();
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = WriteTool { root: None };
        let input = format!(
            r#"{{"path":"{}","content":"replaced"}}"#,
            json_escape(&path.to_string_lossy())
        );
        tool.call(&input, &ctx).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replaced");
        cleanup(&dir);
    }

    #[test]
    fn write_is_atomic_no_partial_file_on_failure() {
        // After a successful write there should be no leftover *.tmp
        // siblings — the rename consumed the staging file. (Injecting a
        // mid-write failure is hard without a fault-injection layer; the
        // observable invariant is "no orphaned tmp files".)
        let dir = unique("write-atomic-no-tmp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.txt");
        let h = Harness::new();
        let ctx = ToolCtx { instance: h.addr() };
        let tool = WriteTool { root: None };
        let input =
            format!(r#"{{"path":"{}","content":"hello"}}"#, json_escape(&path.to_string_lossy()));
        tool.call(&input, &ctx).unwrap();
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        for name in &entries {
            assert!(
                !name.ends_with(".tmp"),
                "found leftover tmp file: {} (entries: {:?})",
                name,
                entries
            );
        }
        assert!(entries.iter().any(|n| n == "out.txt"));
        cleanup(&dir);
    }

    #[test]
    fn concurrent_writes_to_same_path_dont_corrupt() {
        // Four threads racing to write distinct contents to the same
        // path. Because each writer stages to its own uniquely-named
        // tmp file and then renames, the final file must equal one of
        // the inputs byte-for-byte — never a torn merge of two writers'
        // payloads.
        use std::thread;
        let dir = unique("write-atomic-concurrent");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shared.txt");
        // Use big distinctive payloads so any interleave would be
        // obvious (and any partial truncation would change the length).
        let inputs: Vec<String> = (0..4)
            .map(|i| {
                let ch = (b'A' + i as u8) as char;
                std::iter::repeat_n(ch, 4096).collect::<String>()
            })
            .collect();
        let path_arc = std::sync::Arc::new(path.clone());
        let handles: Vec<thread::JoinHandle<()>> = inputs
            .clone()
            .into_iter()
            .map(|content| {
                let p = std::sync::Arc::clone(&path_arc);
                thread::spawn(move || {
                    let h = Harness::new();
                    let ctx = ToolCtx { instance: h.addr() };
                    let tool = WriteTool { root: None };
                    let input = format!(
                        r#"{{"path":"{}","content":"{}"}}"#,
                        json_escape(&p.to_string_lossy()),
                        content
                    );
                    tool.call(&input, &ctx).unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let final_contents = std::fs::read_to_string(&path).unwrap();
        assert!(
            inputs.iter().any(|i| i == &final_contents),
            "final file does not match any writer's input (len={})",
            final_contents.len()
        );
        // And no tmp leftovers.
        for entry in std::fs::read_dir(&dir).unwrap() {
            let name = entry.unwrap().file_name();
            let name = name.to_string_lossy().into_owned();
            assert!(!name.ends_with(".tmp"), "leftover tmp: {name}");
        }
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
        let input =
            format!(r#"{{"path":"{}","content":"x"}}"#, json_escape(&path.to_string_lossy()));
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
        let err =
            EditTool { root: None }.call(r#"{"old_text":"a","new_text":"b"}"#, &ctx).unwrap_err();
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
            .call(r#"{"path":"../etc/passwd","old_text":"a","new_text":"b"}"#, &ctx)
            .unwrap_err();
        assert!(err.0.contains("path traversal not allowed"));
        cleanup(&dir);
    }

    // ---- definitions / metadata ----

    // ---- symlink safety (unix only; we use std::os::unix::fs::symlink) ----

    #[cfg(unix)]
    mod symlink {
        use super::*;
        use std::os::unix::fs::symlink;

        /// Build a sandbox `root` containing `target_kind`-style decoys.
        /// Returns `(root, secret_file_path)` where `secret_file_path`
        /// lives outside `root` and contains "SECRET".
        fn sandbox_with_secret(label: &str) -> (PathBuf, PathBuf) {
            let parent = unique(label);
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let secret = parent.join("secret.txt");
            std::fs::write(&secret, "SECRET").unwrap();
            (root, secret)
        }

        #[test]
        fn read_refuses_symlink_to_outside_file() {
            let (root, secret) = sandbox_with_secret("sym-read-out");
            symlink(&secret, root.join("innocent")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = ReadTool { root: Some(root.clone()) }
                .call(r#"{"path":"innocent"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink") || err.0.contains("escapes"), "got: {}", err.0);
            cleanup(root.parent().unwrap());
        }

        #[test]
        fn read_refuses_symlink_to_inside_file() {
            // Strict policy: even an inside-root symlink is refused.
            let parent = unique("sym-read-in");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let target = root.join("real.txt");
            std::fs::write(&target, "ok").unwrap();
            symlink(&target, root.join("alias")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = ReadTool { root: Some(root.clone()) }
                .call(r#"{"path":"alias"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink"), "got: {}", err.0);
            cleanup(&parent);
        }

        #[test]
        fn read_refuses_inside_symlinked_dir() {
            // <root>/dir is a symlink to <parent>/elsewhere/ which has inner.txt.
            let parent = unique("sym-read-dir");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let elsewhere = parent.join("elsewhere");
            std::fs::create_dir_all(&elsewhere).unwrap();
            std::fs::write(elsewhere.join("inner.txt"), "leaked").unwrap();
            symlink(&elsewhere, root.join("dir")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = ReadTool { root: Some(root.clone()) }
                .call(r#"{"path":"dir/inner.txt"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink"), "got: {}", err.0);
            cleanup(&parent);
        }

        #[test]
        fn write_refuses_symlink_to_outside_file() {
            let (root, secret) = sandbox_with_secret("sym-write-out");
            symlink(&secret, root.join("innocent")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = WriteTool { root: Some(root.clone()) }
                .call(r#"{"path":"innocent","content":"pwned"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink"), "got: {}", err.0);
            // And the target was NOT written through.
            assert_eq!(std::fs::read_to_string(&secret).unwrap(), "SECRET");
            cleanup(root.parent().unwrap());
        }

        #[test]
        fn write_to_symlink_target_is_rejected() {
            // Reproduces the TOCTOU shape the atomic-write change closes
            // off: a pre-existing symlink at `full` would, with the old
            // `fs::write`, follow into the target. The pre-write
            // `symlink_metadata` check fires first; this test pins that
            // behaviour and confirms the target is untouched.
            let parent = unique("sym-write-target");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let target = parent.join("target.txt");
            std::fs::write(&target, "ORIGINAL").unwrap();
            symlink(&target, root.join("link")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = WriteTool { root: Some(root.clone()) }
                .call(r#"{"path":"link","content":"PWNED"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink"), "got: {}", err.0);
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "ORIGINAL");
            cleanup(&parent);
        }

        #[test]
        fn write_refuses_through_intermediate_symlink() {
            // <root>/a -> /var/tmp-ish/somewhere; write a/b/c.txt would
            // end up at .../somewhere/b/c.txt.
            let parent = unique("sym-write-mid");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let elsewhere = parent.join("elsewhere");
            std::fs::create_dir_all(&elsewhere).unwrap();
            symlink(&elsewhere, root.join("a")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = WriteTool { root: Some(root.clone()) }
                .call(r#"{"path":"a/b/c.txt","content":"pwned"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink"), "got: {}", err.0);
            assert!(!elsewhere.join("b").exists(), "wrote through symlink");
            cleanup(&parent);
        }

        #[test]
        fn edit_refuses_symlink_to_outside_file() {
            let (root, secret) = sandbox_with_secret("sym-edit-out");
            symlink(&secret, root.join("innocent")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = EditTool { root: Some(root.clone()) }
                .call(r#"{"path":"innocent","old_text":"SECRET","new_text":"PWNED"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink"), "got: {}", err.0);
            assert_eq!(std::fs::read_to_string(&secret).unwrap(), "SECRET");
            cleanup(root.parent().unwrap());
        }

        #[test]
        fn edit_refuses_inside_symlinked_dir() {
            let parent = unique("sym-edit-dir");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let elsewhere = parent.join("elsewhere");
            std::fs::create_dir_all(&elsewhere).unwrap();
            std::fs::write(elsewhere.join("f.txt"), "alpha").unwrap();
            symlink(&elsewhere, root.join("dir")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = EditTool { root: Some(root.clone()) }
                .call(r#"{"path":"dir/f.txt","old_text":"alpha","new_text":"beta"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("symlink"), "got: {}", err.0);
            assert_eq!(std::fs::read_to_string(elsewhere.join("f.txt")).unwrap(), "alpha");
            cleanup(&parent);
        }

        #[test]
        fn write_create_parents_under_legitimate_dirs_still_works() {
            // Sanity: no symlinks involved, write should create intermediates.
            let parent = unique("sym-write-ok");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            WriteTool { root: Some(root.clone()) }
                .call(r#"{"path":"x/y/z.txt","content":"ok"}"#, &ctx)
                .unwrap();
            assert_eq!(std::fs::read_to_string(root.join("x/y/z.txt")).unwrap(), "ok");
            cleanup(&parent);
        }

        #[test]
        fn read_without_root_follows_symlinks() {
            // Opt-out path: when root: None, the caller is trusted and
            // historical behaviour is preserved — symlinks are followed
            // transparently. Restricting trusted CLI invocations would
            // be a usability regression.
            let parent = unique("sym-noroot");
            std::fs::create_dir_all(&parent).unwrap();
            let target = parent.join("target.txt");
            std::fs::write(&target, "value\n").unwrap();
            let link = parent.join("link");
            symlink(&target, &link).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let input = format!(r#"{{"path":"{}"}}"#, json_escape(&link.to_string_lossy()));
            let out = ReadTool { root: None }.call(&input, &ctx).unwrap();
            assert!(out.contains("value"), "got: {}", out);
            cleanup(&parent);
        }

        #[test]
        fn symlink_loop_does_not_panic() {
            let parent = unique("sym-loop");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            // a -> b, b -> a.
            symlink(root.join("b"), root.join("a")).unwrap();
            symlink(root.join("a"), root.join("b")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err =
                ReadTool { root: Some(root.clone()) }.call(r#"{"path":"a"}"#, &ctx).unwrap_err();
            // Should be a ToolError with non-empty message, not a panic.
            assert!(!err.0.is_empty());
            cleanup(&parent);
        }

        #[test]
        fn write_refuses_absolute_path_under_root() {
            let parent = unique("sym-abs");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = WriteTool { root: Some(root.clone()) }
                .call(r#"{"path":"/etc/passwd","content":"x"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("absolute paths not allowed"), "got: {}", err.0);
            cleanup(&parent);
        }

        #[test]
        fn write_into_existing_inside_root_dir_succeeds() {
            // <root>/dir/ exists as a real dir; write dir/f.txt works.
            let parent = unique("sym-real-dir");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(root.join("dir")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            WriteTool { root: Some(root.clone()) }
                .call(r#"{"path":"dir/f.txt","content":"hello"}"#, &ctx)
                .unwrap();
            assert_eq!(std::fs::read_to_string(root.join("dir/f.txt")).unwrap(), "hello");
            cleanup(&parent);
        }

        #[test]
        fn edit_inside_root_real_file_succeeds() {
            let parent = unique("sym-edit-real");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("f.txt"), "hello world\n").unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            EditTool { root: Some(root.clone()) }
                .call(r#"{"path":"f.txt","old_text":"world","new_text":"there"}"#, &ctx)
                .unwrap();
            assert_eq!(std::fs::read_to_string(root.join("f.txt")).unwrap(), "hello there\n");
            cleanup(&parent);
        }

        #[test]
        fn read_refuses_hard_linked_file() {
            let parent = unique("hl-read");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("a.txt"), b"shared content").unwrap();
            std::fs::hard_link(root.join("a.txt"), root.join("b.txt")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = ReadTool { root: Some(root.clone()) }
                .call(r#"{"path":"a.txt"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("hard-linked"), "expected hard-link refusal, got: {}", err.0);
            cleanup(&parent);
        }

        #[test]
        fn write_refuses_hard_linked_file() {
            let parent = unique("hl-write");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("a.txt"), b"original").unwrap();
            std::fs::hard_link(root.join("a.txt"), root.join("b.txt")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = WriteTool { root: Some(root.clone()) }
                .call(r#"{"path":"a.txt","content":"replacement"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("hard-linked"), "got: {}", err.0);
            // Confirm neither name was overwritten.
            assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "original");
            cleanup(&parent);
        }

        #[test]
        fn edit_refuses_hard_linked_file() {
            let parent = unique("hl-edit");
            std::fs::create_dir_all(&parent).unwrap();
            let root = parent.join("root");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("a.txt"), b"hello world").unwrap();
            std::fs::hard_link(root.join("a.txt"), root.join("b.txt")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let err = EditTool { root: Some(root.clone()) }
                .call(r#"{"path":"a.txt","old_text":"world","new_text":"there"}"#, &ctx)
                .unwrap_err();
            assert!(err.0.contains("hard-linked"), "got: {}", err.0);
            cleanup(&parent);
        }

        #[test]
        fn rootless_hard_link_follows() {
            // Without a configured root, no sandboxing — hard links are
            // legitimate. Read should succeed.
            let parent = unique("hl-rootless");
            std::fs::create_dir_all(&parent).unwrap();
            std::fs::write(parent.join("a.txt"), b"shared content").unwrap();
            std::fs::hard_link(parent.join("a.txt"), parent.join("b.txt")).unwrap();
            let h = Harness::new();
            let ctx = ToolCtx { instance: h.addr() };
            let out = ReadTool { root: None }
                .call(&format!(r#"{{"path":"{}"}}"#, parent.join("a.txt").to_string_lossy()), &ctx)
                .unwrap();
            assert!(out.contains("shared content"));
            cleanup(&parent);
        }
    }

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
