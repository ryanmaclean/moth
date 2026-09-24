//! File-backed structured event log.
//!
//! Subscribes to `harness::StreamEvent` over a `mpsc::Receiver` and writes
//! one JSONL record per event to `<dir>/<run_id>.jsonl`. Production audit
//! trail: every iteration, tool call, model delta gets a record.
//!
//! Record shape (one per line):
//! ```text
//! {"seq": <u64>, "ts_ms": <epoch-millis>, "run_id": "<id>", "kind": "<kind>", "data": {...}}
//! ```
//!
//! When the log was opened via `open_with_request_id`, every record also
//! carries the inbound HTTP correlation id between `run_id` and `kind`:
//! ```text
//! {"seq": ..., "ts_ms": ..., "run_id": "...", "request_id": "...", "kind": "...", "data": ...}
//! ```
//!
//! ## Identity, order, durability
//!
//! - **Identity.** `run_id` is supplied by the caller. Moth does not own
//!   run identity (BOP does); `agent run` accepts it via `--run-id` /
//!   `AGENT_RUN_ID` / `BOP_RUN_ID` and only mints a local id as a
//!   fallback. Because the id becomes a filename it must pass
//!   [`validate_run_id`]; [`RunLog::open`] refuses anything else rather
//!   than silently substituting a different identity.
//! - **Order.** `seq` is the single ordering source for records within a
//!   run: it is assigned under the same lock as the file write, so `seq`
//!   order == file order, starting at 0 and strictly increasing by one.
//!   Re-opening an existing run file (a retry of the same logical run)
//!   continues from the last committed `seq` instead of restarting.
//!   `ts_ms` is human/observation metadata only — never sort on it.
//! - **Durability.** Writes are visible on return but not durable. The
//!   drain calls `sync_data` after a terminal record (`done` /
//!   `cancelled` / `error`); a returned [`TerminalSummary`] with
//!   `final_event: Some(_)` therefore means the terminal record reached
//!   stable storage. [`RunLog::sync`] is exposed for callers' own markers.
//! - **Crash tail.** A crash mid-write can leave a torn final line with no
//!   trailing `\n`. On re-open the torn fragment is terminated with a
//!   newline so the next record starts on its own line; readers skip
//!   lines that fail to parse. The torn fragment never carries a
//!   committed `seq`, so it is not counted when resuming.
//!
//! Atomic writes: each record is built up in a `String`, then written via a
//! single `write_all` of `record + "\n"`. The `Mutex` serialises writers
//! and the `seq` counter together.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use anthropic::json::escape_into;
use harness::{PromptResult, SessionError, StreamEvent};

/// Maximum accepted `run_id` length in bytes.
pub const MAX_RUN_ID_LEN: usize = 128;

struct Inner {
    file: File,
    /// `seq` the next record will carry.
    next_seq: u64,
}

pub struct RunLog {
    inner: Mutex<Inner>,
    run_id: String,
    /// Optional HTTP correlation id. When `Some`, every emitted record
    /// gains a `"request_id":"..."` field after `run_id`.
    request_id: Option<String>,
    started_at: SystemTime,
}

#[derive(Debug)]
pub struct TerminalSummary {
    pub final_event: Option<&'static str>,
    pub turns: usize,
    pub completed: bool,
    /// `seq` of the last record this drain wrote, if any.
    pub last_seq: Option<u64>,
}

#[derive(Debug)]
pub enum RunLogError {
    Io(io::Error),
    /// The run id is not usable as a single filename component.
    InvalidRunId(String),
}

impl std::fmt::Display for RunLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunLogError::Io(e) => write!(f, "io: {e}"),
            RunLogError::InvalidRunId(why) => write!(f, "invalid run id: {why}"),
        }
    }
}

impl std::error::Error for RunLogError {}

impl From<io::Error> for RunLogError {
    fn from(e: io::Error) -> Self {
        RunLogError::Io(e)
    }
}

/// Check that `run_id` is safe to use as the `<run_id>.jsonl` filename:
/// 1..=[`MAX_RUN_ID_LEN`] bytes of `[A-Za-z0-9._:-]`, not starting with
/// `.`, and not containing `..`. Rejects path separators, whitespace and
/// control bytes, so an externally supplied id (CLI flag, env var,
/// `X-Request-ID` header) can never escape the runlog directory.
pub fn validate_run_id(run_id: &str) -> Result<(), RunLogError> {
    let bad = |why: String| Err(RunLogError::InvalidRunId(why));
    if run_id.is_empty() {
        return bad("empty".into());
    }
    if run_id.len() > MAX_RUN_ID_LEN {
        return bad(format!("{} bytes > {MAX_RUN_ID_LEN}", run_id.len()));
    }
    if run_id.starts_with('.') {
        return bad(format!("may not start with '.': {run_id:?}"));
    }
    if run_id.contains("..") {
        return bad(format!("may not contain '..': {run_id:?}"));
    }
    if let Some(c) =
        run_id.chars().find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ':')))
    {
        return bad(format!("disallowed character {c:?} in {run_id:?}"));
    }
    Ok(())
}

/// `true` when [`validate_run_id`] accepts `run_id`.
pub fn is_valid_run_id(run_id: &str) -> bool {
    validate_run_id(run_id).is_ok()
}

impl RunLog {
    /// Open `<dir>/<run_id>.jsonl` for append. Creates parent dirs. Writes
    /// an opening `start` record so every file is non-empty and timestamped.
    /// Fails with [`RunLogError::InvalidRunId`] if `run_id` is not a safe
    /// filename component.
    pub fn open(dir: impl AsRef<Path>, run_id: impl Into<String>) -> Result<Self, RunLogError> {
        Self::open_inner(dir.as_ref(), run_id.into(), None)
    }

    /// Like [`open`] but stamps every record with `request_id`. Used by
    /// the HTTP server path so log records can be joined against
    /// request-scoped metrics by the per-request correlation key.
    pub fn open_with_request_id(
        dir: impl AsRef<Path>,
        run_id: impl Into<String>,
        request_id: impl Into<String>,
    ) -> Result<Self, RunLogError> {
        Self::open_inner(dir.as_ref(), run_id.into(), Some(request_id.into()))
    }

    fn open_inner(
        dir: &Path,
        run_id: String,
        request_id: Option<String>,
    ) -> Result<Self, RunLogError> {
        validate_run_id(&run_id)?;
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{run_id}.jsonl"));
        let mut file = OpenOptions::new().create(true).read(true).append(true).open(&path)?;
        let next_seq = recover_tail(&mut file)?;
        let started_at = SystemTime::now();
        let log =
            Self { inner: Mutex::new(Inner { file, next_seq }), run_id, request_id, started_at };
        let started_ms = unix_ms(started_at);
        let mut payload = String::new();
        payload.push_str(r#"{"started_at_unix_ms":"#);
        push_u128(&mut payload, started_ms);
        payload.push('}');
        log.write_record("start", &payload)?;
        Ok(log)
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn started_at(&self) -> SystemTime {
        self.started_at
    }

    /// `seq` the next record will carry.
    pub fn next_seq(&self) -> u64 {
        self.inner.lock().expect("runlog mutex poisoned").next_seq
    }

    /// Flush written records to stable storage (`fdatasync`).
    pub fn sync(&self) -> Result<(), RunLogError> {
        let g = self.inner.lock().expect("runlog mutex poisoned");
        g.file.sync_data()?;
        Ok(())
    }

    /// One-shot blocking drain: receives `StreamEvent`s until the sender
    /// closes, writes one JSONL record per event. Returns the run summary;
    /// `final_event` is the variant name of the last terminal event seen
    /// (`done` / `cancelled` / `error`) or `None` if the sender dropped
    /// before emitting one. Each terminal record is `sync_data`'d before
    /// the drain continues, so `final_event: Some(_)` implies durability.
    pub fn drain(&self, rx: Receiver<StreamEvent>) -> Result<TerminalSummary, RunLogError> {
        let mut final_event: Option<&'static str> = None;
        let mut turns: usize = 0;
        let mut completed = false;
        let mut last_seq: Option<u64> = None;
        while let Ok(ev) = rx.recv() {
            let (kind, payload) = render_event(&ev);
            last_seq = Some(self.write_record_seq(kind, &payload)?);
            let terminal = match ev {
                StreamEvent::TurnComplete { turn, .. } => {
                    if turn > turns {
                        turns = turn;
                    }
                    false
                }
                StreamEvent::Done(pr) => {
                    final_event = Some("done");
                    if pr.turns > turns {
                        turns = pr.turns;
                    }
                    completed = pr.completed;
                    true
                }
                StreamEvent::Cancelled => {
                    final_event = Some("cancelled");
                    true
                }
                StreamEvent::Error(_) => {
                    final_event = Some("error");
                    true
                }
                _ => false,
            };
            if terminal {
                self.sync()?;
            }
        }
        Ok(TerminalSummary { final_event, turns, completed, last_seq })
    }

    /// Spawn the drain on its own thread. The caller waits on the
    /// `JoinHandle` after dropping the sender.
    pub fn drain_in_background(
        self: Arc<Self>,
        rx: Receiver<StreamEvent>,
    ) -> JoinHandle<Result<TerminalSummary, RunLogError>> {
        thread::spawn(move || self.drain(rx))
    }

    /// Manually write a record. Used internally for `start`; exposed so
    /// callers can emit their own markers around a streaming prompt.
    pub fn write_record(&self, kind: &str, payload: &str) -> Result<(), RunLogError> {
        self.write_record_seq(kind, payload).map(|_| ())
    }

    /// Like [`write_record`] but returns the `seq` assigned to the record.
    /// The counter only advances when the write succeeds.
    pub fn write_record_seq(&self, kind: &str, payload: &str) -> Result<u64, RunLogError> {
        let mut g = self.inner.lock().expect("runlog mutex poisoned");
        let seq = g.next_seq;
        let ts = unix_ms(SystemTime::now());
        let mut line = String::with_capacity(payload.len() + 80);
        line.push_str(r#"{"seq":"#);
        push_u128(&mut line, u128::from(seq));
        line.push_str(r#","ts_ms":"#);
        push_u128(&mut line, ts);
        line.push_str(r#","run_id":""#);
        escape_into(&mut line, &self.run_id);
        line.push('"');
        if let Some(rid) = &self.request_id {
            line.push_str(r#","request_id":""#);
            escape_into(&mut line, rid);
            line.push('"');
        }
        line.push_str(r#","kind":""#);
        escape_into(&mut line, kind);
        line.push_str(r#"","data":"#);
        line.push_str(payload);
        line.push_str("}\n");
        g.file.write_all(line.as_bytes())?;
        g.next_seq = seq + 1;
        Ok(seq)
    }
}

/// Prepare an existing run file for appending and return the next `seq`.
///
/// Terminates a torn (newline-less) final line so the next record starts
/// cleanly, then scans complete lines for the highest committed `seq`.
/// Lines without a leading `{"seq":N,` or a closing `}` (legacy pre-seq
/// records, torn fragments) are ignored. A torn record's seq can never
/// exceed the true next seq anyway: the counter only advances after a
/// successful write, so the crashed write's seq is reissued on resume.
fn recover_tail(file: &mut File) -> Result<u64, RunLogError> {
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(0);
    }
    file.seek(SeekFrom::Start(len - 1))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)?;
    if last[0] != b'\n' {
        // O_APPEND: lands at EOF regardless of the read cursor.
        file.write_all(b"\n")?;
    }
    file.seek(SeekFrom::Start(0))?;
    let mut next: u64 = 0;
    let mut reader = BufReader::new(&*file);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        // Only complete, well-formed records count. A torn fragment that an
        // earlier re-open newline-terminated still lacks its closing `}`.
        if buf.last() != Some(&b'\n') {
            break;
        }
        let rec = &buf[..buf.len() - 1];
        if rec.last() != Some(&b'}') {
            continue;
        }
        if let Some(seq) = parse_seq_prefix(rec) {
            next = next.max(seq.saturating_add(1));
        }
    }
    Ok(next)
}

/// Parse `N` out of a line beginning `{"seq":N,`.
fn parse_seq_prefix(line: &[u8]) -> Option<u64> {
    let rest = line.strip_prefix(br#"{"seq":"#)?;
    let end = rest.iter().position(|b| !b.is_ascii_digit())?;
    if end == 0 || rest[end] != b',' {
        return None;
    }
    std::str::from_utf8(&rest[..end]).ok()?.parse().ok()
}

fn unix_ms(t: SystemTime) -> u128 {
    t.duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

fn push_u128(out: &mut String, n: u128) {
    use std::fmt::Write;
    let _ = write!(out, "{n}");
}

fn push_usize(out: &mut String, n: usize) {
    use std::fmt::Write;
    let _ = write!(out, "{n}");
}

fn render_event(ev: &StreamEvent) -> (&'static str, String) {
    match ev {
        StreamEvent::TextDelta(s) => ("text_delta", obj_text(s)),
        StreamEvent::ToolUseStart { id, name } => {
            let mut p = String::new();
            p.push_str(r#"{"id":""#);
            escape_into(&mut p, id);
            p.push_str(r#"","name":""#);
            escape_into(&mut p, name);
            p.push_str(r#""}"#);
            ("tool_use_start", p)
        }
        StreamEvent::ToolUseInputDelta(s) => ("tool_use_input_delta", obj_text(s)),
        StreamEvent::BlockStop => ("block_stop", "{}".into()),
        StreamEvent::ToolResult { tool_use_id, content, is_error } => {
            let mut p = String::new();
            p.push_str(r#"{"tool_use_id":""#);
            escape_into(&mut p, tool_use_id);
            p.push_str(r#"","is_error":"#);
            p.push_str(if *is_error { "true" } else { "false" });
            p.push_str(r#","content":""#);
            escape_into(&mut p, content);
            p.push_str(r#""}"#);
            ("tool_result", p)
        }
        StreamEvent::TurnComplete { turn, stop_reason } => {
            let mut p = String::new();
            p.push_str(r#"{"turn":"#);
            push_usize(&mut p, *turn);
            p.push_str(r#","stop_reason":"#);
            match stop_reason {
                Some(s) => {
                    p.push('"');
                    escape_into(&mut p, s);
                    p.push('"');
                }
                None => p.push_str("null"),
            }
            p.push('}');
            ("turn_complete", p)
        }
        StreamEvent::Done(pr) => ("done", render_done(pr)),
        StreamEvent::Cancelled => ("cancelled", "{}".into()),
        StreamEvent::Error(e) => ("error", obj_message(&render_session_error(e))),
    }
}

fn obj_text(s: &str) -> String {
    let mut p = String::new();
    p.push_str(r#"{"text":""#);
    escape_into(&mut p, s);
    p.push_str(r#""}"#);
    p
}

fn obj_message(s: &str) -> String {
    let mut p = String::new();
    p.push_str(r#"{"message":""#);
    escape_into(&mut p, s);
    p.push_str(r#""}"#);
    p
}

fn render_done(pr: &PromptResult) -> String {
    let mut p = String::new();
    p.push_str(r#"{"text":""#);
    escape_into(&mut p, &pr.text);
    p.push_str(r#"","turns":"#);
    push_usize(&mut p, pr.turns);
    p.push_str(r#","completed":"#);
    p.push_str(if pr.completed { "true" } else { "false" });
    p.push('}');
    p
}

fn render_session_error(e: &SessionError) -> String {
    match e {
        SessionError::Model(s) => format!("model: {s}"),
        SessionError::Mailbox => "mailbox closed".into(),
        SessionError::TurnLimitExceeded => "turn limit exceeded".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::sync_channel;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch() -> PathBuf {
        let n =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        let c = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("runlog-test-{n}-{c}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(p: &Path) {
        let _ = std::fs::remove_dir_all(p);
    }

    fn read_lines(dir: &Path, run_id: &str) -> Vec<String> {
        let path = dir.join(format!("{run_id}.jsonl"));
        let bytes = std::fs::read(&path).expect("file");
        let s = String::from_utf8(bytes).expect("utf8");
        // trailing newline → last split is empty; drop it
        s.split('\n').filter(|l| !l.is_empty()).map(|l| l.to_string()).collect()
    }

    /// Find the substring `data":` and return what follows, so tests can
    /// match on payload shape without depending on the variable `ts_ms`.
    fn data_part(line: &str) -> &str {
        let pos = line.find(r#""data":"#).expect("data field");
        &line[pos + r#""data":"#.len()..line.len() - 1]
    }

    fn kind_of(line: &str) -> String {
        let needle = r#""kind":""#;
        let pos = line.find(needle).expect("kind");
        let rest = &line[pos + needle.len()..];
        let end = rest.find('"').unwrap();
        rest[..end].to_string()
    }

    #[test]
    fn open_creates_file_and_writes_start_record() {
        let dir = scratch();
        let _log = RunLog::open(&dir, "r1").unwrap();
        let lines = read_lines(&dir, "r1");
        assert_eq!(lines.len(), 1);
        assert_eq!(kind_of(&lines[0]), "start");
        assert!(lines[0].contains(r#""run_id":"r1""#));
        assert!(lines[0].contains(r#""started_at_unix_ms":"#));
        cleanup(&dir);
    }

    #[test]
    fn open_creates_parent_dirs() {
        let dir = scratch();
        let nested = dir.join("a/b/c");
        let _log = RunLog::open(&nested, "r1").unwrap();
        assert!(nested.join("r1.jsonl").exists());
        cleanup(&dir);
    }

    #[test]
    fn drain_writes_text_delta() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::TextDelta("hi".into())).unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert!(summary.final_event.is_none());
        let lines = read_lines(&dir, "r1");
        assert_eq!(lines.len(), 2);
        assert_eq!(kind_of(&lines[1]), "text_delta");
        assert_eq!(data_part(&lines[1]), r#"{"text":"hi"}"#);
        cleanup(&dir);
    }

    #[test]
    fn text_delta_escapes_quotes_and_newlines() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::TextDelta("he said \"hi\"\nbye".into())).unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        let data = data_part(&lines[1]);
        assert_eq!(data, r#"{"text":"he said \"hi\"\nbye"}"#);
        cleanup(&dir);
    }

    #[test]
    fn tool_use_start_records_id_and_name() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::ToolUseStart { id: "tu_1".into(), name: "bash".into() }).unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        assert_eq!(data_part(&lines[1]), r#"{"id":"tu_1","name":"bash"}"#);
        cleanup(&dir);
    }

    #[test]
    fn tool_result_with_is_error_true() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::ToolResult {
            tool_use_id: "tu_1".into(),
            content: "boom".into(),
            is_error: true,
        })
        .unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        let data = data_part(&lines[1]);
        assert!(data.contains(r#""is_error":true"#), "got: {data}");
        assert!(data.contains(r#""tool_use_id":"tu_1""#));
        assert!(data.contains(r#""content":"boom""#));
        cleanup(&dir);
    }

    #[test]
    fn tool_result_with_is_error_false() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::ToolResult {
            tool_use_id: "tu_1".into(),
            content: "ok".into(),
            is_error: false,
        })
        .unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        assert!(data_part(&lines[1]).contains(r#""is_error":false"#));
        cleanup(&dir);
    }

    #[test]
    fn turn_complete_with_and_without_stop_reason() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::TurnComplete { turn: 2, stop_reason: Some("end_turn".into()) })
            .unwrap();
        tx.send(StreamEvent::TurnComplete { turn: 3, stop_reason: None }).unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        assert_eq!(data_part(&lines[1]), r#"{"turn":2,"stop_reason":"end_turn"}"#);
        assert_eq!(data_part(&lines[2]), r#"{"turn":3,"stop_reason":null}"#);
        cleanup(&dir);
    }

    #[test]
    fn block_stop_and_input_delta() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::ToolUseInputDelta("{\"x\":1}".into())).unwrap();
        tx.send(StreamEvent::BlockStop).unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        assert_eq!(kind_of(&lines[1]), "tool_use_input_delta");
        assert_eq!(data_part(&lines[1]), r#"{"text":"{\"x\":1}"}"#);
        assert_eq!(kind_of(&lines[2]), "block_stop");
        assert_eq!(data_part(&lines[2]), "{}");
        cleanup(&dir);
    }

    #[test]
    fn sequence_preserves_order() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(16);
        tx.send(StreamEvent::TextDelta("a".into())).unwrap();
        tx.send(StreamEvent::TextDelta("b".into())).unwrap();
        tx.send(StreamEvent::TextDelta("c".into())).unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        assert_eq!(lines.len(), 4);
        assert_eq!(data_part(&lines[1]), r#"{"text":"a"}"#);
        assert_eq!(data_part(&lines[2]), r#"{"text":"b"}"#);
        assert_eq!(data_part(&lines[3]), r#"{"text":"c"}"#);
        cleanup(&dir);
    }

    #[test]
    fn drop_sender_mid_stream_no_terminal() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(4);
        tx.send(StreamEvent::TextDelta("partial".into())).unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert!(summary.final_event.is_none());
        assert_eq!(summary.turns, 0);
        assert!(!summary.completed);
        cleanup(&dir);
    }

    #[test]
    fn terminal_summary_done() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(4);
        tx.send(StreamEvent::Done(PromptResult {
            text: "final".into(),
            structured: None,
            completed: true,
            turns: 3,
        }))
        .unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert_eq!(summary.final_event, Some("done"));
        assert_eq!(summary.turns, 3);
        assert!(summary.completed);
        let lines = read_lines(&dir, "r1");
        assert_eq!(kind_of(&lines[1]), "done");
        assert_eq!(data_part(&lines[1]), r#"{"text":"final","turns":3,"completed":true}"#);
        cleanup(&dir);
    }

    #[test]
    fn terminal_summary_cancelled() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(4);
        tx.send(StreamEvent::Cancelled).unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert_eq!(summary.final_event, Some("cancelled"));
        assert!(!summary.completed);
        let lines = read_lines(&dir, "r1");
        assert_eq!(kind_of(&lines[1]), "cancelled");
        assert_eq!(data_part(&lines[1]), "{}");
        cleanup(&dir);
    }

    #[test]
    fn terminal_summary_error() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel(4);
        tx.send(StreamEvent::Error(SessionError::Model("nope".into()))).unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert_eq!(summary.final_event, Some("error"));
        let lines = read_lines(&dir, "r1");
        assert_eq!(kind_of(&lines[1]), "error");
        assert_eq!(data_part(&lines[1]), r#"{"message":"model: nope"}"#);
        cleanup(&dir);
    }

    #[test]
    fn drain_in_background_join_returns_ok() {
        let dir = scratch();
        let log = Arc::new(RunLog::open(&dir, "r1").unwrap());
        let (tx, rx) = sync_channel(4);
        let handle = log.clone().drain_in_background(rx);
        tx.send(StreamEvent::TextDelta("hi".into())).unwrap();
        tx.send(StreamEvent::Done(PromptResult {
            text: "bye".into(),
            structured: None,
            completed: true,
            turns: 1,
        }))
        .unwrap();
        drop(tx);
        let summary = handle.join().expect("thread").expect("ok");
        assert_eq!(summary.final_event, Some("done"));
        assert!(summary.completed);
        cleanup(&dir);
    }

    #[test]
    fn concurrent_write_record_lines_dont_interleave() {
        let dir = scratch();
        let log = Arc::new(RunLog::open(&dir, "r1").unwrap());
        // Build a payload that makes the full JSONL line exactly 100 bytes,
        // including the trailing '\n'. The overhead is:
        //   {"seq":<N>,"ts_ms":<MS>,"run_id":"r1","kind":"big","data":"<PAYLOAD>"}\n
        // where data is treated as a raw string in the payload (we feed it
        // as a verbatim JSON value via write_record).
        // To keep things simple, we just check each line ends with '\n' and
        // begins with `{"seq":` and that we get the expected line count.
        let mut handles = Vec::new();
        let payload = "x".repeat(40);
        for _ in 0..4 {
            let log = log.clone();
            let payload = payload.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..25 {
                    let p = format!(r#"{{"v":"{payload}"}}"#);
                    log.write_record("big", &p).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let lines = read_lines(&dir, "r1");
        // start + 4*25 records
        assert_eq!(lines.len(), 1 + 100);
        for (i, l) in lines.iter().enumerate() {
            assert!(l.starts_with(r#"{"seq":"#), "bad start: {l}");
            assert!(l.ends_with('}'), "bad end: {l}");
            // seq is assigned under the write lock: file order == seq order.
            assert_eq!(parse_seq_prefix(l.as_bytes()), Some(i as u64), "line {i}: {l}");
        }
        // Count exactly 100 "big" lines.
        let big = lines.iter().filter(|l| kind_of(l) == "big").count();
        assert_eq!(big, 100);
        cleanup(&dir);
    }

    #[test]
    fn run_id_with_special_chars_is_escaped() {
        let dir = scratch();
        // run_id used as filename — keep filename simple, but the field is
        // still escaped in the JSON.
        let log = RunLog::open(&dir, "with-quote").unwrap();
        log.write_record("foo", r#"{"k":"v"}"#).unwrap();
        let lines = read_lines(&dir, "with-quote");
        assert!(lines[1].contains(r#""run_id":"with-quote""#));
        cleanup(&dir);
    }

    #[test]
    fn append_to_existing_file_does_not_truncate() {
        let dir = scratch();
        {
            let log = RunLog::open(&dir, "r1").unwrap();
            let (tx, rx) = sync_channel(4);
            tx.send(StreamEvent::TextDelta("first".into())).unwrap();
            drop(tx);
            log.drain(rx).unwrap();
        }
        {
            let log = RunLog::open(&dir, "r1").unwrap();
            let (tx, rx) = sync_channel(4);
            tx.send(StreamEvent::TextDelta("second".into())).unwrap();
            drop(tx);
            log.drain(rx).unwrap();
        }
        let lines = read_lines(&dir, "r1");
        // start + delta, then start + delta again
        assert_eq!(lines.len(), 4);
        assert!(lines[1].contains("first"));
        assert!(lines[3].contains("second"));
        cleanup(&dir);
    }

    /// `open_with_request_id` stamps both the opening `start` record and
    /// every drained event with the same `request_id`, in the documented
    /// position (after `run_id`, before `kind`).
    #[test]
    fn open_with_request_id_stamps_start_and_subsequent_events() {
        let dir = scratch();
        let log = RunLog::open_with_request_id(&dir, "r1", "req-abc-12345678").unwrap();
        assert_eq!(log.run_id(), "r1");
        assert_eq!(log.request_id(), Some("req-abc-12345678"));

        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::TextDelta("hi".into())).unwrap();
        tx.send(StreamEvent::Done(PromptResult {
            text: "bye".into(),
            structured: None,
            completed: true,
            turns: 1,
        }))
        .unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert_eq!(summary.final_event, Some("done"));

        let lines = read_lines(&dir, "r1");
        // start + text_delta + done
        assert_eq!(lines.len(), 3);
        for line in &lines {
            assert!(
                line.contains(r#""request_id":"req-abc-12345678""#),
                "missing request_id on line: {line}"
            );
            // Field order: seq, ts_ms, run_id, request_id, kind, data
            let run_pos = line.find(r#""run_id":"#).unwrap();
            let req_pos = line.find(r#""request_id":"#).unwrap();
            let kind_pos = line.find(r#""kind":"#).unwrap();
            assert!(run_pos < req_pos && req_pos < kind_pos, "field order wrong: {line}");
        }
        // Sanity: omitting request_id (plain `open`) still works and the
        // field is absent.
        let dir2 = scratch();
        let log2 = RunLog::open(&dir2, "r2").unwrap();
        assert!(log2.request_id().is_none());
        let lines2 = read_lines(&dir2, "r2");
        assert!(!lines2[0].contains("request_id"), "unexpected: {}", lines2[0]);
        cleanup(&dir);
        cleanup(&dir2);
    }

    fn seqs(dir: &Path, run_id: &str) -> Vec<Option<u64>> {
        read_lines(dir, run_id).iter().map(|l| parse_seq_prefix(l.as_bytes())).collect()
    }

    #[test]
    fn seq_starts_at_zero_and_is_contiguous() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        assert_eq!(log.next_seq(), 1, "start record took seq 0");
        let (tx, rx) = sync_channel(8);
        tx.send(StreamEvent::TextDelta("a".into())).unwrap();
        tx.send(StreamEvent::TurnComplete { turn: 1, stop_reason: None }).unwrap();
        tx.send(StreamEvent::Done(PromptResult {
            text: "x".into(),
            structured: None,
            completed: true,
            turns: 1,
        }))
        .unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert_eq!(summary.last_seq, Some(3));
        assert_eq!(seqs(&dir, "r1"), vec![Some(0), Some(1), Some(2), Some(3)]);
        assert!(read_lines(&dir, "r1")[0].starts_with(r#"{"seq":0,"ts_ms":"#));
        cleanup(&dir);
    }

    #[test]
    fn drain_with_no_events_reports_no_last_seq() {
        let dir = scratch();
        let log = RunLog::open(&dir, "r1").unwrap();
        let (tx, rx) = sync_channel::<StreamEvent>(1);
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert_eq!(summary.last_seq, None);
        cleanup(&dir);
    }

    /// Re-opening the same run (a retry reusing the logical run id)
    /// continues the sequence instead of restarting at 0.
    #[test]
    fn reopen_continues_seq() {
        let dir = scratch();
        {
            let log = RunLog::open(&dir, "r1").unwrap();
            log.write_record("a", "{}").unwrap();
        }
        let log = RunLog::open(&dir, "r1").unwrap();
        assert_eq!(log.next_seq(), 3);
        assert_eq!(log.write_record_seq("b", "{}").unwrap(), 3);
        assert_eq!(seqs(&dir, "r1"), vec![Some(0), Some(1), Some(2), Some(3)]);
        cleanup(&dir);
    }

    /// A crash mid-write leaves a torn final line. Re-open terminates it so
    /// new records stay on their own lines, and the fragment's (uncommitted)
    /// seq is not counted.
    #[test]
    fn torn_tail_is_isolated_on_reopen() {
        let dir = scratch();
        {
            let log = RunLog::open(&dir, "r1").unwrap();
            log.write_record("a", "{}").unwrap();
        }
        let path = dir.join("r1.jsonl");
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(br#"{"seq":99,"ts_ms":1,"run_id":"r1","ki"#).unwrap();
        }
        let log = RunLog::open(&dir, "r1").unwrap();
        // committed: 0 (start), 1 (a); torn 99 ignored; reopen start = 2
        assert_eq!(log.next_seq(), 3);
        let lines = read_lines(&dir, "r1");
        assert_eq!(lines.len(), 4);
        assert!(lines[2].ends_with(r#""ki"#), "torn fragment kept on its own line");
        assert_eq!(kind_of(&lines[3]), "start");
        assert_eq!(parse_seq_prefix(lines[3].as_bytes()), Some(2));
        drop(log);
        // A later re-open must still not count the (now newline-terminated)
        // fragment.
        let log = RunLog::open(&dir, "r1").unwrap();
        assert_eq!(log.next_seq(), 4);
        cleanup(&dir);
    }

    /// Files written before `seq` existed are appended to cleanly; their
    /// records carry no seq, so numbering starts at 0.
    #[test]
    fn legacy_file_without_seq_starts_at_zero() {
        let dir = scratch();
        let path = dir.join("old.jsonl");
        std::fs::write(&path, "{\"ts_ms\":1,\"run_id\":\"old\",\"kind\":\"start\",\"data\":{}}\n")
            .unwrap();
        let log = RunLog::open(&dir, "old").unwrap();
        assert_eq!(log.next_seq(), 1);
        assert_eq!(seqs(&dir, "old"), vec![None, Some(0)]);
        cleanup(&dir);
    }

    #[test]
    fn parse_seq_prefix_cases() {
        assert_eq!(parse_seq_prefix(br#"{"seq":0,"ts_ms":1}"#), Some(0));
        assert_eq!(parse_seq_prefix(br#"{"seq":42,"x":1}"#), Some(42));
        assert_eq!(parse_seq_prefix(br#"{"seq":,"x":1}"#), None);
        assert_eq!(parse_seq_prefix(br#"{"seq":12"#), None);
        assert_eq!(parse_seq_prefix(br#"{"seq":1x,"#), None);
        assert_eq!(parse_seq_prefix(br#"{"ts_ms":1,"seq":3,"#), None);
    }

    #[test]
    fn validate_run_id_accepts_external_ids() {
        for ok in
            ["r1", "run-1726000000000", "bop:3fa2b9c1", "card.a_b-c", "A".repeat(128).as_str()]
        {
            assert!(validate_run_id(ok).is_ok(), "should accept {ok:?}");
        }
    }

    #[test]
    fn validate_run_id_rejects_unsafe_ids() {
        let long = "a".repeat(MAX_RUN_ID_LEN + 1);
        for bad in [
            "",
            "../escape",
            "a/b",
            "a\\b",
            ".hidden",
            "a..b",
            "has space",
            "tab\there",
            "nul\0",
            "caf\u{e9}",
            long.as_str(),
        ] {
            assert!(
                matches!(validate_run_id(bad), Err(RunLogError::InvalidRunId(_))),
                "should reject {bad:?}"
            );
            assert!(!is_valid_run_id(bad));
        }
    }

    /// An `X-Request-ID`-style traversal attempt must not create a file
    /// outside the runlog directory.
    #[test]
    fn open_rejects_traversal_without_touching_fs() {
        let dir = scratch();
        let inner = dir.join("logs");
        let err = RunLog::open_with_request_id(&inner, "../pwned", "../pwned").err().unwrap();
        assert!(matches!(err, RunLogError::InvalidRunId(_)), "{err}");
        assert!(!dir.join("pwned.jsonl").exists());
        assert!(!inner.exists(), "invalid id must fail before create_dir_all");
        cleanup(&dir);
    }
}
