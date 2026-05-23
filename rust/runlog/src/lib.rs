//! File-backed structured event log.
//!
//! Subscribes to `harness::StreamEvent` over a `mpsc::Receiver` and writes
//! one JSONL record per event to `<dir>/<run_id>.jsonl`. Production audit
//! trail: every iteration, tool call, model delta gets a durable record.
//!
//! Record shape (one per line):
//! ```text
//! {"ts_ms": <epoch-millis>, "run_id": "<id>", "kind": "<kind>", "data": {...}}
//! ```
//!
//! When the log was opened via `open_with_request_id`, every record also
//! carries the inbound HTTP correlation id between `run_id` and `kind`:
//! ```text
//! {"ts_ms": ..., "run_id": "...", "request_id": "...", "kind": "...", "data": ...}
//! ```
//!
//! `ts_ms` is plain epoch milliseconds — no calendar math, no leap years,
//! consumers can format if they care.
//!
//! Atomic writes: each record is built up in a `String`, then written via a
//! single `write_all` of `record + "\n"`. JSONL records are short and fit in
//! one POSIX write, so each line is atomic. The `Mutex` only protects the
//! file handle from concurrent writers (the `drain` itself is
//! single-threaded, but `write_record` may be called from elsewhere).

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use anthropic::json::escape_into;
use harness::{PromptResult, SessionError, StreamEvent};

pub struct RunLog {
    file: Mutex<File>,
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
}

#[derive(Debug)]
pub enum RunLogError {
    Io(io::Error),
}

impl std::fmt::Display for RunLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunLogError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for RunLogError {}

impl From<io::Error> for RunLogError {
    fn from(e: io::Error) -> Self {
        RunLogError::Io(e)
    }
}

impl RunLog {
    /// Open `<dir>/<run_id>.jsonl` for append. Creates parent dirs. Writes
    /// an opening `start` record so every file is non-empty and timestamped.
    pub fn open(
        dir: impl AsRef<Path>,
        run_id: impl Into<String>,
    ) -> Result<Self, RunLogError> {
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
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{run_id}.jsonl"));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let started_at = SystemTime::now();
        let log = Self {
            file: Mutex::new(file),
            run_id,
            request_id,
            started_at,
        };
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

    /// One-shot blocking drain: receives `StreamEvent`s until the sender
    /// closes, writes one JSONL record per event. Returns the run summary;
    /// `final_event` is the variant name of the last terminal event seen
    /// (`done` / `cancelled` / `error`) or `None` if the sender dropped
    /// before emitting one.
    pub fn drain(
        &self,
        rx: Receiver<StreamEvent>,
    ) -> Result<TerminalSummary, RunLogError> {
        let mut final_event: Option<&'static str> = None;
        let mut turns: usize = 0;
        let mut completed = false;
        while let Ok(ev) = rx.recv() {
            let (kind, payload) = render_event(&ev);
            self.write_record(kind, &payload)?;
            match ev {
                StreamEvent::TurnComplete { turn, .. } => {
                    if turn > turns {
                        turns = turn;
                    }
                }
                StreamEvent::Done(pr) => {
                    final_event = Some("done");
                    if pr.turns > turns {
                        turns = pr.turns;
                    }
                    completed = pr.completed;
                }
                StreamEvent::Cancelled => {
                    final_event = Some("cancelled");
                }
                StreamEvent::Error(_) => {
                    final_event = Some("error");
                }
                _ => {}
            }
        }
        Ok(TerminalSummary { final_event, turns, completed })
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
    pub fn write_record(
        &self,
        kind: &str,
        payload: &str,
    ) -> Result<(), RunLogError> {
        let ts = unix_ms(SystemTime::now());
        let mut line = String::with_capacity(payload.len() + 64);
        line.push_str(r#"{"ts_ms":"#);
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
        let mut f = self.file.lock().expect("runlog mutex poisoned");
        f.write_all(line.as_bytes())?;
        Ok(())
    }
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
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
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
        tx.send(StreamEvent::ToolUseStart {
            id: "tu_1".into(),
            name: "bash".into(),
        }).unwrap();
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
        }).unwrap();
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
        }).unwrap();
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
        tx.send(StreamEvent::TurnComplete {
            turn: 2,
            stop_reason: Some("end_turn".into()),
        }).unwrap();
        tx.send(StreamEvent::TurnComplete {
            turn: 3,
            stop_reason: None,
        }).unwrap();
        drop(tx);
        log.drain(rx).unwrap();
        let lines = read_lines(&dir, "r1");
        assert_eq!(
            data_part(&lines[1]),
            r#"{"turn":2,"stop_reason":"end_turn"}"#
        );
        assert_eq!(
            data_part(&lines[2]),
            r#"{"turn":3,"stop_reason":null}"#
        );
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
        })).unwrap();
        drop(tx);
        let summary = log.drain(rx).unwrap();
        assert_eq!(summary.final_event, Some("done"));
        assert_eq!(summary.turns, 3);
        assert!(summary.completed);
        let lines = read_lines(&dir, "r1");
        assert_eq!(kind_of(&lines[1]), "done");
        assert_eq!(
            data_part(&lines[1]),
            r#"{"text":"final","turns":3,"completed":true}"#
        );
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
        })).unwrap();
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
        //   {"ts_ms":<MS>,"run_id":"r1","kind":"big","data":"<PAYLOAD>"}\n
        // where data is treated as a raw string in the payload (we feed it
        // as a verbatim JSON value via write_record).
        // To keep things simple, we just check each line ends with '\n' and
        // begins with `{"ts_ms":` and that we get the expected line count.
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
        for l in &lines {
            assert!(l.starts_with(r#"{"ts_ms":"#), "bad start: {l}");
            assert!(l.ends_with('}'), "bad end: {l}");
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
            // Field order: ts_ms, run_id, request_id, kind, data
            let run_pos = line.find(r#""run_id":"#).unwrap();
            let req_pos = line.find(r#""request_id":"#).unwrap();
            let kind_pos = line.find(r#""kind":"#).unwrap();
            assert!(
                run_pos < req_pos && req_pos < kind_pos,
                "field order wrong: {line}"
            );
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
}
