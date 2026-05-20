//! SSE frame → Event(s) mapping for the OpenAI chat-completions stream.
//!
//! Each frame is a single `data: <json>` payload (chat completions doesn't
//! emit `event:` lines). The payload itself can encode several semantic
//! events at once — a text chunk, a tool-call start, a tool-call args delta,
//! and a finish reason can all appear in the same `choices[0].delta` — so
//! `parse_frame` returns a `Vec<Event>` rather than a single one.
//!
//! Canonical stop reasons (matching the anthropic crate, so callers can
//! share downstream logic):
//!   - `finish_reason: "stop"`       → `Stop { reason: Some("end_turn") }`
//!   - `finish_reason: "tool_calls"` → `Stop { reason: Some("tool_use") }`
//!   - everything else (incl. `length`) is forwarded unchanged.

use crate::json::{Json, parse as parse_json};
use crate::{Error, Event};

/// Result of parsing one SSE frame.
///
/// `Terminate` means the stream itself is over (we saw `data: [DONE]`).
/// `Events` is a (possibly empty) list of events extracted from the frame —
/// empty for pings, comment lines, or frames that carry only metadata.
#[derive(Debug, PartialEq)]
pub(crate) enum Parsed {
    Events(Vec<Event>),
    Terminate,
}

pub(crate) fn parse_frame(frame: &[u8]) -> Result<Parsed, Error> {
    let mut data: Vec<u8> = Vec::new();

    for line in frame.split(|&b| b == b'\n') {
        // SSE comments start with ':' — ignore.
        if line.first() == Some(&b':') {
            continue;
        }
        if let Some(rest) = strip_prefix(line, b"data:") {
            if !data.is_empty() {
                data.push(b'\n');
            }
            data.extend_from_slice(trim(rest));
        }
        // We don't care about `event:`, `id:`, `retry:` here.
    }

    if data.is_empty() {
        return Ok(Parsed::Events(Vec::new()));
    }

    if data == b"[DONE]" {
        return Ok(Parsed::Terminate);
    }

    let v = parse_json(&data)?;

    // Surface server-side errors. OpenAI signals these as `{"error": {...}}`
    // either as a non-stream response that got proxied through, or as the
    // last SSE event when something blew up mid-stream.
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(Json::as_str)
            .unwrap_or("openai stream error")
            .to_string();
        return Err(Error::Http(msg));
    }

    let choices = v
        .get("choices")
        .and_then(Json::as_arr)
        .ok_or_else(|| Error::InvalidResponse("missing choices".into()))?;
    let Some(choice) = choices.first() else {
        return Ok(Parsed::Events(Vec::new()));
    };

    let mut out = Vec::new();
    if let Some(delta) = choice.get("delta") {
        emit_delta(delta, &mut out);
    }

    if let Some(reason) = choice.get("finish_reason").and_then(Json::as_str) {
        // A finished tool-call run terminates the current block too — emit a
        // ContentBlockStop so callers don't have to special-case the boundary.
        let canonical = match reason {
            "stop" => Some("end_turn".to_string()),
            "tool_calls" => {
                out.push(Event::ContentBlockStop);
                Some("tool_use".to_string())
            }
            other => Some(other.to_string()),
        };
        out.push(Event::Stop { reason: canonical });
    }

    Ok(Parsed::Events(out))
}

fn emit_delta(delta: &Json, out: &mut Vec<Event>) {
    if let Some(text) = delta.get("content").and_then(Json::as_str)
        && !text.is_empty()
    {
        out.push(Event::TextDelta(text.to_string()));
    }

    if let Some(calls) = delta.get("tool_calls").and_then(Json::as_arr) {
        for call in calls {
            // Some providers omit `id` after the opening delta; presence of
            // both `id` and `function.name` is what defines "start of a new
            // tool_use block". After the start, subsequent deltas typically
            // carry only `function.arguments`.
            let id = call.get("id").and_then(Json::as_str);
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Json::as_str);
            if let (Some(id), Some(name)) = (id, name)
                && !id.is_empty()
                && !name.is_empty()
            {
                out.push(Event::ToolUseStart {
                    id: id.to_string(),
                    name: name.to_string(),
                });
            }
            if let Some(args) = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Json::as_str)
                && !args.is_empty()
            {
                out.push(Event::ToolUseInputDelta(args.to_string()));
            }
        }
    }
}

fn strip_prefix<'a>(line: &'a [u8], prefix: &[u8]) -> Option<&'a [u8]> {
    if line.len() >= prefix.len() && &line[..prefix.len()] == prefix {
        Some(&line[prefix.len()..])
    } else {
        None
    }
}

fn trim(s: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = s.len();
    while start < end && matches!(s[start], b' ' | b'\t' | b'\r') {
        start += 1;
    }
    while end > start && matches!(s[end - 1], b' ' | b'\t' | b'\r') {
        end -= 1;
    }
    &s[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(frame: &[u8]) -> Vec<Event> {
        match parse_frame(frame).unwrap() {
            Parsed::Events(v) => v,
            Parsed::Terminate => panic!("unexpected terminate"),
        }
    }

    #[test]
    fn text_delta() {
        let frame = br#"data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#;
        assert_eq!(events(frame), vec![Event::TextDelta("Hel".into())]);
    }

    #[test]
    fn text_delta_with_escapes() {
        let frame = br#"data: {"choices":[{"delta":{"content":"a\nb"}}]}"#;
        assert_eq!(events(frame), vec![Event::TextDelta("a\nb".into())]);
    }

    #[test]
    fn tool_call_start_then_args() {
        let frame = br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"bash","arguments":""}}]}}]}"#;
        assert_eq!(
            events(frame),
            vec![Event::ToolUseStart {
                id: "call_abc".into(),
                name: "bash".into()
            }]
        );

        let frame2 = br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]}}]}"#;
        assert_eq!(
            events(frame2),
            vec![Event::ToolUseInputDelta("{\"command\":".into())]
        );
    }

    #[test]
    fn tool_call_start_with_initial_args_emits_both() {
        let frame = br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"x\":1}"}}]}}]}"#;
        assert_eq!(
            events(frame),
            vec![
                Event::ToolUseStart { id: "call_1".into(), name: "bash".into() },
                Event::ToolUseInputDelta("{\"x\":1}".into()),
            ]
        );
    }

    #[test]
    fn finish_reason_stop_canonicalises_to_end_turn() {
        let frame = br#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert_eq!(
            events(frame),
            vec![Event::Stop { reason: Some("end_turn".into()) }]
        );
    }

    #[test]
    fn finish_reason_tool_calls_canonicalises_and_emits_block_stop() {
        let frame = br#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        assert_eq!(
            events(frame),
            vec![
                Event::ContentBlockStop,
                Event::Stop { reason: Some("tool_use".into()) },
            ]
        );
    }

    #[test]
    fn finish_reason_length_forwarded_verbatim() {
        let frame = br#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#;
        assert_eq!(
            events(frame),
            vec![Event::Stop { reason: Some("length".into()) }]
        );
    }

    #[test]
    fn done_terminator() {
        let frame = b"data: [DONE]";
        assert_eq!(parse_frame(frame).unwrap(), Parsed::Terminate);
    }

    #[test]
    fn empty_frame_is_no_events() {
        // Ping-style keepalive — a frame with only a comment line.
        let frame = b": keep-alive";
        assert_eq!(parse_frame(frame).unwrap(), Parsed::Events(Vec::new()));
    }

    #[test]
    fn empty_data_frame_is_no_events() {
        let frame = b"";
        assert_eq!(parse_frame(frame).unwrap(), Parsed::Events(Vec::new()));
    }

    #[test]
    fn empty_content_string_is_skipped() {
        // Some providers emit a {"role":"assistant","content":""} bootstrap
        // frame; don't surface that as a no-op TextDelta.
        let frame = br#"data: {"choices":[{"delta":{"role":"assistant","content":""}}]}"#;
        assert_eq!(events(frame), Vec::<Event>::new());
    }

    #[test]
    fn malformed_json_errors() {
        let frame = b"data: {not json}";
        assert!(matches!(
            parse_frame(frame),
            Err(Error::InvalidResponse(_))
        ));
    }

    #[test]
    fn error_payload_returns_http_err() {
        let frame = br#"data: {"error":{"message":"rate limited","type":"rate_limit"}}"#;
        match parse_frame(frame) {
            Err(Error::Http(m)) => assert!(m.contains("rate limited")),
            other => panic!("expected Http err, got {other:?}"),
        }
    }

    #[test]
    fn text_and_finish_in_one_frame() {
        // Not common from OpenAI itself but some compatible servers (e.g. some
        // local llama.cpp builds) fuse the last delta with the finish reason.
        let frame = br#"data: {"choices":[{"delta":{"content":"bye"},"finish_reason":"stop"}]}"#;
        assert_eq!(
            events(frame),
            vec![
                Event::TextDelta("bye".into()),
                Event::Stop { reason: Some("end_turn".into()) },
            ]
        );
    }

    #[test]
    fn multi_line_data_concatenated() {
        // SSE spec: multiple data: lines in a frame are joined with '\n'.
        // OpenAI doesn't actually emit these split, but the parser should
        // tolerate it.
        let frame = b"data: {\"choices\":[{\"delta\":\ndata: {\"content\":\"hi\"}}]}";
        assert_eq!(events(frame), vec![Event::TextDelta("hi".into())]);
    }
}
