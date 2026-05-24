//! SSE frame → Event mapping for the Anthropic Messages stream.
//!
//! Each frame is an `event:` line and one or more `data:` lines (with optional
//! `id:` etc. we ignore). We use `event:` only to short-circuit `ping` and
//! to label `Other(name)` for forward-compat. Everything we actually parse is
//! the JSON in `data:` — the `type` field there is the source of truth.

use crate::json::{Json, parse as parse_json};
use crate::{Error, Event};

pub(crate) fn parse_frame(frame: &[u8]) -> Result<Event, Error> {
    let mut event_name: Option<&[u8]> = None;
    let mut data: Vec<u8> = Vec::new();

    for line in frame.split(|&b| b == b'\n') {
        if let Some(rest) = strip_prefix(line, b"event:") {
            event_name = Some(trim(rest));
        } else if let Some(rest) = strip_prefix(line, b"data:") {
            if !data.is_empty() {
                data.push(b'\n');
            }
            data.extend_from_slice(trim(rest));
        }
    }

    if let Some(name) = event_name
        && name == b"ping"
    {
        return Ok(Event::Ping);
    }

    if data.is_empty() {
        return Ok(Event::Other(
            event_name.and_then(|n| std::str::from_utf8(n).ok()).unwrap_or("").to_string(),
        ));
    }

    let v = parse_json(&data)?;
    let ty = v
        .get("type")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::InvalidResponse("missing type".into()))?;

    match ty {
        "message_start" => {
            let msg = v
                .get("message")
                .ok_or_else(|| Error::InvalidResponse("message_start: no message".into()))?;
            let id = msg
                .get("id")
                .and_then(Json::as_str)
                .ok_or_else(|| Error::InvalidResponse("message_start: no id".into()))?
                .to_string();
            let model = msg
                .get("model")
                .and_then(Json::as_str)
                .ok_or_else(|| Error::InvalidResponse("message_start: no model".into()))?
                .to_string();
            Ok(Event::MessageStart { id, model })
        }
        "content_block_start" => {
            let block = v.get("content_block").ok_or_else(|| {
                Error::InvalidResponse("content_block_start: no content_block".into())
            })?;
            let block_ty = block.get("type").and_then(Json::as_str).ok_or_else(|| {
                Error::InvalidResponse("content_block_start: no block type".into())
            })?;
            if block_ty == "tool_use" {
                let id = block
                    .get("id")
                    .and_then(Json::as_str)
                    .ok_or_else(|| Error::InvalidResponse("tool_use start: no id".into()))?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or_else(|| Error::InvalidResponse("tool_use start: no name".into()))?
                    .to_string();
                Ok(Event::ToolUseStart { id, name })
            } else {
                Ok(Event::Other(format!("content_block_start:{block_ty}")))
            }
        }
        "content_block_delta" => {
            let delta = v
                .get("delta")
                .ok_or_else(|| Error::InvalidResponse("content_block_delta: no delta".into()))?;
            let delta_ty = delta.get("type").and_then(Json::as_str).ok_or_else(|| {
                Error::InvalidResponse("content_block_delta: no delta type".into())
            })?;
            match delta_ty {
                "text_delta" => {
                    let text = delta
                        .get("text")
                        .and_then(Json::as_str)
                        .ok_or_else(|| Error::InvalidResponse("text_delta: no text".into()))?
                        .to_string();
                    Ok(Event::TextDelta(text))
                }
                "input_json_delta" => {
                    let partial = delta
                        .get("partial_json")
                        .and_then(Json::as_str)
                        .ok_or_else(|| {
                            Error::InvalidResponse("input_json_delta: no partial_json".into())
                        })?
                        .to_string();
                    Ok(Event::ToolUseInputDelta(partial))
                }
                other => Ok(Event::Other(format!("content_block_delta:{other}"))),
            }
        }
        "content_block_stop" => Ok(Event::ContentBlockStop),
        "message_delta" => {
            let stop_reason =
                v.get("delta").and_then(|d| d.get("stop_reason")).and_then(|sr| match sr {
                    Json::Str(s) => Some(s.clone()),
                    _ => None,
                });
            Ok(Event::MessageDelta { stop_reason })
        }
        "message_stop" => Ok(Event::MessageStop),
        "ping" => Ok(Event::Ping),
        "error" => {
            let msg = v
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(Json::as_str)
                .unwrap_or("anthropic stream error")
                .to_string();
            Err(Error::Http(msg))
        }
        other => Ok(Event::Other(other.to_string())),
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

    #[test]
    fn text_delta() {
        let frame = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}";
        assert_eq!(parse_frame(frame).unwrap(), Event::TextDelta("Hello".into()));
    }

    #[test]
    fn text_delta_with_escapes() {
        let frame = br#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a\n\"b\""}}"#;
        assert_eq!(parse_frame(frame).unwrap(), Event::TextDelta("a\n\"b\"".into()));
    }

    #[test]
    fn message_start() {
        let frame = br#"event: message_start
data: {"type":"message_start","message":{"id":"msg_01","model":"claude-opus-4-7","role":"assistant","content":[],"stop_reason":null}}"#;
        assert_eq!(
            parse_frame(frame).unwrap(),
            Event::MessageStart { id: "msg_01".into(), model: "claude-opus-4-7".into() }
        );
    }

    #[test]
    fn tool_use_start() {
        let frame = br#"event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}"#;
        assert_eq!(
            parse_frame(frame).unwrap(),
            Event::ToolUseStart { id: "toolu_01".into(), name: "get_weather".into() }
        );
    }

    #[test]
    fn input_json_delta() {
        let frame = br#"event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}"#;
        assert_eq!(parse_frame(frame).unwrap(), Event::ToolUseInputDelta("{\"location\":".into()));
    }

    #[test]
    fn content_block_stop() {
        let frame = br#"event: content_block_stop
data: {"type":"content_block_stop","index":0}"#;
        assert_eq!(parse_frame(frame).unwrap(), Event::ContentBlockStop);
    }

    #[test]
    fn message_delta_with_stop_reason() {
        let frame = br#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#;
        assert_eq!(
            parse_frame(frame).unwrap(),
            Event::MessageDelta { stop_reason: Some("end_turn".into()) }
        );
    }

    #[test]
    fn message_delta_null_stop_reason() {
        let frame = br#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":null}}"#;
        assert_eq!(parse_frame(frame).unwrap(), Event::MessageDelta { stop_reason: None });
    }

    #[test]
    fn message_stop() {
        let frame = br#"event: message_stop
data: {"type":"message_stop"}"#;
        assert_eq!(parse_frame(frame).unwrap(), Event::MessageStop);
    }

    #[test]
    fn ping_via_event_name() {
        let frame = b"event: ping\ndata: {}";
        assert_eq!(parse_frame(frame).unwrap(), Event::Ping);
    }

    #[test]
    fn unknown_event_type() {
        let frame = br#"event: future_thing
data: {"type":"future_thing","x":1}"#;
        assert_eq!(parse_frame(frame).unwrap(), Event::Other("future_thing".into()));
    }

    #[test]
    fn malformed_json_bubbles_up() {
        let frame = b"event: message_start\ndata: {not json}";
        assert!(parse_frame(frame).is_err());
    }

    #[test]
    fn error_event_returns_err() {
        let frame = br#"event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"server overloaded"}}"#;
        match parse_frame(frame) {
            Err(Error::Http(m)) => assert!(m.contains("overloaded")),
            other => panic!("expected Http err, got {other:?}"),
        }
    }

    #[test]
    fn multi_line_data_concat() {
        let frame = b"event: message_stop\ndata: {\"type\":\ndata: \"message_stop\"}";
        assert_eq!(parse_frame(frame).unwrap(), Event::MessageStop);
    }
}
