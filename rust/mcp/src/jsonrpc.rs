//! JSON-RPC 2.0 framing.
//!
//! Tiny helpers to build requests and notifications, and to classify an
//! inbound frame as a response (carries `id`) or notification (does not).
//! Parsing the payload itself is done by `anthropic::json` — we just
//! inspect the top-level fields.

use anthropic::json::Json;

use crate::McpError;

/// What kind of inbound frame this is. `Response` carries the request id
/// and an outcome; `Notification` is silently dropped by the v1 client;
/// `Unknown` becomes a protocol error.
pub(crate) enum FrameKind {
    Response { id: u64, outcome: Outcome },
    Notification,
    Unknown,
}

/// Outcome of a response: either a `result` slice (still borrowing the
/// owning `Json`) or a server error. The caller takes ownership by
/// re-serialising the slice via [`crate::write_json`].
pub(crate) enum Outcome {
    Result,
    Error(McpError),
}

/// Classify a parsed top-level JSON-RPC value. Doesn't extract the
/// `result` subtree — the caller looks it up directly to avoid the
/// `Clone` round-trip.
pub(crate) fn classify(v: &Json) -> FrameKind {
    let id = v.get("id");
    let has_method = v.get("method").is_some();

    match (id, has_method) {
        (Some(id_v), false) => {
            let id_num = match id_num(id_v) {
                Some(n) => n,
                None => return FrameKind::Unknown,
            };
            let outcome = if let Some(err_v) = v.get("error") {
                match parse_error(err_v) {
                    Some(e) => Outcome::Error(e),
                    None => return FrameKind::Unknown,
                }
            } else {
                Outcome::Result
            };
            FrameKind::Response { id: id_num, outcome }
        }
        (None, true) | (Some(Json::Null), true) => FrameKind::Notification,
        _ => FrameKind::Unknown,
    }
}

/// Parse a request id. JSON-RPC allows string ids; we only send integers
/// so anything else is treated as unknown.
fn id_num(v: &Json) -> Option<u64> {
    match v {
        Json::Num(n) => n.parse::<u64>().ok(),
        _ => None,
    }
}

/// Parse the `error` object. We propagate `code` + `message`; `data` is
/// ignored (rarely populated, not worth a typed surface).
fn parse_error(v: &Json) -> Option<McpError> {
    let code = match v.get("code")? {
        Json::Num(n) => n.parse::<i64>().ok()?,
        _ => return None,
    };
    let message = v
        .get("message")
        .and_then(Json::as_str)
        .unwrap_or("")
        .to_string();
    Some(McpError::Server { code, message })
}

/// Build a JSON-RPC request line. `params` is a raw JSON object string
/// inserted verbatim; `None` omits the field.
pub(crate) fn build_request(id: u64, method: &str, params: Option<&str>) -> String {
    let mut s = String::with_capacity(48 + method.len() + params.map(str::len).unwrap_or(0));
    s.push_str(r#"{"jsonrpc":"2.0","id":"#);
    s.push_str(&id.to_string());
    s.push_str(r#","method":""#);
    anthropic::json::escape_into(&mut s, method);
    s.push('"');
    if let Some(p) = params {
        s.push_str(r#","params":"#);
        s.push_str(p);
    }
    s.push('}');
    s
}

/// Build a JSON-RPC notification line (no `id`).
pub(crate) fn build_notification(method: &str, params: Option<&str>) -> String {
    let mut s = String::with_capacity(32 + method.len() + params.map(str::len).unwrap_or(0));
    s.push_str(r#"{"jsonrpc":"2.0","method":""#);
    anthropic::json::escape_into(&mut s, method);
    s.push('"');
    if let Some(p) = params {
        s.push_str(r#","params":"#);
        s.push_str(p);
    }
    s.push('}');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use anthropic::json::parse as parse_json;

    #[test]
    fn build_request_with_params() {
        let s = build_request(7, "tools/call", Some(r#"{"name":"x"}"#));
        assert_eq!(
            s,
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"x"}}"#
        );
    }

    #[test]
    fn build_request_without_params() {
        let s = build_request(1, "tools/list", None);
        assert_eq!(s, r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#);
    }

    #[test]
    fn build_notification_has_no_id() {
        let s = build_notification("notifications/initialized", None);
        assert_eq!(s, r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
    }

    #[test]
    fn classify_response_with_result() {
        let v = parse_json(br#"{"jsonrpc":"2.0","id":2,"result":{"ok":true}}"#).unwrap();
        match classify(&v) {
            FrameKind::Response { id, outcome: Outcome::Result } => assert_eq!(id, 2),
            _ => panic!("not response"),
        }
    }

    #[test]
    fn classify_response_with_error() {
        let v = parse_json(
            br#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"no such method"}}"#,
        )
        .unwrap();
        match classify(&v) {
            FrameKind::Response {
                id,
                outcome: Outcome::Error(McpError::Server { code, message }),
            } => {
                assert_eq!(id, 1);
                assert_eq!(code, -32601);
                assert_eq!(message, "no such method");
            }
            _ => panic!("not error response"),
        }
    }

    #[test]
    fn classify_notification_has_method_no_id() {
        let v = parse_json(br#"{"jsonrpc":"2.0","method":"notifications/log","params":{}}"#)
            .unwrap();
        assert!(matches!(classify(&v), FrameKind::Notification));
    }

    #[test]
    fn classify_null_id_with_method_is_notification() {
        // Strict JSON-RPC says notifications must omit id, but some clients
        // emit `id: null`. Tolerate.
        let v = parse_json(br#"{"jsonrpc":"2.0","id":null,"method":"x"}"#).unwrap();
        assert!(matches!(classify(&v), FrameKind::Notification));
    }

    #[test]
    fn classify_unknown_shape() {
        let v = parse_json(br#"{"jsonrpc":"2.0"}"#).unwrap();
        assert!(matches!(classify(&v), FrameKind::Unknown));
    }
}
