//! JSON parser benches (anthropic's hand-rolled parser).

use std::hint::black_box;

use anthropic::json::parse;

use crate::bench_helper::bench;

#[test]
fn bench_json_text_delta_event() {
    // Representative SSE event body: a content_block_delta with a
    // ~200-character text_delta. This is what the streaming loop sees
    // many times per response.
    let delta = "a".repeat(200);
    let payload = format!(
        r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{delta}"}}}}"#
    );
    let bytes = payload.as_bytes();
    bench("anthropic::json::parse text_delta (~200 char)", || {
        let v = black_box(parse(black_box(bytes)).unwrap());
        drop(black_box(v));
    });
}

#[test]
fn bench_json_1k_object() {
    // 1 KiB-ish JSON object with mixed scalars and a nested array.
    // Stays within RFC 8259, no string escapes.
    let mut s = String::from(
        r#"{"id":"msg_01ABCDEFG","type":"message","role":"assistant","model":"claude-opus-4-7","content":[{"type":"text","text":""#,
    );
    while s.len() < 950 {
        s.push_str("lorem ipsum dolor sit amet ");
    }
    s.push_str(r#""}],"stop_reason":"end_turn","usage":{"input_tokens":42,"output_tokens":128}}"#);
    let bytes = s.as_bytes();
    assert!(bytes.len() >= 1000, "test payload too small: {}", bytes.len());
    bench("anthropic::json::parse ~1 KiB object", || {
        let v = black_box(parse(black_box(bytes)).unwrap());
        drop(black_box(v));
    });
}
