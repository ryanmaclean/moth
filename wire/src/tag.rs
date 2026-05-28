//! XML-style tag body extraction.
//!
//! `find_tag(haystack, "output")` returns the bytes between the first
//! `<output>` and the next `</output>`. Case-sensitive. No nesting
//! support — the first `</tag>` after the matching open wins. Designed
//! for promise-tag completion signals (`<promise>COMPLETE</promise>`)
//! and structured-output extraction.

use crate::scan::scan_for_byte;

/// Find the body of the first `<tag>...</tag>` occurrence in `haystack`.
/// Returns `None` if either marker is missing.
///
/// Panics in debug builds if `tag` is empty.
pub fn find_tag<'a>(haystack: &'a [u8], tag: &[u8]) -> Option<&'a [u8]> {
    debug_assert!(!tag.is_empty(), "tag name must be non-empty");

    let mut open = Vec::with_capacity(tag.len() + 2);
    open.push(b'<');
    open.extend_from_slice(tag);
    open.push(b'>');

    let mut close = Vec::with_capacity(tag.len() + 3);
    close.push(b'<');
    close.push(b'/');
    close.extend_from_slice(tag);
    close.push(b'>');

    let start = find_subslice(haystack, &open)?;
    let body_start = start + open.len();
    let close_off = find_subslice(&haystack[body_start..], &close)?;
    Some(&haystack[body_start..body_start + close_off])
}

/// Find the first occurrence of `needle` in `haystack`. Anchors on the
/// first byte via SIMD scan, then memcmps. Cheap for short needles (tag
/// names are typically <20 bytes).
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    let first = needle[0];
    let mut from = 0;
    while from + needle.len() <= haystack.len() {
        let off = scan_for_byte(&haystack[from..], first)?;
        let pos = from + off;
        if pos + needle.len() > haystack.len() {
            return None;
        }
        if &haystack[pos..pos + needle.len()] == needle {
            return Some(pos);
        }
        from = pos + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(find_tag(b"<promise>COMPLETE</promise>", b"promise"), Some(&b"COMPLETE"[..]));
    }

    #[test]
    fn with_surrounding_text() {
        let s = b"prelude <output>{\"x\": 1}</output> trailer";
        assert_eq!(find_tag(s, b"output"), Some(&b"{\"x\": 1}"[..]));
    }

    #[test]
    fn empty_body() {
        assert_eq!(find_tag(b"<x></x>", b"x"), Some(&b""[..]));
    }

    #[test]
    fn missing_open() {
        assert_eq!(find_tag(b"COMPLETE</promise>", b"promise"), None);
    }

    #[test]
    fn missing_close() {
        assert_eq!(find_tag(b"<promise>COMPLETE", b"promise"), None);
    }

    #[test]
    fn first_occurrence_wins() {
        let s = b"<x>1</x>noise<x>2</x>";
        assert_eq!(find_tag(s, b"x"), Some(&b"1"[..]));
    }

    #[test]
    fn nested_uses_first_close() {
        // <x><x>inner</x></x> — the first </x> closes the outer match.
        let s = b"<x><x>inner</x></x>";
        assert_eq!(find_tag(s, b"x"), Some(&b"<x>inner"[..]));
    }

    #[test]
    fn case_sensitive() {
        assert_eq!(find_tag(b"<TAG>body</TAG>", b"tag"), None);
        assert_eq!(find_tag(b"<TAG>body</TAG>", b"TAG"), Some(&b"body"[..]));
    }

    #[test]
    fn similar_tag_name_does_not_match() {
        let s = b"<outputs>bad</outputs><output>good</output>";
        assert_eq!(find_tag(s, b"output"), Some(&b"good"[..]));
    }

    #[test]
    fn tag_at_start_and_end() {
        let s = b"<a>x</a>";
        assert_eq!(find_tag(s, b"a"), Some(&b"x"[..]));
    }

    #[test]
    fn long_buffer_to_force_simd() {
        let mut buf = vec![b'.'; 300];
        buf.extend_from_slice(b"<p>");
        buf.extend_from_slice(&vec![b'!'; 300]);
        buf.extend_from_slice(b"</p>");
        let body = find_tag(&buf, b"p").unwrap();
        assert_eq!(body.len(), 300);
        assert!(body.iter().all(|&b| b == b'!'));
    }
}
