//! Tokenizer.
//!
//! Splits a script into tokens while resolving quoting and escapes, but
//! leaves variable expansion and command substitution as deferred parts
//! to be executed at run time (env state may mutate between commands).

use crate::ShellError;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WordPart {
    Literal(String),
    Var(String),
    /// Captured source for the inner `$(...)` body. Parsed lazily.
    CmdSub(String),
}

pub(crate) type Word = Vec<WordPart>;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Tok {
    Word(Word),
    /// `=`-style assignment prefix: `KEY=value`. The right side is a Word
    /// because it can contain expansions.
    Assign(String, Word),
    Semi,       // ; or newline
    AndIf,      // &&
    OrIf,       // ||
    Pipe,       // |
    Great,      // >
    DGreat,     // >>
    Less,       // <
    Great2,     // 2>
    DGreat2,    // 2>>
    Err2Out,    // 2>&1
}

pub(crate) fn tokenize(src: &str) -> Result<Vec<(Tok, usize)>, ShellError> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        // Whitespace (not newline).
        if c == b' ' || c == b'\t' {
            i += 1;
            continue;
        }
        // Comment to end of line.
        if c == b'#' && (i == 0 || is_blank(bytes[i - 1])) {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Line continuation handled inside word reader; bare backslash-newline
        // here just means an empty word boundary — skip.
        if c == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
            i += 2;
            continue;
        }
        if c == b'\n' || c == b';' {
            out.push((Tok::Semi, i));
            i += 1;
            continue;
        }
        if c == b'|' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                out.push((Tok::OrIf, i));
                i += 2;
            } else {
                out.push((Tok::Pipe, i));
                i += 1;
            }
            continue;
        }
        if c == b'&' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'&' {
                out.push((Tok::AndIf, i));
                i += 2;
                continue;
            }
            return Err(ShellError::UnsupportedSyntax(
                "background jobs (`&`) not supported".into(),
            ));
        }
        if c == b'>' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                out.push((Tok::DGreat, i));
                i += 2;
            } else {
                out.push((Tok::Great, i));
                i += 1;
            }
            continue;
        }
        if c == b'<' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'<' {
                return Err(ShellError::UnsupportedSyntax("heredocs not supported".into()));
            }
            if i + 1 < bytes.len() && bytes[i + 1] == b'(' {
                return Err(ShellError::UnsupportedSyntax(
                    "process substitution not supported".into(),
                ));
            }
            out.push((Tok::Less, i));
            i += 1;
            continue;
        }
        if c == b'`' {
            return Err(ShellError::UnsupportedSyntax(
                "backtick command substitution not supported; use $(...)".into(),
            ));
        }
        if c == b'(' {
            return Err(ShellError::UnsupportedSyntax("subshells `(...)` not supported".into()));
        }
        // 2> / 2>> / 2>&1 — only when "2" sits flush against the operator
        // (no whitespace). Otherwise it's a normal word.
        if c == b'2' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            if i + 2 < bytes.len() && bytes[i + 2] == b'&' {
                if i + 3 < bytes.len() && bytes[i + 3] == b'1' {
                    out.push((Tok::Err2Out, i));
                    i += 4;
                    continue;
                }
                return Err(ShellError::UnsupportedSyntax(
                    "only `2>&1` is supported among `n>&m` forms".into(),
                ));
            }
            if i + 2 < bytes.len() && bytes[i + 2] == b'>' {
                out.push((Tok::DGreat2, i));
                i += 3;
            } else {
                out.push((Tok::Great2, i));
                i += 2;
            }
            continue;
        }
        // Otherwise it's a word. Read until an unquoted delimiter.
        let (word, consumed) = read_word(&bytes[i..], i)?;
        let start = i;
        i += consumed;
        // Check for assignment: leading literal of the form NAME= followed by
        // more parts. The first part must be a Literal whose string contains
        // a `=` after a valid identifier prefix.
        if let Some(asn) = try_assign(&word) {
            out.push((Tok::Assign(asn.0, asn.1), start));
        } else {
            out.push((Tok::Word(word), start));
        }
    }
    Ok(out)
}

fn is_blank(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b';' || b == b'|' || b == b'&'
}

fn try_assign(word: &[WordPart]) -> Option<(String, Word)> {
    let first = word.first()?;
    let lit = match first {
        WordPart::Literal(s) => s,
        _ => return None,
    };
    let eq = lit.find('=')?;
    if eq == 0 {
        return None;
    }
    let name = &lit[..eq];
    if !is_ident(name) {
        return None;
    }
    let mut rest: Word = Vec::with_capacity(word.len());
    let tail = &lit[eq + 1..];
    if !tail.is_empty() {
        rest.push(WordPart::Literal(tail.to_string()));
    }
    rest.extend(word.iter().skip(1).cloned());
    Some((name.to_string(), rest))
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// Read a single word from `bytes`. Returns the word parts and how many
/// bytes were consumed. The slice starts at the word's first byte.
fn read_word(bytes: &[u8], abs_start: usize) -> Result<(Word, usize), ShellError> {
    let mut parts: Word = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\n' | b';' | b'|' | b'&' | b'<' | b'>' | b'`' | b'(' | b')' => break,
            b'\'' => {
                i += 1;
                let start = i;
                while i < bytes.len() && bytes[i] != b'\'' {
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err(ShellError::ParseError {
                        msg: "unterminated single quote".into(),
                        pos: abs_start + start - 1,
                    });
                }
                cur.push_str(std::str::from_utf8(&bytes[start..i]).map_err(|e| {
                    ShellError::ParseError { msg: format!("invalid utf8: {e}"), pos: abs_start + start }
                })?);
                i += 1;
            }
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    let b = bytes[i];
                    if b == b'\\' && i + 1 < bytes.len() {
                        let n = bytes[i + 1];
                        // Inside double quotes, backslash only escapes a few.
                        match n {
                            b'$' | b'`' | b'"' | b'\\' => {
                                cur.push(n as char);
                                i += 2;
                            }
                            b'\n' => {
                                i += 2;
                            }
                            _ => {
                                cur.push('\\');
                                i += 1;
                            }
                        }
                    } else if b == b'$' {
                        flush(&mut cur, &mut parts);
                        i = read_dollar(bytes, i, abs_start, &mut parts)?;
                    } else if b == b'`' {
                        return Err(ShellError::UnsupportedSyntax(
                            "backtick command substitution not supported".into(),
                        ));
                    } else {
                        cur.push(b as char);
                        i += 1;
                    }
                }
                if i >= bytes.len() {
                    return Err(ShellError::ParseError {
                        msg: "unterminated double quote".into(),
                        pos: abs_start + i,
                    });
                }
                i += 1; // closing "
            }
            b'\\' => {
                if i + 1 >= bytes.len() {
                    return Err(ShellError::ParseError {
                        msg: "trailing backslash".into(),
                        pos: abs_start + i,
                    });
                }
                let n = bytes[i + 1];
                if n == b'\n' {
                    i += 2;
                } else {
                    cur.push(n as char);
                    i += 2;
                }
            }
            b'$' => {
                flush(&mut cur, &mut parts);
                i = read_dollar(bytes, i, abs_start, &mut parts)?;
            }
            b'*' | b'?' | b'[' => {
                return Err(ShellError::UnsupportedSyntax(format!(
                    "glob character `{}` not supported",
                    c as char
                )));
            }
            _ => {
                cur.push(c as char);
                i += 1;
            }
        }
    }
    flush(&mut cur, &mut parts);
    Ok((parts, i))
}

fn flush(cur: &mut String, parts: &mut Word) {
    if !cur.is_empty() {
        parts.push(WordPart::Literal(std::mem::take(cur)));
    }
}

/// Called with `bytes[i] == b'$'`. Returns the new position.
fn read_dollar(
    bytes: &[u8],
    i: usize,
    abs_start: usize,
    parts: &mut Word,
) -> Result<usize, ShellError> {
    debug_assert_eq!(bytes[i], b'$');
    if i + 1 >= bytes.len() {
        parts.push(WordPart::Literal("$".into()));
        return Ok(i + 1);
    }
    let n = bytes[i + 1];
    if n == b'(' {
        // $(...) — capture matching close, track nested parens.
        let start = i + 2;
        let mut depth = 1usize;
        let mut j = start;
        while j < bytes.len() && depth > 0 {
            let b = bytes[j];
            if b == b'(' {
                depth += 1;
                j += 1;
            } else if b == b')' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                j += 1;
            } else if b == b'\'' {
                j += 1;
                while j < bytes.len() && bytes[j] != b'\'' {
                    j += 1;
                }
                if j < bytes.len() {
                    j += 1;
                }
            } else if b == b'"' {
                j += 1;
                while j < bytes.len() && bytes[j] != b'"' {
                    if bytes[j] == b'\\' && j + 1 < bytes.len() {
                        j += 2;
                    } else {
                        j += 1;
                    }
                }
                if j < bytes.len() {
                    j += 1;
                }
            } else if b == b'\\' && j + 1 < bytes.len() {
                j += 2;
            } else {
                j += 1;
            }
        }
        if depth != 0 {
            return Err(ShellError::ParseError {
                msg: "unterminated command substitution".into(),
                pos: abs_start + i,
            });
        }
        let body = std::str::from_utf8(&bytes[start..j]).map_err(|e| ShellError::ParseError {
            msg: format!("invalid utf8 in $(): {e}"),
            pos: abs_start + start,
        })?;
        parts.push(WordPart::CmdSub(body.to_string()));
        Ok(j + 1)
    } else if n == b'{' {
        let start = i + 2;
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'}' {
            // We don't allow `:-`, `:+`, etc. — bail if `}` doesn't follow a plain name.
            if !(bytes[j] == b'_' || bytes[j].is_ascii_alphanumeric()) {
                return Err(ShellError::UnsupportedSyntax(
                    "only `${NAME}` is supported (no parameter expansion modifiers)".into(),
                ));
            }
            j += 1;
        }
        if j >= bytes.len() {
            return Err(ShellError::ParseError {
                msg: "unterminated `${`".into(),
                pos: abs_start + i,
            });
        }
        let name = std::str::from_utf8(&bytes[start..j]).unwrap();
        if name.is_empty() {
            return Err(ShellError::ParseError {
                msg: "empty `${}`".into(),
                pos: abs_start + i,
            });
        }
        parts.push(WordPart::Var(name.to_string()));
        Ok(j + 1)
    } else if n == b'_' || n.is_ascii_alphabetic() {
        let start = i + 1;
        let mut j = start;
        while j < bytes.len() && (bytes[j] == b'_' || bytes[j].is_ascii_alphanumeric()) {
            j += 1;
        }
        let name = std::str::from_utf8(&bytes[start..j]).unwrap();
        parts.push(WordPart::Var(name.to_string()));
        Ok(j)
    } else if matches!(n, b'$' | b'?' | b'!' | b'#' | b'@' | b'*') {
        // `$?` is reasonable to add later; for now report clearly.
        Err(ShellError::UnsupportedSyntax(format!(
            "special parameter `${}` not supported",
            n as char
        )))
    } else {
        // Literal `$` followed by something we don't recognise.
        parts.push(WordPart::Literal("$".into()));
        Ok(i + 1)
    }
}
