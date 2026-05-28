//! `{{KEY}}` prompt-template substitution and Markdown skill/role loader.
//!
//! Two jobs:
//!
//! 1. `substitute(template, args)` — replace `{{KEY}}` placeholders with
//!    values. KEYs are `[A-Z][A-Z0-9_]*`. A placeholder with no matching
//!    arg is an error; an arg with no matching placeholder is a warning
//!    (use `substitute_warn_unused` to surface them).
//!
//! 2. `load_skill` / `load_role` — read a Markdown file under
//!    `<root>/.agents/{skills,roles}/<name>.md`, parse a tiny
//!    YAML-frontmatter subset (`name`, `description`, `args: - x`), and
//!    return the body. Skills can then be `.render(&args)`'d through
//!    `substitute`.
//!
//! The placeholder scan is `wire::scan_for_pair(.., b'{', b'{')`, so on
//! aarch64 (NEON) and x86_64-with-AVX2 the hot loop is vectorised. The
//! frontmatter parser is line-based and deliberately dumb — no quoted
//! strings, no nested objects, no escapes. Anyone who wants real YAML can
//! omit the frontmatter and put their metadata in the body.
//!
//! No external deps: `std::fs` + `wire`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use wire::scan_for_pair;

#[derive(Debug)]
pub enum TmplError {
    /// Template referenced `{{KEY}}` but no value supplied.
    MissingArg(String),
    /// Caller passed an arg the template did not use. Only surfaced via
    /// `substitute_warn_unused`; never returned from `substitute`.
    UnknownArg(String),
    Io(std::io::Error),
    /// Bad placeholder syntax in a template (unterminated, bad char,
    /// nested braces) or bad frontmatter in a skill/role file. Carries a
    /// human-readable reason.
    BadFrontmatter(String),
}

impl std::fmt::Display for TmplError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TmplError::MissingArg(k) => write!(f, "missing template arg: {{{{{k}}}}}"),
            TmplError::UnknownArg(k) => write!(f, "unused template arg: {k}"),
            TmplError::Io(e) => write!(f, "io: {e}"),
            TmplError::BadFrontmatter(s) => write!(f, "bad template/frontmatter: {s}"),
        }
    }
}

impl std::error::Error for TmplError {}

impl From<std::io::Error> for TmplError {
    fn from(e: std::io::Error) -> Self {
        TmplError::Io(e)
    }
}

/// Substitute `{{KEY}}` placeholders in `template` using `args`.
///
/// Missing arg → `MissingArg`. Unused arg → silently ignored (use
/// `substitute_warn_unused` to see them). Invalid KEY char or
/// unterminated `{{` → `BadFrontmatter` (the catch-all "this template
/// is malformed" variant).
pub fn substitute(template: &str, args: &HashMap<&str, String>) -> Result<String, TmplError> {
    let (out, _used) = substitute_impl(template, args)?;
    Ok(out)
}

/// Like `substitute`, but also returns the list of args that were
/// supplied but never referenced. Callers (e.g. an agent harness) can
/// log these as warnings.
pub fn substitute_warn_unused(
    template: &str,
    args: &HashMap<&str, String>,
) -> Result<(String, Vec<String>), TmplError> {
    let (out, used) = substitute_impl(template, args)?;
    let unused: Vec<String> =
        args.keys().filter(|k| !used.contains(&k.to_string())).map(|k| k.to_string()).collect();
    Ok((out, unused))
}

fn substitute_impl(
    template: &str,
    args: &HashMap<&str, String>,
) -> Result<(String, Vec<String>), TmplError> {
    let bytes = template.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut used: Vec<String> = Vec::new();
    let mut cursor: usize = 0;

    while cursor < bytes.len() {
        // Locate next `{{` from cursor.
        let rel = match scan_for_pair(&bytes[cursor..], b'{', b'{') {
            Some(r) => r,
            None => {
                out.push_str(&template[cursor..]);
                break;
            }
        };
        let open = cursor + rel;
        // Copy bytes up to `{{` verbatim.
        out.push_str(&template[cursor..open]);

        // Find closing `}}` after the opening pair.
        let key_start = open + 2;
        if key_start > bytes.len() {
            return Err(TmplError::BadFrontmatter("unterminated '{{' at end of template".into()));
        }
        let close_rel = scan_for_pair(&bytes[key_start..], b'}', b'}').ok_or_else(|| {
            TmplError::BadFrontmatter(format!(
                "unterminated '{{{{' starting at byte {open}: no matching '}}}}'"
            ))
        })?;
        let close = key_start + close_rel;
        let key = &template[key_start..close];
        validate_key(key, open)?;

        match args.get(key) {
            Some(v) => {
                out.push_str(v);
                if !used.iter().any(|u| u == key) {
                    used.push(key.to_string());
                }
            }
            None => return Err(TmplError::MissingArg(key.to_string())),
        }
        cursor = close + 2;
    }
    Ok((out, used))
}

fn validate_key(key: &str, at: usize) -> Result<(), TmplError> {
    if key.is_empty() {
        return Err(TmplError::BadFrontmatter(format!("empty key in '{{{{}}}}' at byte {at}")));
    }
    let mut chars = key.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_uppercase() {
        return Err(TmplError::BadFrontmatter(format!(
            "invalid template key {key:?} at byte {at}: must start with [A-Z]"
        )));
    }
    for c in chars {
        if !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_') {
            return Err(TmplError::BadFrontmatter(format!(
                "invalid template key {key:?} at byte {at}: must match [A-Z][A-Z0-9_]*"
            )));
        }
    }
    Ok(())
}

/// A loaded Markdown skill: post-frontmatter body plus parsed metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub body: String,
    pub description: Option<String>,
    pub args: Vec<String>,
}

impl Skill {
    /// Run `substitute` over `self.body`. Pure convenience.
    pub fn render(&self, args: &HashMap<&str, String>) -> Result<String, TmplError> {
        substitute(&self.body, args)
    }
}

/// A loaded Markdown role. Roles only carry a name + the body as a
/// system prompt; no args, no description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub system_prompt: String,
}

/// Load `<root>/.agents/skills/<name>.md`.
pub fn load_skill(root: &Path, name: &str) -> Result<Skill, TmplError> {
    let path = skill_path(root, name);
    let raw = fs::read_to_string(&path)?;
    let (fm, body) = split_frontmatter(&raw)?;
    let mut skill = Skill {
        name: name.to_string(),
        body: body.to_string(),
        description: None,
        args: Vec::new(),
    };
    if let Some(fm) = fm {
        let parsed = parse_frontmatter(fm)?;
        if let Some(n) = parsed.name {
            skill.name = n;
        }
        skill.description = parsed.description;
        skill.args = parsed.args;
    }
    Ok(skill)
}

/// Load `<root>/.agents/roles/<name>.md`.
pub fn load_role(root: &Path, name: &str) -> Result<Role, TmplError> {
    let path = role_path(root, name);
    let raw = fs::read_to_string(&path)?;
    let (fm, body) = split_frontmatter(&raw)?;
    let mut role = Role { name: name.to_string(), system_prompt: body.to_string() };
    if let Some(fm) = fm {
        let parsed = parse_frontmatter(fm)?;
        if let Some(n) = parsed.name {
            role.name = n;
        }
        // description / args are ignored for roles by design.
    }
    Ok(role)
}

fn skill_path(root: &Path, name: &str) -> PathBuf {
    root.join(".agents").join("skills").join(format!("{name}.md"))
}

fn role_path(root: &Path, name: &str) -> PathBuf {
    root.join(".agents").join("roles").join(format!("{name}.md"))
}

/// Split a file into (Some(frontmatter_text), body) if it starts with
/// `---\n`, else (None, whole_file).
fn split_frontmatter(raw: &str) -> Result<(Option<&str>, &str), TmplError> {
    // Frontmatter must be the very first line.
    let after_open = match raw.strip_prefix("---\n") {
        Some(rest) => rest,
        None => {
            // Allow CRLF too; otherwise no frontmatter.
            match raw.strip_prefix("---\r\n") {
                Some(rest) => rest,
                None => return Ok((None, raw)),
            }
        }
    };

    // Find a line that is exactly `---`. We scan line by line so the
    // closer must be on its own line.
    let mut offset = 0usize;
    for line in after_open.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            let fm = &after_open[..offset];
            let body_start = offset + line.len();
            let body = &after_open[body_start..];
            // Strip a single leading blank line for cleanliness.
            let body = body.strip_prefix('\n').unwrap_or(body);
            return Ok((Some(fm), body));
        }
        offset += line.len();
    }
    Err(TmplError::BadFrontmatter(
        "frontmatter opened with '---' but no closing '---' found".into(),
    ))
}

#[derive(Default)]
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    args: Vec<String>,
}

/// Parse the tiny subset:
///   name: <string>
///   description: <string>
///   args:
///     - <item>
///     - <item>
/// Blank lines and `# comments` are ignored. Indentation under `args:`
/// must be at least one space before the `-`. Anything else is an error.
fn parse_frontmatter(src: &str) -> Result<Frontmatter, TmplError> {
    let mut out = Frontmatter::default();
    let mut in_args = false;
    for (lineno, raw_line) in src.lines().enumerate() {
        let line = raw_line.trim_end();
        if line.trim().is_empty() {
            in_args = false;
            continue;
        }
        if line.trim_start().starts_with('#') {
            continue;
        }

        // List item under `args:`.
        if in_args {
            let stripped = line.trim_start();
            if let Some(rest) = stripped.strip_prefix("- ") {
                out.args.push(rest.trim().to_string());
                continue;
            }
            if stripped == "-" {
                return Err(TmplError::BadFrontmatter(format!(
                    "line {}: empty list item under 'args:'",
                    lineno + 1
                )));
            }
            // Fall through: a non-list line ends the args block.
            in_args = false;
        }

        // Top-level `key: value` or `key:` (for args).
        if line.starts_with(' ') || line.starts_with('\t') {
            return Err(TmplError::BadFrontmatter(format!(
                "line {}: unexpected indentation",
                lineno + 1
            )));
        }
        let colon = line.find(':').ok_or_else(|| {
            TmplError::BadFrontmatter(format!(
                "line {}: expected 'key: value' or 'key:'",
                lineno + 1
            ))
        })?;
        let key = line[..colon].trim();
        let value = line[colon + 1..].trim();
        match key {
            "name" => {
                if value.is_empty() {
                    return Err(TmplError::BadFrontmatter(format!(
                        "line {}: 'name' requires a value",
                        lineno + 1
                    )));
                }
                out.name = Some(value.to_string());
            }
            "description" => {
                if value.is_empty() {
                    return Err(TmplError::BadFrontmatter(format!(
                        "line {}: 'description' requires a value",
                        lineno + 1
                    )));
                }
                out.description = Some(value.to_string());
            }
            "args" => {
                if !value.is_empty() {
                    return Err(TmplError::BadFrontmatter(format!(
                        "line {}: 'args:' must be followed by a list, not a value",
                        lineno + 1
                    )));
                }
                in_args = true;
            }
            other => {
                return Err(TmplError::BadFrontmatter(format!(
                    "line {}: unknown frontmatter key {:?}",
                    lineno + 1,
                    other
                )));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn args(pairs: &[(&'static str, &str)]) -> HashMap<&'static str, String> {
        pairs.iter().map(|(k, v)| (*k, v.to_string())).collect()
    }

    #[test]
    fn substitute_zero_args() {
        let out = substitute("hello world", &HashMap::new()).unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn substitute_one_arg_middle() {
        let out = substitute("a {{X}} b", &args(&[("X", "Y")])).unwrap();
        assert_eq!(out, "a Y b");
    }

    #[test]
    fn substitute_key_at_start() {
        let out = substitute("{{X}}!", &args(&[("X", "hi")])).unwrap();
        assert_eq!(out, "hi!");
    }

    #[test]
    fn substitute_key_at_end() {
        let out = substitute("end:{{X}}", &args(&[("X", "Z")])).unwrap();
        assert_eq!(out, "end:Z");
    }

    #[test]
    fn substitute_many_args() {
        let out =
            substitute("{{A}}-{{B}}-{{C}}", &args(&[("A", "1"), ("B", "2"), ("C", "3")])).unwrap();
        assert_eq!(out, "1-2-3");
    }

    #[test]
    fn substitute_repeated_key() {
        let out = substitute("{{X}} and {{X}}", &args(&[("X", "ok")])).unwrap();
        assert_eq!(out, "ok and ok");
    }

    #[test]
    fn substitute_underscore_and_digit_keys() {
        let out = substitute(
            "{{ISSUE_NUMBER}}/{{SEV2}}",
            &args(&[("ISSUE_NUMBER", "42"), ("SEV2", "hi")]),
        )
        .unwrap();
        assert_eq!(out, "42/hi");
    }

    #[test]
    fn substitute_missing_arg_errors_with_key() {
        let err = substitute("a {{NOPE}} b", &HashMap::new()).unwrap_err();
        match err {
            TmplError::MissingArg(k) => assert_eq!(k, "NOPE"),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn substitute_lowercase_key_is_syntax_error() {
        let err = substitute("{{lower}}", &HashMap::new()).unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn substitute_leading_digit_key_is_syntax_error() {
        let err = substitute("{{1X}}", &HashMap::new()).unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn substitute_punctuation_in_key_is_syntax_error() {
        let err = substitute("{{X-Y}}", &HashMap::new()).unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn substitute_empty_key_is_syntax_error() {
        let err = substitute("{{}}", &HashMap::new()).unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn substitute_single_brace_is_literal() {
        let out = substitute("a { b } c", &HashMap::new()).unwrap();
        assert_eq!(out, "a { b } c");
    }

    #[test]
    fn substitute_unterminated_open_errors() {
        let err = substitute("hello {{X", &args(&[("X", "v")])).unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn substitute_nested_braces_refused() {
        // `{{{{X}}}}` opens at byte 0 with `{{`, sees key "{{X" (bad
        // char `{`), and must error — not silently treat as an escape.
        let err = substitute("{{{{X}}}}", &args(&[("X", "v")])).unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn substitute_warn_unused_returns_unused_keys() {
        let (out, unused) =
            substitute_warn_unused("use {{A}}", &args(&[("A", "1"), ("B", "2"), ("C", "3")]))
                .unwrap();
        assert_eq!(out, "use 1");
        let mut unused = unused;
        unused.sort();
        assert_eq!(unused, vec!["B".to_string(), "C".to_string()]);
    }

    #[test]
    fn substitute_warn_unused_empty_when_all_used() {
        let (_out, unused) =
            substitute_warn_unused("{{A}}{{B}}", &args(&[("A", "x"), ("B", "y")])).unwrap();
        assert!(unused.is_empty());
    }

    // --- file loading ---

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_root(label: &str) -> PathBuf {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("tmpl-test-{label}-{t}-{n}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_skill(root: &Path, name: &str, contents: &str) {
        let dir = root.join(".agents").join("skills");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{name}.md")), contents).unwrap();
    }

    fn write_role(root: &Path, name: &str, contents: &str) {
        let dir = root.join(".agents").join("roles");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{name}.md")), contents).unwrap();
    }

    #[test]
    fn load_skill_with_frontmatter_and_render() {
        let root = unique_root("skill-fm");
        write_skill(
            &root,
            "triage",
            "---\nname: triage\ndescription: Triage incoming issues\nargs:\n  - issue_number\n  - severity\n---\n\nIssue {{ISSUE_NUMBER}} (sev {{SEVERITY}}).\n",
        );
        let s = load_skill(&root, "triage").unwrap();
        assert_eq!(s.name, "triage");
        assert_eq!(s.description.as_deref(), Some("Triage incoming issues"));
        assert_eq!(s.args, vec!["issue_number".to_string(), "severity".to_string()]);
        let rendered = s.render(&args(&[("ISSUE_NUMBER", "42"), ("SEVERITY", "high")])).unwrap();
        assert_eq!(rendered, "Issue 42 (sev high).\n");
    }

    #[test]
    fn load_skill_without_frontmatter_body_is_whole_file() {
        let root = unique_root("skill-no-fm");
        let body = "plain markdown with {{X}}.\nNo frontmatter.\n";
        write_skill(&root, "plain", body);
        let s = load_skill(&root, "plain").unwrap();
        assert_eq!(s.name, "plain"); // defaults to filename stem
        assert_eq!(s.description, None);
        assert!(s.args.is_empty());
        assert_eq!(s.body, body);
    }

    #[test]
    fn load_skill_missing_file_returns_io_error() {
        let root = unique_root("skill-missing");
        let err = load_skill(&root, "nope").unwrap_err();
        assert!(matches!(err, TmplError::Io(_)), "got {err:?}");
    }

    #[test]
    fn load_skill_unclosed_frontmatter_errors() {
        let root = unique_root("skill-unclosed");
        write_skill(&root, "broken", "---\nname: x\nno closer here\n");
        let err = load_skill(&root, "broken").unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)), "got {err:?}");
    }

    #[test]
    fn load_skill_unknown_frontmatter_key_errors() {
        let root = unique_root("skill-unknown-key");
        write_skill(&root, "weird", "---\nfoo: bar\n---\nbody\n");
        let err = load_skill(&root, "weird").unwrap_err();
        match err {
            TmplError::BadFrontmatter(msg) => assert!(msg.contains("foo"), "msg={msg}"),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn load_skill_args_value_form_errors() {
        let root = unique_root("skill-args-value");
        // `args: foo` (value on same line) is illegal — must be a list.
        write_skill(&root, "bad", "---\nargs: foo\n---\nbody\n");
        let err = load_skill(&root, "bad").unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn load_role_basic() {
        let root = unique_root("role-basic");
        write_role(&root, "reviewer", "---\nname: code-reviewer\n---\nYou are a code reviewer.\n");
        let r = load_role(&root, "reviewer").unwrap();
        assert_eq!(r.name, "code-reviewer");
        assert_eq!(r.system_prompt, "You are a code reviewer.\n");
    }

    #[test]
    fn load_role_without_frontmatter_uses_filename() {
        let root = unique_root("role-no-fm");
        write_role(&root, "helper", "Be helpful.\n");
        let r = load_role(&root, "helper").unwrap();
        assert_eq!(r.name, "helper");
        assert_eq!(r.system_prompt, "Be helpful.\n");
    }

    #[test]
    fn load_role_malformed_frontmatter_errors() {
        let root = unique_root("role-bad");
        // No colon on a line.
        write_role(&root, "broken", "---\njust some text\n---\nbody\n");
        let err = load_role(&root, "broken").unwrap_err();
        assert!(matches!(err, TmplError::BadFrontmatter(_)));
    }

    #[test]
    fn empty_args_list_under_args_key_is_ok() {
        // `args:` followed immediately by `---` means "no args".
        let root = unique_root("skill-empty-args");
        write_skill(&root, "noargs", "---\nname: x\nargs:\n---\nbody {{X}}\n");
        let s = load_skill(&root, "noargs").unwrap();
        assert!(s.args.is_empty());
        assert_eq!(s.body, "body {{X}}\n");
    }
}
