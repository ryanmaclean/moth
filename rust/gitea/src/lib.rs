//! Minimal Gitea REST API client.
//!
//! Same shape as the `anthropic` / `mcp` HTTP clients: one libcurl easy handle
//! per request, synchronous, no async runtime. JSON parsing reuses
//! `anthropic::json`.
//!
//! Targets Gitea-compatible forges (Gitea, Forgejo, Codeberg). The wire
//! difference from GitHub: Gitea sends `Authorization: token <PAT>` rather
//! than `Bearer`, and the `/api/v1/...` base path replaces GitHub's bare
//! domain. Otherwise the JSON shapes for repos/issues/pulls/comments are
//! close enough to GitHub that the same field names land.

mod http;

use anthropic::json::{self, Json, escape_into};

// ---- public types ----------------------------------------------------------

pub struct Client {
    base_url: String,
    token: String,
}

impl Client {
    pub fn new(base_url: impl Into<String>, token: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), token: token.into() }
    }

    // -- repos --

    pub fn get_repo(&self, owner: &str, repo: &str) -> Result<Repo, Error> {
        let url = self.url(&format!("/repos/{owner}/{repo}"));
        let body = self.get(&url)?;
        parse_repo(&body)
    }

    // -- issues --

    pub fn list_issues(
        &self,
        owner: &str,
        repo: &str,
        state: IssueState,
    ) -> Result<Vec<Issue>, Error> {
        // Gitea returns issues *and* pulls on /issues; filter type=issues so
        // callers don't get a surprise PR mixed in. `state=all` is the spec'd
        // value for both.
        let url = self.url(&format!(
            "/repos/{owner}/{repo}/issues?type=issues&state={}",
            state.as_query()
        ));
        let body = self.get(&url)?;
        parse_array(&body, parse_issue_value)
    }

    pub fn get_issue(&self, owner: &str, repo: &str, number: u64) -> Result<Issue, Error> {
        let url = self.url(&format!("/repos/{owner}/{repo}/issues/{number}"));
        let body = self.get(&url)?;
        let v = json::parse(&body).map_err(map_anthropic_err)?;
        parse_issue_value(&v)
    }

    pub fn create_issue(
        &self,
        owner: &str,
        repo: &str,
        req: NewIssue,
    ) -> Result<Issue, Error> {
        // Note: Gitea's `labels` field on `POST /issues` wants integer label
        // ids, not names. The README contract specifies names, so callers
        // must resolve those out of band. We pass through as strings; the
        // server will 422 on label names. For v1 this is documented and
        // tested at the serialization level only.
        let url = self.url(&format!("/repos/{owner}/{repo}/issues"));
        let body = serialize_new_issue(&req);
        let resp = self.post(&url, body.as_bytes())?;
        let v = json::parse(&resp).map_err(map_anthropic_err)?;
        parse_issue_value(&v)
    }

    pub fn add_issue_comment(
        &self,
        owner: &str,
        repo: &str,
        number: u64,
        body: &str,
    ) -> Result<Comment, Error> {
        let url = self.url(&format!("/repos/{owner}/{repo}/issues/{number}/comments"));
        let payload = serialize_comment(body);
        let resp = self.post(&url, payload.as_bytes())?;
        let v = json::parse(&resp).map_err(map_anthropic_err)?;
        parse_comment_value(&v)
    }

    pub fn list_issue_comments(
        &self,
        owner: &str,
        repo: &str,
        number: u64,
    ) -> Result<Vec<Comment>, Error> {
        let url = self.url(&format!("/repos/{owner}/{repo}/issues/{number}/comments"));
        let body = self.get(&url)?;
        parse_array(&body, parse_comment_value)
    }

    // -- pulls --

    pub fn list_pulls(
        &self,
        owner: &str,
        repo: &str,
        state: IssueState,
    ) -> Result<Vec<PullRequest>, Error> {
        let url = self.url(&format!(
            "/repos/{owner}/{repo}/pulls?state={}",
            state.as_query()
        ));
        let body = self.get(&url)?;
        parse_array(&body, parse_pull_value)
    }

    pub fn get_pull(
        &self,
        owner: &str,
        repo: &str,
        number: u64,
    ) -> Result<PullRequest, Error> {
        let url = self.url(&format!("/repos/{owner}/{repo}/pulls/{number}"));
        let body = self.get(&url)?;
        let v = json::parse(&body).map_err(map_anthropic_err)?;
        parse_pull_value(&v)
    }

    pub fn create_pull(
        &self,
        owner: &str,
        repo: &str,
        req: NewPull,
    ) -> Result<PullRequest, Error> {
        let url = self.url(&format!("/repos/{owner}/{repo}/pulls"));
        let body = serialize_new_pull(&req);
        let resp = self.post(&url, body.as_bytes())?;
        let v = json::parse(&resp).map_err(map_anthropic_err)?;
        parse_pull_value(&v)
    }

    // -- internals --

    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{base}/api/v1{path}")
    }

    fn headers(&self, with_content_type: bool) -> Vec<String> {
        let mut h = vec![
            format!("Authorization: token {}", self.token),
            "Accept: application/json".to_string(),
        ];
        if with_content_type {
            h.push("Content-Type: application/json".to_string());
        }
        h
    }

    fn get(&self, url: &str) -> Result<Vec<u8>, Error> {
        let resp = http::request(url, "GET", &self.headers(false), b"")?;
        check_status(resp.status, resp.body)
    }

    fn post(&self, url: &str, body: &[u8]) -> Result<Vec<u8>, Error> {
        let resp = http::request(url, "POST", &self.headers(true), body)?;
        check_status(resp.status, resp.body)
    }
}

#[derive(Debug, Clone)]
pub struct Repo {
    pub full_name: String,
    pub default_branch: String,
    pub fork: bool,
}

#[derive(Debug, Clone)]
pub struct Issue {
    pub number: u64,
    pub title: String,
    pub body: String,
    pub state: IssueState,
    pub user: User,
    pub labels: Vec<Label>,
}

#[derive(Debug, Clone)]
pub struct PullRequest {
    pub number: u64,
    pub title: String,
    pub body: String,
    pub state: IssueState,
    pub head: BranchRef,
    pub base: BranchRef,
    pub merged: bool,
}

#[derive(Debug, Clone)]
pub struct Comment {
    pub id: u64,
    pub body: String,
    pub user: User,
}

#[derive(Debug, Clone)]
pub struct User {
    pub login: String,
    pub id: u64,
}

#[derive(Debug, Clone)]
pub struct Label {
    pub name: String,
    pub color: String,
}

#[derive(Debug, Clone)]
pub struct BranchRef {
    pub label: String,
    pub ref_: String,
    pub sha: String,
}

#[derive(Debug, Clone)]
pub struct NewIssue {
    pub title: String,
    pub body: Option<String>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct NewPull {
    pub title: String,
    pub body: Option<String>,
    pub head: String,
    pub base: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueState {
    Open,
    Closed,
    All,
}

impl IssueState {
    fn as_query(self) -> &'static str {
        match self {
            IssueState::Open => "open",
            IssueState::Closed => "closed",
            IssueState::All => "all",
        }
    }

    fn from_str(s: &str) -> IssueState {
        match s {
            "closed" => IssueState::Closed,
            // Gitea/GitHub only return open|closed on individual records;
            // "all" is purely a query filter. Default unknown to Open so a
            // future state ("draft" etc.) doesn't crash the parser.
            _ => IssueState::Open,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Http(String),
    Status { code: u16, body: String },
    InvalidResponse(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Http(m) => write!(f, "http: {m}"),
            Error::Status { code, body } => write!(f, "HTTP {code}: {body}"),
            Error::InvalidResponse(m) => write!(f, "invalid response: {m}"),
        }
    }
}

impl std::error::Error for Error {}

// ---- response helpers ------------------------------------------------------

fn check_status(code: u16, body: Vec<u8>) -> Result<Vec<u8>, Error> {
    if (200..300).contains(&code) {
        Ok(body)
    } else {
        Err(Error::Status {
            code,
            body: String::from_utf8_lossy(&body).into_owned(),
        })
    }
}

fn map_anthropic_err(e: anthropic::Error) -> Error {
    // anthropic::json::parse only ever returns InvalidResponse; map it across.
    Error::InvalidResponse(e.to_string())
}

fn parse_array<T>(
    body: &[u8],
    parse_one: impl Fn(&Json) -> Result<T, Error>,
) -> Result<Vec<T>, Error> {
    let v = json::parse(body).map_err(map_anthropic_err)?;
    let Json::Arr(items) = v else {
        return Err(Error::InvalidResponse("expected JSON array".into()));
    };
    items.iter().map(parse_one).collect()
}

fn field_str(v: &Json, key: &str) -> Result<String, Error> {
    match v.get(key) {
        Some(Json::Str(s)) => Ok(s.clone()),
        // Missing or null collapses to empty string. Gitea sometimes omits
        // optional body/description fields entirely; treating those as ""
        // keeps the struct field non-Option and avoids per-callsite checks.
        Some(Json::Null) | None => Ok(String::new()),
        Some(other) => Err(Error::InvalidResponse(format!(
            "field {key} is not a string: {other:?}"
        ))),
    }
}

fn field_u64(v: &Json, key: &str) -> Result<u64, Error> {
    match v.get(key) {
        Some(Json::Num(n)) => n
            .parse::<u64>()
            .map_err(|_| Error::InvalidResponse(format!("field {key} is not u64: {n}"))),
        other => Err(Error::InvalidResponse(format!(
            "field {key} missing or not numeric: {other:?}"
        ))),
    }
}

fn field_bool(v: &Json, key: &str) -> Result<bool, Error> {
    match v.get(key) {
        Some(Json::Bool(b)) => Ok(*b),
        // Missing fork/merged flags default false rather than erroring —
        // some Gitea versions omit them on resources where they're trivially
        // false (e.g. a closed-but-unmerged PR doesn't always carry merged).
        None | Some(Json::Null) => Ok(false),
        Some(other) => Err(Error::InvalidResponse(format!(
            "field {key} is not bool: {other:?}"
        ))),
    }
}

fn field_obj<'a>(v: &'a Json, key: &str) -> Result<&'a Json, Error> {
    v.get(key)
        .ok_or_else(|| Error::InvalidResponse(format!("field {key} missing")))
}

// ---- per-type parsers ------------------------------------------------------

fn parse_repo(body: &[u8]) -> Result<Repo, Error> {
    let v = json::parse(body).map_err(map_anthropic_err)?;
    Ok(Repo {
        full_name: field_str(&v, "full_name")?,
        default_branch: field_str(&v, "default_branch")?,
        fork: field_bool(&v, "fork")?,
    })
}

fn parse_issue_value(v: &Json) -> Result<Issue, Error> {
    let state_str = field_str(v, "state")?;
    let labels = match v.get("labels") {
        Some(Json::Arr(items)) => items.iter().map(parse_label_value).collect::<Result<_, _>>()?,
        // labels: null shows up on issues with none, despite the docs.
        None | Some(Json::Null) => Vec::new(),
        Some(other) => {
            return Err(Error::InvalidResponse(format!(
                "labels is not array: {other:?}"
            )));
        }
    };
    Ok(Issue {
        number: field_u64(v, "number")?,
        title: field_str(v, "title")?,
        body: field_str(v, "body")?,
        state: IssueState::from_str(&state_str),
        user: parse_user_value(field_obj(v, "user")?)?,
        labels,
    })
}

fn parse_pull_value(v: &Json) -> Result<PullRequest, Error> {
    let state_str = field_str(v, "state")?;
    Ok(PullRequest {
        number: field_u64(v, "number")?,
        title: field_str(v, "title")?,
        body: field_str(v, "body")?,
        state: IssueState::from_str(&state_str),
        head: parse_branch_ref(field_obj(v, "head")?)?,
        base: parse_branch_ref(field_obj(v, "base")?)?,
        merged: field_bool(v, "merged")?,
    })
}

fn parse_comment_value(v: &Json) -> Result<Comment, Error> {
    Ok(Comment {
        id: field_u64(v, "id")?,
        body: field_str(v, "body")?,
        user: parse_user_value(field_obj(v, "user")?)?,
    })
}

fn parse_user_value(v: &Json) -> Result<User, Error> {
    Ok(User {
        login: field_str(v, "login")?,
        id: field_u64(v, "id")?,
    })
}

fn parse_label_value(v: &Json) -> Result<Label, Error> {
    Ok(Label {
        name: field_str(v, "name")?,
        color: field_str(v, "color")?,
    })
}

fn parse_branch_ref(v: &Json) -> Result<BranchRef, Error> {
    Ok(BranchRef {
        label: field_str(v, "label")?,
        // `ref` is a reserved word in Rust; the JSON key is bare `ref`.
        ref_: field_str(v, "ref")?,
        sha: field_str(v, "sha")?,
    })
}

// ---- serializers -----------------------------------------------------------

fn serialize_new_issue(req: &NewIssue) -> String {
    let mut s = String::with_capacity(64);
    s.push_str("{\"title\":\"");
    escape_into(&mut s, &req.title);
    s.push('"');
    if let Some(body) = &req.body {
        s.push_str(",\"body\":\"");
        escape_into(&mut s, body);
        s.push('"');
    }
    if !req.labels.is_empty() {
        // Gitea wants integer label ids here, but we accept names from the
        // public type and let the server reject if names aren't ints. The
        // alternative is forcing a /labels lookup in the client, which is
        // outside this crate's scope.
        s.push_str(",\"labels\":[");
        for (i, lbl) in req.labels.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('"');
            escape_into(&mut s, lbl);
            s.push('"');
        }
        s.push(']');
    }
    s.push('}');
    s
}

fn serialize_new_pull(req: &NewPull) -> String {
    let mut s = String::with_capacity(64);
    s.push_str("{\"title\":\"");
    escape_into(&mut s, &req.title);
    s.push_str("\",\"head\":\"");
    escape_into(&mut s, &req.head);
    s.push_str("\",\"base\":\"");
    escape_into(&mut s, &req.base);
    s.push('"');
    if let Some(body) = &req.body {
        s.push_str(",\"body\":\"");
        escape_into(&mut s, body);
        s.push('"');
    }
    s.push('}');
    s
}

fn serialize_comment(body: &str) -> String {
    let mut s = String::with_capacity(body.len() + 16);
    s.push_str("{\"body\":\"");
    escape_into(&mut s, body);
    s.push_str("\"}");
    s
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests run against an in-process `TcpListener` HTTP/1.1 mock. Each
    //! test scripts one or more canned responses (status + body) and
    //! captures the inbound request for assertion (auth header, URL,
    //! method, body). The mock closes the connection after each scripted
    //! reply, matching `connection: close`.

    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    #[derive(Clone)]
    struct Reply {
        status: u16,
        body: Vec<u8>,
    }

    fn ok(body: impl Into<Vec<u8>>) -> Reply {
        Reply { status: 200, body: body.into() }
    }

    fn status(code: u16, body: impl Into<Vec<u8>>) -> Reply {
        Reply { status: code, body: body.into() }
    }

    #[derive(Debug, Default, Clone)]
    struct Captured {
        method: String,
        path: String,
        authorization: Option<String>,
        content_type: Option<String>,
        body: Vec<u8>,
    }

    struct Mock {
        base_url: String,
        captured: Arc<Mutex<Vec<Captured>>>,
        _handle: thread::JoinHandle<()>,
    }

    fn start_mock(replies: Vec<Reply>) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::<Captured>::new()));
        let cap_clone = Arc::clone(&captured);
        let total = replies.len();
        let idx = Arc::new(AtomicUsize::new(0));
        let replies = Arc::new(replies);

        let handle = thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let i = idx.fetch_add(1, Ordering::SeqCst);
                if i >= total {
                    break;
                }
                let reply = replies[i].clone();
                if let Some(c) = handle_one(stream, &reply) {
                    cap_clone.lock().unwrap().push(c);
                }
                if i + 1 >= total {
                    break;
                }
            }
        });

        Mock {
            base_url: format!("http://{addr}"),
            captured,
            _handle: handle,
        }
    }

    fn handle_one(mut stream: TcpStream, reply: &Reply) -> Option<Captured> {
        let mut reader = BufReader::new(stream.try_clone().ok()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line).ok()?;
        let request_line = request_line.trim_end_matches(['\r', '\n']).to_string();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();

        let mut captured = Captured {
            method,
            path,
            ..Captured::default()
        };
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).ok()?;
            if line == "\r\n" || line == "\n" || line.is_empty() {
                break;
            }
            let line = line.trim_end_matches(['\r', '\n']);
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            let lc = name.to_ascii_lowercase();
            match lc.as_str() {
                "content-length" => content_length = value.parse().unwrap_or(0),
                "authorization" => captured.authorization = Some(value.to_string()),
                "content-type" => captured.content_type = Some(value.to_string()),
                _ => {}
            }
        }

        if content_length > 0 {
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body).ok()?;
            captured.body = body;
        }

        let body = &reply.body;
        let mut resp = format!("HTTP/1.1 {} {}\r\n", reply.status, status_text(reply.status));
        resp.push_str("content-type: application/json\r\n");
        resp.push_str(&format!("content-length: {}\r\n", body.len()));
        resp.push_str("connection: close\r\n\r\n");
        stream.write_all(resp.as_bytes()).ok()?;
        stream.write_all(body).ok()?;
        stream.flush().ok()?;
        Some(captured)
    }

    fn status_text(code: u16) -> &'static str {
        match code {
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            422 => "Unprocessable Entity",
            500 => "Internal Server Error",
            _ => "Status",
        }
    }

    // -- serializer unit tests --

    #[test]
    fn serialize_new_issue_minimal() {
        let s = serialize_new_issue(&NewIssue {
            title: "hello".into(),
            body: None,
            labels: vec![],
        });
        assert_eq!(s, r#"{"title":"hello"}"#);
    }

    #[test]
    fn serialize_new_issue_with_body_and_labels() {
        let s = serialize_new_issue(&NewIssue {
            title: "t".into(),
            body: Some("b\nlines".into()),
            labels: vec!["bug".into(), "p1".into()],
        });
        assert_eq!(
            s,
            r#"{"title":"t","body":"b\nlines","labels":["bug","p1"]}"#
        );
    }

    #[test]
    fn serialize_new_pull_round_trip() {
        let s = serialize_new_pull(&NewPull {
            title: "PR".into(),
            body: Some("desc".into()),
            head: "feature".into(),
            base: "main".into(),
        });
        assert_eq!(
            s,
            r#"{"title":"PR","head":"feature","base":"main","body":"desc"}"#
        );
    }

    // -- end-to-end mock tests --

    #[test]
    fn auth_header_uses_token_not_bearer() {
        let mock = start_mock(vec![ok(
            br#"{"full_name":"o/r","default_branch":"main","fork":false}"#.to_vec(),
        )]);
        let client = Client::new(&mock.base_url, "secretpat");
        let _ = client.get_repo("o", "r").unwrap();
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap.len(), 1);
        assert_eq!(cap[0].authorization.as_deref(), Some("token secretpat"));
        // And not a Bearer prefix.
        assert!(!cap[0].authorization.as_deref().unwrap().starts_with("Bearer"));
    }

    #[test]
    fn get_repo_parses_body_and_uses_api_v1_path() {
        let mock = start_mock(vec![ok(
            br#"{"full_name":"acme/widgets","default_branch":"trunk","fork":true}"#.to_vec(),
        )]);
        let client = Client::new(&mock.base_url, "t");
        let repo = client.get_repo("acme", "widgets").unwrap();
        assert_eq!(repo.full_name, "acme/widgets");
        assert_eq!(repo.default_branch, "trunk");
        assert!(repo.fork);
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].method, "GET");
        assert_eq!(cap[0].path, "/api/v1/repos/acme/widgets");
    }

    #[test]
    fn list_issues_state_filter_open() {
        let mock = start_mock(vec![ok(br#"[]"#.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let issues = client.list_issues("o", "r", IssueState::Open).unwrap();
        assert!(issues.is_empty());
        let cap = mock.captured.lock().unwrap();
        assert!(cap[0].path.contains("state=open"));
        // Gitea returns issues+pulls by default; we filter to issues.
        assert!(cap[0].path.contains("type=issues"));
    }

    #[test]
    fn list_issues_state_filter_closed_and_all() {
        let mock = start_mock(vec![ok(br#"[]"#.to_vec()), ok(br#"[]"#.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        client.list_issues("o", "r", IssueState::Closed).unwrap();
        client.list_issues("o", "r", IssueState::All).unwrap();
        let cap = mock.captured.lock().unwrap();
        assert!(cap[0].path.contains("state=closed"));
        assert!(cap[1].path.contains("state=all"));
    }

    #[test]
    fn get_issue_parses_user_and_labels() {
        let body = br#"{
            "number":42,
            "title":"crash on startup",
            "body":"steps:\n1. boot",
            "state":"open",
            "user":{"login":"alice","id":7},
            "labels":[{"name":"bug","color":"ee0000"},{"name":"p1","color":"ff8800"}]
        }"#;
        let mock = start_mock(vec![ok(body.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let issue = client.get_issue("o", "r", 42).unwrap();
        assert_eq!(issue.number, 42);
        assert_eq!(issue.title, "crash on startup");
        assert_eq!(issue.state, IssueState::Open);
        assert_eq!(issue.user.login, "alice");
        assert_eq!(issue.user.id, 7);
        assert_eq!(issue.labels.len(), 2);
        assert_eq!(issue.labels[0].name, "bug");
        assert_eq!(issue.labels[1].color, "ff8800");
    }

    #[test]
    fn create_issue_posts_json_with_labels() {
        let returned = br#"{
            "number":1,
            "title":"t",
            "body":"b",
            "state":"open",
            "user":{"login":"u","id":1},
            "labels":[]
        }"#;
        let mock = start_mock(vec![status(201, returned.to_vec())]);
        let client = Client::new(&mock.base_url, "tok");
        let issue = client
            .create_issue(
                "o",
                "r",
                NewIssue {
                    title: "t".into(),
                    body: Some("b".into()),
                    labels: vec!["bug".into()],
                },
            )
            .unwrap();
        assert_eq!(issue.number, 1);
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].method, "POST");
        assert_eq!(cap[0].path, "/api/v1/repos/o/r/issues");
        assert_eq!(cap[0].content_type.as_deref(), Some("application/json"));
        let sent = std::str::from_utf8(&cap[0].body).unwrap();
        assert!(sent.contains("\"title\":\"t\""));
        assert!(sent.contains("\"body\":\"b\""));
        assert!(sent.contains("\"labels\":[\"bug\"]"));
    }

    #[test]
    fn add_issue_comment_posts_body() {
        let returned = br#"{"id":99,"body":"thanks","user":{"login":"u","id":1}}"#;
        let mock = start_mock(vec![status(201, returned.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let c = client
            .add_issue_comment("o", "r", 5, "thanks")
            .unwrap();
        assert_eq!(c.id, 99);
        assert_eq!(c.body, "thanks");
        assert_eq!(c.user.login, "u");
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].method, "POST");
        assert_eq!(cap[0].path, "/api/v1/repos/o/r/issues/5/comments");
        assert_eq!(
            std::str::from_utf8(&cap[0].body).unwrap(),
            r#"{"body":"thanks"}"#
        );
    }

    #[test]
    fn list_issue_comments_parses_array() {
        let body = br#"[
            {"id":1,"body":"first","user":{"login":"a","id":1}},
            {"id":2,"body":"second","user":{"login":"b","id":2}}
        ]"#;
        let mock = start_mock(vec![ok(body.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let comments = client.list_issue_comments("o", "r", 9).unwrap();
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].body, "first");
        assert_eq!(comments[1].user.login, "b");
    }

    #[test]
    fn list_pulls_parses_array() {
        let body = br#"[{
            "number":3,
            "title":"refactor",
            "body":"",
            "state":"open",
            "head":{"label":"u:feat","ref":"feat","sha":"deadbeef"},
            "base":{"label":"o:main","ref":"main","sha":"cafef00d"},
            "merged":false
        }]"#;
        let mock = start_mock(vec![ok(body.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let pulls = client.list_pulls("o", "r", IssueState::Open).unwrap();
        assert_eq!(pulls.len(), 1);
        assert_eq!(pulls[0].number, 3);
        assert_eq!(pulls[0].head.ref_, "feat");
        assert_eq!(pulls[0].base.sha, "cafef00d");
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].path, "/api/v1/repos/o/r/pulls?state=open");
    }

    #[test]
    fn get_pull_parses_merged_flag() {
        let body = br#"{
            "number":7,
            "title":"done",
            "body":"yes",
            "state":"closed",
            "head":{"label":"u:f","ref":"f","sha":"aa"},
            "base":{"label":"o:m","ref":"m","sha":"bb"},
            "merged":true
        }"#;
        let mock = start_mock(vec![ok(body.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let pr = client.get_pull("o", "r", 7).unwrap();
        assert_eq!(pr.number, 7);
        assert!(pr.merged);
        assert_eq!(pr.state, IssueState::Closed);
    }

    #[test]
    fn create_pull_posts_head_and_base() {
        let returned = br#"{
            "number":11,
            "title":"new",
            "body":"",
            "state":"open",
            "head":{"label":"u:f","ref":"f","sha":"aa"},
            "base":{"label":"o:m","ref":"m","sha":"bb"},
            "merged":false
        }"#;
        let mock = start_mock(vec![status(201, returned.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let pr = client
            .create_pull(
                "o",
                "r",
                NewPull {
                    title: "new".into(),
                    body: None,
                    head: "f".into(),
                    base: "m".into(),
                },
            )
            .unwrap();
        assert_eq!(pr.number, 11);
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].method, "POST");
        assert_eq!(cap[0].path, "/api/v1/repos/o/r/pulls");
        let sent = std::str::from_utf8(&cap[0].body).unwrap();
        assert!(sent.contains("\"head\":\"f\""));
        assert!(sent.contains("\"base\":\"m\""));
        // No body key when None.
        assert!(!sent.contains("\"body\""));
    }

    #[test]
    fn status_404_includes_body() {
        let mock = start_mock(vec![status(
            404,
            br#"{"message":"Not Found"}"#.to_vec(),
        )]);
        let client = Client::new(&mock.base_url, "t");
        let err = client.get_repo("o", "missing").unwrap_err();
        match err {
            Error::Status { code, body } => {
                assert_eq!(code, 404);
                assert!(body.contains("Not Found"));
            }
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[test]
    fn status_500_includes_body() {
        let mock = start_mock(vec![status(500, br#"server boom"#.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let err = client.list_issues("o", "r", IssueState::All).unwrap_err();
        match err {
            Error::Status { code, body } => {
                assert_eq!(code, 500);
                assert_eq!(body, "server boom");
            }
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[test]
    fn malformed_json_is_invalid_response() {
        let mock = start_mock(vec![ok(br#"{not-json"#.to_vec())]);
        let client = Client::new(&mock.base_url, "t");
        let err = client.get_repo("o", "r").unwrap_err();
        assert!(matches!(err, Error::InvalidResponse(_)), "got {err:?}");
    }

    #[test]
    fn trailing_slash_in_base_url_is_tolerated() {
        let mock = start_mock(vec![ok(
            br#"{"full_name":"o/r","default_branch":"main","fork":false}"#.to_vec(),
        )]);
        let with_slash = format!("{}/", mock.base_url);
        let client = Client::new(&with_slash, "t");
        client.get_repo("o", "r").unwrap();
        let cap = mock.captured.lock().unwrap();
        // No double slash in the path.
        assert_eq!(cap[0].path, "/api/v1/repos/o/r");
    }
}
