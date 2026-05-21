//! Minimal GitHub REST API client.
//!
//! Parallel to `gitea`: same struct shapes (Repo, Issue, PullRequest, Comment,
//! User, Label, BranchRef, NewIssue, NewPull, IssueState, Error), same
//! synchronous libcurl pattern, JSON via `anthropic::json`.
//!
//! Wire differences from `gitea`:
//!   * `Authorization: Bearer <PAT>` rather than `token <PAT>`.
//!   * `Accept: application/vnd.github+json`.
//!   * `X-GitHub-Api-Version: 2022-11-28`.
//!   * `User-Agent` is required — GitHub returns 403 without one. Default
//!     `sandcastle-agent`, overridable via [`Client::with_user_agent`].
//!   * Base path is `<base_url>/repos/...` — no `/api/v1/` prefix.
//!   * `GET /repos/.../issues` returns issues *and* pull requests; PRs are
//!     identified by a `pull_request` field on the issue payload.
//!     [`Client::list_issues`] filters them out client-side.
//!   * 422 validation errors carry an `errors` array — the message is
//!     surfaced in [`Error::Status`]'s body unchanged so callers can inspect
//!     it.
//!
//! For GitHub Enterprise, pass the Enterprise API base
//! (`https://ghe.example.com/api/v3`) via [`Client::with_base_url`].

mod http;

use anthropic::json::{self, Json, escape_into};

// ---- public types ----------------------------------------------------------

pub struct Client {
    base_url: String,
    token: String,
    user_agent: String,
}

impl Client {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.github.com".to_string(),
            token: token.into(),
            user_agent: "sandcastle-agent".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
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
        // GitHub's /issues endpoint mixes PRs and issues. There is no
        // server-side filter for "issues only" (unlike Gitea's `type=issues`),
        // so we filter PRs out client-side by skipping any entry that has a
        // `pull_request` field present.
        let url = self.url(&format!(
            "/repos/{owner}/{repo}/issues?state={}",
            state.as_query()
        ));
        let body = self.get(&url)?;
        let v = json::parse(&body).map_err(map_anthropic_err)?;
        let Json::Arr(items) = v else {
            return Err(Error::InvalidResponse("expected JSON array".into()));
        };
        let mut out = Vec::with_capacity(items.len());
        for item in &items {
            if is_pull_request_entry(item) {
                continue;
            }
            out.push(parse_issue_value(item)?);
        }
        Ok(out)
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
        // Unlike Gitea, GitHub accepts label *names* directly on issue
        // creation — no id lookup required.
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
        format!("{base}{path}")
    }

    fn headers(&self, with_content_type: bool) -> Vec<String> {
        let mut h = vec![
            format!("Authorization: Bearer {}", self.token),
            "Accept: application/vnd.github+json".to_string(),
            "X-GitHub-Api-Version: 2022-11-28".to_string(),
            format!("User-Agent: {}", self.user_agent),
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
            // Per-record state is only ever open|closed; "all" is a query
            // filter. Unknown values collapse to Open so a future GitHub
            // state addition doesn't crash the parser.
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
        // Missing or null collapses to empty string. GitHub frequently omits
        // optional body fields (e.g. an issue posted with no description has
        // `body: null`); treating those as "" keeps the struct field
        // non-Option and avoids per-callsite checks.
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
        // Missing fork/merged flags default false. On the PR object, `merged`
        // is only present on the detail endpoint — the list endpoint omits it.
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

fn is_pull_request_entry(v: &Json) -> bool {
    // GitHub flags a /issues entry as a PR by including a non-null
    // `pull_request` object alongside the usual issue fields. The presence
    // (and non-null-ness) of the key is the discriminator.
    match v.get("pull_request") {
        None | Some(Json::Null) => false,
        Some(_) => true,
    }
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
        // GitHub accepts label names here as an array of strings.
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
    //! captures the inbound request for assertion (headers, URL, method,
    //! body). The mock closes the connection after each scripted reply,
    //! matching `connection: close`.

    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
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
        headers: HashMap<String, String>,
        body: Vec<u8>,
    }

    impl Captured {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
        }
    }

    struct Mock {
        base_url: String,
        captured: Arc<Mutex<Vec<Captured>>>,
        _handle: thread::JoinHandle<()>,
        running: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for Mock {
        fn drop(&mut self) {
            self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn start_mock(replies: Vec<Reply>) -> Mock {
        use std::sync::atomic::{AtomicBool, Ordering};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();

        let captured = Arc::new(Mutex::new(Vec::<Captured>::new()));
        let cap_clone = Arc::clone(&captured);
        let total = replies.len();
        let replies = Arc::new(replies);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let handle = thread::spawn(move || {
            let mut idx = 0usize;
            while running_clone.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if idx < total {
                            let reply = replies[idx].clone();
                            idx += 1;
                            handle_one(stream, &reply, &cap_clone);
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });

        Mock {
            base_url: format!("http://{addr}"),
            captured,
            _handle: handle,
            running,
        }
    }

    fn handle_one(
        mut stream: TcpStream,
        reply: &Reply,
        captured_sink: &Arc<Mutex<Vec<Captured>>>,
    ) {
        let mut reader = match stream.try_clone() {
            Ok(s) => BufReader::new(s),
            Err(_) => return,
        };
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            return;
        }
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
            if reader.read_line(&mut line).is_err() {
                return;
            }
            if line == "\r\n" || line == "\n" || line.is_empty() {
                break;
            }
            let line = line.trim_end_matches(['\r', '\n']);
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim().to_string();
            let lc = name.to_ascii_lowercase();
            if lc == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            captured.headers.insert(lc, value);
        }

        if content_length > 0 {
            let mut body = vec![0u8; content_length];
            if reader.read_exact(&mut body).is_err() {
                return;
            }
            captured.body = body;
        }

        // Record the request BEFORE we write the response. The client's
        // `perform` completes as soon as it has the response bytes, and the
        // test thread will assert on `captured` immediately after that —
        // pushing after the write left a window where the test could read
        // an empty Vec.
        captured_sink.lock().unwrap().push(captured);

        let body = &reply.body;
        let mut resp = format!("HTTP/1.1 {} {}\r\n", reply.status, status_text(reply.status));
        resp.push_str("content-type: application/json\r\n");
        resp.push_str(&format!("content-length: {}\r\n", body.len()));
        resp.push_str("connection: close\r\n\r\n");
        let _ = stream.write_all(resp.as_bytes());
        let _ = stream.write_all(body);
        let _ = stream.flush();
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

    fn repo_body() -> Vec<u8> {
        br#"{"full_name":"o/r","default_branch":"main","fork":false}"#.to_vec()
    }

    // -- serializer unit tests --

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
    fn auth_header_uses_bearer_not_token() {
        let mock = start_mock(vec![ok(repo_body())]);
        let client = Client::new("secretpat").with_base_url(&mock.base_url);
        client.get_repo("o", "r").unwrap();
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap.len(), 1);
        assert_eq!(cap[0].header("authorization"), Some("Bearer secretpat"));
        // Bearer, not Gitea's `token` scheme.
        assert!(!cap[0].header("authorization").unwrap().starts_with("token "));
    }

    #[test]
    fn user_agent_is_sent_on_every_request() {
        let mock = start_mock(vec![ok(repo_body()), ok(br#"[]"#.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        client.get_repo("o", "r").unwrap();
        client.list_issues("o", "r", IssueState::Open).unwrap();
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap.len(), 2);
        for c in cap.iter() {
            // Default UA. GitHub returns 403 without one — we must always set it.
            assert_eq!(c.header("user-agent"), Some("sandcastle-agent"));
        }
    }

    #[test]
    fn with_user_agent_overrides_default() {
        let mock = start_mock(vec![ok(repo_body())]);
        let client = Client::new("t")
            .with_base_url(&mock.base_url)
            .with_user_agent("my-bot/1.0");
        client.get_repo("o", "r").unwrap();
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].header("user-agent"), Some("my-bot/1.0"));
    }

    #[test]
    fn accept_and_api_version_headers_present() {
        let mock = start_mock(vec![ok(repo_body())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        client.get_repo("o", "r").unwrap();
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].header("accept"), Some("application/vnd.github+json"));
        assert_eq!(cap[0].header("x-github-api-version"), Some("2022-11-28"));
    }

    #[test]
    fn get_repo_happy_path_no_api_v1_prefix() {
        let mock = start_mock(vec![ok(
            br#"{"full_name":"acme/widgets","default_branch":"trunk","fork":true}"#.to_vec(),
        )]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        let repo = client.get_repo("acme", "widgets").unwrap();
        assert_eq!(repo.full_name, "acme/widgets");
        assert_eq!(repo.default_branch, "trunk");
        assert!(repo.fork);
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].method, "GET");
        // No `/api/v1/` — that's a Gitea-ism.
        assert_eq!(cap[0].path, "/repos/acme/widgets");
    }

    #[test]
    fn list_issues_filters_out_pull_requests() {
        // First entry has a `pull_request` field — it's actually a PR. The
        // second is a real issue. Only the issue should come back.
        let body = br#"[
            {
                "number":10,
                "title":"a pr in disguise",
                "body":"",
                "state":"open",
                "user":{"login":"a","id":1},
                "labels":[],
                "pull_request":{"url":"https://api.github.com/.../pulls/10"}
            },
            {
                "number":11,
                "title":"real issue",
                "body":"text",
                "state":"open",
                "user":{"login":"b","id":2},
                "labels":[]
            }
        ]"#;
        let mock = start_mock(vec![ok(body.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        let issues = client.list_issues("o", "r", IssueState::Open).unwrap();
        assert_eq!(issues.len(), 1, "PR entry should be filtered out");
        assert_eq!(issues[0].number, 11);
        assert_eq!(issues[0].title, "real issue");
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].path, "/repos/o/r/issues?state=open");
    }

    #[test]
    fn list_issues_state_filter_closed_and_all() {
        let mock = start_mock(vec![ok(br#"[]"#.to_vec()), ok(br#"[]"#.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
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
        let client = Client::new("t").with_base_url(&mock.base_url);
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
        let client = Client::new("tok").with_base_url(&mock.base_url);
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
        assert_eq!(cap[0].path, "/repos/o/r/issues");
        assert_eq!(cap[0].header("content-type"), Some("application/json"));
        let sent = std::str::from_utf8(&cap[0].body).unwrap();
        assert!(sent.contains("\"title\":\"t\""));
        assert!(sent.contains("\"body\":\"b\""));
        assert!(sent.contains("\"labels\":[\"bug\"]"));
    }

    #[test]
    fn add_issue_comment_posts_body() {
        let returned = br#"{"id":99,"body":"thanks","user":{"login":"u","id":1}}"#;
        let mock = start_mock(vec![status(201, returned.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        let c = client.add_issue_comment("o", "r", 5, "thanks").unwrap();
        assert_eq!(c.id, 99);
        assert_eq!(c.body, "thanks");
        assert_eq!(c.user.login, "u");
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].method, "POST");
        assert_eq!(cap[0].path, "/repos/o/r/issues/5/comments");
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
        let client = Client::new("t").with_base_url(&mock.base_url);
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
            "base":{"label":"o:main","ref":"main","sha":"cafef00d"}
        }]"#;
        let mock = start_mock(vec![ok(body.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        let pulls = client.list_pulls("o", "r", IssueState::Open).unwrap();
        assert_eq!(pulls.len(), 1);
        assert_eq!(pulls[0].number, 3);
        assert_eq!(pulls[0].head.ref_, "feat");
        assert_eq!(pulls[0].base.sha, "cafef00d");
        // `merged` is absent on the list endpoint; defaults to false.
        assert!(!pulls[0].merged);
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[0].path, "/repos/o/r/pulls?state=open");
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
        let client = Client::new("t").with_base_url(&mock.base_url);
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
        let client = Client::new("t").with_base_url(&mock.base_url);
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
        assert_eq!(cap[0].path, "/repos/o/r/pulls");
        let sent = std::str::from_utf8(&cap[0].body).unwrap();
        assert!(sent.contains("\"head\":\"f\""));
        assert!(sent.contains("\"base\":\"m\""));
        // No body key when None.
        assert!(!sent.contains("\"body\""));
    }

    #[test]
    fn status_404_includes_body() {
        let mock = start_mock(vec![status(404, br#"{"message":"Not Found"}"#.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
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
        let client = Client::new("t").with_base_url(&mock.base_url);
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
    fn status_422_validation_body_surfaced_verbatim() {
        // GitHub's validation errors carry an `errors` array. We surface the
        // whole body unchanged so callers can read it.
        let body = br#"{"message":"Validation Failed","errors":[{"resource":"Issue","code":"missing_field","field":"title"}],"documentation_url":"https://docs.github.com/rest/issues/issues#create-an-issue"}"#;
        let mock = start_mock(vec![status(422, body.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        let err = client
            .create_issue(
                "o",
                "r",
                NewIssue {
                    title: String::new(),
                    body: None,
                    labels: vec![],
                },
            )
            .unwrap_err();
        match err {
            Error::Status { code, body } => {
                assert_eq!(code, 422);
                // Surface verbatim — caller can parse `errors` if they need it.
                assert!(body.contains("Validation Failed"));
                assert!(body.contains("missing_field"));
                assert!(body.contains("\"field\":\"title\""));
            }
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[test]
    fn malformed_json_is_invalid_response() {
        let mock = start_mock(vec![ok(br#"{not-json"#.to_vec())]);
        let client = Client::new("t").with_base_url(&mock.base_url);
        let err = client.get_repo("o", "r").unwrap_err();
        assert!(matches!(err, Error::InvalidResponse(_)), "got {err:?}");
    }

    #[test]
    fn trailing_slash_in_base_url_is_tolerated() {
        let mock = start_mock(vec![ok(repo_body())]);
        let with_slash = format!("{}/", mock.base_url);
        let client = Client::new("t").with_base_url(&with_slash);
        client.get_repo("o", "r").unwrap();
        let cap = mock.captured.lock().unwrap();
        // No double slash in the path.
        assert_eq!(cap[0].path, "/repos/o/r");
    }

    #[test]
    fn default_base_url_is_api_github_com() {
        // No network call — just inspect the field via a request URL build.
        // We construct without `with_base_url` and verify the prefix.
        let client = Client::new("t");
        assert_eq!(client.url("/repos/o/r"), "https://api.github.com/repos/o/r");
    }
}
